"""Canonical SmolLM2 system benchmark and full-runtime profiler.

Run from the repository root after rebuilding the native extension:

    build/python3119/python.exe benchmarks/benchmark_flux.py --mode system

The benchmark uses the pinned FP32 checkpoint and the canonical retained Flux
category set.  CUDA-event samples are collected in rotating same-process order;
model loading, deterministic input construction, correctness checks, cache
setup for steady-state decode, and native runtime construction are outside
decode timings.
JSON output is optional so the final report can retain exact measurements.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from flux.model.smollm2 import MODEL_ID, MODEL_REVISION, load_model
from flux.model.smollm2_flux import (
    FINAL_FLUX_OPERATOR_CATEGORIES,
    enable_flux_ops,
    flux_operator_counts,
)
from flux.runtime import NativeSmolLM2Prefill
from benchmarks.benchmark_utils import (
    configure_runtime,
    deterministic_input_ids,
    parse_positive_int_list,
)


DTYPE = torch.float32
SEED = 0
RTOL = 2e-4
ATOL = 2e-5
DEFAULT_LENGTHS = (128, 512, 1024, 2048, 4096, 8192)
DEFAULT_GENERATION_PROMPTS = (128, 1024, 4096)


@dataclass(frozen=True)
class PrefillResult:
    length: int
    reference_ms: float
    flux_ms: float
    native_ms: float
    max_logits_error: float
    max_native_logits_error: float
    max_native_key_error: float
    max_native_value_error: float
    cache_position: int
    cache_length: int
    stable_addresses: bool
    cache_bytes: int
    prefill_workspace_bytes: int
    decode_workspace_bytes: int
    stable_buffer_bytes: int


@dataclass(frozen=True)
class CorrectnessResult:
    effective_length: int
    max_logits_error: float
    max_key_error: float
    max_value_error: float
    positions_correct: bool
    stable_addresses: bool
    greedy_equal: bool


@dataclass(frozen=True)
class DecodeResult:
    effective_length: int
    window_start: int
    window_stop: int
    reference_ms: float
    flux_eager_ms: float
    native_ms: float


@dataclass(frozen=True)
class GenerationResult:
    prompt_length: int
    output_tokens: int
    path: str
    prefill_ms: float
    ttft_ms: float
    decode_ms: float
    total_execution_ms: float
    total_observed_ms: float


@dataclass(frozen=True)
class RuntimeAudit:
    capacity: int
    replays: int
    launches_per_replay: int
    flux_launches_per_replay: int
    cublas_launches_per_replay: int
    framework_launches_per_replay: int
    replay_allocation_growth_bytes: int
    stable_addresses: bool
    cpu_synchronize_events: int
    cpu_allocation_events: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("system", "prefill", "decode", "generation", "profile"),
        default="system",
        help="benchmark responsibility to run (default: system)",
    )
    parser.add_argument(
        "--lengths", type=parse_positive_int_list, default=DEFAULT_LENGTHS
    )
    parser.add_argument(
        "--generation-prompts",
        type=parse_positive_int_list,
        default=DEFAULT_GENERATION_PROMPTS,
    )
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--stabilization-iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--correctness-tokens", type=int, default=8)
    parser.add_argument("--generation-repetitions", type=int, default=3)
    parser.add_argument("--audit-capacity", type=int, default=4096)
    parser.add_argument("--audit-replays", type=int, default=10)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument(
        "--decode-contexts",
        type=parse_positive_int_list,
        default=(128, 1024, 4096, 8192),
        help="effective decode lengths used by profile mode",
    )
    parser.add_argument(
        "--prefill-lengths",
        type=parse_positive_int_list,
        default=(128, 1024, 4096, 8192),
        help="prompt lengths used by profile mode",
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--skip-prefill", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.warmup is None:
        args.warmup = 3 if args.mode == "profile" else 5
    if args.samples is None:
        args.samples = 5 if args.mode == "profile" else 20
    positive = (
        "output_tokens",
        "samples",
        "rounds",
        "correctness_tokens",
        "generation_repetitions",
        "audit_capacity",
        "audit_replays",
    )
    if any(getattr(args, name) < 1 for name in positive):
        parser.error("sample, round, token, and audit counts must be positive")
    if args.warmup < 0 or args.stabilization_iterations < 0:
        parser.error("warmup and stabilization counts may be zero, not negative")
    if args.mode == "profile" and min(args.repetitions, args.top_k) < 1:
        parser.error("repetitions and top-k must be positive")
    if args.mode == "profile" and args.skip_prefill and args.skip_decode:
        parser.error("cannot skip both profile workloads")
    if (
        args.mode in {"system", "generation"}
        and not args.skip_generation
        and args.output_tokens < 2
    ):
        parser.error("--output-tokens must be at least two when generation is enabled")
    if (
        args.mode in {"system", "decode"}
        and not args.skip_audit
        and args.audit_capacity <= args.audit_replays + 3
    ):
        parser.error("audit capacity must exceed audit replays plus three warmups")
    return args


def _command_line(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    lines = (result.stdout or result.stderr).strip().splitlines()
    return lines[-1].strip() if lines else "unavailable"


def _environment(args: argparse.Namespace) -> dict[str, Any]:
    capability = torch.cuda.get_device_capability()
    return {
        "mode": args.mode,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "pytorch": torch.__version__,
        "pytorch_cuda_build": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "cuda_toolkit": _command_line(["nvcc", "--version"]),
        "driver": _command_line(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ]
        ),
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "dtype": str(DTYPE),
        "tf32": False,
        "deterministic_algorithms": True,
        "deterministic_safety_fill": torch.utils.deterministic.fill_uninitialized_memory,
        "seed": SEED,
        "stabilization_iterations": args.stabilization_iterations,
        "warmup": args.warmup,
        "timed_samples_per_round": args.samples,
        "rounds": args.rounds,
        "statistic": "median of all CUDA-event samples",
        "final_flux_categories": sorted(FINAL_FLUX_OPERATOR_CATEGORIES),
    }


def _input_ids(length: int, vocab_size: int) -> torch.Tensor:
    return deterministic_input_ids(length, vocab_size)


def _max_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    result = float((actual - expected).abs().max().item())
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    return result


def _cache_length(cache: Any) -> int:
    value = cache.get_seq_length()
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def _prefill_correctness(
    reference: torch.nn.Module, flux: torch.nn.Module, length: int
) -> float:
    input_ids = _input_ids(length, reference.config.vocab_size)
    with torch.inference_mode():
        expected = reference(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        actual = flux(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    maximum = _max_error(actual.logits, expected.logits)
    if _cache_length(expected.past_key_values) != length:
        raise AssertionError("reference prefill cache length is incorrect")
    if _cache_length(actual.past_key_values) != length:
        raise AssertionError("Flux prefill cache length is incorrect")
    del input_ids, expected, actual
    return maximum


def _event_sample(operation: Callable[[], Any]) -> tuple[float, Any]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end), output


def _benchmark_prefill(
    reference: torch.nn.Module,
    flux: torch.nn.Module,
    length: int,
    warmup: int,
    samples: int,
    rounds: int,
) -> PrefillResult:
    maximum = _prefill_correctness(reference, flux, length)
    input_ids = _input_ids(length, reference.config.vocab_size)
    with torch.inference_mode():
        native = NativeSmolLM2Prefill.capture(flux, input_ids)
        flux_output = flux(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    native_error = _max_error(native.logits, flux_output.logits)
    max_key_error = 0.0
    max_value_error = 0.0
    for layer_index, layer in enumerate(flux_output.past_key_values.layers):
        max_key_error = max(
            max_key_error,
            _max_error(native.key_cache[layer_index, ..., :length, :], layer.keys),
        )
        max_value_error = max(
            max_value_error,
            _max_error(native.value_cache[layer_index, ..., :length, :], layer.values),
        )
    if native.cache_position != length or native.cache_length != length:
        raise AssertionError("native prefill cache position or length is incorrect")
    addresses = native.stable_addresses()
    memory = native.memory
    del flux_output
    values: dict[str, list[float]] = {
        "reference": [],
        "Flux eager": [],
        "native": [],
    }
    paths = ("reference", "Flux eager", "native")
    output = None
    with torch.inference_mode():
        for round_index in range(rounds):
            order = paths[round_index % 3 :] + paths[: round_index % 3]
            for name in order:
                for _ in range(warmup):
                    if name == "native":
                        output = native.prefill(input_ids)
                    else:
                        model = reference if name == "reference" else flux
                        output = model(
                            input_ids=input_ids, use_cache=True, logits_to_keep=1
                        )
            torch.cuda.synchronize()
            for sample_index in range(samples):
                offset = (sample_index + round_index) % 3
                order = paths[offset:] + paths[:offset]
                for name in order:
                    if name == "native":
                        elapsed, output = _event_sample(
                            lambda: native.prefill(input_ids)
                        )
                    else:
                        model = reference if name == "reference" else flux
                        elapsed, output = _event_sample(
                            lambda model=model: model(
                                input_ids=input_ids,
                                use_cache=True,
                                logits_to_keep=1,
                            )
                        )
                    values[name].append(elapsed)
    result = PrefillResult(
        length,
        statistics.median(values["reference"]),
        statistics.median(values["Flux eager"]),
        statistics.median(values["native"]),
        maximum,
        native_error,
        max_key_error,
        max_value_error,
        native.cache_position,
        native.cache_length,
        addresses == native.stable_addresses(),
        memory.cache_bytes,
        memory.prefill_workspace_bytes,
        memory.decode_workspace_bytes,
        memory.stable_buffer_bytes,
    )
    del input_ids, output, native
    return result


def _validate_decode(
    reference: torch.nn.Module,
    flux: torch.nn.Module,
    effective_length: int,
    tokens: int,
) -> CorrectnessResult:
    steps = min(tokens, effective_length - 1)
    prompt_length = effective_length - steps
    prompt = _input_ids(prompt_length, reference.config.vocab_size)
    with torch.inference_mode():
        reference_output = reference(input_ids=prompt, use_cache=True, logits_to_keep=1)
        flux_output = flux(input_ids=prompt, use_cache=True, logits_to_keep=1)
        native = NativeSmolLM2Prefill.capture(
            flux, prompt, max_decode_steps=steps
        )
        reference_cache = reference_output.past_key_values
        flux_cache = flux_output.past_key_values
        reference_token = reference_output.logits.argmax(dim=-1)
        flux_token = flux_output.logits.argmax(dim=-1)
        native_token = native.logits.argmax(dim=-1)
        reference_tokens = [reference_token]
        flux_tokens = [flux_token]
        native_tokens = [native_token]
        max_logits = max(
            _max_error(flux_output.logits, reference_output.logits),
            _max_error(native.logits, flux_output.logits),
        )
        max_key = 0.0
        max_value = 0.0
        positions = native.cache_position == native.cache_length == prompt_length
        addresses = native.stable_addresses()
        for step in range(steps):
            reference_output = reference(
                input_ids=reference_token,
                past_key_values=reference_cache,
                use_cache=True,
                logits_to_keep=1,
            )
            flux_output = flux(
                input_ids=flux_token,
                past_key_values=flux_cache,
                use_cache=True,
                logits_to_keep=1,
            )
            native_logits = native.replay(native_token)
            max_logits = max(
                max_logits,
                _max_error(flux_output.logits, reference_output.logits),
                _max_error(native_logits, flux_output.logits),
            )
            expected_length = prompt_length + step + 1
            positions &= (
                native.cache_position == expected_length
                and native.cache_length == expected_length
                and _cache_length(reference_cache) == expected_length
                and _cache_length(flux_cache) == expected_length
            )
            for layer_index, (reference_layer, flux_layer) in enumerate(
                zip(reference_cache.layers, flux_cache.layers, strict=True)
            ):
                native_keys = native.key_cache[
                    layer_index, ..., :expected_length, :
                ]
                native_values = native.value_cache[
                    layer_index, ..., :expected_length, :
                ]
                key_error = float((flux_layer.keys - reference_layer.keys).abs().max().item())
                value_error = float(
                    (flux_layer.values - reference_layer.values).abs().max().item()
                )
                native_key_error = _max_error(native_keys, flux_layer.keys)
                native_value_error = _max_error(native_values, flux_layer.values)
                max_key = max(max_key, key_error, native_key_error)
                max_value = max(max_value, value_error, native_value_error)
                expected_shape = reference_layer.keys.shape
                if (
                    flux_layer.keys.shape != expected_shape
                    or flux_layer.values.shape != reference_layer.values.shape
                    or native_keys.shape != expected_shape
                    or native_values.shape != reference_layer.values.shape
                ):
                    raise AssertionError("KV-cache tensor shapes differ")
                if not all(
                    bool(torch.all(torch.isfinite(tensor)))
                    for tensor in (
                        flux_layer.keys,
                        flux_layer.values,
                        native_keys,
                        native_values,
                    )
                ):
                    raise AssertionError("KV cache contains non-finite values")
            reference_token = reference_output.logits.argmax(dim=-1)
            flux_token = flux_output.logits.argmax(dim=-1)
            native_token = native_logits.argmax(dim=-1)
            reference_tokens.append(reference_token)
            flux_tokens.append(flux_token)
            native_tokens.append(native_token)
        stable = addresses == native.stable_addresses()
        greedy = torch.equal(
            torch.cat(reference_tokens, dim=-1), torch.cat(flux_tokens, dim=-1)
        ) and torch.equal(
            torch.cat(flux_tokens, dim=-1), torch.cat(native_tokens, dim=-1)
        )
    del prompt, native, reference_cache, flux_cache
    return CorrectnessResult(
        effective_length, max_logits, max_key, max_value, positions, stable, greedy
    )


def _decode_eager_samples(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    warmup: int,
    samples: int,
) -> list[float]:
    values = []
    with torch.inference_mode():
        output = model(input_ids=prompt, use_cache=True, logits_to_keep=1)
        cache = output.past_key_values
        token = output.logits.argmax(dim=-1)
        for _ in range(warmup):
            output = model(
                input_ids=token,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            token = output.logits.argmax(dim=-1)
        torch.cuda.synchronize()
        for _ in range(samples):
            elapsed, output = _event_sample(
                lambda: model(
                    input_ids=token,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
            )
            values.append(elapsed)
            token = output.logits.argmax(dim=-1)
    return values


def _decode_native_samples(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    warmup: int,
    samples: int,
) -> list[float]:
    native = NativeSmolLM2Prefill.capture(
        model, prompt, max_decode_steps=warmup + samples
    )
    for _ in range(warmup):
        native.replay()
    torch.cuda.synchronize()
    values = []
    for _ in range(samples):
        elapsed, _ = _event_sample(native.replay)
        values.append(elapsed)
    del native
    return values


def _benchmark_decode(
    reference: torch.nn.Module,
    flux: torch.nn.Module,
    effective_length: int,
    warmup: int,
    samples: int,
    rounds: int,
) -> DecodeResult:
    total_steps = warmup + samples
    if effective_length <= total_steps:
        raise ValueError(
            f"effective length {effective_length} must exceed warmup + samples "
            f"({total_steps})"
        )
    prompt_length = effective_length - total_steps
    values: dict[str, list[float]] = {
        "reference": [],
        "Flux eager": [],
        "native": [],
    }
    paths = ("reference", "Flux eager", "native")
    for round_index in range(rounds):
        order = paths[round_index % 3 :] + paths[: round_index % 3]
        for name in order:
            prompt = _input_ids(prompt_length, reference.config.vocab_size)
            if name == "reference":
                batch = _decode_eager_samples(reference, prompt, warmup, samples)
            elif name == "Flux eager":
                batch = _decode_eager_samples(flux, prompt, warmup, samples)
            else:
                batch = _decode_native_samples(flux, prompt, warmup, samples)
            values[name].extend(batch)
            del prompt
    return DecodeResult(
        effective_length,
        effective_length - samples + 1,
        effective_length,
        statistics.median(values["reference"]),
        statistics.median(values["Flux eager"]),
        statistics.median(values["native"]),
    )


def _generation_eager_once(
    model: torch.nn.Module, prompt: torch.Tensor, output_tokens: int
) -> tuple[float, float]:
    with torch.inference_mode():
        prefill_start = torch.cuda.Event(enable_timing=True)
        prefill_end = torch.cuda.Event(enable_timing=True)
        prefill_start.record()
        output = model(input_ids=prompt, use_cache=True, logits_to_keep=1)
        token = output.logits.argmax(dim=-1)
        prefill_end.record()
        prefill_end.synchronize()
        prefill_ms = prefill_start.elapsed_time(prefill_end)
        cache = output.past_key_values
        decode_start = torch.cuda.Event(enable_timing=True)
        decode_end = torch.cuda.Event(enable_timing=True)
        decode_start.record()
        for _ in range(output_tokens - 1):
            output = model(
                input_ids=token,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            token = output.logits.argmax(dim=-1)
        decode_end.record()
        decode_end.synchronize()
    return prefill_ms, decode_start.elapsed_time(decode_end)


def _generation_native_once(
    model: torch.nn.Module, prompt: torch.Tensor, output_tokens: int
) -> tuple[float, float, float]:
    setup_start = time.perf_counter()
    native = NativeSmolLM2Prefill.capture(
        model, prompt, max_decode_steps=output_tokens - 1
    )
    torch.cuda.synchronize()
    setup_ms = (time.perf_counter() - setup_start) * 1000.0
    prefill_ms, logits = _event_sample(lambda: native.prefill(prompt))
    token = logits.argmax(dim=-1)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(output_tokens - 1):
        token = native.replay(token).argmax(dim=-1)
    end.record()
    end.synchronize()
    result = (prefill_ms, setup_ms, start.elapsed_time(end))
    del native
    return result


def _benchmark_generation(
    reference: torch.nn.Module,
    flux: torch.nn.Module,
    prompt_length: int,
    output_tokens: int,
    repetitions: int,
) -> list[GenerationResult]:
    samples: dict[str, list[tuple[float, float, float]]] = {
        "reference": [],
        "Flux eager": [],
        "native": [],
    }
    paths = tuple(samples)
    for repetition in range(repetitions):
        order = paths[repetition % 3 :] + paths[: repetition % 3]
        for name in order:
            prompt = _input_ids(prompt_length, reference.config.vocab_size)
            if name == "reference":
                prefill, decode = _generation_eager_once(
                    reference, prompt, output_tokens
                )
                samples[name].append((prefill, prefill, decode))
            elif name == "Flux eager":
                prefill, decode = _generation_eager_once(flux, prompt, output_tokens)
                samples[name].append((prefill, prefill, decode))
            else:
                samples[name].append(
                    _generation_native_once(flux, prompt, output_tokens)
                )
            del prompt
    results = []
    for name in paths:
        prefill = statistics.median(item[0] for item in samples[name])
        ttft = statistics.median(item[1] for item in samples[name])
        decode = statistics.median(item[2] for item in samples[name])
        results.append(
            GenerationResult(
                prompt_length,
                output_tokens,
                name,
                prefill,
                ttft,
                decode,
                prefill + decode,
                ttft + decode,
            )
        )
    return results


def _kernel_owner(name: str) -> str:
    lowered = name.lower()
    if "_zn4flux" in lowered or "flux" in lowered:
        return "Flux"
    if "cublas" in lowered or "gemm" in lowered or "gemv" in lowered:
        return "cuBLAS"
    return "framework"


def _runtime_audit(
    flux: torch.nn.Module, capacity: int, replays: int
) -> RuntimeAudit:
    prompt = _input_ids(capacity - replays - 3, flux.config.vocab_size)
    native = NativeSmolLM2Prefill.capture(
        flux, prompt, max_decode_steps=replays + 3
    )
    for _ in range(3):
        native.replay()
    torch.cuda.synchronize()
    addresses = native.stable_addresses()
    allocated_before = torch.cuda.memory_allocated()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        for _ in range(replays):
            native.replay()
        torch.cuda.synchronize()
    allocated_after = torch.cuda.memory_allocated()
    cuda_events = [
        event for event in profiler.events() if event.device_type == DeviceType.CUDA
    ]
    owners = {"Flux": 0, "cuBLAS": 0, "framework": 0}
    for event in cuda_events:
        owners[_kernel_owner(event.name)] += 1
    cpu_names = [
        event.name.lower()
        for event in profiler.events()
        if event.device_type == DeviceType.CPU
    ]
    result = RuntimeAudit(
        capacity,
        replays,
        round(len(cuda_events) / replays),
        round(owners["Flux"] / replays),
        round(owners["cuBLAS"] / replays),
        round(owners["framework"] / replays),
        allocated_after - allocated_before,
        addresses == native.stable_addresses(),
        sum("synchronize" in name for name in cpu_names),
        sum("cudamalloc" in name or "cudafree" in name for name in cpu_names),
    )
    del native, prompt
    return result


def _stabilize(flux: torch.nn.Module, iterations: int) -> None:
    if not iterations:
        return
    prompt = _input_ids(64, flux.config.vocab_size)
    with torch.inference_mode():
        output = flux(input_ids=prompt, use_cache=True, logits_to_keep=1)
        token = output.logits.argmax(dim=-1)
        cache = output.past_key_values
        for _ in range(iterations):
            output = flux(
                input_ids=token,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            token = output.logits.argmax(dim=-1)
    torch.cuda.synchronize()
    del prompt, output, token, cache


def _print_results(
    environment: dict[str, Any],
    correctness: list[CorrectnessResult],
    prefill: list[PrefillResult],
    decode: list[DecodeResult],
    generation: list[GenerationResult],
    audit: RuntimeAudit | None,
) -> None:
    print("\nEnvironment")
    for name, value in environment.items():
        print(f"  {name}: {value}")
    print("\nCorrectness")
    print(
        f"{'effective':>9} {'logits max':>12} {'K max':>12} {'V max':>12} "
        f"{'positions':>10} {'addresses':>10} {'greedy':>8}"
    )
    for item in correctness:
        print(
            f"{item.effective_length:>9} {item.max_logits_error:>12.6g} "
            f"{item.max_key_error:>12.6g} {item.max_value_error:>12.6g} "
            f"{str(item.positions_correct):>10} {str(item.stable_addresses):>10} "
            f"{str(item.greedy_equal):>8}"
        )
    print("\nPrefill (last-token logits, use_cache=True)")
    print(
        f"{'tokens':>8} {'reference':>11} {'Flux eager':>11} {'native':>11} "
        f"{'native tok/s':>12} {'HF/native':>10} {'Flux/native':>12} "
        f"{'HF/Flux err':>12} {'native err':>11}"
    )
    for item in prefill:
        print(
            f"{item.length:>8} {item.reference_ms:>11.3f} {item.flux_ms:>11.3f} "
            f"{item.native_ms:>11.3f} {1000 * item.length / item.native_ms:>12.1f} "
            f"{item.reference_ms / item.native_ms:>9.3f}x "
            f"{item.flux_ms / item.native_ms:>11.3f}x "
            f"{item.max_logits_error:>12.6g} {item.max_native_logits_error:>11.6g}"
        )
    print("\nNative prefill cache and persistent storage")
    for item in prefill:
        print(
            f"  {item.length}: K={item.max_native_key_error:.6g}, "
            f"V={item.max_native_value_error:.6g}, "
            f"position/length={item.cache_position}/{item.cache_length}, "
            f"addresses={item.stable_addresses}, cache={item.cache_bytes} B, "
            f"prefill_workspace={item.prefill_workspace_bytes} B, "
            f"decode_workspace={item.decode_workspace_bytes} B, "
            f"stable_buffers={item.stable_buffer_bytes} B"
        )
    print("\nSteady-state decode (effective attention length; median window shown)")
    print(
        f"{'length':>8} {'window':>13} {'reference':>11} {'Flux eager':>11} "
        f"{'native':>11} {'native/ref':>10} {'native/eager':>12}"
    )
    for item in decode:
        print(
            f"{item.effective_length:>8} "
            f"{item.window_start:>5}-{item.window_stop:<5} "
            f"{item.reference_ms:>11.4f} {item.flux_eager_ms:>11.4f} "
            f"{item.native_ms:>11.4f} "
            f"{item.reference_ms / item.native_ms:>9.3f}x "
            f"{item.flux_eager_ms / item.native_ms:>11.3f}x"
        )
    if generation:
        print("\nGeneration (CUDA-event execution; native observed total includes setup)")
        print(
            f"{'prompt':>7} {'path':>11} {'prefill':>10} {'TTFT':>10} "
            f"{'decode':>10} {'total exec':>11} {'total seen':>11} {'gen tok/s':>10}"
        )
        for item in generation:
            print(
                f"{item.prompt_length:>7} {item.path:>11} {item.prefill_ms:>10.3f} "
                f"{item.ttft_ms:>10.3f} {item.decode_ms:>10.3f} "
                f"{item.total_execution_ms:>11.3f} {item.total_observed_ms:>11.3f} "
                f"{1000 * item.output_tokens / item.total_observed_ms:>10.2f}"
            )
    if audit is not None:
        print("\nCUDA-Graph runtime audit")
        for name, value in asdict(audit).items():
            print(f"  {name}: {value}")



# --- Full-system profile mode ---

@dataclass(frozen=True)
class ProfileRow:
    workload: str
    size: int
    repetitions: int
    median_ms: float
    profiled_ms: float
    launches: int
    allocation_growth_bytes: int
    stable_addresses: bool
    owner_counts: dict[str, int]
    owner_ms: dict[str, float]
    top_kernels_ms: tuple[tuple[str, int, float], ...]

def _owner(name: str) -> str:
    lowered = name.lower()
    if 'memcpy' in lowered or 'memset' in lowered:
        return 'CUDA state/copy'
    if 'gqa_decode' in lowered or 'streaming_prefill' in lowered:
        return 'Flux attention'
    if '_zn4flux' in lowered or 'flux' in lowered:
        return 'Flux other'
    if 'cublas' in lowered or 'gemm' in lowered or 'gemv' in lowered:
        return 'cuBLAS/cuBLASLt'
    return 'framework CUDA'

def _short_name(name: str) -> str:
    for marker, short in (('streaming_prefill', 'streaming prefill GQA'), ('gqa_decode_attention_grouped_chunk', 'decode GQA grouped chunk'), ('gqa_decode_attention_reduce', 'decode GQA reduce'), ('packed_gate_up_swiglu', 'fused gate/up + SwiGLU'), ('packed_qkv_rope_cache', 'packed QKV/RoPE/cache'), ('residual_rmsnorm', 'residual-RMSNorm'), ('rmsnorm', 'RMSNorm'), ('prepare_full_decode', 'decode state prepare'), ('advance_full_decode', 'decode state advance'), ('gemv', 'GEMV'), ('gemm', 'GEMM'), ('memcpy', 'CUDA memcpy'), ('memset', 'CUDA memset')):
        if marker in name.lower():
            return short
    return name if len(name) <= 100 else name[:97] + '...'

def _profile_runtime(workload: str, size: int, runtime: NativeSmolLM2Prefill, operation: Callable[[], object], warmup: int, samples: int, repetitions: int, top_k: int) -> ProfileRow:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    medians = [_event_sample(operation)[0] for _ in range(samples)]
    addresses = runtime.stable_addresses()
    allocated = torch.cuda.memory_allocated()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        start.record()
        for _ in range(repetitions):
            operation()
        end.record()
    end.synchronize()
    growth = torch.cuda.memory_allocated() - allocated
    owner_counts: Counter[str] = Counter()
    owner_times: dict[str, float] = defaultdict(float)
    kernel_counts: Counter[str] = Counter()
    kernel_times: dict[str, float] = defaultdict(float)
    for event in profiler.events():
        if event.device_type != DeviceType.CUDA:
            continue
        milliseconds = float(event.self_device_time_total) / 1000.0 / repetitions
        owner = _owner(event.name)
        short = _short_name(event.name)
        owner_counts[owner] += 1
        owner_times[owner] += milliseconds
        kernel_counts[short] += 1
        kernel_times[short] += milliseconds
    owners = sorted(set(owner_counts) | set(owner_times))
    top = sorted(kernel_times, key=kernel_times.get, reverse=True)[:top_k]
    return ProfileRow(workload=workload, size=size, repetitions=repetitions, median_ms=statistics.median(medians), profiled_ms=start.elapsed_time(end) / repetitions, launches=round(sum(owner_counts.values()) / repetitions), allocation_growth_bytes=growth, stable_addresses=addresses == runtime.stable_addresses(), owner_counts={name: round(owner_counts[name] / repetitions) for name in owners}, owner_ms={name: owner_times[name] for name in owners}, top_kernels_ms=tuple(((name, round(kernel_counts[name] / repetitions), kernel_times[name]) for name in top)))

def _decode_profile(model: torch.nn.Module, context: int, args: argparse.Namespace) -> ProfileRow:
    total = args.warmup + args.samples + args.repetitions
    runtime = NativeSmolLM2Prefill.capture(model, _input_ids(context - total, model.config.vocab_size), max_decode_steps=total)
    return _profile_runtime('native decode', context, runtime, runtime.replay, args.warmup, args.samples, args.repetitions, args.top_k)

def _prefill_profile(model: torch.nn.Module, length: int, args: argparse.Namespace) -> ProfileRow:
    ids = _input_ids(length, model.config.vocab_size)
    runtime = NativeSmolLM2Prefill.capture(model, ids)
    return _profile_runtime('native prefill', length, runtime, lambda: runtime.prefill(ids), args.warmup, args.samples, args.repetitions, args.top_k)

def _print(rows: list[ProfileRow]) -> None:
    for row in rows:
        print(f'\n{row.workload}, size={row.size}: median={row.median_ms:.4f} ms, profiled={row.profiled_ms:.4f} ms, launches={row.launches}, allocation_growth={row.allocation_growth_bytes}, stable_addresses={row.stable_addresses}')
        print('  Owners')
        for owner, milliseconds in sorted(row.owner_ms.items(), key=lambda item: item[1], reverse=True):
            print(f'    {owner:<20} {row.owner_counts[owner]:>4} launches {milliseconds:>10.4f} ms')
        print('  Top kernels')
        for name, launches, milliseconds in row.top_kernels_ms:
            print(f'    {name:<60} {launches:>4} {milliseconds:>10.4f} ms')


def _run_profile(args: argparse.Namespace) -> int:
    configure_runtime(seed=SEED)
    model = enable_flux_ops(
        load_model("cuda"), operators=FINAL_FLUX_OPERATOR_CATEGORIES
    )
    maximum = int(model.config.max_position_embeddings)
    if any(length > maximum for length in args.prefill_lengths):
        raise ValueError(f"prefill length exceeds model maximum {maximum}")
    decode_steps = args.warmup + args.samples + args.repetitions
    if any(
        context > maximum or context <= decode_steps
        for context in args.decode_contexts
    ):
        raise ValueError(
            "decode contexts must exceed all profiling steps and fit the model maximum"
        )
    print(
        f"Model: {MODEL_ID} @ {MODEL_REVISION}\n"
        f"GPU: {torch.cuda.get_device_name()}\n"
        f"PyTorch/CUDA: {torch.__version__} / {torch.version.cuda}\n"
        "Backend: native prefill and attached native decode"
    )
    rows: list[ProfileRow] = []
    if not args.skip_decode:
        for context in args.decode_contexts:
            rows.append(_decode_profile(model, context, args))
    if not args.skip_prefill:
        for length in args.prefill_lengths:
            rows.append(_prefill_profile(model, length, args))
    _print(rows)
    if args.json_output is not None:
        payload = {
            "mode": "profile",
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "rows": [asdict(row) for row in rows],
        }
        args.json_output.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nWrote {args.json_output}")
    return 0


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.mode == "profile":
        return _run_profile(args)
    configure_runtime(seed=SEED)
    print("Loading reference and final Flux model instances...", flush=True)
    reference = load_model("cuda")
    flux = enable_flux_ops(
        load_model("cuda"), operators=FINAL_FLUX_OPERATOR_CATEGORIES
    )
    maximum = int(reference.config.max_position_embeddings)
    if args.mode in {"system", "prefill", "decode"} and any(
        length > maximum for length in args.lengths
    ):
        raise ValueError(f"lengths exceed model maximum {maximum}")
    if args.mode in {"system", "generation"} and any(
        prompt + args.output_tokens - 1 > maximum
        for prompt in args.generation_prompts
    ):
        raise ValueError("generation prompt plus decode tokens exceeds model maximum")
    print(f"Final Flux categories: {sorted(FINAL_FLUX_OPERATOR_CATEGORIES)}")
    print(f"Installed modules: {flux_operator_counts(flux)}")
    reference_state = reference.state_dict()
    flux_state = flux.state_dict()
    if reference_state.keys() != flux_state.keys():
        raise AssertionError("reference and Flux state-dict keys differ")
    for name in reference_state:
        if not torch.equal(reference_state[name], flux_state[name]):
            raise AssertionError(f"checkpoint tensor differs: {name}")
    del reference_state, flux_state
    print("Checkpoint/state-dict compatibility: exact", flush=True)
    _stabilize(flux, args.stabilization_iterations)

    correctness = []
    prefill_results = []
    decode_results = []
    if args.mode in {"system", "prefill", "decode"}:
        for length in args.lengths:
            print(f"Validating effective length {length}...", flush=True)
            correctness.append(
                _validate_decode(reference, flux, length, args.correctness_tokens)
            )
            if args.mode in {"system", "prefill"}:
                print(f"Benchmarking prefill length {length}...", flush=True)
                prefill_results.append(
                    _benchmark_prefill(
                        reference,
                        flux,
                        length,
                        args.warmup,
                        args.samples,
                        args.rounds,
                    )
                )
            if args.mode in {"system", "decode"}:
                print(
                    f"Benchmarking decode through effective length {length}...",
                    flush=True,
                )
                decode_results.append(
                    _benchmark_decode(
                        reference,
                        flux,
                        length,
                        args.warmup,
                        args.samples,
                        args.rounds,
                    )
                )

    generation_results = []
    if args.mode in {"system", "generation"} and not args.skip_generation:
        for prompt_length in args.generation_prompts:
            print(
                f"Benchmarking {args.output_tokens}-token generation at prompt "
                f"{prompt_length}...",
                flush=True,
            )
            generation_results.extend(
                _benchmark_generation(
                    reference,
                    flux,
                    prompt_length,
                    args.output_tokens,
                    args.generation_repetitions,
                )
            )
    audit = None
    if args.mode in {"system", "decode"} and not args.skip_audit:
        print(f"Auditing native runtime at capacity {args.audit_capacity}...", flush=True)
        audit = _runtime_audit(flux, args.audit_capacity, args.audit_replays)

    environment = _environment(args)
    _print_results(
        environment,
        correctness,
        prefill_results,
        decode_results,
        generation_results,
        audit,
    )
    if args.json_output is not None:
        payload = {
            "mode": args.mode,
            "environment": environment,
            "flux_modules": flux_operator_counts(flux),
            "correctness": [asdict(item) for item in correctness],
            "prefill": [asdict(item) for item in prefill_results],
            "decode": [asdict(item) for item in decode_results],
            "generation": [asdict(item) for item in generation_results],
            "runtime_audit": None if audit is None else asdict(audit),
        }
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
