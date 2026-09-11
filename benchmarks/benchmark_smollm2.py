"""Benchmark and profile reference versus Flux-integrated SmolLM2 inference.

Run from the repository root after building the native extension:

    build/python3119/python.exe benchmarks/benchmark_smollm2.py

The benchmark compares distinct model instances loaded from the same pinned
FP32 checkpoint. Timed regions contain model forwards only; model loading,
input construction, cache cloning, correctness checks, and profiler setup are
excluded. CUDA event samples are alternated between implementations and the
reported primary statistic is median latency.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import statistics
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile
from transformers import DynamicCache

from flux.model.smollm2 import MODEL_ID, MODEL_REVISION, load_model
from flux.model.smollm2_flux import enable_flux_ops, flux_operator_counts
from flux.ops import (
    native_residual_rmsnorm_is_available,
    native_rmsnorm_is_available,
    native_softmax_is_available,
)


DTYPE = torch.float32
SEED = 0
RTOL = 2e-4
ATOL = 2e-5
DEFAULT_WARMUP = 5
DEFAULT_REPETITIONS = 30
DEFAULT_SEQUENCE_LENGTHS = (1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
DEFAULT_DECODE_CONTEXTS = (16, 128, 512, 1024, 2048, 4096, 8192)
DEFAULT_PROFILE_LENGTHS = (128, 1024, 2048)
CORRECTNESS_CHUNK_TOKENS = 256


@dataclass(frozen=True)
class Comparison:
    size: int
    reference_ms: float
    flux_ms: float
    max_absolute_error: float

    @property
    def speedup(self) -> float:
        return self.reference_ms / self.flux_ms


@dataclass(frozen=True)
class MemoryResult:
    path: str
    workload: str
    size: int
    baseline_mib: float
    peak_mib: float
    incremental_peak_mib: float


@dataclass(frozen=True)
class ProfileResult:
    path: str
    workload: str
    size: int
    baseline_ms: float
    event_ms: float
    kernel_ms: float
    components_ms: dict[str, float]
    custom_counts: dict[str, int]
    custom_ms: dict[str, float]


def _parse_int_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("lengths must be positive integers")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument(
        "--sequence-lengths",
        type=_parse_int_list,
        default=DEFAULT_SEQUENCE_LENGTHS,
        help="comma-separated full-forward and prefill lengths",
    )
    parser.add_argument(
        "--decode-contexts",
        type=_parse_int_list,
        default=DEFAULT_DECODE_CONTEXTS,
        help="comma-separated cached-decode context lengths",
    )
    parser.add_argument(
        "--profile-lengths",
        type=_parse_int_list,
        default=DEFAULT_PROFILE_LENGTHS,
        help="comma-separated prefill lengths to profile",
    )
    parser.add_argument("--memory-length", type=int, default=1024)
    parser.add_argument("--skip-profiles", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.memory_length < 1:
        parser.error("--memory-length must be positive")
    return args


def _configure_runtime() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _print_environment(args: argparse.Namespace) -> None:
    print("SmolLM2 model benchmark environment")
    print(f"  Model: {MODEL_ID}")
    print(f"  Revision: {MODEL_REVISION}")
    print(f"  Python executable: {sys.executable}")
    print(f"  Python version: {sys.version.split()[0]}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  PyTorch CUDA build: {torch.version.cuda}")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    capability = torch.cuda.get_device_capability()
    print(f"  GPU compute capability: {capability[0]}.{capability[1]}")
    print(f"  dtype: {DTYPE}")
    print("  attention backend: eager")
    print("  TF32: disabled")
    print("  deterministic algorithms: enabled")
    print(f"  deterministic input seed: {SEED}")
    print(f"  warm-up calls per implementation/configuration: {args.warmup}")
    print(f"  measured CUDA-event samples per implementation: {args.repetitions}")
    print(f"  correctness tolerances: rtol={RTOL}, atol={ATOL}")


def _input_ids(length: int, vocab_size: int) -> torch.Tensor:
    values = (torch.arange(length, dtype=torch.long) * 17 + 11) % vocab_size
    return values.unsqueeze(0).to("cuda")


def _cache_length(cache: DynamicCache) -> int:
    return int(cache.get_seq_length())


def _clone_cache(cache: DynamicCache, config: Any) -> DynamicCache:
    """Clone a populated dynamic cache before a one-token timed decode.

    DynamicCache grows in place. Constructing one clone per sample outside its
    CUDA-event interval keeps every measured decode at the requested context
    length instead of silently timing progressively longer contexts.
    """
    data = []
    for layer in cache.layers:
        if layer.keys is None or layer.values is None:
            data.append((None, None))
        else:
            data.append((layer.keys.detach().clone(), layer.values.detach().clone()))
    return DynamicCache(data, config=config)


def _assert_logits_close(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.shape != expected.shape:
        raise AssertionError(f"logit shape mismatch: {actual.shape} != {expected.shape}")
    max_absolute = 0.0
    maximum_excess = 0.0
    total_failures = 0
    for start in range(0, actual.shape[1], CORRECTNESS_CHUNK_TOKENS):
        stop = min(start + CORRECTNESS_CHUNK_TOKENS, actual.shape[1])
        actual_chunk = actual[:, start:stop]
        expected_chunk = expected[:, start:stop]
        difference = (actual_chunk - expected_chunk).abs()
        allowed = ATOL + RTOL * expected_chunk.abs()
        if not bool(torch.all(torch.isfinite(actual_chunk))):
            raise AssertionError("Flux logits contain non-finite values")
        excess = difference - allowed
        total_failures += int(torch.count_nonzero(excess > 0).item())
        maximum_excess = max(maximum_excess, float(excess.max().item()))
        max_absolute = max(max_absolute, float(difference.max().item()))
    torch.cuda.synchronize()
    if total_failures:
        raise AssertionError(
            f"logits differ beyond rtol={RTOL}, atol={ATOL}; "
            f"failures={total_failures}/{actual.numel()}, "
            f"max absolute error={max_absolute:.9g}, "
            f"max tolerance excess={maximum_excess:.9g}"
        )
    return max_absolute


def _event_latencies(
    operations: dict[str, Callable[[], object]],
    warmup: int,
    repetitions: int,
    prepare: dict[str, Callable[[], Callable[[], object]]] | None = None,
) -> dict[str, float]:
    outputs: dict[str, object] = {}
    names = tuple(operations)
    for warmup_index in range(warmup):
        order = names if warmup_index % 2 == 0 else tuple(reversed(names))
        for name in order:
            operation = prepare[name]() if prepare is not None else operations[name]
            outputs[name] = operation()
    torch.cuda.synchronize()

    events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
        name: [] for name in names
    }
    final_event = None
    for sample_index in range(repetitions):
        order = names if sample_index % 2 == 0 else tuple(reversed(names))
        for name in order:
            operation = prepare[name]() if prepare is not None else operations[name]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            outputs[name] = operation()
            end.record()
            events[name].append((start, end))
            final_event = end

    assert final_event is not None
    final_event.synchronize()
    medians = {
        name: statistics.median(start.elapsed_time(end) for start, end in samples)
        for name, samples in events.items()
    }
    del outputs
    return medians


def _full_or_prefill_correctness(
    reference: torch.nn.Module,
    flux: torch.nn.Module,
    input_ids: torch.Tensor,
    use_cache: bool,
) -> float:
    reference_output = reference(input_ids=input_ids, use_cache=use_cache)
    flux_output = flux(input_ids=input_ids, use_cache=use_cache)
    maximum = _assert_logits_close(flux_output.logits, reference_output.logits)
    if use_cache:
        reference_length = _cache_length(reference_output.past_key_values)
        flux_length = _cache_length(flux_output.past_key_values)
        if reference_length != flux_length or reference_length != input_ids.shape[1]:
            raise AssertionError(
                f"prefill cache lengths differ: reference={reference_length}, "
                f"Flux={flux_length}, expected={input_ids.shape[1]}"
            )
    del reference_output, flux_output
    return maximum


def _benchmark_full_or_prefill(
    reference: torch.nn.Module,
    flux: torch.nn.Module,
    input_ids: torch.Tensor,
    use_cache: bool,
    warmup: int,
    repetitions: int,
) -> Comparison:
    maximum = _full_or_prefill_correctness(reference, flux, input_ids, use_cache)
    operations = {
        "reference": lambda: reference(input_ids=input_ids, use_cache=use_cache),
        "Flux": lambda: flux(input_ids=input_ids, use_cache=use_cache),
    }
    medians = _event_latencies(operations, warmup, repetitions)
    return Comparison(input_ids.shape[1], medians["reference"], medians["Flux"], maximum)


def _prefill_cache(model: torch.nn.Module, input_ids: torch.Tensor) -> DynamicCache:
    output = model(input_ids=input_ids, use_cache=True)
    cache = output.past_key_values
    del output
    return cache


def _benchmark_decode(
    reference: torch.nn.Module,
    flux: torch.nn.Module,
    context_ids: torch.Tensor,
    next_token: torch.Tensor,
    warmup: int,
    repetitions: int,
) -> Comparison:
    reference_cache = _prefill_cache(reference, context_ids)
    flux_cache = _prefill_cache(flux, context_ids)
    context_length = context_ids.shape[1]
    if _cache_length(reference_cache) != context_length:
        raise AssertionError("reference prefill cache has an unexpected length")
    if _cache_length(flux_cache) != context_length:
        raise AssertionError("Flux prefill cache has an unexpected length")

    reference_check_cache = _clone_cache(reference_cache, reference.config)
    flux_check_cache = _clone_cache(flux_cache, flux.config)
    reference_output = reference(
        input_ids=next_token,
        past_key_values=reference_check_cache,
        use_cache=True,
    )
    flux_output = flux(
        input_ids=next_token,
        past_key_values=flux_check_cache,
        use_cache=True,
    )
    maximum = _assert_logits_close(flux_output.logits, reference_output.logits)
    expected_length = context_length + 1
    lengths = (
        _cache_length(reference_output.past_key_values),
        _cache_length(flux_output.past_key_values),
    )
    if lengths != (expected_length, expected_length):
        raise AssertionError(
            f"decode cache lengths differ: reference={lengths[0]}, "
            f"Flux={lengths[1]}, expected={expected_length}"
        )
    del reference_output, flux_output, reference_check_cache, flux_check_cache

    def make_decode(model: torch.nn.Module, base: DynamicCache) -> Callable[[], object]:
        cache = _clone_cache(base, model.config)
        return lambda: model(
            input_ids=next_token,
            past_key_values=cache,
            use_cache=True,
        )

    operations = {"reference": lambda: None, "Flux": lambda: None}
    prepare = {
        "reference": lambda: make_decode(reference, reference_cache),
        "Flux": lambda: make_decode(flux, flux_cache),
    }
    medians = _event_latencies(operations, warmup, repetitions, prepare=prepare)
    del reference_cache, flux_cache
    return Comparison(context_length, medians["reference"], medians["Flux"], maximum)


def _clear_after_oom() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _run_comparisons(
    reference: torch.nn.Module,
    flux: torch.nn.Module,
    args: argparse.Namespace,
) -> tuple[list[Comparison], list[Comparison], list[Comparison]]:
    full_results = []
    prefill_results = []
    decode_results = []
    maximum_position = int(reference.config.max_position_embeddings)

    for length in args.sequence_lengths:
        if length > maximum_position:
            print(f"SKIP full/prefill length {length}: exceeds max position {maximum_position}")
            continue
        input_ids = _input_ids(length, reference.config.vocab_size)
        try:
            print(f"Measuring full forward length {length}...", flush=True)
            full_results.append(
                _benchmark_full_or_prefill(
                    reference, flux, input_ids, False, args.warmup, args.repetitions
                )
            )
        except AssertionError as error:
            print(f"SKIP full forward length {length}: correctness guard failed: {error}")
        except torch.OutOfMemoryError:
            print(f"SKIP full forward length {length}: CUDA out of memory")
            _clear_after_oom()
        try:
            print(f"Measuring prefill length {length}...", flush=True)
            prefill_results.append(
                _benchmark_full_or_prefill(
                    reference, flux, input_ids, True, args.warmup, args.repetitions
                )
            )
        except AssertionError as error:
            print(f"SKIP prefill length {length}: correctness guard failed: {error}")
        except torch.OutOfMemoryError:
            print(f"SKIP prefill length {length}: CUDA out of memory")
            _clear_after_oom()
        del input_ids

    for context_length in args.decode_contexts:
        if context_length >= maximum_position:
            print(
                f"SKIP decode context {context_length}: no room for next token "
                f"within max position {maximum_position}"
            )
            continue
        context_ids = _input_ids(context_length, reference.config.vocab_size)
        next_token = _input_ids(context_length + 1, reference.config.vocab_size)[:, -1:]
        try:
            print(f"Measuring cached decode after context {context_length}...", flush=True)
            decode_results.append(
                _benchmark_decode(
                    reference,
                    flux,
                    context_ids,
                    next_token,
                    args.warmup,
                    args.repetitions,
                )
            )
        except AssertionError as error:
            print(
                f"SKIP decode context {context_length}: "
                f"correctness guard failed: {error}"
            )
        except torch.OutOfMemoryError:
            print(f"SKIP decode context {context_length}: CUDA out of memory")
            _clear_after_oom()
        del context_ids, next_token
    return full_results, prefill_results, decode_results


def _geometric_mean_speedup(results: Sequence[Comparison]) -> float | None:
    if not results:
        return None
    return math.exp(statistics.fmean(math.log(result.speedup) for result in results))


def _format_geometric_mean(results: Sequence[Comparison]) -> str:
    value = _geometric_mean_speedup(results)
    return "n/a" if value is None else f"{value:.3f}x"


def _print_forward_table(title: str, results: Sequence[Comparison]) -> None:
    print(f"\n{title}")
    print(f"{'sequence':>9} {'reference ms':>14} {'Flux ms':>12} {'speedup':>10} {'max abs err':>13}")
    for result in results:
        print(
            f"{result.size:>9} {result.reference_ms:>14.3f} "
            f"{result.flux_ms:>12.3f} {result.speedup:>9.3f}x "
            f"{result.max_absolute_error:>13.6g}"
        )


def _print_decode_table(results: Sequence[Comparison]) -> None:
    print("\nCached decode")
    print(
        f"{'context':>9} {'reference ms/token':>19} {'Flux ms/token':>15} "
        f"{'reference tok/s':>17} {'Flux tok/s':>12} {'speedup':>10} "
        f"{'max abs err':>13}"
    )
    for result in results:
        print(
            f"{result.size:>9} {result.reference_ms:>19.3f} "
            f"{result.flux_ms:>15.3f} {1000.0 / result.reference_ms:>17.2f} "
            f"{1000.0 / result.flux_ms:>12.2f} {result.speedup:>9.3f}x "
            f"{result.max_absolute_error:>13.6g}"
        )


def _measure_memory_once(
    path: str,
    workload: str,
    size: int,
    operation: Callable[[], object],
) -> MemoryResult:
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = operation()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del output
    torch.cuda.empty_cache()
    mib = 1024.0**2
    return MemoryResult(path, workload, size, baseline / mib, peak / mib, (peak - baseline) / mib)


def _measure_memory(
    reference: torch.nn.Module,
    flux: torch.nn.Module,
    length: int,
) -> list[MemoryResult]:
    input_ids = _input_ids(length, reference.config.vocab_size)
    next_token = _input_ids(length + 1, reference.config.vocab_size)[:, -1:]
    results = []
    for path, model in (("reference", reference), ("Flux", flux)):
        results.append(
            _measure_memory_once(
                path,
                "prefill",
                length,
                lambda model=model: model(input_ids=input_ids, use_cache=True),
            )
        )
        base_cache = _prefill_cache(model, input_ids)
        torch.cuda.synchronize()
        results.append(
            _measure_memory_once(
                path,
                "decode",
                length,
                lambda model=model, cache=base_cache: model(
                    input_ids=next_token,
                    past_key_values=cache,
                    use_cache=True,
                ),
            )
        )
        del base_cache
    del input_ids, next_token
    return results


def _shape(value: object) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)) and all(isinstance(item, int) for item in value):
        return tuple(value)
    return ()


def _profile_components(profiler: Any) -> dict[str, float]:
    """Classify non-overlapping outer ATen/custom-op CUDA times in milliseconds."""
    totals_us = {
        "qkv projections": 0.0,
        "QK matmul": 0.0,
        "attention mask/elementwise": 0.0,
        "softmax": 0.0,
        "P@V matmul": 0.0,
        "output projection": 0.0,
        "MLP gate/up projections": 0.0,
        "SiLU/gating": 0.0,
        "MLP down projection": 0.0,
        "RMSNorm": 0.0,
        "residual + RMSNorm": 0.0,
        "residual adds": 0.0,
        "RoPE (excluding cat)": 0.0,
        "GQA repeat copies": 0.0,
        "LM head": 0.0,
        "embedding": 0.0,
    }
    square_projection_index = 0
    for event in profiler.events():
        if event.device_type != DeviceType.CPU or event.device_time_total <= 0:
            continue
        shapes = event.input_shapes or []
        first = _shape(shapes[0]) if shapes else ()
        second = _shape(shapes[1]) if len(shapes) > 1 else ()
        duration = float(event.device_time_total)

        if event.name == "aten::linear" and len(second) == 2:
            output_width, input_width = second
            if (output_width, input_width) == (576, 576):
                category = "qkv projections" if square_projection_index % 2 == 0 else "output projection"
                square_projection_index += 1
                totals_us[category] += duration
            elif (output_width, input_width) == (192, 576):
                totals_us["qkv projections"] += duration
            elif (output_width, input_width) == (1536, 576):
                totals_us["MLP gate/up projections"] += duration
            elif (output_width, input_width) == (576, 1536):
                totals_us["MLP down projection"] += duration
            elif output_width == 49152:
                totals_us["LM head"] += duration
        elif event.name == "aten::bmm" and len(first) == 3 and len(second) == 3:
            if first[-1] == 64 and second[-2] == 64:
                totals_us["QK matmul"] += duration
            elif second[-1] == 64:
                totals_us["P@V matmul"] += duration
        elif event.name in {"aten::_softmax", "flux::softmax"}:
            totals_us["softmax"] += duration
        elif event.name == "aten::rms_norm" or event.name == "flux::rmsnorm":
            totals_us["RMSNorm"] += duration
        elif event.name == "flux::residual_rmsnorm":
            totals_us["residual + RMSNorm"] += duration
        elif event.name in {"aten::pow", "aten::mean", "aten::rsqrt"} and (
            (first and first[-1] in {1, 576})
        ):
            totals_us["RMSNorm"] += duration
        elif event.name == "aten::add" and len(first) == 3 and first[-1] == 1:
            totals_us["RMSNorm"] += duration
        elif event.name == "aten::mul" and (
            (len(first) == 3 and first[-1] == 576 and len(second) == 3 and second[-1] == 1)
            or (first == (576,) and len(second) == 3 and second[-1] == 576)
        ):
            totals_us["RMSNorm"] += duration
        elif event.name == "aten::silu":
            totals_us["SiLU/gating"] += duration
        elif event.name == "aten::mul" and first and first[-1] == 1536:
            totals_us["SiLU/gating"] += duration
        elif event.name == "aten::add" and first and first[-1] == 576 and len(first) == 3:
            totals_us["residual adds"] += duration
        elif event.name == "aten::neg" and len(first) == 4 and first[-1] == 32:
            totals_us["RoPE (excluding cat)"] += duration
        elif (
            event.name in {"aten::mul", "aten::add"}
            and len(first) == 4
            and first[-1] == 64
            and len(second) == 4
            and second[-1] == 64
        ):
            totals_us["RoPE (excluding cat)"] += duration
        elif event.name == "aten::copy_" and len(first) == 5 and first[-1] == 64:
            totals_us["GQA repeat copies"] += duration
        elif event.name == "aten::embedding":
            totals_us["embedding"] += duration
        elif (
            (event.name == "aten::mul" and len(first) == 4 and first[-2] != 64)
            or (event.name == "aten::add" and len(first) == 4 and first[-2] != 64)
            or event.name in {"aten::le", "aten::where", "aten::fill_"}
        ):
            totals_us["attention mask/elementwise"] += duration
    return {name: value / 1000.0 for name, value in totals_us.items()}


def _profile_operation(
    path: str,
    workload: str,
    size: int,
    baseline_ms: float,
    operation: Callable[[], object],
) -> ProfileResult:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as profiler:
        start.record()
        output = operation()
        end.record()
    end.synchronize()
    event_ms = start.elapsed_time(end)
    kernel_ms = sum(
        float(event.self_device_time_total)
        for event in profiler.events()
        if event.device_type == DeviceType.CUDA
    ) / 1000.0
    components = _profile_components(profiler)
    averages = {event.key: event for event in profiler.key_averages()}
    custom_names = ("flux::rmsnorm", "flux::residual_rmsnorm", "flux::softmax")
    custom_counts = {
        name: int(averages[name].count) if name in averages else 0 for name in custom_names
    }
    custom_ms = {
        name: float(averages[name].device_time_total) / 1000.0 if name in averages else 0.0
        for name in custom_names
    }
    del output
    return ProfileResult(
        path,
        workload,
        size,
        baseline_ms,
        event_ms,
        kernel_ms,
        components,
        custom_counts,
        custom_ms,
    )


def _run_profiles(
    reference: torch.nn.Module,
    flux: torch.nn.Module,
    profile_lengths: Sequence[int],
    prefill_results: Sequence[Comparison],
    decode_results: Sequence[Comparison],
) -> list[ProfileResult]:
    results = []
    prefill_baselines = {
        ("reference", result.size): result.reference_ms for result in prefill_results
    }
    prefill_baselines.update(
        {("Flux", result.size): result.flux_ms for result in prefill_results}
    )
    decode_baselines = {
        ("reference", result.size): result.reference_ms for result in decode_results
    }
    decode_baselines.update(
        {("Flux", result.size): result.flux_ms for result in decode_results}
    )
    for length in profile_lengths:
        if length > reference.config.max_position_embeddings or ("reference", length) not in prefill_baselines:
            continue
        input_ids = _input_ids(length, reference.config.vocab_size)
        for path, model in (("reference", reference), ("Flux", flux)):
            print(f"Profiling {path} prefill length {length}...", flush=True)
            results.append(
                _profile_operation(
                    path,
                    "prefill",
                    length,
                    prefill_baselines[(path, length)],
                    lambda model=model: model(input_ids=input_ids, use_cache=True),
                )
            )
        del input_ids

    context_length = 1024
    if ("reference", context_length) not in decode_baselines:
        print("Skipping decode profile: context 1024 was not benchmarked")
        return results
    context_ids = _input_ids(context_length, reference.config.vocab_size)
    next_token = _input_ids(context_length + 1, reference.config.vocab_size)[:, -1:]
    for path, model in (("reference", reference), ("Flux", flux)):
        base_cache = _prefill_cache(model, context_ids)
        cache = _clone_cache(base_cache, model.config)
        print(f"Profiling {path} decode after context {context_length}...", flush=True)
        results.append(
            _profile_operation(
                path,
                "decode",
                context_length,
                decode_baselines[(path, context_length)],
                lambda model=model, cache=cache: model(
                    input_ids=next_token,
                    past_key_values=cache,
                    use_cache=True,
                ),
            )
        )
        del base_cache, cache
    del context_ids, next_token
    return results


def _print_memory(results: Sequence[MemoryResult]) -> None:
    print("\nRepresentative CUDA peak allocated memory")
    print(
        f"{'path':>10} {'workload':>10} {'size':>7} {'baseline MiB':>14} "
        f"{'peak MiB':>12} {'increment MiB':>14}"
    )
    for result in results:
        print(
            f"{result.path:>10} {result.workload:>10} {result.size:>7} "
            f"{result.baseline_mib:>14.1f} {result.peak_mib:>12.1f} "
            f"{result.incremental_peak_mib:>14.1f}"
        )


def _print_profiles(results: Sequence[ProfileResult]) -> None:
    print("\nFocused profiler component breakdown")
    print("Component time is inclusive CUDA time attributed to non-overlapping outer ops.")
    for result in results:
        launch_gap = max(0.0, result.baseline_ms - result.kernel_ms)
        print(
            f"\n{result.path} {result.workload} {result.size}: "
            f"unprofiled median={result.baseline_ms:.3f} ms, "
            f"profiled event={result.event_ms:.3f} ms, kernels={result.kernel_ms:.3f} ms, "
            f"approx. kernel-gap/launch={launch_gap:.3f} ms "
            f"({100.0 * launch_gap / result.baseline_ms:.1f}% of unprofiled median)"
        )
        print(f"  {'component':<28} {'ms':>9} {'baseline share':>15}")
        for name, milliseconds in result.components_ms.items():
            if milliseconds > 0:
                print(
                    f"  {name:<28} {milliseconds:>9.3f} "
                    f"{100.0 * milliseconds / result.baseline_ms:>11.1f}%"
                )
        gemm = sum(
            result.components_ms[name]
            for name in (
                "qkv projections",
                "QK matmul",
                "P@V matmul",
                "output projection",
                "MLP gate/up projections",
                "MLP down projection",
                "LM head",
            )
        )
        attention = sum(
            result.components_ms[name]
            for name in (
                "qkv projections",
                "QK matmul",
                "attention mask/elementwise",
                "softmax",
                "P@V matmul",
                "output projection",
                "RoPE (excluding cat)",
                "GQA repeat copies",
            )
        )
        mlp = sum(
            result.components_ms[name]
            for name in (
                "MLP gate/up projections",
                "SiLU/gating",
                "MLP down projection",
            )
        )
        print(
            f"  grouped: GEMMs={gemm:.3f} ms ({100.0 * gemm / result.baseline_ms:.1f}%), "
            f"attention={attention:.3f} ms ({100.0 * attention / result.baseline_ms:.1f}%), "
            f"MLP={mlp:.3f} ms ({100.0 * mlp / result.baseline_ms:.1f}%)"
        )
        if result.path == "Flux":
            print("  Flux custom operators:")
            for name in ("flux::rmsnorm", "flux::residual_rmsnorm", "flux::softmax"):
                count = result.custom_counts[name]
                total_ms = result.custom_ms[name]
                per_call_us = total_ms * 1000.0 / count if count else 0.0
                print(
                    f"    {name:<28} calls={count:>2}, total={total_ms:.3f} ms, "
                    f"per-call={per_call_us:.3f} us, "
                    f"share={100.0 * total_ms / result.baseline_ms:.1f}%"
                )


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the SmolLM2 model benchmark")
    if not all(
        (
            native_rmsnorm_is_available(),
            native_residual_rmsnorm_is_available(),
            native_softmax_is_available(),
        )
    ):
        raise RuntimeError("all three built Flux native operators are required")
    _configure_runtime()
    _print_environment(args)

    print("\nLoading two independent copies of the pinned FP32 model...", flush=True)
    reference = load_model("cuda")
    flux = enable_flux_ops(load_model("cuda"))
    print(f"  reference parameters: {sum(parameter.numel() for parameter in reference.parameters()):,}")
    print(f"  Flux parameters: {sum(parameter.numel() for parameter in flux.parameters()):,}")
    print(f"  Flux modules: {flux_operator_counts(flux)}")
    reference_state = reference.state_dict()
    flux_state = flux.state_dict()
    if reference_state.keys() != flux_state.keys():
        raise AssertionError("reference and Flux state-dict keys differ")
    for name in reference_state:
        if not torch.equal(reference_state[name], flux_state[name]):
            raise AssertionError(f"independently loaded checkpoint tensor differs: {name}")
    print("  checkpoint tensors: all identical")

    with torch.inference_mode():
        full, prefill, decode = _run_comparisons(reference, flux, args)
        memory = _measure_memory(reference, flux, args.memory_length)
        profiles = [] if args.skip_profiles else _run_profiles(
            reference,
            flux,
            args.profile_lengths,
            prefill,
            decode,
        )

    _print_forward_table("Full forward (use_cache=False)", full)
    _print_forward_table("Prefill (use_cache=True)", prefill)
    _print_decode_table(decode)
    print("\nGeometric-mean speedups (reference / Flux)")
    print(f"  full forward: {_format_geometric_mean(full)}")
    print(f"  prefill:      {_format_geometric_mean(prefill)}")
    print(f"  decode:       {_format_geometric_mean(decode)}")
    _print_memory(memory)
    if profiles:
        _print_profiles(profiles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
