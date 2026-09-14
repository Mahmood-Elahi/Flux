"""Validate and benchmark the full native SmolLM2 one-token decode runtime."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.profiler import DeviceType, ProfilerActivity, profile
from transformers import DynamicCache

from benchmarks.smollm2_benchmark_utils import (
    ATOL,
    RTOL,
    alternating_event_medians,
    configure_runtime,
    deterministic_input_ids,
    parse_positive_int_list,
)
from flux.model import FINAL_FLUX_OPERATOR_CATEGORIES, enable_flux_ops
from flux.model.smollm2 import MODEL_ID, MODEL_REVISION, load_model
from flux.model.smollm2_cuda_graph import FluxCUDAGraphDecode
from flux.runtime import NativeSmolLM2Decode


DEFAULT_LENGTHS = (128, 512, 1024, 2048, 4096, 8192)


@dataclass(frozen=True)
class DecodeResult:
    effective_length: int
    reference_ms: float
    flux_eager_ms: float
    flux_graph_ms: float
    native_graph_ms: float
    reference_tokens_per_second: float
    flux_eager_tokens_per_second: float
    flux_graph_tokens_per_second: float
    native_graph_tokens_per_second: float
    native_vs_python_graph_speedup: float
    reference_vs_native_speedup: float
    python_graph_host_us: float
    native_graph_host_us: float
    max_reference_native_logit_error: float
    max_flux_native_logit_error: float
    max_python_graph_native_logit_error: float
    max_key_error: float
    max_value_error: float
    max_key_error_layer: int
    max_value_error_layer: int
    token_identity: bool
    stable_addresses: bool


@dataclass(frozen=True)
class AuditResult:
    capacity: int
    replays: int
    launches_per_replay: int
    flux_launches_per_replay: int
    cublas_launches_per_replay: int
    framework_launches_per_replay: int
    gqa_launches_per_replay: int
    state_launches_per_replay: int
    allocation_growth_bytes: int
    synchronize_events: int
    allocation_events: int
    stable_addresses: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=parse_positive_int_list, default=DEFAULT_LENGTHS)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--generation-prompt", type=int, default=128)
    parser.add_argument("--generation-tokens", type=int, default=16)
    parser.add_argument("--audit-capacity", type=int, default=4096)
    parser.add_argument("--audit-replays", type=int, default=10)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 1:
        parser.error("warmup may be zero and samples must be positive")
    if args.generation_prompt < 1 or args.generation_tokens < 2:
        parser.error("generation prompt must be positive and tokens at least two")
    if args.audit_replays < 1 or args.audit_capacity <= args.audit_replays + 3:
        parser.error("audit capacity must exceed audit replays plus three")
    return args


def _clone_cache(cache: DynamicCache, config: Any) -> DynamicCache:
    data = []
    for layer in cache.layers:
        data.append((layer.keys.detach().clone(), layer.values.detach().clone()))
    return DynamicCache(data, config=config)


def _max_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    return float((actual - expected).abs().max().item())


def _host_enqueue_median_us(
    prepare: Callable[[], Callable[[], object]], samples: int
) -> float:
    values = []
    for _ in range(samples):
        operation = prepare()
        started = time.perf_counter_ns()
        operation()
        values.append((time.perf_counter_ns() - started) / 1000.0)
        torch.cuda.synchronize()
    return statistics.median(values)


@torch.inference_mode()
def _benchmark_length(
    reference: torch.nn.Module,
    flux: torch.nn.Module,
    effective_length: int,
    warmup: int,
    samples: int,
) -> DecodeResult:
    prompt = deterministic_input_ids(effective_length - 1, flux.config.vocab_size)
    token = deterministic_input_ids(1, flux.config.vocab_size)
    with torch.inference_mode():
        reference_prefill = reference(input_ids=prompt, use_cache=True, logits_to_keep=1)
        flux_prefill = flux(input_ids=prompt, use_cache=True, logits_to_keep=1)
    python_graph = FluxCUDAGraphDecode.capture(
        flux, prompt, max_decode_steps=1, warmup_steps=3
    )
    native = NativeSmolLM2Decode.capture(
        flux, prompt, max_decode_steps=1, initial_token=token
    )

    with torch.inference_mode():
        reference_output = reference(
            input_ids=token,
            past_key_values=_clone_cache(reference_prefill.past_key_values, reference.config),
            use_cache=True,
            logits_to_keep=1,
        )
        flux_output = flux(
            input_ids=token,
            past_key_values=_clone_cache(flux_prefill.past_key_values, flux.config),
            use_cache=True,
            logits_to_keep=1,
        )
    python_logits = python_graph.replay(token)
    native_logits = native.replay(token)
    torch.cuda.synchronize()
    reference_error = _max_error(native_logits, reference_output.logits)
    flux_error = _max_error(native_logits, flux_output.logits)
    python_graph_error = _max_error(native_logits, python_logits)

    max_key = 0.0
    max_value = 0.0
    max_key_layer = 0
    max_value_layer = 0
    for index, layer in enumerate(python_graph.cache.layers):
        key_error = _max_error(
            native.key_cache[index, ..., :effective_length, :],
            layer.keys[..., :effective_length, :],
        )
        value_error = _max_error(
            native.value_cache[index, ..., :effective_length, :],
            layer.values[..., :effective_length, :],
        )
        if key_error > max_key:
            max_key, max_key_layer = key_error, index
        if value_error > max_value:
            max_value, max_value_layer = value_error, index

    addresses = native.stable_addresses()

    def prepare_reference() -> Callable[[], object]:
        cache = _clone_cache(reference_prefill.past_key_values, reference.config)
        return lambda: reference(
            input_ids=token, past_key_values=cache, use_cache=True, logits_to_keep=1
        )

    def prepare_flux_eager() -> Callable[[], object]:
        cache = _clone_cache(flux_prefill.past_key_values, flux.config)
        return lambda: flux(
            input_ids=token, past_key_values=cache, use_cache=True, logits_to_keep=1
        )

    def prepare_python_graph() -> Callable[[], object]:
        python_graph._restore_logical_state(clear_decode_slots=True)
        python_graph.input_ids.copy_(token)
        python_graph.steps_replayed = 0
        torch.cuda.synchronize()
        return lambda: python_graph.replay(None)

    source_keys = [layer.keys for layer in flux_prefill.past_key_values.layers]
    source_values = [layer.values for layer in flux_prefill.past_key_values.layers]

    def prepare_native() -> Callable[[], object]:
        native.reset(
            token,
            source_keys,
            source_values,
            cache_position=effective_length - 1,
        )
        return lambda: native.replay(None)

    medians = alternating_event_medians(
        {
            "reference": lambda: None,
            "Flux eager": lambda: None,
            "Flux graph": lambda: None,
            "native graph": lambda: None,
        },
        warmup,
        samples,
        prepare={
            "reference": prepare_reference,
            "Flux eager": prepare_flux_eager,
            "Flux graph": prepare_python_graph,
            "native graph": prepare_native,
        },
    )
    python_host = _host_enqueue_median_us(prepare_python_graph, samples)
    native_host = _host_enqueue_median_us(prepare_native, samples)
    stable = addresses == native.stable_addresses()
    token_identity = (
        native_logits.argmax(dim=-1).item()
        == python_logits.argmax(dim=-1).item()
        == flux_output.logits.argmax(dim=-1).item()
        == reference_output.logits.argmax(dim=-1).item()
    )
    return DecodeResult(
        effective_length,
        medians["reference"],
        medians["Flux eager"],
        medians["Flux graph"],
        medians["native graph"],
        1000.0 / medians["reference"],
        1000.0 / medians["Flux eager"],
        1000.0 / medians["Flux graph"],
        1000.0 / medians["native graph"],
        medians["Flux graph"] / medians["native graph"],
        medians["reference"] / medians["native graph"],
        python_host,
        native_host,
        reference_error,
        flux_error,
        python_graph_error,
        max_key,
        max_value,
        max_key_layer,
        max_value_layer,
        token_identity,
        stable,
    )


def _greedy(model: torch.nn.Module, prompt: torch.Tensor, count: int) -> torch.Tensor:
    with torch.inference_mode():
        output = model(input_ids=prompt, use_cache=True, logits_to_keep=1)
        token = output.logits.argmax(dim=-1)
        generated = [prompt, token]
        for _ in range(count - 1):
            output = model(
                input_ids=token,
                past_key_values=output.past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            token = output.logits.argmax(dim=-1)
            generated.append(token)
    return torch.cat(generated, dim=-1)


@torch.inference_mode()
def _validate_generation(
    reference: torch.nn.Module,
    flux: torch.nn.Module,
    prompt_length: int,
    count: int,
) -> dict[str, Any]:
    prompt = deterministic_input_ids(prompt_length, flux.config.vocab_size)
    reference_tokens = _greedy(reference, prompt, count)
    flux_tokens = _greedy(flux, prompt, count)
    runtime = NativeSmolLM2Decode.capture(
        flux, prompt, max_decode_steps=count - 1
    )
    token = runtime.prefill_logits.argmax(dim=-1)
    generated = [prompt, token]
    for _ in range(count - 1):
        token = runtime.replay(token).argmax(dim=-1)
        generated.append(token)
    native_tokens = torch.cat(generated, dim=-1)
    torch.cuda.synchronize()
    return {
        "prompt_length": prompt_length,
        "generated_tokens": count,
        "reference_flux_exact": bool(torch.equal(reference_tokens, flux_tokens)),
        "reference_native_exact": bool(torch.equal(reference_tokens, native_tokens)),
        "flux_native_exact": bool(torch.equal(flux_tokens, native_tokens)),
        "token_ids": native_tokens[0, prompt_length:].tolist(),
    }


def _owner(name: str) -> str:
    lowered = name.lower()
    if "gqa_decode" in lowered:
        return "gqa"
    if "prepare_full_decode" in lowered or "advance_full_decode" in lowered:
        return "state"
    if "_zn4flux" in lowered or "flux" in lowered:
        return "Flux"
    if "cublas" in lowered or "gemm" in lowered or "gemv" in lowered:
        return "cuBLAS"
    return "framework"


@torch.inference_mode()
def _audit(flux: torch.nn.Module, capacity: int, replays: int) -> AuditResult:
    prompt = deterministic_input_ids(capacity - replays - 3, flux.config.vocab_size)
    runtime = NativeSmolLM2Decode.capture(
        flux, prompt, max_decode_steps=replays + 3
    )
    for _ in range(3):
        runtime.replay(None)
    torch.cuda.synchronize()
    addresses = runtime.stable_addresses()
    allocated = torch.cuda.memory_allocated()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        for _ in range(replays):
            runtime.replay(None)
    torch.cuda.synchronize()
    cuda_events = [
        event for event in profiler.events() if event.device_type == DeviceType.CUDA
    ]
    owners = {name: 0 for name in ("Flux", "cuBLAS", "framework", "gqa", "state")}
    for event in cuda_events:
        owners[_owner(event.name)] += 1
    cpu_names = [
        event.name.lower()
        for event in profiler.events()
        if event.device_type == DeviceType.CPU
    ]
    return AuditResult(
        capacity,
        replays,
        round(len(cuda_events) / replays),
        round((owners["Flux"] + owners["gqa"] + owners["state"]) / replays),
        round(owners["cuBLAS"] / replays),
        round(owners["framework"] / replays),
        round(owners["gqa"] / replays),
        round(owners["state"] / replays),
        torch.cuda.memory_allocated() - allocated,
        sum("synchronize" in name for name in cpu_names),
        sum("cudamalloc" in name or "cudafree" in name for name in cpu_names),
        addresses == runtime.stable_addresses(),
    )


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    configure_runtime(seed=0)
    print("Loading pinned reference and Flux models...", flush=True)
    reference = load_model("cuda")
    flux = enable_flux_ops(load_model("cuda"), operators=FINAL_FLUX_OPERATOR_CATEGORIES)
    reference_state = reference.state_dict()
    flux_state = flux.state_dict()
    if reference_state.keys() != flux_state.keys():
        raise AssertionError("reference and Flux state-dict keys differ")
    for name in reference_state:
        if not torch.equal(reference_state[name], flux_state[name]):
            raise AssertionError(f"checkpoint tensor differs: {name}")
    del reference_state, flux_state
    print("Checkpoint/state-dict compatibility: exact", flush=True)
    results = []
    for length in args.lengths:
        print(f"Validating and benchmarking effective length {length}...", flush=True)
        results.append(
            _benchmark_length(reference, flux, length, args.warmup, args.samples)
        )
    generation = None
    if not args.skip_generation:
        generation = _validate_generation(
            reference, flux, args.generation_prompt, args.generation_tokens
        )
    audit = None if args.skip_audit else _audit(
        flux, args.audit_capacity, args.audit_replays
    )
    memory_runtime = NativeSmolLM2Decode.capture(
        flux,
        deterministic_input_ids(1, flux.config.vocab_size),
        max_decode_steps=max(args.lengths) - 1,
    )
    payload = {
        "environment": {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "dtype": "torch.float32",
            "tf32": False,
        },
        "decode": [asdict(result) for result in results],
        "generation": generation,
        "audit": None if audit is None else asdict(audit),
        "memory_at_maximum_capacity": asdict(memory_runtime.memory),
    }
    print(json.dumps(payload, indent=2))
    if args.json_output is not None:
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
