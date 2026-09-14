"""Validate and benchmark native SmolLM2 prompt-to-cache prefill.

The harness compares the pinned Hugging Face model, retained Python Flux path,
and native prefill on identical deterministic FP32 prompts. Setup, model
loading, runtime construction, and correctness checks remain outside CUDA-event
timing. Native timing exercises fixed-shape reuse of persistent storage.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from benchmarks.smollm2_benchmark_utils import (
    configure_runtime,
    deterministic_input_ids,
    parse_positive_int_list,
)
from flux.model.smollm2 import MODEL_ID, MODEL_REVISION, load_model
from flux.model.smollm2_flux import FINAL_FLUX_OPERATOR_CATEGORIES, enable_flux_ops
from flux.runtime import NativeSmolLM2Prefill


RTOL = 2e-4
ATOL = 2e-5
LENGTHS = (128, 512, 1024, 2048, 4096, 8192)


@dataclass(frozen=True)
class Result:
    length: int
    huggingface_ms: float
    python_flux_ms: float
    native_flux_ms: float
    huggingface_tokens_per_second: float
    python_flux_tokens_per_second: float
    native_flux_tokens_per_second: float
    native_vs_huggingface: float
    native_vs_python_flux: float
    native_huggingface_logit_error: float
    native_python_flux_logit_error: float
    maximum_huggingface_key_error: float
    maximum_huggingface_value_error: float
    maximum_python_flux_key_error: float
    maximum_python_flux_value_error: float
    maximum_key_layer: int
    maximum_value_layer: int
    maximum_key_index: tuple[int, ...]
    maximum_value_index: tuple[int, ...]
    cache_position: int
    cache_length: int
    continuation_tokens: int
    maximum_continuation_huggingface_error: float
    maximum_continuation_python_flux_error: float
    greedy_identity: bool
    cache_bytes: int
    prefill_workspace_bytes: int
    decode_workspace_bytes: int
    stable_buffer_bytes: int


@dataclass(frozen=True)
class Audit:
    length: int
    repetitions: int
    launches_per_prefill: int
    native_launches_per_prefill: int
    library_launches_per_prefill: int
    framework_launches_per_prefill: int
    native_cuda_ms_per_prefill: float
    library_cuda_ms_per_prefill: float
    framework_cuda_ms_per_prefill: float
    allocation_growth_bytes: int
    allocator_events: int
    synchronization_events: int
    stable_addresses: bool
    top_cuda_events_ms: tuple[tuple[str, float], ...]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=parse_positive_int_list, default=LENGTHS)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--continuation-tokens", type=int, default=3)
    parser.add_argument("--audit-length", type=int, default=1024)
    parser.add_argument("--audit-repetitions", type=int, default=5)
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument(
        "--report-cache-drift",
        action="store_true",
        help=("record reordered long-context cache deltas without applying the "
              "elementwise Flux-cache assertion; logits and continuation "
              "checks remain enforced"),
    )
    parser.add_argument("--json-output", type=Path)
    result = parser.parse_args()
    if result.warmup < 0 or result.samples < 1:
        parser.error("warmup may be zero and samples must be positive")
    if result.continuation_tokens < 0 or result.audit_repetitions < 1:
        parser.error("continuation may be zero and audit repetitions must be positive")
    return result


def _maximum_error(
    actual: torch.Tensor, expected: torch.Tensor, *, assert_close: bool = True
) -> float:
    result = float((actual - expected).abs().max().item())
    if assert_close:
        torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    return result


def _time(operation: Callable[[], Any], warmup: int, samples: int) -> float:
    output = None
    for _ in range(warmup):
        output = operation()
    torch.cuda.synchronize()
    pairs = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = operation()
        end.record()
        pairs.append((start, end))
    pairs[-1][1].synchronize()
    result = statistics.median(start.elapsed_time(end) for start, end in pairs)
    del output
    return result


def _cache_errors(
    native: torch.Tensor,
    expected_layers: list[torch.Tensor],
    length: int,
    *,
    assert_close: bool = True,
) -> tuple[float, int, tuple[int, ...]]:
    maximum = 0.0
    maximum_layer = 0
    maximum_index: tuple[int, ...] = ()
    for index, expected in enumerate(expected_layers):
        actual = native[index, ..., :length, :]
        difference = (actual - expected).abs()
        error = float(difference.max().item())
        if assert_close:
            torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
        if error > maximum:
            maximum = error
            maximum_layer = index
            flat_index = int(difference.argmax().item())
            coordinates = []
            for size in reversed(difference.shape):
                coordinates.append(flat_index % size)
                flat_index //= size
            maximum_index = tuple(reversed(coordinates))
    return maximum, maximum_layer, maximum_index


def _validate_and_time(
    reference: torch.nn.Module,
    flux: torch.nn.Module,
    length: int,
    warmup: int,
    samples: int,
    requested_continuation: int,
    report_cache_drift: bool = False,
) -> tuple[Result, NativeSmolLM2Prefill]:
    input_ids = deterministic_input_ids(length, reference.config.vocab_size)
    continuation = min(requested_continuation, 8192 - length)
    with torch.inference_mode():
        reference_output = reference(
            input_ids=input_ids, use_cache=True, logits_to_keep=1
        )
        flux_output = flux(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        native = NativeSmolLM2Prefill.capture(
            flux, input_ids, max_decode_steps=continuation
        )
    torch.cuda.synchronize()

    native_hf_logits = _maximum_error(native.logits, reference_output.logits)
    native_flux_logits = _maximum_error(native.logits, flux_output.logits)
    # The established packed-QKV Flux path is the exact implementation oracle
    # for cache contents.  It is not elementwise-close to HF at every long-
    # context cache element because one packed GEMM and HF's separate Q/K/V
    # GEMMs have different FP32 accumulation orders.  Record the HF deltas, but
    # assert the native caches against Flux without changing the repository's
    # established tolerances.
    hf_key, hf_key_layer, hf_key_index = _cache_errors(
        native.key_cache,
        [layer.keys for layer in reference_output.past_key_values.layers],
        length,
        assert_close=False,
    )
    hf_value, hf_value_layer, hf_value_index = _cache_errors(
        native.value_cache,
        [layer.values for layer in reference_output.past_key_values.layers],
        length,
        assert_close=False,
    )
    flux_key, flux_key_layer, flux_key_index = _cache_errors(
        native.key_cache,
        [layer.keys for layer in flux_output.past_key_values.layers],
        length,
        assert_close=not report_cache_drift,
    )
    flux_value, flux_value_layer, flux_value_index = _cache_errors(
        native.value_cache,
        [layer.values for layer in flux_output.past_key_values.layers],
        length,
        assert_close=not report_cache_drift,
    )

    native_hf_continuation = 0.0
    native_flux_continuation = 0.0
    greedy_identity = True
    reference_token = reference_output.logits.argmax(dim=-1)
    flux_token = flux_output.logits.argmax(dim=-1)
    native_token = native.logits.argmax(dim=-1)
    for _ in range(continuation):
        with torch.inference_mode():
            reference_output = reference(
                input_ids=reference_token,
                past_key_values=reference_output.past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            flux_output = flux(
                input_ids=flux_token,
                past_key_values=flux_output.past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            native_logits = native.replay(native_token)
        torch.cuda.synchronize()
        native_hf_continuation = max(
            native_hf_continuation,
            _maximum_error(native_logits, reference_output.logits),
        )
        native_flux_continuation = max(
            native_flux_continuation,
            _maximum_error(native_logits, flux_output.logits),
        )
        reference_token = reference_output.logits.argmax(dim=-1)
        flux_token = flux_output.logits.argmax(dim=-1)
        native_token = native_logits.argmax(dim=-1)
        greedy_identity &= bool(
            torch.equal(reference_token, flux_token)
            and torch.equal(reference_token, native_token)
        )

    # Restore prompt state before timing native reuse.
    with torch.inference_mode():
        native.prefill(input_ids)
    paths: tuple[tuple[str, Callable[[], Any]], ...] = (
        (
            "huggingface",
            lambda: reference(input_ids=input_ids, use_cache=True, logits_to_keep=1),
        ),
        (
            "python_flux",
            lambda: flux(input_ids=input_ids, use_cache=True, logits_to_keep=1),
        ),
        ("native_flux", lambda: native.prefill(input_ids)),
    )
    timings: dict[str, float] = {}
    with torch.inference_mode():
        for name, operation in paths:
            timings[name] = _time(operation, warmup, samples)

    memory = native.memory
    result = Result(
        length=length,
        huggingface_ms=timings["huggingface"],
        python_flux_ms=timings["python_flux"],
        native_flux_ms=timings["native_flux"],
        huggingface_tokens_per_second=1000.0 * length / timings["huggingface"],
        python_flux_tokens_per_second=1000.0 * length / timings["python_flux"],
        native_flux_tokens_per_second=1000.0 * length / timings["native_flux"],
        native_vs_huggingface=timings["huggingface"] / timings["native_flux"],
        native_vs_python_flux=timings["python_flux"] / timings["native_flux"],
        native_huggingface_logit_error=native_hf_logits,
        native_python_flux_logit_error=native_flux_logits,
        maximum_huggingface_key_error=hf_key,
        maximum_huggingface_value_error=hf_value,
        maximum_python_flux_key_error=flux_key,
        maximum_python_flux_value_error=flux_value,
        maximum_key_layer=(hf_key_layer if hf_key >= flux_key else flux_key_layer),
        maximum_value_layer=(
            hf_value_layer if hf_value >= flux_value else flux_value_layer
        ),
        maximum_key_index=(
            hf_key_index if hf_key >= flux_key else flux_key_index
        ),
        maximum_value_index=(
            hf_value_index if hf_value >= flux_value else flux_value_index
        ),
        cache_position=native.cache_position,
        cache_length=native.cache_length,
        continuation_tokens=continuation,
        maximum_continuation_huggingface_error=native_hf_continuation,
        maximum_continuation_python_flux_error=native_flux_continuation,
        greedy_identity=greedy_identity,
        cache_bytes=memory.cache_bytes,
        prefill_workspace_bytes=memory.prefill_workspace_bytes,
        decode_workspace_bytes=memory.decode_workspace_bytes,
        stable_buffer_bytes=memory.stable_buffer_bytes,
    )
    del reference_output, flux_output, input_ids
    return result, native


def _owner(name: str) -> str:
    lowered = name.lower()
    if "native_prefill" in lowered or "flux" in lowered or "memcpy" in lowered:
        return "native"
    if (
        "cublas" in lowered
        or "gemm" in lowered
        or "gemv" in lowered
        or "memset" in lowered
    ):
        return "library"
    return "framework"


def _audit(runtime: NativeSmolLM2Prefill, repetitions: int) -> Audit:
    for _ in range(2):
        runtime.prefill()
    torch.cuda.synchronize()
    addresses = runtime.stable_addresses()
    allocated_before = torch.cuda.memory_allocated()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        for _ in range(repetitions):
            runtime.prefill()
        torch.cuda.synchronize()
    allocated_after = torch.cuda.memory_allocated()
    cuda_events = [
        event for event in profiler.events() if event.device_type == DeviceType.CUDA
    ]
    owners = {"native": 0, "library": 0, "framework": 0}
    owner_times = {"native": 0.0, "library": 0.0, "framework": 0.0}
    event_times: dict[str, float] = {}
    for event in cuda_events:
        owner = _owner(event.name)
        owners[owner] += 1
        milliseconds = float(event.self_device_time_total) / 1000.0
        owner_times[owner] += milliseconds
        event_times[event.name] = event_times.get(event.name, 0.0) + milliseconds
    cpu_names = [
        event.name.lower()
        for event in profiler.events()
        if event.device_type == DeviceType.CPU
    ]
    return Audit(
        length=runtime.prompt_length,
        repetitions=repetitions,
        launches_per_prefill=round(len(cuda_events) / repetitions),
        native_launches_per_prefill=round(owners["native"] / repetitions),
        library_launches_per_prefill=round(owners["library"] / repetitions),
        framework_launches_per_prefill=round(owners["framework"] / repetitions),
        native_cuda_ms_per_prefill=owner_times["native"] / repetitions,
        library_cuda_ms_per_prefill=owner_times["library"] / repetitions,
        framework_cuda_ms_per_prefill=owner_times["framework"] / repetitions,
        allocation_growth_bytes=allocated_after - allocated_before,
        allocator_events=sum(
            "cudamalloc" in name or "cudafree" in name for name in cpu_names
        ),
        synchronization_events=sum("synchronize" in name for name in cpu_names),
        stable_addresses=addresses == runtime.stable_addresses(),
        top_cuda_events_ms=tuple(
            sorted(event_times.items(), key=lambda item: item[1], reverse=True)[:12]
        ),
    )


def _print(results: list[Result], audit: Audit | None) -> None:
    print("\nNative prefill results")
    print(
        f"{'length':>7} {'HF ms':>10} {'Python ms':>10} {'native ms':>10} "
        f"{'native tok/s':>13} {'vs HF':>8} {'vs Python':>10} {'logit max':>11}"
    )
    for row in results:
        print(
            f"{row.length:7d} {row.huggingface_ms:10.3f} "
            f"{row.python_flux_ms:10.3f} {row.native_flux_ms:10.3f} "
            f"{row.native_flux_tokens_per_second:13.1f} "
            f"{row.native_vs_huggingface:7.3f}x "
            f"{row.native_vs_python_flux:9.3f}x "
            f"{row.native_huggingface_logit_error:11.6g}"
        )
    print("\nCorrectness and state")
    for row in results:
        print(
            f"  {row.length}: K={row.maximum_huggingface_key_error:.6g} "
            f"V={row.maximum_huggingface_value_error:.6g}; "
            f"position/length={row.cache_position}/{row.cache_length}; "
            f"continuation={row.maximum_continuation_huggingface_error:.6g}; "
            f"greedy={row.greedy_identity}"
        )
    if audit is not None:
        print("\nRuntime audit")
        for name, value in asdict(audit).items():
            print(f"  {name}: {value}")


def main() -> int:
    args = _args()
    configure_runtime(seed=0)
    print("Loading pinned Hugging Face and Flux models...", flush=True)
    reference = load_model("cuda")
    flux = enable_flux_ops(
        load_model("cuda"), operators=FINAL_FLUX_OPERATOR_CATEGORIES
    )
    results = []
    audit_runtime = None
    for length in args.lengths:
        if length > 8192:
            raise ValueError("native prefill length exceeds 8192")
        print(f"Validating and timing length {length}...", flush=True)
        result, runtime = _validate_and_time(
            reference,
            flux,
            length,
            args.warmup,
            args.samples,
            args.continuation_tokens,
            args.report_cache_drift,
        )
        results.append(result)
        if length == args.audit_length:
            audit_runtime = runtime
        else:
            del runtime
    audit = None
    if not args.skip_audit:
        if audit_runtime is None:
            ids = deterministic_input_ids(args.audit_length, flux.config.vocab_size)
            audit_runtime = NativeSmolLM2Prefill.capture(flux, ids)
        audit = _audit(audit_runtime, args.audit_repetitions)
    _print(results, audit)
    if args.json_output is not None:
        payload = {
            "environment": {
                "model": MODEL_ID,
                "revision": MODEL_REVISION,
                "pytorch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "dtype": "torch.float32",
                "rtol": RTOL,
                "atol": ATOL,
                "warmup": args.warmup,
                "samples": args.samples,
                "statistic": "CUDA-event median",
                "report_cache_drift": args.report_cache_drift,
            },
            "results": [asdict(result) for result in results],
            "audit": None if audit is None else asdict(audit),
        }
        args.json_output.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nWrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
