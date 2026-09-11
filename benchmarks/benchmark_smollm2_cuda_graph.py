"""Validate, benchmark, and profile fixed-shape SmolLM2 CUDA-Graph decode.

Run from the repository root after building the Flux native extension:

    build/python3119/python.exe benchmarks/benchmark_smollm2_cuda_graph.py

Prefill, graph capture, model loading, input construction, and tokenizer work
are excluded from per-token decode timing.  Practical timings include greedy
argmax, stable-input update, and the one-token model/replay dispatch.
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile
from transformers import DynamicCache

from flux.model.smollm2 import MODEL_ID, MODEL_REVISION, load_model
from flux.model.smollm2_cuda_graph import FluxCUDAGraphDecode
from flux.model.smollm2_flux import enable_flux_ops, flux_operator_counts
from flux.ops import (
    native_residual_rmsnorm_is_available,
    native_rmsnorm_is_available,
    native_rope_is_available,
    native_softmax_is_available,
)


CONTEXTS = (16, 128, 512, 1024, 2048, 4096)
RTOL = 2e-4
ATOL = 2e-5
MIB = 1024.0**2


@dataclass(frozen=True)
class CorrectnessResult:
    context: int
    max_logits_error: float
    max_key_error: float
    max_value_error: float
    positions_correct: bool
    greedy_equal: bool


@dataclass(frozen=True)
class TimingResult:
    context: int
    reference_ms: float
    flux_eager_ms: float
    graph_practical_ms: float
    graph_device_ms: float
    graph_setup_ms: float
    graph_capture_ms: float


@dataclass(frozen=True)
class MemoryResult:
    context: int
    eager_cache_mib: float
    eager_peak_mib: float
    static_cache_mib: float
    graph_pool_mib: float
    graph_setup_peak_mib: float
    replay_growth_bytes: int


@dataclass(frozen=True)
class ProfileResult:
    path: str
    event_ms: float
    kernel_ms: float


def _parse_int_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("contexts must be positive")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=_parse_int_list, default=CONTEXTS)
    parser.add_argument("--correctness-tokens", type=int, default=8)
    parser.add_argument("--warmup-tokens", type=int, default=5)
    parser.add_argument("--timing-tokens", type=int, default=30)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--memory-context", type=int, default=1024)
    parser.add_argument("--profile-context", type=int, default=1024)
    parser.add_argument("--skip-profile", action="store_true")
    args = parser.parse_args()
    if args.correctness_tokens < 2:
        parser.error("--correctness-tokens must be at least 2")
    if args.warmup_tokens < 0 or args.timing_tokens < 1 or args.repetitions < 1:
        parser.error("timing tokens must be positive (warmup may be zero)")
    return args


def _configure_runtime() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _input_ids(length: int, vocab_size: int) -> torch.Tensor:
    values = (torch.arange(length, dtype=torch.long) * 17 + 11) % vocab_size
    return values.unsqueeze(0).cuda()


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = (actual - expected).abs()
    maximum = float(difference.max().item())
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    return maximum


def _validate_context(
    reference: torch.nn.Module,
    flux: torch.nn.Module,
    context: int,
    tokens: int,
) -> CorrectnessResult:
    prompt = _input_ids(context, reference.config.vocab_size)
    with torch.inference_mode():
        reference_output = reference(input_ids=prompt, use_cache=True, logits_to_keep=1)
        flux_output = flux(input_ids=prompt, use_cache=True, logits_to_keep=1)
        reference_cache = reference_output.past_key_values
        flux_cache = flux_output.past_key_values
        graph = FluxCUDAGraphDecode.capture(
            flux,
            prompt,
            max_decode_steps=tokens - 1,
        )
        reference_token = reference_output.logits.argmax(dim=-1)
        flux_token = flux_output.logits.argmax(dim=-1)
        graph_token = graph.prefill_logits.argmax(dim=-1)
        reference_tokens = [reference_token]
        flux_tokens = [flux_token]
        graph_tokens = [graph_token]
        maximum_logits = _assert_close(graph.prefill_logits, flux_output.logits)
        maximum_key = 0.0
        maximum_value = 0.0
        positions_correct = graph.cache_position == context

        for step in range(tokens - 1):
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
            graph_logits = graph.replay(graph_token)
            maximum_logits = max(
                maximum_logits, _assert_close(graph_logits, flux_output.logits)
            )
            expected_length = context + step + 1
            positions_correct &= (
                graph.cache_position == expected_length
                and int(graph.cache.get_seq_length().item()) == expected_length
            )
            for graph_layer, flux_layer in zip(
                graph.cache.layers, flux_cache.layers, strict=True
            ):
                graph_keys = graph_layer.keys[..., :expected_length, :]
                graph_values = graph_layer.values[..., :expected_length, :]
                maximum_key = max(
                    maximum_key,
                    float((graph_keys - flux_layer.keys).abs().max().item()),
                )
                maximum_value = max(
                    maximum_value,
                    float((graph_values - flux_layer.values).abs().max().item()),
                )
                torch.testing.assert_close(graph_keys, flux_layer.keys, rtol=0, atol=0)
                torch.testing.assert_close(graph_values, flux_layer.values, rtol=0, atol=0)
            reference_token = reference_output.logits.argmax(dim=-1)
            flux_token = flux_output.logits.argmax(dim=-1)
            graph_token = graph_logits.argmax(dim=-1)
            reference_tokens.append(reference_token)
            flux_tokens.append(flux_token)
            graph_tokens.append(graph_token)

    greedy_equal = torch.equal(
        torch.cat(reference_tokens, dim=-1), torch.cat(flux_tokens, dim=-1)
    ) and torch.equal(
        torch.cat(flux_tokens, dim=-1), torch.cat(graph_tokens, dim=-1)
    )
    del prompt, reference_cache, flux_cache, graph
    return CorrectnessResult(
        context,
        maximum_logits,
        maximum_key,
        maximum_value,
        positions_correct,
        greedy_equal,
    )


def _time_eager_loop(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    warmup_tokens: int,
    timing_tokens: int,
) -> float:
    with torch.inference_mode():
        output = model(input_ids=prompt, use_cache=True, logits_to_keep=1)
        cache = output.past_key_values
        token = output.logits.argmax(dim=-1)
        for _ in range(warmup_tokens):
            output = model(
                input_ids=token,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            token = output.logits.argmax(dim=-1)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(timing_tokens):
            output = model(
                input_ids=token,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            token = output.logits.argmax(dim=-1)
        torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0 / timing_tokens


def _time_graph_practical(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    warmup_tokens: int,
    timing_tokens: int,
) -> tuple[float, float, float]:
    graph = FluxCUDAGraphDecode.capture(
        model,
        prompt,
        max_decode_steps=warmup_tokens + timing_tokens,
    )
    setup_ms = graph.setup_timing.total_ms
    capture_ms = graph.setup_timing.capture_ms
    token = graph.prefill_logits.argmax(dim=-1)
    for _ in range(warmup_tokens):
        token = graph.replay(token).argmax(dim=-1)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(timing_tokens):
        token = graph.replay(token).argmax(dim=-1)
    torch.cuda.synchronize()
    result = (time.perf_counter() - started) * 1000.0 / timing_tokens
    del graph
    return result, setup_ms, capture_ms


def _time_pure_graph(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    warmup_tokens: int,
    timing_tokens: int,
) -> float:
    graph = FluxCUDAGraphDecode.capture(
        model,
        prompt,
        max_decode_steps=warmup_tokens + timing_tokens,
    )
    for _ in range(warmup_tokens):
        graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(timing_tokens):
        graph.replay()
    end.record()
    end.synchronize()
    result = start.elapsed_time(end) / timing_tokens
    del graph
    return result


def _benchmark_context(
    reference: torch.nn.Module,
    flux: torch.nn.Module,
    context: int,
    warmup_tokens: int,
    timing_tokens: int,
    repetitions: int,
) -> TimingResult:
    prompt = _input_ids(context, reference.config.vocab_size)
    samples: dict[str, list[float]] = {
        "reference": [],
        "Flux": [],
        "graph_practical": [],
        "graph_device": [],
        "graph_setup": [],
        "graph_capture": [],
    }
    for repetition in range(repetitions):
        eager_order = (
            (("reference", reference), ("Flux", flux))
            if repetition % 2 == 0
            else (("Flux", flux), ("reference", reference))
        )
        for name, model in eager_order:
            samples[name].append(
                _time_eager_loop(model, prompt, warmup_tokens, timing_tokens)
            )
        practical_ms, setup_ms, capture_ms = _time_graph_practical(
            flux, prompt, warmup_tokens, timing_tokens
        )
        samples["graph_practical"].append(practical_ms)
        samples["graph_setup"].append(setup_ms)
        samples["graph_capture"].append(capture_ms)
        samples["graph_device"].append(
            _time_pure_graph(flux, prompt, warmup_tokens, timing_tokens)
        )
    del prompt
    return TimingResult(
        context,
        statistics.median(samples["reference"]),
        statistics.median(samples["Flux"]),
        statistics.median(samples["graph_practical"]),
        statistics.median(samples["graph_device"]),
        statistics.median(samples["graph_setup"]),
        statistics.median(samples["graph_capture"]),
    )


def _measure_memory(
    flux: torch.nn.Module,
    context: int,
    replay_tokens: int,
) -> MemoryResult:
    prompt = _input_ids(context, flux.config.vocab_size)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        output = flux(input_ids=prompt, use_cache=True, logits_to_keep=1)
        cache = output.past_key_values
        assert isinstance(cache, DynamicCache)
        torch.cuda.synchronize()
        eager_cache = sum(
            layer.keys.numel() * layer.keys.element_size()
            + layer.values.numel() * layer.values.element_size()
            for layer in cache.layers
        )
        token = output.logits.argmax(dim=-1)
        for _ in range(replay_tokens):
            output = flux(
                input_ids=token,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            token = output.logits.argmax(dim=-1)
        torch.cuda.synchronize()
        eager_peak = torch.cuda.max_memory_allocated() - baseline
    del output, cache, token
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    graph = FluxCUDAGraphDecode.capture(
        flux,
        prompt,
        max_decode_steps=replay_tokens,
    )
    token = graph.prefill_logits.argmax(dim=-1)
    allocated_before = torch.cuda.memory_allocated()
    for _ in range(replay_tokens):
        token = graph.replay(token).argmax(dim=-1)
    torch.cuda.synchronize()
    replay_growth = torch.cuda.memory_allocated() - allocated_before
    result = MemoryResult(
        context,
        eager_cache / MIB,
        eager_peak / MIB,
        graph.cache_bytes / MIB,
        graph.memory.graph_pool_bytes / MIB,
        graph.memory.setup_peak_bytes / MIB,
        replay_growth,
    )
    del graph, token, prompt
    return result


def _profile_operation(path: str, operation: Any) -> ProfileResult:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        start.record()
        operation()
        end.record()
    end.synchronize()
    kernel_ms = sum(
        float(event.self_device_time_total)
        for event in prof.events()
        if event.device_type == DeviceType.CUDA
    ) / 1000.0
    return ProfileResult(path, start.elapsed_time(end), kernel_ms)


def _profile_decode(
    flux: torch.nn.Module,
    context: int,
) -> tuple[ProfileResult, ProfileResult]:
    prompt = _input_ids(context, flux.config.vocab_size)
    with torch.inference_mode():
        eager_output = flux(input_ids=prompt, use_cache=True, logits_to_keep=1)
        eager_cache = eager_output.past_key_values
        eager_token = eager_output.logits.argmax(dim=-1)
        graph = FluxCUDAGraphDecode.capture(flux, prompt, max_decode_steps=1)
        graph_token = graph.prefill_logits.argmax(dim=-1)
        eager_profile = _profile_operation(
            "Flux eager",
            lambda: flux(
                input_ids=eager_token,
                past_key_values=eager_cache,
                use_cache=True,
                logits_to_keep=1,
            ),
        )
        graph.input_ids.copy_(graph_token)
        graph_profile = _profile_operation("Flux graph", graph.graph.replay)
    del prompt, eager_cache, graph
    return eager_profile, graph_profile


def _print_environment(args: argparse.Namespace) -> None:
    print("SmolLM2 fixed-shape CUDA-Graph decode benchmark")
    print(f"  Model: {MODEL_ID}")
    print(f"  Revision: {MODEL_REVISION}")
    print(f"  Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"  PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}")
    print(f"  Transformers: {__import__('transformers').__version__}")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  contexts: {args.contexts}")
    print(
        f"  practical loop: {args.warmup_tokens} warmup + "
        f"{args.timing_tokens} measured tokens, {args.repetitions} repetitions"
    )
    print(f"  correctness continuation: {args.correctness_tokens} tokens")
    print(f"  tolerances: rtol={RTOL}, atol={ATOL}")


def _print_results(
    correctness: list[CorrectnessResult],
    timings: list[TimingResult],
) -> None:
    print("\nCorrectness")
    print(
        f"{'context':>8} {'max logits':>12} {'max K':>10} {'max V':>10} "
        f"{'positions':>10} {'greedy IDs':>11}"
    )
    for result in correctness:
        print(
            f"{result.context:>8} {result.max_logits_error:>12.6g} "
            f"{result.max_key_error:>10.3g} {result.max_value_error:>10.3g} "
            f"{str(result.positions_correct):>10} {str(result.greedy_equal):>11}"
        )

    print("\nDecode latency")
    print(
        f"{'context':>8} {'reference':>11} {'Flux eager':>11} "
        f"{'graph loop':>11} {'pure replay':>12} {'vs Flux':>9} {'vs ref':>9}"
    )
    print(
        f"{'':>8} {'ms/token':>11} {'ms/token':>11} {'ms/token':>11} "
        f"{'ms/token':>12} {'speedup':>9} {'speedup':>9}"
    )
    for result in timings:
        print(
            f"{result.context:>8} {result.reference_ms:>11.3f} "
            f"{result.flux_eager_ms:>11.3f} {result.graph_practical_ms:>11.3f} "
            f"{result.graph_device_ms:>12.3f} "
            f"{result.flux_eager_ms / result.graph_practical_ms:>8.3f}x "
            f"{result.reference_ms / result.graph_practical_ms:>8.3f}x"
        )
    print("\nGraph setup (one-time, excluded from decode timing)")
    print(f"{'context':>8} {'total ms':>12} {'capture ms':>12}")
    for result in timings:
        print(
            f"{result.context:>8} {result.graph_setup_ms:>12.3f} "
            f"{result.graph_capture_ms:>12.3f}"
        )
    print("\nTokens/second (practical loops)")
    print(f"{'context':>8} {'reference':>12} {'Flux eager':>12} {'Flux graph':>12}")
    for result in timings:
        print(
            f"{result.context:>8} {1000/result.reference_ms:>12.2f} "
            f"{1000/result.flux_eager_ms:>12.2f} "
            f"{1000/result.graph_practical_ms:>12.2f}"
        )


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not all(
        (
            native_rmsnorm_is_available(),
            native_residual_rmsnorm_is_available(),
            native_rope_is_available(),
            native_softmax_is_available(),
        )
    ):
        raise RuntimeError("all three built Flux native operators are required")
    _configure_runtime()
    _print_environment(args)
    print("\nLoading reference and Flux model instances...", flush=True)
    reference = load_model("cuda")
    flux = enable_flux_ops(load_model("cuda"))
    print(f"  Flux modules: {flux_operator_counts(flux)}")

    maximum = int(reference.config.max_position_embeddings)
    largest_decode = max(
        args.correctness_tokens - 1,
        args.warmup_tokens + args.timing_tokens,
    )
    contexts = tuple(
        context for context in args.contexts if context + largest_decode <= maximum
    )
    skipped = set(args.contexts) - set(contexts)
    for context in sorted(skipped):
        print(f"SKIP context {context}: decode window exceeds model maximum {maximum}")

    correctness = []
    timings = []
    for context in contexts:
        print(f"\nValidating context {context}...", flush=True)
        correctness.append(
            _validate_context(
                reference, flux, context, args.correctness_tokens
            )
        )
        print(f"Benchmarking context {context}...", flush=True)
        timings.append(
            _benchmark_context(
                reference,
                flux,
                context,
                args.warmup_tokens,
                args.timing_tokens,
                args.repetitions,
            )
        )

    print(f"\nMeasuring memory at context {args.memory_context}...", flush=True)
    memory = _measure_memory(flux, args.memory_context, args.correctness_tokens)
    profiles = None
    if not args.skip_profile:
        print(f"Profiling context {args.profile_context}...", flush=True)
        profiles = _profile_decode(flux, args.profile_context)

    _print_results(correctness, timings)
    print("\nMemory")
    print(f"  eager DynamicCache storage: {memory.eager_cache_mib:.2f} MiB")
    print(f"  eager cached-decode incremental peak: {memory.eager_peak_mib:.2f} MiB")
    print(f"  graph StaticCache storage: {memory.static_cache_mib:.2f} MiB")
    print(f"  graph memory-pool overhead: {memory.graph_pool_mib:.2f} MiB")
    print(f"  graph setup incremental peak: {memory.graph_setup_peak_mib:.2f} MiB")
    print(f"  allocated-memory growth across replays: {memory.replay_growth_bytes} bytes")

    if profiles is not None:
        eager_profile, graph_profile = profiles
        timing = next(
            (item for item in timings if item.context == args.profile_context), None
        )
        print(f"\nProfiler at context {args.profile_context}")
        for result in profiles:
            practical = (
                timing.flux_eager_ms
                if timing is not None and result.path == "Flux eager"
                else timing.graph_practical_ms
                if timing is not None
                else result.event_ms
            )
            if result.path == "Flux graph" and timing is not None:
                remaining = max(
                    0.0, timing.graph_practical_ms - timing.graph_device_ms
                )
                extra = (
                    f"pure replay device={timing.graph_device_ms:.3f} ms, "
                    f"remaining host-loop overhead={remaining:.3f} ms"
                )
            else:
                remaining = max(0.0, practical - result.kernel_ms)
                extra = f"remaining practical overhead={remaining:.3f} ms"
            print(
                f"  {result.path}: practical={practical:.3f} ms/token, "
                f"profiled event={result.event_ms:.3f} ms, "
                f"summed profiled kernels={result.kernel_ms:.3f} ms, {extra}"
            )
        print(
            "  graph replay preserves captured Flux modules: "
            f"{flux_operator_counts(flux)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
