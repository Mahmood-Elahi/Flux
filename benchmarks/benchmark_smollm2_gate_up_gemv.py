"""Validate and benchmark retained fused SmolLM2 gate/up GEMV + SwiGLU."""

from __future__ import annotations

import argparse
import gc
import multiprocessing
import os
import statistics
from collections.abc import Callable
from typing import Any

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile
from transformers import DynamicCache

from flux.model.smollm2 import MODEL_ID, MODEL_REVISION, load_model
from flux.model.smollm2_cuda_graph import FluxCUDAGraphDecode
from flux.model.smollm2_flux import (
    FLUX_CUBLASLT_PROJECTION_CATEGORY,
    FLUX_FUSED_GATE_UP_SWIGLU_CATEGORY,
    FLUX_GQA_DECODE_ATTENTION_CATEGORY,
    FLUX_OPERATOR_CATEGORIES,
    FLUX_PACKED_MLP_CATEGORY,
    FLUX_PACKED_QKV_CATEGORY,
    FLUX_PACKED_QKV_ROPE_CACHE_CATEGORY,
    FLUX_PACKED_SWIGLU_CATEGORY,
    enable_flux_ops,
)
from flux.ops import packed_gate_up_swiglu_native_out, packed_swiglu_native_out


FULL_OPERATORS = FLUX_OPERATOR_CATEGORIES | {
    FLUX_PACKED_MLP_CATEGORY,
    FLUX_PACKED_QKV_CATEGORY,
    FLUX_PACKED_SWIGLU_CATEGORY,
    FLUX_GQA_DECODE_ATTENTION_CATEGORY,
    FLUX_PACKED_QKV_ROPE_CACHE_CATEGORY,
    FLUX_CUBLASLT_PROJECTION_CATEGORY,
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--capacities", default="512,1024,2048,4096,8192")
    parser.add_argument("--correctness-replays", type=int, default=4)
    parser.add_argument("--skip-eager", action="store_true")
    parser.add_argument("--skip-layer", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--independent-memory", action="store_true")
    args = parser.parse_args()
    if args.warmup < 1 or args.repetitions < 3 or args.correctness_replays < 1:
        parser.error("warmup must be positive, repetitions >= 3, correctness-replays positive")
    return args


def _configure() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _ids(length: int, vocab_size: int) -> torch.Tensor:
    return (((torch.arange(length) * 17 + 11) % vocab_size).unsqueeze(0)).cuda()


def _event_median_us(
    operation: Callable[[], object], warmup: int, repetitions: int, batch: int = 1
) -> float:
    for _ in range(warmup):
        operation()
    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(batch):
            operation()
        end.record()
        samples.append((start, end))
    samples[-1][1].synchronize()
    return statistics.median(
        start.elapsed_time(end) * 1000.0 / batch for start, end in samples
    )


def _paired_ms(
    baseline: Callable[[], object],
    fused: Callable[[], object],
    warmup: int,
    repetitions: int,
) -> tuple[float, float]:
    operations = (baseline, fused)
    for _ in range(warmup):
        for operation in operations:
            operation()
    samples: tuple[list[tuple[torch.cuda.Event, torch.cuda.Event]], ...] = ([], [])
    for repetition in range(repetitions):
        order = (0, 1) if repetition % 2 == 0 else (1, 0)
        for index in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            operations[index]()
            end.record()
            samples[index].append((start, end))
    torch.cuda.synchronize()
    return tuple(
        statistics.median(start.elapsed_time(end) for start, end in group)
        for group in samples
    )  # type: ignore[return-value]


def _capture(operation: Callable[[], object]) -> tuple[torch.cuda.CUDAGraph, object]:
    for _ in range(3):
        held = operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.inference_mode():
        held = operation()
    return graph, held


def _set_fused(model: torch.nn.Module, enabled: bool) -> tuple[tuple[str, ...], tuple[bool, ...]]:
    categories = model._flux_operator_categories
    flags = tuple(layer.mlp.fuse_gate_up_swiglu for layer in model.model.layers)
    selected = set(categories)
    if enabled:
        selected.add(FLUX_FUSED_GATE_UP_SWIGLU_CATEGORY)
    else:
        selected.discard(FLUX_FUSED_GATE_UP_SWIGLU_CATEGORY)
    model._flux_operator_categories = tuple(sorted(selected))
    for layer in model.model.layers:
        layer.mlp.fuse_gate_up_swiglu = enabled
    return categories, flags


def _restore_fused(
    model: torch.nn.Module, state: tuple[tuple[str, ...], tuple[bool, ...]]
) -> None:
    model._flux_operator_categories = state[0]
    for layer, flag in zip(model.model.layers, state[1], strict=True):
        layer.mlp.fuse_gate_up_swiglu = flag


def _state(
    model: torch.nn.Module, capacity: int, steps: int, fused: bool
) -> FluxCUDAGraphDecode:
    original = _set_fused(model, fused)
    try:
        return FluxCUDAGraphDecode.capture(
            model,
            _ids(capacity - steps, int(model.config.vocab_size)),
            max_decode_steps=steps,
        )
    finally:
        _restore_fused(model, original)


def _isolated(model: torch.nn.Module, warmup: int, repetitions: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(1234)
    input = torch.randn((1, 1, 576), generator=generator, device="cuda")
    weight = model.model.layers[0].mlp.gate_up_proj.weight
    packed = torch.empty((1, 1, 3072), device="cuda")
    output = torch.empty((1, 1, 1536), device="cuda")

    def projection() -> torch.Tensor:
        return torch.nn.functional.linear(input, weight)

    def boundary() -> torch.Tensor:
        return packed_swiglu_native_out(projection(), output)

    def fused() -> torch.Tensor:
        return packed_gate_up_swiglu_native_out(input, weight, output)

    expected_packed = projection()
    expected = torch.nn.functional.silu(expected_packed[..., :1536]) * expected_packed[..., 1536:]
    packed.copy_(expected_packed)
    output.fill_(float("nan"))
    fused()
    error = (output - expected).abs()
    projection_us = _event_median_us(projection, warmup, repetitions, batch=30)
    boundary_us = _event_median_us(boundary, warmup, repetitions, batch=30)
    fused_us = _event_median_us(fused, warmup, repetitions, batch=30)
    weights = tuple(layer.mlp.gate_up_proj.weight for layer in model.model.layers)
    packed_inputs = tuple(torch.nn.functional.linear(input, item) for item in weights)
    projection_graph = _capture(
        lambda: tuple(torch.nn.functional.linear(input, item) for item in weights)
    )
    swiglu_graph = _capture(
        lambda: tuple(packed_swiglu_native_out(item, output) for item in packed_inputs)
    )
    boundary_graph = _capture(
        lambda: tuple(
            packed_swiglu_native_out(
                torch.nn.functional.linear(input, item), output
            )
            for item in weights
        )
    )
    fused_graph = _capture(
        lambda: tuple(
            packed_gate_up_swiglu_native_out(input, item, output)
            for item in weights
        )
    )
    projection_graph_us = _event_median_us(
        projection_graph[0].replay, warmup, repetitions
    ) / 30.0
    swiglu_graph_us = _event_median_us(
        swiglu_graph[0].replay, warmup, repetitions
    ) / 30.0
    boundary_graph_us = _event_median_us(
        boundary_graph[0].replay, warmup, repetitions
    ) / 30.0
    fused_graph_us = _event_median_us(
        fused_graph[0].replay, warmup, repetitions
    ) / 30.0
    print("\nIsolated layer-0 exact shape (30-call batches; CUDA-event medians)")
    print(f"  current packed projection: {projection_us:.3f} us, one library launch")
    print(f"  current projection + packed SwiGLU: {boundary_us:.3f} us, two launches")
    print(f"  fused GEMV + SwiGLU: {fused_us:.3f} us, one launch, {boundary_us/fused_us:.3f}x")
    print(
        "  graph, 30 distinct layer weights: "
        f"projection={projection_graph_us:.3f} us; SwiGLU={swiglu_graph_us:.3f} us; "
        f"boundary={boundary_graph_us:.3f} us; fused={fused_graph_us:.3f} us; "
        f"speedup={boundary_graph_us/fused_graph_us:.3f}x"
    )
    projection_kernels = _kernel_profile(projection, repeats=3)
    fused_kernels = _kernel_profile(fused, repeats=3)
    print("  current CUDA kernels:")
    for name, (count, milliseconds) in projection_kernels.items():
        print(f"    {name}: {count // 3} launch, {milliseconds / 3 * 1000.0:.3f} us")
    print("  fused CUDA kernels:")
    for name, (count, milliseconds) in fused_kernels.items():
        print(f"    {name}: {count // 3} launch, {milliseconds / 3 * 1000.0:.3f} us")
    print(f"  fused activation error max/mean={error.max().item():.3e}/{error.mean().item():.3e}")


def _correctness(
    baseline: FluxCUDAGraphDecode,
    fused: FluxCUDAGraphDecode,
    replays: int,
) -> tuple[float, float, float, float]:
    maximum = mean = cache_maximum = cache_mean = 0.0
    token = baseline.prefill_logits.argmax(dim=-1)
    addresses = fused.stable_addresses()
    for _ in range(replays):
        if fused.scratch is not None:
            fused.scratch.swiglu_output.fill_(float("nan"))
        expected = baseline.replay(token)
        actual = fused.replay(token)
        difference = (actual - expected).abs()
        maximum = max(maximum, difference.max().item())
        mean = max(mean, difference.mean().item())
        assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))
        token = expected.argmax(dim=-1)
    differences = []
    for expected_layer, actual_layer in zip(
        baseline.cache.layers, fused.cache.layers, strict=True
    ):
        differences.append((actual_layer.keys - expected_layer.keys).abs().flatten())
        differences.append((actual_layer.values - expected_layer.values).abs().flatten())
    cache_difference = torch.cat(differences)
    cache_maximum = cache_difference.max().item()
    cache_mean = cache_difference.mean().item()
    assert addresses == fused.stable_addresses()
    return maximum, mean, cache_maximum, cache_mean


def _kernel_profile(operation: Callable[[], object], repeats: int = 3) -> dict[str, tuple[int, float]]:
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as result:
        for _ in range(repeats):
            operation()
        torch.cuda.synchronize()
    rows: dict[str, tuple[int, float]] = {}
    for event in result.events():
        if event.device_type != DeviceType.CUDA:
            continue
        name = event.name
        count, milliseconds = rows.get(name, (0, 0.0))
        rows[name] = (count + 1, milliseconds + float(event.self_device_time_total) / 1000.0)
    return rows


def _summarize_profile(
    rows: dict[str, tuple[int, float]], repeats: int = 3
) -> tuple[int, int, int, float, float]:
    launches = sum(count for count, _ in rows.values()) // repeats
    custom = sum(count for name, (count, _) in rows.items() if "packed_gate_up_swiglu" in name) // repeats
    gqa = sum(count for name, (count, _) in rows.items() if "gqa_decode_attention" in name) // repeats
    device_ms = sum(milliseconds for _, milliseconds in rows.values()) / repeats
    custom_ms = sum(
        milliseconds
        for name, (_, milliseconds) in rows.items()
        if "packed_gate_up_swiglu" in name
    ) / repeats
    return launches, custom, gqa, device_ms, custom_ms


def _graphs(
    model: torch.nn.Module,
    capacities: tuple[int, ...],
    warmup: int,
    repetitions: int,
    correctness_replays: int,
    profile_enabled: bool,
) -> None:
    print("\nFixed-shape CUDA Graph replay (paired/interleaved)")
    print(" capacity  baseline ms   fused ms  speedup  logits max/mean  cache max/mean  scratch base/fused")
    for capacity in capacities:
        steps = 2 * warmup + 2 * repetitions + correctness_replays + 16
        if steps >= capacity:
            steps = capacity - 1
        baseline = _state(model, capacity, steps, False)
        if capacity < 513:
            baseline_ms = _event_median_us(
                baseline.graph.replay, warmup, repetitions
            ) / 1000.0
            print(
                f" {capacity:8d} {baseline_ms:12.4f} {baseline_ms:10.4f}    1.000x "
                "0.000e+00/0.000e+00 0.000e+00/0.000e+00 0/0 (fallback)"
            )
            del baseline
            gc.collect()
            continue
        fused = _state(model, capacity, steps, True)
        errors = _correctness(baseline, fused, correctness_replays)
        baseline_ms, fused_ms = _paired_ms(
            baseline.graph.replay, fused.graph.replay, warmup, repetitions
        )
        print(
            f" {capacity:8d} {baseline_ms:12.4f} {fused_ms:10.4f} {baseline_ms/fused_ms:8.3f}x "
            f"{errors[0]:.3e}/{errors[1]:.3e} {errors[2]:.3e}/{errors[3]:.3e} "
            f"{baseline.memory.stable_scratch_bytes}/{fused.memory.stable_scratch_bytes}"
        )
        if capacity == 4096 and profile_enabled:
            baseline_profile = _summarize_profile(_kernel_profile(baseline.graph.replay))
            fused_profile = _summarize_profile(_kernel_profile(fused.graph.replay))
            print(
                "  profile launches/custom-GEMV/GQA/device-ms/custom-ms: "
                f"baseline={baseline_profile}; fused={fused_profile}"
            )
        del baseline, fused
        gc.collect()


def _layer_graph(
    model: torch.nn.Module, capacity: int, steps: int, fused: bool
) -> tuple[
    torch.cuda.CUDAGraph,
    FluxCUDAGraphDecode,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
]:
    state = _state(model, capacity, steps, fused)
    layer = model.model.layers[0]
    original = _set_fused(model, fused)
    generator = torch.Generator(device="cuda").manual_seed(1901)
    hidden = torch.randn((1, 1, 576), generator=generator, device="cuda")
    position_embeddings = model.model.rotary_emb(hidden, state.position_ids)
    state.attention_mask.zero_()

    def operation() -> torch.Tensor:
        return layer(
            hidden,
            attention_mask=state.attention_mask,
            position_ids=state.position_ids,
            past_key_values=state.cache,
            use_cache=True,
            cache_position=state.position_ids,
            position_embeddings=position_embeddings,
        )[0]

    try:
        with FluxCUDAGraphDecode._installed_decode_scratch(model, state.scratch):
            for _ in range(3):
                output = operation()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), torch.inference_mode():
                output = operation()
    finally:
        _restore_fused(model, original)
    return graph, state, output, hidden, position_embeddings


def _layer(model: torch.nn.Module, warmup: int, repetitions: int) -> None:
    steps = 2 * warmup + 2 * repetitions + 16
    baseline = _layer_graph(model, 4096, steps, False)
    fused = _layer_graph(model, 4096, steps, True)
    baseline_us, fused_us = _paired_ms(
        baseline[0].replay, fused[0].replay, warmup, repetitions
    )
    baseline_us *= 1000.0
    fused_us *= 1000.0
    baseline[0].replay()
    expected = baseline[2].clone()
    fused[0].replay()
    difference = (fused[2] - expected).abs()
    print("\nFirst decoder layer, capacity 4096")
    print(f"  baseline={baseline_us:.3f} us fused={fused_us:.3f} us speedup={baseline_us/fused_us:.3f}x")
    print(f"  output error max/mean={difference.max().item():.3e}/{difference.mean().item():.3e}")


def _clone_cache(cache: Any, config: Any) -> DynamicCache:
    return DynamicCache(
        [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers],
        config=config,
    )


def _eager_latency(
    model: torch.nn.Module, context: int, warmup: int, repetitions: int
) -> float:
    prompt = _ids(context, int(model.config.vocab_size))
    token = _ids(context + 1, int(model.config.vocab_size))[:, -1:]
    with torch.inference_mode():
        base = model(input_ids=prompt, use_cache=True, logits_to_keep=1).past_key_values
    samples = []
    for index in range(warmup + repetitions):
        cache = _clone_cache(base, model.config)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            output = model(
                input_ids=token,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        end.record()
        if index >= warmup:
            samples.append((start, end))
    samples[-1][1].synchronize()
    del output, base, prompt, token
    return statistics.median(start.elapsed_time(end) for start, end in samples)


def _eager(model: torch.nn.Module, capacities: tuple[int, ...], warmup: int, repetitions: int) -> None:
    print("\nEager one-token decode (fused category intentionally falls back)")
    print(" context  baseline/fused ms  speedup")
    for context in capacities:
        latency = _eager_latency(model, context, warmup, repetitions)
        print(f" {context:7d} {latency:18.4f}   1.000x (identical nn.Linear dispatch)")


def _eager_peak(model: torch.nn.Module, context: int) -> int:
    prompt = _ids(context, int(model.config.vocab_size))
    token = _ids(context + 1, int(model.config.vocab_size))[:, -1:]
    with torch.inference_mode():
        base = model(input_ids=prompt, use_cache=True, logits_to_keep=1).past_key_values
    cache = _clone_cache(base, model.config)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        output = model(
            input_ids=token,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
    torch.cuda.synchronize()
    peak = max(0, torch.cuda.max_memory_allocated() - before)
    del prompt, token, base, cache, output
    return peak


def _memory_worker(capacity: int, fused: bool, queue: Any) -> None:
    _configure()
    model = enable_flux_ops(load_model("cuda"), operators=FULL_OPERATORS)
    state = _state(model, capacity, 8, fused)
    queue.put((state.memory.graph_pool_bytes, state.memory.stable_scratch_bytes, state.memory.setup_peak_bytes))


def _memory(capacity: int) -> None:
    context = multiprocessing.get_context("spawn")
    print("\nIndependent-process CUDA Graph memory (bytes)")
    for label, fused in (("baseline", False), ("fused", True)):
        queue = context.Queue()
        process = context.Process(target=_memory_worker, args=(capacity, fused, queue))
        process.start()
        values = queue.get()
        process.join()
        if process.exitcode:
            raise RuntimeError(f"memory child exited with {process.exitcode}")
        print(f"  {label}: graph_pool={values[0]} scratch={values[1]} setup_peak={values[2]}")


def main() -> int:
    args = _args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    _configure()
    capacities = tuple(int(value) for value in args.capacities.split(","))
    model = enable_flux_ops(load_model("cuda"), operators=FULL_OPERATORS)
    print("SmolLM2 fused packed gate/up GEMV + SwiGLU milestone")
    print(f"  model={MODEL_ID}@{MODEL_REVISION}")
    print(f"  PyTorch={torch.__version__}; CUDA={torch.version.cuda}; GPU={torch.cuda.get_device_name()}")
    print("  FP32; TF32 disabled; deterministic algorithms; CUDA-event medians")
    with torch.inference_mode():
        _isolated(model, args.warmup, args.repetitions)
        if not args.skip_layer:
            _layer(model, args.warmup, args.repetitions)
        _graphs(
            model,
            capacities,
            args.warmup,
            args.repetitions,
            args.correctness_replays,
            not args.skip_profile,
        )
        if not args.skip_eager:
            _eager(model, capacities, args.warmup, args.repetitions)
            peak = _eager_peak(model, 4096)
            print(
                f"  capacity 4096 eager incremental peak: baseline={peak} bytes, "
                f"fused={peak} bytes (identical fallback)"
            )
    if args.independent_memory:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        _memory(4096)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
