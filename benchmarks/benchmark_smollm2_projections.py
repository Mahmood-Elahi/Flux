"""Characterize SmolLM2 FP32 one-token projections and bounded cuBLASLt choices."""

from __future__ import annotations

import argparse
import gc
import multiprocessing
import os
import statistics
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile
from transformers import DynamicCache

from flux.model.smollm2 import MODEL_ID, MODEL_REVISION, inspect_config, load_model
from flux.model.smollm2_cuda_graph import FluxCUDAGraphDecode
from flux.model.smollm2_flux import (
    FLUX_GQA_DECODE_ATTENTION_CATEGORY,
    FLUX_CUBLASLT_PROJECTION_CATEGORY,
    FLUX_OPERATOR_CATEGORIES,
    FLUX_PACKED_MLP_CATEGORY,
    FLUX_PACKED_QKV_CATEGORY,
    FLUX_PACKED_QKV_ROPE_CACHE_CATEGORY,
    FLUX_PACKED_SWIGLU_CATEGORY,
    enable_flux_ops,
)
from flux.ops import CublasLtAlgorithm, cublaslt_algorithms, cublaslt_linear_out


FULL_OPERATORS = FLUX_OPERATOR_CATEGORIES | {
    FLUX_PACKED_MLP_CATEGORY,
    FLUX_PACKED_QKV_CATEGORY,
    FLUX_PACKED_SWIGLU_CATEGORY,
    FLUX_GQA_DECODE_ATTENTION_CATEGORY,
    FLUX_PACKED_QKV_ROPE_CACHE_CATEGORY,
}


@dataclass(frozen=True)
class Projection:
    name: str
    input_width: int
    output_width: int
    calls_per_token: int
    weights: tuple[torch.Tensor, ...]
    current_api: str

    @property
    def weight(self) -> torch.Tensor:
        return self.weights[0]


@dataclass(frozen=True)
class Measurement:
    eager_us: float
    graph_us: float
    kernels: tuple[str, ...]

    @property
    def kernel_count(self) -> int:
        return len(self.kernels)


# Indices refer to the bounded 4 MiB heuristic list characterized below. All
# selected experimental candidates require zero workspace. They are deliberately
# local to this benchmark until integrated retention evidence exists.
EXPERIMENTAL_ALGORITHMS = {
    "packed QKV": 5,
    "attention output": 5,
    "packed gate/up": 4,
    "MLP down": 3,
    "LM head": 0,
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--max-algorithms", type=int, default=12)
    parser.add_argument("--workspace-mib", type=int, default=4)
    parser.add_argument("--profile-replays", type=int, default=5)
    parser.add_argument(
        "--nsys-trace-only",
        action="store_true",
        help="emit short NVTX ranges for Nsight Systems tracing",
    )
    parser.add_argument("--integrated", action="store_true")
    parser.add_argument("--integrated-only", action="store_true")
    parser.add_argument("--ablation-capacity", type=int, default=4096)
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--production-only", action="store_true")
    parser.add_argument("--production-capacities", default="512,1024,2048,4096,8192")
    parser.add_argument(
        "--independent-memory",
        action="store_true",
        help="measure each production graph in a fresh process",
    )
    parser.add_argument(
        "--eager-production",
        action="store_true",
        help="benchmark the intentionally unchanged eager fallback",
    )
    parser.add_argument(
        "--layer-capacity",
        type=int,
        default=4096,
        help="capacity for first-decoder-layer CUDA-event timing; zero disables it",
    )
    args = parser.parse_args()
    if args.warmup < 1 or args.repetitions < 3 or args.profile_replays < 1:
        parser.error("warmup must be positive; repetitions >= 3; profile-replays positive")
    if not 1 <= args.max_algorithms <= 64 or args.workspace_mib < 0:
        parser.error("max-algorithms must be in [1,64] and workspace-mib non-negative")
    return args


def _configure() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _inventory(model: torch.nn.Module) -> tuple[Projection, ...]:
    config = model.config
    layers = int(config.num_hidden_layers)
    hidden = int(config.hidden_size)
    intermediate = int(config.intermediate_size)
    head_dim = int(getattr(config, "head_dim", hidden // int(config.num_attention_heads)))
    kv_width = int(config.num_key_value_heads) * head_dim
    layer = model.model.layers[0]
    projections = (
        Projection("packed QKV", hidden, hidden + 2 * kv_width, layers,
                   tuple(item.self_attn.packed_qkv.weight for item in model.model.layers),
                   "nn.Linear/F.linear"),
        Projection("attention output", hidden, hidden, layers,
                   tuple(item.self_attn.o_proj.weight for item in model.model.layers),
                   "nn.Linear/F.linear"),
        Projection("packed gate/up", hidden, 2 * intermediate, layers,
                   tuple(item.mlp.gate_up_proj.weight for item in model.model.layers),
                   "nn.Linear/F.linear"),
        Projection("MLP down", intermediate, hidden, layers,
                   tuple(item.mlp.down_proj.weight for item in model.model.layers),
                   "nn.Linear/F.linear"),
        Projection("LM head", hidden, int(config.vocab_size), 1,
                   (model.lm_head.weight,), "nn.Linear/F.linear"),
    )
    for item in projections:
        assert tuple(item.weight.shape) == (item.output_width, item.input_width)
        assert item.weight.dtype == torch.float32 and item.weight.is_contiguous()
    return projections


def _event_median(operation: Callable[[], object], warmup: int, repetitions: int) -> float:
    for _ in range(warmup):
        operation()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        operation()
        end.record()
    ends[-1].synchronize()
    return statistics.median(
        start.elapsed_time(end) * 1000.0
        for start, end in zip(starts, ends, strict=True)
    )


def _capture(operation: Callable[[], object]) -> tuple[torch.cuda.CUDAGraph, object]:
    for _ in range(3):
        result = operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.inference_mode():
        result = operation()
    return graph, result


def _kernel_names(operation: Callable[[], object], repeats: int) -> tuple[str, ...]:
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as result:
        for _ in range(repeats):
            operation()
        torch.cuda.synchronize()
    names = [
        event.name
        for event in result.events()
        if event.device_type == DeviceType.CUDA
    ]
    if repeats > 1:
        # Repeated graph traces should have the same ordered kernel sequence.
        assert len(names) % repeats == 0
        names = names[: len(names) // repeats]
    return tuple(names)


def _measure(
    operation: Callable[[], object],
    graph_operation: Callable[[], object],
    graph_batch: int,
    warmup: int,
    repetitions: int,
    profile_replays: int,
) -> Measurement:
    eager = _event_median(operation, warmup, repetitions)
    graph, captured = _capture(graph_operation)
    del captured
    graph_us = _event_median(graph.replay, warmup, repetitions) / graph_batch
    batch_kernels = _kernel_names(graph.replay, profile_replays)
    assert len(batch_kernels) % graph_batch == 0
    kernels = batch_kernels[: len(batch_kernels) // graph_batch]
    return Measurement(eager, graph_us, kernels)


def _kernel_label(kernels: tuple[str, ...]) -> str:
    if not kernels:
        return "none"
    compact = []
    for name in kernels:
        lowered = name.lower()
        if "gemv" in lowered:
            label = "GEMV"
        elif "splitk" in lowered or "reduction" in lowered:
            label = "split-K reduction"
        elif "gemm" in lowered or "cublas" in lowered:
            label = "GEMM"
        else:
            label = name
        compact.append(label)
    return " + ".join(compact)


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = (actual - expected).abs()
    return float(difference.max().item()), float(difference.mean().item())


def _print_inventory(model: torch.nn.Module, projections: tuple[Projection, ...]) -> None:
    print("\nProjection inventory (all execute inside fixed-shape decode graph)")
    print(
        f"{'operation':<18} {'input':<13} {'weight':<16} {'output':<14} "
        f"{'M/N/K':<16} {'strides in/wt':<24} {'bias':<5} {'calls':>5} {'output behavior'}"
    )
    for item in projections:
        input_stride = (item.input_width, item.input_width, 1)
        output_shape = (1, 1, item.output_width)
        print(
            f"{item.name:<18} {str((1, 1, item.input_width)):<13} "
            f"{str(tuple(item.weight.shape)):<16} {str(output_shape):<14} "
            f"{str((1, item.output_width, item.input_width)):<16} "
            f"{str(input_stride) + '/' + str(tuple(item.weight.stride())):<24} "
            f"{'no':<5} {item.calls_per_token:>5} allocated eager; graph-private under capture"
        )
    print(f"  config-derived geometry: {inspect_config(model.config)}")
    print("  weights are row-major [N,K]; every logical operation is [1,K] @ weight.T -> [1,N].")


def _nsys_trace(projections: tuple[Projection, ...], workspace_bytes: int) -> None:
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")
    print("\nEmitting Nsight Systems ranges (20 calls per implementation)")
    with torch.inference_mode():
        for item in projections:
            input = torch.randn((1, 1, item.input_width), device="cuda")
            output = torch.empty((1, 1, item.output_width), device="cuda")
            algorithm = cublaslt_algorithms(
                input,
                item.weight,
                max_workspace_bytes=workspace_bytes,
                max_algorithms=16,
            )[EXPERIMENTAL_ALGORITHMS[item.name]]
            for _ in range(3):
                torch.nn.functional.linear(input, item.weight)
                cublaslt_linear_out(
                    input,
                    item.weight,
                    output,
                    workspace,
                    algorithm_index=algorithm.index,
                    max_workspace_bytes=workspace_bytes,
                )
            torch.cuda.nvtx.range_push(f"{item.name}: F.linear")
            for _ in range(20):
                torch.nn.functional.linear(input, item.weight)
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push(f"{item.name}: bounded cuBLASLt")
            for _ in range(20):
                cublaslt_linear_out(
                    input,
                    item.weight,
                    output,
                    workspace,
                    algorithm_index=algorithm.index,
                    max_workspace_bytes=workspace_bytes,
                )
            torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


class _ProjectionPatch:
    def __init__(
        self,
        model: torch.nn.Module,
        projections: tuple[Projection, ...],
        selected: set[str],
        max_workspace_bytes: int,
    ) -> None:
        self.originals: list[tuple[torch.nn.Linear, object]] = []
        self.outputs: list[torch.Tensor] = []
        self.workspace = torch.empty(0, dtype=torch.uint8, device="cuda")
        config = model.config
        modules: dict[str, tuple[torch.nn.Linear, ...]] = {
            "packed QKV": tuple(layer.self_attn.packed_qkv for layer in model.model.layers),
            "attention output": tuple(layer.self_attn.o_proj for layer in model.model.layers),
            "packed gate/up": tuple(layer.mlp.gate_up_proj for layer in model.model.layers),
            "MLP down": tuple(layer.mlp.down_proj for layer in model.model.layers),
            "LM head": (model.lm_head,),
        }
        by_name = {item.name: item for item in projections}
        for name in selected:
            item = by_name[name]
            algorithm_index = EXPERIMENTAL_ALGORITHMS[name]
            # Initialize and validate the exact plan before any graph capture.
            probe_input = torch.zeros((1, 1, item.input_width), device="cuda")
            probe_output = torch.empty((1, 1, item.output_width), device="cuda")
            available = cublaslt_algorithms(
                probe_input,
                item.weight,
                max_workspace_bytes=max_workspace_bytes,
                max_algorithms=16,
            )
            algorithm = next(value for value in available if value.index == algorithm_index)
            assert algorithm.workspace_bytes == 0
            cublaslt_linear_out(
                probe_input,
                item.weight,
                probe_output,
                self.workspace,
                algorithm_index=algorithm_index,
                max_workspace_bytes=max_workspace_bytes,
            )
            for module in modules[name]:
                output = torch.empty((1, 1, item.output_width), device="cuda")
                self.outputs.append(output)
                original = module.forward
                self.originals.append((module, original))

                def forward(
                    this: torch.nn.Linear,
                    input: torch.Tensor,
                    *,
                    _output: torch.Tensor = output,
                    _algorithm_index: int = algorithm_index,
                ) -> torch.Tensor:
                    if (
                        input.shape == (1, 1, this.in_features)
                        and input.dtype == torch.float32
                        and input.device.type == "cuda"
                    ):
                        return cublaslt_linear_out(
                            input,
                            this.weight,
                            _output,
                            self.workspace,
                            algorithm_index=_algorithm_index,
                            max_workspace_bytes=max_workspace_bytes,
                        )
                    return torch.nn.functional.linear(input, this.weight, this.bias)

                module.forward = types.MethodType(forward, module)

    def restore(self) -> None:
        for module, original in self.originals:
            module.forward = original  # type: ignore[method-assign]
        self.originals.clear()


def _ids(length: int, vocab_size: int) -> torch.Tensor:
    return (((torch.arange(length) * 17 + 11) % vocab_size).unsqueeze(0)).cuda()


def _captured_decode(
    model: torch.nn.Module,
    capacity: int,
    steps: int,
    projections: tuple[Projection, ...],
    selected: set[str],
    max_workspace_bytes: int,
) -> FluxCUDAGraphDecode:
    prompt = _ids(capacity - steps, int(model.config.vocab_size))
    with torch.inference_mode():
        patch = _ProjectionPatch(model, projections, selected, max_workspace_bytes)
        try:
            state = FluxCUDAGraphDecode.capture(
                model,
                prompt,
                max_decode_steps=steps,
            )
        finally:
            patch.restore()
    state._flux_projection_benchmark_patch = patch
    return state


def _graph_latency(state: FluxCUDAGraphDecode, warmup: int, repetitions: int) -> float:
    return _event_median(state.graph.replay, warmup, repetitions) / 1000.0


def _integrated(
    model: torch.nn.Module,
    projections: tuple[Projection, ...],
    max_workspace_bytes: int,
    capacity: int,
    warmup: int,
    repetitions: int,
) -> None:
    steps = warmup + repetitions + 2
    candidates = tuple(item.name for item in projections)
    print(f"\nIntegrated category ablation at capacity {capacity}")
    baseline = _captured_decode(
        model, capacity, steps, projections, set(), max_workspace_bytes
    )
    baseline_ms = _graph_latency(baseline, warmup, repetitions)
    print(f"  {'configuration':<22} {'ms/token':>10} {'speedup':>10} {'scratch':>12}")
    print(f"  {'retained PyTorch':<22} {baseline_ms:>10.4f} {'1.000x':>10} {0:>12}")
    results: list[tuple[str, float]] = []
    for name in candidates:
        candidate = _captured_decode(
            model, capacity, steps, projections, {name}, max_workspace_bytes
        )
        candidate_ms = _graph_latency(candidate, warmup, repetitions)
        patch = candidate._flux_projection_benchmark_patch
        scratch = sum(value.numel() * value.element_size() for value in patch.outputs)
        results.append((name, candidate_ms))
        print(
            f"  {name:<22} {candidate_ms:>10.4f} "
            f"{baseline_ms / candidate_ms:>9.3f}x {scratch:>12}"
        )
        del candidate
        gc.collect()
    winners = {name for name, milliseconds in results if milliseconds < baseline_ms * 0.995}
    if winners:
        combined = _captured_decode(
            model, capacity, steps, projections, winners, max_workspace_bytes
        )
        combined_ms = _graph_latency(combined, warmup, repetitions)
        patch = combined._flux_projection_benchmark_patch
        scratch = sum(value.numel() * value.element_size() for value in patch.outputs)
        print(
            f"  {'best isolated combo':<22} {combined_ms:>10.4f} "
            f"{baseline_ms / combined_ms:>9.3f}x {scratch:>12}"
        )
        token = baseline.prefill_logits.argmax(dim=-1)
        maximum = mean = 0.0
        addresses = tuple(value.data_ptr() for value in patch.outputs) + (
            patch.workspace.data_ptr(),
        )
        with torch.inference_mode():
            for _ in range(2):
                expected = baseline.replay(token)
                actual = combined.replay(token)
                difference = (actual - expected).abs()
                maximum = max(maximum, float(difference.max().item()))
                mean = max(mean, float(difference.mean().item()))
                assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))
                token = expected.argmax(dim=-1)
        stable = addresses == tuple(value.data_ptr() for value in patch.outputs) + (
            patch.workspace.data_ptr(),
        )
        print(
            f"  combo correctness: logits max/mean={maximum:.3e}/{mean:.3e}; "
            f"greedy tokens match; addresses stable={stable}; selected={sorted(winners)}"
        )
    else:
        print("  no category crossed the 0.5% integrated retention threshold")


def _production_state(
    model: torch.nn.Module,
    capacity: int,
    steps: int,
    optimized: bool,
) -> FluxCUDAGraphDecode:
    original_categories = model._flux_operator_categories
    attentions = tuple(layer.self_attn for layer in model.model.layers)
    original_flags = tuple(item.use_cublaslt_projection for item in attentions)
    if optimized:
        model._flux_operator_categories = tuple(
            sorted(set(original_categories) | {FLUX_CUBLASLT_PROJECTION_CATEGORY})
        )
        for item in attentions:
            item.use_cublaslt_projection = True
    try:
        return FluxCUDAGraphDecode.capture(
            model,
            _ids(capacity - steps, int(model.config.vocab_size)),
            max_decode_steps=steps,
        )
    finally:
        model._flux_operator_categories = original_categories
        for item, flag in zip(attentions, original_flags, strict=True):
            item.use_cublaslt_projection = flag


def _paired_graph_latency(
    baseline: FluxCUDAGraphDecode,
    optimized: FluxCUDAGraphDecode,
    warmup: int,
    repetitions: int,
) -> tuple[float, float]:
    operations = (baseline.graph.replay, optimized.graph.replay)
    for _ in range(warmup):
        for operation in operations:
            operation()
    samples: tuple[list[tuple[torch.cuda.Event, torch.cuda.Event]], ...] = ([], [])
    for repetition in range(repetitions):
        for index in ((0, 1) if repetition % 2 == 0 else (1, 0)):
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


def _layer_graph(
    model: torch.nn.Module,
    capacity: int,
    steps: int,
    optimized: bool,
) -> tuple[torch.cuda.CUDAGraph, FluxCUDAGraphDecode, torch.Tensor]:
    state = _production_state(model, capacity, steps, optimized)
    layer = model.model.layers[0]
    original_flag = layer.self_attn.use_cublaslt_projection
    layer.self_attn.use_cublaslt_projection = optimized
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
            with torch.inference_mode():
                for _ in range(3):
                    output = operation()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = operation()
    finally:
        layer.self_attn.use_cublaslt_projection = original_flag
    return graph, state, output


def _layer_benchmark(
    model: torch.nn.Module, capacity: int, warmup: int, repetitions: int
) -> None:
    steps = warmup + repetitions + 8
    baseline = _layer_graph(model, capacity, steps, False)
    optimized = _layer_graph(model, capacity, steps, True)
    operations = (baseline[0].replay, optimized[0].replay)
    for _ in range(warmup):
        for operation in operations:
            operation()
    samples: tuple[list[tuple[torch.cuda.Event, torch.cuda.Event]], ...] = ([], [])
    for repetition in range(repetitions):
        for index in ((0, 1) if repetition % 2 == 0 else (1, 0)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            operations[index]()
            end.record()
            samples[index].append((start, end))
    torch.cuda.synchronize()
    medians = tuple(
        statistics.median(start.elapsed_time(end) * 1000.0 for start, end in group)
        for group in samples
    )
    print(
        f"\nFirst decoder layer at capacity {capacity}: "
        f"PyTorch={medians[0]:.3f} us, selected Lt={medians[1]:.3f} us, "
        f"speedup={medians[0] / medians[1]:.3f}x"
    )


def _clone_cache(cache: Any, config: Any) -> DynamicCache:
    return DynamicCache(
        [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers],
        config=config,
    )


def _eager_latency(
    model: torch.nn.Module,
    capacity: int,
    warmup: int,
    repetitions: int,
    optimized: bool,
) -> float:
    prompt = _ids(capacity - 1, int(model.config.vocab_size))
    token = _ids(capacity, int(model.config.vocab_size))[:, -1:]
    with torch.inference_mode():
        base = model(input_ids=prompt, use_cache=True, logits_to_keep=1).past_key_values
    attentions = tuple(layer.self_attn for layer in model.model.layers)
    original_flags = tuple(item.use_cublaslt_projection for item in attentions)
    for item in attentions:
        item.use_cublaslt_projection = optimized
    samples: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    output: object = None
    try:
        with torch.inference_mode():
            for index in range(warmup + repetitions):
                cache = _clone_cache(base, model.config)
                # Cache cloning is setup, not decode work. Synchronization
                # keeps its copies and any allocator bookkeeping outside the
                # timed interval.
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
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
        return statistics.median(start.elapsed_time(end) for start, end in samples)
    finally:
        for item, flag in zip(attentions, original_flags, strict=True):
            item.use_cublaslt_projection = flag
        del output, base, prompt, token


def _eager_benchmark(
    model: torch.nn.Module,
    capacities: tuple[int, ...],
    warmup: int,
    repetitions: int,
) -> None:
    print("\nEager one-token decode (projection scratch intentionally inactive)")
    print(f"  {'context':>8} {'PyTorch':>10} {'selected flag':>14} {'speedup':>9}")
    for capacity in capacities:
        baseline = _eager_latency(model, capacity, warmup, repetitions, False)
        # Production installs the caller-owned projection buffers only during
        # fixed-shape graph capture. Without them the selected flag executes
        # this exact same nn.Linear path, so a second noisy timing batch would
        # not measure a distinct implementation.
        selected = baseline
        print(
            f"  {capacity:>8} {baseline:>10.4f} {selected:>14.4f} "
            f"{baseline / selected:>8.3f}x"
        )


def _production_benchmark(
    model: torch.nn.Module,
    capacities: tuple[int, ...],
    warmup: int,
    repetitions: int,
) -> None:
    print("\nRetained production CUDA Graph path")
    print(
        f"  {'capacity':>8} {'PyTorch':>10} {'selected Lt':>12} {'speedup':>9} "
        f"{'base scratch':>13} {'Lt scratch':>11}"
    )
    for capacity in capacities:
        # Timing, paired correctness replays, and the capacity-4096 profiler
        # all advance the captured static cache. Keep explicit headroom so the
        # diagnostic replays cannot run past the cache boundary.
        steps = warmup + repetitions + 10
        baseline = _production_state(model, capacity, steps, False)
        optimized = _production_state(model, capacity, steps, True)
        baseline_ms, optimized_ms = _paired_graph_latency(
            baseline, optimized, warmup, repetitions
        )
        print(
            f"  {capacity:>8} {baseline_ms:>10.4f} {optimized_ms:>12.4f} "
            f"{baseline_ms / optimized_ms:>8.3f}x "
            f"{baseline.memory.stable_scratch_bytes:>13} "
            f"{optimized.memory.stable_scratch_bytes:>11}"
        )
        with torch.inference_mode():
            token = baseline.prefill_logits.argmax(dim=-1)
            maximum = mean = 0.0
            addresses = optimized.stable_addresses()
            for _ in range(2):
                expected = baseline.replay(token)
                actual = optimized.replay(token)
                difference = (actual - expected).abs()
                maximum = max(maximum, float(difference.max().item()))
                mean = max(mean, float(difference.mean().item()))
                assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))
                token = expected.argmax(dim=-1)
            assert optimized.stable_addresses() == addresses
        print(
            f"           logits errors max/mean={maximum:.3e}/{mean:.3e}; "
            "greedy/stable-address checks passed"
        )
        baseline_kernels = _kernel_names(baseline.graph.replay, 3)
        optimized_kernels = _kernel_names(optimized.graph.replay, 3)

        def counts(kernels: tuple[str, ...]) -> tuple[int, int, int, int]:
            gemms = sum(
                "gemv" in name.lower() or "gemm" in name.lower()
                for name in kernels
            )
            gqa = sum("gqa_decode_attention" in name.lower() for name in kernels)
            return len(kernels), gemms, gqa, len(kernels) - gemms - gqa

        base_counts = counts(baseline_kernels)
        selected_counts = counts(optimized_kernels)
        print(
            "           launches total/GEMV-GEMM/GQA/other="
            f"{'/'.join(map(str, base_counts))}->"
            f"{'/'.join(map(str, selected_counts))}"
        )
        del baseline, optimized
        gc.collect()


def _memory_worker(optimized: bool, capacity: int, queue: Any) -> None:
    _configure()
    model = enable_flux_ops(load_model("cuda"), operators=FULL_OPERATORS)
    state = _production_state(model, capacity, 2, optimized)
    queue.put(
        (
            state.memory.graph_pool_bytes,
            state.memory.stable_scratch_bytes,
            state.memory.setup_peak_bytes,
        )
    )


def _independent_memory(
    optimized: bool, capacity: int
) -> tuple[int, int, int]:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_memory_worker, args=(optimized, capacity, queue)
    )
    process.start()
    result = queue.get()
    process.join()
    if process.exitcode:
        raise RuntimeError(f"memory child exited with {process.exitcode}")
    return result


def _print_independent_memory(capacities: tuple[int, ...]) -> None:
    print("\nIndependent-process CUDA Graph memory")
    print(
        f"  {'capacity':>8} {'path':>10} {'graph pool':>12} "
        f"{'scratch':>10} {'setup peak':>12}"
    )
    for capacity in capacities:
        for label, optimized in (("PyTorch", False), ("selected", True)):
            graph_pool, scratch, setup_peak = _independent_memory(
                optimized, capacity
            )
            print(
                f"  {capacity:>8} {label:>10} {graph_pool:>12} "
                f"{scratch:>10} {setup_peak:>12}"
            )


def main() -> int:
    args = _args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    _configure()
    model = enable_flux_ops(load_model("cuda"), operators=FULL_OPERATORS)
    projections = _inventory(model)
    workspace_bytes = args.workspace_mib * 1024 * 1024
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")

    print("Flux one-token projection characterization")
    print(f"  model={MODEL_ID}@{MODEL_REVISION}")
    print(
        f"  Python={sys.version.split()[0]}; PyTorch={torch.__version__}; "
        f"CUDA={torch.version.cuda}; GPU={torch.cuda.get_device_name()}"
    )
    print("  FP32; TF32 disabled; deterministic algorithms; CUDA-event medians")
    _print_inventory(model, projections)
    if args.nsys_trace_only:
        _nsys_trace(projections, workspace_bytes)
        return 0

    skip_micro = args.integrated_only or args.production_only
    if not skip_micro:
        print("\nPer-shape API and explicit cuBLASLt results (microseconds)")
    for shape_index, item in enumerate(projections if not skip_micro else ()):
        generator = torch.Generator(device="cuda").manual_seed(9100 + shape_index)
        input = torch.randn(
            (1, 1, item.input_width), generator=generator, device="cuda"
        )
        weight = item.weight
        output = torch.empty((1, 1, item.output_width), device="cuda")
        input_2d = input.view(1, item.input_width)
        output_2d = output.view(1, item.output_width)
        weight_t = weight.t()
        graph_weights = item.weights if item.calls_per_token > 1 else item.weights * 10
        graph_batch = len(graph_weights)
        alternatives: tuple[
            tuple[str, Callable[[], object], Callable[[], object]], ...
        ] = (
            ("F.linear", lambda: torch.nn.functional.linear(input, weight),
             lambda: tuple(torch.nn.functional.linear(input, value) for value in graph_weights)),
            ("torch.mm(out=)", lambda: torch.mm(input_2d, weight_t, out=output_2d),
             lambda: tuple(
                 torch.mm(input_2d, value.t(), out=output_2d)
                 for value in graph_weights
             )),
            ("torch.matmul(out=)", lambda: torch.matmul(input_2d, weight_t, out=output_2d),
             lambda: tuple(
                 torch.matmul(input_2d, value.t(), out=output_2d)
                 for value in graph_weights
             )),
            ("torch.addmm(beta=0,out=)", lambda: torch.addmm(
                output_2d, input_2d, weight_t, beta=0, out=output_2d
             ), lambda: tuple(torch.addmm(
                 output_2d, input_2d, value.t(), beta=0, out=output_2d
             ) for value in graph_weights)),
        )
        print(
            f"\n  {item.name}: M/N/K=1/{item.output_width}/{item.input_width}; "
            f"calls/token={item.calls_per_token}"
        )
        print(f"    {'candidate':<28} {'eager':>9} {'graph':>9} {'kernels':>7} kernel family")
        measurements: list[tuple[str, Measurement]] = []
        with torch.inference_mode():
            expected = torch.nn.functional.linear(input, weight)
            for name, operation, graph_operation in alternatives:
                measured = _measure(
                    operation, graph_operation, graph_batch,
                    args.warmup, args.repetitions, args.profile_replays
                )
                measurements.append((name, measured))
                print(
                    f"    {name:<28} {measured.eager_us:>9.3f} {measured.graph_us:>9.3f} "
                    f"{measured.kernel_count:>7} {_kernel_label(measured.kernels)}"
                )

            algorithms = cublaslt_algorithms(
                input,
                weight,
                max_workspace_bytes=workspace_bytes,
                max_algorithms=args.max_algorithms,
            )
            for algorithm in algorithms:
                operation = lambda algorithm=algorithm: cublaslt_linear_out(
                    input,
                    weight,
                    output,
                    workspace,
                    algorithm_index=algorithm.index,
                    max_workspace_bytes=workspace_bytes,
                )
                graph_operation = lambda algorithm=algorithm: tuple(
                    cublaslt_linear_out(
                        input,
                        value,
                        output,
                        workspace,
                        algorithm_index=algorithm.index,
                        max_workspace_bytes=workspace_bytes,
                    )
                    for value in graph_weights
                )
                operation()
                maximum, mean = _errors(output, expected)
                first = output.clone()
                operation()
                deterministic = torch.equal(first, output)
                measured = _measure(
                    operation, graph_operation, graph_batch,
                    args.warmup, args.repetitions, args.profile_replays
                )
                name = f"Lt[{algorithm.index}] id={algorithm.algorithm_id}"
                measurements.append((name, measured))
                print(
                    f"    {name:<28} {measured.eager_us:>9.3f} {measured.graph_us:>9.3f} "
                    f"{measured.kernel_count:>7} {_kernel_label(measured.kernels)}; "
                    f"tile={algorithm.tile_id} stage={algorithm.stages_id} "
                    f"splitK={algorithm.split_k} reduction={algorithm.reduction_scheme} "
                    f"swizzle={algorithm.cta_swizzle} custom={algorithm.custom_option} "
                    f"workspace={algorithm.workspace_bytes} errors={maximum:.3e}/{mean:.3e} "
                    f"repeat={'exact' if deterministic else 'DIFFERS'}"
                )
        baseline = measurements[0][1]
        best_name, best = min(measurements, key=lambda pair: pair[1].graph_us)
        print(
            f"    best graph: {best_name}, {best.graph_us:.3f} us, "
            f"{baseline.graph_us / best.graph_us:.3f}x versus F.linear"
        )
    if args.integrated or args.integrated_only:
        _integrated(
            model,
            projections,
            workspace_bytes,
            args.ablation_capacity,
            args.warmup,
            args.repetitions,
        )
    if args.production or args.production_only:
        capacities = tuple(
            int(value.strip())
            for value in args.production_capacities.split(",")
            if value.strip()
        )
        _production_benchmark(model, capacities, args.warmup, args.repetitions)
        if args.layer_capacity:
            _layer_benchmark(
                model,
                args.layer_capacity,
                args.warmup,
                args.repetitions,
            )
        if args.eager_production:
            _eager_benchmark(model, capacities, args.warmup, args.repetitions)
        if args.independent_memory:
            _print_independent_memory(capacities)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
