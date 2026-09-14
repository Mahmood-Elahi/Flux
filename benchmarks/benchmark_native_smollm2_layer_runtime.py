"""Benchmark the native one-layer executor against the retained Python graph.

Run from the repository root after rebuilding the native extension:

    build/python3119/python.exe benchmarks/benchmark_native_smollm2_layer_runtime.py

Model construction, graph capture, deterministic state creation, correctness,
and profiling are outside CUDA-event timing. The synthetic model has exact
SmolLM2-135M layer geometry; both paths share the same weights and initial K/V.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import LlamaConfig, LlamaForCausalLM, StaticCache

from benchmarks.smollm2_benchmark_utils import configure_runtime, parse_positive_int_list
from flux.model import FINAL_FLUX_OPERATOR_CATEGORIES, enable_flux_ops
from flux.model.smollm2_cuda_graph import FluxCUDAGraphDecode
from flux.runtime import NativeSmolLM2LayerDecode


RTOL = 2e-4
ATOL = 2e-5
DEFAULT_LENGTHS = (128, 512, 1024, 2048, 4096, 8192)


@dataclass(frozen=True)
class LayerResult:
    effective_length: int
    python_graph_ms: float
    native_graph_ms: float
    native_speedup: float
    python_host_us: float
    native_host_us: float
    max_output_error: float
    max_key_error: float
    max_value_error: float


@dataclass(frozen=True)
class LayerAudit:
    capacity: int
    replays: int
    python_launches_per_replay: int
    native_launches_per_replay: int
    native_allocation_growth_bytes: int
    native_addresses_stable: bool
    native_cpu_synchronize_events: int
    native_cpu_allocation_events: int


class _PythonLayerGraph:
    """Current Python-owned CUDA-Graph path narrowed to decoder layer zero."""

    def __init__(
        self,
        model: LlamaForCausalLM,
        hidden: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        position: int,
        capacity: int,
    ) -> None:
        self.hidden = hidden.clone()
        self.position = torch.full((1, 1), position, dtype=torch.long, device="cuda")
        self.cache = StaticCache(config=model.config, max_cache_len=capacity)
        cache_layer = self.cache.layers[0]
        seed = torch.zeros((1, 3, 1, 64), device="cuda")
        cache_layer.lazy_initialization(seed, seed)
        if position:
            cache_layer.keys[..., :position, :].copy_(key[..., :position, :])
            cache_layer.values[..., :position, :].copy_(value[..., :position, :])
        cache_layer.cumulative_length.fill_(position)

        mask_min = torch.finfo(torch.float32).min
        self.mask = torch.full(
            (1, 1, 1, capacity), mask_min, dtype=torch.float32, device="cuda"
        )
        self.mask[..., :position].zero_()
        self.zero_column = torch.zeros((1, 1, 1, 1), device="cuda")
        scratch = FluxCUDAGraphDecode._allocate_decode_scratch(model, capacity)
        FluxCUDAGraphDecode._initialize_projection_plans(model, scratch)
        self._installed = ExitStack()
        self._installed.enter_context(
            FluxCUDAGraphDecode._installed_decode_scratch(model, scratch)
        )
        layer = model.model.layers[0]

        def body() -> torch.Tensor:
            self.mask.index_copy_(3, self.position.reshape(-1), self.zero_column)
            cos, sin = model.model.rotary_emb(self.hidden, self.position)
            result = layer(
                self.hidden,
                attention_mask=self.mask,
                past_key_values=self.cache,
                use_cache=True,
                position_embeddings=(cos, sin),
            )
            self.position.add_(1)
            return result

        current = torch.cuda.current_stream()
        side = torch.cuda.Stream()
        side.wait_stream(current)
        with torch.cuda.stream(side), torch.inference_mode():
            body()
        current.wait_stream(side)
        current.synchronize()
        self._restore(position, mask_min)
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self.output = body()
        self._restore(position, mask_min)
        torch.cuda.synchronize()

    def _restore(self, position: int, mask_min: float) -> None:
        layer = self.cache.layers[0]
        self.position.fill_(position)
        self.mask.fill_(mask_min)
        self.mask[..., :position].zero_()
        layer.cumulative_length.fill_(position)
        layer.keys[..., position:, :].zero_()
        layer.values[..., position:, :].zero_()

    def replay(self, hidden: torch.Tensor | None = None) -> torch.Tensor:
        if hidden is not None:
            self.hidden.copy_(hidden)
        self.graph.replay()
        return self.output

    def close(self) -> None:
        torch.cuda.synchronize()
        self._installed.close()


def _model() -> LlamaForCausalLM:
    config = LlamaConfig(
        hidden_size=576,
        intermediate_size=1536,
        num_hidden_layers=1,
        num_attention_heads=9,
        num_key_value_heads=3,
        head_dim=64,
        vocab_size=128,
        max_position_embeddings=8192,
        rms_norm_eps=1e-5,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(2601)
    return enable_flux_ops(
        LlamaForCausalLM(config).float().cuda().eval(),
        operators=FINAL_FLUX_OPERATOR_CATEGORIES,
    )


def _state(position: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(2602 + position)
    hidden = torch.randn((1, 1, 576), generator=generator, device="cuda")
    key = torch.randn((1, 3, position, 64), generator=generator, device="cuda")
    value = torch.randn((1, 3, position, 64), generator=generator, device="cuda")
    return hidden, key, value


def _pair(
    model: LlamaForCausalLM, capacity: int, position: int
) -> tuple[_PythonLayerGraph, NativeSmolLM2LayerDecode, torch.Tensor]:
    hidden, key, value = _state(position)
    python = _PythonLayerGraph(model, hidden, key, value, position, capacity)
    native = NativeSmolLM2LayerDecode.capture(
        model,
        hidden,
        key,
        value,
        cache_position=position,
        cache_capacity=capacity,
    )
    return python, native, hidden


def _event_ms(operation: Callable[[], torch.Tensor]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = operation()
    end.record()
    end.synchronize()
    del output
    return start.elapsed_time(end)


def _benchmark_length(
    model: LlamaForCausalLM, length: int, warmup: int, samples: int
) -> LayerResult:
    required = 1 + warmup + 2 * samples
    if required >= length:
        raise ValueError(f"length {length} cannot hold {required} benchmark replays")
    python, native, hidden = _pair(model, length, length - required)
    expected = python.replay(hidden)
    actual = native.replay(hidden)
    torch.cuda.synchronize()
    output_error = float((actual - expected).abs().max().item())
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    python_cache = python.cache.layers[0]
    current = length - required + 1
    key_error = float(
        (native.key_cache[..., :current, :] - python_cache.keys[..., :current, :])
        .abs()
        .max()
        .item()
    )
    value_error = float(
        (native.value_cache[..., :current, :] - python_cache.values[..., :current, :])
        .abs()
        .max()
        .item()
    )
    torch.testing.assert_close(
        native.key_cache[..., :current, :],
        python_cache.keys[..., :current, :],
        rtol=RTOL,
        atol=ATOL,
    )
    torch.testing.assert_close(
        native.value_cache[..., :current, :],
        python_cache.values[..., :current, :],
        rtol=RTOL,
        atol=ATOL,
    )

    for _ in range(warmup):
        python.replay(hidden)
        native.replay(hidden)
    torch.cuda.synchronize()
    values = {"python": [], "native": []}
    host = {"python": [], "native": []}
    for sample in range(samples):
        paths = (
            (("python", python.replay), ("native", native.replay))
            if sample % 2 == 0
            else (("native", native.replay), ("python", python.replay))
        )
        for name, operation in paths:
            values[name].append(_event_ms(lambda operation=operation: operation(hidden)))
            started = time.perf_counter_ns()
            operation(hidden)
            host[name].append((time.perf_counter_ns() - started) / 1000.0)
    torch.cuda.synchronize()
    python_ms = statistics.median(values["python"])
    native_ms = statistics.median(values["native"])
    result = LayerResult(
        length,
        python_ms,
        native_ms,
        python_ms / native_ms,
        statistics.median(host["python"]),
        statistics.median(host["native"]),
        output_error,
        key_error,
        value_error,
    )
    python.close()
    return result


def _launch_count(operation: Callable[[], torch.Tensor], replays: int) -> tuple[int, list[str]]:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        with record_function("flux_layer_replay_batch"):
            for _ in range(replays):
                operation()
    torch.cuda.synchronize()
    cuda_events = [
        event for event in profiler.events() if event.device_type == DeviceType.CUDA
    ]
    def inside_replay(event: object) -> bool:
        parent = getattr(event, "cpu_parent", None)
        while parent is not None:
            if parent.name == "flux_layer_replay_batch":
                return True
            parent = parent.cpu_parent
        return False

    cpu_names = [
        event.name.lower()
        for event in profiler.events()
        if event.device_type == DeviceType.CPU and inside_replay(event)
    ]
    return round(len(cuda_events) / replays), cpu_names


def _audit(model: LlamaForCausalLM, capacity: int, replays: int) -> LayerAudit:
    if replays >= capacity:
        raise ValueError("audit replays must be smaller than capacity")
    python, unused_native, _ = _pair(model, capacity, capacity - replays)
    del unused_native
    python_launches, _ = _launch_count(python.replay, replays)
    python.close()

    # Allocation measurement and profiling each consume fixed-capacity state.
    hidden, key, value = _state(capacity - replays)
    allocation_runtime = NativeSmolLM2LayerDecode.capture(
        model,
        hidden,
        key,
        value,
        cache_position=capacity - replays,
        cache_capacity=capacity,
    )
    allocated = torch.cuda.memory_allocated()
    for _ in range(replays):
        allocation_runtime.replay()
    torch.cuda.synchronize()
    allocation_growth = torch.cuda.memory_allocated() - allocated
    del allocation_runtime

    native = NativeSmolLM2LayerDecode.capture(
        model,
        hidden,
        key,
        value,
        cache_position=capacity - replays,
        cache_capacity=capacity,
    )
    addresses = native.stable_addresses()
    native_launches, cpu_names = _launch_count(native.replay, replays)
    return LayerAudit(
        capacity,
        replays,
        python_launches,
        native_launches,
        allocation_growth,
        addresses == native.stable_addresses(),
        sum("synchronize" in name for name in cpu_names),
        sum("cudamalloc" in name or "cudafree" in name for name in cpu_names),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=parse_positive_int_list, default=DEFAULT_LENGTHS)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--audit-capacity", type=int, default=4096)
    parser.add_argument("--audit-replays", type=int, default=10)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 1 or args.audit_replays < 1:
        parser.error("warmup may be zero; samples and audit replays must be positive")
    return args


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    configure_runtime(seed=2601, deterministic_fill=False)
    model = _model()
    results = []
    for length in args.lengths:
        print(f"Benchmarking one layer at effective length {length}...", flush=True)
        results.append(_benchmark_length(model, length, args.warmup, args.samples))
    audit = _audit(model, args.audit_capacity, args.audit_replays)

    print("\nOne-layer CUDA Graph replay")
    print(
        f"{'length':>8} {'Python ms':>11} {'native ms':>11} {'speedup':>9} "
        f"{'Python host':>12} {'native host':>12} {'max error':>11}"
    )
    for item in results:
        print(
            f"{item.effective_length:>8} {item.python_graph_ms:>11.4f} "
            f"{item.native_graph_ms:>11.4f} {item.native_speedup:>8.3f}x "
            f"{item.python_host_us:>10.2f} us {item.native_host_us:>10.2f} us "
            f"{item.max_output_error:>11.6g}"
        )
    print("\nRuntime audit")
    for name, value in asdict(audit).items():
        print(f"  {name}: {value}")
    payload = {
        "environment": {
            "python": __import__("sys").version.split()[0],
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "dtype": "torch.float32",
            "warmup": args.warmup,
            "samples": args.samples,
        },
        "results": [asdict(item) for item in results],
        "audit": asdict(audit),
    }
    if args.json_output is not None:
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
