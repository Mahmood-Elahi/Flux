"""Benchmark the retained allocation-free long-context GQA CUDA path.

Run from the repository root after rebuilding the native extension:

    build/python3119/python.exe benchmarks/benchmark_gqa_long_context.py

The CUDA-event samples batch several calls so that the two short kernel stages
are not dominated by event resolution.  The profiler pass reports the chunk
and final-reduction kernels separately; event medians remain authoritative.
"""

from __future__ import annotations

import argparse
import os
import statistics
from collections import defaultdict
from collections.abc import Callable

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from flux.ops import gqa_decode_attention, gqa_decode_attention_native_out


HEADS = 9
KV_HEADS = 3
HEAD_DIM = 64
SCALE = HEAD_DIM**-0.5


def _int_list(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result or any(item <= 512 or item > 8192 for item in result):
        raise argparse.ArgumentTypeError("capacities must be in [513, 8192]")
    return result


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacities", type=_int_list, default=(1024, 2048, 4096, 8192))
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--calls-per-sample", type=int, default=100)
    parser.add_argument("--profile-calls", type=int, default=100)
    parser.add_argument("--stabilization-iterations", type=int, default=200)
    args = parser.parse_args()
    if min(args.warmup, args.samples, args.calls_per_sample, args.profile_calls) < 1:
        parser.error("warmup and repetition counts must be positive")
    return args


def _configure() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _event_samples(
    operation: Callable[[], object], warmup: int, samples: int, calls: int
) -> tuple[float, tuple[float, ...]]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(calls):
            operation()
        end.record()
        pairs.append((start, end))
    torch.cuda.synchronize()
    values = tuple(start.elapsed_time(end) * 1000.0 / calls for start, end in pairs)
    return statistics.median(values), values


def _profile_stages(operation: Callable[[], object], calls: int) -> dict[str, tuple[int, float]]:
    for _ in range(10):
        operation()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as result:
        for _ in range(calls):
            operation()
        torch.cuda.synchronize()
    totals: dict[str, list[float]] = defaultdict(list)
    for event in result.events():
        if event.device_type != DeviceType.CUDA:
            continue
        if "gqa_decode_attention_" not in event.name:
            continue
        if "reduce_cuda" in event.name:
            stage = "reduce"
        elif "grouped_chunk_cuda" in event.name:
            stage = "grouped chunk"
        elif "chunk_cuda" in event.name:
            stage = "query-head chunk"
        else:
            continue
        totals[stage].append(float(event.self_device_time_total))
    return {
        stage: (len(values) // calls, sum(values) / calls)
        for stage, values in totals.items()
    }


def main() -> int:
    args = _args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    _configure()
    print("Long-context native GQA benchmark")
    print(f"  PyTorch={torch.__version__}; CUDA={torch.version.cuda}")
    print(f"  GPU={torch.cuda.get_device_name()}; capability={torch.cuda.get_device_capability()}")
    print("  FP32; TF32 disabled; deterministic algorithms; allocation-free out operator")
    print(
        f"  warmup={args.warmup}; samples={args.samples}; "
        f"calls/sample={args.calls_per_sample}; statistic=median"
    )

    tensors: dict[int, tuple[torch.Tensor, ...]] = {}
    for capacity in args.capacities:
        generator = torch.Generator(device="cuda").manual_seed(1000 + capacity)
        query = torch.randn((1, HEADS, 1, HEAD_DIM), generator=generator, device="cuda")
        key = torch.randn(
            (1, KV_HEADS, capacity, HEAD_DIM), generator=generator, device="cuda"
        )
        value = torch.randn_like(key)
        mask = torch.zeros((1, 1, 1, capacity), device="cuda")
        length = torch.tensor(capacity, device="cuda", dtype=torch.int64)
        output = torch.empty_like(query)
        workspace = torch.empty(
            (1, HEADS, (capacity + 127) // 128, HEAD_DIM + 2), device="cuda"
        )
        tensors[capacity] = query, key, value, mask, length, output, workspace

    largest = tensors[max(args.capacities)]
    largest_operation = lambda: gqa_decode_attention_native_out(
        largest[0], largest[1], largest[2], largest[3], SCALE, largest[4], largest[5], largest[6]
    )
    for _ in range(args.stabilization_iterations):
        largest_operation()
    torch.cuda.synchronize()

    print("\n capacity  eager out us  graph out us  chunk prof us  reduce prof us  max abs")
    with torch.inference_mode():
        for capacity in args.capacities:
            query, key, value, mask, length, output, workspace = tensors[capacity]
            operation = lambda: gqa_decode_attention_native_out(
                query, key, value, mask, SCALE, length, output, workspace
            )
            expected = gqa_decode_attention(query, key, value, mask, SCALE, length)
            operation()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, expected, rtol=2e-5, atol=1e-6)
            maximum = float((output - expected).abs().max().item())

            eager_us, _ = _event_samples(
                operation, args.warmup, args.samples, args.calls_per_sample
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                operation()
            graph_us, _ = _event_samples(
                graph.replay, args.warmup, args.samples, args.calls_per_sample
            )
            stages = _profile_stages(operation, args.profile_calls)
            chunk_name = "grouped chunk" if capacity > 4096 else "query-head chunk"
            chunk_us = stages[chunk_name][1]
            reduce_us = stages["reduce"][1]
            print(
                f"{capacity:>9} {eager_us:>13.3f} {graph_us:>13.3f} "
                f"{chunk_us:>14.3f} {reduce_us:>15.3f} {maximum:>8.2g}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
