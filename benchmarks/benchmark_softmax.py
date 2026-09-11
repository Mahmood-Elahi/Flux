"""Benchmark Flux CUDA attention softmax against PyTorch CUDA softmax.

Run from the repository root after building the native extension:

    build/python3119/python.exe benchmarks/benchmark_softmax.py

The benchmark uses contiguous FP32 CUDA attention-score tensors and applies
softmax over their final (key-sequence) dimension. GPU latency is measured with
CUDA events in batches, without synchronizing individual operator calls.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass

import torch

from flux.ops import (
    native_softmax_is_available,
    native_softmax_load_error,
    softmax_native,
)


DTYPE = torch.float32
ATTENTION_HEADS = 9
DEFAULT_WARMUP = 100
DEFAULT_ITERATIONS = 1000
MAX_TIMING_SAMPLES = 20
SEED = 0
RTOL = 1e-5
ATOL = 5e-7


def _shape(rows: int, width: int) -> tuple[int, ...]:
    if rows == 1:
        return (1, width)
    if rows % ATTENTION_HEADS != 0:
        raise ValueError(f"row count {rows} is not divisible by {ATTENTION_HEADS}")
    return (1, ATTENTION_HEADS, rows // ATTENTION_HEADS, width)


# This is deliberately not a Cartesian product. Narrow widths include isolated
# and low-row launch regimes, while long contexts use modest query counts.
SHAPES = (
    _shape(1, 1),
    _shape(9, 1),
    _shape(1, 16),
    _shape(9, 16),
    _shape(72, 16),
    _shape(1, 32),
    _shape(9, 32),
    _shape(72, 32),
    _shape(1, 64),
    _shape(9, 64),
    _shape(72, 64),
    _shape(9, 128),
    _shape(72, 128),
    _shape(144, 128),
    _shape(576, 128),
    _shape(1152, 128),
    _shape(9, 256),
    _shape(72, 256),
    _shape(576, 256),
    _shape(9, 512),
    _shape(72, 512),
    _shape(576, 512),
    _shape(9, 1024),
    _shape(72, 1024),
    _shape(576, 1024),
    _shape(9, 2048),
    _shape(72, 2048),
    _shape(144, 2048),
    _shape(9, 4096),
    _shape(72, 4096),
    _shape(9, 8192),
    _shape(72, 8192),
)


@dataclass(frozen=True)
class LatencyStats:
    minimum_us: float
    p25_us: float
    median_us: float
    p75_us: float


@dataclass(frozen=True)
class BenchmarkResult:
    shape: tuple[int, ...]
    flux: LatencyStats
    pytorch: LatencyStats

    @property
    def rows(self) -> int:
        return math.prod(self.shape[:-1])

    @property
    def width(self) -> int:
        return self.shape[-1]

    @property
    def speedup(self) -> float:
        return self.pytorch.median_us / self.flux.median_us


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"warm-up calls per implementation and shape (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=(
            "measured calls per implementation and shape "
            f"(default: {DEFAULT_ITERATIONS})"
        ),
    )
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    return args


def _print_environment(warmup: int, iterations: int) -> None:
    cuda_available = torch.cuda.is_available()
    print("Attention softmax benchmark environment")
    print(f"  GPU name: {torch.cuda.get_device_name() if cuda_available else 'unavailable'}")
    capability = torch.cuda.get_device_capability() if cuda_available else None
    capability_text = f"{capability[0]}.{capability[1]}" if capability else "unavailable"
    print(f"  GPU compute capability: {capability_text}")
    print(f"  Python executable: {sys.executable}")
    print(f"  Python version: {sys.version.split()[0]}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  PyTorch CUDA build: {torch.version.cuda}")
    print(f"  CUDA available: {cuda_available}")
    print(f"  dtype: {DTYPE}")
    print("  operation: softmax over the final dimension")
    print(f"  deterministic input seed: {SEED}")
    print(f"  warm-up calls per implementation and shape: {warmup}")
    print(f"  measured calls per implementation and shape: {iterations}")
    print(
        "  timing: CUDA events; at most "
        f"{MAX_TIMING_SAMPLES} batched samples, one final synchronization per shape"
    )


def _batch_sizes(iterations: int) -> list[int]:
    sample_count = min(iterations, MAX_TIMING_SAMPLES)
    calls_per_sample, remainder = divmod(iterations, sample_count)
    return [
        calls_per_sample + (sample_index < remainder)
        for sample_index in range(sample_count)
    ]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    upper_weight = position - lower_index
    return (
        ordered[lower_index] * (1.0 - upper_weight)
        + ordered[upper_index] * upper_weight
    )


def _cuda_latencies(
    operations: dict[str, Callable[[], torch.Tensor]],
    warmup: int,
    iterations: int,
) -> dict[str, LatencyStats]:
    outputs: dict[str, torch.Tensor] = {}
    for name, operation in operations.items():
        for _ in range(warmup):
            outputs[name] = operation()
    torch.cuda.synchronize()

    batch_sizes = _batch_sizes(iterations)
    names = list(operations)
    recorded_samples: dict[
        str, list[tuple[int, torch.cuda.Event, torch.cuda.Event]]
    ] = {name: [] for name in names}
    final_event = None
    for sample_index, batch_size in enumerate(batch_sizes):
        # Rotate and reverse path order so clock or scheduler drift does not
        # consistently favor the implementation measured first.
        offset = sample_index % len(names)
        sample_order = names[offset:] + names[:offset]
        if sample_index % 2:
            sample_order.reverse()
        for name in sample_order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(batch_size):
                outputs[name] = operations[name]()
            end.record()
            recorded_samples[name].append((batch_size, start, end))
            final_event = end

    assert final_event is not None
    final_event.synchronize()
    results = {}
    for name, samples in recorded_samples.items():
        sample_latencies_us = [
            start.elapsed_time(end) * 1000.0 / batch_size
            for batch_size, start, end in samples
        ]
        results[name] = LatencyStats(
            minimum_us=min(sample_latencies_us),
            p25_us=_percentile(sample_latencies_us, 0.25),
            median_us=statistics.median(sample_latencies_us),
            p75_us=_percentile(sample_latencies_us, 0.75),
        )
    # Keep each path's final result live until all recorded work has completed.
    del outputs
    return results


def _check_correctness(input_tensor: torch.Tensor) -> None:
    flux_output = softmax_native(input_tensor)
    pytorch_output = torch.softmax(input_tensor, dim=-1)
    torch.testing.assert_close(
        flux_output,
        pytorch_output,
        rtol=RTOL,
        atol=ATOL,
    )
    torch.cuda.synchronize()


def _benchmark_shape(
    shape: tuple[int, ...],
    generator: torch.Generator,
    warmup: int,
    iterations: int,
) -> BenchmarkResult:
    input_tensor = torch.randn(
        shape,
        device="cuda",
        dtype=DTYPE,
        generator=generator,
    )
    assert input_tensor.is_contiguous()
    _check_correctness(input_tensor)

    latencies = _cuda_latencies(
        {
            "flux": lambda: softmax_native(input_tensor),
            "pytorch": lambda: torch.softmax(input_tensor, dim=-1),
        },
        warmup,
        iterations,
    )
    return BenchmarkResult(
        shape=shape,
        flux=latencies["flux"],
        pytorch=latencies["pytorch"],
    )


def _geometric_mean_speedup(results: list[BenchmarkResult]) -> float:
    return math.exp(
        math.fsum(math.log(result.speedup) for result in results) / len(results)
    )


def _print_results(results: list[BenchmarkResult]) -> None:
    print("\nCUDA execution latency in microseconds per call")
    print(
        f"{'shape':>22} {'rows':>6} {'width':>6} "
        f"{'Flux med':>10} {'Flux min':>10} {'Flux p25':>10} {'Flux p75':>10} "
        f"{'PT med':>10} {'PT min':>10} {'PT p25':>10} {'PT p75':>10} "
        f"{'speedup':>9}"
    )
    for result in results:
        print(
            f"{str(result.shape):>22} {result.rows:>6} {result.width:>6} "
            f"{result.flux.median_us:>10.3f} "
            f"{result.flux.minimum_us:>10.3f} "
            f"{result.flux.p25_us:>10.3f} "
            f"{result.flux.p75_us:>10.3f} "
            f"{result.pytorch.median_us:>10.3f} "
            f"{result.pytorch.minimum_us:>10.3f} "
            f"{result.pytorch.p25_us:>10.3f} "
            f"{result.pytorch.p75_us:>10.3f} "
            f"{result.speedup:>8.3f}x"
        )

    best = max(results, key=lambda result: result.speedup)
    worst = min(results, key=lambda result: result.speedup)
    print("\nSummary (median latency)")
    print(f"  geometric mean speedup: {_geometric_mean_speedup(results):.3f}x")
    print(
        f"  fastest Flux result: {best.speedup:.3f}x at "
        f"rows={best.rows}, width={best.width}, shape={best.shape}"
    )
    print(
        f"  worst Flux result: {worst.speedup:.3f}x at "
        f"rows={worst.rows}, width={worst.width}, shape={worst.shape}"
    )
    print("  speedup is PyTorch / Flux; >1 means Flux is faster")


def main() -> int:
    args = _parse_args()
    _print_environment(args.warmup, args.iterations)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the attention softmax benchmark")
    load_error = native_softmax_load_error()
    if load_error is not None:
        raise RuntimeError("the built Flux native operator library failed to load") from load_error
    if not native_softmax_is_available():
        raise RuntimeError(
            "Flux native softmax is unavailable; build it with FLUX_BUILD_NATIVE=1 "
            "and `python setup.py build_ext --inplace`"
        )

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    with torch.inference_mode():
        results = [
            _benchmark_shape(shape, generator, args.warmup, args.iterations)
            for shape in SHAPES
        ]
    _print_results(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
