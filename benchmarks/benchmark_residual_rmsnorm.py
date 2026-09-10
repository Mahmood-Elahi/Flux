"""Benchmark fused Flux residual RMSNorm against eager PyTorch on CUDA.

Run from the repository root after building the native extension:

    build/python3119/python.exe benchmarks/benchmark_residual_rmsnorm.py

The primary comparison has identical two-output semantics on both sides:
out-of-place residual addition followed by FP32 RMSNorm. GPU latency is measured
with CUDA events in batches, without synchronizing individual operator calls.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from flux.ops import (
    native_residual_rmsnorm_is_available,
    native_rmsnorm_is_available,
    native_rmsnorm_load_error,
    residual_rmsnorm_native,
    rms_norm_native,
)


DTYPE = torch.float32
EPSILON = 1e-5
DEFAULT_WARMUP = 100
DEFAULT_ITERATIONS = 1000
MAX_TIMING_SAMPLES = 20
RTOL = 1e-5
ATOL = 2e-6

PRIMARY_SHAPES = (
    (1, 1, 576),
    (1, 32, 576),
    (1, 128, 576),
    (1, 512, 576),
    (1, 2048, 576),
    (1, 8192, 576),
    (4, 128, 576),
    (8, 512, 576),
)

BREAKDOWN_SHAPES = (
    (1, 1, 576),
    (1, 512, 576),
    (1, 8192, 576),
)

WIDTH_SHAPES = (
    (1, 512, 575),
    (1, 512, 576),
    (1, 512, 577),
    (1, 512, 1024),
)


@dataclass(frozen=True)
class LatencyStats:
    mean_us: float
    median_us: float
    minimum_us: float


@dataclass(frozen=True)
class ComparisonResult:
    shape: tuple[int, ...]
    pytorch: LatencyStats
    flux: LatencyStats

    @property
    def rows(self) -> int:
        return math.prod(self.shape[:-1])

    @property
    def hidden_size(self) -> int:
        return self.shape[-1]

    @property
    def median_speedup(self) -> float:
        return self.pytorch.median_us / self.flux.median_us

    @property
    def mean_speedup(self) -> float:
        return self.pytorch.mean_us / self.flux.mean_us


@dataclass(frozen=True)
class BreakdownResult:
    shape: tuple[int, ...]
    pytorch_add: LatencyStats
    pytorch_rmsnorm: LatencyStats
    pytorch_add_rmsnorm: LatencyStats
    flux_rmsnorm: LatencyStats
    flux_fused: LatencyStats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"warm-up calls per implementation (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"measured calls per implementation (default: {DEFAULT_ITERATIONS})",
    )
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    return args


def _print_environment(warmup: int, iterations: int) -> None:
    cuda_available = torch.cuda.is_available()
    print("Residual RMSNorm benchmark environment")
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
    print(f"  RMSNorm epsilon: {EPSILON}")
    print(f"  warm-up calls per implementation: {warmup}")
    print(f"  measured calls per implementation: {iterations}")
    print(
        "  timing: CUDA events; at most "
        f"{MAX_TIMING_SAMPLES} batched samples, one final synchronization"
    )


def _batch_sizes(iterations: int) -> list[int]:
    sample_count = min(iterations, MAX_TIMING_SAMPLES)
    calls_per_sample, remainder = divmod(iterations, sample_count)
    return [
        calls_per_sample + (sample_index < remainder)
        for sample_index in range(sample_count)
    ]


def _cuda_latencies(
    operations: dict[str, Callable[[], object]], warmup: int, iterations: int
) -> dict[str, LatencyStats]:
    outputs: dict[str, object] = {}
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
        # Rotate and reverse the path order so clock or scheduler drift affects
        # paired implementations evenly rather than always favoring one side.
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
        total_elapsed_us = sum(
            latency * batch_size
            for latency, (batch_size, _, _) in zip(
                sample_latencies_us, samples, strict=True
            )
        )
        results[name] = LatencyStats(
            mean_us=total_elapsed_us / iterations,
            median_us=statistics.median(sample_latencies_us),
            minimum_us=min(sample_latencies_us),
        )
    # Retain every path's final return value until all GPU work has completed.
    del outputs
    return results


def _pytorch_residual_rmsnorm(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual_out = hidden + residual
    norm_out = functional.rms_norm(
        residual_out,
        (hidden.shape[-1],),
        weight=weight,
        eps=EPSILON,
    )
    return norm_out, residual_out


def _make_inputs(
    shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = torch.randn(shape, device="cuda", dtype=DTYPE)
    residual = torch.randn(shape, device="cuda", dtype=DTYPE)
    weight = torch.randn(shape[-1], device="cuda", dtype=DTYPE)
    return hidden, residual, weight


def _check_correctness(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    expected_norm, expected_residual = _pytorch_residual_rmsnorm(
        hidden, residual, weight
    )
    actual_norm, actual_residual = residual_rmsnorm_native(
        hidden, residual, weight, EPSILON
    )
    torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(actual_norm, expected_norm, rtol=RTOL, atol=ATOL)
    torch.cuda.synchronize()


def _benchmark_comparison(
    shape: tuple[int, ...], warmup: int, iterations: int
) -> ComparisonResult:
    hidden, residual, weight = _make_inputs(shape)
    _check_correctness(hidden, residual, weight)

    latencies = _cuda_latencies(
        {
            "pytorch": lambda: _pytorch_residual_rmsnorm(
                hidden, residual, weight
            ),
            "flux": lambda: residual_rmsnorm_native(
                hidden, residual, weight, EPSILON
            ),
        },
        warmup,
        iterations,
    )
    return ComparisonResult(
        shape=shape, pytorch=latencies["pytorch"], flux=latencies["flux"]
    )


def _benchmark_breakdown(
    shape: tuple[int, ...], warmup: int, iterations: int
) -> BreakdownResult:
    hidden, residual, weight = _make_inputs(shape)
    _check_correctness(hidden, residual, weight)
    precomputed_residual = hidden + residual
    torch.cuda.synchronize()

    operations = {
        "pytorch_add": lambda: hidden + residual,
        "pytorch_rmsnorm": lambda: functional.rms_norm(
            precomputed_residual,
            (shape[-1],),
            weight=weight,
            eps=EPSILON,
        ),
        "pytorch_add_rmsnorm": lambda: _pytorch_residual_rmsnorm(
            hidden, residual, weight
        ),
        "flux_rmsnorm": lambda: rms_norm_native(
            precomputed_residual, weight, EPSILON
        ),
        "flux_fused": lambda: residual_rmsnorm_native(
            hidden, residual, weight, EPSILON
        ),
    }
    latencies = _cuda_latencies(operations, warmup, iterations)
    return BreakdownResult(shape=shape, **latencies)


def _print_comparison_table(
    title: str, results: list[ComparisonResult]
) -> None:
    print(f"\n{title}")
    print("Mean and median CUDA execution latency in microseconds per call")
    print(
        f"{'shape':>17} {'rows':>7} {'width':>7} "
        f"{'PyTorch mean':>13} {'PyTorch med':>12} "
        f"{'Flux mean':>11} {'Flux med':>10} "
        f"{'mean speedup':>13} {'med speedup':>12}"
    )
    for result in results:
        print(
            f"{str(result.shape):>17} {result.rows:>7} {result.hidden_size:>7} "
            f"{result.pytorch.mean_us:>13.3f} "
            f"{result.pytorch.median_us:>12.3f} "
            f"{result.flux.mean_us:>11.3f} "
            f"{result.flux.median_us:>10.3f} "
            f"{result.mean_speedup:>12.3f}x "
            f"{result.median_speedup:>11.3f}x"
        )
    print("Speedup is PyTorch add+RMSNorm / Flux fused; >1 means Flux is faster.")


def _print_breakdown(results: list[BreakdownResult]) -> None:
    print("\nSecondary component breakdown")
    print("Median CUDA execution latency in microseconds per call")
    print(
        f"{'shape':>17} {'PT add':>10} {'PT RMSNorm':>12} "
        f"{'PT add+RMS':>12} {'Flux RMS':>11} {'Flux fused':>11}"
    )
    for result in results:
        print(
            f"{str(result.shape):>17} "
            f"{result.pytorch_add.median_us:>10.3f} "
            f"{result.pytorch_rmsnorm.median_us:>12.3f} "
            f"{result.pytorch_add_rmsnorm.median_us:>12.3f} "
            f"{result.flux_rmsnorm.median_us:>11.3f} "
            f"{result.flux_fused.median_us:>11.3f}"
        )


def main() -> int:
    args = _parse_args()
    _print_environment(args.warmup, args.iterations)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the residual RMSNorm benchmark")
    if not hasattr(functional, "rms_norm"):
        raise RuntimeError("this PyTorch installation does not provide functional.rms_norm")
    load_error = native_rmsnorm_load_error()
    if load_error is not None:
        raise RuntimeError("the built Flux native operator library failed to load") from load_error
    if not native_residual_rmsnorm_is_available():
        raise RuntimeError(
            "Flux native residual RMSNorm is unavailable; build it with "
            "FLUX_BUILD_NATIVE=1 and `python setup.py build_ext --inplace`"
        )
    if not native_rmsnorm_is_available():
        raise RuntimeError("Flux native RMSNorm is required for the component breakdown")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    with torch.inference_mode():
        primary_results = [
            _benchmark_comparison(shape, args.warmup, args.iterations)
            for shape in PRIMARY_SHAPES
        ]
        breakdown_results = [
            _benchmark_breakdown(shape, args.warmup, args.iterations)
            for shape in BREAKDOWN_SHAPES
        ]
        width_results = [
            _benchmark_comparison(shape, args.warmup, args.iterations)
            for shape in WIDTH_SHAPES
        ]

    _print_comparison_table("Primary fused versus unfused comparison", primary_results)
    _print_breakdown(breakdown_results)
    _print_comparison_table("Secondary hidden-width characterization", width_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
