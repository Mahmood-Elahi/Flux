"""Benchmark Flux CUDA RMSNorm against PyTorch RMSNorm.

Run from the repository root after building the native extension:

    python benchmarks/benchmark_rmsnorm.py

The reported values are median per-call GPU latencies from CUDA events. Measured
calls are divided across at most 20 timing samples so short kernels can be timed
without a synchronization between individual calls.
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
    native_rmsnorm_is_available,
    native_rmsnorm_load_error,
    rms_norm,
    rms_norm_native,
)


DTYPE = torch.float32
EPSILON = 1e-5
HIDDEN_SIZE = 576
DEFAULT_WARMUP = 50
DEFAULT_ITERATIONS = 200
MAX_TIMING_SAMPLES = 20
RTOL = 1e-5
ATOL = 2e-6
SHAPES = (
    (1, 1, HIDDEN_SIZE),
    (1, 8, HIDDEN_SIZE),
    (1, 32, HIDDEN_SIZE),
    (1, 128, HIDDEN_SIZE),
    (1, 512, HIDDEN_SIZE),
    (1, 2048, HIDDEN_SIZE),
    (1, 8192, HIDDEN_SIZE),
    (4, 512, HIDDEN_SIZE),
)


@dataclass(frozen=True)
class BenchmarkResult:
    shape: tuple[int, ...]
    rows: int
    flux_us: float
    pytorch_us: float
    reference_us: float

    @property
    def flux_to_pytorch(self) -> float:
        return self.flux_us / self.pytorch_us


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
    print("RMSNorm benchmark environment")
    print(f"  GPU name: {torch.cuda.get_device_name() if cuda_available else 'unavailable'}")
    capability = torch.cuda.get_device_capability() if cuda_available else None
    capability_text = f"{capability[0]}.{capability[1]}" if capability else "unavailable"
    print(f"  GPU compute capability: {capability_text}")
    print(f"  Python version: {sys.version.split()[0]}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  torch.version.cuda: {torch.version.cuda}")
    print(f"  CUDA available: {cuda_available}")
    print(f"  dtype: {DTYPE}")
    print(f"  RMSNorm epsilon: {EPSILON}")
    print(f"  hidden size: {HIDDEN_SIZE}")
    print(f"  warm-up calls per implementation: {warmup}")
    print(f"  measured calls per implementation: {iterations}")


def _batch_sizes(iterations: int) -> list[int]:
    sample_count = min(iterations, MAX_TIMING_SAMPLES)
    calls_per_sample, remainder = divmod(iterations, sample_count)
    return [
        calls_per_sample + (sample_index < remainder)
        for sample_index in range(sample_count)
    ]


def _median_cuda_latency_us(
    operation: Callable[[], torch.Tensor], warmup: int, iterations: int
) -> float:
    output = None
    for _ in range(warmup):
        output = operation()
    torch.cuda.synchronize()

    batch_sizes = _batch_sizes(iterations)
    event_pairs = [
        (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in batch_sizes
    ]
    for batch_size, (start, end) in zip(batch_sizes, event_pairs, strict=True):
        start.record()
        for _ in range(batch_size):
            output = operation()
        end.record()

    event_pairs[-1][1].synchronize()
    sample_latencies_us = [
        start.elapsed_time(end) * 1000.0 / batch_size
        for batch_size, (start, end) in zip(batch_sizes, event_pairs, strict=True)
    ]
    # Keep the final output alive until all recorded work has completed.
    del output
    return statistics.median(sample_latencies_us)


def _check_correctness(
    input_tensor: torch.Tensor, weight: torch.Tensor
) -> None:
    expected = rms_norm(input_tensor, weight, EPSILON)
    flux_output = rms_norm_native(input_tensor, weight, EPSILON)
    pytorch_output = functional.rms_norm(
        input_tensor, (HIDDEN_SIZE,), weight=weight, eps=EPSILON
    )
    torch.testing.assert_close(flux_output, expected, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(pytorch_output, expected, rtol=RTOL, atol=ATOL)
    torch.cuda.synchronize()


def _benchmark_shape(
    shape: tuple[int, ...], weight: torch.Tensor, warmup: int, iterations: int
) -> BenchmarkResult:
    input_tensor = torch.randn(shape, device="cuda", dtype=DTYPE)
    _check_correctness(input_tensor, weight)

    implementations = {
        "flux": lambda: rms_norm_native(input_tensor, weight, EPSILON),
        "pytorch": lambda: functional.rms_norm(
            input_tensor, (HIDDEN_SIZE,), weight=weight, eps=EPSILON
        ),
        "reference": lambda: rms_norm(input_tensor, weight, EPSILON),
    }
    latencies = {
        name: _median_cuda_latency_us(operation, warmup, iterations)
        for name, operation in implementations.items()
    }
    return BenchmarkResult(
        shape=shape,
        rows=math.prod(shape[:-1]),
        flux_us=latencies["flux"],
        pytorch_us=latencies["pytorch"],
        reference_us=latencies["reference"],
    )


def _print_results(results: list[BenchmarkResult]) -> None:
    print("\nMedian CUDA execution latency (microseconds per call)")
    print(
        f"{'shape':>16} {'rows':>7} {'hidden':>7} {'Flux (us)':>11} "
        f"{'PyTorch (us)':>14} {'reference (us)':>15} {'Flux/PyTorch':>14}"
    )
    for result in results:
        print(
            f"{str(result.shape):>16} {result.rows:>7} {HIDDEN_SIZE:>7} "
            f"{result.flux_us:>11.3f} {result.pytorch_us:>14.3f} "
            f"{result.reference_us:>15.3f} "
            f"{result.flux_to_pytorch:>13.3f}x"
        )
    print("\nFlux/PyTorch is a latency ratio: >1 means Flux is slower; <1 means Flux is faster.")


def main() -> int:
    args = _parse_args()
    _print_environment(args.warmup, args.iterations)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the RMSNorm benchmark")
    if not hasattr(functional, "rms_norm"):
        raise RuntimeError("this PyTorch installation does not provide functional.rms_norm")
    load_error = native_rmsnorm_load_error()
    if load_error is not None:
        raise RuntimeError("the built Flux native operator library failed to load") from load_error
    if not native_rmsnorm_is_available():
        raise RuntimeError(
            "Flux native RMSNorm is unavailable; build it with FLUX_BUILD_NATIVE=1 "
            "and `python setup.py build_ext --inplace`"
        )

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    weight = torch.randn(HIDDEN_SIZE, device="cuda", dtype=DTYPE)
    with torch.inference_mode():
        results = [
            _benchmark_shape(shape, weight, args.warmup, args.iterations)
            for shape in SHAPES
        ]
    _print_results(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
