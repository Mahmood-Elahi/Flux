"""Compare eager and CUDA Graph replay latency for Flux and PyTorch softmax.

Run from the repository root after building the native extension:

    build/python3119/python.exe benchmarks/benchmark_softmax_diagnostic.py

This is a diagnostic companion to ``benchmark_softmax.py``.  Each graph holds
many identical softmax operations so the CPU cost of one graph launch is
amortized.  Reported graph latency is the elapsed CUDA-event time divided by
the number of softmax operations captured in the graph.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from flux.ops import (
    native_softmax_is_available,
    native_softmax_load_error,
    softmax_native,
)


DTYPE = torch.float32
ATTENTION_HEADS = 9
SEED = 0
RTOL = 1e-5
ATOL = 5e-7

DEFAULT_RUNS = 3
DEFAULT_EAGER_WARMUP = 100
DEFAULT_EAGER_ITERATIONS = 1000
DEFAULT_GRAPH_WARMUP = 20
DEFAULT_GRAPH_SAMPLES = 20
DEFAULT_MAX_GRAPH_OPERATIONS = 100
MIN_GRAPH_OPERATIONS = 20
DEFAULT_GRAPH_OUTPUT_BUDGET_MIB = 64
MAX_EAGER_SAMPLES = 20


def _shape(rows: int, width: int) -> tuple[int, ...]:
    if rows == 1:
        return (1, width)
    if rows % ATTENTION_HEADS != 0:
        raise ValueError(f"row count {rows} is not divisible by {ATTENTION_HEADS}")
    return (1, ATTENTION_HEADS, rows // ATTENTION_HEADS, width)


SHAPES = (
    _shape(1, 64),
    _shape(72, 64),
    _shape(9, 128),
    _shape(72, 128),
    _shape(576, 128),
    _shape(72, 256),
    _shape(576, 256),
    _shape(72, 512),
    _shape(576, 512),
    _shape(72, 1024),
    _shape(576, 1024),
    _shape(72, 2048),
    _shape(72, 4096),
    _shape(9, 8192),
    _shape(72, 8192),
)


@dataclass(frozen=True)
class LatencyStats:
    minimum_us: float
    median_us: float
    maximum_us: float


@dataclass(frozen=True)
class CapturedGraph:
    graph: torch.cuda.CUDAGraph
    output: torch.Tensor
    operations: int
    capture_allocated_bytes: int
    replay_growth_bytes: int


@dataclass(frozen=True)
class RunResult:
    shape: tuple[int, ...]
    graph_operations: int
    eager_flux_us: float
    eager_pytorch_us: float
    graph_flux_us: float
    graph_pytorch_us: float
    max_capture_allocated_bytes: int
    max_replay_growth_bytes: int

    @property
    def rows(self) -> int:
        return math.prod(self.shape[:-1])

    @property
    def width(self) -> int:
        return self.shape[-1]

    @property
    def eager_speedup(self) -> float:
        return self.eager_pytorch_us / self.eager_flux_us

    @property
    def graph_speedup(self) -> float:
        return self.graph_pytorch_us / self.graph_flux_us


@dataclass(frozen=True)
class AggregateResult:
    shape: tuple[int, ...]
    graph_operations: int
    eager_flux_us: float
    eager_pytorch_us: float
    graph_flux_us: float
    graph_pytorch_us: float
    graph_speedup_min: float
    graph_speedup_max: float

    @property
    def rows(self) -> int:
        return math.prod(self.shape[:-1])

    @property
    def width(self) -> int:
        return self.shape[-1]

    @property
    def eager_speedup(self) -> float:
        return self.eager_pytorch_us / self.eager_flux_us

    @property
    def graph_speedup(self) -> float:
        return self.graph_pytorch_us / self.graph_flux_us


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        action="append",
        metavar="ROWSxK",
        help=(
            "benchmark only the specified row count and width; may be repeated "
            "(default: all diagnostic shapes)"
        ),
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--eager-warmup", type=int, default=DEFAULT_EAGER_WARMUP)
    parser.add_argument(
        "--eager-iterations", type=int, default=DEFAULT_EAGER_ITERATIONS
    )
    parser.add_argument("--graph-warmup", type=int, default=DEFAULT_GRAPH_WARMUP)
    parser.add_argument("--graph-samples", type=int, default=DEFAULT_GRAPH_SAMPLES)
    parser.add_argument(
        "--max-graph-operations", type=int, default=DEFAULT_MAX_GRAPH_OPERATIONS
    )
    parser.add_argument(
        "--graph-output-budget-mib",
        type=int,
        default=DEFAULT_GRAPH_OUTPUT_BUDGET_MIB,
    )
    args = parser.parse_args()
    positive = (
        "runs",
        "eager_iterations",
        "graph_samples",
        "max_graph_operations",
        "graph_output_budget_mib",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.eager_warmup < 0 or args.graph_warmup < 0:
        parser.error("warm-up counts must be non-negative")
    if args.max_graph_operations < MIN_GRAPH_OPERATIONS:
        parser.error(
            f"--max-graph-operations must be at least {MIN_GRAPH_OPERATIONS}"
        )
    if args.shape:
        selected_shapes = []
        for value in args.shape:
            try:
                rows_text, width_text = value.lower().split("x", maxsplit=1)
                rows, width = int(rows_text), int(width_text)
            except ValueError:
                parser.error(f"invalid --shape {value!r}; expected ROWSxK")
            if rows <= 0 or width <= 0:
                parser.error(
                    f"invalid --shape {value!r}; dimensions must be positive"
                )
            try:
                selected_shapes.append(_shape(rows, width))
            except ValueError as error:
                parser.error(str(error))
        args.shapes = tuple(dict.fromkeys(selected_shapes))
    else:
        args.shapes = SHAPES
    return args


def _batch_sizes(iterations: int) -> list[int]:
    sample_count = min(iterations, MAX_EAGER_SAMPLES)
    calls_per_sample, remainder = divmod(iterations, sample_count)
    return [
        calls_per_sample + (sample_index < remainder)
        for sample_index in range(sample_count)
    ]


def _ordered_names(names: Sequence[str], sample_index: int) -> list[str]:
    ordered = list(names)
    offset = sample_index % len(ordered)
    ordered = ordered[offset:] + ordered[:offset]
    if sample_index % 2:
        ordered.reverse()
    return ordered


def _warm_up_on_side_stream(
    operations: dict[str, Callable[[], torch.Tensor]], calls: int, run_index: int
) -> None:
    current_stream = torch.cuda.current_stream()
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(current_stream)
    outputs: dict[str, torch.Tensor] = {}
    with torch.cuda.stream(warmup_stream):
        names = list(operations)
        for call_index in range(calls):
            for name in _ordered_names(names, call_index + run_index):
                outputs[name] = operations[name]()
    current_stream.wait_stream(warmup_stream)
    torch.cuda.synchronize()
    del outputs


def _event_stats(samples_us: list[float]) -> LatencyStats:
    return LatencyStats(
        minimum_us=min(samples_us),
        median_us=statistics.median(samples_us),
        maximum_us=max(samples_us),
    )


def _time_eager(
    operations: dict[str, Callable[[], torch.Tensor]], iterations: int, run_index: int
) -> dict[str, LatencyStats]:
    batch_sizes = _batch_sizes(iterations)
    names = list(operations)
    outputs: dict[str, torch.Tensor] = {}
    recorded: dict[str, list[tuple[int, torch.cuda.Event, torch.cuda.Event]]] = {
        name: [] for name in names
    }
    final_event = None
    for sample_index, batch_size in enumerate(batch_sizes):
        for name in _ordered_names(names, sample_index + run_index):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(batch_size):
                outputs[name] = operations[name]()
            end.record()
            recorded[name].append((batch_size, start, end))
            final_event = end

    assert final_event is not None
    final_event.synchronize()
    results = {}
    for name, samples in recorded.items():
        results[name] = _event_stats(
            [
                start.elapsed_time(end) * 1000.0 / batch_size
                for batch_size, start, end in samples
            ]
        )
    del outputs
    return results


def _graph_operation_count(
    input_tensor: torch.Tensor, max_operations: int, output_budget_mib: int
) -> int:
    bytes_per_output = input_tensor.numel() * input_tensor.element_size()
    output_budget = output_budget_mib * 1024 * 1024
    budget_operations = output_budget // bytes_per_output
    return max(MIN_GRAPH_OPERATIONS, min(max_operations, budget_operations))


def _capture_graph(
    operation: Callable[[], torch.Tensor], operations: int, replay_probe_count: int = 10
) -> CapturedGraph:
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    graph = torch.cuda.CUDAGraph()
    output = None
    with torch.cuda.graph(graph):
        for _ in range(operations):
            output = operation()
    assert output is not None
    torch.cuda.synchronize()
    allocated_after_capture = torch.cuda.memory_allocated()

    # Replays must use the capture's fixed allocations.  Probe repeated replay
    # and fail if live tensor memory grows, which catches accidental retention.
    for _ in range(replay_probe_count):
        graph.replay()
    torch.cuda.synchronize()
    allocated_before_probe = torch.cuda.memory_allocated()
    for _ in range(replay_probe_count):
        graph.replay()
    torch.cuda.synchronize()
    allocated_after_probe = torch.cuda.memory_allocated()
    replay_growth = allocated_after_probe - allocated_before_probe
    if replay_growth != 0:
        raise RuntimeError(
            "CUDA Graph replay changed live tensor memory by "
            f"{replay_growth} bytes"
        )

    return CapturedGraph(
        graph=graph,
        output=output,
        operations=operations,
        capture_allocated_bytes=allocated_after_capture - allocated_before,
        replay_growth_bytes=replay_growth,
    )


def _time_graphs(
    graphs: dict[str, CapturedGraph], warmup: int, samples: int, run_index: int
) -> dict[str, LatencyStats]:
    names = list(graphs)
    for warmup_index in range(warmup):
        for name in _ordered_names(names, warmup_index + run_index):
            graphs[name].graph.replay()
    torch.cuda.synchronize()

    recorded: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
        name: [] for name in names
    }
    final_event = None
    for sample_index in range(samples):
        for name in _ordered_names(names, sample_index + run_index):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].graph.replay()
            end.record()
            recorded[name].append((start, end))
            final_event = end

    assert final_event is not None
    final_event.synchronize()
    return {
        name: _event_stats(
            [
                start.elapsed_time(end) * 1000.0 / graphs[name].operations
                for start, end in events
            ]
        )
        for name, events in recorded.items()
    }


def _check_correctness(input_tensor: torch.Tensor) -> None:
    flux_output = softmax_native(input_tensor)
    pytorch_output = torch.softmax(input_tensor, dim=-1)
    torch.testing.assert_close(flux_output, pytorch_output, rtol=RTOL, atol=ATOL)
    torch.cuda.synchronize()


def _check_graph_correctness(
    input_tensor: torch.Tensor, graphs: dict[str, CapturedGraph]
) -> None:
    for captured in graphs.values():
        captured.graph.replay()
    torch.cuda.synchronize()
    reference = torch.softmax(input_tensor, dim=-1)
    for name, captured in graphs.items():
        torch.testing.assert_close(
            captured.output,
            reference,
            rtol=RTOL,
            atol=ATOL,
            msg=lambda message: f"{name} CUDA Graph output mismatch: {message}",
        )
    torch.cuda.synchronize()


def _benchmark_shape(
    shape: tuple[int, ...], generator: torch.Generator, args: argparse.Namespace, run_index: int
) -> RunResult:
    input_tensor = torch.randn(
        shape, device="cuda", dtype=DTYPE, generator=generator
    )
    assert input_tensor.is_contiguous()
    _check_correctness(input_tensor)
    operations = {
        "flux": lambda: softmax_native(input_tensor),
        "pytorch": lambda: torch.softmax(input_tensor, dim=-1),
    }
    _warm_up_on_side_stream(operations, args.eager_warmup, run_index)
    eager = _time_eager(operations, args.eager_iterations, run_index)

    graph_operations = _graph_operation_count(
        input_tensor, args.max_graph_operations, args.graph_output_budget_mib
    )
    graphs = {}
    for name in _ordered_names(list(operations), run_index):
        graphs[name] = _capture_graph(operations[name], graph_operations)
    _check_graph_correctness(input_tensor, graphs)
    graph = _time_graphs(graphs, args.graph_warmup, args.graph_samples, run_index)

    return RunResult(
        shape=shape,
        graph_operations=graph_operations,
        eager_flux_us=eager["flux"].median_us,
        eager_pytorch_us=eager["pytorch"].median_us,
        graph_flux_us=graph["flux"].median_us,
        graph_pytorch_us=graph["pytorch"].median_us,
        max_capture_allocated_bytes=max(
            captured.capture_allocated_bytes for captured in graphs.values()
        ),
        max_replay_growth_bytes=max(
            captured.replay_growth_bytes for captured in graphs.values()
        ),
    )


def _aggregate(run_results: list[list[RunResult]]) -> list[AggregateResult]:
    shapes = [result.shape for result in run_results[0]]
    by_shape = {
        shape: [
            next(result for result in results if result.shape == shape)
            for results in run_results
        ]
        for shape in shapes
    }
    aggregated = []
    for shape, results in by_shape.items():
        graph_speedups = [result.graph_speedup for result in results]
        aggregated.append(
            AggregateResult(
                shape=shape,
                graph_operations=results[0].graph_operations,
                eager_flux_us=statistics.median(
                    result.eager_flux_us for result in results
                ),
                eager_pytorch_us=statistics.median(
                    result.eager_pytorch_us for result in results
                ),
                graph_flux_us=statistics.median(
                    result.graph_flux_us for result in results
                ),
                graph_pytorch_us=statistics.median(
                    result.graph_pytorch_us for result in results
                ),
                graph_speedup_min=min(graph_speedups),
                graph_speedup_max=max(graph_speedups),
            )
        )
    return aggregated


def _geometric_mean(values: Sequence[float]) -> float:
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _print_environment(args: argparse.Namespace) -> None:
    print("Softmax eager/CUDA Graph diagnostic environment")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    capability = torch.cuda.get_device_capability()
    print(f"  compute capability: {capability[0]}.{capability[1]}")
    print(f"  Python executable: {sys.executable}")
    print(f"  Python version: {sys.version.split()[0]}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  PyTorch CUDA build: {torch.version.cuda}")
    print(f"  dtype: {DTYPE}")
    print(f"  complete runs: {args.runs}")
    print(f"  eager side-stream warm-up calls/path/shape: {args.eager_warmup}")
    print(f"  eager measured calls/path/shape: {args.eager_iterations}")
    print(f"  graph warm-up replays/path/shape: {args.graph_warmup}")
    print(f"  graph timed replay samples/path/shape: {args.graph_samples}")
    print(
        "  graph operations policy: "
        f"{MIN_GRAPH_OPERATIONS}-{args.max_graph_operations}, capped by "
        f"{args.graph_output_budget_mib} MiB logical output volume"
    )
    print("  timing: CUDA events; speedup is PyTorch / Flux")


def _print_run_result(run_number: int, result: RunResult) -> None:
    print(
        f"run={run_number} rows={result.rows:>3} K={result.width:>4} "
        f"ops={result.graph_operations:>3} "
        f"eager_flux={result.eager_flux_us:>8.3f}us "
        f"eager_pt={result.eager_pytorch_us:>8.3f}us "
        f"eager={result.eager_speedup:>6.3f}x "
        f"graph_flux={result.graph_flux_us:>8.3f}us/op "
        f"graph_pt={result.graph_pytorch_us:>8.3f}us/op "
        f"graph={result.graph_speedup:>6.3f}x"
    )


def _print_summary(results: list[AggregateResult], run_results: list[list[RunResult]]) -> None:
    print("\nMedian across complete runs (microseconds)")
    print(
        f"{'rows':>5} {'K':>5} {'ops':>4} "
        f"{'eager Flux':>11} {'eager PT':>10} {'eager':>8} "
        f"{'graph Flux':>11} {'graph PT':>10} {'graph':>8} "
        f"{'graph range':>15}"
    )
    for result in results:
        print(
            f"{result.rows:>5} {result.width:>5} {result.graph_operations:>4} "
            f"{result.eager_flux_us:>11.3f} {result.eager_pytorch_us:>10.3f} "
            f"{result.eager_speedup:>7.3f}x "
            f"{result.graph_flux_us:>11.3f} {result.graph_pytorch_us:>10.3f} "
            f"{result.graph_speedup:>7.3f}x "
            f"{result.graph_speedup_min:>6.3f}-{result.graph_speedup_max:<6.3f}"
        )

    print("\nSummary")
    print(
        "  eager geometric mean speedup: "
        f"{_geometric_mean([result.eager_speedup for result in results]):.3f}x"
    )
    print(
        "  graph geometric mean speedup: "
        f"{_geometric_mean([result.graph_speedup for result in results]):.3f}x"
    )
    max_capture = max(
        result.max_capture_allocated_bytes
        for complete_run in run_results
        for result in complete_run
    )
    max_replay_growth = max(
        result.max_replay_growth_bytes
        for complete_run in run_results
        for result in complete_run
    )
    print(f"  maximum live allocation increase during one capture: {max_capture} bytes")
    print(f"  maximum live allocation growth during replay probes: {max_replay_growth} bytes")


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    load_error = native_softmax_load_error()
    if load_error is not None:
        raise RuntimeError("the built Flux native operator library failed to load") from load_error
    if not native_softmax_is_available():
        raise RuntimeError(
            "Flux native softmax is unavailable; build it with FLUX_BUILD_NATIVE=1 "
            "and `python setup.py build_ext --inplace`"
        )
    _print_environment(args)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    run_results = []
    with torch.inference_mode():
        for run_index in range(args.runs):
            print(f"\nComplete run {run_index + 1}/{args.runs}")
            shape_order = (
                args.shapes
                if run_index % 2 == 0
                else tuple(reversed(args.shapes))
            )
            results = []
            for shape in shape_order:
                result = _benchmark_shape(shape, generator, args, run_index)
                results.append(result)
                _print_run_result(run_index + 1, result)
            run_results.append(results)

    aggregated = _aggregate(run_results)
    _print_summary(aggregated, run_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
