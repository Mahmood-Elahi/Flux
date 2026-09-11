"""Benchmark fused FP32 attention score processing against the Flux sequence."""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable

import torch

from flux.ops import (
    attention_score_softmax_native,
    native_attention_score_softmax_is_available,
    native_softmax_is_available,
    softmax_native,
)


HEADS = 9
SCALE = 64**-0.5
DEFAULT_LENGTHS = (128, 256, 512, 1024, 2048, 4096)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument(
        "--lengths",
        type=lambda value: tuple(int(item) for item in value.split(",")),
        default=DEFAULT_LENGTHS,
    )
    return parser.parse_args()


def _current(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return softmax_native(scores * SCALE + mask)


def _fused(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return attention_score_softmax_native(scores, mask, SCALE)


def _event_median(
    operation: Callable[[], torch.Tensor], warmup: int, repetitions: int
) -> float:
    output = None
    for _ in range(warmup):
        output = operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = operation()
        end.record()
        samples.append((start, end))
    samples[-1][1].synchronize()
    del output
    return statistics.median(start.elapsed_time(end) for start, end in samples)


def _capture(operation: Callable[[], torch.Tensor]) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            output = operation()
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = operation()
    return graph, output


def _graph_median(
    operation: Callable[[], torch.Tensor], warmup: int, repetitions: int
) -> float:
    graph, output = _capture(operation)
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        samples.append((start, end))
    samples[-1][1].synchronize()
    del graph, output
    return statistics.median(start.elapsed_time(end) for start, end in samples)


def _inputs(query_length: int, key_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(query_length + key_length)
    scores = torch.randn(
        (1, HEADS, query_length, key_length),
        generator=generator,
        device="cuda",
    )
    query_positions = torch.arange(
        key_length - query_length, key_length, device="cuda"
    ).reshape(1, 1, -1, 1)
    key_positions = torch.arange(key_length, device="cuda").reshape(1, 1, 1, -1)
    mask = torch.where(
        key_positions <= query_positions,
        torch.tensor(0.0, device="cuda"),
        torch.tensor(torch.finfo(torch.float32).min, device="cuda"),
    )
    return scores, mask


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not native_softmax_is_available() or not native_attention_score_softmax_is_available():
        raise RuntimeError("built Flux softmax operators are required")
    print(
        f"GPU={torch.cuda.get_device_name()} dtype=float32 heads={HEADS} "
        f"scale={SCALE} warmup={args.warmup} repetitions={args.repetitions}"
    )
    print(
        f"{'Q':>6} {'K':>6} {'current ms':>12} {'fused ms':>10} {'speedup':>9} "
        f"{'graph current':>14} {'graph fused':>12} {'graph spd':>10} {'max error':>12}"
    )
    shapes = [(length, length) for length in args.lengths]
    if 8192 not in args.lengths:
        shapes.append((8, 8192))
    with torch.inference_mode():
        for query_length, key_length in shapes:
            scores, mask = _inputs(query_length, key_length)
            expected = _current(scores, mask)
            actual = _fused(scores, mask)
            max_error = float((actual - expected).abs().max())
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=5e-7)
            operations = {
                "current": lambda: _current(scores, mask),
                "fused": lambda: _fused(scores, mask),
            }
            direct = {
                name: _event_median(op, args.warmup, args.repetitions)
                for name, op in operations.items()
            }
            graph = {
                name: _graph_median(op, args.warmup, args.repetitions)
                for name, op in operations.items()
            }
            print(
                f"{query_length:>6} {key_length:>6} {direct['current']:>12.4f} "
                f"{direct['fused']:>10.4f} {direct['current']/direct['fused']:>8.3f}x "
                f"{graph['current']:>14.4f} {graph['fused']:>12.4f} "
                f"{graph['current']/graph['fused']:>9.3f}x {max_error:>12.6g}"
            )
            del scores, mask, expected, actual
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
