"""Benchmark Flux CUDA RoPE and its integrated SmolLM2 model path.

Run after rebuilding the native extension::

    build/python3119/python.exe benchmarks/benchmark_rope.py

CUDA events time only the operation/model call.  Inputs, model loading, cache
prefill, cache cloning, and correctness checks stay outside timed intervals.
"""

from __future__ import annotations

import argparse
import statistics

import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from benchmarks.benchmark_smollm2 import (
    _benchmark_decode,
    _benchmark_full_or_prefill,
    _configure_runtime,
    _event_latencies,
    _input_ids,
)
from flux.model.smollm2 import load_model
from flux.model.smollm2_cuda_graph import FluxCUDAGraphDecode
from flux.model.smollm2_flux import enable_flux_ops
from flux.ops import native_rope_is_available, rope_native


BASE_OPERATORS = ("rmsnorm", "residual_rmsnorm", "softmax")


def _parse_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(","))
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--prefill-lengths", type=_parse_list, default=(16, 128, 512, 1024, 2048))
    parser.add_argument("--decode-positions", type=_parse_list, default=(128, 1024, 4096))
    parser.add_argument("--skip-model", action="store_true")
    return parser.parse_args()


def _rope_inputs(sequence: int, offset: int) -> tuple[torch.Tensor, ...]:
    values = torch.arange(sequence * 576, device="cuda", dtype=torch.float32)
    hidden = torch.sin(values * 0.013).reshape(1, sequence, 576)
    query = hidden.view(1, sequence, 9, 64).transpose(1, 2)
    key = hidden[..., :192].view(1, sequence, 3, 64).transpose(1, 2)
    positions = torch.arange(offset, offset + sequence, device="cuda").float()
    inv_freq = 1.0 / (
        100000.0 ** (torch.arange(0, 64, 2, device="cuda").float() / 64)
    )
    frequencies = torch.outer(positions, inv_freq)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    return query, key, embedding.cos().unsqueeze(0), embedding.sin().unsqueeze(0)


def _operator_benchmarks(args: argparse.Namespace) -> None:
    print("\nRoPE operator CUDA-event median")
    print(f"{'workload':>10} {'tokens':>7} {'position':>9} {'reference us':>13} {'Flux us':>10} {'speedup':>9} {'max err':>10}")
    workloads = [("decode", 1, position) for position in args.decode_positions]
    workloads += [("prefill", length, 0) for length in args.prefill_lengths]
    for name, sequence, offset in workloads:
        inputs = _rope_inputs(sequence, offset)
        expected = apply_rotary_pos_emb(*inputs)
        actual = rope_native(*inputs)
        error = max(float((actual[i] - expected[i]).abs().max()) for i in (0, 1))
        torch.testing.assert_close(actual[0], expected[0], rtol=1e-6, atol=2e-7)
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-6, atol=2e-7)
        medians = _event_latencies(
            {
                "reference": lambda: apply_rotary_pos_emb(*inputs),
                "Flux": lambda: rope_native(*inputs),
            },
            args.warmup,
            args.repetitions,
        )
        print(
            f"{name:>10} {sequence:>7} {offset:>9} {medians['reference'] * 1000:>13.3f} "
            f"{medians['Flux'] * 1000:>10.3f} {medians['reference'] / medians['Flux']:>8.3f}x {error:>10.3g}"
        )


def _model_benchmarks(args: argparse.Namespace) -> None:
    print("\nLoading baseline Flux and Flux+RoPE models...")
    baseline = enable_flux_ops(load_model("cuda"), operators=BASE_OPERATORS)
    integrated = enable_flux_ops(load_model("cuda"))
    print("\nIntegrated model CUDA-event median (baseline Flux vs Flux+RoPE)")
    print(f"{'workload':>10} {'size':>7} {'baseline ms':>12} {'RoPE ms':>10} {'speedup':>9} {'logit err':>11}")
    for length in args.prefill_lengths:
        input_ids = _input_ids(length, baseline.config.vocab_size)
        result = _benchmark_full_or_prefill(
            baseline, integrated, input_ids, True, args.warmup, args.repetitions
        )
        print(
            f"{'prefill':>10} {length:>7} {result.reference_ms:>12.3f} "
            f"{result.flux_ms:>10.3f} {result.speedup:>8.3f}x {result.max_absolute_error:>11.3g}"
        )
    for context in args.decode_positions:
        context_ids = _input_ids(context, baseline.config.vocab_size)
        next_token = _input_ids(context + 1, baseline.config.vocab_size)[:, -1:]
        result = _benchmark_decode(
            baseline,
            integrated,
            context_ids,
            next_token,
            args.warmup,
            args.repetitions,
        )
        print(
            f"{'decode':>10} {context:>7} {result.reference_ms:>12.3f} "
            f"{result.flux_ms:>10.3f} {result.speedup:>8.3f}x {result.max_absolute_error:>11.3g}"
        )

    graph_context = args.decode_positions[len(args.decode_positions) // 2]
    prompt = _input_ids(graph_context, baseline.config.vocab_size)
    capacity = 1 + args.warmup + args.repetitions
    baseline_graph = FluxCUDAGraphDecode.capture(
        baseline, prompt, max_decode_steps=capacity, warmup_steps=3
    )
    integrated_graph = FluxCUDAGraphDecode.capture(
        integrated, prompt, max_decode_steps=capacity, warmup_steps=3
    )
    torch.testing.assert_close(
        integrated_graph.replay(), baseline_graph.replay(), rtol=0, atol=0
    )
    for _ in range(args.warmup):
        baseline_graph.replay()
        integrated_graph.replay()
    torch.cuda.synchronize()
    samples: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
        "baseline": [],
        "RoPE": [],
    }
    for _ in range(args.repetitions):
        for name, state in (
            ("baseline", baseline_graph),
            ("RoPE", integrated_graph),
        ):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            state.replay()
            end.record()
            samples[name].append((start, end))
    samples["RoPE"][-1][1].synchronize()
    baseline_ms = statistics.median(
        start.elapsed_time(end) for start, end in samples["baseline"]
    )
    rope_ms = statistics.median(
        start.elapsed_time(end) for start, end in samples["RoPE"]
    )
    print(
        f"{'graph':>10} {graph_context:>7} {baseline_ms:>12.3f} "
        f"{rope_ms:>10.3f} {baseline_ms / rope_ms:>8.3f}x {'exact':>11}"
    )


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available() or not native_rope_is_available():
        raise RuntimeError("CUDA and the rebuilt Flux RoPE operator are required")
    _configure_runtime()
    print(
        f"GPU={torch.cuda.get_device_name()} dtype=float32 TF32=off "
        f"warmup={args.warmup} repetitions={args.repetitions}"
    )
    with torch.inference_mode():
        _operator_benchmarks(args)
        if not args.skip_model:
            _model_benchmarks(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
