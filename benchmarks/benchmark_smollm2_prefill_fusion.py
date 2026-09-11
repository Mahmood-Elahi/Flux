"""Benchmark reference, current Flux, and fused Flux SmolLM2 prefill."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from benchmarks.benchmark_smollm2 import (
    ATOL,
    RTOL,
    _configure_runtime,
    _event_latencies,
    _input_ids,
    _profile_operation,
)
from flux.model.smollm2 import load_model
from flux.model.smollm2_flux import enable_flux_ops
from flux.ops import (
    native_attention_score_softmax_is_available,
    native_residual_rmsnorm_is_available,
    native_rmsnorm_is_available,
    native_rope_is_available,
    native_softmax_is_available,
)


DEFAULT_LENGTHS = (128, 256, 512, 1024, 2048, 4096)


@dataclass(frozen=True)
class Result:
    length: int
    reference_ms: float
    current_ms: float
    fused_ms: float
    reference_error: float
    fusion_error: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument(
        "--lengths",
        type=lambda value: tuple(int(item) for item in value.split(",")),
        default=DEFAULT_LENGTHS,
    )
    parser.add_argument("--profile-length", type=int, default=1024)
    return parser.parse_args()


def _errors(
    reference: torch.Tensor,
    current: torch.Tensor,
    fused: torch.Tensor,
) -> tuple[float, float]:
    reference_error = 0.0
    fusion_error = 0.0
    for start in range(0, reference.shape[1], 256):
        stop = min(start + 256, reference.shape[1])
        reference_chunk = reference[:, start:stop]
        current_chunk = current[:, start:stop]
        fused_chunk = fused[:, start:stop]
        reference_error = max(
            reference_error,
            float((current_chunk - reference_chunk).abs().max()),
        )
        fusion_error = max(
            fusion_error,
            float((fused_chunk - current_chunk).abs().max()),
        )
        torch.testing.assert_close(
            fused_chunk,
            current_chunk,
            rtol=0,
            atol=0,
        )
    return reference_error, fusion_error


def _attention_total(components: dict[str, float]) -> float:
    return sum(
        components[name]
        for name in (
            "qkv projections",
            "QK matmul",
            "attention mask/elementwise",
            "softmax",
            "fused score post-processing",
            "P@V matmul",
            "output projection",
            "RoPE (excluding cat)",
            "GQA repeat copies",
        )
    )


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not all(
        (
            native_rmsnorm_is_available(),
            native_residual_rmsnorm_is_available(),
            native_rope_is_available(),
            native_softmax_is_available(),
            native_attention_score_softmax_is_available(),
        )
    ):
        raise RuntimeError("all built Flux operators are required")
    _configure_runtime()
    print(
        f"GPU={torch.cuda.get_device_name()} dtype=float32 TF32=off "
        f"warmup={args.warmup} repetitions={args.repetitions}"
    )
    print("Loading reference, current Flux, and fused Flux models...", flush=True)
    reference = load_model("cuda")
    current = enable_flux_ops(load_model("cuda"), fuse_attention_scores=False)
    fused = enable_flux_ops(load_model("cuda"), fuse_attention_scores=True)

    results = []
    profile_inputs = None
    with torch.inference_mode():
        for length in args.lengths:
            input_ids = _input_ids(length, reference.config.vocab_size)
            outputs = [
                model(input_ids=input_ids, use_cache=True)
                for model in (reference, current, fused)
            ]
            reference_error, fusion_error = _errors(
                outputs[0].logits,
                outputs[1].logits,
                outputs[2].logits,
            )
            for output in outputs:
                if int(output.past_key_values.get_seq_length()) != length:
                    raise AssertionError("prefill KV-cache length mismatch")
            del outputs
            operations = {
                "reference": lambda: reference(input_ids=input_ids, use_cache=True),
                "current Flux": lambda: current(input_ids=input_ids, use_cache=True),
                "fused Flux": lambda: fused(input_ids=input_ids, use_cache=True),
            }
            medians = _event_latencies(operations, args.warmup, args.repetitions)
            results.append(
                Result(
                    length,
                    medians["reference"],
                    medians["current Flux"],
                    medians["fused Flux"],
                    reference_error,
                    fusion_error,
                )
            )
            if length == args.profile_length:
                profile_inputs = input_ids
            else:
                del input_ids

        profiles = []
        if profile_inputs is not None:
            baseline = results[args.lengths.index(args.profile_length)]
            for name, model, milliseconds in (
                ("reference", reference, baseline.reference_ms),
                ("current Flux", current, baseline.current_ms),
                ("fused Flux", fused, baseline.fused_ms),
            ):
                print(f"Profiling {name} prefill {args.profile_length}...", flush=True)
                profiles.append(
                    _profile_operation(
                        name,
                        "prefill",
                        args.profile_length,
                        milliseconds,
                        lambda model=model: model(
                            input_ids=profile_inputs,
                            use_cache=True,
                        ),
                    )
                )

    print("\nPrefill CUDA-event median")
    print(
        f"{'length':>7} {'reference ms':>13} {'current ms':>11} {'fused ms':>10} "
        f"{'vs current':>11} {'ref max err':>12} {'fusion err':>11}"
    )
    for result in results:
        print(
            f"{result.length:>7} {result.reference_ms:>13.3f} "
            f"{result.current_ms:>11.3f} {result.fused_ms:>10.3f} "
            f"{result.current_ms/result.fused_ms:>10.3f}x "
            f"{result.reference_error:>12.6g} {result.fusion_error:>11.6g}"
        )

    if profiles:
        print(f"\nPrefill {args.profile_length} profiler")
        print(
            f"{'path':>13} {'mask/elem ms':>13} {'softmax ms':>11} "
            f"{'fused post ms':>14} {'attention ms':>13} {'model ms':>10}"
        )
        for result in profiles:
            components = result.components_ms
            print(
                f"{result.path:>13} {components['attention mask/elementwise']:>13.3f} "
                f"{components['softmax']:>11.3f} "
                f"{components['fused score post-processing']:>14.3f} "
                f"{_attention_total(components):>13.3f} {result.baseline_ms:>10.3f}"
            )
    print(f"\nReference comparison tolerances remain rtol={RTOL}, atol={ATOL}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
