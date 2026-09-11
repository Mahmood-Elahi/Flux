"""Diagnose long-context numerical differences in Flux SmolLM2 inference.

This is intentionally a correctness diagnostic, not a benchmark. It uses the
same pinned model, deterministic token pattern, FP32 settings, eager attention,
and CUDA runtime configuration as ``benchmarks/benchmark_smollm2.py``.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers.masking_utils import create_causal_mask
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from flux.model.smollm2 import MODEL_ID, MODEL_REVISION, load_model
from flux.model.smollm2_flux import enable_flux_ops
from flux.ops import residual_rmsnorm_native, softmax_native


RTOL = 2e-4
ATOL = 2e-5
OPERATOR_RTOL = 1e-5
OPERATOR_ATOL = 5e-7
SEED = 0
DEFAULT_LENGTHS = (512, 1024, 2048, 4096, 8192)
DEFAULT_LAYER_LENGTHS = (512, 2048, 4096, 8192)
VARIANTS: dict[str, tuple[str, ...]] = {
    "rmsnorm": ("rmsnorm",),
    "residual_rmsnorm": ("residual_rmsnorm",),
    "softmax": ("softmax",),
    "all": ("rmsnorm", "residual_rmsnorm", "softmax"),
}


@dataclass
class DifferenceStats:
    max_absolute_error: float
    mean_absolute_error: float
    max_relative_error: float
    failure_count: int
    element_count: int
    failure_fraction: float
    max_tolerance_excess: float
    worst: list[dict[str, float | int]]


def _parse_lengths(value: str) -> tuple[int, ...]:
    lengths = tuple(int(item) for item in value.split(",") if item.strip())
    if not lengths or any(length < 1 for length in lengths):
        raise argparse.ArgumentTypeError("lengths must be positive integers")
    return lengths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-lengths", type=_parse_lengths, default=DEFAULT_LENGTHS)
    parser.add_argument(
        "--layer-lengths", type=_parse_lengths, default=DEFAULT_LAYER_LENGTHS
    )
    parser.add_argument("--attention-length", type=int, default=8192)
    parser.add_argument(
        "--phases",
        choices=("all", "isolation", "layers", "attention", "masked-softmax"),
        default="all",
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument("--worst-count", type=int, default=5)
    args = parser.parse_args()
    if args.attention_length < 1 or args.worst_count < 1:
        parser.error("--attention-length and --worst-count must be positive")
    return args


def _configure_runtime() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _input_ids(length: int, vocab_size: int) -> torch.Tensor:
    values = (torch.arange(length, dtype=torch.long) * 17 + 11) % vocab_size
    return values.unsqueeze(0).cuda()


def _release(*values: object) -> None:
    del values
    gc.collect()
    torch.cuda.empty_cache()


def _difference_stats(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float = RTOL,
    atol: float = ATOL,
    chunk_rows: int = 256,
    worst_count: int = 5,
) -> DifferenceStats:
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {actual.shape} != {expected.shape}")
    if actual.ndim == 0:
        actual = actual.reshape(1, 1)
        expected = expected.reshape(1, 1)
    elif actual.ndim == 1:
        actual = actual.unsqueeze(0)
        expected = expected.unsqueeze(0)

    row_count = math.prod(actual.shape[:-1])
    width = actual.shape[-1]
    actual_rows = actual.reshape(row_count, width)
    expected_rows = expected.reshape(row_count, width)
    total_absolute = 0.0
    maximum_absolute = 0.0
    maximum_relative = 0.0
    failures = 0
    maximum_excess = 0.0
    worst_candidates: list[tuple[float, int, int, float, float]] = []

    for start in range(0, row_count, chunk_rows):
        stop = min(start + chunk_rows, row_count)
        expected_chunk = expected_rows[start:stop].to(actual.device)
        actual_chunk = actual_rows[start:stop]
        difference = (actual_chunk - expected_chunk).abs()
        allowed = atol + rtol * expected_chunk.abs()
        failures += int(torch.count_nonzero(difference > allowed).item())
        maximum_excess = max(
            maximum_excess, float((difference - allowed).max().item())
        )
        total_absolute += float(difference.sum(dtype=torch.float64).item())
        maximum_absolute = max(maximum_absolute, float(difference.max().item()))
        nonzero = expected_chunk != 0
        if bool(torch.any(nonzero)):
            maximum_relative = max(
                maximum_relative,
                float((difference[nonzero] / expected_chunk[nonzero].abs()).max().item()),
            )
        if bool(torch.any(~nonzero & (difference != 0))):
            maximum_relative = math.inf

        flat = difference.flatten()
        count = min(worst_count, flat.numel())
        values, indices = torch.topk(flat, count)
        for value, index in zip(values.tolist(), indices.tolist(), strict=True):
            local_row, column = divmod(index, width)
            row = start + local_row
            worst_candidates.append(
                (
                    value,
                    row,
                    column,
                    float(actual_chunk[local_row, column].item()),
                    float(expected_chunk[local_row, column].item()),
                )
            )

    worst_candidates.sort(reverse=True)
    worst = [
        {
            "row": row,
            "column": column,
            "actual": actual_value,
            "expected": expected_value,
            "absolute_error": absolute,
        }
        for absolute, row, column, actual_value, expected_value in worst_candidates[
            :worst_count
        ]
    ]
    return DifferenceStats(
        max_absolute_error=maximum_absolute,
        mean_absolute_error=total_absolute / actual.numel(),
        max_relative_error=maximum_relative,
        failure_count=failures,
        element_count=actual.numel(),
        failure_fraction=failures / actual.numel(),
        max_tolerance_excess=maximum_excess,
        worst=worst,
    )


def _print_stats(label: str, stats: DifferenceStats) -> None:
    relative = (
        "inf" if math.isinf(stats.max_relative_error) else f"{stats.max_relative_error:.9g}"
    )
    print(
        f"  {label}: max_abs={stats.max_absolute_error:.9g} "
        f"mean_abs={stats.mean_absolute_error:.9g} max_rel={relative} "
        f"failures={stats.failure_count}/{stats.element_count} "
        f"({stats.failure_fraction:.9g}) excess={stats.max_tolerance_excess:.9g}"
    )


def _position_failures(
    actual: torch.Tensor, expected_cpu: torch.Tensor, chunk_tokens: int = 256
) -> dict[str, Any]:
    counts: list[tuple[int, int]] = []
    for start in range(0, actual.shape[1], chunk_tokens):
        stop = min(start + chunk_tokens, actual.shape[1])
        expected = expected_cpu[:, start:stop].to(actual.device)
        failures = (actual[:, start:stop] - expected).abs() > (
            ATOL + RTOL * expected.abs()
        )
        per_position = failures.sum(dim=(0, 2)).cpu().tolist()
        counts.extend(
            (start + offset, count)
            for offset, count in enumerate(per_position)
            if count
        )
    return {
        "affected_position_count": len(counts),
        "first_position": counts[0][0] if counts else None,
        "last_position": counts[-1][0] if counts else None,
        "top_positions": [
            {"position": position, "failures": count}
            for position, count in sorted(counts, key=lambda item: item[1], reverse=True)[
                :10
            ]
        ],
    }


def _load_variant(operators: Sequence[str] | None) -> torch.nn.Module:
    model = load_model("cuda")
    if operators is not None:
        enable_flux_ops(model, operators=operators)
    return model


def run_isolation(lengths: Sequence[int], worst_count: int) -> dict[str, Any]:
    results: dict[str, Any] = {}
    config_model = load_model("cpu")
    vocab_size = config_model.config.vocab_size
    del config_model

    for length in lengths:
        print(f"\nIsolation length {length}", flush=True)
        input_ids = _input_ids(length, vocab_size)
        reference = _load_variant(None)
        with torch.inference_mode():
            reference_logits = reference(input_ids=input_ids, use_cache=False).logits.cpu()
        del reference
        _release()

        zero = DifferenceStats(
            0.0, 0.0, 0.0, 0, reference_logits.numel(), 0.0, 0.0, []
        )
        length_result: dict[str, Any] = {"reference": asdict(zero)}
        _print_stats("reference", zero)
        for name, operators in VARIANTS.items():
            model = _load_variant(operators)
            with torch.inference_mode():
                logits = model(input_ids=input_ids, use_cache=False).logits
            stats = _difference_stats(
                logits, reference_logits, worst_count=worst_count
            )
            positions = _position_failures(logits, reference_logits)
            argmax_mismatches = int(
                torch.count_nonzero(
                    logits.argmax(dim=-1).cpu()
                    != reference_logits.argmax(dim=-1)
                ).item()
            )
            final_stats = _difference_stats(
                logits[:, -1:], reference_logits[:, -1:], worst_count=worst_count
            )
            length_result[name] = {
                **asdict(stats),
                "positions": positions,
                "argmax_mismatches": argmax_mismatches,
                "final_token": asdict(final_stats),
            }
            _print_stats(name, stats)
            print(
                f"    positions={positions} argmax_mismatches={argmax_mismatches} "
                f"final_max_abs={final_stats.max_absolute_error:.9g}"
            )
            print(f"    worst={stats.worst}")
            del logits, model
            _release()
        results[str(length)] = length_result
        del input_ids, reference_logits
        _release()
    return results


def _model_inputs(model: torch.nn.Module, input_ids: torch.Tensor) -> tuple[Any, ...]:
    hidden = model.model.embed_tokens(input_ids)
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    mask = create_causal_mask(
        config=model.config,
        inputs_embeds=hidden,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )
    position_embeddings = model.model.rotary_emb(hidden, position_ids=position_ids)
    return hidden, position_ids, mask, position_embeddings


def run_layer_localization(
    lengths: Sequence[int], worst_count: int
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for length in lengths:
        print(f"\nLayer localization length {length}", flush=True)
        reference = _load_variant(None)
        flux = _load_variant(VARIANTS["all"])
        input_ids = _input_ids(length, reference.config.vocab_size)
        with torch.inference_mode():
            ref_hidden, position_ids, mask, position_embeddings = _model_inputs(
                reference, input_ids
            )
            flux_hidden = flux.model.embed_tokens(input_ids)
            embedding_stats = _difference_stats(
                flux_hidden, ref_hidden, worst_count=worst_count
            )
            _print_stats("embedding", embedding_stats)
            layer_results = []
            first_nonzero = None
            first_failure = None
            for index, (ref_layer, flux_layer) in enumerate(
                zip(reference.model.layers, flux.model.layers, strict=True)
            ):
                ref_hidden = ref_layer(
                    ref_hidden,
                    attention_mask=mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                )
                flux_hidden = flux_layer(
                    flux_hidden,
                    attention_mask=mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                )
                stats = _difference_stats(
                    flux_hidden, ref_hidden, worst_count=worst_count
                )
                if first_nonzero is None and stats.max_absolute_error != 0:
                    first_nonzero = index
                if first_failure is None and stats.failure_count:
                    first_failure = index
                layer_results.append(asdict(stats))
                _print_stats(f"layer {index:02d}", stats)

            ref_final = reference.model.norm(ref_hidden)
            flux_final = flux.model.norm(flux_hidden)
            final_norm_stats = _difference_stats(
                flux_final, ref_final, worst_count=worst_count
            )
            _print_stats("final norm", final_norm_stats)
            ref_logits = reference.lm_head(ref_final)
            flux_logits = flux.lm_head(flux_final)
            logits_stats = _difference_stats(
                flux_logits, ref_logits, worst_count=worst_count
            )
            _print_stats("final logits", logits_stats)

        results[str(length)] = {
            "embedding": asdict(embedding_stats),
            "layers": layer_results,
            "first_nonzero_layer": first_nonzero,
            "first_tolerance_failure_layer": first_failure,
            "final_norm": asdict(final_norm_stats),
            "final_logits": asdict(logits_stats),
        }
        del reference, flux, input_ids, ref_hidden, flux_hidden, mask
        del position_embeddings, ref_final, flux_final, ref_logits, flux_logits
        _release()
    return results


def _project_attention(
    attention: torch.nn.Module,
    hidden: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    input_shape = hidden.shape[:-1]
    hidden_shape = (*input_shape, -1, attention.head_dim)
    query = attention.q_proj(hidden).view(hidden_shape).transpose(1, 2)
    key = attention.k_proj(hidden).view(hidden_shape).transpose(1, 2)
    value = attention.v_proj(hidden).view(hidden_shape).transpose(1, 2)
    rope_query, rope_key = apply_rotary_pos_emb(
        query, key, position_embeddings[0], position_embeddings[1]
    )
    expanded_key = repeat_kv(rope_key, attention.num_key_value_groups)
    expanded_value = repeat_kv(value, attention.num_key_value_groups)
    return query, key, value, rope_query, rope_key, expanded_key, expanded_value


def run_attention_localization(length: int, worst_count: int) -> dict[str, Any]:
    print(f"\nAttention localization length {length}, layer 0", flush=True)
    reference = _load_variant(None)
    flux = _load_variant(VARIANTS["all"])
    input_ids = _input_ids(length, reference.config.vocab_size)
    query_positions = sorted(
        {
            0,
            min(1, length - 1),
            min(7, length - 1),
            min(5075, length - 1),
            min(5792, length - 1),
            min(7753, length - 1),
            length // 2,
            max(0, length - 2),
            length - 1,
        }
    )
    with torch.inference_mode():
        ref_hidden, _, mask, position_embeddings = _model_inputs(reference, input_ids)
        flux_hidden, _, flux_mask, flux_position_embeddings = _model_inputs(
            flux, input_ids
        )
        ref_attention = reference.model.layers[0].self_attn
        flux_attention = flux.model.layers[0].self_attn
        ref_normalized = reference.model.layers[0].input_layernorm(ref_hidden)
        flux_normalized = flux.model.layers[0].input_layernorm(flux_hidden)
        input_norm_stats = _difference_stats(
            flux_normalized, ref_normalized, worst_count=worst_count
        )
        _print_stats("input RMSNorm", input_norm_stats)
        reference_parts = _project_attention(
            ref_attention, ref_normalized, position_embeddings
        )
        flux_parts = _project_attention(
            flux_attention, flux_normalized, position_embeddings
        )
        names = ("q_projection", "k_projection", "v_projection", "post_rope_q", "post_rope_k", "expanded_k", "expanded_v")
        component_results: dict[str, Any] = {}
        for name, actual, expected in zip(
            names, flux_parts, reference_parts, strict=True
        ):
            stats = _difference_stats(actual, expected, worst_count=worst_count)
            component_results[name] = asdict(stats)
            _print_stats(name, stats)

        ref_q, _, _, _, _, ref_k, ref_v = reference_parts
        flux_q, _, _, _, _, flux_k, flux_v = flux_parts
        # Q/K projection tensors above are pre-RoPE. Use the post-RoPE query.
        ref_q = reference_parts[3]
        flux_q = flux_parts[3]
        selected = torch.tensor(query_positions, device="cuda")
        ref_scores = torch.matmul(
            ref_q.index_select(2, selected), ref_k.transpose(2, 3)
        ) * ref_attention.scaling
        flux_scores = torch.matmul(
            flux_q.index_select(2, selected), flux_k.transpose(2, 3)
        ) * flux_attention.scaling
        pre_mask_stats = _difference_stats(
            flux_scores, ref_scores, worst_count=worst_count
        )
        selected_mask = mask.index_select(2, selected)
        ref_masked = ref_scores + selected_mask
        flux_masked = flux_scores + selected_mask
        post_mask_stats = _difference_stats(
            flux_masked, ref_masked, worst_count=worst_count
        )
        ref_probs = torch.softmax(ref_masked, dim=-1, dtype=torch.float32)
        flux_probs = softmax_native(flux_masked)
        torch_probs_for_flux_scores = torch.softmax(
            flux_masked, dim=-1, dtype=torch.float32
        )
        probability_stats = _difference_stats(
            flux_probs,
            ref_probs,
            rtol=OPERATOR_RTOL,
            atol=OPERATOR_ATOL,
            worst_count=worst_count,
        )
        softmax_operator_stats = _difference_stats(
            flux_probs,
            torch_probs_for_flux_scores,
            rtol=OPERATOR_RTOL,
            atol=OPERATOR_ATOL,
            worst_count=worst_count,
        )
        ref_context = torch.matmul(ref_probs, ref_v)
        flux_context = torch.matmul(flux_probs, flux_v)
        context_stats = _difference_stats(
            flux_context, ref_context, worst_count=worst_count
        )
        ref_merged = ref_context.transpose(1, 2).contiguous().reshape(1, len(query_positions), -1)
        flux_merged = flux_context.transpose(1, 2).contiguous().reshape(1, len(query_positions), -1)
        ref_output = ref_attention.o_proj(ref_merged)
        flux_output = flux_attention.o_proj(flux_merged)
        output_stats = _difference_stats(
            flux_output, ref_output, worst_count=worst_count
        )
        _print_stats("scores before mask", pre_mask_stats)
        _print_stats("scores after mask", post_mask_stats)
        _print_stats("softmax probabilities", probability_stats)
        _print_stats("softmax native vs torch on identical scores", softmax_operator_stats)
        _print_stats("P @ V", context_stats)
        _print_stats("attention output projection", output_stats)

        reference_residual = ref_hidden.index_select(1, selected) + ref_output
        reference_post_norm = reference.model.layers[0].post_attention_layernorm(
            reference_residual
        )
        fused_post_norm, fused_residual = residual_rmsnorm_native(
            ref_output,
            ref_hidden.index_select(1, selected),
            reference.model.layers[0].post_attention_layernorm.weight,
            reference.model.layers[0].post_attention_layernorm.variance_epsilon,
        )
        fused_residual_stats = _difference_stats(
            fused_residual, reference_residual, worst_count=worst_count
        )
        fused_norm_stats = _difference_stats(
            fused_post_norm, reference_post_norm, worst_count=worst_count
        )
        _print_stats("fused residual output", fused_residual_stats)
        _print_stats("fused post-attention RMSNorm", fused_norm_stats)

        row_results = []
        for row_index, position in enumerate(query_positions):
            stats = _difference_stats(
                flux_probs[:, :, row_index],
                torch_probs_for_flux_scores[:, :, row_index],
                rtol=OPERATOR_RTOL,
                atol=OPERATOR_ATOL,
                worst_count=worst_count,
            )
            normalization_error = float(
                (flux_probs[:, :, row_index].sum(dim=-1) - 1.0).abs().max().item()
            )
            row_results.append(
                {
                    "query_position": position,
                    "valid_prefix": position + 1,
                    "masked_future_tokens": length - position - 1,
                    "normalization_error": normalization_error,
                    **asdict(stats),
                }
            )
            _print_stats(f"query {position}", stats)

        finite_mask_values = torch.unique(mask[mask != torch.finfo(mask.dtype).min]).tolist()
        mask_summary = {
            "shape": list(mask.shape),
            "dtype": str(mask.dtype),
            "masked_value": float(torch.finfo(mask.dtype).min),
            "finite_values": finite_mask_values,
            "reference_flux_bitwise_equal": bool(torch.equal(mask, flux_mask)),
        }
        print(f"  mask={mask_summary}")

    result = {
        "length": length,
        "layer": 0,
        "query_positions": query_positions,
        "mask": mask_summary,
        "position_embeddings_bitwise_equal": bool(
            torch.equal(position_embeddings[0], flux_position_embeddings[0])
            and torch.equal(position_embeddings[1], flux_position_embeddings[1])
        ),
        "input_rmsnorm": asdict(input_norm_stats),
        "components": component_results,
        "scores_before_mask": asdict(pre_mask_stats),
        "scores_after_mask": asdict(post_mask_stats),
        "softmax_probabilities": asdict(probability_stats),
        "softmax_native_vs_torch": asdict(softmax_operator_stats),
        "p_times_v": asdict(context_stats),
        "attention_output_projection": asdict(output_stats),
        "fused_residual_output": asdict(fused_residual_stats),
        "fused_post_attention_rmsnorm": asdict(fused_norm_stats),
        "rows": row_results,
    }
    del reference, flux, input_ids, mask, flux_mask, reference_parts, flux_parts
    del ref_scores, flux_scores, ref_masked, flux_masked, ref_probs, flux_probs
    _release()
    return result


def run_masked_softmax(lengths: Sequence[int], worst_count: int) -> dict[str, Any]:
    print("\nLong-row masked softmax", flush=True)
    results: dict[str, Any] = {}
    for key_length in lengths:
        prefixes = sorted({1, 2, 8, key_length // 2, key_length - 1, key_length})
        generator = torch.Generator(device="cuda").manual_seed(1234 + key_length)
        query = torch.randn(
            (1, 9, len(prefixes), 64), generator=generator, device="cuda"
        )
        key = torch.randn(
            (1, 9, key_length, 64), generator=generator, device="cuda"
        )
        scores = torch.matmul(query, key.transpose(2, 3)) * (64**-0.5)
        columns = torch.arange(key_length, device="cuda")
        prefix_tensor = torch.tensor(prefixes, device="cuda").view(1, 1, -1, 1)
        mask = torch.where(
            columns.view(1, 1, 1, -1) < prefix_tensor,
            torch.tensor(0.0, device="cuda"),
            torch.tensor(torch.finfo(torch.float32).min, device="cuda"),
        )
        masked_scores = scores + mask
        with torch.inference_mode():
            actual = softmax_native(masked_scores)
            expected = torch.softmax(masked_scores, dim=-1)
        stats = _difference_stats(
            actual,
            expected,
            rtol=OPERATOR_RTOL,
            atol=OPERATOR_ATOL,
            worst_count=worst_count,
        )
        normalization_error = float(
            (actual.sum(dim=-1) - 1.0).abs().max().item()
        )
        masked_nonzero = 0
        row_results = []
        for row, prefix in enumerate(prefixes):
            row_stats = _difference_stats(
                actual[:, :, row],
                expected[:, :, row],
                rtol=OPERATOR_RTOL,
                atol=OPERATOR_ATOL,
                worst_count=worst_count,
            )
            row_masked_nonzero = int(
                torch.count_nonzero(actual[:, :, row, prefix:]).item()
            )
            masked_nonzero += row_masked_nonzero
            row_results.append(
                {
                    "valid_prefix": prefix,
                    "masked_future_tokens": key_length - prefix,
                    "masked_nonzero_count": row_masked_nonzero,
                    **asdict(row_stats),
                }
            )
        _print_stats(f"K={key_length}", stats)
        print(
            f"    prefixes={prefixes} normalization_max_abs={normalization_error:.9g} "
            f"masked_nonzero={masked_nonzero}"
        )
        results[str(key_length)] = {
            **asdict(stats),
            "prefixes": prefixes,
            "normalization_max_absolute_error": normalization_error,
            "masked_nonzero_count": masked_nonzero,
            "rows": row_results,
        }
        del query, key, scores, mask, masked_scores, actual, expected
        _release()
    return results


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    _configure_runtime()
    print("SmolLM2 Flux correctness diagnostic")
    print(f"  model={MODEL_ID} revision={MODEL_REVISION}")
    print(f"  python={sys.version.split()[0]} torch={torch.__version__} CUDA={torch.version.cuda}")
    print(f"  GPU={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()}")
    print("  dtype=torch.float32 attention=eager TF32=disabled deterministic=True")
    print(f"  model tolerance: rtol={RTOL} atol={ATOL}")
    print(f"  operator tolerance: rtol={OPERATOR_RTOL} atol={OPERATOR_ATOL}")

    output: dict[str, Any] = {
        "environment": {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
        }
    }
    if args.phases in ("all", "isolation"):
        output["isolation"] = run_isolation(args.sequence_lengths, args.worst_count)
    if args.phases in ("all", "layers"):
        output["layers"] = run_layer_localization(args.layer_lengths, args.worst_count)
    if args.phases in ("all", "attention"):
        output["attention"] = run_attention_localization(
            args.attention_length, args.worst_count
        )
    if args.phases in ("all", "masked-softmax"):
        output["masked_softmax"] = run_masked_softmax(
            args.sequence_lengths, args.worst_count
        )
    if args.json is not None:
        args.json.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
