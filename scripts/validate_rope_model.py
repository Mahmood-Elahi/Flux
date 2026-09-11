"""Validate Flux RoPE on the pinned FP32 SmolLM2-135M model."""

from __future__ import annotations

import argparse
import copy

import torch

from flux.model.smollm2 import load_model
from flux.model.smollm2_flux import enable_flux_ops


RTOL = 2e-5
ATOL = 2e-6


def _maximum_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    return float((actual - expected).abs().max())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--prompt-length", type=int, default=32)
    parser.add_argument("--new-tokens", type=int, default=6)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    torch.manual_seed(20260911)
    reference = load_model(args.device)
    integrated = copy.deepcopy(reference)
    enable_flux_ops(integrated, operators=("rope",))
    vocab_size = reference.config.vocab_size
    input_ids = (
        (torch.arange(args.prompt_length, device=args.device) * 17 + 11)
        % vocab_size
    ).unsqueeze(0)

    hidden = torch.randn(1, 7, reference.config.hidden_size, device=args.device)
    position_ids = torch.arange(1024, 1031, device=args.device).unsqueeze(0)
    position_embeddings = reference.model.rotary_emb(hidden, position_ids)
    causal_mask = torch.full(
        (1, 1, 7, 7),
        torch.finfo(torch.float32).min,
        device=args.device,
    ).triu(diagonal=1)
    with torch.inference_mode():
        expected_layer = reference.model.layers[0](
            hidden,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
        )
        actual_layer = integrated.model.layers[0](
            hidden,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
        )
        layer_error = _maximum_error(actual_layer, expected_layer)

        expected_prefill = reference(input_ids=input_ids, use_cache=True)
        actual_prefill = integrated(input_ids=input_ids, use_cache=True)
        prefill_error = _maximum_error(
            actual_prefill.logits, expected_prefill.logits
        )
        next_token = expected_prefill.logits[:, -1:].argmax(dim=-1)
        expected_decode = reference(
            input_ids=next_token,
            past_key_values=expected_prefill.past_key_values,
            use_cache=True,
        )
        actual_decode = integrated(
            input_ids=next_token,
            past_key_values=actual_prefill.past_key_values,
            use_cache=True,
        )
        decode_error = _maximum_error(actual_decode.logits, expected_decode.logits)

        expected_tokens = reference.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=args.new_tokens,
            use_cache=True,
        )
        actual_tokens = integrated.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=args.new_tokens,
            use_cache=True,
        )
    if not torch.equal(actual_tokens, expected_tokens):
        raise AssertionError("greedy generation tokens differ")
    print(f"device={args.device}")
    print(f"layer max_abs_error={layer_error:.9g}")
    print(f"prefill logits max_abs_error={prefill_error:.9g}")
    print(f"decode logits max_abs_error={decode_error:.9g}")
    print(f"greedy generation tokens={expected_tokens.tolist()} (exact match)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
