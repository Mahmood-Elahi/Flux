"""Validate fused score processing against the current Flux SmolLM2 path."""

from __future__ import annotations

import torch
from transformers import DynamicCache

from flux.model.smollm2 import load_model
from flux.model.smollm2_flux import enable_flux_ops


def _clone_cache(cache: DynamicCache, config: object) -> DynamicCache:
    return DynamicCache(
        [
            (layer.keys.detach().clone(), layer.values.detach().clone())
            for layer in cache.layers
        ],
        config=config,
    )


def _maximum_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual - expected).abs().max())


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    current = enable_flux_ops(
        load_model("cuda"),
        fuse_attention_scores=False,
    )
    fused = enable_flux_ops(
        load_model("cuda"),
        fuse_attention_scores=True,
    )
    input_ids = ((torch.arange(128, device="cuda") * 17 + 11) % current.config.vocab_size).unsqueeze(0)
    next_token = torch.tensor([[123]], device="cuda")
    current_layers: list[torch.Tensor] = []
    fused_layers: list[torch.Tensor] = []
    current_hook = current.model.layers[0].register_forward_hook(
        lambda _module, _args, output: current_layers.append(output.detach().clone())
    )
    fused_hook = fused.model.layers[0].register_forward_hook(
        lambda _module, _args, output: fused_layers.append(output.detach().clone())
    )

    with torch.inference_mode():
        current_prefill = current(input_ids=input_ids, use_cache=True)
        fused_prefill = fused(input_ids=input_ids, use_cache=True)
    current_hook.remove()
    fused_hook.remove()
    if len(current_layers) != 1 or len(fused_layers) != 1:
        raise AssertionError("first-layer hooks did not run exactly once")
    torch.testing.assert_close(fused_layers[0], current_layers[0], rtol=0, atol=0)
    torch.testing.assert_close(
        fused_prefill.logits,
        current_prefill.logits,
        rtol=0,
        atol=0,
    )
    for current_layer, fused_layer in zip(
        current_prefill.past_key_values.layers,
        fused_prefill.past_key_values.layers,
        strict=True,
    ):
        torch.testing.assert_close(fused_layer.keys, current_layer.keys, rtol=0, atol=0)
        torch.testing.assert_close(fused_layer.values, current_layer.values, rtol=0, atol=0)

    current_cache = _clone_cache(current_prefill.past_key_values, current.config)
    fused_cache = _clone_cache(fused_prefill.past_key_values, fused.config)
    with torch.inference_mode():
        current_decode = current(
            input_ids=next_token,
            past_key_values=current_cache,
            use_cache=True,
        )
        fused_decode = fused(
            input_ids=next_token,
            past_key_values=fused_cache,
            use_cache=True,
        )
        current_generation = current.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=8,
            use_cache=True,
        )
        fused_generation = fused.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=8,
            use_cache=True,
        )
    torch.testing.assert_close(
        fused_decode.logits,
        current_decode.logits,
        rtol=0,
        atol=0,
    )
    if current_decode.past_key_values.get_seq_length() != 129:
        raise AssertionError("current Flux decode cache length is not 129")
    if fused_decode.past_key_values.get_seq_length() != 129:
        raise AssertionError("fused Flux decode cache length is not 129")
    if not torch.equal(fused_generation, current_generation):
        raise AssertionError("greedy generation differs")

    print(f"first-layer max abs error: {_maximum_error(fused_layers[0], current_layers[0]):.9g}")
    print(f"prefill logits max abs error: {_maximum_error(fused_prefill.logits, current_prefill.logits):.9g}")
    print(f"decode logits max abs error: {_maximum_error(fused_decode.logits, current_decode.logits):.9g}")
    print(f"KV-cache length: {fused_decode.past_key_values.get_seq_length()}")
    print(f"greedy tokens identical: {torch.equal(fused_generation, current_generation)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
