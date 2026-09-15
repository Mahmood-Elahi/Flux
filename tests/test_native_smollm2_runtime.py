"""Focused validation for the full-model native SmolLM2 decode runtime."""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from flux.model import FINAL_FLUX_OPERATOR_CATEGORIES, enable_flux_ops
from flux.runtime import NativeSmolLM2Decode, native_smollm2_runtime_is_available


RTOL = 2e-4
ATOL = 2e-5
_AVAILABLE = torch.cuda.is_available() and native_smollm2_runtime_is_available()
_CUDA_ONLY = pytest.mark.skipif(
    not _AVAILABLE, reason="CUDA and the native full-model runtime are required"
)


def _model() -> LlamaForCausalLM:
    """Create a deterministic canonical 30-layer model."""
    torch.manual_seed(2701)
    config = LlamaConfig(
        hidden_size=576,
        intermediate_size=1536,
        num_hidden_layers=30,
        num_attention_heads=9,
        num_key_value_heads=3,
        head_dim=64,
        vocab_size=128,
        max_position_embeddings=8192,
        rms_norm_eps=1e-5,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).float().cuda().eval()
    enable_flux_ops(model, operators=FINAL_FLUX_OPERATOR_CATEGORIES)
    return model


def _reference_and_flux_models() -> tuple[LlamaForCausalLM, LlamaForCausalLM]:
    """Create independent HF and Flux instances with identical weights."""
    torch.manual_seed(2701)
    config = LlamaConfig(
        hidden_size=576,
        intermediate_size=1536,
        num_hidden_layers=30,
        num_attention_heads=9,
        num_key_value_heads=3,
        head_dim=64,
        vocab_size=128,
        max_position_embeddings=8192,
        rms_norm_eps=1e-5,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
    )
    config._attn_implementation = "eager"
    reference = LlamaForCausalLM(config).float().cuda().eval()
    flux = LlamaForCausalLM(config).float().cuda().eval()
    flux.load_state_dict(reference.state_dict())
    enable_flux_ops(flux, operators=FINAL_FLUX_OPERATOR_CATEGORIES)
    return reference, flux


@_CUDA_ONLY
def test_full_native_repeated_replay_matches_hf_flux_and_all_caches() -> None:
    reference, model = _reference_and_flux_models()
    input_ids = (torch.arange(7, device="cuda").unsqueeze(0) * 11 + 3) % 128
    with torch.inference_mode():
        reference_output = reference(
            input_ids=input_ids, use_cache=True, logits_to_keep=1
        )
        flux_output = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    native = NativeSmolLM2Decode.capture(model, input_ids, max_decode_steps=4)
    assert native.cache_position == native.cache_length == 7
    addresses = native.stable_addresses()
    token = native.prefill_logits.argmax(dim=-1)
    reference_token = reference_output.logits.argmax(dim=-1)
    flux_token = flux_output.logits.argmax(dim=-1)
    assert token.equal(reference_token) and token.equal(flux_token)
    returned_logits = None

    for step in range(3):
        with torch.inference_mode():
            reference_output = reference(
                input_ids=reference_token,
                past_key_values=reference_output.past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            flux_output = model(
                input_ids=flux_token,
                past_key_values=flux_output.past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            actual = native.replay(token)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, reference_output.logits, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(actual, flux_output.logits, rtol=RTOL, atol=ATOL)
        actual_token = actual.argmax(dim=-1)
        reference_token = reference_output.logits.argmax(dim=-1)
        flux_token = flux_output.logits.argmax(dim=-1)
        assert actual_token.equal(reference_token) and actual_token.equal(flux_token)
        assert native.cache_position == native.cache_length == 8 + step
        assert native.stable_addresses() == addresses
        for layer_index, (reference_layer, flux_layer) in enumerate(
            zip(
                reference_output.past_key_values.layers,
                flux_output.past_key_values.layers,
                strict=True,
            )
        ):
            length = 8 + step
            torch.testing.assert_close(
                native.key_cache[layer_index, ..., :length, :],
                flux_layer.keys,
                rtol=RTOL,
                atol=ATOL,
            )
            torch.testing.assert_close(
                native.value_cache[layer_index, ..., :length, :],
                flux_layer.values,
                rtol=RTOL,
                atol=ATOL,
            )
            torch.testing.assert_close(
                flux_layer.keys, reference_layer.keys, rtol=RTOL, atol=ATOL
            )
            torch.testing.assert_close(
                flux_layer.values, reference_layer.values, rtol=RTOL, atol=ATOL
            )
        if returned_logits is None:
            returned_logits = actual
        else:
            assert actual.data_ptr() == returned_logits.data_ptr()
        token = actual_token

    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    assert native.replay(None).data_ptr() == native.logits.data_ptr()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated
    with pytest.raises(RuntimeError, match="exhausted"):
        native.replay(token)


@_CUDA_ONLY
def test_full_native_reset_public_api() -> None:
    model = _model()
    input_ids = torch.tensor([[5, 7, 9, 11]], device="cuda")
    native = NativeSmolLM2Decode.capture(model, input_ids, max_decode_steps=2)
    token = native.prefill_logits.argmax(dim=-1)
    keys = native._source_key_caches
    values = native._source_value_caches

    native.reset(token, keys, values, cache_position=4)
    assert native.cache_position == native.cache_length == 4
    with pytest.raises(RuntimeError, match="one K/V tensor per layer"):
        native.reset(token, keys[:-1], values, cache_position=4)
