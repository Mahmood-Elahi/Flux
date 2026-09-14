"""Focused validation for the full-model native SmolLM2 decode runtime."""

from __future__ import annotations

import gc

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from flux.model import FINAL_FLUX_OPERATOR_CATEGORIES, enable_flux_ops
from flux.model.smollm2_cuda_graph import FluxCUDAGraphDecode
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


@_CUDA_ONLY
def test_full_native_repeated_replay_matches_python_graph_and_all_caches() -> None:
    model = _model()
    input_ids = (torch.arange(7, device="cuda").unsqueeze(0) * 11 + 3) % 128
    python_graph = FluxCUDAGraphDecode.capture(
        model, input_ids, max_decode_steps=4, warmup_steps=1
    )
    native = NativeSmolLM2Decode.capture(model, input_ids, max_decode_steps=4)
    assert native.cache_position == native.cache_length == 7
    addresses = native.stable_addresses()
    token = native.prefill_logits.argmax(dim=-1)

    for step in range(3):
        expected = python_graph.replay(token)
        actual = native.replay(token)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
        assert actual.argmax(dim=-1).item() == expected.argmax(dim=-1).item()
        assert native.cache_position == native.cache_length == 8 + step
        assert native.stable_addresses() == addresses
        for layer_index, layer in enumerate(python_graph.cache.layers):
            length = 8 + step
            torch.testing.assert_close(
                native.key_cache[layer_index, ..., :length, :],
                layer.keys[..., :length, :],
                rtol=RTOL,
                atol=ATOL,
            )
            torch.testing.assert_close(
                native.value_cache[layer_index, ..., :length, :],
                layer.values[..., :length, :],
                rtol=RTOL,
                atol=ATOL,
            )
        token = actual.argmax(dim=-1)

    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    assert native.replay(None).data_ptr() == native.logits.data_ptr()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated
    with pytest.raises(RuntimeError, match="exhausted"):
        native.replay(token)


@_CUDA_ONLY
def test_full_native_reset_current_stream_and_recreation() -> None:
    model = _model()
    input_ids = torch.tensor([[5, 7, 9, 11]], device="cuda")
    native = NativeSmolLM2Decode.capture(model, input_ids, max_decode_steps=2)
    token = native.prefill_logits.argmax(dim=-1)
    keys = native._source_key_caches
    values = native._source_value_caches
    addresses = native.stable_addresses()

    stream = torch.cuda.Stream()
    produced = torch.empty_like(token)
    consumed = torch.empty_like(native.logits)
    with torch.cuda.stream(stream):
        produced.copy_(token)
        result = native.replay(produced)
        consumed.copy_(result)
    stream.synchronize()
    torch.testing.assert_close(consumed, native.logits)
    assert native.stable_addresses() == addresses

    native.reset(token, keys, values, cache_position=4)
    assert native.cache_position == native.cache_length == 4
    with pytest.raises(RuntimeError, match="one K/V tensor per layer"):
        native.reset(token, keys[:-1], values, cache_position=4)
    first = native.replay(token).clone()
    torch.cuda.synchronize()
    native.reset(token, keys, values, cache_position=4)
    second = native.replay(token).clone()
    torch.cuda.synchronize()
    torch.testing.assert_close(first, second, rtol=0, atol=0)

    del native
    gc.collect()
    recreated = NativeSmolLM2Decode.capture(model, input_ids, max_decode_steps=1)
    assert recreated.replay(token).shape == (1, 1, 128)
    torch.cuda.synchronize()
