"""Focused native prompt-to-cache and decode-handoff validation."""

from __future__ import annotations

import gc

import pytest
import torch

from flux.runtime import (
    NativeSmolLM2Prefill,
    native_smollm2_greedy_generate,
    native_smollm2_prefill_is_available,
)
from tests.test_native_smollm2_runtime import _model


RTOL = 2e-4
ATOL = 2e-5
_AVAILABLE = torch.cuda.is_available() and native_smollm2_prefill_is_available()
_CUDA_ONLY = pytest.mark.skipif(
    not _AVAILABLE, reason="CUDA and the native prefill runtime are required"
)


def _ids(length: int) -> torch.Tensor:
    return (torch.arange(length, device="cuda").unsqueeze(0) * 11 + 3) % 128


@_CUDA_ONLY
def test_native_prefill_matches_logits_every_layer_cache_and_decode() -> None:
    model = _model()
    input_ids = _ids(17)
    with torch.inference_mode():
        expected = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        native = NativeSmolLM2Prefill.capture(
            model, input_ids, max_decode_steps=3
        )
    torch.cuda.synchronize()
    torch.testing.assert_close(native.logits, expected.logits, rtol=RTOL, atol=ATOL)
    assert native.prompt_length == native.cache_position == native.cache_length == 17
    assert native.capacity == 20
    assert native.key_cache.shape == native.value_cache.shape == (30, 1, 3, 20, 64)
    for layer_index, layer in enumerate(expected.past_key_values.layers):
        torch.testing.assert_close(
            native.key_cache[layer_index, ..., :17, :],
            layer.keys,
            rtol=RTOL,
            atol=ATOL,
        )
        torch.testing.assert_close(
            native.value_cache[layer_index, ..., :17, :],
            layer.values,
            rtol=RTOL,
            atol=ATOL,
        )

    token = expected.logits.argmax(dim=-1)
    with torch.inference_mode():
        expected_decode = model(
            input_ids=token,
            past_key_values=expected.past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
        actual_decode = native.replay(token)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual_decode, expected_decode.logits, rtol=RTOL, atol=ATOL
    )
    assert actual_decode.argmax(dim=-1).item() == expected_decode.logits.argmax(dim=-1).item()
    assert native.cache_position == native.cache_length == 18


@_CUDA_ONLY
def test_native_prefill_reuse_current_stream_memory_and_addresses() -> None:
    model = _model()
    first_ids = _ids(9)
    second_ids = (first_ids + 7) % 128
    native = NativeSmolLM2Prefill.capture(model, first_ids, max_decode_steps=1)
    addresses = native.stable_addresses()
    stream = torch.cuda.Stream()
    consumed = torch.empty_like(native.logits)
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    with torch.cuda.stream(stream), torch.inference_mode():
        consumed.copy_(native.prefill(second_ids))
    stream.synchronize()
    assert torch.cuda.memory_allocated() == allocated
    assert native.stable_addresses() == addresses
    with torch.inference_mode():
        expected = model(input_ids=second_ids, use_cache=True, logits_to_keep=1)
    torch.testing.assert_close(consumed, expected.logits, rtol=RTOL, atol=ATOL)
    for layer_index, layer in enumerate(expected.past_key_values.layers):
        torch.testing.assert_close(
            native.key_cache[layer_index, ..., :9, :],
            layer.keys,
            rtol=RTOL,
            atol=ATOL,
        )
        torch.testing.assert_close(
            native.value_cache[layer_index, ..., :9, :],
            layer.values,
            rtol=RTOL,
            atol=ATOL,
        )

    assert native.memory.cache_bytes == 2 * 30 * 1 * 3 * 10 * 64 * 4
    del native
    gc.collect()
    recreated = NativeSmolLM2Prefill.capture(model, first_ids, max_decode_steps=0)
    assert recreated.cache_position == 9
    with pytest.raises(RuntimeError, match="exhausted"):
        recreated.replay()


@_CUDA_ONLY
def test_native_prefill_rejects_invalid_inputs_and_capacity() -> None:
    model = _model()
    ids = _ids(5)
    with pytest.raises(ValueError, match="non-negative"):
        NativeSmolLM2Prefill.capture(model, ids, max_decode_steps=-1)
    with pytest.raises(ValueError, match="batch of one"):
        NativeSmolLM2Prefill.capture(model, ids.expand(2, -1))
    with pytest.raises(ValueError, match="vocabulary"):
        NativeSmolLM2Prefill.capture(model, torch.full_like(ids, 128))
    native = NativeSmolLM2Prefill.capture(model, ids)
    with pytest.raises(ValueError, match="prompt shape"):
        native.prefill(ids[:, :-1])


@_CUDA_ONLY
def test_native_prefill_single_token_greedy_generation() -> None:
    model = _model()
    ids = _ids(7)
    with torch.inference_mode():
        expected = model(input_ids=ids, use_cache=False, logits_to_keep=1)
        generated = native_smollm2_greedy_generate(
            model, ids, max_new_tokens=1
        )
    assert generated.shape == (1, 8)
    assert generated[:, :-1].equal(ids)
    assert generated[:, -1:].equal(expected.logits.argmax(dim=-1))
