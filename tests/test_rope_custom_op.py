"""Correctness, dispatch, FakeTensor, and stream tests for native RoPE."""

from __future__ import annotations

import pytest
import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from flux.ops import (
    native_rope_is_available,
    native_rope_load_error,
    rope_native,
)


RTOL = 1e-6
ATOL = 2e-7

if native_rope_load_error() is not None:
    raise RuntimeError("The built Flux native operator library failed to load") from (
        native_rope_load_error()
    )

pytestmark = pytest.mark.skipif(
    not native_rope_is_available(),
    reason="Flux native operators have not been built",
)


def _devices() -> list[str]:
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def _inputs(
    batch: int,
    sequence: int,
    offset: int,
    device: str,
    *,
    shared_positions: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    count = batch * sequence * 576
    hidden = torch.sin(torch.arange(count, device=device).float() * 0.013).reshape(
        batch, sequence, 576
    )
    # These are the actual non-contiguous views produced by Llama projections.
    query = hidden.view(batch, sequence, 9, 64).transpose(1, 2)
    key = hidden[..., :192].view(batch, sequence, 3, 64).transpose(1, 2)
    positions = torch.arange(offset, offset + sequence, device=device).float()
    inv_freq = 1.0 / (
        100000.0 ** (torch.arange(0, 64, 2, device=device).float() / 64)
    )
    frequencies = torch.outer(positions, inv_freq)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    rope_batch = 1 if shared_positions else batch
    cos = embedding.cos().unsqueeze(0).expand(rope_batch, -1, -1)
    sin = embedding.sin().unsqueeze(0).expand(rope_batch, -1, -1)
    return query, key, cos, sin


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize(
    ("batch", "sequence", "offset", "shared_positions"),
    [
        (1, 1, 0, True),
        (1, 1, 4095, True),
        (1, 7, 37, True),
        (2, 128, 1024, False),
        (2, 17, 8191, True),
    ],
)
def test_matches_transformers_for_prefill_and_decode(
    device: str,
    batch: int,
    sequence: int,
    offset: int,
    shared_positions: bool,
) -> None:
    query, key, cos, sin = _inputs(
        batch, sequence, offset, device, shared_positions=shared_positions
    )
    # A length-one transpose is contiguity-degenerate; multi-token projection
    # views exercise the real strided prefill layout.
    if sequence > 1:
        assert not query.is_contiguous()
        assert not key.is_contiguous()
    expected = apply_rotary_pos_emb(query, key, cos, sin)
    with torch.inference_mode():
        actual = rope_native(query, key, cos, sin)
    assert actual[0].is_contiguous() and actual[1].is_contiguous()
    torch.testing.assert_close(actual[0], expected[0], rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(actual[1], expected[1], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("device", _devices())
def test_reads_non_contiguous_embeddings_without_copying_contract(device: str) -> None:
    query, key, cos, sin = _inputs(2, 7, 99, device)
    cos_storage = torch.empty(2, 7, 128, device=device)
    sin_storage = torch.empty(2, 7, 128, device=device)
    cos_storage[..., ::2].copy_(cos)
    sin_storage[..., ::2].copy_(sin)
    strided_cos = cos_storage[..., ::2]
    strided_sin = sin_storage[..., ::2]
    assert not strided_cos.is_contiguous() and not strided_sin.is_contiguous()
    expected = apply_rotary_pos_emb(query, key, strided_cos, strided_sin)
    with torch.inference_mode():
        actual = rope_native(query, key, strided_cos, strided_sin)
    torch.testing.assert_close(actual[0], expected[0], rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(actual[1], expected[1], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("device", _devices())
def test_is_deterministic(device: str) -> None:
    inputs = _inputs(1, 31, 2048, device)
    with torch.inference_mode():
        first = rope_native(*inputs)
        second = rope_native(*inputs)
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_uses_current_non_default_cuda_stream() -> None:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.inference_mode():
        inputs = _inputs(1, 128, 777, "cuda")
        expected = apply_rotary_pos_emb(*inputs)
        actual = rope_native(*inputs)
        downstream = (actual[0] + 0.0, actual[1] + 0.0)
    stream.synchronize()
    torch.testing.assert_close(downstream[0], expected[0], rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(downstream[1], expected[1], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("device", _devices())
def test_torch_library_opcheck(device: str) -> None:
    inputs = _inputs(2, 7, 31, device)
    with torch.inference_mode():
        result = torch.library.opcheck(
            torch.ops.flux.rope.default,
            inputs,
            rtol=RTOL,
            atol=ATOL,
        )
    assert all(status == "SUCCESS" for status in result.values())


def test_rejects_invalid_shapes_and_dtypes() -> None:
    query, key, cos, sin = _inputs(1, 3, 0, "cpu")
    with pytest.raises(RuntimeError, match="rank four"):
        rope_native(query[0], key, cos, sin)
    with pytest.raises(RuntimeError, match="float32"):
        rope_native(query.double(), key, cos, sin)
    with pytest.raises(RuntimeError, match="even"):
        rope_native(query[..., :63], key[..., :63], cos[..., :63], sin[..., :63])


def test_rejects_autograd_inputs() -> None:
    query, key, cos, sin = _inputs(1, 3, 0, "cpu")
    query.requires_grad_()
    with pytest.raises(RuntimeError, match="inference-only"):
        rope_native(query, key, cos, sin)
