"""Dispatch, numerical, metadata, stream, and graph tests for fused GQA decode."""

from __future__ import annotations

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from flux.ops import (
    gqa_decode_attention,
    gqa_decode_attention_native,
    gqa_decode_attention_native_out,
    native_gqa_decode_attention_is_available,
)


# The fused kernel changes the FP32 reduction tree relative to cuBLAS QK/PV
# and PyTorch softmax; these tolerances cover that ordering difference. The
# benchmark records errors two orders of magnitude below the absolute bound.
RTOL = 2e-5
ATOL = 1e-6

pytestmark = pytest.mark.skipif(
    not native_gqa_decode_attention_is_available(),
    reason="Flux native GQA decode attention operator has not been built",
)


def _devices() -> list[str]:
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def _inputs(
    device: str, capacity: int, *, batch: int = 1, head_dim: int = 64
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(1234 + capacity)
    query = torch.randn((batch, 9, 1, head_dim), generator=generator, device=device)
    key = torch.randn((batch, 3, capacity, head_dim), generator=generator, device=device)
    value = torch.randn((batch, 3, capacity, head_dim), generator=generator, device=device)
    mask = torch.randn((batch, 1, 1, capacity), generator=generator, device=device) * 0.05
    return query, key, value, mask


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize(
    "capacity,valid_length",
    [(129, 73)],
)
def test_matches_reference_with_dynamic_and_static_cache_lengths(
    device: str, capacity: int, valid_length: int
) -> None:
    query, key, value, mask = _inputs(device, capacity)
    length = None if valid_length == capacity else torch.tensor(valid_length, device=device)

    expected = gqa_decode_attention(query, key, value, mask, 0.125, length)
    actual = gqa_decode_attention_native(query, key, value, mask, 0.125, length)

    assert actual.shape == query.shape
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("device", _devices())
def test_supports_none_and_broadcast_masks(device: str) -> None:
    query, key, value, _ = _inputs(device, 31, head_dim=8)
    for mask in (
        None,
        torch.zeros((1, 1, 1, 1), device=device),
        torch.randn((1, 9, 1, 31), device=device),
    ):
        actual = gqa_decode_attention_native(query, key, value, mask, 8**-0.5)
        expected = gqa_decode_attention(query, key, value, mask, 8**-0.5)
        torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("device", _devices())
def test_reads_noncontiguous_cache_without_copying_or_modifying(device: str) -> None:
    query, key_source, value_source, mask = _inputs(device, 37)
    key = key_source.transpose(2, 3)
    value = value_source.transpose(2, 3)
    # Restore [B, KVH, L, D] with non-unit position stride.
    key = key.transpose(2, 3)[..., ::2, :]
    value = value.transpose(2, 3)[..., ::2, :]
    mask = mask[..., ::2]
    before = (query.clone(), key.clone(), value.clone(), mask.clone())

    actual = gqa_decode_attention_native(query, key, value, mask, 0.125)
    expected = gqa_decode_attention(query, key, value, mask, 0.125)

    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    for tensor, original in zip((query, key, value, mask), before, strict=True):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)


def test_rejects_invalid_arguments() -> None:
    query, key, value, mask = _inputs("cpu", 17)
    batched_query, batched_key, batched_value, batched_mask = _inputs(
        "cpu", 17, batch=2
    )
    with pytest.raises(RuntimeError, match="batch size must be one"):
        gqa_decode_attention_native(
            batched_query, batched_key, batched_value, batched_mask, 0.125
        )
    with pytest.raises(RuntimeError, match="query length must be one"):
        gqa_decode_attention_native(query.expand(1, 9, 2, 64), key, value, mask, 0.125)
    with pytest.raises(RuntimeError, match="divisible"):
        gqa_decode_attention_native(query[:, :8], key, value, mask[:, :, :, :], 0.125)
    with pytest.raises(RuntimeError, match="float32"):
        gqa_decode_attention_native(query.half(), key, value, mask, 0.125)
    with pytest.raises(RuntimeError, match="broadcastable"):
        gqa_decode_attention_native(query, key, value, mask[..., :-1], 0.125)
    with pytest.raises(RuntimeError, match="within cache capacity"):
        gqa_decode_attention_native(query, key, value, mask, 0.125, torch.tensor(18))


def test_rejects_autograd_input() -> None:
    query, key, value, mask = _inputs("cpu", 17)
    query.requires_grad_()
    with pytest.raises(RuntimeError, match="inference-only"):
        gqa_decode_attention_native(query, key, value, mask, 0.125)


def test_fake_tensor_metadata() -> None:
    mode = FakeTensorMode()
    with mode:
        query = torch.empty((1, 9, 1, 64))
        key = torch.empty((1, 3, 129, 64))
        value = torch.empty_like(key)
        mask = torch.empty((1, 1, 1, 129))
        length = torch.empty((), dtype=torch.int64)
        output = gqa_decode_attention_native(query, key, value, mask, 0.125, length)
    assert isinstance(output, FakeTensor)
    assert output.shape == query.shape
    assert output.dtype == query.dtype
    assert output.device == query.device
    assert output.is_contiguous()


@pytest.mark.parametrize("device", _devices())
def test_torch_library_opcheck(device: str) -> None:
    query, key, value, mask = _inputs(device, 17, head_dim=8)
    length = torch.tensor(11, device=device)
    result = torch.library.opcheck(
        torch.ops.flux.gqa_decode_attention.default,
        (query, key, value, mask, 8**-0.5, length),
        rtol=RTOL,
        atol=ATOL,
    )
    assert all(status == "SUCCESS" for status in result.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_out_contract_returns_and_overwrites_supplied_tensor() -> None:
    query, key, value, mask = _inputs("cuda", 1024)
    length = torch.tensor(777, device="cuda")
    output = torch.full_like(query, 67.0)
    workspace = torch.empty((1, 9, 8, 66), device="cuda")
    assert gqa_decode_attention_native_out(
        query, key, value, mask, 0.125, length, output, workspace
    ) is output
    torch.testing.assert_close(
        output,
        gqa_decode_attention_native(query, key, value, mask, 0.125, length),
        rtol=0,
        atol=0,
    )
