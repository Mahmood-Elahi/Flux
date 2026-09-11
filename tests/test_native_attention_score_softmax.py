"""Correctness tests for fused native attention score post-processing."""

from __future__ import annotations

import math

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from flux.ops import (
    attention_score_softmax,
    attention_score_softmax_native,
    native_attention_score_softmax_is_available,
)


RTOL = 1e-5
ATOL = 5e-7

pytestmark = pytest.mark.skipif(
    not native_attention_score_softmax_is_available(),
    reason="Flux native attention score softmax operator has not been built",
)


def _devices() -> list[str]:
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def _scores(shape: tuple[int, ...], device: str) -> torch.Tensor:
    values = torch.arange(math.prod(shape), device=device, dtype=torch.float32)
    return (3.0 * torch.sin(values * 0.013) + torch.cos(values * 0.007)).reshape(shape)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("shape", [(1, 4, 1, 1), (2, 3, 5, 17), (1, 9, 8, 129)])
def test_matches_unfused_expression(device: str, shape: tuple[int, ...]) -> None:
    scores = _scores(shape, device)
    mask = _scores((shape[0], 1, shape[2], shape[3]), device) * 0.25
    scale = shape[-1] ** -0.5

    actual = attention_score_softmax_native(scores, mask, scale)
    expected = scores * scale
    expected = expected + mask
    expected = torch.softmax(expected, dim=-1)

    assert actual.shape == scores.shape
    assert actual.dtype == torch.float32
    assert actual.device == scores.device
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("key_length", [8, 512, 1024, 2048, 4096, 8192])
def test_realistic_causal_masks_and_query_positions(key_length: int) -> None:
    prefixes = sorted({1, 2, min(17, key_length), key_length // 2, key_length - 1, key_length})
    prefixes = [prefix for prefix in prefixes if prefix > 0]
    query_length = len(prefixes)
    generator = torch.Generator(device="cuda").manual_seed(1000 + key_length)
    scores = torch.randn(
        (2, 9, query_length, key_length),
        generator=generator,
        device="cuda",
    )
    columns = torch.arange(key_length, device="cuda").reshape(1, 1, 1, -1)
    valid_prefix = torch.tensor(prefixes, device="cuda").reshape(1, 1, -1, 1)
    mask = torch.where(
        columns < valid_prefix,
        torch.tensor(0.0, device="cuda"),
        torch.tensor(torch.finfo(torch.float32).min, device="cuda"),
    )

    actual = attention_score_softmax_native(scores, mask, 64**-0.5)
    expected = torch.softmax(scores * (64**-0.5) + mask, dim=-1)

    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    for row, prefix in enumerate(prefixes):
        assert torch.count_nonzero(actual[:, :, row, prefix:]) == 0


@pytest.mark.parametrize("device", _devices())
def test_supports_batch_head_and_query_mask_broadcast(device: str) -> None:
    scores = _scores((2, 3, 4, 11), device)
    for mask_shape in (
        (1, 1, 1, 1),
        (1, 1, 1, 11),
        (2, 1, 4, 11),
        (1, 3, 1, 11),
        (2, 3, 4, 11),
    ):
        mask = _scores(mask_shape, device)
        actual = attention_score_softmax_native(scores, mask, 0.125)
        expected = attention_score_softmax(scores, mask, 0.125)
        torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("device", _devices())
def test_makes_noncontiguous_inputs_contiguous_without_modifying_them(device: str) -> None:
    scores = _scores((2, 3, 11, 5), device).transpose(2, 3)
    mask = _scores((2, 1, 11, 5), device).transpose(2, 3)
    scores_before = scores.clone()
    mask_before = mask.clone()

    actual = attention_score_softmax_native(scores, mask, 0.5)

    assert actual.is_contiguous()
    torch.testing.assert_close(scores, scores_before, rtol=0, atol=0)
    torch.testing.assert_close(mask, mask_before, rtol=0, atol=0)
    torch.testing.assert_close(
        actual,
        attention_score_softmax(scores, mask, 0.5),
        rtol=RTOL,
        atol=ATOL,
    )


def test_rejects_invalid_arguments() -> None:
    scores = torch.ones((1, 2, 3, 4))
    mask = torch.zeros((1, 1, 3, 4))
    with pytest.raises(RuntimeError, match="rank four"):
        attention_score_softmax_native(scores.reshape(2, 3, 4), mask, 0.5)
    with pytest.raises(RuntimeError, match="broadcastable"):
        attention_score_softmax_native(scores, torch.zeros((1, 1, 2, 4)), 0.5)
    with pytest.raises(RuntimeError, match="scores must have dtype torch.float32"):
        attention_score_softmax_native(scores.half(), mask, 0.5)
    with pytest.raises(RuntimeError, match="additive_attention_mask"):
        attention_score_softmax_native(scores, mask.half(), 0.5)
    with pytest.raises(RuntimeError, match="finite FP32"):
        attention_score_softmax_native(scores, mask, float("inf"))


def test_rejects_autograd_input() -> None:
    scores = torch.ones((1, 2, 3, 4), requires_grad=True)
    mask = torch.zeros((1, 1, 3, 4))
    with pytest.raises(RuntimeError, match="inference-only"):
        attention_score_softmax_native(scores, mask, 0.5)


def test_fake_tensor_metadata() -> None:
    mode = FakeTensorMode()
    with mode:
        scores = torch.empty((2, 9, 7, 129), dtype=torch.float32)
        mask = torch.empty((2, 1, 7, 129), dtype=torch.float32)
        output = attention_score_softmax_native(scores, mask, 0.125)
    assert isinstance(output, FakeTensor)
    assert output.shape == scores.shape
    assert output.dtype == scores.dtype
    assert output.device == scores.device
    assert output.is_contiguous()


@pytest.mark.parametrize("device", _devices())
def test_torch_library_opcheck(device: str) -> None:
    scores = _scores((2, 3, 4, 17), device)
    mask = _scores((2, 1, 4, 17), device)
    result = torch.library.opcheck(
        torch.ops.flux.attention_score_softmax.default,
        (scores, mask, 0.125),
        rtol=RTOL,
        atol=ATOL,
    )
    assert all(status == "SUCCESS" for status in result.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_uses_current_non_default_cuda_stream() -> None:
    values = _scores((1, 9, 8, 257), "cuda")
    input_scores = torch.zeros_like(values)
    mask = torch.zeros((1, 1, 8, 257), device="cuda")
    expected = attention_score_softmax(values.cpu(), mask.cpu(), 0.125)
    stream = torch.cuda.Stream()
    assert stream != torch.cuda.default_stream()

    with torch.cuda.stream(stream):
        torch.cuda._sleep(10_000_000)
        input_scores.copy_(values)
        output = attention_score_softmax_native(input_scores, mask, 0.125)
        consumed = output + 0.0

    stream.synchronize()
    torch.testing.assert_close(consumed.cpu(), expected, rtol=RTOL, atol=ATOL)
