"""Tests for the FP32 attention score post-processing reference."""

from __future__ import annotations

import pytest
import torch

from flux.ops.attention_score_softmax import attention_score_softmax


def test_matches_exact_unfused_expression() -> None:
    generator = torch.Generator().manual_seed(1234)
    scores = torch.randn((2, 4, 7, 11), generator=generator)
    mask = torch.randn((2, 1, 7, 11), generator=generator)
    scale = 7**-0.5

    actual = attention_score_softmax(scores, mask, scale)
    expected = torch.softmax(scores * scale + mask, dim=-1)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_supports_broadcast_mask_dimensions() -> None:
    scores = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    mask = torch.zeros((1, 1, 4, 5), dtype=torch.float32)

    actual = attention_score_softmax(scores, mask, 0.125)

    assert actual.shape == scores.shape
    torch.testing.assert_close(actual.sum(dim=-1), torch.ones((2, 3, 4)))


@pytest.mark.parametrize(
    ("scores", "mask", "message"),
    [
        (torch.ones(2, 3), torch.ones(2, 3), "rank four"),
        (torch.ones(1, 2, 3, 4), torch.ones(1, 2, 3), "rank four"),
        (torch.ones(1, 2, 3, 4), torch.ones(1, 2, 2, 4), "broadcastable"),
        (torch.ones(1, 2, 3, 4), torch.ones(1, 1, 3, 5), "broadcastable"),
    ],
)
def test_rejects_invalid_shapes(
    scores: torch.Tensor, mask: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        attention_score_softmax(scores, mask, 0.5)


def test_rejects_non_fp32_and_nonfinite_scale() -> None:
    scores = torch.ones((1, 2, 3, 4))
    mask = torch.zeros((1, 1, 3, 4))
    with pytest.raises(TypeError, match="scores"):
        attention_score_softmax(scores.half(), mask, 0.5)
    with pytest.raises(TypeError, match="additive_attention_mask"):
        attention_score_softmax(scores, mask.half(), 0.5)
    with pytest.raises(ValueError, match="finite"):
        attention_score_softmax(scores, mask, float("inf"))
