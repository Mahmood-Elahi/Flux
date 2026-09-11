"""Tests for the explicit FP32 PyTorch RoPE correctness oracle."""

from __future__ import annotations

import pytest
import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from flux.ops import rope


def _inputs(
    batch: int, sequence: int, position_offset: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(191)
    query = torch.randn(batch, 9, sequence, 64)
    key = torch.randn(batch, 3, sequence, 64)
    positions = torch.arange(position_offset, position_offset + sequence).float()
    inv_freq = 1.0 / (100000.0 ** (torch.arange(0, 64, 2).float() / 64))
    frequencies = torch.outer(positions, inv_freq)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cos = embedding.cos().unsqueeze(0).expand(batch, -1, -1)
    sin = embedding.sin().unsqueeze(0).expand(batch, -1, -1)
    return query, key, cos, sin


@pytest.mark.parametrize(
    ("batch", "sequence", "position_offset"),
    [(1, 1, 0), (1, 1, 4095), (1, 7, 37), (2, 128, 1024)],
)
def test_matches_transformers(
    batch: int, sequence: int, position_offset: int
) -> None:
    query, key, cos, sin = _inputs(batch, sequence, position_offset)
    expected = apply_rotary_pos_emb(query, key, cos, sin)
    actual = rope(query, key, cos, sin)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_single_rope_batch_broadcasts_across_attention_batch() -> None:
    query, key, cos, sin = _inputs(2, 5, 123)
    query[1].copy_(query[0])
    key[1].copy_(key[0])
    actual = rope(query, key, cos[:1], sin[:1])
    torch.testing.assert_close(actual[0][0], actual[0][1], rtol=0, atol=0)
    torch.testing.assert_close(actual[1][0], actual[1][1], rtol=0, atol=0)


def test_rejects_odd_head_dimension() -> None:
    with pytest.raises(ValueError, match="even"):
        rope(
            torch.ones(1, 2, 3, 7),
            torch.ones(1, 1, 3, 7),
            torch.ones(1, 3, 7),
            torch.ones(1, 3, 7),
        )
