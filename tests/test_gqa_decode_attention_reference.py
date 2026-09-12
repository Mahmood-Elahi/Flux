"""Correctness tests for the explicit one-token GQA attention reference."""

from __future__ import annotations

import pytest
import torch

from flux.ops import gqa_decode_attention


def _unexpanded_oracle(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
    scale: float,
    length: int,
) -> torch.Tensor:
    outputs = []
    groups = query.shape[1] // key.shape[1]
    for query_head in range(query.shape[1]):
        kv_head = query_head // groups
        scores = torch.matmul(
            query[:, query_head : query_head + 1],
            key[:, kv_head : kv_head + 1, :length].transpose(2, 3),
        ) * scale
        if mask is not None:
            mask_head = 0 if mask.shape[1] == 1 else query_head
            scores = scores + mask[:, mask_head : mask_head + 1, :, :length]
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
        outputs.append(
            torch.matmul(probabilities, value[:, kv_head : kv_head + 1, :length])
        )
    return torch.cat(outputs, dim=1)


@pytest.mark.parametrize("batch,query_heads,kv_heads,head_dim", [(1, 9, 3, 64), (2, 4, 2, 8)])
@pytest.mark.parametrize("capacity,valid_length", [(1, 1), (17, 11), (129, 128)])
def test_reference_maps_heads_and_respects_valid_length(
    batch: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    capacity: int,
    valid_length: int,
) -> None:
    torch.manual_seed(1000 + capacity)
    query = torch.randn(batch, query_heads, 1, head_dim)
    key = torch.randn(batch, kv_heads, capacity, head_dim)
    value = torch.randn_like(key)
    mask = torch.randn(batch, 1, 1, capacity) * 0.1
    length = torch.tensor(valid_length, dtype=torch.int64)

    actual = gqa_decode_attention(query, key, value, mask, head_dim**-0.5, length)
    expected = _unexpanded_oracle(
        query, key, value, mask, head_dim**-0.5, valid_length
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=5e-7)


def test_reference_mask_changes_only_the_selected_positions() -> None:
    query = torch.ones((1, 2, 1, 2))
    key = torch.zeros((1, 1, 3, 2))
    value = torch.tensor([[[[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]]]])
    mask = torch.tensor([[[[0.0, torch.finfo(torch.float32).min, torch.finfo(torch.float32).min]]]])

    actual = gqa_decode_attention(query, key, value, mask, 1.0)

    torch.testing.assert_close(actual, value[:, :, :1].expand(1, 2, 1, 2))

