"""Tests for the explicit PyTorch packed SwiGLU reference."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from flux.ops.packed_swiglu import packed_swiglu


def test_matches_explicit_silu_multiply() -> None:
    packed = torch.linspace(-4.0, 4.0, 2 * 3 * 10).reshape(2, 3, 10)
    gate, up = packed.chunk(2, dim=-1)

    actual = packed_swiglu(packed)

    torch.testing.assert_close(actual, F.silu(gate) * up)


@pytest.mark.parametrize(
    ("packed", "error", "message"),
    [
        (torch.tensor(1.0), ValueError, "at least one dimension"),
        (torch.empty(2, 0), ValueError, "non-empty and even"),
        (torch.ones(2, 3), ValueError, "non-empty and even"),
        (torch.ones(2, 4, dtype=torch.float64), TypeError, "float32"),
    ],
)
def test_rejects_invalid_inputs(
    packed: torch.Tensor, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        packed_swiglu(packed)
