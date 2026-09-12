"""FP32 PyTorch correctness oracle for packed SwiGLU."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def packed_swiglu(packed: torch.Tensor) -> torch.Tensor:
    """Apply SwiGLU to ``[gate; up]`` packed along the final dimension."""
    if packed.ndim == 0:
        raise ValueError("packed must have at least one dimension")
    if packed.shape[-1] == 0 or packed.shape[-1] % 2 != 0:
        raise ValueError("packed final dimension must be non-empty and even")
    if packed.dtype != torch.float32:
        raise TypeError("packed must be a float32 tensor")

    gate, up = packed.chunk(2, dim=-1)
    return F.silu(gate) * up


__all__ = ["packed_swiglu"]
