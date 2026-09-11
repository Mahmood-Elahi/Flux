"""FP32 correctness reference for fused attention score post-processing."""

from __future__ import annotations

import math

import torch


def attention_score_softmax(
    scores: torch.Tensor,
    additive_attention_mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Return ``softmax(scores * scale + mask, dim=-1)`` in FP32.

    ``scores`` and ``additive_attention_mask`` are rank-four tensors.  The
    mask follows ordinary broadcasting rules, which covers the SmolLM2
    ``[batch, 1, query, key]`` causal/padding mask without expanding it over
    attention heads.
    """
    if scores.ndim != 4:
        raise ValueError("scores must be rank four")
    if additive_attention_mask.ndim != 4:
        raise ValueError("additive_attention_mask must be rank four")
    if scores.numel() == 0 or scores.shape[-1] == 0:
        raise ValueError("scores dimensions must be non-empty")
    if scores.dtype != torch.float32:
        raise TypeError("scores must be a float32 tensor")
    if additive_attention_mask.dtype != torch.float32:
        raise TypeError("additive_attention_mask must be a float32 tensor")
    if scores.device != additive_attention_mask.device:
        raise ValueError("scores and additive_attention_mask must share a device")
    for mask_size, score_size in zip(
        additive_attention_mask.shape, scores.shape, strict=True
    ):
        if mask_size not in (1, score_size):
            raise ValueError("additive_attention_mask is not broadcastable to scores")
    if not math.isfinite(scale) or abs(scale) > torch.finfo(torch.float32).max:
        raise ValueError("scale must be a finite FP32 value")

    return torch.softmax(scores * scale + additive_attention_mask, dim=-1)


__all__ = ["attention_score_softmax"]
