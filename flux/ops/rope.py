"""FP32 PyTorch reference for SmolLM2 rotary positional embeddings."""

from __future__ import annotations

import torch


def rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate ``query`` and ``key`` using Hugging Face Llama semantics.

    Query and key use ``[batch, heads, sequence, head_dim]`` layout.  ``cos``
    and ``sin`` use ``[rope_batch, sequence, head_dim]`` where ``rope_batch``
    is either one or the attention batch size.  Position ids and any RoPE
    scaling are already represented by ``cos`` and ``sin``.
    """
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query and key must be rank four")
    if cos.ndim != 3 or sin.ndim != 3:
        raise ValueError("cos and sin must be rank three")
    if query.dtype != torch.float32 or key.dtype != torch.float32:
        raise TypeError("query and key must be float32 tensors")
    if cos.dtype != torch.float32 or sin.dtype != torch.float32:
        raise TypeError("cos and sin must be float32 tensors")
    if not (query.device == key.device == cos.device == sin.device):
        raise ValueError("all tensors must be on the same device")
    batch, _, sequence, head_dim = query.shape
    if head_dim == 0 or head_dim % 2:
        raise ValueError("head_dim must be positive and even")
    if key.shape[0] != batch or key.shape[2:] != (sequence, head_dim):
        raise ValueError("query and key batch, sequence, and head dimensions must match")
    if cos.shape != sin.shape:
        raise ValueError("cos and sin shapes must match")
    if cos.shape[0] not in (1, batch) or cos.shape[1:] != (sequence, head_dim):
        raise ValueError("cos and sin shapes are not broadcastable to query and key")

    half = head_dim // 2
    query_rotated = torch.cat((-query[..., half:], query[..., :half]), dim=-1)
    key_rotated = torch.cat((-key[..., half:], key[..., :half]), dim=-1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return query * cos + query_rotated * sin, key * cos + key_rotated * sin


__all__ = ["rope"]
