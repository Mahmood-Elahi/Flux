"""FP32 correctness reference for one-token grouped-query decode attention."""

from __future__ import annotations

import math

import torch


def gqa_decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    additive_attention_mask: torch.Tensor | None,
    scale: float,
    cache_length: torch.Tensor | None = None,
) -> torch.Tensor:
    """Attend one query token to an unexpanded grouped-query KV cache.

    ``cache_length=None`` means every cache position is valid.  A scalar
    ``cache_length`` is primarily useful for fixed-capacity cache storage.
    This explicit reference groups query heads by KV head and never calls
    ``repeat_kv``.
    """
    if query.ndim != 4 or key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("query, key_cache, and value_cache must be rank four")
    if query.shape[2] != 1:
        raise ValueError("query length must be one")
    if query.dtype != torch.float32:
        raise TypeError("query must be a float32 tensor")
    if key_cache.dtype != torch.float32 or value_cache.dtype != torch.float32:
        raise TypeError("key_cache and value_cache must be float32 tensors")
    if query.device != key_cache.device or query.device != value_cache.device:
        raise ValueError("query, key_cache, and value_cache must share a device")
    if key_cache.shape != value_cache.shape:
        raise ValueError("key_cache and value_cache shapes must match")
    batch, query_heads, _, head_dim = query.shape
    cache_batch, kv_heads, capacity, cache_head_dim = key_cache.shape
    if batch < 1 or query_heads < 1 or kv_heads < 1 or capacity < 1 or head_dim < 1:
        raise ValueError("all tensor dimensions must be non-empty")
    if cache_batch != batch or cache_head_dim != head_dim:
        raise ValueError("cache batch and head dimension must match query")
    if query_heads % kv_heads:
        raise ValueError("query head count must be divisible by KV head count")
    if not math.isfinite(scale) or abs(scale) > torch.finfo(torch.float32).max:
        raise ValueError("scale must be a finite FP32 value")

    if cache_length is None:
        valid_length = capacity
    else:
        if cache_length.numel() != 1 or cache_length.dtype != torch.int64:
            raise TypeError("cache_length must be a scalar torch.int64 tensor")
        if cache_length.device != query.device:
            raise ValueError("cache_length must be on the same device as query")
        valid_length = int(cache_length.item())
        if valid_length < 1 or valid_length > capacity:
            raise ValueError("cache_length must be within cache capacity")

    if additive_attention_mask is not None:
        if additive_attention_mask.ndim != 4:
            raise ValueError("additive_attention_mask must be rank four")
        if additive_attention_mask.dtype != torch.float32:
            raise TypeError("additive_attention_mask must be a float32 tensor")
        if additive_attention_mask.device != query.device:
            raise ValueError("additive_attention_mask must share the query device")
        target = (batch, query_heads, 1, capacity)
        if any(size not in (1, target_size) for size, target_size in zip(
            additive_attention_mask.shape, target, strict=True
        )):
            raise ValueError("additive_attention_mask is not broadcastable")

    groups = query_heads // kv_heads
    grouped_query = query[:, :, 0, :].reshape(batch, kv_heads, groups, head_dim)
    valid_keys = key_cache[:, :, :valid_length, :]
    valid_values = value_cache[:, :, :valid_length, :]
    scores = torch.einsum("bkgd,bkld->bkgl", grouped_query, valid_keys) * scale
    scores = scores.reshape(batch, query_heads, 1, valid_length)
    if additive_attention_mask is not None:
        scores = scores + additive_attention_mask[..., :valid_length]
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
    probabilities = probabilities.reshape(batch, kv_heads, groups, valid_length)
    output = torch.einsum("bkgl,bkld->bkgd", probabilities, valid_values)
    return output.reshape(batch, query_heads, 1, head_dim).contiguous()


__all__ = ["gqa_decode_attention"]
