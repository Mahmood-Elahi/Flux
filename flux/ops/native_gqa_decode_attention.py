"""Python entry point for native FP32 one-token GQA decode attention."""

from __future__ import annotations

import torch

from flux.ops.native_rmsnorm import (
    _operator_is_registered,
    _try_load_native_library,
    native_rmsnorm_load_error,
)


_FAKE_REGISTERED = False


def _register_fake() -> None:
    global _FAKE_REGISTERED
    if _operator_is_registered("gqa_decode_attention") and not _FAKE_REGISTERED:

        @torch.library.register_fake("flux::gqa_decode_attention")
        def _gqa_decode_attention_fake(
            query: torch.Tensor,
            key_cache: torch.Tensor,
            value_cache: torch.Tensor,
            additive_attention_mask: torch.Tensor | None,
            scale: float,
            cache_length: torch.Tensor | None = None,
        ) -> torch.Tensor:
            torch._check(query.dim() == 4, lambda: "query must be rank four")
            torch._check(key_cache.dim() == 4, lambda: "key_cache must be rank four")
            torch._check(value_cache.dim() == 4, lambda: "value_cache must be rank four")
            torch._check(query.shape[2] == 1, lambda: "query length must be one")
            torch._check(query.dtype == torch.float32, lambda: "query must be float32")
            torch._check(key_cache.dtype == torch.float32, lambda: "key_cache must be float32")
            torch._check(value_cache.dtype == torch.float32, lambda: "value_cache must be float32")
            torch._check(query.device == key_cache.device, lambda: "tensor devices must match")
            torch._check(query.device == value_cache.device, lambda: "tensor devices must match")
            for key_size, value_size in zip(key_cache.shape, value_cache.shape, strict=True):
                torch._check(key_size == value_size, lambda: "cache shapes must match")
            torch._check(query.shape[0] == key_cache.shape[0], lambda: "cache batch must match query")
            torch._check(query.shape[3] == key_cache.shape[3], lambda: "head dimensions must match")
            torch._check(key_cache.shape[1] > 0, lambda: "KV head count must be positive")
            torch._check(
                query.shape[1] % key_cache.shape[1] == 0,
                lambda: "query head count must be divisible by KV head count",
            )
            if additive_attention_mask is not None:
                torch._check(additive_attention_mask.dim() == 4, lambda: "mask must be rank four")
                torch._check(additive_attention_mask.dtype == torch.float32, lambda: "mask must be float32")
                torch._check(additive_attention_mask.device == query.device, lambda: "mask device must match")
                target = (query.shape[0], query.shape[1], 1, key_cache.shape[2])
                for mask_size, target_size in zip(additive_attention_mask.shape, target, strict=True):
                    torch._check(mask_size == 1 or mask_size == target_size, lambda: "mask is not broadcastable")
            if cache_length is not None:
                torch._check(cache_length.numel() == 1, lambda: "cache_length must be scalar")
                torch._check(cache_length.dtype == torch.int64, lambda: "cache_length must be int64")
                torch._check(cache_length.device == query.device, lambda: "cache_length device must match")
            torch._check(
                not torch.is_grad_enabled()
                or not any(t.requires_grad for t in (query, key_cache, value_cache))
                and (additive_attention_mask is None or not additive_attention_mask.requires_grad),
                lambda: "flux::gqa_decode_attention is inference-only",
            )
            return torch.empty_like(query, memory_format=torch.contiguous_format)

        _FAKE_REGISTERED = True


def _try_load_native_gqa_decode_attention() -> None:
    _try_load_native_library()
    _register_fake()


def native_gqa_decode_attention_is_available() -> bool:
    """Return whether the compiled GQA decode operator is loaded."""
    _try_load_native_gqa_decode_attention()
    return _operator_is_registered("gqa_decode_attention")


def gqa_decode_attention_native(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    additive_attention_mask: torch.Tensor | None,
    scale: float,
    cache_length: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run fused one-token attention directly over unexpanded FP32 K/V."""
    _try_load_native_gqa_decode_attention()
    if not _operator_is_registered("gqa_decode_attention"):
        message = (
            "Flux native GQA decode attention is not built. Build it with "
            "FLUX_BUILD_NATIVE=1 and `python setup.py build_ext --inplace`."
        )
        error = native_rmsnorm_load_error()
        if error is not None:
            raise RuntimeError(message) from error
        raise RuntimeError(message)
    return torch.ops.flux.gqa_decode_attention(
        query,
        key_cache,
        value_cache,
        additive_attention_mask,
        scale,
        cache_length,
    )


_try_load_native_gqa_decode_attention()


__all__ = [
    "gqa_decode_attention_native",
    "native_gqa_decode_attention_is_available",
]
