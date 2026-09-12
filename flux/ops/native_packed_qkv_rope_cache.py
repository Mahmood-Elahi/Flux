"""Python entry point for fused one-token packed-QKV post-processing."""

from __future__ import annotations

import torch

from flux.ops.native_rmsnorm import (
    _operator_is_registered,
    _try_load_native_library,
    native_rmsnorm_load_error,
)


_FAKE_REGISTERED = False
_OUT_FAKE_REGISTERED = False


def _register_fake() -> None:
    global _FAKE_REGISTERED, _OUT_FAKE_REGISTERED
    if _operator_is_registered("packed_qkv_rope_cache") and not _FAKE_REGISTERED:

        @torch.library.register_fake("flux::packed_qkv_rope_cache")
        def _packed_qkv_rope_cache_fake(
            packed_qkv: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            key_cache: torch.Tensor,
            value_cache: torch.Tensor,
            cache_length: torch.Tensor,
        ) -> torch.Tensor:
            torch._check(
                packed_qkv.shape == (1, 1, 960),
                lambda: "packed_qkv must have shape [1, 1, 960]",
            )
            torch._check(
                cos.shape == (1, 1, 64),
                lambda: "cos must have shape [1, 1, 64]",
            )
            torch._check(
                sin.shape == cos.shape,
                lambda: "cos and sin shapes must match",
            )
            torch._check(
                key_cache.dim() == 4,
                lambda: "key_cache must be rank four",
            )
            torch._check(
                key_cache.shape[0] == 1
                and key_cache.shape[1] == 3
                and key_cache.shape[2] > 0
                and key_cache.shape[3] == 64,
                lambda: "key_cache must have shape [1, 3, capacity, 64]",
            )
            for key_size, value_size in zip(
                key_cache.shape, value_cache.shape, strict=True
            ):
                torch._check(
                    key_size == value_size,
                    lambda: "K/V cache shapes must match",
                )
            torch._check(
                cache_length.dim() == 0 and cache_length.numel() == 1,
                lambda: "cache_length must be scalar",
            )
            torch._check(
                cache_length.dtype == torch.int64,
                lambda: "cache_length must be int64",
            )
            for tensor, name in (
                (packed_qkv, "packed_qkv"),
                (cos, "cos"),
                (sin, "sin"),
                (key_cache, "key_cache"),
                (value_cache, "value_cache"),
            ):
                torch._check(
                    tensor.dtype == torch.float32,
                    lambda name=name: f"{name} must be float32",
                )
                torch._check(
                    tensor.device == packed_qkv.device,
                    lambda: "tensor devices must match",
                )
            torch._check(
                cache_length.device == packed_qkv.device,
                lambda: "tensor devices must match",
            )
            torch._check(
                not torch.is_grad_enabled()
                or not any(
                    tensor.requires_grad
                    for tensor in (
                        packed_qkv,
                        cos,
                        sin,
                        key_cache,
                        value_cache,
                        cache_length,
                    )
                ),
                lambda: "flux::packed_qkv_rope_cache is inference-only",
            )
            return torch.empty(
                (1, 9, 1, 64),
                dtype=packed_qkv.dtype,
                device=packed_qkv.device,
            )

        _FAKE_REGISTERED = True

    if (
        _operator_is_registered("packed_qkv_rope_cache_out")
        and not _OUT_FAKE_REGISTERED
    ):

        @torch.library.register_fake("flux::packed_qkv_rope_cache_out")
        def _packed_qkv_rope_cache_out_fake(
            packed_qkv: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            key_cache: torch.Tensor,
            value_cache: torch.Tensor,
            cache_length: torch.Tensor,
            query_output: torch.Tensor,
        ) -> torch.Tensor:
            del packed_qkv, cos, sin, key_cache, value_cache, cache_length
            return query_output

        _OUT_FAKE_REGISTERED = True


def _try_load_native_packed_qkv_rope_cache() -> None:
    _try_load_native_library()
    _register_fake()


def native_packed_qkv_rope_cache_is_available() -> bool:
    """Return whether the compiled fused post-QKV operator is loaded."""
    _try_load_native_packed_qkv_rope_cache()
    return _operator_is_registered("packed_qkv_rope_cache")


def native_packed_qkv_rope_cache_load_error() -> OSError | None:
    """Return a native-library loading error, if one occurred."""
    _try_load_native_packed_qkv_rope_cache()
    return native_rmsnorm_load_error()


def packed_qkv_rope_cache_native(
    packed_qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_length: torch.Tensor,
) -> torch.Tensor:
    """Rotate Q/K, store K/V in StaticCache, and return compact Q.

    The specialized CUDA path consumes SmolLM2's one-token FP32 packed layout
    ``Q[576] | K[192] | V[192]``. It mutates cache storage and its device-side
    logical length in place and allocates only the ``[1, 9, 1, 64]`` Q result.
    """
    _try_load_native_packed_qkv_rope_cache()
    if not _operator_is_registered("packed_qkv_rope_cache"):
        message = (
            "Flux native packed-QKV RoPE/cache operator is not built. Build "
            "it with FLUX_BUILD_NATIVE=1 and `python setup.py build_ext --inplace`."
        )
        error = native_packed_qkv_rope_cache_load_error()
        if error is not None:
            raise RuntimeError(message) from error
        raise RuntimeError(message)
    return torch.ops.flux.packed_qkv_rope_cache(
        packed_qkv,
        cos,
        sin,
        key_cache,
        value_cache,
        cache_length,
    )


def packed_qkv_rope_cache_native_out(
    packed_qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_length: torch.Tensor,
    query_output: torch.Tensor,
) -> torch.Tensor:
    """Write the compact rotated query into a stable CUDA output."""
    _try_load_native_packed_qkv_rope_cache()
    if not _operator_is_registered("packed_qkv_rope_cache_out"):
        raise RuntimeError("Flux native packed-QKV RoPE/cache out variant is not built")
    return torch.ops.flux.packed_qkv_rope_cache_out(
        packed_qkv,
        cos,
        sin,
        key_cache,
        value_cache,
        cache_length,
        query_output,
    )


_try_load_native_packed_qkv_rope_cache()


__all__ = [
    "native_packed_qkv_rope_cache_is_available",
    "native_packed_qkv_rope_cache_load_error",
    "packed_qkv_rope_cache_native",
    "packed_qkv_rope_cache_native_out",
]
