"""Python entry point for the native FP32 RoPE custom operator."""

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
    if _operator_is_registered("rope") and not _FAKE_REGISTERED:

        @torch.library.register_fake("flux::rope")
        def _rope_fake(
            query: torch.Tensor,
            key: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            torch._check(query.dim() == 4, lambda: "query must be rank four")
            torch._check(key.dim() == 4, lambda: "key must be rank four")
            torch._check(cos.dim() == 3, lambda: "cos must be rank three")
            torch._check(sin.dim() == 3, lambda: "sin must be rank three")
            for tensor, name in (
                (query, "query"),
                (key, "key"),
                (cos, "cos"),
                (sin, "sin"),
            ):
                torch._check(
                    tensor.dtype == torch.float32,
                    lambda name=name: f"{name} must be float32",
                )
                torch._check(
                    tensor.device == query.device,
                    lambda: "tensor devices must match",
                )
            torch._check(query.shape[0] > 0, lambda: "batch must be non-empty")
            torch._check(query.shape[1] > 0, lambda: "query heads must be non-empty")
            torch._check(key.shape[1] > 0, lambda: "key heads must be non-empty")
            torch._check(query.shape[2] > 0, lambda: "sequence must be non-empty")
            torch._check(query.shape[3] > 0, lambda: "head_dim must be non-empty")
            torch._check(query.shape[3] % 2 == 0, lambda: "head_dim must be even")
            torch._check(key.shape[0] == query.shape[0], lambda: "batches must match")
            torch._check(
                key.shape[2] == query.shape[2], lambda: "sequences must match"
            )
            torch._check(
                key.shape[3] == query.shape[3], lambda: "head dimensions must match"
            )
            torch._check(
                cos.shape[0] == 1 or cos.shape[0] == query.shape[0],
                lambda: "cos batch is not broadcastable",
            )
            torch._check(cos.shape[1] == query.shape[2], lambda: "sequence mismatch")
            torch._check(
                cos.shape[2] == query.shape[3], lambda: "head dimension mismatch"
            )
            for cos_size, sin_size in zip(cos.shape, sin.shape, strict=True):
                torch._check(
                    cos_size == sin_size, lambda: "cos and sin shapes must match"
                )
            torch._check(
                not torch.is_grad_enabled()
                or not any(tensor.requires_grad for tensor in (query, key, cos, sin)),
                lambda: "flux::rope is inference-only",
            )
            return (
                torch.empty_like(query, memory_format=torch.contiguous_format),
                torch.empty_like(key, memory_format=torch.contiguous_format),
            )

        _FAKE_REGISTERED = True


def _try_load_native_rope() -> None:
    _try_load_native_library()
    _register_fake()


def native_rope_is_available() -> bool:
    """Return whether the compiled ``flux::rope`` operator is loaded."""
    _try_load_native_rope()
    return _operator_is_registered("rope")


def native_rope_load_error() -> OSError | None:
    """Return a native-library loading error, if one occurred."""
    _try_load_native_rope()
    return native_rmsnorm_load_error()


def rope_native(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply inference-only FP32 Llama RoPE to Q and K.

    Strided inputs are read directly by both native implementations; only the
    two contiguous result tensors are allocated.
    """
    _try_load_native_rope()
    if not _operator_is_registered("rope"):
        message = (
            "Flux native RoPE is not built. Build it with FLUX_BUILD_NATIVE=1 "
            "and `python setup.py build_ext --inplace`."
        )
        error = native_rope_load_error()
        if error is not None:
            raise RuntimeError(message) from error
        raise RuntimeError(message)
    return torch.ops.flux.rope(query, key, cos, sin)


_try_load_native_rope()


__all__ = [
    "native_rope_is_available",
    "native_rope_load_error",
    "rope_native",
]
