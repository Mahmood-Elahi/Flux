"""Python entry point for the native inference-only FP32 softmax operator."""

from __future__ import annotations

import torch

from flux.ops.native_rmsnorm import (
    _operator_is_registered,
    _try_load_native_library,
    native_rmsnorm_load_error,
)


_SOFTMAX_FAKE_REGISTERED = False


def _register_softmax_fake() -> None:
    global _SOFTMAX_FAKE_REGISTERED

    if _operator_is_registered("softmax") and not _SOFTMAX_FAKE_REGISTERED:

        @torch.library.register_fake("flux::softmax")
        def _softmax_fake(input: torch.Tensor) -> torch.Tensor:
            torch._check(
                input.dim() >= 1,
                lambda: "input must have at least one dimension",
            )
            torch._check(
                input.shape[-1] > 0,
                lambda: "input final dimension must be non-empty",
            )
            torch._check(
                input.numel() > 0,
                lambda: "input must contain at least one row",
            )
            torch._check(
                input.dtype == torch.float32,
                lambda: "input must be float32",
            )
            torch._check(
                not torch.is_grad_enabled() or not input.requires_grad,
                lambda: "flux::softmax is inference-only",
            )
            return torch.empty_like(input, memory_format=torch.contiguous_format)

        _SOFTMAX_FAKE_REGISTERED = True


def _try_load_native_softmax() -> None:
    _try_load_native_library()
    _register_softmax_fake()


def native_softmax_is_available() -> bool:
    """Return whether the compiled ``flux::softmax`` operator is loaded."""
    _try_load_native_softmax()
    return _operator_is_registered("softmax")


def native_softmax_load_error() -> OSError | None:
    """Return a native library loading error, if the library failed to load."""
    _try_load_native_softmax()
    return native_rmsnorm_load_error()


def softmax_native(input: torch.Tensor) -> torch.Tensor:
    """Apply native inference-only FP32 softmax over the final dimension.

    The C++ wrappers make the input contiguous when necessary, allocate a new
    contiguous output, and do not perform dtype conversion.
    """
    _try_load_native_softmax()
    if not _operator_is_registered("softmax"):
        message = (
            "Flux native softmax is not built. Build it with "
            "FLUX_BUILD_NATIVE=1 and `python setup.py build_ext --inplace`."
        )
        error = native_rmsnorm_load_error()
        if error is not None:
            raise RuntimeError(message) from error
        raise RuntimeError(message)
    return torch.ops.flux.softmax(input)


_try_load_native_softmax()


__all__ = [
    "native_softmax_is_available",
    "native_softmax_load_error",
    "softmax_native",
]
