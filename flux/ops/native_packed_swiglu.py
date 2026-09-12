"""Python entry point for the native FP32 packed SwiGLU operator."""

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
    if _operator_is_registered("packed_swiglu") and not _FAKE_REGISTERED:

        @torch.library.register_fake("flux::packed_swiglu")
        def _packed_swiglu_fake(packed: torch.Tensor) -> torch.Tensor:
            torch._check(
                packed.dim() >= 1,
                lambda: "packed must have at least one dimension",
            )
            torch._check(
                packed.shape[-1] > 0,
                lambda: "packed final dimension must be non-empty",
            )
            torch._check(
                packed.shape[-1] % 2 == 0,
                lambda: "packed final dimension must be even",
            )
            torch._check(
                packed.numel() > 0,
                lambda: "packed must contain at least one row",
            )
            torch._check(
                packed.dtype == torch.float32,
                lambda: "packed must be float32",
            )
            torch._check(
                packed.is_contiguous(),
                lambda: "packed must be contiguous",
            )
            torch._check(
                not torch.is_grad_enabled() or not packed.requires_grad,
                lambda: "flux::packed_swiglu is inference-only",
            )
            output_shape = (*packed.shape[:-1], packed.shape[-1] // 2)
            return torch.empty(
                output_shape,
                dtype=packed.dtype,
                device=packed.device,
            )

        _FAKE_REGISTERED = True

    if _operator_is_registered("packed_swiglu_out") and not _OUT_FAKE_REGISTERED:

        @torch.library.register_fake("flux::packed_swiglu_out")
        def _packed_swiglu_out_fake(
            packed: torch.Tensor, output: torch.Tensor
        ) -> torch.Tensor:
            del packed
            return output

        _OUT_FAKE_REGISTERED = True


def _try_load_native_packed_swiglu() -> None:
    _try_load_native_library()
    _register_fake()


def native_packed_swiglu_is_available() -> bool:
    """Return whether the compiled ``flux::packed_swiglu`` op is loaded."""
    _try_load_native_packed_swiglu()
    return _operator_is_registered("packed_swiglu")


def packed_swiglu_native(packed: torch.Tensor) -> torch.Tensor:
    """Apply inference-only FP32 SwiGLU to a contiguous packed tensor.

    No implicit contiguous copy is made. Passing a strided tensor is an error.
    """
    _try_load_native_packed_swiglu()
    if not _operator_is_registered("packed_swiglu"):
        message = (
            "Flux native packed SwiGLU is not built. Build it with "
            "FLUX_BUILD_NATIVE=1 and `python setup.py build_ext --inplace`."
        )
        error = native_rmsnorm_load_error()
        if error is not None:
            raise RuntimeError(message) from error
        raise RuntimeError(message)
    return torch.ops.flux.packed_swiglu(packed)


def packed_swiglu_native_out(
    packed: torch.Tensor, output: torch.Tensor
) -> torch.Tensor:
    """Write packed SwiGLU into a validated CUDA output."""
    _try_load_native_packed_swiglu()
    if not _operator_is_registered("packed_swiglu_out"):
        raise RuntimeError("Flux native packed SwiGLU out variant is not built")
    return torch.ops.flux.packed_swiglu_out(packed, output)


_try_load_native_packed_swiglu()


__all__ = [
    "native_packed_swiglu_is_available",
    "packed_swiglu_native",
    "packed_swiglu_native_out",
]
