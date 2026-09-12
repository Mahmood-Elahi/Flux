"""Python entry point and loader for the native FP32 RMSNorm operator."""

from __future__ import annotations

from pathlib import Path

import torch


_NATIVE_LIBRARY: Path | None = None
_NATIVE_LOAD_ERROR: OSError | None = None
_RMSNORM_FAKE_REGISTERED = False
_RESIDUAL_RMSNORM_FAKE_REGISTERED = False
_RMSNORM_OUT_FAKE_REGISTERED = False
_RESIDUAL_RMSNORM_OUT_FAKE_REGISTERED = False


def _operator_is_registered(name: str) -> bool:
    try:
        getattr(torch.ops.flux, name).default
    except AttributeError:
        return False
    return True


def _library_candidates() -> list[Path]:
    package_dir = Path(__file__).resolve().parents[1]
    return sorted((*package_dir.glob("_C*.pyd"), *package_dir.glob("_C*.so")))


def _register_fakes() -> None:
    global _RMSNORM_FAKE_REGISTERED, _RESIDUAL_RMSNORM_FAKE_REGISTERED
    global _RMSNORM_OUT_FAKE_REGISTERED, _RESIDUAL_RMSNORM_OUT_FAKE_REGISTERED

    if _operator_is_registered("rmsnorm") and not _RMSNORM_FAKE_REGISTERED:

        @torch.library.register_fake("flux::rmsnorm")
        def _rmsnorm_fake(
            input: torch.Tensor, weight: torch.Tensor, epsilon: float
        ) -> torch.Tensor:
            torch._check(input.dim() >= 1, lambda: "input must have at least one dimension")
            torch._check(input.shape[-1] > 0, lambda: "input final dimension must be non-empty")
            torch._check(input.numel() > 0, lambda: "input must contain at least one row")
            torch._check(input.dtype == torch.float32, lambda: "input must be float32")
            torch._check(weight.dtype == torch.float32, lambda: "weight must be float32")
            torch._check(weight.dim() == 1, lambda: "weight must be one-dimensional")
            torch._check(
                weight.shape[0] == input.shape[-1],
                lambda: "weight length must match the input final dimension",
            )
            torch._check(input.device == weight.device, lambda: "tensor devices must match")
            torch._check(epsilon > 0.0, lambda: "epsilon must be positive")
            torch._check(
                not torch.is_grad_enabled()
                or (not input.requires_grad and not weight.requires_grad),
                lambda: "flux::rmsnorm is inference-only",
            )
            return torch.empty_like(input, memory_format=torch.contiguous_format)

        _RMSNORM_FAKE_REGISTERED = True

    if _operator_is_registered("rmsnorm_out") and not _RMSNORM_OUT_FAKE_REGISTERED:

        @torch.library.register_fake("flux::rmsnorm_out")
        def _rmsnorm_out_fake(
            input: torch.Tensor,
            weight: torch.Tensor,
            epsilon: float,
            output: torch.Tensor,
        ) -> torch.Tensor:
            del input, weight, epsilon
            return output

        _RMSNORM_OUT_FAKE_REGISTERED = True

    if (
        _operator_is_registered("residual_rmsnorm")
        and not _RESIDUAL_RMSNORM_FAKE_REGISTERED
    ):

        @torch.library.register_fake("flux::residual_rmsnorm")
        def _residual_rmsnorm_fake(
            hidden: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
            epsilon: float,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            torch._check(hidden.dim() >= 1, lambda: "hidden must have at least one dimension")
            torch._check(hidden.shape[-1] > 0, lambda: "hidden final dimension must be non-empty")
            torch._check(hidden.numel() > 0, lambda: "hidden must contain at least one row")
            torch._check(hidden.dim() == residual.dim(), lambda: "tensor ranks must match")
            for hidden_size, residual_size in zip(hidden.shape, residual.shape, strict=True):
                torch._check(
                    hidden_size == residual_size,
                    lambda: "hidden and residual shapes must match",
                )
            torch._check(hidden.dtype == torch.float32, lambda: "hidden must be float32")
            torch._check(residual.dtype == torch.float32, lambda: "residual must be float32")
            torch._check(weight.dtype == torch.float32, lambda: "weight must be float32")
            torch._check(weight.dim() == 1, lambda: "weight must be one-dimensional")
            torch._check(
                weight.shape[0] == hidden.shape[-1],
                lambda: "weight length must match the hidden final dimension",
            )
            torch._check(
                hidden.device == residual.device and hidden.device == weight.device,
                lambda: "tensor devices must match",
            )
            torch._check(
                epsilon >= 0.0 and epsilon <= torch.finfo(torch.float32).max,
                lambda: "epsilon must be a non-negative finite FP32 value",
            )
            torch._check(
                not torch.is_grad_enabled()
                or (
                    not hidden.requires_grad
                    and not residual.requires_grad
                    and not weight.requires_grad
                ),
                lambda: "flux::residual_rmsnorm is inference-only",
            )
            return (
                torch.empty_like(hidden, memory_format=torch.contiguous_format),
                torch.empty_like(hidden, memory_format=torch.contiguous_format),
            )

        _RESIDUAL_RMSNORM_FAKE_REGISTERED = True

    if (
        _operator_is_registered("residual_rmsnorm_out")
        and not _RESIDUAL_RMSNORM_OUT_FAKE_REGISTERED
    ):

        @torch.library.register_fake("flux::residual_rmsnorm_out")
        def _residual_rmsnorm_out_fake(
            hidden: torch.Tensor,
            residual: torch.Tensor,
            weight: torch.Tensor,
            epsilon: float,
            norm_out: torch.Tensor,
            residual_out: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del hidden, residual, weight, epsilon
            return norm_out, residual_out

        _RESIDUAL_RMSNORM_OUT_FAKE_REGISTERED = True


def _try_load_native_library() -> None:
    global _NATIVE_LIBRARY, _NATIVE_LOAD_ERROR
    if _operator_is_registered("rmsnorm"):
        _register_fakes()
        return
    candidates = _library_candidates()
    if not candidates:
        return
    try:
        torch.ops.load_library(str(candidates[0]))
    except OSError as error:
        _NATIVE_LOAD_ERROR = error
        return
    _NATIVE_LIBRARY = candidates[0]
    _register_fakes()


def native_rmsnorm_is_available() -> bool:
    """Return whether the compiled ``flux::rmsnorm`` operator is loaded."""
    _try_load_native_library()
    return _operator_is_registered("rmsnorm")


def native_rmsnorm_load_error() -> OSError | None:
    """Return a native library loading error, if a built library failed to load."""
    _try_load_native_library()
    return _NATIVE_LOAD_ERROR


def rms_norm_native(
    input: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """Apply the inference-only native FP32 RMSNorm custom operator.

    The C++ wrappers make input and weight contiguous when necessary. They do
    not perform dtype conversion and reject tensors that require gradients.
    """
    _try_load_native_library()
    if not _operator_is_registered("rmsnorm"):
        message = (
            "Flux native RMSNorm is not built. Build it with "
            "FLUX_BUILD_NATIVE=1 and `python setup.py build_ext --inplace`."
        )
        if _NATIVE_LOAD_ERROR is not None:
            raise RuntimeError(message) from _NATIVE_LOAD_ERROR
        raise RuntimeError(message)
    return torch.ops.flux.rmsnorm(input, weight, epsilon)


def rms_norm_native_out(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    output: torch.Tensor,
) -> torch.Tensor:
    """Write inference-only FP32 RMSNorm into a validated CUDA output."""
    _try_load_native_library()
    if not _operator_is_registered("rmsnorm_out"):
        raise RuntimeError("Flux native RMSNorm out variant is not built")
    return torch.ops.flux.rmsnorm_out(input, weight, epsilon, output)


def native_residual_rmsnorm_is_available() -> bool:
    """Return whether the compiled ``flux::residual_rmsnorm`` operator is loaded."""
    _try_load_native_library()
    return _operator_is_registered("residual_rmsnorm")


def residual_rmsnorm_native(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(norm_out, residual_out)`` from fused residual RMSNorm.

    The C++ wrappers make all inputs contiguous when necessary, allocate two
    distinct contiguous outputs, and do not perform dtype conversion.
    """
    _try_load_native_library()
    if not _operator_is_registered("residual_rmsnorm"):
        message = (
            "Flux native residual RMSNorm is not built. Build it with "
            "FLUX_BUILD_NATIVE=1 and `python setup.py build_ext --inplace`."
        )
        if _NATIVE_LOAD_ERROR is not None:
            raise RuntimeError(message) from _NATIVE_LOAD_ERROR
        raise RuntimeError(message)
    return torch.ops.flux.residual_rmsnorm(hidden, residual, weight, epsilon)


def residual_rmsnorm_native_out(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    norm_out: torch.Tensor,
    residual_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write both fused residual-RMSNorm results into CUDA outputs."""
    _try_load_native_library()
    if not _operator_is_registered("residual_rmsnorm_out"):
        raise RuntimeError("Flux native residual RMSNorm out variant is not built")
    return torch.ops.flux.residual_rmsnorm_out(
        hidden, residual, weight, epsilon, norm_out, residual_out
    )


_try_load_native_library()


__all__ = [
    "native_residual_rmsnorm_is_available",
    "native_rmsnorm_is_available",
    "native_rmsnorm_load_error",
    "residual_rmsnorm_native",
    "residual_rmsnorm_native_out",
    "rms_norm_native",
    "rms_norm_native_out",
]
