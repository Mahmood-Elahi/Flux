"""Python entry point and loader for the native FP32 RMSNorm operator."""

from __future__ import annotations

from pathlib import Path

import torch


_NATIVE_LIBRARY: Path | None = None
_NATIVE_LOAD_ERROR: OSError | None = None
_FAKE_REGISTERED = False


def _operator_is_registered() -> bool:
    try:
        torch.ops.flux.rmsnorm.default
    except AttributeError:
        return False
    return True


def _library_candidates() -> list[Path]:
    package_dir = Path(__file__).resolve().parents[1]
    return sorted((*package_dir.glob("_C*.pyd"), *package_dir.glob("_C*.so")))


def _register_fake() -> None:
    global _FAKE_REGISTERED
    if _FAKE_REGISTERED:
        return

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

    _FAKE_REGISTERED = True


def _try_load_native_library() -> None:
    global _NATIVE_LIBRARY, _NATIVE_LOAD_ERROR
    if _operator_is_registered():
        _register_fake()
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
    _register_fake()


def native_rmsnorm_is_available() -> bool:
    """Return whether the compiled ``flux::rmsnorm`` operator is loaded."""
    _try_load_native_library()
    return _operator_is_registered()


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
    if not _operator_is_registered():
        message = (
            "Flux native RMSNorm is not built. Build it with "
            "FLUX_BUILD_NATIVE=1 and `python setup.py build_ext --inplace`."
        )
        if _NATIVE_LOAD_ERROR is not None:
            raise RuntimeError(message) from _NATIVE_LOAD_ERROR
        raise RuntimeError(message)
    return torch.ops.flux.rmsnorm(input, weight, epsilon)


_try_load_native_library()


__all__ = [
    "native_rmsnorm_is_available",
    "native_rmsnorm_load_error",
    "rms_norm_native",
]
