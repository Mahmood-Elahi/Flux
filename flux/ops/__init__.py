"""Custom operator interfaces for Flux."""

from flux.ops.native_rmsnorm import (
    native_residual_rmsnorm_is_available,
    native_rmsnorm_is_available,
    native_rmsnorm_load_error,
    residual_rmsnorm_native,
    rms_norm_native,
)
from flux.ops.residual_rmsnorm import residual_rmsnorm
from flux.ops.rmsnorm import rms_norm

__all__ = [
    "native_residual_rmsnorm_is_available",
    "native_rmsnorm_is_available",
    "native_rmsnorm_load_error",
    "residual_rmsnorm",
    "residual_rmsnorm_native",
    "rms_norm",
    "rms_norm_native",
]
