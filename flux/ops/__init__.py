"""Custom operator interfaces for Flux."""

from flux.ops.native_rmsnorm import (
    native_rmsnorm_is_available,
    native_rmsnorm_load_error,
    rms_norm_native,
)
from flux.ops.rmsnorm import rms_norm

__all__ = [
    "native_rmsnorm_is_available",
    "native_rmsnorm_load_error",
    "rms_norm",
    "rms_norm_native",
]
