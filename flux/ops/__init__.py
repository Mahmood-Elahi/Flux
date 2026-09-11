"""Custom operator interfaces for Flux."""

from flux.ops.attention_score_softmax import attention_score_softmax
from flux.ops.native_attention_score_softmax import (
    attention_score_softmax_native,
    native_attention_score_softmax_is_available,
)
from flux.ops.native_rmsnorm import (
    native_residual_rmsnorm_is_available,
    native_rmsnorm_is_available,
    native_rmsnorm_load_error,
    residual_rmsnorm_native,
    rms_norm_native,
)
from flux.ops.native_softmax import (
    native_softmax_is_available,
    native_softmax_load_error,
    softmax_native,
)
from flux.ops.native_rope import (
    native_rope_is_available,
    native_rope_load_error,
    rope_native,
)
from flux.ops.residual_rmsnorm import residual_rmsnorm
from flux.ops.rmsnorm import rms_norm
from flux.ops.rope import rope
from flux.ops.softmax import softmax

__all__ = [
    "attention_score_softmax",
    "attention_score_softmax_native",
    "native_attention_score_softmax_is_available",
    "native_residual_rmsnorm_is_available",
    "native_rmsnorm_is_available",
    "native_rmsnorm_load_error",
    "native_rope_is_available",
    "native_rope_load_error",
    "native_softmax_is_available",
    "native_softmax_load_error",
    "residual_rmsnorm",
    "residual_rmsnorm_native",
    "rms_norm",
    "rms_norm_native",
    "rope",
    "rope_native",
    "softmax",
    "softmax_native",
]
