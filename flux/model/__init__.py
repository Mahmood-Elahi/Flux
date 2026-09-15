"""Reference and optional Flux-integrated SmolLM2 model utilities."""

from flux.model.smollm2_flux import (
    FINAL_FLUX_OPERATOR_CATEGORIES,
    FLUX_PACKED_MLP_CATEGORY,
    FLUX_PACKED_QKV_CATEGORY,
    FLUX_PACKED_SWIGLU_CATEGORY,
    FLUX_FUSED_GATE_UP_SWIGLU_CATEGORY,
    FluxPackedLlamaMLP,
    enable_flux_ops,
    flux_operator_counts,
)

__all__ = [
    "FINAL_FLUX_OPERATOR_CATEGORIES",
    "FLUX_PACKED_MLP_CATEGORY",
    "FLUX_PACKED_QKV_CATEGORY",
    "FLUX_PACKED_SWIGLU_CATEGORY",
    "FLUX_FUSED_GATE_UP_SWIGLU_CATEGORY",
    "FluxPackedLlamaMLP",
    "enable_flux_ops",
    "flux_operator_counts",
]
