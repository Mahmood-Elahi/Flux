"""Reference and optional Flux-integrated SmolLM2 model utilities."""

from flux.model.smollm2_cuda_graph import (
    CUDAGraphMemory,
    CUDAGraphSetupTiming,
    FluxCUDAGraphDecode,
    cuda_graph_greedy_generate,
)
from flux.model.smollm2_flux import (
    FLUX_PACKED_MLP_CATEGORY,
    FLUX_PACKED_QKV_CATEGORY,
    FLUX_PACKED_SWIGLU_CATEGORY,
    FluxPackedLlamaMLP,
    enable_flux_ops,
    flux_operator_counts,
)

__all__ = [
    "FluxCUDAGraphDecode",
    "CUDAGraphMemory",
    "CUDAGraphSetupTiming",
    "FLUX_PACKED_MLP_CATEGORY",
    "FLUX_PACKED_QKV_CATEGORY",
    "FLUX_PACKED_SWIGLU_CATEGORY",
    "FluxPackedLlamaMLP",
    "cuda_graph_greedy_generate",
    "enable_flux_ops",
    "flux_operator_counts",
]
