"""Explicit cuBLASLt FP32 one-token linear operator and tuning interface."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from flux.ops.native_rmsnorm import _operator_is_registered, _try_load_native_library


_FAKES_REGISTERED = False


@dataclass(frozen=True)
class CublasLtAlgorithm:
    """Configuration returned by the bounded cuBLASLt heuristic query."""

    index: int
    algorithm_id: int
    tile_id: int
    split_k: int
    reduction_scheme: int
    cta_swizzle: int
    custom_option: int
    stages_id: int
    workspace_bytes: int
    waves_count: float


def _register_fakes() -> None:
    global _FAKES_REGISTERED
    required = (
        "cublaslt_algorithm_info",
        "cublaslt_linear_out",
        "cublaslt_linear_config_out",
    )
    if _FAKES_REGISTERED or not all(
        _operator_is_registered(name) for name in required
    ):
        return

    @torch.library.register_fake("flux::cublaslt_linear_out")
    def _cublaslt_linear_out_fake(
        input: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        workspace: torch.Tensor,
        algorithm_index: int,
        max_workspace_bytes: int,
    ) -> torch.Tensor:
        del input, weight, workspace, algorithm_index, max_workspace_bytes
        return output

    @torch.library.register_fake("flux::cublaslt_linear_config_out")
    def _cublaslt_linear_config_out_fake(
        input: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        workspace: torch.Tensor,
        algorithm_id: int,
        tile_id: int,
        split_k: int,
        reduction_scheme: int,
        cta_swizzle: int,
        custom_option: int,
        stages_id: int,
    ) -> torch.Tensor:
        del (
            input,
            weight,
            workspace,
            algorithm_id,
            tile_id,
            split_k,
            reduction_scheme,
            cta_swizzle,
            custom_option,
            stages_id,
        )
        return output

    _FAKES_REGISTERED = True


def native_cublaslt_linear_is_available() -> bool:
    """Return whether the native cuBLASLt tuning and execution ops are loaded."""
    _try_load_native_library()
    _register_fakes()
    return all(
        _operator_is_registered(name)
        for name in (
            "cublaslt_algorithm_info",
            "cublaslt_linear_out",
            "cublaslt_linear_config_out",
        )
    )


def cublaslt_algorithms(
    input: torch.Tensor,
    weight: torch.Tensor,
    *,
    max_workspace_bytes: int,
    max_algorithms: int = 16,
) -> tuple[CublasLtAlgorithm, ...]:
    """Return a bounded heuristic list for the exact input/weight geometry."""
    _try_load_native_library()
    if not _operator_is_registered("cublaslt_algorithm_info"):
        raise RuntimeError("Flux native cuBLASLt benchmark operator is not built")
    rows = torch.ops.flux.cublaslt_algorithm_info(
        input, weight, max_workspace_bytes, max_algorithms
    ).tolist()
    return tuple(
        CublasLtAlgorithm(
            index=row[0],
            algorithm_id=row[1],
            tile_id=row[2],
            split_k=row[3],
            reduction_scheme=row[4],
            cta_swizzle=row[5],
            custom_option=row[6],
            stages_id=row[7],
            workspace_bytes=row[8],
            waves_count=row[9] / 1_000_000.0,
        )
        for row in rows
    )


def cublaslt_linear_out(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    workspace: torch.Tensor,
    *,
    algorithm_index: int,
    max_workspace_bytes: int,
) -> torch.Tensor:
    """Write ``input @ weight.T`` using one enumerated cuBLASLt algorithm."""
    _try_load_native_library()
    _register_fakes()
    if not _operator_is_registered("cublaslt_linear_out"):
        raise RuntimeError("Flux native cuBLASLt benchmark operator is not built")
    return torch.ops.flux.cublaslt_linear_out(
        input,
        weight,
        output,
        workspace,
        algorithm_index,
        max_workspace_bytes,
    )


def cublaslt_linear_config_out(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    workspace: torch.Tensor,
    algorithm: CublasLtAlgorithm,
) -> torch.Tensor:
    """Write ``input @ weight.T`` using one exact cuBLASLt configuration."""
    _try_load_native_library()
    _register_fakes()
    if not _operator_is_registered("cublaslt_linear_config_out"):
        raise RuntimeError("Flux native explicit cuBLASLt operator is not built")
    return torch.ops.flux.cublaslt_linear_config_out(
        input,
        weight,
        output,
        workspace,
        algorithm.algorithm_id,
        algorithm.tile_id,
        algorithm.split_k,
        algorithm.reduction_scheme,
        algorithm.cta_swizzle,
        algorithm.custom_option,
        algorithm.stages_id,
    )


_try_load_native_library()
_register_fakes()


__all__ = [
    "CublasLtAlgorithm",
    "cublaslt_algorithms",
    "cublaslt_linear_config_out",
    "cublaslt_linear_out",
    "native_cublaslt_linear_is_available",
]
