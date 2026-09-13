"""Fixed-shape FP32 packed gate/up GEMV plus SwiGLU CUDA operator."""

from __future__ import annotations

import torch

from flux.ops.native_rmsnorm import _operator_is_registered, _try_load_native_library


_FAKES_REGISTERED = False


def _register_fake() -> None:
    global _FAKES_REGISTERED
    if _FAKES_REGISTERED:
        return
    if _operator_is_registered("packed_gate_up_swiglu_out"):
        @torch.library.register_fake("flux::packed_gate_up_swiglu_out")
        def _fused_fake(
            input: torch.Tensor,
            weight: torch.Tensor,
            output: torch.Tensor,
        ) -> torch.Tensor:
            del input, weight
            return output

        _FAKES_REGISTERED = True


def _load() -> None:
    _try_load_native_library()
    _register_fake()


def native_packed_gate_up_gemv_is_available() -> bool:
    _load()
    return _operator_is_registered("packed_gate_up_swiglu_out")


def packed_gate_up_swiglu_native_out(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    _load()
    if not _operator_is_registered("packed_gate_up_swiglu_out"):
        raise RuntimeError("Flux native fused packed gate/up SwiGLU is not built")
    return torch.ops.flux.packed_gate_up_swiglu_out(input, weight, output)


_load()


__all__ = [
    "native_packed_gate_up_gemv_is_available",
    "packed_gate_up_swiglu_native_out",
]
