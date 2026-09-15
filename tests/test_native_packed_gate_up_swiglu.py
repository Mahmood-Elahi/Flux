"""Correctness and contract tests for fused SmolLM2 gate/up GEMV + SwiGLU."""

from __future__ import annotations

import pytest
import torch

from flux.ops import (
    native_packed_gate_up_gemv_is_available,
    packed_gate_up_swiglu_native_out,
)


pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and native_packed_gate_up_gemv_is_available()),
    reason="Flux native packed gate/up GEMV + SwiGLU is not built",
)


def _inputs(seed: int = 1234) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    input = torch.randn((1, 1, 576), generator=generator, device="cuda")
    weight = torch.randn((3072, 576), generator=generator, device="cuda")
    output = torch.full((1, 1, 1536), float("nan"), device="cuda")
    return input, weight, output


def _reference(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    packed = torch.nn.functional.linear(input, weight)
    return torch.nn.functional.silu(packed[..., :1536]) * packed[..., 1536:]


def test_matches_fp32_reference_and_overwrites_poisoned_output() -> None:
    input, weight, output = _inputs()
    expected = _reference(input, weight)
    with torch.inference_mode():
        returned = packed_gate_up_swiglu_native_out(input, weight, output)
    assert returned is output
    assert not torch.isnan(output).any()
    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-3)


def test_rejects_invalid_shape_dtype_layout_output_and_autograd() -> None:
    input, weight, output = _inputs()
    with pytest.raises(RuntimeError, match=r"\[1, 1, 576\]"):
        packed_gate_up_swiglu_native_out(input[..., :-1], weight, output)
    with pytest.raises(RuntimeError, match=r"\[3072, 576\]"):
        packed_gate_up_swiglu_native_out(input, weight[:-1], output)
    with pytest.raises(RuntimeError, match="float32"):
        packed_gate_up_swiglu_native_out(input.double(), weight, output)
    with pytest.raises(RuntimeError, match="contiguous"):
        packed_gate_up_swiglu_native_out(
            input, weight.t().contiguous().t(), output
        )
    with pytest.raises(RuntimeError, match="output shape"):
        packed_gate_up_swiglu_native_out(input, weight, output[..., :-1])
    weight.requires_grad_()
    with pytest.raises(RuntimeError, match="inference-only"):
        packed_gate_up_swiglu_native_out(input, weight, output)


def test_fake_tensor_and_schema_contract() -> None:
    input, weight, output = _inputs()
    torch.library.opcheck(
        torch.ops.flux.packed_gate_up_swiglu_out.default,
        (input, weight, output),
        test_utils=("test_schema", "test_faketensor"),
    )
