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


def test_matches_fp32_reference_over_random_inputs_and_poisoned_output() -> None:
    for seed in (7, 1234, 9001):
        input, weight, output = _inputs(seed)
        expected = _reference(input, weight)
        with torch.inference_mode():
            returned = packed_gate_up_swiglu_native_out(input, weight, output)
        assert returned is output
        assert not torch.isnan(output).any()
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-3)


def test_repeated_invocation_is_bit_exact_and_uses_current_stream() -> None:
    input, weight, output = _inputs(44)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.inference_mode():
        input.fill_(0.125)
        packed_gate_up_swiglu_native_out(input, weight, output)
        consumed = output + 0.0
    stream.synchronize()
    expected = _reference(input, weight)
    torch.testing.assert_close(consumed, expected, rtol=1e-4, atol=1e-3)
    first = output.clone()
    with torch.inference_mode():
        packed_gate_up_swiglu_native_out(input, weight, output)
    torch.testing.assert_close(output, first, rtol=0, atol=0)


def test_cuda_graph_replay_has_stable_output_and_overwrites_poison() -> None:
    input, weight, output = _inputs(81)
    with torch.inference_mode():
        packed_gate_up_swiglu_native_out(input, weight, output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.inference_mode():
        captured = packed_gate_up_swiglu_native_out(input, weight, output)
    address = output.data_ptr()
    for value in (0.25, -0.5, 0.75):
        input.fill_(value)
        output.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(
            output, _reference(input, weight), rtol=1e-4, atol=1e-3
        )
        assert output.data_ptr() == address == captured.data_ptr()


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
