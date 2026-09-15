"""Correctness and registration tests for native fused residual RMSNorm."""

from __future__ import annotations

import math

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from flux.ops import (
    native_residual_rmsnorm_is_available,
    native_rmsnorm_load_error,
    residual_rmsnorm,
    residual_rmsnorm_native,
    residual_rmsnorm_native_out,
)


EPSILON = 1e-5
RTOL = 1e-5
ATOL = 2e-6

if native_rmsnorm_load_error() is not None:
    raise RuntimeError("The built Flux native operator library failed to load") from (
        native_rmsnorm_load_error()
    )

pytestmark = pytest.mark.skipif(
    not native_residual_rmsnorm_is_available(),
    reason="Flux native residual RMSNorm operator has not been built",
)


def _devices() -> list[str]:
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def _inputs(
    shape: tuple[int, ...], device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    element_count = math.prod(shape)
    positions = torch.arange(element_count, device=device, dtype=torch.float32)
    hidden = (torch.sin(positions * 0.013) + torch.cos(positions * 0.007)).reshape(
        shape
    )
    residual = (
        torch.cos(positions * 0.017) - torch.sin(positions * 0.011)
    ).reshape(shape)
    weight = torch.linspace(0.25, 1.75, shape[-1], device=device, dtype=torch.float32)
    return hidden, residual, weight


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize(
    "shape",
    [
        (2, 32, 576),
    ],
)
def test_matches_reference_across_shapes(
    device: str, shape: tuple[int, ...]
) -> None:
    hidden, residual, weight = _inputs(shape, device)
    expected_norm, expected_residual = residual_rmsnorm(
        hidden, residual, weight, EPSILON
    )

    actual_norm, actual_residual = residual_rmsnorm_native(
        hidden, residual, weight, EPSILON
    )

    for output in (actual_residual, actual_norm):
        assert output.shape == hidden.shape
        assert output.dtype == torch.float32
        assert output.device == hidden.device
        assert output.is_contiguous()
    torch.testing.assert_close(
        actual_residual, expected_residual, rtol=0, atol=0
    )
    torch.testing.assert_close(actual_norm, expected_norm, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("device", _devices())
def test_allocates_distinct_outputs_without_modifying_inputs(device: str) -> None:
    hidden, residual, weight = _inputs((2, 3, 63), device)
    hidden_before = hidden.clone()
    residual_before = residual.clone()
    weight_before = weight.clone()

    norm_out, residual_out = residual_rmsnorm_native(
        hidden, residual, weight, EPSILON
    )

    torch.testing.assert_close(hidden, hidden_before, rtol=0, atol=0)
    torch.testing.assert_close(residual, residual_before, rtol=0, atol=0)
    torch.testing.assert_close(weight, weight_before, rtol=0, atol=0)
    input_pointers = {hidden.data_ptr(), residual.data_ptr(), weight.data_ptr()}
    assert residual_out.data_ptr() not in input_pointers
    assert norm_out.data_ptr() not in input_pointers
    assert residual_out.data_ptr() != norm_out.data_ptr()


@pytest.mark.parametrize("device", _devices())
def test_internally_makes_all_inputs_contiguous(device: str) -> None:
    base_hidden, base_residual, _ = _inputs((2, 576, 7), device)
    hidden = base_hidden.transpose(1, 2)
    residual = base_residual.transpose(1, 2)
    weight = torch.linspace(
        0.25, 1.75, 1152, device=device, dtype=torch.float32
    )[::2]
    assert not hidden.is_contiguous()
    assert not residual.is_contiguous()
    assert not weight.is_contiguous()

    actual = residual_rmsnorm_native(hidden, residual, weight, EPSILON)
    expected = residual_rmsnorm(hidden, residual, weight, EPSILON)

    assert actual[0].is_contiguous()
    assert actual[1].is_contiguous()
    torch.testing.assert_close(actual[0], expected[0], rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("hidden", "residual", "weight", "epsilon", "message"),
    [
        (
            torch.ones(2, 3),
            torch.ones(1, 3),
            torch.ones(3),
            EPSILON,
            "same shape",
        ),
        (
            torch.ones(2, 3),
            torch.ones(2, 3),
            torch.ones(2),
            EPSILON,
            "weight length",
        ),
        (
            torch.ones(2, 3),
            torch.ones(2, 3),
            torch.ones(1, 3),
            EPSILON,
            "one-dimensional",
        ),
        (
            torch.ones(2, 3, dtype=torch.float64),
            torch.ones(2, 3),
            torch.ones(3),
            EPSILON,
            "hidden must have dtype torch.float32",
        ),
        (
            torch.ones(2, 3),
            torch.ones(2, 3, dtype=torch.float64),
            torch.ones(3),
            EPSILON,
            "residual must have dtype torch.float32",
        ),
        (
            torch.ones(2, 3),
            torch.ones(2, 3),
            torch.ones(3, dtype=torch.float64),
            EPSILON,
            "weight must have dtype torch.float32",
        ),
        (torch.tensor(1.0), torch.tensor(1.0), torch.ones(1), EPSILON, "at least one dimension"),
        (torch.empty(2, 0), torch.empty(2, 0), torch.empty(0), EPSILON, "non-empty final dimension"),
        (torch.empty(0, 3), torch.empty(0, 3), torch.ones(3), EPSILON, "at least one row"),
        (torch.ones(2, 3), torch.ones(2, 3), torch.ones(3), -EPSILON, "non-negative"),
        (torch.ones(2, 3), torch.ones(2, 3), torch.ones(3), float("nan"), "finite"),
        (torch.ones(2, 3), torch.ones(2, 3), torch.ones(3), float("inf"), "finite"),
    ],
)
def test_rejects_invalid_arguments(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        residual_rmsnorm_native(hidden, residual, weight, epsilon)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_rejects_mismatched_devices() -> None:
    with pytest.raises(RuntimeError, match="same device|requires CUDA tensors"):
        residual_rmsnorm_native(
            torch.ones(2, 3, device="cuda"),
            torch.ones(2, 3),
            torch.ones(3, device="cuda"),
            EPSILON,
        )


@pytest.mark.parametrize("argument", ["hidden", "residual", "weight"])
def test_rejects_autograd_inputs(argument: str) -> None:
    hidden = torch.ones(2, 3)
    residual = torch.ones(2, 3)
    weight = torch.ones(3)
    tensors = {"hidden": hidden, "residual": residual, "weight": weight}
    tensors[argument].requires_grad_(True)

    with pytest.raises(RuntimeError, match="inference-only"):
        residual_rmsnorm_native(hidden, residual, weight, EPSILON)


def test_inference_mode_accepts_parameters_that_require_grad() -> None:
    hidden = torch.ones(2, 3, requires_grad=True)
    residual = torch.full((2, 3), 0.5, requires_grad=True)
    weight = torch.ones(3, requires_grad=True)

    with torch.inference_mode():
        actual = residual_rmsnorm_native(hidden, residual, weight, EPSILON)

    assert not actual[0].requires_grad
    assert not actual[1].requires_grad
    expected = residual_rmsnorm(hidden, residual, weight, EPSILON)
    torch.testing.assert_close(actual[0], expected[0], rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_fake_tensor_returns_two_distinct_contiguous_outputs() -> None:
    mode = FakeTensorMode()
    with mode:
        hidden = torch.empty((2, 32, 576), dtype=torch.float32)
        residual = torch.empty_like(hidden)
        weight = torch.empty(576, dtype=torch.float32)

        norm_out, residual_out = residual_rmsnorm_native(
            hidden, residual, weight, EPSILON
        )

    assert isinstance(residual_out, FakeTensor)
    assert isinstance(norm_out, FakeTensor)
    assert residual_out is not norm_out
    for output in (residual_out, norm_out):
        assert output.shape == hidden.shape
        assert output.dtype == hidden.dtype
        assert output.device == hidden.device
        assert output.is_contiguous()


@pytest.mark.parametrize("device", _devices())
def test_torch_library_opcheck(device: str) -> None:
    hidden, residual, weight = _inputs((2, 7, 576), device)

    result = torch.library.opcheck(
        torch.ops.flux.residual_rmsnorm.default,
        (hidden, residual, weight, EPSILON),
        rtol=RTOL,
        atol=ATOL,
    )

    assert all(status == "SUCCESS" for status in result.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_out_contract_returns_and_overwrites_supplied_tensors() -> None:
    hidden, residual, weight = _inputs((2, 7, 576), "cuda")
    norm_out = torch.full_like(hidden, 41.0)
    residual_out = torch.full_like(hidden, -41.0)
    actual = residual_rmsnorm_native_out(
        hidden, residual, weight, EPSILON, norm_out, residual_out
    )
    assert actual[0] is norm_out and actual[1] is residual_out
    expected = residual_rmsnorm_native(hidden, residual, weight, EPSILON)
    torch.testing.assert_close(norm_out, expected[0])
    torch.testing.assert_close(residual_out, expected[1])
