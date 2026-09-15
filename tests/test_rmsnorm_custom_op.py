"""Correctness and registration tests for the native PyTorch RMSNorm op."""

from __future__ import annotations

import pytest
import torch

from flux.ops import (
    native_rmsnorm_is_available,
    native_rmsnorm_load_error,
    rms_norm,
    rms_norm_native,
    rms_norm_native_out,
)


EPSILON = 1e-5
RTOL = 1e-5
ATOL = 2e-6

if native_rmsnorm_load_error() is not None:
    raise RuntimeError("The built Flux native operator library failed to load") from (
        native_rmsnorm_load_error()
    )

pytestmark = pytest.mark.skipif(
    not native_rmsnorm_is_available(),
    reason="Flux native operators have not been built",
)


def _devices() -> list[str]:
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def _inputs(shape: tuple[int, ...], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    element_count = 1
    for dimension in shape:
        element_count *= dimension
    positions = torch.arange(element_count, device=device, dtype=torch.float32)
    input = (torch.sin(positions * 0.013) + torch.cos(positions * 0.007)).reshape(shape)
    weight = torch.linspace(0.25, 1.75, shape[-1], device=device, dtype=torch.float32)
    return input, weight


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize(
    "shape",
    [(2, 7, 576)],
)
def test_matches_reference_across_shapes(device: str, shape: tuple[int, ...]) -> None:
    input, weight = _inputs(shape, device)
    expected = rms_norm(input, weight, EPSILON)

    actual = rms_norm_native(input, weight, EPSILON)

    assert actual.shape == input.shape
    assert actual.dtype == torch.float32
    assert actual.device == input.device
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("device", _devices())
def test_internally_makes_tensors_contiguous(device: str) -> None:
    base_input, _ = _inputs((2, 576, 7), device)
    input = base_input.transpose(1, 2)
    weight = torch.linspace(0.25, 1.75, 1152, device=device, dtype=torch.float32)[::2]
    assert not input.is_contiguous()
    assert not weight.is_contiguous()

    actual = rms_norm_native(input, weight, EPSILON)
    expected = rms_norm(input, weight, EPSILON)

    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    ("input", "weight", "epsilon", "message"),
    [
        (torch.tensor(1.0), torch.ones(1), EPSILON, "at least one dimension"),
        (torch.ones(2, 3), torch.ones(2), EPSILON, "weight length"),
        (torch.ones(2, 3), torch.ones(1, 3), EPSILON, "one-dimensional"),
        (
            torch.ones(2, 3, dtype=torch.float64),
            torch.ones(3),
            EPSILON,
            "input must have dtype torch.float32",
        ),
        (
            torch.ones(2, 3),
            torch.ones(3, dtype=torch.float64),
            EPSILON,
            "weight must have dtype torch.float32",
        ),
        (torch.ones(2, 3), torch.ones(3), 0.0, "epsilon must be a positive"),
        (torch.ones(2, 3), torch.ones(3), -EPSILON, "epsilon must be a positive"),
    ],
)
def test_rejects_invalid_arguments(
    input: torch.Tensor, weight: torch.Tensor, epsilon: float, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        rms_norm_native(input, weight, epsilon)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_rejects_mismatched_devices() -> None:
    with pytest.raises(RuntimeError, match="same device|requires CUDA tensors"):
        rms_norm_native(torch.ones(2, 3, device="cuda"), torch.ones(3), EPSILON)


def test_rejects_autograd_inputs() -> None:
    input = torch.ones(2, 3, requires_grad=True)
    with pytest.raises(RuntimeError, match="inference-only"):
        rms_norm_native(input, torch.ones(3), EPSILON)


def test_inference_mode_accepts_parameters_that_require_grad() -> None:
    input = torch.ones(2, 3, requires_grad=True)
    weight = torch.ones(3, requires_grad=True)

    with torch.inference_mode():
        actual = rms_norm_native(input, weight, EPSILON)

    assert not actual.requires_grad
    torch.testing.assert_close(actual, rms_norm(input, weight, EPSILON))


@pytest.mark.parametrize("device", _devices())
def test_torch_library_opcheck(device: str) -> None:
    input, weight = _inputs((2, 7, 576), device)

    result = torch.library.opcheck(
        torch.ops.flux.rmsnorm.default,
        (input, weight, EPSILON),
        rtol=RTOL,
        atol=ATOL,
    )

    assert all(status == "SUCCESS" for status in result.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_out_contract_identity_poison_alias_and_schema() -> None:
    input, weight = _inputs((2, 7, 576), "cuda")
    output = torch.full_like(input, 73.0)
    assert rms_norm_native_out(input, weight, EPSILON, output) is output
    torch.testing.assert_close(output, rms_norm_native(input, weight, EPSILON))
    with pytest.raises(RuntimeError, match="must not alias"):
        rms_norm_native_out(input, weight, EPSILON, input)
    result = torch.library.opcheck(
        torch.ops.flux.rmsnorm_out.default,
        (input, weight, EPSILON, output),
        test_utils=("test_schema", "test_faketensor"),
    )
    assert all(status == "SUCCESS" for status in result.values())
