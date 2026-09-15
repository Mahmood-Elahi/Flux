"""Correctness and registration tests for native attention softmax."""

from __future__ import annotations

import math

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from flux.ops import (
    native_softmax_is_available,
    native_softmax_load_error,
    softmax,
    softmax_native,
)


RTOL = 1e-5
ATOL = 5e-7
WIDTHS = [129]

if native_softmax_load_error() is not None:
    raise RuntimeError("The built Flux native operator library failed to load") from (
        native_softmax_load_error()
    )

pytestmark = pytest.mark.skipif(
    not native_softmax_is_available(),
    reason="Flux native softmax operator has not been built",
)


def _devices() -> list[str]:
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def _input(shape: tuple[int, ...], device: str) -> torch.Tensor:
    positions = torch.arange(math.prod(shape), device=device, dtype=torch.float32)
    return (3.0 * torch.sin(positions * 0.013) + torch.cos(positions * 0.007)).reshape(
        shape
    )


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("width", WIDTHS)
def test_matches_reference_across_widths(device: str, width: int) -> None:
    row_count = 2 if width == 8192 else 5
    input = _input((row_count, width), device)

    actual = softmax_native(input)
    expected = softmax(input)

    assert actual.shape == input.shape
    assert actual.dtype == torch.float32
    assert actual.device == input.device
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("device", _devices())
def test_makes_non_contiguous_input_contiguous_without_modifying_it(device: str) -> None:
    input = _input((2, 129, 7), device).transpose(1, 2)
    before = input.clone()
    assert not input.is_contiguous()

    actual = softmax_native(input)

    assert actual.is_contiguous()
    assert actual.data_ptr() != input.data_ptr()
    torch.testing.assert_close(input, before, rtol=0, atol=0)
    torch.testing.assert_close(actual, softmax(input), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float16, torch.bfloat16, torch.int32])
def test_rejects_unsupported_dtypes(dtype: torch.dtype) -> None:
    with pytest.raises(RuntimeError, match="dtype torch.float32"):
        softmax_native(torch.ones((2, 3), dtype=dtype))


@pytest.mark.parametrize(
    ("input", "message"),
    [
        (torch.tensor(1.0), "at least one dimension"),
        (torch.empty(2, 0), "non-empty final dimension"),
        (torch.empty(0, 3), "at least one row"),
    ],
)
def test_rejects_invalid_shapes(input: torch.Tensor, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        softmax_native(input)


def test_rejects_autograd_input() -> None:
    input = torch.ones((2, 3), requires_grad=True)

    with pytest.raises(RuntimeError, match="inference-only"):
        softmax_native(input)


def test_inference_mode_accepts_input_that_requires_grad() -> None:
    input = _input((2, 3), "cpu").requires_grad_(True)

    with torch.inference_mode():
        actual = softmax_native(input)

    assert not actual.requires_grad
    torch.testing.assert_close(actual, softmax(input), rtol=RTOL, atol=ATOL)


def test_fake_tensor_returns_contiguous_output_with_matching_metadata() -> None:
    mode = FakeTensorMode()
    with mode:
        input = torch.empty((2, 9, 7, 129), dtype=torch.float32)
        output = softmax_native(input)

    assert isinstance(output, FakeTensor)
    assert output.shape == input.shape
    assert output.dtype == input.dtype
    assert output.device == input.device
    assert output.is_contiguous()
    assert output is not input


@pytest.mark.parametrize("device", _devices())
def test_torch_library_opcheck(device: str) -> None:
    input = _input((2, 7, 129), device)

    result = torch.library.opcheck(
        torch.ops.flux.softmax.default,
        (input,),
        rtol=RTOL,
        atol=ATOL,
    )

    assert all(status == "SUCCESS" for status in result.values())
