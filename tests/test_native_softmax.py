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
WIDTHS = [1, 3, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 512, 2048, 8192]

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
@pytest.mark.parametrize("shape", [(1, 9, 8, 64), (2, 9, 7, 129)])
def test_matches_reference_for_attention_shapes(
    device: str, shape: tuple[int, ...]
) -> None:
    input = _input(shape, device)

    actual = softmax_native(input)
    expected = softmax(input)

    assert actual.shape == shape
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(
        actual.sum(dim=-1),
        torch.ones(shape[:-1], device=device),
        rtol=RTOL,
        atol=ATOL,
    )


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


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("case", ["constant", "random", "large_positive", "large_negative"])
def test_numerical_properties(device: str, case: str) -> None:
    if case == "constant":
        input = torch.full((4, 257), 7.0, device=device)
    elif case == "random":
        generator = torch.Generator(device=device).manual_seed(1234)
        input = torch.randn((4, 257), generator=generator, device=device)
    elif case == "large_positive":
        input = _input((4, 257), device) + 10_000.0
    else:
        input = _input((4, 257), device) - 10_000.0

    actual = softmax_native(input)

    torch.testing.assert_close(actual, softmax(input), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(
        actual.sum(dim=-1), torch.ones(4, device=device), rtol=RTOL, atol=ATOL
    )
    assert torch.all(actual >= 0)
    assert torch.all(torch.isfinite(actual))


@pytest.mark.parametrize("device", _devices())
def test_shift_invariance(device: str) -> None:
    input = _input((4, 129), device)
    shifts = torch.tensor([-1000.0, -17.0, 23.0, 1000.0], device=device).unsqueeze(1)

    baseline = softmax_native(input)
    shifted = softmax_native(input + shifts)

    torch.testing.assert_close(shifted, baseline, rtol=5e-5, atol=2e-6)


@pytest.mark.parametrize("device", _devices())
def test_repeated_invocation_is_deterministic(device: str) -> None:
    input = _input((8, 513), device)

    first = softmax_native(input)
    second = softmax_native(input)

    torch.testing.assert_close(first, second, rtol=0, atol=0)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_uses_current_non_default_cuda_stream_for_producer_and_consumer() -> None:
    shape = (32, 257)
    values = _input(shape, "cuda")
    expected = softmax(values.cpu())
    input = torch.zeros_like(values)
    stream = torch.cuda.Stream()
    assert stream != torch.cuda.default_stream()

    with torch.cuda.stream(stream):
        torch.cuda._sleep(10_000_000)
        input.copy_(values)
        output = softmax_native(input)
        consumed = output + 0.0

    stream.synchronize()
    torch.testing.assert_close(consumed.cpu(), expected, rtol=RTOL, atol=ATOL)
