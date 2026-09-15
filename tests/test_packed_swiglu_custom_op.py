"""Correctness and registration tests for native packed FP32 SwiGLU."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from flux.ops import (
    native_packed_swiglu_is_available,
    native_rmsnorm_load_error,
    packed_swiglu_native,
    packed_swiglu_native_out,
)


RTOL = 2e-6
ATOL = 2e-6

if native_rmsnorm_load_error() is not None:
    raise RuntimeError("The built Flux native operator library failed to load") from (
        native_rmsnorm_load_error()
    )

pytestmark = pytest.mark.skipif(
    not native_packed_swiglu_is_available(),
    reason="Flux native packed SwiGLU operator has not been built",
)


def _devices() -> list[str]:
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def _input(shape: tuple[int, ...], device: str) -> torch.Tensor:
    positions = torch.arange(math.prod(shape), device=device, dtype=torch.float32)
    return (torch.sin(positions * 0.013) + torch.cos(positions * 0.007)).reshape(shape)


def _expected(packed: torch.Tensor) -> torch.Tensor:
    gate, up = packed.chunk(2, dim=-1)
    return F.silu(gate) * up


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize(
    "shape",
    [
        (2, 3, 10),
        (1, 128, 3072),
    ],
)
def test_matches_pytorch_across_shapes(
    device: str, shape: tuple[int, ...]
) -> None:
    packed = _input(shape, device)

    actual = packed_swiglu_native(packed)
    expected = _expected(packed)

    assert actual.shape == (*shape[:-1], shape[-1] // 2)
    assert actual.dtype == packed.dtype
    assert actual.device == packed.device
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("device", _devices())
def test_rejects_non_contiguous_packed_input_without_copy(device: str) -> None:
    packed = _input((2, 3072, 3), device).transpose(1, 2)
    assert not packed.is_contiguous()

    with pytest.raises(RuntimeError, match="must be contiguous"):
        packed_swiglu_native(packed)


@pytest.mark.parametrize(
    ("packed", "message"),
    [
        (torch.tensor(1.0), "at least one dimension"),
        (torch.empty(2, 0), "non-empty"),
        (torch.ones(2, 3), "must be even"),
        (torch.empty(0, 4), "at least one row"),
        (torch.ones(2, 4, dtype=torch.float64), "dtype torch.float32"),
    ],
)
def test_rejects_invalid_arguments(packed: torch.Tensor, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        packed_swiglu_native(packed)


def test_rejects_autograd_input() -> None:
    packed = torch.ones(2, 4, requires_grad=True)
    with pytest.raises(RuntimeError, match="inference-only"):
        packed_swiglu_native(packed)


def test_inference_mode_accepts_input_that_requires_grad() -> None:
    packed = torch.ones(2, 4, requires_grad=True)
    with torch.inference_mode():
        actual = packed_swiglu_native(packed)

    assert not actual.requires_grad
    torch.testing.assert_close(actual, _expected(packed), rtol=RTOL, atol=ATOL)


def test_fake_tensor_shape_dtype_device_and_layout() -> None:
    mode = FakeTensorMode()
    with mode:
        packed = torch.empty((2, 32, 3072), dtype=torch.float32)
        output = packed_swiglu_native(packed)

    assert isinstance(output, FakeTensor)
    assert output.shape == (2, 32, 1536)
    assert output.dtype == packed.dtype
    assert output.device == packed.device
    assert output.is_contiguous()


@pytest.mark.parametrize("device", _devices())
def test_torch_library_opcheck(device: str) -> None:
    packed = _input((2, 7, 3072), device)

    result = torch.library.opcheck(
        torch.ops.flux.packed_swiglu.default,
        (packed,),
        rtol=RTOL,
        atol=ATOL,
    )

    assert all(status == "SUCCESS" for status in result.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_out_contract_returns_and_overwrites_supplied_tensor() -> None:
    packed = _input((2, 7, 3072), "cuda")
    output = torch.full((2, 7, 1536), 59.0, device="cuda")
    assert packed_swiglu_native_out(packed, output) is output
    torch.testing.assert_close(output, packed_swiglu_native(packed), rtol=0, atol=0)
