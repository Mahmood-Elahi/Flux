"""Contract tests for the FP32 PyTorch attention-softmax reference."""

import pytest
import torch

from flux.ops import softmax


RTOL = 1e-5
ATOL = 1e-6
SMOLLM2_ATTENTION_HEADS = 9


@pytest.mark.parametrize(
    "shape",
    [
        (7,),
        (3, 5),
        (2, 3, 7),
        (1, SMOLLM2_ATTENTION_HEADS, 17, 33),
        (2, SMOLLM2_ATTENTION_HEADS, 1, 129),
    ],
)
def test_matches_torch_softmax_across_shapes(shape: tuple[int, ...]) -> None:
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32)

    actual = softmax(x)
    expected = torch.softmax(x, dim=-1)

    assert actual.shape == x.shape
    assert actual.device == x.device
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "width",
    [1, 3, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129],
)
def test_reduction_boundary_widths_are_probabilities(width: int) -> None:
    torch.manual_seed(width)
    x = torch.randn((2, 3, width), dtype=torch.float32)

    actual = softmax(x)

    torch.testing.assert_close(
        actual.sum(dim=-1),
        torch.ones((2, 3), dtype=torch.float32),
        rtol=RTOL,
        atol=ATOL,
    )
    assert torch.all(actual >= 0)


def test_large_magnitude_logits_are_numerically_stable() -> None:
    x = torch.tensor(
        [
            [1000.0, 1001.0, 999.0],
            [-1000.0, -1001.0, -999.0],
            [1000.0, 0.0, -1000.0],
        ],
        dtype=torch.float32,
    )
    # Direct exponentiation overflows, demonstrating why max subtraction is
    # part of the reference contract.
    assert not torch.isfinite(torch.exp(x)).all()

    actual = softmax(x)
    expected = torch.softmax(x, dim=-1)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("width", [1, 7, 128])
def test_constant_rows_are_uniform(width: int) -> None:
    x = torch.full((2, 3, width), 42.0, dtype=torch.float32)

    actual = softmax(x)
    expected = torch.full_like(x, 1.0 / width)

    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def test_is_invariant_to_row_wise_constant_shifts() -> None:
    torch.manual_seed(1)
    x = torch.randn((2, 4, 33), dtype=torch.float32)
    row_shifts = torch.tensor([5.5, -3.25], dtype=torch.float32).view(2, 1, 1)

    torch.testing.assert_close(
        softmax(x), softmax(x + row_shifts), rtol=RTOL, atol=ATOL
    )


def test_preserves_shape_and_cpu_device() -> None:
    x = torch.randn((2, SMOLLM2_ATTENTION_HEADS, 3, 64), dtype=torch.float32)

    actual = softmax(x)

    assert actual.shape == x.shape
    assert actual.device == x.device


def test_is_deterministic() -> None:
    torch.manual_seed(2)
    x = torch.randn((2, SMOLLM2_ATTENTION_HEADS, 5, 65), dtype=torch.float32)

    first = softmax(x)
    second = softmax(x)

    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_supports_non_contiguous_input() -> None:
    torch.manual_seed(3)
    x = torch.randn((2, 7, 3), dtype=torch.float32).transpose(1, 2)
    assert x.shape == (2, 3, 7)
    assert not x.is_contiguous()

    actual = softmax(x)
    expected = torch.softmax(x, dim=-1)

    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def test_autograd_matches_torch_softmax() -> None:
    torch.manual_seed(4)
    x = torch.randn((2, 3, 7), dtype=torch.float32, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    output_gradient = torch.randn_like(x)

    actual_gradient = torch.autograd.grad(softmax(x), x, output_gradient)[0]
    expected_gradient = torch.autograd.grad(
        torch.softmax(reference_x, dim=-1), reference_x, output_gradient
    )[0]

    torch.testing.assert_close(
        actual_gradient, expected_gradient, rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float64, torch.int64],
)
def test_rejects_unsupported_dtypes(dtype: torch.dtype) -> None:
    x = torch.ones((2, 3), dtype=dtype)

    with pytest.raises(TypeError, match="float32"):
        softmax(x)


@pytest.mark.parametrize("x", [torch.tensor(1.0), torch.empty((2, 0))])
def test_rejects_inputs_without_non_empty_final_dimension(x: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="non-empty final dimension"):
        softmax(x)
