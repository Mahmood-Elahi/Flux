"""Contract tests for the dual-output residual RMSNorm reference."""

import pytest
import torch

from flux.ops import residual_rmsnorm


SMOLLM2_EPS = 1e-5
RTOL = 1e-5
ATOL = 1e-6


def _manual_residual_rmsnorm(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual_out = hidden + residual
    mean_square = residual_out.pow(2).mean(dim=-1, keepdim=True)
    norm_out = weight * residual_out * torch.rsqrt(mean_square + eps)
    return residual_out, norm_out


def _nontrivial_weight(hidden_size: int) -> torch.Tensor:
    return torch.linspace(0.25, 1.75, hidden_size, dtype=torch.float32)


@pytest.mark.parametrize("shape", [(576,), (4, 576), (2, 32, 576)])
def test_matches_independent_reference_across_shapes(
    shape: tuple[int, ...],
) -> None:
    torch.manual_seed(0)
    hidden = torch.randn(shape, dtype=torch.float32)
    residual = torch.randn(shape, dtype=torch.float32)
    weight = _nontrivial_weight(shape[-1])

    actual_residual, actual_norm = residual_rmsnorm(
        hidden, residual, weight, SMOLLM2_EPS
    )
    expected_residual, expected_norm = _manual_residual_rmsnorm(
        hidden, residual, weight, SMOLLM2_EPS
    )

    assert actual_residual.shape == shape
    assert actual_norm.shape == shape
    assert actual_residual.dtype == torch.float32
    assert actual_norm.dtype == torch.float32
    torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(actual_norm, expected_norm, rtol=RTOL, atol=ATOL)


def test_preserves_unnormalized_residual_as_first_output() -> None:
    hidden = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32)
    residual = torch.tensor([[0.5, -0.5, 1.0]], dtype=torch.float32)
    weight = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)

    residual_out, norm_out = residual_rmsnorm(
        hidden, residual, weight, SMOLLM2_EPS
    )

    torch.testing.assert_close(residual_out, hidden + residual, rtol=0, atol=0)
    assert not torch.allclose(residual_out, norm_out)


@pytest.mark.parametrize("hidden_size", [1, 7, 63, 127, 575, 577])
def test_supports_arbitrary_hidden_sizes(hidden_size: int) -> None:
    torch.manual_seed(hidden_size)
    hidden = torch.randn((2, 3, hidden_size), dtype=torch.float32)
    residual = torch.randn_like(hidden)
    weight = _nontrivial_weight(hidden_size)

    actual = residual_rmsnorm(hidden, residual, weight, SMOLLM2_EPS)
    expected = _manual_residual_rmsnorm(hidden, residual, weight, SMOLLM2_EPS)

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("eps", [0.0, 1e-6, SMOLLM2_EPS, 1e-3])
def test_supports_non_negative_epsilon_values(eps: float) -> None:
    torch.manual_seed(1)
    hidden = torch.randn((2, 3, 127), dtype=torch.float32)
    residual = torch.randn_like(hidden)
    weight = _nontrivial_weight(127)

    actual = residual_rmsnorm(hidden, residual, weight, eps)
    expected = _manual_residual_rmsnorm(hidden, residual, weight, eps)

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=RTOL, atol=ATOL)


def test_does_not_modify_inputs() -> None:
    torch.manual_seed(2)
    hidden = torch.randn((2, 4, 63), dtype=torch.float32)
    residual = torch.randn_like(hidden)
    weight = _nontrivial_weight(63)
    hidden_before = hidden.clone()
    residual_before = residual.clone()
    weight_before = weight.clone()

    residual_rmsnorm(hidden, residual, weight, SMOLLM2_EPS)

    torch.testing.assert_close(hidden, hidden_before, rtol=0, atol=0)
    torch.testing.assert_close(residual, residual_before, rtol=0, atol=0)
    torch.testing.assert_close(weight, weight_before, rtol=0, atol=0)


def test_gradients_from_both_outputs_match_manual_expression() -> None:
    torch.manual_seed(3)
    hidden = torch.randn((2, 3, 7), dtype=torch.float32, requires_grad=True)
    residual = torch.randn((2, 3, 7), dtype=torch.float32, requires_grad=True)
    weight = torch.randn(7, dtype=torch.float32, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    reference_residual = residual.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    residual_grad = torch.randn_like(hidden)
    norm_grad = torch.randn_like(hidden)

    actual_outputs = residual_rmsnorm(hidden, residual, weight, SMOLLM2_EPS)
    expected_outputs = _manual_residual_rmsnorm(
        reference_hidden, reference_residual, reference_weight, SMOLLM2_EPS
    )
    actual_gradients = torch.autograd.grad(
        actual_outputs,
        (hidden, residual, weight),
        grad_outputs=(residual_grad, norm_grad),
    )
    expected_gradients = torch.autograd.grad(
        expected_outputs,
        (reference_hidden, reference_residual, reference_weight),
        grad_outputs=(residual_grad, norm_grad),
    )

    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def test_supports_non_contiguous_inputs() -> None:
    torch.manual_seed(4)
    hidden = torch.randn((2, 9, 3), dtype=torch.float32).transpose(1, 2)
    residual = torch.randn((2, 9, 3), dtype=torch.float32).transpose(1, 2)
    weight = torch.linspace(0.25, 1.75, 18, dtype=torch.float32)[::2]
    assert not hidden.is_contiguous()
    assert not residual.is_contiguous()
    assert not weight.is_contiguous()

    actual = residual_rmsnorm(hidden, residual, weight, SMOLLM2_EPS)
    expected = _manual_residual_rmsnorm(hidden, residual, weight, SMOLLM2_EPS)

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    ("hidden", "residual", "weight", "exception", "message"),
    [
        (
            torch.ones(2, 3),
            torch.ones(1, 3),
            torch.ones(3),
            ValueError,
            "same shape",
        ),
        (
            torch.ones(2, 3),
            torch.ones(2, 3),
            torch.ones(2),
            ValueError,
            "one-dimensional",
        ),
        (
            torch.tensor(1.0),
            torch.tensor(1.0),
            torch.ones(1),
            ValueError,
            "non-empty final dimension",
        ),
        (
            torch.empty(2, 0),
            torch.empty(2, 0),
            torch.empty(0),
            ValueError,
            "non-empty final dimension",
        ),
        (
            torch.ones(2, 3, dtype=torch.float64),
            torch.ones(2, 3),
            torch.ones(3),
            TypeError,
            "float32",
        ),
        (
            torch.ones(2, 3),
            torch.ones(2, 3, dtype=torch.float64),
            torch.ones(3),
            TypeError,
            "float32",
        ),
        (
            torch.ones(2, 3),
            torch.ones(2, 3),
            torch.ones(3, dtype=torch.float64),
            TypeError,
            "float32",
        ),
    ],
)
def test_rejects_invalid_inputs(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        residual_rmsnorm(hidden, residual, weight, SMOLLM2_EPS)


def test_rejects_negative_epsilon() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        residual_rmsnorm(
            torch.ones(2, 3), torch.ones(2, 3), torch.ones(3), -1e-5
        )
