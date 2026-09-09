"""Offline correctness tests for the FP32 PyTorch RMSNorm reference."""

import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from flux.ops.rmsnorm import rms_norm


SMOLLM2_EPS = 1e-5
# Native implementations may use a different reduction order in float32.
RTOL = 1e-5
ATOL = 1e-6


def _nontrivial_weight(hidden_size: int) -> torch.Tensor:
    return torch.linspace(0.25, 1.75, hidden_size, dtype=torch.float32)


@pytest.mark.parametrize(
    "shape",
    [
        (576,),
        (2, 576),
        (1, 1, 576),
        (2, 7, 576),
        (4, 3, 64),
    ],
)
def test_matches_torch_rmsnorm_across_shapes(shape: tuple[int, ...]) -> None:
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32)
    weight = _nontrivial_weight(shape[-1])

    expected_module = torch.nn.RMSNorm(
        shape[-1], eps=SMOLLM2_EPS, dtype=torch.float32
    )
    with torch.no_grad():
        expected_module.weight.copy_(weight)

    actual = rms_norm(x, weight, SMOLLM2_EPS)
    expected = expected_module(x)

    assert actual.shape == x.shape
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def test_matches_transformers_llama_rmsnorm_form() -> None:
    torch.manual_seed(1)
    x = torch.randn((2, 5, 576), dtype=torch.float32)
    weight = _nontrivial_weight(576)
    expected_module = LlamaRMSNorm(576, eps=SMOLLM2_EPS)
    with torch.no_grad():
        expected_module.weight.copy_(weight)

    actual = rms_norm(x, weight, SMOLLM2_EPS)
    expected = expected_module(x)

    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def test_zero_input() -> None:
    x = torch.zeros((2, 3, 576), dtype=torch.float32)
    weight = _nontrivial_weight(576)

    actual = rms_norm(x, weight, SMOLLM2_EPS)

    torch.testing.assert_close(actual, torch.zeros_like(x), rtol=0, atol=0)


def test_is_deterministic() -> None:
    torch.manual_seed(2)
    x = torch.randn((3, 4, 576), dtype=torch.float32)
    weight = _nontrivial_weight(576)

    first = rms_norm(x, weight, SMOLLM2_EPS)
    second = rms_norm(x, weight, SMOLLM2_EPS)

    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("x", "weight", "exception"),
    [
        (torch.tensor(1.0), torch.ones(1), ValueError),
        (torch.ones(2, 3), torch.ones(2), ValueError),
        (torch.ones(2, 3), torch.ones(1, 3), ValueError),
        (torch.ones(2, 3, dtype=torch.float64), torch.ones(3), TypeError),
        (torch.ones(2, 3), torch.ones(3, dtype=torch.float64), TypeError),
    ],
)
def test_validates_fp32_shape_assumptions(
    x: torch.Tensor, weight: torch.Tensor, exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        rms_norm(x, weight, SMOLLM2_EPS)


def test_rejects_negative_epsilon() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        rms_norm(torch.ones(2, 3), torch.ones(3), -1e-5)
