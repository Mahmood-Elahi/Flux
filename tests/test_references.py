"""Consolidated Python suite; see docs/PYTHON_SUITE_CONSOLIDATION_MILESTONE.md."""

from __future__ import annotations


# --- Migrated from test_residual_rmsnorm_ref.py ---

import pytest
import torch
from flux.ops import residual_rmsnorm as _rr_residual_rmsnorm
_rr_SMOLLM2_EPS = 1e-05
_rr_RTOL = 1e-05
_rr_ATOL = 1e-06

def _rr_manual_residual_rmsnorm(hidden: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    residual_out = hidden + residual
    mean_square = residual_out.pow(2).mean(dim=-1, keepdim=True)
    norm_out = weight * residual_out * torch.rsqrt(mean_square + eps)
    return (norm_out, residual_out)

def _rr_nontrivial_weight(hidden_size: int) -> torch.Tensor:
    return torch.linspace(0.25, 1.75, hidden_size, dtype=torch.float32)

@pytest.mark.parametrize('shape', [(576,), (4, 576), (2, 32, 576)])
def test_residual_rmsnorm_matches_independent_reference_across_shapes(shape: tuple[int, ...]) -> None:
    torch.manual_seed(0)
    hidden = torch.randn(shape, dtype=torch.float32)
    residual = torch.randn(shape, dtype=torch.float32)
    weight = _rr_nontrivial_weight(shape[-1])
    actual_norm, actual_residual = _rr_residual_rmsnorm(hidden, residual, weight, _rr_SMOLLM2_EPS)
    expected_norm, expected_residual = _rr_manual_residual_rmsnorm(hidden, residual, weight, _rr_SMOLLM2_EPS)
    assert actual_residual.shape == shape
    assert actual_norm.shape == shape
    assert actual_residual.dtype == torch.float32
    assert actual_norm.dtype == torch.float32
    torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(actual_norm, expected_norm, rtol=_rr_RTOL, atol=_rr_ATOL)

def test_residual_rmsnorm_returns_normalized_then_unnormalized_residual() -> None:
    hidden = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32)
    residual = torch.tensor([[0.5, -0.5, 1.0]], dtype=torch.float32)
    weight = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
    norm_out, residual_out = _rr_residual_rmsnorm(hidden, residual, weight, _rr_SMOLLM2_EPS)
    torch.testing.assert_close(residual_out, hidden + residual, rtol=0, atol=0)
    assert not torch.allclose(residual_out, norm_out)

@pytest.mark.parametrize('hidden_size', [1, 7, 63, 127, 575, 577])
def test_residual_rmsnorm_supports_arbitrary_hidden_sizes(hidden_size: int) -> None:
    torch.manual_seed(hidden_size)
    hidden = torch.randn((2, 3, hidden_size), dtype=torch.float32)
    residual = torch.randn_like(hidden)
    weight = _rr_nontrivial_weight(hidden_size)
    actual = _rr_residual_rmsnorm(hidden, residual, weight, _rr_SMOLLM2_EPS)
    expected = _rr_manual_residual_rmsnorm(hidden, residual, weight, _rr_SMOLLM2_EPS)
    torch.testing.assert_close(actual[0], expected[0], rtol=_rr_RTOL, atol=_rr_ATOL)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

@pytest.mark.parametrize('eps', [0.0, 1e-06, _rr_SMOLLM2_EPS, 0.001])
def test_residual_rmsnorm_supports_non_negative_epsilon_values(eps: float) -> None:
    torch.manual_seed(1)
    hidden = torch.randn((2, 3, 127), dtype=torch.float32)
    residual = torch.randn_like(hidden)
    weight = _rr_nontrivial_weight(127)
    actual = _rr_residual_rmsnorm(hidden, residual, weight, eps)
    expected = _rr_manual_residual_rmsnorm(hidden, residual, weight, eps)
    torch.testing.assert_close(actual[0], expected[0], rtol=_rr_RTOL, atol=_rr_ATOL)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

def test_residual_rmsnorm_does_not_modify_inputs() -> None:
    torch.manual_seed(2)
    hidden = torch.randn((2, 4, 63), dtype=torch.float32)
    residual = torch.randn_like(hidden)
    weight = _rr_nontrivial_weight(63)
    hidden_before = hidden.clone()
    residual_before = residual.clone()
    weight_before = weight.clone()
    _rr_residual_rmsnorm(hidden, residual, weight, _rr_SMOLLM2_EPS)
    torch.testing.assert_close(hidden, hidden_before, rtol=0, atol=0)
    torch.testing.assert_close(residual, residual_before, rtol=0, atol=0)
    torch.testing.assert_close(weight, weight_before, rtol=0, atol=0)

def test_residual_rmsnorm_gradients_from_both_outputs_match_manual_expression() -> None:
    torch.manual_seed(3)
    hidden = torch.randn((2, 3, 7), dtype=torch.float32, requires_grad=True)
    residual = torch.randn((2, 3, 7), dtype=torch.float32, requires_grad=True)
    weight = torch.randn(7, dtype=torch.float32, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    reference_residual = residual.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    residual_grad = torch.randn_like(hidden)
    norm_grad = torch.randn_like(hidden)
    actual_outputs = _rr_residual_rmsnorm(hidden, residual, weight, _rr_SMOLLM2_EPS)
    expected_outputs = _rr_manual_residual_rmsnorm(reference_hidden, reference_residual, reference_weight, _rr_SMOLLM2_EPS)
    actual_gradients = torch.autograd.grad(actual_outputs, (hidden, residual, weight), grad_outputs=(norm_grad, residual_grad))
    expected_gradients = torch.autograd.grad(expected_outputs, (reference_hidden, reference_residual, reference_weight), grad_outputs=(norm_grad, residual_grad))
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=_rr_RTOL, atol=_rr_ATOL)

def test_residual_rmsnorm_supports_non_contiguous_inputs() -> None:
    torch.manual_seed(4)
    hidden = torch.randn((2, 9, 3), dtype=torch.float32).transpose(1, 2)
    residual = torch.randn((2, 9, 3), dtype=torch.float32).transpose(1, 2)
    weight = torch.linspace(0.25, 1.75, 18, dtype=torch.float32)[::2]
    assert not hidden.is_contiguous()
    assert not residual.is_contiguous()
    assert not weight.is_contiguous()
    actual = _rr_residual_rmsnorm(hidden, residual, weight, _rr_SMOLLM2_EPS)
    expected = _rr_manual_residual_rmsnorm(hidden, residual, weight, _rr_SMOLLM2_EPS)
    torch.testing.assert_close(actual[0], expected[0], rtol=_rr_RTOL, atol=_rr_ATOL)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

@pytest.mark.parametrize(('hidden', 'residual', 'weight', 'exception', 'message'), [(torch.ones(2, 3), torch.ones(1, 3), torch.ones(3), ValueError, 'same shape'), (torch.ones(2, 3), torch.ones(2, 3), torch.ones(2), ValueError, 'one-dimensional'), (torch.tensor(1.0), torch.tensor(1.0), torch.ones(1), ValueError, 'non-empty final dimension'), (torch.empty(2, 0), torch.empty(2, 0), torch.empty(0), ValueError, 'non-empty final dimension'), (torch.ones(2, 3, dtype=torch.float64), torch.ones(2, 3), torch.ones(3), TypeError, 'float32'), (torch.ones(2, 3), torch.ones(2, 3, dtype=torch.float64), torch.ones(3), TypeError, 'float32'), (torch.ones(2, 3), torch.ones(2, 3), torch.ones(3, dtype=torch.float64), TypeError, 'float32')])
def test_residual_rmsnorm_rejects_invalid_inputs(hidden: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, exception: type[Exception], message: str) -> None:
    with pytest.raises(exception, match=message):
        _rr_residual_rmsnorm(hidden, residual, weight, _rr_SMOLLM2_EPS)

def test_residual_rmsnorm_rejects_negative_epsilon() -> None:
    with pytest.raises(ValueError, match='non-negative'):
        _rr_residual_rmsnorm(torch.ones(2, 3), torch.ones(2, 3), torch.ones(3), -1e-05)

@pytest.mark.parametrize('eps', [float('nan'), float('inf')])
def test_residual_rmsnorm_rejects_non_finite_epsilon(eps: float) -> None:
    with pytest.raises(ValueError, match='finite'):
        _rr_residual_rmsnorm(torch.ones(2, 3), torch.ones(2, 3), torch.ones(3), eps)


# --- Migrated from test_softmax_ref.py ---

import pytest
import torch
from flux.ops import softmax as _soft_softmax
_soft_RTOL = 1e-05
_soft_ATOL = 1e-06
_soft_SMOLLM2_ATTENTION_HEADS = 9

@pytest.mark.parametrize('shape', [(7,), (3, 5), (2, 3, 7), (1, _soft_SMOLLM2_ATTENTION_HEADS, 17, 33), (2, _soft_SMOLLM2_ATTENTION_HEADS, 1, 129)])
def test_softmax_matches_torch_softmax_across_shapes(shape: tuple[int, ...]) -> None:
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32)
    actual = _soft_softmax(x)
    expected = torch.softmax(x, dim=-1)
    assert actual.shape == x.shape
    assert actual.device == x.device
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=_soft_RTOL, atol=_soft_ATOL)

@pytest.mark.parametrize('width', [1, 3, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129])
def test_softmax_reduction_boundary_widths_are_probabilities(width: int) -> None:
    torch.manual_seed(width)
    x = torch.randn((2, 3, width), dtype=torch.float32)
    actual = _soft_softmax(x)
    torch.testing.assert_close(actual.sum(dim=-1), torch.ones((2, 3), dtype=torch.float32), rtol=_soft_RTOL, atol=_soft_ATOL)
    assert torch.all(actual >= 0)

def test_softmax_large_magnitude_logits_are_numerically_stable() -> None:
    x = torch.tensor([[1000.0, 1001.0, 999.0], [-1000.0, -1001.0, -999.0], [1000.0, 0.0, -1000.0]], dtype=torch.float32)
    assert not torch.isfinite(torch.exp(x)).all()
    actual = _soft_softmax(x)
    expected = torch.softmax(x, dim=-1)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=_soft_RTOL, atol=_soft_ATOL)

@pytest.mark.parametrize('width', [1, 7, 128])
def test_softmax_constant_rows_are_uniform(width: int) -> None:
    x = torch.full((2, 3, width), 42.0, dtype=torch.float32)
    actual = _soft_softmax(x)
    expected = torch.full_like(x, 1.0 / width)
    torch.testing.assert_close(actual, expected, rtol=_soft_RTOL, atol=_soft_ATOL)

def test_softmax_is_invariant_to_row_wise_constant_shifts() -> None:
    torch.manual_seed(1)
    x = torch.randn((2, 4, 33), dtype=torch.float32)
    row_shifts = torch.tensor([5.5, -3.25], dtype=torch.float32).view(2, 1, 1)
    torch.testing.assert_close(_soft_softmax(x), _soft_softmax(x + row_shifts), rtol=_soft_RTOL, atol=_soft_ATOL)

def test_softmax_supports_non_contiguous_input() -> None:
    torch.manual_seed(3)
    x = torch.randn((2, 7, 3), dtype=torch.float32).transpose(1, 2)
    assert x.shape == (2, 3, 7)
    assert not x.is_contiguous()
    actual = _soft_softmax(x)
    expected = torch.softmax(x, dim=-1)
    torch.testing.assert_close(actual, expected, rtol=_soft_RTOL, atol=_soft_ATOL)

def test_softmax_autograd_matches_torch_softmax() -> None:
    torch.manual_seed(4)
    x = torch.randn((2, 3, 7), dtype=torch.float32, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    output_gradient = torch.randn_like(x)
    actual_gradient = torch.autograd.grad(_soft_softmax(x), x, output_gradient)[0]
    expected_gradient = torch.autograd.grad(torch.softmax(reference_x, dim=-1), reference_x, output_gradient)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=_soft_RTOL, atol=_soft_ATOL)

@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float64, torch.int64])
def test_softmax_rejects_unsupported_dtypes(dtype: torch.dtype) -> None:
    x = torch.ones((2, 3), dtype=dtype)
    with pytest.raises(TypeError, match='float32'):
        _soft_softmax(x)

@pytest.mark.parametrize('x', [torch.tensor(1.0), torch.empty((2, 0))])
def test_softmax_rejects_inputs_without_non_empty_final_dimension(x: torch.Tensor) -> None:
    with pytest.raises(ValueError, match='non-empty final dimension'):
        _soft_softmax(x)


# --- Migrated from test_rmsnorm_ref.py ---

import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm as _rms_LlamaRMSNorm
from flux.ops.rmsnorm import rms_norm as _rms_rms_norm
_rms_SMOLLM2_EPS = 1e-05
_rms_RTOL = 1e-05
_rms_ATOL = 1e-06

def _rms_nontrivial_weight(hidden_size: int) -> torch.Tensor:
    return torch.linspace(0.25, 1.75, hidden_size, dtype=torch.float32)

@pytest.mark.parametrize('shape', [(576,), (2, 576), (1, 1, 576), (2, 7, 576), (4, 3, 64)])
def test_rmsnorm_matches_torch_rmsnorm_across_shapes(shape: tuple[int, ...]) -> None:
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32)
    weight = _rms_nontrivial_weight(shape[-1])
    expected_module = torch.nn.RMSNorm(shape[-1], eps=_rms_SMOLLM2_EPS, dtype=torch.float32)
    with torch.no_grad():
        expected_module.weight.copy_(weight)
    actual = _rms_rms_norm(x, weight, _rms_SMOLLM2_EPS)
    expected = expected_module(x)
    assert actual.shape == x.shape
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=_rms_RTOL, atol=_rms_ATOL)

def test_rmsnorm_matches_transformers_llama_rmsnorm_form() -> None:
    torch.manual_seed(1)
    x = torch.randn((2, 5, 576), dtype=torch.float32)
    weight = _rms_nontrivial_weight(576)
    expected_module = _rms_LlamaRMSNorm(576, eps=_rms_SMOLLM2_EPS)
    with torch.no_grad():
        expected_module.weight.copy_(weight)
    actual = _rms_rms_norm(x, weight, _rms_SMOLLM2_EPS)
    expected = expected_module(x)
    torch.testing.assert_close(actual, expected, rtol=_rms_RTOL, atol=_rms_ATOL)

def test_rmsnorm_zero_input() -> None:
    x = torch.zeros((2, 3, 576), dtype=torch.float32)
    weight = _rms_nontrivial_weight(576)
    actual = _rms_rms_norm(x, weight, _rms_SMOLLM2_EPS)
    torch.testing.assert_close(actual, torch.zeros_like(x), rtol=0, atol=0)

@pytest.mark.parametrize(('x', 'weight', 'exception'), [(torch.tensor(1.0), torch.ones(1), ValueError), (torch.ones(2, 3), torch.ones(2), ValueError), (torch.ones(2, 3), torch.ones(1, 3), ValueError), (torch.ones(2, 3, dtype=torch.float64), torch.ones(3), TypeError), (torch.ones(2, 3), torch.ones(3, dtype=torch.float64), TypeError)])
def test_rmsnorm_validates_fp32_shape_assumptions(x: torch.Tensor, weight: torch.Tensor, exception: type[Exception]) -> None:
    with pytest.raises(exception):
        _rms_rms_norm(x, weight, _rms_SMOLLM2_EPS)

def test_rmsnorm_rejects_negative_epsilon() -> None:
    with pytest.raises(ValueError, match='non-negative'):
        _rms_rms_norm(torch.ones(2, 3), torch.ones(3), -1e-05)


# --- Migrated from test_gqa_decode_decode_ref.py ---

import pytest
import torch
from flux.ops import gqa_decode_attention as _gqa_gqa_decode_attention

def _gqa_unexpanded_oracle(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None, scale: float, length: int) -> torch.Tensor:
    outputs = []
    groups = query.shape[1] // key.shape[1]
    for query_head in range(query.shape[1]):
        kv_head = query_head // groups
        scores = torch.matmul(query[:, query_head:query_head + 1], key[:, kv_head:kv_head + 1, :length].transpose(2, 3)) * scale
        if mask is not None:
            mask_head = 0 if mask.shape[1] == 1 else query_head
            scores = scores + mask[:, mask_head:mask_head + 1, :, :length]
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
        outputs.append(torch.matmul(probabilities, value[:, kv_head:kv_head + 1, :length]))
    return torch.cat(outputs, dim=1)

@pytest.mark.parametrize('batch,query_heads,kv_heads,head_dim', [(1, 9, 3, 64), (2, 4, 2, 8)])
@pytest.mark.parametrize('capacity,valid_length', [(1, 1), (17, 11), (129, 128)])
def test_gqa_decode_reference_maps_heads_and_respects_valid_length(batch: int, query_heads: int, kv_heads: int, head_dim: int, capacity: int, valid_length: int) -> None:
    torch.manual_seed(1000 + capacity)
    query = torch.randn(batch, query_heads, 1, head_dim)
    key = torch.randn(batch, kv_heads, capacity, head_dim)
    value = torch.randn_like(key)
    mask = torch.randn(batch, 1, 1, capacity) * 0.1
    length = torch.tensor(valid_length, dtype=torch.int64)
    actual = _gqa_gqa_decode_attention(query, key, value, mask, head_dim ** (-0.5), length)
    expected = _gqa_unexpanded_oracle(query, key, value, mask, head_dim ** (-0.5), valid_length)
    torch.testing.assert_close(actual, expected, rtol=1e-05, atol=5e-07)

def test_gqa_decode_reference_mask_changes_only_the_selected_positions() -> None:
    query = torch.ones((1, 2, 1, 2))
    key = torch.zeros((1, 1, 3, 2))
    value = torch.tensor([[[[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]]]])
    mask = torch.tensor([[[[0.0, torch.finfo(torch.float32).min, torch.finfo(torch.float32).min]]]])
    actual = _gqa_gqa_decode_attention(query, key, value, mask, 1.0)
    torch.testing.assert_close(actual, value[:, :, :1].expand(1, 2, 1, 2))


# --- Migrated from test_rope_ref.py ---

import pytest
import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as _rope_apply_rotary_pos_emb
from flux.ops import rope as _rope_rope

def _rope_inputs(batch: int, sequence: int, position_offset: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(191)
    query = torch.randn(batch, 9, sequence, 64)
    key = torch.randn(batch, 3, sequence, 64)
    positions = torch.arange(position_offset, position_offset + sequence).float()
    inv_freq = 1.0 / 100000.0 ** (torch.arange(0, 64, 2).float() / 64)
    frequencies = torch.outer(positions, inv_freq)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cos = embedding.cos().unsqueeze(0).expand(batch, -1, -1)
    sin = embedding.sin().unsqueeze(0).expand(batch, -1, -1)
    return (query, key, cos, sin)

@pytest.mark.parametrize(('batch', 'sequence', 'position_offset'), [(1, 1, 0), (1, 1, 4095), (1, 7, 37), (2, 128, 1024)])
def test_rope_matches_transformers(batch: int, sequence: int, position_offset: int) -> None:
    query, key, cos, sin = _rope_inputs(batch, sequence, position_offset)
    expected = _rope_apply_rotary_pos_emb(query, key, cos, sin)
    actual = _rope_rope(query, key, cos, sin)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

def test_rope_single_rope_batch_broadcasts_across_attention_batch() -> None:
    query, key, cos, sin = _rope_inputs(2, 5, 123)
    query[1].copy_(query[0])
    key[1].copy_(key[0])
    actual = _rope_rope(query, key, cos[:1], sin[:1])
    torch.testing.assert_close(actual[0][0], actual[0][1], rtol=0, atol=0)
    torch.testing.assert_close(actual[1][0], actual[1][1], rtol=0, atol=0)

def test_rope_rejects_odd_head_dimension() -> None:
    with pytest.raises(ValueError, match='even'):
        _rope_rope(torch.ones(1, 2, 3, 7), torch.ones(1, 1, 3, 7), torch.ones(1, 3, 7), torch.ones(1, 3, 7))


# --- Migrated from test_attention_score_softmax_ref.py ---

import pytest
import torch
from flux.ops.attention_score_softmax import attention_score_softmax as _attention_score_soft_attention_score_softmax

def test_attention_score_soft_matches_exact_unfused_expression() -> None:
    generator = torch.Generator().manual_seed(1234)
    scores = torch.randn((2, 4, 7, 11), generator=generator)
    mask = torch.randn((2, 1, 7, 11), generator=generator)
    scale = 7 ** (-0.5)
    actual = _attention_score_soft_attention_score_softmax(scores, mask, scale)
    expected = torch.softmax(scores * scale + mask, dim=-1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

def test_attention_score_soft_supports_broadcast_mask_dimensions() -> None:
    scores = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    mask = torch.zeros((1, 1, 4, 5), dtype=torch.float32)
    actual = _attention_score_soft_attention_score_softmax(scores, mask, 0.125)
    assert actual.shape == scores.shape
    torch.testing.assert_close(actual.sum(dim=-1), torch.ones((2, 3, 4)))

@pytest.mark.parametrize(('scores', 'mask', 'message'), [(torch.ones(2, 3), torch.ones(2, 3), 'rank four'), (torch.ones(1, 2, 3, 4), torch.ones(1, 2, 3), 'rank four'), (torch.ones(1, 2, 3, 4), torch.ones(1, 2, 2, 4), 'broadcastable'), (torch.ones(1, 2, 3, 4), torch.ones(1, 1, 3, 5), 'broadcastable')])
def test_attention_score_soft_rejects_invalid_shapes(scores: torch.Tensor, mask: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _attention_score_soft_attention_score_softmax(scores, mask, 0.5)

def test_attention_score_soft_rejects_non_fp32_and_nonfinite_scale() -> None:
    scores = torch.ones((1, 2, 3, 4))
    mask = torch.zeros((1, 1, 3, 4))
    with pytest.raises(TypeError, match='scores'):
        _attention_score_soft_attention_score_softmax(scores.half(), mask, 0.5)
    with pytest.raises(TypeError, match='additive_attention_mask'):
        _attention_score_soft_attention_score_softmax(scores, mask.half(), 0.5)
    with pytest.raises(ValueError, match='finite'):
        _attention_score_soft_attention_score_softmax(scores, mask, float('inf'))


# --- Migrated from test_swiglu_ref.py ---

import pytest
import torch
import torch.nn.functional as F
from flux.ops.packed_swiglu import packed_swiglu as _swiglu_packed_swiglu

def test_swiglu_matches_explicit_silu_multiply() -> None:
    packed = torch.linspace(-4.0, 4.0, 2 * 3 * 10).reshape(2, 3, 10)
    gate, up = packed.chunk(2, dim=-1)
    actual = _swiglu_packed_swiglu(packed)
    torch.testing.assert_close(actual, F.silu(gate) * up)

@pytest.mark.parametrize(('packed', 'error', 'message'), [(torch.tensor(1.0), ValueError, 'at least one dimension'), (torch.empty(2, 0), ValueError, 'non-empty and even'), (torch.ones(2, 3), ValueError, 'non-empty and even'), (torch.ones(2, 4, dtype=torch.float64), TypeError, 'float32')])
def test_swiglu_rejects_invalid_inputs(packed: torch.Tensor, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        _swiglu_packed_swiglu(packed)
