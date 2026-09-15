"""PyTorch-facing contracts for Flux native operators.

Native CTest owns CUDA numerical sweeps, streams, graphs, allocation behavior,
and kernel boundary cases. This suite keeps dispatcher/FakeTensor contracts,
one representative numerical oracle per operator, and public argument/output
semantics.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from flux.ops import (
    attention_score_softmax_native,
    cublaslt_algorithms,
    cublaslt_linear_out,
    gqa_decode_attention,
    gqa_decode_attention_native,
    gqa_decode_attention_native_out,
    native_attention_score_softmax_is_available,
    native_cublaslt_linear_is_available,
    native_gqa_decode_attention_is_available,
    native_packed_gate_up_gemv_is_available,
    native_packed_qkv_rope_cache_is_available,
    native_packed_swiglu_is_available,
    native_residual_rmsnorm_is_available,
    native_rmsnorm_is_available,
    native_rmsnorm_load_error,
    native_rope_is_available,
    native_softmax_is_available,
    packed_gate_up_swiglu_native_out,
    packed_qkv_rope_cache_native,
    packed_qkv_rope_cache_native_out,
    packed_swiglu_native,
    packed_swiglu_native_out,
    residual_rmsnorm,
    residual_rmsnorm_native,
    residual_rmsnorm_native_out,
    rms_norm,
    rms_norm_native,
    rms_norm_native_out,
    rope_native,
    softmax,
    softmax_native,
)

EPSILON = 1e-5
WORKSPACE_BYTES = 4 * 1024 * 1024

if native_rmsnorm_load_error() is not None:
    raise RuntimeError("The built Flux native operator library failed to load") from native_rmsnorm_load_error()


def _devices() -> list[str]:
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def _opcheck_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _assert_opcheck(result: dict[str, str]) -> None:
    assert all(status == "SUCCESS" for status in result.values())


def _wave(shape: tuple[int, ...], device: str) -> torch.Tensor:
    values = torch.arange(math.prod(shape), device=device, dtype=torch.float32)
    return (torch.sin(values * 0.013) + torch.cos(values * 0.007)).reshape(shape)


def test_registered_operator_schemas() -> None:
    for name in (
        "attention_score_softmax",
        "cublaslt_linear_config_out",
        "cublaslt_linear_out",
        "gqa_decode_attention",
        "gqa_decode_attention_out",
        "packed_gate_up_swiglu_out",
        "packed_qkv_rope_cache",
        "packed_qkv_rope_cache_out",
        "packed_swiglu",
        "packed_swiglu_out",
        "residual_rmsnorm",
        "residual_rmsnorm_out",
        "rmsnorm",
        "rmsnorm_out",
        "rope",
        "softmax",
    ):
        assert hasattr(torch.ops.flux, name), name


# RMSNorm and residual-RMSNorm

@pytest.mark.skipif(not native_rmsnorm_is_available(), reason="native RMSNorm is unavailable")
@pytest.mark.parametrize("device", _devices())
def test_rmsnorm_representative_numerical_and_layout_contract(device: str) -> None:
    input = _wave((2, 576, 7), device).transpose(1, 2)
    weight = torch.linspace(0.25, 1.75, 1152, device=device)[::2]
    assert not input.is_contiguous() and not weight.is_contiguous()
    actual = rms_norm_native(input, weight, EPSILON)
    assert actual.shape == input.shape
    assert actual.dtype == input.dtype and actual.device == input.device
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, rms_norm(input, weight, EPSILON), rtol=1e-5, atol=2e-6)


@pytest.mark.skipif(not native_rmsnorm_is_available(), reason="native RMSNorm is unavailable")
def test_rmsnorm_error_contract() -> None:
    cases = (
        (torch.tensor(1.0), torch.ones(1), EPSILON, "at least one dimension"),
        (torch.ones(2, 3), torch.ones(2), EPSILON, "weight length"),
        (torch.ones(2, 3), torch.ones(3, dtype=torch.float64), EPSILON, "weight must have dtype torch.float32"),
        (torch.ones(2, 3), torch.ones(3), 0.0, "epsilon must be a positive"),
    )
    for input, weight, epsilon, message in cases:
        with pytest.raises(RuntimeError, match=message):
            rms_norm_native(input, weight, epsilon)
    with pytest.raises(RuntimeError, match="inference-only"):
        rms_norm_native(torch.ones(2, 3, requires_grad=True), torch.ones(3), EPSILON)


@pytest.mark.skipif(not native_rmsnorm_is_available(), reason="native RMSNorm is unavailable")
def test_rmsnorm_opcheck() -> None:
    device = _opcheck_device()
    _assert_opcheck(torch.library.opcheck(
        torch.ops.flux.rmsnorm.default,
        (_wave((2, 7, 576), device), torch.ones(576, device=device), EPSILON),
        rtol=1e-5,
        atol=2e-6,
    ))


@pytest.mark.skipif(not (torch.cuda.is_available() and native_rmsnorm_is_available()), reason="CUDA RMSNorm is unavailable")
def test_rmsnorm_out_identity_alias_and_fake_tensor_contract() -> None:
    input = _wave((2, 7, 576), "cuda")
    weight = torch.linspace(0.25, 1.75, 576, device="cuda")
    output = torch.full_like(input, 73.0)
    assert rms_norm_native_out(input, weight, EPSILON, output) is output
    torch.testing.assert_close(output, rms_norm(input, weight, EPSILON), rtol=1e-5, atol=2e-6)
    with pytest.raises(RuntimeError, match="must not alias"):
        rms_norm_native_out(input, weight, EPSILON, input)
    _assert_opcheck(torch.library.opcheck(
        torch.ops.flux.rmsnorm_out.default,
        (input, weight, EPSILON, output),
        test_utils=("test_schema", "test_faketensor"),
    ))


@pytest.mark.skipif(not native_residual_rmsnorm_is_available(), reason="native residual RMSNorm is unavailable")
@pytest.mark.parametrize("device", _devices())
def test_residual_rmsnorm_representative_numerical_and_output_contract(device: str) -> None:
    hidden = _wave((2, 7, 576), device)
    residual = _wave((2, 7, 576), device).mul_(0.25)
    weight = torch.linspace(0.25, 1.75, 576, device=device)
    hidden_before = hidden.clone()
    expected_norm, expected_residual = residual_rmsnorm(hidden, residual, weight, EPSILON)
    actual_norm, actual_residual = residual_rmsnorm_native(hidden, residual, weight, EPSILON)
    assert actual_norm.shape == actual_residual.shape == hidden.shape
    assert actual_norm.is_contiguous() and actual_residual.is_contiguous()
    assert actual_norm.data_ptr() != actual_residual.data_ptr()
    torch.testing.assert_close(hidden, hidden_before, rtol=0, atol=0)
    torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(actual_norm, expected_norm, rtol=1e-5, atol=2e-6)


@pytest.mark.skipif(not native_residual_rmsnorm_is_available(), reason="native residual RMSNorm is unavailable")
def test_residual_rmsnorm_error_contract() -> None:
    cases = (
        (torch.ones(2, 3), torch.ones(1, 3), torch.ones(3), EPSILON, "same shape"),
        (torch.ones(2, 3), torch.ones(2, 3), torch.ones(2), EPSILON, "weight length"),
        (torch.ones(2, 3, dtype=torch.float64), torch.ones(2, 3), torch.ones(3), EPSILON, "hidden must have dtype torch.float32"),
        (torch.ones(2, 3), torch.ones(2, 3), torch.ones(3), float("inf"), "finite"),
    )
    for hidden, residual, weight, epsilon, message in cases:
        with pytest.raises(RuntimeError, match=message):
            residual_rmsnorm_native(hidden, residual, weight, epsilon)
    with pytest.raises(RuntimeError, match="inference-only"):
        residual_rmsnorm_native(torch.ones(2, 3, requires_grad=True), torch.ones(2, 3), torch.ones(3), EPSILON)


@pytest.mark.skipif(not native_residual_rmsnorm_is_available(), reason="native residual RMSNorm is unavailable")
def test_residual_rmsnorm_fake_tensor_and_opcheck() -> None:
    with FakeTensorMode():
        hidden = torch.empty((2, 7, 576))
        norm, residual = residual_rmsnorm_native(hidden, torch.empty_like(hidden), torch.empty(576), EPSILON)
    assert isinstance(norm, FakeTensor) and isinstance(residual, FakeTensor)
    assert norm is not residual and norm.shape == residual.shape == hidden.shape
    device = _opcheck_device()
    values = (_wave((2, 7, 576), device), _wave((2, 7, 576), device), torch.ones(576, device=device), EPSILON)
    _assert_opcheck(torch.library.opcheck(
        torch.ops.flux.residual_rmsnorm.default, values, rtol=1e-5, atol=2e-6
    ))


@pytest.mark.skipif(not (torch.cuda.is_available() and native_residual_rmsnorm_is_available()), reason="CUDA residual RMSNorm is unavailable")
def test_residual_rmsnorm_out_identity_contract() -> None:
    hidden = _wave((2, 7, 576), "cuda")
    residual = hidden * 0.25
    weight = torch.ones(576, device="cuda")
    norm_out = torch.full_like(hidden, 41.0)
    residual_out = torch.full_like(hidden, -41.0)
    returned = residual_rmsnorm_native_out(hidden, residual, weight, EPSILON, norm_out, residual_out)
    assert returned[0] is norm_out and returned[1] is residual_out
    expected = residual_rmsnorm(hidden, residual, weight, EPSILON)
    torch.testing.assert_close(norm_out, expected[0], rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(residual_out, expected[1], rtol=0, atol=0)


# Softmax, attention score softmax, RoPE, and packed SwiGLU

@pytest.mark.skipif(not native_softmax_is_available(), reason="native softmax is unavailable")
@pytest.mark.parametrize("device", _devices())
def test_softmax_representative_numerical_contract(device: str) -> None:
    input = 3.0 * _wave((5, 129), device)
    actual = softmax_native(input)
    assert actual.shape == input.shape and actual.is_contiguous()
    torch.testing.assert_close(actual, softmax(input), rtol=1e-5, atol=5e-7)


@pytest.mark.skipif(not native_softmax_is_available(), reason="native softmax is unavailable")
def test_softmax_error_fake_tensor_and_opcheck() -> None:
    for input, message in (
        (torch.tensor(1.0), "at least one dimension"),
        (torch.empty(2, 0), "non-empty final dimension"),
        (torch.ones(2, 3, dtype=torch.float16), "dtype torch.float32"),
    ):
        with pytest.raises(RuntimeError, match=message):
            softmax_native(input)
    with pytest.raises(RuntimeError, match="inference-only"):
        softmax_native(torch.ones(2, 3, requires_grad=True))
    with FakeTensorMode():
        fake_input = torch.empty((2, 9, 7, 129))
        output = softmax_native(fake_input)
    assert isinstance(output, FakeTensor) and output.shape == fake_input.shape
    device = _opcheck_device()
    _assert_opcheck(torch.library.opcheck(
        torch.ops.flux.softmax.default, (_wave((2, 7, 129), device),), rtol=1e-5, atol=5e-7
    ))


@pytest.mark.skipif(not native_attention_score_softmax_is_available(), reason="native attention softmax is unavailable")
@pytest.mark.parametrize("device", _devices())
def test_attention_softmax_representative_numerical_contract(device: str) -> None:
    scores = 3.0 * _wave((2, 3, 5, 17), device)
    mask = _wave((2, 1, 5, 17), device) * 0.25
    actual = attention_score_softmax_native(scores, mask, 17**-0.5)
    expected = torch.softmax(scores * (17**-0.5) + mask, dim=-1)
    assert actual.shape == scores.shape and actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=5e-7)


@pytest.mark.skipif(not native_attention_score_softmax_is_available(), reason="native attention softmax is unavailable")
def test_attention_softmax_error_fake_tensor_and_opcheck() -> None:
    scores = torch.ones((1, 2, 3, 4))
    mask = torch.zeros((1, 1, 3, 4))
    for invalid_scores, invalid_mask, scale, message in (
        (scores.reshape(2, 3, 4), mask, 0.5, "rank four"),
        (scores, torch.zeros((1, 1, 2, 4)), 0.5, "broadcastable"),
        (scores.half(), mask, 0.5, "scores must have dtype torch.float32"),
        (scores, mask, float("inf"), "finite FP32"),
    ):
        with pytest.raises(RuntimeError, match=message):
            attention_score_softmax_native(invalid_scores, invalid_mask, scale)
    with pytest.raises(RuntimeError, match="inference-only"):
        attention_score_softmax_native(scores.requires_grad_(), mask, 0.5)
    with FakeTensorMode():
        fake_scores = torch.empty((2, 9, 7, 129))
        output = attention_score_softmax_native(fake_scores, torch.empty((2, 1, 7, 129)), 0.125)
    assert isinstance(output, FakeTensor) and output.shape == fake_scores.shape
    device = _opcheck_device()
    op_scores = _wave((2, 3, 4, 17), device)
    op_mask = _wave((2, 1, 4, 17), device)
    _assert_opcheck(torch.library.opcheck(
        torch.ops.flux.attention_score_softmax.default,
        (op_scores, op_mask, 0.125),
        rtol=1e-5,
        atol=5e-7,
    ))


def _rope_inputs(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = _wave((2, 7, 576), device)
    query = hidden.view(2, 7, 9, 64).transpose(1, 2)
    key = hidden[..., :192].view(2, 7, 3, 64).transpose(1, 2)
    positions = torch.arange(31, 38, device=device, dtype=torch.float32)
    inv_freq = 1.0 / 100000.0 ** (torch.arange(0, 64, 2, device=device).float() / 64)
    frequencies = torch.outer(positions, inv_freq)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    return query, key, embedding.cos().unsqueeze(0).expand(2, -1, -1), embedding.sin().unsqueeze(0).expand(2, -1, -1)


@pytest.mark.skipif(not native_rope_is_available(), reason="native RoPE is unavailable")
@pytest.mark.parametrize("device", _devices())
def test_rope_representative_transformers_contract(device: str) -> None:
    inputs = _rope_inputs(device)
    actual = rope_native(*inputs)
    expected = apply_rotary_pos_emb(*inputs)
    assert actual[0].is_contiguous() and actual[1].is_contiguous()
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-6, atol=2e-7)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-6, atol=2e-7)


@pytest.mark.skipif(not native_rope_is_available(), reason="native RoPE is unavailable")
def test_rope_error_and_opcheck_contract() -> None:
    query, key, cos, sin = _rope_inputs("cpu")
    for invalid_query, invalid_key, invalid_cos, invalid_sin, message in (
        (query[0], key, cos, sin, "rank four"),
        (query.double(), key, cos, sin, "float32"),
        (query[..., :63], key[..., :63], cos[..., :63], sin[..., :63], "even"),
    ):
        with pytest.raises(RuntimeError, match=message):
            rope_native(invalid_query, invalid_key, invalid_cos, invalid_sin)
    with pytest.raises(RuntimeError, match="inference-only"):
        rope_native(query.requires_grad_(), key, cos, sin)
    device = _opcheck_device()
    with torch.inference_mode():
        _assert_opcheck(torch.library.opcheck(
            torch.ops.flux.rope.default, _rope_inputs(device), rtol=1e-6, atol=2e-7
        ))


@pytest.mark.skipif(not native_packed_swiglu_is_available(), reason="native packed SwiGLU is unavailable")
@pytest.mark.parametrize("device", _devices())
def test_packed_swiglu_representative_numerical_contract(device: str) -> None:
    packed = _wave((1, 7, 3072), device)
    gate, up = packed.chunk(2, dim=-1)
    actual = packed_swiglu_native(packed)
    assert actual.shape == (1, 7, 1536) and actual.is_contiguous()
    torch.testing.assert_close(actual, F.silu(gate) * up, rtol=2e-6, atol=2e-6)


@pytest.mark.skipif(not native_packed_swiglu_is_available(), reason="native packed SwiGLU is unavailable")
def test_packed_swiglu_error_fake_tensor_and_opcheck() -> None:
    for packed, message in (
        (torch.tensor(1.0), "at least one dimension"),
        (torch.ones(2, 3), "must be even"),
        (torch.ones(2, 4, dtype=torch.float64), "dtype torch.float32"),
    ):
        with pytest.raises(RuntimeError, match=message):
            packed_swiglu_native(packed)
    with pytest.raises(RuntimeError, match="inference-only"):
        packed_swiglu_native(torch.ones(2, 4, requires_grad=True))
    with FakeTensorMode():
        packed = torch.empty((2, 32, 3072))
        output = packed_swiglu_native(packed)
    assert isinstance(output, FakeTensor) and output.shape == (2, 32, 1536)
    device = _opcheck_device()
    _assert_opcheck(torch.library.opcheck(
        torch.ops.flux.packed_swiglu.default,
        (_wave((2, 7, 3072), device),),
        rtol=2e-6,
        atol=2e-6,
    ))


@pytest.mark.skipif(not (torch.cuda.is_available() and native_packed_swiglu_is_available()), reason="CUDA packed SwiGLU is unavailable")
def test_packed_swiglu_out_identity_contract() -> None:
    packed = _wave((2, 7, 3072), "cuda")
    output = torch.full((2, 7, 1536), 59.0, device="cuda")
    assert packed_swiglu_native_out(packed, output) is output
    gate, up = packed.chunk(2, dim=-1)
    torch.testing.assert_close(output, F.silu(gate) * up, rtol=2e-6, atol=2e-6)


# Decode-only fused operators

def _gqa_inputs(device: str, capacity: int = 73) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(1307)
    query = torch.randn((1, 9, 1, 64), generator=generator, device=device)
    keys = torch.randn((1, 3, capacity, 64), generator=generator, device=device)
    values = torch.randn((1, 3, capacity, 64), generator=generator, device=device)
    mask = torch.randn((1, 1, 1, capacity), generator=generator, device=device) * 0.05
    return query, keys, values, mask


@pytest.mark.skipif(not native_gqa_decode_attention_is_available(), reason="native GQA decode attention is unavailable")
@pytest.mark.parametrize("device", _devices())
def test_gqa_decode_representative_numerical_and_mask_contract(device: str) -> None:
    query, keys, values, mask = _gqa_inputs(device)
    length = torch.tensor(41, device=device)
    actual = gqa_decode_attention_native(query, keys, values, mask, 0.125, length)
    expected = gqa_decode_attention(query, keys, values, mask, 0.125, length)
    assert actual.shape == query.shape and actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)
    without_mask = gqa_decode_attention_native(query, keys, values, None, 0.125, length)
    torch.testing.assert_close(without_mask, gqa_decode_attention(query, keys, values, None, 0.125, length), rtol=2e-5, atol=1e-6)


@pytest.mark.skipif(not native_gqa_decode_attention_is_available(), reason="native GQA decode attention is unavailable")
def test_gqa_decode_error_fake_tensor_and_opcheck() -> None:
    query, keys, values, mask = _gqa_inputs("cpu", 17)
    cases = (
        ((query.expand(2, -1, -1, -1), keys.expand(2, -1, -1, -1), values.expand(2, -1, -1, -1), mask.expand(2, -1, -1, -1), 0.125, None), "batch size must be one"),
        ((query.half(), keys, values, mask, 0.125, None), "float32"),
        ((query, keys, values, mask[..., :-1], 0.125, None), "broadcastable"),
        ((query, keys, values, mask, 0.125, torch.tensor(18)), "within cache capacity"),
    )
    for arguments, message in cases:
        with pytest.raises(RuntimeError, match=message):
            gqa_decode_attention_native(*arguments)
    with pytest.raises(RuntimeError, match="inference-only"):
        gqa_decode_attention_native(query.requires_grad_(), keys, values, mask, 0.125)
    with FakeTensorMode():
        fake_query = torch.empty((1, 9, 1, 64))
        fake_keys = torch.empty((1, 3, 129, 64))
        output = gqa_decode_attention_native(fake_query, fake_keys, torch.empty_like(fake_keys), torch.empty((1, 1, 1, 129)), 0.125, torch.empty((), dtype=torch.int64))
    assert isinstance(output, FakeTensor) and output.shape == fake_query.shape
    device = _opcheck_device()
    inputs = _gqa_inputs(device, 17)
    length = torch.tensor(11, device=device)
    _assert_opcheck(torch.library.opcheck(
        torch.ops.flux.gqa_decode_attention.default,
        (*inputs, 0.125, length),
        rtol=2e-5,
        atol=1e-6,
    ))


@pytest.mark.skipif(not (torch.cuda.is_available() and native_gqa_decode_attention_is_available()), reason="CUDA GQA decode attention is unavailable")
def test_gqa_decode_out_identity_contract() -> None:
    query, keys, values, mask = _gqa_inputs("cuda", 1024)
    length = torch.tensor(777, device="cuda")
    output = torch.full_like(query, 67.0)
    workspace = torch.empty((1, 9, 8, 66), device="cuda")
    assert gqa_decode_attention_native_out(query, keys, values, mask, 0.125, length, output, workspace) is output
    torch.testing.assert_close(output, gqa_decode_attention(query, keys, values, mask, 0.125, length), rtol=2e-5, atol=1e-6)


def _qkv_inputs() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(1245)
    packed = torch.randn((1, 1, 960), generator=generator, device="cuda")
    frequencies = torch.randn((1, 1, 64), generator=generator, device="cuda")
    keys = torch.randn((1, 3, 17, 64), generator=generator, device="cuda")
    values = torch.randn((1, 3, 17, 64), generator=generator, device="cuda")
    return packed, frequencies.cos(), frequencies.sin(), keys, values, torch.tensor(11, device="cuda")


def _qkv_reference(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed, cos, sin, keys, values, _ = inputs
    query = packed[..., :576].view(1, 1, 9, 64).transpose(1, 2)
    key = packed[..., 576:768].view(1, 1, 3, 64).transpose(1, 2)
    value = packed[..., 768:].view(1, 1, 3, 64).transpose(1, 2)
    query, key = rope_native(query, key, cos, sin)
    expected_keys = keys.clone()
    expected_values = values.clone()
    expected_keys[..., 11:12, :].copy_(key)
    expected_values[..., 11:12, :].copy_(value)
    return query, expected_keys, expected_values


@pytest.mark.skipif(not (torch.cuda.is_available() and native_packed_qkv_rope_cache_is_available()), reason="CUDA packed QKV/RoPE/cache is unavailable")
def test_packed_qkv_rope_cache_representative_numerical_contract() -> None:
    inputs = _qkv_inputs()
    expected_query, expected_keys, expected_values = _qkv_reference(inputs)
    actual = packed_qkv_rope_cache_native(*inputs)
    assert actual.shape == (1, 9, 1, 64) and actual.is_contiguous()
    assert inputs[-1].item() == 12
    torch.testing.assert_close(actual, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(inputs[3], expected_keys, rtol=0, atol=0)
    torch.testing.assert_close(inputs[4], expected_values, rtol=0, atol=0)


@pytest.mark.skipif(not native_packed_qkv_rope_cache_is_available(), reason="native packed QKV/RoPE/cache is unavailable")
def test_packed_qkv_rope_cache_fake_tensor_contract() -> None:
    with FakeTensorMode():
        packed = torch.empty((1, 1, 960), device="cuda")
        cos = torch.empty((1, 1, 64), device="cuda")
        keys = torch.empty((1, 3, 17, 64), device="cuda")
        output = packed_qkv_rope_cache_native(packed, cos, torch.empty_like(cos), keys, torch.empty_like(keys), torch.empty((), dtype=torch.int64, device="cuda"))
    assert isinstance(output, FakeTensor)
    assert output.shape == (1, 9, 1, 64) and output.is_contiguous()


@pytest.mark.skipif(not (torch.cuda.is_available() and native_packed_qkv_rope_cache_is_available()), reason="CUDA packed QKV/RoPE/cache is unavailable")
def test_packed_qkv_rope_cache_error_opcheck_and_out_contract() -> None:
    inputs = _qkv_inputs()
    with pytest.raises(RuntimeError, match=r"\[1, 1, 960\]"):
        packed_qkv_rope_cache_native(inputs[0][..., :-1], *inputs[1:])
    with pytest.raises(RuntimeError, match="float32"):
        packed_qkv_rope_cache_native(inputs[0].double(), *inputs[1:])
    autograd_inputs = _qkv_inputs()
    autograd_inputs[0].requires_grad_()
    with pytest.raises(RuntimeError, match="inference-only"):
        packed_qkv_rope_cache_native(*autograd_inputs)
    with torch.inference_mode():
        _assert_opcheck(torch.library.opcheck(
            torch.ops.flux.packed_qkv_rope_cache.default, _qkv_inputs(), rtol=0, atol=0
        ))
    out_inputs = _qkv_inputs()
    expected_query, expected_keys, expected_values = _qkv_reference(out_inputs)
    output = torch.full((1, 9, 1, 64), 83.0, device="cuda")
    assert packed_qkv_rope_cache_native_out(*out_inputs, output) is output
    torch.testing.assert_close(output, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(out_inputs[3], expected_keys, rtol=0, atol=0)
    torch.testing.assert_close(out_inputs[4], expected_values, rtol=0, atol=0)


# CUDA library-backed projection operators

@pytest.mark.skipif(not (torch.cuda.is_available() and native_cublaslt_linear_is_available()), reason="CUDA cuBLASLt is unavailable")
def test_cublaslt_representative_numerical_and_fake_tensor_contract() -> None:
    generator = torch.Generator(device="cuda").manual_seed(1536)
    input = torch.randn((1, 1, 576), generator=generator, device="cuda")
    weight = torch.randn((960, 576), generator=generator, device="cuda")
    output = torch.full((1, 1, 960), float("nan"), device="cuda")
    workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")
    algorithms = cublaslt_algorithms(input, weight, max_workspace_bytes=WORKSPACE_BYTES, max_algorithms=16)
    assert algorithms and all(item.workspace_bytes <= WORKSPACE_BYTES for item in algorithms)
    returned = cublaslt_linear_out(
        input,
        weight,
        output,
        workspace,
        algorithm_index=algorithms[0].index,
        max_workspace_bytes=WORKSPACE_BYTES,
    )
    assert returned is output
    torch.testing.assert_close(output, F.linear(input, weight), rtol=1e-5, atol=1e-5)
    _assert_opcheck(torch.library.opcheck(
        torch.ops.flux.cublaslt_linear_out.default,
        (input, weight, output, workspace, 0, WORKSPACE_BYTES),
        test_utils=("test_schema", "test_faketensor"),
    ))
    algorithm = algorithms[min(5, len(algorithms) - 1)]
    _assert_opcheck(torch.library.opcheck(
        torch.ops.flux.cublaslt_linear_config_out.default,
        (input, weight, output, torch.empty(0, dtype=torch.uint8, device="cuda"), algorithm.algorithm_id, algorithm.tile_id, algorithm.split_k, algorithm.reduction_scheme, algorithm.cta_swizzle, algorithm.custom_option, algorithm.stages_id),
        test_utils=("test_schema", "test_faketensor"),
    ))


@pytest.mark.skipif(not (torch.cuda.is_available() and native_packed_gate_up_gemv_is_available()), reason="CUDA packed gate/up GEMV is unavailable")
def test_gate_up_representative_numerical_and_output_contract() -> None:
    generator = torch.Generator(device="cuda").manual_seed(1234)
    input = torch.randn((1, 1, 576), generator=generator, device="cuda")
    weight = torch.randn((3072, 576), generator=generator, device="cuda")
    output = torch.full((1, 1, 1536), float("nan"), device="cuda")
    assert packed_gate_up_swiglu_native_out(input, weight, output) is output
    packed = F.linear(input, weight)
    torch.testing.assert_close(output, F.silu(packed[..., :1536]) * packed[..., 1536:], rtol=1e-4, atol=1e-3)


@pytest.mark.skipif(not (torch.cuda.is_available() and native_packed_gate_up_gemv_is_available()), reason="CUDA packed gate/up GEMV is unavailable")
def test_gate_up_error_fake_tensor_and_schema_contract() -> None:
    input = torch.ones((1, 1, 576), device="cuda")
    weight = torch.ones((3072, 576), device="cuda")
    output = torch.empty((1, 1, 1536), device="cuda")
    for arguments, message in (
        ((input[..., :-1], weight, output), r"\[1, 1, 576\]"),
        ((input, weight[:-1], output), r"\[3072, 576\]"),
        ((input.double(), weight, output), "float32"),
        ((input, weight, output[..., :-1]), "output shape"),
    ):
        with pytest.raises(RuntimeError, match=message):
            packed_gate_up_swiglu_native_out(*arguments)
    with pytest.raises(RuntimeError, match="inference-only"):
        packed_gate_up_swiglu_native_out(input, weight.requires_grad_(), output)
    weight.requires_grad_(False)
    _assert_opcheck(torch.library.opcheck(
        torch.ops.flux.packed_gate_up_swiglu_out.default,
        (input, weight, output),
        test_utils=("test_schema", "test_faketensor"),
    ))
