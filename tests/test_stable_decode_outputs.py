"""Correctness coverage for fixed-shape CUDA decode output buffers."""

from __future__ import annotations

import copy

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from flux.model.smollm2_cuda_graph import FluxCUDAGraphDecode
from flux.model.smollm2_flux import (
    FLUX_GQA_DECODE_ATTENTION_CATEGORY,
    FLUX_OPERATOR_CATEGORIES,
    FLUX_PACKED_MLP_CATEGORY,
    FLUX_PACKED_QKV_CATEGORY,
    FLUX_PACKED_QKV_ROPE_CACHE_CATEGORY,
    FLUX_PACKED_SWIGLU_CATEGORY,
    enable_flux_ops,
)
from flux.ops import (
    gqa_decode_attention_native,
    gqa_decode_attention_native_out,
    native_gqa_decode_attention_is_available,
    native_packed_qkv_rope_cache_is_available,
    native_packed_swiglu_is_available,
    native_residual_rmsnorm_is_available,
    native_rmsnorm_is_available,
    packed_qkv_rope_cache_native,
    packed_qkv_rope_cache_native_out,
    packed_swiglu_native,
    packed_swiglu_native_out,
    residual_rmsnorm_native,
    residual_rmsnorm_native_out,
    rms_norm_native,
    rms_norm_native_out,
)


_AVAILABLE = (
    torch.cuda.is_available()
    and native_rmsnorm_is_available()
    and native_residual_rmsnorm_is_available()
    and native_packed_swiglu_is_available()
    and native_gqa_decode_attention_is_available()
    and native_packed_qkv_rope_cache_is_available()
)
_CUDA_ONLY = pytest.mark.skipif(
    not _AVAILABLE, reason="CUDA and rebuilt Flux native operators are required"
)
_FULL_OPERATORS = FLUX_OPERATOR_CATEGORIES | {
    FLUX_PACKED_MLP_CATEGORY,
    FLUX_PACKED_QKV_CATEGORY,
    FLUX_PACKED_SWIGLU_CATEGORY,
    FLUX_GQA_DECODE_ATTENTION_CATEGORY,
    FLUX_PACKED_QKV_ROPE_CACHE_CATEGORY,
}


@_CUDA_ONLY
def test_out_variants_overwrite_sentinel_buffers_repeatedly() -> None:
    generator = torch.Generator(device="cuda").manual_seed(9301)
    hidden = torch.randn((1, 1, 576), generator=generator, device="cuda")
    residual = torch.randn_like(hidden)
    weight = torch.randn((576,), generator=generator, device="cuda")
    packed_mlp = torch.randn((1, 1, 3072), generator=generator, device="cuda")
    packed_qkv = torch.randn((1, 1, 960), generator=generator, device="cuda")
    frequencies = torch.randn((1, 1, 64), generator=generator, device="cuda")
    cos, sin = frequencies.cos(), frequencies.sin()
    query = torch.randn((1, 9, 1, 64), generator=generator, device="cuda")
    keys = torch.randn((1, 3, 1024, 64), generator=generator, device="cuda")
    values = torch.randn_like(keys)
    mask = torch.zeros((1, 1, 1, 1024), device="cuda")
    cache_length = torch.tensor(777, device="cuda")

    expected_rms = rms_norm_native(hidden, weight, 1e-5)
    expected_norm, expected_residual = residual_rmsnorm_native(
        hidden, residual, weight, 1e-5
    )
    expected_swiglu = packed_swiglu_native(packed_mlp)
    expected_attention = gqa_decode_attention_native(
        query, keys, values, mask, 0.125, cache_length
    )
    expected_qkv_keys = keys.clone()
    expected_qkv_values = values.clone()
    expected_qkv_length = cache_length.clone()
    expected_query = packed_qkv_rope_cache_native(
        packed_qkv,
        cos,
        sin,
        expected_qkv_keys,
        expected_qkv_values,
        expected_qkv_length,
    )

    rms_output = torch.empty_like(hidden)
    norm_output = torch.empty_like(hidden)
    residual_output = torch.empty_like(hidden)
    swiglu_output = torch.empty_like(expected_swiglu)
    attention_output = torch.empty_like(query)
    workspace = torch.empty((1, 9, 8, 66), device="cuda")
    query_output = torch.empty_like(query)
    qkv_keys = keys.clone()
    qkv_values = values.clone()
    qkv_length = cache_length.clone()
    addresses = tuple(
        item.data_ptr()
        for item in (
            rms_output,
            norm_output,
            residual_output,
            swiglu_output,
            attention_output,
            workspace,
            query_output,
        )
    )

    for sentinel in (17.0, -23.0):
        for output in (
            rms_output,
            norm_output,
            residual_output,
            swiglu_output,
            attention_output,
            workspace,
            query_output,
        ):
            output.fill_(sentinel)
        qkv_keys.copy_(keys)
        qkv_values.copy_(values)
        qkv_length.copy_(cache_length)

        assert rms_norm_native_out(hidden, weight, 1e-5, rms_output) is rms_output
        actual_norm, actual_residual = residual_rmsnorm_native_out(
            hidden,
            residual,
            weight,
            1e-5,
            norm_output,
            residual_output,
        )
        assert actual_norm is norm_output and actual_residual is residual_output
        assert packed_swiglu_native_out(packed_mlp, swiglu_output) is swiglu_output
        assert gqa_decode_attention_native_out(
            query,
            keys,
            values,
            mask,
            0.125,
            cache_length,
            attention_output,
            workspace,
        ) is attention_output
        assert packed_qkv_rope_cache_native_out(
            packed_qkv,
            cos,
            sin,
            qkv_keys,
            qkv_values,
            qkv_length,
            query_output,
        ) is query_output

        torch.testing.assert_close(rms_output, expected_rms, rtol=0, atol=0)
        torch.testing.assert_close(norm_output, expected_norm, rtol=0, atol=0)
        torch.testing.assert_close(residual_output, expected_residual, rtol=0, atol=0)
        torch.testing.assert_close(swiglu_output, expected_swiglu, rtol=0, atol=0)
        torch.testing.assert_close(attention_output, expected_attention, rtol=0, atol=0)
        torch.testing.assert_close(query_output, expected_query, rtol=0, atol=0)
        torch.testing.assert_close(qkv_keys, expected_qkv_keys, rtol=0, atol=0)
        torch.testing.assert_close(qkv_values, expected_qkv_values, rtol=0, atol=0)
        assert int(qkv_length.item()) == 778

    assert addresses == tuple(
        item.data_ptr()
        for item in (
            rms_output,
            norm_output,
            residual_output,
            swiglu_output,
            attention_output,
            workspace,
            query_output,
        )
    )


@_CUDA_ONLY
def test_out_variants_use_non_default_stream_and_validate_outputs() -> None:
    hidden = torch.randn((1, 1, 576), device="cuda")
    residual = torch.randn_like(hidden)
    weight = torch.randn((576,), device="cuda")
    output = torch.full_like(hidden, 99.0)
    norm_output = torch.full_like(hidden, 99.0)
    residual_output = torch.full_like(hidden, 99.0)
    packed_mlp = torch.randn((1, 1, 3072), device="cuda")
    swiglu_output = torch.full((1, 1, 1536), 99.0, device="cuda")
    query = torch.randn((1, 9, 1, 64), device="cuda")
    keys = torch.randn((1, 3, 1024, 64), device="cuda")
    values = torch.randn_like(keys)
    mask = torch.zeros((1, 1, 1, 1024), device="cuda")
    length = torch.tensor(900, device="cuda")
    attention_output = torch.full_like(query, 99.0)
    workspace = torch.full((1, 9, 8, 66), 99.0, device="cuda")
    packed_qkv = torch.randn((1, 1, 960), device="cuda")
    frequencies = torch.randn((1, 1, 64), device="cuda")
    cos, sin = frequencies.cos(), frequencies.sin()
    qkv_keys, qkv_values, qkv_length = keys.clone(), values.clone(), length.clone()
    query_output = torch.full_like(query, 99.0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        rms_norm_native_out(hidden, weight, 1e-5, output)
        residual_rmsnorm_native_out(
            hidden, residual, weight, 1e-5, norm_output, residual_output
        )
        packed_swiglu_native_out(packed_mlp, swiglu_output)
        gqa_decode_attention_native_out(
            query,
            keys,
            values,
            mask,
            0.125,
            length,
            attention_output,
            workspace,
        )
        packed_qkv_rope_cache_native_out(
            packed_qkv,
            cos,
            sin,
            qkv_keys,
            qkv_values,
            qkv_length,
            query_output,
        )
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(output, rms_norm_native(hidden, weight, 1e-5))
    expected_norm, expected_residual = residual_rmsnorm_native(
        hidden, residual, weight, 1e-5
    )
    torch.testing.assert_close(norm_output, expected_norm)
    torch.testing.assert_close(residual_output, expected_residual)
    torch.testing.assert_close(swiglu_output, packed_swiglu_native(packed_mlp))
    torch.testing.assert_close(
        attention_output,
        gqa_decode_attention_native(query, keys, values, mask, 0.125, length),
    )
    expected_keys, expected_values, expected_length = (
        keys.clone(),
        values.clone(),
        length.clone(),
    )
    expected_query = packed_qkv_rope_cache_native(
        packed_qkv,
        cos,
        sin,
        expected_keys,
        expected_values,
        expected_length,
    )
    torch.testing.assert_close(query_output, expected_query)
    torch.testing.assert_close(qkv_keys, expected_keys)
    torch.testing.assert_close(qkv_values, expected_values)
    torch.testing.assert_close(qkv_length, expected_length)

    with pytest.raises(RuntimeError, match="must not alias"):
        rms_norm_native_out(hidden, weight, 1e-5, hidden)
    with pytest.raises(RuntimeError, match="output shape"):
        rms_norm_native_out(hidden, weight, 1e-5, torch.empty((1, 576), device="cuda"))


@_CUDA_ONLY
def test_out_variant_fake_tensor_and_schema_checks() -> None:
    hidden = torch.randn((1, 1, 576), device="cuda")
    weight = torch.randn((576,), device="cuda")
    output = torch.empty_like(hidden)
    result = torch.library.opcheck(
        torch.ops.flux.rmsnorm_out.default,
        (hidden, weight, 1e-5, output),
        test_utils=("test_schema", "test_faketensor"),
    )
    assert all(value == "SUCCESS" for value in result.values())


def _smollm_geometry_model() -> LlamaForCausalLM:
    config = LlamaConfig(
        hidden_size=576,
        intermediate_size=1536,
        num_hidden_layers=1,
        num_attention_heads=9,
        num_key_value_heads=3,
        head_dim=64,
        vocab_size=128,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(9302)
    return LlamaForCausalLM(config).float().eval()


@_CUDA_ONLY
def test_stable_scratch_graph_replay_has_no_stale_data_and_preserves_state() -> None:
    baseline_model = _smollm_geometry_model().cuda()
    stable_model = copy.deepcopy(baseline_model)
    enable_flux_ops(baseline_model, operators=_FULL_OPERATORS)
    enable_flux_ops(stable_model, operators=_FULL_OPERATORS)
    prompt = ((torch.arange(1278, device="cuda") * 17 + 11) % 128).unsqueeze(0)

    with torch.inference_mode():
        baseline = FluxCUDAGraphDecode.capture(
            baseline_model,
            prompt,
            max_decode_steps=3,
            warmup_steps=1,
            use_stable_buffers=False,
        )
        stable = FluxCUDAGraphDecode.capture(
            stable_model,
            prompt,
            max_decode_steps=3,
            warmup_steps=1,
        )
        assert stable.scratch is not None
        addresses = stable.stable_addresses()
        token = baseline.prefill_logits.argmax(dim=-1)
        for step, sentinel in enumerate((31.0, -47.0, 89.0), start=1):
            for tensor in (
                stable.scratch.norm_output,
                stable.scratch.residual_output,
                stable.scratch.query_output,
                stable.scratch.attention_output,
                stable.scratch.attention_workspace,
                stable.scratch.swiglu_output,
            ):
                tensor.fill_(sentinel)
            expected = baseline.replay(token)
            actual = stable.replay(token)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert stable.cache_position == baseline.cache_position == 1278 + step
            for stable_layer, baseline_layer in zip(
                stable.cache.layers, baseline.cache.layers, strict=True
            ):
                length = 1278 + step
                torch.testing.assert_close(
                    stable_layer.keys[..., :length, :],
                    baseline_layer.keys[..., :length, :],
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    stable_layer.values[..., :length, :],
                    baseline_layer.values[..., :length, :],
                    rtol=0,
                    atol=0,
                )
            token = expected.argmax(dim=-1)
        assert stable.stable_addresses() == addresses
        assert stable.memory.stable_scratch_bytes == stable.scratch.bytes


@_CUDA_ONLY
def test_short_capacity_preserves_existing_fallback_without_scratch() -> None:
    model = _smollm_geometry_model().cuda()
    enable_flux_ops(model, operators=_FULL_OPERATORS)
    prompt = torch.tensor([[1, 7, 11]], device="cuda")
    with torch.inference_mode():
        state = FluxCUDAGraphDecode.capture(model, prompt, max_decode_steps=2)
    assert state.scratch is None
    assert state.memory.stable_scratch_bytes == 0
