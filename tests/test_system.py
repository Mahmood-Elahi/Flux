"""Whole-model Hugging Face, Flux eager, and native-runtime validation."""

from __future__ import annotations

import copy

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from flux.model import smollm2_flux
from flux.ops import (
    gqa_decode_attention,
    native_attention_score_softmax_is_available,
    native_gqa_decode_attention_is_available,
    native_residual_rmsnorm_is_available,
    native_rmsnorm_is_available,
    native_rope_is_available,
    native_softmax_is_available,
)
from flux.runtime import (
    NativeSmolLM2Decode,
    NativeSmolLM2Prefill,
    native_smollm2_greedy_generate,
    native_smollm2_prefill_is_available,
    native_smollm2_runtime_is_available,
)

RTOL = 2e-4
ATOL = 2e-5


def _config(num_hidden_layers: int = 2) -> LlamaConfig:
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=128,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
    )
    config._attn_implementation = "eager"
    return config


def _model(num_hidden_layers: int = 2) -> LlamaForCausalLM:
    torch.manual_seed(1234)
    return LlamaForCausalLM(_config(num_hidden_layers)).float().eval()


@pytest.fixture
def python_flux_ops(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Exercise model integration with exact Python formulations."""
    counts = {
        "rmsnorm": 0,
        "residual_rmsnorm": 0,
        "attention_score_softmax": 0,
        "rope": 0,
        "gqa_decode_attention": 0,
    }
    for name in (
        "native_rmsnorm_is_available",
        "native_residual_rmsnorm_is_available",
        "native_softmax_is_available",
        "native_rope_is_available",
        "native_packed_swiglu_is_available",
        "native_packed_gate_up_gemv_is_available",
        "native_attention_score_softmax_is_available",
        "native_gqa_decode_attention_is_available",
    ):
        monkeypatch.setattr(smollm2_flux, name, lambda: True)

    def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        counts["rmsnorm"] += 1
        return weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

    def residual_rmsnorm(
        hidden: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        counts["residual_rmsnorm"] += 1
        residual_out = hidden + residual
        normalized = weight * residual_out * torch.rsqrt(
            residual_out.pow(2).mean(dim=-1, keepdim=True) + eps
        )
        return normalized, residual_out

    def attention_softmax(
        scores: torch.Tensor, mask: torch.Tensor, scale: float
    ) -> torch.Tensor:
        counts["attention_score_softmax"] += 1
        return torch.softmax(scores * scale + mask, dim=-1, dtype=torch.float32)

    def rope(
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        counts["rope"] += 1
        return smollm2_flux.apply_rotary_pos_emb(query, key, cos, sin)

    def gqa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None,
        scale: float,
        cache_length: torch.Tensor | None,
    ) -> torch.Tensor:
        counts["gqa_decode_attention"] += 1
        return gqa_decode_attention(query, key, value, mask, scale, cache_length)

    monkeypatch.setattr(smollm2_flux, "rms_norm_native", rmsnorm)
    monkeypatch.setattr(smollm2_flux, "residual_rmsnorm_native", residual_rmsnorm)
    monkeypatch.setattr(smollm2_flux, "attention_score_softmax_native", attention_softmax)
    monkeypatch.setattr(smollm2_flux, "rope_native", rope)
    monkeypatch.setattr(smollm2_flux, "gqa_decode_attention_native", gqa)
    monkeypatch.setattr(
        smollm2_flux,
        "packed_swiglu_native",
        lambda packed: torch.nn.functional.silu(packed.chunk(2, dim=-1)[0])
        * packed.chunk(2, dim=-1)[1],
    )
    return counts


def test_final_flux_configuration_is_the_complete_retained_set() -> None:
    assert smollm2_flux.FINAL_FLUX_OPERATOR_CATEGORIES == {
        "rmsnorm",
        "residual_rmsnorm",
        "rope",
        "softmax",
        "mlp",
        "packed_swiglu",
        "qkv",
        "gqa_decode_attention",
        "packed_qkv_rope_cache",
        "cublaslt_projection",
        "fused_gate_up_swiglu",
    }


def test_packed_projection_weight_view_and_output_semantics() -> None:
    source = _model(1)
    attention_reference = copy.deepcopy(source.model.layers[0].self_attn)
    attention = smollm2_flux.FluxLlamaAttention(
        source.model.layers[0].self_attn,
        use_rope=False,
        use_softmax=False,
        use_packed_qkv=True,
    )
    hidden = torch.randn(1, 7, 32)
    with torch.inference_mode():
        query, key, value = attention.project_qkv(hidden)
        expected_q = attention_reference.q_proj(hidden).view(1, 7, 4, 8).transpose(1, 2)
        expected_k = attention_reference.k_proj(hidden).view(1, 7, 2, 8).transpose(1, 2)
        expected_v = attention_reference.v_proj(hidden).view(1, 7, 2, 8).transpose(1, 2)
    assert tuple(attention.packed_qkv.weight.shape) == (64, 32)
    torch.testing.assert_close(attention.packed_qkv.weight[:32], attention_reference.q_proj.weight)
    torch.testing.assert_close(attention.packed_qkv.weight[32:48], attention_reference.k_proj.weight)
    torch.testing.assert_close(attention.packed_qkv.weight[48:], attention_reference.v_proj.weight)
    storage = query.untyped_storage().data_ptr()
    assert key.untyped_storage().data_ptr() == storage
    assert value.untyped_storage().data_ptr() == storage
    for actual, expected in zip((query, key, value), (expected_q, expected_k, expected_v), strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    mlp_reference = copy.deepcopy(source.model.layers[0].mlp)
    mlp = smollm2_flux.FluxPackedLlamaMLP(source.model.layers[0].mlp)
    with torch.inference_mode():
        packed = mlp.gate_up_proj(hidden)
        gate, up = packed.chunk(2, dim=-1)
        actual = mlp(hidden)
        expected = mlp_reference(hidden)
    assert gate.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
    assert up.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
    torch.testing.assert_close(gate, mlp_reference.gate_proj(hidden), rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(up, mlp_reference.up_proj(hidden), rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_packed_weights_preserve_exact_state_dict_and_checkpoint(tmp_path) -> None:
    reference = _model(2)
    expected_state = copy.deepcopy(reference.state_dict())
    packed = copy.deepcopy(reference)
    smollm2_flux.enable_flux_ops(packed, operators=("qkv", "mlp"))
    assert not hasattr(packed.model.layers[0].self_attn, "q_proj")
    assert not hasattr(packed.model.layers[0].mlp, "gate_proj")
    assert set(packed.state_dict()) == set(expected_state)
    assert not any("packed_qkv" in name or "gate_up_proj" in name for name in packed.state_dict())
    for name, expected in expected_state.items():
        torch.testing.assert_close(packed.state_dict()[name], expected, rtol=0, atol=0)
    packed.load_state_dict(expected_state, strict=True)
    packed.save_pretrained(tmp_path, safe_serialization=True)
    restored = LlamaForCausalLM.from_pretrained(tmp_path, local_files_only=True)
    assert set(restored.state_dict()) == set(expected_state)
    for name, expected in expected_state.items():
        torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)


def test_model_transformation_and_eager_hf_equivalence(
    python_flux_ops: dict[str, int],
) -> None:
    reference = _model(2)
    flux = copy.deepcopy(reference)
    original_weight = flux.model.layers[0].input_layernorm.weight
    selected = smollm2_flux.FLUX_OPERATOR_CATEGORIES | {
        smollm2_flux.FLUX_PACKED_QKV_CATEGORY,
        smollm2_flux.FLUX_PACKED_MLP_CATEGORY,
        smollm2_flux.FLUX_PACKED_SWIGLU_CATEGORY,
    }
    returned = smollm2_flux.enable_flux_ops(flux, operators=selected)
    input_ids = torch.tensor([[1, 17, 42, 9, 3]])
    with torch.inference_mode():
        expected = reference(input_ids=input_ids, use_cache=True)
        actual = flux(input_ids=input_ids, use_cache=True)
    assert returned is flux
    assert flux.model.layers[0].input_layernorm.weight is original_weight
    assert flux._flux_operator_categories == tuple(sorted(selected))
    counts = smollm2_flux.flux_operator_counts(flux)
    assert counts["decoder_layers"] == 2
    assert counts["packed_qkv_modules"] == counts["packed_mlp_modules"] == 2
    assert python_flux_ops["rmsnorm"] == 3
    assert python_flux_ops["residual_rmsnorm"] == 2
    assert python_flux_ops["attention_score_softmax"] == 2
    assert python_flux_ops["rope"] == 2
    torch.testing.assert_close(actual.logits, expected.logits, rtol=RTOL, atol=ATOL)
    for actual_layer, expected_layer in zip(
        actual.past_key_values.layers, expected.past_key_values.layers, strict=True
    ):
        torch.testing.assert_close(actual_layer.keys, expected_layer.keys, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(actual_layer.values, expected_layer.values, rtol=RTOL, atol=ATOL)


def test_gqa_decode_category_is_opt_in_and_decode_only(
    python_flux_ops: dict[str, int],
) -> None:
    reference = _model(2)
    flux = copy.deepcopy(reference)
    selected = smollm2_flux.FLUX_OPERATOR_CATEGORIES | {
        smollm2_flux.FLUX_GQA_DECODE_ATTENTION_CATEGORY
    }
    smollm2_flux.enable_flux_ops(flux, operators=selected)
    prompt = torch.tensor([[1, 17, 42, 9, 3]])
    token = torch.tensor([[11]])
    with torch.inference_mode():
        expected_prefill = reference(prompt, use_cache=True)
        actual_prefill = flux(prompt, use_cache=True)
        assert python_flux_ops["gqa_decode_attention"] == 0
        expected_decode = reference(
            token, past_key_values=expected_prefill.past_key_values, use_cache=True
        )
        actual_decode = flux(
            token, past_key_values=actual_prefill.past_key_values, use_cache=True
        )
    assert flux._flux_gqa_decode_attention_enabled
    assert python_flux_ops["gqa_decode_attention"] == 2
    torch.testing.assert_close(actual_prefill.logits, expected_prefill.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual_decode.logits, expected_decode.logits, rtol=RTOL, atol=ATOL)


def test_enable_flux_ops_public_error_contract(python_flux_ops: dict[str, int]) -> None:
    del python_flux_ops
    with pytest.raises(ValueError, match="inference-only"):
        smollm2_flux.enable_flux_ops(_model(1).train())
    with pytest.raises(TypeError, match="torch.float32"):
        smollm2_flux.enable_flux_ops(_model(1).half())
    for operators in ((), ("unknown",)):
        with pytest.raises(ValueError, match="operator categor"):
            smollm2_flux.enable_flux_ops(_model(1), operators=operators)
    dependencies = (
        ((smollm2_flux.FLUX_PACKED_SWIGLU_CATEGORY,), "requires.*mlp"),
        ((smollm2_flux.FLUX_PACKED_QKV_ROPE_CACHE_CATEGORY,), 'requires the "rope", "qkv", and'),
        ((smollm2_flux.FLUX_CUBLASLT_PROJECTION_CATEGORY,), 'requires the "packed_qkv_rope_cache"'),
        ((smollm2_flux.FLUX_FUSED_GATE_UP_SWIGLU_CATEGORY,), "requires.*mlp.*packed_swiglu"),
    )
    for operators, message in dependencies:
        with pytest.raises(ValueError, match=message):
            smollm2_flux.enable_flux_ops(_model(1), operators=operators)
    model = _model(1)
    smollm2_flux.enable_flux_ops(model)
    with pytest.raises(ValueError, match="already enabled"):
        smollm2_flux.enable_flux_ops(model)


NATIVE_OPS_AVAILABLE = all(
    check()
    for check in (
        native_attention_score_softmax_is_available,
        native_gqa_decode_attention_is_available,
        native_residual_rmsnorm_is_available,
        native_rmsnorm_is_available,
        native_rope_is_available,
        native_softmax_is_available,
    )
)
NATIVE_RUNTIME_AVAILABLE = (
    torch.cuda.is_available()
    and NATIVE_OPS_AVAILABLE
    and native_smollm2_runtime_is_available()
    and native_smollm2_prefill_is_available()
)
CUDA_RUNTIME_ONLY = pytest.mark.skipif(
    not NATIVE_RUNTIME_AVAILABLE,
    reason="CUDA and the native SmolLM2 runtimes are required",
)


def _runtime_config() -> LlamaConfig:
    config = LlamaConfig(
        hidden_size=576,
        intermediate_size=1536,
        num_hidden_layers=30,
        num_attention_heads=9,
        num_key_value_heads=3,
        head_dim=64,
        vocab_size=128,
        max_position_embeddings=8192,
        rms_norm_eps=1e-5,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
    )
    config._attn_implementation = "eager"
    return config


def _reference_and_flux_models() -> tuple[LlamaForCausalLM, LlamaForCausalLM]:
    torch.manual_seed(2701)
    reference = LlamaForCausalLM(_runtime_config()).float().cuda().eval()
    flux = LlamaForCausalLM(_runtime_config()).float().cuda().eval()
    flux.load_state_dict(reference.state_dict())
    smollm2_flux.enable_flux_ops(
        flux, operators=smollm2_flux.FINAL_FLUX_OPERATOR_CATEGORIES
    )
    return reference, flux


def _runtime_model() -> LlamaForCausalLM:
    return _reference_and_flux_models()[1]


def _ids(length: int) -> torch.Tensor:
    return (torch.arange(length, device="cuda").unsqueeze(0) * 11 + 3) % 128


@CUDA_RUNTIME_ONLY
def test_native_prefill_decode_matches_hf_flux_and_all_layer_caches() -> None:
    reference, flux = _reference_and_flux_models()
    input_ids = _ids(17)
    with torch.inference_mode():
        reference_output = reference(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        flux_output = flux(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        native = NativeSmolLM2Prefill.capture(flux, input_ids, max_decode_steps=3)
    torch.testing.assert_close(flux_output.logits, reference_output.logits, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(native.logits, reference_output.logits, rtol=RTOL, atol=ATOL)
    assert native.prompt_length == native.cache_position == native.cache_length == 17
    assert native.key_cache.shape == native.value_cache.shape == (30, 1, 3, 20, 64)
    for layer_index, (reference_layer, flux_layer) in enumerate(zip(
        reference_output.past_key_values.layers,
        flux_output.past_key_values.layers,
        strict=True,
    )):
        torch.testing.assert_close(flux_layer.keys, reference_layer.keys, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(flux_layer.values, reference_layer.values, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(native.key_cache[layer_index, ..., :17, :], reference_layer.keys, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(native.value_cache[layer_index, ..., :17, :], reference_layer.values, rtol=RTOL, atol=ATOL)

    reference_token = reference_output.logits.argmax(dim=-1)
    flux_token = flux_output.logits.argmax(dim=-1)
    native_token = native.logits.argmax(dim=-1)
    for step in range(3):
        with torch.inference_mode():
            reference_output = reference(
                input_ids=reference_token,
                past_key_values=reference_output.past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            flux_output = flux(
                input_ids=flux_token,
                past_key_values=flux_output.past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            native_logits = native.replay(native_token)
        torch.testing.assert_close(flux_output.logits, reference_output.logits, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(native_logits, reference_output.logits, rtol=RTOL, atol=ATOL)
        reference_token = reference_output.logits.argmax(dim=-1)
        flux_token = flux_output.logits.argmax(dim=-1)
        native_token = native_logits.argmax(dim=-1)
        assert reference_token.equal(flux_token) and flux_token.equal(native_token)
        assert native.cache_position == native.cache_length == 18 + step

    length = 20
    for layer_index, (reference_layer, flux_layer) in enumerate(zip(
        reference_output.past_key_values.layers,
        flux_output.past_key_values.layers,
        strict=True,
    )):
        torch.testing.assert_close(flux_layer.keys, reference_layer.keys, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(flux_layer.values, reference_layer.values, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(native.key_cache[layer_index, ..., :length, :], reference_layer.keys, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(native.value_cache[layer_index, ..., :length, :], reference_layer.values, rtol=RTOL, atol=ATOL)
    with pytest.raises(RuntimeError, match="exhausted"):
        native.replay(native_token)


@CUDA_RUNTIME_ONLY
def test_native_runtime_reset_and_prefill_reuse_public_apis() -> None:
    model = _runtime_model()
    first_ids = _ids(9)
    second_ids = (first_ids + 7) % 128
    native = NativeSmolLM2Prefill.capture(model, first_ids, max_decode_steps=1)
    with torch.inference_mode():
        actual = native.prefill(second_ids).clone()
        expected = model(input_ids=second_ids, use_cache=True, logits_to_keep=1)
    torch.testing.assert_close(actual, expected.logits, rtol=RTOL, atol=ATOL)
    for layer_index, layer in enumerate(expected.past_key_values.layers):
        torch.testing.assert_close(native.key_cache[layer_index, ..., :9, :], layer.keys, rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(native.value_cache[layer_index, ..., :9, :], layer.values, rtol=RTOL, atol=ATOL)

    decode = NativeSmolLM2Decode.capture(model, _ids(4), max_decode_steps=2)
    token = decode.prefill_logits.argmax(dim=-1)
    decode.reset(token, decode._source_key_caches, decode._source_value_caches, cache_position=4)
    assert decode.cache_position == decode.cache_length == 4
    with pytest.raises(RuntimeError, match="one K/V tensor per layer"):
        decode.reset(token, decode._source_key_caches[:-1], decode._source_value_caches, cache_position=4)


@CUDA_RUNTIME_ONLY
def test_native_prefill_public_input_errors() -> None:
    model = _runtime_model()
    ids = _ids(5)
    with pytest.raises(ValueError, match="non-negative"):
        NativeSmolLM2Prefill.capture(model, ids, max_decode_steps=-1)
    with pytest.raises(ValueError, match="batch of one"):
        NativeSmolLM2Prefill.capture(model, ids.expand(2, -1))
    with pytest.raises(ValueError, match="vocabulary"):
        NativeSmolLM2Prefill.capture(model, torch.full_like(ids, 128))
    native = NativeSmolLM2Prefill.capture(model, ids)
    with pytest.raises(ValueError, match="prompt shape"):
        native.prefill(ids[:, :-1])


@CUDA_RUNTIME_ONLY
def test_native_multitoken_greedy_generation_matches_hf_exactly() -> None:
    reference, flux = _reference_and_flux_models()
    input_ids = _ids(7)
    with torch.inference_mode():
        expected = reference.generate(
            input_ids=input_ids,
            do_sample=False,
            num_beams=1,
            max_new_tokens=8,
            use_cache=True,
            pad_token_id=0,
        )
        python_runtime = NativeSmolLM2Prefill.capture(
            flux, input_ids, max_decode_steps=7
        )
        token = python_runtime.logits.argmax(dim=-1)
        python_tokens = [input_ids, token]
        for _ in range(7):
            token = python_runtime.replay(token).argmax(dim=-1)
            python_tokens.append(token)
        python_orchestrated = torch.cat(python_tokens, dim=-1)

        native = NativeSmolLM2Prefill.capture(flux, input_ids, max_decode_steps=7)
        addresses = native.stable_addresses()
        generated = native.generate_greedy(8)
        actual = torch.cat((input_ids, generated), dim=-1)
    assert actual.shape == (1, 15)
    assert python_orchestrated.equal(expected)
    assert actual.equal(expected)
    assert generated[:, :1].equal(native.prefill_logits.argmax(dim=-1))
    assert native.cache_position == input_ids.shape[1] + 7
    assert native.cache_length == input_ids.shape[1] + 7
    assert native.generation_step == 8
    assert native.stable_addresses() == addresses
    with pytest.raises(RuntimeError, match="prefill or reset"):
        native.generate_greedy(1)

    with torch.inference_mode():
        native.prefill(input_ids)
        repeated = native.generate_greedy(8).clone()
    assert repeated.equal(expected[:, input_ids.shape[1] :])
    assert native.stable_addresses() == addresses
