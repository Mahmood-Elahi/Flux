"""Correctness tests for the optional Flux SmolLM2 execution path."""

from __future__ import annotations

import copy

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from flux.model import smollm2_flux
from flux.ops import (
    gqa_decode_attention as gqa_decode_attention_reference,
    native_attention_score_softmax_is_available,
    native_gqa_decode_attention_is_available,
    native_residual_rmsnorm_is_available,
    native_rope_is_available,
    native_rmsnorm_is_available,
    native_softmax_is_available,
)


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
    """Use exact Python formulations while exercising the model adapter."""
    counts = {
        "rmsnorm": 0,
        "residual_rmsnorm": 0,
        "softmax": 0,
        "attention_score_softmax": 0,
        "rope": 0,
    }

    monkeypatch.setattr(smollm2_flux, "native_rmsnorm_is_available", lambda: True)
    monkeypatch.setattr(
        smollm2_flux,
        "native_residual_rmsnorm_is_available",
        lambda: True,
    )
    monkeypatch.setattr(smollm2_flux, "native_softmax_is_available", lambda: True)
    monkeypatch.setattr(smollm2_flux, "native_rope_is_available", lambda: True)
    monkeypatch.setattr(
        smollm2_flux,
        "native_packed_swiglu_is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        smollm2_flux,
        "native_attention_score_softmax_is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        smollm2_flux,
        "native_gqa_decode_attention_is_available",
        lambda: True,
    )

    def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        counts["rmsnorm"] += 1
        inverse_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        return weight * (x * inverse_rms)

    def residual_rmsnorm(
        hidden: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        counts["residual_rmsnorm"] += 1
        residual_out = hidden + residual
        inverse_rms = torch.rsqrt(
            residual_out.pow(2).mean(dim=-1, keepdim=True) + eps
        )
        return weight * (residual_out * inverse_rms), residual_out

    def softmax(x: torch.Tensor) -> torch.Tensor:
        counts["softmax"] += 1
        return torch.nn.functional.softmax(x, dim=-1, dtype=torch.float32)

    def attention_score_softmax(
        scores: torch.Tensor,
        mask: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        counts["attention_score_softmax"] += 1
        return torch.nn.functional.softmax(
            scores * scale + mask,
            dim=-1,
            dtype=torch.float32,
        )

    def rope(
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        counts["rope"] += 1
        return smollm2_flux.apply_rotary_pos_emb(query, key, cos, sin)

    def gqa_decode_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None,
        scale: float,
        cache_length: torch.Tensor | None,
    ) -> torch.Tensor:
        counts["gqa_decode_attention"] = counts.get("gqa_decode_attention", 0) + 1
        return gqa_decode_attention_reference(
            query, key, value, mask, scale, cache_length
        )

    monkeypatch.setattr(smollm2_flux, "rms_norm_native", rmsnorm)
    monkeypatch.setattr(smollm2_flux, "residual_rmsnorm_native", residual_rmsnorm)
    monkeypatch.setattr(smollm2_flux, "softmax_native", softmax)
    monkeypatch.setattr(
        smollm2_flux,
        "attention_score_softmax_native",
        attention_score_softmax,
    )
    monkeypatch.setattr(smollm2_flux, "rope_native", rope)
    monkeypatch.setattr(
        smollm2_flux,
        "gqa_decode_attention_native",
        gqa_decode_attention,
    )
    monkeypatch.setattr(
        smollm2_flux,
        "packed_swiglu_native",
        lambda packed: torch.nn.functional.silu(packed.chunk(2, dim=-1)[0])
        * packed.chunk(2, dim=-1)[1],
    )
    return counts


def _causal_mask(sequence_length: int) -> torch.Tensor:
    mask = torch.full(
        (1, 1, sequence_length, sequence_length),
        torch.finfo(torch.float32).min,
    )
    return torch.triu(mask, diagonal=1)


def _unique_parameter_storage_bytes(model: torch.nn.Module) -> int:
    storages: dict[int, int] = {}
    for parameter in model.parameters():
        storage = parameter.untyped_storage()
        storages.setdefault(storage.data_ptr(), storage.nbytes())
    return sum(storages.values())


@pytest.mark.parametrize("sequence_length", [1, 7, 32])
def test_packed_qkv_projection_slices_shapes_strides_and_storage(
    sequence_length: int,
) -> None:
    source = _model(1).model.layers[0].self_attn
    reference = copy.deepcopy(source)
    packed = smollm2_flux.FluxLlamaAttention(
        source,
        use_rope=False,
        use_softmax=False,
        use_packed_qkv=True,
    )
    hidden_states = torch.randn(1, sequence_length, 32)

    with torch.inference_mode():
        packed_output = packed.packed_qkv(hidden_states)
        query, key, value = packed.project_qkv(hidden_states)
        expected = (
            reference.q_proj(hidden_states)
            .view(1, sequence_length, 4, 8)
            .transpose(1, 2),
            reference.k_proj(hidden_states)
            .view(1, sequence_length, 2, 8)
            .transpose(1, 2),
            reference.v_proj(hidden_states)
            .view(1, sequence_length, 2, 8)
            .transpose(1, 2),
        )

    assert tuple(packed.packed_qkv.weight.shape) == (64, 32)
    torch.testing.assert_close(packed.packed_qkv.weight[:32], reference.q_proj.weight)
    torch.testing.assert_close(packed.packed_qkv.weight[32:48], reference.k_proj.weight)
    torch.testing.assert_close(packed.packed_qkv.weight[48:64], reference.v_proj.weight)
    packed_storage = query.untyped_storage().data_ptr()
    assert key.untyped_storage().data_ptr() == packed_storage
    assert value.untyped_storage().data_ptr() == packed_storage
    # A second call has a distinct packed output, but every returned Q/K/V
    # tensor from that call is still a view of its one allocation.
    assert packed_output.untyped_storage().data_ptr() != packed_storage
    assert query.shape == (1, 4, sequence_length, 8)
    assert key.shape == value.shape == (1, 2, sequence_length, 8)
    if sequence_length == 1:
        assert query.stride() == (32, 8, 32, 1)
        assert key.stride() == value.stride() == (16, 8, 16, 1)
    else:
        assert query.stride() == (sequence_length * 64, 8, 64, 1)
        assert key.stride() == value.stride() == (sequence_length * 64, 8, 64, 1)
    for actual, wanted in zip((query, key, value), expected, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=2e-5, atol=2e-5)


def test_packed_qkv_owns_one_weight_storage_and_preserves_standard_state_dict() -> None:
    reference = _model(2)
    standard_state = copy.deepcopy(reference.state_dict())
    expected_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in reference.parameters()
    )
    expected_storage_bytes = _unique_parameter_storage_bytes(reference)

    packed = copy.deepcopy(reference)
    smollm2_flux.enable_flux_ops(packed, operators=("qkv",))
    attention = packed.model.layers[0].self_attn

    assert isinstance(attention, smollm2_flux.FluxLlamaAttention)
    assert attention.use_packed_qkv
    assert not hasattr(attention, "q_proj")
    assert not hasattr(attention, "k_proj")
    assert not hasattr(attention, "v_proj")
    assert sum(
        parameter.numel() * parameter.element_size()
        for parameter in packed.parameters()
    ) == expected_parameter_bytes
    assert _unique_parameter_storage_bytes(packed) == expected_storage_bytes
    assert set(packed.state_dict()) == set(standard_state)
    assert not any("packed_qkv" in key for key in packed.state_dict())

    packed.load_state_dict(standard_state, strict=True)
    exported = packed.state_dict()
    restored_reference = _model(2)
    restored_reference.load_state_dict(exported, strict=True)
    for key, expected in standard_state.items():
        torch.testing.assert_close(exported[key], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            restored_reference.state_dict()[key], expected, rtol=0, atol=0
        )


def test_packed_qkv_save_pretrained_exports_an_ordinary_checkpoint(tmp_path) -> None:
    packed = _model(1)
    standard_state = copy.deepcopy(packed.state_dict())
    smollm2_flux.enable_flux_ops(packed, operators=("qkv",))

    packed.save_pretrained(tmp_path, safe_serialization=True)
    restored = LlamaForCausalLM.from_pretrained(tmp_path, local_files_only=True)

    assert not isinstance(
        restored.model.layers[0].self_attn, smollm2_flux.FluxLlamaAttention
    )
    assert set(restored.state_dict()) == set(standard_state)
    for key, expected in standard_state.items():
        torch.testing.assert_close(
            restored.state_dict()[key], expected, rtol=0, atol=0
        )


def test_packed_qkv_is_independently_opt_in_and_matches_reference(
    python_flux_ops: dict[str, int],
) -> None:
    reference = _model(2)
    packed = copy.deepcopy(reference)
    smollm2_flux.enable_flux_ops(packed, operators=("qkv",))
    input_ids = torch.tensor([[1, 17, 42, 9, 3]])

    with torch.inference_mode():
        expected_prefill = reference(input_ids, use_cache=True)
        actual_prefill = packed(input_ids, use_cache=True)
        next_token = torch.tensor([[11]])
        expected_decode = reference(
            next_token,
            past_key_values=expected_prefill.past_key_values,
            use_cache=True,
        )
        actual_decode = packed(
            next_token,
            past_key_values=actual_prefill.past_key_values,
            use_cache=True,
        )
        expected_tokens = reference.generate(
            input_ids, do_sample=False, max_new_tokens=4, use_cache=True
        )
        actual_tokens = packed.generate(
            input_ids, do_sample=False, max_new_tokens=4, use_cache=True
        )

    torch.testing.assert_close(
        actual_prefill.logits, expected_prefill.logits, rtol=2e-4, atol=2e-5
    )
    torch.testing.assert_close(
        actual_decode.logits, expected_decode.logits, rtol=2e-4, atol=2e-5
    )
    for actual_layer, expected_layer in zip(
        actual_decode.past_key_values.layers,
        expected_decode.past_key_values.layers,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_layer.keys, expected_layer.keys, rtol=2e-4, atol=2e-5
        )
        torch.testing.assert_close(
            actual_layer.values, expected_layer.values, rtol=2e-4, atol=2e-5
        )
    assert torch.equal(actual_tokens, expected_tokens)
    assert packed._flux_operator_categories == ("qkv",)
    assert packed._flux_packed_qkv_enabled
    assert smollm2_flux.flux_operator_counts(packed)["packed_qkv_modules"] == 2
    assert not any(python_flux_ops.values())


def test_packed_qkv_views_feed_flux_rope_directly(
    python_flux_ops: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _model(1)
    packed = copy.deepcopy(source)
    observed: dict[str, object] = {}

    def rope(
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        python_flux_ops["rope"] += 1
        observed["query_stride"] = query.stride()
        observed["key_stride"] = key.stride()
        observed["shared_storage"] = (
            query.untyped_storage().data_ptr() == key.untyped_storage().data_ptr()
        )
        return smollm2_flux.apply_rotary_pos_emb(query, key, cos, sin)

    monkeypatch.setattr(smollm2_flux, "rope_native", rope)
    smollm2_flux.enable_flux_ops(packed, operators=("qkv", "rope"))
    input_ids = torch.tensor([[1, 17, 42, 9, 3]])
    with torch.inference_mode():
        expected = source(input_ids, use_cache=False).logits
        actual = packed(input_ids, use_cache=False).logits

    assert observed == {
        "query_stride": (320, 8, 64, 1),
        "key_stride": (320, 8, 64, 1),
        "shared_storage": True,
    }
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)


def test_packed_qkv_composes_with_all_retained_flux_categories(
    python_flux_ops: dict[str, int],
) -> None:
    reference = _model(2)
    packed = copy.deepcopy(reference)
    operators = smollm2_flux.FLUX_OPERATOR_CATEGORIES | {
        smollm2_flux.FLUX_PACKED_QKV_CATEGORY,
        smollm2_flux.FLUX_PACKED_MLP_CATEGORY,
        smollm2_flux.FLUX_PACKED_SWIGLU_CATEGORY,
    }
    smollm2_flux.enable_flux_ops(packed, operators=operators)
    input_ids = torch.tensor([[1, 17, 42, 9, 3]])

    with torch.inference_mode():
        expected = reference(input_ids, use_cache=False).logits
        actual = packed(input_ids, use_cache=False).logits

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    assert packed._flux_packed_qkv_enabled
    assert packed._flux_packed_mlp_enabled
    assert packed._flux_packed_swiglu_enabled
    assert python_flux_ops["rope"] == 2
    assert python_flux_ops["attention_score_softmax"] == 2


@pytest.mark.parametrize("sequence_length", [1, 7, 32])
def test_packed_mlp_projection_slices_and_output_match_reference(
    sequence_length: int,
) -> None:
    source = _model(1).model.layers[0].mlp
    reference = copy.deepcopy(source)
    packed = smollm2_flux.FluxPackedLlamaMLP(source)
    hidden_states = torch.randn(1, sequence_length, 32)

    with torch.inference_mode():
        expected_gate = reference.gate_proj(hidden_states)
        expected_up = reference.up_proj(hidden_states)
        gate_up = packed.gate_up_proj(hidden_states)
        actual_gate, actual_up = gate_up.chunk(2, dim=-1)
        expected_output = reference(hidden_states)
        actual_output = packed(hidden_states)

    assert actual_gate.untyped_storage().data_ptr() == gate_up.untyped_storage().data_ptr()
    assert actual_up.untyped_storage().data_ptr() == gate_up.untyped_storage().data_ptr()
    torch.testing.assert_close(actual_gate, expected_gate, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual_up, expected_up, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual_output, expected_output, rtol=2e-5, atol=2e-5)


def test_packed_mlp_owns_one_weight_storage_and_preserves_standard_state_dict() -> None:
    reference = _model(2)
    standard_state = copy.deepcopy(reference.state_dict())
    expected_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in reference.parameters()
    )
    expected_storage_bytes = _unique_parameter_storage_bytes(reference)

    packed = copy.deepcopy(reference)
    smollm2_flux.enable_flux_ops(packed, operators=("mlp",))
    packed_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in packed.parameters()
    )
    first_mlp = packed.model.layers[0].mlp

    assert isinstance(first_mlp, smollm2_flux.FluxPackedLlamaMLP)
    assert tuple(first_mlp.gate_up_proj.weight.shape) == (128, 32)
    assert not hasattr(first_mlp, "gate_proj")
    assert not hasattr(first_mlp, "up_proj")
    assert packed_parameter_bytes == expected_parameter_bytes
    assert _unique_parameter_storage_bytes(packed) == expected_storage_bytes
    assert set(packed.state_dict()) == set(standard_state)
    assert not any("gate_up_proj" in key for key in packed.state_dict())

    # A standard checkpoint loads strictly into the packed runtime form.
    packed.load_state_dict(standard_state, strict=True)
    # Export from the packed form also loads strictly into ordinary HF Llama.
    exported = packed.state_dict()
    restored_reference = _model(2)
    restored_reference.load_state_dict(exported, strict=True)
    for key, expected in standard_state.items():
        torch.testing.assert_close(exported[key], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            restored_reference.state_dict()[key], expected, rtol=0, atol=0
        )


def test_packed_mlp_save_pretrained_exports_an_ordinary_checkpoint(tmp_path) -> None:
    packed = _model(1)
    standard_state = copy.deepcopy(packed.state_dict())
    smollm2_flux.enable_flux_ops(packed, operators=("mlp",))

    packed.save_pretrained(tmp_path, safe_serialization=True)
    restored = LlamaForCausalLM.from_pretrained(tmp_path, local_files_only=True)

    assert not isinstance(restored.model.layers[0].mlp, smollm2_flux.FluxPackedLlamaMLP)
    assert set(restored.state_dict()) == set(standard_state)
    for key, expected in standard_state.items():
        torch.testing.assert_close(restored.state_dict()[key], expected, rtol=0, atol=0)


def test_packed_mlp_layer_prefill_cache_decode_and_generation_match_reference(
    python_flux_ops: dict[str, int],
) -> None:
    reference = _model(2)
    packed = copy.deepcopy(reference)
    selected = smollm2_flux.FLUX_OPERATOR_CATEGORIES | {
        smollm2_flux.FLUX_PACKED_MLP_CATEGORY
    }
    smollm2_flux.enable_flux_ops(packed, operators=selected)

    layer_input = torch.randn(1, 7, 32)
    position_ids = torch.arange(7).unsqueeze(0)
    position_embeddings = reference.model.rotary_emb(layer_input, position_ids)
    causal_mask = _causal_mask(7)
    input_ids = torch.tensor([[1, 17, 42, 9, 3]])
    next_token = torch.tensor([[11]])

    with torch.inference_mode():
        expected_layer = reference.model.layers[0](
            layer_input,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
        )
        actual_layer = packed.model.layers[0](
            layer_input,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
        )
        expected_prefill = reference(input_ids, use_cache=True)
        actual_prefill = packed(input_ids, use_cache=True)
        expected_decode = reference(
            next_token,
            past_key_values=expected_prefill.past_key_values,
            use_cache=True,
        )
        actual_decode = packed(
            next_token,
            past_key_values=actual_prefill.past_key_values,
            use_cache=True,
        )
        expected_tokens = reference.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=4,
            use_cache=True,
        )
        actual_tokens = packed.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=4,
            use_cache=True,
        )

    torch.testing.assert_close(actual_layer, expected_layer, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(
        actual_prefill.logits, expected_prefill.logits, rtol=2e-4, atol=2e-5
    )
    torch.testing.assert_close(
        actual_decode.logits, expected_decode.logits, rtol=2e-4, atol=2e-5
    )
    assert actual_decode.past_key_values.get_seq_length() == input_ids.shape[1] + 1
    for actual_cache_layer, expected_cache_layer in zip(
        actual_decode.past_key_values.layers,
        expected_decode.past_key_values.layers,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_cache_layer.keys, expected_cache_layer.keys, rtol=2e-4, atol=2e-5
        )
        torch.testing.assert_close(
            actual_cache_layer.values,
            expected_cache_layer.values,
            rtol=2e-4,
            atol=2e-5,
        )
    assert torch.equal(actual_tokens, expected_tokens)
    assert smollm2_flux.flux_operator_counts(packed)["packed_mlp_modules"] == 2
    assert python_flux_ops["attention_score_softmax"] > 0


def test_packed_swiglu_is_separately_opt_in_and_matches_packed_mlp(
    python_flux_ops: dict[str, int],
) -> None:
    source = _model(2)
    packed = copy.deepcopy(source)
    fused = copy.deepcopy(source)
    packed_selection = {smollm2_flux.FLUX_PACKED_MLP_CATEGORY}
    fused_selection = packed_selection | {
        smollm2_flux.FLUX_PACKED_SWIGLU_CATEGORY
    }
    smollm2_flux.enable_flux_ops(packed, operators=packed_selection)
    smollm2_flux.enable_flux_ops(fused, operators=fused_selection)
    input_ids = torch.tensor([[1, 17, 42, 9, 3]])

    with torch.inference_mode():
        packed_logits = packed(input_ids=input_ids, use_cache=False).logits
        fused_logits = fused(input_ids=input_ids, use_cache=False).logits

    assert not packed.model.layers[0].mlp.use_packed_swiglu
    assert fused.model.layers[0].mlp.use_packed_swiglu
    assert not packed._flux_packed_swiglu_enabled
    assert fused._flux_packed_swiglu_enabled
    torch.testing.assert_close(fused_logits, packed_logits, rtol=0, atol=0)


def test_packed_swiglu_requires_packed_mlp_category(
    python_flux_ops: dict[str, int],
) -> None:
    with pytest.raises(ValueError, match="packed_swiglu.*requires.*mlp"):
        smollm2_flux.enable_flux_ops(
            _model(1),
            operators=(smollm2_flux.FLUX_PACKED_SWIGLU_CATEGORY,),
        )


def test_rmsnorm_substitution_reuses_weight_and_matches_reference(
    python_flux_ops: dict[str, int],
) -> None:
    model = _model(1)
    reference = copy.deepcopy(model.model.layers[0].input_layernorm)
    original = model.model.layers[0].input_layernorm
    weight = original.weight
    integrated = smollm2_flux.FluxRMSNorm(original)
    hidden_states = torch.randn(2, 5, 32)

    with torch.inference_mode():
        expected = reference(hidden_states)
        actual = integrated(hidden_states)

    assert integrated.weight is weight
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert python_flux_ops["rmsnorm"] == 1


def test_attention_preserves_causal_mask_gqa_rope_and_scale(
    python_flux_ops: dict[str, int],
) -> None:
    model = _model(1)
    reference = copy.deepcopy(model.model.layers[0].self_attn).eval()
    integrated = smollm2_flux.FluxLlamaAttention(
        model.model.layers[0].self_attn
    ).eval()
    hidden_states = torch.randn(1, 5, 32)
    position_ids = torch.arange(5).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    attention_mask = _causal_mask(5)

    with torch.inference_mode():
        expected_output, expected_probs = reference(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        actual_output, actual_probs = integrated(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )

    assert integrated.num_key_value_groups == 2
    assert integrated.scaling == 8**-0.5
    assert actual_probs.shape == (1, 4, 5, 5)
    assert torch.count_nonzero(actual_probs.triu(diagonal=1)) == 0
    torch.testing.assert_close(actual_probs, expected_probs, rtol=0, atol=0)
    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
    assert python_flux_ops["attention_score_softmax"] == 1
    assert python_flux_ops["rope"] == 1


def test_decoder_layer_matches_reference(
    python_flux_ops: dict[str, int],
) -> None:
    model = _model(1)
    reference = copy.deepcopy(model.model.layers[0]).eval()
    integrated = smollm2_flux.FluxLlamaDecoderLayer(
        model.model.layers[0]
    ).eval()
    hidden_states = torch.randn(2, 7, 32)
    position_ids = torch.arange(7).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    attention_mask = _causal_mask(7)

    with torch.inference_mode():
        expected = reference(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        actual = integrated(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert python_flux_ops == {
        "rmsnorm": 1,
        "residual_rmsnorm": 1,
        "softmax": 0,
        "attention_score_softmax": 1,
        "rope": 1,
    }


def test_enable_flux_ops_preserves_parameters_and_invokes_every_operator(
    python_flux_ops: dict[str, int],
) -> None:
    model = _model(2)
    original_input_norm_weight = model.model.layers[0].input_layernorm.weight
    original_post_norm_weight = model.model.layers[0].post_attention_layernorm.weight
    original_q_proj_weight = model.model.layers[0].self_attn.q_proj.weight
    reference = copy.deepcopy(model)
    reference_state_keys = tuple(reference.state_dict())

    returned = smollm2_flux.enable_flux_ops(model)
    counts = smollm2_flux.flux_operator_counts(model)
    input_ids = torch.tensor([[1, 17, 42, 9, 3]])
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        expected = reference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        actual = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits

    assert returned is model
    assert tuple(model.state_dict()) == reference_state_keys
    assert model.model.layers[0].input_layernorm.weight is original_input_norm_weight
    assert (
        model.model.layers[0].post_attention_layernorm.weight
        is original_post_norm_weight
    )
    assert model.model.layers[0].self_attn.q_proj.weight is original_q_proj_weight
    assert counts == {
        "decoder_layers": 2,
        "rmsnorm_modules": 5,
        "attention_modules": 2,
        "packed_mlp_modules": 0,
        "packed_qkv_modules": 0,
        "gqa_decode_attention_modules": 0,
        "packed_qkv_rope_cache_modules": 0,
    }
    # Two input norms, two fused post-attention norms, and one final norm.
    assert python_flux_ops == {
        "rmsnorm": 3,
        "residual_rmsnorm": 2,
        "softmax": 0,
        "attention_score_softmax": 2,
        "rope": 2,
    }
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_enable_flux_ops_rejects_training_and_non_fp32(
    python_flux_ops: dict[str, int],
) -> None:
    training_model = _model(1).train()
    with pytest.raises(ValueError, match="inference-only"):
        smollm2_flux.enable_flux_ops(training_model)

    half_model = _model(1).half()
    with pytest.raises(TypeError, match="torch.float32"):
        smollm2_flux.enable_flux_ops(half_model)


def test_enable_flux_ops_is_explicit_and_single_use(
    python_flux_ops: dict[str, int],
) -> None:
    model = _model(1)
    assert not hasattr(model, "_flux_ops_enabled")
    smollm2_flux.enable_flux_ops(model)
    with pytest.raises(ValueError, match="already enabled"):
        smollm2_flux.enable_flux_ops(model)


@pytest.mark.parametrize(
    ("operators", "expected_counts"),
    [
        (
            ("rmsnorm",),
            {
                "rmsnorm": 5,
                "residual_rmsnorm": 0,
                "softmax": 0,
                "attention_score_softmax": 0,
                "rope": 0,
            },
        ),
        (
            ("residual_rmsnorm",),
            {
                "rmsnorm": 0,
                "residual_rmsnorm": 2,
                "softmax": 0,
                "attention_score_softmax": 0,
                "rope": 0,
            },
        ),
        (
            ("softmax",),
            {
                "rmsnorm": 0,
                "residual_rmsnorm": 0,
                "softmax": 0,
                "attention_score_softmax": 2,
                "rope": 0,
            },
        ),
        (
            ("rope",),
            {
                "rmsnorm": 0,
                "residual_rmsnorm": 0,
                "softmax": 0,
                "attention_score_softmax": 0,
                "rope": 2,
            },
        ),
    ],
)
def test_enable_flux_ops_can_select_operator_categories(
    python_flux_ops: dict[str, int],
    operators: tuple[str, ...],
    expected_counts: dict[str, int],
) -> None:
    model = _model(2)
    reference = copy.deepcopy(model)
    smollm2_flux.enable_flux_ops(model, operators=operators)
    input_ids = torch.tensor([[1, 17, 42, 9, 3]])

    with torch.inference_mode():
        expected = reference(input_ids=input_ids, use_cache=False).logits
        actual = model(input_ids=input_ids, use_cache=False).logits

    assert python_flux_ops == expected_counts
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("operators", [(), ("unknown",)])
def test_enable_flux_ops_rejects_invalid_operator_selection(
    python_flux_ops: dict[str, int], operators: tuple[str, ...]
) -> None:
    model = _model(1)
    with pytest.raises(ValueError, match="operator categor"):
        smollm2_flux.enable_flux_ops(model, operators=operators)


def test_packed_qkv_rope_cache_requires_all_retained_dependencies(
    python_flux_ops: dict[str, int],
) -> None:
    model = _model(1)
    with pytest.raises(ValueError, match='requires the "rope", "qkv", and'):
        smollm2_flux.enable_flux_ops(
            model,
            operators=(smollm2_flux.FLUX_PACKED_QKV_ROPE_CACHE_CATEGORY,),
        )


def test_greedy_generation_with_kv_cache_matches_reference(
    python_flux_ops: dict[str, int],
) -> None:
    model = _model(2)
    reference = copy.deepcopy(model)
    smollm2_flux.enable_flux_ops(model)
    input_ids = torch.tensor([[1, 17, 42, 9]])

    with torch.inference_mode():
        expected = reference.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=4,
            use_cache=True,
        )
        actual = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=4,
            use_cache=True,
        )

    assert torch.equal(actual, expected)
    assert python_flux_ops["rmsnorm"] > 0
    assert python_flux_ops["residual_rmsnorm"] > 0
    assert python_flux_ops["softmax"] > 0
    assert python_flux_ops["attention_score_softmax"] > 0
    assert python_flux_ops["rope"] > 0


def test_gqa_decode_attention_is_separately_opt_in_and_decode_only(
    python_flux_ops: dict[str, int],
) -> None:
    source = _model(2)
    current = copy.deepcopy(source)
    fused = copy.deepcopy(source)
    smollm2_flux.enable_flux_ops(current)
    smollm2_flux.enable_flux_ops(
        fused,
        operators=smollm2_flux.FLUX_OPERATOR_CATEGORIES
        | {smollm2_flux.FLUX_GQA_DECODE_ATTENTION_CATEGORY},
    )
    prompt = torch.tensor([[1, 17, 42, 9, 3]])
    token = torch.tensor([[11]])

    with torch.inference_mode():
        current_prefill = current(prompt, use_cache=True)
        fused_prefill = fused(prompt, use_cache=True)
        assert python_flux_ops.get("gqa_decode_attention", 0) == 0
        current_decode = current(
            token,
            past_key_values=current_prefill.past_key_values,
            use_cache=True,
        )
        fused_decode = fused(
            token,
            past_key_values=fused_prefill.past_key_values,
            use_cache=True,
        )
        current_tokens = current.generate(
            prompt, do_sample=False, max_new_tokens=4, use_cache=True
        )
        fused_tokens = fused.generate(
            prompt, do_sample=False, max_new_tokens=4, use_cache=True
        )

    assert fused._flux_gqa_decode_attention_enabled
    assert not current._flux_gqa_decode_attention_enabled
    assert smollm2_flux.flux_operator_counts(fused)[
        "gqa_decode_attention_modules"
    ] == 2
    assert python_flux_ops["gqa_decode_attention"] > 0
    torch.testing.assert_close(
        fused_prefill.logits, current_prefill.logits, rtol=0, atol=0
    )
    torch.testing.assert_close(
        fused_decode.logits, current_decode.logits, rtol=2e-4, atol=2e-5
    )
    for layer_index, (fused_layer, current_layer) in enumerate(zip(
        fused_decode.past_key_values.layers,
        current_decode.past_key_values.layers,
        strict=True,
    )):
        relative_tolerance = 0 if layer_index == 0 else 2e-4
        absolute_tolerance = 0 if layer_index == 0 else 2e-5
        torch.testing.assert_close(
            fused_layer.keys,
            current_layer.keys,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
        torch.testing.assert_close(
            fused_layer.values,
            current_layer.values,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
    assert torch.equal(fused_tokens, current_tokens)


def test_gqa_decode_attention_category_falls_back_without_cached_decode(
    python_flux_ops: dict[str, int],
) -> None:
    model = _model(1)
    reference = copy.deepcopy(model)
    smollm2_flux.enable_flux_ops(
        model,
        operators=(smollm2_flux.FLUX_GQA_DECODE_ATTENTION_CATEGORY,),
    )
    input_ids = torch.tensor([[7]])

    with torch.inference_mode():
        expected = reference(input_ids=input_ids, use_cache=False).logits
        actual = model(input_ids=input_ids, use_cache=False).logits

    assert python_flux_ops.get("gqa_decode_attention", 0) == 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_gqa_decode_attention_category_falls_back_for_batch_above_one(
    python_flux_ops: dict[str, int],
) -> None:
    model = _model(1)
    smollm2_flux.enable_flux_ops(
        model,
        operators=(smollm2_flux.FLUX_GQA_DECODE_ATTENTION_CATEGORY,),
    )
    prompt = torch.tensor([[1, 17, 42], [9, 3, 28]])
    token = torch.tensor([[11], [5]])

    with torch.inference_mode():
        prefill = model(prompt, use_cache=True)
        model(token, past_key_values=prefill.past_key_values, use_cache=True)

    assert python_flux_ops.get("gqa_decode_attention", 0) == 0


def test_gqa_decode_attention_falls_back_when_probabilities_are_requested(
    python_flux_ops: dict[str, int],
) -> None:
    model = _model(1)
    smollm2_flux.enable_flux_ops(
        model,
        operators=(smollm2_flux.FLUX_GQA_DECODE_ATTENTION_CATEGORY,),
    )
    prompt = torch.tensor([[1, 17, 42]])
    token = torch.tensor([[9]])

    with torch.inference_mode():
        prefill = model(prompt, use_cache=True)
        decode = model(
            token,
            past_key_values=prefill.past_key_values,
            use_cache=True,
            output_attentions=True,
        )

    assert python_flux_ops.get("gqa_decode_attention", 0) == 0
    assert len(decode.attentions) == 1
    assert decode.attentions[0].shape == (1, 4, 1, 4)


def test_hidden_state_capture_survives_module_replacement(
    python_flux_ops: dict[str, int],
) -> None:
    model = _model(2)
    input_ids = torch.tensor([[1, 17, 42, 9]])
    with torch.inference_mode():
        before = model(
            input_ids,
            use_cache=False,
            output_hidden_states=True,
        ).hidden_states
        smollm2_flux.enable_flux_ops(model)
        after = model(
            input_ids,
            use_cache=False,
            output_hidden_states=True,
        ).hidden_states

    assert len(before) == len(after) == 3
    for actual, expected in zip(after, before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


_NATIVE_OPS_AVAILABLE = (
    native_attention_score_softmax_is_available()
    and native_rmsnorm_is_available()
    and native_residual_rmsnorm_is_available()
    and native_rope_is_available()
    and native_softmax_is_available()
)


@pytest.mark.skipif(
    not _NATIVE_OPS_AVAILABLE,
    reason="Flux native custom operators have not been built",
)
@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def test_native_full_model_matches_reference(device: str) -> None:
    model = _model(2).to(device)
    reference = copy.deepcopy(model)
    smollm2_flux.enable_flux_ops(model)
    input_ids = torch.tensor([[1, 17, 42, 9, 3]], device=device)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        expected = reference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        actual = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)


@pytest.mark.skipif(
    not _NATIVE_OPS_AVAILABLE,
    reason="Flux native custom operators have not been built",
)
@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def test_native_fused_attention_matches_current_flux_layer_and_model(
    device: str,
) -> None:
    source = _model(2).to(device)
    current = copy.deepcopy(source)
    fused = copy.deepcopy(source)
    smollm2_flux.enable_flux_ops(current, fuse_attention_scores=False)
    smollm2_flux.enable_flux_ops(fused, fuse_attention_scores=True)
    input_ids = torch.tensor([[0, 0, 1, 17, 42, 9, 3]], device=device)
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1]], device=device)

    layer_input = torch.randn((1, 7, 32), device=device)
    position_ids = torch.arange(7, device=device).unsqueeze(0)
    position_embeddings = source.model.rotary_emb(layer_input, position_ids)
    causal_mask = _causal_mask(7).to(device)
    current_attention = current.model.layers[0].self_attn
    fused_attention = fused.model.layers[0].self_attn
    with torch.inference_mode():
        current_layer, current_probs = current_attention(
            layer_input,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
        )
        fused_layer, fused_probs = fused_attention(
            layer_input,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
        )
        current_output = current(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        fused_output = fused(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )

    torch.testing.assert_close(fused_probs, current_probs, rtol=0, atol=0)
    torch.testing.assert_close(fused_layer, current_layer, rtol=0, atol=0)
    torch.testing.assert_close(
        fused_output.logits,
        current_output.logits,
        rtol=0,
        atol=0,
    )
    assert fused_output.past_key_values.get_seq_length() == input_ids.shape[1]
    for fused_cache_layer, current_cache_layer in zip(
        fused_output.past_key_values.layers,
        current_output.past_key_values.layers,
        strict=True,
    ):
        torch.testing.assert_close(
            fused_cache_layer.keys,
            current_cache_layer.keys,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            fused_cache_layer.values,
            current_cache_layer.values,
            rtol=0,
            atol=0,
        )


@pytest.mark.skipif(
    not _NATIVE_OPS_AVAILABLE,
    reason="Flux native custom operators have not been built",
)
@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def test_native_fused_greedy_generation_matches_current_flux(device: str) -> None:
    source = _model(2).to(device)
    current = copy.deepcopy(source)
    fused = copy.deepcopy(source)
    smollm2_flux.enable_flux_ops(current, fuse_attention_scores=False)
    smollm2_flux.enable_flux_ops(fused, fuse_attention_scores=True)
    input_ids = torch.tensor([[1, 17, 42, 9]], device=device)

    with torch.inference_mode():
        expected = current.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=4,
            use_cache=True,
        )
        actual = fused.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=4,
            use_cache=True,
        )

    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not (_NATIVE_OPS_AVAILABLE and native_gqa_decode_attention_is_available()),
    reason="Flux native operators including GQA decode attention have not been built",
)
@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def test_native_gqa_dynamic_decode_layer_logits_cache_and_generation(device: str) -> None:
    source = _model(2).to(device)
    current = copy.deepcopy(source)
    fused = copy.deepcopy(source)
    smollm2_flux.enable_flux_ops(current)
    smollm2_flux.enable_flux_ops(
        fused,
        operators=smollm2_flux.FLUX_OPERATOR_CATEGORIES
        | {smollm2_flux.FLUX_GQA_DECODE_ATTENTION_CATEGORY},
    )
    prompt = torch.tensor([[1, 17, 42, 9, 3]], device=device)
    token = torch.tensor([[11]], device=device)

    with torch.inference_mode():
        current_prefill = current(prompt, use_cache=True)
        fused_prefill = fused(prompt, use_cache=True)
        current_decode = current(
            token, past_key_values=current_prefill.past_key_values, use_cache=True
        )
        fused_decode = fused(
            token, past_key_values=fused_prefill.past_key_values, use_cache=True
        )
        current_tokens = current.generate(
            prompt, do_sample=False, max_new_tokens=4, use_cache=True
        )
        fused_tokens = fused.generate(
            prompt, do_sample=False, max_new_tokens=4, use_cache=True
        )

    torch.testing.assert_close(
        fused_decode.logits, current_decode.logits, rtol=2e-4, atol=2e-5
    )
    for layer_index, (fused_layer, current_layer) in enumerate(zip(
        fused_decode.past_key_values.layers,
        current_decode.past_key_values.layers,
        strict=True,
    )):
        relative_tolerance = 0 if layer_index == 0 else 2e-4
        absolute_tolerance = 0 if layer_index == 0 else 2e-5
        torch.testing.assert_close(
            fused_layer.keys,
            current_layer.keys,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
        torch.testing.assert_close(
            fused_layer.values,
            current_layer.values,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
    assert torch.equal(fused_tokens, current_tokens)
