"""Correctness tests for the optional Flux SmolLM2 execution path."""

from __future__ import annotations

import copy

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from flux.model import smollm2_flux
from flux.ops import (
    native_residual_rmsnorm_is_available,
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
    counts = {"rmsnorm": 0, "residual_rmsnorm": 0, "softmax": 0}

    monkeypatch.setattr(smollm2_flux, "native_rmsnorm_is_available", lambda: True)
    monkeypatch.setattr(
        smollm2_flux,
        "native_residual_rmsnorm_is_available",
        lambda: True,
    )
    monkeypatch.setattr(smollm2_flux, "native_softmax_is_available", lambda: True)

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

    monkeypatch.setattr(smollm2_flux, "rms_norm_native", rmsnorm)
    monkeypatch.setattr(smollm2_flux, "residual_rmsnorm_native", residual_rmsnorm)
    monkeypatch.setattr(smollm2_flux, "softmax_native", softmax)
    return counts


def _causal_mask(sequence_length: int) -> torch.Tensor:
    mask = torch.full(
        (1, 1, sequence_length, sequence_length),
        torch.finfo(torch.float32).min,
    )
    return torch.triu(mask, diagonal=1)


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
    assert python_flux_ops["softmax"] == 1


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
        "softmax": 1,
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
    }
    # Two input norms, two fused post-attention norms, and one final norm.
    assert python_flux_ops == {
        "rmsnorm": 3,
        "residual_rmsnorm": 2,
        "softmax": 2,
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
    native_rmsnorm_is_available()
    and native_residual_rmsnorm_is_available()
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
