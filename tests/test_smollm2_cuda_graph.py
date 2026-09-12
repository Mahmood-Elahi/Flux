"""Correctness tests for opt-in fixed-capacity CUDA-Graph decode."""

from __future__ import annotations

import copy

import pytest
import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

from flux.model import smollm2_flux

from flux.model.smollm2_cuda_graph import (
    FluxCUDAGraphDecode,
    cuda_graph_greedy_generate,
)
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
    native_gqa_decode_attention_is_available,
    native_packed_swiglu_is_available,
    native_packed_qkv_rope_cache_is_available,
    native_residual_rmsnorm_is_available,
    native_rope_is_available,
    native_rmsnorm_is_available,
    native_softmax_is_available,
)


_FULL_DECODE_OPERATORS = FLUX_OPERATOR_CATEGORIES | {
    FLUX_PACKED_MLP_CATEGORY,
    FLUX_PACKED_QKV_CATEGORY,
    FLUX_PACKED_SWIGLU_CATEGORY,
    FLUX_GQA_DECODE_ATTENTION_CATEGORY,
}


def _config() -> LlamaConfig:
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
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


def _model() -> LlamaForCausalLM:
    torch.manual_seed(1234)
    return LlamaForCausalLM(_config()).float().eval()


def _smollm2_geometry_model() -> LlamaForCausalLM:
    config = LlamaConfig(
        hidden_size=576,
        intermediate_size=64,
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
    torch.manual_seed(4321)
    return LlamaForCausalLM(config).float().eval()


def test_decode_state_requires_capture_factory() -> None:
    with pytest.raises(TypeError, match="capture"):
        FluxCUDAGraphDecode()


def test_capture_requires_flux_enabled_model() -> None:
    model = _model()
    input_ids = torch.tensor([[1, 2, 3]])
    with pytest.raises((RuntimeError, ValueError), match="CUDA|Flux"):
        FluxCUDAGraphDecode.capture(model, input_ids, max_decode_steps=1)


_NATIVE_CUDA_AVAILABLE = (
    torch.cuda.is_available()
    and native_rmsnorm_is_available()
    and native_residual_rmsnorm_is_available()
    and native_rope_is_available()
    and native_softmax_is_available()
    and native_packed_swiglu_is_available()
)


@pytest.mark.skipif(
    not _NATIVE_CUDA_AVAILABLE,
    reason="CUDA and all Flux native custom operators are required",
)
@pytest.mark.parametrize("mlp_path", ["standard", "packed", "fused"])
@pytest.mark.parametrize("packed_qkv", [False, True])
def test_repeated_cuda_graph_decode_matches_eager_flux(
    mlp_path: str, packed_qkv: bool
) -> None:
    eager = _model().cuda()
    graph_model = copy.deepcopy(eager)
    operators = FLUX_OPERATOR_CATEGORIES
    if mlp_path in {"packed", "fused"}:
        operators = operators | {"mlp"}
    if mlp_path == "fused":
        operators = operators | {"packed_swiglu"}
    if packed_qkv:
        operators = operators | {"qkv"}
    enable_flux_ops(eager, operators=operators)
    enable_flux_ops(graph_model, operators=operators)
    prompt = torch.tensor([[1, 17, 42, 9, 3, 28, 11, 5]], device="cuda")

    with torch.inference_mode():
        eager_output = eager(input_ids=prompt, use_cache=True, logits_to_keep=1)
        eager_cache = eager_output.past_key_values
        assert isinstance(eager_cache, DynamicCache)
        state = FluxCUDAGraphDecode.capture(
            graph_model,
            prompt,
            max_decode_steps=8,
            warmup_steps=2,
        )
        torch.testing.assert_close(
            state.prefill_logits,
            eager_output.logits,
            rtol=2e-4,
            atol=2e-5,
        )
        token = eager_output.logits.argmax(dim=-1)
        addresses = state.stable_addresses()
        qkv_addresses = (
            tuple(
                layer.self_attn.packed_qkv.weight.data_ptr()
                for layer in graph_model.model.layers
            )
            if packed_qkv
            else ()
        )

        for step in range(8):
            eager_output = eager(
                input_ids=token,
                past_key_values=eager_cache,
                use_cache=True,
                logits_to_keep=1,
            )
            graph_logits = state.replay(token)
            torch.testing.assert_close(
                graph_logits,
                eager_output.logits,
                rtol=2e-4,
                atol=2e-5,
            )
            assert torch.equal(
                graph_logits.argmax(dim=-1),
                eager_output.logits.argmax(dim=-1),
            )
            expected_length = prompt.shape[1] + step + 1
            assert state.cache_position == expected_length
            assert int(state.cache.get_seq_length().item()) == expected_length
            for graph_layer, eager_layer in zip(
                state.cache.layers, eager_cache.layers, strict=True
            ):
                torch.testing.assert_close(
                    graph_layer.keys[..., :expected_length, :],
                    eager_layer.keys,
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    graph_layer.values[..., :expected_length, :],
                    eager_layer.values,
                    rtol=0,
                    atol=0,
                )
            token = eager_output.logits.argmax(dim=-1)

        assert state.stable_addresses() == addresses
        if packed_qkv:
            assert tuple(
                layer.self_attn.packed_qkv.weight.data_ptr()
                for layer in graph_model.model.layers
            ) == qkv_addresses
        assert state.cache_bytes > 0
        with pytest.raises(RuntimeError, match="exhausted"):
            state.replay(token)


@pytest.mark.skipif(
    not _NATIVE_CUDA_AVAILABLE,
    reason="CUDA and all Flux native custom operators are required",
)
def test_eight_token_graph_greedy_continuation_matches_reference() -> None:
    reference = _model().cuda()
    graph_model = copy.deepcopy(reference)
    enable_flux_ops(graph_model)
    prompt = torch.tensor([[1, 17, 42, 9, 3, 28, 11, 5]], device="cuda")

    with torch.inference_mode():
        expected = reference.generate(
            prompt,
            do_sample=False,
            max_new_tokens=8,
            use_cache=True,
        )
        actual = cuda_graph_greedy_generate(
            graph_model,
            prompt,
            max_new_tokens=8,
            warmup_steps=2,
        )

    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not (_NATIVE_CUDA_AVAILABLE and native_gqa_decode_attention_is_available()),
    reason="CUDA and all Flux native operators including GQA decode are required",
)
def test_gqa_decode_cuda_graph_replay_matches_current_flux_and_preserves_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force the fused operator for this tiny correctness model. Production
    # retains the measured short-StaticCache fallback threshold.
    monkeypatch.setattr(smollm2_flux, "_GQA_DECODE_GRAPH_MINIMUM_CAPACITY", 1)
    current = _model().cuda()
    fused = copy.deepcopy(current)
    enable_flux_ops(current)
    enable_flux_ops(
        fused,
        operators=FLUX_OPERATOR_CATEGORIES | {FLUX_GQA_DECODE_ATTENTION_CATEGORY},
    )
    prompt = torch.tensor([[1, 17, 42, 9, 3, 28, 11, 5]], device="cuda")

    with torch.inference_mode():
        current_state = FluxCUDAGraphDecode.capture(
            current, prompt, max_decode_steps=8, warmup_steps=2
        )
        fused_state = FluxCUDAGraphDecode.capture(
            fused, prompt, max_decode_steps=8, warmup_steps=2
        )
        token = current_state.prefill_logits.argmax(dim=-1)
        addresses = fused_state.stable_addresses()
        for step in range(8):
            current_logits = current_state.replay(token)
            fused_logits = fused_state.replay(token)
            torch.testing.assert_close(
                fused_logits, current_logits, rtol=2e-4, atol=2e-5
            )
            assert torch.equal(
                fused_logits.argmax(dim=-1), current_logits.argmax(dim=-1)
            )
            expected_length = prompt.shape[1] + step + 1
            assert fused_state.cache_position == expected_length
            for layer_index, (fused_layer, current_layer) in enumerate(zip(
                fused_state.cache.layers, current_state.cache.layers, strict=True
            )):
                relative_tolerance = 0 if layer_index == 0 else 2e-4
                absolute_tolerance = 0 if layer_index == 0 else 2e-5
                torch.testing.assert_close(
                    fused_layer.keys[..., :expected_length, :],
                    current_layer.keys[..., :expected_length, :],
                    rtol=relative_tolerance,
                    atol=absolute_tolerance,
                )
                torch.testing.assert_close(
                    fused_layer.values[..., :expected_length, :],
                    current_layer.values[..., :expected_length, :],
                    rtol=relative_tolerance,
                    atol=absolute_tolerance,
                )
            token = current_logits.argmax(dim=-1)

    assert fused_state.stable_addresses() == addresses


@pytest.mark.skipif(
    not (
        _NATIVE_CUDA_AVAILABLE
        and native_gqa_decode_attention_is_available()
        and native_packed_qkv_rope_cache_is_available()
    ),
    reason="CUDA and all fused post-QKV dependencies are required",
)
def test_fused_packed_qkv_rope_cache_graph_matches_retained_path() -> None:
    current = _smollm2_geometry_model().cuda()
    fused = copy.deepcopy(current)
    enable_flux_ops(current, operators=_FULL_DECODE_OPERATORS)
    enable_flux_ops(
        fused,
        operators=_FULL_DECODE_OPERATORS
        | {FLUX_PACKED_QKV_ROPE_CACHE_CATEGORY},
    )
    prompt = torch.tensor([[1, 17, 42, 9, 3, 28, 11, 5]], device="cuda")

    with torch.inference_mode():
        # Capacity 1281 activates the retained native-GQA crossover and the
        # specialized fused post-QKV path while keeping the logical context 8.
        current_state = FluxCUDAGraphDecode.capture(
            current, prompt, max_decode_steps=1273, warmup_steps=2
        )
        fused_state = FluxCUDAGraphDecode.capture(
            fused, prompt, max_decode_steps=1273, warmup_steps=2
        )
        token = current_state.prefill_logits.argmax(dim=-1)
        addresses = fused_state.stable_addresses()
        for step in range(3):
            expected = current_state.replay(token)
            actual = fused_state.replay(token)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))
            valid_length = prompt.shape[1] + step + 1
            torch.testing.assert_close(
                fused_state.cache.layers[0].keys[..., :valid_length, :],
                current_state.cache.layers[0].keys[..., :valid_length, :],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                fused_state.cache.layers[0].values[..., :valid_length, :],
                current_state.cache.layers[0].values[..., :valid_length, :],
                rtol=0,
                atol=0,
            )
            token = expected.argmax(dim=-1)

    assert fused_state.stable_addresses() == addresses
    assert fused_state.cache_position == current_state.cache_position == 11


@pytest.mark.skipif(
    not (
        _NATIVE_CUDA_AVAILABLE
        and native_gqa_decode_attention_is_available()
        and native_packed_qkv_rope_cache_is_available()
    ),
    reason="CUDA and all fused post-QKV dependencies are required",
)
def test_fused_packed_qkv_rope_cache_falls_back_for_dynamic_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _smollm2_geometry_model().cuda()
    enable_flux_ops(
        model,
        operators=_FULL_DECODE_OPERATORS
        | {FLUX_PACKED_QKV_ROPE_CACHE_CATEGORY},
    )
    calls = 0
    original = smollm2_flux.packed_qkv_rope_cache_native

    def counted(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(smollm2_flux, "packed_qkv_rope_cache_native", counted)
    prompt = torch.tensor([[1, 17, 42, 9]], device="cuda")
    with torch.inference_mode():
        prefill = model(prompt, use_cache=True)
        model(
            torch.tensor([[3]], device="cuda"),
            past_key_values=prefill.past_key_values,
            use_cache=True,
        )
    assert calls == 0
