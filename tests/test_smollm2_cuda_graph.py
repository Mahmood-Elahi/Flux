"""Correctness tests for opt-in fixed-capacity CUDA-Graph decode."""

from __future__ import annotations

import copy

import pytest
import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

from flux.model.smollm2_cuda_graph import (
    FluxCUDAGraphDecode,
    cuda_graph_greedy_generate,
)
from flux.model.smollm2_flux import enable_flux_ops
from flux.ops import (
    native_residual_rmsnorm_is_available,
    native_rmsnorm_is_available,
    native_softmax_is_available,
)


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
    and native_softmax_is_available()
)


@pytest.mark.skipif(
    not _NATIVE_CUDA_AVAILABLE,
    reason="CUDA and all Flux native custom operators are required",
)
def test_repeated_cuda_graph_decode_matches_eager_flux() -> None:
    eager = _model().cuda()
    graph_model = copy.deepcopy(eager)
    enable_flux_ops(eager)
    enable_flux_ops(graph_model)
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

