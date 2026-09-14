"""Validation for the native lifetime-managed one-layer decode runtime."""

from __future__ import annotations

import gc
from contextlib import ExitStack
from dataclasses import dataclass

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, StaticCache

from flux.model import FINAL_FLUX_OPERATOR_CATEGORIES, enable_flux_ops
from flux.model.smollm2_cuda_graph import FluxCUDAGraphDecode
from flux.runtime import (
    NativeSmolLM2LayerDecode,
    native_smollm2_layer_runtime_is_available,
)


RTOL = 2e-4
ATOL = 2e-5
CONTEXT_LENGTHS = (128, 512, 1024, 2048, 4096, 8192)


def _model() -> LlamaForCausalLM:
    config = LlamaConfig(
        hidden_size=576,
        intermediate_size=1536,
        num_hidden_layers=1,
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
    torch.manual_seed(2601)
    model = LlamaForCausalLM(config).float().cuda().eval()
    enable_flux_ops(model, operators=FINAL_FLUX_OPERATOR_CATEGORIES)
    return model


def _state(position: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(2602 + position)
    hidden = torch.randn((1, 1, 576), generator=generator, device="cuda")
    key = torch.randn((1, 3, position, 64), generator=generator, device="cuda")
    value = torch.randn((1, 3, position, 64), generator=generator, device="cuda")
    return hidden, key, value


@dataclass
class _PythonLayerGraph:
    """The current retained Python/Flux graph path, narrowed to one layer."""

    model: LlamaForCausalLM
    capacity: int
    start_position: int
    hidden: torch.Tensor
    position: torch.Tensor
    attention_mask: torch.Tensor
    zero_mask_column: torch.Tensor
    cache: StaticCache
    graph: torch.cuda.CUDAGraph
    output: torch.Tensor
    scratch_context: ExitStack

    @classmethod
    def capture(
        cls,
        model: LlamaForCausalLM,
        hidden: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        position: int,
        capacity: int,
    ) -> _PythonLayerGraph:
        stable_hidden = hidden.clone()
        stable_position = torch.full((1, 1), position, dtype=torch.long, device="cuda")
        mask_min = torch.finfo(torch.float32).min
        attention_mask = torch.full(
            (1, 1, 1, capacity), mask_min, dtype=torch.float32, device="cuda"
        )
        attention_mask[..., :position].zero_()
        zero_mask_column = torch.zeros((1, 1, 1, 1), device="cuda")

        cache = StaticCache(config=model.config, max_cache_len=capacity)
        cache_layer = cache.layers[0]
        seed_key = torch.zeros((1, 3, 1, 64), device="cuda")
        seed_value = torch.zeros_like(seed_key)
        cache_layer.lazy_initialization(seed_key, seed_value)
        if position:
            cache_layer.keys[..., :position, :].copy_(key[..., :position, :])
            cache_layer.values[..., :position, :].copy_(value[..., :position, :])
        cache_layer.cumulative_length.fill_(position)

        scratch = FluxCUDAGraphDecode._allocate_decode_scratch(model, capacity)
        FluxCUDAGraphDecode._initialize_projection_plans(model, scratch)
        installed = ExitStack()
        installed.enter_context(
            FluxCUDAGraphDecode._installed_decode_scratch(model, scratch)
        )
        layer = model.model.layers[0]

        def body() -> torch.Tensor:
            attention_mask.index_copy_(
                3, stable_position.reshape(-1), zero_mask_column
            )
            cos, sin = model.model.rotary_emb(stable_hidden, stable_position)
            result = layer(
                stable_hidden,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                position_embeddings=(cos, sin),
            )
            stable_position.add_(1)
            return result

        # Warm all retained library plans on a side stream, then restore the
        # logical cache/position before capture just like the full Python path.
        current = torch.cuda.current_stream()
        warmup = torch.cuda.Stream()
        warmup.wait_stream(current)
        with torch.cuda.stream(warmup), torch.inference_mode():
            body()
        current.wait_stream(warmup)
        current.synchronize()
        stable_position.fill_(position)
        attention_mask.fill_(mask_min)
        attention_mask[..., :position].zero_()
        cache_layer.cumulative_length.fill_(position)
        cache_layer.keys[..., position:, :].zero_()
        cache_layer.values[..., position:, :].zero_()

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            output = body()
        stable_position.fill_(position)
        attention_mask.fill_(mask_min)
        attention_mask[..., :position].zero_()
        cache_layer.cumulative_length.fill_(position)
        cache_layer.keys[..., position:, :].zero_()
        cache_layer.values[..., position:, :].zero_()
        torch.cuda.synchronize()
        return cls(
            model,
            capacity,
            position,
            stable_hidden,
            stable_position,
            attention_mask,
            zero_mask_column,
            cache,
            graph,
            output,
            installed,
        )

    def replay(self, hidden: torch.Tensor) -> torch.Tensor:
        self.hidden.copy_(hidden)
        self.graph.replay()
        return self.output

    def close(self) -> None:
        torch.cuda.synchronize()
        self.scratch_context.close()


_NATIVE_AVAILABLE = torch.cuda.is_available() and native_smollm2_layer_runtime_is_available()
_CUDA_ONLY = pytest.mark.skipif(
    not _NATIVE_AVAILABLE,
    reason="CUDA and the Flux native one-layer runtime are required",
)


@_CUDA_ONLY
@pytest.mark.parametrize("effective_length", CONTEXT_LENGTHS)
def test_native_layer_matches_current_python_graph(effective_length: int) -> None:
    model = _model()
    position = effective_length - 1
    hidden, key, value = _state(position)
    python_graph = _PythonLayerGraph.capture(
        model,
        hidden,
        key,
        value,
        position=position,
        capacity=effective_length,
    )
    native = NativeSmolLM2LayerDecode.capture(
        model,
        hidden,
        key,
        value,
        cache_position=position,
        cache_capacity=effective_length,
    )
    addresses = native.stable_addresses()

    expected = python_graph.replay(hidden)
    actual = native.replay(hidden)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    reference_cache = python_graph.cache.layers[0]
    torch.testing.assert_close(
        native.key_cache[..., :effective_length, :],
        reference_cache.keys[..., :effective_length, :],
        rtol=RTOL,
        atol=ATOL,
    )
    torch.testing.assert_close(
        native.value_cache[..., :effective_length, :],
        reference_cache.values[..., :effective_length, :],
        rtol=RTOL,
        atol=ATOL,
    )
    assert native.cache_position == native.cache_length == effective_length
    assert int(python_graph.position.item()) == effective_length
    assert int(reference_cache.cumulative_length.item()) == effective_length
    assert native.stable_addresses() == addresses
    with pytest.raises(RuntimeError, match="exhausted"):
        native.replay(hidden)
    python_graph.close()


@_CUDA_ONLY
def test_repeated_replay_reset_workspace_and_allocation_stability() -> None:
    model = _model()
    start = 1000
    capacity = 1024
    hidden, key, value = _state(start)
    python_graph = _PythonLayerGraph.capture(
        model, hidden, key, value, position=start, capacity=capacity
    )
    native = NativeSmolLM2LayerDecode.capture(
        model,
        hidden,
        key,
        value,
        cache_position=start,
        cache_capacity=capacity,
    )
    addresses = native.stable_addresses()
    replay_inputs = tuple(hidden.add((step + 1) * 0.03125) for step in range(3))
    allocated = torch.cuda.memory_allocated()

    for step, next_hidden in enumerate(replay_inputs):
        native.native_object().workspace().fill_(float("nan"))
        expected = python_graph.replay(next_hidden)
        actual = native.replay(next_hidden)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
        assert torch.isfinite(actual).all()
        assert native.cache_position == start + step + 1
        assert native.stable_addresses() == addresses
        assert torch.cuda.memory_allocated() == allocated

    native.reset(hidden, key, value, cache_position=start)
    assert native.cache_position == start
    assert native.replay_count == 0
    expected = python_graph.cache.layers[0]
    torch.testing.assert_close(
        native.key_cache[..., :start, :], expected.keys[..., :start, :], rtol=0, atol=0
    )
    torch.testing.assert_close(
        native.value_cache[..., :start, :],
        expected.values[..., :start, :],
        rtol=0,
        atol=0,
    )
    assert native.stable_addresses() == addresses
    python_graph.close()


@_CUDA_ONLY
def test_replay_obeys_non_default_current_stream() -> None:
    model = _model()
    hidden, key, value = _state(511)
    native = NativeSmolLM2LayerDecode.capture(
        model,
        hidden,
        key,
        value,
        cache_position=511,
        cache_capacity=512,
    )
    stream = torch.cuda.Stream()
    produced = torch.empty_like(hidden)
    with torch.cuda.stream(stream):
        produced.copy_(hidden)
        produced.add_(0.125)
        output = native.replay(produced)
        consumed = output.square().clone()
    stream.synchronize()
    torch.testing.assert_close(consumed, output.square(), rtol=0, atol=0)
    assert native.cache_position == 512


@_CUDA_ONLY
def test_runtime_can_be_destroyed_and_recreated() -> None:
    model = _model()
    hidden, key, value = _state(127)
    first = NativeSmolLM2LayerDecode.capture(
        model,
        hidden,
        key,
        value,
        cache_position=127,
        cache_capacity=128,
    )
    expected = first.replay().clone()
    torch.cuda.synchronize()
    del first
    gc.collect()

    second = NativeSmolLM2LayerDecode.capture(
        model,
        hidden,
        key,
        value,
        cache_position=127,
        cache_capacity=128,
    )
    actual = second.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert second.cache_position == 128
