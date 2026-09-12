"""Validate and benchmark the production packed QKV Flux SmolLM2 path.

Run from the repository root after building the native Flux extension::

    build/python3119/python.exe benchmarks/benchmark_smollm2_qkv.py

The opt-in production candidate owns one bias-free PyTorch ``nn.Linear`` with
output ordering ``[Q | K | V]`` and uses allocation-free split/view operations
before the existing Flux RoPE operator. CUDA-event samples alternate paths and
report medians; TF32 is disabled by the shared SmolLM2 benchmark setup.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from benchmarks.benchmark_smollm2 import (
    _clone_cache,
    _configure_runtime,
    _event_latencies,
    _input_ids,
    _prefill_cache,
)
from flux.model.smollm2 import MODEL_ID, MODEL_REVISION, load_model
from flux.model.smollm2_cuda_graph import FluxCUDAGraphDecode
from flux.model.smollm2_flux import (
    FLUX_OPERATOR_CATEGORIES,
    FLUX_PACKED_MLP_CATEGORY,
    FLUX_PACKED_QKV_CATEGORY,
    FLUX_PACKED_SWIGLU_CATEGORY,
    FluxLlamaAttention,
    enable_flux_ops,
)
from flux.ops import (
    native_attention_score_softmax_is_available,
    native_packed_swiglu_is_available,
    native_residual_rmsnorm_is_available,
    native_rmsnorm_is_available,
    native_rope_is_available,
    native_softmax_is_available,
    rope_native,
)


LENGTHS = (1, 128, 512, 1024, 2048, 4096)
PREFILL_LENGTHS = (128, 512, 1024, 2048, 4096)
DECODE_CONTEXTS = (128, 512, 1024, 2048, 4096)
DEFAULT_WARMUP = 10
DEFAULT_OPERATOR_REPETITIONS = 100
DEFAULT_MODEL_REPETITIONS = 30
DEFAULT_STABILIZATION_ITERATIONS = 50
MIB = 1024.0**2
RTOL = 2e-4
ATOL = 3e-5
RELATIVE_FLOOR = 1e-7
ALL_OPTIMIZED_OPERATORS = FLUX_OPERATOR_CATEGORIES | {
    FLUX_PACKED_MLP_CATEGORY,
    FLUX_PACKED_SWIGLU_CATEGORY,
}


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    contiguous: bool


@dataclass(frozen=True)
class Error:
    maximum_absolute: float
    maximum_relative: float


@dataclass(frozen=True)
class ProjectionTiming:
    length: int
    query_us: float
    key_us: float
    value_us: float
    separate_us: float
    packed_us: float


@dataclass(frozen=True)
class SubpathTiming:
    length: int
    separate_us: float
    packed_us: float


@dataclass(frozen=True)
class ViewTiming:
    length: int
    separate_us: float
    packed_us: float


@dataclass(frozen=True)
class ModelTiming:
    size: int
    separate_ms: float
    packed_ms: float
    logits_error: Error


@dataclass(frozen=True)
class GraphTiming:
    model: ModelTiming
    separate_pool_mib: float
    packed_pool_mib: float


@dataclass(frozen=True)
class ConversionMemory:
    parameter_bytes_before: int
    parameter_bytes_after: int
    storage_bytes_before: int
    storage_bytes_after: int
    qkv_parameter_bytes: int
    peak_increment_bytes: int


@dataclass(frozen=True)
class ActivationMemory:
    length: int
    separate_retained_bytes: int
    packed_retained_bytes: int
    separate_peak_bytes: int
    packed_peak_bytes: int
    separate_model_peak_bytes: int
    packed_model_peak_bytes: int


@dataclass(frozen=True)
class CacheMemory:
    length: int
    logical_bytes: int
    separate_storage_bytes: int
    packed_storage_bytes: int


def _spec(tensor: torch.Tensor) -> TensorSpec:
    return TensorSpec(
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.is_contiguous(),
    )


def _parse_int_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=_parse_int_list, default=LENGTHS)
    parser.add_argument("--prefill-lengths", type=_parse_int_list, default=PREFILL_LENGTHS)
    parser.add_argument("--decode-contexts", type=_parse_int_list, default=DECODE_CONTEXTS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument(
        "--operator-repetitions", type=int, default=DEFAULT_OPERATOR_REPETITIONS
    )
    parser.add_argument("--model-repetitions", type=int, default=DEFAULT_MODEL_REPETITIONS)
    parser.add_argument(
        "--stabilization-iterations",
        type=int,
        default=DEFAULT_STABILIZATION_ITERATIONS,
    )
    parser.add_argument("--memory-length", type=int, default=4096)
    parser.add_argument("--skip-graphs", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.stabilization_iterations < 0:
        parser.error("warmup counts must be non-negative")
    if args.operator_repetitions < 1 or args.model_repetitions < 1:
        parser.error("repetition counts must be positive")
    return args


def _required_ops_available() -> bool:
    return all(
        (
            native_attention_score_softmax_is_available(),
            native_packed_swiglu_is_available(),
            native_residual_rmsnorm_is_available(),
            native_rmsnorm_is_available(),
            native_rope_is_available(),
            native_softmax_is_available(),
        )
    )


def _parameter_bytes(model: nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def _unique_parameter_storage_bytes(model: nn.Module) -> int:
    storages: dict[tuple[str, int], int] = {}
    for parameter in model.parameters():
        storage = parameter.untyped_storage()
        storages.setdefault((str(parameter.device), storage.data_ptr()), storage.nbytes())
    return sum(storages.values())


def _convert_to_packed_qkv(model: nn.Module) -> ConversionMemory:
    before_parameters = _parameter_bytes(model)
    before_storage = _unique_parameter_storage_bytes(model)
    qkv_bytes = sum(
        projection.weight.numel() * projection.weight.element_size()
        for layer in model.model.layers
        for projection in (
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
        )
    )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    enable_flux_ops(
        model,
        operators=ALL_OPTIMIZED_OPERATORS | {FLUX_PACKED_QKV_CATEGORY},
    )
    torch.cuda.synchronize()
    peak_increment = torch.cuda.max_memory_allocated() - baseline
    return ConversionMemory(
        before_parameters,
        _parameter_bytes(model),
        before_storage,
        _unique_parameter_storage_bytes(model),
        qkv_bytes,
        peak_increment,
    )


def _error(actual: torch.Tensor, expected: torch.Tensor) -> Error:
    difference = (actual - expected).abs()
    return Error(
        float(difference.max().item()),
        float((difference / expected.abs().clamp_min(RELATIVE_FLOOR)).max().item()),
    )


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> Error:
    result = _error(actual, expected)
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    return result


def _merge_error(current: Error, candidate: Error) -> Error:
    return Error(
        max(current.maximum_absolute, candidate.maximum_absolute),
        max(current.maximum_relative, candidate.maximum_relative),
    )


def _repeat(operation: Callable[[], Any], repetitions: int) -> Any:
    output = operation()
    for _ in range(repetitions - 1):
        output = operation()
    return output


def _projection_inputs(length: int, hidden_size: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(1901 + length)
    return torch.randn(
        (1, length, hidden_size),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )


def _separate_views(
    query_raw: torch.Tensor,
    key_raw: torch.Tensor,
    value_raw: torch.Tensor,
    query_heads: int,
    key_value_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, length, _ = query_raw.shape
    query = query_raw.view(batch, length, query_heads, head_dim).transpose(1, 2)
    key = key_raw.view(batch, length, key_value_heads, head_dim).transpose(1, 2)
    value = value_raw.view(batch, length, key_value_heads, head_dim).transpose(1, 2)
    return query, key, value


def _packed_views(
    qkv: torch.Tensor,
    widths: tuple[int, int, int],
    query_heads: int,
    key_value_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_raw, key_raw, value_raw = qkv.split(widths, dim=-1)
    return _separate_views(
        query_raw, key_raw, value_raw, query_heads, key_value_heads, head_dim
    )


def _time_views(
    length: int,
    separate_raw: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    qkv: torch.Tensor,
    widths: tuple[int, int, int],
    query_heads: int,
    key_value_heads: int,
    head_dim: int,
) -> ViewTiming:
    iterations = 2000
    samples = 15
    operations = {
        "separate": lambda: _separate_views(
            *separate_raw, query_heads, key_value_heads, head_dim
        ),
        "packed": lambda: _packed_views(
            qkv, widths, query_heads, key_value_heads, head_dim
        ),
    }
    results: dict[str, list[float]] = {name: [] for name in operations}
    torch.cuda.synchronize()
    for sample in range(samples):
        order = tuple(operations) if sample % 2 == 0 else tuple(reversed(operations))
        for name in order:
            started = time.perf_counter_ns()
            output = _repeat(operations[name], iterations)
            elapsed = time.perf_counter_ns() - started
            results[name].append(elapsed / iterations / 1000.0)
            del output
    return ViewTiming(
        length,
        statistics.median(results["separate"]),
        statistics.median(results["packed"]),
    )


def _benchmark_projection_and_subpath(
    model: nn.Module,
    packed_model: nn.Module,
    length: int,
    warmup: int,
    repetitions: int,
) -> tuple[ProjectionTiming, SubpathTiming, ViewTiming, dict[str, Error]]:
    separate = model.model.layers[0].self_attn
    packed = packed_model.model.layers[0].self_attn
    if not isinstance(packed, FluxLlamaAttention) or not packed.use_packed_qkv:
        raise TypeError("production packed-QKV attention was not installed")
    hidden = _projection_inputs(length, int(model.config.hidden_size))
    position_ids = torch.arange(length, device="cuda").unsqueeze(0)
    cos, sin = model.model.rotary_emb(hidden, position_ids)
    widths = (packed.query_width, packed.key_width, packed.value_width)
    query_heads = int(model.config.num_attention_heads)
    key_value_heads = int(model.config.num_key_value_heads)
    head_dim = int(separate.head_dim)

    query_raw = separate.q_proj(hidden)
    key_raw = separate.k_proj(hidden)
    value_raw = separate.v_proj(hidden)
    qkv = packed.packed_qkv(hidden)
    packed_raw = qkv.split(widths, dim=-1)
    errors = {
        "Q": _assert_close(packed_raw[0], query_raw),
        "K": _assert_close(packed_raw[1], key_raw),
        "V": _assert_close(packed_raw[2], value_raw),
    }
    separate_views = _separate_views(
        query_raw, key_raw, value_raw, query_heads, key_value_heads, head_dim
    )
    candidate_views = _packed_views(
        qkv, widths, query_heads, key_value_heads, head_dim
    )
    q_rope, k_rope = rope_native(separate_views[0], separate_views[1], cos, sin)
    packed_q_rope, packed_k_rope = rope_native(
        candidate_views[0], candidate_views[1], cos, sin
    )
    errors["RoPE Q"] = _assert_close(packed_q_rope, q_rope)
    errors["RoPE K"] = _assert_close(packed_k_rope, k_rope)
    errors["attention V"] = _assert_close(candidate_views[2], separate_views[2])

    packed_storage = qkv.untyped_storage().data_ptr()
    for tensor in (*packed_raw, *candidate_views):
        if tensor.untyped_storage().data_ptr() != packed_storage:
            raise AssertionError("packed split/view unexpectedly allocated storage")

    inner = max(1, min(50, 4096 // length))
    projection_times = _event_latencies(
        {
            "Q": lambda: _repeat(lambda: separate.q_proj(hidden), inner),
            "K": lambda: _repeat(lambda: separate.k_proj(hidden), inner),
            "V": lambda: _repeat(lambda: separate.v_proj(hidden), inner),
            "separate": lambda: _repeat(
                lambda: (
                    separate.q_proj(hidden),
                    separate.k_proj(hidden),
                    separate.v_proj(hidden),
                ),
                inner,
            ),
            "packed": lambda: _repeat(lambda: packed.packed_qkv(hidden), inner),
        },
        warmup,
        repetitions,
    )

    def separate_subpath() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        views = _separate_views(
            separate.q_proj(hidden),
            separate.k_proj(hidden),
            separate.v_proj(hidden),
            query_heads,
            key_value_heads,
            head_dim,
        )
        query, key = rope_native(views[0], views[1], cos, sin)
        return query, key, views[2]

    def packed_subpath() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        views = _packed_views(
            packed.packed_qkv(hidden),
            widths,
            query_heads,
            key_value_heads,
            head_dim,
        )
        query, key = rope_native(views[0], views[1], cos, sin)
        return query, key, views[2]

    subpath_times = _event_latencies(
        {
            "separate": lambda: _repeat(separate_subpath, inner),
            "packed": lambda: _repeat(packed_subpath, inner),
        },
        warmup,
        repetitions,
    )
    view_times = _time_views(
        length,
        (query_raw, key_raw, value_raw),
        qkv,
        widths,
        query_heads,
        key_value_heads,
        head_dim,
    )
    del hidden, cos, sin, position_ids, query_raw, key_raw, value_raw, qkv
    return (
        ProjectionTiming(
            length,
            projection_times["Q"] * 1000.0 / inner,
            projection_times["K"] * 1000.0 / inner,
            projection_times["V"] * 1000.0 / inner,
            projection_times["separate"] * 1000.0 / inner,
            projection_times["packed"] * 1000.0 / inner,
        ),
        SubpathTiming(
            length,
            subpath_times["separate"] * 1000.0 / inner,
            subpath_times["packed"] * 1000.0 / inner,
        ),
        view_times,
        errors,
    )


def _measure_operation_memory(operation: Callable[[], Any]) -> tuple[int, int]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = operation()
    torch.cuda.synchronize()
    retained = torch.cuda.memory_allocated() - baseline
    peak = torch.cuda.max_memory_allocated() - baseline
    del output
    torch.cuda.synchronize()
    return max(0, retained), max(0, peak)


def _measure_activation_memory(
    model: nn.Module, packed_model: nn.Module, length: int
) -> ActivationMemory:
    separate = model.model.layers[0].self_attn
    packed = packed_model.model.layers[0].self_attn
    hidden = _projection_inputs(length, int(model.config.hidden_size))
    positions = torch.arange(length, device="cuda").unsqueeze(0)
    cos, sin = model.model.rotary_emb(hidden, positions)
    qh = int(model.config.num_attention_heads)
    kvh = int(model.config.num_key_value_heads)
    hd = int(separate.head_dim)
    widths = (packed.query_width, packed.key_width, packed.value_width)

    def separate_path() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        views = _separate_views(
            separate.q_proj(hidden), separate.k_proj(hidden), separate.v_proj(hidden),
            qh, kvh, hd,
        )
        query, key = rope_native(views[0], views[1], cos, sin)
        return query, key, views[2]

    def packed_path() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        views = _packed_views(packed.packed_qkv(hidden), widths, qh, kvh, hd)
        query, key = rope_native(views[0], views[1], cos, sin)
        return query, key, views[2]

    separate_retained, separate_peak = _measure_operation_memory(separate_path)
    packed_retained, packed_peak = _measure_operation_memory(packed_path)
    input_ids = _input_ids(length, model.config.vocab_size)
    _, separate_model_peak = _measure_operation_memory(
        lambda: model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
    )
    _, packed_model_peak = _measure_operation_memory(
        lambda: packed_model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
    )
    del hidden, positions, cos, sin, input_ids
    return ActivationMemory(
        length,
        separate_retained,
        packed_retained,
        separate_peak,
        packed_peak,
        separate_model_peak,
        packed_model_peak,
    )


def _cache_storage_bytes(cache: Any) -> int:
    storages: dict[tuple[str, int], int] = {}
    for layer in cache.layers:
        for tensor in (layer.keys, layer.values):
            storage = tensor.untyped_storage()
            storages.setdefault((str(tensor.device), storage.data_ptr()), storage.nbytes())
    return sum(storages.values())


def _measure_cache_memory(
    separate: nn.Module, packed: nn.Module, length: int
) -> CacheMemory:
    input_ids = _input_ids(length, separate.config.vocab_size)
    separate_cache = _prefill_cache(separate, input_ids)
    packed_cache = _prefill_cache(packed, input_ids)
    logical = sum(
        tensor.numel() * tensor.element_size()
        for layer in separate_cache.layers
        for tensor in (layer.keys, layer.values)
    )
    result = CacheMemory(
        length,
        logical,
        _cache_storage_bytes(separate_cache),
        _cache_storage_bytes(packed_cache),
    )
    del input_ids, separate_cache, packed_cache
    return result


def _benchmark_prefill(
    separate: nn.Module,
    packed: nn.Module,
    length: int,
    warmup: int,
    repetitions: int,
) -> ModelTiming:
    input_ids = _input_ids(length, separate.config.vocab_size)
    expected = separate(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    actual = packed(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    error = _assert_close(actual.logits, expected.logits)
    del expected, actual
    times = _event_latencies(
        {
            "separate": lambda: separate(
                input_ids=input_ids, use_cache=True, logits_to_keep=1
            ),
            "packed": lambda: packed(
                input_ids=input_ids, use_cache=True, logits_to_keep=1
            ),
        },
        warmup,
        repetitions,
    )
    del input_ids
    return ModelTiming(length, times["separate"], times["packed"], error)


def _cache_error(actual: Any, expected: Any) -> Error:
    result = Error(0.0, 0.0)
    for actual_layer, expected_layer in zip(actual.layers, expected.layers, strict=True):
        for actual_tensor, expected_tensor in (
            (actual_layer.keys, expected_layer.keys),
            (actual_layer.values, expected_layer.values),
        ):
            result = _merge_error(result, _assert_close(actual_tensor, expected_tensor))
    return result


def _benchmark_decode(
    separate: nn.Module,
    packed: nn.Module,
    context: int,
    warmup: int,
    repetitions: int,
) -> tuple[ModelTiming, Error]:
    context_ids = _input_ids(context, separate.config.vocab_size)
    next_token = _input_ids(context + 1, separate.config.vocab_size)[:, -1:]
    separate_cache = _prefill_cache(separate, context_ids)
    packed_cache = _prefill_cache(packed, context_ids)

    separate_check = _clone_cache(separate_cache, separate.config)
    packed_check = _clone_cache(packed_cache, packed.config)
    expected = separate(
        input_ids=next_token,
        past_key_values=separate_check,
        use_cache=True,
        logits_to_keep=1,
    )
    actual = packed(
        input_ids=next_token,
        past_key_values=packed_check,
        use_cache=True,
        logits_to_keep=1,
    )
    logits_error = _assert_close(actual.logits, expected.logits)
    cache_error = _cache_error(actual.past_key_values, expected.past_key_values)
    del expected, actual, separate_check, packed_check

    def prepare(model: nn.Module, cache: Any) -> Callable[[], Any]:
        sample_cache = _clone_cache(cache, model.config)
        return lambda: model(
            input_ids=next_token,
            past_key_values=sample_cache,
            use_cache=True,
            logits_to_keep=1,
        )

    times = _event_latencies(
        {"separate": lambda: None, "packed": lambda: None},
        warmup,
        repetitions,
        prepare={
            "separate": lambda: prepare(separate, separate_cache),
            "packed": lambda: prepare(packed, packed_cache),
        },
    )
    del context_ids, next_token, separate_cache, packed_cache
    return ModelTiming(context, times["separate"], times["packed"], logits_error), cache_error


def _qkv_parameter_addresses(model: nn.Module) -> tuple[int, ...]:
    return tuple(
        layer.self_attn.packed_qkv.weight.data_ptr() for layer in model.model.layers
    )


def _benchmark_graph_decode(
    separate: nn.Module,
    packed: nn.Module,
    context: int,
    warmup: int,
    repetitions: int,
) -> GraphTiming:
    prompt = _input_ids(context, separate.config.vocab_size)
    token = _input_ids(context + 1, separate.config.vocab_size)[:, -1:]
    capacity = 2 + warmup + repetitions
    parameter_addresses = _qkv_parameter_addresses(packed)

    # Probe each graph in isolation first.  Measuring the two initial captures
    # back-to-back otherwise attributes CUDA allocator segment reuse (commonly
    # 32 MiB here) to whichever model happens to capture second.
    separate_probe = FluxCUDAGraphDecode.capture(
        separate, prompt, max_decode_steps=2
    )
    separate_pool_mib = separate_probe.memory.graph_pool_bytes / MIB
    del separate_probe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    packed_probe = FluxCUDAGraphDecode.capture(packed, prompt, max_decode_steps=2)
    packed_pool_mib = packed_probe.memory.graph_pool_bytes / MIB
    del packed_probe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    separate_graph = FluxCUDAGraphDecode.capture(
        separate, prompt, max_decode_steps=capacity
    )
    packed_graph = FluxCUDAGraphDecode.capture(packed, prompt, max_decode_steps=capacity)
    if _qkv_parameter_addresses(packed) != parameter_addresses:
        raise AssertionError("packed QKV parameter addresses changed during capture")
    _assert_close(packed_graph.prefill_logits, separate_graph.prefill_logits)
    expected = separate_graph.replay(token)
    actual = packed_graph.replay(token)
    error = _assert_close(actual, expected)
    separate_addresses = separate_graph.stable_addresses()
    packed_addresses = packed_graph.stable_addresses()
    allocated_before_replay = torch.cuda.memory_allocated()
    separate_graph.replay(token)
    packed_graph.replay(token)
    torch.cuda.synchronize()
    if torch.cuda.memory_allocated() != allocated_before_replay:
        raise AssertionError("CUDA-Graph replay changed live CUDA allocation")
    times = _event_latencies(
        {
            "separate": lambda: separate_graph.replay(token),
            "packed": lambda: packed_graph.replay(token),
        },
        warmup,
        repetitions,
    )
    if separate_graph.stable_addresses() != separate_addresses:
        raise AssertionError("separate graph addresses changed across replay")
    if packed_graph.stable_addresses() != packed_addresses:
        raise AssertionError("packed graph addresses changed across replay")
    if _qkv_parameter_addresses(packed) != parameter_addresses:
        raise AssertionError("packed QKV parameter addresses changed across replay")
    result = GraphTiming(
        ModelTiming(context, times["separate"], times["packed"], error),
        separate_pool_mib,
        packed_pool_mib,
    )
    del prompt, token, separate_graph, packed_graph
    gc.collect()
    torch.cuda.synchronize()
    return result


def _module_errors_and_generation(
    separate: nn.Module, packed: nn.Module
) -> tuple[Error, Error, bool]:
    input_ids = _input_ids(128, separate.config.vocab_size)
    outputs: dict[str, torch.Tensor] = {}

    def capture(name: str) -> Callable[..., None]:
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            outputs[name] = tensor.detach().clone()
        return hook

    handles = (
        separate.model.layers[0].self_attn.register_forward_hook(
            capture("separate attention")
        ),
        packed.model.layers[0].self_attn.register_forward_hook(
            capture("packed attention")
        ),
        separate.model.layers[0].register_forward_hook(capture("separate")),
        packed.model.layers[0].register_forward_hook(capture("packed")),
    )
    try:
        separate(input_ids=input_ids, use_cache=False, logits_to_keep=1)
        packed(input_ids=input_ids, use_cache=False, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
    attention_error = _assert_close(
        outputs["packed attention"], outputs["separate attention"]
    )
    layer_error = _assert_close(outputs["packed"], outputs["separate"])
    prompt = _input_ids(8, separate.config.vocab_size)
    separate_tokens = separate.generate(
        prompt, do_sample=False, max_new_tokens=8, use_cache=True
    )
    packed_tokens = packed.generate(
        prompt, do_sample=False, max_new_tokens=8, use_cache=True
    )
    greedy_equal = bool(torch.equal(packed_tokens, separate_tokens))
    if not greedy_equal:
        raise AssertionError("packed QKV changed greedy generation token IDs")
    del input_ids, outputs, prompt, separate_tokens, packed_tokens
    return attention_error, layer_error, greedy_equal


def _print_environment(args: argparse.Namespace) -> None:
    print("SmolLM2 production packed-QKV benchmark")
    print(f"  Model: {MODEL_ID}")
    print(f"  Revision: {MODEL_REVISION}")
    print(f"  Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"  PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}")
    print(f"  Transformers: {__import__('transformers').__version__}")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print("  dtype: torch.float32; TF32: disabled; attention: eager")
    print(f"  lengths: {args.lengths}")
    print(
        f"  warmup={args.warmup}, operator samples={args.operator_repetitions}, "
        f"model samples={args.model_repetitions}"
    )


def _print_timing_table(
    title: str, rows: Sequence[Any], fields: tuple[str, str]
) -> None:
    print(f"\n{title}")
    print(f"{'tokens':>7} {'separate':>12} {'packed':>12} {'speedup':>10}")
    first, second = fields
    for row in rows:
        separate = getattr(row, first)
        packed = getattr(row, second)
        print(f"{row.length:>7} {separate:>12.3f} {packed:>12.3f} {separate/packed:>9.3f}x")


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not _required_ops_available():
        raise RuntimeError("all built Flux operators are required")
    _configure_runtime()
    _print_environment(args)

    print("\nLoading current fully optimized Flux model...", flush=True)
    separate = enable_flux_ops(load_model("cuda"), operators=ALL_OPTIMIZED_OPERATORS)
    print("Loading identical packed-QKV benchmark candidate...", flush=True)
    packed = load_model("cuda")
    conversion = _convert_to_packed_qkv(packed)

    config = separate.config
    attention = separate.model.layers[0].self_attn
    candidate = packed.model.layers[0].self_attn
    print("\nProjection configuration")
    print(
        f"  hidden={config.hidden_size}, query_heads={config.num_attention_heads}, "
        f"kv_heads={config.num_key_value_heads}, head_dim={attention.head_dim}, "
        f"GQA={attention.num_key_value_groups}"
    )
    print(f"  Q weight: {tuple(attention.q_proj.weight.shape)}")
    print(f"  K weight: {tuple(attention.k_proj.weight.shape)}")
    print(f"  V weight: {tuple(attention.v_proj.weight.shape)}")
    print(f"  packed [Q|K|V] weight: {tuple(candidate.packed_qkv.weight.shape)}")

    maximum = int(config.max_position_embeddings)
    lengths = tuple(length for length in args.lengths if length <= maximum)
    prefills = tuple(length for length in args.prefill_lengths if length <= maximum)
    graph_capacity = 2 + args.warmup + args.model_repetitions
    contexts = tuple(
        context for context in args.decode_contexts
        if context + (graph_capacity if not args.skip_graphs else 1) <= maximum
    )

    print("\nStabilizing GPU clocks with alternating untimed prefills...", flush=True)
    stabilization_ids = _input_ids(min(prefills), config.vocab_size)
    with torch.inference_mode():
        for iteration in range(args.stabilization_iterations):
            models = (separate, packed) if iteration % 2 == 0 else (packed, separate)
            for model in models:
                output = model(input_ids=stabilization_ids, use_cache=True, logits_to_keep=1)
                del output
    torch.cuda.synchronize()
    del stabilization_ids

    projection_results: list[ProjectionTiming] = []
    subpath_results: list[SubpathTiming] = []
    view_results: list[ViewTiming] = []
    numerical: dict[str, Error] = {}
    with torch.inference_mode():
        for length in lengths:
            print(f"Profiling QKV and attention setup at {length} tokens...", flush=True)
            projection, subpath, views, errors = _benchmark_projection_and_subpath(
                separate,
                packed,
                length,
                args.warmup,
                args.operator_repetitions,
            )
            projection_results.append(projection)
            subpath_results.append(subpath)
            view_results.append(views)
            for name, error in errors.items():
                numerical[name] = _merge_error(
                    numerical.get(name, Error(0.0, 0.0)), error
                )

        layouts_length = next((value for value in lengths if value > 1), lengths[0])
        layout_hidden = _projection_inputs(layouts_length, int(config.hidden_size))
        q_raw = attention.q_proj(layout_hidden)
        k_raw = attention.k_proj(layout_hidden)
        v_raw = attention.v_proj(layout_hidden)
        separate_layouts = {
            "Q raw": _spec(q_raw),
            "K raw": _spec(k_raw),
            "V raw": _spec(v_raw),
        }
        separate_views = _separate_views(
            q_raw,
            k_raw,
            v_raw,
            int(config.num_attention_heads),
            int(config.num_key_value_heads),
            int(attention.head_dim),
        )
        separate_layouts.update(
            {name: _spec(tensor) for name, tensor in zip(("Q", "K", "V"), separate_views)}
        )
        qkv = candidate.packed_qkv(layout_hidden)
        packed_raw = qkv.split(
            (candidate.query_width, candidate.key_width, candidate.value_width),
            dim=-1,
        )
        packed_views = _packed_views(
            qkv,
            (candidate.query_width, candidate.key_width, candidate.value_width),
            candidate.query_heads,
            candidate.key_value_heads,
            candidate.head_dim,
        )
        packed_layouts = {
            "qkv": _spec(qkv),
            "query_raw": _spec(packed_raw[0]),
            "key_raw": _spec(packed_raw[1]),
            "value_raw": _spec(packed_raw[2]),
            "query": _spec(packed_views[0]),
            "key": _spec(packed_views[1]),
            "value": _spec(packed_views[2]),
        }
        del layout_hidden, q_raw, k_raw, v_raw, separate_views

        activation_memory = _measure_activation_memory(
            separate, packed, args.memory_length
        )
        cache_memory = _measure_cache_memory(separate, packed, args.memory_length)
        attention_error, layer_error, greedy_equal = _module_errors_and_generation(
            separate, packed
        )
        numerical["attention output"] = attention_error
        numerical["decoder layer"] = layer_error

        prefill_results = []
        for length in prefills:
            print(f"Benchmarking integrated prefill at {length} tokens...", flush=True)
            prefill_results.append(
                _benchmark_prefill(
                    separate, packed, length, args.warmup, args.model_repetitions
                )
            )
        numerical["prefill logits"] = Error(
            max(row.logits_error.maximum_absolute for row in prefill_results),
            max(row.logits_error.maximum_relative for row in prefill_results),
        )

        decode_results = []
        cache_errors = []
        for context in contexts:
            print(f"Benchmarking eager decode at context {context}...", flush=True)
            timing, cache_error = _benchmark_decode(
                separate, packed, context, args.warmup, args.model_repetitions
            )
            decode_results.append(timing)
            cache_errors.append(cache_error)
        numerical["decode logits"] = Error(
            max(row.logits_error.maximum_absolute for row in decode_results),
            max(row.logits_error.maximum_relative for row in decode_results),
        )
        numerical["decode cache K/V"] = Error(
            max(error.maximum_absolute for error in cache_errors),
            max(error.maximum_relative for error in cache_errors),
        )

        graph_results = []
        if not args.skip_graphs:
            for context in contexts:
                print(f"Benchmarking CUDA-Graph decode at context {context}...", flush=True)
                graph_results.append(
                    _benchmark_graph_decode(
                        separate,
                        packed,
                        context,
                        args.warmup,
                        args.model_repetitions,
                    )
                )
            numerical["graph decode logits"] = Error(
                max(row.model.logits_error.maximum_absolute for row in graph_results),
                max(row.model.logits_error.maximum_relative for row in graph_results),
            )

    print("\nRuntime tensor layouts (batch=1, sequence=" f"{layouts_length})")
    for path, layouts in (("separate", separate_layouts), ("packed", packed_layouts)):
        print(f"  {path}:")
        for name, spec in layouts.items():
            print(
                f"    {name:<10} shape={spec.shape}, stride={spec.stride}, "
                f"offset={spec.storage_offset}, contiguous={spec.contiguous}"
            )

    print("\nIsolated projection latency (microseconds per layer)")
    print(
        f"{'tokens':>7} {'Q':>9} {'K':>9} {'V':>9} {'3x total':>11} "
        f"{'packed':>10} {'speedup':>9}"
    )
    for row in projection_results:
        print(
            f"{row.length:>7} {row.query_us:>9.3f} {row.key_us:>9.3f} "
            f"{row.value_us:>9.3f} {row.separate_us:>11.3f} "
            f"{row.packed_us:>10.3f} {row.separate_us/row.packed_us:>8.3f}x"
        )
    _print_timing_table(
        "Attention subpath: projections -> views -> Flux RoPE (microseconds per layer)",
        subpath_results,
        ("separate_us", "packed_us"),
    )
    _print_timing_table(
        "Host metadata overhead: split/view/transpose (microseconds)",
        view_results,
        ("separate_us", "packed_us"),
    )

    def print_model_table(title: str, rows: Sequence[ModelTiming]) -> None:
        print(f"\n{title}")
        print(f"{'tokens/context':>14} {'separate ms':>13} {'packed ms':>11} {'speedup':>9}")
        for row in rows:
            print(
                f"{row.size:>14} {row.separate_ms:>13.3f} {row.packed_ms:>11.3f} "
                f"{row.separate_ms/row.packed_ms:>8.3f}x"
            )

    print_model_table("Integrated prefill", prefill_results)
    print_model_table("Eager one-token cached decode", decode_results)
    if graph_results:
        print_model_table("CUDA-Graph one-token decode", [row.model for row in graph_results])
        print("\nCUDA-Graph pool allocation")
        print(f"{'context':>8} {'separate MiB':>14} {'packed MiB':>12} {'delta MiB':>11}")
        for row in graph_results:
            print(
                f"{row.model.size:>8} {row.separate_pool_mib:>14.3f} "
                f"{row.packed_pool_mib:>12.3f} "
                f"{row.packed_pool_mib-row.separate_pool_mib:>11.3f}"
            )

    print("\nNumerical differences (packed versus separate; max over tested sizes)")
    print(f"{'value':<20} {'max absolute':>14} {'max relative*':>15}")
    for name, error in numerical.items():
        print(
            f"{name:<20} {error.maximum_absolute:>14.7g} "
            f"{error.maximum_relative:>15.7g}"
        )
    print(f"  * relative denominator is clamped to {RELATIVE_FLOOR:g}")
    print(f"  greedy generation token IDs equal: {greedy_equal}")

    layers = len(separate.model.layers)
    print("\nMemory and ownership")
    print(
        f"  Q/K/V parameters: {conversion.qkv_parameter_bytes:,} bytes total "
        f"across {layers} layers ({conversion.qkv_parameter_bytes/layers:,.0f}/layer)"
    )
    print(
        f"  model parameter bytes: {conversion.parameter_bytes_before:,} -> "
        f"{conversion.parameter_bytes_after:,}"
    )
    print(
        f"  unique parameter storage: {conversion.storage_bytes_before:,} -> "
        f"{conversion.storage_bytes_after:,} bytes"
    )
    print(
        f"  measured full optimized one-layer-at-a-time conversion peak increment: "
        f"{conversion.peak_increment_bytes/MIB:.3f} MiB"
    )
    packed_output_bytes = args.memory_length * (
        candidate.query_width + candidate.key_width + candidate.value_width
    ) * 4
    print(
        f"  packed output at {args.memory_length} tokens: "
        f"{packed_output_bytes:,} bytes ({packed_output_bytes/MIB:.3f} MiB)"
    )
    print(
        f"  attention-subpath retained allocation: "
        f"{activation_memory.separate_retained_bytes/MIB:.3f} -> "
        f"{activation_memory.packed_retained_bytes/MIB:.3f} MiB"
    )
    print(
        f"  attention-subpath peak increment: "
        f"{activation_memory.separate_peak_bytes/MIB:.3f} -> "
        f"{activation_memory.packed_peak_bytes/MIB:.3f} MiB"
    )
    print(
        f"  full-model no-cache peak increment: "
        f"{activation_memory.separate_model_peak_bytes/MIB:.3f} -> "
        f"{activation_memory.packed_model_peak_bytes/MIB:.3f} MiB"
    )
    print(
        f"  DynamicCache at {cache_memory.length}: "
        f"logical={cache_memory.logical_bytes/MIB:.3f} MiB, "
        f"separate backing={cache_memory.separate_storage_bytes/MIB:.3f} MiB, "
        f"packed backing={cache_memory.packed_storage_bytes/MIB:.3f} MiB"
    )
    print("\nView and cache compatibility")
    print("  split/view operations are metadata-only; packed Q/K feed Flux RoPE directly")
    print(
        "  packed V feeds DynamicCache/StaticCache update without an explicit "
        "copy before the cache API"
    )
    if graph_results:
        print(
            "  CUDA-Graph capture/replay succeeded without replay allocation and "
            "with stable model I/O, cache, logits, and QKV parameter addresses"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
