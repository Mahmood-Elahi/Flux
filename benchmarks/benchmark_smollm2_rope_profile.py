"""Profile the exact Transformers and Flux SmolLM2 RoPE execution paths.

Run from the repository root after building the native Flux extension::

    build/python3119/python.exe benchmarks/benchmark_smollm2_rope_profile.py

The production model is not modified.  Two otherwise identical, fully
optimized Flux models are used: one calls Transformers' eager RoPE helper and
one calls the existing fused Flux RoPE operator.  CUDA-event samples alternate
the two paths and report medians.  Model loading, cache preparation, graph
capture, input construction, and correctness checks are outside timed regions.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import transformers
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile
from transformers.models.llama import modeling_llama
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

import flux.model.smollm2_flux as smollm2_flux
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
    FLUX_PACKED_SWIGLU_CATEGORY,
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


PREFILL_LENGTHS = (128, 512, 1024, 2048, 4096)
DECODE_CONTEXTS = (128, 512, 1024, 2048, 4096)
DEFAULT_WARMUP = 10
DEFAULT_OPERATOR_REPETITIONS = 100
DEFAULT_MODEL_REPETITIONS = 30
DEFAULT_STABILIZATION_ITERATIONS = 50
MIB = 1024.0**2
ALL_OPTIMIZED_OPERATORS = FLUX_OPERATOR_CATEGORIES | {
    FLUX_PACKED_MLP_CATEGORY,
    FLUX_PACKED_SWIGLU_CATEGORY,
}
TRANSFORMERS_ROPE_OPERATORS = ALL_OPTIMIZED_OPERATORS - {"rope"}


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    contiguous: bool
    storage_offset: int


@dataclass(frozen=True)
class RopeTiming:
    size: int
    position_generation_us: float
    transformers_layer_us: float
    transformers_layers_ms: float
    flux_layer_us: float
    flux_layers_ms: float


@dataclass(frozen=True)
class ModelTiming:
    size: int
    transformers_ms: float
    flux_ms: float


@dataclass(frozen=True)
class AllocationResult:
    retained_bytes: int
    peak_increment_bytes: int


@dataclass(frozen=True)
class ProfileResult:
    label: str
    aten_counts: dict[str, int]
    kernel_counts: dict[str, int]
    kernel_us: dict[str, float]
    positive_allocation_bytes: int


def _parse_int_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-lengths", type=_parse_int_list, default=PREFILL_LENGTHS)
    parser.add_argument("--decode-contexts", type=_parse_int_list, default=DECODE_CONTEXTS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument(
        "--operator-repetitions",
        type=int,
        default=DEFAULT_OPERATOR_REPETITIONS,
    )
    parser.add_argument(
        "--model-repetitions",
        type=int,
        default=DEFAULT_MODEL_REPETITIONS,
    )
    parser.add_argument(
        "--stabilization-iterations",
        type=int,
        default=DEFAULT_STABILIZATION_ITERATIONS,
    )
    parser.add_argument("--profile-length", type=int, default=1024)
    parser.add_argument("--skip-profiler", action="store_true")
    parser.add_argument("--skip-graphs", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.stabilization_iterations < 0:
        parser.error("warmup counts must be non-negative")
    if args.operator_repetitions < 1 or args.model_repetitions < 1:
        parser.error("repetition counts must be positive")
    if args.profile_length < 1:
        parser.error("profile length must be positive")
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


def _spec(tensor: torch.Tensor) -> TensorSpec:
    return TensorSpec(
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        tensor.is_contiguous(),
        tensor.storage_offset(),
    )


def _make_rope_inputs(
    model: torch.nn.Module,
    sequence: int,
    offset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct exact projection-view layouts and model-generated cos/sin."""
    attention = model.model.layers[0].self_attn
    batch = 1
    head_dim = int(attention.head_dim)
    query_heads = int(model.config.num_attention_heads)
    key_heads = int(model.config.num_key_value_heads)
    generator = torch.Generator(device="cuda").manual_seed(sequence + offset + 1701)
    query_projection = torch.randn(
        (batch, sequence, query_heads * head_dim),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    key_projection = torch.randn(
        (batch, sequence, key_heads * head_dim),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    query = query_projection.view(batch, sequence, query_heads, head_dim).transpose(1, 2)
    key = key_projection.view(batch, sequence, key_heads, head_dim).transpose(1, 2)
    hidden = torch.empty(
        (batch, sequence, int(model.config.hidden_size)),
        device="cuda",
        dtype=torch.float32,
    )
    position_ids = torch.arange(offset, offset + sequence, device="cuda").unsqueeze(0)
    cos, sin = model.model.rotary_emb(hidden, position_ids)
    return query, key, cos, sin


def _position_inputs(
    model: torch.nn.Module,
    sequence: int,
    offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = torch.empty(
        (1, sequence, int(model.config.hidden_size)),
        device="cuda",
        dtype=torch.float32,
    )
    position_ids = torch.arange(offset, offset + sequence, device="cuda").unsqueeze(0)
    return hidden, position_ids


def _repeat_rope(
    operation: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    inputs: tuple[torch.Tensor, ...],
    layer_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = operation(*inputs)
    for _ in range(layer_count - 1):
        output = operation(*inputs)
    return output


def _repeat_position_generation(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor],
    repetitions: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = model.model.rotary_emb(*inputs)
    for _ in range(repetitions - 1):
        output = model.model.rotary_emb(*inputs)
    return output


def _measure_allocations(operation: Callable[[], object]) -> AllocationResult:
    operation()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = operation()
    torch.cuda.synchronize()
    retained = torch.cuda.memory_allocated() - baseline
    peak = torch.cuda.max_memory_allocated() - baseline
    del output
    torch.cuda.synchronize()
    return AllocationResult(max(0, retained), max(0, peak))


def _short_kernel_name(name: str) -> str:
    if "rope_cuda_fp32_kernel" in name:
        return "rope_cuda_fp32_kernel"
    if "MulFunctor" in name:
        return "elementwise_mul"
    if "neg_kernel_cuda" in name:
        return "elementwise_neg"
    if "CatArrayBatchedCopy" in name:
        return "cat_copy"
    if "CUDAFunctor_add" in name:
        return "elementwise_add"
    if "cos_kernel_cuda" in name:
        return "elementwise_cos"
    if "sin_kernel_cuda" in name:
        return "elementwise_sin"
    if "FillFunctor" in name:
        return "deterministic_empty_fill"
    if "copy_device_to_device" in name or "direct_copy_kernel_cuda" in name:
        return "device_copy_or_cast"
    if "gem" in name.lower() or "ampere" in name.lower():
        return "matmul"
    return name


def _profile_operation(label: str, operation: Callable[[], object]) -> ProfileResult:
    operation()
    torch.cuda.synchronize()
    with torch.inference_mode(), profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
    ) as result:
        output = operation()
    torch.cuda.synchronize()
    del output

    aten_counts: Counter[str] = Counter()
    positive_allocations = 0
    for event in result.key_averages():
        if event.key.startswith("aten::") or event.key.startswith("flux::"):
            aten_counts[event.key] += int(event.count)
            positive_allocations += max(0, int(event.self_device_memory_usage))

    kernel_counts: Counter[str] = Counter()
    kernel_us: defaultdict[str, float] = defaultdict(float)
    for event in result.events():
        if event.device_type != DeviceType.CUDA:
            continue
        name = _short_kernel_name(event.name)
        kernel_counts[name] += 1
        kernel_us[name] += float(event.device_time_total)
    return ProfileResult(
        label,
        dict(aten_counts),
        dict(kernel_counts),
        dict(kernel_us),
        positive_allocations,
    )


def _benchmark_rope(
    model: torch.nn.Module,
    sequence: int,
    offset: int,
    warmup: int,
    repetitions: int,
) -> tuple[RopeTiming, AllocationResult, AllocationResult]:
    inputs = _make_rope_inputs(model, sequence, offset)
    expected = apply_rotary_pos_emb(*inputs)
    actual = rope_native(*inputs)
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-6, atol=2e-7)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-6, atol=2e-7)
    layers = int(model.config.num_hidden_layers)
    position_input = _position_inputs(model, sequence, offset)
    generation_repetitions = layers
    generation = _event_latencies(
        {
            "position": lambda: _repeat_position_generation(
                model, position_input, generation_repetitions
            )
        },
        warmup,
        repetitions,
    )["position"] / generation_repetitions
    totals = _event_latencies(
        {
            "Transformers": lambda: _repeat_rope(
                apply_rotary_pos_emb, inputs, layers
            ),
            "Flux": lambda: _repeat_rope(rope_native, inputs, layers),
        },
        warmup,
        repetitions,
    )
    transformers_alloc = _measure_allocations(lambda: apply_rotary_pos_emb(*inputs))
    flux_alloc = _measure_allocations(lambda: rope_native(*inputs))
    del expected, actual, inputs, position_input
    return (
        RopeTiming(
            sequence if sequence > 1 else offset,
            generation * 1000.0,
            totals["Transformers"] * 1000.0 / layers,
            totals["Transformers"],
            totals["Flux"] * 1000.0 / layers,
            totals["Flux"],
        ),
        transformers_alloc,
        flux_alloc,
    )


def _assert_logits_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-5)


def _benchmark_prefill_model(
    baseline: torch.nn.Module,
    flux: torch.nn.Module,
    length: int,
    warmup: int,
    repetitions: int,
) -> ModelTiming:
    input_ids = _input_ids(length, baseline.config.vocab_size)
    baseline_output = baseline(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    flux_output = flux(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    _assert_logits_close(flux_output.logits, baseline_output.logits)
    del baseline_output, flux_output
    times = _event_latencies(
        {
            "Transformers": lambda: baseline(
                input_ids=input_ids, use_cache=True, logits_to_keep=1
            ),
            "Flux": lambda: flux(input_ids=input_ids, use_cache=True, logits_to_keep=1),
        },
        warmup,
        repetitions,
    )
    del input_ids
    return ModelTiming(length, times["Transformers"], times["Flux"])


def _benchmark_decode_model(
    baseline: torch.nn.Module,
    flux: torch.nn.Module,
    context: int,
    warmup: int,
    repetitions: int,
) -> ModelTiming:
    context_ids = _input_ids(context, baseline.config.vocab_size)
    next_token = _input_ids(context + 1, baseline.config.vocab_size)[:, -1:]
    baseline_cache = _prefill_cache(baseline, context_ids)
    flux_cache = _prefill_cache(flux, context_ids)

    def prepare(model: torch.nn.Module, cache: Any) -> Callable[[], object]:
        sample_cache = _clone_cache(cache, model.config)
        return lambda: model(
            input_ids=next_token,
            past_key_values=sample_cache,
            use_cache=True,
            logits_to_keep=1,
        )

    check_baseline = prepare(baseline, baseline_cache)()
    check_flux = prepare(flux, flux_cache)()
    _assert_logits_close(check_flux.logits, check_baseline.logits)
    del check_baseline, check_flux
    times = _event_latencies(
        {"Transformers": lambda: None, "Flux": lambda: None},
        warmup,
        repetitions,
        prepare={
            "Transformers": lambda: prepare(baseline, baseline_cache),
            "Flux": lambda: prepare(flux, flux_cache),
        },
    )
    del context_ids, next_token, baseline_cache, flux_cache
    return ModelTiming(context, times["Transformers"], times["Flux"])


def _capture_standalone_graph(operation: Callable[[], object]) -> torch.cuda.CUDAGraph:
    current = torch.cuda.current_stream()
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(current)
    with torch.cuda.stream(warmup_stream), torch.inference_mode():
        for _ in range(3):
            operation()
    current.wait_stream(warmup_stream)
    current.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        operation()
    torch.cuda.synchronize()
    return graph


def _benchmark_graph_rope_path(
    model: torch.nn.Module,
    context: int,
    warmup: int,
    repetitions: int,
) -> tuple[float, float]:
    query, key, _, _ = _make_rope_inputs(model, 1, context)
    hidden, positions = _position_inputs(model, 1, context)
    layers = int(model.config.num_hidden_layers)

    def transformers_path() -> object:
        cos, sin = model.model.rotary_emb(hidden, positions)
        return _repeat_rope(
            apply_rotary_pos_emb, (query, key, cos, sin), layers
        )

    def flux_path() -> object:
        cos, sin = model.model.rotary_emb(hidden, positions)
        return _repeat_rope(rope_native, (query, key, cos, sin), layers)

    transformers_graph = _capture_standalone_graph(transformers_path)
    flux_graph = _capture_standalone_graph(flux_path)
    times = _event_latencies(
        {"Transformers": transformers_graph.replay, "Flux": flux_graph.replay},
        warmup,
        repetitions,
    )
    del query, key, hidden, positions, transformers_graph, flux_graph
    return times["Transformers"], times["Flux"]


def _benchmark_decode_graphs(
    baseline: torch.nn.Module,
    flux: torch.nn.Module,
    context: int,
    warmup: int,
    repetitions: int,
) -> ModelTiming:
    prompt = _input_ids(context, baseline.config.vocab_size)
    capacity = 1 + warmup + repetitions
    baseline_graph = FluxCUDAGraphDecode.capture(
        baseline, prompt, max_decode_steps=capacity, warmup_steps=3
    )
    flux_graph = FluxCUDAGraphDecode.capture(
        flux, prompt, max_decode_steps=capacity, warmup_steps=3
    )
    baseline_logits = baseline_graph.replay()
    flux_logits = flux_graph.replay()
    _assert_logits_close(flux_logits, baseline_logits)
    times = _event_latencies(
        {"Transformers": baseline_graph.replay, "Flux": flux_graph.replay},
        warmup,
        repetitions,
    )
    del prompt, baseline_graph, flux_graph
    gc.collect()
    torch.cuda.synchronize()
    return ModelTiming(context, times["Transformers"], times["Flux"])


def _capture_runtime_specs(
    model: torch.nn.Module,
    *,
    length: int,
    decode: bool,
    use_flux: bool,
) -> tuple[dict[str, TensorSpec], int, int]:
    records: list[dict[str, Any]] = []
    attribute = "rope_native" if use_flux else "apply_rotary_pos_emb"
    original = getattr(smollm2_flux, attribute)

    def capture(
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = (
            original(query, key, cos, sin)
            if use_flux
            else original(query, key, cos, sin, unsqueeze_dim)
        )
        records.append(
            {
                "query": _spec(query),
                "key": _spec(key),
                "cos": _spec(cos),
                "sin": _spec(sin),
                "query_output": _spec(output[0]),
                "key_output": _spec(output[1]),
                "cos_pointer": cos.data_ptr(),
                "sin_pointer": sin.data_ptr(),
            }
        )
        return output

    input_ids = _input_ids(length, model.config.vocab_size)
    cache = _prefill_cache(model, input_ids) if decode else None
    setattr(smollm2_flux, attribute, capture)
    try:
        if decode:
            token = _input_ids(length + 1, model.config.vocab_size)[:, -1:]
            output = model(
                input_ids=token,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            del cache, token
        else:
            output = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize()
        del input_ids, output
    finally:
        setattr(smollm2_flux, attribute, original)
    if not records:
        raise AssertionError("failed to capture Transformers RoPE inputs")
    first = records[0]
    specs = {name: value for name, value in first.items() if isinstance(value, TensorSpec)}
    unique_cos = len({record["cos_pointer"] for record in records})
    unique_sin = len({record["sin_pointer"] for record in records})
    return specs, unique_cos, unique_sin


def _print_environment(args: argparse.Namespace, model: torch.nn.Module) -> None:
    attention = model.model.layers[0].self_attn
    print("SmolLM2 RoPE profile environment")
    print(f"  model={MODEL_ID}; revision={MODEL_REVISION}")
    print(f"  Python={sys.version.split()[0]}; PyTorch={torch.__version__}")
    print(f"  Transformers={transformers.__version__}; CUDA build={torch.version.cuda}")
    print(f"  GPU={torch.cuda.get_device_name()}; capability={torch.cuda.get_device_capability()}")
    print("  dtype=torch.float32; eager attention; TF32=off; deterministic algorithms=on")
    print(
        f"  layers={model.config.num_hidden_layers}; query_heads={model.config.num_attention_heads}; "
        f"key_value_heads={model.config.num_key_value_heads}; head_dim={attention.head_dim}"
    )
    print("  both model paths retain packed gate/up and fused packed SwiGLU")
    print(
        "  deterministic empty-fill safety="
        f"{torch.utils.deterministic.fill_uninitialized_memory}"
    )
    print(
        f"  warmup={args.warmup}; operator samples={args.operator_repetitions}; "
        f"model samples={args.model_repetitions}; statistic=median"
    )


def _print_specs(label: str, specs: dict[str, TensorSpec], cos_ptrs: int, sin_ptrs: int) -> None:
    print(f"\nExact runtime tensor metadata: {label}")
    for name, value in specs.items():
        print(
            f"  {name:>12}: shape={value.shape}; stride={value.stride}; "
            f"dtype={value.dtype}; contiguous={value.contiguous}; "
            f"storage_offset={value.storage_offset}"
        )
    print(f"  cos/sin unique storage pointers across layers: {cos_ptrs}/{sin_ptrs}")


def _print_rope_table(title: str, results: list[RopeTiming]) -> None:
    print(f"\n{title}")
    print(
        f"{'size':>7} {'cos/sin us':>11} {'HF/layer us':>12} {'HF/30 ms':>10} "
        f"{'Flux/layer us':>14} {'Flux/30 ms':>11} {'apply speedup':>14}"
    )
    for row in results:
        print(
            f"{row.size:>7} {row.position_generation_us:>11.3f} "
            f"{row.transformers_layer_us:>12.3f} {row.transformers_layers_ms:>10.4f} "
            f"{row.flux_layer_us:>14.3f} {row.flux_layers_ms:>11.4f} "
            f"{row.transformers_layers_ms / row.flux_layers_ms:>13.3f}x"
        )


def _print_model_table(
    title: str,
    rope: list[RopeTiming],
    models: list[ModelTiming],
) -> None:
    print(f"\n{title}")
    print(
        f"{'size':>7} {'HF model ms':>12} {'HF RoPE %':>10} {'Flux model ms':>13} "
        f"{'Flux RoPE %':>12} {'model delta us':>14}"
    )
    for rope_row, model_row in zip(rope, models, strict=True):
        hf_rope_ms = rope_row.transformers_layers_ms + rope_row.position_generation_us / 1000.0
        flux_rope_ms = rope_row.flux_layers_ms + rope_row.position_generation_us / 1000.0
        print(
            f"{model_row.size:>7} {model_row.transformers_ms:>12.4f} "
            f"{100.0 * hf_rope_ms / model_row.transformers_ms:>9.3f}% "
            f"{model_row.flux_ms:>13.4f} "
            f"{100.0 * flux_rope_ms / model_row.flux_ms:>11.3f}% "
            f"{(model_row.transformers_ms - model_row.flux_ms) * 1000.0:>14.3f}"
        )


def _print_allocations(
    title: str,
    sizes: tuple[int, ...],
    transformer_results: list[AllocationResult],
    flux_results: list[AllocationResult],
) -> None:
    print(f"\n{title}")
    print(
        f"{'size':>7} {'output MiB':>11} {'HF temp volume MiB':>19} "
        f"{'HF peak MiB':>12} {'Flux temp':>11} {'Flux peak MiB':>14}"
    )
    for size, hf, flux in zip(sizes, transformer_results, flux_results, strict=True):
        sequence = size if "Prefill" in title else 1
        output_bytes = sequence * (9 + 3) * 64 * 4
        # The helper allocates two multiply results, one half-sized negation,
        # one cat, and one add/output per Q/K element: 4.5x output allocation
        # volume, of which 3.5x is temporary.  Profiler output below validates
        # the formula at representative shapes.
        hf_temp_volume = 3.5 * output_bytes
        # The native implementation allocates only its two logical outputs.
        # Allocator block rounding can make retained_bytes slightly larger for
        # the tiny one-token case; it is not an operator intermediate.
        flux_temp_volume = 0
        print(
            f"{size:>7} {output_bytes / MIB:>11.4f} {hf_temp_volume / MIB:>19.4f} "
            f"{hf.peak_increment_bytes / MIB:>12.4f} "
            f"{flux_temp_volume / MIB:>11.4f} {flux.peak_increment_bytes / MIB:>14.4f}"
        )


def _print_profile(result: ProfileResult) -> None:
    print(f"\nProfiler breakdown: {result.label}")
    print(f"  cumulative positive CUDA allocation: {result.positive_allocation_bytes / MIB:.6f} MiB")
    print("  ATen/custom operator counts:")
    interesting = {
        name: count
        for name, count in result.aten_counts.items()
        if name
        in {
            "aten::unsqueeze",
            "aten::slice",
            "aten::neg",
            "aten::cat",
            "aten::mul",
            "aten::add",
            "aten::_to_copy",
            "aten::matmul",
            "aten::bmm",
            "aten::transpose",
            "aten::cos",
            "aten::sin",
            "aten::expand",
            "flux::rope",
        }
    }
    for name, count in interesting.items():
        print(f"    {name}: {count}")
    print("  CUDA kernels:")
    for name, count in result.kernel_counts.items():
        print(f"    {name}: count={count}; total={result.kernel_us[name]:.3f} us")


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the SmolLM2 RoPE profile")
    if not _required_ops_available():
        raise RuntimeError("all built Flux operators are required for the optimized model")
    _configure_runtime()

    print("Loading optimized model with Transformers RoPE...", flush=True)
    baseline = enable_flux_ops(
        load_model("cuda"), operators=TRANSFORMERS_ROPE_OPERATORS
    )
    print("Loading optimized model with existing Flux RoPE...", flush=True)
    flux = enable_flux_ops(load_model("cuda"), operators=ALL_OPTIMIZED_OPERATORS)
    _print_environment(args, flux)
    maximum = int(flux.config.max_position_embeddings)
    prefill_lengths = tuple(length for length in args.prefill_lengths if length <= maximum)
    decode_contexts = tuple(context for context in args.decode_contexts if context + 1 <= maximum)
    if not prefill_lengths or not decode_contexts:
        raise RuntimeError("no requested workloads fit max_position_embeddings")

    with torch.inference_mode():
        prefill_specs, prefill_cos_ptrs, prefill_sin_ptrs = _capture_runtime_specs(
            baseline, length=prefill_lengths[0], decode=False, use_flux=False
        )
        prefill_flux_specs, prefill_flux_cos_ptrs, prefill_flux_sin_ptrs = (
            _capture_runtime_specs(
                flux, length=prefill_lengths[0], decode=False, use_flux=True
            )
        )
        decode_specs, decode_cos_ptrs, decode_sin_ptrs = _capture_runtime_specs(
            baseline,
            length=decode_contexts[len(decode_contexts) // 2],
            decode=True,
            use_flux=False,
        )
        decode_flux_specs, decode_flux_cos_ptrs, decode_flux_sin_ptrs = (
            _capture_runtime_specs(
                flux,
                length=decode_contexts[len(decode_contexts) // 2],
                decode=True,
                use_flux=True,
            )
        )
    _print_specs(
        f"Transformers apply, prefill length {prefill_lengths[0]}",
        prefill_specs,
        prefill_cos_ptrs,
        prefill_sin_ptrs,
    )
    _print_specs(
        f"Flux apply, prefill length {prefill_lengths[0]}",
        prefill_flux_specs,
        prefill_flux_cos_ptrs,
        prefill_flux_sin_ptrs,
    )
    _print_specs(
        f"Transformers apply, one-token decode after context "
        f"{decode_contexts[len(decode_contexts) // 2]}",
        decode_specs,
        decode_cos_ptrs,
        decode_sin_ptrs,
    )
    _print_specs(
        f"Flux apply, one-token decode after context "
        f"{decode_contexts[len(decode_contexts) // 2]}",
        decode_flux_specs,
        decode_flux_cos_ptrs,
        decode_flux_sin_ptrs,
    )

    print("\nTransformers helper source used at runtime:")
    print(inspect.getsource(modeling_llama.rotate_half).rstrip())
    print(inspect.getsource(modeling_llama.apply_rotary_pos_emb).rstrip())

    print("\nStabilizing GPU clocks with alternating untimed optimized prefills...", flush=True)
    stabilization_ids = _input_ids(prefill_lengths[0], flux.config.vocab_size)
    with torch.inference_mode():
        for iteration in range(args.stabilization_iterations):
            models = (baseline, flux) if iteration % 2 == 0 else (flux, baseline)
            for model in models:
                output = model(
                    input_ids=stabilization_ids,
                    use_cache=False,
                    logits_to_keep=1,
                )
                del output
    torch.cuda.synchronize()
    del stabilization_ids

    prefill_rope: list[RopeTiming] = []
    decode_rope: list[RopeTiming] = []
    prefill_hf_alloc: list[AllocationResult] = []
    prefill_flux_alloc: list[AllocationResult] = []
    decode_hf_alloc: list[AllocationResult] = []
    decode_flux_alloc: list[AllocationResult] = []
    prefill_models: list[ModelTiming] = []
    decode_models: list[ModelTiming] = []
    graph_models: list[ModelTiming] = []

    with torch.inference_mode():
        for length in prefill_lengths:
            print(f"Profiling {length}-token prefill RoPE and model...", flush=True)
            rope_result, hf_alloc, flux_alloc = _benchmark_rope(
                flux, length, 0, args.warmup, args.operator_repetitions
            )
            prefill_rope.append(rope_result)
            prefill_hf_alloc.append(hf_alloc)
            prefill_flux_alloc.append(flux_alloc)
            try:
                prefill_models.append(
                    _benchmark_prefill_model(
                        baseline,
                        flux,
                        length,
                        args.warmup,
                        args.model_repetitions,
                    )
                )
            except torch.OutOfMemoryError as error:
                raise RuntimeError(f"prefill length {length} ran out of CUDA memory") from error

        for context in decode_contexts:
            print(f"Profiling one-token decode after context {context}...", flush=True)
            rope_result, hf_alloc, flux_alloc = _benchmark_rope(
                flux, 1, context, args.warmup, args.operator_repetitions
            )
            decode_rope.append(rope_result)
            decode_hf_alloc.append(hf_alloc)
            decode_flux_alloc.append(flux_alloc)
            decode_models.append(
                _benchmark_decode_model(
                    baseline,
                    flux,
                    context,
                    args.warmup,
                    args.model_repetitions,
                )
            )
            if not args.skip_graphs:
                graph_models.append(
                    _benchmark_decode_graphs(
                        baseline,
                        flux,
                        context,
                        args.warmup,
                        args.model_repetitions,
                    )
                )

        graph_rope = None
        if not args.skip_graphs:
            graph_rope = _benchmark_graph_rope_path(
                flux,
                decode_contexts[len(decode_contexts) // 2],
                args.warmup,
                args.operator_repetitions,
            )

        profiles: list[ProfileResult] = []
        if not args.skip_profiler and args.profile_length <= maximum:
            profile_inputs = _make_rope_inputs(flux, args.profile_length, 0)
            position_inputs = _position_inputs(flux, args.profile_length, 0)
            profiles.extend(
                (
                    _profile_operation(
                        f"Transformers apply, sequence={args.profile_length}",
                        lambda: apply_rotary_pos_emb(*profile_inputs),
                    ),
                    _profile_operation(
                        f"Flux apply, sequence={args.profile_length}",
                        lambda: rope_native(*profile_inputs),
                    ),
                    _profile_operation(
                        f"cos/sin generation, sequence={args.profile_length}",
                        lambda: flux.model.rotary_emb(*position_inputs),
                    ),
                )
            )
            longest = max(prefill_lengths)
            if longest != args.profile_length:
                long_inputs = _make_rope_inputs(flux, longest, 0)
                profiles.extend(
                    (
                        _profile_operation(
                            f"Transformers apply, sequence={longest}",
                            lambda: apply_rotary_pos_emb(*long_inputs),
                        ),
                        _profile_operation(
                            f"Flux apply, sequence={longest}",
                            lambda: rope_native(*long_inputs),
                        ),
                    )
                )
            decode_inputs = _make_rope_inputs(flux, 1, decode_contexts[0])
            profiles.extend(
                (
                    _profile_operation(
                        "Transformers apply, one-token decode",
                        lambda: apply_rotary_pos_emb(*decode_inputs),
                    ),
                    _profile_operation(
                        "Flux apply, one-token decode",
                        lambda: rope_native(*decode_inputs),
                    ),
                )
            )

    _print_rope_table("Prefill targeted RoPE CUDA-event medians", prefill_rope)
    _print_model_table(
        "Complete optimized prefill and RoPE percentage",
        prefill_rope,
        prefill_models,
    )
    _print_allocations(
        "Prefill per-layer apply allocations",
        prefill_lengths,
        prefill_hf_alloc,
        prefill_flux_alloc,
    )
    _print_rope_table("One-token cached-decode targeted RoPE CUDA-event medians", decode_rope)
    _print_model_table(
        "Complete optimized eager decode and RoPE percentage",
        decode_rope,
        decode_models,
    )
    _print_allocations(
        "Decode per-layer apply allocations",
        decode_contexts,
        decode_hf_alloc,
        decode_flux_alloc,
    )

    if graph_rope is not None:
        hf_graph_rope, flux_graph_rope = graph_rope
        print("\nCUDA-Graph one-token decode")
        print(
            f"  isolated full RoPE path (one cos/sin generation + 30 applies): "
            f"Transformers={hf_graph_rope * 1000.0:.3f} us; "
            f"Flux={flux_graph_rope * 1000.0:.3f} us"
        )
        print(
            f"{'context':>9} {'HF graph ms':>12} {'Flux graph ms':>14} "
            f"{'Flux RoPE %':>12} {'graph delta us':>15}"
        )
        for row in graph_models:
            print(
                f"{row.size:>9} {row.transformers_ms:>12.4f} {row.flux_ms:>14.4f} "
                f"{100.0 * flux_graph_rope / row.flux_ms:>11.3f}% "
                f"{(row.transformers_ms - row.flux_ms) * 1000.0:>15.3f}"
            )

    for profiler_result in profiles:
        _print_profile(profiler_result)

    print("\nStructural conclusions encoded by the measured paths")
    print("  Transformers applies Q and K independently: 10 CUDA kernels per layer.")
    print("  unsqueeze and half-dimension slices are views; neg/cat/mul/add allocate.")
    print("  No explicit cast or contiguous copy occurs in FP32 apply_rotary_pos_emb.")
    print("  The same precomputed cos/sin storage is broadcast across heads and reused by all layers.")
    print("  Flux reads strided projection views directly and combines Q/K rotation in one RoPE kernel.")
    print("  This deterministic benchmark also launches two safety fills for its two empty outputs.")
    print("  Both eager paths and the existing Flux operator are CUDA-Graph capturable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
