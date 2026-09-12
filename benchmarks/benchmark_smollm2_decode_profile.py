"""Profile the remaining maximally optimized Flux SmolLM2 inference path.

Run from the repository root after building the Flux native extension:

    build/python3119/python.exe benchmarks/benchmark_smollm2_decode_profile.py

The benchmark keeps production execution unchanged. Uninstrumented CUDA-event
medians establish eager decode, CUDA-Graph replay, and prefill latency. A single
profiler pass then inventories natural full-model launches. Phase estimates use
non-overlapping outer operators from that pass and are corroborated by batched
isolated phase timings, avoiding CUDA events between operations in a live layer.
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile
from transformers.cache_utils import DynamicLayer, StaticLayer
from transformers.models.llama.modeling_llama import repeat_kv

from flux.model.smollm2 import MODEL_ID, MODEL_REVISION, inspect_config, load_model
from flux.model.smollm2_cuda_graph import FluxCUDAGraphDecode
from flux.model.smollm2_flux import (
    FLUX_OPERATOR_CATEGORIES,
    FLUX_PACKED_MLP_CATEGORY,
    FLUX_PACKED_QKV_CATEGORY,
    FLUX_PACKED_SWIGLU_CATEGORY,
    enable_flux_ops,
    flux_operator_counts,
)
from flux.ops import (
    native_attention_score_softmax_is_available,
    native_packed_swiglu_is_available,
    native_residual_rmsnorm_is_available,
    native_rmsnorm_is_available,
    native_rope_is_available,
    native_softmax_is_available,
    packed_swiglu_native,
    residual_rmsnorm_native,
    rope_native,
    softmax_native,
)


DECODE_CONTEXTS = (128, 512, 1024, 2048, 4096)
PREFILL_LENGTHS = (512, 1024, 4096)
ALL_OPERATORS = FLUX_OPERATOR_CATEGORIES | {
    FLUX_PACKED_QKV_CATEGORY,
    FLUX_PACKED_MLP_CATEGORY,
    FLUX_PACKED_SWIGLU_CATEGORY,
}
SEED = 0
MIB = 1024.0**2


@dataclass(frozen=True)
class TimingRow:
    size: int
    milliseconds: float


@dataclass(frozen=True)
class PhaseRow:
    name: str
    calls: int
    milliseconds: float


@dataclass(frozen=True)
class ProfileSummary:
    workload: str
    size: int
    baseline_ms: float
    profiled_ms: float
    kernel_ms: float
    kernel_launches: int
    phases: tuple[PhaseRow, ...]
    kernel_inventory: tuple[tuple[str, int, float], ...]
    fill_inventory: tuple[tuple[str, int, float], ...]


@dataclass(frozen=True)
class IntermediateRow:
    name: str
    shape: tuple[int, ...]
    size_mib: float
    allocation: str
    producer: str
    consumer: str


def _parse_int_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("values must be positive")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-contexts", type=_parse_int_list, default=DECODE_CONTEXTS)
    parser.add_argument("--prefill-lengths", type=_parse_int_list, default=PREFILL_LENGTHS)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--graph-replays", type=int, default=30)
    parser.add_argument("--stabilization-iterations", type=int, default=50)
    parser.add_argument(
        "--profile-contexts",
        type=_parse_int_list,
        default=DECODE_CONTEXTS,
        help="decode contexts receiving one eager and one graph profiler pass",
    )
    parser.add_argument("--skip-profiler", action="store_true")
    parser.add_argument("--skip-microbenchmarks", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.repetitions < 1 or args.graph_replays < 1:
        parser.error("warmup may be zero; repetitions and graph replays must be positive")
    if args.stabilization_iterations < 0:
        parser.error("stabilization iterations must be non-negative")
    return args


def _configure_runtime() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _input_ids(length: int, vocab_size: int) -> torch.Tensor:
    values = (torch.arange(length, dtype=torch.long) * 17 + 11) % vocab_size
    return values.unsqueeze(0).to("cuda")


def _event_median(
    operation: Callable[[], object],
    warmup: int,
    repetitions: int,
    prepare: Callable[[], Callable[[], object]] | None = None,
) -> float:
    output: object = None
    for _ in range(warmup):
        call = prepare() if prepare is not None else operation
        output = call()
    torch.cuda.synchronize()
    samples = []
    final = None
    for _ in range(repetitions):
        call = prepare() if prepare is not None else operation
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = call()
        end.record()
        samples.append((start, end))
        final = end
    assert final is not None
    final.synchronize()
    result = statistics.median(start.elapsed_time(end) for start, end in samples)
    del output
    return result


def _clone_dynamic_cache(cache: Any, config: Any) -> Any:
    from transformers import DynamicCache

    return DynamicCache(
        [(layer.keys.detach().clone(), layer.values.detach().clone()) for layer in cache.layers],
        config=config,
    )


def _prefill_cache(model: torch.nn.Module, context_ids: torch.Tensor) -> Any:
    output = model(input_ids=context_ids, use_cache=True, logits_to_keep=1)
    cache = output.past_key_values
    del output
    return cache


def _time_eager_decode(
    model: torch.nn.Module, context: int, warmup: int, repetitions: int
) -> float:
    context_ids = _input_ids(context, model.config.vocab_size)
    token = _input_ids(context + 1, model.config.vocab_size)[:, -1:]
    base_cache = _prefill_cache(model, context_ids)

    def prepare() -> Callable[[], object]:
        cache = _clone_dynamic_cache(base_cache, model.config)
        return lambda: model(
            input_ids=token,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )

    result = _event_median(lambda: None, warmup, repetitions, prepare)
    del context_ids, token, base_cache
    return result


def _make_graph(model: torch.nn.Module, context: int, steps: int) -> FluxCUDAGraphDecode:
    prompt = _input_ids(context, model.config.vocab_size)
    graph = FluxCUDAGraphDecode.capture(model, prompt, max_decode_steps=steps)
    del prompt
    return graph


def _time_graph_decode(
    model: torch.nn.Module, context: int, warmup: int, repetitions: int
) -> float:
    graph = _make_graph(model, context, warmup + repetitions)
    for _ in range(warmup):
        graph.replay()
    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        samples.append((start, end))
    samples[-1][1].synchronize()
    result = statistics.median(start.elapsed_time(end) for start, end in samples)
    del graph
    return result


def _time_prefill(
    model: torch.nn.Module, length: int, warmup: int, repetitions: int
) -> float:
    input_ids = _input_ids(length, model.config.vocab_size)
    result = _event_median(
        lambda: model(input_ids=input_ids, use_cache=True, logits_to_keep=1),
        warmup,
        repetitions,
    )
    del input_ids
    return result


def _shape(value: object) -> tuple[int, ...]:
    if isinstance(value, (tuple, list)) and all(isinstance(item, int) for item in value):
        return tuple(value)
    return ()


def _phase_for_event(event: Any, config: Any, workload: str) -> str | None:
    """Classify selected non-overlapping CPU ops by semantic decoder phase."""
    shapes = event.input_shapes or []
    first = _shape(shapes[0]) if shapes else ()
    second = _shape(shapes[1]) if len(shapes) > 1 else ()
    name = event.name
    hidden = int(config.hidden_size)
    intermediate = int(config.intermediate_size)
    q_width = hidden
    kv_width = int(config.num_key_value_heads) * (hidden // int(config.num_attention_heads))

    if name == "aten::embedding":
        return "input embedding"
    if name == "flux::rmsnorm":
        return "RMSNorm"
    if name == "flux::residual_rmsnorm":
        return "residual + RMSNorm"
    if name == "flux::rope":
        return "RoPE"
    if name == "flux::softmax":
        return "score scale/mask/softmax"
    if name == "flux::attention_score_softmax":
        return "score scale/mask/softmax"
    if name == "flux::packed_swiglu":
        return "packed SwiGLU"
    if name == "aten::linear" and len(second) == 2:
        output_width, input_width = second
        if (output_width, input_width) == (q_width + 2 * kv_width, hidden):
            return "packed QKV projection"
        if (output_width, input_width) == (hidden, hidden):
            return "o_proj"
        if (output_width, input_width) == (2 * intermediate, hidden):
            return "packed gate/up projection"
        if (output_width, input_width) == (hidden, intermediate):
            return "down_proj"
        if output_width == int(config.vocab_size):
            return "LM head"
    if name == "aten::bmm" and len(first) == 3 and len(second) == 3:
        head_dim = hidden // int(config.num_attention_heads)
        if first[-1] == head_dim and second[-2] == head_dim:
            return "QK^T GEMM"
        if second[-1] == head_dim:
            return "attention x V GEMM"
    if name == "aten::index_copy_" and len(first) == 4:
        return "KV cache update" if first[-1] == hidden // int(config.num_attention_heads) else "input/mask/position setup"
    if name == "aten::copy_" and len(first) == 5:
        return "GQA repeat-KV copies"
    if name == "aten::copy_" and len(first) == 4 and first[-1] == hidden // int(config.num_attention_heads):
        return "attention output layout copy"
    if name in {"aten::mul", "aten::add"} and len(first) == 4 and int(config.num_attention_heads) in first:
        return "score scale/mask/softmax"
    if name == "aten::add" and first and first[-1] == hidden and len(first) == 3:
        return "residual add"
    return None


def _short_kernel_name(name: str) -> str:
    for marker, short in (
        ("fill", "deterministic/safety fill"),
        ("residual_rmsnorm_cuda_fp32_kernel", "Flux residual-RMSNorm"),
        ("rmsnorm_cuda_fp32_kernel", "Flux RMSNorm"),
        ("rope_cuda_fp32_kernel", "Flux RoPE"),
        ("packed_swiglu_cuda_fp32_kernel", "Flux packed SwiGLU"),
        ("softmax_cuda_fp32_kernel", "Flux softmax"),
        ("radix_sort", "index_copy radix sort"),
        ("assert_async", "index_copy async assertions"),
        ("indexing_backward_kernel", "index_copy scatter"),
        ("index_copy", "index_copy"),
        ("CatArrayBatchedCopy", "cat copy"),
        ("Li5ELb0", "packed gate/up GEMV"),
        ("Li9ELb0", "down_proj GEMV"),
        ("Li2ELi2", "packed QKV/o_proj GEMV"),
        ("Li4ELi4", "LM-head GEMV"),
        ("Li6ELb0", "attention x V batched GEMV"),
        ("Li6ELb1", "QK^T batched GEMV"),
        ("gemv", "GEMV"),
        ("elementwise_kernel", "elementwise"),
        ("vectorized_elementwise_kernel", "vectorized elementwise"),
        ("reduce_kernel", "reduction"),
        ("gemm", "GEMM"),
        ("sgemm", "GEMM"),
    ):
        if marker.lower() in name.lower():
            return short
    return name if len(name) <= 100 else name[:97] + "..."


def _summarize_profile(
    profiler: Any,
    workload: str,
    size: int,
    baseline_ms: float,
    profiled_ms: float,
    config: Any,
) -> ProfileSummary:
    phase_times: dict[str, float] = defaultdict(float)
    phase_counts: Counter[str] = Counter()
    kernels: dict[str, list[float]] = defaultdict(list)
    fills: dict[str, list[float]] = defaultdict(list)
    kernel_ms = 0.0
    kernel_launches = 0
    eager_decode_cat_count = 0
    for event in profiler.events():
        if event.device_type == DeviceType.CPU and event.device_time_total > 0:
            if event.name == "aten::cat" and workload == "eager decode":
                # The model-level rotary embedding concatenates freqs once
                # before the layers; the following 60 cats are K/V updates.
                phase_name = (
                    "input/mask/position setup"
                    if eager_decode_cat_count == 0
                    else "KV cache update"
                )
                eager_decode_cat_count += 1
            else:
                phase_name = _phase_for_event(event, config, workload)
            if phase_name is not None:
                phase_times[phase_name] += float(event.device_time_total) / 1000.0
                phase_counts[phase_name] += 1
        elif event.device_type == DeviceType.CUDA:
            duration = float(event.self_device_time_total) / 1000.0
            kernel_ms += duration
            kernel_launches += 1
            short = _short_kernel_name(event.name)
            kernels[short].append(duration)
            lowered = event.name.lower()
            if "fill" in lowered or "uninitialized" in lowered:
                fills[short].append(duration)
    phases = tuple(
        PhaseRow(name, phase_counts[name], phase_times[name])
        for name in sorted(phase_times, key=phase_times.get, reverse=True)
    )
    inventory = tuple(
        sorted(
            ((name, len(times), sum(times)) for name, times in kernels.items()),
            key=lambda row: row[2],
            reverse=True,
        )
    )
    fill_inventory = tuple(
        sorted(
            ((name, len(times), sum(times)) for name, times in fills.items()),
            key=lambda row: row[2],
            reverse=True,
        )
    )
    return ProfileSummary(
        workload,
        size,
        baseline_ms,
        profiled_ms,
        kernel_ms,
        kernel_launches,
        phases,
        inventory,
        fill_inventory,
    )


def _profile_call(
    workload: str,
    size: int,
    baseline_ms: float,
    config: Any,
    operation: Callable[[], object],
) -> ProfileSummary:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as profiler:
        start.record()
        output = operation()
        end.record()
    end.synchronize()
    result = _summarize_profile(
        profiler, workload, size, baseline_ms, start.elapsed_time(end), config
    )
    del output
    return result


def _profile_eager_decode(
    model: torch.nn.Module, context: int, baseline_ms: float
) -> ProfileSummary:
    context_ids = _input_ids(context, model.config.vocab_size)
    token = _input_ids(context + 1, model.config.vocab_size)[:, -1:]
    cache = _prefill_cache(model, context_ids)
    result = _profile_call(
        "eager decode",
        context,
        baseline_ms,
        model.config,
        lambda: model(
            input_ids=token,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        ),
    )
    del context_ids, token, cache
    return result


def _profile_graph_decode(
    model: torch.nn.Module, context: int, baseline_ms: float, capacity_steps: int
) -> ProfileSummary:
    graph = _make_graph(model, context, capacity_steps)
    result = _profile_call(
        "graph replay", context, baseline_ms, model.config, graph.graph.replay
    )
    del graph
    return result


def _profile_prefill(
    model: torch.nn.Module, length: int, baseline_ms: float
) -> ProfileSummary:
    input_ids = _input_ids(length, model.config.vocab_size)
    result = _profile_call(
        "prefill",
        length,
        baseline_ms,
        model.config,
        lambda: model(input_ids=input_ids, use_cache=True, logits_to_keep=1),
    )
    del input_ids
    return result


def _batched_phase_median(
    operation: Callable[[], object], warmup: int, repetitions: int, batch: int = 30
) -> float:
    def repeated() -> object:
        output = None
        for _ in range(batch):
            output = operation()
        return output

    return _event_median(repeated, warmup, repetitions) / batch


def _microbenchmark_layer(
    model: torch.nn.Module, context: int, warmup: int, repetitions: int
) -> tuple[PhaseRow, ...]:
    """Time isolated exact-shape layer-0 phases in 30-call batches."""
    layer = model.model.layers[0]
    attention = layer.self_attn
    hidden = int(model.config.hidden_size)
    heads = int(model.config.num_attention_heads)
    kv_heads = int(model.config.num_key_value_heads)
    head_dim = hidden // heads
    groups = heads // kv_heads
    intermediate = int(model.config.intermediate_size)
    x = torch.randn((1, 1, hidden), device="cuda", dtype=torch.float32)
    residual = torch.randn_like(x)
    normalized = layer.input_layernorm(x)
    q, k, v = attention.project_qkv(normalized)
    cos = torch.randn((1, 1, head_dim), device="cuda", dtype=torch.float32)
    sin = torch.randn_like(cos)
    q_rope, k_rope = rope_native(q, k, cos, sin)
    cached_k = torch.randn((1, kv_heads, context, head_dim), device="cuda")
    cached_v = torch.randn_like(cached_k)
    expanded_k = repeat_kv(cached_k, groups)
    expanded_v = repeat_kv(cached_v, groups)
    scores = torch.matmul(q_rope, expanded_k.transpose(2, 3))
    mask = torch.zeros((1, 1, 1, context), device="cuda")
    probabilities = softmax_native(scores * attention.scaling + mask)
    attended = torch.matmul(probabilities, expanded_v)
    layout = attended.transpose(1, 2).contiguous().reshape(1, 1, hidden).contiguous()
    attention_output = attention.o_proj(layout)
    mlp_input, mlp_residual = residual_rmsnorm_native(
        attention_output,
        residual,
        layer.post_attention_layernorm.weight,
        layer.post_attention_layernorm.variance_epsilon,
    )
    packed = layer.mlp.gate_up_proj(mlp_input)
    activated = packed_swiglu_native(packed)
    down = layer.mlp.down_proj(activated)

    operations: tuple[tuple[str, Callable[[], object]], ...] = (
        ("RMSNorm", lambda: layer.input_layernorm(x)),
        ("packed QKV projection", lambda: attention.project_qkv(normalized)),
        ("RoPE", lambda: rope_native(q, k, cos, sin)),
        ("GQA repeat K", lambda: repeat_kv(cached_k, groups)),
        ("GQA repeat V", lambda: repeat_kv(cached_v, groups)),
        ("QK^T GEMM", lambda: torch.matmul(q_rope, expanded_k.transpose(2, 3))),
        (
            "score scale/mask/softmax",
            lambda: softmax_native(scores * attention.scaling + mask),
        ),
        ("attention x V GEMM", lambda: torch.matmul(probabilities, expanded_v)),
        (
            "attention output layout",
            lambda: attended.transpose(1, 2).contiguous().reshape(1, 1, hidden).contiguous(),
        ),
        ("o_proj", lambda: attention.o_proj(layout)),
        (
            "residual + RMSNorm",
            lambda: residual_rmsnorm_native(
                attention_output,
                residual,
                layer.post_attention_layernorm.weight,
                layer.post_attention_layernorm.variance_epsilon,
            ),
        ),
        ("packed gate/up projection", lambda: layer.mlp.gate_up_proj(mlp_input)),
        ("packed SwiGLU", lambda: packed_swiglu_native(packed)),
        ("down_proj", lambda: layer.mlp.down_proj(activated)),
        ("residual add", lambda: mlp_residual + down),
    )
    rows = [
        PhaseRow(name, 1, _batched_phase_median(operation, warmup, repetitions))
        for name, operation in operations
    ]

    new_k = k_rope.contiguous()
    new_v = v.contiguous()

    def prepare_dynamic() -> Callable[[], object]:
        cache = DynamicLayer()
        cache.keys = cached_k.clone()
        cache.values = cached_v.clone()
        cache.is_initialized = True
        return lambda: cache.update(new_k, new_v)

    dynamic_ms = _event_median(lambda: None, warmup, repetitions, prepare_dynamic)

    static = StaticLayer(max_cache_len=context + 1)
    static.lazy_initialization(new_k, new_v)
    static.keys[..., :context, :].copy_(cached_k)
    static.values[..., :context, :].copy_(cached_v)

    def static_update() -> object:
        static.cumulative_length.fill_(context)
        return static.update(new_k, new_v)

    static_ms = _event_median(static_update, warmup, repetitions)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_update()
    captured_static_ms = _event_median(graph.replay, warmup, repetitions)
    rows.extend(
        (
            PhaseRow("DynamicCache update, eager", 1, dynamic_ms),
            PhaseRow("StaticCache update, eager", 1, static_ms),
            PhaseRow("StaticCache update, captured", 1, captured_static_ms),
        )
    )
    del (
        x,
        residual,
        normalized,
        q,
        k,
        v,
        q_rope,
        k_rope,
        cached_k,
        cached_v,
        expanded_k,
        expanded_v,
        scores,
        mask,
        probabilities,
        attended,
        layout,
        attention_output,
        mlp_input,
        mlp_residual,
        packed,
        activated,
        down,
        new_k,
        new_v,
        static,
        graph,
    )
    return tuple(rows)


def _intermediates(config: Any, length: int, prefill: bool) -> tuple[IntermediateRow, ...]:
    hidden = int(config.hidden_size)
    heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    head_dim = hidden // heads
    query = length if prefill else 1
    packed_width = hidden + 2 * kv_heads * head_dim
    rows = (
        ("packed QKV", (1, query, packed_width), "allocation", "packed QKV GEMM", "split views"),
        ("Q after RoPE", (1, heads, query, head_dim), "allocation", "Flux RoPE", "QK^T"),
        ("K after RoPE", (1, kv_heads, query, head_dim), "allocation", "Flux RoPE", "cache update"),
        ("repeated K", (1, heads, length, head_dim), "materialized allocation", "repeat_kv reshape", "QK^T"),
        ("repeated V", (1, heads, length, head_dim), "materialized allocation", "repeat_kv reshape", "attention x V"),
        ("attention scores", (1, heads, query, length), "allocation", "QK^T", "score post-processing"),
        ("attention probabilities", (1, heads, query, length), "allocation", "Flux softmax", "attention x V"),
        ("attention output", (1, heads, query, head_dim), "allocation", "attention x V", "layout transpose"),
        ("attention layout", (1, query, hidden), "allocation" if query > 1 else "view/no copy", "transpose + contiguous", "o_proj"),
        ("packed gate/up", (1, query, 2 * int(config.intermediate_size)), "allocation", "packed gate/up GEMM", "packed SwiGLU"),
        ("SwiGLU output", (1, query, int(config.intermediate_size)), "allocation", "Flux packed SwiGLU", "down_proj"),
    )
    return tuple(
        IntermediateRow(name, shape, torch.tensor(shape).prod().item() * 4 / MIB, allocation, producer, consumer)
        for name, shape, allocation, producer, consumer in rows
    )


def _print_environment(model: torch.nn.Module, args: argparse.Namespace) -> None:
    print("SmolLM2 remaining-path profile")
    print(f"  Model: {MODEL_ID}@{MODEL_REVISION}")
    print(f"  Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"  PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}")
    print(f"  Transformers: {__import__('transformers').__version__}")
    print(f"  GPU: {torch.cuda.get_device_name()}; capability={torch.cuda.get_device_capability()}")
    print(f"  config: {inspect_config(model.config)}")
    print(f"  Flux modules: {flux_operator_counts(model)}")
    print(f"  Flux categories: {getattr(model, '_flux_operator_categories')}")
    print("  FP32; eager attention; TF32 disabled; deterministic algorithms enabled")
    print(f"  warmup={args.warmup}; samples={args.repetitions}; statistic=median")


def _print_timings(title: str, rows: Sequence[TimingRow]) -> None:
    print(f"\n{title}")
    print(f"{'tokens':>8} {'median ms':>12} {'tokens/s':>12}")
    for row in rows:
        print(f"{row.size:>8} {row.milliseconds:>12.4f} {1000.0 / row.milliseconds:>12.2f}")


def _print_profiles(profiles: Sequence[ProfileSummary], layers: int) -> None:
    for item in profiles:
        print(f"\nProfiler: {item.workload}, size={item.size}")
        print(
            f"  unprofiled median={item.baseline_ms:.4f} ms; profiled event={item.profiled_ms:.4f} ms; "
            f"summed kernels={item.kernel_ms:.4f} ms; launches={item.kernel_launches}"
        )
        gemm_names = {
            "packed gate/up GEMV",
            "down_proj GEMV",
            "packed QKV/o_proj GEMV",
            "LM-head GEMV",
            "attention x V batched GEMV",
            "QK^T batched GEMV",
            "GEMV",
            "GEMM",
        }
        custom_count = sum(count for name, count, _ in item.kernel_inventory if name.startswith("Flux "))
        custom_ms = sum(milliseconds for name, _, milliseconds in item.kernel_inventory if name.startswith("Flux "))
        gemm_count = sum(count for name, count, _ in item.kernel_inventory if name in gemm_names)
        gemm_ms = sum(milliseconds for name, _, milliseconds in item.kernel_inventory if name in gemm_names)
        movement_names = {
            "deterministic/safety fill",
            "cat copy",
            "index_copy",
            "index_copy radix sort",
            "index_copy async assertions",
            "index_copy scatter",
        }
        movement_count = sum(count for name, count, _ in item.kernel_inventory if name in movement_names)
        movement_ms = sum(milliseconds for name, _, milliseconds in item.kernel_inventory if name in movement_names)
        print(
            f"  semantic GEMM/GEMV operations=181; detected cuBLAS kernels={gemm_count} ({gemm_ms:.4f} ms), "
            f"Flux custom={custom_count} ({custom_ms:.4f} ms), "
            f"fill/cat/index={movement_count} ({movement_ms:.4f} ms)"
        )
        if item.phases:
            print(f"  {'phase':<31} {'calls':>7} {'total ms':>11} {'us/layer':>11} {'model %':>9}")
            for row in item.phases:
                per_layer = row.milliseconds * 1000.0 / layers
                share = 100.0 * row.milliseconds / item.baseline_ms
                print(f"  {row.name:<31} {row.calls:>7} {row.milliseconds:>11.4f} {per_layer:>11.3f} {share:>8.2f}%")
        print("  CUDA kernel families (all launches, descending device time):")
        for name, count, milliseconds in item.kernel_inventory[:20]:
            print(f"    {name:<53} count={count:>4} total={milliseconds:>8.4f} ms")
        if item.fill_inventory:
            print("  fill/uninitialized-data kernels:")
            for name, count, milliseconds in item.fill_inventory:
                print(f"    {name:<53} count={count:>4} total={milliseconds:>8.4f} ms")
        else:
            print("  fill/uninitialized-data kernels: none observed")


def _print_microbenchmarks(context: int, rows: Sequence[PhaseRow]) -> None:
    print(f"\nIsolated layer-0 phase corroboration at context {context}")
    print("  Each ordinary phase is the median of 30-call batches, divided by 30.")
    print(f"  {'phase':<39} {'us/call':>12} {'x30 ms':>12}")
    for row in rows:
        print(f"  {row.name:<39} {row.milliseconds * 1000.0:>12.3f} {row.milliseconds * 30:>12.4f}")


def _print_intermediates(config: Any, length: int, prefill: bool) -> None:
    label = "prefill" if prefill else "decode"
    print(f"\nMaterialized intermediates: {label}, effective attention length {length}")
    print(f"  {'tensor':<24} {'shape':<24} {'MiB':>9} {'storage':<24} {'producer -> consumer'}")
    for row in _intermediates(config, length, prefill):
        print(
            f"  {row.name:<24} {str(row.shape):<24} {row.size_mib:>9.3f} "
            f"{row.allocation:<24} {row.producer} -> {row.consumer}"
        )


def _print_gemm_inventory(config: Any, decode_contexts: Sequence[int], prefill_lengths: Sequence[int]) -> None:
    hidden = int(config.hidden_size)
    heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    head_dim = hidden // heads
    intermediate = int(config.intermediate_size)
    print("\nExact GEMM/GEMV inventory per complete forward")
    print("  Decode uses M=1; prefill uses M=sequence length except the logits_to_keep=1 LM head.")
    print(f"  packed QKV: 30 x [M,{hidden}] @ [{hidden},{hidden + 2 * kv_heads * head_dim}]")
    print(f"  QK^T:       30 x batch={heads}, [M,{head_dim}] @ [{head_dim},attention_length]")
    print(f"  attention V:30 x batch={heads}, [M,attention_length] @ [attention_length,{head_dim}]")
    print(f"  o_proj:     30 x [M,{hidden}] @ [{hidden},{hidden}]")
    print(f"  gate/up:    30 x [M,{hidden}] @ [{hidden},{2 * intermediate}]")
    print(f"  down_proj:  30 x [M,{intermediate}] @ [{intermediate},{hidden}]")
    print(f"  LM head:     1 x [1,{hidden}] @ [{hidden},{int(config.vocab_size)}]")
    print("  Total: 181 GEMM/GEMV launches per forward (120 projections + 60 attention + 1 LM head).")
    print(f"  eager decode attention lengths: {tuple(context + 1 for context in decode_contexts)}")
    print("  graph attention length equals context + captured replay capacity; masked future slots are still multiplied.")
    print(f"  prefill M/attention lengths: {tuple(prefill_lengths)}")


def _print_custom_inventory(prefill: bool) -> None:
    print("\nFlux custom operator inventory per complete forward")
    print("  RMSNorm=31; residual-RMSNorm=30; RoPE=30; packed SwiGLU=30")
    if prefill:
        print("  fused attention-score softmax=30; total custom calls/kernels=151")
    else:
        print("  softmax=30 (scale and optional mask add remain separate); total custom calls/kernels=151")


def _required_ops_available() -> bool:
    return all(
        (
            native_rmsnorm_is_available(),
            native_residual_rmsnorm_is_available(),
            native_rope_is_available(),
            native_softmax_is_available(),
            native_attention_score_softmax_is_available(),
            native_packed_swiglu_is_available(),
        )
    )


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not _required_ops_available():
        raise RuntimeError("all retained Flux native operators must be built")
    _configure_runtime()
    model = enable_flux_ops(load_model("cuda"), operators=ALL_OPERATORS)
    _print_environment(model, args)
    maximum = int(model.config.max_position_embeddings)
    decode_window = max(args.repetitions, args.graph_replays)
    decode_contexts = tuple(
        value
        for value in args.decode_contexts
        if value + args.warmup + decode_window <= maximum
    )
    prefill_lengths = tuple(value for value in args.prefill_lengths if value <= maximum)
    if not decode_contexts or not prefill_lengths:
        raise RuntimeError("all requested workloads exceed max_position_embeddings")

    print("\nStabilizing clocks with untimed optimized prefill and graph decode...", flush=True)
    if args.stabilization_iterations:
        stabilization_ids = _input_ids(min(prefill_lengths), model.config.vocab_size)
        with torch.inference_mode():
            for _ in range(args.stabilization_iterations):
                output = model(
                    input_ids=stabilization_ids,
                    use_cache=True,
                    logits_to_keep=1,
                )
                del output
        graph = _make_graph(model, min(decode_contexts), args.stabilization_iterations)
        for _ in range(args.stabilization_iterations):
            graph.replay()
        torch.cuda.synchronize()
        del graph, stabilization_ids

    eager_rows = []
    graph_rows = []
    prefill_rows = []
    with torch.inference_mode():
        for context in decode_contexts:
            print(f"Timing eager decode at context {context}...", flush=True)
            eager_rows.append(TimingRow(context, _time_eager_decode(model, context, args.warmup, args.repetitions)))
            print(f"Timing graph replay at context {context}...", flush=True)
            graph_rows.append(TimingRow(context, _time_graph_decode(model, context, args.warmup, args.graph_replays)))
        for length in prefill_lengths:
            print(f"Timing prefill length {length}...", flush=True)
            prefill_rows.append(TimingRow(length, _time_prefill(model, length, args.warmup, args.repetitions)))

    _print_timings("One-token eager cached decode", eager_rows)
    _print_timings("Fixed-shape CUDA-Graph pure replay", graph_rows)
    _print_timings("Optimized prefill (logits_to_keep=1)", prefill_rows)

    profiles: list[ProfileSummary] = []
    if not args.skip_profiler:
        eager_baseline = {row.size: row.milliseconds for row in eager_rows}
        graph_baseline = {row.size: row.milliseconds for row in graph_rows}
        with torch.inference_mode():
            for context in args.profile_contexts:
                if context not in eager_baseline:
                    continue
                print(f"Profiling eager decode at context {context}...", flush=True)
                profiles.append(_profile_eager_decode(model, context, eager_baseline[context]))
                print(f"Profiling graph replay at context {context}...", flush=True)
                profiles.append(
                    _profile_graph_decode(
                        model,
                        context,
                        graph_baseline[context],
                        args.warmup + args.graph_replays,
                    )
                )
            for row in prefill_rows:
                print(f"Profiling prefill length {row.size}...", flush=True)
                profiles.append(_profile_prefill(model, row.size, row.milliseconds))
        _print_profiles(profiles, int(model.config.num_hidden_layers))

    if not args.skip_microbenchmarks:
        with torch.inference_mode():
            for context in decode_contexts:
                print(f"Microbenchmarking layer phases at context {context}...", flush=True)
                rows = _microbenchmark_layer(model, context, args.warmup, args.repetitions)
                _print_microbenchmarks(context, rows)

    _print_gemm_inventory(model.config, decode_contexts, prefill_lengths)
    _print_custom_inventory(prefill=False)
    _print_custom_inventory(prefill=True)
    _print_intermediates(model.config, max(decode_contexts) + 1, False)
    for length in prefill_lengths:
        _print_intermediates(model.config, length, True)
    print("\nNotes")
    print("  repeat_kv expand is a view, but its following reshape materializes repeated K/V when groups > 1.")
    print("  DynamicCache update concatenates and reallocates K and V; StaticCache update uses two index_copy_ operations per layer.")
    print("  Graph attention uses the full fixed cache capacity, including masked future slots.")
    print("  Profiler phase times are diagnostic device-time attribution; unprofiled medians remain authoritative.")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
