"""Validate and benchmark fused FP32 one-token SmolLM2 GQA attention.

Run from the repository root after rebuilding the native extension:

    build/python3119/python.exe benchmarks/benchmark_smollm2_gqa_decode_attention.py
"""

from __future__ import annotations

import argparse
import copy
import gc
import multiprocessing
import os
import statistics
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile
from transformers import DynamicCache
from transformers.models.llama.modeling_llama import repeat_kv

from flux.model.smollm2 import MODEL_ID, MODEL_REVISION, load_model
from flux.model.smollm2_cuda_graph import FluxCUDAGraphDecode
from flux.model.smollm2_flux import (
    FLUX_GQA_DECODE_ATTENTION_CATEGORY,
    FLUX_OPERATOR_CATEGORIES,
    FLUX_PACKED_MLP_CATEGORY,
    FLUX_PACKED_QKV_CATEGORY,
    FLUX_PACKED_SWIGLU_CATEGORY,
    enable_flux_ops,
)
from flux.ops import (
    gqa_decode_attention_native,
    native_gqa_decode_attention_is_available,
)


CONTEXTS = (128, 512, 1024, 2048, 4096)
HEADS = 9
KV_HEADS = 3
HEAD_DIM = 64
SCALE = HEAD_DIM**-0.5
MIB = 1024.0**2
# QK, softmax, and PV use different FP32 reduction trees in the two paths.
# Model tolerances additionally cover propagation through all 30 layers.
OP_RTOL = 2e-5
OP_ATOL = 1e-6
MODEL_RTOL = 2e-4
MODEL_ATOL = 2e-5
BASE_OPERATORS = FLUX_OPERATOR_CATEGORIES | {
    FLUX_PACKED_QKV_CATEGORY,
    FLUX_PACKED_MLP_CATEGORY,
    FLUX_PACKED_SWIGLU_CATEGORY,
}
FUSED_OPERATORS = BASE_OPERATORS | {FLUX_GQA_DECODE_ATTENTION_CATEGORY}


@dataclass(frozen=True)
class Pair:
    current_ms: float
    fused_ms: float

    @property
    def speedup(self) -> float:
        return self.current_ms / self.fused_ms


@dataclass(frozen=True)
class Numerical:
    max_absolute: float
    mean_absolute: float
    max_relative: float
    mean_relative: float
    per_head_absolute: tuple[float, ...]


@dataclass(frozen=True)
class ProfileResult:
    kernels: int
    device_ms: float
    names: tuple[tuple[str, int], ...]


def _parse_int_list(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return result


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=_parse_int_list, default=CONTEXTS)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--model-repetitions", type=int, default=15)
    parser.add_argument("--generation-tokens", type=int, default=8)
    parser.add_argument("--stabilization-iterations", type=int, default=50)
    args = parser.parse_args()
    if (args.warmup < 0 or args.repetitions < 1 or args.model_repetitions < 1
            or args.stabilization_iterations < 0):
        parser.error("warmup may be zero; repetition counts must be positive")
    return args


def _configure() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _ids(length: int, vocab: int) -> torch.Tensor:
    return (((torch.arange(length) * 17 + 11) % vocab).unsqueeze(0)).cuda()


def _current_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    expanded_key = repeat_kv(key, HEADS // KV_HEADS)
    expanded_value = repeat_kv(value, HEADS // KV_HEADS)
    scores = torch.matmul(query, expanded_key.transpose(2, 3))
    probabilities = torch.softmax(scores * SCALE + mask, dim=-1)
    return torch.matmul(probabilities, expanded_value)


def _record(operation: Callable[[], object]) -> tuple[torch.cuda.Event, torch.cuda.Event]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    operation()
    end.record()
    return start, end


def _paired(
    current: Callable[[], object],
    fused: Callable[[], object],
    warmup: int,
    repetitions: int,
) -> Pair:
    for _ in range(warmup):
        current()
        fused()
    torch.cuda.synchronize()
    samples: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
        "current": [], "fused": []
    }
    for index in range(repetitions):
        order = (("current", current), ("fused", fused))
        if index % 2:
            order = tuple(reversed(order))
        for name, operation in order:
            samples[name].append(_record(operation))
    torch.cuda.synchronize()
    return Pair(*(
        statistics.median(start.elapsed_time(end) for start, end in samples[name])
        for name in ("current", "fused")
    ))


def _paired_prepared(
    current_prepare: Callable[[], Callable[[], object]],
    fused_prepare: Callable[[], Callable[[], object]],
    warmup: int,
    repetitions: int,
) -> Pair:
    samples = {"current": [], "fused": []}
    for index in range(warmup + repetitions):
        order = (("current", current_prepare), ("fused", fused_prepare))
        if index % 2:
            order = tuple(reversed(order))
        for name, prepare in order:
            operation = prepare()
            event = _record(operation)
            if index >= warmup:
                samples[name].append(event)
    torch.cuda.synchronize()
    return Pair(*(
        statistics.median(start.elapsed_time(end) for start, end in samples[name])
        for name in ("current", "fused")
    ))


def _numerical(actual: torch.Tensor, expected: torch.Tensor) -> Numerical:
    difference = (actual - expected).abs()
    relative = difference / expected.abs().clamp_min(1e-8)
    return Numerical(
        float(difference.max().item()),
        float(difference.mean().item()),
        float(relative.max().item()),
        float(relative.mean().item()),
        tuple(float(difference[:, head].max().item()) for head in range(actual.shape[1])),
    )


def _cache_numerical(actual_layers: Any, expected_layers: Any, field: str) -> Numerical:
    max_absolute = 0.0
    max_relative = 0.0
    absolute_sum = 0.0
    relative_sum = 0.0
    elements = 0
    for actual_layer, expected_layer in zip(
        actual_layers, expected_layers, strict=True
    ):
        actual = getattr(actual_layer, field)
        expected = getattr(expected_layer, field)
        difference = (actual - expected).abs()
        relative = difference / expected.abs().clamp_min(1e-8)
        max_absolute = max(max_absolute, float(difference.max().item()))
        max_relative = max(max_relative, float(relative.max().item()))
        absolute_sum += float(difference.sum().item())
        relative_sum += float(relative.sum().item())
        elements += difference.numel()
    return Numerical(
        max_absolute,
        absolute_sum / elements,
        max_relative,
        relative_sum / elements,
        (),
    )


def _clone_cache(cache: Any, config: Any) -> DynamicCache:
    return DynamicCache(
        [(layer.keys.detach().clone(), layer.values.detach().clone()) for layer in cache.layers],
        config=config,
    )


def _isolated(context: int, warmup: int, repetitions: int) -> tuple[Pair, Numerical]:
    generator = torch.Generator(device="cuda").manual_seed(1000 + context)
    query = torch.randn((1, HEADS, 1, HEAD_DIM), generator=generator, device="cuda")
    key = torch.randn((1, KV_HEADS, context, HEAD_DIM), generator=generator, device="cuda")
    value = torch.randn_like(key)
    mask = torch.zeros((1, 1, 1, context), device="cuda")
    expected = _current_attention(query, key, value, mask)
    actual = gqa_decode_attention_native(query, key, value, mask, SCALE)
    torch.testing.assert_close(actual, expected, rtol=OP_RTOL, atol=OP_ATOL)
    timing = _paired(
        lambda: _current_attention(query, key, value, mask),
        lambda: gqa_decode_attention_native(query, key, value, mask, SCALE),
        warmup,
        repetitions,
    )
    return timing, _numerical(actual, expected)


def _prefill(model: torch.nn.Module, context: int) -> tuple[torch.Tensor, Any]:
    output = model(input_ids=_ids(context, model.config.vocab_size), use_cache=True, logits_to_keep=1)
    return output.logits, output.past_key_values


def _eager_pair(
    current: torch.nn.Module,
    fused: torch.nn.Module,
    context: int,
    warmup: int,
    repetitions: int,
) -> tuple[Pair, Numerical, Numerical, Numerical]:
    token = _ids(context + 1, current.config.vocab_size)[:, -1:]
    _, current_base = _prefill(current, context)
    _, fused_base = _prefill(fused, context)

    def prepare(model: torch.nn.Module, base: Any) -> Callable[[], object]:
        cache = _clone_cache(base, model.config)
        return lambda: model(
            input_ids=token, past_key_values=cache, use_cache=True, logits_to_keep=1
        )

    timing = _paired_prepared(
        lambda: prepare(current, current_base),
        lambda: prepare(fused, fused_base),
        warmup,
        repetitions,
    )
    current_call = prepare(current, current_base)
    fused_call = prepare(fused, fused_base)
    current_output = current_call()
    fused_output = fused_call()
    logits_error = _numerical(fused_output.logits, current_output.logits)
    key_error = _cache_numerical(
        fused_output.past_key_values.layers,
        current_output.past_key_values.layers,
        "keys",
    )
    value_error = _cache_numerical(
        fused_output.past_key_values.layers,
        current_output.past_key_values.layers,
        "values",
    )
    torch.testing.assert_close(
        fused_output.logits,
        current_output.logits,
        rtol=MODEL_RTOL,
        atol=MODEL_ATOL,
    )
    return timing, logits_error, key_error, value_error


def _layer_pair(
    current: torch.nn.Module,
    fused: torch.nn.Module,
    context: int,
    warmup: int,
    repetitions: int,
) -> tuple[Pair, Numerical]:
    hidden = torch.randn((1, 1, current.config.hidden_size), device="cuda")
    position = torch.tensor([[context]], device="cuda")
    current_position = current.model.rotary_emb(hidden, position)
    fused_position = fused.model.rotary_emb(hidden, position)
    mask = torch.zeros((1, 1, 1, context + 1), device="cuda")
    _, current_base = _prefill(current, context)
    _, fused_base = _prefill(fused, context)

    def prepare(model: torch.nn.Module, base: Any, embeddings: Any) -> Callable[[], object]:
        cache = _clone_cache(base, model.config)
        return lambda: model.model.layers[0](
            hidden,
            attention_mask=mask,
            position_ids=position,
            past_key_values=cache,
            use_cache=True,
            position_embeddings=embeddings,
        )

    timing = _paired_prepared(
        lambda: prepare(current, current_base, current_position),
        lambda: prepare(fused, fused_base, fused_position),
        warmup,
        repetitions,
    )
    expected = prepare(current, current_base, current_position)()
    actual = prepare(fused, fused_base, fused_position)()
    error = _numerical(actual, expected)
    torch.testing.assert_close(
        actual, expected, rtol=MODEL_RTOL, atol=MODEL_ATOL
    )
    return timing, error


def _graph_pair(
    current: torch.nn.Module,
    fused: torch.nn.Module,
    context: int,
    warmup: int,
    repetitions: int,
) -> tuple[Pair]:
    steps = warmup + repetitions
    prompt = _ids(context, current.config.vocab_size)
    current_graph = FluxCUDAGraphDecode.capture(current, prompt, max_decode_steps=steps)
    fused_graph = FluxCUDAGraphDecode.capture(fused, prompt, max_decode_steps=steps)
    timing = _paired(current_graph.graph.replay, fused_graph.graph.replay, warmup, repetitions)
    return (timing,)


def _graph_memory_worker(
    connection: Any, context: int, steps: int, fused: bool
) -> None:
    """Measure one graph in a fresh process to avoid capture-order attribution."""
    try:
        _configure()
        model = load_model("cuda")
        operators = FUSED_OPERATORS if fused else BASE_OPERATORS
        enable_flux_ops(model, operators=operators)
        prompt = _ids(context, model.config.vocab_size)
        state = FluxCUDAGraphDecode.capture(
            model, prompt, max_decode_steps=steps
        )
        connection.send(("ok", state.memory.graph_pool_bytes))
    except BaseException as error:
        connection.send(("error", repr(error)))
    finally:
        connection.close()


def _independent_graph_pool_bytes(context: int, steps: int) -> tuple[int, int]:
    process_context = multiprocessing.get_context("spawn")
    results: list[int] = []
    for fused in (False, True):
        receive, send = process_context.Pipe(duplex=False)
        process = process_context.Process(
            target=_graph_memory_worker,
            args=(send, context, steps, fused),
        )
        process.start()
        send.close()
        process.join()
        if process.exitcode != 0 or not receive.poll():
            raise RuntimeError(
                f"independent graph-memory probe failed with exit code "
                f"{process.exitcode}"
            )
        status, payload = receive.recv()
        receive.close()
        if status != "ok":
            raise RuntimeError(
                f"independent graph-memory probe failed: {payload}"
            )
        results.append(payload)
    return results[0], results[1]


def _profile(operation: Callable[[], object]) -> ProfileResult:
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as result:
        operation()
    torch.cuda.synchronize()
    events = [event for event in result.events() if event.device_type == DeviceType.CUDA]
    names = Counter(event.name for event in events)
    return ProfileResult(
        len(events),
        sum(float(event.self_device_time_total) for event in events) / 1000.0,
        tuple(names.most_common()),
    )


def _peak_decode(model: torch.nn.Module, context: int) -> int:
    token = _ids(context + 1, model.config.vocab_size)[:, -1:]
    _, cache = _prefill(model, context)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    model(input_ids=token, past_key_values=cache, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - baseline


def main() -> int:
    args = _args()
    if not torch.cuda.is_available() or not native_gqa_decode_attention_is_available():
        raise RuntimeError("CUDA and the rebuilt Flux GQA operator are required")
    _configure()
    print("Fused one-token GQA decode-attention milestone")
    print(f"  model={MODEL_ID}@{MODEL_REVISION}")
    print(f"  Python={sys.version.split()[0]}; PyTorch={torch.__version__}; CUDA={torch.version.cuda}")
    print(f"  GPU={torch.cuda.get_device_name()}; capability={torch.cuda.get_device_capability()}")
    print("  FP32; TF32 disabled; deterministic algorithms; CUDA events; alternating samples")
    print(f"  warmup={args.warmup}; isolated samples={args.repetitions}; model samples={args.model_repetitions}")

    isolated: dict[int, tuple[Pair, Numerical]] = {}
    with torch.inference_mode():
        for context in args.contexts:
            print(f"Isolated attention context {context}...", flush=True)
            isolated[context] = _isolated(context, args.warmup, args.repetitions)

    print("Loading two identically weighted optimized models...", flush=True)
    source = load_model("cuda")
    current = copy.deepcopy(source)
    fused = copy.deepcopy(source)
    del source
    enable_flux_ops(current, operators=BASE_OPERATORS)
    enable_flux_ops(fused, operators=FUSED_OPERATORS)
    if args.stabilization_iterations:
        print("Stabilizing GPU clocks...", flush=True)
        stabilization_ids = _ids(min(1024, max(args.contexts)), current.config.vocab_size)
        with torch.inference_mode():
            for _ in range(args.stabilization_iterations):
                fused(
                    input_ids=stabilization_ids,
                    use_cache=True,
                    logits_to_keep=1,
                )
        torch.cuda.synchronize()
        del stabilization_ids

    layer: dict[int, tuple[Pair, Numerical]] = {}
    eager: dict[int, tuple[Pair, Numerical, Numerical, Numerical]] = {}
    graph: dict[int, tuple[Pair]] = {}
    with torch.inference_mode():
        for context in args.contexts:
            print(f"Layer/eager/graph context {context}...", flush=True)
            layer[context] = _layer_pair(
                current, fused, context, args.warmup, args.model_repetitions
            )
            eager[context] = _eager_pair(
                current, fused, context, args.warmup, args.model_repetitions
            )
            graph[context] = _graph_pair(
                current, fused, context, args.warmup, args.model_repetitions
            )

        prompt = _ids(min(args.contexts), current.config.vocab_size)
        current_tokens = current.generate(
            prompt, do_sample=False, max_new_tokens=args.generation_tokens, use_cache=True
        )
        fused_tokens = fused.generate(
            prompt, do_sample=False, max_new_tokens=args.generation_tokens, use_cache=True
        )
        greedy_equal = torch.equal(current_tokens, fused_tokens)

        profile_context = max(args.contexts)
        query = torch.randn((1, HEADS, 1, HEAD_DIM), device="cuda")
        key = torch.randn((1, KV_HEADS, profile_context, HEAD_DIM), device="cuda")
        value = torch.randn_like(key)
        mask = torch.zeros((1, 1, 1, profile_context), device="cuda")
        current_profile = _profile(lambda: _current_attention(query, key, value, mask))
        fused_profile = _profile(
            lambda: gqa_decode_attention_native(query, key, value, mask, SCALE)
        )
        current_peak = _peak_decode(current, profile_context)
        fused_peak = _peak_decode(fused, profile_context)

    print("Measuring independent CUDA-Graph pools...", flush=True)
    current_pool, fused_pool = _independent_graph_pool_bytes(
        profile_context, args.warmup + args.model_repetitions
    )

    print("\nIsolated attention")
    print(
        f"{'L':>6} {'current us':>12} {'fused us':>11} {'speedup':>9} "
        f"{'max abs':>11} {'mean abs':>11} {'max rel':>11} {'mean rel':>11}"
    )
    for context, (timing, numerical) in isolated.items():
        print(f"{context:>6} {timing.current_ms*1000:>12.3f} {timing.fused_ms*1000:>11.3f} "
              f"{timing.speedup:>8.3f}x {numerical.max_absolute:>11.3g} "
              f"{numerical.mean_absolute:>11.3g} {numerical.max_relative:>11.3g} "
              f"{numerical.mean_relative:>11.3g}")
        print("       per-head max abs: " + ", ".join(f"{value:.3g}" for value in numerical.per_head_absolute))

    for title, rows in (("Complete decoder layer", layer), ("Full eager decode", eager), ("CUDA-Graph replay", graph)):
        print(f"\n{title}")
        print(f"{'L':>6} {'current ms':>12} {'fused ms':>11} {'speedup':>9}")
        for context, row in rows.items():
            timing = row[0]
            print(f"{context:>6} {timing.current_ms:>12.4f} {timing.fused_ms:>11.4f} {timing.speedup:>8.3f}x")

    print("\nIntegrated numerical differences")
    print(
        f"{'L':>6} {'layer max/mean':>23} {'logit max/mean':>23} "
        f"{'cache K max/mean':>23} {'cache V max/mean':>23}"
    )
    for context in args.contexts:
        errors = (layer[context][1], *eager[context][1:])
        formatted = " ".join(
            f"{error.max_absolute:>10.3g}/{error.mean_absolute:<10.3g}"
            for error in errors
        )
        print(f"{context:>6} {formatted}")
    print(f"  greedy token equality ({args.generation_tokens} tokens): {greedy_equal}")

    repeated_bytes = 2 * HEADS * profile_context * HEAD_DIM * 4
    score_bytes = 2 * HEADS * profile_context * 4
    workspace_bytes = HEADS * ((profile_context + 127) // 128) * (HEAD_DIM + 2) * 4
    cache_bytes = 30 * 2 * KV_HEADS * profile_context * HEAD_DIM * 4
    repeat_copy_traffic = 2 * (KV_HEADS + HEADS) * profile_context * HEAD_DIM * 4
    print(f"\nProfiler and memory at context {profile_context}")
    print(f"  current attention: kernels={current_profile.kernels}, device={current_profile.device_ms:.4f} ms")
    print(f"  fused attention: kernels={fused_profile.kernels}, device={fused_profile.device_ms:.4f} ms")
    print(f"  current kernel names: {current_profile.names}")
    print(f"  fused kernel names: {fused_profile.names}")
    print("  attention allocations: current=7, fused=2 (output + compact partial workspace)")
    print(f"  repeated K/V eliminated per layer: {repeated_bytes/MIB:.3f} MiB")
    print(f"  minimum repeat-copy traffic eliminated per layer: {repeat_copy_traffic/MIB:.3f} MiB")
    print(f"  score/probability storage eliminated per layer: {score_bytes/MIB:.3f} MiB")
    print(f"  fused partial workspace per layer call: {workspace_bytes/MIB:.3f} MiB")
    print(f"  unchanged 30-layer unexpanded K/V cache storage: {cache_bytes/MIB:.3f} MiB")
    print(f"  eager decode incremental peak: current={current_peak/MIB:.3f} MiB, fused={fused_peak/MIB:.3f} MiB")
    print(
        f"  independent graph pool: current={current_pool/MIB:.3f} MiB, "
        f"fused={fused_pool/MIB:.3f} MiB"
    )
    print("\nKernel design: 128-token partial online softmax + per-head max-rescaled reduction; "
          "KV-grouped stage 1 above capacity 4096; caller current stream; "
          "valid length read from the StaticCache device scalar.")
    print("\nNative GQA algorithmic traffic (requested bytes; hardware counters may differ)")
    print(
        f"{'L':>6} {'chunks':>7} {'KV minimum':>12} {'KV stage 1':>12} "
        f"{'Q read':>10} {'mask':>10} {'workspace W/R':>24} {'output':>10}"
    )
    for context in (512, 1024, 1280, 2048, 4096, 8192):
        chunks = (context + 127) // 128
        logical_kv = 2 * KV_HEADS * context * HEAD_DIM * 4
        stage1_kv = logical_kv if context > 4096 else 2 * HEADS * context * HEAD_DIM * 4
        query_bytes = HEADS * chunks * HEAD_DIM * 4
        mask_bytes = HEADS * context * 4
        workspace_write = HEADS * chunks * (HEAD_DIM + 2) * 4
        # The reduction issues three scalar-state reads and two reads per
        # output dimension and chunk. Caches can reduce physical DRAM traffic.
        workspace_read = HEADS * chunks * (3 + 2 * HEAD_DIM) * 4
        output_bytes = HEADS * HEAD_DIM * 4
        print(
            f"{context:>6} {chunks:>7} {logical_kv/MIB:>11.3f}M "
            f"{stage1_kv/MIB:>11.3f}M {query_bytes/1024:>9.1f}K "
            f"{mask_bytes/1024:>9.1f}K "
            f"{workspace_write/1024:>9.1f}K/{workspace_read/1024:<9.1f}K "
            f"{output_bytes/1024:>9.2f}K"
        )
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
