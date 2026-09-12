"""Validate and benchmark fused one-token packed-QKV post-processing.

The retained baseline already uses packed QKV, fused residual RMSNorm, Flux
score processing, native one-token GQA attention, and fixed-shape CUDA Graphs.
This benchmark changes only the packed-QKV -> RoPE -> StaticCache boundary.
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
from transformers import StaticCache
from transformers.cache_utils import StaticLayer

from flux.model.smollm2 import MODEL_ID, MODEL_REVISION, load_model
from flux.model.smollm2_cuda_graph import FluxCUDAGraphDecode
from flux.model.smollm2_flux import (
    FLUX_GQA_DECODE_ATTENTION_CATEGORY,
    FLUX_OPERATOR_CATEGORIES,
    FLUX_PACKED_MLP_CATEGORY,
    FLUX_PACKED_QKV_CATEGORY,
    FLUX_PACKED_QKV_ROPE_CACHE_CATEGORY,
    FLUX_PACKED_SWIGLU_CATEGORY,
    enable_flux_ops,
)
from flux.ops import packed_qkv_rope_cache_native, rope_native


CONTEXTS = (128, 512, 1024, 2048, 4096)
MIB = 1024.0**2
BASE_OPERATORS = FLUX_OPERATOR_CATEGORIES | {
    FLUX_PACKED_QKV_CATEGORY,
    FLUX_PACKED_MLP_CATEGORY,
    FLUX_PACKED_SWIGLU_CATEGORY,
    FLUX_GQA_DECODE_ATTENTION_CATEGORY,
}
FUSED_OPERATORS = BASE_OPERATORS | {FLUX_PACKED_QKV_ROPE_CACHE_CATEGORY}


@dataclass(frozen=True)
class Pair:
    current_ms: float
    fused_ms: float

    @property
    def speedup(self) -> float:
        return self.current_ms / self.fused_ms


@dataclass(frozen=True)
class Error:
    maximum: float
    mean: float


@dataclass(frozen=True)
class Profile:
    launches: int
    device_ms: float
    names: tuple[tuple[str, int], ...]


def _parse_int_list(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
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
    if (
        args.warmup < 0
        or args.repetitions < 1
        or args.model_repetitions < 1
        or args.generation_tokens < 1
        or args.stabilization_iterations < 0
    ):
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


def _ids(length: int, vocab_size: int) -> torch.Tensor:
    return (((torch.arange(length) * 17 + 11) % vocab_size).unsqueeze(0)).cuda()


def _error(actual: torch.Tensor, expected: torch.Tensor) -> Error:
    difference = (actual - expected).abs().float()
    return Error(float(difference.max().item()), float(difference.mean().item()))


def _paired(
    current: Callable[[], object],
    fused: Callable[[], object],
    warmup: int,
    repetitions: int,
    *,
    prepare_current: Callable[[], None] | None = None,
    prepare_fused: Callable[[], None] | None = None,
) -> Pair:
    for _ in range(warmup):
        if prepare_current is not None:
            prepare_current()
        current()
        if prepare_fused is not None:
            prepare_fused()
        fused()
    torch.cuda.synchronize()
    samples: tuple[list[tuple[torch.cuda.Event, torch.cuda.Event]], ...] = ([], [])
    operations = (current, fused)
    preparations = (prepare_current, prepare_fused)
    final = None
    output: object = None
    for repetition in range(repetitions):
        for index in ((0, 1) if repetition % 2 == 0 else (1, 0)):
            if preparations[index] is not None:
                preparations[index]()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = operations[index]()
            end.record()
            samples[index].append((start, end))
            final = end
    assert final is not None
    final.synchronize()
    del output
    medians = tuple(
        statistics.median(start.elapsed_time(end) for start, end in group)
        for group in samples
    )
    return Pair(*medians)


def _packed_inputs(position: int, capacity: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(1000 + position)
    packed = torch.randn((1, 1, 960), generator=generator, device="cuda")
    frequencies = torch.randn((1, 1, 64), generator=generator, device="cuda")
    cos = frequencies.cos()
    sin = frequencies.sin()
    query = packed[..., :576].view(1, 1, 9, 64).transpose(1, 2)
    key = packed[..., 576:768].view(1, 1, 3, 64).transpose(1, 2)
    value = packed[..., 768:].view(1, 1, 3, 64).transpose(1, 2)
    current_cache = StaticLayer(max_cache_len=capacity)
    fused_cache = StaticLayer(max_cache_len=capacity)
    current_cache.lazy_initialization(key, value)
    fused_cache.lazy_initialization(key, value)
    current_cache.keys.zero_()
    current_cache.values.zero_()
    fused_cache.keys.zero_()
    fused_cache.values.zero_()
    current_cache.cumulative_length.fill_(position)
    fused_cache.cumulative_length.fill_(position)
    return packed, cos, sin, query, key, value, current_cache, fused_cache


def _isolated_post_qkv(
    position: int, warmup: int, repetitions: int
) -> tuple[Pair, Error, Error, Error]:
    capacity = position + 1
    packed, cos, sin, query, key, value, current_cache, fused_cache = (
        _packed_inputs(position, capacity)
    )
    original_keys = current_cache.keys.clone()
    original_values = current_cache.values.clone()

    def prepare_current() -> None:
        current_cache.cumulative_length.fill_(position)

    def prepare_fused() -> None:
        fused_cache.cumulative_length.fill_(position)

    def current() -> torch.Tensor:
        query_output, key_output = rope_native(query, key, cos, sin)
        current_cache.update(key_output, value)
        return query_output

    def fused() -> torch.Tensor:
        return packed_qkv_rope_cache_native(
            packed,
            cos,
            sin,
            fused_cache.keys,
            fused_cache.values,
            fused_cache.cumulative_length,
        )

    timing = _paired(
        current,
        fused,
        warmup,
        repetitions,
        prepare_current=prepare_current,
        prepare_fused=prepare_fused,
    )
    current_cache.keys.copy_(original_keys)
    current_cache.values.copy_(original_values)
    fused_cache.keys.copy_(original_keys)
    fused_cache.values.copy_(original_values)
    prepare_current()
    prepare_fused()
    expected_q = current()
    actual_q = fused()
    torch.cuda.synchronize()
    result = (
        timing,
        _error(actual_q, expected_q),
        _error(fused_cache.keys, current_cache.keys),
        _error(fused_cache.values, current_cache.values),
    )
    del current_cache, fused_cache
    return result


def _prefill_dynamic(model: torch.nn.Module, context: int) -> Any:
    output = model(
        input_ids=_ids(context, model.config.vocab_size),
        use_cache=True,
        logits_to_keep=1,
    )
    return output.past_key_values


def _make_static_cache(
    model: torch.nn.Module, context: int, capacity: int
) -> StaticCache:
    dynamic = _prefill_dynamic(model, context)
    static = StaticCache(config=model.config, max_cache_len=capacity)
    FluxCUDAGraphDecode._initialize_static_cache(static, dynamic)
    return static


def _reset_cache(cache: StaticCache, context: int) -> None:
    for layer in cache.layers:
        layer.cumulative_length.fill_(context)


def _mask(capacity: int, valid_length: int) -> torch.Tensor:
    result = torch.full(
        (1, 1, 1, capacity),
        torch.finfo(torch.float32).min,
        device="cuda",
    )
    result[..., :valid_length].zero_()
    return result


def _layer_pair(
    current_model: torch.nn.Module,
    fused_model: torch.nn.Module,
    context: int,
    warmup: int,
    repetitions: int,
) -> tuple[Pair, Error]:
    capacity = context + 1
    current_cache = _make_static_cache(current_model, context, capacity)
    fused_cache = _make_static_cache(fused_model, context, capacity)
    generator = torch.Generator(device="cuda").manual_seed(2000 + context)
    hidden = torch.randn(
        (1, 1, current_model.config.hidden_size),
        generator=generator,
        device="cuda",
    )
    position_ids = torch.tensor([[context]], device="cuda")
    position_embeddings = current_model.model.rotary_emb(hidden, position_ids)
    attention_mask = _mask(capacity, context + 1)
    current_layer = current_model.model.layers[0]
    fused_layer = fused_model.model.layers[0]

    def current() -> torch.Tensor:
        return current_layer(
            hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            past_key_values=current_cache,
            use_cache=True,
        )

    def fused() -> torch.Tensor:
        return fused_layer(
            hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            past_key_values=fused_cache,
            use_cache=True,
        )

    timing = _paired(
        current,
        fused,
        warmup,
        repetitions,
        prepare_current=lambda: _reset_cache(current_cache, context),
        prepare_fused=lambda: _reset_cache(fused_cache, context),
    )
    _reset_cache(current_cache, context)
    _reset_cache(fused_cache, context)
    expected = current()
    actual = fused()
    torch.cuda.synchronize()
    return timing, _error(actual, expected)


def _eager_static_pair(
    current_model: torch.nn.Module,
    fused_model: torch.nn.Module,
    context: int,
    warmup: int,
    repetitions: int,
) -> tuple[Pair, Error, Error, Error]:
    capacity = context + 1
    current_cache = _make_static_cache(current_model, context, capacity)
    fused_cache = _make_static_cache(fused_model, context, capacity)
    token = _ids(context + 1, current_model.config.vocab_size)[:, -1:]
    position_ids = torch.tensor([[context]], device="cuda")
    attention_mask = _mask(capacity, context + 1)

    def current() -> Any:
        return current_model(
            input_ids=token,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=current_cache,
            use_cache=True,
            logits_to_keep=1,
        )

    def fused() -> Any:
        return fused_model(
            input_ids=token,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=fused_cache,
            use_cache=True,
            logits_to_keep=1,
        )

    timing = _paired(
        current,
        fused,
        warmup,
        repetitions,
        prepare_current=lambda: _reset_cache(current_cache, context),
        prepare_fused=lambda: _reset_cache(fused_cache, context),
    )
    _reset_cache(current_cache, context)
    _reset_cache(fused_cache, context)
    expected = current()
    actual = fused()
    torch.cuda.synchronize()
    key_errors = []
    value_errors = []
    for current_layer, fused_layer in zip(
        current_cache.layers, fused_cache.layers, strict=True
    ):
        key_errors.append(_error(fused_layer.keys, current_layer.keys))
        value_errors.append(_error(fused_layer.values, current_layer.values))
    return (
        timing,
        _error(actual.logits, expected.logits),
        Error(
            max(item.maximum for item in key_errors),
            statistics.mean(item.mean for item in key_errors),
        ),
        Error(
            max(item.maximum for item in value_errors),
            statistics.mean(item.mean for item in value_errors),
        ),
    )


def _graph_pair(
    current_model: torch.nn.Module,
    fused_model: torch.nn.Module,
    context: int,
    warmup: int,
    repetitions: int,
) -> Pair:
    steps = warmup + repetitions + 1
    prompt = _ids(context, current_model.config.vocab_size)
    current = FluxCUDAGraphDecode.capture(
        current_model, prompt, max_decode_steps=steps
    )
    fused = FluxCUDAGraphDecode.capture(fused_model, prompt, max_decode_steps=steps)
    return _paired(
        current.graph.replay,
        fused.graph.replay,
        warmup,
        repetitions,
    )


def _graph_correctness_and_generation(
    current_model: torch.nn.Module,
    fused_model: torch.nn.Module,
    context: int,
    tokens: int,
) -> tuple[Error, Error, Error, bool, bool]:
    prompt = _ids(context, current_model.config.vocab_size)
    current = FluxCUDAGraphDecode.capture(
        current_model, prompt, max_decode_steps=tokens
    )
    fused = FluxCUDAGraphDecode.capture(
        fused_model, prompt, max_decode_steps=tokens
    )
    token = current.prefill_logits.argmax(dim=-1)
    current_tokens = []
    fused_tokens = []
    logit_errors = []
    addresses = fused.stable_addresses()
    for _ in range(tokens):
        expected = current.replay(token)
        actual = fused.replay(token)
        logit_errors.append(_error(actual, expected))
        current_token = expected.argmax(dim=-1)
        fused_token = actual.argmax(dim=-1)
        current_tokens.append(current_token.clone())
        fused_tokens.append(fused_token.clone())
        token = current_token
    key_errors = []
    value_errors = []
    valid_length = context + tokens
    for current_layer, fused_layer in zip(
        current.cache.layers, fused.cache.layers, strict=True
    ):
        key_errors.append(
            _error(
                fused_layer.keys[..., :valid_length, :],
                current_layer.keys[..., :valid_length, :],
            )
        )
        value_errors.append(
            _error(
                fused_layer.values[..., :valid_length, :],
                current_layer.values[..., :valid_length, :],
            )
        )
    return (
        Error(
            max(item.maximum for item in logit_errors),
            statistics.mean(item.mean for item in logit_errors),
        ),
        Error(
            max(item.maximum for item in key_errors),
            statistics.mean(item.mean for item in key_errors),
        ),
        Error(
            max(item.maximum for item in value_errors),
            statistics.mean(item.mean for item in value_errors),
        ),
        torch.equal(torch.cat(current_tokens, dim=-1), torch.cat(fused_tokens, dim=-1)),
        fused.stable_addresses() == addresses,
    )


def _profile(operation: Callable[[], object]) -> Profile:
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as result:
        output = operation()
        torch.cuda.synchronize()
    events = [
        event for event in result.events() if event.device_type == DeviceType.CUDA
    ]
    names = Counter(event.name for event in events)
    device_ms = sum(event.device_time_total for event in events) / 1000.0
    del output
    return Profile(len(events), device_ms, tuple(names.most_common()))


def _post_profiles(position: int) -> tuple[Profile, Profile]:
    packed, cos, sin, query, key, value, current_cache, fused_cache = (
        _packed_inputs(position, position + 1)
    )

    def current() -> torch.Tensor:
        current_cache.cumulative_length.fill_(position)
        query_output, key_output = rope_native(query, key, cos, sin)
        current_cache.update(key_output, value)
        return query_output

    def fused() -> torch.Tensor:
        fused_cache.cumulative_length.fill_(position)
        return packed_qkv_rope_cache_native(
            packed,
            cos,
            sin,
            fused_cache.keys,
            fused_cache.values,
            fused_cache.cumulative_length,
        )

    return _profile(current), _profile(fused)


def _model_profile(model: torch.nn.Module, context: int) -> Profile:
    cache = _make_static_cache(model, context, context + 1)
    token = _ids(context + 1, model.config.vocab_size)[:, -1:]
    position_ids = torch.tensor([[context]], device="cuda")
    attention_mask = _mask(context + 1, context + 1)
    return _profile(
        lambda: model(
            input_ids=token,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
    )


def _graph_profile(model: torch.nn.Module, context: int) -> Profile:
    prompt = _ids(context, model.config.vocab_size)
    state = FluxCUDAGraphDecode.capture(model, prompt, max_decode_steps=16)
    return _profile(state.graph.replay)


def _eager_peak(model: torch.nn.Module, context: int) -> int:
    cache = _make_static_cache(model, context, context + 1)
    token = _ids(context + 1, model.config.vocab_size)[:, -1:]
    position_ids = torch.tensor([[context]], device="cuda")
    attention_mask = _mask(context + 1, context + 1)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = model(
        input_ids=token,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    del output
    return peak


def _graph_pool_worker(connection: Any, context: int, fused: bool) -> None:
    try:
        _configure()
        model = load_model("cuda")
        enable_flux_ops(model, operators=FUSED_OPERATORS if fused else BASE_OPERATORS)
        prompt = _ids(context, model.config.vocab_size)
        state = FluxCUDAGraphDecode.capture(
            model, prompt, max_decode_steps=16
        )
        torch.cuda.synchronize()
        connection.send(("ok", state.memory.graph_pool_bytes))
    except BaseException as error:
        connection.send(("error", repr(error)))
    finally:
        connection.close()


def _independent_graph_pool(context: int) -> tuple[int, int]:
    process_context = multiprocessing.get_context("spawn")
    values = []
    for fused in (False, True):
        receive, send = process_context.Pipe(duplex=False)
        process = process_context.Process(
            target=_graph_pool_worker, args=(send, context, fused)
        )
        process.start()
        send.close()
        status, payload = receive.recv()
        process.join()
        if process.exitcode != 0 or status != "ok":
            raise RuntimeError(
                f"independent graph pool probe failed: exit={process.exitcode}, "
                f"status={status}, payload={payload}"
            )
        values.append(int(payload))
    return values[0], values[1]


def _print_pairs(title: str, rows: dict[int, Pair], unit: str = "ms") -> None:
    multiplier = 1000.0 if unit == "us" else 1.0
    print(f"\n{title}")
    print(
        f"{'position':>9} {'current ' + unit:>13} {'fused ' + unit:>13} "
        f"{'speedup':>10}"
    )
    for position, timing in rows.items():
        print(
            f"{position:>9} {timing.current_ms * multiplier:>13.3f} "
            f"{timing.fused_ms * multiplier:>13.3f} "
            f"{timing.speedup:>9.3f}x"
        )


def main() -> int:
    args = _args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    _configure()
    print("Fused packed-QKV RoPE + StaticCache milestone")
    print(f"  model={MODEL_ID}@{MODEL_REVISION}")
    print(
        f"  Python={sys.version.split()[0]}; PyTorch={torch.__version__}; "
        f"CUDA={torch.version.cuda}"
    )
    print(
        f"  GPU={torch.cuda.get_device_name()}; "
        f"capability={torch.cuda.get_device_capability()}"
    )
    print(
        "  FP32; TF32 disabled; deterministic algorithms; CUDA-event medians; "
        "alternating samples"
    )

    isolated: dict[int, tuple[Pair, Error, Error, Error]] = {}
    with torch.inference_mode():
        for context in args.contexts:
            print(f"Isolated post-QKV position {context}...", flush=True)
            isolated[context] = _isolated_post_qkv(
                context, args.warmup, args.repetitions
            )
        print("Validating near upper cache bound at position 8190...", flush=True)
        upper = _isolated_post_qkv(8190, 1, 1)

    print("Loading retained and fused models...", flush=True)
    source = load_model("cuda")
    current_model = copy.deepcopy(source)
    fused_model = copy.deepcopy(source)
    del source
    enable_flux_ops(current_model, operators=BASE_OPERATORS)
    enable_flux_ops(fused_model, operators=FUSED_OPERATORS)

    if args.stabilization_iterations:
        print("Stabilizing GPU clocks...", flush=True)
        ids = _ids(min(1024, max(args.contexts)), fused_model.config.vocab_size)
        with torch.inference_mode():
            for _ in range(args.stabilization_iterations):
                output = fused_model(
                    input_ids=ids, use_cache=True, logits_to_keep=1
                )
                del output
        torch.cuda.synchronize()
        del ids

    layer: dict[int, tuple[Pair, Error]] = {}
    eager: dict[int, tuple[Pair, Error, Error, Error]] = {}
    graph: dict[int, Pair] = {}
    graph_correctness: dict[int, tuple[Error, Error, Error, bool, bool]] = {}
    with torch.inference_mode():
        for context in args.contexts:
            print(f"Layer/eager/graph position {context}...", flush=True)
            layer[context] = _layer_pair(
                current_model,
                fused_model,
                context,
                args.warmup,
                args.model_repetitions,
            )
            eager[context] = _eager_static_pair(
                current_model,
                fused_model,
                context,
                args.warmup,
                args.model_repetitions,
            )
            graph[context] = _graph_pair(
                current_model,
                fused_model,
                context,
                args.warmup,
                args.model_repetitions,
            )
            graph_correctness[context] = _graph_correctness_and_generation(
                current_model,
                fused_model,
                context,
                args.generation_tokens,
            )

        profile_context = max(args.contexts)
        print(f"Profiling at position {profile_context}...", flush=True)
        current_post_profile, fused_post_profile = _post_profiles(profile_context)
        current_model_profile = _model_profile(current_model, profile_context)
        fused_model_profile = _model_profile(fused_model, profile_context)
        current_graph_profile = _graph_profile(current_model, profile_context)
        fused_graph_profile = _graph_profile(fused_model, profile_context)
        current_peak = _eager_peak(current_model, profile_context)
        fused_peak = _eager_peak(fused_model, profile_context)

    print("Measuring independent CUDA Graph pools...", flush=True)
    current_pool, fused_pool = _independent_graph_pool(max(args.contexts))

    _print_pairs(
        "Isolated packed QKV -> RoPE -> StaticCache",
        {position: row[0] for position, row in isolated.items()},
        unit="us",
    )
    _print_pairs(
        "Complete decoder layer",
        {position: row[0] for position, row in layer.items()},
    )
    _print_pairs(
        "Full eager StaticCache decode",
        {position: row[0] for position, row in eager.items()},
    )
    _print_pairs("Fixed-shape CUDA Graph replay", graph)

    print("\nNumerical differences (max / mean absolute error)")
    print(
        f"{'position':>9} {'Q RoPE':>21} {'new K':>21} {'new V':>21} "
        f"{'layer':>21} {'logits':>21}"
    )
    for context in args.contexts:
        post = isolated[context]
        print(
            f"{context:>9} "
            f"{post[1].maximum:>9.3g}/{post[1].mean:<9.3g} "
            f"{post[2].maximum:>9.3g}/{post[2].mean:<9.3g} "
            f"{post[3].maximum:>9.3g}/{post[3].mean:<9.3g} "
            f"{layer[context][1].maximum:>9.3g}/{layer[context][1].mean:<9.3g} "
            f"{eager[context][1].maximum:>9.3g}/{eager[context][1].mean:<9.3g}"
        )
    print(
        "  upper-bound position 8190 Q/K/V max/mean: "
        + ", ".join(
            f"{item.maximum:.3g}/{item.mean:.3g}" for item in upper[1:]
        )
    )

    print("\nFull-cache and repeated-replay correctness")
    print(
        f"{'position':>9} {'eager K max/mean':>22} {'eager V max/mean':>22} "
        f"{'graph logits':>22} {'graph K':>22} {'graph V':>22} "
        f"{'tokens':>8} {'address':>9}"
    )
    for context in args.contexts:
        graph_row = graph_correctness[context]
        print(
            f"{context:>9} "
            f"{eager[context][2].maximum:>9.3g}/{eager[context][2].mean:<9.3g} "
            f"{eager[context][3].maximum:>9.3g}/{eager[context][3].mean:<9.3g} "
            f"{graph_row[0].maximum:>9.3g}/{graph_row[0].mean:<9.3g} "
            f"{graph_row[1].maximum:>9.3g}/{graph_row[1].mean:<9.3g} "
            f"{graph_row[2].maximum:>9.3g}/{graph_row[2].mean:<9.3g} "
            f"{str(graph_row[3]):>8} {str(graph_row[4]):>9}"
        )

    print(f"\nProfiles at position {profile_context}")
    for label, item in (
        ("current post-QKV", current_post_profile),
        ("fused post-QKV", fused_post_profile),
        ("current full eager", current_model_profile),
        ("fused full eager", fused_model_profile),
        ("current graph replay", current_graph_profile),
        ("fused graph replay", fused_graph_profile),
    ):
        print(
            f"  {label}: launches={item.launches}, "
            f"summed device={item.device_ms:.4f} ms"
        )
        print(f"    kernels={item.names}")

    q_bytes = 9 * 64 * 4
    k_bytes = 3 * 64 * 4
    packed_bytes = 960 * 4
    print("\nAllocation, lifetime, traffic, and graph pool")
    print(
        "  visible post-QKV temporaries: current=3 "
        f"({(q_bytes + k_bytes + 8) / 1024:.3f} KiB: Q, K, cache index), "
        f"fused=1 ({q_bytes / 1024:.3f} KiB: compact Q)"
    )
    print(
        f"  removed temporary K allocation: {k_bytes / 1024:.3f} KiB/layer; "
        "V remains a view in the retained path"
    )
    print(
        "  cache-write traffic: both write 0.750 KiB K + 0.750 KiB V per layer; "
        "fused removes the temporary rotated-K write/read"
    )
    print(
        f"  packed output: {packed_bytes / 1024:.3f} KiB/layer; fused releases it "
        "immediately after the post-QKV call instead of retaining its V view through cache update"
    )
    print(
        f"  eager incremental CUDA peak: current={current_peak / MIB:.3f} MiB, "
        f"fused={fused_peak / MIB:.3f} MiB"
    )
    print(
        f"  independent graph pool: current={current_pool / MIB:.3f} MiB, "
        f"fused={fused_pool / MIB:.3f} MiB"
    )
    print(
        "  activation rule: StaticCache capacity >=1281 and <=8192; smaller graph "
        "capacities retain the measured native-GQA crossover fallback"
    )
    print(
        "  kernel: one 256-thread block handles all 960 Q|K|V elements, uses exact "
        "FP32 multiply/add rounding, writes compact Q plus cache K/V, then advances "
        "the device-resident cache length after a block barrier"
    )
    print(
        "  rejected experiment: aligned float4 loads/stores reduced the profiled "
        "kernel subtotal but was neutral or slower across representative isolated "
        "positions, so the production kernel retains scalar access"
    )

    del current_model, fused_model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
