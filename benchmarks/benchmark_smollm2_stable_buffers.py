"""Profile and benchmark fixed-shape one-token stable decode outputs.

This compares the retained fully fused Flux CUDA Graph with and without the
explicit scratch set. It also performs the diagnostic-only deterministic
uninitialized-memory-fill toggle requested by this milestone.
"""

from __future__ import annotations

import argparse
import copy
import gc
import multiprocessing
import os
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile
from transformers import DynamicCache

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
from flux.ops import (
    gqa_decode_attention_native,
    packed_qkv_rope_cache_native,
    packed_swiglu_native,
    residual_rmsnorm_native,
    rms_norm_native,
)


CAPACITIES = (128, 512, 1024, 2048, 4096)
FULL_OPERATORS = FLUX_OPERATOR_CATEGORIES | {
    FLUX_PACKED_MLP_CATEGORY,
    FLUX_PACKED_QKV_CATEGORY,
    FLUX_PACKED_SWIGLU_CATEGORY,
    FLUX_GQA_DECODE_ATTENTION_CATEGORY,
    FLUX_PACKED_QKV_ROPE_CACHE_CATEGORY,
}
MIB = 1024.0**2


@dataclass(frozen=True)
class Pair:
    baseline: float
    stable: float

    @property
    def speedup(self) -> float:
        return self.baseline / self.stable


@dataclass(frozen=True)
class ProfileSummary:
    launches: int
    device_ms: float
    categories: tuple[tuple[str, int, float], ...]
    kernels: tuple[tuple[str, int, float], ...]

    def category(self, name: str) -> tuple[int, float]:
        return next(
            ((count, milliseconds) for item, count, milliseconds in self.categories if item == name),
            (0, 0.0),
        )


def _parse_int_list(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item < 2 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated integers >= 2")
    return result


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacities", type=_parse_int_list, default=CAPACITIES)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--correctness-tokens", type=int, default=8)
    parser.add_argument("--profile-capacity", type=int, default=4096)
    parser.add_argument("--skip-independent-pools", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.repetitions < 1 or args.correctness_tokens < 1:
        parser.error("warmup may be zero; other counts must be positive")
    if min(args.capacities) <= args.warmup + args.repetitions:
        parser.error("each capacity must exceed warmup + repetitions")
    if args.profile_capacity not in args.capacities:
        parser.error("--profile-capacity must be present in --capacities")
    return args


def _configure() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _ids(length: int, vocab_size: int) -> torch.Tensor:
    return (((torch.arange(length) * 17 + 11) % vocab_size).unsqueeze(0)).cuda()


def _model() -> torch.nn.Module:
    return enable_flux_ops(load_model("cuda"), operators=FULL_OPERATORS)


def _graph(
    model: torch.nn.Module,
    capacity: int,
    steps: int,
    stable: bool,
) -> FluxCUDAGraphDecode:
    prompt = _ids(capacity - steps, model.config.vocab_size)
    result = FluxCUDAGraphDecode.capture(
        model,
        prompt,
        max_decode_steps=steps,
        use_stable_buffers=stable,
    )
    assert result.max_cache_len == capacity
    return result


def _paired_graph(
    model: torch.nn.Module, capacity: int, warmup: int, repetitions: int
) -> Pair:
    steps = warmup + repetitions
    baseline = _graph(model, capacity, steps, False)
    stable = _graph(model, capacity, steps, True)
    if stable.scratch is None:
        for _ in range(warmup):
            baseline.graph.replay()
        samples = []
        for _ in range(repetitions):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            baseline.graph.replay()
            end.record()
            samples.append((start, end))
        samples[-1][1].synchronize()
        milliseconds = statistics.median(
            start.elapsed_time(end) for start, end in samples
        )
        del baseline, stable
        return Pair(milliseconds, milliseconds)
    operations = (baseline.graph.replay, stable.graph.replay)
    for _ in range(warmup):
        for operation in operations:
            operation()
    samples: tuple[list[tuple[torch.cuda.Event, torch.cuda.Event]], ...] = ([], [])
    for repetition in range(repetitions):
        for index in ((0, 1) if repetition % 2 == 0 else (1, 0)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            operations[index]()
            end.record()
            samples[index].append((start, end))
    torch.cuda.synchronize()
    medians = tuple(
        statistics.median(start.elapsed_time(end) for start, end in group)
        for group in samples
    )
    del baseline, stable
    return Pair(*medians)


def _paired_fill_toggle(
    model: torch.nn.Module, capacity: int, warmup: int, repetitions: int
) -> Pair:
    previous = torch.utils.deterministic.fill_uninitialized_memory
    steps = warmup + repetitions
    try:
        torch.utils.deterministic.fill_uninitialized_memory = True
        enabled = _graph(model, capacity, steps, False)
        torch.utils.deterministic.fill_uninitialized_memory = False
        disabled = _graph(model, capacity, steps, False)
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = previous
    operations = (enabled.graph.replay, disabled.graph.replay)
    for _ in range(warmup):
        for operation in operations:
            operation()
    torch.testing.assert_close(enabled.logits, disabled.logits, rtol=0, atol=0)
    samples: tuple[list[tuple[torch.cuda.Event, torch.cuda.Event]], ...] = ([], [])
    for repetition in range(repetitions):
        for index in ((0, 1) if repetition % 2 == 0 else (1, 0)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            operations[index]()
            end.record()
            samples[index].append((start, end))
    torch.cuda.synchronize()
    medians = tuple(
        statistics.median(start.elapsed_time(end) for start, end in group)
        for group in samples
    )
    del enabled, disabled
    return Pair(*medians)


def _clone_cache(cache: Any, config: Any) -> DynamicCache:
    return DynamicCache(
        [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers],
        config=config,
    )


def _eager_latency(
    model: torch.nn.Module, capacity: int, warmup: int, repetitions: int
) -> float:
    prompt = _ids(capacity - 1, model.config.vocab_size)
    token = _ids(capacity, model.config.vocab_size)[:, -1:]
    base = model(input_ids=prompt, use_cache=True, logits_to_keep=1).past_key_values
    samples = []
    output: object = None
    for index in range(warmup + repetitions):
        cache = _clone_cache(base, model.config)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = model(
            input_ids=token,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        end.record()
        if index >= warmup:
            samples.append((start, end))
    samples[-1][1].synchronize()
    result = statistics.median(start.elapsed_time(end) for start, end in samples)
    del output, base, prompt, token
    return result


def _kernel_category(name: str) -> str:
    lowered = name.lower()
    if "fillfunctor" in lowered or "fill_kernel" in lowered:
        return "deterministic fill"
    if "_zn4flux" in lowered:
        return "custom Flux"
    if any(marker in lowered for marker in ("cublas", "gemv", "gemmk", "sgemm")):
        return "cuBLAS"
    if any(marker in lowered for marker in ("copy", "indexselect")):
        return "tensor copy"
    if any(marker in lowered for marker in ("index_put", "indexing", "onself_add")):
        return "cache/state management"
    if any(marker in lowered for marker in ("elementwise", "cos_kernel", "sin_kernel")):
        return "PyTorch pointwise"
    if any(marker in lowered for marker in ("cast", "transpose", "permute", "contiguous")):
        return "dtype/layout"
    if any(marker in lowered for marker in ("memset", "memcpy")):
        return "memset/memcpy"
    return "other framework"


def _profile(operation: Callable[[], object]) -> ProfileSummary:
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as result:
        output = operation()
        torch.cuda.synchronize()
    categories: dict[str, list[float]] = defaultdict(list)
    kernels: dict[str, list[float]] = defaultdict(list)
    for event in result.events():
        if event.device_type != DeviceType.CUDA:
            continue
        milliseconds = float(event.self_device_time_total) / 1000.0
        categories[_kernel_category(event.name)].append(milliseconds)
        kernels[event.name].append(milliseconds)
    summary = ProfileSummary(
        sum(len(values) for values in kernels.values()),
        sum(sum(values) for values in kernels.values()),
        tuple(sorted(
            ((name, len(values), sum(values)) for name, values in categories.items()),
            key=lambda item: item[1], reverse=True,
        )),
        tuple(sorted(
            ((name, len(values), sum(values)) for name, values in kernels.items()),
            key=lambda item: item[2], reverse=True,
        )),
    )
    del output
    return summary


def _profile_graph(
    model: torch.nn.Module, capacity: int, stable: bool, fill: bool
) -> ProfileSummary:
    previous = torch.utils.deterministic.fill_uninitialized_memory
    torch.utils.deterministic.fill_uninitialized_memory = fill
    try:
        state = _graph(model, capacity, 16, stable)
        for _ in range(5):
            state.graph.replay()
        batch = 5
        result = _profile(
            lambda: tuple(state.graph.replay() for _ in range(batch))
        )
        result = ProfileSummary(
            result.launches // batch,
            result.device_ms / batch,
            tuple(
                (name, count // batch, milliseconds / batch)
                for name, count, milliseconds in result.categories
            ),
            tuple(
                (name, count // batch, milliseconds / batch)
                for name, count, milliseconds in result.kernels
            ),
        )
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = previous
    del state
    return result


def _profile_eager(model: torch.nn.Module, capacity: int, fill: bool) -> ProfileSummary:
    previous = torch.utils.deterministic.fill_uninitialized_memory
    torch.utils.deterministic.fill_uninitialized_memory = fill
    try:
        prompt = _ids(capacity - 1, model.config.vocab_size)
        token = _ids(capacity, model.config.vocab_size)[:, -1:]
        cache = model(input_ids=prompt, use_cache=True, logits_to_keep=1).past_key_values
        result = _profile(
            lambda: model(
                input_ids=token,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        )
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = previous
    del prompt, token, cache
    return result


def _attribution(capacity: int) -> tuple[tuple[str, int, int, int, float], ...]:
    generator = torch.Generator(device="cuda").manual_seed(4400)
    hidden = torch.randn((1, 1, 576), generator=generator, device="cuda")
    residual = torch.randn_like(hidden)
    weight = torch.randn((576,), generator=generator, device="cuda")
    packed_mlp = torch.randn((1, 1, 3072), generator=generator, device="cuda")
    packed_qkv = torch.randn((1, 1, 960), generator=generator, device="cuda")
    frequencies = torch.randn((1, 1, 64), generator=generator, device="cuda")
    cos, sin = frequencies.cos(), frequencies.sin()
    keys = torch.randn((1, 3, capacity, 64), generator=generator, device="cuda")
    values = torch.randn_like(keys)
    length = torch.tensor(capacity - 1, device="cuda")
    query = torch.randn((1, 9, 1, 64), generator=generator, device="cuda")
    mask = torch.zeros((1, 1, 1, capacity), device="cuda")
    operations: tuple[tuple[str, int, int, Callable[[], object]], ...] = (
        ("GQA output + workspace", 30, (9 * 64 + 9 * ((capacity + 255) // 256) * 66) * 4,
         lambda: gqa_decode_attention_native(query, keys, values, mask, 0.125, length)),
        ("residual RMSNorm outputs", 30, 2 * 576 * 4,
         lambda: residual_rmsnorm_native(hidden, residual, weight, 1e-5)),
        ("RMSNorm output", 31, 576 * 4,
         lambda: rms_norm_native(hidden, weight, 1e-5)),
        ("packed SwiGLU output", 30, 1536 * 4,
         lambda: packed_swiglu_native(packed_mlp)),
        ("packed-QKV compact Q", 30, 9 * 64 * 4,
         lambda: packed_qkv_rope_cache_native(
             packed_qkv, cos, sin, keys, values, length)),
    )
    rows = []
    for name, frequency, bytes_per_call, operation in operations:
        item = _profile(operation)
        fill_count, fill_ms = item.category("deterministic fill")
        rows.append((name, frequency, fill_count * frequency, bytes_per_call * frequency, fill_ms * frequency))
    return tuple(rows)


def _correctness(model: torch.nn.Module, capacity: int, tokens: int) -> tuple[float, bool, bool, bool]:
    baseline = _graph(model, capacity, tokens, False)
    stable = _graph(model, capacity, tokens, True)
    maximum = 0.0
    token = baseline.prefill_logits.argmax(dim=-1)
    baseline_tokens = []
    stable_tokens = []
    addresses = stable.stable_addresses()
    for step in range(tokens):
        if stable.scratch is not None:
            for tensor in (
                stable.scratch.norm_output,
                stable.scratch.residual_output,
                stable.scratch.query_output,
                stable.scratch.attention_output,
                stable.scratch.attention_workspace,
                stable.scratch.swiglu_output,
            ):
                tensor.fill_(13.0 + step)
        expected = baseline.replay(token)
        actual = stable.replay(token)
        maximum = max(maximum, float((actual - expected).abs().max().item()))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        baseline_token = expected.argmax(dim=-1)
        stable_token = actual.argmax(dim=-1)
        baseline_tokens.append(baseline_token.clone())
        stable_tokens.append(stable_token.clone())
        token = baseline_token
    caches_equal = all(
        torch.equal(left.keys, right.keys) and torch.equal(left.values, right.values)
        and torch.equal(left.cumulative_length, right.cumulative_length)
        for left, right in zip(baseline.cache.layers, stable.cache.layers, strict=True)
    )
    return (
        maximum,
        torch.equal(torch.cat(baseline_tokens, -1), torch.cat(stable_tokens, -1)),
        caches_equal,
        stable.stable_addresses() == addresses,
    )


def _eager_peak(model: torch.nn.Module, capacity: int) -> int:
    prompt = _ids(capacity - 1, model.config.vocab_size)
    token = _ids(capacity, model.config.vocab_size)[:, -1:]
    cache = model(input_ids=prompt, use_cache=True, logits_to_keep=1).past_key_values
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = model(input_ids=token, past_key_values=cache, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    result = torch.cuda.max_memory_allocated() - baseline
    del prompt, token, cache, output
    return result


def _pool_worker(stable: bool, capacity: int, queue: Any) -> None:
    _configure()
    model = _model()
    state = _graph(model, capacity, 2, stable)
    queue.put((state.memory.graph_pool_bytes, state.memory.stable_scratch_bytes))


def _independent_pool(stable: bool, capacity: int) -> tuple[int, int]:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_pool_worker, args=(stable, capacity, queue))
    process.start()
    result = queue.get()
    process.join()
    if process.exitcode:
        raise RuntimeError(f"graph-pool child exited with {process.exitcode}")
    return result


def _print_profile(label: str, item: ProfileSummary) -> None:
    print(f"  {label}: launches={item.launches}, device={item.device_ms:.4f} ms")
    for name, count, milliseconds in item.categories:
        print(f"    {name:<24} count={count:>4} device={milliseconds:.4f} ms")


def main() -> int:
    args = _args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    _configure()
    print("Flux stable decode-output milestone")
    print(f"  model={MODEL_ID}@{MODEL_REVISION}")
    print(f"  Python={sys.version.split()[0]}; PyTorch={torch.__version__}; CUDA={torch.version.cuda}")
    print(f"  GPU={torch.cuda.get_device_name()}; capability={torch.cuda.get_device_capability()}")
    print("  FP32 B=1 Q=1; TF32 disabled; deterministic algorithms and safety fills enabled")
    model = _model()
    print("Stabilizing GPU clocks...", flush=True)
    stabilization = _graph(model, max(args.capacities), 20, True)
    for _ in range(20):
        stabilization.graph.replay()
    torch.cuda.synchronize()
    del stabilization

    graph_timings = []
    eager_timings = []
    with torch.inference_mode():
        for capacity in args.capacities:
            print(f"Benchmarking capacity {capacity}...", flush=True)
            graph_timings.append((capacity, _paired_graph(model, capacity, args.warmup, args.repetitions)))
            eager_timings.append((capacity, _eager_latency(model, capacity, args.warmup, args.repetitions)))

        print("Profiling graph launch breakdowns...", flush=True)
        graph_profiles = {}
        for capacity in args.capacities:
            graph_profiles[capacity] = (
                _profile_graph(model, capacity, False, True),
                _profile_graph(model, capacity, True, True),
            )
        print(f"Profiling detailed capacity {args.profile_capacity} diagnostics...", flush=True)
        fill_latency = _paired_fill_toggle(
            model, args.profile_capacity, args.warmup, args.repetitions
        )
        baseline_profile, stable_profile = graph_profiles[args.profile_capacity]
        nofill_profile = _profile_graph(model, args.profile_capacity, False, False)
        eager_profile = _profile_eager(model, args.profile_capacity, True)
        eager_nofill = _profile_eager(model, args.profile_capacity, False)
        attribution = _attribution(args.profile_capacity)
        correctness = _correctness(model, args.profile_capacity, args.correctness_tokens)
        eager_peak = _eager_peak(model, args.profile_capacity)

    pools = None
    if not args.skip_independent_pools:
        print("Measuring graph pools in independent processes...", flush=True)
        pools = (
            _independent_pool(False, args.profile_capacity),
            _independent_pool(True, args.profile_capacity),
        )

    print("\nCUDA Graph pure replay")
    print(f"{'capacity':>9} {'baseline ms':>12} {'stable ms':>11} {'speedup':>9}")
    for capacity, item in graph_timings:
        print(f"{capacity:>9} {item.baseline:>12.4f} {item.stable:>11.4f} {item.speedup:>8.3f}x")
    print("\nCUDA Graph launch breakdown")
    print(
        f"{'capacity':>9} {'path':>9} {'total':>7} {'fills':>7} "
        f"{'Flux':>7} {'cuBLAS':>7}"
    )
    for capacity in args.capacities:
        for label, item in zip(("baseline", "stable"), graph_profiles[capacity], strict=True):
            print(
                f"{capacity:>9} {label:>9} {item.launches:>7} "
                f"{item.category('deterministic fill')[0]:>7} "
                f"{item.category('custom Flux')[0]:>7} "
                f"{item.category('cuBLAS')[0]:>7}"
            )
    print("\nEager one-token decode (stable buffers intentionally inactive)")
    print(f"{'context':>9} {'baseline ms':>12} {'stable ms':>11} {'speedup':>9}")
    for capacity, milliseconds in eager_timings:
        print(f"{capacity:>9} {milliseconds:>12.4f} {milliseconds:>11.4f} {'1.000x':>9}")

    print(f"\nProfiles at exact StaticCache capacity {args.profile_capacity}")
    _print_profile("baseline graph", baseline_profile)
    _print_profile("stable graph", stable_profile)
    _print_profile("eager", eager_profile)
    print("\nDiagnostic-only deterministic uninitialized-memory fill toggle")
    _print_profile("graph fill enabled", baseline_profile)
    _print_profile("graph fill disabled", nofill_profile)
    _print_profile("eager fill enabled", eager_profile)
    _print_profile("eager fill disabled", eager_nofill)
    print(
        f"  paired unprofiled graph replay median: enabled={fill_latency.baseline:.4f} ms, "
        f"disabled={fill_latency.stable:.4f} ms, impact={fill_latency.speedup:.3f}x"
    )

    print("\nRanked fully overwritten custom outputs")
    print(f"{'candidate':<30} {'calls':>6} {'fills':>7} {'MiB/token':>11} {'fill ms':>10}")
    for name, frequency, fills, bytes_per_token, fill_ms in sorted(
        attribution, key=lambda item: (item[2], item[4]), reverse=True
    ):
        print(f"{name:<30} {frequency:>6} {fills:>7} {bytes_per_token/MIB:>11.4f} {fill_ms:>10.4f}")
    print("  lifetimes: each output ends at its immediate consumer; GQA workspace ends at its reduction")
    print("  graph-static: yes for every row; all producers overwrite every element")

    selected_bytes = sum(row[3] for row in attribution)
    stable_bytes = 0
    if args.profile_capacity >= 1281:
        stable_bytes = (2 * 576 + 2 * 9 * 64 + 9 * ((args.profile_capacity + 255) // 256) * 66 + 1536) * 4
    print("\nMemory")
    print(f"  selected visible temporary allocations: baseline=211/token, stable=0/token")
    print(f"  selected temporary bytes: baseline={selected_bytes/MIB:.4f} MiB/token, stable=0")
    print(f"  stable scratch added: {stable_bytes} bytes ({stable_bytes/1024:.3f} KiB)")
    print(f"  eager incremental CUDA peak (same eager path): {eager_peak/MIB:.3f} MiB")
    if pools is not None:
        print(f"  independent graph pool: baseline={pools[0][0]/MIB:.3f} MiB, stable={pools[1][0]/MIB:.3f} MiB")
        print(f"  child-reported scratch: baseline={pools[0][1]} bytes, stable={pools[1][1]} bytes")

    print("\nCorrectness with scratch poisoned before every replay")
    print(
        f"  max logits={correctness[0]:.6g}; greedy IDs={correctness[1]}; "
        f"full cache/state={correctness[2]}; stable addresses={correctness[3]}"
    )
    print("\nEliminated kernels at 4096")
    baseline_fills = baseline_profile.category("deterministic fill")
    stable_fills = stable_profile.category("deterministic fill")
    print(
        f"  deterministic fills: {baseline_fills[0]} -> {stable_fills[0]} "
        f"(removed {baseline_fills[0]-stable_fills[0]}, "
        f"profiled device reduction {baseline_fills[1]-stable_fills[1]:.4f} ms)"
    )
    print(f"  total launches: {baseline_profile.launches} -> {stable_profile.launches}")
    print("  copies and other pointwise work: unchanged")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
