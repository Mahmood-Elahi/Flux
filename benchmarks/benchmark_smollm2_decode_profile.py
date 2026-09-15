"""Profile the canonical native SmolLM2 prefill and decode runtimes.

This production-level profiler attributes complete native runtime replays by
CUDA owner and kernel family. Isolated operator timing belongs to
``flux_cuda_microbenchmarks``.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from benchmarks.smollm2_benchmark_utils import (
    configure_runtime,
    deterministic_input_ids,
    parse_positive_int_list,
)
from flux.model import FINAL_FLUX_OPERATOR_CATEGORIES, enable_flux_ops
from flux.model.smollm2 import MODEL_ID, MODEL_REVISION, load_model
from flux.runtime import NativeSmolLM2Prefill


@dataclass(frozen=True)
class ProfileRow:
    workload: str
    size: int
    repetitions: int
    median_ms: float
    profiled_ms: float
    launches: int
    allocation_growth_bytes: int
    stable_addresses: bool
    owner_counts: dict[str, int]
    owner_ms: dict[str, float]
    top_kernels_ms: tuple[tuple[str, int, float], ...]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decode-contexts",
        type=parse_positive_int_list,
        default=(128, 1024, 4096, 8192),
    )
    parser.add_argument(
        "--prefill-lengths",
        type=parse_positive_int_list,
        default=(128, 1024, 4096, 8192),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--skip-prefill", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup may be zero, not negative")
    if min(args.repetitions, args.samples, args.top_k) < 1:
        parser.error("repetitions, samples, and top-k must be positive")
    if args.skip_prefill and args.skip_decode:
        parser.error("cannot skip both workloads")
    return args


def _owner(name: str) -> str:
    lowered = name.lower()
    if "memcpy" in lowered or "memset" in lowered:
        return "CUDA state/copy"
    if "gqa_decode" in lowered or "streaming_prefill" in lowered:
        return "Flux attention"
    if "_zn4flux" in lowered or "flux" in lowered:
        return "Flux other"
    if "cublas" in lowered or "gemm" in lowered or "gemv" in lowered:
        return "cuBLAS/cuBLASLt"
    return "framework CUDA"


def _short_name(name: str) -> str:
    for marker, short in (
        ("streaming_prefill", "streaming prefill GQA"),
        ("gqa_decode_attention_grouped_chunk", "decode GQA grouped chunk"),
        ("gqa_decode_attention_reduce", "decode GQA reduce"),
        ("packed_gate_up_swiglu", "fused gate/up + SwiGLU"),
        ("packed_qkv_rope_cache", "packed QKV/RoPE/cache"),
        ("residual_rmsnorm", "residual-RMSNorm"),
        ("rmsnorm", "RMSNorm"),
        ("prepare_full_decode", "decode state prepare"),
        ("advance_full_decode", "decode state advance"),
        ("gemv", "GEMV"),
        ("gemm", "GEMM"),
        ("memcpy", "CUDA memcpy"),
        ("memset", "CUDA memset"),
    ):
        if marker in name.lower():
            return short
    return name if len(name) <= 100 else name[:97] + "..."


def _event_ms(operation: Callable[[], object]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = operation()
    end.record()
    end.synchronize()
    del output
    return start.elapsed_time(end)


def _profile_runtime(
    workload: str,
    size: int,
    runtime: NativeSmolLM2Prefill,
    operation: Callable[[], object],
    warmup: int,
    samples: int,
    repetitions: int,
    top_k: int,
) -> ProfileRow:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    medians = [_event_ms(operation) for _ in range(samples)]
    addresses = runtime.stable_addresses()
    allocated = torch.cuda.memory_allocated()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        start.record()
        for _ in range(repetitions):
            operation()
        end.record()
    end.synchronize()
    growth = torch.cuda.memory_allocated() - allocated

    owner_counts: Counter[str] = Counter()
    owner_times: dict[str, float] = defaultdict(float)
    kernel_counts: Counter[str] = Counter()
    kernel_times: dict[str, float] = defaultdict(float)
    for event in profiler.events():
        if event.device_type != DeviceType.CUDA:
            continue
        milliseconds = float(event.self_device_time_total) / 1000.0 / repetitions
        owner = _owner(event.name)
        short = _short_name(event.name)
        owner_counts[owner] += 1
        owner_times[owner] += milliseconds
        kernel_counts[short] += 1
        kernel_times[short] += milliseconds
    owners = sorted(set(owner_counts) | set(owner_times))
    top = sorted(kernel_times, key=kernel_times.get, reverse=True)[:top_k]
    return ProfileRow(
        workload=workload,
        size=size,
        repetitions=repetitions,
        median_ms=statistics.median(medians),
        profiled_ms=start.elapsed_time(end) / repetitions,
        launches=round(sum(owner_counts.values()) / repetitions),
        allocation_growth_bytes=growth,
        stable_addresses=addresses == runtime.stable_addresses(),
        owner_counts={name: round(owner_counts[name] / repetitions) for name in owners},
        owner_ms={name: owner_times[name] for name in owners},
        top_kernels_ms=tuple(
            (name, round(kernel_counts[name] / repetitions), kernel_times[name])
            for name in top
        ),
    )


def _decode_profile(
    model: torch.nn.Module, context: int, args: argparse.Namespace
) -> ProfileRow:
    total = args.warmup + args.samples + args.repetitions
    runtime = NativeSmolLM2Prefill.capture(
        model,
        deterministic_input_ids(context - total, model.config.vocab_size),
        max_decode_steps=total,
    )
    return _profile_runtime(
        "native decode", context, runtime, runtime.replay,
        args.warmup, args.samples, args.repetitions, args.top_k,
    )


def _prefill_profile(
    model: torch.nn.Module, length: int, args: argparse.Namespace
) -> ProfileRow:
    ids = deterministic_input_ids(length, model.config.vocab_size)
    runtime = NativeSmolLM2Prefill.capture(model, ids)
    return _profile_runtime(
        "native prefill", length, runtime, lambda: runtime.prefill(ids),
        args.warmup, args.samples, args.repetitions, args.top_k,
    )


def _print(rows: list[ProfileRow]) -> None:
    for row in rows:
        print(
            f"\n{row.workload}, size={row.size}: median={row.median_ms:.4f} ms, "
            f"profiled={row.profiled_ms:.4f} ms, launches={row.launches}, "
            f"allocation_growth={row.allocation_growth_bytes}, "
            f"stable_addresses={row.stable_addresses}"
        )
        print("  Owners")
        for owner, milliseconds in sorted(
            row.owner_ms.items(), key=lambda item: item[1], reverse=True
        ):
            print(
                f"    {owner:<20} {row.owner_counts[owner]:>4} launches "
                f"{milliseconds:>10.4f} ms"
            )
        print("  Top kernels")
        for name, launches, milliseconds in row.top_kernels_ms:
            print(f"    {name:<60} {launches:>4} {milliseconds:>10.4f} ms")


def main() -> int:
    args = _args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    configure_runtime(seed=0)
    model = enable_flux_ops(
        load_model("cuda"), operators=FINAL_FLUX_OPERATOR_CATEGORIES
    )
    maximum = int(model.config.max_position_embeddings)
    if any(length > maximum for length in args.prefill_lengths):
        raise ValueError(f"prefill length exceeds model maximum {maximum}")
    decode_steps = args.warmup + args.samples + args.repetitions
    if any(context > maximum or context <= decode_steps for context in args.decode_contexts):
        raise ValueError(
            "decode contexts must exceed all profiling steps and fit the model maximum"
        )
    print(
        f"Model: {MODEL_ID} @ {MODEL_REVISION}\n"
        f"GPU: {torch.cuda.get_device_name()}\n"
        f"PyTorch/CUDA: {torch.__version__} / {torch.version.cuda}\n"
        "Backend: native prefill and attached native decode"
    )
    rows: list[ProfileRow] = []
    if not args.skip_decode:
        for context in args.decode_contexts:
            rows.append(_decode_profile(model, context, args))
    if not args.skip_prefill:
        for length in args.prefill_lengths:
            rows.append(_prefill_profile(model, length, args))
    _print(rows)
    if args.json_output is not None:
        payload = {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "rows": [asdict(row) for row in rows],
        }
        args.json_output.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nWrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
