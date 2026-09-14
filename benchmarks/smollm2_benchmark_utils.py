"""Shared deterministic setup and timing helpers for SmolLM2 benchmarks.

These helpers deliberately preserve the established benchmark methodology:
model/cache preparation and correctness checks occur outside CUDA-event timing,
and paired implementations alternate order before reporting medians.
"""

from __future__ import annotations

import argparse
import os
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from transformers import DynamicCache


RTOL = 2e-4
ATOL = 2e-5
CORRECTNESS_CHUNK_TOKENS = 256


@dataclass(frozen=True)
class Comparison:
    size: int
    reference_ms: float
    flux_ms: float
    max_absolute_error: float

    @property
    def speedup(self) -> float:
        return self.reference_ms / self.flux_ms


def parse_positive_int_list(value: str) -> tuple[int, ...]:
    """Parse a non-empty comma-separated list of positive integers."""
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return result


def configure_runtime(*, seed: int = 0, deterministic_fill: bool | None = None) -> None:
    """Apply the deterministic FP32 CUDA settings used by model benchmarks."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if deterministic_fill is not None:
        torch.utils.deterministic.fill_uninitialized_memory = deterministic_fill
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def deterministic_input_ids(length: int, vocab_size: int) -> torch.Tensor:
    """Construct the repository's canonical deterministic CUDA token pattern."""
    values = (torch.arange(length, dtype=torch.long) * 17 + 11) % vocab_size
    return values.unsqueeze(0).to("cuda")


def event_median(
    operation: Callable[[], object],
    warmup: int,
    repetitions: int,
    prepare: Callable[[], Callable[[], object]] | None = None,
) -> float:
    """Return median CUDA-event milliseconds with optional untimed preparation."""
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


def alternating_event_medians(
    operations: dict[str, Callable[[], object]],
    warmup: int,
    repetitions: int,
    prepare: dict[str, Callable[[], Callable[[], object]]] | None = None,
) -> dict[str, float]:
    """Time named operations in alternating order and return CUDA-event medians."""
    outputs: dict[str, object] = {}
    names = tuple(operations)
    for warmup_index in range(warmup):
        order = names if warmup_index % 2 == 0 else tuple(reversed(names))
        for name in order:
            operation = prepare[name]() if prepare is not None else operations[name]
            outputs[name] = operation()
    torch.cuda.synchronize()

    events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
        name: [] for name in names
    }
    final_event = None
    for sample_index in range(repetitions):
        order = names if sample_index % 2 == 0 else tuple(reversed(names))
        for name in order:
            operation = prepare[name]() if prepare is not None else operations[name]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            outputs[name] = operation()
            end.record()
            events[name].append((start, end))
            final_event = end

    assert final_event is not None
    final_event.synchronize()
    medians = {
        name: statistics.median(start.elapsed_time(end) for start, end in samples)
        for name, samples in events.items()
    }
    del outputs
    return medians


def _cache_length(cache: DynamicCache) -> int:
    return int(cache.get_seq_length())


def _clone_cache(cache: DynamicCache, config: Any) -> DynamicCache:
    data = []
    for layer in cache.layers:
        if layer.keys is None or layer.values is None:
            data.append((None, None))
        else:
            data.append((layer.keys.detach().clone(), layer.values.detach().clone()))
    return DynamicCache(data, config=config)


def _assert_logits_close(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.shape != expected.shape:
        raise AssertionError(f"logit shape mismatch: {actual.shape} != {expected.shape}")
    max_absolute = 0.0
    maximum_excess = 0.0
    total_failures = 0
    for start in range(0, actual.shape[1], CORRECTNESS_CHUNK_TOKENS):
        stop = min(start + CORRECTNESS_CHUNK_TOKENS, actual.shape[1])
        actual_chunk = actual[:, start:stop]
        expected_chunk = expected[:, start:stop]
        difference = (actual_chunk - expected_chunk).abs()
        allowed = ATOL + RTOL * expected_chunk.abs()
        if not bool(torch.all(torch.isfinite(actual_chunk))):
            raise AssertionError("Flux logits contain non-finite values")
        excess = difference - allowed
        total_failures += int(torch.count_nonzero(excess > 0).item())
        maximum_excess = max(maximum_excess, float(excess.max().item()))
        max_absolute = max(max_absolute, float(difference.max().item()))
    torch.cuda.synchronize()
    if total_failures:
        raise AssertionError(
            f"logits differ beyond rtol={RTOL}, atol={ATOL}; "
            f"failures={total_failures}/{actual.numel()}, "
            f"max absolute error={max_absolute:.9g}, "
            f"max tolerance excess={maximum_excess:.9g}"
        )
    return max_absolute


def benchmark_full_or_prefill(
    reference: torch.nn.Module,
    flux: torch.nn.Module,
    input_ids: torch.Tensor,
    use_cache: bool,
    warmup: int,
    repetitions: int,
) -> Comparison:
    reference_output = reference(input_ids=input_ids, use_cache=use_cache)
    flux_output = flux(input_ids=input_ids, use_cache=use_cache)
    maximum = _assert_logits_close(flux_output.logits, reference_output.logits)
    if use_cache:
        reference_length = _cache_length(reference_output.past_key_values)
        flux_length = _cache_length(flux_output.past_key_values)
        if reference_length != flux_length or reference_length != input_ids.shape[1]:
            raise AssertionError(
                f"prefill cache lengths differ: reference={reference_length}, "
                f"Flux={flux_length}, expected={input_ids.shape[1]}"
            )
    del reference_output, flux_output
    medians = alternating_event_medians(
        {
            "reference": lambda: reference(input_ids=input_ids, use_cache=use_cache),
            "Flux": lambda: flux(input_ids=input_ids, use_cache=use_cache),
        },
        warmup,
        repetitions,
    )
    return Comparison(input_ids.shape[1], medians["reference"], medians["Flux"], maximum)


def benchmark_decode(
    reference: torch.nn.Module,
    flux: torch.nn.Module,
    context_ids: torch.Tensor,
    next_token: torch.Tensor,
    warmup: int,
    repetitions: int,
) -> Comparison:
    reference_context = reference(input_ids=context_ids, use_cache=True)
    flux_context = flux(input_ids=context_ids, use_cache=True)
    reference_cache = reference_context.past_key_values
    flux_cache = flux_context.past_key_values
    del reference_context, flux_context
    context_length = context_ids.shape[1]
    if _cache_length(reference_cache) != context_length:
        raise AssertionError("reference prefill cache has an unexpected length")
    if _cache_length(flux_cache) != context_length:
        raise AssertionError("Flux prefill cache has an unexpected length")

    reference_check_cache = _clone_cache(reference_cache, reference.config)
    flux_check_cache = _clone_cache(flux_cache, flux.config)
    reference_output = reference(
        input_ids=next_token,
        past_key_values=reference_check_cache,
        use_cache=True,
    )
    flux_output = flux(
        input_ids=next_token,
        past_key_values=flux_check_cache,
        use_cache=True,
    )
    maximum = _assert_logits_close(flux_output.logits, reference_output.logits)
    expected_length = context_length + 1
    lengths = (
        _cache_length(reference_output.past_key_values),
        _cache_length(flux_output.past_key_values),
    )
    if lengths != (expected_length, expected_length):
        raise AssertionError(
            f"decode cache lengths differ: reference={lengths[0]}, "
            f"Flux={lengths[1]}, expected={expected_length}"
        )
    del reference_output, flux_output, reference_check_cache, flux_check_cache

    def make_decode(model: torch.nn.Module, base: DynamicCache) -> Callable[[], object]:
        cache = _clone_cache(base, model.config)
        return lambda: model(
            input_ids=next_token,
            past_key_values=cache,
            use_cache=True,
        )

    medians = alternating_event_medians(
        {"reference": lambda: None, "Flux": lambda: None},
        warmup,
        repetitions,
        prepare={
            "reference": lambda: make_decode(reference, reference_cache),
            "Flux": lambda: make_decode(flux, flux_cache),
        },
    )
    return Comparison(context_length, medians["reference"], medians["Flux"], maximum)
