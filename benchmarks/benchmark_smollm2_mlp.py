"""Validate and benchmark the opt-in packed SmolLM2 gate/up projection.

Run from the repository root after building the native Flux extension:

    build/python3119/python.exe benchmarks/benchmark_smollm2_mlp.py

The benchmark compares the established Flux model path against the same path
with the ``mlp`` category enabled. Timed regions contain only the operation
under test. CUDA-event samples alternate implementation order and report the
median. Inputs, weight packing, cache cloning, and correctness checks remain
outside timed regions.
"""

from __future__ import annotations

import argparse
import gc
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch.profiler import ProfilerActivity, profile

from benchmarks.benchmark_smollm2 import (
    ATOL,
    RTOL,
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
    FluxPackedLlamaMLP,
    enable_flux_ops,
)
from flux.ops import (
    native_attention_score_softmax_is_available,
    native_residual_rmsnorm_is_available,
    native_rmsnorm_is_available,
    native_rope_is_available,
    native_softmax_is_available,
)


DEFAULT_LENGTHS = (128, 512, 1024, 2048, 4096)
DEFAULT_DECODE_CONTEXTS = (128, 512, 1024, 2048, 4096)
DEFAULT_WARMUP = 5
DEFAULT_REPETITIONS = 30
DEFAULT_STABILIZATION_ITERATIONS = 50
MIB = 1024.0**2
MLP_RTOL = 1e-3
MLP_ATOL = 2e-4
LOGITS_RTOL = 2e-4
LOGITS_ATOL = 3e-5
CORRECTNESS_CHUNK_TOKENS = 256


@dataclass(frozen=True)
class Comparison:
    size: int
    separate_ms: float
    packed_ms: float
    max_absolute_error: float

    @property
    def speedup(self) -> float:
        return self.separate_ms / self.packed_ms


@dataclass(frozen=True)
class LengthResult:
    projection: Comparison
    mlp: Comparison
    prefill: Comparison
    gate_max_absolute_error: float
    up_max_absolute_error: float


@dataclass(frozen=True)
class DecodeResult:
    comparison: Comparison
    cache_max_absolute_error: float


@dataclass(frozen=True)
class MemoryResult:
    parameter_mib_before: float
    parameter_mib_after: float
    storage_mib_before: float
    storage_mib_after: float
    gate_up_mib: float
    allocated_mib_before: float
    allocated_mib_after: float
    conversion_peak_increment_mib: float


@dataclass(frozen=True)
class CorrectnessResult:
    layer_max_absolute_error: float
    greedy_equal: bool


@dataclass(frozen=True)
class ProfileRow:
    path: str
    name: str
    count: int
    cuda_ms: float
    self_cuda_memory_mib: float


def _parse_int_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument(
        "--stabilization-iterations",
        type=int,
        default=DEFAULT_STABILIZATION_ITERATIONS,
    )
    parser.add_argument("--lengths", type=_parse_int_list, default=DEFAULT_LENGTHS)
    parser.add_argument(
        "--decode-contexts",
        type=_parse_int_list,
        default=DEFAULT_DECODE_CONTEXTS,
    )
    parser.add_argument("--profile-length", type=int, default=1024)
    parser.add_argument("--skip-profiler", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.stabilization_iterations < 0:
        parser.error("warmup counts must be non-negative")
    if args.repetitions < 1 or args.profile_length < 1:
        parser.error("repetitions and profile length must be positive")
    return args


def _required_flux_ops_available() -> bool:
    return all(
        (
            native_attention_score_softmax_is_available(),
            native_residual_rmsnorm_is_available(),
            native_rmsnorm_is_available(),
            native_rope_is_available(),
            native_softmax_is_available(),
        )
    )


def _parameter_bytes(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )


def _unique_parameter_storage_bytes(model: torch.nn.Module) -> int:
    storages: dict[tuple[str, int], int] = {}
    for parameter in model.parameters():
        storage = parameter.untyped_storage()
        key = (str(parameter.device), storage.data_ptr())
        storages.setdefault(key, storage.nbytes())
    return sum(storages.values())


def _assert_packed_logits_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    if actual.shape != expected.shape:
        raise AssertionError(f"logit shape mismatch: {actual.shape} != {expected.shape}")
    maximum = 0.0
    failures = 0
    for start in range(0, actual.shape[1], CORRECTNESS_CHUNK_TOKENS):
        stop = min(start + CORRECTNESS_CHUNK_TOKENS, actual.shape[1])
        difference = (actual[:, start:stop] - expected[:, start:stop]).abs()
        allowed = LOGITS_ATOL + LOGITS_RTOL * expected[:, start:stop].abs()
        failures += int(torch.count_nonzero(difference > allowed).item())
        maximum = max(maximum, float(difference.max().item()))
    if failures:
        raise AssertionError(
            f"packed logits differ beyond rtol={LOGITS_RTOL}, atol={LOGITS_ATOL}; "
            f"failures={failures}/{actual.numel()}, max absolute error={maximum:.9g}"
        )
    return maximum


def _load_models() -> tuple[torch.nn.Module, torch.nn.Module, MemoryResult]:
    print("Loading established Flux model...", flush=True)
    separate = enable_flux_ops(load_model("cuda"))
    print("Loading packed-MLP candidate...", flush=True)
    packed_source = load_model("cuda")
    parameter_before = _parameter_bytes(packed_source)
    storage_before = _unique_parameter_storage_bytes(packed_source)
    gate_up_bytes = sum(
        layer.mlp.gate_proj.weight.numel() * layer.mlp.gate_proj.weight.element_size()
        + layer.mlp.up_proj.weight.numel() * layer.mlp.up_proj.weight.element_size()
        for layer in packed_source.model.layers
    )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    selected = FLUX_OPERATOR_CATEGORIES | {FLUX_PACKED_MLP_CATEGORY}
    packed = enable_flux_ops(packed_source, operators=selected)
    torch.cuda.synchronize()
    conversion_peak = torch.cuda.max_memory_allocated() - allocated_before
    gc.collect()
    torch.cuda.synchronize()
    allocated_after = torch.cuda.memory_allocated()

    memory = MemoryResult(
        parameter_before / MIB,
        _parameter_bytes(packed) / MIB,
        storage_before / MIB,
        _unique_parameter_storage_bytes(packed) / MIB,
        gate_up_bytes / MIB,
        allocated_before / MIB,
        allocated_after / MIB,
        conversion_peak / MIB,
    )
    return separate, packed, memory


def _capture_mlp_inputs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    captured: list[torch.Tensor | None] = [None] * len(model.model.layers)

    def make_hook(index: int) -> Callable[..., None]:
        def capture(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            captured[index] = inputs[0].detach().clone()

        return capture

    handles = [
        layer.mlp.register_forward_pre_hook(make_hook(index))
        for index, layer in enumerate(model.model.layers)
    ]
    try:
        output = model(input_ids=input_ids, use_cache=False)
        torch.cuda.synchronize()
        del output
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in captured):
        raise AssertionError("failed to capture every MLP input")
    return tuple(value for value in captured if value is not None)


def _repeat_separate_projections(
    mlps: Sequence[torch.nn.Module],
    inputs: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    output = (mlps[0].gate_proj(inputs[0]), mlps[0].up_proj(inputs[0]))
    for mlp, hidden_states in zip(mlps[1:], inputs[1:], strict=True):
        output = (mlp.gate_proj(hidden_states), mlp.up_proj(hidden_states))
    return output


def _repeat_packed_projections(
    mlps: Sequence[FluxPackedLlamaMLP],
    inputs: Sequence[torch.Tensor],
) -> torch.Tensor:
    output = mlps[0].gate_up_proj(inputs[0])
    for mlp, hidden_states in zip(mlps[1:], inputs[1:], strict=True):
        output = mlp.gate_up_proj(hidden_states)
    return output


def _repeat_mlps(
    mlps: Sequence[torch.nn.Module],
    inputs: Sequence[torch.Tensor],
) -> torch.Tensor:
    output = mlps[0](inputs[0])
    for mlp, hidden_states in zip(mlps[1:], inputs[1:], strict=True):
        output = mlp(hidden_states)
    return output


def _maximum_projection_and_mlp_errors(
    separate_mlps: Sequence[torch.nn.Module],
    packed_mlps: Sequence[FluxPackedLlamaMLP],
    inputs: Sequence[torch.Tensor],
) -> tuple[float, float, float]:
    gate_max = 0.0
    up_max = 0.0
    mlp_max = 0.0
    for separate_mlp, packed_mlp, hidden_states in zip(
        separate_mlps, packed_mlps, inputs, strict=True
    ):
        expected_gate = separate_mlp.gate_proj(hidden_states)
        expected_up = separate_mlp.up_proj(hidden_states)
        gate_up = packed_mlp.gate_up_proj(hidden_states)
        actual_gate, actual_up = gate_up.chunk(2, dim=-1)
        if actual_gate.untyped_storage().data_ptr() != gate_up.untyped_storage().data_ptr():
            raise AssertionError("packed gate split unexpectedly allocated storage")
        if actual_up.untyped_storage().data_ptr() != gate_up.untyped_storage().data_ptr():
            raise AssertionError("packed up split unexpectedly allocated storage")
        gate_max = max(gate_max, float((actual_gate - expected_gate).abs().max().item()))
        up_max = max(up_max, float((actual_up - expected_up).abs().max().item()))
        expected_mlp = separate_mlp(hidden_states)
        actual_mlp = packed_mlp(hidden_states)
        torch.testing.assert_close(actual_gate, expected_gate, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(actual_up, expected_up, rtol=2e-5, atol=2e-5)
        # Joining the two output-column groups can select a different cuBLAS
        # algorithm. The projection tolerance remains tight; the MLP tolerance
        # accounts for amplification by SiLU, gating, and down projection.
        torch.testing.assert_close(
            actual_mlp,
            expected_mlp,
            rtol=MLP_RTOL,
            atol=MLP_ATOL,
        )
        mlp_max = max(mlp_max, float((actual_mlp - expected_mlp).abs().max().item()))
    return gate_max, up_max, mlp_max


def _benchmark_length(
    separate: torch.nn.Module,
    packed: torch.nn.Module,
    length: int,
    warmup: int,
    repetitions: int,
) -> LengthResult:
    input_ids = _input_ids(length, separate.config.vocab_size)
    separate_output = separate(input_ids=input_ids, use_cache=True)
    packed_output = packed(input_ids=input_ids, use_cache=True)
    logits_error = _assert_packed_logits_close(packed_output.logits, separate_output.logits)
    if separate_output.past_key_values.get_seq_length() != length:
        raise AssertionError("separate prefill cache has incorrect length")
    if packed_output.past_key_values.get_seq_length() != length:
        raise AssertionError("packed prefill cache has incorrect length")
    del separate_output, packed_output

    prefill_times = _event_latencies(
        {
            "separate": lambda: separate(input_ids=input_ids, use_cache=True),
            "packed": lambda: packed(input_ids=input_ids, use_cache=True),
        },
        warmup,
        repetitions,
    )

    inputs = _capture_mlp_inputs(separate, input_ids)
    separate_mlps = tuple(layer.mlp for layer in separate.model.layers)
    packed_mlps = tuple(layer.mlp for layer in packed.model.layers)
    if not all(isinstance(mlp, FluxPackedLlamaMLP) for mlp in packed_mlps):
        raise AssertionError("packed candidate did not replace every MLP")
    gate_error, up_error, mlp_error = _maximum_projection_and_mlp_errors(
        separate_mlps, packed_mlps, inputs
    )
    projection_times = _event_latencies(
        {
            "separate": lambda: _repeat_separate_projections(separate_mlps, inputs),
            "packed": lambda: _repeat_packed_projections(packed_mlps, inputs),
        },
        warmup,
        repetitions,
    )
    mlp_times = _event_latencies(
        {
            "separate": lambda: _repeat_mlps(separate_mlps, inputs),
            "packed": lambda: _repeat_mlps(packed_mlps, inputs),
        },
        warmup,
        repetitions,
    )
    del input_ids, inputs
    return LengthResult(
        Comparison(length, projection_times["separate"], projection_times["packed"], max(gate_error, up_error)),
        Comparison(length, mlp_times["separate"], mlp_times["packed"], mlp_error),
        Comparison(length, prefill_times["separate"], prefill_times["packed"], logits_error),
        gate_error,
        up_error,
    )


def _cache_max_absolute_error(actual: object, expected: object) -> float:
    maximum = 0.0
    for actual_layer, expected_layer in zip(actual.layers, expected.layers, strict=True):
        for actual_tensor, expected_tensor in (
            (actual_layer.keys, expected_layer.keys),
            (actual_layer.values, expected_layer.values),
        ):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=RTOL, atol=ATOL)
            maximum = max(maximum, float((actual_tensor - expected_tensor).abs().max().item()))
    return maximum


def _benchmark_decode(
    separate: torch.nn.Module,
    packed: torch.nn.Module,
    context_length: int,
    warmup: int,
    repetitions: int,
) -> DecodeResult:
    context_ids = _input_ids(context_length, separate.config.vocab_size)
    next_token = _input_ids(context_length + 1, separate.config.vocab_size)[:, -1:]
    separate_cache = _prefill_cache(separate, context_ids)
    packed_cache = _prefill_cache(packed, context_ids)

    separate_check_cache = _clone_cache(separate_cache, separate.config)
    packed_check_cache = _clone_cache(packed_cache, packed.config)
    separate_output = separate(input_ids=next_token, past_key_values=separate_check_cache, use_cache=True)
    packed_output = packed(input_ids=next_token, past_key_values=packed_check_cache, use_cache=True)
    logits_error = _assert_packed_logits_close(packed_output.logits, separate_output.logits)
    expected_length = context_length + 1
    if separate_output.past_key_values.get_seq_length() != expected_length:
        raise AssertionError("separate decode cache has incorrect length")
    if packed_output.past_key_values.get_seq_length() != expected_length:
        raise AssertionError("packed decode cache has incorrect length")
    cache_error = _cache_max_absolute_error(packed_output.past_key_values, separate_output.past_key_values)
    del separate_output, packed_output, separate_check_cache, packed_check_cache

    def prepare_decode(model: torch.nn.Module, cache: object) -> Callable[[], object]:
        sample_cache = _clone_cache(cache, model.config)
        return lambda: model(input_ids=next_token, past_key_values=sample_cache, use_cache=True)

    times = _event_latencies(
        {"separate": lambda: None, "packed": lambda: None},
        warmup,
        repetitions,
        prepare={
            "separate": lambda: prepare_decode(separate, separate_cache),
            "packed": lambda: prepare_decode(packed, packed_cache),
        },
    )
    del context_ids, next_token, separate_cache, packed_cache
    return DecodeResult(
        Comparison(context_length, times["separate"], times["packed"], logits_error),
        cache_error,
    )


def _benchmark_graph_decode(
    separate: torch.nn.Module,
    packed: torch.nn.Module,
    context_length: int,
    warmup: int,
    repetitions: int,
) -> Comparison:
    prompt = _input_ids(context_length, separate.config.vocab_size)
    next_token = _input_ids(context_length + 1, separate.config.vocab_size)[:, -1:]
    capacity = 1 + warmup + repetitions
    separate_graph = FluxCUDAGraphDecode.capture(
        separate,
        prompt,
        max_decode_steps=capacity,
    )
    packed_graph = FluxCUDAGraphDecode.capture(
        packed,
        prompt,
        max_decode_steps=capacity,
    )
    _assert_packed_logits_close(
        packed_graph.prefill_logits,
        separate_graph.prefill_logits,
    )
    separate_logits = separate_graph.replay(next_token)
    packed_logits = packed_graph.replay(next_token)
    logits_error = _assert_packed_logits_close(packed_logits, separate_logits)

    times = _event_latencies(
        {
            "separate": lambda: separate_graph.replay(next_token),
            "packed": lambda: packed_graph.replay(next_token),
        },
        warmup,
        repetitions,
    )
    del prompt, next_token, separate_graph, packed_graph
    return Comparison(
        context_length,
        times["separate"],
        times["packed"],
        logits_error,
    )


def _layer_and_generation_correctness(
    separate: torch.nn.Module,
    packed: torch.nn.Module,
) -> CorrectnessResult:
    input_ids = _input_ids(128, separate.config.vocab_size)
    captured: dict[str, torch.Tensor] = {}

    def capture(name: str) -> Callable[..., None]:
        def hook(_module: torch.nn.Module, _inputs: object, output: torch.Tensor) -> None:
            captured[name] = output.detach().clone()

        return hook

    separate_handle = separate.model.layers[0].register_forward_hook(capture("separate"))
    packed_handle = packed.model.layers[0].register_forward_hook(capture("packed"))
    try:
        separate_output = separate(input_ids=input_ids, use_cache=False)
        packed_output = packed(input_ids=input_ids, use_cache=False)
        del separate_output, packed_output
    finally:
        separate_handle.remove()
        packed_handle.remove()
    torch.testing.assert_close(captured["packed"], captured["separate"], rtol=RTOL, atol=ATOL)
    layer_error = float((captured["packed"] - captured["separate"]).abs().max().item())

    prompt = input_ids[:, :32]
    separate_tokens = separate.generate(prompt, do_sample=False, max_new_tokens=8, use_cache=True)
    packed_tokens = packed.generate(prompt, do_sample=False, max_new_tokens=8, use_cache=True)
    greedy_equal = bool(torch.equal(packed_tokens, separate_tokens))
    if not greedy_equal:
        raise AssertionError("packed MLP changed greedy generation token IDs")
    del input_ids, prompt, separate_tokens, packed_tokens
    return CorrectnessResult(layer_error, greedy_equal)


def _profile_mlp(path: str, mlp: torch.nn.Module, hidden_states: torch.Tensor) -> list[ProfileRow]:
    output = mlp(hidden_states)
    torch.cuda.synchronize()
    del output
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        output = mlp(hidden_states)
    torch.cuda.synchronize()
    del output
    wanted = {"aten::linear", "aten::mm", "aten::silu", "aten::mul", "aten::split", "aten::narrow"}
    rows = []
    for event in profiler.key_averages():
        if event.key in wanted:
            rows.append(
                ProfileRow(
                    path,
                    event.key,
                    int(event.count),
                    float(getattr(event, "device_time_total", 0.0)) / 1000.0,
                    float(getattr(event, "self_device_memory_usage", 0.0)) / MIB,
                )
            )
    return sorted(rows, key=lambda row: (row.path, row.name))


def _print_environment(args: argparse.Namespace) -> None:
    print("SmolLM2 packed gate/up benchmark")
    print(f"  Model: {MODEL_ID}")
    print(f"  Revision: {MODEL_REVISION}")
    print(f"  Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"  PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    capability = torch.cuda.get_device_capability()
    print(f"  Compute capability: {capability[0]}.{capability[1]}")
    print("  FP32; eager attention; TF32 disabled; deterministic algorithms enabled")
    print(f"  warmup={args.warmup}; repetitions={args.repetitions}; statistic=median")


def _print_comparison(title: str, results: Sequence[Comparison], unit: str = "ms") -> None:
    print(f"\n{title}")
    print(f"{'tokens':>7} {'separate ' + unit:>14} {'packed ' + unit:>12} {'speedup':>10} {'max abs err':>13}")
    for result in results:
        print(
            f"{result.size:>7} {result.separate_ms:>14.4f} {result.packed_ms:>12.4f} "
            f"{result.speedup:>9.3f}x {result.max_absolute_error:>13.6g}"
        )


def _print_results(
    lengths: Sequence[LengthResult],
    decodes: Sequence[DecodeResult],
    graph_decodes: Sequence[Comparison],
    memory: MemoryResult,
    correctness: CorrectnessResult,
    profiles: Sequence[ProfileRow],
    profile_length: int,
) -> None:
    print("\nPersistent parameter/storage memory")
    print(f"  parameter bytes before: {memory.parameter_mib_before:.3f} MiB")
    print(f"  parameter bytes after:  {memory.parameter_mib_after:.3f} MiB")
    print(f"  unique storage before:  {memory.storage_mib_before:.3f} MiB")
    print(f"  unique storage after:   {memory.storage_mib_after:.3f} MiB")
    print(f"  gate/up storage packed: {memory.gate_up_mib:.3f} MiB")
    print(f"  CUDA allocated before conversion: {memory.allocated_mib_before:.3f} MiB")
    print(f"  CUDA allocated after conversion:  {memory.allocated_mib_after:.3f} MiB")
    print(f"  conversion peak increment:        {memory.conversion_peak_increment_mib:.3f} MiB")

    _print_comparison("Isolated gate/up projections across all layers", [result.projection for result in lengths])
    _print_comparison("Complete MLP across all layers", [result.mlp for result in lengths])
    _print_comparison("Integrated full-model prefill", [result.prefill for result in lengths])
    _print_comparison("One-token cached decode", [result.comparison for result in decodes], "ms/token")
    _print_comparison("CUDA-Graph cached decode replay", graph_decodes, "ms/token")

    print("\nNumerical maxima")
    for result in lengths:
        print(
            f"  {result.prefill.size:>4} tokens: gate={result.gate_max_absolute_error:.9g}, "
            f"up={result.up_max_absolute_error:.9g}, mlp={result.mlp.max_absolute_error:.9g}, "
            f"logits={result.prefill.max_absolute_error:.9g}"
        )
    print(f"  decoder layer (128 tokens): {correctness.layer_max_absolute_error:.9g}")
    for result in decodes:
        print(
            f"  decode context {result.comparison.size:>4}: logits={result.comparison.max_absolute_error:.9g}, "
            f"cache={result.cache_max_absolute_error:.9g}"
        )
    print(f"  greedy generation token IDs equal: {correctness.greedy_equal}")

    if profiles:
        print(f"\nDiagnostic one-layer profiler at {profile_length} tokens")
        print(f"{'path':>10} {'operator':>16} {'count':>7} {'CUDA ms':>10} {'self alloc MiB':>15}")
        for row in profiles:
            print(
                f"{row.path:>10} {row.name:>16} {row.count:>7} "
                f"{row.cuda_ms:>10.4f} {row.self_cuda_memory_mib:>15.3f}"
            )


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the SmolLM2 MLP benchmark")
    if not _required_flux_ops_available():
        raise RuntimeError("all built Flux operators are required for integrated prefill")
    _configure_runtime()
    _print_environment(args)
    separate, packed, memory = _load_models()
    maximum_position = int(separate.config.max_position_embeddings)
    valid_lengths = tuple(length for length in args.lengths if length <= maximum_position)
    valid_contexts = tuple(context for context in args.decode_contexts if context + 1 <= maximum_position)
    if not valid_lengths:
        raise RuntimeError("all requested lengths exceed max_position_embeddings")

    print("Stabilizing GPU clocks with alternating untimed prefills...", flush=True)
    stabilization_ids = _input_ids(min(valid_lengths), separate.config.vocab_size)
    with torch.inference_mode():
        for iteration in range(args.stabilization_iterations):
            models = (separate, packed) if iteration % 2 == 0 else (packed, separate)
            for model in models:
                output = model(input_ids=stabilization_ids, use_cache=True)
                del output
    torch.cuda.synchronize()
    del stabilization_ids

    length_results = []
    decode_results = []
    graph_decode_results = []
    profiles: list[ProfileRow] = []
    with torch.inference_mode():
        correctness = _layer_and_generation_correctness(separate, packed)
        for length in valid_lengths:
            print(f"Benchmarking {length}-token projection, MLP, and prefill...", flush=True)
            try:
                length_results.append(_benchmark_length(separate, packed, length, args.warmup, args.repetitions))
            except torch.OutOfMemoryError as error:
                print(f"Skipping length {length}: CUDA out of memory ({error})", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        for context in valid_contexts:
            print(f"Benchmarking one-token decode after context {context}...", flush=True)
            try:
                decode_results.append(_benchmark_decode(separate, packed, context, args.warmup, args.repetitions))
                graph_decode_results.append(
                    _benchmark_graph_decode(
                        separate,
                        packed,
                        context,
                        args.warmup,
                        args.repetitions,
                    )
                )
            except torch.OutOfMemoryError as error:
                print(f"Skipping decode context {context}: CUDA out of memory ({error})", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        if not args.skip_profiler and args.profile_length <= maximum_position:
            profile_ids = _input_ids(args.profile_length, separate.config.vocab_size)
            profile_input = _capture_mlp_inputs(separate, profile_ids)[0]
            profiles.extend(_profile_mlp("separate", separate.model.layers[0].mlp, profile_input))
            profiles.extend(_profile_mlp("packed", packed.model.layers[0].mlp, profile_input))
            del profile_ids, profile_input

    if not length_results:
        raise RuntimeError("no requested prefill length completed")
    _print_results(
        length_results,
        decode_results,
        graph_decode_results,
        memory,
        correctness,
        profiles,
        args.profile_length,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
