"""Opt-in CUDA-Graph execution for fixed-shape Flux SmolLM2 decode.

The ordinary Flux model path remains eager.  This module owns the stable input,
position, attention-mask, output, and Hugging Face ``StaticCache`` tensors
needed to capture one batch-preserving, single-token cached-decode step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import DynamicCache, StaticCache


@dataclass(frozen=True)
class CUDAGraphMemory:
    """Persistent CUDA allocations owned by a captured decode state."""

    static_cache_bytes: int
    graph_pool_bytes: int
    setup_peak_bytes: int


@dataclass(frozen=True)
class CUDAGraphSetupTiming:
    """Synchronized one-time setup costs in milliseconds."""

    prefill_ms: float
    static_cache_ms: float
    warmup_ms: float
    capture_ms: float
    total_ms: float


class FluxCUDAGraphDecode:
    """Captured one-token decode state for an already Flux-enabled model.

    Use :meth:`capture` to prefill normally and create the state.  Each call to
    :meth:`replay` copies one CUDA token into a stable input buffer, replays the
    same graph, updates the static KV cache in place, and advances the logical
    position without recapture.
    """

    def __init__(self) -> None:
        # Construction is intentionally centralized in ``capture`` because a
        # partially initialized CUDA graph state is not useful or safe.
        raise TypeError("use FluxCUDAGraphDecode.capture(...) to create a state")

    @classmethod
    def capture(
        cls,
        model: nn.Module,
        input_ids: torch.Tensor,
        *,
        max_decode_steps: int,
        warmup_steps: int = 3,
    ) -> FluxCUDAGraphDecode:
        """Prefill ``input_ids`` and capture a reusable one-token decode graph.

        ``max_decode_steps`` is the number of subsequent one-token replays the
        fixed-capacity cache must hold.  Graph capture and warmup do not consume
        logical decode steps.
        """
        cls._validate_capture_inputs(model, input_ids, max_decode_steps, warmup_steps)

        self = object.__new__(cls)
        self.model = model
        self.device = input_ids.device
        self.batch_size = input_ids.shape[0]
        self.context_length = input_ids.shape[1]
        self.max_decode_steps = max_decode_steps
        self.max_cache_len = self.context_length + max_decode_steps
        self.steps_replayed = 0

        torch.cuda.synchronize(self.device)
        setup_started = time.perf_counter()
        setup_baseline = torch.cuda.memory_allocated(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)

        # Normal eager prefill remains the correctness/performance oracle.  The
        # populated DynamicCache is copied exactly once into fixed storage.
        with torch.inference_mode():
            prefill_output = model(
                input_ids=input_ids,
                use_cache=True,
                logits_to_keep=1,
            )
        dynamic_cache = prefill_output.past_key_values
        if not isinstance(dynamic_cache, DynamicCache):
            raise TypeError(
                "Flux CUDA-Graph prefill expected Transformers DynamicCache, "
                f"got {type(dynamic_cache).__name__}"
            )
        self.prefill_logits = prefill_output.logits.detach()
        del prefill_output
        torch.cuda.synchronize(self.device)
        prefill_finished = time.perf_counter()

        static_cache_started = time.perf_counter()
        self.cache = StaticCache(
            config=model.config,
            max_cache_len=self.max_cache_len,
        )
        cls._initialize_static_cache(self.cache, dynamic_cache)
        del dynamic_cache
        torch.cuda.synchronize(self.device)
        static_cache_finished = time.perf_counter()
        static_cache_bytes = sum(
            layer.keys.numel() * layer.keys.element_size()
            + layer.values.numel() * layer.values.element_size()
            for layer in self.cache.layers
        )

        self.input_ids = torch.empty(
            (self.batch_size, 1), dtype=torch.long, device=self.device
        )
        self.input_ids.zero_()
        self.position_ids = torch.full(
            (1, 1), self.context_length, dtype=torch.long, device=self.device
        )
        mask_min = torch.finfo(self.prefill_logits.dtype).min
        self.attention_mask = torch.full(
            (self.batch_size, 1, 1, self.max_cache_len),
            mask_min,
            dtype=self.prefill_logits.dtype,
            device=self.device,
        )
        self.attention_mask[..., : self.context_length].zero_()
        self._zero_mask_column = torch.zeros(
            (self.batch_size, 1, 1, 1),
            dtype=self.attention_mask.dtype,
            device=self.device,
        )
        self._mask_min = mask_min

        # CUDA Graphs require allocator warmup on a side stream.  Restore all
        # logical state after every warm step so setup never consumes a token.
        warmup_started = time.perf_counter()
        if warmup_steps:
            current_stream = torch.cuda.current_stream(self.device)
            warmup_stream = torch.cuda.Stream(device=self.device)
            warmup_stream.wait_stream(current_stream)
            with torch.cuda.stream(warmup_stream), torch.inference_mode():
                for _ in range(warmup_steps):
                    self._decode_body()
                    self._restore_logical_state()
            current_stream.wait_stream(warmup_stream)
            current_stream.synchronize()
        warmup_finished = time.perf_counter()

        before_capture = torch.cuda.memory_allocated(self.device)
        capture_started = time.perf_counter()
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self.logits = self._decode_body()

        # Capture executes the body once.  Its result is intentionally discarded
        # and the cache/mask/position are restored before the first real replay.
        self._restore_logical_state(clear_decode_slots=True)
        torch.cuda.synchronize(self.device)
        capture_finished = time.perf_counter()
        after_capture = torch.cuda.memory_allocated(self.device)
        setup_peak = torch.cuda.max_memory_allocated(self.device)
        self.memory = CUDAGraphMemory(
            static_cache_bytes=static_cache_bytes,
            graph_pool_bytes=max(0, after_capture - before_capture),
            setup_peak_bytes=max(0, setup_peak - setup_baseline),
        )
        self.setup_timing = CUDAGraphSetupTiming(
            prefill_ms=(prefill_finished - setup_started) * 1000.0,
            static_cache_ms=(static_cache_finished - static_cache_started) * 1000.0,
            warmup_ms=(warmup_finished - warmup_started) * 1000.0,
            capture_ms=(capture_finished - capture_started) * 1000.0,
            total_ms=(capture_finished - setup_started) * 1000.0,
        )
        return self

    @staticmethod
    def _validate_capture_inputs(
        model: nn.Module,
        input_ids: torch.Tensor,
        max_decode_steps: int,
        warmup_steps: int,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Flux CUDA-Graph decode requires CUDA")
        if not getattr(model, "_flux_ops_enabled", False):
            raise ValueError("enable Flux operators on the model before graph capture")
        if model.training:
            raise ValueError("Flux CUDA-Graph decode requires model.eval()")
        if input_ids.device.type != "cuda":
            raise ValueError("input_ids must be a CUDA tensor")
        model_device = next(model.parameters()).device
        if model_device != input_ids.device:
            raise ValueError(
                f"model is on {model_device}, but input_ids are on {input_ids.device}"
            )
        if input_ids.dtype != torch.long or input_ids.ndim != 2:
            raise ValueError("input_ids must be a rank-2 torch.long tensor")
        if input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
            raise ValueError("input_ids batch and sequence dimensions must be non-empty")
        if max_decode_steps < 1:
            raise ValueError("max_decode_steps must be positive")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        maximum = int(model.config.max_position_embeddings)
        requested = input_ids.shape[1] + max_decode_steps
        if requested > maximum:
            raise ValueError(
                f"requested cache capacity {requested} exceeds model maximum {maximum}"
            )
        required = {"rmsnorm", "residual_rmsnorm", "softmax"}
        selected = set(getattr(model, "_flux_operator_categories", ()))
        optional = {"rope", "mlp", "packed_swiglu"}
        if not required.issubset(selected) or selected - required - optional:
            raise ValueError(
                "Flux CUDA-Graph decode requires rmsnorm, residual_rmsnorm, "
                "and softmax operators, with optional rope, packed MLP, and "
                "packed SwiGLU"
            )

    @staticmethod
    def _initialize_static_cache(
        static_cache: StaticCache,
        dynamic_cache: DynamicCache,
    ) -> None:
        if len(static_cache.layers) != len(dynamic_cache.layers):
            raise ValueError("dynamic and static cache layer counts differ")
        for layer_index, (static_layer, dynamic_layer) in enumerate(
            zip(static_cache.layers, dynamic_cache.layers, strict=True)
        ):
            keys = dynamic_layer.keys
            values = dynamic_layer.values
            if keys is None or values is None:
                raise ValueError(f"dynamic cache layer {layer_index} is uninitialized")
            context_length = keys.shape[-2]
            if context_length > static_layer.max_cache_len:
                raise ValueError("prefill cache does not fit fixed-capacity storage")
            static_layer.lazy_initialization(keys, values)
            static_layer.keys[..., :context_length, :].copy_(keys)
            static_layer.values[..., :context_length, :].copy_(values)
            static_layer.cumulative_length.fill_(context_length)

    def _decode_body(self) -> torch.Tensor:
        # The 4D mask bypasses Transformers' capture-incompatible eager mask
        # scalar construction.  Only the new token's column changes each step.
        self.attention_mask.index_copy_(
            3,
            self.position_ids.reshape(-1),
            self._zero_mask_column,
        )
        output = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            position_ids=self.position_ids,
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=1,
        )
        self.position_ids.add_(1)
        return output.logits

    def _restore_logical_state(self, *, clear_decode_slots: bool = False) -> None:
        self.position_ids.fill_(self.context_length)
        self.attention_mask.fill_(self._mask_min)
        self.attention_mask[..., : self.context_length].zero_()
        for layer in self.cache.layers:
            layer.cumulative_length.fill_(self.context_length)
            if clear_decode_slots:
                layer.keys[..., self.context_length :, :].zero_()
                layer.values[..., self.context_length :, :].zero_()

    def replay(self, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Replay one decode step and return the graph-owned logits tensor.

        The returned tensor has a stable address and is overwritten by the next
        replay.  ``input_ids=None`` reuses the token already in the input buffer,
        which is useful for pure graph-replay measurement.
        """
        if self.steps_replayed >= self.max_decode_steps:
            raise RuntimeError("fixed-capacity decode state is exhausted")
        if input_ids is not None:
            if (
                input_ids.shape != self.input_ids.shape
                or input_ids.dtype != self.input_ids.dtype
                or input_ids.device != self.input_ids.device
            ):
                raise ValueError(
                    "replay input must match the captured CUDA input's shape, "
                    "dtype, and device"
                )
            self.input_ids.copy_(input_ids)
        self.graph.replay()
        self.steps_replayed += 1
        return self.logits

    @property
    def cache_position(self) -> int:
        """Current logical cache position (synchronizes to read CUDA state)."""
        return int(self.position_ids.item())

    @property
    def cache_bytes(self) -> int:
        """Exact bytes in the static K/V backing tensors."""
        return sum(
            layer.keys.numel() * layer.keys.element_size()
            + layer.values.numel() * layer.values.element_size()
            for layer in self.cache.layers
        )

    def stable_addresses(self) -> dict[str, Any]:
        """Return captured tensor data pointers for diagnostics and tests."""
        return {
            "input_ids": self.input_ids.data_ptr(),
            "position_ids": self.position_ids.data_ptr(),
            "attention_mask": self.attention_mask.data_ptr(),
            "logits": self.logits.data_ptr(),
            "keys": tuple(layer.keys.data_ptr() for layer in self.cache.layers),
            "values": tuple(layer.values.data_ptr() for layer in self.cache.layers),
        }


def cuda_graph_greedy_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    warmup_steps: int = 3,
) -> torch.Tensor:
    """Generate greedily with one prefill followed by captured decode replays."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if max_new_tokens == 1:
        with torch.inference_mode():
            logits = model(input_ids=input_ids, use_cache=False, logits_to_keep=1).logits
        return torch.cat((input_ids, logits.argmax(dim=-1)), dim=-1)

    state = FluxCUDAGraphDecode.capture(
        model,
        input_ids,
        max_decode_steps=max_new_tokens - 1,
        warmup_steps=warmup_steps,
    )
    token = state.prefill_logits.argmax(dim=-1)
    generated = [input_ids, token]
    for _ in range(max_new_tokens - 1):
        logits = state.replay(token)
        token = logits.argmax(dim=-1)
        generated.append(token)
    return torch.cat(generated, dim=-1)


__all__ = [
    "CUDAGraphMemory",
    "CUDAGraphSetupTiming",
    "FluxCUDAGraphDecode",
    "cuda_graph_greedy_generate",
]
