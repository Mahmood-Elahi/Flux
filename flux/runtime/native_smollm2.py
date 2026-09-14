"""Full-model native CUDA-Graph decode runtime for SmolLM2-135M."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn
from transformers import DynamicCache

from flux.ops.native_rmsnorm import _try_load_native_library


_ADDRESS_NAMES = (
    "input_token",
    "logits",
    "key_cache",
    "value_cache",
    "device_position",
    "device_cache_length",
    "workspace",
    "hidden_a",
    "hidden_b",
    "final_norm_output",
    "attention_workspace",
)

_PREFILL_ADDRESS_NAMES = (
    "prefill_logits",
    "prefill_workspace",
    "streaming_attention_output",
    "final_hidden",
) + _ADDRESS_NAMES


@dataclass(frozen=True)
class NativeDecodeRuntimeMemory:
    """Persistent memory attributable to the full native runtime."""

    cache_bytes: int
    workspace_bytes: int
    stable_buffer_bytes: int
    referenced_weight_bytes: int
    graph_resource_bytes: int | None = None

    @property
    def owned_tensor_bytes(self) -> int:
        return self.cache_bytes + self.workspace_bytes + self.stable_buffer_bytes


@dataclass(frozen=True)
class NativePrefillRuntimeMemory:
    """Persistent native prefill and directly attached decode storage."""

    cache_bytes: int
    prefill_workspace_bytes: int
    decode_workspace_bytes: int
    stable_buffer_bytes: int
    referenced_weight_bytes: int

    @property
    def owned_tensor_bytes(self) -> int:
        return (
            self.cache_bytes
            + self.prefill_workspace_bytes
            + self.decode_workspace_bytes
            + self.stable_buffer_bytes
        )


def native_smollm2_runtime_is_available() -> bool:
    """Return whether the full-model native custom class is registered."""
    _try_load_native_library()
    try:
        getattr(torch.classes.flux, "NativeSmolLM2Decode")
    except RuntimeError:
        return False
    return True


def native_smollm2_prefill_is_available() -> bool:
    """Return whether the native prompt-to-cache custom class is registered."""
    _try_load_native_library()
    try:
        getattr(torch.classes.flux, "NativeSmolLM2Prefill")
    except RuntimeError:
        return False
    return True


class NativeSmolLM2Prefill:
    """Native FP32 prompt prefill with direct native-decode continuation.

    The native object owns the compact cache used by its captured decode graph.
    Prefill writes that storage directly, so no DynamicCache construction,
    expanded GQA cache, or Python cache import occurs at the handoff.
    """

    def __init__(self) -> None:
        raise TypeError("use NativeSmolLM2Prefill.capture(...) to create a runtime")

    @classmethod
    def capture(
        cls,
        model: nn.Module,
        input_ids: torch.Tensor,
        *,
        max_decode_steps: int = 0,
    ) -> NativeSmolLM2Prefill:
        if max_decode_steps < 0:
            raise ValueError("max_decode_steps must be non-negative")
        requested = input_ids.shape[1] + max_decode_steps if input_ids.ndim == 2 else 0
        NativeSmolLM2Decode._validate_capture_model_input(
            model, input_ids, requested
        )
        if int(input_ids.min().item()) < 0 or int(input_ids.max().item()) >= int(
            model.config.vocab_size
        ):
            raise ValueError("input_ids contain a token outside the vocabulary")
        _try_load_native_library()
        if not native_smollm2_prefill_is_available():
            raise RuntimeError(
                "Flux native prefill runtime is not built. Set "
                "FLUX_BUILD_NATIVE=1 and rebuild the extension in place."
            )

        layers = list(model.model.layers)
        positions = torch.arange(
            requested, dtype=torch.long, device=input_ids.device
        ).unsqueeze(0)
        rope_seed = model.model.embed_tokens.weight[:1].view(1, 1, 576)
        with torch.inference_mode():
            cos, sin = model.model.rotary_emb(rope_seed, positions)
        cos = cos.reshape(requested, 64).contiguous()
        sin = sin.reshape(requested, 64).contiguous()

        self = object.__new__(cls)
        self.model = model
        self.input_ids = input_ids
        self._rope_cos = cos
        self._rope_sin = sin
        self._native = torch.classes.flux.NativeSmolLM2Prefill(
            input_ids,
            model.model.embed_tokens.weight,
            [layer.input_layernorm.weight for layer in layers],
            [layer.self_attn.packed_qkv.weight for layer in layers],
            [layer.self_attn.o_proj.weight for layer in layers],
            [layer.post_attention_layernorm.weight for layer in layers],
            [layer.mlp.gate_up_proj.weight for layer in layers],
            [layer.mlp.down_proj.weight for layer in layers],
            model.model.norm.weight,
            model.lm_head.weight,
            cos,
            sin,
            requested,
            [float(layer.input_layernorm.variance_epsilon) for layer in layers],
            [float(layer.self_attn.scaling) for layer in layers],
        )
        self.prefill_logits = self._native.logits()
        return self

    def prefill(self, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Re-run the fixed-shape prompt on the caller's current CUDA stream."""
        ids = self.input_ids if input_ids is None else input_ids
        if ids.shape != self.input_ids.shape:
            raise ValueError("reused native prefill requires the captured prompt shape")
        if ids.dtype != torch.long or ids.device != self.input_ids.device:
            raise ValueError("reused input_ids must preserve CUDA device and long dtype")
        if not ids.is_contiguous():
            raise ValueError("reused input_ids must be contiguous")
        self.input_ids = ids
        return self._native.prefill(ids)

    def replay(self, token: torch.Tensor | None = None) -> torch.Tensor:
        """Continue directly with one captured native decode step."""
        return self._native.replay(token)

    @property
    def logits(self) -> torch.Tensor:
        return self._native.logits()

    @property
    def final_hidden(self) -> torch.Tensor:
        return self._native.final_hidden()

    @property
    def key_cache(self) -> torch.Tensor:
        return self._native.key_cache()

    @property
    def value_cache(self) -> torch.Tensor:
        return self._native.value_cache()

    @property
    def device_position(self) -> torch.Tensor:
        return self._native.device_position()

    @property
    def device_cache_length(self) -> torch.Tensor:
        return self._native.device_cache_length()

    @property
    def cache_position(self) -> int:
        return int(self._native.position())

    @property
    def cache_length(self) -> int:
        return int(self._native.cache_length())

    @property
    def prompt_length(self) -> int:
        return int(self._native.prompt_length())

    @property
    def capacity(self) -> int:
        return int(self._native.capacity())

    @property
    def replay_count(self) -> int:
        return int(self._native.replay_count())

    @property
    def memory(self) -> NativePrefillRuntimeMemory:
        referenced = sum(t.numel() * t.element_size() for t in self.model.parameters())
        decode_workspace = NativeSmolLM2Decode._decode_workspace_bytes(self.capacity)
        return NativePrefillRuntimeMemory(
            cache_bytes=int(self._native.cache_bytes()),
            prefill_workspace_bytes=int(self._native.workspace_bytes()),
            decode_workspace_bytes=decode_workspace,
            stable_buffer_bytes=int(self._native.stable_buffer_bytes()),
            referenced_weight_bytes=referenced,
        )

    def stable_addresses(self) -> dict[str, int]:
        return dict(zip(_PREFILL_ADDRESS_NAMES, self._native.addresses(), strict=True))


class NativeSmolLM2Decode:
    """Thin Python owner around a token-to-logits native decode graph.

    Python performs checkpoint loading and eager prefill. The native object
    then owns the compact 30-layer cache, stable token/logit/state buffers,
    shared workspace, execution descriptors, stream resources, and CUDA Graph.
    """

    def __init__(self) -> None:
        raise TypeError("use NativeSmolLM2Decode.capture(...) to create a runtime")

    @classmethod
    def capture(
        cls,
        model: nn.Module,
        input_ids: torch.Tensor,
        *,
        max_decode_steps: int,
        initial_token: torch.Tensor | None = None,
    ) -> NativeSmolLM2Decode:
        """Prefill eagerly, then capture native token-to-logits decode.

        ``max_decode_steps`` is the number of native replays the fixed cache can
        hold. When ``initial_token`` is omitted, the prefill argmax token is
        installed as the first replay input.
        """
        cls._validate_capture(model, input_ids, max_decode_steps)
        with torch.inference_mode():
            prefill = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        if not isinstance(prefill.past_key_values, DynamicCache):
            raise TypeError(
                "native SmolLM2 prefill expected Transformers DynamicCache, got "
                f"{type(prefill.past_key_values).__name__}"
            )
        token = (
            prefill.logits.argmax(dim=-1)
            if initial_token is None
            else initial_token
        )
        key_caches = [layer.keys for layer in prefill.past_key_values.layers]
        value_caches = [layer.values for layer in prefill.past_key_values.layers]
        if any(tensor is None for tensor in key_caches + value_caches):
            raise ValueError("prefill produced an uninitialized decoder cache")
        self = cls.capture_from_state(
            model,
            token,
            key_caches,  # type: ignore[arg-type]
            value_caches,  # type: ignore[arg-type]
            cache_position=input_ids.shape[1],
            cache_capacity=input_ids.shape[1] + max_decode_steps,
        )
        self.prefill_logits = prefill.logits.detach()
        self.prefill_input_ids = input_ids
        self._prefill_cache = prefill.past_key_values
        return self

    @classmethod
    def capture_from_state(
        cls,
        model: nn.Module,
        initial_token: torch.Tensor,
        key_caches: Sequence[torch.Tensor],
        value_caches: Sequence[torch.Tensor],
        *,
        cache_position: int,
        cache_capacity: int,
    ) -> NativeSmolLM2Decode:
        """Capture from an already-prefilled compact cache."""
        cls._validate_model_and_state(
            model,
            initial_token,
            key_caches,
            value_caches,
            cache_position=cache_position,
            cache_capacity=cache_capacity,
        )
        _try_load_native_library()
        if not native_smollm2_runtime_is_available():
            raise RuntimeError(
                "Flux native full-model decode runtime is not built. Set "
                "FLUX_BUILD_NATIVE=1 and rebuild the extension in place."
            )

        layers = list(model.model.layers)
        positions = torch.arange(
            cache_capacity, dtype=torch.long, device=initial_token.device
        ).unsqueeze(0)
        rope_seed = model.model.embed_tokens.weight[:1].view(1, 1, 576)
        with torch.inference_mode():
            cos, sin = model.model.rotary_emb(rope_seed, positions)
        cos = cos.reshape(cache_capacity, 64).contiguous()
        sin = sin.reshape(cache_capacity, 64).contiguous()

        self = object.__new__(cls)
        self.model = model
        self._rope_cos = cos
        self._rope_sin = sin
        self._source_key_caches = tuple(key_caches)
        self._source_value_caches = tuple(value_caches)
        self._native = torch.classes.flux.NativeSmolLM2Decode(
            initial_token,
            model.model.embed_tokens.weight,
            [layer.input_layernorm.weight for layer in layers],
            [layer.self_attn.packed_qkv.weight for layer in layers],
            [layer.self_attn.o_proj.weight for layer in layers],
            [layer.post_attention_layernorm.weight for layer in layers],
            [layer.mlp.gate_up_proj.weight for layer in layers],
            [layer.mlp.down_proj.weight for layer in layers],
            model.model.norm.weight,
            model.lm_head.weight,
            cos,
            sin,
            list(key_caches),
            list(value_caches),
            cache_capacity,
            cache_position,
            [float(layer.input_layernorm.variance_epsilon) for layer in layers],
            [float(layer.self_attn.scaling) for layer in layers],
        )
        self.prefill_logits = None
        self.prefill_input_ids = None
        self._prefill_cache = None
        return self

    @staticmethod
    def _validate_capture(
        model: nn.Module,
        input_ids: torch.Tensor,
        max_decode_steps: int,
    ) -> None:
        if input_ids.dtype != torch.long or input_ids.ndim != 2:
            raise ValueError("input_ids must be a rank-2 torch.long tensor")
        if input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise ValueError("native full decode requires a non-empty batch of one")
        if input_ids.device != next(model.parameters()).device:
            raise ValueError("input_ids must be on the model device")
        if max_decode_steps < 1:
            raise ValueError("max_decode_steps must be positive")
        requested = input_ids.shape[1] + max_decode_steps
        NativeSmolLM2Decode._validate_capture_model_input(model, input_ids, requested)

    @staticmethod
    def _validate_capture_model_input(
        model: nn.Module,
        input_ids: torch.Tensor,
        requested: int,
    ) -> None:
        if input_ids.dtype != torch.long or input_ids.ndim != 2:
            raise ValueError("input_ids must be a rank-2 torch.long tensor")
        if input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise ValueError("native SmolLM2 runtime requires a non-empty batch of one")
        if not input_ids.is_contiguous():
            raise ValueError("input_ids must be contiguous")
        if input_ids.device != next(model.parameters()).device:
            raise ValueError("input_ids must be on the model device")
        if requested < input_ids.shape[1]:
            raise ValueError("requested capacity must include the complete prompt")
        if requested > int(model.config.max_position_embeddings):
            raise ValueError("requested native cache exceeds the model position limit")
        # Reuse the established model/configuration validation with empty
        # compact cache sources and a valid placeholder token.
        empty = torch.empty(
            (1, 3, 0, 64), dtype=torch.float32, device=input_ids.device
        )
        NativeSmolLM2Decode._validate_model_and_state(
            model,
            input_ids[:, :1],
            [empty] * 30,
            [empty] * 30,
            cache_position=0,
            cache_capacity=max(requested, 1),
        )

    @staticmethod
    def _decode_workspace_bytes(capacity: int) -> int:
        chunks = (capacity + 127) // 128
        values = (
            4 * 576
            + 960
            + 2 * 9 * 64
            + 576
            + 1536
            + 576
            + 2 * 64
            + 9 * chunks * (64 + 2)
        )
        return values * 4

    @staticmethod
    def _validate_model_and_state(
        model: nn.Module,
        initial_token: torch.Tensor,
        key_caches: Sequence[torch.Tensor],
        value_caches: Sequence[torch.Tensor],
        *,
        cache_position: int,
        cache_capacity: int,
    ) -> None:
        from flux.model.smollm2_flux import (
            FINAL_FLUX_OPERATOR_CATEGORIES,
            FluxLlamaDecoderLayer,
            FluxPackedLlamaMLP,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("Flux native full decode requires CUDA")
        if model.training:
            raise ValueError("Flux native full decode requires model.eval()")
        if set(getattr(model, "_flux_operator_categories", ())) != set(
            FINAL_FLUX_OPERATOR_CATEGORIES
        ):
            raise ValueError("native full decode requires the retained Flux configuration")
        config = model.config
        geometry = (
            int(config.num_hidden_layers),
            int(config.hidden_size),
            int(config.intermediate_size),
            int(config.num_attention_heads),
            int(config.num_key_value_heads),
            int(config.head_dim),
        )
        if geometry != (30, 576, 1536, 9, 3, 64):
            raise ValueError(
                "native full decode requires SmolLM2-135M geometry "
                "(30, 576, 1536, 9, 3, 64)"
            )
        if len(model.model.layers) != 30 or not all(
            isinstance(layer, FluxLlamaDecoderLayer)
            and isinstance(layer.mlp, FluxPackedLlamaMLP)
            for layer in model.model.layers
        ):
            raise TypeError("model does not contain 30 fully enabled Flux layers")
        if next(model.parameters()).dtype != torch.float32:
            raise TypeError("native full decode currently requires FP32 weights")
        norm_epsilons = {float(model.model.norm.variance_epsilon)}
        for layer in model.model.layers:
            norm_epsilons.add(float(layer.input_layernorm.variance_epsilon))
            norm_epsilons.add(float(layer.post_attention_layernorm.variance_epsilon))
        if len(norm_epsilons) != 1:
            raise ValueError("native full decode requires one shared RMSNorm epsilon")
        if cache_capacity < 1 or cache_capacity > 8192:
            raise ValueError("cache_capacity must be in [1, 8192]")
        if cache_capacity > int(config.max_position_embeddings):
            raise ValueError("cache_capacity exceeds the model position limit")
        if cache_position < 0 or cache_position >= cache_capacity:
            raise ValueError("cache_position must be inside cache_capacity")
        device = next(model.parameters()).device
        if (
            initial_token.shape != (1, 1)
            or initial_token.dtype != torch.long
            or initial_token.device != device
            or not initial_token.is_contiguous()
        ):
            raise ValueError("initial_token must be contiguous CUDA long [1, 1]")
        if len(key_caches) != 30 or len(value_caches) != 30:
            raise ValueError("one K/V source is required for each of 30 layers")
        for name, tensors in (("key", key_caches), ("value", value_caches)):
            for index, tensor in enumerate(tensors):
                if (
                    tensor.shape[:2] != (1, 3)
                    or tensor.ndim != 4
                    or tensor.shape[2] < cache_position
                    or tensor.shape[3] != 64
                    or tensor.dtype != torch.float32
                    or tensor.device != device
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(f"{name} cache layer {index} is unsupported")
        if any(key.shape != value.shape for key, value in zip(key_caches, value_caches)):
            raise ValueError("each layer's K and V source shapes must match")

    def replay(self, token: torch.Tensor | None = None) -> torch.Tensor:
        """Asynchronously enqueue an optional token copy and one graph launch."""
        return self._native.replay(token)

    def reset(
        self,
        token: torch.Tensor,
        key_caches: Sequence[torch.Tensor],
        value_caches: Sequence[torch.Tensor],
        *,
        cache_position: int,
    ) -> None:
        """Synchronously restore prefilled state outside steady replay."""
        self._native.reset(
            token, list(key_caches), list(value_caches), cache_position
        )

    @property
    def logits(self) -> torch.Tensor:
        return self._native.logits()

    @property
    def key_cache(self) -> torch.Tensor:
        return self._native.key_cache()

    @property
    def value_cache(self) -> torch.Tensor:
        return self._native.value_cache()

    @property
    def device_position(self) -> torch.Tensor:
        return self._native.device_position()

    @property
    def device_cache_length(self) -> torch.Tensor:
        return self._native.device_cache_length()

    @property
    def cache_position(self) -> int:
        return int(self._native.position())

    @property
    def cache_length(self) -> int:
        return int(self._native.cache_length())

    @property
    def capacity(self) -> int:
        return int(self._native.capacity())

    @property
    def replay_count(self) -> int:
        return int(self._native.replay_count())

    @property
    def memory(self) -> NativeDecodeRuntimeMemory:
        referenced = sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.model.parameters()
        )
        return NativeDecodeRuntimeMemory(
            cache_bytes=(self.key_cache.numel() + self.value_cache.numel())
            * self.key_cache.element_size(),
            workspace_bytes=int(self._native.workspace_bytes()),
            stable_buffer_bytes=int(self._native.stable_buffer_bytes()),
            referenced_weight_bytes=referenced,
        )

    def stable_addresses(self) -> dict[str, int]:
        return dict(zip(_ADDRESS_NAMES, self._native.addresses(), strict=True))

    def native_object(self) -> Any:
        return self._native


def native_smollm2_greedy_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
) -> torch.Tensor:
    """Generate greedily with native prefill and full-model decode replay."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    runtime = NativeSmolLM2Prefill.capture(
        model, input_ids, max_decode_steps=max_new_tokens - 1
    )
    token = runtime.logits.argmax(dim=-1)
    generated = [input_ids, token]
    for _ in range(max_new_tokens - 1):
        token = runtime.replay(token).argmax(dim=-1)
        generated.append(token)
    return torch.cat(generated, dim=-1)


__all__ = [
    "NativeDecodeRuntimeMemory",
    "NativePrefillRuntimeMemory",
    "NativeSmolLM2Decode",
    "NativeSmolLM2Prefill",
    "native_smollm2_greedy_generate",
    "native_smollm2_prefill_is_available",
    "native_smollm2_runtime_is_available",
]
