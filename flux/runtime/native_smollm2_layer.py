"""Narrow Python adapter for the native one-layer SmolLM2 decode runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from flux.ops.native_rmsnorm import _try_load_native_library


_ADDRESS_NAMES = (
    "input_hidden",
    "output",
    "key_cache",
    "value_cache",
    "device_position",
    "workspace",
    "norm_output",
    "residual_output",
    "qkv_output",
    "query_output",
    "attention_output",
    "attention_projection_output",
    "swiglu_output",
    "down_projection_output",
    "attention_workspace",
)


@dataclass(frozen=True)
class NativeLayerRuntimeMemory:
    """Persistent tensor bytes owned by one native layer runtime."""

    cache_bytes: int
    workspace_bytes: int
    input_output_state_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.cache_bytes + self.workspace_bytes + self.input_output_state_bytes


def native_smollm2_layer_runtime_is_available() -> bool:
    """Return whether the native lifetime-managed layer class is registered."""
    _try_load_native_library()
    try:
        getattr(torch.classes.flux, "NativeSmolLM2LayerDecode")
    except RuntimeError:
        return False
    return True


class NativeSmolLM2LayerDecode:
    """Python-owned model references around a native one-layer graph runtime.

    Python supplies one decoder layer's retained weights, a layer-boundary
    hidden state, and the prefilled K/V prefix. The native object allocates and
    owns the fixed-capacity cache, stable input/output/state tensors, scratch
    workspace, capture stream, CUDA Graph, and graph executable.
    """

    def __init__(self) -> None:
        raise TypeError("use NativeSmolLM2LayerDecode.capture(...) to create a runtime")

    @classmethod
    def capture(
        cls,
        model: nn.Module,
        hidden_state: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        *,
        cache_position: int,
        cache_capacity: int,
        layer_index: int = 0,
    ) -> NativeSmolLM2LayerDecode:
        """Capture one fixed-shape decoder layer from an enabled Flux model."""
        cls._validate_model_and_state(
            model,
            hidden_state,
            key_cache,
            value_cache,
            cache_position=cache_position,
            cache_capacity=cache_capacity,
            layer_index=layer_index,
        )
        _try_load_native_library()
        if not native_smollm2_layer_runtime_is_available():
            raise RuntimeError(
                "Flux native one-layer decode runtime is not built. Set "
                "FLUX_BUILD_NATIVE=1 and rebuild the extension in place."
            )

        layer = model.model.layers[layer_index]
        attention = layer.self_attn
        mlp = layer.mlp
        positions = torch.arange(
            cache_capacity, dtype=torch.long, device=hidden_state.device
        ).unsqueeze(0)
        with torch.inference_mode():
            cos, sin = model.model.rotary_emb(hidden_state, positions)
        cos = cos.reshape(cache_capacity, 64).contiguous()
        sin = sin.reshape(cache_capacity, 64).contiguous()

        self = object.__new__(cls)
        self.model = model
        self.layer_index = layer_index
        self._native = torch.classes.flux.NativeSmolLM2LayerDecode(
            hidden_state,
            layer.input_layernorm.weight,
            attention.packed_qkv.weight,
            attention.o_proj.weight,
            layer.post_attention_layernorm.weight,
            mlp.gate_up_proj.weight,
            mlp.down_proj.weight,
            cos,
            sin,
            key_cache,
            value_cache,
            cache_capacity,
            cache_position,
            float(layer.input_layernorm.variance_epsilon),
            float(attention.scaling),
        )
        self._rope_cos = cos
        self._rope_sin = sin
        return self

    @staticmethod
    def _validate_model_and_state(
        model: nn.Module,
        hidden_state: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        *,
        cache_position: int,
        cache_capacity: int,
        layer_index: int,
    ) -> None:
        from flux.model.smollm2_flux import (
            FINAL_FLUX_OPERATOR_CATEGORIES,
            FluxLlamaDecoderLayer,
            FluxPackedLlamaMLP,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("Flux native one-layer decode requires CUDA")
        if model.training:
            raise ValueError("Flux native one-layer decode requires model.eval()")
        selected = set(getattr(model, "_flux_operator_categories", ()))
        if selected != set(FINAL_FLUX_OPERATOR_CATEGORIES):
            raise ValueError(
                "native one-layer decode requires the complete retained Flux "
                "operator configuration"
            )
        if layer_index < 0 or layer_index >= len(model.model.layers):
            raise ValueError("layer_index is outside the model's decoder layers")
        layer = model.model.layers[layer_index]
        if not isinstance(layer, FluxLlamaDecoderLayer) or not isinstance(
            layer.mlp, FluxPackedLlamaMLP
        ):
            raise TypeError("selected layer is not a fully enabled Flux decoder layer")
        config = model.config
        geometry = (
            int(config.hidden_size),
            int(config.intermediate_size),
            int(config.num_attention_heads),
            int(config.num_key_value_heads),
            int(config.head_dim),
        )
        if geometry != (576, 1536, 9, 3, 64):
            raise ValueError(
                "native one-layer decode currently requires SmolLM2 geometry "
                "(576, 1536, 9, 3, 64)"
            )
        if cache_capacity < 1 or cache_capacity > 8192:
            raise ValueError("cache_capacity must be in [1, 8192]")
        if cache_capacity > int(config.max_position_embeddings):
            raise ValueError("cache_capacity exceeds the model position limit")
        if cache_position < 0 or cache_position >= cache_capacity:
            raise ValueError("cache_position must be inside cache_capacity")
        expected_device = next(model.parameters()).device
        expected = {
            "hidden_state": ((1, 1, 576), hidden_state),
            "key_cache": ((1, 3, None, 64), key_cache),
            "value_cache": ((1, 3, None, 64), value_cache),
        }
        for name, (shape, tensor) in expected.items():
            if tensor.device != expected_device or tensor.device.type != "cuda":
                raise ValueError(f"{name} must be CUDA on the model device")
            if tensor.dtype != torch.float32 or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous torch.float32")
            if len(shape) != tensor.ndim or any(
                expected_size is not None and actual != expected_size
                for expected_size, actual in zip(shape, tensor.shape, strict=True)
            ):
                raise ValueError(f"{name} has an unsupported shape {tuple(tensor.shape)}")
        if key_cache.shape != value_cache.shape:
            raise ValueError("key_cache and value_cache shapes must match")
        if key_cache.shape[2] < cache_position:
            raise ValueError("prefilled K/V tensors are shorter than cache_position")
        if next(model.parameters()).dtype != torch.float32:
            raise TypeError("native one-layer decode currently requires FP32 weights")
        if layer.input_layernorm.variance_epsilon != (
            layer.post_attention_layernorm.variance_epsilon
        ):
            raise ValueError("layer RMSNorm epsilon values must match")

    def replay(self, hidden_state: torch.Tensor | None = None) -> torch.Tensor:
        """Enqueue an optional D2D input copy and one native graph launch."""
        return self._native.replay(hidden_state)

    def reset(
        self,
        hidden_state: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        *,
        cache_position: int,
    ) -> None:
        """Synchronously restore cache/input/device state outside steady replay."""
        self._native.reset(hidden_state, key_cache, value_cache, cache_position)

    @property
    def output(self) -> torch.Tensor:
        return self._native.output()

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
    def cache_position(self) -> int:
        """Read the device-authoritative position (diagnostic synchronization)."""
        return int(self._native.position())

    @property
    def cache_length(self) -> int:
        """Read the one-layer cache length (diagnostic synchronization)."""
        return int(self._native.cache_length())

    @property
    def capacity(self) -> int:
        return int(self._native.capacity())

    @property
    def replay_count(self) -> int:
        return int(self._native.replay_count())

    @property
    def memory(self) -> NativeLayerRuntimeMemory:
        cache_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.key_cache, self.value_cache)
        )
        state_bytes = (
            self.output.numel() * self.output.element_size()
            + 576 * 4
            + 8
        )
        return NativeLayerRuntimeMemory(
            cache_bytes=cache_bytes,
            workspace_bytes=int(self._native.workspace_bytes()),
            input_output_state_bytes=state_bytes,
        )

    def stable_addresses(self) -> dict[str, int]:
        return dict(zip(_ADDRESS_NAMES, self._native.addresses(), strict=True))

    def native_object(self) -> Any:
        """Return the custom-class object for low-level diagnostics."""
        return self._native


__all__ = [
    "NativeLayerRuntimeMemory",
    "NativeSmolLM2LayerDecode",
    "native_smollm2_layer_runtime_is_available",
]
