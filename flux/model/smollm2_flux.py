"""Optional FP32 Flux custom-operator execution path for SmolLM2.

The adapter replaces modules only on the model instance passed to
``enable_flux_ops``.  It reuses every learned parameter and leaves the
Hugging Face reference loader and global Transformers behavior untouched.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)

from flux.ops import (
    native_residual_rmsnorm_is_available,
    native_rmsnorm_is_available,
    native_softmax_is_available,
    residual_rmsnorm_native,
    rms_norm_native,
    softmax_native,
)


class FluxRMSNorm(LlamaRMSNorm):
    """Llama RMSNorm module backed by the Flux FP32 custom operator."""

    def __init__(self, source: LlamaRMSNorm) -> None:
        # Avoid allocating or initializing replacement parameters. Assigning the
        # original Parameter also preserves state-dict names and weight tying.
        nn.Module.__init__(self)
        self.weight = source.weight
        self.variance_epsilon = source.variance_epsilon

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return rms_norm_native(
            hidden_states,
            self.weight,
            self.variance_epsilon,
        )


class FluxLlamaAttention(LlamaAttention):
    """Llama eager attention with only its final-dimension softmax replaced."""

    def __init__(self, source: LlamaAttention) -> None:
        nn.Module.__init__(self)
        self.config = source.config
        self.layer_idx = source.layer_idx
        self.head_dim = source.head_dim
        self.num_key_value_groups = source.num_key_value_groups
        self.scaling = source.scaling
        self.attention_dropout = source.attention_dropout
        self.is_causal = source.is_causal
        self.q_proj = source.q_proj
        self.k_proj = source.k_proj
        self.v_proj = source.v_proj
        self.o_proj = source.o_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_embeddings is None:
            raise ValueError("position_embeddings must be provided")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
        )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
            )

        # Preserve Llama eager attention exactly around the Flux softmax:
        # grouped-query expansion, scaled QK, additive mask, softmax, then V.
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(
            query_states,
            key_states.transpose(2, 3),
        ) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = softmax_native(attn_weights)
        # Flux integration is inference-only, so dropout is intentionally absent.
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class FluxLlamaDecoderLayer(LlamaDecoderLayer):
    """Llama decoder layer using Flux norm, residual-norm, and softmax ops."""

    def __init__(self, source: LlamaDecoderLayer) -> None:
        nn.Module.__init__(self)
        self.hidden_size = source.hidden_size
        self.self_attn = FluxLlamaAttention(source.self_attn)
        self.mlp = source.mlp
        self.input_layernorm = FluxRMSNorm(source.input_layernorm)
        self.post_attention_layernorm = FluxRMSNorm(
            source.post_attention_layernorm
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_output, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # This is exactly the reference layer's post-attention addition followed
        # by post_attention_layernorm. Both outputs are needed: the normalized
        # value enters the MLP and the unnormalized sum is its residual.
        hidden_states, residual = residual_rmsnorm_native(
            attention_output,
            residual,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.variance_epsilon,
        )
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


def _check_model(model: nn.Module) -> LlamaForCausalLM:
    if not isinstance(model, LlamaForCausalLM):
        raise TypeError("Flux SmolLM2 integration requires LlamaForCausalLM")
    if model.training:
        raise ValueError("Flux custom operators are inference-only; call model.eval() first")
    if getattr(model, "_flux_ops_enabled", False):
        raise ValueError("Flux custom operators are already enabled on this model")

    floating_dtypes = {
        parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()
    }
    if floating_dtypes != {torch.float32}:
        raise TypeError(
            "Flux SmolLM2 integration currently requires all floating-point "
            f"parameters to be torch.float32, found {sorted(map(str, floating_dtypes))}"
        )
    if not all(isinstance(layer, LlamaDecoderLayer) for layer in model.model.layers):
        raise TypeError("model contains a decoder layer unsupported by Flux integration")
    if not isinstance(model.model.norm, LlamaRMSNorm):
        raise TypeError("model final norm is not LlamaRMSNorm")
    return model


def _check_native_ops() -> None:
    missing = []
    if not native_rmsnorm_is_available():
        missing.append("rmsnorm")
    if not native_residual_rmsnorm_is_available():
        missing.append("residual_rmsnorm")
    if not native_softmax_is_available():
        missing.append("softmax")
    if missing:
        raise RuntimeError(
            "Flux native operators are not built: "
            + ", ".join(missing)
            + ". Build the extension before enabling the model path."
        )


def enable_flux_ops(model: nn.Module) -> LlamaForCausalLM:
    """Replace supported modules on an evaluated FP32 Llama causal LM in place.

    The model retains its embeddings, projections, MLPs, RoPE module, causal
    mask construction, LM head, and cache machinery. Learned Parameters are
    reused rather than copied. The returned object is the same model instance.
    """
    llama_model = _check_model(model)
    _check_native_ops()

    llama_model.model.layers = nn.ModuleList(
        [FluxLlamaDecoderLayer(layer) for layer in llama_model.model.layers]
    )
    llama_model.model.norm = FluxRMSNorm(llama_model.model.norm)
    # Transformers installs output-capture hooks lazily on the original layer
    # instances. Force hook discovery to run again after module replacement.
    for module in (llama_model, llama_model.model):
        if hasattr(module, "_output_capturing_hooks_installed"):
            module._output_capturing_hooks_installed = False
    llama_model._flux_ops_enabled = True
    return llama_model


def flux_operator_counts(model: nn.Module) -> dict[str, int]:
    """Return the number of installed Flux modules for diagnostics."""
    return {
        "decoder_layers": sum(
            isinstance(module, FluxLlamaDecoderLayer) for module in model.modules()
        ),
        "rmsnorm_modules": sum(
            isinstance(module, FluxRMSNorm) for module in model.modules()
        ),
        "attention_modules": sum(
            isinstance(module, FluxLlamaAttention) for module in model.modules()
        ),
    }


__all__ = [
    "FluxLlamaAttention",
    "FluxLlamaDecoderLayer",
    "FluxRMSNorm",
    "enable_flux_ops",
    "flux_operator_counts",
]
