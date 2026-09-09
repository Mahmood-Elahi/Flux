"""Hugging Face / PyTorch reference path for SmolLM2-135M.

Importing this module does not load a tokenizer, configuration, or weights.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
# Pin weights and tokenizer together so upstream changes cannot alter a rerun.
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"


def load_tokenizer() -> PreTrainedTokenizerBase:
    """Load the canonical tokenizer, downloading into the HF cache if needed."""
    return AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)


def load_model(device: str | torch.device = "cpu") -> PreTrainedModel:
    """Load the FP32 reference model on the requested device in evaluation mode."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        # This spelling also supports the declared Transformers >=4.46 minimum.
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    return model


def inspect_config(config: PretrainedConfig) -> dict[str, object]:
    """Return operator-relevant configuration fields, or None when absent."""
    fields = (
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "vocab_size",
        "max_position_embeddings",
        "rms_norm_eps",
        "hidden_act",
    )
    return {name: getattr(config, name, None) for name in fields}
