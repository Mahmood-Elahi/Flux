"""Offline tests: no model weights or tokenizer files are downloaded."""

import os
import subprocess
import sys
from unittest.mock import Mock

import torch
from transformers import LlamaConfig, PretrainedConfig

from flux.model import smollm2


def test_model_identity_and_helpers() -> None:
    assert smollm2.MODEL_ID == "HuggingFaceTB/SmolLM2-135M"
    assert len(smollm2.MODEL_REVISION) == 40
    int(smollm2.MODEL_REVISION, 16)
    assert callable(smollm2.load_tokenizer)
    assert callable(smollm2.load_model)
    assert callable(smollm2.inspect_config)


def test_import_does_not_load_pretrained_assets() -> None:
    # Use a fresh process so a previous test's import cannot hide side effects.
    code = """
from unittest.mock import patch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

with patch.object(AutoConfig, 'from_pretrained') as config, \
     patch.object(AutoModelForCausalLM, 'from_pretrained') as model, \
     patch.object(AutoTokenizer, 'from_pretrained') as tokenizer:
    import flux.model.smollm2
    config.assert_not_called()
    model.assert_not_called()
    tokenizer.assert_not_called()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_inspect_synthetic_config() -> None:
    config = LlamaConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        vocab_size=128,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        hidden_act="silu",
    )
    assert smollm2.inspect_config(config) == {
        "model_type": "llama",
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 64,
        "vocab_size": 128,
        "max_position_embeddings": 256,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
    }


def test_inspect_missing_fields() -> None:
    result = smollm2.inspect_config(PretrainedConfig())
    assert result["hidden_size"] is None
    assert result["num_key_value_heads"] is None
    assert result["rms_norm_eps"] is None


def test_load_tokenizer_uses_pinned_model(monkeypatch) -> None:
    loader = Mock(return_value=object())
    monkeypatch.setattr(smollm2.AutoTokenizer, "from_pretrained", loader)
    assert smollm2.load_tokenizer() is loader.return_value
    loader.assert_called_once_with(smollm2.MODEL_ID, revision=smollm2.MODEL_REVISION)


def test_load_model_uses_fp32_eager_and_eval(monkeypatch) -> None:
    model = torch.nn.Linear(2, 2)
    loader = Mock(return_value=model)
    monkeypatch.setattr(smollm2.AutoModelForCausalLM, "from_pretrained", loader)
    loaded = smollm2.load_model(torch.device("cpu"))
    loader.assert_called_once_with(
        smollm2.MODEL_ID,
        revision=smollm2.MODEL_REVISION,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    assert loaded is model
    assert not loaded.training
    assert loaded.weight.device == torch.device("cpu")
