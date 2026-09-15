"""Deterministic setup and input helpers for SmolLM2 system benchmarks."""

from __future__ import annotations

import argparse
import os

import torch


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
