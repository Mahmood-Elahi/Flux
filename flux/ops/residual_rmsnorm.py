"""PyTorch correctness oracle for dual-output residual RMSNorm."""

import torch

from flux.ops.rmsnorm import rms_norm


def residual_rmsnorm(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add the residual path and return both its raw and normalized values.

    Normalization is over the final dimension. The addition is deliberately
    out-of-place so that neither input is modified and the unnormalized sum is
    preserved for the transformer residual path.
    """
    if hidden.shape != residual.shape:
        raise ValueError("hidden and residual must have the same shape")
    if hidden.dtype != torch.float32 or residual.dtype != torch.float32:
        raise TypeError("hidden and residual must be float32 tensors")
    if hidden.device != residual.device:
        raise ValueError("hidden and residual must be on the same device")

    residual_out = hidden + residual
    norm_out = rms_norm(residual_out, weight, eps)
    return residual_out, norm_out
