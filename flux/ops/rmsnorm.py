"""FP32 PyTorch RMSNorm correctness oracle for future native/CUDA versions."""

import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply RMSNorm across the final (hidden) dimension.

    Unlike LayerNorm, RMSNorm does not subtract the mean. SmolLM2 uses a hidden
    size of 576 and ``eps=1e-5``; both remain explicit inputs here.
    """
    if x.ndim == 0 or x.shape[-1] == 0:
        raise ValueError("x must have a non-empty final dimension")
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError("weight must be one-dimensional and match x.shape[-1]")
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        raise TypeError("x and weight must be float32 tensors")
    if x.device != weight.device:
        raise ValueError("x and weight must be on the same device")
    if eps < 0:
        raise ValueError("eps must be non-negative")

    mean_square = x.pow(2).mean(dim=-1, keepdim=True)
    return weight * x * torch.rsqrt(mean_square + eps)
