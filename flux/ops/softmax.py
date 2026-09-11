"""FP32 PyTorch softmax correctness reference for future native/CUDA versions."""

import torch


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Apply numerically stable FP32 softmax over the final dimension.

    Max subtraction keeps exponentiation stable. This explicit formulation is
    the correctness reference for future native and CUDA implementations.
    """
    if x.ndim == 0 or x.shape[-1] == 0:
        raise ValueError("x must have a non-empty final dimension")
    if x.dtype != torch.float32:
        raise TypeError("x must be a float32 tensor")

    row_max = x.max(dim=-1, keepdim=True).values
    exp_x = torch.exp(x - row_max)
    row_sum = exp_x.sum(dim=-1, keepdim=True)
    return exp_x / row_sum


__all__ = ["softmax"]
