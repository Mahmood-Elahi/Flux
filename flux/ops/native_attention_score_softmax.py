"""Python entry point for fused native FP32 attention score processing."""

from __future__ import annotations

import torch

from flux.ops.native_rmsnorm import (
    _operator_is_registered,
    _try_load_native_library,
    native_rmsnorm_load_error,
)


_FAKE_REGISTERED = False


def _register_fake() -> None:
    global _FAKE_REGISTERED
    if _operator_is_registered("attention_score_softmax") and not _FAKE_REGISTERED:

        @torch.library.register_fake("flux::attention_score_softmax")
        def _attention_score_softmax_fake(
            scores: torch.Tensor,
            additive_attention_mask: torch.Tensor,
            scale: float,
        ) -> torch.Tensor:
            torch._check(scores.dim() == 4, lambda: "scores must be rank four")
            torch._check(
                additive_attention_mask.dim() == 4,
                lambda: "additive_attention_mask must be rank four",
            )
            torch._check(scores.numel() > 0, lambda: "scores must be non-empty")
            torch._check(scores.dtype == torch.float32, lambda: "scores must be float32")
            torch._check(
                additive_attention_mask.dtype == torch.float32,
                lambda: "additive_attention_mask must be float32",
            )
            torch._check(
                scores.device == additive_attention_mask.device,
                lambda: "tensor devices must match",
            )
            for mask_size, score_size in zip(
                additive_attention_mask.shape, scores.shape, strict=True
            ):
                torch._check(
                    mask_size == 1 or mask_size == score_size,
                    lambda: "additive_attention_mask is not broadcastable to scores",
                )
            torch._check(
                not torch.is_grad_enabled()
                or (
                    not scores.requires_grad
                    and not additive_attention_mask.requires_grad
                ),
                lambda: "flux::attention_score_softmax is inference-only",
            )
            return torch.empty_like(scores, memory_format=torch.contiguous_format)

        _FAKE_REGISTERED = True


def _try_load_native_attention_score_softmax() -> None:
    _try_load_native_library()
    _register_fake()


def native_attention_score_softmax_is_available() -> bool:
    """Return whether the fused compiled operator is loaded."""
    _try_load_native_attention_score_softmax()
    return _operator_is_registered("attention_score_softmax")


def attention_score_softmax_native(
    scores: torch.Tensor,
    additive_attention_mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Apply fused FP32 scale, additive mask, and final-dimension softmax."""
    _try_load_native_attention_score_softmax()
    if not _operator_is_registered("attention_score_softmax"):
        message = (
            "Flux native attention score softmax is not built. Build it with "
            "FLUX_BUILD_NATIVE=1 and `python setup.py build_ext --inplace`."
        )
        error = native_rmsnorm_load_error()
        if error is not None:
            raise RuntimeError(message) from error
        raise RuntimeError(message)
    return torch.ops.flux.attention_score_softmax(
        scores, additive_attention_mask, scale
    )


_try_load_native_attention_score_softmax()


__all__ = [
    "attention_score_softmax_native",
    "native_attention_score_softmax_is_available",
]
