"""Compare reference and Flux-integrated SmolLM2-135M FP32 inference."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable

import torch

from flux.model.smollm2 import MODEL_ID, MODEL_REVISION, load_model, load_tokenizer
from flux.model.smollm2_flux import (
    FINAL_FLUX_OPERATOR_CATEGORIES,
    enable_flux_ops,
    flux_operator_counts,
)
from flux.runtime import native_smollm2_greedy_generate


def _latency_ms(
    operation: Callable[[], object],
    device: torch.device,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(iterations):
        operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1_000 / iterations


def _generation_kwargs(tokenizer: object, max_new_tokens: int) -> dict[str, object]:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id")
    return {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": pad_token_id,
        "use_cache": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="The future of artificial intelligence is")
    parser.add_argument("--device", default=None, help="Default: CUDA if available, else CPU")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--native",
        action="store_true",
        help="also generate through native prefill and CUDA-Graph decode",
    )
    args = parser.parse_args()
    if not args.prompt.strip():
        parser.error("--prompt must not be empty or whitespace")
    if (
        args.max_new_tokens < 1
        or args.warmup < 0
        or args.iterations < 1
    ):
        parser.error("token and timing counts must be positive (warmup may be zero)")

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = load_tokenizer()
    model = load_model(device)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    generation_kwargs = _generation_kwargs(tokenizer, args.max_new_tokens)

    def full_forward() -> torch.Tensor:
        return model(**inputs, use_cache=False).logits

    with torch.inference_mode():
        reference_output = model(
            **inputs,
            use_cache=False,
            output_hidden_states=True,
        )
        reference_logits = reference_output.logits.detach().cpu()
        reference_layer_one = reference_output.hidden_states[1].detach().cpu()
        reference_tokens = model.generate(
            **inputs,
            **generation_kwargs,
        ).detach().cpu()
        reference_ms = _latency_ms(
            full_forward,
            device,
            args.warmup,
            args.iterations,
        )

        enable_flux_ops(model, operators=FINAL_FLUX_OPERATOR_CATEGORIES)
        flux_output = model(
            **inputs,
            use_cache=False,
            output_hidden_states=True,
        )
        flux_logits = flux_output.logits.detach().cpu()
        flux_layer_one = flux_output.hidden_states[1].detach().cpu()
        flux_tokens = model.generate(
            **inputs,
            **generation_kwargs,
        ).detach().cpu()
        native_tokens = None
        if args.native:
            if device.type != "cuda":
                parser.error("--native requires a CUDA device")
            native_tokens = native_smollm2_greedy_generate(
                model,
                inputs["input_ids"],
                max_new_tokens=args.max_new_tokens,
            ).detach().cpu()
        flux_ms = _latency_ms(
            full_forward,
            device,
            args.warmup,
            args.iterations,
        )

    absolute = (flux_logits - reference_logits).abs()
    relative = absolute / reference_logits.abs().clamp_min(1e-7)
    layer_one_absolute = (flux_layer_one - reference_layer_one).abs()

    print(f"Model: {MODEL_ID}")
    print(f"Revision: {MODEL_REVISION}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print("Dtype: float32")
    print("Reference attention: eager")
    print("Flux attention: explicit eager with flux::softmax")
    print(f"Input shape: {tuple(inputs['input_ids'].shape)}")
    print(f"Flux modules: {flux_operator_counts(model)}")
    print(
        "Decoder layer 1 max absolute difference: "
        f"{layer_one_absolute.max().item():.9g}"
    )
    print(f"Logits max absolute difference: {absolute.max().item():.9g}")
    print(f"Logits mean absolute difference: {absolute.mean().item():.9g}")
    print(f"Logits max relative difference: {relative.max().item():.9g}")
    print(f"Greedy token IDs equal: {torch.equal(reference_tokens, flux_tokens)}")
    print(f"Reference token IDs: {reference_tokens[0].tolist()}")
    print(f"Flux token IDs: {flux_tokens[0].tolist()}")
    if native_tokens is not None:
        print(f"Native greedy token IDs equal: {torch.equal(native_tokens, flux_tokens)}")
        print(f"Native token IDs: {native_tokens[0].tolist()}")
    print(
        f"Reference full-forward latency ({args.iterations} iterations): "
        f"{reference_ms:.3f} ms"
    )
    print(
        f"Flux full-forward latency ({args.iterations} iterations): "
        f"{flux_ms:.3f} ms"
    )


if __name__ == "__main__":
    main()
