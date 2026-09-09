"""Run the SmolLM2-135M FP32 correctness baseline (after installing Flux)."""

import argparse
import os
import platform

import torch
import transformers

from flux.model.smollm2 import (
    MODEL_ID,
    MODEL_REVISION,
    inspect_config,
    load_model,
    load_tokenizer,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="The future of artificial intelligence is")
    parser.add_argument("--device", default=None, help="Default: CUDA if available, else CPU")
    args = parser.parse_args()
    if not args.prompt.strip():
        parser.error("--prompt must not be empty or whitespace")

    # Configure cuBLAS before any CUDA work. These settings belong to this
    # standalone run, rather than changing process state in the model loader.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Model: {MODEL_ID}")
    print(f"Revision: {MODEL_REVISION}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print("Seed: 0; dtype: float32; attention: eager; deterministic algorithms: enabled")

    tokenizer = load_tokenizer()
    model = load_model(device)
    print("Model configuration:")
    for name, value in inspect_config(model.config).items():
        print(f"  {name}: {value}")

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    print(f"Prompt: {args.prompt!r}")
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False).logits
        final_logits = logits[0, -1].float().cpu()
        print(f"Input shape: {tuple(inputs['input_ids'].shape)}")
        print(f"Logits shape: {tuple(logits.shape)}")
        print(f"Logits dtype: {logits.dtype}")
        print(f"Logits device: {logits.device}")
        print(f"Final-token argmax ID: {final_logits.argmax().item()}")
        print(f"Final-token logits[0:8]: {final_logits[:8].tolist()}")

        generated = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=16,
            pad_token_id=(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            ),
        )
    print(f"Generated text: {tokenizer.decode(generated[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
