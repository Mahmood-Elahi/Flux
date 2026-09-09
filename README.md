# Flux

Flux is a long-term systems and machine-learning project for building a CUDA-accelerated transformer inference system around **SmolLM2-135M**. It uses Python, C++, CUDA C++, and PyTorch.

Development begins with reproducible PyTorch reference inference and will progressively replace important transformer operations with custom native and CUDA implementations. Planned work includes FP32 RMSNorm, fused residual + RMSNorm, attention softmax, PyTorch custom-operator integration, correct current CUDA stream handling, correctness testing, benchmarking, profiling, and integration into the SmolLM2 inference path.

The repository provides a Hugging Face / PyTorch SmolLM2-135M reference inference baseline. Custom operators, custom CUDA kernels, and performance results have not yet been implemented.

## Development setup

Flux requires Python 3.10 or newer. From the repository root, install the project with its test dependencies:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

## Reference inference

After installation, run from the repository root:

```bash
python scripts/reference_inference.py
```

The script selects CUDA when available, otherwise CPU. Optional `--device cpu`
and `--prompt "Your prompt"` overrides are supported. The first run downloads
[HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
into the standard Hugging Face cache, outside Git. The model and tokenizer use
the same pinned revision, defined in `flux/model/smollm2.py`.

The baseline uses float32, evaluation mode, inference mode, eager attention,
seed 0, deterministic PyTorch algorithms, and disabled TF32. It prints runtime
versions, model configuration, tensor metadata, an actual final-token logits
fingerprint, and up to 16 greedily generated tokens (`do_sample=False`). These
settings support repeatability in the same environment; bit-for-bit agreement
is not guaranteed across GPUs, PyTorch/Transformers versions, CUDA versions,
or dtypes. Lightweight tests use synthetic configurations and mocked loading
boundaries and do not download model weights or tokenizers.

See [docs/ROADMAP.md](docs/ROADMAP.md) for the planned progression toward the integrated system.
