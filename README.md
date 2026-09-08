# Flux

Flux is a long-term systems and machine-learning project for building a CUDA-accelerated transformer inference system around **SmolLM2-135M**. It uses Python, C++, CUDA C++, and PyTorch.

Development will begin from reproducible PyTorch reference inference and progressively replace important transformer operations with custom native and CUDA implementations. Planned work includes FP32 RMSNorm, fused residual + RMSNorm, attention softmax, PyTorch custom-operator integration, correct current CUDA stream handling, correctness testing, benchmarking, profiling, and integration into the SmolLM2 inference path.

The repository currently provides the project foundation only. Custom operators, CUDA kernels, model integration, and performance results have not yet been implemented.

## Development setup

Flux requires Python 3.10 or newer. From the repository root, install the project with its test dependencies:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

See [docs/ROADMAP.md](docs/ROADMAP.md) for the planned progression toward the integrated system.
