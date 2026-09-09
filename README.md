# Flux

Flux is a long-term systems and machine-learning project for building a CUDA-accelerated transformer inference system around **SmolLM2-135M**. It uses Python, C++, CUDA C++, and PyTorch.

Development begins with reproducible PyTorch reference inference and will progressively replace important transformer operations with custom native and CUDA implementations. The FP32 RMSNorm correctness oracle, standalone native implementations, PyTorch custom operator with CPU and CUDA dispatch, and initial performance benchmark are now implemented. Planned work includes fused residual + RMSNorm, attention softmax, profiling, and integration into the SmolLM2 inference path.

The repository provides a Hugging Face / PyTorch SmolLM2-135M reference inference baseline and FP32 RMSNorm implementations in PyTorch, native C++, and CUDA. The native implementations are exposed as `torch.ops.flux.rmsnorm` through PyTorch's CPU and CUDA dispatch keys. RMSNorm optimization and model-path integration have not yet been performed.

## Development setup

Flux requires Python 3.10 or newer. From the repository root, install the project with its test dependencies:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

### Native PyTorch RMSNorm operator

The optional native library is an explicit development build so PEP 517 build
isolation does not install another PyTorch. From an x64 MSVC Developer Command
Prompt, with the project environment activated, run:

```bat
set DISTUTILS_USE_SDK=1
set FLUX_BUILD_NATIVE=1
python setup.py build_ext --inplace
```

If `cl.exe` is not visible, first initialize the compiler environment:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64
```

Importing `flux.ops` loads a built `flux._C` registration library. The Python
entry point `flux.ops.rms_norm_native(input, weight, epsilon)` invokes
`torch.ops.flux.rmsnorm`. The operator is currently FP32 and inference-only; it
does not implement autograd. Input and weight may be non-contiguous because the
C++ wrappers make contiguous copies before calling the standalone row-major
implementations. No dtype conversion is performed.

The RMSNorm stack is correctness-complete, but its CUDA kernel remains the naive
pre-optimization baseline. With the native extension built, benchmark that
baseline against PyTorch's native RMSNorm from the repository root:

```bash
python benchmarks/benchmark_rmsnorm.py
```

The benchmark checks correctness before timing and prints machine-specific
results to standard output; it does not save them in the repository.

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

The native RMSNorm test can be built directly from an x64 MSVC Developer Command Prompt:

```bat
if not exist build\rmsnorm mkdir build\rmsnorm
cl /nologo /std:c++17 /EHsc /W4 /permissive- /Od /I csrc\rmsnorm csrc\rmsnorm\rmsnorm.cpp csrc\rmsnorm\test_rmsnorm.cpp /Fo:build\rmsnorm\ /Fe:build\rmsnorm\test_rmsnorm.exe
build\rmsnorm\test_rmsnorm.exe
```

The implementation accumulates each sum of squares sequentially in `float`.
PyTorch may reduce in a different order, so cross-language checks use FP32
tolerances rather than requiring bit-for-bit equality.

The standalone CUDA baseline uses one 256-thread block per row, FP32 shared-memory
reduction, and a caller-provided CUDA stream. For an RTX 5070 Ti (compute
capability 12.0), build and validate it from the same developer prompt with:

```bat
nvcc -std=c++17 -arch=sm_120 -Xcompiler=/W4 -Xcompiler=/EHsc -I csrc\rmsnorm csrc\rmsnorm\rmsnorm_cuda.cu csrc\rmsnorm\test_rmsnorm_cuda.cu -o build\rmsnorm\test_rmsnorm_cuda.exe
nvcc -std=c++17 -arch=sm_120 -Xcompiler=/W4 -Xcompiler=/EHsc -I csrc\rmsnorm csrc\rmsnorm\rmsnorm_cuda.cu csrc\rmsnorm\rmsnorm_cuda_cli.cu -o build\rmsnorm\rmsnorm_cuda_cli.exe
build\rmsnorm\test_rmsnorm_cuda.exe
python scripts\check_rmsnorm_cuda.py build\rmsnorm\rmsnorm_cuda_cli.exe
```
