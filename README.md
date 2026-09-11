# Flux

Flux is a long-term systems and machine-learning project for building a CUDA-accelerated transformer inference system around **SmolLM2-135M**. It uses Python, C++, CUDA C++, and PyTorch.

Development began with reproducible PyTorch reference inference and progressively replaces important transformer operations with custom native and CUDA implementations. FP32 RMSNorm, fused residual + RMSNorm, RoPE, attention softmax, and fused attention score post-processing now have validated native PyTorch operators with CPU and CUDA dispatch. An optional integrated SmolLM2 path uses those operators while retaining Hugging Face projections, grouped-query attention, MLPs, and KV-cache management.

The reference model remains unchanged as the numerical oracle. Call `enable_flux_ops(model)` explicitly on an evaluated FP32 `LlamaForCausalLM` to replace supported modules on that model instance; importing Flux never mutates a Hugging Face model or global Transformers behavior.

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

The RMSNorm stack is correctness-complete, and its CUDA kernel uses warp
shuffles plus a small shared-memory reduction across warp partials. With the
native extension built, benchmark it against PyTorch's native RMSNorm from the
repository root:

```bash
python benchmarks/benchmark_rmsnorm.py
```

An aligned `float4` memory-access path was evaluated as the final standalone
RMSNorm optimization, but was not retained. Across three repeated 5,000-call
runs on an otherwise idle RTX 5070 Ti, the `(1, 8192, 576)` median increased
from 29.502 microseconds for scalar warp reduction to 37.166 microseconds with
vectorized access. A 128-thread vector variant remained slower at 38.106
microseconds. The production kernel therefore retains scalar memory access for
all hidden sizes.

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

## Flux-integrated inference

With the native extension built, compare the pinned reference model with the
integrated path on identical FP32 inputs:

```bash
python scripts/flux_inference.py
```

The script reports decoder-layer and full-logit differences, greedy token-ID
equality with KV caching, installed Flux module counts, and a basic full-forward
latency sanity check. The Flux attention path remains explicit eager attention.
Its fused Q/K RoPE operator consumes the exact position-dependent cosine and
sine tensors produced by Hugging Face, preserving position ids, offsets, and
RoPE scaling. It reads non-contiguous projection views directly and supports
SmolLM2's grouped-query head counts. Multi-token masked attention fuses score
scaling, additive masking, and
softmax after the QK matmul. Pass `fuse_attention_scores=False` to
`enable_flux_ops` to retain the previous separate sequence for comparison. QK
and P@V matmuls remain outside the operator; this is not a FlashAttention-style
or fused-SDPA implementation.

Validate the pinned model and benchmark RoPE in isolation and in integrated
prefill/cached-decode paths with:

```bash
python scripts/validate_rope_model.py
python benchmarks/benchmark_rope.py
```

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

The standalone CUDA kernel uses one 256-thread block per row, scalar FP32 memory
access, a two-level FP32 warp reduction, and a caller-provided CUDA stream. For
an RTX 5070 Ti (compute
capability 12.0), build and validate it from the same developer prompt with:

```bat
nvcc -std=c++17 -arch=sm_120 -Xcompiler=/W4 -Xcompiler=/EHsc -I csrc\rmsnorm csrc\rmsnorm\rmsnorm_cuda.cu csrc\rmsnorm\test_rmsnorm_cuda.cu -o build\rmsnorm\test_rmsnorm_cuda.exe
nvcc -std=c++17 -arch=sm_120 -Xcompiler=/W4 -Xcompiler=/EHsc -I csrc\rmsnorm csrc\rmsnorm\rmsnorm_cuda.cu csrc\rmsnorm\rmsnorm_cuda_cli.cu -o build\rmsnorm\rmsnorm_cuda_cli.exe
build\rmsnorm\test_rmsnorm_cuda.exe
python scripts\check_rmsnorm_cuda.py build\rmsnorm\rmsnorm_cuda_cli.exe
```
