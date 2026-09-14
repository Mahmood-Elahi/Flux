# Flux

Flux is a completed, integrated FP32 CUDA inference path for
**SmolLM2-135M**. It combines explicit PyTorch model adapters with custom
C++/CUDA operators, packed checkpoint-compatible projections, native
one-token grouped-query attention, and fixed-shape CUDA-Graph decode. The
ordinary pinned Hugging Face/PyTorch model remains unchanged as the numerical
and performance reference; Flux is enabled only on the model instance passed
to `enable_flux_ops` and never monkey-patches Transformers globally.

## Final integrated system

The canonical production category set is exported as
`FINAL_FLUX_OPERATOR_CATEGORIES`:

```python
from flux.model import FINAL_FLUX_OPERATOR_CATEGORIES, enable_flux_ops

model = enable_flux_ops(
    model,
    operators=FINAL_FLUX_OPERATOR_CATEGORIES,
    fuse_attention_scores=True,
)
```

It retains FP32 RMSNorm, fused residual-RMSNorm, RoPE, fused attention score
processing/softmax, packed QKV and MLP storage, packed SwiGLU, native one-token
GQA, fused packed-QKV/RoPE/StaticCache update, tuned zero-workspace cuBLASLt
QKV/output projections, and fused gate/up GEMV+SwiGLU. Fixed-shape graph decode
uses stable caller-owned buffers, device-resident cache position/mask state, and
unexpanded 3-head K/V storage; it never materializes `repeat_kv` in the
optimized 513--8192-capacity path.

On the target RTX 5070 Ti, the final 30-sample run measured 1.435, 1.464,
1.707, and 2.053 ms/token at effective attention lengths 1024, 2048, 4096,
and 8192. These are 16.82x, 16.12x, 13.62x, and 10.92x faster than the ordinary
Hugging Face/PyTorch reference path. A 4096-capacity replay contains 315 GPU
launches (181 Flux, 92 cuBLAS/cuBLASLt, 42 remaining framework), grows PyTorch
allocated memory by zero bytes, and preserves all graph tensor addresses.
Reference, Flux eager, and Flux graph greedy tokens matched through every
tested continuation; the largest observed end-to-end logit difference was
`8.965e-5` under the established FP32 tolerance.

Rebuild, validate, and reproduce the final matrix with:

```powershell
$env:FLUX_BUILD_NATIVE='1'
build\python3119\python.exe setup.py build_ext --inplace --parallel 8
build\python3119\python.exe -m pytest -q
build\python3119\python.exe benchmarks\benchmark_final_system.py `
  --json-output build\final_system_results.json
```

The benchmark covers 128--8192-token prefill, steady one-token decode,
32-token generation, exact state-dict compatibility, logits/KV/cache state,
stable addresses, and replay launch/allocation behavior. See
[the final system milestone](docs/FINAL_SYSTEM_MILESTONE.md) for the complete
configuration, methodology, tables, bottleneck interpretation, limitations,
and optimization history.

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

The packed MLP remains separately opt-in so the established Flux path is
unchanged. It concatenates each layer's standard gate/up checkpoint weights
once, releases their old runtime storage, and retains the Hugging Face SiLU,
multiply, and down projection:

```python
from flux.model.smollm2_flux import FLUX_OPERATOR_CATEGORIES, enable_flux_ops

enable_flux_ops(model, operators=FLUX_OPERATOR_CATEGORIES | {"mlp"})
```

The independent `"qkv"` category similarly concatenates the bias-free Q, K,
and V weights once in `[Q | K | V]` order and replaces the three projections
with one standard `nn.Linear`/cuBLAS GEMM. For SmolLM2-135M its packed weight is
`[960, 576]`. Allocation-free split, reshape, and transpose views feed the
existing Flux RoPE and cache paths directly; no custom GEMM or RoPE change is
involved:

```python
enable_flux_ops(model, operators=FLUX_OPERATOR_CATEGORIES | {"qkv"})
```

Packed QKV remains independently selectable from packed MLP, packed SwiGLU,
RoPE, softmax, and CUDA-Graph decode. At runtime it is the sole Q/K/V weight
storage, while strict loading, `state_dict()`, and `save_pretrained` retain the
ordinary `q_proj.weight`, `k_proj.weight`, and `v_proj.weight` checkpoint
interface.

Add the separately controlled `"packed_swiglu"` category to replace the
SiLU/multiply pair with `torch.ops.flux.packed_swiglu`. The operator consumes
the original contiguous `[gate; up]` projection result directly, produces a
last dimension half as large, and performs no implicit input copy:

```python
enable_flux_ops(
    model,
    operators=FLUX_OPERATOR_CATEGORIES | {"mlp", "packed_swiglu"},
)
```

Packed models continue to load and export the standard `gate_proj.weight` and
`up_proj.weight` state-dict keys. Compare isolated projection/MLP latency,
integrated prefill, cached decode, numerical error, and parameter storage with:

```bash
python -m benchmarks.benchmark_smollm2_mlp
```

The independent `"gqa_decode_attention"` category replaces the one-token
cached-decode sequence—K/V repetition, QK, scaling/mask/softmax, and P@V—with
a native operator that reads unexpanded `[B, KVH, capacity, D]` cache storage.
The initial native contract is FP32, batch size one, and query length one;
unsupported inputs retain the existing attention path. DynamicCache decode
uses the fused path at every supported length. StaticCache
CUDA-Graph decode uses it above the measured 512-token capacity crossover and
retains the existing shared-memory path through 512. Prefill is unchanged. The
FP32 CUDA head-dimension-64 path uses one shared-score block through 512 tokens and
128-token online-softmax partials plus a max-rescaled reduction beyond 512.
Capacities through 4096 use query-head partials; larger SmolLM2 capacities use
KV-head-grouped partials that reuse each K/V load across three query heads;
other supported head dimensions use the general single-block kernel.

```python
enable_flux_ops(
    model,
    operators=FLUX_OPERATOR_CATEGORIES | {"gqa_decode_attention"},
)
```

Validate and benchmark isolated attention, a decoder layer, eager decode,
CUDA-Graph replay, numerical behavior, kernel inventory, and memory with:

```bash
python benchmarks/benchmark_smollm2_gqa_decode_attention.py
```

The separately controlled `"packed_qkv_rope_cache"` category fuses the work
between the retained packed QKV projection and native one-token GQA attention.
For the FP32 SmolLM2 `B=1`, `Q=1` StaticCache path, one CUDA block applies Q/K
RoPE, writes rotated K and unmodified V directly into unexpanded cache storage,
returns compact head-major Q, and advances the device-resident cache length.
It is CUDA-Graph safe and removes the separate RoPE/cache-update sequence
without a workspace or graph-pool increase. It requires `"rope"`, `"qkv"`, and
`"gqa_decode_attention"`; unsupported inputs use the retained path. As with
native GQA attention, StaticCache capacities through 512 retain the measured
short-context fallback. The specialized fused path supports capacities 513
through 8192.

```python
enable_flux_ops(
    model,
    operators=FLUX_OPERATOR_CATEGORIES
    | {"qkv", "gqa_decode_attention", "packed_qkv_rope_cache"},
)
```

Validate and benchmark the fused boundary, including isolated, layer, eager,
and CUDA-Graph measurements plus cache, generation, launch, and memory checks:

```bash
python benchmarks/benchmark_smollm2_packed_qkv_rope_cache.py
```

For the fully fused FP32 `B=1`, one-token StaticCache graph path at capacities
513 through 8192, graph capture also preallocates a small state-owned scratch
set. Narrow internal out variants reuse it for RMSNorm, residual RMSNorm,
packed SwiGLU, packed-QKV post-processing, and native GQA attention. The
buffers are shared only where producer/consumer lifetimes do not overlap,
remain owned by the captured runtime object, and are not used by eager or
unsupported paths. PyTorch deterministic algorithms and uninitialized-memory
safety filling remain enabled; removing the captured `empty` allocations
removes their redundant replay fills. Profile the fill attribution, launch
breakdown, correctness, latency, and graph-pool behavior with:

```bash
python benchmarks/benchmark_smollm2_stable_buffers.py
```

The separately controlled `"cublaslt_projection"` category replaces only the
FP32 one-token packed-QKV and attention-output projections inside the supported
stable-buffer CUDA-Graph path. It uses a measured zero-workspace cuBLASLt
configuration with caller-owned outputs; eager, short-cache, and unsupported
paths retain `nn.Linear`. It requires `"packed_qkv_rope_cache"` and that
category's dependencies.

```python
enable_flux_ops(
    model,
    operators=FLUX_OPERATOR_CATEGORIES
    | {
        "mlp",
        "packed_swiglu",
        "qkv",
        "gqa_decode_attention",
        "packed_qkv_rope_cache",
        "cublaslt_projection",
        "fused_gate_up_swiglu",
    },
)
```

Reproduce the shape inventory, PyTorch API comparison, bounded cuBLASLt search,
category ablation, decoder-layer timing, full CUDA-Graph sweep, eager fallback,
correctness checks, and independent-process memory accounting with:

```bash
python benchmarks/benchmark_smollm2_projections.py --production --independent-memory --eager-production
```

See [the projection milestone report](docs/PROJECTION_MILESTONE.md) for the
retention evidence and rejected candidates.

The separately controlled `"fused_gate_up_swiglu"` category owns only the
FP32 SmolLM2 one-token packed gate/up shape `[1,1,576] @ [3072,576].T`. In the
supported fixed-shape CUDA-Graph path it computes matched gate/up dot products,
applies SwiGLU, and writes directly to the existing stable 1,536-element MLP
activation buffer. Eager, prefill, non-FP32, non-unit batch/query, and other
geometry use the existing packed `nn.Linear` plus packed-SwiGLU fallback.

Add `"fused_gate_up_swiglu"` to the fully optimized category set shown above.
Reproduce the retained benchmark with:

```bash
python benchmarks/benchmark_smollm2_gate_up_gemv.py
```

See [the gate/up GEMV milestone report](docs/GATE_UP_GEMV_MILESTONE.md) for the
bounded standalone and fused experiments, correctness, launch, traffic, and
memory results.

The subsequent retained-system re-profile found the LM head to be the largest
individual remaining library kernel. A bounded FP32 M=1 custom GEMV experiment
produced only a marginal isolated win and regressed integrated graph decode at
the primary and longer capacities, so no LM-head operator or dispatch was
retained. See [the LM-head GEMV milestone report](docs/LM_HEAD_GEMV_MILESTONE.md)
for the fresh category profile, rejected architectures, correctness, traffic,
resource, graph, eager, and memory results.

The follow-up long-context GQA investigation retained shared per-chunk
rescaling in the final reduction, reducing redundant workspace reads and
rescaling exponentials without changing launches or operator semantics. A
coalesced stage-1 candidate was removed after its isolated 4096 win regressed
the integrated graph. Reproduce the retained kernel/stage benchmark with:

```bash
python benchmarks/benchmark_gqa_long_context.py
```

See [the GQA reduction milestone report](docs/GQA_REDUCTION_MILESTONE.md) for
the baseline structure, traffic/resource analysis, alternating full-graph A/B
results, rejected experiment, correctness, and recommendation.

Validate and benchmark the retained production packed-QKV path, including
projection and attention-setup latency, prefill, eager and CUDA-Graph decode,
view layouts, numerical error, and memory lifetime, with:

```bash
python benchmarks/benchmark_smollm2_qkv.py
```

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
