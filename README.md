# Flux

[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus\&logoColor=white)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.2-76B900?logo=nvidia\&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?logo=pytorch\&logoColor=white)](https://pytorch.org/)

Flux is a PyTorch integrated CUDA inference optimization engine for [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M). It starts from the standard PyTorch implementation and selectively replaces expensive operations with custom CUDA kernels and native C++ paths to reduce memory traffic, kernel launches, tensor allocations, and runtime overhead. Flux uses compact KV caching, native prefill, CUDA Graph decode, and device resident generation while validating every optimization against the original FP32 model.

[Core Features](#core-features) ·
[Architecture](#architecture) ·
[Metrics](#metrics) ·
[Technical Highlights](#technical-highlights) ·
[Validation](#correctness-and-validation) ·
[Setup](#setup-and-build) ·
[Running Flux](#running-flux) ·
[Future Work](#future-work)

## Core Features

* Custom fused CUDA kernels for RMSNorm, RoPE, attention processing, cache updates, and SwiGLU to reduce kernel launches, intermediate tensors, and memory traffic.
* Checkpoint-compatible packed QKV and MLP paths with direct compact KV-cache updates and no expanded K/V materialization during one-token GQA decode.
* Tiled online-softmax native prefill with bounded long-context workspace, avoiding the large attention-score materialization used by the original path.
* Fixed-shape CUDA Graph decode with stable device addresses and device-resident generation state, removing per-token Python participation.
* Native C++ fixed-length replay for low-latency generation, with greedy token selection and feedback performed directly on the GPU.
* Nsight Systems and Nsight Compute profiling used to analyze register pressure, occupancy, dependency latency, memory behavior, and kernel-launch overhead.


## Architecture

```text
Hugging Face SmolLM2-135M checkpoint
                  |
                  v
        PyTorch / Flux integration
                  |
        +---------+----------------------------+
        |                                      |
        v                                      v
custom C++/CUDA operators                 native prefill
  |-- RMSNorm / residual RMSNorm               |
  |-- packed QKV + RoPE                         v
  |-- compact cache update              compact StaticCache
  |-- one-token GQA attention                   |
  `-- fused gate/up + SwiGLU                    v
                                         CUDA Graph decode
                                                |
                                                v
                                    device-resident greedy state
                                                |
                                                v
                                      native C++ replay loop
                                                |
                                                v
                                         generated tokens
```

`enable_flux_ops` changes only the model instance passed to it; Flux does not globally monkey-patch Transformers. Standard checkpoint keys remain intact while packed runtime layouts remove redundant projections and intermediate tensors.

For the fully native path, prompt token IDs enter a 30-layer FP32 prefill runtime that writes compact K/V directly into the decode runtime's cache. The handoff does not construct a Python cache, expand grouped K/V, or copy cache storage.

Decode then replays a captured token-to-logits CUDA Graph whose buffers, cache position, mask, logits, current token, and generated-token storage remain at stable addresses.

The graph performs exact lowest-index greedy selection and feeds the selected token into the next step. A single C++ call submits the requested fixed number of graph replays, removing Python work from the per-token execution path.

## Metrics

Final measurements used:

* NVIDIA GeForce RTX 5070 Ti (`sm_120`)
* Python 3.11.9
* PyTorch 2.14.0+cu132
* CUDA Toolkit 13.2
* NVIDIA driver 616.64
* Transformers 5.16.1
* Batch size 1

Latencies are CUDA-event median ± median absolute deviation (MAD). Model loading, input construction, correctness checks, runtime capture, and cache setup are outside the timed regions.

| Headline Result                          |                       Result |
| ---------------------------------------- | ---------------------------: |
| Prefill geometric mean, 128–8192 tokens  |  **3.078x** vs. Hugging Face |
| 8192-token prefill                       |  **4.178x** vs. Hugging Face |
| Decode geometric mean, contexts 128–8192 | **14.216x** vs. Hugging Face |
| 128-output generation geometric mean     | **13.185x** vs. Hugging Face |
| Native decode at context 8192            | **1.9772 ± 0.0254 ms/token** |

The decode and generation gains come from combining specialized CUDA kernels, compact KV-cache access, operation fusion, stable device memory, and CUDA Graph replay to reduce framework overhead, intermediate work, and per-token execution cost.

### Prefill

| Tokens |         Hugging Face |        Native Flux | Speedup |
| -----: | -------------------: | -----------------: | ------: |
|    128 |   36.414 ± 11.077 ms |   5.516 ± 0.055 ms |  6.601x |
|    512 |    24.852 ± 0.195 ms |  12.348 ± 0.042 ms |  2.013x |
|  1,024 |    33.630 ± 0.266 ms |  21.651 ± 0.092 ms |  1.553x |
|  2,048 |    94.574 ± 0.160 ms |  38.441 ± 0.172 ms |  2.460x |
|  4,096 |   298.641 ± 0.309 ms |  96.167 ± 0.101 ms |  3.105x |
|  8,192 | 1,075.734 ± 5.932 ms | 257.457 ± 1.870 ms |  4.178x |

The geometric-mean speedup across this representative grid is **3.078x**.

At 8,192 tokens, the retained streaming GQA prefill path avoids the former **2.25 GiB attention-score materialization** by using compact K/V storage and bounded persistent workspace.

### Decode

Each row is a steady one-token decode window ending at the listed effective context. Native Flux uses CUDA Graph replay while Hugging Face uses eager execution.

| Context |              Hugging Face |              Native Flux | Speedup |
| ------: | ------------------------: | -----------------------: | ------: |
|     128 | 25.2768 ± 1.7552 ms/token | 1.4469 ± 0.0156 ms/token | 17.469x |
|     512 | 23.4304 ± 0.2684 ms/token | 2.4040 ± 0.0618 ms/token |  9.747x |
|   1,024 | 24.4220 ± 0.4095 ms/token | 1.4364 ± 0.0124 ms/token | 17.003x |
|   2,048 | 24.0012 ± 0.2364 ms/token | 1.4602 ± 0.0098 ms/token | 16.437x |
|   4,096 | 23.8780 ± 0.2055 ms/token | 1.6791 ± 0.0121 ms/token | 14.220x |
|   8,192 | 24.1153 ± 0.2447 ms/token | 1.9772 ± 0.0254 ms/token | 12.197x |

The geometric-mean decode speedup is **14.216x**.

The 512-token discontinuity comes from the conservative exact-capacity one-CTA GQA path.

At capacity 4,096, one captured replay executes **304 GPU launches** — 183 Flux kernels and 121 cuBLAS/cuBLASLt launches — with **zero framework launches, zero replay allocation growth, and stable device addresses**.

### Generation

Complete generation includes native prefill and all fixed-length decode replays but excludes runtime construction and graph capture. Every timed workload was validated against Hugging Face token-for-token before measurement.

| Prompt | Outputs |          Hugging Face |        Native Flux | Speedup |
| -----: | ------: | --------------------: | -----------------: | ------: |
|    128 |     128 |  3,033.647 ± 5.266 ms | 216.035 ± 0.565 ms | 14.042x |
|    512 |     128 | 3,063.687 ± 11.107 ms | 201.990 ± 0.165 ms | 15.168x |
|  1,024 |     128 |  3,037.912 ± 3.042 ms | 211.669 ± 0.263 ms | 14.352x |
|  2,048 |     128 | 3,182.821 ± 78.909 ms | 233.351 ± 0.836 ms | 13.640x |
|  4,096 |     128 | 3,354.230 ± 50.830 ms | 351.010 ± 8.785 ms |  9.556x |

The geometric-mean speedup for 128 generated tokens is **13.185x**.

Native C++ replay removes Python submission from the per-token path, although its end-to-end gain over Python-controlled native graph replay is modest because GPU execution dominates the final runtime.

### GPU Profiling

Nsight Compute profiling of the long-context streaming prefill attention kernel identified **dependency latency and register pressure** as the primary bottlenecks rather than DRAM bandwidth.

| Metric                                             |                       Result |
| -------------------------------------------------- | ---------------------------: |
| Registers per thread                               |                      **162** |
| Register spills                                    |                        **0** |
| Theoretical occupancy                              |                    **25.0%** |
| Achieved occupancy at 2,048 / 4,096 / 8,192 tokens | **19.57% / 22.33% / 24.34%** |
| Issued warps per scheduler-cycle                   |       **0.54 / 0.61 / 0.65** |
| DRAM bandwidth utilization                         |       **2.31–5.18% of peak** |
| L1/shared LSU utilization at 8,192 tokens          |                   **63.26%** |

The kernel is primarily dependency-latency limited rather than memory-bandwidth limited. A two-lane prototype reduced register usage from **162 to 95 registers/thread** and raised theoretical occupancy from **25.0% to 41.67%**, but regressed performance by **82.7–87.1%**, so it was rejected.

Flux retains optimizations only when they improve measured end-to-end performance, not just isolated GPU metrics.


## Technical Highlights

### Custom CUDA Operators

Flux replaces operations only where the custom contract matches SmolLM2 semantics.

Retained custom paths include normalization, residual fusion, attention-score processing, RoPE, grouped-query attention, cache updates, and fused activation work.

These changes reduce:

* kernel launches
* dispatcher overhead
* global-memory traffic
* intermediate tensor creation
* redundant data movement

Native launchers preserve PyTorch's current CUDA stream semantics.

### Packed QKV and Cache Integration

Q, K, and V weights are packed while preserving standard checkpoint and state-dict keys.

During one-token decode, a fused post-projection path:

1. applies RoPE to Q and K,
2. writes rotated K and V directly into compact cache storage,
3. emits head-major Q,
4. advances graph-resident cache state.

This eliminates separate projection views, standalone cache updates, and expanded K/V storage.

### Native GQA Decode

SmolLM2-135M uses nine query heads and three KV heads.

Flux's one-token grouped-query attention kernel reads directly from compact three-head K/V cache storage instead of materializing repeated K/V tensors for each query head.

The kernel uses stable online softmax and separately measured strategies for short and long contexts.

### CUDA Graph Decode

The native runtime owns:

* `StaticCache`
* lifetime-planned workspaces
* logits/output buffers
* input token state
* cache position
* attention-mask state
* generation state

Fixed shapes and stable addresses allow the complete token-to-logits path to be captured and replayed without allocations.

Stream and event handling preserve asynchronous execution on the caller's current CUDA stream.

### Device-Resident Greedy Generation

Greedy argmax, exact lowest-index tie resolution, generated-token writes, cache position, and next-token feedback remain on the GPU.

The C++ runtime submits repeated CUDA Graph replays in one fixed-length generation call, eliminating Python participation between generated tokens.

Native generation is currently fixed-length and does not inspect EOS for early termination.

### Native Prefill

Prefill is optimized separately from one-token decode.

Its tiled FP32 grouped-query attention computes online-softmax partials over compact K/V storage, merges long-context partitions, and bounds workspace growth.

The retained native path reaches a validated **4.178x speedup at 8,192 tokens** versus Hugging Face.

## GPU Performance Analysis

Flux retained optimizations only after correctness validation and end-to-end measurement. CUDA events, same-process comparisons, Nsight Systems, Nsight Compute, compiler resource reports, and SASS inspection were used to separate framework overhead, kernel-launch cost, register pressure, memory behavior, and actual device execution.

Native C++ graph replay reduced host submission overhead from roughly **24–25 µs/token to 3–4 µs/token**, yet improved end-to-end runtime by less than **1%** because GPU execution already dominated. This confirmed that further host-side optimization offered little practical value.

Nsight Compute showed that long-context streaming prefill was primarily dependency-latency and register-pressure limited rather than DRAM-bandwidth limited.

A two-lane-per-query experiment reduced register usage from **162 to 95 registers/thread** and increased theoretical occupancy, but regressed performance by **82.7–87.1%** at 2,048–8,192 tokens because added softmax, shuffle, and control overhead outweighed the occupancy gain.

The experiment was removed.

Other experimental paths, including vectorized RMSNorm, alternative GQA mappings, and a custom LM-head path, were also rejected when their integrated performance did not justify the added complexity.


## Correctness and Validation

The finalized system passes:

* **146 Python tests**
* **11/11 native CTests**

Validation covers:

* explicit Python/PyTorch operator references
* Hugging Face equivalence
* exact greedy-token equality for every timed generation workload
* checkpoint and state-dict compatibility
* logits comparisons
* every-layer KV-cache comparisons
* FakeTensor/meta behavior
* `torch.library.opcheck`
* non-default CUDA stream correctness
* caller-stream native execution
* stable CUDA Graph addresses
* zero replay-allocation growth
* exact lowest-index greedy argmax tie semantics
* fixed-length native generation
* direct native prefill-to-decode handoff

One deterministic sparse 4,096-token continuation case has two of 49,152 logits outside the unchanged `rtol=2e-4`, `atol=2e-5` comparison policy, with a maximum absolute difference of:

```text
3.3736228942871094e-05
```

Full 4,096-token prefill passes, greedy tokens remain equal, and the tolerance was not loosened to hide this edge case.

## Repository Layout

| Path             | Purpose                                                                  |
| ---------------- | ------------------------------------------------------------------------ |
| `flux/`          | PyTorch operators, SmolLM2 integration, and native runtime adapter       |
| `csrc/`          | Production C++/CUDA kernels, runtimes, native tests, and microbenchmarks |
| `tests/`         | Consolidated Python correctness and integration suite                    |
| `benchmarks/`    | Canonical system benchmark and shared timing infrastructure              |
| `scripts/`       | Inference, native build, testing, and validation entry points            |
| `CMakeLists.txt` | Standalone native test and microbenchmark build                          |
| `setup.py`       | Opt-in PyTorch native extension build                                    |
| `pyproject.toml` | Python package metadata and dependencies                                 |

## Setup and Build

The validated native environment is Windows with:

* Python 3.11.9
* MSVC Build Tools
* CUDA Toolkit 13.2
* NVIDIA GPU supporting the configured `sm_120` target

From the repository root:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Build the PyTorch extension against the active environment:

```powershell
$env:FLUX_BUILD_NATIVE = '1'
python setup.py build_ext --inplace --parallel 8
```

The first inference run downloads the pinned model/tokenizer revision through the Hugging Face cache.

The standalone native build interface discovers MSVC, uses CUDA from `CUDA_PATH`, and resolves PyTorch's CMake package from the active environment.

## Running Flux

Compare Hugging Face, PyTorch-integrated Flux, and native fixed-length generation on the same prompt:

```powershell
python scripts\flux_inference.py --device cuda --native `
  --prompt "The future of efficient inference is" `
  --max-new-tokens 32
```

Run the Python validation suite:

```powershell
python -m pytest -q
```

Run the native validation suite:

```powershell
scripts\native.cmd test
```

Run the canonical full-system benchmark:

```powershell
python benchmarks\benchmark_flux.py --mode system
```

Run focused benchmark modes with:

```powershell
python benchmarks\benchmark_flux.py --mode prefill
python benchmarks\benchmark_flux.py --mode decode
python benchmarks\benchmark_flux.py --mode generation
python benchmarks\benchmark_flux.py --mode profile
```

Reproduce the final prefill grid with:

```powershell
python benchmarks\benchmark_flux.py --mode prefill `
  --lengths 1,8,32,64,128,256,512,1024,2048,4096,8192 `
  --warmup 5 `
  --samples 20 `
  --rounds 3 `
  --correctness-tokens 0 `
  --report-cache-drift `
  --json-output build\final_prefill_results.json
```

Build and run the standalone CUDA microbenchmarks with:

```powershell
scripts\native.cmd benchmark
```

## What I Learned

* How to design and integrate custom C++/CUDA operators into PyTorch while preserving model and checkpoint behavior.
* How to optimize transformer inference by reducing memory traffic, kernel launches, tensor allocations, and redundant intermediate operations.
* How to build specialized decode paths using static KV caches, stable device memory, and CUDA Graph capture and replay.
* How to use Nsight Compute and benchmark data to analyze register pressure, occupancy, dependency chains, instruction throughput, and memory behavior.
* How to validate optimized kernels against Hugging Face and reject changes that improve theoretical metrics but regress end-to-end performance.


## Future Work

* Add FP16 and BF16 execution paths and evaluate Tensor Core acceleration.
* Explore quantized inference, including INT8 or lower-precision weight formats.
* Generalize the optimized runtime to additional transformer architectures and model sizes.
* Add EOS-aware native generation so fixed-length decoding can terminate directly on device.
* Investigate alternative long-context attention designs that reduce dependency latency without increasing instruction or synchronization overhead.
