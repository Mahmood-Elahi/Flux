# Retained decode re-profile and LM-head GEMV experiment

Date: 2026-09-13

## Decision

Do not retain a custom LM-head GEMV. A fresh profile selected the LM head over
GQA as the bounded experiment because it was the largest individual remaining
library kernel, but not the largest category. The best isolated candidate was
an aligned eight-warp `float4` kernel, and its captured microbenchmark was only
1.006x faster than cuBLAS. In the authoritative integrated comparison it was
neutral at capacity 1024 and reproducibly slower at 2048, 4096, and 8192. All
LM-head CUDA, binding, model-dispatch, and stable-scratch experiment code was
therefore removed from production source.

The retained production path remains unchanged: the full FP32 logits tensor is
produced by the existing `nn.Linear`/cuBLAS path. No LM-head weight was copied,
transposed, or repacked. The next measured target is long-context native GQA;
this milestone does not implement a second optimization target.

All results used SmolLM2-135M revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, Python 3.11.9, PyTorch
2.14.0+cu132, CUDA 13.2, Transformers 5.16.1, and an RTX 5070 Ti (`sm_120`,
70 SMs, 48 MiB L2). TF32 was disabled, deterministic algorithms and safety
fills remained enabled, and CUDA-event results are warmup/repeated medians.

## Fresh retained-system profile

The uninstrumented current production configuration was stabilized with 100
untimed graph replays, then measured with 20 warmups and 100 samples. These are
fresh results, not values carried forward from the gate/up milestone.

| graph capacity | retained ms/token | launches/token |
|---:|---:|---:|
| 1024 | 1.4119 | 315 |
| 2048 | 1.4458 | 315 |
| 4096 | 1.5532 | 315 |
| 8192 | 2.0631 | 315 |

Profiler device-time attribution is diagnostic and separate from those event
medians. Ten natural replays at every capacity produced the following average
category totals. Packed QKV and attention-output projections share one cuBLASLt
kernel family; their occurrences were separated in execution order.

| category | 1024 ms | 2048 ms | 4096 ms | 8192 ms |
|---|---:|---:|---:|---:|
| embedding / position / RoPE framework work | 0.0095 | 0.0096 | 0.0096 | 0.0099 |
| RMSNorm / residual RMSNorm | 0.1393 | 0.1383 | 0.1390 | 0.1383 |
| packed QKV projection | 0.1239 | 0.1240 | 0.1249 | 0.1414 |
| fused QKV/RoPE/cache update | 0.0546 | 0.0546 | 0.0546 | 0.0550 |
| native GQA attention | 0.3632 | 0.3996 | 0.6479 | 1.0156 |
| attention output projection | 0.0945 | 0.0939 | 0.1113 | 0.0946 |
| fused gate/up GEMV + SwiGLU | 0.2981 | 0.2815 | 0.2984 | 0.2814 |
| MLP down projection | 0.1751 | 0.1750 | 0.1745 | 0.1917 |
| LM head | 0.1353 | 0.1353 | 0.1349 | 0.1557 |
| cache/state management | 0.0037 | 0.0036 | 0.0036 | 0.0036 |
| remaining framework kernels | 0.0300 | 0.0296 | 0.0295 | 0.0448 |
| other | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **summed device time** | **1.4272** | **1.4450** | **1.7282** | **2.1320** |

The primary capacity-4096 breakdown is:

| category | time/token ms | us/call | calls/token | launches/token | replay device % |
|---|---:|---:|---:|---:|---:|
| embedding / position / RoPE framework work | 0.0096 | 1.066 | 9 | 9 | 0.56% |
| RMSNorm / residual RMSNorm | 0.1390 | 2.278 | 61 | 61 | 8.04% |
| packed QKV projection | 0.1249 | 4.163 | 30 | 30 | 7.23% |
| fused QKV/RoPE/cache update | 0.0546 | 1.822 | 30 | 30 | 3.16% |
| native GQA attention | 0.6479 | 21.597 | 30 | 60 | 37.49% |
| attention output projection | 0.1113 | 3.710 | 30 | 30 | 6.44% |
| fused gate/up GEMV + SwiGLU | 0.2984 | 9.947 | 30 | 30 | 17.27% |
| MLP down projection | 0.1745 | 5.818 | 30 | 30 | 10.10% |
| LM head | 0.1349 | 134.905 | 1 | 1 | 7.81% |
| cache/state management | 0.0036 | 0.910 | 4 | 4 | 0.21% |
| remaining framework kernels | 0.0295 | 0.982 | 30 | 30 | 1.70% |
| other | 0.0000 | - | 0 | 0 | 0.00% |

Here, a "call" for the framework-only rows is one kernel invocation. A native
GQA call deliberately launches a chunk kernel and a reduction kernel.

| owner | time/token ms | launches/token | time/launch us | replay device % |
|---|---:|---:|---:|---:|
| custom Flux CUDA | 1.1399 | 181 | 6.298 | 65.96% |
| cuBLAS/cuBLASLt | 0.5466 | 92 | 5.942 | 31.63% |
| framework CUDA | 0.0417 | 42 | 0.993 | 2.41% |

GQA is the largest category, while the LM head is the largest single library
kernel and has a simple M=1 boundary. That combination satisfied the requested
rule for trying the LM head first without implying that it would necessarily
become the next retained operator.

## Exact LM-head workload and baseline

The pinned production configuration derives `hidden_size=576` and
`vocab_size=49152`. The exact operation is:

`[1, 1, 576] x [49152, 576]^T -> [1, 1, 49152]`

Thus `M=1`, `N=49152`, and `K=576`. Input, weight, output, and accumulation are
FP32. The existing weight is contiguous row-major with shape `[49152,576]` and
stride `(576,1)`. It occupies 113,246,208 bytes; the full logits output occupies
196,608 bytes. Counting one input read, the minimum logical payload is
113,445,120 bytes.

The production call is bias-free `nn.Linear`/`F.linear`. cuBLAS selected:

`gemv2T_kernel_val<...,128,16,4,4,...cublasGemvTensorStridedBatched...>`

It is one launch. The isolated real-weight CUDA-event medians were 144.879 us
eager and 142.704 us captured. The capacity-4096 full-graph trace attributed
134.905 us, or 7.81% of summed replay device time, to this kernel.

## CUDA candidates and isolated benchmark

All scalar kernels assigned one vocabulary row to one warp. Lanes traversed
the contiguous hidden dimension, accumulated with FP32 FMA, and reduced with
warp shuffles. Four-warps/CTA used grid `(12288,1,1)` and block `(128,1,1)`;
eight-warps/CTA used grid `(6144,1,1)` and block `(256,1,1)`. Shared variants
staged the 2,304-byte input once per CTA. The vector variant used aligned
`float4` loads; Torch allocation alignment and the 2,304-byte row stride made
every vector address 16-byte aligned.

Timings used the real production LM-head weight, 30-call batches, 20 warmups,
and 60 median samples. Effective bandwidth is minimum logical payload divided
by captured latency, not a hardware-counter measurement.

| candidate | eager us | captured us | captured speedup | launches | estimated payload | effective GB/s |
|---|---:|---:|---:|---:|---:|---:|
| current cuBLAS | 144.879 | 142.704 | 1.000x | 1 | 113,445,120 B | 795.0 |
| 4 warps, direct x | 145.037 | 142.772 | 1.000x | 1 | 113,445,120 B | 794.6 |
| 8 warps, direct x | 144.836 | 142.583 | 1.001x | 1 | 113,445,120 B | 795.6 |
| 4 warps, shared x | 144.943 | 142.463 | 1.002x | 1 | 113,445,120 B | 796.3 |
| 8 warps, shared x | 145.017 | 142.840 | 0.999x | 1 | 113,445,120 B | 794.2 |
| 8 warps, `float4` | 143.452 | 141.849 | 1.006x | 1 | 113,445,120 B | 799.8 |

The 108 MiB weight is more than twice the 48 MiB L2, and the ~795-800 GB/s
payload estimates show why geometry and shared-input changes barely moved the
result. The input vector is tiny and naturally cache resident; shared staging
added synchronization without reducing the dominant physical traffic. The
`float4` kernel was the only justified integrated candidate, but its isolated
advantage was under one microsecond.

`cuobjdump --dump-resource-usage` reported:

| candidate | registers/thread | static shared | local | stack | spills | theoretical thread occupancy |
|---|---:|---:|---:|---:|---:|---:|
| 4 warps, direct x | 40 | 0 B | 0 B | 0 B | none | 100% |
| 8 warps, direct x | 40 | 0 B | 0 B | 0 B | none | 100% |
| 4 warps, shared x | 35 | 3,328 B | 0 B | 0 B | none | 100% |
| 8 warps, shared x | 35 | 3,328 B | 0 B | 0 B | none | 100% |
| 8 warps, `float4` | 42 | 0 B | 0 B | 0 B | none | 100% |

The shared figure includes tool-reported static shared overhead in addition to
the 2,304-byte staged vector. Occupancy is a launch/resource-limit calculation
using 1,536 threads and 65,536 registers per SM, not an achieved measurement.
Nsight Compute failed with `ERR_NVGPUCTRPERM`, so achieved occupancy and DRAM
bandwidth are not claimed.

## Correctness

No tolerance was loosened. Before removal, the best `float4` candidate passed
poisoned caller output, return identity, repeated invocation, bit-exact custom
repeatability, a non-default current CUDA stream, CUDA Graph capture, changing
input replay, repeated replay, and stable output addresses.

| comparison | maximum absolute | mean absolute |
|---|---:|---:|
| random FP32 inputs/weights, worst of 3 seeds | 2.289e-5 | 3.028e-6 |
| real final hidden state and production weight | 1.526e-5 | 2.393e-6 |
| full graph logits, worst capacity/replay | 2.098e-5 | 3.116e-6 |
| full K/V cache | 0.000e+0 | 0.000e+0 |

Four generated-token comparisons at every capacity preserved exact greedy
token IDs. Full cache state and all stable addresses were unchanged. The cache
is exactly equal because the LM head executes after transformer state updates.
The rejected operator was inference-only, used `CUDAGuard` and PyTorch's current
CUDA stream, wrote caller-owned output, allocated nothing during execution, and
was graph safe.

## Integrated graph and eager results

The production graph and the best `float4` experiment were captured from one
model into separate states and measured in paired/interleaved order with 20
warmups and 100 samples. These paired numbers are the retention evidence and
are intentionally distinct from the standalone fresh-profile medians above.

| capacity | baseline ms/token | custom ms/token | speedup | total launches | custom CUDA | cuBLAS/Lt | GQA launches |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1.361824 | 1.361648 | 1.000x | 315 | 182 | 91 | 60 |
| 2048 | 1.379104 | 1.382160 | 0.998x | 315 | 182 | 91 | 60 |
| 4096 | 1.415040 | 1.422384 | 0.995x | 315 | 182 | 91 | 60 |
| 8192 | 1.766160 | 1.776960 | 0.994x | 315 | 182 | 91 | 60 |

Baseline ownership is 181 custom Flux, 92 cuBLAS/cuBLASLt, and 42 framework
launches. The experiment exchanges the one LM-head cuBLAS launch for one custom
launch, so total and GQA launch counts do not change. The small isolated win did
not survive integrated execution and does not meet the retention criterion.

Eager execution deliberately retained the identical `nn.Linear` dispatch. A
fresh secondary run measured 10.9300, 16.1280, 10.9918, and 14.8995 ms/token at
contexts 1024, 2048, 4096, and 8192; baseline and experiment are exactly 1.000x
because they execute the same path. Eager timing is secondary and visibly more
variable than fixed-shape graph replay.

## Memory and optional greedy fusion

Fresh-process capacity-4096 measurements were:

| item | baseline | experiment | retained after milestone |
|---|---:|---:|---:|
| logits/output bytes | 196,608 | 196,608 | 196,608 |
| stable scratch bytes | 97,536 | 294,144 | 97,536 |
| graph private pool | 33,751,040 | 33,554,432 | 33,751,040 |
| pool + explicit scratch | 33,848,576 | 33,848,576 | 33,848,576 |
| eager incremental peak | 3,240,960 | 3,240,960 | 3,240,960 |
| persistent model weights | 538,060,032 | 538,060,032 | 538,060,032 |
| persistent LM-head weight | 113,246,208 | 113,246,208 | 113,246,208 |

The experiment merely moved the 196,608-byte logits allocation from graph
private storage to explicit stable scratch. It saved no total graph storage and
added no persistent weight. Ordinary Flux model and graph APIs return full
logits at this call site. Because the standalone kernel failed retention and a
fused argmax would not preserve that general API, LM-head + argmax was not
implemented. Greedy-only sampling machinery was not broadened in this milestone.

## Updated profile and next milestone

Because the custom LM head was removed, the updated retained profile is exactly
the fresh baseline profile above. At capacity 4096, native GQA owns 0.6479 ms
(37.49%) and grows to 1.0156 ms at capacity 8192. It is the clear next target.
The recommended next milestone is a bounded re-profile and optimization study
of the existing native long-context GQA implementation. Do not automatically
start another projection GEMV, down projection, custom QKV, or general GEMM
framework.

## Reproduction

The retained-system profiler now enables all production categories, including
native GQA, fused QKV/RoPE/cache, selective cuBLASLt projections, and fused
gate/up GEMV + SwiGLU:

```powershell
$env:FLUX_BUILD_NATIVE='1'
build\python3119\python.exe setup.py build_ext --inplace
build\python3119\python.exe benchmarks\benchmark_flux.py --mode profile `
  --decode-contexts 1024,2048,4096 --skip-prefill `
  --warmup 10 --samples 30 --repetitions 30 --top-k 12
```

Capacity 8192 is measured by the fixed-capacity gate/up benchmark harness,
because the older context-oriented profiler reserves additional replay slots:

```powershell
build\python3119\python.exe benchmarks\benchmark_smollm2_gate_up_gemv.py `
  --warmup 10 --repetitions 30 --capacities 1024,2048,4096,8192 `
  --correctness-replays 4 --skip-eager --skip-layer
```
