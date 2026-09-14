# Retained one-token decode profile

Date: 2026-09-13

## Decision

The single highest-value next optimization target is the retained native
long-context GQA attention path, not the LM head. The LM head is the largest
individual remaining library kernel, but it is nearly flat at about 0.135 ms
through capacity 4096. Native GQA grows from 0.382 ms at capacity 1024 to
0.678 ms at 4096 and 1.036 ms at 8192. It is 38.6% and 48.3% of profiled device
time at the two longest capacities.

The exact proposed next milestone is: re-profile and investigate bounded
improvements to the existing native FP32 one-token grouped-GQA chunk and
reduction kernels at fixed capacities 4096 and 8192, preserving current-stream,
mask, cache-length, stable-output, CUDA-Graph, and numerical semantics. Retain
only changes that improve repeated integrated decode medians. Do not couple
that milestone to an LM-head or another projection experiment.

This task changed profiling support only. It did not change model dispatch,
operators, kernels, tolerances, or the optimized execution path.

## Configuration

- Model: `HuggingFaceTB/SmolLM2-135M`, revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- Shape: 30 layers, hidden 576, intermediate 1536, 9 query heads, 3 KV
  heads, head dimension 64, vocabulary 49152
- Python 3.11.9; PyTorch 2.14.0+cu132; CUDA 13.2; Transformers 5.16.1
- NVIDIA GeForce RTX 5070 Ti, `sm_120`, 70 SMs, 16,275 MiB, 48 MiB L2;
  driver 616.64
- FP32, TF32 disabled, deterministic algorithms and safety filling enabled
- Uninstrumented latency: 100 clock-stabilization iterations, 20 warmups,
  100 CUDA-event samples, median
- Attribution: ten natural profiler replays, divided by ten; no inserted
  per-component events or synchronization

## End-to-end latency

The eager rows use the stated active context. The exact graph rows use the
stated fixed cache capacity and time natural replays near the end of that
capacity.

| tokens | eager cached ms/token | exact graph ms/token | graph launches |
|---:|---:|---:|---:|
| 128 | 10.6415 | 2.6013 | 1816 (fallback) |
| 512 | 11.0189 | 2.8091 | 1816 (fallback) |
| 1024 | 11.1627 | 1.4074 | 315 |
| 2048 | 11.0097 | 1.4508 | 315 |
| 4096 | 10.8952 | 1.6766 | 315 |
| 8192 | 24.3729 | 2.0611 | 315 graph / 850 eager |

Capacities 128 and 512 intentionally remain on the existing safe graph
fallback because stable optimized scratch is enabled only for capacities
513--8192. A context-oriented capture reserving 120 decode slots measured
2.5972, 1.4111, 1.4159, 1.4630, and 1.9413 ms at prompt contexts 128, 512,
1024, 2048, and 4096 respectively; the context-512 capture has capacity 632
and therefore uses the retained optimized graph path.

The eager 8192 row is supported by the existing benchmark, but its next token
has effective attention length 8193. This exceeds the native GQA operator's
8192-token contract and deliberately falls back to Transformers/PyTorch.

## Optimized graph attribution

Percentages below are shares of summed profiler device time. Profiler kernel
sums were within about 1--5% of uninstrumented medians; the medians above are
the authoritative latency values.

| component | 1024 ms / % | 2048 ms / % | 4096 ms / % | 8192 ms / % |
|---|---:|---:|---:|---:|
| native GQA attention | 0.3823 / 26.1% | 0.4008 / 27.3% | 0.6784 / 38.6% | 1.0364 / 48.3% |
| fused gate/up GEMV + SwiGLU | 0.2815 / 19.2% | 0.2820 / 19.2% | 0.2817 / 16.0% | 0.2817 / 13.1% |
| MLP down projection | 0.1944 / 13.3% | 0.1754 / 12.0% | 0.1747 / 9.9% | 0.1755 / 8.2% |
| RMSNorm + residual RMSNorm | 0.1399 / 9.5% | 0.1569 / 10.7% | 0.1399 / 8.0% | 0.1395 / 6.5% |
| LM-head projection | 0.1354 / 9.2% | 0.1349 / 9.2% | 0.1349 / 7.7% | 0.1780 / 8.3% |
| packed QKV projection | 0.1247 / 8.5% | 0.1245 / 8.5% | 0.1245 / 7.1% | 0.1248 / 5.8% |
| fused QKV/RoPE/cache update | 0.0706 / 4.8% | 0.0546 / 3.7% | 0.0708 / 4.0% | 0.0541 / 2.5% |
| attention output projection | 0.0945 / 6.4% | 0.0941 / 6.4% | 0.1110 / 6.3% | 0.1126 / 5.2% |
| remaining elementwise kernels | 0.0297 / 2.0% | 0.0299 / 2.0% | 0.0298 / 1.7% | 0.0300 / 1.4% |
| embedding/position/RoPE framework + state | 0.0133 / 0.9% | 0.0134 / 0.9% | 0.0133 / 0.8% | 0.0134 / 0.6% |

The optimized graph has 181 custom Flux launches, 92 cuBLAS/cuBLASLt
launches, and 42 framework launches. At capacity 4096 these account for
66.6%, 31.1%, and 2.4% of device time respectively. The 30 GQA calls each
launch a chunk kernel and a reduction kernel. The other principal counts are
61 norm kernels, 30 packed-QKV projections, 30 fused QKV/RoPE/cache kernels,
30 attention-output projections, 30 fused gate/up+SwiGLU kernels, 30 down
projections, 30 residual adds, and one LM head.

At exact capacities 128 and 512, framework cache/memory work alone accounts
for about 37--38% of profiled device time and 1,170 of 1,816 launches. The
fallback also retains separate projection/SwiGLU and PyTorch attention work.
This is a real small-capacity limitation, but ordinary context-512 capture
crosses the optimized threshold once decode capacity is reserved.

## Eager attribution and overhead

Eager decode has 610 launches at context 128 and 670 at contexts 512--4096.
Only about 1.73--2.37 ms/token is summed kernel execution. The remaining
8.5--9.4 ms/token, or 78--84% of end-to-end latency, is GPU idle time between
kernels caused primarily by Python/dispatcher launch issuance and allocation
work. A five-token CPU trace at context 4096 attributed 7.35 ms/token of self
CPU time to 669 `cudaLaunchKernel` calls per token.

The same trace observed one `cudaStreamSynchronize` per token, approximately
0.38 ms/token under profiler instrumentation, and no per-token `cudaMalloc` or
`cudaFree` event. DynamicCache concatenation nevertheless allocated about
183 MiB/token at context 4096 (914.83 MiB across five tokens) and its K/V copy
kernels grew from about 0.073 ms at context 128 to 0.342 ms at 4096. Thus eager
decode is launch/host-bound first, with increasing cache-copy and allocation
pressure at long context.

At context 4096, the eager device-time leaders were native GQA 0.698 ms,
DynamicCache management 0.342 ms, separate gate/up+SwiGLU 0.369 ms,
QKV+RoPE 0.220 ms, norms 0.218 ms, down projection 0.177 ms, LM head
0.161 ms, and attention output projection 0.121 ms. The optimized graph avoids
the eager stable-scratch fallbacks and host launch gaps.

At eager context 8192 the expected Transformers/PyTorch fallback adds 60
repeat-KV copies, 60 attention GEMMs, separate score processing, and 180 more
launches. Its single-pass device trace summed to 5.102 ms, while end-to-end
latency was 24.373 ms.

## Bound classification and LM head

Eager decode is launch/host- and allocation-bound. Optimized graph decode is a
mixture of launch latency and bandwidth-limited one-token work at 1024--2048,
then increasingly memory/parallelism-bound in native GQA as context grows. Its
latency increase from capacity 1024 to 8192 is 0.654 ms; the profiled GQA
increase over the same range is also about 0.654 ms.

The LM head is exactly `[1,1,576] x [49152,576]^T -> [1,1,49152]`, bias-free
FP32. Its weight is 113,246,208 bytes and it is one cuBLAS GEMV launch. It is
the largest single remaining library kernel, but only 7.7--9.2% of optimized
graph device time through capacity 4096 and it does not explain context
scaling. A targeted FP32 M=1 LM-head GEMV is therefore not the next milestone
from this evidence.

No unexpected PyTorch or Transformers compute kernels occur in the optimized
513--8192 graph path. The observed fallbacks are the documented eager stable-
scratch behavior, exact graph capacities below 513, and eager effective
attention length 8193.

## Validation

```powershell
build\python3119\python.exe -m pytest -q `
  tests\test_smollm2_flux.py tests\test_smollm2_cuda_graph.py `
  tests\test_stable_decode_outputs.py tests\test_native_gqa_decode_attention.py `
  tests\test_native_packed_gate_up_swiglu.py
# 105 passed in 5.14s

build\python3119\python.exe -m pytest -q
# 460 passed in 9.77s
```
