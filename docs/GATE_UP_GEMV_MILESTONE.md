# Fixed-shape packed gate/up GEMV milestone

Date: 2026-09-13

## Decision

Retain only the fused FP32 SmolLM2 one-token gate/up GEMV + SwiGLU kernel as
the separately opt-in `"fused_gate_up_swiglu"` category. The standalone custom
GEMV beat the existing library projection in isolation and improved integrated
replay, but fusion improved further, removed an additional launch, required no
new stable scratch, and cleanly subsumed the decode use case. The standalone,
shared-input, and explicit-vector variants were removed from production source.

The retained path is deliberately exact: input `[1,1,576]`, row-major packed
weight `[3072,576]`, output `[1,1,1536]`, FP32, no bias, batch/query length one.
It is active only with supported stable-buffer CUDA Graph decode. Eager,
prefill, CPU, non-FP32, other shapes/layouts, and unsupported graph capacities
retain packed `nn.Linear` followed by the existing packed SwiGLU operator. No
state-dict key, weight layout, or persistent weight storage changed.

All measurements used SmolLM2-135M revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, Python 3.11.9, PyTorch
2.14.0+cu132, CUDA 13.2, Transformers 5.16.1, and an RTX 5070 Ti (`sm_120`,
70 SMs, 48 MiB L2). TF32 was disabled, deterministic algorithms and safety
fills remained enabled, and CUDA-event results are warmup/repeated medians.

## Production baseline

The existing `nn.Linear`/`F.linear` dispatch executes one cuBLAS GEMV kernel.
The exact profiler symbol was:

`internal::gemvx::kernel<...cublasGemvTensorStridedBatched...>`

A layer-0 trace measured 9.205 us of device time for this kernel. A 30-layer,
distinct-weight CUDA Graph measured 10.134 us per projection and 0.914 us per
packed SwiGLU, or 10.950 us for the combined boundary. The previously retained
full-model profile attributed approximately 0.305-0.313 ms per token to the 30
gate/up projections and about 0.026 ms to their SwiGLU consumers. There are 30
gate/up calls per generated token.

The baseline projection output is graph-private stable storage under capture;
the packed SwiGLU result uses the existing caller-owned 6,144-byte stable
buffer. Eager returns an ordinary allocated projection output.

Exact logical payloads per layer are:

| payload | bytes |
|---|---:|
| packed gate/up weight | 7,077,888 |
| input vector | 2,304 |
| packed projection output | 12,288 |
| final SwiGLU output | 6,144 |

The projection minimum is 7,092,480 bytes. The unfused boundary minimum is
7,110,912 bytes because the 12,288-byte packed output is written and then read.

## Standalone GEMV investigation

Every scalar candidate assigned one output row to one warp. Lanes traversed
`K=576` in steps of 32, accumulated FP32 with FMA, and used warp shuffles for
the final reduction. Adjacent lanes therefore read adjacent row-major weight
elements. Four-warps/CTA used grid `(768,1,1)` and block `(128,1,1)`;
eight-warps/CTA used grid `(384,1,1)` and block `(256,1,1)`.

These bounded candidate timings used random unit-normal tensors, one hot weight
matrix, 100 operations per captured graph, 20 warmups, and 60 median samples.
They select architecture; the distinct real model weights and full decode graph
below are the retention evidence.

| standalone candidate | eager us | captured us | captured speedup vs library |
|---|---:|---:|---:|
| current `F.linear` | 14.522 | 5.381 | 1.000x |
| 4 warps, direct x | 9.973 | 3.216 | 1.673x |
| 8 warps, direct x | 10.017 | 3.047 | 1.766x |
| 4 warps, shared x | 9.985 | 3.375 | 1.595x |
| 8 warps, shared x | 10.122 | 3.529 | 1.525x |
| 8 warps, aligned `float4` | 13.402 | 3.646 | 1.476x |

The standalone 8-warp direct candidate plus retained packed SwiGLU measured
3.889 us in the same captured test versus 5.908 us for the current boundary.
Its integrated experimental speedups were 1.021x, 1.025x, 1.027x, and 1.021x
at capacities 1024, 2048, 4096, and 8192. It preserved 345 total launches but
replaced 30 library GEMVs with custom launches and needed one additional
12,288-byte stable packed-output buffer.

Shared staging loaded 2,304 bytes once per CTA but added a block-wide barrier
and regressed every captured comparison. Direct x loads are repeatedly
requested by warps but are naturally served from cache; explicit staging was
not beneficial. The `float4` variant was legal because Torch allocations are
16-byte aligned and the 2,304-byte row stride preserves alignment. It reduced
load instruction count but changed reduction grouping slightly and was slower,
so it was removed. The scalar accesses are already naturally coalesced.

The standalone result differed from the library reduction by maximum/mean
absolute error `1.717e-5 / 3.536e-6` on random unit-normal inputs. Gate and up
halves were both covered by this packed-output comparison. Caller output,
poison overwrite, repeat determinism, a non-default current stream, graph
capture/replay, and stable output addresses were validated before removal.

## Fused experiment and retained kernel

The retained kernel assigns one warp to one matched gate/up pair. Each input
load feeds two FP32 accumulators, and the warp reduces gate and up separately
before lane zero evaluates the existing FP32
`(gate / (1 + expf(-gate))) * up` expression. Eight warps form a 256-thread
CTA, giving grid `(192,1,1)`. It requests no dynamic shared memory and launches
on PyTorch's current stream under `CUDAGuard` into the existing caller-owned
SwiGLU stable buffer. Execution allocates nothing and is graph safe.

The direct fused candidate measured 3.077 us in the hot captured candidate
test versus 3.769 us for shared-x. With all 30 distinct real layer weights, the
current boundary measured 10.950 us/layer and fused measured 9.515 us/layer, a
1.151x speedup. A single hot layer profiler recorded 9.205 us for the library
projection kernel and 4.650 us for the fused kernel; the latter is not treated
as a DRAM-bandwidth result because its weight was cache-hot.

Fusing removes 12,288 bytes of packed-output writes and 12,288 bytes of reads
per layer: 24,576 bytes/layer and 737,280 bytes/token across 30 layers. The
theoretical fused minimum is 7,086,336 bytes (weight + one input read + final
output). Direct source-level instructions request the 2,304-byte input once per
matched output pair, about 3.375 MiB, but the 48 MiB L2 can serve almost all of
that reuse; physical DRAM traffic was not measured.

Using distinct-weight graph medians, minimum-traffic effective rates are about
700 GB/s for the 10.134-us library projection and 745 GB/s for the 9.515-us
fused boundary. The full-model profile's 0.281-ms fused total corresponds to
about 9.37 us/layer and 756 GB/s. These are simple payload/latency estimates,
not achieved hardware bandwidth.

`cuobjdump` reports 40 registers/thread, zero stack bytes, zero local memory,
and zero static shared memory. The launch also requests zero dynamic shared
memory. With 65,536 registers/SM, 256 threads/block, and
1,536 threads/SM, registers permit six blocks (1,536 threads), or 100%
theoretical thread occupancy. The rejected standalone scalar direct kernels
used 39 registers/thread, shared-x used 35, and `float4` used 42; the rejected
fused shared-x kernel used 40. None spilled. Shared variants requested 2,304
dynamic bytes/block. Nsight hardware counters remain unavailable because the
driver reports `ERR_NVGPUCTRPERM`, so achieved occupancy and DRAM bandwidth are
not claimed.

## Integrated results

The authoritative production comparison used one model, separate captured
states, interleaved baseline/fused samples, 10 warmups, and 30 samples. Capacity
512 cannot install the long-context stable scratch and is the exact existing
fallback.

| graph capacity | baseline ms/token | fused ms/token | speedup |
|---:|---:|---:|---:|
| 512 | same dispatch | same dispatch | 1.000x |
| 1024 | 1.4681 | 1.4156 | 1.037x |
| 2048 | 1.5043 | 1.4525 | 1.036x |
| 4096 | 1.6943 | 1.6387 | 1.034x |
| 8192 | 2.1221 | 2.0739 | 1.023x |

The capacity-4096 first-decoder-layer graph measured 46.768 us baseline and
45.920 us fused, a 1.018x speedup. A separate final capacity-4096 rerun measured
1.7601 -> 1.7024 ms/token (1.034x), confirming the result after the rejected
source was removed.

At capacity 4096, the natural graph changes from 345 to 315 launches. Library
GEMM/GEMV launches fall from 122 to 92; the 60 retained explicit cuBLASLt
QKV/output launches are unchanged, leaving 32 other library matrix launches.
There are 30 fused gate/up launches and 60 native GQA launches. Profiled total
device time changed from 1.7621 to 1.7247 ms; the 30 fused kernels contributed
0.2810 ms. The profiler run is diagnostic and is separate from event medians.

Eager dispatch is intentionally identical. The measured baseline/fused values
were 13.7889, 14.3952, 14.4477, 14.2828, and 14.5035 ms/token at contexts 512,
1024, 2048, 4096, and 8192 respectively, each 1.000x because both flags execute
the same `nn.Linear` path. The category cannot affect prefill.

## Correctness

No model tolerance was weakened. Final retained-path results were:

| comparison | maximum absolute | mean absolute |
|---|---:|---:|
| random fused activation | 8.240e-4 | 5.755e-5 |
| real layer-0 fused activation | 4.959e-5 | 2.442e-6 |
| transformer layer output | 1.907e-6 | 3.081e-7 |
| cached/graph decode logits (worst sweep row) | 8.678e-5 | 3.794e-5 |
| full graph K/V cache (worst maxima across rows) | 8.249e-5 | 8.061e-9 |

The larger random activation delta follows amplification of small FP32 dot
product order differences by unit-normal weights/inputs; real model weights are
substantially smaller. Three random seeds, poisoned output, repeated invocation,
bit-exact custom repeatability, caller-output identity, non-default stream,
FakeTensor/schema `torch.library.opcheck`, graph capture, changing-input replay,
and stable addresses pass. Four repeated graph decode steps at every retained
capacity preserve greedy token IDs. Layer outputs, logits, all graph cache
storage, and full-cache address stability are checked. Unsupported eager and
small-geometry model paths are bit-identical fallbacks.

## Memory

The fused path reuses the 6,144-byte stable SwiGLU output and adds no scratch.
Stable scratch is identical before/after: 0, 40,512, 59,520, 97,536, and
173,568 bytes at capacities 512, 1024, 2048, 4096, and 8192. The 12,288-byte
packed gate/up graph-private temporary is eliminated per layer lifetime, but
the allocator high-water mark does not fall.

Fresh-process capacity-4096 measurements are identical: graph pool 33,751,040
bytes, stable scratch 97,536 bytes, setup peak 1,552,240,640 bytes. Eager
incremental peak is 4,268,032 bytes for both paths because eager falls back.
Persistent model-weight storage is 538,060,032 bytes, including 212,336,640
bytes of packed gate/up weights across 30 layers, and is unchanged. No weight
is transposed, repacked, duplicated, or added.

## Recommendation

The custom ownership evidence is positive for this exact boundary, but it does
not justify a general GEMM/GEMV framework or automatic ownership of another
projection. The updated 4096 profile still attributes roughly 0.7 ms to native
GQA and leaves the LM head as the largest individual library projection. The
next milestone should profile the retained system again and make a bounded
choice between LM-head GEMV and further long-context GQA work; current evidence
slightly favors the LM head as the next projection experiment because gate/up
is now custom and QKV/output already use tuned cuBLASLt. Do not start QKV,
down-projection, or arbitrary GEMM work without a new integrated profile.

## Reproduction

```powershell
$env:FLUX_BUILD_NATIVE='1'
build\python3119\python.exe setup.py build_ext --inplace
build\python3119\python.exe -m pytest tests\test_native_packed_gate_up_swiglu.py -q
build\python3119\python.exe benchmarks\benchmark_smollm2_gate_up_gemv.py `
  --warmup 10 --repetitions 30 --capacities 512,1024,2048,4096,8192 `
  --correctness-replays 4
build\python3119\python.exe benchmarks\benchmark_smollm2_gate_up_gemv.py `
  --warmup 1 --repetitions 3 --capacities 4096 --skip-eager `
  --skip-layer --skip-profile --independent-memory
```
