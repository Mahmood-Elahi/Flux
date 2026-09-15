# FP32 one-token projection characterization and cuBLASLt retention

Date: 2026-09-12

## Decision

Retain an explicitly selected, zero-workspace cuBLASLt configuration for the
packed QKV and attention-output projections in the fixed-shape FP32 CUDA-Graph
decode path. Keep packed gate/up, MLP down, and the LM head on PyTorch
`nn.Linear`. The optimized category is separately opt-in as
`"cublaslt_projection"`, requires the retained packed-QKV/RoPE/cache path, and
does not change eager or unsupported execution.

The retained configuration is cuBLASLt algorithm ID 13, tile 0, split-K 1,
reduction 0, CTA swizzle 0, custom option 91, stages 0, and zero workspace.
It was selected for this exact RTX 5070 Ti / CUDA 13.2 / FP32 workload. Flux
initializes and checks it before capture, supplies caller-owned output and
workspace tensors, uses PyTorch's current CUDA stream and device guard, and
performs no replay-time allocation or search.

No custom GEMM, CUTLASS, Triton, code generation, quantization, batching,
speculative decoding, or fused epilogue was introduced.

## Environment and method

Measurements used SmolLM2-135M revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, Python 3.11.9, PyTorch
2.14.0+cu132, CUDA 13.2, Transformers 5.16.1, and an RTX 5070 Ti. All model
weights and activations were FP32. TF32 was disabled, deterministic algorithms
and deterministic safety filling were enabled, and CUDA-event medians used
warmups and repeated samples. Weight loading, random generation, output and
workspace allocation, heuristic enumeration, and graph capture were outside
timed regions.

The extension was compiled for `sm_120` by CUDA 13.2 `nvcc` and the MSVC
14.51 toolset. Nsight Compute performance counters were not collected, so this
report makes no DRAM-throughput or achieved-occupancy claims.

The bounded tuning search requested at most eight of cuBLASLt's heuristic
results for each exact shape with a 4 MiB cap. The API comparison covered
`F.linear`, `torch.mm(out=...)`, `torch.matmul(out=...)`, and
`torch.addmm(beta=0, out=...)`. Nsight Systems 2025.6.3 traced CUDA, cuBLAS,
and NVTX ranges. It showed that current PyTorch reaches the classic cuBLAS
strided-batched GEMV path (`cublasGemvTensorStridedBatched`), while every
candidate still launched GEMV-shaped kernels. Split-K candidates added a
second reduction kernel and lost.

## Exact projection inventory

All operations are bias-free, contiguous row-major `[N,K]` weights evaluating
`[1,K] @ weight.T -> [1,N]`. Eager `nn.Linear` owns a fresh result. Under graph
capture the ordinary result is graph-private; the retained two projections
instead write into stable state-owned buffers.

| operation | input | weight | output | M/N/K | calls/token |
|---|---|---|---|---|---:|
| packed QKV | `[1,1,576]` | `[960,576]` | `[1,1,960]` | 1/960/576 | 30 |
| attention output | `[1,1,576]` | `[576,576]` | `[1,1,576]` | 1/576/576 | 30 |
| packed gate/up | `[1,1,576]` | `[3072,576]` | `[1,1,3072]` | 1/3072/576 | 30 |
| MLP down | `[1,1,1536]` | `[576,1536]` | `[1,1,576]` | 1/576/1536 | 30 |
| LM head | `[1,1,576]` | `[49152,576]` | `[1,1,49152]` | 1/49152/576 | 1 |

The dimensions and multiplicities are derived from the loaded configuration:
hidden size 576, 30 layers, 9 query heads, 3 KV heads, head dimension 64,
intermediate size 1536, and vocabulary size 49152.

## Isolated results

The table reports a 20-warmup, 60-sample CUDA-Graph median. Layer projections
were captured as a batch over the 30 real layer weights and divided by 30; the
LM head used repeated calls to amortize graph launch cost. “Best bounded Lt” is
diagnostic even where it was rejected.

| operation | `F.linear` us | best other API us | best bounded Lt us | speedup | kernels/workspace | decision |
|---|---:|---:|---:|---:|---:|---|
| packed QKV | 4.532 | 4.564 (`mm`) | 4.223 | 1.073x | 1 / 0 | retain custom 91 |
| attention output | 3.609 | 3.605 (`addmm`) | 2.724 | 1.325x | 1 / 0 | retain custom 91 |
| packed gate/up | 10.169 | 10.165 (`matmul`) | 10.026 | 1.014x | 1 / 0 | reject |
| MLP down | 5.931 | 5.930 (`matmul`) | 5.792 | 1.024x | 1 / 0 | reject |
| LM head | 135.429 | 135.379 (`mm`) | 135.382 | 1.000x | 1 / 0 | reject |

All four PyTorch API forms selected one GEMV launch and had effectively equal
graph latency. The LM head is bandwidth dominated for this matrix-vector
geometry. Its 196,608-byte logits output is graph-owned and written by the
single GEMV with no separate output kernel; 135.429 us is about 7.6% of the
4096-capacity token time. Extra workspace or split-K did not help. The best
split-K LM-head candidate was 150.704 us and used two launches. Gate/up and
down improvements were too small and variable to justify replacing the mature
path.

The bounded results below record every returned configuration tested. Entries
are `algorithm ID / split-K / custom option / workspace bytes / graph us`;
tile, stage, reduction, and swizzle were zero except that split-K candidates
used reduction scheme 2.

| shape | tested cuBLASLt configurations |
|---|---|
| QKV | `13/1/12/0/4.563`, `13/4/72/15376/5.750`, `13/6/63/23056/6.301`, `13/10/53/38416/7.702`, `13/3/66/11536/6.099`, `13/1/91/0/4.223`, `13/8/78/30736/5.756`, `13/1/75/0/5.520` |
| attention output | `13/1/12/0/3.606`, `13/4/72/9232/4.389`, `13/6/63/13840/5.341`, `13/10/53/23056/5.414`, `13/3/66/6928/5.278`, `13/1/91/0/2.724`, `13/8/78/18448/4.185`, `13/1/75/0/3.062` |
| gate/up | `13/1/71/0/10.226`, `13/8/78/98320/12.287`, `13/2/72/24592/12.173`, `13/4/72/49168/12.301`, `13/1/11/0/10.026`, `13/1/12/0/10.094` |
| down | `13/1/93/0/5.930`, `13/10/53/23056/9.337`, `13/5/101/11536/9.486`, `13/1/75/0/5.792`, `13/36/96/82960/9.098`, `14/9/0/20739/8.419`, `13/8/78/18448/7.306`, `13/5/67/11536/8.141` |
| LM head | `13/1/11/0/135.382`, `13/1/71/0/135.771`, `13/1/77/0/155.502`, `13/1/75/0/177.843`, `13/2/72/393232/163.419`, `13/4/72/786448/150.704`, `13/3/66/589840/159.438` |

Nsight Systems' 20-call ranges measured current-versus-selected kernel medians
of 2.496/2.010 us for QKV and 2.512/1.728 us for attention output. The larger
per-call graph numbers above include each operation's amortized share of the
outer graph replay launch. Current QKV, output, and LM-head ranges displayed
`gemv2T_kernel_val`; gate/up and down displayed the profiler's base `kernel`
name but were likewise single GEMV-family launches. No output-only kernel
appeared.

## Integrated ablation and interaction effects

Benchmark-only module swaps independently replaced all 30 instances of each
layer category and the single LM head. One representative 4096-capacity run:

| category | baseline ms/token | candidate ms/token | speedup |
|---|---:|---:|---:|
| packed QKV | 1.7901 | 1.7723 | 1.010x |
| attention output | 1.7901 | 1.7621 | 1.016x |
| packed gate/up | 1.7901 | 1.7769 | 1.007x |
| MLP down | 1.7901 | 1.7770 | 1.007x |
| LM head | 1.7901 | 1.7866 | 1.002x |

A repeat measured gate/up at 1.003x and down at 1.001x, below the 0.5%
retention threshold, while QKV plus attention output together measured 1.019x.
An all-four-layer-projection diagnostic reached 1.026x in the stronger run,
but its extra marginal gain did not remain above threshold independently.
Consequently only QKV and attention output survived the retention gate. This
also avoids 438,000 bytes of per-module benchmark scratch; production shares
two sequential-lifetime buffers totaling 6,144 bytes.

## Retained production results

These paired, interleaved results used five warmups and 20 samples. Capacity
512 remains on the pre-existing short-context fallback, so its 0.2% difference
is noise and its outputs are bit-identical. Capacities 1024-8192 execute the
new category.

| capacity | retained PyTorch ms/token | selected Lt ms/token | speedup |
|---:|---:|---:|---:|
| 512 | 2.7361 | 2.7298 | 1.002x |
| 1024 | 1.4909 | 1.4583 | 1.022x |
| 2048 | 1.5289 | 1.4958 | 1.022x |
| 4096 | 1.7892 | 1.7509 | 1.022x |
| 8192 | 2.1487 | 2.1170 | 1.015x |

At capacity 4096, a graph containing only the first real decoder layer measured
51.376 us for PyTorch projections and 49.952 us with the selected projections,
a 1.029x layer speedup. The full graph remains at 345 launches and 122
GEMV/GEMM-family launches: the change selects faster one-launch algorithms; it
does not claim launch elimination.

Launch composition is unchanged at capacities 1024-8192: 345 total, 122
GEMV/GEMM-family, 60 native GQA, and 163 other launches. Of the 122 library
matrix launches, 60 retained QKV/output launches use the explicit cuBLASLt
configuration; the other projections, LM head, and small framework matrix work
remain on their current paths. Capacity 512 is the established short-context
fallback at 1,816 total, 182 GEMV/GEMM-family, zero native-GQA, and 1,634 other
launches; the projection category is inactive there.

Using the isolated projection medians to attribute the selected 4096-capacity
total gives approximately 0.827 ms (47.2%) for projection plus LM-head work,
0.676 ms (38.6%) for native GQA, and 0.248 ms (14.2%) for everything else.
This combines separate median measurements and is therefore an attribution,
not a hardware-counter claim.

Eager decode deliberately has no caller-owned graph scratch and therefore
executes the same `nn.Linear` path with or without the category. The benchmark
reports 1.000x at capacities 512, 1024, 2048, 4096, and 8192 rather than treating
two timing batches of identical device work as distinct implementations.

## Correctness

Every unique projection shape was checked against PyTorch and for bit-exact
repeatability. For real model inputs, retained QKV maximum/mean absolute error
was `5.722e-6 / 6.126e-7`; attention output was
`7.153e-7 / 1.611e-7`. A unit-normal stress test observed a maximum QKV
accumulation-order difference of `1.240e-5`. Repeated calls were bit-exact.

The end-to-end capacity sweep preserved greedy tokens and stable addresses.
The largest retained-path logits maximum/mean absolute errors were
`4.673e-5 / 1.926e-5`. A one-layer exact-geometry CUDA-Graph test starts from a
510-token prompt, compares transformer-layer outputs and logits on every step,
compares all valid K/V entries, reaches the full 513-token cache boundary, and
then verifies exhaustion. The native tests also cover non-default current
streams, CUDA-Graph capture/replay, caller output identity, repeated inputs, and
FakeTensor/schema `torch.library.opcheck` for heuristic-index and exact-config
operators.

## Memory

Same-process graph-pool deltas are invalid here because PyTorch reuses private
pools across captures. Fresh spawned processes show that the graph pool and
setup peak are unchanged; only stable scratch grows by 6,144 bytes.

| capacity | graph pool, both paths | baseline scratch | selected scratch | setup peak, both paths |
|---:|---:|---:|---:|---:|
| 512 | 33,751,040 | 0 | 0 | 126,879,232 |
| 1024 | 33,751,040 | 34,368 | 40,512 | 175,528,448 |
| 2048 | 33,751,040 | 53,376 | 59,520 | 478,799,360 |
| 4096 | 33,751,040 | 91,392 | 97,536 | 1,556,788,224 |
| 8192 | 33,751,040 | 167,424 | 173,568 | 5,626,356,224 |

The cuBLASLt workspace is empty. Output and workspace addresses remain stable
across replay, and production performs neither heuristic search nor tensor
allocation during replay.

Eager incremental peak and additional persistent parameter storage are both
zero: the category has no eager buffers and reuses canonical checkpoint
weights. cuBLASLt receives the existing row-major weights with a transposed-B
operation and row-major matrix descriptors, so no runtime transpose is needed.
An alternate permanent layout was not pursued: it had no evidenced need and a
second copy of all five projection groups would cost 537,919,488 bytes
(513 MiB), with checkpoint and initialization complexity for no demonstrated
integrated benefit.

## Recommendation and next step

Retain the explicit cuBLASLt configuration only for packed QKV and attention
output projections during the optimized CUDA-Graph decode path. Keep gate/up,
down, and the LM head on their existing PyTorch paths: their isolated cuBLASLt
gains were marginal or absent and did not survive the integrated retention
criterion consistently. Eager decode remains unchanged.

This closes the useful library-level search for these exact FP32 M=1 shapes.
The next milestone should be a bounded, targeted custom one-token GEMV study,
starting with packed gate/up `(M, N, K) = (1, 3072, 576)`. Across 30 decoder
layers it is the largest remaining aggregate projection cost (about 0.305 ms
from the measured 10.169 us median), and it received no robust integrated win
from the available PyTorch or cuBLASLt paths. Any custom implementation should
first match the current operator's correctness, stream, graph-capture, stable
storage, and general-dimension contracts, then be retained only if it improves
the full decode graph rather than merely the isolated kernel.

## Reproduction

The bounded-search and category-ablation commands below record the original
milestone methodology. Their one-off harness was retired during final benchmark
consolidation after the selected configuration was frozen. The exact harness is
available at commit `b374d0423bdff5fe083eb4ded6890c358d265757`.

For the maintained production path, run
`tests/test_ops.py`, `tests/test_system.py`,
`benchmarks/benchmark_flux.py --mode profile`, and
`benchmarks/benchmark_flux.py --mode system`. These own the retained algorithm's
numerical/current-stream/graph contract and final integrated measurements.

```powershell
$env:FLUX_BUILD_NATIVE='1'
build\python3119\python.exe setup.py build_ext --inplace
New-Item -ItemType Directory -Force build\historical | Out-Null
git show b374d0423bdff5fe083eb4ded6890c358d265757:benchmarks/benchmark_smollm2_projections.py |
  Out-File -Encoding utf8 build\historical\benchmark_smollm2_projections.py
build\python3119\python.exe build\historical\benchmark_smollm2_projections.py `
  --warmup 20 --repetitions 60 --max-algorithms 8 --profile-replays 3 `
  --layer-capacity 0
build\python3119\python.exe build\historical\benchmark_smollm2_projections.py `
  --warmup 20 --repetitions 60 --integrated-only --layer-capacity 0
build\python3119\python.exe build\historical\benchmark_smollm2_projections.py `
  --warmup 5 --repetitions 20 --production-only `
  --production-capacities 512,1024,2048,4096,8192 `
  --layer-capacity 4096 --independent-memory --eager-production
```

The short Nsight Systems trace is generated with:

```powershell
nsys profile --trace=cuda,cublas,nvtx --force-overwrite=true `
  --output=build\projection_nsys build\python3119\python.exe `
  build\historical\benchmark_smollm2_projections.py --nsys-trace-only
```
