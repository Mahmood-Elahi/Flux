# Milestone 36: long-context streaming GQA optimization

## Retained result

Long-context native prefill now partitions each causal context across multiple
CTAs, writes bounded FP32 online-softmax partial states, and merges those states
in a second kernel. The measured production dispatch keeps the existing short
paths, uses two context partitions from 1025 through 4096, and uses four above
4096. Native prefill retains compact `[3, capacity, 64]` K/V, causal masking,
FP32 accumulation, and no `repeat_kv`, score matrix, or probability matrix.

The retained CUDA-event medians improve complete 30-layer native prefill from
45.0723/103.6828/281.6308 ms to 38.225/95.685/254.368 ms at lengths
2048/4096/8192: 1.179x, 1.084x, and 1.107x respectively.

## Baseline architecture and drift investigation

The Milestone 35 kernel carried `(m, l, o[64])` in registers for one query row
per thread. A 128-thread CTA owned 128 rows for one query head and reused shared
32-token K/V tiles. It launched only `ceil(S/128) * 9` CTAs per layer and had a
long triangular tail: later rows did much more work, while one CTA could not
divide a row's context across SMs. It required zero attention-specific global
workspace but consumed 164 registers/thread and 17,408 bytes shared/CTA.

The reported cache edge predates Milestone 35. It is already documented in the
Milestone 30 retained implementation, including per-layer amplification and
length-8192 cache maxima. The only Milestone 35 prefill change replaced the
post-LM-head argmax kernel; attention, hidden-state, QKV, and cache production
were unchanged. Therefore Milestone 35 did not introduce the drift.

Fresh diagnostics reproduced the strict comparison failure at length 2048
(`2.371054e-5` at cache index `[0,0,1797,2]` after continuation) and the known
length-4096 edge. In a direct prefill layer audit, layer 0 K and V were bitwise
identical. Layer 1 was the first cache layer with any difference. Thus input
embedding, layer-0 normalization, QKV projection, RoPE, K, V, and cache writes
do not diverge first; the first reordered computation is layer-0 streaming
attention output, followed by the layer-0 residual/MLP hidden state and then
layer-1 K/V. Direct attention remains within about `1.6e-7` of the independent
cuBLAS/causal-softmax/cuBLAS oracle. Per-layer cache maxima rise and fall rather
than growing monotonically. Greedy tokens remain identical. These facts identify
deterministic FP32 accumulation-order amplification, not a semantic cache,
causal-mask, RoPE, or GQA-head mapping defect. No tolerance was changed.

## Split-context algorithm

For every `(query head, query row, context partition)`, stage 1 computes a local
FP32 state `(m_i, l_i, o_i[64])`. Empty causal partitions write the identity
state `(-infinity, 0, 0)`. Partitions are contiguous, 32-token aligned, and
cover the context exactly. K/V stay compact and are staged with aligned
`float4` loads into the same shared 32-by-64 tiles as the baseline.

The merge uses the stable online-softmax composition. With
`m = max_i(m_i)`, it computes:

```text
w_i = exp(m_i - m)
l   = sum_i(l_i * w_i)
o   = sum_i(o_i * w_i)
output = o / l
```

One warp merges one query row; each lane owns two output dimensions. Lane zero
computes each scalar weight and normalization and broadcasts them with warp
shuffles. Partitions are merged in increasing context order. Both stages launch
asynchronously on the caller's current CUDA stream and perform no allocation or
synchronization.

## Candidate measurements and decisions

CUDA-event medians used 10 warmups and 51 samples on the RTX 5070 Ti. The old
`query_tile_128` path and all serious split candidates used 128 threads unless
shown otherwise.

| Candidate | 2048 | 4096 | 8192 | Decision |
| --- | ---: | ---: | ---: | --- |
| existing single CTA/row | 0.6528 ms | 1.8291 ms | 6.2750 ms | retain for comparison/short dispatch |
| split 2 | 0.4489 ms | 1.6127 ms | 5.6240 ms | keep through 4096 |
| split 4 | 0.4700 ms | 1.4406 ms | 5.3477 ms | keep above 4096 |
| split 8 | 0.4798 ms | 1.5520 ms | 5.4695 ms | remove; extra state/merge traffic |
| split 4, 64 threads | 0.4586 ms | 1.5755 ms | 5.7845 ms | remove; insufficient tile reuse |

Four partitions was the isolated winner at 4096, but it produced a sparse
full-model K-cache maximum of `3.47614e-4`. Two partitions kept the 4096 cache
envelope much closer to the old kernel (`7.91550e-5` K, `5.81741e-5` V) and
still improved complete prefill, so numerical behavior determined the retained
4096 choice. At 8192, four partitions improved the old documented cache maxima
to approximately `8.73566e-4` K and `4.33683e-4` V. The rejected eight-way and
64-thread instantiations were removed. This work does not repeat the rejected
three-query-head K/V-sharing design.

The final isolated speedups are 1.454x at 2048, 1.134x at 4096, and 1.173x at
8192. The final dispatch is:

| Regime | Production path |
| --- | --- |
| `S <= 128` | existing streaming short kernel |
| `129 <= S <= 1024` | existing bounded legacy path (maximum 36 MiB scores) |
| `1025 <= S <= 4096` | two-way split context |
| `4097 <= S <= 8192` | four-way split context |

## Workspace and kernel resources

Each partial is 66 FP32 values. Attention workspace is exactly
`9 * S * partitions * 66 * 4` bytes and is a slice of the existing stable
runtime workspace.

| Length | Partitions | Attention partials | Total prefill workspace | Old quadratic scores |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 2 | 9,732,096 B | 93,096,192 B | 150,994,944 B |
| 4096 | 2 | 19,464,192 B | 186,190,080 B | 603,979,776 B |
| 8192 | 4 | 77,856,768 B | 411,306,240 B | 2,415,919,104 B |

The 8192 attention workspace is 74.25 MiB, 31 times smaller than the former
2.25 GiB score matrix, and remains linear rather than quadratic. Addresses are
stable and repeated prefills show zero allocation growth.

`cuobjdump --dump-resource-usage` reports 162 registers/thread and 17,408 bytes
shared/CTA for stage 1. On the target SM (65,536 registers, 100 KiB shared,
1,536 threads maximum), registers permit three 128-thread CTAs/SM, or 12 active
warps (25% thread occupancy). The merge uses 19 registers/thread for two-way or
26 for four-way, no shared memory, and eight warps/CTA. Stage-1 CTA counts per
layer are 288/576/2,304 at 2048/4096/8192; merge CTA counts are
2,304/4,608/9,216. Split context increases useful stage-1 parallelism and
reduces triangular tail imbalance without sharing three Q heads in one CTA.

## Integrated benchmark and profile

The canonical benchmark used deterministic FP32 inputs, TF32 disabled, three
warmups, 10 samples per round, three rotating rounds, and medians of all CUDA
event samples.

| Length | Baseline native | Final native | Speedup | Baseline attention | Final attention | Final attention share |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 45.0723 ms | 38.225 ms | 1.179x | 20.6517 ms | 13.6303 ms | 35.7% |
| 4096 | 103.6828 ms | 95.685 ms | 1.084x | 56.3487 ms | 48.2104 ms | 50.6% |
| 8192 | 281.6308 ms | 254.368 ms | 1.107x | 187.9269 ms | 160.3586 ms | 62.7% |

The final profile records two attention launches per layer. A second
stage-separated profile attributed 13.8010/48.7820/163.9964 ms to stage 1 and
0.1565/0.3268/3.5137 ms to merge at 2048/4096/8192; profiling variance explains
the difference from the aggregate profile. Attention is no longer the largest
2048 component (projection GEMMs are), but remains the largest at 4096/8192.

Decode medians were 1.6852 ms at 4096 and 2.1031 ms at 8192, compared with the
Milestone 35 profile's 1.7934 and 2.1029 ms. Addresses remained stable and
allocation growth was zero, so there is no decode regression attributable to
this prefill-only dispatch.

## Correctness

The native direct test covers arbitrary scores, causal/tile boundaries, all
nine Q heads and three KV groups, non-default streams, repeated deterministic
execution, stable allocation state, and production lengths. Against the
independent retained attention oracle, final-dispatch maximum absolute errors
were `1.34110e-7`, `1.41561e-7`, and `1.30385e-7` at 2048/4096/8192. Uniform
score tests through 8192 peaked at `5.96046e-8`.

Canonical model validation preserved exact state-dict compatibility, compact
K/V shape/state, positions, stable addresses, finite values, prefill-to-decode
continuation, and greedy identity. Final prefill logit errors versus Python Flux
were `2.38419e-5`, `3.14713e-5`, and `8.20160e-5` at 2048/4096/8192, within the
existing combined FP32 tolerance. Known sparse cache outliers remain reportable
with `--report-cache-drift`; strict cache assertions remain the default.

## Validation status and next measurement

The rebuilt Python extension, canonical benchmark/profiler, resource audit,
all 11 native CTest targets (eight CUDA and three CPU), all 146 Python tests,
the 12-test FakeTensor/opcheck selection, system/generation smoke tests,
compileall, and `git diff --check` passed. A transient Windows Smart App Control
block on newly linked executables was cleared by rebuilding the affected exact
targets; the final complete CTest run is clean.

The measured next target remains streaming GQA stage 1 at 4096/8192. A future
milestone should investigate lowering its 162-register footprint or improving
context work distribution while preserving the retained merge order and cache
envelope. It should not optimize merge first (only 0.33/3.51 ms), replace the
winning library GEMMs, or revive multi-Q-head CTA sharing without new evidence.

**KEEP:** two-way/four-way split-context stage 1, warp merge, stable linear
workspace, and measured dispatch. **REMOVE:** eight partitions and the
64-thread stage-1 geometry. **DEFER:** further stage-1 optimization.

## Source accounting

The recursive repository metric excludes Git/build/virtual-environment/cache
and temporary directories.

| Language | Milestone 35 | Milestone 36 | Delta | Final share |
| --- | ---: | ---: | ---: | ---: |
| Python (`.py`) | 266,084 B | 267,004 B | +920 B | 35.429010% |
| CUDA (`.cu` + `.cuh`) | 321,764 B | 335,035 B | +13,271 B | 44.456107% |
| C++ (`.cpp` + `.cc` + `.cxx`) | 133,693 B | 133,693 B | 0 B | 17.739849% |
| headers (`.h` + `.hpp`) | 17,810 B | 17,899 B | +89 B | 2.375035% |

Total counted source is 753,631 bytes. CUDA remains exactly 83,561 bytes short
of equal CUDA/non-CUDA source. The added CUDA implements retained kernels and
their direct benchmark/test coverage; no source-byte target shaped the design.
