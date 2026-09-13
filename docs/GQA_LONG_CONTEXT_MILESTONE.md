# Long-context one-token GQA optimization

## Outcome

The retained CUDA path now uses 128-token partials above length 512. It keeps
the query-head decomposition through cache capacity 4096 and switches to one
CTA per `(KV head, context chunk)` above 4096, where measured KV reuse finally
outweighs the reduced grid size. The existing single-block shared-score kernel
is unchanged through length 512. Static CUDA Graph dispatch now crosses to the
native/stable path at capacity 513.

All measurements below used the repository Python 3.11 environment, PyTorch
2.14.0+cu132, CUDA 13.2, deterministic algorithms, TF32 disabled, and an RTX
5070 Ti (`sm_120`, 70 SMs, 48 MiB L2). CUDA events used warmups and repeated
median samples. Model loading, prefill, allocation setup, and random generation
were outside timed regions.

Nsight Compute 2026.1.1 was available, but the driver denied performance
counter access with `ERR_NVGPUCTRPERM`. Consequently, DRAM throughput and
achieved occupancy are not claimed. The profile uses PyTorch CUDA device
traces, exact source-level byte accounting, and `cuobjdump` compiler resource
usage instead.

## Baseline characterization

The old stage-1 grid was `(query head, 256-token chunk)`. Since
`kv_head = query_head / 3`, three independent CTAs requested the same K and V
payload for every KV head. This confirms threefold algorithmic K/V reads:

`2 * 9 * L * 64 * 4`, versus the logical minimum `2 * 3 * L * 64 * 4` bytes.

Physical DRAM traffic could be lower because the 48 MiB L2 can satisfy reuse
between query-head CTAs. The nearly flat old 1K-4K latency and weak grouped
gain at 4K are consistent with effective cache reuse, but this is an inference,
not a hardware-counter measurement.

| L | old chunks | old stage-1 CTAs | logical K/V | old requested K/V | old stable-out latency |
|---:|---:|---:|---:|---:|---:|
| 1024 | 4 | 36 | 1.500 MiB | 4.500 MiB | 22.30 us |
| 1280 | 5 | 45 | 1.875 MiB | 5.625 MiB | 23.95 us |
| 2048 | 8 | 72 | 3.000 MiB | 9.000 MiB | 24.54 us |
| 4096 | 16 | 144 | 6.000 MiB | 18.000 MiB | 24.57 us |
| 8192 | 32 | 288 | 12.000 MiB | 36.000 MiB | 37.03 us |

Every long-context invocation has two launches: partial accumulation and
max-rescaled reduction. The old partial kernel used 56 registers/thread,
2,336 bytes shared memory, and no stack/local memory; the reduction used 40
registers/thread, 1,040 bytes shared memory, and no stack/local memory. With
256 threads and 65,536 registers per SM, registers cap the old partial kernel
at four blocks/SM (1,024/1,536 threads, 66.7% theoretical occupancy). Achieved
occupancy was unavailable.

## Experiments

| Candidate | 1024 | 1280 | 2048 | 4096 | 8192 | Decision |
|---|---:|---:|---:|---:|---:|---|
| old query-head, 256t/256 | 22.30 | 23.95 | 24.54 | 24.57 | 37.03 | baseline |
| grouped register reuse, 256t/256 | 29.11 | 27.18 | 27.37 | 28.38 | 31.20 | reject |
| grouped register reuse, 128t/128 | 24.81 | 24.03 | 24.04 | 24.23 | 30.34 | retain only above 4096 |
| grouped shared K/V tile, 256t/128 | 24.43 | 22.17 | 22.36 | 24.53 | 41.06 | reject |
| query-head, 256t/128 | 16.40 | 16.52 | 17.90 | 23.37 | 37.55 | retain through 4096 |
| retained hybrid | 16.40 | 16.52 | 17.90 | 23.37 | 30.34 | retain |

Times are microseconds per stable-output operator call. The single-CTA-per-KV
streaming and explicit vector-load experiments were also timed in a standalone
CUDA module against the retained operator in the same process. These paired
runs used contiguous tensors, no mask, the current Torch CUDA stream, 100
warmups, 30 samples of 100 calls, and CUDA-event medians:

| Candidate | 1024 | 1280 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|---:|
| retained hybrid | 16.00 | 18.03 | 15.53 | 20.78 | 24.80 |
| one streaming CTA/KV head | 233.75 | 292.45 | 465.90 | 927.57 | 1853.46 |
| query-head partial with aligned `float4` K/V | 40.99 | 29.71 | 55.31 | 40.68 | 40.33 |

The streaming design performs one launch and reads K/V only once, but exposes
only three CTAs and requires a block-wide synchronization after each eight
context positions. It used 39 registers/thread, 1,948 bytes shared memory, and
zero stack/local memory, so the 11.7x-74.7x loss is underutilization and
synchronization rather than spilling. The explicit `float4` design preserved
the two-stage algorithm; it used 40 registers/thread, 5,920 bytes shared, and
zero stack/local memory, but its shared transpose and reduction cost made it
1.63x-3.56x slower. Both matched the reference (maximum absolute error at most
`1.12e-7`) and were removed.

The retained grouped kernel uses 48 registers/thread, 4,880 bytes shared
memory, and zero stack/local memory, so it has no spills. The retained
query-head kernel remains at 56 registers/thread and zero spills; reducing the
chunk lowers its shared memory to 1,824 bytes. Scalar K loads are coalesced,
and grouped V loads map adjacent lanes to adjacent dimensions. Registers cap
the 128-thread grouped kernel at ten blocks/SM, or 1,280/1,536 threads (83.3%
theoretical occupancy); achieved occupancy could not be collected.

## Retained traffic

The table reports requested source-level bytes. Workspace reads count the
instructions in stage 2; physical traffic can be smaller through cache hits.

| L | chunks | K/V minimum | retained K/V | Q | mask | workspace W/R | output | estimated total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 8 | 1.500 MiB | 4.500 MiB | 18 KiB | 36 KiB | 18.6/36.8 KiB | 2.25 KiB | 4.609 MiB |
| 1280 | 10 | 1.875 MiB | 5.625 MiB | 22.5 KiB | 45 KiB | 23.2/46.1 KiB | 2.25 KiB | 5.761 MiB |
| 2048 | 16 | 3.000 MiB | 9.000 MiB | 36 KiB | 72 KiB | 37.1/73.7 KiB | 2.25 KiB | 9.216 MiB |
| 4096 | 32 | 6.000 MiB | 18.000 MiB | 72 KiB | 144 KiB | 74.3/147.4 KiB | 2.25 KiB | 18.430 MiB |
| 8192 | 64 | 12.000 MiB | 12.000 MiB | 144 KiB | 288 KiB | 148.5/294.8 KiB | 2.25 KiB | 12.857 MiB |

At 8K this reduces total requested traffic from approximately 36.57 MiB to
12.86 MiB per layer. At 4K the retained win comes from better grid sizing, not
KV traffic reduction; the doubled compact workspace adds only about 146 KiB
of read/write requests per layer.

## Integrated measurements

Comparable pre-change and retained exact-capacity stable CUDA Graph runs:

| capacity | pre-change ms/token | retained ms/token | speedup |
|---:|---:|---:|---:|
| 2048 | 1.9453 | 1.7179 | 1.132x |
| 4096 | 2.0425 | 1.9909 | 1.026x |
| 8192 | 2.4584 | 2.2771 | 1.080x |

A final full-capacity rerun after rebuilding the cleaned retained source gave:

| capacity | replay ms/token | total launches | GQA launches | GQA device ms | total device ms |
|---:|---:|---:|---:|---:|---:|
| 128 | 2.5818 | 1816 | fallback | n/a | n/a |
| 512 | 2.7385 | 1816 | fallback | n/a | n/a |
| 1024 | 1.4929 | 345 | 60 | 0.371 | 1.574 |
| 1280 | 1.5130 | 345 | 60 | 0.386 | 1.587 |
| 2048 | 1.5337 | 345 | 60 | 0.409 | 1.625 |
| 4096 | 1.7950 | 345 | 60 | 0.676 | 1.879 |
| 8192 | 2.2184 | 345 | 60 | 1.055 | 2.260 |

Replay columns are CUDA-event medians; device columns are profiler traces from
separate replays and therefore are not expected to equal the event medians.
The 128/512 rows deliberately retain the existing fallback.

Paired same-process crossover measurements against the fallback were 1.829x,
1.856x, 1.882x, 1.934x, and 1.971x at capacities 513, 640, 768, 1024, and
1280 respectively. Capacity 512 still uses the original shared-memory path.

The retained graph still launches 345 kernels total and 60 GQA kernels (two
per layer). Device-trace GQA totals were 0.371, 0.386, 0.409, 0.676, and 1.055
ms at capacities 1024, 1280, 2048, 4096, and 8192. At 4096 the supplied
0.734 ms baseline becomes 0.676 ms; its share of a 1.879 ms profiled replay is
approximately 36.0%, down from about 40% of the supplied 1.83 ms baseline.

Layer-only CUDA-event medians for the retained fused model were 0.422, 0.438,
0.441, 0.418, 0.400, and 0.234 ms at 512, 1024, 1280, 2048, 4096, and 8192.
Eager full-decode medians were 12.520, 12.236, 12.132, 12.233, 11.854, and
36.729 ms respectively. The 8K DynamicCache measurement includes full-cache
growth/copy behavior; eager movement and Python overhead dominate throughout,
so eager dispatch was not broadened separately. At 8K the allocation-returning
isolated operator was 0.066 ms versus 0.155 ms for the retained PyTorch
attention expression, with maximum/mean absolute error `3.35e-8`/`6.74e-9`.

The 4K GQA workspace grows from 38,016 to 76,032 bytes and total stable scratch
from 53,376 to 91,392 bytes. At 8K those values grow from 76,032 to 152,064 and
from 91,392 to 167,424 bytes. The measured independent graph pool remained
32.188 MiB and the 4K eager incremental CUDA peak remained 3.082 MiB.

## Correctness

The native operator was checked against the retained reference at capacities
128, 512, 1024, 1280, 1281, 2048, 4096, and 8190, including partial device
cache lengths, head-broadcast and head-specific masks, random inputs, extreme
finite FP32 scores, non-default streams, deterministic repeats, graph capture
and replay, poisoned output/workspace buffers, and stable addresses. Existing
model tests cover layer output, logits, complete cache state and lengths,
full-cache behavior, and multi-token greedy generation.

The benchmarked isolated max absolute error was at most `1.56e-7`; mean
absolute errors were `8.55e-9` to `1.74e-8`. Layer max/mean error was at most
`1.91e-6`/`1.79e-7`, logits at most `2.48e-5`/`1.01e-5`, and eight-token greedy
generation matched exactly. Existing tolerances were unchanged.

## Next milestone

The next target should be systematic one-token projection characterization
with cuBLASLt/GEMV algorithm selection. At 4K the remaining projection and LM
head work is larger than any plausible additional GQA gain, while grouped KV
reuse is already limited by lost parallelism and effective cache reuse.
