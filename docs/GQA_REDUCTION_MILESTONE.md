# Retained long-context GQA reduction optimization

Date: 2026-09-13

## Decision

Retain one bounded change to the FP32 one-token grouped-GQA path: the final
max-rescaled reduction now computes each chunk's rescaling exponential once
and shares it across the 64 output dimensions. The stage-1 chunk kernels,
workspace layout, dispatch contract, launch count, cache reads, current-stream
behavior, and eager/graph integration remain unchanged.

Two same-process alternating CUDA-Graph A/B runs reproduced integrated gains
at capacities 2048, 4096, and 8192. The primary long-context improvements were
0.0315--0.0366 ms/token at 4096 and 0.0598--0.0789 ms/token at 8192. A
coalesced 128-thread stage-1 experiment was removed because its isolated 4096
operator win became a 0.1188 ms/token integrated graph regression.

## Baseline kernel structure

The native operator has three capacity regimes:

- Through capacity 512, one 256-thread CTA handles each query head using a
  complete shared-memory score row.
- From 513 through 4096, stage 1 launches one 256-thread CTA per
  `(query head, 128-token chunk)`. Each CTA computes QK scores, a local stable
  softmax maximum and sum, and 64 unnormalized weighted-V partials.
- Above 4096 for the SmolLM2 9:3:64 shape, stage 1 launches one 128-thread CTA
  per `(KV head, 128-token chunk)`. It reuses each K/V fetch across the three
  query heads mapped to that KV head.

The long-context workspace is contiguous
`[batch, query_head, chunk, head_dim + 2]`. Each 66-float record contains a
chunk maximum, exponential sum, and 64 weighted-V partials. Stage 2 launches
one 64-thread CTA per query head, finds the global maximum, rescales all chunk
sums and output partials, normalizes, and writes the existing stable output.
Every long-context call therefore remains two launches; the 30-layer graph has
60 GQA launches and 315 total launches.

At 4096, stage 1 has 288 CTAs and requests approximately 18.0 MiB of K/V data
plus 74.3 KiB of workspace writes. At 8192, grouped stage 1 has 192 CTAs and
requests the logical-minimum 12.0 MiB of K/V data plus 148.5 KiB of workspace
writes. Physical DRAM traffic can be lower where L2 supplies repeated
query-head reads.

`cuobjdump --dump-resource-usage` reports 56 registers/thread and 1,824 bytes
shared memory for the query-head chunk kernel, 48 registers/thread and 4,880
bytes shared memory for the grouped chunk kernel, and no local memory or
stack for either. The final reduction uses 40 registers/thread, 1,296 bytes
shared memory, and no local memory or stack. Nsight Compute 2026.1.1 again
reported `ERR_NVGPUCTRPERM`, so achieved occupancy and hardware DRAM counters
are not claimed.

## Measured bottleneck

The fresh allocation-free profiler split showed that stage 1 remains the
dominant native-GQA cost. Before this change it accounted for about 85% of the
profiled operator kernel time at 4096 and 78% at 8192. The old reduction grew
from 1.533 microseconds at 1024 to 5.292 microseconds at 8192 because every one
of 64 output threads reread each chunk maximum and independently recomputed
the same `exp(chunk_max - global_max)`.

For `N` chunks and nine heads, the old reduction requested `131*N*9` FP32
workspace reads and evaluated `65*N*9` exponentials. The retained version
requests `67*N*9` FP32 global reads and evaluates `N*9` exponentials. Its
256-byte shared scale table changes the total compiler-reported shared memory
from 1,040 to 1,296 bytes without changing its 40-register count.

Repeated profiler averages per allocation-free operator call were:

| capacity | chunk before us | reduce before us | kernel sum before us | chunk after us | reduce after us | kernel sum after us |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 7.704 | 1.533 | 9.237 | 7.686 | 1.401 | 9.087 |
| 2048 | 8.478 | 2.025 | 10.503 | 8.506 | 1.626 | 10.132 |
| 4096 | 17.325 | 3.081 | 20.406 | 15.530 | 2.239 | 17.769 |
| 8192 | 19.218 | 5.292 | 24.510 | 17.712 | 3.449 | 21.161 |

Stage 1 did not change; its before/after variation reflects separate profiler
runs. The reduction improvement is monotonic and reaches 1.843 microseconds,
or 34.8%, at 8192. Separately run CUDA-event operator medians were noisy at
4096, but improved from 25.982 to 24.005 microseconds at 8192. This is why the
integrated alternating A/B result, rather than a standalone sequential run,
is the retention criterion.

## Experiments

### Retained: shared chunk rescaling

The global maximum is still accumulated serially by thread zero in the same
chunk order. One thread per chunk then computes the rescaling exponential into
shared memory. Thread zero accumulates the global sum in the same order, and
each output thread consumes the shared scale. This preserves the FP32
reduction ordering while eliminating redundant exponentials and global reads.

### Rejected: 128-thread coalesced-V query-head chunk

This variant mapped adjacent lanes to adjacent V dimensions, used two
position groups per dimension, and combined them through a 512-byte shared
partial tile. Against the retained stage 1 plus the new reduction, isolated
CUDA-Graph operator medians changed from 10.948 to 14.345 microseconds at
1024, 12.326 to 14.660 at 2048, and 20.631 to 17.945 at 4096. Although 4096
looked promising in isolation, a same-process alternating full-graph test
changed 1.5634 to 1.6823 ms/token, a 0.1188 ms/token regression. The variant
was removed completely.

Larger chunk, explicit vector-load, grouped-at-4096, and one-CTA streaming
architectures were not repeated: the preceding long-context milestone already
measured and rejected them for lost parallelism, synchronization, or shared
transpose cost. The present evidence did not justify reopening those designs.

## Integrated graph results

Each comparison below captured baseline and candidate graphs in the same
process, alternated replay order, used 50 warmups and 300 CUDA-event samples,
and excluded model loading, capture, allocation, and random generation. The
temporary baseline selection used for the experiment was removed from the
final source and the final extension was rebuilt.

| capacity | A/B run 1 before/after ms | A/B run 2 before/after ms | repeated long-context delta |
|---:|---:|---:|---:|
| 1024 | 1.4942 / 1.5521 | 1.4199 / 1.4119 | mixed; run 1 was a clock/system outlier |
| 2048 | 1.4587 / 1.4446 | 1.4610 / 1.4448 | -0.0141 / -0.0163 ms |
| 4096 | 1.5665 / 1.5350 | 1.5838 / 1.5473 | -0.0315 / -0.0366 ms |
| 8192 | 2.2484 / 2.1886 | 2.2284 / 2.1495 | -0.0598 / -0.0789 ms |

Absolute clocks varied across independent processes, including the supplied
1.4074/1.4508/1.6766/2.0611 ms baseline, but alternating comparisons removed
that drift at the decision points. At 1024 the isolated reduction itself fell
from 1.533 to 1.401 microseconds, and the second integrated run improved by
0.0080 ms; the first integrated pair was discarded as an outlier because both
graphs shifted far outside the other retained-system measurements. There is
no material retained 1024 or 2048 regression.

The final repeated component profile measured native GQA at 0.6520 ms/token
for capacity 4096 and 1.0104 ms/token for 8192, versus the fresh baseline's
0.6784 and 1.0364 ms/token. Launches remain 60 GQA and 315 total per replay.

## Correctness and validation

The retained source was rebuilt after all experimental paths and benchmark
switches were removed. Existing tests exercise operator correctness at short
and long capacities, partial device cache lengths, broadcast and per-head
masks, extreme finite values, non-default streams, poisoned stable buffers,
stable storage, CUDA-Graph capture/replay, cached layer/logit/cache behavior,
and deterministic greedy generation. Tolerances were unchanged.

```powershell
$env:FLUX_BUILD_NATIVE='1'
build\python3119\python.exe setup.py build_ext --inplace --parallel 8
# succeeded; final flux/_C.cp311-win_amd64.pyd rebuilt

build\python3119\python.exe -m pytest -q `
  tests\test_native_gqa_decode_attention.py `
  tests\test_stable_decode_outputs.py `
  tests\test_smollm2_cuda_graph.py `
  tests\test_smollm2_flux.py
# 100 passed in 5.39s

build\python3119\python.exe -m pytest -q
# 460 passed in 10.10s
```

The final benchmark's maximum absolute operator error was `1.0e-7` at 1024
and at most `4.7e-8` at the longer capacities. Output and workspace shapes,
addresses, allocation behavior, and API contracts did not change.

## Recommendation

The reduction cleanup is worthwhile because it is small, reproducible in
integrated long-context graph decode, and removes clearly redundant work. The
remaining cost is overwhelmingly stage 1, where Flux already balances query-
head parallelism against 3:1 KV reuse. This investigation's coalescing result,
together with the earlier streaming/grouping/chunk experiments, indicates that
another GQA-only round is not justified without access to hardware counters or
a materially different design with a credible integrated advantage. Flux
should proceed to the final integrated-system milestone rather than continue
local GQA tuning.
