# Native greedy-generation profiling audit

Date: 2026-09-15

## Decision

The retained native greedy-generation control path does not justify another
optimization milestone.  It has one Python/custom-class call per generated
sequence, no Python work per post-prefill token, no per-token allocation or
copy, no device-to-host scalar readback, and no host synchronization inside the
generation call.  Over a 127-replay generation, its regular unprofiled host
submission cost is about 0.0034--0.0038 ms/token, or roughly 0.2% of important
long-context decode latency.

No production optimization was attempted or retained.  The device computation,
not the native replay loop, remains limiting.  Grouped GQA is the only component
large enough to support a theoretical five-percent integrated improvement at
long context, but previous milestones already tested and rejected several stage-1
alternatives.  This audit found no new measured candidate that meets the
retention gate.  The existing roadmap's Milestone 37 prefill investigation
therefore remains next; there is no new greedy-runtime milestone.

## Baseline

The clean baseline was `fde378a` (`Add native greedy generation and optimize
streaming GQA`) with an empty `git status --short`.  Although the task brief
described a post-Milestone-35 state, this commit also contains retained
Milestone 36 split-context prefill GQA.  That newer work was preserved and the
audit was limited to generation.

Retained generation state before the audit:

- prefill computes the first exact greedy token and seeds stable device state;
- each full native decode graph ends with exact greedy argmax and state update;
- one C++ loop issues `N - 1` graph launches on PyTorch's current CUDA stream;
- one completion event is recorded after the loop;
- the API returns a view of stable CUDA token storage; and
- generation is fixed length.  EOS early stopping is not implemented and is
  explicitly deferred by the Milestone 35 contract.

The exact tracked-source baseline was:

| Language | Bytes | Share |
|---|---:|---:|
| Python (`.py`) | 267,004 | 35.429% |
| CUDA (`.cu` + `.cuh`) | 335,035 | 44.456% |
| C++ (`.cpp` + `.cc` + `.cxx`) | 133,693 | 17.739% |
| Headers (`.h` + `.hpp`) | 17,899 | 2.375% |
| Total | 753,631 | 100.000% |

CUDA remains 83,561 bytes short of equal CUDA/non-CUDA source under the
repository metric.  The audit adds no counted source bytes.

## Validation baseline

| Command | Result |
|---|---|
| `build\python3119\python.exe -m pytest -q` | 146 passed in 15.89 s |
| `scripts\native.cmd test` | rebuilt/configured; 11/11 CTests passed in 3.38 s (8 CUDA, 3 CPU) |

The native tests cover ATen argmax equivalence, lowest-index ties, all-negative
and extreme values, graph state updates, stable allocation/address behavior,
and non-default/current-stream execution.  The full Python suite includes the
FakeTensor/opcheck and checkpoint-backed integration coverage.

Before timing, the audit independently compared the previous Python-controlled
native path and device-native path with pinned Hugging Face greedy generation.
Both were exactly equal for 128 generated tokens at prompt lengths 128, 512,
1024, 2048, and 4096.  Both were also exactly equal for the one-token boundary
at prompt length 8192.  Every timed shorter result was checked against the
corresponding Hugging Face prefix.

## Method

Measurements used the repository's verified Python 3.11.9 environment,
PyTorch 2.14.0+cu132, CUDA 13.2, and NVIDIA GeForce RTX 5070 Ti.  Deterministic
FP32 settings were enabled and TF32 was disabled.  Prompt creation, model
loading, and correctness checks were outside timing.

Complete-generation measurements used seven samples in rotating native-device
and native-Python order.  Values are median +/- median absolute deviation
(MAD).  Complete time starts immediately before native runtime construction,
includes graph setup/capture and the initial prefill, includes synchronized
post-prefill generation, and ends after the final CUDA `torch.cat` that forms
the ordinary prompt-plus-generation tensor.  It therefore matches the public
helper boundary apart from already-created prompt/model inputs.

Post-prefill decode was enclosed by CUDA events and synchronized only at the
final event.  The device-native measurement calls `generate_greedy` once.  The
comparison path retains the available prior behavior: Python calls one native
replay, ATen argmax, and token installation per post-prefill token.

Steady-state decode used 10 warmups and 21 CUDA-event samples.  Each row spans
the 21 effective contexts ending at the reported context.  The profiler used
five natural graph replays per context and inserted no component events or
synchronization.

## Native complete-generation latency

All values are milliseconds, median +/- MAD.  The length-8192 model boundary
supports the prefill-selected first token but no post-prefill replay, so only
the one-token result is valid there.

| Prompt | 1 output | 8 outputs | 32 outputs | 128 outputs |
|---:|---:|---:|---:|---:|
| 128 | 12.174 +/- 0.206 | 22.702 +/- 0.046 | 59.515 +/- 0.103 | 219.430 +/- 0.284 |
| 512 | 18.876 +/- 0.131 | 29.516 +/- 0.191 | 64.756 +/- 0.118 | 205.116 +/- 0.161 |
| 1024 | 28.553 +/- 0.634 | 38.748 +/- 0.143 | 73.719 +/- 0.106 | 214.874 +/- 0.214 |
| 2048 | 45.171 +/- 0.407 | 55.559 +/- 0.029 | 91.498 +/- 0.148 | 235.972 +/- 0.380 |
| 4096 | 102.053 +/- 0.230 | 115.019 +/- 0.156 | 160.457 +/- 0.075 | 342.788 +/- 0.636 |
| 8192 | 260.839 +/- 0.476 | unsupported | unsupported | unsupported |

The first token is produced by prefill, so the one-output rows contain no
decode replay.  Their approximately 0.04--0.06 ms event intervals are event and
call-boundary floors, not token-decode work.

## Post-prefill native decode

CUDA-event milliseconds per post-prefill token, median +/- MAD:

| Prompt | 8 outputs (7 replays) | 32 outputs (31 replays) | 128 outputs (127 replays) |
|---:|---:|---:|---:|
| 128 | 1.5160 +/- 0.0017 | 1.5333 +/- 0.0063 | 1.6302 +/- 0.0006 |
| 512 | 1.4960 +/- 0.0022 | 1.4693 +/- 0.0014 | 1.4646 +/- 0.0007 |
| 1024 | 1.5041 +/- 0.0050 | 1.4759 +/- 0.0016 | 1.4701 +/- 0.0016 |
| 2048 | 1.5470 +/- 0.0023 | 1.5145 +/- 0.0031 | 1.5062 +/- 0.0018 |
| 4096 | 1.9287 +/- 0.0012 | 1.9081 +/- 0.0038 | 1.9025 +/- 0.0015 |

Longer output sequences advance through different contexts.  In particular,
the prompt-128 sequence crosses the short-capacity boundary, and the
prompt-4096 sequence runs at capacities/effective contexts above 4096.  The
steady-state context table is the cleaner fixed-context comparison:

| Ending context | Sample context window | Median +/- MAD ms/token |
|---:|---:|---:|
| 128 | 108--128 | 1.4331 +/- 0.0141 |
| 512 | 492--512 | 2.2638 +/- 0.0353 |
| 1024 | 1004--1024 | 1.4237 +/- 0.0095 |
| 2048 | 2028--2048 | 1.4546 +/- 0.0088 |
| 4096 | 4076--4096 | 1.6640 +/- 0.0065 |
| 8192 | 8172--8192 | 1.9463 +/- 0.0146 |

The exact-capacity 512 discontinuity is expected.  Capacities through 512 use
the documented safe one-CTA GQA path; ordinary generation from a 512-token
prompt reserves additional decode capacity and uses the optimized chunked
path, as the multi-output table demonstrates.

## Previous Python-controlled comparison

The device-native path wins every paired post-prefill decode comparison.  The
table shows synchronized CUDA-event decode totals; change is lower latency.

| Prompt | Outputs | Python-controlled ms | Device-native ms | Change |
|---:|---:|---:|---:|---:|
| 128 | 32 | 47.939 | 47.531 | -0.852% |
| 512 | 32 | 45.959 | 45.550 | -0.890% |
| 1024 | 32 | 46.248 | 45.752 | -1.074% |
| 2048 | 32 | 47.508 | 46.950 | -1.175% |
| 4096 | 32 | 59.603 | 59.151 | -0.757% |
| 128 | 128 | 209.160 | 207.029 | -1.019% |
| 512 | 128 | 187.623 | 186.005 | -0.862% |
| 1024 | 128 | 188.124 | 186.704 | -0.755% |
| 2048 | 128 | 192.965 | 191.288 | -0.869% |
| 4096 | 128 | 243.120 | 241.615 | -0.619% |

For 127 replays, native submission took 0.430--0.485 ms per sequence, versus
3.009--3.204 ms for the Python-controlled loop.  This removes about 85% of
host submission work (approximately 3.4--3.8 microseconds versus
23.7--25.2 microseconds per replay).  End-to-end improvement remains below one
percent because graph device execution overlaps submission and dominates the
critical path.

## Profile attribution

`Profiled total` is the CUDA-event interval during the profiler run.  Owner
times are summed device-kernel time; their difference from the event interval
contains launch gaps and profiling/runtime effects.  Independent unprofiled
steady-state medians above are authoritative for latency.

| Context | Profiled total | Launches | cuBLAS | GQA | Flux other | Greedy/state |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1.7170 | 274 | 0.6277 | 0.3626 | 0.5197 | 0.0471 |
| 512 | 2.4810 | 274 | 0.6566 | 1.1712 | 0.5234 | 0.0507 |
| 1024 | 1.5935 | 304 | 0.5954 | 0.3472 | 0.5222 | 0.0504 |
| 2048 | 1.6415 | 304 | 0.6022 | 0.3774 | 0.5203 | 0.0482 |
| 4096 | 1.8799 | 304 | 0.5930 | 0.6116 | 0.5195 | 0.0489 |
| 8192 | 2.1964 | 304 | 0.6026 | 0.8543 | 0.5578 | 0.0503 |

At 8192, grouped GQA stage 1 is 0.7521 ms and its reduction is 0.1023 ms.
The 91 projection/LM-head GEMV launches sum to 0.5710 ms.  Fused gate/up plus
SwiGLU is 0.3171 ms, both norm families sum to 0.1394 ms, packed QKV/RoPE/cache
is 0.0482 ms, greedy/state is 0.0503 ms, and decode state preparation is
0.0027 ms.

## Native-loop audit

A host/runtime trace of one 128-output generation at prompt 4096 observed
exactly 127 `cudaGraphLaunch` calls, one asynchronous
`cudaStreamWaitEvent` ordering prefill, one completion `cudaEventRecord`, and
one each of the return-view `narrow`/`slice`/`as_strided`/`view` operations.
There was no per-token ATen/PyTorch dispatch and no Python callback.  The
trace's command-buffer-full waits were profiler artifacts; the regular
unprofiled submission values above are authoritative.

Source inspection and native tests confirm:

- `CUDAGuard` and `getCurrentCUDAStream(device_index)` select the caller's
  current device/stream;
- the prefill dependency is an asynchronous stream wait, not a host wait;
- the loop contains only checked `cudaGraphLaunch` calls;
- one event is recorded after all launches, not once per token;
- generated-token, next-token, position, cache-length, and generation-step
  updates occur in the captured graph;
- no tensor or CUDA allocation occurs in the loop (the returned tensor view is
  created once per sequence on the host);
- scalar D2H reads and `cudaEventSynchronize` occur only in explicit diagnostic,
  reset, and destruction paths, not generation; and
- setup performs two token-range `.item()` validations, but these are outside
  steady-state generation.

Production generation does not retrieve tokens to the CPU.  Returning the
stable CUDA view triggers no device copy.  Final prompt-plus-token assembly
cost 0.0346--0.0535 ms of GPU time for the whole measured sequence; at 128
outputs this was about 0.00036 ms/token.  The benchmark's one final event wait
is the consumer/timing boundary and mostly waits for device computation; it is
not independent runtime overhead that can be removed from a result-consuming
call.

There is no EOS kernel, scalar EOS readback, or termination check to attribute.
The current API always emits the requested count.  Adding EOS would be new
functionality requiring a device active/output-length contract, not a
preserved behavior of this baseline.

## Ranked bottlenecks

Percentages and ceilings use the 2.1964 ms profiled 8192 interval.  Ceilings
are deliberately theoretical complete removal, not performance forecasts.

| Rank | Cost | Approx. latency / share | Removable? | Complexity | Maximum if removed | Confidence |
|---:|---|---:|---|---|---:|---|
| 1 | Grouped GQA stage 1 + reduce | 0.8543 ms / 38.9% | Partly | High | 1.64x | High attribution; low new-design gain |
| 2 | cuBLAS/cuBLASLt projections | 0.6026 ms / 27.4% | Computation required | Very high | 1.38x | High attribution; low replacement confidence |
| 3 | Fused gate/up + SwiGLU | 0.3171 ms / 14.4% | Partly; already fused | High | 1.17x | High attribution; low further-gain confidence |
| 4 | Non-kernel graph/runtime gaps | about 0.1817 ms / 8.3% | Partly | Medium/high | 1.09x | Low; subtraction includes profiler effects |
| 5 | RMSNorm + residual RMSNorm | 0.1394 ms / 6.3% | Partly | Medium | 1.07x | High attribution; low integrated ceiling |
| 6 | Greedy argmax/state update | 0.0503 ms / 2.3% | Exact selection required | Medium | 1.02x | High |
| 7 | Packed QKV/RoPE/cache update | 0.0482 ms / 2.2% | State work required | High | 1.02x | High |
| 8 | C++/driver graph submission | about 0.0034 ms/token / 0.17% | Partly | Low/medium | about 1.002x | High |
| 9 | Final CUDA output assembly | about 0.00036 ms/token / 0.02% | Optional API choice | Low | about 1.0002x | High |

A 15% GQA improvement would project to roughly 5.8% at the profiled 8192
workload, so it is the only generation component with enough leverage.  That is
not sufficient evidence to implement: the retained GQA path already balances
KV reuse and exposed parallelism, prior coalesced/larger-chunk/grouped variants
were rejected by integrated measurements, and hardware counters remain
unavailable.  A future attempt must begin with a materially different measured
stage-1 candidate and same-process alternating full-graph A/B evidence.

## Final status

No optimization experiment or production source change was retained.  The
temporary measurement drivers and JSON profile were removed.  This document is
the only audit artifact.  A commit is not warranted as a performance milestone;
the working-tree documentation change should remain available for review.
