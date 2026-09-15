# Milestone 37: prefill investigation

## Decision

No production prefill optimization is retained. The current native runtime is
already substantially faster than Hugging Face and Python Flux prefill, and its
long-context attention path is the specialized compact-GQA streaming
implementation retained by Milestones 30 and 36. Attention remains the largest
4096/8192 component, so the *bottleneck-share* half of the bounded-attention
gate passes. The *credible projected gain* half does not: a further 10% stage-1
gain would improve complete prefill by only 5.3% at 4096 and 6.6% at 8192, while
a direct register-pressure prototype made stage 1 more than twice as slow.

The retained changes are measurement/support changes only: the canonical
benchmark records median absolute deviation (MAD) for all prefill paths and
permits zero continuation steps for a strict prefill-only timing run, and the
public inference smoke selects the canonical final Flux category set so its
documented native path is usable. Production C++/CUDA is unchanged.

## Baseline and environment

The investigation used the repository Python 3.11.9 environment, PyTorch
2.14.0+cu132, Transformers 5.16.1, CUDA Toolkit 13.2, driver 616.64, and an
NVIDIA GeForce RTX 5070 Ti (`sm_120`). Inputs were deterministic FP32, TF32 was
disabled, and model loading, runtime construction, graph construction, and
input generation were outside CUDA-event timings.

Before source changes, all 146 Python tests and all 11 native CTests passed.
The public Hugging Face/Flux inference smoke produced identical greedy tokens.
Its optional native branch exposed a pre-existing configuration mismatch: it
enabled the old default subset although the native runtime requires the final
retained set. After selecting the final set, Hugging Face, Flux eager, and
native fixed-length greedy tokens all matched.

The measured starting source baseline exactly matched the expected baseline:

| Language | Bytes |
| --- | ---: |
| Python | 267,004 |
| CUDA (`.cu` + `.cuh`) | 335,035 |
| C++ | 133,693 |
| Headers | 17,899 |
| Total | 753,631 |

The enabled production categories were `cublaslt_projection`,
`fused_gate_up_swiglu`, `gqa_decode_attention`, `mlp`,
`packed_qkv_rope_cache`, `packed_swiglu`, `qkv`, `residual_rmsnorm`,
`rmsnorm`, `rope`, and `softmax`. They installed 30 decoder, attention,
packed-MLP, packed-QKV, GQA-decode, packed-QKV/RoPE/cache,
cuBLASLt-projection, and fused-gate/up modules, plus 61 RMSNorm modules.

## Comprehensive prefill benchmark

These are medians and MADs over 60 CUDA-event samples per path: 5 warmups, 20
samples per round, and 3 rotating rounds. Native throughput uses the complete
30-layer prefill time. `HF/native` and `Flux/native` are speedups.

| Tokens | HF ms (MAD) | Flux eager ms (MAD) | Native ms (MAD) | Native tok/s | HF/native | Flux/native |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 25.100 (0.292) | 12.638 (0.140) | 2.992 (0.194) | 334 | 8.389x | 4.224x |
| 8 | 25.056 (0.215) | 16.237 (0.201) | 3.281 (0.134) | 2,438 | 7.637x | 4.949x |
| 32 | 27.158 (0.275) | 17.610 (0.183) | 4.570 (0.081) | 7,002 | 5.943x | 3.853x |
| 64 | 26.859 (0.197) | 17.283 (0.117) | 4.265 (0.033) | 15,005 | 6.297x | 4.052x |
| 128 | 27.036 (0.171) | 17.477 (0.148) | 5.487 (0.036) | 23,327 | 4.927x | 3.185x |
| 256 | 26.869 (0.201) | 17.170 (0.133) | 8.810 (0.064) | 29,058 | 3.050x | 1.949x |
| 512 | 26.853 (0.232) | 16.289 (0.136) | 12.393 (0.050) | 41,314 | 2.167x | 1.314x |
| 1024 | 34.031 (0.139) | 26.887 (0.209) | 21.727 (0.090) | 47,131 | 1.566x | 1.238x |
| 2048 | 95.378 (0.312) | 71.331 (0.343) | 38.617 (0.203) | 53,034 | 2.470x | 1.847x |
| 4096 | 303.801 (5.263) | 229.017 (5.852) | 97.364 (1.137) | 42,069 | 3.120x | 2.352x |
| 8192 | 1,122.987 (18.620) | 972.409 (16.825) | 267.563 (6.824) | 30,617 | 4.197x | 3.634x |

The geometric-mean native speedups across the 11 lengths are 3.979x versus
Hugging Face and 2.684x versus Python Flux.

Strict prefill checks passed at every length: checkpoint/state-dict identity,
finite logits, existing combined elementwise tolerance, exact positions and
lengths, stable addresses, direct compact caches, and greedy identity. The
known deterministic long-context cache reordering was reported, not hidden:
the largest K/V deltas were `7.9155e-5`/`5.8174e-5` at 4096 and
`8.73566e-4`/`4.33683e-4` at 8192.

The ordinary eight-step continuation validation also exposed one sparse 4096
edge: two of 49,152 logits exceeded the unchanged elementwise tolerance after
a decode step, with maximum absolute difference `3.37362e-5` at a near-zero
element. No tolerance was changed and the run was not reported as passing.
The prefill timing matrix used zero continuation steps so this separate decode
edge could not discard valid prefill measurements. Fresh fixed-length
generation still matched exactly at prompt lengths 128, 1024, and 4096.

## Profile and ranked bottlenecks

The standard profiler used 5 warmups, 10 independent CUDA-event samples, and
10 profiled repetitions. It recorded zero allocation growth, stable addresses,
and 609/399/369/369 launches at 512/2048/4096/8192. A five-repetition
launch-order diagnostic separated the otherwise generically named cuBLAS
kernels. Times below are measured device times; `gap` is the measured profiled
envelope minus summed device-event time and includes launch/dispatcher gaps and
profiler overhead.

| Component (ms) | 512 | 2048 | 4096 | 8192 |
| --- | ---: | ---: | ---: | ---: |
| Attention stage 1 | n/a | 13.910 | 49.094 | 160.224 |
| Attention merge | n/a | 0.156 | 0.288 | 3.529 |
| QK score GEMMs | 0.732 | n/a | n/a | n/a |
| Scale/mask/softmax | 0.495 | n/a | n/a | n/a |
| Attention-value GEMMs | 1.465 | n/a | n/a | n/a |
| Packed gate/up projection | 2.840 | 9.562 | 19.542 | 40.024 |
| Down projection | 2.172 | 5.808 | 10.741 | 20.714 |
| QKV projection | 1.236 | 3.296 | 6.273 | 12.827 |
| Output projection | 1.063 | 2.222 | 4.162 | 7.428 |
| SwiGLU | 0.164 | 0.580 | 2.034 | 5.676 |
| RoPE + direct cache write | 0.120 | 0.537 | 1.010 | 2.714 |
| Residual + RMSNorm | 0.115 | 0.395 | 0.837 | 2.116 |
| Attention layout conversion | 0.088 | 0.283 | 0.545 | 1.279 |
| Residual add | 0.070 | 0.353 | 0.702 | 1.525 |
| Input RMSNorm | 0.113 | 0.234 | 0.410 | 0.823 |
| Profiled envelope | 13.246 | 38.915 | 96.881 | 260.353 |
| Device-event sum | 10.883 | 37.541 | 95.848 | 259.115 |
| Gap | 2.363 | 1.374 | 1.033 | 1.238 |

The ranked long-context bottlenecks are: (1) streaming attention stage 1,
(2) packed gate/up GEMM, (3) down-projection GEMM, (4) QKV GEMM, (5) output
GEMM, then SwiGLU and cache/RoPE work. At 512, packed gate/up and down
projections rank ahead of total attention processing.

## Bounded-prefill-attention gate

Attention includes QK + score processing + P@V at 512 and stage 1 + merge at
long context. The removal ceiling is an Amdahl-law bound, not a projection.
The realistic column assumes a further 10% reduction of long-context stage 1.

| Length | Total profile | Attention | Share | Removal ceiling | 10% stage-1 gain gives | Stage-1 gain required for 1.10x end-to-end |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 13.246 ms | 2.692 ms | 20.3% | 1.255x | n/a | n/a |
| 2048 | 38.915 ms | 14.066 ms | 36.1% | 1.566x | 1.037x | 25.4% |
| 4096 | 96.881 ms | 49.382 ms | 51.0% | 2.040x | 1.053x | 17.9% |
| 8192 | 260.353 ms | 163.752 ms | 62.9% | 2.695x | 1.066x | 14.8% |

Thus attention exceeds the roughly 25% share threshold at 2048, 4096, and
8192, but no tested small change supports a credible 10% end-to-end projection.
The bounded-attention gate is not fully met for a new production subsystem.

## Actual attention execution

The runtime is batch-one FP32 with nine query heads, three K/V heads, three
query heads per K/V head, head dimension 64, hidden size 576, and 30 layers.
All buffers are contiguous:

- packed QKV is `[1,S,960]`;
- Q and attention output are `[9,S,64]` with strides `[S*64,64,1]`;
- the owned cache is `[30,3,capacity,64]`; each layer passes a direct
  `[3,capacity,64]` slice with strides `[capacity*64,64,1]`;
- split partial state is logically `[9,S,P,66]`, where `P=2` through 4096 and
  `P=4` above 4096;
- the bounded fallback alone stores `[9,S,S]` scores, capped at 36 MiB at 1024.

The packed-QKV/RoPE kernel rotates Q and K and writes compact K/V directly into
the decode-owned cache. There is no `repeat_kv`, expanded K/V tensor, Python
cache construction, or cache handoff copy. Causality is implicit: the fallback
softmax derives the valid range from the query row, while streaming kernels
never visit future keys. Long contexts use online FP32 `(m,l,o[64])` state over
32-key shared-memory K/V tiles, followed by a stable two- or four-partition
merge; scores and probabilities are never materialized.

Each long-context layer issues four `cublasSgemm` projection calls (QKV,
output, gate/up, and down) and two attention kernels. The bounded fallback
adds three grouped strided-batched QK calls, one causal scale/mask/softmax
kernel, and three grouped strided-batched P@V calls per layer. All calls use
the caller's current PyTorch CUDA stream.

## Candidate evaluation

Previously measured alternatives already reject multi-Q-head CTA sharing,
larger key tiles, tilewise probability storage, eight context partitions, and
a 64-thread split kernel. Four partitions at 4096 offered only about a 5--6%
whole-prefill projection and worsened the established cache-drift envelope; at
8192 the retained four-partition path already beats the tested alternatives.

The new minimal prototype applied `__launch_bounds__(128,4)` to split stage 1,
attempting to reduce its approximately 162-register footprint and raise
residency from three to four CTAs/SM. It was rebuilt into the actual Python
extension and measured through the production native-prefill API:

| Length | Baseline prefill | Prototype prefill | Baseline stage 1 | Prototype stage 1 | Result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2048 | 38.595 ms | 60.858 ms | 13.955 ms | 36.124 ms | 57.7% slower prefill |
| 4096 | 96.918 ms | 162.133 ms | 49.527 ms | 118.483 ms | 67.3% slower prefill |
| 8192 | 260.286 ms | 471.595 ms | 160.491 ms | 375.692 ms | 81.2% slower prefill |

The compiler's forced register reduction causes spill/local-memory cost that
overwhelms the occupancy benefit. The prototype was removed, the extension was
rebuilt again, and restored medians were 38.657/97.222/259.781 ms. No
experimental CUDA remains.

## Decode/generation protection and validation

Fresh steady one-token decode medians after restoring the baseline were:

| Effective length | HF eager | Flux eager | Native CUDA Graph |
| ---: | ---: | ---: | ---: |
| 128 | 23.5764 ms | 11.5164 ms | 1.4438 ms |
| 1024 | 24.1756 ms | 12.0927 ms | 1.4374 ms |
| 4096 | 23.4916 ms | 12.0729 ms | 1.6778 ms |
| 8192 | 23.7755 ms | 12.0994 ms | 1.9608 ms |

The runtime audit remained 304 launches/replay (183 Flux, 121 cuBLAS, zero
framework), zero allocation growth, stable addresses, and zero allocator
events. Thirty-two-token generation at prompts 128, 1024, and 4096 preserved
exact Hugging Face/Flux/native token identity. Device-native decode medians were
1.5871, 1.4957, and 1.9353 ms/token respectively. Generation remains
fixed-length; EOS early termination is not implemented.

Final validation commands and results:

```text
build\python3119\python.exe -m pytest -q
146 passed

scripts\native.cmd test
11/11 passed

build\python3119\python.exe scripts\flux_inference.py --device cuda --native \
  --max-new-tokens 8 --warmup 2 --iterations 5
Hugging Face == Flux eager == native greedy tokens
```

The final recursive source metric is 754,388 bytes: Python 267,761 B, CUDA
335,035 B, C++ 133,693 B, and headers 17,899 B. The +757 B delta is entirely
Python benchmark/smoke support; production CUDA/C++ and headers are unchanged.

## Result and next task

Files retained by Milestone 37 are `benchmarks/benchmark_flux.py`,
`scripts/flux_inference.py`, and this report. The pre-existing untracked native
generation audit remains separate. This is a meaningful measurement milestone
but not an optimization milestone, so it warrants a milestone commit only as
an investigation/support result, not as a claimed performance improvement.

The next evidence-supported task is not another generic prefill-attention
rewrite. If long-context prefill optimization is resumed, first use Nsight
Compute to quantify stage-1 spills, instruction mix, memory throughput, and
tail efficiency, then prototype an algorithmic reduction in per-thread
`o[64]` live state without imposing a compiler register cap. Such work should
proceed only with a credible path to at least a 15% stage-1 gain at 8192 or an
equivalent representative 5%+ complete-prefill improvement.
