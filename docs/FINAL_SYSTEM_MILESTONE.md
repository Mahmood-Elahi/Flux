# Final integrated Flux inference system

Date: 2026-09-13

> Historical milestone note: this document records the completed Python-owned
> Flux CUDA-Graph system at Milestone 24. Milestones 26-27 subsequently added a
> lifetime-managed native one-layer runtime and then a complete native
> token-to-logits 30-layer runtime. See `NATIVE_FULL_DECODE_RUNTIME_MILESTONE.md`.
> The separate 50% CUDA source-share release requirement remains unmet, so the
> later native-runtime phase does not declare the overall project complete.

## 1. System overview

Flux is complete as one integrated FP32 CUDA inference path for the pinned
`HuggingFaceTB/SmolLM2-135M` checkpoint. The final system preserves the
ordinary Hugging Face/PyTorch model as its correctness and performance oracle,
installs Flux only through an explicit model-instance adapter, and retains only
optimizations that survived operator, model, and integrated measurements.

The final architecture is frozen. This milestone added a canonical production
category constant and a final benchmark/validation harness; it did not add a
new kernel or reopen a rejected optimization.

## 2. Final architecture

The exact retained configuration is:

```python
from flux.model import FINAL_FLUX_OPERATOR_CATEGORIES, enable_flux_ops

model = enable_flux_ops(
    model,
    operators=FINAL_FLUX_OPERATOR_CATEGORIES,
    fuse_attention_scores=True,
)
```

`FINAL_FLUX_OPERATOR_CATEGORIES` is the following frozen set:

```text
cublaslt_projection
fused_gate_up_swiglu
gqa_decode_attention
mlp
packed_qkv_rope_cache
packed_swiglu
qkv
residual_rmsnorm
rmsnorm
rope
softmax
```

The production path consists of:

- native FP32 RMSNorm and dual-output fused residual + RMSNorm;
- native RoPE and fused attention scaling, additive masking, and softmax for
  multi-token attention;
- checkpoint-compatible packed Q/K/V and gate/up parameter storage;
- native packed SwiGLU;
- native FP32, batch-one, one-query-token grouped-GQA decode over unexpanded
  K/V, including the retained query-head/KV-head hybrid chunking and shared
  final-reduction rescaling;
- fused packed-QKV post-processing that applies Q/K RoPE, writes K/V directly
  into StaticCache, emits compact Q, and advances device-resident cache state;
- measured zero-workspace cuBLASLt configurations for one-token packed-QKV and
  attention-output projections;
- fused one-token packed gate/up GEMV + SwiGLU;
- fixed-shape CUDA-Graph replay with graph-owned stable outputs and scratch,
  stable input/mask/logit/cache addresses, preallocated StaticCache, and
  device-resident position and cache-length state.

Prefill and unsupported shapes take the established eager fallbacks. Flux eager
uses the retained operators that match DynamicCache semantics, including
packed projections, fused score processing, packed SwiGLU, and native GQA. The
fused StaticCache update, selected cuBLASLt output variants, stable scratch, and
fused gate/up GEMV + SwiGLU activate only in the supported graph path.

Capacities 513 through 8192 use the fully optimized stable-scratch graph.
Capacities 512 and below intentionally retain the measured safe graph fallback.
The optimized graph stores three KV heads and never materializes `repeat_kv` or
expanded nine-head K/V.

## 3. Environment

| Item | Final benchmark value |
|---|---|
| Model/revision | `HuggingFaceTB/SmolLM2-135M` / `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` |
| Model shape | 30 layers; hidden 576; intermediate 1536; 9 Q heads; 3 KV heads; head dim 64; vocabulary 49152 |
| Python | 3.11.9 |
| PyTorch | 2.14.0+cu132 |
| CUDA build/toolkit | 13.2 / CUDA 13.2 compiler build 37953736 |
| Transformers | 5.16.1 |
| GPU | NVIDIA GeForce RTX 5070 Ti, compute capability 12.0 |
| Driver | 616.64 |
| Precision | FP32 |
| TF32 | disabled for cuDNN and matmul |
| Determinism | seed 0; deterministic algorithms and deterministic safety filling enabled; cuDNN benchmark disabled |

The retained `flux/_C.cp311-win_amd64.pyd` used for final validation was newer
than the final GQA CUDA source. No dependency or toolchain upgrade was made.

## 4. Correctness methodology and results

The final harness loaded two independent copies of the pinned checkpoint,
enabled Flux only on one, and established exact equality of all exported
state-dict keys and tensors after packing. At every effective length it then
checked:

- reference versus Flux prefill and decode logits using the established
  `rtol=2e-4`, `atol=2e-5` model tolerance;
- an eight-token cached greedy continuation ending at the stated effective
  length for reference, Flux eager, and Flux graph;
- DynamicCache and StaticCache logical lengths and positions on every step;
- complete valid K/V shapes, finite values, and maximum numerical differences;
- stable graph input, position, mask, logit, K/V, and scratch addresses;
- identical greedy token IDs across all three paths.

The full-cache K/V maximum is reported rather than hidden. Packed projection
and RoPE accumulation order can make individual near-zero internal elements
exceed the absolute term of the end-to-end model tolerance at long context, so
the final harness did not loosen that tolerance or mislabel the complete cache
as elementwise identical. Dedicated operator and integrated model tests remain
the cache-content correctness gates.

| Effective length | Max logits abs. | Max K abs. | Max V abs. | Position/length | Stable addresses | Greedy IDs |
|---:|---:|---:|---:|:---:|:---:|:---:|
| 128 | 2.670e-5 | 1.574e-5 | 1.121e-5 | pass | pass | pass |
| 512 | 2.098e-5 | 2.003e-5 | 1.550e-5 | pass | pass | pass |
| 1024 | 2.670e-5 | 5.007e-5 | 2.098e-5 | pass | pass | pass |
| 2048 | 3.242e-5 | 4.482e-5 | 3.481e-5 | pass | pass | pass |
| 4096 | 2.480e-5 | 1.116e-4 | 5.555e-5 | pass | pass | pass |
| 8192 | 8.965e-5 | 1.116e-4 | 5.555e-5 | pass | pass | pass |

The existing suite additionally covers causal and additive masking, partial
device cache lengths, broadcast and per-head masks, extreme finite values,
non-default current CUDA streams, FakeTensor/meta behavior,
`torch.library.opcheck`, poisoned output/workspace buffers, repeated graph
replay, graph exhaustion, and stable-output identity.

Final validation commands and exact results:

```powershell
build\python3119\python.exe -m pytest -q `
  tests\test_smollm2_flux.py tests\test_smollm2_cuda_graph.py `
  tests\test_stable_decode_outputs.py tests\test_native_gqa_decode_attention.py `
  tests\test_native_packed_qkv_rope_cache.py `
  tests\test_native_packed_gate_up_swiglu.py `
  tests\test_native_cublaslt_linear.py
# 127 passed in 5.76s

build\python3119\python.exe -m pytest -q
# 461 passed in 10.12s
```

No tolerance was changed.

## 5. Final benchmark methodology

The canonical run was:

```powershell
build\python3119\python.exe benchmarks\benchmark_final_system.py `
  --stabilization-iterations 100 --warmup 5 --samples 10 --rounds 3 `
  --correctness-tokens 8 --generation-repetitions 3 `
  --json-output build\final_system_results.json
```

The benchmark compares three configurations loaded from identical weights:

- **Reference:** unmodified FP32 Hugging Face/PyTorch eager-attention model;
- **Flux eager:** all final retained categories, DynamicCache, no graph replay;
- **Flux graph:** the same final Flux model with fixed-capacity StaticCache,
  stable scratch, and CUDA-Graph decode.

Deterministic input IDs, precision, masks, cache semantics, and greedy settings
are shared. Prefill uses `use_cache=True` and last-token logits, matching the
generation boundary. CUDA Graph does not change prefill compute, so the Flux
prefill column applies to both Flux eager and Flux graph.

Each primary row uses 100 preliminary stabilization iterations, five warmup
calls per path per round, three rotating same-process rounds, ten CUDA-event
samples per path per round, and the median of all 30 samples. Loading, input
construction, correctness, eager cache setup for the decode window, graph
capture, and random work are outside steady-state decode timing. Decode samples
span the final ten valid attention lengths shown in each row. Graph capacity is
the stated effective length.

Generation uses three rotating same-process repetitions and 32 generated
tokens. Its eager timing includes the model, DynamicCache growth, and greedy
argmax. Graph decode includes input copies, replay, and argmax. Because capture
is prompt-specific in the current runtime, graph TTFT and observed total include
prefill, StaticCache construction, warmup, and capture. An execution-only total
is also retained in the JSON output to separate graph setup from GPU work.

A second independent process repeated every prefill/decode row with ten samples
and one round. Optimized graph latency differed from the primary run by +0.80%,
+0.03%, -0.11%, +0.18%, -0.13%, and -1.53% from 128 through 8192. The tables
below use the more robust 30-sample primary medians, not the faster observation
from either process.

## 6. Prefill results

| Tokens | Reference ms | Flux ms | Reference tok/s | Flux tok/s | Flux speedup |
|---:|---:|---:|---:|---:|---:|
| 128 | 27.139 | 17.468 | 4,716.5 | 7,327.8 | 1.554x |
| 512 | 26.676 | 16.105 | 19,193.2 | 31,790.8 | 1.656x |
| 1024 | 34.099 | 27.251 | 30,030.1 | 37,576.7 | 1.251x |
| 2048 | 95.189 | 71.378 | 21,515.1 | 28,692.3 | 1.334x |
| 4096 | 310.451 | 236.051 | 13,193.7 | 17,352.2 | 1.315x |
| 8192 | 1,104.184 | 962.724 | 7,419.1 | 8,509.2 | 1.147x |

The maximum prefill last-token logit error was `2.861e-5`. Flux helps prefill
through packed projections, fused score processing/softmax, packed SwiGLU,
RoPE, and norms. The one-token graph-only fusions do not participate. At 8192,
quadratic eager attention dominates and reduces the relative benefit.

## 7. Decode results

Lengths are effective attention lengths. Thus the 8192 row ends with 8191
cached tokens plus the new token and stays within the model's 8192-position
contract.

| Effective length | Reference ms / tok/s | Flux eager ms / tok/s | Flux graph ms / tok/s | Eager vs ref | Graph vs ref | Graph vs eager |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 23.5835 / 42.40 | 11.6575 / 85.78 | 2.6801 / 373.12 | 2.023x | 8.799x | 4.350x |
| 512 | 23.7133 / 42.17 | 11.9717 / 83.53 | 2.8651 / 349.03 | 1.981x | 8.277x | 4.178x |
| 1024 | 24.1360 / 41.43 | 11.8486 / 84.40 | 1.4353 / 696.73 | 2.037x | 16.816x | 8.255x |
| 2048 | 23.6006 / 42.37 | 12.0517 / 82.98 | 1.4643 / 682.91 | 1.958x | 16.117x | 8.230x |
| 4096 | 23.2544 / 43.00 | 11.8857 / 84.13 | 1.7072 / 585.77 | 1.957x | 13.622x | 6.962x |
| 8192 | 22.4131 / 44.62 | 12.7623 / 78.36 | 2.0530 / 487.09 | 1.756x | 10.917x | 6.216x |

The 128 and 512 graphs are the documented 1,816-launch fallback, which explains
their discontinuity relative to the optimized 1024 row. A normal capture from
a 512-token prompt with a 32-token generation budget has capacity 543 and uses
the optimized path; an exact capacity of 512 intentionally does not.

An effective length above 8192 remains outside the native GQA contract. The
existing eager diagnostic that starts with 8192 cached tokens therefore asks
for an 8193-token attention operation and deliberately falls back to the
Transformers/PyTorch expanded-K/V path. The final matrix does not disguise that
unsupported operation as an 8192 result.

## 8. End-to-end generation results

Each row generates 32 tokens; decode time covers the 31 cached token steps.
"Observed total" is the user-visible completion time under the current API. For
graph it includes one-time prompt-specific setup/capture; for eager paths it is
prefill plus decode.

| Prompt | Path | Prefill contribution ms | TTFT ms | Decode ms (ms/token) | Observed total ms | Effective generated tok/s |
|---:|---|---:|---:|---:|---:|---:|
| 128 | Reference | 27.377 | 27.377 | 738.689 (23.829) | 766.065 | 41.77 |
| 128 | Flux eager | 17.661 | 17.661 | 358.995 (11.580) | 376.656 | 84.96 |
| 128 | Flux graph | 18.194 | 121.259 | 84.287 (2.719) | 205.546 | 155.68 |
| 1024 | Reference | 37.964 | 37.964 | 744.222 (24.007) | 782.186 | 40.91 |
| 1024 | Flux eager | 28.446 | 28.446 | 370.742 (11.959) | 399.188 | 80.16 |
| 1024 | Flux graph | 27.362 | 67.710 | 46.958 (1.515) | 114.668 | 279.07 |
| 4096 | Reference | 305.491 | 305.491 | 730.581 (23.567) | 1,036.072 | 30.89 |
| 4096 | Flux eager | 222.443 | 222.443 | 371.135 (11.972) | 593.577 | 53.91 |
| 4096 | Flux graph | 223.011 | 261.936 | 62.877 (2.028) | 324.813 | 98.52 |

Observed 32-token generation speedups for Flux graph versus reference are
3.727x, 6.821x, and 3.190x at prompts 128, 1024, and 4096. Flux graph versus
Flux eager is 1.832x, 3.481x, and 1.827x. The prompt-128 graph still wins in
total, but capture increases TTFT from an 18.2 ms prefill contribution to
121.3 ms; setup cost must not be omitted when judging short generations.

## 9. Launch and runtime characteristics

The fresh capacity-4096 audit reproduced:

| Property | Result |
|---|---:|
| GPU launches per token | 315 |
| Flux CUDA launches | 181 |
| cuBLAS/cuBLASLt launches | 92 |
| Remaining framework/state launches | 42 |
| Native GQA launches | 60 (chunk + reduction for each of 30 layers) |
| PyTorch allocated-memory growth across ten replays | 0 bytes |
| Tensor addresses across replay | unchanged |
| CUDA malloc/free events in profile | 0 |

The two CPU synchronization events in the audit trace are deliberate profiler
and final measurement boundaries, not one synchronization per replay. Normal
`graph.replay()` does not read the device cache length back to the host and does
not synchronize. The diagnostic `cache_position` property uses `.item()` and
therefore synchronizes only when the caller explicitly asks for that host
value. Graph setup and benchmark timing boundaries also synchronize by design.

The 315-launch graph consists principally of 61 norm kernels, 30 packed-QKV
projections, 30 fused QKV/RoPE/cache kernels, 60 native-GQA kernels, 30
attention-output projections, 30 fused gate/up+SwiGLU kernels, 30 down
projections, 30 residual adds, and one LM head, plus embedding, position, state,
and small framework work. The exact-capacity 128/512 fallback has 1,816
launches and retains framework cache/attention/MLP work.

Flux eager uses DynamicCache and growing concatenations. The prior retained
profile measured 610 launches at context 128 and 670 at 512--4096, with only
about 1.73--2.37 ms of kernel execution inside an 11 ms token. It allocated and
copied growing cache storage even though no per-token `cudaMalloc`/`cudaFree`
event appeared in the trace. The graph path instead reserves StaticCache once,
writes compact K/V in place, and keeps all mutable decode state on device.

## 10. Final bottleneck analysis

### Why Flux eager remains much slower

This is measured as a host/launch problem, not a failure of the native kernels.
The retained eager profile assigned roughly 78--84% of token time to GPU gaps
between hundreds of launches. Python, dispatcher, allocation bookkeeping, and
DynamicCache growth prevent the GPU kernels from running back-to-back. CUDA
Graph submits the captured 315-launch sequence as one replay and replaces
growing cache construction with stable in-place state, producing the measured
6.2--8.3x advantage over Flux eager in the fully optimized length range.

### Context scaling and native GQA

Graph replay changes from 1.435 ms at 1024 to 2.053 ms at 8192, an increase of
0.618 ms. Projection, norm, MLP, and LM-head dimensions are fixed per token.
The retained component profile measured native GQA at about 0.382 ms at 1024,
0.652 ms at 4096, and 1.010 ms at 8192 after the final reduction optimization.
Its context-dependent K/V scan therefore explains essentially all important
long-context graph scaling.

At 1024--2048 the optimized graph is a mixture of launch latency and
bandwidth-limited one-token matrix/vector work. At 4096--8192 it becomes
increasingly memory-traffic and exposed-parallelism bound in native GQA. The
retained shared-rescaling reduction saves 0.0315--0.0366 ms/token at 4096 and
0.0598--0.0789 ms/token at 8192 in paired integrated runs, but stage 1 remains
the dominant GQA cost.

At short optimized contexts, fused gate/up+SwiGLU, the LM head, packed-QKV,
down/output projections, and norms remain material fixed costs. Exact graph
capacities 128/512 are instead dominated by the fallback's framework cache and
memory work.

### Why LM-head GEMV was not prioritized

The LM head is one bias-free FP32 GEMV over a 108 MiB weight and costs about
0.135 ms through capacity 4096. It is the largest individual remaining library
kernel but is nearly context-invariant and cannot explain long-context scaling.
The best custom `float4` candidate improved the isolated captured operation by
only 0.6%, then changed integrated graph decode from 1.4150 to 1.4224 ms at 4096
and 1.7662 to 1.7770 ms at 8192. It was removed. The weight exceeds L2 and the
operation is already close to a bandwidth-streaming regime, so another LM-head
project was not justified.

## 11. Retained optimization summary

"No isolated claim" means the change was retained on correctness and integrated
evidence recorded by its original milestone, but no trustworthy standalone
speedup is invented here.

| Milestone | Change and reason | Decision | Representative measured benefit |
|---|---|---|---|
| FP32 norms | Warp-reduced RMSNorm and dual-output residual+RMSNorm replace multi-op normalization/residual boundaries | retained | Scalar warp path beat rejected vector path; fused boundary remains part of all integrated results |
| Attention processing | Native RoPE, softmax, and fused scale+mask+softmax reduce eager framework work while preserving masks/positions | retained | No isolated claim combined across these early milestones |
| Packed MLP | Pack gate/up checkpoint weights and fuse their SwiGLU consumer | retained | No isolated claim across combined final system |
| Packed QKV | Replace three projection calls with one packed projection while preserving checkpoint keys | retained | Structural launch/allocation reduction; no standalone final claim |
| Native one-token GQA | Read unexpanded 3-head K/V directly with stable online softmax | retained | Eliminates `repeat_kv`; hybrid long-context update improved exact graphs 1.132x at 2048, 1.026x at 4096, 1.080x at 8192 in its milestone |
| Packed-QKV/RoPE/cache fusion | Fuse post-projection RoPE, StaticCache writes, compact-Q output, and device cache-length advance | retained | Removed the separate one-token boundary without graph-pool growth; no isolated final claim |
| Stable graph outputs | Preallocate five producer outputs/scratch sets to remove deterministic `empty` safety fills | retained | Removed 211 captured fill launches while leaving deterministic safety enabled |
| Tuned projections | Exact zero-workspace cuBLASLt algorithm for QKV/output only | retained | 1.022x full graph at 1024/2048/4096 and 1.015x at 8192 in retained sweep |
| Fused gate/up GEMV+SwiGLU | Directly produce the stable 1536-element activation | retained | 345 to 315 launches; 1.034x at 4096 and 1.023x at 8192 |
| GQA reduction | Share each chunk's max-rescaling exponential across 64 output dimensions | retained | 0.0315--0.0366 ms at 4096; 0.0598--0.0789 ms at 8192 |
| Final integration | Freeze all above categories and use fixed-shape graph replay | retained | 16.816x reference speedup at 1024; 13.622x at 4096; 10.917x at 8192 |

## 12. Rejected optimization summary

| Candidate | Why attempted | Measured result | Final disposition |
|---|---|---|---|
| RMSNorm aligned `float4` | Increase memory transaction width | Largest `(1,8192,576)` case regressed 29.502 to 37.166 us; 128-thread vector variant was 38.106 us | removed; scalar warp reduction retained |
| Residual-accumulating projections | Remove projection temporary/read and residual-add boundary | Attention graph 0.986x, MLP 0.970x, combined 0.962x at 4096 because `addmm` selected extra split-K work | removed |
| GQA larger chunks/streaming/early grouping | Reduce workspace or K/V rereads | Lost parallelism, synchronization, or transpose costs; grouping retained only above 4096 | rejected variants removed |
| Coalesced 128-thread GQA stage 1 | Improve adjacent V access at 4096 | Isolated 4096 win, but full graph regressed 1.5634 to 1.6823 ms/token | removed |
| Standalone/shared/vector gate/up GEMVs | Replace library one-token gate/up | Several variants regressed; standalone direct helped, but the fused consumer was better | standalone candidates removed; fused kernel retained |
| Custom LM-head GEMV | Target largest individual library kernel | Best isolated candidate 1.006x; integrated 0.995x at 4096 and 0.994x at 8192 | removed |

These results are the intended evidence of measurement-driven selection: an
isolated kernel win was not sufficient when the full decode graph regressed.

## 13. Limitations

- The system and native optimized operators are FP32 and inference-only; there
  is no backward/training path.
- The most specialized graph path targets SmolLM2-135M, batch size one, one
  decode token, head dimension 64, and fixed cache capacity 513--8192.
- Exact capacities through 512 use the safe captured fallback. Unsupported
  shapes and effective lengths above 8192 use established eager/framework
  fallbacks where model position semantics allow them.
- CUDA-Graph state is fixed-shape and prompt-specific. Capture/setup has to be
  amortized; it materially increases TTFT for short generations.
- The selected cuBLASLt algorithms are empirical for the documented target
  stack and GPU. Other hardware/software stacks must revalidate them.
- Prefill remains eager quadratic attention and becomes the dominant cost at
  long prompts.
- The LM head still computes the full 49,152-token FP32 logits vector.
- Nsight Compute hardware counters were unavailable because of
  `ERR_NVGPUCTRPERM`; bandwidth/occupancy classifications are based on logical
  traffic, compiler resources, scaling, and profiler timing, not claimed DRAM
  counter values.

## 14. Final conclusion

Flux now has one coherent final state: an opt-in, checkpoint-compatible
SmolLM2-135M system whose reference, eager, and graph paths are tested together
from 128 through the valid 8192-token boundary. The optimized graph is
allocation-free in steady replay, carries compact K/V and cache state in stable
device storage, removes the dominant eager host gaps, and reaches 1.435--2.053
ms/token across the fully optimized context range. Long-context performance is
now governed primarily by the unavoidable native-GQA K/V scan, while the
remaining fixed work has been bounded and measured. No speculative next
milestone is added.
