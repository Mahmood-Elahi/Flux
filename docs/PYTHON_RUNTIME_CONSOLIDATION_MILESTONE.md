# Native runtime and Python consolidation milestone

> Historical milestone 31 report. Milestone 32 subsequently consolidated the
> separate native-prefill benchmark into `benchmark_flux.py --mode system`; see
> `PYTHON_INTEGRATION_LAYER_MILESTONE.md` for the current inventory and counts.

## Outcome

This milestone makes native prefill plus the attached full-model native decode
runtime the canonical optimized Flux backend. It adds no kernel, precision,
model support, sampling path, or mathematical change. The pinned Hugging Face
model and Flux eager path remain independent correctness oracles.

The historical Python-owned CUDA Graph backend and one-layer Python development
adapter are retired. Their last complete revision is canonical commit
`0cb266939d7e26939fa5aaec6337bf261eee551c`; the measurements that justified
their retained operator choices remain in the milestone documents named below.
Git history is the archive for reproducing the deleted implementations.

## Oracle migration

`test_native_smollm2_runtime.py` no longer imports or compares with
`FluxCUDAGraphDecode`. It constructs independent Hugging Face and Flux model
instances with exactly equal weights, advances their `DynamicCache` objects
beside three native replays, and checks:

- logits against both independent paths at the unchanged `rtol=2e-4`,
  `atol=2e-5` policy;
- exact greedy token identity;
- all 30 complete valid K/V prefixes;
- cache position and length;
- stable returned-logits identity;
- stable native addresses and capacity exhaustion.

`test_native_smollm2_prefill_runtime.py` retains prompt logits, every-layer K/V,
direct prefill-to-decode handoff, cache shape/position/length, input validation,
prefill reuse, and expands its greedy check to eight tokens against Hugging
Face generation. The focused migration command passed 6/6 tests before any
legacy source was removed. The Python graph is therefore no longer a test or
benchmark oracle.

## Retired implementations and benchmarks

The following production-obsolete files are removed:

- `flux/model/smollm2_cuda_graph.py`;
- `flux/runtime/native_smollm2_layer.py`;
- `tests/test_smollm2_cuda_graph.py`;
- `tests/test_stable_decode_outputs.py`;
- `tests/test_native_smollm2_layer_runtime.py`;
- `benchmarks/benchmark_smollm2_gqa_decode_attention.py`;
- `benchmarks/benchmark_smollm2_gate_up_gemv.py`;
- `benchmarks/benchmark_rope.py`;
- `benchmarks/benchmark_native_smollm2_runtime.py`.

The reusable native layer executor and native class implementation remain in
the native decode translation unit because the full 30-layer runtime executes
that implementation. Only its redundant public Python development adapter is
removed.

Historical measurements remain in:

- `FINAL_SYSTEM_MILESTONE.md` and `DECODE_PROFILE_MILESTONE.md` for the old
  Python graph;
- `NATIVE_DECODE_RUNTIME_MILESTONE.md` for the one-layer runtime;
- `NATIVE_FULL_DECODE_RUNTIME_MILESTONE.md` for the full native decoder;
- `GQA_LONG_CONTEXT_MILESTONE.md` and `GQA_REDUCTION_MILESTONE.md` for GQA;
- `GATE_UP_GEMV_MILESTONE.md` for fused gate/up;
- `NATIVE_VALIDATION_MILESTONE.md` for current native microbenchmarks;
- `NATIVE_PREFILL_RUNTIME_MILESTONE.md` and
  `STREAMING_PREFILL_GQA_MILESTONE.md` for native prefill.

README commands now point to native microbenchmark filters or retained system
benchmarks. No active source or README command imports or names a deleted path.

## Coverage replacement table

| Removed Python coverage | Replacement | Why equivalent or stronger |
| --- | --- | --- |
| Complete `FluxCUDAGraphDecode` implementation and constructor/capture errors | `NativeSmolLM2Prefill`, attached `NativeSmolLM2Decode`, focused Python runtime tests, `flux_test_native_runtimes` | The production object owns prompt execution, all 30 layers, compact cache, device state, stable storage, graph capture/replay, final norm, LM head, and lifecycle directly. |
| Old-graph multi-step logits, K/V, tokens, and capacity in `test_smollm2_cuda_graph.py` | Independent HF/Flux-eager multi-step validation in `test_native_smollm2_runtime.py` and native-prefill generation validation | Uses independent model/cache oracles rather than native-vs-native comparison and checks all 30 cache layers. |
| Old-graph category A/B tests | Current full native integration tests; `flux_test_decode_ops`; canonical final benchmark | Retained categories are exercised together in production; direct operator math is checked at the launcher boundary. Historical selection A/B data remains documented. |
| Python scratch installation and short-capacity fallback tests | No replacement | These assertions described private behavior of the intentionally removed backend, not a production contract. |
| Stable-output repeated overwrite and stream tests | Per-operator Python out identity/poison/alias/schema checks; `flux_test_rmsnorm`, `flux_test_residual_rmsnorm`, `flux_test_transformer_ops`, `flux_test_decode_ops` | Python retains dispatcher-visible contracts; native tests check repeated overwrite and current-stream execution directly. |
| Stable Python graph scratch replay/address/allocation tests | `flux_test_native_runtimes` plus native-vs-HF full-runtime tests and canonical runtime audit | Native CTest owns the actual production graph's address, replay, state, stream, allocation, and lifecycle invariants. |
| One-layer learned comparison and lifecycle tests | All-layer HF/Flux/native cache comparison; `flux_test_native_runtimes`; `NATIVE_DECODE_RUNTIME_MILESTONE.md` | The 30-layer runtime exercises the same reusable executor and checks each layer's cache consequence. |
| Standalone Python GQA development benchmark | `flux_cuda_microbenchmarks -Filter gqa`; `flux_test_decode_ops`; final benchmark and native profiler | Direct production launcher correctness/timing and integrated runtime attribution replace historical model A/B orchestration. |
| Standalone Python gate/up development benchmark | `flux_cuda_microbenchmarks -Filter gate_up`; `flux_test_decode_ops`; final benchmark and native profiler | Exact-shape math, timing, and production integration remain covered without the superseded Python graph comparison. |
| Standalone Python RoPE development benchmark | `flux_cuda_microbenchmarks -Filter rope`; `flux_test_transformer_ops`; retained Transformers/opcheck tests | Native math/timing and the independent Transformers contract remain covered. |
| Separate full native decode benchmark | Canonical `benchmark_flux.py --mode system` native decode column and reduced native runtime profiler | The canonical benchmark now contains HF, Flux eager, native correctness/timing/generation/audit in one system matrix. |
| Old profiler eager/component microbenchmarks and Python-graph indexing | Reduced `benchmark_flux.py --mode profile`; selectable native microbenchmarks | The retained profiler attributes complete native prefill/decode kernels and owners; isolated timing uses correctness-gated native launchers. |
| Shared cache-clone and historical eager benchmark helpers | Retained benchmark-local system logic | No remaining caller used the deleted helpers; deterministic configuration, input generation, and argument parsing remain shared. |
| Attention-score CUDA shape/causal/stream sweeps | `flux_test_transformer_ops::test_attention_score_softmax`; retained broadcast/dispatcher/FakeTensor/opcheck tests | CTest directly checks production CUDA math, preservation, and stream; Python keeps framework semantics. |
| cuBLASLt stream/capture development test | `flux_test_native_runtimes`; retained nonzero plan comparisons and FakeTensor/schema tests | Production selected plans execute on the non-default stream inside the full runtime graph; numerical selected-plan checks remain Python. |
| GQA CUDA boundary/extreme/stream/graph sweeps | `flux_test_decode_ops::test_gqa_short_and_long_context`; `flux_test_native_runtimes`; retained mask/cache/opcheck tests | Native tests cover short and 8192-token math, graph/device state, and stream behavior at the launcher/runtime boundary. |
| Packed-QKV CUDA parameter sweep, stream, and graph tests | `flux_test_decode_ops::test_packed_qkv_rope_cache_and_graph`; retained layout/cache/schema/opcheck tests | Native coverage directly validates Q/K/V math, mutation, device length, graph replay, and current stream. |
| Gate/up repeated stream and graph tests | `flux_test_decode_ops::test_packed_gate_up_swiglu`; `flux_test_native_runtimes`; retained Python schema/FakeTensor/error test | Native tests own exact production execution; Python owns the wrapper contract. |
| Softmax width/stability/causal/stream/repetition sweeps | `flux_test_softmax`; retained dispatcher/rank/layout/error/FakeTensor/opcheck tests | Native tests cover widths 1--8192, stability, row sums, determinism, preservation, and streams. |
| Packed-SwiGLU CUDA shape/stream/determinism sweeps | `flux_test_transformer_ops::test_packed_swiglu`; retained dispatcher/layout/error/FakeTensor/opcheck tests | Direct native math/preservation/stream coverage is stronger; public PyTorch behavior remains tested. |
| RMSNorm and residual-RMSNorm CUDA shape/epsilon/stream/determinism sweeps | `flux_test_rmsnorm`, `flux_test_residual_rmsnorm`; retained dispatcher/layout/alias/error/FakeTensor/opcheck tests | Native launchers own numerical and stream boundaries; Python retains all dispatcher and inference-only contracts. |
| RoPE determinism/current-stream duplication and extra CUDA shapes | `flux_test_transformer_ops::test_rope`; retained Transformers prefill/decode/layout/opcheck tests | Native stream/math coverage and independent Transformers equivalence are both preserved. |
| Python runtime allocation/address/stream/recreation duplication | `flux_test_native_runtimes`; retained learned-model equivalence and thin public reset/reuse validation | Lifecycle mechanics are tested on the production C++ objects; Python tests focus on model and adapter contracts. |

## Retained Python responsibilities

Python continues to own pinned Hugging Face equivalence, Flux eager comparison,
checkpoint/state-dict identity, model transformation and opt-in behavior,
public Python adapters and exceptions, dispatcher/CPU/CUDA selection, tensor
layout semantics, FakeTensor, `torch.library.opcheck`, final logits, all-layer
cache comparison, greedy generation, and end-to-end system benchmarking.
`test_smollm2_flux.py` and every explicit Python reference test remain intact.

## Canonical benchmark hierarchy

1. `benchmarks/benchmark_flux.py --mode system` is the canonical checkpoint-backed
   HF / Flux eager / native prefill-decode correctness and performance matrix.
2. `benchmarks/benchmark_native_smollm2_prefill.py` retains its unique detailed
   every-layer prompt/cache/handoff validation, long-context cache-drift report,
   workspace report, and prefill-specific audit.
3. `benchmarks/benchmark_flux.py --mode profile` retains only full native
   prefill/decode CUDA owner and top-kernel attribution.
4. `flux_cuda_microbenchmarks` owns selectable direct CUDA launcher timing.

## Validation

Validation on Windows, Python 3.11.9, PyTorch 2.14.0+cu132, CUDA 13.2, and an
RTX 5070 Ti produced:

| Command | Result |
| --- | --- |
| focused native runtime oracle migration pytest | 6 passed |
| focused moved out-contract pytest | 5 passed |
| `scripts\native.cmd test` after oracle migration | 10/10 CTest targets passed |
| full remaining Python suite after consolidation | 288 passed |
| canonical final-system length-128 smoke | checkpoint exact; HF/Flux/native logits, K/V, positions, stable addresses, greedy generation, and audit passed |
| reduced native profiler length-128 smoke | native prefill and decode profiles completed with zero allocation growth and stable addresses |
| native prefill length-128 smoke | HF/Flux/native logits and K/V passed; continuation and greedy identity passed; zero allocation growth and framework launches |
| `scripts\native.cmd benchmark -Warmup 1 -Samples 1` | all retained correctness-gated native microbenchmarks completed |

No tolerance was changed.

## Exact source accounting

The same recursive extension accounting excludes `.git`, `build`, virtual
environments, and `__pycache__`.

| Language | Before | After | Change |
| --- | ---: | ---: | ---: |
| Python (`.py`) | 551,695 B | 342,517 B | -209,178 B |
| CUDA (`.cu` + `.cuh`) | 306,421 B | 306,421 B | 0 B |
| C++ (`.cpp` + `.cc` + `.cxx`) | 132,894 B | 132,894 B | 0 B |
| headers (`.h` + `.hpp`) | 16,553 B | 16,553 B | 0 B |

There are 52 Python files. Total counted source is 798,385 bytes. CUDA is
38.380105% of counted source. Non-CUDA source is 491,964 bytes, leaving an
exact 185,543-byte deficit to equal CUDA and non-CUDA source.

The implementation does not force the audit's projected byte count. Further
large semantic deletion is not defensible: the remaining Python is primarily
the model/reference/API surface and independent integration oracle. Mechanical
table-driven consolidation remains possible, especially in
`test_smollm2_flux.py` and operator argument-validation tables, but it would
trade review clarity for source accounting without removing an obsolete
execution responsibility. It is therefore left as optional maintenance rather
than claimed as functional consolidation.

## Limitations and assumptions

- The native backend retains its documented B=1, FP32, SmolLM2-135M geometry
  and maximum-context contracts.
- The native one-layer C++ class remains registered internally with the shared
  native implementation; only its unsupported Python adapter is removed.
- Historical commands in old milestone reports intentionally remain as records
  of the commit at which their measurements were made. Active README commands
  use only retained paths.
