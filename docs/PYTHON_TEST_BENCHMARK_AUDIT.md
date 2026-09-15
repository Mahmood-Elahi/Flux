# Python test and benchmark ownership audit

## Scope and conclusion

This is an audit of every tracked Python file under `benchmarks/` and `tests/`
at canonical commit `0cb2669`. It does not propose a new kernel, change
inference behavior, or delete source. File sizes are filesystem bytes at that
commit.

The current inventory is:

| Area | Files | Bytes |
| --- | ---: | ---: |
| `benchmarks/*.py` | 8 | 175,014 |
| `tests/*.py` | 25 | 200,239 |
| audited total | 33 | 375,253 |
| all tracked Python | 61 | 551,695 |
| CUDA (`.cu` + `.cuh`) | 21 | 306,421 |
| C++ and headers, held fixed in the projections | 40 | 149,447 |

The production architecture has moved past two Python-owned development
boundaries:

1. `FluxCUDAGraphDecode` is no longer the natural owner of production decode.
   `NativeSmolLM2Prefill` now owns prompt execution, compact cache creation,
   direct handoff, device state, and the complete 30-layer native decode graph.
2. `NativeSmolLM2LayerDecode` is a completed intermediate milestone, not a
   production boundary. The full runtime exercises the same layer sequence in
   all 30 layers.

The old graph is nevertheless still a *test and benchmark dependency*. The
full native decode test compares against it, and the canonical final-system
benchmark and retained decode profiler still instantiate it. Therefore the
implementation can be retired only after the integration assertions listed
below migrate. The one-layer test has no equivalent remaining dependency and
can be retired now.

The recommended outcome is the **moderate** scenario: immediately retire the
three operator-era model benchmarks and the one-layer runtime test; port the
canonical benchmark/profiler to the native runtime; retire the Python graph
test after moving its still-useful learned-model assertions; and shrink custom
operator tests to their Python dispatcher, schema, FakeTensor, opcheck, and
framework-integration responsibilities.

## Classification rules used

- **A — must remain Python:** Hugging Face/PyTorch oracle, checkpoint, public
  Python API, dispatcher, FakeTensor/opcheck, model, cache abstraction, or
  generation/token semantics.
- **B — keep but reduce:** unique Python ownership is mixed with direct CUDA
  numerical, stream, graph, address, allocation, or timing coverage now owned
  natively.
- **C — fully superseded by native CUDA:** all live behavior has an equal or
  stronger direct native owner.
- **D — historical/obsolete:** the file principally exercises a development
  path that is no longer canonical; retained results are documented or the
  behavior is intentionally no longer a production promise.

`Estimated removable bytes` is an implementation-planning estimate, not a
claim that functions can be cut mechanically. Exact whole-file retirements use
the current file size. Reductions include the expected removal of local helpers,
fixtures, and imports made dead by the named tests. Migration code added to a
retained file is not known yet, so projections are gross reductions from the
current baseline and must be replaced with measured post-change accounting.

## Benchmark audit

| File | Bytes | Class | Unique capability and current owner | Native supersession / historical record | Recommendation | Estimated removable bytes |
| --- | ---: | :---: | --- | --- | :---: | ---: |
| `benchmark_final_system.py` | 30,135 | A | The canonical checkpoint-backed reference/Flux/full-system matrix: state dict, prefill, decode, generation, logits, K/V, token identity, environment, and production summary. This capability is still required, but its graph column must be implemented with the native prompt-to-decode runtime. | Native CTest cannot replace learned Hugging Face equivalence or end-to-end timing. Published old-graph results are in `FINAL_SYSTEM_MILESTONE.md`. | **KEEP** | 0 moderate; up to 5,135 only in the aggressive consolidation |
| `benchmark_native_smollm2_prefill.py` | 18,099 | A | Checkpoint-backed three-path prefill validation/timing, every-layer compact cache comparison, direct decode handoff, generation identity, and native prefill profiling. | `flux_test_native_runtimes`, `flux_test_streaming_prefill_gqa`, and the `streaming_prefill_gqa` microbenchmark cover low-level mechanics, not learned-model equivalence or prefill latency. Results are in `NATIVE_PREFILL_RUNTIME_MILESTONE.md` and `STREAMING_PREFILL_GQA_MILESTONE.md`. | **KEEP** | 0 moderate; up to 4,099 aggressive |
| `benchmark_native_smollm2_runtime.py` | 15,342 | B | Current full 30-layer native decode versus Flux/Python graph, learned weights, cache/logit comparison, greedy generation, host enqueue timing, device timing, and profiler ownership. | `flux_test_native_runtimes` supersedes its allocation/address/current-stream/capacity audit. Its full-model comparison and timing remain Python until absorbed by `benchmark_final_system.py` and the native decode profiler. Results are recorded in `NATIVE_FULL_DECODE_RUNTIME_MILESTONE.md`. | **REDUCE**, then consolidate | 4,500 moderate; entire 15,342 aggressive after consolidation |
| `benchmark_smollm2_decode_profile.py` | 49,970 | B | Only current full-model kernel/category attribution and bottleneck inventory. That diagnostic capability remains valuable, but it profiles the old Python graph and also carries eager/component microbench machinery. | Direct operator timing is covered by `flux_cuda_microbenchmarks`; old profile results are in `DECODE_PROFILE_MILESTONE.md`, `GQA_LONG_CONTEXT_MILESTONE.md`, `LM_HEAD_GEMV_MILESTONE.md`, and `FINAL_SYSTEM_MILESTONE.md`. Retain a smaller native-runtime profiler, not the old graph implementation. | **REDUCE** | 24,000 moderate; 29,970 aggressive |
| `benchmark_smollm2_gqa_decode_attention.py` | 23,038 | D | Historical isolated/layer/eager/Python-graph A/B, numerical, launch, and memory investigation for selecting one-token GQA variants. It no longer answers a distinct production question. | Direct current kernel timing is `flux_cuda_microbenchmarks -Filter gqa`; numerical/long-context coverage is `flux_test_decode_ops`; integrated current performance belongs to the final benchmark/native profiler. Results and rejected variants are preserved in `GQA_LONG_CONTEXT_MILESTONE.md`, `GQA_REDUCTION_MILESTONE.md`, `DECODE_PROFILE_MILESTONE.md`, and Git history. | **RETIRE** | 23,038 |
| `benchmark_smollm2_gate_up_gemv.py` | 20,656 | D | Historical standalone/boundary/layer/eager/Python-graph A/B and memory investigation for the retained fused gate/up+SwiGLU choice. The choice is frozen and the compared Python graph is superseded. | `flux_cuda_microbenchmarks -Filter gate_up` owns direct timing and `flux_test_decode_ops` owns exact-shape math; final-system/native-runtime measurement owns integration. Detailed historical results are in `GATE_UP_GEMV_MILESTONE.md`, `DECODE_PROFILE_MILESTONE.md`, and `FINAL_SYSTEM_MILESTONE.md`. | **RETIRE** | 20,656 |
| `benchmark_rope.py` | 7,494 | D | Combines direct RoPE timing with reference/Flux prefill, decode, and old Python-graph A/B. Direct timing and operator integration are useful, but none now requires this standalone model harness. | `flux_cuda_microbenchmarks -Filter rope` owns direct timing; `flux_test_transformer_ops` owns CUDA math; `test_rope_custom_op.py`, `test_rope_reference.py`, and full-system/native prefill validation own Transformers/model behavior. There is no dedicated historical RoPE result table; `NATIVE_VALIDATION_MILESTONE.md` records the native median, while `ROADMAP.md`, `README.md`, final-system results, and Git history record the integrated outcome. The old isolated model A/B is no longer a canonical promise. | **RETIRE** | 7,494 |
| `smollm2_benchmark_utils.py` | 10,280 | B | Shared deterministic setup/input generation and CUDA-event comparison utilities used by all retained Python system benchmarks. | Native timing supersedes only helpers used exclusively by retiring operator benchmarks. Keep deterministic inputs and alternating timing for checkpoint-backed system benchmarks. | **REDUCE** | 4,000 moderate; 4,280 aggressive |

### Benchmark portions to retain or remove

- In `benchmark_native_smollm2_runtime.py`, retain `_benchmark_length`, learned
  logits/K/V/token comparisons, `_validate_generation`, and host-versus-device
  timing until those move into the canonical benchmark. Remove `_audit` and its
  allocation/address/launch bookkeeping after mapping it to
  `flux_test_native_runtimes`; remove Python-graph construction after the
  reference oracle is direct Hugging Face/Flux eager.
- In `benchmark_smollm2_decode_profile.py`, retain environment reporting,
  `_profile_call`, kernel ownership/category aggregation, summary printing, and
  a native-runtime replay entry point. Remove `_make_graph`,
  `_make_capacity_graph`, Python-graph component indexing, the local layer
  microbenchmark, cache-update microbenchmarks, intermediate/GEMM inventories,
  and old eager-versus-graph experiment branches. Direct operator timing moves
  to `flux_cuda_microbenchmarks`.
- In `smollm2_benchmark_utils.py`, retain `Comparison`,
  `parse_positive_int_list`, `configure_runtime`, `deterministic_input_ids`,
  `event_median`, and `alternating_event_medians`. If `benchmark_rope.py` is
  retired, remove `_cache_length`, `_clone_cache`, `_assert_logits_close`,
  `benchmark_full_or_prefill`, and `benchmark_decode` once `rg` confirms no
  remaining callers.

## Test audit

### Python reference and integration tests

| File | Bytes | Class | Python responsibility and duplicate/obsolete content | Recommendation | Estimated removable bytes |
| --- | ---: | :---: | --- | :---: | ---: |
| `test_import.py` | 77 | A | Package import smoke; no native equivalent. | **KEEP** | 0 |
| `test_smollm2_config.py` | 3,277 | A | Pinned model/revision helpers, import-without-loading, synthetic config inspection, and mocked Hugging Face tokenizer/model loading options. | **KEEP** | 0 |
| `test_smollm2_flux.py` | 45,715 | A | The central Python/Hugging Face contract: explicit opt-in categories, module substitution, packed QKV/MLP storage and slices, ordinary state dict and `save_pretrained`, dependency/fallback dispatch, layer/model logits, DynamicCache, causal mask/GQA/RoPE/scale/residual semantics, and greedy token identity. Native CTest cannot replace these. The bottom CUDA model tests overlap numerically with native runtime tests but still validate the eager PyTorch integration path. | **KEEP** | 0 moderate; up to 13,715 aggressive by table-driven consolidation, without dropping contracts |
| `test_attention_score_softmax_reference.py` | 2,046 | A | Exact PyTorch expression, broadcast semantics, shape/dtype/scale validation. This is the oracle used by native integration tests. | **KEEP** | 0 |
| `test_gqa_decode_attention_reference.py` | 2,464 | A | Explicit unexpanded grouped-head mapping, valid-length truncation, and mask oracle. | **KEEP** | 0 |
| `test_packed_swiglu_reference.py` | 989 | A | Explicit SiLU-times-up oracle and Python argument validation. | **KEEP** | 0 |
| `test_residual_rmsnorm_reference.py` | 7,992 | A | Independent two-output math, output ordering, arbitrary sizes/epsilon, input preservation, noncontiguous inputs, autograd from both outputs, and validation. | **KEEP** | 0 |
| `test_rmsnorm_reference.py` | 2,979 | A | PyTorch RMSNorm and Transformers Llama RMSNorm equivalence, determinism, and reference validation. | **KEEP** | 0 |
| `test_rope_reference.py` | 2,081 | A | Transformers RoPE equivalence, batch broadcasting, and odd-dimension rejection. | **KEEP** | 0 |
| `test_softmax_reference.py` | 4,402 | A | PyTorch/autograd oracle, CPU semantics, noncontiguous behavior, stability, and validation. | **KEEP** | 0 |

The ten files above total 72,022 bytes and are not candidates for semantic
removal. The aggressive projection allows only mechanical consolidation of
about 5,000 bytes across the small reference files and 13,715 bytes in
`test_smollm2_flux.py`; it does not drop any listed contract.

### Mixed custom-operator tests

| File | Bytes | Class | Exact Python tests/coverage to retain | Low-level cases to remove or collapse and native owner | Recommendation | Estimated removable bytes |
| --- | ---: | :---: | --- | --- | :---: | ---: |
| `test_native_attention_score_softmax.py` | 6,911 | B | Retain one CPU/CUDA dispatcher comparison in `test_matches_unfused_expression`, broadcast-mask tests, noncontiguous wrapper behavior, invalid/autograd checks, `test_fake_tensor_metadata`, and `test_torch_library_opcheck`. | Collapse the CUDA shape sweep and remove `test_realistic_causal_masks_and_query_positions` and `test_uses_current_non_default_cuda_stream`; `flux_test_transformer_ops::test_attention_score_softmax` owns CUDA math/input preservation/stream, and the `attention_softmax` native benchmark owns timing. | **REDUCE** | 3,000 |
| `test_native_cublaslt_linear.py` | 7,844 | B | Retain `test_cublaslt_linear_matches_pytorch_and_repeats`, `test_retained_explicit_configuration_matches_pytorch`, and `test_cublaslt_linear_fake_tensor_contract`. These are the only nonzero learned/random numerical checks of the selected PyTorch-visible plan. | Remove the stream/capture half of `test_cublaslt_linear_uses_current_stream_and_captures` after retaining a small Python graph registration smoke if desired. `flux_test_native_runtimes` executes the production plan on a non-default stream and inside the runtime graph, but its zero-weight model does **not** replace the numerical tests. | **REDUCE** | 2,800 |
| `test_native_gqa_decode_attention.py` | 9,378 | B | Retain dynamic/device cache-length API behavior, `None` and broadcast masks, noncontiguous cache contract, invalid/autograd checks, FakeTensor, opcheck, and one representative reference comparison. | Remove/collapse repeated CUDA determinism/current-stream, graph replay, SmolLM2 boundary-length sweep, and extreme-finite sweep. `flux_test_decode_ops::test_gqa_short_and_long_context` owns direct short/8192 math and non-default stream; runtime CTest owns graph/device-state; the `gqa` microbenchmark owns timing. | **REDUCE** | 4,300 |
| `test_native_packed_gate_up_swiglu.py` | 4,136 | B | Retain `test_rejects_invalid_shape_dtype_layout_output_and_autograd` and `test_fake_tensor_and_schema_contract`, plus at most one PyTorch-wrapper numerical smoke. | Direct random/exact-shape math, repeated determinism/current stream, graph replay, output overwrite, and stable output are owned by `flux_test_decode_ops::test_packed_gate_up_swiglu`, `flux_test_native_runtimes`, and the `gate_up` microbenchmark. | **REDUCE** | 2,300 |
| `test_native_packed_qkv_rope_cache.py` | 8,728 | B | Retain positive-stride packed projection layout, StaticCache-compatible tensor contract, invalid dtype/shape/autograd checks, FakeTensor, opcheck, and one wrapper numerical smoke. | Remove broad CUDA numerical parameterization, repeated current-stream/determinism, and direct graph/device-length tests. `flux_test_decode_ops::test_packed_qkv_rope_cache_and_graph` directly checks Q/K/V math, cache mutation, advancing/non-advancing state, graph capture/replay, and stream; `packed_qkv` owns timing. | **REDUCE** | 4,200 |
| `test_native_softmax.py` | 8,496 | B | Retain representative CPU/CUDA dispatcher comparison, attention-rank shape behavior, noncontiguous wrapper behavior, dtype/shape/autograd/inference checks, FakeTensor, and opcheck. | Remove the duplicated required-width sweep, numerical-property cases, long-causal sweep, shift invariance, repeat determinism, and current-stream producer/consumer case. `flux_test_softmax` covers widths 1 through 8192, stability, row sums, determinism, input preservation, invalid launch geometry, default/non-default streams; `softmax` owns timing. | **REDUCE** | 4,300 |
| `test_packed_swiglu_custom_op.py` | 4,997 | B | Retain representative CPU/CUDA dispatch, no-copy layout rejection, invalid/autograd/inference behavior, FakeTensor metadata, and opcheck. | Collapse the CUDA shape sweep and remove standalone determinism/current-stream duplication. `flux_test_transformer_ops::test_packed_swiglu` owns direct math/input preservation/stream and `packed_swiglu` owns timing. | **REDUCE** | 1,700 |
| `test_residual_rmsnorm_custom_op.py` | 11,074 | B | Retain dispatcher CPU/CUDA smoke, distinct non-aliasing two-output contract, internal-contiguous wrapper behavior, invalid/mismatched-device/autograd/inference checks, FakeTensor, and opcheck. | Remove broad CUDA shape/epsilon sweeps, repeated determinism, and explicit producer/consumer stream test. `flux_test_residual_rmsnorm` owns arbitrary-width/two-output math, epsilon, determinism, input preservation, launcher validation, and non-default stream; `residual_rmsnorm` owns timing. | **REDUCE** | 4,700 |
| `test_rmsnorm_custom_op.py` | 5,879 | B | Retain representative dispatcher comparison, noncontiguous wrapper behavior, invalid/mismatched-device/autograd/inference checks, and opcheck/FakeTensor coverage. | Remove duplicated zero/broad CUDA numeric cases, repeated determinism, and current-stream test. `flux_test_rmsnorm` owns direct math, size/epsilon boundaries, validation, and default/non-default stream; `rmsnorm` owns timing. | **REDUCE** | 2,400 |
| `test_rope_custom_op.py` | 5,899 | B | Retain `test_matches_transformers_for_prefill_and_decode`, noncontiguous embedding contract, invalid/autograd checks, and opcheck/FakeTensor behavior. The Transformers comparison cannot move native. | Remove only standalone determinism and current-stream duplication; `flux_test_transformer_ops::test_rope` owns direct GQA-aware Q/K math, odd dimensions, and non-default stream, while `rope` owns timing. | **REDUCE** | 1,000 |

These files should not be retired wholesale. CTest validates launchers; it does
not validate `torch.library` schemas, fake implementations, dispatcher/device
selection, inference-mode interaction, or PyTorch tensor/view contracts. In an
aggressive consolidation, their retained responsibilities can be organized
into shared/table-driven operator contract tests, reducing the current 73,342
bytes to about 25,000 bytes. That is a layout change, not permission to discard
the contracts.

### Runtime and old-graph tests

| File | Bytes | Class | Exact integration coverage to retain or migrate | Superseded/obsolete coverage and native owner | Recommendation | Estimated removable bytes |
| --- | ---: | :---: | --- | --- | :---: | ---: |
| `test_smollm2_cuda_graph.py` | 20,684 | D | Before retirement, migrate multi-step logits and greedy-token identity, complete valid K/V comparison, model/category integration for packed QKV/RoPE/cache + selected cuBLASLt + fused gate/up, capacity exhaustion, and stable-address assertions into `test_native_smollm2_runtime.py` / `test_native_smollm2_prefill_runtime.py` using a direct Hugging Face/Flux eager oracle. | Constructor/capture errors, short-capacity dispatch, Python scratch installation, and all `FluxCUDAGraphDecode`-specific fallback behavior disappear with the backend. Stream/device-state/address/capacity mechanics are also covered by `flux_test_native_runtimes`; fused operator math/graph behavior is in `flux_test_decode_ops`. | **RETIRE after migration** | 20,684 |
| `test_stable_decode_outputs.py` | 13,536 | B | Retain a compact PyTorch-visible out-operator contract: returned object identity, output non-alias validation, schema, FakeTensor/opcheck, and one poisoned-output wrapper smoke. Move it to the relevant custom-op tests rather than retaining an old graph test file. | `test_out_variants_overwrite_sentinel_buffers_repeatedly`, most of the non-default-stream test, `test_stable_scratch_graph_replay_has_no_stale_data_and_preserves_state`, and `test_short_capacity_preserves_existing_fallback_without_scratch` are low-level or old-backend behavior. Map to `flux_test_rmsnorm`, `flux_test_residual_rmsnorm`, `flux_test_transformer_ops`, `flux_test_decode_ops`, and `flux_test_native_runtimes`; the short fallback is intentionally obsolete. | **REDUCE**, then retire the file after moving the small contracts | 10,500 |
| `test_native_smollm2_layer_runtime.py` | 10,852 | D | No unique production/API purpose remains. Its learned one-layer comparison was useful while constructing the full runtime, but is subsumed by learned 30-layer every-cache comparison. | All four tests map to `test_native_smollm2_runtime.py`, `test_native_smollm2_prefill_runtime.py`, and `flux_test_native_runtimes`: full-graph math/cache, replay/reset/allocation/address stability, current stream, destruction/recreation. Historical one-layer measurements and architecture are in `NATIVE_DECODE_RUNTIME_MILESTONE.md` and Git history. | **RETIRE** | 10,852 |
| `test_native_smollm2_runtime.py` | 4,703 | B | Retain full learned 30-layer native replay versus an independent HF/Flux eager oracle, all-layer K/V, logits and token identity, public Python capture/replay/reset validation, and stable returned-logits identity. Replace its dependency on `FluxCUDAGraphDecode`; do not drop the learned oracle. | Allocation stability, current stream, capacity, stable addresses, reset state, and recreation are duplicated by `flux_test_native_runtimes`; keep at most thin Python API assertions for reset errors. | **REDUCE** | 2,100 |
| `test_native_smollm2_prefill_runtime.py` | 5,100 | B | Retain HF/Flux equivalence for prefill logits and every layer's compact K/V, prompt-to-decode handoff, public input/capacity validation, and greedy token identity. Expand generation from one token to several before retiring the old graph generation test. | Reuse allocation stability, address stability, non-default stream, capacity exhaustion, and recreation are directly covered by `flux_test_native_runtimes::test_prefill_handoff_and_reuse`; retain only thin Python property/error checks. | **REDUCE** | 1,500 |

## Old Python CUDA-Graph backend decision

The native full decode runtime **does supersede**
`flux/model/smollm2_cuda_graph.py` as the production execution path. The native
prefill runtime additionally supersedes the Python graph's prompt-specific
StaticCache creation and handoff. The old path should not remain merely to
serve as a second CUDA-Graph implementation or as the oracle for the native
one.

Retirement has four prerequisites:

1. Change `test_native_smollm2_runtime.py` to advance an independent
   Hugging Face/Flux eager `DynamicCache` beside native replay and compare
   logits, greedy IDs, and all 30 K/V prefixes for multiple steps.
2. Extend `test_native_smollm2_prefill_runtime.py` from one generated token to
   a multi-token greedy continuation, and retain prompt logits, direct handoff,
   position/length, cache shape, and exhaustion assertions.
3. Port `benchmark_final_system.py` so its production column uses
   `NativeSmolLM2Prefill` and its attached native decode runtime. Preserve
   setup/TTFT versus steady replay timing boundaries and the HF reference.
4. Point the reduced decode profiler at native runtime replay. Once no imports
   remain, retire `test_smollm2_cuda_graph.py`, the old-graph portions of
   `test_stable_decode_outputs.py`, and then the implementation in a separate
   source-cleanup task.

Assertions that should **not** migrate are the small-capacity scratch fallback,
old graph scratch installation, private `_allocate_decode_scratch` and
`_installed_decode_scratch` behavior, and category A/Bs whose only purpose was
choosing already-retained kernels.

## One-layer native runtime decision

`flux/runtime/native_smollm2_layer.py` has no remaining unique production
purpose. Its useful semantic boundary—one native layer versus the then-current
Python graph—was a construction oracle for the 30-layer runtime. The full
learned-model test now checks the output and cache consequence of every layer,
while `flux_test_native_runtimes` checks lifecycle/state mechanics directly.

Retire `test_native_smollm2_layer_runtime.py` immediately. Removal of the
Python adapter and native class registration should be a later source task,
after `rg` confirms no external/public compatibility commitment is intended.
The historical layer timing remains reproducible from commit `ac962fa` and is
fully tabulated in `NATIVE_DECODE_RUNTIME_MILESTONE.md`.

## Native replacement mapping

Every proposed deletion or reduced test group has the following owner:

| Python file or test group | Replacement owner |
| --- | --- |
| `benchmark_smollm2_gqa_decode_attention.py` | `flux_cuda_microbenchmarks -Filter gqa`; `flux_test_decode_ops::test_gqa_short_and_long_context`; retained `test_native_gqa_decode_attention.py` dispatcher/cache/mask/opcheck subset; native final benchmark/profiler; GQA milestone documents |
| `benchmark_smollm2_gate_up_gemv.py` | `flux_cuda_microbenchmarks -Filter gate_up`; `flux_test_decode_ops::test_packed_gate_up_swiglu`; retained schema/FakeTensor wrapper test; native final benchmark/profiler; `GATE_UP_GEMV_MILESTONE.md` |
| `benchmark_rope.py` | `flux_cuda_microbenchmarks -Filter rope`; `flux_test_transformer_ops::test_rope`; retained Transformers/custom-op tests; native prefill/final benchmark |
| direct timing/component sections of `benchmark_smollm2_decode_profile.py` | Corresponding filters in `flux_cuda_microbenchmarks`; keep only native full-runtime category attribution |
| `_audit` in `benchmark_native_smollm2_runtime.py` | `flux_test_native_runtimes::test_full_decode_runtime_on_non_default_stream`; canonical benchmark runtime audit if a report field is still required |
| `test_smollm2_cuda_graph.py` | Migrated native-vs-HF multi-step logits/K/V/token tests; `test_native_smollm2_prefill_runtime.py`; `flux_test_native_runtimes`; `flux_test_decode_ops` |
| `test_stable_decode_outputs.py::test_out_variants_overwrite_sentinel_buffers_repeatedly` | Direct native operator CTests above plus one retained PyTorch out-schema/identity/poison smoke |
| `test_stable_decode_outputs.py::test_out_variants_use_non_default_stream_and_validate_outputs` | `flux_test_rmsnorm`, `flux_test_residual_rmsnorm`, `flux_test_transformer_ops`, and `flux_test_decode_ops`; retain Python alias/error checks |
| `test_stable_decode_outputs.py::test_stable_scratch_graph_replay_has_no_stale_data_and_preserves_state` | `flux_test_native_runtimes` address/allocation/replay invariants plus native-vs-HF full runtime test; old scratch object itself is obsolete |
| `test_stable_decode_outputs.py::test_short_capacity_preserves_existing_fallback_without_scratch` | No replacement: the assertion describes an intentionally retired Python-backend dispatch path |
| `test_native_smollm2_layer_runtime.py` | `test_native_smollm2_runtime.py`, `test_native_smollm2_prefill_runtime.py`, `flux_test_native_runtimes`, and the historical one-layer milestone |
| CUDA numerical/stream/determinism portions of `test_rmsnorm_custom_op.py` | `flux_test_rmsnorm`; `flux_cuda_microbenchmarks -Filter rmsnorm` |
| CUDA numerical/stream/determinism portions of `test_residual_rmsnorm_custom_op.py` | `flux_test_residual_rmsnorm`; `flux_cuda_microbenchmarks -Filter residual_rmsnorm` |
| CUDA numerical/width/stability/stream portions of `test_native_softmax.py` | `flux_test_softmax`; `flux_cuda_microbenchmarks -Filter softmax` |
| CUDA numerical/stream portions of `test_rope_custom_op.py` | `flux_test_transformer_ops::test_rope`; `flux_cuda_microbenchmarks -Filter rope` |
| CUDA numerical/stream portions of `test_packed_swiglu_custom_op.py` | `flux_test_transformer_ops::test_packed_swiglu`; `flux_cuda_microbenchmarks -Filter packed_swiglu` |
| CUDA causal/numerical/stream portions of `test_native_attention_score_softmax.py` | `flux_test_transformer_ops::test_attention_score_softmax`; `flux_cuda_microbenchmarks -Filter attention_softmax` |
| CUDA boundary/extreme/stream/graph portions of `test_native_gqa_decode_attention.py` | `flux_test_decode_ops::test_gqa_short_and_long_context`; `flux_test_native_runtimes`; `flux_cuda_microbenchmarks -Filter gqa` |
| CUDA math/state/stream/graph portions of `test_native_packed_qkv_rope_cache.py` | `flux_test_decode_ops::test_packed_qkv_rope_cache_and_graph`; `flux_cuda_microbenchmarks -Filter packed_qkv` |
| CUDA math/stream/graph portions of `test_native_packed_gate_up_swiglu.py` | `flux_test_decode_ops::test_packed_gate_up_swiglu`; graph execution in `flux_test_native_runtimes`; `flux_cuda_microbenchmarks -Filter gate_up` |
| stream/graph portion of `test_native_cublaslt_linear.py` | Production plan execution in `flux_test_native_runtimes`; retain nonzero PyTorch numerical and FakeTensor tests |
| allocation/address/current-stream/lifecycle parts of full decode/prefill Python tests | `flux_test_native_runtimes`; retain learned HF equivalence and public adapter checks |

## Immediate retirements, reductions, and required Python

### Can be retired immediately

- `benchmarks/benchmark_smollm2_gqa_decode_attention.py` (23,038 B)
- `benchmarks/benchmark_smollm2_gate_up_gemv.py` (20,656 B)
- `benchmarks/benchmark_rope.py` (7,494 B)
- `tests/test_native_smollm2_layer_runtime.py` (10,852 B)

Total immediate exact retirement: **62,040 B**.

README reproduction commands that name the three retired benchmarks must be
updated in the same implementation change to point to native benchmark filters,
the final system benchmark, and the preserved milestone documents.

### Can be reduced or consolidated

- `benchmark_native_smollm2_runtime.py`
- `benchmark_smollm2_decode_profile.py`
- `smollm2_benchmark_utils.py`
- all ten mixed custom-operator test files in the table above
- `test_stable_decode_outputs.py`
- `test_native_smollm2_runtime.py`
- `test_native_smollm2_prefill_runtime.py`

`test_smollm2_cuda_graph.py` is a delayed retirement, not a permanent reduced
file: first move its native-relevant integration assertions.

### Must remain Python

- `benchmark_final_system.py`
- `benchmark_native_smollm2_prefill.py`
- the reduced native full-decode benchmark/profiler capability until merged
- `test_import.py`
- `test_smollm2_config.py`
- `test_smollm2_flux.py`
- all seven explicit `*_reference.py` files
- reduced PyTorch custom-op contract coverage for every public operator
- reduced full native prefill/decode learned-model and public Python adapter
  coverage

## Byte projections

The source-share denominator is CUDA + remaining Python + unchanged C++ and
headers. Thus:

```text
CUDA share = 306,421 / (306,421 + remaining Python + 149,447)
distance to 50% = (remaining Python + 149,447) - 306,421
```

The distance is how many additional non-CUDA bytes would have to be removed
(with CUDA unchanged) for CUDA and non-CUDA detected source to be equal. These
scenarios count only changes to the audited Python test/benchmark files; they
do not count a later deletion of `smollm2_cuda_graph.py` or
`native_smollm2_layer.py`.

| Scenario | Included work | Python bytes removed | Python bytes remaining | Projected CUDA share | Remaining distance to 50% |
| --- | --- | ---: | ---: | ---: | ---: |
| Current | No change | 0 | 551,695 | 30.412093% | 394,721 B |
| Conservative | Four immediate whole-file retirements only | 62,040 | 489,655 | 32.407567% | 332,681 B |
| Moderate (recommended) | Conservative plus delayed old-graph test retirement and all per-file reductions estimated above | 160,024 | 391,671 | 36.154206% | 234,697 B |
| Aggressive but defensible | Moderate; absorb the native decode benchmark into the canonical benchmark, shrink final/prefill/profiler harness duplication, table-drive retained operator contracts, and mechanically consolidate reference/model tests without dropping semantics | 222,707 | 328,988 | 39.041684% | 172,014 B |

Moderate consists of 83,688 benchmark bytes and 76,336 test bytes. Aggressive
leaves approximately 65,000 benchmark bytes and 87,546 test bytes, so it still
retains substantial Python coverage where Python is the system under test.
Even the aggressive scenario does not reach 50%; deleting legitimate Python
oracles merely to reach a language ratio would violate the project's
correctness rules.

## Recommended consolidation sequence

1. **Break the oracle dependency first.** Make full native decode compare
   directly with an independently advanced HF/Flux eager cache, and extend
   native greedy generation beyond one token. Run focused Python tests and the
   full suite.
2. **Move the canonical benchmark to the actual production runtime.** Preserve
   all final-system output fields and timing boundaries while replacing
   `FluxCUDAGraphDecode` with `NativeSmolLM2Prefill`/attached decode.
3. **Reduce and port the profiler.** Keep only native full-runtime replay and
   kernel/category attribution; use native microbenchmark filters for isolated
   operators.
4. **Retire historical harnesses.** Remove the three operator model benchmarks,
   the one-layer runtime test, and their now-dead utility functions; update
   README commands and audit references.
5. **Retire the Python graph validation path.** Once `rg` shows no retained
   test/benchmark imports, remove `test_smollm2_cuda_graph.py` and old graph
   sections of `test_stable_decode_outputs.py`. Source deletion is a separate
   explicitly reviewed task.
6. **Trim mixed operator tests one family at a time.** For each operator, keep
   dispatcher/schema/FakeTensor/opcheck/framework semantics, verify its mapped
   CTest target, run that target plus focused pytest, then run both full suites.
7. **Recount measured bytes.** Replace estimates with filesystem counts and
   recalculate the CUDA share; do not optimize the source ratio by weakening
   coverage.

## Exact first implementation step

Change only the native runtime integration tests first:

1. Rewrite
   `test_native_smollm2_runtime.py::test_full_native_repeated_replay_matches_python_graph_and_all_caches`
   so the expected path is an independently seeded HF/Flux eager model with a
   `DynamicCache`, not `FluxCUDAGraphDecode`.
2. Preserve its three-step logits, argmax token, every-layer K/V prefix,
   position/length, stable returned-logits, and capacity-exhaustion assertions.
3. Extend
   `test_native_smollm2_prefill_runtime.py::test_native_prefill_single_token_greedy_generation`
   to a small multi-token greedy continuation and compare exact token IDs with
   Hugging Face generation.
4. Run:

   ```powershell
   build\python3119\python.exe -m pytest -q `
     tests\test_native_smollm2_runtime.py `
     tests\test_native_smollm2_prefill_runtime.py
   scripts\native.cmd test
   build\python3119\python.exe -m pytest -q
   ```

This is the first safe cut because it establishes the independent correctness
oracle required before any old graph test or implementation can be retired.
