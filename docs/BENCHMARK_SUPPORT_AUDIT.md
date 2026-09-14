# Benchmark and Python support audit

This audit records the final-system consolidation performed after commit
`b374d0423bdff5fe083eb4ded6890c358d265757`. The governing rule was functional:
files were retired only when their result was already recorded and maintained
tests or benchmarks fully covered the useful path. No implementation, tolerance,
benchmark timing region, or published number was changed.

## Result

The benchmark suite is now centered on one production benchmark, one retained
decode profiler, and focused operator benchmarks. Shared deterministic runtime,
input, alternating CUDA-event timing, cache cloning, and model-comparison logic
lives in `benchmarks/smollm2_benchmark_utils.py`.

| Area | Before | After | Byte change |
| --- | ---: | ---: | ---: |
| `benchmarks/` Python | 20 files / 467,114 B | 11 files / 185,224 B | -281,890 B |
| `scripts/` Python | 11 files / 72,748 B | 8 files / 36,713 B | -36,035 B |
| `tests/` Python | 22 files / 179,584 B | unchanged | 0 B |
| all detected Python | 77 files / 852,221 B | 65 files / 534,296 B | -317,925 B |

The benchmark count includes the new shared utility. The total Python count also
includes package/build Python outside these three directories.

## Benchmark inventory

### Maintained

| File | Ownership and reason retained |
| --- | --- |
| `benchmark_final_system.py` | Canonical reference/Flux eager/Flux graph benchmark; owns published production prefill, decode, generation, correctness, launch, allocation, and environment results. |
| `benchmark_native_smollm2_layer_runtime.py` | Active native-runtime milestone harness; compares one native-owned decoder-layer graph with the retained Python-owned layer graph, including correctness, CUDA-event latency, host enqueue cost, launch inventory, replay allocation, and synchronization audits. It is intentionally not an end-to-end model benchmark. |
| `benchmark_smollm2_decode_profile.py` | Final decode profiling infrastructure; uniquely owns full-model kernel/category attribution and bottleneck evidence. |
| `benchmark_attention_score_softmax.py` | Focused eager/graph A/B for the retained fused scale+mask+softmax operator. |
| `benchmark_softmax.py` | General standalone softmax shape sweep and PyTorch comparison. |
| `benchmark_rmsnorm.py` | General standalone RMSNorm shape sweep and PyTorch comparison. |
| `benchmark_residual_rmsnorm.py` | Two-output semantic comparison and fused-boundary breakdown. |
| `benchmark_rope.py` | RoPE operator microbenchmark plus retained integrated prefill/decode/graph comparison. Its former dependency on the general historical model harness moved to the shared utility. |
| `benchmark_smollm2_gqa_decode_attention.py` | Unique isolated, layer, eager, graph, numerical, kernel-inventory, and memory A/B for retained native one-token GQA. |
| `benchmark_gqa_long_context.py` | Small authoritative long-context GQA stage benchmark used by the retained reduction milestone. |
| `benchmark_smollm2_gate_up_gemv.py` | Exact retained fused gate/up GEMV+SwiGLU operator and integrated graph A/B, including launch and memory checks. |
| `smollm2_benchmark_utils.py` | Shared setup only; no independently runnable benchmark. |

### Retired

| File | Why retirement is safe |
| --- | --- |
| `benchmark_smollm2.py` | The final-system benchmark supersedes its reference/eager comparisons; the decode profiler supersedes its profiling. RoPE's only imports were moved to the shared utility. |
| `benchmark_smollm2_cuda_graph.py` | Its fixed-shape graph correctness, timing, generation, memory, and profile paths are a subset of the final benchmark, decode profiler, and graph tests. |
| `benchmark_smollm2_mlp.py` | Milestone-only packed-MLP/SwiGLU variant harness. Layout/state-dict/model behavior remains in `test_smollm2_flux.py`; operator behavior remains in packed-SwiGLU tests; final performance is canonical. The final milestone records that no isolated combined claim is carried forward. |
| `benchmark_smollm2_qkv.py` | Milestone-only packed-QKV A/B. Layout, storage, checkpoint, model, and graph contracts remain tested; final performance is canonical. |
| `benchmark_smollm2_rope_profile.py` | Historical Transformers-versus-Flux profiling harness. The smaller RoPE benchmark retains both isolated and model measurements; final profiling is centralized. |
| `benchmark_smollm2_prefill_fusion.py` | Early three-model prefill experiment superseded by final prefill coverage and fused-attention tests/microbenchmark. |
| `benchmark_smollm2_packed_qkv_rope_cache.py` | Milestone-only boundary A/B. Native operator/cache/stream/graph behavior remains tested and the production path remains in final benchmark/profile runs. |
| `benchmark_smollm2_stable_buffers.py` | The fill-attribution experiment and retained result are recorded in the residual-projection and final-system reports. Stable/stale/stream/fallback contracts remain in `test_stable_decode_outputs.py`; final runtime audit owns allocation and address checks. |
| `benchmark_smollm2_projections.py` | Bounded algorithm search and category-ablation harness for a frozen choice. `test_native_cublaslt_linear.py` owns the selected configuration contract; final benchmark/profile owns production latency, launches, and memory. Original search source remains available at the pre-consolidation commit. |
| `benchmark_softmax_diagnostic.py` | Historical launch-overhead diagnosis. The maintained softmax benchmark owns direct operator performance; fused-attention and final graph benchmarks cover graph use. |

Detailed historical tables and rejected experiments remain in `docs/` and the
pre-consolidation source remains available in Git history. Current production
claims do not require checking out an old harness: they are reproduced by
`benchmark_final_system.py`, while current bottleneck claims use
`benchmark_smollm2_decode_profile.py`.

## Script inventory

| File | Decision |
| --- | --- |
| `reference_inference.py` | Keep: simple public pinned Hugging Face correctness baseline. |
| `flux_inference.py` | Keep: simple public reference-versus-Flux inference smoke path, distinct from the exhaustive benchmark. |
| `check_rmsnorm_cpp.py`, `check_rmsnorm_cuda.py` | Keep: cross-language validation of standalone CLI binaries against the Python oracle. |
| `check_residual_rmsnorm_cpp.py`, `check_residual_rmsnorm_cuda.py` | Keep: same unique cross-language role for dual-output residual RMSNorm. |
| `check_softmax_cpp.py`, `check_softmax_cuda.py` | Keep: same unique cross-language role for softmax, including multiple cases. |
| `diagnose_smollm2_flux.py` | Retire: one-off long-context drift localization for a resolved milestone; comprehensive model/cache/numerical tests now own the behavior. |
| `validate_attention_score_softmax_model.py` | Retire: model substitution and numerical behavior are maintained by fused-attention and full-model tests and by the focused benchmark. |
| `validate_rope_model.py` | Retire: the maintained RoPE tests and benchmark cover pinned-model, operator, prefill, decode, and graph behavior. |

## Test audit and future native ownership

No tests were removed or weakened. Reference expressions and all
PyTorch/Transformers integration tests must remain Python. Some CUDA-specific
cores can later gain native counterparts, but Python tests remain necessary for
dispatcher and framework-boundary coverage.

| Test file | Required Python responsibility | Candidate native core |
| --- | --- | --- |
| `test_import.py` | Package import contract | none |
| `test_smollm2_config.py` | Pinned HF loading/configuration boundary | none |
| `test_smollm2_flux.py` | HF module substitution, opt-in categories, state dicts, cache/generation/model semantics | packed-layout and single-layer numerical cases may be duplicated natively |
| `test_smollm2_cuda_graph.py` | Python public graph API and HF cache/model equivalence | native runtime state progression and layer/full-decode replay |
| `test_stable_decode_outputs.py` | PyTorch out-op schemas, FakeTensor, scratch installation, graph integration | poisoned-buffer overwrite, address stability, stream ordering |
| `test_rmsnorm_reference.py` | PyTorch/autograd and Transformers oracle | none |
| `test_residual_rmsnorm_reference.py` | PyTorch/autograd two-output oracle | none |
| `test_softmax_reference.py` | PyTorch/autograd numerical oracle | none |
| `test_rope_reference.py` | Transformers RoPE oracle/broadcast semantics | none |
| `test_packed_swiglu_reference.py` | PyTorch SiLU/multiply oracle | none |
| `test_attention_score_softmax_reference.py` | Exact PyTorch fused-expression oracle | none |
| `test_gqa_decode_attention_reference.py` | Explicit unexpanded-GQA/mask oracle | none |
| `test_rmsnorm_custom_op.py` | dispatcher, inference-only, noncontiguous, FakeTensor/opcheck | CUDA numeric, invalid geometry, deterministic/current-stream cases |
| `test_residual_rmsnorm_custom_op.py` | dispatcher, output aliasing, inference-only, FakeTensor/opcheck | CUDA numeric, arbitrary size, deterministic/current-stream cases |
| `test_native_softmax.py` | dispatcher, PyTorch comparison, FakeTensor/opcheck | widths/boundaries, stability, deterministic/current-stream cases |
| `test_rope_custom_op.py` | Transformers comparison, dispatcher, FakeTensor/opcheck | offsets/GQA shapes, deterministic/current-stream cases |
| `test_packed_swiglu_custom_op.py` | dispatcher/layout/inference/FakeTensor/opcheck | CUDA shapes, deterministic/current-stream cases |
| `test_native_attention_score_softmax.py` | broadcast/dispatcher/FakeTensor/opcheck | causal shapes, numerical and current-stream cases |
| `test_native_gqa_decode_attention.py` | cache abstractions, masks, dispatcher/FakeTensor/opcheck | boundary lengths, device-length advance, numerical stability, stream/graph cases |
| `test_native_packed_qkv_rope_cache.py` | StaticCache-compatible tensor/schema/opcheck behavior | fused state update, layouts, stream/graph cases |
| `test_native_cublaslt_linear.py` | PyTorch tensor/schema/FakeTensor boundary | selected-algorithm numerical, stream, graph cases |
| `test_native_packed_gate_up_swiglu.py` | PyTorch schema/FakeTensor boundary | exact-shape numerical, overwrite, stream, graph cases |

Native duplicates should use the same deterministic vectors and tolerances and
must not replace the Python oracle or dispatcher/opcheck coverage.

## Reproducibility map

The following current capabilities remain directly reproducible:

- final prefill, eager decode, graph decode, generation, state-dict, KV/cache,
  stable-address, allocation, and launch claims: `benchmark_final_system.py`;
- retained full-model bottleneck and launch-owner analysis:
  `benchmark_smollm2_decode_profile.py`;
- retained GQA long-context stage and PyTorch-baseline behavior:
  `benchmark_gqa_long_context.py` and
  `benchmark_smollm2_gqa_decode_attention.py`;
- retained gate/up fused boundary and graph contribution:
  `benchmark_smollm2_gate_up_gemv.py`;
- RMSNorm, residual-RMSNorm, softmax, fused attention-score processing, and RoPE:
  their focused benchmarks listed above.

Historical milestone numbers remain immutable evidence, not current promises.
Where a one-off harness was retired, its report names the pre-consolidation
commit and the maintained current validation paths.

## Source-language footprint

Accounting recursively by extension and excluding `.git`, `build`, virtual
environments, and `__pycache__`:

| Language | Before files / bytes / KiB | After files / bytes / KiB |
| --- | ---: | ---: |
| Python `.py` | 77 / 852,221 / 832.2 | 65 / 534,296 / 521.8 |
| CUDA `.cu`, `.cuh` | 14 / 127,772 / 124.8 | unchanged |
| C++ `.cpp`, `.cc`, `.cxx` | 22 / 137,612 / 134.4 | unchanged |
| headers `.h`, `.hpp` | 15 / 8,545 / 8.3 | unchanged |

Python decreased by 317,925 bytes (310.5 KiB). CUDA's share of these detected
source languages rises from 11.35% to 15.81% solely because obsolete Python was
removed; this milestone does not attempt the 50% target.

After consolidation, non-CUDA detected source is 680,453 bytes. With current
CUDA at 127,772 bytes, the exact remaining footprint shift is 552,681 bytes
(539.7 KiB):

```text
new CUDA bytes + legitimately removed non-CUDA bytes >= 552,681
```

This is an accounting constraint, not a source-generation target. Future bytes
must come from the native runtime architecture and its real validation needs.

## Validation

Validation used the repository's verified Python 3.11.9/PyTorch CUDA 13.2
environment:

```powershell
build\python3119\python.exe -m compileall -q benchmarks scripts tests flux
build\python3119\python.exe -m pytest -q
build\python3119\python.exe benchmarks\benchmark_final_system.py `
  --lengths 128 --skip-generation --skip-audit `
  --stabilization-iterations 0 --warmup 1 --samples 1 --rounds 1 `
  --correctness-tokens 1
build\python3119\python.exe benchmarks\benchmark_rope.py `
  --warmup 1 --repetitions 3 --prefill-lengths 16 --decode-positions 128
```

Results: compilation succeeded; all 461 tests passed in 9.85 seconds; the final
benchmark smoke completed with exact state-dict compatibility, matching greedy
tokens, stable addresses, and bounded logits/K/V error; and the RoPE operator,
prefill, cached-decode, and graph paths completed with exact comparison error in
the selected smoke workloads. Smoke timings are intentionally not published as
performance claims because their sample counts are validation-only.
