# Thin Python integration layer milestone

## Outcome

This milestone audits all 22 remaining Python test files and the four-program
Python benchmark layer at the 342,517-byte baseline. It does not change model
mathematics, CUDA, C++, tolerances, or the production Python package.

The separate native-prefill benchmark is retired after its remaining system
responsibilities are absorbed by `benchmark_final_system.py`. The canonical
benchmark now reports every-layer native prefill K/V errors, cache position and
length, stable addresses, cache bytes, prefill and decode workspace bytes, and
stable-buffer bytes. Prefill-to-decode continuation, Hugging Face and Flux
logits, greedy identity, state-dict identity, and three-path timing were already
canonical benchmark responsibilities. Native prefill/decode launch and kernel
attribution remains in `benchmark_smollm2_decode_profile.py`.

Three category-4-only repetitions are removed from Python: a second softmax
CUDA numerical shape, a second residual-RMSNorm epsilon sweep, and two extra
gate/up random seeds. Each operator retains a representative PyTorch numerical
integration check in Python.

## Every-function test audit

Categories are:

1. Python, PyTorch, or Hugging Face integration.
2. Dispatcher, schema, layout, error, inference-only, FakeTensor, or opcheck
   behavior.
3. Independent mathematical reference behavior.
4. Low-level CUDA behavior already owned by native CTest.

The tables below classify all 152 retained test functions. “All functions” is
an exhaustive classification of every `test_*` function in the named file.

### Integration suites: category 1

| File | Classification |
| --- | --- |
| `test_import.py` | All functions (1): package import/API integration. |
| `test_smollm2_config.py` | All functions (6): pinned identity, lazy import, config inspection, and mocked Hugging Face loading. |
| `test_smollm2_flux.py` | All functions (35): model transformation, category selection/dependencies, packed weights, checkpoint/state dict, eager/native/HF layers and logits, cache, fallbacks, hidden-state capture, and greedy generation. |
| `test_native_smollm2_prefill_runtime.py` | All functions (4): HF/Flux/native prompt logits and every-layer cache, public reuse/errors, native handoff, and greedy generation. |
| `test_native_smollm2_runtime.py` | All functions (2): repeated full-runtime HF/Flux/native logits and every-layer cache plus public reset. |

### Independent references: category 3

| File | Classification |
| --- | --- |
| `test_attention_score_softmax_reference.py` | All functions (4): independent unfused expression, broadcast, and reference validation. |
| `test_gqa_decode_attention_reference.py` | All functions (2): explicit grouped-head/valid-length and mask mathematics. |
| `test_packed_swiglu_reference.py` | All functions (2): explicit SiLU-times-up mathematics and reference validation. |
| `test_residual_rmsnorm_reference.py` | All functions (10): independent two-output mathematics, arbitrary dimensions/epsilon, preservation, gradients, layout, and validation. |
| `test_rmsnorm_reference.py` | All functions (6): PyTorch and Transformers equivalence, zero/determinism, and reference validation. |
| `test_rope_reference.py` | All functions (3): Transformers equivalence, batch broadcast, and mathematical dimension contract. |
| `test_softmax_reference.py` | All functions (11): PyTorch/autograd oracle, stability, probability invariants, determinism, layout, and validation. |

### Mixed custom-op suites: categories 1 and 2

For each row, the named numerical functions are category 1. Every other
retained function in that file is category 2; this explicitly includes each
argument/error contract, inference-only/autograd contract, device/layout
contract, output identity/alias contract, FakeTensor check, schema check, and
`torch.library.opcheck` invocation.

| File | Category-1 numerical integration functions | Category-2 remainder |
| --- | --- | ---: |
| `test_native_attention_score_softmax.py` | `test_matches_unfused_expression`; `test_supports_batch_head_and_query_mask_broadcast`; `test_makes_noncontiguous_inputs_contiguous_without_modifying_them` | 4 functions |
| `test_native_cublaslt_linear.py` | `test_cublaslt_linear_matches_pytorch_and_repeats`; `test_retained_explicit_configuration_matches_pytorch` | 1 function |
| `test_native_gqa_decode_attention.py` | `test_matches_reference_with_dynamic_and_static_cache_lengths`; `test_supports_none_and_broadcast_masks`; `test_reads_noncontiguous_cache_without_copying_or_modifying` | 5 functions |
| `test_native_packed_gate_up_swiglu.py` | `test_matches_fp32_reference_and_overwrites_poisoned_output` | 2 functions |
| `test_native_packed_qkv_rope_cache.py` | `test_matches_retained_rope_and_static_cache_update`; `test_reads_positive_stride_packed_projection_layout` | 4 functions |
| `test_native_softmax.py` | `test_matches_reference_across_widths` | 7 functions |
| `test_packed_swiglu_custom_op.py` | `test_matches_pytorch_across_shapes` | 7 functions |
| `test_residual_rmsnorm_custom_op.py` | `test_matches_reference_across_shapes` | 9 functions |
| `test_rmsnorm_custom_op.py` | `test_matches_reference_across_shapes` | 7 functions |
| `test_rope_custom_op.py` | `test_matches_transformers_for_prefill_and_decode`; `test_reads_non_contiguous_embeddings_without_copying_contract` | 3 functions |

No retained test function is category 4. Small device and deterministic-input
helpers remain local because their input contracts differ by operator; moving
them behind a shared abstraction would save negligible source and make the
independent integration cases less direct.

## Python CUDA coverage removed and native replacement

| Python reduction | Native owner | Retained Python boundary |
| --- | --- | --- |
| Removed the additional 4-D attention-shaped softmax numerical test. | `flux_test_softmax` checks native softmax widths, row behavior, stability, preservation, determinism, and streams. | One CPU/CUDA numerical integration case, noncontiguous behavior, errors, inference-only behavior, FakeTensor, and opcheck remain. |
| Removed the additional residual-RMSNorm epsilon sweep. | `flux_test_residual_rmsnorm` checks epsilon and shape matrices, exact residual output, numerical RMSNorm output, preservation, repeated execution, and streams. | One production-shape CPU/CUDA numerical integration case plus allocation, layout, errors, inference-only, FakeTensor, opcheck, and out contracts remain. |
| Reduced the gate/up direct CUDA numerical loop from three seeds to one. | `flux_test_decode_ops::test_packed_gate_up_swiglu` checks the production launcher directly; `flux_test_native_runtimes` exercises it in all 30 layers. | One poisoned-output PyTorch numerical comparison, wrapper validation, and FakeTensor/schema coverage remain. |

The broader replacement mapping from the preceding runtime consolidation still
applies: `flux_test_transformer_ops` owns attention-softmax, RoPE, and packed
SwiGLU launcher matrices; `flux_test_decode_ops` owns packed-QKV/cache, GQA, and
gate/up; and `flux_test_native_runtimes` owns production stream, graph,
allocation, address, device-state, handoff, reuse, and lifecycle behavior.

## Benchmark consolidation

The maintained Python benchmark layer is now:

1. `benchmark_final_system.py`: canonical checkpoint-backed HF/Flux/native
   logits, state dict, prefill, all-layer K/V and cache state, workspace/storage,
   prefill-to-decode continuation, generation, timing, and runtime audit.
2. `benchmark_smollm2_decode_profile.py`: distinct full-runtime native prefill
   and decode owner/launch/top-kernel attribution.
3. `smollm2_benchmark_utils.py`: shared deterministic runtime configuration,
   input construction, and argument parsing.

`benchmark_native_smollm2_prefill.py` is removed. Its duplicate model loading,
argument parsing, event timing, continuation, generation, cache validation,
audit, and reporting are no longer maintained separately. Its useful workspace
and exact all-layer native-versus-Flux prefill checks are in the canonical
benchmark. Fine-grained worst-element cache coordinates and the permissive
cache-drift diagnostic were development diagnostics, not final performance
claims; maximum cache errors remain reported and enforced at unchanged
tolerances. Prefill bottleneck attribution remains in the native profiler.

## Production Python

All production Python at the 342,517-byte baseline is retained. The audit found
no additional dead export, compatibility wrapper, duplicate production helper,
or wrapper for a removed execution path. In particular, `smollm2_flux.py`,
`native_smollm2.py`, every public custom-op wrapper, FakeTensor registration,
and dispatcher binding remain unchanged in this milestone.

## Validation

Validation used Windows, Python 3.11.9, PyTorch 2.14.0+cu132, CUDA 13.2, and an
NVIDIA GeForce RTX 5070 Ti (`sm_120`). No tolerance changed.

| Command | Result |
| --- | --- |
| `build\python3119\python.exe -m pytest -q tests\test_native_softmax.py tests\test_residual_rmsnorm_custom_op.py tests\test_native_packed_gate_up_swiglu.py` | 46 passed |
| `build\python3119\python.exe -m pytest -q` | 282 passed |
| `build\python3119\python.exe -m pytest -q -k "opcheck or fake_tensor or fake"` | 23 passed, 259 deselected |
| `scripts\native.cmd test` | 10/10 CTest targets passed |
| canonical final-system length-128 smoke | Checkpoint exact; HF/Flux/native logits, all-layer K/V, cache state, continuation, greedy generation, workspace report, stable addresses, and zero replay allocation growth passed. |
| reduced profiler length-128 prefill/decode smoke | Both workloads completed; zero allocation growth and stable addresses; native owner and top-kernel attribution reported. |
| `scripts\native.cmd benchmark -Warmup 1 -Samples 1` | All correctness-gated CUDA microbenchmarks completed. |
| `build\python3119\python.exe -m compileall -q benchmarks flux scripts tests` | Passed. |
| `git diff --check` | Passed. |

## Exact source accounting

The recursive metric excludes `.git`, `build`, virtual environments, caches,
generated metadata, and temporary directories.

| Language | Before | After | Change | Final share |
| --- | ---: | ---: | ---: | ---: |
| Python (`.py`) | 342,517 B | 325,056 B | -17,461 B | 41.624537% |
| CUDA (`.cu` + `.cuh`) | 306,421 B | 306,421 B | 0 B | 39.238261% |
| C++ (`.cpp` + `.cc` + `.cxx`) | 132,894 B | 132,894 B | 0 B | 17.017533% |
| headers (`.h` + `.hpp`) | 16,553 B | 16,553 B | 0 B | 2.119668% |

Total counted source is 780,924 bytes. With C++ and headers unchanged, CUDA
would have to equal the 474,503 non-CUDA bytes to be exactly 50% of the total.
The exact additional CUDA required is therefore **168,082 bytes**.

There is no defensible final CUDA subsystem that should be designed to a
168,082-byte source target. The native inference backend is complete, and
adding batched inference, sampling, another precision, or replacement GEMMs
solely to close this gap would expand scope and violate measurement-led
optimization. The recommendation is no byte-sized subsystem: use the retained
native profiler to select only a real end-to-end bottleneck, then size any
future CUDA work by its correct implementation and measured benefit.
