# Python semantic reduction milestone

## Result

This final Python-reduction pass leaves Python with three validation duties:

1. independent mathematical reference behavior;
2. PyTorch schema, dispatcher, FakeTensor, opcheck, and public operator contracts;
3. whole-model Hugging Face, Flux eager, and native-runtime integration.

CUDA implementation detail is now owned by native CTest. No production file,
model computation, tolerance, C++, CUDA, or independent reference was changed.
`tests/test_references.py`, `tests/test_config.py`, and
`benchmarks/benchmark_utils.py` are unchanged.

## Operator-suite reduction

`tests/test_ops.py` decreased from 53,810 to 28,015 bytes. Each native
operator retains registration/schema coverage, one representative numerical
comparison against PyTorch or an independent Python/Transformers expression,
one CUDA invocation, CPU dispatch where it is public, FakeTensor/opcheck where
applicable, essential shape/dtype/layout/autograd errors, and output identity
or alias behavior where it is part of the API.

| Area | Removed or consolidated | Representative Python coverage retained |
| --- | --- | --- |
| RMSNorm and residual-RMSNorm | Separate allocation, contiguity, inference-mode, device mismatch, and large parametrized error tests | CPU/CUDA numerical and layout checks, compact error contracts, opcheck/FakeTensor, distinct outputs, out identity and aliasing |
| Softmax and attention softmax | Dtype/shape case parametrization and duplicate CPU/CUDA opcheck cases | One CPU/CUDA numerical oracle, compact errors, FakeTensor, one opcheck |
| RoPE and packed SwiGLU | Multiple prefill/decode shapes and equivalent shape matrices | One representative noncontiguous RoPE/Transformers comparison and one packed-SwiGLU comparison, plus errors/opcheck/out identity |
| GQA decode attention | Separate mask permutations and repeated cache-length cases | One dynamic-length numerical comparison with masked and unmasked calls, compact public errors, FakeTensor/opcheck, out identity |
| Packed QKV/RoPE/cache | Long-context Python case and repeated native-vs-native out comparison | One short representative cache mutation checked against RoPE plus explicit K/V updates, FakeTensor/opcheck/errors/out identity |
| cuBLASLt and gate/up GEMV | Second width, repeat/determinism assertion, and repeated setup | One representative PyTorch comparison and required schema/FakeTensor/output contracts |

Removed Python cases were CUDA correctness repetitions: shape and seed
matrices, long-context and boundary cases, current/non-default stream checks,
CUDA Graph replay, stable-address and allocation assertions, and device/runtime
lifecycle details. Native targets already exercise those behaviors directly:
`flux_test_rmsnorm`, `flux_test_residual_rmsnorm`, `flux_test_softmax`,
`flux_test_transformer_ops`, `flux_test_decode_ops`,
`flux_test_streaming_prefill_gqa`, and `flux_test_native_runtimes`.

## System-suite reduction

`tests/test_system.py` decreased from 52,076 to 19,823 bytes. The previous
small-model layer, prefill, decode, cache, and generation permutations were
replaced by deliberately distinct system checks:

- packed QKV and MLP weight/view/output semantics at one representative shape;
- exact ordinary state-dict and saved-checkpoint compatibility for both packed
  projection forms;
- one transformed Flux eager model against its independent Hugging Face model,
  including logits, K/V, installed categories, parameter reuse, and operator
  invocation;
- one GQA category test proving explicit opt-in and decode-only use;
- one consolidated public transformation/dependency/error contract;
- one canonical 30-layer HF/Flux/native prefill-to-repeated-decode chain,
  checking prefill and decode logits, cache position/length, greedy tokens, and
  every layer of compact K/V after prefill and continuation;
- public prefill reuse and decode reset behavior;
- Python-visible native input and capacity errors;
- exact eight-token native greedy generation identity against Hugging Face.

Removed system repetitions include three equivalent packed-projection lengths,
separate save/state-dict tests for each packed form, standalone layer and
module-equivalence paths already covered by the complete model, four CPU/CUDA
small-native-model permutations, three GQA fallback permutations, and separate
prefill/decode tests that repeated logits and every-layer cache comparisons.
Low-level stable-address, allocation, graph-lifecycle, and replay-buffer checks
were removed from pytest because `flux_test_native_runtimes` is authoritative.
System-level cache validation remains independent and compares native state to
both Hugging Face and Flux eager state.

## Benchmark audit

All five canonical modes remain: `system`, `prefill`, `decode`, `generation`,
and `profile`. CUDA-event regions, defaults, rotating timing order, deterministic
setup, correctness gates, workspace reporting, stable-state checks,
zero-allocation replay audit, generation correctness, and profiler attribution
are unchanged.

`benchmarks/benchmark_flux.py` decreased from 43,570 to 43,247 bytes. Profile
mode now reuses the canonical CUDA-event sample helper and canonical input-ID
helper instead of maintaining second implementations. No larger benchmark
framework was introduced and no mode or methodology was removed.

## Final ownership

| Responsibility | Python | CTest/CUDA |
| --- | --- | --- |
| Independent math | Yes | No |
| PyTorch schema/FakeTensor/opcheck | Yes | No |
| HF model equivalence | Yes | No |
| CUDA numerical sweeps | No | Yes |
| CUDA stream behavior | No | Yes |
| Graph lifecycle | No | Yes |
| Stable addresses | System smoke only if useful | Yes |
| Allocation invariants | No | Yes |
| Long-context kernel cases | No | Yes |

## Counts

| Measure | Before | After | Change |
| --- | ---: | ---: | ---: |
| Test files | 4 | 4 | 0 |
| Test functions | 140 | 79 | -61 |
| Collected pytest cases | 263 | 146 | -117 |
| `test_ops.py` | 53,810 B | 28,015 B | -25,795 B |
| `test_system.py` | 52,076 B | 19,823 B | -32,253 B |
| `benchmark_flux.py` | 43,570 B | 43,247 B | -323 B |

The final function distribution is 27 operator, 10 system, 35 reference, and
7 configuration tests. The final case distribution is 34 operator, 10 system,
95 reference, and 7 configuration cases.

## Validation

| Command | Result |
| --- | --- |
| `build\python3119\python.exe -m pytest -q` | 146 passed in 14.33 s |
| `build\python3119\python.exe -m pytest -q -k "opcheck or fake_tensor or fake"` | 12 passed, 134 deselected in 3.80 s |
| focused exact checkpoint, HF/Flux eager logits, HF/Flux/native prefill/decode/K/V/handoff, and greedy generation selection | 4 passed, 142 deselected in 7.21 s |
| `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\native.ps1 -Action test` | 10/10 CTest targets passed in 3.14 s |
| `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\native.ps1 -Action benchmark -Warmup 1 -Samples 1` | all correctness-gated CUDA microbenchmarks passed |
| `benchmark_flux.py --mode system` smoke | passed; exact checkpoint, correctness, stable addresses, and zero allocation growth |
| `benchmark_flux.py --mode prefill` smoke | passed |
| `benchmark_flux.py --mode decode` smoke | passed; stable addresses and zero allocation growth |
| `benchmark_flux.py --mode generation` smoke | passed |
| `benchmark_flux.py --mode profile` smoke | passed; decode and prefill attribution completed with zero allocation growth |

The benchmark smokes used length/prompt 8, zero warmup where accepted, one
sample/round/repetition, two generated tokens, and profile top-k 3. They are
correctness and CLI smoke runs, not performance claims.

## Source accounting

| Language | Files | Bytes | Distribution |
| --- | ---: | ---: | ---: |
| Python (`.py`) | 32 | 261,662 | 36.467047% |
| CUDA (`.cu` + `.cuh`) | 21 | 306,421 | 42.704974% |
| C++ (`.cpp` + `.cc` + `.cxx`) | 21 | 132,894 | 18.521037% |
| Headers (`.h` + `.hpp`) | 19 | 16,553 | 2.306942% |
| Total | 93 | 717,530 | 100% |

Python decreased by 58,371 bytes from 320,033 to 261,662 bytes. CUDA, C++,
and header bytes are unchanged. The exact remaining CUDA deficit to 50% is:

```text
(261,662 + 132,894 + 16,553) - 306,421 = 104,688 bytes
```

The reduction therefore decreased the deficit by exactly 58,371 bytes. This is
the final semantic reduction pass; further Python deletion is not recommended
unless a concrete dead-code defect is discovered.
