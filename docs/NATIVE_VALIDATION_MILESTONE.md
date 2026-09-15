# Native CUDA validation and benchmark milestone

This milestone starts from canonical commit `ac962fa` and the source baseline
recorded there: 635,115 Python bytes, 228,861 CUDA bytes, 144,435 C++ bytes,
and 15,701 header bytes. The initial CUDA share is 22.3473% and the exact
CUDA/non-CUDA deficit is 566,390 bytes.

## Pre-change Python validation audit

The tables below record the classification made before any retirement. The
classification numbers are: **1**, Python integration that remains; **2**,
CUDA-specific coverage moving native; **3**, mixed coverage retained or split;
**4**, duplicate/superseded infrastructure; and **5**, historical development
infrastructure whose result is already recorded.

### Benchmarks

| File | Class | Decision and ownership |
| --- | ---: | --- |
| `benchmark_flux.py --mode system` | 1 | Retain as the canonical Hugging Face/Flux eager/Flux graph full-system benchmark. |
| `benchmark_flux.py --mode profile` | 1 | Retain unique full-model kernel/category attribution. |
| `benchmark_smollm2_gqa_decode_attention.py` | 3 | Retain model, graph, cache-memory, and Python-versus-native A/B coverage; native timing moves to the CUDA benchmark. |
| `benchmark_smollm2_gate_up_gemv.py` | 3 | Retain exact full-layer/full-graph A/B coverage; direct kernel timing moves native. |
| `benchmark_rope.py` | 3 | Retain Transformers and integrated prefill/decode comparisons; direct kernel timing moves native. |
| `benchmark_native_smollm2_runtime.py` | 1 | Retain full 30-layer decode/Hugging Face/Python-graph correctness and benchmark orchestration. |
| `benchmark_native_smollm2_prefill.py` | 1 | Retain full prompt-to-cache Hugging Face/Python-Flux equivalence and prefill/decode handoff benchmark. Its timed workload is distinct from decode. |
| `benchmark_native_smollm2_layer_runtime.py` | 5 | Retire: the complete 30-layer runtime supersedes the one-layer development harness; results remain in `NATIVE_DECODE_RUNTIME_MILESTONE.md` and Git history. |
| `benchmark_rmsnorm.py` | 2 | Retire after equivalent direct production-kernel correctness and CUDA-event timing exist natively. |
| `benchmark_residual_rmsnorm.py` | 2 | Retire after equivalent two-output native correctness and timing exist. |
| `benchmark_softmax.py` | 2 | Retire after native width coverage and the native 8192-wide benchmark exist. |
| `benchmark_attention_score_softmax.py` | 2 | Retire after native broadcast/mask correctness and direct fused-kernel timing exist. |
| `benchmark_gqa_long_context.py` | 2 | Retire after native short/8192 correctness and 8192 grouped-reduction timing exist; historical reduction evidence remains in `GQA_LONG_CONTEXT_MILESTONE.md`. |
| `benchmark_utils.py` | 1 | Retain shared full-model setup, deterministic inputs, cache cloning, and alternating event timing. |
| `.gitkeep` | 4 | Remove because the directory is populated. |

The native microbenchmark does not replace full-model benchmarks. It calls the
production launchers directly, validates output before timing, uses a
non-default stream, CUDA events, warmups, repeated samples, and median latency.

### Tests

| File | Class | Decision and Python responsibility retained |
| --- | ---: | --- |
| `test_import.py` | 1 | Package import contract. |
| `test_smollm2_config.py` | 1 | Pinned Hugging Face configuration and loading. |
| `test_smollm2_flux.py` | 1 | Opt-in model substitution, checkpoint/state-dict, cache, logits, generation, and model semantics. |
| `test_smollm2_cuda_graph.py` | 3 | Python graph API plus model/cache equivalence; native kernels now own low-level launch/stream behavior. |
| `test_stable_decode_outputs.py` | 3 | PyTorch out-op schemas, FakeTensor, stable scratch installation, and Python graph integration. |
| `test_rmsnorm_reference.py` | 1 | Explicit PyTorch/autograd/Transformers oracle. |
| `test_residual_rmsnorm_reference.py` | 1 | Explicit PyTorch/autograd two-output oracle. |
| `test_softmax_reference.py` | 1 | Explicit PyTorch/autograd softmax oracle. |
| `test_rope_reference.py` | 1 | Transformers RoPE oracle and broadcast semantics. |
| `test_packed_swiglu_reference.py` | 1 | PyTorch SiLU/multiply oracle. |
| `test_attention_score_softmax_reference.py` | 1 | PyTorch fused-expression oracle. |
| `test_gqa_decode_attention_reference.py` | 1 | Explicit unexpanded-GQA and mask oracle. |
| `test_rmsnorm_custom_op.py` | 3 | Retain dispatcher, CPU/CUDA boundary, inference-only behavior, FakeTensor, and opcheck; standalone CUDA owns geometry/stream/numeric coverage too. |
| `test_residual_rmsnorm_custom_op.py` | 3 | Retain dispatcher, aliasing, inference-only, FakeTensor, and opcheck coverage. |
| `test_native_softmax.py` | 3 | Retain dispatcher, PyTorch comparison, FakeTensor, and opcheck coverage. |
| `test_rope_custom_op.py` | 3 | Retain Transformers comparison, dispatcher, FakeTensor, and opcheck coverage. |
| `test_packed_swiglu_custom_op.py` | 3 | Retain layout, dispatcher, inference-only, FakeTensor, and opcheck coverage. |
| `test_native_attention_score_softmax.py` | 3 | Retain tensor broadcasting, dispatcher, FakeTensor, and opcheck coverage. |
| `test_native_gqa_decode_attention.py` | 3 | Retain PyTorch cache abstraction, masks, schemas, FakeTensor, and opcheck; short/long direct kernel math moves native too. |
| `test_native_packed_qkv_rope_cache.py` | 3 | Retain StaticCache-compatible tensor/schema/opcheck behavior; device state and graph launch move native too. |
| `test_native_cublaslt_linear.py` | 3 | Retain tensor/schema/FakeTensor boundary and selected cuBLASLt plan behavior. The raw launcher shares PyTorch's cached plan implementation, so this remains its direct production test. |
| `test_native_packed_gate_up_swiglu.py` | 3 | Retain PyTorch schema/FakeTensor boundary; exact-shape math and stream launch move native too. |
| `test_native_smollm2_layer_runtime.py` | 3 | Retain one-layer construction/API regression coverage even though its benchmark is historical. |
| `test_native_smollm2_runtime.py` | 3 | Retain full-runtime Python/graph equivalence, every-layer cache comparison, reset, capacity, addresses, allocation, and current-stream checks. Runtime construction requires ATen tensors and the 30-layer model, so it remains integration coverage. |
| `test_native_smollm2_prefill_runtime.py` | 3 | Retain full-model logits/cache/handoff equivalence, reuse, capacity, allocation, addresses, current stream, and generation. |

No Python test is retired merely because the low-level CUDA behavior now also
has a standalone native test. PyTorch public contracts and cross-implementation
oracles remain in Python.

### Scripts

| File | Class | Decision and ownership |
| --- | ---: | --- |
| `reference_inference.py` | 1 | Retain public pinned Hugging Face inference entry point. |
| `flux_inference.py` | 1 | Retain public reference-versus-Flux inference smoke entry point. |
| `check_rmsnorm_cpp.py` | 4 | Retire; CTest runs the production C++ implementation directly. |
| `check_rmsnorm_cuda.py` | 4 | Retire; CTest runs the production CUDA implementation directly. |
| `check_residual_rmsnorm_cpp.py` | 4 | Retire; direct native two-output correctness supersedes binary-file IPC. |
| `check_residual_rmsnorm_cuda.py` | 4 | Retire; direct native CUDA correctness supersedes binary-file IPC. |
| `check_softmax_cpp.py` | 4 | Retire; CTest runs native CPU softmax directly. |
| `check_softmax_cuda.py` | 4 | Retire; native width/stability/current-stream coverage supersedes binary-file IPC. |
| `.gitkeep` | 4 | Remove because the directory is populated. |

`scripts/native.ps1` replaces the six executable-wrapper layers with one small
build/test/benchmark entry point. It discovers Visual Studio through
`vswhere`, imports the active 64-bit compiler environment, and uses the CUDA
toolkit selected by `CUDA_PATH`.

The paired `rmsnorm_cli.cpp`/`rmsnorm_cuda_cli.cu`,
`residual_rmsnorm_cli.cpp`/`residual_rmsnorm_cuda_cli.cu`, and
`softmax_cli.cpp`/`softmax_cuda_cli.cu` translation units are class 4 native
infrastructure as well: they only exchange binary files with the retiring
Python wrappers. The direct CPU and CUDA CTest targets supersede them, so the
six CLI sources are retired while the production launchers and native tests
remain.

## Native hierarchy

`csrc/tests/` contains the shared CUDA-aware test utility and CTest targets.
The pre-existing RMSNorm, residual-RMSNorm, and softmax CUDA tests are registered
without duplicating them. New direct tests cover fused attention-score softmax,
RoPE, packed SwiGLU, packed QKV/RoPE/cache, one-token grouped GQA at short and
8192-token context, and exact-shape packed gate/up + SwiGLU. The packed-QKV test
also captures and replays the production launcher in a CUDA Graph and verifies
device-resident position advancement and the non-advancing variant.

An ATen-backed native C++/CUDA executable constructs the production 30-layer
decode and prefill classes directly. It uses a non-default PyTorch CUDA stream
and validates graph replay, reset, capacity exhaustion, stable addresses,
device-state advancement, no change in device allocation state across replay
or prefill reuse, compact cache layout, and prefill/decode handoff. Its decode
graph also executes the retained production cuBLASLt projection configuration,
so the raw wrapper is covered without creating a second plan implementation.

`csrc/benchmarks/` contains one selectable microbenchmark executable. Each case
allocates and initializes outside the timed region, checks correctness before
timing, launches on a non-default stream, uses CUDA events, performs warmups,
and reports median latency. It covers RMSNorm, residual RMSNorm, softmax,
attention-score softmax, RoPE, packed SwiGLU, packed QKV/RoPE/cache, 8192-token
GQA, and packed gate/up + SwiGLU.

The cuBLASLt projection wrapper and full prefill/decode objects also remain in
the Python integration suite because those tests own Hugging Face/Python Flux
equivalence, public custom-class behavior, and full learned model weights. The
native executable complements rather than replaces those integration oracles.

## Reproducible commands

```powershell
scripts\native.cmd build-tests
scripts\native.cmd test
scripts\native.cmd build-benchmarks
scripts\native.cmd benchmark -Filter gqa -Warmup 10 -Samples 51
build\python3119\python.exe -m pytest -q
```

The one-layer benchmark is reproducible from canonical commit `ac962fa`; its
published measurements and configuration remain in
`docs/NATIVE_DECODE_RUNTIME_MILESTONE.md`. Retired operator benchmark results
remain in their milestone reports and at the same canonical commit. Current
measurements are reproduced by the native executable, while integrated claims
continue to use the retained Python full-model benchmarks.

## Validation results

The verified Windows Python 3.11.9, PyTorch 2.14.0+cu132, CUDA 13.2, MSVC
19.51, and RTX 5070 Ti (`sm_120`) environment produced:

| Validation | Result |
| --- | --- |
| `scripts\native.cmd test` | 9/9 CTest targets passed: 3 CPU, 6 CUDA, including the native runtime target. |
| `scripts\native.cmd benchmark -Warmup 2 -Samples 5` | All 9 correctness-gated microbenchmarks completed. |
| `build\python3119\python.exe -m pytest -q` | 476 passed in 15.58 s. |
| canonical final-system length-128 smoke | Exact state dict; logits/K/V within tolerance; positions, addresses, and greedy token identity passed. |
| native prefill length-128 smoke | Logits/K/V and continuation within tolerance; greedy identity and position/cache length passed. |
| native decode length-128 smoke | Reference/Flux/Python-graph logits and caches within tolerance; token identity and stable addresses passed. |

The validation-only five-sample native medians were 0.010688 ms RMSNorm,
0.011200 ms residual RMSNorm, 0.030336 ms softmax, 0.030688 ms fused attention
softmax, 0.014784 ms RoPE, 0.033984 ms packed SwiGLU, 0.009184 ms packed
QKV/RoPE/cache, 0.026592 ms 8192-token GQA, and 0.011200 ms packed gate/up +
SwiGLU. These low-sample smoke numbers verify the benchmark path and are not
new performance claims.

## Updated 8192-token prefill-attention profile

A fresh correctness run measured native prefill at 534.854 ms for 8192 tokens
(one timing sample, so this is diagnostic rather than a publication-quality
benchmark). A separate warmed profiler replay reported 517 launches and
531.481 ms of device time: 355.330 ms in library kernels and 176.151 ms in
native kernels, with zero allocation growth and stable addresses.

The in-place causal softmax kernel alone consumed 162.028 ms, 91.98% of native
kernel time and 30.49% of the profiled total. The two dominant aggregated SGEMM
kernel families consumed 233.668 ms and 121.522 ms; this aggregation includes
both projection and attention GEMMs, so it is evidence of the matrix-multiply
load but not a claim that all 355.190 ms belongs to attention.

At length 8192, the score tensor is exactly
`9 * 8192 * 8192 * sizeof(float) = 2,415,919,104` bytes (2.25 GiB). It accounts
for 87.87% of the runtime's 2,749,368,576-byte (2.561 GiB) persistent prefill
workspace. The quadratic score materialization is therefore both the dominant
workspace consumer and the input to the largest identifiable native kernel.

The recommended final milestone is a tiled/streaming FP32 grouped-GQA prefill
attention implementation with online softmax. It should preserve compact
three-head K/V, avoid `repeat_kv`, use current-stream asynchronous launches,
and be validated against the retained native prefill and Hugging Face oracles.
The first acceptance gate should be removal of the full score allocation;
latency should then be measured at 128 through 8192 rather than assumed from
reduced traffic. This milestone does not implement that redesign.

## Source accounting

Recursive extension accounting excludes `.git`, `build`, `.venv`, and
`__pycache__`:

| Language | Before | After | Change |
| --- | ---: | ---: | ---: |
| Python `.py` | 635,115 B | 550,093 B | -85,022 B |
| CUDA `.cu`, `.cuh` | 228,861 B | 272,017 B | +43,156 B |
| C++ `.cpp`, `.cc`, `.cxx` | 144,435 B | 132,894 B | -11,541 B |
| headers `.h`, `.hpp` | 15,701 B | 15,701 B | 0 B |

Detected source totals 970,705 bytes, of which CUDA is 28.022623%. Non-CUDA is
698,688 bytes, leaving an exact 426,671-byte deficit to equal CUDA and
non-CUDA. Progress under the required net metric is:

```text
CUDA added + non-CUDA removed - non-CUDA added
= 43,156 + 96,563
= 139,719 bytes
```

That resolves 24.668338% of the initial 566,390-byte deficit. Flux is not yet
at the 50% CUDA source-share completion threshold.
