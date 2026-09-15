# Milestone 35: device-resident native greedy generation

Milestone 35 closes the remaining Python control loop in the production
SmolLM2-135M generation path. Prompt prefill selects the first token with the
shared native argmax. Every later decode replay selects its successor, writes
it into the stable next-token input, appends it to a device-owned output buffer,
and advances generation/cache state in the captured graph. One C++ call
launches that graph a fixed number of times on PyTorch's current CUDA stream.

The pinned Hugging Face model and manually Python-orchestrated native path remain
independent correctness and performance references. This milestone does not
change weights, logits, GQA, RoPE, masks, cache layout, attention scaling,
residual ordering, RMSNorm, or retained projection implementations.

## Retained architecture

| Stable CUDA state | Shape / type | Purpose |
| --- | --- | --- |
| current token | `[1, 1]`, `int64` | embedding input for the next replay |
| generated tokens | `[cache_capacity]`, `int64` | fixed-capacity result storage |
| generation step | scalar `int64` | next generated-output index |
| cache position / length | scalar `int64` | existing decode/cache state |
| logits | `[1, 1, 49152]`, `float32` | LM-head output and argmax input |

Prefill writes token zero and `generation_step=1`. The decode CUDA Graph then
executes the retained token-to-logits graph followed by one
`greedy_argmax_update` kernel. That kernel reduces all 49,152 FP32 logits,
installs the selected token as the next replay's input, appends it at the current
generation step, increments the step, and advances position from cache length.

`NativeSmolLM2Prefill.generate_greedy(max_new_tokens)` validates once. Its
attached decode runtime waits for prefill once, submits `max_new_tokens - 1`
graph launches in C++, records one completion event, and returns a view of the
stable output buffer. There is no Python per-token loop, token `.item()`,
per-token allocation, host/device token transfer, event record, or
synchronization in steady-state generation.

```python
generated = native_smollm2_greedy_generate(
    flux_model, prompt_ids, max_new_tokens=32
)
```

The helper performs setup, one native generation call, and one final
`torch.cat` with the prompt. The result is an ordinary CUDA tensor.

### Selection semantics

The retained kernel is a one-CTA, 256-thread strided scan plus deterministic
shared-memory pair reduction. Candidates are ordered by value and then lower
vocabulary index, preserving PyTorch's first-index tie behavior. All-negative
values, infinities, endpoint maxima, and equal maxima are covered directly.
NaNs follow observed PyTorch CUDA behavior: NaN wins over non-NaN and the first
NaN wins ties. The launcher is stream-aware, allocation-free, asynchronous,
and reports launch errors without synchronizing.

### Fixed-length contract and EOS

Requests must fit captured cache capacity, and a runtime must be prefilled or
reset before reuse. EOS early termination is deferred: it needs a separate
device active-state and output-length contract. The current API deliberately
generates exactly `max_new_tokens` tokens.

## Correctness and validation

The CUDA test covers deterministic random 49,152-wide logits, all-negative
inputs, extreme and endpoint maxima, ties, all negative infinity, repeated
determinism, fused state updates, and a non-default stream. Runtime tests compare
directly with ATen CUDA argmax and verify graph chaining, positions, capacity,
stable addresses, and reuse protection. Python validates actual checkpoint
sequences across Hugging Face, manual Python-native decode, and device-native
generation.

| Command | Result |
| --- | --- |
| `scripts\\native.cmd test` | 11/11 CTest targets passed; 8 CUDA and 3 CPU |
| `build\\python3119\\python.exe -m pytest -q` | 146 passed |
| `build\\python3119\\python.exe -m pytest -q -k "opcheck or fake_tensor or fake"` | 12 passed, 134 deselected |
| focused native prefill/decode/generation selection | 3 passed, 7 deselected |
| canonical `--mode system --lengths 128` smoke | passed: checkpoint, logits/K/V, state, greedy identity, addresses, and allocation audit |

No tolerance changed. A broader pre-existing system run stopped on an FP32
cache-value tolerance edge: maximum absolute differences were `2.371054e-5` at
length 2048 and `3.389269e-5` at 4096 against the established `2e-5` absolute
tolerance. Milestone generation gates passed at prompts 128, 512, 1024, 2048,
and 4096.

## Argmax measurements and decision

CUDA-event medians on RTX 5070 Ti (`sm_120`), CUDA 13.2, PyTorch
2.14.0+cu132, with 50 warmups and 201 samples:

| Candidate / retained case | Median |
| --- | ---: |
| PyTorch CUDA `argmax` primitive | 0.019520 ms |
| native one-CTA argmax | 0.040256 ms |
| native argmax + token/buffer/step/position update | 0.041216 ms |

PyTorch argmax wins in isolation, but placing it inside this raw native graph
requires dispatcher/current-stream coupling and a second kernel for persistent
state. The one-kernel native epilogue is retained because it is allocation-free,
capture-safe, owns the state transition, and improves integrated generation. A
multi-stage partial reduction was rejected before retention: 192 KiB of logits
does not justify workspace plus another launch in this launch-heavy graph. No
rejected implementation remains in production source.

## End-to-end generation

The benchmark validates exact checkpoint tokens before timing. Decode covers
the `N-1` post-prefill tokens; values are medians of three repetitions.

| Output | Prompt | Native Python decode | Native device decode | Change |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 128 | 23.253 ms | 22.989 ms | -1.14% |
| 16 | 1024 | 22.668 ms | 22.509 ms | -0.70% |
| 16 | 4096 | 29.314 ms | 29.064 ms | -0.85% |
| 32 | 128 | 48.309 ms | 47.738 ms | -1.18% |
| 32 | 512 | 46.500 ms | 45.977 ms | -1.12% |
| 32 | 1024 | 46.530 ms | 46.224 ms | -0.66% |
| 32 | 2048 | 48.009 ms | 47.344 ms | -1.39% |
| 32 | 4096 | 60.027 ms | 59.683 ms | -0.57% |
| 64 | 128 | 100.427 ms | 99.392 ms | -1.03% |
| 64 | 1024 | 94.577 ms | 93.625 ms | -1.01% |
| 64 | 4096 | 122.037 ms | 121.413 ms | -0.51% |

The retained path wins decode execution at every pair. Setup noise can still
move short-output observed latency. At 64 tokens, device-native observed rates
were 568.92, 521.79, and 275.99 generated tokens/s for prompts 128, 1024, and
4096. Before this milestone Python submitted one replay and ATen argmax per
post-prefill token, including a device-to-device token install; it did not use
`.item()` or a host token copy. The retained path makes one native call and zero
external per-token copies.

The complete 32-token comparison, including capture/setup in native observed
totals, was:

| Prompt | HF total | Native Python total | Native device total | HF / Python / device generated tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1083.627 ms | 62.588 ms | 61.787 ms | 29.53 / 511.28 / 517.90 |
| 512 | 914.262 ms | 65.992 ms | 65.610 ms | 35.00 / 484.91 / 487.73 |
| 1024 | 770.984 ms | 75.281 ms | 76.211 ms | 41.51 / 425.08 / 419.89 |
| 2048 | 787.114 ms | 99.489 ms | 99.034 ms | 40.65 / 321.64 / 323.12 |
| 4096 | 1198.264 ms | 169.668 ms | 170.342 ms | 26.71 / 188.60 / 187.86 |

| Per post-prefill token | Before | After |
| --- | ---: | ---: |
| Python loop iterations | 1 | 0 |
| Python graph replay calls | 1 | 0 |
| Python/ATen argmax dispatches | 1 | 0 |
| external token D2D installs | 1 | 0 |
| token H2D / D2H copies | 0 / 0 | 0 / 0 |
| host synchronization / `.item()` | 0 / 0 | 0 / 0 |

The improvement is removal of Python/dispatcher/copy orchestration, not a claim
that the old implementation performed a host token readback.

## Fresh retained-system profile

The profile uses five measurements/repetitions. `Profiled GPU` is summed GPU
kernel time; `Median` is independent CUDA-event latency. Host/runtime overhead
is not a GPU kernel and is bounded by the difference between those independently
sampled numbers, not subtracted as paired samples. Addresses stayed stable and
allocation growth was zero everywhere.

| Context | Median | Profiled GPU | Launches | cuBLAS | GQA | Greedy/state |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1.4498 ms | 1.6261 ms | 274 | 0.6021 ms | 0.3628 ms | 0.0490 ms |
| 512 | 2.4833 ms | 2.4743 ms | 274 | 0.5647 ms | 1.2325 ms | 0.0486 ms |
| 1024 | 1.5463 ms | 1.6167 ms | 304 | 0.5616 ms | 0.3470 ms | 0.0496 ms |
| 2048 | 1.5775 ms | 1.6977 ms | 304 | 0.5960 ms | 0.3787 ms | 0.0467 ms |
| 4096 | 1.7934 ms | 1.8894 ms | 304 | 0.6264 ms | 0.5853 ms | 0.0502 ms |
| 8192 | 2.1029 ms | 2.1650 ms | 304 | 0.5634 ms | 0.8889 ms | 0.0501 ms |

`Flux other` is 0.5264--0.5561 ms at contexts 512--8192 (0.5522 ms at 128)
and includes the greedy epilogue. Decode's 91 identically named GEMV launches
aggregate projections and LM head, so one duration cannot safely be assigned to
LM head. The same profile isolates the single post-prefill LM-head GEMV at
0.1398 ms (length 2048), 0.1404 ms (4096), and 0.1812 ms (8192). Greedy/state is
about 2.3--3.4% of decode GPU time. At 4096 and 8192, grouped GQA remains the
dominant scaling term and the next evidence-based target.

Prefill is unchanged except that final-token selection uses the shared argmax.
Its fresh profile is:

| Length | Median | Profiled GPU | Streaming GQA | cuBLAS projections | Packed QKV/RoPE/cache | Packed SwiGLU | Norm kernels | Workspace | Launches |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 45.0723 ms | 45.3383 ms | 20.6517 ms | 21.3571 ms | 0.5745 ms | 0.5772 ms | 0.5780 ms | 83,364,096 B | 369 |
| 4096 | 103.6828 ms | 104.0366 ms | 56.3487 ms | 40.7799 ms | 1.1644 ms | 2.2888 ms | 1.3046 ms | 166,725,888 B | 339 |
| 8192 | 281.6308 ms | 283.2383 ms | 187.9269 ms | 79.9896 ms | 2.6457 ms | 5.6438 ms | 3.1313 ms | 333,449,472 B | 339 |

`Norm kernels` sums RMSNorm and residual-RMSNorm. The profiler reports the 120
per-layer GEMMs as one cuBLAS group, so it cannot truthfully split packed QKV,
attention-output, gate/up, and down-projection GEMM time. Their non-GEMM
consumers are shown separately. Residual-add and attention-head row-layout work
account for another 0.6359, 1.2369, and 2.8838 ms at the three lengths. Prefill
argmax is 0.0339, 0.0338, and 0.0356 ms. Addresses remain stable and allocation
growth is zero.

## Source accounting

The recursive metric excludes `.git`, `build`, virtual environments, caches,
generated metadata, and temporary directories.

| Language | Before | After | Change | Final share |
| --- | ---: | ---: | ---: | ---: |
| Python (`.py`) | 261,662 B | 266,084 B | +4,422 B | 35.988861% |
| CUDA (`.cu` + `.cuh`) | 306,421 B | 321,764 B | +15,343 B | 43.519790% |
| C++ (`.cpp` + `.cc` + `.cxx`) | 132,894 B | 133,693 B | +799 B | 18.082480% |
| headers (`.h` + `.hpp`) | 16,553 B | 17,810 B | +1,257 B | 2.408869% |

Total counted source is 739,351 bytes. CUDA is 95,823 bytes short of equal
CUDA/non-CUDA bytes. Added native code implements the generation boundary and
its direct evidence; no padding, duplicate subsystem, or reclassification was
introduced.

## Decision

**KEEP** the shared exact FP32 argmax, fused graph epilogue, stable device state,
and fixed-count C++ replay loop. Keep manual Python-native generation only as a
benchmark oracle; production uses device-resident generation. **DEFER** EOS
early stopping pending a device output-length contract. **NEXT:** investigate
long-context grouped GQA only through measured retained-system profiles.
