# Streaming FP32 grouped-GQA prefill attention

## Retained result

Milestone 30 replaces the long-context native prefill attention score matrix
with a single-launch streaming CUDA kernel specialized for SmolLM2-135M. The
kernel reads query as `[9, S, 64]`, reads compact K/V as `[3, capacity, 64]`,
maps each query head to `kv_head = query_head / 3`, applies the causal boundary
inside the tile loop, and writes `[9, S, 64]`. It never calls `repeat_kv`, never
creates nine-head K/V storage, and requires zero global attention workspace.

A measured bounded fallback retains the previous cuBLAS QK, native causal
softmax, and cuBLAS PV path for lengths 129 through 1024. Length 128 uses the
streaming kernel; lengths 2048 through 8192 use it unconditionally. Therefore
the only quadratic score allocation remaining in production is capped at
9 x 1024 x 1024 FP32 values (36 MiB), independent of maximum context. The
2.25 GiB length-8192 score matrix is eliminated.

## Previous architecture and bottleneck

Each layer previously launched three grouped QK GEMMs, materialized a complete
`[9,S,S]` score tensor, launched a separate scale/mask/softmax kernel, then
launched three grouped probability-V GEMMs. At length 8192 this used
2,415,919,104 bytes for scores and 2,749,368,576 bytes for the full prefill
workspace. A warmed profile measured 531.481 ms device time, including 162.028
ms in causal softmax; the complete native prefill took approximately 534.854 ms
in that diagnostic run.

## Streaming algorithm and numerical formulation

For each causal query row the kernel carries FP32 state `(m, l, o)`. Given a
new score `s = scale * dot(q, k)`, it applies:

```text
m_new = max(m, s)
alpha = exp(m - m_new)
beta  = exp(s - m_new)
l_new = alpha * l + beta
o_new = alpha * o + beta * v
```

After the last causal key, the result is `o / l`. The rescaling prevents
overflow and remains stable through length 8192. Scores, masks, and
probabilities are never written to global memory. The only attention-local
storage is a shared 32 x 64 FP32 K tile and matching V tile (16 KiB total), plus
register-held query, maximum, normalization, and output state.

Tiles wholly beyond the last query row owned by a CTA are not loaded. Within a
partly causal tile each row computes exactly
`min(valid_keys, query_position - first_key + 1)` keys, so future positions are
neither scored nor accumulated.

## Kernel organization and dispatcher

All retained variants instantiate one readable template; kernels are not
duplicated per head or layer. A CTA is scheduled for one query head, derives
the compact KV head by integer division, and reuses each shared K/V tile across
several query rows. Aligned `float4` transactions stage K/V. Subgroup shuffle
reductions form D=64 dot products where more than one lane owns a row.

The measured fixed dispatcher is:

| Sequence length | Query rows/CTA | Threads | Lanes/query |
| ---: | ---: | ---: | ---: |
| <=128 | 8 | 256 | 32 |
| 129--384 | 32 | 256 | 8 |
| >=385 | 128 | 128 | 1 |

The one-lane form removes unnecessary shuffle instructions and keeps all 64
query and output values in registers. It is selected for every production
streaming length above the bounded fallback.

The production runtime uses the caller's PyTorch current CUDA stream. The
launcher is asynchronous apart from normal launch-error reporting and performs
no allocation or device synchronization.

## Tile investigation

CUDA-event medians guided the retained organization. Representative length-8192
results were:

| Candidate | 8192 latency |
| --- | ---: |
| Grouped CTA, three Q heads reuse K/V, query tiles 8/16 | 35.5/35.7 ms |
| Cooperative matrix tiles 8/16 | 43/46 ms |
| Independent query-head CTA, tiles 8/16/32 | 43.6/29.6/20.57 ms |
| Independent query-head CTA, tiles 64/128 | 17.35/17.40 ms |
| Key tile 64 instead of 32 | 17.36 ms (no improvement) |
| Tilewise softmax, query tiles 32/64/128 | 28.6/40.4/73.1 ms |
| One lane/query, 256/128/64/32 threads | 10.205/8.204/9.507/16.119 ms |
| Retained one lane/query after shuffle removal | 6.156 ms |

The grouped-query CTA was rejected despite shared KV reuse: tripling the
per-row Q/output state increased register pressure and reduced useful
parallelism. A 64-key tile did not beat 32. Tilewise probability storage and
cooperative matrix organizations increased synchronization and register/shared
memory pressure. Compensated and multi-pass softmax experiments cost up to
14.4 ms at 8192 and did not improve full-model drift. All rejected production
variants were removed.

At the short crossover points, 31-sample medians were:

| Length | tile 8 | tile 32 | tile 128 |
| ---: | ---: | ---: | ---: |
| 128 | 0.0320 | 0.0357 | 0.0482 ms |
| 192 | 0.0644 | 0.0508 | 0.0580 ms |
| 256 | 0.0728 | 0.0689 | 0.0771 ms |
| 384 | 0.1461 | 0.1057 | 0.1098 ms |
| 448 | 0.1797 | 0.1364 | 0.1203 ms |
| 512 | 0.2331 | 0.1550 | 0.1427 ms |

Although streaming narrowly won the isolated 512 case, whole-prefill results
favored the retained implementation at 512 and 1024. The simple 129--1024
fallback was therefore kept; it also preserves exact Python Flux accumulation
ordering in that range.

## Isolated correctness and performance

The native CUDA test compares against two independent oracles: a double-precision
CPU causal-attention implementation for arbitrary scores, and the retained
cuBLAS QK -> native softmax -> cuBLAS PV sequence at every required length.
It covers all nine query heads, all three KV groups, tile boundaries,
non-aligned lengths, a non-default stream, repetition, determinism, invalid
arguments, allocation stability, and zero workspace.

| Length | Maximum error vs retained | Retained | Streaming |
| ---: | ---: | ---: | ---: |
| 128 | 8.94070e-8 | 0.145760 ms | 0.033504 ms |
| 512 | 1.19209e-7 | 0.166912 ms | 0.143168 ms |
| 1024 | 1.22935e-7 | 0.252800 ms | 0.293376 ms |
| 2048 | 1.34110e-7 | 0.975520 ms | 0.650112 ms |
| 4096 | 1.56462e-7 | 3.712576 ms | 1.801440 ms |
| 8192 | 1.56462e-7 | 14.422624 ms | 6.155904 ms |

Uniform-score length tests through 8192 had maximum absolute error
5.96046e-8. Arbitrary short/boundary cases peaked at 1.19209e-7. All are well
inside the established `rtol=2e-4`, `atol=2e-5` operator policy.

## Workspace

"Other prefill" is the lifetime-planned non-score workspace. Cache numbers use
the benchmark's three-token continuation capacity where space remains.

| Length | Previous attention | New attention | Other prefill | New total prefill | K/V cache | Persistent state |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 589,824 B | 0 B | 5,212,416 B | 5,212,416 B | 6,036,480 B | 393,248 B |
| 512 | 9,437,184 B | 9,437,184 B fallback | 20,842,752 B | 30,279,936 B | 23,731,200 B | 393,248 B |
| 1024 | 37,748,736 B | 37,748,736 B fallback | 41,683,200 B | 79,431,936 B | 47,324,160 B | 393,248 B |
| 2048 | 150,994,944 B | 0 B | 83,364,096 B | 83,364,096 B | 94,510,080 B | 393,248 B |
| 4096 | 603,979,776 B | 0 B | 166,725,888 B | 166,725,888 B | 188,881,920 B | 393,248 B |
| 8192 | 2,415,919,104 B | 0 B | 333,449,472 B | 333,449,472 B | 377,487,360 B | 393,248 B |

At 8192 the prefill workspace falls by 2,415,919,104 bytes, from about 2.561
GiB to 318.00 MiB. The fallback's score storage has a fixed 36 MiB maximum;
the long-context production workspace is linear in sequence length.

## Integrated performance

The after results are 10-sample CUDA-event medians with three warmups. Before
values are the canonical Milestone 29 run from the same target system.

| Length | HF | Python Flux | Native before | Native after | After tok/s | Before/after |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 25.176 ms | 26.973 ms | 6.592 ms | 6.116 ms | 20,929.6 | 1.078x |
| 512 | 25.795 ms | 14.869 ms | 12.314 ms | 12.339 ms | 41,494.5 | 0.998x |
| 1024 | 33.365 ms | 26.982 ms | 21.769 ms | 21.778 ms | 47,021.0 | 1.000x |
| 2048 | 94.787 ms | 70.587 ms | 55.647 ms | 44.610 ms | 45,908.7 | 1.247x |
| 4096 | 297.702 ms | 221.829 ms | 162.576 ms | 103.683 ms | 39,505.0 | 1.568x |
| 8192 | 1,073.509 ms | 923.300 ms | 548.288 ms | 283.349 ms | 28,911.3 | 1.935x |

There is no material integrated regression: the bounded fallback keeps 512 and
1024 within measurement noise of their retained results while long-context
speed improves substantially.

## Full-model numerical behavior

The standard logit and continuation assertions remain enabled. The explicit
`--report-cache-drift` benchmark option disables only the elementwise cache
assertion so reordered long-context cache differences can be measured after
the isolated kernel has passed its independent oracle checks.

| Length | Logit error vs Python Flux | Max K error vs Python Flux | Max V error vs Python Flux | Continuation error vs Python Flux | Greedy identity |
| ---: | ---: | ---: | ---: | ---: | --- |
| 128 | 1.90735e-5 | 1.23978e-5 | 1.04904e-5 | 2.28882e-5 | yes |
| 512 | 0 | 0 | 0 | 0 | yes |
| 1024 | 0 | 0 | 0 | 0 | yes |
| 2048 | 2.86102e-5 | 4.67300e-5 | 3.40939e-5 | 3.81470e-5 | yes |
| 4096 | 2.47955e-5 | 6.67572e-5 | 5.24521e-5 | 4.00543e-5 | yes |
| 8192 | 1.41144e-4 | 1.08433e-3 | 5.40018e-4 | n/a (capacity full) | yes |

The maximum overall K/V deltas at 8192 occur in layer 29: K index
`[batch=0, kv_head=1, position=7753, dimension=24]` and V index
`[0,1,7753,50]`. The isolated attention delta remains 1.56462e-7, showing that
the larger last-layer cache
numbers are deterministic amplification of FP32 operation reordering over 30
layers rather than softmax instability. The strict default cache assertion
passes through length 2048; at 4096 and 8192 it reports sparse outliers and is
not silently relaxed. Greedy tokens match pinned Hugging Face and Python Flux
for every established prompt and continuation case.

## Updated profile and launches

At 8192, the production replay falls from 517 to 337 GPU launches: 216 native,
121 library, and zero framework launches. This is exactly 180 fewer launches,
because seven attention launches per layer become one across 30 layers.

The new three-replay profile attributes about 188.40 ms/prefill to streaming
attention, 80.06 ms to library kernels, and about 14.42 ms to all other native
kernels. Streaming attention is now the dominant component (about 66.5% of
device time), followed by projection GEMMs. The old global QK score writes,
162.028 ms causal softmax, probability matrix, and separate PV reads no longer
appear at long context. Per the milestone boundary, no unrelated operator was
optimized after this profile.

The audit recorded zero allocation growth, zero allocator events, stable
addresses, zero framework GPU launches, and only the two profiler-induced
synchronization events.

## Source accounting

| Language | Before | After | Change |
| --- | ---: | ---: | ---: |
| Python | 550,093 B | 551,695 B | +1,602 B |
| CUDA (`.cu` + `.cuh`) | 272,017 B | 306,421 B | +34,404 B |
| C++ | 132,894 B | 132,894 B | 0 B |
| Headers | 15,701 B | 16,553 B | +852 B |

The counted source total is 1,007,563 bytes. CUDA is 30.412093%, up from
28.022623%. Non-CUDA totals 701,142 bytes, so the exact remaining distance to
50% CUDA is 394,721 bytes. Net milestone progress is 31,950 bytes: CUDA added
minus non-CUDA added. The milestone does not attempt to close the remaining
deficit with unrelated code.

## Validation

The retained implementation passed:

- all 10 native CTest targets;
- all 476 Python tests;
- native prefill correctness/timing through 8192;
- native decode benchmark smoke at length 1024;
- canonical final-system benchmark smoke;
- native prefill-to-decode continuation and greedy-token checks;
- Python compileall and `git diff --check` (run at milestone close).

## Limitations

- Production is specialized to B=1, FP32, Hq=9, Hkv=3, D=64, causal masks,
  and capacity at most 8192.
- The 129--1024 bounded fallback still materializes scores, capped at 36 MiB.
- Long-context reordered cache state is not elementwise-close to Python Flux at
  every near-zero element under the old assertion, although the isolated
  operator passes, logits/continuations remain bounded, and greedy output is
  identical.
- Attention remains the principal 8192 device-time component; this milestone
  does not pursue unrelated kernels.
