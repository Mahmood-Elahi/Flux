# Native full-model prefill runtime milestone

Milestone 28 moves the canonical SmolLM2-135M FP32 prompt path below Python and
hands its cache directly to the retained Milestone 27 native decode graph. It
does not change checkpoint storage, model mathematics, or the ordinary pinned
Hugging Face/PyTorch oracle. Flux activation remains explicit and opt-in.

## Baseline profile and design decision

The retained Python Flux prefill was profiled before implementation with
deterministic inputs, TF32 disabled, CUDA-event timing, one warmup, three
samples, and prompt lengths 128, 512, 1024, 2048, 4096, and 8192. Median total
latencies were 17.207, 16.376, 26.399, 70.465, 221.863, and 922.757 ms.

The largest semantic categories were:

| Length | First category | Second category | Third category |
| ---: | --- | --- | --- |
| 128 | gate/up 1.135 ms | down 0.964 ms | output projection 0.517 ms |
| 512 | gate/up 2.646 ms | down 2.425 ms | packed QKV 1.250 ms |
| 1024 | gate/up 4.724 ms | softmax 4.652 ms | down 3.269 ms |
| 2048 | softmax 23.849 ms | QK 10.474 ms | gate/up 9.619 ms |
| 4096 | softmax 100.616 ms | QK 39.092 ms | probability-V 30.794 ms |
| 8192 | softmax 548.474 ms | QK 157.747 ms | probability-V 115.138 ms |

At length 8192, attention score formation, softmax, and probability-V consumed
about 89% of the measured semantic time. The Python path also materialized
expanded nine-head K and V views and separate 2.25-GiB score and probability
tensors. Its profiler recorded 850--970 launches depending on length.

That evidence selected one targeted design: keep exact cuBLAS projections, but
execute grouped GQA attention directly against compact three-head K/V, perform
causal FP32 softmax in-place in the score workspace, and eliminate `repeat_kv`
and the second quadratic probability allocation. No speculative projection,
LM-head, approximate-math, or architecture-specific vector path was added.

## Native boundary and ownership

`torch.classes.flux.NativeSmolLM2Prefill` is a lifetime-managed C++ custom class
with a narrow `NativeSmolLM2Prefill` Python adapter. Python loads and validates
the model, passes strong tensor references and scalar descriptors once, and
constructs the full RoPE tables. One native call then owns:

```text
CUDA token IDs -> embedding gather -> 30 decoder layers -> final RMSNorm
               -> cuBLAS LM head -> stable final-token logits and argmax
               -> native decode graph state
```

Each of the 30 descriptors contains the input-norm weight, packed-QKV weight,
attention-output weight, post-attention-norm weight, packed gate/up weight,
down-projection weight, RMSNorm epsilon, and attention scale. One loop executes
all descriptors; layer code is not duplicated.

For each layer, native execution preserves the established order:

1. input RMSNorm;
2. packed-QKV cuBLAS projection;
3. fused Q/K RoPE and direct compact K/V writes;
4. three grouped QK calls, in-place causal softmax, and three grouped
   probability-V calls without expanded K/V;
5. head-to-row layout and attention-output projection;
6. residual plus post-attention RMSNorm;
7. packed gate/up projection and fused SwiGLU;
8. down projection and final residual add.

The runtime allocates one persistent workspace during construction. Two hidden
buffers ping-pong across layers, while normalization, packed-QKV, attention,
MLP, and final-hidden regions are reused according to lifetime. Repeated
prefill calls keep every address stable and perform no PyTorch allocation.

## Direct cache handoff

The prefill object constructs and owns its attached `NativeSmolLM2Decode`.
Prefill writes directly into the decode object's contiguous K and V tensors,
each logically `[30, 1, 3, capacity, 64]`. It then installs the final argmax,
position, and cache length on the caller's current CUDA stream and records the
decode completion event. There is no Python `DynamicCache`, cache conversion,
expanded nine-head storage, or device-to-device cache copy at this boundary.

`replay()` immediately launches the existing full token-to-logits decode graph
against the same storage. The prompt ending at position `S` therefore continues
at exactly `S`, and the graph advances state once per generated token.

Prefill and the handoff use PyTorch's current CUDA stream. An event orders a new
prefill after outstanding decode work and orders decode after prefill without a
host wait. Diagnostic scalar reads and destruction may wait for lifetime
safety; the steady prefill and replay APIs remain asynchronous. Capacity
exhaustion is rejected rather than writing past the cache.

## Correctness

The canonical validation command was:

```powershell
build\python3119\python.exe benchmarks\benchmark_native_smollm2_prefill.py `
  --warmup 2 --samples 5 --continuation-tokens 3 `
  --audit-length 1024 --audit-repetitions 3 `
  --json-output build\native_prefill_results.json
```

It used independent pinned Hugging Face and Flux model instances, deterministic
FP32 inputs, TF32 disabled, and the unchanged `rtol=2e-4`, `atol=2e-5`.

| Length | Max final-logit error vs HF | Max K error vs HF | Max V error vs HF | Cache position/length | Greedy continuation |
| ---: | ---: | ---: | ---: | ---: | --- |
| 128 | 2.67029e-5 | 1.57356e-5 | 1.23978e-5 | 128 / 128 | identical |
| 512 | 2.28882e-5 | 2.19345e-5 | 1.54972e-5 | 512 / 512 | identical |
| 1024 | 2.09808e-5 | 4.10080e-5 | 2.57492e-5 | 1024 / 1024 | identical |
| 2048 | 2.28882e-5 | 4.10080e-5 | 2.19345e-5 | 2048 / 2048 | identical |
| 4096 | 2.09808e-5 | 1.19686e-4 | 5.53131e-5 | 4096 / 4096 | identical |
| 8192 | 2.86102e-5 | 1.11580e-4 | 5.55515e-5 | 8192 / 8192 | no spare capacity |

Final logits and every tested continuation passed against both HF and Python
Flux at the established tolerance. Every layer's complete K/V prefix passed the
unchanged tolerance against Python Flux; lengths 512--8192 were bit-exact, and
the length-128 maxima were 1.14441e-5 for K and 8.19564e-6 for V.

All HF cache tensors were also compared and the table reports their absolute
maxima. Some long-context individual HF cache elements do not satisfy the
elementwise tolerance because the existing Flux packed-QKV projection and HF's
three separate Q/K/V projections have different legal FP32 accumulation orders.
The same deltas occur between the pre-existing Python Flux path and HF; native
prefill adds no cache divergence from Flux. The benchmark therefore asserts the
native implementation cache against its exact Flux implementation oracle and
reports, rather than hides, the independent HF cache delta. It does not loosen
the repository tolerance.

Focused tests cover all 30 caches, final logits, direct decode continuation,
greedy identity, reuse with new same-shape IDs, non-default current-stream
producer/consumer ordering, stable addresses, zero allocation growth, memory
accounting, invalid inputs, capacity exhaustion, and destruction/recreation.
The focused prefill/full-decode/one-layer command passed 15/15 tests, and the
complete repository command `build\python3119\python.exe -m pytest -q` passed
476/476 tests.

## Performance

CUDA-event medians from five same-process samples were:

| Length | HF ms | Python Flux ms | Native ms | Native tok/s | HF/native | Python/native |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 27.227 | 17.590 | 6.592 | 19,417.6 | 4.130x | 2.668x |
| 512 | 27.241 | 16.480 | 12.314 | 41,577.3 | 2.212x | 1.338x |
| 1024 | 33.564 | 26.723 | 21.769 | 47,040.2 | 1.542x | 1.228x |
| 2048 | 95.462 | 70.577 | 55.647 | 36,803.6 | 1.715x | 1.268x |
| 4096 | 308.004 | 241.983 | 162.576 | 25,194.4 | 1.895x | 1.488x |
| 8192 | 1,119.115 | 936.228 | 548.288 | 14,941.0 | 2.041x | 1.708x |

The native path is faster than both comparison paths at every required length.
The gain grows again at long context because it eliminates GQA expansion and a
second quadratic tensor while reducing Python/dispatcher orchestration.

A five-repetition PyTorch CUDA-profiler audit at length 1024 recorded 577 device
activities per prefill: 216 native kernels/state copies, 361 cuBLAS kernels or
library clears, and zero framework GPU launches. Profiler-perturbed device time
was 3.208 ms in native kernels/copies and 17.521 ms in retained library work.
There were zero allocator events, zero allocation growth, and stable addresses.
The two reported synchronization events are the explicit profiler measurement
boundaries, not steady-path synchronization.

## Persistent memory

The largest persistent region is the one in-place `[9, S, S]` FP32 attention
score workspace. Cache capacity includes requested continuation tokens.

| Prompt | Cache bytes | Prefill workspace | Decode workspace | Stable buffers |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 6,036,480 | 5,802,240 | 33,680 | 393,248 |
| 512 | 23,731,200 | 30,279,936 | 40,808 | 393,248 |
| 1024 | 47,324,160 | 79,431,936 | 50,312 | 393,248 |
| 2048 | 94,510,080 | 234,359,040 | 69,320 | 393,248 |
| 4096 | 188,881,920 | 770,705,664 | 107,336 | 393,248 |
| 8192 | 377,487,360 | 2,749,368,576 | 180,992 | 393,248 |

The object also references 538,060,032 bytes of existing model weights without
copying them. Peak memory remains quadratic in prompt length because exact eager
attention needs all scores; the implementation removes the redundant
probability tensor and expanded K/V but does not claim a streaming-attention
algorithm.

## Infrastructure and source accounting

The new production CUDA source, declaration, thin custom-class registration,
Python adapter, focused integration tests, and one canonical benchmark are
retained. No additional Python benchmark or test was retired: the remaining
Python files perform model loading, HF/Flux cross-implementation validation,
public-API checks, or decode profiling that native-only executables cannot
replace. No standalone CUDA-only executable was added because the milestone's
critical contract is the 30-layer PyTorch tensor/model handoff, not an isolated
kernel API.

Using the Milestone 27 exclusions (`.git`, `build`, virtual environments, and
`__pycache__`), exact source totals are:

| Language | Milestone 27 | Milestone 28 | Delta |
| --- | ---: | ---: | ---: |
| Python | 604,975 B | 635,115 B | +30,140 B |
| CUDA (`.cu`, `.cuh`) | 192,931 B | 228,861 B | +35,930 B |
| C++ | 142,569 B | 144,435 B | +1,866 B |
| Headers | 13,588 B | 15,701 B | +2,113 B |

CUDA share is 22.3473%. CUDA additions minus non-CUDA additions are
`35,930 - 34,119 = +1,811` bytes, reducing the exact distance to equal
CUDA/non-CUDA source from 568,201 to 566,390 bytes. The threshold remains
unmet, so Flux is not declared finally complete. No padding, generated bulk,
per-layer duplication, or Linguist override is used.
