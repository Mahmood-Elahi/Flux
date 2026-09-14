# Native full-model decode runtime milestone

Milestone 27 extends the proven lifetime-managed one-layer runtime into one
native token-to-logits CUDA Graph for the complete pinned SmolLM2-135M FP32
decoder. It adds no speculative operator optimization and does not claim that
the project has met its separate 50% CUDA release requirement.

## Architecture and boundary

`torch.classes.flux.NativeSmolLM2Decode` is a lifetime-managed C++ custom class
implemented in the existing CUDA runtime translation unit. Its thin Python
adapter performs eager prefill, extracts the retained model tensors and compact
DynamicCache prefixes, constructs full-capacity RoPE tables, and installs the
first token. Native replay covers:

```text
CUDA token ID -> embedding gather -> 30 decoder layers -> final RMSNorm
              -> cuBLAS LM head -> stable [1, 1, 49152] logits
              -> one position/cache-length advance
```

Each compact layer descriptor owns strong references to input-norm, packed-QKV,
attention-output, post-attention-norm, packed gate/up, and down-projection
weights plus epsilon and attention scale. All descriptors call one shared layer
executor. Each layer preserves the Milestone 26 order: RMSNorm, selected
zero-workspace cuBLASLt packed-QKV, Q/K RoPE and compact cache write, grouped-GQA
decode, selected attention projection, residual plus RMSNorm, fused gate/up plus
SwiGLU, cuBLAS down projection, and cuBLAS residual addition.

K/V tensors are contiguous `[30, 1, 3, capacity, 64]`; no `repeat_kv` or
expanded nine-head cache exists. A single shared layer workspace is reused
across all sequential layers. Two 576-element hidden buffers ping-pong between
layers. The final norm reuses the norm region only after layer execution has
finished. This lifetime plan avoids 30 independent maximal workspaces.

The captured preparation kernel reads the device token and position, gathers
the embedding and RoPE row, and writes `attention_length = position + 1`.
Every layer writes K/V at the unchanged position and attends through the shared
length. A final state kernel advances position exactly once after logits.

## Graph and stream lifecycle

Construction allocates all stable storage, imports prefill K/V on the caller's
current stream, orders a private non-blocking capture stream with an event,
warms all retained library and CUDA paths, restores state, captures all 304
device launches, and instantiates the graph. Replay optionally copies one CUDA
token D2D, launches the graph on the caller's current PyTorch stream, records a
completion event, and returns the stable logits tensor without host
synchronization. An event orders consecutive and cross-current-stream calls.
Reset and diagnostic host reads are explicitly synchronous and outside replay.
Destruction waits only for outstanding work before releasing graph, stream,
event, and cuBLAS resources.

## Correctness

The canonical command was:

```powershell
build\python3119\python.exe benchmarks\benchmark_native_smollm2_runtime.py `
  --warmup 3 --samples 10 --generation-prompt 128 --generation-tokens 16 `
  --audit-capacity 4096 --audit-replays 10 `
  --json-output build\native_full_runtime_results.json
```

It loaded independent pinned Hugging Face and Flux model instances, used FP32
with TF32 disabled, and retained `rtol=2e-4`, `atol=2e-5`. At effective lengths
128, 512, 1024, 2048, 4096, and 8192, native logits passed against Hugging Face,
Flux eager, and Flux CUDA Graph. Every layer's K and V prefix passed against the
Python graph. Position and cache length matched exactly and stable addresses
did not change.

The largest native-versus-Hugging-Face logit error was `2.86102e-5` at length
8192. The largest native-versus-Flux-eager logit error was `2.28882e-5` at
length 1024. The largest native-versus-Python-graph logit error was
`1.71661e-5` at length 512. The largest K error was `5.72205e-6` at layer 18,
length 128; the largest V error was `4.17233e-6` at layer 29, length 128.
K/V were bit-exact at every layer for lengths 1024 through 8192. The short
differences reflect the retained Python graph's intentional short-cache
fallback, while native consistently executes the retained layer sequence.

A 16-token greedy continuation from a 128-token prompt matched the Python Flux
and pinned Hugging Face token IDs exactly. Focused tests additionally cover
construction, capture from installed state, repeated replay, all-layer cache
equality, stable addresses, zero replay allocation growth, non-default-stream
producer/runtime/consumer ordering, reset determinism, capacity exhaustion,
error handling, and destruction/recreation.

The focused native-runtime and retained one-layer commands passed 11/11 tests.
The complete repository regression command
`build\python3119\python.exe -m pytest -q` passed 472/472 tests.

## Performance

CUDA-event medians from ten same-process alternating samples were:

| Length | HF ms/token | Flux eager | Python graph | Native graph | Native tok/s | Native/Python graph | HF/native |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 22.5130 | 10.9134 | 2.6931 | 1.3719 | 728.9 | 1.963x | 16.410x |
| 512 | 39.7933 | 24.4612 | 2.8320 | 2.2394 | 446.6 | 1.265x | 17.770x |
| 1024 | 22.7572 | 11.4767 | 1.4379 | 1.3991 | 714.7 | 1.028x | 16.266x |
| 2048 | 35.7063 | 11.3216 | 1.4762 | 1.4569 | 686.4 | 1.013x | 24.508x |
| 4096 | 22.5369 | 11.3061 | 1.8503 | 1.6516 | 605.5 | 1.120x | 13.645x |
| 8192 | 22.0957 | 11.1081 | 2.2149 | 1.9163 | 521.9 | 1.156x | 11.531x |

Python-graph host enqueue medians ranged from 9.15 to 15.75 microseconds;
native custom-class replay ranged from 12.7 to 29.0 microseconds. The native
graph reduced device latency at every measured length. Exact capacities 128 and
512 compare the native retained sequence with the established safe Python
fallback and therefore show a structurally larger improvement.

At capacity 4096 the full native graph contains 304 launches per token versus
the comparable historical 315-launch Python graph: 183 Flux launches and 121
cuBLAS/cuBLASLt launches, with no remaining framework launch. The Flux count is
60 GQA launches (chunk plus reduction for 30 layers), 120 other per-layer Flux
kernels, final RMSNorm, and two state-management kernels. The library count is
four launches per layer plus the LM head. Across ten audited replays there was
zero allocation growth, zero allocator events, stable addresses, and no replay
host synchronization; the single profiler-visible synchronization is its
measurement boundary.

## Persistent memory

At capacity 8192 the runtime owns 377,487,360 K/V bytes, 180,992 shared
workspace bytes, and 196,632 stable token/logit/state bytes. Graph/runtime
resource bytes are driver-owned and not exposed by CUDA, so they are reported
as unmeasured rather than guessed. The object stably references 538,060,032
model-weight bytes already owned by the Flux model; it does not copy them.

## Remaining release constraint

Milestone 27 completes native full-model one-token decode control-plane
integration. With the same exclusions used by Milestone 26 (`.git`, `build`,
virtual environments, and `__pycache__`), exact source totals are:

| Language | Milestone 26 | Milestone 27 | Delta |
| --- | ---: | ---: | ---: |
| Python | 570,319 B | 604,975 B | +34,656 B |
| CUDA | 157,923 B | 192,931 B | +35,008 B |
| C++ | 140,734 B | 142,569 B | +1,835 B |
| Headers | 11,020 B | 13,588 B | +2,568 B |

CUDA share is now 20.22%. Under the required net metric, CUDA added minus
non-CUDA added is `35,008 - 39,059 = -4,051` bytes. The exact remaining
distance to equal CUDA and non-CUDA source is therefore 568,201 bytes. The
control plane necessarily added its thin Python integration, focused tests,
benchmark harness, bindings, and declarations; no implementation was duplicated
or padded for classification.

Flux is not declared finally complete: the remaining distance to 50% CUDA is
tracked in `ROADMAP.md`, and padding, per-layer duplication, Linguist overrides,
and resurrecting rejected operators remain prohibited.
