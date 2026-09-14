# Native fixed-shape one-token decode runtime design

## Implemented prompt-to-decode boundary (Milestone 28)

The production native boundary now begins with CUDA prompt token IDs rather
than after Python eager prefill. `NativeSmolLM2Prefill` runs embedding lookup,
all 30 decoder layers, final RMSNorm, the FP32 cuBLAS LM head, and argmax on the
caller's current stream. It writes K/V directly into the attached
`NativeSmolLM2Decode` object's compact `[30, 1, 3, capacity, 64]` storage and
installs position/cache length through the existing completion event. The next
`replay()` launches the retained full-model decode graph without a DynamicCache
conversion or cache copy.

Prefill uses one persistent lifetime-planned workspace and data-driven layer
descriptors. Its grouped three-head attention reads compact K/V directly,
performs causal softmax in-place in one `[9, S, S]` score tensor, and never
materializes `repeat_kv`. Exact cuBLAS paths remain responsible for learned
projections and the LM head. Repeated same-shape prompts are allocation-free and
address-stable; an event orders prefill and graph replay across current streams.
The detailed profile, validation, timing, memory, and source accounting are in
`NATIVE_PREFILL_RUNTIME_MILESTONE.md`.

## Implemented full-model boundary (Milestone 27)

The Milestone 27 decode boundary is token-to-logits. Python still loads the
checkpoint and tokenizer, validates configuration, performs eager prefill, and
selects the next token. After prefill, `NativeSmolLM2Decode` accepts one stable
CUDA `int64` token, performs the embedding lookup, runs all 30 decoder layers,
applies final RMSNorm and the unchanged FP32 library LM head, writes stable
logits, mutates all compact K/V caches, and advances device state. Python does
not dispatch decoder layers during replay.

The native object owns one contiguous K tensor and one contiguous V tensor with
layout `[30, 1, 3, capacity, 64]`. It also owns the stable input token, logits,
position, attention length, two ping-pong hidden buffers, one lifetime-reused
layer workspace, RoPE row storage, capture stream/events, cuBLAS handle, CUDA
Graph, and graph executable. It retains strong tensor references to embedding,
layer, final-normalization, LM-head, and full RoPE-table storage. Per-layer
descriptors hold the six learned tensors plus epsilon and attention scaling;
one executor consumes every descriptor rather than duplicating layer code.

One captured preparation kernel gathers the token embedding and RoPE row and
derives `attention_length = position + 1`. Every layer reads the same position
for its cache write and the same attention length for GQA. The packed-QKV
launcher therefore has an internal non-advancing form used by the full runtime,
while its existing public operator continues to advance its cache length.
After logits, one captured state kernel copies attention length to position.
This is the only state advance in a full decode step.

Layer scratch is shared across all 30 sequential layers. Two hidden buffers
alternate between layer input and output; norm, residual, QKV, query, attention,
attention projection, SwiGLU, down projection, RoPE-row, and GQA reduction
regions are reused once their consumers finish. No per-layer maximal workspace
is allocated. K/V storage is the only capacity-scaled per-layer allocation.

Construction orders a private non-blocking capture stream after the caller's
current PyTorch stream, warms every retained kernel/library path, restores
logical state, captures the complete token-to-logits sequence, instantiates the
graph, and completes setup synchronously. Replay optionally enqueues one D2D
token copy, launches the graph on the caller's current PyTorch CUDA stream, and
records a completion event. Consecutive calls and calls made from different
current streams are ordered by an event without a host wait. Reset and explicit
diagnostics synchronize; steady replay does not.

The token-ID boundary was chosen because embedding lookup is a simple stable
gather and avoids returning model-internal orchestration to Python. The new
kernel exists for that functional boundary, not source-language accounting.
The LM head remains an exact FP32 cuBLAS GEMM; the rejected custom LM-head GEMV
was not restored.

## Historical design context

## Boundary and invariants

The next phase moves steady-state decode ownership below Python without changing
the model's mathematics or the ordinary reference path. Python continues to own
checkpoint/tokenizer loading, configuration, public convenience APIs, prefill,
and high-level reference validation. A native C++/CUDA runtime owns one captured,
fixed-capacity, batch-one, one-token decode state.

The first supported contract is the final SmolLM2-135M FP32 inference geometry.
The internal descriptors should still carry layer count, head counts, head
dimension, hidden/intermediate/vocabulary sizes, epsilon, RoPE parameters, and
capacity explicitly so validation is not hidden in hard-coded pointer offsets.
Unsupported shape/dtype/training paths stay in Python; there is no silent native
fallback inside a captured runtime.

## Current Python ownership to move

`flux/model/smollm2_cuda_graph.py` currently owns all control-plane state:

- eager prefill and DynamicCache-to-StaticCache copying;
- stable token, position, mask, logits, K/V, and per-operator scratch tensors;
- projection-plan initialization;
- side-stream allocator warmup;
- `torch.cuda.CUDAGraph` capture/replay;
- installing scratch attributes on every Python layer/module;
- per-replay input copying and host `steps_replayed` bookkeeping;
- mask-column update, model invocation, position increment, reset, memory
  accounting, and diagnostic address reporting.

`flux/model/smollm2_flux.py` currently performs per-layer Python dispatch,
chooses retained/fallback branches, reaches into Hugging Face `StaticCache`, and
threads the shared scratch object through patched module attributes. These are
the steady-state responsibilities that move to native code.

## Proposed components

### Python adapter

Add a narrow `NativeDecodeRuntime` Python wrapper alongside the existing graph
runtime. It validates that a model is eval/FP32/final-category enabled, extracts
ordered retained weight tensors and scalar configuration, runs eager prefill,
and passes the populated K/V tensors into native construction. It retains a
reference to the model/runtime so all borrowed weight storages outlive graph
execution. The existing Python graph implementation remains the correctness
oracle until end-to-end parity is proven.

Public operations should be minimal:

```text
capture(model, prompt, max_decode_steps) -> runtime
runtime.replay(optional_cuda_token) -> stable_cuda_logits
runtime.reset(prefill_cache, position)   # outside steady-state timing
runtime.memory() / runtime.addresses()   # diagnostics only
```

### C++ runtime object

Register a lifetime-managed C++ object through the existing extension (a
`torch::CustomClassHolder` or equivalently narrow pybind class). It owns:

- strong `at::Tensor` references for embeddings, all layer weights, final norm,
  and LM head;
- stable input IDs, position/cache-length state, attention mask or equivalent
  validity state, logits, K/V cache, and all scratch tensors;
- validated per-layer descriptors and preselected cuBLASLt plans;
- CUDA graph, executable graph, capture stream/event bookkeeping, capacity, and
  replay count;
- explicit device guard and stream-contract checks.

Construction, cache import, warmup, and capture may synchronize because they
are reported setup costs. `replay` must enqueue token copy, graph launch, and
state advance on the caller's current PyTorch CUDA stream and return immediately.
Diagnostic host reads may synchronize but must be visibly separate from replay.

### CUDA state and cache

Use native-owned contiguous per-layer K/V tensors in unexpanded GQA layout,
logically `[layers, batch, kv_heads, capacity, head_dim]` (or an equivalent set
of per-layer views). Import the eager prefill once with device-to-device copies.
The steady path must not depend on Python `StaticCache` objects.

Keep the authoritative position/valid length in device memory. Prefer one
runtime position scalar shared by all layers; per-layer cache-length fields are
unnecessary when layer execution is lockstep. A small captured state-update
kernel should expose the current position to RoPE/cache/attention, update mask
or validity metadata if still required, and advance exactly once after all
layers. Host replay bounds remain defensive API bookkeeping, not the computation
source of truth.

### Per-layer executor

Represent each layer as a native descriptor of tensor handles, projection plans,
epsilon, head geometry, and scratch offsets. The captured layer sequence is:

1. input RMSNorm into stable scratch;
2. retained packed-QKV projection through the selected cuBLASLt plan;
3. retained fused QKV/RoPE/cache update;
4. retained native one-token GQA attention;
5. retained attention-output projection through cuBLASLt;
6. fused residual plus post-attention RMSNorm;
7. retained fused gate/up GEMV+SwiGLU;
8. down projection through the existing exact library path;
9. residual update for the next layer.

After the last layer, run final RMSNorm and the unchanged exact LM-head library
operation into stable logits. Residual ordering, RMSNorm epsilon/weights, RoPE,
GQA mapping/scale/mask, and cache writes must match the Python final path.

Existing operator launchers should be factored behind small internal C++ headers
where needed so the runtime calls them directly on the current stream. It should
not redispatch through Python or allocate operator outputs during capture. This
refactoring is code ownership, not permission to introduce new kernels.

### Workspace

Allocate one lifetime-analyzed workspace during construction. Preserve the
already-proven buffer sharing only when producer/consumer lifetimes do not
overlap. Include norm/residual, QKV projection, compact query, GQA output and
reduction workspace, attention projection, SwiGLU activation, down-projection
output, final norm, logits, and cuBLASLt workspace. Record offsets/sizes and
expose total bytes for validation. No steady-state allocator calls are allowed.

### Graph lifecycle and streams

Capture is owned natively with CUDA graph APIs on a PyTorch-compatible capture
stream after all lazy library initialization and warmup. The runtime must use
`CUDAGuard`, the caller's current CUDA stream, and explicit events when moving
between the current and capture streams. It must never silently use the legacy
default stream. Graph instantiation errors and asynchronous launch errors must
surface through the extension's normal error handling.

Replay on a different current stream is supported only with correct event
ordering; otherwise the first milestone should explicitly bind the runtime to
its capture stream/device and reject unsafe use. There are no per-token device
synchronizations. Reset/destruction must wait only where resource lifetime
requires it.

## Validation gates

The native runtime is not production-retained until it passes:

- deterministic single-layer intermediate comparisons against the Python Flux
  layer at short and long capacities;
- complete logits and every-layer K/V comparison for repeated one-token replay;
- identical greedy token continuation against Python Flux and the pinned HF
  oracle;
- position/mask/cache-length boundary tests at 512/513 and maximum capacity;
- stable-address, poisoned-workspace overwrite, zero replay-allocation, and graph
  reset/exhaustion tests;
- non-default-current-stream producer/runtime/consumer ordering;
- graph launch inventory proving native layer orchestration actually replaced
  Python/dispatcher work;
- the full existing Python suite and the canonical final-system benchmark with
  unchanged tolerances and methodology.

Native unit tests should directly cover state-update code, descriptor validation,
workspace layout, cache import/reset, current-stream behavior, and graph replay.
Python tests continue to own HF integration and public API behavior.

## Historical Milestone 26 implementation boundary

Implement the native decode control plane and one-layer executor, without adding
new performance kernels:

1. expose reusable internal launch interfaces for the already-retained kernels
   and selected cuBLASLt paths;
2. add a native state object that owns stable tensors, one layer's K/V cache,
   workspace, device position, and CUDA graph lifecycle;
3. capture and replay the exact final one-token computation for a single
   SmolLM2 decoder layer using existing launchers/library calls;
4. validate intermediates, K/V mutation, repeated replay, stable addresses,
   zero allocations, and non-default-stream ordering against the Python layer;
5. benchmark only to verify overhead/launch behavior, not to claim an
   end-to-end model speedup yet.

That milestone established the hard ownership boundary used by the implemented
30-layer runtime above. The remaining project-level release work is not another
decode algorithm: preserve this validated architecture while addressing the
separate 50% CUDA source-share requirement without padding, duplication, or
speculative operators.
