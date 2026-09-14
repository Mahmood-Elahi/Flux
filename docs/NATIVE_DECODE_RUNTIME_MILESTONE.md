# Native one-layer decode runtime milestone

Milestone 26 establishes the native lifetime and execution boundary required by
the final SmolLM2 runtime. It deliberately stops after one canonical decoder
layer; it does not claim end-to-end model integration or replace the Python
prefill and Hugging Face correctness paths.

## Implemented architecture

`torch.classes.flux.NativeSmolLM2LayerDecode` is a lifetime-managed C++ custom
class whose implementation resides in a CUDA translation unit. A narrow Python
adapter validates a fully enabled Flux model, extracts one layer's retained
weights and RoPE tables, and exchanges the initial layer-boundary hidden state
and prefilled K/V prefix. Python does not dispatch the layer's steady-state
operators.

The native object owns:

- stable input and output tensors;
- fixed-capacity unexpanded `[1, 3, capacity, 64]` K and V tensors;
- one scalar CUDA `int64` position/cache length, advanced by the retained packed
  QKV/RoPE/cache kernel;
- one flat FP32 workspace with lifetime-safe views for both norm outputs, packed
  QKV, compact query, GQA output/reduction, attention projection, SwiGLU, down
  projection, and two gathered RoPE rows;
- stable references to all layer weights and full-capacity RoPE tables;
- a non-blocking capture stream, setup/completion events, cuBLAS handle, CUDA
  graph, and executable graph.

At capacity 4096, native tensor ownership is 6,291,456 cache bytes, 100,352
workspace bytes, and 4,616 input/output/state bytes. Replay does not allocate.

Construction initializes state on the caller's current stream, orders the
private capture stream with an event, warms lazy library state, restores the
logical state, captures with CUDA runtime APIs, instantiates the executable, and
synchronizes only to finish setup. Replay waits on the previous completion event
when needed, optionally copies a new hidden state device-to-device, launches the
graph on the caller's current PyTorch CUDA stream, and records completion without
host synchronization. Reset and diagnostics are explicitly synchronous, outside
the steady-state contract. Destruction waits for outstanding work and releases
all graph, event, stream, and library resources; partial construction also has
exception-safe cleanup.

## Captured layer sequence

The graph composes the already-retained production launchers and exact library
paths in model order:

1. gather one RoPE table row with the sole new, narrowly scoped state kernel;
2. retained input RMSNorm;
3. retained selected zero-workspace cuBLASLt packed-QKV projection;
4. retained packed-QKV to RoPE to direct K/V cache update, including position
   advancement;
5. retained native GQA decode attention over unexpanded cache storage;
6. retained selected zero-workspace cuBLASLt attention-output projection;
7. retained fused residual plus post-attention RMSNorm;
8. retained fused packed gate/up GEMV plus SwiGLU;
9. exact FP32 cuBLAS down projection;
10. exact FP32 cuBLAS residual addition into stable layer output.

No new mathematical operator was added. Cache length is sufficient to enforce
one-token causal visibility, so the native path does not own an additive mask.

## Validation

The focused test constructs a deterministic canonical one-layer SmolLM2 model
and a literal one-layer form of the retained Python/Flux CUDA-Graph path. The
same weights, hidden state, K/V prefix, position, and capacity feed both paths.
The established `rtol=2e-4`, `atol=2e-5` remains unchanged.

| Effective length | Output max error | K max error | V max error | State/address result |
| ---: | ---: | ---: | ---: | --- |
| 128 | 2.38419e-7 | 2.68221e-7 | 2.38419e-7 | exact length/position; stable |
| 512 | 2.38419e-7 | 2.38419e-7 | 2.38419e-7 | exact length/position; stable |
| 1024 | 0 | 0 | 0 | exact length/position; stable |
| 2048 | 0 | 0 | 0 | exact length/position; stable |
| 4096 | 0 | 0 | 0 | exact length/position; stable |
| 8192 | 0 | 0 | 0 | exact length/position; stable |

The short-capacity difference is only FP32 rounding: the current Python graph
uses its established short-cache fallback while native execution consistently
uses the retained projection/GQA sequence. Tests also cover three consecutive
replays, poisoned workspace overwrite, reset, capacity exhaustion, destruction
and recreation, and a non-default-stream producer/runtime/consumer chain.

The profiler measured 10 native graph launches versus 50 in the short-fallback
Python graph at capacity 128, and 11 versus 19 at capacity 4096. Across ten
4096-capacity native replays it observed zero allocation growth, stable
addresses, no allocator events, and no host synchronization inside the marked
replay region.

## Performance

The reproducible command was:

```powershell
build\python3119\python.exe benchmarks\benchmark_native_smollm2_layer_runtime.py `
  --json-output build\native_layer_results.json
```

This used Python 3.11.9, PyTorch 2.14.0+cu132, CUDA 13.2, FP32, and an RTX 5070
Ti, with five warmups and 20 CUDA-event samples. Setup and validation were
untimed. The model has exact SmolLM2 layer geometry and deterministic synthetic
weights/state; this is a control-plane benchmark, not a model-quality workload.

| Length | Python graph ms | Native graph ms | Native speedup | Python host us | Native host us |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.0697 | 0.0342 | 2.039x | 15.30 | 18.30 |
| 512 | 0.0710 | 0.0462 | 1.535x | 15.15 | 17.70 |
| 1024 | 0.0414 | 0.0329 | 1.258x | 15.05 | 17.25 |
| 2048 | 0.0451 | 0.0369 | 1.223x | 14.70 | 18.20 |
| 4096 | 0.0567 | 0.0474 | 1.196x | 30.80 | 33.75 |
| 8192 | 0.0566 | 0.0448 | 1.264x | 14.50 | 18.15 |

Native replay reduces device latency at every measured capacity. Its bounded
Python custom-class call and stream/event bookkeeping add roughly 2-4 us of host
enqueue cost; that is the next control-plane optimization target, not a reason to
weaken lifetime or cross-stream ordering.

## Source footprint and next boundary

Using the milestone-25 exclusions (`.git`, `build`, virtual environments, and
`__pycache__`), the source totals are:

| Language | Before | After | Delta |
| --- | ---: | ---: | ---: |
| Python | 534,296 B | 570,319 B | +36,023 B |
| CUDA | 127,772 B | 157,923 B | +30,151 B |
| C++ | 137,612 B | 140,734 B | +3,122 B |
| Headers | 8,545 B | 11,020 B | +2,475 B |

CUDA share rises from 15.81% to 17.95%. Under the required distance metric,
`CUDA added + non-CUDA removed - non-CUDA added`, net progress is -11,469 bytes;
the distance to equal CUDA/non-CUDA source therefore changes from 552,681 to
564,150 bytes. This honest regression comes from the required Python oracle,
integration tests, benchmark orchestration, binding, and headers; no source was
padded or removed for classification.

The remaining work is milestone 27: generalize/describe multiple layers, own all
30 layer caches/workspaces and their graph sequence, add embedding input,
final RMSNorm, LM-head/logit output, and generation orchestration, then compare
complete logits, every-layer K/V, and greedy tokens with both Python Flux and the
pinned Hugging Face model. FP16/BF16, new projections, a new LM-head kernel, and
other models remain out of scope.
