# Residual-accumulating one-token projection investigation

Date: 2026-09-12

## Decision

Reject both residual-accumulating projection candidates. The retained production
path is unchanged.

`torch.addmm(residual, input, weight.t(), out=residual)` did use the intended
in-place C input / nonzero-beta cuBLAS path. Profiler traces contained no
separate copy or pointwise-add kernel. On the tested PyTorch/cuBLAS stack,
however, each `addmm` selected a two-kernel implementation: a GEMV followed by a
cuBLASLt split-K reduction. Ordinary bias-free `linear` used one GEMV kernel.
This made attention strictly worse and made MLP slower despite removing its
pointwise residual-add kernel.

Environment: SmolLM2-135M revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, FP32, B=1, Q=1, Python 3.11.9,
PyTorch 2.14.0+cu132, CUDA 13.2, Transformers 5.16.1, RTX 5070 Ti (`sm_120`),
TF32 disabled, deterministic algorithms enabled, and deterministic safety fills
enabled. Timings use CUDA events with warmups and medians. Device-category times
come from 20 replay profiler batches and therefore differ slightly from
unprofiled replay medians.

The retained baseline was reproduced with:

```powershell
build/python3119/python.exe benchmarks/benchmark_smollm2_stable_buffers.py `
  --capacities 2048,4096 --warmup 3 --repetitions 10 `
  --correctness-tokens 2 --profile-capacity 4096 --skip-independent-pools
```

Candidate measurements used the same loader, runtime configuration, warmups,
CUDA-event timing, and profiler categorization. The experimental model switches
were removed after the rejection decision, as required by the retention policy.

## Retained CUDA Graph launch profile

The retained stable-buffer graph has 345 launches at both capacities.

| Category | Launches | 2048 device ms | 4096 device ms |
|---|---:|---:|---:|
| Packed QKV projection GEMMs | 30 | 0.1348 | 0.1425 |
| Attention output projection GEMMs | 30 | 0.1151 | 0.1142 |
| Packed gate/up MLP GEMMs | 30 | 0.3133 | 0.3125 |
| MLP down-projection GEMMs | 30 | 0.1746 | 0.1745 |
| LM-head GEMM | 1 | 0.1450 | 0.1351 |
| Native GQA attention (chunk + reduce) | 60 | 0.6410 | 0.7338 |
| Packed-QKV/RoPE/cache kernel | 30 | 0.0544 | 0.0550 |
| RMSNorm/residual-RMSNorm kernels | 61 | 0.1475 | 0.1397 |
| Packed SwiGLU | 30 | 0.0257 | 0.0255 |
| Cache/state kernels | 2 | 0.0031 | 0.0031 |
| Remaining framework kernels | 36 | 0.0375 | 0.0383 |
| Copies/fills/other | 5 | 0.0038 | 0.0039 |
| **Total** | **345** | **1.7959** | **1.8781** |

The 122 cuBLAS launches are the 121 listed model GEMMs plus one small rotary
position-frequency GEMM classified as framework work. Flux custom kernels
account for 181 launches. The other 42 launches are framework, state, copy, and
fill work.

## Retained residual boundaries

### Attention boundary

Per layer, the sequence is one cuBLAS `o_proj` GEMV followed by one Flux fused
residual-RMSNorm kernel. Across 30 layers this is 60 launches and 0.1821 ms at
capacity 2048 or 0.1813 ms at 4096 (about 0.0061 ms per layer).

The `[1, 1, 576]` projection result is 2,304 bytes. It is written by `o_proj`,
read once by residual-RMSNorm, and dead immediately afterward. The fused kernel
also reads the 2,304-byte carried residual and writes the 2,304-byte normalized
MLP input plus the 2,304-byte residual sum. The normalized and residual outputs
use graph-owned stable buffers; their lifetimes end at the packed gate/up GEMM
and the MLP residual boundary, respectively.

### MLP boundary

Per layer, the sequence is one cuBLAS `down_proj` GEMV followed by one PyTorch
pointwise residual-add kernel. Across 30 layers this is 60 launches and 0.2058
ms at capacity 2048 or 0.2056 ms at 4096 (about 0.0069 ms per layer).

The `[1, 1, 576]` down-projection result is 2,304 bytes. It is written once,
read once by the add, and dead immediately afterward. The add also reads the
2,304-byte carried residual and writes the 2,304-byte layer output. That sum
lives through the next layer's input RMSNorm and attention residual connection.
After layer 29 it instead feeds the final model RMSNorm. The stable packed
SwiGLU buffer ends its lifetime at `down_proj`; the projection and sum are graph
private-pool tensors rather than explicit scratch tensors.

## Candidate architecture and isolated results

The attention candidate accumulated `o_proj` into the carried residual in place,
then invoked the existing RMSNorm out-path. The MLP candidate accumulated
`down_proj` into its residual in place. In-place aliasing was limited to the
exact full-tensor input/output alias accepted by `torch.addmm`; no projection
input was aliased. MLP-only integration required a 2,304-byte layer-parity
ping-pong residual buffer so the following layer's residual-RMSNorm output did
not alias its input. The experiment also handled layer 29 by carrying its sum to
the existing final model RMSNorm.

| Boundary | Retained median ms | Candidate median ms | Speedup | Retained launches | Candidate launches | Retained device ms | Candidate device ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Attention | 0.00810 | 0.00988 | 0.820x | 2 | 3 | 0.00437 | 0.00534 |
| MLP | 0.00843 | 0.01033 | 0.816x | 2 | 2 | 0.00375 | 0.00770 |

These exact-shape boundaries do not depend on cache capacity, so the same
measurements apply at 2048 and 4096. A diagnostic complete-decoder-layer graph
(StaticCache/native GQA, without shared output scratch) measured 0.05296 ms
baseline versus 0.04896 ms attention candidate and 0.05296 ms MLP candidate at
2048; at 4096 it measured 0.05504 ms baseline versus 0.05501 ms attention and
0.05709 ms MLP. These isolated layer results were not retention evidence because
they did not reproduce as an integrated 30-layer improvement.

## Integrated CUDA Graph results

The short capacities retain the existing fallback architecture and were not
given a residual-accumulation path.

| Capacity | Baseline ms/token | Attention ms/token | Speedup | MLP ms/token | Speedup | Both ms/token | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 2.5137 | fallback | 1.000x | fallback | 1.000x | fallback | 1.000x |
| 512 | 2.6668 | fallback | 1.000x | fallback | 1.000x | fallback | 1.000x |
| 1024 | 2.9416 | fallback | 1.000x | fallback | 1.000x | fallback | 1.000x |
| 2048 | 1.7553 | 1.7793 | 0.987x | 1.8081 | 0.971x | 1.8285 | 0.960x |
| 4096 | 1.8317 | 1.8586 | 0.986x | 1.8892 | 0.970x | 1.9033 | 0.962x |

| Variant | Total launches | cuBLAS | Flux custom | Remaining framework/state/copy | Profiled device ms at 4096 |
|---|---:|---:|---:|---:|---:|
| Baseline | 345 | 122 | 181 | 42 | 1.8689 |
| Attention | 375 | 152 | 181 | 42 | 1.8854 |
| MLP | 345 | 152 | 181 | 12 | 1.9213 |
| Both | 375 | 182 | 181 | 12 | 1.9785 |

The attention candidate replaces 30 ordinary projection GEMVs with 60 cuBLAS
kernels and replaces 30 residual-RMSNorm kernels with 30 RMSNorm kernels. The
MLP candidate removes 30 pointwise adds but replaces 30 ordinary down-projection
GEMVs with 60 cuBLAS kernels.

Eager decode was effectively neutral and inconsistent, so eager execution was
not changed. Attention measured 10.9346 -> 10.8793 ms at 2048 and 10.8701 ->
10.8968 ms at 4096. MLP measured 10.9224 -> 10.9078 ms at 2048 and 10.8054 ->
10.8276 ms at 4096. These sub-percent changes did not reproduce in the target
CUDA Graph path.

## Correctness

No tolerance was loosened. The differences are expected from moving the FP32
residual addition into the GEMM reduction order.

| Result | 2048 max | 2048 mean | 4096 max | 4096 mean |
|---|---:|---:|---:|---:|
| Attention accumulated projection | 9.54e-7 | 1.55e-7 | 7.15e-7 | 1.67e-7 |
| Attention normalized output | 1.19e-7 | 8.76e-9 | 7.82e-8 | 1.05e-8 |
| MLP accumulated projection/layer sum | 5.01e-6 | 1.06e-6 | 7.63e-6 | 1.10e-6 |
| Attention-candidate layer outputs | 1.53e-4 | 2.64e-6 | 8.01e-5 | 1.86e-6 |
| MLP-candidate layer outputs | 1.83e-4 | 2.57e-6 | 1.22e-4 | 2.13e-6 |
| Attention-candidate eager logits | 1.53e-5 | 2.89e-6 | 1.72e-5 | 3.97e-6 |
| MLP-candidate eager logits | 1.72e-5 | 3.12e-6 | 1.53e-5 | 2.80e-6 |

Eight repeated capacity-4096 graph replays, with stable scratch poisoned before
each replay, produced maximum/mean logit differences of 3.43e-5/4.66e-6 for
attention, 5.15e-5/7.45e-6 for MLP, and 7.44e-5/8.51e-6 together. Maximum full
KV-cache differences were 7.30e-5, 3.62e-5, and 4.01e-5, respectively. Greedy
token IDs matched for every replay, stable addresses were unchanged, and the
eight-replay run reached the exact full-cache boundary. Cache position/length
and exhaustion logic were unchanged by the candidate. The isolated accumulated projection also matched on a
non-default CUDA stream (attention maximum/mean 9.54e-7/1.55e-7). Multi-token
greedy generation therefore remained unchanged over the tested continuation.

## Memory and traffic

Each retained projection temporary costs 2,304 bytes. Per token, either
candidate would eliminate 69,120 bytes of distinct projection-result writes and
69,120 bytes of subsequent reads across 30 layers; both would eliminate 138,240
bytes in each direction. The accumulated sum itself still has to read and write
the residual through cuBLAS. The observed split-K reduction adds internal
traffic not represented by the logical tensor accounting and outweighed the
removed boundary traffic.

Baseline stable scratch was 34,368 bytes at capacity 2048 and 53,376 bytes at
4096. Attention and the combined candidate needed no additional explicit
scratch. MLP-only needed one 2,304-byte ping-pong residual, increasing 4096
scratch to 55,680 bytes. Independent-process graph private-pool size remained
33,751,040 bytes (32.188 MiB) for every variant. Capacity-4096 eager incremental
peak was 5,044,736 bytes for baseline, attention, and MLP; the combined run was
5,042,176 bytes. Thus neither eliminated temporary reduced the allocator high
water mark.

## Conclusion

Both experimental model variants and the ping-pong buffer were removed. Packed
QKV, packed gate/up, native GQA, fused packed-QKV/RoPE/cache, packed SwiGLU,
StaticCache, state-dict compatibility, eager defaults, and the final RMSNorm path
remain unchanged.

The next highest-device-time non-cuBLAS work is native GQA attention: 0.6410 ms
at capacity 2048 and 0.7338 ms at 4096. It is not yet small compared with the
projection/LM-head cuBLAS total (about 0.88 ms), so projection GEMMs are not the
only remaining bottleneck. A future milestone should first characterize the
long-context native GQA chunk/reduce path; a separate cuBLAS/cuBLASLt one-token
projection algorithm-characterization milestone is also warranted, but should
not be mixed with residual-boundary changes.
