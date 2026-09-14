# Flux Roadmap

Flux is one continuous project whose completed outcome is an integrated,
CUDA-accelerated SmolLM2-135M inference system. The milestones below are
sequential stages of that project, not separate projects, versions, or
independent finish lines.

1. **Completed:** Establish reproducible SmolLM2-135M PyTorch reference inference. Validated with seven offline tests and real FP32 CUDA forward inference and greedy generation on an RTX 5070 Ti (Python 3.11.9, PyTorch 2.14.0+cu132, Transformers 5.16.1).
2. **Completed:** Implement an FP32 PyTorch reference RMSNorm and validate it against PyTorch and the Transformers Llama form used by SmolLM2.
3. **Completed:** Implement native C++ RMSNorm and validate it with standalone native tests and a cross-language check against the PyTorch reference.
4. **Completed:** Implement naive FP32 CUDA RMSNorm with one block per row, FP32 shared-memory reduction, and a stream-aware standalone launcher; validate it natively and against the PyTorch reference.
5. **Completed:** Integrate FP32 RMSNorm as an inference-only PyTorch custom operator with CPU and CUDA dispatcher implementations and correct current-stream CUDA launches.
6. **Completed:** Add CPU and CUDA custom-operator correctness tests against the PyTorch reference, including FakeTensor/opcheck and non-default CUDA stream coverage.
7. **Completed:** Establish a reproducible pre-optimization CUDA RMSNorm benchmark against PyTorch's native RMSNorm on SmolLM2-relevant FP32 shapes.
8. **Completed:** Optimize FP32 CUDA RMSNorm with a measured warp-level reduction, and evaluate aligned `float4` memory access; retain scalar access after the vector path regressed the largest benchmark workload.
9. **Completed:** Implement dual-output fused residual + RMSNorm.
10. **Completed:** Implement transformer attention softmax and perform two evidence-driven CUDA optimization rounds.
11. **Completed:** Ensure all CUDA operators use the correct current CUDA stream.
12. **Completed:** Integrate RMSNorm, post-attention fused residual + RMSNorm, and eager attention softmax into an optional FP32 SmolLM2 execution path, with layer, logits, generation, and KV-cache validation.
13. **Completed:** Implement and integrate FP32 RoPE with CPU/CUDA dispatch, current-stream execution, GQA-aware Q/K rotation, positional-offset and cache-decode validation, and measured model-level gains.
14. **Completed:** Profile the SmolLM2 MLP, retain a separately opt-in packed gate/up projection after correctness and end-to-end benchmarking, and preserve standard checkpoint compatibility.
15. **Completed:** Implement and retain separately opt-in FP32 packed SwiGLU with CPU/CUDA dispatch, current-stream execution, FakeTensor/opcheck coverage, model and CUDA-Graph integration, and measured MLP/prefill/decode/memory results.
16. **Completed:** Profile and retain separately opt-in packed QKV as a structural PyTorch/cuBLAS optimization, preserving standard checkpoints, allocation-free strided RoPE/cache views, CUDA-Graph decode, and unchanged parameter/cache storage.
17. **Completed:** Implement and retain separately opt-in FP32 one-token GQA decode attention over unexpanded DynamicCache/StaticCache K/V storage, with stable online softmax, current-stream execution, FakeTensor/opcheck and CUDA-Graph coverage, measured short-cache graph fallback, and isolated/layer/eager/long-context graph gains.
18. **Completed:** Profile and retain a separately opt-in FP32 one-token packed-QKV post-projection CUDA path that applies Q/K RoPE, writes K/V directly into unexpanded StaticCache storage, emits compact Q for native GQA attention, advances graph-resident cache length, and measurably reduces long-context eager and CUDA-Graph decode latency without increasing peak or graph-pool memory.
19. **Completed:** Attribute remaining fully optimized fixed-shape decode launches, prove PyTorch deterministic uninitialized-memory filling as the source of 214 captured FP32 fill kernels, and retain graph-owned stable outputs for the five Flux producers responsible for 211 of them, without disabling deterministic safety globally or changing eager/general fallbacks.
20. **Completed:** Systematically characterize all FP32 one-token SmolLM2 projection and LM-head shapes, add a narrow current-stream and CUDA-Graph-safe cuBLASLt tuning interface, and retain only the measured zero-workspace packed-QKV and attention-output configurations after isolated, category, layer, full-decode, correctness, launch, and independent-memory validation.
21. **Completed:** Design and evaluate exact-shape FP32 one-token packed gate/up GEMV kernels; reject standalone/shared/vector variants in favor of a separately opt-in fused gate/up GEMV + SwiGLU CUDA-Graph path that removes 30 launches and the packed projection intermediate with a reproducible integrated decode gain.
22. **Completed:** Re-profile the fully retained one-token decode system, identify native GQA as the largest category and the LM head as the largest individual library kernel, and reject a bounded custom FP32 LM-head GEMV after its marginal isolated win regressed integrated graph decode at the primary and longer capacities.
23. **Completed:** Re-profile the retained native long-context GQA path, retain shared per-chunk rescaling in the final reduction after reproducible alternating full-graph gains at 4096 and 8192, and reject a coalesced stage-1 variant that won in isolation but regressed integrated decode.
24. **Completed:** Freeze the final retained category set, validate checkpoint,
    logits, prefill, cached and multi-token decode, KV-cache, masking, stream,
    CUDA-Graph replay, stable-address, allocation, greedy-generation, and
    long-context behavior, and benchmark the complete reference, Flux eager,
    and Flux graph systems from 128 through 8192 tokens.
25. **Completed:** Audit and consolidate the Python benchmark/support surface
    around the canonical final-system benchmark, retained decode profiler, and
    focused operator benchmarks; preserve historical evidence and define the
    native fixed-shape decode architecture.
26. **Completed:** Implement the native fixed-shape decode control plane and a
    one-layer executor using retained kernels and exact cuBLAS/cuBLASLt paths,
    with lifetime-managed stable buffers, device position/cache length, K/V,
    workspace, capture stream, CUDA Graph executable, asynchronous current-stream
    replay, reset, diagnostics, and validation from 128 through 8192 tokens.
27. **Completed:** Scale the validated executor across all 30 layers with
    compact per-layer descriptors and shared lifetime-planned workspace; add
    native token embedding, final RMSNorm, retained cuBLAS LM head, one-step
    device state advancement, and complete token-to-logits CUDA Graph replay;
    validate logits, every layer's compact K/V, greedy tokens, streams,
    allocation behavior, launch structure, and performance against Python Flux
    and the pinned Hugging Face oracle from length 128 through 8192.
28. **Completed:** Implement native full-model FP32 prompt prefill from CUDA
    token IDs through all 30 layers and final logits, with compact direct K/V
    writes into the attached native decode runtime, grouped GQA without
    `repeat_kv`, in-place causal softmax, persistent lifetime-planned workspace,
    current-stream/event ordering, allocation-free reuse, and validation and
    performance coverage from length 128 through 8192.
29. **Completed:** Establish a coherent native CTest and CUDA-event benchmark
    hierarchy around production kernels and the native prefill/decode runtime;
    add direct correctness, stream, graph, state, storage, allocation, and
    long-context validation; retire superseded Python executable wrappers,
    operator-only benchmarks, CLIs, and the historical one-layer benchmark
    while retaining all PyTorch/Hugging Face integration coverage.
30. **Completed:** Replace native prefill's full quadratic score materialization
    with tiled/streaming FP32 grouped-GQA attention using online softmax,
    compact three-head K/V, and no `repeat_kv`; retain a measured 36 MiB-bounded
    short-context fallback, remove the 2.25 GiB length-8192 score matrix, and
    validate memory, correctness, and performance through length 8192 without
    changing decode semantics.
31. **Next:** Reassess the remaining integrated architecture and source balance
    from the retained streaming-prefill profile without adding unnecessary
    attention complexity or replacing performant library GEMMs for accounting.

Milestone 30 source totals are Python 551,695 B, CUDA 306,421 B, C++ 132,894 B,
and headers 16,553 B. CUDA is 30.412093% of these sources and remains 394,721 B
short of equal CUDA/non-CUDA bytes under the required exact metric. This
milestone makes 31,950 bytes of net progress (`34,404` CUDA bytes added minus
`2,454` non-CUDA bytes added) without padding, duplication, or Linguist
overrides.

The operator-optimization roadmap through milestone 24 is complete. The final
native-runtime phase continues the same integrated codebase; rejected experiments
remain removed, the ordinary pinned Hugging Face/PyTorch path remains the
correctness oracle, and the retained operator architecture is recorded in
`docs/FINAL_SYSTEM_MILESTONE.md`.
