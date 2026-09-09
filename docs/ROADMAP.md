# Flux Roadmap

Flux is one continuous project progressing toward a single final outcome: an integrated, CUDA-accelerated SmolLM2-135M inference system. The milestones below are sequential stages of that project, not separate projects, versions, or independent finish lines.

1. **Completed:** Establish reproducible SmolLM2-135M PyTorch reference inference. Validated with seven offline tests and real FP32 CUDA forward inference and greedy generation on an RTX 5070 Ti (Python 3.11.9, PyTorch 2.14.0+cu132, Transformers 5.16.1).
2. **Completed:** Implement an FP32 PyTorch reference RMSNorm and validate it against PyTorch and the Transformers Llama form used by SmolLM2.
3. **Completed:** Implement native C++ RMSNorm and validate it with standalone native tests and a cross-language check against the PyTorch reference.
4. **Completed:** Implement naive FP32 CUDA RMSNorm with one block per row, FP32 shared-memory reduction, and a stream-aware standalone launcher; validate it natively and against the PyTorch reference.
5. **Completed:** Integrate FP32 RMSNorm as an inference-only PyTorch custom operator with CPU and CUDA dispatcher implementations and correct current-stream CUDA launches.
6. **Completed:** Add CPU and CUDA custom-operator correctness tests against the PyTorch reference, including FakeTensor/opcheck and non-default CUDA stream coverage.
7. **Completed:** Establish a reproducible pre-optimization CUDA RMSNorm benchmark against PyTorch's native RMSNorm on SmolLM2-relevant FP32 shapes.
8. **Completed:** Optimize FP32 CUDA RMSNorm with a measured warp-level reduction, and evaluate aligned `float4` memory access; retain scalar access after the vector path regressed the largest benchmark workload.
9. Implement dual-output fused residual + RMSNorm.
10. Implement transformer attention softmax.
11. Ensure all CUDA operators use the correct current CUDA stream.
12. Integrate the custom operators into the SmolLM2 inference path.
13. Profile and benchmark the integrated system.
14. Optimize measured bottlenecks.
15. Reach the final integrated Flux inference system.

Each implementation and optimization milestone remains part of the same evolving codebase. Correctness against the reference path precedes performance work, and optimization decisions are driven by reproducible measurements.
