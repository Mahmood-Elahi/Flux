# Flux Roadmap

Flux is one continuous project progressing toward a single final outcome: an integrated, CUDA-accelerated SmolLM2-135M inference system. The milestones below are sequential stages of that project, not separate projects, versions, or independent finish lines.

1. Establish reproducible SmolLM2-135M PyTorch reference inference.
2. Implement a Python reference RMSNorm.
3. Implement native C++ RMSNorm.
4. Implement naive FP32 CUDA RMSNorm.
5. Integrate RMSNorm as a PyTorch custom operator.
6. Add correctness tests against the PyTorch reference.
7. Benchmark and optimize RMSNorm.
8. Implement dual-output fused residual + RMSNorm.
9. Implement transformer attention softmax.
10. Ensure all CUDA operators use the correct current CUDA stream.
11. Integrate the custom operators into the SmolLM2 inference path.
12. Profile and benchmark the integrated system.
13. Optimize measured bottlenecks.
14. Reach the final integrated Flux inference system.

Each implementation and optimization milestone remains part of the same evolving codebase. Correctness against the reference path precedes performance work, and optimization decisions are driven by reproducible measurements.
