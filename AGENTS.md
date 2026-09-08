# Flux Agent Instructions

- Preserve Flux as one continuous project with one final integrated end state; do not split milestones into unrelated mini-projects.
- Correctness comes before optimization. Treat PyTorch and other reference implementations as correctness oracles.
- Every custom operator must have correctness tests.
- CUDA code must use the correct current CUDA stream.
- Do not introduce unnecessary abstractions or dependencies.
- Do not report performance improvements without reproducible benchmarks.
- Keep Python, C++, and CUDA code clean and readable.
- Prefer small, reviewable changes.
- Do not silently change project scope.
- Do not implement functionality merely to make the repository appear more complete.
