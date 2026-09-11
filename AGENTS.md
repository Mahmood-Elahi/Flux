# Flux Agent Instructions

## Project Direction

* Preserve Flux as one continuous long-term project with one final integrated end state. Do not split milestones into unrelated mini-projects or repeatedly redefine the project.
* Follow the existing roadmap and repository architecture unless a concrete technical issue requires a change.
* Do not silently expand or change project scope.
* Do not implement functionality merely to make the repository appear more complete.
* Prefer incremental work that directly advances the integrated SmolLM2 inference system.

## Workflow

* Inspect the relevant existing repository files before making changes.
* Follow established Flux conventions and reuse existing infrastructure rather than creating parallel systems.
* Prefer small, reviewable changes.
* Do not modify unrelated code.
* Do not commit or push unless explicitly instructed.
* Leave the working tree changes available for review at the end of each task.
* Report all files created and modified and show `git status --short` at the end of implementation tasks.

## Correctness

* Correctness comes before optimization.
* Treat PyTorch, Hugging Face, and existing Flux reference implementations as correctness oracles where applicable.
* Every custom operator must have explicit correctness tests.
* Preserve mathematical and model semantics exactly when replacing existing operations.
* Do not weaken tests or tolerances merely to make an implementation pass.
* Use deterministic inputs/seeds where practical for validation.
* After focused validation, run the relevant regression tests and the full Flux test suite.
* Report exact test commands and results.

## Python / PyTorch

* Keep explicit Python reference implementations separate from optimized native implementations when they serve as correctness oracles.
* Follow the existing Flux custom-operator architecture rather than creating independent extensions.
* Preserve FakeTensor/meta support and `torch.library.opcheck` coverage for custom PyTorch operators where applicable.
* Preserve device, dtype, shape, and layout semantics unless an operator's documented contract says otherwise.
* Native optimized operators are currently inference-oriented; do not add training/backward support unless explicitly requested.

## C++ / CUDA

* Keep C++ and CUDA implementations small, readable, and explicit.
* CUDA code must use the caller's correct current CUDA stream.
* PyTorch CUDA integrations must use PyTorch's current CUDA stream and appropriate device guarding.
* Production CUDA launchers must remain asynchronous unless synchronization is explicitly part of the API contract.
* Do not silently fall back to the default CUDA stream.
* Do not add unnecessary device synchronization.
* Preserve correct launch-error handling.
* Support arbitrary valid dimensions unless the operator explicitly documents a narrower contract.
* Do not hardcode SmolLM2 dimensions where a general implementation is practical.
* Avoid unnecessary temporary global-memory allocations.

## Optimization

* Never optimize before establishing a correct baseline.
* Do not report performance improvements without reproducible benchmarks.
* Benchmark the actual workload and API path that matters.
* Use CUDA events or another synchronization-correct GPU timing method; do not use naive unsynchronized wall-clock timing for CUDA work.
* Keep allocations, random generation, model loading, tokenizer work, and unrelated setup outside timed regions.
* Use warmups and repeated measurements.
* Prefer median latency as the primary statistic unless another metric is justified.
* Compare against an appropriate PyTorch/Hugging Face baseline.
* Treat optimization as empirical: implement, benchmark, keep only measured improvements, and remove rejected experimental code.
* Do not retain theoretically appealing optimizations that regress important workloads.
* Distinguish kernel/device performance from Python, dispatcher, launch, and framework overhead when relevant.
* Avoid premature architecture-specific complexity, fusion, vectorization, approximate math, or third-party CUDA libraries unless measurements justify them.

## SmolLM2 Integration

* Preserve the existing reference SmolLM2 path as a correctness oracle.
* Flux integration must be explicit/opt-in and must not globally monkey-patch third-party libraries.
* Do not modify files inside PyTorch, Transformers, or other third-party packages.
* Reuse original model weights and preserve state-dict compatibility where practical.
* Preserve GQA, RoPE, attention scaling, masking, residual ordering, RMSNorm epsilon/weights, and KV-cache semantics exactly.
* Only substitute a Flux operator where its semantics exactly match the model computation.
* Do not force a custom operator into a model location where its contract does not match.
* Do not assume standalone kernel speedups imply model-level speedups; validate end-to-end behavior separately.

## Environment

* Use the repository's verified working Python/PyTorch/CUDA environment.
* Do not make broad dependency or toolchain upgrades unless explicitly required.
* Current target environment is Windows with Python 3.11, PyTorch CUDA, CUDA Toolkit, MSVC, and an NVIDIA RTX 5070 Ti / `sm_120`; inspect the repository/current environment rather than blindly relying on stale values.
* Ensure native extensions are actually rebuilt after C++/CUDA changes before validating or benchmarking them.
* Do not benchmark stale binaries.

## Reporting

At the end of implementation or optimization tasks, concisely report:

* what changed
* important design decisions
* exact validation results
* benchmark results when relevant
* any rejected experiments when relevant
* remaining limitations or assumptions
* `git status --short`

Do not repeat large amounts of repository background already captured here unless it is necessary to explain the current task.
