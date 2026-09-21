# Developer Guide

This section is for developers who want to contribute to vLLM Ascend or understand its internal architecture.

## 310P3 Qwen3.6 FP16 o_proj + MemFabric fusion

This feature is intentionally documented as a small AI-native project surface:

- **[Requirements / Acceptance Contract](310p_memfabric_o_proj_requirements.md)** — fixed scope, hard invariants, non-goals and Definition of Done.
- **[Current Architecture](310p_memfabric_o_proj.md)** — source-accurate model dispatch, runtime, memory layout, device kernels and protocol.
- **[AI-native Development State](310p_memfabric_o_proj_development_plan.md)** — current source status, latest valid evidence, ordered Task IDs, blockers and iteration protocol. Start here for the next development iteration.
- **[Build / Run / Hardware Validation](../user_guide/feature_guide/310p_memfabric_o_proj_usage.md)** — reproducible build, single-layer benchmark, profiler, eager and graph validation procedures.

The branch name still contains `w8a8`, but the current implementation is the target checkpoint's unquantized **FP16 runtime** o_proj path. Do not infer implementation semantics from the legacy branch name.

## Contribution

- **[Contribution Guide](contribution/index.md)** — How to contribute to vLLM Ascend
- **[Testing](contribution/testing.md)** — Write and run unit, E2E, and nightly tests
- **[Doc Writing](contribution/doc_writing.md)** — Documentation contribution guide
- **[Multi-Node Test](contribution/multi_node_test.md)** — Multi-node testing guide
- **[Nightly CI Test](contribution/nightly_ci_test.md)** — Nightly CI testing
- **[E2E CI Test](contribution/e2e_ci_test.md)** — E2E CI testing

## Design Documents

Explore the design documents covering patch architecture, CPU binding, model runner internals, disaggregated prefill, EPLB, ACL Graph, KV Cache Pool, custom operators, context parallel, quantization, and NPUGraph.

## Evaluation

- **[Using EvalScope](evaluation/using_evalscope.md)** — Model evaluation with EvalScope
- **[Using lm_eval](evaluation/using_lm_eval.md)** — Model evaluation with lm_eval
- **[Using AISBench](evaluation/using_ais_bench.md)** — Model evaluation with AISBench
- **[Using OpenCompass](evaluation/using_opencompass.md)** — Model evaluation with OpenCompass

## Performance and Debug

- **[Performance Benchmark](performance_and_debug/performance_benchmark.md)** — Benchmarking guide
- **[Optimization and Tuning](performance_and_debug/optimization_and_tuning.md)** — Performance optimization
- **[Service Profiling Guide](performance_and_debug/service_profiling_guide.md)** — Service profiling
- **[msprobe Guide](performance_and_debug/msprobe_guide.md)** — Debugging with msprobe
