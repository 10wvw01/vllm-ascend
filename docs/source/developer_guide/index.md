# Developer Guide

This section is for developers who want to contribute to vLLM Ascend or understand its internal architecture.

## 310P3 Qwen3.6 W8A8 o_proj + MemFabric fusion

- **[Requirements / Acceptance Baseline](310p_memfabric_o_proj_requirements.md)** — Project north star: fixed scope, hard constraints, non-goals, success criteria and final acceptance contract
- **[Design](310p_memfabric_o_proj.md)** — Architecture, integration points, memory ownership, pipeline and synchronization invariants
- **[Development Plan](310p_memfabric_o_proj_development_plan.md)** — Overall roadmap, current completion status, remaining tasks and acceptance criteria
- **[Build / Deployment / Usage](../user_guide/feature_guide/310p_memfabric_o_proj_usage.md)** — 310P server build, custom MemFabric dependency, hardware benchmark and Qwen3.6 launch procedure
- **[OpenCode Handoff Prompt](310p_memfabric_o_proj_opencode_prompt.md)** — Prompt for continuing bring-up and development directly on the 310P server

The requirements document is authoritative for this feature. Design and implementation may change based on verified 310P/custom-MemFabric behavior, but they should continue to satisfy that requirements baseline.

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
