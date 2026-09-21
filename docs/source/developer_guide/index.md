# Developer Guide

This section is for developers who want to contribute to vLLM Ascend or understand its internal architecture.

## 310P3 Qwen3.6 o_proj + MemFabric fusion

- **[Requirements / Acceptance Baseline](310p_memfabric_o_proj_requirements.md)** — Current scope, invariants and final acceptance contract
- **[Design](310p_memfabric_o_proj.md)** — Current public-API ABI v6 architecture and synchronization model
- **[Hardware Validation Plan](310p_memfabric_o_proj_development_plan.md)** — Current 310P3 build/correctness/Graph/performance gates
- **[Build / Deployment / Usage](../user_guide/feature_guide/310p_memfabric_o_proj_usage.md)** — Current installation, build and hardware-test commands

The requirements document is authoritative for this feature. The other documents describe only the current implementation and current validation workflow; historical implementation details are intentionally excluded.

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
- **[Using AISBench](evaluation/using_ais_bench.md)** — Benchmarking with AISBench
- **[Using OpenCompass](evaluation/using_opencompass.md)** — Model evaluation with OpenCompass

## Performance and Debug

- **[Performance Benchmark](performance_and_debug/performance_benchmark.md)** — Benchmarking guide
- **[Optimization and Tuning](performance_and_debug/optimization_and_tuning.md)** — Performance optimization
- **[Service Profiling Guide](performance_and_debug/service_profiling_guide.md)** — Service profiling
- **[msprobe Guide](performance_and_debug/msprobe_guide.md)** — Debugging with msprobe
