# 310P3 V2 MemFabric MM+AllReduce 工程

> 当前状态：G3 的算子级与模型 Eager 功能正确性已验证，Eager 性能与长稳 FAIL，
> Graph 实际未调用 V2。**请先读 [阶段归档](ARCHIVE_20261019_G3.md)**；
> 禁止把“构建成功”或“Graph 输出与 Stock 一致”误作融合收益证明。

## 目录职责

- `../src/op_kernel/`：Ascend C Kernel（.asc），bisheng dav-2002，计算/SDMA/Add；
- `../src/op_host/`：MemFabric 公共 Host API/对称内存/调度与适配 ABI；
- `../src/torch_adapter/`：注册 `_C_ascend` Torch Op，V2 Serving 生命周期；
- `../src/CMakeLists.txt`：CANN 直调工程源码清单；主工程入口
  `cmake/memfabric_310p_v2.cmake` 在仓库根目录；
- `../test/`：G0/G1/G2/T1/T2、G3 脚本、回归与归档探针。
- `../test/ut/`：V2 import-free 静态源码单测；
- `../test/e2e/`：V2 TP2 路由测试、Stock/V2 三会话评估；
- `../test/g3_e2e_scripts/`：G3 实机验证驱动及未合入的 Graph 实验快照。

## 正式算子与诊断模块的源码隔离

| 子目录 | 正式运行代码 | 调试、诊断及验证代码 | 共享的内部头文件 |
|---|---|---|---|
| `op_kernel/` | `mf310p_v2_kernel.asc`：T0 MM+MemFabric+Add 和 Serving 需要的 IB 自检 | `mf310p_v2_diagnostics.asc`：阶段诊断、T1 NZ 小 M、T2 AR-only、参数探针 | `mf310p_v2_kernel_common.h` |
| `op_host/` | `mf310p_v2_trial_adapter.cpp`：建池、运行、状态、清理 | `mf310p_v2_trial_diagnostics.cpp`：T1/T2 Host 入口、地址及数据探针 | `mf310p_v2_trial_internal.h`；对应独立 C ABI 头文件 |
| `torch_adapter/` | `mf310p_v2_serve.cpp`：真实 Qwen V2 Serving Torch Op | `mf310p_v2_serve_diagnostics.cpp`：主动读取协议状态的 Torch 调试 Op | `mf310p_v2_serve_internal.h` |

正式入口保持原 ABI 与符号，诊断代码不再嵌入正式实现文件。**IB 安全自检保留在正式 Kernel**，因为真实 Serving 初始化必须运行该检查，不能依赖可选诊断模块。

构建开关 `VLLM_ASCEND_310P_V2_BUILD_DIAGNOSTICS` **默认 ON**，以兼容历史 G0/G1/T1/T2/G3 测试。使用干净的构建目录时，可指定
`-DVLLM_ASCEND_310P_V2_BUILD_DIAGNOSTICS=OFF` 验证正式算子最小构建；
此时不会导出 T1/T2、扩展探针和 `memfabric_mm_ar_v2_debug_status`。
OFF 不代表完成生产验收，必须先在 310P3 实机编译和回归。

原始实验报告与问题结论保持归档状态，本次只拆分实现文件和构建清单，不处理先前的 Graph 旁路、Eager 负收益或偶发双卡等待超时。

## 为什么不直接迁成 ACLNN 算子包？

这是**CANN AscendC Kernel 直调工程**，设备侧自编译成
`libmf310p_v2_trial_device.so`，Host 经 MemFabric 公共 ABI 连接
vllm-ascend 的 torch 扩展。目前没有 `op_proto`、`op_host` tiling 注册或
ACLNN `npu_op_package` 调用合同。为“标准化”强行新增 ACLNN 算子定义、
tiling 系统与独立 run 包会改变已验证的调用 ABI 和 SDMA 生命周期，
所以本轮只采用标准 CANN 的目录分层，
保留现有编译流程、Kernel 符号及实际算法。

## 文档索引

- [阶段归档](ARCHIVE_20261019_G3.md)
- [G3 原始实机报告](V2_G3_QWEN_E2E_REPORT.md)
- [G3 目标与实机验证规范](V2_G3_QWEN_E2E_TRIAL.md)
- [T2 原生 MM+MemFabric 对照](T2_AR_ONLY_REPORT.md)
- [T1 NZ 小 M 验证](T1_NZ_DIRECT_REPORT.md)
- [总体方案](DESIGN.md) / [阶段计划](PLAN.md)
- [G3 实机脚本说明](G3_E2E_SCRIPTS.md)

## 从仓库根目录运行

```bash
# V2 源码静态回归，不依赖 NPU
pytest -q csrc/_310P/memfabric_mm_ar_v2/test/ut

# 在 310P3 服务器，用两卡按 G3 文档设置 CANN/MemFabric 环境后
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  csrc/_310P/memfabric_mm_ar_v2/test/e2e/tp2_v2_mixed_route_correctness.py
```

共享的 **V1/Stock Serving 脚手架**仍由仓库
`tests/e2e/_310p/{test_memfabric_mm_ar_functional.py,memfabric_mm_ar_serve_lib.py,
bench_memfabric_mm_ar_serve.sh}` 拥有，因 V1/Stock 也依赖，不能挪走而破坏 V1。
V2 专用 UT/TP2/G3 驱动已集中于本工程 `test/`。
后续修改 V2 的真实模型功能时仍复用上述公共脚手架，不复制两套实现。
