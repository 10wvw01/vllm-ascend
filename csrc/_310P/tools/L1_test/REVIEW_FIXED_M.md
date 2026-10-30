# 固定M统一适配器与批次复核（2026-10-30）

基底：dev/310p-memfabric-mm-ar @ 9b2e00875a35e8dc09af25a7d5a6128579414297。
本轮实现Stock/V1/V2/V3薄适配器，准备Stock+所选V2+V3八种迭代的完整固定M矩阵。
V1不进入本轮测量；无CANN/NPU新执行，无算子Kernel/Host/协议改动。

## 第一轮：调用与测量合同

- Stock严格一次NZ MM+HCCL SUM；V1一次Tensor op，q参数明确传递。
- V2绑定基础/私有/所选API；仅B1绑定AR-only符号，诊断OFF不阻止基础路径。
- V2 raw输出在建池前分配；V3 raw输出和探针存储同样提前，pool-return不额外分配闲置输出。
- 原始V3使用1MiB packet，其他本轮V3使用4MiB；serial/pipeline、classic/merged、A3席位均显式登记。
- V3-0/2/4从冻结模块加载，独立包隔离相对import，不可落到全局当前V3Session。
- 适配器只有run/after_sample/close，不增设版本私有计时和统计，不隐藏必需同步、包装、seed或拷贝。
- 原ready/chain、正确性和逐样本双Rank max后均值公式保持，设备阶段耗时缺失仍UNKNOWN。

## 第二轮：矩阵、隔离与失败

- 完整43组/170会话；active30组/118会话；可选V2历史64组/254会话；host/profiler完整三段510会话。
- 每配置独立ABBA，所选V2与历史私有V2单独runtime_key；没有跨配置合并Stock或混淆参数。
- W20/N100/R1，两形状ready+chain，默认每臂每形状每场景200配对样本。
- 所有工件/历史身份/准入在设备前统一预检，任一缺失整批BLOCKED，不静默删实验。
- 框架身份覆盖每个插件模块；bundle固定每份plan/manifest；启动后不允许改API/参数/库。
- SDK v11/v28按进程隔离，Stock不加载融合依赖；沿用物理卡守护和SDK/Device实际装载检查。
- 首个错误STOP，不重试通信波、不进入下一组；汇总失败也写明确STOP，保留原始状态与未运行清单。
- 每组输出源码HEAD、配置、对应Stock、均值/分位数/std、漂移和原判定，诊断不冒充正式收益。

## CPU验证

```text
python -m pytest -q csrc/_310P/memfabric_mm_ar_v2/test \
  csrc/_310P/memfabric_mm_ar_v3/test tools/310p_mm_ar csrc/_310P/tools/L1_test
475 passed, 36 subtests passed
```

本轮新增23项薄适配/历史模块隔离/完整矩阵/冻结/STOP/汇总回归。
CMake测试通过当前Python环境寻找cmake，去掉对临时虚拟环境绝对路径的依赖。
35份版本文档重新生成，V3历史分支现在明确说明内置冻结session适配；CLI可从任意目录生成计划。
未运行格式检查。所有PASS仅指CPU假后端、结构与元数据合同，不代表SDK/CANN/NPU准入。
