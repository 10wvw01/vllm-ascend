# 310P3 V2 T1：原始 NZ 直接 ND，小 M 实机验证

> **2026-10-17/18 首轮实机验证已完成：单卡 MM 隔离 1028/1028 bit-exact；双卡 160 组合矩阵、随机 M/动态指针/4 权重种子/千次复用全部 PASS；修复 `basicM=256` UB 工作区越界一处。详见 [T1_NZ_DIRECT_REPORT.md](T1_NZ_DIRECT_REPORT.md)。**
> 本试验只验证 V2 小 M 新路线，不接入正式 vLLM 模型、不修改 V1。T0 大 M 的实机正确性结论不等于此路径通过。

## 技术主线

- 复用 Qwen o_proj/out_proj 现有的 **原始 8MiB NZ 权重**，不拷贝或重排为每核 B slice。
- 采用官方 [MatmulBlock](https://gitee.com/ascend/samples/tree/master/operator/ascendc/4_best_practices/23_matmul_all_reduce_custom/MatmulAllReduceCustom) 的 NZ 列偏移规则：转置 B 的 `nOffset × BLOCK_CUBE(16)` 个 FP16 元素。
- 通过 Matmul `SetOrgShape(M,2048,2048,2048,2048)` + `SetSingleShape(M,2048/mmCores,2048)` 将各核的 N 子块写回**最终 ND C0/C1 的对应列片**。
- MM 与 Add 分核数量独立：`mmCores/addCores∈{1,2,4,8}`；Add 核按完整行分工，无跨核 cache line 写共享。
- 单 Kernel 执行 MM→官方 IB 扇入→Core0 MemFabric 公共 signal/wait→Add→quiet+双端 DONE。不新增输入 staging、不做 NZ weight slices，也不额外拼接输出。
- 单数据邮件为 `M×2048×2` 字节（`M=1..256`）。

**待实机确认的第一风险：** Matmul 静态 Tiling、动态 `SetSingleShape`、B NZ 子视图与非紧凑 ND C 的 2048 行步长，在 `dav-2002` 是否完全兼容。若有问题，应优先定位设备侧 Matmul API 约束，而非悄悄退回 V1 blocked 布局。

## 验证步骤（双 310P3）

先按既有方式设置 CANN 9.1、bisheng 和 MemFabric_Hybrid 已安装公共包，再重编仓库：

```bash
export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR=1
export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR_V2=1

# 重编：沿用 2026-10-17 T0 已验证的 CMake/editable 构建命令
# 推荐先确认 20 个静态 Matmul 模板均通过编译和链接。
```

先极小正确性，而非一次全矩阵性能扫：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  csrc/_310P/memfabric_mm_ar_v2/test/bench_nz_direct_small.py \
  --rows 1 16 17 255 256 --mm-cores 1 2 4 8 \
  --add-cores 1 8 --repeat 3
```

确认 bit-exact 后执行完整性能矩阵：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  csrc/_310P/memfabric_mm_ar_v2/test/bench_nz_direct_small.py \
  --rows 1 2 4 8 16 32 64 128 255 256 \
  --mm-cores 1 2 4 8 --add-cores 1 2 4 8 --repeat 30
```

测试脚本在 MemFabric 初始化之前计算 Stock HCCL 参考，避免把 HCCL/SMEM 混用问题当成 MM 正确性问题；每次 T1 调用后核对 FP16 **逐位相等**和协议 status=0；分别输出首次和稳定延迟。继续用已存在的 `benchmarks/scripts/bench_310p_memfabric_mm_ar_layer.py` 单独测 V1（尤其 N-split M<256），不能在同一个正在运行的 V2 MemFabric 池进程内调用 V1。

## 提交验收数据

1. 编译通过/失败记录（设备 architecture、编译参数、模板报错）。
2. 列边界检查：至少 MM=1/2/4/8；重点 M=1、16、17、255、256 的 C(ND) 步长和 NZ 取列。
3. 10×16 完整 M/MM/Add 组合的首调用、median、P95、Stock、V1 对照。**小 M 是否超过 Stock 不做预设。**
4. 动态输出地址、随机 M=1..256、随机权重 4 seeds、跨调用 1000 次和旧 T0 G1/G2 回归。
5. profiler 分离 Matmul、IB ready、SDMA、Add、quiet/DONE；解释固定成本与选核策略。
6. BUG 修复后重测全部 G0、G1、G2 与 T1，报告保存 `T1_NZ_DIRECT_REPORT.md`。

## 已知工程边界

- T1 仍是仅为实验配置的 C API：`mf310p_v2_trial_run_nz_direct_small`；不会自动路由小 M 到 V2。
- `SetOrgShape/SetSingleShape` 配合原始 NZ 子矩阵 + ND 非紧凑行步长已在 dav-2002 实机证实可用；`basicM` 必须 ≤128（`basicM*basicN*2*2 > 可用 UB` 时 release 构建不检查、满块直接 MTE on-chip read OOB）。
- T1 kernel 首启/新 y 首写的 MTE3 丢弃用 launch-stream 同步守护（<1ms，实测全部治愈）；T0 kernel 路径该法无效，只能 `aclrtSynchronizeDevice`（带池随机 0-15s）。两者都不是 Graph-safe 方案。
- 当前 V2 不支持同池多 Stream 并发、正式 Qwen 推理路由或大于 8192 多波。
- 不可修改 MemFabric 私有 ring/TAIL/HEAD，继续使用公开 signal/wait/quiet。
