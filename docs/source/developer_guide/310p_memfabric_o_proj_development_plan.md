# 310P3 TP=2 未量化 o_proj + MemFabric AllReduce 研发计划

> 本文用于项目推进和服务器交接，记录整体研发计划、当前已完成事项、待办事项、实机验证顺序和验收门槛。
>
> 设计文档：[310p_memfabric_o_proj.md](310p_memfabric_o_proj.md)
>
> 使用说明：[310p_memfabric_o_proj_usage.md](../user_guide/feature_guide/310p_memfabric_o_proj_usage.md)

## 1. 最终目标

在 Ascend 310P3 单卡双 die、TP=2 条件下，对 `Eco-Tech/Qwen3.6-35B-A3B-w8a8` 的 full-attention `self_attn.o_proj` 实现：

```text
未量化 FP16 o_proj matmul + TP=2 reduction
```

融合为一条计算/通信 overlap 流水：

```text
MM[t] -> publish[t] -> SDMA[t] -> peer arrival[t] -> local reduce[t]
  |
  +--------------------------------------------------> MM[t+1]
```

目标是让 `MM[t+1]` 不等待 `SDMA[t]`，最终以定制 `wgm-dev-310p` MemFabric 实现双 die 数据搬运。

## 2. 研发阶段总览

| 阶段 | 目标 | 当前状态 |
| --- | --- | --- |
| M0 | 明确模型、硬件、TP、量化和通信契约 | 已完成 |
| M1 | vLLM model-side 精确挂接和 feature gate | 已完成 |
| M2 | 定制 MemFabric build/link + persistent runtime 骨架 | 已完成(实机编译+加载验证, 2026-09-18) |
| M3 | TP=2 bilateral SDMA + repeated-wave correctness | 已完成(M=1~2048 全 PASS; 1000 wave 逐次校验 PASS) |
| M4 | Phase-1 staged 管线 overlap 验证（V4，已废弃） | 已完成（P4 验收：overlap 真实但性能为负） |
| M5 | Phase-2 direct AscendC producer + V5 epoch API | 已完成（2026-09-21 单层 bit-exact 9/9 rows） |
| M6 | P6 性能优化 + ACL graph + 端到端验收 | P6 进行中 |

## 3. 当前已完成事项

### 3.1 模型侧挂接

已完成：

- 仅允许 `qwen3_5_moe_text`；
- 仅允许 `*.layers.N.self_attn.o_proj`；
- 额外要求 `config.layer_types[N] == "full_attention"`；
- TP 必须为 2；
- local K 必须为 2048；
- output N 必须为 2048；
- 310P 未量化路由（owner decision A，原 W8A8 契约废弃）；
- FP16 output；
- 命中后 `RowParallelLinear.reduce_results=False`，避免 generic TP allreduce 再执行一次。

### 3.2 Build/feature gate

已完成：

- feature 默认关闭；
- `wgm-dev-310p` 独立编译安装；
- vLLM-Ascend 直接 include/link 定制 MemFabric；
- 不使用运行时 `dlopen`；
- 不部署第二个 adapter `.so`；
- CMake 只从显式指定的 custom install prefix 找头文件；
- feature 打开但 MemFabric root/libs/device object 缺失时 configure 直接失败；
- `.asc` device object 当前作为外部 build input 传入。

### 3.3 Runtime/内存所有权

已完成代码骨架：

- process-persistent MemFabric context；
- symmetric send arena；
- symmetric recv/final arena；
- V5 mailbox 保留区（48 KiB，memfabric 托管）；
- producer scratch + per-bucket/waiter warmup bitmap；
- send/recv 不 alias；
- failed wave poison runtime，防止残留 polling/SQE/flag 被继续复用。

### 3.4 融合管线（V5，phase-1 staged 管线已删除）

```text
fused AscendC producer kernel（matmul -> 行级 clean -> signal_at，逐 chunk）
    -> AICPU epoch kernel 代投 SDMA（与下一 chunk matmul 重叠）
waiter kernel（quiet + wait x N）
at::add_out(final = send + recv)
```

单条 torch stream，chunk 之间零 host 同步；wave 边界 join（stream 同步 +
control barrier）防跨 wave recv 覆写。

### 3.5 测试基础设施

已完成：

- import-free source regression；
- eligibility/fallback regression；
- no-dlopen/direct-link regression；
- non-aliasing arena regression；
- failure poison regression；
- TP=2 单层 bit-exact/latency benchmark：
  `benchmarks/scripts/bench_310p_memfabric_o_proj_layer.py`；
- mailbox 现场诊断 op：`memfabric_o_proj_debug_snapshot`。

## 4. P0-P4 实机 bring-up 与验收记录（历史，2026-09-17~20）

服务器 bring-up 首先只解决“能正确编译和链接”，不要同时优化性能。

### 4.9 实机 bring-up 状态（2026-09-17，服务器首轮）

环境与工具链（已就绪并实测）：

- Python 栈：`torch 2.10.0+cpu` / `torch-npu 2.10.0.post4` / `vllm 0.27.1+empty`
  （源码装于 `/vllm-workspace/vllm` @ v0.27.1，`VLLM_TARGET_DEVICE=empty`，对齐
  `Dockerfile.310p`）；ray 2.48.0 / modelscope / MPI（openmpi 4.1.2 发行版）。
- `wgm-dev-310p` @ `e07ef883`（本地为 `d177bfaf`，含 orchestrator json 模式自判别
  与失败面包屑诊断两处实机验证改动）：主构建（libmf_smem/libmf_hybm_core/
  libacc_tcp_net 静态库+so）与 AICPU kernel standalone 构建
  （`cmake -S src/hybm/ops -B <dir> -DASCEND_HOME_PATH=...`，产出
  `libmf_sdma_orch.so` + CUST/launch 双 json）均通过。
- example 08 用 `bisheng --npu-arch=dav-2002` 统一编译通过（host+AICore 可执行）；
  实机双 rank 建池/`smem_shm_sdma_submit` 成功。
- `smem_shm_control_barrier` 在 wgm-dev-310p 真实存在（smem_shm.h:123），仓库
  adapter 的用法与实机 ABI 一致。

**当前唯一阻塞（外部条件）**：AICPU 编排 kernel 必须部署到设备
`/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/`（CP1 搜索路径）才能以
AICPUKernel 模式执行；原部署工具 `deployer`/`mini_kfc` 在本容器缺失（全盘无
残留）。deployer 功能已用 `aclrtBinaryLoadFromFile`(CUST json, mode=1) 自研
复现（可把 ELF 迁移落盘到设备 `/home/CustAiCpuUser/lib/`），但设备内跨目录
复制（原 mini_kfc 经 KFC 通道的 shell）无法复现。六条免部署替代通道已系统性
实测排除（详见 memfabric 仓库 `docs/310p/部署档案.md` 第 6 节），其中
CUST 直接 launch 的结构性根因是：CUST aicpusd 为独立进程，查不到 host CP 流
注册的 SQ（mainRet=1300）。310P host 编排路径（HostDataOpAclMemcpy）为
"async 降级 sync"，无 overlap 能力，不可作为替代。

### 4.10 实机 bring-up 状态（2026-09-18，服务器第二轮：P0/P1 全部打通）

**P0 编译（已完成）**：

- `wgm-dev-310p` 合并远程后为 `d7f11c6d`（含 823697d5 部署链工具入库 +
  075856a9 初始化自动部署 + 44e3c4e3 三件套接入主构建），主构建
  `cmake -B build -DXPU_TYPE=NPU -DBUILD_PYTHON=OFF && cmake --build build -j`
  通过；安装前缀组装于 `/opt/memfabric-wgm-dev-310p`（include/ + lib64/ +
  hybm/aicpu_kernel/ 三件套，与运行时安装前缀发现路径一致）。
- `.asc` 实机编译命令（必须 `-fPIC`，产物为可链接 ELF relocatable）：
  `bisheng --npu-arch=dav-2002 -O2 -std=c++17 -w -fPIC -x asc
  csrc/memfabric_o_proj/external/memfabric310p_device.asc -x none -c -o
  build_310p_artifacts/memfabric310p_device.o -I$ASCEND_HOME_PATH/include
  -I$MF_ROOT/include`。链接额外需要 CANN 静态库 `libascendc_runtime.a`
  （AscendC launch stubs，已写入 cmake/memfabric_310p.cmake）。
- 首轮记录的"AICPU kernel 部署阻塞"由远程三笔提交解决：初始化内嵌自动
  部署（CUST 迁移落盘 + KFC 就位 + 完整性校验），example 08 与本管线均
  零手工部署通过。
- feature-on build `pip install -e . --no-build-isolation --no-deps` 通过，
  extension 加载、`memfabric_o_proj_*` 四个 op 注册齐全，RUNPATH 指向
  定制安装前缀。运行时需 `LD_LIBRARY_PATH=$MF_ROOT/lib64:$LD_LIBRARY_PATH`
  （RUNPATH 不传递到 libmf_smem.so 的间接依赖）。

**P1 通信 correctness + repeated-wave（已完成）**：

- M = 1/8/32/64/128/512/2048 双 rank 结果与 HCCL reference 全部 allclose。
- 1000 连续 wave × M∈{1,32,128,512,2048} 逐 wave 校验（`--stress-check`）
  全部通过：无 hang、无 mismatch、无 stale flag，进程干净退出。
- **发现并修复的实机 race（重要）**：reduce consumer 读本地 send arena
  原先没有任何门控（arrival flag 只门控对端数据），对端 SDMA 先到时读
  到未初始化内存（表现为间歇性、rank 不对称的 mismatch，常数模式冒烟
  测试测不出）。修复：reduce 逐 chunk 双门控自旋（邮箱槽 words =
  本地已产出 + arrival flag = 对端已到达）+ 读 send 前按 chunk 失效
  （跨 wave 陈旧缓存行）。
- **退出崩溃**：静态析构晚于 NPU runtime 终结导致双 rank SIGSEGV；修复
  为 LIFO atexit 自动拆卸 + 显式 `memfabric_o_proj_shutdown()` op。

**实机新事实（已登记，防止回退）**：

- 本机 4 张 310P3 卡中，设备 0,1 卡全功能可用；设备 4,5 所在卡
  `AclrtMemSetAccess` 返回 507899（跨 die VMM 授权失败），不可用于本
  管线；设备 2,3 卡 npu-smi 显示 Alarm。**bring-up 一律用 0,1**。
- 310P HCCL（CANN 9.1.0）不支持 BF16 allreduce（HcclAllreduce 直接拒
  绝）；reference 路径用 FP32 allreduce 后转回 BF16。生产路径不受影响。
- dav-2002 AICore 标量环境无 `bfloat16_t`（仅 float16_t）；BF16 reduce
  用位级转换（round-to-nearest-even）。
- KFC 通道（KFCKernel/system dlsym 回退）在本机可用：`test -f` 缺失
  文件按设计回传 507018，退出码 0 命令 sync=0；同流非零退出后可继续
  launch（流不毒化）。
- `libascendc_runtime.a` 为设备对象链接必需（AscendLaunchKernelWithHostArgs
  / AscendGetFuncFromBinary / AscendProf* 桩）。
- 性能基线（Phase-1 标量 reduce + 每 wave 双 barrier）：M=1 约 3.9ms vs
  HCCL 0.6ms；M=2048 约 161ms vs 5.3ms。**正确性已达标，性能差距为
  Phase-2/reduce 优化的明确目标**（研发计划 §9/§10）。

### 4.11 目标模型 o_proj 实为 FLOAT（已裁决：方案 A 重定基线为 BF16）

2026-09-18 实机/仓库核查（P2 W8A8 单层验证启动时发现）：

- **目标检查点的 full-attention `self_attn.o_proj` 未量化**：本地
  `/home/models/Qwen/Qwen3.6-35B-A3B-w8a8` 与 modelscope 上游
  `Eco-Tech/Qwen3.6-35B-A3B-w8a8`（仅下载 quant_model_description.json
  核对）完全一致：11 个 o_proj 条目全部 FLOAT；仅 routed MoE experts 为
  W8A8_DYNAMIC（92251 条目）。需求基线"310P static W8A8 o_proj"的
  eligibility（AscendW8A8LinearMethod310 + input_size 4096 → 2048 → N 2048
  + BF16）在真实目标模型上**永不命中**，E2E 验收"full-attention o_proj
  确实走融合路径"无法按当前需求达成。
- **310P npu_quant_matmul 不支持 BF16 输出**：CANN 9.1.0
  `aclnnQuantMatmulWeightNz` 仅接受 INT8/FLOAT16 输出（实机报错
  EZ1001 DT_BFLOAT16 拒绝）；plain 布局权重在 310P 上不可用（必须
  FRACTAL_NZ + transpose）。需求"output dtype = BF16"在 W8A8 语义下
  本机不可实现；现有 W8A8 生产路径只可能以 FP16 输出运行。
- 本地 `Qwen3-30B-A3B-w8a8`（qwen3_moe，48 层 dense attention，
  torch_dtype bfloat16）的 o_proj 为完整 W8A8 static（weight/scale/
  offset/input_scale/quant_bias/deq_scale 齐备）——是唯一满足 W8A8
  语义的本地检查点，但模型与需求指定的 Qwen3.6 不同（且同样受
  BF16 输出限制）。

**裁决结果（2026-09-18，项目负责人）：方案 A** —— 按真实检查点重定基线：

- 融合目标改为 Qwen3.6 未量化 BF16 full-attention o_proj（per-tile BF16
  matmul + MemFabric exchange），W8A8 机制（deq_scale/quant_bias/NZ）整体
  移除；BF16 payload 语义保留。
- requirements/设计文档已同步修订（标题、范围、eligibility、数值语义、
  Phase-2 producer、验收 C）。
- 310P 实机已验证 BF16 matmul（torch.mm）可用。
- 代码改造点：eligibility 改查 `AscendUnquantizedLinearMethod`；Python
  编排 MM 改 `torch.mm`；dispatch 挂接点从 w8a8_static 迁移到未量化
  linear 路径；单层验收 harness 改 BF16 字面比较。
- 曾评估的 B（换 Qwen3-30B）/C（重新量化）不再采用。

> **后续修正（2026-09-20，见 9.1/9.2）**：本节裁决时的 BF16 假设随后被实机
> 推翻——本机 CANN 的 BF16 NZ linear 不支持，模型整体以 FP16 运行，融合
> 管线交换/规约均为 FP16。当前契约以设计文档 §2 为准。

### 4.12 P3 验收记录：Qwen3.6 TP=2 eager 端到端通过（2026-09-20）

环境事实（本轮新增）：

- `wgm-dev-310p` 上游已更新（本地 `d7f11c6d`）：编排 kernel 三件套接入主构建，
  并内置 **KFC 通道自部署**（`SdmaOrchestrator::KfcExec`/`MigrateViaCustChannel`
  /`DeployOrchestratorKernel`，`KFCKernel` json + `system` 函数 dlsym 回退）——
  设备侧 kernel 缺失时按"验证门 → CUST 迁移 → 设备内 cat → 加载"自动部署，
  ex08 实测部署成功（加速比 2.31x）。
- vllm worker 场景注意事项：multiproc 默认 fork 与 MemFabric 库链的多线程
  状态组合会导致 dlopen 重定位极慢（10min+），须 `VLLM_WORKER_MULTIPROC_METHOD=spawn`。
- KFC 自动部署在已初始化 HCCL 的 torch worker 中曾报 launch 507018（当时的直接
  原因是设备 kernel 缺失走到自动部署分支）；设备 kernel 经 ex08 自动部署就位后，
  bench/torchrun/vllm 的 gate 直接命中已部署路径，全链正常。

P3 验收结果（TP=2 eager，die0/1，FP16）：

- 模型加载 18.85 GB，KV cache 7.9 GiB，warmup 正常；
- **full-attention `o_proj` 命中融合路径**（worker 日志 "Enable 310P3 TP=2
  MemFabric unquantized full-attention o_proj pipeline (tile_m=32,
  chunk_bytes=131072, FP16 exchange)"）；linear-attention 层因 prefix/
  layer_types 检查结构性不命中；
- 3 轮 × 4 prompt 共 12 次 generate 全部成功，同 prompt 跨轮输出逐字一致；
- feature-off baseline 对比：12 输出中 10 个逐字一致，2 个存在 token 级细微
  分歧（如"和助手"vs"与助手"）——根因是 HCCL allreduce 内部累加精度与融合
  路径 FP32-math reduce 不同（P2 单层验收已证融合与 FP32-math 语义 bit 级
  一致），属预期数值行为而非正确性错误。

### 4.13 P4 验收记录：profiler 证明 MM 与通信/规约真实重叠（2026-09-20）

采集方法：`torch_npu.profiler`（`_ExperimentalConfig(export_type=Text,
profiler_level=Level1)`）包裹融合调用，其 `export_only_prof_dir` 产出 CANN
kernel 级时间线（`task_time_*.csv`：kernel_name/AI_CORE|AI_CPU/MEMCPY_ASYNC、
stream_id、task_start/stop）。分析工具已固化为
`benchmarks/scripts/analyze_310p_memfabric_overlap.py`。

代表性 workload（rows=2048，tile_m=32，单 wave 64 tiles，FP16，TP=2）：

```text
waves detected: [64, 64]
== wave: 64 matmul tiles, compute span 8.859 ms, tile avg 32.6 us ==
reduce consumer: active window overlaps compute span by 8.859 ms (dur 84.158 ms, stream 49)
AICPU orchestrator: active window overlaps compute span by 8.859 ms (dur 9.219 ms, stream 2)
matmul tiles (t>=1) overlapping reduce/AICPU activity: 63/63
OVERLAP PROVEN
```

结论：compute stream（stream 24，te_matmul + mf310pPublishKernel 交错）的
**每一个后续 MM tile（63/63）**都与独立流上的 reduce consumer（stream 49）
和 AICPU SDMA 编排 kernel（stream 2，窗口覆盖全 wave）存在真实设备侧时间
重叠 —— R4/验收 E 的 timeline 证据达成（非"无显式 wait"推断）。wave 内
还可见 63 条 MEMCPY_ASYNC（mailbox/flag 清理与数据通路）与计算并行。

证据文件：`/tmp/opencode/p4_evidence/task_time_rank0.csv`（随 PR 归档路径
见 usage 文档）。

### P0.1 确认定制 MemFabric install 产物

在 `wgm-dev-310p` 编译安装后记录：

- install prefix；
- `smem.h` 路径；
- `smem_shm.h` 路径；
- `smem_shm_aicore_sdma.h` 路径；
- 真实生成的 `.so/.a`；
- transitive link dependencies；
- runtime `LD_LIBRARY_PATH` / RPATH 要求。

将实际 library list 写入：

```bash
VLLM_ASCEND_310P_MEMFABRIC_LIBRARIES
```

不要猜官方 MemFabric library 名。

### P0.2 确认 `.asc` 编译规则

找到 `wgm-dev-310p` example 08 实际使用的 device build 规则，用同一套 310P toolchain 编译：

```text
csrc/memfabric_o_proj/external/memfabric310p_device.asc
```

得到可被 `vllm_ascend_C` 链接的 object，并设置：

```bash
VLLM_ASCEND_310P_MEMFABRIC_DEVICE_OBJECT=/abs/path/memfabric310p_device.o
```

如果当前 `.asc` 语法、launch stub 或 object 形式不兼容，应优先按 example 08 的实际编译方式修改，不要引入另一套猜测工具链。

### P0.3 修复所有编译错误

优先级：

1. header/API signature；
2. namespace/type；
3. device launch stub；
4. linker symbol；
5. RPATH/runtime loader。

每解决一个错误，记录原始错误信息和修复 commit。

## 5. P0-P4 历史记录（V4 时代，已验收，规划文本已废弃）

P0-P4 在 V4（kfc/workspace API）形态下完成并验收，其过程规划文本（V4
host API 核对、phase-1 staged 管线、wave flag 协议等）已随 V5 迁移整体
废弃并删除。验收结论存档：

- **P0 编译打通**（2026-09-17）：cmake/adapter/.asc/bisheng 工具链全链路
  打通；CUST 迁移通道与 CP1 部署模式实测定论。
- **P1 通信协议正确性**：bilateral SDMA + local reduce 与 HCCL reference
  对齐；1000-wave stress 无 stale/丢件；wave boundary 双 barrier 协议
  （sync + clear + barrier）定型。
- **P2 单层 correctness**：staged 管线（per-tile F.linear + publish）对
  `F.linear + FP32 allreduce` bit-exact。
- **P3 端到端 bring-up**：Qwen3.6-35B-A3B TP=2 eager 生成正确。
- **P4 profiler overlap 验证**：profiler timeline 证明 MM[t+1] 与
  SDMA[t] 实际重叠；phase-1 性能为负（per-tile launch + staging copy
  开销主导）——这是 P5 direct producer 的直接动机。

V4 链路在该服务器的失效记录（2026-09-20）：设备侧 v4 kernel 消失、
kfc_min.so 缺失导致 kfc 通道不可用、CUST 直跑必 0x2a。全部证据与迁移动机
见 §9.2。

V4 时代的旧 benchmark `bench_310p_memfabric_o_proj.py` 与 phase-1
begin/publish/finish op 面已随迁移删除（git 历史可查）。


## 9. P5：direct producer（已完成）

研发目标（均已达成）：

- 310P Cube matmul primitive（AscendC Matmul API）直写 symmetric send
  arena，保持未量化 FP16 语义；
- tile 完成后立即 clean+signal，一个 producer kernel 内继续下一 tile；
- 移除 local copy 与 per-tile launch（phase-1 的两大负开销）。

### 9.1 实现记录（2026-09-20，dav-2002 / CANN 9.1.0 / bisheng）

P5 producer 已按上述目标实现（FP16 未量化路径，owner decision A）：
`Mf310pDirectProducerKernel`（csrc/memfabric_o_proj/external/
memfabric310p_device.asc），`memfabric_direct_o_proj_allreduce` 唯一入口
op（V5 迁移后 phase-1 staged 管线已整体删除，见 9.2）。

AscendC Matmul 在 dav-2002 上的位级正确配方（对照 `F.linear(x, w_nz)`，
m=1..4096 全 bucket bit-exact，探针 /tmp/opencode/p5_matmul_probe/probe9）：

- A `MatmulType<GM, ND, fp16, false>`，B `<GM, NZ, fp16, true>`，C
  `<GM, ND, fp16>`，`CONFIG_MDL`，base tile 128^3；
- 静态 tiling `GetMatmulApiTiling`，M 按 2 的幂 bucket 特化
  （16/32/64/128/256/512/1024/2048/4096）；bucket < 256 时必须传
  `l1Size=128KB` 抑制 A-full-load 调度（默认调度的 UB 中转需求超过
  2002 的 256KB UB，触发 aicore 异常 0x26 MTE 越界）；
- `enVecND2NZ=true`：OnTheFly ND2NZ 路径在 2002 上有 M 行广播 bug，必须走
  向量路径；
- `pipe.InitBuffer(ubBuf, TOTAL_UB_SIZE); mm.SetLocalWorkspace(ubBuf)` 为
  生产必选（mat_mul_base_kernel.h:82-85 同款）；
- `SetTensorB(bGm, true)` 必须显式运行时转置 flag；
- `REGIST_MATMUL_OBJ` + 零初始化 `TCubeTiling`；
- 宿主侧：每个 kernel symbol 首次 launch 是静默哑弹（binary eager-load
  no-op），必须 warmup 两次（chunk_count==0 零副作用 shape）。

producer 形态（probe10 验证）：单 kernel 循环
`SetTensorA/B -> SetOrgShape -> IterateAll -> clean slot -> notify slot ->
下一 chunk`，一个 Matmul/TPipe/UB 复用多次 IterateAll，bit-exact。partial
tail chunk 用 `bucket_of(rows_last)` 的第二实例，从 host 准备的 scratch
（memset+memcpy）读取避免 x 越界读，slot 未覆盖行由 host memset 归零保持
确定性。

wave 协议（V5 定稿，见 9.2）：join -> warmup -> producer(s) -> waiter ->
add_out；multi-wave 由 op 内循环处理（max_rows_per_wave = 64 * tile_m）。
tile_m 必须为 [16,4096] 内 2 的幂。

已知问题/后续优化：M<16 仍按 bucket 16 计算（decode M=1 粒度下限）；
单 core（usedCoreNum=1, <<<1>>>）为 correctness baseline，多 core/N-panel
为 P6+ 优化项。

### 9.2 V5 epoch API 迁移（2026-09-21，dav-2002 双卡实测）

V4 链路在该服务器失效（设备侧 v4 kernel 被移除、kfc 通道因 kfc_min.so
缺失不可用、CUST 直跑必 0x2a），整体迁移到 V5
（memfabric_hybrid origin/wgm-dev-310p，24de19cb，安装于
/opt/memfabric-wgm-dev-310p-v5）。厂商示例 08（ex08）双卡 PASS 证明
AICPU/epoch/SDMA 数据面健康，v6 kernel 已由同事部署在 CP1 路径。

V5 契约与代码变化：

- 设备 API：`smem_shm_sdma_reserved(gva)` 预计算 + `signal_at/wait_at/
  quiet_at`（32B 请求邮件环 64 槽、到达环 1024、段尾保留区 48KB）；
  kernel 入口 `HybmSdmaOrchEpoch`（MF_SDMA_ORCH_KVER=6，json 须显式
  MF_SDMA_ORCH_JSON 指定——split 布局下自动发现失效且 kfc 回退不可用）；
- producer kernel：matmul -> 64B 行级 clean（ex08 姿势，替代
  ENTIRE_DATA_CACHE）-> `signal_at`（dst=对端 recv arena 槽，imm=chunk）；
- 新增 waiter kernel：`quiet_at`（本端信号全落地，send 可复用）->
  `wait_at` x chunks（按序收件，status != OK 早退）；
- host 数据面全部废止：op 协议变为 per-wave
  `join（stream 同步 + control barrier，防跨 wave recv 覆写竞态）->
  warmup（每 bucket 2 次 + waiter 2 次，哑弹规避）-> producer(s) ->
  waiter -> at::add_out(out, send, recv)`，全程单流无 chunk 间 host 同步；
- ABI v3：layout 增 `pool_base`（kernel 侧保留区推导）与 `local_size`，
  删 arrival flags/2MB 保留区/mailbox 常量；删 phase-1 全套
  begin/publish/finish/mark_failed op 与
  `VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_DIRECT` env（direct 成为唯一路径）；
- cmake：V5 split 布局（smem/include/{host,device}、四目录 lib64、
  rpath 覆盖全部、json 必须显式指定）。

E2E 验证（bench_310p_memfabric_o_proj_layer.py，tile_m=32，repeat=5）：
rows = 1/8/32/33/64/128/512/2048/4096 全部 bit-exact PASS
（max_abs_diff=0.000000），含单 wave 64-chunk、4096 双 wave、tail
bucket。热路径 4096 行 ~25ms（未调优，P6 目标）。

### 9.3 V5 平台约束（本机实测，集成必读）

V5 的"常驻 epoch kernel"设计（~22s 自限 + 监督线程立即续投 = 设备上
永远有活着的 AICPU 任务）在本机引出三个硬约束，bench 已逐一绕开，
生产化需 memfabric 侧跟进：

1. **device 级同步死锁**：`torch.npu.synchronize()`
   （aclrtSynchronizeDevice）等待所有设备任务，包括常驻 epoch →
   永久挂死；约 25-28s 后 epoch 被 launch-timeout 击杀 → tsdaemon 杀
   aicpu-sd → HDC 断连（507901）→ 设备 AICPU 报废直至自愈。实测
   torch_npu 的 **HCCL 集合通信（barrier/allreduce）内部同样做
   device 级同步**——池存活期间两者都不可用。对策（已实现）：池存活
   期间只用 stream 级同步；reference HCCL 全部前置到建池之前。
2. **epoch 25s launch-timeout 击杀窗口**：本机 usleep(100µs) 实际
   ~127µs+，epoch 自限实际 ~28s > 25s launch attr → 任何池寿命
   超过 ~25s 必然被击杀并砖化 AICPU。对策（已实现）：单进程内将
   池生命周期控制在 ~20s 内。**根治需要 kernel 侧缩短
   HYBM_SDMA_ORCH_EPOCH_MAX_LOOPS（220000 -> ~120000）并 bump KVER
   重部署设备侧 .so——kfc 已死无法自动部署，需同事/厂商操作。**
3. **teardown HDC 静默断连**：`smem_shm_destroy` 期间段 unmap 与
   epoch 停止存在窗口，epoch 可被无日志 fault-kill → HDC 断连 →
   destroy 之后任何设备操作（含 HCCL）失败。对策（已实现）：bench
   在 shutdown 后不再做任何设备/集合操作，直接退出。

诊断工具：`torch.ops._C_ascend.memfabric_o_proj_debug_snapshot()` 同步
D2H 转储双端保留区协议字（reqHead/reqTail、quiet/arrival 戳、邮件像、
arena 首字），可在 waiter 挂死时判定卡点（本轮用它证实了 wave 完成、
挂点在 device 同步）。

## 10. P6：性能优化（进行中）

V5 迁移后的实测基线（tile_m=32，首 call 计时）：小 M ~90ms、4096 行
（2 wave / 128 chunk）~25ms。其中存在三个可分离的大头：

1. **per-wave host join（stream 同步 + control barrier）**：TCP rendezvous
   波动 ~0-70ms，decode 小 M 时是单项最大开销。
   优化方向：以设备侧 ack/gate 替代——对端 add_out 后由 ack kernel 回签
   8B（imm=wave id），本端下一 wave 前由 gate kernel wait_at 收签；
   到达环 FIFO 序 [data x N, ack] 保证双 kernel 交错消费自洽。join 可
   整体删除（首 wave 除外）。
2. **单 core matmul**：producer 用 usedCoreNum=1 / <<<1>>>（correctness
   baseline），128 chunk 串行。优化方向：多 block 每 block 独立 chunk
   的 GEMM（配方不变保 bit-exact），signal 以"块序自旋 + 顺序递增
   REQ_TAIL"保持 FIFO，避免并发环写。
3. **clean 成本**：128 KiB/chunk 的 64B 行级 dcci 循环，评估批量行
   clean 或每 chunk 一次 ENTIRE_DATA_CACHE 的取舍。

执行顺序：先给 bench 增加 per-repeat 计时与分相分解（区分 join/matmul/
clean/wait/add），用数据定优先级；每步保留 9/9 rows bit-exact 回归；
同时用 profiler timeline 重新量化 overlap（V5 下 R4 证据需重做）。

硬约束（9.3）：调优运行必须把池寿命控制在 ~25s 内，profiler 场景
尤其注意 device 级同步入口（torch profiler 自身会 device-sync！需要
验证或改用 msprof/task_time 导出方式）。

## 11. P7：ACL graph

Eager 路径通过后：

1. 去掉 `--enforce-eager`；
2. capture；
3. replay 多轮；
4. 检查 persistent GVA；
5. 检查 wave flag generation/reset；
6. 检查 graph replay 是否重复执行 host-only barrier；
7. 检查不同 M bucket/graph 的资源复用。

如果 graph 失败，不要先修改算法；先最小化复现为一个融合 op graph capture case。

## 12. 性能验收指标

至少保存以下 baseline/fused 数据：

- 单层 `o_proj + allreduce` latency；
- prefill latency；
- TTFT；
- decode TPOT；
- output tokens/s；
- prompt tokens/s；
- peak memory；
- SDMA bandwidth；
- overlap ratio/timeline；
- CPU/AICPU overhead。

测试至少覆盖：

- decode 小 M；
- 中等 batch；
- 大 prefill M；
- 长时间稳定运行。

## 13. 当前风险清单

### R1：V5 平台约束对宿主进程的侵入（最高风险）

常驻 epoch 使池存活期间 device 级同步/HCCL 集合通信不可用、池寿命
<25s、teardown 后 HDC 不可靠（见 9.3）。当前 bench 已绕开，但 vLLM
model runner 内的 synchronize 调用点、graph capture 路径、worker 生命
周期管理都需逐点排查。根治依赖 memfabric 侧改造（epoch 常驻策略 /
暂停 API），需厂商跟进。

### R2：epoch 25s 击杀窗口未根治

kernel 侧缩短自限 + KVER bump 重部署尚未执行（kfc 不可用，需同事/
厂商）。根治前，任何 >25s 的池存活场景（包括未来 vLLM 长稳测试）都会
砖化 AICPU。

### R3：overlap 证据需在 V5 下重做

V4 时代的 profiler 证据不再适用；且 torch profiler 的 device-sync 行为
可能与约束 1 冲突，需要 msprof/task_time 导出方式重新量化。

### R4：小 M 没有足够流水深度

decode M=1 单 chunk 无流水可言，收益主要靠去除 join（P6 第 1 项）；
若仍不达标需评估 N-panel。

### R5：ACL graph（P7）

host join 若保留在 wave 协议中则不 graph-safe；P6 的 ack/gate 设备化
顺带改善 graph 兼容性，但 capture/replay 边界仍需单独设计。

### R6：generic `memfabric_hybrid` Python dependency

仓库 `requirements.txt` 仍存在通用 `memfabric_hybrid`。服务器 bring-up 必须确保使用的是定制 `wgm-dev-310p` 安装，不要让 pip 自动下载一个上游版本后误以为已经满足本项目依赖。

## 14. 每次服务器迭代建议记录

每轮提交至少记录：

```text
Date:
Commit:
CANN:
Driver/Firmware:
SOC_VERSION:
PyTorch:
torch-npu:
vLLM:
vLLM-Ascend branch:
MemFabric commit:
MemFabric install prefix:
MemFabric libraries:
Device .asc compile command:
Feature env:
Test command:
Result:
Profiler artifact:
Known issue:
Next action:
```

建议把关键日志、编译命令和性能结果持续回填到本文或 PR #1，避免后续交接重复踩坑。

## 15. 完成定义

该项目可以从“WIP”转为“完成”时，应至少满足（括号为当前状态）：

- custom MemFabric（V5）从源码可重复编译安装（已达成）；
- vLLM-Ascend 可一条明确流程编译（已达成）；
- TP=2 repeated-wave correctness（已达成，repeat 逐次校验）；
- 融合 op 单层 correctness bit-exact（已达成，9/9 rows）；
- 端到端 Qwen3.6 generation 通过（待 P6 后执行；受 R1/R2 约束，
  需先完成 vLLM 进程内 device-sync 排查与 epoch 击杀根治）；
- ACL graph replay 稳定（P7）；
- profiler 证明 V5 下 overlap（P6，R3）；
- 性能相对 baseline 有明确收益或至少能解释瓶颈（P6）；
- 文档中的编译、部署和启动命令经过服务器实测并更新为最终命令
  （usage guide 已按 V5 实测更新）。
