# 310P3 MemFabric o_proj 融合算子（ABI v7）实机测试报告

日期: 2026-09-22 | 联调: w00804146 (10wvw01)

## 1. 环境与版本

| 项 | 值 |
|---|---|
| 硬件 | Ascend 310P3 ×2 die（卡5664, dev0/dev1, bus 18:00.0），TP=2 |
| CANN | 9.1.0 (/usr/local/Ascend/cann-9.1.0) |
| 框架 | torch 2.10.0 + torch_npu 2.10.0.post4, vllm 0.27.1 (source /vllm-workspace/vllm), python 3.12.13 |
| vllm-ascend | feat/310p-memfabric-public-api @ **77cfc8e7**（本轮提交；基线 27ca425b） |
| memfabric_hybrid | wgm-dev-310p @ **8e24b56b**（本轮提交；自 10784295 重建，v9 内核 HYBM_SDMA_ORCH_KVER=9） |
| 设备侧内核 | libmf_sdma_orch_v9.so 经 deployer+mini_kfc 预部署于 /usr/lib64/aicpu_kernels/0/aicpu_kernels_device/ |
| 模型 | Qwen3.6-35B-A3B-w8a8（qwen3_5_moe_text, 40 层, full_attention_interval=4 → 10 个 full-attn o_proj; K_global=4096/local=2048, N=2048, FP16） |
| 关键约束 | MemFabric 仅支持 ASCEND_RT_VISIBLE_DEVICES=0,1（meta GVA 授权按逻辑 device id，其余 die 对 507899 实测失败） |
| 构建 | bisheng --npu-arch=dav-2002；feature-on/off 双构建均验证通过 |

## 2. BUG 清单、根因与修复（均已修复并回归）

### BUG#1 feature-off 构建导入崩溃（vllm-ascend）
- 现象: `ImportError: undefined symbol _ZN11vllm_ascend31memfabric_o_proj_debug_snapshotEv`
- 根因: binding 无条件注册 debug op，但 `memfabric_o_proj_debug_snapshot()` 仅在 feature-on `#ifdef` 分支定义。
- 修复: `#else` 分支补 stub（返回 `zeros({24}, kLong)`，镜像 feature-on 无上下文结果）。
- 回归: feature-off 构建 import/注册/fused op 清晰报错/snapshot 全零 → PASS。

### BUG#2 层基准 q=1/4 双侧 MAIL_MISMATCH trap（bench 脚本）
- 现象: q=1/q=4 首次运行 `mf310pWaitBatchKernel` aicore trap。
- 根因: `VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_BATCH_BASEM_COUNT` 只在 rank0 设置（rank1 用默认 q=2）→ batch_bytes/mail imm-len 两侧期望不一致。
- 修复: env 在 dist init 前对全部 rank 设置。
- 回归: q=1、q=4 各 9 个 M 全部 PASS（修复前 100% trap）。

### BUG#3 融合 MM 效率（核心，.asc 设备内核）
- 现象: producer 1024µs/批 vs stock ~246µs/批；fused 17.8ms@M=8192 vs stock ~6.93ms。
- 根因: AscendC 经典 `Matmul` 是**单核语义**（每个 block 从 tensor 基址消费同一 tiling），原代码把整批形状填入 `MatmulShapeParams` singleCore 字段 → 8 核全量冗余计算整批（正确但 ~8×工作量）。
- 修复: 8 核协作改为显式 per-block M 分区——A/C 连续行切片视图（`aGm = xGm + block·(BATCH_M/8)·K` 等），B(NZ) 全量共享；tiling `{MB/8, 2048, 2048}` basic `{MB/8, 256, 64}`，**CONFIG_NORM**（CONFIG_MDL 在该 per-block M 下实测片上 MTE 越界 aicore 507015 崩溃）；cache clean 收窄为各核只清自己写的连续 C 行。
- 结果: producer 273µs/批；fused@8192 **5.65ms（-18% vs stock 6.93ms）**；tiling 扫描 {basicN=512:283µs, 256:273µs, basicK=128:290µs} → 取 256/64。
- 回归: 全部正确性矩阵 bit-exact。

### BUG#4 运行包缺 310P 编排内核三件套（memfabric_hybrid）
- 现象: .run 安装后 orchestrator 加载失败（除非手设 MF_SDMA_ORCH_JSON）。
- 根因: `make_run.sh` 未打包 `output/hybm/aicpu_kernel`（launch json + CUST json + so）；运行时按 `<prefix>/<ARCH_OS>/lib64/../hybm/aicpu_kernel` 自动发现。
- 修复: 三件套随包打入 `${PKG_DIR}/${ARCH_OS}/hybm/`。已验证自动发现生效。

### BUG#5 安装器在 310P3 上中断（memfabric_hybrid）
- 现象: `install.sh` 芯片检测失败 exit 1 → set_env.sh 不生成。
- 根因: npu-smi 正则仅匹配 Ascend9xx，310P3 报 "310P3"；copy_extend 仅为 A2/A5 构建。
- 修复: 检测不到时 WARNING 跳过可选 extend lib 并继续安装。已验证 set_env.sh 正常生成。

### 环境问题（非代码缺陷，已规避/工具化）
1. **库内自动部署在 torch 进程内必败**: KFC probe miss（`test -f` 退出码 1）经流 sync 以 507018 返回后，torch_npu 注册的 ACL 错误回调将进程内 ACL 运行时闩锁（后续建流/launch 立即 507018）；独立进程无此现象。→ 必须外部预部署（`tools/aicpu_kernel_deploy/deployer`+`mini_kfc` P0 链），已在 orchestrator 代码中注释记录。
2. **设备侧 kernel 文件被延迟 janitor 删除** → 每次运行前 `ensure_kernel.sh` 重新探测/补拷。
3. **pip 残留 memfabric_hybrid 包抢先加载 site-packages 的 libmf_smem.so** → serve 内 json 自动发现失败（`mf310p_create ret=-7`）。处理: 卸载 pip 残留 + serve 脚本显式 `MF_SDMA_ORCH_JSON`。

## 3. 正确性（atol=rtol=0，全部 bit-exact）

| 项 | 结果 |
|---|---|
| q=2 全矩阵 11 个 M（255,256,257,511,512,513,1024,2048,4096,6144,8192） | **ALL PASS** |
| q=1 / q=4 各 9 个 M | **ALL PASS** |
| diverge-va（rank1 预占 4GB 使 GVA 基址分离）511/2048/8192 | **PASS** |
| UT 源码守卫 tests/ut/_310p/test_memfabric_o_proj_source.py | **19/19 PASS** |
| feature-off 构建（不含 MemFabric 依赖） | PASS |
| ACL Graph（gate7）: M=2048×32 replay、M=8192×16 replay、replay 后 eager 连续性 | **双 rank 全 bit-exact** |
| serve 级输出（Qwen eager, TP=2, greedy） | 统计等价（见下） |

serve 级说明: 基线栈自身对大 prompt 跨 run 非确定（stock 服务器 3 次自比, p0/p1/p3 均发散）。负对照: n=512 prompt 在 fused 服务器走纯 stock 路径（M<4096 不融合）仍与 stock 服务器输出不一致 → 证明输出发散 ≠ 融合路径因果。前缀一致长度中位数: p0 S~F 0.34 落在 S~S 0.18 / F~F 0.42 包络内；p1 三组均 ~0.01；确定性 prompt（4095/64）双端完全一致；融合路径 prompt（4100）存在共享变体。结论: 融合输出分布与 stock 统计等价，未见融合特异性发散。

## 4. 性能

### 4.1 算子层头对头对比（核心指标）：stock（o_proj matmul + HCCL allreduce 总耗时）为基准

同层同权重、同进程交替计时（3 轮×33 次取中位，/tmp/opencode/ab_headtohead{,_small}.log）：

| M (q=2) | stock 基准 | fused 总耗时 | 相对基准 | 判定 |
|---|---|---|---|---|
| 512 | 0.466ms | 0.838ms | 慢 79.7% | **劣化** |
| 1024 | 0.830ms | 1.100ms | 慢 32.5% | **劣化** |
| 2048 | 1.486ms | 1.737ms | 慢 16.9% | **劣化** |
| 4096 | 3.245ms | 3.028ms | 快 6.7% | **提升** |
| 6144 | 5.140ms | 4.349ms | 快 15.4% | **提升** |
| 8192 | 6.836ms | 5.654ms | **快 17.3%（吞吐 +20.9%）** | **提升** |

边界补充行: M=256 慢 286.7%（半批 padding）；M=511 慢 213.1%；M=513 慢 281.3%（尾批仅 1/512 有效行，比整批 512 慢 2×）。

**结论**: 盈亏交叉点实测在 M=4096——以下劣化且 M 越小劣化越重，以上提升且 M 越大提升越大（可掩盖的 allreduce 通信量 ∝ M）。生产配置 `MIN_M=4096` 路由使 M<4096 自动走 stock 路径（劣化不会发生），融合仅在 M≥4096 启用（兑现 +6.7%~+17.3%）。

### 4.2 小 M 劣化原因（关键点）

1. **固定协议开销摊不平**: 每次融合调用固定付 gate/quiet/ack + 16×(producer/wait/add) 内核发射 ≈ 0.4~0.9ms，与 M 无关；M=512 时 stock 全程才 0.466ms，纯开销即超过它。
2. **尾批 padding 白算**: batch_m=512 对齐，非整倍数 M 的尾批（如 513→2 批，尾批 1/512 有效）仍整批计算+搬运。
3. **可掩盖通信量 ∝ M**: M=512 仅 2MB（HCCL ~0.2ms），掩盖收益 < 固定开销 → 净劣化；M≥4096（16~32MB）才反转为净收益。
4. **小批量 8 核 MM tiling 效率低**: per-core M = batch_m/8 = 64 行，producer 273µs/批 vs stock 等价 ~246µs/批（+11%）。

### 4.3 serve 级 TTFT 验证（稀释效应，o_proj M = prompt tokens）

单请求并发 1、max_tokens=1（TTFT≈纯 prefill），每点 5 次取中位（/tmp/opencode/sweep_{stock,fused,forced}.json）：

| M | stock (ms) | fused 生产路由 (ms) | Δ | fused 强制全融合 MIN_M=1 (ms) | Δ |
|---|---|---|---|---|---|
| 512 | 560 | 579 | +3.3% | 577 | +3.0% |
| 1024 | 914 | 932 | +1.9% | 928 | +1.5% |
| 2048 | 1696 | 1697 | +0.1% | 1706 | +0.6% |
| 4096 | 3371 | 3388 | +0.5% | 3394 | +0.7% |
| 6144 | 5111 | 5126 | +0.3% | 5121 | +0.2% |
| 8192 | 6799 | 6817 | +0.3% | 6827 | +0.4% |

三组配置 serve 级全部持平（差异 ≤3%，其中 512/1024 的 +1.5~3.3% 为服务器实例间噪声——生产路由下 M<4096 本走 stock 路径，与强制融合几乎同值可证）。**稀释效应**: o_proj 仅 10 层 × 毫秒级 = prefill TTFT 的 0.1~1%（如 M=8192 层级省 1.18ms×10=11.8ms，占 6799ms 的 0.17%），算子层 ±17% 效应映射到 serve 为 ±0.2~0.7%，低于实例噪声。另有并发负载 A/B（32req×4096-token, conc=4）: TTFT med 7.749 vs 7.773s、e2e med 22.83 vs 23.34s、decode 4.24 vs 4.12 tok/s——中位持平略优，尾部见 §5。

### 4.4 其他性能数据

- Producer 内核中位 **273µs/批**（8 核 M-split）vs stock per-批等价 ~246µs（+11%，超出 P0 门 ~5%，端到端反超见 4.1）。
- Overlap 证据（M=8192）: wait 中位 1.8µs vs 串行 SDMA 下界 104.9µs（2MiB@20GB/s）→ SDMA 完全被后续 MM 掩盖；add 40.8µs 与下一批 SDMA 重叠；`Block Num=8`。
- 诊断开关 `VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TRACE=1` 有真实开销（decode -27%），默认关闭。
- 未采纳路线（勘明未实现）: N-split per-core B/C（诊断实测 producer 147µs/批潜力），需权重按核 NZ 重排 + gather-add，列为后续优化。

## 5. 稳定性与已知上游问题

| 项 | 结果 |
|---|---|
| repeat=1000 @8192 | PASS（med 5.65ms, max 6.7ms） |
| 60s soak（34k 次 @2048） | PASS（max 33ms） |
| mixed-M 交错（10 序列×100） | PASS |
| 5~12min 连续负载 soak | **3/3 wedge**（r1 ~10-12min, r2 ~10min, r3 ≤5min 内） |
| 进程生命周期 | 每次 torchrun 完整 create→use→destroy ~20+ 次全部干净退出；graph 使用后保守跳过 destroy（设计行为） |

**Wedge 签名**（3 次一致）: 双 rank 同时 `mf310pWaitBatchKernel`/`mf310pQuietKernel` aicore 超时（507014），orchestrator 已加载、aicpu 无异常日志、kernel 文件完好、设备利用率归零、SIGTERM 不可杀需 SIGKILL，数分钟~10min 自愈。
**归属**: 上游 MemFabric epoch 续命（18s 网格）与 aicpu 调度/运行时交互（R6/D3 已知类别；发生率约每 6-20 分钟连续负载 1 次，与 serve 下 ~20s 可恢复冻结同源——serve 引擎内其他活动使其可恢复，纯融合 bench 下退化为永久 wedge）。**非本集成代码缺陷**；上游已反馈（D3 报告 + 本轮 3 次复现证据）。
**serve 影响**: 尾部延迟被冻结事件拉高（fused ttft_p95 59.3s vs stock 14.3s，1-2 次/4min 运行），中位数不受影响。

## 6. HCCL 共存（受控微基准）

MemFabric 池存活前后 HCCL allreduce（fp16 [1/8/64,2048]）中位延迟 0.209→0.205 / 0.217→0.218 / 0.232→0.232 ms → **无退化**；stock o_proj fallback（matmul+allreduce）建池后 0.32-0.35ms 健康；serve 级 decode（全 HCCL 路径）速率与 stock 持平。

## 7. 剩余风险与验收结论

风险（按严重度）:
1. **上游 D3/R6 wedge/冻结**: 连续融合负载 ≥5min 有 wedge 风险，serve 下为 ~20s 冻结拉高尾部延迟——生产化阻塞项，待上游修复。
2. **部署链**: torch 进程内自动部署不可用（ACL 闩锁）+ kernel 文件 janitor 删除 → 依赖外部 deployer 工具链，需服务化固化。
3. 纯 MM +11%（端到端已 -18%）；N-split 优化路线已勘明。
4. graph 使用后进程退出 SIGABRT（上游缺"只停线程不销毁池"API，已反馈）。
5. 多副本 libmf_smem.so 环境陷阱（pip 残留/opt 旧装）→ 部署需唯一安装源或显式 MF_SDMA_ORCH_JSON。

**验收结论（实机）**:
- 正确性（层级 bit-exact 全矩阵 + diverge-va + graph replay + serve 统计等价）: **PASS**
- 性能（M=8192 端到端 -18%；serve 中位持平略优；overlap 有 profiler 证据）: **PASS**
- 稳定性（60s/mixed/repeat1000/进程生命周期/HCCL 共存）: **PASS**；≥5min 连续负载受上游 D3 限制: **条件通过**（<5min 负载窗口内可靠，或待上游修复）

## 8. 交付物

- vllm-ascend `feat/310p-memfabric-public-api` @ 77cfc8e7（signed-off: 10wvw01）
- memfabric_hybrid `wgm-dev-310p` @ 8e24b56b（signed-off: 10wvw01）
- 工具: /tmp/opencode/{ensure_kernel.sh, v7_env.sh, run_layer.sh, v7_profile.py, stock_profile.py, gate7_graph.py, hccl_coexist.py, ab_headtohead.py, ab_serve.sh, ab_capture.py, ab_compare.py, bench_ab.py, patch_tiling.py}
- 证据: /tmp/opencode/{v7_correct_q*.log, v7_final_*.log, v7_soak_*.log, v7_prof_*.log/csv, gate7_m*.log, hccl_coexist.log, ab_*.csv, ab_*out*.json, ab_*serve*.log}
