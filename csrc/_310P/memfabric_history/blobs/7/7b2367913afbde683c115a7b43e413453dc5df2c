# V2 MTE2/Scalar 专项 Gate A/B 实机核验报告（2026-10-22）

> 状态：**Gate A 现场核验完成（除设备侧二进制直验）；Gate B 计时能力实机判定为不可用；Gate C 准入条件更新**。
> 分支：`diag/310p-v2-mte2-scalar-gate-ab-20261022`。
> 前序：`V2_MTE2_SCALAR_GATE_A_EVIDENCE_AUDIT_20261022.md`（离线审计）、`V2_MTE2_SCALAR_GATE_B_FORENSIC_DESIGN_20261022.md`（设计）。
> 本轮不修改正式 V2 Kernel、MemFabric 协议与 KVER11 部署；不开展长压测与算子优化。原始双 Rank Trace 保留服务器，仅提交脱敏证据。

## 0. 执行摘要

| 核验项 | 结论 |
|---|---|
| Gate A 加载身份 | Host SO / KVER11 JSON / 插件 SO 与 prefill provenance **逐字节一致**；历史版本差异已解释（§1） |
| 站点统计复算 | 从原始 CSV 独立重算，Gate A 文档数字**全部复现**（§2） |
| 74ms 异常 | 精确定位：窗口第 1 个请求、第 9 站、双 Rank 同时；根因 **unknown**（§2.2） |
| 跨 Rank 时钟 | compare2.py 单中位数校准**概念性无效**：偏移呈单调漂移，是真实算力偏斜而非时钟差（§2.3） |
| 拷贝归因 | 上轮"V2 每站点 +66us contiguous"结论**被推翻**：两臂 contiguous 均属 conv 路径，V2 目标站点净拷贝开销 **0**（§3） |
| Gate B 计时能力 | `GetSystemCycle()` 在 310P3 **编译可过、上机恒返回 0**（实机探针证实）；Kernel 内计时 **不可用**（§4.1） |
| 诊断变体 | 可用于受控隔离，但现有入口仅单 Chunk；LOOP/IBDONE16 底层支持 chunks=4（§4.2） |
| 已证实的瓶颈 | V2 站点级 +28.06/+16.78ms/步全部来自融合核自身（2.57 vs 1.94ms/站），与拷贝无关（§5.1） |

## 1. Gate A 现场核验：加载身份与历史版本差异

### 1.1 实测身份（观测值见 `evidence/gate_ab_verification_20261022/env_identity.json`）

| 组件 | 路径 | SHA256 前缀 | 与 provenance 一致 |
|---|---|---|---|
| Host core SO（site-packages，`__init__.py` CDLL 实际加载点） | `.../site-packages/memfabric_hybrid/lib/libmf_hybm_core.so` | `88d39285` | **一致** |
| Host core SO（/usr/local，set_env.sh 加入 LD_LIBRARY_PATH） | `/usr/local/memfabric_hybrid/1.2.0/.../lib64/libmf_hybm_core.so` | `88d39285` | **一致（同内容）** |
| KVER11 JSON | `/usr/local/memfabric_hybrid/1.2.0/.../libmf_sdma_orch_v11.json` | `25e16b0a` | **一致** |
| 插件 SO | `vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so`（editable 安装指向本仓库） | `929b53c2` | **一致** |

- 加载链确认：`import memfabric_hybrid` → `ctypes.CDLL(<pkg>/lib/libmf_hybm_core.so, RTLD_GLOBAL)`；采集脚本 `ORCH_JSON` 解析唯一版本目录 1.2.0，恰指向 v11 JSON。两臂 serve 进程已结束，`/proc/<pid>/maps` 不可倒推（符合 Gate A §9 预期），只能核验在位文件。
- 设备侧 `libmf_sdma_orch_v11.so`：host 文件系统无实体（KFC 部署于设备 FS，host 不可见），**无法直接验 SHA** → `unknown`。间接证据：`HybmSdmaOrchEpoch`（AI_CPU）恰各 1 次出现在 V2 双 Rank Trace（时长 14.92s/15.02s，跨全窗口），Stock Trace 为 0。Gate C0 须在下次设备会话用 deploy-helper 日志或新 Trace 复验。
- v12 残留：安装路径无 v12；`/tmp/.../kernel_v12_stage` 为惰性暂存；`/tmp/mf_kfc_<死PID>_*.json` 为 10-17 残留，按 PID 命名不影响新进程；当前 shell 无任何 MF_* 环境变量。

### 1.2 历史版本差异的裁定（关闭 Gate A §2 遗留问题）

- 当前在位 `88d39285` = G1-D 原装恢复态 = prefill 轮 = **fix-310p-kfc-deploy-helper@131ee5ff 构建产物**（10-19 02:15 热替换，见《310P3 版本溯源与资产保全报告》）。wheel `VERSION` 仍写 `2f59dedb`——**元数据低估了实际部署的修复线**，属已记录现状。
- W1/W4 ABBA 的诊断 SO `6e9dc2d6` 已从全部 host 路径清除（恢复态确认）。
- 结论：prefill Stock/V2 同会话对照内部一致；不得宣称与 W1/W4 环境逐字节相同；W1/W4 绝对数值与 prefill 不可直接比较。

## 2. 证据复核：基于原始双 Rank Trace 的离线审计

审计脚本：`evidence/.../scripts/audit_pass3.py`（另含 pass1/pass2 过程稿），只读原始 CSV；步定义改为**锚点站点间距 >80ms**（异步调度下全时间线最大 gap 仅 10.3ms，历史 >20ms 聚类在两臂都必然坍缩为 1 步——这也解释了为何必须用锚点间距）。修正后的统计（`gate_ab_audit.json`）：

### 2.1 站点统计（全部复现 Gate A 文档值）

| 每步站点和（ms） | Rank0 | Rank1 |
|---|---:|---:|
| Stock（5 步各自） | 77.87/77.90/77.82/77.85/77.82 | 93.05/91.34/93.82/93.64/94.17 |
| V2 稳态 4 请求均值 | **105.92** | **109.99** |
| 净差（站点事件和） | **+28.06** | **+16.78** |
| 稳态步周期差 | +32ms（2.396 vs 2.364s） | +31ms（2.411 vs 2.380s） |

复现项还包括：Stock MM 形状 `4096,2048;128,128,16,16`（M=4096）；MM p50 795.1us + AR p50 1149.1us；V2 融合核 p50 2574.3/2672.1us；非目标 hcom = 41 AR + 1 AG = 42/步/Rank；pipe 计数器 p50（mac 458.9/mte2 1711.4/scalar 873.0 vs Stock 456.6/335.9/186.5）与 `kernel_internal_breakdown.json` 逐位一致。

### 2.2 74ms 异常精确定位

- 位置：**窗口第 1 个 profiled 请求（step 0）、第 9 个目标站点（site_idx=8）**，双 Rank 同时（进入步后 505.4ms / 508.7ms）。
- r0 74231.7us / r1 74827.4us；`Wait Time` 仅 130.8/148.8us（排除队头等待）；同流前驱为正常 `aclnnMul`（gap 131/149us）。
- 单次出现（200 站点中各 1 次）；其所在请求含异常站点和 176.65/182.85ms，去除异常后 ≈102/108ms，与稳态一致。
- 根因 **unknown**。候选（未证实，禁止择一采信）：跨 Rank mail/SDMA 单次长尾、AICPU 编排核 epoch 交互、profiler 启动后首个请求的采集路径效应。注意 6 次预热请求在窗口外已跑过同形状，"首调"解释不成立。

### 2.3 跨 Rank 事件配对与时钟校准（修正 A-02）

- **配对方法**：hcom 内核名含全局代序号（`hcom_allReduce__503_<seq>_1`），同名即同代调用。Stock 410/410、V2 210/210 全配对、**零孤儿**（无丢行/多行）。V2 比 Stock 少 200 个 hcom = 被融合核替换的目标 AR，账目闭合。
- **修正符号后的关键发现**：`offset = t0 − t1` 并非常数，而是**单调漂移**——两臂均从约 −4ms 漂到约 −75ms（~−7ms/s，即每请求 −15.5ms）。若解释为时钟漂移需要 ~6500ppm，物理上不可能（晶振典型 <50ppm）。
- **独立佐证**：每步全内核忙时和显示 Rank1 比 Rank0 每请求多 +11~26ms（Stock）/ +13~18ms（V2）设备工作；Stock 的站点和差 +15.35ms/步与之吻合。
- **裁定**：漂移是**真实的渐进 Rank 偏斜**（Rank1 每步多做 ~15ms 设备工作，Rank0 在集合通信处等待吸收差额），不是可校准的时钟差。历史 `compare2.py` 的单中位数校准在符号与概念上双双无效；**跨 Rank 绝对时间结论继续不可采信**；每 Rank 本地时长、同代配对、事件先后关系有效。
- 附带事实：hcom 时长比 r1/r0 p50，Stock 1.105（Rank1 的 AR 实例更长）、V2 0.945；V2 中 Rank1 站点内仅 +4.07ms/步，其余 ~11ms/步偏斜在**非目标算子**（MoE/conv/非目标 AR）。

### 2.4 附带澄清：15 秒"拷贝"为 Profiler 归因伪影

V2 `operator_details.csv` 中单个 ~14.92s/15.02s 的 `aclnnInplaceCopy` Device Self Duration 与持久 `HybmSdmaOrchEpoch` 内核时长相差 ~1us——是 Profiler 把持久 AICPU 核归因到相邻 Host 调用的伪影，**不是真实阻塞拷贝**。

## 3. 拷贝归因：contiguous 与 D2D 的真实归属

| 证据 | Stock | V2 | 裁定 |
|---|---|---|---|
| host `aclnnContiguous` 调用（api_statistic） | 200 次/37.5ms | 200 次/16.1ms | 两臂都有，上轮"V2 无同名 Host API"只是没进 Top-25 |
| 设备 `aclnnContiguous_Slice` 内核 | 200 个，p50 531us，合计 81.1ms/窗口 | 200 个，p50 452us，合计 71.4ms/窗口 | **两臂同源同数量** |
| 这些内核的位置 | 后接 `Contiguous_Transpose` + `CausalConv1dV310`，领先下一目标站点 10–25ms（p50 23.9ms） | 同左 | **全部属于 conv/Mamba 路径，与目标站点无关** |
| 目标站点 10ms 内同流 contiguous 前驱 | **0/200** | **0/200** | 两臂目标站点输入均无紧邻 contiguous |
| V2 显式 `x.contiguous()`（memfabric_mm_ar.py:405） | — | 未产生任何邻接设备拷贝 | 输入已连续，设备层为空操作 |
| D2D `MEMCPY_ASYNC` 任务（>1us） | 449 个 / 82.6ms | 246 个 / 41.7ms | 数量随 hcom 集合数（410 vs 210）变化 = **HCCL 内部拷贝**；Stock 反而多付 ~2 倍 D2D 时间 |

**结论（推翻上轮 A-03 的条件估计）**：

1. "V2 每站点 +66us `x.contiguous()`、Stock 零拷贝"不成立——66us 实为 host 侧 launch 时长，"199/199 站点"是 ≥25ms 回溯窗把 conv 路径拷贝误配到融合核前。
2. **V2 相对 Stock 的目标路径净拷贝开销为 0**；"净 +3.6ms"条件估计撤销。
3. 站点级 +28.06/+16.78ms/步的差额**全部来自融合核自身时长**（2.57ms vs 0.80+1.15=1.94ms/站，p50），拷贝不是贡献项。

## 4. Gate B 能力验证

### 4.1 Kernel 内部计时：310P3 不可用（实机证据）

按 Gate B 设计 §3 的最小探针路线执行（独立于正式/诊断内核，纯 ACL，空闲 dev0，6 次发射 <2s）：

1. **编译级**：`bisheng --npu-arch=dav-2002`（与正式 `libmf310p_v2_trial_device.so` 同参）**接受** `AscendC::GetSystemCycle()`；阴性对照（守卫内放未定义函数）证明探针路径真实编译。→ 仅有编译通过本会给出"支持"的**假阳性**。
2. **运行级**：8 核 × 6 次发射，`magic=0xC0DE2026` 与 200k 循环累加值全部正确回读（内核确实执行），但 **`GetSystemCycle()` 恒返回 0**（全部核、全部发射、三点采样均为 0）。单调性/单位/跨核属性全部不可观测。
3. 与 CANN 9.1 公开文档"Atlas 推理系列 AI Core 不支持 GetSystemCycle(ISASI)/PrintTimeStamp"一致；寄存器猜测按设计禁令未尝试。

**裁定：310P3 无可靠 Kernel 内计时源。Gate B"新设备埋点"维持 No-Go（从文档推断升级为实机证据）。任何基于该 API 的"Kernel 内分段耗时"都会是伪造的零值。**

### 4.2 现有诊断变体：可用于受控隔离，但有两个等价性缺口

实测盘点（`mf310p_v2_diagnostics.asc` + `mf310p_v2_trial_diagnostics.cpp`，随正式库一起编入 `libmf310p_v2_trial_device.so`，经 `mf310p_v2_trial_diag` diag 0–7 派发，bench 入口 `bench_ar_only.py`/`t1_diag_single.py`）：

| diag | 变体 | 入口能力 | 隔离意义 |
|---|---|---|---|
| 0 / 2 | MM-only / V1-MM | 单 chunk（256/512/1024） | 纯 MM 地板 |
| 1 | NoIB | 单 chunk | MM+SDMA+Add，无 IB 会合 |
| 3 | IB | 单 chunk | +IB 会合 |
| 4 | IBDONE | 单 chunk | +DONE 握手 |
| 5 | 正式核单 chunk 串行 | chunks=1 | 正式核同形状锚点 |
| 6 / 7 | IBDONE16 / LOOP | **底层 C API 支持 chunks+overlap 参数；现有 host 入口硬编码 chunks=1** | 多 chunk 串行 |

缺口一：**单 Chunk 入口**——除 diag 5/6/7 外全部单 chunk；diag 6/7 经现有 host 入口也是 chunks=1。跑 4-chunk 需在诊断 host wrapper 增加 chunks 透传（独立诊断入口改动，不动正式核，属 Gate C 准备工作）。
缺口二：**LOOP 不实现 overlap=1**——源码 `(void)overlap`，每 chunk 顺序 `signal(k)→wait(k)→add(k)`；正式核是 `signal(k)` 后 Add `k−1` 的错峰流水。LOOP 是**串行化隔离探针**，不是正式核等价物；其与正式核的差只能报告为"交叠收益的界"，不能当正式核分段耗时。

## 5. 总结审视

### 5.1 已证实（observed/derived，可复查）

1. **瓶颈定位收窄到融合核内部**：站点级净差 +28.06（r0）/+16.78（r1）ms/步完全由融合核时长差构成；拷贝（contiguous/D2D）净贡献为 0；非目标 AR 两臂无系统差。
2. **融合核比 Stock MM+AR 每站多 ~0.63ms**（p50 2574 vs 1944us r0），且 pipe 计数器显示多出的活动集中在 mte2（+1375us）与 scalar（+686us）通道——**只能作为候选方向（传输/缓存维护/消息轮询），不能换算为阶段耗时**。
3. **Rank1 为慢侧**：每请求多 ~15ms 设备忙时，两臂一致；V2 下其超额大部分在非目标算子。
4. **版本身份链**：prefill 证据的 Host/KVER11/插件身份与当前在位一致；历史差异（W1/W4 `6e9dc2d6`、`2f59dedb` 元数据）已解释并关闭。
5. **计量纪律**：跨 Rank 绝对时间不可用（真实偏斜不可校零）；每 Rank 本地统计与同代配对有效；74ms 异常为请求 0 第 9 站双 Rank 同时单次事件，根因 unknown。

### 5.2 仍缺失（unknown，不推测补全）

- 74ms 异常根因；融合核内部 MM 等待/SDMA/IB/CleanLines/Add 各自的独占或重叠结构；overlap=1 交叠是否真实生效；MTE2/Scalar 计数器到具体行为的映射；设备侧 v11 SO 的直接 SHA；Rank1 非目标算子慢 ~11ms/步（V2）的机理。

### 5.3 Gate C 准入条件与最小取证路线（更新）

**准入（C0，新增本轮结论）**：Gate A 加载身份已核（本报告 §1）；设备空闲、无 v12、KVER11 JSON/SO 在位复验（下次设备会话用 helper 日志或新 Trace 确认 `HybmSdmaOrchEpoch` 唯一实例）；诊断构建与正式库同源同参；`GetSystemCycle` 禁用（恒 0）。

**最小取证路线（无时间戳的受控隔离矩阵，全部短窗口、双 Rank、带校验和）**：

1. **锚点**：正式 V2 全形状（M4096/chunk1024/overlap1）重现 ~2.57ms；同时跑 diag 5（正式核单 chunk 串行）取 4×单 chunk 和作对照。
2. **增量链**：diag 0（MM-only）→ 1（NoIB）→ 3（IB）→ 4（IBDONE）→ 7（LOOP，chunks=4，需 host wrapper 透传小改）——只报告**相邻变体差值与方向**，标注"非正式核分段耗时"。
3. **交叠收益界**：LOOP(chunks=4) − 正式核 = 串行化代价上界；diag5×4 − 正式核 = 另一侧参照。
4. **异常探针**：在矩阵中观察 74ms 是否复现（尤其首请求位）；不复现则记录"未复现"，不追打。
5. **外部观测**：全程 torch_npu profiler 采全核时间线；跨 Rank 只用同代配对与每 Rank 本地统计。
6. **停止条件**：照 Gate B §7 C4（507014/507018、协议状态非零、首调失配、设备占用、误加载 v12、Trace 不完整即停）。

**产出承诺（不超额）**：各变体整核时长分布与增量方向、交叠收益界、74ms 复现性、正确性校验和；**不产出**"SDMA 等待=xxus"式伪精确分段。若隔离矩阵无法把 ~0.63ms 收窄到可行动假设，明确 No-Go，不进入无证据调参。

## 6. 证据索引

| 文件（`evidence/gate_ab_verification_20261022/`） | 内容 | 分级 |
|---|---|---|
| `env_identity.json` | Gate A 现场身份核验（SHA/路径/加载链/残留检查/版本裁定） | observed |
| `gate_ab_audit.json` | 原始 Trace 独立复算全部结论（§2/§3 数字源） | derived |
| `gate_b_capability.json` | 计时能力判定与探针方法学 | observed+derived |
| `raw_probe/getsystemcycle_probe_result.json` | 探针原始回读（6 发射 × 8 核，cycle 全零、magic/acc 正确） | observed |
| `trace_sha256.json` | 服务器原始 Trace 16 个关键 CSV 的 SHA256/大小/mtime | observed |
| `scripts/` | 审计脚本 3 份 + 探针源码/runner（可复现全部 derived） | — |

原始 Trace（`/tmp/opencode/v2_prefill_captures/{stock,v2}_prefill_v2attrib/`，4×~125MB）仅存服务器。

## 7. 下一步建议

1. **优先**：按 §5.3 执行 Gate C 受控隔离矩阵（先补诊断 host wrapper 的 chunks 透传，独立分支、不触正式核）。
2. 并行推进 MTE2/Scalar 计数器的**行为级解释**实验设计（微基准触发已知行为比对计数器形态），替代直接计时。
3. 74ms 异常挂起为独立观察项，在 Gate C 矩阵中顺带观测，不单独投入长压测。
4. Rank1 非目标算子偏斜（V2 下 ~11ms/步）值得单列一次离线分析（现有 Trace 即可），判断是否与融合核争用资源。
5. 记录未关闭的 G1-D Gate E P0（通信邮件可见性/写入缺失分支）：性能取证不等于稳定性结案。
