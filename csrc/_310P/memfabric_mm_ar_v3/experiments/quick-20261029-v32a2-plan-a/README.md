# V3.2 方案A2（pool-return）私有编译、准入与快速验证记录（2026-10-29）

执行：OpenCode（按 `OPENCODE_EXECUTION.md`、`V32_ADMISSION.md` 固定顺序，用户委托一次性完成）。
范围：隔离环境 → CPU 合同 → 私有 MemFabric SDK → 私有 V3 Host/Device 库 → TorchNPU Storage 适配冻结 →
G0 与双 Die 独占 → `--plan-a` 六门禁 → M2048/4096 × chunk512/1024/2048 六配置两轮短测。
仅测试方案 A2；不实现/不测试方案 B；不重跑 V2/Stock 历史臂；不跑模型/TTFT/Graph/Confirm/重型Profiler。

## 总结论

| 阶段 | 结果 |
|---|---|
| CPU 合同（V3+V2 UT，起点 213cfcedb） | **102 passed** |
| 私有 MemFabric SDK（db38d75e，fix 分支强制 helper） | PASS；AICPU 三件套 + `orch_deploy_helper` 齐全 |
| 私有 V3 Host/Device SO（BiSheng dav-2002） | PASS（`device_compile: PASS`） |
| TorchNPU Storage 适配（冻结） | **两轮编译 BUG 修复后 PASS**（见下），SHA256 见指纹表 |
| G0 + 38144 双 Die 连续独占 | PASS；全程无 STOP，仅一次合法部署窗口瞬态（7 秒自愈） |
| Plan-A 六门禁（双 Rank） | **全 PASS**：ib → add → output_lifetime（含切片别名/双槽/单边释放/busy close）→ protocol（含单 chunk 排空与环回绕）→ fused（含下游 ND 消费）→ tail |
| 六配置两轮短测（4 预热 + 30 样本 ×2 轮，4MiB 包） | 完成，`PASS_QUICK_SCREEN_ONLY`，跨轮漂移 1.17%–2.69% |
| 相对历史基线 | **无收益**：最佳配置仍慢于历史 V3.1/V2 E3-G/真实 NZ Stock，详见性能节 |

## 指纹

| 项 | 值 |
|---|---|
| vllm-ascend 分支 | `work/310p-mm-ar-v3-sdma-add-20261029` |
| vllm-ascend 起点 | `213cfcedb41c6465e525dd184b7c5f74d3d4fba9`（交接锁定） |
| vllm-ascend 执行后 HEAD | `0f638d5112de0b11b656f296ff6fb3c6e81d13d3`（含两个执行 BUG 修复，见 git log） |
| MemFabric 分支/提交 | `fix/310p-v3-orch-deploy-helper-20261029` @ `db38d75e8485d31451302219eb7a1bfeb1dd107c`（tree `4fe624cabbf62d6475f79cc63bd19dcbedacb0e7`，上游 `ba3e7f42`，未修改） |
| 私有 SDK 助手 | `orch_deploy_helper` SHA256 `849f93c0b4a96877d3e839ae7ceec4b6c479a8de331213f0d7a270c1198bd310`（与生产验证版逐字节一致），25320 B |
| V3 Host SO | `9c719a631390d5a5…`（manifest 全量），27608 B |
| V3 Device SO | `141ac6065e835332…`，1326144 B |
| TorchNPU Storage 适配 SO | `2d0ac4d3d7142fc9…`，1477152 B |
| KVER / launch JSON | 28；`libmf_sdma_orch_v28.json` SHA256 `b34034cceccdbaee…` |
| 公共头 | `smem_shm_aicore_base_sdma.h` / `smem_shm_sdma_layout.h` 与锁定源逐字节一致（`dependency.public_headers`） |
| 设备/环境 | 38144 双 Die dev4/5（物理直选，无可见设备重映射）；aarch64 / CANN 9.1.0 / Python 3.12.13 / torch 2.10.0+cpu / torch_npu 2.10.0.post4 / Ascend310P3 |
| 进程内实际装载 | 双 Rank `/proc/self/maps` 仅 SDK `libmf_hybm_core.so`、`libmf_smem.so` 与私有 `libmf310p_v3_host/device.so`，指纹与 manifest 一致；运行前后装载集不变 |

构建产物与 SDK 二进制不入库（治理要求）。完整 SHA256 清单见 `manifests/v3_build_manifest.json`
（含 `installed_files`/`public_headers`/`artifacts`/`tensor_adapter`）与 `manifests/sdk_manifest.json`。
产物保留在服务器 `/tmp/opencode/310p-v32a2-20261029-001/`（v3-build-r3/、v3-tensor-r3/、memfabric-sdk/）。

## 轮次时间线与执行 BUG 记录

两个 BUG 均为**编译/接口适配类**，由 OpenCode 按执行单第 0 节修复并提交新 SHA；未改 SDMA ADD 方案、
协议、采样口径或精度标准，未修改 MemFabric。

### B1（`r1-tensor-include-stop/`）：Storage 适配编译缺 CANN ACL 头路径

- 现象：`build_tensor.py` 编译 `pool_tensor.cpp` 时 `torch_npu/.../acl/acl_base.h:14` 报
  `fatal error: acl/acl_base_rt.h: No such file or directory`。
- 根因：torch_npu 捆绑 ACL 头（`include/third_party/acl/inc`）不是自含集合——`NPUStorageImpl.h`
  以 `third_party/acl/inc/...` 前缀引用 `acl_rt.h`，但 `acl_base.h` 内部以 `acl/acl_base_rt.h`
  与 `error_codes/…` 的 CANN 相对前缀再引用，只有 CANN 安装 include（
  `/usr/local/Ascend/cann-9.1.0/aarch64-linux/include`）能解析；`load_tensor_module` 只传了
  torch_npu 两条 include。
- 修复：`263ae4bc8` —— `load_tensor_module` 增加 `cann_home`（manifest 传入或
  `ASCEND_HOME_PATH` 兜底），追加 CANN include 并显式检查 `acl/acl_base_rt.h` 存在。

### B2（`r2-tensor-setdesc-stop/`）：`StorageDescHelper::SetDesc` 未从 libtorch_npu 导出

- 现象：B1 修复后编译、链接通过，但 import 时
  `undefined symbol: _ZN6at_npu6native17StorageDescHelper7SetDescERN2at6TensorERKN3c108ArrayRefIlEES9_`。
- 根因：锁定版 `libtorch_npu.so`（stripped）只导出 9148 个动态符号，`StorageDescHelper` 全类、
  `NPUStorageImpl` 构造与 `make_npu_storage_impl` 均**不在动态符号表**，外部扩展无法链接调用。
- 修复：`0f638d511` —— `pool_tensor.cpp` 改用 torch 2.10 的 `c10::make_storage_impl`：
  torch_npu 在 import 时经 `REGISTER_PRIVATEUSE1_BACKEND` → `c10::SetStorageImplCreate(PrivateUse1,
  &make_npu_storage_impl)` 注册了设备级 StorageImpl 工厂，`make_storage_impl` 将调用方 DataPtr
  原样透传给该工厂（torch_npu v2.10.0 开源源码 `NPUStorageImpl.cpp` 证实：DataPtr 非空即不重新分配），
  因此得到真正的 `NPUStorageImpl`；随后按开源 `StorageDescHelper::SetDesc(dst, size, strides)`
  的语义（`StorageDescHelper.cpp` / `FormatHelper.cpp` / `InferFormat.cpp`）直接填充公开成员
  `npu_desc_`：2-D 尺寸 → `GuessBaseFormat`/`GuessFormatUnit` 均为 `ACL_FORMAT_ND`，
  `InferShapeofND` 保持 storage sizes 不变；`base_sizes_={rows,2048}`、`base_strides_={2048,1}`、
  `storage_sizes_={rows,2048}`、`origin_format_=npu_format_=ACL_FORMAT_ND`、`data_type_=fp16`。
  TensorImpl 用 `at::detail::make_tensor<TensorImpl>(Storage, DispatchKeySet(PrivateUse1), fp16)` +
  `set_sizes_contiguous` 构造。**没有退回普通 from_blob**（普通 `from_blob`/`TensorMaker::make_tensor`
  经反汇编确认只创建普通 `c10::StorageImpl`，不含 NPU 描述，仍不可用）。
- 设备侧验证：fused 门禁中的下游 ND 消费（对输出执行真实 `torch.add`）+ 输出生命周期门禁
  （切片别名持有、双槽、busy close）在锁定版上全部 PASS，证明描述字段填充正确。

每轮均新建构建/结果目录（v3-build → r2 → r3；v3-tensor → r2 → r3），失败轮证据保留于本目录；
修复提交后 CPU 102 项重跑 PASS 再重编。

## 设备会话（`pass/`）

- 启动：`tools/quick.py start --plan-a --manifest …/v3-build-r3/manifest.json --card 38144 --devices 4,5
  --out …/results`，独立 session detached，600s 上限；守护器 2s 轮询健康/占用/跨卡。
- 独占与部署窗口：PREFLIGHT 双 Die 空闲 OK；池创建窗口内出现 3 个目标卡进程
  （两 Rank + 一 helper/编译基础设施形态），7 秒自愈（`DEPLOY_WINDOW_TRANSIENT→HEALED`），
  全程 54 次 `LEASE_SAMPLE` 无 STOP，`STAGE_EXIT 0`；结束时双 Die OK、无残留进程。
- 门禁顺序与结果（两个 Rank 各自落盘 `gate_*_rank{0,1}.json`）：
  1. `ib`：SDMA IB 探测 PASS。
  2. `add`：4096/1MiB 两档裸缓冲 + FP16 边界值（±65504、2^-24、0.333251953125 等）真实 ADD，
     逐位模式比对 PASS。
  3. `output_lifetime`：pool 首包（Storage 包装/元数据）、第二活跃槽、rank1 释放别名而 rank0
     持有时双方分配被拒、单边持槽下集体 close 返回 busy、全部释放后正常关闭 PASS。
  4. `protocol`：group512/1024/2048 × rows=group/2048/4096（含单 chunk 排空）×8 次 + group512
     48 波环回绕（>1024 到达环）PASS。
  5. `fused`：全 chunk 融合 MM+ADD+Gather，输出直接被 `torch.add` 下游 ND 消费 PASS。
  6. `tail`：2051 行奇数尾块（显式补零、输出切片、无隐藏 copy）PASS。
- 数值口径：全部与 Gloo CPU 参考按 FP16 位模式逐元素比对；宿主进程仅初始化 Gloo，未初始化 HCCL。

## 性能结果（六配置 × 两轮，4MiB 包，完整接口窗口）

计时窗口包含：输出槽 acquire（含原流同步与控制 allgather）→ Storage 包装 → 双端 prepare（Host
allgather）→ Kernel → 流完成；同时记录事件时延与 Host Wall。**无 Kernel-only 口径数字**。
统计口径：同一样本编号先取双 Rank max，再算分位数（`pass/analyze_timing.py` 可复算）。

事件时延（µs，双 Rank max 后分位）：

| M | chunk | R0 p50/p90/p99 | R1 p50/p90/p99 | 跨轮 p50 漂移 | Wall p50（R0/R1） |
|---|---|---|---|---|---|
| 2048 | 512 | 2297.1 / 2419.6 / 2543.3 | 2332.1 / 2416.2 / 2422.2 | 1.52% | 2393.5 / 2421.5 |
| 2048 | 1024 | 2049.5 / 2151.8 / 2158.7 | 2018.7 / 2095.6 / 2145.4 | 1.53% | 2146.1 / 2107.3 |
| 2048 | 2048 | 2034.7 / 2108.3 / 2170.7 | 2089.4 / 2183.5 / 2255.4 | 2.69% | 2121.1 / 2177.7 |
| 4096 | 512 | 3365.6 / 3491.3 / 3655.0 | 3405.1 / 3614.1 / 3685.2 | 1.17% | 3458.3 / 3501.2 |
| 4096 | 1024 | 2727.1 / 2889.3 / 2978.4 | 2786.5 / 2881.5 / 2894.7 | 2.18% | 2823.6 / 2865.2 |
| 4096 | 2048 | 2756.8 / 2846.6 / 2903.3 | 2797.0 / 2858.7 / 2899.2 | 1.46% | 2848.6 / 2893.4 |

- 最佳配置：M2048 → chunk1024（2018.7µs）；M4096 → chunk1024（2727.1µs）。
- Wall p50 普遍比事件 p50 高 3–5%，为完整接口的 Host 侧开销体现。
- 逐样本原始数据：`pass/timing_r{0,1}_m{M}_g{G}_rank{0,1}.json`（24 份）。

## 历史对照与限制

`pass/REPORT.md` 为工具自动生成的全量对照表（E3/E4/R2_screen/V3.1 各臂）。要点（以最佳配置 vs
历史最佳）：

| 对照 | M2048 | M4096 |
|---|---|---|
| V3.2-A2 最佳 p50 | 2018.7 | 2727.1 |
| 历史 V3.1 最佳 p50 | 1655.9（g1024） | 2516.1（g2048） |
| V3.2 相对 V3.1 | **慢 21.9%** | **慢 8.4%** |
| 历史 V2 E3-G p50 | 1451.0 | 2119.5 |
| V3.2 相对 V2 E3-G | **慢 39.1%** | **慢 28.7%** |
| 真实 NZ Stock p50 | 1078.4 | 2022.6 |
| V3.2 相对 Stock | **慢 87.2%** | **慢 34.8%** |

结论与限制：

1. **方案 A2 未达到性能目标，方向性上比 V3.1 更慢**。差距主因是准入单已预告的逐波接口成本：
   每波 acquire 的原流同步、acquire/prepare 两次 Host 控制协商、Storage 包装与双端就绪 barrier
   全部计入完整窗口——这是 A2 为换取输出生命周期安全（输出即池内 Storage、别名持有、busy close）
   支付的成本；V3.1 数字不含这些输出管理成本。本轮未做独立归因轮（超出授权范围），上述分解
   依据为时延结构（Wall−Event 差）与 V3.1/V3.2 测量窗口差异，非实测分解。
2. 跨批限制：历史 V3.1/V2/Stock 数字来自既往批次（环境同卡同 CANN 但非同批），漂移与批次差异
   依治理判为方向性对照，**不签同批正式收益结论**。
3. KVER28 仍有占位自拷贝源码差异（manifest `findings` 提示），本轮不能签长稳或发布 GO；
   仅一次短测会话，无长稳/Graph/模型传导结论。
4. Serving 默认不变；本实验不改变任何共享安装、CP1 或生产路由。

## 复现命令

```bash
# 0) 固定提交（修复后 HEAD）
git clone --single-branch --branch work/310p-mm-ar-v3-sdma-add-20261029 \
  git@github.com:10wvw01/vllm-ascend.git && git checkout 0f638d511
git clone --single-branch --branch fix/310p-v3-orch-deploy-helper-20261029 \
  git@github.com:10wvw01/memfabric_hybrid.git && git checkout db38d75e

# 1) CPU 合同
python -m pytest -q csrc/_310P/memfabric_mm_ar_v3/test \
  csrc/_310P/memfabric_mm_ar_v2/test/ut --import-mode=importlib   # 102 passed

# 2) 私有 SDK + V3 库 + Tensor 适配（新目录）
python csrc/_310P/memfabric_mm_ar_v3/tools/prepare_sdk.py \
  --source <memfabric-src> --out <sdk> --cann /usr/local/Ascend/cann-9.1.0 --jobs 8
python csrc/_310P/memfabric_mm_ar_v3/tools/build.py \
  --memfabric-source <memfabric-src> --memfabric-home <sdk> \
  --cann /usr/local/Ascend/cann-9.1.0 --out <build>
python csrc/_310P/memfabric_mm_ar_v3/tools/build_tensor.py \
  --manifest <build>/manifest.json --out <tensor-build>

# 3) 设备短测（先取得 38144 双 Die 独占窗口）
python csrc/_310P/memfabric_mm_ar_v3/tools/quick.py start --plan-a \
  --manifest <build>/manifest.json --card 38144 --devices 4,5 --out <fresh-result-dir> \
  --lease-note '已协调的38144双Die独占窗口'
python csrc/_310P/memfabric_mm_ar_v3/tools/quick.py status --out <fresh-result-dir>
```

## 证据索引

| 内容 | 位置 |
|---|---|
| 运行结果（result/REPORT/g0/launch/rank×2/historical_comparison） | `pass/` |
| 六门禁双 Rank JSON（12 份） | `pass/gate_*.json` |
| 逐样本计时（24 份，event+wall） | `pass/timing_*.json` |
| 配对统计与脚本 | `pass/perf_summary_paired_rank_max.json`、`pass/analyze_timing.py` |
| 守护器全程日志 / 控制台 / 会话内 CPU | `pass/lease_guard.jsonl`、`pass/console.log`、`pass/cpu.log` |
| V3 构建身份（manifest 含全部 SO/头/helper SHA256 与命令） | `manifests/v3_build_manifest.json` |
| SDK 构建（sdk_manifest/version.info/构建日志） | `manifests/` |
| 失败轮 B1（ACL include） | `r1-tensor-include-stop/` |
| 失败轮 B2（SetDesc 未导出） | `r2-tensor-setdesc-stop/` |
| 环境/健康/前后 npu-smi | `env/` |
| 性能瓶颈可视化分析（5图，纯SVG自包含） | `bottleneck_analysis.html`（`gen_bottleneck_html.py` 生成） |

## L0+L1 归因轮（`attribution-l0l1/`，2026-10-29 追加）

授权后同日执行。工具：`tools/attribution.py` + `test/attribution_worker.py`（提交 `b0bdb2cba`），
构建 `v3-build-r5`（HEAD `f8b5b30a9`，修复部署窗口声明顺序后）。同会话四臂：IB探针 →
裸ADD传输(1/2/4MiB) → partial协议臂 → fused臂（含逐波Host相位记录）。
统计：配对双Rank max后p50；Host相位另取各Rank p50的min作无宿主抖动基线。
宿主背景：loadavg≈13（192核，其他租户活跃），已记入 `attr-host-load*.txt`。

### 首轮失败（`r1-guard-window-stop/`）

worker把部署窗口标记写在会话创建之后，建池fork的torch_npu编译基础设施子进程
（目标卡4进程>2且无窗口）被守护器正确STOP(97)。修复 `f8b5b30a9`：标记先于任何设备算子。

### L1：每波Host接口分解（基线=Rank-min p50）

| 相位 | 基线µs | 抖动峰值µs | 内容 |
|---|---|---|---|
| acquire | ~250 | ~460 | 流同步 + 控制allgather + 槽交集 |
| wrap | ~40 | ~48 | NPUStorageImpl包装 |
| prepare | ~180 | ~400 | 控制allgather（双方就绪门） |
| run | ~80 | ~89 | epoch清零 + 入队 |
| **合计** | **~550** | **~890** | |

两次控制allgather合计~430µs是Host接口的主成本；宿主负载抖动可再抬升150~350µs
（loadavg 13下rank1在部分配置段膨胀，建议Confirm级测量绑核+错峰）。

### L0：差分与传输原语

- **MM≈拷入**：Δ(fused−partial) = MM − copy_in：M2048为+645/+279/+88µs（g512/1024/2048），
  M4096为+184/−7/+119µs。8核MM的每行成本与MTE整块拷入同量级——**MM不是瓶颈**，
  g2048下MM相对拷入几乎零边际。
- **裸ADD传输**（含3×MTE+2次控制往返+2次quiet+内核固定开销）：1MiB=613 / 2MiB=708 /
  4MiB=923µs → 固定成本~500µs/次 + 边际~103µs/MiB。小包传递被控制往返与quiet的固定
  延迟支配——定量证实复核文档"减少包与同步"的方向。
- **IB**：每三次核间交换14.3µs（~4.8µs/次），可忽略。

### 综合分解（M4096 g1024，fused p50 2670µs，同批）

Host接口 ~570µs（21%）+ 设备侧 ~2100µs（79%）。设备侧=MM(≈copy_in等价) +
R/G传输(~103µs/MiB量级的MTE+SDMA混合边际) + 每组READY往返/quiet固定延迟。
主轮fused 2727µs与本轮2670µs差2.1%，同批一致性良好。

### 去向

1. **砍每波双allgather**（~430µs基线）：批量预约/设备侧标志/随内核捎带——最大单项；
2. 控制邮件往返~250µs级固定延迟是邮箱协议的结构成本，与研发侧"减包减quiet"重排同向；
3. MM与IB无需优化。以上为归因结论，不构成性能GO；未做L2内核内分段（另行授权）。
