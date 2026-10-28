# V3 融合算子编译部署、正确性门禁与快速性能测试记录（2026-10-29）

执行：OpenCode（按 `OPENCODE_EXECUTION.md` 与研发侧逐步指令）。
范围：隔离环境 → CPU 检查 → 私有 MemFabric SDK → 私有 V3 库 → 设备五关门禁 → V3-only 短测。
只测 V3；历史对照读取既有 E3/E4/R2_screen 数据，不重跑旧基线，不跑模型/TTFT。

## 总结论

| 阶段 | 结果 |
|---|---|
| CPU 检查（V3+V2 UT） | **87 passed**（修复轮后；首轮 83） |
| 私有 MemFabric SDK（fix 分支 db38d75e） | PASS，AICPU 三件套 + `orch_deploy_helper` 入 `lib64` |
| 私有 V3 Host/Device 库 | PASS（`device_compile: PASS`） |
| 纯 ACL 独立进程通道验证（dev4 首次部署 v28） | PASS（`deployed and verified size=14072`；重复运行 rc=3 幂等） |
| R3 设备五关门禁（38144 dev4/5，TP=2 单卡双Die） | **双 Rank 全 PASS**：ib → add → protocol → fused → tail |
| R3 V3 性能短测 | 4 形状 × 2 轮 × 30 样本完成，`PASS_QUICK_SCREEN_ONLY` |
| 相对历史基线 | **无方向性收益**：V3 全面慢于历史 E3/E4 Stock（真实 NZ）与 V2 各臂（-13%~-76%），详见下文 |

## 指纹

| 项 | 值 |
|---|---|
| vllm-ascend 分支 | `work/310p-mm-ar-v3-sdma-add-20261029` |
| vllm-ascend 起点 | `c89b32a80b58b8199b25c9a57fa678bdd19bf4c6`（交接锁定） |
| vllm-ascend 修复后 HEAD | `f2c3eaf0f…`（本仓提交，见 git log） |
| MemFabric 分支 | `fix/310p-v3-orch-deploy-helper-20261029`（基于同步锁 `b9d46f97`，上游 `ba3e7f42`） |
| MemFabric 修复提交 | `db38d75e8485d31451302219eb7a1bfeb1dd107c`（tree `4fe624cab…`） |
| 私有 SDK 助手 SHA256 | `849f93c0b4a96877d3e839ae7ceec4b6c479a8de331213f0d7a270c1198bd310`（与生产验证版逐字节一致） |
| 环境 | aarch64 / CANN 9.1.0 / Python 3.12.13 / torch 2.10.0+cpu / torch_npu 2.10.0.post4 |

传输适配说明：GitHub HTTPS 在本服务器不可达（TLS 中断），双仓均改用 SSH 克隆同一仓库；
commit/tree/整树校验全部保留并通过。

## 轮次时间线与 BUG 记录

### R1（2026-10-28 17:23，`r1-inprocess-kfc-stop/`）：进程内 KFC 部署链不可用 → rc=-8 STOP

- 现象：`smem_shm_create` → `hybm_create_entity` → `InitTransManager` →
  `sdma orchestrator acquire failed` → V3 create rc=-8，双 Rank 同时失败，未入门禁。
- 直接原因：`MigrateViaCustChannel` 清残留步骤 `rm -f /home/CustAiCpuUser/lib/...` 经
  KFC（`kfc_min.so` 的 system 内核）执行，launch 本身失败 507018 / AICPU errorCode=0x2a。
- 根因（三层）：
  1. 本服务器设备侧 `kfc_min.so` 缺失：2026-09-18 以来全部设备调度器日志
     2263 条 `Get system api from kfc_min.so unsuccessfully`，0 次成功（含 10-23/10-28）。
  2. torch_npu 宿主进程内 KFC 命令不可用且会污染上下文（acd626cc 于 10-18 实测；
     本轮表现为 launch 直接 507018）。
  3. 锁定的上游源（`b9d46f97`）缺少生产安装版（`88d39285` 谱系，
     `fix-310p-kfc-deploy-helper@131ee5ff`）验证过的 `orch_deploy_helper`
     fork+exec 部署机制——V3 首次实机执行才暴露。
- 修复（按研发侧指令）：MemFabric `fix/310p-v3-orch-deploy-helper-20261029`（db38d75e）
  移植 helper 并强制独占：探测/CUST 迁移/复制/校验全部在助手子进程执行，宿主零 KFC；
  助手缺失或失败即报错，无进程内回退、无“假定已部署”；ADD/KVER28/SetAccess 不变。
  V3 侧（cb07896db）重锁依赖：`prepare_sdk.py` 编译打包助手并记 SHA，
  `build.py` 校验存在/可执行/同目录，`quick.py` 冻结其指纹。
- 参照证据：生产基线 10-23 d4_2 会话中纯 ACL helper 内 KFC 真实执行 + CUST 落盘
  13960B（`device-side-log-excerpts.md` 末节）。

### R2（18:02，`r2-guard-helper-window-stop/`）：守护器把自有 helper 误判外部租户 → STOP(97)

- 现象：两 Rank + 两 fork+exec helper 子进程（宿主 PID 3409467/3409468）触
  `unexpected target process count 4>2`，守护器 SIGTERM 全组。
- 根因：E4 谱系守护器先于强制 helper 机制设计；npu-smi 宿主 PID 无法映射回容器
  （`/proc/*/status NSpid` 仅暴露最内层命名空间），无法证明 4 个进程属本会话。
- 修复（f2c3eaf0f）：worker 建池前后声明 `deploy_window_rank*.flag`；守护器仅在
  声明窗口内容忍 ≤max+2 且持续 ≤60s（TRANSIENT/HEALED 事件留痕），超额/持续/未声明
  仍立即 STOP；健康/跨卡/解析检查不变。CPU 87 项含新增窗口语义回归。
- 附注：R2 期间 dev5 的 helper 已被 fork 并执行探测（“kernel missing”路径），其部署
  完成态因 SIGTERM 早于 stdout 缓冲刷出而不可观测；R3 的 probe 命中与 dev5 全部
  门禁通过证明 v28 内核在 dev5 已在位且功能完好。

### Stage-A（17:52，`stage-a-pure-acl-helper/`）：纯 ACL 独立进程通道验证

- dev4 首次部署：`[helper] kfc sync=507018`（探测缺失，纯 ACL 内 KFC 真实执行）→
  `deployed and verified size=14072`（与 SDK 源 SO 逐字节尺寸一致）。
- 重复运行 rc=3（already present，幂等探测路径验证）。

### R3（18:14，`r3-pass/`）：全链路通过

- 守护器：`DEPLOY_WINDOW_TRANSIENT`（4 PID，18:14:55）→ `DEPLOY_WINDOW_HEALED`
  （18:15:02，~7s）→ 50 次正常采样 → `STAGE_EXIT 0`。
- 两 Rank 的 helper 均走 rc=3 快路径（dev4/dev5 内核已在位）→ `TryLoadKernel` 成功。
- 五关门禁双 Rank 全 PASS（FP16 逐位比对、协议/回绕 49mail×24wave、融合/尾块 2051→3072）。

## R3 V3 性能数据（p50，逐样本双 Rank max，µs；漂移=两轮差）

| M | group | V3 p50 | 跨轮漂移 | vs E3/stock(真实NZ) | vs E4a2/stock | vs E3g(V2分块) | vs R2_screen/v2_d4 |
|---|---|---|---|---|---|---|---|
| 2048 | 1024 | 1897.604 | 0.58% | -75.96% | -74.35% | -30.78% | -38.73% |
| 2048 | 2048 | 1709.941 | 1.37% | -58.56% | -57.11% | -17.85% | -25.01% |
| 4096 | 1024 | 3343.041 | 0.28% | -65.29% | -64.06% | -57.73% | -31.62% |
| 4096 | 2048 | 2883.197 | 0.82% | -42.55% | -41.49% | -36.03% | -13.52% |

负值=V3 更慢。**结论：本轮快测未显示 V3 相对历史 V2/真实 NZ Stock 的方向性收益**，
且为跨批对照（不同日、不同进程形态），不能签正式性能判定；group=2048 优于
group=1024（大分组长包优势成立）。完整数据见 `r3-pass/historical_comparison.json`
与 `r3-pass/REPORT.md`。

## 复现命令（服务器 qwen-test，38144 双Die 独占窗口内）

```bash
export V3_RUN_ROOT=/tmp/opencode/310p-v3-20261029-001   # 本轮根目录（证据已随仓库分发）
V3_PYTHON=$V3_RUN_ROOT/venv/bin/python
source /usr/local/Ascend/cann-9.1.0/set_env.sh
git clone --single-branch --branch fix/310p-v3-orch-deploy-helper-20261029 \
  git@github.com:10wvw01/memfabric_hybrid.git $V3_RUN_ROOT/memfabric-src-r2
git -C $V3_RUN_ROOT/memfabric-src-r2 checkout --detach db38d75e8485d31451302219eb7a1bfeb1dd107c
cd <vllm-ascend@work/310p-mm-ar-v3-sdma-add-20261029>
$V3_PYTHON csrc/_310P/memfabric_mm_ar_v3/tools/prepare_sdk.py \
  --source $V3_RUN_ROOT/memfabric-src-r2 --out $V3_RUN_ROOT/memfabric-sdk-r2 \
  --cann /usr/local/Ascend/cann-9.1.0 --jobs 8
$V3_PYTHON csrc/_310P/memfabric_mm_ar_v3/tools/build.py \
  --memfabric-source $V3_RUN_ROOT/memfabric-src-r2 --memfabric-home $V3_RUN_ROOT/memfabric-sdk-r2 \
  --cann /usr/local/Ascend/cann-9.1.0 --out $V3_RUN_ROOT/v3-build-r3
$V3_PYTHON csrc/_310P/memfabric_mm_ar_v3/tools/quick.py start \
  --manifest $V3_RUN_ROOT/v3-build-r3/manifest.json --card 38144 --devices 4,5 \
  --out <新结果目录> --lease-note '<真实独占窗口说明>'
$V3_PYTHON csrc/_310P/memfabric_mm_ar_v3/tools/quick.py status --out <新结果目录>
```

## 证据索引（均在本目录；SHA256 见各 manifest/`sdk_manifest.json`）

| 内容 | 位置 |
|---|---|
| R1 失败现场（result/stop_rank×2/console/g0/lease_guard/npu-before/health-before/npu-after） | `r1-inprocess-kfc-stop/` |
| R2 守护器 STOP 现场（result/g0/console/lease_guard/lease_guard_stop.npu） | `r2-guard-helper-window-stop/` |
| 纯 ACL 助手验证（dev4 两次运行日志 + npu 前后） | `stage-a-pure-acl-helper/` |
| R3 通过现场（result/g0/rank×2/gate×10/timing×16/历史对照/REPORT/console/lease_guard/cpu/npu-after） | `r3-pass/` |
| SDK/V3 构建指纹（sdk_manifest/memfabric_build.log/helper 构建日志/version.info/v3 manifest/build logs/Python 环境/CPU 测试） | `manifests/` |
| AICPU 设备侧日志关键行摘录（服务器 `/root/ascend/log/run/` 不可随仓库分发，此处保留决定性行） | `device-side-log-excerpts.md` |

## 限制与遗留（交研发侧）

1. KVER28 仍有占位自拷贝源码差异（`KVER28_PADDING_SELF_COPY_PRESENT`），本轮不签长稳/发布。
2. R3 为跨批历史对照 + 短测（30样本×2轮）；V3 相对历史基线慢 13%~76%，方向性收益未成立，
   原因归因（新增 ADD+Gather 全量在算子内、1MiB 分包、MTE2/MTE3 排布等）需研发侧决策后续路径。
3. torch 宿主内“完整部署到校验”路径未被本轮完整复现（R3 双 Die 均走 already-present 快路径；
   R2 已验证 fork+exec 与探测、dev5 部署起跑）。如需补证，可在守护窗口内清 CP1 目录后重跑
   ——但该操作违反执行单“不清理 CP1 目录”约束，须研发侧批准后另行安排。
4. 设备侧文件（`/usr/lib64/aicpu_kernels/.../libmf_sdma_orch_v28.so`、
   `/home/CustAiCpuUser/lib/...`）未做逐字节读取校验（执行单已声明该边界，
   功能门由 ADD 真实数值把关）。
