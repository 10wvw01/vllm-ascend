# OpenCode执行单：V3依赖同步、私有编译部署和一次快速验证

本单由研发侧制定。OpenCode负责执行、执行中的BUG定位与修复、实验结果汇总、数据及证据采集，
并将修复代码和实验记录提交远程工作分支。研发侧负责方案、算法与协议决策、结果审查和后续推进。
编译或设备门禁失败先停止当前轮并保存现场；允许修复实现BUG后重编、重新验证，
不得通过删除检查、放宽精度或切回旧实现使测试通过。涉及方案变更的事项交回研发侧决策。

## 0. BUG修复与远程交付职责

- OpenCode可修复本方案范围内的编译、接口适配、资源管理、同步及数值实现BUG，并解释根因。
  不自行改变SDMA ADD归约方案、通信协议设计、性能目标、采样口径或精度标准。
- 固定SHA是首轮起点；修复后记录新HEAD/tree、diff和构建manifest，禁止仍以旧SHA标记修复产物。
  V3修复提交到本次V3工作分支；推送前检查远程HEAD，有并发更新先核对，不强推或覆盖他人提交。
- 若确需修复MemFabric依赖，从锁定提交创建独立工作分支，提交到`10wvw01/memfabric_hybrid`，
  不直接改`dev-v1.2.0`或权威上游。同步更新V3依赖锁定、来源记录和私有SDK证据，重新构建；
  不禁用commit/tree/哈希校验。方案或协议变更先回报研发侧。
- 每轮使用新的构建/结果目录，保留失败轮。修复后重跑受影响CPU检查和构建；
  新设备会话必须重新通过全部前置门禁，不能在已失败的池或Session中重发ADD继续。
  健康告警、占用冲突或独占窗口丢失仍须STOP，不能靠代码修改绕过守护器。
- 将实验配置、复现命令、根因与修复、逐轮结果和证据索引提交V3仓库的
  `csrc/_310P/memfabric_mm_ar_v3/experiments/<run-id>/`。
  小体积原始JSON/日志可随提交保存；大体积证据使用仓库附件或已有制品存储，
  索引记录可获取地址、大小与SHA256，不能只提供服务器临时路径。
  不提交虚拟环境、SDK/构建二进制、模型、凭据或无关环境数据。
- 最终回传远程分支、完整提交SHA、修复摘要、实验汇总及原始证据位置，不只报告PASS/FAIL。

## 1. 哪些仓库需要更新

| 仓库 | 作用 | 分支/锁定值 | 服务器动作 |
|---|---|---|---|
| `gitcode.com/GDD_ESCC/memfabric_hybrid` | 权威上游，只读来源 | `dev-v1.2.0`；`ba3e7f42873b6f2a4e8d118c03a14d1f43d39ab4` | 不修改，不必再克隆 |
| `github.com/10wvw01/memfabric_hybrid` | 已同步的SDK构建源码 | `dev-v1.2.0`；`b9d46f97955b11a8b7a892e9fd8d516e66346a65` | 新克隆，detached HEAD固定提交 |
| `github.com/10wvw01/vllm-ascend` | V3算子、适配器和执行工具 | `work/310p-mm-ar-v3-sdma-add-20261029`；完整SHA以本次交接消息为准 | 新克隆，detached HEAD固定该SHA |

不更新vLLM主框架、现有Serving worktree、已安装vllm-ascend wheel、模型权重或DFlash。
不更新/提交`HaonanWu1/vllm-ascend`。不把V3 merge进研发主线或切换为生产默认。

GitHub镜像使用保留现有历史的快照提交，所以它的commit SHA不同于上游。
两边完整tree均为`59f1fb6f1e4b8e4e8fdc8128c74d605bc8f35103`，代码逐字节一致。
此次新增SetAccess物理号修复，ADD接口、公共设备头及KVER28不变。来源证明见[UPSTREAM_SYNC.json](UPSTREAM_SYNC.json)。

## 2. 冻结环境与创建隔离目录

现有记录中的验证环境是aarch64 / CANN9.1.0 / Python3.12 / torch2.10+torch_npu2.10。
使用同一套环境。下列Python和CANN路径若不存在，先STOP回传实际安装信息，不自行换成另一版本。
CPU编译不需要占用NPU，可以先完成；任何设备测试都必须等到双Die独占窗口。

在Shell先设置本次交接给出的完整V3提交号：

```bash
export V3_EXPECTED_SHA=填入交接消息中的完整40位V3提交SHA
```

然后执行下列命令。根目录必须是新目录，已存在则不覆盖；不得使用共享研发worktree。

```bash
set -euo pipefail
V3_RUN_ROOT=/tmp/opencode/310p-v3-20261029-001
V3_BASE_PYTHON=/usr/local/python3.12.13/bin/python3
V3_CANN=/usr/local/Ascend/cann-9.1.0
test -x "$V3_BASE_PYTHON"
test -f "$V3_CANN/set_env.sh"
test "${#V3_EXPECTED_SHA}" -eq 40
mkdir -p /tmp/opencode
mkdir "$V3_RUN_ROOT"
set +u
source "$V3_CANN/set_env.sh"
set -u
"$V3_BASE_PYTHON" -m venv --system-site-packages "$V3_RUN_ROOT/venv"
V3_PYTHON="$V3_RUN_ROOT/venv/bin/python"
if ! "$V3_PYTHON" -c 'import pytest' >/dev/null 2>&1; then
  "$V3_PYTHON" -m pip install pytest==9.2.0
fi

git clone --single-branch --branch work/310p-mm-ar-v3-sdma-add-20261029 \
  https://github.com/10wvw01/vllm-ascend.git "$V3_RUN_ROOT/vllm-ascend"
git -C "$V3_RUN_ROOT/vllm-ascend" checkout --detach "$V3_EXPECTED_SHA"
test "$(git -C "$V3_RUN_ROOT/vllm-ascend" rev-parse HEAD)" = "$V3_EXPECTED_SHA"

git clone --single-branch --branch dev-v1.2.0 \
  https://github.com/10wvw01/memfabric_hybrid.git "$V3_RUN_ROOT/memfabric-src"
git -C "$V3_RUN_ROOT/memfabric-src" checkout --detach b9d46f97955b11a8b7a892e9fd8d516e66346a65
test "$(git -C "$V3_RUN_ROOT/memfabric-src" rev-parse HEAD^{tree})" = \
  59f1fb6f1e4b8e4e8fdc8128c74d605bc8f35103

cd "$V3_RUN_ROOT/vllm-ascend"
"$V3_PYTHON" -c 'import torch, torch_npu; print(torch.__version__, torch_npu.__version__)' \
  > "$V3_RUN_ROOT/python-environment.txt"
"$V3_PYTHON" -m pytest -q csrc/_310P/memfabric_mm_ar_v3/test \
  csrc/_310P/memfabric_mm_ar_v2/test/ut --import-mode=importlib \
  > "$V3_RUN_ROOT/cpu-tests.log" 2>&1
```

私有venv复用现有torch/torch_npu，只有缺失的pytest安装到本次venv；不升级系统Python或推理依赖。
首轮构建前两个克隆内均不应有tracked改动。执行中修复须提交新版本并重编，按第0节记录证据。

## 3. 先构建完整私有MemFabric SDK

执行专用脚本；它核验镜像commit/tree、拒绝已有build/output的源目录，
调用上游`build_and_pack_run.sh`，关闭Python、测试包及HCOM，最多8个编译任务。

```bash
"$V3_PYTHON" csrc/_310P/memfabric_mm_ar_v3/tools/prepare_sdk.py \
  --source "$V3_RUN_ROOT/memfabric-src" \
  --out "$V3_RUN_ROOT/memfabric-sdk" \
  --cann "$V3_CANN" --jobs 8
```

脚本不执行run安装器。上游打包器未复制AICPU目录，因此不能仅安装run包后就假设SDK完整。
脚本从同次真实构建中复制公共头、Host库及AICPU三件套，记录每个源/目标文件SHA。

| 私有产物 | 位置 |
|---|---|
| 公共头 | `memfabric-sdk/include/smem/{host,device}` |
| Host动态库 | `memfabric-sdk/aarch64-linux/lib64` |
| AICPU设备源SO | `memfabric-sdk/aarch64-linux/hybm/aicpu_kernel/libmf_sdma_orch.so` |
| CUST迁移JSON | 同目录`libmf_sdma_orch.json` |
| KVER28 launch JSON | 同目录`libmf_sdma_orch_v28.json` |
| SDK构建/复制证据 | `memfabric-sdk/memfabric_build.log`、`sdk_manifest.json`、`version.info` |

不用旧的KVER11补丁SO，不复制系统安装中的同名库，不临时修改SetAccess源码。
上游新修复已经包含在这次锁定的源码中。AICPU hcc编译器缺失或任一三件套缺失都STOP。

## 4. 再构建私有V3 Host/Device库

```bash
"$V3_PYTHON" csrc/_310P/memfabric_mm_ar_v3/tools/build.py \
  --memfabric-source "$V3_RUN_ROOT/memfabric-src" \
  --memfabric-home "$V3_RUN_ROOT/memfabric-sdk" \
  --cann "$V3_CANN" \
  --out "$V3_RUN_ROOT/v3-build"
```

生成`libmf310p_v3_device.so`、`libmf310p_v3_host.so`、两份编译日志和`manifest.json`。
不重编整个vllm-ascend扩展，不执行pip editable安装；Python薄适配直接从本worktree加载。
此阶段只编译Host/Device文件，不创建MemFabric池、不执行设备Kernel。

## 5. 设备部署与快速测试

本轮固定优先使用物理卡38144的dev4/5。先取得连续独占窗口并记录健康/占用。
若卡忙或Alarm则STOP回传，不自行切卡或等待后无限重试。不要杀其他租户。

```bash
npu-smi info > "$V3_RUN_ROOT/npu-before.txt"
npu-smi info -t health -i 38144 > "$V3_RUN_ROOT/health-before.txt"
"$V3_PYTHON" csrc/_310P/memfabric_mm_ar_v3/tools/quick.py start \
  --manifest "$V3_RUN_ROOT/v3-build/manifest.json" \
  --card 38144 --devices 4,5 \
  --out "$V3_RUN_ROOT/results" \
  --lease-note '本轮已协调的38144双Die独占窗口'
```

只有取得真实独占窗口后才能使用上面的lease-note，不能把文字参数当成租约证明。
脚本以独立session启动，设备阶段默认最多600秒；OpenCode轮询结果，不用前台长命令等待：

```bash
"$V3_PYTHON" csrc/_310P/memfabric_mm_ar_v3/tools/quick.py status \
  --out "$V3_RUN_ROOT/results"
```

每次观察间隔约10～20秒，直到result.json出现。脚本每2秒检查NPU占用和健康，异常仅清理自有PGID。
两个Rank直接选择物理DeviceID并经CANN核验，不设置可见设备重映射；父Shell环境不变。

设备部署归MemFabric自动初始化负责：**首次获准建池时**自动按需迁移并加载v28。
不预部署四卡，不手动运行deployer/mini_kfc，不清理CP1目录，不替换/删除其他版本设备文件。
三件套源SO与launch JSON的SHA在manifest中；实际Host/Device直调库通过进程maps记录。
CP1实际文件若未取得单独读取校验，不把其SHA写成已验证；本轮功能门依靠ADD真实数值继续核验。

单次会话固定顺序：IB → FP16 ADD真实数值 → 通信协议/回绕 → 融合/尾块正确性 → V3短测。
任一关失败停止后续阶段。默认只测V3流水的M2048/4096、group1024/2048、两轮各30样本。
不启动V2/Stock性能进程、不初始化HCCL、不跑模型/Profiler/Confirm/TTFT。
建池前少量MM+CPU Gloo仅用于数值参考，不计为Stock性能重测。

## 6. 回传与收尾

必须回传以下原始文件，不能只粘贴“PASS”或手算加速比：

1. 两个仓库HEAD/tree、`UPSTREAM_SYNC.json`、Python版本和CPU日志。
2. SDK的`sdk_manifest.json`、`memfabric_build.log`和V3的`manifest.json`、`build_*.log`。
3. `results/result.json`、`rank0.json`、`rank1.json`、全部`gate_*`/`timing_*`。
4. `REPORT.md`、`historical_comparison.json`、`console.log`、`lease_guard.jsonl`。
5. 失败时附`stop_rank*.json`、`lease_guard_stop.npu.txt`及结束时的`npu-smi info`。

报告只回答V3算子相对历史V2/真实NZ Stock的方向性收益，不升级历史漂移或Screen结论。
KVER28仍有占位自拷贝源码差异，本次不能签长稳或发布GO。
不在没有定位或修复的情况下反复盲跑，不删失败样本、不切旧SDK、不调整包长，
不把任何库拷回共享环境。完成BUG修复后允许按第0节创建新轮次验证并提交远程。
成功后正常退出自有Session即可；失败按守护器STOP保留目录，不用清KFC“恢复环境”。
