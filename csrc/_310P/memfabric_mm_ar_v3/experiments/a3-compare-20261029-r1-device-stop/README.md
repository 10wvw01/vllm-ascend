# A3/A3-2/A3-tail 同批对照：私有编译完成、设备阶段因38144被占用STOP（2026-10-29 R1）

执行：OpenCode（按 `A3_SEAT_SCHEDULE.md` 同批合同与用户 2026-10-29 一次性委托）。
范围：固定 SHA 隔离 checkout → CPU/治理/源码检查 → 私有 MemFabric SDK → 私有 V3 Host/Device →
TorchNPU Storage 适配冻结 → dav-2002 编译核查 → 设备 G0 预检 → **STOP（独占窗口不成立）**。
未启动 compare_a3 批次，未触发任何设备 Kernel，无设备侧风险。

## 总结论

| 阶段 | 结果 |
|---|---|
| 隔离 checkout（vllm-ascend） | PASS：新 worktree `/home/w00804146/310P/vllm-ascend-a3v3`，detached 分支尖端 = 81780f34（tree 6b94a0b3，与交接一致，全程无 tracked 改动） |
| 隔离 checkout（MemFabric） | PASS：新克隆 SSH，detached db38d75e（tree 4fe624ca，fix 分支锁校验通过） |
| CPU 检查（V3 81 + V2 UT 54） | **135 passed**（与 A3_SEAT_SCHEDULE 记录的 135 项一致） |
| 项目治理 | **38 tests OK**（unittest discover tools/310p_mm_ar）；`project_gate.py --static` PASS |
| Ruff / 源码 | V3 src/tools/test + 三个 ops 适配 **全部通过**；仅历史 experiments 归档脚本与 v2 遗留文件存在 E501 等既有告警（非本轮引入）；`git diff --check` OK |
| 私有 MemFabric SDK | PASS：137 构建步（含 310P sdma orchestrator kernel 编译、AICPU 交叉编译打包）；AICPU 三件套 + `orch_deploy_helper` 入 lib64；helper SHA256 与生产验证版逐字节一致 |
| 私有 V3 Host/Device SO | PASS（BiSheng `--npu-arch=dav-2002`，`device_compile: PASS`） |
| dav-2002 编译核查（三候选） | PASS：详见下文"编译核查" |
| TorchNPU Storage 适配 | PASS：编译并冻结进 manifest（`tensor_adapter`，storage_device_gate NOT_RUN 待设备） |
| 设备 G0（38144 dev4/5） | **STOP**：dev4 被外部租户进程持续占用（19.8GB、51% DDR 带宽），dev5 间歇租户探测；独占窗口不成立。未换卡、未杀租户、未启动任何会话 |
| compare_a3 八会话批次 | **NOT_RUN**（被 G0 STOP 阻断） |

## 指纹

| 项 | 值 |
|---|---|
| vllm-ascend 分支 | `work/310p-mm-ar-v3-sdma-add-20261029` |
| vllm-ascend 起点 | `81780f342e2bfbcf628814d608c1185f984b5fae`（tree `6b94a0b375264d2e0819d13c1261eefd5e13a515`，交接锁定） |
| MemFabric | `fix/310p-v3-orch-deploy-helper-20261029` @ `db38d75e8485d31451302219eb7a1bfeb1dd107c`（tree `4fe624cabbf62d6475f79cc63bd19dcbedacb0e7`，上游 `ba3e7f42`，未修改） |
| 私有 SDK 助手 | `orch_deploy_helper` SHA256 `849f93c0b4a96877d3e839ae7ceec4b6c479a8de331213f0d7a270c1198bd310`（与生产验证版一致） |
| AICPU 源 SO | `libmf_sdma_orch.so` SHA256 `db3903da4617dead67af5847209202d37e25e2f2853769e4c646c10d0ce6a709`（同源构建产出） |
| V3 Host SO | `147f883d3ec3a628…`（完整值见 `manifests/v3_build_manifest.json` `artifacts`） |
| V3 Device SO | `fd6807c94a6d3614…` |
| TorchNPU Storage 适配 SO | `6be0308072673a3f…` |
| 环境 | aarch64 / CANN 9.1.0 / Python 3.12.13（venv --system-site-packages + pytest 9.1.1）/ torch 2.10.0+cpu / torch_npu 2.10.0.post4 / 310P3 |

构建产物与 SDK 二进制不入库（治理要求）。本轮产物保留在服务器运行根
`/tmp/opencode/310p-a3cmp-20261029-001/`（`memfabric-sdk/`、`v3-build/`、`v3-tensor/`，
SHA256 全量登记于 manifest）；该目录为临时路径，任何后续使用必须按上表 SHA 重新校验，
或以本 README 复现命令从锁定的源重新构建（可逐字节复核）。

## 编译核查（三候选 dav-2002）

对照 `A3_SEAT_SCHEDULE.md`"新增kernel需核对dav-2002编译、Matmul注册/UB/同步资源、
64B标志与跨核/SDMA可见性"逐项核对 `src/op_kernel/mf310p_v3_kernel.asc`（81780f34）：

1. **模板实例化**：`tiling512`（V3_TILING basicM64）/`tiling1024`（basicM128）静态 tiling；
   `mf310p_a3_device_run` 按 role（admit 高位 0/1/2）分发
   `A3Kernel<tilingM, M, MM>`（原A3）与 `A3SeatKernel<tilingM, M, MM, TAIL>`
   （A3-2=TAIL false、A3-tail=TAIL true），compute∈{512,1024}×mode∈{MM,copy} 全组合实例化；
   入口对 rows/compute/comm/epoch/packet/slot/mode/role 参数边界校验，非法返回 -1。
2. **Matmul 注册**：`REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &runtime)` +
   `mm.SetLocalWorkspace(mmBuf)` + `SetTensorA/B/SetOrgShape/IterateAll`；
   A3-2（专职通信）Core0 在 Matmul 注册与计算缓冲初始化**之前**走独立分支提前 return，
   满足"Core0在Matmul注册前走独立分支"的席位合同。
3. **UB/NZ2ND 临时空间**：`pipe.InitBuffer(mmBuf, TOTAL_UB_SIZE - 2*kCopyBytes - 32)`，
   `static_assert(TOTAL_UB_SIZE > 2*kCopyBytes + 32 + 128*256*2*2, "A3 seat UB budget")`，
   显式保留 basicM128 Matmul NZ2ND 临时预算；syncBuf 独立 32B。
4. **双 MTE 缓冲**：`pipe.InitBuffer(copyBuf, 2*kCopyBytes)`（两个 16KiB 槽）+
   `FetchEventID(MTE2_MTE3/MTE3_MTE2)` read0/read1、write0/write1 事件对，
   MTE2(tile n) 与 MTE3(tile n-1) 交叠，`ReleaseEventID` 全部成对归还。
5. **同步资源**：IB `FanIn/FanOut`（kReady/kCopyRelease）+ A3WaitSeats 标量退避自旋
   （初值 32、每 256 次翻倍至 1024，上限 1e8 次 → status 341）；
   首 READY 64B（DeviceTag phase/epoch/group/packet 位域）+ ready=0 拒绝路径（301/329/310）。
6. **生产标志地址**：座位 epoch 标志 `ibPtr + kIbBytes + band*64`（IB 第二半、每座位独占
   64B、≤64 座位 4096B 不越界）；`A3ProduceSeat` 写 epoch 后 `FlushLine(flag)`，
   `A3WaitSeats` 读侧同样 `FlushLine` 保证跨核/SDMA 可见性。
7. **A3-tail 调度不变量**：Core0 `nextTail` 从 `leading`（前段末尾）起，先补齐当批自己的
   座位再 `A3WaitSeats` + `progress.Submit(g)`，ADD 在途时再生产一个末尾座位——
   不会等待只有自己能生产的座位。

宿主侧（`mf310p_v3_host.cpp`）：`IsA3` 接受 3302/3303/3304；`mf310p_a3_create/_2_create/
_tail_create` 三个导出符号齐备（nm 验证），admit 低位 admit/高位 role 编码与席位合同一致。
Device SO 导出 `mf310p_a3_device_run`；Host SO 链接 device SO 与 `libmf_smem.so`。

## 设备阶段 STOP（R1）

观察窗口 10:07–10:25（18 分钟连续采样，间隔 50–60s，共 16 次）：

- **dev4（Chip0）**：外部租户进程 PID 2834401（python，宿主 PID，容器内不可见）全程在场，
  恒定 19867 MB；DDR 带宽占用率稳定 51%，Aicore 0%（带宽型负载）。
  上一轮同根目录记录（quick-20261029-r1r2r3 的 npu-before）中 38144 无任何进程，
  本占用为本轮开始前新出现，非本轮遗留。
- **dev5（Chip1）**：间歇出现小体量租户探测进程（PID 2931389 等，75→1632 MB→消失，周期性）。
- 卡健康：双 Die Health OK；其他卡：5664 chip1、16384、43520 chip1 均有各自租户/告警。
- 容器内无自有 python 进程；未杀任何租户、未换卡（用户指令优先于 ADR-014 的换卡条款）。

按用户指令"卡忙、告警或独占不成立即STOP，不自动换卡、不杀其他租户"与
OPENCODE_EXECUTION §5，设备阶段停止；无失败会话、无双 Rank 现场需要保留（未启动）。
即便内存上勉强可容纳，51% 的 DDR 带宽污染会使八会话同批对照的时延结论失效，
且 lease-note 必须记录真实协调过的独占窗口——当前不满足，不得伪称。

## 复现与恢复

CPU/构建阶段复现（任意时刻）：

```bash
V3_RUN_ROOT=/tmp/opencode/310p-a3cmp-20261029-001
V3_PYTHON=$V3_RUN_ROOT/venv/bin/python
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cd /home/w00804146/310P/vllm-ascend-a3v3   # worktree @ 81780f342
git clone --single-branch --branch fix/310p-v3-orch-deploy-helper-20261029 \
  git@github.com:10wvw01/memfabric_hybrid.git $V3_RUN_ROOT/memfabric-src
git -C $V3_RUN_ROOT/memfabric-src checkout --detach db38d75e8485d31451302219eb7a1bfeb1dd107c
$V3_PYTHON -m pytest -q csrc/_310P/memfabric_mm_ar_v3/test \
  csrc/_310P/memfabric_mm_ar_v2/test/ut --import-mode=importlib       # 135 passed
$V3_PYTHON -m unittest discover -s tools/310p_mm_ar -p 'test_*.py'    # 38 OK
$V3_PYTHON tools/310p_mm_ar/project_gate.py --static                  # PASS
$V3_PYTHON csrc/_310P/memfabric_mm_ar_v3/tools/prepare_sdk.py \
  --source $V3_RUN_ROOT/memfabric-src --out $V3_RUN_ROOT/memfabric-sdk \
  --cann /usr/local/Ascend/cann-9.1.0 --jobs 8
$V3_PYTHON csrc/_310P/memfabric_mm_ar_v3/tools/build.py \
  --memfabric-source $V3_RUN_ROOT/memfabric-src --memfabric-home $V3_RUN_ROOT/memfabric-sdk \
  --cann /usr/local/Ascend/cann-9.1.0 --out $V3_RUN_ROOT/v3-build
$V3_PYTHON csrc/_310P/memfabric_mm_ar_v3/tools/build_tensor.py \
  --manifest $V3_RUN_ROOT/v3-build/manifest.json --out $V3_RUN_ROOT/v3-tensor
```

设备批次恢复（取得 38144 dev4/5 覆盖整批 ≈1 小时的真实独占窗口后）：

```bash
$V3_PYTHON csrc/_310P/memfabric_mm_ar_v3/tools/compare_a3.py start \
  --manifest $V3_RUN_ROOT/v3-build/manifest.json \
  --out <全新批次结果目录> --card 38144 --devices 4,5 \
  --lease-note '<现场真实独占窗口记录>'
$V3_PYTHON csrc/_310P/memfabric_mm_ar_v3/tools/compare_a3.py status --out <全新批次结果目录>
```

批次顺序（compare_a3.py ORDER）：compute512：A3前→A3-2→A3-tail→A3后；compute1024 同序。
每会话六门禁（IB→裸ADD→输出生命周期→协议/回绕→融合/下游ND→尾块）通过后
六配置 M2048/4096×comm512/1024/2048、4MiB 包、两轮各 4 预热+30 样本。
manifest 已含 tensor_adapter 冻结，卡片空闲后无需重新构建即可启动。

## 证据索引（均在本目录）

| 内容 | 位置 |
|---|---|
| 本轮停止说明（含占用时间线） | `device-stop/DEVICE_STOP_NOTE.md` |
| G0 预检 npu-smi/health（开始时，dev4 已被占） | `device-stop/npu-before.txt`、`health-before.txt` |
| STOP 时占用/进程/带宽快照 | `device-stop/npu-stop-occupancy.txt`、`proc-mem-38144-stop.txt`、`usages-38144-stop.txt` |
| 等待 18 分钟后快照（仍被占） | `device-stop/npu-after-wait.txt`、`health-after-wait.txt` |
| CPU 测试（135 passed） | `manifests/cpu-tests.log` |
| 治理 38 项 + 静态门 | `manifests/governance-tests.log`、`manifests/project-gate-static.log` |
| Ruff（V3 范围 clean） | `manifests/ruff.log` |
| SDK 构建身份与产物 SHA | `manifests/sdk_manifest.json`、`version.info`、`orch_deploy_helper_build.log`、`prepare_sdk_console.log`；上游全量构建日志 `memfabric_build.log`（408行，SHA256 `04b9e65f20c7e66a78fdb9200cc5ffa5dc706f45cfd86d688648247ba571e3a9`）保留在服务器运行根 `memfabric-sdk/`（含原始cmake尾随空格，依惯例不入库） |
| V3 构建身份（head/依赖/命令/SO SHA/tensor_adapter） | `manifests/v3_build_manifest.json`、`build_0.log`、`build_1.log` |
| Python 环境 | `manifests/python-environment.txt` |

## 遗留（交研发侧）

1. 设备批次未执行：需要研发侧协调 38144 双 Die 独占窗口（当前 dev4 有外部租户
   19.8GB/51% 带宽的 python 进程，dev5 有周期性探测）；或指示换卡（ADR-014 允许，
   但本次用户指令禁止自动换卡）。
2. KVER28 占位自拷贝源码差异（`KVER28_PADDING_SELF_COPY_PRESENT`）仍在，本轮不签长稳。
3. dev5 的间歇探测进程意味着"会话间隙维持独占"同样需要租约保障，与 A3_SEAT_SCHEDULE
   的"工具不声称进程间隙仍有连续守护"一致——恢复执行时 lease-note 须覆盖整批。
