# 设备阶段STOP记录（2026-10-29 10:25 local）

## 停止原因：38144 dev4 独占窗口不成立（卡忙）

观察窗口：10:07 ~ 10:25（约18分钟连续采样）。

- dev4（Chip0）：外部租户进程 PID 2834401（python，宿主PID，容器内不可见），
  持续占用 19867 MB，DDR 带宽占用率稳定 51%，Aicore 0%（带宽型负载）。
  上一轮（2026-10-28，/tmp/opencode/310p-v3-20261029-001/npu-before.txt）该卡无任何进程，
  本占用为本轮开始前新出现。
- dev5（Chip1）：间歇出现小体量租户探测进程（PID 2931389 等，75MB→1632MB→消失，周期性出现）。
- 卡健康：双 Die Health OK（无 Alarm）。
- 容器内无自有python进程（ps 为空），未杀任何租户、未换卡、未启动任何设备会话。

按用户指令与 OPENCODE_EXECUTION §5 / ADR-014：卡忙即 STOP，不自动换卡、不杀其他租户、
不等待无限重试。本轮未运行 compare_a3 批次，未触发任何设备 Kernel，无设备侧风险。

## 已完成（CPU阶段全部通过，产物已冻结可用）

1. CPU：V3+V2 UT 135 passed；治理 38 tests OK；project_gate --static PASS；
   V3范围 Ruff clean；git diff --check OK（日志见本目录 *.log）。
2. 私有 MemFabric SDK：db38d75e（tree 4fe624ca）构建 PASS，
   orch_deploy_helper SHA256 849f93c0...（与生产验证版一致）。
3. 私有 V3 Host/Device SO：HEAD 81780f342 构建 PASS，三候选 create 符号齐全。
4. TorchNPU Storage 适配器：已编译并冻结进 manifest（6be03080...）。

## 恢复方式（卡空闲后）

取得 38144 dev4/5 覆盖整批（8会话，约1小时）的真实独占窗口后，直接运行：

```bash
V3_PYTHON=/tmp/opencode/310p-a3cmp-20261029-001/venv/bin/python
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cd /home/w00804146/310P/vllm-ascend-a3v3
$V3_PYTHON csrc/_310P/memfabric_mm_ar_v3/tools/compare_a3.py start \
  --manifest /tmp/opencode/310p-a3cmp-20261029-001/v3-build/manifest.json \
  --out <全新批次结果目录> --card 38144 --devices 4,5 \
  --lease-note '<现场真实独占窗口记录>'
$V3_PYTHON csrc/_310P/memfabric_mm_ar_v3/tools/compare_a3.py status --out <全新批次结果目录>
```

## 现场证据文件

- npu-before.txt / health-before.txt：开始时状态（dev4 已被占用）
- npu-stop-occupancy.txt / proc-mem-38144-stop.txt / usages-38144-stop.txt：STOP 时占用
- npu-after-wait.txt / health-after-wait.txt：等待后状态（仍被占用）
