# 310P3 TP=2 未量化 o_proj + MemFabric AllReduce 使用说明

> 适用目标：Ascend 310P3 单卡双 die、TP=2、`Eco-Tech/Qwen3.6-35B-A3B-w8a8` 的
> **未量化 full-attention `o_proj`**（负责人裁决 A，2026-09-18：该 checkpoint 的
> o_proj 为 FLOAT；模型在 310P 上以 FP16 运行，融合管线交换 FP16 partial）。
>
> 当前状态：P5 完成（V5 epoch API + direct producer，单层 bit-exact 9/9 rows
> 验收通过，2026-09-21）；P6 性能优化进行中。定制 MemFabric 为
> `origin/wgm-dev-310p`（V5），epoch kernel（libmf_sdma_orch_v6.so）已部署在
> 设备 CP1 路径。
>
> 设计说明：[310p_memfabric_o_proj.md](../../developer_guide/310p_memfabric_o_proj.md)
>
> 研发计划：[310p_memfabric_o_proj_development_plan.md](../../developer_guide/310p_memfabric_o_proj_development_plan.md)

## 1. 你最终要部署什么

依赖关系如下：

```text
CANN + Driver/Firmware
        |
        +--> PyTorch 2.10.0 + torch-npu 2.10.0.post4
        |
        +--> matching vLLM
        |
        +--> customized MemFabric wgm-dev-310p (V5, origin/wgm-dev-310p)
        |       + smem host/device headers
        |       + libmf_smem / libmf_hybm_core / libacc_tcp_net
        |       + AICPU epoch kernel (libmf_sdma_orch_v6, 已部署设备侧 CP1 路径)
        |
        +--> this vLLM-Ascend feature branch
                + memfabric310p_adapter.cpp (编入 vllm_ascend_C)
                + memfabric310p_device.asc (CMake 自动 bisheng 编译)
                + Qwen3.6 未量化 o_proj 集成
```

不要把官方/上游 MemFabric 当成本功能依赖。需要先独立编译安装 `wgm-dev-310p`
V5，然后再编译本分支的 vLLM-Ascend。

## 2. 推荐环境

- Linux；
- Python >=3.10,<3.13；
- CANN 9.1.0；
- PyTorch 2.10.0；
- torch-npu 2.10.0.post4；
- vLLM 与当前 vLLM-Ascend 分支兼容；
- cmake >=3.26；
- pybind11；
- Ascend 310P3 driver/firmware；
- 定制 `wgm-dev-310p` MemFabric（V5）。

先确认服务器：

```bash
npu-smi info
python3 --version
cmake --version
```

## 3. 设备选择（实机要点）

以 `npu-smi` 实测为准。本服务器 4 张 310P3 卡中只有设备 0,1 所在卡全功能
可用（设备 4,5 所在卡 `AclrtMemSetAccess` 交叉 die 授权失败 507899，设备
2,3 卡有 Alarm）。**bring-up 固定用 0,1**。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
```

## 4. 编译安装定制 MemFabric（V5）

V5 = `origin/wgm-dev-310p` 分支（mailbox-ring epoch API，入口
`HybmSdmaOrchEpoch`，`MF_SDMA_ORCH_KVER=6`）。旧 V4（kfc/workspace API）
链路已废弃：该服务器设备侧 kfc_min.so 缺失、kfc 通道不可用。

```bash
git clone <内部镜像> memfabric_hybrid && cd memfabric_hybrid
git checkout origin/wgm-dev-310p   # V5

cmake -B build -DXPU_TYPE=NPU -DBUILD_PYTHON=OFF
cmake --build build -j
```

注意该分支 CMake 的一个坑：configure 时 `CMAKE_INSTALL_PREFIX` 目录若不
存在，`cmake --install` 会装到 `build` 旁的 `output/` 树。两种安装方式任选：

```bash
# 方式 A：先建 prefix 再 install
mkdir -p /opt/memfabric-wgm-dev-310p-v5
cmake --install build --prefix /opt/memfabric-wgm-dev-310p-v5

# 方式 B：直接拷贝 output 树
cp -a output/. /opt/memfabric-wgm-dev-310p-v5/
```

安装后必须能找到（V5 split 布局）：

```text
$MF_ROOT/smem/include/host/smem.h
$MF_ROOT/smem/include/host/smem_shm.h
$MF_ROOT/smem/include/device/smem_shm_aicore_base_sdma.h
$MF_ROOT/smem/lib64/libmf_smem.so
$MF_ROOT/hybm/lib64/libmf_hybm_core.so
$MF_ROOT/acc_links/lib64/libacc_tcp_net.so
$MF_ROOT/hybm/aicpu_kernel/libmf_sdma_orch_v6.json
```

epoch kernel 的设备侧部署：`libmf_sdma_orch_v6.so` 需位于设备
`/usr/lib64/aicpu_kernels/<id>/aicpu_kernels_device/`（CP1 搜索路径）。当前
两卡均已由同事部署；V5 的自动部署通道依赖 kfc（本机不可用），**kernel 侧
任何改动（如缩短 epoch 自限）后的重部署需要同事/厂商手动操作**。

## 5. 防止误用上游 Python 包

当前仓库 `requirements.txt` 仍包含通用 `memfabric_hybrid`。bring-up 时建议：

1. 编译本项目时使用 `--no-deps`，避免 pip 临时从公共源拉取另一个 MemFabric 实现；
2. 本融合路径的 C++ 链接最终以 `VLLM_ASCEND_310P_MEMFABRIC_ROOT` 和显式
   library list 为准，不依赖某个官方 soname。

## 6. 配置 vLLM-Ascend 编译环境

310P3 建议明确指定：

```bash
export SOC_VERSION=ascend310p3
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export COMPILE_CUSTOM_KERNELS=1
export MAX_JOBS=8

export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1
export MF_ROOT=/opt/memfabric-wgm-dev-310p-v5
export VLLM_ASCEND_310P_MEMFABRIC_ROOT="$MF_ROOT"
export VLLM_ASCEND_310P_MEMFABRIC_LIBRARIES="$MF_ROOT/smem/lib64/libmf_smem.so;$MF_ROOT/hybm/lib64/libmf_hybm_core.so;$MF_ROOT/acc_links/lib64/libacc_tcp_net.so"

# 运行时参数（编译期无关）
export VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TILE_M=32     # [16,4096] 内 2 的幂
export VLLM_ASCEND_310P_MEMFABRIC_STORE_URL=tcp://127.0.0.1:8581
export VLLM_ASCEND_310P_MEMFABRIC_LOCAL_BYTES=$((32 * 1024 * 1024))
```

`memfabric310p_device.asc` 由 `cmake/memfabric_310p.cmake` 自动用 bisheng
（dav-2002）编译成共享库并链接，无需手动编译。CMake 在 feature 开启时会
fail-fast 检查：

- SOC 必须是 `ascend310p*`；
- V5 host/device 头文件与 epoch launch json 存在；
- library list 非空且文件存在。

## 7. 安装 Python/vLLM 环境

先保证 compatible vLLM 已经安装，再编译安装本分支：

```bash
cd /path/to/vllm-ascend
python3 -m pip install -v -e . --no-build-isolation --no-deps
```

缺的普通 Python 依赖请显式按仓库版本安装。

## 8. 编译后快速检查

### 8.1 op 面是否注册

```bash
python3 - <<'PY'
import torch
import vllm_ascend.vllm_ascend_C  # noqa

ns = torch.ops._C_ascend
for op in [
    "memfabric_direct_o_proj_allreduce",
    "memfabric_o_proj_shutdown",
    "memfabric_o_proj_debug_snapshot",
]:
    print(op, hasattr(ns, op))
PY
```

预期均为 `True`。旧 phase-1 的 `memfabric_o_proj_begin/publish/finish/
mark_failed` 已随 V5 迁移删除。

### 8.2 检查最终 extension 动态依赖

```bash
python3 -c "import vllm_ascend.vllm_ascend_C as C; print(C.__file__)"
ldd /path/to/vllm_ascend_C*.so | sort
```

确认 DT_NEEDED 含 `libmf_smem.so / libmf_hybm_core.so / libacc_tcp_net.so /
libmf310p_device.so`，RUNPATH 覆盖 V5 四个 lib64 目录（无需再手工设置
LD_LIBRARY_PATH）。

## 9. 先运行单层 benchmark

不要第一步就启动 35B 模型。

**运行前必须设置**（split 布局下 json 自动发现失效，kfc 回退不可用）：

```bash
export MF_SDMA_ORCH_JSON=$MF_ROOT/hybm/aicpu_kernel/libmf_sdma_orch_v6.json
```

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1
export VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TILE_M=32

torchrun --standalone --nproc-per-node=2 \
  benchmarks/scripts/bench_310p_memfabric_o_proj_layer.py \
  --rows 1 8 32 33 64 128 512 2048 4096 \
  --repeat 5
```

预期输出：每个 rows 一行 `PASS`（bit-exact，max_abs_diff=0.000000），
最后一行 `ALL PASS`。

该 bench 的结构内建了 V5 平台约束（见下节）：reference HCCL 全部前置到
建池之前；fused 阶段只用 stream 级同步；池寿命控制在 ~20s 内；shutdown
后不再触碰设备。

## 10. V5 平台硬约束（务必阅读）

以下三条在当前服务器实测定论，任何基于本融合路径的宿主程序都必须遵守
（详细证据见研发计划 §9.3）：

1. **池存活期间禁止 device 级同步**：`torch.npu.synchronize()` 以及
   torch_npu 的 HCCL 集合通信（barrier/allreduce，内部同样 device-sync）
   会等待永续 epoch AICPU 任务而永久挂死；随后 epoch 被 25s
   launch-timeout 击杀并砖化设备 AICPU（507901）。只能用
   `torch.npu.current_stream().synchronize()`。
2. **单进程池寿命 < ~25s**：超过后 epoch 必然被击杀（本机 usleep 粒度
   导致 epoch 自限 ~28s > 25s）。bench 以"reference 前置 + 快速 fused
   循环"满足。生产化前需要 memfabric kernel 侧修复（缩短
   `HYBM_SDMA_ORCH_EPOCH_MAX_LOOPS` + KVER bump 重部署，厂商跟进）。
3. **shutdown 之后禁止任何设备操作**：`smem_shm_destroy` 可能静默断开
   HDC 会话，之后的 HCCL/launch 都会 507901。

诊断工具：`torch.ops._C_ascend.memfabric_o_proj_debug_snapshot()` 返回
CPU int64 张量，含双端 mailbox 环协议字（reqHead/Tail、quiet/arrival 戳、
邮件像、arena 首字），用于 waiter 挂死时的现场判定。

## 11. 下载目标模型

```bash
python3 -m pip install modelscope
mkdir -p /models/Qwen3.6-35B-A3B-w8a8

modelscope download \
  --model Eco-Tech/Qwen3.6-35B-A3B-w8a8 \
  --local_dir /models/Qwen3.6-35B-A3B-w8a8
```

## 12. 第一次启动模型：Eager correctness 模式

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1
export VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TILE_M=32
export VLLM_ASCEND_310P_MEMFABRIC_STORE_URL=tcp://127.0.0.1:8581
export VLLM_ASCEND_310P_MEMFABRIC_LOCAL_BYTES=$((32 * 1024 * 1024))
export MF_SDMA_ORCH_JSON=/opt/memfabric-wgm-dev-310p-v5/hybm/aicpu_kernel/libmf_sdma_orch_v6.json
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL=/models/Qwen3.6-35B-A3B-w8a8

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3.6-35b-a3b-w8a8 \
  --tensor-parallel-size 2 \
  --quantization ascend \
  --dtype float16 \
  --trust-remote-code \
  --enforce-eager \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 2>&1 | tee /tmp/qwen36-mf-bringup.log
```

说明：

- TP 必须为 2；`--dtype float16`（checkpoint 无 torch_dtype，本机 BF16 NZ
  linear 不支持）；
- 第一次不要额外开启 EP/sequence-parallel MoE，减少排查变量；
- `--enforce-eager` 是为了先验证算子，不代表最终部署必须 eager；
- **注意 §10 约束同样适用于 vLLM 进程**：在池创建（首个目标层 forward）
  之后，vLLM 代码路径中任何 device 级同步或 HCCL 之外的异常路径都可能
  触发挂死——P6/P7 需要逐点排查 model runner 中的 synchronize 调用。

## 13. 发一个最小请求

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-35b-a3b-w8a8",
    "messages": [{"role": "user", "content": "你好，请用一句话介绍你自己。"}],
    "temperature": 0,
    "max_tokens": 32
  }'
```

先检查：server 不 hang、两个 TP worker 都存活、输出非空、没有重复
allreduce、日志中出现融合路径 enable 信息、多次请求结果稳定。

## 14. Baseline 对比

关闭融合（需重新编译 extension）：

```bash
export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=0
```

保留同样模型/TP/dtype/输入/eager 模式，比较 generation correctness 和
单层/端到端性能。baseline/fused 最好分别保存 build，避免反复覆盖后无法
确认当前加载的是哪一版。

## 15. ACL Graph 验证（P7）

Eager 稳定后移除 `--enforce-eager`，重点观察：capture 是否成功、replay
是否 hang、repeated request 是否 stale、symmetric GVA 是否稳定。如果
graph 模式失败但 eager 正常，先做最小化 fusion-op graph capture 复现。

## 16. 常见失败排查

### CMake 找不到 V5 头文件/json

`VLLM_ASCEND_310P_MEMFABRIC_ROOT` 不是实际 install prefix，或装的是
V4 布局（`<root>/include` 扁平布局已废弃，需要 V5 split 布局）。

### 运行报 "sdma orchestration not ready" / orchestrator json not accessible

`MF_SDMA_ORCH_JSON` 未设置或路径错误（必须指向 v6 json）。

### fused call 挂死 + ~28s 后 507901

基本必为 §10 约束 1/2：宿主程序在池存活期间做了 device 级同步，或池寿命
超过 25s。用 `memfabric_o_proj_debug_snapshot()` 判定 wave 是否完成：
若 reqHead/quiet/arrival 戳均已推进而流不返回，则挂点在宿主的
device-sync，而非数据面。

### 模型启动后没有命中融合

检查：feature gate、model type、`*.self_attn.o_proj` prefix、
`layer_types[N]`、TP=2、FP16、未量化路由、shape 4096 -> local 2048 ->
output 2048。
