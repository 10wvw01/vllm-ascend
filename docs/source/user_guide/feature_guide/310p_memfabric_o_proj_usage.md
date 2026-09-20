# 310P3 TP=2 W8A8 o_proj + MemFabric AllReduce 使用说明

> 适用目标：Ascend 310P3 单卡双 die、TP=2、`Eco-Tech/Qwen3.6-35B-A3B-w8a8` 的
> **未量化 full-attention `o_proj`**（负责人裁决 A，2026-09-18：该 checkpoint 的
> o_proj 为 FLOAT；模型在 310P 上以 FP16 运行，融合管线交换 FP16 partial）。
>
> 当前状态：P0/P1/P2 实机验收已通过（build、TP=2 correctness、1000-wave
> stress、单层 bit 级一致）。文档中的 custom MemFabric library 名称和 `.asc`
> 编译命令以 `wgm-dev-310p` 服务器实际源码/构建产物为准。
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
        +--> customized MemFabric wgm-dev-310p
        |       + headers
        |       + libraries
        |       + AICore/AICPU/SDMA runtime
        |
        +--> this vLLM-Ascend feature branch
                + memfabric310p_adapter.cpp
                + memfabric310p_device.asc object
                + Qwen3.6 W8A8 integration
```

不要把官方/上游 MemFabric 当成本功能依赖。需要先独立编译安装 `wgm-dev-310p`，然后再编译本分支的 vLLM-Ascend。

## 2. 推荐环境

仓库当前依赖要求至少包括：

- Linux；
- Python >=3.10,<3.13；
- CANN 9.1.0；
- PyTorch 2.10.0；
- torch-npu 2.10.0.post4；
- vLLM 与当前 vLLM-Ascend 分支兼容；
- cmake >=3.26；
- pybind11；
- Ascend 310P3 driver/firmware；
- 定制 `wgm-dev-310p` MemFabric。

先确认服务器：

```bash
npu-smi info
python3 --version
cmake --version
python3 - <<'PY'
import torch
import torch_npu
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu available:", torch.npu.is_available())
PY
```

建议保存：

```bash
npu-smi info > /tmp/npu-info.txt
```

## 3. 拉取本项目分支

```bash
git clone https://github.com/10wvw01/vllm-ascend.git
cd vllm-ascend
git checkout feat/310p-w8a8-o-proj-memfabric-ar
git rev-parse HEAD
```

后续问题排查时始终记录 `git rev-parse HEAD`。

## 4. 获取并安装定制 MemFabric

定制分支：

```text
https://gitcode.com/GDD_ESCC/memfabric_hybrid/tree/wgm-dev-310p
```

如果服务器支持 GitCode HTTPS clone，可使用类似：

```bash
git clone -b wgm-dev-310p https://gitcode.com/GDD_ESCC/memfabric_hybrid.git
cd memfabric_hybrid
git rev-parse HEAD
```

如果使用内部镜像，请确保 commit 与 `wgm-dev-310p` 对应。

### 4.1 按定制分支自身方式编译安装

这里不要套用官方 MemFabric 的 library 名或安装布局。先阅读该分支的 README/CMake/example 08，再按它自己的方式 build/install。

建议最终安装到一个独立 prefix，例如：

```bash
export MF_ROOT=/opt/memfabric-wgm-dev-310p
```

安装完成后至少要能找到：

```bash
find "$MF_ROOT/include" -name smem.h -o -name smem_shm.h -o -name smem_shm_aicore_sdma.h
find "$MF_ROOT" \( -name '*.so' -o -name '*.a' \) -type f | sort
```

必须确认以下三个头文件来自定制安装：

```text
smem.h
smem_shm.h
smem_shm_aicore_sdma.h
```

### 4.2 防止误用上游 Python 包

当前仓库 `requirements.txt` 仍包含通用 `memfabric_hybrid`。bring-up 时建议：

1. 先安装 `wgm-dev-310p` 自己生成的 Python/package 产物（如果它提供）；
2. 记录安装位置；
3. 编译本项目时使用 `--no-deps`，避免 pip 临时从公共源拉取另一个 MemFabric 实现。

检查：

```bash
python3 - <<'PY'
try:
    import importlib.metadata as md
    d = md.distribution("memfabric_hybrid")
    print("memfabric_hybrid version:", d.version)
    print("location:", d.locate_file(""))
except Exception as e:
    print("memfabric_hybrid distribution not found:", e)
PY
```

本融合路径的 C++ 链接最终以 `VLLM_ASCEND_310P_MEMFABRIC_ROOT` 和显式 library list 为准，不依赖某个官方 soname。

## 5. 编译 `memfabric310p_device.asc`

源码：

```text
csrc/memfabric_o_proj/external/memfabric310p_device.asc
```

实机验证过的编译命令（2026-09-18，CANN 9.1.0；`-fPIC` 必需，产物为可
直接链接进 `vllm_ascend_C` 的 ELF relocatable）：

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export MF_ROOT=/opt/memfabric-wgm-dev-310p

bisheng --npu-arch=dav-2002 -O2 -std=c++17 -w -fPIC \
  -x asc csrc/memfabric_o_proj/external/memfabric310p_device.asc -x none \
  -c -o build_310p_artifacts/memfabric310p_device.o \
  -I$ASCEND_HOME_PATH/include -I$MF_ROOT/include
```

链接说明（已固化在 `cmake/memfabric_310p.cmake`）：设备对象引用的
AscendC launch 桩（`AscendLaunchKernelWithHostArgs` 等）由 CANN 静态库
`libascendc_runtime.a` 提供，CMake 会在 `ASCEND_HOME_PATH` 下自动查找并
追加链接；`aclrt*` 符号在加载期由 `libascendcl.so` 解析。

## 6. 配置 MemFabric library list

从定制安装实际产物确定链接库。示例：

```bash
export MF_LIBS='/opt/memfabric-wgm-dev-310p/lib64/libA.so;/opt/memfabric-wgm-dev-310p/lib64/libB.so'
```

这里只是格式示例，`libA.so/libB.so` 不是约定名称。

如果链接失败，优先从 example 08 的 link line/CMake target 中提取真实依赖和顺序。

## 7. 配置 vLLM-Ascend 编译环境

310P3 建议明确指定：

```bash
export SOC_VERSION=ascend310p3
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export COMPILE_CUSTOM_KERNELS=1
export MAX_JOBS=8

export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1
export VLLM_ASCEND_310P_MEMFABRIC_ROOT="$MF_ROOT"
export VLLM_ASCEND_310P_MEMFABRIC_LIBRARIES="$MF_LIBS"
export VLLM_ASCEND_310P_MEMFABRIC_DEVICE_OBJECT="$MF_DEVICE_OBJ"
export VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TILE_M=32

# Runtime rendezvous，默认也是 tcp://127.0.0.1:8581
export VLLM_ASCEND_310P_MEMFABRIC_STORE_URL=tcp://127.0.0.1:8581

# 每 rank symmetric pool physical contribution，默认 32 MiB
export VLLM_ASCEND_310P_MEMFABRIC_LOCAL_BYTES=$((32 * 1024 * 1024))
```

如果 `ASCEND_HOME_PATH` 实际不同，以服务器 CANN 安装为准。

CMake 在 feature 开启时会 fail-fast 检查：

- SOC 必须是 `ascend310p*`；
- custom MemFabric root 存在；
- 三个必要头文件存在；
- library list 非空；
- device object 存在。

## 8. 安装 Python/vLLM 环境

先保证 compatible vLLM 已经安装。确认：

```bash
python3 - <<'PY'
import vllm
print(vllm.__version__)
PY
```

再确认仓库要求的核心依赖：

```bash
python3 - <<'PY'
import torch, torch_npu
assert torch.__version__.startswith("2.10.0"), torch.__version__
assert torch_npu.__version__.startswith("2.10.0"), torch_npu.__version__
print("core versions OK")
PY
```

在受控 310P bring-up 环境中，建议先显式准备依赖，再这样编译安装本分支：

```bash
cd /path/to/vllm-ascend
python3 -m pip install -v -e . --no-build-isolation --no-deps
```

使用 `--no-deps` 的目的，是避免 pip 在此阶段自动替换/下载通用 `memfabric_hybrid`。缺的普通 Python 依赖请显式按仓库版本安装。

如果你确认当前定制 `memfabric_hybrid` package 已经正确满足 `requirements.txt`，也可以按团队正常 vLLM-Ascend 安装流程构建。

## 9. 编译后快速检查

### 9.1 extension 能否加载

```bash
python3 - <<'PY'
import torch
import vllm_ascend
import vllm_ascend.vllm_ascend_C

ns = torch.ops._C_ascend
for op in [
    "memfabric_o_proj_begin",
    "memfabric_o_proj_publish",
    "memfabric_o_proj_finish",
    "memfabric_o_proj_mark_failed",
]:
    print(op, hasattr(ns, op))
PY
```

预期均为 `True`。

### 9.2 检查最终 extension 动态依赖

找到 extension：

```bash
python3 - <<'PY'
import vllm_ascend.vllm_ascend_C as C
print(C.__file__)
PY
```

然后：

```bash
ldd /path/to/vllm_ascend_C*.so | sort
```

确认实际链接的是 custom `wgm-dev-310p` 安装产物，而不是系统里的其他 MemFabric。

必要时：

```bash
export LD_LIBRARY_PATH="$MF_ROOT/lib64:$MF_ROOT/lib:$LD_LIBRARY_PATH"
```

## 10. 先运行独立 MemFabric benchmark

不要第一步就启动 35B 模型。

实机要点（2026-09-18 验证）：

- **设备选择**：以 `npu-smi` 实测为准。本服务器 4 张 310P3 卡中只有
  设备 0,1 所在卡全功能可用（设备 4,5 所在卡 `AclrtMemSetAccess` 交叉
  die 授权失败 507899，设备 2,3 卡有 Alarm）。**bring-up 固定用 0,1**。
- **运行时库路径**：`export LD_LIBRARY_PATH=$MF_ROOT/lib64:$LD_LIBRARY_PATH`
  （extension 的 RUNPATH 不传递到 `libmf_smem.so` 的间接依赖）。
- 首次运行含 AICPU 编排 kernel 自动部署（CUST 迁移 + KFC 就位 + 校验），
  秒级一次性开销，之后零部署复用。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1
export VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TILE_M=32
export LD_LIBRARY_PATH=/opt/memfabric-wgm-dev-310p/lib64:$LD_LIBRARY_PATH

torchrun --standalone --nproc-per-node=2 \
  benchmarks/scripts/bench_310p_memfabric_o_proj.py \
  --rows 1 8 32 64 128 512 2048 \
  --warmup 5 \
  --repeat 20
```

这个 benchmark：

- MemFabric 路径执行 bilateral SDMA + local reduce；
- HCCL 只作为 reference（注意：310P CANN 9.1.0 的 HCCL 不支持 BF16
  allreduce，reference 以 FP32 归约后转回 BF16）；
- correctness 会先检查，再统计 latency。

repeated-wave 逐次校验 stress（验收门槛 1000 次）：

```bash
torchrun --standalone --nproc-per-node=2 \
  benchmarks/scripts/bench_310p_memfabric_o_proj.py \
  --rows 1 32 128 512 2048 \
  --warmup 20 \
  --repeat 1000 \
  --stress-check
```

如果这里出现随机 hang 或偶发 mismatch，优先处理 wave barrier/flag reset，不要先启动模型。

## 11. 下载目标模型

推荐先下载到本地目录，避免两 rank 启动时同时远程拉模型。

```bash
python3 -m pip install modelscope
mkdir -p /models/Qwen3.6-35B-A3B-w8a8

modelscope download \
  --model Eco-Tech/Qwen3.6-35B-A3B-w8a8 \
  --local_dir /models/Qwen3.6-35B-A3B-w8a8
```

检查：

```bash
ls -lah /models/Qwen3.6-35B-A3B-w8a8
```

## 12. 第一次启动模型：Eager correctness 模式

先确定 310P3 两个逻辑 die ID。以下 `0,1` 只是示例：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
```

融合 feature runtime 环境：

```bash
export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1
export VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TILE_M=32
export VLLM_ASCEND_310P_MEMFABRIC_STORE_URL=tcp://127.0.0.1:8581
export VLLM_ASCEND_310P_MEMFABRIC_LOCAL_BYTES=$((32 * 1024 * 1024))
export LD_LIBRARY_PATH="$MF_ROOT/lib64:$MF_ROOT/lib:$LD_LIBRARY_PATH"
```

第一次 bring-up 建议：

```bash
MODEL=/models/Qwen3.6-35B-A3B-w8a8

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3.6-35b-a3b-w8a8 \
  --tensor-parallel-size 2 \
  --quantization ascend \
  --dtype bfloat16 \
  --trust-remote-code \
  --enforce-eager \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 2>&1 | tee /tmp/qwen36-mf-bringup.log
```

说明：

- TP 必须为 2；
- 第一次不要额外开启 EP/sequence-parallel MoE，减少排查变量；
- `--enforce-eager` 是为了先验证算子，不代表最终部署必须 eager；
- 若 OOM，先降低 `max-model-len` / `max-num-seqs`，不要先改融合算法；
- 如果环境中 vLLM CLI 对某参数名称有变化，以当前安装版本的 `vllm serve --help` 为准，并把最终可用命令回填本文。

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

先检查：

- server 不 hang；
- 两个 TP worker 都存活；
- 输出非空；
- 没有重复 allreduce；
- 日志中出现融合路径 enable 信息；
- 多次请求结果稳定。

## 14. Baseline 对比

关闭融合：

```bash
export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=0
```

重新编译/启动一个 baseline 环境，保留：

- 同样模型；
- TP=2；
- 同样 dtype/quantization；
- 同样输入；
- 同样 eager/graph 模式。

比较 generation correctness 和单层/端到端性能。

注意：build-time feature gate 会影响 extension 编译，因此 baseline/fused 最好分别保存 wheel 或 build directory，避免反复覆盖后无法确认当前加载的是哪一版。

## 15. ACL Graph 验证

Eager 稳定后，移除：

```text
--enforce-eager
```

再启动模型，重点观察：

- capture 是否成功；
- replay 是否 hang；
- repeated request 是否 stale flag；
- symmetric GVA 是否稳定；
- 不同 batch/M 是否触发错误资源复用。

如果 graph 模式失败但 eager 正常，先做最小化 fusion-op graph capture 复现，不要同时修改通信协议和 matmul。

## 16. 常见失败排查

### CMake 找不到 `smem_shm_aicore_sdma.h`

说明 `VLLM_ASCEND_310P_MEMFABRIC_ROOT` 不是实际 install prefix，或者定制分支没有安装该 header。不要让 CMake 去系统目录兜底；修正 custom install。

### Linker undefined reference

从 example 08 的真实 link line 找缺失 library，补到：

```bash
VLLM_ASCEND_310P_MEMFABRIC_LIBRARIES
```

并保持依赖顺序。

### `.asc` object 链不上

说明当前 device compiler 输出形式与普通 object 假设不一致。把 example 08 的实际 build target 移植到本项目 CMake，不要用 host C++ compiler 编 `.asc`。

### Benchmark 双 rank hang

优先检查：

- rank0/rank1 是否都进入同一个 wave；
- config-store URL；
- submit 顺序；
- mailbox；
- arrival flag；
- wave clear/barrier race；
- AICPU SDMA result stage。

### 第一次能跑，第二次 hang/mismatch

高度怀疑 stale flag / late clear / mailbox reuse。先解决 reusable-wave 协议。

### 模型启动后没有命中融合

检查：

- feature gate；
- model type；
- `*.self_attn.o_proj` prefix；
- `layer_types[N]`；
- TP=2；
- BF16；
- static 310P W8A8；
- shape 4096 -> local 2048 -> output 2048。

## 17. 实机验证完成后必须回填

请把以下内容更新到文档/PR：

- 定制 MemFabric commit；
- 实际 build/install 命令；
- 实际 library names；
- `.asc` 编译命令；
- device object/stub 形式；
- barrier/flag 最终语义；
- `smem_shm_sdma_submit` offset/flag 语义；
- 310P3 `npu-smi` topology；
- 最终可复现 vLLM build 命令；
- 最终可复现 model serve 命令；
- correctness 数据；
- profiler timeline；
- baseline/fused 性能。

这些内容完成后，本文应从“bring-up guide”收敛为最终部署手册。
