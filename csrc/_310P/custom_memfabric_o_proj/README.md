# CUSTOM 310P3 MemFabric o_proj fusion

> **CUSTOMIZED 310P3 ONLY**
>
> Qwen3.6 full-attention `self_attn.o_proj + TP=2 reduction` 的 310P3
> 定制实现。所有平台特化 C++/AscendC 代码隔离在本目录。

## 当前合同

- Hardware: Ascend 310P3 / dav-2002
- Model: `qwen3_5_moe_text` full-attention `self_attn.o_proj`
- TP: 2
- Shape: global K=4096, local K=2048, N=2048
- Compute / exchange / output: FP16
- Transport: installed `wgm-dev-310p` MemFabric public SHM/SDMA API
- Adapter ABI: **v7**
- Native MM base tiling: **256 / 256 / 64**
- Producer blockDim: **8**
- Communication batch: `batch_m = 256 * q`, `q in {1,2,4}`, default 2

2 MiB 是默认 `q=2`、N=2048、FP16 时推导出的 batch payload，不是协议常量。

## 数据流

```text
wave gate (fixed credit)
        |
        v
P0: 8-core cooperative MM
    all cores clean + ready[batch][core]=generation
    core0 sole signal owner
        |
P1 -------------------------- SDMA0
        |
W0 -> A0 -------------------- SDMA1
        |
P2 -------------------------- ...
...
drain last wait/add
        |
public quiet (once per wave)
        |
wave credit ack
```

8 个 AI Core 全部参加 MM；core0 不独占通信角色，只在完成自身 MM 后兼任唯一
data `signal()` owner。wait/reduce 以 batch 为单位，credit 仍以 wave 为单位。

## 文件职责

- `memfabric_o_proj_binding.cpp`：PyTorch custom op
- `memfabric_o_proj_runtime.cpp`：arena/wave、lookahead pipeline、Graph/lifecycle
- `memfabric_o_proj_torch_adpt.h`：torch-facing 校验
- `memfabric310p_adapter_api.h`：内部 ABI v7
- `memfabric310p_adapter.cpp`：MemFabric public host API bridge
- `memfabric310p_device.asc`：8-core cooperative MM、public signal/wait/quiet、FP16 add

## 必须保持的边界

1. MemFabric 是外部黑盒；不得读取 request/arrival ring、head/tail、reserved
   layout、AICPU orchestrator、SQE 等私有状态。
2. device 数据面只调用 public `smem_shm_sdma_signal/wait/quiet`。
3. data `signal()` 只有 core0 调用。
4. data/credit mail 严格校验 `status/dst/len/imm`。
5. 协议异常写 status + `AscendC::Trap()`，失败 context 不继续复用。
6. ready 为 vLLM-owned `[batch][8 cores]` 独立 64B cache line，使用 generation。
7. arena 按 rows/内存预算定容，不随 batch payload 成比例放大。
8. tail zero-pad 到完整 `batch_m` 后仍走 8-core cooperative MM。
9. eager 单 stream；Graph capture 前完成 context/scratch/protocol/kernel warmup。
10. feature-off 不要求安装 MemFabric。

当前 cache visibility 使用 correctness-first 保守实现：每个 core 在 cooperative MM
结束后 clean 自己的 cache view；后续只有在 dav-2002 profiler/ownership 证据充分
时才能缩小 clean 范围。

## 构建

`cmake/memfabric_310p.cmake` 从 `MEMFABRIC_HYBRID_HOME_PATH` 读取 public
headers/`libmf_smem.so`，并用 bisheng `--npu-arch=dav-2002` 编译当前
`memfabric310p_device.asc`。仓库不保存正式预编译 device object。
