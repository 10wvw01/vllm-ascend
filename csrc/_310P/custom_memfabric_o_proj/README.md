# CUSTOM 310P3 MemFabric o_proj fusion

> **CUSTOMIZED 310P3 ONLY**
>
> 本目录包含 Qwen3.6 full-attention `self_attn.o_proj + TP=2 reduction`
> 在 Ascend 310P3 上的项目定制实现。它与通用 `csrc/` 代码隔离，避免把
> 平台特化逻辑扩散到通用算子。

## 当前适用范围

- Hardware: Ascend 310P3 / dav-2002
- Model route: `qwen3_5_moe_text` 的 full-attention `self_attn.o_proj`
- TP: 2
- Shape: global K=4096, local K=2048, N=2048
- Compute / payload / output: FP16
- Target layer: checkpoint 中未量化 FLOAT 条目
- Transport: 安装版 `wgm-dev-310p` MemFabric **public SHM/SDMA API**
- Adapter ABI: v6

## 当前数据流

```text
block 1..7
  MM(chunk)
    -> cache clean
    -> ready[chunk]
                 \
                  \ 
block 0 coordinator
  wait ready[chunk]
    -> smem_shm_sdma_signal()
            |
            v
      MemFabric / SDMA
            |
            v
  quiet + strict wait
            |
            v
       FP16 local add
            |
            v
     fixed-credit ack
```

同一 chunk 必须满足 `MM -> clean -> ready -> signal`；后续 MM 不等待前一个
chunk 的 SDMA 完成，目标是 `MM(later) || SDMA(earlier)`。

## 文件职责

- `memfabric_o_proj_binding.cpp`：PyTorch custom op 注册
- `memfabric_o_proj_runtime.cpp`：persistent runtime、wave、fixed credit、Graph/lifecycle
- `memfabric_o_proj_torch_adpt.h`：torch-facing 声明
- `memfabric310p_adapter_api.h`：vLLM 内部 adapter ABI
- `memfabric310p_adapter.cpp`：MemFabric public host API bridge
- `memfabric310p_device.asc`：7 MM workers + 1 coordinator、public signal/wait/quiet、FP16 add、fail-stop

## 必须保持的边界

1. MemFabric 是只读外部依赖，不修改其仓库。
2. vLLM 不读取 request/arrival ring、head/tail、reserved layout、AICPU
   orchestrator、SQE 等内部实现。
3. device 数据面只调用 public `smem_shm_sdma_signal/wait/quiet`。
4. 数据 `signal()` 只由 coordinator block 发起。
5. data/credit mail 都必须校验 `status/dst/len/imm`。
6. 协议异常写入 vLLM-owned status 并 `AscendC::Trap()`，不得继续 add/ack。
7. correctness 不依赖 host `wave_count`；wave 复用由 fixed credit 保护。
8. eager context 只允许一个执行 stream；ACL Graph capture side stream 是
   framework-managed 例外。
9. Graph capture 前必须完成 context、scratch、protocol 和需要的 bucket warmup。
10. feature-off 时不得要求系统安装 MemFabric。

## 构建

- 根 `CMakeLists.txt` 显式编译本目录的 binding/runtime。
- `cmake/memfabric_310p.cmake` 在 feature-on 时读取
  `MEMFABRIC_HYBRID_HOME_PATH`，查 public headers 与 `libmf_smem.so`。
- `memfabric310p_device.asc` 由当前 CANN bisheng 以 dav-2002 目标实时编译。

仓库不保存本功能的预编译 `.o`。任何 device artifact 都必须由当前源码、
当前 CANN 和当前 MemFabric 安装环境重新生成。
