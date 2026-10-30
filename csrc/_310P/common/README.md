# 310P公开Torch Tensor调用层

`torch_tensor_entry.py`注册`torch.ops.memfabric310p.mm_ar(X,W,context)->Y`。
它是可独立调用的算子入口，不导入L1_test；V2/V3的家族实现分别放在各工程src/torch_adapter/tensor_entry.py。
公开入口在每次调用内处理输入格式验证、输出槽获取/Storage包装、租约和异步输入保活。
底层算法仍走本版C ABI或冻结会话；两者不直接暴露为L1裸指针out接口。

下面是V2-7/M2048的Eager调用形状示意，实际执行必须先完成该设备/SDK/工件的准入：

```python
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
import torch_npu

repo = Path('/actual/vllm-ascend')
sys.path.insert(0, str(repo / 'csrc/_310P/common'))
from torch_tensor_entry import create_operator

# 各Rank已设置正确物理NPU、通信地址，X/W已就绪；初始化在计时外。
ctx = SimpleNamespace(torch=torch, linear=F.linear, rank=rank, url=store_url,
                      device=str(X.device), stream=torch.npu.current_stream(),
                      shapes=[2048], output_slots=2)
spec = dict(api='v2_selected_v1', library='/actual/vllm_ascend_C.so',
            tensor_adapter='/actual/mf310p_tensor_bridge.so')
op = create_operator('v2', spec, {'id': 'V2-7'}, ctx)
Y = op.run(X, W_NZ)  # 内部真实Torch dispatcher调用；不是外部传out。
Z = Y + residual    # 同Stream立即消费，无需中间Host等待。
ctx.stream.synchronize()
op.after_block()   # 已完成后审计协议。
del Y, Z           # 所有Y的Storage别名也必须释放。
op.close()         # 显式生命周期，不在GC中执行集合销毁。
```

只支持原Stream的FP16 ND X[M,2048]和NZ29 W[2048,2048]，权重按F.linear语义。
两Rank必须一致调用，任一异常停止双方；不自动重试、不回退Stock。
上下文强注册至显式close，避免丢弃Python对象时提前回收在途内存；失败上下文保留到进程退出。
A2/A3最多持有两个Y（含别名），A2原有内部同步保留；raw版内部槽数受首写表容量约束。
调用方不得跨Stream消费/复用这些池输出，Graph/模型调用点集成尚未验证。

`build_tensor_bridge.py`可在指定fresh build子目录独立编译Storage桥，不链接MemFabric；
复用既有pool_tensor.cpp，保留TorchNPU Storage描述及别名析构租约机制。
与Stock对齐的是公开Tensor输入/完整输出边界，内部分配策略可以不同。
[完整测量合同](../tools/L1_test/MEASUREMENT_V4.md)明确冷/热成本与证据强度。
