import ctypes, os, sys
import torch, torch_npu
d = int(sys.argv[1]) if len(sys.argv) > 1 else 4
torch.npu.set_device(d)
dev = f'npu:{d}'
ctrl = torch.ones(1024, 1024, dtype=torch.float16, device=dev)
z = ctrl * 2  # aclnn control op
torch.npu.synchronize()
x = torch.arange(0, 1024, dtype=torch.uint8).to(dev)
y = torch.zeros(1024, dtype=torch.uint8).to(dev)
torch.npu.synchronize()
lib = ctypes.CDLL('/tmp/opencode/direct-launch-test/libmini.so')
lib.mini_add_launch.argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
lib.mini_add_launch.restype=ctypes.c_int
s = torch.npu.current_stream().npu_stream
for i in range(5):
    rc = lib.mini_add_launch(x.data_ptr(), y.data_ptr(), 1024, ctypes.c_void_p(s))
    if rc: print('launch rc', rc); break
torch.npu.synchronize()
print('mini result ok:', bool((y[:16].cpu() == x[:16].cpu() + 1).all().item()), flush=True)
