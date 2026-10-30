import ctypes, os, subprocess, sys
import torch, torch_npu
d = 4
torch.npu.set_device(d)
dev = f'npu:{d}'
x = torch.arange(0, 1024, dtype=torch.uint8).to(dev)
y = torch.zeros(1024, dtype=torch.uint8).to(dev)
torch.npu.synchronize()
lib = ctypes.CDLL('/tmp/opencode/direct-launch-test/libmini.so')
lib.mini_add_launch.argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
lib.mini_add_launch.restype=ctypes.c_int
s = torch.npu.current_stream().npu_stream
for i in range(3):
    lib.mini_add_launch(x.data_ptr(), y.data_ptr(), 1024, ctypes.c_void_p(s))
torch.npu.synchronize()
print('PRE_FORK_DONE', flush=True)
r = subprocess.run(['/home/w00804146/310P/a3ut-20261029/memfabric-sdk/aarch64-linux/lib64/orch_deploy_helper'],
                   capture_output=True, text=True, timeout=120)
print('HELPER_RC', r.returncode, r.stdout.strip()[:80], flush=True)
for i in range(3):
    lib.mini_add_launch(x.data_ptr(), y.data_ptr(), 1024, ctypes.c_void_p(s))
torch.npu.synchronize()
print('POST_FORK_DONE', bool((y[:16].cpu() == x[:16].cpu() + 1).all().item()), flush=True)
