#!/usr/bin/env python3
# Discriminating experiments for shadow-kernel first-launch garbage:
#   V1: format_cast -> sync -> prewarm(noop) -> real call   (transdata-race theory)
#   V3: format_cast -> real call, NO prewarm at all          (does prewarm matter)
#   V4: format_cast -> prewarm -> sync -> 30 real calls      (arm-script pattern, stability)
#   V5: format_cast -> prewarm(noop) -> real -> real ...    (no sync anywhere; how many bad)
import ctypes, os
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu

K = N = 2048
rank = int(os.environ["RANK"]); local = int(os.environ["LOCAL_RANK"])
assert int(os.environ["WORLD_SIZE"]) == 2
torch.npu.set_device(local)
torch.npu.config.allow_internal_format = True
torch_npu.npu.set_compile_mode(jit_compile=False)
device = torch.device(f"npu:{local}")
dist.init_process_group(backend="gloo")

LIB = "/tmp/opencode/a1-mm-shadow-20261027/precheck/build/libmf310p_v2_mm_shadow.so"
lib = ctypes.CDLL(LIB)
fn = lib.mf310p_v2_mm_shadow_launch_async
fn.argtypes = [ctypes.c_uint64]*3 + [ctypes.c_uint32]*3 + [ctypes.c_void_p]
fn.restype = ctypes.c_int
stream = torch.npu.current_stream()
raw = ctypes.c_void_p(stream.npu_stream)

m = 2048
torch.manual_seed(4242)
w = torch.randn(N, K, dtype=torch.float16, device=device)
torch.manual_seed(1200 + rank*17 + m)
x = torch.randn(m, K, dtype=torch.float16, device=device)
ref = F.linear(x, w)
out = torch.empty_like(ref)

def real():
    rc = fn(x.data_ptr(), w_nz.data_ptr(), out.data_ptr(), m, 0, 1, raw)
    assert rc == 0, rc

def prewarm():
    for _ in range(2):
        rc = fn(0, 0, 0, m, 0, 0, raw)
        assert rc == 0, rc

def check(tag):
    stream.synchronize()
    ok = torch.equal(out, ref)
    print(f"V rank{rank} {tag} equal={ok}", flush=True)
    return ok

# V1: cast -> sync -> prewarm (no sync) -> 3 real calls
w_nz = torch_npu.npu_format_cast(w, 29)
stream.synchronize()
prewarm()
real(); check("V1-cast_sync_prewarm_nosync_call1")
real(); check("V1-call2")
real(); check("V1-call3")
dist.barrier()

# V3: fresh process state can't be reset here; use M4096 with NO prewarm, no cast-sync
m2 = 4096
torch.manual_seed(1200 + rank*17 + m2)
x2 = torch.randn(m2, K, dtype=torch.float16, device=device)
ref2 = F.linear(x2, w)
out2 = torch.empty_like(ref2)
w_nz2 = torch_npu.npu_format_cast(w, 29)
for _ in range(2):
    fn(0, 0, 0, m2, 0, 0, raw)
for i in range(3):
    fn(x2.data_ptr(), w_nz2.data_ptr(), out2.data_ptr(), m2, 0, 1, raw)
    stream.synchronize()
    print(f"V rank{rank} V3-m4096-nocastsync-noprewarmsync call{i+1} equal={torch.equal(out2, ref2)}", flush=True)
dist.barrier()

# V4: arm-script pattern on M4096: prewarm -> sync -> 30 calls
for _ in range(2):
    fn(0, 0, 0, m2, 0, 0, raw)
stream.synchronize()
bad = 0
for i in range(30):
    fn(x2.data_ptr(), w_nz2.data_ptr(), out2.data_ptr(), m2, 0, 1, raw)
    stream.synchronize()
    if not torch.equal(out2, ref2):
        bad += 1
print(f"V rank{rank} V4-armscript-pattern-30calls bad={bad}", flush=True)
dist.barrier()

# V5: continuous no-sync launch chain on M2048 (arm pattern but no sync): launch 5 back-to-back, sync once
real(); real(); real(); real(); real()
stream.synchronize()
print(f"V rank{rank} V5-5backtoback-1sync equal={torch.equal(out, ref)}", flush=True)
dist.destroy_process_group()
