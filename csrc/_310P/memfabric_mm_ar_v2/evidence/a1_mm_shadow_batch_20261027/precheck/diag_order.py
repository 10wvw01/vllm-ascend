#!/usr/bin/env python3
# Precise ordering bisect:
#  T1: w -> cast -> x -> ref=F.linear(x,w) -> out -> prewarm -> real -> check   [profile order, known FAIL]
#  T2: w -> x -> ref=F.linear(x,w) -> out -> cast -> prewarm -> real -> check   [cast after ref]
#  T3: w -> cast -> x -> ref=F.linear(x,w) -> out -> SYNC -> prewarm -> real    [sync after ref]
#  T4: w -> cast -> x -> ref2 (separate out buffer) ... same as T1 but F.linear AFTER out alloc
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
def fresh():
    torch.manual_seed(4242)
    w = torch.randn(N, K, dtype=torch.float16, device=device)
    torch.manual_seed(1200 + rank*17 + m)
    x = torch.randn(m, K, dtype=torch.float16, device=device)
    return w, x

def run(tag, w, x, w_nz, ref, sync_before_real=False):
    out = torch.empty_like(ref)
    for _ in range(2):
        rc = fn(0, 0, 0, m, 0, 0, raw); assert rc == 0
    if sync_before_real:
        stream.synchronize()
    rc = fn(x.data_ptr(), w_nz.data_ptr(), out.data_ptr(), m, 0, 1, raw); assert rc == 0
    stream.synchronize()
    ok = torch.equal(out, ref)
    print(f"T rank{rank} {tag} equal={ok}", flush=True)
    return ok

# T1: exact profile order (cast BEFORE ref)
w, x = fresh()
w_nz = torch_npu.npu_format_cast(w, 29)
ref = F.linear(x, w)
run("T1-cast-before-ref", w, x, w_nz, ref)
dist.barrier()

# T2: cast AFTER ref
w, x = fresh()
ref = F.linear(x, w)
w_nz = torch_npu.npu_format_cast(w, 29)
run("T2-cast-after-ref", w, x, w_nz, ref)
dist.barrier()

# T3: cast before ref, but SYNC between out alloc and prewarm
w, x = fresh()
w_nz = torch_npu.npu_format_cast(w, 29)
ref = F.linear(x, w)
run("T3-cast-before-ref-sync", w, x, w_nz, ref, sync_before_real=True)
dist.barrier()

# T4: cast before ref, native check in between (like native arm pattern)
w, x = fresh()
w_nz = torch_npu.npu_format_cast(w, 29)
ref = F.linear(x, w)
y = F.linear(x, w_nz)   # native NZ matmul consumes w_nz first
run("T4-cast-ref-native-consume", w, x, w_nz, ref)
dist.barrier()

# T5: repeat T1 with a SECOND fresh set to confirm determinism
w, x = fresh()
w_nz2 = torch_npu.npu_format_cast(w, 29)
ref = F.linear(x, w)
run("T5-T1-repeat", w, x, w_nz2, ref)
dist.destroy_process_group()
