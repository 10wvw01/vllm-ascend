#!/usr/bin/env python3
# Reproduce nd-arm correctness failure under torchrun dual-rank; dump mismatch pattern.
import ctypes, os
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
torch_npu.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = True

K = N = 2048
rank = int(os.environ["RANK"]); local = int(os.environ["LOCAL_RANK"])
assert int(os.environ["WORLD_SIZE"]) == 2
torch.npu.set_device(local)
device = torch.device(f"npu:{local}")
dist.init_process_group(backend="gloo")

LIB = "/tmp/opencode/a1-mm-shadow-20261027/precheck/build/libmf310p_v2_mm_shadow.so"
lib = ctypes.CDLL(LIB)
fn = lib.mf310p_v2_mm_shadow_launch_async
fn.argtypes = [ctypes.c_uint64]*3 + [ctypes.c_uint32]*3 + [ctypes.c_void_p]
fn.restype = ctypes.c_int
stream = torch.npu.current_stream()
raw = ctypes.c_void_p(stream.npu_stream)

torch.manual_seed(4242)
w = torch.randn(N, K, dtype=torch.float16, device=device)
nz = torch_npu.npu_format_cast(w, 29)

for m in (2048, 4096):
    torch.manual_seed(1200 + rank*17 + m)
    x = torch.randn(m, K, dtype=torch.float16, device=device)
    ref = F.linear(x, w)
    out = torch.empty_like(ref)
    for _ in range(2):
        rc = fn(0, 0, 0, m, 0, 0, raw)
        assert rc == 0, rc
    stream.synchronize()
    ok_all = True
    for it in range(5):
        rc = fn(x.data_ptr(), nz.data_ptr(), out.data_ptr(), m, 0, 1, raw)
        assert rc == 0, rc
        stream.synchronize()
        if torch.equal(out, ref):
            print(f"rank{rank} M{m} iter{it} EQUAL", flush=True)
        else:
            ok_all = False
            diff = (out.float() - ref.float()).abs()
            n_bad = int((diff > 0).sum().item())
            print(f"rank{rank} M{m} iter{it} MISMATCH n_bad={n_bad}/{m*N} "
                  f"max_abs={diff.max().item():.6f} "
                  f"out_absmax={out.float().abs().max().item():.4f} "
                  f"ref_absmax={ref.float().abs().max().item():.4f}", flush=True)
            if it == 0:
                bad = (diff > 0)
                for blk in range(0, m, 1024):
                    cnt = int(bad[blk:blk+1024].sum().item())
                    print(f"  rank{rank} rows[{blk}:{blk+1024}] bad={cnt}", flush=True)
                badflat = bad.nonzero()
                if len(badflat):
                    r0 = int(badflat[0][0])
                    print(f"  first bad row={r0} col={int(badflat[0][1])}", flush=True)
                    nzr = int((out[r0].float().abs().sum() == 0).item())
                    print(f"  out row {r0} all-zero={nzr}", flush=True)
                    # is the row equal to some other ref row?
                    hit = None
                    for s in (-2,-1,1,2,1024,-1024,2048,-2048):
                        rr = r0+s
                        if 0 <= rr < m and torch.equal(out[r0], ref[rr]):
                            hit = s; break
                    print(f"  row-shift match: {hit}", flush=True)
    print(f"rank{rank} M{m} overall ok={ok_all}", flush=True)
    dist.barrier()
dist.destroy_process_group()
