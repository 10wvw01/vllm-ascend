#!/usr/bin/env python3
# A1 MM shadow correctness failure diagnosis: characterize mismatch pattern.
import ctypes, sys
import torch
import torch.nn.functional as F
import torch_npu

torch.npu.set_device(0)
torch.npu.config.allow_internal_format = True
torch_npu.npu.set_compile_mode(jit_compile=False)
device = torch.device("npu:0")

K = N = 2048
LIB = "/tmp/opencode/a1-mm-shadow-20261027/precheck/build/libmf310p_v2_mm_shadow.so"
lib = ctypes.CDLL(LIB)
fn = lib.mf310p_v2_mm_shadow_launch_async
fn.argtypes = [ctypes.c_uint64]*3 + [ctypes.c_uint32]*3 + [ctypes.c_void_p]
fn.restype = ctypes.c_int

stream = torch.npu.current_stream()
ptr = ctypes.c_void_p(stream.npu_stream)

torch.manual_seed(4242)
w = torch.randn(N, K, dtype=torch.float16, device=device)
w_nz = torch_npu.npu_format_cast(w, 29)
print("w_nz shape:", tuple(w_nz.shape), "stride:", w_nz.stride(),
      "storage numel:", w_nz.untyped_storage().nbytes()//2,
      "ptr:", hex(w_nz.data_ptr()))

# symbol warmup (enable=0)
for _ in range(2):
    rc = fn(0, 0, 0, 2048, 0, 0, ptr)
    assert rc == 0, rc
stream.synchronize()

for m in (2048, 4096):
    torch.manual_seed(1200 + m)
    x = torch.randn(m, K, dtype=torch.float16, device=device)
    ref = F.linear(x, w)
    out = torch.empty((m, N), dtype=torch.float16, device=device)
    rc = fn(x.data_ptr(), w_nz.data_ptr(), out.data_ptr(), m, 0, 1, ptr)
    assert rc == 0, rc
    stream.synchronize()
    equal = torch.equal(out, ref)
    diff = (out.float() - ref.float()).abs()
    n_bad = int((diff > 0).sum().item())
    print(f"M={m} equal={equal} n_bad={n_bad}/{m*N} "
          f"max_abs={diff.max().item():.6f} "
          f"out_absmax={out.float().abs().max().item():.4f} "
          f"ref_absmax={ref.float().abs().max().item():.4f}")
    if not equal and n_bad > 0:
        bad = (diff > 0).nonzero()
        print("  first bad idx:", bad[:5].tolist())
        # per-1024-row block bad counts
        for blk in range(0, m, 1024):
            cnt = int((diff[blk:blk+1024] > 0).sum().item())
            print(f"  rows[{blk}:{blk+1024}] bad={cnt}")
        # per-column-half pattern for first bad row
        r = int(bad[0][0].item())
        rowbad = (diff[r] > 0).nonzero().flatten().tolist()
        print(f"  row {r}: bad cols head={rowbad[:10]} count={len(rowbad)}")
        # check if out rows are a permutation / shifted
        # compare row r of out with nearby rows of ref
        for shift in (-2, -1, 0, 1, 2):
            rr = r + shift
            if 0 <= rr < m and torch.equal(out[r], ref[rr]):
                print(f"  MATCH: out row {r} == ref row {rr} (shift {shift})")
        # check zero rows in out
        zero_rows = int((out.float().abs().sum(dim=1) == 0).sum().item())
        print("  zero rows in out:", zero_rows)
