# SPDX-License-Identifier: Apache-2.0
# Single-card T1 MM isolation: mf310p_v2_diag_nzdirect_mm_async vs F.linear
# (original-NZ N-split + direct-ND SetOrgShape/SetSingleShape contract).
# No MemFabric pool, no distributed, no IB/SDMA - pure matmul correctness.
# Set DIAG_PRESYNC=1 to heal the first-MTE3-write drop hazard (see
# T1_NZ_DIRECT_REPORT.md); without it, fresh-buffer results are unreliable.
# Usage: python t1_diag_single.py [m ...]   (default boundary sweep)
# Single-card T1 MM isolation: mf310p_v2_diag_nzdirect_mm_async vs F.linear.
# No MemFabric pool, no distributed, no IB/SDMA - pure matmul correctness of
# the original-NZ N-split + direct-ND SetOrgShape/SetSingleShape path.
import ctypes
import os
import sys

import torch
import torch.nn.functional as F
import torch_npu

ACL_FORMAT_FRACTAL_NZ = 29
try:
    import vllm_ascend.vllm_ascend_C as _ascend_ext

    _DEFAULT_LIB = _ascend_ext.__file__
except Exception:
    _DEFAULT_LIB = "/tmp/opencode/v2/libmf310p_v2_trial_device.so"
DEVLIB = os.environ.get("V2_DEVLIB", _DEFAULT_LIB)


def main():
    dev = int(os.environ.get("DIAG_DEV", "0"))
    torch.npu.set_device(dev)
    device = torch.device(f"npu:{dev}")
    lib = ctypes.CDLL(DEVLIB)
    lib.mf310p_v2_diag_nzdirect_mm_async.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    lib.mf310p_v2_diag_nzdirect_mm_async.restype = ctypes.c_int
    stream = torch.npu.current_stream().npu_stream

    torch.manual_seed(4242)
    w = torch.randn(2048, 2048, dtype=torch.float16, device=device)
    w_nz = torch_npu.npu_format_cast(w, ACL_FORMAT_FRACTAL_NZ)

    def templ(m):
        for t in (16, 32, 64, 128, 256):
            if m <= t:
                return t
        raise ValueError(m)

    # order: safe values first, known-crash (m=256, mm=1) last
    plan = []
    for m in [int(a) for a in sys.argv[1:]] or [16, 32, 64, 128, 1, 2, 3, 15, 17, 31, 33, 63, 65, 127, 129, 254, 255]:
        for mm in (1, 2, 4, 8):
            plan.append((m, mm))
    for mm in (8, 4, 2, 1):
        plan.append((256, mm))

    n_pass = n_fail = 0
    for m, mm in plan:
        t = templ(m)
        torch.manual_seed(1200 + m)
        x = torch.randn(m, 2048, dtype=torch.float16, device=device)
        ref = F.linear(x, w)
        y = torch.zeros(m, 2048, dtype=torch.float16, device=device)
        if os.environ.get("DIAG_PRESYNC"):
            # T0 BUG#3 hazard probe: heal first-MTE3-write drops into a
            # freshly allocated y with an explicit pre-launch sync.
            torch.npu.current_stream().synchronize()
        # symbol warmup x2 (enable=0), then the real launch
        for _ in range(2):
            rc = lib.mf310p_v2_diag_nzdirect_mm_async(x.data_ptr(), w_nz.data_ptr(), y.data_ptr(), t, m, mm, 0, stream)
            assert rc == 0, f"launch rc={rc} (warm)"
        rc = lib.mf310p_v2_diag_nzdirect_mm_async(x.data_ptr(), w_nz.data_ptr(), y.data_ptr(), t, m, mm, 1, stream)
        assert rc == 0, f"launch rc={rc}"
        torch.npu.current_stream().synchronize()
        if torch.equal(y, ref):
            n_pass += 1
            print(f"DIAGNZ\tT={t}\tm={m}\tmm={mm}\tPASS\ty_abs_max={y.float().abs().max().item():.3f}", flush=True)
        else:
            diff = (y.float() - ref.float()).abs()
            bad = diff > 0
            n_fail += 1
            print(
                f"DIAGNZ\tT={t}\tm={m}\tmm={mm}\tFAIL\t"
                f"max_abs={diff.max().item():.4f}\t"
                f"n_bad={int(bad.sum().item())}\t"
                f"y_abs_max={y.float().abs().max().item():.3f}\t"
                f"ref_abs_max={ref.float().abs().max().item():.3f}",
                flush=True,
            )
    print(f"DIAGNZ\tsummary\tpass={n_pass}\tfail={n_fail}", flush=True)


if __name__ == "__main__":
    main()
