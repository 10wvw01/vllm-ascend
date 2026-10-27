#!/usr/bin/env python3
# Raw-byte dump of w_nz buffer around F.linear(x, w) via aclrtMemcpy (no torch format ops).
import ctypes
import numpy as np
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
acl = ctypes.CDLL("libascendcl.so")
P = ctypes.c_void_p
acl.aclrtMemcpy.argtypes = [P, ctypes.c_size_t, P, ctypes.c_size_t, ctypes.c_int]
acl.aclrtMemcpy.restype = ctypes.c_int
acl.aclrtSynchronizeDevice.restype = ctypes.c_int

stream = torch.npu.current_stream()
raw = ctypes.c_void_p(stream.npu_stream)
m = 2048

def raw_read(ptr, nbytes):
    buf = ctypes.create_string_buffer(nbytes)
    rc = acl.aclrtMemcpy(buf, nbytes, P(ptr), nbytes, 2)
    assert rc == 0, rc
    return np.frombuffer(buf, dtype=np.uint8).copy()

NZ_BYTES = N * K * 2
torch.manual_seed(4242)
w = torch.randn(N, K, dtype=torch.float16, device=device)
torch.manual_seed(1200)
x = torch.randn(m, K, dtype=torch.float16, device=device)

w_nz = torch_npu.npu_format_cast(w, 29)
nz_ptr = w_nz.data_ptr()
print("after cast: nz_ptr:", hex(nz_ptr), "w_ptr:", hex(w.data_ptr()))
rc = acl.aclrtSynchronizeDevice(); assert rc == 0
snapA = raw_read(nz_ptr, NZ_BYTES)
print("snapA nonzero bytes:", int((snapA != 0).sum()), "/", NZ_BYTES,
      "first fp16s:", np.frombuffer(snapA.tobytes(), dtype="<f2")[:8])

ref = F.linear(x, w)
rc = acl.aclrtSynchronizeDevice(); assert rc == 0
snapB = raw_read(nz_ptr, NZ_BYTES)
print("snapB nonzero bytes:", int((snapB != 0).sum()), "/", NZ_BYTES,
      "first fp16s:", np.frombuffer(snapB.tobytes(), dtype="<f2")[:8])
print("A==B:", bool((snapA == snapB).all()))
print("nz_ptr still:", hex(w_nz.data_ptr()), "same as captured:", w_nz.data_ptr() == nz_ptr)

y = F.linear(x, w_nz)
rc = acl.aclrtSynchronizeDevice(); assert rc == 0
print("native F.linear(x, w_nz) == ref:", torch.equal(y, ref))

out = torch.empty_like(ref)
rc = fn(0, 0, 0, m, 0, 0, raw); assert rc == 0
rc = fn(x.data_ptr(), w_nz.data_ptr(), out.data_ptr(), m, 0, 1, raw); assert rc == 0
rc = acl.aclrtSynchronizeDevice(); assert rc == 0
print("shadow equal ref:", torch.equal(out, ref))
snapC = raw_read(nz_ptr, NZ_BYTES)
print("A==C:", bool((snapA == snapC).all()))

# reference NZ bytes from a second independent cast in same process
w_nz2 = torch_npu.npu_format_cast(w, 29)
rc = acl.aclrtSynchronizeDevice(); assert rc == 0
snapD = raw_read(w_nz2.data_ptr(), NZ_BYTES)
print("A==D (second cast same buffer?):", bool((snapA == snapD).all()),
      "ptr2:", hex(w_nz2.data_ptr()))
