#!/usr/bin/env python3
# Buffer archaeology: which buffer holds NZ data and when.
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

acl = ctypes.CDLL("libascendcl.so")
P = ctypes.c_void_p
acl.aclrtMemcpy.argtypes = [P, ctypes.c_size_t, P, ctypes.c_size_t, ctypes.c_int]
acl.aclrtMemcpy.restype = ctypes.c_int
acl.aclrtSynchronizeDevice.restype = ctypes.c_int

def raw_read(ptr, nbytes):
    buf = ctypes.create_string_buffer(nbytes)
    rc = acl.aclrtMemcpy(buf, nbytes, P(ptr), nbytes, 2)
    assert rc == 0, rc
    return np.frombuffer(buf, dtype=np.uint8).copy()

def sync():
    rc = acl.aclrtSynchronizeDevice(); assert rc == 0

NZ_BYTES = N * K * 2
torch.manual_seed(4242)
w = torch.randn(N, K, dtype=torch.float16, device=device)
torch.manual_seed(1200)
x = torch.randn(2048, K, dtype=torch.float16, device=device)

W1 = raw_read(w.data_ptr(), NZ_BYTES); sync()
print("W1 (raw ND w) first fp16s:", np.frombuffer(W1.tobytes(), dtype="<f2")[:6])

w_nz = torch_npu.npu_format_cast(w, 29)
sync()
A = raw_read(w_nz.data_ptr(), NZ_BYTES)
W2 = raw_read(w.data_ptr(), NZ_BYTES)
print("cast lazy? A==W2(ND):", bool((A == W2).all()),
      "| A==W1:", bool((A == W1).all()),
      "| A first fp16s:", np.frombuffer(A.tobytes(), dtype="<f2")[:6])

ref = F.linear(x, w)
sync()
W3 = raw_read(w.data_ptr(), NZ_BYTES)
B = raw_read(w_nz.data_ptr(), NZ_BYTES)
print("after F.linear(x,w): W3==W1 (w unchanged):", bool((W3 == W1).all()),
      "| B==A (nz buf unchanged):", bool((B == A).all()))

y = F.linear(x, w_nz)
sync()
C = raw_read(w_nz.data_ptr(), NZ_BYTES)
print("after native consume: C==A:", bool((C == A).all()),
      "| C==W2 (became ND?):", bool((C == W2).all()),
      "| native bit-exact:", bool(torch.equal(y, ref)))

# What does a KNOWN-good NZ buffer look like? Do an eager-style sequence:
w2 = w.clone()
ref2_pre = F.linear(x, w2)          # consume w2 ND first (internal cast)
w2_nz = torch_npu.npu_format_cast(w2, 29)
sync()
D = raw_read(w2_nz.data_ptr(), NZ_BYTES)
print("T2-style cast buffer D==W2(ND):", bool((D == W2).all()),
      "| D==A:", bool((D == A).all()))
w2_nz2 = torch_npu.npu_format_cast(w2, 29)
sync()
E = raw_read(w2_nz2.data_ptr(), NZ_BYTES)
print("second cast E==D:", bool((E == D).all()), "ptr:", hex(w2_nz2.data_ptr()), hex(w2_nz.data_ptr()))
