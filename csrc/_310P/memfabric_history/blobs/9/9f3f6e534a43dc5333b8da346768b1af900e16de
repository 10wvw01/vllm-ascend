#!/usr/bin/env python3
# MemFabric private host SO (1dc46d7d) real loading precheck:
# dual-rank private pool create/barrier/geometry/layout/destroy via ctypes,
# mirroring a1_mm_shadow_arm.Symmetric exactly. KFC auto-deploys at first pool.
import ctypes, os, sys
import torch
import torch.distributed as dist

sys.path.insert(0, "/tmp/opencode/a1-r0-worktree/csrc/_310P/memfabric_mm_ar_v2/test/perf")
from a1_mm_shadow_arm import Layout, CHUNK

rank = int(os.environ["RANK"]); local = int(os.environ["LOCAL_RANK"])
assert int(os.environ["WORLD_SIZE"]) == 2
torch.npu.set_device(local)
dist.init_process_group(backend="gloo")

LIB = os.environ["U1_CORE_SO_LIB64"]
url = "tcp://127.0.0.1:18885"
lib = ctypes.CDLL(LIB)
P = ctypes.c_void_p
lib.mf310p_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
                              ctypes.c_uint64, ctypes.c_uint32,
                              ctypes.c_uint32, ctypes.POINTER(P)]
lib.mf310p_create.restype = ctypes.c_int
lib.mf310p_control_barrier.argtypes = [P]
lib.mf310p_control_barrier.restype = ctypes.c_int
lib.mf310p_exchange_geometry.argtypes = [P]
lib.mf310p_exchange_geometry.restype = ctypes.c_int
lib.mf310p_get_layout.argtypes = [P, ctypes.POINTER(Layout)]
lib.mf310p_get_layout.restype = ctypes.c_int
lib.mf310p_destroy.argtypes = [P]
lib.mf310p_destroy.restype = ctypes.c_int

ctx = P()
rc = lib.mf310p_create(rank, 2, url.encode(), 96 * 1024 * 1024, 8192, CHUNK,
                       ctypes.byref(ctx))
print(f"rank{rank} mf310p_create rc={rc} ctx={'set' if ctx.value else 'null'}", flush=True)
assert rc == 0 and ctx.value
try:
    for name in ("mf310p_control_barrier", "mf310p_exchange_geometry"):
        rc = getattr(lib, name)(ctx)
        print(f"rank{rank} {name} rc={rc}", flush=True)
        assert rc == 0
    layout = Layout()
    rc = lib.mf310p_get_layout(ctx, ctypes.byref(layout))
    assert rc == 0
    ptr = layout.send_arena if rank == 0 else layout.recv_arena
    print(f"rank{rank} layout: pool_base={hex(layout.pool_base)} "
          f"own={hex(layout.own_segment)} peer={hex(layout.peer_segment)} "
          f"sym_size={layout.symmetric_size} arena_bytes={layout.arena_bytes} "
          f"arena_rows={layout.arena_rows} batch_m={layout.batch_m} "
          f"max_batches={layout.max_batches} arena={hex(ptr)}", flush=True)
    assert ptr and layout.pool_base and layout.arena_bytes >= 4096 * 2048 * 2
    torch.npu.synchronize()
    acl = ctypes.CDLL("libascendcl.so")
    rc = acl.aclrtSynchronizeDevice()
    print(f"rank{rank} protected first device sync rc={rc}", flush=True)
    assert rc == 0
    dist.barrier()
    print(f"rank{rank} SYM_POOL_PRECHECK_PASS", flush=True)
finally:
    rc = lib.mf310p_destroy(ctx)
    print(f"rank{rank} mf310p_destroy rc={rc}", flush=True)
    assert rc == 0
dist.destroy_process_group()
