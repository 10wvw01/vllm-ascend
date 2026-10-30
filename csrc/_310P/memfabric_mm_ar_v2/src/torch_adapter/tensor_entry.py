# SPDX-License-Identifier: Apache-2.0
"""Operator-owned V2 Tensor entries over the preserved base/private/selected ABIs."""
import ctypes
from mf310p_tensor_entry import TensorOperator, load_tensor_bridge

class Trial(TensorOperator):
    def __init__(self, spec, entry, ctx):
        bridge = load_tensor_bridge(spec)
        self.spec, self.entry, self.ctx = spec, entry, ctx
        self.lib = ctypes.CDLL(spec["library"], mode=ctypes.RTLD_LOCAL)
        self.handle = ctypes.c_void_p()
        self.raw = ctypes.c_void_p(ctx.stream.npu_stream)
        p, u, q = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64
        signatures = {"create": [ctypes.c_int, ctypes.c_char_p, u, ctypes.POINTER(p)],
                      "ib_probe": [p, u, p], "run": [p, q, q, q, u, u, p],
                      "check_status": [p, ctypes.POINTER(q)], "destroy": [p]}
        if entry["runtime"] == "v2_b1":
            signatures["run_ar_only"] = [p, q, q, u, u, p]
        for name, args in signatures.items():
            fn = getattr(self.lib, "mf310p_v2_trial_" + name)
            fn.argtypes, fn.restype = args, ctypes.c_int
        self.config = {**entry.get("config", {}), **spec.get("config", {})}
        self.outputs = {m: [ctx.torch.empty((m, 2048), dtype=ctx.torch.float16, device=ctx.device) for _ in range(getattr(ctx,'output_slots',2))]
                        for m in ctx.shapes}
        ctx.stream.synchronize()
        self.check(self.lib.mf310p_v2_trial_create(ctx.rank, ctx.url.encode(), self.config.get("chunk_rows", 1024),
                                                ctypes.byref(self.handle)))
        self.check(self.lib.mf310p_v2_trial_ib_probe(self.handle, 64, self.raw))
        self.private = None
        self.pending = []
        if entry["runtime"] == "v2_private":
            dll = ctypes.CDLL(spec["private_library"], mode=ctypes.RTLD_LOCAL)
            self.private = getattr(dll, entry["symbol"])
            self.private.argtypes = [p, q, q, q, u, u, p]
            self.private.restype = ctypes.c_int
            self.private_library = dll
        self._publish(ctx, bridge, self.outputs)

    def _take_backend_inputs(self):
        pending = tuple(self.pending)
        self.pending.clear()
        return pending

    @staticmethod
    def check(rc):
        if rc:
            raise RuntimeError(f"V2 rc={rc}; fail-stop, no ADD retry")

    def _backend_run(self, x, w, variant):
        out = self.outputs[x.shape[0]][variant]
        if self.private is not None:
            self.check(self.private(self.handle, x.data_ptr(), w.data_ptr(), out.data_ptr(), x.shape[0],
                                    self.entry["kind"], self.raw))
        elif self.entry["runtime"] == "v2_b1":
            partial = self.ctx.linear(x, w)
            self.ctx.stream.synchronize()  # Required by historical B1, inside timed call.
            self.pending.append(partial)
            self.check(self.lib.mf310p_v2_trial_run_ar_only(self.handle, partial.data_ptr(), out.data_ptr(),
                                                          x.shape[0], 8, self.raw))
        else:
            self.check(self.lib.mf310p_v2_trial_run(self.handle, x.data_ptr(), w.data_ptr(), out.data_ptr(),
                                                  x.shape[0], self.config.get("overlap", 1), self.raw))
        return out

    def _after_backend(self):
        status = ctypes.c_uint64()
        self.check(self.lib.mf310p_v2_trial_check_status(self.handle, ctypes.byref(status)))
        self.check(status.value)
        self.pending.clear()
        return {}

    def _close_backend(self):
        self.ctx.stream.synchronize()
        self.check(self.lib.mf310p_v2_trial_destroy(self.handle))



class SelectedV2(TensorOperator):
    """Fixed-shape selected C API; historical private adapters remain available."""
    def __init__(self, spec, entry, ctx):
        bridge = load_tensor_bridge(spec)
        expected = {"V2-7": 2048, "V2-15": 4096}
        self.m = expected[entry["id"]]
        if list(ctx.shapes) != [self.m]:
            raise ValueError("selected V2 requires one matching fixed M per plan")
        self.ctx = ctx
        # Pin outputs before creating the MemFabric pool; no late allocations.
        self.outputs = [ctx.torch.empty((self.m, 2048), dtype=ctx.torch.float16, device=ctx.device)
                         for _ in range(getattr(ctx,'output_slots',2))]
        ctx.stream.synchronize()
        self.lib = ctypes.CDLL(spec["library"], mode=ctypes.RTLD_LOCAL)
        self.handle = ctypes.c_void_p()
        self.raw = ctypes.c_void_p(ctx.stream.npu_stream)
        p, u, q = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64
        signatures = {"create": [ctypes.c_int, ctypes.c_char_p, u, ctypes.POINTER(p)],
                      "ib_probe": [p, u, p], "run": [p, q, q, q, p],
                      "check_status": [p, ctypes.POINTER(q)], "destroy": [p]}
        for name, args in signatures.items():
            fn = getattr(self.lib, "mf310p_v2_selected_" + name)
            fn.argtypes, fn.restype = args, ctypes.c_int
        Trial.check(self.lib.mf310p_v2_selected_create(ctx.rank, ctx.url.encode(), self.m,
                                                    ctypes.byref(self.handle)))
        Trial.check(self.lib.mf310p_v2_selected_ib_probe(self.handle, 64, self.raw))
        self._publish(ctx, bridge, self.outputs)

    def _backend_run(self, x, w, variant):
        if tuple(x.shape) != (self.m, 2048):
            raise ValueError("selected V2 input differs from its fixed M")
        out = self.outputs[variant]
        Trial.check(self.lib.mf310p_v2_selected_run(
            self.handle, x.data_ptr(), w.data_ptr(), out.data_ptr(), self.raw))
        return out

    def _after_backend(self):
        status = ctypes.c_uint64()
        Trial.check(self.lib.mf310p_v2_selected_check_status(self.handle, ctypes.byref(status)))
        Trial.check(status.value)
        return {}

    def _close_backend(self):
        Trial.check(self.lib.mf310p_v2_selected_destroy(self.handle))
        self.handle = ctypes.c_void_p()

