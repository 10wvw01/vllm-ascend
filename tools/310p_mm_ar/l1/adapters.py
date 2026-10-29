# SPDX-License-Identifier: Apache-2.0
"""Version-specific execution only; scheduling/timing/statistics belong to the common framework."""
import ctypes
import importlib.util
import sys
from pathlib import Path

from .core import ROOT, catalog


class Trial:
    def __init__(self, spec, entry, ctx):
        self.spec, self.entry, self.ctx = spec, entry, ctx
        self.lib = ctypes.CDLL(spec["library"], mode=ctypes.RTLD_LOCAL)
        self.handle = ctypes.c_void_p()
        self.raw = ctypes.c_void_p(ctx.stream.npu_stream)
        p, u, q = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64
        signatures = {"create": [ctypes.c_int, ctypes.c_char_p, u, ctypes.POINTER(p)],
                      "ib_probe": [p, u, p], "run": [p, q, q, q, u, u, p],
                      "run_ar_only": [p, q, q, u, u, p], "check_status": [p, ctypes.POINTER(q)], "destroy": [p]}
        for name, args in signatures.items():
            fn = getattr(self.lib, "mf310p_v2_trial_" + name)
            fn.argtypes, fn.restype = args, ctypes.c_int
        self.config = {**entry.get("config", {}), **spec.get("config", {})}
        self.check(self.lib.mf310p_v2_trial_create(ctx.rank, ctx.url.encode(), self.config.get("chunk_rows", 1024),
                                                ctypes.byref(self.handle)))
        self.check(self.lib.mf310p_v2_trial_ib_probe(self.handle, 64, self.raw))
        self.outputs = {m: [ctx.torch.empty((m, 2048), dtype=ctx.torch.float16, device=ctx.device) for _ in range(2)]
                        for m in ctx.shapes}
        self.private = None
        self.pending = []
        if entry["runtime"] == "v2_private":
            dll = ctypes.CDLL(spec["private_library"], mode=ctypes.RTLD_LOCAL)
            self.private = getattr(dll, entry["symbol"])
            self.private.argtypes = [p, q, q, q, u, u, p]
            self.private.restype = ctypes.c_int
            self.private_library = dll

    @staticmethod
    def check(rc):
        if rc:
            raise RuntimeError(f"V2 rc={rc}; fail-stop, no ADD retry")

    def run(self, x, w, variant):
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

    def after_sample(self):
        status = ctypes.c_uint64()
        self.check(self.lib.mf310p_v2_trial_check_status(self.handle, ctypes.byref(status)))
        self.check(status.value)
        self.pending.clear()
        return {}

    def close(self):
        self.ctx.stream.synchronize()
        self.check(self.lib.mf310p_v2_trial_destroy(self.handle))


class Native:
    def __init__(self, ctx):
        self.ctx = ctx

    def run(self, x, w, variant):
        out = self.ctx.linear(x, w)
        self.ctx.dist.all_reduce(out)
        return out

    def after_sample(self):
        return {}

    def close(self):
        pass


class V1(Native):
    def __init__(self, spec, entry, ctx):
        super().__init__(ctx)
        ctx.torch.ops.load_library(spec["library"])
        self.config = {**entry.get("config", {}), **spec.get("config", {})}

    def run(self, x, w, variant):
        # The Tensor custom op performs its own checks, pool acquisition, MM/AR and output wrapping.
        return self.ctx.torch.ops._C_ascend.memfabric_mm_ar_allreduce(
            x, w, self.ctx.rank, self.config.get("batch_basem_count", 1))

    def close(self):
        self.ctx.stream.synchronize()
        self.ctx.torch.ops._C_ascend.memfabric_mm_ar_shutdown()


class V3(Native):
    def __init__(self, spec, entry, ctx, collection):
        super().__init__(ctx)
        sys.path.insert(0, str(ROOT / "vllm_ascend/_310p/ops"))
        from memfabric_mm_ar_v3 import V3Session

        self.entry = entry
        self.config = {**entry.get("config", {}), **spec.get("config", {})}
        kind = entry["runtime"]
        if kind in ("v32", "a3", "a3-2", "a3-tail"):
            module_spec = importlib.util.spec_from_file_location("mf310p_v32_tensor", spec["tensor_adapter"])
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            if kind == "v32":
                from memfabric_mm_ar_v32 import V32Session
                self.session = V32Session(spec["library"], ctx.rank, ctx.url, tensor_module=module)
                self.session.merged_wave = True
            else:
                from memfabric_mm_ar_a3 import A3Session
                self.session = A3Session(spec["library"], ctx.rank, ctx.url, tensor_module=module, variant=kind)
        else:
            self.session = V3Session(spec["library"], ctx.rank, ctx.url)
        self.session.ib_probe()
        src = ctx.torch.ones((512, 2048), dtype=ctx.torch.float16, device=ctx.device)
        seed, out = src.clone(), ctx.torch.empty_like(src)
        self.session.probe_add(src, seed, out)
        self.session.synchronize_check()
        if not ctx.torch.equal(out, src + seed):
            raise AssertionError("raw ADD gate")
        if kind.startswith("a3"):
            self.session.initialize()
        if collection == "host" and kind in ("v32", "a3", "a3-2", "a3-tail"):
            self.session.phase_log = []
        self.outputs = {m: [ctx.torch.empty((m, 2048), dtype=ctx.torch.float16, device=ctx.device) for _ in range(2)]
                        for m in ctx.shapes}

    def run(self, x, w, variant):
        kind, conf = self.entry["runtime"], self.config
        if kind.startswith("a3"):
            return self.session.run_pool(x, w, compute_rows=conf.get("compute_rows", 1024),
                                         comm_rows=conf.get("comm_rows", 1024))
        if kind == "v32":
            return self.session.run_pool(x, w, group=conf.get("comm_rows", 1024))
        return self.session.run(x, w, self.outputs[x.shape[0]][variant], group=conf.get("comm_rows", 1024),
                                pipeline=conf.get("pipeline", True))

    def after_sample(self):
        self.session.synchronize_check()
        phases = getattr(self.session, "phase_log", None)
        return {"host_us": phases.pop()} if phases else {}

    def close(self):
        self.session.close()


def create(version, spec, ctx, collection):
    if version == "STOCK":
        return Native(ctx)
    entry = catalog()[version]
    kind = entry["runtime"]
    if kind == "external":
        module_spec = importlib.util.spec_from_file_location("frozen_l1_adapter", spec["factory_file"])
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        adapter = module.create_adapter(ctx=ctx, specification=spec)
        if adapter.version != version:
            raise ValueError("external factory returned the wrong VN-M")
        return adapter
    if kind == "v1":
        return V1(spec, entry, ctx)
    if kind.startswith("v2"):
        return Trial(spec, entry, ctx)
    return V3(spec, entry, ctx, collection)
