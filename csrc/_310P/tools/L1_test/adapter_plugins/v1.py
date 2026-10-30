# SPDX-License-Identifier: Apache-2.0
"""One Tensor call into the explicitly frozen V1 operator."""
from .stock import Native

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
