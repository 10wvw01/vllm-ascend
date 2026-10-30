# SPDX-License-Identifier: Apache-2.0
"""Native NZ MM plus HCCL SUM, with no MemFabric dependency."""

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
