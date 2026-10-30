# SPDX-License-Identifier: Apache-2.0
"""V3 iteration adapters: invoke frozen sessions; keep all timing in the worker."""
import hashlib
import importlib.util
import sys
import types
from pathlib import Path

from ..core import ROOT, sha
from .stock import Native

ROLES = {
    'v3': ('memfabric_mm_ar_v3',),
    'v32': ('memfabric_mm_ar_v3', 'memfabric_mm_ar_v32'),
    'a3': ('memfabric_mm_ar_v3', 'memfabric_mm_ar_a3'),
}


def family(entry):
    kind = entry.get('adapter_family', entry['runtime'])
    return 'a3' if kind in ('a3', 'a3-2', 'a3-tail') else kind


def load_sessions(spec, entry):
    """Use a private package so relative imports cannot pick another iteration."""
    roles = ROLES[family(entry)]
    explicit = spec.get('api') == 'python_session_v1'
    paths = spec.get('python_modules') if explicit else {
        role: str(ROOT/'vllm_ascend/_310p/ops'/f'{role}.py') for role in roles}
    if not paths or set(paths) != set(roles):
        raise ValueError('exact frozen Python session module roles required')
    identities = []
    for role in roles:
        path = Path(paths[role]).resolve(strict=True)
        digest = sha(path)
        if explicit and spec.get('files', {}).get(str(path)) != digest:
            raise ValueError('Python session is absent from frozen files or changed')
        identities.append((role, str(path), digest))
    package_name = '_mf310p_l1_' + hashlib.sha256(repr(identities).encode()).hexdigest()[:20]
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = []
        sys.modules[package_name] = package
    modules = {}
    for role, path, _ in identities:
        name = package_name + '.' + role
        if name not in sys.modules:
            descriptor = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(descriptor)
            sys.modules[name] = module
            try:
                descriptor.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(name, None)
                raise
        modules[role] = sys.modules[name]
    return modules


class V3(Native):
    def __init__(self, spec, entry, ctx, collection):
        super().__init__(ctx)
        self.entry = entry
        self.config = {**entry.get('config', {}), **spec.get('config', {})}
        self.family = family(entry)
        # Allocate raw outputs and ADD-probe storage before pool creation.
        self.outputs = {m: [ctx.torch.empty((m, 2048), dtype=ctx.torch.float16, device=ctx.device)
                            for _ in range(2)] for m in ctx.shapes} if self.family == 'v3' else {}
        src = ctx.torch.ones((512, 2048), dtype=ctx.torch.float16, device=ctx.device)
        seed, out = src.clone(), ctx.torch.empty_like(src)
        ctx.stream.synchronize()
        modules = load_sessions(spec, entry)
        tensor = None
        if self.family in ('v32', 'a3'):
            descriptor = importlib.util.spec_from_file_location('mf310p_v32_tensor', spec['tensor_adapter'])
            tensor = importlib.util.module_from_spec(descriptor)
            descriptor.loader.exec_module(tensor)
        common = dict(packet_bytes=self.config.get('packet_bytes', 4 << 20))
        if self.family == 'v3':
            self.session = modules['memfabric_mm_ar_v3'].V3Session(spec['library'], ctx.rank, ctx.url, **common)
        elif self.family == 'v32':
            self.session = modules['memfabric_mm_ar_v32'].V32Session(
                spec['library'], ctx.rank, ctx.url, tensor_module=tensor, **common)
            merged = self.config.get('merged_wave', True)
            if merged and not hasattr(self.session, 'merged_wave'):
                raise ValueError('frozen session does not implement merged-wave')
            if hasattr(self.session, 'merged_wave'):
                self.session.merged_wave = merged
        else:
            variant = entry.get('adapter_variant', entry['runtime'])
            extra = {'variant': variant} if variant in ('a3-2', 'a3-tail') else {}
            self.session = modules['memfabric_mm_ar_a3'].A3Session(
                spec['library'], ctx.rank, ctx.url, tensor_module=tensor, **common, **extra)
        self.session.ib_probe()
        self.session.probe_add(src, seed, out)
        self.session.synchronize_check()
        if not ctx.torch.equal(out, src + seed):
            raise AssertionError('raw ADD gate')
        if self.family == 'a3':
            self.session.initialize()
        if collection == 'host' and self.family in ('v32', 'a3'):
            self.session.phase_log = []

    def run(self, x, w, variant):
        conf = self.config
        if self.family == 'a3':
            return self.session.run_pool(x, w, compute_rows=conf['compute_rows'], comm_rows=conf['comm_rows'])
        if self.family == 'v32':
            return self.session.run_pool(x, w, group=conf['comm_rows'])
        return self.session.run(x, w, self.outputs[x.shape[0]][variant],
                                group=conf['comm_rows'], pipeline=conf['pipeline'])

    def after_sample(self):
        self.session.synchronize_check()
        phases = getattr(self.session, 'phase_log', None)
        return {'host_us': phases.pop()} if phases else {}

    def close(self):
        self.session.close()
