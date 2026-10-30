# SPDX-License-Identifier: Apache-2.0
"""Public V3 Torch Tensor entries; frozen algorithm sessions remain unchanged."""
import hashlib
import importlib.util
import sys
import types
from pathlib import Path

from mf310p_tensor_entry import ROOT, TensorOperator, load_tensor_bridge

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

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
    package_name = '_mf310p_sessions_' + hashlib.sha256(repr(identities).encode()).hexdigest()[:20]
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


class V3(TensorOperator):
    def __init__(self, spec, entry, ctx, collection):
        self.ctx = ctx
        bridge = load_tensor_bridge(spec)
        self.entry = entry
        self.config = {**entry.get('config', {}), **spec.get('config', {})}
        self.family = family(entry)
        # Allocate raw outputs and ADD-probe storage before pool creation.
        self.outputs = {m: [ctx.torch.empty((m, 2048), dtype=ctx.torch.float16, device=ctx.device)
                            for _ in range(getattr(ctx,'output_slots',2))] for m in ctx.shapes} if self.family == 'v3' else {}
        src = ctx.torch.ones((512, 2048), dtype=ctx.torch.float16, device=ctx.device)
        seed, out = src.clone(), ctx.torch.empty_like(src)
        ctx.stream.synchronize()
        modules = load_sessions(spec, entry)
        tensor = bridge
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
        self._publish(ctx, bridge, self.outputs if self.family == 'v3' else None)

    def _backend_run(self, x, w, variant):
        conf = self.config
        if self.family == 'a3':
            return self.session.run_pool(x, w, compute_rows=conf['compute_rows'], comm_rows=conf['comm_rows'])
        if self.family == 'v32':
            return self.session.run_pool(x, w, group=conf['comm_rows'])
        return self.session.run(x, w, self.outputs[x.shape[0]][variant],
                                group=conf['comm_rows'], pipeline=conf['pipeline'])

    def _take_backend_inputs(self):
        refs = tuple(self.session.pending)
        self.session.pending.clear()
        return refs

    def _after_backend(self):
        self.session.synchronize_check()
        phases = getattr(self.session, 'phase_log', None)
        calls = list(phases) if phases else []
        if phases is not None:
            phases.clear()
        return {'host_calls': calls} if calls else {}

    def _close_backend(self):
        self.session.close()
