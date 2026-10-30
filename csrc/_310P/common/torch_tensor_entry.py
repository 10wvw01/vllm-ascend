# SPDX-License-Identifier: Apache-2.0
"""Public eager Torch operator boundary shared by the 310P V2/V3 implementations.

Persistent communication setup is explicit, like creating an HCCL process group.
All per-call validation, output acquisition/wrapping and input lifetime work is
inside torch.ops.memfabric310p.mm_ar. No benchmark module is imported here.
"""
import hashlib
import importlib.util
import itertools
import sys
import threading
import types
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.modules.setdefault('mf310p_tensor_entry', sys.modules[__name__])
sys.modules.setdefault('torch_tensor_entry', sys.modules[__name__])
_LIBRARY = None
_CONTEXTS = {}  # Explicit close owns lifetime; never free an in-flight pool in GC.
_TOKENS = itertools.count(1)
_LOCK = threading.RLock()


def _dispatch(x, weight, context):
    with _LOCK:
        owner = _CONTEXTS.get(context)
    if owner is None:
        raise RuntimeError('closed or unknown MemFabric Tensor operator context')
    return owner._run_tensor(x, weight)


def prepare_dispatch(torch):
    global _LIBRARY
    with _LOCK:
        if _LIBRARY is None:
            library = torch.library.Library('memfabric310p', 'FRAGMENT')
            library.define('mm_ar(Tensor x, Tensor weight, int context) -> Tensor')
            library.impl('mm_ar', _dispatch, 'CompositeExplicitAutograd')
            _LIBRARY = library


def load_module(path, prefix):
    path = Path(path).resolve(strict=True)
    name = prefix + hashlib.sha256(path.read_bytes()).hexdigest()[:20]
    if name not in sys.modules:
        descriptor = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(descriptor)
        sys.modules[name] = module
        try:
            descriptor.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def load_tensor_bridge(spec):
    path = Path(spec['tensor_adapter']).resolve(strict=True)
    package = '_mf310p_storage_' + hashlib.sha256(path.read_bytes()).hexdigest()[:20]
    if package not in sys.modules:
        parent = types.ModuleType(package)
        parent.__path__ = []
        sys.modules[package] = parent
    # The final component must match PyInit_mf310p_v32_tensor for the SO.
    name = package + '.' + path.name.split('.')[0]
    if name not in sys.modules:
        descriptor = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(descriptor)
        sys.modules[name] = module
        try:
            descriptor.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name,None)
            raise
    module = sys.modules[name]
    if not callable(getattr(module, 'wrap', None)):
        raise RuntimeError('TorchNPU Storage bridge must expose wrap')
    return module


class TensorOperator:
    """A real dispatcher entry, with persistent internal buffers owned by the op."""
    def _publish(self, ctx, bridge, raw_outputs):
        prepare_dispatch(ctx.torch)
        self.ctx = ctx
        self._bridge = bridge
        self._raw_outputs = raw_outputs
        self._entry_lock = threading.RLock()
        self._entry_failed = False
        self._entry_closed = False
        self._leases = {}
        self._cursor = {}
        self._inflight = deque()
        self._uncertain_inputs = []
        self._events = []
        if raw_outputs is not None:
            for m in ctx.shapes:
                bases = raw_outputs[m] if isinstance(raw_outputs, dict) else raw_outputs
                self._leases[m] = [False] * len(bases)
                self._cursor[m] = 0
            # Event first-touch is cold setup; input references remain alive until
            # the owning stream passes the recorded completion event.
            self._events = [ctx.torch.npu.Event() for _ in range(64)]
            for event in self._events:
                event.record(ctx.stream)
            ctx.stream.synchronize()
        with _LOCK:
            self._token = next(_TOKENS)
            _CONTEXTS[self._token] = self

    def run(self, x, weight, variant=None):
        # variant is accepted only for caller compatibility. The operator owns
        # output selection; a test script cannot choose or pre-wrap its output.
        return self.ctx.torch.ops.memfabric310p.mm_ar(x, weight, self._token)

    def _validate(self, x, weight):
        import torch_npu
        torch = self.ctx.torch
        if self._entry_closed or self._entry_failed:
            raise RuntimeError('closed/failed MemFabric Tensor operator')
        if torch.npu.current_stream() != self.ctx.stream:
            raise RuntimeError('MemFabric Tensor operator requires its owning Stream')
        capture = getattr(torch.npu, 'is_current_stream_capturing', None)
        if capture is not None and capture():
            raise RuntimeError('MemFabric Tensor entry is eager-only')
        if (str(x.device) != str(self.ctx.device) or str(weight.device) != str(self.ctx.device)
                or x.dtype != torch.float16 or weight.dtype != torch.float16
                or x.requires_grad or weight.requires_grad):
            raise ValueError('owning NPU FP16 inference tensors required')
        if (x.ndim != 2 or x.shape[1] != 2048 or x.shape[0] not in self.ctx.shapes
                or tuple(weight.shape) != (2048, 2048) or not x.is_contiguous()
                or torch_npu.get_npu_format(x) != 2 or torch_npu.get_npu_format(weight) != 29):
            raise ValueError('ND X[M,2048] and NZ29 W[2048,2048] required')

    def _reap_inputs(self):
        while self._inflight and self._inflight[0][0].query():
            event, _inputs = self._inflight.popleft()
            self._events.append(event)

    def _poison(self):
        # The strong context registry preserves uncertain inputs/pools until
        # process exit. There is no collective destructor or implicit retry.
        self._entry_failed = True

    def _take_backend_inputs(self):
        return ()

    def _run_tensor(self, x, weight):
        with self._entry_lock:
            self._validate(x, weight)
            try:
                if self._raw_outputs is None:
                    # A2/A3 already acquire, wrap and retain inputs internally.
                    return self._backend_run(x, weight, 0)
                self._reap_inputs()
                if not self._events:
                    raise RuntimeError('input completion capacity exhausted; no silent drain/retry')
                m = x.shape[0]
                flags = self._leases[m]
                choices = [(self._cursor[m]+i) % len(flags) for i in range(len(flags))]
                slot = next((i for i in choices if not flags[i]), None)
                if slot is None:
                    raise RuntimeError('all output leases held; release outputs/aliases before another call')
                self._cursor[m] = (slot+1) % len(flags)
                flags[slot] = True
                released = False
                def release():
                    nonlocal released
                    with self._entry_lock:
                        if not released:
                            flags[slot] = False
                            released = True
                bases = self._raw_outputs[m] if isinstance(self._raw_outputs, dict) else self._raw_outputs
                try:
                    output = self._bridge.wrap(bases[slot].data_ptr(), m, x.device.index, self, release)
                except BaseException:
                    release()
                    raise
                self._uncertain_inputs = [x, weight]
                self._backend_run(x, weight, slot)
                self._uncertain_inputs.extend(self._take_backend_inputs())
                event = self._events.pop()
                self._inflight.append((event, self._uncertain_inputs))
                event.record(self.ctx.stream)
                self._uncertain_inputs = []
                return output
            except BaseException:
                self._poison()
                raise

    def after_sample(self):
        with self._entry_lock:
            try:
                result = self._after_backend()
                if self._raw_outputs is not None:
                    self._reap_inputs()
                return result
            except BaseException:
                self._poison()
                raise

    def after_block(self):
        return self.after_sample()

    def close(self):
        with self._entry_lock:
            if self._entry_closed:
                return
            if self._entry_failed:
                raise RuntimeError('failed Tensor operator: preserve context and stop both ranks')
            if any(any(flags) for flags in self._leases.values()):
                raise RuntimeError('output/alias leases still held; context remains alive')
            self.ctx.stream.synchronize()
            self.after_sample()
            self._close_backend()
            self._entry_closed = True
            with _LOCK:
                _CONTEXTS.pop(self._token, None)


def create_operator(family, spec, entry, ctx, collection='off'):
    """Public constructor; no L1/test framework imports or output arguments."""
    if family not in ('v2','v3'):
        raise ValueError('family must be v2 or v3')
    prepare_dispatch(ctx.torch)
    path = ROOT / 'csrc/_310P' / ('memfabric_mm_ar_v2' if family == 'v2' else 'memfabric_mm_ar_v3') / 'src/torch_adapter/tensor_entry.py'
    module = load_module(path, '_mf310p_backend_')
    if family == 'v2':
        cls = module.SelectedV2 if spec.get('api') == 'v2_selected_v1' else module.Trial
        return cls(spec, entry, ctx)
    return module.V3(spec, entry, ctx, collection)
