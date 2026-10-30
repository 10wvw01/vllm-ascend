# SPDX-License-Identifier: Apache-2.0
"""CPU fixtures; these do not emulate NPU execution or prove device correctness."""
import itertools
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

_PTRS = itertools.count(4096, 128 << 20)
_BASES = weakref.WeakValueDictionary()


class Lease:
    def __init__(self, owner, release): self.owner, self.release = owner, release
    def __del__(self): self.release()


class Tensor:
    def __init__(self, value=0, shape=(2048, 2048), fmt=2, *, base=None, lease=None):
        self.shape, self.fmt = shape, fmt
        self.device = SimpleNamespace(index=0)
        self.dtype, self.requires_grad = 'half', False
        self.state = [value] if base is None else base.state
        self.pointer = next(_PTRS) if base is None else base.pointer
        self.lease = lease
        if base is None: _BASES[self.pointer] = self
    @property
    def value(self): return self.state[0]
    @value.setter
    def value(self, value): self.state[0] = value
    @property
    def ndim(self): return len(self.shape)
    def is_contiguous(self): return True
    def clone(self): return Tensor(self.value, self.shape, self.fmt)
    def alias(self): return Tensor(shape=self.shape, fmt=self.fmt, base=self, lease=self.lease)
    def __add__(self, other): return Tensor(self.value + other.value, self.shape)
    def data_ptr(self): return self.pointer


def wrap(pointer, m, device, owner, release):
    return Tensor(shape=(m, 2048), base=_BASES[pointer], lease=Lease(owner, release))


def bridge_file(tmp_path):
    path = tmp_path / 'cpu_storage_bridge.py'
    path.write_text('from L1_test.testing import wrap\n')
    return str(path)


def context(events, shapes=(2048, 4096)):
    from .adapter_plugins.tensor_api import public_entry
    public = public_entry()
    class Event:
        complete = True
        def record(self, stream): pass
        def query(self): return self.complete
    class Library:
        def __init__(self, *args): pass
        def define(self, *args): pass
        def impl(self, *args): pass
    def dispatch(x, w, token):
        events.append(('dispatch_begin',))
        try: return public._dispatch(x, w, token)
        finally: events.append(('dispatch_end',))
    def empty(shape, **kw):
        events.append(('allocate', shape))
        return Tensor(shape=shape)
    stream = SimpleNamespace(npu_stream=17, synchronize=lambda: events.append(('sync',)))
    torch = SimpleNamespace(float16='half', empty=empty, ones=lambda shape, **kw: empty(shape)+Tensor(1, shape),
        empty_like=lambda t: empty(t.shape), equal=lambda a,b: a.value == b.value,
        library=SimpleNamespace(Library=Library), ops=SimpleNamespace(memfabric310p=SimpleNamespace(mm_ar=dispatch)),
        npu=SimpleNamespace(Event=Event,current_stream=lambda:stream,is_current_stream_capturing=lambda:False))
    ctx = SimpleNamespace(rank=0,url='tcp://CPU-fixture',device=Tensor().device,shapes=list(shapes),stream=stream,torch=torch)
    # Fake format validation is separate from the actual CPU Torch dispatcher test.
    sys.modules.setdefault('torch_npu', SimpleNamespace(get_npu_format=lambda t:t.fmt))
    return ctx


def public_files(version):
    from .core import ROOT, sha
    family = 'v2' if version.startswith('V2-') else 'v3'
    paths = [ROOT/'csrc/_310P/common/torch_tensor_entry.py',
             ROOT/f'csrc/_310P/memfabric_mm_ar_{family}/src/torch_adapter/tensor_entry.py']
    return {str(p):sha(p) for p in paths}


def reports(plan, job, values=None, identity=None):
    from .core import CONTRACT, execution_contract
    from .measurement import units
    result = []
    for rank in (0,1):
        cases = []
        for m in plan['M']:
            for scenario in plan['scenarios']:
                samples = []
                for first, count in units(plan['N'], plan['batch_size'], scenario):
                    value = values(rank, first) if values else [10,100][(first+rank)%2]
                    samples.append(dict(i=first,calls=count,variants=[i%plan['working_set'] for i in range(first,first+count)],
                                        wall_us=value*count,event_us=value*count))
                cases.append(dict(M=m,scenario=scenario,warmup_count=plan['W'],samples=samples,
                    checking=plan['checking'],validation=dict(status='PASS',scope='pre_post_and_last_group'),
                    calibration=dict(subtracted=False),timer_overhead_ratio=.01,warmup=dict(status='STABLE_TIMING_WINDOW')))
        result.append(dict(contract=CONTRACT,plan=plan,job=job,rank=rank,status='PASS',cases=cases,
            execution_contract=execution_contract(job['version'],identity or {},plan),
            submission_validation=dict(status='PASS',timed=False,cases=[dict(M=m,calls=plan['batch_size'],correct=True,disjoint=True) for m in plan['M']]),
            downstream_validation=dict(status='PASS',timed=False,same_stream=True,cases=[
                dict(phase=phase,M=m,variant=v,correct=True) for phase in ('before','after') for m in plan['M'] for v in range(plan['gate_variants'])]),
            runtime_identity=identity or {'frozen':'one'},route={'verified':True}))
    return result
