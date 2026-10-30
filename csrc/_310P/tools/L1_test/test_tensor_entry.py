# SPDX-License-Identifier: Apache-2.0
"""Public dispatch and async lifetime tests; NPU binary correctness is NOT_RUN."""
import gc
import subprocess
import sys
import weakref
from pathlib import Path

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from L1_test.testing import Tensor, context, wrap
from L1_test.adapter_plugins.tensor_api import public_entry
from types import SimpleNamespace


def raw_operator(events):
    public=public_entry();ctx=context(events,shapes=[2048])
    class Operator(public.TensorOperator):
        def _backend_run(self,x,w,slot):
            events.append(('backend',slot));self.outputs[2048][slot].value=x.value+w.value
        def _after_backend(self): return {}
        def _close_backend(self): events.append(('destroy',))
    op=Operator();op.outputs={2048:[Tensor(),Tensor()]}
    def wrapping(*args):
        events.append(('wrap',));return wrap(*args)
    op._publish(ctx,SimpleNamespace(wrap=wrapping),op.outputs)
    return op


def test_wrap_acquire_and_input_retention_are_inside_dispatch_and_alias_retains_lease():
    events=[];op=raw_operator(events);events.clear()
    x,w=Tensor(2),Tensor(3,fmt=29)
    y=op.run(x,w);alias=y.alias();y2=op.run(x,w)
    assert y.value==5 and y2.value==5 and y.data_ptr()!=y2.data_ptr()
    assert events[:4]==[('dispatch_begin',),('wrap',),('backend',0),('dispatch_end',)]
    del y,y2;gc.collect()
    with pytest.raises(RuntimeError,match='leases still held'):op.close()
    assert sum(op._leases[2048])==1
    del alias;gc.collect();op.close()
    with pytest.raises(RuntimeError,match='unknown'):op.run(x,w)


def test_inputs_survive_until_completion_event_not_python_call_return():
    events=[];op=raw_operator(events)
    for event in op._events:event.complete=False
    x,w=Tensor(2),Tensor(3,fmt=29);refs=[weakref.ref(x),weakref.ref(w)]
    y=op.run(x,w);del x,w,y;gc.collect()
    assert all(ref() is not None for ref in refs)
    assert len(op._inflight)==1
    op._inflight[0][0].complete=True
    op.after_block();gc.collect()
    assert all(ref() is None for ref in refs)
    op.close()



def test_dropping_python_owner_does_not_free_an_inflight_pool():
    events=[];op=raw_operator(events);token=op._token
    reference=weakref.ref(op)
    for event in op._events:event.complete=False
    y=op.run(Tensor(),Tensor(fmt=29));del y,op;gc.collect()
    assert reference() is not None
    owner=public_entry()._CONTEXTS[token]
    assert owner._inflight
    owner._inflight[0][0].complete=True
    owner.close()
    assert token not in public_entry()._CONTEXTS
    del owner;gc.collect()
    assert reference() is None

def test_capacity_failure_poisoned_without_overwriting_held_outputs():
    events=[];op=raw_operator(events);x,w=Tensor(),Tensor(fmt=29)
    y,z=op.run(x,w),op.run(x,w)
    before=events.count(('backend',0))+events.count(('backend',1))
    with pytest.raises(RuntimeError,match='leases held'):op.run(x,w)
    assert events.count(('backend',0))+events.count(('backend',1))==before
    with pytest.raises(RuntimeError,match='failed Tensor'):op.close()
    del y,z


def test_real_torch_dispatcher_encloses_operator_work_in_fresh_process():
    # An actual CPU Torch dispatcher, not the fake dispatch used above. The
    # backend overrides NPU validation solely for this CPU registration test.
    script=r'''
import importlib.util, sys
from types import SimpleNamespace
import torch
path=sys.argv[1]
spec=importlib.util.spec_from_file_location('mf310p_tensor_entry',path)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
class Operator(module.TensorOperator):
    def _validate(self,x,w): pass
    def _backend_run(self,x,w,slot): return x.mm(w)
    def _after_backend(self): return {}
    def _close_backend(self): pass
ctx=SimpleNamespace(torch=torch,shapes=[2],stream=SimpleNamespace(synchronize=lambda:None))
op=Operator();op._publish(ctx,None,None)
x=torch.ones(2,2);w=torch.ones(2,2)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
    y=op.run(x,w)
assert torch.equal(y,torch.full((2,2),2.))
events=profile.events();entry=next(e for e in events if e.name=='memfabric310p::mm_ar')
assert any(e.name=='aten::mm' and e.cpu_parent is entry for e in events)
op.close()
try: op.run(x,w)
except RuntimeError as error: assert 'closed or unknown' in str(error)
else: raise AssertionError('closed token accepted')
print('REAL_TORCH_DISPATCH_PASS')
'''
    result=subprocess.run([sys.executable,'-c',script,str(Path(public_entry().__file__))],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'REAL_TORCH_DISPATCH_PASS' in result.stdout
