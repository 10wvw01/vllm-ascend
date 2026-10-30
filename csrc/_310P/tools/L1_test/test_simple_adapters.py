# SPDX-License-Identifier: Apache-2.0
"""Thin operator adapters tested through fake sessions, without importing NPU."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from L1_test import adapters
from L1_test.adapter_plugins import v3
from L1_test.core import catalog, sha, validate_runtime


from L1_test.testing import Tensor, context, bridge_file, public_files


@pytest.mark.parametrize('version',[f'V3-{n}' for n in range(8)])
def test_each_v3_iteration_passes_only_its_registered_configuration(tmp_path,monkeypatch,version):
    events=[];ctx=context(events)
    class Session:
        def __init__(self,*args,**kw):
            events.append(('create',kw));self.phase_log=None;self.merged_wave=True;self.pending=[]
        def ib_probe(self):events.append(('probe',))
        def probe_add(self,a,b,out):out.value=a.value+b.value;events.append(('add',))
        def synchronize_check(self):events.append(('check',))
        def initialize(self):events.append(('initialize',))
        def run(self,x,w,out,**kw):events.append(('raw_run',kw));return out
        def run_pool(self,x,w,**kw):
            events.append(('pool_run',kw))
            if self.phase_log is not None:self.phase_log.append({'enqueue':1.0})
            return Tensor(2,x.shape)
        def close(self):events.append(('close',))
    modules={key:SimpleNamespace(**{name:Session}) for key,name in [
        ('memfabric_mm_ar_v3','V3Session'),('memfabric_mm_ar_v32','V32Session'),('memfabric_mm_ar_a3','A3Session')]}
    monkeypatch.setattr(v3._impl,'load_sessions',lambda spec,entry:modules)
    tensor=Path(bridge_file(tmp_path))
    entry=catalog()[version]
    config=dict(comm_rows=2048)
    if entry['adapter_family']=='v3':config['pipeline']=False
    if entry['adapter_family']=='a3':config['compute_rows']=512
    spec=dict(api='python_session_v1',library='CPU-only.so',tensor_adapter=str(tensor),config=config)
    op=adapters.create(version,spec,ctx,'host')
    created=next(i for i,e in enumerate(events) if e[0]=='create')
    assert all(i<created for i,e in enumerate(events) if e[0]=='allocate')
    assert ('sync',) in events[:created]
    assert op.version==version
    assert next(e[1] for e in events if e[0]=='create')['packet_bytes']==entry['config']['packet_bytes']
    result=op.run(Tensor(),Tensor(fmt=29),0)
    assert isinstance(result,Tensor)
    if entry['adapter_family']=='v3':
        assert ('raw_run',{'group':2048,'pipeline':False}) in events
        assert op.outputs and op.after_sample()=={}
    elif entry['adapter_family']=='v32':
        assert op.session.merged_wave==(version=='V3-3')
        assert ('pool_run',{'group':2048}) in events and not op.outputs
        assert op.after_sample()=={'host_calls':[{'enqueue':1.0}]}
    else:
        assert ('initialize',) in events
        assert ('pool_run',{'compute_rows':512,'comm_rows':2048}) in events and not op.outputs
        kwargs=next(e[1] for e in events if e[0]=='create')
        assert kwargs.get('variant')==({'V3-6':'a3-2','V3-7':'a3-tail'}.get(version))
        assert op.after_sample()=={'host_calls':[{'enqueue':1.0}]}
    del result
    op.close();assert events[-1]==('close',)


def test_frozen_session_relative_import_never_uses_current_global_module(tmp_path,monkeypatch):
    base=tmp_path/'base.py';base.write_text("class V3Session:\n    marker='frozen-original'\n")
    derived=tmp_path/'v32.py';derived.write_text('from .memfabric_mm_ar_v3 import V3Session\nclass V32Session(V3Session): pass\n')
    monkeypatch.setitem(sys.modules,'memfabric_mm_ar_v3',SimpleNamespace(V3Session=type('Wrong',(),{'marker':'current'})))
    spec=dict(api='python_session_v1',python_modules={'memfabric_mm_ar_v3':str(base),'memfabric_mm_ar_v32':str(derived)},
              files={str(p):sha(p) for p in (base,derived)})
    modules=v3.load_sessions(spec,catalog()['V3-2'])
    assert modules['memfabric_mm_ar_v32'].V32Session.marker=='frozen-original'
    spec['files'][str(base)]='0'*64
    with pytest.raises(ValueError,match='frozen'):v3.load_sessions(spec,catalog()['V3-2'])


def test_v2_without_diagnostics_does_not_bind_ar_only_and_allocates_before_pool(tmp_path,monkeypatch):
    events=[];ctx=context(events)
    class Function:
        def __init__(self,name):self.name=name
        def __call__(self,*args):
            events.append((self.name,))
            if self.name=='create':args[-1]._obj.value=99
            if self.name=='check_status':args[-1]._obj.value=0
            return 0
    library=SimpleNamespace(**{'mf310p_v2_trial_'+key:Function(key) for key in ('create','ib_probe','run','check_status','destroy')})
    monkeypatch.setattr(adapters.ctypes,'CDLL',lambda *a,**k:library)
    op=adapters.create('V2-1',{'library':'CPU-fixture','tensor_adapter':bridge_file(tmp_path)},ctx,'off')
    assert events.index(('create',))>max(i for i,e in enumerate(events) if e[0]=='allocate')
    result=op.run(Tensor(),Tensor(fmt=29),0)
    assert result.data_ptr()==op.outputs[2048][0].data_ptr() and result is not op.outputs[2048][0]
    del result
    op.after_sample();op.close()


def test_v1_single_call_and_stock_do_not_hide_additional_execution(monkeypatch):
    events=[];ctx=context(events)
    monkeypatch.setattr(adapters.ctypes,'CDLL',lambda *a,**k:pytest.fail('Stock must not load a fusion library'))
    ctx.linear=lambda x,w:events.append(('MM',)) or Tensor(3)
    ctx.dist=SimpleNamespace(all_reduce=lambda out:events.append(('HCCL',)))
    op=adapters.create('STOCK',{},ctx,'off');op.run(Tensor(),Tensor(fmt=29),0)
    assert events==[('MM',),('HCCL',)]
    ctx.torch.ops=SimpleNamespace(load_library=lambda p:events.append(('load',p)),_C_ascend=SimpleNamespace(
        memfabric_mm_ar_allreduce=lambda x,w,r,q:events.append(('V1',r,q)) or Tensor(6),
        memfabric_mm_ar_shutdown=lambda:events.append(('shutdown',))))
    op=adapters.create('V1-0',{'library':'frozen-V1.so','config':{'batch_basem_count':4}},ctx,'off')
    op.run(Tensor(),Tensor(fmt=29),0);op.close()
    assert ('V1',0,4) in events and events[-2:]==[('sync',),('shutdown',)]


def test_historical_sessions_require_explicit_frozen_roles_and_no_merged_alias(tmp_path):
    files=[]
    for name in ('host.so','tensor.so','base.py','v32.py'):
        p=tmp_path/name;p.write_bytes(b'CPU fixture');files.append(p)
    spec=dict(version='V3-2',source_head='a'*40,api='python_session_v1',library=str(files[0]),
              tensor_adapter=str(files[1]),supported_M=[2048,4096],files={str(p):sha(p) for p in files},
              python_modules={'memfabric_mm_ar_v3':str(files[2]),'memfabric_mm_ar_v32':str(files[3])})
    with pytest.raises(ValueError,match='historical'):validate_runtime(spec,'V3-2')
    with pytest.raises(ValueError,match='historical'):
        validate_runtime({**spec,'historical_identity_verified':'false'},'V3-2')
    spec['files'].update(public_files('V3-2'))
    spec['historical_identity_verified']=True;validate_runtime(spec,'V3-2')
    with pytest.raises(ValueError,match='classic/merged'):
        validate_runtime({**spec,'config':{'merged_wave':True}},'V3-2')
    bad=copy.deepcopy(spec);bad['python_modules'].pop('memfabric_mm_ar_v3')
    with pytest.raises(ValueError,match='roles'):validate_runtime(bad,'V3-2')
