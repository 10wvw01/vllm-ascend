# SPDX-License-Identifier: Apache-2.0
"""New public V2 API identity and complete-call adapter, without NPU imports."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from L1_test import adapters
from L1_test.core import sha, validate_runtime
from L1_test.worker import verify_maps


def specification(tmp_path, version='V2-7'):
    lib = tmp_path/'extension.so'
    device = tmp_path/'libmf310p_v2_selected_device.so'
    lib.write_bytes(b'CPU fixture; not an NPU binary')
    device.write_bytes(b'CPU fixture; not an NPU binary')
    return dict(version=version, source_head='a'*40, api='v2_selected_v1', library=str(lib),
                selected_device_library=str(device), files={str(p):sha(p) for p in (lib, device)},
                supported_M=[2048 if version=='V2-7' else 4096])


@pytest.mark.parametrize('version', ['V2-7', 'V2-15'])
def test_selected_api_requires_its_fixed_shape_and_frozen_device(tmp_path, version):
    spec = specification(tmp_path, version)
    validate_runtime(spec, version)
    for changed in [dict(supported_M=[2048,4096]), dict(config={'kind':6}), dict(private_library=spec['library']),
                    dict(selected_device_library='/unfrozen.so'), dict(api='unknown')]:
        with pytest.raises(ValueError):
            validate_runtime({**spec, **changed}, version)
    other = {**spec, 'version':'V2-1'}
    with pytest.raises(ValueError, match='selected V2'):
        validate_runtime(other, 'V2-1')


@pytest.mark.parametrize('version,m', [('V2-7',2048),('V2-15',4096)])
def test_selected_adapter_pins_outputs_before_pool_and_calls_only_public_api(tmp_path, monkeypatch, version, m):
    events = []
    spec = specification(tmp_path, version)
    class Function:
        def __init__(self, name): self.name=name
        def __call__(self, *args):
            events.append(self.name)
            if self.name=='create':
                assert args[2]==m
                args[-1]._obj.value=123
            if self.name=='check_status': args[-1]._obj.value=0
            return 0
    class Library:
        def __init__(self):
            for name in ('create','ib_probe','run','check_status','destroy'):
                setattr(self,'mf310p_v2_selected_'+name,Function(name))
    def load(*args, **kwargs):
        events.append('load')
        return Library()
    monkeypatch.setattr(adapters.ctypes, 'CDLL', load)
    class Tensor:
        shape=(m,2048)
        def data_ptr(self): return id(self)
    def empty(*args, **kwargs):
        events.append('allocate')
        return Tensor()
    ctx=SimpleNamespace(shapes=[m], rank=0, url='tcp://test', device='CPU-fixture',
                        torch=SimpleNamespace(float16='half',empty=empty),
                        stream=SimpleNamespace(npu_stream=7,synchronize=lambda:events.append('sync')))
    selected=adapters.create(version,spec,ctx,'off')
    assert events[:6]==['allocate','allocate','sync','load','create','ib_probe']
    x,w=Tensor(),Tensor()
    assert selected.run(x,w,0) is selected.outputs[0]
    assert selected.after_sample()=={}
    bad=Tensor();bad.shape=(3072,2048)
    before=events.count('run')
    with pytest.raises(ValueError): selected.run(bad,w,0)
    assert events.count('run')==before
    selected.close()
    assert events[-1]=='destroy'
    ctx.shapes=[2048,4096]
    with pytest.raises(ValueError,match='one matching fixed M'):
        adapters.create(version,spec,ctx,'off')


def test_maps_reject_unfrozen_base_v2_device_in_selected_process(tmp_path, monkeypatch):
    spec=specification(tmp_path)
    base=tmp_path/'libmf310p_v2_trial_device.so'
    base.write_bytes(b'CPU fixture')
    mapped=[spec['library'],spec['selected_device_library'],str(base)]
    maps='\n'.join('0000-1000 r-xp 0000 00:00 0 '+p for p in mapped)
    monkeypatch.setattr(Path,'read_text',lambda self:maps)
    with pytest.raises(RuntimeError,match='unfrozen MemFabric'):
        verify_maps(spec,'V2-7')
    spec['files'][str(base)]=sha(base)
    assert verify_maps(spec,'V2-7')==sorted(mapped[1:])


def test_stock_rejects_loaded_v2_device_without_mf_sdk_mapping(tmp_path, monkeypatch):
    base=tmp_path/'libmf310p_v2_trial_device.so'
    base.write_bytes(b'CPU fixture')
    monkeypatch.setattr(Path,'read_text',lambda self:'0000-1000 r-xp 0000 00:00 0 '+str(base))
    with pytest.raises(RuntimeError,match='Stock contains MemFabric'):
        verify_maps({'files':{}},'STOCK')
