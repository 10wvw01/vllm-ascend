# SPDX-License-Identifier: Apache-2.0
"""Load the operator-owned public Tensor API; never allocate/wrap an output here."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]


def public_entry():
    name = 'mf310p_tensor_entry'
    path = ROOT/'csrc/_310P/common/torch_tensor_entry.py'
    if name not in sys.modules:
        descriptor = importlib.util.spec_from_file_location(name,path)
        module = importlib.util.module_from_spec(descriptor)
        sys.modules[name] = module
        descriptor.loader.exec_module(module)
    return sys.modules[name]


def backend_module(family):
    module = public_entry()
    directory = 'memfabric_mm_ar_v2' if family == 'v2' else 'memfabric_mm_ar_v3'
    return module.load_module(ROOT/'csrc/_310P'/directory/'src/torch_adapter/tensor_entry.py',
                              '_mf310p_backend_')
