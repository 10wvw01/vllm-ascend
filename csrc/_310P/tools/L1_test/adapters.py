# SPDX-License-Identifier: Apache-2.0
"""Adapter factory. Add an operator call here, never a separate timing loop."""
import ctypes  # Kept public for existing adapter tests/integrations.
import importlib.util

from .core import catalog
from .adapter_plugins.stock import Native
from .adapter_plugins.v1 import V1
from .adapter_plugins.v2 import Trial, SelectedV2
from .adapter_plugins.v3 import V3


def create(version, spec, ctx, collection):
    if version == "STOCK":
        adapter = Native(ctx)
    else:
        entry = catalog()[version]
        kind = entry["runtime"]
        if spec.get("api") == "v2_selected_v1":
            adapter = SelectedV2(spec, entry, ctx)
        elif spec.get("api") == "v2_trial_v1":
            adapter = Trial(spec, entry, ctx)
        elif spec.get("api") == "python_session_v1" or kind in ("v3", "v32", "a3", "a3-2", "a3-tail"):
            adapter = V3(spec, entry, ctx, collection)
        elif kind == "external":
            module_spec = importlib.util.spec_from_file_location("frozen_l1_adapter", spec["factory_file"])
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            adapter = module.create_adapter(ctx=ctx, specification=spec)
            if adapter.version != version:
                raise ValueError("external factory returned the wrong VN-M")
        elif kind == "v1":
            adapter = V1(spec, entry, ctx)
        elif kind.startswith("v2"):
            adapter = Trial(spec, entry, ctx)
        else:
            raise ValueError("operator requires an explicit matching adapter")
    adapter.version = version
    return adapter
