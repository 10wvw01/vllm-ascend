# SPDX-License-Identifier: Apache-2.0
"""Thin aliases to the operator-owned public Torch Tensor entry."""
from .tensor_api import backend_module

_impl = backend_module("v2")
Trial, SelectedV2 = _impl.Trial, _impl.SelectedV2
