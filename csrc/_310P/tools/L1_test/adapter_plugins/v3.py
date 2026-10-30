# SPDX-License-Identifier: Apache-2.0
"""Thin aliases to the operator-owned public Torch Tensor entry."""
from .tensor_api import backend_module

_impl = backend_module("v3")
V3, family, load_sessions = _impl.V3, _impl.family, _impl.load_sessions
