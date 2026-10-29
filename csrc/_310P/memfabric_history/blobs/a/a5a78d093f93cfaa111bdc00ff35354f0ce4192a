# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Runtime hooks that adapt MemFabric mm_ar to the 310P DFlash runner.

Keep the DFlash runner itself owned by the main development branch. The
MemFabric customization only needs two lifecycle adjustments:

1. optionally wrap *profile* dummy runs in the existing MemFabric warmup
   fallback context, while leaving graph pre-capture eager warmups fused; and
2. avoid device-wide synchronization once the long-lived MemFabric SDMA pool
   has started, because its supervised epoch kernel lives on a separate stream.

The hooks are installed lazily when an eligible mm_ar layer is selected, which
happens during model construction before profile/dummy runs. Installation is
idempotent and preserves the complete DFlash implementation of ``_dummy_run``
(including FDO/FAP capture state and async metadata handling).
"""

from __future__ import annotations

from functools import wraps
from inspect import signature

import torch

_HOOKS_INSTALLED = False


# 作用：在选中合格投影层时，给 NPUModelRunner310 安装幂等生命周期适配。
# 入参：无。返回：None；副作用是替换 _dummy_run/_sync_device 两个方法。
# 原因：只包装原 Runner 的必要行为，保留 DFlash 原始 profile/Graph warmup 控制流。
def install_memfabric_mm_ar_runner_hooks() -> None:
    """Install the minimal MemFabric lifecycle adaptation on NPUModelRunner310."""

    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return

    from vllm_ascend._310p.model_runner_310p import NPUModelRunner310
    from vllm_ascend._310p.ops.memfabric_mm_ar import (
        memfabric_mm_ar_pool_started,
        memfabric_mm_ar_warmup_fallback,
    )

    from vllm_ascend._310p.ops.memfabric_mm_ar_v2_serve import v2_pool_started

    # Preserve the DFlash-owned implementations wholesale. The dummy-run
    # wrapper only enables fallback for memory/profile runs. Graph pre-capture
    # eager warmups must execute the fused path so lazy MemFabric state (most
    # importantly small-M per-weight slices) exists before torch.npu.graph
    # capture begins.
    original_dummy_run = NPUModelRunner310._dummy_run
    original_sync_device = NPUModelRunner310._sync_device
    dummy_run_signature = signature(original_dummy_run)

    # 包装原 _dummy_run：入参 self、原函数任意 args/kwargs，返回原方法的值。
    # 通过 inspect.signature 识别 is_profile，只有 profile 用受控 Stock fallback；
    # Graph capture 前的真实 eager warmup 仍遵循各自融合逻辑，不应一概回退。
    @wraps(original_dummy_run)
    def _dummy_run_with_memfabric(self, *args, **kwargs):
        bound = dummy_run_signature.bind_partial(self, *args, **kwargs)
        is_profile = bool(bound.arguments.get("is_profile", False))
        if is_profile:
            with memfabric_mm_ar_warmup_fallback():
                return original_dummy_run(self, *args, **kwargs)
        return original_dummy_run(self, *args, **kwargs)

    # 包装原 _sync_device：入参 self，返回 None。
    # 建池前沿用 Runner 原同步；建池后仅同步当前模型 stream，
    # 避免 device-wide sync 把 MemFabric 长驻 orchestrator 流也当成待排空计算。
    @wraps(original_sync_device)
    def _sync_device_with_memfabric(self) -> None:
        # The MemFabric orchestrator runs a supervised epoch kernel on its own
        # launch stream for the lifetime of the pool. A device-wide sync would
        # wait on that stream; after pool creation only drain the model stream.
        if memfabric_mm_ar_pool_started() or v2_pool_started():
            torch.npu.current_stream().synchronize()
            return
        # Before pool creation retain the runner's original synchronization
        # contract rather than duplicating its current implementation here.
        return original_sync_device(self)

    # 在类方法层安装一次；保持 @wraps 元数据以免破坏上游 Runner 观察/调试行为。
    NPUModelRunner310._dummy_run = _dummy_run_with_memfabric
    NPUModelRunner310._sync_device = _sync_device_with_memfabric
    _HOOKS_INSTALLED = True
