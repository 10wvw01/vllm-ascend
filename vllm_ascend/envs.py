#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# This file is mainly Adapted from vllm-project/vllm/vllm/envs.py
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
from collections.abc import Callable
from typing import Any

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

env_variables: dict[str, Callable[[], Any]] = {
    # max compile thread number for package building. Usually, it is set to
    # the number of CPU cores. If not set, the default value is None, which
    # means all number of CPU cores will be used.
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # The build type of the package. It can be one of the following values:
    # Release, Debug, RelWithDebugInfo. If not set, the default value is Release.
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # Whether to compile custom kernels. If not set, the default is True.
    "COMPILE_CUSTOM_KERNELS": lambda: bool(int(os.getenv("COMPILE_CUSTOM_KERNELS", "1"))),
    # The CXX compiler used for compiling the package.
    "CXX_COMPILER": lambda: os.getenv("CXX_COMPILER", None),
    # The C compiler used for compiling the package.
    "C_COMPILER": lambda: os.getenv("C_COMPILER", None),
    # The version of the Ascend chip. It's used for package building.
    "SOC_VERSION": lambda: os.getenv("SOC_VERSION", None),
    # If set, vllm-ascend will print verbose logs during compilation
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
    # The home path for CANN toolkit.
    "ASCEND_HOME_PATH": lambda: os.getenv("ASCEND_HOME_PATH", None),
    # The path for HCCL library.
    "HCCL_SO_PATH": lambda: os.getenv("HCCL_SO_PATH", None),
    # The version of vllm is installed.
    "VLLM_VERSION": lambda: os.getenv("VLLM_VERSION", None),
    # Whether to enable FlashComm optimization when tensor parallel is enabled.
    # DEPRECATED: use additional_config.enable_flashcomm1 instead.
    "VLLM_ASCEND_ENABLE_FLASHCOMM1": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0"))),
    # Whether to enable msMonitor tool to monitor the performance of vllm-ascend.
    "MSMONITOR_USE_DAEMON": lambda: bool(int(os.getenv("MSMONITOR_USE_DAEMON", "0"))),
    # Whether to enable MLAPO optimization for DeepSeek W8A8 series models.
    "VLLM_ASCEND_ENABLE_MLAPO": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_MLAPO", "1"))),
    # Whether to enable weight cast format to FRACTAL_NZ.
    # 0: close nz; 1: only quant case enable nz; 2: enable nz as long as possible.
    "VLLM_ASCEND_ENABLE_NZ": lambda: int(os.getenv("VLLM_ASCEND_ENABLE_NZ", 1)),
    # Whether to anbale dynamic EPLB
    "DYNAMIC_EPLB": lambda: os.getenv("DYNAMIC_EPLB", "false").lower(),
    # Whether to enable fused MC2 (`dispatch_ffn_combine/mega_moe`).
    "VLLM_ASCEND_ENABLE_FUSED_MC2": lambda: int(os.getenv("VLLM_ASCEND_ENABLE_FUSED_MC2", "0")),
    # DEPRECATED: use --additional-config '{"enable_balance_scheduling": true}' instead.
    "VLLM_ASCEND_BALANCE_SCHEDULING": lambda: bool(int(os.getenv("VLLM_ASCEND_BALANCE_SCHEDULING", "0"))),
    # use fused op transpose_kv_cache_by_block, default is True
    "VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK": lambda: bool(
        int(os.getenv("VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK", "1"))
    ),
    # Control the aclrtMemcpyBatchAsync compile path for KV cache offloading.
    "VLLM_ASCEND_ENABLE_BATCH_MEMCPY": lambda: os.getenv("VLLM_ASCEND_ENABLE_BATCH_MEMCPY", None),

    # Experimental 310P3-only path: Qwen3.5/3.6 full-attention W8A8 o_proj
    # plus TP=2 all-reduce implemented with the separately built and installed
    # wgm-dev-310p MemFabric SDK. Disabled by default.
    "VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ": lambda: bool(
        int(os.getenv("VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ", "0"))
    ),
    # Installation prefix of the independently built customized MemFabric SDK.
    # Expected layout is SDK-defined; CMake searches ROOT/include and ROOT/lib{,64}
    # but does not assume an upstream/official library package exists.
    "VLLM_ASCEND_310P_MEMFABRIC_ROOT": lambda: os.getenv(
        "VLLM_ASCEND_310P_MEMFABRIC_ROOT", None
    ),
    # Semicolon-separated libraries exported by the customized SDK. Entries may
    # be absolute library paths or linker names. We intentionally do not hardcode
    # any library name until the wgm-dev-310p install ABI is provided.
    "VLLM_ASCEND_310P_MEMFABRIC_LIBRARIES": lambda: os.getenv(
        "VLLM_ASCEND_310P_MEMFABRIC_LIBRARIES", None
    ),
    # Config-store rendezvous used by the customized MemFabric SHM runtime.
    "VLLM_ASCEND_310P_MEMFABRIC_STORE_URL": lambda: os.getenv(
        "VLLM_ASCEND_310P_MEMFABRIC_STORE_URL", "tcp://127.0.0.1:8581"
    ),
    # Per-rank physical contribution to the symmetric pool. 32 MiB matches the
    # supplied 310P AICore/AICPU/SDMA example.
    "VLLM_ASCEND_310P_MEMFABRIC_LOCAL_BYTES": lambda: int(
        os.getenv("VLLM_ASCEND_310P_MEMFABRIC_LOCAL_BYTES", str(32 * 1024 * 1024))
    ),
    # M tile 32 => 32 * 2048 * BF16(2B) = 128 KiB per SDMA chunk, matching the
    # supplied 310P overlap example.
    "VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TILE_M": lambda: int(
        os.getenv("VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TILE_M", "32")
    ),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
