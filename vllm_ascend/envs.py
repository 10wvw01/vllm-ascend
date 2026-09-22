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
    # Experimental 310P3-only path: Qwen3.5/3.6 full-attention unquantized
    # o_proj plus TP=2 reduction through the installed wgm-dev-310p
    # MemFabric public SHM/SDMA API.
    "VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ": lambda: bool(
        int(os.getenv("VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ", "0"))
    ),
    # Config-store rendezvous used by customized MemFabric SHM.
    "VLLM_ASCEND_310P_MEMFABRIC_STORE_URL": lambda: os.getenv(
        "VLLM_ASCEND_310P_MEMFABRIC_STORE_URL", "tcp://127.0.0.1:8581"
    ),
    # Per-rank physical contribution to the symmetric pool. ABI v7 derives
    # arena_rows from this budget instead of multiplying it by communication
    # batch size. 96 MiB keeps the default arena at up to 8192 rows while
    # leaving transport headroom.
    "VLLM_ASCEND_310P_MEMFABRIC_LOCAL_BYTES": lambda: int(
        os.getenv("VLLM_ASCEND_310P_MEMFABRIC_LOCAL_BYTES", str(96 * 1024 * 1024))
    ),
    # Communication/reduction batch is an integer multiple of native
    # baseM=256. Legal values are 1/2/4; q=2 derives batch_m=512 and a 2 MiB
    # FP16 payload for N=2048. 2 MiB is not a protocol constant.
    "VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_BATCH_BASEM_COUNT": lambda: int(
        os.getenv("VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_BATCH_BASEM_COUNT", "2")
    ),
    # D2 workaround (validation aid, default off): during profile/warmup
    # dummy runs the routed o_proj falls back to stock matmul + HCCL
    # all-reduce, deferring MemFabric pool creation to the first real
    # request. The 310P AICPU watchdog kills the SDMA orchestrator epoch
    # task 28s after pool creation (MemFabric self-limit arithmetic
    # assumes 22s), so eager-mode init must not create the pool.
    "VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_WARMUP_FALLBACK": lambda: bool(
        int(os.getenv("VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_WARMUP_FALLBACK", "0"))
    ),
    # Minimum M (token rows) routed to the fused MemFabric o_proj path.
    # Below this the layer uses the stock NZ matmul + HCCL all-reduce: the
    # v7 keeps the conservative M=4096 route threshold inherited from the
    # validated v6 baseline until cooperative-MM hardware measurements are
    # collected. Range: 1 (always fused) to unlimited.
    "VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_MIN_M": lambda: int(
        os.getenv("VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_MIN_M", "4096")
    ),
    # Wave-level host-phase trace + device-duration outlier monitor (debug
    # aid, default off). C++ runtime prints one "[mf310p-trace]" line per
    # fused call; the Python wrapper additionally records NPU event pairs
    # and a daemon thread logs "[mf310p-slow]" lines for calls whose device
    # duration exceeds 500 ms (e.g. delayed SDMA mail relay at an epoch
    # relaunch boundary).
    "VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TRACE": lambda: bool(
        int(os.getenv("VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TRACE", "0"))
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
