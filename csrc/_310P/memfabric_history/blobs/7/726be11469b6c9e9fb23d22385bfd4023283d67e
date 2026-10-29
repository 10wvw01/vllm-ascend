#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# F1-C 现场环境口径：源化 CANN/MemFabric 环境 + 锁定部署指纹（G0 已核验）。
source /usr/local/memfabric_hybrid/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=6,7
export MF_SDMA_ORCH_JSON=/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/hybm/aicpu_kernel/libmf_sdma_orch_v11.json
export V2_HOST_LIB=/home/w00804146/310P/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so
export F1C_DEVICE_SO_PATH=/home/w00804146/310P/vllm-ascend/vllm_ascend/libmf310p_v2_trial_device.so
export F1C_CORE_SO_LIB64=/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/lib64/libmf_hybm_core.so
export F1C_CORE_SO_PY=/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid/lib/libmf_hybm_core.so
export F1C_DEVICE_SHA256="$(sha256sum "$F1C_DEVICE_SO_PATH" | awk '{print $1}')"
export F1C_PLUGIN_SHA256="$(sha256sum "$V2_HOST_LIB" | awk '{print $1}')"
export F1C_HOST_SHA256="$(sha256sum "$F1C_CORE_SO_LIB64" | awk '{print $1}')"
export F1C_KVER_SHA256="$(sha256sum "$MF_SDMA_ORCH_JSON" | awk '{print $1}')"
