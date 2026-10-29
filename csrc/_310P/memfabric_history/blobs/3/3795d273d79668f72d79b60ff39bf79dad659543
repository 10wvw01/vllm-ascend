#!/usr/bin/env bash
# E4 attempt3（保护器实机验收轮）：本轮G0重新审签的私有装载环境。
# G0事实：2026-10-28 16384(dev2,3)双Die+MCU健康、零进程、hugepages 0/0；
# 四冻结SHA与e4_layered_20261028/g0/frozen_binaries_g0.sha256逐字节一致（recheck OK）。
# 硬性约束：E4_CONFIRM_APPROVED 保持未授权（不进入十四场Confirm），Screen后自动停止。
set -Eeuo pipefail
B=/tmp/opencode/a1-batch
export PATH=/usr/local/python3.12.13/bin:$PATH
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
export ASCEND_RT_VISIBLE_DEVICES=4,5
export E4_PHYSICAL_CARD_ID=38144
export MF_SDMA_ORCH_JSON=/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/hybm/aicpu_kernel/libmf_sdma_orch_v11.json
export V2_HOST_LIB=$B/site/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so
export U1_DEVICE_SO_PATH=$B/site/vllm_ascend/libmf310p_v2_trial_device.so
export U1_DEVICE_SHA256=2a19390c83c3fe6b0bdb8946d88990cca66b331f765ca445dbda2e787c2d2a19
export U1_PLUGIN_SHA256=8eb65de98c3d8664ff10b421c1d7ad31318dc720c3f5679cd4d34e187203bea1
export U1_HOST_SHA256=1dc46d7dcbe583a5dbb78b4c65a0b5e22f9f352976c53ed31b761fa744f91a30
export U1_KVER_SHA256=25e16b0af2f16989231ff0443698261257bd8142834783a0c1e4ad57cd9ebc78
export A1_FIVE_ARM_G0_PASSED=YES
export A1_FIVE_ARM_CARD_EXCLUSIVE=YES
export FIVE_ARM_STORE_PORT=18894
export LD_LIBRARY_PATH=$B/priv-libs:/usr/local/memfabric_hybrid/latest/aarch64-linux/lib64:/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$B/site:${PYTHONPATH:-}
unset E4_CONFIRM_APPROVED
