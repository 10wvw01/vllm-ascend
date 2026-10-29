#!/usr/bin/env bash
# A1五层五臂十场：私有装载环境（复用R1验证的私有site+priv-libs方案，共享系统文件不动）
set -Eeuo pipefail
FIVE_ARM_WT=/home/w00804146/310P/vllm-ascend-five-arm
B=/tmp/opencode/a1-batch
PERF=$FIVE_ARM_WT/csrc/_310P/memfabric_mm_ar_v2/test/perf
export ASCEND_RT_VISIBLE_DEVICES=2,3
export MF_SDMA_ORCH_JSON=/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/hybm/aicpu_kernel/libmf_sdma_orch_v11.json
export V2_HOST_LIB=$B/site/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so
export U1_DEVICE_SO_PATH=$B/site/vllm_ascend/libmf310p_v2_trial_device.so
export U1_DEVICE_SHA256=2a19390c83c3fe6b0bdb8946d88990cca66b331f765ca445dbda2e787c2d2a19
export U1_PLUGIN_SHA256=8eb65de98c3d8664ff10b421c1d7ad31318dc720c3f5679cd4d34e187203bea1
export U1_HOST_SHA256=1dc46d7dcbe583a5dbb78b4c65a0b5e22f9f352976c53ed31b761fa744f91a30
export U1_KVER_SHA256=25e16b0af2f16989231ff0443698261257bd8142834783a0c1e4ad57cd9ebc78
export A1_FIVE_ARM_G0_PASSED=YES
export A1_FIVE_ARM_CARD_EXCLUSIVE=YES
export FIVE_ARM_STORE_PORT=18884
export LD_LIBRARY_PATH=$B/priv-libs:/usr/local/memfabric_hybrid/latest/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$B/site:${PYTHONPATH:-}
cd "$FIVE_ARM_WT"
exec bash "$PERF/five_arm_campaign.sh" /tmp/opencode/five-arm-campaign-new /tmp/opencode/five-arm-build-new
