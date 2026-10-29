#!/usr/bin/env bash
# A1-MM-SHADOW campaign launcher: R2-proven private loading + campaign entry.
# Run from neutral cwd. Result dir must NOT exist (campaign creates it).
set -Eeuo pipefail
B=/tmp/opencode/a1-batch
W=/tmp/opencode/a1-r0-worktree
P=$W/csrc/_310P/memfabric_mm_ar_v2/test/perf
OUT="$1"
# 全新或续跑(由campaign自身的同源校验语义决定跳过/重跑);setsid启动防会话清理击杀。
[[ -n "$OUT" && "$OUT" == /* ]] || { echo "absolute OUT required" >&2; exit 2; }

export ASCEND_RT_VISIBLE_DEVICES=2,3
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
export MEMFABRIC_HYBRID_HOME_PATH=/usr/local/memfabric_hybrid/1.2.0
export MF_SDMA_ORCH_JSON=/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/hybm/aicpu_kernel/libmf_sdma_orch_v11.json
export U1_DEVICE_SO_PATH=$B/priv-libs/libmf310p_v2_trial_device.so
export U1_CORE_SO_LIB64=$B/priv-libs/libmf_hybm_core.so
export A1_MM_SHADOW_V1_DEVICE_LIB=$B/site/vllm_ascend/libmf310p_device.so
# approval gates (set after G0/exclusive/restore verified this batch)
export U1_G0_APPROVED=YES
export U1_EXCLUSIVE_APPROVED=YES
export U1_RESTORE_PLAN_VERIFIED=YES
export A1_MM_SHADOW_EXCLUSIVE_APPROVED=YES
export A1_MM_SHADOW_STORE_PORT=18884
# safety: never bypass KFC / first-write / epoch budget
unset MF_KFC_BYPASS MF310P_V2_NOWARMUP MF_SDMA_ORCH_EPOCH_BUDGET_S 2>/dev/null || true
# R2-proven private loading order (campaign re-prepends deterministically):
export LD_LIBRARY_PATH=$B/priv-libs:$B/site/vllm_ascend:/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/cann-9.1.0/python/site-packages:/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe

cd /tmp/opencode
bash "$P/a1_mm_shadow_campaign.sh" "$OUT"
