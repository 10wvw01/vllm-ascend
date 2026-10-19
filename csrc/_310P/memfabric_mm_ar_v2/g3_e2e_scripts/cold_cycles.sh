#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 20 fresh torchrun cycles of the TP2 mixed-route test: cold-start robustness.
# Records per-cycle pass/fail; used to characterize first-load behavior of a
# freshly deployed orchestrator kernel and steady-state cold-start stability.
set -uo pipefail
OUT_DIR=${1:-/tmp/opencode/g3/cold_cycles}
DEVS=${DEVS:-2,3}
STORE_PORT=${STORE_PORT:-8582}
CYCLES=${CYCLES:-20}
mkdir -p "$OUT_DIR"
cd /home/w00804146/310P/vllm-ascend
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=$DEVS
export VLLM_ASCEND_310P_MEMFABRIC_STORE_URL="tcp://127.0.0.1:${STORE_PORT}"
pass=0; fail=0
for i in $(seq 1 "$CYCLES"); do
    timeout 600 torchrun --standalone --nproc-per-node=2 \
        tests/e2e/_310p/tp2_v2_mixed_route_correctness.py \
        --rows 2048 2049 3073 4095 4097 --chunk-rows 1024 --reps 1 \
        > "$OUT_DIR/cycle_$i.log" 2>&1
    rc=$?
    # The per-device fixed-address meta region is exclusive and releases
    # asynchronously after process exit (observed 2026-10-18: back-to-back
    # cycles self-sustain MetaSetDeviceAccess 507018/507899 failures).
    # Drain between cycles so the next pool creation starts clean.
    sleep 75
    if [ $rc -eq 0 ] && grep -q "TP2 V2 mixed-route PASS" "$OUT_DIR/cycle_$i.log"; then
        pass=$((pass+1)); echo "cycle $i: PASS" | tee -a "$OUT_DIR/summary.txt"
    else
        fail=$((fail+1)); reason=$(grep -m1 -E "AssertionError|RuntimeError|Error" "$OUT_DIR/cycle_$i.log" | head -c 200)
        echo "cycle $i: FAIL rc=$rc $reason" | tee -a "$OUT_DIR/summary.txt"
    fi
done
echo "TOTAL pass=$pass fail=$fail" | tee -a "$OUT_DIR/summary.txt"
