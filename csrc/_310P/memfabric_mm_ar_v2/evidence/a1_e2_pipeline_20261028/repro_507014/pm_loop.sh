#!/usr/bin/env bash
# 循环运行post-mortem取证复现，直到捕获HANG_CAUGHT或达到最大尝试次数。
set -uo pipefail
E2_WT=/home/w00804146/310P/e2wt
B=/tmp/opencode/a1-batch
BUILD=/tmp/opencode/e2-build-20261028
PERF=$E2_WT/csrc/_310P/memfabric_mm_ar_v2/test/perf
REPRO=/tmp/opencode/e2-repro
MANIFEST=/tmp/opencode/e2-diag/diagbuild/manifest_forensic.json
MAX=${1:-6}
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
export ASCEND_RT_VISIBLE_DEVICES=2,3
export MF_SDMA_ORCH_JSON=/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/hybm/aicpu_kernel/libmf_sdma_orch_v11.json
export V2_HOST_LIB=$B/site/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so
export U1_DEVICE_SO_PATH=$B/site/vllm_ascend/libmf310p_v2_trial_device.so
export U1_DEVICE_SHA256=2a19390c83c3fe6b0bdb8946d88990cca66b331f765ca445dbda2e787c2d2a19
export U1_PLUGIN_SHA256=8eb65de98c3d8664ff10b421c1d7ad31318dc720c3f5679cd4d34e187203bea1
export U1_HOST_SHA256=1dc46d7dcbe583a5dbb78b4c65a0b5e22f9f352976c53ed31b761fa744f91a30
export U1_KVER_SHA256=25e16b0af2f16989231ff0443698261257bd8142834783a0c1e4ad57cd9ebc78
export PYTHONPATH=$B/site:$PERF:$ASCEND_HOME_PATH/python/site-packages:$ASCEND_HOME_PATH/aarch64-linux/python:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$B/priv-libs:/usr/local/memfabric_hybrid/latest/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}
mkdir -p "$REPRO/logs"
cd "$E2_WT"
for attempt in $(seq 1 "$MAX"); do
  PORT=$((18893 + attempt))
  LOG="$REPRO/logs/pm_attempt${attempt}_$(date +%H%M%S).log"
  echo "=== PM ATTEMPT $attempt start $(date -Is) port $PORT ===" | tee -a "$REPRO/pm_attempts.log"
  torchrun --standalone --nproc-per-node=2 "$REPRO/e2_repro_trace.py" \
    --arm e2p --mode sequence --watcher off --out "$REPRO" --warmup 5 --iters 300 \
    --build-manifest "$MANIFEST" \
    --store-url "tcp://127.0.0.1:$PORT" \
    > "$LOG" 2>&1
  rc=$?
  echo "=== PM ATTEMPT $attempt rc=$rc $(date -Is) ===" | tee -a "$REPRO/pm_attempts.log"
  if grep -q "HANG_CAUGHT" "$LOG"; then
    echo "HANG CAPTURED in attempt $attempt; stopping loop" | tee -a "$REPRO/pm_attempts.log"
    break
  fi
  sleep 5
done
echo "PM_LOOP_DONE $(date -Is)" | tee -a "$REPRO/pm_attempts.log"
