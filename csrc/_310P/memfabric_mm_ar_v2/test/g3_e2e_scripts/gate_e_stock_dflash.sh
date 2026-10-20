#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Gate E 附加场景（用户 2026-10-20 追加要求，随后指示移除投机配置）：
# STOCK + FULL_DECODE_ONLY 图模式，无投机推理。
# 与原 Gate E 的差异：无 --enforce-eager；新增 --compilation-config（FULL_DECODE_ONLY, capture sizes [10]）；
# 负载 output 2048、50 prompts。
# 其余（TP=2、两卡 0/1、模型、并发 10、620s、结果入 gate_e_forensics）保持不变。
# 注：--speculative-config dflash 已按用户指示移除；其连带要求 VLLM_USE_V2_MODEL_RUNNER=1
# （进而硬性要求 Triton，本机无 triton-ascend）一并移除，走默认 V1 model runner。
set -uo pipefail
V2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OUT_DIR=${1:-"${V2_ROOT}/tmp/opencode/gate_e_forensics/manual_$(date -u +%Y%m%dT%H%M%SZ)"}
mkdir -p "$OUT_DIR"
cd "${V2_WORKTREE:-/home/w00804146/310P/vllm-ascend}"

MODEL=${GATE_E_MODEL:-/home/models/Qwen/Qwen3.6-35B-A3B-w8a8}
DFLASH_MODEL=${GATE_E_DFLASH_MODEL:-/home/models/Qwen/Qwen3.6-35B-A3B-DFlash}
PORT=${GATE_E_PORT:-8000}
DEVICES=${GATE_E_DEVICES:-0,1}
if [[ "${MF_SDMA_ORCH_EPOCH_BUDGET_S:-}" == "4294967295" ]]; then
    echo "禁止在 Gate E 使用常驻 Epoch；请取消 MF_SDMA_ORCH_EPOCH_BUDGET_S=4294967295" >&2
    exit 2
fi
DURATION_S=${GATE_E_DURATION_S:-620}

export ASCEND_RT_VISIBLE_DEVICES=$DEVICES
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR VLLM_ASCEND_310P_MEMFABRIC_STORE_URL \
      VLLM_ASCEND_310P_MEMFABRIC_LOCAL_BYTES VLLM_ASCEND_310P_MEMFABRIC_MM_AR_BATCH_BASEM_COUNT \
      VLLM_ASCEND_310P_MEMFABRIC_MM_AR_WARMUP_FALLBACK VLLM_ASCEND_310P_MEMFABRIC_MM_AR_V2_SERVE \
      VLLM_ASCEND_310P_MEMFABRIC_V2_CHUNK_ROWS VLLM_ASCEND_310P_MEMFABRIC_V2_OUTPUT_SLOTS MF_SDMA_ORCH_JSON
source /usr/local/memfabric_hybrid/set_env.sh

LOG="$OUT_DIR/serve.log"
echo "[$(date +%H:%M:%S)] starting STOCK+GRAPH serve (FULL_DECODE_ONLY, no speculative)"
setsid vllm serve "$MODEL" \
    --host 0.0.0.0 --port $PORT --tensor-parallel-size 2 \
    --quantization ascend --served-model-name qwen3.6-35b-a3b-w8a8 \
    --trust-remote-code --dtype float16 --max-num-seqs 16 \
    --max-num-batched-tokens 4096 --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching --enable-chunked-prefill \
    --async-scheduling \
    --mamba-ssm-cache-dtype float16 --mamba-cache-mode align \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[10]}' \
    --additional-config '{"enable_flashcomm1":false,"enable_flashcomm2_parallel_size":0,"enable_prefill_mc2":false,"enable_fused_mc2":0,"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":false}}' \
    > "$LOG" 2>&1 &
SERVE_PID=$!

DEADLINE=$((SECONDS + 2400))
until curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; do
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
        echo "serve exited early"; tail -40 "$LOG"; exit 1
    fi
    [ "$SECONDS" -ge "$DEADLINE" ] && { echo "health timeout"; kill -TERM -- -"$SERVE_PID"; exit 1; }
    sleep 5
done
echo "[$(date +%H:%M:%S)] healthy (stock+dflash); v2_layers=$(grep -c 'Enable 310P3 TP=2 MemFabric unquantized mm_ar' "$LOG" || true) spec=$(grep -ciE 'dflash|speculative' "$LOG" || true)"

END=$((SECONDS + DURATION_S))
REP=0
while [ $SECONDS -lt $END ]; do
    REP=$((REP+1))
    HF_HUB_OFFLINE=1 vllm bench serve --backend vllm \
        --endpoint /v1/completions --base-url "http://127.0.0.1:${PORT}" \
        --model qwen3.6-35b-a3b-w8a8 --tokenizer "$MODEL" \
        --dataset-name random --random-input-len 4096 --random-output-len 2048 \
        --num-prompts 50 --max-concurrency 10 --ignore-eos \
        --percentile-metrics ttft,tpot,itl --metric-percentiles "50,90,99,99.9" \
        --save-result --result-dir "$OUT_DIR" --result-filename "stability_rep${REP}.json" \
        >> "$OUT_DIR/bench.log" 2>&1 || { echo "bench rep $REP failed"; break; }
done
echo "[$(date +%H:%M:%S)] load done after $REP reps"

sleep 5
kill -TERM -- -"$SERVE_PID" 2>/dev/null || true
wait "$SERVE_PID" 2>/dev/null || true

REQS=$(python - <<'EOF'
import glob, json, os
total = 0
errs = 0
for f in sorted(glob.glob(os.environ["OUT_DIR"] + "/stability_rep*.json")):
    d = json.load(open(f))
    total += d.get("completed", 0)
    errs += d.get("errors", 0) if isinstance(d.get("errors"), int) else 0
print(f"{total} {errs}")
EOF
)
echo "requests_completed errors: $REQS"
TIMEOUTS=$(grep -ciE "507014|timeout.*kernel|aicpu.*timeout" "$LOG" || true)
DEADLOCK=$(grep -ciE "deadlock|hang|watchdog" "$LOG" || true)
echo "timeout_hits=$TIMEOUTS deadlock_hits=$DEADLOCK"
grep -cE "ERROR|Error" "$LOG" || true
