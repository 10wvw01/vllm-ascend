#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SEL-01 三臂设备 Trace 路由审计（独立、不计入性能）。
# 每臂起一个与 SEL-01 完全同参（4096/2048 TP2 eager max-num-seqs 10）的短会话：
# 暖机 3×(4096入,1出) 后 /start_profile 捕获 3×(4096入,8出)（含 prefill+decode 相位），
# 等待双 Rank ascend_pt 导出，随后在服务器日志与设备 Trace 两侧核对：
#   stock: 无任何融合层使能日志；Trace 无 Mf310pWaveFusedB/V2TrialKernel/HybmSdmaOrchEpoch
#   v1:    每 Rank 40 层 Enable 日志；Trace 出现 Mf310pWaveFusedBKernel
#   v2:    每 Rank 40 层 Enable 日志 + [mf310p-v2] M= 标记；Trace 出现 V2TrialKernel
# 用法: sel01_route_audit.sh <arm:stock|v1|v2> <outdir> [port] [store_port]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
ARM=${1:?arm must be stock, v1 or v2}
OUT=${2:?outdir required}
PORT=${3:-8100}
STORE_PORT=${4:-8691}

MODEL=${MEMFABRIC_MM_AR_E2E_MODEL:-/home/models/Qwen/Qwen3.6-35B-A3B-w8a8}
DEVICES=${MEMFABRIC_MM_AR_E2E_DEVICES:-6,7}
ORCH_JSON=${MEMFABRIC_MM_AR_E2E_ORCH_JSON:-/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/hybm/aicpu_kernel/libmf_sdma_orch_v11.json}
mkdir -p "$OUT"

preflight_device_idle() {
    local chip dev bus aicore mem rest busy=0
    while read -r chip dev bus aicore mem rest; do
        [[ "$bus" =~ ^0000:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]$ ]] || continue
        case ",$DEVICES," in *",$dev,"*) ;; *) continue ;; esac
        mem=${mem%%/*}
        if [ "${aicore:-0}" -gt 10 ] || [ "${mem:-0}" -gt 4096 ]; then
            echo "device $dev BUSY (aicore=${aicore}% mem=${mem}MB) - refusing" >&2
            busy=1
        fi
    done < <(npu-smi info 2>/dev/null | tr -d '|' | tr -s ' ')
    return "$busy"
}
wait_devices_drained() {
    local deadline=$((SECONDS + 300)) chip dev bus aicore mem rest
    while [ $SECONDS -lt $deadline ]; do
        local dirty=0
        while read -r chip dev bus aicore mem rest; do
            [[ "$bus" =~ ^0000:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]$ ]] || continue
            case ",$DEVICES," in *",$dev,"*) ;; *) continue ;; esac
            mem=${mem%%/*}
            [ "${mem:-0}" -gt 2500 ] && dirty=1
        done < <(npu-smi info 2>/dev/null | tr -d '|' | tr -s ' ')
        [ "$dirty" = 0 ] && return 0
        sleep 5
    done
    echo "warning: devices still not drained after 300s" >&2
}

preflight_device_idle || { echo "[$ARM] devices busy, aborting" >&2; exit 1; }

export ASCEND_RT_VISIBLE_DEVICES=$DEVICES
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR \
    VLLM_ASCEND_310P_MEMFABRIC_STORE_URL \
    VLLM_ASCEND_310P_MEMFABRIC_LOCAL_BYTES \
    VLLM_ASCEND_310P_MEMFABRIC_MM_AR_BATCH_BASEM_COUNT \
    VLLM_ASCEND_310P_MEMFABRIC_MM_AR_WARMUP_FALLBACK \
    VLLM_ASCEND_310P_MEMFABRIC_MM_AR_V2_SERVE \
    VLLM_ASCEND_310P_MEMFABRIC_V2_CHUNK_ROWS \
    VLLM_ASCEND_310P_MEMFABRIC_V2_OUTPUT_SLOTS MF_SDMA_ORCH_JSON
[ -f /usr/local/memfabric_hybrid/set_env.sh ] && source /usr/local/memfabric_hybrid/set_env.sh

case "$ARM" in
  stock) ;;
  v1)
    export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR=1
    export VLLM_ASCEND_310P_MEMFABRIC_STORE_URL="tcp://127.0.0.1:${STORE_PORT}"
    export VLLM_ASCEND_310P_MEMFABRIC_LOCAL_BYTES=$((96 * 1024 * 1024))
    export VLLM_ASCEND_310P_MEMFABRIC_MM_AR_BATCH_BASEM_COUNT=1
    export VLLM_ASCEND_310P_MEMFABRIC_MM_AR_WARMUP_FALLBACK=1
    export MF_SDMA_ORCH_JSON="$ORCH_JSON" ;;
  v2)
    export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR=1
    export VLLM_ASCEND_310P_MEMFABRIC_STORE_URL="tcp://127.0.0.1:${STORE_PORT}"
    export VLLM_ASCEND_310P_MEMFABRIC_LOCAL_BYTES=$((96 * 1024 * 1024))
    export VLLM_ASCEND_310P_MEMFABRIC_MM_AR_BATCH_BASEM_COUNT=1
    export VLLM_ASCEND_310P_MEMFABRIC_MM_AR_WARMUP_FALLBACK=1
    export VLLM_ASCEND_310P_MEMFABRIC_MM_AR_V2_SERVE=1
    export VLLM_ASCEND_310P_MEMFABRIC_V2_CHUNK_ROWS=1024
    export VLLM_ASCEND_310P_MEMFABRIC_V2_OUTPUT_SLOTS=4
    export MF_SDMA_ORCH_JSON="$ORCH_JSON" ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac

LOG="$OUT/serve.log"
echo "[$ARM] starting audit serve $(date +%H:%M:%S)"
setsid vllm serve "$MODEL" \
    --host 0.0.0.0 --port "$PORT" --tensor-parallel-size 2 \
    --quantization ascend --served-model-name qwen3.6-35b-a3b-w8a8 \
    --trust-remote-code --dtype float16 --max-num-seqs 10 \
    --max-num-batched-tokens 4096 --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching --enable-chunked-prefill \
    --async-scheduling \
    --mamba-ssm-cache-dtype float16 --mamba-cache-mode align \
    --additional-config '{"enable_flashcomm1":false,"enable_flashcomm2_parallel_size":0,"enable_prefill_mc2":false,"enable_fused_mc2":0,"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":false}}' \
    --enforce-eager \
    --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$OUT\",\"torch_profiler_with_stack\":false}" \
    > "$LOG" 2>&1 &
SPID=$!

cleanup() { kill -TERM -- -"$SPID" 2>/dev/null || kill -TERM "$SPID" 2>/dev/null || true; wait "$SPID" 2>/dev/null || true; wait_devices_drained; }
trap cleanup EXIT

DEADLINE=$((SECONDS + 2400))
until curl -sf -m 3 "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; do
    kill -0 "$SPID" 2>/dev/null || { echo "[$ARM] serve died" >&2; tail -30 "$LOG" >&2; exit 1; }
    [ "$SECONDS" -ge "$DEADLINE" ] && { echo "[$ARM] health timeout" >&2; exit 1; }
    sleep 5
done
echo "[$ARM] healthy $(date +%H:%M:%S) enabled_layers_log=$(grep -c 'Enable 310P3 TP=2 MemFabric unquantized mm_ar' "$LOG" || true)"

python3 - "$PORT" "$OUT/requests.json" <<'PYEOF'
import json, sys, time, urllib.request
port, out_json = sys.argv[1], sys.argv[2]
URL = f"http://127.0.0.1:{port}"
ids = [64] * 4096
def send(max_tokens):
    payload = json.dumps({"model": "qwen3.6-35b-a3b-w8a8", "prompt": ids,
                          "max_tokens": max_tokens, "ignore_eos": True}).encode()
    req = urllib.request.Request(URL + "/v1/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    body = urllib.request.urlopen(req, timeout=600).read()
    return {"latency_s": round(time.perf_counter() - t0, 3),
            "completion_tokens": json.loads(body)["usage"]["completion_tokens"]}
def profile(cmd):
    urllib.request.urlopen(urllib.request.Request(
        URL + f"/{cmd}", data=b"", method="POST",
        headers={"Content-Type": "application/json"}), timeout=600).read()
warm = [send(1) for _ in range(3)]
profile("start_profile")
win = [send(8) for _ in range(3)]
profile("stop_profile")
json.dump({"warmup": warm, "window": win}, open(out_json, "w"), indent=2)
print("warmup:", [r["latency_s"] for r in warm], "window:", [r["latency_s"] for r in win])
PYEOF
[ $? -eq 0 ] || { echo "[$ARM] request phase failed" >&2; exit 1; }

for i in $(seq 1 90); do
    n=$(find "$OUT" -maxdepth 1 -name "*_ascend_pt" -type d | wc -l)
    [ "$n" -ge 2 ] && break
    sleep 5
done
N_TRACE=$(find "$OUT" -maxdepth 1 -name "*_ascend_pt" -type d | wc -l)
echo "[$ARM] trace_dirs=$N_TRACE"
[ "$N_TRACE" -ge 2 ] || { echo "[$ARM] missing rank trace exports" >&2; exit 1; }
sleep 10  # let the exporter finish writing csv

trap - EXIT
cleanup
echo "[$ARM] DONE $(date +%H:%M:%S)"
