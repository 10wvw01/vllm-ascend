#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# One isolated Qwen3.6 Eager Prefill session; no auto reruns or card switching.
set -Eeuo pipefail
[[ "$#" == 4 ]] || { echo "usage: $0 stock|v2 l2_profile|l3_event SESSION_ID NEW_ABS_DIR" >&2; exit 2; }
ARM="$1"; MODE="$2"; SID="$3"; OUT="$4"
[[ "$ARM" == stock || "$ARM" == v2 ]] || exit 2
[[ "$MODE" == l2_profile || "$MODE" == l3_event ]] || exit 2
[[ "$SID" =~ ^[A-Za-z0-9_-]{2,32}$ ]] || exit 2
[[ "$OUT" == /* && ! -e "$OUT" && -d "$(dirname "$OUT")" ]] || { echo "output must be fresh absolute dir" >&2; exit 2; }
: "${A1_SERVE_MODEL_PATH:?explicit verified model directory required}"
: "${A1_SERVE_MODEL_NAME:?verified model service name required}"
: "${A1_SERVE_PORT:?explicit private idle port required}"
: "${A1_RUNTIME_IDENTITY_JSON:?current R1 load identity JSON required}"
: "${ASCEND_RT_VISIBLE_DEVICES:?current approved physical Die pair required}"
: "${MF_SDMA_ORCH_JSON:?verified KVER11 path required}"
[[ "$A1_SERVE_MODEL_PATH" == /* && -d "$A1_SERVE_MODEL_PATH" ]] || exit 2
[[ "$A1_RUNTIME_IDENTITY_JSON" == /* && -f "$A1_RUNTIME_IDENTITY_JSON" ]] || exit 2
[[ "$MF_SDMA_ORCH_JSON" == /*/libmf_sdma_orch_v11.json && -f "$MF_SDMA_ORCH_JSON" ]] || exit 2
[[ "$ASCEND_RT_VISIBLE_DEVICES" =~ ^[0-9]+,[0-9]+$ ]] || exit 2
[[ "${ASCEND_RT_VISIBLE_DEVICES%,*}" != "${ASCEND_RT_VISIBLE_DEVICES#*,}" ]] || exit 2
[[ "$A1_SERVE_PORT" =~ ^[0-9]+$ && "$A1_SERVE_PORT" -ge 1024 && "$A1_SERVE_PORT" -le 65535 ]] || exit 2
[[ "${U1_G0_APPROVED:-}" == YES && "${U1_EXCLUSIVE_APPROVED:-}" == YES &&
   "${U1_RESTORE_PLAN_VERIFIED:-}" == YES ]] || {
  echo "same-batch G0/exclusive/rollback attestation required" >&2; exit 2;
}
[[ -z "${MF_KFC_BYPASS+x}" && -z "${MF310P_V2_NOWARMUP+x}" &&
   -z "${MF310P_V2_AR_ONLY_NOWARMUP+x}" &&
   -z "${MF_SDMA_ORCH_DEPLOY_HELPER+x}" ]] || { echo "unsafe MemFabric env override" >&2; exit 2; }
[[ "${MF_SDMA_ORCH_EPOCH_BUDGET_S:-}" != 4294967295 ]] || { echo "persistent epoch forbidden" >&2; exit 2; }
# 仅R2诊断V2会话允许显式5s/15s；非R2保持原脚本语义。
if [[ -n "${A1_R2_EPOCH_ARM:-}" ]]; then
  [[ "$ARM" == v2 && "$MODE" == l3_event &&
     "${MF_SDMA_ORCH_EPOCH_BUDGET_S:-}" == "$A1_R2_EPOCH_ARM" &&
     ( "$A1_R2_EPOCH_ARM" == 5 || "$A1_R2_EPOCH_ARM" == 15 ) ]] || {
    echo "R2 epoch requires matched 5/15 second V2-only no-profiler session" >&2; exit 2;
  }
fi
for cmd in python3 vllm setsid curl timeout sha256sum npu-smi; do
  command -v "$cmd" >/dev/null || { echo "missing $cmd" >&2; exit 2; }
done
PERF="$(cd "$(dirname "$0")" && pwd)"
# Before creating the directory, require an unused local port. Never attach to another service.
python3 - "$A1_SERVE_PORT" <<'PY'
import socket,sys
with socket.socket() as s:
    s.settimeout(.5)
    if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0:
        raise SystemExit("refusing occupied service port")
PY
mkdir -m 700 "$OUT"
# Manifest identity is supplied by R1 verified runtime audit, never inferred from a candidate file's existence.
python3 - "$A1_RUNTIME_IDENTITY_JSON" "$OUT/manifest.json" "$ARM" "$MODE" "$SID" <<'PY'
import json,os,sys
src,dest,arm,mode,sid=sys.argv[1:]
identity=json.load(open(src,encoding="utf-8"))
required=("model","model_revision","physical_devices","host_so_sha256",
          "device_so_sha256","plugin_so_sha256","kver_sha256")
if not isinstance(identity,dict) or any(not isinstance(identity.get(k),str)
                                       or not identity[k] for k in required):
    raise SystemExit("missing frozen runtime identity")
if identity["physical_devices"] != os.environ["ASCEND_RT_VISIBLE_DEVICES"]:
    raise SystemExit("runtime identity physical mapping differs")
if identity["host_so_sha256"] != "1dc46d7dcbe583a5dbb78b4c65a0b5e22f9f352976c53ed31b761fa744f91a30":
    raise SystemExit("runtime Host identity not verified")
if identity["device_so_sha256"] != "2a19390c83c3fe6b0bdb8946d88990cca66b331f765ca445dbda2e787c2d2a19":
    raise SystemExit("runtime Device D4 identity not verified")
if identity["plugin_so_sha256"] != "8eb65de98c3d8664ff10b421c1d7ad31318dc720c3f5679cd4d34e187203bea1":
    raise SystemExit("runtime Plugin identity not verified")
if identity["kver_sha256"] != "25e16b0af2f16989231ff0443698261257bd8142834783a0c1e4ad57cd9ebc78":
    raise SystemExit("runtime KVER identity not verified")
assert arm in ("stock","v2") and mode in ("l2_profile","l3_event")
m={"session_id":sid,"arm":arm,"input_tokens":4096,"output_tokens":1,
   "epoch_budget_s": (int(os.environ["MF_SDMA_ORCH_EPOCH_BUDGET_S"])
                      if os.environ.get("A1_R2_EPOCH_ARM") else None),
   "epoch_budget_source": ("explicit_R2_controlled_env"
                           if os.environ.get("A1_R2_EPOCH_ARM") else "not_recorded"),
   "concurrency":1,"tp":2,"eager":True,"profiler":mode=="l2_profile",
   "warmup_requests":6,"measured_requests":5 if mode=="l2_profile" else 30,
   "identity":identity,"identity_kind":"site-audited runtime; must retain raw evidence"}
with open(dest,"x",encoding="utf-8") as f: json.dump(m,f,indent=2)
PY
# Strict opt-in for test-only event observer, never for profiler and never on production default.
if [[ "$MODE" == l3_event ]]; then
  export A1_310P_PREFILL_EVENTS_DIR="$OUT"
  unset VLLM_TORCH_PROFILER_DIR
else
  unset A1_310P_PREFILL_EVENTS_DIR
fi
# Both Stock/V2 run on fresh independent vLLM serve sessions, SAME software/model/card.
unset VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR \
  VLLM_ASCEND_310P_MEMFABRIC_STORE_URL \
  VLLM_ASCEND_310P_MEMFABRIC_LOCAL_BYTES \
  VLLM_ASCEND_310P_MEMFABRIC_MM_AR_V2_SERVE \
  VLLM_ASCEND_310P_MEMFABRIC_MM_AR_WARMUP_FALLBACK \
  VLLM_ASCEND_310P_MEMFABRIC_V2_CHUNK_ROWS \
  VLLM_ASCEND_310P_MEMFABRIC_V2_OUTPUT_SLOTS
if [[ "$ARM" == v2 ]]; then
  : "${A1_SERVE_STORE_PORT:?explicit verified unused MemFabric store port required}"
  [[ "$A1_SERVE_STORE_PORT" =~ ^[0-9]+$ && "$A1_SERVE_STORE_PORT" -ge 1024 &&
     "$A1_SERVE_STORE_PORT" -le 65535 && "$A1_SERVE_STORE_PORT" != "$A1_SERVE_PORT" ]] || {
    echo "invalid or conflicting MemFabric store port" >&2; exit 2;
  }
  python3 - "$A1_SERVE_STORE_PORT" <<'PYSTORE'
import socket,sys
with socket.socket() as s:
    s.settimeout(.5)
    if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0:
        raise SystemExit("refusing occupied MemFabric store port")
PYSTORE
  export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR=1
  export VLLM_ASCEND_310P_MEMFABRIC_STORE_URL="tcp://127.0.0.1:${A1_SERVE_STORE_PORT}"
  export VLLM_ASCEND_310P_MEMFABRIC_LOCAL_BYTES=$((96 * 1024 * 1024))
  export VLLM_ASCEND_310P_MEMFABRIC_MM_AR_WARMUP_FALLBACK=1
  export VLLM_ASCEND_310P_MEMFABRIC_MM_AR_V2_SERVE=1
  export VLLM_ASCEND_310P_MEMFABRIC_V2_CHUNK_ROWS=1024
  export VLLM_ASCEND_310P_MEMFABRIC_V2_OUTPUT_SLOTS=4
fi
export VLLM_WORKER_MULTIPROC_METHOD=spawn
cmd=(vllm serve "$A1_SERVE_MODEL_PATH" --host 127.0.0.1 --port "$A1_SERVE_PORT"
     --tensor-parallel-size 2 --quantization ascend
     --served-model-name "$A1_SERVE_MODEL_NAME"
     --trust-remote-code --dtype float16
     --max-num-seqs 16 --max-num-batched-tokens 4096 --max-model-len 8192
     --gpu-memory-utilization 0.90 --no-enable-prefix-caching
     --enable-chunked-prefill --async-scheduling
     --mamba-ssm-cache-dtype float16 --mamba-cache-mode align
     --additional-config '{"enable_flashcomm1":false,"enable_flashcomm2_parallel_size":0,"enable_prefill_mc2":false,"enable_fused_mc2":0,"ascend_compilation_config":{"fuse_norm_quant":false,"enable_npugraph_ex":false}}'
     --enforce-eager)
if [[ "$MODE" == l2_profile ]]; then
  cmd+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$OUT\",\"torch_profiler_with_stack\":false}")
fi
pid=''
cleanup() {
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    # Only terminate the process group whose session leader was created here.
    sid="$(ps -o sid= -p "$pid" 2>/dev/null | tr -d ' ')"
    if [[ "$sid" == "$pid" ]]; then
      kill -TERM -- "-$pid" 2>/dev/null || true
      # 只检查/清理我们创建的会话组，最多30秒，避免遗留新实验Serve。
      for ((i=0; i<30; i++)); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
      done
      if kill -0 "$pid" 2>/dev/null; then
        kill -KILL -- "-$pid" 2>/dev/null || true
      fi
    fi
    wait "$pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
npu-smi info > "$OUT/npu_before.txt" 2>&1
setsid "${cmd[@]}" > "$OUT/serve.log" 2>&1 &
pid="$!"
deadline=$((SECONDS + 2400))
while ! curl -sf --max-time 3 "http://127.0.0.1:${A1_SERVE_PORT}/health" >/dev/null; do
  kill -0 "$pid" 2>/dev/null || { echo "own serve exited: $OUT/serve.log" >&2; exit 1; }
  ((SECONDS < deadline)) || { echo "own serve startup timeout" >&2; exit 1; }
  sleep 5
done
req=(python3 "$PERF/a1_prefill_http_requests.py"
     --port "$A1_SERVE_PORT" --model "$A1_SERVE_MODEL_NAME"
     --session "$SID" --out "$OUT/client_requests.jsonl"
     --warmup 6 --measured 30)
if [[ "$MODE" == l2_profile ]]; then
  req[13]=5
  req+=(--profile)
fi
"${req[@]}" > "$OUT/client.log" 2>&1 || { echo "Prefill client failed; save logs" >&2; exit 1; }
# Profiler export is asynchronous; never kill own Serve before both Rank traces appear.
if [[ "$MODE" == l2_profile ]]; then
  traced=0
  for ((attempt=0; attempt<60; attempt++)); do
    traced="$(find "$OUT" -maxdepth 1 -name '*_ascend_pt' -type d | wc -l)"
    [[ "$traced" -ge 2 ]] && break
    sleep 5
  done
  [[ "$traced" -ge 2 ]] || { echo "both Rank profiler dirs not exported" >&2; exit 1; }
else
  # Event flush never synchronizes devices. If a Rank cannot produce samples,
  # mark the session NOT_READY rather than inventing an L3 timing.
  have=0
  for ((attempt=0; attempt<30; attempt++)); do
    if compgen -G "$OUT/a1_prefill_rank0_pid*.jsonl" >/dev/null &&
       compgen -G "$OUT/a1_prefill_rank1_pid*.jsonl" >/dev/null; then
      have=1
      break
    fi
    sleep 2
  done
  [[ "$have" == 1 ]] || { echo "missing full dual-Rank event traces: L3_MEASUREMENT_NOT_READY" >&2; exit 1; }
fi
npu-smi info > "$OUT/npu_after.txt" 2>&1
echo "A1 session $SID/$MODE completed; raw trace/route/actual loaded identity MUST be independently audited"
