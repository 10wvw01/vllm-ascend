#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 固定输入4096 / 输出2048 / TP=2 / 并发10 / Eager，禁止混用D4的512输出。
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
BENCH="$ROOT/tests/e2e/_310p/bench_memfabric_mm_ar_serve.sh"
ANALYZER="$ROOT/tests/e2e/_310p/sel01_analyze.py"
OUT="$1"
case "$OUT" in /*) ;; *) echo "output directory must be absolute" >&2; exit 2 ;; esac
[ ! -e "$OUT" ] || { echo "output exists, refusing overwrite" >&2; exit 2; }
[ -d "$MEMFABRIC_MM_AR_E2E_MODEL" ] || { echo "pin model directory" >&2; exit 2; }
case "$MEMFABRIC_MM_AR_E2E_MODEL" in *Qwen3.6-35B-A3B-w8a8*) ;; *) echo "wrong model" >&2; exit 2;; esac
[ -f "$MEMFABRIC_MM_AR_E2E_ORCH_JSON" ] || { echo "KVER11 JSON missing" >&2; exit 2; }
case "$MEMFABRIC_MM_AR_E2E_ORCH_JSON" in /*/libmf_sdma_orch_v11.json) ;; *) echo "pin absolute KVER11 JSON" >&2; exit 2;; esac
[[ "$MEMFABRIC_MM_AR_E2E_DEVICES" =~ ^[0-9]+,[0-9]+$ ]] || { echo "set two physical die IDs" >&2; exit 2; }
[ "$SEL01_APPROVED_ENV" = YES ] || { echo "first check exclusive die, runtime SO SHA/KVER, then set SEL01_APPROVED_ENV=YES" >&2; exit 2; }
[ -z "$(printenv MF310P_V2_NOWARMUP 2>/dev/null || :)" ] || { echo "do not bypass safety warmup" >&2; exit 2; }
[ -z "$(printenv VLLM_ASCEND_310P_MEMFABRIC_MM_AR_TRACE 2>/dev/null || :)" ] || { echo "trace must be off for performance" >&2; exit 2; }
[ -z "$(printenv MEMFABRIC_MM_AR_E2E_V2 2>/dev/null || :)" ] || { echo "SEL01 owns route" >&2; exit 2; }
[ -z "$(printenv MEMFABRIC_MM_AR_E2E_BATCH_BASEM_COUNT 2>/dev/null || :)" ] || { echo "SEL01 owns q" >&2; exit 2; }
cd "$ROOT"
export MEMFABRIC_MM_AR_E2E_INPUT_LEN=4096
export MEMFABRIC_MM_AR_E2E_OUTPUT_LEN=2048
export MEMFABRIC_MM_AR_E2E_CONCURRENCY=10
export MEMFABRIC_MM_AR_E2E_MAX_NUM_SEQS=10
export MEMFABRIC_MM_AR_E2E_NUM_PROMPTS=50
export MEMFABRIC_MM_AR_E2E_REPEAT=1
export MEMFABRIC_MM_AR_E2E_DFLASH=0
export MEMFABRIC_MM_AR_E2E_BATCH_BASEM_COUNT=1
export MEMFABRIC_MM_AR_E2E_V2_CHUNK_ROWS=1024
export MEMFABRIC_MM_AR_E2E_V2_OUTPUT_SLOTS=4
mkdir -p "$OUT"
python3 - "$OUT" <<'PY'
import hashlib, json, os, pathlib, subprocess, sys
out=pathlib.Path(sys.argv[1])
model=pathlib.Path(os.environ["MEMFABRIC_MM_AR_E2E_MODEL"])
orch=pathlib.Path(os.environ["MEMFABRIC_MM_AR_E2E_ORCH_JSON"])
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()
data={"git_sha":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
"model":str(model),"config_sha256":sha(model/"config.json"),
"orch_json":str(orch),"orch_json_sha256":sha(orch),
"devices":os.environ["MEMFABRIC_MM_AR_E2E_DEVICES"],
"input_tokens":4096,"output_tokens":2048,"tp":2,"max_num_seqs":10,
"max_concurrency":10,"num_prompts":50,"mode":"eager","V1_q":1,"V2_chunk_rows":1024}
(out/"manifest.json").write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
PY
i=0
# 两轮正反交叉，各场启动独立server，遇失败即保全并终止，不自动筛掉挂死。
for arm in stock v1 v2 v2 v1 stock stock v1 v2 v2 v1 stock; do
    i=$((i+1))
    sid="$(printf '%02d' "$i")"
    session="$OUT/$sid-$arm"
    mkdir "$session"
    export MEMFABRIC_MM_AR_E2E_V2=0
    case "$arm" in
      stock) combo="stock eager" ;;
      v1) combo="fused eager" ;;
      v2) combo="fused eager"; export MEMFABRIC_MM_AR_E2E_V2=1 ;;
    esac
    echo "SEL-01 $i/12 arm=$arm input4096 output2048 tp2 batch-cap10"
    if ! bash "$BENCH" "$session" "$combo" >"$session/session.log" 2>&1; then
      echo "FAIL $sid $arm: keep all logs, check Gate E and environment" >&2
      exit 1
    fi
    python3 "$ANALYZER" verify-one --session "$session" --arm "$arm" --output "$session/verified.json"
done
python3 "$ANALYZER" summarize --root "$OUT" --output "$OUT/sel01_summary.json"
