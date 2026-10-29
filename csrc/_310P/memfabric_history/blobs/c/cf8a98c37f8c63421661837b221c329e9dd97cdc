#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SEL-01 容错续跑器：主驱动 sel01_three_arm_eager.sh 在任一场失败即停（正确的
# 停止门行为）。本脚本用于停止门触发、现场已保全后的继续执行：
#   - 会话级配置/校验与主驱动逐字节一致（同一 bench、同一 verify-one）；
#   - 差别仅是：某场失败时记录 FAILED 原因并继续后续场，不删改任何产物；
#   - 每场墙钟上限 SESSION_CAP_S（默认 7200s），超时组杀进程组、标 HUNG，
#     防止真实挂死（G3 §8 P0 双 die 自旋）导致 bench 永久卡死；
#   - 用法: sel01_continue.sh <run_root> <session_spec>...
#     session_spec 形如 03:v2 04:v2 05:v1 ...（编号:臂）
# 环境要求与主驱动相同（MEMFABRIC_MM_AR_E2E_* / SEL01_APPROVED_ENV=YES）。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
BENCH="$ROOT/tests/e2e/_310p/bench_memfabric_mm_ar_serve.sh"
ANALYZER="$ROOT/tests/e2e/_310p/sel01_analyze.py"
OUT="$1"; shift
SESSION_CAP_S="${SEL01_SESSION_CAP_S:-7200}"

case "$OUT" in /*) ;; *) echo "output directory must be absolute" >&2; exit 2 ;; esac
[ -d "$OUT" ] || { echo "run root missing" >&2; exit 2; }
[ -f "$OUT/manifest.json" ] || { echo "manifest missing (not a SEL-01 run root)" >&2; exit 2; }
[ "$SEL01_APPROVED_ENV" = YES ] || { echo "set SEL01_APPROVED_ENV=YES" >&2; exit 2; }
[ -z "$(printenv MF310P_V2_NOWARMUP 2>/dev/null || :)" ] || { echo "do not bypass safety warmup" >&2; exit 2; }
[ -z "$(printenv VLLM_ASCEND_310P_MEMFABRIC_MM_AR_TRACE 2>/dev/null || :)" ] || { echo "trace must be off" >&2; exit 2; }

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
cd "$ROOT"

for spec in "$@"; do
    sid="${spec%%:*}"
    arm="${spec##*:}"
    case "$arm" in stock|v1|v2) ;; *) echo "bad arm in $spec" >&2; exit 2 ;; esac
    session="$OUT/$sid-$arm"
    [ ! -e "$session" ] || { echo "SKIP $sid-$arm (exists)"; continue; }
    mkdir "$session"
    export MEMFABRIC_MM_AR_E2E_V2=0
    case "$arm" in
      stock) combo="stock eager" ;;
      v1) combo="fused eager" ;;
      v2) combo="fused eager"; export MEMFABRIC_MM_AR_E2E_V2=1 ;;
    esac
    echo "SEL-01-continue $sid arm=$arm input4096 output2048 tp2 batch-cap10 cap=${SESSION_CAP_S}s"
    rm -f "$session/.failed"
    if timeout -k 30 "$SESSION_CAP_S" bash "$BENCH" "$session" "$combo" >"$session/session.log" 2>&1; then
        bench_rc=0
    else
        bench_rc=$?
    fi
    if [ "$bench_rc" -eq 124 ] || [ "$bench_rc" -eq 137 ]; then
        # 墙钟超时：组杀残留 vllm（setsid 会话），保全现场后继续
        pkill -9 -f 'vllm serve.*Qwen3.6-35B-A3B-w8a8' 2>/dev/null || true
        sleep 30
        echo "HUNG wall-clock cap ${SESSION_CAP_S}s" > "$session/.failed"
        echo "FAIL $sid $arm: HUNG (cap)"
        continue
    fi
    if [ "$bench_rc" -ne 0 ]; then
        echo "BENCH_RC=$bench_rc" > "$session/.failed"
        echo "FAIL $sid $arm: bench rc=$bench_rc (logs preserved)"
        pkill -9 -f 'vllm serve.*Qwen3.6-35B-A3B-w8a8' 2>/dev/null || true
        sleep 30
        continue
    fi
    if ! python3 "$ANALYZER" verify-one --session "$session" --arm "$arm" --output "$session/verified.json"; then
        echo "VERIFY_FAILED (not 50/50, wrong lens, or route mismatch)" > "$session/.failed"
        echo "FAIL $sid $arm: verify-one (logs preserved)"
        continue
    fi
    echo "OK $sid $arm"
done
echo "SEL-01-continue schedule complete"
