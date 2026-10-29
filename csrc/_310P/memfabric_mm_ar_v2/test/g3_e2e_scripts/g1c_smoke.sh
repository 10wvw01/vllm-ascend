#!/usr/bin/env bash
# G1-C 第一轮实机观测门禁：先跑短时原工况，再验证双 Rank 8 核阶段与 D2H。
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_ROOT="$(cd "$HERE/../.." && pwd)"
SESSION="${1:-G1C_SMOKE_$(date -u +%Y%m%dT%H%M%SZ)}"
export MF310P_G1C_TRACE=1
export GATE_E_DURATION_S="${G1C_SMOKE_DURATION_S:-180}"
export V2_WORKTREE="$(git -C "$V2_ROOT" rev-parse --show-toplevel)"
RC=0
bash "$HERE/gate_e_forensic_run.sh" fused "$SESSION" || RC=$?
OUT="$V2_ROOT/tmp/opencode/gate_e_forensics/$SESSION"
if [[ "$RC" -ne 0 ]]; then
  echo "G1-C 短测异常 rc=$RC；先保存现场，不执行观测门禁或长稳" >&2
  exit "$RC"
fi
python3 "$HERE/g1c_observability_gate.py" "$OUT/g1c" \
  > "$OUT/evidence/g1c_observability.json"
echo "G1-C 双 Rank 观测门禁通过。结果：$OUT/evidence/g1c_observability.json"
echo "注意：未取得本轮 G1-C 静态/CANN/TP2 功能门禁通过证明前，仍禁止开展长稳。"
