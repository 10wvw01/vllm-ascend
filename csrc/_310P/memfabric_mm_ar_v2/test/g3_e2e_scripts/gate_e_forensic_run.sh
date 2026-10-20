#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# G1-A：不改设备 Kernel 的 Gate E 取证包装器；一轮只跑一个模式。
# 用法：bash gate_e_forensic_run.sh fused|stock [实验编号]
set -Eeuo pipefail

MODE="${1:-}"
case "$MODE" in fused|stock|stock_dflash) ;; *) echo "用法: bash $0 fused|stock|stock_dflash [实验编号]" >&2; exit 2 ;; esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(git -C "$V2_ROOT" rev-parse --show-toplevel)"
SESSION="${2:-$(date -u +%Y%m%dT%H%M%SZ)_${MODE}}"
case "$SESSION" in *[!A-Za-z0-9_.-]*|"") echo "实验编号仅允许字母数字._-" >&2; exit 2 ;; esac
ROOT="${V2_ROOT}/tmp/opencode/gate_e_forensics"
OUT_DIR="${ROOT}/${SESSION}"
if [[ -e "$OUT_DIR" ]]; then
  echo "拒绝覆盖已有实验目录: $OUT_DIR" >&2
  exit 3
fi
mkdir -p "${OUT_DIR}/samples" "${OUT_DIR}/evidence"
export OUT_DIR

# 不导出所有环境变量（可能包含令牌），只记录必要的可复核版本、设备及路径。
{
  date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
  echo "mode=$MODE"
  echo "vllm_tree=$REPO_ROOT"
  git -C "$REPO_ROOT" rev-parse HEAD
  git -C "$REPO_ROOT" status --porcelain=v1
  echo "devices=${GATE_E_DEVICES:-0,1}"
  echo "model=${GATE_E_MODEL:-/home/models/Qwen/Qwen3.6-35B-A3B-w8a8}"
  echo "duration_s=${GATE_E_DURATION_S:-620}"
  echo "mf_host_diag=${MF_SDMA_ORCH_DIAG:-unset}"
  echo "mf_epoch_budget=${MF_SDMA_ORCH_EPOCH_BUDGET_S:-default}"
  # 实机布局：/usr/local/memfabric_hybrid/latest -> 1.2.0，库在 aarch64-linux/lib64；
  # vLLM 进程实际经 site-packages ctypes 加载包内 lib/libmf_hybm_core.so，两处都取哈希。
  MF_HOME=/usr/local/memfabric_hybrid/latest
  MF_SITE=$(python3 -c "import memfabric_hybrid,os;print(os.path.dirname(memfabric_hybrid.__file__))" 2>/dev/null || true)
  for file in \
      "${MF_HOME}/version.info" \
      "${MF_HOME}/aarch64-linux/lib64/libmf_hybm_core.so" \
      "${MF_HOME}/aarch64-linux/lib64/orch_deploy_helper" \
      "${MF_HOME}/aarch64-linux/lib64/libmf_smem.so" \
      "${MF_HOME}/aarch64-linux/hybm/aicpu_kernel/libmf_sdma_orch.so" \
      "${MF_SITE}/lib/libmf_hybm_core.so"; do
    if [[ -n "$file" && -f "$file" ]]; then
      stat -c '%n %s bytes %y' "$file"
      sha256sum "$file"
    fi
  done
  if [[ -n "${MF_SDMA_ORCH_JSON:-}" && -f "$MF_SDMA_ORCH_JSON" ]]; then
    sha256sum "$MF_SDMA_ORCH_JSON"
  fi
  command -v npu-smi || true
  command -v vllm || true
} > "${OUT_DIR}/evidence/provenance.txt" 2>&1

# 取样仅调用查询命令，不触发 device synchronize。
STOP_FILE="${OUT_DIR}/samples/.stop"
sample_loop() {
  while [[ ! -e "$STOP_FILE" ]]; do
    {
      date -u '+----- %Y-%m-%dT%H:%M:%SZ -----'
      npu-smi info 2>&1 || true
    } >> "${OUT_DIR}/samples/npu_smi.log" 2>&1
    {
      date -u '+----- %Y-%m-%dT%H:%M:%SZ -----'
      ps -eo pid,ppid,stat,etime,args | grep -E 'vllm|EngineCore|Worker_TP|gate_e' | grep -v grep || true
    } >> "${OUT_DIR}/samples/processes.log" 2>&1
    sleep "${GATE_E_SAMPLE_PERIOD_S:-15}"
  done
}
sample_loop &
SAMPLER_PID=$!
cleanup() {
  touch "$STOP_FILE"
  # 无需中止服务：服务生命周期由原 Gate E 脚本管理。
  if kill -0 "$SAMPLER_PID" 2>/dev/null; then
    kill "$SAMPLER_PID" 2>/dev/null || true
  fi
  wait "$SAMPLER_PID" 2>/dev/null || true
}
trap cleanup EXIT

if [[ "$MODE" = fused ]]; then
  RUNNER="${SCRIPT_DIR}/gate_e_eager.sh"
elif [[ "$MODE" = stock_dflash ]]; then
  RUNNER="${SCRIPT_DIR}/gate_e_stock_dflash.sh"
else
  RUNNER="${SCRIPT_DIR}/gate_e_eager_stock.sh"
fi
echo "[$(date -u '+%FT%TZ')] Gate E $MODE 启动，证据目录 $OUT_DIR" | tee "${OUT_DIR}/runner.log"
set +e
bash "$RUNNER" "$OUT_DIR" >> "${OUT_DIR}/runner.log" 2>&1
RC=$?
set -e
echo "$RC" > "${OUT_DIR}/runner_exit_code.txt"

# 保持历史证据：本脚本不强杀、清空或重建其他运行中的设备进程。
python3 "${SCRIPT_DIR}/gate_e_forensic_summary.py" "$OUT_DIR" > "${OUT_DIR}/evidence/summary.txt" 2>&1 || true

# 独立验收的退出码不能被原 Runner 的 0 覆盖。
# F01 只完成 29/多轮应判 FAIL；不依赖 vllm bench 是否设置非零退出码。
PROMPTS_PER_REP=25
if [[ "$MODE" = stock_dflash ]]; then PROMPTS_PER_REP=50; fi
VERDICT_RC=0
python3 "${SCRIPT_DIR}/gate_e_forensic_verdict.py" "$OUT_DIR" \
    --prompts-per-rep "$PROMPTS_PER_REP" \
    --json-output "$OUT_DIR/evidence/verdict.json" \
    > "$OUT_DIR/evidence/verdict.stdout.json" 2> "$OUT_DIR/evidence/verdict.stderr" || VERDICT_RC=$?
if [[ "$VERDICT_RC" -gt 2 ]]; then
  echo "判定器执行失败: rc=$VERDICT_RC" >> "$OUT_DIR/evidence/verdict.stderr"
  VERDICT_RC=2
fi
{
  echo "原始 runner 退出码: $RC"
  echo "若负载发生 hang/timeout，请在恢复设备前保全对应时段的 plog、slog、aicore 错误日志。"
  echo "路径清单（需要根据服务器实际部署核实）："
  for logdir in /root/ascend/log /var/log/npu/slog "$HOME/ascend/log"; do
    if [[ -d "$logdir" ]]; then
      find "$logdir" -type f -mmin -90 -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' 2>/dev/null | tail -150
    fi
  done
} > "${OUT_DIR}/evidence/device_log_inventory.txt"
FINAL_RC="$VERDICT_RC"
if [[ "$RC" -ne 0 ]]; then FINAL_RC=1; fi
echo "[$(date -u '+%FT%TZ')] runner原始码=$RC，离线判定码=$VERDICT_RC，最终码=$FINAL_RC，报告见 $OUT_DIR/evidence/verdict.json" | tee -a "${OUT_DIR}/runner.log"
exit "$FINAL_RC"
