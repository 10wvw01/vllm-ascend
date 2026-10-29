#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# G1-D0：只读版本/环境锁定。拒绝自动安装、切分支、覆盖设备库。
set -Eeuo pipefail
umask 077
[[ ${1:-} == "record" ]] || { echo "用法: bash $0 record; 设置 G1D_MF_TREE/G1D_OUT_DIR" >&2; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_TREE="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
: "${G1D_MF_TREE:?必须提供 G1D_MF_TREE（MemFabric G1-D worktree 的绝对路径）}"
: "${G1D_OUT_DIR:?必须提供 G1D_OUT_DIR（全新、非覆盖证据目录）}"
[[ "$G1D_MF_TREE" = /* && "$G1D_OUT_DIR" = /* ]] || {
    echo "MF 工作树和输出目录必须是绝对路径" >&2; exit 2;
}
[[ ! -e "$G1D_OUT_DIR" ]] || { echo "拒绝覆盖证据目录: $G1D_OUT_DIR" >&2; exit 3; }
mkdir -p "$G1D_OUT_DIR"
report="$G1D_OUT_DIR/provenance.txt"
{
    date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
    for entry in "V2:$V2_TREE" "MF:$G1D_MF_TREE"; do
        tag="${entry%%:*}"; dir="${entry#*:}"
        echo "=== $tag ==="
        git -C "$dir" rev-parse --show-toplevel
        git -C "$dir" rev-parse HEAD
        git -C "$dir" branch --show-current
        git -C "$dir" status --porcelain=v1
        git -C "$dir" remote -v | sed -E 's#(https?://)[^/@]+:[^/@]+@#\1[REDACTED]@#g' || true
    done
    echo "=== selected runtime facts ==="
    echo "npu_device_map=${ASCEND_RT_VISIBLE_DEVICES:-unset}"
    echo "mf_json=${MF_SDMA_ORCH_JSON:-unset}"
    echo "mf_epoch_budget=${MF_SDMA_ORCH_EPOCH_BUDGET_S:-default15}"
    echo "mf_trace=${MF_SDMA_ORCH_G1D_TRACE:-0}"
    echo "v2_trace=${MF310P_V2_G1D_TRACE:-0}"
    command -v npu-smi || true
    command -v bisheng || true
    command -v python3 || true
    (npu-smi info 2>&1 || true) | head -60
    echo "=== installed library sha256 ==="
    mhome=/usr/local/memfabric_hybrid/latest
    mfsite="$(python3 -c "import memfabric_hybrid,os;print(os.path.dirname(memfabric_hybrid.__file__))" 2>/dev/null || true)"
    for f in "$mhome/aarch64-linux/lib64/libmf_hybm_core.so" \
        "$mfsite/lib/libmf_hybm_core.so" \
        "$mhome/aarch64-linux/hybm/aicpu_kernel/libmf_sdma_orch.so" \
        "${MF_SDMA_ORCH_JSON:-}"; do
        [[ -n "$f" && -f "$f" ]] && sha256sum "$f"
    done
    echo "=== version info ==="
    if [[ -f "$mhome/version.info" ]]; then cat "$mhome/version.info"; fi
} > "$report" 2>&1
sha256sum "$report" > "$report.sha256"
echo "$report"
