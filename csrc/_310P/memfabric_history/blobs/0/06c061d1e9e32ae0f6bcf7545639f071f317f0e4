#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# G1-D 分段执行入口。不安装、不部署、不切换 Git 分支，不自动覆盖原库。
set -Eeuo pipefail
CMD="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_TREE="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
expect_branch=diag/310p3-g1d-minimal-forensics-20261021
case "$CMD" in
  preflight)
    exec bash "$SCRIPT_DIR/g1d_preflight.sh" record
    ;;
  verify-baseline)
    : "${G1D_ORIGINAL_SHA_MANIFEST:?请提供此前保全的设备安装库 SHA256SUMS 绝对路径}"
    [[ -f "$G1D_ORIGINAL_SHA_MANIFEST" ]] || exit 3
    exec sha256sum -c "$G1D_ORIGINAL_SHA_MANIFEST"
    ;;
  c1|async|gate-e)
    : "${G1D_MF_TREE:?必须指定 MemFabric G1-D 独立 worktree}"
    [[ "$(git -C "$V2_TREE" branch --show-current)" == "$expect_branch" ]] || {
      echo "错误 V2 分支: $(git -C "$V2_TREE" branch --show-current)" >&2; exit 3;
    }
    [[ "$(git -C "$G1D_MF_TREE" branch --show-current)" == "$expect_branch" ]] || {
      echo "错误 MemFabric 分支" >&2; exit 3;
    }
    [[ -z "$(git -C "$V2_TREE" status --porcelain)" &&
       -z "$(git -C "$G1D_MF_TREE" status --porcelain)" ]] || {
      echo "工作树不洁净，请先归档本地构建/修改（构建产物应忽略）" >&2; exit 3;
    }
    : "${MF_SDMA_ORCH_JSON:?必须显式指定已验证的 v12 launch JSON；不能自动猜测最新版本}"
    [[ "$MF_SDMA_ORCH_JSON" == *libmf_sdma_orch_v12.json &&
       -f "$MF_SDMA_ORCH_JSON" ]] || {
      echo "当前第一轮 G1-D 必须沿用 G1-C v12 设备取证库，不可使用 v9/v11/v13" >&2; exit 3;
    }
    # D0 版本门禁：不能仅检查源码 worktree，必须验证 Worker 将加载的
    # 两份 Host 动态库与 v12 JSON 字节级一致。未部署好直接拒绝执行。
    : "${G1D_HOST_SHA256_EXPECTED:?先完成 Host 双位置部署，设置 G1D_HOST_SHA256_EXPECTED}"
    : "${G1D_V12_JSON_SHA256_EXPECTED:?设置经 SHA256 校验的 v12 launch JSON 摘要}"
    actual_json_sha="$(sha256sum "$MF_SDMA_ORCH_JSON" | awk '{print $1}')"
    [[ "$actual_json_sha" == "$G1D_V12_JSON_SHA256_EXPECTED" ]] || {
        echo "v12 JSON SHA256 不一致，拒绝运行" >&2; exit 3;
    }
    mf_host_global=/usr/local/memfabric_hybrid/latest/aarch64-linux/lib64/libmf_hybm_core.so
    mf_python_site="$(python3 -c 'import memfabric_hybrid,os;print(os.path.dirname(memfabric_hybrid.__file__))')"
    mf_host_site="$mf_python_site/lib/libmf_hybm_core.so"
    for lib in "$mf_host_global" "$mf_host_site"; do
        [[ -f "$lib" ]] || { echo "缺少 Host 运行库 $lib" >&2; exit 3; }
        actual_host_sha="$(sha256sum "$lib" | awk '{print $1}')"
        [[ "$actual_host_sha" == "$G1D_HOST_SHA256_EXPECTED" ]] || {
            echo "Host 运行库 SHA256 不一致: $lib" >&2; exit 3;
        }
    done
    [[ "${ASCEND_RT_VISIBLE_DEVICES:-0,1}" == "0,1" ]] || {
      echo "当前只允许 0,1 两设备" >&2; exit 3;
    }
    if [[ "${MF_SDMA_ORCH_EPOCH_BUDGET_S:-}" == "4294967295" ]]; then
      echo "禁止常驻 Epoch" >&2; exit 3;
    fi
    export ASCEND_RT_VISIBLE_DEVICES=0,1
    export MF_SDMA_ORCH_DIAG=1
    export MF_SDMA_ORCH_G1C_TRACE=1
    export MF_SDMA_ORCH_G1D_TRACE=1
    export MF310P_V2_G1C_TRACE=1
    export MF310P_V2_G1D_TRACE=1
    export V2_WORKTREE="$V2_TREE"
    # 真实 Serving 的 import resolver 需沿用 G1-C 的 PathFinder 优先级方案。
    # 不提供可验证的注入目录就拒绝 Gate E，避免实际加载其他 vllm-ascend。
    if [[ "$CMD" == gate-e ]]; then
        : "${G1D_PYINJ:?必须指定已核验的 sitecustomize 注入目录}"
        [[ -f "$G1D_PYINJ/sitecustomize.py" ]] || {
            echo "缺少 $G1D_PYINJ/sitecustomize.py" >&2; exit 3;
        }
        export PYTHONPATH="$G1D_PYINJ:$V2_TREE:${PYTHONPATH:-}"
    else
        export PYTHONPATH="$V2_TREE:${PYTHONPATH:-}"
    fi
    export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR=1
    export VLLM_ASCEND_310P_MEMFABRIC_STORE_URL=tcp://127.0.0.1:8581
    export VLLM_ASCEND_310P_MEMFABRIC_LOCAL_BYTES=$((96 * 1024 * 1024))
    export VLLM_ASCEND_310P_MEMFABRIC_MM_AR_BATCH_BASEM_COUNT=1
    export VLLM_ASCEND_310P_MEMFABRIC_MM_AR_WARMUP_FALLBACK=1
    export VLLM_ASCEND_310P_MEMFABRIC_MM_AR_V2_SERVE=1
    export VLLM_ASCEND_310P_MEMFABRIC_V2_CHUNK_ROWS=1024
    export VLLM_ASCEND_310P_MEMFABRIC_V2_OUTPUT_SLOTS=4
    if [[ "$CMD" == c1 ]]; then
       cd "$V2_TREE"
       exec torchrun --nproc-per-node=2 --master-port=29531 \
           csrc/_310P/memfabric_mm_ar_v2/test/e2e/tp2_v2_mixed_route_correctness.py \
           --rows 2047 2048 2049 4095 4096 4097 --reps 3
    elif [[ "$CMD" == async ]]; then
       shift
       cd "$V2_TREE"
       exec torchrun --nproc-per-node=2 --master-port=29531 \
           csrc/_310P/memfabric_mm_ar_v2/test/e2e/g1d_tp2_async_window.py "$@"
    else
       session="${2:?需要唯一的 Gate E 会话编号}"
       export GATE_E_DURATION_S="${GATE_E_DURATION_S:-620}"
       cd "$SCRIPT_DIR"
       exec bash gate_e_forensic_run.sh fused "$session"
    fi
    ;;
  *)
    echo "用法: $0 preflight|verify-baseline|c1|async [参数]|gate-e <新会话编号>" >&2
    exit 2
    ;;
esac
