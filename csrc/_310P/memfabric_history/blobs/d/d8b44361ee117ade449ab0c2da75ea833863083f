#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# G1-C 设备侧取证执行入口（实机验证版，2026-10-20）。
# 用法:
#   bash g1c_forensic_run.sh env      # 打印/导出 G1-C 诊断环境(供人工检查)
#   bash g1c_forensic_run.sh c1       # Gate C1: 双 Rank 混合路由最小正确性自检
#   bash g1c_forensic_run.sh gate-e <会话编号>   # Gate C2: 原 Gate E 工况取证
#   bash g1c_forensic_run.sh restore  # 恢复原运行库并核对 SHA256
#
# 前置条件(缺一不可,脚本只做最小检查):
#   - MemFabric G1-C worktree 已按 -DMF_SDMA_ORCH_KVER=12
#     -DMF_SDMA_ORCH_G1C_DEVICE_TRACE=ON 编译,产物路径见 MF_BUILD 变量
#   - vllm-ascend G1-C worktree 已按 -DVLLM_ASCEND_310P_V2_G1C_TRACE=ON
#     构建并 --inplace 放置 vllm_ascend_C + libmf310p_v2_trial_device_g1c.so
#   - kernel 三件套已在 STAGE 目录(so + CUST json + v12 launch json)
#   - Host core.so(G1-C 版)已替换到 /usr/local 与 site-packages 双位置
#     (备份留存),v12 kernel 已部署到设备搜索路径(新文件名,不动 v11)
# 详细步骤与全部哈希见 docs/G1C_DEVICE_FIRST_STOP_REPORT_20261020.md。
set -euo pipefail

G1C_TREE=${G1C_TREE:-/home/w00804146/310P/vllm-ascend-g1c}
MF_G1C=${MF_G1C:-/home/w00804146/310P/memfabric_hybrid-g1c}
STAGE=${G1C_STAGE:-/tmp/opencode/g1c/kernel_v12_stage}
PYINJ=${G1C_PYINJ:-/tmp/opencode/g1c/pyinject}
MF_BUILD=$MF_G1C/build-g1c

CMD=${1:-}
case "$CMD" in
  env)
    cat <<EOF
G1C_TREE=$G1C_TREE
MF_G1C=$MF_G1C
STAGE=$STAGE
export V2_WORKTREE=$G1C_TREE
export MF_SDMA_ORCH_JSON=$STAGE/libmf_sdma_orch_v12.json
export MF_SDMA_ORCH_DEPLOY_HELPER=$MF_G1C/tools/aicpu_kernel_deploy/orch_deploy_helper
export MF_SDMA_ORCH_DIAG=1
export MF_SDMA_ORCH_G1C_TRACE=1
export MF310P_V2_G1C_TRACE=1
export PYTHONPATH="$PYINJ:$G1C_TREE\$PYTHONPATH"
unset MF_SDMA_ORCH_EPOCH_BUDGET_S   # 15s 默认 Epoch
EOF
    ;;
  c1)
    export ASCEND_RT_VISIBLE_DEVICES=0,1
    export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR=1
    export VLLM_ASCEND_310P_MEMFABRIC_STORE_URL=tcp://127.0.0.1:8581
    export VLLM_ASCEND_310P_MEMFABRIC_LOCAL_BYTES=$((96 * 1024 * 1024))
    export VLLM_ASCEND_310P_MEMFABRIC_MM_AR_BATCH_BASEM_COUNT=1
    export VLLM_ASCEND_310P_MEMFABRIC_MM_AR_WARMUP_FALLBACK=1
    export VLLM_ASCEND_310P_MEMFABRIC_MM_AR_V2_SERVE=1
    export VLLM_ASCEND_310P_MEMFABRIC_V2_CHUNK_ROWS=1024
    export VLLM_ASCEND_310P_MEMFABRIC_V2_OUTPUT_SLOTS=4
    eval "$(bash "$0" env | grep '^export\|^unset')"
    cd "$G1C_TREE"
    exec torchrun --nproc-per-node=2 --master-port=29531 \
      csrc/_310P/memfabric_mm_ar_v2/test/e2e/tp2_v2_mixed_route_correctness.py \
      --rows 2047 2048 2049 4095 4096 4097 --reps 3
    ;;
  gate-e)
    SESSION=${2:-C2R1}
    export GATE_E_DURATION_S=${GATE_E_DURATION_S:-620}
    eval "$(bash "$0" env | grep '^export\|^unset')"
    cd "$G1C_TREE/csrc/_310P/memfabric_mm_ar_v2/test/g3_e2e_scripts"
    exec bash gate_e_forensic_run.sh fused "$SESSION"
    ;;
  restore)
    # 逐字节恢复原库;完整脚本与备份哈希见报告 §8。
    echo "按 /tmp/opencode/g1c/restore_g1c.sh 流程执行(备份目录 + SHA256SUMS 校验)"
    exit 3
    ;;
  *)
    echo "用法: $0 env|c1|gate-e <会话编号>|restore" >&2
    exit 2
    ;;
esac
