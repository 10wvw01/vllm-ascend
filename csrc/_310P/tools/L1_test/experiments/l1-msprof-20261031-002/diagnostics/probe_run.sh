#!/bin/bash
set -euo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cd /home/w00804146/vllm-ascend
export LD_LIBRARY_PATH=/home/w00804146/vllm-ascend/build/310p_l1/l1-fixed-20261031-001/v3-build:/home/w00804146/310P/a3ut-20261029/memfabric-sdk/aarch64-linux/lib64:/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64:$LD_LIBRARY_PATH
export MF_SDMA_ORCH_JSON=/home/w00804146/310P/a3ut-20261029/memfabric-sdk/aarch64-linux/hybm/aicpu_kernel/libmf_sdma_orch_v28.json
export MEMFABRIC_HYBRID_HOME_PATH=/home/w00804146/310P/a3ut-20261029/memfabric-sdk
export MEMFABRIC_HYBRID_EXTEND_LIB_PATH=/home/w00804146/310P/a3ut-20261029/memfabric-sdk/aarch64-linux/lib64
export PYTHONPATH=/home/w00804146/vllm-ascend/csrc/_310P/tools:$PYTHONPATH
OUT=$1
mkdir -p "$OUT"
# rank shim: exec into msprof per rank (mirrors L1 rank_launcher)
cat > "$OUT/rank_shim.py" <<'PYEOF'
import os, sys
rank = os.environ['RANK']
out = sys.argv[1]
app = ['/usr/local/python3.12.13/bin/python3', '/tmp/opencode/a3-msprof-probe/probe_app.py', '--devices', '4,5']
argv = ['/usr/local/Ascend/cann-9.1.0/tools/profiler/bin/msprof',
        '--output=' + out + '/msprof-rank' + rank,
        '--application=' + ' '.join(app),
        '--task-time=on', '--runtime-api=on', '--ascendcl=on', '--aicpu=on', '--msproftx=on']
os.execv(argv[0], argv)
PYEOF
exec /usr/local/python3.12.13/bin/python3 -m torch.distributed.run --standalone --nproc-per-node=2 "$OUT/rank_shim.py" "$OUT"
