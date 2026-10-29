#!/usr/bin/env bash
# E4续跑：G0复核（16384独占/健康/SHA/工作树洁净）通过后启动campaign attempt2。
# Confirm保持未授权；任何复核失败立即退出，不重试、不换卡。
set -Eeuo pipefail
E4_WT=/home/w00804146/310P/e4wt
B=/tmp/opencode/a1-batch
PERF=$E4_WT/csrc/_310P/memfabric_mm_ar_v2/test/perf
OUT=/tmp/opencode/e4-campaign-20261028b
BUILD=/tmp/opencode/e4-build2
G0DIR=/tmp/opencode/e4_attempt2_g0
mkdir -m 700 -p "$G0DIR"

echo "[$(date -Is)] relaunch: G0 re-verification start"

# 1) 卡16384当前快照与二Die健康
npu-smi info > "$G0DIR/npu_relaunch.txt" 2>&1
npu-smi info -t health -i 16384 > "$G0DIR/health_relaunch.txt" 2>&1
card_lines=$(grep -m2 "^| 16384" "$G0DIR/npu_relaunch.txt" || true)
if echo "$card_lines" | grep -q "Alarm"; then
  echo "[$(date -Is)] STOP: 16384 Alarm present"; exit 1; fi
if ! grep -q "Health                         : OK" "$G0DIR/health_relaunch.txt"; then
  echo "[$(date -Is)] STOP: 16384 health not OK"; exit 1; fi
if grep -qE "^\| 16384    [01] +\|" "$G0DIR/npu_relaunch.txt"; then
  echo "[$(date -Is)] STOP: 16384 still has processes"; exit 1; fi
hp2=$(grep -m1 "^| 16384" "$G0DIR/npu_relaunch.txt" | awk '{print $(NF-3)}')
hp3=$(grep -m2 "^| 16384" "$G0DIR/npu_relaunch.txt" | tail -1 | awk '{print $(NF-3)}')
if [ "${hp2:-9999}" -gt 100 ] || [ "${hp3:-9999}" -gt 100 ]; then
  echo "[$(date -Is)] STOP: hugepages not reclaimed hp2=$hp2 hp3=$hp3"; exit 1; fi
echo "[$(date -Is)] card 16384 free: hp2=$hp2 hp3=$hp3 health OK"

# 2) 四冻结SHA
check_sha() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ] || { echo "[$(date -Is)] STOP: SHA mismatch $1"; exit 1; }; }
check_sha "$B/priv-libs/libmf_hybm_core.so" 1dc46d7dcbe583a5dbb78b4c65a0b5e22f9f352976c53ed31b761fa744f91a30
check_sha "$B/site/vllm_ascend/libmf310p_v2_trial_device.so" 2a19390c83c3fe6b0bdb8946d88990cca66b331f765ca445dbda2e787c2d2a19
check_sha "$B/site/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so" 8eb65de98c3d8664ff10b421c1d7ad31318dc720c3f5679cd4d34e187203bea1
check_sha /usr/local/memfabric_hybrid/1.2.0/aarch64-linux/hybm/aicpu_kernel/libmf_sdma_orch_v11.json 25e16b0af2f16989231ff0443698261257bd8142834783a0c1e4ad57cd9ebc78
echo "[$(date -Is)] frozen binary SHAs verified"

# 3) 工作树洁净 + HEAD + 私有构建manifest身份
[ -z "$(git -C "$E4_WT" status --porcelain --untracked-files=no)" ] || {
  echo "[$(date -Is)] STOP: worktree dirty"; exit 1; }
head=$(git -C "$E4_WT" rev-parse HEAD)
[ "$head" = "5b727e9f9153e8292c6022a14b63e974dfb0db03" ] || {
  echo "[$(date -Is)] STOP: unexpected HEAD $head"; exit 1; }
python3 - "$BUILD/manifest.json" "$head" <<'PY' || exit 1
import json,sys,pathlib,hashlib
m=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert m["work_head"]==sys.argv[2]
for k,p in (("candidate_sha256","candidate_so"),("private_host_sha256","private_host_so")):
    assert hashlib.sha256(pathlib.Path(m[p]).read_bytes()).hexdigest()==m[k]
PY
echo "[$(date -Is)] worktree HEAD $head clean; build manifest verified"

# 4) MemFabric store端口可用性（占用则换备选）
PORT=18894
python3 - "$PORT" <<'PY' || PORT=18895
import socket,sys
s=socket.socket();s.settimeout(0.25)
sys.exit(0 if s.connect_ex(("127.0.0.1",int(sys.argv[1])))!=0 else 1)
PY
echo "[$(date -Is)] store port $PORT"

# 5) 启动campaign attempt2（E4_CONFIRM_APPROVED保持未设置）
export PATH=/usr/local/python3.12.13/bin:$PATH
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
export ASCEND_RT_VISIBLE_DEVICES=2,3
export MF_SDMA_ORCH_JSON=/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/hybm/aicpu_kernel/libmf_sdma_orch_v11.json
export V2_HOST_LIB=$B/site/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so
export U1_DEVICE_SO_PATH=$B/site/vllm_ascend/libmf310p_v2_trial_device.so
export U1_DEVICE_SHA256=2a19390c83c3fe6b0bdb8946d88990cca66b331f765ca445dbda2e787c2d2a19
export U1_PLUGIN_SHA256=8eb65de98c3d8664ff10b421c1d7ad31318dc720c3f5679cd4d34e187203bea1
export U1_HOST_SHA256=1dc46d7dcbe583a5dbb78b4c65a0b5e22f9f352976c53ed31b761fa744f91a30
export U1_KVER_SHA256=25e16b0af2f16989231ff0443698261257bd8142834783a0c1e4ad57cd9ebc78
export A1_FIVE_ARM_G0_PASSED=YES
export A1_FIVE_ARM_CARD_EXCLUSIVE=YES
export FIVE_ARM_STORE_PORT=$PORT
export LD_LIBRARY_PATH=$B/priv-libs:/usr/local/memfabric_hybrid/latest/aarch64-linux/lib64:/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$B/site:${PYTHONPATH:-}
unset E4_CONFIRM_APPROVED
cd "$E4_WT"
echo "[$(date -Is)] launching e4_campaign.sh OUT=$OUT BUILD=$BUILD PORT=$PORT"
exec bash "$PERF/e4_campaign.sh" "$OUT" "$BUILD"
