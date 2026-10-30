#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dual-rank smoke for the new public Tensor entry (this round's bootstrap driver).

Runs one candidate TP2 session per runtime binding through the real L1 worker
under the shared-card stage guard. The worker itself performs the public-entry
gates: bit-exact outputs, retained-output submission, immediate same-stream
downstream ADD, storage alias release/reuse, protocol audit and real route
capture. Results become the honest dual_rank_smoke admission evidence for the
frozen campaign of this round.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('/home/w00804146/vllm-ascend')
sys.path.insert(0, str(ROOT / 'csrc/_310P/tools'))
sys.path.insert(0, str(ROOT / 'csrc/_310P/memfabric_mm_ar_v3/tools'))

from L1_test.core import make_plan, sha  # noqa: E402
from L1_test.cli import freeze_data, clean_environment  # noqa: E402
from L1_test.worker import save  # noqa: E402
from stage_guard import run_stage  # noqa: E402

RUN = ROOT / 'csrc/_310P/tools/L1_test/experiments/l1-fixed-20261031-001'
BUILD = ROOT / 'build/310p_l1/l1-fixed-20261031-001'
SDK = Path('/home/w00804146/310P/a3ut-20261029/memfabric-sdk')
CANN = Path('/usr/local/Ascend/cann-9.1.0')
CANN_LD = Path('/tmp/opencode/cann-ld-library-path.txt').read_text().strip()
HEAD = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
BRIDGE = BUILD / 'tensor_bridge/mf310p_tensor_bridge.so'
V3DIR = BUILD / 'v3-build'
V2DIR = BUILD / 'v2-selected'
SDK_LIB64 = SDK / 'aarch64-linux/lib64'
ORCH_JSON = SDK / 'aarch64-linux/hybm/aicpu_kernel/libmf_sdma_orch_v28.json'
COMMON_ENTRY = ROOT / 'csrc/_310P/common/torch_tensor_entry.py'
BINDINGS = ('STOCK', 'V2-7', 'V2-15', 'V3-5', 'V3-6', 'V3-7')
PUBLIC_SOURCES = {
    'STOCK': [],
    'V2-7': [COMMON_ENTRY, ROOT / 'csrc/_310P/memfabric_mm_ar_v2/src/torch_adapter/tensor_entry.py'],
    'V2-15': [COMMON_ENTRY, ROOT / 'csrc/_310P/memfabric_mm_ar_v2/src/torch_adapter/tensor_entry.py'],
    'V3-5': [COMMON_ENTRY, ROOT / 'csrc/_310P/memfabric_mm_ar_v3/src/torch_adapter/tensor_entry.py'],
    'V3-6': [COMMON_ENTRY, ROOT / 'csrc/_310P/memfabric_mm_ar_v3/src/torch_adapter/tensor_entry.py'],
    'V3-7': [COMMON_ENTRY, ROOT / 'csrc/_310P/memfabric_mm_ar_v3/src/torch_adapter/tensor_entry.py'],
}
A3_MODULES = {'memfabric_mm_ar_v3': ROOT / 'vllm_ascend/_310p/ops/memfabric_mm_ar_v3.py',
              'memfabric_mm_ar_a3': ROOT / 'vllm_ascend/_310p/ops/memfabric_mm_ar_a3.py'}
ROUTE = {'V2-7': 'V2TrialKernel', 'V2-15': 'E4Group2048Kernel',
         'V3-5': 'A3Kernel', 'V3-6': 'A3SeatKernel', 'V3-7': 'A3SeatKernel'}
MF_LIBS = [SDK_LIB64 / 'libmf_smem.so', SDK_LIB64 / 'libmf_hybm_core.so', ORCH_JSON]
EVIDENCE = [RUN / 'admission/g0.json', RUN / 'admission/cpu-l1-test.log',
            RUN / 'admission/cpu-v3-v2.log', RUN / 'admission/v2-build-manifest.json',
            RUN / 'admission/v3-build-manifest.json',
            RUN / 'admission/tensor-bridge-manifest.json']
SMOKES = [('STOCK', [], [2048]), ('V2-7', ['V2-7'], [2048]), ('V2-15', ['V2-15'], [4096]),
          ('V3-5', ['V3-5'], [2048]), ('V3-6', ['V3-6'], [2048]), ('V3-7', ['V3-7'], [2048])]
ORIGINAL_ENV = os.environ.copy()


def private_ld(*dirs):
    return ':'.join([*dirs, str(SDK_LIB64), str(CANN / 'aarch64-linux/lib64'), CANN_LD])


def spec(binding):
    base = dict(version=binding, source_head=HEAD, environment={})
    if binding == 'STOCK':
        base['files'] = ['/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/lib/libtorch_npu.so']
        return base
    if binding.startswith('V2-'):
        m = 2048 if binding == 'V2-7' else 4096
        base.update(api='v2_selected_v1', library=str(V2DIR / 'vllm_ascend_C.so'),
                    selected_device_library=str(V2DIR / 'libmf310p_v2_selected_device.so'),
                    supported_M=[m], route_pattern=ROUTE[binding],
                    files=[V2DIR / 'vllm_ascend_C.so', V2DIR / 'libmf310p_v2_trial_device.so',
                           V2DIR / 'libmf310p_v2_selected_device.so', V2DIR / 'libmf310p_device.so',
                           *MF_LIBS, BRIDGE, *PUBLIC_SOURCES[binding]],
                    environment={'LD_LIBRARY_PATH': private_ld(str(V2DIR)),
                                 'MF_SDMA_ORCH_JSON': str(ORCH_JSON),
                                 'MEMFABRIC_HYBRID_HOME_PATH': str(SDK),
                                 'MEMFABRIC_HYBRID_EXTEND_LIB_PATH': str(SDK_LIB64)},
                    tensor_adapter=str(BRIDGE))
        return base
    base.update(api='python_session_v1', library=str(V3DIR / 'libmf310p_v3_host.so'),
                supported_M=[2048, 4096], route_pattern=ROUTE[binding],
                python_modules={k: str(v) for k, v in A3_MODULES.items()},
                files=[V3DIR / 'libmf310p_v3_host.so', V3DIR / 'libmf310p_v3_device.so',
                       *A3_MODULES.values(), *MF_LIBS, BRIDGE, *PUBLIC_SOURCES[binding]],
                environment={'LD_LIBRARY_PATH': private_ld(str(V3DIR)),
                             'MF_SDMA_ORCH_JSON': str(ORCH_JSON),
                             'MEMFABRIC_HYBRID_HOME_PATH': str(SDK),
                             'MEMFABRIC_HYBRID_EXTEND_LIB_PATH': str(SDK_LIB64)},
                tensor_adapter=str(BRIDGE))
    return base


def admission():
    return dict(g0_readonly='PASS', cpu_contract='PASS', private_compile='PASS',
                binary_identity='PASS', dual_rank_smoke='IN_PROGRESS_THIS_RUN',
                evidence_files=[str(p) for p in EVIDENCE])


def campaign_template(smoke_status='PASS', smoke_evidence=None):
    def adm():
        base = admission()
        base['dual_rank_smoke'] = smoke_status
        if smoke_evidence:
            base['evidence_files'] = base['evidence_files'] + [str(p) for p in smoke_evidence]
        return base
    return dict(card=38144, devices='4,5', cann=str(CANN),
                runtimes={b: spec(b) for b in BINDINGS},
                admission_by_runtime={b: adm() for b in BINDINGS})


def restore_env():
    os.environ.clear()
    os.environ.update(ORIGINAL_ENV)


def run_smoke(binding, versions, shapes):
    out = RUN / 'smoke' / binding
    out.mkdir(parents=True, exist_ok=False)
    plan = make_plan(versions, samples=2, warmup=2, rounds=1, mode='shared',
                     scenarios=['ready'], shapes=shapes, collection='off',
                     batch_size=2, working_set=4)
    runtimes = {'STOCK': spec('STOCK')}
    if binding != 'STOCK':
        runtimes[binding] = spec(binding)
    manifest = freeze_data(dict(card=38144, devices='4,5', cann=str(CANN),
                                admission=admission(), runtimes=runtimes))
    save(out / 'manifest.json', manifest)
    job = next(j for j in plan['jobs'] if j['version'] == binding)
    save(out / 'run.json', dict(plan=plan, manifest=manifest,
                                manifest_sha256=sha(out / 'manifest.json'),
                                store_url='tcp://127.0.0.1:8594', job_timeout=900))
    job_dir = out / job['id']
    job_dir.mkdir()
    env = clean_environment(manifest['runtimes'][binding])
    restore_env()
    os.environ.clear()
    os.environ.update(env)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nproc-per-node=2', '--module', 'L1_test.worker',
               '--run', str(out / 'run.json'), '--job', job['id']]
    try:
        rc = run_stage(38144, job_dir / 'lease_guard.jsonl', 2, command, timeout=900,
                       deploy_flag_dir=job_dir, allow_external_target=True)
    finally:
        restore_env()
    print(json.dumps(dict(binding=binding, job=job['id'], rc=rc)), flush=True)
    return rc


def main():
    only = sys.argv[1:] or [b for b, _, _ in SMOKES]
    failures = []
    for binding, versions, shapes in SMOKES:
        if binding not in only:
            continue
        rc = run_smoke(binding, versions, shapes)
        if rc:
            failures.append(binding)
            break
    print(json.dumps(dict(smoke_failures=failures)), flush=True)


if __name__ == '__main__':
    main()
