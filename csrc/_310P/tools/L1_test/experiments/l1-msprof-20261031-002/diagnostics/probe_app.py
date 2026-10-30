#!/usr/bin/env python3
"""Minimal discriminator: A3Session pool + direct launches under external msprof.

Ranks via torchrun. Prints device-visible markers so the exported trace can be
checked for whether direct-launched A3 kernels keep device timing after pool
creation, isolating the msprof capture gap seen in the L1 diagnostic round.
"""
import argparse
import importlib.util
import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu

REPO = Path('/home/w00804146/vllm-ascend')
BUILD = REPO / 'build/310p_l1/l1-fixed-20261031-001'
SDK = Path('/home/w00804146/310P/a3ut-20261029/memfabric-sdk')
sys.path.insert(0, str(REPO / 'vllm_ascend/_310p/ops'))
os.environ.setdefault('LD_LIBRARY_PATH', '')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--devices', default='4,5')
    p.add_argument('--store-url', default='tcp://127.0.0.1:8597')
    a = p.parse_args()
    rank = int(os.environ['RANK'])
    selected = int(a.devices.split(',')[rank])
    torch.npu.set_device(selected)
    torch.npu.config.allow_internal_format = True
    torch.npu.set_compile_mode(jit_compile=False)
    dist.init_process_group('gloo', timeout=timedelta(seconds=120))
    stream = torch.npu.current_stream()
    from memfabric_mm_ar_a3 import A3Session

    spec = importlib.util.spec_from_file_location(
        'mf310p_tensor_bridge', str(BUILD / 'tensor_bridge/mf310p_tensor_bridge.so'))
    tensor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tensor)
    session = A3Session(str(BUILD / 'v3-build/libmf310p_v3_host.so'), rank, a.store_url,
                        packet_bytes=4 << 20, tensor_module=tensor, variant='a3-2')
    session.ib_probe()
    src = torch.ones((512, 2048), dtype=torch.float16, device=f'npu:{selected}')
    seed = src.clone()
    out = torch.empty_like(src)
    session.probe_add(src, seed, out)
    session.synchronize_check()
    assert torch.equal(out, src + seed)
    del out, src, seed
    session.initialize()
    print('POOL_CREATED', flush=True)

    torch.manual_seed(4242)
    w = torch.randn((2048, 2048), dtype=torch.float16, device=f'npu:{selected}')
    wnz = torch_npu.npu_format_cast(w, 29)
    x = torch.randn((2048, 2048), dtype=torch.float16, device=f'npu:{selected}')
    ref = F.linear(x, wnz).float().cpu()
    dist.all_reduce(ref)
    ref = ref.to(device=f'npu:{selected}', dtype=torch.float16)
    stream.synchronize()
    print('SETUP_DONE', flush=True)
    for i in range(6):
        o = session.run_pool(x, wnz, compute_rows=512, comm_rows=1024)
        stream.synchronize()
        assert torch.equal(o, ref), f'numeric fail at {i}'
        del o
    print('RUN_POOL_OK', flush=True)
    session.close()
    print('CLOSED', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
