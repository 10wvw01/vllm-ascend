#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build one SDK-independent TorchNPU Storage bridge for the V2/V3 Tensor ops."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cann',type=Path,required=True)
    args=parser.parse_args()
    import torch
    import torch_npu
    from torch.utils.cpp_extension import load
    root=Path(__file__).resolve().parents[3]
    source=root/'csrc/_310P/memfabric_mm_ar_v3/src/torch_adapter/pool_tensor.cpp'
    home=Path(torch_npu.__file__).resolve().parent
    include=args.cann.resolve(strict=True)/'aarch64-linux/include'
    if not (include/'acl/acl_base_rt.h').is_file():
        raise ValueError('CANN ACL headers missing')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    module=load(name='mf310p_tensor_bridge',sources=[str(source)],
                extra_include_paths=[str(home/'include'),str(home/'include/third_party/hccl/inc'),str(include)],
                extra_ldflags=[f'-L{home / "lib"}','-ltorch_npu',f'-Wl,-rpath,{home / "lib"}'],
                build_directory=str(out),verbose=True)
    artifact=Path(module.__file__).resolve()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=dict(schema='MF310P_TENSOR_BRIDGE_V1',source=str(source),source_sha256=sha(source),
                  path=str(artifact),sha256=sha(artifact),torch=str(torch.__version__),
                  torch_npu=str(torch_npu.__version__),compile='PASS',device_validation='NOT_RUN',
                  memfabric_dependency=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))


if __name__=='__main__':
    main()
