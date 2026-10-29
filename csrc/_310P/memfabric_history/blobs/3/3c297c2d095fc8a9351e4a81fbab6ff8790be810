# 310P MemFabric bring-up artifacts

Device object compiled from `csrc/memfabric_o_proj/external/memfabric310p_device.asc`
with the same customized 310P toolchain used by the MemFabric example 08
(verified on the real 310P3 server, CANN 9.1.0):

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export MF_ROOT=/opt/memfabric-wgm-dev-310p   # wgm-dev-310p install prefix

bisheng --npu-arch=dav-2002 -O2 -std=c++17 -w -fPIC \
  -x asc csrc/memfabric_o_proj/external/memfabric310p_device.asc -x none \
  -c -o build_310p_artifacts/memfabric310p_device.o \
  -I$ASCEND_HOME_PATH/include -I$MF_ROOT/include
```

Notes (real-machine facts, 2026-09-18):
- `-fPIC` is mandatory: the object links into the `vllm_ascend_C` shared library.
- The link additionally needs CANN's static `libascendc_runtime.a`
  (AscendC launch stubs); handled by `cmake/memfabric_310p.cmake`.
- The dav-2002 AICore scalar environment has no `bfloat16_t`; BF16 reduce uses
  bit-level conversions in the kernel.
