# SPDX-License-Identifier: Apache-2.0
"""G1-C 单变量审查：剔除追踪语句后，设备执行主体与正式 V2 完全一致。"""
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"


def _body(path: str) -> list[str]:
    text = (SRC / "op_kernel" / path).read_text(encoding="utf-8")
    start = text.index("KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);")
    end = text.index("using namespace AscendC;", start)
    lines = text[start:end].splitlines()
    # 审查范围只在 Kernel 体内，去掉该函数后的 namespace 注释。
    return [line.strip() for line in lines
            if line.strip() and not line.strip().startswith("//")]


def test_g1c_kernel_has_identical_nontrace_instruction_sequence():
    production = _body("mf310p_v2_kernel.asc")
    diagnosed = _body("mf310p_v2_g1c_kernel.asc")
    diagnosed = [
        line for line in diagnosed if not line.startswith("G1CMark(")
        and line != "uint32_t traceStep = 0;"
    ]
    assert production == diagnosed, (
        "G1-C 诊断核发生了与追踪无关的算法/通信指令改变，禁止上机"
    )


def test_g1c_build_is_disabled_by_default():
    cmake = (SRC / "CMakeLists.txt").read_text(encoding="utf-8")
    bridge = (SRC.parents[3] / "cmake/memfabric_310p_v2.cmake").read_text(encoding="utf-8")
    assert 'option(VLLM_ASCEND_310P_G1C_TRACE' in cmake
    assert '"Build opt-in 310P3 G1-C phase-trace kernel and host observer" OFF' in cmake
    assert "mf310p_v2_g1c_kernel.asc" in cmake
    assert "mf310p_v2_g1c_observer.cpp" in cmake
    assert "if(VLLM_ASCEND_310P_G1C_TRACE)" in bridge
    assert "MF310P_G1C_TRACE=1" in bridge


def test_public_torch_op_is_unchanged():
    serve = (SRC / "torch_adapter/mf310p_v2_serve.cpp").read_text(encoding="utf-8")
    assert 'memfabric_mm_ar_v2_allreduce(Tensor x, Tensor weight' in serve
    assert 'memfabric_mm_ar_v2_shutdown() -> ()' in serve
    assert 'MF310P_G1C_TRACE' in serve
    assert 'g1c_stop_observer(s)' in serve


def test_device_wire_is_not_touched():
    header = (SRC / "op_kernel/mf310p_v2_g1c_trace.h").read_text(encoding="utf-8")
    assert "kG1CWaitEnter" in header
    assert "DataCacheCleanAndInvalid" in header
    assert "64ULL * core" in header
