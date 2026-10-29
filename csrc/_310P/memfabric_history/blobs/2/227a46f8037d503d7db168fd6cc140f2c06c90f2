"""G1-C 取证埋点源码防回归测试，不要求 CANN/NPU。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def test_trace_isolated_from_protocol_and_opt_in():
    header = (SRC / "op_kernel/mf310p_v2_g1c_trace.h").read_text()
    kernel = (SRC / "op_kernel/mf310p_v2_kernel.asc").read_text()
    cm = (SRC / "CMakeLists.txt").read_text()
    assert "kG1cTraceOffset = 64" in header
    assert "kG1cTraceSlotBytes = 64" in header
    assert "MF310P_V2_G1C_TRACE" in header
    assert "G1cStage(status, core" in kernel
    assert "CheckOrTrap(smem_shm_sdma_quiet_at" in kernel
    assert '"Enable invasive G1-C per-core progress stamps (diagnosis only)" OFF' in cm


def test_stage_entry_generation_and_sampler():
    header = (SRC / "op_kernel/mf310p_v2_g1c_trace.h").read_text()
    host = (SRC / "op_host/mf310p_v2_trial_adapter.cpp").read_text()
    assert "if (stage == 10)" in header
    assert "slot + 8" in header
    assert "MF310P_V2_G1C_TRACE" in host
    assert "aclrtSetCurrentContext(context)" in host
    assert "ACL_MEMCPY_DEVICE_TO_HOST" in host
    assert "g1c_stop.store(true)" in host


def test_device_library_separated_by_filename():
    cm = (ROOT.parents[2] / "cmake/memfabric_310p_v2.cmake").read_text()
    assert "libmf310p_v2_trial_device_g1c.so" in cm
    assert "MF310P_V2_G1C_TRACE=1" in cm
    assert "libmf310p_v2_trial_device.so" in cm
