# SPDX-License-Identifier: Apache-2.0
"""C3 单变量：不能删除缓存维护；源代码合同不代替设备/性能验证。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROD = (ROOT / "src/op_kernel/mf310p_v2_kernel.asc").read_text()
COMMON = (ROOT / "src/op_kernel/mf310p_v2_kernel_common.h").read_text()
DIAG = (ROOT / "src/op_kernel/mf310p_v2_diagnostics.asc").read_text()


def test_every_cache_line_clean_preserved():
    start = PROD.index("inline void CleanLinesUnrolled4")
    stop = PROD.index("template <const MatmulApiStaticTiling& TILING", start)
    helper = PROD[start:stop]
    assert helper.count("DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE>(g);") == 4
    assert helper.count("g.SetGlobalBuffer(") == 4
    assert "b += 256" in helper
    for offset in ("p + b)", "p + b + 64)", "p + b + 128)", "p + b + 192)"):
        assert offset in helper
    assert "ENTIRE_DATA_CACHE" not in helper
    assert "CleanLinesUnrolled4(ownC + start + coreOff, perCoreBytes)" in PROD
    assert PROD.count("CleanLinesUnrolled4(ownPeerC + base, perCoreBytes)") == 2
    assert "CleanLines(status, 64)" in COMMON  # 错误路径仍旧实现


def test_all_chunk_geometries_are_aligned():
    assert "static_assert((CHUNK * kRowBytes / kCores) % 256 == 0" in PROD
    for chunk in (256, 512, 1024):
        per_core_bytes = chunk * 2048 * 2 // 8
        assert per_core_bytes % 256 == 0
        assert per_core_bytes // 64 == 4 * (per_core_bytes // 256)
    # 几何：M=4096，CHUNK=1024，8 核，源/目的各 4 个 Chunk。
    per_core_all = (4096 * 2048 * 2 // 8)
    assert 2 * per_core_all // 64 == 65536


def test_protocol_and_d3_add_untouched():
    assert "kProductionAddTile = 2048;" in PROD
    assert "pipe.InitBuffer(mmUb, TOTAL_UB_SIZE - kProductionUBForVector - kUBForIB)" in PROD
    assert "mm.IterateAll(cg);" in PROD
    assert "smem_shm_sdma_signal_at(" in PROD
    assert "smem_shm_sdma_wait_at(" in PROD
    assert "WaitReady(ibGm, ibUb, core)" in PROD
    assert "ReleaseAdd(ibGm, ibUb, core)" in PROD
    assert "WaitAddDone(ibGm, ibUb, core)" in PROD
    assert "smem_shm_sdma_quiet_at(reserved)" in PROD
    assert "kTagDone" in PROD
    assert "kProductionAddTile" not in DIAG
    assert "CleanLinesUnrolled4" not in COMMON
    assert "CleanLinesUnrolled4" not in DIAG
