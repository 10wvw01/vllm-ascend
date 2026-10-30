# SPDX-License-Identifier: Apache-2.0
"""Configure the actual V2 build graph with a fake SDK; never invoke BiSheng."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]
SRC = ROOT / 'csrc/_310P/memfabric_mm_ar_v2/src'


@pytest.mark.parametrize('selected', ['ON', 'OFF'])
@pytest.mark.parametrize('diagnostics', ['ON', 'OFF'])
def test_selected_and_legacy_build_groups_are_independent(tmp_path, selected, diagnostics):
    cmake = shutil.which('cmake')
    if not cmake:
        candidate = Path('/tmp/mf310p-l1-move-venv/bin/cmake')
        cmake = str(candidate) if candidate.is_file() else None
    assert cmake, 'CMake is required for build-graph regression'
    linked_src = tmp_path/'csrc/_310P/memfabric_mm_ar_v2/src'
    linked_src.parent.mkdir(parents=True)
    linked_src.symlink_to(SRC, target_is_directory=True)
    cann = tmp_path/'cann'
    (cann/'bin').mkdir(parents=True)
    (cann/'bin/bisheng').symlink_to('/bin/true')
    sdk = tmp_path/'sdk'
    (sdk/'include/smem/device').mkdir(parents=True)
    (sdk/'include/smem/device/smem_shm_aicore_base_sdma.h').write_text('// CPU configuration fixture')
    (tmp_path/'dummy.cpp').write_text('int fixture() { return 0; }')
    (tmp_path/'CMakeLists.txt').write_text(f'''cmake_minimum_required(VERSION 3.16)
project(V2SelectedBuildGraph LANGUAGES CXX)
set(SOC_VERSION ascend310p)
set(ASCEND_HOME_PATH "{cann}")
add_library(extension SHARED dummy.cpp)
include("{ROOT / 'cmake/memfabric_310p_v2.cmake'}")
vllm_ascend_configure_310p_memfabric_v2_trial(extension)
get_target_property(srcs extension SOURCES)
file(WRITE "${{CMAKE_BINARY_DIR}}/actual_sources.txt" "${{srcs}}")
get_target_property(links extension LINK_LIBRARIES)
file(WRITE "${{CMAKE_BINARY_DIR}}/actual_links.txt" "${{links}}")
''')
    env = dict(os.environ, VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR='1', MEMFABRIC_HYBRID_HOME_PATH=str(sdk))
    build = tmp_path/'build'
    result = subprocess.run([cmake, '-S', str(tmp_path), '-B', str(build), '-G', 'Unix Makefiles',
                             '-DVLLM_ASCEND_310P_V2_BUILD_SELECTED='+selected,
                             '-DVLLM_ASCEND_310P_V2_BUILD_DIAGNOSTICS='+diagnostics],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    sources = (build/'actual_sources.txt').read_text()
    links = (build/'actual_links.txt').read_text()
    assert 'mf310p_v2_trial_adapter.cpp' in sources
    assert 'mf310p_v2_serve.cpp' in sources
    assert ('mf310p_v2_trial_diagnostics.cpp' in sources) == (diagnostics == 'ON')
    assert ('mf310p_v2_selected_adapter.cpp' in sources) == (selected == 'ON')
    assert ('libmf310p_v2_selected_device.so' in links) == (selected == 'ON')
    if selected == 'ON':
        graph = (build/'CMakeFiles/mf310p_v2_selected_device_lib.dir/build.make').read_text()
        assert 'mf310p_v2_dcci_entire_candidate.asc' in graph
        assert 'mf310p_v2_e4_layered_candidates.asc' in graph
        assert 'mf310p_v2_kernel_common.h' in graph
        assert 'mf310p_v2_e3_group2048.asc' not in graph
        assert 'private_host.cpp' not in graph
    help_result = subprocess.run([cmake, '--build', str(build), '--target', 'help'], text=True, capture_output=True)
    assert help_result.returncode == 0, help_result.stderr
    assert ('mf310p_v2_selected_device_lib' in help_result.stdout) == (selected == 'ON')
