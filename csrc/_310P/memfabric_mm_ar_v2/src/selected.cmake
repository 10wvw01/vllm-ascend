# SPDX-License-Identifier: Apache-2.0
# Selected source package: do not duplicate or edit the measured device kernels.
option(VLLM_ASCEND_310P_V2_BUILD_SELECTED
    "Build fixed-shape performance-selected V2 E1/T8 C API" ON)

function(mf310p_v2_add_selected target bisheng memfabric_device_include source_dir)
    if(NOT VLLM_ASCEND_310P_V2_BUILD_SELECTED)
        return()
    endif()
    # Explicit root keeps compatibility with the repository's CMake 3.16 floor.
    set(_src "${source_dir}")
    set(_kernels
        "${_src}/op_kernel/mf310p_v2_dcci_entire_candidate.asc"
        "${_src}/op_kernel/mf310p_v2_e4_layered_candidates.asc")
    set(_lib "${CMAKE_CURRENT_BINARY_DIR}/libmf310p_v2_selected_device.so")
    add_custom_command(
        OUTPUT "${_lib}"
        COMMAND "${bisheng}"
            --npu-arch=dav-2002 -O2 -std=c++17 -w
            -shared -fPIC -x asc ${_kernels} -x none -o "${_lib}"
            -I${ASCEND_HOME_PATH}/aarch64-linux/include
            -I${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include
            -I${memfabric_device_include}
            -L${ASCEND_HOME_PATH}/aarch64-linux/lib64
            -lascendcl -lruntime -ldl -pthread -lm
            -Wl,-rpath,${ASCEND_HOME_PATH}/aarch64-linux/lib64
        DEPENDS ${_kernels} "${_src}/op_kernel/mf310p_v2_kernel_common.h"
        COMMENT "Build selected V2 E1/T8 kernels (dav-2002); no device validation implied"
        VERBATIM)
    add_custom_target(mf310p_v2_selected_device_lib DEPENDS "${_lib}")
    add_dependencies(${target} mf310p_v2_selected_device_lib)
    target_sources(${target} PRIVATE "${_src}/op_host/mf310p_v2_selected_adapter.cpp")
    target_link_libraries(${target} PRIVATE
        "-Wl,--no-as-needed" "${_lib}" "-Wl,--as-needed")
    install(FILES "${_lib}" DESTINATION .)
    message(STATUS "V2 selected C API: fixed M2048/E1 or M4096/T8; Serving routing unchanged")
endfunction()
