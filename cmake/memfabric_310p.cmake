# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The customized wgm-dev-310p MemFabric is an independently built/installed
# external SDK.  Do not assume that it is part of CANN or that an upstream
# MemFabric package/library is present on the system.

set(VLLM_ASCEND_310P_MEMFABRIC_ROOT
    "$ENV{VLLM_ASCEND_310P_MEMFABRIC_ROOT}"
    CACHE PATH
    "Install prefix of the independently built customized 310P MemFabric SDK")

set(VLLM_ASCEND_310P_MEMFABRIC_LIBRARIES
    "$ENV{VLLM_ASCEND_310P_MEMFABRIC_LIBRARIES}"
    CACHE STRING
    "Semicolon-separated libraries exported by the customized 310P MemFabric SDK")

function(vllm_ascend_configure_310p_memfabric target)
    if(NOT VLLM_ASCEND_310P_MEMFABRIC_ROOT)
        message(FATAL_ERROR
            "310P MemFabric backend requested but VLLM_ASCEND_310P_MEMFABRIC_ROOT is empty. "
            "Build/install the customized wgm-dev-310p MemFabric first and point this variable at its install prefix.")
    endif()

    set(_mf_include_hints
        "${VLLM_ASCEND_310P_MEMFABRIC_ROOT}/include"
        "${VLLM_ASCEND_310P_MEMFABRIC_ROOT}/include/memfabric"
        "${VLLM_ASCEND_310P_MEMFABRIC_ROOT}/include/smem")

    find_path(VLLM_ASCEND_310P_MEMFABRIC_INCLUDE_DIR
        NAMES smem.h
        HINTS ${_mf_include_hints}
        NO_DEFAULT_PATH)
    find_path(VLLM_ASCEND_310P_MEMFABRIC_SHM_INCLUDE_DIR
        NAMES smem_shm.h
        HINTS ${_mf_include_hints}
        NO_DEFAULT_PATH)
    find_path(VLLM_ASCEND_310P_MEMFABRIC_SDMA_INCLUDE_DIR
        NAMES smem_shm_aicore_sdma.h
        HINTS ${_mf_include_hints}
        NO_DEFAULT_PATH)

    if(NOT VLLM_ASCEND_310P_MEMFABRIC_INCLUDE_DIR OR
       NOT VLLM_ASCEND_310P_MEMFABRIC_SHM_INCLUDE_DIR OR
       NOT VLLM_ASCEND_310P_MEMFABRIC_SDMA_INCLUDE_DIR)
        message(FATAL_ERROR
            "Customized 310P MemFabric headers were not found under "
            "${VLLM_ASCEND_310P_MEMFABRIC_ROOT}. Required: smem.h, smem_shm.h, smem_shm_aicore_sdma.h")
    endif()

    target_include_directories(${target} PRIVATE
        ${VLLM_ASCEND_310P_MEMFABRIC_INCLUDE_DIR}
        ${VLLM_ASCEND_310P_MEMFABRIC_SHM_INCLUDE_DIR}
        ${VLLM_ASCEND_310P_MEMFABRIC_SDMA_INCLUDE_DIR})

    # The customized branch is installed independently and its final library
    # names are branch/build specific.  Keep the link contract explicit rather
    # than guessing an upstream library name.
    target_link_directories(${target} PRIVATE
        "${VLLM_ASCEND_310P_MEMFABRIC_ROOT}/lib64"
        "${VLLM_ASCEND_310P_MEMFABRIC_ROOT}/lib")

    if(VLLM_ASCEND_310P_MEMFABRIC_LIBRARIES)
        target_link_libraries(${target} PRIVATE ${VLLM_ASCEND_310P_MEMFABRIC_LIBRARIES})
    endif()

    target_compile_definitions(${target} PRIVATE VLLM_ASCEND_HAS_310P_MEMFABRIC_SDK=1)

    message(STATUS
        "Configured external customized 310P MemFabric SDK: root=${VLLM_ASCEND_310P_MEMFABRIC_ROOT}, "
        "libraries=${VLLM_ASCEND_310P_MEMFABRIC_LIBRARIES}")
endfunction()
