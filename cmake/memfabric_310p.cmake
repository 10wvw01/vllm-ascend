# 310P customized MemFabric integration for the Qwen3.6 W8A8 o_proj pipeline.
#
# Contract:
#   1. Build/install GDD_ESCC/memfabric_hybrid:wgm-dev-310p first.
#   2. Point VLLM_ASCEND_310P_MEMFABRIC_ROOT at that install prefix.
#   3. Explicitly provide the libraries required by that customized install.
#   4. Compile memfabric310p_device.asc with the same customized 310P toolchain
#      used by the MemFabric example and provide the resulting object.
#
# No upstream/official MemFabric installation is searched or used.

function(vllm_ascend_configure_310p_memfabric target)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "Unknown target passed to vllm_ascend_configure_310p_memfabric: ${target}")
    endif()

    if(NOT DEFINED ENV{VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ} OR
       NOT "$ENV{VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ}" STREQUAL "1")
        message(STATUS "310P customized MemFabric o_proj fusion is disabled")
        return()
    endif()

    if(NOT SOC_VERSION MATCHES "ascend310p.*")
        message(FATAL_ERROR
            "VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1 is only valid for ascend310p*, got ${SOC_VERSION}")
    endif()

    set(_mf_root "$ENV{VLLM_ASCEND_310P_MEMFABRIC_ROOT}")
    if(_mf_root STREQUAL "")
        message(FATAL_ERROR
            "VLLM_ASCEND_310P_MEMFABRIC_ROOT is required. It must point to the install prefix produced by wgm-dev-310p MemFabric")
    endif()
    if(NOT IS_DIRECTORY "${_mf_root}")
        message(FATAL_ERROR "wgm-dev-310p MemFabric install prefix does not exist: ${_mf_root}")
    endif()

    # The customized branch may install headers either flat under include/ or
    # into a subdirectory. Search only inside the explicitly selected install
    # prefix so an upstream/system MemFabric can never be picked accidentally.
    find_path(_mf_smem_include
        NAMES smem.h
        PATHS "${_mf_root}/include"
        PATH_SUFFIXES "" memfabric memfabric_hybrid
        NO_DEFAULT_PATH)
    find_path(_mf_shm_include
        NAMES smem_shm.h
        PATHS "${_mf_root}/include"
        PATH_SUFFIXES "" memfabric memfabric_hybrid
        NO_DEFAULT_PATH)
    find_path(_mf_sdma_include
        NAMES smem_shm_aicore_sdma.h
        PATHS "${_mf_root}/include"
        PATH_SUFFIXES "" memfabric memfabric_hybrid
        NO_DEFAULT_PATH)

    if(NOT _mf_smem_include OR NOT _mf_shm_include OR NOT _mf_sdma_include)
        message(FATAL_ERROR
            "The selected wgm-dev-310p install is missing required 310P headers. "
            "Need smem.h, smem_shm.h and smem_shm_aicore_sdma.h under ${_mf_root}/include")
    endif()

    set(_mf_libs_raw "$ENV{VLLM_ASCEND_310P_MEMFABRIC_LIBRARIES}")
    if(_mf_libs_raw STREQUAL "")
        message(FATAL_ERROR
            "VLLM_ASCEND_310P_MEMFABRIC_LIBRARIES is required and must list the libraries produced/required by the installed wgm-dev-310p MemFabric")
    endif()
    # Environment uses CMake's native semicolon-separated list syntax.
    set(_mf_libs ${_mf_libs_raw})

    set(_mf_device_object "$ENV{VLLM_ASCEND_310P_MEMFABRIC_DEVICE_OBJECT}")
    if(_mf_device_object STREQUAL "")
        message(FATAL_ERROR
            "VLLM_ASCEND_310P_MEMFABRIC_DEVICE_OBJECT is required. Compile csrc/memfabric_o_proj/external/memfabric310p_device.asc with the same wgm-dev-310p 310P toolchain used by example 08")
    endif()
    if(NOT EXISTS "${_mf_device_object}")
        message(FATAL_ERROR "310P MemFabric device object does not exist: ${_mf_device_object}")
    endif()

    # The adapter is not a separately installed shared library. It is compiled
    # directly into vllm_ascend_C and calls the selected customized MemFabric.
    target_sources(${target} PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/../csrc/memfabric_o_proj/external/memfabric310p_adapter.cpp"
        "${_mf_device_object}"
    )

    target_include_directories(${target} PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/../csrc/memfabric_o_proj/external"
        "${_mf_smem_include}"
        "${_mf_shm_include}"
        "${_mf_sdma_include}"
    )

    # Support the conventional lib/lib64 layouts produced by install prefixes.
    if(IS_DIRECTORY "${_mf_root}/lib64")
        target_link_directories(${target} PRIVATE "${_mf_root}/lib64")
    endif()
    if(IS_DIRECTORY "${_mf_root}/lib")
        target_link_directories(${target} PRIVATE "${_mf_root}/lib")
    endif()

    target_link_libraries(${target} PRIVATE ${_mf_libs})
    target_compile_definitions(${target} PRIVATE VLLM_ASCEND_ENABLE_310P_MEMFABRIC_O_PROJ)

    # Keep runtime lookup inside the selected customized installation when the
    # caller supplied library names instead of absolute paths.
    set_property(TARGET ${target} APPEND PROPERTY BUILD_RPATH
        "${_mf_root}/lib64;${_mf_root}/lib")
    set_property(TARGET ${target} APPEND PROPERTY INSTALL_RPATH
        "${_mf_root}/lib64;${_mf_root}/lib")

    message(STATUS "310P customized MemFabric o_proj fusion enabled")
    message(STATUS "  MemFabric install: ${_mf_root}")
    message(STATUS "  MemFabric libraries: ${_mf_libs}")
    message(STATUS "  Device object: ${_mf_device_object}")
endfunction()
