# 310P MemFabric integration for the Qwen3.6 o_proj fused path.
#
# MemFabric is an external black box.  vLLM-Ascend consumes only the installed
# public host/device headers and libmf_smem.so exposed by the current
# memfabric_hybrid run package.  AICPU deployment, orchestration, mailbox
# layout and all transitive runtime dependencies belong to MemFabric.
#
# Runtime/build contract:
#   1. Install memfabric_hybrid:wgm-dev-310p with its run package.
#   2. source <install>/set_env.sh so MEMFABRIC_HYBRID_HOME_PATH is exported.
#   3. Set VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1 and build vLLM-Ascend.
#
# No MemFabric internal library, AICPU json, mailbox header or split source-tree
# layout is part of the vLLM-Ascend contract.

function(vllm_ascend_configure_310p_memfabric target)
    if(NOT TARGET ${target})
        message(FATAL_ERROR
            "Unknown target passed to vllm_ascend_configure_310p_memfabric: ${target}")
    endif()

    if(NOT DEFINED ENV{VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ} OR
       NOT "$ENV{VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ}" STREQUAL "1")
        message(STATUS "310P MemFabric o_proj fusion is disabled")
        return()
    endif()

    if(NOT SOC_VERSION MATCHES "ascend310p.*")
        message(FATAL_ERROR
            "VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1 is only valid for "
            "ascend310p*, got ${SOC_VERSION}")
    endif()

    set(_mf_home "$ENV{MEMFABRIC_HYBRID_HOME_PATH}")
    if(_mf_home STREQUAL "")
        message(FATAL_ERROR
            "MEMFABRIC_HYBRID_HOME_PATH is not set. Install the current "
            "memfabric_hybrid run package and source its set_env.sh first.")
    endif()
    if(NOT IS_DIRECTORY "${_mf_home}")
        message(FATAL_ERROR
            "MEMFABRIC_HYBRID_HOME_PATH does not exist: ${_mf_home}")
    endif()

    # Current run packages expose public headers under <home>/include/smem.
    # Keep the documented arch-qualified form as a compatibility search path;
    # both remain within the single installed package selected by _mf_home.
    set(_mf_arch_dir "${CMAKE_SYSTEM_PROCESSOR}-linux")
    find_path(_mf_host_include
        NAMES smem_shm.h
        PATHS
            "${_mf_home}/include/smem/host"
            "${_mf_home}/${_mf_arch_dir}/include/smem/host"
        NO_DEFAULT_PATH)
    find_path(_mf_device_include
        NAMES smem_shm_aicore_base_sdma.h
        PATHS
            "${_mf_home}/include/smem/device"
            "${_mf_home}/${_mf_arch_dir}/include/smem/device"
        NO_DEFAULT_PATH)

    # libmf_smem.so is the only MemFabric library in the public API contract.
    # The installer adds the arch lib64 directory to LD_LIBRARY_PATH; rpath is
    # also recorded below so editable/regular installs resolve consistently.
    find_library(_mf_smem_lib
        NAMES mf_smem
        PATHS
            "${_mf_home}/${_mf_arch_dir}/lib64"
            "${_mf_home}/${_mf_arch_dir}/lib"
        NO_DEFAULT_PATH)

    # Some package builders use an explicit aarch64-linux directory even when
    # CMAKE_SYSTEM_PROCESSOR is reported differently. Search installed lib64
    # children without depending on any internal MemFabric component names.
    if(NOT _mf_smem_lib)
        file(GLOB _mf_public_lib_dirs
            LIST_DIRECTORIES true
            "${_mf_home}/*/lib64"
            "${_mf_home}/*/lib")
        find_library(_mf_smem_lib
            NAMES mf_smem
            PATHS ${_mf_public_lib_dirs}
            NO_DEFAULT_PATH)
    endif()

    if(NOT _mf_host_include OR NOT _mf_device_include OR NOT _mf_smem_lib)
        message(FATAL_ERROR
            "Selected MemFabric installation is missing its public contract. "
            "Need smem_shm.h, smem_shm_aicore_base_sdma.h and libmf_smem.so "
            "under MEMFABRIC_HYBRID_HOME_PATH=${_mf_home}.")
    endif()

    # ---- vLLM-owned AscendC fused-kernel library ----
    set(_mf_asc_src
        "${CMAKE_SOURCE_DIR}/csrc/memfabric_o_proj/external/memfabric310p_device.asc")
    if(NOT EXISTS "${_mf_asc_src}")
        message(FATAL_ERROR "memfabric310p_device.asc not found: ${_mf_asc_src}")
    endif()

    set(_mf_device_lib_override
        "$ENV{VLLM_ASCEND_310P_MEMFABRIC_DEVICE_LIBRARY}")
    if(NOT _mf_device_lib_override STREQUAL "")
        if(NOT EXISTS "${_mf_device_lib_override}")
            message(FATAL_ERROR
                "VLLM_ASCEND_310P_MEMFABRIC_DEVICE_LIBRARY does not exist: "
                "${_mf_device_lib_override}")
        endif()
        set(_mf_device_lib "${_mf_device_lib_override}")
    else()
        find_program(BISHENG_COMPILER
            NAMES bisheng
            HINTS
                "${ASCEND_HOME_PATH}/bin"
                "${ASCEND_HOME_PATH}/compiler/aarch64-linux/bin"
            NO_DEFAULT_PATH)
        if(NOT BISHENG_COMPILER)
            message(FATAL_ERROR
                "bisheng compiler not found under ASCEND_HOME_PATH; it is "
                "required for the dav-2002 fused device library.")
        endif()

        set(_mf_device_lib "${CMAKE_CURRENT_BINARY_DIR}/libmf310p_device.so")
        add_custom_command(
            OUTPUT "${_mf_device_lib}"
            COMMAND ${BISHENG_COMPILER}
                    --npu-arch=dav-2002 -O2 -std=c++17 -w
                    -shared -fPIC
                    -x asc "${_mf_asc_src}" -x none
                    -o "${_mf_device_lib}"
                    -I${ASCEND_HOME_PATH}/aarch64-linux/include
                    -I${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include
                    -I${_mf_device_include}
                    -L${ASCEND_HOME_PATH}/aarch64-linux/lib64
                    -lascendcl -lruntime -ldl -pthread -lm
                    -Wl,-rpath,${ASCEND_HOME_PATH}/aarch64-linux/lib64
            DEPENDS "${_mf_asc_src}"
            COMMENT
                "Compiling 310P public-MemFabric fused device library (dav-2002)"
            VERBATIM)
    endif()

    target_sources(${target} PRIVATE
        "${CMAKE_SOURCE_DIR}/csrc/memfabric_o_proj/external/memfabric310p_adapter.cpp")

    if(NOT _mf_device_lib_override STREQUAL "")
        target_link_libraries(${target} PRIVATE "${_mf_device_lib}")
    else()
        add_custom_target(mf310p_device_lib DEPENDS "${_mf_device_lib}")
        add_dependencies(${target} mf310p_device_lib)
        target_link_libraries(${target} PRIVATE "${_mf_device_lib}")
    endif()

    target_include_directories(${target} PRIVATE
        "${CMAKE_SOURCE_DIR}/csrc/memfabric_o_proj/external"
        "${_mf_host_include}")

    target_link_libraries(${target} PRIVATE
        "-Wl,--no-as-needed"
        "${_mf_smem_lib}"
        "-Wl,--as-needed")

    target_compile_definitions(
        ${target} PRIVATE VLLM_ASCEND_ENABLE_310P_MEMFABRIC_O_PROJ)

    get_filename_component(_mf_lib_dir "${_mf_smem_lib}" DIRECTORY)
    target_link_options(
        ${target} PRIVATE "-Wl,-rpath,$ORIGIN:${_mf_lib_dir}")

    install(FILES "${_mf_device_lib}" DESTINATION .)

    message(STATUS "310P MemFabric o_proj fusion enabled (public API)")
    message(STATUS "  MemFabric home: ${_mf_home}")
    message(STATUS "  Host include: ${_mf_host_include}")
    message(STATUS "  Device include: ${_mf_device_include}")
    message(STATUS "  Public library: ${_mf_smem_lib}")
    message(STATUS "  vLLM device library: ${_mf_device_lib}")
endfunction()
