# 310P customized MemFabric integration for the Qwen3.6 o_proj pipeline.
#
# Contract (all verified on the real 310P3 server):
#   1. Build/install GDD_ESCC/memfabric_hybrid:wgm-dev-310p (V5, mailbox ring
#      epoch API) first:
#        cmake -B build -DXPU_TYPE=NPU -DBUILD_PYTHON=OFF && cmake --build build -j
#      then copy the produced output/ tree (or cmake --install with an
#      existing prefix) to e.g. /opt/memfabric-wgm-dev-310p-v5.
#   2. Point VLLM_ASCEND_310P_MEMFABRIC_ROOT at that install prefix. The V5
#      install uses a split layout:
#        <root>/smem/include/host/       smem.h, smem_shm.h (host adapter)
#        <root>/smem/include/device/     smem_shm_aicore_base_sdma.h (.asc)
#        <root>/smem/lib64/libmf_smem.so
#        <root>/hybm/lib64/libmf_hybm_core.so
#        <root>/acc_links/lib64/libacc_tcp_net.so
#        <root>/hybm/aicpu_kernel/       libmf_sdma_orch_v6.json (epoch kernel)
#   3. Explicitly provide the libraries produced by that install via
#      VLLM_ASCEND_310P_MEMFABRIC_LIBRARIES (semicolon-separated list):
#        <root>/smem/lib64/libmf_smem.so;<root>/hybm/lib64/libmf_hybm_core.so;<root>/acc_links/lib64/libacc_tcp_net.so
#   4. csrc/memfabric_o_proj/external/memfabric310p_device.asc is compiled with
#      the same customized 310P toolchain used by MemFabric example 08
#      (bisheng --npu-arch=dav-2002, unified host+AICore compilation) into a
#      shared library whose kernels launch through the public ACL binary API.
#      The build is performed automatically by this CMake module; pass
#      VLLM_ASCEND_310P_MEMFABRIC_DEVICE_LIBRARY to override with a prebuilt .so.
#   5. At runtime, MF_SDMA_ORCH_JSON must point at
#      <root>/hybm/aicpu_kernel/libmf_sdma_orch_v6.json. The V5 in-process
#      auto-discovery assumes a flat <prefix>/lib64 layout that the split
#      install does not have, and the kfc fallback channel is unavailable on
#      the target server (device-side kfc_min.so is missing), so the launch
#      json must always be provided explicitly.
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
            "VLLM_ASCEND_310P_MEMFABRIC_ROOT is required. It must point to the V5 install prefix produced by wgm-dev-310p MemFabric")
    endif()
    if(NOT IS_DIRECTORY "${_mf_root}")
        message(FATAL_ERROR "wgm-dev-310p MemFabric install prefix does not exist: ${_mf_root}")
    endif()

    # The V5 (mailbox ring epoch) install splits host headers under
    # smem/include/host and device headers under smem/include/device. Search
    # only inside the explicitly selected install prefix so an upstream/system
    # MemFabric can never be picked accidentally.
    find_path(_mf_host_include
        NAMES smem_shm.h
        PATHS "${_mf_root}/smem/include/host"
        NO_DEFAULT_PATH)
    find_path(_mf_device_include
        NAMES smem_shm_aicore_base_sdma.h
        PATHS "${_mf_root}/smem/include/device"
        NO_DEFAULT_PATH)
    find_path(_mf_orch_json
        NAMES libmf_sdma_orch_v6.json
        PATHS "${_mf_root}/hybm/aicpu_kernel"
        NO_DEFAULT_PATH)

    if(NOT _mf_host_include OR NOT _mf_device_include OR NOT _mf_orch_json)
        message(FATAL_ERROR
            "The selected install is missing the required V5 wgm-dev-310p pieces. "
            "Need smem_shm.h under ${_mf_root}/smem/include/host, "
            "smem_shm_aicore_base_sdma.h under ${_mf_root}/smem/include/device, and "
            "libmf_sdma_orch_v6.json under ${_mf_root}/hybm/aicpu_kernel. "
            "The V4 (kfc workspace) layout is no longer supported.")
    endif()

    set(_mf_libs_raw "$ENV{VLLM_ASCEND_310P_MEMFABRIC_LIBRARIES}")
    if(_mf_libs_raw STREQUAL "")
        message(FATAL_ERROR
            "VLLM_ASCEND_310P_MEMFABRIC_LIBRARIES is required and must list the libraries produced by the installed V5 wgm-dev-310p MemFabric, e.g. '${_mf_root}/smem/lib64/libmf_smem.so;${_mf_root}/hybm/lib64/libmf_hybm_core.so;${_mf_root}/acc_links/lib64/libacc_tcp_net.so'")
    endif()
    # Environment uses CMake's native semicolon-separated list syntax.
    set(_mf_libs ${_mf_libs_raw})
    foreach(_mf_lib IN LISTS _mf_libs)
        if(NOT EXISTS "${_mf_lib}")
            message(FATAL_ERROR "VLLM_ASCEND_310P_MEMFABRIC_LIBRARIES entry does not exist: ${_mf_lib}")
        endif()
    endforeach()

    # Collect every lib64 directory of the split install for the runtime
    # lookup path (the V5 libraries themselves carry no rpath).
    set(_mf_lib_dirs "")
    foreach(_mf_subdir smem hybm acc_links acc_offload)
        foreach(_mf_libdir lib64 lib)
            if(IS_DIRECTORY "${_mf_root}/${_mf_subdir}/${_mf_libdir}")
                list(APPEND _mf_lib_dirs "${_mf_root}/${_mf_subdir}/${_mf_libdir}")
            endif()
        endforeach()
    endforeach()

    # ---- Device-side AICore cooperation library (memfabric310p_device.asc) ----
    # Compiled with the example-08 bisheng toolchain into a shared library whose
    # kernel launches resolve through the public ACL binary API (verified: no
    # private runtime symbols are referenced). The .asc includes the V5 device
    # header (mailbox ring API) but needs no MemFabric library at link time;
    # signal/wait/quiet are inlined into the kernels.
    set(_mf_asc_src "${CMAKE_SOURCE_DIR}/csrc/memfabric_o_proj/external/memfabric310p_device.asc")
    if(NOT EXISTS "${_mf_asc_src}")
        message(FATAL_ERROR "memfabric310p_device.asc not found: ${_mf_asc_src}")
    endif()

    set(_mf_device_lib_override "$ENV{VLLM_ASCEND_310P_MEMFABRIC_DEVICE_LIBRARY}")
    if(NOT _mf_device_lib_override STREQUAL "")
        if(NOT EXISTS "${_mf_device_lib_override}")
            message(FATAL_ERROR "VLLM_ASCEND_310P_MEMFABRIC_DEVICE_LIBRARY does not exist: ${_mf_device_lib_override}")
        endif()
        set(_mf_device_lib "${_mf_device_lib_override}")
        message(STATUS "310P MemFabric device library (prebuilt): ${_mf_device_lib}")
    else()
        find_program(BISHENG_COMPILER
            NAMES bisheng
            HINTS "${ASCEND_HOME_PATH}/bin" "${ASCEND_HOME_PATH}/compiler/aarch64-linux/bin"
            NO_DEFAULT_PATH)
        if(NOT BISHENG_COMPILER)
            message(FATAL_ERROR
                "bisheng compiler not found under ${ASCEND_HOME_PATH}. It is required to compile memfabric310p_device.asc (same toolchain as MemFabric example 08)")
        endif()

        set(_mf_device_lib "${CMAKE_CURRENT_BINARY_DIR}/libmf310p_device.so")
        add_custom_command(
            OUTPUT "${_mf_device_lib}"
            COMMAND ${BISHENG_COMPILER} --npu-arch=dav-2002 -O2 -std=c++17 -w
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
            COMMENT "Compiling 310P MemFabric device cooperation library (bisheng dav-2002)"
            VERBATIM)
        message(STATUS "310P MemFabric device library will be built from: ${_mf_asc_src}")
    endif()

    # The adapter is not a separately installed shared library. It is compiled
    # directly into vllm_ascend_C and calls the selected customized MemFabric.
    target_sources(${target} PRIVATE
        "${CMAKE_SOURCE_DIR}/csrc/memfabric_o_proj/external/memfabric310p_adapter.cpp"
    )

    # The device cooperation library is a bisheng-built shared object. Driving
    # it through a custom target triggers the .asc build; it must also be
    # linked explicitly (a .so listed in target_sources is not linked).
    if(NOT _mf_device_lib_override STREQUAL "")
        target_link_libraries(${target} PRIVATE "${_mf_device_lib}")
    else()
        add_custom_target(mf310p_device_lib DEPENDS "${_mf_device_lib}")
        add_dependencies(${target} mf310p_device_lib)
        target_link_libraries(${target} PRIVATE "${_mf_device_lib}")
    endif()

    target_include_directories(${target} PRIVATE
        "${CMAKE_SOURCE_DIR}/csrc/memfabric_o_proj/external"
        "${_mf_host_include}"
    )

    # Link the customized libraries explicitly. --no-as-needed keeps them in
    # DT_NEEDED even though vllm_ascend_C references symbols only from
    # libmf_smem.so directly. The V5 libraries carry no rpath themselves, so
    # every lib directory of the split install must be resolvable through the
    # extension's own runtime path.
    foreach(_mf_libdir IN LISTS _mf_lib_dirs)
        target_link_directories(${target} PRIVATE "${_mf_libdir}")
    endforeach()
    target_link_libraries(${target} PRIVATE
        "-Wl,--no-as-needed"
        ${_mf_libs}
        "-Wl,--as-needed")

    target_compile_definitions(${target} PRIVATE VLLM_ASCEND_ENABLE_310P_MEMFABRIC_O_PROJ)

    # Keep runtime lookup inside the selected customized installation. The
    # main CMakeLists sets rpath via target_link_options, so the same mechanism
    # is required here (set_property INSTALL_RPATH is overridden by it).
    set(_mf_rpath "$ORIGIN")
    foreach(_mf_libdir IN LISTS _mf_lib_dirs)
        string(APPEND _mf_rpath ":${_mf_libdir}")
    endforeach()
    target_link_options(${target} PRIVATE "-Wl,-rpath,${_mf_rpath}")

    # Ship the device cooperation library next to vllm_ascend_C so the $ORIGIN
    # rpath resolves after editable/regular installs.
    install(FILES "${_mf_device_lib}" DESTINATION .)

    message(STATUS "310P customized MemFabric o_proj fusion enabled (V5 epoch API)")
    message(STATUS "  MemFabric install: ${_mf_root}")
    message(STATUS "  MemFabric libraries: ${_mf_libs}")
    message(STATUS "  Epoch launch json: ${_mf_orch_json}")
    message(STATUS "  Device library: ${_mf_device_lib}")
endfunction()
