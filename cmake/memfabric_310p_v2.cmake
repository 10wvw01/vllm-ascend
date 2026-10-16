# Experimental 310P3 V2 native prototype; NOT enabled by default.
function(vllm_ascend_configure_310p_memfabric_v2_trial target)
    if(NOT "$ENV{VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR}" STREQUAL "1")
        message(FATAL_ERROR "V2 trial requires V1's installed MemFabric public API build")
    endif()
    if(NOT SOC_VERSION MATCHES "ascend310p.*")
        message(FATAL_ERROR "V2 trial is restricted to ascend310p*")
    endif()
    set(_dir "${CMAKE_SOURCE_DIR}/csrc/_310P/memfabric_mm_ar_v2")
    set(_mf_home "$ENV{MEMFABRIC_HYBRID_HOME_PATH}")
    set(_arch "${CMAKE_SYSTEM_PROCESSOR}-linux")
    find_path(_v2_mf_device_inc NAMES smem_shm_aicore_base_sdma.h
        PATHS "${_mf_home}/include/smem/device"
              "${_mf_home}/${_arch}/include/smem/device"
        NO_DEFAULT_PATH)
    find_program(_v2_bisheng NAMES bisheng
        HINTS "${ASCEND_HOME_PATH}/bin"
              "${ASCEND_HOME_PATH}/compiler/aarch64-linux/bin"
        NO_DEFAULT_PATH)
    if(NOT _v2_mf_device_inc OR NOT _v2_bisheng)
        message(FATAL_ERROR "V2 trial needs bisheng and installed MemFabric SDMA public header")
    endif()
    set(_lib "${CMAKE_CURRENT_BINARY_DIR}/libmf310p_v2_trial_device.so")
    add_custom_command(
        OUTPUT "${_lib}"
        COMMAND ${_v2_bisheng}
            --npu-arch=dav-2002 -O2 -std=c++17 -w
            -shared -fPIC -x asc "${_dir}/mf310p_v2_trial_device.asc"
            -x none -o "${_lib}"
            -I${ASCEND_HOME_PATH}/aarch64-linux/include
            -I${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include
            -I${_v2_mf_device_inc}
            -L${ASCEND_HOME_PATH}/aarch64-linux/lib64
            -lascendcl -lruntime -ldl -pthread -lm
            -Wl,-rpath,${ASCEND_HOME_PATH}/aarch64-linux/lib64
        DEPENDS "${_dir}/mf310p_v2_trial_device.asc"
        COMMENT "Building experimental 310P V2 single-kernel MM+AR (dav-2002)"
        VERBATIM)
    add_custom_target(mf310p_v2_trial_device_lib DEPENDS "${_lib}")
    add_dependencies(${target} mf310p_v2_trial_device_lib)
    target_sources(${target} PRIVATE "${_dir}/mf310p_v2_trial_adapter.cpp")
    target_include_directories(${target} PRIVATE "${_dir}")
    target_link_libraries(${target} PRIVATE "-Wl,--no-as-needed" "${_lib}" "-Wl,--as-needed")
    install(FILES "${_lib}" DESTINATION .)
    message(WARNING "V2 trial compiled, but it has NO model-router integration: prototype only")
endfunction()
