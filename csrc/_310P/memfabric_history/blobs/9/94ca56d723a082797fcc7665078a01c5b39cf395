# Experimental 310P3 V2 native prototype; NOT enabled by default.
# 正式 V2 的主工程接线函数：target 为主 _C_ascend 扩展目标；不产生 ACLNN op_proto 算子包。
# 依赖 V1 安装的 MemFabric 公共 API，但 V2 设备代码从独立 src/op_kernel 编译与导出。
function(vllm_ascend_configure_310p_memfabric_v2_trial target)
    if(NOT "$ENV{VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR}" STREQUAL "1")
        message(FATAL_ERROR "V2 trial requires V1's installed MemFabric public API build")
    endif()
    if(NOT SOC_VERSION MATCHES "ascend310p.*")
        message(FATAL_ERROR "V2 trial is restricted to ascend310p*")
    endif()
    set(_dir "${CMAKE_SOURCE_DIR}/csrc/_310P/memfabric_mm_ar_v2/src")
    # src/CMakeLists.txt 集中定义正式 Host/Kernel 清单，诊断代码通过独立选项追加。
    include("${_dir}/CMakeLists.txt")
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
    # bisheng 针对 dav-2002 编译单独设备库，再通过 Host ABI 与 PyTorch 扩展对接；
    # 不运行 ACNN 算子包 tiling 注册，Host 显式负责静态模板 dispatch。
    add_custom_command(
        OUTPUT "${_lib}"
        COMMAND ${_v2_bisheng}
            --npu-arch=dav-2002 -O2 -std=c++17 -w
            -shared -fPIC -x asc ${MF310P_V2_KERNEL_SOURCES}
            -x none -o "${_lib}"
            -I${ASCEND_HOME_PATH}/aarch64-linux/include
            -I${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include
            -I${_v2_mf_device_inc}
            -L${ASCEND_HOME_PATH}/aarch64-linux/lib64
            -lascendcl -lruntime -ldl -pthread -lm
            -Wl,-rpath,${ASCEND_HOME_PATH}/aarch64-linux/lib64
        DEPENDS ${MF310P_V2_KERNEL_SOURCES}
            "${_dir}/op_kernel/mf310p_v2_kernel_common.h"
        COMMENT "构建 310P3 V2 正式 MM+AR Kernel 和可选诊断 Kernel (dav-2002)"
        VERBATIM)
    add_custom_target(mf310p_v2_trial_device_lib DEPENDS "${_lib}")
    add_dependencies(${target} mf310p_v2_trial_device_lib)
    # Host 与 Torch adapter 编入主扩展；设备 so 作为显式链接依赖并随包部署，
    # 不允许只替换 Python 文件而继续装载旧设备库。
    target_sources(${target} PRIVATE ${MF310P_V2_HOST_SOURCES})
    target_include_directories(${target} PRIVATE ${MF310P_V2_HOST_INCLUDE_DIRS})
    target_link_libraries(${target} PRIVATE "-Wl,--no-as-needed" "${_lib}" "-Wl,--as-needed")
    install(FILES "${_lib}" DESTINATION .)
    message(STATUS "V2 trial and opt-in 310P3 serving op compiled; V2 serving routing remains disabled by default")
endfunction()
