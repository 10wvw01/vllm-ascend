cmake_minimum_required(VERSION 3.16)
project(vllm_ascend_memfabric310p_adapter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# These paths point to artifacts produced by building/installing wgm-dev-310p
# MemFabric. No official/upstream MemFabric installation is used here.
set(MEMFABRIC_INCLUDE_DIRS "" CACHE STRING "Semicolon-separated include directories from installed wgm-dev-310p MemFabric")
set(MEMFABRIC_LIBRARY_DIR "" CACHE PATH "Library directory from installed wgm-dev-310p MemFabric")
set(MEMFABRIC_LIBRARIES "" CACHE STRING "Semicolon-separated libraries produced/required by wgm-dev-310p MemFabric")
set(MEMFABRIC_DEVICE_OBJECT "" CACHE FILEPATH "Object generated from memfabric310p_device.asc with the 310P customized MemFabric example toolchain")
set(ASCEND_HOME_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "CANN toolkit root")

if(NOT MEMFABRIC_INCLUDE_DIRS)
    message(FATAL_ERROR "MEMFABRIC_INCLUDE_DIRS must point at headers installed by wgm-dev-310p MemFabric")
endif()
if(NOT MEMFABRIC_LIBRARY_DIR)
    message(FATAL_ERROR "MEMFABRIC_LIBRARY_DIR must point at libraries installed by wgm-dev-310p MemFabric")
endif()
if(NOT MEMFABRIC_LIBRARIES)
    message(FATAL_ERROR "MEMFABRIC_LIBRARIES must explicitly list the wgm-dev-310p MemFabric libraries to link")
endif()
if(NOT MEMFABRIC_DEVICE_OBJECT)
    message(FATAL_ERROR "MEMFABRIC_DEVICE_OBJECT is required. Compile memfabric310p_device.asc using the same 310P toolchain/flags as example 08 first")
endif()
if(NOT EXISTS "${MEMFABRIC_DEVICE_OBJECT}")
    message(FATAL_ERROR "MEMFABRIC_DEVICE_OBJECT does not exist: ${MEMFABRIC_DEVICE_OBJECT}")
endif()

# Fail during configure rather than later in C++ compilation if the selected
# customized MemFabric installation does not contain the required 310P APIs.
find_path(MF310P_SMEM_H_DIR smem.h PATHS ${MEMFABRIC_INCLUDE_DIRS} NO_DEFAULT_PATH)
find_path(MF310P_SHM_H_DIR smem_shm.h PATHS ${MEMFABRIC_INCLUDE_DIRS} NO_DEFAULT_PATH)
find_path(MF310P_SDMA_H_DIR smem_shm_aicore_sdma.h PATHS ${MEMFABRIC_INCLUDE_DIRS} NO_DEFAULT_PATH)
if(NOT MF310P_SMEM_H_DIR OR NOT MF310P_SHM_H_DIR OR NOT MF310P_SDMA_H_DIR)
    message(FATAL_ERROR
        "The selected wgm-dev-310p MemFabric installation is missing one of: "
        "smem.h, smem_shm.h, smem_shm_aicore_sdma.h. include dirs=${MEMFABRIC_INCLUDE_DIRS}")
endif()

add_library(vllm_ascend_memfabric310p_adapter SHARED
    memfabric310p_adapter.cpp
    ${MEMFABRIC_DEVICE_OBJECT}
)

target_include_directories(vllm_ascend_memfabric310p_adapter PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${MEMFABRIC_INCLUDE_DIRS}
    ${ASCEND_HOME_PATH}/include
)

target_link_directories(vllm_ascend_memfabric310p_adapter PRIVATE
    ${MEMFABRIC_LIBRARY_DIR}
    ${ASCEND_HOME_PATH}/lib64
)

target_link_libraries(vllm_ascend_memfabric310p_adapter PRIVATE
    ascendcl
    ${MEMFABRIC_LIBRARIES}
)

set_target_properties(vllm_ascend_memfabric310p_adapter PROPERTIES
    OUTPUT_NAME "vllm_ascend_memfabric310p_adapter"
)

install(TARGETS vllm_ascend_memfabric310p_adapter
    LIBRARY DESTINATION lib
)
install(FILES memfabric310p_adapter_api.h
    DESTINATION include
)
