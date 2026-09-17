#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${MF310P_BRIDGE_BUILD_DIR:-${SCRIPT_DIR}/build}"
INSTALL_DIR="${MF310P_BRIDGE_INSTALL_DIR:-${BUILD_DIR}/install}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
MF_ROOT="${VLLM_ASCEND_310P_MEMFABRIC_ROOT:-}"
MF_LIBS="${VLLM_ASCEND_310P_MEMFABRIC_LIBRARIES:-}"
DEVICE_OBJECT="${MEMFABRIC_DEVICE_OBJECT:-}"

fail() {
    echo "[mf310p bridge] ERROR: $*" >&2
    exit 1
}

[[ -n "${MF_ROOT}" ]] || fail \
    "VLLM_ASCEND_310P_MEMFABRIC_ROOT is required and must point to the install prefix produced by wgm-dev-310p MemFabric."
[[ -d "${MF_ROOT}" ]] || fail \
    "custom MemFabric install prefix does not exist: ${MF_ROOT}"
[[ -n "${MF_LIBS}" ]] || fail \
    "VLLM_ASCEND_310P_MEMFABRIC_LIBRARIES is required. List the libraries produced/required by the wgm-dev-310p install; library names are intentionally not guessed."
[[ -n "${DEVICE_OBJECT}" ]] || fail \
    "MEMFABRIC_DEVICE_OBJECT is required. Compile memfabric310p_device.asc with the same 310P AscendC/custom-MemFabric toolchain used by example 08, then pass the resulting object here."
[[ -f "${DEVICE_OBJECT}" ]] || fail \
    "MEMFABRIC_DEVICE_OBJECT does not exist: ${DEVICE_OBJECT}"

# An explicit include list wins. Otherwise collect only directories underneath
# the selected wgm-dev-310p installation. This never searches system/upstream
# MemFabric locations.
if [[ -n "${MEMFABRIC_INCLUDE_DIRS:-}" ]]; then
    MF_INCLUDE_DIRS="${MEMFABRIC_INCLUDE_DIRS}"
else
    candidates=(
        "${MF_ROOT}/include"
        "${MF_ROOT}/include/smem"
        "${MF_ROOT}/include/smem/host"
        "${MF_ROOT}/include/smem/device"
    )
    while IFS= read -r -d '' dir; do
        candidates+=("${dir}")
    done < <(find "${MF_ROOT}" -mindepth 2 -maxdepth 4 -type d \
        \( -path '*/include' -o -path '*/include/smem' -o -path '*/include/smem/host' -o -path '*/include/smem/device' \) \
        -print0 2>/dev/null || true)

    include_list=()
    for dir in "${candidates[@]}"; do
        [[ -d "${dir}" ]] || continue
        duplicate=0
        for existing in "${include_list[@]:-}"; do
            if [[ "${existing}" == "${dir}" ]]; then
                duplicate=1
                break
            fi
        done
        [[ ${duplicate} -eq 0 ]] && include_list+=("${dir}")
    done
    [[ ${#include_list[@]} -gt 0 ]] || fail \
        "no include directories found below ${MF_ROOT}; set MEMFABRIC_INCLUDE_DIRS explicitly."
    MF_INCLUDE_DIRS="$(IFS=';'; echo "${include_list[*]}")"
fi

if [[ -n "${MEMFABRIC_LIBRARY_DIR:-}" ]]; then
    MF_LIBRARY_DIR="${MEMFABRIC_LIBRARY_DIR}"
else
    MF_LIBRARY_DIR=""
    for dir in "${MF_ROOT}/lib64" "${MF_ROOT}/lib"; do
        if [[ -d "${dir}" ]]; then
            MF_LIBRARY_DIR="${dir}"
            break
        fi
    done
    if [[ -z "${MF_LIBRARY_DIR}" ]]; then
        while IFS= read -r -d '' dir; do
            MF_LIBRARY_DIR="${dir}"
            break
        done < <(find "${MF_ROOT}" -mindepth 2 -maxdepth 4 -type d \
            \( -name lib64 -o -name lib \) -print0 2>/dev/null || true)
    fi
    [[ -n "${MF_LIBRARY_DIR}" ]] || fail \
        "no library directory found below ${MF_ROOT}; set MEMFABRIC_LIBRARY_DIR explicitly."
fi

mkdir -p "${BUILD_DIR}" "${INSTALL_DIR}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DASCEND_HOME_PATH="${ASCEND_HOME_PATH}" \
    -DMEMFABRIC_INCLUDE_DIRS="${MF_INCLUDE_DIRS}" \
    -DMEMFABRIC_LIBRARY_DIR="${MF_LIBRARY_DIR}" \
    -DMEMFABRIC_LIBRARIES="${MF_LIBS}" \
    -DMEMFABRIC_DEVICE_OBJECT="${DEVICE_OBJECT}"

cmake --build "${BUILD_DIR}" -j "${MAX_JOBS:-$(nproc)}"
cmake --install "${BUILD_DIR}"

BRIDGE_SO="${INSTALL_DIR}/lib/libvllm_ascend_memfabric310p_adapter.so"
[[ -f "${BRIDGE_SO}" ]] || fail "bridge build completed but output is missing: ${BRIDGE_SO}"

cat <<EOF
[mf310p bridge] build complete
  custom MemFabric install: ${MF_ROOT}
  include dirs:             ${MF_INCLUDE_DIRS}
  library dir:              ${MF_LIBRARY_DIR}
  bridge:                   ${BRIDGE_SO}

Before launching vLLM:
  export VLLM_ASCEND_310P_MEMFABRIC_ADAPTER_SO='${BRIDGE_SO}'
  export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1
EOF
