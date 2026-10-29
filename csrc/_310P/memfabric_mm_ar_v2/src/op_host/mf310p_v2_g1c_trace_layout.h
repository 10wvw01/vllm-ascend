#pragma once
// G1-C Host/设备侧阶段槽布局（仅诊断用途，不修改 MemFabric wire ABI）。
#include <cstdint>
struct alignas(64) MF310PG1CTraceSlot {
    uint32_t magic; // 0x47314331
    uint32_t core;
    uint64_t seq;
    uint32_t phase;
    uint32_t chunk;
    uint32_t step;
    uint32_t reserved[9];
};
static_assert(sizeof(MF310PG1CTraceSlot) == 64, "G1-C slot must occupy one cache line");
static_assert(alignof(MF310PG1CTraceSlot) == 64, "G1-C slot must be cache-line aligned");
constexpr uint32_t MF310P_G1C_MAGIC = 0x47314331u;
constexpr uint32_t MF310P_G1C_CORES = 8;
constexpr uint64_t MF310P_G1C_TRACE_BYTES = 64ULL * MF310P_G1C_CORES;
