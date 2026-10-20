#pragma once
// G1-C 设备端独立进度槽：每个 AICore 独占 64B，禁止与通信信箱复用。
// 实验版内部契约，不改变 MemFabric wire 或公开 Torch Op ABI。
#include <cstdint>
#include "kernel_operator.h"

namespace {
constexpr uint64_t kG1CMagic = 0x4731435048415345ULL; // ASCII G1CPHASE
constexpr uint32_t kG1CEntry = 1;
constexpr uint32_t kG1CMmEnter = 10;
constexpr uint32_t kG1CMmExit = 11;
constexpr uint32_t kG1CReadyEnter = 20;
constexpr uint32_t kG1CReadyExit = 21;
constexpr uint32_t kG1CSignalEnter = 30;
constexpr uint32_t kG1CSignalExit = 31;
constexpr uint32_t kG1CWaitEnter = 40;
constexpr uint32_t kG1CWaitExit = 41;
constexpr uint32_t kG1CReleaseEnter = 50;
constexpr uint32_t kG1CReleaseExit = 51;
constexpr uint32_t kG1CAddEnter = 60;
constexpr uint32_t kG1CAddExit = 61;
constexpr uint32_t kG1CAddDoneEnter = 70;
constexpr uint32_t kG1CAddDoneExit = 71;
constexpr uint32_t kG1CQuietEnter = 80;
constexpr uint32_t kG1CQuietExit = 81;
constexpr uint32_t kG1CDoneSignalEnter = 90;
constexpr uint32_t kG1CDoneSignalExit = 91;
constexpr uint32_t kG1CDoneWaitEnter = 100;
constexpr uint32_t kG1CDoneWaitExit = 101;
constexpr uint32_t kG1CExit = 110;
// 8×64B：GM 槽 [magic,u64 seq,u64 chunk,u64 phase,u64 core,u64 step,0,0]。
// 只写本人独立缓存行；打印由 Host 线程完成，设备核不能使用 printf。
__aicore__ inline void G1CMark(
    __gm__ uint8_t* trace, uint64_t launchSeq, uint32_t core,
    uint32_t chunk, uint32_t phase, uint32_t step)
{
    __gm__ uint8_t* slot = trace + 64ULL * core;
    __gm__ volatile uint64_t* words =
        reinterpret_cast<__gm__ volatile uint64_t*>(slot);
    words[0] = kG1CMagic;
    words[1] = launchSeq;
    words[2] = chunk;
    words[3] = phase;
    words[4] = core;
    words[5] = step;
    AscendC::GlobalTensor<int32_t> g;
    g.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(slot), 16);
    AscendC::DataCacheCleanAndInvalid<int32_t,
        AscendC::CacheLine::SINGLE_CACHE_LINE>(g);
}
} // 匿名命名空间
