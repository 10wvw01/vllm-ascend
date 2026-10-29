// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace mfv3 {
constexpr uint32_t kAbi = 1;
constexpr uint32_t kCores = 8;
constexpr uint32_t kWidth = 2048;
constexpr uint32_t kMaxRows = 4096;
constexpr uint64_t kRowBytes = kWidth * 2ULL;
constexpr uint64_t kArenaBytes = kMaxRows * kRowBytes;
// A3 sessions lay the pool out as [Y slot0 | Y slot1 | S slot0 | S slot1 | ctrl];
// V2/V3 paths only touch the first three arenas, leaving the layout compatible.
constexpr uint64_t kPoolBytes = 5 * kArenaBytes;
// User control lines are disjoint from data and from MemFabric's reserved tail.
constexpr uint64_t kCtrlSrcOff = 2 * kArenaBytes;
constexpr uint64_t kCtrlDstOff = kCtrlSrcOff + 64;
constexpr uint64_t kIbBytes = 8192;
constexpr uint32_t kMaxEpoch = (1u << 24) - 1;
constexpr uint32_t kMaxPacket = 4u << 20;
constexpr uint32_t kDefaultPacket = 4u << 20;
enum Phase : uint32_t { READY = 1, REDUCE = 2, GATHER = 3, DONE = 4, RAW_ADD = 5 };
enum Mode : uint32_t { MATMUL = 0, PARTIAL = 1, RAW = 2 };
// Exactly 40 bits: phase[39:36], epoch[35:12], group[11:6], packet[5:0].
constexpr uint64_t Tag(uint32_t phase, uint32_t epoch, uint32_t group, uint32_t packet)
{
    return (uint64_t{phase} << 36) | (uint64_t{epoch} << 12) | (uint64_t{group} << 6) | packet;
}
constexpr bool GroupValid(uint32_t rows)
{
    return rows == 256 || rows == 512 || rows == 1024 || rows == 2048;
}
constexpr bool PacketValid(uint32_t bytes)
{
    return bytes >= 512 && bytes <= kMaxPacket && bytes % 512 == 0;
}
constexpr bool ShapeValid(uint32_t rows, uint32_t group)
{
    return GroupValid(group) && rows > 0 && rows <= kMaxRows && rows % group == 0;
}
constexpr uint32_t PaddedRows(uint32_t rows, uint32_t group)
{
    return GroupValid(group) && rows > 0 && rows <= kMaxRows ? ((rows + group - 1) / group) * group : 0;
}
constexpr uint32_t PacketCount(uint64_t bytes, uint32_t packet)
{
    return static_cast<uint32_t>((bytes + packet - 1) / packet);
}
// L2 attribution trace layout, in uint64 slots; mirrored by the Python adapter
// and the attribution worker's parser. Null trace pointer disables stamping.
constexpr uint32_t kTraceBytes = 4096;
constexpr uint32_t kTraceU64 = kTraceBytes / 8;
constexpr uint64_t kTraceMagic = 0x4C325431ULL; // "L2T1"
constexpr uint32_t kTraceHeader = 16; // magic, rows, group, mode, rank, epoch, groups
constexpr uint32_t kTraceStride = 32; // per-core region: start, end, prod[8], barrier[8], step[8], tail[4]
constexpr uint32_t kTraceSubBase = kTraceHeader + kCores * kTraceStride; // 5 core0 sub-stamps per group
} // namespace mfv3
