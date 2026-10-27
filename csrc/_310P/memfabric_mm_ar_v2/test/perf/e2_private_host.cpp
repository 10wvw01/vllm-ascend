// SPDX-License-Identifier: Apache-2.0
// A1五层实验：借用已验证trial上下文，私有发射B2/E设备核。
// 不改变正式插件/Device SO/Serving ABI；此模块只能在相同源码ABI、经SO身份门禁的进程加载。
#include "../../src/op_host/mf310p_v2_trial_internal.h"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>

extern "C" int mf310p_v2_b2_full_mm_ar_launch_async(
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, aclrtStream);
extern "C" int mf310p_v2_dcci_entire_launch_async(
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, aclrtStream);

// E2独立公共符号，不与E1/B2正式候选相互覆盖。
extern "C" int mf310p_v2_e2_pipeline_launch_async(
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, aclrtStream);

namespace {
struct State {
    bool prewarmed = false;
    bool first_real = false;
    std::array<uint64_t,64> output_addresses{};
    uint32_t next = 0;
};
thread_local std::unordered_map<uintptr_t,State> states;

int invoke(mf310p_v2_trial_t* s, uint64_t x, uint64_t w, uint64_t y,
           uint32_t m, uint32_t kind, aclrtStream stream, uint32_t enable)
{
    const auto& l=s->layout;
    const bool rank0=(s->rank==0);
    const uint64_t ownC=rank0?l.send_arena:l.recv_arena;
    const uint64_t peerC=rank0?l.recv_arena:l.send_arena;
    const uint64_t peerDst=rank0?l.peer_segment:l.peer_recv_arena;
    const uint64_t expectedRecvDst=
        rank0?l.expected_recv_base:l.expected_recv_base-l.arena_bytes;
    auto launch=(kind==1)?mf310p_v2_b2_full_mm_ar_launch_async:
               (kind==2)?mf310p_v2_dcci_entire_launch_async:
                          mf310p_v2_e2_pipeline_launch_async;
    return launch(l.pool_base,x,w,y,ownC,peerC,peerDst,expectedRecvDst,
                  reinterpret_cast<uint64_t>(s->ib),
                  reinterpret_cast<uint64_t>(s->status),
                  l.ack_slot,l.peer_ack_slot,l.expected_credit_dst,
                  s->chunk_rows,m/s->chunk_rows,kind==1?0u:kind==3?1u:kind>=4?2u:1u,enable,stream);
}
} // namespace

extern "C" int mf310p_e2_run_private(
    mf310p_v2_trial_t* s, uint64_t x, uint64_t w, uint64_t y,
    uint32_t m, uint32_t kind, void* stream_ptr)
{
    // kind=1 B2;2 E1;3 E2-R-1024;4 E2-P-1024;5 E2-P-512.
    if (!s || !s->v1_pool || !s->ib_probed || !stream_ptr ||
        !x || !w || !y || (kind<1 || kind>5) ||
        s->chunk_rows!=(kind==5?512u:1024u) || !m || (m!=2048 && m!=4096) ||
        (m/s->chunk_rows)*s->chunk_rows!=m)
        return -1;
    const auto stream=static_cast<aclrtStream>(stream_ptr);
    // 状态按指针+kind隔离；服务启动不加载本库，各arm单独torchrun工作进程。
    const uintptr_t key=reinterpret_cast<uintptr_t>(s)*4+kind;
    State& guard=states[key];
    if (!guard.prewarmed) {
        for(int i=0;i<2;++i) {
            const int err=invoke(s,0,0,0,m,kind,stream,0);
            if(err) return err;
        }
        guard.prewarmed=true;
    }
    bool known=false;
    for(uint64_t p:guard.output_addresses) {
        if(p==y) { known=true; break; }
    }
    // 强制沿用正式D4首写与生命周期防护；前期已知地址后不引入重复设备同步。
    if(!guard.first_real || !known) {
        const int err=static_cast<int>(aclrtSynchronizeDevice());
        if(err) return err;
        if(!known) {
            guard.output_addresses[guard.next]=y;
            guard.next=(guard.next+1)%guard.output_addresses.size();
        }
    }
    guard.first_real=true;
    const int clear=static_cast<int>(
        aclrtMemsetAsync(s->status,kStatusSize,0,kStatusSize,stream));
    if(clear) return clear;
    return invoke(s,x,w,y,m,kind,stream,1);
}

// 2026-10-28 507014取证新增：以下三个只读探针仅在Python异常路径上调用，
// 正常测量路径零开销、不改变任何每次调用行为；用于挂死/超时后的现场保全。
// 依据 smem_shm_sdma_layout.h：保留区48KB，偏移 REQ_RING=0x0 / REQ_TAIL=0x800 /
// REQ_HEAD=0x808 / ARR_STAMP=0xA20 / ARR_MAIL=0x2A20 / ARR_HEAD=0xAA20。
// reserved = pool_base + symmetric_size*rank + (local_size - 48KB)。
extern "C" int mf310p_e2_diag_read_pool(
    mf310p_v2_trial_t* s, uint64_t offset, uint32_t len, void* out)
{
    if (!s || !s->v1_pool || !out || len == 0 || len > 65536) return -1;
    return static_cast<int>(aclrtMemcpy(out, len,
        reinterpret_cast<const void*>(s->layout.pool_base + offset), len,
        ACL_MEMCPY_DEVICE_TO_HOST));
}

extern "C" int mf310p_e2_diag_read_abs(
    mf310p_v2_trial_t* s, uint64_t addr, uint32_t len, void* out)
{
    if (!s || !out || len == 0 || len > 65536 || addr < 0x1000) return -1;
    return static_cast<int>(aclrtMemcpy(out, len,
        reinterpret_cast<const void*>(addr), len, ACL_MEMCPY_DEVICE_TO_HOST));
}

extern "C" int mf310p_e2_diag_pool_info(mf310p_v2_trial_t* s, uint64_t* out6)
{
    if (!s || !s->v1_pool || !out6) return -1;
    out6[0] = s->layout.pool_base;
    out6[1] = s->layout.symmetric_size;
    out6[2] = s->layout.local_size;
    out6[3] = s->rank;
    out6[4] = s->layout.symmetric_size * s->rank +
              (s->layout.local_size - 48ULL * 1024ULL);
    out6[5] = reinterpret_cast<uint64_t>(s->ib);
    return 0;
}
