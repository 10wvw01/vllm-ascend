// SPDX-License-Identifier: Apache-2.0
// E3独立实验：借用已验证trial上下文，安全发射E1或两种Chunk2048私有Device核。
// 不改变正式插件/Device SO/Serving ABI；此模块只能在相同源码ABI、经SO身份门禁的进程加载。
#include "../../src/op_host/mf310p_v2_trial_internal.h"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>

extern "C" int mf310p_v2_dcci_entire_launch_async(
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, aclrtStream);

extern "C" int mf310p_v2_e3_group2048_launch_async(
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
    auto launch=(kind==2)?mf310p_v2_dcci_entire_launch_async:
                           mf310p_v2_e3_group2048_launch_async;
    const uint32_t logicalChunk=(kind==2)?1024u:2048u;
    const uint32_t kernelMode=(kind==2)?1u:(kind==3?1024u:2048u);
    return launch(l.pool_base,x,w,y,ownC,peerC,peerDst,expectedRecvDst,
                  reinterpret_cast<uint64_t>(s->ib),
                  reinterpret_cast<uint64_t>(s->status),
                  l.ack_slot,l.peer_ack_slot,l.expected_credit_dst,
                  logicalChunk,m/logicalChunk,kernelMode,enable,stream);
}
} // namespace

extern "C" int mf310p_e3_run_private(
    mf310p_v2_trial_t* s, uint64_t x, uint64_t w, uint64_t y,
    uint32_t m, uint32_t kind, void* stream_ptr)
{
    // kind=2: E1原样1024，kind=3: E3组2048+2次1024子MM，
    // kind=4: E3组2048+一次2048真实MM；Pool ABI仍保持1024。
    if (!s || !s->v1_pool || !s->ib_probed || !stream_ptr ||
        !x || !w || !y || (kind!=2 && kind!=3 && kind!=4) ||
        s->chunk_rows!=1024 || (m!=2048 && m!=4096) ||
        (m%2048)!=0 || s->layout.arena_bytes<static_cast<uint64_t>(m)*4096)
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
