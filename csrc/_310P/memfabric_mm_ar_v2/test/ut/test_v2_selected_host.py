# SPDX-License-Identifier: Apache-2.0
"""Compile the real selected Host against fake ACL/launch backends on CPU.

Checks dispatch, first-write guards, context lifetime and failure propagation;
this does not execute or validate an Ascend device kernel.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / 'src'


@pytest.fixture(scope='module')
def host_executable(tmp_path_factory):
    compiler = shutil.which('g++')
    assert compiler, 'g++ is required for selected Host lifecycle regression'
    tmp = tmp_path_factory.mktemp('v2-selected-host')
    (tmp / 'acl').mkdir()
    (tmp / 'acl/acl_rt.h').write_text('''#pragma once
#include <cstddef>
using aclrtStream = void*;
using aclError = int;
extern "C" int aclrtSynchronizeDevice();
extern "C" int aclrtSynchronizeStream(aclrtStream);
extern "C" int aclrtMemsetAsync(void*, size_t, int, size_t, aclrtStream);
''')
    harness = tmp / 'test.cpp'
    harness.write_text(r'''
#include "mf310p_v2_selected_api.h"
#include "mf310p_v2_trial_internal.h"
#include <array>
#include <cassert>
#include <string>
#include <vector>

int creates=0, frees=0, probes=0, device_syncs=0, stream_syncs=0, clears=0;
int device_rc=0, stream_rc=0, clear_rc=0, launch_rc=0, probe_rc=0, status_rc=0;
uint64_t protocol=0;
struct Call { bool e1; std::array<uint64_t,13> p; uint32_t group,chunks,mode,enable; void* stream; };
std::vector<Call> calls;
void* stream=reinterpret_cast<void*>(0x100);

extern "C" int mf310p_v2_trial_create(int rank,const char*,uint32_t chunk,mf310p_v2_trial_t** out) {
    ++creates;
    assert(chunk==1024);
    auto* s=new mf310p_v2_trial_t;
    s->rank=rank;s->chunk_rows=chunk;
    s->v1_pool=reinterpret_cast<mf310p_context_t*>(1);
    s->ib=reinterpret_cast<void*>(0x1010);s->status=&protocol;
    s->layout.pool_base=0x11000000;s->layout.send_arena=0x22000000;
    s->layout.recv_arena=0x33000000;s->layout.peer_segment=0x44000000;
    s->layout.peer_recv_arena=0x55000000;s->layout.arena_bytes=8192ull*4096;
    s->layout.expected_recv_base=0x66000000;s->layout.ack_slot=0x77000000;
    s->layout.peer_ack_slot=0x88000000;s->layout.expected_credit_dst=0x99000000;
    *out=s;return 0;
}
extern "C" int mf310p_v2_trial_ib_probe(mf310p_v2_trial_t* p,uint32_t,void*) {
    ++probes;p->ib_probed=probe_rc==0;return probe_rc;
}
extern "C" int mf310p_v2_trial_check_status(mf310p_v2_trial_t*,uint64_t* out) {
    if(!status_rc)*out=protocol;return status_rc;
}
extern "C" int mf310p_v2_trial_destroy(mf310p_v2_trial_t* p) { ++frees;delete p;return 0; }
extern "C" int aclrtSynchronizeDevice() { ++device_syncs;return device_rc; }
extern "C" int aclrtSynchronizeStream(void* s) { assert(s==stream);++stream_syncs;return stream_rc; }
extern "C" int aclrtMemsetAsync(void* p,size_t cap,int value,size_t n,void* s) {
    assert(p==&protocol && cap==64 && value==0 && n==64 && s==stream);
    ++clears;return clear_rc;
}
#define ARGS uint64_t g,uint64_t x,uint64_t w,uint64_t y,uint64_t own,uint64_t peer,uint64_t dst,uint64_t exp,uint64_t ib,uint64_t st,uint64_t ack,uint64_t pack,uint64_t credit,uint32_t group,uint32_t chunks,uint32_t mode,uint32_t enable,void* s
#define SAVE(e) calls.push_back({e,{g,x,w,y,own,peer,dst,exp,ib,st,ack,pack,credit},group,chunks,mode,enable,s});return launch_rc
extern "C" int mf310p_v2_dcci_entire_launch_async(ARGS) { SAVE(true); }
extern "C" int mf310p_v2_e4_group2048_launch_async(ARGS) { SAVE(false); }

mf310p_v2_selected_t* make(uint32_t m=2048,int rank=0) {
    mf310p_v2_selected_t* p=nullptr;
    assert(mf310p_v2_selected_create(rank,"tcp://test",m,&p)==0);
    assert(p);return p;
}
void probe(mf310p_v2_selected_t* p) { assert(mf310p_v2_selected_ib_probe(p,64,stream)==0); }
int run(mf310p_v2_selected_t* p,uint64_t y=0x900) { return mf310p_v2_selected_run(p,0x700,0x800,y,stream); }
void close(mf310p_v2_selected_t* p) { assert(mf310p_v2_selected_destroy(p)==0); }
int main(int argc,char** argv) {
    assert(argc==2);std::string mode=argv[1];
    if(mode=="plan") {
        mf310p_v2_selected_plan_t plan{};
        assert(mf310p_v2_selected_get_plan(2048,&plan)==0 && plan.version==7 && plan.group_rows==1024 && plan.add_elements==2048);
        assert(mf310p_v2_selected_get_plan(4096,&plan)==0 && plan.version==15 && plan.mm_rows==1024 && plan.group_rows==2048 && plan.add_elements==8192);
        for(uint32_t m:{0u,2047u,2049u,3072u,4095u,4097u}) {
            assert(mf310p_v2_selected_get_plan(m,&plan)==MF310P_V2_SELECTED_UNSUPPORTED);
            assert(plan.abi==0);
            mf310p_v2_selected_t* p=reinterpret_cast<mf310p_v2_selected_t*>(1);
            assert(mf310p_v2_selected_create(0,"url",m,&p)==MF310P_V2_SELECTED_UNSUPPORTED && !p);
        }
        assert(creates==0);
    } else if(mode=="routing") {
        for(uint32_t m:{2048u,4096u}) for(int rank:{0,1}) {
            auto* p=make(m,rank);probe(p);calls.clear();assert(run(p)==0);
            assert(calls.size()==3);
            for(size_t i=0;i<3;++i) {
                const auto& c=calls[i];assert(c.e1==(m==2048));
                assert(c.group==(m==2048?1024u:2048u) && c.chunks==2);
                assert(c.mode==(m==2048?1u:2u) && c.enable==(i==2?1u:0u));
                assert(c.p[4]==(rank==0?0x22000000u:0x33000000u));
                assert(c.p[5]==(rank==0?0x33000000u:0x22000000u));
                assert(c.p[6]==(rank==0?0x44000000u:0x55000000u));
                assert(c.p[7]==(rank==0?0x66000000u:0x66000000u-8192ull*4096));
                assert(c.p[8]==0x1010 && c.p[10]==0x77000000 && c.p[11]==0x88000000 && c.p[12]==0x99000000);
            }
            close(p);
        }
    } else if(mode=="warm_lifetime") {
        for(int j=0;j<2;++j) {
            auto* p=make();probe(p);int before=device_syncs;size_t n=calls.size();
            assert(run(p)==0 && device_syncs==before+1 && calls.size()==n+3);
            assert(run(p)==0 && device_syncs==before+1 && calls.size()==n+4);
            assert(run(p,0x901)==0 && device_syncs==before+2 && calls.size()==n+5);
            assert(run(p)==0 && device_syncs==before+2);close(p);
        }
        assert(creates==2 && frees==2 && clears==8);
    } else if(mode=="preconditions") {
        auto* p=make();assert(run(p)==MF310P_V2_SELECTED_NOT_READY && calls.empty());
        assert(mf310p_v2_selected_ib_probe(p,0,stream)==MF310P_V2_SELECTED_INVALID && probes==0);
        probe(p);
        assert(mf310p_v2_selected_ib_probe(p,64,stream)==MF310P_V2_SELECTED_INVALID && probes==1);
        assert(mf310p_v2_selected_run(p,1,2,3,reinterpret_cast<void*>(0x200))==MF310P_V2_SELECTED_STREAM_MISMATCH && calls.empty());
        assert(mf310p_v2_selected_run(p,0,2,3,stream)==MF310P_V2_SELECTED_INVALID && calls.empty());close(p);
    } else if(mode=="launch_failure") {
        auto* p=make();probe(p);launch_rc=91;assert(run(p)==91);size_t n=calls.size();
        assert(run(p)==MF310P_V2_SELECTED_POISONED && calls.size()==n);close(p);
    } else if(mode=="real_launch_failure") {
        auto* p=make();probe(p);assert(run(p)==0);launch_rc=92;assert(run(p)==92);
        size_t n=calls.size();assert(run(p)==MF310P_V2_SELECTED_POISONED && calls.size()==n);close(p);
    } else if(mode=="sync_failure") {
        auto* p=make();probe(p);device_rc=93;assert(run(p)==93 && clears==0);
        assert(run(p)==MF310P_V2_SELECTED_POISONED);device_rc=0;close(p);
    } else if(mode=="clear_failure") {
        auto* p=make();probe(p);clear_rc=94;assert(run(p)==94 && calls.size()==2);
        assert(run(p)==MF310P_V2_SELECTED_POISONED && calls.size()==2);close(p);
    } else if(mode=="protocol_failure") {
        auto* p=make();probe(p);assert(run(p)==0);protocol=9;uint64_t status=0;
        assert(mf310p_v2_selected_check_status(p,&status)==0 && status==9);
        assert(run(p)==MF310P_V2_SELECTED_POISONED);
        assert(mf310p_v2_selected_destroy(p)==MF310P_V2_SELECTED_PROTOCOL && frees==0);
        protocol=0;close(p); // Test cleanup only; production failure ends both workers.
    } else if(mode=="destroy_inflight") {
        auto* p=make();probe(p);assert(run(p)==0);stream_rc=95;
        assert(mf310p_v2_selected_destroy(p)==95 && frees==0);
        assert(run(p)==MF310P_V2_SELECTED_POISONED);stream_rc=0;close(p);
    } else if(mode=="probe_failure") {
        auto* p=make();probe_rc=96;
        assert(mf310p_v2_selected_ib_probe(p,64,stream)==96);
        assert(run(p)==MF310P_V2_SELECTED_POISONED && calls.empty());close(p);
    } else if(mode=="status_failure") {
        auto* p=make();probe(p);uint64_t status=0;status_rc=97;
        assert(mf310p_v2_selected_check_status(p,&status)==97);
        assert(run(p)==MF310P_V2_SELECTED_POISONED);
        assert(mf310p_v2_selected_destroy(p)==97 && frees==0);status_rc=0;close(p);
    } else { return 2; }
    assert(creates==frees);
}
''')
    executable = tmp / 'host-contract'
    result = subprocess.run([compiler, '-std=c++17', '-O2', '-I'+str(tmp), '-I'+str(SRC/'op_host'),
                             str(SRC/'op_host/mf310p_v2_selected_adapter.cpp'), str(harness), '-o', str(executable)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.parametrize('scenario', ['plan', 'routing', 'warm_lifetime', 'preconditions', 'launch_failure',
                                      'real_launch_failure', 'sync_failure', 'clear_failure', 'protocol_failure',
                                      'destroy_inflight', 'probe_failure', 'status_failure'])
def test_selected_host_contract(host_executable, scenario):
    result = subprocess.run([str(host_executable), scenario], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
