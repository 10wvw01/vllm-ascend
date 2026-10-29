# SPDX-License-Identifier: Apache-2.0
"""Compile the real host adapter against fake ACL/SMEM; exercise slot ownership."""

import subprocess
from pathlib import Path


def test_real_host_slots_and_generation(tmp_path):
    (tmp_path / "acl").mkdir()
    (tmp_path / "acl/acl_rt.h").write_text(r"""
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
using aclrtStream=void*;
constexpr int ACL_MEM_MALLOC_HUGE_FIRST=0,ACL_MEMCPY_DEVICE_TO_HOST=0;
inline int aclrtGetDevice(int32_t* p){*p=0;return 0;}
inline int aclrtMalloc(void** p,size_t n,int){*p=malloc(n);return 0;}
inline int aclrtFree(void* p){free(p);return 0;}
inline int aclrtMemset(void* p,size_t,int v,size_t n){
if(uintptr_t(p)<0x10000000||uintptr_t(p)>0x20000000)memset(p,v,n);return 0;}
inline int aclrtMemsetAsync(void* p,size_t a,int v,size_t n,void*){return aclrtMemset(p,a,v,n);}
inline int aclrtMemcpy(void* p,size_t,const void* q,size_t n,int){memcpy(p,q,n);return 0;}
inline int aclrtSynchronizeDevice(){return 0;}
inline int aclrtSynchronizeStream(void*){return 0;}
""")
    (tmp_path / "smem.h").write_text(
        "#pragma once\ninline int smem_init(int){return 0;}\ninline void smem_uninit(){}\n"
    )
    (tmp_path / "smem_shm.h").write_text(r"""
#pragma once
#include <cstring>
using smem_shm_t=void*;
struct smem_shm_config_t{bool startConfigStoreServer;};
constexpr int SMEMS_DATA_OP_SDMA=0;
inline int smem_shm_config_init(smem_shm_config_t*){return 0;}
inline int smem_shm_init(const char*,int,int,uint16_t,smem_shm_config_t*){return 0;}
inline void smem_shm_uninit(int){}
inline void* smem_shm_create(int,int,int,uint64_t,int,int,void** p){*p=(void*)0x10000000;return (void*)1;}
inline void* smem_shm_sdma_get_workspace(void*){return (void*)1;}
inline uint64_t smem_shm_get_symmetric_size(void*){return 64ULL<<20;}
inline int smem_shm_control_barrier(void*){return 0;}
unsigned peerMask=3,peerReady=1,peerHeld=0,peerOp=0;int destroyed=0;
inline int smem_shm_destroy(void*,int){destroyed++;return 0;}
inline int smem_shm_control_allgather(void*,const char* in,size_t n,char* out,size_t){
memcpy(out,in,n);memcpy(out+n,in,n);
if(n==36){auto p=reinterpret_cast<unsigned*>(out+n);
if(p[0]==1)p[3]=peerMask;else if(p[0]==2)p[3]=peerReady;else if(p[0]==3)p[3]=peerHeld;
if(peerOp)p[0]=peerOp;}
return 0;}
""")
    host = (Path(__file__).parents[1] / "src/op_host/mf310p_v3_host.cpp").resolve()
    source = tmp_path / "host.cpp"
    source.write_text(
        f'#include "{host}"\n'
        + r"""
#include <cassert>
extern "C" int mf310p_a3_device_run(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,
uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint32_t,uint32_t,uint32_t,
uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,aclrtStream){return 0;}
int enabledLaunches=0;
uint64_t g_traceSeen=0;
extern "C" int mf310p_v3_device_run(uint64_t,uint64_t,uint64_t,uint64_t y,uint64_t work,uint64_t,
uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint32_t,uint32_t,uint32_t,uint32_t,
uint32_t,uint32_t pipeline,uint32_t,uint64_t trace,uint32_t enabled,aclrtStream){
g_traceSeen=trace;
if(enabled){assert(pipeline==2 && y==work);enabledLaunches++;}return 0;}
extern "C" int mf310p_v3_device_raw(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,
uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint32_t,uint32_t,uint32_t,aclrtStream){return 0;}
extern "C" int mf310p_v3_device_ib(uint64_t,uint64_t result,uint32_t reps,uint32_t enabled,aclrtStream){
if(enabled)for(unsigned c=0;c<8;c++)reinterpret_cast<unsigned*>(result)[c*16]=reps;return 0;}
int main(){mf310p_v3_context* s=nullptr;assert(!mf310p_v32_create(0,"mock",4<<20,&s));
assert(!mf310p_v3_ib_probe(s,64,(void*)1));
unsigned slot0,gen0,slot1,gen1,slot2,gen2;uint64_t addr0,addr1,addr2;
assert(!mf310p_v32_acquire(s,4096,1024,0,(void*)1,&slot0,&gen0,&addr0));
assert(mf310p_v32_run(s,4096,8192,4096,1024,0,slot0,gen0,(void*)1)==-24);
assert(!mf310p_v32_prepare(s,slot0,gen0,1,(void*)1));
assert(!mf310p_v32_run(s,4096,8192,4096,1024,0,slot0,gen0,(void*)1));
assert(!mf310p_v32_acquire(s,4096,1024,0,(void*)1,&slot1,&gen1,&addr1));
assert(slot0!=slot1 && addr0!=addr1);
assert(mf310p_v32_run(s,4096,8192,4096,1024,0,slot0,gen0,(void*)1)==-24);
assert(mf310p_v32_acquire(s,4096,1024,0,(void*)1,&slot2,&gen2,&addr2)==-23);
assert(mf310p_v3_destroy(s)==-26 && destroyed==0);
assert(mf310p_v32_release(s,slot0,gen0+1)==-2);
assert(!mf310p_v32_release(s,slot0,gen0));
peerMask=0;assert(mf310p_v32_acquire(s,4096,1024,0,(void*)1,&slot2,&gen2,&addr2)==-23);
peerMask=3;assert(!mf310p_v32_acquire(s,4096,1024,0,(void*)1,&slot2,&gen2,&addr2));
assert(slot2==slot0 && gen2!=gen0);
assert(mf310p_v32_release(s,slot0,gen0)==-2);
assert(!mf310p_v32_prepare(s,slot2,gen2,1,(void*)1));
assert(mf310p_v32_run(s,addr2,8192,4096,1024,0,slot2,gen2,(void*)1)==-25);
assert(mf310p_v32_run(s,4096,8192,4096,1024,0,slot2,gen0,(void*)1)==-24);
assert(!mf310p_v32_release(s,slot2,gen2));assert(!mf310p_v32_release(s,slot1,gen1));
peerHeld=1;assert(mf310p_v3_destroy(s)==-26 && destroyed==0);
peerHeld=0;assert(!mf310p_v3_destroy(s));assert(destroyed==1 && enabledLaunches==1);
// Peer failed Storage wrapping: prepare must poison this rank before launch.
assert(!mf310p_v32_create(0,"mock",4<<20,&s));assert(!mf310p_v3_ib_probe(s,64,(void*)1));
assert(!mf310p_v32_acquire(s,4096,1024,0,(void*)1,&slot0,&gen0,&addr0));
peerReady=0;assert(mf310p_v32_prepare(s,slot0,gen0,1,(void*)1)==-27);
assert(mf310p_v32_run(s,4096,8192,4096,1024,0,slot0,gen0,(void*)1)==-27);
assert(mf310p_v3_destroy(s)==-27 && destroyed==1);
assert(!mf310p_v32_release(s,slot0,gen0));delete s;peerReady=1;
// Mismatched collective operations must fail, never free one rank's pool.
assert(!mf310p_v32_create(0,"mock",4<<20,&s));assert(!mf310p_v3_ib_probe(s,64,(void*)1));
peerOp=1;assert(mf310p_v3_destroy(s)==-22 && destroyed==1);delete s;
}
"""
    )
    binary = tmp_path / "host"
    subprocess.run(
        ["g++", "-std=c++17", "-Wall", "-Wextra", "-pthread", f"-I{tmp_path}", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, timeout=10)


def test_real_host_two_rank_prepare_and_close(tmp_path):
    # Reuse only the fake platform and device symbols; replace the mirrored
    # allgather with a real two-thread rendezvous around the actual Host code.
    test_real_host_slots_and_generation(tmp_path)
    header = tmp_path / "smem_shm.h"
    text = header.read_text()
    text = text[: text.index("inline int smem_shm_control_allgather")]
    text = text.replace("#include <cstring>", "#include <cstring>\n#include <atomic>")
    text = text.replace("int destroyed=0;", "std::atomic<int> destroyed{0};")
    text = text.replace(
        "inline void* smem_shm_create(int,int,int,uint64_t,int,int,void** p)",
        "inline void* smem_shm_create(int,int,int rank,uint64_t,int,int,void** p)",
    )
    text = text.replace(
        "return (void*)1;}\ninline void* smem_shm_sdma_get_workspace",
        "return (void*)(uintptr_t(rank)+1);}\ninline void* smem_shm_sdma_get_workspace",
    )
    header.write_text(
        text
        + r"""
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <vector>
std::mutex mx;std::condition_variable cv;unsigned sequence[2]={};
struct Round{std::vector<char> input[2];unsigned arrived=0,read=0;};
std::map<unsigned,std::shared_ptr<Round>> rounds;
inline int smem_shm_control_allgather(void* handle,const char* in,size_t n,char* out,size_t cap){
unsigned rank=uintptr_t(handle)-1;std::unique_lock<std::mutex> lock(mx);
unsigned seq=sequence[rank]++;auto& entry=rounds[seq];if(!entry)entry=std::make_shared<Round>();
auto round=entry;round->input[rank].assign(in,in+n);round->arrived++;cv.notify_all();
cv.wait(lock,[&]{return round->arrived==2;});
if(round->input[0].size()!=n||round->input[1].size()!=n||cap!=2*n)return -99;
memcpy(out,round->input[0].data(),n);memcpy(out+n,round->input[1].data(),n);
if(++round->read==2)rounds.erase(seq);return 0;}
"""
    )
    source = tmp_path / "host.cpp"
    text = source.read_text().split("int main(){")[0]
    text = text.replace("int enabledLaunches=0;", "std::atomic<int> enabledLaunches{0};")
    text = text.replace("uint64_t g_traceSeen=0;", "std::atomic<uint64_t> g_traceSeen{0};")
    source.write_text(
        text
        + r"""
#include <thread>
int main(){
for(unsigned scenario=0;scenario<3;++scenario){
 auto worker=[&](unsigned rank){mf310p_v3_context* s=nullptr;
 assert(!mf310p_v32_create(rank,"mock",4<<20,&s));assert(!mf310p_v3_ib_probe(s,64,(void*)1));
 unsigned slot,gen;uint64_t address;
 if(scenario==2){int rc=rank==0?mf310p_v3_destroy(s):
   mf310p_v32_acquire(s,4096,1024,0,(void*)1,&slot,&gen,&address);
   assert(rc==-22&&s->failed);delete s;return;}
 assert(!mf310p_v32_acquire(s,4096,1024,0,(void*)1,&slot,&gen,&address));
 if(scenario==0){assert(mf310p_v32_prepare(s,slot,gen,rank==0,(void*)1)==-27);
   assert(s->failed);assert(mf310p_v3_destroy(s)==-27);delete s;return;}
 assert(!mf310p_v32_prepare(s,slot,gen,1,(void*)1));
 assert(!mf310p_v32_run(s,4096,8192,4096,1024,0,slot,gen,(void*)1));
 if(rank==0)assert(!mf310p_v32_release(s,slot,gen));
 assert(mf310p_v3_destroy(s)==-26&&!s->failed); // asymmetric held output
 if(rank==1)assert(!mf310p_v32_release(s,slot,gen));
 assert(!mf310p_v3_destroy(s));};
 std::thread a(worker,0),b(worker,1);a.join();b.join();
}
assert(enabledLaunches==2&&destroyed==2&&rounds.empty());}
"""
    )
    binary = tmp_path / "two_rank_host"
    subprocess.run(["g++", "-std=c++17", "-pthread", f"-I{tmp_path}", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=15)


def test_real_host_merged_wave_fallback(tmp_path):
    # Interface Phase 1: speculative wave acquire (no allgather) plus the
    # single reconciling prepare; -29 falls back to the classic intersection.
    test_real_host_slots_and_generation(tmp_path)
    header = tmp_path / "smem_shm.h"
    text = header.read_text()
    text = text.replace(
        "unsigned peerMask=3,peerReady=1,peerHeld=0,peerOp=0;",
        "unsigned peerMask=3,peerReady=1,peerHeld=0,peerOp=0,peerSlotOv=0;",
    )
    text = text.replace("if(peerOp)p[0]=peerOp;}", "if(peerOp)p[0]=peerOp;if(peerSlotOv)p[7]=peerSlotOv;}")
    header.write_text(text)
    host = (Path(__file__).parents[1] / "src/op_host/mf310p_v3_host.cpp").resolve()
    source = tmp_path / "wave.cpp"
    source.write_text(
        f'#include "{host}"\n'
        + r"""
#include <cassert>
extern "C" int mf310p_a3_device_run(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,
uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint32_t,uint32_t,uint32_t,
uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,aclrtStream){return 0;}
int enabledLaunches=0;
uint64_t g_traceSeen=0;
extern "C" int mf310p_v3_device_run(uint64_t,uint64_t,uint64_t,uint64_t y,uint64_t work,uint64_t,
uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint32_t,uint32_t,uint32_t,uint32_t,
uint32_t,uint32_t pipeline,uint32_t,uint64_t trace,uint32_t enabled,aclrtStream){
g_traceSeen=trace;
if(enabled){assert(pipeline==2 && y==work);enabledLaunches++;}return 0;}
extern "C" int mf310p_v3_device_raw(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,
uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint32_t,uint32_t,uint32_t,aclrtStream){return 0;}
extern "C" int mf310p_v3_device_ib(uint64_t,uint64_t result,uint32_t reps,uint32_t enabled,aclrtStream){
if(enabled)for(unsigned c=0;c<8;c++)reinterpret_cast<unsigned*>(result)[c*16]=reps;return 0;}
int main(){mf310p_v3_context* s=nullptr;
assert(!mf310p_v32_create(0,"mock",4<<20,&s));assert(!mf310p_v3_ib_probe(s,64,(void*)1));
unsigned slotA,genA,slotB,genB,slotC,genC;uint64_t addrA,addrB,addrC;
// Happy path: speculative pick agrees (fake mirrors both peers); no acquire allgather.
assert(!mf310p_v32_wave_acquire(s,4096,1024,0,(void*)1,&slotA,&genA,&addrA));
assert(slotA==0);
assert(!mf310p_v32_prepare(s,slotA,genA,1,(void*)1));
assert(!mf310p_v32_run(s,4096,8192,4096,1024,0,slotA,genA,(void*)1));
assert(!mf310p_v32_release(s,slotA,genA));
// Divergent speculative picks: recoverable -29, session alive, lease releasable.
assert(!mf310p_v32_wave_acquire(s,4096,1024,0,(void*)1,&slotB,&genB,&addrB));
peerSlotOv=1-slotB;
assert(mf310p_v32_prepare(s,slotB,genB,1,(void*)1)==-29);
peerSlotOv=0;
assert(!s->failed);
assert(!mf310p_v32_release(s,slotB,genB));
// Classic intersection acquire still works after a burned speculation.
assert(!mf310p_v32_acquire(s,4096,1024,0,(void*)1,&slotC,&genC,&addrC));
assert(!mf310p_v32_prepare(s,slotC,genC,1,(void*)1));
assert(!mf310p_v32_run(s,4096,8192,4096,1024,0,slotC,genC,(void*)1));
assert(!mf310p_v32_release(s,slotC,genC));
// Local exhaustion is busy without any negotiation.
assert(!mf310p_v32_wave_acquire(s,4096,1024,0,(void*)1,&slotA,&genA,&addrA));
assert(!mf310p_v32_wave_acquire(s,4096,1024,0,(void*)1,&slotB,&genB,&addrB));
assert(slotA!=slotB);
assert(mf310p_v32_wave_acquire(s,4096,1024,0,(void*)1,&slotC,&genC,&addrC)==-23);
assert(!mf310p_v32_release(s,slotA,genA));
assert(!mf310p_v32_release(s,slotB,genB));
assert(!mf310p_v3_destroy(s));assert(destroyed==1 && enabledLaunches==2);
}
"""
    )
    binary = tmp_path / "wave_host"
    subprocess.run(
        ["g++", "-std=c++17", "-Wall", "-Wextra", "-pthread", f"-I{tmp_path}", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, timeout=10)


def test_real_host_trace_control_and_fetch(tmp_path):
    # L2 attribution plumbing: stamps off by default, enable gates the launch
    # argument, fetch validates size and copies the device buffer.
    test_real_host_slots_and_generation(tmp_path)
    host = (Path(__file__).parents[1] / "src/op_host/mf310p_v3_host.cpp").resolve()
    source = tmp_path / "trace.cpp"
    source.write_text(
        f'#include "{host}"\n'
        + r"""
#include <cassert>
extern "C" int mf310p_a3_device_run(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,
uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint32_t,uint32_t,uint32_t,
uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,aclrtStream){return 0;}
int enabledLaunches=0;
uint64_t g_traceSeen=0;
extern "C" int mf310p_v3_device_run(uint64_t,uint64_t,uint64_t,uint64_t y,uint64_t work,uint64_t,
uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint32_t,uint32_t,uint32_t,uint32_t,
uint32_t,uint32_t pipeline,uint32_t,uint64_t trace,uint32_t enabled,aclrtStream){
g_traceSeen=trace;
if(enabled){assert(pipeline==2 && y==work);enabledLaunches++;}return 0;}
extern "C" int mf310p_v3_device_raw(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,
uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint32_t,uint32_t,uint32_t,aclrtStream){return 0;}
extern "C" int mf310p_v3_device_ib(uint64_t,uint64_t result,uint32_t reps,uint32_t enabled,aclrtStream){
if(enabled)for(unsigned c=0;c<8;c++)reinterpret_cast<unsigned*>(result)[c*16]=reps;return 0;}
int main(){mf310p_v3_context* s=nullptr;
assert(!mf310p_v32_create(0,"mock",4<<20,&s));assert(!mf310p_v3_ib_probe(s,64,(void*)1));
unsigned slot,gen;uint64_t address;uint64_t buffer[512];
assert(mf310p_v3_trace_fetch(s,buffer,4096)==-5);   // not allocated yet
assert(mf310p_v3_trace_control(nullptr,1)==-1);
assert(!mf310p_v32_acquire(s,4096,1024,0,(void*)1,&slot,&gen,&address));
assert(!mf310p_v32_prepare(s,slot,gen,1,(void*)1));
assert(!mf310p_v32_run(s,4096,8192,4096,1024,0,slot,gen,(void*)1));
assert(g_traceSeen==0);                              // stamps off: null trace pointer
assert(!mf310p_v32_release(s,slot,gen));
assert(!mf310p_v3_trace_control(s,1));              // allocate + enable
assert(!mf310p_v3_trace_control(s,1));              // idempotent re-enable
assert(!mf310p_v32_wave_acquire(s,4096,1024,0,(void*)1,&slot,&gen,&address));
assert(!mf310p_v32_prepare(s,slot,gen,1,(void*)1));
assert(!mf310p_v32_run(s,4096,8192,4096,1024,0,slot,gen,(void*)1));
assert(g_traceSeen!=0);                              // stamps on: buffer pointer passed
assert(mf310p_v3_trace_fetch(s,buffer,4095)==-1);   // exact size required
assert(mf310p_v3_trace_fetch(s,buffer,4096)==0);
assert(!mf310p_v3_trace_control(s,0));              // disable keeps the buffer
assert(!mf310p_v32_release(s,slot,gen));
assert(!mf310p_v32_wave_acquire(s,4096,1024,0,(void*)1,&slot,&gen,&address));
assert(!mf310p_v32_prepare(s,slot,gen,1,(void*)1));
assert(!mf310p_v32_run(s,4096,8192,4096,1024,0,slot,gen,(void*)1));
assert(g_traceSeen==0);                              // disabled again: null pointer
assert(!mf310p_v32_release(s,slot,gen));
assert(!mf310p_v3_destroy(s));assert(destroyed==1 && enabledLaunches==3);
}
"""
    )
    binary = tmp_path / "trace_host"
    subprocess.run(
        ["g++", "-std=c++17", "-Wall", "-Wextra", "-pthread", f"-I{tmp_path}", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, timeout=10)
