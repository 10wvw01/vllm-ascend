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
unsigned peerMask=3;int destroyed=0;
inline int smem_shm_destroy(void*,int){destroyed++;return 0;}
inline int smem_shm_control_allgather(void*,const char* in,size_t n,char* out,size_t){
memcpy(out,in,n);memcpy(out+n,in,n);
if(n==28)reinterpret_cast<unsigned*>(out+n)[0]=peerMask;
return 0;}
""")
    host = (Path(__file__).parents[1] / "src/op_host/mf310p_v3_host.cpp").resolve()
    source = tmp_path / "host.cpp"
    source.write_text(
        f'#include "{host}"\n'
        + r"""
#include <cassert>
int enabledLaunches=0;
extern "C" int mf310p_v3_device_run(uint64_t,uint64_t,uint64_t,uint64_t y,uint64_t work,uint64_t,
uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint32_t,uint32_t,uint32_t,uint32_t,
uint32_t,uint32_t pipeline,uint32_t,uint32_t enabled,aclrtStream){
if(enabled){assert(pipeline==2 && y==work);enabledLaunches++;}return 0;}
extern "C" int mf310p_v3_device_raw(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,
uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint32_t,uint32_t,uint32_t,aclrtStream){return 0;}
extern "C" int mf310p_v3_device_ib(uint64_t,uint64_t result,uint32_t reps,uint32_t enabled,aclrtStream){
if(enabled)for(unsigned c=0;c<8;c++)reinterpret_cast<unsigned*>(result)[c*16]=reps;return 0;}
int main(){mf310p_v3_context* s=nullptr;assert(!mf310p_v32_create(0,"mock",4<<20,&s));
assert(!mf310p_v3_ib_probe(s,64,(void*)1));
unsigned slot0,gen0,slot1,gen1,slot2,gen2;uint64_t addr0,addr1,addr2;
assert(!mf310p_v32_acquire(s,4096,1024,0,(void*)1,&slot0,&gen0,&addr0));
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
assert(mf310p_v32_run(s,addr2,8192,4096,1024,0,slot2,gen2,(void*)1)==-25);
assert(mf310p_v32_run(s,4096,8192,4096,1024,0,slot2,gen0,(void*)1)==-24);
assert(!mf310p_v32_release(s,slot2,gen2));assert(!mf310p_v32_release(s,slot1,gen1));
assert(!mf310p_v3_destroy(s));assert(destroyed==1 && enabledLaunches==1);}
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
