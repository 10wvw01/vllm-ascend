# SPDX-License-Identifier: Apache-2.0
"""Compile real A3 Host, verify hot-path calls and fixed-seat failure."""

import importlib.util
import subprocess
from pathlib import Path


def test_actual_a3_host_no_hot_sync_or_collective(tmp_path):
    path = Path(__file__).with_name("test_v32_host.py")
    spec = importlib.util.spec_from_file_location("a3_host_platform", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.test_real_host_slots_and_generation(tmp_path)
    header = tmp_path / "acl/acl_rt.h"
    text = header.read_text().replace(
        "inline int aclrtSynchronizeDevice(){return 0;}",
        "int syncCalls=0;\ninline int aclrtSynchronizeDevice(){syncCalls++;return 0;}",
    )
    text = text.replace(
        "inline int aclrtSynchronizeStream(void*){return 0;}",
        "inline int aclrtSynchronizeStream(void*){syncCalls++;return 0;}",
    )
    text = text.replace(
        "inline int aclrtMemsetAsync(void* p,size_t a,int v,size_t n,void*){",
        "int asyncClears=0;\ninline int aclrtMemsetAsync(void* p,size_t a,int v,size_t n,void*){asyncClears++;",
    )
    header.write_text(text)
    header = tmp_path / "smem_shm.h"
    text = header.read_text().replace("unsigned peerMask=3", "int controlCalls=0;\nunsigned peerMask=3")
    text = text.replace("memcpy(out,in,n);", "controlCalls++;memcpy(out,in,n);")
    header.write_text(text)
    source = tmp_path / "host.cpp"
    text = source.read_text().split("int main(){")[0]
    a = text.index('extern "C" int mf310p_a3_device_run')
    b = text.index("int enabledLaunches", a)
    text = (
        text[:a]
        + r"""
int a3Launches=0,lastAdmit=-1;
extern "C" int mf310p_a3_device_run(uint64_t,uint64_t,uint64_t,uint64_t y,uint64_t s,uint64_t,
uint64_t,uint64_t,uint64_t ctrl,uint64_t,uint64_t,uint64_t,uint32_t,uint32_t,uint32_t,
uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t admit,uint32_t enabled,aclrtStream){
if(enabled){a3Launches++;lastAdmit=admit;assert(s==0x10000000+2*mfv3::kArenaBytes);
assert(ctrl==s+mfv3::kArenaBytes);assert(y==0x10000000||y==0x10000000+mfv3::kArenaBytes);}return 0;}
"""
        + text[b:]
    )
    source.write_text(
        text
        + r"""
int main(){mf310p_v3_context* s=nullptr;assert(!mf310p_a3_create(0,"mock",4<<20,&s));
assert(!mf310p_v3_ib_probe(s,64,(void*)1));assert(!mf310p_a3_warm(s,(void*)1));
int sync=syncCalls,ctl=controlCalls,clears=asyncClears;unsigned slot,gen;uint64_t addr;
for(unsigned i=0;i<2;i++){
assert(!mf310p_a3_reserve(s,4096,1024,512,0,(void*)1,&slot,&gen,&addr));
assert(slot==i&&gen==i+1);
assert(!mf310p_a3_run(s,4096,8192,4096,1024,512,0,slot,gen,1,(void*)1));
assert(lastAdmit==1);}
assert(mf310p_a3_reserve(s,4096,1024,512,0,(void*)1,&slot,&gen,&addr)==-23);
assert(slot==0&&gen==3);
assert(!mf310p_a3_run(s,4096,8192,4096,1024,512,0,slot,gen,0,(void*)1));
assert(lastAdmit==0&&s->failed&&a3Launches==3);
assert(sync==syncCalls&&ctl==controlCalls&&clears==asyncClears);
assert(mf310p_a3_reserve(s,4096,1024,512,0,(void*)1,&slot,&gen,&addr)==-27);delete s;}
"""
    )
    binary = tmp_path / "a3_host"
    subprocess.run(["g++", "-std=c++17", "-pthread", f"-I{tmp_path}", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=10)
