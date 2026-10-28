# SPDX-License-Identifier: Apache-2.0
"""Execute the actual device Progress implementation with asynchronous FIFO SDMA."""
import subprocess
from pathlib import Path


def test_actual_progress_with_asymmetric_ranks(tmp_path):
    kernel = (Path(__file__).parents[1] / 'src/op_kernel/mf310p_v3_kernel.asc').read_text()
    progress = kernel[kernel.index('struct Progress {'):kernel.index('__aicore__ inline void FinishGroup')]
    harness = r'''
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <cassert>
#define __aicore__
constexpr unsigned READY=1, REDUCE=2, GATHER=3;
constexpr int SMEM_SHM_SDMA_MAIL_OK=0, SMEM_SHM_SDMA_OP_COPY=0, SMEM_SHM_SDMA_OP_ADD_FP16=2;
uint64_t DeviceTag(unsigned p,unsigned e,unsigned g,unsigned b) {return uint64_t(p)<<36|uint64_t(e)<<12|g<<6|b;}
void Check(bool ok, uint8_t*,uint64_t) {if(!ok) throw std::runtime_error("invalid mail");}
struct Mail {uint64_t dst,len,imm;unsigned status=0;};
struct Request {uint8_t* src;uint8_t* dst;uint64_t len,tag;int op;};
std::mutex mu; std::condition_variable cv;
std::deque<Request> requests[2];std::deque<Mail> arrivals[2];
std::atomic<bool> stopped{false};
bool smem_shm_sdma_signal_at(uint8_t* res,uint8_t* src,uint8_t* dst,uint64_t n,uint64_t tag,int op=0) {
 unsigned r=reinterpret_cast<uintptr_t>(res)-1;
 {std::lock_guard<std::mutex> l(mu);requests[r].push_back({src,dst,n,tag,op});}cv.notify_all();return true;
}
Mail smem_shm_sdma_wait_at(uint8_t* res) {
 unsigned r=reinterpret_cast<uintptr_t>(res)-1;std::unique_lock<std::mutex> l(mu);
 cv.wait(l,[&]{return !arrivals[r].empty();});auto m=arrivals[r].front();arrivals[r].pop_front();return m;
}
struct Transport {
 uint8_t *res,*work,*peer,*status,*ctrlSrc,*peerCtrl;
 uint64_t expectedWork,expectedCtrl;unsigned rank,epoch,packet;
 void Send(unsigned phase,unsigned g,uint64_t off,uint64_t bytes,int op) {
  for(uint64_t b=0;b<bytes;b+=packet)smem_shm_sdma_signal_at(res,work+off+b,peer+off+b,
    bytes-b<packet?bytes-b:packet,DeviceTag(phase,epoch,g,b/packet),op);
 }
};
'''
    runner = r'''
int main() {
 for(unsigned seed=0;seed<32;++seed)for(unsigned groups: {1,2,4,8,16})for(unsigned packets:{1,2,4}) {
  constexpr unsigned packet=32;unsigned bytes=2*packets*packet;
  std::array<uint64_t,2048> work[2];uint64_t ctrl[2]={};stopped=false;
  std::thread sdma([&]{std::mt19937 rng(seed);while(!stopped){
   std::unique_lock<std::mutex> l(mu);unsigned r=rng()%2;
   if(requests[r].empty()){l.unlock();std::this_thread::yield();continue;}
   auto q=requests[r].front();requests[r].pop_front();
   if(q.op==2)for(unsigned i=0;i<q.len/8;++i)
     reinterpret_cast<uint64_t*>(q.dst)[i]+=reinterpret_cast<uint64_t*>(q.src)[i];
   else std::memcpy(q.dst,q.src,q.len);
   arrivals[1-r].push_back({reinterpret_cast<uint64_t>(q.dst),q.len,q.tag,0});l.unlock();cv.notify_all();
  }});
  auto run=[&](unsigned r){
   Transport t{reinterpret_cast<uint8_t*>(uintptr_t(r+1)),reinterpret_cast<uint8_t*>(work[r].data()),
    reinterpret_cast<uint8_t*>(work[1-r].data()),nullptr,reinterpret_cast<uint8_t*>(&ctrl[r]),
    reinterpret_cast<uint8_t*>(&ctrl[1-r]),reinterpret_cast<uint64_t>(work[r].data()),reinterpret_cast<uint64_t>(&ctrl[r]),r,17,packet};
   Progress p{&t,bytes,groups};
   auto finish=[&](unsigned g){p.Complete(g);std::lock_guard<std::mutex> l(mu);
     for(unsigned j=0;j<bytes/8;++j)assert(work[r][g*bytes/8+j]==100+2*(g*100+j));};
   for(unsigned g=0;g<groups;++g){
    {std::lock_guard<std::mutex> l(mu);for(unsigned j=0;j<bytes/8;++j)work[r][g*bytes/8+j]=r*100+g*100+j;}
    if((g+r+seed)%3==0)std::this_thread::yield();p.Submit(g);if(g>=2)finish(g-2);
   }
   for(unsigned g=groups>2?groups-2:0;g<groups;++g)finish(g);
  };
  std::thread a(run,0),b(run,1);a.join();b.join();stopped=true;sdma.join();
  assert(requests[0].empty()&&requests[1].empty()&&arrivals[0].empty()&&arrivals[1].empty());
 }
}
'''
    runner = runner.replace('int main() {', r"""
int main() {
 { Transport t{reinterpret_cast<uint8_t*>(uintptr_t(1)),nullptr,nullptr,nullptr,nullptr,nullptr,4096,8192,0,17,32};
   Progress p{&t,128,1};p.published=1;
   auto reject=[&](Mail m){arrivals[0].push_back(m);bool failed=false;
    try{p.Pump();}catch(const std::runtime_error&){failed=true;}assert(failed);};
   reject({8192,8,DeviceTag(READY,16,0,0),0});
   reject({8193,8,DeviceTag(READY,17,0,0),0});
   arrivals[0].push_back({8192,8,DeviceTag(READY,17,0,0),0});p.Pump();
   reject({8192,8,DeviceTag(READY,17,0,0),0});
   reject({4096,32,DeviceTag(REDUCE,17,1,0),0});
   reject({4096,32,DeviceTag(REDUCE,17,0,2),0});
   arrivals[0].push_back({4096,32,DeviceTag(REDUCE,17,0,0),0});p.Pump();
   reject({4096,32,DeviceTag(REDUCE,17,0,0),0});
 }
""")
    source = tmp_path / 'progress.cpp' 
    source.write_text(harness + progress + runner)
    binary = tmp_path / 'progress'
    subprocess.run(['g++', '-std=c++17', '-O2', '-pthread', str(source), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=45)
