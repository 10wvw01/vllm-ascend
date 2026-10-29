# SPDX-License-Identifier: Apache-2.0
"""Execute actual A3 dispatcher with delayed source reads and bounded queues."""

import ast
import subprocess
from pathlib import Path

HERE = Path(__file__).parent


def test_actual_a3_progress(tmp_path):
    kernel = (HERE.parent / "src/op_kernel/mf310p_v3_kernel.asc").read_text()
    tree = ast.parse((HERE / "test_v31.py").read_text())
    harness = next(
        n.value.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Constant)
        and any(isinstance(t, ast.Name) and t.id == "harness" for t in n.targets)
    )
    harness = harness.split("struct Transport {")[0]
    harness = harness.replace(
        "#define __aicore__",
        "#define TRACE_NOW() 0\n#define __gm__\nusing smem_shm_sdma_op_t=int;\n#define __aicore__",
    )
    harness = harness.replace("#include <cassert>", "#include <cassert>\n#include <vector>")
    harness += "\nvoid FlushLine(uint8_t*){}\n"
    transport = kernel[kernel.index("struct Transport {") : kernel.index("// One mailbox consumer")]
    progress = kernel[
        kernel.index("// A3 first READY") : kernel.index(
            "template <const MatmulApiStaticTiling& TILING, uint32_t MM_ROWS, bool MM, bool TRACE = false>"
        )
    ]
    runner = r"""
int main(){
for(unsigned scenario=0;scenario<3;++scenario)for(unsigned seed=0;seed<32;++seed)
for(unsigned groups:{1,2,4,8})for(unsigned packets:{1,2,4,64}){
 constexpr unsigned packet=32;unsigned bytes=packets*packet;
 std::vector<uint64_t> source[2],dest[2];uint64_t ctrl[2][8]={},incoming[2][8]={};std::atomic<unsigned> rejected{0};
 for(unsigned r=0;r<2;++r){source[r].resize(groups*bytes/8);dest[r].resize(groups*bytes/8);}
 stopped=false;
 std::thread dma([&]{std::mt19937 rng(seed);while(!stopped){
  std::unique_lock<std::mutex> lock(mu);unsigned r=rng()%2;
  if(requests[r].empty()){lock.unlock();std::this_thread::yield();continue;}
  auto q=requests[r].front();requests[r].pop_front();
  if(q.op==2)for(unsigned j=0;j<q.len/8;++j)
   reinterpret_cast<uint64_t*>(q.dst)[j]+=reinterpret_cast<uint64_t*>(q.src)[j];
  else std::memcpy(q.dst,q.src,q.len);
  arrivals[1-r].push_back({reinterpret_cast<uint64_t>(q.dst),q.len,q.tag,0});
  assert(arrivals[1-r].size()<1024);lock.unlock();cv.notify_all();
 }});
 auto run=[&](unsigned r){
 Transport t{reinterpret_cast<uint8_t*>(uintptr_t(r+1)),reinterpret_cast<uint8_t*>(source[r].data()),
  reinterpret_cast<uint8_t*>(dest[1-r].data()),nullptr,reinterpret_cast<uint8_t*>(ctrl[r]),
  reinterpret_cast<uint8_t*>(incoming[1-r]),reinterpret_cast<uint64_t>(dest[r].data()),
  reinterpret_cast<uint64_t>(incoming[r]),r,11,packet};
 A3PrepareHeader(t,4096,1024,scenario==2&&r==1?1024:512,1,0,scenario==1&&r==1?0:1);
 A3Progress p{&t,bytes,groups,reinterpret_cast<uint8_t*>(incoming[r])};
 try {
 if(scenario==1&&r==1){p.Submit(0);assert(false);} // failed rank never seeds or computes
 for(unsigned g=0;g<groups;++g){
  {std::lock_guard<std::mutex> lock(mu);for(unsigned j=0;j<bytes/8;++j){
   unsigned i=g*bytes/8+j;source[r][i]=dest[r][i]=r*100000+i;}}
  if((seed+g+r)%3==0)std::this_thread::yield();p.Submit(g);
 }
 p.Drain();
 }catch(const std::runtime_error&){assert(p.submitted==0);rejected++;return;}
 assert(scenario==0);
 {std::lock_guard<std::mutex> lock(mu);for(unsigned i=0;i<groups*bytes/8;++i){
 assert(dest[r][i]==100000+2*i);assert(source[r][i]==r*100000+i);}}
 };
 std::thread a(run,0),b(run,1);a.join();b.join();stopped=true;dma.join();
 assert(rejected==(scenario?2u:0u));
 assert(arrivals[0].empty()&&arrivals[1].empty()&&requests[0].empty()&&requests[1].empty());
}}
"""
    source = tmp_path / "a3.cpp"
    source.write_text("#define TRACE_NOW() 0\n" + harness + transport + progress + runner)
    binary = tmp_path / "a3"
    subprocess.run(["g++", "-std=c++17", "-O2", "-pthread", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=90)


def test_compute_and_transfer_ranges():
    for rows in (512, 1024, 2048, 4096):
        for compute in (512, 1024):
            for comm in (512, 1024, 2048):
                if rows % compute or rows % comm:
                    continue
                seen = []
                ready, published = 0, []
                for q in range(rows // compute):
                    for core in range(8):
                        seen.extend(
                            range(q * compute + core * (compute // 8), q * compute + (core + 1) * (compute // 8))
                        )
                    ready = (q + 1) * compute
                    while (len(published) + 1) * comm <= ready:
                        published.append(len(published))
                assert seen == list(range(rows))
                assert len(published) == rows // comm


def test_no_hot_collective_or_sync():
    host = (HERE.parent / "src/op_host/mf310p_v3_host.cpp").read_text()
    hot = host[host.index('extern "C" int mf310p_a3_reserve') : host.index('extern "C" int mf310p_v3_destroy')]
    assert "Synchronize" not in hot and "allgather" not in hot and "Exchange(" not in hot
    assert "NextEpoch(s)" not in hot and "aclrtMemsetAsync" not in hot
    assert "if (ready) s->leases[slot] = 3" in hot
    kernel = (HERE.parent / "src/op_kernel/mf310p_v3_kernel.asc").read_text()
    a3 = kernel[kernel.index("// A3 first READY") : kernel.index("__global__ __aicore__ void V3RawAddKernel")]
    assert "GATHER" not in a3 and "FlushLine(flag)" in a3


def test_actual_a3_mail_rejection(tmp_path):
    kernel = (HERE.parent / "src/op_kernel/mf310p_v3_kernel.asc").read_text()
    progress = kernel[
        kernel.index("// A3 first READY") : kernel.index(
            "template <const MatmulApiStaticTiling& TILING, uint32_t MM_ROWS, bool MM, bool TRACE = false>"
        )
    ]
    source = tmp_path / "reject.cpp"
    source.write_text(
        r"""
#define TRACE_NOW() 0
#include <cstdint>
#include <stdexcept>
#include <cassert>
#define __aicore__
#define __gm__
void FlushLine(uint8_t*){}
constexpr unsigned READY=1,REDUCE=2;
constexpr int SMEM_SHM_SDMA_MAIL_OK=0,SMEM_SHM_SDMA_OP_ADD_FP16=2;
uint64_t DeviceTag(unsigned p,unsigned e,unsigned g,unsigned b){return uint64_t(p)<<36|uint64_t(e)<<12|g<<6|b;}
void Check(bool ok,uint8_t*,uint64_t){if(!ok)throw std::runtime_error("bad mail");}
struct Mail{uint64_t dst,len,imm;int status=0;};Mail incoming;
Mail smem_shm_sdma_wait_at(uint8_t*){return incoming;}
bool smem_shm_sdma_signal_at(uint8_t*,uint8_t*,uint8_t*,uint64_t,uint64_t){return true;}
struct Transport{uint8_t *res=nullptr,*ctrlSrc=nullptr,*peerCtrl=nullptr,*status=nullptr;
uint64_t expectedWork=4096,expectedCtrl=8192;unsigned epoch=17,packet=512;
void Send(unsigned,unsigned,uint64_t,uint64_t,int){};};
"""
        + progress
        + r"""
int main(){Transport t;uint64_t local[8],peer[8];t.ctrlSrc=reinterpret_cast<uint8_t*>(local);
A3PrepareHeader(t,2048,1024,1024,1,0,1);for(unsigned i=0;i<8;i++)peer[i]=local[i];
A3Progress p{&t,1024,1,reinterpret_cast<uint8_t*>(peer)};p.published=1;
auto reject=[&](Mail m){incoming=m;bool caught=false;try{p.Pump();}catch(...){caught=true;}assert(caught);};
reject({8192,8,DeviceTag(READY,16,0,0),0});
reject({8192,8,DeviceTag(READY,17,1,0),0});
reject({8193,8,DeviceTag(READY,17,0,0),0});
reject({8192,8,DeviceTag(READY,17,0,0),0}); // old protocol's short first READY
for(unsigned field=0;field<8;++field){peer[field]++;reject({8192,64,DeviceTag(READY,17,0,0),0});peer[field]--;}
peer[7]=0;reject({8192,64,DeviceTag(READY,17,0,0),0});peer[7]=1;
incoming={8192,64,DeviceTag(READY,17,0,0),0};p.Pump();reject(incoming);
reject({4096,512,DeviceTag(3,17,0,0),0});
reject({4096,512,DeviceTag(REDUCE,17,0,2),0});
reject({4097,512,DeviceTag(REDUCE,17,0,0),0});
reject({4096,513,DeviceTag(REDUCE,17,0,0),0});
reject({4096,512,DeviceTag(REDUCE,17,0,0),1});
incoming={4096,512,DeviceTag(REDUCE,17,0,0),0};p.Pump();reject(incoming);
p.published=0;reject({4608,512,DeviceTag(REDUCE,17,0,1),0});}
"""
    )
    binary = tmp_path / "reject"
    subprocess.run(["g++", "-std=c++17", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_first_ready_replaces_start_without_initial_fanout():
    text = (HERE.parent / "src/op_kernel/mf310p_v3_kernel.asc").read_text()
    body = text[text.index("void A3Kernel(") : text.index("__global__ __aicore__ void V3RawAddKernel")]
    assert "A3Start" not in body and "kSubmitted" not in body and "DeviceTag(6," not in body
    assert body.index("if (!admit)") < body.index("REGIST_MATMUL_OBJ")
    assert "g == 0 ? 64 : 8" in text
    assert "kA3Protocol = 3302" in (HERE.parent / "src/op_host/mf310p_v3_host.cpp").read_text()


def test_a3_tail_single_quiet_per_slot_source_and_amortized_invalidation():
    kernel = (HERE.parent / "src/op_kernel/mf310p_v3_kernel.asc").read_text()
    a3 = kernel[kernel.index("// A3 first READY") : kernel.index("__global__ __aicore__ void V3RawAddKernel")]
    # Every A3 tail keeps Drain -> quiet -> DONE but drops the final quiet:
    # the DONE exchange proves the peer drained our outbound ADDs, and the
    # next call seeds the other S slot anyway.
    assert "status, 106" not in a3
    assert a3.count("status, 105") == 3
    # Amortized invalidation on both wait families: bounded staleness, never unbounded.
    assert a3.count("(spins & 7u) == 0) FlushLine") == 2
    # The inbound READY header tracks the ctrl arena, not the (now per-slot) S arena.
    assert a3.count("ctrlSrc + 64") == 2
    host = (HERE.parent / "src/op_host/mf310p_v3_host.cpp").read_text()
    assert "kA3CtrlOff = 4 * kArenaBytes" in host
    assert "s->work + kA3SourceOff + slot * kArenaBytes" in host
    assert "kPoolBytes = 5 * kArenaBytes" in (HERE.parent / "src/mf310p_v3_protocol.h").read_text()
