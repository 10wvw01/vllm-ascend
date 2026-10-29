# SPDX-License-Identifier: Apache-2.0
"""Run the actual Plan A transport and scheduler against delayed FIFO DMA."""

import ast
import subprocess
from pathlib import Path

HERE = Path(__file__).parent


def test_plan_a_actual_cpp_schedule(tmp_path):
    kernel = (HERE.parent / "src/op_kernel/mf310p_v3_kernel.asc").read_text()
    # Reuse only the asynchronous fake DMA engine, not V3.1 scheduling code.
    tree = ast.parse((HERE / "test_v31.py").read_text())
    values = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("harness", "runner"):
                    values[target.id] = node.value.value
    harness = values["harness"].split("struct Transport {")[0]
    # The extracted OrderedProgress block stamps L2 segments via TRACE_NOW and
    # mirrors mfv3::kTraceSubBase from mf310p_v3_protocol.h.
    harness = harness.replace(
        "#define __aicore__",
        "#define __gm__\nusing smem_shm_sdma_op_t = int;\n"
        "namespace mfv3{constexpr uint32_t kTraceSubBase=272;}\n#define TRACE_NOW() 0\n#define __aicore__",
    )
    transport = kernel[kernel.index("struct Transport {") : kernel.index("// One mailbox consumer")]
    progress = kernel[
        kernel.index("struct OrderedProgress {") : kernel.index(
            "template <const MatmulApiStaticTiling& TILING, uint32_t MM_ROWS", kernel.index("struct OrderedProgress {")
        )
    ]
    runner = values["runner"]
    runner = runner.replace("Progress p{", "OrderedProgress p{").replace(
        "p.Submit(g);if(g>=2)finish(g-2);", "p.Step(g);"
    )
    runner = runner.replace("p.Complete(g);", "")
    runner = runner.replace(
        "for(unsigned g=groups>2?groups-2:0;g<groups;++g)finish(g);",
        "p.Drain();for(unsigned g=0;g<groups;++g)finish(g);",
    )
    # 16 groups remains a transport stress case; actual Plan A supports <=8.
    source = tmp_path / "ordered.cpp"
    source.write_text(harness + transport + progress + runner)
    binary = tmp_path / "ordered"
    subprocess.run(["g++", "-std=c++17", "-O2", "-pthread", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=90)


def test_transport_rejects_bad_mail(tmp_path):
    kernel = (HERE.parent / "src/op_kernel/mf310p_v3_kernel.asc").read_text()
    transport = kernel[kernel.index("struct Transport {") : kernel.index("// One mailbox consumer")]
    source = tmp_path / "bad.cpp"
    source.write_text(
        r"""
#include <cstdint>
#include <stdexcept>
#include <cassert>
#define __aicore__
#define __gm__
using smem_shm_sdma_op_t=int;
constexpr int SMEM_SHM_SDMA_MAIL_OK=0;
uint64_t DeviceTag(unsigned p,unsigned e,unsigned g,unsigned b){return uint64_t(p)<<36|uint64_t(e)<<12|g<<6|b;}
void Check(bool ok,uint8_t*,uint64_t){if(!ok)throw std::runtime_error("bad mail");}
struct Mail{uint64_t dst,len,imm;int status;}; Mail incoming;
Mail smem_shm_sdma_wait_at(uint8_t*){return incoming;}
bool smem_shm_sdma_signal_at(uint8_t*,uint8_t*,uint8_t*,uint64_t,uint64_t,int=0){return true;}
"""
        + transport
        + r"""
int main(){Transport t{nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,4096,8192,0,17,512};
Mail good{4096,512,DeviceTag(2,17,0,0),0};incoming=good;t.Wait(2,0,0,4096,512);
for(int k=0;k<7;++k){incoming=good;
switch(k){case 0:incoming.dst++;break;case 1:incoming.len++;break;
case 2:incoming.imm=DeviceTag(2,16,0,0);break;case 3:incoming.imm=DeviceTag(3,17,0,0);break;
case 4:incoming.imm=DeviceTag(2,17,1,0);break;case 5:incoming.imm=DeviceTag(2,17,0,1);break;
case 6:incoming.status=1;break;}
bool caught=false;try{t.Wait(2,0,0,4096,512);}catch(...){caught=true;}assert(caught);}}
"""
    )
    binary = tmp_path / "bad"
    subprocess.run(["g++", "-std=c++17", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_assessor_requires_plan_a_storage_gate():
    import importlib.util

    spec = importlib.util.spec_from_file_location("v3_test_helpers", HERE / "test_v3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a, b = mod.sample_pair()
    for r in (a, b):
        r["configuration"]["path"] = "V32_PLAN_A_POOL_RETURN"
    try:
        mod.history.assess(a, b)
    except ValueError:
        pass
    else:
        raise AssertionError("Plan A must require Storage lifetime gate")
    for r in (a, b):
        r["gates"]["output_lifetime"] = "PASS"
    assert not mod.history.assess(a, b)["formal_performance_go"]


def test_historical_v31_is_recomputed_from_fixed_session():
    import importlib.util

    spec = importlib.util.spec_from_file_location("v32_history", HERE / "historical.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = [x for x in mod.baselines() if x["batch"] == "V3.1_default_r2"]
    assert len(rows) == 6
    best = {m: min(x["p50_us"] for x in rows if x["M"] == m) for m in (2048, 4096)}
    assert abs(best[2048] - 1656.0) < 0.2
    assert abs(best[4096] - 2516.1) < 0.2
