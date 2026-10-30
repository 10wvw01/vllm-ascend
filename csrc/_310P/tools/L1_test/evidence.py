# SPDX-License-Identifier: Apache-2.0
"""Recompute source-linked historical Host-phase ranges, never invent kernel phase clocks."""
import json
from pathlib import Path
from .core import ROOT, sha


def write_evidence():
    base=ROOT/'csrc/_310P/memfabric_mm_ar_v3/experiments/a3-unified-shared-card-20261029-r1'
    paths=sorted((base/'campaign-r4-paired').glob('a3*.paired.json'))
    rows=[]
    sources=[]
    def span(values):
        return f'{min(values):.1f}–{max(values):.1f}'
    for path in paths:
        data=json.loads(path.read_text())
        conf=data['configuration']
        arm=conf['arm']
        for m in conf['M']:
            for comm in conf['comm_rows']:
                cases=[c for c in data['cases'] if c['M']==m and c['comm_rows']==comm and c['stage'].startswith('off')]
                host=[x for x in data['attribution'] if x['M']==m and x['comm_rows']==comm and x['stage']=='host']
                values=[span([x['host_phase_p50_us'][key] for x in host])
                        for key in ('validate_reap','reserve','wrap','enqueue','input_event')]
                wall=span([x['wall_us']['p50'] for x in cases])
                rows.append('| '+' | '.join([arm,str(conf['compute_rows']),str(m),str(comm),wall,*values])+' |')
        sources.append(f'- [{path.name}](../../memfabric_mm_ar_v3/experiments/a3-unified-shared-card-20261029-r1/campaign-r4-paired/{path.name})：SHA256 `{sha(path)}`。')
    text='''# 统一L1性能证据：已测数值、瓶颈候选与缺口

2026-10-30。这里只读取历史原件；新框架CANN/NPU测量NOT_RUN。统一框架的新算术均值不能拿历史p50冒充。

## 当前能回答到哪一层

| 范围 | 已有证据 | 可下的判断 | 不能下的判断 |
|---|---|---|---|
| V1系列 | 原设计、原分支快照和旧测试 | 批流水/小M分支/生命周期的依赖关系可审查 | 新统一合同下每步µs/占比尚UNKNOWN |
| V2 T0/D4及B/E系列 | 历史完整链、MM/AR隔离与Profiler实验 | 可复用各自原合同的完整耗时和失败限制 | 不把正式−诊断差额当作一个步骤的真实耗时 |
| A2/A3系列Host准备 | 五段Host记录，见下表 | validate/reap、reserve、wrap、enqueue、事件提交存在可测成本 | 不能将这些段与设备耗时直接相加凑100% |
| A3设备MM/seed/wait/ADD/drain | 旧打点时钟不可用；新增MAC/Scalar/MTE计数 | 多管线持续忙；有就绪、提交与访存相关候选瓶颈 | Scalar=wait、MTE2=通信、MAC=全部MM墙钟均不成立 |
| Stock | 同批NZ+HCCL完整链及真实路由 | 是后续完整对照的唯一主基线 | MM-only、旧row-major数字不能替代完整Stock |

所有35个设计的步骤表见各工程VERSION_INDEX；UNKNOWN表示缺失本版分段独占墙钟证据，不是设备步骤为0。
已有历史总性能结论仍见[主性能基线](PERFORMANCE_BASELINES.md)，不混算不同卡、共享/独占或不同计时合同。

## A3系列现有Host阶段数值

单位µs。每个单元格是**各Rank/两轮源p50的最小–最大范围**；完整Wall为无打点前后两轮的p50范围。
Host段来自仅Host打点样本，Wall来自无打点样本，二者不相加、不相减推算设备阶段。
这些是共享卡历史场次，不能签稳定正式收益。这里不把p50改称平均耗时。

| 版本 | compute行 | M | comm行 | 完整Wall p50范围 | validate/reap | reserve | wrap | enqueue | input_event |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

## 对瓶颈的审慎判断

1. Host固定开销客观存在，尤其小M时可能占比较大；上表支持定位具体调用段，但Host下发空隙会进入Event，不能用Wall−Event代替Host实测。
2. 原A3的Core0在计算波边界推进通信；A3-2计算核少一核且按7核取整；tail会面对末尾自身座位与连续范围空洞。这些依赖可能暴露等待。原A3对A3-2在同批配置实际9/12更快，不能只凭“专职通信”推断一定更快。
3. 新硬件Profiler里的MAC、Scalar、MTE1/MTE2、Vector为重叠管线指标；高Scalar可能含轮询、地址计算、同步与提交，高MTE可能含MM数据搬运和缓存操作。单看计数尚不能证明“等待完全解释了差距”。
4. 同名CSV被各comm配置反复写入，归档实际只有6份最终CSV；索引列出的18次采集不是18份完整原件。保留原索引，不能据此宣称每个配置的原CSV都已保全。新框架按形状/场景独立目录避免覆盖。
5. 若要得到核内各步骤准确独占µs，仍需要310P可用且经过校准的时间源/硬件时序证据。当前公共框架会报告缺失能力，同时先提供可靠的完整链、Host阶段和Kernel/管线采集，不再增加无效的设备时钟方案。

## 下一批统一采集的交付物

无打点完整算术均值与全部原始样本；每个M真实NZ/fused路由；ready和chain；独占/共享标签与漂移；Host/Profiler前后控制和扰动；各版本支持/缺失阶段列表。故障或错算立即STOP，不继续下一臂。发布GO始终单独验收。

来源及校验：

'''+ '\n'.join(sources)+'''

[原硬件Profiler归档](../../memfabric_mm_ar_v3/experiments/a3-unified-shared-card-20261029-r1/pipe-profiler/)仅作为原始证据；上面没有修改原报告或其CSV。
'''
    target=ROOT/'csrc/_310P/memfabric_mm_ar_v2/docs/L1_PERFORMANCE_EVIDENCE.md'
    target.write_text(text)
    print('historical Host-phase rows:',len(rows))


if __name__=='__main__':
    write_evidence()
