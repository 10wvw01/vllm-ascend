# SPDX-License-Identifier: Apache-2.0
"""Generate reproducible VN-M flow documents and a source-backed family index."""
import gzip
import json
from pathlib import Path

from .core import ROOT, catalog


def build():
    history=ROOT/'csrc/_310P/memfabric_history'
    manifest=json.loads((history/'manifest.json').read_text())
    evidence='../../../memfabric_mm_ar_v2/docs/L1_PERFORMANCE_EVIDENCE.md'
    indexes={family:[] for family in ('V1','V2','V3')}
    for entry in catalog().values():
        vid=entry['id']
        family=vid.split('-')[0]
        target=ROOT/entry['document']
        target.parent.mkdir(parents=True,exist_ok=True)
        source=ROOT/entry['source']
        if source.exists():
            source_link='../../../../../'+entry['source']
            provenance=f'当前主线来源：[源码/既有设计]({source_link})。历史版本的行为须与冻结源码和二进制同时核对。'
        else:
            match=None
            for branch in manifest['branches']:
                if entry.get('snapshot_branch') and branch['branch']!=entry['snapshot_branch']:
                    continue
                snap=json.loads(gzip.decompress((history/branch['snapshot']).read_bytes()))
                if entry['source'] in snap['files']:
                    match=(branch,snap['files'][entry['source']]['sha'])
                    break
            if match is None:
                raise ValueError('unresolved source '+vid)
            branch,sha=match
            provenance=f'来源：`{branch["branch"]}` @ `{branch["head"]}`；原路径`{entry["source"]}`。'
            provenance+=f' [原始blob](../../../memfabric_history/blobs/{sha[0]}/{sha})；通过主线快照恢复，不创建替代实现。'
        flow=entry['flow']
        diagram=['flowchart TD','  I["就绪输入 X 与 NZ W"]']
        previous='I'
        for n,step in enumerate(flow):
            diagram.append(f'  {previous} --> S{n}["{step}"]')
            previous=f'S{n}'
        diagram.append(f'  {previous} --> O["输出完成与下游读取"]')
        if vid in ('V3-5','V3-6','V3-7'):
            producer='Core1～7按步长7独立生产' if vid=='V3-6' else '各核按本版座位顺序生产'
            scheduler={'V3-5':'Core0兼计算；计算波边界推进通信',
                       'V3-6':'Core0专职通信；不执行MM',
                       'V3-7':'Core0持末尾座位；补齐自有行并穿插通信'}[vid]
            diagram=[
                'flowchart TD',
                '  I["就绪ND输入与NZ权重"] --> P["准备固定S与独立Y输出租约"]',
                f'  P --> C["{producer}：NZ MM、seed与可见性"]',
                '  C --> F["发布本座位epoch；其他核继续生产"]',
                '  F --> B{"仍有自有座位？"}',
                '  B -->|是| C',
                '  B -->|否| M["本核生产完成"]',
                f'  P --> H["{scheduler}"]',
                '  H --> R{"下一连续通信范围齐全？"}',
                '  F -.-> R',
                '  R -->|否| W["依版本补算自有行、泵收件或等待"]',
                '  W --> R',
                '  R -->|是| V["首READY校验；按通信批提交Direct ADD"]',
                '  V --> A["SDMA异步搬运与ADD；可与后续MM重叠"]',
                '  V --> N{"仍有通信批？"}',
                '  N -->|是| H',
                '  N -->|否| D["统一排空、quiet、DONE与核间会合"]',
                '  A --> D',
                '  M --> D',
                '  D --> O["独立Y完整就绪；下游读取"]',
            ]
        rows='\n'.join(f'| {i+1} | {step} | UNKNOWN | 待采集；不凭名称判定瓶颈 |' for i,step in enumerate(flow))
        if entry.get('session_api'):
            usage=f'统一L1的`runtimes.{vid}`使用`api=python_session_v1`；冻结匹配的Python会话、Host/Device/SDK和Storage扩展，复用本家族薄适配器，无需复制测量循环。详见[适配器合同](../../../tools/L1_test/ADAPTER_GUIDE.md)。'
            if entry.get('historical_source_hint'):
                usage+=f' 历史恢复线索：`{entry["historical_source_hint"]}`，必须核对实际工件，不用当前SO替代。'
        elif entry['runtime']=='external':
            usage='该历史/诊断版本没有用当前SO替代执行。恢复匹配快照，按共同适配器协议提供冻结的`factory_file`；框架核验返回的`adapter.version`。'
        else:
            usage=f'在运行manifest的`runtimes.{vid}`填写本版冻结工件，运行统一入口；不变更产品Serving路由。'
        if entry.get('selected_api'):
            usage+=f' 本轮性能选中版本可填写`api={entry["selected_api"]}`，改用固定M={entry["selected_M"]}的公共C接口；详见[正式入口与代码梳理](../SELECTED_API.md)。旧私有ABI仍用于复现历史，不能把旧SO成绩赋给新构建。'
        if entry.get('category'):
            usage+=' 本版为'+entry['category']+'，Eager完整MM+AR计划主动拒绝把它计作性能候选；原始专用脚本保留。'
        example_m=str(entry['selected_M']) if entry.get('selected_api') else '2048 4096'
        window=' --batch-size 1' if entry['runtime'] in ('v1','external') else ''
        example='' if entry.get('category') else f'''```bash
python csrc/_310P/tools/L1_test/l1_bench.py plan --versions {vid} --M {example_m} \\
  --warmup 20 --samples 100 --rounds 2{window} --out /tmp/{vid}-plan.json
```

上述M仅为统一示例；本版实际合法M由匹配实现与`runtime.supported_M`约束，小M路径必须改为其已准入形状。
'''
        text=f'''# {vid}｜{entry['legacy']}

统一编号中N为实现代际，M为设计序号；不是矩阵行数。旧名称保留作检索别名。
状态：{entry['status']}。新采集框架尚未完成CANN/NPU验证。

## 来源和设计边界

{provenance}

{entry['notes']}

## 算子使用、输入与输出

{usage}

- 输入：{entry['input']}。
- 输出：{entry['output']}。
- 数学语义：`P_rank=FP16(linear(X_rank,W_NZ)); Y=FP16(P_0+P_1)`。这里只规定TP2局部FP16 MM后归约；不替代原量化包装层的输入合同。
- 统一计时：{entry['timing']}。
- 调用层：Stock的公开Tensor输入/输出边界；内建V2/V3由算子源码中的`torch.ops.memfabric310p.mm_ar`管理输出获取、包装和异步输入保活，L1不代办。详见[共同测量合同](../../../tools/L1_test/MEASUREMENT_V4.md)。
- 权重NZ物化、持久会话初始化和首写保护属于冷态准备并单列；每次调用必需的动作仍计入热态。
- 临时/池输出的租约由各实现保留到下游消费结束；不能提前释放或将同一底层地址当作两个独立输出。

{example}
## 整体流程

```mermaid
{chr(10).join(diagram)}
```

步骤顺序表示依赖；标为交叠的设计可以并行执行，图中箭头不表示所有步骤全程串行。

## 每步耗时与瓶颈判断

| 步骤 | 工作 | 本版内部独占耗时 | 判定依据 |
|---|---|---|---|
{rows}

这里的UNKNOWN专指新统一合同下的核内分段独占墙钟耗时；不是否认历史完整算子测量。
历史实测与当前可复算的Host阶段见[性能证据总表]({evidence if family!='V2' else '../L1_PERFORMANCE_EVIDENCE.md'})。
统一输出完整Wall/Event、Host可测阶段及硬件Profiler原始CSV；不支持的设备阶段明确缺失，不能写0。
Host各段在同一调用内可比较；其间可能与设备交叠，不能和设备Kernel耗时求和凑成100%。

## 复核与待验证项

- 实际NZ MM或本版Kernel路由、双Rank输入/输出、相同N及下游消费必须逐项通过。
- 同路径不同源码/二进制的历史实现须单独冻结；`historical_identity_verified`记录人工审计，不代表自动证明源码到SO可复现。
- 优先检查本版流程中的缓存可见性、就绪空洞、阻塞wait、提交/环背压、局部Add、输出整理与尾部排空；这些是候选瓶颈，不是已测占比。
- 不移除IB、cache clean、credit、quiet/DONE或首写保护来获取虚假性能。
- 本次文档/CPU框架工作不改变Stock默认，不构成实机或发布GO。
'''
        target.write_text(text)
        execution=entry.get('session_api',entry['runtime'])
        if entry.get('selected_api'):
            execution+=' / '+entry['selected_api']
        indexes[family].append(f'| [{vid}](versions/{vid}.md) | {entry["legacy"]} | {execution} | {entry.get("category","完整候选/历史实现")} |')
    for family,rows in indexes.items():
        path=ROOT/next(v['document'] for v in catalog().values() if v['id'].startswith(family+'-'))
        (path.parent.parent/'VERSION_INDEX.md').write_text(f'''# {family} 设计版本索引

统一采用VN-M编号，保留旧名称；同一实现的不同参数点不冒充新架构。
C2选型/D2设计/D3编码/D4实测属于同一个AddTile2048实现，统一V2-1；C3指2026-10-24的CleanLines unroll4候选。
Graph、MM shadow及G1C/G1D诊断单独列明，不能混入Eager完整MM+AR收益统计。

| 编号 | 原名称 | 执行适配 | 类别 |
|---|---|---|---|
{chr(10).join(rows)}

框架合同见仓库`csrc/_310P/tools/L1_test/README.md`。旧设计、实验数据和失败记录保持原文；这些文档补充统一导航，不覆盖原记录。
''')
    print(f'generated {sum(map(len,indexes.values()))} source-resolved version documents and 3 indexes')


if __name__=='__main__':
    build()
