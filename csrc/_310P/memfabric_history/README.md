# V1 / V2 / V3 主线整合与历史快照（2026-10-30）

用户已授权暂停算法推进，将已实现源码、优化候选、测试脚本、实验及数据集中到
`dev/310p-memfabric-mm-ar`。本目录不是新的开发分支，也不改变Serving默认或任何设备安装。

## 主线当前源码

- V1：`../memfabric_mm_ar/`，保留现有实现与公共适配接口。
- V2：`../memfabric_mm_ar_v2/`，保留D4正式源与构建清单；合入可独立存在的G1C取证、
  B2/E1/E2/E3/E4候选、MM shadow、NZ路由、参数扫描、测试工具及实验数据。
- V3/A2/A3/A3-2/A3-tail及统一计时：`../memfabric_mm_ar_v3/`，以d59fec7d的最新实机归档为基底。
- 测量工具冲突采用large-m A1分支的NZ-aware U1版本；旧版也在快照中完整保留。

合入表示研发资产集中，不表示所有历史方案通过发布验收。NO-GO、失败现场、共享卡限制、
设备时钟不可用、未关闭的Gate E与Graph/Serving限制全部继续有效。本轮没有实机测试。

## 完整保全方式

`manifest.json`列出54条项目分支的冻结HEAD及快照位置；每个`snapshots/*.json.gz`
是无损压缩的原路径→Git blob SHA→文件模式映射，合计30987个项目文件版本引用。
`blobs/<SHA首位>/<SHA>`保存4862个独立文件内容，复用原Git对象，未经文本重编码、日志改写或二进制转换。
相同文件跨分支去重；相同路径的不同实现不会互相覆盖。每条源分支历史由整合提交的祖先保留。

快照覆盖所有310P/MemFabric/G1C/G1D命名的代码、脚本、配置、文档和实验数据，
另保留AGENTS、CMake/setup、envs、ModelSlim与Runner关键集成文件的版本。
仓库非项目部分不恢复为旧版；完整原仓上下文仍可从已保留的源提交还原。
所有内容来自我方10wvw01/vllm-ascend，不改外部HaonanWu1仓库或MemFabric发行分支。

## 浏览与恢复

查看`manifest.json`中的branch、head、snapshot字段；原文内容可直接查看对应`blobs/<SHA首位>/<SHA>`。
快照抽取器只写全新目录，逐文件验证Git blob SHA并恢复可执行位，不切换当前工作区：

```bash
python tools/310p_mm_ar/restore_snapshot.py --list
python tools/310p_mm_ar/restore_snapshot.py \
  --branch work/310p-mm-ar-e4-layered-pipeline-20261028 --out /tmp/e4-project-snapshot
```

如需包括仓库其他依赖文件的完整历史测试上下文，应直接使用源HEAD独立worktree：

```bash
git worktree add --detach /tmp/e4-full-repository 5cff1b2d79e28c588607dbacb5bdeec098087349
```

源HEAD均在本轮整合的历史中保留，不依赖未来远程分支仍存在；本轮不删除任何分支。

## 历史不兼容测试与废止计划

`legacy/`中的以下原件完整保留，要求匹配历史源码上下文，不并入当前自动收集目录：

- `test_g1c_forensic_source.py`、`test_g1c_kernel_isolation.py`：要求对应G1C专用CMake、Torch注册与源指令布局。
- `test_v2_c3_cleanlines_unroll4.py`：要求C3未采纳的生产源改写；当前默认仍是D4。
- `test_g1b_epoch_trace.py`、`test_g1c_trace_analyze.py`：分别依赖不同代的取证输出格式/阶段头定义。
- `V2_RUNNER_TRITON_FREE_MASTER_PLAN_20261025.md`：废止R2计划只作历史材料，不恢复成当前路线图。

不是删除或放宽这些测试；逐字节原件与完整匹配分支均保留，可恢复独立执行。
R2 Runner重建与C4一般模型优化也仅保留快照，不重新启用其模块或路线。

## 合并验证

V2全部当前测试（含perf/g3工具测试）、V3测试、项目治理及快照恢复工具：407项PASS，
另36个subtest PASS。新恢复工具Ruff通过；项目入口/增量台账检查及保护文件SHA核对另见审计记录。
旧源码、历史报告和实验数据按原SHA保留，不批量重格式化。全仓format.sh ci若环境缺少
pre-commit则未完成（本次实际未安装）；本轮不把CPU通过冒充CANN、设备或发布GO。


## 2026-10-30增量归档：A3管线Profiler证据

V3工作分支新增b34901ba（基于d59fec7d）已纳入主线：6份Rank CSV、索引、两份采集脚本、
归因摘要及实验README更新，合计11个文件。原始内容按Git blob逐项核对，未改写。
来源及文件校验见[增量记录](INCREMENTAL_ARCHIVES.json)。manifest中的V3入口指向新完整快照，
旧d59fec7d快照仍保留在previous_snapshots；上方30987/4862与INTEGRATION_AUDIT.json为首次整合统计。
归档保留原报告结论，不把管线计数升级为互斥阶段时长或新的性能GO。
本次只归档已存在证据，暂停中的L1统一框架/版本文档草稿未合入，也未执行新实机实验。
