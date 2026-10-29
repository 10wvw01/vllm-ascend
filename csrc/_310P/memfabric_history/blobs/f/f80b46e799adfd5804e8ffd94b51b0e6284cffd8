# 310P Stock+DFlash 启动失败｜上游源码与依赖溯源（第一轮）

> 2026-10-25；**静态源码审查 / GitHub 比对结论**，不是历史服务器日志重放。项目主线 `10wvw01/vllm-ascend:dev/310p-memfabric-mm-ar`。上游只读基线 `HaonanWu1/vllm-ascend:dev_v24/feat-310p-dflash@1989b04c04e6ec5cb8c2e394bc8ebd107422196a`。

## 1. 故障问题和不能跳过的前提

历史有过 Stock+DFlash 投机推理启动失败的反馈，疑似依赖缺失；**本轮没有找到对应原始 traceback、pip freeze、两处源代码真实导入路径或复现的 Run ID**。因此不能把某个具体包、某段补丁或Graph宣判为已知根因。历史另已证实：**V2 Graph未通过验收；旧Graph A/B发生实际Stock-vs-Stock伪命中**，不能与上游 Stock+DFlash 失败混为一个故障。

固定本次业务：`--speculative-config '{"method":"dflash","model":"/home/models/Qwen3.6-35B-A3B-DFlash","num_speculative_tokens":7}'`；目标模型 `/home/models/Qwen3.6-35B-A3B-w8a8`；单卡双Die、TP=2。**第一次只验证Stock+DFlash Eager**，不加载MemFabric服务池，也不讨论V2 Graph是否通过。

## 2. 上游和自有主线对比

Git比较显示：`1989b04c04e6` 与自有主线已经 **diverged**，上游在共同祖先之后独有33提交；我方独有大量研发/取证提交。此数字描述提交拓扑，**不等于缺33个必要依赖、也不能做自动merge**。我方旧`dev_v24/feat-310p-dflash` 是上游祖先旁的旧快照，不能凭分支名称判断源码追平。

逐文件SHA比较，以下与指定上游完全一致：`requirements.txt`、`pyproject.toml`、`setup.py`、`Dockerfile.310p`、`worker_310p.py`、`dflash_full_decode_only.py`、`dflash_full_and_piecewise.py`、`dflash_piecewise.py`、`spec_decode/dflash_proposer_310.py`、`dflash_model_310.py`、`dflash_vocab.py`、`llm_base_proposer_310.py`、`dflash_mrope.py`、`sample/rejection_sampler.py`、`npu_input_batch.py`、`patch/worker/patch_idex_310.py`、上游Eager/FDO验收工具。

已确认的主线差异：`model_runner_310p.py` 仅额外为`vllm.v1.worker.cp_utils.get_total_cp_world_size`添加`ImportError`兼容分支；`modelslim_config.py` 增加MemFabric选择性`LinearMethod`；`memfabric_mm_ar_runner_hooks.py`与`memfabric_mm_ar_v2_serve.py`是我方新增，外部上游无对应文件。**这降低了“缺核心DFlash源文件”的可能性，不排除代码接口不兼容或环境异常。**

## 3. 依赖层的重要证据和风险

- **源代码合同**：上游`Dockerfile.310p`默认安装`vllm-project/vllm v0.24.0`，且验收工具`run_310p_dflash_piecewise_acceptance.py`锁定的vLLM源码SHA为`ee0da84ab9e04ac7610e28580af62c365e898389`。vLLM此tag确有`vllm/model_executor/models/qwen3_dflash.py`与`vllm/v1/worker/cp_utils.py`及相关接口。**vllm-ascend分支一致不等于服务器真正装载的vllm一致**。
- **依赖声明**：两侧`requirements.txt`和`pyproject.toml`完全一致；requirements声明`torch==2.10.0`、`torch-npu==2.10.0.post2`、`transformers==5.13.0`、`arctic-inference==0.1.1`、`numba`、`regex`等；`vllm`来自另行安装的Docker步骤，不能用`pip show vllm-ascend`代替检查。上游Docker最终卸载`triton-ascend`和`triton`是310P的有意处理，不可把这两者缺失不加分析地当作故障根因。
- **我们自己的Stock入口潜在耦合**：`modelslim_config.py`即使MemFabric功能标志关闭，在未量化Linear分支仍会导入`memfabric_mm_ar.py`去执行资格判定。该模块顶层导入`regex`等模块。它没有直接证明历史故障由此引起，但这确实是一处上游没有的导入路径风险；未来若traceback落在这里应优先做有门禁的最小修复，不能直接把异常吞掉当作Stock PASS。
- **上游验收脚本不可直接搬运**：`run_310p_dflash_piecewise_acceptance.py`硬编码 `/home/whn/vllm-ascend-openspec-fix`、`/home/whn/vllm-v0.24.0-clean`、`/home/qzh/datasets/gsm8k/test_vllm.jsonl`；`run_310p_dflash_full_decode_only_acceptance.py`的35B预设默认测试`num_speculative_tokens=15`、还要求Graph证据，不符合本次`num_speculative_tokens=7`和**先Eager**的任务合同。

## 4. 最小证伪计划：环境先行，再决定是否补源码

1. **无需实机，已完成**：记录上游和主线逐文件SHA、引用的 vLLM v0.24.0 tag、两份依赖清单和增量差异；暂不改正式Model Runner或Kernel。
2. **下次进入310P服务器时首个G0**：在Stock所有MemFabric标志关闭且无设备工作时运行 `python3 tools/310p_mm_ar/dflash_stock_preflight.py --out /tmp/opencode/<new_run_id>/stock_dflash_preflight.json`；这只做路径、Python分发版本、所需API源文件、模型配置、`pip check`只读盘点，**不启动服务、不安装软件、不跑NPU、不替换SO**。使用真实Python执行器及明确新目录。
3. **G0通过后可选导入探针**：在隔离子进程执行相同脚本的`--import-smoke`以获取真实ImportError栈；仍不启动模型，但可能加载torch-npu动态库，须核实现场环境安全。
4. 只有看到具体的`ModuleNotFoundError`/`ImportError`、`pip check`依赖冲突、版本或文件加载来源，才对症处理：若上游声明已有且服务器缺包→**环境安装偏差**；若上游声明缺失但源码硬依赖→**上游依赖声明缺陷**；若上游无但仅我方新增导入报错→**我们的融合隔离/移植缺陷**；若vLLM版本或API不同→**底座版本不匹配**。修复时锁定组合，不执行无脑`pip install -U`。
5. 先完成 **Stock+DFlash Eager** 的最小单请求及 batch10 的正确性、接受率和进程退出检查；确认Stock启动PASS后再测独立Stock+DFlash Graph（由上游合同及现场实测支持哪些模式就报告哪些模式）。**Stock Graph PASS也不能证明V2 Graph PASS**。只有此阶段通过，才把V2注入同一模型并研究Graph真实Kernel命中与生命周期。

## 5. 当前判定

- **DFlash关键源码是否明显漏合**：目前核对的关键文件都相同，**未见明显漏合**；全仓所有文件尚未做穷尽核对。
- **缺少Python依赖是根因吗**：**UNKNOWN**（没有历史原始Traceback与服务器安装事实，不能确认）。
- **实际vLLM底座是否正确**：**UNKNOWN**（上游要求v0.24.0，服务器实际导入版本和路径未核验）。
- **Graph/V2已经可用吗**：**NO**。历史未通过，Graph伪命中不可算验收。
- **是否需要310P3立即执行**：**否**，本期仅做上游静态归因、检查工具和CPU测试；下一轮进入实机前还需用户指派OpenCode执行G0。
