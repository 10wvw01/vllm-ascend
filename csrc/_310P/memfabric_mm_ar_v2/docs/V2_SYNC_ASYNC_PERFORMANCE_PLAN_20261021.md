# 310P3 V2 同步与连续异步执行性能差异专项验证（2026-10-21）

> 状态：GitHub 测试代码与首轮静态审视准备中；**尚未在 310P3 上运行**。本文件定义验证条件与验收结果，不将编译通过视为设备实测完成。
>
> 不切换 `10wvw01/memfabric_hybrid@wgm-dev-310p`；继续使用已知镜像修复线的 **KVER11** Host + AICPU；不修改 MemFabric 协议/设备库。V2 从 `feat/310p-memfabric-public-api` 派生独立 `perf/310p-v2-sync-async-value-20261021` 测试分支。

## 1. 为什么做这项验证

此前 T2 已使用资源复用/预热后的 Stock MM+FP16 HCCL、V2 融合同步微基准（各进程单次创建通信资源，计时循环不建池），M=4096 的 Stock 3.2735ms、V2 2.6875ms，**V2 单调用延迟降低17.9%**。但 G3 真实 Qwen3.6 Eager Serve 的吞吐三会话变化中位 -3.51%，TTFT p50 稳定退化 2.1%~2.5%；Graph 中 fused 实际为 Stock，不能拿来证明 V2 性能。

因此本轮只回答：**对相同 M/N/K、同一类型权重、同一 Rank、稳态预热、两种提交/同步节奏，V2 的相对优势是否改变？** 若不改变，下一轮再查真实 Serving 中 Prefill/Decode 相互干扰、路由覆盖率与 Host 整合。

这不是 HCCL “AI CPU Cache 开关实验”，也不是 Gate E 复现或 Graph 优化。

## 2. 测试几何及真实性

- 每 Rank FP16 激活 `X=[M,2048]`、NZ 权重 `W=[2048,2048]`、输出 `Y=[M,2048]`，TP=2。
- Stock 使用 `torch.ops.vllm.unquantized_gemm(x,w,None)` + `torch.distributed.all_reduce(..., async_op=True)`；采用 vLLM 当前 unquantized_gemm 实际入口，其底层为 `F.linear`。FP16 Reduce Op SUM。
- V2 通过 `mf310p_v2_trial_create/ib_probe/run/check_status/destroy` 已有 ABI；设备 Kernel overlap=1；无需更改、编译或替换 MemFabric Host/SDMA 协议。
- V2 Host Handle **一次创建、全部预热、计时循环中不创建或销毁**。输出地址在建池之前分配，固定存活，跨窗口仅在全部通信完成后复用。
- Stock HCCL Process Group **进程启动时创建一次**，计时期间不建立新进程组；首调用/预热不混入稳态。
- `steps=40` 是 40 组**数据与权重各异**的连续调用。窗口=1 每个调用后 wait/sync；窗口=2/4/8 每窗口所有调用入队后统一 wait/sync，再复用 output；它是 **异步排队压力包络**，不是 Qwen 40 层完整依赖图。真实推理中的层间依赖、Decode 与 Prefill 交叠需后续 Serving 实验复现。
- 模型路由 `2048≤M≤4096`；原始 trial API 要求 `M%chunk_rows==0`，所以本轮主测试 M=4096，第二步 M=2048/3072，chunk_rows=1024。M=2049/4095 的 Input Stage 补齐属另一个待测环节，不在第一轮混入。
- Raw ABI V2 测试 **不包含** Python Serving 路由的阈值判断、首形状位精确校验等一次性逻辑；本阶段旨在隔离调度差异。后续 P1/Serving 验证再测这些开销。

## 3. 四组对照

| 实验 | Stock | V2 | 判定重点 |
|---|---|---|---|
| W1（同步） | 每次 MM→HCCL，Work.wait + Stream sync | 每次融合 Kernel 后 Stream sync | 再现 T2 同步优势 |
| W2 | 连续两个算子/AR 才同步 | 连续两个融合 Kernel 才同步 | 少同步时是否有相对收益变化 |
| W4 | 连续四次异步提交后同步 | 连续四次异步提交后同步 | 本轮核心：V2 相对 Stock 是否反转 |
| W8（可选） | 同上，每批 8 次 | 同上，预先分配/预热 8 个输出槽 | 只用于有证据支持继续扩窗时 |

记录每个配置两 Rank 的 wall 总时间/40、enqueue 总时间/40、中位/P95。以较慢 Rank 的会话中位数为主（max-rank median 近似）。**不**把该统计冒充“单次每 Rank 取 max 再求中位数”或设备单 Kernel latency。

两种方法的每窗口同步次数相同，但 HCCL 需要 Work.wait 形成通信流依赖；Stock 的 MM 输出引用必须保留至窗口完成；V2 不可提前复用仍在途的输出地址。计时期间不做 `aclrtMemcpy`/D2H status 读、NumPy 校验或 profiler；正确性与协议查询在计时前后单独执行。

## 4. 环境与门禁

**G0：现场锁定。**

1. dev0/dev1 独占，原始 Worker 已退出；Host v11、设备 v11、JSON 实际版本可追溯；CANN 9.1.0、torch_npu/vllm-ascend 版本、机器温度/频率等记录。源码分支必须为本独立 perf worktree，且干净。
2. MemFabric 使用已部署的修复线，不切到原始 `wgm-dev-310p`，不合并两个 MemFabric 历史分支。
3. 不安装、覆盖、卸载 Host 或 AICPU；如果原系统找不到 `libmf_sdma_orch_v11.json`，立即停止，不去自动部署。
4. 新 worktree 中完成扩展源的 `setup.py build_ext --inplace`（根据当前设备编译环境），实际 import 的 `vllm_ascend_C` 必须来自本 worktree；不得指向旧 system Python package。运行器在进程内硬性校验。
5. 先跑 CPU-only `test_sync_async_summary.py` 和源文件 `py_compile`；检查扩展 ABI `trial_*` 导出，实机验证完全遵循正确性检查。
6. 当前还未做运行时结果。若编译/导入失效，只能修复测试驱动兼容性；不得为了跑通测试而改变 V2 算法或 MemFabric 协议。

**G1：短 Smoke（节省实机调试成本）。** 先 M4096、steps=8、W1/W2/W4、repeat=3，按 ABBA stock→v2→v2→stock，确认全部正确、0 status 错误、无 hang/507014，并能计算报表。

**G2：正式初步 A/B。** M4096、steps=40、W1/W2/W4、repeat=8。ABBA 四个独立进程：Stock→V2→V2→Stock。第二组窗口顺序 W4/W2/W1，抵消进程内部预热/温度递增。通过后补 M2048 和 M3072 的相同配置。可根据波动额外增至 ≥3 个独立 ABBA 循环做统计结论（现有 2 对仅探索性，不能声称有统计显著性）。

**G3：原因判定。** 仅在 W1→W4 发生明显相对优势变化时，增加最小 profiler 或 host enqueue 时序采集；否则不做无目标的大型 profiling。Profiler 与正式计时独立会话。

## 5. 执行命令（交给 OpenCode，在 310P3 服务器运行）

先用**新建的独立 worktree**检出本分支。完整构建由 OpenCode 依据 `docs/README.md` 和服务器已安装 CANN 做；不要更换 MemFabric 分支。

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1
export MF_SDMA_ORCH_JSON=/usr/local/memfabric_hybrid/latest/aarch64-linux/hybm/aicpu_kernel/libmf_sdma_orch_v11.json

# 在新 perf worktree 根目录；确认 extension 由当前 worktree 编译和加载
python3 -m py_compile \
  csrc/_310P/memfabric_mm_ar_v2/test/perf/sync_async_tp2.py \
  csrc/_310P/memfabric_mm_ar_v2/test/perf/run_sync_async_abba.py \
  csrc/_310P/memfabric_mm_ar_v2/test/perf/summarize_sync_async.py
python3 csrc/_310P/memfabric_mm_ar_v2/test/perf/test_sync_async_summary.py

# 第一次 smoke：输出目录必须不存在，绝对路径，不覆盖原日志
python3 csrc/_310P/memfabric_mm_ar_v2/test/perf/run_sync_async_abba.py \
  --out /tmp/v2_sync_async_smoke_20261021_01 \
  --rows 4096 --steps 8 --windows 1 2 4 --warmup 2 --repeat 3

# Smoke 通过后再跑 40 步
python3 csrc/_310P/memfabric_mm_ar_v2/test/perf/run_sync_async_abba.py \
  --out /tmp/v2_sync_async_m4096_20261021_01 \
  --rows 4096 --steps 40 --windows 1 2 4 --warmup 3 --repeat 8

# 如果前两轮均 PASS 且无设备故障，扩展形状
python3 csrc/_310P/memfabric_mm_ar_v2/test/perf/run_sync_async_abba.py \
  --out /tmp/v2_sync_async_m2048_3072_20261021_01 \
  --rows 2048 3072 --steps 40 --windows 1 2 4 --warmup 3 --repeat 8
```

每个 `out` 内生成 `provenance.json`（当前分支 SHA、v11 JSON SHA、系统 Host SO SHA）、四个独立进程日志、`run_index.jsonl`（每日志哈希、退出码）、`sync_async_summary.json`。命令返回码必须为 0、两个 Rank 均有 `complete PASS`、全部窗口都有 `validated=true`；不得只看“模型未报错”。

## 6. 性能归因判据

定义每窗口 V2 延迟改善：

`Gain(W)=100%×[1−T_V2(W)/T_Stock(W)]`。

核心指标为 `Gain(W4)-Gain(W1)`，单位百分点。它把 Stock 同样由于降低同步频率产生的收益纳入了对照。

| 模式 | 解释 | 后续动作 |
|---|---|---|
| W1 V2 快，W4 V2 变慢/收益显著缩小 | 连续异步排队可能损害融合优势 | 追踪 Host enqueue、设备 SDMA 等待、SQ 资源、Kernel 并发 |
| W1/W4 V2 都更快、优势稳定 | 同步→异步本身不是性能反转主因 | 直接进入真实 Serving Prefill/Decode 交叠与覆盖率统计 |
| W1 已无法复现 T2 单算子优势 | 当前代码/编译/基线/环境或形状口径漂移 | 先重建 T2 对照，不能先大改 Kernel |
| 两 Rank/两会话严重漂移、或请求超时 | 当前数据不可归因或出现 P0 | 先报告和保全证据，停止自动重试 |

**不能**用单次 40 独立调用 W4 的优势预测真实 Qwen40层吞吐，因为真实层间还有 attention、GDN、MoE 与后续依赖。也不能把任何延迟差简单解释为 HCCL AICPU Cache 的独有贡献。

## 7. STOP 与交付

出现 bit-exact FAIL、状态非零、507014/507018、两个 Rank 配对错误、内存/地址非法、扩展路径不一致、SO 哈希不一致、900s 单进程超时，**立即 STOP**。不自动回退系统库、不执行 `wgm-dev-310p` 切换、不随意关闭首写保护，也不要跳过测试继续压测。

OpenCode 交付 `V2_SYNC_ASYNC_EXECUTION_REPORT.md` 和脱敏证据：每场景的 raw 两 Rank 日志、样本汇总、SHA、总线/设备利用率快照与具体环境、问题修复前后差分、两轮静态审视补充、资源/系统未变动的证明。原始含 GVA/内部路径日志留服务器，不全量公开。

本轮的最低成功标准不是“V2 一定更快”，而是**明确区分同步与有界连续异步条件下 Stock/V2 的相对性能变化，并排除资源重建与预热成本的干扰**。

### 已有证据

- [T2 实机原始性能与 profiler](T2_AR_ONLY_REPORT.md)
- [2026-10-17 T0/G2 同步微基准](20261017实机测验及修复.md)
- [G3 Eager Serving 性能与 Graph 路由缺陷](V2_G3_QWEN_E2E_REPORT.md)
