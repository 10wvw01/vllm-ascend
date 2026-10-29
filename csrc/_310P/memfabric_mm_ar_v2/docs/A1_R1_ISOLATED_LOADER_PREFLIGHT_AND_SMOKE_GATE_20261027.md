# A1-R1｜隔离装载、环境恢复及 M2048 Smoke 准入合同（2026-10-27）

> **2026-10-27 更新：此文档已被[同一批次贯通 Smoke/L1/L2/L3 的最新授权与执行合同](A1_R1_ONE_BATCH_THROUGH_L3_20261027.md)替代。** 用户明确批准环境准入通过后首次M2048 Smoke及其成功后的连续性能验证，**不再要求每个正常阶段分别返回用户审批**。本文件仍可参考隔离装载/回滚细节，但其中“R1A后必须再申请R1B”等旧管理约束不再生效；异常STOP、他人环境隔离、数值与实际加载真实性硬门仍有效。

> 主线[大M实验总账](https://github.com/10wvw01/vllm-ascend/blob/dev/310p-memfabric-mm-ar/csrc/_310P/memfabric_mm_ar_v2/docs/LARGE_M_PREFILL_TTFT_EXPERIMENT_LEDGER.md)负责状态；本文件属于A1实施分支[PR#9](https://github.com/10wvw01/vllm-ascend/pull/9)。用户已授权着手**准备**，R1A可交OpenCode进行纯CPU和只读检查；**真实部署、建池、KFC、M2048双Rank Kernel依然必须R1A审证后另外放行**。服务器正在运行其它任务，旧卡余量及环境身份不自动沿用。

## 1. R0已审核事实与新变化

- R0 `a1-r0-20261027-001`：原仓unittest34/34、pytest88+27subtests、独立CPU DeviceID 15/15 PASS。唯一暂定单卡16384(dev2,3)，R0时健康无可见进程；5664被占，38144 dev4孤儿内存/HugePages全占，43520 Die7 Alarm且被占；**本批重扫四卡**。
- 历史修复Host SO SHA`1dc46d7dcbe583a5dbb78b4c65a0b5e22f9f352976c53ed31b761fa744f91a30`两份归档完好（`/tmp/opencode/mf-fix/build/src/hybm/csrc/libmf_hybm_core.so`、`/tmp/opencode/gate_e_s3r/fixed_host_lib/libmf_hybm_core.so`）。当前共享系统/Python双Host路径均是**未修复**`88d392850733d7f166ce28615a8fdbf164b604979df9d3fc8cd0c10c64fd7bc1`；在装Device SO为baseline`b8ca96f0...`，归档D4正确候选为`2a19390c83c3fe6b0bdb8946d88990cca66b331f765ca445dbda2e787c2d2a19`；插件`8eb65de9...`、KVER11 JSON `25e16b0a...`。
- 旧服务器**原始源码修复commit**`5802f55026c16e153e783d8ef26a29f813d63dd9`（3处`aclrtMemSetAccess`用物理ID，`MallocPhysical`仍逻辑ID）已严格按原Patch恢复。三份Git Blob全哈希与历史目标完全相同，经[MemFabric镜像仓PR#1](https://github.com/10wvw01/memfabric_hybrid_310p/pull/1)合入`10wvw01/memfabric_hybrid_310p:main@0670f3c7db7cfb090e8a312b9347fa039ada7da2`。没有新Host编译或NPU实验；另一仓行为不同的兜底Draft PR已关闭，禁止误选。
- **危险假阳性**：`u1_run.sh`会计算`U1_DEVICE_SO_PATH`候选SHA和`U1_CORE_SO_LIB64/U1_CORE_SO_PY`指定路径SHA，`u1_arm.py`的`ctypes.CDLL(V2_HOST_LIB)`是导入vllm_ascend C++插件；这些并不独立证明**实际加载的Device SO**或所有Host MemFabric符号均来自正确1dc46d7d。当前插件editable导入指向其它任务正在使用的工作树。仅按候选SHA通过即运行是禁止的。

## 2. BATCH-R1A：一次完成的CPU/只读/隔离预备（当前可委托）

1. 在新的**独立worktree和全新绝对输出目录**读取主线入口、总账、R0验收及本合同。A1具体实现源仅使用`work/310p-mm-ar-large-m-a1-20261026`当前HEAD；MemFabric只读参考已恢复的镜像主线`0670f3c7...`与历史本地`5802f550`；记录实际分支/HEAD/tracked状态，**不操作其它任务工作树**。
2. **重扫四卡且不启动设备**：`npu-smi info`、`npu-smi info -t proc`、hugepages、容器PID命名空间/报警、逻辑↔物理Die映射、端口及租户。健康空闲且独占的`16384(dev2,3)`只是候选；无安全卡时标`CARD_NO_GO`并继续其余纯CPU审计，不结束其它任务、不换到Alarm卡。
3. 生成本批`paths.json`（下附参考），执行新增`test/perf/a1_r1_loader_readonly.py`；保存完整SHA、ELF `readelf -d` NEEDED/RPATH/RUNPATH、路径真实inode和Symlink、装机版与候选版差异。**脚本不导入torch/CANN、不dlopen、不写SO、不签发GO**，且将`actual_host_dlopen_path_proven`和`actual_device_kernel_so_loaded_proven`固定为false；这代表静态证据不足，不能手改。
4. 审查`u1_run.sh`、`u1_arm.py`、`vllm_ascend_C` CMake/packaging、`libmf310p_v2_trial_device.so`注册和`memfabric_hybrid` Python包装/动态链接全链；画出Rank0/1分别在何处读Host SO/插件/Device SO/KVER11，每段标**CONFIRMED/SOURCE-INFERRED/UNKNOWN**。不能把`LD_PRELOAD`“曾在S3-R2成功”直接外推为Device SO隔离成功。
5. **只在本批scratch目录**存放由旧二进制复制得到的私有Host SO和D4 Device SO，各自SHA复验，给出无需更改生产前缀、其它工作树或pip安装的“私有Python import+Host LD_PRELOAD+真实Device注册”候选路线及必要验证方法。当前版本是否支持完整隔离属于待取证问题：无法证明则`ISOLATION_UNPROVEN`，不自动退化为共享覆盖。对实际Host进程地图、Runtime设备加载的证据设计属于R1B用户批准后才能执行。
6. 如私有隔离确实不可能，提出**共享维护窗口替代方案**：列出系统/Python双Host SO及editable插件目录Device SO每个受影响进程、租户、权限、原文件完整SHA、新私有备份路径、恢复字节校验、阻断和回滚顺序，要求所有共享使用者退出证据及**用户另行批准**；此批不得复制覆盖共享文件。
7. 在原A1隔离worktree运行`py_compile`、`unittest discover -s test/ut -p 'test_a1_r1_loader_readonly.py' -v`、`pytest -q test/ut`、`bash -n test/perf/u1_run.sh`及静态门禁核查，保存所有原始日志/exit。全部步骤可在**同一次OpenCode委托**内执行，最终提交脱敏证据包和`R1A_PASS/NO_GO`建议，不宣称设备已准备完毕。

示例paths.json；**现场必须核对每个绝对路径**，特别是在装Device SO的真实文件名：

```json
{
  "host_archive_fixed": "/tmp/opencode/mf-fix/build/src/hybm/csrc/libmf_hybm_core.so",
  "host_system": "/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/lib64/libmf_hybm_core.so",
  "host_python": "/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid/lib/libmf_hybm_core.so",
  "plugin_so": "/home/w00804146/310P/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so",
  "d4_device_candidate": "/tmp/opencode/d4/build/device_candidate_diag_on.so",
  "device_so_installed": "/home/w00804146/310P/vllm-ascend/vllm_ascend/libmf310p_v2_trial_device.so",
  "kver11_json": "/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/hybm/aicpu_kernel/libmf_sdma_orch_v11.json"
}
```

```bash
# 从新隔离A1工作树根目录执行；输出路径必须预先确认其父目录已存在
P=csrc/_310P/memfabric_mm_ar_v2
python3 -m py_compile "$P/test/perf/a1_r1_loader_readonly.py"
python3 -m unittest discover -s "$P/test/ut" -p 'test_a1_r1_loader_readonly.py' -v
python3 -m pytest -q "$P/test/ut"
bash -n "$P/test/perf/u1_run.sh"
python3 "$P/test/perf/a1_r1_loader_readonly.py" \
  --paths-json /本批绝对输出父目录/paths.json \
  --out /本批绝对输出父目录/new_r1_loader_inventory
```

**回传**：`r1_loader_static_inventory.json`、`paths.json`、四卡+租户占用清单、MemFabric源Blob核对、依赖/注册路径审计、私有目录SHA、原仓CPU日志、`ISOLATION_FEASIBLE/UNPROVEN/NO_GO`、若需要维护窗口则备份/恢复与受影响进程清单。R1A只有**只读/隔离证据验收意义**，没有任何NPU许可。

## 3. BATCH-R1B：只有R1A审查+独立用户批准，才能运行

**路线I（首选）**：建立能复核真实装载的私有子进程（独立PID、受控`LD_PRELOAD`及私有Python依赖、Device注册）。Host以`/proc/<rank_pid>/maps`/运行时加载日志/真实符号路径核对`1dc46d7d`；设备SO可能不是进程ELF文件映射，因此需Kernel注册/CANN真实加载日志和Trace证明使用`2a19390c`，不能只根据候选路径SHA判断。确保没改全局SO或其它任务editable目录。

**路线S（非首选）**：隔离不成立且必须共享安装时，在有明确维护窗口、受影响租户都退出和用户明确批准后，先按sha备份可恢复双Host+Device SO，逐项验证权限与覆盖内容，再进行受控恢复；有任何共享占用/校验失败直接`NO_GO`，不能影响其它业务。

**首次最小正确性**：再次重扫16384(dev2,3)双Die健康/独占、确认两Rank可见设备`2,3`、真实SO/KVER11身份，严格U1三批准门（Confirm额外门仍禁用），用户单独授权后运行原`test/perf/u1_run.sh NEW_ABSOLUTE_DIR large smoke`两场Stock与V2。首次MemFabric建池时才自动KFC部署v11；检查双Rank首写bit-exact/status 0/实际D4正式Kernel路由/文件身份，任何507014/507899/507018、KFC/IB/Die异常立即STOP、保全日志与原装库备份，**绝不自动换卡、手工mini_kfc rm、连跑Screen**。Smoke仅说明最低限度正确性，不等于L1性能GO，更不等于发布级Gate E。

**路线X**：不能证明隔离且无法获得安全共享维护窗口，则`R1B_NO_GO`，提交完整阻塞，禁止首次KFC和Smoke。

## 4. 不变工程边界

未获得实际新G0、私有加载证据和用户批次许可前：不编译替换旧`1dc46d7d`、不重建DFlash Runner、不更改Stock默认、不对设备执行显式同步探针、不设置`U1_G0_APPROVED`等变量、不让KFC helper常驻Serving、不尝试一键R1→Screen→Confirm。L3无Profiler真实设备Step计量仍需独立研发审查，L4只在L3_GO后。
