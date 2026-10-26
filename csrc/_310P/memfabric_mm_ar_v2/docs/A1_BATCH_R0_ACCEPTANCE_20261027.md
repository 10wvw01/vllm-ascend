# A1 BATCH-R0｜服务器重进场证据审核与R1准入裁决（2026-10-27）

> **审核类型：**仓库已上传取证/原始CPU日志的远端静态复核；**非**审核者在310P3直接执行 NPU/编译。批次 `a1-r0-20261027-001`，现场A1执行时HEAD `235de9f6d13a394ffaaddde1db8843154aeb8813`；证据最终归档于A1工作分支 `73114fd12d725827f9b314fd716903ea3e48b582`。来源：[现场报告](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r0_reentry_20261027/BATCH_R0_REPORT_20261027.md)、[取证索引](https://github.com/10wvw01/vllm-ascend/blob/work/310p-mm-ar-large-m-a1-20261026/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r0_reentry_20261027/evidence_index.json)及相邻完整SHA/CPU原始文件。主线仍为唯一正式任务管理入口，本报告不构成E2共享部署、建池/KFC或Kernel运行的自动许可。

## 1. 分项结论

| 验收项 | 复核结果 | 限制 |
|---|---|---|
| A1原仓CPU | **PASS**：`py_compile` 0；`unittest test_a1_*.py` 34/34；`pytest test/ut` 88 passed + 27 subtests；`bash -n` 0 | 原始文本日志已提交仓库；真实NPU路径仍未执行 |
| A1/U1安全闸 | **静态核对PASS**：三项U1许可变量、Confirm额外变量及5份二进制冻结身份未删除/修改 | 代码闸存在≠人工批准，U1下一轮实际装载身份仍需检查 |
| 四卡只读取证 | **已完成**：卡5664已占用，16384(dev2,3)健康、无可见进程且hugepages 0/0；38144 dev4约13GB残留并满HugePages；43520 dev7 MCU/Chip Alarm且占用 | 16384仅**临时候选**，设备初始化前再取健康、权限和独占快照；其它卡排除，不干预租户 |
| 历史MemFabric来源 | **找回**：本地未推送修复提交`5802f55026c16e153e783d8ef26a29f813d63dd9`（基于镜像`131ee5ff`），修复三个SetAccess物理号，MallocPhysical逻辑号不变；Patch原文和修复提交栈已归档 | 对应历史CANN9.1.0实机验证链S3-R/S3-R2在报告中有引用；本轮未在新环境再次执行设备测试 |
| 历史Host SO | **可复用**：本地build与S3-R staging两份各3,240,592B，SHA256 **`1dc46d7dcbe583a5dbb78b4c65a0b5e22f9f352976c53ed31b761fa744f91a30`** | 两份**当前共享装机**Host SO仍是未修复`88d392850733d7f166ce28615a8fdbf164b604979df9d3fc8cd0c10c64fd7bc1`；不能签发在装G0 PASS |
| V2插件/KVER11 | **归档匹配**：Python插件SO `8eb65de9…`、KVER11 JSON `25e16b0a…` | 插件editable指向正在供其它任务使用的主工作树，运行时可能被其它任务影响，需要冻结/隔离装载与二次SHA复查 |
| 正式D4 Device SO | **正确候选在位**：历史冻结 `2a19390c…`，来自历史构建 | **当前插件目录实际在装的是`b8ca96f0…`基线版本**，不能仅凭`U1_DEVICE_SO_PATH`是正确候选路径宣称真实Kernel已装载 |
| 本轮Host重编译 | **NOT_ATTEMPTED（合理）**：验证过的历史SO可复用，不无谓重编；单独g++编译CPU映射测试15组PASS | 新MemFabric Draft PR #1是未经CANN/NPU验证的兜底代码，不是历史原版，不得替换旧产物或改U1冻结SHA |
| E2部署 / 首调KFC / M2048Smoke / L1性能 | **NOT_RUN** | `G0_APPROVED=DEPLOY_APPROVED=SMOKE_APPROVED=false`，没有任何新D4 Stock/V2完整性能数据 |

### 1.1 特别确认：修复源码的“失踪”已解释

现场从`/tmp/opencode/mf-fix`恢复**真实历史本地提交** `5802f550`，其变更只有：`src/hybm/csrc/common/hybm_define.h`定义`HybmPhysicalDeviceIdOf`，`src/hybm/csrc/driver/hybm_gva.cpp`授权一次、`src/hybm/csrc/mm/hybm_acl_vmm_segment.cpp`授权两次；未修改底层AICPU协议。确证当时只在服务器本地，镜像未推送。提交的`memfabric_fix_5802f550_setaccess_physical.patch`现在是远端可恢复证据。**最优方案是将原修复提交补推/归档到正确MemFabric仓库，而非合入此次新写的Draft PR#1“相似修复”。**

**工具限制**：只读`a1_reentry_readonly.py`中的`three_setaccess_mapper_static_hint=false`是**检查布局不适配历史原始实现**（新候选header名不同，历史helper位于`hybm_define.h`），不能把该布尔值当成实际三处源码修复缺失；已单独人工按归档patch核对。未来若继续使用该hint应先修正并在原仓CPU复验，保留当次原始输出不篡改。

## 2. 下轮E2/R1高价值风险和明确前置要求

**R0工程准备通过 ≠ 当前运行环境G0通过。** 目前唯一真实部署阻塞：**生产两路径Host SO仍旧、Device SO实际在装旧基线**。已恢复的可用源和字节产物不需重新发明，但**服务器仍有其他工作负载**：

1. **优先隔离而非全局覆盖**。历史S3-R/S3-R2用进程级`LD_PRELOAD`验证过修复版Host SO；E2先静态核对正式U1中的`ctypes.CDLL`、Python MemFabric动态绑定、`dlopen`和Device SO装载路径，判断能否用独立进程/工作树/私有路径挂载**真实修复Host SO + 真正的D4 Device SO + 冻结Python插件**。尤其`U1_DEVICE_SO_PATH`通过SHA检查**不等于**CANN实际上加载这份ELF，必须另有加载路径证据和正式trial符号。不能直接从正在运行其它任务的可编辑插件目录替换Device SO。
2. 如完全隔离不可行，**不得自行升级为写共享库**；明确`/usr/local/memfabric_hybrid/...`及`site-packages`两份SO的受影响服务/进程及安装维护窗口，检查属主/权限/可回滚备份、既有文件SHA、重启/装载边界，经用户单独批准后才有写权限。不能静默覆盖主工作树中的D4 SO。
3. 下一次建池前**再次对16384（dev2,3）做四卡只读检查与独占取证**；先审查当期新G0、`ASCEND_RT_VISIBLE_DEVICES=2,3`、Host/Device/插件/KVER真正加载的SHA、CANN9.1.0/ACL逻辑-物理映射。历史其它卡有占用/报警不得绕道使用。
4. **KFC不预部署**：仅在已批准首次M2048双Rank短Smoke建池时允许MemFabric自动部署KVER11；观察两Rank与CP1加载、KFC helper失败、507014/507899/507018和首写 bit-exact；故障立即STOP，封存首次出错日志/设备状态，**禁止自动换卡、执行长测或用rm清CP1**。
5. **每阶段的授权必须可追溯**：可以在一份现场委托里高密度批处理E2前的隔离预备；真正部署与首轮Smoke需明确许可；`large screen`和`large confirm`仍分别在前一阶段证据审核后申请，不得一键把所有脚本接通。
6. **项目主线保持Stock默认**。本次既未证明现D4完整L1胜过Stock，也未测L3或L4；历史S3-R2的25/25请求通过仅能说明当时相同SHA的短场正确性旁证，不能直接替代当前卡上S0/Smoke和发布级Gate E。

## 3. 决策和移交

- **R0阶段验收：`R0_EVIDENCE_ACCEPTED / CPU_PASS / PROVENANCE_FOUND / ARTIFACT_REUSABLE`**，不需要重新跑CPU或盲目重编Host SO。
- **本轮G0：`G0_READONLY_OBSERVED / APPROVAL_PENDING`**。16384为唯一候选，数据取证采样时健康；不能签发`G0_GO`或`DEPLOYMENT_READY`，仍需E2恢复方案确定、实际装载二次身份和使用授权。
- **下一步：**研发侧落实修复源码长期GitHub归档与隔离部署可行性静审，再安排用户/OpenCode单批次`E2 preflight→（许可后）受控加载/恢复→（独立许可后）M2048短Smoke`。不执行L1 Screen/Confirm和L3模型长测。
- **安全/项目状态**：Gate E首写潜在静默损坏未被关闭，非0/1映射仍需真实CPU/NPU链确认，`L3_MEASUREMENT_NOT_READY`未解决，PR#9依旧Draft。仓库A1版本仅用于解释R0，不自动合主线代码。

**来源保真说明：**本报告中的CPU测试计数、卡余量/负载、哈希和历史链均来自已上传的BATCH-R0证据；`/tmp/opencode/a1_r0_evidence_20261027`的原始npu-smi/ps/ss因敏感信息保留SERVER_ONLY，没有从聊天或从GitHub凭空声称审核过这些原文。以上R1隔离装载建议是本次静态审查的**后续工程建议**，不是R0现场已经验证的事实。
