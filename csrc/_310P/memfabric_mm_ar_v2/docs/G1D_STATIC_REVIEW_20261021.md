# G1-D 两轮静态审视记录（2026-10-21）

> 说明：本文件记录 **GitHub 代码级静态审视**，尚未在 CANN/Bisheng/310P3 编译或执行。CPU-only 测试脚本已准备，但无实机或本地完整依赖环境执行结果，不得标记“实机 PASS”。执行侧需补充构建日志、依赖/二进制哈希与实测结果。

## Round 1：协议语义、布局与故障区间

| 检查项 | 结果 |
|---|---|
| G1-D 默认 OFF，编译需同时启用 G1-C | 源码确认；V2 G1-D header 有编译期约束 |
| V2 两处 wait 语义 | 均保留原 wait/邮件字段校验/CheckOrTrap，只在返回和失败前加阶段记录 |
| MemFabric Host 与设备分工 | 仅 Host Supervisor 新增本端保留区只读快照，不改 AICPU SQE、Doorbell 和 Epoch |
| 缓存行精确边界 | 0x2A00=STAMP[1020..1023]+MAIL[0]；0xAA00=MAIL[1023]+ArrHead，Host 新增 static_assert |
| request/arrival/quiet 布局 | 依赖原布局常量，不写控制区、不改游标、未新增 ABI |
| 诊断 SO 保护 | V2 独立 g1d SO；MemFabric AICPU 沿用 G1-C v12 哈希产物，需实际部署核验 |
| 取证等级 | D2H 非原子、设备与 AICore 缓存观察不一致、SQ 排空非正确完成证明均已写入解析器/计划 |

**Round 1 发现并修正：**

1. G1-D 最初只记录 generation，存在“同一代上一封邮件已返回，却被误报成当前 chunk 返回”的风险；后续把 chunk 编号打包到专属字段中，日志现在含 waitChunk。
2. 独立编译新宏时须使用 CMake 列表传参，不可将两个 bisheng -D 选项拼为一个空格字符串；已使用 list(APPEND) 修正。
3. 设备 G1-D 诊断 SO 必须有独立文件名，不能复用 g1c SO 名称；已增加 g1d 后缀。
4. G1-C 测试结束后设备 v12 已被清理；计划已修正为重新验哈希、独立部署，不能假定现场 v12 仍存在。

## Round 2：源码参数、脚本门禁与证据复核

| 检查项 | 结果 |
|---|---|
| V2 重叠循环 | 只在 prev 对应的 wait 后记录 prev；收尾 wait 只记录 last |
| stage 定义 | 50=入口、52=wait 返回、53=校验失败/Trap 前、51=校验成功 |
| 返回邮件的状态持久化 | Core0 单独 64B 槽；generation + chunk + seq/status/dst/len/imm 无跨核共享 |
| G1C summary 兼容 | 52/53 增加对应文本；原 50/51 仍可识别 |
| MemFabric 本端采样 | Host 根据当前本地 reqBase 读取 ARR/quiet，不把 peer GVA 当本地 AclrtMemcpy 地址 |
| 异步窗口限制 | 1<=window<=output_slots<=4；按窗口 sync+bit-exact，避免立即把 40 层完全无同步作为首轮 |
| Gate E 路由真实性 | 启动脚本要求独立 V2/MF G1-D 工作树和显式 v12 JSON；仍需实机检查 Python import/链接加载路径 |
| 结果解释 | 原始回绕计数包括预热/DONE，TP2 脚本给出的理论邮件数不可当精确 reqSeq |

**Round 2 发现并修正：**

1. 修改调用点传参时曾发生两处 prev/last 混用；提交前二次逐行核对修复，当前两处分别为 prev 和 last，未再直接复用错误参数。
2. 原 G1C 离线解析器没有 52/53 定义；已补齐，防止新增阶段被误报 unknown。
3. 单窗口压力设计可能触发输出槽引用数限制，故 D2 显式将 window 上限约束为 output_slots<=4；无同步 40 调用不是首轮承诺。

## 剩余必须上机解决的事项（不得伪造完成）

- **编译**：CANN 9.1.0 / bisheng(dav-2002) 与 C++ 编译器实编；设备 CMake flags、header typedef、动态链接、ELF 所依赖正确。
- **部署**：MF Host so 两处实测加载路径、v12 原始产物 SHA256 与新 Host 链接；恢复脚本演练；确保 system v11 不被覆盖。
- **CPU-only**：运行 test_g1d_evidence_summary.py 与原 G1-C 测试，检查 log 格式和样本解析。
- **C1/D2**：FP16 bit-exact、协议 status、环多次回绕、1024边界健康对照；窗口4若因当前输出槽引用计数失败需降级而非强行扩槽。
- **长稳**：D1 Gate E 和 D3 统计在新埋点条件下的时序扰动；无埋点版本才用于性能/修复终验。

**结论：** 静态准备已完成两轮，发现的明显设计/参数问题已修；本文件**不构成“可以免审直接实机长稳”证明**。执行侧必须先过 D0、D1.0 构建和 D1.1 正确性 Gate。
