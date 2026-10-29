# V3.2 方案 A 实现与准入

## 实现范围

- 显式 `V32Session.run_pool`，chunk512/1024/2048，保留对应64/128行MM tiling。
- 八核MM，Core1～7不等逐chunk放行；Core0汇总本chunk后按固定R/D/G顺序收件。
- 本地就绪采用独立group/core缓存行及epoch，位于IB专用区域之后；不把IB事件当计数器。
- 单槽Y直接承担MM、ADD目标和Gather目标，输出返回同一SMEM地址，无末尾全量copy。
- 两个有界槽，每槽16MiB；两Rank allgather空闲交集及调用几何，避免单边释放造成覆盖。
  每波acquire先同步原流。该同步、控制allgather、Tensor包装全部纳入完整测量窗口。
- Torch Storage deleter持有session和槽；切片/别名保留Storage即保留槽。
  release只标记空闲，下一波双方流完成后才复用；close遇到仍持有的Storage返回busy。
- create时协商31/32协议版本，禁止混合V3.1/V3.2数据波；同一generation只能发起一次run。
- 原V3.1接口保留，方案B不实现、不修改MemFabric。

本版A按核发布完整chunk就绪；chunk2048仍含两个128行子MM。
不是逐行带自驱收件，也不假装消除了Core0汇总等待。独立producer的收益是其他核可向前计算。
普通Tensor consumer仅支持session拥有流；调用方不得把池内Storage交给其他流、resize或改写。

## 已完成的本地检查与限制

实际OrderedProgress及Transport提取后执行480场双Rank异步FIFO测试，
包括1/2/4/8/16组、1/2/4包，DMA延迟读源、最终逐元素和及完整排空。
实际Transport拒绝status/dst/len/epoch/phase/group/packet错误。
实际Host C++在ACL/SMEM mock下编译，验证双槽、双Rank空闲交集、generation、
busy close、输入别名拒绝和重复释放。
这不模拟CANN缓存、NPU Storage或真实设备资源；不能签设备GO。

**本地无CANN/BiSheng、TorchNPU和310P。设备编译、Storage映射、数值、生命周期及性能NOT_RUN。**
既有V3.1设备SO的成功不能替代新Kernel与新Storage的验证。

## 给服务器的单次执行顺序（用户委托后）

遵循[执行治理](OPENCODE_EXECUTION.md)：新隔离克隆/工作目录、固定完整HEAD，
MemFabric仍锁定db38d75e8485d31451302219eb7a1bfeb1dd107c，复用SDK必须完整核验。
不改共享环境，不动CP1/KVER，不重跑旧V2/Stock，不跑模型/TTFT/Graph或独立归因轮。

1. CPU项目合同及新增测试。
2. 私有`tools/build.py --memfabric-source ... --memfabric-home ... --cann ... --out ...`。
   检查dav-2002新Kernel、两段IB/flag内存、双UB、NZ2ND工作区、公开接口与Host链接。
3. 同一服务器Torch/TorchNPU环境执行：
   `tools/build_tensor.py --manifest <build>/manifest.json --out <fresh-tensor-build>`。
   编译TorchNPU Storage适配，写入SO路径和SHA256到manifest。缺少Storage工厂或
   StorageDescHelper/SetDesc不兼容，记录为准入失败，不退回普通from_blob冒充可用。
4. G0健康且双Die连续独占，源码/SDK/helper/所有SO指纹一致后启动：
   `tools/quick.py start --plan-a --manifest <build>/manifest.json --out <fresh-result-dir> --card 38144 --devices 4,5`
   卡与设备以实际获准独占窗口为准；使用原守护和detached/status流程。
5. 自动执行IB→ADD→输出/别名/持有/双槽/单边释放/close门禁→协议（含单chunk和环回绕）
   →全部chunk融合/下游ND消费→尾块。失败STOP，禁止在不确定ADD会话中重试。
6. 全门禁通过才继续六配置两轮（4预热+30样本）。pool-return的事件和Host Wall完整保留；
   不以Kernel-only或排除acquire成本的数字代替完整接口。没有out-copy路径性能声称。
7. 归档两个Rank、SO/SDK/helper、守护/编译日志及全部样本，按配对Rank max统计。
   对照历史V3.1/V2 E3-G及真实NZ Stock；不只比较旧D4，不签跨批正式性能GO。

设备前置编译/G0通过仅允许进入门禁；输出生命周期/数值门禁通过才允许性能。
不得将本地CPU PASS写成“满足全部实机准入”。

## 本地检查记录

- V3测试与V2 CPU合同：94项PASS（含实际新调度器480场FIFO及Host mock）。
- Ruff检查、项目静态合同及变更台账检查通过。
- 两轮源码审查：协议源消费/目标初始化/固定收件序列/首尾排空；
  Storage持有/代次/双Rank释放不一致/重复run/完整计时/新门禁与汇总入口。
- 仓库全量`bash format.sh ci`已尝试，但环境缺少pre-commit，未完成全量hook；
  不能把定点Ruff和项目合同代称全仓格式CI通过。
- TorchNPU适配参考官方StorageDescHelper/SetDesc与cpp_extension安装include路径；
  仍必须使用现场锁定版本编译并验证，不以master源码兼容性取代实测。
