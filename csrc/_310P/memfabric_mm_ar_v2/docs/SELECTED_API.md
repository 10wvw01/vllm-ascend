# V2 所选算子的公共接口与使用

正式选型为V2-7/E1（M2048）与V2-15/E4-T8（M4096），公共C接口ABI1。
[代码调用链与计算流程](CODE_MAP.md)、[源码包身份](../../releases/v1-v2-20261030/V2_CODE_PACKAGE.json)。
接口整理不改已有设备核，不把旧SO的成绩登记为新编译SO实测成绩。

## 构建

沿用项目原来的310P MemFabric/V2编译开关、工具链和SDK。V2构建现在额外包含：

- 扩展中的`mf310p_v2_selected_*` Host符号；
- 独立`libmf310p_v2_selected_device.so`，源为原E1、E4文件与共同头；
- 原`libmf310p_v2_trial_device.so`仍负责基础探针、D4及可选历史诊断。

`-DVLLM_ASCEND_310P_V2_BUILD_SELECTED=OFF`可关闭新增公共入口，回到旧构建集合。
`VLLM_ASCEND_310P_V2_BUILD_DIAGNOSTICS`独立控制诊断，不影响selected的可用性。
没有新增环境变量，也不默认启动设备任务、部署SDK或变更Serving路由。

核验实际扩展的selected Host符号和两个Device SO装载路径，冻结新SO SHA、SDK/KVER11与物理Die。
构建/短功能准入留待下次设备窗口；当前没有CANN编译或NPU PASS，不以本次整理要求重跑旧性能实验。

## C接口与生命周期

包含`src/op_host/mf310p_v2_selected_api.h`，调用方已经选择正确ACL设备并创建Stream。
以下为调用顺序，所有返回码必须检查：

```cpp
auto check = [](int rc) {
    if (rc != 0) throw std::runtime_error("V2 failed; stop both workers");
};
mf310p_v2_selected_plan_t plan{};
check(mf310p_v2_selected_get_plan(m, &plan)); // CPU，仅2048或4096
mf310p_v2_selected_t* handle = nullptr;
check(mf310p_v2_selected_create(rank, url, m, &handle)); // rank=0/1
check(mf310p_v2_selected_ib_probe(handle, 64, stream));
check(mf310p_v2_selected_run(handle, x, w_nz, y, stream)); // 已入队
// 同Stream可直接提交下游读取；Host校验前同步Stream。
check(aclrtSynchronizeStream(stream));
uint64_t protocol = 0;
check(mf310p_v2_selected_check_status(handle, &protocol));
if (protocol != 0) throw std::runtime_error("V2 protocol failed; stop both workers");
check(mf310p_v2_selected_destroy(handle)); // 完成确认后才回收
handle = nullptr; // 仅在destroy成功后清除
```

每个句柄固定一个M、版本和Stream；两Rank使用相同配置与顺序，多个活跃句柄使用不同rendezvous URL。
只允许串行Host调用，不支持并发、多Stream或Graph capture。输入输出缓冲区在建池前分配/初始化并保持存活，
W确为NZ29，X/Y为连续ND FP16；设备地址、容量、无别名由调用方保证。Y为完整TP2 SUM，不是局部结果。
输出的“入队返回”不是Host ready；同Stream下游按依赖消费，Host访问需先等待完成。

| 接口 | 工作 | 同步边界 |
|---|---|---|
| get_plan | 查询固定M/版本/MM/通信/ADD粒度 | CPU，无设备副作用 |
| create | 基础池/对称布局及通信初始化 | 冷态，可有集合/同步 |
| ib_probe | 既有IB探针，绑定唯一Stream | 冷态，完成后才能run |
| run | 首用保护、status清零及完整融合核入队 | 首用/新输出保留设备同步；热态无新增同步 |
| check_status | 返回协议状态，失败poison句柄 | 调用方先完成Stream |
| destroy | 确认完成/协议后销毁池与本句柄状态 | Stream同步，失败不释放资源 |

任何backend错误/非零协议状态都终止本轮，后续run拒绝，不自动重试通信波。
仅指针不变不能证明释放重分配后的存储仍被预热；不要在句柄生命周期内释放并复用已登记输出存储。

## 统一L1调用

现有V2-7/V2-15登记保留历史私有ABI。新入口通过runtime的`api: "v2_selected_v1"`显式选择；
不能偷偷用新API/新SO替换历史工件身份。模板为
[runtime-v2-selected-template.json](../../tools/L1_test/runtime-v2-selected-template.json)。

每份计划只指定一个匹配M：

```bash
python csrc/_310P/tools/L1_test/l1_bench.py plan --versions V2-7 --M 2048 \
  --warmup 20 --samples 100 --rounds 2 --out /tmp/v2-e1-plan.json
python csrc/_310P/tools/L1_test/l1_bench.py plan --versions V2-15 --M 4096 \
  --warmup 20 --samples 100 --rounds 2 --out /tmp/v2-t8-plan.json
```

以上只生成CPU计划，不执行实机。实际执行仍须冻结准入manifest；模板含NOT_RUN和占位路径，不能直接运行。
`library`为新扩展，`selected_device_library`必须位于冻结files中；文件清单还须包含实际映射的基础Device及SDK依赖。
框架拒绝错版本/M、额外kind配置、历史private_library混入或缺失selected Device身份。

计时继续采用统一ready/chain边界；持久输出建池前分配，首写及路由预热单列冷态，run内必需清零与提交都计时。
本次只完成入口整理和CPU验证；不运行上述性能计划，不把V1历史查找作为V2前置。
