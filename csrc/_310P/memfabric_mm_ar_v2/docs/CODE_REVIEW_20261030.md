# V2 所选代码包复核记录（2026-10-30）

基底：项目主线56485231a0dde5fa83a461e5810954d2e8dc48ea。
交付固定M公共入口、独立构建与L1调用；V1查历史按用户指令延期到下一次实机。

## 第一轮：设备参数与协议复核

- E1、E4、共同头及D4/基础Host/内部结构保持原文，设备源逐SHA与原始计时manifest核对。
- E1固定M2048、pool chunk1024、device chunk1024、chunks2、overlap1。
- T8固定M4096、pool chunk1024、device group2048、groups2、variant2（不是旧Host kind6）。
- rank0/rank1 ownC、peerC、peerDst、expectedRecvDst沿用原私有Host映射；64B status、IB、ack/credit均传递。
- Cache维护、Ready、wait/release、ADD、quiet和DONE保留；新Host未新增逐调用集合或已热输出同步。
- 每句柄固定版本/形状/Stream，2049～4095在建池前拒绝；不造未经测量的区间选择阈值。

## 第二轮：Host生命周期、构建和测量复核

- 不再依赖test/perf私有Host；新Host与基础池代码在同一扩展编译，未改基础trial结构ABI。
- 预热/输出地址/poison状态随句柄持有和销毁，不使用全局裸指针哈希表。
- 保留两次空符号预热、首次真实/新输出设备同步；输出存储生命周期仍由调用方保证。
- launch/清零/同步/探针/状态失败注入后禁止继续run；退出Stream同步失败不销毁仍可能使用的池。
- CMake按仓库3.16最低版本使用显式源码目录参数；实际CPU配置使用CMake4.4.4。
- selected和diagnostics开关四组合均实际配置生成构建图；旧D4源与所选Device库分开，未混入E2/E3或私有Host。
- L1通过api=v2_selected_v1显式选择；仍沿用V2-7/V2-15编号，历史ABI默认不换。
- L1在建池前分配输出；完整计时/采样/统计与下游消费代码不变；实际装载selected Device也须在冻结files内。

## 验证结果

```text
python -m pytest -q csrc/_310P/memfabric_mm_ar_v2/test \
  csrc/_310P/memfabric_mm_ar_v3/test tools/310p_mm_ar csrc/_310P/tools/L1_test
450 passed, 36 subtests passed
```

其中新增20项：12个真实Host源码+fake ACL/launch的C++故障/生命周期用例，
4个实际CMake构建图组合，4个L1新API身份/分配时机/调用合同测试。
未执行格式检查。

## 实机与封存边界

CPU假后端不执行NPU。新组合BiSheng编译、二进制装载、双Rank正确性/首写/下游消费均NOT_RUN。
下一设备窗口按新源码SHA构建并核验新入口；历史E1/T8性能证据继续复用，本轮不安排重跑。
原有Stock默认、旧Serving/D4调用与所有历史实验保留；源码封存不是生产默认启用或新二进制性能GO。
