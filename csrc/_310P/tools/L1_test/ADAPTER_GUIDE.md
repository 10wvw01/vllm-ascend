# L1_test 简易算子适配器

所有适配器统一使用：`create(version, spec, ctx, collection)`，返回`version`、`run(x,w,variant)`、
`after_sample()`、`close()`。适配层只持有本版会话、输出和调用参数；公共worker负责预热、计时、
参考值、下游消费、双Rank配对和统计，不为各实验复制测试循环。

| 家族 | 实现文件 | run的工作 |
|---|---|---|
| Stock | [stock.py](adapter_plugins/stock.py) | 原生NZ F.linear，再HCCL SUM |
| V1 | [v1.py](adapter_plugins/v1.py) | 一次V1 Tensor op，显式q；本轮不安排V1测量 |
| V2 | [v2.py](adapter_plugins/v2.py) | 基础T0/D4/C3/F2、历史私有候选或所选公共API |
| V3各迭代 | [v3.py](adapter_plugins/v3.py) | 冻结V3Session/V32Session/A3Session的一次完整run/run_pool |

外部工厂继续支持`create_adapter(ctx, specification)`，用于尚未匹配ABI的早期原型与专用诊断。
Graph、MM shadow和G1取证不作为Eager完整算子混入本轮排名；小M专用核也不凭名字强行跑大M。

## 最小接口

- `ctx`提供torch、dist、linear、rank、url、stream、device、shapes；输入均已是就绪ND FP16与NZ29 FP16。
- `run(x,w,variant)`返回完整ND FP16 Y。可以异步入队，但须建立同Stream下游可安全消费的依赖。
- `variant`只表示两个输入/输出地址轮换，不是算法编号。图形模式、partial输出和Stock回退不能偷偷混入。
- `after_sample()`在计时结束及正确性检查前读取协议状态，可返回`{'host_us': {...}}`；不支持细分则返回空dict。
- `close()`完成必要同步和句柄销毁；异常通信波不自动重试、不自动回退Stock。
- 每次调用必需的padding、复制、租约/包装、seed、同步/事件都留在run；持久初始化和真实首用由公共冷态门禁单列。

统一合同`L1_TENSOR_IO_V3`只对完整MM+AR的ready边界计时。下游ADD在采样前后作为独立功能门禁：
`run`返回后立即同Stream提交ADD，最后才同步，不读取时钟或计时Event；热态循环不执行ADD。
适配器不需为消费验证提供另一套计时API。每样本的状态/完整输出检查仍保留，前后门禁缺失或失败则STOP。

## V2接口选择

`api=v2_selected_v1`调用已整理的公共入口，V2-7只跑M2048，V2-15只跑M4096。
`api=v2_trial_v1`可调用匹配T0/D4/C3/F2基础ABI。私有B/E候选仍用冻结的private_library与登记symbol/kind。
基础算子不再无条件绑定AR-only诊断符号，因此诊断OFF仍可接入。
原始输出在建池前分配，保留首写保护，热态不增加新同步。

## V3显式冻结Python会话

本轮各组使用`api=python_session_v1`，不仅冻结SO，还冻结这些Python模块：

| 家族 | python_modules键 |
|---|---|
| 原始V3/V3.1 | memfabric_mm_ar_v3 |
| A2 classic/merged | memfabric_mm_ar_v3、memfabric_mm_ar_v32 |
| A3 START/首READY/A3-2/tail | memfabric_mm_ar_v3、memfabric_mm_ar_a3 |

值为文件绝对路径，并同时列入files。加载器建立按路径与SHA隔离的私有包，
相对导入只能拿到该组冻结base；不通过全局sys.path寻找同名当前会话。
V3-0、V3-2、V3-4还要求人工核对源码/工件并登记historical_identity_verified，不用最新SO替代历史实现。
原始V3 packet=1MiB；其余本轮packet=4MiB。classic与merged标识固定，不能通过改参数把两个VN-M互换。

raw版本直接复用预分配输出；pool-return版本保留原Storage租约，不额外分配一套闲置输出。
A3-2/tail显式传variant；旧A3 START没有该参数时不强塞新接口。Host phase_log只在host阶段启用，
旧模块没有相位数据则保持UNKNOWN，不创建假的阶段耗时。

## 新增V3实验

优先复用现有session家族，只在versions.json登记版本/参数，再在campaign.py登记配置组。
若ABI确有变化，在adapter_plugins下增加一个同样的薄类；仍由同一worker计时，禁止复制一份独立统计脚本。
源码HEAD、Python、Host/Device、SDK/KVER、Tensor Storage扩展均须冻结，实际装载的库也要匹配。

[本轮矩阵与操作](ROUND_FIXED_M.md)；[委托OpenCode执行单](OPENCODE_ROUND.md)。
