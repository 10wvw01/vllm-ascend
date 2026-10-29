# A3-2与A3-tail：连续行固定座位调度

## 布局与就绪合同

沿用A3完整S、独立双槽Y、ND激活与NZ权重。S/Y是ND行主序，不是NZ输出；
K=N=2048、FP16，每行4096字节。第r行地址为base+r×4096，连续h行占h×4096字节。
通信批量512/1024/2048行，对应2/4/8MiB；4MiB包意味着8MiB批次拆成两个ADD包。
每个MM座位计算全部K、全部2048列，不能把未完成的K累加中间量当成整行就绪。

每座位按次序完成MM→S可见→seed S到本地Y→Y可见→发布epoch标志。
Core0确认该批所有座位齐全才发READY。双方相同范围READY及首帧校验通过后才ADD S→peerY。
因此“连续”是地址无空洞且数据全部就绪，不要求不同核按行号顺序完成。
S全波只写一次，无逐chunk源回收；最终Drain/quiet/DONE（无最终quiet）和核间完成放行不变。

## 三个独立候选

| 候选 | profile | 核分工 | Python variant |
|---|---|---|---|
| 原A3 | 3302 | 8核MM，Core0兼通信，原实现保留 | a3 |
| A3-2 | 3303 | Core1～7计算全部座位，Core0专职通信 | a3-2 |
| A3-tail | 3304 | Core1～7计算前段，Core0计算末尾约1/8座位并调度通信 | a3-tail |

compute_rows仍取512/1024，表示原8核计算单元；实际每核MM tile为64/128行，
新候选不将它误解成7核共同计算512/1024行。座位总数bands=M/tile_rows。
A3-2按band=core-1、步长7分配；Core0在Matmul注册与计算缓冲初始化前走独立分支。
A3-tail给Core0末尾ceil(bands/8)个座位，其余前段由7核同样步长分配。

每座位独占64B epoch标志，位于既有IB分配第二半；最多64座位、4096字节，
不增加Host热态清零/任务或内存分配。复用依赖严格递增epoch、同流和上一波最终完成。
公开create新增mf310p_a3_2_create/mf310p_a3_tail_create；reserve/run公共admit仍为0/1。
内部Host→device的admit参数低位为admit、高位为role（0原A3、1专职、2末尾），
不改变导出函数签名；两端首READY profile不同会拒绝，Host/Device必须同版构建。

## A3-tail调度及限制

Core0先等待前段首批，提交ADD后计算一个末尾座位，再调度下一批。
如果下一批已覆盖末尾区，先补齐该批属于自己的所有座位，再等待工作核；
这样不会发生Core0等自己尚未计算的数据。每次Submit仍使用现有阻塞FIFO进度器，
对端落后时Core0可能等待；这不是方案B的公共非阻塞收件接口。

末尾排布消除前段因Core0兼任通信产生的座位空洞，但不保证更快：首批由7核生成，
后续Core0的MM会延迟再次调度；末尾批次仍依赖Core0。M2048/comm2048只有一个批次，
必须整矩阵完成才提交，无法用该粒度实现前段交叠。

7核负载取整需要实测：M2048、tile128共16座位，最忙核由2增为3；
M4096、tile128共32座位，由4增为5。tile64时分别32/64座位，由4/8增为5/10。
这些是座位数量，不是预测耗时；不能仅以8/7推断性能代价。

## 同批实机合同

保持MemFabric db38d75e8485d31451302219eb7a1bfeb1dd107c、相同HEAD/SDK/Host与Device SO、
4MiB包及同卡双Die独占窗口。先完成A3_ADMISSION既有私有构建、Storage冻结、G0及六门禁。
新增kernel需核对dav-2002编译、Matmul注册/UB/同步资源、64B标志与跨核/SDMA可见性。

一次compare_a3批次顺序：compute512四会话，再compute1024四会话；每组顺序
A3-before→A3-2→A3-tail→A3-after。每会话六配置M2048/4096×comm512/1024/2048，
每配置两轮、每轮4预热+30样本，复用quick既有六门禁，不另跑旧V2/Stock性能臂。

```bash
python csrc/_310P/memfabric_mm_ar_v3/tools/compare_a3.py start \
  --manifest <frozen-build>/manifest.json --out <fresh-batch-dir> \
  --card 38144 --devices 4,5 --lease-note '<现场覆盖整批的独占租约>'
python csrc/_310P/memfabric_mm_ar_v3/tools/compare_a3.py status --out <fresh-batch-dir>
```

每会话由quick守护，连续会话间的独占须由现场租约保证；工具不声称进程间隙仍有连续守护。
任一异常STOP整批、不自动重试ADD、不自动换卡。执行时间超限仍需保存现场。
报告校验八会话、variant/tiling身份及每场两轮六配置完整性。
每轮先逐样本取双Rank max，再取p50；比较两轮p50中位数。
以A3前后中位值为同批基线，前后漂移或任一候选两轮漂移>5%则该case不判稳定收益。
完整窗口包含本地Storage/事件、seed、MM、所有通信及排空，同时保留Host Wall。
历史Stock比较仍为跨批方向参考；此短测不签正式Stock性能GO。

## 审核与验证边界

第一轮审核地址与依赖：所有座位恰好一个生产者，完整S不可覆盖，Y在READY前已seed；
专职Core0不生产座位，tail Core0不会等待自己的未生产座位，最多64标志不越界。
第二轮审核接口与测试：variant实际传入Session而不只记入报告；profile互斥；
公共admit与内部role分别校验；失败首READY、生命周期、最终排空沿用原合同。
CPU随机调度覆盖512/1024/2048/3072/4096、两种tile及全部可整除通信批量。
Host mock实际编译验证三个profile热态无新增同步/集合/ACL memset及失败admit编码。

本地无CANN/BiSheng、TorchNPU和NPU，设备编译、缓存可见性、数值与性能均NOT_RUN。
CPU逻辑通过不能替代设备准入；候选实现可以交OpenCode先编译再门禁，尚无设备GO。

本轮本地验证：V3+V2 UT共132项PASS，项目治理38项PASS，Ruff与静态项目检查PASS。
修复既有phase_log测试对纳秒转浮点求和的精度敏感断言（不改生产计时）。
全仓format.sh ci因缺少pre-commit未完成，不宣称全仓CI通过。

发布前同步远程a762514a735c83c71e0be46c319cf811ff9565a3（旧A的L2分段计时），
保留其改动后全组回归135项PASS，增量项目台账检查PASS。
