# E3 Bugfix 1：E3-T首版tiling运行时L0C满载AICore异常

- run_id: e3-chunk2048-20261028-001（首次campaign尝试）
- 时间：2026-10-28 00:52~00:55 (+08)
- 现象：E3-G双Rank Smoke全部bit-exact PASS后，E3-T Smoke在首个真实kernel
  （warmup i=0）即双Rank对称失败，SIGABRT，错误码507015
  （rtStreamSynchronizeWithTimeout aicore exception, retCode=0x26,
  failure_mode=1, PreCheckTaskErr, fault kernel
  `E3Group2048Kernel<e3Tiling2048, 2048>`，blockDim=8）。
- 根因：dav-2002 L0C总计256KB。首版e3Shape2048={256,kN,kK,256,256,64}
  的baseM=256×baseN=256 FP32累加器恰为256KB=100%占满L0C且零余量
  （L0A双缓冲同时64KB=100%满载）。BiSheng编译期资源检查未拦截，
  运行期AICore触发异常。对照G模式tiling1024 base tile 128×256×4B=128KB
  （50%），且E3-G Smoke同日同卡同SO全部通过，证明组同步/双包/Add
  逻辑无问题，异常隔离在T的MM tiling。
- 修复：e3Shape2048改为{256,kN,kK,128,256,64}。singleCoreM=256保持不变，
  T模式语义保持"每组每核一次256行MM调用（mm.SetOrgShape(256,..)+
  IterateAll单次发射）"；仅内部基础tile的basicM回落到与已验证
  tiling1024完全一致的128行几何，L0C占用128KB(50%)、L0A 32KB、
  L0B 64KB均为已实机验证水平。不改变G模式、双4MiB分包、ReadyTree、
  Add2048或任何同步协议。
- 影响：仅E3-T内核tiling常量与说明注释、UT合同新增几何锁定断言。
  UT 71项全数通过；重建后manifest的work_head与源SHA同步更新。
- 证据文件：本目录progress.txt / smoke_e3t.log（原始torchrun日志）、
  plog_rank{0,1}_aicore_excerpt.log（device侧RUNTIME/IDEDD异常摘录）、
  smoke_e3g.log（对照：G在同批次通过）。
