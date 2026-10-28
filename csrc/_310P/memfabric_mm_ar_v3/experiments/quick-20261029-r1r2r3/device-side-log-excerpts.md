# AICPU 设备侧调度器日志关键行摘录（来源 /root/ascend/log/run/device-*/，服务器路径不随仓库分发）

## R1 rank0 (device-4, device-1877661_20261028172341289.log)：kfc_min.so 缺失 → 探测任务 0x2a
    [INFO] CCECPU(28325,aicpu_scheduler):2026-10-29-01:23:44.016.834 [ae_so_manager.cc:664][CreateSingleSoMgr][tid:28364][AICPU_PROCESSER] So not exist, try to load at another path. currentSoFile=/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/kfc_min.so.
    [INFO] CCECPU(28325,aicpu_scheduler):2026-10-29-01:23:44.016.862 [ae_so_manager.cc:724][GetSoInHostPidPath][tid:28364][AICPU_PROCESSER] So not exist, try to load at another path. currentSoFile=/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/3316127_0/kfc_min.so.
    [WARNING] CCECPU(28325,aicpu_scheduler):2026-10-29-01:23:44.016.884 [ae_so_manager.cc:680][CreateSingleSoMgr][tid:28364][AICPU_PROCESSER] Single so manager init failed, soFile is /usr/lib64/aicpu_kernels/0/aicpu_kernels_device/3316127_0/kfc_min.so.
    [WARNING] CCECPU(28325,aicpu_scheduler):2026-10-29-01:23:44.016.893 [ae_so_manager.cc:357][GetApi][tid:28364][AICPU_PROCESSER] Load so kfc_min.so failed.
    [WARNING] CCECPU(28325,aicpu_scheduler):2026-10-29-01:23:44.016.901 [ae_kernel_lib_aicpu_kfc.cpp:95][GetApiWhenSonameNotEmpty][tid:28364][AICPU_PROCESSER] Get system api from kfc_min.so unsuccessfully.

## 历史佐证：2026-09-18 以来全部设备日志 kfc_min.so 加载 2263 次失败 0 成功
    grep -rh 'Get system api from kfc_min.so unsuccessfully' /root/ascend/log/run/device-*/*.log | wc -l == 2263

## R2 dev5 helper (device-5, device-1911603_20261028180231538.log, hostPid=3409468)：
## aicpu scheduler 启动于 18:02:28.84，probe 任务 18:02:29.27，无 CUST 写入记录（18:02:31.59 被守护器 SIGTERM）
    [INFO] CCECPU(4615,aicpu_scheduler):2026-10-29-02:02:29.270.440 [ae_so_manager.cc:664][CreateSingleSoMgr][tid:4663][AICPU_PROCESSER] So not exist, try to load at another path. currentSoFile=/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/kfc_min.so.
    [INFO] CCECPU(4615,aicpu_scheduler):2026-10-29-02:02:29.270.470 [ae_so_manager.cc:724][GetSoInHostPidPath][tid:4663][AICPU_PROCESSER] So not exist, try to load at another path. currentSoFile=/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/3409468_1/kfc_min.so.
    [WARNING] CCECPU(4615,aicpu_scheduler):2026-10-29-02:02:29.270.492 [ae_so_manager.cc:680][CreateSingleSoMgr][tid:4663][AICPU_PROCESSER] Single so manager init failed, soFile is /usr/lib64/aicpu_kernels/0/aicpu_kernels_device/3409468_1/kfc_min.so.
    [WARNING] CCECPU(4615,aicpu_scheduler):2026-10-29-02:02:29.270.502 [ae_so_manager.cc:357][GetApi][tid:4663][AICPU_PROCESSER] Load so kfc_min.so failed.
    [WARNING] CCECPU(4615,aicpu_scheduler):2026-10-29-02:02:29.270.510 [ae_kernel_lib_aicpu_kfc.cpp:95][GetApiWhenSonameNotEmpty][tid:4663][AICPU_PROCESSER] Get system api from kfc_min.so unsuccessfully.

## R3 dev5 helper (device-5, device-1916356_20261028181457640.log, hostPid=3437351)：
## 仅一次 probe 任务解析（test -f 命中，rc=3 already-present 快路径），无部署活动
    [INFO] CCECPU(23730,aicpu_scheduler):2026-10-29-02:14:55.959.453 [ae_so_manager.cc:664][CreateSingleSoMgr][tid:23776][AICPU_PROCESSER] So not exist, try to load at another path. currentSoFile=/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/kfc_min.so.
    [INFO] CCECPU(23730,aicpu_scheduler):2026-10-29-02:14:55.959.481 [ae_so_manager.cc:724][GetSoInHostPidPath][tid:23776][AICPU_PROCESSER] So not exist, try to load at another path. currentSoFile=/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/3437351_1/kfc_min.so.
    [WARNING] CCECPU(23730,aicpu_scheduler):2026-10-29-02:14:55.959.504 [ae_so_manager.cc:680][CreateSingleSoMgr][tid:23776][AICPU_PROCESSER] Single so manager init failed, soFile is /usr/lib64/aicpu_kernels/0/aicpu_kernels_device/3437351_1/kfc_min.so.
    [WARNING] CCECPU(23730,aicpu_scheduler):2026-10-29-02:14:55.959.514 [ae_so_manager.cc:357][GetApi][tid:23776][AICPU_PROCESSER] Load so kfc_min.so failed.
    [WARNING] CCECPU(23730,aicpu_scheduler):2026-10-29-02:14:55.959.523 [ae_kernel_lib_aicpu_kfc.cpp:95][GetApiWhenSonameNotEmpty][tid:23776][AICPU_PROCESSER] Get system api from kfc_min.so unsuccessfully.

## 参照（生产验证基线, 2026-10-23 d4_2 会话, device-6, device-98292_20261023153825967.log）：
## 纯 ACL helper 内 KFC system() 真实执行 + CUST 迁移落盘 13960B（v11）
    [INFO] CCECPU(28586,aicpu_scheduler):2026-10-23-23:38:24.095.679 [aicpusd_cust_so_manager.cpp:321][WriteBufToSoFile][tid:28615] Write buffer to so success, path=/home/CustAiCpuUser/lib/libmf_sdma_orch.so.0.0.4167660, hash=12714778748295993450, fileSize=13960
    [INFO] CCECPU(28586,aicpu_scheduler):2026-10-23-23:38:24.095.789 [aicpusd_cust_so_manager.cpp:360][CreateSoftLinkToSoFile][tid:28615] Create soft link so success, linkPath=/home/CustAiCpuUser/cust_aicpu_0_0_4167660/libmf_sdma_orch.so, targetPath=/home/CustAiCpuUser/lib/libmf_sdma_orch.so.0.0.4167660
