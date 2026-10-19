# 310P3 V2 Ascend C 直调工程源码清单。
# 由上层 vllm_ascend CMake 集成加载；本项目
# 不是常规 ACLNN 自定义算子包，形状及内核注册与
# Torch 绑定归原有 vllm-ascend 扩展负责，强行改成
# npu_op_package() 会改变 ABI 并使已验证结果失效。
#
# 遵循 CANN 标准的工程目录分层：
# op_kernel/：Ascend C 设备内核，由 bisheng 编译至 dav-2002。
# op_host/：MemFabric 公共 Host API 及 ABI 声明。
# torch_adapter/：供 vLLM 使用的 PyTorch/Torch Library 接入层。
set(MF310P_V2_KERNEL_SOURCE
    "${CMAKE_CURRENT_LIST_DIR}/op_kernel/mf310p_v2_trial_device.asc")
set(MF310P_V2_HOST_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/op_host/mf310p_v2_trial_adapter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/torch_adapter/mf310p_v2_serve.cpp")
set(MF310P_V2_HOST_INCLUDE_DIRS
    "${CMAKE_CURRENT_LIST_DIR}/op_host"
    "${CMAKE_CURRENT_LIST_DIR}/torch_adapter")
