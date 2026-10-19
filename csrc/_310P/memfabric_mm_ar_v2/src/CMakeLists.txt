# 310P3 V2 Ascend C direct-invoke source manifest.
# Loaded from the parent vllm_ascend cmake integration. This project is
# NOT a conventional aclnn custom-op package: shape/kernel registration and
# torch binding are owned by the existing vllm-ascend extension. Turning this
# into npu_op_package() would change the ABI and invalidate hardware results.
#
# Standard CANN engineering split:
#   op_kernel/    Ascend C device kernel, bisheng dav-2002
#   op_host/      MemFabric public host API bridge + ABI declaration
#   torch_adapter/ PyTorch/Torch Library vLLM-facing entrypoint
set(MF310P_V2_KERNEL_SOURCE
    "${CMAKE_CURRENT_LIST_DIR}/op_kernel/mf310p_v2_trial_device.asc")
set(MF310P_V2_HOST_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/op_host/mf310p_v2_trial_adapter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/torch_adapter/mf310p_v2_serve.cpp")
set(MF310P_V2_HOST_INCLUDE_DIRS
    "${CMAKE_CURRENT_LIST_DIR}/op_host"
    "${CMAKE_CURRENT_LIST_DIR}/torch_adapter")
