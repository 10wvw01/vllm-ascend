# G1-B 日志级别注入器（实机验证版，2026-10-20 B02 会话）。
# 用法：将本文件所在目录前置到 PYTHONPATH，并设 MF_SDMA_ORCH_DIAG=1：
#   env -u MF_SDMA_ORCH_EPOCH_BUDGET_S MF_SDMA_ORCH_DIAG=1 \
#       PYTHONPATH="$(dirname $0):$PYTHONPATH" GATE_E_DURATION_S=620 \
#       bash gate_e_forensic_run.sh fused <会话编号>
#
# 背景：飞行记录器（hybm_sdma_orchestrator.cpp SupervisorLoop）的 [MF_DIAG]
# 行全部经 BM_LOG_INFO 输出，而 hybm core 的 OutLogger 单例（各 .so 独立，
# 头文件内联实现）默认 WARN_LEVEL，会静默过滤 INFO——不提级则 G1-B 整轮无效
# （serve.log 中无任何 [MF_DIAG]，且极易误判为"诊断库未加载"）。
# 库内无环境变量覆盖路径（MF_LOG_LEVEL 仅被 smem 侧解析），故经 PYTHONPATH
# 注入本钩子：ctypes 加载与 memfabric_hybrid 包 __init__.py 完全相同的绝对
# 路径（SONAME 命中同一动态库实例，RTLD_GLOBAL 同包语义），调用导出符号
# hybm_set_log_level(1)。仅 MF_SDMA_ORCH_DIAG=1 的进程激活，其余进程零影响。
# 生效确认：serve.log 应出现 "[G1B] sitecustomize: hybm_set_log_level(1) rc=0"
# 及 "[MF_DIAG] start epoch: 1 ..."；二者缺一则该轮无效（前者=注入失败，
# 后者=诊断库未加载或未被触发）。
import os
import sys

if os.environ.get("MF_SDMA_ORCH_DIAG") == "1":
    try:
        import ctypes

        _LIB = "/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid/lib/libmf_hybm_core.so"
        _h = ctypes.CDLL(_LIB, mode=ctypes.RTLD_GLOBAL)
        _h.hybm_set_log_level.argtypes = [ctypes.c_int32]
        _h.hybm_set_log_level.restype = ctypes.c_int32
        _rc = _h.hybm_set_log_level(1)  # 0=DEBUG 1=INFO 2=WARN(默认) 3=ERROR 4=FATAL
        print(f"[G1B] sitecustomize: hybm_set_log_level(1) rc={_rc} lib={_LIB}", file=sys.stderr, flush=True)
    except Exception as _e:  # noqa: BLE001
        print(f"[G1B] sitecustomize FAILED: {_e!r}", file=sys.stderr, flush=True)
