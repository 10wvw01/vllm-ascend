#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""V2 同步 vs 连续异步提交，310P3 TP=2 算子级性能价值验证。

每个进程只测一个 arm：stock 或 v2。两边独立新进程，避免在计时
阶段交替创建/释放通信资源。Stock 使用 vLLM 当前 unquantized_gemm
+ TP=2 FP16 HCCL；V2 使用已经公开的 trial_create/run ABI。

所有输入/权重、V2 输出槽均在 create 前分配；通信建池只执行一次。
每个窗口至多使用 window 个输出槽，在窗口末同步之后才允许复用。
40 次独立目标层调用是连续提交压力包络，并非真实 Qwen 40 层
依赖图；真实模型的层间数据依赖不能被这份脚本假装消失。
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C as extension
from vllm_ascend.ops import linear as _linear_registration  # noqa: F401
from vllm_ascend.utils import maybe_trans_nz

WIDTH = 2048


def fail_if_not(cond: bool, reason: str) -> None:
    if not cond:
        raise RuntimeError(reason)


def checked(rc: int, name: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{name} rc={rc}")


def quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    fail_if_not(bool(ordered), "empty samples")
    n = len(ordered)
    return {
        "count": n,
        "min_ms": round(ordered[0], 5),
        "median_ms": round(statistics.median(ordered), 5),
        "p95_ms": round(ordered[min(n - 1, (95 * n + 99) // 100 - 1)], 5),
        "max_ms": round(ordered[-1], 5),
    }


def emit(rank: int, phase: str, **data: object) -> None:
    # torchrun 下两个 Rank 共享同一个 stdout 文件描述符；print 在非缓冲
    # stdout 上会把文本与换行拆成两次 write，跨进程交错后两条 JSON 记录
    # 会挤在同一物理行。含换行的单次 os.write 使每条记录原子落盘。
    line = json.dumps({
        "tag": "V2_SYNC_ASYNC", "rank": rank, "phase": phase, **data
    }, ensure_ascii=False) + "\n"
    sys.stdout.flush()
    os.write(sys.stdout.fileno(), line.encode("utf-8"))


def bind_v2():
    lib = ctypes.CDLL(extension.__file__)
    P, U32, U64 = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64
    signatures = {
        "mf310p_v2_trial_create": ([ctypes.c_int, ctypes.c_char_p, U32, ctypes.POINTER(P)], ctypes.c_int),
        "mf310p_v2_trial_ib_probe": ([P, U32, P], ctypes.c_int),
        "mf310p_v2_trial_run": ([P, U64, U64, U64, U32, U32, P], ctypes.c_int),
        "mf310p_v2_trial_check_status": ([P, ctypes.POINTER(U64)], ctypes.c_int),
        "mf310p_v2_trial_destroy": ([P], ctypes.c_int),
    }
    for name, (argtypes, restype) in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = argtypes, restype
    return lib


def stock_forward(x: torch.Tensor, w: torch.Tensor):
    # 与 Serving 的 AscendUnquantizedLinearMethod.apply() 完全相同的入口。
    partial = torch.ops.vllm.unquantized_gemm(x, w, None)
    work = dist.all_reduce(partial, op=dist.ReduceOp.SUM, async_op=True)
    return partial, work


class V2Context:
    def __init__(self, lib, rank, raw_stream, rows, chunk, store, outputs, x, w,
                 refs, ctrl):
        self.lib = lib
        self.handle = ctypes.c_void_p()
        self.raw_stream = raw_stream
        self.rows = rows
        self.outputs = outputs
        checked(lib.mf310p_v2_trial_create(
            rank, store.encode(), chunk, ctypes.byref(self.handle)), "trial_create")
        fail_if_not(bool(self.handle.value), "null V2 context")
        checked(lib.mf310p_v2_trial_ib_probe(
            self.handle, 1000, raw_stream), "ib_probe")
        # 每个输出地址的首写保护可能触发 0-15 秒 device sync。
        # 分槽交替 Rank 会合，避免首写保护形成双 Rank 过大偏斜。
        for slot, y in enumerate(outputs):
            dist.barrier(group=ctrl)
            self.launch(x[slot % len(x)], w[slot % len(w)], y)
            torch.npu.current_stream().synchronize()
            self.check()
            if not torch.equal(y, refs[slot % len(refs)]):
                raise AssertionError(f"V2 warmup slot={slot} not bit-exact")
            dist.barrier(group=ctrl)

    def launch(self, x, w, y):
        checked(self.lib.mf310p_v2_trial_run(
            self.handle, x.data_ptr(), w.data_ptr(), y.data_ptr(),
            self.rows, 1, self.raw_stream), "trial_run")

    def check(self):
        status = ctypes.c_uint64(0)
        checked(self.lib.mf310p_v2_trial_check_status(
            self.handle, ctypes.byref(status)), "check_status")
        fail_if_not(status.value == 0, f"V2 status={status.value}")

    def close(self):
        if self.handle.value:
            torch.npu.current_stream().synchronize()
            checked(self.lib.mf310p_v2_trial_destroy(self.handle), "trial_destroy")
            self.handle = ctypes.c_void_p()


def execute_window(arm, context, x_list, w_list, outputs, expected, start,
                   stop, verify):
    works = []
    pending = []
    t_launch = 0.0
    for index in range(start, stop):
        a = time.perf_counter()
        if arm == "stock":
            y, work = stock_forward(x_list[index], w_list[index])
            works.append(work)
        else:
            y = outputs[index - start]
            context.launch(x_list[index], w_list[index], y)
        t_launch += time.perf_counter() - a
        pending.append((index, y))
    # Work.wait 是 HCCL->当前计算流的必要依赖。只有窗口末允许同步。
    for work in works:
        work.wait()
    torch.npu.current_stream().synchronize()
    if verify:
        if context is not None:
            context.check()
        for index, y in pending:
            if not torch.equal(y, expected[index]):
                bad = int(torch.count_nonzero(y != expected[index]).item())
                raise AssertionError(
                    f"{arm}: rank={dist.get_rank()} index={index} mismatch={bad}")
    # CPU Tensor 引用在窗口内一直有效；退出本函数后方可复用。
    del pending, works
    return t_launch * 1000.0


def run_sequence(arm, context, x_list, w_list, outputs, expected, window,
                 verify):
    n = len(x_list)
    start_wall = time.perf_counter()
    enqueue_ms = 0.0
    for start in range(0, n, window):
        enqueue_ms += execute_window(
            arm, context, x_list, w_list, outputs, expected, start,
            min(n, start + window), verify)
    return (time.perf_counter() - start_wall) * 1000, enqueue_ms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("stock", "v2"))
    parser.add_argument("--rows", type=int, choices=(2048, 3072, 4096), default=4096)
    parser.add_argument("--chunk-rows", type=int, choices=(256, 512, 1024), default=1024)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4400)
    parser.add_argument("--store-url", default=os.environ.get(
        "VLLM_ASCEND_310P_MEMFABRIC_STORE_URL", "tcp://127.0.0.1:8581"))
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        parser.error("必须 torchrun --nproc-per-node=2")
    if (not 1 <= args.steps <= 40 or not 1 <= args.repeat <= 100
            or not 1 <= args.warmup <= 30 or not args.windows
            or any(not 1 <= v <= min(args.steps, 16) for v in args.windows)
            or len(set(args.windows)) != len(args.windows)
            or args.rows % args.chunk_rows != 0):
        parser.error("1<=steps<=40; windows 唯一且 <=min(steps,16); 行数须整除 chunk")
    if os.getenv("MF_KFC_BYPASS") == "1":
        parser.error("禁止使用 MF_KFC_BYPASS 规避控制面错误")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group(backend="hccl")
    ctrl = dist.new_group(backend="gloo")
    stream = torch.npu.current_stream()
    raw_stream = ctypes.c_void_p(stream.npu_stream)
    # 避免导入了机器上另一份已安装的旧扩展造成伪性能结果。
    expected_repo = os.getenv("V2_SYNC_ASYNC_REPO_ROOT")
    if expected_repo:
        actual_so = str(Path(extension.__file__).resolve())
        root = str(Path(expected_repo).resolve())
        fail_if_not(os.path.commonpath([actual_so, root]) == root,
                    f"wrong extension origin: {actual_so}; expected under {root}")
    emit(rank, "environment", arm=args.arm, rows=args.rows,
         steps=args.steps, windows=args.windows,
         chunk_rows=args.chunk_rows, extension=str(Path(extension.__file__).resolve()),
         expansion_mode=os.getenv("HCCL_OP_EXPANSION_MODE", "<unset>"))
    ctx = None
    try:
        # 所有 Tensor 在 V2 建池之前分配，符合历史 MTE3 首写安全约束。
        # 40 层不同输入/权重保证误复用输出不能蒙混通过正确性阶段。
        x_list = []
        w_list = []
        for layer in range(args.steps):
            torch.manual_seed(args.seed + rank * 103 + layer * 17)
            x_list.append(torch.randn(args.rows, WIDTH, dtype=torch.float16, device=device))
            w_list.append(maybe_trans_nz(torch.randn(
                WIDTH, WIDTH, dtype=torch.float16, device=device)))
        outputs = [torch.empty((args.rows, WIDTH), dtype=torch.float16, device=device)
                   for _ in range(max(args.windows))]
        # FP32 SUM参考，必须在池创建前构造；不计入基准。
        expected = []
        for x, w in zip(x_list, w_list):
            ref = F.linear(x, w).float()
            dist.all_reduce(ref, op=dist.ReduceOp.SUM)
            expected.append(ref.half())
        torch.npu.synchronize()
        dist.barrier(group=ctrl)

        if args.arm == "v2":
            lib = bind_v2()
            ctx = V2Context(lib, rank, raw_stream, args.rows, args.chunk_rows,
                            args.store_url, outputs, x_list, w_list, expected, ctrl)
        for window in args.windows:
            # 先通过全部 40 调用逐窗口位精确测试，再进行无核验计时。
            dist.barrier(group=ctrl)
            elapsed_ms, _ = run_sequence(
                args.arm, ctx, x_list, w_list, outputs, expected, window, True)
            emit(rank, "correctness", arm=args.arm, rows=args.rows, window=window,
                 steps=args.steps, pass_bit_exact=True,
                 validation_wall_ms=round(elapsed_ms, 4))
            for _ in range(args.warmup):
                run_sequence(args.arm, ctx, x_list, w_list, outputs, expected, window, False)
            if ctx is not None:
                ctx.check()  # 预热结束后验状态；不能计入稳态耗时
            dist.barrier(group=ctrl)
            wall_samples = []
            enqueue_samples = []
            for _ in range(args.repeat):
                wall, enqueue = run_sequence(
                    args.arm, ctx, x_list, w_list, outputs, expected, window, False)
                wall_samples.append(wall)
                enqueue_samples.append(enqueue)
            # 计时窗口结束后再次对所有步执行一次真实性校验。
            run_sequence(args.arm, ctx, x_list, w_list, outputs, expected, window, True)
            emit(rank, "result", arm=args.arm, rows=args.rows,
                 chunk_rows=args.chunk_rows, steps=args.steps, window=window,
                 warmup=args.warmup, repeat=args.repeat, validated=True,
                 wall=quantiles(wall_samples), enqueue=quantiles(enqueue_samples),
                 wall_per_call_median_ms=round(
                     statistics.median(wall_samples) / args.steps, 6))
        emit(rank, "complete", arm=args.arm, status="PASS")
        dist.barrier(group=ctrl)
    finally:
        if ctx is not None:
            ctx.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
