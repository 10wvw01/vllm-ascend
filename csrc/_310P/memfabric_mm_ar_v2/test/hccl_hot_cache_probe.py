#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""310P3 TP2 HCCL 冷热/同 Shape/动态 Shape/异步数据正确性快速实验。

限制：只测 torch.distributed HCCL，未触及 vLLM、自定义 V2 或 MemFabric。
首次调用变快不能单独证明 AI_CPU cache 命中；HOST/AI_CPU 是算法展开模式，
不是 cache 开关。每个阶段在热态与冷态分别观察，两 Rank 都输出独立 JSON。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


def value(rank: int, layer: int, step: int) -> float:
    # FP16 可精确表示的数据，连续调用必须随代际和层编号发生变化。
    return float(rank + 1) + (layer % 5) * 0.5 + (step % 11) * 0.125


def expected(layer: int, step: int) -> float:
    # TP2 SUM 的结果；小值且用二进制可精确表示的增量，支持严格相等校验。
    return 3.0 + (layer % 5) + (step % 11) * 0.25


def check_tensor(t: torch.Tensor, layer: int, step: int, where: str) -> None:
    bad = int(torch.count_nonzero(t != expected(layer, step)).item())
    if bad:
        raise RuntimeError(
            f"{where}: rank={dist.get_rank()} layer={layer} step={step} "
            f"shape={tuple(t.shape)} mismatched_elements={bad}"
        )


def report(phase: str, rank: int, **metrics: object) -> None:
    print(json.dumps(
        {"tag": "HCCL_CACHE_PROBE", "phase": phase, "rank": rank,
         "expansion_mode": os.getenv("HCCL_OP_EXPANSION_MODE", "<unset/default>"),
         **metrics}, ensure_ascii=False), flush=True)


def stats(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_ms": round(statistics.median(samples), 5),
        "p95_ms": round(ordered[max(0, int(len(samples) * 0.95 + 0.999) - 1)], 5),
        "min_ms": round(ordered[0], 5),
        "max_ms": round(ordered[-1], 5),
    }


def timed_allreduces(tensors: list[torch.Tensor]) -> tuple[float, float]:
    # 不包含数据填充；计时包含 Python/HCCL 排队、设备实际执行和收尾同步。
    begin = time.perf_counter()
    works = [dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
             for t in tensors]
    launched = time.perf_counter()
    for work in works:
        work.wait()
    torch.npu.synchronize()
    end = time.perf_counter()
    return (launched - begin) * 1000, (end - begin) * 1000


def fill_inputs(tensors: list[torch.Tensor], rank: int, step: int) -> None:
    for layer, t in enumerate(tensors):
        t.fill_(value(rank, layer, step))
    # 通信测时不要把数据填充混进去；两 Rank 在本地先完成数据准备。
    torch.npu.synchronize()


def singles(rank: int, dev: torch.device, args: argparse.Namespace) -> None:
    for m in (2048, 3072, 4096):
        t = torch.empty((m, 2048), dtype=torch.float16, device=dev)
        fill_inputs([t], rank, 0)
        first_launch, first_elapsed = timed_allreduces([t])
        check_tensor(t, 0, 0, "single_first")
        warm = []
        for step in range(1, args.warmup + 1):
            fill_inputs([t], rank, step)
            _, dt = timed_allreduces([t])
            warm.append(dt)
            check_tensor(t, 0, step, "single_warm")
        hot: list[float] = []
        for step in range(args.warmup + 1, args.warmup + args.single_reps + 1):
            fill_inputs([t], rank, step)
            _, dt = timed_allreduces([t])
            hot.append(dt)
            if step % args.verify_every == 0:
                check_tensor(t, 0, step, "single_hot")
        check_tensor(t, 0, args.warmup + args.single_reps, "single_last")
        report("single", rank, shape=[m, 2048], bytes_per_rank=t.numel() * 2,
               first_ms=round(first_elapsed, 5),
               first_launch_ms=round(first_launch, 5),
               warmup=stats(warm), hot=stats(hot), hot_samples=len(hot))
        dist.barrier()


def bursts(rank: int, dev: torch.device, args: argparse.Namespace,
           name: str, shapes: list[int]) -> None:
    # 每层不同地址，40次同Shape不能通过“返回上一次数据”蒙混过关。
    tensors = [torch.empty((m, 2048), dtype=torch.float16, device=dev)
               for m in shapes]
    used_bytes = sum(t.numel() * t.element_size() for t in tensors)
    elapsed: list[float] = []
    launched: list[float] = []
    for step in range(args.burst_warmup + args.burst_reps):
        fill_inputs(tensors, rank, step)
        launch_ms, dt = timed_allreduces(tensors)
        if step >= args.burst_warmup:
            elapsed.append(dt)
            launched.append(launch_ms)
        if (step == 0 or step % args.verify_every == 0
                or step == args.burst_warmup + args.burst_reps - 1):
            for layer, t in enumerate(tensors):
                check_tensor(t, layer, step, name)
    report(name, rank, ops_per_round=len(tensors), bytes_per_rank=used_bytes,
           per_op_bytes_mean=round(used_bytes / len(tensors)),
           async_ops=True, batch=stats(elapsed), enqueue=stats(launched),
           amortized_ms_per_op=round(statistics.median(elapsed) / len(tensors), 5))
    del tensors
    dist.barrier()


def dependency_chain(rank: int, dev: torch.device, args: argparse.Namespace) -> None:
    """异步前驱写入→HCCL→当前计算流的后继读取，验证数据依赖隔离。"""
    buffers = [torch.empty((2048, 2048), dtype=torch.float16, device=dev)
               for _ in range(min(args.layers, 40))]
    for step in range(args.chain_reps):
        reads = []
        for layer, t in enumerate(buffers):
            # 上游填充不立即 sync；HCCL 调用必须等待调用方流已排队的数据准备。
            t.fill_(value(rank, layer, step))
            work = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
            # 必须显式建立当前计算流等待通信完成的依赖。
            work.wait()
            # 后继只取首 64 元素，减少额外内存消耗；整张量另做抽样校验。
            reads.append(t.flatten()[:64] * 2.0)
        torch.npu.synchronize()
        for layer, (t, y) in enumerate(zip(buffers, reads)):
            if not bool(torch.all(y == expected(layer, step) * 2).item()):
                raise RuntimeError(f"chain stale dependency rank={rank} layer={layer} step={step}")
            if step == 0 or step == args.chain_reps - 1:
                check_tensor(t, layer, step, "chain_full")
    report("async_dependency_chain", rank, layers=len(buffers),
           rounds=args.chain_reps, result="PASS", stream="default + HCCL internal")
    dist.barrier()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layers", type=int, default=40)
    p.add_argument("--warmup", type=int, default=6)
    p.add_argument("--single-reps", type=int, default=24)
    p.add_argument("--burst-warmup", type=int, default=3)
    p.add_argument("--burst-reps", type=int, default=8)
    p.add_argument("--chain-reps", type=int, default=4)
    p.add_argument("--verify-every", type=int, default=4)
    args = p.parse_args()
    if int(os.getenv("WORLD_SIZE", "0")) != 2:
        p.error("仅允许 torchrun --nproc-per-node=2")
    if (not 1 <= args.layers <= 40 or args.warmup < 1 or args.single_reps < 1
            or args.burst_warmup < 1 or args.burst_reps < 1
            or args.chain_reps < 1 or args.verify_every < 1):
        p.error("参数必须为正，且 1 <= layers <= 40")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dev = torch.device(f"npu:{local_rank}")
    # HCCL init 放在数据 Tensor 分配之前，以满足 300I Duo 初始化约束。
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        report("environment", rank,
               torch_version=torch.__version__,
               torch_npu_version=getattr(torch_npu, "__version__", "unknown"),
               current_device=int(torch.npu.current_device()),
               note="不通过加速结果直接推断 AI CPU cache 命中")
        singles(rank, dev, args)
        fixed = [3072] * args.layers
        mixed = [2048, 3072, 4096] * (args.layers // 3)
        mixed += [2048, 3072, 4096][:args.layers % 3]
        bursts(rank, dev, args, "fixed_same_shape_independent_buffers", fixed)
        bursts(rank, dev, args, "mixed_shapes_similar_total_bytes", mixed)
        dependency_chain(rank, dev, args)
        report("complete", rank, result="PASS")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
