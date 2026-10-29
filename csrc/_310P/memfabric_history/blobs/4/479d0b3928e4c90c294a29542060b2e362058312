"""310P3 Target GDN align 的独立 Torch-NPU 数据搬运候选（未注册）。

这是一个可实际执行 torch.index_select/index_copy_ 的设备操作原型。
CPU gold 可运行；NPU 仅在调用方显式启用诊断且提供当前 Stream 时才可尝试。
NPU op 支持/物理内存 32B 写行为/异步生命周期尚未经 310P3 实测。
绝不在 Worker / Runner 注册，不导入私有 Draft、Triton 或 CUDA。
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch


@dataclass(frozen=True)
class GDNDeviceLayer310:
    conv: torch.Tensor
    temporal: torch.Tensor
    conv_layout: str = "SD"


@dataclass(frozen=True)
class GDNDeviceGroup310:
    group_id: int
    # CPU int32 每行代表当前 batch 的物理块表；必须来自同步的权威快照。
    block_table: torch.Tensor
    layers: tuple[GDNDeviceLayer310, ...]


@dataclass(frozen=True)
class GDNMove310:
    """已审计算法生产的逻辑搬运指令；不包含权重或 Draft 私有数据。"""
    batch_row: int
    source_column: int
    destination_column: int
    accepted_bias: int


@dataclass(frozen=True)
class _PreparedWrite:
    storage: torch.Tensor
    index: torch.Tensor
    replacement: torch.Tensor
    destination_key: tuple[int, int]


def _require_device(t: torch.Tensor, device: torch.device) -> None:
    if t.device != device:
        raise ValueError("GDN 缓存不在候选执行设备")


def _index(value: int, device: torch.device) -> torch.Tensor:
    # device-native 索引 Tensor；源数值必须来自经过验证的 CPU 块表。
    return torch.tensor([value], dtype=torch.int64, device=device)


class GDNWholeBlockCopy310Candidate:
    """T30-B/T31-B: 整物理块覆盖，避免直接对持久 cache 做窄切片写。

    相比 CPU 原型，这里实际使用目标设备上的 index_select/index_copy_
    完成复制，不读出设备 Tensor 标量；但是逻辑 move/BT 由权威 CPU
    元数据生成，是保守的 Eager 同步正确性实验架构，不是生产异步方案。
    """

    def __init__(
        self,
        groups: Sequence[GDNDeviceGroup310],
        *,
        device: torch.device | str = "cpu",
        allow_npu_diagnostic: bool = False,
        execution_stream: object | None = None,
    ) -> None:
        if isinstance(device, str) and device.startswith("npu") and not hasattr(torch, "npu"):
            raise NotImplementedError("本环境没有 torch_npu 注册，禁止启动设备候选")
        self.device = torch.device(device)
        self.poisoned = False
        self._groups: tuple[GDNDeviceGroup310, ...] = ()
        self._in_flight = False
        self._device_pending = False
        self._bound_stream = execution_stream if self.device.type == "npu" else None
        self._test_fail_after_write: int | None = None
        if self.device.type not in ("cpu", "npu"):
            raise NotImplementedError("310P3 候选只支持 CPU gold 或 NPU")
        if self.device.type == "npu":
            if not allow_npu_diagnostic or execution_stream is None:
                raise NotImplementedError("设备入口须显式授权诊断并绑定 NPU Stream")
            if not hasattr(torch, "npu"):
                raise NotImplementedError("torch_npu 尚未注册")
            current = torch.npu.current_stream(self.device)
            if (not hasattr(current, "npu_stream") or
                    not hasattr(execution_stream, "npu_stream") or
                    current.npu_stream != execution_stream.npu_stream):
                raise RuntimeError("GDN 数据流必须与当前 NPU Stream 一致")
        if not groups:
            raise ValueError("必须包含至少一个 Mamba KV Group")
        seen_groups: set[int] = set()
        seen_storages: set[int] = set()
        for group in groups:
            if group.group_id in seen_groups or group.group_id < 0:
                raise ValueError("Mamba KV Group ID 不唯一")
            seen_groups.add(group.group_id)
            bt = group.block_table
            if (bt.device.type != "cpu" or bt.dtype != torch.int32 or
                    bt.ndim != 2 or bt.shape[1] < 1):
                raise NotImplementedError("块表必须是已同步、权威的 CPU int32 快照")
            if not group.layers:
                raise ValueError("空 Mamba KV Group")
            for layer in group.layers:
                if layer.conv_layout not in ("SD", "DS"):
                    raise ValueError("不支持的 Conv 状态布局")
                conv, temporal = layer.conv, layer.temporal
                for tensor in (conv, temporal):
                    _require_device(tensor, self.device)
                    if (tensor.ndim < 2 or tensor.shape[0] < 1 or
                            not tensor.is_contiguous()):
                        raise NotImplementedError("要求完整连续物理状态块")
                    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                        raise NotImplementedError("未审核的 GDN 缓存 dtype")
                    # 明确保守：每次 index_copy_ 覆盖完整 >=32B 的物理行。
                    row_bytes = tensor[0].numel() * tensor.element_size()
                    if (row_bytes < 32 or row_bytes % 32 or
                            tensor.stride(0) * tensor.element_size() % 32 or
                            tensor.data_ptr() % 32):
                        raise NotImplementedError("GDN 物理状态块未满足 32B 全行要求")
                    pointer = tensor.untyped_storage().data_ptr()
                    if pointer in seen_storages:
                        raise ValueError("不同组/层共享物理存储尚未审计")
                    seen_storages.add(pointer)
                if conv.ndim != 3 or temporal.ndim < 2:
                    raise ValueError("GDN conv/temporal rank 不合法")
                width = conv.shape[1] if layer.conv_layout == "SD" else conv.shape[2]
                if width <= 7:
                    raise ValueError("Conv 历史宽度无法容纳 K7 偏移")
        self._groups = tuple(groups)

    def _check(self) -> None:
        if self.poisoned:
            raise RuntimeError("GDN 设备搬运已中毒，禁止重试")

    @staticmethod
    def _block_id(bt: torch.Tensor, row: int, col: int, limit: int) -> int:
        if col < 0 or col >= bt.shape[1]:
            raise ValueError("逻辑块表越界")
        physical = int(bt[row, col])
        if not 0 <= physical < limit:
            raise ValueError("物理块表越界")
        return physical

    def _prepare(self, moves: Sequence[GDNMove310]) -> list[_PreparedWrite]:
        seen_dest: set[tuple[int, int]] = set()
        writes: list[_PreparedWrite] = []
        for move in moves:
            if (not 0 <= move.accepted_bias <= 7 or move.batch_row < 0 or
                    move.source_column < 0 or move.destination_column < 0):
                raise ValueError("非法 K7 状态搬运意图")
            for group in self._groups:
                if move.batch_row >= group.block_table.shape[0]:
                    raise ValueError("块表行不是有效 batch row")
                for layer in group.layers:
                    conv, temporal = layer.conv, layer.temporal
                    width = (conv.shape[1] if layer.conv_layout == "SD"
                             else conv.shape[2])
                    bias = move.accepted_bias
                    if bias >= width:
                        raise ValueError("K7 偏移超过 Conv 历史长度")
                    src = self._block_id(
                        group.block_table, move.batch_row,
                        move.source_column, conv.shape[0])
                    dst = self._block_id(
                        group.block_table, move.batch_row,
                        move.destination_column, conv.shape[0])
                    # 同物理块 bias=0 时 conv 不需要读写；但 temporal
                    # 的来源也必须通过 src_col+bias 独立确定。
                    if src != dst or bias != 0:
                        src_idx = _index(src, self.device)
                        dst_idx = _index(dst, self.device)
                        source_block = conv.index_select(0, src_idx)
                        dest_block = conv.index_select(0, dst_idx)
                        if layer.conv_layout == "SD":
                            replacement = torch.cat(
                                (source_block[:, bias:, :],
                                 dest_block[:, width - bias:, :]), dim=1)
                        else:
                            replacement = torch.cat(
                                (source_block[:, :, bias:],
                                 dest_block[:, :, width - bias:]), dim=2)
                        key = (conv.untyped_storage().data_ptr(), dst)
                        if key in seen_dest:
                            raise ValueError("多个请求试图覆盖同一物理 Conv 块")
                        seen_dest.add(key)
                        writes.append(_PreparedWrite(conv, dst_idx, replacement, key))
                    t_src = self._block_id(
                        group.block_table, move.batch_row,
                        move.source_column + bias, temporal.shape[0])
                    t_dst = self._block_id(
                        group.block_table, move.batch_row,
                        move.destination_column, temporal.shape[0])
                    if t_src != t_dst:
                        key = (temporal.untyped_storage().data_ptr(), t_dst)
                        if key in seen_dest:
                            raise ValueError("多个请求试图覆盖同一 Temporal 块")
                        seen_dest.add(key)
                        writes.append(_PreparedWrite(
                            temporal, _index(t_dst, self.device),
                            temporal.index_select(0, _index(t_src, self.device)),
                            key,
                        ))
        return writes

    def execute(self, moves: Sequence[GDNMove310]) -> int:
        """只执行已验证意图。全批先从源做 device 快照，再写目标整块。"""
        self._check()
        if self._in_flight or self._device_pending:
            raise RuntimeError("前一次 GDN 设备操作未确认完成")
        if self.device.type == "npu":
            current = torch.npu.current_stream(self.device)
            if (not hasattr(current, "npu_stream") or
                    current.npu_stream != self._bound_stream.npu_stream):
                raise RuntimeError("NPU 当前 Stream 已变化")
        self._in_flight = True
        try:
            staged = self._prepare(moves)
            for n, write in enumerate(staged):
                if self._test_fail_after_write == n:
                    raise RuntimeError("模拟 NPU 部分写入故障")
                # 物理 Tensor 整行写；32B/支持性仍须310P3设备取证。
                write.storage.index_copy_(0, write.index, write.replacement)
            if self.device.type == "npu":
                self._device_pending = True
            return len(staged)
        except Exception:
            self.poisoned = True
            raise
        finally:
            # NPU 发射完成不等于设备执行完成；保持 device_pending
            # 直到显式同步成功，不允许回收物理缓存。
            self._in_flight = False

    def confirm_completed(self) -> None:
        """诊断路径显式等待；非310P3准入证据。"""
        self._check()
        if self._in_flight:
            raise RuntimeError("GDN 操作尚在提交")
        if self.device.type == "npu" and self._device_pending:
            try:
                self._bound_stream.synchronize()
            except Exception:
                self.poisoned = True
                raise
            self._device_pending = False

    @property
    def device_pending(self) -> bool:
        return self._device_pending


class PaddedInt32State310Candidate:
    """T30-A/T31-A的32B整行 int32 元信息同步写原型。

    CPU 输入代表显式同步且已检验的Host metadata；生产异步K7必须
    替换成 device-owned 计数计算/写回，不能借用乐观 Host 镜像。
    """

    def __init__(self, max_slots: int, *, device: torch.device | str = "cpu",
                 allow_npu_diagnostic: bool = False,
                 execution_stream: object | None = None) -> None:
        if max_slots <= 0:
            raise ValueError("max_slots must be positive")
        if isinstance(device, str) and device.startswith("npu") and not hasattr(torch, "npu"):
            raise NotImplementedError("本环境没有 torch_npu 注册，禁止启动 int32 设备状态")
        self.device = torch.device(device)
        if self.device.type == "npu":
            if not allow_npu_diagnostic or execution_stream is None:
                raise NotImplementedError("NPU int32 状态行仅允许显式诊断")
            if not hasattr(torch, "npu"):
                raise NotImplementedError("torch_npu 不可用")
            current = torch.npu.current_stream(self.device)
            if (not hasattr(current, "npu_stream") or
                    current.npu_stream != getattr(execution_stream, "npu_stream", None)):
                raise RuntimeError("int32 元数据更新 Stream 不匹配")
        elif self.device.type != "cpu":
            raise NotImplementedError("不支持的设备")
        self.max_slots = max_slots
        self._bound_stream = execution_stream if self.device.type == "npu" else None
        self._device_pending = False
        self.data = torch.ones(
            ((max_slots + 7) // 8, 8), dtype=torch.int32, device=self.device)
        self.poisoned = False

    def scatter_authoritative_counts(
        self, slots: Sequence[int], accepted: Sequence[int],
    ) -> None:
        if self.poisoned:
            raise RuntimeError("接受数元信息已中毒")
        if self._device_pending:
            raise RuntimeError("NPU 接受数写尚未确认完成")
        if self.device.type == "npu":
            stream = torch.npu.current_stream(self.device)
            if (not hasattr(stream, "npu_stream") or
                    stream.npu_stream != self._bound_stream.npu_stream):
                raise RuntimeError("接受数当前 NPU Stream 不匹配")
        if len(slots) != len(accepted) or len(set(slots)) != len(slots):
            raise ValueError("重复/不一致请求槽位")
        staged: dict[int, list[tuple[int, int]]] = {}
        for slot, count in zip(slots, accepted):
            if (not isinstance(slot, int) or not 0 <= slot < self.max_slots or
                    not isinstance(count, int) or not 0 <= count <= 8):
                raise ValueError("接受数必须是已核实的 K7 计数")
            staged.setdefault(slot // 8, []).append((slot % 8, max(count, 1)))
        try:
            # 同一物理32B行内的所有req_slot一次合并，防止彼此覆盖。
            full_writes: list[tuple[torch.Tensor, torch.Tensor]] = []
            for row, parts in staged.items():
                row_index = _index(row, self.device)
                existing = self.data.index_select(0, row_index)
                cols = torch.arange(8, dtype=torch.int64, device=self.device)
                for col, value in parts:
                    existing = torch.where(
                        cols[None, :] == col,
                        torch.full_like(existing, value),
                        existing)
                full_writes.append((row_index, existing))
            for row_index, full in full_writes:
                self.data.index_copy_(0, row_index, full)
            if self.device.type == "npu" and full_writes:
                self._device_pending = True
        except Exception:
            self.poisoned = True
            raise

    @property
    def device_pending(self) -> bool:
        return self._device_pending

    def confirm_completed(self) -> None:
        """等待当前诊断批次元数据写完成；不充当生产Event接口。"""
        if self.poisoned:
            raise RuntimeError("接受数元信息已中毒")
        if self.device.type == "npu" and self._device_pending:
            try:
                self._bound_stream.synchronize()
            except Exception:
                self.poisoned = True
                raise
            self._device_pending = False
