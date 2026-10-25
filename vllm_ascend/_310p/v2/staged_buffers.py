"""310P3 V2 Eager staging primitives; no UVA, Triton or CUDA dispatch.

This is an opt-in building block, NOT wired into GPUModelRunner. The upstream
GPUModelRunner still constructs UVA RequestState and Triton BlockTables before
Ascend overrides them. Keep NPUWorker310 fail-closed until that constructor
and every inherited caller are replaced/reviewed.

Matrix H2D writes below are explicitly BLOCKING and 32-byte-row padded.
1-D state writes use device index_copy_ to preserve concurrent GPU-owned slots.
This is a correctness-first Eager contract, not a performance/async claim.
"""
from __future__ import annotations

from collections.abc import Iterable


class StagedWritePlan:
    """CPU reference and transactional write plan for a 1-D or 2-D int tensor.

    Pending writes are applied in insertion order (last overlapping write wins).
    Bounds are checked at staging time; no partial mutation on invalid writes.
    The physical row stride is rounded up to 32 bytes for 310P3 H2D safety.
    """

    def __init__(self, size: int | tuple[int, int], itemsize: int = 4):
        if isinstance(size, int):
            self.vector = True
            rows, cols = 1, size
        elif isinstance(size, tuple) and len(size) == 2:
            self.vector = False
            rows, cols = size
        else:
            raise ValueError("only 1-D or 2-D staging is supported")
        if rows <= 0 or cols <= 0 or itemsize not in (4, 8):
            raise ValueError("invalid staging shape or dtype size")
        self.rows, self.cols = rows, cols
        self.padded_cols = ((cols * itemsize + 31) // 32) * 32 // itemsize
        self.data = [[0] * self.padded_cols for _ in range(rows)]
        self._pending: list[tuple[int, int, tuple[int | float, ...]]] = []

    def stage_write(self, index: int, start: int, values: Iterable[int | float]) -> None:
        values = tuple(values)
        row = 0 if self.vector else index
        offset = index if self.vector else start
        if self.vector and start != 0:
            raise ValueError("1-D staging requires start=0")
        if not (0 <= row < self.rows and 0 <= offset <= self.cols):
            raise ValueError("staged write row/offset out of bounds")
        if len(values) > self.cols - offset:
            raise ValueError("staged write exceeds logical row")
        if values:
            self._pending.append((row, offset, values))

    def stage_write_elem(self, index: int, value: int | float) -> None:
        self.stage_write(index, 0, (value,))

    def commit(self) -> tuple[int, ...]:
        dirty = set()
        for row, offset, values in self._pending:
            self.data[row][offset:offset + len(values)] = values
            dirty.add(row)
        self._pending.clear()
        return tuple(sorted(dirty))

    def clear_staged_writes(self) -> None:
        self._pending.clear()

    @property
    def pending_count(self) -> int:
        return len(self._pending)


class NPUStagedWriteTensor:
    """Drop-in staged-write surface with *device-backed* storage, not UVA.

    torch is imported only on construction; CPU plan tests require no torch.
    Full padded-row H2D copies avoid unaligned short device writes. The copies
    are deliberately synchronous; the caller must not assume async overlap.
    """

    def __init__(
        self,
        size: int | tuple[int, int],
        dtype,
        device,
        max_concurrency: int | None = None,
        uva_instead_of_gpu: bool = False,
    ):
        import torch

        if dtype not in (torch.int32, torch.int64, torch.float32):
            raise ValueError("unsupported staging dtype")
        self.dtype, self.device = dtype, device
        self.plan = StagedWritePlan(size, torch.empty((), dtype=dtype).element_size())
        self.num_rows = self.plan.rows if not self.plan.vector else self.plan.cols
        shape = (self.plan.padded_cols,) if self.plan.vector else (
            self.plan.rows, self.plan.padded_cols
        )
        self._storage = torch.zeros(shape, dtype=dtype, device=device)
        self.gpu = (
            self._storage[:self.plan.cols]
            if self.plan.vector
            else self._storage[:, :self.plan.cols]
        )
        # An explicitly device-backed buffer is used even if upstream requested
        # UVA. It is NOT a host-mapped pointer and cannot be used as one.
        self.uva_instead_of_gpu = False

    def stage_write(self, index: int, start: int, x: Iterable[int | float]) -> None:
        self.plan.stage_write(index, start, x)

    def stage_write_elem(self, index: int, x: int | float) -> None:
        self.plan.stage_write_elem(index, x)

    def apply_write(self) -> None:
        import torch

        # 1-D state (num_computed_tokens/total_len) is also mutated by the
        # sampling kernel on device. Copying a stale CPU mirror of the whole
        # vector would SILENTLY erase accepted tokens in other live requests.
        # Scatter only the newly staged indices via a device kernel.
        pending = tuple(self.plan._pending)
        dirty = self.plan.commit()
        if self.plan.vector:
            offsets = sorted({
                offset + i
                for _row, offset, values in pending
                for i in range(len(values))
            })
            if offsets:
                indices = torch.tensor(offsets, dtype=torch.int64, device=self.gpu.device)
                values = torch.tensor(
                    [self.plan.data[0][i] for i in offsets],
                    dtype=self.dtype, device=self.gpu.device
                )
                self._storage.index_copy_(0, indices, values)
            return

        # Matrix rows are CPU-authoritative only at add/replace time; callers
        # MUST NOT stage an active GPU-mutated row. Full padded-row H2D copies
        # avoid 310P short unaligned DMA. This restriction is a GO gate.
        for row in dirty:
            source = torch.tensor(self.plan.data[row], dtype=self.dtype, device="cpu")
            self._storage[row].copy_(source, non_blocking=False)

    def clear_staged_writes(self) -> None:
        self.plan.clear_staged_writes()


class NPUBackedTensor:
    """Non-UVA replacement exposing upstream .np/.gpu/copy_to_uva surface.

    The historical method name is retained only for call-site compatibility.
    It copies the entire aligned allocation to device synchronously, including
    when n is supplied, so no partial-row copy can overwrite an adjacent row.
    """

    def __init__(self, size: int | tuple[int, int], dtype, device):
        import torch

        if dtype not in (torch.int32, torch.int64, torch.float32):
            raise ValueError("unsupported buffer dtype")
        self.dtype = dtype
        self.plan = StagedWritePlan(size, torch.empty((), dtype=dtype).element_size())
        shape = (self.plan.padded_cols,) if self.plan.vector else (
            self.plan.rows, self.plan.padded_cols
        )
        self._cpu_storage = torch.zeros(shape, dtype=dtype, device="cpu")
        self._gpu_storage = torch.zeros(shape, dtype=dtype, device=device)
        self.cpu = (
            self._cpu_storage[:self.plan.cols]
            if self.plan.vector
            else self._cpu_storage[:, :self.plan.cols]
        )
        self.np = self.cpu.numpy()
        self.gpu = (
            self._gpu_storage[:self.plan.cols]
            if self.plan.vector
            else self._gpu_storage[:, :self.plan.cols]
        )

    def copy_to_uva(self, n: int | None = None):
        if n is not None and not 0 <= n <= len(self.cpu):
            raise ValueError("invalid row count")
        self._gpu_storage.copy_(self._cpu_storage, non_blocking=False)
        return self.gpu


class NPUFusedStagedWriter:
    """T14 multi-group fallback: apply each owned buffer without pointer kernels.

    output_ptrs/strides are intentionally unused; we never dereference a device
    address from a Python integer. Eager-only sequential ordering is explicit.
    """

    def __init__(self, device, max_writes: int, max_concurrency: int | None = None):
        if max_writes <= 0:
            raise ValueError("max_writes must be positive")
        self.device = device

    def apply(self, tensors, output_ptrs=None, output_strides=None) -> None:
        for tensor in tensors:
            if not isinstance(tensor, NPUStagedWriteTensor):
                raise TypeError("unsafe foreign staged tensor")
        for tensor in tensors:
            tensor.apply_write()
