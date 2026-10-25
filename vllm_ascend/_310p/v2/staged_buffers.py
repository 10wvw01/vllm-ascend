"""310P3 V2 Eager staging primitives; no UVA, Triton or CUDA dispatch.

This is an opt-in building block, NOT wired into GPUModelRunner. The upstream
GPUModelRunner still constructs UVA RequestState and Triton BlockTables before
Ascend overrides them. Keep NPUWorker310 fail-closed until that constructor
and every inherited caller are replaced/reviewed.

Matrix H2D writes below are explicitly BLOCKING and 32-byte-row padded.
1-D state writes merge device-owned values into full 32-byte rows before\nindex_copy_; same-stream ordering remains mandatory and NPU op support NO-GO.
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

        if getattr(self, "_poisoned", False):
            raise RuntimeError("310P3 staged buffer poisoned by failed device commit")
        pending = tuple(self.plan._pending)
        if not pending:
            return
        # Do NOT commit the CPU plan before device writes complete. The
        # previous implementation lost pending writes if an NPU copy failed.
        # Construct the proposed state without mutating the committed mirror.
        dirty = tuple(sorted({row for row, _offset, _values in pending}))
        try:
            if self.plan.vector:
                # The device owns all other live indices (post_update mutates
                # them). Never upload the stale full-vector CPU mirror.
                changes = {}
                for _row, offset, values in pending:
                    for i, value in enumerate(values):
                        changes[offset + i] = value
                # Never issue narrow persistent int32/int64 writes on
                # 310P3. Read the currently device-owned 32B row, merge
                # only the newly staged slots, then write a complete row.
                # This preserves other requests advanced by post_update.
                lanes = 32 // self._storage.element_size()
                changes_by_row = {}
                for offset, value in changes.items():
                    row, col = divmod(offset, lanes)
                    changes_by_row.setdefault(row, {})[col] = value
                row_ids = sorted(changes_by_row)
                rows = torch.tensor(
                    row_ids, dtype=torch.int64, device=self.gpu.device
                )
                row_storage = self._storage.view(-1, lanes)
                current = row_storage.index_select(0, rows)
                masks = [
                    [col in changes_by_row[row] for col in range(lanes)]
                    for row in row_ids
                ]
                values = [
                    [changes_by_row[row].get(col, 0) for col in range(lanes)]
                    for row in row_ids
                ]
                merged = torch.where(
                    torch.tensor(masks, dtype=torch.bool,
                                 device=self.gpu.device),
                    torch.tensor(values, dtype=self.dtype,
                                 device=self.gpu.device),
                    current,
                )
                row_storage.index_copy_(0, rows, merged)
            else:
                # Only newly owned or explicitly recycled rows may be staged:
                # post_update owns active rows on device. Full 32B-aligned
                # rows avoid short, potentially unsafe 310P3 DMA writes.
                proposed = {
                    row: list(self.plan.data[row]) for row in dirty
                }
                for row, offset, values in pending:
                    proposed[row][offset:offset + len(values)] = values
                for row in dirty:
                    source = torch.tensor(
                        proposed[row], dtype=self.dtype, device="cpu"
                    )
                    self._storage[row].copy_(source, non_blocking=False)
        except Exception:
            # Some groups/rows may already have been copied; retrying could
            # expose a half-committed KV state. Poison until Runner aborts
            # the affected batch and tears down this adapter.
            self._poisoned = True
            raise
        self.plan.commit()

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
        return self.gpu if n is None else self.gpu[:n]


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
        if any(getattr(tensor, "_poisoned", False) for tensor in tensors):
            raise RuntimeError("310P3 fused write includes a poisoned buffer")
        try:
            for tensor in tensors:
                tensor.apply_write()
        except Exception:
            # Cross-group Eager writes are not atomic. Never continue after
            # a partial group commit and silently mix different KV epochs.
            for tensor in tensors:
                tensor._poisoned = True
            raise
