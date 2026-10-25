"""310P3 V2 RequestState candidate without GPU UVA construction.

Matches vLLM 0.27.1 RequestState's public attributes and methods. Does not
subclass/call GPU RequestState.__init__: that constructor allocates UVA pools
before Ascend's post-super replacement. Not yet wired into a 310P V2 runner.
"""
from __future__ import annotations

from .staged_buffers import NPUBackedTensor, NPUStagedWriteTensor


class NPURequestState:
    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        num_speculative_steps: int,
        vocab_size: int,
        device,
        num_prefill_lookahead: int = 1,
    ):
        import numpy as np
        import torch

        if (min(max_num_reqs, max_model_len, max_num_batched_tokens,
                vocab_size, num_prefill_lookahead) <= 0 or num_speculative_steps < 0):
            raise ValueError("invalid RequestState capacity")
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.num_speculative_steps = num_speculative_steps
        self.vocab_size = vocab_size
        self.device = device
        self.req_id_to_index: dict[str, int] = {}
        self.index_to_req_id: dict[int, str] = {}
        self.free_indices = list(range(max_num_reqs))
        # All token IDs are physically device-backed, NOT host UVA mapped.
        self.all_token_ids = NPUStagedWriteTensor(
            (max_num_reqs, max_model_len), torch.int32, device,
            uva_instead_of_gpu=False,
        )
        self.prompt_len = NPUBackedTensor(max_num_reqs, torch.int32, device)
        self.prefill_len = NPUBackedTensor(max_num_reqs, torch.int32, device)
        self.total_len = NPUStagedWriteTensor(max_num_reqs, torch.int32, device)
        self.num_computed_prefill_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_computed_tokens = NPUStagedWriteTensor(max_num_reqs, torch.int32, device)
        self.num_computed_tokens_np = np.zeros(max_num_reqs, dtype=np.int32)
        # Physical rows are 32-byte aligned: resetting a reused request
        # must never issue a short unaligned 8-byte D2D write on 310P3.
        self._last_sampled_storage = torch.zeros(
            max_num_reqs, 4, dtype=torch.int64, device=device
        )
        self.last_sampled_tokens = self._last_sampled_storage[:, :1]
        self.max_seq_len = np.zeros(max_num_reqs, dtype=np.int32)
        draft_width = ((num_speculative_steps + 3) // 4) * 4
        self._draft_storage = torch.zeros(
            max_num_reqs, draft_width, dtype=torch.int64, device=device
        )
        self.draft_tokens = self._draft_storage[:, :num_speculative_steps]
        self.next_prefill_tokens = torch.zeros(
            num_prefill_lookahead, max_num_reqs, dtype=torch.int32, device=device
        )
        # Ascend V2 attention expects an explicit CPU mirror.
        self.num_computed_tokens_cpu = torch.zeros(max_num_reqs, dtype=torch.int32)

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    def add_request(
        self,
        req_id: str,
        prompt_len: int,
        all_token_ids: list[int],
        num_computed_tokens: int,
        max_tokens: int,
    ) -> None:
        if req_id in self.req_id_to_index:
            raise ValueError("duplicate request id")
        if not self.free_indices:
            raise ValueError("no free request slots")
        prefill_len = len(all_token_ids)
        if not (0 <= prompt_len <= prefill_len <= self.max_model_len):
            raise ValueError("invalid prompt/prefill length")
        if max_tokens < 0 or prompt_len + max_tokens > self.max_model_len:
            raise ValueError("request exceeds model capacity")
        if not (0 <= num_computed_tokens <= prefill_len):
            raise ValueError("invalid computed token count")
        idx = self.free_indices.pop()
        self.req_id_to_index[req_id] = idx
        self.index_to_req_id[idx] = req_id
        self.max_seq_len[idx] = prompt_len + max_tokens
        self.prompt_len.np[idx] = prompt_len
        self.prefill_len.np[idx] = prefill_len
        self.total_len.stage_write_elem(idx, prefill_len)
        # A recycled row may still contain the previous request's tokens
        # beyond the new prompt. Stage a full reset BEFORE the new prefix;
        # only the committed, 32-byte-aligned row is copied to the device.
        # The Runner must separately ensure no in-flight kernel owns idx.
        self.all_token_ids.stage_write(
            idx, 0, [0] * self.max_model_len
        )
        self.all_token_ids.stage_write(idx, 0, all_token_ids)
        self.num_computed_prefill_tokens[idx] = num_computed_tokens
        self.num_computed_tokens_np[idx] = num_computed_tokens
        self.num_computed_tokens.stage_write_elem(idx, num_computed_tokens)
        self.num_computed_tokens_cpu[idx] = num_computed_tokens
        self._last_sampled_storage[idx].zero_()
        self._draft_storage[idx].zero_()
        # Prefill lookahead is indexed by request slot, not by req_id.
        # A reused slot must not inherit the previous request's predictions.
        # Device-side index_fill avoids a CPU-origin 4-byte partial H2D.
        import torch
        slot = torch.tensor([idx], dtype=torch.int64, device=self.device)
        self.next_prefill_tokens.index_fill_(1, slot, 0)

    def apply_staged_writes(self) -> None:
        self.prompt_len.copy_to_uva()
        self.prefill_len.copy_to_uva()
        self.total_len.apply_write()
        self.all_token_ids.apply_write()
        self.num_computed_tokens.apply_write()

    def remove_request(self, req_id: str) -> int | None:
        idx = self.req_id_to_index.pop(req_id, None)
        if idx is None:
            return None
        # Removing a slot with uncommitted writes would allow a later
        # apply_staged_writes to write into a different request's row.
        # The scheduler must drain the batch before recycling a slot.
        if any(row == idx for row, _off, _values
               in self.all_token_ids.plan._pending):
            self.req_id_to_index[req_id] = idx
            raise RuntimeError("cannot release request with pending token writes")
        self.index_to_req_id.pop(idx)
        self.free_indices.append(idx)
        return idx
