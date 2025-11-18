# src/yourpkg/utils/buffers.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List
import torch
import torch.distributed as dist

def _ddp_on() -> bool:
    return dist.is_available() and dist.is_initialized()

def _rank() -> int:
    return dist.get_rank() if _ddp_on() else 0

def _world() -> int:
    return dist.get_world_size() if _ddp_on() else 1

@dataclass
class EpochLogitBuffer:
    """Accumulates per-epoch logits/labels/(optional) indices on each rank, then all_gathers to rank 0.
       By default assumes equal per-rank lengths (use DistributedSampler(drop_last=True))."""
    keep_indices: bool = False
    keep_domains: bool = False
    assume_equal_lengths: bool = True  # set False if you ever allow unequal per-rank counts

    def __post_init__(self):
        self._logit_diffs: List[torch.Tensor] = []
        self._labels: List[torch.Tensor] = []
        self._indices: Optional[List[torch.Tensor]] = [] if self.keep_indices else None
        self._domains: Optional[List[torch.Tensor]] = [] if self.keep_domains else None

    @torch.no_grad()
    def add(self, *, logit_diffs: torch.Tensor, labels: torch.Tensor,
            indices: Optional[torch.Tensor] = None, domains: Optional[torch.Tensor] = None):
        # Detach to avoid graph retention; KEEP ON DEVICE for NCCL all_gather
        self._logit_diffs.append(logit_diffs.detach())
        self._labels.append(labels.detach())
        if self._indices is not None and indices is not None:
            self._indices.append(indices.detach())
        if self._domains is not None and domains is not None:
            self._domains.append(domains.detach())

    def clear(self):
        self._logit_diffs.clear(); self._labels.clear()
        if self._indices is not None: self._indices.clear()
        if self._domains is not None: self._domains.clear()

    # ---------- local stitch ----------
    def _stitch_local(self) -> Dict[str, torch.Tensor]:
        dev = self._logit_diffs[0].device if self._logit_diffs else torch.device("cpu")
        out = {
            "logit_diffs": torch.cat(self._logit_diffs, dim=0) if self._logit_diffs else torch.empty(0, device=dev),
            "labels": torch.cat(self._labels, dim=0) if self._labels else torch.empty(0, dtype=torch.long, device=dev),
        }
        if self._indices is not None:
            out["indices"] = torch.cat(self._indices, dim=0) if self._indices else torch.empty(0, dtype=torch.long, device=dev)
        if self._domains is not None:
            out["domains"] = torch.cat(self._domains, dim=0) if self._domains else torch.empty(0, dtype=torch.long, device=dev)
        return out

    # ---------- equal-length gather (fast path) ----------
    @torch.no_grad()
    def _gather_equal(self, x: torch.Tensor) -> torch.Tensor:
        if not _ddp_on():
            return x
        buf = [torch.empty_like(x) for _ in range(_world())]
        dist.all_gather(buf, x)
        return torch.cat(buf, dim=0) if _rank() == 0 else x.new_empty((0, *x.shape[1:]))

    # ---------- variable-length gather (robust path) ----------
    @torch.no_grad()
    def _gather_varlen(self, x: torch.Tensor, n_local: int) -> torch.Tensor:
        if not _ddp_on():
            return x
        # Gather lengths
        n_t = torch.tensor([n_local], device=x.device, dtype=torch.int64)
        lens = [torch.empty_like(n_t) for _ in range(_world())]
        dist.all_gather(lens, n_t)
        lens = [int(t.item()) for t in lens]
        Lmax = max(lens)
        # Pad to Lmax on dim 0
        if n_local < Lmax:
            pad_shape = (Lmax - n_local, *x.shape[1:])
            x = torch.cat([x, torch.zeros(pad_shape, device=x.device, dtype=x.dtype)], dim=0)
        # Gather padded tensors
        bufs = [torch.empty_like(x) for _ in range(_world())]
        dist.all_gather(bufs, x)
        if _rank() != 0:
            return x.new_empty((0, *x.shape[1:]))
        parts = [bufs[r][:lens[r]] for r in range(_world())]
        return torch.cat(parts, dim=0)

    # ---------- main API ----------
    @torch.no_grad()
    def gather_to_rank0(self, *, cast_fp16: bool = True, reorder_by_indices: bool = False) -> Optional[Dict[str, torch.Tensor]]:
        """Return a CPU payload on rank 0 (or None on others). Does NOT save to disk."""
        local = self._stitch_local()
        logit_diffs = local["logit_diffs"]; labels = local["labels"]
        n_local = int(labels.shape[0])

        # Choose gather path
        if self.assume_equal_lengths:
            logit_diffs_all = self._gather_equal(logit_diffs)
            labels_all = self._gather_equal(labels)
            indices_all = self._gather_equal(local["indices"]) if "indices" in local else None
            domains_all = self._gather_equal(local["domains"]) if "domains" in local else None
        else:
            logit_diffs_all = self._gather_varlen(logit_diffs, n_local)
            labels_all = self._gather_varlen(labels, n_local)
            indices_all = self._gather_varlen(local["indices"], n_local) if "indices" in local else None
            domains_all = self._gather_varlen(local["domains"], n_local) if "domains" in local else None

        if _ddp_on() and _rank() != 0:
            return None  # non-master returns nothing

        # Rank 0: move to CPU, optional reorder & dtype shrink
        if cast_fp16:
            logit_diffs_all = logit_diffs_all.to(torch.float16)
        payload = {
            "logit_diffs": logit_diffs_all.cpu(),
            "labels": labels_all.cpu(),
        }
        if indices_all is not None:
            idx_cpu = indices_all.cpu()
            if reorder_by_indices and idx_cpu.numel() > 0:
                order = torch.argsort(idx_cpu)
                payload["logit_diffs"] = payload["logit_diffs"][order]
                payload["labels"] = payload["labels"][order]
                payload["indices"] = idx_cpu[order]
            else:
                payload["indices"] = idx_cpu
        if domains_all is not None:
            payload["domains"] = domains_all.cpu()
        return payload
