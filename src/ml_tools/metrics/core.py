#src/ml_tools/metrics/core.py

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Any, Optional, Mapping
import time
from .segmentation import segmentation_batch_metrics
from .classification import classification_batch_metrics

import torch


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def _to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if torch.is_tensor(x):
        return int(x.detach().cpu().item())
    return int(x)

@torch.no_grad()
def get_batch_metrics(
    batch: Mapping[str, Any],
    loss_fns: Mapping[str, Any],
    *,
    task: str = "classification",
    pred_threshold: float = 0.5,
    label_threshold: float = 0.5,
    valid_mask_key: Optional[str] = None,
    topk: int = 1
) -> dict[str, Any]:
    """
    Task-aware wrapper.

    Expects for segmentation:
      batch["pred"]  : (N, 1, H, W) logits
      batch["label"] : (N, 1, H, W) labels
      optionally batch[valid_mask_key] : (N, 1, H, W) bool

    loss_fns should contain:
      loss_fns["ce"] : callable for logits/labels (e.g. BCEWithLogitsLoss)
    """
    if task == "classification":
        logits = batch["pred"]    # (N, C)
        labels = batch["label"]   # (N,)
        loss = loss_fns["ce"](logits, labels)  # or whatever key you choose
        return classification_batch_metrics(logits, labels, loss=loss, topk=topk)

    
    elif task == "segmentation":
        logits = batch["pred"]
        labels = batch["label"]
        valid_mask = batch.get(valid_mask_key) if valid_mask_key else None

        loss = loss_fns["bce"](logits, labels)

        return segmentation_batch_metrics(
            logits,
            labels,
            loss_fn=loss_fns['ce'],
            pred_threshold=pred_threshold,
            label_threshold=label_threshold,
            valid_mask=valid_mask,
        )
    raise ValueError(f"Unsupported task={task!r}")


@dataclass
class RollingWindow:
    """Stores last W values. Supports weighted mean and raw sum."""
    window: int = 50
    _vals: deque[float] = field(init=False)
    _wts: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        w = max(1, int(self.window))
        self._vals = deque(maxlen=w)
        self._wts = deque(maxlen=w)

    def update(self, value: float, weight: float = 1.0) -> None:
        self._vals.append(float(value))
        self._wts.append(float(weight))

    def total(self) -> float:
        return float(sum(self._vals))

    def weighted_total(self) -> float:
        return float(sum(v * w for v, w in zip(self._vals, self._wts)))

    def weighted_mean(self) -> float:
        denom = sum(self._wts)
        return float(self.weighted_total() / denom) if denom > 0 else 0.0

    def reset(self) -> None:
        self._vals.clear()
        self._wts.clear()


@dataclass
class RunningStats:
    """
    One-stop metrics aggregator:
      - rolling windows (means and sums)
      - epoch totals (sums)

    You configure which keys are:
      - rolling weighted means (losses): `mean_keys`
      - rolling sums (counts): `sum_keys`
    All keys (mean+sum) also get epoch totals accumulated.

    For weighted means you also configure which key provides the weight (e.g. 'batch_size' or 'num_pixels').
    """
    window: int = 50
    phase: str = "train"
    ddp_sync: bool = False  # optional, only used in sync_snapshot_ddp

    # which metrics to track
    weight_key: str = "batch_size"  # used for mean_keys
    mean_keys: tuple[str, ...] = ("loss",)  # rolling weighted mean + epoch weighted sum
    sum_keys: tuple[str, ...] = ("correct",)  # rolling sum + epoch sum

    # internal
    _roll: dict[str, RollingWindow] = field(init=False, default_factory=dict)
    _epoch_sums: dict[str, float] = field(init=False, default_factory=dict)
    _timer_start: float = field(init=False, default_factory=time.time)
    _last_time: float = field(init=False, default_factory=time.time)
    _seen_batches: int = field(init=False, default=0)
    _last_seen_batches: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        # create rolling windows for each tracked key
        for k in set(self.mean_keys) | set(self.sum_keys) | {self.weight_key}:
            # weight_key also gets a window if you want rolling denominators
            self._roll[k] = RollingWindow(self.window)
        self.reset_epoch()

    def reset_epoch(self) -> None:
        self._epoch_sums.clear()
        for rw in self._roll.values():
            rw.reset()
        now = time.time()
        self._timer_start = now
        self._last_time = now
        self._seen_batches = 0
        self._last_seen_batches = 0

    def update(self, metrics: Mapping[str, Any]) -> None:
        """
        Update from a single dict (your Trainer passes once).

        Expected:
          - metrics[self.weight_key] exists if any mean_keys are present
          - mean_keys values can be tensors/floats
          - sum_keys values can be tensors/ints
        """
        # weight (for loss-like means)
        w = _to_float(metrics.get(self.weight_key, None))
        if w is None:
            w = 0.0

        # Always track weight_key itself as a rolling sum and epoch sum (useful for denominators)
        self._roll[self.weight_key].update(w, weight=1.0)
        self._epoch_sums[self.weight_key] = self._epoch_sums.get(self.weight_key, 0.0) + w

        # rolling means + epoch weighted sums
        for k in self.mean_keys:
            v = _to_float(metrics.get(k, None))
            if v is None:
                continue
            self._roll[k].update(v, weight=w)
            # store epoch weighted sum for k: sum(v * w)
            self._epoch_sums[f"{k}_sum"] = self._epoch_sums.get(f"{k}_sum", 0.0) + v * w

        # rolling sums + epoch sums
        for k in self.sum_keys:
            v = _to_int(metrics.get(k, None))
            if v is None:
                continue
            self._roll[k].update(v, weight=1.0)
            self._epoch_sums[k] = self._epoch_sums.get(k, 0.0) + v

        self._seen_batches += 1

    # ---- timing ----
    def avg_batch_time(self) -> float:
        cur = time.time()
        elapsed = cur - self._last_time
        self._last_time = cur
        nb = self._seen_batches - self._last_seen_batches
        self._last_seen_batches = self._seen_batches
        return elapsed / max(1, nb)

    def epoch_time(self) -> float:
        return time.time() - self._timer_start

    # ---- rolling access ----
    def rolling_mean(self, key: str) -> float:
        return self._roll[key].weighted_mean() if key in self._roll else 0.0

    def rolling_sum(self, key: str) -> float:
        return self._roll[key].total() if key in self._roll else 0.0

    def rolling_denominator(self) -> float:
        return self.rolling_sum(self.weight_key)
    
    def rolling_snapshot(self) -> dict[str, float]:
        snap = {k: self.rolling_sum(k) for k in self.sum_keys}
        snap[self.weight_key] = self.rolling_sum(self.weight_key)
        for k in self.mean_keys:
            snap[k] = self.rolling_mean(k)
        return snap
    
    def epoch_snapshot_ddp(self, *, device: Optional[torch.device] = None) -> dict[str, float]:
        """
        DDP-safe epoch snapshot:
        - reduces internal epoch sums across ranks
        - returns the same 'epoch_snapshot()' schema
        """
        reduced = self.sync_epoch_snapshot_ddp(device=device)

        snap: dict[str, float] = {}
        denom = float(reduced.get(self.weight_key, 0.0))
        snap[self.weight_key] = denom

        for k in self.mean_keys:
            num = float(reduced.get(f"{k}_sum", 0.0))
            snap[k] = (num / denom) if denom > 0 else 0.0

        for k in self.sum_keys:
            snap[k] = float(reduced.get(k, 0.0))

        return snap


    def epoch_snapshot(self) -> dict[str, float]:
        snap = {k: self.epoch_sum(k) for k in self.sum_keys}
        snap[self.weight_key] = self.epoch_denominator()
        for k in self.mean_keys:
            snap[k] = self.epoch_weighted_mean(k)
        return snap


    # ---- epoch access ----
    def epoch_weighted_mean(self, key: str) -> float:
        denom = self._epoch_sums.get(self.weight_key, 0.0)
        num = self._epoch_sums.get(f"{key}_sum", 0.0)
        return float(num / denom) if denom > 0 else 0.0

    def epoch_sum(self, key: str) -> float:
        return float(self._epoch_sums.get(key, 0.0))

    def epoch_denominator(self) -> float:
        return float(self._epoch_sums.get(self.weight_key, 0.0))

    # ---- optional DDP snapshot (sync sums only) ----
    def sync_epoch_snapshot_ddp(self, *, device: Optional[torch.device] = None) -> dict[str, float]:
        """
        Sum-reduce epoch sums across ranks.
        Call on all ranks; print from rank 0.
        """
        snap = dict(self._epoch_sums)  # local
        if not (self.ddp_sync and torch.distributed.is_available() and torch.distributed.is_initialized()):
            return snap

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        keys = sorted(snap.keys())
        t = torch.tensor([snap[k] for k in keys], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        return {k: float(v) for k, v in zip(keys, t.tolist())}