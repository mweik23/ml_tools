# src/ml_tools/metrics/segmentation.py
from __future__ import annotations
from typing import Optional, Any

import torch


from dataclasses import dataclass, field
from .core import RollingWindow

@dataclass
class RollingConfusion:
    window: int = 50
    tp: RollingWindow = field(init=False)
    fp: RollingWindow = field(init=False)
    fn: RollingWindow = field(init=False)
    tn: RollingWindow = field(init=False)

    def __post_init__(self):
        self.tp = RollingWindow(self.window)
        self.fp = RollingWindow(self.window)
        self.fn = RollingWindow(self.window)
        self.tn = RollingWindow(self.window)

    def update(self, *, tp: int, fp: int, fn: int, tn: int) -> None:
        self.tp.update(tp)
        self.fp.update(fp)
        self.fn.update(fn)
        self.tn.update(tn)

    def totals(self) -> dict[str, int]:
        return {
            "tp": int(self.tp.total()),
            "fp": int(self.fp.total()),
            "fn": int(self.fn.total()),
            "tn": int(self.tn.total()),
        }

    def metrics(self) -> dict[str, float]:
        t = self.totals()
        return segmentation_agg_metrics(t["tp"], t["fp"], t["fn"])

    def reset(self) -> None:
        self.tp.reset(); self.fp.reset(); self.fn.reset(); self.tn.reset()


@torch.no_grad()
def binary_confusion_from_logits(
    logits: torch.Tensor,          # (N, 1, H, W)
    labels: torch.Tensor,          # (N, 1, H, W) soft or hard in [0,1]
    *,
    pred_threshold: float = 0.5,
    label_threshold: float = 0.5,
    valid_mask: Optional[torch.Tensor] = None,  # (N, 1, H, W) bool, True=keep
) -> dict[str, int]:
    """
    Compute TP/FP/FN/TN for binary segmentation from logits + (soft) labels.

    Returns python ints: tp, fp, fn, tn
    """
    probs = torch.sigmoid(logits)
    pred = (probs >= pred_threshold)
    targ = (labels >= label_threshold)

    if valid_mask is not None:
        pred = pred[valid_mask]
        targ = targ[valid_mask]
    else:
        pred = pred.reshape(-1)
        targ = targ.reshape(-1)

    tp = int((pred & targ).sum().item())
    fp = int((pred & ~targ).sum().item())
    fn = int((~pred & targ).sum().item())
    tn = int((~pred & ~targ).sum().item())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def segmentation_agg_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    tp = int(tp); fp = int(fp); fn = int(fn)
    prec_denom = tp + fp
    rec_denom = tp + fn
    f1_denom = 2 * tp + fp + fn

    precision = tp / prec_denom if prec_denom > 0 else 0.0
    recall = tp / rec_denom if rec_denom > 0 else 0.0
    f1 = (2 * tp) / f1_denom if f1_denom > 0 else 0.0
    dice = f1  # identical for binary
    return {"precision": precision, "recall": recall, "f1": f1, "dice": dice}


@dataclass
class BinarySegmentationAccumulator:
    """
    Accumulates TP/FP/FN/TN across batches.
    Compute metrics at any time (running) or at end of epoch (final).
    """
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update_batch(self, *, tp: int, fp: int, fn: int, tn: int) -> None:
        self.tp += int(tp)
        self.fp += int(fp)
        self.fn += int(fn)
        self.tn += int(tn)

    def metrics(self) -> dict[str, float]:
        return segmentation_agg_metrics(self.tp, self.fp, self.fn)

    def reset(self) -> None:
        self.tp = self.fp = self.fn = self.tn = 0


@torch.no_grad()
def segmentation_batch_metrics(
    logits: torch.Tensor,          # (N, 1, H, W) raw logits
    labels: torch.Tensor,          # (N, 1, H, W) soft or hard in [0,1]
    *,
    loss: Optional[torch.Tensor] = None,   # if you already computed it
    loss_fn: Optional[Any] = None,         # or provide loss_fn to compute it here
    pred_threshold: float = 0.5,
    label_threshold: float = 0.5,
    valid_mask: Optional[torch.Tensor] = None,  # (N, 1, H, W) bool; True=keep
) -> dict[str, Any]:
    """
    Returns a dict suitable for RunningStats.update(...).

    Keys:
      - loss: scalar tensor or float (optional if neither loss nor loss_fn provided)
      - num_pixels: number of evaluated pixels (denominator for weighted mean loss)
      - tp, fp, fn, tn: per-batch confusion counts (python ints)
    """
    if loss is None and loss_fn is not None:
        loss = loss_fn(logits, labels)

    probs = torch.sigmoid(logits)
    pred = probs >= pred_threshold
    targ = labels >= label_threshold

    if valid_mask is not None:
        # ignore pixels where valid_mask is False
        pred = pred[valid_mask]
        targ = targ[valid_mask]
        num_pixels = int(valid_mask.sum().item())
    else:
        pred = pred.reshape(-1)
        targ = targ.reshape(-1)
        num_pixels = int(targ.numel())

    tp = int((pred & targ).sum().item())
    fp = int((pred & ~targ).sum().item())
    fn = int((~pred & targ).sum().item())
    tn = int((~pred & ~targ).sum().item())

    out: dict[str, Any] = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "num_pixels": num_pixels,
    }
    if loss is not None:
        out["loss"] = loss  # tensor is fine; RunningStats can detach
    return out