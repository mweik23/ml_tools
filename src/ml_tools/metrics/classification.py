# src/ml_tools/metrics/classification.py
from __future__ import annotations

from typing import Optional, Any
import torch


@torch.no_grad()
def topk_correct(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    k: int = 1,
    dim: int = 1,
) -> int:
    """
    Count top-k correct predictions (multiclass).

    logits: (N, C, ...)
    labels: (N, ...) int class ids
    Returns: python int
    """
    if k <= 1:
        pred = logits.argmax(dim=dim)
        return int((pred == labels).sum().item())

    topk = logits.topk(k, dim=dim).indices  # (N, k, ...)
    # For standard classification, labels are (N,) and logits are (N,C).
    # We support extra dims by aligning and comparing elementwise.
    # Expand labels to match topk shape along the k dimension.
    labels_exp = labels.unsqueeze(dim).expand_as(topk)
    return int((topk == labels_exp).any(dim=dim).sum().item())


@torch.no_grad()
def binary_correct_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float = 0.5,
    label_threshold: Optional[float] = None,
    valid_mask: Optional[torch.Tensor] = None,
) -> int:
    """
    Count correct predictions for binary classification from logits.

    logits: (N, 1, ...) or (N, ...)
    labels: same shape; can be soft in [0,1] or hard {0,1}

    - threshold applies to sigmoid(logits)
    - label_threshold (default threshold if labels are soft) defaults to threshold
    - valid_mask: bool mask (same shape) indicating which elements to include
    """
    probs = torch.sigmoid(logits)
    pred = probs >= threshold

    if label_threshold is None:
        label_threshold = threshold
    targ = labels >= label_threshold

    if valid_mask is not None:
        pred = pred[valid_mask]
        targ = targ[valid_mask]

    return int((pred == targ).sum().item())


@torch.no_grad()
def multiclass_confusion_matrix(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: Optional[int] = None,
    dim: int = 1,
) -> torch.Tensor:
    """
    Compute a confusion matrix for multiclass classification.

    Returns: (C, C) tensor on CPU, where rows=true, cols=pred.
    """
    pred = logits.argmax(dim=dim)

    # flatten
    pred = pred.reshape(-1)
    labels = labels.reshape(-1)

    if ignore_index is not None:
        keep = labels != ignore_index
        pred = pred[keep]
        labels = labels[keep]

    # bincount trick
    idx = labels * num_classes + pred
    cm = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm.cpu()


@torch.no_grad()
def accuracy_from_correct(correct: int, total: int) -> float:
    return float(correct / total) if total > 0 else 0.0


@torch.no_grad()
def classification_batch_metrics(
    logits: torch.Tensor,          # (N, C)
    labels: torch.Tensor,          # (N,)
    *,
    loss: Optional[torch.Tensor] = None,
    loss_fn: Optional[Any] = None,
    topk: int = 1,
) -> dict[str, Any]:
    if loss is None and loss_fn is not None:
        loss = loss_fn(logits, labels)

    correct = topk_correct(logits, labels, k=topk)
    batch_size = int(labels.shape[0])

    out: dict[str, Any] = {
        "correct": correct,
        "batch_size": batch_size,
    }
    if loss is not None:
        out["loss"] = loss
    return out