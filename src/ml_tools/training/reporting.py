from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import Optional
from scipy.stats import norm
from typing import Mapping, Optional, Callable, Any

def _fmt_metrics(metrics: Mapping[str, Any], *, order: tuple[str, ...] = ()) -> str:
    """
    Format a flat dict of scalar metrics as: 'Loss 0.1234, Acc 0.9876, ...'
    - Skips None.
    - Uses `order` first, then remaining keys alphabetically.
    """
    def fmt_one(k: str, v: Any) -> Optional[str]:
        if v is None:
            return None
        # ints as ints, floats as 4dp by default
        if isinstance(v, bool):
            return f"{k} {v}"
        if isinstance(v, int):
            return f"{k} {v}"
        try:
            fv = float(v)
        except Exception:
            return f"{k} {v}"
        # heuristics: time-like keys or large values
        if k.lower().endswith("time_s") or k.lower().endswith("time"):
            return f"{k} {fv:.1f}s"
        return f"{k} {fv:.4f}"

    keys = []
    seen = set()
    for k in order:
        if k in metrics:
            keys.append(k); seen.add(k)
    for k in sorted(metrics.keys()):
        if k not in seen:
            keys.append(k)

    parts = []
    for k in keys:
        s = fmt_one(k, metrics[k])
        if s is not None:
            parts.append(s)
    return ", ".join(parts)


def display_epoch_summary(
    *,
    partition: str,
    epoch: int,
    tot_epochs: int,
    time_s: float,
    metrics: Mapping[str, Any],
    domain: str = "Source",
    best_epoch: int | None = None,
    best_val: float | None = None,
    logger=None,
    metric_order: tuple[str, ...] = ("loss", "bce", "mmd", "acc", "dice", "precision", "recall", "auc", "r30"),
) -> str:
    log: Callable[[str], None] = logger.info if logger else print

    header = 124 * "-" + "\n"
    body = (
        f"Domain: {domain} [{partition}] Epoch {epoch}/{tot_epochs} — "
        + _fmt_metrics({**metrics, "time_s": time_s}, order=metric_order + ("time_s",))
    )
    log(header + body)

    tail = ""
    if partition == "validation" and best_epoch is not None and best_val is not None:
        tail += f"  (best val epoch {best_epoch} with loss {best_val:.4f})\n"
    tail += 124 * "-"
    log(tail)

    return header + body + "\n" + tail

def display_status(
    *,
    phase: str,
    epoch: int,
    tot_epochs: int,
    batch_idx: int,
    num_batches: int,
    metrics: Mapping[str, Any],
    avg_batch_time: float,
    domain: str = "Source",
    logger=None,
    is_master: bool = True,
    metric_order: tuple[str, ...] = ("loss", "bce", "mmd", "acc", "dice", "precision", "recall"),
) -> str:
    if not is_master:
        return ""

    log: Callable[[str], None] = logger.info if logger else print

    msg = (
        f">> {phase} ({domain}):\tEpoch {epoch}/{tot_epochs}\t"
        f"Batch {batch_idx}/{num_batches}\t"
        + _fmt_metrics({**metrics, "avg_batch_time_s": avg_batch_time}, order=metric_order + ("avg_batch_time_s",))
    )
    log(msg)
    return msg


def finish_roc_plot(path, ax, is_primary=True):
    if is_primary:
        ax.set_xlabel('tpr')
        ax.set_ylabel('1/fpr')
        ax.set_xlim([0, 1])
        ax.set_yscale('log')
        ax.legend(frameon=False)
        fig = ax.figure
        fig.tight_layout()
        fig.savefig(f"{path}/ROC_curve.pdf", dpi=300, bbox_inches="tight")                                        
        return ax
    return None



def make_train_plt(train_metrics, path, pretrained=False, do_MMD=False, rename_map={}, main_loss_name='BCE'):
    main_train_loss_key = f'train_{main_loss_name}'
    main_val_loss_key = f'val_{main_loss_name}'
    keys_needed = ['epochs', main_train_loss_key, main_val_loss_key]
    if do_MMD:
        keys_needed += ['train_MMD', 'val_MMD', 'train_loss', 'val_loss']

    for k, v in rename_map.items():
        if k in train_metrics:
            if v not in keys_needed: 
                print(f"Warning: Attempting to rename {k} to {v} but {v} not in keys_needed")
            train_metrics[v] = train_metrics.pop(k)
    assert all(k in train_metrics for k in keys_needed), f"Missing keys in train_metrics. Found {list(train_metrics.keys())}, need {keys_needed}"
    if pretrained:
        train_start = 1
    else:
        train_start = 0
    fig, ax = plt.subplots()
    ax.plot(train_metrics['epochs'][train_start:], train_metrics[main_train_loss_key], color='b', linestyle='dotted' if do_MMD else 'solid', label=main_train_loss_key)
    ax.plot(train_metrics['epochs'], train_metrics[main_val_loss_key], color='r', linestyle='dotted' if do_MMD else 'solid', label=main_val_loss_key)

    if do_MMD:
        ax.plot(train_metrics['epochs'][train_start:], train_metrics['train_MMD'], color='b', linestyle='dashed', label='train MMD')
        ax.plot(train_metrics['epochs'][train_start:], train_metrics['train_loss'], color='b', linestyle='solid', label='train total')
        ax.plot(train_metrics['epochs'], train_metrics['val_MMD'], color='r', linestyle='dashed', label='val MMD')
        ax.plot(train_metrics['epochs'], train_metrics['val_loss'], color='r', linestyle='solid', label='val total')
    ax.legend(frameon=False)
    ax.set_ylim([-0.1, .6])
    fig.savefig(f"{path}/loss_vs_epochs.pdf")
    plt.close(fig)
    return None

def make_logits_plt(logit_diffs, path, name='final', domains=None):
    if domains is not None and list(logit_diffs.keys())==['Mixed']:
        # split Mixed into Source and Target
        logit_diffs = {'Source': logit_diffs['Mixed'][domains['Mixed']==0], 'Target': logit_diffs['Mixed'][domains['Mixed']==1]}   
    bins = np.histogram_bin_edges(np.concatenate(list(logit_diffs.values())), bins='auto')
    xlims = get_xlims(logit_diffs)
    plt.figure()
    for d, l in logit_diffs.items():
        plt.hist(l, bins=bins, histtype='step', label=d, density=True)
    plt.xlabel('logit difference')
    plt.xlim(xlims)
    plt.legend(frameon=False)
    plt.savefig(f"{path}/logit_diff_{name}.pdf")
    plt.close()
    return None

# set xlims assuming gaussian-like tails. Outliers will be cut off, but plot will be focused on the main distribution.
def get_xlims(logit_diffs, z_lim=4, z_set=2):
    quant_set = norm.cdf(z_set)
    quants = np.array([1-quant_set, 0.5, quant_set])
    vals = [np.quantile(v, quants) for v in logit_diffs.values()]
    min_idx = np.argmin(np.array([v[0] for v in vals]))
    max_idx = np.argmax(np.array([v[-1] for v in vals]))
    xmin = vals[min_idx][0] - (z_lim/z_set) * (vals[min_idx][1]-vals[min_idx][0])
    xmax = vals[max_idx][0] + (z_lim/z_set) * (vals[max_idx][-1]-vals[max_idx][0])
    return [xmin, xmax]