from pathlib import Path
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.stats import norm
SRC_DIR = (Path(__file__).parents[1]).resolve()
import sys
sys.path.append(str(SRC_DIR))
from utils.distributed import is_master

def display_epoch_summary(*, 
                          partition: str, 
                          epoch: int, 
                          tot_epochs: int, 
                          bce: float, 
                          mmd: float, 
                          acc: float, 
                          time_s: float, 
                          domain: str = 'Source',
                          best_epoch: int = None,
                          best_val: float = None,
                          auc: float = None,
                          r30: float = None,
                          logger=None):
    if not is_master(): 
        return
    msg = (124 * "-" + "\n" +
           f"Domain: {domain} [{partition}] Epoch {epoch}/{tot_epochs} — "
           f"BCE {bce:.4f}, MMD {mmd:.4f}, Acc {acc:.4f}, Time {time_s:.1f}s")
    if auc is not None:
        msg += f", AUC {auc:.4f}"
    if r30 is not None:
        msg += f", R30 {r30:.4f}"
    (logger.info if logger else print)(msg)
    msg=''
    if partition == 'validation' and best_epoch is not None and best_val is not None:
        msg += f"  (best val epoch {best_epoch} with loss {best_val:.4f})\n"
    msg += 124 * "-"
    (logger.info if logger else print)(msg)
    return msg

def display_status(*, phase: str, domain: str, epoch: int, tot_epochs: int,
                   batch_idx: int, num_batches: int,
                   running_bce: float, running_mmd: float,
                   running_acc: float, avg_batch_time: float,
                   logger=None):
    if not is_master():
        return

    msg = (f">> {phase} ({domain}):\tEpoch {epoch}/{tot_epochs}\t"
           f"Batch {batch_idx}/{num_batches}\t"
           f"BCE {running_bce:.4f}\tMMD {running_mmd:.4f}\t"
           f"RunAcc {running_acc:.3f}\t"
           f"AvgBatchTime {avg_batch_time:.4f}s")
    (logger.info if logger else print)(msg)
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



def make_train_plt(train_metrics, path, pretrained=False, do_MMD=False, rename_map={}):
    keys_needed = ['epochs', 'train_BCE', 'val_BCE']
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
    ax.plot(train_metrics['epochs'][train_start:], train_metrics['train_BCE'], color='b', linestyle='dotted' if do_MMD else 'solid', label='train BCE')
    ax.plot(train_metrics['epochs'], train_metrics['val_BCE'], color='r', linestyle='dotted' if do_MMD else 'solid', label='val BCE')

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