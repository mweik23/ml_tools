import torch
from pathlib import Path
from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from collections import deque
from dataclasses import dataclass
import time

def get_correct(pred, label):
        predict = pred.max(1).indices
        correct = torch.sum(predict == label).item()
        return correct
    
#masks controls the number of domains to calculate metrics for
def get_batch_metrics(batch, loss_fns, mmd_coef=1.0, use_tar_labels=False, domains=['Source']):
    if type(domains) is str:
        domains = [domains]
    if 'mmd' in loss_fns:
        if use_tar_labels:
            mmd_val = (loss_fns['mmd'](batch['Source']['encoder'][batch['Source']['label']==0], batch['Target']['encoder'][batch['Target']['label']==0]) 
                    + loss_fns['mmd'](batch['Source']['encoder'][batch['Source']['label']==1], batch['Target']['encoder'][batch['Target']['label']==1]))/2
        else:
            mmd_val = loss_fns['mmd'](batch['Source']['encoder'], batch['Target']['encoder'])
    else:
        mmd_val=None
    out = {d: {} for d in domains}
    for d in domains:
        out[d]['batch_size'] = batch[d]['label'].size(0)
        out[d]['correct'] = get_correct(batch[d]['pred'], batch[d]['label'])
        out[d]['BCE_loss'] = loss_fns['bce'](batch[d]['pred'], batch[d]['label'])
        if mmd_val is not None:
            out[d]['MMD_loss'] = mmd_val * mmd_coef
    return out

@dataclass
class RunningStats:
    """Keeps a rolling window for display AND full-epoch totals (no DDP)."""
    window: int
    domain: str = 'Source'  # for display purposes only

    def __post_init__(self):
        w = max(1, int(self.window))
        self._bce = deque(maxlen=w)
        self._mmd = deque(maxlen=w)
        self._correct = deque(maxlen=w)
        self._batch_sizes = deque(maxlen=w)
        self._initial_time = time.time()
        self._last_time = self._initial_time
        self._seen_batches = 0
        self._last_seen_batches = 0
        self.reset_epoch()

    # ---- per-batch update ----
    def update(self, *, BCE_loss: float, correct: int, batch_size: int, MMD_loss: Any = None):
        #detach
        BCE_loss = BCE_loss.detach().cpu().item()
        if MMD_loss is not None:
            MMD_loss = MMD_loss.detach().cpu().item()

        # window buffers
        self._bce.append(float(BCE_loss))
        if MMD_loss is not None:
            self._mmd.append(float(MMD_loss))
        self._correct.append(int(correct))
        self._batch_sizes.append(int(batch_size))
        self._seen_batches += 1
        # epoch totals
        self.epoch_bce_sum += float(BCE_loss) * int(batch_size)   # sample-weighted
        if MMD_loss is not None:
            self.epoch_mmd_sum += float(MMD_loss) * int(batch_size)   # sample-weighted
        self.epoch_correct += int(correct)
        self.epoch_count   += int(batch_size)

    def avg_batch_time(self) -> float:
        cur_time = time.time()
        
        # time since last call
        elapsed = cur_time-self._last_time
        self._last_time = cur_time
        # batches since last call
        num_batches = self._seen_batches - self._last_seen_batches
        self._last_seen_batches = self._seen_batches
        return elapsed / max(1, num_batches)

    # ---- windowed properties (for periodic display) ----
    @property
    def running_bce(self) -> float:
        bs = sum(self._batch_sizes)
        return (sum(self._bce[i] * self._batch_sizes[i] for i in range(len(self._bce))) / max(1, bs)) if bs else 0.0

    @property
    def running_mmd(self) -> float:
        bs = sum(self._batch_sizes)
        return (sum(self._mmd[i] * self._batch_sizes[i] for i in range(len(self._mmd))) / max(1, bs)) if bs else 0.0

    @property
    def running_acc(self) -> float:
        bs = sum(self._batch_sizes)
        return (sum(self._correct) / max(1, bs)) if bs else 0.0

    # ---- epoch aggregates (local, pre-DDP) ----
    def epoch_loss_avgs(self) -> tuple[float, float]:
        """Return (bce_avg, mmd_avg) weighted by samples across the whole epoch (local rank only)."""
        if self.epoch_count == 0:
            return 0.0, 0.0
        return self.epoch_bce_sum / self.epoch_count, self.epoch_mmd_sum / self.epoch_count

    def epoch_accuracy(self) -> float:
        return (self.epoch_correct / self.epoch_count) if self.epoch_count else 0.0
    
    def epoch_time(self) -> float:
        """Return the total time elapsed since the start of the epoch."""
        return time.time() - self._initial_time

    def reset_epoch(self):
        self.epoch_bce_sum = 0.0
        self.epoch_mmd_sum = 0.0
        self.epoch_correct = 0
        self.epoch_count   = 0
        self._bce.clear(); self._mmd.clear(); self._correct.clear(); self._batch_sizes.clear()
        self._initial_time = time.time()
        self._last_time = self._initial_time 
        self._seen_batches = 0
        self._last_snap_batches = self._seen_batches

def buildROC(labels, score, targetEff=[0.3,0.5]):
    r''' ROC curve is a plot of the true positive rate (Sensitivity) in the function of the false positive rate
    (100-Specificity) for different cut-off points of a parameter. Each point on the ROC curve represents a
    sensitivity/specificity pair corresponding to a particular decision threshold. The Area Under the ROC
    curve (AUC) is a measure of how well a parameter can distinguish between two diagnostic groups.
    '''
    if not isinstance(targetEff, list):
        targetEff = [targetEff]
    fpr, tpr, threshold = roc_curve(labels, score)
    idx = [np.argmin(np.abs(tpr - Eff)) for Eff in targetEff]
    eB, eS = fpr[idx], tpr[idx]
    return fpr, tpr, threshold, eB, eS
def make_roc_curve(fpr, tpr, domain='Source', ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    mask = fpr>0
    ax.plot(tpr[mask], 1/(fpr[mask]), label=domain)
    return ax

def get_test_metrics(labels, scores, targetEff=[0.3, 0.5], **kwargs):
    fpr, tpr, _, eB, eS = buildROC(labels, scores, targetEff=targetEff)
    auc = np.trapz(tpr, fpr)
    ax = make_roc_curve(fpr, tpr, **kwargs)
    metrics = {f'1/eB ~ {eff}': 1/eB[i] for i, eff in enumerate(targetEff)}
    metrics['auc'] = auc
    return metrics, ax