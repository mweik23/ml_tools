from dataclasses import dataclass
from typing import Iterator, Optional, Tuple, Any, TypedDict
import torch

class LossDict(TypedDict):
    bce: torch.nn.Module
    mmd: torch.nn.Module
    label: str

def split_output(batch_vals, n_s=None, domains=('Source', 'Target')):
    out = {}
    for d in domains:
        if d=='Source':
            out[d] = {
                k: v[:n_s] for k, v in batch_vals.items()
            }
        elif d=='Target':
            out[d] = {
                k: v[n_s:] for k, v in batch_vals.items()
            }
        else:
            raise ValueError(f"Unknown domain: {d}")
    return out  

class MetricHistory:
    def __init__(self):
        self.storage = {}  # e.g., {'train_loss': [..], 'val_acc': [..]}

    def append(self, **kwargs):
        # usage: history.append(train_loss=0.23, val_loss=0.31, train_acc=0.88)
        for k, v in kwargs.items():
            self.storage.setdefault(k, []).append(v)
            
    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.storage[k] = v

    def get(self, key, default=None):
        return self.storage.get(key, default)
    
    @property
    def mmd_norm(self):
        if 'init_loss' in self.storage:
            init_loss = self.storage['init_loss']
            if isinstance(init_loss, dict):
                bce = init_loss.get('BCE', None)
                mmd = init_loss.get('MMD', None)
                if bce is not None and mmd is not None and mmd != 0:
                    return bce / mmd
        return None

    def to_dict(self):
        return dict(self.storage)

    def reset(self):
        self.storage.clear()

def _iter_cycle_no_cache(loader: Iterator):
    """Cycle a DataLoader by recreating its iterator on exhaustion (DDP-safe, no caching)."""
    it = iter(loader)
    while True:
        try:
            yield next(it)
        except StopIteration:
            it = iter(loader)  # restart within the same epoch

def pair_by_source_len(
    primary_loader: Optional[Iterator] = None,
    secondary_loader: Optional[Iterator] = None,
    *,
    repeat_secondary: bool = True,   # repeat target when shorter; if False, yield None after it exhausts
):
    """
    Yields idx, (batch_prim, batch_sec).

    - If primary_loader is None and secondary_loader is not None: iterate target only.
        -> Yields idx, (None, batch_sec)
    - If primary_loader is provided:
        * run exactly len(primary_loader) steps
        * if secondary_loader is provided and repeat_secondary==True, target repeats safely
        * if repeat_secondary==False, target becomes None after it exhausts
    - If both are None: raises.
    """
    if primary_loader is None and secondary_loader is None:
        raise ValueError("Need at least one loader")

    if primary_loader is None:
        for idx, b_sec in enumerate(secondary_loader):
            yield idx, (None, b_sec)
        return

    # Source present → epoch length == len(source)
    prim_it = iter(primary_loader)
    sec_it = None
    if secondary_loader is not None:
        sec_it = _iter_cycle_no_cache(secondary_loader) if repeat_secondary else iter(secondary_loader)

    for idx in range(len(primary_loader)):
        b_prim = next(prim_it)
        if sec_it is None:
            b_sec = None
        else:
            try:
                b_sec = next(sec_it)
            except StopIteration:
                # only happens when repeat_secondary=False and target exhausted
                b_sec = None
        yield idx, (b_prim, b_sec)

def split_st(data, domains=('Source', 'Target')):
    if len(domains)==1:
        return [data, None] if domains[0]=='Source' else [None, data]
    ns = data['n_s']
    split_data=[{} for _ in range(2)]
    for k, v in data.items():
        if k != 'n_s':
            split_data[0][k] = v[:ns]
            split_data[1][k] = v[ns:]
    split_data[0]['n_s'] = ns
    return split_data

def merge_st(data_s, data_t, ns=None):
    merged = {}
    for k in data_s.keys():
        if k != 'n_s':
            merged[k] = torch.cat([data_s[k], data_t[k]], dim=0)
    merged['n_s'] = ns if ns is not None else len(data_s['is_signal'])
    return merged

class Event:
    pass

@dataclass(frozen=True)
class Initialization(Event):
    bce_estimate: Optional[float] = None  # initial BCE loss estimate (for MMD norm)
    mmd_estimate: Optional[float] = None  # initial MMD loss estimate

@dataclass(frozen=True)
class TrainEpochStart(Event):
    pass
@dataclass(frozen=True)
class ValidEpochStart(Event):
    pass
@dataclass(frozen=True)
class TestEpochStart(Event):
    pass

@dataclass(frozen=True)
class EndFirstVal(Event):
    bce: float
    mmd: Optional[float] = None
    
    