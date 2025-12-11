import torch
import numpy as np
import random
import os
import torch.distributed as dist

PHASE_OFFSET = {
    "collate": 0,
    "model": 1_00_000,
    "train": 2_000_000,
    "val":   3_000_000,
    "test":  4_000_000,
}

def make_and_set_seed(base_seed: int, phase: str, epoch: int=0, rank: int=0) -> int:
    seed = make_epoch_seed(base_seed, phase, epoch, rank)
    set_global_seed(seed)
    return seed

def make_epoch_seed(base_seed: int, phase: str, epoch: int, rank: int) -> int:
    return base_seed + PHASE_OFFSET[phase] + epoch * 10_000 + rank

def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_base(_wid: int, rank: int = -1, verbose: bool = False):
    seed = torch.initial_seed() % 2**32
    if verbose:
        print(f"[Rank {rank}] Worker {_wid} seed = {seed}")
    np.random.seed(seed)
    random.seed(seed)