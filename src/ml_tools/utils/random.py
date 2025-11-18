import torch
import numpy as np
import random
import os
import torch.distributed as dist

def set_global_seed(seed: int, rank: int = 0):
    seed = seed + rank
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def worker_init_base(_wid: int, rank: int = -1, verbose: bool = False):
    seed = torch.initial_seed() % 2**32
    if verbose:
        print(f"[Rank {rank}] Worker {_wid} seed = {seed}")
    np.random.seed(seed)
    random.seed(seed)