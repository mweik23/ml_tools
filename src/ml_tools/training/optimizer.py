import torch
import torch.nn as nn
import torch.optim as optim
from typing import List


def get_param_groups(model: nn.Module, lr: float, weight_decay: float) -> List[dict]:
    """Get parameter groups for optimizer."""
    params = model.parameters()
    param_groups = [{
        "params": list(params),
        "lr": lr,
        "weight_decay": weight_decay,
    }]
    total_params = sum(p.numel() for p in params)
    trainable_params = sum(p.numel() for p in params if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    return param_groups

def get_optimizer(model: nn.Module,
                  optimizer_type: str,
                  lr: float,
                  weight_decay: float,
                  momentum: float = 0.9) -> optim.Optimizer:
    param_groups = get_param_groups(model, lr, weight_decay)
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(param_groups)
    elif optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(param_groups)
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(param_groups, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    for g in optimizer.param_groups:
        g.setdefault('initial_lr', g['lr'])
    return optimizer