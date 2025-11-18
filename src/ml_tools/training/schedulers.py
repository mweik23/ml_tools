import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, ConstantLR, ExponentialLR, ChainedScheduler, LinearLR
import numpy as np
# types
from dataclasses import dataclass
from typing import Literal, Callable, Optional

class WarmupThenPlateau:
    def __init__(self, optimizer, *, warmup_steps:Optional[int]=None, warmup_epochs:Optional[int]=None,
                 start_factor:float=0.0, end_factor:float=1.0, plateau_kwargs:Optional[dict]=None):
        plateau_kwargs = plateau_kwargs or dict(mode="min", factor=0.5, patience=5)
        assert (warmup_steps is None) ^ (warmup_epochs is None), "Specify steps OR epochs."

        total_iters = warmup_steps if warmup_steps is not None else warmup_epochs
        self.per_step = warmup_steps is not None

        self.warmup = LinearLR(optimizer, start_factor=start_factor, total_iters=total_iters)
        self.plateau = ReduceLROnPlateau(optimizer, **plateau_kwargs)
        self.mode = plateau_kwargs.get("mode", "min")

    def step_batch(self):
        if self.per_step and self.warmup.last_epoch < self.warmup.total_iters:
            self.warmup.step()
            
    #call one step before the first epoch
    def step_epoch(self, val_metric: float = None):
        if not self.per_step and self.warmup.last_epoch < self.warmup.total_iters:
            self.warmup.step()
        else:
            if val_metric is None:
                if self.mode == "min":
                    val_metric = float('inf')
                elif self.mode == 'max':
                    val_metric = float('-inf')
                else:
                    raise ValueError(f"Unknown mode {self.mode} for ReduceLROnPlateau")
            self.plateau.step(val_metric)
'''
call for WarmupThenPlateau::

SchedConfig(
    kind="warmup_plateau",
    lr_min=1e-5,
    warmup_epochs=5,
    factor = 0.5,
    patience = 3,
    mode = "min"
    )
'''
@dataclass
class SchedConfig:
    kind: Literal[
        "cosine_warmup", "onecycle", "step", "multistep", "plateau", "cosine_warmup"
    ] =  "warmup_plateau"
    lr_min: float = 0.0
    warmup_epochs: int = 10
    step_size: int = 30
    gamma: float = 0.1
    milestones: tuple[int, ...] = ()
    pct_start: float = 0.3  # for OneCycle
    mode: Literal["min","max"] = "min"  # for Plateau
    factor: float = 0.5
    patience: int = 5
    threshold: float = 1e-4  # for Plateau

def make_scheduler(optimizer, cfg: SchedConfig, *, total_steps: Optional[int]=None, steps_per_epoch: int=1):
    if cfg.kind == "cosine_warmup":
        def lf(step):
            if step < cfg.warmup_steps:
                return (step + 1) / max(1, cfg.warmup_steps)
            # cosine from 1.0 -> lr_min ratio
            prog = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
            return cfg.lr_min + (1 - cfg.lr_min) * 0.5 * (1 + torch.cos(torch.pi * prog))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lf)

    if cfg.kind == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[g['lr'] for g in optimizer.param_groups],
            total_steps=total_steps,
            pct_start=cfg.pct_start,
            anneal_strategy="cos",
            cycle_momentum=False,
        )

    if cfg.kind == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

    if cfg.kind == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(cfg.milestones), gamma=cfg.gamma)

    if cfg.kind == "plateau":
        return ReduceLROnPlateau(
            optimizer, mode=cfg.mode, factor=cfg.factor, patience=cfg.patience, verbose=False
        )
    if cfg.kind == "warmup_plateau":
        if steps_per_epoch == 1:
            return WarmupThenPlateau(
            optimizer,
            warmup_epochs=cfg.warmup_epochs,
            start_factor=cfg.lr_min,
            plateau_kwargs=dict(mode=cfg.mode, factor=cfg.factor, patience=cfg.patience, threshold=cfg.threshold)
        )
        else:
            return WarmupThenPlateau(
                optimizer,
                warmup_steps=cfg.warmup_epochs * steps_per_epoch,
                start_factor=cfg.lr_min,
                plateau_kwargs=dict(mode=cfg.mode, factor=cfg.factor, patience=cfg.patience, threshold=cfg.threshold)
            )
    raise ValueError(f"Unknown scheduler kind: {cfg.kind}")


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        warmup_epoch: target learning rate is reached at warmup_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    Reference:
        https://github.com/ildoonet/pytorch-gradual-warmup-lr
    """

    def __init__(self, optimizer, multiplier, warmup_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.warmup_epoch = warmup_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    @property
    def _warmup_lr(self):
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch + 1) / self.warmup_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * (self.last_epoch + 1) / self.warmup_epoch + 1.) for base_lr in self.base_lrs]

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch - 1:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return self._warmup_lr

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        self.last_epoch = self.last_epoch + 1 if epoch==None else epoch
        if self.last_epoch >= self.warmup_epoch - 1:
            if not self.finished:
                warmup_lr = [base_lr * self.multiplier for base_lr in self.base_lrs]
                for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                    param_group['lr'] = lr
                self.finished = True
                return
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.warmup_epoch)
            return

        for param_group, lr in zip(self.optimizer.param_groups, self._warmup_lr):
            param_group['lr'] = lr

    def step(self, metrics=None, epoch=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.warmup_epoch)
                self.last_epoch = self.after_scheduler.last_epoch + self.warmup_epoch + 1
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        result = {key: value for key, value in self.__dict__.items() if key != 'optimizer' or key != "after_scheduler"}
        if self.after_scheduler:
            result.update({"after_scheduler": self.after_scheduler.state_dict()})
        return result

    def load_state_dict(self, state_dict):
        after_scheduler_state = state_dict.pop("after_scheduler", None)
        self.__dict__.update(state_dict)
        if after_scheduler_state:
            self.after_scheduler.load_state_dict(after_scheduler_state)

def make_chained(optimizer, factors, epochs):
    const_sched = ConstantLR(optimizer, factor=factors[0], total_iters=epochs)
    gamma_val = (factors[1]/factors[0])**(1/(epochs-1))
    exp_sched = ExponentialLR(optimizer, gamma=gamma_val)
    chained_scheduler = ChainedScheduler([const_sched, exp_sched])
    #chained_scheduler.last_epoch=-1
    return chained_scheduler

class LinearLambda:

    def __init__(self, rates, epochs):
        self.slope = (rates[1]-rates[0])/(epochs[1]-epochs[0])
        self.intercept = rates[0] - self.slope*epochs[0]

    def __call__(self, epoch):
        return self.slope*epoch + self.intercept 

class ExpLambda:

    def __init__(self, rates, epochs):
        self.factor = (rates[1]/rates[0])**(1/(epochs[1]-epochs[0]))
        self.start_rate = rates[0]
        self.start_epoch = epochs[0]

    def __call__(self, epoch):
        return self.start_rate*self.factor**(epoch-self.start_epoch)

class ParticleNetLambda:

    def __init__(self, rates, epochs, types):
        self.lambdas = [LinearLambda(r, e) if t=='linear' else ExpLambda(r, e) for r, e, t in zip(rates, epochs, types)]
        self.epochs = epochs
    
    def __call__(self, epoch):
        #print('epoch', epoch)
        epoch = float(epoch)
        if epoch < self.epochs[0][0]:
            return self.lambdas[0](self.epochs[0][0])
        elif epoch > self.epochs[-1][-1]:
            return self.lambdas[-1](self.epochs[-1][-1])
        condlist = [epoch>=e[0] for e in self.epochs]
        return float(np.piecewise(epoch, condlist, self.lambdas))