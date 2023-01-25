import os
import numpy as np
from scipy.stats import poisson
from skimage.transform import rescale, resize
from torch.autograd import Variable
import torch
import torch.nn as nn
import re

from torch.optim.lr_scheduler import StepLR
from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer, required

## CosineAnnealingWarmUpRestarts
import math
from torch.optim.lr_scheduler import _LRScheduler

class LrHandler():
    def __init__(self, train_loader, **kwargs):
        self.final_lr = 1e-7
        self.epoch = kwargs.get('nEpochs')
        self.lr_policy = kwargs.get('lr_policy')
        self.step_size = kwargs.get('lr_step')
        self.base_lr = kwargs.get('lr_init')
        self.kwargs = kwargs
        self.final_lr = 1e-7
        self.num_iterations  = len(train_loader) # number of batches / world_size
        self.epoch = kwargs.get('nEpochs')
        self.total_iterations = self.num_iterations * self.epoch
        self.lr_policy = kwargs.get('lr_policy')
        self.gamma = 0.5 if self.lr_policy == 'SGDR' else kwargs.get('lr_gamma')
        self.warmup = int(self.total_iterations * 0.05) if kwargs.get('lr_warmup') is None else kwargs.get('lr_warmup')
        self.T_0 = int(0.3 * self.total_iterations)
        self.T_mult = 1 if kwargs.get('lr_T_mult') is None else kwargs.get('lr_T_mult')


    def set_lr(self,dict_lr):
        if self.base_lr is None:
            self.base_lr = dict_lr

    def set_schedule(self,optimizer):
        self.schedule = self.get_scheduler(optimizer) #StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

    def schedule_check_and_update(self, optimizer):
        if self.lr_policy == 'step':
            if optimizer.param_groups[0]['lr'] > self.final_lr:
                self.schedule.step() # 각 iteration 마다 update
                if (self.schedule._step_count - 1) % self.step_size == 0:
                    print('current lr: {}'.format(optimizer.param_groups[0]['lr']))
        else: 
            if int(self.schedule.last_epoch) == 0 : 
                print('initializing warmup')
            elif int(self.schedule.last_epoch) == int(self.warmup) : 
                print('finished warmup')
            self.schedule.step()
            print('current lr: {}'.format(optimizer.param_groups[0]['lr']))
            
                    
    def get_scheduler(self,optimizer):
        
        if self.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        
        elif self.lr_policy == 'SGDR':
            # warmup을 위해 초기 시작 lr 을 0으로 지정.
            # https://gaussian37.github.io/dl-pytorch-lr_scheduler/ 참고
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0 
            scheduler = CosineAnnealingWarmUpRestarts(optimizer, first_cycle_steps=self.T_0, cycle_mult=self.T_mult, max_lr=self.base_lr,min_lr=1e-9, warmup_steps=self.warmup, gamma=self.gamma)
            # T_0 : 최초 주기값 (T_mult에 의해 점점 주기가 커짐)
            # T_mult : 주기가 반복되면서 최초 주기값에 비해 얼만큼 주기를 늘려나갈 것인지 스케일 값에 해당
            # eta_max : lr의 최대값 (warmup 시 eta_max까지 값이 증가)
            # T_up : Warm up 시 필요한 epoch 수를 지정하며 일반적으로 짧은 epoch 수를 지정
            # gamma : lr를 얼마나 줄여나갈 것인지.(eta_max에 곱해지는 스케일값)
        elif self.lr_policy == 'OneCycle':
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.base_lr, steps_per_epoch=self.num_iterations, epochs=self.kwargs.get('nEpochs'))
        elif self.lr_policy == 'CosAnn':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', self.lr_policy)
        return scheduler

    
    
# https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
class CosineAnnealingWarmUpRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
    
    
'''
class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
'''        