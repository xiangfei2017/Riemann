# BSD 3-Clause License
#
# Copyright (c) 2025, Fei Xiang
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""
Learning Rate Scheduler Module for the Riemann Library

This module implements various learning rate scheduling algorithms for training neural networks in the Riemann library.

Learning rate schedulers adjust the learning rate during training to improve model convergence and performance.
This file provides implementations of several widely used scheduling algorithms with PyTorch-compatible interfaces.

Implemented schedulers:
- LRScheduler: Base class providing common scheduler functionality
- StepLR: Decays the learning rate by a factor every step_size epochs
- MultiStepLR: Decays the learning rate by a factor at specified milestones
- ExponentialLR: Decays the learning rate exponentially
- CosineAnnealingLR: Anneals the learning rate using a cosine function
- ReduceLROnPlateau: Reduces the learning rate when a metric has stopped improving

All schedulers support PyTorch-compatible interfaces and can be used with any Riemann optimizer.
"""
from __future__ import annotations
from typing import Any
import math
from .optim import Optimizer


class LRScheduler:
    """
    学习率调度器基类，提供调度器的通用接口和功能。
    
    所有具体的学习率调度算法都应继承此类，并实现get_lr方法以计算新的学习率。
    
    参数:
        optimizer: 待调整学习率的优化器
        last_epoch: 上一个epoch的索引，默认为-1表示从头开始
        verbose: 是否在每次更新时打印学习率信息
    
    属性:
        optimizer: 关联的优化器
        last_epoch: 当前的epoch索引
        verbose: 是否打印学习率信息
    """
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1, verbose: bool = False) -> None:
        """
        初始化学习率调度器
        
        参数:
            optimizer: 待调整学习率的优化器
            last_epoch: 上一个epoch的索引，默认为-1表示从头开始
            verbose: 是否在每次更新时打印学习率信息
        """
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"optimizer must be an Optimizer instance, but got {type(optimizer).__name__}")
        if last_epoch < -1:
            raise ValueError(f"last_epoch must be >= -1, but got {last_epoch}")
        
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.verbose = verbose
        
        # 初始化学习率
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self._last_lr = self.base_lrs.copy()
        
        # 首次调用step()时，last_epoch会从-1变为0
        if last_epoch == -1:
            for i, group in enumerate(optimizer.param_groups):
                group['lr'] = self.base_lrs[i]
    
    def step(self, epoch: int | None = None) -> None:
        """
        执行单个调度步骤，更新学习率
        
        参数:
            epoch: 当前的epoch索引，如果为None则自动递增
        """
        if epoch is None:
            self.last_epoch += 1
        else:
            if epoch < 0:
                raise ValueError(f"epoch must be >= 0, but got {epoch}")
            self.last_epoch = epoch
        
        # 计算新的学习率
        new_lrs = self.get_lr()
        
        # 更新优化器中的学习率
        for i, (group, lr) in enumerate(zip(self.optimizer.param_groups, new_lrs)):
            old_lr = group['lr']
            group['lr'] = lr
            self._last_lr[i] = lr
            
            # 打印学习率信息
            if self.verbose:
                print(f"Epoch {self.last_epoch}: adjusting learning rate of group {i} from {old_lr:.6f} to {lr:.6f}")
    
    def get_lr(self) -> list[float]:
        """
        计算当前epoch的学习率

        返回:
            list[float]: 每个参数组的学习率

        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("Subclasses must implement get_lr method")

    def get_last_lr(self) -> list[float]:
        """
        返回上一次计算的学习率
        
        返回:
            list[float]: 每个参数组的上一次学习率
        """
        return self._last_lr

    def state_dict(self) -> dict[str, Any]:
        """
        返回调度器状态的字典，可用于保存和恢复调度器状态
        
        返回:
            dict: 包含调度器状态的字典
        """
        return {
            'last_epoch': self.last_epoch,
            'base_lrs': self.base_lrs,
            '_last_lr': self._last_lr,
        }
    
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        从state_dict中加载调度器状态
        
        参数:
            state_dict: 包含调度器状态的字典，通常由state_dict()方法生成
        """
        self.last_epoch = state_dict['last_epoch']
        self.base_lrs = state_dict['base_lrs']
        self._last_lr = state_dict['_last_lr']
        
        # 更新优化器中的学习率
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self._last_lr[i]


class StepLR(LRScheduler):
    """
    每step_size个epoch将学习率乘以gamma因子
    
    参数更新公式: lr = base_lr * gamma ^ (epoch // step_size)
    
    适用场景:
        - 训练过程中需要定期降低学习率
        - 简单的学习率调度策略
    """
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1, verbose: bool = False) -> None:
        """
        初始化StepLR调度器
        
        参数:
            optimizer: 待调整学习率的优化器
            step_size: 学习率衰减的步长（单位：epoch）
            gamma: 学习率衰减因子，默认为0.1
            last_epoch: 上一个epoch的索引，默认为-1表示从头开始
            verbose: 是否在每次更新时打印学习率信息
        
        异常:
            ValueError: 当step_size <= 0或gamma <= 0时抛出
        """
        if step_size <= 0:
            raise ValueError(f"step_size must be > 0, but got {step_size}")
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, but got {gamma}")
        
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> list[float]:
        """
        计算当前epoch的学习率

        返回:
            list[float]: 每个参数组的学习率
        """
        return [base_lr * (self.gamma ** ((self.last_epoch + 1) // self.step_size)) for base_lr in self.base_lrs]


class MultiStepLR(LRScheduler):
    """
    在指定的milestones处将学习率乘以gamma因子
    
    参数更新公式: lr = base_lr * gamma ^ (sum(milestone <= epoch))
    
    适用场景:
        - 训练过程中需要在特定epoch降低学习率
        - 基于经验或先验知识设置学习率衰减点
    """
    def __init__(self, optimizer: Optimizer, milestones: list[int], gamma: float = 0.1, last_epoch: int = -1, verbose: bool = False) -> None:
        """
        初始化MultiStepLR调度器
        
        参数:
            optimizer: 待调整学习率的优化器
            milestones: 学习率衰减的epoch列表，必须按升序排列
            gamma: 学习率衰减因子，默认为0.1
            last_epoch: 上一个epoch的索引，默认为-1表示从头开始
            verbose: 是否在每次更新时打印学习率信息
        
        异常:
            ValueError: 当milestones为空、gamma <= 0或milestones未按升序排列时抛出
        """
        if not milestones:
            raise ValueError("milestones must be a non-empty list")
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, but got {gamma}")
        if not all(milestones[i] < milestones[i+1] for i in range(len(milestones)-1)):
            raise ValueError("milestones must be in ascending order")
        
        self.milestones = milestones
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> list[float]:
        """
        计算当前epoch的学习率

        返回:
            list[float]: 每个参数组的学习率
        """
        # 计算当前epoch已经经过了多少个milestone
        num_decays = sum(1 for milestone in self.milestones if milestone <= self.last_epoch + 1)
        return [base_lr * (self.gamma ** num_decays) for base_lr in self.base_lrs]


class ExponentialLR(LRScheduler):
    """
    指数衰减学习率
    
    参数更新公式: lr = base_lr * gamma ^ epoch
    
    适用场景:
        - 训练过程中需要持续平滑地降低学习率
        - 适合长时间训练的模型
    """
    def __init__(self, optimizer: Optimizer, gamma: float, last_epoch: int = -1, verbose: bool = False) -> None:
        """
        初始化ExponentialLR调度器
        
        参数:
            optimizer: 待调整学习率的优化器
            gamma: 学习率衰减因子，必须大于0且小于1
            last_epoch: 上一个epoch的索引，默认为-1表示从头开始
            verbose: 是否在每次更新时打印学习率信息
        
        异常:
            ValueError: 当gamma <= 0时抛出
        """
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, but got {gamma}")
        
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> list[float]:
        """
        计算当前epoch的学习率

        返回:
            list[float]: 每个参数组的学习率
        """
        return [base_lr * (self.gamma ** (self.last_epoch + 1)) for base_lr in self.base_lrs]


class CosineAnnealingLR(LRScheduler):
    """
    使用余弦函数退火学习率
    
    参数更新公式: lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(π * epoch / T_max))
    
    适用场景:
        - 需要精细调整学习率以获得更好的最终性能
        - 适合对学习率敏感的模型
    """
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0, last_epoch: int = -1, verbose: bool = False) -> None:
        """
        初始化CosineAnnealingLR调度器
        
        参数:
            optimizer: 待调整学习率的优化器
            T_max: 余弦退火的周期（单位：epoch）
            eta_min: 最小学习率，默认为0
            last_epoch: 上一个epoch的索引，默认为-1表示从头开始
            verbose: 是否在每次更新时打印学习率信息
        
        异常:
            ValueError: 当T_max <= 0或eta_min < 0时抛出
        """
        if T_max <= 0:
            raise ValueError(f"T_max must be > 0, but got {T_max}")
        if eta_min < 0:
            raise ValueError(f"eta_min must be >= 0, but got {eta_min}")
        
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> list[float]:
        """
        计算当前epoch的学习率

        返回:
            list[float]: 每个参数组的学习率
        """
        # 计算当前epoch在余弦循环中的位置
        # 与PyTorch一致，使用(1 + cos(pi * epoch / T_max)) / 2的形式
        # 并且在T_max后重新开始循环
        epoch = self.last_epoch + 1
        # 计算余弦退火学习率
        cos_term = math.cos(math.pi * epoch / self.T_max)
        return [self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + cos_term) for base_lr in self.base_lrs]


class ReduceLROnPlateau:
    """
    当指标停止改善时降低学习率
    
    当监控的指标在patience个epoch内没有改善时，将学习率乘以factor因子。
    
    适用场景:
        - 基于验证集性能自动调整学习率
        - 适合需要自适应学习率的复杂模型
    """
    def __init__(self, optimizer: Optimizer, mode: str = 'min', factor: float = 0.1, patience: int = 10,
                 verbose: bool = False, threshold: float = 1e-4, threshold_mode: str = 'rel',
                 cooldown: int = 0, min_lr: float | list[float] = 0, eps: float = 1e-8) -> None:
        """
        初始化ReduceLROnPlateau调度器
        
        参数:
            optimizer: 待调整学习率的优化器
            mode: 指标模式，'min'表示指标越小越好，'max'表示指标越大越好
            factor: 学习率衰减因子，默认为0.1
            patience: 等待的epoch数，默认为10
            verbose: 是否在每次更新时打印学习率信息
            threshold: 指标改善的阈值，默认为1e-4
            threshold_mode: 阈值模式，'rel'表示相对改善，'abs'表示绝对改善
            cooldown: 冷却期，在调整学习率后等待的epoch数，默认为0
            min_lr: 最小学习率，默认为0
            eps: 学习率变化的最小阈值，默认为1e-8
        
        异常:
            ValueError: 当mode、threshold_mode无效，或factor、patience、threshold、cooldown、eps为负数时抛出
        """
        if mode not in {'min', 'max'}:
            raise ValueError(f"mode must be 'min' or 'max', but got {mode}")
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError(f"threshold_mode must be 'rel' or 'abs', but got {threshold_mode}")
        if factor >= 1.0:
            raise ValueError(f"factor must be < 1.0, but got {factor}")
        if patience < 0:
            raise ValueError(f"patience must be >= 0, but got {patience}")
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0, but got {threshold}")
        if cooldown < 0:
            raise ValueError(f"cooldown must be >= 0, but got {cooldown}")
        if eps < 0:
            raise ValueError(f"eps must be >= 0, but got {eps}")
        
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.eps = eps
        
        # 处理min_lr
        if isinstance(min_lr, (int, float)):
            self.min_lrs = [float(min_lr) for _ in optimizer.param_groups]
        else:
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(f"min_lr length must match the number of parameter groups, but got {len(min_lr)} vs {len(optimizer.param_groups)}")
            self.min_lrs = min_lr
        
        # 初始化状态
        self.best = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.mode_worse = None  # 用于比较指标是否恶化
        self._init_is_better(mode=mode, threshold=threshold, threshold_mode=threshold_mode)
        self._reset()
    
    def _init_is_better(self, mode: str, threshold: float, threshold_mode: str) -> None:
        """
        初始化is_better函数，用于判断指标是否改善
        """
        if mode == 'min' and threshold_mode == 'rel':
            self.is_better = lambda current, best: current < best * (1 - threshold)
            self.mode_worse = float('inf')
        elif mode == 'min' and threshold_mode == 'abs':
            self.is_better = lambda current, best: current < best - threshold
            self.mode_worse = float('inf')
        elif mode == 'max' and threshold_mode == 'rel':
            self.is_better = lambda current, best: current > best * (1 + threshold)
            self.mode_worse = -float('inf')
        else:  # mode == 'max' and threshold_mode == 'abs'
            self.is_better = lambda current, best: current > best + threshold
            self.mode_worse = -float('inf')
    
    def _reset(self) -> None:
        """
        重置调度器状态
        """
        self.best = self.mode_worse
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
    
    def step(self, metrics: float, epoch: int | None = None) -> None:
        """
        执行单个调度步骤，根据指标更新学习率
        
        参数:
            metrics: 监控的指标值
            epoch: 当前的epoch索引，默认为None
        """
        current = metrics
        
        # 检查是否在冷却期
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
        else:
            # 检查指标是否改善
            if self.is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1
                
                # 检查是否需要降低学习率
                if self.num_bad_epochs > self.patience:
                    self._reduce_lr(epoch)
                    self.cooldown_counter = self.cooldown
                    self.num_bad_epochs = 0
    
    def _reduce_lr(self, epoch: int | None) -> None:
        """
        降低学习率
        """
        for i, (group, min_lr) in enumerate(zip(self.optimizer.param_groups, self.min_lrs)):
            old_lr = group['lr']
            new_lr = max(old_lr * self.factor, min_lr)
            
            # 检查学习率变化是否大于eps
            if old_lr - new_lr > self.eps:
                group['lr'] = new_lr
                
                # 打印学习率信息
                if self.verbose:
                    print(f"Epoch {epoch}: reducing learning rate of group {i} from {old_lr:.6f} to {new_lr:.6f}")
    
    def state_dict(self) -> dict[str, Any]:
        """
        返回调度器状态的字典，可用于保存和恢复调度器状态
        
        返回:
            dict: 包含调度器状态的字典
        """
        return {
            'mode': self.mode,
            'factor': self.factor,
            'patience': self.patience,
            'verbose': self.verbose,
            'threshold': self.threshold,
            'threshold_mode': self.threshold_mode,
            'cooldown': self.cooldown,
            'min_lrs': self.min_lrs,
            'eps': self.eps,
            'best': self.best,
            'num_bad_epochs': self.num_bad_epochs,
            'cooldown_counter': self.cooldown_counter,
        }
    
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        从state_dict中加载调度器状态
        
        参数:
            state_dict: 包含调度器状态的字典，通常由state_dict()方法生成
        """
        self.mode = state_dict['mode']
        self.factor = state_dict['factor']
        self.patience = state_dict['patience']
        self.verbose = state_dict['verbose']
        self.threshold = state_dict['threshold']
        self.threshold_mode = state_dict['threshold_mode']
        self.cooldown = state_dict['cooldown']
        self.min_lrs = state_dict['min_lrs']
        self.eps = state_dict['eps']
        self.best = state_dict['best']
        self.num_bad_epochs = state_dict['num_bad_epochs']
        self.cooldown_counter = state_dict['cooldown_counter']
        
        # 重新初始化is_better函数
        self._init_is_better(self.mode, self.threshold, self.threshold_mode)

