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
Optimizer Module for the Riemann Library

This module implements various optimization algorithms for training neural networks in the Riemann library.

Optimizers are responsible for adjusting model parameters to minimize the loss function during training.
This file provides implementations of several widely used optimization algorithms with PyTorch-compatible interfaces.

Implemented optimizers:
- Optimizer: Base class providing common optimizer functionality
- GD: Simple Gradient Descent optimizer
- SGD: Stochastic Gradient Descent with momentum support
- Adam: Adaptive Moment Estimation optimizer with bias correction
- Adagrad: Adaptive Gradient Algorithm with per-parameter learning rates
- LBFGS: Limited-memory BFGS algorithm for second-order optimization

All optimizers support parameter groups, weight decay (L2 regularization), and state saving/restoration.
"""
import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union, Callable, Iterable, Generator
from .tensordef import *
from .nn import *

class Optimizer:
    """
    优化器基类，提供优化器的通用接口和功能。
    
    所有具体的优化器算法都应继承此类，并实现step方法以执行参数更新。
    
    参数:
        params: 待优化的参数组，可以是参数列表、生成器或参数字典列表
        defaults: 优化器的默认超参数（如学习率、权重衰减等）
    
    属性:
        param_groups: 存储参数组的列表，每个参数组是一个包含参数和超参数的字典
        defaults: 优化器的默认超参数
        state: 优化器的状态字典，存储每个参数的历史信息（如动量项等）
    """
    def __init__(self, params: Union[Iterable[Union[TN, Parameter]], List[Dict[str, Any]]], defaults: Dict[str, Any]) -> None:
        """
        初始化优化器
        
        参数:
            params: 待优化的参数组，可以是参数列表、生成器或参数字典列表
            defaults: 优化器的默认超参数（如学习率、权重衰减等）
        
        异常:
            ValueError: 当参数列表为空时抛出
        """
        self.param_groups: List[Dict[str, Any]] = []
        self.defaults = defaults
        # 使用defaultdict简化状态管理，避免检查不存在的键
        self.state: defaultdict[Any, Dict[str, Any]] = defaultdict(dict)
        
        # 检查空参数列表
        if isinstance(params, (list, tuple)) and len(params) == 0:
            raise ValueError("optimizer got an empty parameter list")
        
        if isinstance(params, list) and len(params) > 0 and isinstance(params[0], dict):
            # 如params是字典的列表，将每个字典作为一个组加入param_groups
            for param_group in params:
                self.add_param_group(param_group)
        else:
            # 如果params不是list而是来自Module的参数生成器，
            # 将字典{'params': params}作为一个group加入param_groups
            # 将params生成器转换成list列表，以便可以重复访问参数
            para_list = list(params)
            if len(para_list) == 0:
                raise ValueError("optimizer got an empty parameter list")
            self.add_param_group({'params': para_list})

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """
        向优化器添加一个参数组
        
        参数:
            param_group: 包含参数和超参数的字典，必须包含'params'键
        
        异常:
            TypeError: 当param_group不是字典或参数类型不正确时抛出
            ValueError: 当参数不需要梯度时抛出
        """
        # 验证param_group是字典
        if not isinstance(param_group, dict):
            raise TypeError(f"param_group must be a dict, got {type(param_group).__name__}")
        
        # 检查'params'键是否存在
        if 'params' not in param_group:
            raise KeyError("param_group must contain 'params' key")
        
        params = param_group['params']
        
        # 处理单个参数的情况
        if isinstance(params, (TN, Parameter)):
            # 确保参数需要梯度
            if not params.requires_grad:
                raise ValueError("optimizer can only optimize TNs with requires_grad=True")
            param_group['params'] = [params]
        
        # 处理参数列表/元组的情况
        elif isinstance(params, (list, tuple)):
            if len(params) == 0:
                raise ValueError("param_group['params'] cannot be empty")
            for param in params:
                if not isinstance(param, (TN, Parameter)):
                    raise TypeError(f"optimizer can only optimize TN or Parameter objects, but one of the params is {type(param).__name__}")
                if not param.requires_grad:
                    raise ValueError("optimizer can only optimize TNs with requires_grad=True")
        
        # 其他情况报错
        else:
            raise TypeError(f"params argument given to the optimizer should be an iterable of TNs/Parameters or dicts, but got {type(params).__name__}")
        
        # 合并默认参数（PyTorch风格，不检查重复组）
        param_group = {**self.defaults, **param_group}
        self.param_groups.append(param_group)

    def zero_grad(self, set_to_none: bool = False) -> None:
        """
        将所有参数的梯度清零
        
        参数:
            set_to_none: 如果为True，则将梯度设置为None而非零张量
                         这可以在某些情况下节省内存，但可能会改变计算图的行为
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()
    
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        执行单个优化步骤
        
        参数:
            closure: 一个可选的闭包函数，重新评估模型并返回损失
        
        返回:
            如果提供了closure，则返回损失值；否则返回None
        
        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("Subclasses must implement step method")

    def state_dict(self) -> Dict[str, Any]:
        """
        返回优化器状态的字典，可用于保存和恢复优化器状态
        
        返回:
            dict: 包含优化器状态的字典，包括参数组设置和每个参数的状态
        """
        # 保存参数组设置，深拷贝以避免修改原始数据
        param_groups = []
        for group in self.param_groups:
            param_group = {}
            for key, value in group.items():
                if key == 'params':
                    # 保存参数对象的信息而不是对象本身
                    param_group[key] = [(id(p), p.shape, p.dtype) if hasattr(p, 'shape') else id(p) for p in value]
                else:
                    param_group[key] = value
            param_groups.append(param_group)
        
        # 保存每个参数的状态
        state_mapping = {}
        for group in self.param_groups:
            for p in group['params']:
                param_id = id(p)
                if param_id in self.state:
                    state_mapping[param_id] = self.state[param_id]
        
        return {
            'state': state_mapping,
            'param_groups': param_groups
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        从state_dict中加载优化器状态
        
        参数:
            state_dict: 包含优化器状态的字典，通常由state_dict()方法生成
        
        异常:
            TypeError: 当state_dict不是字典时抛出
            KeyError: 当state_dict缺少必要键时抛出
        """
        # 验证state_dict
        if not isinstance(state_dict, dict):
            raise TypeError(f"state_dict must be a dict, got {type(state_dict).__name__}")
        
        if 'state' not in state_dict or 'param_groups' not in state_dict:
            raise KeyError("state_dict must contain 'state' and 'param_groups' keys")
        
        # 保存当前参数组的参数引用，用于后续匹配
        current_params = []
        for group in self.param_groups:
            current_params.extend(group['params'])
        
        # 加载参数组
        self.param_groups = []
        for saved_group in state_dict['param_groups']:
            if 'params' not in saved_group:
                raise KeyError("param_group must contain 'params' key")
            
            # 重建参数引用
            param_refs = saved_group['params']
            actual_params = []
            
            for param_ref in param_refs:
                if isinstance(param_ref, (list, tuple)) and len(param_ref) >= 3:
                    # 新格式：(id, shape, dtype)
                    param_id, shape, dtype = param_ref[0], param_ref[1], param_ref[2]
                    # 在当前参数列表中查找匹配的参数
                    found = False
                    for p in current_params:
                        if id(p) == param_id and hasattr(p, 'shape') and p.shape == shape and str(p.dtype) == str(dtype):
                            actual_params.append(p)
                            found = True
                            break
                    if not found:
                        raise ValueError(f"Cannot find parameter with id {param_id}, shape {shape}, dtype {dtype}")
                else:
                    # 旧格式：直接使用id
                    param_id = param_ref
                    found = False
                    for p in current_params:
                        if id(p) == param_id:
                            actual_params.append(p)
                            found = True
                            break
                    if not found:
                        raise ValueError(f"Cannot find parameter with id {param_id}")
            
            # 重建参数组
            new_group = {k: v for k, v in saved_group.items() if k != 'params'}
            new_group['params'] = actual_params
            self.param_groups.append(new_group)
        
        # 加载状态
        self.state.clear()
        self.state.update(state_dict['state'])

    def __repr__(self) -> str:
        """
        返回优化器的字符串表示，显示优化器的类型和参数组设置
        
        返回:
            str: 优化器的字符串表示
        """
        format_string = self.__class__.__name__ + '('
        for i, group in enumerate(self.param_groups):
            format_string += f'Parameter Group {i}\n'
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += f'    {key}: {group[key]}\n'
                else:
                    format_string += f'    params: [{len(group[key])} parameters]\n'
        format_string += ')'
        return format_string

class GD(Optimizer):
    """
    普通梯度下降（Gradient Descent）优化器
    
    最简单的优化算法，每次迭代都沿着负梯度方向更新参数。
    支持权重衰减（L2正则化）以防止过拟合。
    
    参数更新公式: θ = θ - η * ∇θL(θ)
    其中，θ是参数，η是学习率，∇θL(θ)是损失函数关于参数的梯度
    
    适用场景:
        - 小规模数据集和简单模型
        - 作为其他复杂优化算法的基础对比
    """
    def __init__(self, params, lr=0.01, weight_decay=0.0):
        """
        初始化梯度下降优化器
        
        参数:
            params: 待优化参数组 (需包含requires_grad=True的TN对象)
            lr: 学习率 (默认0.01)
            weight_decay: L2正则化系数 (默认0.0)
        
        异常:
            ValueError: 当学习率或权重衰减系数为负数时抛出
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    def step(self):
        """
        执行单个优化步骤（参数更新）
        
        应用L2正则化（如果启用）并根据负梯度方向更新参数
        """
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # 添加梯度形状验证
                # if p.grad.data.shape != p.data.shape:
                #     raise ValueError(f"Gradient shape {p.grad.data.shape} does not match parameter shape {p.data.shape}")
                
                # 应用L2正则化
                if weight_decay > 0:
                    grad = p.grad.data + weight_decay * p.data
                else:
                    grad = p.grad.data
                    
                # 参数更新：θ = θ - η * ∇θ
                p.data -= lr * grad
                


class SGD(Optimizer):
    """
    随机梯度下降（Stochastic Gradient Descent）优化器
    
    梯度下降的随机版本，每次迭代使用一个批次的数据计算梯度。
    支持动量（momentum）以加速收敛和减少震荡，支持Nesterov动量。
    
    参数更新公式: 
        - 标准动量: v = μ*v + ∇θL(θ), θ = θ - η*v
        - Nesterov动量: v = μ*v + ∇θL(θ) + μ*η*∇θL(θ), θ = θ - η*v
    
    适用场景:
        - 大规模数据集训练
        - 需要逃离局部最优解的复杂优化问题
        - 大多数深度学习任务的默认选择
    """
    def __init__(self, params: Union[Iterable[Union[TN, Parameter]], List[Dict[str, Any]]], 
                 lr: float = 0.01, momentum: float = 0.0, weight_decay: float = 0.0,
                 dampening: float = 0.0, nesterov: bool = False) -> None:
        """
        初始化随机梯度下降优化器
        
        参数:
            params: 待优化参数组 (需包含requires_grad=True的TN对象)
            lr: 学习率 (默认0.01)
            momentum: 动量系数 (默认0.0)
            weight_decay: L2正则化系数 (默认0.0)
            dampening: 动量抑制系数 (默认0.0)
            nesterov: 是否启用Nesterov动量 (默认False)
        
        异常:
            ValueError: 当学习率、动量系数、权重衰减系数或抑制系数为负数时抛出
            ValueError: 当启用nesterov但momentum为0时抛出
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if dampening < 0.0:
            raise ValueError(f"Invalid dampening value: {dampening}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                       dampening=dampening, nesterov=nesterov)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        执行单个优化步骤（参数更新）
        
        应用权重衰减（如果启用），更新动量项，然后根据动量方向更新参数。
        支持dampening和Nesterov动量。
        
        参数:
            closure: 一个可选的闭包函数，重新评估模型并返回损失
        
        返回:
            如果提供了closure，则返回损失值；否则返回None
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            # 一次性获取所有参数，避免重复的get调用
            lr = group['lr']
            momentum = group.get('momentum', 0)
            weight_decay = group.get('weight_decay', 0)
            dampening = group.get('dampening', 0)
            nesterov = group.get('nesterov', False)
            
            # 预计算常用值，避免重复计算
            has_momentum = momentum > 0
            has_weight_decay = weight_decay > 0
            has_dampening = dampening > 0
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # 添加梯度形状验证
                # if p.grad.data.shape != p.data.shape:
                #     raise ValueError(f"Gradient shape {p.grad.data.shape} does not match parameter shape {p.data.shape}")
                
                grad = p.grad.data
                param_data = p.data
                
                # 原地应用权重衰减，避免创建新数组
                if has_weight_decay:
                    grad += weight_decay * param_data
                
                if has_momentum:
                    param_id = id(p)
                    
                    # 获取或初始化动量状态
                    if param_id not in self.state:
                        self.state[param_id] = {'velocity': np.zeros_like(param_data)}
                    
                    velocity = self.state[param_id]['velocity']
                    
                    if nesterov:
                        # 保存更新前的速度用于Nesterov更新
                        old_velocity = velocity.copy()
                        
                        # 更新速度
                        if has_dampening:
                            velocity *= momentum
                            velocity += (1.0 - dampening) * grad
                        else:
                            velocity *= momentum
                            velocity += grad
                        
                        # Nesterov更新: param = param - lr * (grad + momentum * old_velocity)
                        update = lr * (grad + momentum * old_velocity)
                        param_data -= update
                    else:
                        # 标准动量更新
                        if has_dampening:
                            velocity *= momentum
                            velocity += (1.0 - dampening) * grad
                        else:
                            velocity *= momentum
                            velocity += grad
                        
                        # 原地参数更新
                        param_data -= lr * velocity
                else:
                    # 无动量，直接使用梯度
                    param_data -= lr * grad
        
        return loss

class Adam(Optimizer):
    """
    Adam（Adaptive Moment Estimation）优化器
    
    结合了动量法和自适应学习率的优化算法，维护每个参数的学习率和动量项。
    支持偏差校正和AMSGrad变体。
    
    参数更新公式:
        m = β1*m + (1-β1)*∇θL(θ)  # 一阶矩估计
        v = β2*v + (1-β2)*(∇θL(θ))²  # 二阶矩估计
        m̂ = m/(1-β1^t)  # 偏差校正的一阶矩估计
        v̂ = v/(1-β2^t)  # 偏差校正的二阶矩估计
        θ = θ - η*m̂/(√v̂ + ε)  # 参数更新
    
    适用场景:
        - 需要快速收敛的深度学习任务
        - 非平稳目标函数
        - 大规模数据和参数的场景
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        """
        初始化Adam优化器
        
        参数:
            params: 待优化参数组 (需包含requires_grad=True的TN对象)
            lr: 学习率 (默认1e-3)
            betas: 用于计算一阶和二阶矩估计的系数，格式为(beta1, beta2) (默认(0.9, 0.999))
            eps: 数值稳定性参数，防止除零错误 (默认1e-8)
            weight_decay: 权重衰减系数 (默认0)
            amsgrad: 是否使用AMSGrad变体 (默认False)
        
        异常:
            ValueError: 当学习率、betas、eps或权重衰减系数为负数时抛出
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)
        # 移除state初始化，采用延迟初始化策略

    def step(self):
        """
        执行单个优化步骤（参数更新）
        
        应用权重衰减（如果启用），更新一阶和二阶矩估计，执行偏差校正，
        最后根据Adam更新规则调整参数
        """
        # 延迟初始化state
        if not hasattr(self, 'state'):
            self.state = {}
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            weight_decay = group.get('weight_decay', 0)
            amsgrad = group.get('amsgrad', False)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # 添加梯度形状验证
                # if p.grad.data.shape != p.data.shape:
                #     raise ValueError(f"Gradient shape {p.grad.data.shape} does not match parameter shape {p.data.shape}")
                
                grad = p.grad.data
                
                # 添加权重衰减
                if weight_decay > 0:
                    grad = grad + weight_decay * p.data
                
                param_id = id(p)
                
                # 初始化状态
                if param_id not in self.state:
                    self.state[param_id] = {
                        'step': 0,
                        'exp_avg': np.zeros_like(p.data),
                        'exp_avg_sq': np.zeros_like(p.data)
                    }
                    if amsgrad:
                        self.state[param_id]['max_exp_avg_sq'] = np.zeros_like(p.data)
                
                state = self.state[param_id]
                state['step'] += 1  # 先递增步数
                
                # 优化更新计算，使用原地操作减少内存分配
                np.multiply(beta1, state['exp_avg'], out=state['exp_avg'])
                np.add(state['exp_avg'], (1.0 - beta1) * grad, out=state['exp_avg'])
                
                np.multiply(beta2, state['exp_avg_sq'], out=state['exp_avg_sq'])
                np.add(state['exp_avg_sq'], (1.0 - beta2) * np.square(grad), out=state['exp_avg_sq'])
                
                # 偏差校正
                bias_correction1 = 1.0 - beta1 ** state['step']
                bias_correction2 = 1.0 - beta2 ** state['step']
                
                # AMSGrad处理
                if amsgrad:
                    np.maximum(state['max_exp_avg_sq'], state['exp_avg_sq'], out=state['max_exp_avg_sq'])
                    denom = np.sqrt(state['max_exp_avg_sq']) + eps
                else:
                    denom = np.sqrt(state['exp_avg_sq']) + eps
                
                # 计算更新量
                step_size = lr * np.sqrt(bias_correction2) / bias_correction1
                p.data -= step_size * (state['exp_avg'] / denom)

class Adagrad(Optimizer):
    """
    Adagrad（Adaptive Gradient Algorithm）优化器
    
    自适应学习率优化算法，为每个参数维护独立的学习率。
    根据历史梯度的平方和来自适应调整学习率，梯度大的参数学习率小，梯度小的参数学习率大。
    
    参数更新公式: θ = θ - (η / (√(G + ε))) * ∇θL(θ)
    其中，G是历史梯度的平方和，η是学习率，ε是数值稳定参数
    
    适用场景:
        - 处理稀疏数据的任务
        - 训练词嵌入模型和某些类型的CNN
        - 需要不同参数有不同学习率的场景
    """
    def __init__(self, params, lr=0.01, lr_decay=0.0, weight_decay=0.0, initial_accumulator_value=0.0, eps=1e-10):
        """
        初始化Adagrad优化器
        
        参数:
            params: 待优化参数组 (需包含requires_grad=True的TN对象)
            lr: 学习率 (默认: 0.01)
            lr_decay: 学习率衰减率 (默认: 0.0)
            weight_decay: 权重衰减 (L2正则化系数) (默认: 0.0)
            initial_accumulator_value: 累加器的初始值 (默认: 0.0)
            eps: 数值稳定性参数，防止除零错误 (默认: 1e-10)
        
        异常:
            ValueError: 当任何参数值无效时抛出
        """
        if initial_accumulator_value < 0.0:
            raise ValueError(f"Invalid initial_accumulator_value: {initial_accumulator_value}")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if lr_decay < 0.0:
            raise ValueError(f"Invalid lr_decay value: {lr_decay}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
            
        defaults = dict(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay, 
                       initial_accumulator_value=initial_accumulator_value, eps=eps)
        super().__init__(params, defaults)
        
    def step(self):
        """
        执行单个优化步骤（参数更新）
        
        算法步骤：
        1. 应用权重衰减（如果启用）
        2. 更新梯度平方累加器
        3. 计算衰减的学习率
        4. 根据Adagrad更新规则调整参数
        
        数学公式：
        - 梯度平方累加: sum_sq += grad^2
        - 学习率衰减: lr_t = lr / (1 + (step-1) * lr_decay)
        - 参数更新: θ = θ - (lr_t / (sqrt(sum_sq) + eps)) * grad
        """
        # 延迟初始化state
        if not hasattr(self, 'state'):
            self.state = {}
            
        for group in self.param_groups:
            lr = group['lr']
            lr_decay = group['lr_decay']
            weight_decay = group.get('weight_decay', 0)
            initial_accumulator_value = group['initial_accumulator_value']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # 验证梯度形状
                # if p.grad.data.shape != p.data.shape:
                #     raise ValueError(f"Gradient shape {p.grad.data.shape} does not match parameter shape {p.data.shape}")
                
                # 验证数据类型
                # if p.grad.data.dtype != p.data.dtype:
                #     raise TypeError(f"Gradient data type {p.grad.data.dtype} does not match parameter data type {p.data.dtype}")
                
                # 保存原始梯度值
                grad_original = p.grad.data.copy()
                grad = grad_original
                param_id = id(p)
                
                # 初始化状态
                if param_id not in self.state:
                    self.state[param_id] = {
                        'step': 0,
                        'sum': np.full_like(p.data, initial_accumulator_value)
                    }
                
                state = self.state[param_id]
                state['step'] += 1
                
                # 应用权重衰减
                if weight_decay > 0:
                    # 使用原地操作减少内存分配
                    np.add(grad, weight_decay * p.data, out=grad)
                
                # 计算梯度平方
                grad_squared = np.square(grad)
                
                # 更新梯度平方累加器（原地操作）
                sum_sq = state['sum']
                np.add(sum_sq, grad_squared, out=sum_sq)
                
                # 计算学习率衰减
                lr_t = lr / (1.0 + (state['step'] - 1.0) * lr_decay)
                
                # 计算更新分母（增强数值稳定性）
                denom = np.sqrt(np.maximum(sum_sq, 1e-16)) + eps
                
                # 计算更新量并应用（使用原地操作减少内存分配）
                # θ = θ - (lr_t / denom) * grad_original
                update = lr_t * grad_original / denom
                np.subtract(p.data, update, out=p.data)

class LBFGS(Optimizer):
    """
    L-BFGS（Limited-memory Broyden–Fletcher–Goldfarb–Shanno）优化器
    
    一种拟牛顿优化算法，通过维护有限数量的历史梯度和参数变化来近似Hessian矩阵的逆。
    相比于标准BFGS，它使用更少的内存，适用于大规模优化问题。
    
    注意：与其他优化器不同，LBFGS需要一个闭包函数来重新计算损失和梯度。
    
    参数:
        params: 待优化的参数组
        lr: 学习率（默认: 1.0）
        max_iter: 每个优化步骤中的最大迭代次数（默认: 20）
        max_eval: 每个优化步骤中的最大函数评估次数（默认: None，即max_iter * 1.25）
        tolerance_grad: 梯度收敛阈值（默认: 1e-5）
        tolerance_change: 参数变化收敛阈值（默认: 1e-9）
        history_size: 历史记录大小（默认: 100）
        line_search_fn: 线搜索函数（默认: None，使用内置的强Wolfe条件线搜索）
    """
    def __init__(self, params, lr=1.0, max_iter=20, max_eval=None, tolerance_grad=1e-5, 
                 tolerance_change=1e-9, history_size=100, line_search_fn=None):
        """
        初始化LBFGS优化器
        
        参数:
            params: 待优化参数组
            lr: 学习率
            max_iter: 每个优化步骤中的最大迭代次数
            max_eval: 每个优化步骤中的最大函数评估次数
            tolerance_grad: 梯度收敛阈值
            tolerance_change: 参数变化收敛阈值
            history_size: 历史记录大小
            line_search_fn: 线搜索函数
        """
        if max_eval is None:
            max_eval = max_iter * 5 // 4
            
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if max_iter <= 0:
            raise ValueError(f"Invalid max_iter value: {max_iter}")
        if max_eval <= 0:
            raise ValueError(f"Invalid max_eval value: {max_eval}")
        if tolerance_grad < 0:
            raise ValueError(f"Invalid tolerance_grad value: {tolerance_grad}")
        if tolerance_change < 0:
            raise ValueError(f"Invalid tolerance_change value: {tolerance_change}")
        if history_size <= 0:
            raise ValueError(f"Invalid history_size value: {history_size}")
            
        defaults = dict(lr=lr, max_iter=max_iter, max_eval=max_eval,
                        tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
                        history_size=history_size, line_search_fn=line_search_fn)
        super(LBFGS, self).__init__(params, defaults)
        
        # 为每个参数组初始化状态
        for group in self.param_groups:
            group.setdefault('func_evals', 0)
            group.setdefault('old_dirs', [])
            group.setdefault('old_stps', [])
            group.setdefault('H_diag', 1.0)
            # 预计算并缓存参数信息，避免每次step重复计算
            params = group['params']
            param_info, total_params = self._compute_param_info(params)
            group['param_info'] = param_info
            group['total_params'] = total_params
            # 预创建向量转换函数
            group['vector_funcs'] = self._create_vector_functions(param_info, total_params)
    
    def _create_vector_functions(self, param_info, total_params):
        """
        创建参数和梯度向量转换的辅助函数
        
        Args:
            param_info: 参数信息列表
            total_params: 参数总数
            
        Returns:
            tuple: (params_to_vector, vector_to_params, grads_to_vector) 三个辅助函数
        """
        # 将参数列表转换为单一向量
        def params_to_vector(param_list):
            """将参数列表转换为单一向量"""
            vec = np.zeros(total_params)
            for info in param_info:
                p = info['param']
                offset = info['offset']
                numel = info['numel']
                if p.data.ndim == 0:  # 标量参数
                    vec[offset] = p.item()
                else:
                    vec[offset:offset + numel] = p.data.flatten()
            return vec
        
        # 将向量值应用到参数列表
        def vector_to_params(vec, param_list):
            """将向量值应用到参数列表"""
            for info in param_info:
                p = info['param']
                offset = info['offset']
                numel = info['numel']
                shape = info['shape']
                if len(shape) == 0:  # 标量参数
                    p.data = vec[offset]
                else:
                    p.data = vec[offset:offset + numel].reshape(shape)
        
        # 将梯度转换为单一向量
        def grads_to_vector():
            """将梯度转换为单一向量"""
            vec = np.zeros(total_params)
            for info in param_info:
                p = info['param']
                offset = info['offset']
                numel = info['numel']
                if p.grad is None:
                    continue
                if p.grad.data.ndim == 0:  # 标量梯度
                    vec[offset] = p.grad.item()
                else:
                    vec[offset:offset + numel] = p.grad.data.flatten()
            return vec
            
        return params_to_vector, vector_to_params, grads_to_vector
    
    def _compute_lbfgs_direction(self, grad_vector, old_dirs, old_stps, n_old, H_diag, np_dot):
        """
        计算L-BFGS搜索方向（双循环算法）
        
        Args:
            grad_vector: 当前梯度向量
            old_dirs: 历史参数变化列表
            old_stps: 历史梯度变化列表
            n_old: 历史记录数量
            H_diag: Hessian对角线估计
            np_dot: numpy点积函数引用
            
        Returns:
            tuple: (search_dir, H_diag) 搜索方向和更新后的Hessian对角线估计
        """
        # 预计算ys值和点积，避免重复计算
        ys_values = np.array([np_dot(old_stps[i], old_dirs[i]) for i in range(n_old)])
        
        # 过滤掉不稳定的项
        valid_indices = np.where(ys_values > 1e-12)[0]
        
        # 初始化搜索方向
        search_dir = np.zeros_like(grad_vector)
        
        if len(valid_indices) == 0:
            # 没有有效历史，使用最陡下降
            search_dir = -grad_vector.copy()
            H_diag = 1.0
        else:
            # 反向循环 - 使用向量化操作和原地操作
            q = grad_vector.copy()
            alphas = np.zeros(len(valid_indices))
            
            # 向量化计算alphas（不能预计算点积，因为q在循环中会改变）
            for idx, i in enumerate(reversed(valid_indices)):
                s_k = old_dirs[i]
                ys = ys_values[i]
                alpha_k = np_dot(s_k, q) / ys
                alphas[idx] = alpha_k
                # 使用原地操作减少内存分配
                np.subtract(q, alpha_k * old_stps[i], out=q)
            
            # 初始Hessian近似
            np.multiply(q, H_diag, out=q)  # 原地操作，q = H_diag * q
            
            # 正向循环 - 使用向量化操作和原地操作
            for idx, i in enumerate(valid_indices):
                s_k = old_dirs[i]
                y_k = old_stps[i]
                ys = ys_values[i]
                beta_k = np_dot(y_k, q) / ys
                # 使用原地操作减少内存分配
                np.add(q, s_k * (alphas[len(valid_indices) - 1 - idx] - beta_k), out=q)
            
            # 搜索方向为负梯度方向
            np.negative(q, out=search_dir)  # 原地操作，search_dir = -q
        
        return search_dir, H_diag
    
    def _line_search(self, closure, params, current_params_vec, search_dir, grad_vector, 
                     loss_val, gtd, lr, vector_to_params, grads_to_vector, 
                     np_dot, np_isfinite, c1=1e-4, c2=0.9, max_backtracks=10):
        """
        执行回溯线搜索，寻找满足Wolfe条件的步长
        
        Args:
            closure: 重新计算损失的闭包函数
            params: 参数列表
            current_params_vec: 当前参数向量
            search_dir: 搜索方向
            grad_vector: 当前梯度向量
            loss_val: 当前损失值
            gtd: 梯度与搜索方向的点积
            lr: 学习率（初始步长）
            vector_to_params: 向量到参数的转换函数
            grads_to_vector: 梯度到向量的转换函数
            np_dot: numpy点积函数引用
            np_isfinite: numpy有限值检查函数引用
            c1: Armijo条件常数
            c2: 曲率条件常数
            max_backtracks: 最大回溯次数
            
        Returns:
            tuple: (line_search_success, s, y, func_evals) 线搜索结果
        """
        old_loss = loss_val
        eta = lr  # 使用学习率作为初始步长，而不是固定的1.0
        func_evals = 0
        line_search_success = False
        s = None
        y = None
        
        # 执行线搜索
        for backtracks in range(max_backtracks):
            # 计算新的参数位置：current_params + eta * search_dir
            new_params_vec = current_params_vec + eta * search_dir
            
            # 应用新参数
            vector_to_params(new_params_vec, params)
            
            # 计算新的损失
            loss = closure()
            func_evals += 1
            
            new_loss = loss.item()
            
            # Armijo条件检查（充分下降条件）
            if new_loss <= old_loss + c1 * eta * gtd:
                # 只有在满足Armijo条件时才计算梯度，减少不必要的计算
                new_grad_vector = grads_to_vector()
                
                # 曲率条件检查
                new_gtd = np_dot(new_grad_vector, search_dir)
                if new_gtd >= c2 * gtd:
                    # 满足强Wolfe条件，线搜索成功
                    line_search_success = True
                    
                    # 计算参数变化和梯度变化
                    s = new_params_vec - current_params_vec
                    y = new_grad_vector - grad_vector
                    
                    break
            
            # 减少步长
            eta *= 0.5
            
        return line_search_success, s, y, func_evals
    
    def _update_history(self, old_dirs, old_stps, s, y, n_old, history_size):
        """
        更新L-BFGS历史记录
        
        Args:
            old_dirs: 历史参数变化列表
            old_stps: 历史梯度变化列表
            s: 当前参数变化
            y: 当前梯度变化
            n_old: 当前历史记录数量
            history_size: 最大历史记录大小
            
        Returns:
            int: 更新后的历史记录数量
        """
        # 计算内积，用于稳定性检查
        ys = np.dot(y, s)
        yy = np.dot(y, y)
        
        # 只有在满足稳定性条件时才更新历史记录
        if ys > 1e-10 and yy > 1e-10 and np.isfinite(ys) and np.isfinite(yy):
            # 更新历史记录
            if n_old == history_size:
                # 移除最旧的历史记录
                old_dirs.pop(0)
                old_stps.pop(0)
            else:
                n_old += 1
            old_dirs.append(s)
            old_stps.append(y)
        
        return n_old
    
    def _check_convergence(self, grad_norm, tolerance_grad, old_dirs, tolerance_change):
        """
        检查收敛条件
        
        Args:
            grad_norm: 梯度范数
            tolerance_grad: 梯度收敛阈值
            old_dirs: 历史参数变化列表
            tolerance_change: 参数变化收敛阈值
            
        Returns:
            bool: 是否收敛
        """
        # 检查梯度收敛条件
        if grad_norm < tolerance_grad:
            return True
            
        # 检查参数变化收敛条件（如果有历史记录）
        if len(old_dirs) > 0:
            # 优化：使用向量化操作计算所有范数，避免循环
            param_norms = np.array([float(np.linalg.norm(s)) for s in old_dirs])
            max_param_change = np.max(param_norms)  # 使用numpy的max函数，更高效
            if max_param_change < tolerance_change:
                return True
                
        return False
    
    def _initialize_group_state(self, group):
        """
        初始化参数组状态（如果不存在）
        
        Args:
            group: 参数组字典
        """
        if 'old_dirs' not in group:
            group['old_dirs'] = []
            group['old_stps'] = []
            group['H_diag'] = 1.0  # 初始Hessian对角线估计
        # 重要修改：每次调用step时重置函数评估计数
        # 这样可以确保每次调用都能进行完整的优化
        group['func_evals'] = 0
    
    def _compute_param_info(self, params):
        """
        计算参数信息，避免重复计算
        
        Args:
            params: 参数列表
            
        Returns:
            tuple: (param_info, total_params) 参数信息列表和总参数数
        """
        param_info = []
        total_params = 0
        for p in params:
            if p.grad is not None:
                numel = p.numel()
                shape = p.shape
                param_info.append({
                    'param': p,
                    'numel': numel,
                    'shape': shape,
                    'offset': total_params
                })
                total_params += numel
        return param_info, total_params
    
    def _restore_best_params(self, params, best_params_vec, vector_to_params):
        """
        恢复最佳参数
        
        Args:
            params: 当前参数列表
            best_params_vec: 最佳参数向量
            vector_to_params: 向量到参数的转换函数
        """
        # 使用向量形式恢复参数，减少克隆开销
        if best_params_vec is not None:
            vector_to_params(best_params_vec, params)
    
    def _compute_final_loss(self, closure, best_loss):
        """
        计算最终损失值
        
        Args:
            closure: 重新计算损失的闭包函数
            best_loss: 最佳损失值
            
        Returns:
            Tensor: 最终损失值
        """
        np_isfinite = np.isfinite
        # 优化：只有在需要时才重新计算损失
        # 如果最佳损失是有限的，则直接返回，避免重复计算
        if np_isfinite(best_loss):
            # closure()总是返回TN张量，直接返回
            return tensor(best_loss)
        else:
            # 只有在最佳损失无效时才重新计算
            loss = closure()
            return loss
    
    # LBFGS优化器的step方法
    def step(self, closure=None):
        """
        执行单步优化
        
        Args:
            closure: 重新计算模型并返回损失的闭包函数
        
        Returns:
            Tensor: 损失值
        """
        if closure is None:
            raise RuntimeError('LBFGS optimizer requires a closure function')
        
        # 预先获取内置函数引用，提高性能
        np_dot = np.dot
        np_isfinite = np.isfinite
        np_linalg_norm = np.linalg.norm
        
        # 回溯线搜索常量
        c1 = 1e-4  # Armijo条件常数
        c2 = 0.9   # 曲率条件常数
        max_backtracks = 10  # 最大回溯次数
        
        # 对每个参数组进行优化
        loss = None
        for group in self.param_groups:
            # 获取参数组配置
            params = group['params']
            lr = group.get('lr', 1.0)
            tolerance_grad = group.get('tolerance_grad', 1e-5)
            tolerance_change = group.get('tolerance_change', 1e-9)
            history_size = group.get('history_size', 10)
            max_iter = group.get('max_iter', 20)
            max_eval = group.get('max_eval', max_iter * 5 // 4)
            
            # 初始化组状态（如果不存在）
            if 'old_dirs' not in group:
                self._initialize_group_state(group)
            
            # 获取组状态
            old_dirs = group['old_dirs']
            old_stps = group['old_stps']
            H_diag = group['H_diag']
            n_old = len(old_dirs)
            func_evals = 0  # 本地计数器，独立于group状态
            
            # 使用缓存的向量转换函数
            params_to_vector, vector_to_params, grads_to_vector = group['vector_funcs']
            
            # 如果参数信息为空（初始化时没有梯度），重新计算
            if not group['param_info']:
                # 计算一次梯度以获取参数信息
                loss = closure()
                loss.backward()
                param_info, total_params = self._compute_param_info(params)
                group['param_info'] = param_info
                group['total_params'] = total_params
                # 重新创建向量转换函数
                group['vector_funcs'] = self._create_vector_functions(param_info, total_params)
                params_to_vector, vector_to_params, grads_to_vector = group['vector_funcs']
            
            # 保存当前最佳参数和损失
            best_loss = float('inf')
            best_params_vec = None  # 使用向量形式保存最佳参数，减少克隆开销
            
            # 执行多次迭代，直到达到最大迭代次数或最大函数评估次数
            for iteration in range(max_iter):
                # 1. 保存当前参数向量（在closure调用之前，确保参数向量和梯度匹配）
                current_params_vec = params_to_vector(params)
                
                # 2. 计算当前损失和梯度
                loss = closure()
                func_evals += 1
                loss_val = loss.item()
                
                # 3. 获取梯度向量并检查收敛条件
                grad_vector = grads_to_vector()
                grad_norm = np_linalg_norm(grad_vector)
                
                # 保存最佳状态（使用已获取的参数向量，避免重复转换）
                if loss_val < best_loss and np_isfinite(loss_val):
                    best_loss = loss_val
                    best_params_vec = current_params_vec.copy()  # 使用已获取的参数向量

                # 检查是否达到最大函数评估次数
                if func_evals >= max_eval:
                    break
                
                # 检查收敛条件
                if self._check_convergence(grad_norm, tolerance_grad, old_dirs, tolerance_change):
                    break
                
                # 4. 计算L-BFGS搜索方向（双循环算法）
                search_dir, H_diag = self._compute_lbfgs_direction(
                    grad_vector, old_dirs, old_stps, n_old, H_diag, np_dot)
                
                # 5. 检查是否为下降方向
                gtd = np_dot(grad_vector, search_dir)
                if gtd >= 0:
                    # 非下降方向，使用最陡下降
                    search_dir = -grad_vector.copy()
                    H_diag = 1.0  # 重置Hessian对角线估计
                    gtd = -np_dot(grad_vector, grad_vector)
                
                # 6. 执行回溯线搜索
                line_search_success, s, y, line_search_func_evals = self._line_search(
                    closure, params, current_params_vec, search_dir, grad_vector, 
                    loss_val, gtd, lr, vector_to_params, grads_to_vector, 
                    np_dot, np_isfinite, c1, c2, max_backtracks)
                
                # 更新函数评估计数
                func_evals += line_search_func_evals
                
                # 检查是否达到最大函数评估次数
                if func_evals >= max_eval:
                    break
                
                # 7. 处理线搜索结果
                if line_search_success:
                    # 线搜索成功，更新历史记录
                    # 计算Hessian对角线估计
                    ys = np_dot(y, s)
                    yy = np_dot(y, y)
                    if ys > 1e-10 and yy > 1e-10 and np_isfinite(ys) and np_isfinite(yy):
                        H_diag = ys / yy
                    
                    n_old = self._update_history(old_dirs, old_stps, s, y, n_old, history_size)
                else:
                    # 线搜索失败，恢复参数并使用小步长
                    vector_to_params(current_params_vec, params)
                    
                    # 使用最陡下降作为安全策略
                    safe_eta = 1e-3  # 使用固定的小步长
                    new_params_vec = current_params_vec - safe_eta * grad_vector
                    vector_to_params(new_params_vec, params)
                    
                    # 重新计算损失
                    loss = closure()
                    func_evals += 1
                    
                    # 检查是否达到最大函数评估次数
                    if func_evals >= max_eval:
                        break
                    
                    # 保存最佳状态
                    new_loss = loss.item()
                    if new_loss < best_loss and np_isfinite(new_loss):
                        best_loss = new_loss
                        best_params_vec = current_params_vec - safe_eta * grad_vector  # 使用已知的新参数向量，避免重复转换
            
            # 8. 更新组状态
            group.update({
                'H_diag': H_diag,
                'old_dirs': old_dirs,
                'old_stps': old_stps
            })
            
            # 9. 恢复最佳参数并返回最终损失
            self._restore_best_params(params, best_params_vec, vector_to_params)
            loss = self._compute_final_loss(closure, best_loss)
                    
        return loss


