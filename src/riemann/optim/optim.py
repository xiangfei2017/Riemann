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

Key features:
- PyTorch-compatible optimizer interfaces
- Support for parameter groups with different hyperparameters
- Weight decay (L2 regularization) support
- State saving and restoration capabilities
- Device-aware implementation (supports both CPU and CUDA)
- Efficient state management using defaultdict

Implemented optimizers:
- Optimizer: Base class providing common optimizer functionality and interfaces
- GD: Simple Gradient Descent optimizer with basic parameter updates
- SGD: Stochastic Gradient Descent with momentum and Nesterov acceleration support
- Adam: Adaptive Moment Estimation with bias correction and adaptive learning rates
- Adamax: Adam variant using an exponentially weighted infinity norm for the second moment
- AdamW: Adam with weight decay decoupling for better regularization
- RMSprop: Root Mean Square Propagation with adaptive learning rates
- Adagrad: Adaptive Gradient Algorithm with per-parameter learning rate scaling
- Adadelta: Adagrad variant with an exponentially decaying window and no global learning rate scale
- LBFGS: Limited-memory BFGS algorithm for second-order optimization with line search

Each optimizer implements the standard step() method for parameter updates and supports both
standard optimization steps and closure-based steps (required for LBFGS).
"""

from __future__ import annotations

import numpy as np
import builtins
from collections import defaultdict
from typing import Any, Callable, Iterable, NamedTuple
from ..core import *
from ..nn import *
from ..cuda import cp

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
    def __init__(self, params: Iterable[TN | Parameter] | list[dict[str, Any]], defaults: dict[str, Any]) -> None:
        """
        初始化优化器
        
        参数:
            params: 待优化的参数组，可以是参数列表、生成器或参数字典列表
            defaults: 优化器的默认超参数（如学习率、权重衰减等）
        
        异常:
            ValueError: 当参数列表为空时抛出
        """
        self.param_groups: list[dict[str, Any]] = []
        self.defaults = defaults
        # 使用defaultdict简化状态管理，避免检查不存在的键
        self.state: defaultdict[Any, dict[str, Any]] = defaultdict(dict)
        
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
        
        # 检测参数所在的设备类型，设置arrlib属性
        self.arrlib = np
        if len(self.param_groups) > 0 and len(self.param_groups[0]['params']) > 0:
            first_param = self.param_groups[0]['params'][0]
            if hasattr(first_param, 'device') and first_param.device.type == 'cuda':
                self.arrlib = cp

    def add_param_group(self, param_group: dict[str, Any]) -> None:
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
    
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
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

    def state_dict(self) -> dict[str, Any]:
        """
        返回优化器状态的字典，可用于保存和恢复优化器状态
        
        返回:
            dict: 包含优化器状态的字典，包括参数组设置和每个参数的状态
        """
        # 与 torch 一致：params 保存为按顺序编号的整数索引（跨参数组连续），
        # state 以同一索引为键；加载时按位置匹配参数，不依赖 id(p)，
        # 因此可以在新建模型或新进程中恢复
        param_groups = []
        state_mapping = {}
        offset = 0
        for group in self.param_groups:
            param_group = {}
            for key, value in group.items():
                if key == 'params':
                    indices = []
                    for p in value:
                        indices.append(offset)
                        if id(p) in self.state:
                            state_mapping[offset] = self.state[id(p)]
                        offset += 1
                    param_group[key] = indices
                else:
                    param_group[key] = value
            param_groups.append(param_group)

        return {
            'state': state_mapping,
            'param_groups': param_groups
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        从state_dict中加载优化器状态

        参数按位置匹配（与 torch 一致）：参数组按顺序一一对应、组内参数按
        顺序对应，不要求是同一批参数对象，因此可用于新建模型或新进程。

        参数:
            state_dict: 包含优化器状态的字典，通常由state_dict()方法生成

        异常:
            TypeError: 当state_dict不是字典时抛出
            KeyError: 当state_dict缺少必要键时抛出
            ValueError: 当参数组数量或组内参数数量不匹配时抛出
        """
        # 验证state_dict
        if not isinstance(state_dict, dict):
            raise TypeError(f"state_dict must be a dict, got {type(state_dict).__name__}")

        if 'state' not in state_dict or 'param_groups' not in state_dict:
            raise KeyError("state_dict must contain 'state' and 'param_groups' keys")

        saved_groups = state_dict['param_groups']
        if len(saved_groups) != len(self.param_groups):
            raise ValueError(
                f"loaded state dict has {len(saved_groups)} parameter groups, "
                f"but the optimizer has {len(self.param_groups)}")

        # 按位置匹配参数：保存的索引只用于查 state，参数对象取自当前优化器
        saved_indices = []
        current_params = []
        for saved_group, group in zip(saved_groups, self.param_groups):
            if 'params' not in saved_group:
                raise KeyError("param_group must contain 'params' key")
            refs = saved_group['params']
            params = group['params']
            if len(refs) != len(params):
                raise ValueError(
                    f"loaded state dict group has {len(refs)} params, "
                    f"but the optimizer group has {len(params)}")
            saved_indices.extend(refs)
            current_params.extend(params)

        for index in saved_indices:
            if not isinstance(index, int):
                raise ValueError(
                    "unsupported optimizer state_dict format: expected integer "
                    "parameter indices; re-save the state dict with this version")

        index_to_param = dict(zip(saved_indices, current_params))

        # 重建参数组：超参数取保存值，参数对象用当前优化器持有的
        self.param_groups = [
            {**{k: v for k, v in saved_group.items() if k != 'params'},
             'params': group['params']}
            for saved_group, group in zip(saved_groups, self.param_groups)
        ]

        # 状态在优化器内部以 id(param) 为键，这里把保存的索引映射到当前参数
        self.state.clear()
        for index, value in state_dict['state'].items():
            if index not in index_to_param:
                raise KeyError(
                    f"loaded state dict contains state for unknown parameter index {index}")
            self.state[id(index_to_param[index])] = value

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
                if key == 'params':
                    format_string += f'    params: [{len(group[key])} parameters]\n'
                    continue
                value = group[key]
                # 历史向量等数组列表只显示概要，避免打印巨量内容
                if isinstance(value, (list, tuple)) and value and hasattr(value[0], 'shape'):
                    format_string += (
                        f'    {key}: <{len(value)} arrays of shape {tuple(value[0].shape)}>\n')
                else:
                    format_string += f'    {key}: {value}\n'
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
                # if p.grad._array.shape != p._array.shape:
                #     raise ValueError(f"Gradient shape {p.grad._array.shape} does not match parameter shape {p._array.shape}")
                
                # 应用L2正则化
                if weight_decay > 0:
                    grad = p.grad._array + weight_decay * p._array
                else:
                    grad = p.grad._array
                    
                # 参数更新：θ = θ - η * ∇θ
                p._array -= lr * grad
                


class SGD(Optimizer):
    """
    随机梯度下降（Stochastic Gradient Descent）优化器
    
    梯度下降的随机版本，每次迭代使用一个批次的数据计算梯度。
    支持动量（momentum）以加速收敛和减少震荡，支持Nesterov动量。
    
    参数更新公式: 
        - 标准动量: v = μ*v + ∇θL(θ), θ = θ - η*v
        - Nesterov动量: v = μ*v + ∇θL(θ), θ = θ - η*(∇θL(θ) + μ*v)
    
    适用场景:
        - 大规模数据集训练
        - 需要逃离局部最优解的复杂优化问题
        - 大多数深度学习任务的默认选择
    """
    def __init__(self, params: Iterable[TN | Parameter] | list[dict[str, Any]],
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

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
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
            lr = group['lr']
            momentum = group.get('momentum', 0)
            weight_decay = group.get('weight_decay', 0)
            dampening = group.get('dampening', 0)
            nesterov = group.get('nesterov', False)

            has_momentum = momentum > 0
            has_weight_decay = weight_decay > 0
            has_dampening = dampening > 0

            for p in group['params']:
                grad_tn = p.grad
                if grad_tn is None:
                    continue

                param_data = p._array
                grad_array = grad_tn._array

                if has_momentum:
                    # 与 torch 一致：权重衰减生成新梯度，不回写 p.grad
                    if has_weight_decay:
                        grad = weight_decay * param_data
                        grad += grad_array
                    else:
                        grad = grad_array

                    param_id = id(p)
                    param_state = self.state.get(param_id)

                    if param_state is None:
                        # torch 首个动量缓存直接取梯度，dampening 从第二步起生效
                        velocity = grad.copy()
                        self.state[param_id] = {'velocity': velocity}
                    else:
                        velocity = param_state['velocity']
                        velocity *= momentum
                        if has_dampening:
                            velocity += (1.0 - dampening) * grad
                        else:
                            velocity += grad

                    if nesterov:
                        # 与 torch 一致：使用更新后的动量缓存（原地缩放省一次分配）
                        update = grad + momentum * velocity
                        update *= lr
                        p._array -= update
                    else:
                        p._array -= lr * velocity
                elif has_weight_decay:
                    # 算子融合：p = p*(1 - lr*wd) - lr*g，
                    # 与 p -= lr*(g + wd*p) 代数等价，省去构造衰减梯度的两次全量读写
                    param_data *= (1.0 - lr * weight_decay)
                    param_data -= lr * grad_array
                else:
                    p._array -= lr * grad_array
        
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
        arrlib = self.arrlib

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
                # if p.grad._array.shape != p._array.shape:
                #     raise ValueError(f"Gradient shape {p.grad._array.shape} does not match parameter shape {p._array.shape}")
                
                grad_array = p.grad._array

                # 添加权重衰减（复用临时数组，且不回写 p.grad）
                if weight_decay > 0:
                    grad = weight_decay * p._array
                    grad += grad_array
                else:
                    grad = grad_array

                param_id = id(p)

                # 初始化状态
                if param_id not in self.state:
                    self.state[param_id] = {
                        'step': 0,
                        'exp_avg': arrlib.zeros_like(p._array),
                        'exp_avg_sq': arrlib.zeros_like(p._array)
                    }
                    if amsgrad:
                        self.state[param_id]['max_exp_avg_sq'] = arrlib.zeros_like(p._array)

                state = self.state[param_id]
                state['step'] += 1  # 先递增步数

                # 优化更新计算，使用原地操作减少内存分配
                arrlib.multiply(beta1, state['exp_avg'], out=state['exp_avg'])
                arrlib.add(state['exp_avg'], (1.0 - beta1) * grad, out=state['exp_avg'])

                arrlib.multiply(beta2, state['exp_avg_sq'], out=state['exp_avg_sq'])
                grad_sq = arrlib.square(grad)
                grad_sq *= (1.0 - beta2)
                arrlib.add(state['exp_avg_sq'], grad_sq, out=state['exp_avg_sq'])

                # 偏差校正
                bias_correction1 = 1.0 - beta1 ** state['step']
                bias_correction2 = 1.0 - beta2 ** state['step']
                bias_correction2_sqrt = bias_correction2 ** 0.5

                # 与 torch 一致：sqrt(v) 先除以 sqrt(bias_correction2) 再加 eps，
                # 等价于 eps 放大 sqrt(bias_correction2) 倍，二者不可互换
                if amsgrad:
                    arrlib.maximum(state['max_exp_avg_sq'], state['exp_avg_sq'], out=state['max_exp_avg_sq'])
                    denom = arrlib.sqrt(state['max_exp_avg_sq'])
                else:
                    denom = arrlib.sqrt(state['exp_avg_sq'])
                denom /= bias_correction2_sqrt
                denom += eps

                # 计算更新量
                step_size = lr / bias_correction1
                update = state['exp_avg'] / denom
                update *= step_size
                p._array -= update

# ---------------------------------------------------------------------------
# Adamax 算法设计
#
# 1. 动机
#    Adam 的二阶矩 v 累积 g²（L2 范数），量级随梯度分布变化，必须做偏差校正；
#    Adamax 把二阶矩换成指数加权的无穷范数 u = max(β2·u, |g|+ε)，每个坐标只保留
#    "历史上最大的 |梯度|"。因为 u 是逐元素取 max 而不是求和平均，不需要偏差校正；
#    且对偶发大梯度更稳健：u 被抬高后只按 β2 缓慢衰减。
#
# 2. 状态与递推（t 为该参数被更新的次数）
#    m_t = β1·m_{t-1} + (1-β1)·g_t              # 一阶矩 EMA，m_0 = 0
#    u_t = max(β2·u_{t-1}, |g_t| + ε)           # 无穷范数 EMA，u_0 = 0
#    θ_t = θ_{t-1} - η/(1-β1^t) · m_t / u_t     # 只有一阶矩做偏差校正
#
# 3. 与 torch.optim.Adamax 对齐的关键细节
#    - eps 加在 |g| 上（max 内部），而不是加在 u 上；因此 u 恒 ≥ ε > 0，除法不会除零。
#    - u 不做偏差校正；有效步长是 lr/(1-β1^t)。
#    - weight_decay 是 L2 正则：先把 wd·θ 并进梯度再走上述递推（非解耦），
#      合成梯度用临时数组，不回写 p.grad。
#
# 4. 实现要点
#    - 状态延迟初始化（首个有梯度的 step 才创建），键为 id(param)，值为
#      {'step', 'exp_avg', 'exp_inf'}，可被 state_dict/load_state_dict 原样搬运。
#    - 全部使用 arrlib（CPU=numpy、CUDA=cupy）原地算子，避免每步新增大数组；
#      仅 |g|+ε 需要一个临时数组（abs_grad）。
#    - 参数写回用 p._array -= update，保持 TN 底层存储类型不变。
# ---------------------------------------------------------------------------
class Adamax(Optimizer):
    """
    Adamax 优化器
    
    Adam 的 ∞ 范数变体：把二阶矩的 L2 范数换成指数加权的无穷范数，
    因此二阶矩不需要偏差校正，数值也更稳定。
    
    参数更新公式:
        m = β1*m + (1-β1)*∇θL(θ)  # 一阶矩估计
        u = max(β2*u, |∇θL(θ)| + ε)  # 指数加权的无穷范数
        θ = θ - (η / (1-β1^t)) * m / u  # 参数更新
    
    适用场景:
        - 梯度稀疏或含较大离群值的任务
        - 需要比 Adam 更稳定的二阶矩估计的场景
    """
    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        """
        初始化Adamax优化器
        
        参数:
            params: 待优化参数组 (需包含requires_grad=True的TN对象)
            lr: 学习率 (默认2e-3，与 torch 一致)
            betas: 一阶矩与无穷范数的系数，格式为(beta1, beta2) (默认(0.9, 0.999))
            eps: 数值稳定性参数，加在 |梯度| 上防止除零 (默认1e-8)
            weight_decay: 权重衰减系数 (默认0.0)
        
        异常:
            ValueError: 当学习率、betas、eps或权重衰减系数无效时抛出
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
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        """
        执行单个优化步骤（参数更新）
        
        算法步骤：
        1. 应用权重衰减（如果启用）
        2. 更新一阶矩的指数移动平均
        3. 用 |梯度| + eps 更新指数加权的无穷范数
        4. 按 Adamax 规则调整参数（二阶矩无需偏差校正）
        """
        arrlib = self.arrlib
        
        # 延迟初始化state
        if not hasattr(self, 'state'):
            self.state = {}
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            weight_decay = group.get('weight_decay', 0.0)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad_array = p.grad._array

                # 添加权重衰减（复用临时数组，且不回写 p.grad）
                if weight_decay > 0:
                    grad = weight_decay * p._array
                    grad += grad_array
                else:
                    grad = grad_array

                param_id = id(p)

                # 初始化状态
                if param_id not in self.state:
                    self.state[param_id] = {
                        'step': 0,
                        'exp_avg': arrlib.zeros_like(p._array),
                        'exp_inf': arrlib.zeros_like(p._array)
                    }

                state = self.state[param_id]
                state['step'] += 1

                # 一阶矩：m = β1*m + (1-β1)*g
                arrlib.multiply(beta1, state['exp_avg'], out=state['exp_avg'])
                arrlib.add(state['exp_avg'], (1.0 - beta1) * grad, out=state['exp_avg'])

                # 无穷范数：u = max(β2*u, |g| + ε)，eps 加在 |g| 上（与 torch 一致）
                arrlib.multiply(beta2, state['exp_inf'], out=state['exp_inf'])
                abs_grad = arrlib.abs(grad)
                abs_grad += eps
                arrlib.maximum(state['exp_inf'], abs_grad, out=state['exp_inf'])

                # 一阶矩的偏差校正；无穷范数不做偏差校正
                bias_correction1 = 1.0 - beta1 ** state['step']
                step_size = lr / bias_correction1

                update = state['exp_avg'] / state['exp_inf']
                update *= step_size
                p._array -= update

class AdamW(Optimizer):
    """
    AdamW（Adam with Weight Decay）优化器
    
    Adam的改进版本，将权重衰减作为独立的正则化项处理，而非Adam中的梯度修改。
    这使得权重衰减能够更有效地作为L2正则化，避免了Adam中原有的权重衰减副作用。
    
    参数更新公式:
        m = β1*m + (1-β1)*∇θL(θ)  # 一阶矩估计
        v = β2*v + (1-β2)*(∇θL(θ))²  # 二阶矩估计
        m̂ = m/(1-β1^t)  # 偏差校正的一阶矩估计
        v̂ = v/(1-β2^t)  # 偏差校正的二阶矩估计
        θ = θ * (1 - η * weight_decay)  # 权重衰减
        θ = θ - η*m̂/(√v̂ + ε)  # 参数更新
    
    适用场景:
        - Transformer等大型模型的训练
        - 需要更强正则化的深度学习任务
        - 对权重衰减敏感的模型架构
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False):
        """
        初始化AdamW优化器
        
        参数:
            params: 待优化参数组 (需包含requires_grad=True的TN对象)
            lr: 学习率 (默认1e-3)
            betas: 用于计算一阶和二阶矩估计的系数，格式为(beta1, beta2) (默认(0.9, 0.999))
            eps: 数值稳定性参数，防止除零错误 (默认1e-8)
            weight_decay: 权重衰减系数 (默认0.01，AdamW通常使用较大的权重衰减)
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
        
        1. 更新一阶和二阶矩估计
        2. 执行偏差校正
        3. 应用权重衰减（作为独立的L2正则化项）
        4. 根据Adam更新规则调整参数
        """
        arrlib = self.arrlib

        # 延迟初始化state
        if not hasattr(self, 'state'):
            self.state = {}
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            weight_decay = group.get('weight_decay', 0.01)
            amsgrad = group.get('amsgrad', False)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad._array
                param_id = id(p)
                
                # 初始化状态
                if param_id not in self.state:
                    self.state[param_id] = {
                        'step': 0,
                        'exp_avg': arrlib.zeros_like(p._array),
                        'exp_avg_sq': arrlib.zeros_like(p._array)
                    }
                    if amsgrad:
                        self.state[param_id]['max_exp_avg_sq'] = arrlib.zeros_like(p._array)
                
                state = self.state[param_id]
                state['step'] += 1  # 先递增步数
                
                # 优化更新计算，使用原地操作减少内存分配
                arrlib.multiply(beta1, state['exp_avg'], out=state['exp_avg'])
                arrlib.add(state['exp_avg'], (1.0 - beta1) * grad, out=state['exp_avg'])
                
                arrlib.multiply(beta2, state['exp_avg_sq'], out=state['exp_avg_sq'])
                grad_sq = arrlib.square(grad)
                grad_sq *= (1.0 - beta2)
                arrlib.add(state['exp_avg_sq'], grad_sq, out=state['exp_avg_sq'])

                # 偏差校正
                bias_correction1 = 1.0 - beta1 ** state['step']
                bias_correction2 = 1.0 - beta2 ** state['step']
                bias_correction2_sqrt = bias_correction2 ** 0.5

                # 与 torch 一致：sqrt(v) 先除以 sqrt(bias_correction2) 再加 eps，
                # 等价于 eps 放大 sqrt(bias_correction2) 倍，二者不可互换
                if amsgrad:
                    arrlib.maximum(state['max_exp_avg_sq'], state['exp_avg_sq'], out=state['max_exp_avg_sq'])
                    denom = arrlib.sqrt(state['max_exp_avg_sq'])
                else:
                    denom = arrlib.sqrt(state['exp_avg_sq'])
                denom /= bias_correction2_sqrt
                denom += eps

                # 计算更新量
                step_size = lr / bias_correction1

                # 应用权重衰减（作为独立的L2正则化项）
                if weight_decay > 0:
                    p._array *= (1 - lr * weight_decay)

                # 根据Adam更新规则调整参数
                update = state['exp_avg'] / denom
                update *= step_size
                p._array -= update

class RMSprop(Optimizer):
    """
    RMSprop（Root Mean Square Propagation）优化器
    
    Adagrad的改进版本，通过指数移动平均调整学习率，解决了Adagrad学习率衰减过快的问题。
    适用于递归神经网络（RNN）等序列模型的训练。
    
    参数更新公式:
        v = α*v + (1-α)*(∇θL(θ))²  # 二阶矩的指数移动平均
        θ = θ - η*∇θL(θ)/(√v + ε)  # 参数更新
    
    适用场景:
        - 递归神经网络（RNN）训练
        - 长序列数据处理
        - 需要稳定学习率的优化问题
    """
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0, centered=False):
        """
        初始化RMSprop优化器
        
        参数:
            params: 待优化参数组 (需包含requires_grad=True的TN对象)
            lr: 学习率 (默认0.01)
            alpha: 平滑常数，用于计算二阶矩的指数移动平均 (默认0.99)
            eps: 数值稳定性参数，防止除零错误 (默认1e-8)
            weight_decay: 权重衰减系数 (默认0.0)
            momentum: 动量系数 (默认0.0)
            centered: 是否使用中心化的RMSprop (默认False)
        
        异常:
            ValueError: 当学习率、alpha、eps或权重衰减系数为负数时抛出
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if alpha < 0.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)
        super().__init__(params, defaults)
        # 移除state初始化，采用延迟初始化策略

    def step(self):
        """
        执行单个优化步骤（参数更新）
        
        1. 应用权重衰减（如果启用）
        2. 更新二阶矩的指数移动平均
        3. 如果启用中心化，计算中心化的二阶矩
        4. 如果启用动量，更新动量项
        5. 根据RMSprop更新规则调整参数
        """
        arrlib = self.arrlib

        # 延迟初始化state
        if not hasattr(self, 'state'):
            self.state = {}
        
        for group in self.param_groups:
            alpha = group['alpha']
            eps = group['eps']
            lr = group['lr']
            weight_decay = group.get('weight_decay', 0.0)
            momentum = group.get('momentum', 0.0)
            centered = group.get('centered', False)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad_array = p.grad._array

                # 应用权重衰减（复用临时数组，且不回写 p.grad）
                if weight_decay > 0:
                    grad = weight_decay * p._array
                    grad += grad_array
                else:
                    grad = grad_array

                param_id = id(p)

                # 初始化状态
                if param_id not in self.state:
                    self.state[param_id] = {
                        'step': 0,
                        'square_avg': arrlib.zeros_like(p._array)
                    }
                    if momentum > 0:
                        self.state[param_id]['momentum_buffer'] = arrlib.zeros_like(p._array)
                    if centered:
                        self.state[param_id]['grad_avg'] = arrlib.zeros_like(p._array)

                state = self.state[param_id]
                state['step'] += 1

                # 更新二阶矩的指数移动平均
                square_avg = state['square_avg']
                arrlib.multiply(alpha, square_avg, out=square_avg)
                grad_sq = arrlib.square(grad)
                grad_sq *= (1.0 - alpha)
                arrlib.add(square_avg, grad_sq, out=square_avg)

                # 计算分母
                if centered:
                    grad_avg = state['grad_avg']
                    arrlib.multiply(alpha, grad_avg, out=grad_avg)
                    arrlib.add(grad_avg, (1.0 - alpha) * grad, out=grad_avg)
                    denom = arrlib.sqrt(square_avg - arrlib.square(grad_avg)) + eps
                else:
                    denom = arrlib.sqrt(square_avg)
                    denom += eps

                # 应用动量（与 torch 一致：缓存不含学习率，更新时再乘 lr）
                if momentum > 0:
                    momentum_buffer = state['momentum_buffer']
                    arrlib.multiply(momentum, momentum_buffer, out=momentum_buffer)
                    arrlib.add(momentum_buffer, grad / denom, out=momentum_buffer)
                    p._array -= lr * momentum_buffer
                else:
                    # 直接更新参数：-lr * grad / denom
                    update = lr * grad
                    update /= denom
                    p._array -= update

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
        arrlib = self.arrlib
        
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
                # if p.grad._array.shape != p._array.shape:
                #     raise ValueError(f"Gradient shape {p.grad._array.shape} does not match parameter shape {p._array.shape}")
                
                # 验证数据类型
                # if p.grad._array.dtype != p._array.dtype:
                #     raise TypeError(f"Gradient data type {p.grad._array.dtype} does not match parameter data type {p._array.dtype}")
                
                grad = p.grad._array
                param_id = id(p)

                # 初始化状态
                if param_id not in self.state:
                    self.state[param_id] = {
                        'step': 0,
                        'sum': arrlib.full_like(p._array, initial_accumulator_value)
                    }

                state = self.state[param_id]
                state['step'] += 1

                # 应用权重衰减（生成新梯度，不回写 p.grad）
                if weight_decay > 0:
                    grad = weight_decay * p._array
                    grad += p.grad._array

                # 计算梯度平方
                grad_squared = arrlib.square(grad)

                # 更新梯度平方累加器（原地操作）
                sum_sq = state['sum']
                arrlib.add(sum_sq, grad_squared, out=sum_sq)

                # 计算学习率衰减
                lr_t = lr / (1.0 + (state['step'] - 1.0) * lr_decay)

                # 计算更新分母（增强数值稳定性）
                denom = arrlib.sqrt(arrlib.maximum(sum_sq, 1e-16))
                denom += eps

                # 计算更新量并应用（使用原地操作减少内存分配）
                # θ = θ - (lr_t / denom) * grad
                update = lr_t * grad
                update /= denom
                arrlib.subtract(p._array, update, out=p._array)


# ---------------------------------------------------------------------------
# Adadelta 算法设计
#
# 1. 动机
#    Adagrad 的分母是历史梯度平方的累加，随迭代单调增长，学习率会衰减到 0；
#    Adadelta 用指数移动平均把统计窗口限制在近期（衰减系数 rho），并用"更新量自身的
#    RMS"代替手调学习率，使步长自动随梯度尺度变化。
#
# 2. 状态与递推（t 为该参数被更新的次数）
#    s_t = ρ·s_{t-1} + (1-ρ)·g_t²                # 梯度平方 EMA，s_0 = 0
#    Δ_t = √(u_{t-1} + ε)/√(s_t + ε) · g_t       # 由两个 RMS 的比值缩放更新量
#    θ_t = θ_{t-1} - η·Δ_t                       # η 默认 1.0，仅作整体缩放
#    u_t = ρ·u_{t-1} + (1-ρ)·Δ_t²                # 更新量平方 EMA，u_0 = 0
#
# 3. 与 torch.optim.Adadelta 对齐的关键细节
#    - eps 在开方前分别加到 s 和 u 上（√(s+ε)、√(u+ε)），既防除零也参与缩放；
#      若只加在分母上，首步（u=0）的行为会不同。
#    - 首步 u=0，故 Δ = √ε/√((1-ρ)g²+ε)·g；当 |g| 远大于 ε 时约为
#      √ε/√(1-ρ)·sign(g)，量级约 √ε/√(1-ρ)。所以 Adadelta 起步慢是算法固有特性，
#      随 u 累积逐步恢复，不是实现缺陷。
#    - 分母用已含当前梯度的 s_t（而非 s_{t-1}）；u 的最后更新复用同一个 Δ_t。
#    - weight_decay 是 L2 正则：先把 wd·θ 并进梯度再走上述递推（非解耦）。
#
# 4. 实现要点
#    - 状态延迟初始化，键为 id(param)，值为 {'step', 'square_avg', 'acc_delta'}，
#      可被 state_dict/load_state_dict 原样搬运。
#    - √(s+ε) 与 √(u+ε) 都通过 arrlib 生成新数组，不污染状态；随后"先除后乘"
#      （delta /= std; delta *= grad），再 p._array -= lr*delta，与 torch 的运算顺序一致。
#    - 每步每参数的内存开销固定为两个与参数同形的状态数组（square_avg、acc_delta），
#      计算量为约 6 次逐元素运算 + 2 次 sqrt。
#
# 5. 收敛特性与调参（在 test_060 的二维基准上实测）
#    - eps 实际是"步长下限"的控制量：|g| 很小时 √(s+ε) ≈ √ε 主导分母，步长被限制在
#      ~√ε/√(1-ρ) 附近，无法继续细化。Rosenbrock（20000 步）：
#      ε=1e-6 → 损失 1.8e-1，ε=1e-8 → 4.9e-3，ε=1e-10 → 4.6e-5。
#      因此梯度尺度很小的任务应把 eps 调小。
#    - lr 只整体缩放 Δ，不能突破 eps 造成的瓶颈，且过大会在窄谷中振荡：
#      Rosenbrock 上 lr=2 比 lr=1 更差（2.3e-2 对 4.9e-3），lr=10 时退化到 7.9e-1；
#      Beale 则需要 lr=2 配 ε=1e-8 才能到 3e-2。
#    - 起步阶段 u=0 使有效步长极小（见第 3 点），相同迭代预算下通常比 Adam/Adamax/
#      Adagrad 慢，这是算法特性而非实现问题；与 torch 在同预算下逐步对照，损失差异仅 1e-5 量级。
#
# 6. 调用关系
#    Optimizer.__init__ → add_param_group 建立 param_groups → 本类 step():
#      step()
#       └─ for group（lr、rho、eps、weight_decay）→ for p（跳过 grad 为 None 的参数）
#            ├─ 权重衰减：临时梯度 = wd·θ + g（不回写 p.grad）
#            ├─ 状态延迟初始化：{'step', 'square_avg', 'acc_delta'}
#            ├─ 更新 s：s ← ρ·s + (1-ρ)·g²（原地）
#            ├─ 计算 Δ：√(u+ε)/√(s+ε)·g（两个 sqrt 均生成新数组）
#            ├─ 参数写回：θ ← θ - lr·Δ
#            └─ 更新 u：u ← ρ·u + (1-ρ)·Δ²（原地）
#    外部依赖：zero_grad 负责清梯度；state_dict/load_state_dict 按位置搬运 self.state；
#    本类不维护组级额外状态，全部超参数都在 param_groups 中。
# ---------------------------------------------------------------------------
class Adadelta(Optimizer):
    """
    Adadelta 优化器
    
    Adagrad 的改进版本：用梯度平方与更新量平方的指数移动平均代替全历史累加，
    并用更新量的 RMS 自动缩放步长，因此基本不需要手动调学习率。
    
    参数更新公式:
        s = ρ*s + (1-ρ)*(∇θL(θ))²  # 梯度平方的指数移动平均
        Δ = √(u + ε)/√(s + ε)*∇θL(θ)  # 由 RMS 比值缩放的更新量
        θ = θ - η*Δ  # 参数更新
        u = ρ*u + (1-ρ)*Δ²  # 更新量平方的指数移动平均
    
    适用场景:
        - 不想仔细调学习率的任务
        - 需要自适应步长缩放的中长训练过程
    """
    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.0):
        """
        初始化Adadelta优化器
        
        参数:
            params: 待优化参数组 (需包含requires_grad=True的TN对象)
            lr: 学习率 (默认1.0，与 torch 一致；步长主要由 RMS 比值决定)
            rho: 平方平均的衰减系数 (默认0.9)
            eps: 数值稳定性参数 (默认1e-6)
            weight_decay: 权重衰减系数 (默认0.0)
        
        异常:
            ValueError: 当学习率、rho、eps或权重衰减系数无效时抛出
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"Invalid rho value: {rho}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        """
        执行单个优化步骤（参数更新）
        
        算法步骤：
        1. 应用权重衰减（如果启用）
        2. 更新梯度平方的指数移动平均
        3. 用梯度平方与更新量平方的 RMS 计算缩放后的更新量
        4. 应用更新，并累计更新量平方的指数移动平均
        
        与 torch 一致：eps 在开方前加到两个平方平均上，学习率只作用于最终更新量。
        """
        arrlib = self.arrlib
        
        # 延迟初始化state
        if not hasattr(self, 'state'):
            self.state = {}
        
        for group in self.param_groups:
            lr = group['lr']
            rho = group['rho']
            eps = group['eps']
            weight_decay = group.get('weight_decay', 0.0)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad_array = p.grad._array

                # 应用权重衰减（复用临时数组，且不回写 p.grad）
                if weight_decay > 0:
                    grad = weight_decay * p._array
                    grad += grad_array
                else:
                    grad = grad_array

                param_id = id(p)

                # 初始化状态
                if param_id not in self.state:
                    self.state[param_id] = {
                        'step': 0,
                        'square_avg': arrlib.zeros_like(p._array),
                        'acc_delta': arrlib.zeros_like(p._array)
                    }

                state = self.state[param_id]
                state['step'] += 1

                # s = ρ*s + (1-ρ)*g²
                square_avg = state['square_avg']
                arrlib.multiply(rho, square_avg, out=square_avg)
                grad_sq = arrlib.square(grad)
                grad_sq *= (1.0 - rho)
                arrlib.add(square_avg, grad_sq, out=square_avg)

                # Δ = √(u + ε)/√(s + ε)*g（两个 sqrt 都是新数组，不改动状态）
                std = arrlib.sqrt(square_avg + eps)
                delta = arrlib.sqrt(state['acc_delta'] + eps)
                delta /= std
                delta *= grad

                # θ = θ - η*Δ
                p._array -= lr * delta

                # u = ρ*u + (1-ρ)*Δ²
                acc_delta = state['acc_delta']
                arrlib.multiply(rho, acc_delta, out=acc_delta)
                delta_sq = arrlib.square(delta)
                delta_sq *= (1.0 - rho)
                arrlib.add(acc_delta, delta_sq, out=acc_delta)


class _VectorFuncs(NamedTuple):
    """参数/梯度与扁平向量之间的转换函数集合。

    p2v:   参数列表 -> 扁平向量
    v2p:   扁平向量 -> 写回参数
    g2v:   当前梯度 -> 扁平梯度向量
    gdot:  当前梯度与方向向量的点积（不构造整条梯度向量）
    """
    p2v: Callable
    v2p: Callable
    g2v: Callable
    gdot: Callable


class _LineSearchResult(NamedTuple):
    """线搜索结果。

    success 为 False 时 s/y/params_vec/loss 均为 None。
    """
    success: bool
    s: Any
    y: Any
    evals: int
    params_vec: Any
    loss: Any


# ---------------------------------------------------------------------------
# L-BFGS 算法设计与调用关系
#
# 一、向量化设计
#   一个 step 内把所有参数组的参数拼成一条扁平向量（total_params），在向量空间
#   完成方向计算与线搜索，再写回各参数。超参数与历史状态统一取自
#   param_groups[0]，因此"多参数组"是一次联合优化，而不是逐组独立优化。
#   CPU 用 numpy、CUDA 用 cupy，arrlib 每步按参数实际设备推断（_infer_arrlib），
#   支持"先建优化器、再把模型迁到 CUDA"的用法。
#
# 二、历史与缓存
#   group 上保存 old_dirs(s_k)、old_stps(y_k)、old_ys(y_k·s_k)、H_diag(H0 缩放)，
#   最多保留 history_size 对；old_ys 在入库时算好，双循环中不再重复点积。
#   入库判据是相对曲率 ys > 1e-12·‖s‖·‖y‖（_update_history），非正/退化曲率对
#   直接丢弃以保持 Hessian 近似正定。向量转换闭包、参数信息、arrlib 存于
#   self._group_caches（不写入 param_groups，保证 state_dict 可 pickle）；
#   参数对象集合或设备变化时重建缓存并清空历史
#   （_get_vector_funcs / _rebuild_cache / _validate_history）。
#
# 三、单步流程（step 只做编排，三个阶段各自独立）
#   step
#   ├─ _prepare          读超参、初始化/取出组状态、取向量函数、空梯度时先跑一次闭包
#   ├─ _iterate          主循环（≤ max_iter 轮，闭包调用次数 ≤ max_eval）
#   │  ├─ closure()                          当前点的损失与梯度
#   │  ├─ _check_convergence                 max|g| < tol_grad 或 上次被接受步长 < tol_change
#   │  ├─ _compute_lbfgs_direction           双循环（two-loop）得搜索方向 d，并更新 H0 缩放
#   │  ├─ 下降方向校验                       g·d ≥ 0 时退回最陡下降 -g
#   │  ├─ _line_search                       回溯线搜索（Armijo + 弱 Wolfe 曲率）
#   │  │   └─ _strong_wolfe_search           line_search_fn='strong_wolfe' 时改用括号 + 二分
#   │  ├─ _update_history                    接受步时把曲率对 (s, y) 入库
#   │  └─ _fallback_step                     线搜索失败时回退当前点，并试一个受检的最速下降步
#   └─ _finalize         写回组状态、恢复最佳参数、返回最终损失
#
#   辅助：_record_best 记录最优可行点；_restore_best_params 写回最优参数，若最优点
#   与最后一次评估点不同则重跑一次闭包，保证 p.grad 与参数同步；_loss_value 允许
#   闭包返回 Tensor / 数组标量 / Python 数值；_compute_final_loss 在最优损失有限时
#   避免重复求值；_create_vector_functions 提供 p2v/v2p/g2v/gdot 四个向量算子。
#
# 四、关键不变量
#   - 收敛判据只看本次 step 内被接受的步长（last_step_norm），历史步长不参与，
#     否则手动移动参数后会立即误判收敛。
#   - max_eval 是闭包调用次数的硬上限：线搜索试探预算与兜底步都从剩余预算中扣减。
#   - 历史长度即有效曲率对数量，统一由 len(old_dirs) 给出，不另设计数器。
# ---------------------------------------------------------------------------
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
    # 线搜索常数（Armijo/曲率条件、最大回溯次数、兜底步长）
    _C1 = 1e-4
    _C2 = 0.9
    _MAX_BACKTRACKS = 10
    _SAFE_ETA = 1e-3

    def __init__(self, params, lr=1.0, max_iter=20, max_eval=None, tolerance_grad=1e-5, 
                 tolerance_change=1e-9, history_size=100, line_search_fn=None):
        """
        初始化LBFGS优化器
        
        参数:
            params: 待优化参数组
            lr: 学习率
            max_iter: 每个优化步骤中的最大迭代次数
            max_eval: 每个优化步骤中的最大闭包评估次数（硬上限）。
                默认是 1.25×max_iter 再预留 2 次线搜索试探，保证小 max_iter
                也能完成至少一次线搜索
            tolerance_grad: 梯度收敛阈值（与 max|g| 比较）
            tolerance_change: 参数变化收敛阈值（与 max|Δθ| 比较）
            history_size: 历史记录大小
            line_search_fn: 线搜索方式。None 表示回溯线搜索（Armijo + 弱 Wolfe 曲率），
                'strong_wolfe' 表示括号+二分实现的强 Wolfe 线搜索
        """
        if max_eval is None:
            max_eval = max_iter * 5 // 4 + 2

        if line_search_fn not in (None, 'strong_wolfe'):
            raise ValueError(
                f"Invalid line_search_fn: {line_search_fn}，仅支持 None 或 'strong_wolfe'")
            
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

        # 向量转换函数等运行时缓存放在优化器内部，不写入 param_groups：
        # param_info 持有参数引用、vector_funcs 是闭包，写入参数组会让
        # state_dict 带出不可 pickle 的对象
        self._group_caches = {}

        # 为每个参数组初始化状态
        for group in self.param_groups:
            group.setdefault('old_dirs', [])
            group.setdefault('old_stps', [])
            group.setdefault('old_ys', [])
            group.setdefault('H_diag', 1.0)

    @staticmethod
    def _infer_arrlib(params):
        """按参数实际所在设备推断数组库。

        不使用构造时缓存的设备，从而支持"先建优化器、再把模型迁到 CUDA"的用法。
        """
        for p in params:
            device = getattr(p, 'device', None)
            if device is not None and getattr(device, 'type', None) == 'cuda':
                return cp
        return np

    @staticmethod
    def _loss_value(loss):
        """把闭包返回值规范化为 Python float。

        允许闭包返回 Tensor、0 维数组、numpy 标量或 Python 数值，
        避免要求闭包必须返回 Tensor。
        """
        if isinstance(loss, (int, float)):
            return float(loss)
        if hasattr(loss, 'item'):
            return float(loss.item())
        return float(loss)

    def _get_vector_funcs(self, group, params=None):
        """返回参数组的向量转换函数缓存，首次使用时构建。

        缓存以参数组对象身份为键保存在优化器内部，不进入 state_dict。
        """
        if params is None:
            params = group['params']
        cache = self._group_caches.get(id(group))
        current_ids = [id(p) for p in params]
        current_arrlib = self._infer_arrlib(params)
        if cache is not None and (
                cache.get('param_ids') != current_ids
                or cache.get('arrlib') is not current_arrlib):
            # 参数对象集合或所在设备发生变化：历史向量与新的参数布局/设备不再对应，清空
            self._reset_history(group)
            cache = None
        if cache is None:
            param_info, total_params = self._compute_param_info(params)
            cache = {
                'param_info': param_info,
                'total_params': total_params,
                'param_ids': current_ids,
                'arrlib': current_arrlib,
                'vector_funcs': self._create_vector_functions(
                    param_info, total_params, current_arrlib),
            }
            self._group_caches[id(group)] = cache
        return cache

    def _rebuild_cache(self, group, cache, params=None):
        """按当前参数/梯度重建向量转换函数，并刷新参数身份列表。"""
        if params is None:
            params = group['params']
        param_info, total_params = self._compute_param_info(params)
        cache['param_info'] = param_info
        cache['total_params'] = total_params
        cache['param_ids'] = [id(p) for p in params]
        cache['arrlib'] = self._infer_arrlib(params)
        cache['vector_funcs'] = self._create_vector_functions(
            param_info, total_params, cache['arrlib'])
        return cache['vector_funcs']

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """加载状态，并丢弃持有旧参数组引用的向量转换缓存。"""
        super().load_state_dict(state_dict)
        self._group_caches.clear()

    def _validate_history(self, group, total_params):
        """校验已加载的历史向量与当前参数（展平后）维度一致。

        历史向量的长度由首次 step 时的参数集合决定；若加载到参数数量/形状
        不同的模型，后续点积会出现难以理解的形状错误，这里提前给出明确报错。
        """
        if total_params == 0:
            return
        history = list(group.get('old_dirs', [])) + list(group.get('old_stps', []))
        for vec in history:
            shape = tuple(getattr(vec, 'shape', ()))
            if shape != (total_params,):
                raise ValueError(
                    'LBFGS 状态与当前参数不匹配: 历史向量形状为 '
                    f'{shape}，当前参数展平后长度应为 {total_params}。'
                    '请确认加载的 state_dict 与模型参数一致')

    def _create_vector_functions(self, param_info, total_params, arrlib=None):
        """
        创建参数和梯度向量转换的辅助函数
        
        Args:
            param_info: 参数信息列表
            total_params: 参数总数
            arrlib: 数组库（numpy或cupy），None 表示按参数设备推断
            
        Returns:
            tuple: (params_to_vector, vector_to_params, grads_to_vector) 三个辅助函数
        """
        if arrlib is None:
            arrlib = self._infer_arrlib([info['param'] for info in param_info])
        
        # 同一参数组内的参数必须是同一浮点 dtype：
        # 向量按首个参数的 dtype 创建并回写成视图，混合 dtype 会让高精度参数被
        # 静默降级（例如 float64 参数在一步后变成 float32），因此直接报错
        dtypes = {info['param'].dtype for info in param_info}
        if len(dtypes) > 1:
            raise ValueError(
                'LBFGS 要求同一参数组内所有参数 dtype 一致，当前包含: '
                + ', '.join(sorted(str(d) for d in dtypes)))
        for dtype in dtypes:
            if not np.issubdtype(dtype, np.floating):
                raise ValueError(f'LBFGS 仅支持浮点参数，当前 dtype 为 {dtype}')
        param_dtype = next((info['param'].dtype for info in param_info), None)
        
        # 将参数列表转换为单一向量
        def params_to_vector(param_list):
            """将参数列表转换为单一向量"""
            # 所有参数都会被写入，使用 empty 省去清零的一遍写入
            vec = arrlib.empty(total_params, dtype=param_dtype)
            for info in param_info:
                p = info['param']
                offset = info['offset']
                numel = info['numel']
                if p._array.ndim == 0:  # 标量参数
                    vec[offset] = p.item()
                else:
                    vec[offset:offset + numel] = p._array.flatten()
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
                    # vec[offset] 是数组标量，直接赋值会让 _array 退化为标量而非
                    # 0 维数组，破坏 TN 底层存储类型不变量，故先转回 0 维数组
                    p._array = arrlib.asarray(vec[offset])
                else:
                    p._array = vec[offset:offset + numel].reshape(shape)
        
        # 将梯度转换为单一向量
        def grads_to_vector():
            """将梯度转换为单一向量"""
            vec = arrlib.zeros(total_params, dtype=param_dtype)
            for info in param_info:
                p = info['param']
                offset = info['offset']
                numel = info['numel']
                if p.grad is None:
                    continue
                if p.grad._array.ndim == 0:  # 标量梯度
                    vec[offset] = p.grad.item()
                else:
                    vec[offset:offset + numel] = p.grad._array.flatten()
            return vec

        def grads_dot(direction):
            """直接计算 g·direction，避免为每次试探构造整条梯度向量"""
            total = 0.0
            for info in param_info:
                p = info['param']
                if p.grad is None:
                    continue
                offset = info['offset']
                numel = info['numel']
                grad_array = p.grad._array
                if grad_array.ndim == 0:  # 标量梯度
                    total += float(grad_array) * float(direction[offset])
                else:
                    total += float(arrlib.dot(grad_array.ravel(),
                                              direction[offset:offset + numel]))
            return total

        return _VectorFuncs(params_to_vector, vector_to_params, grads_to_vector, grads_dot)
    
    def _compute_lbfgs_direction(self, grad_vector, old_dirs, old_stps, old_ys, n_old, H_diag, arrlib):
        """
        计算L-BFGS搜索方向（双循环算法）
        
        Args:
            grad_vector: 当前梯度向量
            old_dirs: 历史参数变化列表
            old_stps: 历史梯度变化列表
            old_ys: 历史曲率对点积 y·s 缓存（与 old_dirs/old_stps 对齐）
            n_old: 历史记录数量
            H_diag: Hessian对角线估计
            arrlib: 数组库（numpy或cupy）
            
        Returns:
            tuple: (search_dir, H_diag) 搜索方向和更新后的Hessian对角线估计
        """
        arrlib_dot = arrlib.dot
        arrlib_zeros_like = arrlib.zeros_like
        arrlib_zeros = arrlib.zeros
        arrlib_subtract = arrlib.subtract
        arrlib_multiply = arrlib.multiply
        arrlib_add = arrlib.add
        arrlib_negative = arrlib.negative
        
        # 曲率点积 y·s 在入库时已算好并缓存于 old_ys，正常路径无需重复计算；
        # 仅当缓存与历史长度不一致（如加载旧版状态字典）时补齐一次
        if len(old_ys) != n_old:
            old_ys[:] = [float(arrlib_dot(old_stps[i], old_dirs[i])) for i in range(n_old)]

        # 入库时已按相对曲率阈值保证 ys > 0，正常路径无需过滤；
        # 仅对旧状态字典中的异常值（非正/NaN）做保护
        valid_indices = [i for i in range(n_old) if old_ys[i] > 0.0]
        
        # 初始化搜索方向
        search_dir = arrlib_zeros_like(grad_vector)
        
        if len(valid_indices) == 0:
            # 没有有效历史，使用最陡下降
            search_dir = -grad_vector.copy()
            H_diag = 1.0
        else:
            # 反向循环 - 直接使用缓存的 ys（Python float），避免每次调用重建数组
            q = grad_vector.copy()
            n_valid = len(valid_indices)
            alphas = arrlib_zeros(n_valid)
            # 复用同一个临时缓冲区，避免每个历史对都分配一个参数规模的数组
            tmp = arrlib_zeros_like(grad_vector)
            
            for idx, i in enumerate(reversed(valid_indices)):
                alpha_k = arrlib_dot(old_dirs[i], q) / old_ys[i]
                alphas[idx] = alpha_k
                # q -= alpha_k * y_k（原地操作）
                arrlib_multiply(old_stps[i], alpha_k, out=tmp)
                arrlib_subtract(q, tmp, out=q)
            
            # 初始Hessian近似
            arrlib_multiply(q, H_diag, out=q)  # 原地操作，q = H_diag * q
            
            # 正向循环 - q += (alpha - beta) * s_k（原地操作）
            for idx, i in enumerate(valid_indices):
                beta_k = arrlib_dot(old_stps[i], q) / old_ys[i]
                arrlib_multiply(old_dirs[i], alphas[n_valid - 1 - idx] - beta_k, out=tmp)
                arrlib_add(q, tmp, out=q)
            
            # 搜索方向为负梯度方向
            arrlib_negative(q, out=search_dir)  # 原地操作，search_dir = -q
        
        return search_dir, H_diag
    
    def _line_search(self, closure, params, current_params_vec, search_dir, grad_vector, 
                     loss_val, gtd, lr, vector_to_params, grads_to_vector, 
                     arrlib, c1, c2, max_backtracks,
                     strong_wolfe=False, max_trials=None, grads_dot=None):
        """
        执行线搜索，寻找满足Wolfe条件的步长
        
        strong_wolfe=False 时使用回溯线搜索（Armijo + 弱 Wolfe 曲率条件）；
        strong_wolfe=True 时使用括号+二分搜索强 Wolfe 条件 |g'·d| <= c2·|g·d|。
        
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
            arrlib: 数组库（numpy或cupy）
            c1: Armijo条件常数
            c2: 曲率条件常数
            max_backtracks: 最大回溯/试探次数
            strong_wolfe: 是否使用强 Wolfe 曲率条件
            max_trials: 允许的最大试探次数（用于遵守 max_eval 预算），None 表示仅受 max_backtracks 限制
            
        Returns:
            _LineSearchResult: 线搜索结果；失败时 s/y/params_vec/loss 为 None
        """
        arrlib_dot = arrlib.dot
        arrlib_isfinite = arrlib.isfinite
        
        if max_trials is None:
            trial_limit = max_backtracks
        else:
            # 注意：本模块的 * 导入覆盖了内置 min/max，这里使用 builtins
            trial_limit = builtins.max(0, builtins.min(int(max_trials), max_backtracks))
        if trial_limit <= 0:
            return _LineSearchResult(False, None, None, 0, None, None)
        if strong_wolfe:
            return self._strong_wolfe_search(
                closure, params, current_params_vec, search_dir, grad_vector,
                loss_val, gtd, lr, vector_to_params, grads_to_vector,
                arrlib, c1, c2, trial_limit, grads_dot=grads_dot)

        old_loss = loss_val
        eta = lr  # 使用学习率作为初始步长，而不是固定的1.0
        func_evals = 0
        line_search_success = False
        s = None
        y = None
        new_params_vec = None
        new_loss = None
        
        # 执行线搜索
        for backtracks in range(trial_limit):
            # 计算新的参数位置：current_params + eta * search_dir
            new_params_vec = current_params_vec + eta * search_dir
            
            # 应用新参数
            vector_to_params(new_params_vec, params)
            
            # 计算新的损失
            loss = closure()
            func_evals += 1
            
            new_loss = self._loss_value(loss)
            
            # Armijo条件检查（充分下降条件）
            if new_loss <= old_loss + c1 * eta * gtd:
                # 先用逐参数点积判断曲率，避免每次试探都构造整条梯度向量
                if grads_dot is not None:
                    new_gtd = grads_dot(search_dir)
                else:
                    new_gtd = arrlib_dot(grads_to_vector(), search_dir)
                
                # 曲率条件检查
                if new_gtd >= c2 * gtd:
                    # 满足Wolfe条件，线搜索成功
                    line_search_success = True
                    
                    # 计算参数变化和梯度变化
                    new_grad_vector = grads_to_vector()
                    s = new_params_vec - current_params_vec
                    y = new_grad_vector - grad_vector
                    
                    break
            
            # 减少步长
            eta *= 0.5
            
        if not line_search_success:
            # 线搜索失败：参数停留在最后一次试探点，由调用方负责恢复
            return _LineSearchResult(False, None, None, func_evals, None, None)
        return _LineSearchResult(True, s, y, func_evals, new_params_vec, new_loss)
    
    def _strong_wolfe_search(self, closure, params, current_params_vec, search_dir,
                             grad_vector, loss_val, gtd, lr, vector_to_params,
                             grads_to_vector, arrlib, c1, c2, trial_limit,
                             grads_dot=None):
        """强 Wolfe 线搜索：倍数扩张找括号 + 括号内二分。

        满足 Armijo 且 |g'·d| <= c2·|g·d| 时接受；若预算耗尽仍未满足强 Wolfe，
        则退而接受已找到的 Armijo 可行点中损失最小者（保证下降）。
        """
        arrlib_dot = arrlib.dot
        arrlib_isfinite = arrlib.isfinite

        eta = lr
        eta_lo = 0.0
        eta_hi = None
        eta_cap = builtins.max(1.0, lr * 1e3)
        func_evals = 0
        best = None  # Armijo 可行点中损失最小者: (eta, params_vec, loss, grad_vec)

        for _ in range(trial_limit):
            if not np.isfinite(eta) or eta <= 0.0:
                break
            candidate = current_params_vec + eta * search_dir
            vector_to_params(candidate, params)
            loss = closure()
            func_evals += 1
            candidate_loss = self._loss_value(loss)

            if (arrlib_isfinite(candidate_loss)
                    and candidate_loss <= loss_val + c1 * eta * gtd):
                # 先用逐参数点积判断曲率，仅在被选为最优候选时才构造整条梯度向量
                if grads_dot is not None:
                    candidate_gtd = float(grads_dot(search_dir))
                else:
                    candidate_gtd = float(arrlib_dot(grads_to_vector(), search_dir))
                if best is None or candidate_loss < best[2]:
                    best = (eta, candidate, candidate_loss, grads_to_vector())
                if builtins.abs(candidate_gtd) <= c2 * builtins.abs(float(gtd)):
                    best = (eta, candidate, candidate_loss, grads_to_vector())
                    break
                if candidate_gtd > 0:
                    eta_hi = eta
                else:
                    eta_lo = eta
            else:
                eta_hi = eta

            if eta_hi is None:
                # 步长偏小：扩大步长
                eta = builtins.min(eta * 2.0, eta_cap)
                if eta >= eta_cap:
                    break
            else:
                # 已有括号：二分收缩
                if eta_hi - eta_lo <= 1e-8 * builtins.max(1.0, builtins.abs(eta_hi)):
                    break
                eta = 0.5 * (eta_lo + eta_hi)

        if best is None:
            return _LineSearchResult(False, None, None, func_evals, None, None)

        _, new_params_vec, new_loss, new_grad_vector = best
        # 二分过程中参数可能停留在其他试探点，写回接受点
        vector_to_params(new_params_vec, params)
        s = new_params_vec - current_params_vec
        y = new_grad_vector - grad_vector
        return _LineSearchResult(True, s, y, func_evals, new_params_vec, new_loss)

    def _update_history(self, old_dirs, old_stps, old_ys, s, y, history_size, arrlib):
        """
        更新L-BFGS历史记录（原地维护，历史长度由 old_dirs 的长度给出）
        
        Args:
            old_dirs: 历史参数变化列表
            old_stps: 历史梯度变化列表
            old_ys: 历史曲率对点积 y·s 缓存
            s: 当前参数变化
            y: 当前梯度变化
            history_size: 最大历史记录大小
            arrlib: 数组库（numpy或cupy）
            
        Returns:
            float | None: h_scale 为该曲率对的 H0 缩放因子 ys/yy；
                曲率不满足条件时不入库，h_scale 为 None
        """
        arrlib_dot = arrlib.dot
        arrlib_sqrt = arrlib.sqrt
        arrlib_isfinite = arrlib.isfinite

        # 相对曲率判据：ys > eps * ||s|| * ||y||，
        # 使用相对阈值避免损失/梯度尺度很小时（绝对值小于 1e-10）永远无法积累历史
        ys = float(arrlib_dot(y, s))
        yy = float(arrlib_dot(y, y))
        ss = float(arrlib_dot(s, s))

        if not (arrlib_isfinite(ys) and arrlib_isfinite(yy) and arrlib_isfinite(ss)):
            return None
        if ys <= 1e-12 * arrlib_sqrt(ss) * arrlib_sqrt(yy):
            # 非正曲率或数值上退化的曲率对：跳过，避免破坏正定性
            return None

        # 更新历史记录
        if len(old_dirs) >= history_size:
            # 移除最旧的历史记录
            old_dirs.pop(0)
            old_stps.pop(0)
            if old_ys:
                old_ys.pop(0)
        old_dirs.append(s)
        old_stps.append(y)
        old_ys.append(ys)

        return ys / yy
    
    def _check_convergence(self, grad_norm, tolerance_grad, last_step_norm, tolerance_change):
        """
        检查收敛条件
        
        Args:
            grad_norm: 当前梯度的最大绝对值范数（max|g|）
            tolerance_grad: 梯度收敛阈值
            last_step_norm: 本次 step 内最近一次被接受参数变化的最大绝对值范数（None 表示尚未走步）
            tolerance_change: 参数变化收敛阈值
            
        Returns:
            bool: 是否收敛
        """
        # 检查梯度收敛条件
        if grad_norm < tolerance_grad:
            return True

        # 检查参数变化收敛条件：只看最近一次接受的步长，
        # 不能用历史记录的步长（历史会跨 step 调用保留，导致移动参数后立即误判收敛）
        if last_step_norm is not None and last_step_norm < tolerance_change:
            return True

        return False
    
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

    @staticmethod
    def _reset_history(group):
        """清空 L-BFGS 历史。

        在参数集合/设备变化、或新参数首次参与训练时调用：
        旧历史向量与新参数布局不再对应，继续使用会破坏正定性。
        """
        group['old_dirs'] = []
        group['old_stps'] = []
        group['old_ys'] = []
        group['H_diag'] = 1.0

    @staticmethod
    def _record_best(best_loss, best_params_vec, params_vec, loss_value, arrlib):
        """记录更优的点（不可行/非有限损失不更新）。

        Returns:
            tuple: (best_loss, best_params_vec)
        """
        if arrlib.isfinite(loss_value) and loss_value < best_loss:
            return loss_value, params_vec.copy()
        return best_loss, best_params_vec

    def _fallback_step(self, closure, params, current_params_vec, grad_vector,
                       loss_val, vector_to_params, arrlib, max_eval, func_evals):
        """线搜索失败后的兜底：回退当前点，并尝试一次受检的最速下降步。

        只有当兜底步确实降低损失时才接受；预算耗尽或没有改善时回退到当前点，
        由调用方结束本次优化。

        Returns:
            tuple: (accepted, new_loss, step_norm, evaluated_vec, func_evals)
                evaluated_vec 为 last closure 评估对应的参数向量（未评估时为 None）
        """
        vector_to_params(current_params_vec, params)
        if func_evals >= max_eval:
            return False, None, None, None, func_evals

        safe_params_vec = current_params_vec - self._SAFE_ETA * grad_vector
        vector_to_params(safe_params_vec, params)
        loss = closure()
        func_evals += 1
        new_loss = self._loss_value(loss)

        if arrlib.isfinite(new_loss) and new_loss < loss_val:
            step_norm = float(arrlib.max(arrlib.abs(safe_params_vec - current_params_vec)))
            return True, new_loss, step_norm, safe_params_vec, func_evals

        # 兜底步没有改善：回到当前点
        vector_to_params(current_params_vec, params)
        return False, new_loss, None, safe_params_vec, func_evals
    
    def _compute_final_loss(self, closure, best_loss):
        """
        计算最终损失值
        
        Args:
            closure: 重新计算损失的闭包函数
            best_loss: 最佳损失值
            
        Returns:
            Tensor: 最终损失值
        """
        # 优化：只有在需要时才重新计算损失
        # 如果最佳损失是有限的，则直接返回，避免重复计算
        if np.isfinite(best_loss):
            return as_tensor(best_loss)
        else:
            # 只有在最佳损失无效时才重新计算；闭包允许返回非 Tensor 数值
            return as_tensor(self._loss_value(closure()))
    
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
        
        # 联合优化所有参数组的参数：L-BFGS 需要单一扁平向量，
        # 逐组独立优化会让各组相互踩踏。超参数与历史状态使用第一个参数组
        all_params = [p for g in self.param_groups for p in g['params']]
        if not all_params:
            raise ValueError('LBFGS optimizer got an empty parameter list')
        ctx = self._prepare(all_params, closure)
        ctx = self._iterate(ctx, closure)
        return self._finalize(ctx, closure)

    def _prepare(self, params, closure):
        """读取超参数与组状态、准备向量转换函数，返回本次 step 的上下文。

        参数是所有参数组的并集：L-BFGS 需要在单一扁平向量上联合优化，
        因此超参数与历史状态统一取自第一个参数组。
        """
        group = self.param_groups[0]
        lr = group.get('lr', 1.0)
        tolerance_grad = group.get('tolerance_grad', 1e-5)
        tolerance_change = group.get('tolerance_change', 1e-9)
        history_size = group.get('history_size', 10)
        max_iter = group.get('max_iter', 20)
        max_eval = group.get('max_eval')
        if max_eval is None:
            max_eval = max_iter * 5 // 4 + 2
        strong_wolfe = group.get('line_search_fn') == 'strong_wolfe'

        # 初始化组状态（如果不存在）
        if 'old_dirs' not in group:
            self._reset_history(group)

        # 获取组状态
        old_dirs = group['old_dirs']
        old_stps = group['old_stps']
        old_ys = group.setdefault('old_ys', [])
        H_diag = group['H_diag']

        # 使用缓存的向量转换函数（缓存不写入 param_groups）；
        # arrlib 由参数实际设备决定，支持构造后再迁移到 CUDA
        cache = self._get_vector_funcs(group, params)
        funcs = cache['vector_funcs']
        arrlib = cache['arrlib']

        # 如果参数信息为空（初始化时没有梯度），先执行一次闭包获取梯度
        pending_loss = None
        func_evals = 0  # 本地计数器，独立于group状态
        if not cache['param_info']:
            loss = closure()
            func_evals += 1
            funcs = self._rebuild_cache(group, cache, params)
            arrlib = cache['arrlib']
            if not cache['param_info']:
                raise RuntimeError(
                    'LBFGS closure must compute gradients before optimizer.step()')
            # 参数未发生变化，这次评估结果可直接作为第 0 轮迭代的结果，避免重复调用闭包
            pending_loss = loss

        # 校验历史向量维度与当前参数匹配（加载到不同形状模型时尽早报错）
        self._validate_history(group, cache['total_params'])

        return {
            'group': group,
            'params': params,
            'lr': lr,
            'tolerance_grad': tolerance_grad,
            'tolerance_change': tolerance_change,
            'history_size': history_size,
            'max_iter': max_iter,
            'max_eval': max_eval,
            'strong_wolfe': strong_wolfe,
            'cache': cache,
            'funcs': funcs,
            'arrlib': arrlib,
            'pending_loss': pending_loss,
            'func_evals': func_evals,
            'last_step_norm': None,  # 本次 step 内最近一次被接受步的范数
            'old_dirs': old_dirs,
            'old_stps': old_stps,
            'old_ys': old_ys,
            'H_diag': H_diag,
            'best_loss': float('inf'),
            'best_params_vec': None,  # 使用向量形式保存最佳参数，减少克隆开销
            'last_eval_params_vec': None,  # 最近一次闭包评估对应的参数向量（用于判断梯度是否与参数同步）
        }

    def _iterate(self, ctx, closure):
        """执行 L-BFGS 迭代：闭包评估、方向计算、线搜索与历史更新。"""
        params = ctx['params']
        lr = ctx['lr']
        tolerance_grad = ctx['tolerance_grad']
        tolerance_change = ctx['tolerance_change']
        history_size = ctx['history_size']
        max_iter = ctx['max_iter']
        max_eval = ctx['max_eval']
        strong_wolfe = ctx['strong_wolfe']
        cache = ctx['cache']
        group = ctx['group']
        funcs = ctx['funcs']
        arrlib = ctx['arrlib']
        old_dirs = ctx['old_dirs']
        old_stps = ctx['old_stps']
        old_ys = ctx['old_ys']
        H_diag = ctx['H_diag']
        func_evals = ctx['func_evals']
        last_step_norm = ctx['last_step_norm']
        pending_loss = ctx['pending_loss']
        best_loss = ctx['best_loss']
        best_params_vec = ctx['best_params_vec']
        last_eval_params_vec = ctx['last_eval_params_vec']

        # 执行多次迭代，直到达到最大迭代次数或最大函数评估次数
        for iteration in range(max_iter):
            # 1. 保存当前参数向量（在closure调用之前，确保参数向量和梯度匹配）
            current_params_vec = funcs.p2v(params)

            # 2. 计算当前损失和梯度
            if pending_loss is not None:
                # 复用初始化阶段刚算过、且参数未变的评估结果
                loss = pending_loss
                pending_loss = None
            else:
                loss = closure()
                func_evals += 1
            loss_val = self._loss_value(loss)

            # 首次迭代：若出现缓存中未包含的参数（新增参数、替换参数对象、
            # 或此前未参与损失、现在才获得梯度），重建向量函数并清空历史
            if iteration == 0 and cache['param_info']:
                fresh_ids = {id(p) for p in params if p.grad is not None}
                cached_ids = {id(info['param']) for info in cache['param_info']}
                if not fresh_ids <= cached_ids:
                    self._reset_history(group)
                    old_dirs = group['old_dirs']
                    old_stps = group['old_stps']
                    old_ys = group['old_ys']
                    H_diag = 1.0
                    funcs = self._rebuild_cache(group, cache, params)
                    arrlib = cache['arrlib']
                    current_params_vec = funcs.p2v(params)

            # 3. 获取梯度向量并检查收敛条件
            grad_vector = funcs.g2v()
            last_eval_params_vec = current_params_vec
            # 使用最大绝对值范数，使 tolerance_grad 的含义不随参数规模漂移
            grad_norm = float(arrlib.max(arrlib.abs(grad_vector)))

            # 保存最佳状态（使用已获取的参数向量，避免重复转换）
            best_loss, best_params_vec = self._record_best(
                best_loss, best_params_vec, current_params_vec, loss_val, arrlib)

            # 检查收敛条件（参数变化只看本次 step 内已接受的步长）
            if self._check_convergence(grad_norm, tolerance_grad, last_step_norm, tolerance_change):
                break

            # 4. 计算L-BFGS搜索方向（双循环算法）
            search_dir, H_diag = self._compute_lbfgs_direction(
                grad_vector, old_dirs, old_stps, old_ys, len(old_dirs), H_diag, arrlib)

            # 5. 检查是否为下降方向
            gtd = arrlib.dot(grad_vector, search_dir)
            if gtd >= 0:
                # 非下降方向，使用最陡下降
                search_dir = -grad_vector.copy()
                H_diag = 1.0  # 重置Hessian对角线估计
                gtd = -arrlib.dot(grad_vector, grad_vector)

            # 6. 执行线搜索（受 max_eval 预算约束，避免超出评估上限）
            trial_budget = max_eval - func_evals
            if trial_budget <= 0:
                break
            search_result = self._line_search(
                closure, params, current_params_vec, search_dir, grad_vector,
                loss_val, gtd, lr, funcs.v2p, funcs.g2v,
                arrlib, self._C1, self._C2, self._MAX_BACKTRACKS,
                strong_wolfe=strong_wolfe, max_trials=trial_budget,
                grads_dot=funcs.gdot)

            # 更新函数评估计数
            func_evals += search_result.evals

            # 7. 处理线搜索结果
            if search_result.success:
                # 接受点立即纳入最佳状态，避免评估预算/迭代上限到期时丢弃已接受的步
                best_loss, best_params_vec = self._record_best(
                    best_loss, best_params_vec, search_result.params_vec,
                    search_result.loss, arrlib)

                # 线搜索成功，更新历史记录（曲率对按相对阈值校验）
                h_scale = self._update_history(
                    old_dirs, old_stps, old_ys, search_result.s,
                    search_result.y, history_size, arrlib)
                if h_scale is not None:
                    H_diag = h_scale
                last_step_norm = float(arrlib.max(arrlib.abs(search_result.s)))
                last_eval_params_vec = search_result.params_vec
            else:
                # 线搜索失败：回退到当前点，尝试一次最速下降安全步，并校验是否真的降低损失
                (fallback_accepted, fallback_loss, fallback_step_norm,
                 evaluated_vec, func_evals) = self._fallback_step(
                    closure, params, current_params_vec, grad_vector, loss_val,
                    funcs.v2p, arrlib, max_eval, func_evals)
                if evaluated_vec is not None:
                    last_eval_params_vec = evaluated_vec
                if not fallback_accepted:
                    # 预算耗尽或兜底步没有改善：结束本次优化，避免在坏点上继续迭代
                    break
                # 只接受确实降低损失的兜底步
                best_loss, best_params_vec = self._record_best(
                    best_loss, best_params_vec, evaluated_vec, fallback_loss, arrlib)
                last_step_norm = fallback_step_norm

            # 评估预算检查放在整轮结束之后：保证已接受的一步不会被丢弃，
            # 同时小预算配置（如 max_iter=1）也能真正执行一次迭代
            if func_evals >= max_eval:
                break

        ctx.update({
            'funcs': funcs,
            'arrlib': arrlib,
            'old_dirs': old_dirs,
            'old_stps': old_stps,
            'old_ys': old_ys,
            'H_diag': H_diag,
            'func_evals': func_evals,
            'last_step_norm': last_step_norm,
            'best_loss': best_loss,
            'best_params_vec': best_params_vec,
            'last_eval_params_vec': last_eval_params_vec,
        })
        return ctx

    def _finalize(self, ctx, closure):
        """写回组状态、恢复最佳参数，并返回本次 step 的损失。"""
        group = ctx['group']
        # 8. 更新组状态
        group.update({
            'H_diag': ctx['H_diag'],
            'old_dirs': ctx['old_dirs'],
            'old_stps': ctx['old_stps'],
            'old_ys': ctx['old_ys']
        })

        # 9. 恢复最佳参数；若回退点与最后一次评估点不同，重新评估以保证 p.grad 与参数同步
        best_params_vec = ctx['best_params_vec']
        last_eval_params_vec = ctx['last_eval_params_vec']
        arrlib = ctx['arrlib']
        self._restore_best_params(ctx['params'], best_params_vec, ctx['funcs'].v2p)
        if best_params_vec is not None and (
                last_eval_params_vec is None
                or not bool(arrlib.array_equal(best_params_vec, last_eval_params_vec))):
            return as_tensor(self._loss_value(closure()))
        return self._compute_final_loss(closure, ctx['best_loss'])


