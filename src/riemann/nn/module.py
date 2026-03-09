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
Neural Network Module Implementation for the Riemann Library

This module provides the foundation for building neural networks in the Riemann library.
It contains the core Module class and various neural network components that implement
common deep learning operations.

This file implements the following key components:
- RemovableHandle: Handle for managing hook registration and removal
- Parameter: Special tensor type for module parameters requiring gradients
- Module: Base class for all neural network modules with parameter management
- Linear: Fully connected linear layer
- Sequential: Container for sequential module execution
- ModuleList: List-based module container
- ModuleDict: Dictionary-based module container
- ParameterList: List-based parameter container
- ParameterDict: Dictionary-based parameter container
- Dropout: Regularization technique to prevent overfitting
- Dropout2d: 2D version of Dropout
- Dropout3d: 3D version of Dropout
- Flatten: Layer to flatten input tensors
- BatchNorm1d: 1D batch normalization layer
- BatchNorm2d: 2D batch normalization layer
- BatchNorm3d: 3D batch normalization layer
- LayerNorm: Layer normalization
- Embedding: Embedding layer for categorical data

All modules implement a PyTorch-compatible interface, making it easy to migrate code
between Riemann and PyTorch frameworks.
"""
from typing import Any, Dict
import copy
import numpy as np
from ..tensordef import *
from ..cuda import cp, Device
from .functional import *


class RemovableHandle:
    """
    可移除句柄类 (Removable Handle)
    
    用于管理钩子函数的注册和移除。当调用remove()方法时，
    会从对应的钩子字典中移除该钩子。
    
    Attributes:
        id (int): 钩子的唯一标识符
        hook_dict (dict): 存储钩子的字典引用
        _next_id (classmethod): 类变量，用于生成唯一ID
        
    Examples:
        >>> handle = module.register_forward_hook(my_hook)
        >>> handle.remove()  # 移除钩子
        >>> handle.id  # 访问钩子ID
    """
    _next_id = 0
    
    def __init__(self, hook_dict):
        """
        初始化句柄
        
        Args:
            hook_dict (dict): 存储钩子的字典
        """
        self.hook_dict = hook_dict
        self.id = RemovableHandle._next_id
        RemovableHandle._next_id += 1
    
    def remove(self):
        """
        移除钩子 (Remove Hook)
        
        从钩子字典中删除此句柄对应的钩子。
        如果钩子已被移除，则不做任何操作。
        """
        if self.id in self.hook_dict:
            del self.hook_dict[self.id]
    
    def __enter__(self):
        """支持上下文管理器协议"""
        return self
    
    def __exit__(self, *args):
        """退出上下文时自动移除钩子"""
        self.remove()


class Parameter(TN):
    """
    模型参数类 (Model Parameter Class)
    
    继承自张量类，用于表示神经网络中的可学习参数。
    参数是模型的权重、偏置等需要通过训练优化的张量。
    
    主要特性:
        - 自动设置requires_grad=True，参与梯度计算
        - 设置is_leaf=True，作为计算图的叶子节点
        - 支持模块系统的参数注册和管理
        - 提供与PyTorch兼容的接口
    
    Attributes:
        data: 参数的数值数据
        requires_grad (bool): 是否需要计算梯度，默认为True
        is_leaf (bool): 是否为计算图的叶子节点，默认为True
        
    Examples:
        >>> # 创建参数
        >>> weight = Parameter(rm.randn(10, 5))
        >>> bias = Parameter(rm.zeros(5))
        >>> 
        >>> # 在模块中使用
        >>> class Linear(Module):
        ...     def __init__(self, in_features, out_features):
        ...         super().__init__()
        ...         self.weight = Parameter(rm.randn(out_features, in_features))
        ...         self.bias = Parameter(rm.zeros(out_features))
        ...     
        ...     def forward(self, x):
        ...         return rm.matmul(x, self.weight.T) + self.bias
        >>> 
        >>> # 参数会自动注册到模块
        >>> layer = Linear(10, 5)
        >>> print(list(layer.parameters()))  # 包含weight和bias
        
    Note:
        - 通常不需要直接创建，模块系统会自动注册
        - requires_grad属性控制是否参与梯度计算
        - is_leaf属性影响反向传播的行为
        - 继承自张量类，支持所有张量操作
    """
    def __init__(self, data: Optional[TN] = None, requires_grad:bool=True):
        """
        初始化参数实例 (Initialize Parameter Instance)
        
        从现有张量创建参数，设置梯度计算相关属性。
        
        Args:
            data (TN, optional): 输入张量，作为参数的初始值。如果为None，则创建空参数
            requires_grad (bool): 是否需要计算梯度，默认为True
            
        Raises:
            TypeError: 当输入不是张量类型且不为None时
            
        Examples:
            >>> # 从随机张量创建参数
            >>> weight = Parameter(rm.randn(10, 5))
            >>> 
            >>> # 从常量张量创建参数
            >>> bias = Parameter(rm.zeros(5))
            >>> 
            >>> # 创建空参数
            >>> empty_param = Parameter()
            >>> 
            >>> # 参数会自动设置梯度计算
            >>> weight.requires_grad  # True
            >>> weight.is_leaf        # True
        """
        super().__init__()
        # 处理空参数情况
        if data is None:
            self.data = None # type:ignore
        else:
            # 确保输入是张量
            if not isinstance(data, TN):
                raise TypeError(f"Expected a TN tensor, got {type(data)}")
            self.data = data.data
            
        self.requires_grad = requires_grad
        
    def __str__(self):
        """
        参数的字符串表示 (Parameter String Representation)
        
        特殊处理data为None的情况，其他情况复用父类的__str__方法。
        
        Returns:
            str: 参数的字符串表示
        """
        if self.data is None:
            return 'Parameter containing:\nNone'
        else:
            # 复用父类的__str__方法
            return super().__str__()
        
    def __repr__(self):
        """
        参数的字符串表示 (Parameter String Representation)
        
        返回参数的描述性字符串，显示参数包含的数据。
        与PyTorch的Parameter.__repr__保持一致的格式。
        
        Returns:
            str: 参数的字符串表示
            
        Examples:
            >>> param = Parameter(rm.randn(2, 3))
            >>> print(param)
            Parameter containing:
            tensor([[ 0.1234, -0.5678,  0.9012],
                    [ 0.3456, -0.7890,  0.2345]], requires_grad=True)
            
            >>> empty_param = Parameter()
            >>> print(empty_param)
            Parameter containing:
            None
        """
        if self.data is None:
            return 'Parameter containing:\nNone'
        else:
            # 直接复用父类的__repr__方法，TN类已支持requires_grad属性的显示
            tensor_str = super().__repr__()
            return f'Parameter containing:\n{tensor_str}'
    
    def __reduce__(self):
        """
        实现对象的pickle序列化支持，兼容PyTorch格式
        
        返回:
            tuple: 包含构造函数和参数的元组
        """
        from ..serialization import _parameter_constructor
        
        if self.data is None:
            return (_parameter_constructor, (None, self.requires_grad))
        else:
            data = self.data
            if cp and isinstance(data, cp.ndarray):
                # 使用 .get() 方法显式转换 cupy 数组为 numpy 数组
                data = data.get()
            
            return (_parameter_constructor, (
                data,  # 直接序列化底层数据
                self.shape,
                str(self.dtype),
                str(self.device),
                self.requires_grad
            ))
  
#end of class Parameter

class Module:
    """
    神经网络模块基类 (Neural Network Module Base Class)
    
    所有神经网络模块的基类，提供了模块管理、参数管理、缓冲区管理等核心功能。
    设计与PyTorch的nn.Module兼容，便于代码迁移。
    
    Module是构建神经网络的基础组件，支持：
    - 参数和缓冲区的自动注册和管理
    - 子模块的嵌套和层次化组织
    - 训练/评估模式切换
    - 状态字典的保存和加载
    - 模块的复制和克隆
    - 递归遍历所有子模块和参数
    
    核心属性:
        _modules (dict): 存储子模块的字典
        _parameters (dict): 存储参数的字典  
        _buffers (dict): 存储缓冲区的字典
        training (bool): 训练/评估模式标志，True为训练模式
    
    基本用法:
        >>> class MyNet(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.linear1 = Linear(784, 256)
        ...         self.linear2 = Linear(256, 10)
        ...     
        ...     def forward(self, x):
        ...         x = self.linear1(x)
        ...         x = self.linear2(x)
        ...         return x
        >>> 
        >>> net = MyNet()
        >>> print(list(net.parameters()))  # 访问所有参数
        >>> print(net.state_dict())        # 获取状态字典
        >>> net.eval()                     # 切换到评估模式
    
    Note:
        - 子类必须实现forward()方法定义前向传播逻辑
        - 使用self.xxx = Module()自动注册为子模块
        - 使用self.xxx = Parameter()自动注册为参数
        - 使用register_buffer()注册缓冲区
        - 支持通过属性访问直接访问参数、缓冲区和子模块
    """
    def __init__(self):
        """
        初始化模块实例
        
        创建新的模块实例，初始化核心数据结构：
        - _modules: 存储子模块的字典
        - _parameters: 存储参数的字典
        - _buffers: 存储缓冲区的字典
        - _forward_pre_hooks: 存储前向传播前钩子的字典
        - _forward_hooks: 存储前向传播钩子的字典
        - _backward_pre_hooks: 存储反向传播前钩子的字典
        - _backward_hooks: 存储反向传播钩子的字典
        - training: 训练模式标志，默认为True
        
        所有子类都应该调用super().__init__()来确保正确的初始化。
        """
        
        self._modules = {}
        self._parameters = {}
        self._buffers = {}
        self._forward_pre_hooks = {}
        self._forward_hooks = {}
        self._backward_pre_hooks = {}
        self._backward_hooks = {}
        self.training = True  # 训练/评估模式标志      
    
    def to(self, *args, **kwargs):
        """
        将模块的所有参数和缓冲区移动到指定设备和/或转换为指定数据类型
        
        Args:
            device: 目标设备，可以是字符串（如'cpu'、'cuda'）、整数（设备ID）或Device对象
            dtype: 目标数据类型，可以是Python类型、NumPy dtype、字符串或Riemann dtype
            也可以传入另一个张量，复制其dtype和device
            
        Returns:
            Module: 转换后的模块本身（原地操作）
            
        Examples:
            >>> model = MyModule()
            >>> model.to('cuda')  # 移动到CUDA设备
            >>> model.to('cpu')  # 移动到CPU
            >>> model.to(float32)  # 转换为float32类型
            >>> model.to('cuda', float16)  # 移动到CUDA并转换为float16
            >>> model.to(device='cuda', dtype=float16)  # 使用关键字参数
            >>> model.to(other_tensor)  # 从另一个张量复制dtype和device
        """
        # 移动所有参数
        for name, param in self._parameters.items():
            if param is not None:
                # 直接透传参数给 param.to()
                new_param = param.to(*args, **kwargs)
                if new_param is not param:
                    # 跨设备或跨类型转换时要清除计算图依赖
                    new_param = new_param.detach_()
                    new_param.requires_grad = param.requires_grad
                    # 更新实例属性引用
                    setattr(self, name, new_param)
        
        # 移动所有缓冲区
        for name, buffer in self._buffers.items():
            if buffer is not None:
                # 直接透传参数给 buffer.to()
                new_buffer = buffer.to(*args, **kwargs)
                if new_buffer is not buffer:
                    # 清除计算图依赖
                    new_buffer = new_buffer.detach_()
                    new_buffer.requires_grad = buffer.requires_grad
                    # 更新实例属性引用
                    setattr(self, name, new_buffer)
        
        # 递归移动所有子模块
        for name, module in self._modules.items():
            module.to(*args, **kwargs)
        
        return self

    def cuda(self, device=None):
        """
        将模块的所有参数和缓冲区移动到CUDA设备
        
        参数:
            device: 目标CUDA设备，可以是整数（设备ID）、字符串（如'cuda:0'）或Device对象
                   如果为None，则使用当前默认CUDA设备
        
        返回:
            Module: 移动到CUDA设备后的模块本身（原地操作）
        
        Examples:
            >>> model = MyModule()
            >>> model.cuda()  # 移动到默认CUDA设备
            >>> model.cuda(0)  # 移动到cuda:0
            >>> model.cuda('cuda:1')  # 移动到cuda:1
        """
        if device == 'cpu':
            raise ValueError("cuda() method is not supported for 'cpu'")
        
        if device is None:
            device = 'cuda'
        device = Device(device)
        return self.to(device)
    
    def cpu(self):
        """
        将模块的所有参数和缓冲区移动到CPU设备
        
        返回:
            Module: 移动到CPU设备后的模块本身（原地操作）
        
        Examples:
            >>> model = MyModule()
            >>> model.cuda()  # 移动到CUDA
            >>> model.cpu()  # 移回CPU
        """
        from ..cuda import Device
        
        return self.to(Device('cpu'))
      

    def _get_name(self):
        """
        获取模块类名 (Get Module Class Name)
        
        返回模块的类名字符串，用于在__repr__等显示方法中标识模块类型。
        
        Returns:
            str: 模块的类名
            
        Examples:
            >>> class MyModule(Module):
            ...     pass
            >>> m = MyModule()
            >>> print(m._get_name())  # 输出: 'MyModule'
        """
        return self.__class__.__name__

    def register_parameter(self, name, param):
        """
        注册参数到模块 (Register Parameter to Module)
        
        将一个参数添加到模块的参数字典中。参数是模型的可学习参数，
        会参与梯度计算和优化器更新。
        
        Args:
            name (str): 参数名称，用于后续访问和状态字典中的键
            param (Parameter or None): 要注册的参数对象，None表示删除该参数
            
        Raises:
            TypeError: 当param不是Parameter类型且不为None时
            
        Examples:
            >>> class MyModule(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.weight = Parameter(rm.randn(10, 5))
            ...         self.register_parameter('bias', Parameter(rm.zeros(5)))
            ...
            >>> m = MyModule()
            >>> print(list(m.named_parameters()))
            [('weight', Parameter containing:...), ('bias', Parameter containing:...)]
            
        Note:
            - 通常不需要直接调用此方法，通过self.xxx = Parameter()会自动注册
            - 参数会被包含在parameters()和named_parameters()的返回中
            - 参数会包含在state_dict()和load_state_dict()中
            - 参数的requires_grad默认为True，会参与梯度计算
        """
        if param is not None and not isinstance(param, Parameter):
            raise TypeError(f"Cannot assign '{type(param)}' as parameter '{name}'")
        self._parameters[name] = param

    def register_buffer(self, name: str, a_tensor: Any) -> None:
        """注册缓冲区 (Register Buffer)
        
        向模块添加一个持久化的缓冲区。缓冲区是模型状态的一部分，
        但不是参数（即不参与梯度计算）。常用于存储运行时统计信息，
        如BatchNorm中的均值和方差。
        
        缓冲区可以通过buffers()和named_buffers()方法访问，
        并且会包含在state_dict()和load_state_dict()中。
        
        Args:
            name (str): 缓冲区名称
            a_tensor (Tensor): 要注册的张量。如果为None，则删除该缓冲区
            
        Examples::
        
            >>> class MyModule(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         # 注册运行时统计信息
            ...         self.register_buffer('running_mean', rm.zeros(10))
            ...         self.register_buffer('running_var', rm.ones(10))
            ...         # 注册固定常量
            ...         self.register_buffer('scale', rm.tensor([1.0, 2.0, 3.0]))
            ...
            >>> m = MyModule()
            >>> print(list(m.named_buffers()))
            [('running_mean', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])), 
             ('running_var', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])),
             ('scale', tensor([1., 2., 3.]))]
        
        Note:
            缓冲区的特点和用途：
            - 是模型状态的一部分，会随模型一起保存/加载
            - 不参与梯度计算，requires_grad通常为False
            - 可以通过属性访问，如self.running_mean
            - 常用于BatchNorm、LayerNorm等层的运行时统计
            - 也可用于存储固定的常量或超参数
            - 与参数不同，缓冲区不会被优化器更新
            - 在模型评估时保持稳定，提供一致性
            - 支持None值，用于删除已注册的缓冲区
        """
        if a_tensor is not None and not isinstance(a_tensor, TN):
            raise TypeError(f"Cannot assign '{type(a_tensor)}' as buffer '{name}'. Expected TN tensor or None.")
        self._buffers[name] = a_tensor

    def register_parameters_batch(self, **parameters) -> None:
        """批量注册参数 (Batch Register Parameters)
        
        一次性注册多个参数，提高性能。
        
        Args:
            **parameters: 参数名和参数值的键值对
            
        Examples:
            >>> class MyModule(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.register_parameters_batch(
            ...             weight=Parameter(rm.randn(10, 5)),
            ...             bias=Parameter(rm.zeros(5)),
            ...             scale=Parameter(rm.ones(5))
            ...         )
        """
        for name, param in parameters.items():
            if not isinstance(param, Parameter):
                raise TypeError(f"Cannot assign '{type(param)}' as parameter '{name}'. Expected Parameter.")
            self._parameters[name] = param

    def register_buffers_batch(self, **buffers) -> None:
        """批量注册缓冲区 (Batch Register Buffers)
        
        一次性注册多个缓冲区，提高性能。
        
        Args:
            **buffers: 缓冲区名和张量值的键值对
            
        Examples:
            >>> class MyModule(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.register_buffers_batch(
            ...             running_mean=rm.zeros(10),
            ...             running_var=rm.ones(10),
            ...             scale=rm.tensor([1.0, 2.0, 3.0])
            ...         )
        """
        for name, tensor_obj in buffers.items():
            if tensor_obj is not None and not isinstance(tensor_obj, TN):
                raise TypeError(f"Cannot assign '{type(tensor_obj)}' as buffer '{name}'. Expected TN tensor or None.")
            self._buffers[name] = tensor_obj

    def clear_cache(self) -> None:
        """清除属性访问缓存 (Clear Attribute Cache)
        
        清除所有缓存的属性引用，释放内存。
        在大量修改模块属性后调用此方法。
        """
        self._attr_cache.clear()

    def enable_cache(self, enabled: bool = True) -> None:
        """启用或禁用属性缓存 (Enable/Disable Attribute Cache)
        
        Args:
            enabled (bool): 是否启用缓存，默认为True
        """
        self._cache_enabled = enabled
        if not enabled:
            self.clear_cache()

    def __setattr__(self, name: str, value: Any) -> None:
        """
        设置属性值，自动注册模块、参数和缓冲区 (Set Attribute with Auto-Registration)
        
        重写属性设置方法，实现模块、参数和缓冲区的自动注册：
        - Module实例 -> 自动注册为子模块
        - Parameter实例 -> 自动注册为参数
        - Tensor实例 -> 自动注册为缓冲区（除training属性外）
        - 其他类型 -> 直接设置为普通属性
        
        Args:
            name (str): 属性名称
            value (Any): 属性值，可以是Module、Parameter、Tensor或其他类型
            
        Examples:
            >>> class MyModule(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         # 自动注册子模块
            ...         self.linear = Linear(10, 5)
            ...         # 自动注册参数
            ...         self.weight = Parameter(rm.randn(5, 3))
            ...         # 自动注册缓冲区
            ...         self.scale = rm.tensor([1.0, 2.0])
            ...         # 普通属性
            ...         self.num_layers = 2
            ...
            >>> m = MyModule()
            >>> print(list(m.modules()))      # 包含linear子模块
            >>> print(list(m.parameters()))   # 包含weight参数
            >>> print(list(m.buffers()))      # 包含scale缓冲区
            
        Note:
            - 这是模块系统的核心机制，实现自动注册功能
            - training属性特殊处理，不作为缓冲区注册
            - 避免在__init__外频繁调用，可能影响性能
        """
        # 特殊属性直接设置
        if name in ['_modules', '_parameters', '_buffers', 'training']:
            super().__setattr__(name, value)
            return
        
        # 处理模块注册
        if isinstance(value, Module):
            self._modules[name] = value
        # 处理参数注册（包括Parameter子类）
        elif isinstance(value, Parameter):
            self._parameters[name] = value
        # 处理缓冲区注册（张量但非参数）
        elif isinstance(value, TN) and name != 'training':
            self._buffers[name] = value
        else:
            # 其他属性使用父类方法直接设置
            super().__setattr__(name, value)
    
    def __getattr__(self, name: str) -> Any:
        """
        获取属性值，支持参数、缓冲区和子模块访问 (Get Attribute with Auto-Lookup)
        
        重写属性获取方法，实现从参数、缓冲区和子模块中自动查找：
        1. 首先从参数字典中查找
        2. 然后从缓冲区字典中查找  
        3. 最后从子模块字典中查找
        4. 都找不到则抛出AttributeError
        
        Args:
            name (str): 要获取的属性名称
            
        Returns:
            Any: 找到的参数、缓冲区或子模块
            
        Raises:
            AttributeError: 当属性在所有字典中都找不到时
            
        Examples:
            >>> class MyModule(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.weight = Parameter(rm.randn(10, 5))
            ...         self.register_buffer('scale', rm.tensor([2.0]))
            ...         self.linear = Linear(5, 3)
            ...
            >>> m = MyModule()
            >>> w = m.weight      # 从参数字典获取
            >>> s = m.scale       # 从缓冲区字典获取
            >>> l = m.linear      # 从子模块字典获取
            
        Note:
            - 与__setattr__配合实现透明的属性访问
            - 查找顺序：参数 -> 缓冲区 -> 子模块
            - 支持通过属性名直接访问注册的组件
        """
        # 使用object.__getattribute__直接访问基础属性，避免递归调用
        try:
            _parameters = object.__getattribute__(self, '_parameters')
            if name in _parameters:
                return _parameters[name]
            
            _buffers = object.__getattribute__(self, '_buffers')
            if name in _buffers:
                return _buffers[name]
            
            _modules = object.__getattribute__(self, '_modules')
            if name in _modules:
                return _modules[name]
        except AttributeError:
            # 如果基础属性不存在，继续抛出原始属性错误
            pass
        
        # 触发标准属性错误
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __delattr__(self, name: str):
        """删除属性方法，支持参数、缓冲区和子模块的删除"""
        # 使用object.__getattribute__直接访问基础属性，避免递归调用
        try:
            # 从参数字典删除
            _parameters = object.__getattribute__(self, '_parameters')
            if name in _parameters:
                del _parameters[name]
                return
            
            # 从缓冲区字典删除
            _buffers = object.__getattribute__(self, '_buffers')
            if name in _buffers:
                del _buffers[name]
                return
            
            # 从模块字典删除
            _modules = object.__getattribute__(self, '_modules')
            if name in _modules:
                del _modules[name]
                return
        except AttributeError:
            # 如果基础属性不存在，继续使用父类方法
            pass
        
        # 其他属性使用父类方法删除
        super().__delattr__(name)
    
    def add_module(self, name, module):
        """显式添加子模块""" 
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(type(module)))
        self._modules[name] = module

    def forward(self, *args, **kwargs):
        """
        前向传播方法 (Forward Pass)
        
        定义模块的前向传播逻辑，子类必须实现此方法。
        这是模块的核心功能，接收输入数据并产生输出。
        
        Args:
            *args: 位置参数，根据具体模块定义
            **kwargs: 关键字参数，根据具体模块定义
            
        Returns:
            输出数据，类型和形状由具体模块定义
            
        Raises:
            NotImplementedError: 基类中未实现，子类必须重写
            
        Examples:
            >>> class Linear(Module):
            ...     def __init__(self, in_features, out_features):
            ...         super().__init__()
            ...         self.weight = Parameter(rm.randn(out_features, in_features))
            ...         self.bias = Parameter(rm.zeros(out_features))
            ...     
            ...     def forward(self, x):
            ...         return rm.matmul(x, self.weight.T) + self.bias
            ...
            >>> layer = Linear(10, 5)
            >>> output = layer(input_data)  # 调用forward方法
            
        Note:
            - 子类必须实现此方法
            - 实际调用时通过__call__方法，可能包含钩子等额外逻辑
            - 应该支持批量输入，遵循深度学习惯例
        """ 
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """
        模块调用方法 (Module Call)
        
        使模块实例可调用，实现module(x)的调用方式。
        这是用户使用模块的主要接口，内部调用forward方法。
        同时处理前向传播前钩子和前向传播钩子的调用。
        
        Args:
            *args: 位置参数，传递给forward方法
            **kwargs: 关键字参数，传递给forward方法
            
        Returns:
            forward方法的返回值
            
        Examples:
            >>> layer = Linear(10, 5)
            >>> output = layer(input_data)  # 等价于layer.forward(input_data)
            >>> output = layer(input_data, training=True)  # 传递额外参数
            
        Note:
            - 这是模块的标准调用方式
            - 在forward前后会调用注册的钩子函数
            - 提供与PyTorch兼容的调用接口
        """
        # 调用前向传播前钩子
        for hook in self._forward_pre_hooks.values():
            result = hook(self, args)
            if result is not None:
                if not isinstance(result, tuple):
                    result = (result,)
                args = result
        
        # 执行前向传播
        output = self.forward(*args, **kwargs)
        
        # 调用前向传播钩子
        for hook in self._forward_hooks.values():
            hook_result = hook(self, args, output)
            if hook_result is not None:
                output = hook_result
        
        # 设置输出张量的模块引用（用于反向传播钩子）
        # 只有当存在backward钩子时才设置，避免不必要的开销
        if output is not None and (self._backward_hooks or self._backward_pre_hooks):
            if isinstance(output, TN):
                output._module = self
            elif isinstance(output, (tuple, list)):
                for out in output:
                    if isinstance(out, TN):
                        out._module = self
        
        return output

    def parameters(self, recurse=True):
        """
        返回模块参数的迭代器 (Parameters Iterator)
        
        返回模块所有参数的迭代器。参数是可学习的模型参数，
        会参与梯度计算和优化器更新。
        
        Args:
            recurse (bool, optional): 是否递归返回子模块的参数。默认值: True
            
        Yields:
            Parameter: 模块中的参数对象
            
        Examples:
            >>> class MyNet(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.linear1 = Linear(10, 5)
            ...         self.linear2 = Linear(5, 1)
            ...         self.bias = Parameter(rm.zeros(1))
            ...
            >>> net = MyNet()
            >>> for param in net.parameters():
            ...     print(param.shape)  # 打印所有参数的形状
            >>> 
            >>> # 获取参数数量
            >>> param_count = sum(1 for _ in net.parameters())
            >>> 
            >>> # 仅当前模块的参数（不递归）
            >>> for param in net.linear1.parameters(recurse=False):
            ...     print(param.shape)
            
        Note:
            - 包含当前模块和所有子模块的参数（当recurse=True时）
            - 参数的requires_grad属性决定是否参与梯度计算
            - 常用于优化器的参数列表构建
            - 返回的是生成器，支持惰性求值
        """
        for name, param in self._parameters.items():
            if param is not None:
                yield param
        if recurse:
            for module in self._modules.values():
                yield from module.parameters()

    def named_parameters(self, prefix='', recurse=True):
        """
        返回带名称的模块参数迭代器 (Named Parameters Iterator)
        
        返回模块所有参数及其名称的迭代器。参数名称包含层级路径，
        便于识别参数在模型中的位置。
        
        Args:
            prefix (str, optional): 参数名前缀。默认值: ''
            recurse (bool, optional): 是否递归返回子模块的参数。默认值: True
            
        Yields:
            tuple: (参数名称, 参数对象) 的元组
            
        Examples:
            >>> class MyNet(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.linear1 = Linear(10, 5)
            ...         self.linear2 = Linear(5, 1)
            ...
            >>> net = MyNet()
            >>> for name, param in net.named_parameters():
            ...     print(f"{name}: {param.shape}")
            ...     # 输出: linear1.weight: torch.Size([5, 10])
            ...     #      linear1.bias: torch.Size([5])
            ...     #      linear2.weight: torch.Size([1, 5])
            ...     #      linear2.bias: torch.Size([1])
            >>> 
            >>> # 仅特定前缀的参数
            >>> for name, param in net.named_parameters(prefix='linear1.'):
            ...     print(f"{name}: {param.shape}")
            >>> 
            >>> # 获取参数字典
            >>> param_dict = dict(net.named_parameters())
            
        Note:
            - 参数名称使用点号分隔的层级结构
            - 常用于模型检查点保存和加载
            - 支持前缀过滤，便于定位特定模块的参数
            - 返回的是生成器，支持惰性求值
        """
        for name, param in self._parameters.items():
            if param is not None:
                yield f"{prefix}{name}", param
        if recurse:
            for module_name, module in self._modules.items():
                sub_prefix = f"{prefix}{module_name}."
                yield from module.named_parameters(sub_prefix, recurse)

    def buffers(self, recurse: bool = True):
        """返回模块缓冲区的迭代器 (Buffers Iterator)
        
        返回模块所有缓冲区的迭代器。缓冲区是不参与梯度计算的持久化张量，
        常用于存储运行时统计信息（如BatchNorm中的均值和方差）。
        
        Args:
            recurse (bool, optional): 是否递归返回子模块的缓冲区。默认值: True
            
        Yields:
            Tensor: 模块中的缓冲区张量
            
        Examples::
        
            >>> class MyModule(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.register_buffer('running_mean', rm.zeros(5))
            ...         self.register_buffer('running_var', rm.ones(5))
            ...
            >>> m = MyModule()
            >>> buffers = list(m.buffers())
            >>> print(len(buffers))  # 2
            >>> print(buffers[0].shape)  # (5,)
            
        Note:
            buffers()方法的特点：
            - 只返回缓冲区张量，不包含名称信息
            - 缓冲区不参与梯度计算，requires_grad通常为False
            - 递归模式下会包含所有子模块的缓冲区
            - 常用于需要访问所有持久化状态但不需要名称的场景
            - 与named_buffers()相比，更简洁但信息较少
            - 在模型状态管理、设备转移等操作中很有用
        """
        for name, buffer in self._buffers.items():
            if buffer is not None:
                yield buffer
        if recurse:
            for module in self._modules.values():
                yield from module.buffers(recurse)

    def named_buffers(self, prefix: str = '', recurse: bool = True):
        """返回带名称的模块缓冲区迭代器 (Named Buffers Iterator)
        
        返回模块所有缓冲区的名称和缓冲区的迭代器。缓冲区是不参与梯度计算的
        持久化张量，常用于存储运行时统计信息。
        
        Args:
            prefix (str, optional): 子模块名称前缀。默认值: ''
            recurse (bool, optional): 是否递归返回子模块的缓冲区。默认值: True
            
        Yields:
            (str, Tensor): 缓冲区名称和缓冲区张量的元组
            
        Examples::
        
            >>> class MyModule(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.register_buffer('running_mean', rm.zeros(3))
            ...         self.register_buffer('running_var', rm.ones(3))
            ...
            >>> m = MyModule()
            >>> for name, buffer in m.named_buffers():
            ...     print(f"{name}: {buffer.shape}")
            running_mean: (3,)
            running_var: (3,)
            
        Note:
            named_buffers()方法的特点：
            - 返回缓冲区名称和张量的键值对
            - 支持层级命名，子模块缓冲区会加上前缀
            - 递归模式下可以访问整个模型的缓冲区树
            - 常用于调试、日志记录和状态检查
            - 在模型序列化/反序列化中很重要
            - 与buffers()相比提供了更多的上下文信息
            - 前缀机制支持复杂的模块嵌套结构
        """
        for name, buffer in self._buffers.items():
            if buffer is not None:
                yield f"{prefix}{name}", buffer
        if recurse:
            for module_name, module in self._modules.items():
                sub_prefix = f"{prefix}{module_name}."
                yield from module.named_buffers(sub_prefix, recurse)

    def children(self):
        """生成式返回直接子模块""" 
        for name, module in self._modules.items():
            yield module

    def modules(self):
        """递归返回所有子模块（包括自身）""" 
        yield self
        for module in self._modules.values():
            yield from module.modules()

    def named_modules(self, prefix='', recurse=True):
        """带层级的模块名生成器"""
        # 首先返回自身模块（空键名）
        if prefix == '':
            yield '', self
        for name, module in self._modules.items():
            yield f"{prefix}{name}", module
        if recurse:
            for module_name, module in self._modules.items():
                sub_prefix = f"{prefix}{module_name}."
                yield from module.named_modules(sub_prefix, recurse)

    def register_forward_pre_hook(self, hook):
        """
        注册前向传播前钩子 (Register Forward Pre-Hook)
        
        注册一个钩子函数，该钩子会在模块的forward方法被调用之前执行。
        钩子函数应该具有以下签名：
            hook(module, input) -> None or modified input
        
        Args:
            hook (callable): 钩子函数，接收模块和输入参数，可以返回修改后的输入
            
        Returns:
            RemovableHandle: 一个可调用对象，调用它可以移除这个钩子
            
        Examples:
            >>> def my_hook(module, input):
            ...     print(f"Forward pre-hook called for {module._get_name()}")
            ...     # 可以修改输入
            ...     return input
            >>> 
            >>> layer = Linear(10, 5)
            >>> handle = layer.register_forward_pre_hook(my_hook)
            >>> output = layer(input_data)  # 会调用 my_hook
            >>> handle.remove()  # 移除钩子
            
        Note:
            - 钩子函数可以返回修改后的输入来改变forward的输入
            - 如果返回None，则使用原始输入
            - 多个钩子按注册顺序执行
        """
        handle = RemovableHandle(self._forward_pre_hooks)
        self._forward_pre_hooks[handle.id] = hook
        return handle

    def register_forward_hook(self, hook):
        """
        注册前向传播钩子 (Register Forward Hook)
        
        注册一个钩子函数，该钩子会在模块的forward方法被调用之后执行。
        钩子函数应该具有以下签名：
            hook(module, input, output) -> None or modified output
        
        Args:
            hook (callable): 钩子函数，接收模块、输入和输出，可以返回修改后的输出
            
        Returns:
            RemovableHandle: 一个可调用对象，调用它可以移除这个钩子
            
        Examples:
            >>> def my_hook(module, input, output):
            ...     print(f"Forward hook called for {module._get_name()}")
            ...     print(f"Output shape: {output.shape}")
            ...     # 可以修改输出
            ...     return output * 2
            >>> 
            >>> layer = Linear(10, 5)
            >>> handle = layer.register_forward_hook(my_hook)
            >>> output = layer(input_data)  # 会调用 my_hook
            >>> handle.remove()  # 移除钩子
            
        Note:
            - 钩子可以返回修改后的输出来改变模块的输出
            - 如果返回None，则使用原始输出
            - 多个钩子按注册顺序执行
        """
        handle = RemovableHandle(self._forward_hooks)
        self._forward_hooks[handle.id] = hook
        return handle

    def register_full_backward_pre_hook(self, hook):
        """
        注册反向传播前钩子 (Register Full Backward Pre-Hook)
        
        注册一个钩子函数，该钩子会在模块的反向传播开始之前执行。
        钩子函数应该具有以下签名：
            hook(module, grad_output) -> None or modified grad_output
        
        Args:
            hook (callable): 钩子函数，接收模块和输出梯度，可以返回修改后的梯度
            
        Returns:
            RemovableHandle: 一个可调用对象，调用它可以移除这个钩子
            
        Examples:
            >>> def my_hook(module, grad_output):
            ...     print(f"Backward pre-hook called for {module._get_name()}")
            ...     # 可以修改梯度
            ...     return grad_output
            >>> 
            >>> layer = Linear(10, 5)
            >>> handle = layer.register_full_backward_pre_hook(my_hook)
            >>> output = layer(input_data)
            >>> output.sum().backward()  # 会调用 my_hook
            >>> handle.remove()  # 移除钩子
            
        Note:
            - 钩子在反向传播开始前被调用
            - 可以修改传递给模块的梯度
            - grad_output是一个元组，包含相对于输出的梯度
        """
        handle = RemovableHandle(self._backward_pre_hooks)
        self._backward_pre_hooks[handle.id] = hook
        return handle

    def register_full_backward_hook(self, hook):
        """
        注册反向传播钩子 (Register Full Backward Hook)
        
        注册一个钩子函数，该钩子会在模块的反向传播完成后执行。
        钩子函数应该具有以下签名：
            hook(module, grad_input, grad_output) -> None or modified grad_input
        
        Args:
            hook (callable): 钩子函数，接收模块、输入梯度和输出梯度
            
        Returns:
            RemovableHandle: 一个可调用对象，调用它可以移除这个钩子
            
        Examples:
            >>> def my_hook(module, grad_input, grad_output):
            ...     print(f"Backward hook called for {module._get_name()}")
            ...     print(f"grad_input: {grad_input}")
            ...     print(f"grad_output: {grad_output}")
            ...     # 可以修改输入梯度
            ...     return grad_input
            >>> 
            >>> layer = Linear(10, 5)
            >>> handle = layer.register_full_backward_hook(my_hook)
            >>> output = layer(input_data)
            >>> output.sum().backward()  # 会调用 my_hook
            >>> handle.remove()  # 移除钩子
            
        Note:
            - 钩子在反向传播完成后被调用
            - grad_input和grad_output都是元组
            - 可以返回修改后的grad_input来影响梯度传播
        """
        handle = RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle

    def train(self, mode=True):
        """
        设置模块为训练模式 (Set Training Mode)
        
        将模块及其所有子模块设置为训练模式。在训练模式下，
        某些层（如Dropout、BatchNorm等）会表现出不同的行为。
        
        Args:
            mode (bool, optional): True设置为训练模式，False设置为评估模式。默认值: True
            
        Examples:
            >>> model = MyNet()
            >>> model.train()  # 设置为训练模式
            >>> model.train(False)  # 设置为评估模式
            >>> model.training  # 查看当前模式
            False
            >>> 
            >>> # 训练循环中的典型用法
            >>> model.train()
            >>> for batch in dataloader:
            ...     # 训练代码
            ...     pass
            >>> 
            >>> # 评估阶段
            >>> model.eval()
            >>> with torch.no_grad():
            ...     # 评估代码
            ...     pass
            
        Note:
            - 会递归设置所有子模块的训练模式
            - 影响Dropout、BatchNorm等层的 behavior
            - training属性保存当前模式状态
            - eval()方法是train(False)的简写
        """ 
        self.training = mode
        for module in self.children():
            module.train(mode)

    def eval(self):
        """
        设置模块为评估模式 (Set Evaluation Mode)
        
        将模块及其所有子模块设置为评估模式。在评估模式下，
        某些层会关闭随机性，确保输出的一致性。
        
        Examples:
            >>> model = MyNet()
            >>> model.eval()  # 设置为评估模式
            >>> model.training  # 查看当前模式
            False
            >>> 
            >>> # 评估时的标准用法
            >>> model.eval()
            >>> with torch.no_grad():
            ...     predictions = model(data)
            ...     # 评估指标计算
            ...     pass
            
        Note:
            - 等价于train(False)
            - Dropout会停止随机丢弃神经元
            - BatchNorm会使用运行统计而非批次统计
            - 常与torch.no_grad()配合使用
            - 返回self支持链式调用
        """ 
        self.train(False)
        return self
    
    def zero_grad(self, set_to_none: bool = False):
        """清空所有参数梯度，防止梯度累积 (Zero Gradients)
        
        将模块中所有参数的梯度设置为0或None。这在每次训练迭代开始前调用，
        以防止前一次迭代的梯度累积到当前迭代中。
        
        Args:
            set_to_none (bool, optional): 是否将梯度设置为None而不是0。
                                         设置为None可以节省内存，但会影响
                                         某些优化器的行为。默认值: False
                                         
        Examples::
        
            >>> model = MyModel()
            >>> optimizer = SGD(model.parameters(), lr=0.01)
            >>> 
            >>> # 训练循环
            >>> for data, target in dataloader:
            ...     # 清空梯度
            ...     model.zero_grad()
            ...     # 前向传播
            ...     output = model(data)
            ...     # 计算损失
            ...     loss = criterion(output, target)
            ...     # 反向传播
            ...     loss.backward()
            ...     # 更新参数
            ...     optimizer.step()
            ... 
            >>> # 内存优化版本
            >>> model.zero_grad(set_to_none=True)
            
        Note:
            zero_grad()方法的特点：
            - 在训练循环开始时调用，防止梯度累积
            - set_to_none=True可以节省内存，但可能影响某些优化器
            - set_to_none=False将梯度设置为0，更安全但占用更多内存
            - 递归作用于所有子模块的参数
            - 与optimizer.zero_grad()功能类似，但作用于模型而非优化器
            - 在梯度裁剪、梯度累积等高级技巧中很重要
            - 是训练循环的标准组成部分
            - 支持内存优化和性能调优
        """
        for p in self.parameters():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def requires_grad_(self, requires_grad: bool = True):
        """设置参数是否需要计算梯度 (Requires Grad)
        
        递归地设置模块中所有参数的requires_grad属性。这用于冻结或解冻
        模型参数，在迁移学习、微调等场景中非常有用。
        
        Args:
            requires_grad (bool, optional): 是否需要计算梯度。True表示需要，
                                           False表示不需要。默认值: True
                                           
        Returns:
            Module: 返回自身，支持链式调用
            
        Examples::
        
            >>> model = MyModel()
            >>> 
            >>> # 冻结所有参数
            >>> model.requires_grad_(False)
            >>> 
            >>> # 解冻特定层
            >>> model.layer1.requires_grad_(True)
            >>> 
            >>> # 链式调用
            >>> model.requires_grad_(True).train()
            
        Note:
            requires_grad_()方法的特点：
            - 递归作用于所有子模块的参数
            - 返回自身，支持方法链式调用
            - 在迁移学习中用于冻结预训练模型
            - 在微调中用于分层训练
            - 比单独设置每个参数更高效
            - 影响后续计算的梯度追踪
            - 与no_grad()上下文管理器功能不同
            - 是模型训练控制的重要工具
            - 支持灵活的参数训练策略
        """
        for p in self.parameters():
            p.requires_grad = requires_grad
        return self

    def state_dict(self, destination=None, prefix: str = '', keep_vars: bool = False):
        """返回模块的状态字典 (State Dictionary)
        
        返回包含模块整个状态的字典。状态字典包括参数和持久化缓冲区。
        这个字典可以用于保存模型，后续可以通过load_state_dict()恢复。
        
        Args:
            destination (dict, optional): 目标字典。如果为None，会创建新的有序字典
            prefix (str, optional): 参数和缓冲区名称的前缀。默认值: ''
            keep_vars (bool, optional): 是否保持变量对象而非张量数据。默认值: False
            
        Returns:
            dict: 包含模块状态的有序字典
            
        Examples::
        
            >>> class MyModule(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.weight = Parameter(rm.randn(3, 3))
            ...         self.register_buffer('running_mean', rm.zeros(3))
            ...
            >>> m = MyModule()
            >>> state = m.state_dict()
            >>> print(list(state.keys()))
            ['weight', 'running_mean']
            >>> print(state['weight'].shape)  # (3, 3)
            
        Note:
            state_dict()方法的特点：
            - 包含所有参数和缓冲区
            - 递归包含所有子模块的状态，名称带前缀
            - 默认返回张量的数据而非张量对象本身
            - 返回的字典是有序的，保证一致性
            - 常用于模型保存和迁移学习
            - 与load_state_dict()配对使用
            - keep_vars=True时返回实际张量对象，用于调试
            - 是模型序列化的标准方式
            - 支持前缀机制，便于模块嵌套和状态管理
        """
        if destination is None:
            destination = {}
        
        # 保存参数
        for name, param in self._parameters.items():
            if param is not None:
                if keep_vars:
                    destination[prefix + name] = param
                else:
                    destination[prefix + name] = param.data
        
        # 保存缓冲区
        for name, buffer in self._buffers.items():
            if buffer is not None:
                if keep_vars:
                    destination[prefix + name] = buffer
                else:
                    destination[prefix + name] = buffer.data
        
        # 递归保存子模块
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + '.', keep_vars)
        
        return destination

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """将状态字典复制到模块中 (Load State Dictionary)
        
        将state_dict中的参数和缓冲区复制到当前模块中。这是state_dict()
        的逆操作，常用于加载保存的模型参数。
        
        Args:
            state_dict (dict): 包含参数和缓冲区的状态字典
            strict (bool, optional): 是否严格要求状态字典与模块结构完全匹配。
                                    如果为True，缺失或多余的键会引发错误。
                                    如果为False，会忽略不匹配的键。默认值: True
            
        Returns:
            NamedTuple: (missing_keys, unexpected_keys)
                missing_keys: 模块中存在但state_dict中缺失的键列表
                unexpected_keys: state_dict中存在但模块中不存在的键列表
        
        Raises:
            RuntimeError: 当strict=True且存在缺失或意外的键时
        
        Examples::
        
            >>> # 保存模型状态
            >>> m1 = MyModule()
            >>> state = m1.state_dict()
            >>> 
            >>> # 加载到另一个模型
            >>> m2 = MyModule()
            >>> m2.load_state_dict(state)
            >>> 
            >>> # 非严格模式加载
            >>> m2.load_state_dict(state, strict=False)
        
        Note:
            load_state_dict()方法的特点：
            - 是state_dict()的逆操作，用于模型恢复
            - 支持严格模式和非严格模式加载
            - 严格模式下确保状态完全匹配，适合完整模型加载
            - 非严格模式下支持部分加载，适合迁移学习
            - 返回缺失和意外的键信息，便于调试
            - 会覆盖模块中的现有参数和缓冲区
            - 不影响模块结构，只更新状态值
            - 是模型加载的标准方式
            - 与state_dict()配合实现模型的序列化/反序列化
        """
        from collections import namedtuple
        MissingAndUnexpected = namedtuple('MissingAndUnexpected', ['missing_keys', 'unexpected_keys'])
        
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        
        # 获取当前模块的所有参数和缓冲区
        local_state = {**self._parameters, **self._buffers}
        
        # 分离本地键和子模块键
        local_keys = set(local_state.keys())
        child_state_dict = {}
        
        for key, value in state_dict.items():
            if '.' in key:
                # 子模块的键，格式为 "child_name.param_name"
                child_state_dict[key] = value
            elif key in local_state:
                # 当前模块的键
                if local_state[key] is None:
                    missing_keys.append(key)
                else:
                    try:
                        # 统一处理：获取目标对象的底层数组
                        target_data = local_state[key].data
                        
                        # 如果value是张量对象（keep_vars=True的情况），获取其data属性
                        if isinstance(value, (TN, Parameter)):
                            source_data = value.data
                        else:
                            source_data = value
                        
                        # 处理数组类型转换（CuPy/NumPy互转）
                        is_cupy_to_numpy = False
                        is_numpy_to_cupy = False
                        
                        if cp is not None:
                            is_cupy_to_numpy = isinstance(source_data, cp.ndarray) and isinstance(target_data, np.ndarray)
                            is_numpy_to_cupy = isinstance(source_data, np.ndarray) and isinstance(target_data, cp.ndarray)
                        
                        if is_cupy_to_numpy:
                            source_data = source_data.get()
                        elif is_numpy_to_cupy:
                            source_data = cp.asarray(source_data)
                        
                        # 确保source_data是数组
                        if not isinstance(source_data, (np.ndarray, cp.ndarray if cp is not None else type(None))):
                            # 如果source_data不是数组，尝试将其转换为数组
                            try:
                                # 尝试根据target_data的类型选择合适的数组库
                                if cp is not None and isinstance(target_data, cp.ndarray):
                                    source_data = cp.array(source_data)
                                else:
                                    source_data = np.array(source_data)
                            except Exception:
                                # 如果转换失败，使用默认值
                                import warnings
                                original_shape = getattr(value, 'shape', 'unknown')
                                if cp is not None and isinstance(target_data, cp.ndarray):
                                    source_data = cp.zeros(target_data.shape, dtype=target_data.dtype)
                                else:
                                    source_data = np.zeros(target_data.shape, dtype=target_data.dtype)
                                warnings.warn(
                                    f"Warning: Failed to convert parameter '{key}' from checkpoint to array. "
                                    f"Original shape: {original_shape}, "
                                    f"Expected shape: {target_data.shape}. "
                                    f"Filling with zeros.")
                        
                        # 统一数组赋值（无论维度多少）
                        target_data[()] = source_data
                    except Exception as e:
                        error_msgs.append(f'While copying the parameter named "{key}", '
                                        f'whose dimensions in the model are {local_state[key].shape} and '
                                        f'whose dimensions in the checkpoint are {getattr(value, 'shape', 'unknown')}, '
                                        f'an exception occurred : {e}')
            else:
                unexpected_keys.append(key)
        
        # 递归处理子模块
        for name, module in self._modules.items():
            if module is not None:
                # 筛选属于该子模块的键
                child_prefix = f"{name}."
                child_keys = {k: v for k, v in child_state_dict.items() if k.startswith(child_prefix)}
                
                # 移除前缀，只保留参数名
                child_keys_stripped = {k[len(child_prefix):]: v for k, v in child_keys.items()}
                
                if child_keys_stripped:
                    child_result = module.load_state_dict(child_keys_stripped, strict=False)
                    missing_keys.extend([f"{name}.{k}" for k in child_result.missing_keys])
                    unexpected_keys.extend([f"{name}.{k}" for k in child_result.unexpected_keys])
        
        # 检查缺失的键（只检查当前模块的键）
        state_dict_local_keys = {k for k in state_dict.keys() if '.' not in k}
        missing_keys.extend(list(local_keys - state_dict_local_keys))
        
        # 处理严格模式
        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.append(f'Unexpected key(s) in state_dict: {unexpected_keys}')
            if len(missing_keys) > 0:
                error_msgs.append(f'Missing key(s) in state_dict: {missing_keys}')
        
        if len(error_msgs) > 0:
            raise RuntimeError('\n'.join(error_msgs))
        
        return MissingAndUnexpected(missing_keys, unexpected_keys)

    def apply(self, fn):
        """递归地将函数fn应用到每个子模块 (Apply Function)
        
        将函数fn递归地应用到当前模块及其所有子模块上。这个方法
        提供了一种便捷的方式来对模块树中的所有模块执行相同的操作。
        
        Args:
            fn (callable): 要应用到每个模块的函数，接受Module作为参数
            
        Returns:
            Module: 返回自身，支持链式调用
            
        Examples::
        
            >>> def init_weights(m):
            ...     if isinstance(m, Linear):
            ...         rm.init.normal_(m.weight.data, 0.0, 0.02)
            ...         if m.bias is not None:
            ...             rm.init.constant_(m.bias.data, 0)
            ...
            >>> net = MyNetwork()
            >>> net.apply(init_weights)  # 初始化所有Linear层的权重
            
            >>> def print_modules(m):
            ...     print(f"Module: {type(m).__name__}")
            ...
            >>> net.apply(print_modules)  # 打印所有模块
            
        Note:
            apply()方法的特点：
                - 递归遍历所有子模块，包括深层嵌套的模块
                - 先对子模块应用函数，最后对自身应用函数
                - 返回自身，支持方法链式调用
                - 常用于权重初始化、模块配置、调试等
                - 函数fn应该接受一个Module参数
                - 可以用于批量设置模块属性或状态
                - 在模型构建和配置阶段非常有用
                - 提供了对整个模块树的统一操作接口
                - 支持任意自定义的模块操作函数
        """
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self
    
    def get_parameter(self, target: str):
        """获取指定名称的参数 (Get Parameter)
        
        根据参数名称路径获取对应的参数。参数名称路径使用点号分隔，
        支持访问嵌套模块中的参数，如 'layer1.weight' 或 'layer2.bias'。
        
        Args:
            target (str): 参数名称路径，使用点号分隔层级关系
            
        Returns:
            Parameter: 指定的参数对象
            
        Raises:
            AttributeError: 当参数名称路径不存在时
            
        Examples::
        
            >>> class Net(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.conv1 = Conv2d(3, 64, 3)
            ...         self.conv2 = Conv2d(64, 128, 3)
            ...
            >>> net = Net()
            >>> # 获取顶层参数
            >>> conv1_weight = net.get_parameter('conv1.weight')
            >>> # 获取嵌套参数
            >>> conv2_bias = net.get_parameter('conv2.bias')
            >>> print(conv1_weight.shape)  # (64, 3, 3, 3)
            
        Note:
            get_parameter()方法的特点：
            - 支持点号分隔的参数路径，可访问嵌套模块参数
            - 返回实际的Parameter对象，可用于梯度计算
            - 常用于访问特定参数进行操作或检查
            - 与named_parameters()配合使用，便于参数管理
            - 路径不存在时抛出AttributeError异常
            - 支持深层嵌套的参数访问
            - 在模型优化和参数初始化中很有用
            - 提供了比直接属性访问更安全的参数获取方式
        """
        if '.' in target:
            # 嵌套参数，如 'layer1.weight'
            module_path, param_name = target.rsplit('.', 1)
            module = self.get_submodule(module_path)
            if not hasattr(module, param_name) or param_name not in module._parameters:
                raise AttributeError(f"{self._get_name()} has no parameter '{target}'")
            return getattr(module, param_name)
        else:
            # 本地参数
            if target not in self._parameters:
                raise AttributeError(f"{self._get_name()} has no parameter '{target}'")
            return self._parameters[target]

    def get_submodule(self, target: str, create_if_missing=False):
        """获取指定名称的子模块 (Get Submodule)
        
        根据模块名称路径获取对应的子模块。模块名称路径使用点号分隔，
        支持访问深层嵌套的子模块，如 'layer1.conv1' 或 'features.pool'。
        
        Args:
            target (str): 模块名称路径，使用点号分隔层级关系
            create_if_missing (bool): 如果为True，当中间路径不存在时自动创建空Module对象
            
        Returns:
            Module: 指定的子模块对象
            
        Raises:
            AttributeError: 当模块名称路径不存在且create_if_missing为False时
            
        Examples::
        
            >>> class Net(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.features = Sequential(
            ...             Conv2d(3, 64, 3),
            ...             ReLU(),
            ...             MaxPool2d(2)
            ...         )
            ...         self.classifier = Linear(64 * 14 * 14, 10)
            ...
            >>> net = Net()
            >>> # 获取子模块
            >>> features = net.get_submodule('features')
            >>> conv1 = net.get_submodule('features.0')  # 第一个Conv2d
            >>> print(type(features).__name__)  # Sequential
            >>> print(type(conv1).__name__)      # Conv2d
            >>> # 自动创建中间路径
            >>> new_module = net.get_submodule('new.path.module', create_if_missing=True)
            
        Note:
            get_submodule()方法的特点：
                - 支持点号分隔的模块路径，可访问深层嵌套模块
                - 返回实际的Module对象，可用于进一步操作
                - 常用于动态访问和操作特定子模块
                - 与named_modules()配合使用，便于模块管理
                - 路径不存在时抛出AttributeError异常（除非create_if_missing=True）
                - 支持任意深度的模块嵌套访问
                - 在模型修改和模块替换中很有用
                - 提供了比直接属性访问更安全的模块获取方式
                - 支持通过索引访问Sequential中的模块
                - create_if_missing=True时会自动创建中间路径的空Module对象
        """
        if '.' in target:
            # 嵌套模块，如 'layer1.conv1'
            module_path, submodule_name = target.rsplit('.', 1)
            module = self.get_submodule(module_path, create_if_missing=create_if_missing)
            if not hasattr(module, submodule_name) or submodule_name not in module._modules:
                if create_if_missing:
                    module.add_module(submodule_name, Module())
                else:
                    raise AttributeError(f"{self._get_name()} has no submodule '{target}'")
            return getattr(module, submodule_name)
        else:
            # 本地子模块
            if target not in self._modules:
                if create_if_missing:
                    self.add_module(target, Module())
                else:
                    raise AttributeError(f"{self._get_name()} has no submodule '{target}'")
            return self._modules[target]

    def get_buffer(self, target: str):
        """获取指定名称的缓冲区 (Get Buffer)
        
        根据缓冲区名称路径获取对应的缓冲区张量。缓冲区名称路径使用点号分隔，
        支持访问嵌套模块中的缓冲区，如 'layer1.running_mean' 或 'batchnorm.weight'。
        
        Args:
            target (str): 缓冲区名称路径，使用点号分隔层级关系
            
        Returns:
            Tensor: 指定的缓冲区张量
            
        Raises:
            AttributeError: 当缓冲区名称路径不存在时
            
        Examples::
        
            >>> class Net(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.bn1 = BatchNorm2d(64)
            ...         self.bn2 = BatchNorm2d(128)
            ...
            >>> net = Net()
            >>> # 获取顶层缓冲区
            >>> running_mean = net.get_buffer('bn1.running_mean')
            >>> # 获取嵌套缓冲区
            >>> running_var = net.get_buffer('bn2.running_var')
            >>> print(running_mean.shape)  # (64,)
                
        Note:
            get_buffer()方法的特点：
                - 支持点号分隔的缓冲区路径，可访问嵌套模块缓冲区
                - 返回实际的Tensor对象，可用于数值计算
                - 常用于访问特定缓冲区进行检查或操作
                - 与named_buffers()配合使用，便于缓冲区管理
                - 路径不存在时抛出AttributeError异常
                - 支持深层嵌套的缓冲区访问
                - 在模型调试和状态检查中很有用
                - 提供了比直接属性访问更安全的缓冲区获取方式
                - 缓冲区通常存储运行统计信息，如均值和方差
        """
        if '.' in target:
            # 嵌套缓冲区，如 'layer1.running_mean'
            module_path, buffer_name = target.rsplit('.', 1)
            module = self.get_submodule(module_path)
            if not hasattr(module, buffer_name) or buffer_name not in module._buffers:
                raise AttributeError(f"{self._get_name()} has no buffer '{target}'")
            return getattr(module, buffer_name)
        else:
            # 本地缓冲区
            if target not in self._buffers:
                raise AttributeError(f"{self._get_name()} has no buffer '{target}'")
            return self._buffers[target]

    def has_parameter(self, target: str) -> bool:
        """检查指定名称的参数是否存在 (Has Parameter)
        
        根据参数名称路径检查对应的参数是否存在。参数名称路径使用点号分隔，
        支持检查嵌套模块中的参数，如 'layer1.weight' 或 'layer2.bias'。
        
        Args:
            target (str): 参数名称路径，使用点号分隔层级关系
            
        Returns:
            bool: 参数存在返回True，不存在返回False
            
        Examples::
        
            >>> class Net(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.conv1 = Conv2d(3, 64, 3)
            ...         self.conv2 = Conv2d(64, 128, 3)
            ...
            >>> net = Net()
            >>> # 检查参数存在性
            >>> has_weight = net.has_parameter('conv1.weight')  # True
            >>> has_bias = net.has_parameter('conv1.bias')      # True
            >>> has_fake = net.has_parameter('conv1.fake')      # False
            >>> print(has_weight, has_bias, has_fake)
                
        Note:
            has_parameter()方法的特点：
                - 支持点号分隔的参数路径，可检查嵌套模块参数
                - 返回布尔值，便于条件判断
                - 常用于参数存在性验证和错误处理
                - 与get_parameter()配合使用，提供安全的参数访问
                - 不会抛出异常，总是返回布尔值
                - 支持深层嵌套的参数检查
                - 在动态模型构建中很有用
                - 提供了比try-except更简洁的存在性检查
                - 有助于编写健壮的模型操作代码
        """
        if '.' in target:
            # 嵌套参数，如 'layer1.weight'
            module_path, param_name = target.rsplit('.', 1)
            try:
                module = self.get_submodule(module_path)
                return hasattr(module, param_name) and param_name in module._parameters
            except AttributeError:
                return False
        else:
            # 本地参数
            return target in self._parameters

    def has_buffer(self, target: str) -> bool:
        """检查指定名称的缓冲区是否存在 (Has Buffer)
        
        根据缓冲区名称路径检查对应的缓冲区是否存在。缓冲区名称路径使用点号分隔，
        支持检查嵌套模块中的缓冲区，如 'layer1.running_mean' 或 'batchnorm.weight'。
        
        Args:
            target (str): 缓冲区名称路径，使用点号分隔层级关系
            
        Returns:
            bool: 缓冲区存在返回True，不存在返回False
            
        Examples::
        
            >>> class Net(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.bn1 = BatchNorm2d(64)
            ...         self.register_buffer('custom_buffer', tensor([1.0, 2.0]))
            ...
            >>> net = Net()
            >>> # 检查缓冲区存在性
            >>> has_running_mean = net.has_buffer('bn1.running_mean')  # True
            >>> has_custom = net.has_buffer('custom_buffer')            # True
            >>> has_fake = net.has_buffer('fake_buffer')               # False
            >>> print(has_running_mean, has_custom, has_fake)
                
        Note:
            has_buffer()方法的特点：
                - 支持点号分隔的缓冲区路径，可检查嵌套模块缓冲区
                - 返回布尔值，便于条件判断
                - 常用于缓冲区存在性验证和错误处理
                - 与get_buffer()配合使用，提供安全的缓冲区访问
                - 不会抛出异常，总是返回布尔值
                - 支持深层嵌套的缓冲区检查
                - 在动态模型构建中很有用
                - 提供了比try-except更简洁的存在性检查
                - 有助于编写健壮的模型操作代码
                - 缓冲区通常用于存储运行时状态信息
        """
        if '.' in target:
            # 嵌套缓冲区，如 'layer1.running_mean'
            module_path, buffer_name = target.rsplit('.', 1)
            try:
                module = self.get_submodule(module_path)
                return hasattr(module, buffer_name) and buffer_name in module._buffers
            except AttributeError:
                return False
        else:
            # 本地缓冲区
            return target in self._buffers

    def set_parameter(self, name: str, param):
        """设置指定名称的参数 (Set Parameter)
        
        设置模块的参数，如果参数不存在则添加新参数。参数会被自动注册到模块中，
        并参与梯度计算和优化器更新。支持点号分隔的路径来设置嵌套模块的参数。
        
        Args:
            name (str): 参数名称，支持点号分隔的嵌套路径
            param (Parameter or None): 要设置的参数对象，None表示删除参数
            
        Examples::
        
            >>> class Net(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.conv1 = Conv2d(3, 64, 3)
            ...
            >>> net = Net()
            >>> # 设置新参数
            >>> new_weight = Parameter(randn(64, 3, 3, 3))
            >>> net.set_parameter('conv1.weight', new_weight)
            >>> 
            >>> # 添加新参数
            >>> new_bias = Parameter(randn(64))
            >>> net.set_parameter('conv1.custom_bias', new_bias)
            >>> 
            >>> # 删除参数
            >>> net.set_parameter('conv1.bias', None)
                
        Note:
            set_parameter()方法的特点：
                - 支持点号分隔的参数路径，可设置嵌套模块参数
                - 参数不存在时会自动添加新参数
                - 设置为None时会删除现有参数
                - 参数会被自动注册，参与参数管理
                - 常用于参数初始化、模型修改和参数替换
                - 与register_parameter()不同，支持嵌套路径
                - 提供了比直接属性赋值更安全的参数设置方式
                - 在模型微调和参数操作中很有用
                - 确保参数正确注册到模块系统中
        """
        if '.' in name:
            # 嵌套参数，如 'layer1.weight'
            module_path, param_name = name.rsplit('.', 1)
            module = self.get_submodule(module_path)
            if param is None:
                # 删除参数
                if hasattr(module, param_name) and param_name in module._parameters:
                    delattr(module, param_name)
            else:
                module.register_parameter(param_name, param)
        else:
            # 本地参数
            if param is None:
                # 删除参数
                if hasattr(self, name) and name in self._parameters:
                    delattr(self, name)
            else:
                self.register_parameter(name, param)

    def set_buffer(self, name: str, tensor):
        """设置指定名称的缓冲区 (Set Buffer)
        
        设置模块的缓冲区，如果缓冲区不存在则添加新缓冲区。缓冲区会被自动注册到模块中，
        但不参与梯度计算和优化器更新。支持点号分隔的路径来设置嵌套模块的缓冲区。
        
        Args:
            name (str): 缓冲区名称，支持点号分隔的嵌套路径
            tensor (Tensor or None): 要设置的缓冲区张量，None表示删除缓冲区
            
        Examples::
            
            >>> class Net(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.bn1 = BatchNorm2d(64)
            ...
            >>> net = Net()
            >>> # 设置新缓冲区
            >>> new_mean = tensor([0.0] * 64)
            >>> net.set_buffer('bn1.running_mean', new_mean)
            >>> 
            >>> # 添加新缓冲区
            >>> custom_stats = tensor([1.0, 2.0, 3.0])
            >>> net.set_buffer('custom_stats', custom_stats)
            >>> 
            >>> # 删除缓冲区
            >>> net.set_buffer('bn1.running_var', None)
            
        Note:
            set_buffer()方法的特点：
                - 支持点号分隔的缓冲区路径，可设置嵌套模块缓冲区
                - 缓冲区不存在时会自动添加新缓冲区
                - 设置为None时会删除现有缓冲区
                - 缓冲区会被自动注册，参与缓冲区管理
                - 缓冲区不参与梯度计算，用于存储状态信息
                - 常用于状态初始化、模型修改和状态管理
                - 与register_buffer()不同，支持嵌套路径
                - 提供了比直接属性赋值更安全的缓冲区设置方式
                - 在模型状态管理和调试中很有用
                - 确保缓冲区正确注册到模块系统中
        """
        if '.' in name:
            # 嵌套缓冲区，如 'layer1.running_mean'
            module_path, buffer_name = name.rsplit('.', 1)
            module = self.get_submodule(module_path)
            module.register_buffer(buffer_name, tensor)
        else:
            # 本地缓冲区
            self.register_buffer(name, tensor)

    def type(self, dtype=None):
        """
        返回或转换模块所有参数和缓冲区的数据类型
        
        行为：
        - 如果不传入参数，返回模块中第一个参数的数据类型
        - 如果传入数据类型参数，将模块的所有参数和缓冲区转换为指定数据类型
        
        参数:
            dtype: 数据类型，可以是Python类型、NumPy dtype、字符串或Riemann dtype
                   如果为None，则返回模块中第一个参数的数据类型
                   如果模块没有参数，则返回None
        
        返回:
            如果dtype为None，返回模块中第一个参数的数据类型，或None（如果没有参数）
            否则返回转换后的数据类型的模块本身（原地操作）
        
        Examples:
            >>> model = MyModule()
            >>> model.type()  # 返回第一个参数的数据类型
            >>> model.type(float32)  # 将所有参数转换为float32
            >>> model.type('float64')  # 将所有参数转换为float64
        """
        # 如果不传入参数，返回模块中第一个参数的数据类型
        if dtype is None:
            for name, param in self._parameters.items():
                if param is not None:
                    return param.dtype
            return None
        
        # 直接调用 to() 方法进行数据类型转换
        return self.to(dtype=dtype)

    def float(self):
        """转换为float32类型 (Float Cast)
        
        将模块的所有参数和缓冲区转换为float32类型。这是一个递归操作，
        会影响当前模块及其所有子模块。
        
        Returns:
            Module: 返回自身，支持链式调用
            
        Examples::
            
            >>> class Net(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.conv1 = Conv2d(3, 64, 3)
            ...
            >>> net = Net()
            >>> # 转换为float32类型
            >>> net.float()
            >>> # 链式调用
            >>> net.float().eval()
            
        Note:
            float()方法的特点：
                - 等价于调用 type('float32') 或 type(np.float32)
                - 递归转换所有子模块的参数和缓冲区
                - 返回自身，支持链式调用
                - float32是PyTorch默认的数值类型
                - 在大多数训练场景中使用float32
                - 提供了良好的精度和性能平衡
                - 常用于从其他精度类型恢复到标准精度
                - 在精度调试和测试中很有用
                - 确保计算精度满足训练要求
                - 是深度学习中最常用的数据类型
        """
        return self.type('float32')

    def double(self):
        """转换为float64类型 (Double Cast)
        
        将模块的所有参数和缓冲区转换为float64类型。这是一个递归操作，
        会影响当前模块及其所有子模块。
        
        Returns:
            Module: 返回自身，支持链式调用
            
        Examples::
            
            >>> class Net(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.conv1 = Conv2d(3, 64, 3)
            ...
            >>> net = Net()
            >>> # 转换为float64类型
            >>> net.double()
            >>> # 链式调用
            >>> net.double().eval()
            
        Note:
            double()方法的特点：
                - 等价于调用 type('float64') 或 type(np.float64)
                - 递归转换所有子模块的参数和缓冲区
                - 返回自身，支持链式调用
                - 提供更高的数值精度，减少舍入误差
                - 内存占用约为float32的两倍
                - 在科学计算和高精度要求场景中使用
                - 可能影响计算性能，特别是在GPU上
                - 常用于数值稳定性要求高的任务
                - 在梯度检查和调试中很有用
                - 某些硬件可能不支持float64加速
                - 适合需要高精度的数学运算
        """
        return self.type('float64')

    def half(self):
        """转换为float16类型 (Half Cast)
        
        将模块的所有参数和缓冲区转换为float16类型。这是一个递归操作，
        会影响当前模块及其所有子模块。
        
        Returns:
            Module: 返回自身，支持链式调用
            
        Examples::
            
            >>> class Net(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.conv1 = Conv2d(3, 64, 3)
            ...
            >>> net = Net()
            >>> # 转换为float16类型
            >>> net.half()
            >>> # 链式调用
            >>> net.half().eval()
            
        Note:
            half()方法的特点：
                - 等价于调用 type('float16') 或 type(np.float16)
                - 递归转换所有子模块的参数和缓冲区
                - 返回自身，支持链式调用
                - 内存占用约为float32的一半
                - 在支持的硬件上可以显著加速计算
                - 可能导致数值精度问题和梯度下溢
                - 常用于推理时的模型压缩和加速
                - 在混合精度训练中与float32配合使用
                - 需要确保硬件支持float16计算
                - 在大规模模型训练中节省显存
                - 可能影响模型收敛性和数值稳定性
        """
        return self.type('float16')

    def extra_repr(self) -> str:
        """设置模块的额外表示信息 (Extra Representation)
        
        返回模块的额外表示信息字符串，该字符串会包含在__repr__方法的输出中。
        子类可以重写此方法来添加模块特定的信息，如参数数量、配置选项等。
        
        Returns:
            str: 额外表示信息字符串，如果不需要额外信息则返回空字符串
            
        Examples::
        
            >>> class Linear(Module):
            ...     def __init__(self, in_features, out_features, bias=True):
            ...         super().__init__()
            ...         self.in_features = in_features
            ...         self.out_features = out_features
            ...         self.bias = bias
            ...         self.weight = Parameter(randn(in_features, out_features))
            ...         if bias:
            ...             self.bias_param = Parameter(randn(out_features))
            ...     
            ...     def extra_repr(self):
            ...         return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias}'
            ...
            >>> layer = Linear(10, 20, True)
            >>> print(layer)
            # 输出: Linear(in_features=10, out_features=20, bias=True)
            
        Note:
            extra_repr()方法的特点：
                - 默认返回空字符串，子类可重写添加特定信息
                - 返回字符串会自动包含在__repr__输出中
                - 常用于显示模块的配置参数和结构信息
                - 字符串应该简洁明了，避免过长
                - 与__repr__配合使用，提供完整的模块信息
                - 支持多行信息，但通常使用单行格式
                - 在调试和日志记录中很有用
                - 是PyTorch模块系统的标准组成部分
                - 有助于模型的可视化和理解
        """
        return ''

    def __repr__(self):
        """返回模块的字符串表示 (String Representation)
        
        生成模块的字符串表示，包含模块名称、额外信息和子模块结构。
        这是模块的官方字符串表示，用于调试、日志记录和模型可视化。
        
        Returns:
            str: 模块的字符串表示
            
        Examples::
        
            >>> class SimpleNet(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.conv1 = Conv2d(3, 64, 3)
            ...         self.relu = ReLU()
            ...         self.conv2 = Conv2d(64, 128, 3)
            ...
            ...     def forward(self, x):
            ...         x = self.conv1(x)
            ...         x = self.relu(x)
            ...         x = self.conv2(x)
            ...         return x
            ...
            >>> net = SimpleNet()
            >>> print(net)
            # 输出类似:
            # SimpleNet(
            #   (conv1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1))
            #   (relu): ReLU()
            #   (conv2): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
            # )
            
        Note:
            __repr__()方法的特点：
                - 提供模块的完整结构信息
                - 包含模块名称和extra_repr信息
                - 递归显示所有子模块的结构
                - 使用缩进表示模块层次关系
                - 格式与PyTorch保持一致
                - 用于调试、日志和模型可视化
                - 支持任意深度的模块嵌套
                - 是模块系统的核心表示方法
                - 与print()函数配合使用
                - 提供人类可读的模型结构
        """
        class_name = self.__class__.__name__
        extra_info = self.extra_repr()
        
        # 构建基本字符串
        if extra_info:
            main_str = f'{class_name}({extra_info}'
        else:
            main_str = f'{class_name}('
        
        # 添加子模块信息
        child_lines = []
        for name, module in self._modules.items():
            if module is not None:
                # 获取子模块的repr，并处理多行缩进
                child_repr = repr(module)
                # 如果子模块repr包含换行符，需要处理缩进
                if '\n' in child_repr:
                    # 对多行子模块repr进行缩进处理
                    child_lines_indented = []
                    for line in child_repr.split('\n'):
                        if line.strip():  # 非空行
                            child_lines_indented.append(f'    {line}')
                        else:  # 空行
                            child_lines_indented.append('')
                    child_repr_formatted = '\n'.join(child_lines_indented)
                    child_lines.append(f'  ({name}): {child_repr_formatted}')
                else:
                    child_lines.append(f'  ({name}): {child_repr}')
        
        if child_lines:
            # 有子模块时的格式
            if extra_info:
                main_str += '\n' + ',\n'.join(child_lines) + '\n)'
            else:
                main_str += '\n' + ',\n'.join(child_lines) + '\n)'
        else:
            # 没有子模块时的格式
            main_str += ')'
        
        return main_str

    def set_submodule(self, target: str, module):
        """设置指定路径的子模块 (Set Submodule)
        
        根据子模块名称路径设置对应的子模块。子模块名称路径使用点号分隔，
        支持设置嵌套模块中的子模块，如 'layer1.conv1' 或 'backbone.features'。
        如果路径中的中间模块不存在，会自动创建空的Module对象。
        
        Args:
            target (str): 子模块名称路径，使用点号分隔层级关系
            module (Module): 要设置的子模块对象，None表示删除子模块
                
        Returns:
            Module: 返回自身，支持链式调用
                
        Examples::
            
            >>> class Net(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.layer1 = Module()
            ...
            >>> net = Net()
            >>> # 设置子模块
            >>> conv = Conv2d(3, 64, 3)
            >>> net.set_submodule('layer1.conv1', conv)
            >>> # 设置深层嵌套子模块
            >>> net.set_submodule('backbone.features.conv1', conv)
            >>> # 删除子模块
            >>> net.set_submodule('layer1.conv1', None)
            >>> # 链式调用
            >>> net.set_submodule('layer1.bn1', BatchNorm2d(64)).eval()
                
        Note:
            set_submodule()方法的特点：
                - 支持点号分隔的子模块路径，可设置深层嵌套的子模块
                - 如果中间路径的模块不存在，会自动创建空的Module对象
                - 设置为None时会删除指定的子模块
                - 返回自身，支持链式调用
                - 常用于动态修改模型结构
                - 支持运行时替换模型组件
                - 提供了比直接属性赋值更安全的子模块设置方式
                - 在模型构建和修改中很有用
                - 确保子模块正确注册到模块系统中
                - 与get_submodule()配合使用，便于子模块管理
        """
        if '.' in target:
            # 嵌套子模块，如 'layer1.conv1'
            module_path, submodule_name = target.rsplit('.', 1)
            parent_module = self.get_submodule(module_path, create_if_missing=True)
            parent_module.add_module(submodule_name, module)
        else:
            # 本地子模块
            self.add_module(target, module)
        
        return self

    def has_submodule(self, target: str) -> bool:
        """检查指定名称的子模块是否存在 (Has Submodule)
        
        根据子模块名称路径检查对应的子模块是否存在。子模块名称路径使用点号分隔，
        支持检查嵌套模块中的子模块，如 'layer1.conv1' 或 'backbone.features'。
        
        Args:
            target (str): 子模块名称路径，使用点号分隔层级关系
                
        Returns:
            bool: 子模块存在返回True，不存在返回False
                
        Examples::
            
            >>> class Net(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.layer1 = Module()
            ...         self.layer1.conv1 = Conv2d(3, 64, 3)
            ...
            >>> net = Net()
            >>> # 检查子模块存在性
            >>> net.has_submodule('layer1')  # True
            >>> net.has_submodule('layer1.conv1')  # True
            >>> net.has_submodule('layer1.conv2')  # False
            >>> net.has_submodule('layer2.conv1')  # False
                
        Note:
            has_submodule()方法的特点：
                - 支持点号分隔的子模块路径，可检查深层嵌套的子模块
                - 不会自动创建不存在的中间模块
                - 返回布尔值，便于条件判断
                - 常用于检查模型结构的完整性
                - 在动态模型操作前进行安全检查
                - 与get_submodule()配合使用，提供安全的模块访问
                - 在模型调试和验证中很有用
                - 支持任意深度的模块嵌套检查
                - 路径不存在时不会抛出异常
                - 提供了非侵入式的模块存在性检查方式
        """
        if '.' in target:
            # 嵌套子模块，如 'layer1.conv1'
            module_path, submodule_name = target.rsplit('.', 1)
            try:
                parent_module = self.get_submodule(module_path)
                return hasattr(parent_module, submodule_name) and submodule_name in parent_module._modules
            except AttributeError:
                return False
        else:
            # 本地子模块
            return target in self._modules

    def delete_submodule(self, target: str):
        """删除指定名称的子模块 (Delete Submodule)
        
        根据子模块名称路径删除对应的子模块。子模块名称路径使用点号分隔，
        支持删除嵌套模块中的子模块，如 'layer1.conv1' 或 'backbone.features'。
        
        Args:
            target (str): 子模块名称路径，使用点号分隔层级关系
                
        Returns:
            Module: 返回自身，支持链式调用
                
        Raises:
            AttributeError: 当子模块名称路径不存在时
                
        Examples::
            
            >>> class Net(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.layer1 = Module()
            ...         self.layer1.conv1 = Conv2d(3, 64, 3)
            ...         self.layer1.conv2 = Conv2d(64, 128, 3)
            ...
            >>> net = Net()
            >>> # 删除子模块
            >>> net.delete_submodule('layer1.conv2')
            >>> # 删除整个子模块分支
            >>> net.delete_submodule('layer1')
            >>> # 链式调用
            >>> net.delete_submodule('layer1').eval()
                
        Note:
            delete_submodule()方法的特点：
                - 支持点号分隔的子模块路径，可删除深层嵌套的子模块
                - 删除操作是递归的，会删除子模块及其所有子模块
                - 返回自身，支持链式调用
                - 子模块不存在时会抛出AttributeError异常
                - 常用于动态修改模型结构
                - 支持运行时移除模型组件
                - 提供了比del属性更安全的子模块删除方式
                - 在模型剪枝和结构优化中很有用
                - 确保子模块从模块系统中正确移除
                - 与set_submodule(target, None)效果相同
        """
        if '.' in target:
            # 嵌套子模块，如 'layer1.conv1'
            module_path, submodule_name = target.rsplit('.', 1)
            parent_module = self.get_submodule(module_path)
            if not hasattr(parent_module, submodule_name) or submodule_name not in parent_module._modules:
                raise AttributeError(f"{self._get_name()} has no submodule '{target}'")
            delattr(parent_module, submodule_name)
        else:
            # 本地子模块
            if target not in self._modules:
                raise AttributeError(f"{self._get_name()} has no submodule '{target}'")
            delattr(self, target)
        
        return self

    def delete_parameter(self, target: str):
        """删除指定名称的参数 (Delete Parameter)
        
        根据参数名称路径删除对应的参数。参数名称路径使用点号分隔，
        支持删除嵌套模块中的参数，如 'layer1.weight' 或 'layer2.bias'。
        
        Args:
            target (str): 参数名称路径，使用点号分隔层级关系
                
        Returns:
            Module: 返回自身，支持链式调用
                
        Raises:
            AttributeError: 当参数名称路径不存在时
                
        Examples::
            
            >>> class Net(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.layer1 = Linear(10, 20)
            ...         self.layer2 = Linear(20, 30)
            ...
            >>> net = Net()
            >>> # 删除参数
            >>> net.delete_parameter('layer1.weight')
            >>> # 删除偏置参数
            >>> net.delete_parameter('layer2.bias')
            >>> # 链式调用
            >>> net.delete_parameter('layer1.weight').eval()
                
        Note:
            delete_parameter()方法的特点：
                - 支持点号分隔的参数路径，可删除深层嵌套的参数
                - 删除操作会从参数字典中移除参数
                - 返回自身，支持链式调用
                - 参数不存在时会抛出AttributeError异常
                - 常用于模型剪枝和参数移除
                - 支持运行时动态修改模型参数结构
                - 提供了比del属性更安全的参数删除方式
                - 在模型优化和压缩中很有用
                - 确保参数从模块系统中正确移除
                - 与set_parameter(target, None)效果相同
                - 删除后参数不再参与梯度计算和优化器更新
        """
        if '.' in target:
            # 嵌套参数，如 'layer1.weight'
            module_path, param_name = target.rsplit('.', 1)
            module = self.get_submodule(module_path)
            if not hasattr(module, param_name) or param_name not in module._parameters:
                raise AttributeError(f"{self._get_name()} has no parameter '{target}'")
            delattr(module, param_name)
        else:
            # 本地参数
            if target not in self._parameters:
                raise AttributeError(f"{self._get_name()} has no parameter '{target}'")
            delattr(self, target)
        
        return self

    def delete_buffer(self, target: str):
        """删除指定名称的缓冲区 (Delete Buffer)
        
        根据缓冲区名称路径删除对应的缓冲区。缓冲区名称路径使用点号分隔，
        支持删除嵌套模块中的缓冲区，如 'layer1.running_mean' 或 'batchnorm.weight'。
        
        Args:
            target (str): 缓冲区名称路径，使用点号分隔层级关系
                
        Returns:
            Module: 返回自身，支持链式调用
                
        Raises:
            AttributeError: 当缓冲区名称路径不存在时
                
        Examples::
            
            >>> class Net(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.bn1 = BatchNorm2d(64)
            ...         self.bn2 = BatchNorm2d(128)
            ...
            >>> net = Net()
            >>> # 删除缓冲区
            >>> net.delete_buffer('bn1.running_mean')
            >>> # 删除权重缓冲区
            >>> net.delete_buffer('bn2.weight')
            >>> # 链式调用
            >>> net.delete_buffer('bn1.running_mean').eval()
                
        Note:
            delete_buffer()方法的特点：
                - 支持点号分隔的缓冲区路径，可删除深层嵌套的缓冲区
                - 删除操作会从缓冲区字典中移除缓冲区
                - 返回自身，支持链式调用
                - 缓冲区不存在时会抛出AttributeError异常
                - 常用于清理不需要的状态信息
                - 支持运行时动态修改模型缓冲区结构
                - 提供了比del属性更安全的缓冲区删除方式
                - 在模型状态管理中很有用
                - 确保缓冲区从模块系统中正确移除
                - 与set_buffer(target, None)效果相同
                - 删除后缓冲区不再参与状态保存和加载
        """
        if '.' in target:
            # 嵌套缓冲区，如 'layer1.running_mean'
            module_path, buffer_name = target.rsplit('.', 1)
            module = self.get_submodule(module_path)
            if not hasattr(module, buffer_name) or buffer_name not in module._buffers:
                raise AttributeError(f"{self._get_name()} has no buffer '{target}'")
            delattr(module, buffer_name)
        else:
            # 本地缓冲区
            if target not in self._buffers:
                raise AttributeError(f"{self._get_name()} has no buffer '{target}'")
            delattr(self, target)
        
        return self

    def copy(self):
        """创建模块的浅拷贝 (Copy)
        
        创建模块的浅拷贝，新的模块与原模块共享相同的参数和缓冲区张量数据，
        但是具有独立的模块对象。修改新模块的属性不会影响原模块，
        但对张量数据的修改会影响两个模块。
        
        Returns:
            Module: 模块的浅拷贝对象
                
        Examples::
            
            >>> class Net(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.conv1 = Conv2d(3, 64, 3)
            ...         self.bn1 = BatchNorm2d(64)
            ...
            >>> net = Net()
            >>> # 创建浅拷贝
            >>> net_copy = net.copy()
            >>> # 修改新模块的属性
            >>> net_copy.conv1 = Conv2d(3, 128, 3)  # 不影响原模块
            >>> # 修改张量数据
            >>> net_copy.conv1.weight[0, 0, 0, 0] = 1.0  # 影响原模块
            >>> # 检查是否为不同对象
            >>> id(net) != id(net_copy)  # True
            >>> id(net.conv1) != id(net_copy.conv1)  # True
                
        Note:
            copy()方法的特点：
                - 创建新的模块对象，但共享张量数据
                - 参数和缓冲区张量是引用拷贝，不是数据拷贝
                - 修改模块属性不影响原模块
                - 修改张量数据会影响两个模块
                - 比deepcopy()更高效，内存占用更少
                - 适用于需要独立模块结构但共享权重的场景
                - 常用于模型权重共享和模型集成
                - 与Python的copy.copy()行为一致
                - 支持任意复杂度的模块结构拷贝
                - 提供了比直接构造更便捷的拷贝方式
        """
        # 创建新的模块实例
        cls = self.__class__
        new_module = cls.__new__(cls)
        
        # 复制所有属性，但不深拷贝参数、缓冲区和模块
        for key, value in self.__dict__.items():
            if key in ['_parameters', '_buffers', '_modules']:
                # 浅拷贝参数、缓冲区和模块字典
                setattr(new_module, key, value.copy())
            else:
                # 直接复制其他属性
                setattr(new_module, key, value)
        
        return new_module

    def deepcopy(self):
        """创建模块的深拷贝 (Deep Copy)
        
        创建模块的深拷贝，新的模块与原模块具有完全独立的参数和缓冲区张量数据。
        修改新模块的任何内容都不会影响原模块，包括模块属性和张量数据。
        
        Returns:
            Module: 模块的深拷贝对象
                
        Examples::
            
            >>> class Net(Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.conv1 = Conv2d(3, 64, 3)
            ...         self.bn1 = BatchNorm2d(64)
            ...
            >>> net = Net()
            >>> # 创建深拷贝
            >>> net_deepcopy = net.deepcopy()
            >>> # 修改新模块的属性
            >>> net_deepcopy.conv1 = Conv2d(3, 128, 3)  # 不影响原模块
            >>> # 修改张量数据
            >>> net_deepcopy.conv1.weight[0, 0, 0, 0] = 1.0  # 不影响原模块
            >>> # 检查是否为不同对象
            >>> id(net) != id(net_deepcopy)  # True
            >>> id(net.conv1) != id(net_deepcopy.conv1)  # True
            >>> id(net.conv1.weight) != id(net_deepcopy.conv1.weight)  # True
                
        Note:
            deepcopy()方法的特点：
                - 创建新的模块对象和张量数据
                - 参数和缓冲区张量是完全独立的拷贝
                - 修改任何内容都不影响原模块
                - 比copy()更耗时，内存占用更多
                - 适用于需要完全独立模型的场景
                - 常用于模型权重初始化和模型备份
                - 与Python的copy.deepcopy()行为一致
                - 支持任意复杂度的模块结构拷贝
                - 确保模型完全独立，无任何共享数据
                - 提供了最安全的模型复制方式
                - 在模型实验和权重保存中很有用
        """
        
        # 创建新的模块实例
        cls = self.__class__
        new_module = cls.__new__(cls)
        
        # 深拷贝所有属性
        for key, value in self.__dict__.items():
            if key in ['_parameters', '_buffers', '_modules']:
                # 深拷贝参数、缓冲区和模块
                new_dict = {}
                for sub_key, sub_value in value.items():
                    if sub_value is not None:
                        if isinstance(sub_value, Parameter):
                            # 深拷贝参数
                            new_data = tensor(sub_value.data.copy())
                            new_param = Parameter(new_data)
                            new_param.requires_grad = sub_value.requires_grad
                            new_dict[sub_key] = new_param
                        elif hasattr(sub_value, 'deepcopy'):
                            # 递归深拷贝模块
                            new_dict[sub_key] = sub_value.deepcopy()
                        elif hasattr(sub_value, 'copy'):
                            # 深拷贝张量或具有copy方法的对象
                            new_dict[sub_key] = sub_value.copy()
                        else:
                            new_dict[sub_key] = sub_value
                    else:
                        new_dict[sub_key] = None
                setattr(new_module, key, new_dict)
            else:
                # 深拷贝其他属性
                try:
                    setattr(new_module, key, copy.deepcopy(value))
                except:
                    setattr(new_module, key, value)
        
        return new_module

    def __copy__(self):
        """Python copy.copy()协议支持 (Copy Protocol)
        
        实现Python的copy.copy()协议，支持copy.copy(self)调用。
        创建模块的浅拷贝，与copy()方法行为相同。
        
        Returns:
            Module: 模块的浅拷贝对象
        """
        return self.copy()

    def __deepcopy__(self, memo):
        """Python copy.deepcopy()协议支持 (Deep Copy Protocol)
        
        实现Python的copy.deepcopy()协议，支持copy.deepcopy(self)调用。
        创建模块的深拷贝，与deepcopy()方法行为相同。
        
        Args:
            memo (dict): 深拷贝的备忘录字典，用于处理循环引用
                
        Returns:
            Module: 模块的深拷贝对象
        """
        import copy
        
        # 检查是否已经在备忘录中，避免循环引用
        if id(self) in memo:
            return memo[id(self)]
        
        # 创建新的模块实例
        cls = self.__class__
        new_module = cls.__new__(cls)
        memo[id(self)] = new_module
        
        # 深拷贝所有属性
        for key, value in self.__dict__.items():
            if key in ['_parameters', '_buffers', '_modules']:
                # 深拷贝参数、缓冲区和模块
                new_dict = {}
                for sub_key, sub_value in value.items():
                    if sub_value is not None:
                        if isinstance(sub_value, Parameter):
                            # 深拷贝参数
                            new_data = tensor(sub_value.data.copy())
                            new_param = Parameter(new_data)
                            new_param.requires_grad = sub_value.requires_grad
                            new_dict[sub_key] = new_param
                        elif hasattr(sub_value, 'deepcopy'):
                            # 使用模块的deepcopy方法
                            new_dict[sub_key] = sub_value.deepcopy()
                        elif hasattr(sub_value, 'copy'):
                            # 深拷贝张量或具有copy方法的对象
                            new_dict[sub_key] = sub_value.copy()
                        else:
                            new_dict[sub_key] = sub_value
                    else:
                        new_dict[sub_key] = None
                setattr(new_module, key, new_dict)
            else:
                # 深拷贝其他属性
                try:
                    setattr(new_module, key, copy.deepcopy(value, memo))
                except:
                    setattr(new_module, key, value)
        
        return new_module
    
# end of class Module

class Linear(Module):
    """
    线性层/全连接层 (Linear/Fully Connected Layer)
    
    实现线性变换：y = xA^T + b，其中A是权重矩阵，b是偏置向量。
    这是神经网络中最基础和常用的层类型。
    
    数学公式:
        output = input @ weight.T + bias
        
    其中：
        - input: 输入张量，形状为 (*, in_features)
        - weight: 权重矩阵，形状为 (out_features, in_features)
        - bias: 偏置向量，形状为 (out_features,)
        - output: 输出张量，形状为 (*, out_features)
    
    Args:
        in_features (int): 输入特征数量
        out_features (int): 输出特征数量
        bias (bool, optional): 是否使用偏置。默认值: True
        dtype (np.dtype, optional): 参数的数据类型。默认值: None（使用默认类型）
        device (str|int|Device, optional): 参数的设备。默认值: None（使用当前设备）
        
    Attributes:
        weight (Parameter): 权重矩阵，形状为(out_features, in_features)
        bias (Parameter or None): 偏置向量，形状为(out_features,)，当bias=False时为None
        in_features (int): 输入特征数量
        out_features (int): 输出特征数量
        
    Examples:
        >>> # 创建线性层
        >>> layer = Linear(10, 5)  # 输入10维，输出5维
        >>> 
        >>> # 前向传播
        >>> x = rm.randn(32, 10)  # 批量大小32，输入10维
        >>> y = layer(x)          # 输出形状: (32, 5)
        >>> print(y.shape)
        >>> 
        >>> # 不使用偏置
        >>> layer_no_bias = Linear(10, 5, bias=False)
        >>> 
        >>> # 查看参数
        >>> for name, param in layer.named_parameters():
        ...     print(f"{name}: {param.shape}")
        ... # weight: (5, 10)
        ... # bias: (5,)
        
    Note:
        - 使用Xavier初始化方法初始化权重
        - 支持任意维度的输入，只关心最后一维
        - 权重形状与PyTorch保持一致：(out_features, in_features)
        - 常用于分类器、回归器或网络中的特征变换
    """
    def __init__(self, in_features, out_features, bias=True, device=None, dtype:np.dtype|None=None):
        """
        初始化线性层 (Initialize Linear Layer)
        
        创建线性层实例，初始化权重和偏置参数。
        
        Args:
            in_features (int): 输入特征数量
            out_features (int): 输出特征数量  
            bias (bool, optional): 是否使用偏置项。默认值: True
            dtype (np.dtype, optional): 参数的数据类型。默认值: None（使用默认类型）
            device (str|int|Device, optional): 参数的设备。默认值: None（使用当前设备）
            
        Examples:
            >>> layer = Linear(10, 5)  # 输入10维，输出5维，使用偏置
            >>> layer_no_bias = Linear(10, 5, bias=False)  # 不使用偏置
            >>> 
            >>> # 查看参数形状
            >>> print(layer.weight.shape)  # (5, 10)
            >>> print(layer.bias.shape)    # (5,)
        """
        super().__init__()  # 必须显式调用父类初始化
        
        dt = get_default_dtype() if dtype is None else dtype

        # 与PyTorch保持一致：权重形状为 [out_features, in_features]
        # 先初始化权重张量为 (in_features, out_features) 形状，再转置为 (out_features, in_features) 形状的参数
        # 在forward里前向计算时，权重会再次转置为连续内存布局，这样前向计算性能会提升
        # 这么处理的目的是在与PyTorch保持一致的权重形状的前提下，提升前向计算的性能
        stdv = 1.0 / sqrt(in_features)
        w_para = randn(in_features, out_features, dtype=dt, device=device) * stdv
        self.weight = Parameter(w_para.mT)
        
        # 偏置处理需要完整注册逻辑
        if bias:
            b_para = randn(out_features,dtype=dt, device=device) * stdv
            self.bias = Parameter(b_para)
        else:
            self.register_parameter('bias', None)  # 显式注册空参数
            
    def forward(self, x):
        """
        线性层前向传播 (Linear Layer Forward Pass)
        
        执行线性变换：output = input @ weight.T + bias
        
        Args:
            x: 输入张量，形状为 (*, in_features)，*表示任意前导维度
            
        Returns:
            output: 输出张量，形状为 (*, out_features)
            
        Examples:
            >>> layer = Linear(10, 5)
            >>> x = rm.randn(32, 10)  # 批量输入
            >>> y = layer(x)          # 输出形状: (32, 5)
            >>> 
            >>> # 支持多维输入
            >>> x_3d = rm.randn(4, 8, 10)  # (batch, seq_len, features)
            >>> y_3d = layer(x_3d)         # 输出: (4, 8, 5)
            
        Note:
            - 只对最后一维进行线性变换
            - 使用矩阵乘法实现高效的批量计算
            - 当bias为None时只执行矩阵乘法
        """
        # 执行矩阵乘法: x @ weight.T + bias
        output = x @ self.weight.mT
        if self.bias is not None:
            output = output + self.bias
        return output
    
    def extra_repr(self) -> str:
        """
        返回线性层的额外表示信息 (Extra Representation)
        
        返回层的配置信息，用于__repr__方法中显示层的参数。
        
        Returns:
            str: 层的配置字符串
            
        Examples:
            >>> layer = Linear(10, 5, bias=True)
            >>> print(layer.extra_repr())
            in_features=10, out_features=5, bias=True
            >>> 
            >>> print(layer)  # 包含extra_repr信息
            Linear(in_features=10, out_features=5, bias=True)
        """
        in_features = self.weight.shape[1]
        out_features = self.weight.shape[0]
        return f'in_features={in_features}, out_features={out_features}, bias={self.bias is not None}'

# end of class Linear

class Sequential(Module):
    """
    顺序容器模块 (Sequential Container Module)
    
    将多个模块按顺序包装成一个模块，数据会依次通过所有子模块。
    这是构建神经网络时常用的容器，可以简化网络结构的定义。
    
    数据流向:
        input -> module_0 -> module_1 -> ... -> module_n -> output
        
    Args:
        *modules: 可变数量的模块参数，按顺序执行
        
    Attributes:
        子模块会自动注册为 '0', '1', '2', ... 等数字名称
        
    Examples:
        >>> # 使用位置参数创建
        >>> seq = Sequential(
        ...     Linear(10, 20),
        ...     ReLU(),
        ...     Linear(20, 5)
        ... )
        >>> 
        >>> # 前向传播
        >>> x = rm.randn(32, 10)
        >>> y = seq(x)  # 依次通过三个层
        >>> 
        >>> # 访问子模块
        >>> print(seq[0])  # 第一个层
        >>> print(seq[1])  # 第二个层
        >>> 
        >>> # 动态添加模块
        >>> seq.add_module('3', Dropout(0.5))
        >>> 
        >>> # 使用字典创建
        >>> modules = OrderedDict([
        ...     ('fc1', Linear(10, 20)),
        ...     ('relu', ReLU()),
        ...     ('fc2', Linear(20, 5))
        ... ])
        >>> seq = Sequential(modules)
        
    Note:
        - 子模块按添加顺序执行
        - 支持索引访问子模块
        - 常用于构建简单的线性网络结构
        - 可以与条件分支和循环结合使用
    """
    def __init__(self, *modules):
        """
        初始化顺序容器 (Initialize Sequential Container)
        
        创建顺序容器，添加所有给定的模块。
        
        Args:
            *modules: 可变数量的模块参数，可以是Module实例或包含模块的迭代器
            
        Examples:
            >>> # 直接传入模块
            >>> seq = Sequential(Linear(10, 20), ReLU(), Linear(20, 5))
            >>> 
            >>> # 使用列表
            >>> modules = [Linear(10, 20), ReLU(), Linear(20, 5)]
            >>> seq = Sequential(*modules)
            >>> 
            >>> # 使用OrderedDict
            >>> from collections import OrderedDict
            >>> modules = OrderedDict([('fc1', Linear(10, 20)), ('relu', ReLU())])
            >>> seq = Sequential(modules)
        """
        super().__init__()
        for idx, module in enumerate(modules):
            self.add_module(str(idx), module)

    def forward(self, x):
        """
        顺序容器的前向传播 (Sequential Forward Pass)
        
        将输入依次通过所有子模块进行处理。
        
        Args:
            x: 输入张量
            
        Returns:
            经过所有子模块处理后的输出张量
            
        Examples:
            >>> seq = Sequential(Linear(10, 20), ReLU(), Linear(20, 5))
            >>> x = rm.randn(32, 10)
            >>> y = seq(x)  # 等价于: Linear(20, 5)(ReLU(Linear(10, 20)(x)))
            
        Note:
            - 模块按注册顺序执行
            - 前一个模块的输出是后一个模块的输入
            - 支持任意数量的子模块
        """
        for i, module in enumerate(self.children()):
            x = module(x)
        return x

# end of class Sequential

class ModuleList(Module):
    """
    模块列表容器 (Module List Container)
    
    将子模块存储在列表中，支持索引访问、迭代和动态修改。
    与Sequential不同，ModuleList不定义前向传播逻辑，需要用户自行定义。
    
    主要用途:
        - 在网络中需要条件执行某些模块
        - 动态构建网络结构
        - 在循环中访问多个模块
        
    Args:
        modules (iterable, optional): 模块的迭代器。默认值: None
        
    Attributes:
        子模块以数字字符串 '0', '1', '2', ... 作为名称注册
        
    Examples:
        >>> # 创建模块列表
        >>> layers = ModuleList([
        ...     Linear(10, 20),
        ...     Linear(20, 30),
        ...     Linear(30, 5)
        ... ])
        >>> 
        >>> # 索引访问
        >>> layer1 = layers[0]
        >>> layer2 = layers[1]
        >>> 
        >>> # 迭代访问
        >>> for layer in layers:
        ...     print(layer)
        >>> 
        >>> # 动态添加模块
        >>> layers.append(Linear(5, 1))
        >>> 
        >>> # 在自定义forward中使用
        >>> class MyNet(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.layers = ModuleList([
        ...             Linear(10, 20),
        ...             ReLU(),
        ...             Linear(20, 5)
        ...         ])
        ...     
        ...     def forward(self, x):
        ...         for layer in self.layers:
        ...             x = layer(x)
        ...         return x
        
    Note:
        - 不自动定义前向传播逻辑
        - 支持列表操作：append, extend, 索引访问等
        - 子模块会自动注册，包含在parameters()中
        - 比Sequential更灵活，适用于复杂的网络结构
    """
    def __init__(self, modules=None):
        """
        初始化模块列表 (Initialize Module List)
        
        创建模块列表，可选择性地添加初始模块。
        
        Args:
            modules (iterable, optional): 包含Module对象的迭代器。默认值: None
            
        Examples:
            >>> # 空列表
            >>> layers = ModuleList()
            >>> 
            >>> # 从列表创建
            >>> modules = [Linear(10, 20), ReLU(), Linear(20, 5)]
            >>> layers = ModuleList(modules)
            >>> 
            >>> # 从生成器创建
            >>> layers = ModuleList(Linear(i, i+1) for i in range(5))
        """
        super().__init__()
        if modules is not None:
            self.extend(modules)

    def append(self, module):
        """
        在列表末尾添加模块 (Append Module)
        
        将新模块添加到模块列表的末尾。
        
        Args:
            module (Module): 要添加的模块
            
        Examples:
            >>> layers = ModuleList()
            >>> layers.append(Linear(10, 20))
            >>> layers.append(ReLU())
            >>> print(len(layers))  # 2
        """
        self.add_module(str(len(self)), module)

    def extend(self, modules):
        """
        扩展模块列表 (Extend Module List)
        
        将多个模块添加到模块列表的末尾。
        
        Args:
            modules (iterable): 包含Module对象的迭代器
            
        Examples:
            >>> layers = ModuleList([Linear(10, 20)])
            >>> new_modules = [ReLU(), Linear(20, 5)]
            >>> layers.extend(new_modules)
            >>> print(len(layers))  # 3
        """
        for module in modules:
            self.append(module)

    def __getitem__(self, idx):
        """
        索引访问模块 (Index Access)
        
        支持整数索引访问模块列表中的模块。
        
        Args:
            idx (int): 模块索引
            
        Returns:
            Module: 指定索引处的模块
            
        Examples:
            >>> layers = ModuleList([Linear(10, 20), ReLU(), Linear(20, 5)])
            >>> layer = layers[1]  # 获取ReLU层
            >>> first_layer = layers[0]  # 获取第一个Linear层
        """
        return list(self._modules.values())[idx]

    def __iter__(self):
        """
        迭代器支持 (Iterator Support)
        
        返回模块列表的迭代器，支持for循环遍历。
        
        Returns:
            iterator: 模块的迭代器
            
        Examples:
            >>> layers = ModuleList([Linear(10, 20), ReLU(), Linear(20, 5)])
            >>> for layer in layers:
            ...     print(type(layer).__name__)
        """
        return iter(self._modules.values())
        
    def __len__(self):
        """
        返回模块数量 (Length of Module List)
        
        返回模块列表中模块的数量。
        
        Returns:
            int: 模块数量
            
        Examples:
            >>> layers = ModuleList([Linear(10, 20), ReLU()])
            >>> print(len(layers))  # 2
        """
        return len(self._modules)

# end of class

class ModuleDict(Module):
    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            self.update(modules)

    def __setitem__(self, key, module):
        self.add_module(key, module)

    def __getitem__(self, key):
        return self._modules[key]

    def update(self, modules):
        if isinstance(modules, dict):
            for key, module in modules.items():
                self[key] = module
        else:
            raise TypeError("ModuleDict.update requires a dict")

    def keys(self):
        return self._modules.keys()

# end of class

class ParameterList(Module):
    """
    参数列表容器 (Parameter List Container)
    
    将参数存储在列表中，支持索引访问、迭代和动态修改。
    与ModuleList类似，但专门用于存储Parameter对象。
    
    主要用途:
        - 管理多个参数
        - 动态构建参数集合
        - 在循环中访问多个参数
        
    Args:
        parameters (iterable, optional): 参数的迭代器。可以是列表、元组或任何可迭代对象，
            其中的元素必须是Parameter对象。如果为None，则创建空列表。默认值: None
            
    Attributes:
        参数以数字字符串 '0', '1', '2', ... 作为名称注册到模块中，
        可以通过parameters()和named_parameters()方法访问
        
    Examples:
        >>> # 创建参数列表
        >>> params = ParameterList([
        ...     Parameter(rm.randn(10, 5)),
        ...     Parameter(rm.randn(5)),
        ...     Parameter(rm.randn(3, 3))
        ... ])
        >>> 
        >>> # 索引访问
        >>> weight = params[0]
        >>> bias = params[1]
        >>> 
        >>> # 迭代访问
        >>> for param in params:
        ...     print(param.shape)
        >>> 
        >>> # 动态添加参数
        >>> params.append(Parameter(rm.randn(1, 10)))
        >>> 
        >>> # 在自定义模块中使用
        >>> class MyModule(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.params = ParameterList([
        ...             Parameter(rm.randn(10, 20)),
        ...             Parameter(rm.randn(20))
        ...         ])
        ...     
        ...     def forward(self, x):
        ...         weight, bias = self.params
        ...         return x @ weight + bias
        
    Note:
        - 专门用于存储Parameter对象
        - 支持列表操作：append, extend, 索引访问等
        - 参数会自动注册，包含在parameters()中
        - 接口与torch.nn.ParameterList一致
    """
    def __init__(self, parameters=None):
        """
        初始化参数列表 (Initialize Parameter List)
        
        创建参数列表，可选择性地添加初始参数。
        
        Args:
            parameters (iterable, optional): 包含Parameter对象的迭代器。可以是列表、元组、
                生成器或任何可迭代对象。每个元素必须是Parameter类型。如果为None，
                则创建空的参数列表。默认值: None
                
        Raises:
            TypeError: 如果parameters中的元素不是Parameter对象
            
        Examples:
            >>> # 空列表
            >>> params = ParameterList()
            >>> 
            >>> # 从列表创建
            >>> params = ParameterList([
            ...     Parameter(rm.randn(10, 5)),
            ...     Parameter(rm.randn(5))
            ... ])
            >>> 
            >>> # 从生成器创建
            >>> params = ParameterList(Parameter(rm.randn(i, i+1)) for i in range(3))
        """
        super().__init__()
        if parameters is not None:
            self.extend(parameters)

    def append(self, parameter):
        """
        在列表末尾添加参数 (Append Parameter)
        
        将新参数添加到参数列表的末尾。参数会被自动注册到模块中，
        注册名称为当前列表长度的字符串形式（如'0', '1', '2'等）。
        
        Args:
            parameter (Parameter): 要添加的参数。必须是Parameter类型，
                不能是普通的Tensor或其他类型
                
        Raises:
            TypeError: 如果parameter不是Parameter对象
            
        Returns:
            None
            
        Examples:
            >>> params = ParameterList()
            >>> params.append(Parameter(rm.randn(10, 20)))
            >>> params.append(Parameter(rm.randn(20)))
            >>> print(len(params))  # 2
            >>> 
            >>> # 验证参数已注册
            >>> print(list(params.named_parameters()))  # [('0', Parameter(...)), ('1', Parameter(...))]
        """
        if not isinstance(parameter, Parameter):
            raise TypeError(f"ParameterList only accepts Parameter objects, got {type(parameter)}")
        self.register_parameter(str(len(self)), parameter)

    def extend(self, parameters):
        """
        扩展参数列表 (Extend Parameter List)
        
        将多个参数添加到参数列表的末尾。每个参数会被依次调用append()方法添加，
        因此会继承append()的所有特性，包括类型检查和自动注册。
        
        Args:
            parameters (iterable): 包含Parameter对象的迭代器。可以是列表、元组、
                生成器或任何可迭代对象。每个元素必须是Parameter类型
                
        Raises:
            TypeError: 如果parameters中的任何元素不是Parameter对象
            
        Returns:
            None
            
        Examples:
            >>> params = ParameterList([Parameter(rm.randn(10, 20))])
            >>> 
            >>> # 从列表扩展
            >>> new_params = [Parameter(rm.randn(20)), Parameter(rm.randn(20, 5))]
            >>> params.extend(new_params)
            >>> print(len(params))  # 3
            >>> 
            >>> # 从生成器扩展
            >>> params.extend(Parameter(rm.randn(i, i+1)) for i in range(2))
            >>> print(len(params))  # 5
        """
        for parameter in parameters:
            self.append(parameter)

    def __getitem__(self, idx):
        """
        索引访问参数 (Index Access)
        
        支持整数索引访问参数列表中的参数。索引从0开始，支持负数索引
        （如-1表示最后一个参数）。
        
        Args:
            idx (int): 参数索引。必须是整数，范围在[-len(self), len(self)-1]之间
            
        Returns:
            Parameter: 指定索引处的参数
            
        Raises:
            IndexError: 如果索引超出范围
            TypeError: 如果idx不是整数类型
            
        Examples:
            >>> params = ParameterList([
            ...     Parameter(rm.randn(10, 20)),
            ...     Parameter(rm.randn(20)),
            ...     Parameter(rm.randn(20, 5))
            ... ])
            >>> weight = params[0]  # 获取第一个参数
            >>> bias = params[1]  # 获取第二个参数
            >>> last = params[-1]  # 获取最后一个参数（负数索引）
        """
        if not isinstance(idx, int):
            raise TypeError(f"ParameterList indices must be integers, not {type(idx)}")
        return list(self._parameters.values())[idx]

    def __iter__(self):
        """
        迭代器支持 (Iterator Support)
        
        返回参数列表的迭代器，支持for循环遍历。
        
        Returns:
            iterator: 参数的迭代器
            
        Examples:
            >>> params = ParameterList([
            ...     Parameter(rm.randn(10, 20)),
            ...     Parameter(rm.randn(20))
            ... ])
            >>> for param in params:
            ...     print(param.shape)
        """
        return iter(self._parameters.values())

    def __len__(self):
        """
        获取参数列表长度 (Get Length)
        
        Returns:
            int: 参数列表的长度
            
        Examples:
            >>> params = ParameterList([
            ...     Parameter(rm.randn(10, 20)),
            ...     Parameter(rm.randn(20))
            ... ])
            >>> print(len(params))  # 2
        """
        return len(self._parameters)

# end of class

class ParameterDict(Module):
    """
    参数字典容器 (Parameter Dictionary Container)
    
    将参数存储在字典中，支持按名称访问和动态修改。
    与ModuleDict类似，但专门用于存储Parameter对象。
    
    主要用途:
        - 按名称管理参数
        - 动态构建命名参数集合
        - 方便地访问特定参数
        
    Args:
        parameters (dict, optional): 参数的字典。键必须是字符串类型，
            值必须是Parameter对象。如果为None，则创建空的参数字典。默认值: None
            
    Attributes:
        参数以用户指定的键作为名称注册到模块中，
        可以通过parameters()和named_parameters()方法访问
        
    Examples:
        >>> # 创建参数字典
        >>> params = ParameterDict({
        ...     'weight': Parameter(rm.randn(10, 5)),
        ...     'bias': Parameter(rm.randn(5)),
        ...     'scale': Parameter(rm.randn(1))
        ... })
        >>> 
        >>> # 按名称访问
        >>> weight = params['weight']
        >>> bias = params['bias']
        >>> 
        >>> # 迭代访问
        >>> for name, param in params.items():
        ...     print(f"{name}: {param.shape}")
        >>> 
        >>> # 动态添加参数
        >>> params['new_param'] = Parameter(rm.randn(3, 3))
        >>> 
        >>> # 在自定义模块中使用
        >>> class MyModule(Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.params = ParameterDict({
        ...             'w1': Parameter(rm.randn(10, 20)),
        ...             'b1': Parameter(rm.randn(20)),
        ...             'w2': Parameter(rm.randn(20, 5)),
        ...             'b2': Parameter(rm.randn(5))
        ...         })
        ...     
        ...     def forward(self, x):
        ...         x = x @ self.params['w1'] + self.params['b1']
        ...         x = x @ self.params['w2'] + self.params['b2']
        ...         return x
        
    Note:
        - 专门用于存储Parameter对象
        - 支持字典操作：__setitem__, __getitem__, update等
        - 参数会自动注册，包含在parameters()中
        - 接口与torch.nn.ParameterDict一致
    """
    def __init__(self, parameters=None):
        """
        初始化参数字典 (Initialize Parameter Dict)
        
        创建参数字典，可选择性地添加初始参数。
        
        Args:
            parameters (dict, optional): 包含Parameter对象的字典。键必须是字符串类型，
                值必须是Parameter类型。如果为None，则创建空的参数字典。默认值: None
                
        Raises:
            TypeError: 如果parameters不是字典类型，或键不是字符串，或值不是Parameter对象
            
        Examples:
            >>> # 空字典
            >>> params = ParameterDict()
            >>> 
            >>> # 从字典创建
            >>> params = ParameterDict({
            ...     'weight': Parameter(rm.randn(10, 5)),
            ...     'bias': Parameter(rm.randn(5))
            ... })
            >>> 
            >>> # 验证参数已注册
            >>> print(list(params.named_parameters()))  # [('weight', Parameter(...)), ('bias', Parameter(...))]
        """
        super().__init__()
        if parameters is not None:
            self.update(parameters)

    def __setitem__(self, key, parameter):
        """
        设置参数 (Set Parameter)
        
        按键设置参数。参数会被自动注册到模块中，注册名称为指定的键。
        
        Args:
            key (str): 参数键。必须是字符串类型，作为参数的名称
            parameter (Parameter): 要设置的参数。必须是Parameter类型，
                不能是普通的Tensor或其他类型
                
        Raises:
            TypeError: 如果key不是字符串类型，或parameter不是Parameter对象
            
        Returns:
            None
            
        Examples:
            >>> params = ParameterDict()
            >>> params['weight'] = Parameter(rm.randn(10, 5))
            >>> params['bias'] = Parameter(rm.randn(5))
            >>> 
            >>> # 使用变量名作为键
            >>> w_key = 'encoder_weight'
            >>> params[w_key] = Parameter(rm.randn(20, 10))
            >>> print(w_key in params)  # True
        """
        if not isinstance(key, str):
            raise TypeError(f"ParameterDict keys must be strings, not {type(key)}")
        if not isinstance(parameter, Parameter):
            raise TypeError(f"ParameterDict only accepts Parameter objects, got {type(parameter)}")
        self.register_parameter(key, parameter)

    def __getitem__(self, key):
        """
        按键获取参数 (Get Parameter)
        
        通过键获取对应的参数。如果键不存在，会抛出KeyError。
        
        Args:
            key (str): 参数键。必须是字符串类型，且必须存在于字典中
            
        Returns:
            Parameter: 指定键的参数
            
        Raises:
            KeyError: 如果指定的键不存在于字典中
            TypeError: 如果key不是字符串类型
            
        Examples:
            >>> params = ParameterDict({
            ...     'weight': Parameter(rm.randn(10, 5)),
            ...     'bias': Parameter(rm.randn(5))
            ... })
            >>> weight = params['weight']
            >>> bias = params['bias']
            >>> 
            >>> # 使用变量名作为键
            >>> w_key = 'encoder_weight'
            >>> params[w_key] = Parameter(rm.randn(20, 10))
            >>> encoder_w = params[w_key]
        """
        if not isinstance(key, str):
            raise TypeError(f"ParameterDict keys must be strings, not {type(key)}")
        return self._parameters[key]

    def update(self, parameters):
        """
        更新参数字典 (Update Parameter Dict)
        
        从字典更新参数。对于字典中的每个键值对，会调用__setitem__方法添加参数，
        因此会继承__setitem__的所有特性，包括类型检查和自动注册。
        
        Args:
            parameters (dict): 包含Parameter对象的字典。键必须是字符串类型，
                值必须是Parameter类型。如果键已存在，会覆盖原有参数
                
        Raises:
            TypeError: 如果parameters不是字典类型，或键不是字符串，或值不是Parameter对象
            
        Returns:
            None
            
        Examples:
            >>> params = ParameterDict({
            ...     'weight': Parameter(rm.randn(10, 5))
            ... })
            >>> 
            >>> # 批量添加新参数
            >>> new_params = {
            ...     'bias': Parameter(rm.randn(5)),
            ...     'scale': Parameter(rm.randn(1))
            ... }
            >>> params.update(new_params)
            >>> print(len(params))  # 3
            >>> 
            >>> # 覆盖已有参数
            >>> params.update({'weight': Parameter(rm.randn(10, 5))})  # 覆盖原有的weight
        """
        if not isinstance(parameters, dict):
            raise TypeError("ParameterDict.update requires a dict")
        for key, parameter in parameters.items():
            self[key] = parameter

    def keys(self):
        """
        获取所有参数键 (Get Keys)
        
        返回包含所有参数键的视图对象。返回的是dict_keys对象，
        支持迭代和成员检查，但不支持索引访问。
        
        Returns:
            dict_keys: 参数键的视图，包含所有字符串类型的键
            
        Examples:
            >>> params = ParameterDict({
            ...     'weight': Parameter(rm.randn(10, 5)),
            ...     'bias': Parameter(rm.randn(5))
            ... })
            >>> print(list(params.keys()))  # ['weight', 'bias']
            >>> 
            >>> # 成员检查
            >>> 'weight' in params.keys()  # True
            >>> 'nonexistent' in params.keys()  # False
        """
        return self._parameters.keys()

    def items(self):
        """
        获取所有参数项 (Get Items)
        
        返回包含所有参数键值对的视图对象。返回的是dict_items对象，
        每个元素是一个(key, value)元组，其中key是字符串，value是Parameter对象。
        
        Returns:
            dict_items: 参数项的视图，包含(key, Parameter)元组
            
        Examples:
            >>> params = ParameterDict({
            ...     'weight': Parameter(rm.randn(10, 5)),
            ...     'bias': Parameter(rm.randn(5))
            ... })
            >>> for name, param in params.items():
            ...     print(f"{name}: {param.shape}")
            >>> 
            >>> # 转换为列表
            >>> items_list = list(params.items())
            >>> print(items_list)  # [('weight', Parameter(...)), ('bias', Parameter(...))]
        """
        return self._parameters.items()

    def values(self):
        """
        获取所有参数值 (Get Values)
        
        返回包含所有参数值的视图对象。返回的是dict_values对象，
        包含所有的Parameter对象，支持迭代但不支持索引访问。
        
        Returns:
            dict_values: 参数值的视图，包含所有Parameter对象
            
        Examples:
            >>> params = ParameterDict({
            ...     'weight': Parameter(rm.randn(10, 5)),
            ...     'bias': Parameter(rm.randn(5))
            ... })
            >>> for param in params.values():
            ...     print(param.shape)
            >>> 
            >>> # 转换为列表
            >>> values_list = list(params.values())
            >>> print(len(values_list))  # 2
        """
        return self._parameters.values()

    def __iter__(self):
        """
        迭代器支持 (Iterator Support)
        
        返回参数字典键的迭代器，支持for循环遍历。迭代顺序与keys()方法一致。
        
        Returns:
            iterator: 参数键（字符串类型）的迭代器
            
        Examples:
            >>> params = ParameterDict({
            ...     'weight': Parameter(rm.randn(10, 5)),
            ...     'bias': Parameter(rm.randn(5))
            ... })
            >>> for name in params:
            ...     print(name)
            >>> 
            >>> # 等价于
            >>> for name in params.keys():
            ...     print(name)
        """
        return iter(self._parameters)

    def __len__(self):
        """
        获取参数字典长度 (Get Length)
        
        返回参数字典中键值对的数量，即参数的总数。
        
        Returns:
            int: 参数字典中参数的数量
            
        Examples:
            >>> params = ParameterDict({
            ...     'weight': Parameter(rm.randn(10, 5)),
            ...     'bias': Parameter(rm.randn(5))
            ... })
            >>> print(len(params))  # 2
            >>> 
            >>> # 添加新参数后
            >>> params['scale'] = Parameter(rm.randn(1))
            >>> print(len(params))  # 3
        """
        return len(self._parameters)

# end of class

class Dropout(Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability p must be between 0 and 1, got {p}")
        self.p = p
        self.inplace = inplace
    
    def forward(self, x):
        # 调用dropout函数
        return dropout(x, self.p, self.training, self.inplace)

# end of class

class Dropout2d(Module):
    """2D dropout模块 (Dropout2d Layer)
    
    在训练期间，以概率p随机将输入张量的整个通道置为0，并对剩余通道乘以1/(1-p)进行缩放。
    在评估期间，不执行任何操作。
    
    接口与torch.nn.Dropout2d完全一致
    
    Args:
        p (float, optional): dropout概率，默认为0.5
        inplace (bool, optional): 是否原地操作，默认为False
    """
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability p must be between 0 and 1, got {p}")
        self.p = p
        self.inplace = inplace
    
    def forward(self, x):
        # 调用dropout2d函数
        return dropout2d(x, self.p, self.training, self.inplace)

# end of class

class Dropout3d(Module):
    """3D dropout模块 (Dropout3d Layer)
    
    在训练期间，以概率p随机将输入张量的整个通道置为0，并对剩余通道乘以1/(1-p)进行缩放。
    在评估期间，不执行任何操作。
    
    接口与torch.nn.Dropout3d完全一致
    
    Args:
        p (float, optional): dropout概率，默认为0.5
        inplace (bool, optional): 是否原地操作，默认为False
    """
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability p must be between 0 and 1, got {p}")
        self.p = p
        self.inplace = inplace
    
    def forward(self, x):
        # 调用dropout3d函数
        return dropout3d(x, self.p, self.training, self.inplace)

# end of class

class Flatten(Module):
    """展平模块 (Flatten Layer)
    
    对输入张量在指定维度范围内进行展平操作，将多个维度合并为一个维度。
    常用于将卷积层的多维输出转换为全连接层所需的一维或二维输入。
    
    数学公式::
    
        output_shape = input_shape[:start_dim] + [product(input_shape[start_dim:end_dim+1])] + input_shape[end_dim+1:]
        
        其中 product() 表示计算所有元素的乘积
    
    Args:
        start_dim (int, optional): 开始展平的维度。默认值: 1
        end_dim (int, optional): 结束展平的维度。默认值: -1
    
    Shape:
        - Input: 任意形状的张量 (*)
        - Output: 展平后的张量，形状根据start_dim和end_dim参数确定
    
    Examples::
    
        >>> # 展平除批次维度外的所有维度
        >>> m = Flatten()
        >>> input = rm.randn(4, 3, 224, 224)  # batch_size=4, channels=3, height=224, width=224
        >>> output = m(input)
        >>> print(output.shape)  # (4, 150528) 其中 150528 = 3*224*224
        
        >>> # 展平指定维度范围
        >>> m = Flatten(start_dim=2, end_dim=3)
        >>> input = rm.randn(8, 16, 32, 32, 64)  # batch_size=8, channels=16, height=32, width=32, depth=64
        >>> output = m(input)
        >>> print(output.shape)  # (8, 16, 1024, 64) 其中 1024 = 32*32
        
        >>> # 展平所有维度
        >>> m = Flatten(start_dim=0)
        >>> input = rm.randn(2, 3, 4)
        >>> output = m(input)
        >>> print(output.shape)  # (24,) 其中 24 = 2*3*4
    
    Note:
        Flatten模块的特点和用途：
        - 是神经网络架构中的形状变换层，不涉及数值计算
        - 主要用于连接卷积层和全连接层，将多维特征图转换为一维特征向量
        - start_dim=1是默认值，保留了批次维度，符合深度学习的常见使用模式
        - 支持负数维度索引，与Python和NumPy的索引规则一致
        - 展平操作保持数据的连续性，不改变数据本身的值
        - 在CNN架构中不可或缺，如LeNet、AlexNet、VGG等经典网络都使用展平层
        - 可以灵活控制展平范围，适应不同的网络架构需求
        - 与reshape操作相比，展平更加语义明确，专门用于维度合并
        - 在现代网络设计中，有时被Global Average Pooling替代，但仍有广泛应用
    """
    
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量，任意形状
            
        Returns:
            展平后的张量
        """
        # 处理边界情况：如果输入张量维度不足，直接返回原张量
        if x.ndim <= 1:
            return x
        
        # 调整start_dim和end_dim以确保在有效范围内
        dim = x.ndim
        start_dim = self.start_dim if self.start_dim >= 0 else dim + self.start_dim
        end_dim = self.end_dim if self.end_dim >= 0 else dim + self.end_dim
        
        # 确保维度范围有效
        if start_dim < 0:
            start_dim = 0
        if end_dim >= dim:
            end_dim = dim - 1
        if start_dim > end_dim:
            return x
            
        return x.flatten(start_dim, end_dim)

# end of class

class BatchNorm1d(Module):
    """
    一维批量归一化层 (1D Batch Normalization Layer)
    
    对2D或3D输入张量的通道维度进行归一化，使每个通道的特征分布具有零均值和单位方差。
    批量归一化可以加速训练收敛，提高模型泛化能力，并允许使用更高的学习率。
    
    批量归一化的计算过程：
    1. 计算当前批次每个通道的均值和方差
    2. 使用均值和方差对输入进行归一化
    3. 使用可学习的缩放因子γ和偏移因子β进行线性变换
    
    数学公式::
    
        y = γ * (x - μ) / √(σ² + ε) + β
        
        其中：
        - μ: 当前批次的均值
        - σ²: 当前批次的方差
        - γ: 可学习的缩放参数(weight)
        - β: 可学习的偏移参数(bias)
        - ε: 数值稳定性参数
    
    Args:
        num_features (int): 特征数量(通道数C)
        eps (float, optional): 数值稳定性的小常数。默认值: 1e-5
        momentum (float, optional): 运行时均值和方差的动量。默认值: 0.1
        affine (bool, optional): 是否使用可学习的仿射参数γ和β。默认值: True
        track_running_stats (bool, optional): 是否跟踪运行时均值和方差。默认值: True
        device (optional): 参数和缓冲区的设备，默认为None
        dtype (optional): 参数和缓冲区的数据类型，默认为None
    
    Shape:
        - Input: (N, C) 或 (N, C, L) 批次大小N，通道数C，序列长度L(可选)
        - Output: (N, C) 或 (N, C, L) 与输入形状相同
    
    Attributes:
        weight (Parameter): 可学习的缩放参数γ，形状为(num_features,)
        bias (Parameter): 可学习的偏移参数β，形状为(num_features,)
        running_mean (TN): 运行时均值，形状为(num_features,)
        running_var (TN): 运行时方差，形状为(num_features,)
        num_batches_tracked (TN): 跟踪的批次数，用于动量计算
    
    Examples::
    
        >>> # 带有可学习参数的批量归一化
        >>> m = BatchNorm1d(100)
        >>> input = rm.randn(20, 100)
        >>> output = m(input)
        >>> 
        >>> # 3D输入的批量归一化
        >>> m = BatchNorm1d(100)
        >>> input = rm.randn(20, 100, 35)
        >>> output = m(input)
        >>> 
        >>> # 不带可学习参数的批量归一化
        >>> m = BatchNorm1d(100, affine=False)
        >>> 
        >>> # 不跟踪运行时统计量的批量归一化
        >>> m = BatchNorm1d(100, track_running_stats=False)
    
    Note:
        批量归一化的特点和注意事项：
        - 在训练和评估模式下行为不同：训练时使用当前批次统计量，评估时使用运行时统计量
        - 可以显著减少对初始化的敏感性，允许使用更高的学习率
        - 在全连接层或RNN中通常放在线性变换之后、激活函数之前
        - 小批次情况下效果可能不稳定，可以考虑使用其他归一化方法
        - 与Dropout有类似的正则化效果，可以减少对Dropout的需求
        - 在训练初期会引入噪声，有助于模型的泛化能力
        - 运行时统计量使用指数移动平均计算，动量参数控制更新速度
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, 
                 track_running_stats=True, device=None, dtype=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        # 创建可学习参数
        if self.affine:
            self.weight = Parameter(ones(num_features, device=device, dtype=dtype))
            self.bias = Parameter(zeros(num_features, device=device, dtype=dtype))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        # 创建运行时统计量缓冲区
        if self.track_running_stats:
            self.register_buffer('running_mean', zeros(num_features, device=device, dtype=dtype))
            self.register_buffer('running_var', ones(num_features, device=device, dtype=dtype))
            self.register_buffer('num_batches_tracked', tensor(0, dtype=int32, device=device))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
    
    def forward(self, input):
        """
        前向传播
        
        Args:
            input: 输入张量，形状为(N, C)或(N, C, L)
            
        Returns:
            归一化后的张量，形状与输入相同
        """
                
        # 检查输入形状
        if input.ndim not in [2, 3]:
            raise ValueError(f"BatchNorm1d expects 2D or 3D input (N, C) or (N, C, L), got {input.ndim}D input")
        
        if input.shape[1] != self.num_features:
            raise ValueError(f"Expected input to have {self.num_features} channels, got {input.shape[1]}")
        
        # 使用functional模块中的batch_norm函数
        return batch_norm(
            input=input,
            running_mean=self.running_mean if self.track_running_stats else None,
            running_var=self.running_var if self.track_running_stats else None,
            weight=self.weight if self.affine else None,
            bias=self.bias if self.affine else None,
            training=self.training,
            momentum=self.momentum,
            eps=self.eps
        )
    
    def extra_repr(self):
        """
        模块的额外表示信息
        
        Returns:
            str: 模块的描述字符串
        """
        return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}, track_running_stats={self.track_running_stats}"

# end of class BatchNorm1d

class BatchNorm2d(Module):
    """
    二维批量归一化层 (2D Batch Normalization Layer)
    
    对4D输入张量(N, C, H, W)的通道维度进行归一化，使每个通道的特征分布具有零均值和单位方差。
    批量归一化可以加速训练收敛，提高模型泛化能力，并允许使用更高的学习率。
    
    批量归一化的计算过程：
    1. 计算当前批次每个通道的均值和方差
    2. 使用均值和方差对输入进行归一化
    3. 使用可学习的缩放因子γ和偏移因子β进行线性变换
    
    数学公式::
    
        y = γ * (x - μ) / √(σ² + ε) + β
        
        其中：
        - μ: 当前批次的均值
        - σ²: 当前批次的方差
        - γ: 可学习的缩放参数(weight)
        - β: 可学习的偏移参数(bias)
        - ε: 数值稳定性参数
    
    Args:
        num_features (int): 特征数量(通道数C)
        eps (float, optional): 数值稳定性的小常数。默认值: 1e-5
        momentum (float, optional): 运行时均值和方差的动量。默认值: 0.1
        affine (bool, optional): 是否使用可学习的仿射参数γ和β。默认值: True
        track_running_stats (bool, optional): 是否跟踪运行时均值和方差。默认值: True
        device (optional): 参数和缓冲区的设备，默认为None
        dtype (optional): 参数和缓冲区的数据类型，默认为None
    
    Shape:
        - Input: (N, C, H, W) 批次大小N，通道数C，高度H，宽度W
        - Output: (N, C, H, W) 与输入形状相同
    
    Attributes:
        weight (Parameter): 可学习的缩放参数γ，形状为(num_features,)
        bias (Parameter): 可学习的偏移参数β，形状为(num_features,)
        running_mean (TN): 运行时均值，形状为(num_features,)
        running_var (TN): 运行时方差，形状为(num_features,)
        num_batches_tracked (TN): 跟踪的批次数，用于动量计算
    
    Examples::
    
        >>> # 带有可学习参数的批量归一化
        >>> m = BatchNorm2d(100)
        >>> input = rm.randn(20, 100, 35, 35)
        >>> output = m(input)
        >>> 
        >>> # 不带可学习参数的批量归一化
        >>> m = BatchNorm2d(100, affine=False)
        >>> 
        >>> # 不跟踪运行时统计量的批量归一化
        >>> m = BatchNorm2d(100, track_running_stats=False)
    
    Note:
        批量归一化的特点和注意事项：
        - 在训练和评估模式下行为不同：训练时使用当前批次统计量，评估时使用运行时统计量
        - 可以显著减少对初始化的敏感性，允许使用更高的学习率
        - 在CNN中通常放在卷积层之后、激活函数之前
        - 小批次情况下效果可能不稳定，可以考虑使用其他归一化方法
        - 与Dropout有类似的正则化效果，可以减少对Dropout的需求
        - 在训练初期会引入噪声，有助于模型的泛化能力
        - 运行时统计量使用指数移动平均计算，动量参数控制更新速度
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, 
                 track_running_stats=True, device=None, dtype=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        # 创建可学习参数
        if self.affine:
            self.weight = Parameter(ones(num_features, device=device, dtype=dtype))
            self.bias = Parameter(zeros(num_features, device=device, dtype=dtype))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        # 创建运行时统计量缓冲区
        if self.track_running_stats:
            self.register_buffer('running_mean', zeros(num_features, device=device, dtype=dtype))
            self.register_buffer('running_var', ones(num_features, device=device, dtype=dtype))
            self.register_buffer('num_batches_tracked', tensor(0, dtype=int32, device=device))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
    
    def forward(self, input):
        """
        前向传播
        
        Args:
            input: 输入张量，形状为(N, C, H, W)
            
        Returns:
            归一化后的张量，形状与输入相同
        """
        
        # 检查输入形状
        if input.ndim != 4:
            raise ValueError(f"BatchNorm2d expects 4D input (N, C, H, W), got {input.ndim}D input")
        
        if input.shape[1] != self.num_features:
            raise ValueError(f"Expected input to have {self.num_features} channels, got {input.shape[1]}")
        
        # 使用functional模块中的batch_norm2d函数
        return batch_norm(
            input=input,
            running_mean=self.running_mean if self.track_running_stats else None,
            running_var=self.running_var if self.track_running_stats else None,
            weight=self.weight if self.affine else None,
            bias=self.bias if self.affine else None,
            training=self.training,
            momentum=self.momentum,
            eps=self.eps
        )
    
    def extra_repr(self):
        """
        模块的额外表示信息
        
        Returns:
            str: 模块的描述字符串
        """
        return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}, track_running_stats={self.track_running_stats}"

# end of class BatchNorm2d

class BatchNorm3d(Module):
    """
    三维批量归一化层 (3D Batch Normalization Layer)
    
    对5D输入张量(N, C, D, H, W)的通道维度进行归一化，使每个通道的特征分布具有零均值和单位方差。
    批量归一化可以加速训练收敛，提高模型泛化能力，并允许使用更高的学习率。
    
    批量归一化的计算过程：
    1. 计算当前批次每个通道的均值和方差
    2. 使用均值和方差对输入进行归一化
    3. 使用可学习的缩放因子γ和偏移因子β进行线性变换
    
    数学公式::
    
        y = γ * (x - μ) / √(σ² + ε) + β
        
        其中：
        - μ: 当前批次的均值
        - σ²: 当前批次的方差
        - γ: 可学习的缩放参数(weight)
        - β: 可学习的偏移参数(bias)
        - ε: 数值稳定性参数
    
    Args:
        num_features (int): 特征数量(通道数C)
        eps (float, optional): 数值稳定性的小常数。默认值: 1e-5
        momentum (float, optional): 运行时均值和方差的动量。默认值: 0.1
        affine (bool, optional): 是否使用可学习的仿射参数γ和β。默认值: True
        track_running_stats (bool, optional): 是否跟踪运行时均值和方差。默认值: True
        device (optional): 参数和缓冲区的设备，默认为None
        dtype (optional): 参数和缓冲区的数据类型，默认为None
    
    Shape:
        - Input: (N, C, D, H, W) 批次大小N，通道数C，深度D，高度H，宽度W
        - Output: (N, C, D, H, W) 与输入形状相同
    
    Attributes:
        weight (Parameter): 可学习的缩放参数γ，形状为(num_features,)
        bias (Parameter): 可学习的偏移参数β，形状为(num_features,)
        running_mean (TN): 运行时均值，形状为(num_features,)
        running_var (TN): 运行时方差，形状为(num_features,)
        num_batches_tracked (TN): 跟踪的批次数，用于动量计算
    
    Examples::
    
        >>> # 带有可学习参数的批量归一化
        >>> m = BatchNorm3d(100)
        >>> input = rm.randn(20, 100, 35, 45, 55)
        >>> output = m(input)
        >>> 
        >>> # 不带可学习参数的批量归一化
        >>> m = BatchNorm3d(100, affine=False)
        >>> 
        >>> # 不跟踪运行时统计量的批量归一化
        >>> m = BatchNorm3d(100, track_running_stats=False)
    
    Note:
        批量归一化的特点和注意事项：
        - 在训练和评估模式下行为不同：训练时使用当前批次统计量，评估时使用运行时统计量
        - 可以显著减少对初始化的敏感性，允许使用更高的学习率
        - 在3D CNN中通常放在卷积层之后、激活函数之前
        - 小批次情况下效果可能不稳定，可以考虑使用其他归一化方法
        - 与Dropout有类似的正则化效果，可以减少对Dropout的需求
        - 在训练初期会引入噪声，有助于模型的泛化能力
        - 运行时统计量使用指数移动平均计算，动量参数控制更新速度
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, 
                 track_running_stats=True, device=None, dtype=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        # 创建可学习参数
        if self.affine:
            self.weight = Parameter(ones(num_features, device=device, dtype=dtype))
            self.bias = Parameter(zeros(num_features, device=device, dtype=dtype))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        # 创建运行时统计量缓冲区
        if self.track_running_stats:
            self.register_buffer('running_mean', zeros(num_features, device=device, dtype=dtype))
            self.register_buffer('running_var', ones(num_features, device=device, dtype=dtype))
            self.register_buffer('num_batches_tracked', tensor(0, dtype=int32, device=device))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
    
    def forward(self, input):
        """
        前向传播
        
        Args:
            input: 输入张量，形状为(N, C, D, H, W)
            
        Returns:
            归一化后的张量，形状与输入相同
        """
                
        # 检查输入形状
        if input.ndim != 5:
            raise ValueError(f"BatchNorm3d expects 5D input (N, C, D, H, W), got {input.ndim}D input")
        
        if input.shape[1] != self.num_features:
            raise ValueError(f"Expected input to have {self.num_features} channels, got {input.shape[1]}")
        
        # 使用functional模块中的batch_norm函数
        return batch_norm(
            input=input,
            running_mean=self.running_mean if self.track_running_stats else None,
            running_var=self.running_var if self.track_running_stats else None,
            weight=self.weight if self.affine else None,
            bias=self.bias if self.affine else None,
            training=self.training,
            momentum=self.momentum,
            eps=self.eps
        )
    
    def extra_repr(self):
        """
        模块的额外表示信息
        
        Returns:
            str: 模块的描述字符串
        """
        return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}, track_running_stats={self.track_running_stats}"

# end of class BatchNorm3d

class LayerNorm(Module):
    """层归一化层，对指定维度进行归一化处理
    与 torch.nn.LayerNorm 兼容
    
    参数:
        normalized_shape: 一个整数或元组，指定需要归一化的维度
        eps: 一个很小的常数，添加到方差中以避免除零错误
        affine: 如果为 True，将使用可学习的参数(gamma 和 beta)进行仿射变换
        device: 参数和缓冲区的设备，默认为 None
        dtype: 参数和缓冲区的数据类型，默认为 None
    """
    
    def __init__(self, normalized_shape, eps=1e-05, affine=True, device=None, dtype=None):
        super().__init__()
        self.normalized_shape = normalized_shape if isinstance(normalized_shape, (tuple, list)) else (normalized_shape,)
        self.eps = eps
        self.affine = affine
        
        # 如果启用仿射变换，创建可学习参数
        if self.affine:
            self.weight = Parameter(ones(self.normalized_shape, device=device, dtype=dtype))
            self.bias = Parameter(zeros(self.normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, input):
        """前向传播
        
        参数:
            input: 需要归一化的输入张量
        
        返回值:
            归一化后的张量，形状与输入一致
        """
        return layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
    
    def extra_repr(self):
        """
        返回 LayerNorm 模块的字符串表示
        """
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}, affine={self.affine}"

class Embedding(Module):
    """
    嵌入层 (Embedding Layer)
    
    将整数索引转换为固定大小的密集向量表示。
    嵌入层是神经网络中处理分类特征和序列数据的基础组件。
    
    参数:
        num_embeddings (int): 嵌入向量的数量，即词典大小
        embedding_dim (int): 每个嵌入向量的维度
        padding_idx (int, optional): 如果指定，该索引的嵌入向量不参与梯度计算，
                                     且在训练过程中保持不变。默认为None
        max_norm (float, optional): 如果指定，所有嵌入向量的范数超过max_norm时，
                                    将被重归一化到max_norm。默认为None
        norm_type (float, optional): 计算范数时使用的p值，默认为2（L2范数）
        scale_grad_by_freq (bool, optional): 如果为True，梯度将按mini-batch中每个词的频率进行缩放。
                                             默认为False
        sparse (bool, optional): 如果为True，权重的梯度将是稀疏张量。默认为False
        dtype (np.dtype, optional): 嵌入权重的数据类型。默认为None（使用默认类型）
        device (str|int|Device, optional): 嵌入权重的设备。默认为None（使用当前设备）
    
    形状:
        - 输入: (*) 包含整数索引的任意维度张量
        - 输出: (*, embedding_dim) 嵌入向量张量
    
    示例:
        >>> # 创建一个嵌入层，词典大小为10，嵌入维度为3
        >>> embedding = Embedding(10, 3)
        >>> input = rm.tensor([1, 2, 3])
        >>> output = embedding(input)
        >>> output.shape
        (3, 3)
        
        >>> # 指定padding_idx
        >>> embedding = Embedding(10, 3, padding_idx=0)
        >>> input = rm.tensor([0, 2, 3])
        >>> output = embedding(input)
        >>> # padding_idx=0的嵌入向量将不参与梯度计算
    
    注意:
        - 与PyTorch的nn.Embedding接口兼容
        - 嵌入权重是随机初始化的
        - padding_idx的嵌入向量在初始化时被设为0
        - 支持max_norm限制嵌入向量的范数
        - 支持scale_grad_by_freq按频率缩放梯度
        - 目前不支持sparse参数（会忽略该参数）
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None, 
                 max_norm: Optional[float] = None, norm_type: float = 2.0, scale_grad_by_freq: bool = False, 
                 sparse: bool = False, device=None,dtype: np.dtype|None = None):
        """
        初始化嵌入层
        
        参数:
            num_embeddings (int): 嵌入向量的数量
            embedding_dim (int): 每个嵌入向量的维度
            padding_idx (int, optional): 填充索引
            max_norm (float, optional): 嵌入向量的最大范数
            norm_type (float, optional): 计算范数的p值
            scale_grad_by_freq (bool, optional): 是否按频率缩放梯度
            sparse (bool, optional): 是否使用稀疏梯度
            dtype (np.dtype, optional): 嵌入权重的数据类型
            device (str|int|Device, optional): 嵌入权重的设备
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        
        # 设置数据类型和设备
        dt = get_default_dtype() if dtype is None else dtype
        
        # 创建嵌入权重，形状为(num_embeddings, embedding_dim)
        self.weight = Parameter(randn(num_embeddings, embedding_dim, dtype=dt, device=device))
        
        # 如果指定了padding_idx，将其嵌入向量设为0
        if padding_idx is not None:
            if padding_idx < 0:
                padding_idx += num_embeddings
            if padding_idx >= num_embeddings or padding_idx < 0:
                raise ValueError(f"padding_idx ({padding_idx}) must be within num_embeddings ({num_embeddings})")
            self.weight.data[padding_idx] = 0.0  # type: ignore  # type: ignore  # type: ignore
    
    def forward(self, input: TN) -> TN:
        """
        前向传播
        
        参数:
            input: 包含索引的张量，形状为任意维度
            
        返回:
            嵌入向量张量，形状为(*, embedding_dim)
        """
        return embedding(
            input=input,
            weight=self.weight,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse
        )
    
    def extra_repr(self) -> str:
        """
        模块的额外表示信息
        
        返回:
            str: 模块的描述字符串
        """
        s = f"{self.num_embeddings}, {self.embedding_dim}"
        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"
        if self.max_norm is not None:
            s += f", max_norm={self.max_norm}"
        if self.norm_type != 2:
            s += f", norm_type={self.norm_type}"
        if self.scale_grad_by_freq:
            s += f", scale_grad_by_freq={self.scale_grad_by_freq}"
        if self.sparse:
            s += f", sparse=True"
        return s

# end of class Embedding

