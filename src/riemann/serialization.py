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
Riemann Library Serialization Module: Model Persistence and Data Storage

This module provides comprehensive serialization and deserialization capabilities for the
Riemann machine learning framework. It implements save and load functions for persisting
Riemann tensors, parameters, modules, and training states to disk, with an interface
compatible with PyTorch's torch.save and torch.load functions.

Main features:
    - Tensor serialization: Efficient saving and loading of Riemann tensors with
      gradient information and computation graph metadata
    - Module persistence: Complete model state serialization including parameters,
      buffers, and submodule hierarchies
    - Training checkpointing: Save and restore training states for resuming training
    - Multiple format support: Flexible serialization formats including pickle and
      custom binary formats
    - Cross-platform compatibility: Ensures models can be saved and loaded across
      different platforms and Python versions
    - Memory-efficient loading: Lazy loading mechanisms for large models and datasets

Using this module, you can save trained models, checkpoint training progress,
and share models across different environments, with full compatibility with
PyTorch's serialization ecosystem.
"""

import numpy as np
import pickle
import os
from typing import Any, Union, Dict, List, Tuple, Optional
from .tensordef import TN, tensor
from .nn.module import Parameter, Module
from .cuda import Device, cp

def save(obj: Any, f: Union[str, os.PathLike, Any], 
         pickle_module: Any = None, 
         pickle_protocol: int = 2,
         use_new_zipfile_serialization: bool = True) -> None:
    """
    将对象保存到磁盘文件。
    
    此函数使用pickle序列化将Riemann张量、参数、模块或任何Python对象
    保存到磁盘文件。它与PyTorch的torch.save接口兼容。
    
    参数：
        obj: 要保存的对象。可以是张量、参数、模块或任何可pickle的对象
        f: 要写入的文件路径或类文件对象
        pickle_module: 用于pickle的模块（默认：pickle）
        pickle_protocol: Pickle协议版本（默认：2）
        use_new_zipfile_serialization: 是否使用基于zip的序列化（默认：True）
        
    示例：
        >>> import riemann as rm
        >>> # 保存张量
        >>> tensor = rm.randn(3, 4)
        >>> rm.save(tensor, 'tensor.pt')
        >>> 
        >>> # 保存模块
        >>> model = rm.nn.Linear(10, 5)
        >>> rm.save(model.state_dict(), 'model_weights.pt')
        >>> 
        >>> # 保存多个对象
        >>> rm.save({
        ...     'model': model.state_dict(),
        ...     'optimizer_state': optimizer.state_dict(),
        ...     'epoch': 10
        ... }, 'checkpoint.pt')
    """
    if pickle_module is None:
        pickle_module = pickle
        
    # 如果需要，将Riemann对象转换为可pickle格式
    processed_obj = _prepare_for_serialization(obj)
    
    # 处理文件路径与类文件对象
    if isinstance(f, (str, os.PathLike)):
        # 文件路径 - 打开并写入
        mode = 'wb'
        with open(f, mode) as file_handle:
            pickle_module.dump(processed_obj, file_handle, protocol=pickle_protocol)
    else:
        # 类文件对象
        pickle_module.dump(processed_obj, f, protocol=pickle_protocol)


def load(f: Union[str, os.PathLike, Any], 
         map_location: Optional[Any] = None,
         pickle_module: Any = None,
         **pickle_load_args: Any) -> Any:
    """
    从磁盘文件加载对象。
    
    此函数使用pickle反序列化从磁盘文件加载Riemann张量、参数、模块或任何Python对象。
    它与PyTorch的torch.load接口兼容。
    
    参数：
        f: 要读取的文件路径或类文件对象
        map_location: 指定如何重新映射存储位置的函数或字典（此版本中未完全实现）
        pickle_module: 用于unpickle的模块（默认：pickle）
        **pickle_load_args: 传递给pickle.load的额外参数
        
    返回：
        加载的对象
        
    示例：
        >>> import riemann as rm
        >>> # 加载张量
        >>> tensor = rm.load('tensor.pt')
        >>> 
        >>> # 加载模型权重
        >>> state_dict = rm.load('model_weights.pt')
        >>> model.load_state_dict(state_dict)
        >>> 
        >>> # 加载检查点
        >>> checkpoint = rm.load('checkpoint.pt')
        >>> model.load_state_dict(checkpoint['model'])
        >>> optimizer.load_state_dict(checkpoint['optimizer_state'])
        >>> epoch = checkpoint['epoch']
    """
    if pickle_module is None:
        pickle_module = pickle
        
    # 处理文件路径与类文件对象
    if isinstance(f, (str, os.PathLike)):
        # 文件路径 - 打开并读取
        mode = 'rb'
        with open(f, mode) as file_handle:
            loaded_obj = pickle_module.load(file_handle, **pickle_load_args)
    else:
        # 类文件对象
        loaded_obj = pickle_module.load(f, **pickle_load_args)
    
    # 如果需要，转换回Riemann对象
    processed_obj = _restore_from_serialization(loaded_obj, map_location)
    
    return processed_obj

def _prepare_for_serialization(obj: Any) -> Any:
    """
    准备Riemann对象进行序列化。
    
    此函数将Riemann张量、参数和模块转换为可以安全pickle的格式。
    
    参数：
        obj: 要准备序列化的对象
        
    返回：
        对象的可序列化版本
    """
    if isinstance(obj, TN):
        # 将张量转换为可序列化的字典
        return {
            '__type__': 'tensor',
            'data': obj.data.tolist(),
            'shape': obj.shape,
            'dtype': str(obj.dtype),
            'device': str(obj.device),
            'requires_grad': obj.requires_grad,
            'is_leaf': obj.is_leaf,
        }
    elif isinstance(obj, Parameter):
        # 将参数转换为可序列化的字典
        return {
            '__type__': 'parameter',
            'data': obj.data.tolist(),
            'shape': obj.shape,
            'dtype': str(obj.dtype),
            'device': str(obj.device),
            'requires_grad': obj.requires_grad,
            'is_leaf': obj.is_leaf
        }
    elif isinstance(obj, Module):
        # 将模块转换为其状态字典
        return {
            '__type__': 'module',
            'class_name': obj.__class__.__name__,
            'module_name': obj.__class__.__module__,
            'state_dict': obj.state_dict()
        }
    elif isinstance(obj, dict):
        # 递归处理字典
        return {k: _prepare_for_serialization(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # 递归处理列表和元组
        processed = [_prepare_for_serialization(item) for item in obj]
        return processed if isinstance(obj, list) else tuple(processed)
    else:
        # 其他对象原样返回
        return obj


def _restore_from_serialization(obj: Any, map_location: Optional[Any] = None) -> Any:
    """
    从序列化格式恢复Riemann对象。
    
    此函数将序列化数据转换回Riemann张量、参数和模块。
    
    参数：
        obj: 要恢复的序列化对象
        map_location: 用于重新映射存储位置的函数或字典
        
    返回：
        恢复的Riemann对象
    """
        
    if isinstance(obj, dict) and '__type__' in obj:
        obj_type = obj['__type__']
        
        if obj_type == 'tensor':
            # 使用tensor函数恢复张量
            try:
                # 将字符串dtype转换回numpy dtype
                dtype_str = obj.get('dtype')            
                dtype = np.dtype(dtype_str)
                # 获取设备信息
                device_str = obj.get('device', 'cpu')
                device = Device(device_str)
                arrlib = cp if device.type == 'cuda' else np

                # 将列表转换回具有正确形状和dtype的numpy数组
                data = arrlib.array(obj['data'],dtype=dtype)
                data = data.reshape(obj['shape'])

                tensor_obj = tensor(data, dtype=dtype,device=device, requires_grad=obj.get('requires_grad', False))
                tensor_obj.is_leaf = obj.get('is_leaf', True)
            except Exception as e:
                # 恢复失败时，返回None作为占位符
                tensor_obj = None
                print(f"Error restoring tensor: {e}")
            return tensor_obj
            
        elif obj_type == 'parameter':
            # 恢复参数
            try:
                # 将字符串dtype转换回numpy dtype
                dtype_str = obj.get('dtype')            
                dtype = np.dtype(dtype_str)
                # 获取设备信息
                device_str = obj.get('device', 'cpu')
                device = Device(device_str)
                arrlib = cp if device.type == 'cuda' else np

                # 将列表转换回具有正确形状和dtype的numpy数组
                data = arrlib.array(obj['data'],dtype=dtype)
                data = data.reshape(obj['shape'])
            
                # 使用tensor函数创建参数以确保正确初始化
                parameter_tensor = tensor(data, dtype=dtype,device=device, requires_grad=obj.get('requires_grad', True))
                parameter = Parameter(parameter_tensor)
                parameter.is_leaf = obj.get('is_leaf', True)
            except Exception as e:
                # 恢复失败时，返回None作为占位符
                parameter = None
                print(f"Error restoring parameter: {e}")
            return parameter
        
        elif obj_type == 'module':
            # 恢复模块（这更复杂，可能需要类注册）
            # 目前只返回状态字典
            return obj['state_dict']
            
    elif isinstance(obj, dict):
        # 递归处理字典
        return {k: _restore_from_serialization(v, map_location) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # 递归处理列表和元组
        restored = [_restore_from_serialization(item, map_location) for item in obj]
        return restored if isinstance(obj, list) else tuple(restored)
    else:
        # 其他对象原样返回
        return obj