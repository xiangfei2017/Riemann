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
import zipfile
import io
from typing import Any, Union, Optional
from .tensordef import TN, tensor
from .nn.module import Parameter, Module
from .cuda import Device, cp

def _tensor_constructor(data, shape, dtype_str, device_str, requires_grad):
    """
    从序列化数据重建Riemann张量
    
    参数：
        data: 张量数据（numpy或cupy数组）
        shape: 张量形状
        dtype_str: 数据类型字符串
        device_str: 设备字符串
        requires_grad: 是否需要梯度
        
    返回：
        TN: 重建的Riemann张量
    """
    
    # 创建设备对象
    device = Device(device_str)
    
    # 转换数据类型
    dtype = np.dtype(dtype_str)
    
    # 创建张量
    return tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)


def _parameter_constructor(*args):
    """
    从序列化数据重建Riemann参数
    
    参数：
        *args: 构造参数，可以是以下两种形式：
            1. (None, requires_grad) - 空参数
            2. (data, shape, dtype_str, device_str, requires_grad) - 完整参数
        
    返回：
        Parameter: 重建的Riemann参数
    """
    if len(args) == 2 and args[0] is None:
        # 空参数情况
        return Parameter(None, args[1])
    elif len(args) == 5:
        # 完整参数情况
        data, shape, dtype_str, device_str, requires_grad = args
        # 先创建张量
        device = Device(device_str)
        dtype = np.dtype(dtype_str)
        tensor_obj = tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        # 再创建参数
        return Parameter(tensor_obj, requires_grad)
    else:
        raise ValueError(f"Invalid arguments for Parameter constructor: {args}")


def save(obj: Any, f: Union[str, os.PathLike, Any], 
         pickle_module: Any = None, 
         pickle_protocol: int = 2,
         use_new_zipfile_serialization: bool = True) -> None:
    """
    将对象保存到磁盘文件。
    
    此函数使用pickle序列化将Riemann张量、参数、模块或任何Python对象
    保存到磁盘文件。
    
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
        
    # 准备对象进行序列化
    processed_obj = _prepare_for_serialization(obj)
    
    # 对于所有情况，保持完整的元数据
    # 这样可以确保requires_grad等属性被正确保存
    
    # 处理文件路径与类文件对象
    if isinstance(f, (str, os.PathLike)):
        # 文件路径
        if use_new_zipfile_serialization:
            # 使用基于zip的序列化（PyTorch默认格式）
            # PyTorch期望文件在子目录中
            with zipfile.ZipFile(f, 'w', compression=zipfile.ZIP_STORED) as zf:
                # 创建一个缓冲区来保存pickle数据
                buffer = io.BytesIO()
                pickle_module.dump(processed_obj, buffer, protocol=pickle_protocol)
                buffer.seek(0)
                
                # 将数据写入ZIP文件的子目录
                zf.writestr('archive/data.pkl', buffer.getvalue())
        else:
            # 使用传统的pickle格式
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
        # 文件路径
        try:
            # 尝试作为ZIP文件加载
            with zipfile.ZipFile(f, 'r') as zf:
                # 尝试PyTorch格式（archive/data.pkl）
                try:
                    with zf.open('archive/data.pkl', 'r') as fobj:
                        loaded_obj = pickle_module.load(fobj, **pickle_load_args)
                except KeyError:
                    # 尝试Riemann格式（data.pkl）
                    try:
                        with zf.open('data.pkl', 'r') as fobj:
                            loaded_obj = pickle_module.load(fobj, **pickle_load_args)
                    except KeyError:
                        # 如果都失败，尝试加载第一个找到的.pkl文件
                        for name in zf.namelist():
                            if name.endswith('.pkl'):
                                with zf.open(name, 'r') as fobj:
                                    loaded_obj = pickle_module.load(fobj, **pickle_load_args)
                                break
                        else:
                            # 如果没有找到.pkl文件，抛出异常
                            raise zipfile.BadZipFile("No pickle file found in archive")
        except zipfile.BadZipFile:
            # 如果不是ZIP文件，尝试作为传统pickle文件加载
            mode = 'rb'
            with open(f, mode) as file_handle:
                loaded_obj = pickle_module.load(file_handle, **pickle_load_args)
    else:
        # 类文件对象
        loaded_obj = pickle_module.load(f, **pickle_load_args)
    
    # 恢复对象
    processed_obj = _restore_from_serialization(loaded_obj, map_location)
    
    return processed_obj

def _prepare_for_serialization(obj: Any) -> Any:
    """
    准备Riemann对象进行序列化。
    
    此函数将Riemann张量、参数和模块转换为可以安全pickle的格式，
    
    参数：
        obj: 要准备序列化的对象
        
    返回：
        对象的可序列化版本
    """
    if isinstance(obj, Parameter):
        # 对于参数，转换为字典格式，保存必要的元数据
        return {
            '__type__': 'riemann_parameter',
            'data': obj.data,  # 保存底层数组（numpy或转换后的numpy）
            'shape': obj.shape,
            'dtype': str(obj.dtype),
            'device': str(obj.device),
            'requires_grad': obj.requires_grad
        }
    elif isinstance(obj, TN):
        # 对于张量，转换为字典格式，保存必要的元数据
        return {
            '__type__': 'riemann_tensor',
            'data': obj.data,  # 保存底层数组（numpy或转换后的numpy）
            'shape': obj.shape,
            'dtype': str(obj.dtype),
            'device': str(obj.device),
            'requires_grad': obj.requires_grad,
        }
    elif isinstance(obj, Module):
        # 将模块转换为其状态字典
        return obj.state_dict()
    elif isinstance(obj, dict):
        # 递归处理字典
        return {k: _prepare_for_serialization(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # 递归处理列表和元组
        processed = [_prepare_for_serialization(item) for item in obj]
        return processed if isinstance(obj, list) else tuple(processed)
    elif cp is not None and isinstance(obj, cp.ndarray):
        # 对于cupy数组，保持原样保存，与PyTorch行为一致
        return obj
    else:
        # 其他对象原样返回
        return obj


def _restore_from_serialization(obj: Any, map_location: Optional[Any] = None, is_state_dict: bool = False) -> Any:
    """
    从序列化格式恢复Riemann对象。
    
    此函数将序列化数据转换回Riemann张量、参数和模块。
    
    参数：
        obj: 要恢复的序列化对象
        map_location: 用于重新映射存储位置的函数或字典
        is_state_dict: 是否正在恢复状态字典（默认为False）
        
    返回：
        恢复的Riemann对象
    """
    if isinstance(obj, dict) and '__type__' in obj:
        obj_type = obj['__type__']
        
        if obj_type == 'riemann_tensor':
            # 使用tensor函数恢复张量
            try:
                # 将字符串dtype转换回numpy dtype
                dtype_str = obj.get('dtype')            
                dtype = np.dtype(dtype_str)
                # 获取设备信息
                device_str = obj.get('device', 'cpu')
                # 尝试创建设备，如果失败就回退到CPU
                try:
                    device = Device(device_str)
                except Exception:
                    # 设备创建失败，回退到CPU
                    device = Device('cpu')
                # 检查是否真的支持CUDA
                arrlib = cp if (device.type == 'cuda' and cp is not None) else np

                # 直接使用保存的data（已经是numpy或cupy数组）
                data = obj['data']
                # 如果data是numpy数组且需要CUDA，转换为cupy数组
                if device.type == 'cuda' and cp is not None and isinstance(data, np.ndarray):
                    data = cp.asarray(data)

                tensor_obj = tensor(data, dtype=dtype, device=device, requires_grad=obj.get('requires_grad', False))
            except Exception as e:
                # 恢复失败时，返回None作为占位符
                tensor_obj = None
                print(f"Error restoring tensor: {e}")
            return tensor_obj
            
        elif obj_type == 'riemann_parameter':
            # 恢复参数
            try:
                # 将字符串dtype转换回numpy dtype
                dtype_str = obj.get('dtype')            
                dtype = np.dtype(dtype_str)
                # 获取设备信息
                device_str = obj.get('device', 'cpu')
                # 尝试创建设备，如果失败就回退到CPU
                try:
                    device = Device(device_str)
                except Exception:
                    # 设备创建失败，回退到CPU
                    device = Device('cpu')
                # 检查是否真的支持CUDA
                arrlib = cp if (device.type == 'cuda' and cp is not None) else np

                # 直接使用保存的data（已经是numpy或cupy数组）
                data = obj['data']
                # 如果data是numpy数组且需要CUDA，转换为cupy数组
                if device.type == 'cuda' and cp is not None and isinstance(data, np.ndarray):
                    data = cp.asarray(data)
            
                # 使用tensor函数创建参数以确保正确初始化
                parameter_tensor = tensor(data, dtype=dtype, device=device, requires_grad=obj.get('requires_grad', True))
                parameter = Parameter(parameter_tensor)
            except Exception as e:
                # 恢复失败时，返回None作为占位符
                parameter = None
                print(f"Error restoring parameter: {e}")
            return parameter
 
    elif isinstance(obj, np.ndarray):
        # 如果是状态字典中的numpy数组，保持原样
        if is_state_dict:
            return obj
        # 直接保存的numpy数组也保持原样返回，与PyTorch行为一致
        # 只有当对象是从Riemann张量序列化而来时，才会转换回张量
        # 因此这里直接返回numpy数组
        return obj
    elif hasattr(obj, 'numpy'):
        # 处理PyTorch张量（如果直接传递）
        # 这种情况可能发生在测试环境中，当直接传递PyTorch张量时
        # 将PyTorch张量转换为numpy数组，然后创建Riemann张量
        numpy_data = obj.numpy()
        return tensor(numpy_data)
    elif cp is not None and isinstance(obj, cp.ndarray):
        # 如果是状态字典中的cupy数组，转换为numpy数组
        if is_state_dict:
            return obj.get()  # 转换为numpy数组
        # 直接保存的cupy数组保持原样返回，与PyTorch行为一致
        # 只有当对象是从Riemann张量序列化而来时，才会转换回张量
        # 因此这里直接返回cupy数组
        return obj
    elif isinstance(obj, dict):
        # 检查是否可能是状态字典
        # 状态字典通常包含'weight'和'bias'等键，值为numpy数组
        might_be_state_dict = False
        if not '__type__' in obj:
            # 检查字典内容是否符合状态字典的特征
            for key, value in obj.items():
                if isinstance(value, np.ndarray):
                    might_be_state_dict = True
                    break
        
        # 递归处理字典
        return {k: _restore_from_serialization(v, map_location, is_state_dict=might_be_state_dict) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # 递归处理列表和元组
        restored = [_restore_from_serialization(item, map_location, is_state_dict=is_state_dict) for item in obj]
        return restored if isinstance(obj, list) else tuple(restored)
    else:
        # 其他对象原样返回
        return obj