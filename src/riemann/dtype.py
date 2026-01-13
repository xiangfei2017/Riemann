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
Data Type Utilities Module for the Riemann Library

This module implements data type related functions for the Riemann library,
providing essential utilities for tensordef.py including:

- Predefined data types and aliases (float16, float32, float64, complex64, etc.)
- Default data type management functions:
  - set_default_dtype: Sets the default floating-point data type
  - get_default_dtype: Retrieves the current default floating-point type
  - get_default_complex: Derives the default complex type based on default float
- Data type checking functions:
  - is_floating: Checks if a data type is floating-point or complex
  - is_number: Checks if a value is a numeric type
  - is_numeric_array: Checks if a NumPy array has numeric data type
  - is_scalar: Checks if a value is a scalar (including Riemann Tensor scalar)
- Data type inference:
  - infer_data_type: Infers appropriate data type from Python values, NumPy arrays,
    or collections of values, with type promotion logic

All functions are designed to provide consistent type handling across the Riemann library.
"""

import numpy as np


# 预定义dtype对象
float16 = np.dtype('float16')
float32 = np.dtype('float32')
float64 = np.dtype('float64')
complex64 = np.dtype('complex64')
complex128 = np.dtype('complex128')

half = float16
float_ = float32
double = float64
complex_ = complex64

int8 = np.dtype('int8')
int16 = np.dtype('int16')
int32 = np.dtype('int32')
int64 = np.dtype('int64')

uint8 = np.dtype('uint8')
uint16 = np.dtype('uint16')
uint32 = np.dtype('uint32')
uint64 = np.dtype('uint64')

short = int16
int_ = int32
long = int64

bool_ = np.dtype('bool')

default_float = float32

def set_default_dtype(dtype:np.dtype):
    global default_float
    if dtype.kind == 'f':
        default_float = dtype
    else:
        raise TypeError('only floating-point types are supported as the default type')
    return

def get_default_dtype():
    return default_float

def get_default_complex():
    if default_float.type == np.float32:
        default_complex = complex64
    elif default_float.type == np.float64:
        default_complex = complex128
    else:
        raise TypeError(f'Invalid default type of float:{default_float}')
    return default_complex

def is_numeric_array(numpy_arr):
    return numpy_arr.dtype.kind in {'i', 'u', 'f', 'c', 'b'}

def is_number(v):
    return isinstance(v,(int,float,complex,bool,np.bool_,
                         np.integer,np.floating,np.complexfloating))

def is_float_or_complex(dtype:np.dtype):
    if np.issubdtype(dtype, np.floating) or \
       np.issubdtype(dtype, np.complexfloating):
        return True
    return False

# 预定义类型值映射，避免在每次函数调用时重新创建字典
# 使用更高效的元组结构存储类型信息
type_info = {
    # (size_value, is_signed, numpy_dtype)
    bool: (1, 0, np.dtype('bool')),
    np.bool_: (1, 0, np.dtype('bool')),
    int: (64, 1, np.dtype('int64')),
    np.int8: (8, 1, np.dtype('int8')),
    np.uint8: (8, 0, np.dtype('uint8')),
    np.int16: (16, 1, np.dtype('int16')),
    np.uint16: (16, 0, np.dtype('uint16')),
    np.int32: (32, 1, np.dtype('int32')),
    np.uint32: (32, 0, np.dtype('uint32')),
    np.int64: (64, 1, np.dtype('int64')),
    np.uint64: (64, 0, np.dtype('uint64')),
    float: (64+64, 0, get_default_dtype()),
    np.float16: (64+16, 1, np.dtype('float16')),
    np.float32: (64+32, 1, np.dtype('float32')),
    np.float64: (64+64, 0, np.dtype('float64')),
    complex: (64+128, 1, get_default_complex()),
    np.complex64: (64+64, 1, np.dtype('complex64')),
    np.complex128: (64+128, 1, np.dtype('complex128')),
}

# 预定义常见类型的直接映射
_dtype_cache = {
    np.dtype('bool'): (1, 0),
    np.dtype('int8'): (8, 1),
    np.dtype('uint8'): (8, 0),
    np.dtype('int16'): (16, 1),
    np.dtype('uint16'): (16, 0),
    np.dtype('int32'): (32, 1),
    np.dtype('uint32'): (32, 0),
    np.dtype('int64'): (64, 1),
    np.dtype('uint64'): (64, 0),
    np.dtype('float16'): (64+16, 1),
    np.dtype('float32'): (64+32, 1),
    np.dtype('float64'): (64+64, 0),
    np.dtype('complex64'): (64+64, 1),
    np.dtype('complex128'): (64+128, 1),
}

def infer_data_type(v):
    """
    推断输入数据的最佳numpy数据类型。
    
    优化点：
    1. 预计算类型映射，避免每次调用重新创建字典
    2. 为常见类型提供直接快速路径
    3. 使用迭代而非递归处理嵌套结构
    4. 优化类型比较逻辑，减少分支
    5. 缓存类型检查结果
    6. 使用更高效的类型创建方式
    """
    # 快速路径：标量类型
    v_type = type(v)
    if v_type in type_info:
        return type_info[v_type][2]
    
    # 快速路径：numpy数组
    if isinstance(v, np.ndarray):
        return v.dtype
    
    # 快速路径：numpy标量（处理不在type_info中的特殊情况）
    if hasattr(v, 'dtype'):
        return v.dtype
    
    # 处理列表和元组
    if isinstance(v, (list, tuple)):
        if not v:
            return default_float
        
        # 初始化最大值跟踪
        max_type, max_value, max_sign = None, 0, 0
        
        # 使用迭代处理，避免递归
        stack = list(v)
        while stack:
            item = stack.pop()
            item_type = type(item)
            
            if item_type in type_info:
                # 快速处理已知类型
                size_val, is_signed, dtype = type_info[item_type]
                type_val = size_val
            elif isinstance(item, np.ndarray):
                # 处理numpy数组
                dtype = item.dtype
                type_val, is_signed = _dtype_cache.get(dtype, (64+64, 0))  # 默认float64
            elif hasattr(item, 'dtype'):
                # 处理numpy标量
                dtype = item.dtype
                type_val, is_signed = _dtype_cache.get(dtype, (64+64, 0))  # 默认float64
            elif isinstance(item, (list, tuple)):
                # 处理嵌套结构
                stack.extend(item)
                continue
            else:
                # 未知类型
                raise TypeError('elements of v need to be number')
            
            # 更新最大值
            if type_val > max_value:
                max_type, max_value, max_sign = dtype, type_val, is_signed
            elif type_val == max_value and is_signed != max_sign:
                # 处理同大小但符号不同的情况
                if type_val == 128:  # float64和complex64的情况
                    max_type = np.dtype('complex128')
                    max_sign = 1
                else:
                    # 升级到更大的类型
                    new_type_val = type_val * 2
                    if new_type_val > 64:
                        max_type = np.dtype('float64')
                        max_sign = 0
                    else:
                        # 使用预定义的类型映射，避免字符串格式化
                        if new_type_val == 16:
                            max_type = np.dtype('int16')
                        elif new_type_val == 32:
                            max_type = np.dtype('int32')
                        elif new_type_val == 64:
                            max_type = np.dtype('int64')
                        else:
                            max_type = np.dtype('float64')
                        max_sign = 1
        
        return max_type
    
    # 所有其他情况
    raise TypeError('elements of v need to be number')

def is_scalar(value):
    '''检查value是否为标量：int、float、complex、numpy标量、TN标量'''
    from .tensordef import TN
    scalar_types = (int, float, complex)
    is_numpy_scalar = hasattr(value, 'dtype') and np.isscalar(value)
    is_TN_scalar = isinstance(value, TN) and value.ndim == 0
    return isinstance(value, scalar_types) or is_numpy_scalar or is_TN_scalar

