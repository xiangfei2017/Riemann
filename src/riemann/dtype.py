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
try:
    import cupy as cp
except:
    cp = None
import math

# 预定义常量
inf = float('inf')
ninf = float('-inf')
nan = float('nan')
e = math.e
pi = math.pi

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

def is_float(dtype:np.dtype):
    return np.issubdtype(dtype, np.floating)

def is_complex(dtype:np.dtype):
    return np.issubdtype(dtype, np.complexfloating)

# 预定义类型值映射，避免在每次函数调用时重新创建字典
type_value_dict = { np.dtype('bool'):(1,0),
                    np.dtype('uint8'):(8,0),
                    np.dtype('int8'):(8,1),
                    np.dtype('uint16'):(16,0),
                    np.dtype('int16'):(16,1),
                    np.dtype('uint32'):(32,0),
                    np.dtype('int32'):(32,1),
                    np.dtype('uint64'):(64,0),  
                    np.dtype('int64'):(64,1),               
                    np.dtype('float16'):(64+16,1),
                    np.dtype('float32'):(64+32,1),
                    np.dtype('float64'):(64+64,0),
                    np.dtype('complex64'):(64+64,1),
                    np.dtype('complex128'):(64+128,1),
} 

def infer_data_type(v):
    # 优先检查numpy标量类型、python bool，然后是python整形、浮点、复数
    if isinstance(v,bool):        
        dt = np.dtype('bool')
    elif isinstance(v,int):
        dt = np.dtype('int64')
    elif isinstance(v,(np.bool_,np.integer,np.floating,np.complexfloating)):
        dt = v.dtype
    elif isinstance(v,float):
        dt = get_default_dtype()    
    elif isinstance(v,complex):
        dt = get_default_complex()
    elif isinstance(v,np.ndarray) or (cp and isinstance(v,cp.ndarray)):
        dt = v.dtype
    elif isinstance(v,(list,tuple)):
        if len(v) == 0:
            return default_float
        
        # 初始化max_type以处理空列表/元组情况
        max_type, max_value, max_sign =None, 0, 0
        for e in v:
            data_type = infer_data_type(e)
            type_value,sign = type_value_dict[data_type]
            if type_value > max_value:
                max_type, max_value, max_sign = data_type,type_value,sign
            elif type_value == max_value and sign != max_sign:
                if type_value == 128: # float64和complex64比较
                    max_type = np.dtype('complex128')
                    max_sign = 1
                else:
                    new_type_value = 2*type_value
                    if new_type_value > 64:
                        max_type = np.dtype('float64')
                        max_sign = 0
                    else:
                        max_type = np.dtype(f'int{new_type_value}')
                        max_sign = 1
                
        dt = max_type
    else:
        raise TypeError('elements of v need to be number')
    return dt

def infer_dtype_in_binoper(non_tensor_value,tensor_dtype):
    '''
    推断二元操作符运算中非TN量的数据类型
    '''

    if is_float(tensor_dtype) and \
       isinstance(non_tensor_value,(int,float,bool)):
        return tensor_dtype
    
    if is_complex(tensor_dtype):
        return tensor_dtype
    
    dt = infer_data_type(non_tensor_value)
    return dt

def is_scalar(value):
    '''检查value是否为标量：int、float、complex、numpy标量、TN标量'''
    from .tensordef import TN
    scalar_types = (int, float, complex)
    is_numpy_scalar = hasattr(value, 'dtype') and np.isscalar(value)
    is_TN_scalar = isinstance(value, TN) and value.ndim == 0
    return isinstance(value, scalar_types) or is_numpy_scalar or is_TN_scalar

