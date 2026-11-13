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
Riemann Library Core Module: Tensor Definition and Automatic Differentiation

This module is the core of the Riemann automatic differentiation framework, implementing
the Tensor class (TN) and a complete automatic differentiation mechanism. Tensors are the
basic data structure of the framework, encapsulating NumPy multidimensional arrays and
providing automatic gradient calculation capabilities, supporting complex neural network
construction and training.

Main features:
    - Tensor class (TN): Encapsulates multidimensional arrays with support for mathematical
      operations and automatic differentiation
    - Dynamic computation graph: Automatically builds and tracks computation history
    - Backpropagation algorithm: Efficiently implements gradient calculation
    - Rich tensor operations: Including arithmetic operations, shape transformations,
      indexing operations, etc.
    - Random number generation and tensor creation functions
    - Linear algebra operation support

Using this module, you can build and train various machine learning models, implement
custom gradient descent algorithms, and perform scientific computing and numerical analysis.

Example usage:
    >>> import riemann as rm
    >>> x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
    >>> y = x * 2 + 3
    >>> z = y.mean()
    >>> z.backward()
    >>> print(x.grad)  # Gradient calculation result
"""

from __future__ import annotations
from collections import namedtuple
import builtins
import inspect
import warnings
from typing import Callable, Any, Optional, overload
from contextlib import contextmanager
import numpy as np
from ..cuda import Device, cp, get_default_device
from .dtype import *
from .gradmode import *

class APIRegistry:
    def __init__(self):
        pass

    def register(self, name: str):
        def decorator(fn):
            # Prevent duplicate registration
            if hasattr(self, name):
                raise RuntimeError(f"Name '{name}' is already registered; overwriting is not allowed")
            setattr(self, name, fn)
            return fn
        return decorator

    def get(self, name: str):
        if not hasattr(self, name):
            raise KeyError(f"Registered function not found: {name}")
        return getattr(self, name)


# Global unique registry instance
api_registry = APIRegistry()

class TN:
    """
    张量(Tensor)类是Riemann框架的核心数据结构。
    
    TN(Tensor Numerics)类将NumPy（CPU）或CuPy（GPU）多维数组封装为张量，
    并在数学运算过程中自动构建计算图，从而支持自动微分(自动求导)功能，
    是构建神经网络和实现深度学习算法的基础。
    
    主要特性：
    - 基于NumPy/CuPy多维数组存储数据，支持CPU与GPU设备及彼此间的迁移
    - 自动构建动态计算图，追踪操作历史
    - 支持反向传播算法，自动计算梯度
    - 提供丰富的数学运算和函数接口
    - 支持链式法则进行复杂导数计算
    - 支持数据类型(dtype)与设备(device)管理
    
    核心设计思想：
    1. 计算图构建：每个张量操作都会记录其输入张量和计算函数
    2. 梯度追踪：通过requires_grad标志控制是否追踪梯度
    3. 反向传播：通过backward()方法从输出张量开始反向计算梯度
    4. 参数优化：计算得到的梯度可用于优化模型参数
    
    示例用法：
    ```python
    # 创建张量
    x = tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    # 执行操作，自动构建计算图
    y = x * 2
    z = y.mean()
    
    # 反向传播计算梯度
    z.backward()
    
    # 访问梯度
    print(x.grad)  # 输出: [0.66666667 0.66666667 0.66666667]
    ```
    
    常用属性与方法：
    - 自动微分与计算图: backward()（反向传播计算梯度）、detach()（脱离计算图共享数据）、
        requires_grad（是否追踪梯度）、grad（计算得到的梯度）、is_leaf（是否叶子张量）
    - 张量信息: shape（形状）、ndim（维度数）、dtype（数据类型）、device（设备）、T（转置）
    - 数据访问: numpy()（转为NumPy数组）、array()（返回底层数组）、item()（转为Python标量）
    - 数学运算: abs()（绝对值）、log()/exp()（对数/指数）、sqrt()（平方根）、
        pow()（幂运算）、sin()/cos()/tan()（三角函数）
    - 规约与统计: sum()（求和）、mean()（均值）、prod()（乘积）、max()/min()（最大/最小值）、
        std()/var()（标准差/方差）
    - 设备迁移: cpu()、cuda()
    
    注：该类提供与PyTorch兼容的接口。
    """
    def __init__(
        self,
        data: (np.ndarray | cp.ndarray | np.generic | cp.generic |
               list | tuple | range | int | float | complex | bool | None) = None,
        requires_grad: bool = False
    ):
        """
        初始化张量对象。
        
        创建一个新的张量实例，初始化所有必要的属性。张量是Riemann框架的核心数据结构，
        封装了NumPy数组并支持自动微分功能。
        
        参数:
            data: 张量数据，支持以下形式：
                - None: 创建空张量，dtype 与设备分别跟随 get_default_dtype() 和 get_default_device()
                - numpy/cupy 数组（np.ndarray/cp.ndarray）: 直接使用，不复制数据
                - numpy/cupy 标量（np.generic/cp.generic）: 转换为对应数组库的0维数组
                - Python 数组like（list、tuple、range）: 按当前默认设备转换为 numpy 或 cupy 数组
                - Python 数值标量（int、float、complex、bool）: 按当前默认设备转换为 numpy 或 cupy 的0维数组
            requires_grad: 是否需要计算梯度，默认为False
        
        属性：
            _array (np.ndarray | cp.ndarray): 存储张量数据的NumPy或Cupy数组
            fromvars (tuple): 记录本张量是通过哪些张量计算得来的
            parms (tuple): 对张量进行函数运算时用到的参数，如sum函数的dim和keepdim参数
            gradfuncs (tuple): 张量运算时登记的梯度函数对象，在backward中将调用这些钩子函数
            grad (TN): 与data的shape一致的张量，用于存放梯度，初始化为None以节省空间
            requires_grad (bool): 是否需要计算梯度，默认为False
            retains_grad (bool): 是否保留梯度，默认为False
            rcv_grad_count (int): 在计算图中可接收梯度的计数，计算梯度时临时使用
            grad_value (TN): 用于计算反向传播的梯度，最终计算结果保存在grad里，该变量用于临时计算
            _output_of (Module|None): 记录产生该输出张量的模块，用于反向传播钩子机制
            _module_input_of (Module|None): 记录该张量作为哪个模块的输入，用于反向传播钩子机制
        """
        if data is None:
            # 无参构造：创建空张量，dtype 跟随 get_default_dtype()，
            # 设备跟随 get_default_device()（与 torch.empty(0) 的全局默认设备语义一致）
            default_dtype = get_default_dtype()
            default_device = get_default_device()
            if default_device.type == 'cuda':
                # 显式进入目标设备：cp.empty 落在 CuPy 当前设备上，而
                # set_default_device('cuda:1') 不会切换 CuPy 当前设备
                with cp.cuda.Device(default_device.index):
                    arr = cp.empty(0, dtype=default_dtype)
            else:
                arr = np.empty(0, dtype=default_dtype)
        elif isinstance(data, np.ndarray) :
            arr = data
        elif isinstance(data,np.generic):
            # 标量转换为数组
            arr = np.array(data)
        elif cp and isinstance(data,cp.ndarray):
            arr = data    
        elif cp and isinstance(data,cp.generic):
            # 标量转换为数组
            arr = cp.array(data)
        else:
            try:
                dtype = infer_data_type(data)
                # 按当前默认设备选择数组库：CUDA 设备生成 cupy 数组，否则生成 numpy 数组
                default_device = get_default_device()
                if cp and default_device.type == 'cuda':
                    # 显式进入目标设备，确保数据落在 default_device 对应的 GPU 上
                    with cp.cuda.Device(default_device.index):
                        arr = cp.array(data, dtype=dtype)
                else:
                    arr = np.array(data, dtype=dtype)
            except:
                raise TypeError("data need to be numpy or cupy array or scalar")
        
        self._array:np.ndarray|cp.ndarray = arr
        self.requires_grad = requires_grad
        
        self.fromvars = ()              #张量运算时，运算符函数中记录本张量是通过哪些张量计算得来的
        self.parms = ()                 #对张量进行函数运算时用到的参数，比如sum函数的dim和keepdim参数
        self.gradfuncs = ()             #张量运算时，运算符函数中登记梯度函数对象，在backward中将调用这些钩子函数
        self.grad:TN = None             #与data的shape一致的张量对象，用于存放梯度，
                                        #初始化为None是为了节省空间，计算梯度时才分配空间
        self.retains_grad = False
        
        self.rcv_grad_count = 0         #在具体计算图中，self可接收梯度的计数，计算梯度时临时使用
        self.grad_value:TN = None       #用于计算反向传播的梯度，最终计算结果保存在grad里，该变量用于临时计算
        
        self._output_of = None          #记录该张量是哪个模块的输出（单引用）
        self._module_input_of = None    #记录该张量是哪个模块的输入（单引用）
        
        return
    
    # 用于交互式环境显示对象的数值
    def __disp__(self, format_spec: str):
        """
        带格式字符串参数的显示函数
        
        参数:
            format_spec: 格式规范字符串，例如'8.4f'表示：宽度为8，精度为4的浮点数格式
                        浮点数和复数使用此格式显示，其它数据类型使用numpy默认格式化
                        支持指定部分格式参数：
                        - 未指定width但指定precision时，width使用numpy默认值
                        - 指定width但未指定precision时，precision使用numpy默认值
                        - 都未指定时，使用numpy默认格式
        
        返回:
            str: 格式化后的张量字符串表示
        """
        # 获取默认浮点类型用于比较
        default_float = get_default_dtype()
        default_complex = get_default_complex()

        # 获取数组库
        arrlib = self._get_array_lib()

        # 构建属性字符串        
        attrs = []        
        if self.device != Device('cpu'):
            attrs.append(f"device='{self.device}'")
        if is_float_or_complex(self.dtype) and self.dtype != default_float and self.dtype != default_complex:
            attrs.append(f"dtype={self.dtype.name}")
        if self.requires_grad:
            attrs.append("requires_grad=True")        
        attr_str = ", ".join(attrs)

        # 格式化工具函数
        def format_text(text, prefixStr):
            lines = text.split('\n')
            if not lines:
                return ""
            
            # 处理第一行
            if len(lines) == 1:
                lines[0] = f"{prefixStr}({lines[0]})"
            else:
                lines[0] = f"{prefixStr}({lines[0]}"
            
            # 计算缩进量
            indent = ' ' * (len(prefixStr) + 1)
            
            # 处理后续行
            for i in range(1, len(lines)):
                if i == len(lines) - 1:
                    lines[i] = f"{indent}{lines[i]})"
                else:
                    lines[i] = f"{indent}{lines[i]}"
            
            return '\n'.join(lines)
        
        # 提取宽度、精度和格式类型信息
        format_type = 'f'  # 默认格式类型
        width = None  # None表示使用numpy默认值
        precision = None  # None表示使用numpy默认值
        
        if '.' in format_spec:
            # 提取小数点前的宽度部分
            width_part = format_spec.split('.')[0]
            width_digits = ''.join(filter(str.isdigit, width_part))
            if width_digits:
                width = int(width_digits)
            
            # 提取小数点后的数字和格式类型部分
            precision_part = format_spec.split('.')[1]
            # 提取数字部分（精度）
            precision_digits = ''.join(filter(str.isdigit, precision_part))
            if precision_digits:
                precision = int(precision_digits)
            # 提取格式类型（如果有）
            type_part = ''.join(filter(str.isalpha, precision_part))
            if type_part:
                format_type = type_part[0]  # 使用第一个字母作为格式类型
        else:
            # 如果没有小数点，尝试提取宽度和格式类型
            width_digits = ''.join(filter(str.isdigit, format_spec))
            if width_digits:
                width = int(width_digits)
            type_part = ''.join(filter(str.isalpha, format_spec))
            if type_part:
                format_type = type_part[0]  # 使用第一个字母作为格式类型
        
        # 判断是否需要使用numpy默认格式化
        # 只要指定了precision，就使用自定义格式化
        # 否则使用numpy默认格式
        use_numpy_default = precision is None
        
        # 简化实现：根据数据类型使用不同的格式化策略
        if arrlib.issubdtype(self._array.dtype, arrlib.complexfloating):
            if use_numpy_default:
                # 使用numpy默认格式化
                data_str = arrlib.array2string(
                    self._array,
                    separator=', ',
                    threshold=self._array.size  # 确保显示所有元素
                )
            else:
                # 复数类型的简化格式化 - 右对齐
                def complex_formatter(z):
                    # 格式化实部和虚部，使用右对齐格式
                    real = z.real
                    imag = z.imag
                    
                    # 使用从format_spec提取的宽度、精度和格式类型
                    # 使用右对齐格式化，支持不同格式类型（如'f'、'e'等）
                    if width is not None:
                        real_str = f'{real:>{width}.{precision}{format_type}}'
                        # 处理虚部符号 - 从格式化字符串中移除多余的符号
                        imag_abs = builtins.abs(imag)
                        imag_sign = '+' if imag >= 0 else '-'
                        imag_str_clean = f'{imag_abs:>{width-1}.{precision}{format_type}}'
                    else:
                        real_str = f'{real:.{precision}{format_type}}'
                        # 处理虚部符号 - 从格式化字符串中移除多余的符号
                        imag_abs = builtins.abs(imag)
                        imag_sign = '+' if imag >= 0 else '-'
                        imag_str_clean = f'{imag_abs:.{precision}{format_type}}'
                    
                    return f'{real_str}{imag_sign}{imag_str_clean}j'
                
                formatter_dict = {'complex_kind': complex_formatter}
                data_str = arrlib.array2string(
                    self._array,
                    separator=', ',
                    formatter=formatter_dict,  # type: ignore
                    threshold=self._array.size  # 确保显示所有元素
                )
        elif arrlib.issubdtype(self._array.dtype, arrlib.floating):
            if use_numpy_default:
                # 使用numpy默认格式化
                data_str = arrlib.array2string(
                    self._array,
                    separator=', ',
                    threshold=self._array.size  # 确保显示所有元素
                )
            else:
                # 浮点类型的简化格式化 - 右对齐
                def float_formatter(x):
                    # 为了实现小数点对齐，使用右对齐格式化
                    # 使用从format_spec提取的宽度、精度和格式类型
                    if width is not None:
                        return f'{x:>{width}.{precision}{format_type}}'
                    else:
                        return f'{x:.{precision}{format_type}}'
                
                formatter_dict = {'float_kind': float_formatter}
                data_str = arrlib.array2string(
                    self._array,
                    separator=', ',
                    formatter=formatter_dict,  # type: ignore
                    threshold=self._array.size  # 确保显示所有元素
                )
        else:
            # 对于非浮点、非复数类型，使用默认格式化
            if arrlib.issubdtype(self._array.dtype, arrlib.integer):
                data_str = arrlib.array2string(
                    self._array,
                    separator=', '
                )
            else:
                # 其他类型使用默认格式化
                data_str = arrlib.array2string(self._array, separator=', ')

        
        # 第一步：已经生成了多维数据的显示字符串
        # 第二步：使用format_text函数添加"tensor("前缀
        base_str = format_text(data_str, "tensor")
        
        # 第三步：在最后添加属性信息（format_text已经添加了右括号）
        if attr_str:
            base_str = base_str[:-1]
            base_str += f", {attr_str})"
                
        return base_str

    def __repr__(self):
        """调用__disp__()使用默认格式显示对象"""
        return self.__disp__(format_spec='.f')

    # 用于在print(obj)、str(obj)显示对象数值
    def __str__(self):
        return self.__disp__(format_spec='.f')
    
    # 用于在f'{obj}'显示对象数值
    def __format__(self, format_spec):
        """
        自定义格式化输出，支持浮点数格式规范。
        例如："{:.4f}".format(tensor) 会显示4位小数
        """
        if self.ndim == 0 and not self.is_complex():
            return format(self._array, format_spec)
        else:
            return self.__disp__(format_spec)

    def __array__(self, dtype=None, copy=False):
        '''
        numpy/cupy的array asarray函数处理TN张量参数时，自动调用张量的该接口函数将张量转换为numpy/cupy数组。
        该接口仅返回 CPU 端的 numpy ndarray（np.array 原生不支持 GPU 数组自动拉回）。
        '''
        # cupy 数据：cupy.ndarray 未实现返回 numpy 数组的 __array__，显式 cp.asnumpy 拉回 CPU
        if cp is not None and isinstance(self._array, cp.ndarray):
            arr = cp.asnumpy(self._array)
        else:
            arr = self._array
        
        # astype(None) 会被 numpy 解析为默认类型 float64（np.dtype(None) == float64），
        # dtype 为 None 时必须显式归位为原 dtype，使零拷贝/复制语义与显式传入 dtype 时一致
        if dtype is None:
            dtype = arr.dtype
        # numpy 2.x 可能以 copy=None（NEP 56 "copy if needed"）调用 __array__，
        # astype 对 copy=None 的解析不保证零拷贝，归一为 bool 保持语义明确
        return arr.astype(dtype, copy=bool(copy))

    # 显式退出 ufunc 分派（NEP 13）：numpy/cupy 以 TN 为操作数调用 ufunc（如 np.sin(a)、
    # np.multiply(a, b)）时直接报 TypeError，而不是静默经 __array__ 转成裸数组再计算
    # （静默降级会丢失 requires_grad、改变返回类型与 dtype，不符合用户预期）。
    # 运算符（a_n * a_r 等）不受影响：numpy 会让位，回落到 TN 的 __rxxx__ 得到 TN；
    # 转换接口（np.asarray(a)、np.array(a)）也不受影响，仍走 __array__/__buffer__。
    __array_ufunc__ = None

    def __buffer__(self, flags):
        """供 memoryview(obj)/bytes(obj) 及 numpy缓冲区转换调用；GPU数据先经__array__拉回CPU"""
        return memoryview(self.__array__())
    
    def __release_buffer__(self, buffer):
        pass
    
    @property
    def __cuda_array_interface__(self):
        """供 cupy / torch(cuda) / numba 等调用"""
        if hasattr(self._array, '__cuda_array_interface__'):
            return self._array.__cuda_array_interface__
        raise AttributeError("Data is not on GPU")
    
    def __dlpack__(self, stream=None):
        """供 torch.from_dlpack / jax.dlpack 等调用"""
        return self._array.__dlpack__(stream=stream)
    
    def __dlpack_device__(self):
        """返回 DLPack 设备信息"""
        return self._array.__dlpack_device__()
    
    def __hash__(self):
        """
        使TN对象可哈希，允许将其放入set或用作字典键。
        使用对象的内存地址作为哈希值，确保实例唯一性。
        
        Returns:
            int: 基于对象id的哈希值
        """
        return id(self)
    
    def __reduce__(self):
        """
        实现对象的pickle序列化支持，兼容PyTorch格式
        
        返回:
            tuple: 包含构造函数和参数的元组
        """
        from .serialization import _tensor_constructor

        data = self._array
        if cp and isinstance(data, cp.ndarray):
            # 使用 .get() 方法显式转换 cupy 数组为 numpy 数组
            data = data.get()
                    
        return (_tensor_constructor, (
            data,  # 直接序列化底层数据
            self.shape,
            str(self.dtype),
            str(self.device),
            self.requires_grad
        ))
        
    # 返回张量第一维的大小
    def __len__(self):
        """
        返回张量第一维的大小。
        
        返回张量第一个维度的大小，等同于len(tensor)。
        对于标量张量（0维），此方法会引发TypeError。
        
        Returns:
            int: 张量第一维的大小
            
        Raises:
            TypeError: 如果张量是标量（0维）
        """
        return len(self._array)
    
    def __bool__(self):
        """
        将张量转换为布尔值。
        
        根据张量的值返回True或False，遵循Python的布尔转换规则。
        对于非空张量，如果所有元素都不为零，则返回True；否则返回False。
        对于空张量，返回False。
        
        Returns:
            bool: 张量的布尔值表示
        """
        return bool(self._array)
    
    def __float__(self):
        """
        将张量转换为Python浮点数。
        
        仅适用于单元素张量，将张量的值转换为Python的float类型。
        
        Returns:
            float: 张量值的浮点数表示
            
        Raises:
            ValueError: 如果张量包含多个元素
        """
        return float(self._array)

    def __int__(self):
        """
        将张量转换为Python整数。
        
        仅适用于单元素张量，将张量的值转换为Python的int类型。
        
        Returns:
            int: 张量值的整数表示
            
        Raises:
            ValueError: 如果张量包含多个元素
        """
        return int(self._array)
    
    def _get_array_lib(self):
        """
        根据data属性的数组类型，获得numpy或cupy库的模块名
        """
        return np if isinstance(self._array,np.ndarray) else cp

    def item(self):
        """从单元素张量中提取Python标量值
        
        该方法从任何包含单个元素的张量中提取标量值，
        无论张量的维度如何（如0维、[1]一维、[1,1]二维等）。
        
        返回:
            Python标量值（int、float、complex等）
            
        异常:
            RuntimeError: 当张量包含多个元素时抛出
        """
        if self.numel() != 1:
            raise RuntimeError("only one element tensors can convert to Python scalars")
        return self._array.item()
    
    @property
    def dtype(self):
        """
        返回张量的数据类型。
        
        返回张量中元素的数据类型，与NumPy数组的dtype属性一致。
        常见的数据类型包括float32、float64、int32、int64、complex64等。
        
        Returns:
            numpy.dtype: 张量的数据类型对象
        """
        return self._array.dtype
    
    def is_floating_point(self):
        """
        判断张量是否为浮点类型。
        
        检查张量的数据类型是否为浮点类型，包括float16、float32、float64等。
        浮点类型张量支持梯度计算，通常用于神经网络中的权重和激活值。
        
        Returns:
            bool: 如果张量是浮点类型返回True，否则返回False
        """
        return self._array.dtype.kind == 'f'

    def is_complex(self):
        """
        判断张量是否为复数类型。
        
        检查张量的数据类型是否为复数类型，包括complex64、complex128等。
        复数类型张量包含实部和虚部，支持复数运算和梯度计算。
        
        Returns:
            bool: 如果张量是复数类型返回True，否则返回False
        """
        return self._array.dtype.kind == 'c'
    
    def isreal(self:TN)->TN:
        """
        判断张量元素是否为实数。
        
        对于复数张量，返回一个布尔张量，指示每个元素是否为实数（虚部为零）。
        对于实数张量，返回一个与原张量形状相同且所有元素为True的布尔张量。
        
        Returns:
            TN: 布尔张量，True表示对应元素为实数
        """
        return isreal(self)
    
    def isinf(self:TN)->TN:
        """
        判断张量元素是否为无穷大。
        
        逐元素检查张量中的值是否为正无穷大或负无穷大，返回一个布尔张量。
        对于有限数值和NaN（非数值），返回False。
        
        Returns:
            TN: 布尔张量，True表示对应元素为无穷大
        """
        return isinf(self)
    
    def isnan(self:TN)->TN:
        """
        判断张量元素是否为NaN（非数值）。
        
        逐元素检查张量中的值是否为NaN（Not a Number），返回一个布尔张量。
        NaN通常由未定义的数学运算产生，如0除以0。
        
        Returns:
            TN: 布尔张量，True表示对应元素为NaN
        """
        return isnan(self)

    @property
    def real(self):
        """
        返回复数张量的实部。
        
        对于复数张量，返回一个包含实部的新张量。对于实数张量，返回原张量本身。
        支持自动微分，梯度计算正确处理复数运算。
        
        Returns:
            TN: 包含实部的新张量，或原张量（如果为实数张量）
        """
        if not self.is_complex():
            return self
            
        result = TN(self._array.real, requires_grad=(is_grad_enabled() and self.requires_grad))
        
        if result.requires_grad:
            result.fromvars = (self,)
            result.parms = ()
            # 实部的梯度函数：利用广播特性简化计算
            def real_backward(result, i):
                # 因为self是复数，result.grad_value需要转换为复数后返回
                return result.grad_value + 0.0j

            result.gradfuncs = (real_backward,)
        return result
    
    @property
    def imag(self):
        """
        返回复数张量的虚部。
        
        对于复数张量，返回一个包含虚部的新张量。对于实数张量，会引发RuntimeError。
        支持自动微分，梯度计算正确处理复数运算。
        
        Returns:
            TN: 包含虚部的新张量
            
        Raises:
            RuntimeError: 如果张量不是复数类型
        """
        if not self.is_complex():
            raise RuntimeError("imag property is only defined for complex tensors")
        
        result = TN(self._array.imag, requires_grad=(is_grad_enabled() and self.requires_grad))
        
        if result.requires_grad:
            result.fromvars = (self,)
            result.parms = ()
            # 虚部的梯度函数：利用广播特性简化计算
            def imag_backward(result, i):
                # 因为self是复数，result.grad_value须转换与self虚部对应的复数后返回
                return 1.0j * result.grad_value
                
            result.gradfuncs = (imag_backward,)
        return result

    def conj(self):
        """返回张量的复数共轭。
        
        对于复数张量，此方法返回其元素级别的共轭复数；
        对于实数张量，返回张量本身的副本。
        
        返回:
            TN: 包含共轭元素的新张量
        """

        if not self.is_complex():
            return  self
        
        # 计算共轭
        arrlib = self._get_array_lib()
        conj_data = arrlib.conj(self._array)
        
        # 创建结果张量
        ret = TN(conj_data, requires_grad=(is_grad_enabled() and self.requires_grad))
        
        # 设置梯度传播
        if ret.requires_grad:
            def _conj_backward(result_tensor: TN, i: int) -> TN:
                # 共轭操作的梯度是元素级别的共轭
                return result_tensor.grad_value.conj()
            
            ret.fromvars = (self,)
            ret.gradfuncs = (_conj_backward,)
        
        return ret

    @property
    def shape(self):
        """
        返回张量的形状。
        
        返回张量各维度大小的元组，与NumPy数组的shape属性一致。
        例如，一个2x3矩阵的形状为(2, 3)，标量的形状为()。
        
        Returns:
            tuple: 张量各维度大小的元组
        """
        return self._array.shape
    
    @property
    def ndim(self):
        """
        返回张量的维度数。
        
        返回张量的维度数量，也称为秩(rank)。标量的维度为0，向量的维度为1，
        矩阵的维度为2，以此类推。与NumPy数组的ndim属性一致。
        
        Returns:
            int: 张量的维度数
        """
        return self._array.ndim

    def size(self, dim = None):
        """
        返回张量的大小。
        
        当不提供dim参数时，返回张量的形状（与shape属性相同）。
        当提供dim参数时，返回指定维度的大小。
        
        Args:
            dim (int, optional): 要查询的维度索引，从0开始计数。如果为None，返回整个形状。
            
        Returns:
            tuple or int: 如果dim为None，返回形状元组；否则返回指定维度的大小
            
        Raises:
            IndexError: 如果dim超出张量的维度范围
        """
        if dim is None:
            return self._array.shape
        else:
            return self._array.shape[dim]

    def numel(self):
        """返回张量中元素的总数。
        
        返回:
            int: 张量中元素的数量
        """
        return self._array.size

    def type(self, dtype=None, non_blocking=False):
        """返回或转换张量的数据类型
        
        行为：
        - 如果不传入参数，返回张量的数据类型字符串
        - 如果传入数据类型参数，返回一个转换为指定数据类型的新张量
        
        参数:
            dtype: 数据类型，可以是Python类型、NumPy dtype、字符串或Riemann dtype
                如果为None，则返回当前张量的数据类型
            non_blocking (bool): 如果为True且源在固定内存中，复制到GPU可与主机计算异步。
                                默认为False。仅适用于CPU -> GPU的传输。
                
        返回:
            如果dtype为None，返回当前张量的数据类型字符串
            否则返回转换为指定数据类型的新张量
        """
        # 如果不传入参数，返回张量的数据类型字符串
        if dtype is None:
            return self.dtype.name
        
        return self.to(dtype=dtype, device=self.device, non_blocking=non_blocking)

    def bool(self):
        """
        将张量转换为布尔类型(bool_)。
        
        创建并返回一个新的张量，其数据类型为布尔类型(bool_)。
        如果原张量已经是布尔类型，则返回原张量本身。
        
        Returns:
            TN: 布尔类型的新张量，或原张量（如果已经是布尔类型）
        """
        return self.type(bool_)

    def float(self):
        """
        将张量转换为单精度浮点类型(float32)。
        
        创建并返回一个新的张量，其数据类型为float32，与原张量内容相同。
        如果原张量已经是float32类型，则返回原张量本身。
        
        Returns:
            TN: float32类型的新张量，或原张量（如果已经是float32类型）
        """
        return self.type(float32)
    
    def double(self):
        """
        将张量转换为双精度浮点类型(float64)。
        
        创建并返回一个新的张量，其数据类型为float64，与原张量内容相同。
        如果原张量已经是float64类型，则返回原张量本身。
        
        Returns:
            TN: float64类型的新张量，或原张量（如果已经是float64类型）
        """
        return self.type(float64)

    def type_as(self,other):
        """
        将张量转换为与另一个张量相同的数据类型。
        
        创建并返回一个新的张量，其数据类型与参数张量other相同，与原张量内容相同。
        如果原张量的数据类型已经与other相同，则返回原张量本身。
        
        Args:
            other (TN): 目标数据类型的参考张量
            
        Returns:
            TN: 与other相同数据类型的新张量，或原张量（如果数据类型已匹配）
            
        Raises:
            TypeError: 如果other不是张量类型
        """
        if not isinstance(other,TN):
            raise TypeError("'other' need to be a tensor")
        return self.type(other.dtype)
    
    @property
    def device(self):
        """
        动态获取张量所在的设备。
        
        根据data属性的类型动态返回设备对象：
        - 如果data是CuPy数组（cp.ndarray类型），返回CUDA设备
        - 否则返回CPU设备
        
        Returns:
            device: 张量所在的设备对象
        """
        # 直接检查data是否为cp.ndarray类型
        if cp and isinstance(self._array, cp.ndarray):
            # 使用CuPy数组的device属性获取设备信息
            cupy_device = self._array.device
            device_idx = cupy_device.id
            return Device(f'cuda:{device_idx}')
        else:
            # 对于NumPy数组或其他类型，返回CPU设备
            return Device('cpu')
                
    @classmethod
    def _from_array(cls, array, requires_grad: bool = False):
        """
        按 cls 创建张量，绕过子类构造函数。

        供 to() 等内部路径使用：Parameter 的构造函数与 PyTorch 一致只接受
        TN/None，无法直接传入底层数组，这里复用 TN.__init__ 完成统一初始化，
        从而在保留子类类型的同时不改动子类的公开签名。
        """
        ret = object.__new__(cls)
        TN.__init__(ret, array, requires_grad)
        return ret

    def to(self, *args, **kwargs):
        """
        将张量转换为指定的数据类型和/或设备。
        
        创建并返回一个新的张量，其数据类型和/或设备为指定的值，与原张量内容相同。
        如果原张量的数据类型和设备已经与指定的值相同，则返回原张量本身（除非 copy=True）。
        
        Args:
            dtype (optional): 目标数据类型，可以是Python类型、NumPy dtype、字符串或Riemann dtype
            device (optional): 目标设备，可以是字符串（如'cpu'、'cuda'）或Device对象
            non_blocking (bool, optional): 如果为True且数据在固定内存中，则复制到GPU可以与主机计算异步进行。
                                          默认为False。仅适用于CPU -> GPU的传输。
            copy (bool, optional): 如果为True，则总是返回副本，即使设备和数据类型相同。
                                  默认为False。
            
        Returns:
            TN: 指定数据类型和/或设备的新张量，或原张量（如果已匹配且copy=False）
            
        Examples:
            >>> # 转换数据类型
            >>> x = tensor([1.0, 2.0, 3.0], dtype=float32)
            >>> y = x.to(float64)
            >>> # 转换设备
            >>> x = tensor([1.0, 2.0, 3.0], device='cpu')
            >>> y = x.to('cuda')
            >>> # 同时转换数据类型和设备
            >>> x = tensor([1.0, 2.0, 3.0], dtype=float32, device='cpu')
            >>> y = x.to(float64, device='cuda')
            >>> # 使用关键字参数
            >>> x = tensor([1.0, 2.0, 3.0])
            >>> y = x.to(dtype=float64, device='cuda')
            >>> # 从另一个张量复制dtype和device
            >>> x = tensor([1.0, 2.0, 3.0], dtype=float64, device='cuda')
            >>> y = tensor([4.0, 5.0, 6.0])
            >>> z = y.to(x)
            >>> # 强制复制
            >>> y = x.to(copy=True)
            >>> # 异步传输
            >>> y = x.to('cuda', non_blocking=True)
        """
        # 解析参数
        dtype = kwargs.get('dtype', None)
        device = kwargs.get('device', None)
        non_blocking = kwargs.get('non_blocking', False)
        copy = kwargs.get('copy', False)
        
        # 处理位置参数
        if len(args) > 0:
            if isinstance(args[0], TN):
                # 参数是另一个张量，复制其dtype和device
                other = args[0]
                dtype = other.dtype
                device = other.device
            else:
                # 处理第一个位置参数
                first_arg = args[0]
                
                # 检查第一个参数是否是device（字符串形式）
                if isinstance(first_arg, str):
                    # 字符串参数可能是device或dtype
                    # 先尝试识别为device，如果失败再作为dtype
                    try:
                        # 尝试创建device对象并直接赋值
                        device = _get_device(first_arg)
                        # 检查第二个参数是否是dtype
                        if len(args) > 1:
                            dtype = args[1]
                    except Exception:
                        # 不是有效的device字符串，作为dtype处理
                        dtype = first_arg
                        # 检查第二个参数是否是device
                        if len(args) > 1:
                            second_arg = args[1]
                            if isinstance(second_arg, (int, str, Device)):
                                device = second_arg
                elif isinstance(first_arg, (int, Device)):
                    # 整数或Device对象，作为device处理
                    device = first_arg
                    # 检查第二个参数是否是dtype
                    if len(args) > 1:
                        dtype = args[1]
                else:
                    # 其他类型，作为dtype处理
                    dtype = first_arg
                    # 检查第二个参数是否是device
                    if len(args) > 1:
                        second_arg = args[1]
                        if isinstance(second_arg, (int, str, Device)):
                            device = second_arg
                
                # 检查是否有更多参数
                if len(args) > 2:
                    # 不支持超过2个位置参数
                    raise TypeError(f"to() takes at most 2 positional arguments but {len(args)} were given")
        
        # 解析device参数，确定是否使用CUDA
        target_device = self.device if device is None else _get_device(device)
                
        # 处理数据类型转换
        target_dtype = self.dtype if dtype is None else _get_dtype(dtype)
        
        # 检查当前类型和设备是否已经匹配
        dtype_not_change = (self.dtype == target_dtype)
        device_not_change = (self.device == target_device)
        
        # 如果类型和设备都相同，且不需要复制，则返回自身
        if dtype_not_change and device_not_change and not copy:
            return self
        
        # 处理设备迁移、数据转换
        if device_not_change:
            # 设备相同，只需要转换数据类型（或强制复制）
            if is_complex(self.dtype) and not is_complex(target_dtype):
                # 复数向非复数转换时，为避免warning，不用astype直接转，取real后再转换
                ret = type(self)._from_array(self._array.real.astype(target_dtype, copy=False))
            else:
                # 如果copy=True, 数据复制，否则仅数据类型变化时复制
                ret = type(self)._from_array(self._array.astype(target_dtype, copy = copy))            
        else:
            # 设备不同，需要转换设备
            if target_device.type=='cuda':
                # CPU -> CUDA
                # 利用cp.asarray()支持dtype参数的特性，一步完成转换
                if is_complex(self.dtype) and not is_complex(target_dtype):
                    # 复数向非复数转换时，先取real
                    source_data = self._array.real
                else:
                    source_data = self._array
                
                # 拷贝统一走当前流：cupy 的拷贝都排在当前流上，与后续同流操作天然有序。
                # 旧实现为 non_blocking 单独创建了一条流，该流与默认流之间没有同步点，
                # 返回后立即使用数据会产生竞态。non_blocking 只有在源数据位于固定
                # (pinned) 内存时才会真正异步，且异步拷贝同样排在当前流上，因此无需建流。
                ret = type(self)._from_array(_to_device_array(source_data, target_device.index, target_dtype))
            else:
                # CUDA -> CPU
                # 注意：non_blocking 参数主要适用于 CPU -> GPU 传输
                # GPU -> CPU 使用 cp.asnumpy()，这是同步操作
                # 如果需要异步 GPU -> CPU 传输，需要更复杂的实现（使用 pinned memory + 异步拷贝）
                if is_complex(self.dtype) and not is_complex(target_dtype):
                    # 复数向非复数转换时，先取real
                    ret = type(self)._from_array(cp.asnumpy(self._array.real).astype(target_dtype,copy=False))
                else:
                    ret = type(self)._from_array(cp.asnumpy(self._array).astype(target_dtype,copy=False))
        
        # 设置梯度跟踪信息
        if is_float_or_complex(target_dtype):
            ret.requires_grad = (is_grad_enabled() and self.requires_grad)
            
            if ret.requires_grad:
                # 设置计算图相关属性
                ret.fromvars = (self,)                
                # 保存原始数据类型和设备信息用于反向传播
                ret.parms = ((self.dtype,self.device),)                
                
                def to_backward(result, i):
                    original_dtype = result.parms[i][0]
                    original_device = result.parms[i][1]
                    # 直接使用to函数将梯度转换回原始数据类型和设备
                    return result.grad_value.to(dtype=original_dtype, device=original_device)
                
                ret.gradfuncs = (to_backward,)
        else:
            # 非浮点/复数类型不支持梯度
            ret.requires_grad = False
            
        return ret

    def cuda(self, device=None):
        """
        将张量移动到CUDA设备上。
        
        Args:
            device (optional): 目标CUDA设备ID或设备名称，默认为当前设备
        
        Returns:
            TN: 移动到CUDA设备上的新张量
        """
        if device == 'cpu':
            raise ValueError("cuda() method is not supported for 'cpu'")
        
        if device is None:
            device = "cuda"
        elif isinstance(device, int):
            device = Device('cuda', device)
        else:
            device = Device(device)

        return self.to(device=device)
    
    @property
    def is_cuda(self):
        return self.device.type == 'cuda'
    
    def cpu(self):
        """
        将张量移动到CPU设备上。
        
        Returns:
            TN: 移动到CPU设备上的新张量
        """
        return self.to(device='cpu')

    @property
    def is_cpu(self):
        return self.device.type == 'cpu'

    @property
    def is_leaf(self):
        return self.requires_grad == False or not self.fromvars

    def is_contiguous(self):
        """
        检查张量的内存布局是否是连续的。
        
        在内存中，连续张量的元素是按行主序（row-major）存储的，没有间隙。
        对于基于NumPy或CuPy数组的Riemann张量，此方法会检查底层数组的连续性。
        
        Returns:
            bool: 如果张量是连续的返回True，否则返回False
        """
        return self._array.flags.c_contiguous
        
    def contiguous(self):
        """
        如果张量不是连续的，返回一个具有相同数据但连续内存布局的新张量。
        如果张量已经是连续的，返回张量本身。
        
        连续张量的元素在内存中是按行主序（row-major）存储的，没有间隙，
        这可以提高某些操作的性能。
        
        Returns:
            TN: 连续内存布局的张量
        """
        if self._array.flags.c_contiguous:
            return self
        
        # 获取数组库（NumPy或CuPy）
        arrlib = self._get_array_lib()
        
        # 创建新张量
        ret = TN(arrlib.ascontiguousarray(self._array),
                requires_grad = (is_grad_enabled() and self.requires_grad)
        )
        
        if ret.requires_grad:
            ret.fromvars = (self,)
            ret.gradfuncs = (lambda result, i: result.grad_value,)
        
        return ret

    def __getitem__(self, index):
        # 类型转换：TN索引转为NumPy数组
        index_val = _convert_TNindex_to_numpy(index)
        
        # 执行索引操作并捕获错误
        try:
            indexed_data = self._array[index_val]
        except (IndexError, ValueError) as e:
            # 错误信息增强
            err_msg = f"Index operation failed: {str(e)} (Original index type: {type(index)}, shape: {self.shape})"
            raise type(e)(err_msg) from None
        
        # 使用base是否为None来判断是否为视图，这样可以同时支持numpy和cupy
        # 对于numpy和cupy，视图的base属性指向原始数组，而副本的base属性为None
        is_view = indexed_data.base is not None
        
        # 创建结果张量
        ret = TN(indexed_data, requires_grad = (is_grad_enabled() and self.requires_grad))
        
        # 设置视图属性
        ret._is_view = is_view
        ret._base = self
        ret._view_index = index_val
        
        if ret.requires_grad:
            ret.fromvars = (self,)
            ret.parms = (index_val,)
            ret.gradfuncs = (_getitem_backward,)
        
        return ret
    
    def __setitem__(self, index, val):
        """
        原地赋值入口，负责 a[index] = val 以及 Python 展开后的 a[index] op= val。

        `a[index] op= val` 在 Python 中会拆成三步：
        1. temp = a.__getitem__(index)：基本切片得到视图，整数/花式索引得到副本；
        2. temp op= val：原地操作只作用于 temp；若 temp 是视图，_inplace_oper_at_
           会合并索引后重定向到 a 上直接执行，此时 a 的数据已被改写；
        3. a.__setitem__(index, temp)：把结果落回 a，即本函数。

        按 temp 的形态分四种处理：
        - 右值不是张量：直接 setat_；
        - 张量但 _base 不是 self：直接 setat_，不涉及原地操作语义；
        - 视图且索引一致（场景 A）：第 2 步已改写 a，直接返回 self；
        - 副本（场景 B）：第 2 步只改了副本，需要把副本上的原地操作重放到 a。

        重放依赖两个前提：副本取自同一索引、且所有原地操作都作用于整个副本；
        否则退化为按数据赋值（_assign_without_cycle）。这样数据始终正确，代价是
        不可重放的场景中“副本 → self”这条梯度路径会被切断。

        参数:
            index: 索引，可以是切片、整数、数组等
            val: 右值，可能是张量或标量
        """
        # 1. 叶子节点检查：所有原地操作都需要这个检查
        # 叶子节点是计算图的起点，直接修改会破坏梯度追踪
        if is_grad_enabled() and self.is_leaf and self.requires_grad:
            raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation.')

        # 2. 快速路径：简单赋值（右值不是张量，没有 _is_view 属性）
        if not hasattr(val, '_is_view'):
            return self.setat_(index, val)

        # 3. 快速路径：视图但基础不是 self（来自其他张量的视图）
        if val._base is not self:
            return self.setat_(index, val)

        # 4. 复杂原地操作场景：视图且基础是 self
        # 这种情况发生在 a[index] += b 这样的原地操作中
        # val 是通过 a.__getitem__(index) 创建的临时张量
        index_val = _convert_TNindex_to_numpy(index)

        # 根据 val 是否是视图，分两种情况处理：
        # 
        # 场景 A：val 是视图（如切片索引 a[0:3] += 1）
        # - __getitem__ 返回的视图与 self 共享内存
        # - 原地操作已在 _inplace_oper_at_ 中被重定向到 self 上执行
        # - 这里只需检查索引匹配，直接返回
        #
        # 场景 B：val 不是视图（如整数数组索引 a[[0,4]] += 1）
        # - __getitem__ 返回的是副本，不共享内存
        # - 原地操作只修改了 val 的数据，self 的数据未变
        # - 需要将 val 中记录的原地操作应用到 self 上
        if val._is_view:
            # 场景 A：视图对象
            # 检查 val 的视图索引与当前索引是否一致
            # 使用 _indexes_are_equal 安全比较（处理布尔数组等复杂索引）
            if _indexes_are_equal(val._view_index, index_val):
                # 索引匹配，原地操作已在 self 上执行，直接返回
                return self
            return self._assign_without_cycle(index, val)

        # 场景 B：非视图对象（如整数数组索引 a[[0,4]] += 1）
        # 副本记录着它相对 self 做过的原地操作，按时间顺序全部重放即可完整还原数据，
        # 且不会在计算图中引入环路。重放的成立条件是：副本取自同一索引，且所有
        # 原地操作都作用于整个副本（见 _collect_replay_ops）。
        replay_ops = self._collect_replay_ops(val, index_val)
        if replay_ops:
            # 记录按“最新在前”存放，重放必须按“最早在前”的顺序执行
            for op_name, right_val in reversed(replay_ops):
                getattr(self, op_name)(index_val, right_val)
            return self

        # 无法安全重放：按数据赋值。若副本的计算图依赖 self，需先断开环路，
        # 否则 self 与副本互相引用，反向传播无法收敛（梯度会整体丢失）。
        return self._assign_without_cycle(index, val)

    def _collect_replay_ops(self, val, index_val):
        """收集 val 上记录的原地操作，供 __setitem__ 按时间顺序重放到 self。

        场景：a[[0,4]] += 1 时 __getitem__ 返回副本 val，原地操作只改了 val，
        因此 self 需要“重演”这些操作才能得到相同数据。重演等价于直接在 self 上
        执行 a[idx] op= 右值，既不改动副本、也不会产生 self 与 val 的循环依赖。

        两个前提缺一不可，否则重演结果与 val 的实际数据不一致：
        1. val 必须取自同一索引（val._view_index == index_val）；
        2. 每条原地操作都必须作用于整个副本（stored_index 为 ()）。
           若副本上做过局部原地操作（如 t[0] += 1），重演需要再合并一层索引，
           这里不做该推导，直接放弃重演。

        原地操作记录连续插入在计算图首部且“最新在前”，遇到第一条非原地记录即停止；
        返回的列表同样保持“最新在前”，调用方需 reversed 后按时间顺序执行。

        参数:
            val: __getitem__ 返回的副本张量。
            index_val: 本次赋值使用的索引（已规范化为 numpy 索引）。

        返回值:
            list: [(method_name, right_val), ...]，无可用重放时为空列表
        """
        if not _indexes_are_equal(getattr(val, '_view_index', None), index_val):
            return []

        ops = []
        for i, func in enumerate(val.gradfuncs):
            op_name = _INPLACE_AT_METHODS.get(getattr(func, '__name__', ''))
            if op_name is None:
                # 原地操作记录连续插入在计算图首部，遇到非原地操作即可停止
                break
            parm = val.parms[i]
            # 原地操作的 parms 为 (target_copy, stored_index)，stored_index 为 ()
            # 表示该操作作用于整个副本；否则说明副本上还做过局部原地操作
            if not (isinstance(parm, tuple) and len(parm) >= 2
                    and isinstance(parm[1], tuple) and len(parm[1]) == 0):
                return []
            ops.append((op_name, val.fromvars[i]))
        return ops

    def _assign_without_cycle(self, index, val):
        """按数据把 val 写入 self，必要时断开 val 对 self 的计算图依赖。

        用于无法重放的场景，例如：
        - t = a[[0,4]]; t += 1; a[1:3] = t（索引不一致）；
        - a[[0,4]] = a[[4,0]]（副本直接回写，没有原地操作记录）。

        直接 setat_ 会让 self.fromvars 包含 val、而 val 的图又包含 self，形成环路。
        反向传播按“消费者梯度到齐才调度”的方式推进，环内节点互相等待，导致整条
        链路的梯度都算不出来（不是算错，而是全丢）。因此检测到依赖时先 detach：
        数据写入仍然正确，只是“副本 → self”这条回传路径被截断。

        注意：detach 只影响该副本回传到 self 的梯度，self 的其他来源（如 x → a）
        不受影响。这也是本实现与 torch 的已知差异，torch 通过不可变的图节点
        天然避免该环。

        参数:
            index: 写入索引
            val: 要写入的右值张量
        """
        if isinstance(val, TN) and val.requires_grad and _graph_contains(val, self):
            val = val.detach()
        return self.setat_(index, val)
        
    def merge_indices(self, base_index, current_index):
        return merge_indices(self, base_index, current_index)
    
    def _adjust_right_val(self, target_data, val):
        # 若 val 已是 TN，则校验设备一致性、按需转换 dtype 后直接复用，避免重复创建
        if isinstance(val, TN):
            right_val = val
            # 检查设备一致性
            if right_val.device != self.device:
                raise ValueError(f"Right value device {right_val.device} does not match target device {self.device}")
            # 检查并转换dtype
            if right_val.dtype != self.dtype:
                right_val = right_val.type(self.dtype)
        else:
            right_val = as_tensor(val, dtype=self.dtype, device=self.device)
        
        # 将右值形状、维度转化为和索引值相匹配的形式
        # 注：target_data 可能是numpy数组（切片/花式索引），也可能是numpy标量（整数索引选中单个元素），
        # 二者都提供 size/shape/ndim，故下列判定对两种情形通用
        target_size = target_data.size
        right_size = right_val._array.size  # 直接访问底层数组的size，避免方法调用开销
        
        if right_size > target_size:
            # 唯一例外：向空张量（size=0）赋值仅含1个元素的右值（标量、形状(1,)/(1,1)等）是合法的
            if target_size == 0 and right_size == 1:
                # 归一为0维：numpy对空选择的赋值只接受0维/1维右值，否则会因右值维数报TypeError
                right_val = right_val._reshape(())
            else:
                raise ValueError(f"Right value size {right_size} exceeds target size {target_size}")
        elif right_size == target_size:
            right_val = right_val._reshape(target_data.shape)
        elif right_val.ndim <= target_data.ndim:
            # _expand 内部不做形状检查，形状不兼容时由 broadcast_to 报错，故无需在此重复校验
            right_val = right_val._expand(target_data.shape)
        else:
            # 维度数过多，先squeeze所有大小为1的维度，再expand
            dim = tuple(i for i, size in enumerate(right_val.shape) if size == 1)
            right_val = right_val._squeeze(dim)
            right_val = right_val._expand(target_data.shape)

        return right_val

    def _inplace_oper_at_(self, index, val, oper_func, backward_func, vector_func=None):
        """在指定索引位置对 self 执行原地运算，并记录反向传播所需的依赖。

        参数:
            index: 目标索引；空元组 () 表示整个张量
            val: 右值，标量或张量
            oper_func: 通用执行函数，签名 (数组, 索引, 右值数组)；
                       必须是 ufunc.at 一类支持重复索引累加的实现
            backward_func: 该操作对应的梯度回调，写入 self.gradfuncs
            vector_func: 可选的全量索引加速函数，签名 (数组, 右值数组)；
                         仅当 index 为 () 时使用，为 None 时全部走 oper_func

        视图重定向：若 self 是视图，先合并视图索引与 index，再在 self._base 上执行，
        保证写入落到真实存储；返回值仍是视图对象本身，调用方无需区分。

        计算图记录顺序“最新在前”：每次原地操作把右值、索引（含左值副本）、梯度回调
        分别插入 fromvars/parms/gradfuncs 的首部，反向传播按逆序处理，从而实现
        “后做的原地操作先处理梯度”。
        """
        # 将索引规范化为numpy索引
        index_val = _convert_TNindex_to_numpy(index)
        
        # 检查是否为视图对象
        if hasattr(self, '_is_view') and self._is_view:
            # 将当前操作索引与视图索引合并
            base_index = self._view_index
                        
            # 调用内部函数合并索引
            combined_index = self.merge_indices(base_index, index_val)
            
            # self是视图时，直接在原张量self._base上执行操作
            self._base._inplace_oper_at_(combined_index, val, oper_func, backward_func, vector_func)
            
            # 返回视图对象本身
            return self
        
        # 将右值形状、维度转化为和索引值相匹配的形式
        target_data = self._array[index_val]
        right_val= self._adjust_right_val(target_data,val)

        # 是否记录计算图：no_grad 下既不记录依赖，也不改动 requires_grad，
        # 与 torch 保持一致（梯度开关不应永久改变张量的属性）
        need_grad = is_grad_enabled() and (self.requires_grad or right_val.requires_grad)

        # target 的来源有两种：
        # 1、基本切片返回视图，与self共享内存
        # 2、整数/花式索引返回副本或numpy标量，本就不共享，但形状/维度信息仍需保留
        # 对于共享内存的视图，必须在改动self前备份一份独立且无依赖的副本：
        # 因为反向传播要用到左值，而原地操作后 self 对应位置的数据已被覆盖
        # 只有 mul/div/pow 的反向需要左值（_LEFT_VALUE_BACKWARD_NAMES），
        # add/sub/set 的反向只用索引，可省去这次整块内存拷贝
        if need_grad and getattr(backward_func, '__name__', '') in _LEFT_VALUE_BACKWARD_NAMES:
            if target_data.base is not None:
                target_data = target_data.copy()  # 复制共享内存的视图，确保独立性
            target_copy = TN(target_data)
        else:
            target_copy = None

        # 空元组索引与省略号均表示对整个张量操作，不存在重复索引，可直接用向量化版本，
        # 避免 ufunc.at 的逐元素慢路径；其余索引保持 ufunc.at 的累加语义
        if vector_func is not None and (
                index_val is Ellipsis
                or (isinstance(index_val, tuple) and len(index_val) == 0)):
            vector_func(self._array, right_val._array)
        else:
            oper_func(self._array, index_val, right_val._array)  # 执行原地操作
        
        if need_grad:
            # 原地赋值时，须在self的现有计算图上增加依赖关系，注意:
            # 1、只能记录右值的依赖关系，不能记录self，避免依赖循环
            # 2、右值的梯度回调函数里同时传递右值梯度和更新self的梯度，所以右值requires_grad=False时，
            #    需要复制一个右值的副本用于跟踪梯度，将副本的requires_grad置True，以便backward里副本的梯度跟踪函数被调用
            # 3、原地赋值依赖关系需要优先处理，须插入到fromvars元组前面
            # 4、多次原地赋值时，遵循先赋值的右值后处理梯度跟踪原则，要从0位置插入右值、索引、梯度回调函数
            # 5、若右值依赖 self（如 t = a[[0,4]]; a[1:3] = t），这里会形成环；
            #    该情况由 __setitem__ 的 _assign_without_cycle 提前拦截，不会走到这里
            if right_val.requires_grad == False:
                # 需要一个用于传播梯度的右值变量，但又不要改动右值的属性，detach一个新变量是一个经济、可行的办法
                right_val = right_val.detach().requires_grad_(True)

            self.requires_grad = True
            self.fromvars = (right_val,) + self.fromvars
            # 注意(target_copy,index_val)必须以元组的形式整体插入到parms元组前面
            self.parms = ((target_copy,index_val),) + self.parms
            self.gradfuncs = (backward_func, ) + self.gradfuncs

        return self

    def _inplace_unary_at_(self, index, data_func, backward_func):
        """在指定索引位置对 self 执行一元原地运算，并记录反向传播所需的依赖。

        与 _inplace_oper_at_ 的区别：一元运算没有右值，data_func 直接依据左值
        改写数据，因此梯度只需修正 self 自身历史算子的梯度（由 backward_func
        完成），不用记录右值依赖；占位变量仅用于承载该梯度回调。

        参数:
            index: 目标索引；空元组 () 表示整个张量
            data_func: 执行函数，签名 (数组, 索引)，直接在数组上改写数据
            backward_func: 该操作对应的梯度回调，写入 self.gradfuncs

        视图重定向：若 self 是视图，先合并视图索引与 index，再在 self._base 上执行，
        保证写入落到真实存储；返回值仍是视图对象本身。
        """
        # 将索引规范化为numpy索引
        index_val = _convert_TNindex_to_numpy(index)

        # 检查是否为视图对象
        if hasattr(self, '_is_view') and self._is_view:
            combined_index = self.merge_indices(self._view_index, index_val)
            self._base._inplace_unary_at_(combined_index, data_func, backward_func)
            return self

        # 是否记录计算图：no_grad 下既不记录依赖，也不改动 requires_grad
        need_grad = is_grad_enabled() and self.requires_grad

        # 反向需要操作前的左值（如 abs 的符号），故在改写数据前备份一份独立副本
        target_data = self._array[index_val]
        if need_grad:
            if target_data.base is not None:
                target_data = target_data.copy()  # 复制共享内存的视图，确保独立性
            target_copy = TN(target_data)

        data_func(self._array, index_val)  # 执行原地操作

        if need_grad:
            # 一元原地运算没有右值，用不依赖任何输入的占位变量承载梯度回调；
            # 插入位置与 _inplace_oper_at_ 一致，保证“后做的原地操作先处理梯度”
            placeholder = zeros((), dtype=self.dtype, device=self.device, requires_grad=True)
            self.requires_grad = True
            self.fromvars = (placeholder,) + self.fromvars
            self.parms = ((target_copy,index_val),) + self.parms
            self.gradfuncs = (backward_func, ) + self.gradfuncs

        return self

    def _non_inplace_oper_at(self, index, val, oper_func, left_backward_func, right_backward_func):
        # 将索引规范化为numpy索引
        index_val = _convert_TNindex_to_numpy(index)
        
        # 将右值形状、维度转化为和索引值相匹配的形式
        target = self._array[index_val]        
        right_val= self._adjust_right_val(target, val)

        # 新建一个self张量的独立副本，用于存储结果
        ret = self.copy()
        oper_func(ret._array, index_val, right_val._array)

        ret.requires_grad = (is_grad_enabled() and (self.requires_grad or right_val.requires_grad))
        
        if ret.requires_grad:
            ret.fromvars = (self,right_val)
            ret.parms = (index_val,)
            ret.gradfuncs = (left_backward_func,right_backward_func)

        return ret

    # 原地/非原地成对接口：xxxat_ 直接在 self 上执行并记录依赖、返回 self；
    # xxxat 在副本上执行、返回新张量，self 不变。
    # 每对接口都提供两种执行函数，由 _inplace_oper_at_ 按索引选择：
    # - oper_func：通用路径，基于 ufunc.at，保证花式索引“重复索引累加”的语义；
    # - vector_func：仅用于 index 为 () 的全量操作，走向量化原地运算符
    #   （见文件末尾 _iadd_full 等），避免 ufunc.at 的逐元素慢路径。
    def setat_(self, index, val):
        """原地执行 self[index] = val（覆盖写），返回 self。"""
        def _set_numpy_item(numpy_arr, index, right_numpy_arr):
            numpy_arr[index] = right_numpy_arr
            return numpy_arr
        
        return self._inplace_oper_at_(index, val, _set_numpy_item, _setat_inplace_backward, _iset_full)

    def setat(self, index, val):
        def _set_numpy_item(numpy_arr, index, right_numpy_arr):
            numpy_arr[index] = right_numpy_arr
            return numpy_arr
        
        return self._non_inplace_oper_at(index, val, _set_numpy_item, _setat_backward_left, _setat_backward_right)

    def addat_(self,index,val):
        """原地执行 self[index] += val（重复索引累加），返回 self。"""
        arrlib = self._get_array_lib()
        return self._inplace_oper_at_(index, val, arrlib.add.at, _addat_inplace_backward, _iadd_full)

    def addat(self,index,val):
        arrlib = self._get_array_lib()
        return self._non_inplace_oper_at(index, val, arrlib.add.at, _addat_backward_left, _addat_backward_right)

    def subat_(self,index,val):
        """原地执行 self[index] -= val（重复索引累加），返回 self。

        cupy 的 subtract.at 支持不完整，故 cupy 分支改用切片原地运算。
        """
        arrlib = self._get_array_lib()
        if arrlib == cp:
            def _subat_numpy_item(numpy_arr, index, right_numpy_arr):
                numpy_arr[index] -= right_numpy_arr
                return numpy_arr
            oper_func = _subat_numpy_item
        else:
            oper_func = np.subtract.at
        return self._inplace_oper_at_(index, val, oper_func, _subat_inplace_backward, _isub_full)

    def subat(self,index,val):
        arrlib = self._get_array_lib()
        if arrlib == cp:
            def _subat_numpy_item(numpy_arr, index, right_numpy_arr):
                numpy_arr[index] -= right_numpy_arr
                return numpy_arr
            oper_func = _subat_numpy_item
        else:
            oper_func = np.subtract.at
        return self._non_inplace_oper_at(index, val, oper_func, _subat_backward_left, _subat_backward_right)
    
    def mulat_(self,index,val):
        """原地执行 self[index] *= val（重复索引累加），返回 self。

        反向传播需要操作前的左值，target_copy 会在这里被备份。
        """
        arrlib = self._get_array_lib()
        if arrlib == cp:
            def _mulat_numpy_item(numpy_arr, index, right_numpy_arr):
                numpy_arr[index] *= right_numpy_arr
                return numpy_arr
            oper_func = _mulat_numpy_item
        else:
            oper_func = np.multiply.at
            
        return self._inplace_oper_at_(index, val, oper_func, _mulat_inplace_backward, _imul_full)

    def mulat(self,index,val):
        arrlib = self._get_array_lib()
        if arrlib == cp:
            def _mulat_numpy_item(numpy_arr, index, right_numpy_arr):
                numpy_arr[index] *= right_numpy_arr
                return numpy_arr
            oper_func = _mulat_numpy_item
        else:
            oper_func = np.multiply.at
        
        return self._non_inplace_oper_at(index, val, oper_func, _mulat_backward_left, _mulat_backward_right)

    def divat_(self,index,val):
        """原地执行 self[index] /= val（重复索引累加），返回 self。

        反向传播需要操作前的左值，target_copy 会在这里被备份。
        """
        arrlib = self._get_array_lib()
        if arrlib == cp:
            def _divat_numpy_item(numpy_arr, index, right_numpy_arr):
                numpy_arr[index] /= right_numpy_arr
                return numpy_arr
            oper_func = _divat_numpy_item
        else:
            oper_func = np.divide.at
        return self._inplace_oper_at_(index, val, oper_func, _divat_inplace_backward, _idiv_full)

    def divat(self,index,val):
        arrlib = self._get_array_lib()
        if arrlib == cp:
            def _divat_numpy_item(numpy_arr, index, right_numpy_arr):
                numpy_arr[index] /= right_numpy_arr
                return numpy_arr
            oper_func = _divat_numpy_item
        else:
            oper_func = np.divide.at
        return self._non_inplace_oper_at(index, val, oper_func, _divat_backward_left, _divat_backward_right)

    def powat_(self,index,val):
        """原地执行 self[index] **= val（重复索引累加），返回 self。

        反向传播需要操作前的左值，target_copy 会在这里被备份。
        """
        arrlib = self._get_array_lib()
        if arrlib == cp:
            def _powat_numpy_item(numpy_arr, index, right_numpy_arr):
                numpy_arr[index] **= right_numpy_arr
                return numpy_arr
            oper_func = _powat_numpy_item
        else:
            oper_func = np.power.at
        return self._inplace_oper_at_(index, val, oper_func, _powat_inplace_backward, _ipow_full)

    def powat(self,index,val):
        arrlib = self._get_array_lib()
        if arrlib == cp:
            def _powat_numpy_item(numpy_arr, index, right_numpy_arr):
                numpy_arr[index] **= right_numpy_arr
                return numpy_arr
            oper_func = _powat_numpy_item
        else:
            oper_func = np.power.at
        return self._non_inplace_oper_at(index, val, oper_func, _powat_backward_left, _powat_backward_right)

    def floordivat_(self,index,val):
        """原地执行 self[index] //= val（重复索引累加），返回 self。

        向下取整除法不可导，反向传播与 torch 一致，在 backward 时抛出异常，
        因此无需备份左值。
        """
        arrlib = self._get_array_lib()
        if arrlib == cp:
            def _floordivat_numpy_item(numpy_arr, index, right_numpy_arr):
                numpy_arr[index] //= right_numpy_arr
                return numpy_arr
            oper_func = _floordivat_numpy_item
        else:
            oper_func = np.floor_divide.at
        return self._inplace_oper_at_(index, val, oper_func, _floordivat_inplace_backward, _ifloordiv_full)

    def floordivat(self,index,val):
        arrlib = self._get_array_lib()
        if arrlib == cp:
            def _floordivat_numpy_item(numpy_arr, index, right_numpy_arr):
                numpy_arr[index] //= right_numpy_arr
                return numpy_arr
            oper_func = _floordivat_numpy_item
        else:
            oper_func = np.floor_divide.at
        return self._non_inplace_oper_at(index, val, oper_func, _floordiv_grad_left, _floordiv_grad_right)

    def modat_(self,index,val):
        """原地执行 self[index] %= val（重复索引累加），返回 self。

        反向传播需要操作前的左值，target_copy 会在这里被备份。
        """
        arrlib = self._get_array_lib()
        if arrlib == cp:
            def _modat_numpy_item(numpy_arr, index, right_numpy_arr):
                numpy_arr[index] %= right_numpy_arr
                return numpy_arr
            oper_func = _modat_numpy_item
        else:
            oper_func = np.remainder.at
        return self._inplace_oper_at_(index, val, oper_func, _modat_inplace_backward, _imod_full)

    def modat(self,index,val):
        arrlib = self._get_array_lib()
        if arrlib == cp:
            def _modat_numpy_item(numpy_arr, index, right_numpy_arr):
                numpy_arr[index] %= right_numpy_arr
                return numpy_arr
            oper_func = _modat_numpy_item
        else:
            oper_func = np.remainder.at
        return self._non_inplace_oper_at(index, val, oper_func, _modat_backward_left, _modat_backward_right)

    def gather(self, dim: int, index:TN):
        """
        根据指定的 ​维度和​索引收集元素，
        常用于需要根据条件动态选择数据的场景（如交叉熵损失函数中选择正确类别的预测概率）
        等价于模块级函数 riemann.gather(self, dim, index)
        :param dim: 收集维度（0 ≤ dim < self.ndim）
        :param index: 索引张量（dtype=int64）
        :return: 新TN实例，形状与index相同
        """
        return self._run_registered_api('gather', self, dim, index)

    def index_select(self, dim: int, index: TN) -> 'TN':
        """沿指定维度按索引选择元素，等价于 riemann.index_select(self, dim, index)"""
        return self._run_registered_api('index_select', self, dim, index)

    def masked_select(self, mask: TN) -> 'TN':
        """按布尔掩码选择元素，返回一维张量，等价于 riemann.masked_select(self, mask)"""
        return self._run_registered_api('masked_select', self, mask)

    def take(self, index: TN) -> 'TN':
        """按扁平化下标选择元素，返回与 index 同形状的张量，等价于 riemann.take(self, index)"""
        return self._run_registered_api('take', self, index)

    def index_put_(self, indices, values, accumulate: bool = False) -> 'TN':
        """
        按索引把 values 写入当前张量（原地操作），与 torch.Tensor.index_put_ 语义一致

        参数:
            indices: 索引元组，每个元素为整数索引张量或布尔掩码张量
            values: 写入值，可为标量、TN 或可转换为 TN 的数据，须可广播到被选中区域；
                当传入 TN 时，其 dtype 须与当前张量一致
            accumulate: 为 True 时按 addat_ 语义累加（重复索引累加），否则按 setat_ 覆盖

        返回:
            原地修改后的张量（self）
        """
        return self._run_registered_api('index_put_', self, indices, values, accumulate)

    def scatter(self, dim, index, src=None, *, value=None):
        """
        将值按照index指定的位置填充到与self同形状的一个新张量中
        
        参数:
            dim: 沿着哪个维度进行索引
            index: 索引张量
            src: 源张量，提供要填充的值，与value参数互斥；当为标量且value未提供时，视为value参数
            value: 标量值（命名参数），提供要填充的值，与src参数互斥
        
        返回:
            新的填充后的张量
        """
        return self._run_registered_api('scatter', self, dim, index, src, value=value)

    def scatter_(self, dim, index, src=None, *, value=None):
        """
        将值按照index指定的位置原地填充到张量self中
        
        参数:
            dim: 沿着哪个维度进行索引
            index: 索引张量
            src: 源张量，提供要填充的值，与value参数互斥；当为标量且value未提供时，视为value参数
            value: 标量值（命名参数），提供要填充的值，与src参数互斥
        
        返回:
            原地修改后的张量（self）
        """
        return self._run_registered_api('scatter_', self, dim, index, src, value=value)

    def scatter_add(self, dim, index, src):
        """
        将src中的值按照index指定的位置累加到与self同形状的新张量中
        
        参数:
            dim: 沿着张量src的哪个维度进行索引
            index: 索引张量，形状必须与src（如果是张量）兼容，维度数必须与self相同
            src: 源张量或标量，提供要累加的值
        
        返回:
            原地修改后的张量（self）
        """
        return self._run_registered_api('scatter_add', self, dim, index, src)

    def scatter_add_(self, dim, index, src):
        """
        将src中的值按照index指定的位置原地累加到self张量中
        
        参数:
            dim: 沿着张量src的哪个维度进行索引
            index: 索引张量，形状必须与src（如果是张量）兼容，维度数必须与self相同
            src: 源张量或标量，提供要累加的值
        
        返回:
            原地修改后的张量（self）
        """
        return self._run_registered_api('scatter_add_', self, dim, index, src)

    def numpy(self):
        """
        将张量转换为NumPy数组。
        
        返回张量数据的NumPy数组表示。此操作会断开计算图，
        因此返回的NumPy数组不会参与梯度计算。
        
        Returns:
            np.ndarray: 张量数据的NumPy数组表示
            
        Raises:
            RuntimeError: 如果张量需要计算梯度
        """
        if self.requires_grad:
            raise RuntimeError("Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.")
        # 与 PyTorch 语义一致：CUDA tensor 的 .numpy() 抛 TypeError，
        # 强制用户显式调用 .cpu().numpy() 执行 GPU→CPU 数据迁移。
        if self.is_cuda:
            raise TypeError(
                "can't convert cuda device type tensor to numpy. "
                "Use Tensor.cpu() to copy the tensor to host memory first."
            )
        # CPU tensor 直接返回底层 np.ndarray
        return self._array
    
    def array(self):
        """
        返回张量底层的设备原生数组，零拷贝、与张量共享内存。

        与 .numpy() 的区别：
        - .numpy() 始终返回 np.ndarray，CUDA 张量会抛出 TypeError ；
        - .array() 返回张量所在设备的原生数组类型：CPU 上是 np.ndarray，
          CUDA 上是 cp.ndarray。不做任何设备迁移或数据拷贝，与张量共享底层存储。

        典型用途：
        - 验证 as_tensor / torch.from_dlpack 等操作的内存共享语义；
        - 将 TN 张量零拷贝交给 CuPy / NumPy 原生函数处理。

        Returns:
            np.ndarray | cp.ndarray: 与张量共享内存的底层数组对象。
            CPU 张量返回 np.ndarray，CUDA 张量返回 cp.ndarray。

        Examples:
            >>> cpu_t = rm.tensor([1.0, 2.0, 3.0])
            >>> cpu_arr = cpu_t.array()   # np.ndarray, 与 cpu_t 共享内存
            >>> cpu_arr[0] = 42.0
            >>> cpu_t[0]                  # TN 也看到修改

            >>> cuda_t = rm.tensor([1.0, 2.0], device='cuda')
            >>> cuda_arr = cuda_t.array() # cp.ndarray, 与 cuda_t 共享内存
            >>> type(cuda_arr)            # cupy.ndarray

            >>> # 配合 .detach() 在需要梯度的张量上使用
            >>> t = rm.tensor([1.0, 2.0], requires_grad=True)
            >>> arr = t.detach().array()  # OK
        """
        return self._array
        
    def tolist(self):
        """
        将张量转换为Python列表。
        
        返回张量数据的Python列表表示。对于标量张量，返回Python标量值；
        对于多维张量，返回嵌套列表。此操作会断开计算图，因此返回的列表不会参与梯度计算。
        
        Returns:
            list | int | float | complex: 张量数据的Python列表或标量表示
            
        Raises:
            RuntimeError: 如果张量需要计算梯度
        """
        if self.requires_grad:
            raise RuntimeError("Can't call tolist() on Tensor that requires grad. Use tensor.detach().tolist() instead.")
        
        # 直接使用NumPy的tolist()方法，它会自动处理标量和多维数组
        return self._array.tolist()
    
    def requires_grad_(self, requires_grad : bool = True):  # type: ignore
        """
        就地设置张量是否需要计算梯度。
        
        仅适用于叶子节点张量，非叶子节点张量不能修改requires_grad属性。
        只有浮点类型和复数类型的张量可以设置requires_grad为True。
        
        Args:
            requires_grad (bool, optional): 是否需要计算梯度，默认为True
            
        Raises:
            RuntimeError: 如果张量不是叶子节点或尝试为非浮点/复数类型张量设置requires_grad=True
        """
        if not self.is_leaf:
            raise RuntimeError('requires_grad_:you can only change requires_grad flags of leaf variables.')
        
        requires_grad = bool(requires_grad)
        if requires_grad:
            if not is_float_or_complex(self.dtype):
                raise RuntimeError('requires_grad_:Only floating point tensors can require gradients')
        
        self.requires_grad = requires_grad
        return self
    
    def retain_grad(self):
        """
        设置张量在反向传播后保留梯度。
        
        默认情况下，只有叶子节点张量在反向传播后会保留梯度值。
        调用此方法可以使非叶子节点张量也保留梯度值，便于调试和分析。
        
        Raises:
            RuntimeError: 如果张量的requires_grad为False
        """
        if not self.requires_grad:
            raise RuntimeError("'can't retain_grad on Tensor that has requires_grad=False")
        self.retains_grad = True
        return

    def transpose(self,dim1:int,dim2:int):
        """
        交换张量的两个维度。
        
        返回一个新张量，其中指定的两个维度被交换。对于2D张量，
        transpose(0, 1)等同于矩阵转置。对于高维张量，可以交换任意两个维度。
        
        Args:
            dim1 (int): 要交换的第一个维度索引
            dim2 (int): 要交换的第二个维度索引
            
        Returns:
            TN: 交换指定维度后的新张量
            
        Raises:
            ValueError: 如果张量维度小于2或维度索引超出范围
        """
        if self._array.ndim >= 2:    #只有3D以上多维数组才支持批量矩阵转置
            trandata = self._array.swapaxes(dim1, dim2)
            
            #转置产生一个新对象，但与原对象共用内存数据
            newobj = TN(trandata, 
                        requires_grad = (is_grad_enabled() and self.requires_grad)
            )
            
            if newobj.requires_grad:
                newobj.fromvars = (self,)
                newobj.parms = ((dim1,dim2),)
                newobj.gradfuncs = (_transpose_backward,)
            
            return newobj
        else:
            raise RuntimeError("transpose() can only be used on tensors of 2D or more dimension")

    def trace(self):
        '''
        计算张量的迹（对角线元素的和）
        '''
        return api_registry.get("trace")(self)

    @property
    def mT(self) -> TN:
        """
        矩阵转置，即张量最后两个维度间的转置。
        
        对于高维张量，此属性只交换最后两个维度，保持其他维度不变。
        例如，对于一个形状为(3, 4, 5)的张量，mT的结果形状为(3, 5, 4)。
        对于2D张量，这等同于标准的矩阵转置。
        
        Returns:
            TN: 最后两个维度转置后的新张量
            
        Raises:
            RuntimeError: 如果张量维度小于2
        """
        return self.transpose(-1,-2)
    
    @property
    def mH(self) -> TN:
        """
        矩阵共轭转置，即张量最后两个维度间的共轭转置。
        
        对于复数张量，此属性先对张量进行共轭操作，然后交换最后两个维度。
        对于实数张量，这等同于mT属性，因为实数的共轭是其本身。
        
        Returns:
            TN: 最后两个维度共轭转置后的新张量
            
        Raises:
            RuntimeError: 如果张量维度小于2
        """
        return self.conj().mT

    @property
    def T(self) -> TN:
        """
        返回张量的转置
        - 对于一维张量，返回原张量（不变）
        - 对于二维张量，交换两个维度（标准矩阵转置）
        - 对于高维张量，反转整个维度顺序
        支持自动梯度跟踪
        """
        # 构建转置维度顺序 - 反转整个维度顺序
        transpose_dims = list(reversed(range(self.ndim)))
        
        # 使用permute函数实现转置
        return self.permute(transpose_dims)  # type: ignore

    @property
    def H(self) -> TN:
        """
        全维度共轭转置，张量共轭后反转整个维度顺序。
        
        对于复数张量，此属性先对张量进行共轭操作，然后反转整个维度顺序。
        对于实数张量，这等同于T属性，因为实数的共轭是其本身。
        与mH属性不同，H属性会反转所有维度，而不仅仅是最后两个维度。
        
        Returns:
            TN: 共轭转置后的新张量
        """
        return self.conj().T
    
    def permute(self, *dims: int|tuple) -> 'TN':
        """
        使用numpy.transpose直接实现高效的维度重排
        参数示例：对于形状(2,3,4)的输入，dims=(2,0,1)将输出形状变为(4,2,3)
        支持自动梯度跟踪，保持完整的计算图用于高阶导数计算
        """

        dims = _validate_shape(dims)

        # 校验维度数量
        if len(dims) != self.ndim:
            raise ValueError(f"Dimension mismatch: got {len(dims)}, expected {self.ndim}")
        
        # 转换负数索引为正数索引
        pos_dims = []
        for dim in dims:
            pos_dim = dim + self.ndim if dim < 0 else dim  # type: ignore
            # 检查维度是否在有效范围内
            if pos_dim < 0 or pos_dim >= self.ndim:  # type: ignore
                raise ValueError(f"Dimension {dim} out of range for tensor with {self.ndim} dimensions")
            pos_dims.append(pos_dim)
        
        # 检查是否包含所有轴且无重复
        if sorted(pos_dims) != list(range(self.ndim)):
            raise ValueError("Dimension order must contain all axes without repetition")
        
        # 使用numpy.transpose直接计算维度重排
        new_data = self._array.transpose(dims)
        
        # 创建新张量
        requires_grad = is_grad_enabled() and self.requires_grad
        newobj = TN(new_data, requires_grad = requires_grad)
        
        # 如果需要梯度，设置梯度跟踪信息
        if requires_grad:
            newobj.fromvars = (self,)
            newobj.parms = (dims,)
            newobj.gradfuncs = (_permute_backward,)
        
        return newobj

    def neg_(self):
        """
        就地取负操作，等价于 torch.Tensor.neg_，直接修改当前张量的数据。

        布尔张量不支持取负（与 torch 一致）；无符号整型按位宽回绕并保持 dtype。

        Returns:
            TN: 当前张量（已就地修改）

        Raises:
            RuntimeError: 如果当前张量是需要梯度的叶子节点
            NotImplementedError: 如果当前张量是布尔类型
        """
        if is_grad_enabled() and self.is_leaf and self.requires_grad:
            raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation.')

        # 布尔张量取负没有意义，torch 同样不支持；取反请用 ~ 或 bitwise_not()
        if np.issubdtype(self.dtype, np.bool_):
            raise NotImplementedError(
                "Negation, the `-` operator, on a bool tensor is not supported. "
                "If you are trying to invert a mask, use the `~` or `bitwise_not()` operator instead."
            )

        # 一元原地运算没有右值，传入空索引表示对整个视图操作（视图索引由
        # _inplace_unary_at_ 自动合并），无符号整型取负由底层按位宽回绕
        return self._inplace_unary_at_((), _neg_inplace_data, _negat_inplace_backward)

    def abs_(self):
        """
        就地取绝对值操作，等价于 torch.Tensor.abs_，直接修改当前张量的数据。

        无符号整型与布尔以外类型保持 dtype 不变。

        Returns:
            TN: 当前张量（已就地修改）

        Raises:
            RuntimeError: 如果当前张量是需要梯度的叶子节点，或复数类型（与 torch 一致）
            NotImplementedError: 如果当前张量是布尔类型
        """
        if is_grad_enabled() and self.is_leaf and self.requires_grad:
            raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation.')

        # torch 的 abs_ 不支持布尔与复数类型
        if np.issubdtype(self.dtype, np.bool_):
            raise NotImplementedError("abs_ is not implemented for bool tensors")
        if np.issubdtype(self.dtype, np.complexfloating):
            raise RuntimeError("In-place abs is not supported for complex tensors.")

        return self._inplace_unary_at_((), _abs_inplace_data, _absat_inplace_backward)

    def add_(self,other):
        """
        就地加法操作，将另一个张量或标量值加到当前张量上。
        
        直接修改当前张量的数据，而不是创建新张量。对于需要梯度的叶子节点张量，
        不允许进行就地操作，因为这会破坏计算图。
        
        Args:
            other (TN or scalar): 要加到当前张量上的张量或标量值
            
        Returns:
            TN: 当前张量（已就地修改）
            
        Raises:
            RuntimeError: 如果当前张量是需要梯度的叶子节点
        """
        if is_grad_enabled() and self.is_leaf and self.requires_grad:
            raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation.')
        # 对于in-place操作，我们总是对整个视图进行操作，所以传递空索引
        # 视图的原始索引会在_inplace_oper_at_方法中自动处理
        return self.addat_((), other)
    
    def sub_(self,other):
        """
        就地减法操作，从当前张量中减去另一个张量或标量值。
        
        直接修改当前张量的数据，而不是创建新张量。对于需要梯度的叶子节点张量，
        不允许进行就地操作，因为这会破坏计算图。
        
        Args:
            other (TN or scalar): 要从当前张量中减去的张量或标量值
            
        Returns:
            TN: 当前张量（已就地修改）
            
        Raises:
            RuntimeError: 如果当前张量是需要梯度的叶子节点
        """
        if is_grad_enabled() and self.is_leaf and self.requires_grad:
            raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation.')
        # 对于in-place操作，我们总是对整个视图进行操作，所以传递空索引
        return self.subat_((), other)

    def mul_(self,other):
        """
        就地乘法操作，将当前张量与另一个张量或标量值相乘。
        
        直接修改当前张量的数据，而不是创建新张量。对于需要梯度的叶子节点张量，
        不允许进行就地操作，因为这会破坏计算图。
        
        Args:
            other (TN or scalar): 要与当前张量相乘的张量或标量值
            
        Returns:
            TN: 当前张量（已就地修改）
            
        Raises:
            RuntimeError: 如果当前张量是需要梯度的叶子节点
        """
        if is_grad_enabled() and self.is_leaf and self.requires_grad:
            raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation.')
        # 对于in-place操作，我们总是对整个视图进行操作，所以传递空索引
        return self.mulat_((), other)
    
    def div_(self,other):
        """
        就地除法操作，将当前张量除以另一个张量或标量值。
        
        直接修改当前张量的数据，而不是创建新张量。对于需要梯度的叶子节点张量，
        不允许进行就地操作，因为这会破坏计算图。
        
        Args:
            other (TN or scalar): 要除以的张量或标量值
            
        Returns:
            TN: 当前张量（已就地修改）
            
        Raises:
            RuntimeError: 如果当前张量是需要梯度的叶子节点
        """
        if is_grad_enabled() and self.is_leaf and self.requires_grad:
            raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation.')
        # 对于in-place操作，我们总是对整个视图进行操作，所以传递空索引
        return self.divat_((), other)
    
    def pow_(self,other):
        """
        就地幂运算操作，将当前张量的元素进行幂运算。
        
        直接修改当前张量的数据，而不是创建新张量。对于需要梯度的叶子节点张量，
        不允许进行就地操作，因为这会破坏计算图。
        
        Args:
            other (TN or scalar): 幂运算的指数，可以是张量或标量值
            
        Returns:
            TN: 当前张量（已就地修改）
            
        Raises:
            RuntimeError: 如果当前张量是需要梯度的叶子节点
        """
        if is_grad_enabled() and self.is_leaf and self.requires_grad:
            raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation.')
        # 对于in-place操作，我们总是对整个视图进行操作，所以传递空索引
        return self.powat_((), other)

    def __iadd__(self,other):
        return self.add_(other)
    
    def __isub__(self,other):
        return self.sub_(other)
    
    def __imul__(self,other):
        return self.mul_(other)
    
    def __itruediv__(self,other):
        return self.div_(other)
    
    def __ipow__(self,other):
        return self.pow_(other)

    def floor_divide_(self,other):
        """
        就地向下取整除法操作，将当前张量与另一个张量或标量值做整除。
        
        直接修改当前张量的数据，而不是创建新张量。对于需要梯度的叶子节点张量，
        不允许进行就地操作，因为这会破坏计算图。
        
        Args:
            other (TN or scalar): 除数，可以是张量或标量值
            
        Returns:
            TN: 当前张量（已就地修改）
            
        Raises:
            RuntimeError: 如果当前张量是需要梯度的叶子节点
        """
        if is_grad_enabled() and self.is_leaf and self.requires_grad:
            raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation.')
        return self.floordivat_((), other)

    def remainder_(self,other):
        """
        就地取余操作，将当前张量对另一个张量或标量值取余。
        
        直接修改当前张量的数据，而不是创建新张量。对于需要梯度的叶子节点张量，
        不允许进行就地操作，因为这会破坏计算图。
        
        Args:
            other (TN or scalar): 取余的除数，可以是张量或标量值
            
        Returns:
            TN: 当前张量（已就地修改）
            
        Raises:
            RuntimeError: 如果当前张量是需要梯度的叶子节点
        """
        if is_grad_enabled() and self.is_leaf and self.requires_grad:
            raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation.')
        return self.modat_((), other)

    def __ifloordiv__(self,other):
        return self.floor_divide_(other)
    
    def __imod__(self,other):
        return self.remainder_(other)

    # '+'运算，左值为self，右值为TN，numpy数组，list，tuple，整数或浮点数
    def __add__(self,right_obj):
        return add(self,right_obj)
    
    # '+'运算，右值为self，左值为TN，numpy数组，list，tuple，整数或浮点数
    def __radd__(self,left_obj):
        return add(self,left_obj) #归一到模块函数add

    # '-'运算，左值为self，右值为TN，numpy数组，list，tuple，整数或浮点数
    def __sub__(self,right_obj):
        return sub(self,right_obj)
    
    # '-'运算，右值为self，左值为TN，numpy数组，list，tuple，整数或浮点数
    def __rsub__(self,left_obj):
        return sub(left_obj,self) #保持减数顺序：left_obj - self
    
    # '*'运算，左值为self，右值为TN，numpy数组，list，tuple，整数或浮点数
    def __mul__(self,right_obj):
        return mul(self,right_obj)
    
    # '*'运算，右值为self，左值为TN，numpy数组，list，tuple，整数或浮点数
    def __rmul__(self,left_obj):
        return mul(self,left_obj) #归一到模块函数mul
    
    # '@'运算，左值为self，右值为TN，numpy数组，list，tuple，整数或浮点数
    def __matmul__(self,right_obj):
        return matmul(self,right_obj)

    # '@'运算，右值为self，左值为TN，numpy数组，list，tuple，整数或浮点数
    def __rmatmul__(self,left_obj):
        return matmul(left_obj,self) #保持左矩阵顺序：left_obj @ self
    

    # '/'运算，左值为self，右值为TN，numpy数组，list，tuple，整数或浮点数
    def __truediv__(self,right_obj):
        return div(self,right_obj)
    
    # '/'运算，右值为self，左值为TN，numpy数组，list，tuple，整数或浮点数
    def __rtruediv__(self,left_obj):
        return div(left_obj,self) #保持除数顺序：left_obj / self

    # '//'运算，左值为self，右值为TN，numpy数组，list，tuple，整数或浮点数
    def __floordiv__(self,right_obj):
        return floor_divide(self,right_obj)

    # '//'运算，右值为self，左值为TN，numpy数组，list，tuple，整数或浮点数
    def __rfloordiv__(self,left_obj):
        return floor_divide(left_obj,self) #保持被除数顺序：left_obj // self

    # '%'运算，左值为self，右值为TN，numpy数组，list，tuple，整数或浮点数
    def __mod__(self,right_obj):
        return remainder(self,right_obj)

    # '%'运算，右值为self，左值为TN，numpy数组，list，tuple，整数或浮点数
    def __rmod__(self,left_obj):
        return remainder(left_obj,self) #保持被除数顺序：left_obj % self

    # pow运算，左值为self，右值为TN，numpy数组，list，tuple，整数或浮点数
    def __pow__(self,right_obj):
        return pow(self,right_obj)
    
    # pow运算，右值为self，左值为TN，numpy数组，list，tuple，整数或浮点数
    def __rpow__(self,left_obj):
        return pow(left_obj,self) #保持底数顺序：left_obj ** self
        
    def __pos__(self):
        return positive(self)
        
    def __neg__(self):
        return neg(self)
    
    
    # 比较运算符重载（返回布尔张量，不参与梯度计算）
    def __lt__(self, other):
        if other is None:
            raise TypeError(" '<' not supported between instances of 'TN' and 'NoneType'")
        return lt(self, other)

    def __le__(self, other):
        if other is None:
            raise TypeError(" '<=' not supported between instances of 'TN' and 'NoneType'")
        return le(self, other)

    def __gt__(self, other):
        if other is None:
            raise TypeError(" '>' not supported between instances of 'TN' and 'NoneType'")
        return gt(self, other)

    def __ge__(self, other):
        if other is None:
            raise TypeError(" '>=' not supported between instances of 'TN' and 'NoneType'")
        return ge(self, other)

    def __eq__(self, other):
        if other is None:
            return False
        return eq(self, other)

    def __ne__(self, other):
        if other is None:
            return True
        return ne(self, other)

    def __and__(self, other):
        return bitwise_and(self, other)

    def __or__(self, other):
        return bitwise_or(self, other)
    
    def __xor__(self, other):
        return bitwise_xor(self, other)
    
    def __invert__(self):
        return bitwise_not(self)
    
    def __lshift__(self, other):
        return bitwise_left_shift(self, other)
    
    def __rshift__(self, other):
        return bitwise_right_shift(self, other)

    def __rand__(self, other):
        return bitwise_and(self, other)

    def __ror__(self, other):
        return bitwise_or(self, other)

    def __rxor__(self, other):
        return bitwise_xor(self, other)

    def __rlshift__(self, other):
        return bitwise_left_shift(other, self)

    def __rrshift__(self, other):
        return bitwise_right_shift(other, self)
    
    def detach(self):
        """
        返回一个与当前张量共享数据但断开计算图的新张量。
        
        返回的新张量与原张量共享底层数据，但不参与原张量的计算图。
        这意味着对新张量的操作不会影响原张量的梯度计算。
        常用于在需要将张量转换为NumPy数组或进行不需要梯度的计算时。
        
        Returns:
            TN: 与原张量共享数据但断开计算图的新张量
        """
        return TN(self._array)
        
    @property
    def data(self):
        """
        返回与当前张量共享底层数据、但已脱离计算图的新张量。

        行为与 PyTorch 的 Tensor.data 一致，等价于 detach()：
        新张量与当前张量共享底层数组，但 requires_grad 为 False，
        不参与当前张量的计算图；对返回张量的原地修改会同步反映到当前张量上。

        注意：
        - 每次访问都会创建新的张量对象（与 PyTorch 一致），因此判断是否共享
          内存应比较底层数组指针（如 `.array().ctypes.data`），而不是对象身份；
        - 通过该属性修改数据不会触发自动微分，可能引入难以察觉的错误；
          只是读取数据时建议使用 `.detach()` 或 `.array()`。

        Returns:
            TN: 共享底层数据、requires_grad 为 False 的新张量

        Examples:
            >>> t = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
            >>> d = t.data            # 脱离计算图，与 t 共享内存
            >>> d.requires_grad       # False
            >>> d[0] = 42.0
            >>> t[0]                  # 原张量也看到修改
        """
        return self.detach()

    def detach_(self):
        """
        原地操作，断开当前张量与计算图的连接。
        
        将当前张量从计算图中分离，使其成为叶子节点，并清除所有与梯度计算相关的属性。
        这是一个就地操作，会直接修改当前张量，而不是创建新张量。
        操作后，张量将不再参与梯度计算，也无法通过反向传播接收梯度。
        
        Returns:
            None: 此方法就地修改张量，不返回任何值
        """
        self.requires_grad = False
        self.retains_grad = False
        self.fromvars = ()
        self.gradfuncs = ()
        self.parms = ()
        self.grad_value = None
        self.rcv_grad_count = 0
        return self

    def clone(self):
        """
        Returns a copy of the tensor.
        
        返回一个张量的副本，与原始张量共享相同的数据类型、形状和设备，
        但具有独立的内存。如果原始张量需要梯度，则副本也会需要梯度，
        并且会跟踪对原始张量的依赖关系。
        
        Returns:
            TN: A copy of the tensor.
        """
        ret = TN(self._array.copy(), requires_grad = (is_grad_enabled() and self.requires_grad))
        if ret.requires_grad:
            ret.fromvars=(self,)
            fn = lambda result,i: result.grad_value
            ret.gradfuncs=(fn,)
        return ret
    
    def copy(self):
        '''返回一个新张量，复制self数据，不共享内存，也不依赖self'''
        return TN(self._array.copy())

    def copy_(self,src):
        '''原地复制src到self'''
        if is_grad_enabled() and self.is_leaf and self.requires_grad:
            raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation.')
        
        # 会自动调用__setitem__，触发梯度计算
        # 用省略号...作全局索引，表示对整个张量赋值（':'不能用于0维张量，...可以）
        self[...] = src
        
        return self
    
    def copy_to(self,target:TN|np.ndarray|None=None):
        '''将self数据复制到target'''
        arrtype = (np.ndarray,cp.ndarray) if cp else (np.ndarray,)

        if isinstance(target,TN):
            if target.device != self.device:
                raise ValueError(f'target device must be {self.device} but got {target.device}')
            target.copy_(self)
        elif isinstance(target,arrtype):
            if type(target) != type(self._array):
                raise TypeError(f'target type must be {type(self._array)} but got {type(target)}')
            np.copyto(target,self._array)
        elif target is None:
            target = self.clone()
        else:
            raise TypeError('target must be a TN, numpy/cupy ndarray object or None')

        return target

    def zeros_like(self, dtype=None, device=None, requires_grad=False):
        """
        返回一个与当前张量形状相同的全零张量。
        
        参数:
            dtype: 输出张量的数据类型，默认为None（使用当前张量的数据类型）
            device: 输出张量的设备，默认为None（使用当前张量的设备）
            requires_grad: 是否需要计算梯度，默认为False
            
        返回:
            TN: 与当前张量形状相同的全零张量
        """
        return zeros_like(self, dtype=dtype, device=device, requires_grad=requires_grad)
    
    def ones_like(self, dtype=None, device=None, requires_grad=False):
        """
        返回一个与当前张量形状相同的全一张量。
        
        参数:
            dtype: 输出张量的数据类型，默认为None（使用当前张量的数据类型）
            device: 输出张量的设备，默认为None（使用当前张量的设备）
            requires_grad: 是否需要计算梯度，默认为False
            
        返回:
            TN: 与当前张量形状相同的全一张量
        """
        return ones_like(self, dtype=dtype, device=device, requires_grad=requires_grad)
    
    def empty_like(self, dtype=None, device=None, requires_grad=False):
        """
        返回一个与当前张量形状相同的未初始化张量。
        
        参数:
            dtype: 输出张量的数据类型，默认为None（使用当前张量的数据类型）
            device: 输出张量的设备，默认为None（使用当前张量的设备）
            requires_grad: 是否需要计算梯度，默认为False
            
        返回:
            TN: 与当前张量形状相同的未初始化张量
        """
        return empty_like(self, dtype=dtype, device=device, requires_grad=requires_grad)
    
    def full_like(self, fill_value, dtype=None, device=None, requires_grad=False):
        """
        返回一个与当前张量形状相同并填充了指定值的张量。
        
        参数:
            fill_value: 用于填充张量的值
            dtype: 输出张量的数据类型，默认为None（使用当前张量的数据类型）
            device: 输出张量的设备，默认为None（使用当前张量的设备）
            requires_grad: 是否需要计算梯度，默认为False
            
        返回:
            TN: 与当前张量形状相同并填充了指定值的张量
        """
        return full_like(self, fill_value, dtype=dtype, device=device, requires_grad=requires_grad)
    
    def new_full(self, size, fill_value, dtype=None, device=None, requires_grad=False):
        """
        返回一个指定形状并填充了指定值的新张量，使用当前张量的设备。
        
        参数:
            size: 输出张量的形状
            fill_value: 用于填充张量的值
            dtype: 输出张量的数据类型，默认为None（使用当前张量的数据类型）
            device: 输出张量的设备，默认为None（使用当前张量的设备）
            requires_grad: 是否需要计算梯度，默认为False
            
        返回:
            TN: 指定形状并填充了指定值的新张量
        """
        dt = self.dtype if dtype is None else dtype
        dev = self.device if device is None else _get_device(device)
        if dev.type == 'cpu':
            arr = np.full(size, fill_value, dt)
        else:
            with cp.cuda.Device(dev.index):
                arr = cp.full(size, fill_value, dt)
        
        return TN(arr, requires_grad = requires_grad)

    def zero_(self):
        if is_grad_enabled() and self.is_leaf and self.requires_grad:
            raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation.')
        
        arrlib = self._get_array_lib()
        if isinstance(self._array,arrlib.ndarray):
            self._array.fill(0.)
        else:
            raise ValueError(f'zero_ only supports numpy/cupy ndarray but got {type(self._array)}')
        return self
    
    def fill_(self, value):
        """
        将张量的所有元素填充为指定的值，原地操作。
        
        参数:
            value: 填充值，可以是标量或0D张量
        
        返回:
            self: 填充后的张量
            
        注意:
            - 这是原地操作，会直接修改张量数据
            - 如果张量是需要梯度的叶子节点，会抛出运行时错误
        """
        # 检查是否是需要梯度的叶子节点（仅在梯度启用时检查）
        if is_grad_enabled() and self.requires_grad and self.is_leaf:
            raise RuntimeError("a leaf Variable that requires grad has been used in an in-place operation")
        
        # 处理不同类型的value参数
        if isinstance(value, TN):
            if value.ndim != 0:
                raise RuntimeError(f'fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions.')
        
        # 执行原地填充操作
        self.copy_(value)
        
        # 返回自身以支持链式调用
        return self
    
    def clamp_(self, min: float | TN | None=None, max: float | TN | None=None):
        """
        将张量的每个元素限制在 [min, max] 范围内（原地操作）。
        
        参数:
            min: 最小值，可以是标量或张量。如果为None，则不限制最小值
            max: 最大值，可以是标量或张量。如果为None，则不限制最大值
        
        返回:
            self: 裁剪后的张量
            
        边界相等时的处理（对齐 PyTorch 的 clamp_）:
            - 标量边界：开区间严格比较。元素恰等于 min/max 时同样改写为边界值，其自身梯度为 0
            - 张量边界：闭区间
              * 元素仅等于一侧边界（x==lo<hi 或 lo<x==hi）：梯度对半分给该元素与该边界张量（各 0.5）
              * 元素同时等于上下界（x==lo==hi）：梯度全部归该元素自身（dx=1）
              * lo>hi（区间无效）：结果全部取 hi，梯度全部归 hi
            - min>max（均为标量时）：不报错，所有元素的结果值都等于 max（即先被抬到 min、再统一压到上界 max）

        注意:
            - 这是原地操作，会直接修改张量数据
            - 若本张量是需要梯度的叶子节点，会抛出运行时错误
            - min 与 max 必须同为标量或同为张量
        """
        # 若本张量是需要梯度的叶子节点，原地改写会破坏计算图，直接报错（与 PyTorch 一致）
        if is_grad_enabled() and self.requires_grad and self.is_leaf:
            raise RuntimeError("a leaf Variable that requires grad has been used in an in-place operation")

        # 标量与张量边界不可混用（与 PyTorch 的重载约束一致）
        min_is_tn = isinstance(min, TN)
        max_is_tn = isinstance(max, TN)
        if (min_is_tn and max is not None and not max_is_tn) or \
           (max_is_tn and min is not None and not min_is_tn):
            raise TypeError("clamp_(): min and max must be both scalars or both tensors")
        if not min_is_tn and not max_is_tn:
            # 标量边界：开区间严格比较 —— x<=min 或 x>=max 的元素改写为边界值；
            # 元素恰等于边界时也按越界处理（值取边界、自身梯度为 0）。
            # 标量可直接广播，无需掩码聚集；min>max（均为标量）不报错，
            # 所有元素的结果值都等于 max（与 torch/numpy 一致）
            if min is not None:
                self.setat_(self <= min, min)
            if max is not None:
                self.setat_(self >= max, max)
            return self

        # 张量边界：缺失一侧以 ±inf 作哨兵填充，比较时该侧条件恒为假，等价于不设限
        lo = min if min_is_tn else -float('inf')
        hi = max if max_is_tn else float('inf')

        # 原地操作不得改变形状：边界广播后的形状必须与原形状一致（与 PyTorch 一致）
        out_shape = np.broadcast_shapes(self.shape, np.shape(lo), np.shape(hi))
        if out_shape != self.shape:
            raise RuntimeError(f"output with shape {self.shape} doesn't match the broadcast shape {out_shape}")

        def gather(bound, mask):
            # setat_/addat_ 的掩码右值尺寸须等于被选中元素个数：先把边界广播到 self 形状，再按掩码聚集
            b = bound._expand(self.shape) if bound.shape != self.shape else bound
            return b[mask]

        # 相等掩码须在改写数据前、基于原始数据算出（下面的 setat_ 会改动数据）；
        # (lo < hi) 用于排除无效区间 lo>hi，并让三重相等 x==lo==hi 不落入对半分
        t_lo = (self == lo) & (lo < hi)
        t_hi = (self == hi) & (lo < hi)

        # 数据裁剪：先抬下限、再压上限。above 必须在下限改写之后计算，
        # 这样 lo>hi 时被抬到 lo 的元素也能一并压到 hi，其梯度一并转给 hi
        below = self < lo
        self.setat_(below, gather(lo, below) if min_is_tn else lo)
        above = self > hi
        self.setat_(above, gather(hi, above) if max_is_tn else hi)

        # 单边相等（x==lo<hi 或 lo<x==hi）处需把梯度对半分给元素与边界张量：
        # addat_ 使数据变为 2*bound，再 mulat_(0.5) 既还原数据、又把 self 梯度缩为一半
        #（mulat_ 是缩放而非置零），恰好得到 dx=0.5、dbound=0.5；
        # 双重/三重相等被上面的 (lo<hi) 排除，保持 dx=1
        if min_is_tn:
            self.addat_(t_lo, gather(lo, t_lo))
            self.mulat_(t_lo, 0.5)
        if max_is_tn:
            self.addat_(t_hi, gather(hi, t_hi))
            self.mulat_(t_hi, 0.5)

        # 返回自身以支持链式调用
        return self

    def masked_fill_(self, mask: TN, value: any):  # type: ignore
        """
        masked_fill_(mask, value)
        
        原地将张量中掩码为 True 的元素填充为指定值
        
        掩码 (mask) 必须是可广播到原张量形状的布尔张量
        
        参数:
            mask (TN): 布尔类型的张量掩码，形状需与原张量可广播
            value (any): 用于填充的值，可以是标量或 0 维张量
        
        返回:
            TN: 原地修改后的张量
        
        注意:
            - 这是原地操作，会直接修改张量数据
            - 如果张量是需要梯度的叶子节点，会抛出运行时错误
            - 掩码必须是布尔类型的张量
            - 填充值可以是标量或 0 维张量
            - 掩码形状需与原张量形状可广播
            - 此实现与 PyTorch 的 masked_fill_ 行为保持一致
        
        示例:
            基本用法:
                arr = tensor([[1, 2, 3], [4, 5, 6]])
                mask = tensor([[True, False, True], [False, True, False]])
                arr.masked_fill_(mask, 0)
                print(arr)  # 输出: [[0, 2, 0], [4, 0, 6]]
            
            使用可广播掩码:
                arr = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
                mask = tensor([True, False, True])  # 形状为 (3,)，可广播到 (3, 3)
                arr.masked_fill_(mask, 10)
                print(arr)  # 输出: [[10, 2, 10], [10, 5, 10], [10, 8, 10]]
            
            带梯度计算的用法（跟踪右值梯度）:
                arr = tensor([[1, 2], [3, 4]], requires_grad=True)
                arr_clone = arr.clone()  # 创建非叶子节点
                mask = tensor([[True, False], [False, True]])
                fill_value = tensor(100.0, requires_grad=True)  # 设置填充值为可跟踪梯度
                arr_clone.masked_fill_(mask, fill_value)
                loss = arr_clone.sum()
                loss.backward()
                print(fill_value.grad)  # 输出填充值的梯度，显示被填充位置的数量
            
            使用标量值填充:
                arr = tensor([1, 2, 3, 4, 5])
                mask = tensor([False, True, False, True, False])
                arr.masked_fill_(mask, -1)
                print(arr)  # 输出: [1, -1, 3, -1, 5]
        """
        # 检查是否是需要梯度的叶子节点，原地操作不允许对需要梯度的叶子节点执行
        if is_grad_enabled() and self.requires_grad and self.is_leaf:
            raise RuntimeError("a leaf Variable that requires grad has been used in an in-place operation")
        
        # 检查mask是否为布尔类型
        if mask.dtype != np.bool_:
            raise RuntimeError(f"masked_fill_ only supports boolean masks, but got mask with dtype {mask.dtype}")
        
        # 处理不同类型的value参数
        if isinstance(value, TN):
            if value.ndim != 0:  # type: ignore
                raise RuntimeError(f'masked_fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions.')  # type: ignore
        
        # 执行原地填充操作
        try:
            if mask.shape != self.shape:
                # 如果需要广播，先广播掩码
                broadcasted_mask = mask.broadcast_to(self.shape)
                self[broadcasted_mask] = value
            else:
                self[mask] = value
        except Exception as e:
            raise RuntimeError(f"mask with shape {mask.shape} is not broadcastable to tensor shape {self.shape}") from e
        
        # 返回自身以支持链式调用
        return self
    
    def masked_fill(self, mask: TN, value: Any) -> TN:
        """
        使用标量值填充张量中对应掩码为True的位置（非原地操作）
        
        参数:
            mask: 布尔类型的掩码张量，形状需与原张量广播兼容
            value: 填充值，可以是标量或0维张量
        
        返回:
            填充后的新张量
        
        示例:
            使用标量值填充:
                arr = tensor([1, 2, 3, 4, 5])
                mask = tensor([False, True, False, True, False])
                new_arr = arr.masked_fill(mask, -1)
                print(new_arr)  # 输出: [1, -1, 3, -1, 5]
                print(arr)      # 输出: [1, 2, 3, 4, 5] (原张量不变)
        """
        return self._run_registered_api('masked_fill', self, mask, value)
    
    def squeeze(self, dim: int|tuple|None = None):
        """
        移除张量中大小为1的维度（降维），原张量不变；
        若最终没有任何维度被移除，则原样返回（同一对象）。

        参数:
            dim: 指定要移除的维度，各取值行为如下:
                - None（默认）: 移除所有大小为1的维度。
                  例如 (2,1,3)→(2,3)，(1,1)→()，(2,3)→(2,3)。
                - (): 空元组，无操作，直接返回原张量本身。
                - int: 仅尝试移除该维度。支持负数索引（从末尾计数，
                  有效范围 [-ndim, ndim-1]），越界抛出 IndexError；
                  该维度大小不为1时不做任何操作，原样返回（对齐 PyTorch）。
                  例如形状 (2,1,3): squeeze(1)→(2,3)，
                  squeeze(0)→(2,1,3)，squeeze(-1)→(2,1,3)。
                - tuple: 对每个维度独立应用 int 的规则后一次移除:
                  越界抛出 IndexError；大小不为1的维度静默跳过；
                  重复指定大小为1的维度抛出 RuntimeError。
                  所有维度大小均为1时结果为0维标量张量。
                  例如形状 (2,1,4,1): squeeze((1,3))→(2,4)。

        返回:
            TN: 移除指定大小为1维度后的张量

        异常:
            IndexError: 当指定的维度索引越界时抛出
            RuntimeError: 当元组中重复指定大小为1的维度时抛出
            TypeError: 当 dim 不是 None、int 或 tuple 时抛出
        """
        # dim为None时，收集所有大小为1的维度索引，直接交给_squeeze处理
        # 这些维度天然合法（非负、无重复、大小均为1），无需再走校验流程
        if dim is None:
            new_dim = tuple(i for i, size in enumerate(self._array.shape) if size == 1)
            return self._squeeze(new_dim)

        # 空元组为无操作快速路径，直接返回原张量
        if dim == ():
            return self

        # 单int快速路径：避免构建列表和循环，减少常见调用的开销
        if isinstance(dim, int):
            # 负数维度归一化为非负索引
            actual_dim = dim + self._array.ndim if dim < 0 else dim
            if actual_dim < 0 or actual_dim >= self._array.ndim:
                raise IndexError(f'Dimension out of range (expected to be in range of [{-self._array.ndim}, {self._array.ndim - 1}], but got {dim})')
            # 指定维度大小不为1时按约定不执行操作，直接返回原张量（对齐PyTorch行为）
            if self._array.shape[actual_dim] != 1:
                return self
            return self._squeeze((actual_dim,))

        # 元组路径：逐维度校验并收集实际要挤压的非负维度索引
        if isinstance(dim, tuple):
            new_dim = []
            for d in dim:
                actual_dim = d + self._array.ndim if d < 0 else d
                if actual_dim < 0 or actual_dim >= self._array.ndim:
                    raise IndexError(f'Dimension out of range (expected to be in range of [{-self._array.ndim}, {self._array.ndim - 1}], but got {d})')
                # 大小不为1的维度静默跳过，不影响其他维度的挤压
                if self._array.shape[actual_dim] != 1:
                    continue
                if actual_dim in new_dim:
                    raise RuntimeError(f'dim {actual_dim} appears multiple times in the list of dims')
                new_dim.append(actual_dim)
            return self._squeeze(tuple(new_dim))

        raise TypeError("dim must be None, an integer, or a tuple of integers")

    def _squeeze(self, new_dim: tuple):
        # 快速路径：如果没有需要挤压的维度，直接返回原张量
        if not new_dim:
            return self

        # 根据原始张量的数据类型选择使用np或cp
        arrlib = self._get_array_lib()

        # 执行挤压操作并设置梯度信息
        newarr = arrlib.squeeze(self._array, axis=new_dim)
        ret = TN(newarr, requires_grad = (is_grad_enabled() and self.requires_grad))
        
        if ret.requires_grad:
            ret.fromvars = (self,)
            ret.parms = (new_dim,)
            ret.gradfuncs = (_squeeze_backward,)
        return ret

    def unsqueeze(self, dim: int|tuple|None = None):
        """
        在指定位置插入大小为1的新维度（升维），原张量不变。
        输出维度数 = 原维度数 + 插入的维度个数，元素总数不变。

        参数:
            dim: 指定插入位置，各取值行为如下:
                - None 或 (): 无操作，直接返回原张量本身
                  （框架扩展语义，供 sum/mean 等反向传播复用）。
                - int: 在该位置插入一个大小为1的维度。支持负数索引
                  （基于插入后的维度数从末尾计数，有效范围
                  [-(ndim+1), ndim]，ndim 为原维度数），越界抛出 IndexError。
                  例如形状 (2,3): unsqueeze(0)→(1,2,3)，
                  unsqueeze(1)→(2,1,3)，unsqueeze(-1)→(2,3,1)。
                - tuple: 同时在多个位置插入大小为1的维度。各位置基于
                  插入后的维度数独立归一化负数索引，重复位置抛出
                  ValueError。例如形状 (2,3): unsqueeze((0,2))→(1,2,1,3)。

        返回:
            TN: 插入大小为1维度后的张量

        异常:
            IndexError: 当指定的维度索引越界时抛出
            ValueError: 当元组中存在重复的插入位置时抛出
            TypeError: 当 dim 不是 int 或 tuple 时抛出
        """
        # 无操作快速路径：dim为None或空元组时直接返回原张量
        if dim is None or dim == ():
            return self

        # 性能说明：底层统一用 ndarray.reshape 而非 arrlib.expand_dims 实现维度插入。
        # expand_dims 是 Python 层封装（约1.6µs），直接 reshape 仅约0.16µs，
        # 对 unsqueeze 这类高频调用（sum/mean/matmul 等反向均依赖）收益显著。

        # 单int快速路径：最常见的调用形式，单独处理以减少循环和集合操作开销
        if isinstance(dim, int):
            # 输出张量的维度数 = 原维度数 + 1，负数维度据此归一化为非负索引
            ndim = self._array.ndim + 1
            actual = dim + ndim if dim < 0 else dim
            if actual < 0 or actual >= ndim:
                raise IndexError(f'dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {dim})')
            # 在目标位置插入大小为1的维度后reshape，等价于expand_dims
            shape = list(self._array.shape)
            shape.insert(actual, 1)
            newarr = self._array.reshape(shape)
            ret = TN(newarr, requires_grad = (is_grad_enabled() and self.requires_grad))
            if ret.requires_grad:
                ret.fromvars = (self,)
                # parms 统一存元组形式，保证反向函数拿到的参数类型一致
                ret.parms = ((dim,),)
                ret.gradfuncs = (_unsqueeze_backward,)
            return ret

        # 元组路径：支持一次插入多个维度
        if isinstance(dim, tuple):
            # 1元组快速路径：直接调用int路径，避免元组分支的全部循环和查重开销
            # 实测在 shape=(64,1,32,1,8) 上可将 1元组 从 1.28µs 降至 1.00µs
            if len(dim) == 1:
                return self.unsqueeze(dim[0])

            # 输出张量的维度数 = 原维度数 + 插入维度个数，负数维度据此归一化
            ndim = self._array.ndim + len(dim)
            norm = []
            for d in dim:
                actual = d + ndim if d < 0 else d
                if actual < 0 or actual >= ndim:
                    raise IndexError(f'dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {d})')
                norm.append(actual)

            # 归一化后排序，再相邻元素查重：省去一次 set 创建和独立的全量扫描
            # 实测在 2~3 维度元组上比 set 方案快约 10~11%
            norm.sort()
            for i in range(1, len(norm)):
                if norm[i] == norm[i - 1]:
                    raise ValueError(f'repeated axis {norm[i]}')

            # 按升序逐个插入大小为1的维度，reshape等价于expand_dims
            shape = list(self._array.shape)
            for d in norm:
                shape.insert(d, 1)
            newarr = self._array.reshape(shape)
            ret = TN(newarr, requires_grad = (is_grad_enabled() and self.requires_grad))
            if ret.requires_grad:
                ret.fromvars = (self,)
                # 元组形式原样存入parms，反向函数直接复用
                ret.parms = (dim,)
                ret.gradfuncs = (_unsqueeze_backward,)
            return ret

        raise TypeError("dim must be an integer or a tuple of integers")

    def reshape(self, *new_shape):
        """
        返回具有新形状的张量，不改变其数据。
        
        支持负数维度大小（特别是-1）
        来自动计算该维度的大小，前提是其他维度的大小可以确定。
        支持类似NumPy的负数维度语义（从末尾计数）。
        
        参数:
            *new_shape: 整数序列或单个形状元组，指定新的张量形状。
                       可以包含-1，表示自动计算该维度的大小。
                       最多只能有一个维度可以指定为-1。
                       新形状的元素总数必须与原始形状兼容。
        
        返回:
            TN: 具有新形状的张量
            
        异常:
            RuntimeError: 当新形状与原始形状的元素总数不兼容时抛出
            RuntimeError: 当有多个维度被指定为-1时抛出
            RuntimeError: 当形状包含非整数值时抛出
            RuntimeError: 当形状包含无效的维度值时抛出
        """
        # 处理参数格式：支持 reshape(2,3) 与 reshape((2,3)) / reshape([2,3])
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            shape = new_shape[0]
        else:
            shape = new_shape

        # 列表先归一化为元组，否则下方 self.shape(元组) 与列表比较恒为False，快速路径会被跳过
        if isinstance(shape, list):
            shape = tuple(shape)

        # shape相同直接返回self（最常见的无操作调用，放在校验前以取得最低开销）。
        # 用 all(type(d) is int) 确保数值相同但类型为 float/np.int64 的无效输入
        # 不会静默通过，仍落入下方单趟校验流程抛出异常。
        if self.shape == shape and all(type(d) is int for d in shape):
            return self

        # 单趟扫描完成全部校验与预处理：整数类型检查、-1推断维度定位、
        # 其他负数维度归一化（按新维度数从末尾计数）、已知维度累乘。
        # 相比旧实现的多趟扫描（int校验/count(-1)/负数转换/负值复查/两次math.prod），
        # 此处仅需一趟，实测性能提升约33%~44%。
        new_ndim = len(shape)
        processed_shape = []
        neg_one_idx = -1        # -1所在的维度索引，-1表示未出现
        known_prod = 1          # 已知维度（不含-1）的乘积
        for i, d in enumerate(shape):
            if not isinstance(d, int):
                raise RuntimeError(f'{shape} contains non int numbers')
            if d == -1:
                if neg_one_idx >= 0:
                    raise RuntimeError(f"only one dimension can be inferred (got {shape.count(-1)})")
                neg_one_idx = i
                processed_shape.append(-1)
            elif d < 0:
                # 其他负数维度：转换为正数维度索引（new_ndim + d）
                p = new_ndim + d
                if p < 0:
                    raise RuntimeError(f"Invalid shape dimension {d}")
                processed_shape.append(p)
                known_prod *= p
            else:
                processed_shape.append(d)
                known_prod *= d

        # 计算原始张量的元素总数
        original_size = self._array.size

        if neg_one_idx >= 0:
            # 有-1待推断：整除检查后，推断结果天然使总元素数一致，无需再验
            if original_size % known_prod != 0:
                raise RuntimeError(
                    f"shape '{processed_shape}' is invalid for input of size {original_size}"
                )
            processed_shape[neg_one_idx] = original_size // known_prod
        elif known_prod != original_size:
            # 无-1：单趟累乘的known_prod即为新形状元素总数，直接比较，省去第二次math.prod
            raise RuntimeError(
                f"shape '{processed_shape}' is invalid for input of size {original_size}"
            )

        return self._reshape(tuple(processed_shape))
        
    def view(self, *new_shape):
        """返回具有相同数据但不同形状的新张量视图。
        
        该函数是reshape函数的别名，直接调用reshape来实现相同的功能。
        新形状必须与原张量的元素总数兼容。
        
        参数:
            *new_shape: 整数序列，指定新的张量形状
        
        返回:
            TN: 具有新形状的张量
        
        异常:
            RuntimeError: 当形状参数包含非整数或形状不兼容时抛出
        """
        return self.reshape(*new_shape)

    def _reshape(self, new_shape):
        """
        Simplified version of reshape with better performance.
        
        Optimizations:
        - Removed integer type checking
        - Removed negative dimension handling (including -1 for auto-inference)
        - Direct numpy reshape call with minimal validation
        - Only accepts non-negative integer tuples or sequences
        
        Args:
            new_shape: a single shape tuple, 
                       specifying the new tensor shape.
                       All dimensions must be non-negative integers.
                       The product of dimensions must match the original tensor size.
        
        Returns:
            TN: Tensor with the new shape
        
        Note:
            This function assumes valid input parameters for better performance.
            For robust error handling with -1 support, use reshape() method instead.
        """
        # 缓存shape，避免二次属性访问；与原形状相同则直接返回self
        original_shape = self.shape
        if original_shape == new_shape:
            return self

        ret = TN(self._array.reshape(new_shape), 
                 requires_grad = (is_grad_enabled() and self.requires_grad)
        )
        
        if ret.requires_grad:
            ret.fromvars = (self,)
            ret.parms = ((original_shape, new_shape),)
            ret.gradfuncs = (_reshape_backward,)
        return ret

    def unfold(self, dimension: int, size: int, step: int = None) -> 'TN':
        return api_registry.get("unfold")(self, dimension, size, step)
    
    def fold(self, dimension: int, size: int, step: int = None, output_size: int = None) -> 'TN':
        return api_registry.get("fold")(self, dimension, size, step, output_size)
    
    def flatten(self, start_dim=0, end_dim=-1) -> 'TN':
        """将张量从start_dim到end_dim的维度展平，自动支持梯度跟踪"""
        # 处理负数维度索引
        dim = self._array.ndim
        start_dim = start_dim if start_dim >= 0 else dim + start_dim
        end_dim = end_dim if end_dim >= 0 else dim + end_dim
        assert 0 <= start_dim <= end_dim < dim, "Invalid dimension range"

        # 计算展平后的新形状
        new_shape = list(self.shape)
        # 合并目标维度
        flattened_size = 1
        for i in range(start_dim, end_dim + 1):
            flattened_size *= new_shape[i]
        new_shape = new_shape[:start_dim] + [flattened_size] + new_shape[end_dim + 1:]

        # 通过reshape实现展平（继承梯度传播能力）
        return self._reshape(tuple(new_shape))

    def expand(self, *size):
        """
        将张量扩展为新的形状，仅能扩展大小为1的维度。
        
        参数:
            *size: 目标形状，可以是一个整数元组或多个整数参数
        
        返回:
            TN: 扩展后的张量
        
        异常:
            RuntimeError: 当扩展不兼容时抛出
        """
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            new_shape = size[0]
        else:
            new_shape = size
        
        # 列表先归一化为元组，否则下方与 self.shape（元组）比较恒为 False
        if isinstance(new_shape, list):
            new_shape = tuple(new_shape)
        
        original_shape = self.shape
        
        # 快速路径：如果形状相同，直接返回自身，前移到逐元素 int 转换之前
        if new_shape == original_shape:
            return self
        
        # 仅在未命中快速路径时才做逐元素 int 转换
        new_shape = tuple(int(s) for s in new_shape)
        
        orig_ndim = len(original_shape)
        new_ndim = len(new_shape)
        
        # 处理-1并统一维度对齐后的原始形状
        if orig_ndim == 0:
            # 标量可以扩展到任意形状，直接使用新形状
            aligned_orig_shape = new_shape
        elif new_ndim < orig_ndim:
            raise RuntimeError(
                f"expand(Size({list(new_shape)})): the number of dimensions "
                f"({new_ndim}) must be >= the number of dimensions of the tensor ({orig_ndim})"
            )
        elif new_ndim > orig_ndim:
            # 在前面添加1来对齐维度
            aligned_orig_shape = (1,) * (new_ndim - orig_ndim) + original_shape
        else:
            aligned_orig_shape = original_shape
        
        # 处理-1并检查维度兼容性
        processed_shape = []
        for i, (orig_dim, new_dim) in enumerate(zip(aligned_orig_shape, new_shape)):
            if new_dim == -1:
                if orig_ndim == 0:
                    raise RuntimeError("Cannot use -1 when expanding a scalar tensor")
                processed_shape.append(orig_dim)
            else:
                if orig_dim != 1 and orig_dim != new_dim:
                    raise RuntimeError(
                        f"The expanded size of the tensor ({new_dim}) in dimension {i} must be "
                        f"equal to the existing size ({orig_dim}) or the existing size must be one."
                    )
                processed_shape.append(new_dim)
        
        new_shape = tuple(processed_shape)

        return self._expand(new_shape)

    def _expand(self, new_shape):
        """
        实现张量扩展的内部逻辑，不对new_shape做检查
        该函数被sum、mean的梯度函数调用，以提高效率
        """
        original_shape = self.shape
        
        # 快速路径：如果形状相同，直接返回自身
        if new_shape == original_shape:
            return self
        
        # 根据原始张量的数据类型选择使用np或cp
        arrlib = self._get_array_lib()        
        # 使用numpy的broadcast_to执行扩展
        expanded_data = arrlib.broadcast_to(self._array, new_shape)   
        
        # 创建新的张量对象
        ret = TN(expanded_data, requires_grad = (is_grad_enabled() and self.requires_grad))
        
        # 设置梯度跟踪信息
        if ret.requires_grad:
            ret.fromvars = (self,)
            ret.parms = ((original_shape, new_shape),)
            ret.gradfuncs = (_expand_backward,)
    
        return ret

    def expand_as(self, other: TN) -> TN:
        """
        将当前张量扩展为与其他张量相同的形状。
        
        参数:
            other: 目标张量，用于确定扩展后的形状
        
        返回:
            TN: 扩展后的张量
            
        异常:
            RuntimeError: 当当前张量的维度大于other的维度时抛出
        """
        return self.expand(*other.shape)

    def broadcast_to(self, *size) -> TN:
        """
        将当前张量广播到指定形状。
        
        参数:
            *size: 目标形状，可以是一个整数元组或多个整数参数
        
        返回:
            TN: 扩展后的张量
        
        异常:
            RuntimeError: 当扩展不兼容时抛出
        """
        # 解析参数格式（支持 (2,3) 或 2,3 两种形式）
        new_shape = _validate_shape(size)

        # 直接调用函数版本，避免重复验证
        return broadcast_to(self, new_shape)
    
    def repeat(self, *repeats) -> TN:
        """
        沿着指定的维度重复张量的元素。
        
        参数:
            *repeats: 每个维度的重复次数，可以是一个整数元组或多个整数参数
        
        返回:
            TN: 重复后的新张量
        
        异常:
            ValueError: 当repeats长度不等于张量维度时抛出
        """
        repeats = _validate_shape(repeats)
        return self._run_registered_api('repeat', self, repeats)
    
    def flip(self, dims:list[int]|tuple[int,...]) -> TN:
        return flip(self,dims)

    def clamp(self,min:float|None=None, max:float|None=None):  # type: ignore
        return self._run_registered_api('clamp', self, min, max)
    
    def where(self,cond,other):
        return where(cond,self,other)

    def all(self, dim:int|tuple|None=None, keepdim:bool=False):  # type: ignore
        """
        测试张量中的所有元素是否为True。
        
        参数:
            x: 输入张量
            dim: 沿着哪个轴执行all操作。如果为None，则对所有元素进行操作
            keepdim: 是否保持结果维度不变
        
        返回:
            包含逻辑运算结果的张量
        """
        if dim == ():
            dim = None
        
        # 根据原始张量的数据类型选择使用np或cp
        arrlib = self._get_array_lib()
        # 使用numpy/cupy的all函数计算结果
        result_data = arrlib.all(self._array, axis=dim, keepdims=keepdim)
        
        # 创建结果张量，保持梯度传播
        return TN(result_data)

    def any(self, dim:int|tuple|None=None, keepdim:bool=False):  # type: ignore
        """
        测试张量中的任何元素是否为True。
        
        参数:
            x: 输入张量
            dim: 沿着哪个轴执行any操作。如果为None，则对所有元素进行操作
            keepdim: 是否保持结果维度不变
        
        返回:
            包含逻辑运算结果的张量
        """
        if dim == ():
            dim = None
        
        # 根据原始张量的数据类型选择使用np或cp
        arrlib = self._get_array_lib()
        # 使用numpy/cupy的any函数计算结果
        result_data = arrlib.any(self._array, axis=dim, keepdims=keepdim)
        
        return TN(result_data)

    def equal(self, other):
        return equal(self,other)
    
    def not_equal(self, other):
        return not_equal(self,other)

    def allclose(self:TN,other:TN,rtol:float=1e-5, atol:float=1e-8, equal_nan:bool=False)->bool:  # type: ignore
        return allclose(self,other,rtol,atol,equal_nan)

    def max(self, dim:int|None=None, keepdim:bool=False):  # type: ignore
        return max(self,dim,keepdim)
    
    def min(self, dim:int|None=None, keepdim:bool=False):  # type: ignore
        return min(self,dim,keepdim)

    def argmax(self,dim:int|None=None,keepdim:bool=False):  # type: ignore
        idx = self._array.argmax(axis=dim,keepdims=keepdim)
        ret = tensor(idx, dtype = int64, device=self.device)        
        return ret
    
    def argmin(self,dim:int|None=None,keepdim:bool=False):  # type: ignore
        idx = self._array.argmin(axis=dim,keepdims=keepdim)
        ret = tensor(idx, dtype = int64, device=self.device)
        return ret

    def argsort(self,dim:int=-1,descending:bool=False):  # type: ignore
        return argsort(self,dim=dim,descending=descending)

    def sort(self,dim:int=-1,descending:bool=False):  # type: ignore
        return self._run_registered_api('sort', self, dim=dim, descending=descending)

    def topk(self, k:int, dim:int=-1, largest:bool=True, sorted:bool=True):  # type: ignore
        return self._run_registered_api('topk', self, k, dim=dim, largest=largest, sorted=sorted)

    def multinomial(self, num_samples:int, replacement:bool=False, *, generator=None):  # type: ignore
        return self._run_registered_api('multinomial', self, num_samples, replacement, generator=generator)

    def maximum(self, other):
        return maximum(self,other)
    
    def minimum(self, other):
        return minimum(self,other)

    def sum(self, dim:int|tuple|None=None, keepdim:bool=False):  # type: ignore
        ret = sum(self, dim, keepdim)
        return ret
    
    def cumsum(self, dim: int, *, dtype: str | np.dtype | None = None, out: TN | None = None):
        ret = cumsum(self,dim,dtype=dtype,out=out)
        return ret
    
    def prod(self, dim:int|tuple|None=None, keepdim:bool=False):  # type: ignore
        return self._run_registered_api('prod', self, dim, keepdim)
    
    def mean(self, dim:int|tuple|None=None, keepdim:bool=False):  # type: ignore
        ret = mean(self,dim,keepdim)
        return ret
    
    def var(self, dim:int|tuple|None=None, unbiased:bool=True, keepdim:bool=False):  # type: ignore
        ret = var(self, dim, unbiased, keepdim)
        return ret
    
    def std(self, dim:int|tuple|None=None, unbiased:bool=True, keepdim:bool=False):  # type: ignore
        ret = std(self, dim, unbiased, keepdim)
        return ret

    def abs(self):
        return abs(self)
    
    def pow(self, exponent)->TN:
        return pow(self,exponent)
    
    def sqrt(self):
        return sqrt(self)

    def exp(self):
        return exp(self)
    
    def log(self):
        return log(self)

    def log10(self):
        return log10(self)
    
    def log2(self):
        return log2(self)
    
    def log1p(self):
        return log1p(self)

    def sin(self):
        return self._run_registered_api('sin', self)

    def cos(self):
        return self._run_registered_api('cos', self)

    def tan(self):
        return self._run_registered_api('tan', self)

    def cot(self):
        return self._run_registered_api('cot', self)

    def sec(self):
        return self._run_registered_api('sec', self)

    def csc(self):
        return self._run_registered_api('csc', self)

    def arcsin(self):
        return self._run_registered_api('arcsin', self)

    def arccos(self):
        return self._run_registered_api('arccos', self)

    def arctan(self):
        return self._run_registered_api('arctan', self)

    def sinh(self):
        return self._run_registered_api('sinh', self)

    def cosh(self):
        return self._run_registered_api('cosh', self)

    def tanh(self):
        return self._run_registered_api('tanh', self)

    def coth(self):
        return self._run_registered_api('coth', self)

    def sech(self):
        return self._run_registered_api('sech', self)

    def csch(self):
        return self._run_registered_api('csch', self)

    def arcsinh(self):
        return self._run_registered_api('arcsinh', self)

    def arccosh(self):
        return self._run_registered_api('arccosh', self)

    def arctanh(self):
        return self._run_registered_api('arctanh', self)

    def sign(self):
        return sign(self)

    def diagonal(self,offset:int=0,dim1:int=0,dim2:int=1):
        return diagonal(self,offset,dim1,dim2)
    
    def diag(self,offset: int = 0):
        return diag(self,offset)

    def batch_diag(self):
        return batch_diag(self)

    def fill_diagonal(self, value, offset: int = 0, dim1: int = -2, dim2: int = -1):
        return fill_diagonal(self,value,offset,dim1,dim2)

    def fill_diagonal_(self, value, offset: int = 0, dim1: int = -2, dim2: int = -1):
        return fill_diagonal_(self,value,offset,dim1,dim2)

    def tril(self, diagonal: int = 0):
        ret = tril(self, diagonal)
        return ret
    
    def triu(self, diagonal: int = 0):
        ret = triu(self, diagonal)
        return ret

    def inverse(self):
        return self._run_registered_api('inv', self)

    def pinverse(self):
        return self._run_registered_api('pinv', self)

    def det(self):
        return self._run_registered_api('det', self)

    def norm(self, ord:int|float|str|None = "fro", dim: int|tuple|None = None, keepdim: bool = False):  # type: ignore
        return self._run_registered_api('norm', self, ord=ord, dim=dim, keepdim=keepdim)

    def _init_calc_graph(self) -> bool:
        '''初始化以self为根的计算图中的缓存信息，包括：grad_value、rcv_grad_count

           该函数被backward和grad函数调用，用于：
           1、将从self开始的反向计算图中所有节点的梯度缓存grad_value置None
           2、初始化各节点应收到反向传播梯度的次数rcv_grad_count
           3、检查计算图中是否存在需要钩子处理的模块

           实现细节：
           - 使用visited set跟踪已访问节点，确保即使计算图缓存未清理也能正确初始化
           - 使用rcv_grad_count == 0判断节点是否首次访问，避免重复入栈
           - 钩子检查使用短路逻辑，一旦确定需要钩子处理则跳过后续检查

           返回:
               bool: 是否需要钩子处理（True表示需要）
        '''
        stack = [self]  # 初始化一个栈
        visited = {self}  # 使用set跟踪已访问节点，确保彻底初始化
        self.rcv_grad_count = 0
        need_hook_processing = False

        while stack:
            item: TN = stack.pop(-1)  # pop(-1)效率o(1),表示从list尾弹出
            item.grad_value = None    # 梯度值置None

            # 检查是否需要钩子处理（只在尚未确定需要钩子时检查）
            # 短路逻辑：一旦need_hook_processing为True，跳过所有后续检查
            if not need_hook_processing and item._output_of is not None:
                module = item._output_of
                if module._backward_pre_hooks or module._backward_hooks:
                    need_hook_processing = True

            # 叶子节点不要再往下遍历
            if item.is_leaf:
                continue

            varlist = item.fromvars
            for var in varlist:
                if var.requires_grad:
                    # 使用visited set判断首次访问，确保rcv_grad_count被正确初始化
                    if var not in visited:
                        var.rcv_grad_count = 0  # 显式初始化
                        stack.append(var)
                        visited.add(var)
                    var.rcv_grad_count += 1  # 累计当前节点可接受梯度的次数

        return need_hook_processing

    def _clear_calc_graph_cache(self):
        '''清理以self为根的计算图中的临时梯度状态

        将所有节点的 rcv_grad_count 置 0、grad_value 置 None。
        与 _init_calc_graph 的区别：
          - _init_calc_graph 会重新累加 rcv_grad_count（从子节点数推导正确值）
          - _clear_calc_graph_cache 仅将 rcv_grad_count 置 0，不做重新累加
        因此前者可独立完成初始化，后者仅作为辅助清理工具保留。
        '''

        stack = [self]

        while stack:
            item:TN = stack.pop(-1)
            item.rcv_grad_count = 0
            item.grad_value = None  # type: ignore

            if item.is_leaf:
                continue

            for var in item.fromvars:
                if var.requires_grad:
                    stack.append(var)
        
        return

    def _save_grad(self:TN):
        '''将self的grad_value保存到self的grad中，返回是否是叶子节点
        
        1._backward_with_hooks或_backward_without_hooks里调用_save_grad，
          self是出栈节点，已保证了self是一个requires_grad=True的叶子节点或中间节点。
        2.叶子节点（is_leaf=True）总是保存梯度到self.grad，
          中间节点只有在retains_grad=True时才保存梯度。
        
        '''
        
        is_leaf = not self.fromvars
        
        # 叶子节点：总是保存梯度
        # 注意：is_leaf为True意味着requires_grad=True且没有fromvars
        if is_leaf:
            if self.grad is None:
                # 如还没有梯度值，直接赋值
                self.grad = self.grad_value
            else:
                # 如已有梯度值，累计梯度，注意不要使用原地+=
                self.grad = self.grad + self.grad_value
            
            # 叶子节点存完梯度后，梯度缓存置None
            self.grad_value = None
        
        # 中间节点：只有retains_grad=True时才保存梯度
        elif self.retains_grad:
            if self.grad is None:
                self.grad = self.grad_value
            else:
                self.grad = self.grad + self.grad_value
            
            # 中间节点保存梯度后，不清空grad_value，因为还要传播
        
        return is_leaf

    def _add_received_grad_value(self:TN, grad_value:TN):
        '''将grad_value添加到self的grad_value中
        '''

        # 如果收到的梯度为None，直接返回
        if grad_value is None:
            return
        
        # 确保self和target的dtype一致
        # grad_value是待累加的梯度，由于计算过程中精度会出现提升现象(float32 to float64)
        # 及时精度转换虽然可能会导致梯度张量被复制到一个新类型，但由于保证了梯度一直是低级度，
        # 计算性能反而会提升
        if grad_value.dtype != self.dtype:
            warnings.warn(f'self.dtype={self.dtype}, received grad_value.dtype={grad_value.dtype}')
            grad_value = grad_value.type(self.dtype)
        
        if self.grad_value is None:
            # 如还没有梯度值，直接赋值
            self.grad_value = grad_value
        else:
            # 如self已有梯度值，累计梯度
            # 多个节点可能共享了同一个grad_value张量，
            # 比如c=a+b，c收到的梯度grad_value会原封不动地传递给a和b,
            # a和b的grad_value是同一个张量，所以不能用原地+=累加梯度
            self.grad_value = self.grad_value + grad_value
        return

    def _propagate_grad_to_sources(self: TN, vars_received_grad: set) -> None:
        """将当前节点的梯度传播到其所有来源节点

        反向传播核心方法，遍历当前节点的所有来源节点，计算并传播梯度。
        直接修改vars_received_grad集合，添加已收集完梯度的节点。

        参数：
            vars_received_grad: 已收集完梯度、可入栈的源变量集合（会被直接修改）
        """
        fromvars = self.fromvars
        gradfuncs = self.gradfuncs

        for i in range(len(fromvars)):
            var: TN = fromvars[i]
            fn = gradfuncs[i]

            # 只有requires_grad=True的节点才需要梯度传播
            if var.requires_grad:
                rcv_grad_value: TN = fn(self, i)
                var._add_received_grad_value(rcv_grad_value)
                var.rcv_grad_count -= 1

                # 如var的rcv_grad_count为0，说明所有梯度都已收集，可入栈传播
                if var.rcv_grad_count == 0:
                    vars_received_grad.add(var)

        return

    def _backward_with_hooks(self) -> None:
        """带钩子处理的反向传播核心逻辑

        在反向传播过程中统一处理反向预处理钩子和反向钩子：
        1. 反向预处理钩子：在模块所有输出梯度就绪时调用，可修改输出梯度
        2. 反向钩子：在模块所有输入梯度就绪时调用，可修改输入梯度
        """
        hook_manager = BackwardHookManager()
        stack = [self]

        while stack:
            item: TN = stack.pop(-1)

            # 以下代码调测时用于检查栈内节点是否都是已收到梯度的节点或梯度没有被异常置None
            # if item.grad_value is None:
            #     warnings.warn(f'item poped from stack has no grad_value, skip backward')
            #     continue

            # 在反向传播前统一检查和处理两种钩子
            items_for_propagate = hook_manager.process_module_hooks(item)

            # 在调用钩子之后保存梯度，确保钩子可以访问所有输入的梯度grad_value
            item._save_grad()

            # 如果没有可传播节点（模块输入或输出梯度未收集齐），跳过本次循环
            if not items_for_propagate:
                continue

            # 处理需要钩子的节点
            vars_received_grad = set()
            for current_item in items_for_propagate:
                current_item._propagate_grad_to_sources(vars_received_grad)
                # 传播完梯度的节点，使命完成，清空grad_value节省内存
                current_item.grad_value = None
            stack.extend(vars_received_grad)

        # 清理钩子缓存
        hook_manager.cleanup()

    def _backward_without_hooks(self) -> None:
        """无钩子处理的反向传播核心逻辑（快速路径）

        标准的反向传播实现，无钩子处理开销。
        """
        stack = [self]

        while stack:
            item: TN = stack.pop(-1)
            
            # 保存叶子节点或retains_grad=True的中间节点的梯度，
            # 如item是叶子节点，无须传播梯度，快速进入下一次循环
            if item._save_grad():
                continue
            
            # 传播梯度到来源节点
            vars_received_grad = set()
            item._propagate_grad_to_sources(vars_received_grad)
            item.grad_value = None

            # 将收到梯度的节点入栈
            stack.extend(vars_received_grad)

    def backward(self, gradient: TN|None = None,
                 retain_graph: bool = False,
                 create_graph: bool = False):
        """执行自动微分的反向传播计算。

        该函数从当前张量开始，沿计算图反向传播梯度，计算并存储所有
        叶子节点或设置了retains_grad=True的中间节点的梯度。

        实现结构：
        1. 参数验证和梯度准备
           - 验证当前张量需要梯度
           - 准备初始梯度值（标量隐式为1.0，或用户提供的外部梯度）
        2. 初始化计算图
           - 遍历计算图初始化所有节点的rcv_grad_count和grad_value
           - 检查是否需要钩子处理（计算图中是否存在注册了钩子的模块）
        3. 执行反向传播
           - 根据need_hook_processing选择执行路径：
             * True: 调用_backward_with_hooks，处理反向预处理钩子和反向钩子
             * False: 调用_backward_without_hooks，标准反向传播无额外开销

        钩子处理机制（仅在need_hook_processing为True时启用）：
        - 反向预处理钩子：在模块所有输出梯度就绪时调用，可修改输出梯度
        - 反向钩子：在模块所有输入梯度就绪时调用，可修改输入梯度
        - 由BackwardHookManager统一管理钩子调用时机和状态

        性能优化：
        - 无钩子场景使用快速路径，避免钩子管理开销
        - 钩子场景只处理模块相关节点，避免不必要的检查

        参数:
            gradient (TN | None, 可选): 输出张量的梯度，默认为None
                                       标量张量可省略，隐式使用1.0
                                       非标量张量必须提供与自身形状相同的梯度
            retain_graph (bool, 可选):  本参数为兼容Pytorch的设计，riemann反向传播不依赖该参数，
                                       无论True还是False，riemann都支持多次反向传播，默认为False
            create_graph (bool, 可选):  是否在梯度计算中创建计算图，
                                       设为True时可用于高阶导数计算，默认为False

        返回:
            None: 该函数不返回值，梯度结果直接存储在各节点的.grad属性中

        异常:
            RuntimeError: 当当前张量不需要梯度、非标量且无外部梯度、
                        或外部梯度形状不匹配时抛出
            TypeError: 当gradient参数类型不正确时抛出
        """
        # 参数验证和梯度准备
        if not self.requires_grad:
            raise RuntimeError('Only a tensor require grad can call backward()')

        if gradient is None:
            if self._array.ndim > 0:
                raise RuntimeError('Only a scalar can call backward() without argument grad_outputs')
            if self.is_complex():
                raise RuntimeError(f'grad can be implicitly created only for real scalar outputs but got {self.dtype}')

            # _init_calc_graph返回是否需要钩子处理
            need_hook_processing = self._init_calc_graph()
            self.grad_value = tensor(1.0, dtype=self.dtype, device=self.device, requires_grad=create_graph)

        elif isinstance(gradient, TN):
            if gradient._array.shape == self._array.shape:
                # _init_calc_graph返回是否需要钩子处理
                need_hook_processing = self._init_calc_graph()
                self.grad_value = gradient.to(self.device).detach().requires_grad_(create_graph)
            else:
                raise RuntimeError('shape of gradient need to be same as the shape of outputs')
        else:
            raise TypeError(f'gradient can be either tensor or None, but got {type(gradient)}')

        # 执行反向传播
        with set_grad_enabled(create_graph):
            if need_hook_processing:
                self._backward_with_hooks()
            else:
                self._backward_without_hooks()

        return
    
    def d(self, *args, create_graph=False):
        r"""
        计算当前标量张量对指定变量的高阶偏导数或混合偏导数。

        变量可以是任意形状（标量、向量、矩阵等）。同一变量可以多次出现，
        阶数自动累加。按 args 顺序逐变量求导，输出形状 = 各变量形状按顺序扁平拼接。

        参数:
            *args: 导数规格，两种形式可混用:
                - TN                对该变量求一阶
                - (TN, k)           对该变量求 k 阶 (k >= 1)
                例如 ``f.d(x)``、``f.d(x, x)``、``f.d((x, 3))``、``f.d((x, 2), y)``
            create_graph: 是否追踪结果计算图以支持继续高阶微分。默认 False。

        返回值:
            张量，形状为各变量形状的扁平拼接。
            例：``f.d(x, y)`` 其中 x:(2,) y:(3,) → 输出 shape=(2, 3) 交叉偏导矩阵

        异常:
            ValueError  当 self 非标量或无梯度；args 含非法项

        示例:
            # 标量变量（与旧版完全兼容）
            >>> x, y = tensor(2.0, rg=True), tensor(3.0, rg=True)
            >>> f = x**3 * y**2
            >>> f.d(x)          # 3*x^2*y^2 = 108
            >>> f.d(x, y)       # 6*x^2*y = 72   混合偏导
            >>> f.d((x, 2))     # 6*x*y^2 = 108  显式二阶

            # 向量变量（新增能力）
            >>> x = tensor([1., 2., 3.], rg=True)
            >>> f = (x**3).sum()
            >>> f.d(x)          # [3, 12, 27]     梯度向量
            >>> f.d(x, x)       # diag([6,12,18]) Hessian
            >>> f.d((x, 2))     # 同上

            # 多向量混合偏导
            >>> x = tensor([1., 2.], rg=True)
            >>> y = tensor([3., 4.], rg=True)
            >>> f = (x * y).sum() ** 2
            >>> f.d(x, y)       # (2, 2) 矩阵

        算法说明:
            1. 解析 args 为 [(var, k), ...]，同变量阶数累加
            2. 顺序执行 total_steps = Σk 次 grad:
               - current 是标量 → 直接对 var 一次 grad，返回同形状梯度
               - current 是张量 → 对其每个元素（标量）单独 grad，再 stack + reshape
                 (autograd.grad 只接受标量 outputs，所以必须逐元素)
            3. reshape 回 shape = ⊕ shape(vi) × ki
        """
        _grad = api_registry.get("grad")
        _rstack = api_registry.get("stack")
        
        # ---- 1. 前置校验 ----
        if self.ndim > 0 or not self.requires_grad:
            raise ValueError("self must be a scalar tensor with requires_grad=True")
        if not args:
            raise ValueError("At least one variable must be provided")

        # ---- 2. 解析 args → [(var, 总阶数), ...] ----
        # 先累加同一变量的阶数，再从 dict 构造 specs
        vid_to_order: dict[int, int] = {}   # id(var) → 累计阶数
        var_order: list[TN] = []             # 保持首次出现的顺序

        for item in args:
            if isinstance(item, TN):
                var, k = item, 1
            elif isinstance(item, (tuple, list)) and len(item) == 2 \
                    and isinstance(item[0], TN) and isinstance(item[1], int):
                var, k = item
            else:
                raise ValueError(
                    f"Each arg must be a TN or (TN, int), got {type(item).__name__}"
                )
            if k < 1:
                raise ValueError(f"Derivative order k must be >= 1, got {k}")
            if not var.requires_grad:
                raise ValueError("Variable must have requires_grad=True")

            vid = id(var)
            if vid in vid_to_order:
                vid_to_order[vid] += k
            else:
                vid_to_order[vid] = k
                var_order.append(var)

        specs: list[tuple[TN, int]] = [(var, vid_to_order[id(var)]) for var in var_order]

        # ---- 3. 预计算 ----
        total_steps = 0
        for _, k in specs:
            total_steps += k
        final_shape: tuple[int, ...] = ()
        for var, k in specs:
            final_shape += var.shape * k

        # ---- 4. 核心: total_steps 次顺序 grad ----
        # 局部 helper: 对单个标量 outputs 求一次 grad
        def _grad_once(scalar_out: TN, var_in: TN,
                       need_graph: bool, retain_graph: bool) -> TN | None:
            return _grad(scalar_out, var_in,
                         create_graph=need_graph,
                         allow_unused=True,
                         retain_graph=retain_graph)[0]

        current: TN = self
        step = 0

        for var, order in specs:
            for _ in range(order):
                is_last = (step == total_steps - 1)
                need_graph = create_graph if is_last else True  # 中间步骤必须追踪图
                retain_graph = not is_last

                if current.ndim == 0:
                    # 标量 → 一次 grad
                    g = _grad_once(current, var, need_graph, retain_graph)
                    if g is None:
                        return zeros_like(self, requires_grad=False)
                    current = g
                else:
                    # 张量 → 逐元素标量 grad，再 stack 保持计算图
                    # 注：autograd.grad outputs 必须是标量，所以对张量 outputs
                    #     必须逐个元素求导再拼回形状
                    flat = current.reshape(-1)
                    elem_grads: list[TN] = []
                    for ei in range(flat.shape[0]):
                        g_elem = _grad_once(flat[ei], var, need_graph, retain_graph)
                        if g_elem is None:
                            elem_grads.append(zeros_like(var, requires_grad=False))
                        else:
                            elem_grads.append(g_elem)

                    stacked = _rstack(elem_grads, dim=0)
                    target_shape = current.shape + var.shape
                    current = stacked if stacked.shape == target_shape \
                        else stacked.reshape(*target_shape)

                step += 1

        # ---- 5. reshape 回最终形状 ----
        if current.shape != final_shape:
            current = current.reshape(*final_shape)
        return current

    def _run_registered_api(self, name: str, *args, **kwargs):
        """统一入口，根据名字调用注册函数"""
        return api_registry.get(name)(*args, **kwargs)
# end of class TN

class BackwardHookManager:
    """
    反向传播钩子管理器

    统一管理模块的反向预处理钩子(full_backward_pre_hook)和反向钩子(full_backward_hook)。
    
    核心设计原则：
    1. 反向预处理钩子：模块所有输出梯度收集完成后调用，可修改输出梯度
    2. 反向钩子：模块所有输入梯度收集完成后调用，可修改输入梯度
    3. 统一在节点出栈时处理钩子，确保梯度已就绪
    4. 节点不可能同时是反向预处理模块的输出和反向钩子模块的输入（互斥性）
    
    数据结构：
    - _backward_pre_hooks_processed: 已处理的反向预处理钩子模块
    - _backward_hooks_processed: 已处理的反向钩子模块
    - _module_output_ids_to_cleanup: 需要清理的模块输出ID集合
    - _popped_tensors: 已出栈的节点ID集合（用于判断梯度就绪）
    """
    
    def __init__(self):
        self._backward_pre_hooks_processed: set = set()
        self._backward_hooks_processed: set = set()
        # {module: set(output_ids)} - 用于cleanup时快速清理缓存
        self._module_output_ids_to_cleanup: dict = {}
        # {tensor_id} - 已出栈节点，用于判断输入或输出梯度是否就绪
        self._popped_tensors: set = set()
    
    def wait_for_backward_pre_hook_ready(self, tensor: TN) -> tuple[set, set]:
        """
        检查反向预处理钩子是否就绪，返回可传播节点和就绪模块集合。
        
        在节点出栈时调用，完成以下任务：
        1. 缓存输出梯度（用于反向钩子和预处理钩子）
        2. 检查模块输出梯度是否收集完成
        3. 确定可反向传播的节点集合
        
        关键逻辑：
        - 非模块输出节点：直接返回{tensor}
        - 模块输出节点：缓存梯度，检查是否就绪，返回模块所有输出节点
        
        注意：节点出栈状态由调用方（process_module_hooks）记录，此函数不重复记录。
        
        参数:
            tensor: 刚出栈的节点（grad_value已就绪，且已在_popped_tensors中）
            
        返回:
            (items_for_propagate, pre_hook_modules)
            - items_for_propagate: 可反向传播节点集合
            - pre_hook_modules: 已就绪的反向预处理钩子模块集合
        """
        pre_hook_modules = set()
        
        # 非模块输出节点，直接返回
        if tensor._output_of is None:
            return {tensor}, pre_hook_modules
        
        module = tensor._output_of
        if id(tensor) in module._backward_hook_cache:
            output_id = id(tensor)
            cache_entry = module._backward_hook_cache[output_id]
            
            # 缓存输出梯度
            if tensor.grad_value is not None:
                cache_entry['grad_clone'] = tensor.grad_value.clone()
                
                # 多输出模块：缓存同组其他输出的梯度
                multi_group = cache_entry.get('multi_group')
                if multi_group is not None:
                    for other_output in multi_group:
                        if other_output is tensor:
                            continue
                        if other_output.grad_value is not None:
                            other_cache = module._backward_hook_cache.get(id(other_output))
                            if other_cache is not None and other_cache.get('grad_clone') is None:
                                other_cache['grad_clone'] = other_output.grad_value.clone()
            
            # 记录需要清理的output_id
            if module not in self._module_output_ids_to_cleanup:
                self._module_output_ids_to_cleanup[module] = set()
            self._module_output_ids_to_cleanup[module].add(output_id)
            
            # 检查反向预处理钩子是否就绪
            # 前置检查：module._backward_pre_hooks 非空，避免不必要的 _check_backward_pre_hooks_ready 调用
            if module._backward_pre_hooks and self._check_backward_pre_hooks_ready(module):
                pre_hook_modules.add(module)
        
        # 确定可传播节点
        items_for_propagate = self._get_propagate_items_for_pre_hook(tensor, pre_hook_modules)
        
        return set(items_for_propagate), pre_hook_modules
    
    def _get_propagate_items_for_pre_hook(self, tensor: TN, hook_modules: set) -> list:
        """
        获取可反向传播的节点列表。
        
        核心逻辑：
        1. 非注册钩子模块输入输出的节点（如参数）-> 返回 []
        2. tensor 是已就绪模块的输出 -> 返回该模块的所有输出
        3. tensor 是未就绪的反向预处理钩子模块的输出 -> 返回 []（等待）
           等待条件：多输出模块中，组内有输出未出栈（id(out) not in _popped_tensors）
        4. 其他情况 -> 返回 [tensor]
        
        注意：hook_modules 已由 wait_for_backward_pre_hook_ready 通过 _check_backward_pre_hooks_ready
        检查过就绪状态（基于 rcv_grad_count），此处直接使用，无需重复检查。
        但第3点中的等待检查基于 _popped_tensors，与就绪检查不同，用于确保所有输出都已出栈。
        """
        # 非注册钩子模块输入输出的节点（如参数）不传播
        if tensor._module_input_of is not None and tensor._output_of is None:
            return []
        
        # 有就绪模块：返回该模块的所有输出
        if hook_modules:
            for module in hook_modules:
                if id(tensor) not in module._backward_hook_cache:
                    continue
                cache_entry = module._backward_hook_cache[id(tensor)]
                multi_outputs = cache_entry.get('multi_group')
                if multi_outputs:
                    return [out for out in multi_outputs if id(out) in self._popped_tensors]
                return [tensor]
        
        # 无就绪模块：检查是否是未就绪的反向预处理钩子模块的输出
        # 注意：这里只检查是否有注册钩子，不检查就绪状态（已在 wait_for_backward_pre_hook_ready 中检查）
        if tensor._output_of is not None:
            module = tensor._output_of
            if id(tensor) in module._backward_hook_cache:
                # 是反向预处理钩子模块的输出但未就绪 -> 等待
                if module._backward_pre_hooks:
                    cache_entry = module._backward_hook_cache[id(tensor)]
                    multi_outputs = cache_entry.get('multi_group')
                    if multi_outputs:
                        # 多输出模块：检查是否所有输出都已出栈或从未收到梯度
                        # 条件：需要梯度且 (已出栈 或 从未收到梯度)
                        # 已出栈的输出：id(out) in _popped_tensors
                        # 未参与损失计算的输出：rcv_grad_count == 0（从未收到梯度或已收集完成）
                        for out in multi_outputs:
                            if out.requires_grad and id(out) not in self._popped_tensors and out.rcv_grad_count > 0:
                                return []  # 未就绪，等待
                        # 已就绪但未在hook_modules中（可能已通过其他输出添加）
                        return [out for out in multi_outputs if id(out) in self._popped_tensors]
                    # 单输出模块未就绪
                    return []
        
        return [tensor]
    
    def _should_call_backward_pre_hooks(self, module) -> bool:
        """
        检查是否应该调用反向预处理钩子（准入条件）。
        
        准入条件（与PyTorch一致）：
        - 至少有一个输入需要梯度，或者模块有参数
        - 如果都没有，调用钩子无意义（没有梯度可修改）
        
        注意：这是"准入条件"，不是"就绪条件"。就绪条件（梯度收集完成）
        在 _check_backward_pre_hooks_ready 中检查。
        
        参数:
            module: 要检查的模块
            
        返回:
            bool: 是否满足准入条件
        """
        if not module._backward_pre_hooks:
            return False
        
        # 检查是否至少有一个输入需要梯度
        for cache_entry in module._backward_hook_cache.values():
            for inp in cache_entry.get('inputs', ()):
                if isinstance(inp, TN) and inp.requires_grad:
                    return True
        
        # 没有输入需要梯度，检查是否有参数需要梯度
        return any(param.requires_grad for param in module.parameters())
    
    def _check_backward_pre_hooks_ready(self, module) -> bool:
        """
        检查反向预处理钩子是否就绪（就绪条件）。
        
        就绪条件：
        - 单输出模块：该输出节点的 rcv_grad_count == 0（所有上游梯度已收集完成）
        - 多输出模块：该组内所有需要梯度的输出节点的 rcv_grad_count == 0
        
        关键设计点：
        1. 使用 rcv_grad_count 而非 grad_value 检测就绪状态
           - rcv_grad_count == 0 表示所有上游梯度已收集完成
           - grad_value 可能在收集完成后被清空，不可靠
           - 对于不参与损失计算的输出（grad_value=None），rcv_grad_count 仍为 0
        
        2. 多输出模块的特殊处理
           - 多输出必须作为一个组一起处理（与 PyTorch 行为一致）
           - 必须等待组内所有需要梯度的输出都就绪才能调用钩子
        
        3. 部分输出不参与损失计算的场景
           - 不参与损失计算的输出：rcv_grad_count == 0, grad_value == None
           - 参与损失计算的输出：rcv_grad_count == 0, grad_value != None
           - 只要组内所有需要梯度的输出 rcv_grad_count == 0，就认为组已就绪
        
        注意：这是"就绪条件"，不是"准入条件"。准入条件
        在 _should_call_backward_pre_hooks 中检查。
        
        前置条件：调用此函数前已确认 module._backward_pre_hooks 非空
        
        参数:
            module: 要检查的模块
            
        返回:
            bool: 是否存在至少一个就绪的输出（或输出组）
        """
        # 检查准入条件
        # 注意：_should_call_backward_pre_hooks 内部会检查 module._backward_pre_hooks
        if not self._should_call_backward_pre_hooks(module):
            return False
        
        processed_groups = set()
        
        # 遍历所有输出，找到任意一个就绪的输出（或输出组）即返回True
        for output_id, cache_entry in module._backward_hook_cache.items():
            if (id(module), output_id) in self._backward_pre_hooks_processed:
                continue
            
            output_tensor = cache_entry.get('tensor')
            if output_tensor is None or not output_tensor.requires_grad:
                continue
            
            multi_group = cache_entry.get('multi_group')
            
            if multi_group is None:
                # 单输出：检查该输出是否就绪
                # rcv_grad_count == 0 表示所有上游梯度已收集完成
                if output_tensor.rcv_grad_count == 0:
                    return True
            else:
                # 多输出：按组检查
                group_id = id(multi_group)
                if group_id in processed_groups:
                    continue
                processed_groups.add(group_id)
                
                # 检查组内所有需要梯度的输出是否就绪
                # 注意：不参与损失计算的输出 rcv_grad_count 也为 0
                group_ready = all(
                    out.rcv_grad_count == 0 
                    for out in multi_group 
                    if out.requires_grad
                )
                if group_ready:
                    return True
        
        return False
    
    def _call_backward_pre_hooks_chain(self, module, grad_output_tuple: tuple) -> tuple:
        """
        调用反向预处理钩子链（支持链式处理）。
        
        链式处理：
        - 按注册顺序依次调用每个钩子
        - 钩子返回值作为下一个钩子的输入
        - 支持返回元组或单个张量
        
        参数:
            module: 模块
            grad_output_tuple: 输出梯度元组
            
        返回:
            (处理后的梯度元组, 是否调用了钩子)
        """
        hooks_called = False
        current_grad_output = grad_output_tuple
        
        for hook in module._backward_pre_hooks.values():
            hook_result = hook(module, current_grad_output)
            hooks_called = True
            # 如果钩子返回了值，更新 current_grad_output 供下一个钩子使用
            if hook_result is not None:
                current_grad_output = hook_result if isinstance(hook_result, tuple) else (hook_result,)
        
        return current_grad_output, hooks_called
    
    def _call_backward_pre_hooks(self, module) -> bool:
        """
        调用模块的反向预处理钩子。
        
        处理流程：
        1. 获取所有未处理的输出
        2. 按 multi_group 分组（单输出视为一组）
        3. 收集梯度，调用钩子链
        4. 根据钩子返回值更新输出梯度
        5. 更新 grad_clone 供反向钩子使用
        
        参数:
            module: 模块
            
        返回:
            bool: 是否成功处理了至少一个输出组
        """
        if not module._backward_pre_hooks:
            return False
        
        processed_groups = set()
        processed_any = False
        
        for output_id, cache_entry in module._backward_hook_cache.items():
            if (id(module), output_id) in self._backward_pre_hooks_processed:
                continue
            
            output_tensor = cache_entry.get('tensor')
            if output_tensor is None:
                continue
            
            multi_group = cache_entry.get('multi_group')
            
            if multi_group is None:
                # 单输出：检查是否已出栈
                if id(output_tensor) not in self._popped_tensors:
                    continue
                outputs_to_process = (output_tensor,)
            else:
                # 多输出：检查组是否已处理
                group_id = id(multi_group)
                if group_id in processed_groups:
                    continue
                
                # 检查组内所有需要梯度且会参与反向传播的输出是否都已出栈
                # 注意：不参与损失计算的输出不会入栈（没有收到梯度），不需要等待
                # 通过 rcv_grad_count 判断：rcv_grad_count == 0 表示已就绪（或从未收到梯度）
                # 通过 _popped_tensors 判断：已出栈的节点
                # 条件：需要梯度且 (已出栈 或 从未收到梯度)
                group_ready = all(
                    id(out) in self._popped_tensors or out.rcv_grad_count == 0
                    for out in multi_group
                    if out.requires_grad
                )
                if not group_ready:
                    continue
                
                outputs_to_process = multi_group
                processed_groups.add(group_id)
            
            # 过滤需要梯度的输出
            outputs_requires_grad = tuple(out for out in outputs_to_process if out.requires_grad)
            
            # 无需要梯度的输出，直接标记为已处理
            if not outputs_requires_grad:
                for out in outputs_to_process:
                    self._backward_pre_hooks_processed.add((id(module), id(out)))
                processed_any = True
                continue
            
            # 收集梯度（优先使用缓存的grad_clone）
            grad_outputs = []
            for out in outputs_requires_grad:
                out_cache = module._backward_hook_cache.get(id(out))
                grad = out_cache.get('grad_clone') if out_cache else None
                if grad is None:
                    grad = out.grad_value
                grad_outputs.append(grad)
            
            grad_outputs = tuple(grad_outputs)
            
            # 调用钩子链
            current_grad_output, hooks_called = self._call_backward_pre_hooks_chain(module, grad_outputs)
            
            # 处理钩子返回值
            if hooks_called and current_grad_output is not grad_outputs:
                if isinstance(current_grad_output, (tuple, list)):
                    for i, out in enumerate(outputs_requires_grad):
                        if i < len(current_grad_output):
                            out.grad_value = current_grad_output[i] if current_grad_output[i] is not None else zeros_like(out, requires_grad=False)
                else:
                    outputs_requires_grad[0].grad_value = current_grad_output
            
            # 更新grad_clone供反向钩子使用
            for out in outputs_requires_grad:
                out_cache = module._backward_hook_cache.get(id(out))
                if out_cache is not None:
                    out_cache['grad_clone'] = out.grad_value.clone() if out.grad_value is not None else None
            
            # 标记所有输出为已处理
            for out in outputs_to_process:
                self._backward_pre_hooks_processed.add((id(module), id(out)))
            
            processed_any = True
        
        return processed_any
    
    def process_module_hooks(self, tensor: TN) -> set[TN]:
        """
        统一处理模块钩子（反向预处理钩子和反向钩子）。

        核心逻辑：
        1. 记录节点出栈状态
        2. 获取两种钩子的可传播节点和就绪模块
        3. 融合判断：任一集合为空则不能传播（条件未满足）
        4. 两个集合都非空时，调用钩子并返回并集

        关键设计（互斥性）：
        - 节点不可能同时是反向预处理模块的输出和反向钩子模块的输入
        - 因此正常情况下，items_for_pre_hook 和 items_for_backward_hook 只有一个非空

        参数:
            tensor: 当前出栈的节点张量

        返回:
            set[TN]: 可反向传播节点集合（空集表示不能传播）
        """
        # 记录模块相关节点已出栈（供后续钩子就绪检查使用）
        # 优化：只记录模块输入/输出节点，避免记录所有节点造成内存浪费
        if tensor._output_of is not None or tensor._module_input_of is not None:
            self._popped_tensors.add(id(tensor))
        
        # 获取两种钩子的可传播节点和模块
        items_for_pre_hook, pre_hook_modules = self.wait_for_backward_pre_hook_ready(tensor)
        items_for_backward_hook, backward_hook_modules = self.wait_for_backward_hook_ready(tensor)
        
        # 融合可传播节点集合
        # 如果任一集合为空，说明对应的钩子条件未满足，不能传播
        # 由于互斥性，正常情况下只有一个集合会非空
        if not items_for_pre_hook or not items_for_backward_hook:
            # 至少一个集合为空，不能反向传播，也不调用钩子
            return set()
        
        # 两个集合都非空，可以反向传播
        # 调用已就绪的钩子（wait函数返回的模块已通过就绪检查，直接调用内部函数）
        if pre_hook_modules:
            for module in pre_hook_modules:
                self._call_backward_pre_hooks(module)
        if backward_hook_modules:
            for module in backward_hook_modules:
                self._call_backward_hooks(module)
        
        # 取并集作为可传播节点集合
        items_for_propagate = items_for_pre_hook | items_for_backward_hook
        
        return items_for_propagate

    def _check_backward_hooks_ready(self, module) -> bool:
        """
        检查模块是否准备好调用反向钩子（就绪条件）。
        
        就绪条件：
        1. 所有需要梯度的输入节点的 rcv_grad_count == 0
        2. 所有需要梯度的输入节点都已出栈（_popped_tensors）
        3. 至少有一个输入需要梯度，或者模块有参数
        
        关键设计：
        - 输入节点必须出栈后才能调用反向钩子
        - 防止钩子修改仍在栈中的节点的梯度
        
        参数:
            module: 要检查的模块
            
        返回:
            bool: 模块是否就绪
        """
        if not module._backward_hooks:
            return False

        # 收集所有需要检查的输入（未处理的、有梯度的输出对应的输入）
        inputs_to_check = set()
        
        for output_id, cache_entry in module._backward_hook_cache.items():
            if (id(module), output_id) in self._backward_hooks_processed:
                continue
            # 只检查有梯度副本的输出节点（已参与当前backward的）
            if cache_entry.get('grad_clone') is None:
                continue
            
            for inp in cache_entry.get('inputs', ()):
                if inp.requires_grad:
                    inputs_to_check.add(inp)
        
        # 没有输入需要梯度时，检查是否有参数需要梯度（准入条件）
        if not inputs_to_check:
            return any(param.requires_grad for param in module.parameters())
        
        # 检查所有输入是否都已就绪（梯度收集完成且已出栈）
        for inp in inputs_to_check:
            if inp.rcv_grad_count > 0:
                return False
            if id(inp) not in self._popped_tensors:
                return False
        
        return True
    
    def _call_backward_hooks(self, module):
        """
        调用模块的反向钩子。
        
        处理流程：
        1. 构建 grad_input（从输入节点的 grad_value 获取）
        2. 构建 grad_output（从缓存的 grad_clone 获取）
        3. 调用钩子链
        4. 根据返回值更新输入节点的 grad_value
        
        钩子返回值处理：
        - 返回 None：不修改梯度
        - 返回 tuple/list：按位置更新对应输入的梯度
        - 返回的元素为 None：将该输入的梯度设为 0
        
        参数:
            module: 要调用钩子的模块
        """
        if not module._backward_hooks:
            return

        # 获取有grad_clone的输出ID（已参与当前backward的）
        current_output_ids = [
            output_id for output_id, cache_entry in module._backward_hook_cache.items()
            if cache_entry.get('grad_clone') is not None
        ]

        if not current_output_ids:
            return

        processed_first_ids = set()

        for output_id in current_output_ids:
            if (id(module), output_id) in self._backward_hooks_processed:
                continue

            cache_entry = module._backward_hook_cache.get(output_id)
            if cache_entry is None:
                self._backward_hooks_processed.add((id(module), output_id))
                continue

            forward_inputs = cache_entry.get('inputs', ())
            if not forward_inputs:
                self._backward_hooks_processed.add((id(module), output_id))
                continue

            multi_outputs = cache_entry.get('multi_group')

            if multi_outputs is None:
                # 单输出模块
                output_tensor = cache_entry.get('tensor')
                if output_tensor is None:
                    self._backward_hooks_processed.add((id(module), output_id))
                    continue
                multi_outputs = (output_tensor,)
                first_output_id = output_id
                is_multi_output = False
            else:
                first_output_id = id(multi_outputs[0])
                is_multi_output = len(multi_outputs) > 1
            
            # 多输出组去重
            if is_multi_output and first_output_id in processed_first_ids:
                self._backward_hooks_processed.add((id(module), output_id))
                continue
            
            # 构建grad_input
            all_inputs_no_grad = all(not inp.requires_grad for inp in forward_inputs)
            grad_input_tuple = tuple(
                inp.grad_value if inp.grad_value is not None else zeros_like(inp, requires_grad=False)
                if inp.requires_grad else
                (None if all_inputs_no_grad else zeros_like(inp, requires_grad=False))
                for inp in forward_inputs
            )
            
            # 构建grad_output
            grad_output_tuple = tuple(
                module._backward_hook_cache.get(id(out), {}).get('grad_clone')
                for out in multi_outputs
            )
            
            # 调用钩子链
            for hook in module._backward_hooks.values():
                hook_result = hook(module, grad_input_tuple, grad_output_tuple)
                if hook_result is not None:
                    # 验证返回值
                    if not isinstance(hook_result, (tuple, list)):
                        raise RuntimeError(
                            f"Backward hook returned an invalid number of grad_input, "
                            f"got 1 (non-tuple), but expected {len(forward_inputs)}"
                        )
                    if len(hook_result) != len(forward_inputs):
                        raise RuntimeError(
                            f"Backward hook returned an invalid number of grad_input, "
                            f"got {len(hook_result)}, but expected {len(forward_inputs)}"
                        )
                    
                    # 更新输入梯度
                    for i, inp in enumerate(forward_inputs):
                        if i < len(hook_result):
                            inp.grad_value = hook_result[i] if hook_result[i] is not None else zeros_like(inp, requires_grad=False)
                else:
                    # 钩子返回None，使用grad_input_tuple中的梯度
                    for i, inp in enumerate(forward_inputs):
                        if i < len(grad_input_tuple) and grad_input_tuple[i] is not None:
                            inp.grad_value = grad_input_tuple[i]
            
            # 标记输出为已处理
            for out in multi_outputs:
                self._backward_hooks_processed.add((id(module), id(out)))
            
            if is_multi_output:
                processed_first_ids.add(first_output_id)

    def wait_for_backward_hook_ready(self, tensor: TN) -> tuple[set, set]:
        """
        检查反向钩子是否就绪，返回可传播节点和就绪模块集合。
        
        在节点出栈时调用，完成以下任务：
        1. 检查tensor是否是注册了反向钩子的模块的输入
        2. 检查模块输入梯度是否收集完成
        3. 确定可反向传播的节点集合
        
        关键逻辑：
        - 非反向钩子模块输入节点：直接返回{tensor}
        - 反向钩子模块输入节点，梯度未收集齐 -> 返回空集（等待）
        - 反向钩子模块输入节点，梯度已收集齐 -> 返回模块所有输入节点
        
        注意：节点出栈状态由调用方（process_module_hooks）记录，此函数不重复记录。
        
        参数:
            tensor: 刚出栈的节点（grad_value已就绪，且已在_popped_tensors中）
            
        返回:
            (items_for_propagate, backward_hook_modules)
            - items_for_propagate: 可反向传播节点集合
            - backward_hook_modules: 已就绪的反向钩子模块集合
        """
        backward_hook_modules = set()
        
        # 收集所有相关模块（输入和输出关联的模块）
        # 注意：根据设计原则，节点不会同时是反向预处理模块的输出和反向钩子模块的输入
        # （因为输入被clone，输出也被clone，两者是不同的张量）
        # 但为了代码健壮性，仍检查两种关联
        related_modules = set()
        if tensor._module_input_of is not None:
            module = tensor._module_input_of
            if module._backward_hooks:
                related_modules.add(module)
        if tensor._output_of is not None:
            module = tensor._output_of
            if module._backward_hooks:
                related_modules.add(module)
        
        # 检查哪些模块已就绪
        for module in related_modules:
            if self._check_backward_hooks_ready(module):
                backward_hook_modules.add(module)
        
        # 确定可传播节点
        items_for_propagate = self._get_propagate_items_for_backward_hook(tensor, backward_hook_modules)
        
        return items_for_propagate, backward_hook_modules
    
    def _get_propagate_items_for_backward_hook(self, tensor: TN, hook_modules: set) -> set:
        """
        获取反向钩子处理后可反向传播的节点集合。
        
        核心逻辑：
        1. 非反向钩子模块输入节点 -> 返回{tensor}
        2. 反向钩子模块输入节点，模块未就绪 -> 返回空集（等待）
        3. 反向钩子模块输入节点，模块已就绪 -> 返回模块所有输入节点
        """
        tensor_id = id(tensor)
        
        # 检查tensor是否是反向钩子模块的输入
        if tensor._module_input_of is None:
            return {tensor}
        
        # 找到tensor所属的反向钩子模块
        module = tensor._module_input_of
        if not module._backward_hooks:
            return {tensor}
        
        # 查找包含该输入的缓存条目
        # 多输出模块的所有条目共享相同的inputs元组，找到任意一个即可获取所有输入
        # 多次前向传播的输入/输出都是独立对象（被clone），通过输入id能正确匹配到对应条目
        related_cache_entry = None
        for cache_entry in module._backward_hook_cache.values():
            if any(tensor_id == id(inp) for inp in cache_entry.get('inputs', ())):
                related_cache_entry = cache_entry
                break
        
        # 不是反向钩子模块的输入
        if related_cache_entry is None:
            return {tensor}
        
        # 模块未就绪：检查是否已处理过
        if not hook_modules:
            output_id = id(related_cache_entry.get('tensor'))
            if (id(module), output_id) in self._backward_hooks_processed:
                # 已处理过，正常传播
                return {tensor} if tensor.requires_grad else set()
            # 未处理过，等待
            return set()
        
        # 模块已就绪：返回所有相关输入节点
        items = set()
        for inp in related_cache_entry.get('inputs', ()):
            if inp.requires_grad and id(inp) in self._popped_tensors:
                items.add(inp)
        return items
    
    def cleanup(self):
        """
        清理本次反向传播相关的所有缓存。

        清理内容：
        1. 各模块的 _backward_hook_cache
        2. 本管理器的处理状态集合
        """
        # 快速退出：没有需要清理的模块
        if not self._module_output_ids_to_cleanup:
            return

        for module, output_ids in self._module_output_ids_to_cleanup.items():
            # 优化：清理_backward_hook_cache中指定的output_ids
            for key in list(module._backward_hook_cache.keys()):
                if key in output_ids:
                    del module._backward_hook_cache[key]

        self._backward_pre_hooks_processed.clear()
        self._backward_hooks_processed.clear()
        self._module_output_ids_to_cleanup.clear()
        self._popped_tensors.clear()


@contextmanager
def device_context(tensor_obj: TN):
    """根据张量设备类型，提供相应的设备上下文管理器
    
    对于CPU张量，不产生任何上下文
    对于CUDA张量，设置对应的CUDA设备
    
    参数:
        tensor_obj (TN): 输入张量
        
    示例:
        with device_context(x):
            # 在此上下文中，arrlib操作会在正确的设备上执行
            arr = x._get_array_lib()
            result = arr.some_operation(x._array)
    """
    if tensor_obj.device.type == 'cpu':
        yield
    else:
        with cp.cuda.Device(tensor_obj.device.index):
            yield

def _get_device(device:str|int|Device=None)->Device:
    """
    获取设备，该函数根据输入的设备参数，返回一个Device对象。

    参数:
        device: 可选，指定设备的字符串表示、整数索引或Device对象。
                可以是'cpu'、'cuda'、'cuda:0'、从0开始的整数等。
                如果为None，则返回当前默认设备（Device上下文覆盖优先，
                其次是set_default_device设置的默认设备，最后为CPU）。

    返回值:
        Device: 包含两个属性，type属性为'cpu'或'cuda'，
                index属性为CUDA设备索引（CPU为None）。

    异常:
        RuntimeError: 当指定的device无效或不可用时抛出
    """

    if device is not None:
        # 先将字符串、整数或Device对象统一转换为Device对象
        if isinstance(device, Device):
            return_device = device
        elif isinstance(device, str):
            return_device = Device(device)
        elif isinstance(device, int):
            # 整数表示cuda设备索引
            return_device = Device('cuda', device)
        else:
            raise RuntimeError(f"Invalid device type: {type(device).__name__}")
    else:
        # device参数为None，使用当前默认设备
        return_device = get_default_device()

    return return_device

def _get_device_for_input(device:str|int|Device|None, data:any)->Device:
    """
    为有数据来源的张量创建函数（tensor/as_tensor）解析目标设备。

    与torch的语义保持一致：
    - 显式指定device参数时，以该参数为准；
    - data本身是张量（TN）时，沿用data所在的设备；
    - data为numpy/cupy数组、列表、标量等无设备语义的来源时，
      使用当前默认设备（get_default_device）。

    参数:
        device: 可选，用户显式指定的设备参数，可为None。
        data: 张量创建函数的data参数。

    返回值:
        Device: 解析后的目标设备。

    异常:
        RuntimeError: 当指定的device无效或不可用时抛出
    """
    if device is not None:
        # 显式指定的device参数优先
        return _get_device(device)
    
    # numpy 数组按约定视为"无源数据"，走默认设备解析链；
    # 不能用 __dlpack_device__ 检测，因为 numpy 2.x 也实现了该协议（返回 kDLCPU），
    # 会把 numpy 数组错误归类为 CPU 设备来源，导致无法利用默认设备/Device 上下文迁移到 CUDA。
    if isinstance(data, np.ndarray):
        return _get_device(None)
    
    # 对于TN/torch/Jax等张量或cupy数组等类型的“有源数据”，
    # 通过 __dlpack_device__ 获取 device，该方法通用且不依赖任何框架库，就可以获得device
    # 返回值形如 (device_type, device_id)，kDLCUDA=2、kDLCPU=1、kDLROCM=3 等
    dlpack_device = getattr(data, '__dlpack_device__', None)
    if dlpack_device is not None:
        try:
            dev_type, dev_id = dlpack_device()
        except (TypeError, AttributeError, ValueError):
            # 协议签名异常或返回值非法，按无源数据处理
            dev_type, dev_id = None, None
        if dev_type == 2:  # kDLCUDA
            # 直接给整数索引：_get_device 会解析为 Device('cuda', dev_id)，既保留显式
            # 设备索引（不能写 'cuda'，那会被解析成 current_device()，多卡下误判源设备），
            # 又省去 f-string 拼接
            device = dev_id
        elif dev_type == 1:  # kDLCPU
            device = 'cpu'
        # 其他 dev_type（如 kDLROCM=3、kDLMetal=5）不识别，保持 None 走默认设备
    
    # 其他情况，device is None，_get_device(None)返回当前默认设备
    return _get_device(device)

def _get_dtype(dtype:any)->np.dtype:
    """
    获取数据类型，该函数根据输入的dtype参数，返回一个np.dtype对象。
    
    参数:
        dtype: 可选，指定数据类型的字符串表示或np.dtype对象。
                可以是'float32'、'float64'、'int32'、'int64'等。
                如果为None，则默认使用float32。
    
    返回值:
        np.dtype: 包含数据类型的对象。
    
    异常:
        TypeError: 当dtype参数无效时抛出
    """

    if isinstance(dtype, np.dtype):
        return dtype
    
    # 使用np.dtype转换，numpy可以处理各种字符串表示的类型
    # cupy和numpy的dtype类是统一的，所以用np.dtype处理即可，不用区分
    try:
        dtype_obj = np.dtype(dtype)
    except (TypeError, TypeError):
        raise TypeError(f"Cannot convert {dtype} to a valid dtype")

    return dtype_obj    

def _numpy_for_cuda(data, dtype):
    """
    在 CPU 侧把输入物化为 dtype 对齐的 numpy 数组，供 GPU 目标设备上推。

    先物化再交给 cupy，可让 cupy 只走 ndarray 分支，避开旧版本 cupy 中两条
    以 copy=False 调用 numpy 的回退路径（numpy 2.x 下会抛 ValueError）：
    _array_default 处理列表/标量/range 时，以及 _array_from_nested_sequence
    处理同形状数组列表且无 pinned 缓冲时。
    """
    if isinstance(data, np.ndarray):
        arr = data
    elif hasattr(data, '__array__'):
        # torch/TN 等协议对象：asarray 不探测 __array__ 的 copy 形参，避免旧式实现告警
        arr = np.asarray(data)
    else:
        # 列表/元组/标量/range：必然新建数组
        return np.asarray(data, dtype=dtype)

    if dtype is not None and arr.dtype != np.dtype(dtype):
        arr = arr.astype(dtype)
    return arr


def _to_device_array(data, target_index:int, dtype=None, copy:bool=False):
    """
    把数据搬到 target_index 指定的 GPU，并保证结果确实落在该设备上。

    cp.array/cp.asarray 的跨设备拷贝依赖 CuPy 的实现细节（copy=False 且 dtype
    相同时是否感知设备并不由文档严格保证），一旦某版本直接返回源数组，结果就会
    静默停留在原设备。这里显式校验结果所在设备，不一致时用主机中转兜底。
    copy=True 强制新建数组（tensor 语义），copy=False 允许同设备共享内存。
    """
    with cp.cuda.Device(target_index):
        result = cp.array(data, dtype=dtype) if copy else cp.asarray(data, dtype=dtype)
        if result.device.id != target_index:
            # CuPy 未执行跨设备拷贝：经主机中转，确保结果位于目标设备
            result = cp.array(cp.asnumpy(data), dtype=dtype)
    return result


def tensor(data, dtype:np.dtype|None = None, device:str|int|Device|None = None, requires_grad:bool|None = False)->TN:
    """
    创建一个新的张量对象，与输入数据永不共享内存。

    该函数是创建张量的主要入口，总是复制数据（对应 np.array 语义）。
    需要与输入共享内存时请使用 as_tensor。

    参数:
        data: 输入数据，支持以下类型：
              - Python 标量（int/float/bool/complex）、列表/元组（支持任意嵌套
                形成多维张量）、range 对象
              - numpy 数组、cupy 数组以及 numpy/cupy 标量（np.float32 等）
              - TN 张量（本框架创建的张量，总是独立拷贝、不共享内存）
              - 其他实现 __array__/__dlpack__ 协议的第三方框架张量（torch、jax
                等），经协议接口自动转换为底层数组
        dtype: 目标数据类型（'float32'、np.float32 等）。None 时不做类型转换：
              数组类输入保留原 dtype，标量/列表/元组/range 按框架规则推断（默认 float32）
        device: 目标设备，可为 'cpu'、'cuda'、'cuda:0'、Device 对象或 None。
              None 时不做设备迁移：data 有设备语义（TN、numpy/cupy 数组、框架
              张量，经 __dlpack_device__ 识别）时沿用其所在设备，纯 Python
              数据使用当前默认设备
        requires_grad: 是否需要计算该张量的梯度，默认 False

    返回值:
        TN: 新建的张量对象

    异常:
        RuntimeError: requires_grad=True 但数据类型不是浮点/复数型时抛出
        RuntimeError: 指定的 device 无效或不可用时抛出

    示例:
        >>> x = tensor([1, 2, 3])                               # CPU 整数张量
        >>> y = tensor([1.0, 2.0, 3.0], device='cuda')          # GPU 浮点张量
        >>> z = tensor([1.0, 2.0, 3.0], requires_grad=True)     # 启用梯度
        >>> w = tensor([1, 2, 3], dtype=float64, device='cpu')  # 指定 dtype 与设备

    注意事项:
        1. 与 data 永不共享内存：numpy/cupy 数组、TN、torch 等输入均独立拷贝
        2. 只有浮点型或复数型张量才能启用梯度
        3. 新建张量默认是计算图的叶节点
    """

    # 解析设备索引：显式device优先，其次沿用数据来源设备，否则用默认设备
    target_device = _get_device_for_input(device, data)
    
    # dtype 推断：无 __array__ 协议的对象（range/列表/元组/标量等）按元素类型推断，
    # 其余类型保留底层数组自身的 dtype
    if dtype is None and not hasattr(data, '__array__'):
        try:
            dtype = infer_data_type(data)
        except TypeError:
            dtype = None  # infer 也无法识别时兜底，交给 np.array/cp.array 处理

    if target_device.type == 'cuda':
        # GPU 目标设备：GPU 来源（cupy 数组 / CAI 张量）用 cp.asarray，同设备零拷贝共享、
        # 跨设备迁移；CPU 来源先在本地物化为 dtype 对齐的 numpy 数组再上推（必然新建
        # GPU 内存），这样 cupy 只走 ndarray 分支，避开旧版 cupy 的 numpy 回退路径
        # 以 copy=False 调 numpy（numpy 2.x 下抛 ValueError）
        if isinstance(data, cp.ndarray) or hasattr(data, '__cuda_array_interface__'):
            arr = _to_device_array(data, target_device.index, dtype, copy=True)
        else:
            arr = _to_device_array(_numpy_for_cuda(data, dtype), target_device.index, dtype, copy=False)
    else:
        # CPU 目标设备
        if cp is not None and isinstance(data, cp.ndarray):
            # 裸 cp.ndarray 没有返回 CPU numpy 数组的 __array__，需显式拉回；
            # asnumpy 已产生全新 CPU 数组，asarray 不再拷贝
            arr = np.asarray(cp.asnumpy(data), dtype=dtype)
        elif hasattr(data, '__array__') and not isinstance(data, np.ndarray):
            # TN/torch/jax 等协议对象：np.array 会探测 __array__(dtype, copy) 的 copy 形参，
            # torch 等旧式实现只接受 dtype，numpy 2.x 下会告警，故改用 asarray + astype
            arr = np.asarray(data)
            if dtype is None:
                dtype = arr.dtype  # astype(None) 会被解析为 float64，归位为自然 dtype
            arr = arr.astype(dtype, copy=True)  # 必须新建数组，保证 tensor 不共享源内存
        else:
            # 列表/元组/标量/range/numpy数组：np.array 直接构造或拷贝，无协议探测路径
            arr = np.array(data, dtype=dtype)

    tsobj = TN(arr, requires_grad=bool(requires_grad))
    if tsobj.requires_grad:
        # 浮点/复数 dtype 必然是数值类型，无需再单独做数值检查
        if not is_float_or_complex(tsobj._array.dtype):
            raise RuntimeError('Only floating point tensors can require gradients')
    elif not is_numeric_array(arr):
        raise TypeError(f"cannot create a TN tensor with dtype {arr.dtype}; only numeric or boolean dtypes are supported")
    return tsobj    

def as_tensor(data, dtype:np.dtype|None = None, device:str|int|Device|None = None, requires_grad:bool|None = False)->TN:
    """
    将输入数据转换为张量，尽可能与输入数据共享底层内存。

    与 tensor 共用同一套转换路径，区别仅在内存语义：tensor 总是复制；as_tensor
    在目标设备与 data 所在设备相同、且 dtype 一致（dtype 为 None 时沿用 data 的
    dtype）时不复制，而是共享底层内存（对应 np.asarray 语义）。data 为 TN 且
    dtype 与设备均匹配时，直接返回该对象本身，不做任何拷贝与修改。

    参数:
        data: 输入数据，支持以下类型：
              - Python 标量（int/float/bool/complex）、列表/元组（支持任意嵌套
                形成多维张量）、range 对象
              - numpy 数组、cupy 数组以及 numpy/cupy 标量（np.float32 等）
              - TN 张量（本框架创建的张量，内存语义见上）
              - 其他实现 __array__/__dlpack__ 协议的第三方框架张量（torch、jax
                等），经协议接口自动转换为底层数组，同设备同dtype时共享底层存储
        dtype: 目标数据类型（'float32'、np.float32 等）。None 时不做类型转换：
              数组类输入保留原 dtype，标量/列表/元组/range 按框架规则推断（默认 float32）
        device: 目标设备，可为 'cpu'、'cuda'、'cuda:0'、Device 对象或 None。
              None 时不做设备迁移：data 有设备语义（TN、numpy/cupy 数组、框架
              张量，经 __dlpack_device__ 识别）时沿用其所在设备，纯 Python
              数据使用当前默认设备
        requires_grad: 是否需要计算该张量的梯度，默认 False

    返回值:
        TN: 张量对象（data 为 TN 且 dtype、设备匹配时直接返回原对象，否则新建）

    异常:
        RuntimeError: requires_grad=True 但数据类型不是浮点/复数型时抛出
        RuntimeError: 指定的 device 无效或不可用时抛出

    示例:
        >>> arr = np.array([1.0, 2.0, 3.0])
        >>> x = as_tensor(arr)      # 同设备同 dtype，与 arr 共享内存
        >>> x[0] = 0                # 修改张量会同步影响 arr

    注意事项:
        1. 与 data 尽可能共享内存：numpy 数组、torch 等张量在同设备同 dtype 时
           经协议共享底层存储；设备迁移或 dtype 转换时必然复制
        2. data 为 TN 且 dtype、设备匹配时直接返回原对象，requires_grad 参数被忽略，
           原对象属性保持不变
        3. 只有浮点型或复数型张量才能启用梯度
        4. 新建张量默认是计算图的叶节点
    """
    # TN 输入：dtype 与设备均匹配（参数为 None 视为匹配）时直接返回原对象，对齐
    # torch.as_tensor 的 "data itself is returned" 语义；不匹配时保持 TN 交给后续路径，
    # 由其协议接口读取底层数据（CPU 走 __array__，GPU 走 __cuda_array_interface__）
    if isinstance(data, TN):
        dtype_matched = (dtype is None or _get_dtype(dtype) == data.dtype)
        device_matched = (device is None or _get_device(device) == data.device)
        if dtype_matched and device_matched:
            return data
    
    # 解析设备索引：显式device优先，其次沿用数据来源设备，否则用默认设备
    target_device = _get_device_for_input(device, data)
    
    # dtype 推断：无 __array__ 协议的对象（range/列表/元组/标量等）按元素类型推断，
    # 其余类型保留底层数组自身的 dtype
    if dtype is None and not hasattr(data, '__array__'):
        try:
            dtype = infer_data_type(data)
        except TypeError:
            dtype = None  # infer 也无法识别时兜底，交给 np.asarray/cp.asarray 处理

    if target_device.type == 'cuda':
        # GPU 目标设备：GPU 来源（cupy 数组 / CAI 张量）用 cp.asarray，同设备零拷贝共享、
        # 跨设备迁移；CPU 来源先在本地物化为 dtype 对齐的 numpy 数组再上推（必然新建
        # GPU 内存），这样 cupy 只走 ndarray 分支，避开旧版 cupy 的 numpy 回退路径
        # 以 copy=False 调 numpy（numpy 2.x 下抛 ValueError）
        if isinstance(data, cp.ndarray) or hasattr(data, '__cuda_array_interface__'):
            arr = _to_device_array(data, target_device.index, dtype, copy=False)
        else:
            arr = _to_device_array(_numpy_for_cuda(data, dtype), target_device.index, dtype, copy=False)
    else:
        # CPU 目标设备
        # 裸 cp.ndarray 没有返回 CPU numpy 数组的 __array__，需显式拉回；
        # torch/jax 等框架的 __array__ 都可由 np.asarray 统一读取；
        # numpy 原生数组在 dtype 为 None 时由 asarray 直接返回原对象，实现共享内存
        if cp is not None and isinstance(data, cp.ndarray):
            arr = np.asarray(cp.asnumpy(data), dtype=dtype)
        else:
            arr = np.asarray(data, dtype=dtype)

    tsobj = TN(arr, requires_grad = bool(requires_grad))
    if tsobj.requires_grad:
        # 浮点/复数 dtype 必然是数值类型，无需再单独做数值检查
        if not is_float_or_complex(tsobj._array.dtype):
            raise RuntimeError('Only floating point tensors can require gradients')
    elif not is_numeric_array(arr):
        raise TypeError(f"cannot create a TN tensor with dtype {arr.dtype}; only numeric or boolean dtypes are supported")
    return tsobj

def _as_tn(x):
    """Convert the input of a unary math function to TN via as_tensor
    (TN inputs are returned directly as a fast path; numpy/cupy/torch
    and other convertible objects are converted; dtype validity is checked
    inside as_tensor), raising a TypeError if the conversion fails."""
    if isinstance(x, TN):
        return x
    try:
        return as_tensor(x)
    except Exception as e:
        raise TypeError(
            f"cannot convert input of type {type(x).__name__} to a TN tensor"
        ) from e

def _validate_shape(shape:tuple|list):
    '''
    验证张量形状是否有效，返回有效形状元组。
    当一个函数的参数*arg表示可以整数序列，也可以是元组时，
    _validate_shape用于获得正确的元组并确保元组内元素是整数
    
    参数:
        shape: 输入元组，可包含嵌套元组或整数。
    
    返回:
        有效形状元组。
    
    异常:
        RuntimeError: 如果形状包含非整数元素。
    '''
    
    # 处理参数格式
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        newshape = shape[0]
    else:
        newshape = shape
    
    # 验证形状参数是否为整数
    # if any(not isinstance(dim, (int,np.integer)) for dim in newshape):
    #     raise RuntimeError(f'{shape} contains non int numbers')
    return newshape

def zeros(*shape,dtype:np.dtype|None = None,device:str|int|Device|None=None,requires_grad:bool|None = False)->TN:
    """
    创建一个全零张量。
    
    返回一个形状为指定形状、元素全为零的新张量。
    
    Args:
        *shape: 张量的形状，可以是一个整数序列或一个元组/列表
        dtype (np.dtype, optional): 张量的数据类型，如果为None则使用默认数据类型
        device(str|int|Device,optional): 可选，指定张量所在的设备，可以是'cpu'、'cuda'、CUDA设备索引、device对象或None
        requires_grad (bool, optional): 是否需要计算梯度，默认为False
        
    Returns:
        TN: 全零张量
        
    Examples:
        >>> zeros(3, 4)  # 创建3x4的全零张量
        >>> zeros((2, 3))  # 创建2x3的全零张量
    """
    shape_val = _validate_shape(shape)
    dt = get_default_dtype() if dtype is None else dtype
    dev = _get_device(device)
    if dev.type=='cpu':
        arr = np.zeros(shape_val,dt)
    else:
        with cp.cuda.Device(dev.index):
            arr = cp.zeros(shape_val,dt)
    
    return TN(arr, requires_grad=bool(requires_grad))
        
def zeros_like(tsr:TN, dtype:np.dtype|None = None,device:str|int|Device|None = None,requires_grad:bool|None = False)->TN:
    """
    创建一个与给定张量形状相同的全零张量。
    
    返回一个与输入张量形状相同、元素全为零的新张量。
    
    Args:
        tsr (TN): 参考张量，用于确定输出张量的形状
        dtype (np.dtype, optional): 输出张量的数据类型，如果为None则使用参考张量的数据类型
        device(str|int|Device,optional): 可选，指定张量所在的设备，可以是'cpu'、'cuda'、CUDA设备索引、device对象或None
        requires_grad (bool, optional): 是否需要计算梯度，默认为False
        
    Returns:
        TN: 与参考张量形状相同的全零张量
        
    Examples:
        >>> x = ones(3, 4)
        >>> y = zeros_like(x)  # 创建与x形状相同的全零张量
    """
    dt = tsr.dtype if dtype is None else dtype
    dev = tsr.device if device is None else _get_device(device)
    if dev.type=='cpu':
        arr = np.zeros_like(tsr._array,dt)
    else:
        with cp.cuda.Device(dev.index):
            arr = cp.zeros_like(tsr._array,dt)
    
    return TN(arr, requires_grad=bool(requires_grad))

def ones(*shape,dtype:np.dtype|None = None,device:str|int|Device|None=None,requires_grad:bool|None = False)->TN:
    """
    创建一个全一张量。
    
    返回一个形状为指定形状、元素全为一的新张量。
    
    Args:
        *shape: 张量的形状，可以是一个整数序列或一个元组/列表
        dtype (np.dtype, optional): 张量的数据类型，如果为None则使用默认数据类型
        device(str|int|Device,optional): 可选，指定张量所在的设备，可以是'cpu'、'cuda'、CUDA设备索引、device对象或None
        requires_grad (bool, optional): 是否需要计算梯度，默认为False
        
    Returns:
        TN: 全一张量
        
    Examples:
        >>> ones(3, 4)  # 创建3x4的全一张量
        >>> ones((2, 3))  # 创建2x3的全一张量
    """
    shape_val = _validate_shape(shape)
    dt = get_default_dtype() if dtype is None else dtype
    dev = _get_device(device)
    if dev.type=='cpu':
        arr = np.ones(shape_val,dt)
    else:
        with cp.cuda.Device(dev.index):
            arr = cp.ones(shape_val,dt)
    
    return TN(arr, requires_grad=bool(requires_grad))

def ones_like(tsr:TN, dtype:np.dtype|None = None,device:str|int|Device|None=None,requires_grad:bool|None = False)->TN:  
    """
    创建一个与给定张量形状相同的全一张量。
    
    返回一个与输入张量形状相同、元素全为一的新张量。
    
    Args:
        tsr (TN): 参考张量，用于确定输出张量的形状
        dtype (np.dtype, optional): 输出张量的数据类型，如果为None则使用参考张量的数据类型
        device(str|int|Device,optional): 可选，指定张量所在的设备，可以是'cpu'、'cuda'、CUDA设备索引、device对象或None
        requires_grad (bool, optional): 是否需要计算梯度，默认为False
        
    Returns:
        TN: 与参考张量形状相同的全一张量
        
    Examples:
        >>> x = zeros(3, 4)
        >>> y = ones_like(x)  # 创建与x形状相同的全一张量
    """
    dt = tsr.dtype if dtype is None else dtype
    dev = tsr.device if device is None else _get_device(device)
    if dev.type=='cpu':
        arr = np.ones_like(tsr._array,dt)
    else:
        with cp.cuda.Device(dev.index):
            arr = cp.ones_like(tsr._array,dt)
    
    return TN(arr, requires_grad=bool(requires_grad))

def empty(*shape,dtype:np.dtype|None = None,device:str|int|Device|None=None,requires_grad:bool|None = False)->TN:
    """
    创建一个未初始化的张量。
    
    返回一个形状为指定形状的新张量，但不初始化其元素的值。
    元素的值是未定义的，取决于内存的当前状态。
    
    Args:
        *shape: 张量的形状，可以是一个整数序列或一个元组/列表
        dtype (np.dtype, optional): 张量的数据类型，如果为None则使用默认数据类型
        device(str|int|Device,optional): 可选，指定张量所在的设备，可以是'cpu'、'cuda'、CUDA设备索引、device对象或None
        requires_grad (bool, optional): 是否需要计算梯度，默认为False
        
    Returns:
        TN: 未初始化的张量
        
    Examples:
        >>> x = empty(3, 4)  # 创建3x4的未初始化张量
        >>> y = empty((2, 3))  # 创建2x3的未初始化张量
        
    Note:
        由于张量未初始化，其元素值是不确定的，使用前应当先赋值。
    """
    shape_val = _validate_shape(shape)
    dt = get_default_dtype() if dtype is None else dtype
    dev = _get_device(device)
    if dev.type=='cpu':
        arr = np.empty(shape_val,dt)
    else:
        with cp.cuda.Device(dev.index):
            arr = cp.empty(shape_val,dt)
    
    return TN(arr, requires_grad=bool(requires_grad))

def empty_like(tsr:TN, dtype:np.dtype|None = None,device:str|int|Device|None=None,requires_grad:bool|None = False)->TN:
    """
    创建一个与给定张量形状相同的未初始化张量。
    
    返回一个与输入张量形状相同的新张量，但不初始化其元素的值。
    元素的值是未定义的，取决于内存的当前状态。
    
    Args:
        tsr (TN): 参考张量，用于确定输出张量的形状
        dtype (np.dtype, optional): 输出张量的数据类型，如果为None则使用参考张量的数据类型
        device(str|int|Device,optional): 可选，指定张量所在的设备，可以是'cpu'、'cuda'、CUDA设备索引、device对象或None
        requires_grad (bool, optional): 是否需要计算梯度，默认为False
        
    Returns:
        TN: 与参考张量形状相同的未初始化张量
        
    Examples:
        >>> x = ones(3, 4)
        >>> y = empty_like(x)  # 创建与x形状相同的未初始化张量
        
    Note:
        由于张量未初始化，其元素值是不确定的，使用前应当先赋值。
    """
    dt = tsr.dtype if dtype is None else dtype
    dev = tsr.device if device is None else _get_device(device)
    if dev.type=='cpu':
        arr = np.empty_like(tsr._array,dt)
    else:
        with cp.cuda.Device(dev.index):
            arr = cp.empty_like(tsr._array,dt)
    
    return TN(arr, requires_grad=bool(requires_grad))

def full(*shape,fill_value:Any,dtype:np.dtype|None = None,device:str|int|Device|None=None,requires_grad:bool|None = False)->TN:
    """
    创建一个填充了指定值的张量。
    
    返回一个形状为指定形状、所有元素都填充为指定值的新张量。
    
    Args:
        *shape: 张量的形状，可以是一个整数序列或一个元组/列表
        fill_value: 用于填充张量的值
        dtype (np.dtype, optional): 张量的数据类型，如果为None则根据fill_value推断
        device(str|int|Device,optional): 可选，指定张量所在的设备，可以是'cpu'、'cuda'、CUDA设备索引、device对象或None
        requires_grad (bool, optional): 是否需要计算梯度，默认为False
        
    Returns:
        TN: 填充了指定值的张量
        
    Examples:
        >>> full(3, 4, 5)  # 创建3x4的张量，所有元素为5
        >>> full((2, 3), 1.5)  # 创建2x3的张量，所有元素为1.5
    """
    shape_val = _validate_shape(shape)
    dt = get_default_dtype() if dtype is None else dtype
    dev = _get_device(device)
    if dev.type=='cpu':
        arr = np.full(shape_val,fill_value,dt)
    else:
        with cp.cuda.Device(dev.index):
            arr = cp.full(shape_val,fill_value,dt)

    return TN(arr, requires_grad=bool(requires_grad))

def full_like(tsr:TN,fill_value:Any,dtype:np.dtype|None = None,device:str|int|Device|None=None,requires_grad:bool|None = False)->TN:
    """
    创建一个与给定张量形状相同并填充了指定值的张量。
    
    返回一个与输入张量形状相同、所有元素都填充为指定值的新张量。
    
    Args:
        tsr (TN): 参考张量，用于确定输出张量的形状
        fill_value: 用于填充张量的值
        dtype (np.dtype, optional): 输出张量的数据类型，如果为None则使用参考张量的数据类型
        device(str|int|Device,optional): 可选，指定张量所在的设备，可以是'cpu'、'cuda'、CUDA设备索引、device对象或None
        requires_grad (bool, optional): 是否需要计算梯度，默认为False
        
    Returns:
        TN: 与参考张量形状相同并填充了指定值的张量
        
    Examples:
        >>> x = ones(3, 4)
        >>> y = full_like(x, 5)  # 创建与x形状相同的张量，所有元素为5
    """
    dt = tsr.dtype if dtype is None else dtype
    dev = tsr.device if device is None else _get_device(device)
    if dev.type=='cpu':
        arr = np.full_like(tsr._array,fill_value,dt)
    else:
        with cp.cuda.Device(dev.index):
            arr = cp.full_like(tsr._array,fill_value,dt)
    
    return TN(arr, requires_grad=bool(requires_grad))

def eye(n: int, m: int | None = None, dtype:np.dtype|None = None,device:str|int|Device|None=None,requires_grad:bool|None = False):
    """
    创建一个二维单位矩阵。
    
    返回一个n行m列的二维张量，对角线元素为1，其他元素为0。
    如果未指定m，则默认创建一个n×n的方阵。
    
    Args:
        n (int): 矩阵的行数
        m (int, optional): 矩阵的列数，如果为None则默认为n
        dtype (np.dtype, optional): 张量的数据类型，如果为None则使用默认数据类型
        device(str|int|Device,optional): 可选，指定张量所在的设备，可以是'cpu'、'cuda'、CUDA设备索引、device对象或None
        requires_grad (bool, optional): 是否需要计算梯度，默认为False
        
    Returns:
        TN: 二维单位矩阵张量
        
    Examples:
        >>> eye(3)  # 创建3×3的单位矩阵
        >>> eye(2, 4)  # 创建2×4的单位矩阵
    """
    dt = get_default_dtype() if dtype is None else dtype
    mm = n if m is None else m
    dev = _get_device(device)
    if dev.type=='cpu':
        arr = np.eye(n, mm, 0, dt)
    else:
        with cp.cuda.Device(dev.index):
            arr = cp.eye(n, mm, 0, dt)
    
    return TN(arr, requires_grad=bool(requires_grad))

# 定义函数类型别名
ForwardFunc = Callable[..., Any]  # 前向函数类型：输入任意数量的参数，返回任意类型
GradFunc = Callable[..., tuple[Optional[TN], ...]]  # 梯度函数类型：首参为结果张量，后续为前向输入，返回包含每个输入梯度的元组
BackwardFunc = Callable[[TN, int], TN]  # 反向函数类型：输入一个TN和一个int，返回一个TN
DecoratorFunc = Callable[[Callable[..., Any]], Callable[..., TN]]  # 修饰器类型：输入一个返回任意类型的函数，返回一个返回TN的函数

def track_grad(grad_func:GradFunc)->DecoratorFunc:
    """
    创建一个梯度跟踪修饰器，用于为函数添加自动微分支持。

    这个修饰器工厂接收一个梯度函数，返回一个修饰器，该修饰器可以将普通的张量运算函数
    转换为支持自动微分的函数。它会自动创建反向传播函数，并管理梯度计算图的构建。

    Args:
        grad_func (GradFunc): 梯度计算函数，签名固定为 ``grad_func(result, *args, **kwargs)``：
            第一个参数 result 是前向函数的返回张量，后续参数与前向函数保持相同的输入参数。
            返回值是一个元组，包含每个 TN 类型输入参数对应的对最终标量的梯度。
            元组内元素按 TN 参数出现顺序一一对应（仅包含 TN 参数，不包含非 TN 参数）；
            对于不需要梯度的 TN 参数，对应位置可为 None。

    Returns:
        DecoratorFunc: 一个修饰器函数，用于包装
            前向计算函数，使其支持自动微分

    梯度计算方式：
        grad_func 内部如何计算梯度完全由用户决定，result 及后续各参数都可能用到：
        - ``result.grad_value`` 是上游梯度（损失对该输出的梯度），按链式法则计算各输入
          梯度时必须使用它；
        - ``result`` 本身也可参与计算，例如直接用前向结果替代重新计算中间量，有时能显著
          减小计算量；
        - 其余 TN / 非 TN 参数同样可自由使用。
        因此梯度运算不受形式限制，矩阵乘 @、转置等任意运算均可直接使用。

    注意：
        若前向函数内部使用了可微算子，其返回张量已有的计算图会被本修饰器建立的
        计算图覆盖，因此 grad_func 必须给出完整梯度；若前向返回需要梯度的张量而所有
        形参都不需要梯度，原有梯度路径会被丢弃并发出警告。

    Examples:
        >>> # ========== 示例 1：单输入函数 ==========
        >>> # 定义导数函数：d/dx log(x) = 1/x
        >>> def _log_derivative(result: TN, x: TN) -> tuple[TN]:
        ...     return (result.grad_value * (1. / x.conj()),)
        >>>
        >>> @track_grad(_log_derivative)
        ... def mylog(x: TN) -> TN:
        ...     return tensor(np.log(x._array))
        >>>
        >>> x = tensor(2., requires_grad=True)
        >>> y = mylog(x)
        >>> y.backward()
        >>> print(f'x.grad = {x.grad}')  # 输出: x.grad = 0.5

        >>> # ========== 示例 2：多输入函数（关键字参数顺序无关） ==========
        >>> # track_grad 内部通过参数签名建立映射，因此无论关键字参数以何种顺序传入，
        >>> # 梯度都能正确匹配到对应的张量。
        >>> def _mul_derivative(result: TN, x: TN, y: TN) -> tuple[TN, TN]:
        ...     return (result.grad_value * y.conj(), result.grad_value * x.conj())
        >>>
        >>> @track_grad(_mul_derivative)
        ... def mymul(x: TN, y: TN) -> TN:
        ...     return tensor(x._array * y._array)
        >>>
        >>> x = tensor(2., requires_grad=True)
        >>> y = tensor(3., requires_grad=True)
        >>> # 关键字参数顺序与函数签名不一致：签名是 (x, y)，调用时写成 (y, x)
        >>> z = mymul(y=y, x=x)
        >>> z.backward()
        >>> print(f'x.grad = {x.grad}')  # 输出: x.grad = 3.0 (y 的值)
        >>> print(f'y.grad = {y.grad}')  # 输出: y.grad = 2.0 (x 的值)

        >>> # ========== 示例 3：混合 TN 与非 TN 参数 ==========
        >>> # grad_func 的返回值只包含 TN 类型参数的梯度，非 TN 参数（如标量）被忽略。
        >>> def _pow_derivative(result: TN, x: TN, exponent: float) -> tuple[TN]:
        ...     return (result.grad_value * exponent * (x ** (exponent - 1)).conj(),)
        >>>
        >>> @track_grad(_pow_derivative)
        ... def mypow(x: TN, exponent: float) -> TN:
        ...     return tensor(x._array ** exponent)
        >>>
        >>> x = tensor(2., requires_grad=True)
        >>> y = mypow(x, 3.0)
        >>> y.backward()
        >>> print(f'x.grad = {x.grad}')  # 输出: x.grad = 12.0 (3 * 2^2)

        >>> # ========== 示例 4：部分输入不需要梯度 ==========
        >>> # 不需要梯度的输入不会出现在 fromvars 中，对应的梯度位置可为 None。
        >>> def _add_derivative(result: TN, x: TN, y: TN) -> tuple[TN, TN]:
        ...     return (result.grad_value, result.grad_value)
        >>>
        >>> @track_grad(_add_derivative)
        ... def myadd(x: TN, y: TN) -> TN:
        ...     return tensor(x._array + y._array)
        >>>
        >>> x = tensor(2., requires_grad=True)
        >>> y = tensor(3., requires_grad=False)  # 不需要梯度
        >>> z = myadd(x, y)
        >>> z.backward()
        >>> print(f'x.grad = {x.grad}')  # 输出: x.grad = 1.0
        >>> print(f'y.grad = {y.grad}')  # 输出: y.grad = None

        >>> # ========== 示例 5：矩阵乘法（上游梯度需参与矩阵运算） ==========
        >>> # 矩阵乘反向：dL/dx = dL/dz @ y^T，dL/dy = x^T @ dL/dz。
        >>> # grad_func 通过 result.grad_value 直接完成矩阵乘法，无需求助逐元素乘法。
        >>> def _matmul_derivative(result: TN, x: TN, y: TN):
        ...     g = result.grad_value
        ...     return (g @ y.detach().T, x.detach().T @ g)
        >>>
        >>> @track_grad(_matmul_derivative)
        ... def mymatmul(x: TN, y: TN) -> TN:
        ...     return tensor(x._array @ y._array)
        >>>
        >>> x = tensor([[1., 2.], [3., 4.]], requires_grad=True)
        >>> y = tensor([[5., 6.], [7., 8.]], requires_grad=True)
        >>> z = mymatmul(x, y)
        >>> z.sum().backward()

    工作原理：
        1. 包装后的函数先执行前向计算，并根据“梯度模式是否开启、任一输入是否需要梯度”
           决定结果张量是否参与梯度跟踪
        2. 若需要跟踪，则按 grad_func 的签名顺序展平本次实参，并仅为 requires_grad=True
           的 TN 输入登记图节点（fromvars / gradfuncs）
        3. 反向传播时，每个需要梯度的输入各触发一次 grad_func 调用，按索引取出
           对应输入的梯度返回，因此 grad_func 应保持无副作用；链式法则完全由
           grad_func 借助 result 自行完成
    """
    # grad_func 在装饰阶段固定，预先解析签名，避免每次前向调用重复解析
    grad_sig = inspect.signature(grad_func)
    grad_params = grad_sig.parameters
    if not grad_params:
        raise TypeError("grad_func must accept 'result' as its first parameter")
    first_param = next(iter(grad_params.values()))
    if first_param.kind is inspect.Parameter.VAR_KEYWORD:
        raise TypeError("grad_func's first parameter cannot be **kwargs")
    first_param_name = first_param.name

    def decorator(forward_func:ForwardFunc)->ForwardFunc:
        def wrapper(*xs, **kwargs)->TN:
            ret_val = forward_func(*xs, **kwargs)

            # 前向返回值可能不是 TN（如 numpy 数组），此时用首个 TN 输入的 device 转换
            all_params = list(xs) + list(kwargs.values())
            tn_params = [p for p in all_params if isinstance(p, TN)]
            # 梯度模式关闭或所有输入都不需要梯度时，结果张量不参与计算图
            track = is_grad_enabled() and any(p.requires_grad for p in tn_params)
            if isinstance(ret_val, TN):
                ret = ret_val
                # 前向若使用了可微算子，其内部计算图会被下面的 fromvars/gradfuncs 覆盖
                if is_grad_enabled() and ret.requires_grad and not track:
                    warnings.warn(
                        f"track_grad: forward result of "
                        f"'{getattr(forward_func, '__name__', forward_func)}' requires grad, "
                        f"but none of its TN arguments do; the existing gradient path is discarded"
                    )
            else:
                ret = as_tensor(ret_val, device=tn_params[0].device if tn_params else None)

            ret.requires_grad = track

            if track:
                # grad_func 返回的梯度元组按“签名参数顺序”排列（result 除外），而调用时
                # 关键字参数顺序可能不同。用 sig.bind 建立“参数名 → 实参值”的映射，
                # 再按签名顺序展平，使梯度元组与 TN 参数一一对应。
                bound = grad_sig.bind(ret, *xs, **kwargs)
                bound.apply_defaults()
                ordered_args = []
                for name, value in bound.arguments.items():
                    kind = grad_params[name].kind
                    if name == first_param_name:
                        # 首参 result 本身没有梯度；裸 *args 签名中 args[0] 即 result
                        if kind is inspect.Parameter.VAR_POSITIONAL:
                            ordered_args.extend(value[1:])
                        continue
                    if kind is inspect.Parameter.VAR_POSITIONAL:
                        ordered_args.extend(value)            # *args 展平为多个输入
                    elif kind is inspect.Parameter.VAR_KEYWORD:
                        ordered_args.extend(value.values())   # **kwargs 展平为多个输入
                    else:
                        ordered_args.append(value)

                # 每个需要梯度的输入对应一个反向函数，各自调用一次 grad_func。
                # 注意：多输入函数中 grad_func 会被调用多次，因此其实现应保持无副作用。
                def _make_backward_func(grad_index: int):
                    # backward 机制要求签名为 (result_tensor, index)，但取哪个梯度
                    # 已由闭包 grad_index 固定，因此不使用 index 参数。
                    def backward_func(result_tensor: TN, _index: int) -> TN:
                        grads = grad_func(result_tensor, *xs, **kwargs)
                        # 兼容单输入函数直接返回单个梯度（非元组）的写法
                        if not isinstance(grads, tuple):
                            grads = (grads,)
                        if grad_index >= len(grads):
                            raise ValueError(
                                f"grad_func returned {len(grads)} gradients, "
                                f"but TN parameter at index {grad_index} needs one"
                            )
                        grad = grads[grad_index]
                        if grad is None:
                            raise ValueError(
                                f"Gradient for TN parameter at index {grad_index} is None, "
                                f"but requires_grad is True"
                            )
                        return grad
                    return backward_func

                # 按实参顺序为所有 TN 参数编号，并只为 requires_grad=True 的参数登记图节点。
                # grad_index 对“所有 TN 参数”递增（与 grad_func 返回的梯度元组对齐），
                # 而 fromvars/gradfuncs 只含需要梯度的参数，两者序号不可混用。
                fromvars = []
                gradfuncs = []
                grad_index = 0
                for param in ordered_args:
                    if isinstance(param, TN):
                        if param.requires_grad:
                            fromvars.append(param)
                            gradfuncs.append(_make_backward_func(grad_index))
                        grad_index += 1

                ret.fromvars = tuple(fromvars)
                ret.gradfuncs = tuple(gradfuncs)
            return ret
        return wrapper
    return decorator

def broadcast_to(input: TN, size: tuple[int, ...]) -> TN:
    """
    将输入张量广播到指定的形状。
    
    返回一个原始张量的视图，广播到新的形状。原始张量和新张量共享相同的数据，
    但新张量的形状可能不同。广播遵循NumPy的广播规则。
    
    Args:
        input (TN): 要广播的输入张量
        size (tuple[int, ...]): 目标形状，必须与输入形状兼容
        
    Returns:
        TN: 广播后的张量
        
    Raises:
        TypeError: 如果输入不是TN张量或size不是元组/列表
        RuntimeError: 如果广播失败
        
    Examples:
        >>> x = tensor([[1, 2, 3]])  # 形状(1, 3)
        >>> y = broadcast_to(x, (4, 3))  # 形状(4, 3)，每行都是[1, 2, 3]
        >>> z = tensor([1, 2, 3])  # 形状(3,)
        >>> w = broadcast_to(z, (2, 3))  # 形状(2, 3)，每行都是[1, 2, 3]
    """
    # 验证输入类型
    if not isinstance(input, TN):
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input)}")
    
    # 确保size是元组（支持列表输入）
    size = tuple(size) if isinstance(size, list) else size
    
    # _expand 内部会检查形状相等并返回自身，无需重复检查
    # _expand 内部使用 numpy/cupy 的 broadcast_to，返回视图不复制数据
    try:
        return input._expand(size)
    except ValueError as e:
        raise RuntimeError(f"Failed to broadcast tensor from shape {input.shape} to shape {size}: {e}")

def broadcast_tensors(*tensors: TN) -> tuple[TN, ...]:
    """
    将多个张量广播到相同的形状
    
    按照NumPy的广播规则，将多个输入张量广播到相同的形状。
    如果张量的形状已经兼容，则返回原始张量；否则返回广播后的张量。
    
    Args:
        *tensors: 要广播的张量，至少需要两个张量
        type *tensors: TN
        
    Returns:
        广播后的张量元组，所有张量具有相同的形状
        rtype: tuple[TN, ...]
        
    Raises:
        TypeError: 如果输入不是TN张量
        RuntimeError: 如果广播失败
        
    Examples:
        >>> a = tensor([1, 2, 3])  # 形状(3,)
        >>> b = tensor([[4], [5]])  # 形状(2, 1)
        >>> a_broadcast, b_broadcast = broadcast_tensors(a, b)
        >>> a_broadcast.shape  # (2, 3)
        >>> b_broadcast.shape  # (2, 3)
        
        >>> c = tensor([1, 2, 3])  # 形状(3,)
        >>> d = tensor([[4], [5]])  # 形状(2, 1)
        >>> e = tensor([[[6]]])  # 形状(1, 1, 1)
        >>> c_broadcast, d_broadcast, e_broadcast = broadcast_tensors(c, d, e)
        >>> c_broadcast.shape  # (1, 2, 3)
        >>> d_broadcast.shape  # (1, 2, 3)
        >>> e_broadcast.shape  # (1, 2, 3)
    """
    if len(tensors) < 2:
        raise ValueError(f"broadcast_tensors requires at least 2 tensors, got {len(tensors)}")
    
    # 验证所有输入都是TN类型
    for i, t in enumerate(tensors):
        if not isinstance(t, TN):
            raise TypeError(f"Expected tensor {i} to be TN type, got {type(t)}")
    
    # 使用对应的数组库计算广播后的形状
    # 获取第一个张量的数组库（所有张量应该在同一个库中）
    arrlib = tensors[0]._get_array_lib()
    
    try:
        # 使用对应数组库的 broadcast 函数获取广播后的形状
        broadcast_obj = arrlib.broadcast(*[t._array for t in tensors])
        broadcast_shape = broadcast_obj.shape
    except ValueError as e:
        # 广播失败，抛出更清晰的错误信息
        raise RuntimeError(f"Cannot broadcast tensors with incompatible shapes: {[t.shape for t in tensors]}") from e
    
    # 广播所有张量
    broadcasted_tensors = []
    for t in tensors:
        if t.shape == broadcast_shape:
            # 形状已经匹配，直接返回
            broadcasted_tensors.append(t)
        else:
            # 需要广播
            broadcasted_tensors.append(broadcast_to(t, broadcast_shape))
    
    return tuple(broadcasted_tensors)

def _convert_TNindex_to_numpy(index):
    """
    将 TN 索引转换为 NumPy 兼容的索引格式
    
    此函数是索引处理的单一入口，负责：
    1. 将 TN 张量索引转换为底层数组
    2. 将 None 转换为 (None,) 以统一处理 np.newaxis
    
    Args:
        index: 输入索引，可以是 TN、slice、int、tuple、list、np.ndarray 等
    
    Returns:
        转换后的索引
    """
    if isinstance(index, TN):
        index_val = index._array
    elif index is None:
        # 将 None 转换为 (None,) 以便在反向传播中统一处理
        # 这样可以区分 "没有索引信息" (None) 和 "None 索引" ((None,))
        # 同时 arr[None] 和 arr[(None,)] 在 NumPy 中行为相同
        index_val = (None,)
    elif isinstance(index, tuple):
        index_list = []
        for idx in index:
            if isinstance(idx, TN):
                index_list.append(idx._array)
            else:
                index_list.append(idx)
        index_val = tuple(index_list)
    else:
        index_val = index

    return index_val


def _is_boolean_index(idx):
    """检查索引是否为布尔类型索引"""
    try:
        if hasattr(idx, 'dtype') and idx.dtype == bool:
            return True
        elif isinstance(idx, np.ndarray) and idx.dtype == bool:
            return True
        elif isinstance(idx, tuple):
            return any(_is_boolean_index(i) for i in idx)
        return False
    except Exception:
        return False


# 全量索引（index 为 ()）时使用的向量化原地操作。
# 与 ufunc.at 的分工：
# - oper_func（ufunc.at）面向任意索引，必须保证重复索引“累加”的语义，
#   但在全量索引下会退化成逐元素循环（实测慢两个数量级）；
# - 全量索引不存在重复元素，可直接使用原地运算符，语义与 ufunc.at 等价。
# 传入的右值已由 _adjust_right_val 转成 self 的 dtype 并完成形状对齐，
# 因此这里不会引入额外的类型转换差异，cupy 数组同样适用。
def _iadd_full(numpy_arr, right):
    numpy_arr += right
    return numpy_arr


def _isub_full(numpy_arr, right):
    numpy_arr -= right
    return numpy_arr


def _imul_full(numpy_arr, right):
    numpy_arr *= right
    return numpy_arr


def _idiv_full(numpy_arr, right):
    numpy_arr /= right
    return numpy_arr


def _ipow_full(numpy_arr, right):
    numpy_arr **= right
    return numpy_arr


def _ifloordiv_full(numpy_arr, right):
    numpy_arr //= right
    return numpy_arr


def _imod_full(numpy_arr, right):
    numpy_arr %= right
    return numpy_arr


def _iset_full(numpy_arr, right):
    numpy_arr[...] = right
    return numpy_arr


# 反向传播需要“左值”的原地操作回调。
# mul/div/pow 的反向要用操作前的左值，因此必须在覆盖数据之前备份 target_copy；
# add/sub/set 的反向只用索引和梯度，不需要备份，可省去一次整块内存拷贝。
# 用回调函数名而不是函数对象做判断，避免这些函数定义在本类之后造成前向引用问题。
_LEFT_VALUE_BACKWARD_NAMES = {
    '_mulat_inplace_backward',
    '_divat_inplace_backward',
    '_powat_inplace_backward',
    '_modat_inplace_backward',
}


# 原地操作梯度回调函数名 -> 对应的原地操作方法名，__setitem__ 据此重放副本上的操作。
# 只收录复合赋值运算（+=、-=、*=、/=、**=、%=）；setat_ 不在此列——它本身是覆盖写，
# 直接按数据 setat 即可得到相同结果，无需重放。
_INPLACE_AT_METHODS = {
    '_addat_inplace_backward': 'addat_',
    '_subat_inplace_backward': 'subat_',
    '_mulat_inplace_backward': 'mulat_',
    '_divat_inplace_backward': 'divat_',
    '_powat_inplace_backward': 'powat_',
    '_modat_inplace_backward': 'modat_',
}


def _graph_contains(var, target):
    """判断 target 是否出现在 var 的计算图来源中（含 var 自身）。

    仅沿 requires_grad 为 True 的节点遍历：不需要梯度的节点不会继续向上游传播，
    其祖先也就不参与判断，同时能剪掉大部分无关子图。
    用 id 记录访问过的节点，避免共享子图重复遍历和潜在环路导致的死循环。

    只在 __setitem__ 无法重放的兜底路径上调用，属于低频操作，不做额外优化。
    """
    stack = [var]
    visited = set()
    while stack:
        item = stack.pop()
        if item is target:
            return True
        if id(item) in visited:
            continue
        visited.add(id(item))
        if not item.requires_grad:
            continue
        stack.extend(item.fromvars)
    return False


def _indexes_are_equal(idx1, idx2):
    """判断两个索引是否表示同一选择，用于确认视图/副本索引与回写索引是否一致。

    只做保守判断：
    - 布尔索引直接视为不同，避免布尔数组逐元素比较的歧义；
    - 元组索引逐项比较，数组项用 np.array_equal，任一项不同即返回 False；
    - 无法比较的类型返回 False，让调用方走更安全的赋值路径。
    """
    try:
        # 跳过布尔索引比较
        if _is_boolean_index(idx1) or _is_boolean_index(idx2):
            return False

        if isinstance(idx1, tuple) and isinstance(idx2, tuple):
            # 元组索引比较
            if len(idx1) != len(idx2):
                return False
            for a, b in zip(idx1, idx2):
                try:
                    # 数组类型用逐元素比较，注意不能提前 return，需继续比较剩余元素
                    if hasattr(a, 'shape') or hasattr(b, 'shape'):
                        if not np.array_equal(np.asarray(a), np.asarray(b)):
                            return False
                        continue
                    # 简单类型直接比较
                    if a != b:
                        return False
                except Exception:
                    return False
            return True
        elif isinstance(idx1, (int, slice, type(None))) and isinstance(idx2, (int, slice, type(None))):
            # 单一索引比较
            return idx1 == idx2
        else:
            # 其他类型尝试转换为数组比较
            try:
                return bool(np.array_equal(np.asarray(idx1), np.asarray(idx2)))
            except Exception:
                return False
    except Exception:
        return False


def _squeeze_backward(result_tensor:TN, i: int) -> TN:
    dim = result_tensor.parms[i]
    grad = result_tensor.grad_value.unsqueeze(dim)
    return grad

def _unsqueeze_backward(result_tensor:TN, i: int) -> TN:
    dim = result_tensor.parms[i]
    grad = result_tensor.grad_value._squeeze(dim)
    return grad

def _reshape_backward(result_tensor: TN, i: int) -> TN:
    original_shape, _ = result_tensor.parms[i]
    grad = result_tensor.grad_value._reshape(original_shape)
    return grad

def _expand_backward(result_tensor: TN, i: int) -> TN:
    """
    expand操作的反向传播函数。
    """
    original_shape, _ = result_tensor.parms[i]
    grad_value = result_tensor.grad_value
    
    # 如果原始张量是标量，需要将梯度求和为标量
    if original_shape == ():
        # 使用Riemann的sum函数而非numpy函数，保留计算图
        return grad_value.sum()
    
    orig_ndim = len(original_shape)
    grad_ndim = grad_value.ndim
    
    # 确定需要求和的维度
    sum_dims: list[int] = []
    
    if orig_ndim < grad_ndim:
        # 对于维度数量增加的情况，前面的维度都需要求和
        sum_dims.extend(range(grad_ndim - orig_ndim))
        # 检查剩余维度
        for i, (orig_dim, expanded_dim) in enumerate(zip(
                original_shape, 
                grad_value.shape[grad_ndim - orig_ndim:])):
            if orig_dim == 1 and expanded_dim > 1:
                sum_dims.append(i + (grad_ndim - orig_ndim))
    else:
        # 维度数量相同的情况
        for dim, (orig_size, expanded_size) in enumerate(zip(original_shape, grad_value.shape)):
            if orig_size == 1 and expanded_size > 1:
                sum_dims.append(dim)
    
    # 如果有需要求和的维度，执行求和操作
    if sum_dims:
        # 使用Riemann的sum函数，保留计算图
        summed_grad = grad_value
        # 对维度从高到低求和，避免维度索引变化问题
        for dim in sorted(sum_dims, reverse=True):
            summed_grad = summed_grad.sum(dim=dim, keepdim=True)
        # 使用Riemann的reshape函数，保留计算图
        grad = summed_grad._reshape(original_shape)
    else:
        # 如果没有维度被扩展，直接使用Riemann的reshape函数
        grad = grad_value._reshape(original_shape)
    
    return grad

def _setat_backward_left(result_tensor:TN, i:int)->TN:
    index = result_tensor.parms[0]
    grad = result_tensor.grad_value.setat(index,0.)
    return grad

def _setat_backward_right(result_tensor:TN, i:int)->TN:
    index = result_tensor.parms[0]

    # result_tensor就是原地赋值时的self
    # 将self的grad_value按索引取值就是要传递给所依赖的右值的梯度，索引取值结果是一个新张量
    right_var_grad = result_tensor.grad_value[index]
    return right_var_grad

def _setat_inplace_backward(result_tensor:TN, i:int)->TN: 
    index = result_tensor.parms[i][1]

    # result_tensor就是原地赋值时的self
    # 将self的grad_value按索引取值就是要传递给所依赖的右值的梯度，索引取值结果是一个新张量
    right_var_grad = result_tensor.grad_value[index]
    # 由于self的在原地索引赋值时index位置已被右值覆盖，
    # 所以要将self的grad_value中索引位置的梯度设为0
    clone_self_grad = result_tensor.grad_value.setat(index,0.)
    result_tensor.grad_value = clone_self_grad
    
    return right_var_grad

def _addat_backward_left(result_tensor:TN, i:int)->TN:
    grad = result_tensor.grad_value
    return grad

def _addat_backward_right(result_tensor: TN, i: int) -> TN:
    index = result_tensor.parms[0]

    # result_tensor就是原地赋值时的self
    # 将self的grad_value按索引取值就是要传递给所依赖的右值的梯度，索引取值结果是一个新张量
    right_var_grad = result_tensor.grad_value[index]    
    return right_var_grad

def _addat_inplace_backward(result_tensor:TN, i:int)->TN:    
    index = result_tensor.parms[i][1]

    # result_tensor就是原地赋值时的self
    # 将self的grad_value按索引取值就是要传递给所依赖的右值的梯度，索引取值结果是一个新张量
    right_var_grad = result_tensor.grad_value[index]
    return right_var_grad

def _subat_backward_left(result_tensor:TN, i:int)->TN:
    grad = result_tensor.grad_value
    return grad

def _subat_backward_right(result_tensor: TN, i: int) -> TN:
    index = result_tensor.parms[0]

    # result_tensor就是原地赋值时的self
    # 将self的grad_value按索引取值就是要传递给所依赖的右值的梯度，索引取值结果是一个新张量
    right_var_grad = result_tensor.grad_value[index]    
    return right_var_grad

def _subat_inplace_backward(result_tensor:TN, i:int)->TN:    
    index = result_tensor.parms[i][1]

    # result_tensor就是原地赋值时的self
    # 将self的grad_value按索引取值就是要传递给所依赖的右值的梯度，索引取值结果是一个新张量
    right_var_grad = -result_tensor.grad_value[index]
    return right_var_grad

def _mulat_backward_left(result_tensor:TN, i:int)->TN:
    left_var = result_tensor.fromvars[0]
    right_var = result_tensor.fromvars[1]
    index = result_tensor.parms[0]

    left_var_grad = result_tensor.grad_value[index] * right_var.conj()
    grad = result_tensor.grad_value.setat(index,left_var_grad)
    return grad

def _mulat_backward_right(result_tensor: TN, i: int) -> TN:
    left_var = result_tensor.fromvars[0]
    right_var = result_tensor.fromvars[1]
    index = result_tensor.parms[0]

    right_var_grad = result_tensor.grad_value[index] * left_var[index].conj()
    return right_var_grad

def _mulat_inplace_backward(result_tensor: TN, i: int) -> TN:
    right_var = result_tensor.fromvars[i]
    left_var = result_tensor.parms[i][0]    
    index = result_tensor.parms[i][1]

    result_grad = result_tensor.grad_value[index]
    right_var_grad = result_grad * left_var.conj()

    # 由于self的在原地索引赋值时index位置已被乘积值覆盖，
    # 所以要将self的grad_value中索引位置的梯度设为左值的梯度
    left_var_grad = result_grad * right_var.conj()
    clone_self_grad = result_tensor.grad_value.setat(index,left_var_grad)    
    result_tensor.grad_value = clone_self_grad

    return right_var_grad

def _neg_inplace_data(array, index)->None:
    """neg_ 的数据改写函数：对指定索引位置的元素取负"""
    array[index] = -array[index]

def _abs_inplace_data(array, index)->None:
    """abs_ 的数据改写函数：对指定索引位置的元素取绝对值。

    不能用内置 abs()：本模块的 abs(TN) 会遮蔽内置名，数组会被包成 TN；
    ndarray/numpy 标量/cupy 数组都实现了 __abs__，可直接调用以保持设备无关。
    """
    array[index] = array[index].__abs__()

def _placeholder_zero_grad(result_tensor: TN, i: int) -> TN:
    """为不传播梯度的一元原地运算返回0标量梯度，供占位变量接收"""
    placeholder = result_tensor.fromvars[i]
    return TN(placeholder._get_array_lib().zeros((), dtype=placeholder.dtype))

def _negat_inplace_backward(result_tensor: TN, i: int) -> TN:
    """neg_ 的反向：把流向自身历史算子的梯度取负（d(-x)/dx = -1）"""
    index = result_tensor.parms[i][1]

    result_grad = result_tensor.grad_value[index]
    result_tensor.grad_value = result_tensor.grad_value.setat(index, -result_grad)

    return _placeholder_zero_grad(result_tensor, i)

def _absat_inplace_backward(result_tensor: TN, i: int) -> TN:
    """abs_ 的反向：d|x|/dx = sign(x)，符号取自操作前的左值副本"""
    left_copy = result_tensor.parms[i][0]
    index = result_tensor.parms[i][1]

    result_grad = result_tensor.grad_value[index]
    left_grad = result_grad * left_copy._get_array_lib().sign(left_copy._array)
    result_tensor.grad_value = result_tensor.grad_value.setat(index, left_grad)

    return _placeholder_zero_grad(result_tensor, i)

def _divat_backward_left(result_tensor:TN, i:int)->TN:
    left_var = result_tensor.fromvars[0]
    right_var = result_tensor.fromvars[1]
    index = result_tensor.parms[0]

    left_var_grad = result_tensor.grad_value[index] / right_var.conj()
    grad = result_tensor.grad_value.setat(index,left_var_grad)
    return grad

def _divat_backward_right(result_tensor: TN, i: int) -> TN:
    left_var = result_tensor.fromvars[0]
    right_var = result_tensor.fromvars[1]
    index = result_tensor.parms[0]

    right_var_grad = result_tensor.grad_value[index] * (-result_tensor[index]/right_var).conj()
    return right_var_grad

def _divat_inplace_backward(result_tensor: TN, i: int) -> TN:
    right_var = result_tensor.fromvars[i]
    left_var = result_tensor.parms[i][0]    
    index = result_tensor.parms[i][1]

    result_grad = result_tensor.grad_value[index]
    right_var_grad = result_grad * (-result_tensor[index]/right_var).conj()

    # 由于self的在原地索引赋值时index位置已被覆盖，
    # 所以要将self的grad_value中索引位置的梯度设为左值的梯度
    left_var_grad = result_grad / right_var.conj()
    clone_self_grad = result_tensor.grad_value.setat(index,left_var_grad)    
    result_tensor.grad_value = clone_self_grad

    return right_var_grad

def _modat_backward_left(result_tensor:TN, i:int)->TN:
    # d(left % right)/d(left) = 1，结果梯度即为左值梯度
    return result_tensor.grad_value

def _modat_backward_right(result_tensor: TN, i: int) -> TN:
    left_var = result_tensor.fromvars[0]
    right_var = result_tensor.fromvars[1]
    index = result_tensor.parms[0]

    # d(left % right)/d(right) = -floor(left/right)
    right_var_grad = -result_tensor.grad_value[index] * floor(left_var[index] / right_var)
    return right_var_grad

def _modat_inplace_backward(result_tensor: TN, i: int) -> TN:
    right_var = result_tensor.fromvars[i]
    left_var = result_tensor.parms[i][0]    
    index = result_tensor.parms[i][1]

    result_grad = result_tensor.grad_value[index]
    right_var_grad = -result_grad * floor(left_var / right_var)

    # 由于self在原地索引赋值时index位置已被取余结果覆盖，
    # 所以要将self的grad_value中索引位置的梯度设为左值的梯度（导数恒为1）
    clone_self_grad = result_tensor.grad_value.setat(index, result_grad)    
    result_tensor.grad_value = clone_self_grad

    return right_var_grad

def _powat_backward_left(result_tensor:TN, i:int)->TN:
    left_var = result_tensor.fromvars[0]
    right_var = result_tensor.fromvars[1]
    index = result_tensor.parms[0]

    result_grad = result_tensor.grad_value[index]
    left_var_grad = result_grad * (right_var * (left_var[index] ** (right_var - 1.))).conj()
    grad = result_tensor.grad_value.setat(index,left_var_grad)
    return grad

def _powat_backward_right(result_tensor: TN, i: int) -> TN:
    left_var = result_tensor.fromvars[0]
    right_var = result_tensor.fromvars[1]
    index = result_tensor.parms[0]

    result_grad = result_tensor.grad_value[index]
    right_var_grad = result_grad * result_tensor[index].conj() * log(left_var[index].conj())
    return right_var_grad

def _powat_inplace_backward(result_tensor: TN, i: int) -> TN:
    right_var = result_tensor.fromvars[i]
    left_var = result_tensor.parms[i][0]    
    index = result_tensor.parms[i][1]

    result_grad = result_tensor.grad_value[index]
    right_var_grad = result_grad * result_tensor[index].conj() * log(left_var.conj())

    # 由于self的在原地索引赋值时index位置已被乘积值覆盖，
    # 所以要将self的grad_value中索引位置的梯度设为左值的梯度
    left_var_grad = result_grad * (right_var * (left_var ** (right_var - 1.))).conj()
    clone_self_grad = result_tensor.grad_value.setat(index,left_var_grad)    
    result_tensor.grad_value = clone_self_grad

    return right_var_grad

def _flip_backward(result_tensor:TN, i:int)->TN:
    """flip操作的反向传播梯度计算函数。
    
    Args:
        result_tensor: 前向传播的结果张量
        i: 输入张量在fromvars中的索引
        
    Returns:
        TN: 计算得到的梯度张量
    """
    # flip操作的梯度计算很简单，只需要对梯度值执行相同的flip操作即可
    grad_value = result_tensor.grad_value
    dims = result_tensor.parms[0]  # 获取前向传播时使用的dims参数
    return flip(grad_value,dims)

def flip(input: TN, dims:list[int]|tuple[int,...]) -> TN:
    """沿指定维度翻转张量的顺序。
    
    Reverse the order of an n-D tensor along given axis in dims.
    
    Note:
        Unlike NumPy's `np.flip`, this implementation makes a copy of the input tensor's data.
    
    Args:
        input (TN): 输入张量
        dims (list or tuple): 需要翻转的维度
        
    Returns:
        TN: 翻转后的新张量
        
    Examples:
        >>> import riemann as rm
        >>> x = rm.arange(8).reshape(2, 2, 2)
        >>> x
        TN([[[0 1]
             [2 3]]
            
            [[4 5]
             [6 7]]], requires_grad=False)
        >>> rm.flip(x, [0, 1])
        TN([[[6 7]
             [4 5]]
            
            [[2 3]
             [0 1]]], requires_grad=False)
    """
    # 确保dims是列表或元组
    if not isinstance(dims, (list, tuple)):
        dims = (dims,)
    
    # 使用numpy.flip或cupy.flip执行翻转操作
    arrlib = input._get_array_lib()
    data = arrlib.flip(input._array, tuple(dims)).copy()
    
    # 创建新的张量
    result = TN(data, requires_grad = (is_grad_enabled() and input.requires_grad))
    
    # 如果需要计算梯度，设置计算图信息
    if result.requires_grad:
        result.fromvars = (input,)
        result.parms = (dims,)
        result.gradfuncs = (_flip_backward,)
    
    return result

def _get_broadcast_axis(big_shape,small_shape):
    """计算 small_shape 广播为 big_shape 时发生广播的轴索引。

    广播轴有两个来源：
    1. 前导维：small_shape 右对齐后缺失的前 dims 个轴；
    2. 扩展维：对应轴上 small_shape 为 1 而 big_shape 大于 1（1 -> n 广播）。
    形状相同的轴（含 1 -> 1）不计入。

    参数:
        big_shape: 广播结果的目标形状，维数须不少于 small_shape
        small_shape: 被广播扩展的小形状

    返回:
        广播轴索引元组（升序），无广播时为空元组 ()

    示例:
        >>> _get_broadcast_axis((2, 2, 2), (2,))
        (0, 1)
        >>> _get_broadcast_axis((2, 2, 1), (2, 1))
        (0,)
        >>> _get_broadcast_axis((2, 2, 2), (1, 2))
        (0, 1)
        >>> _get_broadcast_axis((2, 2, 1), ())
        (0, 1, 2)

    异常:
        ValueError: 两形状不兼容（无法广播）时抛出
    """
    len_of_bigshape = len(big_shape)
    dims = len_of_bigshape - len(small_shape)

    # 前 dims 个轴是 small_shape 缺失的前导维，恒为广播轴
    prefix = range(dims)

    # 右对齐逐轴比较剩余轴：big_shape 是广播结果，故各轴只能满足
    # big == small（含 1 -> 1，不计入）或 small == 1（1 -> n，计入）
    suffix = []
    for i in range(dims, len_of_bigshape):
        j = i - dims
        if big_shape[i] != small_shape[j]:
            if small_shape[j] == 1:
                suffix.append(i)
            else:
                raise ValueError(big_shape, small_shape)

    if suffix:
        return tuple(prefix) + tuple(suffix)
    return tuple(prefix)

def _sum_backward(result_tensor:TN,i:int)->TN:
    dim, keepdims = result_tensor.parms[i]
    x_shape = result_tensor.fromvars[i].shape

    # 如果sum计算时结果张量做过维度精简，需要将缩减的维度恢复
    # 然后expand到输入形状，sum的梯度就是直接将输出梯度传播到每个输入元素
    if keepdims == False:
        grad = result_tensor.grad_value.unsqueeze(dim)._expand(x_shape)
    else:
        grad = result_tensor.grad_value._expand(x_shape)
    
    return grad

def sum(x:TN, dim:int|tuple|None=None, keepdim:bool=False)->TN:
    """张量求和函数（内部实现）"""
    if not isinstance(x,TN):
        raise SyntaxError('x must be a tensor')
    
    # dim==()时，numpy.sum返回原数组，但sum函数求总和，需要将dim设置为None
    if dim == ():
        dim = None
    
    sumvalue = x._array.sum(axis=dim, keepdims=keepdim)
    
    # 创建与x在相同设备上的张量
    ret=TN(sumvalue, requires_grad = (is_grad_enabled() and x.requires_grad))
    
    if ret.requires_grad:
        ret.fromvars=(x,)
        ret.parms=((dim,keepdim),)
        ret.gradfuncs=(_sum_backward,)
    
    return ret

def _is_array_type(obj):
    """检查对象是否是数组类型（numpy或cupy）"""
    if isinstance(obj, np.ndarray):
        return True
    if cp is not None and isinstance(obj, cp.ndarray):
        return True
    return False

def _cumsum_backward(result_tensor: TN, i: int) -> TN:
    """
    cumsum函数的反向传播实现。
    累积和的梯度是反向累积和。
    """
    dim, dtype_param = result_tensor.parms[i]  # 直接使用转换后的dtype
    
    # 获取梯度值，使用Riemann的张量操作保持计算图
    grad_value = result_tensor.grad_value
    
    # 反向累积和：从后向前累积
    # 例如：[g1, g2, g3, g4] -> [g1+g2+g3+g4, g2+g3+g4, g3+g4, g4]
    reversed_grad = flip(grad_value, [dim])
    reversed_cumsum = cumsum(reversed_grad, dim=dim)  # 使用Riemann的cumsum函数，不是递归调用
    input_grad = flip(reversed_cumsum, [dim])
    
    # 如果指定了dtype，转换回原始数据类型
    if dtype_param != input_grad.dtype:
        input_grad = input_grad.type(dtype_param)
    
    return input_grad


def _validate_out_tensor(out, result: TN, inputs, name: str, label: str = "out", allow_grad: bool = False) -> None:
    """校验单个 out 张量是否与计算结果兼容（语义对齐 torch 的 out= 参数）

    参数:
        out: 调用方传入的输出张量
        result: 函数的计算结果
        inputs: 参与本次计算的输入序列，用于判断是否需要梯度
        name: 函数名，用于报错信息
        label: out 的显示名称，多输出场景下形如 out[0]
        allow_grad: 输入可带梯度时是否仍允许 out（仅 norm/cond 等 torch 支持的复合算子为 True）
    """
    if not isinstance(out, TN):
        raise TypeError(f"{name}(): {label} must be a tensor of type TN, but got {type(out)}")

    # out 自身需要梯度时无法安全地原地写入（与 torch 的叶节点原地操作限制一致）
    if out.requires_grad:
        if allow_grad:
            raise RuntimeError("a leaf Variable that requires grad is being used in an in-place operation.")
        raise RuntimeError(f"{name}(): functions with out=... arguments don't support automatic differentiation, but one of the arguments requires grad.")

    # 其余算子与 torch 一致：只要输入需要梯度就不支持 out= 形式
    if not allow_grad and builtins.any(isinstance(t, TN) and t.requires_grad for t in inputs):
        raise RuntimeError(f"{name}(): functions with out=... arguments don't support automatic differentiation, but one of the arguments requires grad.")

    if out.shape != result.shape:
        raise RuntimeError(f"{name}(): {label} shape ({out.shape}) is incompatible with result shape ({result.shape})")
    if out.dtype != result.dtype:
        raise RuntimeError(f"Expected out tensor to have dtype {result.dtype}, but got {out.dtype} instead")
    if out.device != result.device:
        raise RuntimeError(f"Expected out tensor to have device {result.device}, but got {out.device} instead")


def _apply_out(out, result: TN, inputs, name: str, allow_grad: bool = False) -> TN:
    """校验 out 并把计算结果写入其中；out 为 None 时直接返回 result"""
    if out is None:
        return result
    _validate_out_tensor(out, result, inputs, name, allow_grad=allow_grad)
    if allow_grad and result.requires_grad:
        # 与 torch 的 CopyBackwards 一致：通过原地拷贝把 out 接入计算图
        out.copy_(result)
    else:
        result._get_array_lib().copyto(out._array, result._array)
    return out


def _apply_out_tuple(out, results: tuple, inputs, name: str) -> tuple:
    """校验多输出场景的 out 元组并把结果写入其中；out 为 None 时原样返回 results"""
    if out is None:
        return results
    if not isinstance(out, (tuple, list)) or len(out) != len(results):
        raise TypeError(f"{name}(): out must be a tuple of {len(results)} tensors, but got {out}")
    # 先整体校验，避免部分写入后才发现不合法
    for i, (out_item, result_item) in enumerate(zip(out, results)):
        _validate_out_tensor(out_item, result_item, inputs, name, label=f"out[{i}]")
    for out_item, result_item in zip(out, results):
        result_item._get_array_lib().copyto(out_item._array, result_item._array)
    return tuple(out)


def cumsum(input: TN, dim: int, *, dtype: str | np.dtype | None = None, out: TN | None = None) -> TN:
    """
    对输入张量沿指定维度进行累积求和。
    
    参数:
        input: 输入张量
        dim: 要计算累积和的维度
        dtype: 可选，输出张量的数据类型。如果为None，则使用输入张量的数据类型。支持字符串或numpy dtype
        out: 可选，输出张量，用于存储结果。注意：out参数不支持梯度跟踪
    
    返回:
        TN: 累积求和结果张量，与输入张量形状相同
    
    示例:
        >>> x = tensor([[1, 2, 3], [4, 5, 6]])
        >>> cumsum(x, dim=0)
        tensor([[1, 2, 3], [5, 7, 9]])
        >>> cumsum(x, dim=1)
        tensor([[1, 3, 6], [4, 9, 15]])
        >>> x = tensor([1, 2, 3])  # 1D向量
        >>> cumsum(x, dim=0)
        tensor([1, 3, 6])
        >>> x = tensor(5.0)  # 0D标量
        >>> cumsum(x, dim=0)
        tensor(5.0)
    """
    if not isinstance(input, TN):
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input)}")
    
    # 处理dtype参数 - 支持字符串格式或numpy dtype
    if dtype is None:
        result_dtype = input.dtype
    elif isinstance(dtype, str):
        result_dtype = np.dtype(dtype)
    else:
        result_dtype = dtype
    
    # 处理0D标量情况
    if input.ndim == 0:
        # 0D标量的累积和就是它自身
        ret = input.type(result_dtype)
        return _apply_out(out, ret, (input,), "cumsum")

    # 处理1D向量情况，验证dim参数的有效性
    if input.ndim == 1:
        # 1D向量只支持dim=0或dim=-1
        if dim not in [0, -1]:
            raise IndexError(f"Dimension out of range (expected to be in range of [-1, 0], but got {dim})")
    
    # 前向计算 - 使用numpy.cumsum
    result_data = input._array.cumsum(axis=dim, dtype=result_dtype)
    
    # 创建结果张量
    ret = TN(result_data, requires_grad=(is_grad_enabled() and input.requires_grad))
    
    # 注册梯度函数
    if ret.requires_grad:
        ret.fromvars = (input,)
        ret.parms = ((dim, result_dtype),)
        ret.gradfuncs = (_cumsum_backward,)
    
    return _apply_out(out, ret, (input,), "cumsum")

def _mean_backward(result_tensor:TN, i:int)->TN:
    dim, keepdim = result_tensor.parms[i]     
    x = result_tensor.fromvars[i]

    if isinstance(dim, int):
        n = x._array.shape[dim]
    elif dim is None:
        n = x.numel()
    elif isinstance(dim, tuple):
        n = 1
        for d in dim:
            n *= x._array.shape[d]
    else:
        raise TypeError(dim)

    n = x.dtype.type(n)       # 确保n与x的dtype一致，避免计算时导致数据类型提升

    # 均值的梯度传播到输入张量，需要缩放1/n，同时考虑keepdim参数，要恢复维度
    # 最后要expand到输入的形状，才得到输入的梯度
    if keepdim:
        # keepdim=True时，直接expand到输入形状，无需unsqueeze
        grad = (result_tensor.grad_value / n)._expand(x.shape)
    else:
        # unsqueeze在dim=None时直接返回原张量，无需额外判断
        grad = (result_tensor.grad_value / n).unsqueeze(dim)._expand(x.shape)
    
    return grad

def mean(x:TN, dim:int|tuple|None=None, keepdim:bool=False)->TN:
    if not isinstance(x,TN):
        raise SyntaxError('x must be a tensor')
    
    if dim == ():
        dim = None
    
    # 根据x的数组类型选择使用np或cp
    value = x._array.mean(axis=dim, keepdims=keepdim)
    ret = TN(value, requires_grad = (is_grad_enabled() and x.requires_grad))
    
    if ret.requires_grad:
        ret.fromvars = (x,)
        ret.parms = ((dim,keepdim),)
        ret.gradfuncs = (_mean_backward,)
    
    return ret

def _abs_backward(result_tensor:TN, i:int)->TN:
    x = result_tensor.fromvars[i]
    # 添加安全检查，避免当result_tensor为0时的除零错误
    # 当result_tensor为0时，梯度应为0
    is_zero = (result_tensor == 0)
    # 使用where操作避免除零，当result_tensor为0时使用1代替
    safe_divisor = where(is_zero, ones_like(result_tensor), result_tensor)
    # 当result_tensor为0时，(x / safe_divisor)将被替换为0
    grad = result_tensor.grad_value * where(is_zero, zeros_like(x), x / safe_divisor)
    
    return grad

def abs(x:TN)->TN:
    """
    计算张量的绝对值。
    
    返回一个新张量，其中每个元素是输入张量对应元素的绝对值。
    
    Args:
        x (TN): 输入张量
        
    Returns:
        TN: 包含输入张量绝对值的新张量
        
    Examples:
        >>> a = tensor([-1, -2, 3])
        >>> abs(a)  # 返回[1, 2, 3]
        
        >>> b = tensor([[-1.5, 2.5], [3.0, -4.0]])
        >>> abs(b)  # 返回[[1.5, 2.5], [3.0, 4.0]]
    """
    x = _as_tn(x)
    arrlib = x._get_array_lib()
    value = arrlib.abs(x._array)
    ret = TN(value, requires_grad = (is_grad_enabled() and x.requires_grad))
    
    if ret.requires_grad:
        ret.fromvars = (x,)
        ret.gradfuncs = (_abs_backward,)
    
    return ret

def sqrt(x:TN)->TN:
    """
    计算张量的平方根。
    
    返回一个新张量，其中每个元素是输入张量对应元素的平方根。
    
    Args:
        x (TN): 输入张量，元素必须为非负数
        
    Returns:
        TN: 包含输入张量平方根的新张量
        
    Examples:
        >>> a = tensor([1, 4, 9])
        >>> sqrt(a)  # 返回[1, 2, 3]
        
        >>> b = tensor([[4.0, 9.0], [16.0, 25.0]])
        >>> sqrt(b)  # 返回[[2.0, 3.0], [4.0, 5.0]]
    """
    x = _as_tn(x)
    return x ** 0.5

def _create_maxmin_mask(arr, dim:int|tuple|None=None, *, indices=None, maxmin_val=None):
    """创建最大值/最小值掩码数组，支持梯度分配到极值位置

    该函数用于在自动微分过程中，为max/min函数的反向传播创建掩码数组。
    当dim=None时，梯度会被平均分配到所有相同的极值位置；
    当指定dim时，梯度仅分配到每个切片中的第一个极值位置。

    参数:
        arr (numpy.ndarray): 输入的numpy数组
        dim (int|tuple|None, 可选): 计算最大值/最小值的轴，默认为None（全局计算）
        indices (numpy.ndarray, 关键字参数): 前向计算时已保存的argmax/argmin索引，
            dim为int或tuple时必须提供
        maxmin_val (numpy.ndarray, 关键字参数): 前向计算时已保存的极值，
            dim为None时必须提供

    返回:
        numpy.ndarray: 掩码数组，形状与输入数组相同，在极值位置为1.0，其他位置为0.0

    异常:
        ValueError: 当dim参数不是None、整数或元组时抛出

    示例:
        >>> arr = np.array([[1., 6., 3.], [4., 6., 5.]])
        >>> _create_maxmin_mask(arr, maxmin_val=6.0)
        array([[0. , 0.5, 0. ],
               [0. , 0.5, 0. ]])

        >>> _create_maxmin_mask(arr, dim=1, indices=np.array([1, 1]))
        array([[0., 1., 0.],
               [0., 1., 0.]])
    """
    arrlib = np if isinstance(arr,np.ndarray) else cp

    if dim is None:
        # 全局极值 - 使用前向保存的极值，梯度平均分配到所有相同极值位置
        equal_mask = (arr == maxmin_val)
        mask_arr = arrlib.zeros_like(arr, dtype=arr.dtype)
        mask_sum = equal_mask.sum()
        if mask_sum > 0:
            mask_arr[equal_mask] = 1.0 / mask_sum
        return mask_arr

    if isinstance(dim, int):
        # 单轴极值 - 使用前向保存的索引和广播比较创建掩码
        idx_shape = [s if d != dim else 1 for d, s in enumerate(arr.shape)]
        indices_reshaped = indices.reshape(idx_shape)
        arange_shape = [1] * arr.ndim
        arange_shape[dim] = arr.shape[dim]
        dim_range = arrlib.arange(arr.shape[dim]).reshape(arange_shape)
        return (dim_range == indices_reshaped).astype(arr.dtype)

    if isinstance(dim, tuple):
        # 多轴极值 - 使用前向保存的索引（合并后的flat索引）
        original_dims = [arr.shape[ax] for ax in dim]
        multi_indices = arrlib.unravel_index(indices, original_dims)
        other_indices = list(arrlib.ogrid[tuple(slice(s) for s in indices.shape)])
        full_indices = []
        idx_iter = iter(multi_indices)
        for i in range(arr.ndim):
            if i in dim:
                full_indices.append(next(idx_iter))
            else:
                full_indices.append(other_indices.pop(0))
        mask_arr = arrlib.zeros_like(arr, dtype=arr.dtype)
        mask_arr[tuple(full_indices)] = 1.0
        return mask_arr

    raise ValueError("`axis` must be None, int, or tuple")


# 返回类型与torch.return_types.max/min一致
ReturnType_max = namedtuple('max', ['values', 'indices'])
ReturnType_min = namedtuple('min', ['values', 'indices'])

# 修改现有的_max_backward函数保持不变
def _max_backward(result_tensor:TN, i:int)->TN:
    x = result_tensor.fromvars[i]
    dim, keepdim, indices, maxmin_val = result_tensor.parms[i]

    with device_context(x):
        # 使用前向保存的索引/极值，避免重复计算
        max_pos_one_like_x = _create_maxmin_mask(x._array, dim, indices=indices, maxmin_val=maxmin_val)

    # 如果sum计算时结果张量做过维度精简，需要将缩减的维度暂时恢复
    if keepdim == False:
        new_result_grad = result_tensor.grad_value.unsqueeze(dim)
    else:
        new_result_grad = result_tensor.grad_value

    grad = new_result_grad * max_pos_one_like_x
    return grad

# 修改现有的_min_backward函数保持不变
def _min_backward(result_tensor:TN, i:int)->TN:
    x = result_tensor.fromvars[i]
    dim, keepdim, indices, maxmin_val = result_tensor.parms[i]

    with device_context(x):
        # 使用前向保存的索引/极值，避免重复计算
        min_pos_one_like_x = _create_maxmin_mask(x._array, dim, indices=indices, maxmin_val=maxmin_val)

    # 如果sum计算时结果张量做过维度精简，需要将缩减的维度暂时恢复
    if keepdim == False:
        new_result_grad = result_tensor.grad_value.unsqueeze(dim)
    else:
        new_result_grad = result_tensor.grad_value

    grad = new_result_grad * min_pos_one_like_x
    return grad

def max(x:TN, dim:int|tuple|None=None, keepdim:bool=False, *, out=None):
    if x.is_complex():
        raise RuntimeError("max() does not support complex input")

    # 规范化负维度：前向虽支持负轴，但反向的掩码重建需要非负维度
    if isinstance(dim, int):
        dim = dim + x.ndim if dim < 0 else dim
    elif isinstance(dim, tuple):
        dim = tuple(d + x.ndim if d < 0 else d for d in dim)

    arrlib = x._get_array_lib()
    # 计算最大值 - 利用numpy原生的axis=None行为
    values_arr = arrlib.max(x._array, axis=dim, keepdims=keepdim)
    needs_grad = is_grad_enabled() and x.requires_grad
    values_tensor = TN(values_arr, requires_grad=needs_grad)

    # 计算索引/极值 - 前向返回值和反向梯度共用，避免反向重复计算
    indices_arr = None
    maxmin_val = None
    if dim is None:
        # dim为None时保存极值，反向传播直接复用
        maxmin_val = values_arr
    elif isinstance(dim, int):
        indices_arr = arrlib.argmax(x._array, axis=dim)
    elif isinstance(dim, tuple):
        # 多轴argmax: 展平后计算，保存flat索引供反向传播复用
        transposed = arrlib.moveaxis(x._array, dim, range(len(dim)))
        merged_shape = (-1,) + transposed.shape[len(dim):]
        flattened = transposed.reshape(merged_shape)
        indices_arr = arrlib.argmax(flattened, axis=0)

    # 为values_tensor设置梯度信息
    if needs_grad:
        values_tensor.fromvars = (x,)
        # 保存索引和极值到parms，反向传播时直接复用
        values_tensor.parms = ((dim, keepdim, indices_arr, maxmin_val),)
        values_tensor.gradfuncs = (_max_backward,)

    # 根据dim是否为None决定返回类型
    if dim is None:
        return _apply_out(out, values_tensor, (x,), "max")
    else:
        if keepdim and isinstance(dim, int):
            indices_arr = arrlib.expand_dims(indices_arr, axis=dim)
        indices_tensor = TN(indices_arr)
        return _apply_out_tuple(out, ReturnType_max(values_tensor, indices_tensor), (x,), "max")

def min(x:TN, dim:int|tuple|None=None, keepdim:bool=False, *, out=None):
    if x.is_complex():
        raise RuntimeError("min() does not support complex input")

    # 规范化负维度：前向虽支持负轴，但反向的掩码重建需要非负维度
    if isinstance(dim, int):
        dim = dim + x.ndim if dim < 0 else dim
    elif isinstance(dim, tuple):
        dim = tuple(d + x.ndim if d < 0 else d for d in dim)

    arrlib = x._get_array_lib()
    # 计算最小值 - 利用numpy原生的axis=None行为
    values_arr = arrlib.min(x._array, axis=dim, keepdims=keepdim)
    needs_grad = is_grad_enabled() and x.requires_grad
    values_tensor = TN(values_arr, requires_grad=needs_grad)

    # 计算索引/极值 - 前向返回值和反向梯度共用，避免反向重复计算
    indices_arr = None
    maxmin_val = None
    if dim is None:
        # dim为None时保存极值，反向传播直接复用
        maxmin_val = values_arr
    elif isinstance(dim, int):
        indices_arr = arrlib.argmin(x._array, axis=dim)
    elif isinstance(dim, tuple):
        # 多轴argmin: 展平后计算，保存flat索引供反向传播复用
        transposed = arrlib.moveaxis(x._array, dim, range(len(dim)))
        merged_shape = (-1,) + transposed.shape[len(dim):]
        flattened = transposed.reshape(merged_shape)
        indices_arr = arrlib.argmin(flattened, axis=0)

    # 为values_tensor设置梯度信息
    if needs_grad:
        values_tensor.fromvars = (x,)
        # 保存索引和极值到parms，反向传播时直接复用
        values_tensor.parms = ((dim, keepdim, indices_arr, maxmin_val),)
        values_tensor.gradfuncs = (_min_backward,)

    # 根据dim是否为None决定返回类型
    if dim is None:
        return _apply_out(out, values_tensor, (x,), "min")
    else:
        if keepdim and isinstance(dim, int):
            indices_arr = arrlib.expand_dims(indices_arr, axis=dim)
        indices_tensor = TN(indices_arr)
        return _apply_out_tuple(out, ReturnType_min(values_tensor, indices_tensor), (x,), "min")

def _var_backward(result_tensor:TN, i:int)->TN:
    dim,unbiased,keepdim = result_tensor.parms[i]     
    x=result_tensor.fromvars[i]
    
    # 计算样本数量n
    if dim is None:
        n = x.numel()
    elif isinstance(dim, int):
        n = x.shape[dim]
    elif isinstance(dim, tuple):
        # 对于多维度，我们需要计算每个维度的大小的乘积
        n = 1
        for d in dim:
            n *= x.shape[d]
    else:
        raise TypeError(f"dim must be int, tuple or None, got {type(dim)}")
    
    # 计算自由度调整
    ddof = 1 if unbiased else 0
    denom = builtins.max(n - ddof, 1)  # 避免除以0
    
    # 计算(x - mean(x))的梯度因子
    x_mean = mean(x, dim=dim, keepdim=True)  # 始终使用keepdim=True以保持广播正确
    diff = x - x_mean
    factor = x.dtype.type(2.0 / denom)       # 确保因子与输入张量的dtype一致，避免计算时导致数据类型提升

    # 如果keepdim为False，需要扩展梯度维度以匹配diff的维度
    # unsqueeze支持int、tuple和None，dim=None时直接返回原张量
    if not keepdim:
        grad_value = result_tensor.grad_value.unsqueeze(dim)
    else:
        grad_value = result_tensor.grad_value
    
    # 计算最终梯度
    grad = grad_value * diff * factor
    return grad

def var(x:TN, dim:int|tuple|None=None, unbiased:bool=True, keepdim:bool=False)->TN:
    if not isinstance(x,TN):
        raise SyntaxError('x must be a tensor')
    
    if dim == ():
        dim = None
    ddof = 1 if unbiased == True else 0

    value = x._array.var(axis=dim, ddof=ddof, keepdims=keepdim)
    ret = TN(value, requires_grad = (is_grad_enabled() and x.requires_grad))
    
    if ret.requires_grad:
        ret.fromvars=(x,)
        ret.parms=((dim,unbiased,keepdim),)
        ret.gradfuncs=(_var_backward,)
    
    return ret

def std(x:TN, dim:int|tuple|None=None, unbiased:bool=True, keepdim:bool=False)->TN:
    return sqrt(var(x,dim,unbiased,keepdim))

def _transpose_backward(result_tensor:TN, i:int)->TN:
    d1,d2=result_tensor.parms[i]
    grad = result_tensor.grad_value.transpose(d1,d2)
    return grad

def _permute_backward(result_tensor:TN, i: int):
    """
    permute操作的反向传播函数，直接返回重排后的梯度
    """
    # 获取维度排列信息
    permutation = result_tensor.parms[i]
    grad_value = result_tensor.grad_value
    
    # 计算逆排列
    inv_permutation = tuple(np.argsort(permutation))
    # 使用O(n)算法计算逆排列，避免使用排序
    # n = len(permutation)
    # inv_permutation = [0] * n
    # for i, p in enumerate(permutation):
    #     inv_permutation[p] = i  # 关键：新索引p映射到原始索引i
    # inv_permutation = tuple(inv_permutation)
    
    # 直接使用permute方法重排梯度维度并返回
    # permute方法会自动处理梯度计算图的记录
    return grad_value.permute(*inv_permutation)


def _getitem_backward(result_tensor: TN, i: int) -> TN:
    x = result_tensor.fromvars[i]
    index = result_tensor.parms[i]
    out_grad = result_tensor.grad_value

    # 初始化ret_grad与x的shape相同，初始值为0
    ret_grad = zeros_like(x) 

    # 将out_grad中的梯度值根据index反向添加到ret_grad中
    ret_grad = ret_grad.addat(index,out_grad)
    return ret_grad

def _batch_slice(tensor: TN, index: tuple) -> TN:
    """vmap 专用快速切片，等价于 tensor[index] 但跳过通用索引分派。

    仅用于基础索引（slice / 整数）且索引已由调用方归一化为非负的情形：
    直接取底层数组的视图，并在需要梯度时保留到原张量的梯度边
    （fromvars / parms / gradfuncs 与 __getitem__ 完全一致）。
    """
    data = tensor._array[index]
    # 标量索引可能返回 numpy/cupy 标量而非 0 维数组，
    # 统一转成数组，保持与 __getitem__ 的一致性（否则 ndim/_get_array_lib 会失效）
    arrlib = tensor._get_array_lib()
    if not isinstance(data, arrlib.ndarray):
        data = arrlib.asarray(data)

    ret = TN(data, requires_grad=is_grad_enabled() and tensor.requires_grad)
    ret._is_view = True
    ret._base = tensor
    ret._view_index = index
    if ret.requires_grad:
        ret.fromvars = (tensor,)
        ret.parms = (index,)
        ret.gradfuncs = (_getitem_backward,)
    return ret


def _add_grad_left(result_tensor:TN, i:int)->TN:
    left_tensor = result_tensor.fromvars[i]
    left_var_shape = left_tensor.shape
    result_shape = result_tensor.shape

    # shape一样时，直接返回grad_value，否则对result_tensor.grad_value进行sum缩减
    if left_var_shape == result_shape:
        grad = result_tensor.grad_value
    else:
        # left_tensor与result_tensor的shape比较，获取需left_tensor广播轴序号的元组
        broadcast_axes=_get_broadcast_axis(result_shape,left_var_shape)
        grad = sum(result_tensor.grad_value,dim=broadcast_axes,keepdim=False)._reshape(left_var_shape)
    return grad

def _add_grad_right(result_tensor:TN, i:int)->TN:
    right_tensor = result_tensor.fromvars[i]
    right_var_shape = right_tensor.shape
    result_shape = result_tensor.shape

    # shape一样时，直接返回grad_value，否则对result_tensor.grad_value进行sum缩减
    if right_var_shape == result_shape:
        grad = result_tensor.grad_value
    else:
        # right_tensor与result_tensor的shape比较，获取需right_tensor广播轴序号的元组
        broadcast_axes=_get_broadcast_axis(result_shape,right_var_shape)
        grad = sum(result_tensor.grad_value,dim=broadcast_axes,keepdim=False)._reshape(right_var_shape)
    return grad

def _sub_grad_left(result_tensor:TN, i:int)->TN:
    return _add_grad_left(result_tensor,i)

def _sub_grad_right(result_tensor:TN, i:int)->TN:
    grad = _add_grad_right(result_tensor,i)
    return -grad

def _mul_grad_left(result_tensor:TN, i:int)->TN:
    left_tensor = result_tensor.fromvars[i]
    # parms为空时，从fromvars中获取右值，否则从parms中获取
    if result_tensor.parms:
        right_value = result_tensor.parms[i]
    else:
        right_value = result_tensor.fromvars[i+1]
    left_var_shape = left_tensor.shape
    result_shape = result_tensor.shape

    # 右值可能是非张量，使用conj()函数获取共轭
    left_grad = result_tensor.grad_value * conj(right_value)

    # shape一样时，直接返回left_grad，否则对left_grad进行sum缩减
    if left_var_shape == result_shape:
        grad = left_grad
    else:
        # left_tensor与result_tensor的shape比较，获取需left_tensor广播轴序号的元组
        broadcast_axes=_get_broadcast_axis(result_shape,left_var_shape)
        grad = sum(left_grad,
                    dim=broadcast_axes,
                    keepdim=False)._reshape(left_var_shape)
    return grad

def _mul_grad_right(result_tensor:TN, i:int)->TN:
    # parms为空时，从fromvars中获取左值，否则从parms中获取
    if result_tensor.parms:
        left_tensor = result_tensor.parms[i]
    else:
        left_tensor = result_tensor.fromvars[i-1]
    
    right_tensor = result_tensor.fromvars[i]
    right_var_shape = right_tensor.shape
    result_shape = result_tensor.shape
    
    # 左值一定是张量，使用conj()成员函数获取共轭效率更高
    right_grad = result_tensor.grad_value * left_tensor.conj()

    # shape一样时，直接返回right_grad，否则对right_grad进行sum缩减
    if right_var_shape == result_shape:
        grad = right_grad
    else:
        # right_tensor与result_tensor的shape比较，获取需right_tensor广播轴序号的元组
        broadcast_axes=_get_broadcast_axis(result_shape,right_var_shape)
        grad = sum(right_grad,
                    dim=broadcast_axes,
                    keepdim=False)._reshape(right_var_shape)
    return grad

def _matmul_grad_left(result_tensor:TN, i:int)->TN:
    """
    计算矩阵乘法 C = A @ B 中左操作数 A 的梯度。
    
    数学公式：dL/dA = grad @ B^T
    其中 grad = dL/dC，是结果张量的梯度。
    
    分支处理策略：
    
    分支1 - 1D @ 1D（向量点积）：
        场景：A(K,) @ B(K,) -> C()
        公式：dL/dA[k] = grad * B[k]
        实现：逐元素乘法，无需额外操作
    
    分支2 - 1D @ 多维（行向量乘矩阵）：
        场景：A(K,) @ B(..., K, N) -> C(..., N)
        公式：dL/dA = sum(批次维度)(grad @ B^T)
             = sum(...)(grad[..., n] * B[..., k, n])
        实现：unsqueeze(-2) + @ + sum
        性能选择：使用矩阵乘法 + 求和，走 BLAS，实测比 einsum 快约 1.6~7.3 倍
    
    分支3 - 多维 @ 1D（矩阵乘列向量）：
        场景：A(..., M, K) @ B(K,) -> C(..., M)
        公式：dL/dA[..., m, k] = grad[..., m] * B[k]（外积）
        实现：einsum('...m,k->...mk', grad, B)
        性能选择：使用 einsum 直接计算外积，无 GEMM 退化开销。实测产出元素数 ≳8192 时比
            unsqueeze + 广播乘快 1.1~1.4 倍；≲4096 时 einsum 约 15µs 的固定解析开销占主导，
            该区间 unsqueeze + 广播乘最多快 2.1 倍。交叉点约 8192 个产出元素
    
    分支4 - 多维 @ 多维（批量矩阵乘法）：
        场景：A(..., M, K) @ B(..., K, N) -> C(..., M, N)
        公式：dL/dA = grad @ B^T
        实现：标准矩阵乘法 @
        性能选择：使用 @ 运算符，走 BLAS，实测比 einsum 快约 1.0~3.0 倍
    
    Args:
        result_tensor: 矩阵乘法结果张量 C
        i: 左操作数 A 在 fromvars 中的索引
        
    Returns:
        左操作数 A 的梯度张量，形状与 A 相同
    """
    left_tensor:TN = result_tensor.fromvars[i]
    right_tensor:TN = result_tensor.fromvars[i+1]

    # 确保grad_value是连续内存，避免后续@运算符性能下降
    grad_value = result_tensor.grad_value.contiguous()

    left_ndim = left_tensor._array.ndim
    right_ndim = right_tensor._array.ndim

    # 分支1：1D @ 1D，逐元素乘法
    if left_ndim == 1 and right_ndim == 1:
        return grad_value * right_tensor.conj()

    # 分支2：1D @ 多维，矩阵乘法 + 批次求和
    if left_ndim == 1 and right_ndim > 1:
        # grad 为 C 的形状 (..., N)，插入行维变为"单行矩阵"才能与 B^T 参与 @ 运算
        new_grad = grad_value.unsqueeze(-2)  # (..., N) -> (..., 1, N)
        mat_grad = new_grad @ right_tensor.mT.conj()  # (..., 1, N) @ (..., N, K) -> (..., 1, K)

        # A 无批次维，前 right_ndim-1 个轴（B 的批次维 + 插入的行维）全由广播引入，
        # 求和消去后恰剩 (K,)，即 A 的形状，无需 reshape
        sum_axes = tuple(range(right_ndim - 1))
        return sum(mat_grad, dim=sum_axes, keepdim=False)

    # 分支3：多维 @ 1D，einsum 计算外积
    if left_ndim > 1 and right_ndim == 1:
        return api_registry.get('einsum')('...m,k->...mk', grad_value, right_tensor.conj())

    # 分支4：多维 @ 多维，标准矩阵乘法 @
    # 只比较批次部分（[:-2] 剥离矩阵维），找出 A 被广播的批次轴
    broadcast_axes = _get_broadcast_axis(result_tensor._array.shape[:-2], left_tensor._array.shape[:-2])
    # 乘积形状 = C 的批次 + (M, K)，矩阵部分已是 dA
    mat_grad = grad_value @ right_tensor.mT.conj()

    if broadcast_axes:
        # 广播使 A 的一份数据被当成多份参与计算（缺前导批次维或 size-1 维被扩大），
        # 多份位置的梯度须沿广播轴求和归还为一份；求和后形状随广播类型而异，
        # 统一 reshape 对齐 A 的形状（_reshape 形状一致时零开销）。
        return sum(mat_grad, dim=broadcast_axes, keepdim=False)._reshape(left_tensor.shape)
    # 无广播时 mat_grad 已是 A 的形状，直接返回
    return mat_grad

def _matmul_grad_right(result_tensor:TN, i:int)->TN:
    """
    计算矩阵乘法 C = A @ B 中右操作数 B 的梯度。
    
    数学公式：dL/dB = A^T @ grad
    其中 grad = dL/dC，是结果张量的梯度。
    
    分支处理策略：
    
    分支1 - 1D @ 1D（向量点积）：
        场景：A(K,) @ B(K,) -> C()
        公式：dL/dB[k] = grad * A[k]
        实现：逐元素乘法，无需额外操作
    
    分支2 - 1D @ 多维（行向量乘矩阵）：
        场景：A(K,) @ B(..., K, N) -> C(..., N)
        公式：dL/dB[..., k, n] = A[k] * grad[..., n]（外积）
        实现：einsum('k,...n->...kn', A, grad)
        性能选择：使用 einsum 直接计算外积，支持批次维度。实测产出元素数 ≳8192 时比
            unsqueeze 方案快 1.1~1.5 倍；≲4096 时 einsum 约 15µs 的固定解析开销占主导，
            该区间 unsqueeze 方案最多快 1.8 倍。交叉点约 8192 个产出元素
    
    分支3 - 多维 @ 1D（矩阵乘列向量）：
        场景：A(..., M, K) @ B(K,) -> C(..., M)
        公式：dL/dB[k] = sum(批次维度, m)(grad[..., m] * A[..., m, k])
             = sum(...)(grad @ A)
        实现：unsqueeze(-1) + @ + sum(批次维度 + 尾部1维)
        性能选择：使用矩阵乘法 + 求和，走 BLAS，实测比 einsum 快约 1.0~1.8 倍
    
    分支4 - 多维 @ 多维（批量矩阵乘法）：
        场景：A(..., M, K) @ B(..., K, N) -> C(..., M, N)
        公式：dL/dB = A^T @ grad
        实现：标准矩阵乘法 @
        性能选择：使用 @ 运算符，走 BLAS，实测比 einsum 快约 1.0~3.0 倍
    
    Args:
        result_tensor: 矩阵乘法结果张量 C
        i: 右操作数 B 在 fromvars 中的索引
        
    Returns:
        右操作数 B 的梯度张量，形状与 B 相同
    """
    left_tensor:TN = result_tensor.fromvars[i-1]
    right_tensor:TN = result_tensor.fromvars[i]
    
    # 确保grad_value是连续内存，避免后续@运算符性能下降
    grad_value = result_tensor.grad_value.contiguous()

    left_ndim = left_tensor._array.ndim
    right_ndim = right_tensor._array.ndim

    # 分支1：1D @ 1D，逐元素乘法
    if left_ndim == 1 and right_ndim == 1:
        return left_tensor.conj() * grad_value

    # 分支2：1D @ 多维，einsum 计算外积
    if left_ndim == 1 and right_ndim > 1:
        # A 是 1D 不贡献批次维，C 的批次即 B 的批次，einsum 直接产出 B 的形状 (..., K, N)
        return api_registry.get('einsum')('k,...n->...kn', left_tensor.conj(), grad_value)

    # 分支3：多维 @ 1D，矩阵乘法 + 批次求和
    if left_ndim > 1 and right_ndim == 1:
        # grad 为 C 的形状 (..., M)，插入列维变为"单列矩阵"才能与 A^T 参与 @ 运算
        new_grad = grad_value.unsqueeze(-1)  # (..., M) -> (..., M, 1)
        mat_grad = left_tensor.mT.conj() @ new_grad  # (..., K, M) @ (..., M, 1) -> (..., K, 1)

        # B 无批次维，对 A 的批次维（A 被广播时聚合多份梯度）与尾部列维 1
        # 一次求和后恰剩 (K,)，即 B 的形状，无需 reshape
        sum_axes = tuple(range(left_ndim - 2)) + (-1,)
        return sum(mat_grad, dim=sum_axes, keepdim=False)

    # 分支4：多维 @ 多维，标准矩阵乘法 @
    # 只比较批次部分（[:-2] 剥离矩阵维），找出 B 被广播的批次轴
    broadcast_axes = _get_broadcast_axis(result_tensor._array.shape[:-2], right_tensor._array.shape[:-2])
    # 乘积形状 = C 的批次 + (K, N)，矩阵部分已是 dB
    mat_grad = left_tensor.mT.conj() @ grad_value

    if broadcast_axes:
        # 广播使 B 的一份数据被当成多份参与计算（缺前导批次维或 size-1 维被扩大），
        # 多份位置的梯度须沿广播轴求和归还为一份；求和后形状随广播类型而异，
        # 统一 reshape 对齐 B 的形状（_reshape 形状一致时零开销）。
        return sum(mat_grad, dim=broadcast_axes, keepdim=False)._reshape(right_tensor.shape)
    # 无广播时 mat_grad 已是 B 的形状，直接返回
    return mat_grad

def _div_grad_left(result_tensor:TN, i:int)->TN:
    left_tensor = result_tensor.fromvars[i]
    # parms为空时，从fromvars中获取右值，否则从parms中获取
    if result_tensor.parms:
        right_value = result_tensor.parms[i]
    else:
        right_value = result_tensor.fromvars[i+1]
    left_var_shape = left_tensor.shape
    result_shape = result_tensor.shape

    # 右值可能是非张量，使用conj()函数获取共轭
    left_grad = result_tensor.grad_value / conj(right_value)

    # shape一样时，直接返回left_grad，否则对left_grad进行sum缩减
    if left_var_shape == result_shape:
        grad = left_grad
    else:
        # left_tensor与result_tensor的shape比较，获取需left_tensor广播轴序号的元组
        broadcast_axes=_get_broadcast_axis(result_shape,left_var_shape)
        grad = sum(left_grad,
                    dim=broadcast_axes,
                    keepdim=False)._reshape(left_var_shape)
    return grad
 
def _div_grad_right(result_tensor:TN, i:int)->TN:
    right_tensor = result_tensor.fromvars[i]
    right_var_shape = right_tensor.shape
    result_shape = result_tensor.shape

    right_grad = result_tensor.grad_value * (-result_tensor/right_tensor).conj()
    
    # shape一样时，直接返回right_grad，否则对right_grad进行sum缩减
    if right_var_shape == result_shape:
        grad = right_grad
    else:
        # right_tensor与result_tensor的shape比较，获取需right_tensor广播轴序号的元组
        broadcast_axes=_get_broadcast_axis(result_shape,right_var_shape)
        grad = sum(right_grad,
                    dim=broadcast_axes,
                    keepdim=False)._reshape(right_var_shape)
    return grad

def _floordiv_grad_left(result_tensor:TN, i:int)->TN:
    # 向下取整除法的导数几乎处处为0，PyTorch 未实现其反向传播，这里保持一致
    raise RuntimeError('derivative for aten::floor_divide is not implemented')

def _floordiv_grad_right(result_tensor:TN, i:int)->TN:
    raise RuntimeError('derivative for aten::floor_divide is not implemented')

def _floordivat_inplace_backward(result_tensor:TN, i:int)->TN:
    raise RuntimeError('derivative for aten::floor_divide is not implemented')

def _mod_grad_left(result_tensor:TN, i:int)->TN:
    left_tensor = result_tensor.fromvars[i]
    left_var_shape = left_tensor.shape
    result_shape = result_tensor.shape

    # d(left % right)/d(left) = 1，结果梯度即为左值梯度
    if left_var_shape == result_shape:
        grad = result_tensor.grad_value
    else:
        # left_tensor与result_tensor的shape比较，获取需left_tensor广播轴序号的元组
        broadcast_axes=_get_broadcast_axis(result_shape,left_var_shape)
        grad = sum(result_tensor.grad_value,dim=broadcast_axes,keepdim=False)._reshape(left_var_shape)
    return grad

def _mod_grad_right(result_tensor:TN, i:int)->TN:
    right_tensor = result_tensor.fromvars[i]
    # 左值：仅右值需要梯度时存于 parms；两个输入都参与时，左值固定是 fromvars[0]
    if result_tensor.parms:
        left_value = result_tensor.parms[i]
    else:
        left_value = result_tensor.fromvars[0]
    right_var_shape = right_tensor.shape
    result_shape = result_tensor.shape

    # d(left % right)/d(right) = -floor(left/right)
    right_grad = -result_tensor.grad_value * floor(_as_tn(left_value) / right_tensor)

    # shape一样时，直接返回right_grad，否则对right_grad进行sum缩减
    if right_var_shape == result_shape:
        grad = right_grad
    else:
        # right_tensor与result_tensor的shape比较，获取需right_tensor广播轴序号的元组
        broadcast_axes=_get_broadcast_axis(result_shape,right_var_shape)
        grad = sum(right_grad,
                    dim=broadcast_axes,
                    keepdim=False)._reshape(right_var_shape)
    return grad

def _pow_grad_left(result_tensor:TN, i:int)->TN:
    left_tensor = result_tensor.fromvars[i]
    # parms为空时，从fromvars中获取右值，否则从parms中获取
    if result_tensor.parms:
        right_value = result_tensor.parms[i]
    else:
        right_value = result_tensor.fromvars[i+1]
    left_var_shape = left_tensor.shape
    result_shape = result_tensor.shape

    # 将right_value放在乘法运算的右边，确保TN乘法运算符被优先调用,避免numpy的乘运算符优先TN的右乘运算符
    d = (left_tensor ** (right_value - 1.0)) * right_value  
    left_grad = result_tensor.grad_value * d.conj()

    # shape一样时，直接返回left_grad，否则对left_grad进行sum缩减
    if left_var_shape == result_shape:
        grad = left_grad
    else:
        # left_tensor与result_tensor的shape比较，获取需left_tensor广播轴序号的元组
        broadcast_axes=_get_broadcast_axis(result_shape,left_var_shape)
        grad = sum(left_grad,
                    dim=broadcast_axes,
                    keepdim=False)._reshape(left_var_shape)
    return grad

def _pow_grad_right(result_tensor:TN, i:int)->TN:
    # parms为空时，从fromvars中获取左值，否则从parms中获取
    if result_tensor.parms:
        left_value = result_tensor.parms[i]
    else:
        left_value = result_tensor.fromvars[i-1]
    right_tensor = result_tensor.fromvars[i]
    right_var_shape = right_tensor.shape
    result_shape = result_tensor.shape

    right_grad = result_tensor.grad_value * (result_tensor * log(left_value)).conj()

    # shape一样时，直接返回right_grad，否则对right_grad进行sum缩减
    if right_var_shape == result_shape:
        grad = right_grad
    else:
        # right_tensor与result_tensor的shape比较，获取需right_tensor广播轴序号的元组
        broadcast_axes=_get_broadcast_axis(result_shape,right_var_shape)
        grad = sum(right_grad,
                    dim=broadcast_axes,
                    keepdim=False)._reshape(right_var_shape)
    return grad

def pow(input, exponent)->TN:
    """
    逐元素幂运算，等价于 input ** exponent，接口与 torch.pow 一致。

    底数统一归一化为 TN 后完成计算，指数支持 TN、numpy/cupy 数组、
    list、tuple、range、整数或浮点数。

    Args:
        input: 底数（左值）
        exponent: 指数（右值）

    Returns:
        TN: 两输入逐元素幂运算的结果
    """
    # 入口归一化：幂运算不满足交换律，左值非 TN、右值是 TN 时交换操作数并记录顺序，
    # 保证主体代码中的左值 left 一定是 TN，同时保持 input ** exponent 的原始顺序
    if isinstance(input, TN):
        left, right = input, exponent
        reversed_order = False
    elif isinstance(exponent, TN):
        left, right = exponent, input
        reversed_order = True
    else:
        left, right = as_tensor(input), exponent
        reversed_order = False

    if isinstance(right,TN):
        if left.device != right.device:
            raise RuntimeError(f'Expected all tensors to be on the same device, but found at least two devices, {left.device} and {right.device}!')
        right_data = right._array
        right_requires_grad = right.requires_grad
    else:
        right_data = right
        right_requires_grad = False

    #requires_grad属性在运算时传递到结果tensor
    requires_grad = (is_grad_enabled() and (left.requires_grad or right_requires_grad))
    if reversed_order:
        # 保持原始顺序 right ** left（list/tuple/range 交给数组的 __rpow__ 逐元素计算）
        data = right_data ** left._array
    else:
        data = left._array ** right_data
    ret = TN(data, requires_grad=requires_grad)
    
    if requires_grad:
        if reversed_order:
            # 只有作为指数的 left 需要梯度，d(x ** left)/d(left) = x ** left * log(x)
            ret.fromvars=(left,) 
            ret.parms = (right,)
            ret.gradfuncs=(_pow_grad_right,)
        elif left.requires_grad and right_requires_grad:
            ret.fromvars=(left,right) 
            ret.gradfuncs=(_pow_grad_left,_pow_grad_right)
        elif left.requires_grad:
            ret.fromvars=(left,) 
            ret.parms = (right,)
            ret.gradfuncs=(_pow_grad_left,)
        else:
            ret.fromvars= (right,)
            ret.parms = (left,)
            ret.gradfuncs= (_pow_grad_right,)
    
    return ret

def add(x, y, *, alpha=1)->TN:
    """
    逐元素加法，等价于 x + alpha * y，接口与 torch.add 一致。

    左值统一归一化为 TN：x 是 TN 时直接使用；x 非 TN 且 y 是 TN 时交换两操作数
    （加法满足交换律，无需把 x 转成 TN）；x、y 都不是 TN 时把 x 包装成 TN。
    右值支持 TN、numpy/cupy 数组、list、tuple、range、整数或浮点数。

    alpha 为关键字参数：alpha != 1 时右值先乘 alpha 再参与计算
    （右值是 TN 时复用乘法节点，反向传播自动按 alpha 缩放）。

    Args:
        x: 加数（左值）
        y: 加数（右值）
        alpha: 右值的乘数（关键字参数），默认为 1

    Returns:
        TN: 两输入逐元素相加的结果

    Examples:
        >>> a = tensor([1., 2.])
        >>> add(a, 10)             # [11., 12.]
        >>> add(a, 10, alpha=2)    # [21., 22.]，等价于 a + 2 * 10
        >>> add([1., 2.], a)       # 左值非TN，交换后计算 [2., 4.]
    """
    # alpha != 1 时先把右值乘 alpha：
    # 右值是 TN 时复用乘法节点，反向传播自动按 alpha 缩放；
    # list/tuple/range 不做类型转换（可能位于 CPU，而 left 在 CUDA 上），
    # 用推导式逐元素乘 alpha 后交给下面的 left + right 处理
    if alpha != 1:
        if isinstance(y, TN):
            y = y * alpha
        elif isinstance(y, (list, tuple, range)):
            y = [e * alpha for e in y]
        else:
            y = y * alpha

    # 入口归一化，保证主体代码中的左值 left 一定是 TN：
    # 左值是 TN 时直接使用；左值非 TN、右值是 TN 时交换（与 __radd__ 的调度一致，
    # 无需把左值转成 TN）；左右都不是 TN 时把左值包装成 TN 后沿用同一主体逻辑
    if isinstance(x, TN):
        left, right = x, y
    elif isinstance(y, TN):
        left, right = y, x
    else:
        # 左右都不是TN：左值包装成TN后沿用同一主体逻辑，
        # 保持与原来 as_tensor(x) + y 一致的逐元素相加语义（list 不会退化成拼接）
        left, right = as_tensor(x), y

    if isinstance(right,TN):
        if left.device != right.device:
            raise RuntimeError(f'Expected all tensors to be on the same device, but found at least two devices, {left.device} and {right.device}!')
        right_data = right._array
        right_requires_grad = right.requires_grad
    else:
        right_data = right
        right_requires_grad = False

    #requires_grad属性在运算时传递到结果tensor
    requires_grad = (is_grad_enabled() and (left.requires_grad or right_requires_grad))
    ret = TN(left._array + right_data, requires_grad=requires_grad)
    
    if requires_grad:
        if left.requires_grad and right_requires_grad:
            ret.fromvars=(left,right) 
            ret.gradfuncs=(_add_grad_left,_add_grad_right)
        elif left.requires_grad:
            ret.fromvars=(left,) 
            ret.gradfuncs=(_add_grad_left,)
        else:
            ret.fromvars= (right,)
            ret.gradfuncs= (_add_grad_right,)
    
    return ret

def sub(x, y, *, alpha=1)->TN:
    """
    逐元素减法，等价于 x - alpha * y，接口与 torch.sub 一致。

    左值统一归一化为 TN 后完成计算，右值支持 TN、numpy/cupy 数组、
    list、tuple、range、整数或浮点数。

    Args:
        x: 被减数（左值）
        y: 减数（右值）
        alpha: 右值的乘数（关键字参数），默认为 1

    Returns:
        TN: 两输入逐元素相减的结果
    """
    # alpha != 1 时先把右值乘 alpha：
    # 右值是 TN 时复用乘法节点，反向传播自动按 alpha 缩放；
    # list/tuple/range 不做类型转换（可能位于 CPU，而 left 在 CUDA 上），
    # 用推导式逐元素乘 alpha 后交给下面的 left - right 处理
    if alpha != 1:
        if isinstance(y, TN):
            y = y * alpha
        elif isinstance(y, (list, tuple, range)):
            y = [e * alpha for e in y]
        else:
            y = y * alpha

    # 入口归一化：减法不满足交换律，左值非 TN、右值是 TN 时交换操作数并记录顺序，
    # 保证主体代码中的左值 left 一定是 TN，同时保持 x - y 的原始顺序
    if isinstance(x, TN):
        left, right = x, y
        reversed_order = False
    elif isinstance(y, TN):
        left, right = y, x
        reversed_order = True
    else:
        left, right = as_tensor(x), y
        reversed_order = False

    if isinstance(right,TN):
        if left.device != right.device:
            raise RuntimeError(f'Expected all tensors to be on the same device, but found at least two devices, {left.device} and {right.device}!')
        right_data = right._array
        right_requires_grad = right.requires_grad
    else:
        right_data = right
        right_requires_grad = False

    #requires_grad属性在运算时传递到结果tensor
    requires_grad = (is_grad_enabled() and (left.requires_grad or right_requires_grad))
    if reversed_order:
        # 保持原始顺序 right - left（list/tuple/range 交给数组的 __rsub__ 逐元素计算）
        data = right_data - left._array
    else:
        data = left._array - right_data
    ret = TN(data, requires_grad=requires_grad)
    
    if requires_grad:
        if reversed_order:
            # 只有作为减数的 left 需要梯度，d(x - left)/d(left) = -1
            ret.fromvars=(left,) 
            ret.gradfuncs=(_sub_grad_right,)
        elif left.requires_grad and right_requires_grad:
            ret.fromvars=(left,right) 
            ret.gradfuncs=(_sub_grad_left,_sub_grad_right)
        elif left.requires_grad:
            ret.fromvars=(left,) 
            ret.gradfuncs=(_sub_grad_left,)
        else:
            ret.fromvars= (right,)
            ret.gradfuncs= (_sub_grad_right,)
    
    return ret

def mul(x, y)->TN:
    """
    逐元素乘法，等价于 x * y，接口与 torch.mul 一致。

    左值统一归一化为 TN 后完成计算，右值支持 TN、numpy/cupy 数组、
    list、tuple、range、整数或浮点数。

    Args:
        x: 乘数（左值）
        y: 乘数（右值）

    Returns:
        TN: 两输入逐元素相乘的结果
    """
    # 入口归一化，保证主体代码中的左值 left 一定是 TN：
    # 乘法满足交换律，左值非 TN、右值是 TN 时直接交换操作数
    if isinstance(x, TN):
        left, right = x, y
    elif isinstance(y, TN):
        left, right = y, x
    else:
        # 左右都不是TN：左值包装成TN后沿用同一主体逻辑
        left, right = as_tensor(x), y

    if isinstance(right,TN):
        if left.device != right.device:
            raise RuntimeError(f'Expected all tensors to be on the same device, but found at least two devices, {left.device} and {right.device}!')
        right_data = right._array
        right_requires_grad = right.requires_grad
    else:
        right_data = right
        right_requires_grad = False

    #requires_grad属性在运算时传递到结果tensor
    requires_grad = (is_grad_enabled() and (left.requires_grad or right_requires_grad))
    ret = TN(left._array * right_data, requires_grad=requires_grad)
    
    if requires_grad:
        if left.requires_grad and right_requires_grad:
            ret.fromvars=(left,right) 
            ret.gradfuncs=(_mul_grad_left,_mul_grad_right)
        elif left.requires_grad:
            ret.fromvars=(left,) 
            ret.parms = (right,)
            ret.gradfuncs=(_mul_grad_left,)
        else:
            ret.fromvars= (right,)
            ret.parms = (left,)
            ret.gradfuncs= (_mul_grad_right,)
    
    return ret

def div(x, y)->TN:
    """
    逐元素除法（真除法），等价于 x / y，接口与 torch.div 一致。

    左值统一归一化为 TN 后完成计算，右值支持 TN、numpy/cupy 数组、
    list、tuple、range、整数或浮点数。

    Args:
        x: 被除数（左值）
        y: 除数（右值）

    Returns:
        TN: 两输入逐元素相除的结果
    """
    # 入口归一化：除法不满足交换律，左值非 TN、右值是 TN 时交换操作数并记录顺序，
    # 保证主体代码中的左值 left 一定是 TN，同时保持 x / y 的原始顺序
    if isinstance(x, TN):
        left, right = x, y
        reversed_order = False
    elif isinstance(y, TN):
        left, right = y, x
        reversed_order = True
    else:
        left, right = as_tensor(x), y
        reversed_order = False

    if isinstance(right,TN):
        if left.device != right.device:
            raise RuntimeError(f'Expected all tensors to be on the same device, but found at least two devices, {left.device} and {right.device}!')
        right_data = right._array
        right_requires_grad = right.requires_grad
    else:
        right_data = right
        right_requires_grad = False

    #requires_grad属性在运算时传递到结果tensor
    requires_grad = (is_grad_enabled() and (left.requires_grad or right_requires_grad))
    if reversed_order:
        # 保持原始顺序 right / left（list/tuple/range 交给数组的 __rtruediv__ 逐元素计算）
        data = right_data / left._array
    else:
        data = left._array / right_data
    ret = TN(data, requires_grad=requires_grad)
    
    if requires_grad:
        if reversed_order:
            # 只有作为除数的 left 需要梯度，d(x / left)/d(left) = -x / left^2
            ret.fromvars=(left,) 
            ret.parms = (right,)
            ret.gradfuncs=(_div_grad_right,)
        elif left.requires_grad and right_requires_grad:
            ret.fromvars=(left,right) 
            ret.gradfuncs=(_div_grad_left,_div_grad_right)
        elif left.requires_grad:
            ret.fromvars=(left,) 
            ret.parms = (right,)
            ret.gradfuncs=(_div_grad_left,)
        else:
            ret.fromvars= (right,)
            ret.parms = (left,)
            ret.gradfuncs= (_div_grad_right,)
    
    return ret

def floor_divide(x, y)->TN:
    """
    逐元素向下取整除法，等价于 x // y，接口与 torch.floor_divide 一致。

    左值统一归一化为 TN 后完成计算，右值支持 TN、numpy/cupy 数组、
    list、tuple、range、整数或浮点数。

    Args:
        x: 被除数（左值）
        y: 除数（右值）

    Returns:
        TN: 两输入逐元素向下取整除的结果
    """
    # 入口归一化：向下取整除法不满足交换律，左值非 TN、右值是 TN 时交换操作数并记录顺序，
    # 保证主体代码中的左值 left 一定是 TN，同时保持 x // y 的原始顺序
    if isinstance(x, TN):
        left, right = x, y
        reversed_order = False
    elif isinstance(y, TN):
        left, right = y, x
        reversed_order = True
    else:
        left, right = as_tensor(x), y
        reversed_order = False

    if isinstance(right,TN):
        if left.device != right.device:
            raise RuntimeError(f'Expected all tensors to be on the same device, but found at least two devices, {left.device} and {right.device}!')
        right_data = right._array
        right_requires_grad = right.requires_grad
    else:
        right_data = right
        right_requires_grad = False

    #requires_grad属性在运算时传递到结果tensor
    requires_grad = (is_grad_enabled() and (left.requires_grad or right_requires_grad))
    if reversed_order:
        # 保持原始顺序 right // left
        data = right_data // left._array
    else:
        data = left._array // right_data
    ret = TN(data, requires_grad=requires_grad)

    if requires_grad:
        # 向下取整除法的导数几乎处处为0，PyTorch 未实现其反向传播，
        # 这里同样在 backward 时显式报错，避免静默给出错误梯度
        if reversed_order:
            ret.fromvars=(left,) 
            ret.gradfuncs=(_floordiv_grad_right,)
        elif left.requires_grad and right_requires_grad:
            ret.fromvars=(left,right) 
            ret.gradfuncs=(_floordiv_grad_left,_floordiv_grad_right)
        elif left.requires_grad:
            ret.fromvars=(left,) 
            ret.gradfuncs=(_floordiv_grad_left,)
        else:
            ret.fromvars= (right,)
            ret.gradfuncs= (_floordiv_grad_right,)
    
    return ret

def remainder(x, y)->TN:
    """
    逐元素取余运算，等价于 x % y，接口与 torch.remainder 一致。

    左值统一归一化为 TN 后完成计算，右值支持 TN、numpy/cupy 数组、
    list、tuple、range、整数或浮点数。

    Args:
        x: 被除数（左值）
        y: 除数（右值）

    Returns:
        TN: 两输入逐元素取余的结果
    """
    # 入口归一化：取余不满足交换律，左值非 TN、右值是 TN 时交换操作数并记录顺序，
    # 保证主体代码中的左值 left 一定是 TN，同时保持 x % y 的原始顺序
    if isinstance(x, TN):
        left, right = x, y
        reversed_order = False
    elif isinstance(y, TN):
        left, right = y, x
        reversed_order = True
    else:
        left, right = as_tensor(x), y
        reversed_order = False

    if isinstance(right,TN):
        if left.device != right.device:
            raise RuntimeError(f'Expected all tensors to be on the same device, but found at least two devices, {left.device} and {right.device}!')
        right_data = right._array
        right_requires_grad = right.requires_grad
    else:
        right_data = right
        right_requires_grad = False

    #requires_grad属性在运算时传递到结果tensor
    requires_grad = (is_grad_enabled() and (left.requires_grad or right_requires_grad))
    if reversed_order:
        # 保持原始顺序 right % left
        data = right_data % left._array
    else:
        data = left._array % right_data
    ret = TN(data, requires_grad=requires_grad)
    
    if requires_grad:
        # d(left % right)/d(left)=1，d(left % right)/d(right)=-floor(left/right)，与 torch 一致
        if reversed_order:
            ret.fromvars=(left,) 
            ret.parms = (right,)
            ret.gradfuncs=(_mod_grad_right,)
        elif left.requires_grad and right_requires_grad:
            ret.fromvars=(left,right) 
            ret.gradfuncs=(_mod_grad_left,_mod_grad_right)
        elif left.requires_grad:
            ret.fromvars=(left,) 
            ret.parms = (right,)
            ret.gradfuncs=(_mod_grad_left,)
        else:
            ret.fromvars= (right,)
            ret.parms = (left,)
            ret.gradfuncs= (_mod_grad_right,)
    
    return ret

def matmul(x, y)->TN:
    """
    矩阵乘法，等价于 x @ y，接口与 torch.matmul 一致。

    左值统一归一化为 TN 后完成计算，右值支持 TN、numpy/cupy 数组、list、tuple。

    Args:
        x: 左矩阵（左值）
        y: 右矩阵（右值）

    Returns:
        TN: 两输入矩阵相乘的结果
    """
    # 入口归一化：矩阵乘法不满足交换律，左值非 TN、右值是 TN 时交换操作数并记录顺序，
    # 保证主体代码中的左值 left 一定是 TN，同时保持 x @ y 的原始顺序
    if isinstance(x, TN):
        left, right = x, y
        reversed_order = False
    elif isinstance(y, TN):
        left, right = y, x
        reversed_order = True
    else:
        left, right = as_tensor(x), y
        reversed_order = False

    #如右值是非TN对象，转化为TN对象，以便让后续处理一至
    if isinstance(right,TN):
        right_tensor = right
    else:
        right_tensor = as_tensor(
            right,
            dtype = infer_dtype_in_binoper(right,left.dtype),
            device = left.device
        )

    if left.device != right_tensor.device:
        raise RuntimeError(f'Expected all tensors to be on the same device, but found at least two devices, {left.device} and {right_tensor.device}!')

    if left._array.ndim == 0 or right_tensor._array.ndim == 0:
        raise RuntimeError('both arguments to matmul need to be at least 1D')

    # requires_grad属性在运算时传递到结果tensor
    requires_grad = (is_grad_enabled() and (left.requires_grad or right_tensor.requires_grad))
    if reversed_order:
        # 保持原始顺序 right_tensor @ left
        data = right_tensor._array @ left._array
    else:
        data = left._array @ right_tensor._array
    ret = TN(data, requires_grad=requires_grad)

    if requires_grad:
        if reversed_order:
            # fromvars 需按 matmul 的操作数顺序排列，梯度函数按相邻位置取另一操作数
            ret.fromvars=(right_tensor,left)
        else:
            ret.fromvars=(left,right_tensor)
        ret.gradfuncs=(_matmul_grad_left,_matmul_grad_right)

    return ret

def neg(x)->TN:
    """
    逐元素取负，等价于 -x，接口与 torch.neg 一致。

    布尔张量不支持取负（与 torch 一致，会抛 NotImplementedError）；
    无符号整型按位宽回绕并保持 dtype（如 uint8 的 -1 得到 255）。

    Args:
        x (TN): 输入张量

    Returns:
        TN: 输入张量逐元素取负的结果
    """
    x = _as_tn(x)
    dtype = x.dtype

    # 布尔张量取负没有意义，torch 同样不支持；取反请用 ~ 或 bitwise_not()
    if np.issubdtype(dtype, np.bool_):
        raise NotImplementedError(
            "Negation, the `-` operator, on a bool tensor is not supported. "
            "If you are trying to invert a mask, use the `~` or `bitwise_not()` operator instead."
        )

    # numpy/cupy 的无符号整型取负本身就是按位宽回绕（如 uint8: -1 -> 255），与 torch 一致
    data = -x._array

    ret = TN(
        data,
        requires_grad = (is_grad_enabled() and x.requires_grad)
    )

    if  ret.requires_grad:
        ret.fromvars=(x,)
        fn = lambda result,i: result.grad_value.__neg__()
        ret.gradfuncs=(fn,)
    return ret

def positive(x)->TN:
    """
    逐元素取正（返回输入本身），接口与 torch.positive 一致。

    Args:
        x (TN): 输入张量

    Returns:
        TN: 输入张量本身
    """
    return _as_tn(x)

def _compare_operands(x, y):
    """比较/位运算的操作数归一化：返回 (left_tn, right, reversed_order)，
    reversed_order 为 True 表示原始顺序是 right <op> left"""
    if isinstance(x, TN):
        return x, (y._array if isinstance(y, TN) else y), False
    if isinstance(y, TN):
        return y, x, True
    return as_tensor(x), y, False

def lt(x, y)->TN:
    """
    逐元素小于比较，等价于 x < y，接口与 torch.lt 一致。

    返回布尔张量，不参与梯度计算。
    """
    left, right, reversed_order = _compare_operands(x, y)
    if reversed_order:
        return TN(right < left._array)
    return TN(left._array < right)

def le(x, y)->TN:
    """
    逐元素小于等于比较，等价于 x <= y，接口与 torch.le 一致。

    返回布尔张量，不参与梯度计算。
    """
    left, right, reversed_order = _compare_operands(x, y)
    if reversed_order:
        return TN(right <= left._array)
    return TN(left._array <= right)

def gt(x, y)->TN:
    """
    逐元素大于比较，等价于 x > y，接口与 torch.gt 一致。

    返回布尔张量，不参与梯度计算。
    """
    left, right, reversed_order = _compare_operands(x, y)
    if reversed_order:
        return TN(right > left._array)
    return TN(left._array > right)

def ge(x, y)->TN:
    """
    逐元素大于等于比较，等价于 x >= y，接口与 torch.ge 一致。

    返回布尔张量，不参与梯度计算。
    """
    left, right, reversed_order = _compare_operands(x, y)
    if reversed_order:
        return TN(right >= left._array)
    return TN(left._array >= right)

def eq(x, y)->TN:
    """
    逐元素相等比较，等价于 x == y，接口与 torch.eq 一致。

    返回布尔张量，不参与梯度计算。
    """
    left, right, _ = _compare_operands(x, y)
    return TN(left._array == right)

def ne(x, y)->TN:
    """
    逐元素不等比较，等价于 x != y，接口与 torch.ne 一致。

    返回布尔张量，不参与梯度计算。
    """
    left, right, _ = _compare_operands(x, y)
    return TN(left._array != right)

def bitwise_and(x, y)->TN:
    """
    逐元素按位与，等价于 x & y，接口与 torch.bitwise_and 一致。

    返回张量，不参与梯度计算。
    """
    left, right, _ = _compare_operands(x, y)
    return TN(left._array & right)

def bitwise_or(x, y)->TN:
    """
    逐元素按位或，等价于 x | y，接口与 torch.bitwise_or 一致。

    返回张量，不参与梯度计算。
    """
    left, right, _ = _compare_operands(x, y)
    return TN(left._array | right)

def bitwise_xor(x, y)->TN:
    """
    逐元素按位异或，等价于 x ^ y，接口与 torch.bitwise_xor 一致。

    返回张量，不参与梯度计算。
    """
    left, right, _ = _compare_operands(x, y)
    return TN(left._array ^ right)

def bitwise_not(x)->TN:
    """
    逐元素按位取反，等价于 ~x，接口与 torch.bitwise_not 一致。

    返回张量，不参与梯度计算。
    """
    return TN(~_as_tn(x)._array)

def bitwise_left_shift(x, y)->TN:
    """
    逐元素左移，等价于 x << y，接口与 torch.bitwise_left_shift 一致。

    返回张量，不参与梯度计算。
    """
    left, right, reversed_order = _compare_operands(x, y)
    shift = int(right)
    if reversed_order:
        return TN(shift << left._array)
    return TN(left._array << shift)

def bitwise_right_shift(x, y)->TN:
    """
    逐元素右移，等价于 x >> y，接口与 torch.bitwise_right_shift 一致。

    返回张量，不参与梯度计算。
    """
    left, right, reversed_order = _compare_operands(x, y)
    shift = int(right)
    if reversed_order:
        return TN(shift >> left._array)
    return TN(left._array >> shift)

def _log_grad(result_tensor:TN, i:int)->TN:
    x = result_tensor.fromvars[i]
    return result_tensor.grad_value * (1. / x.conj())

def log(x:TN)->TN:
    """
    计算张量的自然对数。

    返回一个新张量，其中每个元素是输入张量对应元素的自然对数(ln(x))。

    Args:
        x (TN): 输入张量，元素必须为正数

    Returns:
        TN: 包含输入张量自然对数的新张量

    Examples:
        >>> a = tensor([1, e, e^2])  # e是自然常数
        >>> log(a)  # 返回[0, 1, 2]

        >>> b = tensor([[1.0, 2.718], [7.389, 20.086]])
        >>> log(b)  # 返回[[0.0, 1.0], [2.0, 3.0]]
    """
    x = _as_tn(x)
    arrlib = x._get_array_lib()
    result_data = arrlib.log(x._array)
    requires_grad = (is_grad_enabled() and x.requires_grad)
    ret = TN(result_data, requires_grad)
    
    if requires_grad:
        ret.fromvars = (x,)
        ret.gradfuncs = (_log_grad,)

    return ret

def log1p(x: TN) -> TN:    
    """
    计算张量的log(1+x)。
    
    返回一个新张量，其中每个元素是输入张量对应元素加1后的自然对数，即log(1+x)。
    对于x接近0的小值，这个函数比直接计算log(1+x)更精确。
    
    Args:
        x (TN): 输入张量，元素必须大于-1
        
    Returns:
        TN: 包含输入张量log(1+x)的新张量
        
    Examples:
        >>> a = tensor([0, 1, 2])
        >>> log1p(a)  # 返回[log(1), log(2), log(3)] ≈ [0, 0.693, 1.099]
        
        >>> b = tensor([[0.0, 1.0], [2.0, 3.0]])
        >>> log1p(b)  # 返回[[0.0, 0.693], [1.099, 1.386]]
    """
    return log(x + 1.0)  # 复用现有log函数

def log2(x:TN)->TN:
    """
    计算张量的二进制对数。
    
    返回一个新张量，其中每个元素是输入张量对应元素的二进制对数(log2(x))。
    
    Args:
        x (TN): 输入张量，元素必须为正数
        
    Returns:
        TN: 包含输入张量二进制对数的新张量
    """
    return log(x) / log(2.0)

def log10(x:TN)->TN:
    """
    计算张量的十进制对数。
    
    返回一个新张量，其中每个元素是输入张量对应元素的十进制对数(log10(x))。
    
    Args:
        x (TN): 输入张量，元素必须为正数
        
    Returns:
        TN: 包含输入张量十进制对数的新张量
    """
    return log(x) / log(10.0)
    
def _exp_grad(result_tensor:TN, i:int)->TN:
    return result_tensor.grad_value * result_tensor.conj()

def exp(x:TN)->TN:
    """
    计算张量的指数函数。

    返回一个新张量，其中每个元素是输入张量对应元素的自然指数(e^x)。

    Args:
        x (TN): 输入张量

    Returns:
        TN: 包含输入张量指数函数值的新张量

    Examples:
        >>> a = tensor([0, 1, 2])
        >>> exp(a)  # 返回[e^0, e^1, e^2] ≈ [1, 2.718, 7.389]

        >>> b = tensor([[0.0, 1.0], [2.0, 3.0]])
        >>> exp(b)  # 返回[[1.0, 2.718], [7.389, 20.086]]
    """
    x = _as_tn(x)
    arrlib = x._get_array_lib()
    ret = TN(arrlib.exp(x._array), is_grad_enabled() and x.requires_grad)
    
    if ret.requires_grad:
        ret.fromvars = (x,)
        ret.gradfuncs = (_exp_grad,)

    return ret

def _where_backward(r, i):
    """where函数的梯度计算函数（外部定义以避免重复创建闭包）"""
    grad = r.grad_value * r.parms[i]
    var = r.fromvars[i]
    var_shape = var.shape
    grad_shape = grad.shape

    # 处理广播场景
    if var_shape != grad_shape:
        broadcast_axes = _get_broadcast_axis(grad_shape, var_shape)
        grad = grad.sum(dim=broadcast_axes, keepdim=False)
        grad = grad._reshape(var_shape)
    
    # 仅在类型不一致时进行转换
    grad = grad.type(var.dtype)
    return grad

@overload
def where(cond: TN, x: None, y: None) -> tuple[TN, ...]:
    ...

@overload
def where(cond: TN, x: TN | int | float, y: TN | int | float) -> TN:
    ...

@api_registry.register('where')
def where(cond: TN, x: TN | int | float | None = None, y: TN | int | float | None = None) -> TN | tuple[TN, ...]:
    
    if not isinstance(cond,TN):
        raise ValueError('cond must be a tensor')
    
    arrlib = cond._get_array_lib()

    if x is None and y is None:
        # 在对应设备上计算非零索引，避免跨设备数据传输
        with device_context(cond):
            tup = arrlib.where(cond._array)
        lst = []
        for idx_arr in tup:
            lst.append(TN(idx_arr))
        return tuple(lst)

    if x is None or y is None:
        raise RuntimeError('one of x,y is None while the other is Non None')
    
    # 处理类型提升和梯度追踪，与PyTorch行为保持一致
    x_is_tensor = isinstance(x, TN)
    y_is_tensor = isinstance(y, TN)
    
    # 设备检查
    if x_is_tensor:
        if cond.device != x.device:
            raise ValueError(f"Expected all tensors to be on the same device, "
                           f"but found devices: cond={cond.device}, x={x.device}")
        x_requires_grad = x.requires_grad
        x_data = x._array
    else:
        x_requires_grad = False
        x_data = x
        
    if y_is_tensor:
        if cond.device != y.device:
            raise ValueError(f"Expected all tensors to be on the same device, "
                           f"but found devices: cond={cond.device}, y={y.device}")
        y_requires_grad = y.requires_grad
        y_data = y._array
    else:
        y_requires_grad = False
        y_data = y
        
    # 条件选择，cond不参与梯度计算
    data = arrlib.where(cond._array, x_data, y_data)
    # 使用计算结果的数据类型，避免强制类型转换
    ret = TN(
        data,
        requires_grad = (is_grad_enabled() and (x_requires_grad or y_requires_grad))
    )
    
    if ret.requires_grad:
        if x_requires_grad and y_requires_grad:
            ret.fromvars = (x, y)
            ret.parms = (cond, 1.0 - cond)
            ret.gradfuncs = (_where_backward, _where_backward)
        elif x_requires_grad:
            ret.fromvars = (x, )
            ret.parms = (cond,)
            ret.gradfuncs = (_where_backward, )
        else:
            ret.fromvars = (y, )
            ret.parms = (1.0 - cond,)
            ret.gradfuncs = (_where_backward, )
    
    return ret

def _sign_derivative(result:TN, x:TN)->tuple[TN]:
    # sign函数在x=0处不可导，在其他点处梯度为0
    # 梯度为零，与上游梯度相乘后仍为零
    return (result.grad_value * zeros_like(x),)

@track_grad(_sign_derivative)
def sign(x:TN)->TN:
    """计算张量的符号函数
    
    对于复数 z = a + bi:
    - 如果 z ≠ 0: sign(z) = z / |z| = (a + bi) / √(a² + b²)
    - 如果 z = 0: sign(z) = 0
    
    该函数支持实数和复数张量，并正确处理反向传播。
    """
    x = _as_tn(x)
    arrlib = x._get_array_lib()
    return TN(arrlib.sign(x._array))

def conj(x):
    """计算输入的共轭。
    
    对于复数 z = a + bi，其共轭为 z* = a - bi。
    对于实数，共轭是其本身。
    
    参数:
        x: 输入，可以是TN张量、numpy/cupy数组、python数值（包括复数）
        
    返回:
        TN张量、数组或python数值: 输入的共轭
        
    示例:
        >>> # TN张量
        >>> a = tensor([1+2j, 3-4j])
        >>> conj(a)  # 返回TN张量[1-2j, 3+4j]
        
        >>> # numpy数组
        >>> import numpy as np
        >>> b = np.array([1+2j, 3-4j])
        >>> conj(b)  # 返回numpy数组[1-2j, 3+4j]
        
        >>> # python复数
        >>> conj(1+2j)  # 返回(1-2j)
        
        >>> # python实数
        >>> conj(5.0)  # 返回5.0
    """
    if isinstance(x, TN) or isinstance(x, np.ndarray) or (cp and isinstance(x, cp.ndarray)):
        # 处理TN张量、numpy数组、cupy数组
        return x.conj()
    elif isinstance(x, complex):
        # 处理python内置复数
        return x.conjugate()
    elif isinstance(x, (int, float, bool)):
        # 处理python实数（int, float）
        return x
    else:
        # 处理其他类型
        raise TypeError(f"conj does not support type: {type(x)}")   

def argsort(input: TN, dim: int = -1, descending: bool = False, stable: bool = False, *, out = None) -> TN:
    """    
    返回沿指定维度按值排序的索引张量，排序结果为升序或降序
    
    参数:
        input: 输入张量
        dim: 排序维度，默认为最后一维
        descending: 是否降序排列，默认为False（升序）
        stable: 是否使用稳定排序算法，默认为False
        out: 可选的输出张量
    
    返回:
        排序后的索引张量
        
    注意: 当使用out参数时，不支持自动微分
    """
    if dim < -input.ndim or dim >= input.ndim:
        raise IndexError(f"Dimension out of range (expected to be in range of [-{input.ndim}, {input.ndim-1}], got {dim})")
    
    if dim < 0:
        dim += input.ndim
    
    arrlib = input._get_array_lib()
    sorted_indices = arrlib.argsort(input._array, axis=dim)
    
    if descending:
        with device_context(input):
            reverse_indices = arrlib.empty_like(sorted_indices)
        dim_size = input._array.shape[dim]
        # 注意：nditer在CPU上执行，但操作的是已有数组
        it = arrlib.nditer(sorted_indices, flags=['multi_index', 'refs_ok'])
        while not it.finished:
            idx = list(it.multi_index)
            sub_array = sorted_indices[tuple(idx[:dim] + [slice(None)] + idx[dim+1:])]  # type: ignore
            reverse_sub_array = sub_array[::-1].copy()
            reverse_indices[tuple(idx[:dim] + [slice(None)] + idx[dim+1:])] = reverse_sub_array
            it.iternext()
        sorted_indices = reverse_indices
    
    sorted_indices_tensor = TN(sorted_indices, requires_grad=False)
    
    return _apply_out(out, sorted_indices_tensor, (input,), "argsort")

def unique(input: TN, sorted: bool = True, return_inverse: bool = False, 
           return_counts: bool = False, return_indices: bool = False) -> TN | tuple:
    """
    返回输入张量中所有唯一的元素。
    
    参数:
        input: 输入张量
        sorted: 如果为True，返回的唯一值将按升序排序
        return_inverse: 如果为True，还返回一个索引张量，使得input可以通过output[inverse_indices]重构
        return_counts: 如果为True，还返回每个唯一值在输入中出现的次数
        return_indices: 如果为True，还返回输入张量中每个唯一值首次出现的索引
    
    返回:
        output: 包含唯一值的一维张量
        如果指定了return_inverse、return_indices或return_counts，则返回一个元组
    """
    # 检查输入类型
    if not isinstance(input, TN):
        raise TypeError("input must be a TN tensor")
    
    # 将输入张量展平为一维数组
    flat_data = input._array.flatten()
    
    arrlib = input._get_array_lib()
    dev = input.device

    # 使用NumPy的unique函数获取唯一值
    # 根据不同的参数组合调用不同版本的np.unique
    if return_inverse and return_counts and return_indices:
        # 返回所有可选值
        unique_values, indices, inverse_indices, counts = arrlib.unique(
            flat_data, return_index=True, return_inverse=True, return_counts=True)
        
        # 创建结果张量
        result = as_tensor(unique_values,device=dev)
        indices_tensor = as_tensor(indices, dtype=int64,device=dev)
        inverse_tensor = as_tensor(inverse_indices, dtype=int64,device=dev)
        counts_tensor = as_tensor(counts, dtype=int64,device=dev)
        
        return result, inverse_tensor, indices_tensor, counts_tensor
    elif return_inverse and return_counts:
        # 返回唯一值、逆索引和计数
        unique_values, inverse_indices, counts = arrlib.unique(
            flat_data, return_inverse=True, return_counts=True)
        
        # 创建结果张量
        result = as_tensor(unique_values,device=dev)
        inverse_tensor = as_tensor(inverse_indices, dtype=int64,device=dev)
        counts_tensor = as_tensor(counts, dtype=int64,device=dev)
        
        return result, inverse_tensor, counts_tensor
    elif return_inverse and return_indices:
        # 返回唯一值、逆索引和首次出现索引
        unique_values, indices, inverse_indices = arrlib.unique(
            flat_data, return_index=True, return_inverse=True)
        
        # 创建结果张量
        result = as_tensor(unique_values,device=dev)
        indices_tensor = as_tensor(indices, dtype=int64,device=dev)
        inverse_tensor = as_tensor(inverse_indices, dtype=int64,device=dev)
        
        return result, inverse_tensor, indices_tensor
    elif return_counts and return_indices:
        # 返回唯一值、首次出现索引和计数
        unique_values, indices, counts = arrlib.unique(
            flat_data, return_index=True, return_counts=True)
        
        # 创建结果张量
        result = as_tensor(unique_values,device=dev)
        indices_tensor = as_tensor(indices, dtype=int64,device=dev)
        counts_tensor = as_tensor(counts, dtype=int64,device=dev)
        
        return result, indices_tensor, counts_tensor
    elif return_inverse:
        # 只返回唯一值和逆索引
        unique_values, inverse_indices = arrlib.unique(
            flat_data, return_inverse=True)
        
        # 创建结果张量
        result = as_tensor(unique_values,device=dev)
        inverse_tensor = as_tensor(inverse_indices, dtype=int64,device=dev)
        
        return result, inverse_tensor
    elif return_counts:
        # 只返回唯一值和计数
        unique_values, counts = arrlib.unique(
            flat_data, return_counts=True)
        
        # 创建结果张量
        result = as_tensor(unique_values,device=dev)
        counts_tensor = as_tensor(counts, dtype=int64,device=dev)
        
        return result, counts_tensor
    elif return_indices:
        # 只返回唯一值和首次出现索引
        unique_values, indices = arrlib.unique(
            flat_data, return_index=True)
        
        # 创建结果张量
        result = as_tensor(unique_values,device=dev)
        indices_tensor = as_tensor(indices, dtype=int64,device=dev)
        
        return result, indices_tensor
    else:
        # 只返回唯一值
        unique_values = arrlib.unique(flat_data)
        return as_tensor(unique_values,device=dev)
# end of unique

def maximum(input: TN, other: TN) -> TN:
    """
    计算两个张量的逐元素最大值
    
    参数:
        input (TN): 第一个输入张量
        other (TN): 第二个输入张量，可以是标量或与input广播兼容的张量
        
    返回:
        TN: 包含两个张量逐元素最大值的新张量
        
    异常:
        TypeError: 当输入不是TN张量时抛出
        
    示例:
        >>> a = rm.tensor([[1, 2], [3, 4]])
        >>> b = rm.tensor([[2, 1], [1, 5]])
        >>> rm.maximum(a, b)
        tensor([[2, 2],
                [3, 5]])
    """
    if not isinstance(input, TN):
        raise TypeError(f"Expected input to be TN tensor, got {type(input)}")
    
    arrlib = input._get_array_lib()
    dev = input.device

    if not isinstance(other, TN):
        other = as_tensor(other,device=dev)
    
    if other.device != dev:
        raise RuntimeError(f'Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!')
    
    # 前向计算：使用numpy的maximum
    value = arrlib.maximum(input._array, other._array)
    ret = TN(value, requires_grad=(is_grad_enabled() and (input.requires_grad or other.requires_grad)))
    
    # 注册梯度函数
    if ret.requires_grad:
        ret.fromvars = (input, other)
        ret.gradfuncs = (_maximum_backward_input, _maximum_backward_other)
    
    return ret

def _maximum_backward_input(result_tensor: TN, i: int) -> TN:
    """
    maximum函数对第一个输入的梯度计算
    
    梯度规则：
    - 当input > other时，梯度为1（传递梯度）
    - 当input < other时，梯度为0（阻断梯度）
    - 当input == other时，梯度为0.5（与PyTorch行为一致，平均分配）
    """
    input_tensor = result_tensor.fromvars[i]  # 第一个输入
    other_tensor = result_tensor.fromvars[i+1]  # 第二个输入
    
    
    # 计算梯度掩码
    # input > other: 1.0
    # input == other: 0.5  # 与PyTorch一致
    # input < other: 0.0
    # 使用riemann的where函数，自动处理设备上下文
    mask = where(
        input_tensor > other_tensor, 
        1.0, 
        where(input_tensor < other_tensor, 0.0, 0.5)
    ).type(input_tensor.dtype)
    
    # 应用梯度掩码
    grad = result_tensor.grad_value * mask
    
    # 处理广播情况
    input_shape = input_tensor.shape
    result_shape = result_tensor.shape
    
    if input_shape != result_shape:
        # 获取需要缩减的广播轴
        broadcast_axes = _get_broadcast_axis(result_shape, input_shape)
        grad = sum(grad, dim=broadcast_axes, keepdim=False)._reshape(input_shape)
    
    return grad

def _maximum_backward_other(result_tensor: TN, i: int) -> TN:
    """
    maximum函数对第二个输入的梯度计算
    
    梯度规则：
    - 当other > input时，梯度为1（传递梯度）
    - 当other <= input时，梯度为0（阻断梯度）
    - 当other == input时，梯度为0.5（平均分配）
    """
    other_tensor = result_tensor.fromvars[i]  # 第二个输入
    input_tensor = result_tensor.fromvars[i-1]  # 第一个输入
    
    
    # 计算梯度掩码
    # other > input: 1.0
    # other == input: 0.5  # 与PyTorch一致
    # other < input: 0.0
    # 使用riemann的where函数，自动处理设备上下文
    mask = where(
        other_tensor > input_tensor, 
        1.0,
        where(other_tensor < input_tensor, 0.0, 0.5)
    ).type(other_tensor.dtype)
    
    # 应用梯度掩码
    grad = result_tensor.grad_value * mask
    
    # 处理广播情况
    other_shape = other_tensor.shape
    result_shape = result_tensor.shape
    
    if other_shape != result_shape:
        # 获取需要缩减的广播轴
        broadcast_axes = _get_broadcast_axis(result_shape, other_shape)
        grad = sum(grad, dim=broadcast_axes, keepdim=False)._reshape(other_shape)
    
    return grad

def minimum(input: TN, other: TN) -> TN:
    """
    计算两个张量的逐元素最小值
    
    参数:
        input (TN): 第一个输入张量
        other (TN): 第二个输入张量，可以是标量或与input广播兼容的张量
        
    返回:
        TN: 包含两个张量逐元素最小值的新张量
        
    异常:
        TypeError: 当输入不是TN张量时抛出
        
    示例:
        >>> a = rm.tensor([[1, 2], [3, 4]])
        >>> b = rm.tensor([[2, 1], [1, 5]])
        >>> rm.minimum(a, b)
        tensor([[1, 1],
                [1, 4]])
    """
    if not isinstance(input, TN):
        raise TypeError(f"Expected input to be TN tensor, got {type(input)}")
    
    arrlib = input._get_array_lib()
    dev = input.device

    if not isinstance(other, TN):
        other = as_tensor(other,device=dev)
    
    if other.device != dev:
        raise RuntimeError(f'Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!')
        
    # 前向计算：使用numpy的minimum
    value = arrlib.minimum(input._array, other._array)
    ret = TN(value, requires_grad=(is_grad_enabled() and (input.requires_grad or other.requires_grad)))
    
    # 注册梯度函数
    if ret.requires_grad:
        ret.fromvars = (input, other)
        ret.gradfuncs = (_minimum_grad_input, _minimum_grad_other)
    
    return ret

def _minimum_grad_input(result_tensor: TN, i: int) -> TN:
    """
    minimum函数对第一个输入的梯度计算
    
    梯度规则：
    - 当input < other时，梯度为1（传递梯度）
    - 当input >= other时，梯度为0（阻断梯度）
    - 当input == other时，梯度为0.5（平均分配）
    """
    input_tensor = result_tensor.fromvars[i]  # 第一个输入
    other_tensor = result_tensor.fromvars[i+1]  # 第二个输入
    
    # 计算梯度掩码
    # input < other: 1.0
    # input == other: 0.5  
    # input > other: 0.0
    # 使用riemann的where函数，自动处理设备上下文
    mask = where(
        input_tensor < other_tensor, 
        1.0, 
        where(input_tensor > other_tensor, 0.0, 0.5)
    ).type(input_tensor.dtype)
    
    # 应用梯度掩码
    grad = result_tensor.grad_value * mask
    
    # 处理广播情况
    input_shape = input_tensor.shape
    result_shape = result_tensor.shape
    
    if input_shape != result_shape:
        # 获取需要缩减的广播轴
        broadcast_axes = _get_broadcast_axis(result_shape, input_shape)
        grad = sum(grad, dim=broadcast_axes, keepdim=False)._reshape(input_shape)
    
    return grad

def _minimum_grad_other(result_tensor: TN, i: int) -> TN:
    """
    minimum函数对第二个输入的梯度计算
    
    梯度规则：
    - 当other < input时，梯度为1（传递梯度）
    - 当other >= input时，梯度为0（阻断梯度）
    - 当other == input时，梯度为0.5（平均分配）
    """
    other_tensor = result_tensor.fromvars[i]  # 第二个输入
    input_tensor = result_tensor.fromvars[i-1]  # 第一个输入
    
    # 计算梯度掩码
    # other < input: 1.0
    # other == input: 0.5
    # other > input: 0.0
    # 使用riemann的where函数，自动处理设备上下文
    mask = where(
        other_tensor < input_tensor, 
        1.0,
        where(other_tensor > input_tensor, 0.0, 0.5)
    ).type(other_tensor.dtype)
    
    # 应用梯度掩码
    grad = result_tensor.grad_value * mask
    
    # 处理广播情况
    other_shape = other_tensor.shape
    result_shape = result_tensor.shape
    
    if other_shape != result_shape:
        # 获取需要缩减的广播轴
        broadcast_axes = _get_broadcast_axis(result_shape, other_shape)
        grad = sum(grad, dim=broadcast_axes, keepdim=False)._reshape(other_shape)
    
    return grad

def _get_diagonal_len_start_pop(m:int, n:int, offset:int):
    ''' 根据矩阵长宽、对角线偏移值，计算对角线长度和起点位置的坐标'''

    if offset >= 0:
        # 主对角线以上
        new_n = n - offset
        new_n = new_n if new_n > 0 else 0
        diagonal_len = m if m < new_n else new_n
        start_row, start_col = 0, offset  # 计算起始索引
    else:
        # 主对角线以下
        new_m = m + offset
        new_m = new_m if new_m > 0 else 0
        diagonal_len = n if n < new_m else new_m
        start_row, start_col = -offset, 0 # 计算起始索引

    return diagonal_len,start_row,start_col

def diagonal(
    input: TN, 
    offset: int = 0, 
    dim1: int = 0, 
    dim2: int = 1
) -> TN:
    """
    从输入张量中提取对角线元素
    
    该函数从指定的两个维度之间提取对角线元素。对于高维张量，函数会保留
    其他维度，并在提取对角线的维度上生成一个新的张量，其形状为：
    将dim1和dim2替换为一个新维度，该维度的大小等于对角线长度。
    
    参数:
        input (TN): 输入张量，必须至少是2维的。如果是高维张量，
                   则在指定的两个维度间提取对角线。
        offset (int, 可选): 要考虑的对角线偏移量，默认为0（主对角线）。
            - 如果offset > 0，则提取主对角线以上的对角线
            - 如果offset < 0，则提取主对角线以下的对角线
            - 如果offset = 0，则提取主对角线
        dim1 (int, 可选): 相对于其提取对角线的第一个维度，默认为0。
            支持负索引（-1表示最后一个维度）。
        dim2 (int, 可选): 相对于其提取对角线的第二个维度，默认为1。
            支持负索引（-1表示最后一个维度）。
    
    返回:
        TN: 包含对角线元素的新张量。对于形状为(*, a, b, *)的输入张量，
            如果从dim1和dim2维度提取对角线，则输出形状为(*, 对角线长度, *)
    
    异常:
        TypeError: 当输入不是TN类型时
        ValueError: 当输入张量维度小于2，或dim1等于dim2时
        IndexError: 当指定的维度超出有效范围时
    
    示例:
        >>> x = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> diagonal(x)  # 提取主对角线
        tensor([1, 5, 9])
        >>> diagonal(x, offset=1)  # 提取主对角线以上的对角线
        tensor([2, 6])
        >>> diagonal(x, offset=-1)  # 提取主对角线以下的对角线
        tensor([4, 8])
        >>> y = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        >>> diagonal(y, dim1=1, dim2=2)  # 从高维张量中提取对角线
        tensor([[1, 4], [5, 8]])
    """
    # 检查输入是否为TN类型
    if not isinstance(input, TN):
        raise TypeError("Input must be a tensor")
    
    # 检查输入是否至少是2维的
    if input._array.ndim < 2:
        raise ValueError("Input tensor must be at least 2-dimensional")
    
    # 检查dim1和dim2是否有效
    if dim1 == dim2:
        raise ValueError(f"dim1 and dim2 cannot be the same value ({dim1})")
    
    # 处理负维度
    if dim1 < 0:
        dim1 = input._array.ndim + dim1
    if dim2 < 0:
        dim2 = input._array.ndim + dim2
    
    # 检查维度是否在有效范围内
    if dim1 < 0 or dim1 >= input._array.ndim:
        raise IndexError(f"Dimension out of range (expected to be in range of [-{input._array.ndim}, {input._array.ndim-1}], but got {dim1})")
    if dim2 < 0 or dim2 >= input._array.ndim:
        raise IndexError(f"Dimension out of range (expected to be in range of [-{input._array.ndim}, {input._array.ndim-1}], but got {dim2})")
    
    # 计算对角线长度
    m, n = input.shape[dim1], input.shape[dim2]
    diagonal_len, start1, start2 = _get_diagonal_len_start_pop(m,n,offset)
    
    # 如果对角线长度为0，返回空张量
    if diagonal_len == 0:
        output_shape = list(input.shape)
        output_shape.pop(builtins.max(dim1, dim2))
        output_shape[builtins.min(dim1, dim2)] = 0
        # 将列表转换为元组并解包传递给zeros函数
        return zeros(*tuple(output_shape), dtype=input.dtype, device=input.device, requires_grad=input.requires_grad)
    
    
    # 创建对角线索引
    indices = [slice(None)] * input.ndim
    
    # 创建对角线位置的索引数组（使用numpy即可，索引值不依赖于设备）
    idx = np.arange(diagonal_len, dtype=int)
    
    # 设置dim1和dim2维度的索引
    indices[dim1] = start1 + idx
    indices[dim2] = start2 + idx
    
    # 直接使用TN张量的索引操作获取对角线元素
    result = input[tuple(indices)]
    
    return result    
# end of diagonal

def diag(input: TN, offset: int = 0) -> TN:
    """
    提取张量的对角线元素或从1D张量创建对角矩阵。
    
    此函数有两种行为模式：
    1. 当输入是1D张量时：返回一个以输入为对角线元素的2D对角矩阵
    2. 当输入是2D或更高维张量时：返回指定对角线的元素，作为1D张量
    
    参数:
        input (TN): 输入张量。如果是1D张量，则创建对角矩阵；如果是2D或更高维张量，则提取对角线元素
        offset (int, 可选): 对角线偏移量，默认为0（主对角线）
            - 对于2D及以上输入：如果diagonal > 0，则提取主对角线以上的对角线；如果diagonal < 0，则提取主对角线以下的对角线
            - 对于1D输入：此参数被忽略，始终创建主对角矩阵
    
    返回:
        TN: 
            - 如果输入是1D张量，返回形状为(n, n)的2D对角矩阵
            - 如果输入是2D或更高维张量，返回形状为(...)的1D张量，包含指定对角线的元素
    
    异常:
        TypeError: 当输入不是TN类型时
    
    示例:
        >>> import riemann as rm
        >>> # 从1D张量创建对角矩阵
        >>> x = rm.array([1, 2, 3])
        >>> diag_x = rm.diag(x)
        >>> # diag_x 是: [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        >>> 
        >>> # 提取2D张量的对角线元素
        >>> y = rm.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> diag_y = rm.diag(y)
        >>> # diag_y 是: [1, 5, 9]
        >>> 
        >>> # 提取偏移对角线
        >>> diag_y_offset = rm.diag(y, diagonal=1)
        >>> # diag_y_offset 是: [2, 6]
    """
    # 检查输入是否为TN类型
    if not isinstance(input, TN):
        raise TypeError("Input must be a tensor")
    
    # 处理1D输入 - 创建对角矩阵
    if input.ndim == 1:
        n = input._array.shape[0]
        
        # 创建单位矩阵模板
        identity_tensor = eye(n, device = input.device, dtype = input.dtype)
        
        # 使用广播机制生成对角矩阵
        # 将输入张量扩展维度，然后与单位矩阵相乘
        result = input.unsqueeze(-1) * identity_tensor
        
        return result
    # 处理2D或更高维输入 - 提取对角线元素
    elif input.ndim == 2:
        return diagonal(input, offset=offset, dim1=0, dim2=1)
    else:
        raise RuntimeError('diag(): Supports 1D or 2D tensors. Got {input.ndim}D')

# end of diag

def fill_diagonal(input: TN, value, offset: int = 0, dim1: int = -2, dim2: int = -1) -> TN:
    """
    用指定值填充张量指定维度之间的对角线元素，返回新张量。
    
    参数:
        input (TN): 输入张量，必须至少是2维的
        value: 用于填充对角线的值，可以是标量或与对角线长度相同的1D张量
        offset (int, 可选): 对角线偏移量，默认为0（主对角线）
            - 如果offset > 0，则填充主对角线以上的对角线
            - 如果offset < 0，则填充主对角线以下的对角线
        dim1 (int, 可选): 要考虑的第一个维度，默认为-2（倒数第二个维度）
        dim2 (int, 可选): 要考虑的第二个维度，默认为-1（最后一个维度）
    
    返回:
        TN: 填充对角线后的新张量
    
    异常:
        TypeError: 当输入不是TN类型时
        ValueError: 当输入张量维度小于2，或dim1等于dim2时
        IndexError: 当指定的维度超出有效范围时
    """
    # 检查输入是否为TN类型
    if not isinstance(input, TN):
        raise TypeError("Input must be a tensor")
    
    # 检查输入是否至少是2维的
    if input.ndim < 2:
        raise ValueError("Input tensor must be at least 2-dimensional")
    
    # 处理负维度
    if dim1 < 0:
        dim1 = input.ndim + dim1
    if dim2 < 0:
        dim2 = input.ndim + dim2
    
    # 检查维度是否在有效范围内
    if dim1 < 0 or dim1 >= input.ndim:
        raise IndexError(f"Dimension1 out of range (expected to be in range of [-{input.ndim}, {input.ndim-1}], but got {dim1})")
    if dim2 < 0 or dim2 >= input.ndim:
        raise IndexError(f"Dimension2 out of range (expected to be in range of [-{input.ndim}, {input.ndim-1}], but got {dim2})")
    
    # 检查dim1和dim2是否不同
    if dim1 == dim2:
        raise ValueError(f"dim1 and dim2 cannot be the same value ({dim1})")
    
    m, n = input.shape[dim1], input.shape[dim2]
    diagonal_len, start1, start2 = _get_diagonal_len_start_pop(m,n,offset)
    
    # 如果对角线长度为0，则直接返回输入的副本
    if diagonal_len == 0:
        return input.clone()
    
    # 创建对角线索引
    indices = [slice(None)] * input.ndim
    
    # 创建对角线位置的索引数组（使用numpy即可，索引值不依赖于设备）
    idx = np.arange(diagonal_len, dtype=int)
    
    # 设置dim1和dim2维度的索引，与diagonal函数保持一致
    indices[dim1] = start1 + idx
    indices[dim2] = start2 + idx
 
    # 准备填充值
    if not isinstance(value, TN):
        fill_value = as_tensor(value, dtype = input.dtype, device=input.device)
    else:
        fill_value = value

    # 按索引赋值input张量，setat返回创建的新张量
    result = input.setat(tuple(indices), fill_value)    
    return result

# end of fill_diagonal

def fill_diagonal_(input: TN, value, offset: int = 0, dim1: int = -2, dim2: int = -1) -> TN:
    """
    用指定值原地填充张量指定维度之间的对角线元素，返回原张量。
    
    参数:
        input (TN): 输入张量，必须至少是2维的
        value: 用于填充对角线的值，可以是标量或与对角线长度相同的1D张量
        offset (int, 可选): 对角线偏移量，默认为0（主对角线）
            - 如果offset > 0，则填充主对角线以上的对角线
            - 如果offset < 0，则填充主对角线以下的对角线
        dim1 (int, 可选): 要考虑的第一个维度，默认为-2（倒数第二个维度）
        dim2 (int, 可选): 要考虑的第二个维度，默认为-1（最后一个维度）
    
    返回:
        TN: 原张量
    
    异常:
        TypeError: 当输入不是TN类型时
        ValueError: 当输入张量维度小于2，或dim1等于dim2时
        IndexError: 当指定的维度超出有效范围时
    """
    # 检查输入是否为TN类型
    if not isinstance(input, TN):
        raise TypeError("Input must be a tensor")
    
    # 检查输入是否至少是2维的
    if input.ndim < 2:
        raise ValueError("Input tensor must be at least 2-dimensional")
    
    # 处理负维度
    if dim1 < 0:
        dim1 = input.ndim + dim1
    if dim2 < 0:
        dim2 = input.ndim + dim2
    
    # 检查维度是否在有效范围内
    if dim1 < 0 or dim1 >= input.ndim:
        raise IndexError(f"Dimension1 out of range (expected to be in range of [-{input.ndim}, {input.ndim-1}], but got {dim1})")
    if dim2 < 0 or dim2 >= input.ndim:
        raise IndexError(f"Dimension2 out of range (expected to be in range of [-{input.ndim}, {input.ndim-1}], but got {dim2})")
    
    # 检查dim1和dim2是否不同
    if dim1 == dim2:
        raise ValueError(f"dim1 and dim2 cannot be the same value ({dim1})")
    
    m, n = input.shape[dim1], input.shape[dim2]
    diagonal_len, start1, start2 = _get_diagonal_len_start_pop(m,n,offset)
    
    # 如果对角线长度为0，则直接返回输入张量
    if diagonal_len == 0:
        return input
    
    # 创建对角线索引
    indices = [slice(None)] * input.ndim
    
    # 创建对角线位置的索引数组（使用numpy即可，索引值不依赖于设备）
    idx = np.arange(diagonal_len, dtype=int)
    
    # 设置dim1和dim2维度的索引，与diagonal函数保持一致
    indices[dim1] = start1 + idx
    indices[dim2] = start2 + idx
 
    # 准备填充值
    if not isinstance(value, TN):
        fill_value = as_tensor(value, dtype = input.dtype, device=input.device)
    else:
        fill_value = value

    # 按索引原地赋值input张量
    result = input.setat_(tuple(indices), fill_value)    
    return result

# end of fill_diagonal_

# 从批量1D张量生成批量对角矩阵的函数
def batch_diag(v):
    """从批量1D张量生成批量对角矩阵
    
    参数:
        v: 输入张量，形状为(*, n)，其中*表示任意数量的批处理维度
        
    返回:
        形状为(*, n, n)的对角矩阵，对角线元素为输入张量的值
    """
    # 验证输入
    if not isinstance(v, TN):
        raise TypeError(f"Input must be TN type, got {type(v)}")
    
    # 获取输入形状
    shape = v.shape
    n = shape[-1]  # 最后一维的大小
    
    # 创建单位矩阵模板，确保与输入张量的数据类型一致
    identity_tensor = eye(n, device=v.device, dtype=v.dtype)
    
    # 使用广播机制生成对角矩阵
    # 将输入张量扩展维度，然后与单位矩阵相乘
    # 形状变化：(*, n) -> (*, 1, n) -> (*, n, n)
    batch_eye = identity_tensor.expand(shape[:-1] + (n, n))
    v_expanded = v.unsqueeze(-2)  # 扩展为 (*, 1, n)
    
    # 逐元素相乘，利用广播机制
    result = v_expanded * batch_eye
    return result

# end of batch_diag

def equal(a:TN,b:TN)->bool:
    """
    检查两个张量是否相等，返回一个布尔值。
    
    参数:
        a (TN): 第一个张量
        b (TN): 第二个张量
        
    返回:
        bool: 如果两个张量形状相同且所有元素都相等，则返回True；否则返回False
    """
    # 检查参数是否为None
    if a is None or b is None:
        raise TypeError("equal(): arguments must be Tensor, not NoneType")
    
    # 检查参数是否为TN类型
    if not isinstance(a, TN) or not isinstance(b, TN):
        raise TypeError("equal(): arguments must be Tensor")
    
    # 检查形状是否相同
    if a.shape != b.shape:
        return False
    
    # 检查元素是否相等
    return (a == b).all().item()

def not_equal(a:TN,b:TN)->bool:
    """
    检查两个张量是否不相等，返回一个布尔值。
    
    参数:
        a (TN): 第一个张量
        b (TN): 第二个张量
        
    返回:
        bool: 如果两个张量形状不同或有任何元素不相等，则返回True；否则返回False
    """
    return not equal(a,b)

def allclose(a:TN,b:TN,rtol:float=1e-5, atol:float=1e-8, equal_nan:bool=False)->bool:
    """
    检查两个张量是否在给定的相对和绝对容差内接近相等，返回一个布尔值。
    
    参数:
        a (TN): 第一个张量
        b (TN): 第二个张量
        rtol (float, 可选): 相对容差，默认值为1e-5
        atol (float, 可选): 绝对容差，默认值为1e-8
        equal_nan (bool, 可选): 当为True时，将两个张量中的NaN视为相等，默认值为False
        
    返回:
        bool: 如果两个张量在给定容差内接近相等，则返回True；否则返回False
    """
    # 检查形状是否相同
    if a.shape != b.shape:
        return False
    
    arrlib = a._get_array_lib()
    if a.device != b.device:
        raise RuntimeError(f'Cannot compare tensors on different devices: {a.device} and {b.device}')
    
    # 获取numpy数组
    a_data = a._array
    b_data = b._array
    
    # 如果启用了equal_nan，需要特殊处理NaN值
    if equal_nan:
        # 找出a和b中的NaN位置
        a_nan = arrlib.isnan(a_data)
        b_nan = arrlib.isnan(b_data)
        
        # 检查NaN位置是否相同
        if not arrlib.array_equal(a_nan, b_nan):
            return False
        
        # 在非NaN位置检查接近相等
        mask = ~a_nan  # 非NaN位置的掩码
        if arrlib.any(mask):  # 如果有非NaN位置需要检查
            a_valid = a_data[mask]
            b_valid = b_data[mask]
            diff = arrlib.abs(a_valid - b_valid)
            close = ((diff <= atol + rtol * np.abs(a_valid)) & 
                    (diff <= atol + rtol * np.abs(b_valid)))
            if not arrlib.all(close):
                return False
        return True
    else:
        # 不特殊处理NaN，正常检查接近相等
        diff = abs(a - b)
        return ((diff <= atol + rtol * abs(a)) & 
                (diff <= atol + rtol * abs(b))).all().item()

def isinf(x:TN)->TN:
    """
    检测张量中的无穷大元素。
    
    参数:
        x: 输入张量
    
    返回:
        布尔类型张量，对应位置为True表示输入张量中该位置是无穷大
    """
    if not isinstance(x, TN):
        raise TypeError('x must be a tensor')
    
    arrlib = x._get_array_lib()
    value = arrlib.isinf(x._array)
    ret = as_tensor(value,device=x.device)
    
    return ret

def isnan(x:TN)->TN:
    """
    检测张量中的NaN（不是数字）元素。
    
    参数:
        x: 输入张量
    
    返回:
        布尔类型张量，对应位置为True表示输入张量中该位置是NaN
    """
    if not isinstance(x, TN):
        raise TypeError('x must be a tensor')
    
    arrlib = x._get_array_lib()
    value = arrlib.isnan(x._array)
    ret = as_tensor(value,device=x.device)
    
    return ret

def isreal(x:TN)->TN:
    """
    检测张量中的实数元素。

    参数:
        x: 输入张量

    返回:
        布尔类型张量，对应位置为True表示输入张量中该位置是实数
    """
    if not isinstance(x, TN):
        raise TypeError('x must be a tensor')

    arrlib = x._get_array_lib()
    value = arrlib.isreal(x._array)
    ret = as_tensor(value,device=x.device)

    return ret

def tril(input_tensor: TN, diagonal: int = 0) -> TN:
    """
    返回矩阵的下三角部分
    
    参数:
        input_tensor: 输入张量
        diagonal: 对角线偏移量，默认为0
                 diagonal > 0: 包括主对角线以上的diagonal条对角线
                 diagonal = 0: 仅包括主对角线
                 diagonal < 0: 仅包括主对角线以下的部分
    
    返回:
        包含输入矩阵下三角部分的新张量
    """
    if not isinstance(input_tensor, TN):
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input_tensor)}")
       
    # 前向计算
    arrlib = input_tensor._get_array_lib()
    data = arrlib.tril(input_tensor._array, k=diagonal)
    ret = TN(data, requires_grad = (is_grad_enabled() and input_tensor.requires_grad))
    
    # 注册梯度函数
    if ret.requires_grad:
        ret.fromvars = (input_tensor,)
        ret.parms = (diagonal,)
        ret.gradfuncs = (_tril_backward,)
    
    return ret

def _tril_backward(result_tensor: TN, i: int) -> TN:
    """
    tril函数的反向传播实现。
    下三角矩阵的梯度是保留结果梯度的下三角部分。
    """
    diagonal = result_tensor.parms[i]
    # 重要：使用tril函数获取梯度的下三角部分，而不是numpy.tril
    return tril(result_tensor.grad_value, diagonal)

def triu(input_tensor: TN, diagonal: int = 0) -> TN:
    """
    返回矩阵的上三角部分
    
    参数:
        input_tensor: 输入张量
        diagonal: 对角线偏移量，默认为0
                 diagonal > 0: 仅包括主对角线以上的部分
                 diagonal = 0: 仅包括主对角线
                 diagonal < 0: 包括主对角线以下的-diagonal条对角线
    
    返回:
        包含输入矩阵上三角部分的新张量
    """
    if not isinstance(input_tensor, TN):
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input_tensor)}")
    
    # 前向计算
    arrlib = input_tensor._get_array_lib()
    data = arrlib.triu(input_tensor._array, k=diagonal)
    ret = TN(data, requires_grad = (is_grad_enabled() and input_tensor.requires_grad))
    
    # 注册梯度函数
    if ret.requires_grad:
        ret.fromvars = (input_tensor,)
        ret.parms = (diagonal,)
        ret.gradfuncs = (_triu_backward,)
    
    return ret

def _triu_backward(result_tensor: TN, i: int) -> TN:
    """
    triu函数的反向传播实现。
    上三角矩阵的梯度是保留结果梯度的上三角部分。
    """
    diagonal = result_tensor.parms[i]
    # 重要：使用triu函数获取梯度的上三角部分，而不是numpy.triu
    return triu(result_tensor.grad_value, diagonal)

def floor(x:TN)->TN:
    """
    计算张量的下取整。

    返回:
        包含输入张量下取整的新张量
    """
    x = _as_tn(x)
    arrlib = x._get_array_lib()
    return as_tensor(arrlib.floor(x._array),device=x.device)

def round(x:TN)->TN:
    """
    计算张量的四舍五入取整。

    返回:
        包含输入张量四舍五入取整的新张量
    """
    x = _as_tn(x)
    return as_tensor(x._array.round(),device=x.device)

def _is_cupy_array(arr):
    """
    检查是否为 CuPy 数组
    """
    return cp is not None and isinstance(arr, cp.ndarray)

def _get_array_lib(arr):
    """
    获取数组对应的库（numpy 或 cupy）
    """
    return cp if _is_cupy_array(arr) else np

class _IndexDescriptor:
    """
    统一索引描述符类
    
    用于统一表示不同类型的索引（切片、数组、整数），便于进行索引合并计算。

    - LINEAR：等差数列索引，元素映射为 f(i) = start + i*step，
      其中 i ∈ [0, (stop-start)/step)；正负步长统一适用，合并时可直接做复合映射
    - LOOKUP：显式离散位置表 indices；布尔索引已转为 True 所在位置，
      负索引已按维度大小归一化
    - NONE：保留类型；实际 None（np.newaxis）在 _normalize_index 中返回 Python None，
      表示仅添加维度，不参与描述符合并
    
    属性:
        type: 索引类型（LINEAR、LOOKUP 或 NONE）
        params: 类型特定的参数字典
    
    示例:
        >>> # 表示切片 slice(2, 10, 2)
        >>> desc = _IndexDescriptor(_IndexDescriptor.LINEAR, start=2, stop=10, step=2)
        >>> 
        >>> # 表示数组索引 [0, 2, 4]
        >>> desc = _IndexDescriptor(_IndexDescriptor.LOOKUP, indices=np.array([0, 2, 4]))
    """
    
    LINEAR = 'LINEAR'  # 线性切片索引，如 slice(start, stop, step)
    LOOKUP = 'LOOKUP'  # 离散索引，如 [0, 2, 4] 或 np.array([0, 2, 4])
    NONE = 'NONE'      # None 索引（np.newaxis）
    
    __slots__ = ('type', 'params')
    
    def __init__(self, mapping_type: str, **params):
        """
        初始化索引描述符
        
        Args:
            mapping_type: 索引类型（LINEAR、LOOKUP 或 NONE）
            **params: 类型特定的参数
                - LINEAR: start, stop, step
                - LOOKUP: indices（numpy 数组）
        """
        self.type = mapping_type
        self.params = params
    
    def __repr__(self):
        """返回描述符的字符串表示"""
        if self.type == _IndexDescriptor.LINEAR:
            return f"LINEAR(start={self.params['start']}, stop={self.params['stop']}, step={self.params['step']})"
        elif self.type == _IndexDescriptor.LOOKUP:
            return f"LOOKUP(indices={self.params['indices']})"
        else:
            return f"NONE()"


def _normalize_index_to_tuple(index):
    """将单个索引或索引序列统一标准化为元组。

    merge_indices 内部统一用元组表示索引，避免后续按位置取值时对
    int/slice/list 等类型做分支判断。规则：
      - 已是 tuple：原样返回
      - 单个索引（slice/int/list/np.ndarray/cupy 数组）：包装为单元素元组
      - 其他可迭代对象：tuple(index)

    注意：None 索引会由调用方先转换为 (None,)，因此这里不单独处理 None。
    """
    if isinstance(index, tuple):
        return index
    elif index is Ellipsis:
        # 单省略号包装成元组，交由后续 _expand_ellipsis 按目标维度数展开
        return (Ellipsis,)
    elif isinstance(index, (slice, int, list, np.ndarray)) or _is_cupy_array(index):
        return (index,)
    else:
        return tuple(index)


def _expand_ellipsis(index, target_ndim):
    """展开索引中的 Ellipsis（...）为适当数量的 slice(None)。

    语义与 NumPy/PyTorch 保持一致：
      - target_ndim 表示“需要被索引的维度数”（对 current_index 而言是视图维度数）
      - None（np.newaxis）只添加维度，不消耗维度，因此不计入 num_other
      - 显式非 None 索引超过 target_ndim 时，Ellipsis 展开为 0 个切片
      - 不含 Ellipsis 时原样返回；长度补全由调用方按需处理
    """
    # 标准化为tuple
    if not isinstance(index, tuple):
        if index is Ellipsis:
            return (slice(None),) * target_ndim
        return (index,)
    
    # 检查是否包含 Ellipsis（使用显式循环避免数组索引的歧义）
    ellipsis_idx = -1
    for i, idx in enumerate(index):
        if idx is Ellipsis:
            ellipsis_idx = i
            break
    
    if ellipsis_idx == -1:
        # 没有 Ellipsis，直接返回
        return index
    
    # 计算需要填充的 slice(None) 数量（None 仅添加维度，不消耗维度）
    num_other = len([idx for idx in index if idx is not Ellipsis and idx is not None])
    num_slices_needed = target_ndim - num_other
    
    if num_slices_needed < 0:
        # Ellipsis 可以表示0个切片
        return index[:ellipsis_idx] + index[ellipsis_idx+1:]
    
    # 展开 Ellipsis
    return (index[:ellipsis_idx] + 
            (slice(None),) * num_slices_needed + 
            index[ellipsis_idx+1:])


def _try_merge_special_base_index(base_index, current_index):
    """尝试快速合并几类特殊的 base_index，无法直接合并时返回 None。

    可快速处理的场景：
      1. base_index 为空元组 ()：表示直接索引原始张量，无需合并，返回 current_index
      2. base_index 为 (None,)：视图仅由一个 np.newaxis 构成。
         对常见的单元素 current_index（int/None/slice）给出等价索引，
         其他情况保留 None 并追加 current_index
      3. base_index 全为整数索引：视图没有剩余维度，
         将整数索引与 current_index 直接拼接即可

    Returns:
        tuple 或 None：合并结果；返回 None 表示需要走通用分阶段合并流程。
    """
    # 处理 base_index 为空元组的情况（直接索引原始张量）
    # 注意：current_index 不会是 None，因为 _convert_TNindex_to_numpy 已将 None 转换为 (None,)
    if isinstance(base_index, tuple) and len(base_index) == 0:
        return current_index
    
    # 处理 (None,) 形式的 base_index（通过 np.newaxis 创建的视图）
    if isinstance(base_index, tuple) and len(base_index) == 1 and base_index[0] is None:
        # 标准化 current_index 为元组（current_index 不会是 None，因为 _convert_TNindex_to_numpy 已转换）
        curr_tuple = current_index if isinstance(current_index, tuple) else (current_index,)
        
        if len(curr_tuple) == 1:
            curr_idx = curr_tuple[0]
            if isinstance(curr_idx, int):
                # x[None][0] -> 选取新维度的第0个元素，结果是原始数据的全部
                return (slice(None),)
            elif curr_idx is None:
                # x[None][None] -> 添加两个新维度
                return (None, None)
            elif isinstance(curr_idx, slice):
                # x[None][:] -> 保留新维度，选取全部
                return (None, curr_idx)
        
        # 其他情况，保留 None 并追加 current_index
        return (None,) + curr_tuple
    
    # 标准化 base_index 为元组
    base_tuple = _normalize_index_to_tuple(base_index)
    
    # 检查是否全是整数索引（无视图维度）
    if all(isinstance(idx, int) for idx in base_tuple):
        if isinstance(current_index, int) and len(base_tuple) == 1:
            # 单整数索引链式访问，如 x[5][3]
            return (base_tuple[0], current_index)
        elif isinstance(current_index, tuple):
            # 多索引链式访问，如 x[5][1:3]
            return base_tuple + current_index
        return base_tuple
    
    # 需要进一步处理
    return None


def _preprocess_base_index(base_index, base_shape):
    """预处理基础索引，返回 (base_tuple, view_dims)。

    - base_tuple：标准化后的元组索引，Ellipsis 已展开。这里刻意不补齐长度，
      因为 base_index 可能只索引原始张量的前若干个维度（尾部维度隐式全切片）
    - view_dims：真正构成视图维度的位置列表。只有切片、数组等索引会保留维度；
      整数索引会把该维度消去，None 只添加新维度，二者都不计入 view_dims

    例：
        base_index=(slice(2, 8), 1)    -> view_dims=[0]
        base_index=(None, slice(2, 8)) -> view_dims=[1]
    """
    # 标准化为元组格式
    base_tuple = _normalize_index_to_tuple(base_index)
    
    # 计算基础张量的维度数
    base_ndim = len(base_shape)
    
    # 展开 Ellipsis（不填充长度，因为 base_index 可能只索引部分维度）
    base_tuple = _expand_ellipsis(base_tuple, base_ndim)
    
    # 计算 view_dims（非整数索引的位置）
    view_dims = [i for i, idx in enumerate(base_tuple)
                 if not isinstance(idx, int) and idx is not None]
    
    return base_tuple, view_dims


def _preprocess_current_index(current_index, view_dims):
    """预处理当前索引，使其与视图维度一一对应。

    - 标准化为元组，并按视图维度数 len(view_dims) 展开 Ellipsis
    - 只统计非 None 元素：None 不消耗视图维度
    - 非 None 元素不足时在末尾补 slice(None)，等价于 Python 的隐式全切片

    返回的元组仍保留 None 的位置，供 merge_indices 在组装阶段插入新维度。
    """
    result = _normalize_index_to_tuple(current_index)
    
    ndim = len(view_dims)
    
    # 展开 Ellipsis
    result = _expand_ellipsis(result, ndim)
    
    # 填充长度以匹配 view_dims（None 不消耗视图维度，不计入缺失数量）
    non_none_count = len([idx for idx in result if idx is not None])
    if non_none_count < ndim:
        result = result + (slice(None),) * (ndim - non_none_count)
    
    return result


def _normalize_index(index, dim_size: int):
    """把单个索引转换为 _IndexDescriptor（统一表示线性切片/离散索引）。

    - slice：正步长时 start/stop 缺省为 0/dim_size；负步长时 start 缺省为
      dim_size-1、stop 缺省为 -dim_size-1（表示越过首元素）；负索引按 dim_size 归一化
    - int：转为单元素 LOOKUP（负整数先加 dim_size）
    - list/ndarray/cupy：布尔数组转为 True 所在位置的整数数组；
      整数数组中的负数按 dim_size 归一化
    - None：返回 None（np.newaxis 不参与描述符运算）

    Args:
        index: 单个索引元素
        dim_size: 该维度在“被索引张量”中的大小（合并视图维度时为视图大小）
    """
    
    if isinstance(index, slice):
        # 处理切片索引
        step = index.step if index.step is not None else 1
        
        # 处理负步长切片
        # 对于负步长，start 默认为 dim_size-1，stop 默认为 -dim_size-1（在第一个元素之前）
        if step < 0:
            start = index.start if index.start is not None else dim_size - 1
            # 对于负步长，stop 默认为 -dim_size-1（比任何有效索引都小）
            stop = index.stop if index.stop is not None else -dim_size - 1
        else:
            start = index.start if index.start is not None else 0
            stop = index.stop if index.stop is not None else dim_size
        
        # 处理负数索引（仅当不是 None 转换的默认值时）
        if start < 0:
            start += dim_size
        # 对于负步长，stop 可以是负数（表示在数组开头之前停止）
        if stop < 0 and step > 0:
            stop += dim_size
        
        return _IndexDescriptor(_IndexDescriptor.LINEAR, start=start, stop=stop, step=step)
    
    elif isinstance(index, (int, np.integer)):
        # 处理整数索引，转换为单元素 LOOKUP
        idx = int(index if index >= 0 else index + dim_size)
        return _IndexDescriptor(_IndexDescriptor.LOOKUP, indices=np.array([idx]))
    
    elif isinstance(index, (list, np.ndarray)) or _is_cupy_array(index):
        # 处理列表、NumPy 数组或 CuPy 数组索引
        is_cupy = _is_cupy_array(index)
        if is_cupy:
            # CuPy 数组：保持 CuPy 类型，使用 cupy 的函数
            indices = index
            lib = cp
        else:
            # NumPy 数组或列表：转换为 NumPy 数组
            indices = np.asarray(index)
            lib = np
        
        # 处理布尔数组索引
        # 布尔数组需要转换为整数索引（True 元素的位置）
        if indices.dtype == bool:
            # 将布尔数组转换为整数索引数组
            int_indices = lib.where(indices)[0]
            return _IndexDescriptor(_IndexDescriptor.LOOKUP, indices=int_indices)
        
        # 处理负索引（仅适用于整数数组）
        if np.issubdtype(indices.dtype, np.integer):
            indices = lib.where(indices < 0, indices + dim_size, indices)
        return _IndexDescriptor(_IndexDescriptor.LOOKUP, indices=indices)
    
    elif index is Ellipsis:
        # Ellipsis (...) 应该在 _expand_ellipsis 中处理
        # 如果到达这里，说明 Ellipsis 没有被正确展开
        raise ValueError("Ellipsis (...) 应该在 _expand_ellipsis 中处理，不应该到达 normalize_index")

    elif index is None:
        # None (np.newaxis) 作为特殊标记返回
        return None

    else:
        raise ValueError(f"不支持的索引类型: {type(index)}")


def _merge_descriptors(base_desc: _IndexDescriptor, current_desc: _IndexDescriptor) -> _IndexDescriptor:
    """合并两个索引描述符，得到“在父张量上一步索引”的等价描述符。

    参数含义：
        base_desc:    视图维度到父维度的映射（由 base_index 标准化而来）
        current_desc: 当前索引在该视图维度上的映射（dim_size 为视图大小）

    支持四种组合（公式与推导见各分支注释）：
        1. LINEAR + LINEAR = LINEAR：两个切片复合，新的起点/步长/终点按线性映射计算
        2. LINEAR + LOOKUP = LOOKUP：离散索引经线性映射换算到父维度位置
        3. LOOKUP + LINEAR = LOOKUP：从查找表中截取一段子区间
        4. LOOKUP + LOOKUP = LOOKUP：用第二个查找表从第一个查找表中挑选元素

    Returns:
        合并后的 _IndexDescriptor，可直接交给 _descriptor_to_index 还原为索引对象。
    """
    
    # 情况1: LINEAR + LINEAR = LINEAR
    # 组合两个切片，计算新的 start, stop, step
    # 
    # 数学公式:
    #   设基础切片: A = slice(a_start, a_stop, a_step)
    #   设当前切片: C = slice(c_start, c_stop, c_step)
    #   
    #   则合并后的切片:
    #     new_start = a_start + c_start * a_step
    #     new_step  = a_step * c_step
    #     new_stop  = min(a_start + c_stop * a_step, a_stop)
    #
    # 推导过程:
    #   基础切片映射: f(i) = a_start + i * a_step, 其中 i ∈ [0, (a_stop-a_start)/a_step)
    #   当前切片映射: g(j) = c_start + j * c_step, 其中 j ∈ [0, (c_stop-c_start)/c_step)
    #   
    #   合并映射: h(j) = f(g(j)) = a_start + (c_start + j * c_step) * a_step
    #                    = (a_start + c_start * a_step) + j * (a_step * c_step)
    #   
    #   因此:
    #     新起点 = a_start + c_start * a_step
    #     新步长 = a_step * c_step
    #     新终点 = min(a_start + c_stop * a_step, a_stop)  # 不能超过基础切片的终点
    #
    # 示例:
    #   base = slice(2, 8, 1), current = slice(1, 3, 1)
    #   new_start = 2 + 1*1 = 3
    #   new_step  = 1*1 = 1
    #   new_stop  = min(2 + 3*1, 8) = 5
    #   result = slice(3, 5, 1)  # 即 arr[2:8][1:3] == arr[3:5]
    if base_desc.type == _IndexDescriptor.LINEAR and current_desc.type == _IndexDescriptor.LINEAR:
        a_start, a_stop, a_step = base_desc.params['start'], base_desc.params['stop'], base_desc.params['step']
        c_start, c_stop, c_step = current_desc.params['start'], current_desc.params['stop'], current_desc.params['step']
        
        # 处理负步长（反转切片）
        # 对于负步长，使用数学公式直接计算合并后的切片参数
        # 
        # 数学公式（适用于正负步长）:
        #   设基础切片: A = slice(a_start, a_stop, a_step)
        #   设当前切片: C = slice(c_start, c_stop, c_step)
        #   
        #   则合并后的切片:
        #     new_start = a_start + c_start * a_step
        #     new_step  = a_step * c_step
        #     new_stop  = a_start + c_stop * a_step
        #
        # 注意: 对于负步长，不需要取 min，因为负步长的 stop 本身就可能小于 start
        #
        # 示例（负步长）:
        #   base = slice(9, -1, -1)  # 表示 [9,8,7,6,5,4,3,2,1,0]
        #   current = slice(1, 4, 1)  # 选取第1到第4个元素
        #   new_start = 9 + 1*(-1) = 8
        #   new_step  = (-1)*1 = -1
        #   new_stop  = 9 + 4*(-1) = 5
        #   result = slice(8, 5, -1)  # 表示 [8,7,6]
        if a_step < 0:
            new_start = a_start + c_start * a_step
            new_step = a_step * c_step
            new_stop = a_start + c_stop * a_step
            return _IndexDescriptor(_IndexDescriptor.LINEAR, start=new_start, stop=new_stop, step=new_step)
        
        new_start = a_start + c_start * a_step
        new_step = a_step * c_step
        new_stop = builtins.min(a_start + c_stop * a_step, a_stop)
        
        return _IndexDescriptor(_IndexDescriptor.LINEAR, start=new_start, stop=new_stop, step=new_step)
    
    # 情况2: LINEAR + LOOKUP = LOOKUP
    # 切片后离散索引，应用线性映射
    #
    # 数学公式:
    #   设基础切片: A = slice(start, stop, step)
    #   设当前索引: C = [c_0, c_1, ..., c_n]
    #   
    #   则合并后的索引:
    #     mapped[i] = start + c_i * step
    #
    # 推导过程:
    #   基础切片映射: f(i) = start + i * step
    #   离散索引直接选取: c_i 表示在视图中的第 c_i 个元素
    #   
    #   合并映射: mapped[i] = f(c_i) = start + c_i * step
    #
    # 示例:
    #   base = slice(2, 8, 2), current = [0, 2]
    #   mapped[0] = 2 + 0*2 = 2
    #   mapped[1] = 2 + 2*2 = 6
    #   result = [2, 6]  # 即 arr[2:8:2][[0,2]] == arr[[2,6]]
    elif base_desc.type == _IndexDescriptor.LINEAR and current_desc.type == _IndexDescriptor.LOOKUP:
        start = base_desc.params['start']
        step = base_desc.params['step']
        indices = current_desc.params['indices']
        
        mapped = start + indices * step
        return _IndexDescriptor(_IndexDescriptor.LOOKUP, indices=mapped)
    
    # 情况3: LOOKUP + LINEAR = LOOKUP
    # 离散索引后切片，从查找表中选择子范围
    #
    # 数学公式:
    #   设基础查找表: T = [t_0, t_1, ..., t_m]
    #   设当前切片: C = slice(start, stop, step)
    #   
    #   则合并后的索引:
    #     indices = [start, start+step, start+2*step, ..., min(stop, m)-1]
    #     mapped[i] = T[indices[i]]
    #
    # 推导过程:
    #   基础查找表: T[i] 表示父张量中的第 T[i] 个元素
    #   切片映射: 从查找表中选取索引范围 [start, stop)，步长为 step
    #   
    #   合并映射: mapped[j] = T[start + j * step], 其中 j 满足 start + j*step < min(stop, m)
    #
    # 示例:
    #   base = [2, 4, 6, 8], current = slice(1, 3, 1)
    #   indices = [1, 2]  # 从查找表中选取第1和第2个元素
    #   mapped = [base[1], base[2]] = [4, 6]
    #   result = [4, 6]  # 即 arr[[2,4,6,8]][1:3] == arr[[4,6]]
    elif base_desc.type == _IndexDescriptor.LOOKUP and current_desc.type == _IndexDescriptor.LINEAR:
        table = base_desc.params['indices']
        start = current_desc.params['start']
        stop = current_desc.params['stop']
        step = current_desc.params['step']
        
        # 根据 table 的类型选择库
        lib = _get_array_lib(table)

        actual_stop = builtins.min(stop, table.size)
        indices = lib.arange(start, actual_stop, step)
        
        mapped = table[indices]
        return _IndexDescriptor(_IndexDescriptor.LOOKUP, indices=mapped)
    
    # 情况4: LOOKUP + LOOKUP = LOOKUP
    # 两次离散索引，使用第二次索引从第一次的查找表中选择元素
    #
    # 数学公式:
    #   设基础查找表: T1 = [a_0, a_1, ..., a_m]
    #   设当前查找表: T2 = [b_0, b_1, ..., b_n]
    #   
    #   则合并后的索引:
    #     valid_indices = {b_i | 0 <= b_i < m}
    #     mapped[i] = T1[T2[i]]  # 仅对有效的 T2[i]
    #
    # 推导过程:
    #   第一次查找: T1[i] 表示父张量中的第 T1[i] 个元素
    #   第二次查找: T2[j] 表示在 T1 中的第 T2[j] 个元素
    #   
    #   合并映射: mapped[j] = T1[T2[j]], 需要满足 0 <= T2[j] < len(T1)
    #   
    #   注意: 如果 T2[j] 超出范围，则跳过该索引（通过 valid_mask 过滤）
    #
    # 示例:
    #   base = [2, 4, 6, 8], current = [0, 2]
    #   mapped[0] = base[0] = 2
    #   mapped[1] = base[2] = 6
    #   result = [2, 6]  # 即 arr[[2,4,6,8]][[0,2]] == arr[[2,6]]
    elif base_desc.type == _IndexDescriptor.LOOKUP and current_desc.type == _IndexDescriptor.LOOKUP:
        table1 = base_desc.params['indices']
        table2 = current_desc.params['indices']
        
        # 确定使用哪个数组库（根据 table2 的类型，因为结果应该与 table2 同类型）
        lib = _get_array_lib(table2)
        
        # 如果 table1 和 table2 类型不同，需要转换 table1
        if _is_cupy_array(table1) and not _is_cupy_array(table2):
            # table1 是 CuPy，table2 是 NumPy：将 table1 转换为 NumPy
            table1 = table1.get()
        elif not _is_cupy_array(table1) and _is_cupy_array(table2):
            # table1 是 NumPy，table2 是 CuPy：将 table1 转换到 table2 所在设备
            with table2.device:
                table1 = cp.asarray(table1)

        table1_size = table1.size
        valid_mask = (table2 >= 0) & (table2 < table1_size)
        valid_indices = table2[valid_mask]
        mapped = table1[valid_indices]
        return _IndexDescriptor(_IndexDescriptor.LOOKUP, indices=mapped)
    
    else:
        raise ValueError(f"未知的描述符类型组合: {base_desc.type}, {current_desc.type}")


def _descriptor_to_index(desc: _IndexDescriptor, original_type: type = None, keep_step_none: bool = False):
    """把合并后的 _IndexDescriptor 转回可直接用于张量索引的 Python 对象。

    Args:
        desc: 合并结果描述符（LINEAR/LOOKUP/NONE）
        original_type: 当前索引的原始类型；当 LOOKUP 只含一个元素且原始为 int 时，
            转回 int 而不是数组，以保持索引语义一致
        keep_step_none: 为 True 且 step==1 时，输出 slice(start, stop, None)，
            保持与外部书写的切片形式一致（便于比较/打印）
    """
    if desc.type == _IndexDescriptor.LINEAR:
        start = desc.params['start']
        stop = desc.params['stop']
        step = desc.params['step']
        # 如果 step 为 1 且 keep_step_none 为 True，返回 None（与原版一致）
        if keep_step_none and step == 1:
            step = None
        return slice(start, stop, step)
    elif desc.type == _IndexDescriptor.LOOKUP:
        indices = desc.params['indices']
        if len(indices) == 1 and original_type == int:
            return int(indices[0])
        return indices
    elif desc.type == _IndexDescriptor.NONE:
        return None
    else:
        raise ValueError(f"未知的描述符类型: {desc.type}")

def merge_indices(self, base_index, current_index):
    """
    基于 _IndexDescriptor 的索引合并函数

    将视图索引链（base_index 和 current_index）合并为在原始张量上的等效索引。
    这是原地操作的核心函数，用于确定需要修改原始张量的哪些元素。

    核心思想：
        视图中每个保留的维度都能写成父维度上的线性映射（切片）或离散映射（数组）；
        当前索引在该视图维度上再做一次选择，两次选择可以复合为一次映射
        （详见 _merge_descriptors）。整数维度已被消去，None 维度只添加坐标轴，
        二者都不参与复合运算，最后再按原始维度顺序组装结果。

    职责分层设计：
    ----------------
    本函数采用分层架构，各阶段职责清晰：

    第一阶段 - 尝试处理特殊情况：
        调用 _try_merge_special_base_index 处理可以快速合并的情况：
        - base_index 为空元组（直接索引原始张量）
        - base_index 为 (None,)（np.newaxis 创建的视图）
        - base_index 全是整数索引（链式索引）
        如果可以合并，直接返回结果，跳过后续阶段。
        
        注意：base_index 不会为 None，因为 _convert_TNindex_to_numpy 已将 None 转换为 (None,)

    第二阶段 - 预处理 base_index：
        调用 _preprocess_base_index 进行标准化：
        - 将各种格式（slice、int、tuple 等）统一转换为元组格式
        - 展开 Ellipsis（...），但不填充长度（因为 base_index 可能只索引部分维度）
        - 计算 view_dims（真正构成视图维度的位置：切片/数组等，不含整数和 None）

    第三阶段 - 预处理 current_index：
        调用 _preprocess_current_index 进行标准化：
        - 转换为元组格式
        - 展开 Ellipsis
        - 仅对非 None 元素补齐长度（None 不占视图维度），末尾补 slice(None)

    第四阶段 - 逐个视图维度合并索引：
        将 current_index 中的非 None 元素与 view_dims 一一对应：调用 _normalize_index
        转为 _IndexDescriptor，用 _merge_descriptors 合并，再经 _descriptor_to_index 转回原始格式。
        该阶段只计算“每个视图维度映射到原始张量的哪个位置”，不处理 None。

    第五阶段 - 组装结果：
        按 base_index 的维度顺序输出结果：整数/None 维度原样保留，视图维度输出其合并结果；
        并在第 j 个视图维度之前插入 current_index 中位于第 j 个非 None 元素之前的 None。

    支持的索引类型：
    ----------------
    - 切片索引：slice(start, stop, step)，支持正负步长
    - 整数索引：int，支持负索引
    - 数组索引：list、np.ndarray、cp.ndarray，支持布尔数组
    - 省略号：Ellipsis（...）自动展开
    - None 索引：np.newaxis，仅添加维度，不消耗视图维度

    关键规则（None 的处理）：
    ----------------
    - None 不消耗视图维度，因此 current_index 的非 None 元素按顺序映射到 view_dims；
      非 None 元素不足 view_dims 数量时，末尾用 slice(None) 补齐（等价于 Python 隐式全切片）。
    - 合并结果是用于索引“原始张量”的，所以 None 新维度按其在 current_index 中的
      相对位置插入：位于第 j 个非 None 元素之前的 None，插入到第 j 个视图维度之前；
      j 等于视图维度数时追加到结果末尾。
    - base_index 中的整数维度、None 维度均不参与合并，原样保留在结果中。

    Args:
        base_index: 基础索引，用于创建视图。可以是 slice、int、tuple、list 或 np.ndarray
        current_index: 当前索引，用于在视图上进行索引。可以是 slice、int、tuple 等

    Returns:
        tuple: 合并后的索引元组，可以直接用于原始张量的索引

    示例:
        >>> # 基本切片合并：x[2:8][1:3] 等价于 x[3:5]
        >>> merge_indices(self, slice(2, 8), slice(1, 3))
        (slice(3, 5, None),)
        >>>
        >>> # 负步长切片：x[::-1][1:4] 在 1D 张量 [0..9] 上
        >>> merge_indices(self, slice(None, None, -1), slice(1, 4))
        (slice(8, 5, -1),)  # 选取元素 [8, 7, 6]
        >>>
        >>> # 多维索引合并：x[2:8, 3:7][1:3, 0:2] 等价于 x[3:5, 3:5]
        >>> merge_indices(self, (slice(2, 8), slice(3, 7)), (slice(1, 3), slice(0, 2)))
        (slice(3, 5, None), slice(3, 5, None))
        >>>
        >>> # 数组索引合并：x[[0,2,4]][[0,2]] 等价于 x[[0,4]]
        >>> merge_indices(self, [0, 2, 4], [0, 2])
        (array([0, 4]),)
        >>>
        >>> # np.newaxis 特殊情况：x[None][0] 等价于 x[:]
        >>> merge_indices(self, (None,), (0,))
        (slice(None),)
        >>>
        >>> # 在视图上新增维度：x[2:8, 3:7][None, 1:3] 等价于 x[None, 3:5, 3:7]
        >>> merge_indices(self, (slice(2, 8), slice(3, 7)), (None, slice(1, 3)))
        (None, slice(3, 5, None), slice(3, 7, None))
    """
    
    # 第一阶段：尝试处理特殊情况    
    result = _try_merge_special_base_index(base_index, current_index)
    if result is not None:
        return result
    
    # 提前获取基础张量形状，后续多个阶段复用
    base_shape = self._base.shape if (hasattr(self, '_base') and self._base is not None) else self.shape
    
    # 第二阶段：预处理 base_index    
    base_index, view_dims = _preprocess_base_index(base_index, base_shape)
    
    # 视图由部分索引创建时（尾部维度隐式全切片），base_index 不含这些维度。
    # 若当前索引显式作用于这些尾部维度（如 v[..., 1]，或 current 元素数超过
    # view_dims 数量），需先把 base_index 补齐为全维，否则省略号展开的维度数
    # 与视图实际维度不一致，合并时会把索引维度丢掉。
    if len(base_index) < len(base_shape):
        curr_tuple = _normalize_index_to_tuple(current_index)
        non_none = [idx for idx in curr_tuple if idx is not None]
        has_ellipsis = any(idx is Ellipsis for idx in curr_tuple)
        view_ndim_no_newaxis = self.ndim - len([idx for idx in base_index if idx is None])
        if len(non_none) > len(view_dims) or (has_ellipsis and view_ndim_no_newaxis > len(view_dims)):
            base_index = base_index + (slice(None),) * (len(base_shape) - len(base_index))
            view_dims = [i for i, idx in enumerate(base_index)
                         if not isinstance(idx, int) and idx is not None]
    
    # 第三阶段：预处理 current_index    
    current_index = _preprocess_current_index(current_index, view_dims)
    
    # 第四阶段：逐个维度合并索引
    merged_dims = {}  # 视图维度位置 -> 合并后的索引
    current_non_none = [idx for idx in current_index if idx is not None]
    
    for i, view_dim in enumerate(view_dims):
        base_idx = base_index[view_dim]
        curr_idx = current_non_none[i]
        
        # 第 i 个非 None 索引作用于第 i 个视图维度 view_dims[i]；
        # None 已从当前索引序列中剔除，留到第五阶段按相对位置插入
        
        # 获取该视图维度在原始张量中对应维度的大小（用于解析负索引/负步长）
        dim_size = base_shape[view_dim] if view_dim < len(base_shape) else 1
        
        # 标准化基础索引为 IndexDescriptor
        base_desc = _normalize_index(base_idx, dim_size)

        # 防御性处理：base_tuple 中若残留 None（理论上已被 view_dims 排除），保持原样
        if base_desc is None:
            merged_dims[view_dim] = base_idx
            continue

        # 计算视图大小（该维度在视图中的元素数量）
        if base_desc.type == _IndexDescriptor.LINEAR:
            p = base_desc.params
            view_size = (p['stop'] - p['start'] + p['step'] - 1) // p['step']
        else:
            view_size = base_desc.params['indices'].size
        
        # 标准化当前索引、合并、转回原始格式
        current_desc = _normalize_index(curr_idx, view_size)
        merged = _merge_descriptors(base_desc, current_desc)
        merged_dims[view_dim] = _descriptor_to_index(merged, type(curr_idx), keep_step_none=True)
    
    # 第五阶段：组装结果
    # 统计每个 None 前面有多少个非 None 元素：none_before[j] 表示需要插到
    # 第 j 个视图维度之前的 None 数量；j == 视图维度数时表示追加到末尾
    none_before = {}
    seen = 0
    for idx in current_index:
        if idx is None:
            none_before[seen] = none_before.get(seen, 0) + 1
        else:
            seen += 1

    # 按 base_index 的维度顺序输出，保证整数维度（已被消去）和原始维度顺序不变
    result = []
    ordinal = 0  # 已处理的视图维度序号，用于查 none_before
    for pos, base_idx in enumerate(base_index):
        if pos in merged_dims:
            result.extend([None] * none_before.get(ordinal, 0))
            result.append(merged_dims[pos])
            ordinal += 1
        else:
            result.append(base_idx)
    result.extend([None] * none_before.get(ordinal, 0))

    return tuple(result)


