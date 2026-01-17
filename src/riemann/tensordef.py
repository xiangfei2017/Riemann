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
from itertools import accumulate
import builtins
import warnings
from typing import Callable, Any, List, Tuple, TypeAlias, overload, Union, Optional
import math
import numpy as np
from .cuda import Device, CUPY_AVAILABLE, cp, is_in_cuda_context, get_default_device, current_device
from .dtype import *
from .gradmode import *


class TN:
    """
    张量(Tensor)类是Riemann框架的核心数据结构。
    
    TN(Tensor Numerics)类将NumPy多维数组封装为张量，并在数学运算过程中自动构建计算图，
    从而支持自动微分(自动求导)功能，是构建神经网络和实现深度学习算法的基础。
    
    主要特性：
    - 封装NumPy多维数组作为底层数据存储
    - 自动构建动态计算图，追踪操作历史
    - 支持反向传播算法，自动计算梯度
    - 提供丰富的数学运算和函数接口
    - 支持链式法则进行复杂导数计算
    
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
    
    属性说明：
    - data: 底层NumPy多维数组，存储实际数据
    - fromvars: 元组，记录当前张量由哪些输入张量计算而来
    - parms: 元组，记录计算过程中使用的非张量参数
    - gradfuncs: 元组，记录反向传播时需要调用的梯度函数
    - grad: TN类型，存储计算得到的梯度
    - requires_grad: 布尔值，控制是否需要计算梯度
    - retains_grad: 布尔值，控制是否保留梯度值
    - is_leaf: 布尔值，表示是否为计算图的叶节点
    - rcv_grad_count: 整数，用于梯度计算时记录节点可接收梯度的计数
    - grad_value: TN类型，反向传播中的临时梯度存储
    
    注：该类提供与PyTorch兼容的接口。
    """
    def __init__(self):
        """
        初始化张量对象。
        
        创建一个新的张量实例，初始化所有必要的属性。张量是Riemann框架的核心数据结构，
        封装了NumPy数组并支持自动微分功能。
        
        属性：
            data (np.ndarray): 存储张量数据的NumPy数组
            fromvars (tuple): 记录本张量是通过哪些张量计算得来的
            parms (tuple): 对张量进行函数运算时用到的参数，如sum函数的dim和keepdim参数
            gradfuncs (tuple): 张量运算时登记的梯度函数对象，在backward中将调用这些钩子函数
            grad (TN): 与data的shape一致的张量，用于存放梯度，初始化为None以节省空间
            requires_grad (bool): 是否需要计算梯度，默认为False
            retains_grad (bool): 是否保留梯度，默认为False
            is_leaf (bool): 是否为叶子节点，默认为True
            rcv_grad_count (int): 在计算图中可接收梯度的计数，计算梯度时临时使用
            grad_value (TN): 用于计算反向传播的梯度，最终计算结果保存在grad里，该变量用于临时计算
        """
        #tensor函数构造对象时，data引用一个numpy或cupy数组
        self.data: np.ndarray = None    # type: ignore
        self.fromvars = ()              #张量运算时，运算符函数中记录本张量是通过哪些张量计算得来的
        self.parms = ()                 #对张量进行函数运算时用到的参数，比如sum函数的dim和keepdim参数
        self.gradfuncs = ()             #张量运算时，运算符函数中登记梯度函数对象，在backward中将调用这些钩子函数
        self.grad:TN = None             #与data的shape一致的numpy数组，用于存放梯度，
                                        #初始化为None是为了节省空间，计算梯度时才分配空间
        self.requires_grad = False      #tensor默认不计算梯度，在tensor函数构建对象时指定是否需要计算梯度
        self.retains_grad = False
        self.is_leaf = True

        self.rcv_grad_count = 0         #在具体计算图中，self可接收梯度的计数，计算梯度时临时使用
        self.grad_value:TN = None       #用于计算反向传播的梯度，最终计算结果保存在grad里，该变量用于临时计算
        
        return
    
    # 用于交互式环境显示对象的数值
    def __disp__(self, format_spec: str):
        """
        带格式字符串参数的显示函数
        
        参数:
            format_spec: 格式规范字符串，默认为'8.4f'（宽度为8，精度为4的浮点数格式）
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
        if arrlib.issubdtype(self.data.dtype, arrlib.complexfloating):
            if use_numpy_default:
                # 使用numpy默认格式化
                data_str = arrlib.array2string(
                    self.data,
                    separator=', ',
                    threshold=self.data.size  # 确保显示所有元素
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
                    self.data,
                    separator=', ',
                    formatter=formatter_dict,  # type: ignore
                    threshold=self.data.size  # 确保显示所有元素
                )
        elif arrlib.issubdtype(self.data.dtype, arrlib.floating):
            if use_numpy_default:
                # 使用numpy默认格式化
                data_str = arrlib.array2string(
                    self.data,
                    separator=', ',
                    threshold=self.data.size  # 确保显示所有元素
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
                    self.data,
                    separator=', ',
                    formatter=formatter_dict,  # type: ignore
                    threshold=self.data.size  # 确保显示所有元素
                )
        else:
            # 对于非浮点、非复数类型，使用默认格式化
            if arrlib.issubdtype(self.data.dtype, arrlib.integer):
                data_str = arrlib.array2string(
                    self.data,
                    separator=', '
                )
            else:
                # 其他类型使用默认格式化
                data_str = arrlib.array2string(self.data, separator=', ')

        
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
            return format(self.data, format_spec)
        else:
            return self.__disp__(format_spec)

    def __array__(self, dtype=None, copy=None):
        '''
        numpy函数处理TN张量的参数时，自动调用张量的该方法函数将张量转换为numpy数组。
        '''
        # 获取数据
        arr = self.data
        
        # 根据指定的dtype转换数据
        if dtype is not None and arr.dtype != dtype:
            arr = arr.astype(dtype)
        
        # 处理copy参数（虽然在当前实现中我们总是返回底层数据的引用）
        # 在numpy的array protocol中，copy=False意味着尽可能返回视图而非副本
        if copy == True:
            arr = arr.copy()

        return arr

    def __hash__(self):
        """
        使TN对象可哈希，允许将其放入set或用作字典键。
        使用对象的内存地址作为哈希值，确保实例唯一性。
        
        Returns:
            int: 基于对象id的哈希值
        """
        return id(self)
        
    # 返回张量第一维的大小
    def __len__(self):
        """
        返回张量第一维的大小。
        
        返回张量第一个维度的大小，等同于len(tensor.data)。
        对于标量张量（0维），此方法会引发TypeError。
        
        Returns:
            int: 张量第一维的大小
            
        Raises:
            TypeError: 如果张量是标量（0维）
        """
        return len(self.data)
    
    def __bool__(self):
        """
        将张量转换为布尔值。
        
        根据张量的值返回True或False，遵循Python的布尔转换规则。
        对于非空张量，如果所有元素都不为零，则返回True；否则返回False。
        对于空张量，返回False。
        
        Returns:
            bool: 张量的布尔值表示
        """
        return bool(self.data)
    
    def __float__(self):
        """
        将张量转换为Python浮点数。
        
        仅适用于单元素张量，将张量的值转换为Python的float类型。
        
        Returns:
            float: 张量值的浮点数表示
            
        Raises:
            ValueError: 如果张量包含多个元素
        """
        return float(self.data)

    def __int__(self):
        """
        将张量转换为Python整数。
        
        仅适用于单元素张量，将张量的值转换为Python的int类型。
        
        Returns:
            int: 张量值的整数表示
            
        Raises:
            ValueError: 如果张量包含多个元素
        """
        return int(self.data)
    
    def _get_array_lib(self):
        """
        根据data属性的数组类型，获得numpy或cupy库的模块名
        """
        return np if isinstance(self.data,np.ndarray) else cp

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
        return self.data.item()
    
    @property
    def dtype(self):
        """
        返回张量的数据类型。
        
        返回张量中元素的数据类型，与NumPy数组的dtype属性一致。
        常见的数据类型包括float32、float64、int32、int64、complex64等。
        
        Returns:
            numpy.dtype: 张量的数据类型对象
        """
        return self.data.dtype
    
    def is_floating_point(self):
        """
        判断张量是否为浮点类型。
        
        检查张量的数据类型是否为浮点类型，包括float16、float32、float64等。
        浮点类型张量支持梯度计算，通常用于神经网络中的权重和激活值。
        
        Returns:
            bool: 如果张量是浮点类型返回True，否则返回False
        """
        return self.data.dtype.kind == 'f'

    def is_complex(self):
        """
        判断张量是否为复数类型。
        
        检查张量的数据类型是否为复数类型，包括complex64、complex128等。
        复数类型张量包含实部和虚部，支持复数运算和梯度计算。
        
        Returns:
            bool: 如果张量是复数类型返回True，否则返回False
        """
        return self.data.dtype.kind == 'c'
    
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
            
        result = tensor(self.data.real, device=self.device, requires_grad=(is_grad_enabled() and self.requires_grad))
        result.is_leaf = not result.requires_grad
        
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
        
        result = tensor(self.data.imag, device=self.device, requires_grad=(is_grad_enabled() and self.requires_grad))
        result.is_leaf = not result.requires_grad
        
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
        conj_data = arrlib.conj(self.data)
        
        # 创建结果张量
        ret = tensor(conj_data, device=self.device,requires_grad=(is_grad_enabled() and self.requires_grad))
        ret.is_leaf = not ret.requires_grad
        
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
        return self.data.shape
    
    @property
    def ndim(self):
        """
        返回张量的维度数。
        
        返回张量的维度数量，也称为秩(rank)。标量的维度为0，向量的维度为1，
        矩阵的维度为2，以此类推。与NumPy数组的ndim属性一致。
        
        Returns:
            int: 张量的维度数
        """
        return self.data.ndim

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
            return self.data.shape
        else:
            return self.data.shape[dim]

    def numel(self):
        """返回张量中元素的总数。
        
        返回:
            int: 张量中元素的数量
        """
        return self.data.size

    def type(self, dtype=None):
        """返回或转换张量的数据类型
        
        行为：
        - 如果不传入参数，返回张量的数据类型
        - 如果传入数据类型参数，返回一个转换为指定数据类型的新张量
        
        参数:
            dtype: 数据类型，可以是Python类型、NumPy dtype、字符串或Riemann dtype
                如果为None，则返回当前张量的数据类型
                
        返回:
            如果dtype为None，返回当前张量的数据类型
            否则返回转换为指定数据类型的新张量
        """
        # 如果不传入参数，返回张量的数据类型
        if dtype is None:
            return self.dtype
        
        target_dtype = _get_dtype(dtype)
        
        # 如果当前类型已经匹配，返回自身
        if self.dtype == target_dtype:
            return self
        
        # 创建新的张量并转换数据类型
        ret = TN()
        if np.issubdtype(self.dtype,np.complexfloating) and not np.issubdtype(target_dtype,np.complexfloating):
            # 复数向非复数转换时，为避免warning，不用astype直接转，取real后再转换
            ret.data = self.data.real.astype(target_dtype)
        else:
            ret.data = self.data.astype(target_dtype)
        
        # 只有浮点和复数是可微分的，使用is_floating函数检查
        if is_float_or_complex(target_dtype):
            ret.requires_grad = (is_grad_enabled() and self.requires_grad)
            ret.is_leaf = not ret.requires_grad

            if ret.requires_grad:
                ret.fromvars = (self,)
                ret.parms = (self.dtype,)
                # 反向传播时，需要将梯度转换回原始数据类型
                def type_backward(result, i):
                    # 确保梯度值被转换为原始数据类型
                    original_dtype = result.parms[0]
                    return result.grad_value.type(original_dtype)
                ret.gradfuncs = (type_backward,)

        return ret

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
        if CUPY_AVAILABLE and isinstance(self.data, cp.ndarray):
            # 使用CuPy数组的device属性获取设备信息
            cupy_device = self.data.device
            device_idx = cupy_device.id
            return Device(f'cuda:{device_idx}')
        else:
            # 对于NumPy数组或其他类型，返回CPU设备
            return Device('cpu')
                
    def to(self, *args, **kwargs):
        """
        将张量转换为指定的数据类型和/或设备。
        
        创建并返回一个新的张量，其数据类型和/或设备为指定的值，与原张量内容相同。
        如果原张量的数据类型和设备已经与指定的值相同，则返回原张量本身。
        
        Args:
            dtype (optional): 目标数据类型，可以是Python类型、NumPy dtype、字符串或Riemann dtype
            device (optional): 目标设备，可以是字符串（如'cpu'、'cuda'）或Device对象
            
        Returns:
            TN: 指定数据类型和/或设备的新张量，或原张量（如果已匹配）
            
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
        """
        # 解析参数
        dtype = kwargs.get('dtype', None)
        device = kwargs.get('device', None)
        
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
        if dtype_not_change and device_not_change:
            return self
        
        # 创建新张量
        ret = TN()
        
        # 处理设备迁移、数据转换
        if device_not_change:
            # 设备相同，数据类型一定不同
            if np.issubdtype(self.dtype, np.complexfloating) and not np.issubdtype(target_dtype, np.complexfloating):
                # 复数向非复数转换时，为避免warning，不用astype直接转，取real后再转换
                ret.data = self.data.real.astype(target_dtype)
            else:
                ret.data = self.data.astype(target_dtype)            
        else:
            # 设备不同，需要转换设备
            if target_device.type=='cuda':
                # CPU -> CUDA
                # 利用cp.asarray()支持dtype参数的特性，一步完成转换
                if np.issubdtype(self.dtype, np.complexfloating) and not np.issubdtype(target_dtype, np.complexfloating):
                    # 复数向非复数转换时，先取real
                    ret.data = cp.asarray(self.data.real, dtype=target_dtype)
                else:
                    ret.data = cp.asarray(self.data, dtype=target_dtype)
            else:
                # CUDA -> CPU
                # 先转换设备，再转换数据类型
                if np.issubdtype(self.dtype, np.complexfloating) and not np.issubdtype(target_dtype, np.complexfloating):
                    # 复数向非复数转换时，先取real
                    ret.data = cp.asnumpy(self.data.real).astype(target_dtype)
                else:
                    ret.data = cp.asnumpy(self.data).astype(target_dtype)
        
        # 设置梯度跟踪信息
        if is_float_or_complex(target_dtype):
            ret.requires_grad = (is_grad_enabled() and self.requires_grad)
            ret.is_leaf = not ret.requires_grad
            
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
            ret.is_leaf = True
        
        return ret

    def cuda(self, device=None):
        """
        将张量移动到CUDA设备上。
        
        Args:
            device (optional): 目标CUDA设备ID或设备名称，默认为当前设备
        
        Returns:
            TN: 移动到CUDA设备上的新张量
        """
        if device is None:
            # 使用当前CUDA设备
            device_id = current_device()
            device_name = f'cuda:{device_id}'
        elif isinstance(device, int):
            # 设备ID
            device_name = f'cuda:{device}'
        else:
            # 设备名称
            device_name = device
        
        return self.to(device=device_name)
    
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
    
    def __getitem__(self, index):
        # 类型转换：TN索引转为NumPy数组
        index_val = _convert_TNindex_to_numpy(index)
        
        # 执行索引操作并捕获错误
        try:
            indexed_data = self.data[index_val]
        except (IndexError, ValueError) as e:
            # 错误信息增强
            err_msg = f"Index operation failed: {str(e)} (Original index type: {type(index)}, shape: {self.shape})"
            raise type(e)(err_msg) from None
        
        # 使用NumPy的视图检查机制来判断索引结果是否为视图
        # NumPy会根据索引类型自动决定返回视图还是副本
        is_view = False
        if hasattr(indexed_data, 'base') and indexed_data.base is not None:
            arrlib = self._get_array_lib()
            # 检查indexed_data是否与原数据共享内存
            is_view = arrlib.shares_memory(self.data, indexed_data)
        
        # 创建结果张量
        ret = tensor(indexed_data, device=self.device, requires_grad = (is_grad_enabled() and self.requires_grad))
        ret.is_leaf = not ret.requires_grad
        
        # 根据索引类型设置视图属性或创建副本
        if is_view:
            # 设置视图属性
            ret._is_view = True
            ret._base = self
            ret._view_index = index_val
        else:
            # 创建副本，不设置视图属性
            ret._is_view = False
            ret._base = self
            ret._view_index = index_val
        
        if ret.requires_grad:
            ret.fromvars = (self,)
            ret.parms = (index_val,)
            ret.gradfuncs = (_getitem_backward,)
        
        return ret
    
    def __setitem__(self, index, val):
        if self.is_leaf and self.requires_grad:
            raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation.')
        
        # 检查是否是视图对象的赋值，避免不必要的操作
        if hasattr(val, '_is_view'):
            # 转换索引格式以便比较
            index_val = _convert_TNindex_to_numpy(index)
                
            if val._is_view:
                # 如果赋值的是视图对象，且视图的基础是self，索引也相同，则不需要执行操作
                if val._base is self and val._view_index == index_val:
                    return self
            else:
                if val._base is self and val._view_index == index_val:
                    for i in range(len(val.gradfuncs)-1,-1,-1):
                        funcname = val.gradfuncs[i].__name__
                        if funcname == '_addat_inplace_backward':
                            if val.parms[i][1] == ():
                                return self.addat_(index_val,val.fromvars[i])

                        if funcname == '_subat_inplace_backward':
                            if val.parms[i][1] == ():
                                return self.subat_(index_val,val.fromvars[i])
                        
                        if funcname == '_mulat_inplace_backward':
                            if val.parms[i][1] == ():
                                return self.mulat_(index_val,val.fromvars[i])

                        if funcname == '_divat_inplace_backward':
                            if val.parms[i][1] == ():
                                return self.divat_(index_val,val.fromvars[i])

                        if funcname == '_powat_inplace_backward':
                            if val.parms[i][1] == ():
                                return self.powat_(index_val,val.fromvars[i])
                    pass
        
        # print('in __setitem__, call setat_')
        return self.setat_(index,val)
        
    def _merge_indices(self, base_index, current_index):
        """
        基于坐标转换的索引合并函数（重构版本）
        
        思路：将索引视为坐标，通过numpy向量化操作实现从视图坐标系到基张量坐标系的坐标转换
        重构目标：更清晰的结构、更少的重复代码、更好的可读性
        
        参数:
            base_index: 视图的基础索引（在基张量坐标系中的坐标）
            current_index: 当前操作的索引（在视图坐标系中的坐标）
            
        返回:
            combined_index: 合并后的索引（在基张量坐标系中的坐标）
        """
        arrlib = self._get_array_lib()

        # 辅助函数：检查是否为布尔类型索引
        def _is_boolean_index(index):
            return isinstance(index, arrlib.ndarray) and index.dtype == bool
        
        def _normalize_neg_index(index, dim_size):
            """
            将负索引转换为对应的正索引
            
            参数:
                index: 要转换的索引（可以是整数、切片或整数数组）
                dim_size: 当前维度的大小
                
            返回:
                normalized_index: 转换后的正索引
            """
            if isinstance(index, int):
                if index < 0:
                    index += dim_size
                return index
            elif isinstance(index, slice):
                start = index.start
                stop = index.stop
                step = index.step
                
                if start is not None and start < 0:
                    start += dim_size
                if stop is not None and stop < 0:
                    stop += dim_size
                
                return slice(start, stop, step)
            elif isinstance(index, (arrlib.ndarray, list)) and arrlib.issubdtype(arrlib.array(index).dtype, arrlib.integer):
                index_arr = arrlib.asarray(index)
                # 对负索引进行转换
                index_arr[index_arr < 0] += dim_size
                return index_arr
            else:
                return index

        # 辅助函数：标准化索引格式并处理省略号
        def _normalize_index(index, length):
            if not isinstance(index, tuple):
                index = (index,) if not _is_boolean_index(index) else index
            
            if _is_boolean_index(index):
                return index
                
            # 处理省略号 - 只有当 index 是元组且不包含数组时才检查
            if isinstance(index, tuple):
                # 检查元组中是否包含数组或非基本类型
                contains_array = any(isinstance(idx, (arrlib.ndarray, list)) and not isinstance(idx, (str, bytes)) for idx in index)
                if not contains_array and Ellipsis in index:
                    ellipsis_idx = index.index(Ellipsis)
                    num_regular = len(index) - 1
                    num_missing = length - num_regular
                    if num_missing > 0:
                        index = index[:ellipsis_idx] + (slice(None),) * num_missing + index[ellipsis_idx+1:]
                    else:
                        index = index[:ellipsis_idx] + index[ellipsis_idx+1:]
            
            # 补全索引长度 - 只有当 index 是元组时才补全
            if isinstance(index, tuple) and len(index) < length:
                index += (slice(None),) * (length - len(index))
            
            return index
        
        # 辅助函数：获取基础张量维度大小
        def _get_base_dim_size(view_dim):
            if hasattr(self, '_base') and self._base is not None:
                return self._base.shape[view_dim] if view_dim < len(self._base.shape) else 1
            return self.shape[view_dim] if view_dim < len(self.shape) else 1
        
        # 辅助函数：处理基础索引为数组的情况
        def _process_base_array_index(base_idx, view_dim):
            dim_size = _get_base_dim_size(view_dim)
            base_idx_arr = arrlib.asarray(base_idx)
            
            if arrlib.issubdtype(base_idx_arr.dtype, arrlib.bool_):
                # 布尔数组 -> 整数数组
                base_idx_arr = arrlib.where(base_idx_arr)[0]
            
            # 处理负索引
            if arrlib.any(base_idx_arr < 0):
                base_idx_arr = base_idx_arr.copy()
                base_idx_arr[base_idx_arr < 0] += dim_size
            
            return base_idx_arr, len(base_idx_arr)
        
        # 辅助函数：处理基础索引为切片的情况
        def _process_base_slice_index(base_idx, view_dim):
            dim_size = _get_base_dim_size(view_dim)
            norm_base = _normalize_neg_index(base_idx, dim_size)
            base_start = norm_base.start if norm_base.start is not None else 0
            base_stop = norm_base.stop if norm_base.stop is not None else dim_size
            base_step = norm_base.step  # 严格保留原始步长，不做默认值替换
            step_val = 1 if base_step is None else base_step
            view_dim_size = (base_stop - base_start + step_val - 1) // step_val
            
            return base_start, base_stop, base_step, step_val, view_dim_size
        
        # 第一步：标准化基础索引格式
        if isinstance(base_index, slice):
            base_index = (base_index,)
        elif isinstance(base_index, int):
            base_index = (base_index,)
        elif not isinstance(base_index, tuple):
            base_index = tuple(base_index)
        
        # 计算视图的实际维度（不包括被整数索引折叠的维度）
        view_dims = []
        for i, idx in enumerate(base_index):
            if not isinstance(idx, int):
                view_dims.append(i)
        
        # 第二步：处理特殊情况 - 省略号索引
        if isinstance(current_index, tuple) and any(idx is Ellipsis for idx in current_index):
            num_ellipsis = current_index.count(Ellipsis)
            if num_ellipsis > 1:
                raise ValueError("An index can only contain a single ellipsis ('...')")
            
            # 扩展省略号为相应数量的full slices
            ellipsis_idx = current_index.index(Ellipsis)
            num_full_slices = len(view_dims) - (len(current_index) - num_ellipsis)
            expanded_index = (
                current_index[:ellipsis_idx] + 
                (slice(None),) * builtins.max(0, num_full_slices) + 
                current_index[ellipsis_idx+1:]
            )
            current_index = expanded_index
        
        # 第三步：处理特殊情况 - view_dims为空（base_index全是整数）
        if not view_dims:
            # 处理base_index为空的情况（直接对原始张量进行索引）
            if len(base_index) == 0:
                return _normalize_index(current_index, 0) if isinstance(current_index, tuple) else current_index
            
            # 处理链式索引情况，如x[1][2]或x[1][:3]
            if isinstance(current_index, int):
                if len(base_index) == 1 and isinstance(base_index[0], int):
                    return (base_index[0], current_index)
            elif isinstance(current_index, tuple):
                if len(base_index) == 1 and isinstance(base_index[0], int):
                    return (base_index[0],) + current_index
            return base_index
        
        # 第四步：标准化当前索引
        current_index = _normalize_index(current_index, len(view_dims))
        
        # 第五步：快速路径 - 当前操作是对整个视图进行的
        if isinstance(current_index, tuple):
            contains_array = any(isinstance(idx, (arrlib.ndarray, list)) and not isinstance(idx, (str, bytes)) for idx in current_index)
            if not contains_array:
                all_full_slices = all(
                    isinstance(idx, slice) and 
                    idx.start is None and 
                    idx.stop is None and 
                    idx.step is None 
                    for idx in current_index
                )
                if all_full_slices and len(current_index) == len(view_dims):
                    return base_index
        
        # 第六步：处理布尔数组索引
        if _is_boolean_index(current_index):
            bool_indices = arrlib.where(current_index)
            if len(bool_indices) != len(view_dims):
                raise ValueError(f"Boolean index dimension mismatch: {len(bool_indices)} vs {len(view_dims)}")
            
            combined_coords = list(base_index)
            for i, (view_dim, bool_idx) in enumerate(zip(view_dims, bool_indices)):
                base_idx = base_index[view_dim]
                
                if isinstance(base_idx, slice):
                    # 切片 -> 布尔索引：线性变换
                    _, _, _, step, _ = _process_base_slice_index(base_idx, view_dim)
                    norm_base = _normalize_neg_index(base_idx, _get_base_dim_size(view_dim))
                    start = norm_base.start if norm_base.start is not None else 0
                    combined_coords[view_dim] = start + bool_idx * step
                elif isinstance(base_idx, (arrlib.ndarray, list)):
                    # 数组 -> 布尔索引：查表
                    base_arr, _ = _process_base_array_index(base_idx, view_dim)
                    combined_coords[view_dim] = base_arr[bool_idx]
                else:
                    raise NotImplementedError(f"Unsupported base index type: {type(base_idx)}")
            
            return tuple(combined_coords)
        
        # 第七步：常规索引合并逻辑
        combined_coords = list(base_index)
        
        # 确保当前索引是元组格式
        if not isinstance(current_index, tuple):
            current_index = (current_index,)
        
        # 确保当前索引长度与view_dims长度匹配
        while len(current_index) < len(view_dims):
            current_index += (slice(None),)
        
        # 遍历视图的每个维度和当前索引
        for i, (view_dim, curr_idx) in enumerate(zip(view_dims, current_index)):
            base_idx = base_index[view_dim]
            
            # 处理None/newaxis索引
            if curr_idx is None:
                combined_coords[view_dim] = None
                continue
            
            # 根据基础索引类型选择处理方式
            if isinstance(base_idx, slice):
                base_start, base_stop, base_step, step_val, view_dim_size = _process_base_slice_index(base_idx, view_dim)
            elif isinstance(base_idx, (arrlib.ndarray, list)):
                base_idx_arr, view_dim_size = _process_base_array_index(base_idx, view_dim)
            else:
                raise NotImplementedError(f"Unsupported base index type: {type(base_idx)}")
            
            # 标准化当前索引
            norm_curr_idx = _normalize_neg_index(curr_idx, view_dim_size)
            
            # 应用坐标转换
            if isinstance(norm_curr_idx, int):
                if isinstance(base_idx, slice):
                    # 切片 -> 整数：线性变换
                    actual_idx = base_start + norm_curr_idx * step_val
                    if actual_idx >= base_stop:
                        raise IndexError(f"Index {norm_curr_idx} out of bounds for view dimension size {view_dim_size}")
                    combined_coords[view_dim] = actual_idx
                else:
                    # 数组 -> 整数：直接查表
                    combined_coords[view_dim] = base_idx_arr[norm_curr_idx]
            
            elif isinstance(norm_curr_idx, slice):
                if isinstance(base_idx, slice):
                    # 切片 -> 切片：组合线性变换
                    curr_start = norm_curr_idx.start if norm_curr_idx.start is not None else 0
                    curr_stop = norm_curr_idx.stop
                    curr_step = norm_curr_idx.step
                    
                    # 计算实际数值
                    base_step_val = 1 if base_step is None else base_step
                    curr_step_val = 1 if curr_step is None else curr_step
                    
                    actual_start = base_start + curr_start * base_step_val
                    actual_stop = base_stop if curr_stop is None else (base_start + curr_stop * base_step_val)
                    actual_stop = builtins.min(actual_stop, base_stop)
                    
                    # 确定最终步长
                    if base_step is None and curr_step is None:
                        actual_step = None
                    elif base_step is None:
                        actual_step = curr_step
                    elif curr_step is None:
                        actual_step = base_step
                    else:
                        actual_step = base_step * curr_step
                    
                    combined_coords[view_dim] = slice(actual_start, actual_stop, actual_step)
                else:
                    # 数组 -> 切片：直接索引映射数组
                    combined_coords[view_dim] = base_idx_arr[norm_curr_idx]
            
            elif isinstance(norm_curr_idx, (arrlib.ndarray, list)):
                curr_idx_arr = arrlib.asarray(norm_curr_idx)
                
                if arrlib.issubdtype(curr_idx_arr.dtype, arrlib.bool_):
                    # 布尔数组 -> 整数数组
                    bool_indices = arrlib.where(curr_idx_arr)[0]
                    if len(bool_indices) == 0:
                        combined_coords[view_dim] = arrlib.array([], dtype=int)
                        continue
                    curr_idx_arr = bool_indices
                
                # 确保索引在有效范围内
                if arrlib.any(curr_idx_arr >= view_dim_size):
                    raise IndexError(f"Array index out of bounds for view dimension size {view_dim_size}")
                
                if isinstance(base_idx, slice):
                    # 切片 -> 数组：应用线性变换
                    combined_coords[view_dim] = base_start + curr_idx_arr * step_val
                else:
                    # 数组 -> 数组：向量化查表
                    combined_coords[view_dim] = base_idx_arr[curr_idx_arr]
            
            else:
                raise NotImplementedError(f"Unsupported index type: {type(norm_curr_idx)}")
        
        return tuple(combined_coords)
    
    def _adjust_right_val(self, target_data, val):
        if not isinstance(val,TN):
            right_val = tensor(val, dtype = self.dtype)
        else:
            right_val = val
        
            # 确保val和self的dtype相同
            if right_val.dtype != self.dtype:
                right_val = right_val.type(self.dtype)           
        
        # 将右值形状、维度转化为和索引值相匹配的形式
        target_size = target_data.size
        right_size = right_val.numel()
        if right_size > target_size:
            if target_size == 0 and right_size == 1:
                pass   # 允许向空张量赋值0D标量
            else:
                raise ValueError(f"Right value size {right_size} exceeds target size {target_size}")
        elif right_size == target_size:
            right_val = right_val.reshape(target_data.shape)
        elif right_val.ndim <= target_data.ndim:
            right_val = right_val.broadcast_to(target_data.shape)
        else:
            # 获取所有大小为1的维度
            dim = tuple(i for i, size in enumerate(right_val.shape) if size == 1)
            # 移除所有大小为1的维度
            right_val = right_val.squeeze(dim)
            right_val = right_val.broadcast_to(target_data.shape)

        return right_val

    def _inplace_oper_at_(self, index, val, oper_func, backward_func):
        # 将索引规范化为numpy索引
        index_val = _convert_TNindex_to_numpy(index) 
        
        # 检查是否为视图对象
        if hasattr(self, '_is_view') and self._is_view:
            # 将当前操作索引与视图索引合并
            base_index = self._view_index
                        
            # 调用内部函数合并索引
            combined_index = self._merge_indices(base_index, index_val)
            
            # self是视图时，直接在原张量self._base上执行操作
            self._base._inplace_oper_at_(combined_index, val, oper_func, backward_func)
            
            # 返回视图对象本身
            return self
        
        # 将右值形状、维度转化为和索引值相匹配的形式
        target_data = self.data[index_val]
        right_val= self._adjust_right_val(target_data,val)

        # 因为以下原因：
        # 1、target与self是共享内存的，
        # 2、向右值传播梯度时可能需要左值
        # 3、对self原地操作后，self index位置的数据已被修改，相当于左值消失
        # 所以，原地操作数据前，须备份target的独立且无依赖副本，用于梯度跟踪时使用
        target_copy = tensor(target_data.copy(),dtype=self.dtype)
        oper_func(self.data, index_val, right_val.data)

        self.requires_grad = (is_grad_enabled() and (self.requires_grad or right_val.requires_grad))
        self.is_leaf = not self.requires_grad

        if self.requires_grad:
            # 原地赋值时，须在self的现有计算图上增加依赖关系，注意:
            # 1、只能记录右值的依赖关系，不能记录self，避免依赖循环
            # 2、右值的梯度回调函数里同时传递右值梯度和更新self的梯度，所以右值requires_grad=False时，
            #    需要复制一个右值的副本用于跟踪梯度，将副本的requires_grad置True，以便backward里副本的梯度跟踪函数被调用
            # 3、原地赋值依赖关系需要优先处理，须插入到fromvars元组前面
            # 4、多次原地赋值时，遵循先赋值的右值后处理梯度跟踪原则，要从0位置插入右值、索引、梯度回调函数
            # 5、如果右值已经依赖于self，需要特殊处理以避免循环依赖
            if right_val.requires_grad == False:
                # 需要一个用于传播梯度的右值变量，但又不要改动右值的属性，detach一个新变量是一个经济、可行的办法
                right_val = right_val.detach().requires_grad_(True)
            
            self.fromvars = (right_val,) + self.fromvars
            # 注意(target_copy,index_val)必须以元组的形似整体插入到parms元组前面
            self.parms = ((target_copy,index_val),) + self.parms
            self.gradfuncs = (backward_func, ) + self.gradfuncs

        return self

    def _non_inplace_oper_at(self, index, val, oper_func, left_backward_func, right_backward_func):
        # 将索引规范化为numpy索引
        index_val = _convert_TNindex_to_numpy(index)
        
        # 将右值形状、维度转化为和索引值相匹配的形式
        target = self.data[index_val]        
        right_val= self._adjust_right_val(target, val)

        # 新建一个self张量的独立副本，用于存储结果
        ret = self.copy()
        oper_func(ret.data, index_val, right_val.data)

        ret.requires_grad = (is_grad_enabled() and (self.requires_grad or right_val.requires_grad))
        ret.is_leaf = not ret.requires_grad        

        if ret.requires_grad:
            ret.fromvars = (self,right_val)
            ret.parms = (index_val,)
            ret.gradfuncs = (left_backward_func,right_backward_func)

        return ret

    def setat_(self, index, val):
        def _set_numpy_item(numpy_arr, index, right_numpy_arr):
            numpy_arr[index] = right_numpy_arr
            return numpy_arr
        
        return self._inplace_oper_at_(index, val, _set_numpy_item, _setat_inplace_backward)

    def setat(self, index, val):
        def _set_numpy_item(numpy_arr, index, right_numpy_arr):
            numpy_arr[index] = right_numpy_arr
            return numpy_arr
        
        return self._non_inplace_oper_at(index, val, _set_numpy_item, _setat_backward_left, _setat_backward_right)

    def addat_(self,index,val):
        arrlib = self._get_array_lib()
        return self._inplace_oper_at_(index, val, arrlib.add.at, _addat_inplace_backward)

    def addat(self,index,val):
        arrlib = self._get_array_lib()
        return self._non_inplace_oper_at(index, val, arrlib.add.at, _addat_backward_left, _addat_backward_right)

    def subat_(self,index,val):
        arrlib = self._get_array_lib()
        return self._inplace_oper_at_(index, val, arrlib.subtract.at, _subat_inplace_backward)

    def subat(self,index,val):
        arrlib = self._get_array_lib()
        return self._non_inplace_oper_at(index, val, arrlib.subtract.at, _subat_backward_left, _subat_backward_right)

    def mulat_(self,index,val):
        arrlib = self._get_array_lib()
        return self._inplace_oper_at_(index, val, arrlib.multiply.at, _mulat_inplace_backward)

    def mulat(self,index,val):
        arrlib = self._get_array_lib()
        return self._non_inplace_oper_at(index, val, arrlib.multiply.at, _mulat_backward_left, _mulat_backward_right)

    def divat_(self,index,val):
        arrlib = self._get_array_lib()
        return self._inplace_oper_at_(index, val, arrlib.divide.at, _divat_inplace_backward)

    def divat(self,index,val):
        arrlib = self._get_array_lib()
        return self._non_inplace_oper_at(index, val, arrlib.divide.at, _divat_backward_left, _divat_backward_right)

    def powat_(self,index,val):
        arrlib = self._get_array_lib()
        return self._inplace_oper_at_(index, val, arrlib.power.at, _powat_inplace_backward)

    def powat(self,index,val):
        arrlib = self._get_array_lib()
        return self._non_inplace_oper_at(index, val, arrlib.power.at, _powat_backward_left, _powat_backward_right)

    def _compute_to_direct_indices(self:TN, dim:int, index_data:np.ndarray):
        """
        计算用于setat、setat_、addat的直接索引，
        用在gather、scatter、scatter、scatter_scatter中将间接索引转换为直接索引
        
        参数:
            self: gather或scatter操作的张量，用于获取形状信息
            dim: 要操作的维度索引
            index_data: 包含索引映射信息的NumPy数组
        
        返回:
            一个元组，包含每个维度的索引数组，用于直接索引目标张量
        """

        # 获取self和index_data的形状
        target_shape = self.shape        
        index_shape = index_data.shape

        # 确保index和self的维度数相同
        # index张量的维度必须与读取数据的源张量和写入数据的目的张量的维度相同
        # 用于gather函数时，self是源张量，用于scatter函数时，self是目的张量
        if index_data.ndim != self.ndim:
            raise ValueError(f"Dimension mismatch between index and self: index has {index.ndim} dimensions, self has {self.ndim} dimensions")  # type: ignore
        
        # 确保dim在有效范围内
        if dim >= len(target_shape) or dim < -len(target_shape):
            raise ValueError(f"Dimension {dim} out of range for tensor of shape {target_shape}")
     
        # 添加对负维度索引的支持
        if dim < 0:
            dim = dim + len(target_shape)
        
        arrlib = self._get_array_lib()
        # 检查索引值是否在有效范围内
        if arrlib.any(index_data < 0) or arrlib.any(index_data >= target_shape[dim]):
            raise IndexError(f"Index value out of range [0, {target_shape[dim]}-1]")

        # 使用np.indices一次性创建所有网格索引，这是高效的关键
        grid_indices = arrlib.indices(index_shape)
        
        # 为每个维度创建索引
        direct_indices = []
        
        # 处理与index_data维度对应的部分
        for i in range(len(index_shape)):
            if i == dim and dim < len(index_shape):
                # 如果dim在index的维度范围内，使用index_data的值
                direct_indices.append(index_data)
            else:
                # 否则使用网格索引
                direct_indices.append(grid_indices[i])
        
        # 处理目标张量维度大于index_data的情况 - 简化版本
        for i in range(len(index_shape), len(target_shape)):
            # 对于额外维度，直接使用全0索引
            # 这与gather函数原始实现保持一致
            direct_indices.append(arrlib.zeros(index_shape, dtype=arrlib.int64))
        
        return tuple(direct_indices)
    # end of _compute_direct_indices

    def gather(self, dim: int, index:TN):
        """
        根据指定的 ​维度和​索引收集元素，
        常用于需要根据条件动态选择数据的场景（如交叉熵损失函数中选择正确类别的预测概率）
        :param dim: 收集维度（0 ≤ dim < self.ndim）
        :param index: 索引张量（dtype=int64）
        :return: 新TN实例，形状与index相同
        """
        # 确保索引是int64类型
        arrlib = self._get_array_lib()
        index.data = index.data.astype(arrlib.int64)
                
        gather_indices_tuple = self._compute_to_direct_indices(dim, index.data)
        result = self[gather_indices_tuple]
        
        # 无需设置自定义的梯度函数，__getitem__已经处理了反向传播
        
        return result

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
        
        # 标记value是否来自非命名参数
        is_positional_value = False
        
        arrlib = self._get_array_lib()
        dev = self.device

        # 处理非命名参数作为value的情况
        # 如果src是标量且value未提供，则将src视为value参数
        if src is not None and value is None:
            # 检查非命名参数是否为有效的标量类型
            positional_scalar_types = (int, float, complex)
            # 检查是否为numpy标量
            is_numpy_scalar = hasattr(src, 'dtype') and arrlib.isscalar(src)
            
            if isinstance(src, positional_scalar_types) or is_numpy_scalar:
                value = src
                src = None
                is_positional_value = True
        
        # 检查参数互斥性
        if src is not None and value is not None:
            raise ValueError("src and value parameters are mutually exclusive, please provide only one")
        
        if src is None and value is None:
            raise ValueError("Either src or value parameter must be provided")
        
        # 检查value参数类型
        if value is not None:
            if is_positional_value:
                # 非命名参数时的类型检查：仅限int、float、complex、numpy标量
                positional_scalar_types = (int, float, complex)
                is_numpy_scalar = hasattr(value, 'dtype') and arrlib.isscalar(value)
                
                if not (isinstance(value, positional_scalar_types) or is_numpy_scalar):
                    raise TypeError(f"Positional value must be a scalar (int, float, complex, numpy scalar), got {type(value).__name__}")
            else:
                # 命名参数时的类型检查：int、float、complex、numpy标量、TN标量
                named_scalar_types = (int, float, complex)
                is_numpy_scalar = hasattr(value, 'dtype') and arrlib.isscalar(value)
                is_TN_scalar = isinstance(value, TN) and value.ndim == 0
                
                if not (isinstance(value, named_scalar_types) or is_numpy_scalar or is_TN_scalar):
                    raise TypeError(f"Named value must be a scalar (int, float, complex, numpy scalar, TN scalar), got {type(value).__name__}")
        
        # 将index转换为TN张量（如果它不是）
        if not isinstance(index, TN):
            index = tensor(index, dtype=int64, device=dev)
        
        # 确保dim是有效的维度
        if dim < 0:
            dim += self.ndim
        if dim < 0 or dim >= self.ndim:
            raise IndexError(f"Dimension out of range (expected to be in range of [{-self.ndim}, {self.ndim-1}], got {dim})")
        
        # 确保index和self的维度数相同
        if index.ndim != self.ndim:
            raise ValueError(f"Dimension mismatch between index and self: index has {index.ndim} dimensions, self has {self.ndim} dimensions")
        
        # 处理src参数
        if src is not None:
            # 将src转换为TN张量（如果它不是）
            if not isinstance(src, TN):
                src = tensor(src, dtype=self.dtype, device=dev)
            
            # 确保src和self的dtype相同
            if src.dtype != self.dtype:
                raise RuntimeError(f"scatter(): Expected self.dtype to be equal to src.dtype")

            # 确保src和index的形状相同
            if src.shape != index.shape:
                raise ValueError(f"Shape mismatch between index and src: index has shape {index.shape}, src has shape {src.shape}")
            
            fill_value = src
        else:  # 使用value参数
            # 创建与index形状相同的张量，填充value值
            fill_value = full_like(index, value, dtype=self.dtype,device=dev)            

        # 计算直接索引
        full_index = self._compute_to_direct_indices(dim, index.data)
        
        # 使用setat进行赋值
        return self.setat(full_index, fill_value)
    # end of scatter

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
        
        if self.is_leaf and self.requires_grad:
            raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation.')
        
        # 标记value是否来自非命名参数
        is_positional_value = False
        
        arrlib = self._get_array_lib()
        dev = self.device

        # 处理非命名参数作为value的情况
        # 如果src是标量且value未提供，则将src视为value参数
        if src is not None and value is None:
            # 检查非命名参数是否为有效的标量类型
            positional_scalar_types = (int, float, complex)
            # 检查是否为numpy标量
            is_numpy_scalar = hasattr(src, 'dtype') and arrlib.isscalar(src)
            
            if isinstance(src, positional_scalar_types) or is_numpy_scalar:
                value = src
                src = None
                is_positional_value = True
        
        # 检查参数互斥性
        if src is not None and value is not None:
            raise ValueError("src and value parameters are mutually exclusive, please provide only one")
        
        if src is None and value is None:
            raise ValueError("Either src or value parameter must be provided")
        
        # 检查value参数类型
        if value is not None:
            if is_positional_value:
                # 非命名参数时的类型检查：仅限int、float、complex、numpy标量
                positional_scalar_types = (int, float, complex)
                is_numpy_scalar = hasattr(value, 'dtype') and arrlib.isscalar(value)
                
                if not (isinstance(value, positional_scalar_types) or is_numpy_scalar):
                    raise TypeError(f"Positional value must be a scalar (int, float, complex, numpy scalar), got {type(value).__name__}")
            else:
                # 命名参数时的类型检查：int、float、complex、numpy标量、TN标量
                named_scalar_types = (int, float, complex)
                is_numpy_scalar = hasattr(value, 'dtype') and arrlib.isscalar(value)
                is_TN_scalar = isinstance(value, TN) and value.ndim == 0
                
                if not (isinstance(value, named_scalar_types) or is_numpy_scalar or is_TN_scalar):
                    raise TypeError(f"Named value must be a scalar (int, float, complex, numpy scalar, TN scalar), got {type(value).__name__}")
        
        # 将index转换为TN张量（如果它不是）
        if not isinstance(index, TN):
            index = tensor(index, dtype = int64, device=dev)
        
        # 确保dim是有效的维度
        if dim < 0:
            dim += self.ndim
        if dim < 0 or dim >= self.ndim:
            raise IndexError(f"Dimension out of range (expected to be in range of [{-self.ndim}, {self.ndim-1}], got {dim})")
        
        # 确保index和self的维度数相同
        if index.ndim != self.ndim:
            raise ValueError(f"Dimension mismatch between index and self: index has {index.ndim} dimensions, self has {self.ndim} dimensions")
        
        # 处理src参数
        if src is not None:
            # 将src转换为TN张量（如果它不是）
            if not isinstance(src, TN):
                src = tensor(src, dtype=self.dtype, device=dev)
            
            # 确保src和self的dtype相同
            if src.dtype != self.dtype:
                raise RuntimeError(f"scatter_(): Expected self.dtype to be equal to src.dtype")
            
            # 确保src和index的形状相同
            if src.shape != index.shape:
                raise ValueError(f"Shape mismatch between index and src: index has shape {index.shape}, src has shape {src.shape}")
            
            fill_value = src
        else:  # 使用value参数
            # 创建与index形状相同的张量，填充value值
            fill_value = full_like(index, value, dtype=self.dtype,device=dev)            

        # 计算直接索引        
        full_index = self._compute_to_direct_indices(dim, index.data)
        
        # 使用setat进行赋值
        return self.setat_(full_index, fill_value)
    # end of scatter_

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
        dev = self.device
        
        # 将index转换为TN张量（如果它不是）
        if not isinstance(index, TN):
            index = tensor(index, dtype = int64, device=dev)
        
        # 处理src为标量的情况        
        # src参数的类型检查：int、float、complex、numpy标量、TN标量
        if is_scalar(src):
            src = full_like(index, src, dtype=self.dtype, device=dev)

        if not isinstance(src,TN):
            src = tensor(src, dtype=self.dtype, device=dev)

        # 确保src和self的dtype相同
        if src.dtype != self.dtype:
            raise RuntimeError(f"scatter_add_(): Expected self.dtype to be equal to src.dtype")

        # 对于张量，确保形状与index相同
        if src.shape != index.shape:
            raise ValueError(f"Shape mismatch between index and src: index has shape {index.shape}, src has shape {src.shape}")
        
        # 确保dim是有效的维度
        if dim < 0:
            dim += self.ndim
        if dim < 0 or dim >= self.ndim:
            raise IndexError(f"Dimension out of range (expected to be in range of [{-self.ndim}, {self.ndim-1}], got {dim})")
        
        # 确保index和self的维度数相同
        if index.ndim != self.ndim:
            raise ValueError(f"Dimension mismatch between index and self: index has {index.ndim} dimensions, self has {self.ndim} dimensions")
                
        # 计算直接索引
        full_index = self._compute_to_direct_indices(dim, index.data)
        
        # 直接调用原地操作的addat_函数
        return self.addat(full_index, src)
    # end of scatter_add

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

        if self.is_leaf and self.requires_grad:
            raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation.')
        
        dev = self.device
        
        # 将index转换为TN张量（如果它不是）
        if not isinstance(index, TN):
            index = tensor(index, dtype = int64, device=dev)
        
        # 处理src为标量的情况        
        # src参数的类型检查：int、float、complex、numpy标量、TN标量
        if is_scalar(src):
            src = full_like(index, src, dtype=self.dtype, device=dev)

        if not isinstance(src,TN):
            src = tensor(src, dtype=self.dtype, device=dev)

        # 确保src和self的dtype相同
        if src.dtype != self.dtype:
            raise RuntimeError(f"scatter_add_(): Expected self.dtype to be equal to src.dtype")

        # 对于张量，确保形状与index相同
        if src.shape != index.shape:
            raise ValueError(f"Shape mismatch between index and src: index has shape {index.shape}, src has shape {src.shape}")
        
        # 确保dim是有效的维度
        if dim < 0:
            dim += self.ndim
        if dim < 0 or dim >= self.ndim:
            raise IndexError(f"Dimension out of range (expected to be in range of [{-self.ndim}, {self.ndim-1}], got {dim})")
        
        # 确保index和self的维度数相同
        if index.ndim != self.ndim:
            raise ValueError(f"Dimension mismatch between index and self: index has {index.ndim} dimensions, self has {self.ndim} dimensions")
                
        # 计算直接索引
        full_index = self._compute_to_direct_indices(dim, index.data)
        
        # 直接调用原地操作的addat_函数
        return self.addat_(full_index, src)
    # end of scatter_add_

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
        return self.data
        
    def tolist(self):
        """
        将张量转换为Python列表。
        
        返回张量数据的Python列表表示。对于标量张量，返回Python标量值；
        对于多维张量，返回嵌套列表。此操作会断开计算图，因此返回的列表不会参与梯度计算。
        
        Returns:
            Union[list, int, float, complex]: 张量数据的Python列表或标量表示
            
        Raises:
            RuntimeError: 如果张量需要计算梯度
        """
        if self.requires_grad:
            raise RuntimeError("Can't call tolist() on Tensor that requires grad. Use tensor.detach().tolist() instead.")
        
        # 直接使用NumPy的tolist()方法，它会自动处理标量和多维数组
        return self.data.tolist()
    
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
        if self.data.ndim >= 2:    #只有3D以上多维数组才支持批量矩阵转置
            trandata = self.data.swapaxes(dim1, dim2)
            
            #转置产生一个新对象，但与原对象共用内存数据
            newobj = tensor(trandata,device=self.device,
                            requires_grad = (is_grad_enabled() and self.requires_grad))
            newobj.is_leaf = not newobj.requires_grad

            if newobj.requires_grad:
                newobj.fromvars = (self,)
                newobj.parms = ((dim1,dim2),)
                newobj.gradfuncs = (_transpose_backward,)
            
            return newobj
        else:
            raise RuntimeError("transpose() can only be used on tensors of 2D or more dimension")

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
        arrlib = self._get_array_lib()
        new_data = arrlib.transpose(self.data, dims)  # type: ignore
        
        # 创建新张量
        requires_grad = is_grad_enabled() and self.requires_grad
        newobj = tensor(new_data, device=self.device, requires_grad=requires_grad)
        
        # 设置叶子节点状态
        newobj.is_leaf = not requires_grad
        
        # 如果需要梯度，设置梯度跟踪信息
        if requires_grad:
            newobj.fromvars = (self,)
            newobj.parms = (dims,)
            newobj.gradfuncs = (_permute_backward,)
        
        return newobj

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
        if self.is_leaf and self.requires_grad:
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
        if self.is_leaf and self.requires_grad:
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
        if self.is_leaf and self.requires_grad:
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
        if self.is_leaf and self.requires_grad:
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
        if self.is_leaf and self.requires_grad:
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

    # '+'运算，左值为self，右值为TN，numpy数组，list，tuple，整数或浮点数
    def __add__(self,right_obj):
        # 获取self的设备
        dev = self.device

        #如右值是非TN对象，转化为TN对象，以便让后续处理一至
        right_tensor = right_obj if isinstance(right_obj,TN) else tensor(right_obj,device=dev)
        if dev != right_tensor.device:
            raise RuntimeError(f'Expected all tensors to be on the same device, but found at least two devices, {dev} and {right_tensor.device}!')
        
        #requires_grad属性在运算时传递到结果tensor
        requires_grad = (is_grad_enabled() and (self.requires_grad or right_tensor.requires_grad))
        ret = tensor(self.data + right_tensor.data, device=dev, requires_grad=requires_grad)
        ret.is_leaf = not requires_grad
        
        if requires_grad:
            ret.fromvars=(self,right_tensor) 
            ret.gradfuncs=(_add_grad_left,_add_grad_right)
        
        return ret
    
    # '+'运算，右值为self，左值为TN，numpy数组，list，tuple，整数或浮点数
    def __radd__(self,left_obj):
        if not isinstance(left_obj,TN):
            left_tensor = tensor(left_obj,device=self.device) #如左值是非TN对象，转化为TN对象
        return left_tensor.__add__(self) #归一到左值'+'函数

    # '-'运算，左值为self，右值为TN，numpy数组，list，tuple，整数或浮点数
    def __sub__(self,right_obj):
        # 获取self的设备
        dev = self.device

        #如右值是非TN对象，转化为TN对象，以便让后续处理一至
        right_tensor = right_obj if isinstance(right_obj,TN) else tensor(right_obj,device=dev)
        if dev != right_tensor.device:
            raise RuntimeError(f'Expected all tensors to be on the same device, but found at least two devices, {dev} and {right_tensor.device}!')
        
        #requires_grad属性在运算时传递到结果tensor
        requires_grad = (is_grad_enabled() and (self.requires_grad or right_tensor.requires_grad))
        ret = tensor(self.data - right_tensor.data, device=dev, requires_grad=requires_grad)
        ret.is_leaf = not requires_grad

        if requires_grad:
            ret.fromvars=(self,right_tensor) 
            ret.gradfuncs=(_sub_grad_left,_sub_grad_right)
            
        return ret
    
    # '-'运算，右值为self，左值为TN，numpy数组，list，tuple，整数或浮点数
    def __rsub__(self,left_obj):
        if not isinstance(left_obj,TN):
            left_tensor=tensor(left_obj,device=self.device) #如左值是非TN对象，转化为TN对象
        return left_tensor.__sub__(self) #归一到左值'-'函数
    
    # '*'运算，左值为self，右值为TN，numpy数组，list，tuple，整数或浮点数
    def __mul__(self,right_obj):
        # 获取self的设备
        dev = self.device

        #如右值是非TN对象，转化为TN对象，以便让后续处理一至
        right_tensor = right_obj if isinstance(right_obj,TN) else tensor(right_obj,device=dev)
        if dev != right_tensor.device:
            raise RuntimeError(f'Expected all tensors to be on the same device, but found at least two devices, {dev} and {right_tensor.device}!')
        
        #requires_grad属性在运算时传递到结果tensor
        requires_grad = (is_grad_enabled() and (self.requires_grad or right_tensor.requires_grad))
        ret = tensor(self.data * right_tensor.data, device=dev, requires_grad=requires_grad)
        ret.is_leaf = not requires_grad

        if requires_grad:
            ret.fromvars=(self,right_tensor) 
            ret.gradfuncs=(_mul_grad_left,_mul_grad_right)      

        return ret
    
    # '*'运算，右值为self，左值为TN，numpy数组，list，tuple，整数或浮点数
    def __rmul__(self,left_obj):
        if not isinstance(left_obj,TN):
            left_tensor=tensor(left_obj,device=self.device) #如左值是非TN对象，转化为TN对象
        return left_tensor.__mul__(self) #归一到左值'*'函数
    
    # '@'运算，左值为self，右值为TN，numpy数组，list，tuple，整数或浮点数
    def __matmul__(self,right_obj):
        # 获取self的设备
        dev = self.device

        #如右值是非TN对象，转化为TN对象，以便让后续处理一至
        right_tensor = right_obj if isinstance(right_obj,TN) else tensor(right_obj,device=dev)
        if dev != right_tensor.device:
            raise RuntimeError(f'Expected all tensors to be on the same device, but found at least two devices, {dev} and {right_tensor.device}!')
        
        if self.data.ndim == 0 or right_obj.data.ndim == 0:
            raise RuntimeError('both arguments to matmul need to be at least 1D')

        # requires_grad属性在运算时传递到结果tensor
        requires_grad = (is_grad_enabled() and (self.requires_grad or right_tensor.requires_grad))
        ret = tensor(self.data @ right_tensor.data, device=dev, requires_grad=requires_grad)
        ret.is_leaf=not requires_grad

        if requires_grad:
            ret.fromvars=(self,right_tensor) 
            ret.gradfuncs=(_matmul_grad_left,_matmul_grad_right)
        
        return ret

    # '@'运算，右值为self，左值为TN，numpy数组，list，tuple，整数或浮点数
    def __rmatmul__(self,left_obj):
        if not isinstance(left_obj,TN):
            left_tensor=tensor(left_obj,device=self.device) #如左值是非TN对象，转化为TN对象
        return left_tensor.__matmul__(self) #归一到左值'@'函数
    

    # '/'运算，左值为self，右值为TN，numpy数组，list，tuple，整数或浮点数
    def __truediv__(self,right_obj):
        # 获取self的设备
        dev = self.device

        #如右值是非TN对象，转化为TN对象，以便让后续处理一至
        right_tensor = right_obj if isinstance(right_obj,TN) else tensor(right_obj,device=dev)
        if dev != right_tensor.device:
            raise RuntimeError(f'Expected all tensors to be on the same device, but found at least two devices, {dev} and {right_tensor.device}!')
        
        #requires_grad属性在运算时传递到结果tensor
        requires_grad = (is_grad_enabled() and (self.requires_grad or right_tensor.requires_grad))
        ret=tensor(self.data / right_tensor.data, device=dev, requires_grad=requires_grad)
        ret.is_leaf=not requires_grad

        if requires_grad:
            ret.fromvars=(self,right_tensor)
            ret.gradfuncs=(_div_grad_left,_div_grad_right)
        
        return ret
    
    # '/'运算，右值为self，左值为TN，numpy数组，list，tuple，整数或浮点数
    def __rtruediv__(self,left_obj):
        if not isinstance(left_obj,TN):
            left_tensor=tensor(left_obj,device=self.device) #如左值是非TN对象，转化为TN对象
        return left_tensor.__truediv__(self) #归一到左值'*'函数

    # pow运算，左值为self，右值为TN，numpy数组，list，tuple，整数或浮点数
    def __pow__(self,right_obj):
        # 获取self的设备
        dev = self.device

        #如右值是非TN对象，转化为TN对象，以便让后续处理一至
        right_tensor = right_obj if isinstance(right_obj,TN) else tensor(right_obj,device=dev)
        if dev != right_tensor.device:
            raise RuntimeError(f'Expected all tensors to be on the same device, but found at least two devices, {dev} and {right_tensor.device}!')
        
        #requires_grad属性在运算时传递到结果tensor
        requires_grad = (is_grad_enabled() and (self.requires_grad or right_tensor.requires_grad))
        ret=tensor(self.data ** right_tensor.data, device=dev, requires_grad=requires_grad)
        ret.is_leaf=not requires_grad

        if requires_grad:
            ret.fromvars=(self,right_tensor) 
            ret.gradfuncs=(_pow_grad_left,_pow_grad_right)
        
        return ret
    
    # pow运算，右值为self，左值为TN，numpy数组，list，tuple，整数或浮点数
    def __rpow__(self,left_obj):
        if not isinstance(left_obj,TN):
            left_tensor=tensor(left_obj,device=self.device) #如左值是非TN对象，转化为TN对象
        return left_tensor.__pow__(self) #归一到左值'*'函数
        
    def __pos__(self):
        return self
        
    def __neg__(self):
        # 对于无符号整数类型，需要先将数据转换为有符号类型再取负
        # 否则在numpy中直接对uint8取负会先发生溢出，再转换类型
        data = self.data
        dtype = self.dtype
        
        if np.issubdtype(dtype, np.unsignedinteger):
            # 为不同的无符号类型选择对应的有符号类型
            if dtype == np.uint8:
                target_dtype = np.int16  # 使用int16避免溢出
            elif dtype == np.uint16:
                target_dtype = np.int32
            elif dtype == np.uint32:
                target_dtype = np.int64
            elif dtype == np.uint64:
                target_dtype = np.int64  # Python的int可以处理更大的范围
            
            # 先将数据转换为有符号类型，再执行取负操作
            data = data.astype(target_dtype)
            dtype = target_dtype
        
        # 现在取负操作会得到正确的负值
        ret = tensor(-data, dtype=dtype, device=self.device,
                    requires_grad = (is_grad_enabled() and self.requires_grad))
        ret.is_leaf = not ret.requires_grad
        if  ret.requires_grad:
            ret.fromvars=(self,)
            fn = lambda result,i: -result.grad_value
            ret.gradfuncs=(fn,)
        return ret
    
    # 比较运算符重载（返回布尔张量，不参与梯度计算）
    def __lt__(self, other):
        if other is None:
            raise TypeError(" '<' not supported between instances of 'TN' and 'NoneType'")
        other_data = other.data if isinstance(other, TN) else other        
        return tensor(self.data < other_data, device=self.device)

    def __le__(self, other):
        if other is None:
            raise TypeError(" '<=' not supported between instances of 'TN' and 'NoneType'")
        other_data = other.data if isinstance(other, TN) else other
        return tensor(self.data <= other_data, device=self.device)

    def __gt__(self, other):
        if other is None:
            raise TypeError(" '>' not supported between instances of 'TN' and 'NoneType'")
        other_data = other.data if isinstance(other, TN) else other
        return tensor(self.data > other_data, device=self.device)

    def __ge__(self, other):
        if other is None:
            raise TypeError(" '>=' not supported between instances of 'TN' and 'NoneType'")
        other_data = other.data if isinstance(other, TN) else other
        return tensor(self.data >= other_data, device=self.device)

    def __eq__(self, other):
        if other is None:
            return False
        other_data = other.data if isinstance(other, TN) else other
        return tensor(self.data == other_data, device=self.device)

    def __ne__(self, other):
        if other is None:
            return True
        other_data = other.data if isinstance(other, TN) else other
        return tensor(self.data != other_data, device=self.device)

    def __and__(self, other):
        other_data = other.data if isinstance(other,TN) else other        
        return tensor(self.data & other_data, device=self.device)

    def __or__(self, other):
        other_data = other.data if isinstance(other,TN) else other        
        return tensor(self.data | other_data, device=self.device)
    
    def __xor__(self, other):
        other_data = other.data if isinstance(other,TN) else other        
        return tensor(self.data ^ other_data, device=self.device)
    
    def __invert__(self):
        return tensor(~self.data, device=self.device)
    
    def __lshift__(self, other):
        other_data = int(other.data if isinstance(other,TN) else other)
        return tensor(self.data << other_data, device=self.device)
    
    def __rshift__(self, other):
        other_data = int(other.data if isinstance(other,TN) else other)
        return tensor(self.data >> other_data, device=self.device)
    
    def detach(self):
        """
        返回一个与当前张量共享数据但断开计算图的新张量。
        
        返回的新张量与原张量共享底层数据，但不参与原张量的计算图。
        这意味着对新张量的操作不会影响原张量的梯度计算。
        常用于在需要将张量转换为NumPy数组或进行不需要梯度的计算时。
        
        Returns:
            TN: 与原张量共享数据但断开计算图的新张量
        """
        ret = TN()
        ret.data = self.data
        return ret    
    
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
        self.is_leaf = True
        self.fromvars = ()
        self.gradfuncs = ()
        self.parms = ()
        self.grad_value = None
        self.rcv_grad_count = 0
        return self

    def clone(self):
        '''返回一个新张量，与self共享内存，依赖self'''
        ret=tensor(self.data.copy(),device=self.device,
                    requires_grad=(is_grad_enabled() and self.requires_grad))
        ret.is_leaf = not ret.requires_grad
        if ret.requires_grad:
            ret.fromvars=(self,)
            fn = lambda result,i: result.grad_value
            ret.gradfuncs=(fn,)
        return ret
    
    def copy(self):
        '''返回一个新张量，复制self数据，不共享内存，也不依赖self'''
        return self.clone().detach_()

    def copy_(self,src):
        '''原地复制src到self'''
        if self.is_leaf and self.requires_grad:
            raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation.')
        
        # 会自动调用__setitem__，触发梯度计算
        # 用空元组()作索引，表示对整个张量赋值, 注意':'也是全局索引，但不能用于0维张量
        self[()] = src
        
        return self
    
    def copy_to(self,target:TN|np.ndarray|None=None):
        '''将self数据复制到target'''
        arrtype = (np.ndarray,cp.ndarray) if cp else (np.ndarray,)

        if isinstance(target,TN):
            if target.device != self.device:
                raise ValueError(f'target device must be {self.device} but got {target.device}')
            target.copy_(self)
        elif isinstance(target,arrtype):
            if type(target) != type(self.data):
                raise TypeError(f'target type must be {type(self.data)} but got {type(target)}')
            np.copyto(target,self.data)
        elif target is None:
            target = self.clone()
        else:
            raise TypeError('target must be a TN, numpy/cupy ndarray object or None')

        return target

    def zero_(self):
        if self.is_leaf and self.requires_grad:
            raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation.')
        self.data.fill(0.)               
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
        # 检查是否是需要梯度的叶子节点
        if self.requires_grad and self.is_leaf:
            raise RuntimeError("a leaf Variable that requires grad has been used in an in-place operation")
        
        # 处理不同类型的value参数
        if isinstance(value, TN):
            if value.ndim != 0:
                raise RuntimeError(f'fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions.')
        
        # 执行原地填充操作
        self.copy_(value)
        
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
        if self.requires_grad and self.is_leaf:
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
    
    def squeeze(self, dim: int | tuple = None):  # type: ignore
        # 当dim为None时，计算所有大小为1的维度
        if dim is None:
            # 获取所有大小为1的维度
            dim = tuple(i for i, size in enumerate(self.data.shape) if size == 1)
        
        # 当dim为空元组时，直接返回原张量
        if dim == ():            
            return self
        
        if  isinstance(dim, int):
            dim = (dim,)

        # 统一检查指定维度是否都为1
        # 处理dim为元组的情况
        if isinstance(dim, tuple):
            new_dim = []
            for d in dim:
                # 处理负数索引
                actual_dim = d + self.data.ndim if d < 0 else d
                if actual_dim < 0 or actual_dim >= self.data.ndim:
                    raise IndexError(f'Dimension out of range (expected to be in range of [-2, {self.data.ndim - 1}], but got {d})')
                if self.data.shape[actual_dim] != 1:
                    # 如果任何一个指定维度大小不为1，则不执行操作
                    continue
                if actual_dim in new_dim:
                    raise RuntimeError(f'dim {actual_dim} appears multiple times in the list of dims')
                new_dim.append(actual_dim)
        else:
            raise TypeError("dim must be None, an integer, or a tuple of integers")
        
        # 根据原始张量的数据类型选择使用np或cp
        arrlib = self._get_array_lib()

        # 执行挤压操作并设置梯度信息
        new_dim = tuple(new_dim)  # type: ignore
        newarr = arrlib.squeeze(self.data, axis=new_dim)
        ret = TN()
        ret.data = newarr        
        ret.requires_grad = (is_grad_enabled() and self.requires_grad)
        ret.is_leaf = not ret.requires_grad

        if ret.requires_grad:
            ret.fromvars = (self,)
            ret.parms = (new_dim,)
            ret.gradfuncs = (_squeeze_backward,)
        return ret

    def unsqueeze(self, dim:int|tuple):
        if dim is None or dim == ():            
            return self

        # 根据原始张量的数据类型选择使用np或cp
        arrlib = self._get_array_lib()
        newarr = arrlib.expand_dims(self.data, axis=dim)
        
        ret = TN()
        ret.data = newarr        
        ret.requires_grad = (is_grad_enabled() and self.requires_grad)
        ret.is_leaf = not ret.requires_grad
        
        if ret.requires_grad:
            ret.fromvars = (self,)
            ret.parms = (dim,)
            ret.gradfuncs = (_unsqueeze_backward,)
        return ret

    def reshape(self, *new_shape):
        """
        返回具有新形状的张量，不改变其数据。
        
        支持负数维度大小（特别是-1）
        来自动计算该维度的大小，前提是其他维度的大小可以确定。
        
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
        """
        # 处理参数格式
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            shape = new_shape[0]
        else:
            shape = new_shape
        
        # 验证形状参数是否为整数
        # if builtins.any(not isinstance(dim, int) for dim in shape):
        #     raise RuntimeError(f'{shape} contains non int numbers')
        
        # shape相同是直接返回self
        if self.shape == shape:
            return self

        # 计算原始张量的元素总数
        original_size = self.numel()
        
        # 处理负数维度大小，特别是-1
        processed_shape = list(shape)
        num_neg_ones = processed_shape.count(-1)        
        if num_neg_ones > 1:
            raise RuntimeError(f"only one dimension can be inferred (got {num_neg_ones})")
        
        # 计算推断维度的大小
        if num_neg_ones == 1:
            # 计算已知维度的乘积
            known_size = 1
            unknown_dim_index = processed_shape.index(-1)
            for i, dim in enumerate(processed_shape):
                if i != unknown_dim_index:
                    if dim < 0:
                        raise RuntimeError(f"Invalid shape dimension {dim}")
                    known_size *= dim
            
            # 检查是否可以整除
            if original_size % known_size != 0:
                raise RuntimeError(
                    f"shape '{processed_shape}' is invalid for input of size {original_size}"
                )
            
            # 计算并替换-1为实际维度大小
            processed_shape[unknown_dim_index] = original_size // known_size
        else:
            # 检查是否有负维度（除了-1，但此时已经确定没有-1了）
            if any(dim < 0 for dim in processed_shape):
                invalid_dim = next(dim for dim in processed_shape if dim < 0)
                raise RuntimeError(f"Invalid shape dimension {invalid_dim}")
            
            # 计算新形状的元素总数
            new_size = math.prod(processed_shape)
            if new_size != original_size:
                raise RuntimeError(
                    f"shape '{processed_shape}' is invalid for input of size {original_size}"
                )
        
        # 记录原始形状供反向传播恢复
        original_shape = self.shape
        ret = TN()
        ret.data = self.data.reshape(tuple(processed_shape))
        ret.requires_grad = (is_grad_enabled() and self.requires_grad)
        ret.is_leaf = not ret.requires_grad
        
        if ret.requires_grad:
            ret.fromvars = (self,)
            ret.parms = ((original_shape, tuple(processed_shape)),)
            ret.gradfuncs = (_reshape_backward,)
        return ret

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

    def unfold(self, dimension: int, size: int, step: int = None) -> 'TN':  # type: ignore
        """将张量的指定维度展开为多个连续的切片。
        
        该函数用于将张量的指定维度展开为
        多个连续的窗口切片，常用于卷积操作中的im2col变换。
        
        参数:
            dimension: 要展开的维度
            size: 窗口大小
            step: 步长，默认等于size（不重叠）
        
        返回:
            TN: 展开后的张量，新张量的形状为：
                原形状[:dimension] + (num_windows,) + 原形状[dimension+1:] + (size,)
                其中 num_windows = (原形状[dimension] - size) // step + 1
        
        异常:
            ValueError: 当维度超出范围或参数不合法时抛出
        """
        # 处理默认步长
        if step is None:
            step = size
        
        # 处理负数维度索引
        dim = dimension if dimension >= 0 else self.ndim + dimension
        if dim < 0 or dim >= self.ndim:
            raise ValueError(f"Dimension {dimension} out of range for tensor with {self.ndim} dimensions")
        
        # 获取输入形状
        input_shape = list(self.shape)
        dim_size = input_shape[dim]
        
        # 检查参数合法性
        if size > dim_size:
            raise ValueError(f"Size {size} is larger than dimension size {dim_size}")
        
        # 计算窗口数量
        num_windows = (dim_size - size) // step + 1
        if (dim_size - size) % step != 0:
            warnings.warn(f"Window step {step} is not compatible with dimension size {dim_size} and window size {size}, some elements will be discarded")
        
        # 计算输出形状：原形状[:dim] + [num_windows] + 原形状[dim+1:] + [size]
        output_shape = list(input_shape)
        output_shape[dim] = num_windows
        output_shape.append(size)
        
        # 创建所有窗口
        windows = []
        for i in range(num_windows):
            # 计算当前窗口的起始和结束索引
            start = i * step
            end = start + size
            
            # 创建切片索引
            idx = [slice(None)] * self.ndim
            idx[dim] = slice(start, end)
            
            # 获取窗口
            window = self[tuple(idx)]
            
            # 调整窗口形状：将展开维度移动到最后
            # 例如：对于形状(a, size, b)，转换为(a, b, size)
            perm = list(range(self.ndim))
            perm.append(perm.pop(dim))
            window = window.permute(perm)
            
            # 在原dim位置插入1，形成窗口数量维度
            window_shape = list(window.shape)
            window_shape.insert(dim, 1)
            window = window.reshape(window_shape)
            
            # 将窗口添加到列表中
            windows.append(window)
        
        # 沿窗口数量维度堆叠所有窗口
        result = concatenate(windows, dim=dim)
        
        # 确保输出形状正确
        assert result.shape == tuple(output_shape), f"Expected shape {output_shape}, got {result.shape}"
        
        return result

    def fold(self, dimension: int, size: int, step: int = None, output_size: int = None) -> 'TN':  # type: ignore
        """
        将张量的指定维度从窗口展开状态折叠回原始形状。
        这是unfold()函数的逆操作。

        参数:
            dimension: 要折叠的维度（对应unfold时的维度）
            size: 窗口大小（与unfold时的size相同）
            step: 步长（与unfold时的step相同）
            output_size: 折叠后的目标维度大小。如果不提供，将自动计算为：
                (num_windows - 1) * step + size，其中num_windows是输入张量在dimension处的大小

        返回:
            TN: 折叠后的张量，形状为：
                原形状[:dimension] + (output_size,) + 原形状[dimension+1:-1]

        异常:
            ValueError: 当维度超出范围或参数不合法时抛出
        """
        # 处理默认步长
        if step is None:
            step = size

        # 处理负数维度索引
        dim = dimension if dimension >= 0 else self.ndim + dimension
        if dim < 0 or dim >= self.ndim:
            raise ValueError(f"Dimension {dimension} out of range for tensor with {self.ndim} dimensions")

        # 检查输入张量是否符合预期格式（最后一维应包含窗口元素）
        if self.ndim < 2:
            raise ValueError(f"Input tensor must have at least 2 dimensions, got {self.ndim}")

        # 获取输入形状
        input_shape = list(self.shape)
        num_windows = input_shape[dim]
        window_size = input_shape[-1]

        # 验证窗口大小是否匹配
        if window_size != size:
            raise ValueError(f"Window size mismatch: expected {size}, got {window_size}")

        # 计算输出维度大小
        if output_size is None:
            output_size = (num_windows - 1) * step + size

        # 构建输出形状
        output_shape = list(input_shape)
        output_shape[dim] = output_size
        output_shape.pop()  # 移除最后一维（窗口元素）

        # 创建一个形状为output_shape的零张量作为基础
        output = zeros(output_shape, dtype=self.dtype) #, requires_grad=self.requires_grad)

        # 对于每个窗口，将其内容添加到输出张量的对应位置
        for i in range(num_windows):
            # 计算当前窗口在输出张量中的起始和结束位置
            start = i * step
            end = start + size

            # 创建输出张量的切片索引
            output_idx = [slice(None)] * output.ndim
            output_idx[dim] = slice(start, end)

            # 获取当前窗口内容
            # 输入形状: [batch, channels, num_windows, window_size] 或类似
            window_idx = [slice(None)] * self.ndim
            window_idx[dim] = i  # type: ignore
            window_idx[-1] = slice(None)  # 选择所有窗口元素
            window = self[tuple(window_idx)]

            # 确保窗口形状与输出切片形状匹配
            # 窗口形状应该是: output_shape[:dim] + (size,) + output_shape[dim+1:]
            window_shape = list(output_shape)
            window_shape[dim] = size
            window = window.view(tuple(window_shape))

            # 将窗口贡献添加到输出张量（原地操作）
            output.addat_(tuple(output_idx),window)

        return output

    def flatten(self, start_dim=0, end_dim=-1) -> 'TN':
        """将张量从start_dim到end_dim的维度展平，自动支持梯度跟踪"""
        # 处理负数维度索引
        dim = self.data.ndim
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
        return self.reshape(*new_shape)

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
        
        # 转换为整数元组
        new_shape = tuple(int(s) for s in new_shape)
        
        # 处理负数维度大小，特别是-1
        original_shape = self.shape
        orig_ndim = len(original_shape)
        new_ndim = len(new_shape)
        
        # 替换-1为对应的原始维度大小
        processed_shape = list(new_shape)
        for i in range(new_ndim):
            if processed_shape[i] == -1:
                if new_ndim == orig_ndim:  # 只有当维度数量相同时，才能使用-1
                    processed_shape[i] = original_shape[i]
                else:
                    raise RuntimeError(
                        f"Cannot use -1 as size value for dimension {i} when the number of dimensions changes from {orig_ndim} to {new_ndim}"
                    )
        new_shape = tuple(processed_shape)
        
        # 对于维度数量不同的情况，如果原张量是0维，可以扩展到任意形状
        if orig_ndim != new_ndim:
            if orig_ndim == 0:  # 标量可以扩展到任意形状
                pass
            else:
                # 检查是否可以通过在前面添加1的维度来匹配新维度数
                # 例如：(4,) -> (1,4) -> 可以扩展为 (3,4)
                if new_ndim > orig_ndim:
                    # 创建扩展后的原始形状（在前面添加1）
                    expanded_orig_shape = (1,) * (new_ndim - orig_ndim) + original_shape
                    # 检查扩展后的原始形状是否与新形状兼容
                    for i, (orig_dim, new_dim) in enumerate(zip(expanded_orig_shape, new_shape)):
                        if orig_dim != 1 and orig_dim != new_dim:
                            raise RuntimeError(
                                f"The expanded size of the tensor ({new_dim}) in dimension {i} must be "
                                f"equal to the existing size ({orig_dim}) or the existing size must be one."
                            )
                else:
                    raise RuntimeError(
                        f"expand(Size({list(new_shape)})): the number of dimensions "
                        f"({new_ndim}) must be >= the number of dimensions of the tensor ({orig_ndim})"
                    )
        else:
            # 维度数量相同的情况，检查每个维度的兼容性
            for i, (orig_dim, new_dim) in enumerate(zip(original_shape, new_shape)):
                if orig_dim != 1 and orig_dim != new_dim:
                    raise RuntimeError(
                        f"The expanded size of the tensor ({new_dim}) in dimension {i} must be "
                        f"equal to the existing size ({orig_dim}) or the existing size must be one."
                    )
        
        # 根据原始张量的数据类型选择使用np或cp
        arrlib = self._get_array_lib()        
        # 使用numpy的broadcast_to执行扩展
        expanded_data = arrlib.broadcast_to(self.data, new_shape)   
        
        # 创建新的张量对象
        ret = tensor(expanded_data, 
                     requires_grad=(is_grad_enabled() and self.requires_grad))
        ret.is_leaf = not ret.requires_grad
        
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
        size = _validate_shape(size)
        return broadcast_to(self, size)
    
    def flip(self, dims:List[int]|Tuple[int,...]) -> TN:
        return flip(self,dims)

    def clamp(self,min:float|None=None, max:float|None=None):  # type: ignore
        return clamp(self,min,max)
    
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
        result_data = arrlib.all(self.data, axis=dim, keepdims=keepdim)
        
        # 创建结果张量，保持梯度传播
        ret = tensor(result_data, dtype=bool, device=self.device)  # type: ignore
        return ret

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
        result_data = arrlib.any(self.data, axis=dim, keepdims=keepdim)
        
        # 创建结果张量，保持梯度传播
        ret = tensor(result_data, dtype=bool, device=self.device)  # type: ignore
        return ret

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
        idx = self.data.argmax(axis=dim,keepdims=keepdim)
        ret = tensor(idx, dtype = int64)        
        return ret
    
    def argmin(self,dim:int|None=None,keepdim:bool=False):  # type: ignore
        idx = self.data.argmin(axis=dim,keepdims=keepdim)
        ret = tensor(idx, dtype = int64)
        return ret

    def argsort(self,dim:int=-1,descending:bool=False):  # type: ignore
        return argsort(self,dim=dim,descending=descending)

    def sort(self,dim:int=-1,descending:bool=False):  # type: ignore
        return sort(self,dim=dim,descending=descending)

    def maximum(self, other):
        return maximum(self,other)
    
    def minimum(self, other):
        return minimum(self,other)

    def sum(self, dim:int|tuple|None=None, keepdim:bool=False):  # type: ignore
        ret = sum(self,dim,keepdim)
        return ret
    
    def cumsum(self, dim: int, *, dtype: Optional[Union[str, np.dtype]] = None, out: Optional[TN] = None):
        ret = cumsum(self,dim,dtype=dtype,out=out)
        return ret
    
    def prod(self, dim:int|tuple|None=None, keepdim:bool=False):  # type: ignore
        ret = prod(self,dim,keepdim)
        return ret
    
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
        return pow(self,exponent)  # type: ignore
    
    def sqrt(self):
        return sqrt(self)

    def exp(self):
        return exp(self)
    
    def log(self):
        return log(self)

    def sin(self):
        return sin(self)

    def cos(self):
        return cos(self)

    def tan(self):
        return tan(self)
    
    def cot(self):
        return cot(self)

    def sec(self):
        return sec(self)

    def csc(self):
        return csc(self)
        
    def arcsin(self):
        return arcsin(self)

    def arccos(self):
        return arccos(self)

    def arctan(self):
        return arctan(self)

    def sinh(self):
        return sinh(self)
        
    def cosh(self):
        return cosh(self)
        
    def tanh(self):
        return tanh(self)

    def coth(self):
        return coth(self)

    def sech(self):
        return sech(self)
    
    def csch(self):
        return csch(self)

    def arcsinh(self):
        return arcsinh(self)
        
    def arccosh(self):
        return arccosh(self)
        
    def arctanh(self):
        return arctanh(self)

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
        from .linalg import inv
        return inv(self)
    
    def pinverse(self):
        from .linalg import pinv
        return pinv(self)

    def det(self):
        from .linalg import det
        return det(self)

    def norm(self, ord:int|float|str|None = "fro", dim: int|tuple|None = None, keepdim: bool = False):  # type: ignore
        from .linalg import norm as linalg_norm
        return linalg_norm(self, ord=ord, dim=dim, keepdim=keepdim)

    def _init_calc_graph(self):
        '''初始化以self为根的计算图中的缓存信息，包括：grad_value、rcv_grad_count           
           该函数被backward和grad函数调用，用于：
           1、将从self开始的反向计算图中所有节点的梯度缓存grad_value置None
           2、初始化各节点应收到反向传播梯度的次数rcv_grad_count
           调用该函数前假定计算图中的缓存信息已清理，清理时机：
           1、张量初始化函数__init__()里会初始化缓存信息
           2、backward函数会遍历self为根的计算图中所有节点，执行过程中节点缓存被自动清理
           3、grad函数不会遍历self为根的计算图中所有节点，执行结束前要调用_clear_calc_graph_cache主动清理
        '''
        stack = []  #初始化一个栈
        self.rcv_grad_count = 0
        stack.append(self)
        while stack:
            item:TN = stack.pop(-1)  #pop(-1)效率o(1),表示从list尾弹出
            item.grad_value = None  # type: ignore # 梯度值置None

            # 叶子节点不要再往下遍历
            if item.is_leaf:                    
                continue

            varlist = item.fromvars
            for var in varlist:                
                if var.requires_grad:
                    #首次遍历到的节点入队列，避免同一节点多次入队列导致rcv_grad_count计数出错
                    if var.rcv_grad_count == 0:
                        stack.append(var)
                    var.rcv_grad_count += 1 #累计当前节点可接受梯度的次数
                
        return

    def _init_calc_graph_(self):
        '''初始化以self为根的计算图中的缓存信息，包括：grad_value、rcv_grad_count           
           该函数被backward和grad函数调用，用于：
           1、将从self开始的反向计算图中所有节点的梯度缓存grad_value置None
           2、初始化各节点应收到反向传播梯度的次数rcv_grad_count
           该函数_init_calc_graph的区别：
           1、不假定计算图中的缓存信息已清理干净
           2、性能比_init_calc_graph略差，但逻辑上初始化操作更完备
           3、如backward和grad函数调用此函数初始化计算图，
              执行结束可以不调用_clear_calc_graph_cache主动清理grad_value、rcv_grad_count
        '''
        stack = []  #初始化一个栈
        visited = set()        
        self.rcv_grad_count = 0
        stack.append(self)
        visited.add(self)

        while stack:
            item:TN = stack.pop(-1)  #pop(-1)效率o(1),表示从list尾弹出
            item.grad_value = None  # type: ignore # 梯度值置None

            # 叶子节点不要再往下遍历
            if item.is_leaf:                    
                continue

            varlist = item.fromvars
            for var in varlist:
                if var.requires_grad:
                    if var not in visited:
                        var.rcv_grad_count = 0
                        stack.append(var)
                        visited.add(var)
                    var.rcv_grad_count += 1 #累计当前节点可接受梯度的次数
            
        return

    def _clear_calc_graph_cache(self):
        '''请理以self为根的计算图
           将所有节点的梯度置None、初始化各节点应收到反向传播梯度的次数
        '''

        stack = []  #初始化一个栈
        stack.append(self) 

        while stack:
            item:TN = stack.pop(-1)
            item.rcv_grad_count = 0
            item.grad_value = None  # type: ignore

            # 叶子节点不要再往下遍历
            if item.is_leaf: 
                continue

            varlist = item.fromvars
            for var in varlist:
                if var.requires_grad:
                    stack.append(var)
        
        return
    
    def _addto_grad(self:TN, target:TN, create_graph:bool):  # type: ignore
        '''将self添加到target的grad中
        '''
        # self = self.type(target.dtype)
        if target.grad is None:
            # 如还没有梯度值，直接赋值
            if create_graph:
                target.grad = self
            else:
                # 如果不保存梯度的计算图信息，self清除计算图信息后赋值给target.grad_value
                target.grad = self.detach_()
        else:
            # 如已有梯度值，累计梯度
            if create_graph:
                # 如果梯度也保存计算图信息，用张量加法，但不能用原地+=
                target.grad = target.grad_value + self
            else:
                # 如果不保存梯度的计算图信息，直接对data原地加法
                target.grad.data += self.data

        return

    def _addto_grad_value(self:TN, target:TN, create_graph:bool):  # type: ignore
        '''将self添加到target的grad_value中
        '''

        # 确保self和target的dtype一致
        # self是待累加的梯度，由于计算过程中精度会出现提升现象(float32 to float64)
        # 及时精度转换虽然可能会导致梯度张量被复制到一个新类型，但由于保证了梯度一直是低级度，
        # 计算性能反而会提升
        if self.dtype != target.dtype:
            warnings.warn(f'self.dtype={self.dtype}, target.dtype={target.dtype}')
            self = self.type(target.dtype)
        
        # 处理特殊情况：梯度self是大小为0空张量(如[])，target是单元素张量(如1.、[1.]、[[1.]])
        # 处理空张量某些运算中梯度反向传播也为空张量的情况，比如给空张量原地赋值单元素张量时，单元素张量收到的梯度会是空张量，
        # 梯度需要设置为和单元张量形状一致的0张量
        if self.numel()==0 and target.numel()==1:
            self = zeros_like(target)
        
        if target.grad_value is None:
            # 如还没有梯度值，直接赋值
            if create_graph:
                target.grad_value = self
            else:
                # 如果不保存梯度的计算图信息，self清除计算图信息后赋值给target.grad_value
                target.grad_value = self.detach_()
        else:
            # 如已有梯度值，累计梯度
            if create_graph:
                # 如果梯度也保存计算图信息，用张量加法，但不能用原地+=
                target.grad_value = target.grad_value + self
            else:
                # 如果不保存梯度的计算图信息，直接对data原地加法
                target.grad_value.data += self.data

        return

    def backward(self, gradient: TN|None = None, 
                 retain_graph: bool = False,  # type: ignore
                 create_graph: bool = False):  # type: ignore
        """执行自动微分的反向传播计算。

        该函数从当前张量开始，沿计算图反向传播梯度，计算并存储所有
        叶子节点或设置了retains_grad=True的中间节点的梯度。

        参数:
            gradient (TN | None, 可选): 输出张量的梯度，默认为None
            retain_graph (bool, 可选):  本参数为兼容Pytorch的设计，riemann反向传播不依赖该参数，
                                       无论True还是False，riemann都支持多次反向传播
                                       该参数表示是否在反向传播后保留计算图，设为True时可用于多次反向传播，默认为False
            create_graph (bool, 可选):  是否在梯度计算中创建计算图，
                                       设为True时可用于高阶导数计算，默认为False

        返回:
            None: 该函数不返回值，梯度结果直接存储在各节点的.grad属性中

        异常:
            RuntimeError: 当当前张量是叶子节点、当前张量不是标量、
                        或当前张量不需要梯度时抛出
        """
        if not self.requires_grad:
            raise RuntimeError('Only a tensor require grad can call backward()')
        
        if gradient is None:
            if self.data.ndim > 0:
                raise RuntimeError('Only a scalar can call backward() without argument grad_outputs')
            if self.is_complex():
                raise RuntimeError(f'grad can be implicitly created only for real scalar outputs but got {self.dtype}')
        
            # 使用self作为反向传播的起点
            self._init_calc_graph()
            # 确保默认梯度张量与当前张量在相同的设备上
            self.grad_value = tensor(1.0, dtype=self.dtype, device=self.device, requires_grad=create_graph)
        
        elif isinstance(gradient,TN):
            if gradient.data.shape == self.data.shape:
                # 直接使用原始self，设置其grad_value为gradient
                self._init_calc_graph()
                self.grad_value = gradient.to(self.device).detach().requires_grad_(create_graph)            
            else:
                raise RuntimeError('shape of gradient need to be same as the shape of outputs')
        else:
            raise TypeError(f'gradient can be either tensor or None, but got {type(gradient)}')
        
        stack = []  # 初始化用于存放收集完梯度的节点的堆栈
        stack.append(self)
                
        while stack:  #对栈循环处理直至为空
            item:TN = stack.pop(-1)    #pop(-1)效率o(1),表示从栈尾弹出
            
            # 对于叶子节点或要求保存梯度的中间节点，保存grad
            if item.is_leaf or item.retains_grad:
                item.grad_value._addto_grad(item,create_graph)
            
            fromvars = item.fromvars
            gradfuncs = item.gradfuncs

            #向来源节点传播梯度
            #fromvars和gradfuncs是等长并元素一一对应的，所以可以在一个循环中处理
            # for i,(var,fn) in enumerate(zip(fromvars,gradfuncs)):
            for i in range(len(fromvars)):
                var:TN = fromvars[i]
                fn = gradfuncs[i]
                if var.requires_grad == True:
                    #调用来源节点对应的梯度函数，向该节点传播梯度值,每接收一次梯度传递，计数减1
                    tobe_add_grad:TN = fn(item,i)
                    tobe_add_grad._addto_grad_value(var,create_graph)

                    var.rcv_grad_count -= 1   # 收到一次梯度，计数-1
                    
                    # 如果节点var的梯度已收集完毕，将该节点加入stack，以便后续继续反向传播梯度
                    # stack中永远只存放收集完梯度的节点
                    if var.rcv_grad_count == 0:
                        stack.append(var)
            
            # 节点反向传播完梯度后，grad_value清空，节省空间
            item.grad_value = None  # type: ignore
        #end of while

        return
    
    def d(self,*vars,create_graph = False):
        """
        计算当前标量张量对指定多个标量张量的混合偏导数。
        
        参数:
            *vars: 一个或多个张量，用于计算当前张量对其的混合偏导数
        
        返回值:
            TN: 计算得到的混合偏导数标量张量
        
        异常:
            ValueError: 当指定的变量中包含非张量元素、不是标量张量或未提供任何变量时抛出
        """

        from .autograd import grad
        
        # 检查self是否为标量
        if self.ndim > 0 or self.requires_grad == False:
            raise ValueError("self must be a scalar tensor requires grad")
        
        # 检查是否提供了至少一个变量
        if not vars:
            raise ValueError("At least one variable must be provided to compute derivatives")
                
        # 验证输入参数是否都是张量类型且为标量
        all_0d = True
        for var in vars:
            if not isinstance(var, TN):
                raise ValueError(f"Input variable must be TN type, got {type(var)}")
            if var.ndim > 0:
                all_0d = False
        if not all_0d:
            raise ValueError("All input variables must be 0-dimensional tensors")
        
        # 初始化当前梯度为self
        current_grad = self
        
        # 依次对每个变量求偏导数，计算混合偏导数
        # 最后一次求导单独处理以提高效率
        for i, var in enumerate(vars):
            # 判断是否为最后一次求导
            is_last = (i == len(vars) - 1)
                        
            # 最后一次求导时根据create_graph参数决定是否创建计算图
            next_grad = grad(current_grad, var, 
                             create_graph=not is_last or create_graph,
                             allow_unused=True)[0]
            
            # 如果导数为None，终止后续求导计算
            if next_grad is None:
                # 返回与self同数据类型的0张量
                return zeros_like(self)
            
            current_grad = next_grad
        
        # 如果最终梯度为None，返回0张量
        if current_grad is None:
            return zeros_like(self)
        
        # 非最后一次求导时已经处理了create_graph参数，这里不再需要单独detach
        return current_grad
#end of class

def _get_device(device:str|int|Device=None)->Device:
    """
    获取设备，该函数根据输入的设备参数，返回一个Device对象。
    
    参数:
        device: 可选，指定设备的字符串表示、整数索引或Device对象。
                可以是'cpu'、'cuda'、'cuda:0'、从0开始的整数等。
                如果为None，则默认使用CPU。
    
    返回值:
        Device: 包含两个属性，type属性为‘cpu'或'cuda'，
                index属性为CUDA设备索引（CPU为None）。
    
    异常:
        RuntimeError: 当指定的device无效或不可用时抛出
    """ 

    if device is not None:
        # 先将字符串或Device对象统一转换为Device对象
        if isinstance(device, Device):
            return_device = device
        elif isinstance(device, (str,int)):
            return_device = Device(device)
        else:
            raise RuntimeError(f"Invalid device type: {type(device).__name__}")
    else:
        # device参数为None，优先级：device上下文 > 默认设备
        if CUPY_AVAILABLE and is_in_cuda_context():
            # 当前线程在CUDA设备上下文中，使用当前CUDA设备
            target_device_idx = cp.cuda.runtime.getDevice()
            return_device = Device(target_device_idx)
        else:
            # 否则使用默认设备
            return_device = get_default_device()
    
    return return_device

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

def tensor(data, dtype:np.dtype|None = None, device:str|int|Device|None = None, requires_grad:bool|None = False)->TN:
    """
    创建一个新的张量对象。
    
    该函数是Riemann框架中创建张量的主要入口点,
    它将输入数据转换为张量，并可选择设置数据类型和是否需要梯度计算。
    
    参数:
        data: 可以是任意可转换为numpy数组的数据，包括列表、元组、标量、numpy数组等
        dtype: 可选，指定张量的数据类型。如果为None，则保留原始数据类型
        device: 可选，指定张量所在的设备，可以是'cpu'、'cuda'、device对象或None
        requires_grad: 可选，布尔值，指定是否需要计算该张量的梯度，默认为False
            
    返回值:
        TN: 新创建的张量对象
    
    异常:
        RuntimeError: 当requires_grad=True但数据类型不是浮点型时抛出
        RuntimeError: 当指定的device无效或不可用时抛出
    
    示例:
        >>> # 从列表创建CPU张量
        >>> x = tensor([1, 2, 3])
        >>> # 创建GPU张量
        >>> y = tensor([1.0, 2.0, 3.0], device='cuda')
        >>> # 创建浮点型张量并启用梯度
        >>> z = tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> # 指定数据类型和设备
        >>> w = tensor([1, 2, 3], dtype=float64, device='cpu')
    
    注意事项:
        1. 当dtype与输入数据类型不同时，会创建数据副本，张量与原始numpy数组不共享内存
        2. 只有浮点型张量才能启用梯度计算
        3. 创建的张量默认是计算图的叶节点
        4. 当指定device为'cuda'但CuPy不可用时，会抛出RuntimeError
    """

    tsobj = TN()    # 初始化空对象实例
    
    # 解析设备索引
    dev = _get_device(device)
    use_cuda, target_device_idx = (dev.type == 'cuda'), dev.index
    
    # 根据CUPY_AVAILABLE决定检查的数组类型
    array_type = (np.ndarray, cp.ndarray) if CUPY_AVAILABLE else (np.ndarray,)  # type: ignore
    if isinstance(data, array_type):
        if dtype is None:
            if use_cuda:
                if isinstance(data, np.ndarray):
                    # numpy数组迁移到目标CUDA设备
                    with cp.cuda.Device(target_device_idx):
                        arr = cp.asarray(data)
                else:
                    # 检查cupy数组所在的设备是否与目标设备相同
                    # 所有CuPy数组都有device属性，因此无需hasattr检查
                    if data.device.id != target_device_idx:
                        # 数组在不同设备上，迁移到目标设备
                        with cp.cuda.Device(target_device_idx):
                            arr = cp.asarray(data, dtype=data.dtype)
                    else:
                        arr = data  # 已经是当前设备上的cp.ndarray
            else: 
                if isinstance(data, np.ndarray):
                    arr = data  # 已经是np.ndarray
                else:
                    # cupy数组迁移到cpu
                    arr = cp.asnumpy(data)
                                    
        else: # dtype is not None
            if use_cuda:
                if isinstance(data, np.ndarray):
                    # numpy数组迁移到目标CUDA设备，同时转换dtype
                    with cp.cuda.Device(target_device_idx):
                        arr = cp.asarray(data, dtype=dtype)
                else:
                    # 检查cupy数组所在的设备是否与目标设备相同
                    # 所有CuPy数组都有device属性，因此无需hasattr检查
                    if data.device.id != target_device_idx:
                        # 数组在不同设备上，迁移到目标设备
                        with cp.cuda.Device(target_device_idx):
                            arr = cp.asarray(data, dtype=dtype)
                    else:
                        # 数组在当前设备上，直接转换dtype
                        arr = data.astype(dtype)
            else:
                if isinstance(data, np.ndarray):
                    arr = data.astype(dtype)                    
                else:
                    # cupy数组迁移到cpu
                    arr = cp.asnumpy(data).astype(dtype)
    else:
        # data是python数据类型时转换为相应数组    
        if use_cuda:
            # 在目标CUDA设备上创建新数组
            with cp.cuda.Device(target_device_idx):
                if dtype is None:
                    arr = cp.array(data, dtype=infer_data_type(data))
                else:
                    arr = cp.array(data, dtype=dtype)
        else:
            # 创建numpy数组
            if dtype is None:
                arr = np.array(data, dtype=infer_data_type(data))
            else:
                arr = np.array(data, dtype=dtype)
    
    tsobj.data = arr    
    tsobj.requires_grad = bool(requires_grad)
    if requires_grad:
        if not is_float_or_complex(tsobj.data.dtype):
            raise RuntimeError('Only floating point tensors can require gradients')
    return tsobj

def from_numpy(arr:np.ndarray)->TN:
    """
    从numpy数组创建张量，与原数组共享内存。
    
    该函数将现有的numpy数组转换为Riemann张量对象，与numpy数组共享底层数据内存。
    这意味着对张量的修改会影响原始numpy数组，反之亦然。
    
    参数:
        arr: numpy数组，必须包含数值类型的数据
    
    返回值:
        TN: 新创建的张量对象，与输入的numpy数组共享内存
    
    异常:
        TypeError: 当输入不是numpy数组或数组包含非数值类型数据时抛出
    
    示例:
        >>> import numpy as np
        >>> arr = np.array([1.0, 2.0, 3.0])
        >>> x = from_numpy(arr)
        >>> x[0] = 0  # 修改张量会影响原始numpy数组
        >>> print(arr)  # 输出: [0. 2. 3.]
    
    注意事项:
        1. 创建的张量默认requires_grad=False，不追踪梯度
        2. 张量与原始numpy数组共享内存，修改一方会影响另一方
        3. 输入数组必须是数值类型（整数、浮点数、复数等）
    """
    array_type = (np.ndarray, cp.ndarray) if CUPY_AVAILABLE else (np.ndarray,)  # type: ignore
    if not isinstance(data, array_type):
        raise TypeError("array need to be numpy or cupy array")
    
    if not is_numeric_array(arr):
        raise TypeError("dtype of array need to be numberic")
    
    return tensor(arr)

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
    if any(not isinstance(dim, (int,np.integer)) for dim in newshape):
        raise RuntimeError(f'{shape} contains non int numbers')
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
    return tensor(arr,device=dev,requires_grad=requires_grad)
    
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
        arr = np.zeros_like(tsr.data,dt)
    else:
        with cp.cuda.Device(dev.index):
            arr = cp.zeros_like(tsr.data,dt)
    return tensor(arr,device=dev,requires_grad=requires_grad)

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
    return tensor(arr,device=dev,requires_grad=requires_grad)

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
        arr = np.ones_like(tsr.data,dt)
    else:
        with cp.cuda.Device(dev.index):
            arr = cp.ones_like(tsr.data,dt)
    return tensor(arr,device=dev,requires_grad=requires_grad)

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
    return tensor(arr,device=dev,requires_grad=requires_grad)

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
        arr = np.empty_like(tsr.data,dt)
    else:
        with cp.cuda.Device(dev.index):
            arr = cp.empty_like(tsr.data,dt)
    return tensor(arr,device=dev,requires_grad=requires_grad)

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
    return tensor(arr,device=dev,requires_grad=requires_grad) 

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
        arr = np.full_like(tsr.data,fill_value,dt)
    else:
        with cp.cuda.Device(dev.index):
            arr = cp.full_like(tsr.data,fill_value,dt)
    return tensor(arr,device=dev,requires_grad=requires_grad)

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
    return tensor(arr,device=dev,requires_grad=requires_grad)

def rand(*size, requires_grad=False, dtype:np.dtype|None = None,device:str|int|Device|None=None) -> TN:
    """
    创建一个填充了[0,1)均匀分布随机数的张量。
    
    返回一个形状为指定大小的张量，元素是从[0,1)区间均匀分布的随机数。
    
    Args:
        *size: 张量的形状，可以是一个整数序列或一个元组/列表
        requires_grad (bool, optional): 是否需要计算梯度，默认为False
        dtype (np.dtype, optional): 张量的数据类型，如果为None则使用默认数据类型
        device(str|int|Device,optional): 可选，指定张量所在的设备，可以是'cpu'、'cuda'、CUDA设备索引、device对象或None
        
    Returns:
        TN: 填充了[0,1)均匀分布随机数的张量
        
    Examples:
        >>> rand(3, 4)  # 创建3x4的张量，元素为[0,1)的随机数
        >>> rand((2, 3))  # 创建2x3的张量，元素为[0,1)的随机数
    """
    shape = _validate_shape(size)
    dt = get_default_dtype() if dtype is None else dtype
    dev = _get_device(device)
    if dev.type=='cpu':
        data = np.random.rand(*shape).astype(dt)
    else:
        with cp.cuda.Device(dev.index):
            data = cp.random.rand(*shape,dtype=dt)
    return tensor(data, device=dev, requires_grad=requires_grad)

def randn(*size, requires_grad=False, dtype:np.dtype|None = None,device:str|int|Device|None=None) -> TN:
    """
    创建一个填充了标准正态分布随机数的张量。
    
    返回一个形状为指定大小的张量，元素是从标准正态分布（均值为0，标准差为1）中抽取的随机数。
    
    Args:
        *size: 张量的形状，可以是一个整数序列或一个元组/列表
        requires_grad (bool, optional): 是否需要计算梯度，默认为False
        dtype (np.dtype, optional): 张量的数据类型，如果为None则使用默认数据类型
        device(str|int|Device,optional): 可选，指定张量所在的设备，可以是'cpu'、'cuda'、CUDA设备索引、device对象或None
        
    Returns:
        TN: 填充了标准正态分布随机数的张量
        
    Examples:
        >>> randn(3, 4)  # 创建3x4的张量，元素为标准正态分布的随机数
        >>> randn((2, 3))  # 创建2x3的张量，元素为标准正态分布的随机数
    """
    shape = _validate_shape(size)
    dt = get_default_dtype() if dtype is None else dtype
    dev = _get_device(device)
    if dev.type=='cpu':
        data = np.random.randn(*shape).astype(dt)
    else:
        with cp.cuda.Device(dev.index):
            data = cp.random.randn(*shape,dtype=dt)
    return tensor(data,device=dev, requires_grad=requires_grad)

def randint(low: int, high = None, size = None, dtype = int64,device:str|int|Device|None=None,requires_grad = False) -> TN:
    """
    创建一个填充了指定区间随机整数的张量。
    
    返回一个形状为指定大小的张量，元素是从[low, high)区间均匀分布的随机整数。
    
    Args:
        low (int, optional): 随机整数的最小值（包含），默认为0
        high (int): 随机整数的最大值（不包含）
        size: 张量的形状，可以是一个整数序列或一个元组/列表
        requires_grad (bool, optional): 是否需要计算梯度，默认为False
        dtype (np.dtype, optional): 张量的数据类型，默认为int64
        device(str|int|Device,optional): 可选，指定张量所在的设备，可以是'cpu'、'cuda'、CUDA设备索引、device对象或None
        
    Returns:
        TN: 填充了[low, high)区间随机整数的张量
        
    Examples:
        >>> randint(10, size=(3, 4))  # 创建3x4的张量，元素为0到9的随机整数
        >>> randint(0, 10, (3, 4))  # 创建3x4的张量，元素为0到9的随机整数
        >>> randint(5, 15, 6)  # 创建长度为6的一维张量，元素为5到14的随机整数
    """
    # 处理PyTorch和传统风格的调用方式
    # 检查是否是PyTorch风格的调用方式：randint(high, size)
    # 这种情况下，high参数实际上是size，low参数实际上是high
    if size is None:
        if high is None:
            # 调用方式为randint(high)，这是不允许的
            raise ValueError("Either high or high and size must be specified")
        elif not isinstance(high, (int, np.integer)):
            # 调用方式为randint(high, size)，其中size是一个非整数类型（如元组）
            size = high
            high = low
            low = 0
        elif high < low:
            # 调用方式为randint(high, size)，其中size是一个整数
            # 由于high < low，这不可能是传统的randint(low, high, size)调用方式
            size = high
            high = low
            low = 0
    elif high is None:
        # 调用方式为randint(high, size=size)
        high = low
        low = 0
    
    dt = int64 if dtype is None else dtype
    dev = _get_device(device)
    if dev.type=='cpu':
        data = np.random.randint(low, high, size=size).astype(dt)
    else:
        with cp.cuda.Device(dev.index):
            data = cp.random.randint(low, high, size=size,dtype=dt)
    return tensor(data,device=dev, requires_grad=requires_grad)

def randperm(n: int, requires_grad=False, dtype=int64,device:str|int|Device|None=None) -> TN:
    """
    创建一个包含0到n-1随机排列的张量。
    
    返回一个长度为n的一维张量，包含从0到n-1的整数，这些整数以随机顺序排列。
    
    Args:
        n (int): 排列的长度，生成的张量将包含从0到n-1的整数
        requires_grad (bool, optional): 是否需要计算梯度，默认为False
        dtype (np.dtype, optional): 张量的数据类型，默认为int64
        device(str|int|Device,optional): 可选，指定张量所在的设备，可以是'cpu'、'cuda'、CUDA设备索引、device对象或None
        
    Returns:
        TN: 包含0到n-1随机排列的一维张量
        
    Examples:
        >>> randperm(5)  # 可能返回[3, 1, 4, 0, 2]这样的随机排列
        >>> randperm(10)  # 返回长度为10的0到9的随机排列
    """
    dt = int64 if dtype is None else dtype
    dev = _get_device(device)
    if dev.type=='cpu':
        data = np.random.permutation(n).astype(dt)
    else:
        with cp.cuda.Device(dev.index):
            # cupy.random.permutation不支持dtype参数，先生成排列再转换数据类型
            data = cp.random.permutation(n).astype(dt)
    return tensor(data, device=dev, requires_grad=requires_grad)

def normal(mean:float,std:float,size:int|tuple,dtype:np.dtype|None = None,device:str|int|Device|None=None,requires_grad=False)->TN: 
    """
    创建一个填充了指定正态分布随机数的张量。
    
    返回一个形状为指定大小的张量，元素是从指定均值和标准差的正态分布中抽取的随机数。
    
    Args:
        mean (float): 正态分布的均值
        std (float): 正态分布的标准差
        size (int|tuple): 张量的形状，可以是一个整数或一个元组/列表
        dtype (np.dtype, optional): 张量的数据类型，如果为None则使用默认数据类型
        device(str|int|Device,optional): 可选，指定张量所在的设备，可以是'cpu'、'cuda'、CUDA设备索引、device对象或None
        requires_grad (bool, optional): 是否需要计算梯度，默认为False
        
    Returns:
        TN: 填充了指定正态分布随机数的张量
        
    Examples:
        >>> normal(0, 1, (3, 4))  # 创建3x4的张量，元素为标准正态分布的随机数
        >>> normal(2, 0.5, 5)  # 创建长度为5的一维张量，元素为均值为2、标准差为0.5的正态分布随机数
    """
    dt = get_default_dtype() if dtype is None else dtype
    dev = _get_device(device)
    if dev.type=='cpu':
        data = np.random.normal(mean,std,size).astype(dt)
    else:
        with cp.cuda.Device(dev.index):
            data = cp.random.normal(mean,std,size,dtype=dt)
    return tensor(data,device=dev, requires_grad=requires_grad)

# 添加arange函数
def arange(start: float, end: float | None = None, step: float = 1.0, dtype: np.dtype | None = None, device: str|int|Device|None = None, requires_grad: bool = False) -> TN:
    """
    创建一个一维张量，包含从start到end（不包括end）的等差序列。
    
    参数:
        start: 序列的起始值
        end: 序列的结束值（不包含）。如果省略，start将被视为end，而start将变为0
        step: 相邻两个元素之间的步长，默认为1.0
        dtype: 返回张量的数据类型。如果未指定，将从其他输入推断
        device(str|int|Device,optional): 可选，指定张量所在的设备，可以是'cpu'、'cuda'、CUDA设备索引、device对象或None
        requires_grad: 是否需要计算梯度，默认为False
    
    返回:
        包含等差序列的一维张量
    """
    # 处理参数，支持arange(end)的调用形式
    if end is None:
        end = start  # 先保存用户输入
        # 根据用户输入类型决定起始值类型
        start = 0 if isinstance(end, int) else 0.0  
            
    # 如果没有指定dtype，根据输入类型推断
    if dtype is None:
        # 检查所有输入是否都是整数
        is_all_integer = isinstance(start, int) and isinstance(end, int) and isinstance(step, int)
        if is_all_integer:
            # 所有输入都是整数时使用int64
            dt = np.int64
        else:
            # 否则使用默认浮点类型
            dt = get_default_dtype()
    else:
        dt = dtype

    # 使用numpy/cupy的arange创建数组
    dev = _get_device(device)
    arrlib = np if dev.type == 'cpu' else cp
    data = arrlib.arange(start, end, step, dtype=dt)
    
    # 创建并返回张量
    return tensor(data, device=dev, requires_grad=requires_grad)

# 添加linspace函数
def linspace(start: float, end: float, steps: int = 100, endpoint: bool = True, dtype: np.dtype | None = None, device: str|int|Device|None = None, requires_grad: bool = False) -> TN:
    """
    创建一个一维张量，包含从start到end的均匀间隔的值。
    
    参数:
        start: 序列的起始值
        end: 序列的结束值
        steps: 生成的样本数量，默认为100
        endpoint: 如果为True，序列包含end；否则不包含，默认为True
        dtype: 返回张量的数据类型。如果未指定，将从其他输入推断
        device(str|int|Device,optional): 可选，指定张量所在的设备，可以是'cpu'、'cuda'、CUDA设备索引、device对象或None
        requires_grad: 是否需要计算梯度，默认为False
    
    返回:
        包含均匀间隔值的一维张量
    """
    # 如果没有指定dtype，使用默认浮点类型
    dt = get_default_dtype() if dtype is None else dtype
    
    # 使用numpy的linspace创建数组
    dev = _get_device(device)
    arrlib = np if dev.type == 'cpu' else cp
    data = arrlib.linspace(start, end, num=steps, endpoint=endpoint, dtype=dt)
    
    # 创建并返回张量
    return tensor(data, device=dev, requires_grad=requires_grad)

# 定义函数类型别名
ForwardFunc: TypeAlias = Callable[..., Any]  # 前向函数类型：输入任意数量的参数，返回任意类型
GradFunc: TypeAlias = Callable[..., tuple[TN, ...]]  # 梯度函数类型：输入任意数量的参数，返回包含每个输入梯度的元组
BackwardFunc: TypeAlias = Callable[[TN, int], TN]  # 反向函数类型：输入一个TN和一个int，返回一个TN
DecoratorFunc: TypeAlias = Callable[[Callable[..., Any]], Callable[..., TN]]  # 修饰器类型：输入一个返回任意类型的函数，返回一个返回TN的函数

def track_grad(grad_func:GradFunc)->DecoratorFunc:
    """
    创建一个梯度跟踪修饰器，用于为函数添加自动微分支持。
    
    这个修饰器工厂接收一个梯度函数，返回一个修饰器，该修饰器可以将普通的张量运算函数
    转换为支持自动微分的函数。它会自动创建反向传播函数，并管理梯度计算图的构建。
    
    Args:
        grad_func (GradFunc): 梯度计算函数，接收与前向函数相同的输入参数，
            返回一个元组，包含每个输入张量对应的梯度（偏导数）
            元组内元素需要与前向函数的输入张量一一对应，对于不需要梯度的张量，对应的梯度值应为None
            
    Returns:
        DecoratorFunc: 一个修饰器函数，用于包装
            前向计算函数，使其支持自动微分
    
    Examples:
        >>> # 定义单输入导数函数（d/dx log(x) = 1/x）
        >>> def _log_derivative(x: TN) -> tuple[TN]:
        ...     return (1. / x.conj(),)
        >>> 
        >>> # 使用track_grad修饰器创建支持自动微分的对数函数
        >>> @track_grad(_log_derivative)
        ... def mylog(x: TN) -> TN:
        ...     return tensor(np.log(x.data))
        >>> 
        >>> # 使用带自动微分的对数函数
        >>> x = tensor(2., requires_grad=True)
        >>> y = mylog(x)
        >>> y.backward()
        >>> print(f'x.grad = {x.grad}')  # 输出: x.grad = 0.5
        
        >>> # 定义多输入导数函数（d/dx (x + y) = 1, d/dy (x + y) = 1）
        >>> def _add_derivative(x: TN, y: TN) -> tuple[TN, TN]:
        ...     return (tensor(1.), tensor(1.))
        >>> 
        >>> # 使用track_grad修饰器创建支持自动微分的加法函数
        >>> @track_grad(_add_derivative)
        ... def myadd(x: TN, y: TN) -> TN:
        ...     return tensor(x.data + y.data)
        >>> 
        >>> # 使用带自动微分的加法函数
        >>> x = tensor(2., requires_grad=True)
        >>> y = tensor(3., requires_grad=True)
        >>> z = myadd(x, y)
        >>> z.backward()
        >>> print(f'x.grad = {x.grad}')  # 输出: x.grad = 1.0
        >>> print(f'y.grad = {y.grad}')  # 输出: y.grad = 1.0
    
    工作原理：
        1. 接收一个前向计算函数forward_func，包装它为支持梯度跟踪的函数
        2. 自动创建反向传播函数，用于计算梯度值（利用链式法则：grad = result.grad_value * grad_func(x)）
        3. 包装后的函数会管理梯度计算所需的元数据（requires_grad, is_leaf, fromvars, gradfuncs）
        4. 当进行反向传播时，会自动调用backward_func计算梯度
    """
    def decorator(forward_func:ForwardFunc)->ForwardFunc:
        def wrapper(*xs, **kwargs)->TN:
            # 调用前向函数
            ret_val = forward_func(*xs, **kwargs)
            
            # 如果返回值不是TN类型，将其转换为TN类型
            if not isinstance(ret_val, TN):
                ret = tensor(ret_val)
            else:
                ret = ret_val
            
            # 合并所有参数
            all_params = list(xs) + list(kwargs.values())
            
            # 筛选出所有TN类型的参数
            tn_params = []
            tn_param_indices = []
            for i, param in enumerate(all_params):
                if isinstance(param, TN):
                    tn_params.append(param)
                    tn_param_indices.append(i)
            
            # 设置梯度跟踪标志
            ret.requires_grad = (is_grad_enabled() and any(x.requires_grad for x in tn_params))
            ret.is_leaf = not ret.requires_grad

            if ret.requires_grad:
                # 只保留需要梯度的TN类型输入张量及其原始索引
                grad_required_tn_indices = [i for i, x in enumerate(tn_params) if x.requires_grad]
                fromvars = tuple(tn_params[i] for i in grad_required_tn_indices)
                
                # 为每个需要梯度的TN输入创建对应的反向梯度跟踪函数
                def create_backward_func():
                    def backward_func(result_tensor: TN, index_in_fromvars: int) -> TN:
                        # index_in_fromvars是fromvars中的索引，需要根据它找到对应的TN参数索引
                        tn_index = grad_required_tn_indices[index_in_fromvars]
                        # 再根据TN参数索引找到对应的原始参数索引
                        original_index = tn_param_indices[tn_index]
                        # 同时传递位置参数和关键字参数给梯度函数
                        grad_values = grad_func(*xs, **kwargs)
                        # 确保返回值是元组
                        if not isinstance(grad_values, tuple):
                            grad_values = (grad_values,)
                        
                        return result_tensor.grad_value * grad_values[original_index]
                    return backward_func
                
                gradfuncs = tuple(
                    create_backward_func() for _ in range(len(grad_required_tn_indices))
                )
                ret.fromvars = fromvars
                ret.gradfuncs = gradfuncs
            return ret
        return wrapper
    return decorator
# end of track_grad

def broadcast_to(input: TN, size: Tuple[int, ...]) -> TN:
    """
    将输入张量广播到指定的形状。
    
    返回一个原始张量的视图，广播到新的形状。原始张量和新张量共享相同的数据，
    但新张量的形状可能不同。广播遵循NumPy的广播规则。
    
    Args:
        input (TN): 要广播的输入张量
        size (Tuple[int, ...]): 目标形状，必须与输入形状兼容
        
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
    # 验证输入
    if not isinstance(input, TN):
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input)}")
    
    # 确保size是元组
    if not isinstance(size, (tuple,list)):
        raise TypeError(f"Expected size to be a tuple or list of integers, but received type: {type(size)}")
    
    # 将size转换为元组
    size = tuple(size)
    
    if input.shape == size:
        return input

    # 创建全1张量并与输入相乘，利用numpy的广播机制
    ones_tensor = ones(*size, dtype=input.dtype, requires_grad=False)
    result = input * ones_tensor
    
    # 确保结果的形状正确
    if result.shape != size:
        raise RuntimeError(f"Failed to broadcast tensor to shape {size}")
    
    return result

def dot(x:TN,y:TN)->TN:
    """
    计算两个张量的点积。
    
    对于一维数组，计算向量的内积；对于二维数组，计算矩阵乘法；
    对于N维数组，是x和y的最后一个轴上的点积运算。
    
    Args:
        x (TN): 第一个张量
        y (TN): 第二个张量
        
    Returns:
        TN: 两个张量的点积结果
        
    Examples:
        >>> a = tensor([1, 2, 3])
        >>> b = tensor([4, 5, 6])
        >>> dot(a, b)  # 返回32 (1*4 + 2*5 + 3*6)
        
        >>> c = tensor([[1, 2], [3, 4]])
        >>> d = tensor([[5, 6], [7, 8]])
        >>> dot(c, d)  # 返回[[19, 22], [43, 50]]
    """
    return x @ y

def _convert_TNindex_to_numpy(index):
        if isinstance(index,TN):
            index_val = index.data
        elif isinstance(index,tuple):
            index_list = []
            for idx in index:
                if isinstance(idx,TN):
                    index_list.append(idx.data)
                else:
                    index_list.append(idx)
            index_val = tuple(index_list)
        else:
            index_val = index

        return index_val

def _squeeze_backward(result_tensor:TN, i: int) -> TN:
    dim = result_tensor.parms[i]
    grad = result_tensor.grad_value.unsqueeze(dim)
    return grad

def _unsqueeze_backward(result_tensor:TN, i: int) -> TN:
    dim = result_tensor.parms[i]
    grad = result_tensor.grad_value.squeeze(dim)
    return grad

def _reshape_backward(result_tensor: TN, i: int) -> TN:
    original_shape, _ = result_tensor.parms[i]
    grad = result_tensor.grad_value.reshape(original_shape)
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
        grad = summed_grad.reshape(original_shape)
    else:
        # 如果没有维度被扩展，直接使用Riemann的reshape函数
        grad = grad_value.reshape(original_shape)
    
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

def flip(input: TN, dims:List[int]|Tuple[int,...]) -> TN:
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
    data = arrlib.flip(input.data, tuple(dims)).copy()
    
    # 创建新的张量
    result = tensor(data, device = input.device, requires_grad = (is_grad_enabled() and input.requires_grad))
    result.is_leaf = not result.requires_grad
    
    # 如果需要计算梯度，设置计算图信息
    if result.requires_grad:
        result.fromvars = (input,)
        result.parms = (dims,)
        result.gradfuncs = (_flip_backward,)
    
    return result

def _get_broadcast_axis(big_shape,small_shape):
    """获取在广播（broadcasting）操作中需要扩展的轴。

    当两个形状不同的数组进行二元运算时，根据广播规则，较小的数组会被扩展到较大数组的形状。
    此函数计算较小数组需要在哪些轴上进行扩展。

    参数:
        big_shape: 较大数组的形状（广播后的目标形状）
        small_shape: 需要广播扩展的较小数组的形状

    返回:
        包含需要广播扩展的轴索引的元组

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
        ValueError: 当两个数组形状不兼容（无法进行广播）时抛出
    """
    # 右对齐缺失的维度
    len_of_bigshape= len(big_shape)
    dims = len_of_bigshape - len(small_shape)
    axis_list=[i for i in range(dims)]

    for i in range(dims,len_of_bigshape):
        j = i - dims
        # big_shape是计算结果数组，是广播后的shape，所以big_shape[i]>=small_shape[j]
        # 二者不等时，small_shape[j]一定是1，对应的i轴就需要广播
        if big_shape[i] != small_shape[j]:
            if small_shape[j]==1:
                axis_list.append(i)
            else:
                raise ValueError(big_shape,small_shape)

    return tuple(axis_list)

def _sum_backward(result_tensor:TN,i:int)->TN:
    dim, keepdims = result_tensor.parms[i]

    # 如果sum计算时结果张量做过维度精简，需要将缩减的维度暂时恢复
    if keepdims == False:
        new_result_grad = result_tensor.grad_value.unsqueeze(dim)
    else:
        new_result_grad = result_tensor.grad_value
    
    x=result_tensor.fromvars[i]
    # mask = tensor(np.ones_like(x.data,dtype=x.data.dtype))
    mask = ones_like(x)
    grad = new_result_grad * mask
    
    return grad

def sum(x:TN, dim:int|tuple|None=None, keepdim:bool=False)->TN:
    if not isinstance(x,TN):
        raise SyntaxError('x must be a tensor')
    
    # dim==()时，numpy.sum返回原数组，但sum函数求总和，需要将dim设置为None
    if dim == ():
        dim = None
    
    # 根据x的数组类型选择使用np或cp
    arrlib = x._get_array_lib()
    sumvalue = arrlib.sum(x.data, axis=dim, keepdims=keepdim)
    
    # 创建与x在相同设备上的张量
    ret=tensor(sumvalue, device=x.device, requires_grad = (is_grad_enabled() and x.requires_grad))
    ret.is_leaf=not ret.requires_grad

    if ret.requires_grad:
        ret.fromvars=(x,)
        ret.parms=((dim,keepdim),)
        ret.gradfuncs=(_sum_backward,)
    
    return ret

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

def cumsum(input: TN, dim: int, *, dtype: Optional[Union[str, np.dtype]] = None, out: Optional[TN] = None) -> TN:
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
    
    if out is not None:
        if not isinstance(out, TN):
            raise TypeError(f"Expected out type to be TN tensor or None, but received type: {type(out)}")
    
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
        # 如果指定了输出张量，将结果写入其中
        if out is not None:
            out.data = ret.data
            out.is_leaf = True
        return ret

    # 处理1D向量情况，验证dim参数的有效性
    if input.ndim == 1:
        # 1D向量只支持dim=0或dim=-1
        if dim not in [0, -1]:
            raise IndexError(f"Dimension out of range (expected to be in range of [-1, 0], but got {dim})")
    
    # 根据x的数组类型选择使用np或cp
    arrlib = input._get_array_lib()
    # 前向计算 - 使用numpy.cumsum
    result_data = arrlib.cumsum(input.data, axis=dim, dtype=result_dtype)
    
    # 如果指定了输出张量，将结果写入其中
    if out is not None:
        out.data = result_data
        out.is_leaf = True
    
    # 创建结果张量
    ret = tensor(result_data, device=input.device, requires_grad=(is_grad_enabled() and input.requires_grad))
    ret.is_leaf = not ret.requires_grad
    
    # 注册梯度函数
    if ret.requires_grad:
        ret.fromvars = (input,)
        ret.parms = ((dim, result_dtype),)
        ret.gradfuncs = (_cumsum_backward,)
    
    return ret

def _prod_backward(result_tensor:TN, i:int)->TN:
    dim, keepdims = result_tensor.parms[i]

    # 如果prod计算时结果张量做过维度精简，需要将缩减的维度暂时恢复
    if keepdims == False:
        new_result_grad = result_tensor.grad_value.unsqueeze(dim)
    else:
        new_result_grad = result_tensor.grad_value
    
    x = result_tensor.fromvars[i]
    
    # 创建一个与x相同形状的结果张量 - 保持计算图连续性
    if dim is None:
        # 对于全局乘积（标量结果），我们需要将结果广播到x的形状
        expanded_result = ones_like(x) * result_tensor
    else:
        # 对于指定维度的乘积
        if isinstance(dim, int):
            # 单维度情况
            # 当keepdims=False时，我们需要先unsqueeze恢复维度
            if not keepdims:
                # 先添加被移除的维度
                expanded_result = result_tensor.unsqueeze(dim)
                # 使用ones_like和乘法进行广播
                expanded_result = expanded_result * ones_like(x)
            else:
                # keepdims=True的情况，直接广播
                expanded_result = ones_like(x) * result_tensor
        elif isinstance(dim, tuple):
            # 多维度情况
            expanded_result = result_tensor
            # 对于每个缩减的维度，恢复它
            for d in sorted(dim, reverse=True):  # 反向排序以避免维度索引变化问题
                if not keepdims:
                    expanded_result = expanded_result.unsqueeze(d)
            # 使用ones_like和乘法进行广播
            expanded_result = expanded_result * ones_like(x)
        else:
            raise TypeError(f"dim must be int, tuple or None, got {type(dim)}")
    
    # 关键改进：使用Tensor方法而不是直接操作numpy数组，保持计算图连续性
    # 同时确保数值稳定性
    
    # 创建一个安全的掩码，使用Tensor操作
    # 对于非零元素，使用常规梯度；对于零元素，特殊处理
    # 使用Tensor的where方法避免除零操作
    
    # 更精确地检测零值
    # 对于复数和实数，都使用绝对值进行零检测
    abs_x = abs(x)
    non_zero_x = where(abs_x > 1e-10, ones_like(x), zeros_like(x))
    is_zero = where(abs_x <= 1e-10, ones_like(x), zeros_like(x)).type(bool)
    
    # 创建安全除数：仅当x不为零时使用x，否则使用1（但会被where过滤掉）
    safe_divisor = where(non_zero_x, x, ones_like(x))
    
    # 常规梯度计算，但使用安全除数
    regular_grad = where(non_zero_x, expanded_result / safe_divisor, zeros_like(x))
    
    # 特殊处理：当x为0时，梯度计算逻辑是
    # 如果沿着指定维度，该元素所在切片中只有它一个为0，则梯度为其他元素的乘积
    # 否则，梯度为0
    
    # 对于全局乘积的特殊处理（dim=None）
    if dim is None:
        # 计算张量中零元素的总数
        zero_count = sum(is_zero.type(int))
        # 如果只有一个零元素，则对应该位置的梯度为其他所有元素的乘积
        if zero_count == 1:
            # 创建一个临时张量，将零元素位置设为1，其他位置保持不变
            temp_x = where(is_zero, ones_like(x), x)
            # 计算所有元素的乘积
            other_prod = temp_x.prod()
            # 将结果广播到x的形状
            other_prod_expanded = ones_like(x) * other_prod
            # 只在零元素位置设置特殊梯度
            regular_grad = where(is_zero, other_prod_expanded, regular_grad)
    elif isinstance(dim, int) or (isinstance(dim, tuple) and len(dim) > 0):
        # 对于指定维度的情况
        # 计算每个维度切片中有多少个0元素
        if isinstance(dim, int):
            # 单维度情况
            # 创建维度掩码，排除当前维度
            other_dims = tuple(i for i in range(x.ndim) if i != dim)
            # 计算每个切片中的0元素数量
            zero_counts = sum(is_zero.type(int), dim=other_dims)
            # 确保zero_counts被正确地广播到x的形状
            expanded_zero_counts = zero_counts
            for i in other_dims:
                expanded_zero_counts = expanded_zero_counts.unsqueeze(i)
            # 当且仅当元素为0且切片中只有它一个0时，梯度为其他元素的乘积
            single_zero_mask = is_zero & (expanded_zero_counts == 1)
            
            # 对于满足条件的位置，计算其他元素的乘积
            # 创建一个临时张量，将当前位置的元素设为1，其他位置保持不变
            temp_x = where(single_zero_mask, ones_like(x), x)
            # 计算沿着指定维度的乘积
            other_prod = temp_x.prod(dim=dim, keepdim=True)
            # 扩展结果到x的形状
            other_prod_expanded = other_prod * ones_like(x)
            
            # 根据条件选择常规梯度或特殊处理的梯度
            regular_grad = where(single_zero_mask, other_prod_expanded, regular_grad)
        elif isinstance(dim, tuple):
            # 多维度情况，采用类似的逻辑
            other_dims = tuple(i for i in range(x.ndim) if i not in dim)
            if len(other_dims) > 0:
                # 计算每个切片中的0元素数量
                zero_counts = sum(is_zero.type(int), dim=other_dims)
                # 确保zero_counts被正确地广播到x的形状
                expanded_zero_counts = zero_counts
                for i in other_dims:
                    expanded_zero_counts = expanded_zero_counts.unsqueeze(i)
                # 当且仅当元素为0且切片中只有它一个0时，梯度为其他元素的乘积
                single_zero_mask = is_zero & (expanded_zero_counts == 1)
                
                # 对于满足条件的位置，计算其他元素的乘积
                temp_x = where(single_zero_mask, ones_like(x), x)
                # 计算沿着指定维度的乘积
                other_prod = temp_x.prod(dim=dim, keepdim=True)
                # 扩展结果到x的形状
                for d in sorted(dim, reverse=True):
                    if not keepdims:
                        other_prod = other_prod.unsqueeze(d)
                other_prod_expanded = other_prod * ones_like(x)
                
                # 根据条件选择常规梯度或特殊处理的梯度
                regular_grad = where(single_zero_mask, other_prod_expanded, regular_grad)
    
    # 计算最终梯度
    # 对regular_grad取共轭以支持复数张量的Wirtinger梯度
    grad = new_result_grad * regular_grad.conj()
    
    return grad

def prod(x:TN, dim:int|tuple|None=None, keepdim:bool=False)->TN:
    """
    返回张量在指定维度上的乘积。
    
    参数:
        x: 输入张量
        dim: 计算乘积的维度，可以是整数、元组或None（对所有元素求乘积）
        keepdim: 是否保持维度不变
    
    返回:
        计算乘积后的张量
    """
    if not isinstance(x,TN):
        raise SyntaxError('x must be a tensor')
    
    if dim == ():
        dim = None
    
    # 根据x的数组类型选择使用np或cp
    arrlib = x._get_array_lib()
    # 使用numpy/cupy的prod函数计算乘积
    prod_value = arrlib.prod(x.data, axis=dim, keepdims=keepdim)
    ret = tensor(prod_value, device=x.device, requires_grad = (is_grad_enabled() and x.requires_grad))
    ret.is_leaf = not ret.requires_grad

    if ret.requires_grad:
        ret.fromvars = (x,)
        ret.parms = ((dim, keepdim),)
        ret.gradfuncs = (_prod_backward,)
    
    return ret

def _mean_backward(result_tensor:TN, i:int)->TN:
    dim, keepdim = result_tensor.parms[i]     
    x = result_tensor.fromvars[i]

    if type(dim) == int:
        n = x.data.shape[dim]
    elif dim is None:
        n = 1
        for i in x.data.shape:
            n *= i
    elif isinstance(dim,tuple):
        n=1.
        for i in dim:
            n *= x.data.shape[i]
    else:
        raise TypeError(dim)

    arr = ones_like(x) / tensor(n,dtype=x.dtype,device=x.device)
    
    # 如果mean计算时结果张量做过维度精简，需要将缩减的维度暂时恢复
    if keepdim == False:
        new_result_grad = result_tensor.grad_value.unsqueeze(dim)
    else:
        new_result_grad = result_tensor.grad_value

    grad = new_result_grad * arr
    return grad

def mean(x:TN, dim:int|tuple|None=None, keepdim:bool=False)->TN:
    if not isinstance(x,TN):
        raise SyntaxError('x must be a tensor')
    
    if dim == ():
        dim = None
    
    # 根据x的数组类型选择使用np或cp
    arrlib = x._get_array_lib()
    value = arrlib.mean(x.data, axis=dim, keepdims=keepdim)
    ret = tensor(value, device=x.device, requires_grad = (is_grad_enabled() and x.requires_grad))
    ret.is_leaf=not ret.requires_grad

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
    arrlib = x._get_array_lib()
    value = arrlib.abs(x.data)
    ret = tensor(value, device=x.device, requires_grad = (is_grad_enabled() and x.requires_grad))
    ret.is_leaf = not ret.requires_grad

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
    return x ** 0.5

def _create_maxmin_mask(arr, argmaxmin, dim:int|tuple|None=None):
    """创建最大值/最小值掩码数组，支持梯度分配到极值位置

    该函数用于在自动微分过程中，为max/min函数的反向传播创建掩码数组。
    当dim=None时，梯度会被平均分配到所有相同的极值位置；
    当指定dim时，梯度仅分配到每个切片中的第一个极值位置。

    参数:
        arr (numpy.ndarray): 输入的numpy数组
        argmaxmin (function): 最大值/最小值索引函数(numpy.argmax或numpy.argmin)
        dim (int|tuple|None, 可选): 计算最大值/最小值的轴，默认为None（全局计算）

    返回:
        numpy.ndarray: 掩码数组，形状与输入数组相同，在极值位置为1.0，其他位置为0.0

    异常:
        ValueError: 当dim参数不是None、整数或元组时抛出

    示例:
        >>> arr = np.array([[1., 6., 3.], [4., 6., 5.]])
        >>> _create_maxmin_mask(arr, np.argmax)
        array([[0. , 0.5, 0. ],
               [0. , 0.5, 0. ]])
        
        >>> _create_maxmin_mask(arr, np.argmax, dim=1)
        array([[0., 1., 0.],
               [0., 1., 0.]])
    """
    arrlib = np if isinstance(arr,np.ndarray) else cp
    # 初始化全0掩码
    mask_arr = arrlib.zeros_like(arr, dtype=arr.dtype)
    
    if dim is None:
        # 全局最大值/最小值 - 保持平均分配行为
        maxmin_val = arrlib.max(arr) if argmaxmin == arrlib.argmax else np.min(arr)
        # 创建掩码，标记所有等于最大值/最小值的位置
        equal_mask = (arr == maxmin_val)
        # 归一化掩码，确保总和为1
        if equal_mask.sum() > 0:
            mask_arr[equal_mask] = 1.0 / equal_mask.sum()
    elif isinstance(dim, int):
        # 单轴最大值/最小值 - 只选择第一个最大值位置
        indices = argmaxmin(arr, axis=dim)
        # 为了使用put_along_axis，需要扩展维度
        indices_expanded = arrlib.expand_dims(indices, axis=dim)
        
        # 使用numpy和cupy都支持的向量化操作替代put_along_axis
        # 创建一个与mask_arr形状相同的索引数组，其中dim轴的索引来自indices_expanded
        shape = mask_arr.shape
        
        # 创建一个包含所有轴索引的元组
        grid = arrlib.indices(shape)
        
        # 将dim轴的索引替换为indices_expanded
        grid[dim] = indices_expanded
        
        # 使用高级索引设置值为1.0
        mask_arr[tuple(grid)] = 1.0
    elif isinstance(dim, tuple):
        # 多轴最大值/最小值 - 只选择第一个最大值位置
        # 将指定轴移动到前面
        transposed = arrlib.moveaxis(arr, dim, range(len(dim)))
        # 合并轴并展平
        merged_shape = (-1,) + transposed.shape[len(dim):]
        flattened = transposed.reshape(merged_shape)
        # 沿合并后的轴取argmax或argmin
        maxmin_indices = argmaxmin(flattened, axis=0)
        # 分解索引为原轴坐标
        original_dims = [arr.shape[ax] for ax in dim]
        multi_indices = arrlib.unravel_index(maxmin_indices, original_dims)
        # 生成其他轴索引
        other_indices = list(arrlib.ogrid[tuple(slice(s) for s in maxmin_indices.shape)])
        # 组合所有索引并转置回原轴顺序
        full_indices = []
        idx_iter = iter(multi_indices)
        for i in range(arr.ndim):
            if i in dim:
                full_indices.append(next(idx_iter))
            else:
                full_indices.append(other_indices.pop(0))
        # 设置对应位置为1.0
        mask_arr[tuple(full_indices)] = 1.0
    else:
        raise ValueError("`axis` must be None, int, or tuple")
    
    return mask_arr


# 为max和min函数创建专门的返回类型类
class MaxMinReturnType:
    def __init__(self, values, indices, name):
        self.values = values
        self.indices = indices
        self._name = name  # 存储操作名称用于调试
    
    def __iter__(self):
        # 支持解包操作
        return iter([self.values, self.indices])
    
    def __repr__(self):
        return f"{self._name}(values={self.values}, \nindices={self.indices})"

# 修改现有的_max_backward函数保持不变
def _max_backward(result_tensor:TN, i:int)->TN:
    x = result_tensor.fromvars[i]    
    dim, keepdim = result_tensor.parms[i]
    
    arrlib = x._get_array_lib()
    max_pos_one_like_x = _create_maxmin_mask(x.data, arrlib.argmax, dim)
    max_pos_one_tensor = tensor(max_pos_one_like_x,device=x.device)
    
    # 如果sum计算时结果张量做过维度精简，需要将缩减的维度暂时恢复
    if keepdim == False:
        new_result_grad = result_tensor.grad_value.unsqueeze(dim)
    else:
        new_result_grad = result_tensor.grad_value
    
    grad = new_result_grad * max_pos_one_tensor
    return grad

# 修改现有的_min_backward函数保持不变
def _min_backward(result_tensor:TN, i:int)->TN:
    x = result_tensor.fromvars[i]    
    dim, keepdim = result_tensor.parms[i]
    
    arrlib = x._get_array_lib()
    min_pos_one_like_x = _create_maxmin_mask(x.data, arrlib.argmin, dim)
    min_pos_one_tensor = tensor(min_pos_one_like_x, device=x.device)
    
    # 如果sum计算时结果张量做过维度精简，需要将缩减的维度暂时恢复
    if keepdim == False:
        new_result_grad = result_tensor.grad_value.unsqueeze(dim)
    else:
        new_result_grad = result_tensor.grad_value
    
    grad = new_result_grad * min_pos_one_tensor    
    return grad

def max(x:TN, dim:int|None=None, keepdim:bool=False, *, out=None):
    if x.is_complex():
        raise RuntimeError("max() does not support complex input")
    
    arrlib = x._get_array_lib()
    dev = x.device
    # 计算最大值 - 利用numpy原生的axis=None行为
    values_arr = arrlib.max(x.data, axis=dim, keepdims=keepdim)
    values_tensor = tensor(values_arr, device=dev, requires_grad=(is_grad_enabled() and x.requires_grad))
    values_tensor.is_leaf = not values_tensor.requires_grad
    
    # 为values_tensor设置梯度信息
    if values_tensor.requires_grad:
        values_tensor.fromvars = (x,)
        values_tensor.parms = ((dim, keepdim),)
        values_tensor.gradfuncs = (_max_backward,)
    
    # 根据dim是否为None决定返回类型
    if dim is None:
        # 当dim为None时，直接返回值张量
        return values_tensor
    else:
        # 当dim不为None时，计算索引并返回包含values和indices的对象
        indices_arr = arrlib.argmax(x.data, axis=dim)
        if keepdim:
            indices_arr = arrlib.expand_dims(indices_arr, axis=dim)
        indices_tensor = tensor(indices_arr,device=dev)
        
        return MaxMinReturnType(values_tensor, indices_tensor, "max")

def min(x:TN, dim:int|None=None, keepdim:bool=False, *, out=None):
    if x.is_complex():
        raise RuntimeError("min() does not support complex input")

    arrlib = x._get_array_lib()
    dev = x.device
    # 计算最小值 - 利用numpy原生的axis=None行为
    values_arr = arrlib.min(x.data, axis=dim, keepdims=keepdim)
    values_tensor = tensor(values_arr, device=dev, requires_grad=(is_grad_enabled() and x.requires_grad))
    values_tensor.is_leaf = not values_tensor.requires_grad
    
    # 为values_tensor设置梯度信息
    if values_tensor.requires_grad:
        values_tensor.fromvars = (x,)
        values_tensor.parms = ((dim, keepdim),)
        values_tensor.gradfuncs = (_min_backward,)
    
    # 根据dim是否为None决定返回类型
    if dim is None:
        # 当dim为None时，直接返回值张量
        return values_tensor
    else:
        # 当dim不为None时，计算索引并返回包含values和indices的对象
        indices_arr = arrlib.argmin(x.data, axis=dim)
        if keepdim:
            indices_arr = arrlib.expand_dims(indices_arr, axis=dim)
        indices_tensor = tensor(indices_arr,device=dev)
        
        return MaxMinReturnType(values_tensor, indices_tensor, "min")

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
    factor = tensor(2.0 / denom,dtype = x.dtype, device=x.device)
    
    # 准备梯度值
    grad_value = result_tensor.grad_value
    
    # 如果keepdim为False，需要扩展梯度维度以匹配diff的维度
    if not keepdim:
        if dim is None:
            # 对于全局方差，需要扩展到原始形状
            new_result_grad = grad_value.reshape(tuple(1 for _ in x.shape))
        elif isinstance(dim, int):
            new_result_grad = grad_value.unsqueeze(dim)
        else:  # tuple
            # 对多个dim，按升序逐个unsqueeze以保持维度顺序正确
            # 注意：这里需要按升序处理，因为每次unsqueeze会增加维度，影响后续的索引
            new_result_grad = grad_value
            for d in sorted(dim):
                new_result_grad = new_result_grad.unsqueeze(d)
    else:
        new_result_grad = grad_value
    
    # 计算最终梯度
    grad = new_result_grad * diff * factor
    return grad

def var(x:TN, dim:int|tuple|None=None, unbiased:bool=True, keepdim:bool=False)->TN:
    if not isinstance(x,TN):
        raise SyntaxError('x must be a tensor')
    
    if dim == ():
        dim = None
    ddof = 1 if unbiased == True else 0

    arrlib = x._get_array_lib()
    value = arrlib.var(x.data, axis=dim, ddof=ddof, keepdims=keepdim)
    ret=tensor(value, device=x.device, requires_grad = (is_grad_enabled() and x.requires_grad))
    ret.is_leaf = not ret.requires_grad

    if ret.requires_grad:
        ret.fromvars=(x,)
        ret.parms=((dim,unbiased,keepdim),)
        ret.gradfuncs=(_var_backward,)
    
    return ret

# 利用mean函数计算var的版本，无需构造梯度回调函数
def var2(x:TN, dim:int|tuple|None=None, unbiased:bool=True, keepdim:bool=False)->TN:
    if not isinstance(x,TN):
        raise SyntaxError('x must be a tensor')
    
    if dim == ():
        dim = None
    
    # 基于mean函数实现方差计算
    # 1. 计算均值
    x_mean = mean(x, dim=dim, keepdim=True)
    
    # 2. 计算平方差（使用模的平方，对于复数需要考虑共轭）
    diff = x - x_mean
    # 对于复数，使用diff * diff.conj()计算模的平方
    square_diff = diff * diff.conj()
    
    # 3. 计算平方差的均值
    variance = mean(square_diff, dim=dim, keepdim=keepdim)
    
    # 4. 根据unbiased参数调整自由度
    if unbiased:
        # 计算用于自由度调整的缩放因子
        if dim is None:
            n = x.numel()
        elif isinstance(dim, int):
            n = x.shape[dim]
        elif isinstance(dim, tuple):
            n = 1
            for d in dim:
                n *= x.shape[d]
        else:
            raise TypeError(f"dim must be int, tuple or None, got {type(dim)}")
        
        if n > 1:  # 避免除以0
            scaling_factor = tensor(n / (n - 1),dtype=x.dtype, device=x.device)
            variance = variance * scaling_factor
    
    return variance.real

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
    inv_permutation = tuple(inv_permutation)
    
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
        grad = sum(result_tensor.grad_value,dim=broadcast_axes,keepdim=False).reshape(left_var_shape)
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
        grad = sum(result_tensor.grad_value,dim=broadcast_axes,keepdim=False).reshape(right_var_shape)
    return grad

def _sub_grad_left(result_tensor:TN, i:int)->TN:
    return _add_grad_left(result_tensor,i)

def _sub_grad_right(result_tensor:TN, i:int)->TN:
    grad = _add_grad_right(result_tensor,i)
    return -grad

def _mul_grad_left(result_tensor:TN, i:int)->TN:
    left_tensor = result_tensor.fromvars[i]
    right_tensor = result_tensor.fromvars[i+1]
    left_var_shape = left_tensor.shape
    result_shape = result_tensor.shape

    left_grad = result_tensor.grad_value * right_tensor.conj()

    # shape一样时，直接返回left_grad，否则对left_grad进行sum缩减
    if left_var_shape == result_shape:
        grad = left_grad
    else:
        # left_tensor与result_tensor的shape比较，获取需left_tensor广播轴序号的元组
        broadcast_axes=_get_broadcast_axis(result_shape,left_var_shape)
        grad = sum(left_grad,
                    dim=broadcast_axes,
                    keepdim=False).reshape(left_var_shape)
    return grad

def _mul_grad_right(result_tensor:TN, i:int)->TN:
    left_tensor = result_tensor.fromvars[i-1]
    right_tensor = result_tensor.fromvars[i]
    right_var_shape = right_tensor.shape
    result_shape = result_tensor.shape

    right_grad = result_tensor.grad_value * left_tensor.conj()

    # shape一样时，直接返回right_grad，否则对right_grad进行sum缩减
    if right_var_shape == result_shape:
        grad = right_grad
    else:
        # right_tensor与result_tensor的shape比较，获取需right_tensor广播轴序号的元组
        broadcast_axes=_get_broadcast_axis(result_shape,right_var_shape)
        grad = sum(right_grad,
                    dim=broadcast_axes,
                    keepdim=False).reshape(right_var_shape)
    return grad

def _matmul_grad_left(result_tensor:TN, i:int)->TN:
    left_tensor:TN = result_tensor.fromvars[i]
    right_tensor:TN = result_tensor.fromvars[i+1]
    grad_value = result_tensor.grad_value
    
    left_ndim = left_tensor.data.ndim
    right_ndim = right_tensor.data.ndim
    
    # 1. 一维向量与一维向量的情况
    if left_ndim == 1 and right_ndim == 1:
        return grad_value * right_tensor.conj()
    
    # 2. 一维向量与多维矩阵的情况
    if left_ndim == 1 and right_ndim > 1:
        # 行向量乘矩阵结果还是行向量，result_tensor.grad扩充中行向量扩维为(1,n)
        new_grad = grad_value.unsqueeze(-2)  # 扩维为二维
        mat_grad = new_grad @ right_tensor.mT.conj()
        
        # 计算需要求和的轴
        sum_axes = tuple(range(right_ndim - 1))
        return sum(mat_grad, dim=sum_axes, keepdim=False)
    
    # 3. 多维矩阵与一维向量的情况
    if left_ndim > 1 and right_ndim == 1:
        # 结果如果是行向量，先列化，相当于转置，计算完成后恢复为行向量
        new_grad = grad_value.unsqueeze(-1)  # 扩维为二维
        new_right = right_tensor.unsqueeze(0).conj()  # 转换为(1,n)矩阵
        
        return new_grad @ new_right
    
    # 4. 多维矩阵与多维矩阵的一般情况
    # 比较left_tensor与result_tensor的shape中的广播维，获取需left_tensor广播轴序号的元组
    broadcast_axes = _get_broadcast_axis(result_tensor.data.shape[:-2], left_tensor.data.shape[:-2])
    mat_grad = grad_value @ right_tensor.mT.conj()
    
    # 仅在需要广播时进行求和操作
    if broadcast_axes:
        return sum(mat_grad, dim=broadcast_axes, keepdim=False)
    
    return mat_grad

# 优化的矩阵乘法右梯度函数
def _matmul_grad_right(result_tensor:TN, i:int)->TN:
    left_tensor:TN = result_tensor.fromvars[i-1]
    right_tensor:TN = result_tensor.fromvars[i]
    grad_value = result_tensor.grad_value
    
    left_ndim = left_tensor.data.ndim
    right_ndim = right_tensor.data.ndim
    
    # 1. 一维向量与一维向量的情况
    if left_ndim == 1 and right_ndim == 1:
        return left_tensor.conj() * grad_value
    
    # 2. 一维向量与多维矩阵的情况
    if left_ndim == 1 and right_ndim > 1:
        # 左行向量先列化，计算完成后恢复为行向量
        new_left = left_tensor.unsqueeze(1).conj()
        new_grad = grad_value.unsqueeze(-2)  # 插入倒数第二维
        
        return new_left @ new_grad
    
    # 3. 多维矩阵与一维向量的情况
    if left_ndim > 1 and right_ndim == 1:
        new_grad = grad_value.unsqueeze(-1)  # 扩维为二维
        mat_grad = left_tensor.mT.conj() @ new_grad
        
        # 计算需要求和的轴
        sum_axes = tuple(range(left_ndim - 2))
        if sum_axes:  # 避免对空元组求和导致标量化
            mat_grad = sum(mat_grad, dim=sum_axes, keepdim=False)
        
        # 如果结果是多维的，删除最后一个维度
        if mat_grad.data.ndim > 1:
            return mat_grad.squeeze(-1)
        
        return mat_grad
    
    # 4. 多维矩阵与多维矩阵的一般情况
    broadcast_axes = _get_broadcast_axis(result_tensor.data.shape[:-2], right_tensor.data.shape[:-2])
    mat_grad = left_tensor.mT.conj() @ grad_value
    
    # 仅在需要广播时进行求和操作
    if broadcast_axes:
        return sum(mat_grad, dim=broadcast_axes, keepdim=False)
    
    return mat_grad

def _div_grad_left(result_tensor:TN, i:int)->TN:
    left_tensor = result_tensor.fromvars[i]
    right_tensor = result_tensor.fromvars[i+1]
    left_var_shape = left_tensor.shape
    result_shape = result_tensor.shape

    left_grad = result_tensor.grad_value / right_tensor.conj()

    # shape一样时，直接返回left_grad，否则对left_grad进行sum缩减
    if left_var_shape == result_shape:
        grad = left_grad
    else:
        # left_tensor与result_tensor的shape比较，获取需left_tensor广播轴序号的元组
        broadcast_axes=_get_broadcast_axis(result_shape,left_var_shape)
        grad = sum(left_grad,
                    dim=broadcast_axes,
                    keepdim=False).reshape(left_var_shape)
    return grad
 
def _div_grad_right(result_tensor:TN, i:int)->TN:
    left_tensor = result_tensor.fromvars[i-1]
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
                    keepdim=False).reshape(right_var_shape)
    return grad

def _pow_grad_left(result_tensor:TN, i:int)->TN:
    left_tensor = result_tensor.fromvars[i]
    right_tensor = result_tensor.fromvars[i+1]
    left_var_shape = left_tensor.shape
    result_shape = result_tensor.shape

    left_grad = result_tensor.grad_value * (right_tensor*(left_tensor ** (right_tensor - 1.))).conj()

    # shape一样时，直接返回left_grad，否则对left_grad进行sum缩减
    if left_var_shape == result_shape:
        grad = left_grad
    else:
        # left_tensor与result_tensor的shape比较，获取需left_tensor广播轴序号的元组
        broadcast_axes=_get_broadcast_axis(result_shape,left_var_shape)
        grad = sum(left_grad,
                    dim=broadcast_axes,
                    keepdim=False).reshape(left_var_shape)
    return grad

def _pow_grad_right(result_tensor:TN, i:int)->TN:
    left_tensor = result_tensor.fromvars[i-1]
    right_tensor = result_tensor.fromvars[i]
    right_var_shape = right_tensor.shape
    result_shape = result_tensor.shape

    right_grad = result_tensor.grad_value * result_tensor.conj() * log(left_tensor.conj())
    
    # shape一样时，直接返回right_grad，否则对right_grad进行sum缩减
    if right_var_shape == result_shape:
        grad = right_grad
    else:
        # right_tensor与result_tensor的shape比较，获取需right_tensor广播轴序号的元组
        broadcast_axes=_get_broadcast_axis(result_shape,right_var_shape)
        grad = sum(right_grad,
                    dim=broadcast_axes,
                    keepdim=False).reshape(right_var_shape)
    return grad

def pow(input, exponent)->TN|float:
    return input ** exponent

def _log_derivative(x:TN)->tuple[TN]:
    return (1. / x.conj(),)

@track_grad(_log_derivative)
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
    arrlib = x._get_array_lib()
    return tensor(arrlib.log(x.data), device=x.device)

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

def _exp_derivative(x:TN)->tuple[TN]:
    return (exp(x).conj(),)

@track_grad(_exp_derivative)
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
    arrlib = x._get_array_lib()
    return tensor(arrlib.exp(x.data),device=x.device)

def _sin_derivative(x:TN)->tuple[TN]:
    return (cos(x).conj(),)

@track_grad(_sin_derivative)
def sin(x:TN)->TN:
    """
    计算张量的正弦函数。
    
    返回一个新张量，其中每个元素是输入张量对应元素的正弦值(弧度制)。
    
    Args:
        x (TN): 输入张量，元素为弧度值
        
    Returns:
        TN: 包含输入张量正弦值的新张量
        
    Examples:
        >>> a = tensor([0, pi/2, pi])  # 0, 90度, 180度
        >>> sin(a)  # 返回[0, 1, 0]
        
        >>> b = tensor([[0.0, pi/4], [pi/2, 3*pi/4]])
        >>> sin(b)  # 返回[[0.0, 0.707], [1.0, 0.707]]
    """
    arrlib = x._get_array_lib()
    return tensor(arrlib.sin(x.data),device=x.device)
    
def _cos_derivative(x:TN)->tuple[TN]:
    return (-sin(x).conj(),)

@track_grad(_cos_derivative)
def cos(x:TN)->TN:
    """
    计算张量的余弦函数。
    
    返回一个新张量，其中每个元素是输入张量对应元素的余弦值(弧度制)。
    
    Args:
        x (TN): 输入张量，元素为弧度值
        
    Returns:
        TN: 包含输入张量余弦值的新张量
        
    Examples:
        >>> a = tensor([0, pi/2, pi])  # 0, 90度, 180度
        >>> cos(a)  # 返回[1, 0, -1]
        
        >>> b = tensor([[0.0, pi/4], [pi/2, 3*pi/4]])
        >>> cos(b)  # 返回[[1.0, 0.707], [0.0, -0.707]]
    """
    arrlib = x._get_array_lib()
    return tensor(arrlib.cos(x.data),device=x.device)

def _tan_derivative(x:TN)->tuple[TN]:
    return (1. + (tan(x.conj()))**2.,)

@track_grad(_tan_derivative)
def tan(x:TN)->TN:
    """
    计算张量的正切函数。
    
    返回一个新张量，其中每个元素是输入张量对应元素的正切值(弧度制)。
    
    Args:
        x (TN): 输入张量，元素为弧度值
        
    Returns:
        TN: 包含输入张量正切值的新张量
        
    Examples:
        >>> a = tensor([0, pi/4])  # 0, 45度
        >>> tan(a)  # 返回[0, 1]
        
        >>> b = tensor([[0.0, pi/6], [pi/3, pi/4]])
        >>> tan(b)  # 返回[[0.0, 0.577], [1.732, 1.0]]
    """
    arrlib = x._get_array_lib()
    return tensor(arrlib.tan(x.data),device=x.device)

def cot(x:TN)->TN:
    """
    计算张量的余切函数。
    
    返回一个新张量，其中每个元素是输入张量对应元素的余切值(弧度制)，即cot(x) = 1/tan(x)。
    
    Args:
        x (TN): 输入张量，元素为弧度值
        
    Returns:
        TN: 包含输入张量余切值的新张量
        
    Examples:
        >>> a = tensor([pi/4, pi/6])  # 45度, 30度
        >>> cot(a)  # 返回[1, 1.732]
        
        >>> b = tensor([[pi/4, pi/3], [pi/6, pi/2]])
        >>> cot(b)  # 返回[[1.0, 0.577], [1.732, 0.0]]
    """
    return 1./tan(x)

def sec(x:TN)->TN:
    return 1./cos(x)

def csc(x:TN)->TN:
    return 1./sin(x)

def _arcsin_derivative(x:TN)->tuple[TN]:
    return (1./sqrt(1. - x**2.).conj(),)

@track_grad(_arcsin_derivative)
def arcsin(x:TN)->TN:
    arrlib = x._get_array_lib()
    return tensor(arrlib.arcsin(x.data), device=x.device)

def _arccos_derivative(x:TN)->tuple[TN]:
    return (-1.0/sqrt(1. - x**2.).conj(),)

@track_grad(_arccos_derivative)
def arccos(x:TN)->TN:
    arrlib = x._get_array_lib()
    return tensor(arrlib.arccos(x.data),device=x.device)

def _arctan_derivative(x:TN)->tuple[TN]:
    return (1./(1. + x**2.).conj(),)

@track_grad(_arctan_derivative)
def arctan(x:TN)->TN:
    arrlib = x._get_array_lib()
    return tensor(arrlib.arctan(x.data),device=x.device)

def _sinh_derivative(x:TN)->tuple[TN]:
    return (cosh(x).conj(),)

@track_grad(_sinh_derivative)
def sinh(x:TN)->TN:
    """
    计算张量的双曲正弦函数。
    
    返回一个新张量，其中每个元素是输入张量对应元素的双曲正弦值，即sinh(x) = (e^x - e^(-x))/2。
    
    Args:
        x (TN): 输入张量
        
    Returns:
        TN: 包含输入张量双曲正弦值的新张量
        
    Examples:
        >>> a = tensor([0, 1, -1])
        >>> sinh(a)  # 返回[0, 1.175, -1.175]
        
        >>> b = tensor([[0.0, 1.0], [2.0, -1.0]])
        >>> sinh(b)  # 返回[[0.0, 1.175], [3.627, -1.175]]
    """
    arrlib = x._get_array_lib()
    return tensor(arrlib.sinh(x.data),device=x.device)

def _cosh_derivative(x:TN)->tuple[TN]:
    return (sinh(x).conj(),)

@track_grad(_cosh_derivative)
def cosh(x:TN)->TN:
    """
    计算张量的双曲余弦函数。
    
    返回一个新张量，其中每个元素是输入张量对应元素的双曲余弦值，即cosh(x) = (e^x + e^(-x))/2。
    
    Args:
        x (TN): 输入张量
        
    Returns:
        TN: 包含输入张量双曲余弦值的新张量
        
    Examples:
        >>> a = tensor([0, 1, -1])
        >>> cosh(a)  # 返回[1, 1.543, 1.543]
        
        >>> b = tensor([[0.0, 1.0], [2.0, -1.0]])
        >>> cosh(b)  # 返回[[1.0, 1.543], [3.762, 1.543]]
    """
    arrlib = x._get_array_lib()
    return tensor(arrlib.cosh(x.data),device=x.device)

def _tanh_derivative(x:TN)->tuple[TN]:
    return ((1.0 - tanh(x)**2.0).conj(),)

@track_grad(_tanh_derivative)
def tanh(x:TN)->TN:
    """
    计算张量的双曲正切函数。
    
    返回一个新张量，其中每个元素是输入张量对应元素的双曲正切值，即tanh(x) = sinh(x)/cosh(x)。
    
    Args:
        x (TN): 输入张量
        
    Returns:
        TN: 包含输入张量双曲正切值的新张量
        
    Examples:
        >>> a = tensor([0, 1, -1])
        >>> tanh(a)  # 返回[0, 0.762, -0.762]
        
        >>> b = tensor([[0.0, 1.0], [2.0, -1.0]])
        >>> tanh(b)  # 返回[[0.0, 0.762], [0.964, -0.762]]
    """
    arrlib = x._get_array_lib()
    return tensor(arrlib.tanh(x.data),device=x.device)

def coth(x:TN)->TN:
    """
    计算张量的双曲余切函数。
    
    返回一个新张量，其中每个元素是输入张量对应元素的双曲余切值，即coth(x) = 1/tanh(x)。
    
    Args:
        x (TN): 输入张量，元素不能为0
        
    Returns:
        TN: 包含输入张量双曲余切值的新张量
        
    Examples:
        >>> a = tensor([1, 2])
        >>> coth(a)  # 返回[1.313, 1.037]
        
        >>> b = tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> coth(b)  # 返回[[1.313, 1.037], [1.005, 1.001]]
    """
    return 1.0 / tanh(x)

def sech(x:TN)->TN:
    """
    计算张量的双曲正割函数。
    
    返回一个新张量，其中每个元素是输入张量对应元素的双曲正割值，即sech(x) = 1/cosh(x)。
    
    Args:
        x (TN): 输入张量
        
    Returns:
        TN: 包含输入张量双曲正割值的新张量
        
    Examples:
        >>> a = tensor([0, 1, -1])
        >>> sech(a)  # 返回[1, 0.648, 0.648]
        
        >>> b = tensor([[0.0, 1.0], [2.0, -1.0]])
        >>> sech(b)  # 返回[[1.0, 0.648], [0.266, 0.648]]
    """
    return 1.0 / cosh(x)

def csch(x:TN)->TN:
    """
    计算张量的双曲余割函数。
    
    返回一个新张量，其中每个元素是输入张量对应元素的双曲余割值，即csch(x) = 1/sinh(x)。
    
    Args:
        x (TN): 输入张量，元素不能为0
        
    Returns:
        TN: 包含输入张量双曲余割值的新张量
        
    Examples:
        >>> a = tensor([1, 2])
        >>> csch(a)  # 返回[0.851, 0.276]
        
        >>> b = tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> csch(b)  # 返回[[0.851, 0.276], [0.100, 0.037]]
    """
    return 1.0 / sinh(x)

def _arcsinh_derivative(x:TN)->tuple[TN]:
    return (1. / sqrt(x**2. + 1.).conj(),)
    
@track_grad(_arcsinh_derivative)
def arcsinh(x:TN)->TN:
    """
    计算张量的反双曲正弦函数。
    
    返回一个新张量，其中每个元素是输入张量对应元素的反双曲正弦值，即arcsinh(x) = ln(x + sqrt(x^2 + 1))。
    
    Args:
        x (TN): 输入张量
        
    Returns:
        TN: 包含输入张量反双曲正弦值的新张量
        
    Examples:
        >>> a = tensor([0, 1, -1])
        >>> arcsinh(a)  # 返回[0, 0.881, -0.881]
        
        >>> b = tensor([[0.0, 1.0], [2.0, -1.0]])
        >>> arcsinh(b)  # 返回[[0.0, 0.881], [1.444, -0.881]]
    """
    arrlib = x._get_array_lib()
    return tensor(arrlib.arcsinh(x.data),device=x.device)

def _arccosh_derivative(x:TN)->tuple[TN]:
    return (1. / sqrt(x**2. - 1.).conj(),)

@track_grad(_arccosh_derivative)
def arccosh(x:TN)->TN:
    """
    计算张量的反双曲余弦函数。
    
    返回一个新张量，其中每个元素是输入张量对应元素的反双曲余弦值，即arccosh(x) = ln(x + sqrt(x^2 - 1))。
    
    Args:
        x (TN): 输入张量，元素必须大于等于1
        
    Returns:
        TN: 包含输入张量反双曲余弦值的新张量
        
    Examples:
        >>> a = tensor([1, 2, 3])
        >>> arccosh(a)  # 返回[0, 1.317, 1.763]
        
        >>> b = tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> arccosh(b)  # 返回[[0.0, 1.317], [1.763, 2.063]]
    """
    arrlib = x._get_array_lib()
    return tensor(arrlib.arccosh(x.data),device=x.device)

def _arctanh_derivative(x:TN)->tuple[TN]:
    return (1. / (1. - x**2.0).conj(),)

@track_grad(_arctanh_derivative)
def arctanh(x:TN)->TN:
    """
    计算张量的反双曲正切函数。
    
    返回一个新张量，其中每个元素是输入张量对应元素的反双曲正切值，即arctanh(x) = 0.5 * ln((1+x)/(1-x))。
    
    Args:
        x (TN): 输入张量，元素必须在(-1, 1)区间内
        
    Returns:
        TN: 包含输入张量反双曲正切值的新张量
        
    Examples:
        >>> a = tensor([0, 0.5, -0.5])
        >>> arctanh(a)  # 返回[0, 0.549, -0.549]
        
        >>> b = tensor([[0.0, 0.5], [0.8, -0.8]])
        >>> arctanh(b)  # 返回[[0.0, 0.549], [1.099, -1.099]]
    """
    arrlib = x._get_array_lib()
    return tensor(arrlib.arctanh(x.data),device=x.device)

def _sign_derivative(x:TN)->tuple[TN]:
    # sign函数在x=0处不可导，在其他点处梯度为0
    # 创建一个与输入相同形状的零张量作为梯度
    grad = zeros_like(x)
    return (grad,)

@track_grad(_sign_derivative)
def sign(x:TN)->TN:
    """计算张量的符号函数
    
    对于复数 z = a + bi:
    - 如果 z ≠ 0: sign(z) = z / |z| = (a + bi) / √(a² + b²)
    - 如果 z = 0: sign(z) = 0
    
    该函数支持实数和复数张量，并正确处理反向传播。
    """
    arrlib = x._get_array_lib()
    return tensor(arrlib.sign(x.data),device=x.device)

@overload
def where(cond: TN, x: None, y: None) -> Tuple[TN, ...]:
    ...

@overload
def where(cond: TN, x: TN | int | float, y: TN | int | float) -> TN:
    ...

def where(cond: TN, x: TN | int | float | None = None, y: TN | int | float | None = None) -> TN | Tuple[TN, ...]:
    
    if not isinstance(cond,TN):
        cond = tensor(cond)
    
    arrlib = cond._get_array_lib()

    if x is None and y is None:
        tup = arrlib.where(cond.data)
        lst = []
        for idx_arr in tup:
            lst.append(tensor(idx_arr))
        return tuple(lst)

    if x is None or y is None:
        raise RuntimeError('one of x,y is None while the other is Non None')
    
    if not isinstance(x,TN):
        x = tensor(x,device=cond.device)

    if not isinstance(y,TN):
        y = tensor(y,device=cond.device)

    # 条件选择，cond不参与梯度计算
    data = arrlib.where(cond.data, x.data, y.data)
    ret = tensor(data, device = cond.device, requires_grad = (is_grad_enabled() and (x.requires_grad or y.requires_grad)))
    ret.is_leaf = not ret.requires_grad

    if ret.requires_grad:
        ret.fromvars = (x, y)
        ret.parms = (cond,cond)  # cond存入parms而非fromvars
        ret.gradfuncs = (
            lambda r, i: r.grad_value * (cond.data.astype(r.dtype)),
            lambda r, i: r.grad_value * (1.0 - cond.data.astype(r.dtype))
        )
    
    return ret

def clamp(x: TN, min: float | None = None, max: float | None = None, out: TN | None = None) -> TN:
    # 处理参数缺省逻辑
    if min is None and max is None:
        raise ValueError("clamp()需要至少指定min或max参数")
    
    # 处理参数边界
    if min is not None and max is not None:
        if min > max:
            raise ValueError("clamp(): min不能大于max")
    
    # 检查out参数和梯度需求的冲突
    if out is not None and x.requires_grad:
        raise RuntimeError("clamp(): functions with out=... arguments don't support automatic differentiation, but one of the arguments requires grad.")
    
    arrlib = x._get_array_lib()

    # 将None转换为极值
    np_min = min if min is not None else -arrlib.inf
    np_max = max if max is not None else arrlib.inf
    
    # 如果提供了out参数，执行原地操作
    if out is not None:
        # 确保out是TN类型
        if not isinstance(out, TN):
            raise TypeError(f"out must be a TN type, not {type(out)}")
        
        # 检查out张量的形状是否与x兼容
        if out.shape != x.shape:
            raise RuntimeError(f"out tensor shape ({out.shape}) is incompatible with input tensor shape ({x.shape})")
        
        # 执行原地数值截断操作，直接修改out.data的内容
        arrlib.clip(x.data, np_min, np_max, out=out.data)
        
        # 使用out参数时不设置梯度跟踪
        return out
    
    # 否则创建新的张量
    data = arrlib.clip(x.data, np_min, np_max)
    ret = tensor(data, device=x.device, requires_grad = (is_grad_enabled() and x.requires_grad))
    ret.is_leaf = not ret.requires_grad

    if ret.requires_grad:
        ret.fromvars = (x,)
        ret.parms = ((min, max),)  # 保存原始参数用于反向传播
        ret.gradfuncs = (_clamp_backward,)
    else:
        ret.detach_()
        
    return ret

def _clamp_backward(result_tensor: TN, i: int) -> TN:
    x = result_tensor.fromvars[i]
    min_val, max_val = result_tensor.parms[i]
    
    # 梯度掩码计算
    # grad_mask = np.ones_like(x.data)
    grad_mask = ones_like(x)

    # 下限梯度处理
    if min_val is not None:
        # grad_mask = np.where(x.data <= min_val, 0.0, grad_mask)
        grad_mask = where(x<=min_val,0.0,grad_mask)  # type: ignore

    # 上限梯度处理
    if max_val is not None:
        # grad_mask = np.where(x.data >= max_val, 0.0, grad_mask)
        grad_mask = where(x>=max_val,0.0,grad_mask)  # type: ignore
    
    return result_tensor.grad_value * grad_mask

def split(ts: TN, split_indices, dim: int = 0) -> List[TN]:
    """
    沿指定轴分割TN张量，支持计算图记录和梯度反向传播
    参数：
        ts: 输入TN张量
        split_indices: 分割点列表或分割份数（整数）
        dim: 分割轴（默认0）
    """
    arrlib = ts._get_array_lib()
    # 改用array_split支持不均等分割
    split_data = arrlib.array_split(ts.data, split_indices, axis=dim)
    
    # 创建子张量列表（保留计算图）
    sub_tensors = []
    for i, data in enumerate(split_data):
        subt = tensor(data, device=ts.device, requires_grad = (is_grad_enabled() and ts.requires_grad))
        subt.is_leaf = False
        
        # 记录计算图信息
        if ts.requires_grad:
            subt.fromvars = (ts,)
            subt.parms = ((split_indices, dim, i),)  # 新增索引i记录分割位置
            subt.gradfuncs = (_split_backward,)
        
        sub_tensors.append(subt)
    
    return sub_tensors

def _split_backward(result_tensor: TN, i: int) -> TN:
    split_indices, dim, split_pos = result_tensor.parms[i]
    parent = result_tensor.fromvars[i]
    
    # grad = tensor(np.zeros_like(parent.data,dtype=parent.data.dtype))
    grad = zeros_like(parent)

    # 梯度切片计算逻辑
    if isinstance(split_indices, int):
        total_size = parent.data.shape[dim]
        base_size, remainder = divmod(total_size, split_indices)
        split_sizes = [base_size + 1 if j < remainder else base_size for j in range(split_indices)]
        start = builtins.sum(split_sizes[:split_pos])
        end = start + split_sizes[split_pos]
    else:
        start = builtins.sum(split_indices[:split_pos]) if split_pos > 0 else 0
        end = start + split_indices[split_pos]
    
    # 累加归一化后的梯度
    slice_obj = [slice(None)] * parent.data.ndim
    slice_obj[dim] = slice(int(start), int(end))
    grad[tuple(slice_obj)] = result_tensor.grad_value
    
    return grad


def _sort_backward(result_tensor: TN, i: int) -> TN:
    """
    排序操作的反向传播函数
    对于排序后的张量，梯度需要按照原始索引重新排列
    """
    # 获取原始输入张量
    input_tensor = result_tensor.fromvars[i]
    # 获取排序时使用的索引、排序维度
    sorted_indices, sort_dim = result_tensor.parms[i]
    # 获取梯度值
    grad_value = result_tensor.grad_value
    
    # 创建与输入形状相同的零张量
    grad = zeros_like(input_tensor)
    
    # 将梯度按照原始索引放置到正确的位置
    # 使用高级索引将梯度值分配回原始位置
    if sorted_indices.ndim == 1:
        # 1D情况
        grad[sorted_indices] = grad_value
    else:
        # 多维情况，根据排序维度创建正确的坐标网格
        # 创建与sorted_indices形状相同的坐标网格
        coords = list(np.ogrid[tuple(slice(0, size) for size in sorted_indices.shape)])
        # 用排序后的索引替换排序维度的坐标
        coords[sort_dim] = sorted_indices.data
        # 使用这些坐标将梯度值分配回原始位置
        grad[tuple(coords)] = grad_value
    
    return grad

def sort(input: TN, dim: int = -1, descending: bool = False, stable: bool = False, *, out = None) -> Tuple[TN, TN]:
    """
    对张量沿指定维度排序，返回排序后的张量和对应的索引张量
    
    参数:
        input: 输入张量
        dim: 排序维度，默认为最后一维
        descending: 是否降序排列，默认为False（升序）
        stable: 是否使用稳定排序算法，默认为False
        out: 可选的输出元组 (values, indices)
    
    返回:
        包含两个张量的元组 (sorted_values, sorted_indices)
        
    注意: 当使用out参数时，不支持自动微分
    """
    # 检查out参数和梯度需求的冲突
    if out is not None and input.requires_grad:
        raise RuntimeError("sort(): functions with out=... arguments don't support automatic differentiation, but one of the arguments requires grad.")
    
    # 确保dim在有效范围内
    if dim < -input.ndim or dim >= input.ndim:
        raise IndexError(f"Dimension out of range (expected to be in range of [-{input.ndim}, {input.ndim-1}], got {dim})")
    
    # 标准化维度索引
    if dim < 0:
        dim += input.ndim
    
    arrlib = input._get_array_lib()
    # 使用numpy进行排序
    # numpy的sort不支持stable参数，所以这里忽略stable参数
    sorted_indices = arrlib.argsort(input.data, axis=dim)
    
    # 创建用于索引的坐标网格
    coords = list(arrlib.ogrid[tuple(slice(0, size) for size in input.data.shape)])
    # 用排序后的索引替换指定维度的坐标
    coords[dim] = sorted_indices
    
    # 获取排序后的值
    sorted_values_data = input.data[tuple(coords)]
    
    # 如果需要降序，反转排序结果
    if descending:
        # 创建反转索引
        reverse_slice = [slice(None)] * input.data.ndim
        reverse_slice[dim] = slice(None, None, -1)
        sorted_values_data = sorted_values_data[tuple(reverse_slice)]
        # 调整索引以反映降序
        # 对于每个子数组，计算原始位置的反转
        # 这部分比较复杂，需要根据具体维度处理
        # 创建一个与sorted_indices形状相同的数组，用于存储反转后的索引
        reverse_indices = arrlib.empty_like(sorted_indices)
        # 获取指定维度的大小
        dim_size = input.data.shape[dim]
        # 对每个子数组进行处理
        it = np.nditer(sorted_indices, flags=['multi_index', 'refs_ok'])
        while not it.finished:
            # 获取当前元素的多维索引
            idx = list(it.multi_index)
            # 获取当前子数组
            sub_array = sorted_indices[tuple(idx[:dim] + [slice(None)] + idx[dim+1:])]  # type: ignore
            # 创建反转后的子数组索引
            reverse_sub_array = sub_array[::-1].copy()
            # 将反转后的子数组放回原位置
            reverse_indices[tuple(idx[:dim] + [slice(None)] + idx[dim+1:])] = reverse_sub_array
            it.iternext()
        sorted_indices = reverse_indices
    
    # 创建排序值张量
    sorted_values = tensor(sorted_values_data, device=input.device,requires_grad=(is_grad_enabled() and input.requires_grad))
    sorted_values.is_leaf = not sorted_values.requires_grad
    
    # 创建索引张量（索引张量不需要梯度）
    sorted_indices_tensor = tensor(sorted_indices, device=input.device,requires_grad=False)
    
    # 设置梯度信息
    if sorted_values.requires_grad:
        sorted_values.fromvars = (input,)
        sorted_values.parms = ((sorted_indices_tensor, dim),)
        sorted_values.gradfuncs = (_sort_backward,)
    
    # 处理out参数
    if out is not None:
        if not isinstance(out, tuple) or len(out) != 2:
            raise TypeError("out must be a tuple of two tensors")
        
        values_out, indices_out = out
        
        # 检查out参数类型
        if not isinstance(values_out, TN) or not isinstance(indices_out, TN):
            raise TypeError("out tensors must be of type TN")
        
        # 检查形状兼容性
        if values_out.shape != sorted_values.shape:
            raise RuntimeError(f"out values tensor shape ({values_out.shape}) is incompatible with result shape ({sorted_values.shape})")
        if indices_out.shape != sorted_indices_tensor.shape:
            raise RuntimeError(f"out indices tensor shape ({indices_out.shape}) is incompatible with result shape ({sorted_indices_tensor.shape})")
        
        # 复制数据到out参数
        arrlib.copyto(values_out.data, sorted_values.data)
        arrlib.copyto(indices_out.data, sorted_indices_tensor.data)
        
        # 使用out参数时不设置梯度跟踪
        return (values_out, indices_out)
    
    # 返回排序结果
    return (sorted_values, sorted_indices_tensor)

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
    if out is not None and input.requires_grad:
        raise RuntimeError("argsort(): functions with out=... arguments don't support automatic differentiation, but one of the arguments requires grad.")
    
    if dim < -input.ndim or dim >= input.ndim:
        raise IndexError(f"Dimension out of range (expected to be in range of [-{input.ndim}, {input.ndim-1}], got {dim})")
    
    if dim < 0:
        dim += input.ndim
    
    arrlib = input._get_array_lib()
    sorted_indices = arrlib.argsort(input.data, axis=dim)
    
    if descending:
        reverse_indices = arrlib.empty_like(sorted_indices)
        dim_size = input.data.shape[dim]
        it = arrlib.nditer(sorted_indices, flags=['multi_index', 'refs_ok'])
        while not it.finished:
            idx = list(it.multi_index)
            sub_array = sorted_indices[tuple(idx[:dim] + [slice(None)] + idx[dim+1:])]  # type: ignore
            reverse_sub_array = sub_array[::-1].copy()
            reverse_indices[tuple(idx[:dim] + [slice(None)] + idx[dim+1:])] = reverse_sub_array
            it.iternext()
        sorted_indices = reverse_indices
    
    sorted_indices_tensor = tensor(sorted_indices, device=input.device, requires_grad=False)
    
    if out is not None:
        if not isinstance(out, TN):
            raise TypeError("out must be a tensor of type TN")
        if out.shape != sorted_indices_tensor.shape:
            raise RuntimeError(f"out tensor shape ({out.shape}) is incompatible with result shape ({sorted_indices_tensor.shape})")
        arrlib.copyto(out.data, sorted_indices_tensor.data)
        return out
    
    return sorted_indices_tensor

def stack(tensors: Tuple[TN, ...]|List[TN], dim: int = 0) -> TN:
    """沿新维度堆叠张量"""
    arrlib = tensors[0]._get_array_lib()
    data = arrlib.stack([t.data for t in tensors], axis=dim)
    
    # 梯度需求判断优化为any()
    requires_grad = (is_grad_enabled() and builtins.any(t.requires_grad for t in tensors))
    
    ret = tensor(data, device=tensors[0].device, requires_grad=requires_grad)
    ret.is_leaf = not requires_grad

    if requires_grad:
        ret.fromvars = tuple(tensors)  # 确保tuple类型
        num = len(tensors)
        ret.parms = (dim,) * num
        ret.gradfuncs = (_stack_backward,) * num
    
    return ret

def concatenate(tensors: Tuple[TN, ...]|List[TN], dim: int = 0) -> TN:
    """沿指定轴连接张量"""
    arrlib = tensors[0]._get_array_lib()
    data = arrlib.concatenate([t.data for t in tensors], axis=dim)
    
    requires_grad = (is_grad_enabled() and builtins.any(t.requires_grad for t in tensors))
    
    ret = tensor(data, device=tensors[0].device, requires_grad=requires_grad)
    ret.is_leaf = not requires_grad

    if requires_grad:
        ret.fromvars = tuple(tensors)  # 确保tuple类型
        num = len(tensors)
        ret.parms = (dim,) * num
        ret.gradfuncs = (_concatenate_backward,) * num
    
    return ret

def cat(tensors: Tuple[TN, ...]|List[TN], dim: int = 0) -> TN:
    return concatenate(tensors, dim)

def vstack(tensors: Tuple[TN, ...]|List[TN]) -> TN:
    """垂直堆叠张量
    
    对于一维张量，将它们作为行堆叠（创建二维数组）；
    对于多维张量，沿第0轴（行方向）连接。
    """

    # 检查是否所有张量都是0D的
    all_0d = all(t.ndim == 0 for t in tensors)

    # 检查是否所有张量都是一维的
    all_1d = all(t.ndim == 1 for t in tensors)
    if all_0d:
        # 对于0D张量，添加新维度后沿第0轴连接
        expanded_tensors = [t.reshape(1, 1) for t in tensors]
        return concatenate(expanded_tensors, dim=0)
    elif all_1d:
        # 对于一维张量，添加新维度后沿第0轴连接
        # 这样可以利用concatenate的梯度跟踪能力
        expanded_tensors = [t.reshape(1, -1) for t in tensors]
        return concatenate(expanded_tensors, dim=0)
    else:
        # 对于多维张量，沿第0轴连接
        return concatenate(tensors, dim=0)

def hstack(tensors: Tuple[TN, ...]|List[TN]) -> TN:
    """水平堆叠张量
    
    对于一维张量，水平连接成一维数组；
    对于多维张量，沿第1轴（列方向）连接。
    """
    # 检查是否所有张量都是0D的
    all_0d = all(t.ndim == 0 for t in tensors)

    # 检查是否所有张量都是一维的
    all_1d = all(t.ndim == 1 for t in tensors)
    if all_0d:
        # 对于0D张量，添加新维度后沿第0轴连接
        expanded_tensors = [t.reshape(1) for t in tensors]
        return concatenate(expanded_tensors, dim=0)
    elif all_1d:
        # 对于一维张量，沿第0轴连接（水平堆叠）
        return concatenate(tensors, dim=0)
    else:
        # 对于多维张量，沿第1轴连接
        return concatenate(tensors, dim=1)

def _stack_backward(result_tensor: TN, i: int) -> TN:
    dim = result_tensor.parms[i]
    grad = result_tensor.grad_value
    slices = [slice(None)] * grad.data.ndim
    slices[dim] = i  # type: ignore
    return grad[tuple(slices)]

def _concatenate_backward(result_tensor: TN, i: int) -> TN:
    dim = result_tensor.parms[i]
    grad_data = result_tensor.grad_value
    
    # 获取所有输入张量在拼接轴上的尺寸
    split_sizes = [t.data.shape[dim] for t in result_tensor.fromvars]
    
    # 使用Python内置的accumulate函数计算累积和，避免不必要的数组转换
    split_points = list(accumulate(split_sizes[:-1])) if len(split_sizes) > 1 else []
    
    try:
        grads = split(grad_data,split_points, dim=dim)
        return grads[i]
    
    except ValueError as e:
        raise RuntimeError(f"梯度分割失败: {e}") from None
    
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
    flat_data = input.data.flatten()
    
    arrlib = input._get_array_lib()
    dev = input.device

    # 使用NumPy的unique函数获取唯一值
    # 根据不同的参数组合调用不同版本的np.unique
    if return_inverse and return_counts and return_indices:
        # 返回所有可选值
        unique_values, indices, inverse_indices, counts = arrlib.unique(
            flat_data, return_index=True, return_inverse=True, return_counts=True)
        
        # 创建结果张量
        result = tensor(unique_values,device=dev)
        indices_tensor = tensor(indices, dtype=int64,device=dev)
        inverse_tensor = tensor(inverse_indices, dtype=int64,device=dev)
        counts_tensor = tensor(counts, dtype=int64,device=dev)
        
        return result, inverse_tensor, indices_tensor, counts_tensor
    elif return_inverse and return_counts:
        # 返回唯一值、逆索引和计数
        unique_values, inverse_indices, counts = arrlib.unique(
            flat_data, return_inverse=True, return_counts=True)
        
        # 创建结果张量
        result = tensor(unique_values,device=dev)
        inverse_tensor = tensor(inverse_indices, dtype=int64,device=dev)
        counts_tensor = tensor(counts, dtype=int64,device=dev)
        
        return result, inverse_tensor, counts_tensor
    elif return_inverse and return_indices:
        # 返回唯一值、逆索引和首次出现索引
        unique_values, indices, inverse_indices = arrlib.unique(
            flat_data, return_index=True, return_inverse=True)
        
        # 创建结果张量
        result = tensor(unique_values,device=dev)
        indices_tensor = tensor(indices, dtype=int64,device=dev)
        inverse_tensor = tensor(inverse_indices, dtype=int64,device=dev)
        
        return result, inverse_tensor, indices_tensor
    elif return_counts and return_indices:
        # 返回唯一值、首次出现索引和计数
        unique_values, indices, counts = arrlib.unique(
            flat_data, return_index=True, return_counts=True)
        
        # 创建结果张量
        result = tensor(unique_values,device=dev)
        indices_tensor = tensor(indices, dtype=int64,device=dev)
        counts_tensor = tensor(counts, dtype=int64,device=dev)
        
        return result, indices_tensor, counts_tensor
    elif return_inverse:
        # 只返回唯一值和逆索引
        unique_values, inverse_indices = arrlib.unique(
            flat_data, return_inverse=True)
        
        # 创建结果张量
        result = tensor(unique_values,device=dev)
        inverse_tensor = tensor(inverse_indices, dtype=int64,device=dev)
        
        return result, inverse_tensor
    elif return_counts:
        # 只返回唯一值和计数
        unique_values, counts = arrlib.unique(
            flat_data, return_counts=True)
        
        # 创建结果张量
        result = tensor(unique_values,device=dev)
        counts_tensor = tensor(counts, dtype=int64,device=dev)
        
        return result, counts_tensor
    elif return_indices:
        # 只返回唯一值和首次出现索引
        unique_values, indices = arrlib.unique(
            flat_data, return_index=True)
        
        # 创建结果张量
        result = tensor(unique_values,device=dev)
        indices_tensor = tensor(indices, dtype=int64,device=dev)
        
        return result, indices_tensor
    else:
        # 只返回唯一值
        unique_values = arrlib.unique(flat_data)
        return tensor(unique_values,device=dev)
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
        other = tensor(other,device=dev)
    
    if other.device != dev:
        raise RuntimeError(f'Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!')
    
    # 前向计算：使用numpy的maximum
    value = arrlib.maximum(input.data, other.data)
    ret = tensor(value, device=dev, requires_grad=(is_grad_enabled() and (input.requires_grad or other.requires_grad)))
    ret.is_leaf = not ret.requires_grad
    
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
    
    # 创建比较掩码
    input_data = input_tensor.data
    other_data = other_tensor.data
    
    # 计算梯度掩码
    # input > other: 1.0
    # input == other: 0.5  # 与PyTorch一致
    # input < other: 0.0
    # 使用输入张量的数据类型来创建掩码值，避免类型转换警告
    mask_dtype = input_tensor.dtype
    arrlib = input_tensor._get_array_lib()
    mask = arrlib.where(input_data > other_data, arrlib.array(1.0, dtype=mask_dtype), 
                   arrlib.where(input_data < other_data, arrlib.array(0.0, dtype=mask_dtype), arrlib.array(0.5, dtype=mask_dtype)))
    
    # 应用梯度掩码
    grad = result_tensor.grad_value * mask
    
    # 处理广播情况
    input_shape = input_tensor.shape
    result_shape = result_tensor.shape
    
    if input_shape != result_shape:
        # 获取需要缩减的广播轴
        broadcast_axes = _get_broadcast_axis(result_shape, input_shape)
        grad = sum(grad, dim=broadcast_axes, keepdim=False).reshape(input_shape)
    
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
    
    # 创建比较掩码
    other_data = other_tensor.data
    input_data = input_tensor.data
    
    # 计算梯度掩码
    # other > input: 1.0
    # other == input: 0.5  # 与PyTorch一致
    # other < input: 0.0
    # 使用输入张量的数据类型来创建掩码值，避免类型转换警告
    mask_dtype = other_tensor.dtype
    arrlib = other_tensor._get_array_lib()
    mask = arrlib.where(other_data > input_data, arrlib.array(1.0, dtype=mask_dtype),
                   arrlib.where(other_data < input_data, arrlib.array(0.0, dtype=mask_dtype), arrlib.array(0.5, dtype=mask_dtype)))
    
    # 应用梯度掩码
    grad = result_tensor.grad_value * mask
    
    # 处理广播情况
    other_shape = other_tensor.shape
    result_shape = result_tensor.shape
    
    if other_shape != result_shape:
        # 获取需要缩减的广播轴
        broadcast_axes = _get_broadcast_axis(result_shape, other_shape)
        grad = sum(grad, dim=broadcast_axes, keepdim=False).reshape(other_shape)
    
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
        other = tensor(other,device=dev)
    
    if other.device != dev:
        raise RuntimeError(f'Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!')
        
    # 前向计算：使用numpy的minimum
    value = arrlib.minimum(input.data, other.data)
    ret = tensor(value, device=dev, requires_grad=(is_grad_enabled() and (input.requires_grad or other.requires_grad)))
    ret.is_leaf = not ret.requires_grad
    
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
    
    # 创建比较掩码
    input_data = input_tensor.data
    other_data = other_tensor.data
    
    # 计算梯度掩码
    # input < other: 1.0
    # input == other: 0.5  
    # input > other: 0.0
    # 使用输入张量的数据类型来创建掩码值，避免类型转换警告
    mask_dtype = input_tensor.dtype
    arrlib = input_tensor._get_array_lib()
    mask = arrlib.where(input_data < other_data, arrlib.array(1.0, dtype=mask_dtype), 
                   arrlib.where(input_data > other_data, arrlib.array(0.0, dtype=mask_dtype), arrlib.array(0.5, dtype=mask_dtype)))
    
    # 应用梯度掩码
    grad = result_tensor.grad_value * mask
    
    # 处理广播情况
    input_shape = input_tensor.shape
    result_shape = result_tensor.shape
    
    if input_shape != result_shape:
        # 获取需要缩减的广播轴
        broadcast_axes = _get_broadcast_axis(result_shape, input_shape)
        grad = sum(grad, dim=broadcast_axes, keepdim=False).reshape(input_shape)
    
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
    
    # 创建比较掩码
    other_data = other_tensor.data
    input_data = input_tensor.data
    
    # 计算梯度掩码
    # other < input: 1.0
    # other == input: 0.5
    # other > input: 0.0
    # 使用输入张量的数据类型来创建掩码值，避免类型转换警告
    mask_dtype = other_tensor.dtype
    arrlib = other_tensor._get_array_lib()
    mask = arrlib.where(other_data < input_data, arrlib.array(1.0, dtype=mask_dtype),
                   arrlib.where(other_data > input_data, arrlib.array(0.0, dtype=mask_dtype), arrlib.array(0.5, dtype=mask_dtype)))
    
    # 应用梯度掩码
    grad = result_tensor.grad_value * mask
    
    # 处理广播情况
    other_shape = other_tensor.shape
    result_shape = result_tensor.shape
    
    if other_shape != result_shape:
        # 获取需要缩减的广播轴
        broadcast_axes = _get_broadcast_axis(result_shape, other_shape)
        grad = sum(grad, dim=broadcast_axes, keepdim=False).reshape(other_shape)
    
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
    if input.data.ndim < 2:
        raise ValueError("Input tensor must be at least 2-dimensional")
    
    # 检查dim1和dim2是否有效
    if dim1 == dim2:
        raise ValueError(f"dim1 and dim2 cannot be the same value ({dim1})")
    
    # 处理负维度
    if dim1 < 0:
        dim1 = input.data.ndim + dim1
    if dim2 < 0:
        dim2 = input.data.ndim + dim2
    
    # 检查维度是否在有效范围内
    if dim1 < 0 or dim1 >= input.data.ndim:
        raise IndexError(f"Dimension out of range (expected to be in range of [-{input.data.ndim}, {input.data.ndim-1}], but got {dim1})")
    if dim2 < 0 or dim2 >= input.data.ndim:
        raise IndexError(f"Dimension out of range (expected to be in range of [-{input.data.ndim}, {input.data.ndim-1}], but got {dim2})")
    
    # 计算对角线长度
    m, n = input.shape[dim1], input.shape[dim2]
    diagonal_len, start1, start2 = _get_diagonal_len_start_pop(m,n,offset)
    
    # 如果对角线长度为0，返回空张量
    if diagonal_len == 0:
        output_shape = list(input.shape)
        output_shape.pop(builtins.max(dim1, dim2))
        output_shape[builtins.min(dim1, dim2)] = 0
        # 将列表转换为元组并解包传递给zeros函数
        return zeros(*tuple(output_shape), dtype=input.dtype, requires_grad=input.requires_grad)
    
    
    # 创建对角线索引
    indices = [slice(None)] * input.ndim
    
    arrlib = input._get_array_lib()

    # 创建对角线位置的索引数组
    idx = arrlib.arange(diagonal_len, dtype=int)
    
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
        n = input.data.shape[0]
        
        # 创建单位矩阵模板
        identity_tensor = eye(n)
        
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
    
    # 创建对角线位置的索引数组
    arrlib = input._get_array_lib()
    idx = arrlib.arange(diagonal_len, dtype=int)
    
    # 设置dim1和dim2维度的索引，与diagonal函数保持一致
    indices[dim1] = start1 + idx
    indices[dim2] = start2 + idx
 
    # 准备填充值
    if not isinstance(value, TN):
        fill_value = tensor(value, dtype = input.dtype,device=input.device)
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
    
    # 创建对角线位置的索引数组
    arrlib = input._get_array_lib()
    idx = arrlib.arange(diagonal_len, dtype=int)
    
    # 设置dim1和dim2维度的索引，与diagonal函数保持一致
    indices[dim1] = start1 + idx
    indices[dim2] = start2 + idx
 
    # 准备填充值
    if not isinstance(value, TN):
        fill_value = tensor(value, dtype = input.dtype,device=input.device)
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
    identity_tensor = eye(n, dtype=v.dtype)
    
    # 使用广播机制生成对角矩阵
    # 将输入张量扩展维度，然后与单位矩阵相乘
    # 形状变化：(*, n) -> (*, 1, n) -> (*, n, n)
    batch_eye = identity_tensor.expand(shape[:-1] + (n, n))
    v_expanded = v.unsqueeze(-2)  # 扩展为 (*, 1, n)
    
    # 逐元素相乘，利用广播机制
    result = v_expanded * batch_eye
    return result

# end of batch_diag

def nonzero(input: TN, *, as_tuple: bool = False) -> TN | Tuple[TN, ...]:
    """
    返回输入张量中所有非零元素的索引。
    
    参数:
        input: 输入张量
        as_tuple: 如果为True，返回一个元组，其中每个元素对应一个维度的索引张量；
                 如果为False（默认），返回一个二维张量，每行是一个非零元素的坐标
    
    返回:
        TN或Tuple[TN, ...]: 非零元素的索引
        - 当as_tuple=False时，返回形状为(N, input.ndim)的二维张量，其中N是非零元素的数量
        - 当as_tuple=True时，返回一个元组，包含input.ndim个一维张量，每个对应一个维度的索引
    
    示例:
        >>> x = tensor([[1, 0, 2], [0, 3, 0]])
        >>> nonzero(x)
        tensor([[0, 0], [0, 2], [1, 1]])
        >>> nonzero(x, as_tuple=True)
        (tensor([0, 0, 1]), tensor([0, 2, 1]))
    """
    if not isinstance(input, TN):
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input)}")
    
    # 获取非零元素的索引
    arrlib = input._get_array_lib()
    indices = arrlib.nonzero(input.data)
    
    if as_tuple:
        # 返回元组形式，每个维度一个张量
        result = tuple(tensor(idx) for idx in indices)
    else:
        # 返回二维张量形式，每行是一个坐标
        if len(indices) > 0:
            # 将索引数组堆叠成二维数组
            result_data = arrlib.stack(indices, axis=1)
        else:
            # 如果没有非零元素，返回空的二维数组
            result_data = arrlib.empty((0, input.ndim), dtype=np.int64)
        result = tensor(result_data,device=input.device)  # type: ignore
    
    return result

def equal(a:TN,b:TN)->bool:
    """
    检查两个张量是否相等，返回一个布尔值。
    
    参数:
        a (TN): 第一个张量
        b (TN): 第二个张量
        
    返回:
        bool: 如果两个张量形状相同且所有元素都相等，则返回True；否则返回False
    """
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
    a_data = a.data
    b_data = b.data
    
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
    value = arrlib.isinf(x.data)
    ret = tensor(value,device=x.device)
    
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
    value = arrlib.isnan(x.data)
    ret = tensor(value,device=x.device)
    
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
    value = arrlib.isreal(x.data)
    ret = tensor(value,device=x.device)
    
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
    data = arrlib.tril(input_tensor.data, k=diagonal)
    ret = tensor(data, device=input_tensor.device, requires_grad = (is_grad_enabled() and input_tensor.requires_grad))
    ret.is_leaf = not ret.requires_grad
    
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
    data = arrlib.triu(input_tensor.data, k=diagonal)
    ret = tensor(data, device=input_tensor.device, requires_grad = (is_grad_enabled() and input_tensor.requires_grad))
    ret.is_leaf = not ret.requires_grad
    
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