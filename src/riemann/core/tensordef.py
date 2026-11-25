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
import functools
import inspect
import itertools
import warnings
from typing import Callable, Any, Optional, Iterable, Iterator, Mapping
from contextlib import contextmanager
import numpy as np
from ..cuda import Device, cp, get_default_device
from .dtype import *
from .ad_context import *

class APIRegistry:
    def __init__(self):
        pass

    def register(self, fn):
        name = fn.__name__
        # Prevent duplicate registration
        if hasattr(self, name):
            raise RuntimeError(f"Name '{name}' is already registered; overwriting is not allowed")
        setattr(self, name, fn)
        return fn

    def get(self, name: str):
        if not hasattr(self, name):
            raise KeyError(f"Registered function not found: {name}")
        return getattr(self, name)


# Global unique registry instance
api_registry = APIRegistry()


# 原地操作的自包含记录：反向梯度规则和 __setitem__ 重放都只读这一条记录，
# 不再依赖 _inplace_ops 与 fromvars/parms 的下标一一对应关系。
#   method:       可用于重放的原地方法名；None 表示该操作不可重放
#   right:        本次操作的右值；一元原地运算为占位变量
#   stored_index: 本次操作实际作用的索引；() 表示作用于整个张量
#   left_copy:    覆盖前备份的左值副本，供 mul/div/pow/mod/abs 的反向使用，其余为 None
InplaceRecord = namedtuple(
    'InplaceRecord', ('method', 'right', 'stored_index', 'left_copy')
)


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
        
        self._array:np.ndarray|cp.ndarray = arr     # 存储张量数据的NumPy或Cupy数组
        self.requires_grad = requires_grad          # 是否需要计算梯度，默认为False
        self.retains_grad = False       # 是否保留梯度，默认为False
        self.grad:TN = None             #与_array的shape一致的张量对象，用于存放梯度，
                                        #初始化为None是为了节省空间，计算梯度时才分配空间
        
        self.fromvars = ()              #正向运算顺序记录来源，可同时含 TN 与非 TN 操作数
                                        #（如 add(x, 2.0)、where(cond, x, 1.0)）。反向请求
                                        #（requires_grad=True）或前向请求（forward_ad_trace_active()）
                                        #任一成立时记录；前向建图不改变 requires_grad
        self.parms = ()                 #算子附加参数（如sum的dim、where的cond），不参与图遍历
        self._grad_rule = None          #梯度规则：接收本节点，返回与fromvars等长且按位置对齐
                                        #的梯度元组，非可微来源用None占位
        self._jvp_rule = None           #前向规则：接收(frame, 本节点)，从frame拉取各入边tangent，
                                        #用普通TN算子计算并返回本节点的tangent（None表示零方向）；
                                        #无规则时为None
        self._inplace_ops = ()          #原地操作记录，最新在前，元素为InplaceRecord；
                                        #同时供反向梯度规则和__setitem__重放使用
        self._rcv_grad_count = 0        #在具体计算图中，self可接收梯度的计数，计算梯度时临时使用
        self._grad_value:TN = None      #用于计算反向传播的梯度，最终计算结果保存在grad里，该变量用于临时计算
        
        self._output_of = None          #记录该张量是哪个模块的输出（单引用），用于反向传播钩子机制
        self._module_input_of = None    #记录该张量是哪个模块的输入（单引用），用于反向传播钩子机制
        
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
        return api_registry._disp(self, format_spec)

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
        if self.ndim == 0:
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
        
        if result.requires_grad or forward_ad_trace_active():
            result.fromvars = (self,)
            result.parms = ()
            # 实部梯度：result._grad_value 需转换为复数后返回
            result._grad_rule = lambda result: (result._grad_value + 0.0j,)
            result._jvp_rule = _real_jvp
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
        
        if result.requires_grad or forward_ad_trace_active():
            result.fromvars = (self,)
            result.parms = ()
            # 虚部梯度：result._grad_value 需转换为与虚部对应的复数后返回
            result._grad_rule = lambda result: (1.0j * result._grad_value,)
            result._jvp_rule = _imag_jvp
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
        if ret.requires_grad or forward_ad_trace_active():
            ret.fromvars = (self,)
            # 共轭操作的梯度是元素级别的共轭
            ret._grad_rule = lambda result: (result._grad_value.conj(),)
            ret._jvp_rule = _conj_jvp
        
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
            device_idx = self._array.device.id
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
            
            if ret.requires_grad or forward_ad_trace_active():
                # 设置计算图相关属性
                ret.fromvars = (self,)                
                # 保存原始数据类型和设备信息用于反向传播
                ret.parms = ((self.dtype,self.device),)                
                # 梯度需转换回原始数据类型和设备
                ret._grad_rule = lambda result: (
                    result._grad_value.to(
                        dtype=result.parms[0][0],
                        device=result.parms[0][1],
                    ),
                )
                ret._jvp_rule = _to_jvp
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
        # 依赖不变量：fromvars 仅在需要建图时登记，且非空时至少含一个可微 TN 来源；
        # requires_grad=False 的节点即使有 fromvars 也视为叶子（不会被反向访问）。
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
        
        if ret.requires_grad or forward_ad_trace_active():
            ret.fromvars = (self,)
            ret._grad_rule = lambda result: (result._grad_value,)
            ret._jvp_rule = _contiguous_jvp
        
        return ret

    def __getitem__(self, index):
        return _getitem(self, index)

    def __setitem__(self, index, val):
        return api_registry._setitem(self, index, val)

    # 原地/非原地成对接口：xxxat_ 直接在 self 上执行并记录依赖、返回 self；
    # xxxat 在副本上执行、返回新张量，self 不变。
    # 具体实现（ufunc.at 通用路径与全量索引向量化路径）已迁至 tensorfuncs。
    def setat_(self, index, val):
        """原地执行 self[index] = val（覆盖写），返回 self。"""
        return api_registry.setat_(self, index, val)

    def setat(self, index, val):
        """非原地执行 self[index] = val，返回新张量。"""
        return api_registry.setat(self, index, val)

    def addat_(self, index, val):
        """原地执行 self[index] += val（重复索引累加），返回 self。"""
        return api_registry.addat_(self, index, val)

    def addat(self, index, val):
        """非原地执行 self[index] += val，返回新张量。"""
        return api_registry.addat(self, index, val)

    def subat_(self, index, val):
        """原地执行 self[index] -= val（重复索引累加），返回 self。"""
        return api_registry.subat_(self, index, val)

    def subat(self, index, val):
        """非原地执行 self[index] -= val，返回新张量。"""
        return api_registry.subat(self, index, val)
    
    def mulat_(self, index, val):
        """原地执行 self[index] *= val（重复索引累加），返回 self。"""
        return api_registry.mulat_(self, index, val)

    def mulat(self, index, val):
        """非原地执行 self[index] *= val，返回新张量。"""
        return api_registry.mulat(self, index, val)

    def divat_(self, index, val):
        """原地执行 self[index] /= val（重复索引累加），返回 self。"""
        return api_registry.divat_(self, index, val)

    def divat(self, index, val):
        """非原地执行 self[index] /= val，返回新张量。"""
        return api_registry.divat(self, index, val)

    def powat_(self, index, val):
        """原地执行 self[index] **= val（重复索引累加），返回 self。"""
        return api_registry.powat_(self, index, val)

    def powat(self, index, val):
        """非原地执行 self[index] **= val，返回新张量。"""
        return api_registry.powat(self, index, val)

    def floordivat_(self, index, val):
        """原地执行 self[index] //= val（重复索引累加），返回 self。"""
        return api_registry.floordivat_(self, index, val)

    def floordivat(self, index, val):
        """非原地执行 self[index] //= val，返回新张量。"""
        return api_registry.floordivat(self, index, val)

    def modat_(self, index, val):
        """原地执行 self[index] %= val（重复索引累加），返回 self。"""
        return api_registry.modat_(self, index, val)

    def modat(self, index, val):
        """非原地执行 self[index] %= val，返回新张量。"""
        return api_registry.modat(self, index, val)

    def gather(self, dim: int, index:TN):
        """
        根据指定的 ​维度和​索引收集元素，
        常用于需要根据条件动态选择数据的场景（如交叉熵损失函数中选择正确类别的预测概率）
        等价于模块级函数 riemann.gather(self, dim, index)
        :param dim: 收集维度（0 ≤ dim < self.ndim）
        :param index: 索引张量（dtype=int64）
        :return: 新TN实例，形状与index相同
        """
        return api_registry.gather(self, dim, index)

    def index_select(self, dim: int, index: TN) -> 'TN':
        """沿指定维度按索引选择元素，等价于 riemann.index_select(self, dim, index)"""
        return api_registry.index_select(self, dim, index)

    def masked_select(self, mask: TN) -> 'TN':
        """按布尔掩码选择元素，返回一维张量，等价于 riemann.masked_select(self, mask)"""
        return api_registry.masked_select(self, mask)

    def take(self, index: TN) -> 'TN':
        """按扁平化下标选择元素，返回与 index 同形状的张量，等价于 riemann.take(self, index)"""
        return api_registry.take(self, index)

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
        return api_registry.index_put_(self, indices, values, accumulate)

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
        return api_registry.scatter(self, dim, index, src, value=value)

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
        return api_registry.scatter_(self, dim, index, src, value=value)

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
        return api_registry.scatter_add(self, dim, index, src)

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
        return api_registry.scatter_add_(self, dim, index, src)

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
            
            if newobj.requires_grad or forward_ad_trace_active():
                newobj.fromvars = (self,)
                newobj.parms = ((dim1,dim2),)
                newobj._grad_rule = _transpose_grad
                newobj._jvp_rule = _transpose_jvp
            
            return newobj
        else:
            raise RuntimeError("transpose() can only be used on tensors of 2D or more dimension")

    def trace(self):
        '''
        计算张量的迹（对角线元素的和）
        '''
        return api_registry.trace(self)

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
        if requires_grad or forward_ad_trace_active():
            newobj.fromvars = (self,)
            newobj.parms = (dims,)
            newobj._grad_rule = _permute_grad
            newobj._jvp_rule = _permute_jvp
        
        return newobj

    def neg_(self):
        """就地取负，等价于 torch.Tensor.neg_（实现位于 tensorfuncs）。"""
        return api_registry.neg_(self)

    def abs_(self):
        """就地取绝对值，等价于 torch.Tensor.abs_（实现位于 tensorfuncs）。"""
        return api_registry.abs_(self)

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
        if is_grad_enabled() and not forward_ad_trace_active() and self.is_leaf and self.requires_grad:
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
        if is_grad_enabled() and not forward_ad_trace_active() and self.is_leaf and self.requires_grad:
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
        if is_grad_enabled() and not forward_ad_trace_active() and self.is_leaf and self.requires_grad:
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
        if is_grad_enabled() and not forward_ad_trace_active() and self.is_leaf and self.requires_grad:
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
        if is_grad_enabled() and not forward_ad_trace_active() and self.is_leaf and self.requires_grad:
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
        if is_grad_enabled() and not forward_ad_trace_active() and self.is_leaf and self.requires_grad:
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
        if is_grad_enabled() and not forward_ad_trace_active() and self.is_leaf and self.requires_grad:
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
        self._grad_rule = None
        self._jvp_rule = None
        self._inplace_ops = ()
        self.parms = ()
        self._grad_value = None
        self._rcv_grad_count = 0
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
        if ret.requires_grad or forward_ad_trace_active():
            ret.fromvars=(self,)
            ret._grad_rule = lambda result: (result._grad_value,)
            ret._jvp_rule = _clone_jvp
        return ret
    
    def copy(self):
        '''返回一个新张量，复制self数据，不共享内存，也不依赖self'''
        return TN(self._array.copy())

    def copy_(self,src):
        '''原地复制src到self'''
        if is_grad_enabled() and not forward_ad_trace_active() and self.is_leaf and self.requires_grad:
            raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation.')
        
        # 会自动调用__setitem__，触发梯度计算
        # 用省略号...作全局索引，表示对整个张量赋值（':'不能用于0维张量，...可以）
        self[...] = src
        
        return self
    
    def copy_to(self,target:TN|np.ndarray|None=None):
        '''将self数据复制到target'''
        arrtype = (np.ndarray,cp.ndarray) if cp else (np.ndarray,)

        if isinstance(target,TN):
            if not _same_device(target, self):
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
        if is_grad_enabled() and not forward_ad_trace_active() and self.requires_grad and self.is_leaf:
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
        """将张量的每个元素限制在 [min, max] 范围内（原地操作，实现位于 tensorfuncs）。"""
        return api_registry.clamp_(self, min, max)

    def masked_fill_(self, mask: TN, value: any):  # type: ignore
        """原地将张量中掩码为 True 的元素填充为指定值（实现位于 tensorfuncs）。"""
        return api_registry.masked_fill_(self, mask, value)
    
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
        return api_registry.masked_fill(self, mask, value)
    
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
        
        if ret.requires_grad or forward_ad_trace_active():
            ret.fromvars = (self,)
            ret.parms = (new_dim,)
            ret._grad_rule = _squeeze_grad
            ret._jvp_rule = _squeeze_jvp
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
            if ret.requires_grad or forward_ad_trace_active():
                ret.fromvars = (self,)
                # parms 统一存元组形式，保证反向函数拿到的参数类型一致
                ret.parms = ((dim,),)
                ret._grad_rule = _unsqueeze_grad
                ret._jvp_rule = _unsqueeze_jvp
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
            if ret.requires_grad or forward_ad_trace_active():
                ret.fromvars = (self,)
                # 元组形式原样存入parms，反向函数直接复用
                ret.parms = (dim,)
                ret._grad_rule = _unsqueeze_grad
                ret._jvp_rule = _unsqueeze_jvp
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
        
        if ret.requires_grad or forward_ad_trace_active():
            ret.fromvars = (self,)
            ret.parms = ((original_shape, new_shape),)
            ret._grad_rule = _reshape_grad
            ret._jvp_rule = _reshape_jvp
        return ret

    def unfold(self, dimension: int, size: int, step: int = None) -> 'TN':
        return api_registry.unfold(self, dimension, size, step)
    
    def fold(self, dimension: int, size: int, step: int = None, output_size: int = None) -> 'TN':
        return api_registry.fold(self, dimension, size, step, output_size)
    
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
        if ret.requires_grad or forward_ad_trace_active():
            ret.fromvars = (self,)
            ret.parms = ((original_shape, new_shape),)
            ret._grad_rule = _expand_grad
            ret._jvp_rule = _expand_jvp
    
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
        return api_registry.broadcast_to(self, new_shape)
    
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
        return api_registry.repeat(self, repeats)
    
    def flip(self, dims:list[int]|tuple[int,...]) -> TN:
        return flip(self,dims)

    def clamp(self,min:float|None=None, max:float|None=None):  # type: ignore
        return api_registry.clamp(self, min, max)
    
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
        return api_registry.sort(self, dim=dim, descending=descending)

    def topk(self, k:int, dim:int=-1, largest:bool=True, sorted:bool=True):  # type: ignore
        return api_registry.topk(self, k, dim=dim, largest=largest, sorted=sorted)

    def multinomial(self, num_samples:int, replacement:bool=False, *, generator=None):  # type: ignore
        return api_registry.multinomial(self, num_samples, replacement, generator=generator)

    def maximum(self, other):
        return api_registry.maximum(self, other)
    
    def minimum(self, other):
        return api_registry.minimum(self, other)

    def atan2(self, other):
        return api_registry.atan2(self, other)

    def arctan2(self, other):
        return api_registry.arctan2(self, other)

    def sum(self, dim:int|tuple|None=None, keepdim:bool=False):  # type: ignore
        ret = sum(self, dim, keepdim)
        return ret

    def logsumexp(self, dim:int|tuple|None=None, keepdim:bool=False):
        return api_registry.logsumexp(self, dim=dim, keepdim=keepdim)
    
    def cumsum(self, dim: int, *, dtype: str | np.dtype | None = None, out: TN | None = None):
        return api_registry.cumsum(self,dim,dtype=dtype,out=out)
    
    def prod(self, dim:int|tuple|None=None, keepdim:bool=False):  # type: ignore
        return api_registry.prod(self, dim, keepdim)
    
    def mean(self, dim:int|tuple|None=None, keepdim:bool=False):  # type: ignore
        ret = mean(self,dim,keepdim)
        return ret
    
    def var(self, dim:int|tuple|None=None, unbiased:bool=True, keepdim:bool=False):  # type: ignore
        return api_registry.var(self, dim, unbiased, keepdim)
    
    def std(self, dim:int|tuple|None=None, unbiased:bool=True, keepdim:bool=False):  # type: ignore
        return api_registry.std(self, dim, unbiased, keepdim)

    def abs(self):
        return abs(self)
    
    def pow(self, exponent)->TN:
        return pow(self,exponent)
    
    def sqrt(self):
        return api_registry.sqrt(self)

    def exp(self):
        return api_registry.exp(self)
    
    def log(self):
        return api_registry.log(self)

    def log10(self):
        return api_registry.log10(self)
    
    def log2(self):
        return api_registry.log2(self)
    
    def log1p(self):
        return api_registry.log1p(self)

    def sin(self):
        return api_registry.sin(self)

    def cos(self):
        return api_registry.cos(self)

    def tan(self):
        return api_registry.tan(self)

    def cot(self):
        return api_registry.cot(self)

    def sec(self):
        return api_registry.sec(self)

    def csc(self):
        return api_registry.csc(self)

    def arcsin(self):
        return api_registry.arcsin(self)

    def asin(self):
        return api_registry.asin(self)

    def arccos(self):
        return api_registry.arccos(self)

    def acos(self):
        return api_registry.acos(self)

    def arctan(self):
        return api_registry.arctan(self)

    def atan(self):
        return api_registry.atan(self)

    def sinh(self):
        return api_registry.sinh(self)

    def cosh(self):
        return api_registry.cosh(self)

    def tanh(self):
        return api_registry.tanh(self)

    def coth(self):
        return api_registry.coth(self)

    def sech(self):
        return api_registry.sech(self)

    def csch(self):
        return api_registry.csch(self)

    def arcsinh(self):
        return api_registry.arcsinh(self)

    def asinh(self):
        return api_registry.asinh(self)

    def arccosh(self):
        return api_registry.arccosh(self)

    def acosh(self):
        return api_registry.acosh(self)

    def arctanh(self):
        return api_registry.arctanh(self)

    def atanh(self):
        return api_registry.atanh(self)

    def sign(self):
        return api_registry.sign(self)

    def diagonal(self,offset:int=0,dim1:int=0,dim2:int=1):
        return api_registry.diagonal(self, offset, dim1, dim2)
    
    def diag(self,offset: int = 0):
        return api_registry.diag(self, offset)

    def batch_diag(self):
        return api_registry.batch_diag(self)

    def fill_diagonal(self, value, offset: int = 0, dim1: int = -2, dim2: int = -1):
        return api_registry.fill_diagonal(self, value, offset, dim1, dim2)

    def fill_diagonal_(self, value, offset: int = 0, dim1: int = -2, dim2: int = -1):
        return api_registry.fill_diagonal(self, value, offset, dim1, dim2)

    def tril(self, diagonal: int = 0):
        return api_registry.tril(self, diagonal)
    
    def triu(self, diagonal: int = 0):
        return api_registry.triu(self, diagonal)

    def inverse(self):
        return api_registry.inv(self)

    def pinverse(self):
        return api_registry.pinv(self)

    def det(self):
        return api_registry.det(self)

    def norm(self, ord:int|float|str|None = "fro", dim: int|tuple|None = None, keepdim: bool = False):  # type: ignore
        return api_registry.norm(self, ord=ord, dim=dim, keepdim=keepdim)

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
        - 由_BackwardHookManager统一管理钩子调用时机和状态

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

            # _init_calc_graph 返回是否需要钩子处理
            need_hook_processing = _init_calc_graph(self)
            self._grad_value = tensor(1.0, dtype=self.dtype, device=self.device,
                                    requires_grad=create_graph)

        elif isinstance(gradient, TN):
            if gradient._array.shape == self._array.shape:
                need_hook_processing = _init_calc_graph(self)
                self._grad_value = gradient.to(self.device).detach().requires_grad_(create_graph)
            else:
                raise RuntimeError('shape of gradient need to be same as the shape of outputs')
        else:
            raise TypeError(f'gradient can be either tensor or None, but got {type(gradient)}')

        # 执行反向传播
        with set_grad_enabled(create_graph):
            if need_hook_processing:
                _backward_with_hooks([self])
            else:
                _backward_without_hooks([self])
    
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
                 (未提供 grad_outputs 时 autograd.grad 要求每个输出根是标量，
                  所以必须逐元素)
            3. reshape 回 shape = ⊕ shape(vi) × ki
        """
        
        return api_registry.diff(self, *args, create_graph=create_graph)
# end of class TN

def _init_calc_graph(outputs):
    """从多个反向根初始化计算图缓存，并探测是否需要钩子处理。

    供 TN.backward、autograd.backward、autograd.grad 共用的图初始化入口，
    完成两件事（均为原地修改，不返回计算结果）：

    1. 状态重置：从 outputs 可达的所有节点的 _grad_value 置 None；
    2. 计数重建：把 _rcv_grad_count 先清零，再按图中的边重新累加。

    _rcv_grad_count 的语义是“该节点还需要接收多少份上游梯度”，即它在
    本次可达子图中被多少个可微后继节点当作 fromvars 引用。计数为 0 的
    节点（含叶子与未被其它根引用的根）才可以立即开始反向传播。

    多根时，某个根若同时是另一个根的前驱，它会被重新计数（>0），
    调用方不应立即把它入栈，需要等上游根的梯度传播到达后才入栈。

    本函数只负责图状态初始化，不处理 .grad 累积范围和 inputs 过滤，
    这些语义由调用方维护。

    来源与入栈约定（与 _propagate_grad_to_sources 保持一致）:
        - fromvars 可能混有非 TN 操作数（如 add(x, 2.0)）；反向图只沿
          requires_grad=True 的 TN 来源建边和计数，非 TN 条目直接跳过。
        - requires_grad=False 的节点不会记录 fromvars，故“fromvars 全为非 TN”
          的节点不存在，空 fromvars 等价于叶子。
        - 反向栈只含 TN：根由调用方保证是 TN，传播侧只把 TN 且 requires_grad=True
          的来源入栈。

    参数:
        outputs: 单个 TN 或非空 TN 序列，反向传播的起点（根）。
                 内部函数，调用方保证元素均为 TN。

    返回:
        bool: 计算图中是否存在需要钩子处理的模块。
    """

    # 用 id 而不是 TN 本身作为 visited 元素：TN.__eq__ 是逐元素比较，
    # 不能用于集合成员判断；用 id 还能保证同一节点只被重置/入栈一次。
    visited = set()

    if isinstance(outputs, TN):
        # 单根也先标记为已访问并清零计数：根自身不会被当作别人的前驱
        # 重复入栈，后面的 while 循环负责把它的 _grad_value 置 None。
        visited.add(id(outputs))
        stack = [outputs]
        outputs._rcv_grad_count = 0
    else:
        # 多根：按 id 去重后再入栈，避免同一个根被重复初始化。
        stack = []
        for root in outputs:
            if id(root) not in visited:
                visited.add(id(root))
                root._rcv_grad_count = 0
                stack.append(root)

    need_hook_processing = False
    while stack:
        item = stack.pop()

        # 每个节点只会入栈一次（入栈前已登记 visited），所以这里只重置一次，
        # 不会覆盖反向传播过程中新写入的 _grad_value。
        item._grad_value = None

        # 钩子探测：只看节点是否由某个带反向钩子的模块产生。
        # 短路逻辑：一旦置 True 就跳过后续所有检查。
        if not need_hook_processing and item._output_of is not None:
            module = item._output_of
            if module._backward_pre_hooks or module._backward_hooks:
                need_hook_processing = True

        # 叶子节点没有 fromvars，不必再向更上游遍历。
        if item.is_leaf:
            continue

        for var in item.fromvars:
            # 只有 requires_grad 的 TN 前驱参与反向图，常量/普通张量都被忽略。
            if isinstance(var, TN) and var.requires_grad:
                if id(var) not in visited:
                    # 首次访问该前驱：清零计数并入栈，等待后续处理。
                    visited.add(id(var))
                    var._rcv_grad_count = 0
                    stack.append(var)
                # 每条来自不同后继的可微边，都要累加一次接收计数。
                var._rcv_grad_count += 1

    return need_hook_processing


def _clear_calc_graph_cache(outputs) -> None:
    """清理从多个反向根可达节点中的临时梯度状态。

    把所有可达节点的 _grad_value 置 None、_rcv_grad_count 置 0。
    与 _init_calc_graph 的区别：
      - _init_calc_graph 会在清零后重新累加 _rcv_grad_count（得到后续可用的
        真实计数），并检查计算图是否需要钩子处理；
      - _clear_calc_graph_cache 只做清零，不重新累加、不检查钩子，用于丢弃
        上一次反向传播留下的临时状态。

    grad 与 backward 遍历结束后不会主动调用本函数：下次图初始化时
    _init_calc_graph 会覆盖所有可达节点的状态。此函数作为备用清理工具保留。

    遍历范围与 _init_calc_graph 一致：只沿 requires_grad=True 的 TN 来源，
    fromvars 中的非 TN 条目直接跳过。

    参数:
        outputs: 单个 TN 或 TN 序列，清理范围的起点（根）。
    """
    # 用 id 作为 visited 元素：TN.__eq__ 是逐元素比较，不能用于集合成员判断。
    visited = set()

    if isinstance(outputs, TN):
        visited.add(id(outputs))
        stack = [outputs]
        outputs._rcv_grad_count = 0
    else:
        # 多根：按 id 去重后再入栈，避免同一个根被重复清理。
        stack = []
        for root in outputs:
            if id(root) not in visited:
                visited.add(id(root))
                root._rcv_grad_count = 0
                stack.append(root)

    while stack:
        item = stack.pop(-1)

        # 出栈即清空临时梯度。每个节点只会入栈一次（入栈前已登记 visited），
        # 因此根节点与普通节点都能被清理到。
        item._grad_value = None

        # 叶子节点没有 fromvars，不必再向更上游遍历。
        if item.is_leaf:
            continue

        for var in item.fromvars:
            if isinstance(var, TN) and var.requires_grad and id(var) not in visited:
                visited.add(id(var))
                var._rcv_grad_count = 0
                stack.append(var)

def _save_grad(item: TN, input_ids=None) -> bool:
    """将 item 的 _grad_value 保存到 item.grad 中，返回是否为叶子节点。

    item 是 _backward_with_hooks 或 _backward_without_hooks 的出栈节点，
    已保证 requires_grad=True，且 _grad_value 已收集完成。

    input_ids 为 None 时：叶子节点总是保存，中间节点仅在
    retains_grad=True 时保存。
    input_ids 非 None 时：只向集合中的张量累积 .grad，中间节点
    不受 retains_grad 限制。

    返回是否为叶子节点：叶子没有来源，调用方据此清空 _grad_value 并跳过
    后续传播。

    is_leaf 依据“fromvars 为空”判定：fromvars 记录影响前向值的 TN 来源
    （可能含 requires_grad=False 的实参），非空即说明存在上游 TN，因此空
    fromvars 等价于叶子。
    """
    is_leaf = not item.fromvars

    if input_ids is None:
        should_save = is_leaf or item.retains_grad
    else:
        should_save = id(item) in input_ids

    if should_save:
        if item.grad is None:
            item.grad = item._grad_value
        else:
            # 如已有梯度值，累计梯度，注意不要使用原地 +=
            item.grad = item.grad + item._grad_value

    return is_leaf

def _propagate_grad_to_sources(item: TN, vars_received_grad: set) -> None:
    """将当前节点的梯度传播到其所有来源节点。

    反向传播核心步骤，遍历当前节点的所有来源节点，计算并传播梯度。
    直接修改 vars_received_grad 集合，添加已收集完梯度的节点。

    约定：
        - rule(item) 必须返回与 item.fromvars 等长、按位置对齐的元组；非 TN 或
          requires_grad=False 的来源用 None 占位。这里用 zip 逐位配对，长度不一致
          会被静默截断，因此长度必须严格一致。
        - 只有 TN 且 requires_grad=True 的来源会被累加梯度并计入
          vars_received_grad（即反向栈只含 TN）；fromvars 中的非 TN 条目即使
          返回了梯度也不会入栈。
        - 这里的过滤条件与 _init_calc_graph 的计数条件一致，保证每节点只在其
          _rcv_grad_count 归零后入栈一次。

    参数：
        item: 当前完成梯度收集的节点
        vars_received_grad: 已收集完梯度、可入栈的源变量集合（会被直接修改）
    """
    fromvars = item.fromvars

    # 梯度规则一次调用返回与 fromvars 等长且按位置对齐的梯度元组
    rule = item._grad_rule
    if rule is None:
        return
    
    for var, grad in zip(fromvars, rule(item)):
        if grad is not None and isinstance(var, TN) and var.requires_grad:
            # 确保 var 与 grad 的 dtype 一致：grad 计算过程中精度可能提升
            # （float32 -> float64），累加前先转回 var 的 dtype。
            if grad.dtype != var.dtype:
                warnings.warn(f'dtype mismatch: node dtype={var.dtype}, incoming grad dtype={grad.dtype}')
                grad = grad.type(var.dtype)
            
            # 首次直接赋值；已有梯度则累计。多个节点可能共享同一个 _grad_value，
            # 不能用原地 += 累加。
            var._grad_value = grad if var._grad_value is None else var._grad_value + grad
            var._rcv_grad_count -= 1
            if var._rcv_grad_count == 0:
                # 所有梯度已收集，放入 vars_received_grad 集合，准备入栈
                vars_received_grad.add(var)


class _BackwardHookManager:
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
            tensor: 刚出栈的节点（_grad_value已就绪，且已在_popped_tensors中）
            
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
            if tensor._grad_value is not None:
                cache_entry['grad_clone'] = tensor._grad_value.clone()
                
                # 多输出模块：缓存同组其他输出的梯度
                multi_group = cache_entry.get('multi_group')
                if multi_group is not None:
                    for other_output in multi_group:
                        if other_output is tensor:
                            continue
                        if other_output._grad_value is not None:
                            other_cache = module._backward_hook_cache.get(id(other_output))
                            if other_cache is not None and other_cache.get('grad_clone') is None:
                                other_cache['grad_clone'] = other_output._grad_value.clone()
            
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
        检查过就绪状态（基于 _rcv_grad_count），此处直接使用，无需重复检查。
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
                        # 未参与损失计算的输出：_rcv_grad_count == 0（从未收到梯度或已收集完成）
                        for out in multi_outputs:
                            if out.requires_grad and id(out) not in self._popped_tensors and out._rcv_grad_count > 0:
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
        - 单输出模块：该输出节点的 _rcv_grad_count == 0（所有上游梯度已收集完成）
        - 多输出模块：该组内所有需要梯度的输出节点的 _rcv_grad_count == 0
        
        关键设计点：
        1. 使用 _rcv_grad_count 而非 _grad_value 检测就绪状态
           - _rcv_grad_count == 0 表示所有上游梯度已收集完成
           - _grad_value 可能在收集完成后被清空，不可靠
           - 对于不参与损失计算的输出（_grad_value=None），_rcv_grad_count 仍为 0
        
        2. 多输出模块的特殊处理
           - 多输出必须作为一个组一起处理（与 PyTorch 行为一致）
           - 必须等待组内所有需要梯度的输出都就绪才能调用钩子
        
        3. 部分输出不参与损失计算的场景
           - 不参与损失计算的输出：_rcv_grad_count == 0, _grad_value == None
           - 参与损失计算的输出：_rcv_grad_count == 0, _grad_value != None
           - 只要组内所有需要梯度的输出 _rcv_grad_count == 0，就认为组已就绪
        
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
                # _rcv_grad_count == 0 表示所有上游梯度已收集完成
                if output_tensor._rcv_grad_count == 0:
                    return True
            else:
                # 多输出：按组检查
                group_id = id(multi_group)
                if group_id in processed_groups:
                    continue
                processed_groups.add(group_id)
                
                # 检查组内所有需要梯度的输出是否就绪
                # 注意：不参与损失计算的输出 _rcv_grad_count 也为 0
                group_ready = all(
                    out._rcv_grad_count == 0 
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
                # 通过 _rcv_grad_count 判断：_rcv_grad_count == 0 表示已就绪（或从未收到梯度）
                # 通过 _popped_tensors 判断：已出栈的节点
                # 条件：需要梯度且 (已出栈 或 从未收到梯度)
                group_ready = all(
                    id(out) in self._popped_tensors or out._rcv_grad_count == 0
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
                    grad = out._grad_value
                grad_outputs.append(grad)
            
            grad_outputs = tuple(grad_outputs)
            
            # 调用钩子链
            current_grad_output, hooks_called = self._call_backward_pre_hooks_chain(module, grad_outputs)
            
            # 处理钩子返回值
            if hooks_called and current_grad_output is not grad_outputs:
                if isinstance(current_grad_output, (tuple, list)):
                    for i, out in enumerate(outputs_requires_grad):
                        if i < len(current_grad_output):
                            out._grad_value = current_grad_output[i] if current_grad_output[i] is not None else zeros_like(out, requires_grad=False)
                else:
                    outputs_requires_grad[0]._grad_value = current_grad_output
            
            # 更新grad_clone供反向钩子使用
            for out in outputs_requires_grad:
                out_cache = module._backward_hook_cache.get(id(out))
                if out_cache is not None:
                    out_cache['grad_clone'] = out._grad_value.clone() if out._grad_value is not None else None
            
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
        1. 所有需要梯度的输入节点的 _rcv_grad_count == 0
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
            if inp._rcv_grad_count > 0:
                return False
            if id(inp) not in self._popped_tensors:
                return False
        
        return True
    
    def _call_backward_hooks(self, module):
        """
        调用模块的反向钩子。
        
        处理流程：
        1. 构建 grad_input（从输入节点的 _grad_value 获取）
        2. 构建 grad_output（从缓存的 grad_clone 获取）
        3. 调用钩子链
        4. 根据返回值更新输入节点的 _grad_value
        
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
                inp._grad_value if inp._grad_value is not None else zeros_like(inp, requires_grad=False)
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
                            inp._grad_value = hook_result[i] if hook_result[i] is not None else zeros_like(inp, requires_grad=False)
                else:
                    # 钩子返回None，使用grad_input_tuple中的梯度
                    for i, inp in enumerate(forward_inputs):
                        if i < len(grad_input_tuple) and grad_input_tuple[i] is not None:
                            inp._grad_value = grad_input_tuple[i]
            
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
            tensor: 刚出栈的节点（_grad_value已就绪，且已在_popped_tensors中）
            
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

def _backward_with_hooks(stack, input_ids=None) -> None:
    """从多个反向根执行带钩子的反向传播。

    在标准遍历的基础上，于节点出栈时统一处理模块的反向预处理钩子和
    反向钩子，并在结束后清理钩子缓存。

    参数:
        stack: 已就绪、待处理的根节点列表；函数会消费该列表。
        input_ids: 允许累积 .grad 的张量 id 集合；None 表示不限制。
    """
    hook_manager = _BackwardHookManager()

    while stack:
        item = stack.pop(-1)

        # 在反向传播前统一检查和处理两种钩子
        items_for_propagate = hook_manager.process_module_hooks(item)

        # 在调用钩子之后保存梯度，确保钩子可以访问所有输入的梯度grad_value
        # 叶子节点保存后无须继续传播，清空梯度缓存并快速进入下一次循环
        if _save_grad(item, input_ids):
            item._grad_value = None
            continue

        # 如果没有可传播节点（模块输入或输出梯度未收集齐），跳过本次循环
        if not items_for_propagate:
            continue

        # 处理需要钩子的节点
        vars_received_grad = set()
        for current_item in items_for_propagate:
            _propagate_grad_to_sources(current_item, vars_received_grad)
            # 传播完梯度的节点，使命完成，清空grad_value节省内存
            current_item._grad_value = None
        stack.extend(vars_received_grad)

    # 清理钩子缓存
    hook_manager.cleanup()


def _backward_without_hooks(stack, input_ids=None) -> None:
    """从多个反向根执行无钩子的反向传播（快速路径）。

    不检查也不调用任何模块钩子，适用于计算图中没有反向钩子的场景。

    参数:
        stack: 已就绪、待处理的根节点列表；函数会消费该列表。
        input_ids: 允许累积 .grad 的张量 id 集合；None 表示不限制。
    """

    while stack:
        item = stack.pop(-1)

        # 保存叶子节点或retains_grad=True的中间节点的梯度，
        # 如item是叶子节点，清空梯度缓存并无须继续传播
        if _save_grad(item, input_ids):
            item._grad_value = None
            continue

        # 传播梯度到来源节点
        vars_received_grad = set()
        _propagate_grad_to_sources(item, vars_received_grad)
        item._grad_value = None

        # 将收到梯度的节点入栈
        stack.extend(vars_received_grad)

def _getitem(t: TN, index):
    """TN.__getitem__ 的实现：执行索引并建立视图与梯度依赖。"""
    # 类型转换：TN索引转为NumPy数组
    index_val = _convert_TNindex_to_numpy(index)

    # 执行索引操作并捕获错误
    try:
        indexed_data = t._array[index_val]
    except (IndexError, ValueError) as e:
        # 错误信息增强
        err_msg = f"Index operation failed: {str(e)} (Original index type: {type(index)}, shape: {t.shape})"
        raise type(e)(err_msg) from None

    # 使用base是否为None来判断是否为视图，这样可以同时支持numpy和cupy
    # 对于numpy和cupy，视图的base属性指向原始数组，而副本的base属性为None
    is_view = indexed_data.base is not None

    # 创建结果张量
    ret = TN(indexed_data, requires_grad=(is_grad_enabled() and t.requires_grad))
    ret._is_view = is_view
    ret._base = t
    ret._view_index = index_val

    if ret.requires_grad or forward_ad_trace_active():
        ret.fromvars = (t,)
        ret.parms = (index_val,)
        ret._grad_rule = _getitem_grad
        ret._jvp_rule = _getitem_jvp

    return ret


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

def _require_tn(x, fn_name):
    """校验一元数学函数的入参必须是 TN（与 torch 一致，不做隐式转换）。"""
    if not isinstance(x, TN):
        raise TypeError(f"{fn_name}(): argument 'input' (position 1) must be Tensor, not {type(x).__name__}")

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
JvpFunc = Callable[..., Optional[TN]]  # 切线函数类型：首参为结果张量，后续为前向输入的 (primal, tangent) 配对，返回结果张量方向的切线
DecoratorFunc = Callable[[ForwardFunc], ForwardFunc]  # 修饰器类型：输入前向函数，返回包装后的前向函数
# 实参展开布局：与 bind(...).arguments 逐位对应；None 表示该形参不产生实参，
# (形参种类, skip) 表示展开方式（skip 为展开 *args 时开头丢弃的个数）
ArgLayout = list[Optional[tuple[Any, int]]]


def _argument_layout(sig: inspect.Signature, drop_first: bool = False) -> ArgLayout:
    """按签名顺序记录每个形参的展开方式，供按实参顺序遍历复用。

    返回与 ``sig.bind(...).arguments`` 逐位对应的列表，元素取值为：
      - None         —— 该形参不产生实参（drop_first=True 时的首参 result）
      - (kind, skip) —— kind 为形参种类；skip 表示展开 *args 时开头要丢弃的个数
    展开规则：VAR_POSITIONAL 逐项、VAR_KEYWORD 逐个值、普通形参 1 项。
    装饰阶段调用一次即可，避免每次前向/传播时反复查询签名。
    """
    layout = []
    for index, param in enumerate(sig.parameters.values()):
        if drop_first and index == 0:
            # 首参是 result：普通形参不参与展开；若 grad_func 写成 *args，
            # 则 result 是 args[0]，需要展开时丢弃
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                layout.append((param.kind, 1))
            else:
                layout.append(None)
        else:
            layout.append((param.kind, 0))
    return layout


def _iter_bound_values(layout: ArgLayout, arguments: Mapping[str, Any]) -> Iterator[Any]:
    """按签名顺序展开已绑定的实参（*args 逐项、**kwargs 逐个值、普通形参 1 项）。

    layout 由 _argument_layout 生成，arguments 为 sig.bind(...).arguments，
    两者都按签名顺序，逐位配对即可。
    """
    for entry, value in zip(layout, arguments.values()):
        if entry is None:
            continue
        kind, skip = entry
        if kind is inspect.Parameter.VAR_POSITIONAL:
            yield from value[skip:]
        elif kind is inspect.Parameter.VAR_KEYWORD:
            yield from value.values()
        else:
            yield value


def _scan_operands(values: Iterable[Any]) -> tuple[Optional[TN], bool]:
    """扫描实参序列，返回 (首个 TN, 是否存在 requires_grad=True 的 TN)。

    首个 TN 用于把非 TN 的返回值转换到同一 device；第二个返回值用于决定
    是否建立反向图。没有 TN 时首个元素为 None。
    """
    first_tn = None
    for value in values:
        if isinstance(value, TN):
            if first_tn is None:
                first_tn = value
            if value.requires_grad:
                return first_tn, True
    return first_tn, False


def register_grad_rule(grad_func:GradFunc)->DecoratorFunc:
    """
    把普通的前向函数包装成可自动微分的函数。

    用法是工厂模式：先用梯度函数 grad_func 生成修饰器，再用它修饰前向函数。
    被修饰后的函数会记录计算图，反向时调用 grad_func 求各输入的梯度。

    Args:
        grad_func (GradFunc): 梯度计算函数，签名与前向函数一致，只是首位多一个 result，
            即 ``grad_func(result, *args, **kwargs)``，result 是前向函数的返回值。
            它返回一个元组，按签名顺序给出“全部 TN 实参”各自的梯度；非 TN 参数不占位。
            requires_grad=False 的 TN 位置不会累计梯度，填 None 即可。

    Returns:
        DecoratorFunc: 用于修饰前向函数的修饰器。

    规则约定：
        - grad_func 返回元组，按签名顺序给出每个 TN 实参的梯度，非 TN 参数不占位；
          不需要梯度的 TN 位置填 None。
        - ``result._grad_value`` 是上游梯度，求各输入梯度时必须使用；``result`` 及其余
          实参也可直接参与计算（如复用前向结果、避免重算中间量）。
        - 非 TN 参数不是图节点；不需要梯度的 TN 仍记录在 fromvars 中（前向图需要
          它的边），但反向时不会累计梯度。
        - 本装饰器会用 grad_func 覆盖前向内部已建立的图，因此 grad_func 必须给出完整
          梯度；若前向结果 requires_grad=True 而所有形参都不需要梯度，会丢弃原图并告警。
        - 可与 register_jvp_rule 同时修饰同一函数，装饰顺序任意。

    Examples:
        >>> # ========== 示例 1：单输入函数 ==========
        >>> # 定义导数函数：d/dx log(x) = 1/x
        >>> def _log_derivative(result: TN, x: TN) -> tuple[TN]:
        ...     return (result._grad_value * (1. / x.conj()),)
        >>>
        >>> @register_grad_rule(_log_derivative)
        ... def mylog(x: TN) -> TN:
        ...     return tensor(np.log(x._array))
        >>>
        >>> x = tensor(2., requires_grad=True)
        >>> y = mylog(x)
        >>> y.backward()
        >>> print(f'x.grad = {x.grad}')  # 输出: x.grad = 0.5

        >>> # ========== 示例 2：多输入函数（关键字参数顺序无关） ==========
        >>> # register_grad_rule 内部通过参数签名建立映射，因此无论关键字参数以何种顺序传入，
        >>> # 梯度都能正确匹配到对应的张量。
        >>> def _mul_derivative(result: TN, x: TN, y: TN) -> tuple[TN, TN]:
        ...     return (result._grad_value * y.conj(), result._grad_value * x.conj())
        >>>
        >>> @register_grad_rule(_mul_derivative)
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
        ...     return (result._grad_value * exponent * (x ** (exponent - 1)).conj(),)
        >>>
        >>> @register_grad_rule(_pow_derivative)
        ... def mypow(x: TN, exponent: float) -> TN:
        ...     return tensor(x._array ** exponent)
        >>>
        >>> x = tensor(2., requires_grad=True)
        >>> y = mypow(x, 3.0)
        >>> y.backward()
        >>> print(f'x.grad = {x.grad}')  # 输出: x.grad = 12.0 (3 * 2^2)

        >>> # ========== 示例 4：部分输入不需要梯度 ==========
        >>> # 不需要梯度的输入仍是一条图边，但反向不会为它累计梯度，其梯度位置填 None。
        >>> def _add_derivative(result: TN, x: TN, y: TN) -> tuple[TN, TN]:
        ...     return (result._grad_value, result._grad_value)
        >>>
        >>> @register_grad_rule(_add_derivative)
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
        >>> # grad_func 通过 result._grad_value 直接完成矩阵乘法，无需求助逐元素乘法。
        >>> def _matmul_derivative(result: TN, x: TN, y: TN):
        ...     g = result._grad_value
        ...     return (g @ y.detach().T, x.detach().T @ g)
        >>>
        >>> @register_grad_rule(_matmul_derivative)
        ... def mymatmul(x: TN, y: TN) -> TN:
        ...     return tensor(x._array @ y._array)
        >>>
        >>> x = tensor([[1., 2.], [3., 4.]], requires_grad=True)
        >>> y = tensor([[5., 6.], [7., 8.]], requires_grad=True)
        >>> z = mymatmul(x, y)
        >>> z.sum().backward()

    工作原理：
        1. 先执行前向计算，再由“梯度模式是否开启、是否有形参 requires_grad”决定
           结果张量是否参与梯度跟踪；
        2. 需要跟踪时，把本次实参按签名顺序展平成 TN 实参序列写入 fromvars
           （含 requires_grad=False 的实参，与内置算子一致：图边由 fromvars 定义，
           反向时按各个来源的 requires_grad 决定是否累计梯度）；
        3. 反向时对该节点调用一次 grad_func，其返回元组与 fromvars 逐位对齐。
    """
    # grad_func 在装饰阶段固定，预先解析签名，避免每次前向调用重复解析
    grad_sig = inspect.signature(grad_func)
    grad_params = grad_sig.parameters
    if not grad_params:
        raise TypeError("grad_func must accept 'result' as its first parameter")
    first_param = next(iter(grad_params.values()))
    if first_param.kind is inspect.Parameter.VAR_KEYWORD:
        raise TypeError("grad_func's first parameter cannot be **kwargs")

    # 实参布局：与 grad_sig.bind(...).arguments（“形参名 → 实参值”的有序映射）
    # 逐位对应，记录每个形参的展开方式，首参 result 不产生实参。
    # 装饰阶段算好一次，前向调用时只做 zip + 展开，不再解析签名。
    # 展开后只有 TN 实参进入 fromvars（其顺序就是 grad_func 返回元组的顺序），
    # 因此 grad_func 的返回值与 fromvars 逐位对齐。
    arg_layout = _argument_layout(grad_sig, drop_first=True)

    def decorator(forward_func:ForwardFunc)->ForwardFunc:
        def _build_grad_rule(xs, kwargs, fromvars):
            """构造该节点的反向规则：反向时调用一次 grad_func。

            grad_func 的返回值与 fromvars 逐位对齐（两者都按签名顺序编号，
            非 TN 实参不占位）。requires_grad=False 的来源不参与反向，其位置
            允许为 None；requires_grad=True 的位置为 None 说明规则不完整，报错。
            """
            def grad_rule(result_tensor: TN):
                grads = grad_func(result_tensor, *xs, **kwargs)
                # 兼容单输入函数直接返回单个梯度（非元组）的写法
                if not isinstance(grads, tuple):
                    grads = (grads,)
                if len(grads) != len(fromvars):
                    raise ValueError(
                        f"grad_func returned {len(grads)} gradients, "
                        f"but the forward function has {len(fromvars)} TN arguments"
                    )
                for index, (var, grad) in enumerate(zip(fromvars, grads)):
                    if grad is None and var.requires_grad:
                        raise ValueError(
                            f"Gradient for TN argument at index {index} is None, "
                            f"but requires_grad is True"
                        )
                return grads
            return grad_rule

        def _collect_fromvars(bound_args):
            """按签名顺序展开本次实参，返回全部 TN 实参组成的 fromvars。

            bound_args 是 grad_sig.bind(...).arguments，即“形参名 → 实参值”的有序
            映射（签名顺序，未传参数已补默认值），因此无需区分位置传参和关键字传参。

            这里与内置算子保持同一约定：fromvars 记录“影响前向值的全部 TN 操作数”
            （含 requires_grad=False 的实参），反向传播再按各来源的 requires_grad
            决定是否累计梯度（见 _propagate_grad_to_sources）。这样前向图（JVP）
            与反向图共用同一份边定义，图遍历无需了解规则实现细节。
            """
            return tuple(v for v in _iter_bound_values(arg_layout, bound_args)
                         if isinstance(v, TN))

        @functools.wraps(forward_func)
        def wrapper(*xs, **kwargs)->TN:
            ret_val = forward_func(*xs, **kwargs)

            grad_on = is_grad_enabled()
            first_tn, has_diff_arg = _scan_operands(itertools.chain(xs, kwargs.values()))
            track = grad_on and has_diff_arg

            if isinstance(ret_val, TN):
                ret = ret_val
                # 前向内部用了可微算子（结果 requires_grad=True），却没有任何形参
                # 需要梯度：下面的重设会丢弃原梯度路径，故发出警告
                if grad_on and ret.requires_grad and not track:
                    warnings.warn(
                        f"register_grad_rule: forward result of "
                        f"'{getattr(forward_func, '__name__', forward_func)}' requires grad, "
                        f"but none of its TN arguments do; the existing gradient path is discarded"
                    )
            else:
                # 前向返回值可能不是 TN（如 numpy 数组），用首个 TN 输入的 device 转换
                ret = as_tensor(ret_val, device=first_tn.device if first_tn is not None else None)

            # 恒等映射（直接返回某个输入节点本身）：输出就是这个输入，它自己的
            # 梯度路径已经存在。若继续写入 fromvars，会形成“节点引用自身”的自环，
            # 并覆盖输入节点原有的规则，因此原样返回。
            if any(value is ret for value in itertools.chain(xs, kwargs.values())):
                return ret

            # 自定义梯度会覆盖前向算子登记的图，旧的来源与规则一并失效
            ret._grad_rule = None
            ret.requires_grad = track

            if track:
                # 关键字实参的书写顺序可能与签名不同，先用 bind 归位成
                # “形参名 → 实参值”的有序映射（同时校验参数、补默认值），
                # 再按签名顺序展开，使 fromvars 与签名一一对应
                bound = grad_sig.bind(ret, *xs, **kwargs)
                bound.apply_defaults()
                ret.fromvars = _collect_fromvars(bound.arguments)
                ret._grad_rule = _build_grad_rule(xs, kwargs, ret.fromvars)
            return ret
        return wrapper
    return decorator


class TangentPair(tuple):
    """jvp_func 实参中 ``(primal, tangent)`` 的标记类型。

    继承自 tuple：可以像普通二元组一样解包（``x, x_t = arg``），且
    ``isinstance(arg, tuple)`` 仍为 True，保持“配对元组”的直观用法。
    它同时提供一个明确的类型，供 ``*args`` 这类变参规则区分“配对的 TN 实参”
    和“本身就是元组的非 TN 实参”（例如 ``dim=(0, 1)``）：后者会原样传入，
    不会被误配对。``tangent`` 为 None 表示该输入方向为零。
    """
    __slots__ = ()

    def __new__(cls, primal: TN, tangent: Optional[TN]) -> "TangentPair":
        return tuple.__new__(cls, (primal, tangent))


def _tangent_value(frame: JVPFrame, value: Any) -> Any:
    """把单个实参准备成 jvp_func 看到的形式：TN → TangentPair(primal, tangent)。

    非 TN 实参原样返回（包括元组、列表等容器），因此 ``dim=(0, 1)`` 这类参数
    不会被误认为配对元组；TN 实参的 tangent 通过 frame 读取，未播种时为 None
    （表示该输入方向为零），规则据此决定是否贡献切线。
    """
    if isinstance(value, TN):
        return TangentPair(value, frame.tangent(value))
    return value


def _rebuild_tangent_call(layout: ArgLayout, arguments: Mapping[str, Any],
                          frame: JVPFrame) -> tuple[list[Any], dict[str, Any]]:
    """按签名顺序重建 jvp_func 的 (args, kwargs)。

    位置形参（含 POSITIONAL_ONLY）与 *args 走位置实参，KEYWORD_ONLY 与
    **kwargs 走关键字实参，保证 jvp_func 的调用形状与原算子一致。
    layout 由 _argument_layout 生成，与 arguments（签名顺序）逐位对应。
    """
    call_args = []
    call_kwargs = {}
    for (name, value), entry in zip(arguments.items(), layout):
        kind = entry[0]
        if kind is inspect.Parameter.VAR_POSITIONAL:
            call_args.extend(_tangent_value(frame, v) for v in value)
        elif kind is inspect.Parameter.VAR_KEYWORD:
            for key, v in value.items():
                call_kwargs[key] = _tangent_value(frame, v)
        elif kind is inspect.Parameter.KEYWORD_ONLY:
            call_kwargs[name] = _tangent_value(frame, value)
        else:
            call_args.append(_tangent_value(frame, value))
    return call_args, call_kwargs


def _make_tangent_rule(jvp_func: JvpFunc, layout: ArgLayout,
                       arguments: Mapping[str, Any]
                       ) -> Callable[[JVPFrame, TN], Optional[TN]]:
    """把用户规则 jvp_func(result, *args, **kwargs) 包装成内部 _jvp_rule。

    用户规则不接触 JVPFrame：传播时按签名顺序重建实参，TN 实参替换为
    TangentPair(primal, tangent)（tangent=None 表示零方向），非 TN 实参原样透传。

    本函数不登记任何图信息：edge 由调用方写入 ret.fromvars，图遍历与传播只认
    fromvars，因此不需要在规则对象上附加私有属性。
    """
    def jvp_rule(frame: JVPFrame, result_tensor: TN) -> Optional[TN]:
        call_args, call_kwargs = _rebuild_tangent_call(layout, arguments, frame)
        return jvp_func(result_tensor, *call_args, **call_kwargs)

    return jvp_rule


def register_jvp_rule(jvp_func: JvpFunc) -> DecoratorFunc:
    """
    装饰器：为算子函数登记前向自动微分（JVP）规则。

    规则约定：
        - jvp_func 的调用形状与原算子一致，首参是输出张量 result：
          ``jvp_func(result, *args, **kwargs) -> TN | None``；
        - 每个 **TN 实参**以 ``TangentPair(primal, tangent)``（tuple 子类，可直接
          解包成两个变量）传入；tangent 为 None 表示该输入方向未播种（零方向）。
          非 TN 实参原样透传，包括 ``dim=(0, 1)`` 这类元组参数；变参规则可用
          ``isinstance(arg, TangentPair)`` 区分“配对实参”与“普通元组实参”；
        - 返回本节点 tangent（None 表示零方向），不要直接写 frame，也不读写
          ``_grad_value``；规则内使用的普通 TN 算子会被继续追踪，因此支持二阶导数；
        - 规则不接触 JVPFrame；位置形参、关键字形参、*args/**kwargs 与默认值
          都按签名顺序自动配对，规则只需按原函数签名解包；
        - 可与 register_grad_rule 同时修饰同一函数，装饰顺序任意。单独使用时
          反向仍由函数体内部的可微算子负责，行为不变。

    Args:
        jvp_func (JvpFunc): 前向规则函数，签名为 jvp_func(result, *args, **kwargs)，
            返回与输出同形的 tangent；零方向返回 None。

    Returns:
        装饰器，接受算子函数并返回登记了 _jvp_rule 的包装函数。

    Examples:
        >>> from riemann import tensor
        >>> from riemann.func import jvp
        >>> from riemann.core import register_grad_rule, register_jvp_rule
        >>>
        >>> def _square_derivative(result, x):
        ...     return (2.0 * x * result._grad_value,)
        >>>
        >>> def _square_jvp(result, x):
        ...     x, x_t = x                    # TN 实参 = (primal, tangent)
        ...     if x_t is not None:           # None 表示该方向为零
        ...         return 2.0 * x * x_t
        >>>
        >>> @register_grad_rule(_square_derivative)      # 反向规则
        ... @register_jvp_rule(_square_jvp)         # 前向规则
        ... def square(x: TN) -> TN:
        ...     return tensor(x._array * x._array)
        >>>
        >>> x = tensor([1.0, 2.0], requires_grad=True)
        >>> y, y_t = jvp(square, x, tensor([0.1, 0.1]))
        >>> y.sum().backward()
    """
    if not callable(jvp_func):
        raise TypeError("register_jvp_rule's first argument must be a callable jvp rule")

    def decorator(forward_func: ForwardFunc) -> ForwardFunc:
        # 装饰阶段解析一次签名；若 forward_func 已被 register_grad_rule 包装，
        # functools.wraps 保证 signature 仍指向原算子函数
        forward_sig = inspect.signature(forward_func)
        # 与 grad_sig 的布局同理：记录每个形参的展开方式，传播时只做 zip + 展开
        arg_layout = _argument_layout(forward_sig)

        @functools.wraps(forward_func)
        def wrapper(*xs: Any, **kwargs: Any) -> TN:
            ret_val = forward_func(*xs, **kwargs)
            trace = forward_ad_trace_active()

            if isinstance(ret_val, TN):
                ret = ret_val
            else:
                # 与 register_grad_rule 一致：非 TN 返回值转换成 TN，并放到首个
                # TN 实参的设备上；没有 TN 实参时用默认设备
                first_tn, _ = _scan_operands(itertools.chain(xs, kwargs.values()))
                ret = as_tensor(ret_val, device=first_tn.device if first_tn is not None else None)

            if ret.requires_grad or trace:
                # 按签名归位实参（补默认值、区分 *args/**kwargs），供规则调用时
                # 重建“与算子调用形状一致”的实参；fromvars 记录本次全部 TN 实参
                # （含 requires_grad=False 的实参），与内置算子约定一致：图边由
                # fromvars 定义，反向传播再按各来源的 requires_grad 过滤。
                bound = forward_sig.bind(*xs, **kwargs)
                bound.apply_defaults()
                arguments = bound.arguments
                sources = tuple(v for v in _iter_bound_values(arg_layout, arguments)
                                if isinstance(v, TN))
                # 恒等映射（直接返回某个输入节点本身）：输出就是该输入，它自己的
                # tangent/梯度路径已经存在。此时登记规则会把节点自身写进
                # fromvars 形成自引用边，使图拓扑排序无法结束，也会覆盖输入节点
                # 原有的规则，因此直接跳过。
                if not sources or any(src is ret for src in sources):
                    return ret
                # 函数体内已有可微算子时 fromvars 可能已由内层节点设置，
                # 此时保留它作为图边；仅当为空时才按实参补齐
                if not ret.fromvars:
                    ret.fromvars = sources
                ret._jvp_rule = _make_tangent_rule(jvp_func, arg_layout, arguments)
            return ret
        return wrapper
    return decorator

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

def _squeeze_grad(result_tensor:TN):
    dim = result_tensor.parms[0]
    return (result_tensor._grad_value.unsqueeze(dim),)

def _unsqueeze_grad(result_tensor:TN):
    dim = result_tensor.parms[0]
    return (result_tensor._grad_value._squeeze(dim),)

def _reshape_grad(result_tensor:TN):
    original_shape, _ = result_tensor.parms[0]
    return (result_tensor._grad_value._reshape(original_shape),)

def _expand_grad(result_tensor:TN):
    """expand操作的融合梯度：把被扩展的维度求和回原始形状。"""
    original_shape, _ = result_tensor.parms[0]
    grad_value = result_tensor._grad_value

    # 如果原始张量是标量，需要将梯度求和为标量
    if original_shape == ():
        return (grad_value.sum(),)

    orig_ndim = len(original_shape)
    grad_ndim = grad_value.ndim

    # 确定需要求和的维度
    sum_dims: list[int] = []

    if orig_ndim < grad_ndim:
        # 对于维度数量增加的情况，前面的维度都需要求和
        sum_dims.extend(range(grad_ndim - orig_ndim))
        # 检查剩余维度
        for dim, (orig_dim, expanded_dim) in enumerate(zip(
                original_shape,
                grad_value.shape[grad_ndim - orig_ndim:])):
            if orig_dim == 1 and expanded_dim > 1:
                sum_dims.append(dim + (grad_ndim - orig_ndim))
    else:
        # 维度数量相同的情况
        for dim, (orig_size, expanded_size) in enumerate(zip(original_shape, grad_value.shape)):
            if orig_size == 1 and expanded_size > 1:
                sum_dims.append(dim)

    # 如果有需要求和的维度，从高到低依次求和，避免维度索引变化问题
    if sum_dims:
        summed_grad = grad_value
        for dim in sorted(sum_dims, reverse=True):
            summed_grad = summed_grad.sum(dim=dim, keepdim=True)
        grad = summed_grad._reshape(original_shape)
    else:
        grad = grad_value._reshape(original_shape)

    return (grad,)

def _flip_grad(result_tensor:TN):
    """flip操作的融合梯度：对上游梯度执行相同的flip操作。"""
    return (flip(result_tensor._grad_value, result_tensor.parms[0]),)

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
    if result.requires_grad or forward_ad_trace_active():
        result.fromvars = (input,)
        result.parms = (dims,)
        result._grad_rule = _flip_grad
        result._jvp_rule = _flip_jvp
    
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

def _sum_grad(result_tensor:TN):
    dim, keepdims = result_tensor.parms[0]
    x_shape = result_tensor.fromvars[0].shape

    # 如果sum计算时结果张量做过维度精简，需要将缩减的维度恢复
    # 然后expand到输入形状，sum的梯度就是直接将输出梯度传播到每个输入元素
    if keepdims == False:
        grad = result_tensor._grad_value.unsqueeze(dim)._expand(x_shape)
    else:
        grad = result_tensor._grad_value._expand(x_shape)

    return (grad,)

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
    
    if ret.requires_grad or forward_ad_trace_active():
        ret.fromvars=(x,)
        ret.parms=((dim,keepdim),)
        ret._grad_rule = _sum_grad
        ret._jvp_rule = _sum_jvp
    
    return ret

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
    if not _same_device(out, result):
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

def _mean_grad(result_tensor:TN):
    dim, keepdim = result_tensor.parms[0]
    x = result_tensor.fromvars[0]

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
        grad = (result_tensor._grad_value / n)._expand(x.shape)
    else:
        # unsqueeze在dim=None时直接返回原张量，无需额外判断
        grad = (result_tensor._grad_value / n).unsqueeze(dim)._expand(x.shape)

    return (grad,)

def mean(x:TN, dim:int|tuple|None=None, keepdim:bool=False)->TN:
    if not isinstance(x,TN):
        raise SyntaxError('x must be a tensor')
    
    if dim == ():
        dim = None
    
    # 根据x的数组类型选择使用np或cp
    value = x._array.mean(axis=dim, keepdims=keepdim)
    ret = TN(value, requires_grad = (is_grad_enabled() and x.requires_grad))
    
    if ret.requires_grad or forward_ad_trace_active():
        ret.fromvars = (x,)
        ret.parms = ((dim,keepdim),)
        ret._grad_rule = _mean_grad
        ret._jvp_rule = _mean_jvp
    
    return ret

def _abs_grad(result_tensor:TN):
    x = result_tensor.fromvars[0]
    # 添加安全检查，避免当result_tensor为0时的除零错误
    # 当result_tensor为0时，梯度应为0
    is_zero = (result_tensor == 0)
    # 使用where操作避免除零，当result_tensor为0时使用1代替
    safe_divisor = where(is_zero, ones_like(result_tensor), result_tensor)
    # 当result_tensor为0时，(x / safe_divisor)将被替换为0
    grad = result_tensor._grad_value * where(is_zero, zeros_like(x), x / safe_divisor)

    return (grad,)

def _abs_jvp(frame, result_tensor: TN):
    """绝对值 JVP：y_t = Re(conj(x)·x_t)/|x|（实数时退化为 sign(x)·x_t）。"""
    x = result_tensor.fromvars[0]
    x_t = frame.tangent(x)
    if x_t is None:
        return None
    # 与 _abs_grad 相同的零点保护：|x| 为零时导数为 0
    is_zero = (result_tensor == 0)
    safe_divisor = where(is_zero, ones_like(result_tensor), result_tensor)
    tangent = (x.conj() * x_t).real / safe_divisor
    return where(is_zero, zeros_like(result_tensor), tangent)


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
    _require_tn(x, 'abs')
    arrlib = x._get_array_lib()
    value = arrlib.abs(x._array)
    ret = TN(value, requires_grad = (is_grad_enabled() and x.requires_grad))
    
    if ret.requires_grad or forward_ad_trace_active():
        ret.fromvars = (x,)
        ret._grad_rule = _abs_grad
        ret._jvp_rule = _abs_jvp
    
    return ret

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

def _max_grad(result_tensor:TN):
    x = result_tensor.fromvars[0]
    dim, keepdim, indices, maxmin_val = result_tensor.parms[0]

    with device_context(x):
        # 使用前向保存的索引/极值，避免重复计算
        max_pos_one_like_x = _create_maxmin_mask(x._array, dim, indices=indices, maxmin_val=maxmin_val)

    # 如果sum计算时结果张量做过维度精简，需要将缩减的维度暂时恢复
    if keepdim == False:
        new_result_grad = result_tensor._grad_value.unsqueeze(dim)
    else:
        new_result_grad = result_tensor._grad_value

    return (new_result_grad * max_pos_one_like_x,)

def _min_grad(result_tensor:TN):
    x = result_tensor.fromvars[0]
    dim, keepdim, indices, maxmin_val = result_tensor.parms[0]

    with device_context(x):
        # 使用前向保存的索引/极值，避免重复计算
        min_pos_one_like_x = _create_maxmin_mask(x._array, dim, indices=indices, maxmin_val=maxmin_val)

    # 如果sum计算时结果张量做过维度精简，需要将缩减的维度暂时恢复
    if keepdim == False:
        new_result_grad = result_tensor._grad_value.unsqueeze(dim)
    else:
        new_result_grad = result_tensor._grad_value

    return (new_result_grad * min_pos_one_like_x,)

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
    if needs_grad or forward_ad_trace_active():
        values_tensor.fromvars = (x,)
        # 保存索引和极值到parms，反向传播时直接复用
        values_tensor.parms = ((dim, keepdim, indices_arr, maxmin_val),)
        values_tensor._grad_rule = _max_grad
        values_tensor._jvp_rule = _max_jvp

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
    if needs_grad or forward_ad_trace_active():
        values_tensor.fromvars = (x,)
        # 保存索引和极值到parms，反向传播时直接复用
        values_tensor.parms = ((dim, keepdim, indices_arr, maxmin_val),)
        values_tensor._grad_rule = _min_grad
        values_tensor._jvp_rule = _min_jvp

    # 根据dim是否为None决定返回类型
    if dim is None:
        return _apply_out(out, values_tensor, (x,), "min")
    else:
        if keepdim and isinstance(dim, int):
            indices_arr = arrlib.expand_dims(indices_arr, axis=dim)
        indices_tensor = TN(indices_arr)
        return _apply_out_tuple(out, ReturnType_min(values_tensor, indices_tensor), (x,), "min")

def _transpose_grad(result_tensor:TN):
    d1, d2 = result_tensor.parms[0]
    return (result_tensor._grad_value.transpose(d1, d2),)

def _permute_grad(result_tensor:TN):
    """permute操作的融合梯度：按逆排列重排上游梯度。"""
    permutation = result_tensor.parms[0]
    # 使用O(n)算法计算逆排列（np.argsort）
    inv_permutation = tuple(np.argsort(permutation))
    # permute方法会自动处理梯度计算图的记录
    return (result_tensor._grad_value.permute(*inv_permutation),)

def _getitem_grad(result_tensor:TN):
    x = result_tensor.fromvars[0]
    index = result_tensor.parms[0]
    out_grad = result_tensor._grad_value

    # 初始化ret_grad与x的shape相同，初始值为0，再按index反向累加梯度
    return (zeros_like(x).addat(index, out_grad),)

def _matmul_grad(result_tensor:TN):
    """计算矩阵乘法 C = A @ B 中左右操作数的梯度，返回值与 fromvars 对齐。

    数学公式：dL/dA = grad @ B^T，dL/dB = A^T @ grad，
    其中 grad = dL/dC 是结果张量的梯度。1D 与多维的各分支处理如下：

    分支1 - 1D @ 1D（向量点积）：逐元素乘法
    分支2 - 1D @ 多维：左梯度用矩阵乘法 + 批次求和；右梯度用 einsum 外积
    分支3 - 多维 @ 1D：左梯度用 einsum 外积；右梯度用矩阵乘法 + 批次求和
    分支4 - 多维 @ 多维：标准矩阵乘法，存在广播时沿广播轴求和后 reshape
    """
    
    left_tensor:TN = result_tensor.fromvars[0]
    right_tensor:TN = result_tensor.fromvars[1]
    left_requires = isinstance(left_tensor, TN) and left_tensor.requires_grad
    right_requires = isinstance(right_tensor, TN) and right_tensor.requires_grad

    if not left_requires and not right_requires:
        return None, None

    # 确保grad_value是连续内存，避免后续@运算符性能下降
    grad_value = result_tensor._grad_value.contiguous()

    left_ndim = left_tensor._array.ndim
    right_ndim = right_tensor._array.ndim

    left_grad = None
    if left_requires:
        if left_ndim == 1 and right_ndim == 1:
            # 分支1：逐元素乘法
            left_grad = grad_value * right_tensor.conj()
        elif left_ndim == 1 and right_ndim > 1:
            # 分支2：grad 插入行维后与 B^T 相乘，再对 A 无批次维引入的轴求和
            new_grad = grad_value.unsqueeze(-2)  # (..., N) -> (..., 1, N)
            mat_grad = new_grad @ right_tensor.mT.conj()  # (..., 1, N) @ (..., N, K) -> (..., 1, K)
            sum_axes = tuple(range(right_ndim - 1))
            left_grad = sum(mat_grad, dim=sum_axes, keepdim=False)
        elif right_ndim == 1:
            # 分支3：einsum 直接计算外积
            left_grad = api_registry.einsum('...m,k->...mk', grad_value, right_tensor.conj())
        else:
            # 分支4：只比较批次部分（[:-2] 剥离矩阵维），找出 A 被广播的批次轴
            broadcast_axes = _get_broadcast_axis(
                result_tensor._array.shape[:-2], left_tensor._array.shape[:-2]
            )
            mat_grad = grad_value @ right_tensor.mT.conj()
            if broadcast_axes:
                # 广播使 A 的一份数据被当成多份参与计算，须沿广播轴求和归还为一份
                left_grad = sum(mat_grad, dim=broadcast_axes, keepdim=False)._reshape(left_tensor.shape)
            else:
                left_grad = mat_grad

    right_grad = None
    if right_requires:
        if left_ndim == 1 and right_ndim == 1:
            # 分支1：逐元素乘法
            right_grad = left_tensor.conj() * grad_value
        elif left_ndim == 1 and right_ndim > 1:
            # 分支2：einsum 直接计算外积，C 的批次即 B 的批次
            right_grad = api_registry.einsum('k,...n->...kn', left_tensor.conj(), grad_value)
        elif right_ndim == 1:
            # 分支3：grad 插入列维后与 A^T 相乘，再对批次维与尾部列维求和
            new_grad = grad_value.unsqueeze(-1)  # (..., M) -> (..., M, 1)
            mat_grad = left_tensor.mT.conj() @ new_grad  # (..., K, M) @ (..., M, 1) -> (..., K, 1)
            sum_axes = tuple(range(left_ndim - 2)) + (-1,)
            right_grad = sum(mat_grad, dim=sum_axes, keepdim=False)
        else:
            # 分支4：只比较批次部分，找出 B 被广播的批次轴
            broadcast_axes = _get_broadcast_axis(
                result_tensor._array.shape[:-2], right_tensor._array.shape[:-2]
            )
            mat_grad = left_tensor.mT.conj() @ grad_value
            if broadcast_axes:
                right_grad = sum(mat_grad, dim=broadcast_axes, keepdim=False)._reshape(right_tensor.shape)
            else:
                right_grad = mat_grad

    return left_grad, right_grad


def _add_jvp(frame, result_tensor: TN):
    """加法（含减法）之外的基础 JVP：y = a + b → y_t = a_t + b_t。"""
    left, right = result_tensor.fromvars
    left_t = frame.tangent(left)
    right_t = frame.tangent(right)
    total = None
    if left_t is not None and right_t is not None:
        # tangent 形状等于各自 primal 形状，相加时的广播结果必然等于输出形状
        total = left_t + right_t
    elif left_t is not None:
        # 单边 tangent 需要显式广播到输出形状（常量侧不提供输出形状）
        total = left_t._expand(result_tensor.shape)
    elif right_t is not None:
        total = right_t._expand(result_tensor.shape)
    return total


def _sub_jvp(frame, result_tensor: TN):
    """减法 JVP：fromvars[0] 为被减数、fromvars[1] 为减数。"""
    left, right = result_tensor.fromvars
    left_t = frame.tangent(left)
    right_t = frame.tangent(right)
    total = None
    if left_t is not None and right_t is not None:
        total = left_t - right_t
    elif left_t is not None:
        total = left_t._expand(result_tensor.shape)
    elif right_t is not None:
        total = -right_t._expand(result_tensor.shape)
    return total


def _mul_jvp(frame, result_tensor: TN):
    """乘法 JVP：y = a · b → y_t = a_t · b + a · b_t。"""
    left, right = result_tensor.fromvars
    left_t = frame.tangent(left)
    right_t = frame.tangent(right)
    total = None
    if left_t is not None:
        total = left_t * right
    if right_t is not None:
        term = left * right_t
        total = term if total is None else total + term
    return total


def _div_jvp(frame, result_tensor: TN):
    """除法 JVP：y = a / b → y_t = a_t / b − a · b_t / b²。"""
    left, right = result_tensor.fromvars
    left_t = frame.tangent(left)
    right_t = frame.tangent(right)
    total = None
    if left_t is not None:
        total = left_t / right
    if right_t is not None:
        # 复用前向输出 y = a/b：∂y/∂b = −y/b，省掉 a/b² 的一次乘法与临时量
        term = (result_tensor * right_t) / right
        total = -term if total is None else total - term
    return total


def _pow_jvp(frame, result_tensor: TN):
    """幂运算 JVP：y = a^b → y_t = b·a^(b−1)·a_t + y·log(a)·b_t。

    指数为非 TN 常量时只剩第一项；指数是 TN 时按完整公式。
    """
    left, right = result_tensor.fromvars
    left_t = frame.tangent(left)
    right_t = frame.tangent(right)
    total = None
    if left_t is not None:
        if isinstance(right, (list, tuple)):
            exponent_minus_one = [e - 1 for e in right]
        else:
            exponent_minus_one = right - 1
        total = (left ** exponent_minus_one) * right * left_t
    if right_t is not None:
        # log 定义在 tensorfuncs，运行时导入避免循环依赖
        from .tensorfuncs import log as _log
        # reversed_order 建图时 fromvars[0] 可能是非 TN 底数常量
        left_value = left
        if not isinstance(left_value, TN):
            left_value = as_tensor(left_value, dtype=result_tensor.dtype,
                                   device=result_tensor.device)
        term = result_tensor * _log(left_value) * right_t
        total = term if total is None else total + term
    return total


def _mod_jvp(frame, result_tensor: TN):
    """取余 JVP：y = a % b → y_t = a_t − floor(a / b)·b_t（与 _mod_grad 一致）。"""
    left, right = result_tensor.fromvars
    left_t = frame.tangent(left)
    right_t = frame.tangent(right)
    total = None
    if left_t is not None:
        total = left_t
    if right_t is not None:
        # reversed_order 建图时 fromvars[0] 可能是非 TN 被除数常量
        left_value = left
        if not isinstance(left_value, TN):
            left_value = as_tensor(left_value, dtype=result_tensor.dtype,
                                   device=result_tensor.device)
        term = floor(left_value / right) * right_t
        total = -term if total is None else total - term
    return total


def _clone_jvp(frame, result_tensor: TN):
    """克隆 JVP：y = clone(x) → y_t = clone(x_t)。"""
    tangent = frame.tangent(result_tensor.fromvars[0])
    if tangent is None:
        return None
    return tangent.clone()


def _contiguous_jvp(frame, result_tensor: TN):
    """contiguous JVP：y 与 x 数值相同 → y_t = x_t.contiguous()。"""
    tangent = frame.tangent(result_tensor.fromvars[0])
    if tangent is None:
        return None
    return tangent.contiguous()


def _to_jvp(frame, result_tensor: TN):
    """类型/设备转换 JVP：y = x.to(...) → y_t = x_t.to(与 y 相同的 dtype/device)。"""
    tangent = frame.tangent(result_tensor.fromvars[0])
    if tangent is None:
        return None
    return tangent.to(dtype=result_tensor.dtype, device=result_tensor.device)


def _neg_jvp(frame, result_tensor: TN):
    """取负 JVP：y = -x → y_t = -x_t。"""
    tangent = frame.tangent(result_tensor.fromvars[0])
    if tangent is None:
        return None
    return -tangent


# ---------------------------------------------------------------------------
# 步骤 2：形状变换 / 归约 / 索引选择 / 复杂数视图算子的直接前向规则
#
# 这些算子都是“结构搬运”或在前向值上可解析求导的算子：规则只把入边 tangent
# 做同样的线性搬运（转置/重排/切片/归约），不构造反向图，也不依赖 requires_grad。
# ---------------------------------------------------------------------------

def _transpose_jvp(frame, result_tensor: TN):
    """转置 JVP：y = x.transpose(d1, d2) → y_t = x_t.transpose(d1, d2)。"""
    dim1, dim2 = result_tensor.parms[0]
    tangent = frame.tangent(result_tensor.fromvars[0])
    if tangent is None:
        return None
    return tangent.transpose(dim1, dim2)


def _permute_jvp(frame, result_tensor: TN):
    """维度重排 JVP：y = x.permute(*dims) → y_t = x_t.permute(*dims)。"""
    dims = result_tensor.parms[0]
    tangent = frame.tangent(result_tensor.fromvars[0])
    if tangent is None:
        return None
    return tangent.permute(*dims)


def _reshape_jvp(frame, result_tensor: TN):
    """reshape JVP：y = x.reshape(new_shape) → y_t = x_t.reshape(new_shape)。"""
    new_shape = result_tensor.parms[0][1]
    tangent = frame.tangent(result_tensor.fromvars[0])
    if tangent is None:
        return None
    return tangent.reshape(*new_shape)


def _expand_jvp(frame, result_tensor: TN):
    """expand JVP：y = x.expand(new_shape) → y_t = x_t.expand(new_shape)。

    广播是把同一元素复制到多个输出位置，其切向随之前向复制；
    反向的“沿广播轴求和”只属于 VJP，前向不需要。
    """
    new_shape = result_tensor.parms[0][1]
    tangent = frame.tangent(result_tensor.fromvars[0])
    if tangent is None:
        return None
    return tangent._expand(new_shape)


def _squeeze_jvp(frame, result_tensor: TN):
    """squeeze JVP：y = x.squeeze(dims) → y_t = x_t.squeeze(dims)。"""
    squeezed_dims = result_tensor.parms[0]
    tangent = frame.tangent(result_tensor.fromvars[0])
    if tangent is None:
        return None
    return tangent._squeeze(squeezed_dims)


def _unsqueeze_jvp(frame, result_tensor: TN):
    """unsqueeze JVP：y = x.unsqueeze(dim) → y_t = x_t.unsqueeze(dim)。"""
    dim = result_tensor.parms[0]
    tangent = frame.tangent(result_tensor.fromvars[0])
    if tangent is None:
        return None
    return tangent.unsqueeze(dim)


def _getitem_jvp(frame, result_tensor: TN):
    """索引选择 JVP：对 tangent 施加完全相同的索引。"""
    index = result_tensor.parms[0]
    tangent = frame.tangent(result_tensor.fromvars[0])
    if tangent is None:
        return None
    return tangent[index]


def _flip_jvp(frame, result_tensor: TN):
    """翻转 JVP：y = flip(x, dims) → y_t = flip(x_t, dims)。"""
    dims = result_tensor.parms[0]
    tangent = frame.tangent(result_tensor.fromvars[0])
    if tangent is None:
        return None
    return flip(tangent, dims)


def _sum_jvp(frame, result_tensor: TN):
    """求和 JVP：y = sum(x, dim) → y_t = sum(x_t, dim)。"""
    dim, keepdim = result_tensor.parms[0]
    tangent = frame.tangent(result_tensor.fromvars[0])
    if tangent is None:
        return None
    return sum(tangent, dim=dim, keepdim=keepdim)


def _mean_jvp(frame, result_tensor: TN):
    """均值 JVP：y = mean(x, dim) → y_t = mean(x_t, dim)。"""
    dim, keepdim = result_tensor.parms[0]
    tangent = frame.tangent(result_tensor.fromvars[0])
    if tangent is None:
        return None
    return mean(tangent, dim=dim, keepdim=keepdim)


def _extreme_jvp(frame, result_tensor: TN):
    """max/min 共用 JVP：只让取到极值位置的 tangent 通过（并列时全部计入）。

    前向极值的选择是局部常数映射，因此 JVP 等价于“按选择掩码筛选再归约”。
    掩码用前向值比较得到，避免依赖 parms 中保存的 argmax 展平索引。
    """
    x = result_tensor.fromvars[0]
    tangent = frame.tangent(x)
    if tangent is None:
        return None
    dim, keepdim = result_tensor.parms[0][0], result_tensor.parms[0][1]
    selected_value = result_tensor if (dim is None or keepdim) else result_tensor.unsqueeze(dim)
    mask = where(x == selected_value, ones_like(tangent), zeros_like(tangent))
    return sum(tangent * mask, dim=dim, keepdim=keepdim)


def _max_jvp(frame, result_tensor: TN):
    return _extreme_jvp(frame, result_tensor)


def _min_jvp(frame, result_tensor: TN):
    return _extreme_jvp(frame, result_tensor)


def _where_jvp(frame, result_tensor: TN):
    """where JVP：y = where(cond, a, b) → y_t = where(cond, a_t, b_t)。

    cond 为不参与微分的静态参数；缺失的一侧按零 tangent 处理，并广播到输出形状。
    """
    cond = result_tensor.parms[0]
    left, right = result_tensor.fromvars
    left_t = frame.tangent(left)
    right_t = frame.tangent(right)
    if left_t is None and right_t is None:
        return None
    zero = zeros_like(result_tensor)
    # where 自身按 NumPy 规则广播；cond 或 zeros_like 已提供输出形状
    left_t = zero if left_t is None else left_t
    right_t = zero if right_t is None else right_t
    return where(cond, left_t, right_t)


def _matmul_jvp(frame, result_tensor: TN):
    """矩阵乘 JVP：y = A @ B → y_t = A_t @ B + A @ B_t（双线性）。"""
    left, right = result_tensor.fromvars
    left_t = frame.tangent(left)
    right_t = frame.tangent(right)
    total = None
    if left_t is not None:
        total = left_t @ right
    if right_t is not None:
        term = left @ right_t
        total = term if total is None else total + term
    return total


def _real_jvp(frame, result_tensor: TN):
    """实部 JVP：y = Re(x) → y_t = Re(x_t)。"""
    tangent = frame.tangent(result_tensor.fromvars[0])
    if tangent is None:
        return None
    return tangent.real


def _imag_jvp(frame, result_tensor: TN):
    """虚部 JVP：y = Im(x) → y_t = Im(x_t)。"""
    tangent = frame.tangent(result_tensor.fromvars[0])
    if tangent is None:
        return None
    return tangent.imag


def _conj_jvp(frame, result_tensor: TN):
    """共轭 JVP：y = conj(x) → y_t = conj(x_t)（前向不取共轭梯度约定）。"""
    tangent = frame.tangent(result_tensor.fromvars[0])
    if tangent is None:
        return None
    return tangent.conj()


def _add_grad(result_tensor:TN):
    """一次计算加法的左右梯度，返回值与fromvars的两个位置对齐。"""
    left, right = result_tensor.fromvars
    left_requires = isinstance(left, TN) and left.requires_grad
    right_requires = isinstance(right, TN) and right.requires_grad
    grad_value = result_tensor._grad_value
    result_shape = result_tensor.shape
    left_grad = None
    right_grad = None

    if left_requires:
        if left.shape == result_shape:
            left_grad = grad_value
        else:
            broadcast_axes = _get_broadcast_axis(result_shape, left.shape)
            left_grad = sum(grad_value, dim=broadcast_axes, keepdim=False)._reshape(left.shape)

    if right_requires:
        # 两侧形状相同时广播归约结果完全相同，可直接复用
        if left_requires and right.shape == left.shape:
            right_grad = left_grad
        elif right.shape == result_shape:
            right_grad = grad_value
        else:
            broadcast_axes = _get_broadcast_axis(result_shape, right.shape)
            right_grad = sum(grad_value, dim=broadcast_axes, keepdim=False)._reshape(right.shape)

    return left_grad, right_grad


def _sub_grad(result_tensor:TN):
    """一次计算减法的左右梯度，右值梯度取加法对应结果的负值。"""
    left_grad, right_grad = _add_grad(result_tensor)
    return left_grad, None if right_grad is None else -right_grad


def _mul_grad(result_tensor:TN):
    """一次计算乘法的左右梯度，返回值与fromvars的两个位置对齐。"""
    left, right = result_tensor.fromvars
    left_requires = isinstance(left, TN) and left.requires_grad
    right_requires = isinstance(right, TN) and right.requires_grad
    grad_value = result_tensor._grad_value
    result_shape = result_tensor.shape

    # x*x的两个来源是同一对象，两侧原始梯度及广播归约完全一致
    if left_requires and right_requires and left is right:
        grad = grad_value * conj(left)
        if left.shape != result_shape:
            broadcast_axes = _get_broadcast_axis(result_shape, left.shape)
            grad = sum(grad, dim=broadcast_axes, keepdim=False)._reshape(left.shape)
        return grad, grad

    left_grad = None
    right_grad = None
    if left_requires:
        left_grad = grad_value * conj(right)
        if left.shape != result_shape:
            broadcast_axes = _get_broadcast_axis(result_shape, left.shape)
            left_grad = sum(left_grad, dim=broadcast_axes, keepdim=False)._reshape(left.shape)

    if right_requires:
        right_grad = grad_value * conj(left)
        if right.shape != result_shape:
            broadcast_axes = _get_broadcast_axis(result_shape, right.shape)
            right_grad = sum(right_grad, dim=broadcast_axes, keepdim=False)._reshape(right.shape)

    return left_grad, right_grad


def _div_grad(result_tensor:TN):
    """一次计算除法的左右梯度，并复用右操作数的共轭。"""
    left, right = result_tensor.fromvars
    left_requires = isinstance(left, TN) and left.requires_grad
    right_requires = isinstance(right, TN) and right.requires_grad
    grad_value = result_tensor._grad_value
    result_shape = result_tensor.shape
    right_conj = conj(right) if left_requires or right_requires else None
    left_grad = None
    right_grad = None

    if left_requires:
        left_grad = grad_value / right_conj
        if left.shape != result_shape:
            broadcast_axes = _get_broadcast_axis(result_shape, left.shape)
            left_grad = sum(left_grad, dim=broadcast_axes, keepdim=False)._reshape(left.shape)

    if right_requires:
        # g * (-result/right).conj() == -g * result.conj() / right.conj()
        right_grad = -grad_value * result_tensor.conj() / right_conj
        if right.shape != result_shape:
            broadcast_axes = _get_broadcast_axis(result_shape, right.shape)
            right_grad = sum(right_grad, dim=broadcast_axes, keepdim=False)._reshape(right.shape)

    return left_grad, right_grad


def _floordiv_jvp(frame, result_tensor:TN):
    """向下整除的前向导数未定义：与反向及 PyTorch 一致，直接报错。"""
    raise RuntimeError('derivative for aten::floor_divide is not implemented')


def _floordiv_grad(result_tensor:TN):
    left_var = result_tensor.fromvars[0]
    right_var = result_tensor.fromvars[1]
    # 向下取整除法的导数几乎处处为0，PyTorch 未实现其反向传播，这里保持一致；
    # 仅当某个来源需要梯度时才报错，与 PyTorch 的可观察行为一致
    if ((isinstance(left_var, TN) and left_var.requires_grad)
            or (isinstance(right_var, TN) and right_var.requires_grad)):
        raise RuntimeError('derivative for aten::floor_divide is not implemented')
    return (None, None)


def _mod_grad(result_tensor:TN):
    left_var = result_tensor.fromvars[0]
    right_var = result_tensor.fromvars[1]
    left_requires = isinstance(left_var, TN) and left_var.requires_grad
    right_requires = isinstance(right_var, TN) and right_var.requires_grad
    result_shape = result_tensor.shape

    left_grad = None
    if left_requires:
        # d(left % right)/d(left) = 1，结果梯度即为左值梯度
        if left_var.shape == result_shape:
            left_grad = result_tensor._grad_value
        else:
            broadcast_axes = _get_broadcast_axis(result_shape, left_var.shape)
            left_grad = sum(
                result_tensor._grad_value, dim=broadcast_axes, keepdim=False
            )._reshape(left_var.shape)

    right_grad = None
    if right_requires:
        left_value = left_var
        if not isinstance(left_value, TN):
            left_value = as_tensor(
                left_value,
                dtype=result_tensor.dtype,
                device=result_tensor.device,
            )
        # d(left % right)/d(right) = -floor(left/right)
        right_grad = -result_tensor._grad_value * floor(left_value / right_var)
        if right_var.shape != result_shape:
            broadcast_axes = _get_broadcast_axis(result_shape, right_var.shape)
            right_grad = sum(
                right_grad, dim=broadcast_axes, keepdim=False
            )._reshape(right_var.shape)

    return left_grad, right_grad

def _pow_grad(result_tensor:TN):
    """一次计算幂运算的左右梯度，非TN值仅在实际求导时转换。"""
    left, right = result_tensor.fromvars
    left_requires = isinstance(left, TN) and left.requires_grad
    right_requires = isinstance(right, TN) and right.requires_grad
    grad_value = result_tensor._grad_value
    result_shape = result_tensor.shape
    left_grad = None
    right_grad = None

    if left_requires:
        right_value = right
        if isinstance(right, (list, tuple, range)):
            xp = cp if result_tensor.device.type == 'cuda' else np
            right_value = xp.array(right, dtype=result_tensor.dtype)
        derivative = (left ** (right_value - 1.0)) * right_value
        left_grad = grad_value * derivative.conj()
        if left.shape != result_shape:
            broadcast_axes = _get_broadcast_axis(result_shape, left.shape)
            left_grad = sum(left_grad, dim=broadcast_axes, keepdim=False)._reshape(left.shape)

    if right_requires:
        left_value = left
        if not isinstance(left_value, TN):
            left_value = as_tensor(
                left_value,
                dtype=result_tensor.dtype,
                device=result_tensor.device,
            )
        right_grad = grad_value * (result_tensor * left_value.log()).conj()
        if right.shape != result_shape:
            broadcast_axes = _get_broadcast_axis(result_shape, right.shape)
            right_grad = sum(right_grad, dim=broadcast_axes, keepdim=False)._reshape(right.shape)

    return left_grad, right_grad


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
        if not _same_device(left, right):
            raise RuntimeError(f'Expected all tensors to be on the same device, but found at least two devices, {left.device} and {right.device}!')
        right_data = right._array
        right_requires_grad = right.requires_grad
    else:
        right_data = right
        right_requires_grad = False

    #requires_grad属性在运算时传递到结果tensor
    #图记录与requires_grad解耦：反向请求(requires_grad)或前向请求(JVP frame)任一成立都记录图；
    #requires_grad 只表达反向请求，前向建图不修改它（设计文档 §14.6）
    backward_requested = (is_grad_enabled() and (left.requires_grad or right_requires_grad))
    record_graph = backward_requested or forward_ad_trace_active()
    if reversed_order:
        # 保持原始顺序 right ** left（list/tuple/range 交给数组的 __rpow__ 逐元素计算）
        data = right_data ** left._array
    else:
        data = left._array ** right_data
    ret = TN(data, requires_grad=backward_requested)
    
    if record_graph:
        ret.fromvars = (right, left) if reversed_order else (left, right)
        ret._grad_rule = _pow_grad
        ret._jvp_rule = _pow_jvp

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
        reversed_order = False
    elif isinstance(y, TN):
        left, right = y, x
        reversed_order = True
    else:
        # 左右都不是TN：左值包装成TN后沿用同一主体逻辑，
        # 保持与原来 as_tensor(x) + y 一致的逐元素相加语义（list 不会退化成拼接）
        left, right = as_tensor(x), y
        reversed_order = False

    if isinstance(right,TN):
        if not _same_device(left, right):
            raise RuntimeError(f'Expected all tensors to be on the same device, but found at least two devices, {left.device} and {right.device}!')
        right_data = right._array
        right_requires_grad = right.requires_grad
    else:
        right_data = right
        right_requires_grad = False

    #requires_grad属性在运算时传递到结果tensor
    backward_requested = (is_grad_enabled() and (left.requires_grad or right_requires_grad))
    record_graph = backward_requested or forward_ad_trace_active()
    ret = TN(left._array + right_data, requires_grad=backward_requested)
    
    if record_graph:
        ret.fromvars = (right, left) if reversed_order else (left, right)
        ret._grad_rule = _add_grad
        ret._jvp_rule = _add_jvp

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
        if not _same_device(left, right):
            raise RuntimeError(f'Expected all tensors to be on the same device, but found at least two devices, {left.device} and {right.device}!')
        right_data = right._array
        right_requires_grad = right.requires_grad
    else:
        right_data = right
        right_requires_grad = False

    #requires_grad属性在运算时传递到结果tensor
    backward_requested = (is_grad_enabled() and (left.requires_grad or right_requires_grad))
    record_graph = backward_requested or forward_ad_trace_active()
    if reversed_order:
        # 保持原始顺序 right - left（list/tuple/range 交给数组的 __rsub__ 逐元素计算）
        data = right_data - left._array
    else:
        data = left._array - right_data
    ret = TN(data, requires_grad=backward_requested)
    
    if record_graph:
        ret.fromvars = (right, left) if reversed_order else (left, right)
        ret._grad_rule = _sub_grad
        ret._jvp_rule = _sub_jvp

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
        reversed_order = False
    elif isinstance(y, TN):
        left, right = y, x
        reversed_order = True
    else:
        # 左右都不是TN：左值包装成TN后沿用同一主体逻辑
        left, right = as_tensor(x), y
        reversed_order = False

    if isinstance(right,TN):
        if not _same_device(left, right):
            raise RuntimeError(f'Expected all tensors to be on the same device, but found at least two devices, {left.device} and {right.device}!')
        right_data = right._array
        right_requires_grad = right.requires_grad
    else:
        right_data = right
        right_requires_grad = False

    #requires_grad属性在运算时传递到结果tensor
    backward_requested = (is_grad_enabled() and (left.requires_grad or right_requires_grad))
    record_graph = backward_requested or forward_ad_trace_active()
    ret = TN(left._array * right_data, requires_grad=backward_requested)
    
    if record_graph:
        ret.fromvars = (right, left) if reversed_order else (left, right)
        ret._grad_rule = _mul_grad
        ret._jvp_rule = _mul_jvp

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
        if not _same_device(left, right):
            raise RuntimeError(f'Expected all tensors to be on the same device, but found at least two devices, {left.device} and {right.device}!')
        right_data = right._array
        right_requires_grad = right.requires_grad
    else:
        right_data = right
        right_requires_grad = False

    #requires_grad属性在运算时传递到结果tensor
    backward_requested = (is_grad_enabled() and (left.requires_grad or right_requires_grad))
    record_graph = backward_requested or forward_ad_trace_active()
    if reversed_order:
        # 保持原始顺序 right / left（list/tuple/range 交给数组的 __rtruediv__ 逐元素计算）
        data = right_data / left._array
    else:
        data = left._array / right_data
    ret = TN(data, requires_grad=backward_requested)
    
    if record_graph:
        ret.fromvars = (right, left) if reversed_order else (left, right)
        ret._grad_rule = _div_grad
        ret._jvp_rule = _div_jvp

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
        if not _same_device(left, right):
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
        ret.fromvars = (right, left) if reversed_order else (left, right)
        ret._grad_rule = _floordiv_grad
        ret._jvp_rule = _floordiv_jvp

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
        if not _same_device(left, right):
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
    
    record_graph = requires_grad or forward_ad_trace_active()
    if record_graph:
        # d(left % right)/d(left)=1，d(left % right)/d(right)=-floor(left/right)，与 torch 一致
        ret.fromvars = (right, left) if reversed_order else (left, right)
        ret._grad_rule = _mod_grad
        ret._jvp_rule = _mod_jvp

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

    if not _same_device(left, right_tensor):
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

    if requires_grad or forward_ad_trace_active():
        if reversed_order:
            # fromvars 需按 matmul 的操作数顺序排列，梯度函数按相邻位置取另一操作数
            ret.fromvars=(right_tensor,left)
        else:
            ret.fromvars=(left,right_tensor)
        ret._grad_rule = _matmul_grad
        ret._jvp_rule = _matmul_jvp

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
    _require_tn(x, 'neg')
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

    if  ret.requires_grad or forward_ad_trace_active():
        ret.fromvars=(x,)
        ret._grad_rule = lambda result: (-result._grad_value,)
        ret._jvp_rule = _neg_jvp
    return ret

def positive(x)->TN:
    """
    逐元素取正（返回输入本身），接口与 torch.positive 一致。

    Args:
        x (TN): 输入张量

    Returns:
        TN: 输入张量本身
    """
    return as_tensor(x)

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
    return TN(~as_tensor(x)._array)

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

def _same_device(a: TN, b: TN) -> bool:
    """判断两个张量是否位于同一设备。

    直接按底层数组判断，避免为每次比较构造 Device 对象：
    numpy 数组一律位于 cpu；cupy 数组比较设备号。
    其余情况（一侧numpy、一侧cupy）必然不同设备。
    """
    a_arr, b_arr = a._array, b._array
    if isinstance(a_arr, np.ndarray) and isinstance(b_arr, np.ndarray):
        return True
    if cp and isinstance(a_arr, cp.ndarray) and isinstance(b_arr, cp.ndarray):
        return a_arr.device.id == b_arr.device.id
    return False


def _scalar_where_dtype(x, y):
    """两个标量分支的公共 dtype，与 torch 的标量提升规则一致。

    torch 把标量都当作弱类型：出现复数取默认复数类型，出现浮点取默认浮点类型，
    仅出现整数取 int64，否则为 bool。numpy 的 where 会把 Python 浮点提升为
    float64，因此这里先确定 dtype，再据此构造标量。
    """
    kinds = {np.dtype(infer_data_type(x)).kind, np.dtype(infer_data_type(y)).kind}
    if 'c' in kinds:
        return get_default_complex()
    if 'f' in kinds:
        return get_default_dtype()
    if kinds & {'i', 'u'}:
        return int64
    return np.dtype('bool')


def _where_branch_grad(r, i, mask):
    """按 mask 缩放上游梯度，并在广播场景下归约回源张量形状（单分支）"""
    grad = r._grad_value * mask
    var = r.fromvars[i]
    var_shape = var.shape
    grad_shape = grad.shape

    # 处理广播场景
    if var_shape != grad_shape:
        broadcast_axes = _get_broadcast_axis(grad_shape, var_shape)
        grad = grad.sum(dim=broadcast_axes, keepdim=False)._reshape(var_shape)
    
    # 仅在类型不一致时进行转换
    return grad.type(var.dtype)
    

def _where_grad(result_tensor:TN):
    """where 的融合梯度：fromvars 固定为 (x, y)，按各来源是否需要梯度分别计算。"""
    x, y = result_tensor.fromvars
    cond = result_tensor.parms[0]
    x_requires = isinstance(x, TN) and x.requires_grad
    y_requires = isinstance(y, TN) and y.requires_grad

    x_grad = _where_branch_grad(result_tensor, 0, cond) if x_requires else None

    y_grad = None
    if y_requires:
        # 取反掩码只计算一次；bool 掩码走逻辑非，避免浮点提升
        y_mask = ~cond if cond.dtype == np.bool_ else 1.0 - cond
        y_grad = _where_branch_grad(result_tensor, 1, y_mask)

    return x_grad, y_grad

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
        if not _same_device(cond, x):
            raise ValueError(f"Expected all tensors to be on the same device, "
                           f"but found devices: cond={cond.device}, x={x.device}")
        x_requires_grad = x.requires_grad
        x_data = x._array
    else:
        x_requires_grad = False
        x_data = x
        
    if y_is_tensor:
        if not _same_device(cond, y):
            raise ValueError(f"Expected all tensors to be on the same device, "
                           f"but found devices: cond={cond.device}, y={y.device}")
        y_requires_grad = y.requires_grad
        y_data = y._array
    else:
        y_requires_grad = False
        y_data = y

    # 两个分支都是标量时按 torch 的弱类型规则指定 dtype，
    # 否则 numpy 会把 Python 浮点提升为 float64
    if not x_is_tensor and not y_is_tensor and is_number(x) and is_number(y):
        scalar_dtype = _scalar_where_dtype(x, y)
        x_data = scalar_dtype.type(x)
        y_data = scalar_dtype.type(y)
        
    # 条件选择，cond不参与梯度计算
    data = arrlib.where(cond._array, x_data, y_data)
    # 使用计算结果的数据类型，避免强制类型转换
    ret = TN(
        data,
        requires_grad = (is_grad_enabled() and (x_requires_grad or y_requires_grad))
    )
    
    if ret.requires_grad or forward_ad_trace_active():
        # fromvars 固定保存 (x, y)，由梯度函数按各自 requires_grad 决定是否计算
        ret.fromvars = (x, y)
        ret.parms = (cond,)
        ret._grad_rule = _where_grad
        ret._jvp_rule = _where_jvp
    
    return ret

def conj(x):
    """计算输入的共轭。
    
    对于复数 z = a + bi，其共轭为 z* = a - bi。
    对于实数，共轭是其本身。
    
    参数:
        x: 输入，可以是TN张量、numpy/cupy数组或标量、python数值（包括复数）
        
    返回:
        TN张量、数组、标量或数值容器: 输入的共轭
        
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
    if isinstance(x, (TN, np.ndarray, np.generic)) or \
       (cp and isinstance(x, (cp.ndarray, cp.generic))):
        # 处理TN张量、numpy/cupy数组及标量
        return x.conj()
    elif isinstance(x, complex):
        # 处理python内置复数
        return x.conjugate()
    elif isinstance(x, (int, float, bool)):
        # 处理python实数（int, float）
        return x
    elif isinstance(x, (list, tuple, range)):
        # 递归处理容器中的数值
        return [conj(item) for item in x]
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
    if not _same_device(a, b):
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
    _require_tn(x, 'isinf')    
    arrlib = x._get_array_lib()
    value = arrlib.isinf(x._array)
    return as_tensor(value,device=x.device)

def isnan(x:TN)->TN:
    """
    检测张量中的NaN（不是数字）元素。
    
    参数:
        x: 输入张量
    
    返回:
        布尔类型张量，对应位置为True表示输入张量中该位置是NaN
    """
    _require_tn(x, 'isnan')    
    arrlib = x._get_array_lib()
    value = arrlib.isnan(x._array)
    return as_tensor(value,device=x.device)

def isreal(x:TN)->TN:
    """
    检测张量中的实数元素。

    参数:
        x: 输入张量

    返回:
        布尔类型张量，对应位置为True表示输入张量中该位置是实数
    """
    _require_tn(x, 'isreal')
    arrlib = x._get_array_lib()
    value = arrlib.isreal(x._array)
    return as_tensor(value,device=x.device)

def floor(x:TN)->TN:
    """
    计算张量的下取整。

    返回:
        包含输入张量下取整的新张量

    抛出:
        TypeError: 入参不是 TN（与 torch 一致，不做隐式转换）
    """
    _require_tn(x, 'floor')
    arrlib = x._get_array_lib()
    return as_tensor(arrlib.floor(x._array),device=x.device)

