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
Riemann Library Core Module: Tensor Functional Operations

This module holds tensor functional operations migrated from tensordef.py
(creation helpers, random generators, trigonometric / hyperbolic functions,
comparison & logical ops, stacking / splitting helpers, diagonal utilities,
index-merge helpers, top-k / multinomial sampling, etc.).

Dependency rule: every function here depends on the TN tensor class defined
in tensordef.py (imported below), while tensordef.py does NOT depend on this
module - the dependency is strictly one-way.

Example usage:
    >>> import riemann as rm
    >>> x = rm.sin(rm.tensor([0.0, 1.0]))
"""

from __future__ import annotations
import builtins
import numpy as np
from itertools import accumulate
from collections import Counter, defaultdict
from ..cuda import Device
from .dtype import *
from .ad_context import *
from .tensordef import *
from .tensordef import (
    _validate_shape, _require_tn, _get_device,
    _get_broadcast_axis, _apply_out, _apply_out_tuple,
    _convert_TNindex_to_numpy, _same_device
)


# ---------------------------------------------------------------------------
# 分区目录（分区内按“常量/辅助函数 -> 被依赖的底层实现 -> 对外接口”排列，
# 即被调用者在前、调用者在后）
#   1. 模块常量
#   2. 张量创建与转换（from_numpy / from_dlpack / arange / linspace）
#   3. 随机数（随机种子、均匀/正态分布、随机整数与排列、多项式采样）
#   4. 基础逐元素数学（取整、平方、指数/对数、最值、三角函数、双曲函数及其梯度）
#   5. 归约与统计（逻辑运算、求和、方差、乘积、累积和、排序、topk）
#   6. 索引合并（视图索引链 -> 原始张量索引）
#   7. 原地索引操作（_i*_full 全量向量化、反向回调、*at_/*at 接口、abs_/neg_）
#   8. 索引读写（index_select / masked_select / take / gather / scatter / setitem）
#   9. 形状变换与拼接（split / stack / concatenate / unfold / fold / repeat）
#   10. einsum（方程解析、批量对角线辅助、梯度计算与入口）
#   11. 矩阵与对角线（dot / outer / kron / tril / triu / diagonal / trace / fill_diagonal）
#   12. 其余工具（masked_fill / clamp / 梯度来源共享）
# ---------------------------------------------------------------------------


# ===========================================================================
# 1. 模块常量
# ===========================================================================


# cupy 不可用时为空元组，isinstance(x, ()) 恒为 False，
# 因此判断数组类型时无需再逐个检查 cp 是否为 None
_CP_NDARRAY = (cp.ndarray,) if cp is not None else ()
# 索引归一化中视为“数组/序列索引”的类型
_ARRAY_TYPES = (list, np.ndarray) + _CP_NDARRAY
# 索引归一化中视为“单个索引”（而非索引元组）的类型
_INDEX_TYPES = (slice, int) + _ARRAY_TYPES

# 反向传播需要“左值”的原地操作回调。
# mul/div/pow 的反向要用操作前的左值，因此必须在覆盖数据之前备份 target_copy；
# add/sub/set 的反向只用索引和梯度，不需要备份，可省去一次整块内存拷贝。
# 用回调函数名而不是函数对象做判断，避免这些函数定义在本类之后造成前向引用问题。
_LEFT_VALUE_BACKWARD_NAMES = {
    '_mulat_inplace_grad',
    '_divat_inplace_grad',
    '_powat_inplace_grad',
    '_modat_inplace_grad',
}

# 原地操作梯度规则函数名 -> 对应的原地操作方法名，__setitem__ 据此重放副本上的操作。
# 只收录复合赋值运算（+=、-=、*=、/=、**=、%=）；setat_ 不在此列——它本身是覆盖写，
# 直接按数据 setat 即可得到相同结果，无需重放。
_INPLACE_AT_METHODS = {
    '_addat_inplace_grad': 'addat_',
    '_subat_inplace_grad': 'subat_',
    '_mulat_inplace_grad': 'mulat_',
    '_divat_inplace_grad': 'divat_',
    '_powat_inplace_grad': 'powat_',
    '_modat_inplace_grad': 'modat_',
}


# ===========================================================================
# 2. 张量创建与转换（from_numpy / from_dlpack / arange / linspace）
# ===========================================================================


def from_numpy(arr: np.ndarray | cp.ndarray) -> TN:
    """
    从numpy或cupy数组创建张量，与原数组共享内存。

    该函数将现有的numpy或cupy数组转换为Riemann张量对象，与输入数组共享底层数据内存。
    这意味着对张量的修改会影响原始数组，反之亦然。

    参数:
        arr: numpy或cupy数组，必须包含数值类型的数据。
            如果是cupy数组，默认在CUDA设备上创建张量。
    
    返回值:
        TN: 新创建的张量对象，与输入的数组共享内存

    异常:
        TypeError: 当输入不是numpy/cupy数组或数组包含非数值类型数据时抛出
        
    示例:
        >>> import numpy as np
        >>> # 从numpy数组创建CPU张量
        >>> arr = np.array([1.0, 2.0, 3.0])
        >>> x = from_numpy(arr)
        >>> x[0] = 0  # 修改张量会影响原始numpy数组
        >>> print(arr)  # 输出: [0. 2. 3.]

        >>> # 从cupy数组创建CUDA张量（自动推断设备）
        >>> import cupy as cp
        >>> arr_cuda = cp.array([1.0, 2.0, 3.0])
        >>> x_cuda = from_numpy(arr_cuda)  # 自动在cuda:0上创建

    注意事项:
        1. 创建的张量默认requires_grad=False，不追踪梯度
        2. 张量与原始数组共享内存，修改一方会影响另一方
        3. 输入数组必须是数值类型（整数、浮点数、复数等）
    """

    array_type = (np.ndarray, cp.ndarray) if cp else (np.ndarray,)  # type: ignore
    if not isinstance(arr, array_type):
        raise TypeError("arr need to be numpy or cupy array")

    if not is_numeric_array(arr):
        raise TypeError("dtype of arr need to be numberic")

    return TN(arr)


def from_dlpack(ext_tensor) -> TN:
    """
    通过 DLPack 协议从外部张量/数组创建 Riemann 张量，零拷贝、共享底层存储。

    DLPack 是跨框架张量交换的通用协议，torch.Tensor、numpy.ndarray（2.x）、
    cupy.ndarray、jax.Array、TVM / Paddle 等框架的张量/数组都实现了该协议
    （通过 __dlpack__ 方法）。本函数利用协议把外部张量零拷贝地映射为 Riemann TN，
    设备与 dtype 保持不变。

    参数:
        ext_tensor: 实现了 __dlpack__ 和 __dlpack_device__ 协议的对象，可以是张量
            或数组（ndarray）。典型来源包括 torch.Tensor、numpy.ndarray、
            cupy.ndarray、jax.Array 等。Riemann TN 自身也实现了该协议，可以互转。

    返回值:
        TN: 新创建的张量对象，与 ext_tensor 共享底层存储。
            ext_tensor 在哪个设备上，返回的 TN 就在哪个设备上。

    异常:
        TypeError: 当输入不支持 DLPack 协议时抛出
        RuntimeError: 当输入设备类型无法识别，或 CUDA 张量到达但 CUDA 不可用时抛出

    示例:
        >>> # 从 PyTorch CUDA 张量零拷贝创建 Riemann 张量
        >>> import torch, riemann as rm
        >>> t = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        >>> x = rm.from_dlpack(t)       # 在 cuda:0 上，零拷贝
        >>> x.array()[0] = 42.0         # 修改 Riemann 张量
        >>> t[0]                        # PyTorch 张量也看到修改

        >>> # 从 NumPy / CuPy 数组
        >>> import numpy as np, cupy as cp
        >>> cpu_arr = np.array([1.0, 2.0])
        >>> y = rm.from_dlpack(cpu_arr)  # CPU 上
        >>> gpu_arr = cp.array([1.0, 2.0])
        >>> z = rm.from_dlpack(gpu_arr)  # CUDA 上，type(z.array()) is cp.ndarray

        >>> # 反向：把 Riemann 张量交给 PyTorch / CuPy / NumPy
        >>> t = rm.tensor([1.0, 2.0], device='cuda')
        >>> torch_t = torch.from_dlpack(t)   # PyTorch 零拷贝接入
        >>> cupy_a  = cp.from_dlpack(t)     # CuPy 零拷贝接入

    注意事项:
        1. DLPack 零拷贝共享底层存储，修改一方会影响另一方
        2. 跨设备（CPU↔CUDA）转换不属于 DLPack 的设计目标，
           请使用 rm.tensor / rm.as_tensor 处理需要设备迁移的场景
        3. 创建的张量默认 requires_grad=False，不追踪梯度
    """
    if not hasattr(ext_tensor, '__dlpack__'):
        raise TypeError(
            f"Object of type {type(ext_tensor).__name__} does not support "
            "DLPack protocol (missing __dlpack__ method). "
            "Use rm.tensor / rm.as_tensor for non-DLPack inputs."
        )
    
    # 通过 __dlpack_device__ 判断设备类型（DLPack 标准约定）
    # dev_type: 1=CPU (kDLCPU), 2=CUDA (kDLCUDA), 3=ROCM, 4=OpenCL, 5=Metal
    try:
        dev_type, dev_id = ext_tensor.__dlpack_device__()
    except (AttributeError, TypeError, ValueError) as e:
        raise RuntimeError(f"Failed to query DLPack device info: {e}") from e
    
    # 用 NumPy / CuPy 的 from_dlpack 标准 API 把 DLPack 张量转成原生 ndarray
    # 两者内部会调用 ext_tensor.__dlpack__() 获取 DLPack capsule
    if dev_type == 1:  # kDLCPU
        arr = np.from_dlpack(ext_tensor)
    elif dev_type == 2:  # kDLCUDA
        if cp is None:
            raise RuntimeError(
                "CUDA is not available: received a DLPack tensor on CUDA device "
                f"kDLCUDA={dev_id}, but Riemann has no CUDA support or no CUDA "
                "device is visible. Install cupy-cuda12x / cupy-cuda13x "
                "and ensure a CUDA device is attached."
            )
        arr = cp.from_dlpack(ext_tensor)
    else:
        raise RuntimeError(
            f"Unsupported DLPack device type: {dev_type}. "
            "Riemann currently supports CPU (kDLCPU=1) and CUDA (kDLCUDA=2)."
        )
    
    return TN(arr)


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
    if dev.type == 'cpu':
        data = np.arange(start, end, step, dtype=dt)
    else:
        with cp.cuda.Device(dev.index):
            data = cp.arange(start, end, step, dtype=dt)
    
    # 创建并返回张量
    return TN(data, requires_grad=bool(requires_grad))


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
    
    # 使用numpy/cupy的linspace创建数组
    dev = _get_device(device)
    if dev.type == 'cpu':
        data = np.linspace(start, end, num=steps, endpoint=endpoint, dtype=dt)
    else:
        with cp.cuda.Device(dev.index):
            data = cp.linspace(start, end, num=steps, endpoint=endpoint, dtype=dt)
    
    return TN(data, requires_grad=bool(requires_grad))


# ===========================================================================
# 3. 随机数（随机种子、均匀/正态分布、随机整数与排列、多项式采样）
# ===========================================================================


def manual_seed(seed: int) -> int:
    """
    设置全局随机种子，用于NumPy和CuPy（如果可用）的随机数生成器。
    
    此函数用于确保随机数生成的可重复性。
    设置种子后，后续的随机数生成操作（如rand、randn等）将产生可预测的结果。
    
    Args:
        seed (int): 随机种子值，必须是整数
        
    Returns:
        int: 设置的种子值
        
    Examples:
        >>> manual_seed(42)  # 设置随机种子为42
        42
        >>> x = rand(2, 2)  # 基于种子42生成的随机数
        >>> y = rand(2, 2)  # 基于种子42生成的下一批随机数
        
    Note:
        1. 此函数会同时设置NumPy的随机种子
        2. 如果CuPy可用，也会设置CuPy的随机种子
        3. 为了确保完全的可重复性，应在所有随机操作之前调用此函数
    """
    # 设置NumPy的随机种子
    np.random.seed(seed)
    
    # 如果CuPy可用，也设置CuPy的随机种子
    if cp is not None:
        cp.random.seed(seed)
    
    return seed


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
    
    return TN(data, requires_grad=bool(requires_grad))


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
    
    return TN(data, requires_grad=bool(requires_grad))


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
    
    return TN(data, requires_grad=bool(requires_grad))


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
    
    return TN(data, requires_grad=bool(requires_grad))


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
    
    return TN(data, requires_grad=bool(requires_grad))


@api_registry.register
def multinomial(input: TN, num_samples: int, replacement: bool = False, *, generator=None, out: TN|None = None) -> TN:
    """
    按照输入张量给出的多项分布权重进行抽样，返回类别的索引。

    接口与PyTorch的torch.multinomial()兼容，基于numpy/cupy的random.choice实现：
    - 输入为1-D张量时，返回形状为(num_samples,)的索引张量
    - 输入为2-D张量（m行）时，对每一行独立抽样，返回形状为(m, num_samples)的索引张量

    输入行不需要求和为1，元素值被当作相对权重自动归一化；
    权重必须非负且有限，当某一行权重全为0时按均匀分布抽样（与PyTorch一致）。
    replacement=False时，num_samples不能超过类别数。

    参数:
        input (TN): 1-D或2-D的权重张量，可以是浮点或整型（整型会转为浮点权重）
        num_samples (int): 抽样次数
        replacement (bool): True为有放回抽样，False为无放回抽样，默认为False
        generator: 为与PyTorch签名兼容而保留，当前忽略，使用numpy/cupy全局随机流
                   （可通过np.random.seed/cp.random.seed控制）
        out (TN, optional): 可选的输出张量，形状必须与结果一致

    返回:
        TN: int64类型的类别索引张量，与输入位于同一设备（CPU/GPU），不支持梯度

    异常:
        RuntimeError: 输入维度不是1或2、权重为负或非有限、无放回抽样数超过类别数等

    示例:
        >>> weights = rm.tensor([0., 10., 3., 0.])
        >>> rm.multinomial(weights, 2, replacement=True)
        tensor([1, 2])
    """
    # 输入类型与维度检查
    if not isinstance(input, TN):
        raise TypeError("multinomial: input must be a tensor")
    if input.ndim not in (1, 2):
        raise RuntimeError("multinomial only supports 1-D or 2-D input tensors")

    # num_samples必须是非负整数（排除bool）
    if isinstance(num_samples, bool) or not isinstance(num_samples, (int, np.integer)):
        raise TypeError("multinomial: num_samples must be an integer")
    num_samples = int(num_samples)
    if num_samples < 0:
        raise RuntimeError("multinomial: num_samples must be non-negative")

    n_categories = input.shape[-1]
    if not replacement and num_samples > n_categories:
        raise RuntimeError(
            f"cannot sample n_sample={num_samples} > input.size(-1)={n_categories} "
            f"samples when replacement=False")

    arrlib = input._get_array_lib()

    # choice要求概率为浮点类型，整型权重先转换
    weights = input._array
    if weights.dtype.kind != 'f':
        weights = weights.astype(float64)

    # 与PyTorch一致：权重必须非负且有限
    if bool(arrlib.any(weights < 0)):
        raise RuntimeError("prob_dist must be non-negative")
    if bool(arrlib.any(~arrlib.isfinite(weights))):
        raise RuntimeError("prob_dist must be finite")

    def _normalize(row):
        # 将一行权重归一化为概率；整行为0时退化为均匀分布（与PyTorch行为一致）
        total = row.sum()
        n = row.shape[0]
        if total <= 0:
            return arrlib.full((n,), 1.0 / n, dtype=row.dtype)
        return row / total

    def _sample(probs):
        """按概率 probs 抽样，返回 int64 索引数组。

        - 有放回(replacement=True)：直接用 numpy/cupy 的 random.choice
        - 无放回(replacement=False)：numpy 用 random.choice；
          cupy 的 choice 不支持 replace=False+p 组合，改用 Gumbel-max trick
          (对每类生成 Gumbel 噪声，取 log(p)+Gumbel 分数最大的 k 个)，
          数学上等价于按概率无放回抽样。
        """
        # num_samples=0 时统一跳过 choice 调用，避免无意义的 probs 校验
        if num_samples == 0:
            return arrlib.empty(0, dtype=arrlib.int64)
        if replacement:
            out = arrlib.random.choice(
                n_categories, size=num_samples, replace=True, p=probs)
        else:
            if arrlib is np:
                out = np.random.choice(
                    n_categories, size=num_samples, replace=False, p=probs)
            else:
                # Gumbel-max trick：cupy 不支持 choice(replace=False, p=...)
                u = arrlib.random.uniform(1e-30, 1.0, size=n_categories)
                gumbel = -arrlib.log(-arrlib.log(u))
                # 避免 log(0)：probs=0 处用 log(1)=0 占位，再置为 -inf 使其不被选中
                safe_p = arrlib.where(probs > 0, probs, 1.0)
                log_p = arrlib.where(probs > 0, arrlib.log(safe_p), -arrlib.inf)
                scores = log_p + gumbel
                # 升序后反转切片得到降序，避免创建 -scores 临时浮点数组
                order = arrlib.argsort(scores)[::-1]
                out = order[:num_samples]
        return out.astype(arrlib.int64)

    with device_context(input):
        if input.ndim == 1:
            probs = _normalize(weights)
            samples_arr = _sample(probs)
        else:
            # 2-D输入：逐行独立抽样（choice仅支持1-D概率向量）
            m = input.shape[0]
            samples_arr = arrlib.empty((m, num_samples), dtype=arrlib.int64)
            for row_idx in range(m):
                probs = _normalize(weights[row_idx])
                samples_arr[row_idx] = _sample(probs)

    # 索引结果为int64、无梯度，设备与输入保持一致
    result = as_tensor(samples_arr, device=input.device, requires_grad=False)

    # 处理out参数
    return _apply_out(out, result, (input,), "multinomial")


# ===========================================================================
# 4. 基础逐元素数学（取整、平方、指数/对数、最值、三角函数、双曲函数及其梯度）
# ===========================================================================


def _sign_derivative(result:TN, x:TN)->tuple[TN]:
    # sign函数在x=0处不可导，在其他点处梯度为0
    # 梯度为零，与上游梯度相乘后仍为零
    return (result._grad_value * zeros_like(x),)

@api_registry.register
@track_grad(_sign_derivative)
def sign(x:TN)->TN:
    """计算张量的符号函数
    
    对于复数 z = a + bi:
    - 如果 z ≠ 0: sign(z) = z / |z| = (a + bi) / √(a² + b²)
    - 如果 z = 0: sign(z) = 0
    
    该函数支持实数和复数张量，并正确处理反向传播。
    """
    _require_tn(x, 'sign')
    arrlib = x._get_array_lib()
    return TN(arrlib.sign(x._array))

def exp2(x:TN)->TN:
    _require_tn(x, 'exp2')
    return 2.0 ** x


def ceil(x:TN)->TN:
    """
    计算张量的上取整。

    返回:
        包含输入张量上取整的新张量

    抛出:
        TypeError: 入参不是 TN（与 torch 一致，不做隐式转换）
    """
    if not isinstance(x, TN):
        raise TypeError(f"ceil(): argument 'input' (position 1) must be Tensor, not {type(x).__name__}")
    arrlib = x._get_array_lib()
    return as_tensor(arrlib.ceil(x._array),device=x.device)


def trunc(x:TN)->TN:
    """
    计算张量的截断取整。

    返回:
        包含输入张量截断取整的新张量

    抛出:
        TypeError: 入参不是 TN（与 torch 一致，不做隐式转换）
    """
    if not isinstance(x, TN):
        raise TypeError(f"trunc(): argument 'input' (position 1) must be Tensor, not {type(x).__name__}")
    arrlib = x._get_array_lib()
    return as_tensor(arrlib.trunc(x._array),device=x.device)

def round(x:TN)->TN:
    """
    计算张量的四舍五入取整。

    返回:
        包含输入张量四舍五入取整的新张量

    抛出:
        TypeError: 入参不是 TN（与 torch 一致，不做隐式转换）
    """
    _require_tn(x, 'round')
    return as_tensor(x._array.round(),device=x.device)

def square(x:TN)->TN:
    """
    计算张量的平方。
    
    返回一个新张量，其中每个元素是输入张量对应元素的平方。
    
    Args:
        x (TN): 输入张量
        
    Returns:
        TN: 包含输入张量平方的新张量
        
    Examples:
        >>> a = tensor([1, 2, 3])
        >>> square(a)  # 返回[1, 4, 9]
        
        >>> b = tensor([[1, 2, 3], [4, 5, 6]])
        >>> square(b)  # 返回[[1, 4, 9], [16, 25, 36]]
    """
    _require_tn(x, 'square')
    return x ** 2.0


def _sin_grad(result_tensor:TN):
    x = result_tensor.fromvars[0]
    return (result_tensor._grad_value * cos(x).conj(),)


@api_registry.register
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
    _require_tn(x, 'sin')
    arrlib = x._get_array_lib()
    ret = TN(arrlib.sin(x._array), is_grad_enabled() and x.requires_grad)
        
    if ret.requires_grad:
        ret.fromvars = (x,)
        ret._grad_rule = _sin_grad

    return ret


def _cos_grad(result_tensor:TN):
    x = result_tensor.fromvars[0]
    return (result_tensor._grad_value * (-sin(x).conj()),)


@api_registry.register
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
    _require_tn(x, 'cos')
    arrlib = x._get_array_lib()
    ret = TN(arrlib.cos(x._array), is_grad_enabled() and x.requires_grad)
    
    if ret.requires_grad:
        ret.fromvars = (x,)
        ret._grad_rule = _cos_grad

    return ret


def _tan_grad(result_tensor:TN):
    x = result_tensor.fromvars[0]
    return (result_tensor._grad_value * (1. + (tan(x.conj()))**2.),)


@api_registry.register
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
    _require_tn(x, 'tan')
    arrlib = x._get_array_lib()
    ret = TN(arrlib.tan(x._array), is_grad_enabled() and x.requires_grad)
    
    if ret.requires_grad:
        ret.fromvars = (x,)
        ret._grad_rule = _tan_grad

    return ret


@api_registry.register
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


@api_registry.register
def sec(x:TN)->TN:
    return 1./cos(x)


@api_registry.register
def csc(x:TN)->TN:
    return 1./sin(x)


def _arcsin_derivative(result:TN, x:TN)->tuple[TN]:
    return (result._grad_value * (1./sqrt(1. - x**2.).conj()),)


@api_registry.register
@track_grad(_arcsin_derivative)
def arcsin(x:TN)->TN:
    _require_tn(x, 'arcsin')
    arrlib = x._get_array_lib()
    return TN(arrlib.arcsin(x._array))


def _arccos_derivative(result:TN, x:TN)->tuple[TN]:
    return (result._grad_value * (-1.0/sqrt(1. - x**2.).conj()),)


@api_registry.register
@track_grad(_arccos_derivative)
def arccos(x:TN)->TN:
    _require_tn(x, 'arccos')
    arrlib = x._get_array_lib()
    return TN(arrlib.arccos(x._array))


def _arctan_derivative(result:TN, x:TN)->tuple[TN]:
    return (result._grad_value * (1./(1. + x**2.).conj()),)


@api_registry.register
@track_grad(_arctan_derivative)
def arctan(x:TN)->TN:
    _require_tn(x, 'arctan')
    arrlib = x._get_array_lib()
    return TN(arrlib.arctan(x._array))


def _sinh_derivative(result:TN, x:TN)->tuple[TN]:
    return (result._grad_value * cosh(x).conj(),)


@api_registry.register
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
    _require_tn(x, 'sinh')
    arrlib = x._get_array_lib()
    return TN(arrlib.sinh(x._array))


def _cosh_derivative(result:TN, x:TN)->tuple[TN]:
    return (result._grad_value * sinh(x).conj(),)


@api_registry.register
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
    _require_tn(x, 'cosh')
    arrlib = x._get_array_lib()
    return TN(arrlib.cosh(x._array))


def _tanh_derivative(result:TN, x:TN)->tuple[TN]:
    return (result._grad_value * (1.0 - tanh(x)**2.0).conj(),)


@api_registry.register
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
    _require_tn(x, 'tanh')
    arrlib = x._get_array_lib()
    return TN(arrlib.tanh(x._array))


@api_registry.register
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


@api_registry.register
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


@api_registry.register
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


def _arcsinh_derivative(result:TN, x:TN)->tuple[TN]:
    return (result._grad_value * (1. / sqrt(x**2. + 1.).conj()),)


@api_registry.register
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
    _require_tn(x, 'arcsinh')
    arrlib = x._get_array_lib()
    return TN(arrlib.arcsinh(x._array))


def _arccosh_derivative(result:TN, x:TN)->tuple[TN]:
    return (result._grad_value * (1. / sqrt(x**2. - 1.).conj()),)


@api_registry.register
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
    _require_tn(x, 'arccosh')
    arrlib = x._get_array_lib()
    return TN(arrlib.arccosh(x._array))


def _arctanh_derivative(result:TN, x:TN)->tuple[TN]:
    return (result._grad_value * (1. / (1. - x**2.0).conj()),)


@api_registry.register
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
    _require_tn(x, 'arctanh')
    arrlib = x._get_array_lib()
    return TN(arrlib.arctanh(x._array))


@api_registry.register
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
    _require_tn(x, 'sqrt')
    return x ** 0.5


def _log_grad(result_tensor:TN):
    x = result_tensor.fromvars[0]
    return (result_tensor._grad_value * (1. / x.conj()),)


@api_registry.register
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
    _require_tn(x, 'log')
    arrlib = x._get_array_lib()
    result_data = arrlib.log(x._array)
    requires_grad = (is_grad_enabled() and x.requires_grad)
    ret = TN(result_data, requires_grad)
    
    if requires_grad:
        ret.fromvars = (x,)
        ret._grad_rule = _log_grad

    return ret


@api_registry.register
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


@api_registry.register
def log2(x:TN)->TN:
    """
    计算张量的二进制对数。
    
    返回一个新张量，其中每个元素是输入张量对应元素的二进制对数(log2(x))。
    
    Args:
        x (TN): 输入张量，元素必须为正数
        
    Returns:
        TN: 包含输入张量二进制对数的新张量
    """
    return log(x) / np.log(2.0)


@api_registry.register
def log10(x:TN)->TN:
    """
    计算张量的十进制对数。
    
    返回一个新张量，其中每个元素是输入张量对应元素的十进制对数(log10(x))。
    
    Args:
        x (TN): 输入张量，元素必须为正数
        
    Returns:
        TN: 包含输入张量十进制对数的新张量
    """
    return log(x) / np.log(10.0)
    

def _exp_grad(result_tensor:TN):
    return (result_tensor._grad_value * result_tensor.conj(),)


@api_registry.register
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
    _require_tn(x, 'exp')
    arrlib = x._get_array_lib()
    ret = TN(arrlib.exp(x._array), is_grad_enabled() and x.requires_grad)
    
    if ret.requires_grad:
        ret.fromvars = (x,)
        ret._grad_rule = _exp_grad

    return ret


@api_registry.register
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
    
    if not _same_device(input, other):
        raise RuntimeError('Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!')
    
    # 前向计算：使用numpy的maximum
    value = arrlib.maximum(input._array, other._array)
    ret = TN(value, requires_grad=(is_grad_enabled() and (input.requires_grad or other.requires_grad)))
    
    # 注册梯度函数
    if ret.requires_grad:
        ret.fromvars = (input, other)
        ret._grad_rule = _maximum_grad
    
    return ret

def _maximum_grad(result_tensor: TN):
    """
    maximum函数的融合梯度计算，返回值与 fromvars=(input, other) 对齐。

    梯度规则：
    - 当 input > other 时，input 梯度为 1（传递梯度）
    - 当 input < other 时，input 梯度为 0（阻断梯度）
    - 当 input == other 时，input 梯度为 0.5（与PyTorch行为一致，平均分配）
    other 的梯度按对称规则计算。
    """
    input_tensor = result_tensor.fromvars[0]
    other_tensor = result_tensor.fromvars[1]
    result_shape = result_tensor.shape

    # 计算 input 的梯度掩码
    # input > other: 1.0
    # input == other: 0.5  # 与PyTorch一致
    # input < other: 0.0
    # 使用riemann的where函数，自动处理设备上下文
    input_mask = where(
        input_tensor > other_tensor, 
        1.0, 
        where(input_tensor < other_tensor, 0.0, 0.5)
    ).type(input_tensor.dtype)

    # 应用梯度掩码
    grad = result_tensor._grad_value * input_mask

    # 处理广播情况
    input_shape = input_tensor.shape

    if input_shape != result_shape:
        # 获取需要缩减的广播轴
        broadcast_axes = _get_broadcast_axis(result_shape, input_shape)
        grad = sum(grad, dim=broadcast_axes, keepdim=False)._reshape(input_shape)

    input_grad = grad

    # 计算 other 的梯度掩码
    # other > input: 1.0
    # other == input: 0.5  # 与PyTorch一致
    # other < input: 0.0
    # 使用riemann的where函数，自动处理设备上下文
    other_mask = where(
        other_tensor > input_tensor, 
        1.0,
        where(other_tensor < input_tensor, 0.0, 0.5)
    ).type(other_tensor.dtype)

    # 应用梯度掩码
    grad = result_tensor._grad_value * other_mask

    # 处理广播情况
    other_shape = other_tensor.shape

    if other_shape != result_shape:
        # 获取需要缩减的广播轴
        broadcast_axes = _get_broadcast_axis(result_shape, other_shape)
        grad = sum(grad, dim=broadcast_axes, keepdim=False)._reshape(other_shape)

    other_grad = grad

    return input_grad, other_grad


@api_registry.register
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
    
    if not _same_device(input, other):
        raise RuntimeError('Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!')
        
    # 前向计算：使用numpy的minimum
    value = arrlib.minimum(input._array, other._array)
    ret = TN(value, requires_grad=(is_grad_enabled() and (input.requires_grad or other.requires_grad)))
    
    # 注册梯度函数
    if ret.requires_grad:
        ret.fromvars = (input, other)
        ret._grad_rule = _minimum_grad
    
    return ret

def _minimum_grad(result_tensor: TN):
    """
    minimum函数的融合梯度计算，返回值与 fromvars=(input, other) 对齐。

    梯度规则：
    - 当 input < other 时，input 梯度为 1（传递梯度）
    - 当 input >= other 时，input 梯度为 0（阻断梯度）
    - 当 input == other 时，input 梯度为 0.5（平均分配）
    other 的梯度按对称规则计算。
    """
    input_tensor = result_tensor.fromvars[0]
    other_tensor = result_tensor.fromvars[1]
    result_shape = result_tensor.shape

    # 计算 input 的梯度掩码
    # input < other: 1.0
    # input == other: 0.5  
    # input > other: 0.0
    # 使用riemann的where函数，自动处理设备上下文
    input_mask = where(
        input_tensor < other_tensor, 
        1.0, 
        where(input_tensor > other_tensor, 0.0, 0.5)
    ).type(input_tensor.dtype)

    # 应用梯度掩码
    grad = result_tensor._grad_value * input_mask

    # 处理广播情况
    input_shape = input_tensor.shape

    if input_shape != result_shape:
        # 获取需要缩减的广播轴
        broadcast_axes = _get_broadcast_axis(result_shape, input_shape)
        grad = sum(grad, dim=broadcast_axes, keepdim=False)._reshape(input_shape)

    input_grad = grad

    # 计算 other 的梯度掩码
    # other < input: 1.0
    # other == input: 0.5
    # other > input: 0.0
    # 使用riemann的where函数，自动处理设备上下文
    other_mask = where(
        other_tensor < input_tensor, 
        1.0,
        where(other_tensor > input_tensor, 0.0, 0.5)
    ).type(other_tensor.dtype)

    # 应用梯度掩码
    grad = result_tensor._grad_value * other_mask

    # 处理广播情况
    other_shape = other_tensor.shape

    if other_shape != result_shape:
        # 获取需要缩减的广播轴
        broadcast_axes = _get_broadcast_axis(result_shape, other_shape)
        grad = sum(grad, dim=broadcast_axes, keepdim=False)._reshape(other_shape)

    other_grad = grad

    return input_grad, other_grad


# ===========================================================================
# 5. 归约与统计（逻辑运算、求和、方差、乘积、累积和、排序、topk）
# ===========================================================================


def logical_and(input:TN, other:TN|int|float|bool)->TN:
    """
    逐元素计算逻辑与，非零值视为 True，返回布尔类型张量。

    参数:
        input: 输入张量
        other: 另一个张量或标量值，形状可与 input 广播

    返回:
        布尔类型张量，对应位置为 True 表示两个输入在该位置均非零
    """
    if not isinstance(input, TN):
        raise TypeError('input must be a tensor')

    if isinstance(other, TN):
        if not _same_device(input, other):
            raise ValueError(f"Expected all tensors to be on the same device, "
                             f"but found devices: input={input.device}, other={other.device}")
        other_data = other._array
    else:
        other_data = other

    arrlib = input._get_array_lib()
    value = arrlib.logical_and(input._array, other_data)
    return as_tensor(value,device=input.device)


def logical_or(input:TN, other:TN|int|float|bool)->TN:
    """
    逐元素计算逻辑或，非零值视为 True，返回布尔类型张量。

    参数:
        input: 输入张量
        other: 另一个张量或标量值，形状可与 input 广播

    返回:
        布尔类型张量，对应位置为 True 表示两个输入在该位置至少一个非零
    """
    if not isinstance(input, TN):
        raise TypeError('input must be a tensor')

    if isinstance(other, TN):
        if not _same_device(input, other):
            raise ValueError(f"Expected all tensors to be on the same device, "
                             f"but found devices: input={input.device}, other={other.device}")
        other_data = other._array
    else:
        other_data = other

    arrlib = input._get_array_lib()
    value = arrlib.logical_or(input._array, other_data)
    return as_tensor(value,device=input.device)


def logical_xor(input:TN, other:TN|int|float|bool)->TN:
    """
    逐元素计算逻辑异或，非零值视为 True，返回布尔类型张量。

    参数:
        input: 输入张量
        other: 另一个张量或标量值，形状可与 input 广播

    返回:
        布尔类型张量，对应位置为 True 表示两个输入在该位置恰好一个非零
    """
    if not isinstance(input, TN):
        raise TypeError('input must be a tensor')

    if isinstance(other, TN):
        if not _same_device(input, other):
            raise ValueError(f"Expected all tensors to be on the same device, "
                             f"but found devices: input={input.device}, other={other.device}")
        other_data = other._array
    else:
        other_data = other

    arrlib = input._get_array_lib()
    value = arrlib.logical_xor(input._array, other_data)
    return as_tensor(value,device=input.device)


def logical_not(x:TN)->TN:
    """
    逐元素计算逻辑非，非零值视为 True，返回布尔类型张量。

    参数:
        x: 输入张量

    返回:
        布尔类型张量，对应位置为 True 表示输入张量中该位置为零
    """
    if not isinstance(x, TN):
        raise TypeError('x must be a tensor')

    arrlib = x._get_array_lib()
    value = arrlib.logical_not(x._array)
    return as_tensor(value,device=x.device)


def _sumall_derivative(result: TN, *args) -> tuple:
    """
    计算 sumall() 的梯度。

    对每个输入，按其与结果的广播关系对上游梯度求和，恢复到输入形状。
    非 TN 参数不占梯度元组的位置；不需要梯度的 TN 参数返回 None。

    Args:
        result: sumall() 的输出张量，可通过 result._grad_value 获取上游梯度
        *args: sumall() 的全部输入参数

    Returns:
        tuple: 每个 TN 输入对应的梯度（按参数出现顺序）
    """
    grads = []
    for arg in args:
        if not isinstance(arg, TN):
            continue
        if not arg.requires_grad:
            grads.append(None)
            continue

        input_shape = arg.shape
        if input_shape == result.shape:
            grads.append(result._grad_value)
        else:
            # 计算广播轴并对 _grad_value 进行求和
            broadcast_axes = _get_broadcast_axis(result.shape, input_shape)
            grads.append(sum(result._grad_value, dim=broadcast_axes, keepdim=False)._reshape(input_shape))
    return tuple(grads)


@track_grad(_sumall_derivative)
def sumall(*args) -> TN:
    """
    将多个张量或非张量相加，返回和
        
    Args:
        *args: 任意数量的张量或非张量参数
        
    Returns:
        TN: 所有参数的和
    """
    if not args:
        raise ValueError("sumall() must have at least one argument")

    # 设备：优先使用首个张量的设备，否则使用默认设备
    tensor_args = [arg for arg in args if isinstance(arg, TN)]
    target_device = tensor_args[0].device if tensor_args else _get_device(None)

    # 所有张量必须在同一设备上
    bad_arg = next((arg for arg in tensor_args if not _same_device(arg, tensor_args[0])), None)
    if bad_arg is not None:
        raise ValueError(
            f"All tensor arguments must be on the same device, "
            f"but found {bad_arg.device} and {target_device}"
        )

    arrlib = np if target_device.type == 'cpu' else cp

    def _align(arg):
        """将参数统一到 target_device 对应的数组库"""
        if isinstance(arg, TN):
            return arg._array
        if cp is not None and arrlib is np and isinstance(arg, cp.ndarray):
            return cp.asnumpy(arg)                  # GPU → CPU
        if arrlib is cp and isinstance(arg, np.ndarray):
            with cp.cuda.Device(target_device.index):
                return cp.asarray(arg)              # CPU → GPU
        return arg                                  # python 值 / 同库数组

    # 累加；计算图由 track_grad 装饰器自动建立
    # 注意：模块内定义了名为 sum 的算子，此处必须用 builtins.sum 调用内置实现
    return as_tensor(builtins.sum(_align(arg) for arg in args), device=target_device)


def isum(*args, **kwargs):
    """
    智能sum函数：
    - 如果第一个参数是TN或数组类型，且只有1个位置参数，调用参数对象的sum方法函数
    - 如果多个参数中包含张量、numpy数组或cupy数组、python数值，调用python内置sum(args)
    - 否则直接调用 builtins.sum（Python内置sum）
    
    参数:
        *args: 可变参数
        **kwargs: 关键字参数，可包含 dim 和 keepdim（用于张量求和）
    
    返回:
        根据参数类型返回张量、numpy/cupy数组或Python数值
    
    示例:
        >>> isum(tensor([1, 2, 3]))  # 张量求和
        tensor(6)
        >>> isum(tensor([1, 2, 3]), dim=0)  # 沿维度求和
        tensor(6)
        >>> isum(1, 2, 3)  # 多个数值求和
        6
        >>> isum([1, 2, 3])  # Python列表求和
        6
    """
    
    # 情况1：只有一个位置参数且是TN或numpy/cupy数组类型
    if len(args) == 1 :
        x = args[0]
        if (isinstance(x, TN) or
            isinstance(x, np.ndarray) or
            (cp is not None and isinstance(x, cp.ndarray))
        ):
            # 只有一个位置参数，或有关键字参数 -> 调用 _sum
            return x.sum(**kwargs)
    
    # 情况2：多个位置参数，检查参数是否张量、数组或python数值类型
    if len(args) >= 2:
        data_types = (
            TN, int, float, complex, bool,
            np.integer, np.floating, np.complexfloating, np.bool_,
            np.ndarray
        )
        if cp:
            data_types = data_types + (cp.ndarray,)
        
        if all(isinstance(arg, data_types) for arg in args):
            return builtins.sum(args)
    
    # 情况3：其它情况，调用 builtins.sum
    return builtins.sum(*args, **kwargs)


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


def _prod_grad(result_tensor:TN):
    dim, keepdims = result_tensor.parms[0]

    # 如果prod计算时结果张量做过维度精简，需要将缩减的维度暂时恢复
    if keepdims == False:
        new_result_grad = result_tensor._grad_value.unsqueeze(dim)
    else:
        new_result_grad = result_tensor._grad_value
    
    x = result_tensor.fromvars[0]
    
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
    
    return (grad,)


@api_registry.register
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
    
    # 使用numpy/cupy的prod函数计算乘积
    prod_value = x._array.prod(axis=dim, keepdims=keepdim)
    ret = TN(prod_value, requires_grad = (is_grad_enabled() and x.requires_grad))
    
    if ret.requires_grad:
        ret.fromvars = (x,)
        ret.parms = ((dim, keepdim),)
        ret._grad_rule = _prod_grad
    
    return ret


def _cumsum_grad(result_tensor: TN):
    """
    cumsum函数的融合梯度规则。
    累积和的梯度是反向累积和。
    """
    dim, dtype_param = result_tensor.parms[0]  # 直接使用转换后的dtype
    
    # 获取梯度值，使用Riemann的张量操作保持计算图
    grad_value = result_tensor._grad_value
    
    # 反向累积和：从后向前累积
    # 例如：[g1, g2, g3, g4] -> [g1+g2+g3+g4, g2+g3+g4, g3+g4, g4]
    reversed_grad = flip(grad_value, [dim])
    reversed_cumsum = cumsum(reversed_grad, dim=dim)  # 使用Riemann的cumsum函数，不是递归调用
    input_grad = flip(reversed_cumsum, [dim])
    
    # 如果指定了dtype，转换回原始数据类型
    if dtype_param != input_grad.dtype:
        input_grad = input_grad.type(dtype_param)
    
    return (input_grad,)


@api_registry.register
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
        ret._grad_rule = _cumsum_grad
    
    return _apply_out(out, ret, (input,), "cumsum")


def _sort_grad(result_tensor: TN):
    """
    排序操作的融合梯度规则
    对于排序后的张量，梯度需要按照原始索引重新排列
    """
    # 获取原始输入张量
    input_tensor = result_tensor.fromvars[0]
    # 获取排序时使用的索引、排序维度
    sorted_indices, sort_dim = result_tensor.parms[0]
    # 获取梯度值
    grad_value = result_tensor._grad_value
    
    # 创建与输入形状相同的零张量
    grad = zeros_like(input_tensor)
    
    # 获取输入张量对应的数组库
    arrlib = input_tensor._get_array_lib()
    # 将梯度按照原始索引放置到正确的位置
    # 使用高级索引将梯度值分配回原始位置
    if sorted_indices.ndim == 1:
        # 1D情况
        grad[sorted_indices] = grad_value
    else:
        # 多维情况，根据排序维度创建正确的坐标网格
        # 创建与sorted_indices形状相同的坐标网格（确保在源张量的设备上创建）
        with device_context(input_tensor):
            coords = list(arrlib.ogrid[tuple(slice(0, size) for size in sorted_indices.shape)])
        # 用排序后的索引替换排序维度的坐标
        coords[sort_dim] = sorted_indices._array
        # 使用这些坐标将梯度值分配回原始位置
        grad[tuple(coords)] = grad_value
    
    return (grad,)


@api_registry.register
def sort(input: TN, dim: int = -1, descending: bool = False, stable: bool = False, *, out = None) -> tuple[TN, TN]:
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
    # 确保dim在有效范围内
    if dim < -input.ndim or dim >= input.ndim:
        raise IndexError(f"Dimension out of range (expected to be in range of [-{input.ndim}, {input.ndim-1}], got {dim})")
    
    # 标准化维度索引
    if dim < 0:
        dim += input.ndim
    
    arrlib = input._get_array_lib()
    # 使用numpy进行排序
    # numpy的sort不支持stable参数，所以这里忽略stable参数
    sorted_indices = arrlib.argsort(input._array, axis=dim)
    
    # 创建用于索引的坐标网格（确保在输入张量的设备上创建）
    with device_context(input):
        coords = list(arrlib.ogrid[tuple(slice(0, size) for size in input._array.shape)])
    # 用排序后的索引替换指定维度的坐标
    coords[dim] = sorted_indices
    
    # 获取排序后的值
    sorted_values_data = input._array[tuple(coords)]
    
    # 如果需要降序，反转排序结果
    if descending:
        # 创建反转索引
        reverse_slice = [slice(None)] * input._array.ndim
        reverse_slice[dim] = slice(None, None, -1)
        sorted_values_data = sorted_values_data[tuple(reverse_slice)]
        # 调整索引以反映降序
        # 对于每个子数组，计算原始位置的反转
        # 这部分比较复杂，需要根据具体维度处理
        # 创建一个与sorted_indices形状相同的数组，用于存储反转后的索引
        with device_context(input):
            reverse_indices = arrlib.empty_like(sorted_indices)
        # 对每个子数组进行处理
        # 注意：nditer在CPU上执行，但操作的是已有数组
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
    sorted_values = TN(sorted_values_data, requires_grad=(is_grad_enabled() and input.requires_grad))
    
    # 创建索引张量（索引张量不需要梯度）
    sorted_indices_tensor = TN(sorted_indices, requires_grad=False)
    
    # 设置梯度信息
    if sorted_values.requires_grad:
        sorted_values.fromvars = (input,)
        sorted_values.parms = ((sorted_indices_tensor, dim),)
        sorted_values._grad_rule = _sort_grad
    
    # 处理out参数
    return _apply_out_tuple(out, (sorted_values, sorted_indices_tensor), (input,), "sort")


def _topk_grad(result_tensor: TN):
    """
    topk操作的融合梯度规则。
    将输出位置上的梯度按照topk索引散射回输入张量的对应位置，
    未被选中的输入位置梯度为0。
    """
    # 获取原始输入张量
    input_tensor = result_tensor.fromvars[0]
    # 获取topk选择时使用的索引和维度
    topk_indices, topk_dim = result_tensor.parms[0]
    # 获取梯度值
    grad_value = result_tensor._grad_value

    # 创建与输入形状相同的零张量
    grad = zeros_like(input_tensor)

    # 将梯度按照topk索引放置到正确的位置
    if topk_indices.ndim == 1:
        # 1D情况
        grad[topk_indices] = grad_value
    else:
        # 多维情况，根据topk维度创建正确的坐标网格
        arrlib = input_tensor._get_array_lib()
        with device_context(input_tensor):
            # 创建坐标网格（确保在源张量的设备上创建）
            coords = list(arrlib.ogrid[tuple(slice(0, size) for size in topk_indices.shape)])
        # 用topk的索引替换该维度的坐标
        coords[topk_dim] = topk_indices._array
        # 使用这些坐标将梯度值分配回原始位置
        grad[tuple(coords)] = grad_value

    return (grad,)


@api_registry.register
def topk(input: TN, k: int, dim: int = -1, largest: bool = True, sorted: bool = True, *, out = None) -> tuple[TN, TN]:
    """
    沿指定维度返回输入张量中k个最大（或最小）元素的值和索引。

    接口与PyTorch的torch.topk()兼容，基于numpy/cupy的argpartition实现：
    先通过argpartition以O(n)复杂度将前k个元素划分到一起，
    当sorted=True时再对这k个候选项做argsort，保证返回值有序。

    参数:
        input (TN): 输入张量，至少为1维，不支持复数类型
        k (int): 返回元素的个数，必须满足 0 <= k <= input.size(dim)
        dim (int): 查找的维度，默认为最后一维
        largest (bool): True返回最大的k个元素，False返回最小的k个元素，默认为True
        sorted (bool): True保证返回元素按值有序（最大模式降序、最小模式升序），
                       False时仅保证选出的是top-k元素、顺序不做保证，默认为True
        out: 可选的输出元组 (values, indices)，使用out时不支持自动微分

    返回:
        tuple[TN, TN]: (values, indices)，values为top-k元素的值（支持梯度），
                       indices为对应的int64索引（不支持梯度）

    异常:
        IndexError: dim超出维度范围
        RuntimeError: k超出范围、输入为复数、out与requires_grad冲突等

    示例:
        >>> x = rm.tensor([3., 1., 4., 1., 5.])
        >>> values, indices = rm.topk(x, 2)
        >>> values
        tensor([5., 4.])
        >>> indices
        tensor([4, 2])
    """
    # 输入类型检查
    if not isinstance(input, TN):
        raise TypeError("topk: input must be a tensor")
    if input.is_complex():
        raise RuntimeError("topk() does not support complex input")
    # k必须是整数（排除bool）
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError("topk: k must be an integer")
    k = int(k)
    # torch要求输入至少1维
    if input.ndim == 0:
        raise RuntimeError("topk: input must have at least 1 dimension")

    # 确保dim在有效范围内
    if dim < -input.ndim or dim >= input.ndim:
        raise IndexError(f"Dimension out of range (expected to be in range of "
                         f"[-{input.ndim}, {input.ndim - 1}], got {dim})")
    # 标准化维度索引
    if dim < 0:
        dim += input.ndim

    dim_size = input.shape[dim]
    if k < 0 or k > dim_size:
        raise RuntimeError(f"selected index k out of range, k should be in [0, {dim_size}] but got {k}")

    arrlib = input._get_array_lib()
    data = input._array

    with device_context(input):
        if k == 0:
            # k=0时返回指定维度长度为0的空结果
            result_shape = data.shape[:dim] + (0,) + data.shape[dim+1:]
            values_arr = arrlib.empty(result_shape, dtype=data.dtype)
            indices_arr = arrlib.empty(result_shape, dtype=arrlib.int64)
        else:
            # 第一步：利用argpartition以O(n)复杂度将top-k候选划分到一起
            if largest:
                # 升序划分后，最大的k个位于维度末尾
                part_indices = arrlib.argpartition(data, kth=dim_size - k, axis=dim)
                candidate_pos = slice(dim_size - k, dim_size)
            else:
                # 升序划分后，最小的k个位于维度开头
                part_indices = arrlib.argpartition(data, kth=k - 1, axis=dim)
                candidate_pos = slice(0, k)

            # 沿dim取出前k个候选索引
            take_slice = [slice(None)] * data.ndim
            take_slice[dim] = candidate_pos
            candidate_indices = part_indices[tuple(take_slice)]

            # 用候选索引从原数据中聚集出候选值
            coords = list(arrlib.ogrid[tuple(slice(0, size) for size in data.shape)])
            coords[dim] = candidate_indices
            values_arr = data[tuple(coords)]

            # 第二步：sorted=True时对k个候选项排序，保证输出有序
            # 统一升序argsort，largest模式沿dim反转切片得到降序，
            # 避免对values_arr取负号以兼容无符号整型，且省去临时数组
            if sorted:
                order = arrlib.argsort(values_arr, axis=dim)
                if largest:
                    rev = [slice(None, None, -1) if d == dim else slice(None)
                           for d in range(values_arr.ndim)]
                    order = order[tuple(rev)]
                order_coords = list(arrlib.ogrid[tuple(slice(0, size) for size in values_arr.shape)])
                order_coords[dim] = order
                indices_arr = candidate_indices[tuple(order_coords)]
                values_arr = values_arr[tuple(order_coords)]
            else:
                indices_arr = candidate_indices

    # 创建结果张量：values支持梯度，indices为整型索引不支持梯度
    values_tensor = as_tensor(values_arr, device=input.device,
                           requires_grad=(is_grad_enabled() and input.requires_grad))
    indices_tensor = as_tensor(indices_arr, device=input.device, requires_grad=False)

    if values_tensor.requires_grad:
        values_tensor.fromvars = (input,)
        values_tensor.parms = ((indices_tensor, dim),)
        values_tensor._grad_rule = _topk_grad

    # 处理out参数
    return _apply_out_tuple(out, (values_tensor, indices_tensor), (input,), "topk")


def nonzero(input: TN, *, as_tuple: bool = False) -> TN | tuple[TN, ...]:
    """
    返回输入张量中所有非零元素的索引。
    
    参数:
        input: 输入张量
        as_tuple: 如果为True，返回一个元组，其中每个元素对应一个维度的索引张量；
                 如果为False（默认），返回一个二维张量，每行是一个非零元素的坐标
    
    返回:
        TN或tuple[TN, ...]: 非零元素的索引
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
    indices = arrlib.nonzero(input._array)
    
    if as_tuple:
        # 返回元组形式，每个维度一个张量
        result = tuple(as_tensor(idx,device=input.device) for idx in indices)
    else:
        # 返回二维张量形式，每行是一个坐标
        if len(indices) > 0:
            # 将索引数组堆叠成二维数组
            result_data = arrlib.stack(indices, axis=1)
        else:
            # 如果没有非零元素，返回空的二维数组
            with device_context(input):
                result_data = arrlib.empty((0, input.ndim), dtype=np.int64)
        result = as_tensor(result_data,device=input.device)  # type: ignore
    
    return result


def _var_grad(result_tensor:TN):
    dim,unbiased,keepdim = result_tensor.parms[0]     
    x=result_tensor.fromvars[0]
    
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
        grad_value = result_tensor._grad_value.unsqueeze(dim)
    else:
        grad_value = result_tensor._grad_value
    
    # 计算最终梯度
    grad = grad_value * diff * factor
    return (grad,)


@api_registry.register
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
        ret._grad_rule=_var_grad
    
    return ret


@api_registry.register
def std(x:TN, dim:int|tuple|None=None, unbiased:bool=True, keepdim:bool=False)->TN:
    return sqrt(var(x,dim,unbiased,keepdim))


# ===========================================================================
# 6. 索引合并（视图索引链 -> 原始张量索引）
# ===========================================================================
#
# x[base_index][current_index] 等价于 x[merge_indices(x, base_index, current_index)]：
# 把两段索引复合为一次，用于原地操作定位到原始张量中的真实元素。
#
# 调用关系（按 merge_indices 的五个阶段）：
#   merge_indices(self, base_index, current_index)
#     ├─ 阶段1 _try_merge_special_base_index
#     │        空元组 / (None,) / 全整数三类特例，命中直接返回，否则继续
#     ├─ 阶段2 _preprocess_base_index
#     │        └─ _normalize_index_to_tuple、_expand_ellipsis
#     │        → (base_tuple, view_dims)  view_dims 为保留维度的位置
#     ├─ 阶段3 _preprocess_current_index
#     │        └─ _normalize_index_to_tuple、_expand_ellipsis
#     │        → 与 view_dims 等长的当前索引（保留 None 的位置）
#     ├─ 阶段4 逐个视图维度复合：
#     │        base_idx ─_normalize_index→ base_desc
#     │        curr_idx ─_normalize_index→ current_desc（dim_size 取视图大小）
#     │        两者 ─_merge_descriptors→ merged ─_descriptor_to_index→ 索引对象
#     └─ 阶段5 按 base_index 的维度顺序组装：整数/None 维度原样保留，
#              视图维度输出阶段4结果，并按相对位置插入 current 的 None
#
# 数据表示：_IndexDescriptor 统一描述“一个维度上的选择”
#   LINEAR  f(i) = start + i*step   切片，正负步长通用，可直接复合
#   LOOKUP  indices[i]              整数/数组/布尔，负索引已归一化
#   None    仅添加维度、不消耗视图维度，不进入描述符
# ===========================================================================


class _IndexDescriptor:
    """统一索引描述符：用一个维度上的“选择”表示切片/整数/数组索引。

    索引合并需要“两次选择复合为一次”，因此把各类索引归约为两种可复合形式：

    - LINEAR：等差数列 f(i) = start + i*step，i ∈ [0, (stop-start)/step)；
      正负步长统一适用，两个切片复合后仍是切片
    - LOOKUP：显式离散位置表 indices；布尔索引已转为 True 所在位置，
      负索引已按维度大小归一化，两个查找表复合后仍是查找表
    - NONE：保留类型，仅表示 np.newaxis 语义；实际 None 在 _normalize_index
      中直接返回 Python None，不进入描述符运算

    属性:
        type: 索引类型（LINEAR、LOOKUP 或 NONE）
        params: 类型特定参数，LINEAR 为 {start, stop, step}，LOOKUP 为 {indices}

    示例:
        >>> _IndexDescriptor(_IndexDescriptor.LINEAR, start=2, stop=10, step=2)
        >>> _IndexDescriptor(_IndexDescriptor.LOOKUP, indices=np.array([0, 2, 4]))

    调用者: _normalize_index（构造）、_merge_descriptors（复合）、
    _descriptor_to_index（还原为可直接索引的对象）
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
            return "NONE()"


def _normalize_index_to_tuple(index):
    """将单个索引或索引序列统一标准化为元组。

    merge_indices 内部统一用元组表示索引，避免后续按位置取值时对
    int/slice/list 等类型做分支判断。规则：
      - 已是 tuple：原样返回
      - 单个索引（slice/int/list/np.ndarray/cupy 数组）：包装为单元素元组
      - 其他可迭代对象：tuple(index)

    注意：None 索引会由调用方先转换为 (None,)，因此这里不单独处理 None。

    调用者: merge_indices、_try_merge_special_base_index、_preprocess_base_index、
    _preprocess_current_index
    """
    if isinstance(index, tuple):
        return index
    elif index is Ellipsis:
        # 单省略号包装成元组，交由后续 _expand_ellipsis 按目标维度数展开
        return (Ellipsis,)
    elif isinstance(index, _INDEX_TYPES):
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

    调用者: _preprocess_base_index、_preprocess_current_index
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

    调用者: merge_indices（第一阶段，先尝试快速合并特殊情况）
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

    调用者: merge_indices（第二阶段，预处理 base_index）
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

    调用者: merge_indices（第三阶段，预处理 current_index）
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

    调用者: merge_indices（第四阶段，逐个视图维度标准化）
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
    
    elif isinstance(index, _ARRAY_TYPES):
        # 处理列表、NumPy 数组或 CuPy 数组索引
        is_cupy = isinstance(index, _CP_NDARRAY)
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

    调用者: merge_indices（第四阶段，复合 base/current 两个描述符）
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
        lib = cp if isinstance(table, _CP_NDARRAY) else np

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
        lib = cp if isinstance(table2, _CP_NDARRAY) else np

        # 如果 table1 和 table2 类型不同，需要转换 table1
        is_cupy1 = isinstance(table1, _CP_NDARRAY)
        is_cupy2 = isinstance(table2, _CP_NDARRAY)
        if is_cupy1 and not is_cupy2:
            # table1 是 CuPy，table2 是 NumPy：将 table1 转换为 NumPy
            table1 = table1.get()
        elif not is_cupy1 and is_cupy2:
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

    调用者: merge_indices（第四阶段，把合并后的描述符转回索引对象）
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


@api_registry.register
def merge_indices(self, base_index, current_index):
    """把视图索引链 base_index → current_index 复合为原始张量上的一次索引。

    用于原地操作：求出 x[base_index][current_index] 实际命中的原始张量元素，
    返回可直接用于 x 的索引元组。整数维度已在视图中消去，None 维度只添加
    坐标轴，二者都不参与复合，最后按原始维度顺序组装。

    五个阶段（调用关系图见本节顶部）：
        1. _try_merge_special_base_index：空元组 / (None,) / 全整数三类特例
        2. _preprocess_base_index：标准化 base_index、展开 Ellipsis，得到 view_dims
        3. _preprocess_current_index：标准化 current_index，按视图维度数展开并补齐
        4. 逐视图维度复合：_normalize_index ×2 → _merge_descriptors → _descriptor_to_index
        5. 组装：按 base_index 顺序输出，并插入 current_index 中的 None

    关键规则（None）：
        - None 不消耗视图维度，current 的非 None 元素依次对应 view_dims[0..]
        - 非 None 元素不足时末尾补 slice(None)，等价于 Python 的隐式全切片
        - 结果中的 None 插入到对应视图维度之前；j == len(view_dims) 时追加到末尾

    Args:
        base_index: 创建视图时的索引（slice/int/tuple/list/ndarray）
        current_index: 在视图上继续索引时的索引

    Returns:
        tuple: 可直接用于原始张量的索引

    示例:
        >>> merge_indices(self, slice(2, 8), slice(1, 3))        # x[2:8][1:3] == x[3:5]
        (slice(3, 5, None),)
        >>> merge_indices(self, slice(None, None, -1), slice(1, 4))   # 负步长：取 [8, 7, 6]
        (slice(8, 5, -1),)
        >>> merge_indices(self, (slice(2, 8), slice(3, 7)), (slice(1, 3), slice(0, 2)))
        (slice(3, 5, None), slice(3, 5, None))
        >>> merge_indices(self, [0, 2, 4], [0, 2])              # x[[0,2,4]][[0,2]] == x[[0,4]]
        (array([0, 4]),)
        >>> merge_indices(self, (None,), (0,))                  # x[None][0] == x[:]
        (slice(None),)
        >>> merge_indices(self, (slice(2, 8), slice(3, 7)), (None, slice(1, 3)))
        (None, slice(3, 5, None), slice(3, 7, None))

    调用者: _inplace_unary_at_、_inplace_oper_at_（视图索引重定向时合并两段索引）；
    同时作为对外接口供 rm.merge_indices 调用
    """
    
    # 阶段1：特例快速合并（空元组 / (None,) / 全整数），命中即返回
    result = _try_merge_special_base_index(base_index, current_index)
    if result is not None:
        return result

    # 基础张量形状：后续阶段解析负索引/负步长时复用
    base_shape = self._base.shape if (hasattr(self, '_base') and self._base is not None) else self.shape

    # 阶段2：预处理 base_index → (标准化元组, view_dims)
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
    
    # 阶段3：预处理 current_index → 与 view_dims 等长的索引序列
    current_index = _preprocess_current_index(current_index, view_dims)

    # 阶段4：逐个视图维度复合 base/current 两个描述符
    merged_dims = {}  # 视图维度位置 -> 合并后的索引
    current_non_none = [idx for idx in current_index if idx is not None]

    for i, view_dim in enumerate(view_dims):
        # 第 i 个非 None 索引对应第 i 个视图维度；None 不参与合并，
        # 留到阶段5按相对位置插入
        base_idx = base_index[view_dim]
        curr_idx = current_non_none[i]

        # 该视图维度在原始张量中对应维度的大小（用于解析负索引/负步长）
        dim_size = base_shape[view_dim] if view_dim < len(base_shape) else 1
        
        # 标准化基础索引为 IndexDescriptor
        base_desc = _normalize_index(base_idx, dim_size)

        # view_dims 已排除 None，这里仅作防御：无法标准化时保持原索引
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
    
    # 阶段5：组装结果
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


# ===========================================================================
# 7. 原地索引操作（_i*_full 全量向量化、反向回调、*at_/*at 接口、abs_/neg_）
# ===========================================================================


# 全量索引（index 为 ()）时使用的向量化原地操作。
# 与 ufunc.at 的分工：
# - oper_func（ufunc.at）面向任意索引，必须保证重复索引“累加”的语义，
#   但在全量索引下会退化成逐元素循环（实测慢两个数量级）；
# - 全量索引不存在重复元素，可直接使用原地运算符，语义与 ufunc.at 等价。
# 传入的右值已由 _adjust_right_val 转成 input 的 dtype 并完成形状对齐，
# 因此这里不会引入额外的类型转换差异，cupy 数组同样适用。
def _iadd_full(numpy_arr, right):
    """全量索引（index 为 ()）下执行原地加法 arr += right，绕开 ufunc.at 的逐元素慢路径。

    调用者: addat_（作为 vector_func 实参传入 _inplace_oper_at_）
    """
    numpy_arr += right
    return numpy_arr


def _isub_full(numpy_arr, right):
    """全量索引（index 为 ()）下执行原地减法 arr -= right，绕开 ufunc.at 的逐元素慢路径。

    调用者: subat_（作为 vector_func 实参传入 _inplace_oper_at_）
    """
    numpy_arr -= right
    return numpy_arr


def _imul_full(numpy_arr, right):
    """全量索引（index 为 ()）下执行原地乘法 arr *= right，绕开 ufunc.at 的逐元素慢路径。

    调用者: mulat_（作为 vector_func 实参传入 _inplace_oper_at_）
    """
    numpy_arr *= right
    return numpy_arr


def _idiv_full(numpy_arr, right):
    """全量索引（index 为 ()）下执行原地除法 arr /= right，绕开 ufunc.at 的逐元素慢路径。

    调用者: divat_（作为 vector_func 实参传入 _inplace_oper_at_）
    """
    numpy_arr /= right
    return numpy_arr


def _ipow_full(numpy_arr, right):
    """全量索引（index 为 ()）下执行原地幂运算 arr **= right，绕开 ufunc.at 的逐元素慢路径。

    调用者: powat_（作为 vector_func 实参传入 _inplace_oper_at_）
    """
    numpy_arr **= right
    return numpy_arr


def _ifloordiv_full(numpy_arr, right):
    """全量索引（index 为 ()）下执行原地向下取整除法 arr //= right。

    调用者: floordivat_（作为 vector_func 实参传入 _inplace_oper_at_）
    """
    numpy_arr //= right
    return numpy_arr


def _imod_full(numpy_arr, right):
    """全量索引（index 为 ()）下执行原地取余 arr %= right。

    调用者: modat_（作为 vector_func 实参传入 _inplace_oper_at_）
    """
    numpy_arr %= right
    return numpy_arr


def _iset_full(numpy_arr, right):
    """全量索引（index 为 ()）下执行覆盖写 arr[...] = right。

    调用者: setat_（作为 vector_func 实参传入 _inplace_oper_at_）
    """
    numpy_arr[...] = right
    return numpy_arr


def _setat_grad(result_tensor:TN):
    """非原地 setat 的融合反向：返回 (左值梯度, 右值梯度)。

    左值：索引位置已被覆盖写，故把该位置梯度清零；
    右值：取 self 梯度在索引位置的取值。

    调用者: setat（作为 grad_rule 实参传入 _non_inplace_oper_at）
    """
    index = result_tensor.parms[0]

    # result_tensor就是赋值结果的副本
    # 将 _grad_value 按索引取值就是要传递给所依赖的右值的梯度，索引取值结果是一个新张量
    right_var_grad = result_tensor._grad_value[index]
    left_var_grad = result_tensor._grad_value.setat(index,0.)
    return (left_var_grad, right_var_grad)


def _setat_inplace_grad(result_tensor:TN):
    """原地 setat_ 的反向：右值梯度取索引位置的梯度，并把 self 该位置梯度清零。

    调用者: setat_（作为 grad_rule 实参传入 _inplace_oper_at_）
    """
    index = result_tensor._inplace_ops[0].stored_index

    # result_tensor就是原地赋值时的self
    # 将self._grad_value按索引取值就是要传递给所依赖的右值的梯度，索引取值结果是一个新张量
    right_var_grad = result_tensor._grad_value[index]
    # 由于self的在原地索引赋值时index位置已被右值覆盖，
    # 所以要将self._grad_value中索引位置的梯度设为0
    clone_self_grad = result_tensor._grad_value.setat(index,0.)
    result_tensor._grad_value = clone_self_grad

    return right_var_grad


def _addat_grad(result_tensor:TN):
    """非原地 addat 的融合反向：返回 (左值梯度, 右值梯度)。

    左值：加法不改变左值梯度，原样返回；
    右值：取 self 梯度在索引位置的取值。

    调用者: addat（作为 grad_rule 实参传入 _non_inplace_oper_at）
    """
    index = result_tensor.parms[0]

    # result_tensor就是赋值结果的副本
    # 将 _grad_value 按索引取值就是要传递给所依赖的右值的梯度，索引取值结果是一个新张量
    right_var_grad = result_tensor._grad_value[index]
    return (result_tensor._grad_value, right_var_grad)


def _addat_inplace_grad(result_tensor:TN):
    """原地 addat_ 的反向：右值梯度即 self 梯度在索引位置的取值。

    调用者: addat_（作为 grad_rule 实参传入 _inplace_oper_at_）；函数名亦登记在 _INPLACE_AT_METHODS，供 setitem 重放
    """
    index = result_tensor._inplace_ops[0].stored_index

    # result_tensor就是原地赋值时的self
    # 将self._grad_value按索引取值就是要传递给所依赖的右值的梯度，索引取值结果是一个新张量
    right_var_grad = result_tensor._grad_value[index]
    return right_var_grad


def _subat_grad(result_tensor:TN):
    """非原地 subat 的融合反向：返回 (左值梯度, 右值梯度)。

    左值：减法不改变左值梯度，原样返回；
    右值：索引位置梯度取负。

    调用者: subat（作为 grad_rule 实参传入 _non_inplace_oper_at）
    """
    index = result_tensor.parms[0]

    # result_tensor就是赋值结果的副本
    # 将 _grad_value 按索引取值就是要传递给所依赖的右值的梯度，索引取值结果是一个新张量
    right_var_grad = -result_tensor._grad_value[index]
    return (result_tensor._grad_value, right_var_grad)


def _subat_inplace_grad(result_tensor:TN):
    """原地 subat_ 的反向：右值梯度为 self 在索引位置梯度的相反数。

    调用者: subat_（作为 grad_rule 实参传入 _inplace_oper_at_）；函数名亦登记在 _INPLACE_AT_METHODS，供 setitem 重放
    """
    index = result_tensor._inplace_ops[0].stored_index

    # result_tensor就是原地赋值时的self
    # 将self._grad_value按索引取值就是要传递给所依赖的右值的梯度，索引取值结果是一个新张量
    right_var_grad = -result_tensor._grad_value[index]
    return right_var_grad


def _mulat_grad(result_tensor:TN):
    """非原地 mulat 的融合反向：返回 (左值梯度, 右值梯度)。

    左值：用右值的共轭修正索引位置的梯度；
    右值：索引位置梯度乘左值（按索引）的共轭。

    调用者: mulat（作为 grad_rule 实参传入 _non_inplace_oper_at）
    """
    left_var = result_tensor.fromvars[0]
    right_var = result_tensor.fromvars[1]
    index = result_tensor.parms[0]

    left_var_grad = result_tensor._grad_value[index] * right_var.conj()
    right_var_grad = result_tensor._grad_value[index] * left_var[index].conj()
    return (result_tensor._grad_value.setat(index,left_var_grad), right_var_grad)


def _mulat_inplace_grad(result_tensor: TN):
    """原地 mulat_ 的反向：由备份的左值算出右值梯度，并把 self 索引位置梯度替换为左值梯度。

    调用者: mulat_（作为 grad_rule 实参传入 _inplace_oper_at_）；函数名亦登记在 _INPLACE_AT_METHODS 与 _LEFT_VALUE_BACKWARD_NAMES
    """
    record = result_tensor._inplace_ops[0]
    right_var = result_tensor.fromvars[0]
    left_var = record.left_copy
    index = record.stored_index

    result_grad = result_tensor._grad_value[index]
    right_var_grad = result_grad * left_var.conj()

    # 由于self的在原地索引赋值时index位置已被乘积值覆盖，
    # 所以要将self._grad_value中索引位置的梯度设为左值的梯度
    left_var_grad = result_grad * right_var.conj()
    clone_self_grad = result_tensor._grad_value.setat(index,left_var_grad)
    result_tensor._grad_value = clone_self_grad

    return right_var_grad


def _neg_inplace_data(array, index)->None:
    """neg_ 的数据改写函数：对指定索引位置的元素取负。

    调用者: neg_（作为 data_func 实参传入 _inplace_unary_at_）
    """
    array[index] = -array[index]


def _abs_inplace_data(array, index)->None:
    """abs_ 的数据改写函数：对指定索引位置的元素取绝对值。

    不能用内置 abs()：本模块的 abs(TN) 会遮蔽内置名，数组会被包成 TN；
    ndarray/numpy 标量/cupy 数组都实现了 __abs__，可直接调用以保持设备无关。

    调用者: abs_（作为 data_func 实参传入 _inplace_unary_at_）
    """
    array[index] = array[index].__abs__()


def _placeholder_zero_grad(result_tensor: TN) -> TN:
    """为不传播梯度的一元原地运算返回0标量梯度，供占位变量接收。

    调用者: _negat_inplace_grad、_absat_inplace_grad
    """
    placeholder = result_tensor.fromvars[0]
    return TN(placeholder._get_array_lib().zeros((), dtype=placeholder.dtype))


def _negat_inplace_grad(result_tensor: TN) -> TN:
    """neg_ 的反向：把流向自身历史算子的梯度取负（d(-x)/dx = -1）。

    调用者: neg_（作为 grad_rule 实参传入 _inplace_unary_at_）
    """
    index = result_tensor._inplace_ops[0].stored_index

    result_grad = result_tensor._grad_value[index]
    result_tensor._grad_value = result_tensor._grad_value.setat(index, -result_grad)

    return _placeholder_zero_grad(result_tensor)


def _absat_inplace_grad(result_tensor: TN) -> TN:
    """abs_ 的反向：d|x|/dx = sign(x)，符号取自操作前的左值副本。

    调用者: abs_（作为 grad_rule 实参传入 _inplace_unary_at_）
    """
    record = result_tensor._inplace_ops[0]
    left_copy = record.left_copy
    index = record.stored_index

    result_grad = result_tensor._grad_value[index]
    left_grad = result_grad * left_copy._get_array_lib().sign(left_copy._array)
    result_tensor._grad_value = result_tensor._grad_value.setat(index, left_grad)

    return _placeholder_zero_grad(result_tensor)


def _divat_grad(result_tensor:TN):
    """非原地 divat 的融合反向：返回 (左值梯度, 右值梯度)。

    左值：索引位置梯度除以右值的共轭；
    右值：-x/right 的共轭乘以索引位置梯度。

    调用者: divat（作为 grad_rule 实参传入 _non_inplace_oper_at）
    """
    right_var = result_tensor.fromvars[1]
    index = result_tensor.parms[0]

    left_var_grad = result_tensor._grad_value[index] / right_var.conj()
    right_var_grad = result_tensor._grad_value[index] * (-result_tensor[index]/right_var).conj()
    return (result_tensor._grad_value.setat(index,left_var_grad), right_var_grad)


def _divat_inplace_grad(result_tensor: TN):
    """原地 divat_ 的反向：算出右值梯度，并把 self 索引位置梯度设为左值梯度。

    调用者: divat_（作为 grad_rule 实参传入 _inplace_oper_at_）；函数名亦登记在 _INPLACE_AT_METHODS 与 _LEFT_VALUE_BACKWARD_NAMES
    """
    right_var = result_tensor.fromvars[0]
    index = result_tensor._inplace_ops[0].stored_index

    result_grad = result_tensor._grad_value[index]
    right_var_grad = result_grad * (-result_tensor[index]/right_var).conj()

    # 由于self的在原地索引赋值时index位置已被覆盖，
    # 所以要将self._grad_value中索引位置的梯度设为左值的梯度
    left_var_grad = result_grad / right_var.conj()
    clone_self_grad = result_tensor._grad_value.setat(index,left_var_grad)
    result_tensor._grad_value = clone_self_grad

    return right_var_grad


def _modat_grad(result_tensor:TN):
    """非原地 modat 的融合反向：返回 (左值梯度, 右值梯度)。

    左值：取余对左值的导数为 1，梯度原样返回；
    右值：-floor(left/right) 乘以索引位置梯度。

    调用者: modat（作为 grad_rule 实参传入 _non_inplace_oper_at）
    """
    left_var = result_tensor.fromvars[0]
    right_var = result_tensor.fromvars[1]
    index = result_tensor.parms[0]

    # d(left % right)/d(right) = -floor(left/right)
    right_var_grad = -result_tensor._grad_value[index] * floor(left_var[index] / right_var)
    # d(left % right)/d(left) = 1，结果梯度即为左值梯度
    return (result_tensor._grad_value, right_var_grad)


def _modat_inplace_grad(result_tensor: TN):
    """原地 modat_ 的反向：算出右值梯度，并把 self 索引位置梯度设为左值梯度。

    调用者: modat_（作为 grad_rule 实参传入 _inplace_oper_at_）；函数名亦登记在 _INPLACE_AT_METHODS 与 _LEFT_VALUE_BACKWARD_NAMES
    """
    record = result_tensor._inplace_ops[0]
    right_var = result_tensor.fromvars[0]
    left_var = record.left_copy
    index = record.stored_index

    result_grad = result_tensor._grad_value[index]
    right_var_grad = -result_grad * floor(left_var / right_var)

    # 由于self在原地索引赋值时index位置已被取余结果覆盖，
    # 所以要将self._grad_value中索引位置的梯度设为左值的梯度（导数恒为1）
    clone_self_grad = result_tensor._grad_value.setat(index, result_grad)
    result_tensor._grad_value = clone_self_grad

    return right_var_grad


def _powat_grad(result_tensor:TN):
    """非原地 powat 的融合反向：返回 (左值梯度, 右值梯度)。

    左值：right*left**(right-1) 的共轭乘以索引位置梯度；
    右值：result*log(left) 的共轭乘以索引位置梯度。

    调用者: powat（作为 grad_rule 实参传入 _non_inplace_oper_at）
    """
    left_var = result_tensor.fromvars[0]
    right_var = result_tensor.fromvars[1]
    index = result_tensor.parms[0]

    result_grad = result_tensor._grad_value[index]
    left_var_grad = result_grad * (right_var * (left_var[index] ** (right_var - 1.))).conj()
    right_var_grad = result_grad * result_tensor[index].conj() * log(left_var[index].conj())
    return (result_tensor._grad_value.setat(index,left_var_grad), right_var_grad)


def _powat_inplace_grad(result_tensor: TN):
    """原地 powat_ 的反向：算出右值梯度，并把 self 索引位置梯度设为左值梯度。

    调用者: powat_（作为 grad_rule 实参传入 _inplace_oper_at_）；函数名亦登记在 _INPLACE_AT_METHODS 与 _LEFT_VALUE_BACKWARD_NAMES
    """
    record = result_tensor._inplace_ops[0]
    right_var = result_tensor.fromvars[0]
    left_var = record.left_copy
    index = record.stored_index

    result_grad = result_tensor._grad_value[index]
    right_var_grad = result_grad * result_tensor[index].conj() * log(left_var.conj())

    # 由于self的在原地索引赋值时index位置已被乘积值覆盖，
    # 所以要将self._grad_value中索引位置的梯度设为左值的梯度
    left_var_grad = result_grad * (right_var * (left_var ** (right_var - 1.))).conj()
    clone_self_grad = result_tensor._grad_value.setat(index,left_var_grad)
    result_tensor._grad_value = clone_self_grad

    return right_var_grad


def _floordivat_grad(result_tensor:TN):
    """非原地 floordivat 的融合反向：向下取整除法不可导，有来源需要梯度时抛 RuntimeError。

    调用者: floordivat（作为 grad_rule 实参传入 _non_inplace_oper_at）
    """
    left_var = result_tensor.fromvars[0]
    right_var = result_tensor.fromvars[1]
    if ((isinstance(left_var, TN) and left_var.requires_grad)
            or (isinstance(right_var, TN) and right_var.requires_grad)):
        raise RuntimeError('derivative for aten::floor_divide is not implemented')
    return (None, None)


def _floordivat_inplace_grad(result_tensor:TN):
    """原地 floordivat_ 的反向：向下取整除法不可导，直接抛 RuntimeError（与 torch 一致）。

    调用者: floordivat_（作为 grad_rule 实参传入 _inplace_oper_at_）
    """
    raise RuntimeError('derivative for aten::floor_divide is not implemented')


def _adjust_right_val(input:TN, target_data, val):
    """将右值转为与目标张量一致的 dtype/device，并调整形状以匹配索引选中的区域。

    调用者: _inplace_oper_at_、_non_inplace_oper_at
    """
    # 若 val 已是 TN，则校验设备一致性、按需转换 dtype 后直接复用，避免重复创建
    if isinstance(val, TN):
        right_val = val
        # 检查设备一致性
        if not _same_device(right_val, input):
            raise ValueError(f"Right value device {right_val.device} does not match target device {input.device}")
        # 检查并转换dtype
        if right_val.dtype != input.dtype:
            right_val = right_val.type(input.dtype)
    else:
        right_val = as_tensor(val, dtype=input.dtype, device=input.device)

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


def _compose_inplace_grad(input:TN, grad_rule):
    """把本次原地操作的融合梯度规则叠加到 input 已有的 _grad_rule 上。

    原地操作只能修改 input 自身，因此每做一次操作就要在它的计算图上追加一条依赖。
    新规则先计算本次操作的梯度，再调用旧规则计算历史来源的梯度，并把二者按
    “当前 fromvars 顺序”拼接返回。

    必须在把本次操作插入 fromvars/_inplace_ops 之前调用：旧规则是按它记录时的
    位置计算梯度的，而本次操作会把新条目插到首部使位置整体后移。fused 内部因此
    临时去掉首条 fromvars 与首条 InplaceRecord 后再调用旧规则，调用结束后恢复；
    这里按 result.fromvars[1:] 现取历史记录，而不是捕获调用时刻的张量对象，
    这样在 JVP 的影子节点上调用时仍使用影子自己的来源，不会泄漏到真实计算图。

    parms 不再参与原地操作的插入与剥离：每个原地操作所需的左值副本与索引都由
    InplaceRecord 自带，parms 保持算子自身的静态参数含义。

    调用者: _inplace_unary_at_、_inplace_oper_at_
    """
    old_rule = input._grad_rule

    def fused(result:TN):
        # 先按当前记录计算本次原地操作的梯度（首条记录即本操作的上下文）
        first = grad_rule(result)

        # 临时去掉首条记录，让旧规则按其记录时的位置计算
        new_fromvars = result.fromvars
        new_inplace_ops = result._inplace_ops
        result.fromvars = new_fromvars[1:]
        result._inplace_ops = new_inplace_ops[1:]
        try:
            rest = old_rule(result) if old_rule is not None else ()
        finally:
            result.fromvars = new_fromvars
            result._inplace_ops = new_inplace_ops

        return (first,) + tuple(rest)

    input._grad_rule = fused


def _inplace_unary_at_(input:TN, index, data_func, grad_rule):
    """在指定索引位置对 input 执行一元原地运算，并记录反向传播所需的依赖。

    与 _inplace_oper_at_ 的区别：一元运算没有右值，data_func 直接依据左值
    改写数据，因此梯度只需修正 input 自身历史算子的梯度（由 grad_rule
    完成），不用记录右值依赖；占位变量仅用于承载该梯度规则。

    参数:
        input: 被操作的张量
        index: 目标索引；空元组 () 表示整个张量
        data_func: 执行函数，签名 (数组, 索引)，直接在数组上改写数据
        grad_rule: 该操作对应的融合梯度规则

    视图重定向：若 input 是视图，先合并视图索引与 index，再在 input._base 上执行，
    保证写入落到真实存储；返回值仍是视图对象本身。

    调用者: abs_、neg_（另在视图重定向时递归调用自身）
    """
    # 将索引规范化为numpy索引
    index_val = _convert_TNindex_to_numpy(index)

    # 检查是否为视图对象
    if hasattr(input, '_is_view') and input._is_view:
        combined_index = merge_indices(input, input._view_index, index_val)
        _inplace_unary_at_(input._base, combined_index, data_func, grad_rule)
        return input

    # 是否记录计算图：no_grad 下既不记录依赖，也不改动 requires_grad
    need_grad = is_grad_enabled() and input.requires_grad

    # 反向需要操作前的左值（如 abs 的符号），故在改写数据前备份一份独立副本
    target_data = input._array[index_val]
    if need_grad:
        if target_data.base is not None:
            target_data = target_data.copy()  # 复制共享内存的视图，确保独立性
        target_copy = TN(target_data)

    data_func(input._array, index_val)  # 执行原地操作

    if need_grad:
        # 一元原地运算没有右值，用不依赖任何输入的占位变量承载梯度规则；
        # 插入位置与 _inplace_oper_at_ 一致，保证“后做的原地操作先处理梯度”
        placeholder = zeros((), dtype=input.dtype, device=input.device, requires_grad=True)
        input.requires_grad = True
        _compose_inplace_grad(input, grad_rule)
        input.fromvars = (placeholder,) + input.fromvars
        input._inplace_ops = (
            InplaceRecord(None, placeholder, index_val, target_copy),
        ) + input._inplace_ops

    return input


def _inplace_oper_at_(input:TN, index, val, oper_func, grad_rule, vector_func=None):
    """在指定索引位置对 input 执行原地运算，并记录反向传播所需的依赖。

    参数:
        input: 被操作的张量
        index: 目标索引；空元组 () 表示整个张量
        val: 右值，标量或张量
        oper_func: 通用执行函数，签名 (数组, 索引, 右值数组)；
                   必须是 ufunc.at 一类支持重复索引累加的实现
        grad_rule: 该操作对应的融合梯度规则
        vector_func: 可选的全量索引加速函数，签名 (数组, 右值数组)；
                     仅当 index 为 () 时使用，为 None 时全部走 oper_func

    视图重定向：若 input 是视图，先合并视图索引与 index，再在 input._base 上执行，
    保证写入落到真实存储；返回值仍是视图对象本身，调用方无需区分。

    计算图记录顺序“最新在前”：每次原地操作把右值插入 fromvars 首部，并把索引、
    左值副本和重放信息合并为一条 InplaceRecord 插入 _inplace_ops 首部，同时把
    本次操作的梯度规则组合进 _grad_rule；反向传播按逆序处理，从而实现
    “后做的原地操作先处理梯度”。parms 不再记录原地操作上下文。

    调用者: setat_、addat_、subat_、mulat_、divat_、powat_、floordivat_、modat_
    （另在视图重定向时递归调用自身）
    """
    # 将索引规范化为numpy索引
    index_val = _convert_TNindex_to_numpy(index)

    # 检查是否为视图对象
    if hasattr(input, '_is_view') and input._is_view:
        # 将当前操作索引与视图索引合并
        combined_index = merge_indices(input, input._view_index, index_val)

        # input是视图时，直接在原张量input._base上执行操作
        _inplace_oper_at_(input._base, combined_index, val, oper_func, grad_rule, vector_func)

        # 返回视图对象本身
        return input

    # 将右值形状、维度转化为和索引值相匹配的形式
    target_data = input._array[index_val]
    right_val = _adjust_right_val(input, target_data, val)

    # 是否记录计算图：no_grad 下既不记录依赖，也不改动 requires_grad，
    # 与 torch 保持一致（梯度开关不应永久改变张量的属性）
    need_grad = is_grad_enabled() and (input.requires_grad or right_val.requires_grad)

    # target 的来源有两种：
    # 1、基本切片返回视图，与input共享内存
    # 2、整数/花式索引返回副本或numpy标量，本就不共享，但形状/维度信息仍需保留
    # 对于共享内存的视图，必须在改动input前备份一份独立且无依赖的副本：
    # 因为反向传播要用到左值，而原地操作后 input 对应位置的数据已被覆盖
    # 只有 mul/div/pow 的反向需要左值（_LEFT_VALUE_BACKWARD_NAMES），
    # add/sub/set 的反向只用索引，可省去这次整块内存拷贝
    if need_grad and getattr(grad_rule, '__name__', '') in _LEFT_VALUE_BACKWARD_NAMES:
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
        vector_func(input._array, right_val._array)
    else:
        oper_func(input._array, index_val, right_val._array)  # 执行原地操作

    if need_grad:
        # 原地赋值时，须在input的现有计算图上增加依赖关系，注意:
        # 1、只能记录右值的依赖关系，不能记录input，避免依赖循环
        # 2、本操作的梯度规则同时给出右值梯度与历史来源梯度；右值requires_grad=False时，
        #    复制一个detach副本并置True，使该来源能接收梯度
        # 3、原地赋值依赖关系需要优先处理，须插入到fromvars元组前面
        # 4、多次原地赋值时，遵循后做的操作先处理梯度的原则，从0位置插入右值与索引
        # 5、若右值依赖 input（如 t = a[[0,4]]; a[1:3] = t），这里会形成环；
        #    该情况由 __setitem__ 的 _assign_without_cycle 提前拦截，不会走到这里
        if right_val.requires_grad == False:
            # 需要一个用于传播梯度的右值变量，但又不要改动右值的属性，detach一个新变量是一个经济、可行的办法
            right_val = right_val.detach().requires_grad_(True)

        input.requires_grad = True
        _compose_inplace_grad(input, grad_rule)
        input.fromvars = (right_val,) + input.fromvars
        input._inplace_ops = (
            InplaceRecord(
                _INPLACE_AT_METHODS.get(grad_rule.__name__),
                input.fromvars[0],
                index_val,
                target_copy,
            ),
        ) + input._inplace_ops

    return input


def _non_inplace_oper_at(input:TN, index, val, oper_func, grad_rule):
    """在 input 的副本上按索引执行运算并返回新张量，input 本身不变。

    调用者: setat、addat、subat、mulat、divat、powat、floordivat、modat
    """
    # 将索引规范化为numpy索引
    index_val = _convert_TNindex_to_numpy(index)

    # 将右值形状、维度转化为和索引值相匹配的形式
    target = input._array[index_val]
    right_val = _adjust_right_val(input, target, val)

    # 新建一个input张量的独立副本，用于存储结果
    ret = input.copy()
    oper_func(ret._array, index_val, right_val._array)

    ret.requires_grad = (is_grad_enabled() and (input.requires_grad or right_val.requires_grad))

    if ret.requires_grad:
        ret.fromvars = (input,right_val)
        ret.parms = (index_val,)
        ret._grad_rule = grad_rule

    return ret


# 原地/非原地成对接口：xxxat_ 直接在 input 上执行并记录依赖、返回 input；
# xxxat 在副本上执行、返回新张量，input 不变。
# 每对接口都提供两种执行函数，由 _inplace_oper_at_ 按索引选择：
# - oper_func：通用路径，基于 ufunc.at，保证花式索引“重复索引累加”的语义；
# - vector_func：仅用于 index 为 () 的全量操作，走向量化原地运算符
#   （即 _iadd_full 等），避免 ufunc.at 的逐元素慢路径。


@api_registry.register
def setat_(input:TN, index, val):
    """原地执行 input[index] = val（覆盖写），返回 input。"""
    def _set_numpy_item(numpy_arr, index, right_numpy_arr):
        numpy_arr[index] = right_numpy_arr
        return numpy_arr

    return _inplace_oper_at_(input, index, val, _set_numpy_item, _setat_inplace_grad, _iset_full)


@api_registry.register
def setat(input:TN, index, val):
    """非原地执行 input[index] = val，返回新张量。"""
    def _set_numpy_item(numpy_arr, index, right_numpy_arr):
        numpy_arr[index] = right_numpy_arr
        return numpy_arr

    return _non_inplace_oper_at(input, index, val, _set_numpy_item, _setat_grad)


@api_registry.register
def addat_(input:TN, index, val):
    """原地执行 input[index] += val（重复索引累加），返回 input。"""
    arrlib = input._get_array_lib()
    return _inplace_oper_at_(input, index, val, arrlib.add.at, _addat_inplace_grad, _iadd_full)


@api_registry.register
def addat(input:TN, index, val):
    """非原地执行 input[index] += val，返回新张量。"""
    arrlib = input._get_array_lib()
    return _non_inplace_oper_at(input, index, val, arrlib.add.at, _addat_grad)


@api_registry.register
def subat_(input:TN, index, val):
    """原地执行 input[index] -= val（重复索引累加），返回 input。

    cupy 的 subtract.at 支持不完整，故 cupy 分支改用切片原地运算。
    """
    arrlib = input._get_array_lib()
    if arrlib == cp:
        def _subat_numpy_item(numpy_arr, index, right_numpy_arr):
            numpy_arr[index] -= right_numpy_arr
            return numpy_arr
        oper_func = _subat_numpy_item
    else:
        oper_func = np.subtract.at
    return _inplace_oper_at_(input, index, val, oper_func, _subat_inplace_grad, _isub_full)


@api_registry.register
def subat(input:TN, index, val):
    """非原地执行 input[index] -= val，返回新张量。"""
    arrlib = input._get_array_lib()
    if arrlib == cp:
        def _subat_numpy_item(numpy_arr, index, right_numpy_arr):
            numpy_arr[index] -= right_numpy_arr
            return numpy_arr
        oper_func = _subat_numpy_item
    else:
        oper_func = np.subtract.at
    return _non_inplace_oper_at(input, index, val, oper_func, _subat_grad)


@api_registry.register
def mulat_(input:TN, index, val):
    """原地执行 input[index] *= val（重复索引累加），返回 input。

    反向传播需要操作前的左值，target_copy 会在这里被备份。
    """
    arrlib = input._get_array_lib()
    if arrlib == cp:
        def _mulat_numpy_item(numpy_arr, index, right_numpy_arr):
            numpy_arr[index] *= right_numpy_arr
            return numpy_arr
        oper_func = _mulat_numpy_item
    else:
        oper_func = np.multiply.at

    return _inplace_oper_at_(input, index, val, oper_func, _mulat_inplace_grad, _imul_full)


@api_registry.register
def mulat(input:TN, index, val):
    """非原地执行 input[index] *= val，返回新张量。"""
    arrlib = input._get_array_lib()
    if arrlib == cp:
        def _mulat_numpy_item(numpy_arr, index, right_numpy_arr):
            numpy_arr[index] *= right_numpy_arr
            return numpy_arr
        oper_func = _mulat_numpy_item
    else:
        oper_func = np.multiply.at

    return _non_inplace_oper_at(input, index, val, oper_func, _mulat_grad)


@api_registry.register
def divat_(input:TN, index, val):
    """原地执行 input[index] /= val（重复索引累加），返回 input。

    反向传播需要操作前的左值，target_copy 会在这里被备份。
    """
    arrlib = input._get_array_lib()
    if arrlib == cp:
        def _divat_numpy_item(numpy_arr, index, right_numpy_arr):
            numpy_arr[index] /= right_numpy_arr
            return numpy_arr
        oper_func = _divat_numpy_item
    else:
        oper_func = np.divide.at
    return _inplace_oper_at_(input, index, val, oper_func, _divat_inplace_grad, _idiv_full)


@api_registry.register
def divat(input:TN, index, val):
    """非原地执行 input[index] /= val，返回新张量。"""
    arrlib = input._get_array_lib()
    if arrlib == cp:
        def _divat_numpy_item(numpy_arr, index, right_numpy_arr):
            numpy_arr[index] /= right_numpy_arr
            return numpy_arr
        oper_func = _divat_numpy_item
    else:
        oper_func = np.divide.at
    return _non_inplace_oper_at(input, index, val, oper_func, _divat_grad)


@api_registry.register
def powat_(input:TN, index, val):
    """原地执行 input[index] **= val（重复索引累加），返回 input。

    反向传播需要操作前的左值，target_copy 会在这里被备份。
    """
    arrlib = input._get_array_lib()
    if arrlib == cp:
        def _powat_numpy_item(numpy_arr, index, right_numpy_arr):
            numpy_arr[index] **= right_numpy_arr
            return numpy_arr
        oper_func = _powat_numpy_item
    else:
        oper_func = np.power.at
    return _inplace_oper_at_(input, index, val, oper_func, _powat_inplace_grad, _ipow_full)


@api_registry.register
def powat(input:TN, index, val):
    """非原地执行 input[index] **= val，返回新张量。"""
    arrlib = input._get_array_lib()
    if arrlib == cp:
        def _powat_numpy_item(numpy_arr, index, right_numpy_arr):
            numpy_arr[index] **= right_numpy_arr
            return numpy_arr
        oper_func = _powat_numpy_item
    else:
        oper_func = np.power.at
    return _non_inplace_oper_at(input, index, val, oper_func, _powat_grad)


@api_registry.register
def floordivat_(input:TN, index, val):
    """原地执行 input[index] //= val（重复索引累加），返回 input。

    向下取整除法不可导，反向传播与 torch 一致，在 backward 时抛出异常，
    因此无需备份左值。
    """
    arrlib = input._get_array_lib()
    if arrlib == cp:
        def _floordivat_numpy_item(numpy_arr, index, right_numpy_arr):
            numpy_arr[index] //= right_numpy_arr
            return numpy_arr
        oper_func = _floordivat_numpy_item
    else:
        oper_func = np.floor_divide.at
    return _inplace_oper_at_(input, index, val, oper_func,
                             _floordivat_inplace_grad, _ifloordiv_full)


@api_registry.register
def floordivat(input:TN, index, val):
    """非原地执行 input[index] //= val，返回新张量。"""
    arrlib = input._get_array_lib()
    if arrlib == cp:
        def _floordivat_numpy_item(numpy_arr, index, right_numpy_arr):
            numpy_arr[index] //= right_numpy_arr
            return numpy_arr
        oper_func = _floordivat_numpy_item
    else:
        oper_func = np.floor_divide.at
    return _non_inplace_oper_at(input, index, val, oper_func, _floordivat_grad)


@api_registry.register
def modat_(input:TN, index, val):
    """原地执行 input[index] %= val（重复索引累加），返回 input。

    反向传播需要操作前的左值，target_copy 会在这里被备份。
    """
    arrlib = input._get_array_lib()
    if arrlib == cp:
        def _modat_numpy_item(numpy_arr, index, right_numpy_arr):
            numpy_arr[index] %= right_numpy_arr
            return numpy_arr
        oper_func = _modat_numpy_item
    else:
        oper_func = np.remainder.at
    return _inplace_oper_at_(input, index, val, oper_func, _modat_inplace_grad, _imod_full)


@api_registry.register
def modat(input:TN, index, val):
    """非原地执行 input[index] %= val，返回新张量。"""
    arrlib = input._get_array_lib()
    if arrlib == cp:
        def _modat_numpy_item(numpy_arr, index, right_numpy_arr):
            numpy_arr[index] %= right_numpy_arr
            return numpy_arr
        oper_func = _modat_numpy_item
    else:
        oper_func = np.remainder.at
    return _non_inplace_oper_at(input, index, val, oper_func, _modat_grad)


@api_registry.register
def abs_(x:TN)->TN:
    """
    就地取绝对值，等价于 torch.abs_。

    直接修改输入张量的数据并返回它本身。布尔张量、复数张量不支持（与 torch 一致）。

    Args:
        x (TN): 输入张量

    Returns:
        TN: 修改后的输入张量（与输入是同一对象）

    Raises:
        RuntimeError: 如果输入是需要梯度的叶子节点，或复数类型（与 torch 一致）
        NotImplementedError: 如果输入是布尔类型

    Examples:
        >>> a = tensor([-1., 2., -3.])
        >>> abs_(a)  # a 变为 [1., 2., 3.]，返回 a 本身
    """
    _require_tn(x, 'abs_')
    if is_grad_enabled() and x.is_leaf and x.requires_grad:
        raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation.')

    # torch 的 abs_ 不支持布尔与复数类型
    if np.issubdtype(x.dtype, np.bool_):
        raise NotImplementedError("abs_ is not implemented for bool tensors")
    if np.issubdtype(x.dtype, np.complexfloating):
        raise RuntimeError("In-place abs is not supported for complex tensors.")

    return _inplace_unary_at_(x, (), _abs_inplace_data, _absat_inplace_grad)


@api_registry.register
def neg_(x)->TN:
    """
    就地取负，等价于 torch.neg_。

    直接修改输入张量的数据并返回它本身。布尔张量不支持取负，
    无符号整型按位宽回绕并保持 dtype（如 uint8 的 -1 得到 255）。

    Args:
        x (TN): 输入张量

    Returns:
        TN: 修改后的输入张量（与输入是同一对象）

    Raises:
        RuntimeError: 如果输入是需要梯度的叶子节点
        NotImplementedError: 如果输入是布尔类型

    Examples:
        >>> a = tensor([1., -2.])
        >>> neg_(a)  # a 变为 [-1., 2.]，返回 a 本身
    """
    _require_tn(x, 'neg_')
    if is_grad_enabled() and x.is_leaf and x.requires_grad:
        raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation.')

    # 布尔张量取负没有意义，torch 同样不支持；取反请用 ~ 或 bitwise_not()
    if np.issubdtype(x.dtype, np.bool_):
        raise NotImplementedError(
            "Negation, the `-` operator, on a bool tensor is not supported. "
            "If you are trying to invert a mask, use the `~` or `bitwise_not()` operator instead."
        )

    # 一元原地运算没有右值，传入空索引表示对整个视图操作（视图索引由
    # _inplace_unary_at_ 自动合并），无符号整型取负由底层按位宽回绕
    return _inplace_unary_at_(x, (), _neg_inplace_data, _negat_inplace_grad)


# ===========================================================================
# 8. 索引读写（index_select / masked_select / take / gather / scatter / setitem）
# ===========================================================================


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
        stack.extend(var for var in item.fromvars if isinstance(var, TN))
    return False


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


def _collect_replay_ops(input, val, index_val):
    """收集 val 上记录的原地操作，供 __setitem__ 按时间顺序重放到 self。

    场景：a[[0,4]] += 1 时 __getitem__ 返回副本 val，原地操作只改了 val，
    因此 self 需要“重演”这些操作才能得到相同数据。重演等价于直接在 self 上
    执行 a[idx] op= 右值，既不改动副本、也不会产生 self 与 val 的循环依赖。

    两个前提缺一不可，否则重演结果与 val 的实际数据不一致：
    1. val 必须取自同一索引（val._view_index == index_val）；
    2. 每条原地操作都必须作用于整个副本（stored_index 为 ()）。
        若副本上做过局部原地操作（如 t[0] += 1），重演需要再合并一层索引，
        这里不做该推导，直接放弃重演。

    每条记录都是自包含的：重演只读 InplaceRecord，不再按下标访问
    fromvars/parms，因此 parms 可以始终只表示算子自身的静态参数。

    原地操作记录连续插入在 _inplace_ops 首部且“最新在前”，遇到不可重放记录
    （method 为 None）即停止；返回的列表同样保持“最新在前”，调用方需 reversed
    后按时间顺序执行。

    参数:
        val: __getitem__ 返回的副本张量。
        index_val: 本次赋值使用的索引（已规范化为 numpy 索引）。

    返回值:
        list: [(method_name, right_val), ...]，无可用重放时为空列表
    """
    if not _indexes_are_equal(getattr(val, '_view_index', None), index_val):
        return []

    ops = []
    for record in val._inplace_ops:
        if record.method is None:
            # 原地操作记录连续插入在 _inplace_ops 首部，遇到不可重放记录即可停止
            break
        # stored_index 为 () 表示该操作作用于整个副本；
        # 否则说明副本上还做过局部原地操作，重演需要合并索引，这里直接放弃
        if not (isinstance(record.stored_index, tuple)
                and len(record.stored_index) == 0):
            return []
        ops.append((record.method, record.right))
    return ops


def _assign_without_cycle(input, index, val):
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
    if isinstance(val, TN) and val.requires_grad and _graph_contains(val, input):
        val = val.detach()
    return setat_(input, index, val)


def _compute_to_direct_indices(input:TN, dim:int, index_data:np.ndarray):
    """
    计算用于setat、setat_、addat的直接索引，
    用在gather、scatter、scatter、scatter_scatter中将间接索引转换为直接索引
    
    参数:
        input: gather或scatter操作的张量，用于获取形状信息
        dim: 要操作的维度索引
        index_data: 包含索引映射信息的NumPy数组
    
    返回:
        一个元组，包含每个维度的索引数组，用于直接索引目标张量
    """

    # 获取self和index_data的形状
    target_shape = input.shape        
    index_shape = index_data.shape

    # 确保index和self的维度数相同
    # index张量的维度必须与读取数据的源张量和写入数据的目的张量的维度相同
    # 用于gather函数时，self是源张量，用于scatter函数时，self是目的张量
    if index_data.ndim != input.ndim:
        raise ValueError(f"Dimension mismatch between index and input: index has {index_data.ndim} dimensions, input has {input.ndim} dimensions")
    
    # 确保dim在有效范围内
    if dim >= len(target_shape) or dim < -len(target_shape):
        raise ValueError(f"Dimension {dim} out of range for tensor of shape {target_shape}")
 
    # 添加对负维度索引的支持
    if dim < 0:
        dim = dim + len(target_shape)
    
    # 检查索引值是否在有效范围内
    if np.any(index_data < 0) or np.any(index_data >= target_shape[dim]):
        raise IndexError(f"Index value out of range [0, {target_shape[dim]}-1]")

    # 使用np.indices一次性创建所有网格索引（使用numpy即可，索引值不依赖于设备）
    grid_indices = np.indices(index_shape)
    
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
        # 对于额外维度，直接使用全0索引（使用numpy即可）
        # 这与gather函数原始实现保持一致
        direct_indices.append(np.zeros(index_shape, dtype=np.int64))
    
    return tuple(direct_indices)
# end of _compute_direct_indices


@api_registry.register
def _setitem(input: TN, index, val):
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

    场景 B 的重放依据是副本上的 _inplace_ops：原地操作在 _inplace_oper_at_ /
    _inplace_unary_at_ 中按“最新在前”插入自包含的 InplaceRecord（方法名、
    右值、索引、左值副本）。method 为 None 表示该操作不可重放（setat_、
    floordivat_、abs_、neg_ 等），重放到此为止。

    重放依赖两个前提：副本取自同一索引、且所有原地操作都作用于整个副本；
    否则退化为按数据赋值（_assign_without_cycle）。这样数据始终正确，代价是
    不可重放的场景中“副本 → self”这条梯度路径会被切断。

    参数:
        input: 目标张量
        index: 索引，可以是切片、整数、数组等
        val: 右值，可能是张量或标量
    """
    # 1. 叶子节点检查：所有原地操作都需要这个检查
    # 叶子节点是计算图的起点，直接修改会破坏梯度追踪
    if is_grad_enabled() and input.is_leaf and input.requires_grad:
        raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation.')

    # 2. 快速路径：简单赋值（右值不是张量，没有 _is_view 属性）
    if not hasattr(val, '_is_view'):
        return setat_(input, index, val)

    # 3. 快速路径：视图但基础不是 self（来自其他张量的视图）
    if val._base is not input:
        return setat_(input, index, val)

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
            return input
        return _assign_without_cycle(input, index, val)

    # 场景 B：非视图对象（如整数数组索引 a[[0,4]] += 1）
    # 副本的 _inplace_ops 记录着它相对 self 做过的原地操作，按时间顺序全部重放
    # 即可完整还原数据，且不会在计算图中引入环路。重放的成立条件是：副本取自
    # 同一索引，且所有原地操作都作用于整个副本（见 _collect_replay_ops）。
    replay_ops = _collect_replay_ops(input, val, index_val)
    if replay_ops:
        # 记录按“最新在前”存放，重放必须按“最早在前”的顺序执行
        for op_name, right_val in reversed(replay_ops):
            getattr(input, op_name)(index_val, right_val)
        return input

    # 无法安全重放：按数据赋值。若副本的计算图依赖 self，需先断开环路，
    # 否则 self 与副本互相引用，反向传播无法收敛（梯度会整体丢失）。
    return _assign_without_cycle(input, index, val)


@api_registry.register
def index_select(input: TN, dim: int, index: TN, *, out = None) -> TN:
    """
    沿指定维度按索引选择元素，与 torch.index_select 语义一致

    参数:
        input: 源张量
        dim: 选择维度，支持负维度
        index: 一维整数索引张量，dtype 会被规范化为 int64
        out: 可选的输出张量，形状须与结果一致

    返回:
        形状为 input.shape 但 dim 维长度等于 index 长度的新张量

    注意: 当使用out参数时，不支持自动微分
    """
    if dim < -input.ndim or dim >= input.ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-input.ndim}, {input.ndim-1}], got {dim})"
        )
    if dim < 0:
        dim += input.ndim

    index = as_tensor(index, dtype=int64, device=input.device)
    if index.ndim != 1:
        raise IndexError("index_select(): Index tensor must be a vector")

    index_data = index._array
    size = input.shape[dim]
    arrlib = index._get_array_lib()
    if index_data.size and (bool(arrlib.any(index_data < 0)) or bool(arrlib.any(index_data >= size))):
        raise IndexError(f"index_select(): index out of range for dimension {dim} with size {size}")

    # 复用 __getitem__ 的花式索引：沿 dim 使用索引数组，其余维度全选
    indices = tuple(index_data if i == dim else slice(None) for i in range(input.ndim))
    result = input[indices]

    return _apply_out(out, result, (input,), "index_select")


@api_registry.register
def masked_select(input: TN, mask: TN, *, out = None) -> TN:
    """
    按布尔掩码选择元素，与 torch.masked_select 语义一致

    参数:
        input: 源张量
        mask: 布尔张量，形状须可广播到 input 的形状
        out: 可选的输出张量，形状须与结果一致

    返回:
        包含所有被选中元素的一维新张量

    注意: 当使用out参数时，不支持自动微分
    """
    mask = as_tensor(mask, device=input.device)
    if mask.dtype != np.bool_:
        raise TypeError(f"masked_select(): mask must be a boolean tensor, but got {mask.dtype}")

    # 先按广播规则扩展掩码，再复用 __getitem__ 的布尔索引
    result = input[mask._expand(input.shape)]

    return _apply_out(out, result, (input,), "masked_select")


@api_registry.register
def take(input: TN, index: TN) -> TN:
    """
    按扁平化后的下标选择元素，与 torch.take 语义一致

    参数:
        input: 源张量，按行主序视为一维数据
        index: 整数索引张量，形状任意，dtype 会被规范化为 int64

    返回:
        形状与 index 相同的新张量
    """
    index = as_tensor(index, dtype=int64, device=input.device)

    index_data = index._array
    numel = input._array.size
    arrlib = index._get_array_lib()
    # 与 torch.take 一致：支持负下标（按 Python 规则从末尾计数），仅校验取值范围
    if index_data.size and (
        bool(arrlib.any(index_data >= numel)) or bool(arrlib.any(index_data < -numel))
    ):
        raise IndexError(f"take(): index out of range for tensor with {numel} elements")

    # 展平后复用 __getitem__ 的整数数组索引，梯度经 reshape/getitem 自动回传
    result = input.reshape(-1)[index]

    return result


@api_registry.register
def index_put_(input, indices, values, accumulate: bool = False) -> 'TN':
    """
    按索引把 values 写入当前张量（原地操作），与 torch.Tensor.index_put_ 语义一致

    参数:
        indices: 索引元组，每个元素为整数索引张量或布尔掩码张量
        values: 写入值，可为标量、TN 或可转换为 TN 的数据，须可广播到被选中区域；
            当传入 TN 时，其 dtype 须与当前张量一致
        accumulate: 为 True 时按 addat_ 语义累加（重复索引累加），否则按 setat_ 覆盖

    返回:
        原地修改后的张量（input）
    """
    # 与 torch 一致：禁止对需要梯度的叶子张量执行原地写入
    if is_grad_enabled() and input.is_leaf and input.requires_grad:
        raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation.')
    # 与 torch 一致：写入值张量的 dtype 必须与目标张量一致
    if isinstance(values, TN) and values.dtype != input.dtype:
        raise RuntimeError(
            f"Index put requires the source and destination dtypes match, "
            f"got {input.dtype} for the destination and {values.dtype} for the source."
        )

    # 与 torch 的 tuple 约定保持一致，同时兼容列表形式
    if isinstance(indices, list):
        indices = tuple(indices)
    if accumulate:
        return addat_(input, indices, values)
    return setat_(input, indices, values)


@api_registry.register
def gather(input: TN, dim: int, index: TN, *, sparse_grad: bool = False, out = None) -> TN:
    """
    沿指定维度收集元素，与 torch.gather 语义一致

    参数:
        input: 源张量
        dim: 收集维度，支持负维度
        index: 索引张量，维数须与 input 相同，dtype 会被规范化为 int64
        sparse_grad: 是否产生稀疏梯度；Riemann 暂不支持，仅允许 False
        out: 可选的输出张量，形状须与结果一致

    返回:
        形状与 index 相同的新张量

    注意: 当使用out参数时，不支持自动微分
    """
    if sparse_grad:
        raise NotImplementedError("gather(): sparse_grad=True is not supported")

    arrlib = input._get_array_lib()
    index_data = index._array.astype(arrlib.int64)
    result = input[_compute_to_direct_indices(input, dim, index_data)]

    return _apply_out(out, result, (input,), "gather")


# 启用optimize路径（tensordot/BLAS加速）的最小操作数元素数阈值：
# 低于该值时einsum_path计算开销（约30us）超过BLAS收益
_EINSUM_OPTIMIZE_MIN_ELEMS = 4096


@api_registry.register
def scatter_(input, dim, index, src=None, *, value=None):
    """
    将值按照index指定的位置原地填充到张量self中
    
    参数:
        dim: 沿着哪个维度进行索引
        index: 索引张量
        src: 源张量，提供要填充的值，与value参数互斥；当为标量且value未提供时，视为value参数
        value: 标量值（命名参数），提供要填充的值，与src参数互斥
    
    返回:
        原地修改后的张量（input）
    """
    
    if is_grad_enabled() and input.is_leaf and input.requires_grad:
        raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation.')
    
    # 标记value是否来自非命名参数
    is_positional_value = False
    
    arrlib = input._get_array_lib()
    dev = input.device

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
    
    # 统一将index转换为int64的TN张量（已是int64且同设备的TN会直接返回原对象）
    index = as_tensor(index, dtype=int64, device=dev)
    
    # 确保dim是有效的维度
    if dim < 0:
        dim += input.ndim
    if dim < 0 or dim >= input.ndim:
        raise IndexError(f"Dimension out of range (expected to be in range of [{-input.ndim}, {input.ndim-1}], got {dim})")
    
    # 确保index和self的维度数相同
    if index.ndim != input.ndim:
        raise ValueError(f"Dimension mismatch between index and input: index has {index.ndim} dimensions, input has {input.ndim} dimensions")
    
    # 处理src参数
    if src is not None:
        # 将src转换为TN张量（如果它不是）
        if not isinstance(src, TN):
            src = as_tensor(src, dtype=input.dtype, device=dev)
        
        # 确保src和self的dtype相同
        if src.dtype != input.dtype:
            raise RuntimeError("scatter_(): Expected input.dtype to be equal to src.dtype")
        
        # 确保src和index的形状相同
        if src.shape != index.shape:
            raise ValueError(f"Shape mismatch between index and src: index has shape {index.shape}, src has shape {src.shape}")
        
        fill_value = src
    else:  # 使用value参数
        # 创建与index形状相同的张量，填充value值
        fill_value = full_like(index, value, dtype=input.dtype, device=dev)            

    # 计算直接索引        
    full_index = _compute_to_direct_indices(input, dim, index._array)
    
    # 使用setat进行赋值
    return setat_(input, full_index, fill_value)
# end of scatter_


@api_registry.register
def scatter(input, dim, index, src=None, *, value=None):
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
    
    arrlib = input._get_array_lib()
    dev = input.device

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
    
    # 统一将index转换为int64的TN张量（已是int64且同设备的TN会直接返回原对象）
    index = as_tensor(index, dtype=int64, device=dev)
    
    # 确保dim是有效的维度
    if dim < 0:
        dim += input.ndim
    if dim < 0 or dim >= input.ndim:
        raise IndexError(f"Dimension out of range (expected to be in range of [{-input.ndim}, {input.ndim-1}], got {dim})")
    
    # 确保index和self的维度数相同
    if index.ndim != input.ndim:
        raise ValueError(f"Dimension mismatch between index and input: index has {index.ndim} dimensions, input has {input.ndim} dimensions")
    
    # 处理src参数
    if src is not None:
        # 将src转换为TN张量（如果它不是）
        if not isinstance(src, TN):
            src = as_tensor(src, dtype=input.dtype, device=dev)
        
        # 确保src和self的dtype相同
        if src.dtype != input.dtype:
            raise RuntimeError("scatter(): Expected input.dtype to be equal to src.dtype")

        # 确保src和index的形状相同
        if src.shape != index.shape:
            raise ValueError(f"Shape mismatch between index and src: index has shape {index.shape}, src has shape {src.shape}")
        
        fill_value = src
    else:  # 使用value参数
        # 创建与index形状相同的张量，填充value值
        fill_value = full_like(index, value, dtype=input.dtype, device=dev)            

    # 计算直接索引
    full_index = _compute_to_direct_indices(input, dim, index._array)
    
    # 使用setat进行赋值
    return input.setat(full_index, fill_value)
# end of scatter


@api_registry.register
def scatter_add_(input, dim, index, src):
    """
    将src中的值按照index指定的位置原地累加到self张量中
    
    参数:
        dim: 沿着张量src的哪个维度进行索引
        index: 索引张量，形状必须与src（如果是张量）兼容，维度数必须与self相同
        src: 源张量或标量，提供要累加的值
    
    返回:
        原地修改后的张量（input）
    """

    if is_grad_enabled() and input.is_leaf and input.requires_grad:
        raise RuntimeError('a leaf Variable that requires grad is being used in an in-place operation.')
    
    dev = input.device
    
    # 统一将index转换为int64的TN张量（已是int64且同设备的TN会直接返回原对象）
    index = as_tensor(index, dtype=int64, device=dev)
    
    # 处理src为标量的情况        
    # src参数的类型检查：int、float、complex、numpy标量、TN标量
    if is_scalar(src):
        src = full_like(index, src, dtype=input.dtype, device=dev)

    if not isinstance(src,TN):
        src = as_tensor(src, dtype=input.dtype, device=dev)

    # 确保src和self的dtype相同
    if src.dtype != input.dtype:
        raise RuntimeError("scatter_add_(): Expected input.dtype to be equal to src.dtype")

    # 对于张量，确保形状与index相同
    if src.shape != index.shape:
        raise ValueError(f"Shape mismatch between index and src: index has shape {index.shape}, src has shape {src.shape}")
    
    # 确保dim是有效的维度
    if dim < 0:
        dim += input.ndim
    if dim < 0 or dim >= input.ndim:
        raise IndexError(f"Dimension out of range (expected to be in range of [{-input.ndim}, {input.ndim-1}], got {dim})")
    
    # 确保index和self的维度数相同
    if index.ndim != input.ndim:
        raise ValueError(f"Dimension mismatch between index and input: index has {index.ndim} dimensions, input has {input.ndim} dimensions")
            
    # 计算直接索引
    full_index = _compute_to_direct_indices(input, dim, index._array)
    
    # 直接调用原地操作的addat_函数
    return addat_(input, full_index, src)
# end of scatter_add_


@api_registry.register
def scatter_add(input, dim, index, src):
    """
    将src中的值按照index指定的位置累加到与self同形状的新张量中
    
    参数:
        dim: 沿着张量src的哪个维度进行索引
        index: 索引张量，形状必须与src（如果是张量）兼容，维度数必须与self相同
        src: 源张量或标量，提供要累加的值
    
    返回:
        原地修改后的张量（input）
    """
    dev = input.device
    
    # 统一将index转换为int64的TN张量（已是int64且同设备的TN会直接返回原对象）
    index = as_tensor(index, dtype=int64, device=dev)
    
    # 处理src为标量的情况        
    # src参数的类型检查：int、float、complex、numpy标量、TN标量
    if is_scalar(src):
        src = full_like(index, src, dtype=input.dtype, device=dev)

    if not isinstance(src,TN):
        src = as_tensor(src, dtype=input.dtype, device=dev)

    # 确保src和self的dtype相同
    if src.dtype != input.dtype:
        raise RuntimeError("scatter_add_(): Expected input.dtype to be equal to src.dtype")

    # 对于张量，确保形状与index相同
    if src.shape != index.shape:
        raise ValueError(f"Shape mismatch between index and src: index has shape {index.shape}, src has shape {src.shape}")
    
    # 确保dim是有效的维度
    if dim < 0:
        dim += input.ndim
    if dim < 0 or dim >= input.ndim:
        raise IndexError(f"Dimension out of range (expected to be in range of [{-input.ndim}, {input.ndim-1}], got {dim})")
    
    # 确保index和self的维度数相同
    if index.ndim != input.ndim:
        raise ValueError(f"Dimension mismatch between index and input: index has {index.ndim} dimensions, input has {input.ndim} dimensions")
            
    # 计算直接索引
    full_index = _compute_to_direct_indices(input, dim, index._array)
    
    # 直接调用原地操作的addat_函数
    return input.addat(full_index, src)
# end of scatter_add


# ===========================================================================
# 9. 形状变换与拼接（split / stack / concatenate / unfold / fold / repeat）
# ===========================================================================

@api_registry.register
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

def _split_grad(result_tensor: TN):
    split_sizes, dim, split_pos = result_tensor.parms[0]
    parent = result_tensor.fromvars[0]
    
    grad = zeros_like(parent)

    # 计算起始和结束位置
    start = builtins.sum(split_sizes[:split_pos]) if split_pos > 0 else 0
    end = start + split_sizes[split_pos]
    
    # 累加归一化后的梯度
    slice_obj = [slice(None)] * parent._array.ndim
    slice_obj[dim] = slice(int(start), int(end))
    grad[tuple(slice_obj)] = result_tensor._grad_value
    
    return (grad,)


def _split_with_indices(input: TN, indices_or_sections, dim: int) -> tuple[TN, ...]:
    """
    内部函数：使用 numpy 风格的语义分割张量。

    该函数根据 `indices_or_sections` 的类型决定分割方式：
    - 当 `indices_or_sections` 为整数时，表示将张量分割成 N 段。如果维度大小不能被 N 整除，
      最后一段将包含剩余的元素。
    - 当 `indices_or_sections` 为列表时，表示在指定的索引位置进行分割。

    参数：
        input (TN): 输入的 TN 张量。
        indices_or_sections (int | list): 分割参数。
            - int: 分割的段数。
            - list: 分割点的索引列表。
        dim (int): 进行分割的维度。默认为 0。

    返回：
        tuple[TN, ...]: 包含分割后子张量的元组。

    示例：
        >>> # 按段数分割
        >>> x = rm.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        >>> _split_with_indices(x, 2, dim=1)
        (tensor([[1, 2], [5, 6]]), tensor([[3, 4], [7, 8]]))
        >>>
        >>> # 按索引分割
        >>> _split_with_indices(x, [1, 3], dim=1)
        (tensor([[1], [5]]), tensor([[2, 3], [6, 7]]), tensor([[4], [8]]))
    """
    arrlib = input._get_array_lib()
    
    if isinstance(indices_or_sections, int):
        # 整数：分割成 N 段
        n_sections = indices_or_sections
        if n_sections <= 0:
            raise ValueError(f"number of sections must be positive, got {n_sections}")
        split_data = arrlib.array_split(input._array, n_sections, axis=dim)
    else:
        # 列表：分割点索引
        split_points = list(indices_or_sections)
        if len(split_points) == 0:
            split_data = [input._array]
        else:
            split_data = arrlib.split(input._array, split_points, axis=dim)
    
    # 从分割结果计算每段的大小（用于反向传播）
    split_sizes = [s.shape[dim] for s in split_data]
    
    # 创建子张量列表（保留计算图）
    sub_tensors = []
    for i, data in enumerate(split_data):
        subt = TN(data, requires_grad=(is_grad_enabled() and input.requires_grad))
        
        # 记录计算图信息
        if input.requires_grad:
            subt.fromvars = (input,)
            subt.parms = ((split_sizes, dim, i),)
            subt._grad_rule = _split_grad
        
        sub_tensors.append(subt)
    
    return tuple(sub_tensors)


def split(input: TN, split_size_or_sections, dim: int = 0) -> tuple[TN, ...]:
    """
    沿指定轴分割TN张量，支持计算图记录和梯度反向传播
    
    参数：
        input: 输入TN张量
        split_size_or_sections: 
            - 整数：每段的大小（最后一段可能较小）
            - 列表：每段的大小列表，总和必须等于该维度大小
        dim: 分割轴（默认0）
    
    返回：
        tuple[TN, ...]: 分割后的张量元组
    
    示例：
        >>> x = rm.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])  # shape (2, 4)
        >>> rm.split(x, 2, dim=1)  # 每段大小为2
        (tensor([[1, 2], [5, 6]]), tensor([[3, 4], [7, 8]]))
        >>> rm.split(x, [1, 3], dim=1)  # 段大小为[1, 3]
        (tensor([[1], [5]]), tensor([[2, 3, 4], [6, 7, 8]]))
    """
    dim_size = input.shape[dim]
    
    if isinstance(split_size_or_sections, int):
        # 整数：每段的大小，转换为分割点
        section_size = split_size_or_sections
        if section_size <= 0:
            raise ValueError(f"split_size_or_sections must be positive, got {section_size}")
        # 计算分割点
        split_points = list(range(section_size, dim_size, section_size))
    else:
        # 列表：每段的大小，转换为分割点
        split_sizes = list(split_size_or_sections)
        total_size = builtins.sum(split_sizes)
        if total_size != dim_size:
            raise ValueError(f"sum of split sections ({total_size}) must equal dimension size ({dim_size})")
        # 计算分割点（累积和）
        split_points = list(accumulate(split_sizes[:-1]))
    
    # 使用 _split_with_indices 进行分割
    return _split_with_indices(input, split_points, dim)


def vsplit(input: TN, indices_or_sections):
    """
    垂直分割（沿 dim=0），将张量分割为多个子张量
    
    参数：
        input: 输入张量，至少2维
        indices_or_sections: 
            - 整数：分割成 N 段
            - 列表：分割点索引列表
    
    示例：
        >>> x = rm.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])  # shape (4, 2)
        >>> rm.vsplit(x, 2)  # 分割成 2 段
        (tensor([[1, 2], [3, 4]]), tensor([[5, 6], [7, 8]]))
        >>> rm.vsplit(x, [1, 3])  # 在索引 1 和 3 处分割
        (tensor([[1, 2]]), tensor([[3, 4], [5, 6]]), tensor([[7, 8]]))
    """
    return _split_with_indices(input, indices_or_sections, dim=0)


def hsplit(input: TN, indices_or_sections):
    """
    水平分割（沿 dim=1），将张量分割为多个子张量
    
    参数：
        input: 输入张量，至少2维
        indices_or_sections: 
            - 整数：分割成 N 段
            - 列表：分割点索引列表
    
    示例：
        >>> x = rm.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])  # shape (2, 4)
        >>> rm.hsplit(x, 2)  # 分割成 2 段
        (tensor([[1, 2], [5, 6]]), tensor([[3, 4], [7, 8]]))
    """
    return _split_with_indices(input, indices_or_sections, dim=1)


def dsplit(input: TN, indices_or_sections):
    """
    深度分割（沿 dim=2），将张量分割为多个子张量
    
    参数：
        input: 输入张量，至少3维
        indices_or_sections: 
            - 整数：分割成 N 段
            - 列表：分割点索引列表
    
    示例：
        >>> x = rm.tensor(np.random.randn(2, 3, 6))  # shape (2, 3, 6)
        >>> rm.dsplit(x, 3)  # 分割成 3 段
        # 返回 3 个 shape 为 (2, 3, 2) 的张量
    """
    return _split_with_indices(input, indices_or_sections, dim=2)


def tensor_split(input: TN, indices_or_sections, dim: int = 0) -> tuple[TN, ...]:
    """
    将张量沿指定维度分割为多个子张量。

    该函数根据 `indices_or_sections` 的类型决定分割方式：
    - 当 `indices_or_sections` 为整数时，表示将张量分割成 N 段。如果维度大小不能被 N 整除，
      前 `size % N` 个段的大小为 `size // N + 1`，其余段的大小为 `size // N`。
    - 当 `indices_or_sections` 为列表时，表示在指定的索引位置进行分割。

    参数：
        input (TN): 输入的 TN 张量。
        indices_or_sections (int | list): 分割参数。
            - int: 分割的段数。
            - list: 分割点的索引列表。
        dim (int): 进行分割的维度。默认为 0。

    返回：
        tuple[TN, ...]: 包含分割后子张量的元组。

    示例：
        >>> x = rm.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])  # shape (3, 4)
        >>> # 按段数分割
        >>> tensor_split(x, 3, dim=0)
        (tensor([[1, 2, 3, 4]]), tensor([[5, 6, 7, 8]]), tensor([[9, 10, 11, 12]]))
        >>> # 按索引分割
        >>> tensor_split(x, [1, 2], dim=0)
        (tensor([[1, 2, 3, 4]]), tensor([[5, 6, 7, 8]]), tensor([[9, 10, 11, 12]]))
    """
    return _split_with_indices(input, indices_or_sections, dim)


def _stack_grad(result_tensor: TN):
    dim = result_tensor.parms[0]
    grad = result_tensor._grad_value
    slices = [slice(None)] * grad._array.ndim
    grads = []
    for i in range(len(result_tensor.fromvars)):
        slices[dim] = i  # type: ignore
        grads.append(grad[tuple(slices)])
    return tuple(grads)


@api_registry.register
def stack(tensors: tuple[TN, ...]|list[TN], dim: int = 0) -> TN:
    """沿新维度堆叠张量"""
    arrlib = tensors[0]._get_array_lib()
    data = arrlib.stack([t._array for t in tensors], axis=dim)
    
    # 梯度需求判断优化为any()
    requires_grad = (is_grad_enabled() and builtins.any(t.requires_grad for t in tensors))
    
    ret = TN(data, requires_grad=requires_grad)
    
    if requires_grad:
        ret.fromvars = tuple(tensors)  # 确保tuple类型
        num = len(tensors)
        ret.parms = (dim,) * num
        ret._grad_rule = _stack_grad
    
    return ret


def _concatenate_grad(result_tensor: TN):
    dim = result_tensor.parms[0]
    grad_data = result_tensor._grad_value
    
    # 获取所有输入张量在拼接轴上的尺寸
    split_sizes = [t._array.shape[dim] for t in result_tensor.fromvars]
    
    # 使用Python内置的accumulate函数计算累积和，避免不必要的数组转换
    split_points = list(accumulate(split_sizes[:-1])) if len(split_sizes) > 1 else []
    
    try:
        # 使用 _split_with_indices 进行分割（numpy风格语义）
        grads = _split_with_indices(grad_data, split_points, dim=dim)
    except ValueError as e:
        raise RuntimeError(f"梯度分割失败: {e}") from None

    return tuple(grads)


def concatenate(tensors: tuple[TN, ...]|list[TN], dim: int = 0) -> TN:
    """沿指定轴连接张量"""
    arrlib = tensors[0]._get_array_lib()
    data = arrlib.concatenate([t._array for t in tensors], axis=dim)
    
    requires_grad = (is_grad_enabled() and builtins.any(t.requires_grad for t in tensors))
    
    ret = TN(data, requires_grad=requires_grad)
    
    if requires_grad:
        ret.fromvars = tuple(tensors)  # 确保tuple类型
        num = len(tensors)
        ret.parms = (dim,) * num
        ret._grad_rule = _concatenate_grad
    
    return ret


def cat(tensors: tuple[TN, ...]|list[TN], dim: int = 0) -> TN:
    return concatenate(tensors, dim)


def vstack(tensors: tuple[TN, ...]|list[TN]) -> TN:
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
        expanded_tensors = [t._reshape((1, 1)) for t in tensors]
        return concatenate(expanded_tensors, dim=0)
    elif all_1d:
        # 对于一维张量，添加新维度后沿第0轴连接
        # 这样可以利用concatenate的梯度跟踪能力
        expanded_tensors = [t._reshape((1, -1)) for t in tensors]
        return concatenate(expanded_tensors, dim=0)
    else:
        # 对于多维张量，沿第0轴连接
        return concatenate(tensors, dim=0)


def hstack(tensors: tuple[TN, ...]|list[TN]) -> TN:
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
        expanded_tensors = [t._reshape((1,)) for t in tensors]
        return concatenate(expanded_tensors, dim=0)
    elif all_1d:
        # 对于一维张量，沿第0轴连接（水平堆叠）
        return concatenate(tensors, dim=0)
    else:
        # 对于多维张量，沿第1轴连接
        return concatenate(tensors, dim=1)


def dstack(tensors):
    """
    深度堆叠（沿第三维）
    
    根据输入张量的维度自动处理：
    - 1D 张量 (N,) -> reshape 为 (1, N, 1)，然后沿第2维堆叠
    - 2D 张量 (M, N) -> reshape 为 (M, N, 1)，然后沿第2维堆叠  
    - 3D+ 张量 (M, N, P, ...) -> 沿第2维拼接
    
    参数：
        tensors: 张量列表
    
    返回：
        堆叠后的张量
    """
    if not tensors:
        raise ValueError("dstack requires at least one tensor")
    
    # 获取第一个张量的维度数
    first_tensor = tensors[0]
    ndim = first_tensor.ndim
    
    if ndim == 1:
        # 1D: (N,) -> (1, N, 1)，然后沿第2维堆叠
        reshaped = [t.reshape((1, -1, 1)) for t in tensors]
        return concatenate(reshaped, dim=2)
    elif ndim == 2:
        # 2D: (M, N) -> (M, N, 1)，然后沿第2维堆叠
        reshaped = [t.reshape((*t.shape, 1)) for t in tensors]
        return concatenate(reshaped, dim=2)
    else:
        # 3D+: 沿第2维拼接
        return concatenate(tensors, dim=2)


@api_registry.register
def unfold(tensor_obj: TN, dimension: int, size: int, step: int = None) -> 'TN':  # type: ignore
    """将张量的指定维度展开为多个连续的切片。
    
    该函数用于将张量的指定维度展开为
    多个连续的窗口切片，常用于卷积操作中的im2col变换。
    
    参数:
        tensor_obj: 输入张量
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
    dim = dimension if dimension >= 0 else tensor_obj.ndim + dimension
    if dim < 0 or dim >= tensor_obj.ndim:
        raise ValueError(f"Dimension {dimension} out of range for tensor with {tensor_obj.ndim} dimensions")
    
    # 获取输入形状
    input_shape = list(tensor_obj.shape)
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
        idx = [slice(None)] * tensor_obj.ndim
        idx[dim] = slice(start, end)
        
        # 获取窗口
        window = tensor_obj[tuple(idx)]
        
        # 调整窗口形状：将展开维度移动到最后
        # 例如：对于形状(a, size, b)，转换为(a, b, size)
        perm = list(range(tensor_obj.ndim))
        perm.append(perm.pop(dim))
        window = window.permute(perm)
        
        # 在原dim位置插入1，形成窗口数量维度
        window_shape = list(window.shape)
        window_shape.insert(dim, 1)
        window = window._reshape(tuple(window_shape))
        
        # 将窗口添加到列表中
        windows.append(window)
    
    # 沿窗口数量维度堆叠所有窗口
    result = concatenate(windows, dim=dim)
    
    # 确保输出形状正确
    assert result.shape == tuple(output_shape), f"Expected shape {output_shape}, got {result.shape}"
    
    return result


@api_registry.register
def repeat(input: TN, repeats: tuple[int, ...]) -> TN:
    """
    沿着指定的维度重复张量的元素。
    
    返回一个新的张量，其中输入张量的元素在每个维度上重复指定的次数。
    
    此函数的行为与PyTorch的torch.repeat()完全一致，包括：
    - 当repeats长度小于输入张量维度时，抛出异常
    - 当repeats长度大于输入张量维度时，在前面添加新的维度
    - 对整个维度的内容进行重复，而不是对每个元素单独重复
    
    Args:
        input (TN): 要重复的输入张量
        repeats (tuple[int, ...]): 每个维度的重复次数，可以是整数元组或多个整数参数
            - 如果长度小于输入张量维度，会抛出异常
            - 如果长度大于输入张量维度，会在前面添加新的维度
            - 所有元素必须是正整数
        
    Returns:
        TN: 重复后的新张量
        
    Raises:
        TypeError: 如果输入不是TN张量
        ValueError: 如果repeats包含非正整数
        
    Examples:
        >>> # 一维张量的重复
        >>> x = tensor([1, 2, 3])  # 形状(3,)
        >>> y = repeat(x, (2,))  # 形状(6,)，结果为[1, 2, 3, 1, 2, 3]
        >>> 
        >>> # 二维张量的重复
        >>> z = tensor([[1, 2], [3, 4]])  # 形状(2, 2)
        >>> w = repeat(z, (2, 3))  # 形状(4, 6)，结果为[[1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4], [1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4]]
        >>> 
        >>> # repeats长度小于输入张量维度
        >>> v = repeat(z, (3,))  # 形状(2, 6)，结果为[[1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4]]
        >>> 
        >>> # repeats长度大于输入张量维度
        >>> u = repeat(x, (2, 3))  # 形状(2, 9)，结果为[[1, 2, 3, 1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3, 1, 2, 3]]
        >>> 
        >>> # 使用方法调用
        >>> a = tensor([5, 6, 7])
        >>> b = a.repeat(4)  # 形状(12,)，结果为[5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7]
    """
    # 验证输入
    if not isinstance(input, TN):
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input)}")
    
    # 使用_validate_shape处理repeats参数
    repeats = _validate_shape(repeats)
    
    # 验证所有重复次数都是正整数
    for r in repeats:
        if not isinstance(r, int) or r <= 0:
            raise ValueError(f"All elements in repeats must be positive integers, got {r}")
    
    # 获取底层数组库
    arrlib = input._get_array_lib()
    
    # 保存输入张量的原始形状
    original_shape = input._array.shape
    
    # 处理repeats长度与输入张量维度的关系
    ndim = input._array.ndim
    if len(repeats) < ndim:
        # 如果repeats长度小于输入张量维度，抛出异常，与PyTorch行为一致
        raise ValueError(f"Number of dimensions of repeat dims ({len(repeats)}) can not be smaller than number of dimensions of tensor ({ndim})")
    elif len(repeats) > ndim:
        # 如果repeats长度大于输入张量维度，在前面添加新的维度
        # 首先计算需要添加的维度数
        new_dims = len(repeats) - ndim
        # 在输入张量前面添加新的维度，每个新维度的大小为1
        for _ in range(new_dims):
            input._array = arrlib.expand_dims(input._array, axis=0)
        # 更新ndim值
        ndim = input._array.ndim
    
    # 使用NumPy的repeat功能实现
    
    # 对于每个维度，从最后一个维度开始重复，只处理需要重复的轴
    # 注意：使用tile函数而不是repeat函数，以实现与PyTorch一致的行为
    # PyTorch的repeat是对整个维度的内容进行重复，而NumPy的repeat是对每个元素进行重复
    repeated_data = input._array
    
    # 计算每个维度的重复次数，对于不需要重复的维度，设置为1
    tile_repeats = []
    for i in range(ndim):
        if repeats[i] > 1:
            tile_repeats.append(repeats[i])
        else:
            tile_repeats.append(1)
    
    # 使用tile函数进行重复
    repeated_data = arrlib.tile(repeated_data, tuple(tile_repeats))
    
    # 创建结果张量
    result = TN(repeated_data, requires_grad = (is_grad_enabled() and input.requires_grad))
    
    # 设置梯度传播
    if result.requires_grad:
        def _repeat_grad(result_tensor: TN):
            # 重复操作的梯度是将结果的梯度在重复的维度上求和
            grad_value = result_tensor._grad_value
            
            # 对于每个维度，从最后一个维度开始求和
            for j in reversed(range(ndim)):
                # 计算步长和大小
                size = input._array.shape[j]
                repeat = repeats[j]
                
                # 重塑梯度张量，以便在重复的维度上求和
                shape = list(grad_value.shape)
                shape[j] = repeat
                shape.insert(j+1, size)
                grad_reshaped = grad_value._reshape(tuple(shape))
                
                # 在重复的维度上求和
                grad_value = grad_reshaped.sum(dim=j)
            
            # 将梯度的形状调整为输入张量的原始形状
            grad_value = grad_value._reshape(original_shape)
            
            return (grad_value,)
        
        result.fromvars = (input,)
        result._grad_rule = _repeat_grad
    
    return result


@api_registry.register
def fold(tensor_obj:TN, dimension: int, size: int, step: int = None, output_size: int = None) -> 'TN':  # type: ignore
    """
    将张量的指定维度从窗口展开状态折叠回原始形状。
    这是unfold()函数的逆操作。

    参数:
        tensor_obj: 输入张量
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
    dim = dimension if dimension >= 0 else tensor_obj.ndim + dimension
    if dim < 0 or dim >= tensor_obj.ndim:
        raise ValueError(f"Dimension {dimension} out of range for tensor with {tensor_obj.ndim} dimensions")

    # 检查输入张量是否符合预期格式（最后一维应包含窗口元素）
    if tensor_obj.ndim < 2:
        raise ValueError(f"Input tensor must have at least 2 dimensions, got {tensor_obj.ndim}")

    # 获取输入形状
    input_shape = list(tensor_obj.shape)
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

    # 创建一个形状为output_shape的零张量作为基础，与输入张量在同一设备上
    output = zeros(output_shape, dtype=tensor_obj.dtype, device=tensor_obj.device) 

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
        window_idx = [slice(None)] * tensor_obj.ndim
        window_idx[dim] = i  # type: ignore
        window_idx[-1] = slice(None)  # 选择所有窗口元素
        window = tensor_obj[tuple(window_idx)]

        # 确保窗口形状与输出切片形状匹配
        # 窗口形状应该是: output_shape[:dim] + (size,) + output_shape[dim+1:]
        window_shape = list(output_shape)
        window_shape[dim] = size
        window = window.view(tuple(window_shape))

        # 将窗口贡献添加到输出张量（原地操作）
        addat_(output, tuple(output_idx),window)

    return output


# ===========================================================================
# 10. einsum（方程解析、批量对角线辅助、梯度计算与入口）
# ===========================================================================
#
# 前向
#   einsum(equation, *operands)
#     ├─ _parse_einsum_equation      隐式输出转显式 → (input_eq, input_eqs, output_eq)
#     ├─ _same_device                多操作数设备一致性检查
#     ├─ _einsum_use_optimize        GEMM 型缩并走 optimize（BLAS），否则 naive
#     └─ _precompute_grad_info       需要梯度时逐输入预分类
#          └─ _has_duplicate_indices
#
# 反向（_einsum_grad 一次性计算所有操作数的梯度，逐来源按 grad_infos[i] 路由）
#   _einsum_grad(result_tensor)   依据 grad_infos[i]['grad_type'] 路由
#     ├─ 'simple'       → _compute_simple_einsum_grad    一次梯度方程
#     ├─ 'trace'        → _compute_trace_grad            → batch_diag / eye
#     │                     └─ _ellipsis_after_indices
#     ├─ 'diag_extract' → _compute_diag_extract_grad     → batch_diag / eye
#     │                     └─ _ellipsis_after_indices
#     └─ 'complex'      → _compute_complex_einsum_grad
#                           ├─ 偏梯度 einsum(grad_value, 其他操作数 -> 可用索引)
#                           ├─ _reorder_partial_grad    省略号维按目标真实顺序重排
#                           ├─ _expand_grad_dimensions  广播回目标完整形状
#                           │    └─ _broadcast_expand_shape
#                           └─ _restore_diagonal_tensor 重复索引施加对角线约束
#                                └─ _has_duplicate_indices / batch_diag
#     统一收尾：_unbroadcast_grad（前向发生广播的维度按 torch 语义求和还原）
#
# 数据约定：parms = (equation, input_eqs, output_eq, grad_infos)，
#           grad_infos[i] = {'grad_type', 'target_eq', ...}，
#           四类 grad_type 的判定条件见 _precompute_grad_info 的 docstring。
# ===========================================================================


# 从批量1D张量生成批量对角矩阵的函数
@api_registry.register
def batch_diag(v):
    """从批量1D张量生成批量对角矩阵。

    参数:
        v: 输入张量，形状为(*, n)，其中*表示任意数量的批处理维度

    返回:
        形状为(*, n, n)的对角矩阵，对角线元素为输入张量的值

    调用者: _compute_trace_grad、_compute_diag_extract_grad、
    _restore_diagonal_tensor（梯度构造对角线结构）；同时作为公开接口 rm.batch_diag
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


def _einsum_use_optimize(input_eqs: list[str], output_eq: str, operand_data: list) -> bool:
    """
    判断einsum前向是否启用optimize路径。

    optimize使numpy/cupy经由tensordot调用BLAS，对矩阵乘/批量矩阵乘/多操作数
    缩并类运算（GEMM型）有数倍收益。以下情况保持naive路径（实测更快或开销不匹配）：
    - 单操作数（对角线/转置等无跨操作数缩并可优化）
    - 张量规模低于阈值（einsum_path计算开销约30us超过收益）
    - 点积/矩阵向量乘等仅1-D参与缩并的运算（tensordot的转置拷贝比naive慢数倍）

    另：numpy的c_einsum路径不支持省略号维度被求和的方程（如'...i->i'会直接
    抛出ValueError），此类方程必须走optimize路径，与张量规模无关。

    返回:
        True 表示走 optimize（BLAS）路径，False 表示 naive 路径

    调用者: einsum（前向）
    """
    # 省略号维度被求和（如'...i->i'）：c_einsum不支持，必须走optimize路径
    if any('.' in eq for eq in input_eqs) and '.' not in output_eq:
        return True
    if len(input_eqs) < 2:
        return False
    if builtins.max(d.size for d in operand_data) < _EINSUM_OPTIMIZE_MIN_ELEMS:
        return False

    # 统计各索引在所有输入方程中的出现次数：
    # 出现多次且不在输出中的索引即为被求和的缩并索引，只有存在此类索引才有BLAS收益
    char_counts = Counter()
    for eq in input_eqs:
        char_counts.update(eq.replace('.', ''))
    output_chars = set(output_eq.replace('.', ''))
    contracted = {c for c, n in char_counts.items() if n > 1} - output_chars
    if not contracted:
        return False

    if len(input_eqs) == 2:
        # 双操作数时要求双方都保留非缩并维度（排除点积与矩阵向量乘）
        return all(set(eq.replace('.', '')) - contracted for eq in input_eqs)
    return True


def _parse_einsum_equation(equation: str, num_operands: int) -> tuple[str, list[str], str]:
    """
    解析einsum方程字符串。

    显式输出（含 '->'）直接拆分；隐式输出取「只出现一次」的索引并按字母排序，
    若输入含省略号则在最前面补 '...'。

    参数:
        equation: einsum方程字符串
        num_operands: 操作数数量

    返回:
        (input_eq, input_eqs, output_eq): 输入方程字符串、输入方程列表、输出方程字符串

    示例:
        >>> _parse_einsum_equation('ij,jk->ik', 2)
        ('ij,jk', ['ij', 'jk'], 'ik')
        >>> _parse_einsum_equation('ii->', 1)
        ('ii', ['ii'], '')
        >>> _parse_einsum_equation('ij,jk', 2)  # 隐式输出
        ('ij,jk', ['ij', 'jk'], 'ik')

    调用者: einsum（前向入口）
    """
    if '->' in equation:
        input_eq, output_eq = equation.split('->')
    else:
        # 隐式输出：按字母顺序排列的非求和索引
        input_eq = equation
        # 移除省略号后再统计，避免 '.' 被当作普通字符
        input_eq_no_ellipsis = input_eq.replace('...', '')
        all_indices = input_eq_no_ellipsis.replace(',', '')
        counts = Counter(all_indices)
        has_ellipsis = '...' in input_eq
        non_ellipsis_output = ''.join(sorted(c for c in counts if counts[c] == 1))
        if has_ellipsis:
            output_eq = '...' + non_ellipsis_output
        else:
            output_eq = non_ellipsis_output
    
    input_eqs = input_eq.split(',')
    
    if len(input_eqs) != num_operands:
        raise ValueError(f"Number of subscripts ({len(input_eqs)}) does not match "
                        f"number of operands ({num_operands})")
    
    return input_eq, input_eqs, output_eq


def _has_duplicate_indices(target_eq: str) -> bool:
    """检查 target_eq 是否含重复索引（忽略省略号 '...'）。

    调用者: _precompute_grad_info（分类判据）、_restore_diagonal_tensor（是否需施加约束）
    """
    index_counts = {}
    for c in target_eq:
        if c != '.':  # 忽略省略号
            index_counts[c] = index_counts.get(c, 0) + 1
    return any(count > 1 for count in index_counts.values())


def _broadcast_expand_shape(shape: tuple, target_shape: tuple) -> tuple:
    """
    按广播语义对齐源形状与目标形状，用于 expand。

    目标维为1而源维更大的维度（如省略号批量维被其他操作数广播放大）保留源尺寸，使 expand
    合法；此类维度随后由 _unbroadcast_grad 求和还原为目标尺寸（前向共享同一份数据，梯度
    应按广播份数累加）。其余维度取目标尺寸以正常广播。

    返回:
        可直接用于 expand 的形状元组

    调用者: _expand_grad_dimensions、_compute_trace_grad、_compute_diag_extract_grad
    """
    return tuple(src if tgt == 1 else tgt for src, tgt in zip(shape, target_shape))


def _unbroadcast_grad(grad_tensor: TN, target_shape: tuple) -> TN:
    """
    将按广播形状算出的梯度还原为目标操作数的原始形状。

    目标操作数的省略号批量维可能与输出/其他操作数不一致（如 ...i,...i->... 中某操作数
    批量维为1，或 ...aaa,...b->ba 中该维被广播放大），梯度便按广播后的形状算出。按 torch
    语义需先消除广播差异：
      - 目标秩小于梯度秩：多出的前导维由其他操作数/输出广播而来，逐个求和消去；
      - 某个维度目标为 1 而梯度大于 1：对该维 keepdim 求和（前向共享同一份数据，
        梯度按广播份数累加）；
      - 某个维度目标大于 1 而梯度为 1：梯度与广播份数无关，直接广播到目标尺寸。

    形状本就一致时原样返回，故对迹/对角线等已对齐的路径无副作用。

    调用者: _einsum_grad（所有 grad_type 分支的统一收尾）
    """
    grad = grad_tensor
    while grad.ndim > len(target_shape):
        grad = grad.sum(dim=0)
    axes = tuple(ax for ax, (gs, ts) in enumerate(zip(grad.shape, target_shape))
                 if ts == 1 and gs != 1)
    if axes:
        grad = grad.sum(dim=axes, keepdim=True)
    if grad.shape != target_shape:
        # 目标维大于梯度维（前向该维被广播放大），梯度可视为常数沿该维复制
        grad = grad.expand(target_shape)
    return grad


def _ellipsis_after_indices(target_eq: str, rank: int) -> bool:
    """
    判断目标的省略号是否位于全部显式索引之后（如 'ii...'、'ij...k'）。

    用于区分布局：省略号在索引之前/中间时，重复索引对占据最前两维的反面——省略号在被
    重复的索引对之后时，该索引对才真正落在目标的最前两维上。

    返回:
        True 表示省略号位于全部显式索引之后（如 'ii...'）

    调用者: _compute_trace_grad、_compute_diag_extract_grad
    """
    return '...' in target_eq and rank > len(target_eq.replace('.', '')) and \
        len(target_eq.split('...')[0]) == len(target_eq.replace('.', ''))


def _restore_diagonal_tensor(grad_tensor: TN, target_shape: tuple, target_eq: str) -> TN:
    """
    将梯度张量还原为具有重复索引的对角线形式。

    背景：当目标操作数含重复索引（如 'ii'、'iijj'）时，前向只取了对角元素，
    故反向梯度只应落在对应位置的对角线上、其余位置为0。但
    _expand_grad_dimensions 只把梯度广播到了完整形状，非对角位置被重复填了值，
    必须在此处重新施加对角线约束，将其清零。

    约束方式：对每一对重复位置 (pos1, pos2) 做一次「与单位矩阵逐元素相乘」。
    构造方程 'ij,<result_eq>-><result_eq>'（i、j 为两个临时新字符，分别放到
    pos1、pos2 上），由于两个输入的所有索引都保留在输出中、不发生缩并，该方程
    等价于广播乘法：
        result[pos1, pos2] *= I[pos1, pos2]
    而 I 是单位矩阵，于是 pos1 != pos2 处被置0、pos1 == pos2 处保留原值，从而
    施加对角线约束。该实现完全由已有张量算子组合而成，故可自然支持高阶梯度。

    grad_tensor 为 _expand_grad_dimensions 展开后的结果，维度布局与 target_eq 的真实布局一致
    （省略号维按其在 target_eq 中的位置展开成对应数量的维度），重复索引的首个出现位置持有
    数据，其余重复位置为可广播的哑维度。

    参数:
        grad_tensor: 展开后的梯度张量
        target_shape: 目标形状
        target_eq: 目标索引字符串（包含重复索引）

    返回:
        还原后的梯度张量（仅重复索引的对角线位置非0）

    调用者: _compute_complex_einsum_grad（阶段3）
    """
    if not _has_duplicate_indices(target_eq):
        return grad_tensor

    target_eq_clean = target_eq.replace('.', '')

    # 省略号维数量与「省略号之前的显式索引个数」，用于把索引在 target_eq 中的位置映射到
    # 展开后张量的真实维度下标（省略号维按其在目标中的真实位置占据一段连续维度）
    n_ellipsis = grad_tensor.ndim - len(target_eq_clean)
    n_before = len(target_eq.split('...')[0]) if '...' in target_eq else len(target_eq_clean)

    def real_dim(pos):
        """target_eq_clean 中第 pos 个索引在展开后张量中对应的真实维度下标"""
        return pos + (n_ellipsis if pos >= n_before else 0)

    # 快速路径：恰好2个索引且相同，且重复位置正好是最后两维（如 ii、...ii），
    # 直接广播乘单位矩阵即可（与一般路径等价，省去构造方程的额外 einsum）
    if (len(target_eq_clean) == 2 and target_eq_clean[0] == target_eq_clean[1] and
            real_dim(0) == grad_tensor.ndim - 2):
        n = grad_tensor.shape[-1]
        return grad_tensor * eye(n, device=grad_tensor.device, dtype=grad_tensor.dtype)

    # 一般路径：按字符分组，找出所有重复索引及其在 target_eq 中的位置
    index_positions = defaultdict(list)
    for pos, c in enumerate(target_eq_clean):
        index_positions[c].append(real_dim(pos))

    result = grad_tensor

    for char, positions in index_positions.items():
        if len(positions) > 1:
            # 该索引出现多次，需要强制其各出现位置满足对角线约束
            dim_size = target_shape[positions[0]]
            I = eye(dim_size, device=result.device, dtype=result.dtype)

            # 位置多于2个时，逐对（相邻两位置）施加一次约束即可覆盖全部组合
            for p in range(len(positions) - 1):
                pos1 = positions[p]
                pos2 = positions[p + 1]

                # 为 result 的每个维度分配一个临时字符（'a','b',...），再把 pos1、pos2
                # 两处的字符替换为两个全新的、只出现在单位矩阵方程中的字符。这样两个输入
                # 的所有索引都会保留到输出，不触发缩并，等价于沿该两维做广播逐元素相乘
                n_dims = result.ndim
                result_chars = [chr(ord('a') + d) for d in range(n_dims)]
                i_char = chr(ord('a') + n_dims)
                j_char = chr(ord('a') + n_dims + 1)

                identity_eq = i_char + j_char
                result_eq = result_chars.copy()
                result_eq[pos1] = i_char
                result_eq[pos2] = j_char
                result_eq_str = ''.join(result_eq)

                equation = f'{identity_eq},{result_eq_str}->{result_eq_str}'
                result = einsum(equation, I, result)

    return result


def _reorder_partial_grad(
    partial_grad: TN, target_eq: str, available_target_chars: list[str], has_ellipsis: bool
) -> TN:
    """
    把偏梯度从「省略号维在最前」的标准顺序重排为目标的真实维度顺序。

    偏梯度由方程 '<grad_eq>,<其他操作数> -> ...<可用索引>' 算出，省略号维恒定位于最前，
    其后是可用索引（按 available_target_chars 顺序，即其在 target_eq 中首次出现的顺序）。
    而目标的省略号可能出现在中间或末尾（如 i...j、ij...k、ii...），其真实布局把显式索引
    排在省略号维之前或之间。若重排缺失，后续按真实顺序消耗维度时会张冠李戴。

    重排后顺序为：[省略号之前的可用索引][省略号维][省略号之后的可用索引]；
    省略号维数量为 0、或原本就在最前时无需重排。

    调用者: _compute_complex_einsum_grad（扩展维度前先对齐轴顺序）
    """
    if not has_ellipsis or not available_target_chars:
        return partial_grad

    target_eq_clean = target_eq.replace('.', '')
    n_ellipsis = partial_grad.ndim - len(available_target_chars)
    # 省略号之前的显式索引个数：即真实布局中省略号维左侧的索引数
    n_before = len(target_eq.split('...')[0])
    if n_ellipsis <= 0 or n_before == 0:
        return partial_grad

    first_pos = {c: target_eq_clean.index(c) for c in available_target_chars}
    before = [n_ellipsis + k for k, c in enumerate(available_target_chars)
              if first_pos[c] < n_before]
    after = [n_ellipsis + k for k, c in enumerate(available_target_chars)
             if first_pos[c] >= n_before]
    perm = before + list(range(n_ellipsis)) + after
    if perm == list(range(partial_grad.ndim)):
        return partial_grad
    return partial_grad.permute(perm)


def _expand_grad_dimensions(
    partial_grad: TN, target_eq: str, available_indices: set,
    has_ellipsis: bool, target_shape: tuple
) -> TN:
    """
    把偏梯度扩展（广播）到目标的完整形状。

    偏梯度的轴顺序与「目标真实维度顺序中过滤出可消耗维度」的顺序一致，即
    [省略号之前的可用索引][省略号维][省略号之后的可用索引]（由 _reorder_partial_grad 保证）。
    本函数按同一顺序重建目标的完整秩形状 expand_shape：

    - 省略号维：有来源（has_ellipsis）时依次取偏梯度对应轴的真实尺寸；无来源时说明这些
      批量维前向已被独立求和、梯度对其为常数，占位 1 等待广播；
    - 可用索引：仅其「首次出现」消耗偏梯度的一个轴，重复出现处填 1，留待
      _restore_diagonal_tensor 施加对角线约束；
    - 不可用索引（前向已被独立求和）：填 1，后续通过广播补齐。

    省略号维按其在目标中的真实位置插入，故省略号位于中间/末尾（如 i...j、ij...k）时也能
    与目标形状对齐；由于消耗轴的顺序与该顺序完全一致，reshape 不会打乱已有数据。

    调用者: _compute_complex_einsum_grad（阶段2，仅在偏梯度秩小于目标秩时调用）
    """
    target_eq_clean = target_eq.replace('.', '')
    # 目标中的省略号维数量：目标秩减去显式索引个数
    n_ellipsis = len(target_shape) - len(target_eq_clean)
    # 省略号之前的显式索引个数（无省略号时视为全部索引之后），即省略号维的插入位置
    n_before = len(target_eq.split('...')[0]) if '...' in target_eq else len(target_eq_clean)

    # 按目标真实布局逐个列出「槽位」：省略号前的索引、省略号维、省略号后的索引
    slots = ([('char', p) for p in range(n_before)] +
             [('ellipsis', k) for k in range(n_ellipsis)] +
             [('char', p) for p in range(n_before, len(target_eq_clean))])

    expand_shape = []      # 目标的完整秩形状（含省略号维）
    result_data_idx = 0    # partial_grad 中下一个待消费的轴下标
    seen_available = set()

    for kind, p in slots:
        if kind == 'ellipsis':
            consume = has_ellipsis
        else:
            c = target_eq_clean[p]
            # 仅索引的首次出现消耗一个轴；重复出现处留待对角线约束处理
            consume = c in available_indices and c not in seen_available
            if consume:
                seen_available.add(c)
        if consume and result_data_idx < partial_grad.ndim:
            expand_shape.append(partial_grad.shape[result_data_idx])
            result_data_idx += 1
        else:
            expand_shape.append(1)

    partial_grad = partial_grad.reshape(expand_shape)
    return partial_grad.expand(_broadcast_expand_shape(expand_shape, target_shape))


def _compute_simple_einsum_grad(
    grad_value: TN, output_eq: str, 
    all_operands: list[TN], i: int, grad_equation: str
) -> TN:
    """
    计算简单einsum操作的梯度（无重复索引）。

    直接使用预计算的梯度方程，性能最优。
    grad_value的方程为output_eq（空输出时为''），确保非单位梯度正确参与计算。

    调用者: _einsum_grad（grad_type='simple'）
    """
    grad_inputs = [grad_value]

    for j, op in enumerate(all_operands):
        if j != i:
            grad_inputs.append(op.conj() if op.is_complex() else op)

    return einsum(grad_equation, *grad_inputs)


def _compute_trace_grad(grad_value: TN, target_operand: TN, target_eq: str) -> TN:
    """
    计算迹操作（ii->、bii->b、...ii->...、...ii->、ii...->、ii...->...）的梯度。

    迹在前向只对重复索引对应的两维取对角元素求和，反向即构造一个对角张量：对角元素为
    grad_value 的对应值、非对角为0。注意 grad_value 可能缺少目标的部分前导维度
    （如 ...ii-> 的输出是标量、bii->b 只含批量维），需先广播到目标的批量形状。

    重复索引对默认位于目标最后两维；但当省略号位于该索引对之后（如 ii...）时，该对实际
    落在目标最前两维，此时 grad_value 对应的是尾部的省略号批量维，需按前两维施加对角约束。

    调用者: _einsum_grad（grad_type='trace'）
    """
    target_shape = target_operand.shape
    rank = len(target_shape)

    if _ellipsis_after_indices(target_eq, rank):
        # 重复对在最前两维：先把 grad_value（尾部省略号维，或标量）补前导维后广播到目标形状，
        # 再沿最前两维乘单位矩阵，把非对角位置清零
        n = target_shape[0]
        v = grad_value.reshape((1,) * (rank - grad_value.ndim) + grad_value.shape)
        v = v.expand(_broadcast_expand_shape(v.shape, target_shape))
        eye_shape = (n, n) + (1,) * (rank - 2)
        return v * eye(n, device=v.device, dtype=v.dtype).reshape(eye_shape)

    batch_shape = target_operand.shape[:-1]
    if not batch_shape:
        # 纯2D迹（ii->）：梯度为0维标量，直接构造 (n, n) 的对角矩阵
        n = target_operand.shape[-1]
        v = grad_value * ones(n, device=grad_value.device, dtype=grad_value.dtype)
        return batch_diag(v)
    # 其余情况：在梯度末尾补一维（对角维）后广播到批量形状，再构造对角矩阵。
    # 这样无论 grad_value 是标量（...ii->）还是缺批量维均可正确对齐
    return batch_diag(grad_value.unsqueeze(-1).expand(batch_shape))


def _compute_diag_extract_grad(grad_value: TN, target_operand: TN, target_eq: str) -> TN:
    """
    计算对角线提取操作（ii->i、...ii->i、ii...->i）的梯度。

    梯度是一个对角张量；输出缺少目标的省略号维度时（如 ...ii->i）需先扩展批量维度。
    省略号位于索引对之后时（ii...），对角约束施加在目标最前两维、grad_value 对应首维。

    调用者: _einsum_grad（grad_type='diag_extract'）
    """
    target_shape = target_operand.shape
    rank = len(target_shape)

    if _ellipsis_after_indices(target_eq, rank):
        n = target_shape[0]
        v = grad_value.reshape((n,) + (1,) * (rank - 1))
        v = v.expand(_broadcast_expand_shape(v.shape, target_shape))
        eye_shape = (n, n) + (1,) * (rank - 2)
        return v * eye(n, device=v.device, dtype=v.dtype).reshape(eye_shape)

    if grad_value.ndim < target_operand.ndim - 1:
        grad_value = grad_value.expand(target_operand.shape[:-1])
    return batch_diag(grad_value)


def _compute_complex_einsum_grad(
    grad_value: TN, output_eq: str, other_indices: list[str], 
    other_operands: list[TN], target_eq: str, target_operand: TN,
    info: dict
) -> TN:
    """
    计算复杂 einsum 操作的梯度（目标含重复索引，或存在无法从其他来源获得的索引）。

    分三步：
    1. 偏梯度：把所有可用索引（输出 + 其他操作数）按梯度方程缩并，得到一个只含目标
       可用子集维度的张量（对应前向中未被独立求和的那些维度）；
    2. 扩展维度：把偏梯度 reshape/expand 回目标的完整形状（缺失维、重复维先补1再广播）；
    3. 恢复对角线：对重复索引施加对角线约束，保证非对角位置梯度为0。

    调用者: _einsum_grad（grad_type='complex'）
    """
    available_indices = info['available_indices']
    # 偏梯度中的省略号维度对应「目标自身的批量维度」，能否保留取决于该批量维在反向
    # 中是否有来源，分两种情况：
    # 1. 输出含'...'：grad_value 自带批量维度（如 ...ii,i->...）
    # 2. 其他操作数含'...'：目标的批量维与对方的批量维配对收缩，不能丢弃
    #    （如 ...aaa,...b->ba 中 dx0[q,...] = sum_b g[b,...] * w[q,b]，q 依赖必须保留）
    # 反之，若输出不含'...'且其他操作数也不含'...'，说明目标的批量维已在前向被独立
    # 求和、梯度对其为常数，此时不能加'...'前缀（否则 numpy 会拒绝无来源的输出省略号，
    # 例如 ...cdc->）
    has_ellipsis = ('...' in target_eq
                    and ('...' in output_eq or any('.' in oi for oi in other_indices)))
    target_eq_clean = target_eq.replace('.', '')
    
    # 构造目标的"可用索引串"：按 target_eq 的出现顺序，保留能在梯度输入中找到来源的
    # 索引，并去重（重复索引只保留一次，供后续扩展阶段复用同一维度）
    seen = set()
    available_target_chars = []
    for c in target_eq_clean:
        if c in available_indices and c not in seen:
            available_target_chars.append(c)
            seen.add(c)
    available_target = ''.join(available_target_chars)
    
    if has_ellipsis:
        available_target_with_ellipsis = '...' + available_target
    else:
        available_target_with_ellipsis = available_target
    
    # 组装偏梯度的 einsum 输入：grad_value（方程恒为 output_eq，空输出时用 '' 占位，
    # 确保 grad_value 不参与缩并也不被丢弃）+ 各其他操作数（复数则取共轭）
    grad_input_eqs = [output_eq]
    grad_inputs = [grad_value]

    for idx, op in zip(other_indices, other_operands):
        grad_inputs.append(op.conj() if op.is_complex() else op)
        grad_input_eqs.append(idx)

    # 计算偏梯度，输出方程只保留目标的可用索引（可带 '...' 前缀）。
    # 快捷分支：目标没有任何可用索引、且只有 grad_value 一个输入时，方程退化为 '->'，
    # 偏梯度就是 grad_value 本身（标量），无需再走一次 einsum
    if not available_target_with_ellipsis and len(grad_input_eqs) == 1:
        partial_grad = grad_value
    else:
        grad_equation = ','.join(grad_input_eqs) + '->' + available_target_with_ellipsis
        partial_grad = einsum(grad_equation, *grad_inputs)
    
    # 偏梯度的维度顺序固定为 [省略号维][可用索引]，而目标的真实布局可能把省略号维放在中间
    # （如 i...j 的真实布局是 [i][省略号][j]），故先按目标的真实顺序重排偏梯度
    partial_grad = _reorder_partial_grad(
        partial_grad, target_eq, available_target_chars, has_ellipsis)

    # 偏梯度秩小于目标时，扩展到目标的完整形状
    target_shape = target_operand.shape
    if partial_grad.ndim < len(target_shape):
        partial_grad = _expand_grad_dimensions(
            partial_grad, target_eq, available_indices, has_ellipsis, target_shape
        )

    # 恢复重复索引的对角线结构（非对角位置清零）
    return _restore_diagonal_tensor(partial_grad, target_shape, target_eq)


def _precompute_grad_info(input_eqs: list[str], output_eq: str,
                          operands: list[TN]) -> list[dict]:
    """
    为每个输入预计算反向传播所需的梯度信息，避免反向时重复推导。

    反向梯度统一遵循「用梯度方程再算一次 einsum」的思路：
        grad_target = einsum(grad_value, 其他操作数 -> target_eq)
    但不同形态的方程能构造出的合法梯度方程不同，故在此预先分类，反向时直接路由。
    grad_type 共四类：

    - 'simple'：目标无重复索引，且其所有索引都能从「输出 + 其他操作数」中找到来源。
      此时目标索引在梯度输入中全部出现，可直接拼出合法梯度方程，反向只需一次 einsum。
    - 'trace'：单操作数、恰好2个索引且重复、输出少2维（如 'ii->'），即取迹，
      反向等价于把标量梯度铺到对角线上。
    - 'diag_extract'：单操作数、恰好2个索引且重复、输出少1维（如 'ii->i'），
      即提取对角线，反向同样只需构造对角矩阵。
    - 'complex'：目标含重复索引，或存在无法从其他来源获得的索引。此时先做一次
      「偏梯度」einsum 得到目标的可用子集，再扩展维度、恢复对角线约束。

    参数:
        input_eqs: 每个操作数的索引字符串
        output_eq: 输出索引字符串（隐式输出已由 _parse_einsum_equation 解析为显式形式）
        operands: 操作数列表

    返回:
        每个输入对应一个 info 字典：
          - grad_type: 'simple' / 'trace' / 'diag_extract' / 'complex'
          - target_eq: 该输入的索引字符串
          - 以及 available_indices、available_target_chars、has_ellipsis、
            grad_equation 等辅助字段，由各分支按需取用

    调用者: einsum（前向，仅在需要梯度时调用）
    """
    grad_infos = []
    
    for i, target_eq in enumerate(input_eqs):
        info = {'target_eq': target_eq}
        
        # 反向梯度方程的输入来源：除本目标外的其他操作数索引（它们提供未被求和的原始数据）
        other_indices = [eq for j, eq in enumerate(input_eqs) if j != i]
        # 可获得的索引集合 = 输出索引 ∪ 其他操作数索引；不在此集合中的索引，说明前向
        # 只在目标自身内对其求和、无其他来源，需要走 'complex' 分支特殊处理
        available_indices = set(output_eq) | set(''.join(other_indices))
        available_indices.discard('.')
        
        target_eq_clean = target_eq.replace('.', '')
        has_duplicates = _has_duplicate_indices(target_eq)
        has_ellipsis = '...' in target_eq
        
        # 分类所需的几个结构判据
        is_single_operand = len(input_eqs) == 1
        # 末尾两维索引相同（如 ii、bii、...ii）
        is_last_two_dup = (len(target_eq_clean) >= 2 and 
                          target_eq_clean[-1] == target_eq_clean[-2])
        
        # 目标中存在既不在输出、也不在其他操作数中的索引（前向已把它们各自独立求和）
        has_missing = any(c not in available_indices for c in target_eq_clean)
        
        # 目标的省略号维度能否从梯度输入中找到来源：若目标含'...'，而输出与其他操作数
        # 都不含'...'，说明这些批量维已在前向被独立求和，梯度需广播回目标形状，
        # 无法用 simple 的单次 einsum 表达，改由 'complex' 分支（含维度扩展）处理
        has_unavailable_ellipsis = ('...' in target_eq and
                                    not ('...' in output_eq or
                                         any('.' in eq for eq in other_indices)))
        
        # 迹 / 对角线提取：只适用于「单操作数 + 恰好2个索引重复」的情况（如 ii->、ii->i）；
        # 多索引重复（如 iijj->ij）形态更复杂，统一交给 'complex' 处理
        is_trace_like = (is_single_operand and is_last_two_dup and has_duplicates and
                        len(target_eq_clean) == 2)
        
        if is_trace_like:
            # 依据输出比目标少的维数区分「迹」和「对角线提取」
            output_eq_clean = output_eq.replace('.', '')
            if len(output_eq_clean) == len(target_eq_clean) - 2:
                info['grad_type'] = 'trace'
            elif len(output_eq_clean) == len(target_eq_clean) - 1:
                info['grad_type'] = 'diag_extract'
            else:
                info['grad_type'] = 'complex'
                info['has_duplicates'] = has_duplicates
                info['has_ellipsis'] = has_ellipsis
                info['available_indices'] = available_indices
        elif not has_duplicates and not has_missing and not has_unavailable_ellipsis:
            # 简单情况：无重复索引、无缺失索引、且省略号维有来源，可直接构造梯度方程
            info['grad_type'] = 'simple'
            # 梯度方程 = 'output_eq,其他操作数索引 -> target_eq'。
            # grad_value 的方程固定用 output_eq：空输出（'->'）时为空串 ''，numpy/cupy
            # 支持空下标表示 0-d 标量操作数，从而保证 grad_value 始终参与计算不被丢弃
            grad_input_eqs = [output_eq]
            grad_input_eqs.extend(other_indices)
            info['grad_equation'] = ','.join(grad_input_eqs) + '->' + target_eq
        else:
            # 复杂情况：含重复索引或缺失索引，需偏梯度 + 维度扩展 + 对角线恢复
            info['grad_type'] = 'complex'
            info['has_duplicates'] = has_duplicates
            info['has_ellipsis'] = has_ellipsis
            info['available_indices'] = available_indices
        
        grad_infos.append(info)
    
    return grad_infos


def _einsum_grad(result_tensor: TN):
    """
    einsum 的融合梯度规则。

    一次性计算所有输入操作数的梯度，返回值与 fromvars（操作数列表）按位置对齐，
    不需要梯度的来源返回 None。前向已将反向所需信息存入 parms：
        parms = (equation, input_eqs, output_eq, grad_infos)
    逐来源按 grad_infos[i] 的 grad_type 路由到对应的处理函数。

    调用者: einsum 前向注册的 _grad_rule
    """
    grad_value = result_tensor._grad_value
    input_eqs = result_tensor.parms[1]
    output_eq = result_tensor.parms[2]
    grad_infos = result_tensor.parms[3]
    all_operands = result_tensor.fromvars

    grads = []
    for i, info in enumerate(grad_infos):
        target_operand = all_operands[i]
        if not (isinstance(target_operand, TN) and target_operand.requires_grad):
            grads.append(None)
            continue

        # 获取预计算的梯度信息
        grad_type = info['grad_type']
        target_eq = info['target_eq']

        # 快速路由到对应处理函数
        if grad_type == 'simple':
            # 简单情况：直接使用预计算的梯度方程
            grad = _compute_simple_einsum_grad(
                grad_value, output_eq, all_operands, i, info['grad_equation']
            )
        elif grad_type == 'trace':
            # 迹操作
            grad = _compute_trace_grad(grad_value, target_operand, target_eq)
        elif grad_type == 'diag_extract':
            # 对角线提取
            grad = _compute_diag_extract_grad(grad_value, target_operand, target_eq)
        else:
            # 复杂情况
            other_indices = [eq for j, eq in enumerate(input_eqs) if j != i]
            other_operands = [op for j, op in enumerate(all_operands) if j != i]
            grad = _compute_complex_einsum_grad(
                grad_value, output_eq, other_indices, other_operands, 
                target_eq, target_operand, info
            )

        # 目标省略号批量维可能小于输出（前向发生广播），需按 torch 语义对广播维求和还原形状
        grads.append(_unbroadcast_grad(grad, target_operand.shape))

    return tuple(grads)


@api_registry.register
def einsum(equation: str, *operands: TN) -> TN:
    """
    爱因斯坦求和约定（Einstein Summation Convention）实现。
    
    基于爱因斯坦求和约定，对输入张量进行多维数组的线性代数运算，包括：
    - 矩阵乘法
    - 批量矩阵乘法
    - 张量缩并（contraction）
    - 转置和置换
    - 对角线和迹
    - 外积
    
    本函数接口与 PyTorch 的 torch.einsum() 兼容。

    实现要点（完整调用关系见本节顶部）：
        - 前向：解析方程 → 设备检查 → 选择 optimize/naive 路径 → 记录 parms 与融合梯度规则
        - 反向：按 _precompute_grad_info 预分类的四类 grad_type 路由，
          最后统一经 _unbroadcast_grad 还原前向发生的广播
        - cupy 限制：输出不含 '...' 而输入含 '...' 时，内部改写为保留省略号维再求和
    
    参数:
        equation (str): 爱因斯坦求和方程字符串，格式为 "输入索引->输出索引"
                       或简写形式 "输入索引"（此时隐式输出为按字母顺序排列的非求和索引）
        *operands (TN): 输入张量序列
        
    返回:
        TN: 计算结果张量
        
    示例:
        >>> # 矩阵乘法: ij,jk->ik
        >>> A = rm.randn(3, 4)
        >>> B = rm.randn(4, 5)
        >>> C = rm.einsum('ij,jk->ik', A, B)  # 结果形状 (3, 5)
        
        >>> # 批量矩阵乘法: bij,bjk->bik
        >>> A = rm.randn(10, 3, 4)
        >>> B = rm.randn(10, 4, 5)
        >>> C = rm.einsum('bij,bjk->bik', A, B)  # 结果形状 (10, 3, 5)
        
        >>> # 张量缩并: ijkl,jklm->ijm
        >>> A = rm.randn(3, 4, 5, 6)
        >>> B = rm.randn(4, 5, 6, 7)
        >>> C = rm.einsum('ijkl,jklm->ijm', A, B)  # 结果形状 (3, 7)
        
        >>> # 迹: ii->
        >>> A = rm.randn(5, 5)
        >>> trace = rm.einsum('ii->', A)  # 标量
        
        >>> # 外积: i,j->ij
        >>> a = rm.randn(5)
        >>> b = rm.randn(4)
        >>> C = rm.einsum('i,j->ij', a, b)  # 结果形状 (5, 4)
        
        >>> # 转置: ij->ji
        >>> A = rm.randn(3, 4)
        >>> B = rm.einsum('ij->ji', A)  # 结果形状 (4, 3)
        
        >>> # 对角线提取: ii->i
        >>> A = rm.randn(5, 5)
        >>> diag = rm.einsum('ii->i', A)  # 结果形状 (5,)
        
        >>> # 点积: i,i->
        >>> a = rm.randn(5)
        >>> b = rm.randn(5)
        >>> dot = rm.einsum('i,i->', a, b)  # 标量
    """
    # 验证所有输入都是TN类型
    if not all(isinstance(op, TN) for op in operands):
        raise TypeError("All operands must be TN tensors")
    
    if len(operands) == 0:
        raise ValueError("At least one operand is required")
    
    # 解析方程
    input_eq, input_eqs, output_eq = _parse_einsum_equation(equation, len(operands))
    
    # 获取设备和数组库
    device = operands[0].device
    for op in operands[1:]:
        if not _same_device(operands[0], op):
            raise RuntimeError(f'Expected all tensors to be on the same device, '
                             f'but found at least two devices, {device} and {op.device}!')
    
    arrlib = operands[0]._get_array_lib()
    
    # 检查是否需要梯度
    requires_grad = is_grad_enabled() and any(op.requires_grad for op in operands)
    
    # 前向计算：使用numpy/cupy的einsum，GEMM型大规模缩并启用optimize走BLAS
    operand_data = [op._array for op in operands]
    optimize = _einsum_use_optimize(input_eqs, output_eq, operand_data)
    if arrlib is not np and '...' in input_eq and '...' not in output_eq:
        # cupy的einsum不支持省略号维度被求和的方程（输出不含'...'时直接抛ValueError，
        # 且该校验与optimize无关），改写为输出保留省略号（如'...i->i'改为'...i->...i'），
        # 再对前导省略号维求和，结果与numpy optimize路径及torch语义一致；
        # 反向梯度方程复用本函数，此分支同样生效
        tmp_data = arrlib.einsum(input_eq + '->...' + output_eq, *operand_data,
                                 optimize=optimize)
        n_batch = tmp_data.ndim - len(output_eq)
        result_data = tmp_data.sum(axis=tuple(range(n_batch))) if n_batch > 0 else tmp_data
    else:
        result_data = arrlib.einsum(equation, *operand_data, optimize=optimize)
    
    result = as_tensor(result_data, device=device, requires_grad=requires_grad)
    
    if requires_grad:
        result.fromvars = operands
        # 预计算每个输入的梯度信息，避免反向传播时重复计算
        grad_infos = _precompute_grad_info(input_eqs, output_eq, operands)
        # parms 保存反向所需信息：完整方程、输入/输出索引、每个输入的预计算梯度信息
        result.parms = (equation, input_eqs, output_eq, grad_infos)
        # 融合梯度规则一次性计算所有输入的梯度，按 fromvars 位置对齐
        result._grad_rule = _einsum_grad
    
    return result


# ===========================================================================
# 11. 矩阵与对角线（dot / outer / kron / tril / triu / diagonal / trace / fill_diagonal）
# ===========================================================================


def dot(x:TN,y:TN)->TN:
    """
    计算两个一维张量的点积（内积）。
    
    这个函数与PyTorch的torch.dot()行为一致，只接受1D张量，并计算它们的内积。
    
    Args:
        x (TN): 第一个一维张量
        y (TN): 第二个一维张量
        
    Returns:
        TN: 点积结果，是一个标量张量
        
    Raises:
        RuntimeError: 如果输入张量不是一维的
        
    Examples:
        >>> a = tensor([1, 2, 3])
        >>> b = tensor([4, 5, 6])
        >>> dot(a, b)  # 返回32 (1*4 + 2*5 + 3*6)
    """
    if x._array.ndim != 1:
        raise RuntimeError(f"1D tensors expected, got {x._array.ndim}D tensor")
    if y._array.ndim != 1:
        raise RuntimeError(f"1D tensors expected, got {y._array.ndim}D tensor")
    if not _same_device(x, y):
        raise RuntimeError(f"Input tensors must have the same device, got {x.device} and {y.device}")
    
    # 使用x@y进行计算，利用现有的__matmul__方法的梯度跟踪能力
    return x @ y


def outer(x:TN,y:TN)->TN:
    """
    计算两个一维张量的外积。
    
    这个函数与PyTorch的torch.outer()行为一致，只接受1D张量，并计算它们的外积。
    外积结果是一个形状为(len(x), len(y))的2D张量，其中result[i][j] = x[i] * y[j]。
    
    实现使用einsum('i,j->ij')，相比unsqueeze+乘法，反向梯度计算更快（约2.5倍）。
    
    Args:
        x (TN): 第一个一维张量
        y (TN): 第二个一维张量
        
    Returns:
        TN: 外积结果，是一个形状为(len(x), len(y))的2D张量
        
    Raises:
        RuntimeError: 如果输入张量不是一维的
        
    Examples:
        >>> a = tensor([1, 2, 3])
        >>> b = tensor([4, 5, 6])
        >>> outer(a, b)  # 返回[[4, 5, 6], [8, 10, 12], [12, 15, 18]]
    """
    if x._array.ndim != 1:
        raise RuntimeError(f"1D tensors expected, got {x._array.ndim}D tensor")
    if y._array.ndim != 1:
        raise RuntimeError(f"1D tensors expected, got {y._array.ndim}D tensor")
    if not _same_device(x, y):
        raise RuntimeError(f"Input tensors must have the same device, got {x.device} and {y.device}")
    
    # 使用einsum实现外积，反向梯度计算更高效
    return einsum('i,j->ij', x, y)


def batch_outer(x:TN, y:TN)->TN:
    """
    计算批量向量的外积。
    
    支持任意批次维度的向量外积，结果形状为(..., len(x), len(y))。
    这是outer()的批量扩展版本。
    
    Args:
        x (TN): 输入张量，形状为(..., M)，最后一维为向量长度
        y (TN): 输入张量，形状为(..., N)，最后一维为向量长度
        
    Returns:
        TN: 批量外积结果，形状为(..., M, N)
        
    Raises:
        RuntimeError: 如果输入张量的批次维度不匹配
        
    Examples:
        >>> a = randn(2, 3)  # 2个批次，每个是长度为3的向量
        >>> b = randn(2, 4)  # 2个批次，每个是长度为4的向量
        >>> batch_outer(a, b)  # 形状为(2, 3, 4)
        
        >>> a = randn(2, 3, 4)  # (2,3)批次，每个是长度为4的向量
        >>> b = randn(2, 3, 5)  # (2,3)批次，每个是长度为5的向量
        >>> batch_outer(a, b)  # 形状为(2, 3, 4, 5)
    """
    if not _same_device(x, y):
        raise RuntimeError(f"Input tensors must have the same device, got {x.device} and {y.device}")
    
    # 使用einsum实现批量外积，支持任意批次维度
    return einsum('...i,...j->...ij', x, y)


def kron(input: TN, other: TN, *, out: TN | None = None) -> TN:
    """
    计算 input 和 other 的克罗内克积（Kronecker product），记作 :math:`\\otimes`。
    
    如果 input 是 :math:`(a_0 \\times a_1 \\times \\dots \\times a_n)` 张量，
    other 是 :math:`(b_0 \\times b_1 \\times \\dots \\times b_n)` 张量，
    则结果将是 :math:`(a_0 \\cdot b_0 \\times a_1 \\cdot b_1 \\times \\dots \\times a_n \\cdot b_n)` 张量。
    
    计算公式:
    :math:`(A \\otimes B)_{k_0, k_1, \\dots, k_n} = A_{i_0, i_1, \\dots, i_n} \\cdot B_{j_0, j_1, \\dots, j_n}`
    
    其中 :math:`k_t = i_t \\cdot b_t + j_t`
    
    参数:
        input: 第一个输入张量
        other: 第二个输入张量
        out: 可选的输出张量，用于存储结果
        
    返回:
        克罗内克积结果张量
        
    示例:
        >>> mat1 = rm.eye(3)
        >>> mat2 = rm.ones(5, 3)
        >>> rm.kron(mat1, mat2).shape
        torch.Size([15, 9])
        
        >>> A = rm.tensor([[1, 2], [3, 4]])
        >>> B = rm.tensor([[0, 5], [6, 7]])
        >>> rm.kron(A, B)
        tensor([[ 0,  5,  0, 10],
                [ 6,  7, 12, 14],
                [ 0, 15,  0, 20],
                [18, 21, 24, 28]])
    """
    # 使用einsum计算克罗内克积
    # 对于2D矩阵，克罗内克积可以通过外积和reshape实现
    # A ⊗ B 的 (i0*i1, j0*j1) 位置元素 = A[i0, j0] * B[i1, j1]
    
    input_shape = input.shape
    other_shape = other.shape
    
    # 确保两个张量维度相同
    if len(input_shape) != len(other_shape):
        raise ValueError(f"kron: input and other must have the same number of dimensions, "
                        f"got {len(input_shape)} and {len(other_shape)}")
    
    ndim = len(input_shape)
    
    if ndim == 0:
        # 标量情况
        result = input * other
    elif ndim == 1:
        # 1D向量: 外积后展平
        # input: (a,), other: (b,) -> result: (a*b,)
        result = einsum('i,j->ij', input, other).reshape(-1)
    elif ndim == 2:
        # 2D矩阵: 使用einsum计算外积后reshape
        # input: (a0, a1), other: (b0, b1) -> result: (a0*b0, a1*b1)
        outer = einsum('ij,kl->ikjl', input, other)
        result = outer.reshape(input_shape[0] * other_shape[0], input_shape[1] * other_shape[1])
    else:
        # 高维张量
        # 构建einsum方程: 例如3D时 'abc,def->adbecf'
        input_indices = ''.join(chr(ord('a') + i) for i in range(ndim))
        other_indices = ''.join(chr(ord('a') + ndim + i) for i in range(ndim))
        output_indices = ''.join(f'{input_indices[i]}{other_indices[i]}' for i in range(ndim))
        equation = f'{input_indices},{other_indices}->{output_indices}'
        
        outer = einsum(equation, input, other)
        
        # reshape到最终形状
        result_shape = tuple(input_shape[i] * other_shape[i] for i in range(ndim))
        result = outer.reshape(result_shape)
    
    return _apply_out(out, result, (input, other), "kron")


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


@api_registry.register
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
        ret._grad_rule = _tril_grad
    
    return ret


def _tril_grad(result_tensor: TN):
    """
    tril函数的融合梯度规则。
    下三角矩阵的梯度是保留结果梯度的下三角部分。
    """
    diagonal = result_tensor.parms[0]
    # 重要：使用tril函数获取梯度的下三角部分，而不是numpy.tril
    return (tril(result_tensor._grad_value, diagonal),)


@api_registry.register
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
        ret._grad_rule = _triu_grad
    
    return ret


def _triu_grad(result_tensor: TN):
    """
    triu函数的融合梯度规则。
    上三角矩阵的梯度是保留结果梯度的上三角部分。
    """
    diagonal = result_tensor.parms[0]
    # 重要：使用triu函数获取梯度的上三角部分，而不是numpy.triu
    return (triu(result_tensor._grad_value, diagonal),)


@api_registry.register
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


@api_registry.register
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


@api_registry.register
def trace(input: TN) -> TN:
    """
    返回输入2-D矩阵对角线元素的和。
    
    参数:
        input: 输入的2-D张量
        
    返回:
        对角线元素之和的标量张量
        
    示例:
        >>> x = rm.arange(1., 10.).view(3, 3)
        >>> x
        tensor([[1., 2., 3.],
                [4., 5., 6.],
                [7., 8., 9.]])
        >>> rm.trace(x)
        tensor(15.)
    """
    if input.ndim != 2:
        raise ValueError(f"trace: expected a 2-D tensor, but got {input.ndim}-D tensor")
    
    # 使用 diagonal 函数提取主对角线元素，然后求和
    # diagonal 函数支持非方阵，自动取 min(m, n)
    diag = diagonal(input, offset=0, dim1=0, dim2=1)
    return diag.sum()


@api_registry.register
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


@api_registry.register
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
    result = setat_(input, tuple(indices), fill_value)    
    return result

# end of fill_diagonal_


# ===========================================================================
# 12. 其余工具（masked_fill / clamp / 梯度来源共享）
# ===========================================================================


@api_registry.register
def masked_fill_(input: TN, mask: TN, value: Any) -> TN:
    """
    原地将张量中掩码为 True 的元素填充为指定值，等价于 torch.Tensor.masked_fill_。

    掩码必须是可广播到 input 形状的布尔张量。

    参数:
        input: 输入张量（会被原地修改）
        mask: 布尔类型的张量掩码，形状需与 input 可广播
        value: 用于填充的值，可以是标量或 0 维张量

    返回:
        TN: input 自身（已原地修改）

    异常:
        RuntimeError: input 是需要梯度的叶子节点、mask 不是布尔类型、
            value 不是 0 维张量，或 mask 无法广播到 input 形状
    """
    # 检查是否是需要梯度的叶子节点，原地操作不允许对需要梯度的叶子节点执行
    if is_grad_enabled() and input.requires_grad and input.is_leaf:
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
        if mask.shape != input.shape:
            # 如果需要广播，先广播掩码
            broadcasted_mask = mask.broadcast_to(input.shape)
            _setitem(input, broadcasted_mask, value)
        else:
            _setitem(input, mask, value)
    except Exception as e:
        raise RuntimeError(f"mask with shape {mask.shape} is not broadcastable to tensor shape {input.shape}") from e

    # 返回自身以支持链式调用
    return input


@api_registry.register
def masked_fill(input: TN, mask: TN, value: Any) -> TN:
    """
    使用标量值填充张量中对应掩码为True的位置（非原地操作）
    
    参数:
        input: 输入张量
        mask: 布尔类型的掩码张量，形状需与输入张量广播兼容
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
    # 检查mask是否为布尔类型
    if mask.dtype != np.bool_:
        raise RuntimeError(f"masked_fill only supports boolean masks, but got mask with dtype {mask.dtype}")
    
    # 处理不同类型的value参数
    if isinstance(value, TN):
        if value.ndim != 0:  # type: ignore
            raise RuntimeError(f'masked_fill only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions.')  # type: ignore
    
    # 执行填充操作
    try:
        if mask.shape != input.shape:
            # 如果需要广播，先广播掩码
            broadcasted_mask = mask.broadcast_to(input.shape)
            # 使用where执行非原地填充
            return where(broadcasted_mask, value, input)
        else:
            # 使用where执行非原地填充
            return where(mask, value, input)
    except Exception as e:
        raise RuntimeError(f"mask with shape {mask.shape} is not broadcastable to tensor shape {input.shape}") from e


@api_registry.register
def clamp_(input: TN, min: float | TN | None = None, max: float | TN | None = None) -> TN:
    """
    将张量的每个元素限制在 [min, max] 范围内（原地操作），等价于 torch.Tensor.clamp_。

    参数:
        input: 输入张量（会被原地修改）
        min: 最小值，可以是标量或张量。为 None 时不限制最小值
        max: 最大值，可以是标量或张量。为 None 时不限制最大值

    返回:
        TN: input 自身（已原地修改）

    边界相等时的处理（对齐 PyTorch 的 clamp_）:
        - 标量边界：开区间严格比较。元素恰等于 min/max 时同样改写为边界值，其自身梯度为 0
        - 张量边界：闭区间
          * 元素仅等于一侧边界（x==lo<hi 或 lo<x==hi）：梯度对半分给该元素与该边界张量（各 0.5）
          * 元素同时等于上下界（x==lo==hi）：梯度全部归该元素自身（dx=1）
          * lo>hi（区间无效）：结果全部取 hi，梯度全部归 hi
        - min>max（均为标量时）：不报错，所有元素的结果值都等于 max（即先被抬到 min、再统一压到上界 max）

    注意:
        - 若本张量是需要梯度的叶子节点，会抛出运行时错误
        - min 与 max 必须同为标量或同为张量
    """
    # 若本张量是需要梯度的叶子节点，原地改写会破坏计算图，直接报错（与 PyTorch 一致）
    if is_grad_enabled() and input.requires_grad and input.is_leaf:
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
            setat_(input, input <= min, min)
        if max is not None:
            setat_(input, input >= max, max)
        return input

    # 张量边界：缺失一侧以 ±inf 作哨兵填充，比较时该侧条件恒为假，等价于不设限
    lo = min if min_is_tn else -float('inf')
    hi = max if max_is_tn else float('inf')

    # 原地操作不得改变形状：边界广播后的形状必须与原形状一致（与 PyTorch 一致）
    out_shape = np.broadcast_shapes(input.shape, np.shape(lo), np.shape(hi))
    if out_shape != input.shape:
        raise RuntimeError(f"output with shape {input.shape} doesn't match the broadcast shape {out_shape}")

    def gather(bound, mask):
        # setat_/addat_ 的掩码右值尺寸须等于被选中元素个数：先把边界广播到 input 形状，再按掩码聚集
        b = bound._expand(input.shape) if bound.shape != input.shape else bound
        return b[mask]

    # 相等掩码须在改写数据前、基于原始数据算出（下面的 setat_ 会改动数据）；
    # (lo < hi) 用于排除无效区间 lo>hi，并让三重相等 x==lo==hi 不落入对半分
    t_lo = (input == lo) & (lo < hi)
    t_hi = (input == hi) & (lo < hi)

    # 数据裁剪：先抬下限、再压上限。above 必须在下限改写之后计算，
    # 这样 lo>hi 时被抬到 lo 的元素也能一并压到 hi，其梯度一并转给 hi
    below = input < lo
    setat_(input, below, gather(lo, below) if min_is_tn else lo)
    above = input > hi
    setat_(input, above, gather(hi, above) if max_is_tn else hi)

    # 单边相等（x==lo<hi 或 lo<x==hi）处需把梯度对半分给元素与边界张量：
    # addat_ 使数据变为 2*bound，再 mulat_(0.5) 既还原数据、又把 input 梯度缩为一半
    #（mulat_ 是缩放而非置零），恰好得到 dx=0.5、dbound=0.5；
    # 双重/三重相等被上面的 (lo<hi) 排除，保持 dx=1
    if min_is_tn:
        addat_(input, t_lo, gather(lo, t_lo))
        mulat_(input, t_lo, 0.5)
    if max_is_tn:
        addat_(input, t_hi, gather(hi, t_hi))
        mulat_(input, t_hi, 0.5)

    # 返回自身以支持链式调用
    return input


@api_registry.register
def clamp(x: TN, min: float | TN | None = None, max: float | TN | None = None, out: TN | None = None) -> TN:
    """
    将张量的元素限制在 [min, max] 范围内（非原地）
    
    参数:
        x: 输入张量
        min: 下限值，可以是标量或张量
        max: 上限值，可以是标量或张量
        out: 输出张量（可选）；指定后结果写入 out，且不支持自动微分
    
    返回:
        限制在指定范围内的张量

    边界相等时的处理（对齐 PyTorch 的 clamp）:
        - 标量边界：开区间严格比较。元素恰等于 min/max 时不改写，其自身梯度为 0
        - 张量边界：闭区间
          * 元素仅等于一侧边界（x==lo<hi 或 lo<x==hi）：梯度对半分给该元素与该边界张量（各 0.5）
          * 元素同时等于上下界（x==lo==hi）：梯度全部归该元素（dx=1）
          * lo>hi（区间无效）：结果全部取 hi，梯度全部归 hi
        - min>max（均为标量时）：不报错，所有元素的结果值都等于 max（即先被抬到 min、再统一压到上界 max）
        - min 与 max 必须同为标量或同为张量；两者均为 None 时报 ValueError
    """
    # 参数缺省检查：min 与 max 不能同时为 None
    if min is None and max is None:
        raise ValueError("clamp(): min or max must be specified")
    
    # 标量与张量边界不可混用（与 PyTorch 的重载约束一致）
    min_is_tn = isinstance(min, TN)
    max_is_tn = isinstance(max, TN)
    if (min_is_tn and max is not None and not max_is_tn) or \
       (max_is_tn and min is not None and not min_is_tn):
        raise TypeError("clamp(): min and max must be both scalars or both tensors")

    if not min_is_tn and not max_is_tn:
        # 标量边界：开区间严格比较 —— 仅 x<min 或 x>max 的元素改写为边界值；
        # 元素恰等于边界时不改写、其自身梯度为 0（与 PyTorch 一致）。
        # min>max（均为标量）不报错，所有元素的结果值都等于 max（与 torch/numpy 一致）
        y = x
        if min is not None:
            y = where(x <= min, min, y)
        if max is not None:
            y = where(y >= max, max, y)
    else:
        # 张量边界：缺失一侧以 ±inf 作哨兵填充，比较时该侧条件恒为假，等价于不设限
        lo = min if min_is_tn else -float('inf')
        hi = max if max_is_tn else float('inf')

        # 值：先 max(x,lo) 再 min(hi)；lo>hi 时结果全部落到 hi（与 PyTorch 一致）
        m = where(x < lo, lo, x)
        y = where(m > hi, hi, m)
        # 闭区间下元素可能落在边界上，需按 PyTorch 规则分配梯度：
        # 单边相等（x==lo<hi 或 lo<x==hi）时元素与边界张量各得 0.5 —— 取值用 0.5*(x+bound)，
        # 因 x==bound 数值上恰好等于 bound，而梯度精确为 dx=0.5、dbound=0.5；
        # 三重相等 x==lo==hi 被 (lo<hi) 排除，保持 dx=1；
        # 仅对真实张量边界施加对半分：None 侧为 ±inf 哨兵，若 x 恰为 ±inf 不应参与
        if min_is_tn:
            t_lo = (x == lo) & (lo < hi)
            y = where(t_lo, 0.5 * (x + lo), y)
        if max_is_tn:
            t_hi = (x == hi) & (lo < hi)
            y = where(t_hi, 0.5 * (x + hi), y)

    # 若指定 out，把结果写入其中并返回（out 场景不支持自动微分）
    return _apply_out(out, y, (x, min, max), "clamp")


def attach_zero_grad_sources(input:TN, sources:tuple|list) -> TN:
    """
    返回一个新张量：值不变，梯度透传给 input，同时向 sources 传递全 0 梯度。

    该函数用于把张量加入计算图，但不改变现有计算图的前向计算值和反向传播计算的梯度值。
    与原张量不同，返回的是一个新的计算图节点，其 fromvars 为
    (input, source1, source2, ...)，因此不会原地修改任何已有节点的计算图结构。

    参数:
        input: 要依附来源的张量。
        sources: 要依附的来源张量集合，可以是tuple、list或set。
                只有需要梯度(requires_grad=True)且不是input本身的张量才会被依附。

    返回:
        TN: 新张量；没有有效来源时原样返回 input。

    示例:
        >>> a = tensor([1.0, 2.0], requires_grad=True)
        >>> b = tensor([3.0, 4.0], requires_grad=True)
        >>> c = tensor([5.0, 6.0], requires_grad=True)
        >>> d = attach_zero_grad_sources(c, [a, b])  # 将a,b加入d的计算图
        >>> # d的值与c相同，但backward时d会向a,b传递0梯度
    """
    # 过滤有效来源张量（需要梯度且不是input本身）
    valid_sources = [
        var for var in sources
        if var is not input and isinstance(var, TN) and var.requires_grad
    ]
    if not valid_sources:
        return input

    # 新节点的来源顺序固定为 (原张量, 依附来源...)，融合梯度按同一顺序返回：
    # 第0位透传梯度，其余位置为零梯度
    result = TN(input._array.copy(), requires_grad=True)
    result.fromvars = (input,) + tuple(valid_sources)

    def _attach_zero_grad_sources_grad(result:TN):
        return (result._grad_value,) + tuple(
            zeros_like(var) for var in result.fromvars[1:]
        )

    result._grad_rule = _attach_zero_grad_sources_grad
    return result


def map_value_connect_grad(tensors:tuple|list):
    """
    为一组张量生成一一对应的副本，副本序列与原张量序列构成全连接梯度图。

    每个副本的值与其对应原张量相同。反向传播时，副本与对应张量之间透传梯度，
    副本与非对应张量之间传递零梯度，因此组内每个张量都能获得梯度
    （未参与前向计算的张量得到零梯度），而前向计算值和原有梯度值均不受影响。

    实现上，对副本 t 而言，fromvars 为 (t, 其余原张量...)：第0位透传梯度，
    其余位置为零梯度，由副本的融合梯度规则 _map_value_connect_grad_rule 按序返回。

    参数:
        tensors: 输入张量序列，必须是tuple或list类型。
                序列中的每个张量都会被处理：
                - 需要梯度的张量会被复制，并将其他张量（排除自身）作为零梯度来源依附
                - 不需要梯度的张量或非TN对象保持原样

    返回:
        与输入类型相同的张量序列（tuple或list）。
        返回的张量与输入张量值相同，但计算图已相互连接。
        注意：对于需要梯度的张量，返回的是数据副本而非原始张量。

    异常:
        ValueError: 当tensors不是tuple或list类型时抛出。

    示例:
        >>> a = tensor([1.0, 2.0], requires_grad=True)
        >>> b = tensor([3.0, 4.0], requires_grad=True)
        >>> c = tensor([5.0, 6.0], requires_grad=True)
        >>>
        >>> # 定义前向计算函数，只使用a和b
        >>> def func(t):
        ...     a, b, c = t
        ...     return (a * b).sum()
        >>>
        >>> # 映射前：c不参与计算，不会收到梯度
        >>> y1 = func((a, b, c))
        >>> y1.backward()
        >>> print(f"a.grad={a.grad}, b.grad={b.grad}, c.grad={c.grad}")
        a.grad=tensor([3., 4.]), b.grad=tensor([1., 2.]), c.grad=None
        >>>
        >>> # 重置张量
        >>> a = tensor([1.0, 2.0], requires_grad=True)
        >>> b = tensor([3.0, 4.0], requires_grad=True)
        >>> c = tensor([5.0, 6.0], requires_grad=True)
        >>>
        >>> # 映射后：所有张量相互连接，c会收到0梯度
        >>> a_new, b_new, c_new = map_value_connect_grad((a, b, c))
        >>> y2 = func((a_new, b_new, c_new))
        >>> y2.backward()
        >>> print(f"a.grad={a_new.grad}, b.grad={b_new.grad}, c.grad={c_new.grad}")
        a.grad=tensor([3., 4.]), b.grad=tensor([1., 2.]), c.grad=tensor([0., 0.])
        >>>
        >>> # 验证：前向值不变，a和b梯度不变，c从None变为0
        >>> assert float(y1._array) == float(y2._array)
        >>> assert (a_new.grad == tensor([3., 4.])).all()
        >>> assert (b_new.grad == tensor([1., 2.])).all()
        >>> assert (c_new.grad == tensor([0., 0.])).all()
    """
    if not isinstance(tensors,(tuple,list)):
        raise ValueError("'tensors' need to be tuple or list of TN tensors")

    # 只有需要梯度的张量参与共享，其余张量不进入零梯度依附
    valid_tensors = [t for t in tensors if isinstance(t,TN) and t.requires_grad]

    def _map_value_connect_grad_rule(result:TN):
        # fromvars 为 (原始输出, 其他输出1, ...)：第0位透传梯度，其余为零梯度
        return (result._grad_value,) + tuple(
            zeros_like(var) for var in result.fromvars[1:]
        )

    target_list = []
    for t in tensors:
        if isinstance(t,TN) and t.requires_grad:
            # 直接构造输出副本：fromvars 与融合梯度规则由本函数一并确立，
            # 顺序固定为 (原始输出, 其他输出...)，与 _map_value_connect_grad_rule 的返回顺序一致
            others = tuple(s for s in valid_tensors if s is not t)
            node = TN(t._array.copy(), requires_grad=is_grad_enabled())
            if node.requires_grad:
                node.fromvars = (t,) + others
                node._grad_rule = _map_value_connect_grad_rule
            target_list.append(node)
        else:
            target_list.append(t)

    return type(tensors)(target_list)


# ---------------------------------------------------------------------------
# 张量显示：_disp 及其辅助函数
#
# 调用关系：
#   _disp(t, format_spec)
#     ├─ 步骤1 属性后缀：device / size / dtype / requires_grad（规则与torch一致）
#     ├─ 步骤2 _parse_format_spec：解析出 (width, precision, format_type)
#     ├─ 步骤3 _tensor_data_str：生成数据部分（二选一）
#     │    ├─ torch默认规则：_print_summarized_values → _print_number_formatter
#     │    │                或 _print_plain_formatter → _torch_layout
#     │    └─ 显式 format_spec：_print_spec_formatter + numpy array2string
#     ├─ 步骤4 _wrap_tensor_text：加 "tensor(" 前缀与续行缩进
#     └─ 步骤5 _add_suffixes：追加属性并补齐右括号，超行宽时折行
#
# 约定：所有 formatter 均为 (format_value, width) 二元组，
#       format_value(value) 返回已按 width 右对齐的字符串。
# ---------------------------------------------------------------------------
def _print_number_formatter(values, arrlib):
    """按torch默认打印规则构造浮点/复数的格式化器（precision=4）。

    values: 用于判断的数值数组（实部或虚部，大张量时只传首尾采样值）。
    与torch一致：非零有限值全为整数时使用整数模式（末尾补'.'）；
    max/min>1000、max>1e8 或 min<1e-4 时使用科学计数法；否则固定4位小数。

    返回:
        (format_value, width): 单值格式化函数（已按width右对齐）与对齐宽度
    """
    finite = arrlib.isfinite(values)
    nonzero = values[finite & (values != 0)]
    int_mode = True
    sci_mode = False
    vmin = vmax = 0.0
    if nonzero.size:
        abs_nz = arrlib.abs(nonzero)
        abs_min = float(abs_nz.min())
        abs_max = float(abs_nz.max())
        # 对齐宽度取带符号极值，避免无负值时多算一个符号位
        vmin = float(nonzero.min())
        vmax = float(nonzero.max())
        int_mode = bool(arrlib.all(nonzero == arrlib.floor(nonzero)))
        sci_mode = (abs_max / abs_min > 1000.0) or (abs_max > 1.0e8) or (abs_min < 1.0e-4)

    if sci_mode:
        width = builtins.max(1, len(f'{vmax:.4e}'), len(f'{vmin:.4e}'))

        def fmt(x):
            return f'{x:>{width}.4e}'
    elif int_mode:
        width = builtins.max(1, len(f'{vmax:.0f}') + 1, len(f'{vmin:.0f}') + 1)

        def fmt(x):
            # 整数模式下非有限值不加'.'，用于标识浮点类型
            if x != x or builtins.abs(x) == builtins.float('inf'):
                return f'{x:>{width}}'
            return f'{x:>{width - 1}.0f}.'
    else:
        width = builtins.max(1, len(f'{vmax:.4f}'), len(f'{vmin:.4f}'))

        def fmt(x):
            return f'{x:>{width}.4f}'

    return fmt, width


def _print_plain_formatter(values, arrlib):
    """整数/布尔的默认格式化器：按最大字符宽度右对齐（与torch一致）。

    返回:
        (format_value, width): 单值格式化函数与对齐宽度
    """
    items = values.reshape(-1).tolist()
    width = builtins.max([1] + [len(str(v)) for v in items])

    def fmt(x):
        s = str(x)
        return ' ' * (width - len(s)) + s

    return fmt, width


def _print_summarized_values(arr, arrlib, edgeitems: int = 3):
    """按torch get_summarized_data 的规则收集元素，用于确定格式化参数。

    大张量（元素数超过阈值）只显示每维首尾各 edgeitems 个元素，
    因此列宽、整数模式、科学计数法也只需基于这些采样值判断。
    """
    if arr.ndim <= 1:
        if arr.shape[0] > 2 * edgeitems:
            return arrlib.concatenate([arr[:edgeitems], arr[-edgeitems:]])
        return arr
    n = arr.shape[0]
    if n > 2 * edgeitems:
        rows = list(range(edgeitems)) + list(range(n - edgeitems, n))
    else:
        rows = list(range(n))
    parts = [_print_summarized_values(arr[i], arrlib, edgeitems).reshape(-1) for i in rows]
    return arrlib.concatenate(parts) if parts else arr


def _print_scalar(value, fmt1, fmt2=None):
    """格式化单个元素。

    fmt1/fmt2: 实部/虚部的 (format_value, width) 二元组，虚部为空表示实数。
    复数与torch一致：虚部先格式化再去空白，按首字符判断是否需要补 '+'。
    """
    if fmt2 is None:
        return fmt1[0](value)
    real_str = fmt1[0](value.real)
    imag_str = (fmt2[0](value.imag) + 'j').lstrip()
    if imag_str[0] in ('+', '-'):
        return real_str + imag_str
    return real_str + '+' + imag_str


def _torch_layout(arr, fmt1, fmt2=None, indent: int = 0, edgeitems: int = 3,
                  summarize: bool = False, linewidth: int = 80,
                  prefix_width: int = 7) -> str:
    """按torch的排布算法生成数值部分的字符串（不含 tensor() 前缀与属性）。

    fmt1/fmt2: (format_value, width) 二元组，fmt2 仅复数（虚部）使用。
    - 每行元素数由元素宽度、行宽与缩进决定（与torch的 _vector_str 一致）
    - 超过阈值时每维只显示首尾 edgeitems 个元素，中间用 '...' 省略
    - 多维按 ',' + '\\n'*(dim-1) 连接，嵌套层相对缩进 indent+1
      （绝对缩进由外层 format_text 的 tensor( 前缀补齐，prefix_width 参与行宽计算）
    """
    dim = arr.ndim
    if dim == 0:
        return _print_scalar(arr.item(), fmt1, fmt2)

    if dim == 1:
        element_length = fmt1[1] + 2
        if fmt2 is not None:
            element_length += fmt2[1] + 1
        per_line = builtins.max(1, (linewidth - prefix_width - indent) // element_length)
        n = arr.shape[0]
        if summarize and n > 2 * edgeitems:
            data = ([_print_scalar(v, fmt1, fmt2) for v in arr[:edgeitems].tolist()]
                    + [' ...']
                    + [_print_scalar(v, fmt1, fmt2) for v in arr[-edgeitems:].tolist()])
        else:
            data = [_print_scalar(v, fmt1, fmt2) for v in arr.tolist()]
        lines = [', '.join(data[i:i + per_line]) for i in range(0, len(data), per_line)]
        return '[' + (',' + '\n' + ' ' * (indent + 1)).join(lines) + ']'

    n = arr.shape[0]
    if summarize and n > 2 * edgeitems:
        slices = ([_torch_layout(arr[i], fmt1, fmt2, indent + 1, edgeitems, summarize,
                                 linewidth, prefix_width)
                   for i in range(edgeitems)]
                  + ['...']
                  + [_torch_layout(arr[i], fmt1, fmt2, indent + 1, edgeitems, summarize,
                                   linewidth, prefix_width)
                     for i in range(n - edgeitems, n)])
    else:
        slices = [_torch_layout(arr[i], fmt1, fmt2, indent + 1, edgeitems, summarize,
                                linewidth, prefix_width)
                  for i in range(n)]
    return '[' + (',' + '\n' * (dim - 1) + ' ' * (indent + 1)).join(slices) + ']'


def _add_suffixes(tensor_str: str, suffixes, indent: int = 7, linewidth: int = 80) -> str:
    """追加属性后缀并补齐右括号；超出行宽时折行并对齐到 indent 个空格。

    tensor_str 为不含右括号的 "tensor(...)" 主体，返回值始终以 ')' 结尾。
    """
    parts = [tensor_str]
    last_line_len = len(tensor_str) - (tensor_str.rfind('\n') + 1)
    for suffix in suffixes:
        suffix_len = len(suffix)
        if last_line_len + suffix_len + 2 > linewidth:
            parts.append(',\n' + ' ' * indent + suffix)
            last_line_len = indent + suffix_len
        else:
            parts.append(', ' + suffix)
            last_line_len += suffix_len + 2
    parts.append(')')
    return ''.join(parts)


def _parse_format_spec(format_spec: str):
    """解析 format_spec，返回 (width, precision, format_type)。

    支持 '.2f'、'8.4f'、'.1e'、'05d' 等写法；
    precision 为 None 表示未显式指定，此时使用torch默认规则。
    """
    format_type = 'f'  # 默认格式类型
    width = None  # None 表示不指定宽度
    precision = None  # None 表示使用torch默认规则（4位小数等）

    if '.' in format_spec:
        # 小数点前为宽度
        width_digits = ''.join(filter(str.isdigit, format_spec.split('.')[0]))
        if width_digits:
            width = int(width_digits)

        # 小数点后为精度，末尾字母为格式类型
        precision_part = format_spec.split('.')[1]
        precision_digits = ''.join(filter(str.isdigit, precision_part))
        if precision_digits:
            precision = int(precision_digits)
        type_part = ''.join(filter(str.isalpha, precision_part))
        if type_part:
            format_type = type_part[0]
    else:
        # 没有小数点时只解析宽度与格式类型
        width_digits = ''.join(filter(str.isdigit, format_spec))
        if width_digits:
            width = int(width_digits)
        type_part = ''.join(filter(str.isalpha, format_spec))
        if type_part:
            format_type = type_part[0]

    return width, precision, format_type


def _print_spec_formatter(width, precision: int, format_type: str, offset: int = 0):
    """按 format_spec 构造定宽右对齐的格式化器。

    offset: 宽度偏移，复数虚部用 width-1（符号单独拼接，不占用虚部宽度）。
    """
    if width is None:
        return lambda x: f'{x:.{precision}{format_type}}'
    target_width = width - offset
    return lambda x: f'{x:>{target_width}.{precision}{format_type}}'


def _tensor_data_str(t: TN, arrlib, width, precision, format_type: str) -> str:
    """生成张量数据部分的字符串（不含 tensor() 前缀与属性）。

    未显式指定 precision，或类型为整数/布尔时，走torch默认规则
    （整数/布尔忽略 spec）；否则浮点/复数按 format_spec 用 numpy 格式化。
    """
    is_complex = arrlib.issubdtype(t._array.dtype, arrlib.complexfloating)
    is_floating = arrlib.issubdtype(t._array.dtype, arrlib.floating)

    if precision is None or not (is_complex or is_floating):
        # 与torch一致：超过1000个元素时每维只显示首尾3个元素，
        # 并按torch的排布算法决定每行元素数与省略号形式
        edgeitems = 3
        summarize = t._array.size > 1000
        # 列宽/整数模式/科学计数法只需基于实际显示的采样值判断
        params = (_print_summarized_values(t._array, arrlib, edgeitems)
                  if summarize else t._array)
        fmt2 = None  # 只有复数才有虚部格式化器
        if is_complex:
            fmt1 = _print_number_formatter(params.real, arrlib)
            fmt2 = _print_number_formatter(params.imag, arrlib)
        elif is_floating:
            fmt1 = _print_number_formatter(params, arrlib)
        else:
            fmt1 = _print_plain_formatter(params, arrlib)
        # 空张量统一显示为 []，形状信息由 size 后缀补充
        if t._array.size == 0:
            return '[]'
        return _torch_layout(t._array, fmt1, fmt2, 0, edgeitems, summarize)

    # 显式 format_spec：实部与虚部共用同一规则，虚部宽度减1
    real_fmt = _print_spec_formatter(width, precision, format_type)
    if is_complex:
        imag_fmt = _print_spec_formatter(width, precision, format_type, offset=1)

        def complex_formatter(z):
            imag_sign = '+' if z.imag >= 0 else '-'
            return f'{real_fmt(z.real)}{imag_sign}{imag_fmt(builtins.abs(z.imag))}j'

        formatter = {'complex_kind': complex_formatter}
    else:
        formatter = {'float_kind': real_fmt}
    return arrlib.array2string(
        t._array,
        separator=', ',
        formatter=formatter,  # type: ignore
        threshold=1000  # 与torch一致：超过1000个元素时省略中间部分
    )


def _wrap_tensor_text(text: str, prefix: str = 'tensor') -> str:
    """给数据部分加前缀与左括号，续行缩进 len(prefix)+1。

    右括号不在此处添加，由 _add_suffixes 统一收尾。
    """
    # 去掉各行尾随空格；块间空行保持为空行，避免缩进后留下纯空白行
    lines = [line.rstrip() for line in text.split('\n')]
    if not lines:
        return ""

    lines[0] = f"{prefix}({lines[0]}"
    indent = ' ' * (len(prefix) + 1)
    for i in range(1, len(lines)):
        if not lines[i]:
            continue
        lines[i] = f"{indent}{lines[i]}"

    return '\n'.join(lines)

@api_registry.register
def _disp(t: TN, format_spec: str) -> str:
    """TN.__disp__ 的实现：按 format_spec 返回张量的显示字符串。"""
    # 获取默认浮点类型用于比较
    default_float = get_default_dtype()
    default_complex = get_default_complex()

    # 获取数组库
    arrlib = t._get_array_lib()

    # 步骤1：构建属性后缀（顺序与torch一致：device → size → dtype → requires_grad）
    attrs = []
    if t.device != Device('cpu'):
        attrs.append(f"device='{t.device}'")
    if t._array.size == 0:
        # 与torch一致：空张量显式给出形状（1维时省略），dtype 仅与默认浮点比较
        if t.ndim != 1:
            attrs.append(f"size={tuple(t.shape)}")
        if t.dtype != default_float:
            attrs.append(f"dtype={t.dtype.name}")
    elif t.dtype not in (default_float, default_complex, np.int64, np.bool_):
        attrs.append(f"dtype={t.dtype.name}")
    if t.requires_grad:
        attrs.append("requires_grad=True")

    # 步骤2：解析 format_spec（width/precision/format_type）
    width, precision, format_type = _parse_format_spec(format_spec)

    # 步骤3：生成数据部分字符串
    data_str = _tensor_data_str(t, arrlib, width, precision, format_type)

    # 步骤4：加 tensor( 前缀（不含右括号）
    # 步骤5：追加属性后缀并统一补齐右括号（超过行宽时折行）
    return _add_suffixes(_wrap_tensor_text(data_str), attrs)
