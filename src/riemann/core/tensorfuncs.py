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
from .gradmode import *
from .tensordef import *
from .tensordef import (
    _validate_shape,
    _get_broadcast_axis, _apply_out, _apply_out_tuple, 
    _as_tn, _get_device, _is_array_type
)

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
    if x.device != y.device:
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
    if x.device != y.device:
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
    if x.device != y.device:
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

def abs_(x:TN)->TN:
    """
    就地取绝对值，等价于 torch.abs_。

    直接修改输入张量的数据并返回它本身。布尔张量、复数张量不支持（与 torch 一致）。

    Args:
        x (TN): 输入张量

    Returns:
        TN: 修改后的输入张量（与输入是同一对象）

    Examples:
        >>> a = tensor([-1., 2., -3.])
        >>> abs_(a)  # a 变为 [1., 2., 3.]，返回 a 本身
    """
    x = _as_tn(x)
    return x.abs_()

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
    x = _as_tn(x)
    return x ** 2.0

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

def neg_(x)->TN:
    """
    就地取负，等价于 torch.neg_。

    直接修改输入张量的数据并返回它本身。布尔张量不支持取负，
    无符号整型按位宽回绕并保持 dtype（如 uint8 的 -1 得到 255）。

    Args:
        x (TN): 输入张量

    Returns:
        TN: 修改后的输入张量（与输入是同一对象）

    Examples:
        >>> a = tensor([1., -2.])
        >>> neg_(a)  # a 变为 [-1., 2.]，返回 a 本身
    """
    x = _as_tn(x)
    return x.neg_()

def exp2(x:TN)->TN:
    x = _as_tn(x)
    return 2.0 ** x


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
            subt.gradfuncs = (_split_backward,)
        
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

def _split_backward(result_tensor: TN, i: int) -> TN:
    split_sizes, dim, split_pos = result_tensor.parms[i]
    parent = result_tensor.fromvars[i]
    
    grad = zeros_like(parent)

    # 计算起始和结束位置
    start = builtins.sum(split_sizes[:split_pos]) if split_pos > 0 else 0
    end = start + split_sizes[split_pos]
    
    # 累加归一化后的梯度
    slice_obj = [slice(None)] * parent._array.ndim
    slice_obj[dim] = slice(int(start), int(end))
    grad[tuple(slice_obj)] = result_tensor.grad_value
    
    return grad

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

@api_registry.register("stack")
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
        ret.gradfuncs = (_stack_backward,) * num
    
    return ret

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
        ret.gradfuncs = (_concatenate_backward,) * num
    
    return ret

def _stack_backward(result_tensor: TN, i: int) -> TN:
    dim = result_tensor.parms[i]
    grad = result_tensor.grad_value
    slices = [slice(None)] * grad._array.ndim
    slices[dim] = i  # type: ignore
    return grad[tuple(slices)]

def _concatenate_backward(result_tensor: TN, i: int) -> TN:
    dim = result_tensor.parms[i]
    grad_data = result_tensor.grad_value
    
    # 获取所有输入张量在拼接轴上的尺寸
    split_sizes = [t._array.shape[dim] for t in result_tensor.fromvars]
    
    # 使用Python内置的accumulate函数计算累积和，避免不必要的数组转换
    split_points = list(accumulate(split_sizes[:-1])) if len(split_sizes) > 1 else []
    
    try:
        # 使用 _split_with_indices 进行分割（numpy风格语义）
        grads = _split_with_indices(grad_data, split_points, dim=dim)
        return grads[i]
    
    except ValueError as e:
        raise RuntimeError(f"梯度分割失败: {e}") from None

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

@api_registry.register("unfold")
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

@api_registry.register("fold")
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
        output.addat_(tuple(output_idx),window)

    return output


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
        if other.device != input.device:
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
        if other.device != input.device:
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
        if other.device != input.device:
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
        result: sumall() 的输出张量，可通过 result.grad_value 获取上游梯度
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
            grads.append(result.grad_value)
        else:
            # 计算广播轴并对grad_value进行求和
            broadcast_axes = _get_broadcast_axis(result.shape, input_shape)
            grads.append(sum(result.grad_value, dim=broadcast_axes, keepdim=False)._reshape(input_shape))
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
    bad_arg = next((arg for arg in tensor_args if arg.device != target_device), None)
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
           _is_array_type(x)
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

def ceil(x:TN)->TN:
    """
    计算张量的上取整。

    返回:
        包含输入张量上取整的新张量
    """
    x = _as_tn(x)
    arrlib = x._get_array_lib()
    return as_tensor(arrlib.ceil(x._array),device=x.device)

def trunc(x:TN)->TN:
    """
    计算张量的截断取整。

    返回:
        包含输入张量截断取整的新张量
    """
    x = _as_tn(x)
    arrlib = x._get_array_lib()
    return as_tensor(arrlib.trunc(x._array),device=x.device)

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

def _topk_backward(result_tensor: TN, i: int) -> TN:
    """
    topk操作的反向传播函数。
    将输出位置上的梯度按照topk索引散射回输入张量的对应位置，
    未被选中的输入位置梯度为0。
    """
    # 获取原始输入张量
    input_tensor = result_tensor.fromvars[i]
    # 获取topk选择时使用的索引和维度
    topk_indices, topk_dim = result_tensor.parms[i]
    # 获取梯度值
    grad_value = result_tensor.grad_value

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

    return grad


@api_registry.register('topk')
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
        values_tensor.gradfuncs = (_topk_backward,)

    # 处理out参数
    return _apply_out_tuple(out, (values_tensor, indices_tensor), (input,), "topk")


@api_registry.register('multinomial')
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


def _sin_grad(result_tensor:TN, i:int)->TN:
    x = result_tensor.fromvars[i]
    return result_tensor.grad_value * cos(x).conj()

@api_registry.register('sin')
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
    x = _as_tn(x)
    arrlib = x._get_array_lib()
    ret = TN(arrlib.sin(x._array), is_grad_enabled() and x.requires_grad)
        
    if ret.requires_grad:
        ret.fromvars = (x,)
        ret.gradfuncs = (_sin_grad,)

    return ret
    
def _cos_grad(result_tensor:TN, i:int)->TN:
    x = result_tensor.fromvars[i]
    return result_tensor.grad_value * (-sin(x).conj())

@api_registry.register('cos')
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
    x = _as_tn(x)
    arrlib = x._get_array_lib()
    ret = TN(arrlib.cos(x._array), is_grad_enabled() and x.requires_grad)
    
    if ret.requires_grad:
        ret.fromvars = (x,)
        ret.gradfuncs = (_cos_grad,)

    return ret

def _tan_grad(result_tensor:TN, i:int)->TN:
    x = result_tensor.fromvars[i]
    return result_tensor.grad_value * (1. + (tan(x.conj()))**2.)
    
@api_registry.register('tan')
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
    x = _as_tn(x)
    arrlib = x._get_array_lib()
    ret = TN(arrlib.tan(x._array), is_grad_enabled() and x.requires_grad)
    
    if ret.requires_grad:
        ret.fromvars = (x,)
        ret.gradfuncs = (_tan_grad,)

    return ret

@api_registry.register('cot')
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

@api_registry.register('sec')
def sec(x:TN)->TN:
    return 1./cos(x)

@api_registry.register('csc')
def csc(x:TN)->TN:
    return 1./sin(x)

def _arcsin_derivative(result:TN, x:TN)->tuple[TN]:
    return (result.grad_value * (1./sqrt(1. - x**2.).conj()),)

@api_registry.register('arcsin')
@track_grad(_arcsin_derivative)
def arcsin(x:TN)->TN:
    x = _as_tn(x)
    arrlib = x._get_array_lib()
    return TN(arrlib.arcsin(x._array))

def _arccos_derivative(result:TN, x:TN)->tuple[TN]:
    return (result.grad_value * (-1.0/sqrt(1. - x**2.).conj()),)

@api_registry.register('arccos')
@track_grad(_arccos_derivative)
def arccos(x:TN)->TN:
    x = _as_tn(x)
    arrlib = x._get_array_lib()
    return TN(arrlib.arccos(x._array))

def _arctan_derivative(result:TN, x:TN)->tuple[TN]:
    return (result.grad_value * (1./(1. + x**2.).conj()),)

@api_registry.register('arctan')
@track_grad(_arctan_derivative)
def arctan(x:TN)->TN:
    x = _as_tn(x)
    arrlib = x._get_array_lib()
    return TN(arrlib.arctan(x._array))

def _sinh_derivative(result:TN, x:TN)->tuple[TN]:
    return (result.grad_value * cosh(x).conj(),)

@api_registry.register('sinh')
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
    x = _as_tn(x)
    arrlib = x._get_array_lib()
    return TN(arrlib.sinh(x._array))

def _cosh_derivative(result:TN, x:TN)->tuple[TN]:
    return (result.grad_value * sinh(x).conj(),)

@api_registry.register('cosh')
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
    x = _as_tn(x)
    arrlib = x._get_array_lib()
    return TN(arrlib.cosh(x._array))

def _tanh_derivative(result:TN, x:TN)->tuple[TN]:
    return (result.grad_value * (1.0 - tanh(x)**2.0).conj(),)

@api_registry.register('tanh')
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
    x = _as_tn(x)
    arrlib = x._get_array_lib()
    return TN(arrlib.tanh(x._array))

@api_registry.register('coth')
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

@api_registry.register('sech')
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

@api_registry.register('csch')
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
    return (result.grad_value * (1. / sqrt(x**2. + 1.).conj()),)
    
@api_registry.register('arcsinh')
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
    x = _as_tn(x)
    arrlib = x._get_array_lib()
    return TN(arrlib.arcsinh(x._array))

def _arccosh_derivative(result:TN, x:TN)->tuple[TN]:
    return (result.grad_value * (1. / sqrt(x**2. - 1.).conj()),)

@api_registry.register('arccosh')
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
    x = _as_tn(x)
    arrlib = x._get_array_lib()
    return TN(arrlib.arccosh(x._array))

def _arctanh_derivative(result:TN, x:TN)->tuple[TN]:
    return (result.grad_value * (1. / (1. - x**2.0).conj()),)

@api_registry.register('arctanh')
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
    x = _as_tn(x)
    arrlib = x._get_array_lib()
    return TN(arrlib.arctanh(x._array))


@api_registry.register('repeat')
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
        def _repeat_backward(result_tensor: TN, i: int) -> TN:
            # 重复操作的梯度是将结果的梯度在重复的维度上求和
            grad_value = result_tensor.grad_value
            
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
            
            return grad_value
        
        result.fromvars = (input,)
        result.gradfuncs = (_repeat_backward,)
    
    return result


@api_registry.register('masked_fill')
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


@api_registry.register('clamp')
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
    
    return grad

@api_registry.register('sort')
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
        # 获取指定维度的大小
        dim_size = input._array.shape[dim]
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
        sorted_values.gradfuncs = (_sort_backward,)
    
    # 处理out参数
    return _apply_out_tuple(out, (sorted_values, sorted_indices_tensor), (input,), "sort")


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

@api_registry.register('prod')
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
        ret.gradfuncs = (_prod_backward,)
    
    return ret


@api_registry.register('index_select')
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


@api_registry.register('masked_select')
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


@api_registry.register('take')
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


@api_registry.register('index_put_')
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
        return input.addat_(indices, values)
    return input.setat_(indices, values)


@api_registry.register('scatter')
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
            raise RuntimeError(f"scatter(): Expected input.dtype to be equal to src.dtype")

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


@api_registry.register('scatter_')
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
            raise RuntimeError(f"scatter_(): Expected input.dtype to be equal to src.dtype")
        
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
    return input.setat_(full_index, fill_value)
# end of scatter_


@api_registry.register('scatter_add')
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
        raise RuntimeError(f"scatter_add_(): Expected input.dtype to be equal to src.dtype")

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


@api_registry.register('scatter_add_')
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
        raise RuntimeError(f"scatter_add_(): Expected input.dtype to be equal to src.dtype")

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
    return input.addat_(full_index, src)
# end of scatter_add_


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


@api_registry.register('gather')
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
    """检查target_eq是否包含重复索引（忽略省略号...）。"""
    index_counts = {}
    for c in target_eq:
        if c != '.':  # 忽略省略号
            index_counts[c] = index_counts.get(c, 0) + 1
    return any(count > 1 for count in index_counts.values())


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


@api_registry.register('einsum')
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
        if op.device != device:
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
        # 所有输入共享同一个外置梯度函数，反向时用 i 参数区分当前求导的输入
        result.gradfuncs = (_einsum_backward,) * len(operands)
    
    return result

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
        每个输入对应一个 info 字典，含 grad_type 及该类所需的辅助字段
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

def _einsum_backward(result_tensor: TN, i: int) -> TN:
    """
    einsum 的反向传播函数（外置版本）。

    所有操作数共享这同一个梯度函数，通过参数 i（当前输入在操作数列表中的位置）
    区分本次要计算的是哪个输入的梯度。前向已将反向所需信息存入 parms：
        parms = (equation, input_eqs, output_eq, grad_infos)
    本函数按 grad_infos[i] 的 grad_type 路由到对应的处理函数。
    """
    grad_value = result_tensor.grad_value
    input_eqs = result_tensor.parms[1]
    output_eq = result_tensor.parms[2]
    grad_infos = result_tensor.parms[3]
    all_operands = result_tensor.fromvars
    
    # 获取预计算的梯度信息
    info = grad_infos[i]
    grad_type = info['grad_type']
    target_eq = info['target_eq']
    target_operand = all_operands[i]
    
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
    return _unbroadcast_grad(grad, target_operand.shape)


def _compute_simple_einsum_grad(
    grad_value: TN, output_eq: str, 
    all_operands: list[TN], i: int, grad_equation: str
) -> TN:
    """
    计算简单einsum操作的梯度（无重复索引）。

    直接使用预计算的梯度方程，性能最优。
    grad_value的方程为output_eq（空输出时为''），确保非单位梯度正确参与计算。
    """
    grad_inputs = [grad_value]

    for j, op in enumerate(all_operands):
        if j != i:
            grad_inputs.append(op.conj() if op.is_complex() else op)

    return einsum(grad_equation, *grad_inputs)


def _ellipsis_after_indices(target_eq: str, rank: int) -> bool:
    """
    判断目标的省略号是否位于全部显式索引之后（如 'ii...'、'ij...k'）。

    用于区分布局：省略号在索引之前/中间时，重复索引对占据最前两维的反面——省略号在被
    重复的索引对之后时，该索引对才真正落在目标的最前两维上。
    """
    return '...' in target_eq and rank > len(target_eq.replace('.', '')) and \
        len(target_eq.split('...')[0]) == len(target_eq.replace('.', ''))


def _compute_trace_grad(grad_value: TN, target_operand: TN, target_eq: str) -> TN:
    """
    计算迹操作（ii->、bii->b、...ii->...、...ii->、ii...->、ii...->...）的梯度。

    迹在前向只对重复索引对应的两维取对角元素求和，反向即构造一个对角张量：对角元素为
    grad_value 的对应值、非对角为0。注意 grad_value 可能缺少目标的部分前导维度
    （如 ...ii-> 的输出是标量、bii->b 只含批量维），需先广播到目标的批量形状。

    重复索引对默认位于目标最后两维；但当省略号位于该索引对之后（如 ii...）时，该对实际
    落在目标最前两维，此时 grad_value 对应的是尾部的省略号批量维，需按前两维施加对角约束。
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


def _broadcast_expand_shape(shape: tuple, target_shape: tuple) -> tuple:
    """
    按广播语义对齐源形状与目标形状，用于 expand。

    目标维为1而源维更大的维度（如省略号批量维被其他操作数广播放大）保留源尺寸，使 expand
    合法；此类维度随后由 _unbroadcast_grad 求和还原为目标尺寸（前向共享同一份数据，梯度
    应按广播份数累加）。其余维度取目标尺寸以正常广播。
    """
    return tuple(src if tgt == 1 else tgt for src, tgt in zip(shape, target_shape))

@api_registry.register("trace")
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


def attach_zero_grad_sources(self:TN, sources:tuple|list):
    """
    将多个张量作为来源张量依附到self张量上，不改变self的值，但self可以向这些张量传递0梯度。

    该函数用于将张量加入计算图，但不改变现有计算图的前向计算值和反向传播计算的梯度值。
    当调用backward时，self会向依附的来源张量传递全0梯度。

    参数:
        sources: 要依附的来源张量集合，可以是tuple、list或set。
                只有需要梯度(requires_grad=True)且不是self本身的张量才会被依附。

    返回:
        self: 返回修改后的self张量，便于链式调用。

    示例:
        >>> a = tensor([1.0, 2.0], requires_grad=True)
        >>> b = tensor([3.0, 4.0], requires_grad=True)
        >>> c = tensor([5.0, 6.0], requires_grad=True)
        >>> c.attach_zero_grad_sources([a, b])  # 将a,b加入c的计算图
        >>> # c的值不变，但backward时c会向a,b传递0梯度
    """
    def _zero_grad_backward(result:TN, i):
        return zeros_like(result.fromvars[i])

    # 过滤有效来源张量（需要梯度且不是self本身）
    valid_sources = [
        var for var in sources
        if var is not self and isinstance(var, TN) and var.requires_grad
    ]
    if not valid_sources:
        return self

    # 直接扩展fromvars和gradfuncs，避免创建中间列表
    self.fromvars = self.fromvars + tuple(valid_sources)
    self.gradfuncs = self.gradfuncs + tuple(_zero_grad_backward for _ in valid_sources)
    self.requires_grad = True

    return self

def share_grad_map(tensors:tuple|list):
    """
    将一组张量相互连接到一个共享的计算图中，但不改变现有计算图的前向计算值和反向传播计算的梯度值。

    核心机制：
    1. 对每个需要梯度的张量进行克隆，克隆张量通过clone操作依赖原始张量（梯度透传）
    2. 将其他张量（排除自身）作为零梯度来源依附到克隆张量上
    3. 这样每个张量既保持与原始张量的梯度关系，又与其他张量形成0梯度连接

    设计优化：
    - 使用 `[s for s in tensors if s is not t]` 排除自身，避免双路径依赖
    - 单路径设计：克隆张量只通过clone依赖原始张量，不通过attach重复依赖

    参数:
        tensors: 输入张量序列，必须是tuple或list类型。
                序列中的每个张量都会被处理：
                - 需要梯度的张量会被克隆，并将其他张量（排除自身）作为零梯度来源依附到克隆张量
                - 不需要梯度的张量或非TN对象保持原样

    返回:
        与输入类型相同的张量序列（tuple或list）。
        返回的张量与输入张量值相同，但计算图已相互连接。
        注意：对于需要梯度的张量，返回的是克隆张量而非原始张量。

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
        >>> a_new, b_new, c_new = share_grad_map((a, b, c))
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

    target_list = []
    for t in tensors:
        if isinstance(t,TN) and t.requires_grad:
            # 克隆张量并通过clone依赖原始张量（梯度透传）
            # 将其他张量（排除自身）作为零梯度来源依附，避免双路径依赖
            target_list.append(
                attach_zero_grad_sources(t.clone(),[s for s in tensors if s is not t])
            )
        else:
            target_list.append(t)

    return type(tensors)(target_list)
