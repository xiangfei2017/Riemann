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
Riemann Library Linear Algebra Module: Matrix Operations and Decompositions

This module provides comprehensive linear algebra operations for the Riemann
machine learning framework. It implements matrix operations, decompositions,
and norm calculations with full gradient support and automatic differentiation
compatibility, following the interface conventions of PyTorch's torch.linalg.

Main features:
    - Matrix operations: Matrix multiplication, transpose, inverse, and
      other fundamental linear algebra operations
    - Matrix decompositions: LU, QR, SVD, eigenvalue decomposition, and
      other factorization methods with gradient support
    - Norm calculations: Vector norms, matrix norms, and multi-dimensional
      norms with automatic differentiation
    - Linear system solving: Direct and iterative methods for solving
      systems of linear equations
    - Tensor linear algebra: Extended linear algebra operations for
      multi-dimensional tensors
    - Numerical stability: Robust implementations with proper handling of
      edge cases and numerical precision issues

Using this module, you can perform advanced linear algebra computations within
the Riemann automatic differentiation framework, enabling implementation of
sophisticated machine learning algorithms that rely on matrix operations
and decompositions.
"""

import numpy as np
import scipy  # type: ignore
from .tensordef import *
import builtins

def matmul(A, B, *, out=None):
    """计算两个张量的矩阵乘法
    
    根据输入张量的维度自动选择适当的矩阵乘法方式：
    - 1D × 1D: 向量内积（标量）
    - 2D × 1D: 矩阵乘向量
    - 1D × 2D: 向量乘矩阵
    - 2D × 2D: 矩阵乘法
    - ≥3D: 批量矩阵乘法（支持广播）
    
    参数:
        A: 第一个张量
        type A: riemann.TN
        B: 第二个张量
        type B: riemann.TN
        out: 输出张量（可选）
        type out: riemann.TN, optional
        
    返回:
        矩阵乘法结果
        rtype: riemann.TN
        
    示例:
        >>> A = tensor([[1, 2], [3, 4]])
        >>> B = tensor([[5, 6], [7, 8]])
        >>> C = matmul(A, B)  # 返回 [[19, 22], [43, 50]]
        
        >>> a = tensor([1, 2, 3])
        >>> b = tensor([4, 5, 6])
        >>> c = matmul(a, b)  # 返回标量 32
        
        >>> A = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 形状 (2, 2, 2)
        >>> B = tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])  # 形状 (2, 2, 2)
        >>> C = matmul(A, B)  # 返回形状 (2, 2, 2) 的张量
    """
    # 输入验证
    if not isinstance(A, TN):
        raise TypeError(f"matmul: Expected A to be TN type, got {type(A)}")
    if not isinstance(B, TN):
        raise TypeError(f"matmul: Expected B to be TN type, got {type(B)}")
    
    # 执行矩阵乘法
    result = A @ B
    
    # 处理out参数
    if out is not None:
        if not isinstance(out, TN):
            raise TypeError("out must be TN type")
        if out.shape != result.shape:
            raise ValueError(f"out has wrong shape: expected {result.shape}, got {out.shape}")
        
        # 将计算结果复制到out中
        return out.copy_(result)
    
    return result

def norm(A, ord:int|float|str|None=None, dim=None, keepdim=False, out=None, dtype=None):
    """计算张量的向量范数、矩阵范数或多轴范数
    
    支持向量范数、矩阵范数和多轴范数
    
    参数:
        A: 输入张量
        ord: 范数类型，支持None, 0, 1, 2, -1, -2, inf, -inf, 'fro', 'nuc'
        dim: 计算范数的维度，可以是整数、元组或None
        keepdim: 是否保持维度
        out: 输出张量
        dtype: 输出数据类型
    
    返回:
        范数计算结果
    """
    # 输入验证
    if not isinstance(A, TN):
        raise TypeError(f"Input must be TN type, got {type(A)}")
    
    # 类型检查：只有浮点型或复数型张量才能用于范数计算，与PyTorch行为一致
    if A.dtype.kind not in ['f', 'c']:
        raise RuntimeError(f"linalg.norm: Expected a floating point or complex tensor as input. Got {A.dtype.name}")
    
    dev = A.device

    # 处理dtype参数
    if dtype is not None:
        A = A.type(dtype)
    
    # 处理dim参数
    if dim == ():
        dim = None
    
    # 预先验证一些无效的p值
    valid_ps = [None, 0, 1, 2, -1, -2, float('inf'), float('-inf'), 'fro', 'nuc']
    if ord not in valid_ps and not isinstance(ord, (int, float)):
        raise ValueError(f"Unsupported norm type: ord={ord}")
    
    # 扁平化范数情况
    if dim is None:
        if ord is None:
            # L2范数，支持复数
            abs_A = abs(A)
            squared_A = abs_A * abs_A
            ret = sqrt(sum(squared_A))
        else:
            # dim=None且p!=None的情况
            if A.ndim > 2:
                raise ValueError("When dim=None and ord!=None, input must be 1D or 2D tensor")
            
            if A.ndim == 1:
                # 向量范数
                ret = _vector_norm(A, ord)
            else:  # A.ndim == 2
                # 矩阵范数
                ret = _matrix_norm(A, ord)                
    else:
        # 向量范数（dim是整数）
        if isinstance(dim, int):
            if ord == 'fro' or ord == 'nuc':
                raise ValueError(f"Norm type '{ord}' is not applicable to vectors")
            if ord == 0:
                # L0范数: 非零元素的个数
                epsilon = tensor(np.finfo(float).eps, device=dev, requires_grad=False)
                abs_A = abs(A)
                mask = abs_A > epsilon
                ones = tensor(1.0, device=dev, requires_grad=False)
                zeros = tensor(0.0, device=dev, requires_grad=False)
                mask_float = where(mask, ones, zeros)
                ret = sum(mask_float, dim=dim, keepdim=keepdim)
            else:
                ret = _compute_axis_norm(A, ord, dim, keepdim)
        # 矩阵范数或多轴范数（dim是元组）
        elif isinstance(dim, tuple):
            if len(dim) != 2:
                raise ValueError(f"dim must be a tuple of length 2, got {len(dim)}")
            
            # 对于两个维度的情况，根据p值决定是矩阵范数还是多轴范数
            # 关键修复：在高维张量上，即使是矩阵范数类型，也应该根据张量维度决定调用哪个函数
            if ord in ['fro', 'nuc']:
                # 特殊矩阵范数类型始终调用矩阵范数函数
                ret = _compute_matrix_norm(A, ord, dim, keepdim)
            elif A.ndim == 2:
                # 2D张量始终调用矩阵范数函数
                ret = _compute_matrix_norm(A, ord, dim, keepdim)
            else:
                # 高维张量上，p为1, -1, inf, -inf时也应该调用矩阵范数函数
                # 这是与PyTorch行为一致的关键
                if ord in [1, -1, float('inf'), float('-inf')]:
                    ret = _compute_matrix_norm(A, ord, dim, keepdim)
                else:
                    # 其他情况调用多轴范数函数
                    ret = _compute_multi_axis_norm(A, ord, dim, keepdim)
        else:
            raise TypeError(f"dim must be an integer, tuple, or None, got {type(dim).__name__}")
    
    # 处理keepdim参数 - 确保维度正确
    if dim is not None and not keepdim:
        if isinstance(dim, int):
            if ret.ndim > 0 and dim < ret.ndim and ret.shape[dim] == 1:
                ret = ret.squeeze(dim)
        elif isinstance(dim, tuple):
            for d in sorted(dim, reverse=True):
                if ret.ndim > 0 and d < ret.ndim and ret.shape[d] == 1:
                    ret = ret.squeeze(d)
    
    # 在返回前强制转换数据类型
    if dtype is not None:
        ret = ret.type(dtype)
    
    # 处理out参数
    if out is not None:
        if not isinstance(out, TN):
            raise TypeError("out must be TN type")
        if out.shape != ret.shape:
            raise ValueError(f"out has wrong shape: expected {ret.shape}, got {out.shape}")
        
        # 将计算结果复制到out中
        return out.copy_(ret)
    
    return ret


def _vector_norm(A, p):
    """计算1D向量的范数（内部函数）"""
    arrlib = A._get_array_lib()
    dev = A.device

    # 检查是否是全零向量
    is_all_zero = all(abs(A) < tensor(arrlib.finfo(float).eps, device=dev, requires_grad=False))
    if is_all_zero:
        # 对于全零向量，负阶范数返回0，与PyTorch行为一致
        if p in [-1, -2]:
            return tensor(0.0, device=dev, requires_grad=False)
        # 其他范数正常处理
    
    if p == 0:
        # L0范数: 非零元素的个数        
        epsilon = tensor(arrlib.finfo(float).eps, device=dev, requires_grad=False)
        abs_A = abs(A)
        mask = abs_A > epsilon
        ones = tensor(1.0, device=dev, requires_grad=False)
        zeros = tensor(0.0, device=dev, requires_grad=False)
        mask_float = where(mask, ones, zeros)
        return sum(mask_float)
    elif p == 1:
        return sum(abs(A))
    elif p == 2 or p is None:
        abs_A = abs(A)
        squared_A = abs_A * abs_A
        return sqrt(sum(squared_A))
    elif p == float('inf'):
        abs_A = abs(A)
        max_result = max(abs_A)
        return max_result.values if hasattr(max_result, 'values') else max_result
    elif p == float('-inf'):
        abs_A = abs(A)
        min_result = min(abs_A)
        return min_result.values if hasattr(min_result, 'values') else min_result
    elif isinstance(p, (int, float)) and p > 0:
        abs_A = abs(A)
        float_p = get_default_dtype().type(p)
        value = abs_A ** float_p
        return sum(value) ** (1.0/float_p)
    elif p == -1:
        abs_A = abs(A)
        # 避免除以零
        epsilon = tensor(arrlib.finfo(float).eps, device=dev, requires_grad=False)
        safe_abs_A = where(abs_A > epsilon, abs_A, tensor(float('inf'), device=dev, requires_grad=False))
        # L-1范数: (sum(|x_i|^(-1)))^(-1)
        value = 1.0 / safe_abs_A  # p=-1, 相当于1/|x_i|
        return 1.0 / sum(value)   # 1/p=-1
    elif p == -2:
        abs_A = abs(A)
        # 避免除以零
        epsilon = tensor(arrlib.finfo(float).eps, device=dev, requires_grad=False)
        safe_abs_A = where(abs_A > epsilon, abs_A, tensor(float('inf'), device=dev, requires_grad=False))
        # L-2范数: (sum(|x_i|^(-2)))^(-1/2)
        value = safe_abs_A ** (-2.0)  # p=-2, 相当于1/(|x_i|^2)
        return sum(value) ** (-0.5)  # 1/p=-1/2
    else:
        raise ValueError(f"vector norm does not support p value: {p}")


def _matrix_norm(A, p):
    """计算2D矩阵的范数（内部函数）"""
    arrlib = A._get_array_lib()
    dev = A.device
    
    # 计算矩阵范数
    if p == 'fro' or (p is None):
        abs_A = abs(A)
        squared_A = abs_A * abs_A
        return sqrt(sum(squared_A))
    elif p == 'nuc':
        _, S, _ = svd(A, full_matrices=False)
        return sum(S)
    elif p == float('inf'):
        abs_A = abs(A)
        sum_abs = sum(abs_A, dim=1)
        max_result = max(sum_abs)
        return max_result.values if hasattr(max_result, 'values') else max_result
    elif p == float('-inf'):
        abs_A = abs(A)
        sum_abs = sum(abs_A, dim=1)
        min_result = min(sum_abs)
        return min_result.values if hasattr(min_result, 'values') else min_result
    elif p == 1:
        abs_A = abs(A)
        sum_abs = sum(abs_A, dim=0)
        max_result = max(sum_abs)
        return max_result.values if hasattr(max_result, 'values') else max_result
    elif p == -1:
        abs_A = abs(A)
        sum_abs = sum(abs_A, dim=0)
        min_result = min(sum_abs)
        return min_result.values if hasattr(min_result, 'values') else min_result
    elif p == 2:
        _, S, _ = svd(A, full_matrices=False)
        max_result = max(S)
        return max_result.values if hasattr(max_result, 'values') else max_result
    elif p == -2:
        _, S, _ = svd(A, full_matrices=False)
        epsilon = tensor(arrlib.finfo(float).eps * S.max().data, device=dev, requires_grad=False)
        S_non_zero = where(S > epsilon, S, tensor(float('inf'), device=dev, requires_grad=False))
        min_result = min(S_non_zero)
        ret = min_result.values if hasattr(min_result, 'values') else min_result
        if (ret.data == float('inf')).any():
            ret = tensor(arrlib.finfo(float).eps, device=dev, requires_grad=A.requires_grad)
        return ret
    else:
        raise ValueError(f"Matrix norm does not support p value: {p}")


def _compute_axis_norm(A, p, dim, keepdim):
    """计算指定单轴的范数（内部函数）"""
    arrlib = A._get_array_lib()
    dev = A.device
    
    # 对于负阶范数，检查是否在指定维度上全为零
    if p in [-1, -2]:
        # 计算指定维度上的最大绝对值
        max_abs = max(abs(A), dim=dim, keepdim=True)
        max_abs_value = max_abs.values if hasattr(max_abs, 'values') else max_abs
        # 如果所有元素都接近零，返回0
        epsilon = tensor(arrlib.finfo(float).eps, device=dev, requires_grad=False)
        all_zero_mask = max_abs_value < epsilon
        # 创建与结果形状匹配的零张量
        result_shape = list(A.shape)
        result_shape[dim] = 1
        if not keepdim:
            result_shape.pop(dim)
        zero_result = zeros(result_shape, dtype=A.dtype, device=dev, requires_grad=A.requires_grad)
        # 如果是全零，返回0；否则执行正常计算
        if all_zero_mask.all():
            return zero_result
    
    if p == 1:
        return sum(abs(A), dim=dim, keepdim=keepdim)
    elif p == 2 or p is None:
        abs_A = abs(A)
        squared_A = abs_A * abs_A
        sum_squared = sum(squared_A, dim=dim, keepdim=keepdim)
        return sqrt(sum_squared)
    elif p == float('inf'):
        abs_A = abs(A)
        max_result = max(abs_A, dim=dim, keepdim=keepdim)
        return max_result.values if hasattr(max_result, 'values') else max_result
    elif p == float('-inf'):
        abs_A = abs(A)
        min_result = min(abs_A, dim=dim, keepdim=keepdim)
        return min_result.values if hasattr(min_result, 'values') else min_result
    elif isinstance(p, (int, float)) and p > 0:
        abs_A = abs(A)
        float_p = get_default_dtype().type(p)
        value = abs_A ** float_p
        return sum(value, dim=dim, keepdim=keepdim) ** (1.0/float_p)
    elif p == -1:
        abs_A = abs(A)
        # 避免除以零
        epsilon = tensor(arrlib.finfo(float).eps, device=dev, requires_grad=False)
        safe_abs_A = where(abs_A > epsilon, abs_A, tensor(float('inf'), device=dev, requires_grad=False))
        # L-1范数: (sum(|x_i|^(-1)))^(-1)
        value = -1.0 / safe_abs_A  # p=-1, 相当于1/|x_i|
        return -1.0 / sum(value, dim=dim, keepdim=keepdim)  # 1/p=-1
    elif p == -2:
        abs_A = abs(A)
        # 避免除以零
        epsilon = tensor(arrlib.finfo(float).eps, device=dev, requires_grad=False)
        safe_abs_A = where(abs_A > epsilon, abs_A, tensor(float('inf'), device=dev, requires_grad=False))
        # L-2范数: (sum(|x_i|^(-2)))^(-1/2)
        value = safe_abs_A ** (-2.0)  # p=-2, 相当于1/(|x_i|^2)
        return sum(value, dim=dim, keepdim=keepdim) ** (-0.5)  # 1/p=-1/2
    else:
        raise ValueError(f"Single-axis norm does not support p value: {p}")


# 修复_compute_matrix_norm函数，确保正确处理高维张量
def _compute_matrix_norm(A, p, dim, keepdim):
    """计算指定维度的矩阵范数（内部函数）"""
    arrlib = A._get_array_lib()
    dev = A.device
    
    # 验证维度索引是否有效
    for d in dim:
        if d < 0:
            d = A.ndim + d
        if d < 0 or d >= A.ndim:
                raise ValueError(f"Dimension index {d} out of range for tensor with {A.ndim} dimensions")
    
    if p == 'fro' or p is None:
        # Frobenius范数: 平方和的平方根
        # 直接对指定维度求和，避免多次嵌套sum
        abs_A = abs(A)
        squared_A = abs_A * abs_A
        sum_squared = sum(squared_A, dim=dim, keepdim=keepdim)
        return sqrt(sum_squared)
    elif p == 'nuc':
        # 核范数: 奇异值之和
        if A.ndim > 2:
            # 对于高维张量，我们需要对每个2D子矩阵应用SVD
            raise ValueError("Nuclear norm only supports 2D tensors")
        else:
            _, S, _ = svd(A, full_matrices=False)
            result = sum(S)
            if keepdim:
                return result.unsqueeze(0).unsqueeze(0)
            return result
    elif p == float('inf'):
        # 矩阵无穷范数: max(sum(abs(x), dim=dim[1]))
        abs_A = abs(A)
        # 确保在sum后保持维度，以便正确应用max
        sum_abs = sum(abs_A, dim=dim[1], keepdim=True)
        max_result = max(sum_abs, dim=dim[0], keepdim=keepdim)
        return max_result.values if hasattr(max_result, 'values') else max_result
    elif p == float('-inf'):
        # 矩阵负无穷范数: min(sum(abs(x), dim=dim[1]))
        abs_A = abs(A)
        sum_abs = sum(abs_A, dim=dim[1], keepdim=True)
        min_result = min(sum_abs, dim=dim[0], keepdim=keepdim)
        return min_result.values if hasattr(min_result, 'values') else min_result
    elif p == 1:
        # 矩阵L1范数: max(sum(abs(x), dim=dim[0]))
        abs_A = abs(A)
        sum_abs = sum(abs_A, dim=dim[0], keepdim=True)
        max_result = max(sum_abs, dim=dim[1], keepdim=keepdim)
        return max_result.values if hasattr(max_result, 'values') else max_result
    elif p == -1:
        # 矩阵-1范数: min(sum(abs(x), dim=dim[0]))
        abs_A = abs(A)
        sum_abs = sum(abs_A, dim=dim[0], keepdim=True)
        min_result = min(sum_abs, dim=dim[1], keepdim=keepdim)
        return min_result.values if hasattr(min_result, 'values') else min_result
    elif p == 2 and A.ndim == 2:
        # 矩阵2范数(谱范数): 最大奇异值
        _, S, _ = svd(A, full_matrices=False)
        max_result = max(S)
        ret = max_result.values if hasattr(max_result, 'values') else max_result
        if keepdim:
            ret = ret.unsqueeze(0).unsqueeze(0)
        return ret
    elif p == -2 and A.ndim == 2:
        # 矩阵-2范数: 最小奇异值
        _, S, _ = svd(A, full_matrices=False)
        epsilon = tensor(arrlib.finfo(float).eps * S.max().data, device=dev, requires_grad=False)
        S_non_zero = where(S > epsilon, S, tensor(float('inf'), device=dev, requires_grad=False))
        min_result = min(S_non_zero)
        ret = min_result.values if hasattr(min_result, 'values') else min_result
        if (ret.data == float('inf')).any():
            ret = tensor(arrlib.finfo(float).eps, device=dev, requires_grad=A.requires_grad)
        if keepdim:
            ret = ret.unsqueeze(0).unsqueeze(0)
        return ret
    else:
        raise ValueError(f"Matrix norm does not support p value: {p}")

def _compute_max_singular_value_norm(A: TN, dim: tuple, keepdim: bool = False) -> TN:
    """
    计算多维数组在指定两个轴上的谱范数（最大奇异值）
    
    参数:
        A: 输入张量，形状为任意维度
        dim: 元组，指定两个轴，沿着这些轴计算谱范数
        keepdim: 布尔值，是否在结果中保持计算维度
        
    返回:
        TN: 计算得到的谱范数值，形状为A.shape但移除了指定的两个维度（除非keepdim=True）
    """
    # 输入验证 
    if not isinstance(A, TN): 
        raise TypeError(f"Input must be TN type, got {type(A)}") 
    
    arrlib = A._get_array_lib()
    dev = A.device
    
    if not isinstance(dim, tuple) or len(dim) != 2: 
        raise ValueError(f"dim must be a tuple of length 2, got {dim}") 
    
    # 验证维度索引是否有效并转换为正数索引
    pos_dim = []
    for d in dim:
        if d < 0:
            d = A.ndim + d
        if d < 0 or d >= A.ndim:
            raise ValueError(f"Dimension index {d} out of range for tensor with {A.ndim} dimensions")
        pos_dim.append(d)
    
    # 确保维度索引不重复
    if pos_dim[0] == pos_dim[1]:
        raise ValueError(f"dim cannot contain duplicate dimensions, got {dim}")
    
    # 获取所有轴的索引
    all_dims = list(range(A.ndim))
    
    # 确定非计算轴（即除了dim之外的轴）
    non_compute_axes = [ax for ax in all_dims if ax not in pos_dim]
    
    # 重塑数组，将计算轴放在最后两个维度
    new_per = tuple(non_compute_axes + pos_dim)
    
    # 验证维度列表长度是否正确
    if len(new_per) != A.ndim:
        raise ValueError(f"Internal error: Invalid dimension permutation. Expected {A.ndim} dimensions, got {len(new_per)}")
    
    # 使用permute重排维度
    A_permuted = A.permute(new_per)
    
    # 检查是否全为零矩阵
    is_all_zero = (abs(A_permuted) < tensor(arrlib.finfo(float).eps, device=dev, requires_grad=False)).all()
    if is_all_zero:
        # 确定结果形状
        output_shape = list(A.shape)
        for d in pos_dim:
            output_shape[d] = 1
        # 创建与结果形状匹配的零张量
        zero_result = zeros(tuple(output_shape), dtype=A.dtype, device=dev, requires_grad=A.requires_grad)
        if not keepdim:
            # 如果不需要保持维度，移除计算的两个维度
            for d in sorted(pos_dim, reverse=True):
                zero_result = zero_result.squeeze(d)
        return zero_result
    
    # 重排后的张量形状：(batch_dims..., m, n)，其中m和n是指定轴的大小
    # 使用svd获取奇异值
    _, S, _ = svd(A_permuted, full_matrices=False)
    
    # 正确的max_dim设置：奇异值维度是最后一维
    max_dim = S.ndim - 1  # 只在奇异值维度上取最大值
    
    # 计算最大奇异值，正确传递keepdim参数
    max_result = max(S, dim=max_dim, keepdim=True)  # 这里先保持维度
    spectral_norm = max_result.values if hasattr(max_result, 'values') else max_result
    
    # 处理keepdim参数
    if keepdim:
        # 我们需要将结果的形状调整回原始张量的形状，但在计算的两个维度上大小为1
        # 1. 首先确定要恢复的形状
        output_shape = list(A.shape)
        # 将指定的两个维度大小设为1
        for d in pos_dim:
            output_shape[d] = 1
        # 2. 重塑结果张量
        spectral_norm = spectral_norm.reshape(tuple(output_shape))
    else:
        # 如果不需要保持维度，移除计算的两个维度
        # 由于我们已经在奇异值维度上取了最大值，现在需要移除原始的两个计算维度
        # 先创建一个不包含计算维度的新形状
        output_shape = []
        for i in range(A.ndim):
            if i not in pos_dim:
                output_shape.append(A.shape[i])
        # 重塑结果张量
        spectral_norm = spectral_norm.reshape(tuple(output_shape))
    
    return spectral_norm


# 计算最小奇异值范数
def _compute_min_singular_value_norm(A: TN, dim: tuple, keepdim: bool = False) -> TN:
    """
    计算多维数组在指定两个轴上的最小奇异值（-2范数）
    
    参数:
        A: 输入张量，形状为任意维度
        dim: 元组，指定两个轴，沿着这些轴计算最小奇异值
        keepdim: 布尔值，是否在结果中保持计算维度
        
    返回:
        TN: 计算得到的最小奇异值，形状为A.shape但移除了指定的两个维度（除非keepdim=True）
    """
    # 输入验证 
    if not isinstance(A, TN): 
        raise TypeError(f"Input must be TN type, got {type(A)}") 
    
    if not isinstance(dim, tuple) or len(dim) != 2: 
        raise ValueError(f"dim must be a tuple of length 2, got {dim}") 
    
    arrlib = A._get_array_lib()
    dev = A.device
    
    # 验证维度索引是否有效并转换为正数索引
    pos_dim = []
    for d in dim:
        if d < 0:
            d = A.ndim + d
        if d < 0 or d >= A.ndim:
            raise ValueError(f"Dimension index {d} out of range for tensor with {A.ndim} dimensions")
        pos_dim.append(d)
    
    # 确保维度索引不重复
    if pos_dim[0] == pos_dim[1]:
        raise ValueError(f"dim cannot contain duplicate dimensions, got {dim}")
    
    # 获取所有轴的索引
    all_dims = list(range(A.ndim))
    
    # 确定非计算轴（即除了dim之外的轴）
    non_compute_axes = [ax for ax in all_dims if ax not in pos_dim]
    
    # 重塑数组，将计算轴放在最后两个维度
    new_per = tuple(non_compute_axes + pos_dim)
    
    # 使用permute重排维度
    A_permuted = A.permute(new_per)
    
    # 使用svd获取奇异值
    _, S, _ = svd(A_permuted, full_matrices=False)
    
    # 设置min_dim
    min_dim = S.ndim - 1
    
    # 计算最小非零奇异值
    epsilon = tensor(arrlib.finfo(float).eps * S.max().data, device=dev, requires_grad=False)
    S_non_zero = where(S > epsilon, S, tensor(float('inf'), device=dev, requires_grad=False))
    min_result = min(S_non_zero, dim=min_dim, keepdim=True)
    min_singular = min_result.values if hasattr(min_result, 'values') else min_result
    
    # 处理无穷大情况（当所有奇异值都接近零时）
    if (min_singular.data == float('inf')).any():
        # 对于全零矩阵，返回0而不是eps，与PyTorch行为一致
        min_singular = tensor(0.0, device=dev, requires_grad=A.requires_grad)
    
    # 处理keepdim参数
    if keepdim:
        # 调整结果形状，保持原始张量的形状，但计算的两个维度大小为1
        output_shape = list(A.shape)
        for d in pos_dim:
            output_shape[d] = 1
        min_singular = min_singular.reshape(tuple(output_shape))
    else:
        # 移除计算的两个维度
        output_shape = []
        for i in range(A.ndim):
            if i not in pos_dim:
                output_shape.append(A.shape[i])
        min_singular = min_singular.reshape(tuple(output_shape))
    
    return min_singular

# 计算多维数组在指定两个轴上的范数
def _compute_multi_axis_norm(A: TN, p: int | float | str | None, dim: tuple, keepdim: bool = False) -> TN:
    """
    计算多维数组在指定两个轴上的范数（内部函数）
    参数:
        A: 输入张量，形状为任意维度
        p: 范数的阶数，或None表示Frobenius范数
        dim: 元组，指定两个轴，沿着这些轴计算范数
        keepdim: 布尔值，是否在结果中保持计算维度
        
    返回:
        TN: 计算得到的范数，形状为A.shape但移除了指定的两个维度（除非keepdim=True）
    """
    
    arrlib = A._get_array_lib()
    dev = A.device
    
    # 对于负阶范数，检查是否在指定维度上全为零
    if p == -1:
        # 计算指定维度上的最大绝对值
        abs_A = abs(A)
        # 先沿第一个维度取最大
        max_abs_dim0 = max(abs_A, dim=dim[0], keepdim=True)
        max_abs_dim0 = max_abs_dim0.values if hasattr(max_abs_dim0, 'values') else max_abs_dim0
        # 再沿第二个维度取最大
        max_abs_result = max(max_abs_dim0, dim=dim[1], keepdim=True)
        max_abs_value = max_abs_result.values if hasattr(max_abs_result, 'values') else max_abs_result
        
        # 如果所有元素都接近零，返回0
        epsilon = tensor(arrlib.finfo(float).eps, device=dev, requires_grad=False)
        all_zero_mask = max_abs_value < epsilon
        
        # 创建与结果形状匹配的零张量
        if all_zero_mask.all():
            # 确定结果形状
            result_shape = list(A.shape)
            for d in sorted(dim, reverse=True):
                if keepdim:
                    result_shape[d] = 1
                else:
                    result_shape.pop(d)
            return zeros(tuple(result_shape), device=dev, requires_grad=A.requires_grad)
    
    if p == 0:
        raise ValueError("Multi-axis norm does not support p=0")
    elif p == 1:
        # 多轴L1范数：对指定维度内的元素取绝对值后求和
        abs_A = abs(A)
        return sum(abs_A, dim=dim, keepdim=keepdim)
    elif p == 2:
        return _compute_max_singular_value_norm(A, dim=dim, keepdim=keepdim)
    elif p == -2:  # 修复：为-2范数添加特殊处理
        # 对于-2范数，在_compute_min_singular_value_norm函数中已经处理了全零情况
        return _compute_min_singular_value_norm(A, dim=dim, keepdim=keepdim)
    elif p is None:
        # 使用Frobenius范数计算矩阵范数
        abs_A = abs(A)
        squared_A = abs_A * abs_A
        sum_squared = sum(squared_A, dim=dim, keepdim=keepdim)  # 沿指定维度求和
        result = sqrt(sum_squared)  # 开平方根
        return result        
    elif isinstance(p, (int, float)) and p > 0:
        # 通用p-范数
        abs_A = abs(A)
        float_p = get_default_dtype().type(p)
        value = abs_A ** float_p
        return sum(value, dim=dim, keepdim=keepdim) ** (1.0/float_p)
    elif p == float('inf'):
        # 多轴无穷范数：最大值
        abs_A = abs(A)
        # 对第一个维度应用max
        max_result = max(abs_A, dim=dim[0], keepdim=True)
        # 对第二个维度应用max
        max_result = max(max_result.values if hasattr(max_result, 'values') else max_result, 
                         dim=dim[1], keepdim=keepdim)
        return max_result.values if hasattr(max_result, 'values') else max_result
    elif p == float('-inf'):
        # 多轴负无穷范数：最小值 - 分别处理每个维度
        abs_A = abs(A)
        # 对第一个维度应用min
        min_result = min(abs_A, dim=dim[0], keepdim=True)
        # 对第二个维度应用min
        min_result = min(min_result.values if hasattr(min_result, 'values') else min_result, 
                         dim=dim[1], keepdim=keepdim)
        return min_result.values if hasattr(min_result, 'values') else min_result
    elif p == -1:
        # 多轴-1范数：(sum(|x_ij|^(-1)))^(-1)
        abs_A = abs(A)
        # 避免除以零
        epsilon = tensor(arrlib.finfo(float).eps, device=dev, requires_grad=False)
        safe_abs_A = where(abs_A > epsilon, abs_A, tensor(float('inf'), device=dev, requires_grad=False))
        # 计算每个元素的-1次方
        value = 1. / safe_abs_A  # p=-1
        # 对两个维度求和
        sum_value = sum(value, dim=dim, keepdim=keepdim)
        # 取-1次方根
        return 1. / sum_value  # 1/p=-1
    else:
        raise ValueError(f"Multi-axis norm does not support p value: {p}")

def vector_norm(x, p=None, dim=None, keepdim=False, out=None, dtype=None):
    """计算向量的向量范数
    
    专门用于计算向量范数的函数，接口与PyTorch的torch.linalg.vector_norm一致
    
    参数:
        x: 输入张量
        p: 范数类型，支持None(默认2范数), 0, 1, 2, -1, -2, inf, -inf
        dim: 计算范数的维度，可以是整数、元组或None
        keepdim: 是否保持维度
        out: 输出张量
        dtype: 输出数据类型
    
    返回:
        范数计算结果
    """
    # 类型检查：只有浮点型或复数型张量才能用于范数计算，与PyTorch行为一致
    if not isinstance(x, TN) or x.dtype.kind not in ['f', 'c']:
        raise RuntimeError(f"linalg.vector_norm: Expected a floating point or complex tensor as input. Got {x.dtype.name if isinstance(x, TN) else type(x).__name__}")
    
    # 验证p参数，vector_norm不支持'fro'和'nuc'范数
    if p in ['fro', 'nuc']:
        raise ValueError(f"vector_norm does not support norm type: p={p}")
    
    # 直接调用通用的norm函数处理
    return norm(x, ord=p, dim=dim, keepdim=keepdim, out=out, dtype=dtype)


def matrix_norm(A, p=None, dim=None, keepdim=False, out=None, dtype=None):
    """计算矩阵的矩阵范数
    
    专门用于计算矩阵范数的函数，接口与PyTorch的torch.linalg.matrix_norm一致
    
    参数:
        A: 输入张量
        p: 范数类型，支持None(默认Frobenius范数), 1, 2, -1, -2, inf, -inf, 'fro', 'nuc'
        dim: 计算范数的维度，必须是长度为2的元组或None
        keepdim: 是否保持维度
        out: 输出张量
        dtype: 输出数据类型
    
    返回:
        范数计算结果
    """
    # 类型检查：只有浮点型或复数型张量才能用于范数计算，与PyTorch行为一致
    if not isinstance(A, TN) or A.dtype.kind not in ['f', 'c']:
        raise RuntimeError(f"linalg.matrix_norm: Expected a floating point or complex tensor as input. Got {A.dtype.name if isinstance(A, TN) else type(A).__name__}")
    
    # 验证p参数，matrix_norm不支持0范数
    if p == 0:
        raise ValueError(f"matrix_norm does not support norm type: p={p}")
    
    # 如果dim是整数，转换为元组
    if isinstance(dim, int):
        raise ValueError(f"matrix_norm's dim must be a tuple of length 2, got {type(dim).__name__}")
    elif dim is not None and (not isinstance(dim, tuple) or len(dim) != 2):
        raise ValueError(f"matrix_norm's dim must be a tuple of length 2, got {dim}")
    
    # 直接调用通用的norm函数处理
    return norm(A, ord=p, dim=dim, keepdim=keepdim, out=out, dtype=dtype)


# - slogdet ：计算方阵行列式的符号和对数绝对值
# - cond ：计算矩阵相对于矩阵范数的条件数
# - matrix_rank ：计算矩阵的数值秩

def cond(A: TN, p=None, *, out=None) -> TN:
    """计算矩阵相对于矩阵范数的条件数。

    条件数衡量了线性系统`AX = B`相对于矩阵范数的数值稳定性。

    支持float、double、cfloat和cdouble数据类型的输入。
    也支持批量矩阵，如果A是批量矩阵，则输出具有相同的批量维度。

    :attr:`p`定义了要计算的矩阵范数。支持以下范数：

    =========    ==================================
    :attr:`p`    矩阵范数
    =========    ==================================
    None         2-范数（最大奇异值）
    'fro'        Frobenius范数
    'nuc'        核范数
    inf          max(sum(abs(x), dim=1))
    -inf         min(sum(abs(x), dim=1))
    1            max(sum(abs(x), dim=0))
    -1           min(sum(abs(x), dim=0))
    2            最大奇异值
    -2           最小奇异值
    =========    ==================================

    其中inf指的是float('inf')，NumPy的inf对象或任何等效对象。

    对于:attr:p为('fro', 'nuc', inf, -inf, 1, -1)之一时，此函数使用
    `riemann.linalg.norm`和`riemann.linalg.inv`。
    因此，在这种情况下，矩阵（或批量中的每个矩阵）:attr:A必须是方阵且可逆。

    对于:attr:p为`(2, -2)`时，此函数可以根据奇异值计算条件数。
    在这些情况下，它使用:riemann.linalg.svdvals计算。对于这些范数，矩阵
    （或批量中的每个矩阵）:attr:A可以具有任何形状。

    参数:
        A (TN): 张量，形状为`(*, m, n)`，其中`*`是零个或多个批量维度
                对于:attr:`p`为`(2, -2)`，以及形状为`(*, n, n)`，其中每个矩阵
                对于:attr:`p`为`('fro', 'nuc', inf, -inf, 1, -1)`时必须可逆。
        p (int, inf, -inf, 'fro', 'nuc', optional):
            用于计算的矩阵范数类型（见上文）。默认值: `None`

    关键字参数:
        out (TN, optional): 输出张量。如果为`None`则忽略。默认值: `None`。

    返回:
        一个实值张量，即使:attr:`A`是复数。

    异常:
        RuntimeError:
            如果:attr:`p`是`('fro', 'nuc', inf, -inf, 1, -1)`之一
            且:attr:`A`矩阵或批量:attr:`A`中的任何矩阵不是方阵
            或不可逆。
        
    注意:
        对于奇异矩阵，当p=2时，条件数理论上是无穷大，函数会返回无穷大值。
        对于奇异矩阵，当p=-2时，条件数理论上是0，函数会返回0值。
    """
    # 输入验证
    if not isinstance(A, TN):
        raise TypeError(f"Input must be TN type, got {type(A)}")
    
    # 类型检查：只有浮点型或复数型张量才能用于条件数计算
    if A.dtype.kind not in ['f', 'c']:
        raise TypeError(f"Expected a floating point or complex tensor as input. Got {A.dtype.name}")
    
    arrlib = A._get_array_lib()
    dev = A.device
    
    # 处理默认参数
    if p is None:
        p = 2
    
    # 检查p是否是支持的值
    supported_p = [2, -2, 'fro', 'nuc', np.inf, -np.inf, 1, -1]
    if p not in supported_p:
        raise ValueError(f"p must be one of {supported_p}, got {p}")
    
    # 对于p=2或p=-2，使用奇异值计算，不需要矩阵是方阵
    if p == 2 or p == -2:
        # 获取奇异值
        S = svdvals(A)
        
        # 最大奇异值
        max_S = max(S, dim=-1, keepdim=False)
        max_S = max_S.values if hasattr(max_S, 'values') else max_S
        # 最小奇异值
        min_S = min(S, dim=-1, keepdim=False)
        min_S = min_S.values if hasattr(min_S, 'values') else min_S
        
        # 定义一个小的阈值，用于判断奇异值是否接近0（考虑数值计算误差）
        eps = arrlib.finfo(float).eps
        
        # 对于p=2，条件数是最大奇异值除以最小奇异值
        if p == 2:
            # 检查矩阵是否奇异（最小奇异值接近0）
            cond_value = max_S / min_S
        else:  # p == -2
            # 对于p=-2，条件数是最小奇异值除以最大奇异值
            cond_value = min_S / max_S
        
        # 确保结果是实数
        if A.dtype.kind == 'c':
            cond_value = cond_value.abs()
    else:
        # 对于其他p值，矩阵必须是方阵
        # 检查是否是方阵
        if A.ndim < 2:
            raise ValueError(f"Input matrix must be at least 2-dimensional, got dimension {A.ndim}")
        
        # 对于非2/-2的p值，要求矩阵是方阵
        if A.shape[-1] != A.shape[-2]:
            raise RuntimeError(f"Input must be a square matrix for p={p}, got shape {A.shape}")
        
        # 获取奇异值，用于检测矩阵是否奇异
        S = svdvals(A)
        max_S = max(S, dim=-1, keepdim=False)
        max_S = max_S.values if hasattr(max_S, 'values') else max_S
        min_S = min(S, dim=-1, keepdim=False)
        min_S = min_S.values if hasattr(min_S, 'values') else min_S
        
        # 定义一个小的阈值，用于判断奇异值是否接近0（考虑数值计算误差）
        eps = arrlib.finfo(float).eps
        
        # 检查矩阵是否奇异（最小奇异值接近0）
        is_singular = (min_S < eps).any()
        
        if is_singular:
            # 对于奇异矩阵，创建与max_S相同形状的无穷大张量
            # 使用full_like保持与输入张量相同的形状和类型
            cond_value = full_like(max_S, float('inf'), device=dev)
        else:
            try:
                # 对于可逆矩阵，根据条件数的数学定义：矩阵范数与逆矩阵范数的乘积
                # 计算A的范数
                norm_A = norm(A, ord=p)
                # 计算逆矩阵
                inv_A = inv(A)
                # 计算逆矩阵的范数
                norm_inv_A = norm(inv_A, ord=p)
                # 计算条件数
                cond_value = norm_A * norm_inv_A
            except RuntimeError as e:
                # 如果在计算过程中发现矩阵不可逆，返回无穷大
                if "not invertible" in str(e).lower():
                    cond_value = full_like(max_S, float('inf'), device=dev)
                else:
                    # 其他运行时错误重新抛出
                    raise
        
        # 确保结果是实数
        if A.dtype.kind == 'c':
            cond_value = cond_value.abs()
    
    # 处理out参数
    if out is not None:
        # 检查out参数是否是TN类型
        if not isinstance(out, TN):
            raise TypeError(f"out must be TN type, got {type(out)}")
        
        # 检查out参数的形状是否与计算结果一致
        if out.shape != cond_value.shape:
            raise RuntimeError(f"out has wrong shape: expected {cond_value.shape}, got {out.shape}")
        
        # 将计算结果复制到out中
        return out.copy_(cond_value)
            
    return cond_value

def svdvals(A: TN) -> TN:
    """返回矩阵的奇异值。
    
    该函数计算矩阵A的奇异值，与torch.linalg.svdvals保持一致。
    
    参数:
        A (TN): 输入张量，形状为(*, m, n)
    
    返回:
        TN: 奇异值，形状为(*, k)，其中k = min(m, n)
    """
    # 输入验证
    if not isinstance(A, TN):
        raise TypeError(f"Input must be TN type, got {type(A)}")
    
    # 使用svd函数获取奇异值
    _, S, _ = svd(A, full_matrices=False)
    
    return S

# - det ：计算方阵的行列式
def det(A:TN):
    """计算方阵的行列式。
    
    该函数计算方阵A的行列式，与torch.linalg.det保持一致。
    矩阵必须是方阵。
    
    参数:
        A (TN): 输入张量，必须是方阵
    
    返回:
        TN: 计算得到的行列式值
    
    异常:
        TypeError: 当输入不是TN类型时
        ValueError: 当输入不是方阵时
    """
    # 输入验证
    if not isinstance(A, TN):
        raise TypeError(f"Input must be TN type, got {type(A)}")
    
    # 检查是否是方阵
    if A.ndim < 2:
        raise ValueError(f"Input matrix must be at least 2-dimensional, got dimension {A.ndim}")
    
    if A.shape[-1] != A.shape[-2]:
        raise ValueError(f"Input must be a square matrix, got shape {A.shape}")
    
    arrlib = A._get_array_lib()
    dev = A.device
    
    # 计算行列式
    detvalue = arrlib.linalg.det(A.data)
    
    ret = tensor(detvalue, device=dev, requires_grad=(is_grad_enabled() and A.requires_grad))
    ret.is_leaf = not ret.requires_grad
    
    if ret.requires_grad:
        ret.fromvars = (A,)
        ret.gradfuncs = (_matdet_backward,)
    
    return ret

def _matdet_backward(result_tensor:TN, i:int)->TN:
    """计算行列式的反向传播梯度。"""
    
    A = result_tensor.fromvars[0]
    grad_value = result_tensor.grad_value

    # 将行列式值从标量恢复为2D矩阵，并取共轭（考虑复数情况）
    new_result = result_tensor.unsqueeze((-1,-2)).conj()
    # 计算A的逆矩阵的共轭转置
    inv_trans = pinv(A).mH

    grad = grad_value * new_result * inv_trans
    return grad

def inv(A:TN):
    """计算方阵的逆矩阵。
    
    该函数计算方阵A的逆矩阵，与torch.linalg.inv保持一致。
    矩阵必须是方阵且非奇异（行列式不为零）。
    
    参数:
        A (TN): 输入张量，必须是方阵
    
    返回:
        TN: 计算得到的逆矩阵
    
    异常:
        TypeError: 当输入不是TN类型时
        ValueError: 当输入不是方阵时
        RuntimeError: 当矩阵不可逆时
    """
    # 输入验证
    if not isinstance(A, TN):
        raise TypeError(f"Input must be TN type, got {type(A)}")
    
    # 检查是否是方阵
    if A.ndim < 2:
        raise ValueError(f"Input matrix must be at least 2-dimensional, got dimension {A.ndim}")
    
    if A.shape[-1] != A.shape[-2]:
        raise ValueError(f"Input must be a square matrix, got shape {A.shape}")
    
    arrlib = A._get_array_lib()
    dev = A.device
    
    # 计算逆矩阵并处理异常
    try:
        invarr = arrlib.linalg.inv(A.data)
        
    except arrlib.linalg.LinAlgError as e:
        raise RuntimeError(f"linalg.inv: Matrix is not invertible: {str(e)}")
    
    ret = tensor(invarr, device=dev, requires_grad=(is_grad_enabled() and A.requires_grad))
    ret.is_leaf = not ret.requires_grad
    
    if ret.requires_grad:
        ret.fromvars = (A,)
        ret.gradfuncs = (_matinv_backward,)
    
    return ret

def _matinv_backward(result_tensor:TN, i:int)->TN:
    """计算矩阵求逆的反向传播梯度。"""
    
    inv_trans = result_tensor.mH    
    grad = - inv_trans @ result_tensor.grad_value @ inv_trans
    
    return grad

def skew(A:TN):
    """计算方阵的反对称部分。
    
    反对称部分的定义为：(M - M^T) / 2，其中M^T是矩阵M的转置。
    
    参数:
        A (TN): 输入张量，必须是方阵
    
    返回:
        TN: 计算得到的反对称矩阵
    
    异常:
        TypeError: 当输入不是TN类型时
        ValueError: 当输入不是方阵时
    """
    # 输入验证
    if not isinstance(A, TN):
        raise TypeError(f"Input must be TN type, got {type(A)}")
    
    # 检查是否是方阵
    if A.ndim < 2:
        raise ValueError(f"Input matrix must be at least 2-dimensional, got dimension {A.ndim}")
    
    if A.shape[-1] != A.shape[-2]:
        raise ValueError(f"Input must be a square matrix, got shape {A.shape}")
    
    # 计算反对称部分：(A - A^T) / 2
    A_T = A.mT  # 获取转置
    result = (A - A_T) * 0.5
    
    return result


# 替换整个svd函数及其梯度计算函数
def svd(A, full_matrices=True, driver=None, out=None):
    """计算矩阵的奇异值分解(SVD)。
    
    该函数计算矩阵A的奇异值分解，与torch.linalg.svd保持一致。
    
    参数:
        A (TN): 输入张量，形状为(*, m, n)，其中*表示任意数量的批处理维度
        full_matrices (bool, optional): 控制是否计算完整或精简的SVD，默认为True
        driver (str, optional): CUDA设备上使用的cuSOLVER方法名称，默认为None
        out (tuple, optional): 输出元组的三个张量，默认为None
        
    返回:
        一个包含(U, S, Vh)的元组，其中:
        - U: 左奇异向量矩阵，形状为(*, m, m)（如果full_matrices=True）或(*, m, min(m,n))（如果full_matrices=False）
        - S: 奇异值向量，形状为(*, min(m,n))
        - Vh: 右奇异向量的共轭转置矩阵，形状为(*, n, n)（如果full_matrices=True）或(*, min(m,n), n)（如果full_matrices=False）
    """
    # 输入验证 
    if not isinstance(A, TN): 
        raise TypeError(f"Input must be TN type, got {type(A)}") 
    
    arrlib = A._get_array_lib()
    dev = A.device
    
    # 验证输入维度至少为2 
    if A.ndim < 2: 
        raise ValueError(f"Input matrix must be at least 2-dimensional, got dimension {A.ndim}") 
    
    # 暂时不支持driver参数，因为我们使用numpy实现 
    if driver is not None: 
        print("Warning: driver parameter is ignored in the current implementation because numpy is used as the backend") 
    
    # 处理out参数 
    if out is not None: 
        if not isinstance(out, tuple) or len(out) != 3: 
            raise TypeError("out must be a tuple containing 3 tensors") 
        for i, out_tensor in enumerate(out): 
            if not isinstance(out_tensor, TN): 
                raise TypeError(f"out[{i}] must be TN type") 
        if out[0].requires_grad or out[1].requires_grad or out[2].requires_grad or A.requires_grad:
            raise RuntimeError(f"svd(): functions with out=... arguments don't support automatic differentiation, but one of the arguments requires grad.")

    # 使用numpy.linalg.svd计算SVD，正确传递full_matrices参数 
    try: 
        U_data, S_data, Vh_data = arrlib.linalg.svd(A.data, full_matrices=full_matrices) 
    except arrlib.linalg.LinAlgError as e: 
        raise RuntimeError(f"SVD computation failed: {str(e)}")
    
    # 创建结果张量
    requires_grad = is_grad_enabled() and A.requires_grad
    U = tensor(U_data, device=dev, requires_grad=requires_grad)
    S = tensor(S_data, device=dev, requires_grad=requires_grad)
    Vh = tensor(Vh_data, device=dev, requires_grad=requires_grad)
    
    # 设置叶子节点状态
    U.is_leaf = not U.requires_grad
    S.is_leaf = not S.requires_grad
    Vh.is_leaf = not Vh.requires_grad
    
    # 如果需要梯度，设置fromvars和gradfuncs，并保存中间计算结果到parms中
    if requires_grad:
        U.fromvars = (A,)
        U.gradfuncs = (_svd_backward_u,)
        U.parms = ((U, S, Vh, full_matrices),)  # 保存full_matrices参数供梯度计算使用
        
        S.fromvars = (A,)
        S.gradfuncs = (_svd_backward_s,)
        S.parms = ((U, S, Vh, full_matrices),)  # 保存full_matrices参数供梯度计算使用
        
        Vh.fromvars = (A,)
        Vh.gradfuncs = (_svd_backward_vh,)
        Vh.parms = ((U, S, Vh, full_matrices),)  # 保存full_matrices参数供梯度计算使用
    
    # 处理out参数
    if out is not None:
        if out[0].shape != U.shape:
            raise ValueError(f"out[0] must have shape {U.shape}, got {out[0].shape}")
        if out[1].shape != S.shape:
            raise ValueError(f"out[1] must have shape {S.shape}, got {out[1].shape}")
        if out[2].shape != Vh.shape:
            raise ValueError(f"out[2] must have shape {Vh.shape}, got {out[2].shape}")
        
        out[0].copy_(U)
        out[1].copy_(S)
        out[2].copy_(Vh)
        return out
        
    return U, S, Vh

# U的梯度计算函数 - 支持full_matrices参数
def _svd_backward_u(result_tensor: TN, i: int) -> TN:
    """根据数学公式修正：计算SVD中U矩阵的反向传播梯度。"""
    # 从parms中获取保存的SVD张量结果
    U, S, Vh, full_matrices = result_tensor.parms[i]
    A = result_tensor.fromvars[i]
    
    # 获取上游传来的梯度 dL/dU
    G = result_tensor.grad_value
    
    # 获取矩阵维度
    m, n = A.shape[-2], A.shape[-1]
    r = builtins.min(m, n)  # 矩阵的最大可能秩，实际秩≤r
    # 数值稳定性处理时将接近0数值换为epsilon
    epsilon = 1e-12

    # 处理精简SVD情况：裁剪U, Vh到有效秩r
    if full_matrices:
        if m > r:
            U = U[..., :, :r]  # 形状 (m, r)
            G = G[..., :, :r]  # 形状 (m, r)
        if n > r:
            Vh = Vh[..., :r, :] # 形状 (r, n)

    # 计算奇异值的倒数，小于阈值的奇异值被设置为0
    # 为了避免除零错误，先创建一个掩码
    mask = S > epsilon
    # 只对非零奇异值计算倒数，0奇异值位置保持为0
    S_reciprocal = where(mask, 1.0 / where(mask, S, 1.0), 0.0)

    # 创建奇异值S的对角矩阵及其逆矩阵(非0奇异值取倒数，0奇异值不变，实际为伪逆矩阵)
    S_diag = batch_diag(S)
    S_inv_diag = batch_diag(S_reciprocal)
        
    # 使用矩阵广播直接计算分母矩阵 (S_i² - S_j²)
    S_squared = S ** 2.0
    S_squared_col = S_squared.unsqueeze(-1)
    S_squared_row = S_squared.unsqueeze(-2)
    denominator = S_squared_col - S_squared_row

    # 数值稳定性处理，避免除零错误
    # 将denominator对角线置1.0
    # 当denominator为0时，使用epsilon而不是0作为替代值
    denominator = denominator + eye(r, dtype=A.dtype, device=A.device)
    safe_denominator = where(denominator.abs() < epsilon, 
                            where(denominator >= 0, epsilon, -epsilon), 
                            denominator)
    
    # F矩阵计算：F = G^T @ U
    F = G.mH @ U
    FS = (F - F.mH )/safe_denominator
    term1 = U @ FS @ S_diag @ Vh  # 形状 (m, n)

    if m > n:
        I_m = eye(m, dtype=A.dtype, device=A.device)
        proj = I_m - U @ U.mH  # 形状 (m, m)
    
        # 计算第2项
        term2 = proj @ G @ S_inv_diag @ Vh  # 形状 (m, n)
        
        # 最终梯度为两项之和
        grad_A = term1 + term2  # 形状 (m, n)
    else:
        # 最终梯度为两项之和
        grad_A = term1  # 形状 (m, n)
    
    return grad_A

# Vh的梯度计算函数 - 支持full_matrices参数
def _svd_backward_vh(result_tensor: TN, i: int) -> TN:
    """计算SVD中Vh矩阵的反向传播梯度"""
    U, S, Vh, full_matrices = result_tensor.parms[i]
    A = result_tensor.fromvars[i]
    
    # 获取上游传来的梯度 dL/dVh
    H = result_tensor.grad_value  # 形状 (n, n)
    
    # 获取矩阵维度
    m, n = A.shape[-2], A.shape[-1]
    r = builtins.min(m, n)
    # 数值稳定性处理时将接近0数值换为epsilon
    epsilon = 1e-12
    
    # 处理精简SVD情况
    if full_matrices:
        if m > r:
            U = U[..., :, :r]  # 形状 (m, r)
        if n > r:
            Vh = Vh[..., :r, :] # 形状 (r, n)
            H = H[..., :r, :] # 形状 (r, n)

    # 计算奇异值的倒数，小于阈值的奇异值被设置为0
    # 为了避免除零错误，先创建一个掩码
    mask = S > epsilon
    # 只对非零奇异值计算倒数，0奇异值位置保持为0
    S_reciprocal = where(mask, 1.0 / where(mask, S, 1.0), 0.0)

    # 创建奇异值S的对角矩阵及其逆矩阵(非0奇异值取倒数，0奇异值不变，实际为伪逆矩阵)
    S_diag = batch_diag(S)
    S_inv_diag = batch_diag(S_reciprocal)
        
    # 计算分母矩阵 K
    S_squared = S ** 2.0
    S_squared_col = S_squared.unsqueeze(-1)
    S_squared_row = S_squared.unsqueeze(-2)
    denominator = S_squared_col - S_squared_row
    
    # 数值稳定性处理，避免除零错误
    # 将denominator对角线置1.0
    # 当denominator为0时，使用epsilon而不是0作为替代值
    denominator = denominator + eye(r, dtype=A.dtype, device=A.device)
    safe_denominator = where(denominator.abs() < epsilon, 
                            where(denominator >= 0, epsilon, -epsilon), 
                            denominator)
    
    # 计算反对称部分 - 修正复数矩阵的梯度计算
    V = Vh.mH  # 形状 (n, r)
    F = H @ V  # 形状 (r, n) @ (n, r) = (r, r)
    FS = (F - F.mH) / safe_denominator  # 形状 (r, r)

    # 调整term1的计算顺序以匹配PyTorch的实现
    term1 = U @ S_diag @ FS @ Vh  # 形状 (m, n)

    if m < n:
        I_n = eye(n, dtype=A.dtype, device=A.device)
        # 对于复数矩阵，投影矩阵计算需要考虑共轭
        proj = I_n - V @ Vh  # 形状 (n, n)
        
        # 调整term2的计算以正确处理复数矩阵
        term2 = U @ S_inv_diag @ H @ proj  # 形状 (m, n)
    
        # 最终梯度
        grad_A = term1 + term2
    else:
        grad_A = term1
    
    return grad_A

# S的梯度计算函数 - 支持full_matrices参数
def _svd_backward_s(result_tensor, i: int):
    """SVD分解中S的反向梯度传播函数 - 支持full_matrices参数"""
    # 从result_tensor中获取所需参数
    U, S, Vh, full_matrices = result_tensor.parms[i]
    A = result_tensor.fromvars[i]  # 原始输入张量
    grad_S = result_tensor.grad_value
    
    # 1. 首先检查并获取正确的维度信息
    orig_shape = A.shape
    
    # 2. 创建正确大小的对角矩阵
    # 确定对角矩阵的大小 - 它应该是min(orig_shape[-2], orig_shape[-1])
    m, n = A.shape[-2], A.shape[-1]
    r = builtins.min(m, n)
    
    # 处理精简SVD情况
    if full_matrices:
        if m > r:
            U = U[:, :r]  # 形状 (m, r)
        if n > r:
            Vh = Vh[:r, :] # 形状 (r, n)

    # 3. 创建对角矩阵
    S_grad_diag = batch_diag(grad_S)
    
    # 4. 计算梯度
    grad_A = U @ S_grad_diag @ Vh
    
    return grad_A

def pinv(A, atol=None, rtol=None, hermitian=False, out=None):
    """
    计算矩阵的Moore-Penrose伪逆。
    
    参数:
        A (TN): 输入张量，形状为(*, m, n)，其中*表示任意数量的批处理维度
        atol (float, optional): 绝对阈值，用于判断奇异值是否为零。如果为None，则使用默认值
        rtol (float, optional): 相对阈值，用于判断奇异值是否为零。如果为None，则使用默认值
        hermitian (bool, optional): 是否假设输入是厄米特矩阵或实对称矩阵，默认为False，当前不支持该参数
        
    返回:
        伪逆矩阵，形状为(*, n, m)
    """
    # 输入验证
    if not isinstance(A, TN):
        raise TypeError(f"Input must be TN type, got {type(A)}")
    
    # 验证输入维度至少为2
    if A.ndim < 2:
        raise ValueError(f"Input matrix must be at least 2-dimensional, got dimension {A.ndim}")
    
    # 对于hermitian=True的情况，在riemann中暂时不做特殊处理，与普通情况使用相同逻辑
    
    # 使用SVD分解
    U, S, Vh = svd(A, full_matrices=False)
    
    # 计算阈值
    # 如果没有提供atol和rtol，使用PyTorch的默认值
    if rtol is None:
        rtol = S.shape[-1] * np.finfo(S.dtype).eps
    
    # 确定哪些奇异值被认为是非零的
    # 首先计算有效的阈值
    if atol is None:
        # 使用最大奇异值乘以相对阈值
        # 对于批量矩阵，需要处理每个批次的最大奇异值
        if S.ndim > 1:
            # 批量矩阵情况
            max_singular_values = S[..., 0].unsqueeze(-1)  # 获取每个批次的最大奇异值并扩展维度
            threshold = max_singular_values * rtol
        else:
            # 单个矩阵情况
            max_singular_value = S[0] if len(S) > 0 else 0
            threshold = max_singular_value * rtol
    else:
        # 使用绝对阈值
        threshold = atol
    
    # 计算奇异值的倒数，小于阈值的奇异值被设置为0
    # 为了避免除零错误，先创建一个掩码
    mask = S > threshold
    # 只对非零奇异值计算倒数
    S_reciprocal = where(mask, 1.0 / where(mask, S, 1.0), 0.0)
    
    # 确保S_reciprocal的数据类型与输入矩阵A一致，避免反向传播时的类型不匹配
    if A.dtype in (np.complex64, np.complex128):
        S_reciprocal = S_reciprocal.type(A.dtype)
    
    # 创建S的伪逆对角矩阵
    S_inv = batch_diag(S_reciprocal)
    
    # 计算伪逆：pinv(A) = V @ S_inv @ U^H
    pinv_A = Vh.mH @ S_inv @ U.mH
    
    # 处理out参数
    if out is not None:
        # 检查out参数是否是TN类型
        if not isinstance(out, TN):
            raise TypeError(f"out must be TN type, got {type(out)}")
        
        # 检查out参数的形状是否与计算结果一致
        if out.shape != pinv_A.shape:
            raise RuntimeError(f"out has wrong shape: expected {pinv_A.shape}, got {out.shape}")
        
        # 将计算结果复制到out中
        return out.copy_(pinv_A)

    return pinv_A


def eig(A, *, out=None):
    """计算方阵的特征值和特征向量。
    
    该函数计算矩阵A的特征值分解
    
    参数:
        A (TN): 输入张量，形状为(*, n, n)，其中*表示任意数量的批处理维度
        out (tuple, optional): 输出元组的两个张量，默认为None
        
    返回:
        一个包含(w, V)的元组，其中:
        - w: 特征值，形状为(*, n)
        - V: 特征向量矩阵，形状为(*, n, n)，每一列是一个特征向量
        
    注意:
        - 输入必须是方阵
        - 特征值可能为复数，即使输入是实数矩阵
        - 特征向量矩阵V满足：A @ V = V @ diag(w)
    """
    # 输入验证 
    if not isinstance(A, TN): 
        raise TypeError(f"Input must be TN type, got {type(A)}") 
    
    # 验证输入维度至少为2 
    if A.ndim < 2: 
        raise ValueError(f"Input matrix must be at least 2-dimensional, got dimension {A.ndim}") 
    
    arrlib = A._get_array_lib()
    dev = A.device
    
    # 验证输入是方阵
    m, n = A.shape[-2], A.shape[-1]
    if m != n:
        raise ValueError(f"Input matrix must be square, got shape {A.shape}")
    
    # 处理out参数 
    if out is not None: 
        if not isinstance(out, tuple) or len(out) != 2: 
            raise TypeError("out must be a tuple containing 2 tensors") 
        for i, out_tensor in enumerate(out): 
            if not isinstance(out_tensor, TN): 
                raise TypeError(f"out[{i}] must be TN type") 
        if out.requires_grad or A.requires_grad:
            raise RuntimeError(f"eig(): functions with out=... arguments don't support automatic differentiation, but one of the arguments requires grad.")
        
    # 使用numpy.linalg.eig计算特征值分解 - 支持批量计算
    try: 
        # numpy.linalg.eig 原生支持批处理维度
        w_data, V_data = arrlib.linalg.eig(A.data)
    except arrlib.linalg.LinAlgError as e: 
        raise RuntimeError(f"Eig computation failed: {str(e)}")
        
    # 创建结果张量
    requires_grad = is_grad_enabled() and A.requires_grad
    # data_type = get_default_complex()
    w = tensor(w_data, device=dev, requires_grad = requires_grad)
    V = tensor(V_data, device=dev, requires_grad = requires_grad)
        
    # 设置叶子节点状态
    w.is_leaf = not w.requires_grad
    V.is_leaf = not V.requires_grad

    # 如果需要梯度，设置fromvars和gradfuncs，并保存中间计算结果到parms中
    if requires_grad:
        U = inv(V).mH # 计算左特征向量矩阵U

        w.fromvars = (A,)
        w.gradfuncs = (_eig_backward_w,)
        w.parms = ((w, V, U),)  # 保存特征值和特征向量供梯度计算使用
        
        V.fromvars = (A,)
        V.gradfuncs = (_eig_backward_V,)
        V.parms = ((w, V, U),)  # 保存特征值和特征向量供梯度计算使用
    
    # 处理out参数
    if out is not None:
        if out[0].shape != w.shape:
            raise ValueError(f"out[0] must have shape {w.shape}, got {out[0].shape}")
        if out[1].shape != V.shape:
            raise ValueError(f"out[1] must have shape {V.shape}, got {out[1].shape}")

        out[0].copy_(w)
        out[1].copy_(V)
        return out
        
    return w, V


def _eig_backward_w(result_tensor: TN, i: int) -> TN:
    """计算特征值的反向传播梯度 - 使用正确的解析公式"""
    w, V, U = result_tensor.parms[i]
    A = result_tensor.fromvars[i]
    
    # 获取上游传来的梯度 dL/dw
    G = result_tensor.grad_value
    
    # 使用特征值梯度解析公式：dL/dA = V^(-T) @ diag(G) @ V^T
    # 基于矩阵微分理论：如果 A = V @ diag(w) @ V^(-1)，那么
    # dL/dA = V^(-T) @ diag(dL/dw) @ V^T， U = V^(-T)
    
    # 创建对角矩阵 diag(G) - 使用Riemann张量操作
    G_diag = batch_diag(G)
    
    # 计算梯度：dL/dA = V^(-H) @ G_diag @ V^H - 使用Riemann张量计算
    grad_A = U @ G_diag @ V.mH
    
    return grad_A

def _eig_backward_V(result_tensor: TN, i: int) -> TN:
    """
    基于严格数学推导的非对称矩阵特征向量反向传播
    """
    w, V, U = result_tensor.parms[i]  # V_l 是左特征向量
    A = result_tensor.fromvars[i]
    dL_dV_r = result_tensor.grad_value  # ∂L/∂V_r
    
    n = A.shape[-1]
    
    # 1. 构建 Cauchy 矩阵 F
    w_col = w.unsqueeze(-1)      # (n, 1)
    w_row = w.unsqueeze(-2)      # (1, n)  
    w_diff = w_row - w_col       # (n, n)
    
    # 处理特征值相等的情况
    epsilon = 1e-12
    eye_mask = eye(n, dtype=bool_, device=A.device)
    w_diff = where(eye_mask, 1.0, w_diff)  # 对角线设为1避免除零
    w_diff_safe = where(w_diff.abs() < epsilon,
                        where(w_diff>=0.0, epsilon, -epsilon), 
                        w_diff)
    
    F = where(eye_mask, 0.0, 1.0 / w_diff_safe)
    
    # 2. 计算投影梯度
    proj_grad = U.mH @ dL_dV_r  # U^mH @ (∂L/∂V_r)
    
    # 3. Hadamard 积
    M = F * proj_grad  # 逐元素相乘
    
    # 4. 变换回原空间
    grad_A = U @ M @ V.mH
    
    return grad_A

def lstsq(A, B, *, rcond=None, out=None):
    """
    求解线性最小二乘问题：min_X ||AX - B||₂²
    
    参数:
        A (TN): 系数矩阵，形状为(*, m, n)，其中*表示任意数量的批处理维度
        B (TN): 观测值矩阵或向量，形状为(*, m, k), (*, m)
        rcond (float, optional): 相对条件数阈值，用于奇异值截断
        out (tuple, optional): 输出元组，包含(X, residuals, rank, singular_values)
        
    返回:
        一个包含四个元素的元组 (X, residuals, rank, singular_values):
        - X: 最小二乘解，形状为(*, n, k)
        - residuals: 残差平方和，形状为(*, k)
        - rank: 矩阵A的数值秩，形状为(*)
        - singular_values: 奇异值向量，形状为(*, min(m, n))
    """
    # 输入验证
    if not isinstance(A, TN):
        raise TypeError(f"Input A must be TN type, got {type(A)}")
    if not isinstance(B, TN):
        raise TypeError(f"Input B must be TN type, got {type(B)}")
    if A.device != B.device:
        raise ValueError(f"Input A and B must have the same device, got {A.device} and {B.device}")
    
    arrlib = A._get_array_lib()
        
    # 验证输入维度至少为2
    if A.ndim < 2:
        raise ValueError(f"Input matrix A must be at least 2-dimensional, got dimension {A.ndim}")
    
    # 获取批量维度 - 由A决定
    batch_dim = A.ndim - 2
    batch_shape = A.shape[:batch_dim]
    A_data_shape = A.shape[-2:]
    m = A_data_shape[0]

    # 记录是否需要在最后压缩维度
    needs_squeeze = False
    
    if B.ndim == 1: # B没有批量维
        # B是1D向量，将其扩展为2D列向量
        B = B.unsqueeze(-1)  # 转换为形状为(m, 1)
        needs_squeeze = True
        if B.shape[-2] != m:
            raise ValueError(f"Input B must be a 1D vector of length {m}, got shape {B.shape}")
    elif B.ndim >= 2: 
        if B.shape[-1] == m:
            B = B.unsqueeze(-1)  # 转换为形状为(*，m, 1)
            needs_squeeze = True
        elif B.shape[-2] == m:
            pass
        else:
            raise ValueError(f"Input B must be a 1D or 2D vector of length {m}, got shape {B.shape}")

    # 如果A、B有批量维，检查批量维形状是否匹配
    if B.shape[:-2] != () and batch_shape !=() and B.shape[:-2] != batch_shape:
        raise ValueError(f"Batch shape of B ({B.shape[:-2]}) must match batch shape of A ({batch_shape})")

    # 使用SVD分解
    U, S, Vh = svd(A, full_matrices=False)
    
    # 计算阈值
    if rcond is None:
        # 使用PyTorch默认的rcond值
        m, n = A_data_shape
        rcond = builtins.max(m, n) * arrlib.finfo(S.dtype).eps
    
    # 确定哪些奇异值被认为是非零的
    if S.ndim > 1:
        # 批量矩阵情况
        max_singular_values = S[..., 0].unsqueeze(-1)  # 获取每个批次的最大奇异值并扩展维度
        threshold = max_singular_values * rcond
    else:
        # 单个矩阵情况
        max_singular_value = S[0] if len(S) > 0 else 0
        threshold = max_singular_value * rcond
    
    # 计算奇异值的倒数，小于阈值的奇异值被设置为0
    mask = S > threshold
    # 只对非零奇异值计算倒数
    S_reciprocal = where(mask, 1.0 / where(mask, S, 1.0), 0.0)
    
    # 创建S的伪逆对角矩阵
    S_inv = batch_diag(S_reciprocal)
    
    # 计算伪逆矩阵：pinv_A = V @ S_inv @ U^H
    pinv_A = Vh.mH @ S_inv @ U.mH
    
    # 计算最小二乘解 X = pinv(A) @ B
    X = pinv_A @ B
    
    # 计算残差平方和 residuals = ||B - A @ X||₂²
    # 先计算 A @ X
    A_X = A @ X
    
    # 计算残差
    residuals = B - A_X
    # 计算残差平方和（对每个批次和每个输出列）
    # 对于复数情况，我们需要计算每个元素的模的平方
    residuals_squared = abs(residuals) ** 2.0
    # 在最后两个维度求和（对于所有情况）
    residuals = sum(residuals_squared, dim=(-2, -1))
    
    # 计算秩（非零奇异值的个数）
    # 对每个批次计算非零奇异值的个数
    rank = sum(mask, dim=-1)
    
    # 根据原始B的维度调整X的形状
    # 如果需要压缩维度（原始B是1D或B.ndim > batch_dim且remaining_dim=1）
    if needs_squeeze:
        X = X.squeeze(-1)
    
    # 处理out参数
    if out is not None:
        if not isinstance(out, tuple) or len(out) != 4:
            raise TypeError("out must be a tuple containing 4 tensors")
        for i, out_tensor in enumerate(out):
            if not isinstance(out_tensor, TN):
                raise TypeError(f"out[{i}] must be TN type")
        
        # 检查形状
        if out[0].shape != X.shape:
            raise ValueError(f"out[0] must have shape {X.shape}, got {out[0].shape}")
        if out[1].shape != residuals.shape:
            raise ValueError(f"out[1] must have shape {residuals.shape}, got {out[1].shape}")
        if out[2].shape != rank.shape:
            raise ValueError(f"out[2] must have shape {rank.shape}, got {out[2].shape}")
        if out[3].shape != S.shape:
            raise ValueError(f"out[3] must have shape {S.shape}, got {out[3].shape}")
        
        # 复制结果
        out[0].copy_(X)
        out[1].copy_(residuals)
        out[2].copy_(rank)
        out[3].copy_(S)
        return out
    
    return X, residuals, rank, S

def eigh(A, *, UPLO='L', out=None):
    """计算Hermitian（复数域对称）或实对称矩阵的特征值分解。
    
    该函数计算Hermitian矩阵的特征值分解，与torch.linalg.eigh保持一致。
    Hermitian矩阵满足 Aᴴ = A（复数域）或 Aᵀ = A（实数域）。
    
    参数:
        A (TN): 输入张量，形状为(*, n, n)，其中*表示任意数量的批处理维度
        UPLO (str, 可选): 指定使用矩阵的上三角('U')还是下三角('L')部分，默认为'L'
        out (tuple, 可选): 输出元组的两个张量，默认为None
        
    返回:
        一个包含(w, V)的元组，其中:
        - w: 特征值（实数），形状为(*, n)，按升序排列
        - V: 特征向量矩阵，形状为(*, n, n)，每一列是一个特征向量
        
    注意:
        - 输入必须是方阵
        - 特征值总是实数，即使输入是复数矩阵
        - 特征向量矩阵V满足：A @ V = V @ diag(w) 且 Vᴴ @ V = I
        - 特征值按升序排列
    """
    # 输入验证 
    if not isinstance(A, TN): 
        raise TypeError(f"Input must be TN type, got {type(A)}") 
    
    # 验证UPLO参数
    if UPLO not in ['L', 'U']:
        raise ValueError(f"UPLO must be 'L' or 'U', got {UPLO}")
    
    # 验证输入维度至少为2 
    if A.ndim < 2: 
        raise ValueError(f"Input matrix must be at least 2-dimensional, got dimension {A.ndim}") 
    
    # 验证输入是方阵
    m, n = A.shape[-2], A.shape[-1]
    if m != n:
        raise ValueError(f"Input matrix must be square, got shape {A.shape}")
    
    # 处理out参数 
    if out is not None: 
        if not isinstance(out, tuple) or len(out) != 2: 
            raise TypeError("out must be a tuple containing 2 tensors") 
        for i, out_tensor in enumerate(out): 
            if not isinstance(out_tensor, TN): 
                raise TypeError(f"out[{i}] must be TN type")  
        if out.requires_grad or A.requires_grad:
            raise RuntimeError(f"eigh(): functions with out=... arguments don't support automatic differentiation, but one of the arguments requires grad.")
        
    arrlib = A._get_array_lib()
    dev = A.device
    
    # 使用numpy.linalg.eigh计算特征值分解 - 支持批量计算
    try: 
        # numpy.linalg.eig 原生支持批处理维度
        w_data, V_data = arrlib.linalg.eigh(A.data,UPLO)
    except arrlib.linalg.LinAlgError as e: 
        raise RuntimeError(f"Eigh computation failed: {str(e)}")
    
    # 创建结果张量
    requires_grad = is_grad_enabled() and A.requires_grad
    w = tensor(w_data, device=dev, requires_grad = requires_grad)
    V = tensor(V_data, device=dev, requires_grad=requires_grad)
    
    # 设置叶子节点状态
    w.is_leaf = not w.requires_grad
    V.is_leaf = not V.requires_grad
    
    # 如果需要梯度，设置fromvars和gradfuncs，并保存中间计算结果到parms中
    if requires_grad:
        w.fromvars = (A,)
        w.gradfuncs = (_eigh_backward_w,)
        w.parms = ((w, V),)  # 保存特征值和特征向量供梯度计算使用
        
        V.fromvars = (A,)
        V.gradfuncs = (_eigh_backward_V,)
        V.parms = ((w, V),)  # 保存特征值和特征向量供梯度计算使用
    
    # 处理out参数
    if out is not None:
        if out[0].shape != w.shape:
            raise ValueError(f"out[0] must have shape {w.shape}, got {out[0].shape}")
        if out[1].shape != V.shape:
            raise ValueError(f"out[1] must have shape {V.shape}, got {out[1].shape}")

        out[0].copy_(w)
        out[1].copy_(V)
        return out
        
    return w, V

def _eigh_backward_w(result_tensor: TN, i: int) -> TN:
    """计算特征值的反向传播梯度 - 使用正确的解析公式"""
    w, V = result_tensor.parms[i]
    A = result_tensor.fromvars[i]
    
    # 获取上游传来的梯度 dL/dw
    G = result_tensor.grad_value
    
    # 使用特征值梯度解析公式：dL/dA = V^(-T) @ diag(G) @ V^T
    # 基于矩阵微分理论：如果 A = V @ diag(w) @ V^(-1)，那么
    # dL/dA = V^(-T) @ diag(dL/dw) @ V^T
    
    # 计算 V^(-H) = (V^H)^(-1)
    # 对于对称矩阵或Hermitian矩阵的特征向量矩阵（单位化后），V^(-1) = V.mH
    # V_inv_H = inv(V).mH = V
    V_inv_H = V
    
    # 创建对角矩阵 diag(G) - 使用Riemann张量操作
    G_diag = batch_diag(G)
    
    # 计算梯度：dL/dA = V^(-H) @ G_diag @ V^H
    grad_A = V_inv_H @ G_diag @ V.mH
    
    return grad_A

def _eigh_backward_V(result_tensor: TN, i: int) -> TN:
    """改进的特征向量梯度计算"""
    w, V = result_tensor.parms[i]
    A = result_tensor.fromvars[i]
    G = result_tensor.grad_value
    
    n = A.shape[-1]
        
    # 向量化计算特征值差异矩阵
    w_col = w.unsqueeze(-1)  # (..., n, 1)
    w_row = w.unsqueeze(-2)  # (..., 1, n)  
    w_diff = w_row - w_col   # (..., n, n)
    
    # w_diff对角线置为1，非对角线接近0元素置为epsilon，避免除零错误
    epsilon = 1e-12
    eye_mask = eye(n, dtype=bool_, device=A.device)
    w_diff = where(eye_mask, 1.0, w_diff)
    w_diff_safe = where(w_diff.abs() < epsilon,
                        where(w_diff>=0.0,epsilon,-epsilon), 
                        w_diff)

    # 计算 F 矩阵
    Vh = V.mH
    Vh_G = Vh @ G  # (..., n, n)
    Vh_G_div_w_diff = Vh_G / w_diff_safe
    # 处理对角线元素：F_ii = 0
    F = where(eye_mask, 0.0, Vh_G_div_w_diff)

    # 计算最终梯度
    V_inv_H = inv(Vh)
    grad = V_inv_H @ F @ Vh    
    grad_A = (grad + grad.mH) / 2.0

    return grad_A


def _squared_mat_lu(A, *, pivot=True):
    """
    计算矩阵的LU分解。
    
    该函数计算矩阵的LU分解，与torch.lu保持一致。分解形式取决于pivot参数：
    - 当pivot=True时：A = PLU，其中P是置换矩阵
    - 当pivot=False时：A = LU，不使用行交换
    
    参数:
        A (TN): 输入张量，形状为(*, m, n)，其中*表示任意数量的批处理维度
        pivot (bool, 可选): 是否使用主元交换，默认为True; pivot=False时，目前代码未实现
        
    返回:
        一个包含(P, L, U)的元组，其中:
        - P: 置换矩阵，形状为(*, m, m)。当pivot = False时为空矩阵
        - L: 单位下三角矩阵，形状为(*, m, m) 
        - U: 上三角矩阵，形状为(*, m, n)
        
    注意:
        - 输入矩阵可以是矩形矩阵(m×n)
        - 当pivot=True时，使用scipy.linalg.lu(..., permute_l=False)得到(P,L,U)，A=PLU
        - 当pivot=False时，不使用主元交换，数值稳定性可能降低,目前未实现
        - L的对角线元素始终为1
        - 支持批量处理
    """
    # 输入验证 
    if not isinstance(A, TN): 
        raise TypeError(f"Input must be TN type, got {type(A)}") 
    
    dev = A.device
    
    # 验证输入维度至少为2 
    if A.ndim < 2: 
        raise ValueError(f"Input matrix must be at least 2-dimensional, got dimension {A.ndim}") 
    
    # 使用scipy.linalg.lu计算LU分解 - 支持批量计算
    try: 
        # 使用scipy.linalg.lu计算LU分解
        # permute_l=False: 返回(P, L, U)满足 PA = LU
        # permute_l=True: 返回(PL, U) 当前未实现
        
        if pivot:
            # 使用主元交换，返回(P, L, U)
            P_data, L_data, U_data = scipy.linalg.lu(A.data, permute_l=False)
        else:
            raise NotImplementedError("When pivot = False, LU is not implemented yet")
            
    except Exception as e: 
        raise RuntimeError(f"LU decomposition failed: {str(e)}")
    
    # 创建结果张量
    requires_grad = is_grad_enabled() and A.requires_grad
    # 当pivot=False时，P_data为None，此时P也设为None
    P = tensor(P_data, device=dev)  # 置换矩阵不需要梯度
    L = tensor(L_data, device=dev, requires_grad=requires_grad)
    U = tensor(U_data, device=dev, requires_grad=requires_grad)
    
    # 设置叶子节点状态
    L.is_leaf = not L.requires_grad
    U.is_leaf = not U.requires_grad
    
    # 如果需要梯度，设置fromvars和gradfuncs，并保存中间计算结果到parms中
    if requires_grad:
        L.fromvars = (A,)
        L.gradfuncs = (_lu_backward_l,)
        L.parms = ((P, L, U),)  # 保存P, L, U和pivot参数供梯度计算使用
        
        U.fromvars = (A,)
        U.gradfuncs = (_lu_backward_u,)
        U.parms = ((P, L, U),)  # 保存P, L, U和pivot参数供梯度计算使用
    
    return P, L, U

def _lu_backward_l(result_tensor: TN, i: int) -> TN:
    """L的反向传播梯度计算
    梯度公式: d y / d A = P^T * L^{-T} * tril( L^T * (d y / d L) ) * U^{-T}
    """
    # scipy lu分解A=PLU，此时的P就是理论分解(PA=LU)中P的转置P^T
    P, L, U = result_tensor.parms[i] 
    
    # 获取上游传来的梯度 dL/dL
    G = result_tensor.grad_value
    L_T = L.mH
    
    grad_A = P @ pinv(L_T) @ tril(L_T @ G, diagonal=-1) @ pinv(U.mH)
    return grad_A


def _lu_backward_u(result_tensor: TN, i: int) -> TN:
    """U的反向传播梯度计算。
    
    梯度公式: d y / d A = P^T * L^{-T} * upper_triangular((d y / d U) * U^T) * U^{-T}
    
    """
    # scipy lu分解A=PLU，此时的P就是理论分解(PA=LU)中P的转置P^T
    P, L, U = result_tensor.parms[i]
    
    # 获取上游传来的梯度 dL/dU
    G = result_tensor.grad_value
    U_T = U.mH
    grad_A = P @ pinv(L.mH) @ triu(G @ U_T) @ pinv(U_T)
    
    return grad_A

def lu(A, *, pivot=True, out=None):
    """
    计算矩阵的LU分解。
    
    该函数计算矩阵的LU分解，与torch.lu保持一致。分解形式取决于pivot参数：
    - 当pivot=True时：A = PLU，其中P是置换矩阵
    - 当pivot=False时：A = LU，不使用行交换
    
    参数:
        A (TN): 输入张量，形状为(*, m, n)，其中*表示任意数量的批处理维度
        pivot (bool, 可选): 是否使用主元交换，默认为True; pivot=False时，目前代码未实现
        out (tuple, 可选): 输出元组的三个张量(P, L, U)，默认为None
        
    返回:
        一个包含(P, L, U)的元组，其中:
        - P: 置换矩阵，形状为(*, m, m)。当pivot = False时为空矩阵
        - L: 单位下三角矩阵，形状为(*, m, m) 
        - U: 上三角矩阵，形状为(*, m, n)
        
    注意:
        - 输入矩阵可以是矩形矩阵(m×n)
        - 当pivot=True时，使用scipy.linalg.lu(..., permute_l=False)得到(P,L,U)，A=PLU
        - 当pivot=False时，不使用主元交换，数值稳定性可能降低,目前未实现
        - L的对角线元素始终为1
        - 支持批量处理
    """
    # 输入验证 
    if not isinstance(A, TN): 
        raise TypeError(f"linalg.lu:Input must be TN type, got {type(A)}") 
    
    # 验证输入维度至少为2 
    if A.ndim < 2: 
        raise ValueError(f"linalg.lu:Input matrix must be at least 2-dimensional, got dimension {A.ndim}") 
    
    if not pivot:
        raise NotImplementedError("linalg.lu: When pivot = False, LU is not implemented yet")
            
    # 处理out参数 
    if out is not None: 
        if not isinstance(out, tuple) or len(out) != 3: 
            raise TypeError("out must be a tuple containing 3 tensors") 
        for i, out_tensor in enumerate(out): 
            if not isinstance(out_tensor, TN): 
                raise TypeError(f"out[{i}] must be TN type")  
        if out.requires_grad or A.requires_grad:
            raise RuntimeError(f"lu(): functions with out=... arguments don't support automatic differentiation, but one of the arguments requires grad.")
    
    # 获取矩阵维度
    m, n = A.shape[-2], A.shape[-1]

    if m == n :
        # 直接调用方阵的LU分解
        ret_P,ret_L,ret_U = _squared_mat_lu(A, pivot=pivot)
    
    elif m > n:
        # 对高廋形状矩阵，右侧补0列，转为方阵后再计算LU分解
        # 创建具有正确批量维度的padding矩阵
        batch_dims = A.shape[:-2]
        padding = zeros(batch_dims + (m, m - n), dtype=A.dtype, device=A.device)

        # 按列批量合并A和padding，得到A_p
        A_p = concatenate((A, padding), dim=-1)
        P, L, U = _squared_mat_lu(A_p, pivot=pivot)

        # 按A原始形状，剪除P、L、U的多余部分，P(m,m)，L(m,n)，U(n,n)
        ret_P,ret_L,ret_U = P, L[...,:,:n], U[...,:n,:n]

    else: # m < n
        # 对宽廋形状矩阵，下侧补斜对角为1的行，转为方阵后再计算LU分解
        # 创建具有正确批量维度的padding矩阵
        batch_dims = A.shape[:-2]
        padding = zeros(batch_dims + (n - m, n), dtype=A.dtype, device=A.device)        
        # 在对角线位置设置1
        padding[..., range(n - m), range(m,n)] = 1.0
        
        # 按行批量合并A和padding，得到A_p
        A_p = concatenate((A, padding), dim=-2)
        P, L, U = _squared_mat_lu(A_p, pivot=pivot)

        # 按A原始形状，剪除P、L、U的多余部分，P(m,m)，L(m,m)，U(m,n)
        ret_P,ret_L,ret_U = P[...,:m,:m], L[...,:m,:m], U[...,:m,:n]

    # 处理out参数
    if out is not None:
        if out[0].shape != ret_P.shape:
            raise ValueError(f"out[0] must have shape {ret_P.shape}, got {out[0].shape}")
        if out[1].shape != ret_L.shape:
            raise ValueError(f"out[1] must have shape {ret_L.shape}, got {out[1].shape}")
        if out[2].shape != ret_U.shape:
            raise ValueError(f"out[2] must have shape {ret_U.shape}, got {out[2].shape}")
        
        out[0].copy_(ret_P)
        out[1].copy_(ret_L)
        out[2].copy_(ret_U)
        return out
    
    return ret_P,ret_L,ret_U

def solve(A, B, *, left=True, out=None):
    """
    求解线性方程组 AX = B 或 XA = B。
    
    该函数求解线性方程组，与torch.linalg.solve保持一致。支持批处理矩阵和
    多种数据类型（float、double、cfloat、cdouble）。
    
    参数:
        A (TN): 系数矩阵，形状为(*, n, n)，其中*表示任意数量的批处理维度
        B (TN): 右侧矩阵，形状为(*, n)或(*, n, k)或(n,)或(n, k)
        left (bool, 可选): 是否求解AX=B（left=True）或XA=B（left=False）。默认True
        out (TN, 可选): 输出张量。默认为None
        
    返回:
        TN: 解矩阵X，形状与B相同
        
    异常:
        RuntimeError: 如果A不可逆或批处理中的任何矩阵不可逆
        TypeError: 如果输入不是TN类型
        ValueError: 如果维度不匹配
        
    注意:
        - 使用X = A^(-1) @ B公式计算
        - 矩阵求逆已支持自动梯度跟踪，无需额外回调函数
        - 支持复数张量
        - 当left=False时，通过转置输入和输出实现XA=B的求解
    """
    # 输入验证
    if not isinstance(A, TN):
        raise TypeError(f"A must be TN type, got {type(A)}")
    if not isinstance(B, TN):
        raise TypeError(f"B must be TN type, got {type(B)}")
    if A.device != B.device:
        raise ValueError(f"A and B must have the same device, got {A.device} and {B.device}")
    
    if A.dtype.kind not in ['f', 'c']:
        raise RuntimeError(f"linalg.solve: A must be floating point or complex tensor, got {A.dtype.name}")
    if B.dtype.kind not in ['f', 'c']:
        raise RuntimeError(f"linalg.solve: B must be floating point or complex tensor, got {B.dtype.name}")
    
    # 检查维度
    if A.ndim < 2:
        raise ValueError(f"A must be at least 2-dimensional, got {A.ndim}")
    if B.ndim < 1:
        raise ValueError(f"B must be at least 1-dimensional, got {B.ndim}")
    
    # 检查A是否为方阵
    if A.shape[-1] != A.shape[-2]:
        raise ValueError(f"A must be square in the last two dimensions, got shape {A.shape}")

    # 处理left=False的情况（求解XA=B）
    if not left:
        # XA = B 等价于 A^T X^T = B^T
        # 所以先求解 A^T Y = B^T，然后返回 Y^T
        A_T = A.mT  # A的转置

        B_is_converted_to_2D_row_vec = False
        if B.ndim >= 2:
            if B.shape[-2] == 1:     # (*,1,n) 2D行向量
                B_T = B.mT
            else:                    # (*,n) 批量1D向量
                B_T = B.unsqueeze(-2).mT  # (*,n)>>(*,1,n) 批量2D列向量
                B_is_converted_to_2D_row_vec = True
        else:
            B_T = B

        Y = solve(A_T, B_T, left=True)

        if B.ndim >= 2:
            if B_is_converted_to_2D_row_vec:
                Y = Y.mT.squeeze(-2)  # (*,1,n)>>(*,n) 批量1D向量
            else:
                Y = Y.mT
        return Y
    
    # 检查批处理维度兼容性
    batch_dims_A = A.shape[:-2]
    A_row_size = A.shape[-2]
    B_is_converted_to_2D_col_vec = False

    if B.ndim == 1:              # (n,) 1D向量
        batch_dims_B = ()
        B_vector_size = B.shape[0]
    elif B.ndim >= 2:
        if B.shape[-1] == 1:     # (*,n,1) 2D列向量
            batch_dims_B = B.shape[:-2]
            B_vector_size = B.shape[-2]
        else:                    # (*,n) 批量1D向量
            batch_dims_B = B.shape[:-1]
            B_vector_size = B.shape[-1]
            B = B.unsqueeze(-1)  # (*,n)>>(*,n,1) 批量2D列向量
            B_is_converted_to_2D_col_vec = True
    
    if A_row_size != B_vector_size:
        if left:
            raise ValueError(f"Row size of A ({A_row_size}) miss match vector size of B ({B_vector_size})")
        else:
            raise ValueError(f"Column size of A ({A_row_size}) miss match vector size of B ({B_vector_size})")
    
    if batch_dims_A != batch_dims_B:
        raise ValueError(f"Batch dimensions of A {batch_dims_A} and B {batch_dims_B} are not compatible")
    
    # 使用矩阵求逆和矩阵乘法求解 - inv函数已支持梯度跟踪
    try:
        A_inv = inv(A)  # 计算A的逆矩阵，自动支持梯度跟踪
        # if batch_dims_B == ():
        X = A_inv @ B
        if B_is_converted_to_2D_col_vec:
             X = X.squeeze(-1)
        
    except RuntimeError as e:
        if "not invertible" in str(e):
            raise RuntimeError("Matrix is not invertible")
        else:
            raise RuntimeError(f"Linear system solution failed: {str(e)}")
    
    # 处理out参数
    if out is not None:
        if not isinstance(out,TN):
            raise TypeError(f"out must be TN type, got {type(out)}")
        
        if out.shape != X.shape:
            raise ValueError(f"out has wrong shape: expected {X.shape}, got {out.shape}")
        
        if out.requires_grad or A.requires_grad or B.requires_grad:
            raise RuntimeError(f"solve(): functions with out=... arguments don't support automatic differentiation, but one of the arguments requires grad.")
        
        out.copy_(X)
        return out
        
    return X

def qr(A, mode='reduced', *, out=None):
    """计算矩阵的QR分解。
    
    该函数计算矩阵A的QR分解，与torch.linalg.qr保持一致。支持批处理矩阵和
    多种数据类型（float、double、cfloat、cdouble）。
    
    QR分解将矩阵A分解为正交矩阵Q和上三角矩阵R，使得A = QR。
    根据mode参数的不同，可以返回简化QR分解、完全QR分解或仅返回R矩阵。
    
    参数:
        A (TN): 输入张量，形状为(*, m, n)，其中*表示任意数量的批处理维度
        mode (str, 可选): QR分解模式，可选值为'reduced'、'complete'、'r'。默认为'reduced'
            - 'reduced': 返回简化QR分解，Q形状为(*, m, k)，R形状为(*, k, n)，k=min(m,n)
            - 'complete': 返回完全QR分解，Q形状为(*, m, m)，R形状为(*, m, n)
            - 'r': 仅返回R矩阵，Q为空张量，R形状为(*, k, n)，k=min(m,n)
        out (tuple, 可选): 输出元组的两个张量，默认为None
        
    返回:
        tuple: (Q, R)元组，其中Q和R是QR分解的结果
        
    异常:
        TypeError: 如果输入不是TN类型
        ValueError: 如果mode参数无效或维度不匹配
        RuntimeError: 如果QR分解计算失败
        
    注意:
        - mode='r'不支持梯度反向传播
        - R矩阵的对角线元素不一定为正
        - QR分解仅在A的前k=min(m,n)列线性独立时才可计算
    """
    # 输入验证
    if not isinstance(A, TN):
        raise TypeError(f"Input must be TN type, got {type(A)}")
    
    arrlib = A._get_array_lib()
    dev = A.device
    
    # 验证输入维度至少为2
    if A.ndim < 2:
        raise ValueError(f"Input matrix must be at least 2-dimensional, got dimension {A.ndim}")
    
    # 验证mode参数
    valid_modes = ['reduced', 'complete', 'r']
    if mode not in valid_modes:
        raise ValueError(f"qr() got an invalid mode: '{mode}'. Expected one of {valid_modes}")
    
    # 获取矩阵维度
    m, n = A.shape[-2], A.shape[-1]
    k = builtins.min(m, n)
    
    # 处理out参数
    if out is not None:
        if not isinstance(out, tuple) or len(out) != 2:
            raise TypeError("out must be a tuple containing 2 tensors")
        for i, out_tensor in enumerate(out):
            if not isinstance(out_tensor, TN):
                raise TypeError(f"out[{i}] must be TN type")
        
        # # mode='r'不支持梯度
        # if mode == 'r' and (out[0].requires_grad or out[1].requires_grad or A.requires_grad):
        #     raise RuntimeError("qr(..., mode='r') does not support automatic differentiation")
        
        # 不支持out参数的梯度
        if A.requires_grad or out[0].requires_grad or out[1].requires_grad or A.requires_grad:
            raise RuntimeError(f"qr() with out=... arguments don't support automatic differentiation, but one of the arguments requires grad.")
    
    # 使用numpy.linalg.qr计算QR分解
    try:
        if mode == 'r':
            # mode='r'时，只计算R矩阵
            R_data = arrlib.linalg.qr(A.data, mode='r')
            Q_data = arrlib.empty((0,), dtype=A.dtype)
        else:
            # mode='reduced'或'complete'时，计算完整的QR分解
            Q_data, R_data = arrlib.linalg.qr(A.data, mode=mode)
    except arrlib.linalg.LinAlgError as e:
        raise RuntimeError(f"QR decomposition failed: {str(e)}")
    
    # 创建结果张量
    requires_grad = is_grad_enabled() and A.requires_grad and mode != 'r'
    
    if mode == 'r':
        Q = tensor(Q_data, device=dev, requires_grad=False)  # mode='r'的Q始终不需要梯度
        R = tensor(R_data, device=dev, requires_grad=False)  # mode='r'的R始终不需要梯度
    else:
        Q = tensor(Q_data, device=dev, requires_grad=requires_grad)
        R = tensor(R_data, device=dev, requires_grad=requires_grad)
    
    # 设置叶子节点状态
    Q.is_leaf = not Q.requires_grad
    R.is_leaf = not R.requires_grad
    
    # 如果需要梯度，设置fromvars和gradfuncs（mode='r'除外）
    if requires_grad and mode != 'r':
        Q.fromvars = (A,)
        Q.gradfuncs = (_qr_backward_q,)
        Q.parms = ((Q, R, mode),)  # 保存Q, R和mode参数供梯度计算使用
        
        R.fromvars = (A,)
        R.gradfuncs = (_qr_backward_r,)
        R.parms = ((Q, R, mode),)  # 保存Q, R和mode参数供梯度计算使用
    
    # 处理out参数
    if out is not None:
        # 检查形状匹配
        if Q.shape != out[0].shape:
            raise ValueError(f"out[0] must have shape {Q.shape}, got {out[0].shape}")
        if R.shape != out[1].shape:
            raise ValueError(f"out[1] must have shape {R.shape}, got {out[1].shape}")
        
        out[0].copy_(Q)
        out[1].copy_(R)
        return out
        
    return Q, R

# Q的梯度计算函数
def _qr_backward_q(result_tensor: TN, i: int) -> TN:
    """QR分解中Q矩阵的反向传播梯度计算。"""
    Q, R, mode = result_tensor.parms[i]
    A = result_tensor.fromvars[i]
    
    # 获取上游传来的梯度 dL/dQ
    G = result_tensor.grad_value
    
    # 获取矩阵维度
    m, n = A.shape[-2], A.shape[-1]
    
    if mode == 'complete':
        if m > n:
            # Q(m,m),R(m,n)
            # 完整分解时，剪除Q、G右侧的列，剪除R下方的行
            G = G[..., :n]      # (*, m, n)
            Q = Q[..., :n]      # (*, m, n)
            R_squared = R[..., :n, :]   # (*, n, n)
        elif m < n:
            # Q(m,m),R(m,n)
            # m < n时剪除R右侧列        
            R_squared = R[..., :m]          # (*, m, m)
        else:
            R_squared = R
    elif mode == 'reduced':
        if m < n: 
            # Q(m,m),R(m,n)
            # m < n时剪除R右侧列       
            R_squared = R[..., :m]          # (*, m, m)
        else:
            # Q(m,n),R(n,n)
            R_squared = R
    else:
        pass
    
    arrlib = A._get_array_lib()
    dev = A.device
    
    # 计算 R^{-T}
    try:
        R_inv_T = inv(R_squared.mH)
    except RuntimeError:
        return full_like(A, arrlib.nan, device=dev, dtype=A.dtype)
    
    # R^{-T}右侧补0，恢复形状为(*, m, n)
    if m < n:
        shape = R.shape[:-2] + (m, n-m)
        padding = zeros(shape, dtype=R.dtype, device=dev)
        R_inv_T = concatenate((R_inv_T, padding), dim=-1)  # (*, m, n)

    # 根据数学公式计算梯度
    # dL/dA = Q * (Q^T(dL/dQ) - (dL/dQ)^T * R^{-T} + (I - Q Q^T) * (dL/dQ) * R^{-T}
    
    # 计算 Q^T * G
    Qt_G = Q.mH @ G       # m>n, (*, n, n); m<n, (*, m, m)
    
    # 计算反对称部分的严格下三角
    # skew_part = (Qt_G - Qt_G.mH ).tril(diagonal = -1)
    skew_part = (Qt_G - Qt_G.mH )   # m>n, (*, n, n); m<n, (*, m, m)

    # 计算第一项: Q * (Qt_G - Gt_Q) * R^{-T}
    term1 = Q @ skew_part @ R_inv_T  # (*, m, n)
        
    if m > n:
        # 计算第二项: (I - Q Q^T) * G * R^{-T}
        G_R_inv_T = G @ R_inv_T
        I = eye(m, dtype=A.dtype, device=dev)
        proj = I - Q @ Q.mH    
        term2 = proj @ G_R_inv_T    
        grad_A = term1 + term2
    else:
        grad_A = term1
    
    return grad_A

# R的梯度计算函数
def _qr_backward_r(result_tensor: TN, i: int) -> TN:
    """QR分解中R矩阵的反向传播梯度计算。"""
    Q, R, mode = result_tensor.parms[i]
    A = result_tensor.fromvars[i]
    
    # 获取上游传来的梯度 dL/dR
    H = result_tensor.grad_value
    
    # 获取矩阵维度
    m, n = A.shape[-2], A.shape[-1]
    
    if mode == 'complete':
        if m > n:
            # 完整分解时，剪除Q、G右侧的列，剪除R下方的行
            H = H[..., :n, :]      # (*, m, n)
            Q = Q[..., :n]      # (*, m, n)
            R_squared = R[..., :n, :]   # (*, n, n)
        elif m < n:
            # m < n时剪除R右侧列        
            R_squared = R[..., :m]          # (*, m, m)
        else:
            R_squared = R
    elif mode == 'reduced':
        if m < n: 
            # m < n时剪除R右侧列       
            R_squared = R[..., :m]          # (*, m, m)
        else:
            R_squared = R
    else:
        pass
    
    arrlib = A._get_array_lib()
    dev = A.device
    
    # 计算 R^{-T}
    try:
        R_inv_T = inv(R_squared.mH)
    except RuntimeError:
        return full_like(A, arrlib.nan, device=dev, dtype=A.dtype)
    
    # R^{-T}右侧补0，恢复形状为(*, m, n)
    if m < n:
        shape = R.shape[:-2] + (m, n-m)
        padding = zeros(shape, dtype=R.dtype, device=dev)
        R_inv_T = concatenate((R_inv_T, padding), dim=-1)  # (*, m, n)
    
    # 计算梯度: Q @ (H + (R @ H^T- H @ R^T).tril() @ R^(-T))
    R_H_T = R @ H.mH
    skew_part:TN = R_H_T - R_H_T.mH
    # grad_A = Q @ (H + skew_part.tril(diagonal = -1) @ R_inv_T)
    grad_A = Q @ (H + skew_part @ R_inv_T)

    return grad_A