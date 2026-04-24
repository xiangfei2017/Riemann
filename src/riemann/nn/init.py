# BSD 3-Clause License
#
# Copyright (c) 2025, Fei Xiang
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
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

"""神经网络参数初始化工具模块 (Neural Network Parameter Initialization)

此模块包含用于初始化神经网络参数的实用函数，与PyTorch的nn.init模块保持接口一致。
"""

import math
import warnings
from typing import Literal, Optional, Union

import numpy as np

from ..tensordef import TN, no_grad
from ..cuda import cp


__all__ = [
    "calculate_gain",
    "uniform_",
    "normal_",
    "trunc_normal_",
    "constant_",
    "ones_",
    "zeros_",
    "eye_",
    "dirac_",
    "xavier_uniform_",
    "xavier_normal_",
    "kaiming_uniform_",
    "kaiming_normal_",
    "orthogonal_",
    "sparse_",
]


_NonlinearityType = Literal[
    "linear",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "sigmoid",
    "tanh",
    "relu",
    "leaky_relu",
    "selu",
]

_FanMode = Literal["fan_in", "fan_out"]


def _get_array_lib(tensor: TN):
    """获取张量使用的数组库 (numpy 或 cupy)"""
    if tensor.device.type == 'cpu':
        return np
    else:
        return cp


def _no_grad_uniform_(tensor: TN, a: float, b: float) -> TN:
    """无梯度均匀分布初始化 (Uniform Initialization without Gradient)"""
    arrlib = _get_array_lib(tensor)
    if tensor.device.type == 'cpu':
        tensor.data[:] = arrlib.random.uniform(a, b, tensor.shape).astype(tensor.dtype)
    else:
        with cp.cuda.Device(tensor.device.index):
            tensor.data[:] = arrlib.random.uniform(a, b, tensor.shape).astype(tensor.dtype)
    return tensor


def _no_grad_normal_(tensor: TN, mean: float, std: float) -> TN:
    """无梯度正态分布初始化 (Normal Initialization without Gradient)"""
    arrlib = _get_array_lib(tensor)
    if tensor.device.type == 'cpu':
        tensor.data[:] = arrlib.random.normal(mean, std, tensor.shape).astype(tensor.dtype)
    else:
        with cp.cuda.Device(tensor.device.index):
            tensor.data[:] = arrlib.random.normal(mean, std, tensor.shape).astype(tensor.dtype)
    return tensor


def _no_grad_trunc_normal_(
    tensor: TN, mean: float, std: float, a: float, b: float
) -> TN:
    """无梯度截断正态分布初始化 (Truncated Normal Initialization without Gradient)
    
    方法基于 https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """
    def norm_cdf(x: float) -> float:
        # 计算标准正态累积分布函数
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # 使用截断均匀分布生成值，然后使用正态分布的逆CDF
    # 获取上下CDF值
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # 用[l, u]中的值均匀填充张量，然后转换到[2l-1, 2u-1]
    arrlib = _get_array_lib(tensor)
    if tensor.device.type == 'cpu':
        uniform_vals = arrlib.random.uniform(2 * l - 1, 2 * u - 1, tensor.shape)
    else:
        with cp.cuda.Device(tensor.device.index):
            uniform_vals = arrlib.random.uniform(2 * l - 1, 2 * u - 1, tensor.shape)

    # 使用逆CDF变换得到截断标准正态
    # erfinv(x) = sqrt(2) * inverse_cdf((x+1)/2)
    normal_vals = math.sqrt(2.0) * std * _erfinv(uniform_vals) + mean
    
    # 裁剪确保在正确范围内
    normal_vals = arrlib.clip(normal_vals, a, b)
    
    tensor.data[:] = normal_vals.astype(tensor.dtype)
    return tensor


def _erfinv(x):
    """逆误差函数 (Inverse Error Function)
    
    使用近似方法计算erfinv。
    """
    # 使用numpy的erfinv（如果可用）或近似实现
    if hasattr(np, 'erfinv'):
        return np.erfinv(x)
    else:
        # 简单的近似实现
        # 基于Winitzki的近似公式
        a = 8 * (np.pi - 3) / (3 * np.pi * (4 - np.pi))
        y = np.log(1 - x * x)
        z = 2 / (np.pi * a) + y / 2
        return np.sign(x) * np.sqrt(np.sqrt(z * z - y / a) - z)


def _no_grad_fill_(tensor: TN, val: float) -> TN:
    """无梯度填充 (Fill without Gradient)"""
    with no_grad():
        tensor.fill_(val)
    return tensor


def _no_grad_zero_(tensor: TN) -> TN:
    """无梯度置零 (Zero without Gradient)"""
    with no_grad():
        tensor.zero_()
    return tensor


def calculate_gain(
    nonlinearity: _NonlinearityType, param: Optional[Union[int, float]] = None
) -> float:
    r"""返回给定非线性函数的推荐增益值 (Calculate Gain for Nonlinearity)

    各非线性函数对应的增益值如下:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    SELU              :math:`\frac{3}{4}`
    ================= ====================================================

    Args:
        nonlinearity: 非线性函数名称 (`nn.functional` 中的名称)
        param: 非线性函数的可选参数

    Returns:
        float: 推荐的增益值

    Examples:
        >>> gain = rm.init.calculate_gain("leaky_relu", 0.2)
    """
    linear_fns = [
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    ]
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        if param is None:
            negative_slope = 0.01
        elif (
            not isinstance(param, bool)
            and isinstance(param, int)
            or isinstance(param, float)
        ):
            negative_slope = param
        else:
            raise ValueError(f"negative_slope {param} not a valid number")
        return math.sqrt(2.0 / (1 + negative_slope**2))
    elif nonlinearity == "selu":
        return 3.0 / 4
    else:
        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def uniform_(tensor: TN, a: float = 0.0, b: float = 1.0) -> TN:
    r"""用均匀分布的值填充输入张量 (Uniform Initialization)

    从均匀分布 :math:`\mathcal{U}(a, b)` 中采样值填充张量。

    Args:
        tensor: n维张量
        a: 均匀分布的下界
        b: 均匀分布的上界

    Returns:
        TN: 填充后的张量

    Examples:
        >>> w = rm.empty(3, 5)
        >>> rm.init.uniform_(w)
    """
    return _no_grad_uniform_(tensor, a, b)


def normal_(tensor: TN, mean: float = 0.0, std: float = 1.0) -> TN:
    r"""用正态分布的值填充输入张量 (Normal Initialization)

    从正态分布 :math:`\mathcal{N}(\text{mean}, \text{std}^2)` 中采样值填充张量。

    Args:
        tensor: n维张量
        mean: 正态分布的均值
        std: 正态分布的标准差

    Returns:
        TN: 填充后的张量

    Examples:
        >>> w = rm.empty(3, 5)
        >>> rm.init.normal_(w)
    """
    return _no_grad_normal_(tensor, mean, std)


def trunc_normal_(
    tensor: TN, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> TN:
    r"""用截断正态分布的值填充输入张量 (Truncated Normal Initialization)

    从截断正态分布 :math:`\mathcal{N}(\text{mean}, \text{std}^2)` 中采样值填充张量，
    超出 :math:`[a, b]` 范围的值会被重新采样直到落在范围内。

    Args:
        tensor: n维张量
        mean: 正态分布的均值
        std: 正态分布的标准差
        a: 最小截断值
        b: 最大截断值

    Returns:
        TN: 填充后的张量

    Examples:
        >>> w = rm.empty(3, 5)
        >>> rm.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def constant_(tensor: TN, val: float) -> TN:
    r"""用常数值填充输入张量 (Constant Initialization)

    Args:
        tensor: n维张量
        val: 用于填充张量的值

    Returns:
        TN: 填充后的张量

    Examples:
        >>> w = rm.empty(3, 5)
        >>> rm.init.constant_(w, 0.3)
    """
    return _no_grad_fill_(tensor, val)


def ones_(tensor: TN) -> TN:
    r"""用标量值1填充输入张量 (Ones Initialization)

    Args:
        tensor: n维张量

    Returns:
        TN: 填充后的张量

    Examples:
        >>> w = rm.empty(3, 5)
        >>> rm.init.ones_(w)
    """
    return _no_grad_fill_(tensor, 1.0)


def zeros_(tensor: TN) -> TN:
    r"""用标量值0填充输入张量 (Zeros Initialization)

    Args:
        tensor: n维张量

    Returns:
        TN: 填充后的张量

    Examples:
        >>> w = rm.empty(3, 5)
        >>> rm.init.zeros_(w)
    """
    return _no_grad_zero_(tensor)


def eye_(tensor: TN) -> TN:
    r"""用单位矩阵填充2维输入张量 (Identity Matrix Initialization)

    在Linear层中保留输入的恒等性，尽可能保留多的输入。

    Args:
        tensor: 2维张量

    Returns:
        TN: 填充后的张量

    Examples:
        >>> w = rm.empty(3, 5)
        >>> rm.init.eye_(w)
    """
    if tensor.ndim != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")
    
    # 直接操作底层数据，与PyTorch的out参数行为一致
    arrlib = np if tensor.device.type == 'cpu' else cp
    tensor.data = arrlib.eye(tensor.shape[0], tensor.shape[1], dtype=tensor.dtype)
    return tensor


def dirac_(tensor: TN, groups: int = 1) -> TN:
    r"""用Dirac delta函数填充{3,4,5}维输入张量 (Dirac Initialization)

    在卷积层中保留输入的恒等性，尽可能保留多的输入通道。
    当groups>1时，每组通道保留恒等性。

    Args:
        tensor: {3,4,5}维张量
        groups: 卷积层中的组数 (默认: 1)

    Returns:
        TN: 填充后的张量

    Examples:
        >>> w = rm.empty(3, 16, 5, 5)
        >>> rm.init.dirac_(w)
        >>> w = rm.empty(3, 24, 5, 5)
        >>> rm.init.dirac_(w, 3)
    """
    dimensions = tensor.ndim
    if dimensions not in [3, 4, 5]:
        raise ValueError("Only tensors with 3, 4, or 5 dimensions are supported")

    sizes = tensor.shape

    if sizes[0] % groups != 0:
        raise ValueError("dim 0 must be divisible by groups")

    out_chans_per_grp = sizes[0] // groups
    min_dim = min(out_chans_per_grp, sizes[1])

    tensor.zero_()

    for g in range(groups):
        for d in range(min_dim):
            if dimensions == 3:  # 时序卷积
                tensor[g * out_chans_per_grp + d, d, tensor.shape[2] // 2] = 1
            elif dimensions == 4:  # 空间卷积
                tensor[
                    g * out_chans_per_grp + d,
                    d,
                    tensor.shape[2] // 2,
                    tensor.shape[3] // 2,
                ] = 1
            else:  # 体积卷积
                tensor[
                    g * out_chans_per_grp + d,
                    d,
                    tensor.shape[2] // 2,
                    tensor.shape[3] // 2,
                    tensor.shape[4] // 2,
                ] = 1
    return tensor


def _calculate_fan_in_and_fan_out(tensor: TN) -> tuple[int, int]:
    """计算张量的fan_in和fan_out (Calculate Fan In and Fan Out)
    
    Args:
        tensor: 输入张量
        
    Returns:
        tuple: (fan_in, fan_out)
    """
    dimensions = tensor.ndim
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if tensor.ndim > 2:
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _calculate_correct_fan(tensor: TN, mode: _FanMode) -> int:
    """计算正确的fan值 (Calculate Correct Fan)
    
    Args:
        tensor: 输入张量
        mode: 'fan_in' 或 'fan_out'
        
    Returns:
        int: fan值
    """
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out"]
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out


def xavier_uniform_(tensor: TN, gain: float = 1.0) -> TN:
    r"""使用Xavier均匀分布填充输入张量 (Xavier Uniform Initialization)

    方法描述于 `Understanding the difficulty of training deep feedforward neural networks`
    - Glorot, X. & Bengio, Y. (2010)。结果张量的值从 :math:`\mathcal{U}(-a, a)` 采样，其中

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}

    也称为Glorot初始化。

    Args:
        tensor: n维张量
        gain: 可选的缩放因子

    Returns:
        TN: 填充后的张量

    Examples:
        >>> w = rm.empty(3, 5)
        >>> rm.init.xavier_uniform_(w, gain=rm.init.calculate_gain("relu"))
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # 从标准差计算均匀分布边界

    return _no_grad_uniform_(tensor, -a, a)


def xavier_normal_(tensor: TN, gain: float = 1.0) -> TN:
    r"""使用Xavier正态分布填充输入张量 (Xavier Normal Initialization)

    方法描述于 `Understanding the difficulty of training deep feedforward neural networks`
    - Glorot, X. & Bengio, Y. (2010)。结果张量的值从 :math:`\mathcal{N}(0, \text{std}^2)` 采样，其中

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}

    也称为Glorot初始化。

    Args:
        tensor: n维张量
        gain: 可选的缩放因子

    Returns:
        TN: 填充后的张量

    Examples:
        >>> w = rm.empty(3, 5)
        >>> rm.init.xavier_normal_(w)
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    return _no_grad_normal_(tensor, 0.0, std)


def kaiming_uniform_(
    tensor: TN,
    a: float = 0,
    mode: _FanMode = "fan_in",
    nonlinearity: _NonlinearityType = "leaky_relu",
) -> TN:
    r"""使用Kaiming均匀分布填充输入张量 (Kaiming Uniform Initialization)

    方法描述于 `Delving deep into rectifiers: Surpassing human-level performance on 
    ImageNet classification` - He, K. et al. (2015)。结果张量的值从 
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` 采样，其中

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    也称为He初始化。

    Args:
        tensor: n维张量
        a: 该层后使用的整流器的负斜率（仅用于'leaky_relu'）
        mode: 'fan_in' (默认) 或 'fan_out'。选择'fan_in'保留前向传播中权重的方差大小，
              选择'fan_out'保留反向传播中的大小。
        nonlinearity: 非线性函数名称，建议仅用于'relu'或'leaky_relu'（默认）。

    Returns:
        TN: 填充后的张量

    Examples:
        >>> w = rm.empty(3, 5)
        >>> rm.init.kaiming_uniform_(w, mode="fan_in", nonlinearity="relu")
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # 从标准差计算均匀分布边界
    
    return _no_grad_uniform_(tensor, -bound, bound)


def kaiming_normal_(
    tensor: TN,
    a: float = 0,
    mode: _FanMode = "fan_in",
    nonlinearity: _NonlinearityType = "leaky_relu",
) -> TN:
    r"""使用Kaiming正态分布填充输入张量 (Kaiming Normal Initialization)

    方法描述于 `Delving deep into rectifiers: Surpassing human-level performance on 
    ImageNet classification` - He, K. et al. (2015)。结果张量的值从 
    :math:`\mathcal{N}(0, \text{std}^2)` 采样，其中

    .. math::
        \text{std} = \frac{\text{gain}}{\sqrt{\text{fan\_mode}}}

    也称为He初始化。

    Args:
        tensor: n维张量
        a: 该层后使用的整流器的负斜率（仅用于'leaky_relu'）
        mode: 'fan_in' (默认) 或 'fan_out'。选择'fan_in'保留前向传播中权重的方差大小，
              选择'fan_out'保留反向传播中的大小。
        nonlinearity: 非线性函数名称，建议仅用于'relu'或'leaky_relu'（默认）。

    Returns:
        TN: 填充后的张量

    Examples:
        >>> w = rm.empty(3, 5)
        >>> rm.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    
    return _no_grad_normal_(tensor, 0.0, std)


def orthogonal_(tensor: TN, gain: float = 1.0) -> TN:
    r"""用（半）正交矩阵填充输入张量 (Orthogonal Initialization)

    描述于 `Exact solutions to the nonlinear dynamics of learning in deep linear 
    neural networks` - Saxe, A. et al. (2013)。输入张量必须至少有2维，
    对于多于2维的张量，尾部维度会被展平。

    Args:
        tensor: n维张量，其中 :math:`n \geq 2`
        gain: 可选的缩放因子

    Returns:
        TN: 填充后的张量

    Examples:
        >>> w = rm.empty(3, 5)
        >>> rm.init.orthogonal_(w)
    """
    if tensor.ndim < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    # 计算总元素数
    total_size = 1
    for s in tensor.shape:
        total_size *= s
    
    if total_size == 0:
        # 无操作
        return tensor
    
    rows = tensor.shape[0]
    cols = int(total_size // rows)
    
    # 创建随机矩阵
    flattened = np.random.normal(0, 1, (rows, cols)).astype(tensor.dtype)
    
    if rows < cols:
        flattened = flattened.T
    
    # 计算QR分解
    q, r = np.linalg.qr(flattened)
    
    # 使Q均匀分布
    d = np.diag(r)
    ph = np.sign(d)
    q = q * ph
    
    if rows < cols:
        q = q.T
    
    # 复制到tensor
    if tensor.device.type == 'cpu':
        tensor.data[:] = q.reshape(tensor.shape)
    else:
        tensor.data[:] = cp.asarray(q.reshape(tensor.shape))
    tensor.mul_(gain)
    return tensor


def sparse_(tensor: TN, sparsity: float, std: float = 0.01) -> TN:
    r"""将2D输入张量填充为稀疏矩阵 (Sparse Initialization)

    非零元素将从正态分布 :math:`\mathcal{N}(0, 0.01)` 中抽取，
    如 `Deep learning via Hessian-free optimization` - Martens, J. (2010) 所述。

    Args:
        tensor: n维张量
        sparsity: 每列中设为0的元素比例
        std: 用于生成非零值的正态分布的标准差

    Returns:
        TN: 填充后的张量

    Examples:
        >>> w = rm.empty(3, 5)
        >>> rm.init.sparse_(w, sparsity=0.1)
    """
    if tensor.ndim != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    rows, cols = tensor.shape
    num_zeros = int(math.ceil(sparsity * rows))

    _no_grad_normal_(tensor, 0, std)
    arrlib = _get_array_lib(tensor)
    for col_idx in range(cols):
        row_indices = list(range(rows))
        import random
        random.shuffle(row_indices)
        zero_indices = row_indices[:num_zeros]
        for idx in zero_indices:
            tensor[idx, col_idx] = 0
    return tensor
