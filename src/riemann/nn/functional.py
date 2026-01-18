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

# Copyright (c) 2024 Riemann Contributors
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Neural Network Functional Module for the Riemann Library

This module implements essential building blocks for neural networks in the Riemann library,
including various activation functions, loss functions, linear transformations, convolution operations,
and pooling functions. These functions serve as the foundation for the higher-level modules 
defined in activation.py, loss.py, conv.py, and pool.py.

Implemented Functions:

Linear Functions:
- linear: Linear transformation y = xA^T + b (compatible with torch.nn.functional.linear)

Activation Functions:
- sigmoid: Element-wise sigmoid function
- silu: Sigmoid Linear Unit (Swish) activation function
- tanh: Hyperbolic tangent activation function
- softmax: Softmax function along specified dimension
- log_softmax: Logarithm of softmax function for numerical stability
- relu: Rectified Linear Unit activation
- leaky_relu: Leaky ReLU with learnable negative slope
- prelu: Parametric ReLU with learnable parameters
- rrelu: Randomized ReLU with learnable bounds
- gelu: Gaussian Error Linear Unit
- softplus: Smooth approximation to ReLU

Convolution Functions:
- conv1d: 1D convolution operation (compatible with torch.nn.functional.conv1d)
- conv2d: 2D convolution operation (compatible with torch.nn.functional.conv2d)
- conv3d: 3D convolution operation (compatible with torch.nn.functional.conv3d)

Pooling Functions:
- max_pool1d: 1D max pooling operation (compatible with torch.nn.functional.max_pool1d)
- max_pool2d: 2D max pooling operation (compatible with torch.nn.functional.max_pool2d)
- max_pool3d: 3D max pooling operation (compatible with torch.nn.functional.max_pool3d)
- avg_pool1d: 1D average pooling operation (compatible with torch.nn.functional.avg_pool1d)
- avg_pool2d: 2D average pooling operation (compatible with torch.nn.functional.avg_pool2d)
- avg_pool3d: 3D average pooling operation (compatible with torch.nn.functional.avg_pool3d)

Loss Functions:
- mse_loss: Mean Squared Error loss
- l1_loss: L1 (absolute error) loss
- smooth_l1_loss: Smooth L1 loss
- cross_entropy: Cross entropy loss with optional label smoothing
- binary_cross_entropy_with_logits: Binary cross entropy with logits
- huber_loss: Huber loss
- nll_loss: Negative log likelihood loss

Utility Functions:
- one_hot: Convert class indices to one-hot encoded tensors
- _get_reduction: Helper function to handle reduction parameters

All functions in this module are designed to work with the Riemann Tensor (TN) type and
support automatic differentiation through properly implemented backward functions.
"""

import numpy as np
from typing import Optional
from ..tensordef import *

def linear(input: TN, weight: TN, bias: Optional[TN] = None) -> TN:
    """
    应用线性变换：y = xA^T + b
    
    接口与torch.nn.functional.linear完全一致
    
    参数:
        input (TN): 输入张量，形状为 (*, in_features)
        weight (TN): 权重张量，形状为 (out_features, in_features)
        bias (TN, optional): 偏置张量，形状为 (out_features,)。默认为None
    
    返回:
        TN: 输出张量，形状为 (*, out_features)
    
    数学公式:
        output = input @ weight.T + bias
        
    其中：
        - input: 输入张量，形状为 (*, in_features)
        - weight: 权重矩阵，形状为 (out_features, in_features)
        - bias: 偏置向量，形状为 (out_features,)
        - output: 输出张量，形状为 (*, out_features)
    
    示例:
        >>> input = tensor([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
        >>> weight = tensor([[1.0, 0.0], [0.0, 1.0]])  # (2, 2)
        >>> bias = tensor([1.0, 2.0])  # (2,)
        >>> output = linear(input, weight, bias)
        >>> # 结果: [[2.0, 4.0], [4.0, 6.0]]
    
    注意:
        - 支持任意维度的输入张量，只要最后一个维度是in_features
        - 如果bias为None，则不添加偏置
        - 自动支持梯度计算
    """
    if not isinstance(input, TN):
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input)}")
    if not isinstance(weight, TN):
        raise TypeError(f"Expected weight type to be TN tensor, but received type: {type(weight)}")
    if bias is not None and not isinstance(bias, TN):
        raise TypeError(f"Expected bias type to be TN tensor or None, but received type: {type(bias)}")
    
    if input.device != weight.device or (bias is not None and bias.device != weight.device):
        raise ValueError("input, weight, and bias must have the same device")
    
    # 检查维度兼容性
    if input.ndim < 1:
        raise ValueError("input must have at least 1 dimension")
    if weight.ndim != 2:
        raise ValueError("weight must be 2-dimensional")
    if bias is not None and bias.ndim != 1:
        raise ValueError("bias must be 1-dimensional")
    
    # 检查特征维度匹配
    in_features = input.shape[-1]
    weight_in_features = weight.shape[1]
    out_features = weight.shape[0]
    
    if in_features != weight_in_features:
        raise ValueError(f"input.shape[-1] ({in_features}) must equal weight.shape[1] ({weight_in_features})")
    
    if bias is not None and bias.shape[0] != out_features:
        raise ValueError(f"bias.shape[0] ({bias.shape[0]}) must equal weight.shape[0] ({out_features})")
    
    # 执行线性变换：input @ weight.T
    output = input @ weight.mT
    
    # 添加偏置（如果提供）
    if bias is not None:
        # 广播bias到output的形状
        # bias需要广播到output的所有维度，除了最后一个特征维度
        output = output + bias
    
    return output


def sigmoid(x: TN) -> TN:
    """前向传播：1.0 / (1.0 + exp(-x))"""
    if not isinstance(x, TN): 
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(x)}")

    x_data = x.data
    # 对每个元素单独处理，提高数值稳定性
    # 对于较大的正值，直接返回1.0以避免exp溢出
    # 对于较小的负值，直接返回0.0以避免exp溢出
    # 对于中间值，使用标准公式
    arrlib = x._get_array_lib()
    data = arrlib.where(x_data > 20, 1.0, 
                   arrlib.where(x_data < -20, 0.0, 
                           1. / (1. + arrlib.exp(-x_data))))
    
    ret = tensor(data, device=x.device, requires_grad=x.requires_grad)
    ret.is_leaf = not ret.requires_grad
    
    if ret.requires_grad:
        ret.fromvars = (x,)
        ret.gradfuncs = (_sigmoid_backward,)
    return ret

def _sigmoid_backward(result_tensor: TN, i: int) -> TN:
    """梯度计算：sigmoid(x) * (1. - sigmoid(x))"""
    return result_tensor.grad_value * result_tensor * (1.0 - result_tensor)

def silu(input: TN) -> TN:
    """
    Sigmoid Linear Unit (SiLU) 激活函数，也称为Swish
    
    参数:
        input (TN): 输入张量
    
    返回:
        TN: SiLU激活后的张量
    
    数学公式:
        silu(x) = x * sigmoid(x)
        
    其中：
        - input: 输入张量
        - sigmoid: sigmoid函数，1/(1 + exp(-x))
        - output: 输出张量，形状与输入相同
    
    示例:
        >>> input = tensor([-2.0, 0.0, 2.0])
        >>> output = silu(input)
        >>> # 结果约: [-0.2384, 0.0, 1.7616]
    
    注意:
        - SiLU是平滑的非单调激活函数
        - 相比ReLU，在某些任务上表现更好
        - 利用已有的sigmoid函数，自动支持梯度计算
        - 与PyTorch的F.silu行为完全一致
    """
    if not isinstance(input, TN):
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input)}")
    
    # SiLU公式: x * sigmoid(x)
    # 利用已有的sigmoid函数，自动支持梯度计算
    sigmoid_output = sigmoid(input)
    return input * sigmoid_output

# softmax函数定义
def softmax(x:TN, dim:int)->TN:
    """计算输入张量的softmax函数"""
    if not isinstance(x, TN): 
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(x)}")

    # 确保dim是整数类型
    if not isinstance(dim, int):
        raise ValueError(f"dim必须是整数类型，当前类型: {type(dim)}")
    # 当dim是整数时，max函数应返回带有values属性的对象
    max_result = max(x, dim=dim, keepdim=True)
    x_max = x - max_result.values
    ex = exp(x_max)
    return ex / sum(ex, dim=dim, keepdim=True)

# 修改log_softmax函数，添加显式类型检查
def log_softmax(x: TN, dim: int = -1) -> TN:
    """数值稳定的log(softmax)实现，使用log-sum-exp技巧"""
    if not isinstance(x, TN): 
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(x)}")

    # 确保dim是整数类型
    if not isinstance(dim, int):
        raise ValueError(f"dim必须是整数类型，当前类型: {type(dim)}")
    
    # 当dim是整数时，max函数应返回带有values属性的对象
    _max = max(x, dim=dim, keepdim=True).values
    x_max = x - _max

    # 计算指数并求和
    exp_x = exp(x_max)
    sum_exp_x = sum(exp_x, dim=dim, keepdim=True)
    # 应用log-sum-exp公式
    return x_max - log(sum_exp_x)

def relu(x: TN) -> TN:
    """前向传播：max(0, x)"""
    if not isinstance(x, TN): 
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(x)}")
    
    # 前向计算：使用np.maximum实现条件选择
    # 对于输入为负值，直接返回0.0以避免数值不稳定性
    arrlib = x._get_array_lib()
    data = arrlib.maximum(0, x.data)
    ret = tensor(data, device=x.device, requires_grad=x.requires_grad)
    ret.is_leaf = not ret.requires_grad
    
    # 注册梯度函数
    if ret.requires_grad:
        ret.fromvars = (x,)
        ret.gradfuncs = (_relu_backward,)
    return ret

def _relu_backward(result_tensor: TN, i: int) -> TN:
    """梯度计算：输入>0时梯度为1，否则为0"""
    x = result_tensor.fromvars[0]
    mask = (x > 0).type(x.dtype)
    return result_tensor.grad_value * mask

def leaky_relu(x: TN, alpha: float = 0.01) -> TN:
    # 前向计算：使用np.where实现条件选择
    # 对于输入为负值，返回alpha倍的输入值
    arrlib = x._get_array_lib()
    data = arrlib.where(x.data > 0, x.data, alpha * x.data)
    ret = tensor(data, device=x.device, requires_grad=x.requires_grad)
    ret.is_leaf = not ret.requires_grad
    
    if ret.requires_grad:
        ret.fromvars = (x,)
        ret.parms = (alpha,)  # 保存alpha值供反向传播使用
        ret.gradfuncs = (_leaky_relu_backward,)
    return ret

def _leaky_relu_backward(result_tensor: TN, i: int) -> TN:
    alpha = result_tensor.parms[0]
    x = result_tensor.fromvars[0]
    # 计算梯度：x>0时梯度为1，否则为alpha
    mask = (x > 0).type(x.dtype)
    grad = result_tensor.grad_value * (mask + (1.0 - mask) * alpha)
    return grad

def prelu(x: TN, alpha: TN) -> TN:
    if not isinstance(x, TN): 
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(x)}")

    # 前向计算：alpha为可训练参数，用于控制负值的缩放
    arrlib = x._get_array_lib()
    data = arrlib.where(x.data > 0, x.data, alpha.data * x.data)
    ret = tensor(data, device=x.device, requires_grad=(x.requires_grad or alpha.requires_grad))
    ret.is_leaf = not ret.requires_grad
    
    if ret.requires_grad:
        ret.fromvars = (x, alpha)
        # 梯度函数分别处理x和alpha的梯度
        ret.gradfuncs = (_prelu_grad_x, _prelu_grad_alpha)
    return ret

def _prelu_grad_x(result_tensor: TN, i: int) -> TN:
    x, alpha = result_tensor.fromvars
    mask = (x > 0).type(x.dtype)
    grad_x = result_tensor.grad_value * (mask + (1.0 - mask) * alpha.data)
    return grad_x

def _prelu_grad_alpha(result_tensor: TN, i: int) -> TN:
    x, alpha = result_tensor.fromvars
    mask = (x <= 0)
    # 对alpha的梯度需要求和（假设alpha为标量或向量）
    grad_alpha = sum(result_tensor.grad_value * mask * x)
    return grad_alpha

def rrelu(x: TN, lower: float = 1.0/8.0, upper: float = 1.0/3.0, training: bool = True) -> TN:
    if not isinstance(x, TN): 
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(x)}")

    # 训练时随机生成alpha，测试时用平均值作为alpha
    arrlib = x._get_array_lib()
    if training:
        alpha = arrlib.random.uniform(low=lower, high=upper, size=x.data.shape)
    else:
        alpha = (lower + upper) / 2.0 * arrlib.ones_like(x.data)
    alpha = alpha.astype(x.dtype)

    # 前向计算
    data = arrlib.where(x.data > 0, x.data, alpha * x.data)
    ret = tensor(data, device=x.device, requires_grad=x.requires_grad)
    ret.is_leaf = not ret.requires_grad
    
    if x.requires_grad:
        ret.fromvars = (x,)
        ret.parms = (training, lower, upper, alpha)
        ret.gradfuncs = (_rrelu_grad_x,)
    return ret

def _rrelu_grad_x(result_tensor: TN, i: int) -> TN:
    x = result_tensor.fromvars[0]
    alpha = result_tensor.parms[3]
    mask = (x > 0).type(x.dtype)
    grad = result_tensor.grad_value * (mask + (1.0 - mask) * alpha)
    return grad

# 修改gelu函数实现，使用PyTorch兼容的精确版本
def gelu(x: TN) -> TN:
    """高斯误差线性单元，使用与PyTorch完全一致的实现"""
    if not isinstance(x, TN): 
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(x)}")

    # 使用PyTorch使用的GELU公式（Hendrycks & Gimpel 2016）
    # 精确的GELU公式: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    arrlib = x._get_array_lib()
    c = arrlib.sqrt(2.0 / arrlib.pi, dtype=x.dtype)
    x_data = x.data
    data = 0.5 * x_data * (1.0 + arrlib.tanh(c * (x_data + 0.044715 * x_data**3.0)))
    ret = tensor(data, device=x.device, requires_grad=x.requires_grad)
    ret.is_leaf = not ret.requires_grad
    
    if ret.requires_grad:
        ret.fromvars = (x,)
        ret.gradfuncs = (_gelu_backward,)
    return ret

def _gelu_backward(result_tensor: TN, i: int) -> TN:
    """GELU导数计算，使用与PyTorch兼容的实现"""
    x_data = result_tensor.fromvars[0].data

    c = np.sqrt(2. / np.pi, dtype=x_data.dtype)
    tanh_term = np.tanh(c * (x_data + 0.044715 * x_data**3))
    grad = 0.5 * (1. + tanh_term) + 0.5 * x_data * (1. - tanh_term**2.) * c * (1. + 0.134145*x_data**2.)
    return result_tensor.grad_value * grad

def softplus(x: TN, beta: float = 1.0, threshold: float = 20.0) -> TN:
    """
    带阈值切换的稳定softplus实现[12,13](@ref)
    特性：
    1. beta参数控制曲线陡峭度
    2. 输入超过阈值时切换为线性计算
    """

    if not isinstance(x, TN): 
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(x)}")

    scaled_x = x * beta
    
    # 分段计算避免大值溢出
    # 修复softplus函数中的where()调用
    
    return where(
        scaled_x > threshold,  # 条件（位置参数）
        x,  # 满足条件时的值（位置参数）
        log(1. + exp(scaled_x)) / beta  # 不满足条件时的值（位置参数）
    )

# 修改nll_loss函数，使其在non-reduction模式下与PyTorch保持一致
# PyTorch在non-reduction模式下保留所有样本位置，只是将被忽略样本的损失值设为0
def nll_loss(input: TN, target: TN, weight: Optional[TN] = None, 
             ignore_index: int = -100, reduction: str = 'mean') -> TN:
    if not isinstance(input, TN): 
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input)}")

    # 维度校验
    if input.ndim < 2:
        raise ValueError("Input must be at least 2-dimensional")
    if input.shape[0] != target.shape[0]:
        raise ValueError(f"Expected input batch_size ({input.shape[0]}) to match target batch_size ({target.shape[0]})")

    original_shape = input.shape
    original_target_shape = target.shape
    
    # 高维输入处理（对齐PyTorch维度逻辑）
    if input.ndim > 2:
        # 保存原始形状，用于non-reduction模式下恢复
        # input_flat_shape = (-1, input.shape[1])
        # target_flat_shape = (-1,)
        
        input = input.reshape(input.shape[0], input.shape[1], -1)  # (N, C, D)
        input = input.permute(0, 2, 1).reshape(-1, input.shape[1])  # (N*D, C) → 分类维度为dim=1
        target = target.reshape(-1).type(int64)
    
    # 创建忽略索引掩码
    mask = (target != ignore_index)
    # valid_samples = sum(mask.type(int32))  # 有效样本数
    valid_samples = sum(mask)  # 有效样本数
    
    # 处理边界情况：没有有效样本时
    if valid_samples == 0:
        if reduction == 'none':
            # 返回与输入形状匹配的零张量
            if input.ndim > 2:
                # return tensor(np.zeros(original_target_shape, dtype=np.float32))
                return zeros_like(target)
            else:
                # return tensor(np.zeros(input.shape[0], dtype=np.float32))
                return zeros(input.shape[0],input.dtype,device=input.device)
        elif reduction == 'sum':
            return tensor(0.0, dtype=input.dtype,device=input.device)
        elif reduction == 'mean':
            # PyTorch在这种情况下返回nan（除以零）
            return tensor(float('nan'),dtype=input.dtype,device=input.device)
    
    # 核心计算 - 计算所有样本的损失
    # 对于被忽略的样本，target值可能超出范围，所以我们需要使用临时target
    temp_target = where(mask, target, zeros_like(target))
    target_expanded = temp_target.unsqueeze(1)
    
    selected_log_probs = input.gather(1, target_expanded).squeeze(1)
    loss = -selected_log_probs
    
    # 权重处理
    weight_sum = None
    if weight is not None:
        if weight.ndim != 1:
            raise ValueError('weight must be 1D tensor')
        if weight.shape[0] != original_shape[1]:
            raise ValueError(f'weight size ({weight.shape}) must match number of classes ({original_shape[1]})')
        
        # 获取对应类别的权重
        class_weights = weight.gather(0, temp_target)
        loss = loss * class_weights
        # weight_sum = sum(class_weights * mask.type(float32))
        weight_sum = sum(class_weights * mask)
    
    # 应用ignore_index掩码 - 将被忽略样本的损失设为0
    loss = where(mask, loss, zeros_like(loss))
    
    # 对于non-reduction模式，恢复原始形状（如果需要）
    if reduction == 'none' and input.ndim > 2:
        # 恢复为与target原始形状匹配的损失张量
        loss = loss.reshape(original_target_shape)
        return loss
    elif reduction == 'none':
        return loss
    
    # 对于其他reduction模式，只考虑有效样本
    valid_loss = loss[mask]
    
    if reduction == 'sum':
        return valid_loss.sum()
    elif reduction == 'mean':
        if weight is None:
            # 没有权重时，直接除以有效样本数
            # return valid_loss.sum() / valid_loss.numel()
            return valid_loss.mean()
        else:
            # 有权重时，使用权重和进行归一化
            return valid_loss.sum() / weight_sum            
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
# endof nll_loss


def one_hot(target: TN, num_classes: int) -> TN:
    """
    将类别索引转换为one-hot编码
    参数：
        target: (N, *) 整数张量，元素值在[0, num_classes-1]范围内
        num_classes: 类别总数
    返回：
        (N, *, num_classes) 的one-hot编码张量，与PyTorch一致
    """
    if not isinstance(target, TN): 
        raise TypeError(f"Expected target type to be TN tensor, but received type: {type(target)}")

    # 检查负索引
    if (target < 0).any():  # 检查是否有任何元素小于0
        raise RuntimeError("Class values must be non-negative.")

    # 创建结果形状 (N, *, num_classes)
    result_shape = target.shape + (num_classes,)
    
    # 创建全零张量
    result = zeros(result_shape,device=target.device)
    
    # 获取最后一个维度作为scatter的维度
    dim = -1
    
    # 使用scatter_在指定位置设置1.0
    # target扩展维度以匹配result的维度数
    index = target.unsqueeze(dim)
    result.scatter_(dim, index, 1.0)
    
    return result

# 辅助函数：处理旧版参数
def _get_reduction(size_average, reduce):
    """
    处理PyTorch中旧版的size_average和reduce参数
    """
    if reduce is None:
        reduce = True
    
    if reduce:
        if size_average is None:
            return 'mean'
        else:
            return 'mean' if size_average else 'sum'
    else:
        return 'none'

# 新增损失函数实现
def mse_loss(input: TN, target: TN, size_average=None, reduce=None, reduction: str = 'mean') -> TN:
    """
    计算均方误差损失
    接口与torch.nn.functional.mse_loss一致
    """
    if not isinstance(input, TN): 
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input)}")
    if not isinstance(target, TN): 
        raise TypeError(f"Expected target type to be TN tensor, but received type: {type(target)}")
    if input.device != target.device:
        raise ValueError("Input and target must be on the same device")
    # 处理旧版参数
    if size_average is not None or reduce is not None:
        reduction = _get_reduction(size_average, reduce)
    
    # 计算平方差
    diff = input - target
    square_diff = diff ** 2.0
    
    # 根据reduction参数聚合结果
    if reduction == 'none':
        return square_diff
    elif reduction == 'sum':
        return square_diff.sum()
    elif reduction == 'mean':
        return square_diff.mean()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

def l1_loss(input: TN, target: TN, size_average=None, reduce=None, reduction: str = 'mean') -> TN:
    """
    计算L1损失（绝对误差）
    接口与torch.nn.functional.l1_loss一致
    """
    if not isinstance(input, TN): 
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input)}")
    if not isinstance(target, TN): 
        raise TypeError(f"Expected target type to be TN tensor, but received type: {type(target)}")
    if input.device != target.device:
        raise ValueError("Input and target must be on the same device")
    # 处理旧版参数
    if size_average is not None or reduce is not None:
        reduction = _get_reduction(size_average, reduce)
    
    # 计算绝对差
    abs_diff = abs(input - target)
    
    # 根据reduction参数聚合结果
    if reduction == 'none':
        return abs_diff
    elif reduction == 'sum':
        return abs_diff.sum()
    elif reduction == 'mean':
        return abs_diff.mean()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

# 修改smooth_l1_loss函数，使用正确的方式创建beta常量张量
def smooth_l1_loss(input: TN, target: TN, size_average=None, reduce=None, reduction: str = 'mean', beta: float = 1.0) -> TN:
    """
    计算平滑L1损失
    接口与torch.nn.functional.smooth_l1_loss一致
    """
    if not isinstance(input, TN): 
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input)}")
    if not isinstance(target, TN): 
        raise TypeError(f"Expected target type to be TN tensor, but received type: {type(target)}")
    if input.device != target.device:
        raise ValueError("Input and target must be on the same device")
    # 处理旧版参数
    if size_average is not None or reduce is not None:
        reduction = _get_reduction(size_average, reduce)
    
    # 计算绝对差
    diff = abs(input - target)
    
    # 分段计算：diff <= beta时使用0.5*dif^2f/beta，否则使用diff - 0.5*beta
    condition = diff < beta
    loss = where(condition, 
                 0.5 * diff ** 2.0 / beta,
                 diff - 0.5 * beta)

    # 根据reduction参数聚合结果
    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

def cross_entropy(input: TN, target: TN, weight: Optional[TN] = None,
                 size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean',
                 label_smoothing: float = 0.0) -> TN:
    """
    计算交叉熵损失
    接口与torch.nn.functional.cross_entropy一致
    """
    if not isinstance(input, TN): 
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input)}")
    if not isinstance(target, TN): 
        raise TypeError(f"Expected target type to be TN tensor, but received type: {type(target)}")
    if input.device != target.device:
        raise ValueError("Input and target must be on the same device")
    # 处理旧版参数
    if size_average is not None or reduce is not None:
        reduction = _get_reduction(size_average, reduce)
    
    # 创建有效样本掩码
    mask = (target != ignore_index)
    
    # 如果不需要标签平滑，直接使用nll_loss
    if label_smoothing == 0:
        return nll_loss(
            log_softmax(input, dim=1), 
            target, 
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction
        )
    
    # 标签平滑处理
    num_classes = input.shape[1]
    
    # 创建安全的临时target，确保所有值都在有效范围内
    temp_target = where(mask, target, zeros_like(target))
    
    # 计算log_softmax
    log_probs = log_softmax(input, dim=1)
    
    # 创建平滑目标分布 - 严格按照PyTorch实现
    # 关键点1: 创建one-hot编码
    target_onehot = zeros_like(input)
    target_onehot = target_onehot.scatter(1, temp_target.unsqueeze(1), 1.0)

    # 关键点2: 应用标签平滑
    epsilon = label_smoothing
    smooth_target = target_onehot * (1.0 - epsilon) + epsilon / num_classes
    
    # 关键点3: 如果有权重，需要特殊处理
    if weight is not None:
        # 权重需要应用到每个类别的贡献上，而不仅仅是最终损失
        # 这是与之前实现的关键区别
        smooth_target = smooth_target * weight.view(1, -1)
    
    # 计算交叉熵损失
    loss = -sum(smooth_target * log_probs, dim=1)
    
    # 应用ignore_index掩码
    loss = where(mask, loss, zeros_like(loss))
    
    # 应用reduction参数
    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        # 计算有效样本数
        valid_count = sum(mask.type(input.dtype))
        if valid_count > 0:
            if weight is not None:
                # 关键点4: 对于带权重的mean reduction，分母是有效样本的权重总和
                # 但这里我们已经在smooth_target中应用了权重，所以需要重新计算有效权重
                target_weights = weight.gather(0, temp_target.view(-1)).view(loss.shape)
                weight_sum = sum(target_weights * mask.type(input.dtype))
                if weight_sum > 0:
                    return loss.sum() / weight_sum
                else:
                    return zeros_like(loss.sum())
            else:
                return loss.sum() / valid_count
        else:
            return zeros_like(loss.sum())
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

def binary_cross_entropy_with_logits(input: TN, target: TN, weight: Optional[TN] = None, 
                                    size_average=None, reduce=None, 
                                    reduction: str = 'mean', 
                                    pos_weight: Optional[TN] = None) -> TN:
    """
    计算带logits的二分类交叉熵损失
    接口与torch.nn.functional.binary_cross_entropy_with_logits一致
    """
    if not isinstance(input, TN): 
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input)}")
    if not isinstance(target, TN): 
        raise TypeError(f"Expected target type to be TN tensor, but received type: {type(target)}")

    # 处理旧版参数
    if size_average is not None or reduce is not None:
        reduction = _get_reduction(size_average, reduce)
    
    # 数值稳定性处理：避免log(0)问题
    max_val = where(-input > 0, -input, 0.0)
    
    # 稳定的交叉熵计算：input - input * target + log(1 + exp(-input))
    loss = input - input * target + max_val + log(1.0 + exp(-abs(input)))
    
    # 应用正类权重
    if pos_weight is not None:
        # pos_weight * target * (-log(sigmoid(input))) + (1 - target) * (-log(1 - sigmoid(input)))
        # 重新计算以应用pos_weight
        log_weight = pos_weight
        loss_pos = log_weight * (-log(sigmoid(input)))
        loss_neg = -log(1.0 - sigmoid(input))
        loss = target * loss_pos + (1.0 - target) * loss_neg
    
    # 应用样本权重
    if weight is not None:
        loss = loss * weight
    
    # 根据reduction参数聚合结果
    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        # 修正：PyTorch的行为是简单地对加权损失求平均值，而不是除以权重总和
        return loss.mean()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
    
def huber_loss(input: TN, target: TN, delta: float = 1.0, 
               size_average=None, reduce=None, reduction: str = 'mean') -> TN:
    """
    计算Huber损失
    接口与torch.nn.functional.huber_loss一致
    """
    if not isinstance(input, TN): 
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input)}")
    if not isinstance(target, TN): 
        raise TypeError(f"Expected target type to be TN tensor, but received type: {type(target)}")
    if input.device != target.device:
        raise ValueError("Input and target must be on the same device")
    
    # 处理旧版参数
    if size_average is not None or reduce is not None:
        reduction = _get_reduction(size_average, reduce)
    
    # 计算绝对差
    diff = abs(input - target)
    
    # 分段计算：diff <= delta时使用0.5*diff^2，否则使用delta*(diff - 0.5*delta)
    loss = where(diff <= delta, 0.5 * diff ** 2.0, delta * (diff - 0.5 * delta))
    
    # 根据reduction参数聚合结果
    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

def _single(x):
    '''
    确保输入参数是一个整数。
    如果输入是一个整数，直接返回它。
    如果输入是一个长度为1的元组，返回元组中的唯一元素。
    否则，抛出 ValueError 异常。
    '''
    if isinstance(x, int):
        return x
    elif isinstance(x, tuple) and len(x) == 1:
        return x[0]
    else:
        raise ValueError(f"Expected int or tuple of length 1, but got {type(x)}")


def _pair(x):
    '''
    确保输入参数是一个2元组。
    如果输入是一个整数，返回一个2元组，其中的元素都是该整数。
    如果输入是一个2元组，直接返回它。
    否则，抛出 ValueError 异常。
    '''
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple) and len(x) == 2:
        return x
    else:
        raise ValueError(f"Expected int or tuple of length 2, but got {type(x)}")

def _triple(x):
    '''
    确保输入参数是一个3元组。
    如果输入是一个整数，返回一个3元组，其中的元素都是该整数。
    如果输入是一个3元组，直接返回它。
    否则，抛出 ValueError 异常。
    '''
    if isinstance(x, int):
        return (x, x, x)
    elif isinstance(x, tuple) and len(x) == 3:
        return x
    else:
        raise ValueError(f"Expected int or tuple of length 3, but got {type(x)}")

def _get_slide_win_count_and_size(input_size, kernel_size, dilation, padding, stride, ceil_mode):
    '''
    计算单个维度滑动窗口的数量和有效大小
    '''
    # 计算有效kernel大小 - 与im2col一致
    effective_K = dilation * (kernel_size - 1) + 1
    
    # 计算输出维度 - 与im2col逻辑一致
    numerator = input_size + 2 * padding - effective_K
    
    if not ceil_mode:
        # ceil_mode=False时的正确公式
        L_out = (numerator + stride) // stride if numerator >= 0 else 1
    else:
        # ceil_mode=True时的计算 - 与im2col一致
        # 先用地板除法计算基础输出尺寸
        L_out = (numerator + stride - 1) // stride if numerator >= 0 else 1
        
        # 检查是否需要再增加一个窗口
        # 如果再移动一个步长后仍包含原始图像元素，则增加输出尺寸
        next_l_start = L_out * stride
        next_orig_l_start = next_l_start - padding
        next_orig_l_end = next_l_start + effective_K - padding
        next_contains_orig = next_orig_l_start < input_size and next_orig_l_end > 0
        
        # 如果下一个窗口包含原始元素，则增加输出尺寸
        if next_contains_orig:
            L_out += 1
    
    # 确保输出尺寸至少为1
    L_out = L_out if L_out > 1 else 1
    return L_out, effective_K

def unfold(input: TN, kernel_size, dilation=1, padding=0, stride=1) -> TN:
    r"""从批处理输入张量中提取滑动局部块。

    .. warning::
        目前仅支持4D输入张量（批处理图像类张量）。

    .. warning::

        展开后的张量中可能有多个元素引用单个内存位置。因此，原地操作（尤其是向量化的操作）可能会导致不正确的行为。
        如果需要写入张量，请先克隆它。

    参数:
        input (TN): 输入张量，形状为 (N, C, H, W)
        kernel_size (int 或 tuple): 滑动块的大小
        dilation (int 或 tuple, 可选): 内核点之间的间距。默认值: 1
        padding (int 或 tuple, 可选): 两侧的隐式零填充。默认值: 0
        stride (int 或 tuple, 可选): 滑动块的步幅。默认值: 1

    返回:
        TN: 展开后的张量，形状为 (N, C * kernel_size[0] * kernel_size[1], L)
        其中 L 是滑动块的总数。
    """
    if not isinstance(input, TN):
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input)}")
    
    if input.ndim != 4:
        raise ValueError(f"Expected 4D tensor as input, but got {input.ndim}D tensor")
        
    kernel_size = _pair(kernel_size)
    padding = _pair(padding)
    stride = _pair(stride)
    dilation = _pair(dilation)
    # 检查膨胀率是否大于0
    if dilation[0] <= 0 or dilation[1] <= 0:
        raise RuntimeError(f"dilation should be greater than zero, but got dilation_height: {dilation[0]} dilation_width: {dilation[1]}")
        
    # 计算输出维度
    N, C, H, W = input.shape
    kh, kw = kernel_size

    H_out = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    W_out = (W + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
        
    # 创建展开张量
    # 首先，根据需要对输入进行填充
    if padding != (0, 0):
        # 使用riemann张量函数创建带零填充的张量，保持计算图
        # 计算填充后的形状
        padded_shape = (N, C, H + 2 * padding[0], W + 2 * padding[1])
        # 直接创建形状为padded_shape的零张量
        padded_input = zeros(padded_shape, dtype=input.dtype)

        # 使用索引赋值将原始数据复制到填充后的张量中
        # padded_input[:, :, padding[0]:padding[0]+H, padding[1]:padding[1]+W] = input
        index = (slice(None), slice(None), slice(padding[0],padding[0]+H), slice(padding[1],padding[1]+W))
        padded_input.setat_(index,input)
    else:
        padded_input = input
    
    # 完全向量化版本：展开输入，避免双重循环
    arrlib = input._get_array_lib()
    h_starts = arrlib.arange(H_out) * stride[0]
    w_starts = arrlib.arange(W_out) * stride[1]
    h_k_range = arrlib.arange(kernel_size[0]) * dilation[0]
    w_k_range = arrlib.arange(kernel_size[1]) * dilation[1]
    
    # 生成所有内核位置的坐标偏移 (kh, kw)
    kh_indices, kw_indices = arrlib.meshgrid(h_k_range, w_k_range, indexing='ij')
    
    # 生成所有输出位置的坐标网格 (H_out, W_out)
    h_indices, w_indices = arrlib.meshgrid(h_starts, w_starts, indexing='ij')
    
    # 计算所有需要提取的高度和宽度索引，形状为 (kh, kw, H_out, W_out)
    all_h_indices = h_indices[arrlib.newaxis, arrlib.newaxis, :, :] + kh_indices[:, :, arrlib.newaxis, arrlib.newaxis]
    all_w_indices = w_indices[arrlib.newaxis, arrlib.newaxis, :, :] + kw_indices[:, :, arrlib.newaxis, arrlib.newaxis]
    
    # 一次性提取所有展开块，避免双重循环，直接得到形状 (N, C, kh, kw, H_out, W_out)
    unfolded_blocks = padded_input[:, :, all_h_indices, all_w_indices]
    
    # 直接返回池化函数所需的形状 (N, C*kh*kw, H_out*W_out)
    # 只需要一次reshape即可
    unfolded_result = unfolded_blocks.reshape(N, C*kh*kw, H_out*W_out)
    
    return unfolded_result

def unfold2d(input: TN, kernel_size, dilation=1, padding=0, padvalue=0.0, stride=None, ceil_mode=False, check_pad=True) -> tuple[TN, int, int]:
    r"""将输入张量从(N, C, H, W)转换为(N, C, kernel_size[0], kernel_size[1], H_out, W_out)的格式。

    .. warning::
        目前仅支持4D输入张量（批处理图像类张量）。

    参数:
        input (TN): 输入张量，形状为 (N, C, H, W)
        kernel_size (tuple): 滑动块的大小
        dilation (tuple, 可选): 内核点之间的间距。默认值: 1
        padding (tuple, 可选): 两侧的隐式零填充。默认值: 0
        stride (tuple, 可选): 滑动块的步幅。默认值: 1
        ceil_mode (bool, 可选): 是否使用向上取整计算输出尺寸。默认值: False

    返回:
        TN: 转换后的张量，形状为 (N, C, kernel_size[0], kernel_size[1], H_out, W_out)
        其中 H_out, W_out 是输出张量的高度和宽度。
    """
    if not isinstance(input, TN):
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input)}")
    
    if input.ndim != 4:
        raise ValueError(f"Expected 4D tensor as input, but got {input.ndim}D tensor")
     
    # 计算输出维度 - 与PyTorch实际行为一致
    N, C, H_in, W_in = input.shape
    kernel_size = _pair(kernel_size)
    padding = _pair(padding)
    stride = _pair(stride) if stride is not None else kernel_size
    dilation = _pair(dilation)
    # 检查膨胀率是否大于0
    if dilation[0] <= 0 or dilation[1] <= 0:
        raise RuntimeError(f"dilation should be greater than zero, but got dilation_height: {dilation[0]} dilation_width: {dilation[1]}")
    
    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    H_out, effective_kh = _get_slide_win_count_and_size(H_in, 
                                                       kh, 
                                                       dh, 
                                                       ph, 
                                                       sh, 
                                                       ceil_mode)
    W_out, effective_kw = _get_slide_win_count_and_size(W_in, 
                                                       kw, 
                                                       dw, 
                                                       pw, 
                                                       sw, 
                                                       ceil_mode)
        
    # 检查padding是否超过有效kernel大小的一半（与PyTorch行为一致）
    if check_pad:
        max_pad_h = effective_kh // 2
        max_pad_w = effective_kw // 2
        
        if ph > max_pad_h:
            raise RuntimeError(f"pad should be at most half of effective kernel size, but got pad={ph}, kernel_size={kh} and dilation={dh}")
        if pw > max_pad_w:
            raise RuntimeError(f"pad should be at most half of effective kernel size, but got pad={pw}, kernel_size={kw} and dilation={dw}")
    
    L = H_out * W_out
    
    # 创建展开张量
    # 首先，根据需要对输入进行填充
    if padding != (0, 0) or ceil_mode:
        # 计算基础填充后的形状（仅包括显式padding）
        base_padded_height = H_in + 2 * ph
        base_padded_width = W_in + 2 * pw
        
        # 当ceil_mode=True时，计算需要的最小填充大小
        if ceil_mode:
            min_padded_height = (H_out - 1) * sh + effective_kh
            min_padded_width = (W_out - 1) * sw + effective_kw
        else:
            min_padded_height = base_padded_height
            min_padded_width = base_padded_width
        
        # 计算需要额外添加的填充（只在底部和右侧）
        # 显式padding已经包含了顶部和左侧的填充
        extra_pad_bottom = builtins.max(0, min_padded_height - base_padded_height)
        extra_pad_right = builtins.max(0, min_padded_width - base_padded_width)
        
        # 计算最终填充后的形状
        final_padded_height = base_padded_height + extra_pad_bottom
        final_padded_width = base_padded_width + extra_pad_right
        
        # 创建最终填充后的张量
        padded_shape = (N, C, final_padded_height, final_padded_width)
        padded_input = full(padded_shape, fill_value=padvalue, dtype=input.dtype)
        
        # 将原始数据复制到填充后的张量中（考虑显式padding和额外填充）
        data_index = (slice(None), slice(None), slice(ph, ph + H_in), slice(pw, pw + W_in))
        padded_input.setat_(data_index, input)
    else:
        padded_input = input
        
    # 完全向量化版本：展开输入，避免双重循环
    arrlib = input._get_array_lib()
    h_starts = arrlib.arange(H_out) * stride[0]
    w_starts = arrlib.arange(W_out) * stride[1]
    h_k_range = arrlib.arange(kernel_size[0]) * dilation[0]
    w_k_range = arrlib.arange(kernel_size[1]) * dilation[1]
    
    kh, kw = kernel_size
    
    # 生成所有内核位置的坐标偏移 (kh, kw)
    kh_indices, kw_indices = arrlib.meshgrid(h_k_range, w_k_range, indexing='ij')
    
    # 生成所有输出位置的坐标网格 (H_out, W_out)
    h_indices, w_indices = arrlib.meshgrid(h_starts, w_starts, indexing='ij')
    
    # 计算所有需要提取的高度和宽度索引，形状为 (kh, kw, H_out, W_out)
    all_h_indices = h_indices[arrlib.newaxis, arrlib.newaxis, :, :] + kh_indices[:, :, arrlib.newaxis, arrlib.newaxis]
    all_w_indices = w_indices[arrlib.newaxis, arrlib.newaxis, :, :] + kw_indices[:, :, arrlib.newaxis, arrlib.newaxis]
    
    # 一次性提取所有展开块，避免双重循环，直接得到形状 (N, C, kh, kw, H_out, W_out)
    unfolded_blocks = padded_input[:, :, all_h_indices, all_w_indices]
    
    return unfolded_blocks, H_out, W_out

def fold(input: TN, output_size, kernel_size, dilation=1, padding=0, stride=1) -> TN:
    r"""将展开的张量折叠回原始形状。

    .. warning::
        目前仅支持4D输入张量（批处理图像类张量）的逆操作。

    参数:
        input (TN): 输入张量，形状为 (N, C * kernel_size[0] * kernel_size[1], L)
            其中 L 是滑动块的总数。
        output_size (int 或 tuple): 输出张量的空间维度 (H, W)
        kernel_size (int 或 tuple): 滑动块的大小
        dilation (int 或 tuple, 可选): 内核点之间的间距。默认值: 1
        padding (int 或 tuple, 可选): 两侧的隐式零填充。默认值: 0
        stride (int 或 tuple, 可选): 滑动块的步幅。默认值: 1

    返回:
        TN: 折叠后的张量，形状为 (N, C, H, W)
    """
    if not isinstance(input, TN):
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input)}")
    
    if input.ndim != 3:
        raise ValueError(f"Expected 3D tensor as input, but got {input.ndim}D tensor")
        
    kernel_size = _pair(kernel_size)
    padding = _pair(padding)
    stride = _pair(stride)
    output_size = _pair(output_size)
    dilation = _pair(dilation)
    if dilation[0] <= 0 or dilation[1] <= 0:
        raise RuntimeError(f"dilation should be greater than zero, but got dilation_height: {dilation[0]} dilation_width: {dilation[1]}")
    
    # 计算输出维度
    N, C_kern, L = input.shape
    C = C_kern // (kernel_size[0] * kernel_size[1])
    
    if C_kern % (kernel_size[0] * kernel_size[1]) != 0:
        raise ValueError(f"Expected input channel dimension to be divisible by kernel size product, but got C * kernel_size: {C_kern} and kernel_size product: {kernel_size[0] * kernel_size[1]}")
    
    # 计算H_out和W_out
    H_out = (output_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    W_out = (output_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    
    if H_out * W_out != L:
        raise ValueError(f"Expected input L dimension to be equal to H_out * W_out, but got L: {L} and H_out * W_out: {H_out * W_out}")
    
    # 创建输出张量（填充后的形状）
    padded_H = output_size[0] + 2 * padding[0]
    padded_W = output_size[1] + 2 * padding[1]
    
    # 创建形状为(N, C, padded_H, padded_W)的零张量
    output = zeros((N, C, padded_H, padded_W), dtype=input.dtype,device=input.device)
    
    # 向量化版本：将展开的块放回输出张量的正确位置
    arrlib = input._get_array_lib()
    h_starts = arrlib.arange(H_out) * stride[0]
    w_starts = arrlib.arange(W_out) * stride[1]
    h_k_range = arrlib.arange(kernel_size[0]) * dilation[0]
    w_k_range = arrlib.arange(kernel_size[1]) * dilation[1]
    
    kh, kw = kernel_size
    
    # 生成所有内核位置的坐标偏移 (kh, kw)
    kh_indices, kw_indices = arrlib.meshgrid(h_k_range, w_k_range, indexing='ij')
    
    # 生成所有输出位置的坐标网格 (H_out, W_out)
    h_indices, w_indices = arrlib.meshgrid(h_starts, w_starts, indexing='ij')
    
    # 计算所有需要更新的高度和宽度索引，形状为 (kh, kw, H_out, W_out)
    all_h_indices = h_indices[arrlib.newaxis, arrlib.newaxis, :, :] + kh_indices[:, :, arrlib.newaxis, arrlib.newaxis]
    all_w_indices = w_indices[arrlib.newaxis, arrlib.newaxis, :, :] + kw_indices[:, :, arrlib.newaxis, arrlib.newaxis]
    
    # 直接将输入重塑为 (N, C, kh, kw, H_out, W_out)，无需额外reshape
    folded_input = input.reshape(N, C, kh, kw, H_out, W_out)
    
    # 使用高级索引一次性将所有块添加到输出张量的正确位置
    # output[:, :, all_h_indices, all_w_indices] += folded_input
    output.addat_((slice(None), slice(None), all_h_indices, all_w_indices), folded_input)

    # 去除填充
    if padding != (0, 0):
        output = output[:, :, padding[0]:padding[0]+output_size[0], padding[1]:padding[1]+output_size[1]]
    
    return output

def unfold3d(input: TN, kernel_size, dilation=1, padding=0, padvalue=0.0, stride=None, ceil_mode=False, check_pad=True) -> tuple[TN, int, int, int]:
    r"""将输入张量从(N, C, D_in, H_in, W_in)转换为(N, C, kd, kh, kw, D_out, H_out, W_out)的格式。

    .. warning::
        目前仅支持5D输入张量（批处理3D数据类张量）。

    参数:
        input (TN): 输入张量，形状为 (N, C, D_in, H_in, W_in)
        kernel_size (tuple): 滑动块的大小，格式为 (kd, kh, kw)，分别对应深度、高度、宽度方向的内核大小
        dilation (tuple, 可选): 内核点之间的间距，格式为 (dd, dh, dw)。默认值: (1,1,1)
        padding (tuple, 可选): 两侧的隐式零填充，格式为 (pd, ph, pw)。默认值: (0,0,0)
        padvalue (float, 可选): 填充值。默认值: 0.0
        stride (tuple, 可选): 滑动块的步幅，格式为 (sd, sh, sw)。默认值: None（使用kernel_size）
        ceil_mode (bool, 可选): 是否使用向上取整计算输出尺寸。默认值: False
        check_pad (bool, 可选): 是否检查padding是否超过有效kernel大小的一半。默认值: True

    返回:
        tuple: 包含以下元素的元组：
            - TN: 转换后的张量，形状为 (N, C, kd, kh, kw, D_out, H_out, W_out)
            - int: 深度方向的输出尺寸 D_out
            - int: 高度方向的输出尺寸 H_out  
            - int: 宽度方向的输出尺寸 W_out

    形状说明:
        - 输入: (N, C, D_in, H_in, W_in)
        - 输出张量: (N, C, kd, kh, kw, D_out, H_out, W_out)
        - 其中:
            D_out = floor((D_in + 2*pd - dd*(kd-1) - 1)/sd + 1)
            H_out = floor((H_in + 2*ph - dh*(kh-1) - 1)/sh + 1)  
            W_out = floor((W_in + 2*pw - dw*(kw-1) - 1)/sw + 1)

    """
    if not isinstance(input, TN):
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input)}")
    
    if input.ndim != 5:
        raise ValueError(f"Expected 5D tensor as input, but got {input.ndim}D tensor")
     
    # 计算输出维度 - 与PyTorch实际行为一致
    N, C, D_in, H_in, W_in = input.shape
    kernel_size = _triple(kernel_size)
    padding = _triple(padding)
    stride = _triple(stride) if stride is not None else kernel_size
    dilation = _triple(dilation)
    
    # 检查膨胀率是否大于0
    if dilation[0] <= 0 or dilation[1] <= 0 or dilation[2] <= 0:
        raise RuntimeError(f"dilation should be greater than zero, but got dilation_height: {dilation[0]} dilation_width: {dilation[1]} dilation_depth: {dilation[2]}")
    
    kd, kh, kw = kernel_size
    sd, sh, sw = stride
    pd, ph, pw = padding
    dd, dh, dw = dilation

    H_out, effective_kh = _get_slide_win_count_and_size(H_in, 
                                                       kh, 
                                                       dh, 
                                                       ph, 
                                                       sh, 
                                                       ceil_mode)
    W_out, effective_kw = _get_slide_win_count_and_size(W_in, 
                                                       kw, 
                                                       dw, 
                                                       pw, 
                                                       sw, 
                                                       ceil_mode)
    D_out, effective_kd = _get_slide_win_count_and_size(D_in, 
                                                       kd, 
                                                       dd, 
                                                       pd, 
                                                       sd, 
                                                       ceil_mode)
        
    # 检查padding是否超过有效kernel大小的一半（与PyTorch行为一致）
    if check_pad:
        max_pad_h = effective_kh // 2
        max_pad_w = effective_kw // 2
        max_pad_d = effective_kd // 2
        
        if ph > max_pad_h:
            raise RuntimeError(f"pad should be at most half of effective kernel size, but got pad={ph}, kernel_size={kh} and dilation={dh}")
        if pw > max_pad_w:
            raise RuntimeError(f"pad should be at most half of effective kernel size, but got pad={pw}, kernel_size={kw} and dilation={dw}")
        if pd > max_pad_d:
            raise RuntimeError(f"pad should be at most half of effective kernel size, but got pad={pd}, kernel_size={kd} and dilation={dd}")
    
    # 创建展开张量
    # 首先，根据需要对输入进行填充
    if padding != (0, 0, 0) or ceil_mode:
        # 计算基础填充后的形状（仅包括显式padding）
        base_padded_depth = D_in + 2 * pd
        base_padded_height = H_in + 2 * ph
        base_padded_width = W_in + 2 * pw
        
        # 当ceil_mode=True时，计算需要的最小填充大小
        if ceil_mode:
            min_padded_depth = (D_out - 1) * sd + effective_kd
            min_padded_height = (H_out - 1) * sh + effective_kh
            min_padded_width = (W_out - 1) * sw + effective_kw
        else:
            min_padded_depth = base_padded_depth
            min_padded_height = base_padded_height
            min_padded_width = base_padded_width
        
        # 计算需要额外添加的填充（只在底部、右侧和后侧）
        # 显式padding已经包含了顶部、左侧和前侧的填充
        extra_pad_back = builtins.max(0, min_padded_depth - base_padded_depth)
        extra_pad_bottom = builtins.max(0, min_padded_height - base_padded_height)
        extra_pad_right = builtins.max(0, min_padded_width - base_padded_width)
        
        # 计算最终填充后的形状
        final_padded_depth = base_padded_depth + extra_pad_back
        final_padded_height = base_padded_height + extra_pad_bottom
        final_padded_width = base_padded_width + extra_pad_right
        
        # 创建最终填充后的张量
        padded_shape = (N, C, final_padded_depth, final_padded_height, final_padded_width)
        padded_input = full(padded_shape, fill_value=padvalue, dtype=input.dtype,device=input.device)
        
        # 将原始数据复制到填充后的张量中（考虑显式padding和额外填充）
        data_index = (slice(None), slice(None), 
                      slice(pd, pd + D_in), 
                      slice(ph, ph + H_in), 
                      slice(pw, pw + W_in))
        padded_input.setat_(data_index, input)
    else:
        padded_input = input
        
    # 完全向量化版本：展开输入，避免三重循环
    arrlib = input._get_array_lib()
    d_starts = arrlib.arange(D_out) * stride[0]
    h_starts = arrlib.arange(H_out) * stride[1]
    w_starts = arrlib.arange(W_out) * stride[2]
    d_k_range = arrlib.arange(kernel_size[0]) * dilation[0]
    h_k_range = arrlib.arange(kernel_size[1]) * dilation[1]
    w_k_range = arrlib.arange(kernel_size[2]) * dilation[2]
    
    kd, kh, kw = kernel_size
    
    # 生成所有内核位置的坐标偏移 (kd, kh, kw)
    kd_indices, kh_indices, kw_indices = arrlib.meshgrid(d_k_range, h_k_range, w_k_range, indexing='ij')
    
    # 生成所有输出位置的坐标网格 (D_out, H_out, W_out)
    d_indices, h_indices, w_indices = arrlib.meshgrid(d_starts, h_starts, w_starts, indexing='ij')
    
    # 计算所有需要提取的深度、高度和宽度索引，形状为 (kd, kh, kw, D_out, H_out, W_out)
    all_d_indices = d_indices[arrlib.newaxis, arrlib.newaxis, arrlib.newaxis, :, :, :] + kd_indices[:, :, :, arrlib.newaxis, arrlib.newaxis, arrlib.newaxis]
    all_h_indices = h_indices[arrlib.newaxis, arrlib.newaxis, arrlib.newaxis, :, :, :] + kh_indices[:, :, :, arrlib.newaxis, arrlib.newaxis, arrlib.newaxis]
    all_w_indices = w_indices[arrlib.newaxis, arrlib.newaxis, arrlib.newaxis, :, :, :] + kw_indices[:, :, :, arrlib.newaxis, arrlib.newaxis, arrlib.newaxis]
    
    # 一次性提取所有展开块，避免三重循环，直接得到形状 (N, C, kd, kh, kw, D_out, H_out, W_out)
    unfolded_blocks = padded_input[:, :, all_d_indices, all_h_indices, all_w_indices]
    
    return unfolded_blocks, D_out, H_out, W_out

def conv1d(input: TN, weight: TN, bias: Optional[TN] = None, stride=1, padding=0, dilation=1, groups=1) -> TN:
    r"""对输入张量应用1D卷积。

    参数:
        input (TN): 输入张量，形状为 (N, C_in, L_in)
        weight (TN): 卷积核，形状为 (C_out, C_in/groups, K)
        bias (TN, 可选): 偏置项，形状为 (C_out)。默认值: None
        stride (int 或 tuple, 可选): 卷积步长。默认值: 1
        padding (int 或 tuple, 可选): 两侧的隐式零填充。默认值: 0
        dilation (int 或 tuple, 可选): 内核点之间的间距。默认值: 1
        groups (int, 可选): 分组卷积组数。默认值: 1

    返回:
        TN: 输出张量，形状为 (N, C_out, L_out)

    形状转换:
        - 输入: (N, C_in, L_in)
        - 卷积核: (C_out, C_in/groups, K)
        - 偏置: (C_out)
        - 输出: (N, C_out, L_out)
        其中:
        L_out = floor((L_in + 2*padding - dilation*(K-1) - 1)/stride + 1)
    """
    if not isinstance(input, TN):
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input)}")
    if not isinstance(weight, TN):
        raise TypeError(f"Expected weight type to be TN tensor, but received type: {type(weight)}")
    if bias is not None and not isinstance(bias, TN):
        raise TypeError(f"Expected bias type to be TN tensor or None, but received type: {type(bias)}")
    
    if input.device != weight.device or (bias is not None and bias.device != weight.device):
        raise ValueError("input, weight, and bias must have the same device")
    
    if input.ndim != 3 or weight.ndim != 3:
        raise ValueError(f"Expected 3D tensors for input and weight, but got input dim: {input.ndim}, weight dim: {weight.ndim}")
    
    stride = _single(stride)
    padding = _single(padding)
    dilation = _single(dilation)
    
    # 检查groups参数
    if not isinstance(groups, int) or groups <= 0:
        raise ValueError(f"groups must be a positive integer, but got {groups}")
    
    N, C_in, L_in = input.shape
    C_out, C_in_per_group, K = weight.shape
    
    # 检查输入通道数是否与卷积核匹配
    if C_in != C_in_per_group * groups:
        raise ValueError(f"input channels ({C_in}) must match weight channels ({C_in_per_group} * {groups})")
    
    # 计算输出维度
    L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    
    # 为1D卷积实现展开操作
    # 先对输入进行填充
    if padding != 0:
        # 使用非原地操作创建填充后的输入，确保计算图完整
        left_pad = zeros((N, C_in, padding), dtype=input.dtype,device=input.device)
        right_pad = zeros((N, C_in, padding), dtype=input.dtype,device=input.device)
        padded_input = concatenate((left_pad, input, right_pad), dim=2)
    else:
        padded_input = input
    
    # 展开输入 - 向量化实现
    # 使用numpy.arange生成所有内核索引的起始位置
    arrlib = input._get_array_lib()
    l_starts = arrlib.arange(L_out) * stride
    
    # 预计算所有内核块的索引范围
    k_range = arrlib.arange(K) * dilation
    
    # 一次性生成所有需要的索引
    all_indices = l_starts[:, arrlib.newaxis] + k_range
    
    # 使用高级索引一次性提取所有内核块
    # 首先创建一个形状为 (N, C_in, L_out, K) 的临时张量
    unfolded_temp = padded_input[:, :, all_indices]
    
    # 转置维度，将K维度放在L_out前面，确保正确的展开顺序
    unfolded_temp = unfolded_temp.transpose(2, 3)
    
    # 重塑为 (N, C_in*K, L_out) 形状
    unfolded_input = unfolded_temp.reshape(N, C_in*K, L_out)

    # 处理分组卷积
    if groups > 1:
        # 将输入和权重分成groups组
        unfolded_input = unfolded_input.reshape(N, groups, C_in_per_group * K, L_out)
        weight_reshaped = weight.reshape(groups, C_out // groups, C_in_per_group * K)
        
        # 对每个组进行矩阵乘法
        output = zeros((N, groups, C_out // groups, L_out), dtype=input.dtype,device=input.device)
        for g in range(groups):
            # weight_reshaped[g]形状: (C_out_per_group, C_in_per_group*K)
            # unfolded_input[:, g, :, :]形状: (N, C_in_per_group*K, L_out)
            # 矩阵乘法后直接得到: (N, C_out_per_group, L_out)
            index = (slice(None), g, slice(None), slice(None))
            output.setat_(index, weight_reshaped[g] @ unfolded_input[index])

        # 合并所有组的结果
        output = output.reshape(N, C_out, L_out)
    else:
        # 将权重reshape为 (C_out, C_in*K)
        weight_reshaped = weight.reshape(C_out, C_in * K)
        
        # 直接使用批量矩阵乘法
        # weight_reshaped形状: (C_out, C_in*K)
        # unfolded_input形状: (N, C_in*K, L_out)
        # 矩阵乘法后直接得到: (N, C_out, L_out)
        output = weight_reshaped @ unfolded_input
    
    # 添加偏置项
    if bias is not None:
        if not isinstance(bias, TN):
            raise TypeError(f"Expected bias to be TN tensor, but received type: {type(bias)}")
        if bias.shape != (C_out,):
            raise ValueError(f"bias shape ({bias.shape}) must match output channels ({C_out})")
        
        # 广播偏置到输出张量
        output = output + bias.reshape(1, C_out, 1)
    
    return output

def conv2d(input: TN, weight: TN, bias: Optional[TN] = None, stride=1, padding=0, dilation=1, groups=1) -> TN:
    r"""对输入张量应用2D卷积。

    参数:
        input (TN): 输入张量，形状为 (N, C_in, H_in, W_in)
        weight (TN): 卷积核，形状为 (C_out, C_in/groups, K_h, K_w)
        bias (TN, 可选): 偏置项，形状为 (C_out)。默认值: None
        stride (int 或 tuple, 可选): 卷积步长。默认值: 1
        padding (int 或 tuple, 可选): 两侧的隐式零填充。默认值: 0
        dilation (int 或 tuple, 可选): 内核点之间的间距。默认值: 1
        groups (int, 可选): 分组卷积组数。默认值: 1

    返回:
        TN: 输出张量，形状为 (N, C_out, H_out, W_out)

    形状转换:
        - 输入: (N, C_in, H_in, W_in)
        - 卷积核: (C_out, C_in/groups, K_h, K_w)
        - 偏置: (C_out)
        - 输出: (N, C_out, H_out, W_out)
        其中:
        H_out = floor((H_in + 2*padding[0] - dilation[0]*(K_h-1) - 1)/stride[0] + 1)
        W_out = floor((W_in + 2*padding[1] - dilation[1]*(K_w-1) - 1)/stride[1] + 1)
    """
    if not isinstance(input, TN):
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input)}")
    if not isinstance(weight, TN):
        raise TypeError(f"Expected weight type to be TN tensor, but received type: {type(weight)}")
    if bias is not None and not isinstance(bias, TN):
        raise TypeError(f"Expected bias type to be TN tensor or None, but received type: {type(bias)}")
    
    if input.device != weight.device or (bias is not None and bias.device != weight.device):
        raise ValueError("input, weight, and bias must have the same device")
    
    if input.ndim != 4 or weight.ndim != 4:
        raise ValueError(f"Expected 4D tensors for input and weight, but got input dim: {input.ndim}, weight dim: {weight.ndim}")
    
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    
    # 检查groups参数
    if not isinstance(groups, int) or groups <= 0:
        raise ValueError(f"groups must be a positive integer, but got {groups}")
    
    N, C_in, H_in, W_in = input.shape
    C_out, C_in_per_group, K_h, K_w = weight.shape
    
    # 检查输入通道数是否与卷积核匹配
    if C_in != C_in_per_group * groups:
        raise ValueError(f"input channels ({C_in}) must match weight channels ({C_in_per_group} * {groups})")
    
    # 使用unfold2d函数展开输入并获取H_out和W_out
    unfolded_input, H_out, W_out = unfold2d(input, 
                                            kernel_size=(K_h, K_w), 
                                            dilation=dilation, 
                                            padding=padding, 
                                            padvalue=0.0, 
                                            stride=stride, 
                                            ceil_mode=False,
                                            check_pad=False)
        
    # 处理分组卷积
    if groups > 1:
        # 将输入和权重分成groups组,每组为一个矩阵，滑窗内通道数据的列向量化为矩阵的列，矩阵列数为滑窗数目
        unfolded_result = unfolded_input.reshape(N, groups, C_in_per_group * K_h * K_w, H_out * W_out)
        weight_reshaped = weight.reshape(groups, C_out // groups, C_in_per_group * K_h * K_w)
        
        # 对每个组进行矩阵乘法
        output = zeros((N, groups, C_out // groups, H_out * W_out), dtype=input.dtype,device=input.device)
        for g in range(groups):
            # 直接使用批量矩阵乘法，避免冗余转置
            # weight_reshaped[g]形状: (C_out_per_group, C_in_per_group*K_h*K_w)
            # unfolded_input[:, g, :, :]形状: (N, C_in_per_group*K_h*K_w, L)
            # 矩阵乘法后直接得到: (N, C_out_per_group, L)
            # output[:, g, :, :] = matmul(weight_reshaped[g], unfolded_input[:, g, :, :])
            index = (slice(None), g, slice(None), slice(None))
            output.setat_(index, weight_reshaped[g] @ unfolded_result[index])

        # 合并所有组的结果
        output = output.reshape(N, C_out, H_out * W_out)
    else:
        # 将展开后输入reshape为形状 (N, C_in*K_h*K_w, H_out*W_out)
        unfolded_result = unfolded_input.reshape(N, C_in*K_h*K_w, H_out*W_out)

        # 将权重reshape为 (C_out, C_in*K_h*K_w)
        weight_reshaped = weight.reshape(C_out, C_in * K_h * K_w)
        
        # 直接使用批量矩阵乘法，避免冗余转置
        # weight_reshaped形状: (C_out, C_in*K_h*K_w)
        # unfolded_input形状: (N, C_in*K_h*K_w, L)
        # 矩阵乘法后直接得到: (N, C_out, L)
        output = weight_reshaped @ unfolded_result
    
    # 将结果reshape为 (N, C_out, H_out, W_out)
    output = output.reshape(N, C_out, H_out, W_out)
    
    # 添加偏置项
    if bias is not None:
        if bias.shape != (C_out,):
            raise ValueError(f"bias shape ({bias.shape}) must match output channels ({C_out})")
        
        # 广播偏置到输出张量
        output = output + bias.reshape(1, C_out, 1, 1)
    
    return output

def conv3d(input: TN, weight: TN, bias: Optional[TN] = None, stride=1, padding=0, dilation=1, groups=1) -> TN:
    r"""对输入张量应用3D卷积。

    参数:
        input (TN): 输入张量，形状为 (N, C_in, D_in, H_in, W_in)
        weight (TN): 卷积核，形状为 (C_out, C_in/groups, K_d, K_h, K_w)
        bias (TN, 可选): 偏置项，形状为 (C_out)。默认值: None
        stride (int 或 tuple, 可选): 卷积步长。默认值: 1
        padding (int 或 tuple, 可选): 两侧的隐式零填充。默认值: 0
        dilation (int 或 tuple, 可选): 内核点之间的间距。默认值: 1
        groups (int, 可选): 分组卷积组数。默认值: 1

    返回:
        TN: 输出张量，形状为 (N, C_out, D_out, H_out, W_out)

    形状转换:
        - 输入: (N, C_in, D_in, H_in, W_in)
        - 卷积核: (C_out, C_in/groups, K_d, K_h, K_w)
        - 偏置: (C_out)
        - 输出: (N, C_out, D_out, H_out, W_out)
        其中:
        D_out = floor((D_in + 2*padding[0] - dilation[0]*(K_d-1) - 1)/stride[0] + 1)
        H_out = floor((H_in + 2*padding[1] - dilation[1]*(K_h-1) - 1)/stride[1] + 1)
        W_out = floor((W_in + 2*padding[2] - dilation[2]*(K_w-1) - 1)/stride[2] + 1)
    """
    if not isinstance(input, TN):
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input)}")
    if not isinstance(weight, TN):
        raise TypeError(f"Expected weight type to be TN tensor, but received type: {type(weight)}")
    if bias is not None and not isinstance(bias, TN):
        raise TypeError(f"Expected bias type to be TN tensor or None, but received type: {type(bias)}")
    
    if input.device != weight.device or (bias is not None and bias.device != weight.device):
        raise ValueError("input, weight, and bias must have the same device")
    
    if input.ndim != 5 or weight.ndim != 5:
        raise ValueError(f"Expected 5D tensors for input and weight, but got input dim: {input.ndim}, weight dim: {weight.ndim}")
    
    stride = _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)
    
    # 检查groups参数
    if not isinstance(groups, int) or groups <= 0:
        raise ValueError(f"groups must be a positive integer, but got {groups}")
    
    N, C_in, D_in, H_in, W_in = input.shape
    C_out, C_in_per_group, K_d, K_h, K_w = weight.shape
    
    # 检查输入通道数是否与卷积核匹配
    if C_in != C_in_per_group * groups:
        raise ValueError(f"input channels ({C_in}) must match weight channels ({C_in_per_group} * {groups})")
    
    # 使用unfold3d函数展开输入并获取D_out, H_out和W_out
    unfolded_input, D_out, H_out, W_out = unfold3d(input, 
                                            kernel_size=(K_d, K_h, K_w), 
                                            dilation=dilation, 
                                            padding=padding, 
                                            padvalue=0.0, 
                                            stride=stride, 
                                            ceil_mode=False,
                                            check_pad=False)
        
    # 处理分组卷积
    if groups > 1:
        # 将输入和权重分成groups组,每组为一个矩阵，滑窗内通道数据的列向量化为矩阵的列，矩阵列数为滑窗数目
        unfolded_result = unfolded_input.reshape(N, groups, C_in_per_group * K_d * K_h * K_w, D_out * H_out * W_out)
        weight_reshaped = weight.reshape(groups, C_out // groups, C_in_per_group * K_d * K_h * K_w)
        
        # 对每个组进行矩阵乘法
        output = zeros((N, groups, C_out // groups, D_out * H_out * W_out), dtype=input.dtype,device=input.device)
        for g in range(groups):
            # 直接使用批量矩阵乘法，避免冗余转置
            # weight_reshaped[g]形状: (C_out_per_group, C_in_per_group*K_d*K_h*K_w)
            # unfolded_input[:, g, :, :]形状: (N, C_in_per_group*K_d*K_h*K_w, L)
            # 矩阵乘法后直接得到: (N, C_out_per_group, L)
            # output[:, g, :, :] = matmul(weight_reshaped[g], unfolded_input[:, g, :, :])
            index = (slice(None), g, slice(None), slice(None))
            output.setat_(index, weight_reshaped[g] @ unfolded_result[index])

        # 合并所有组的结果
        output = output.reshape(N, C_out, D_out * H_out * W_out)
    else:
        # 将展开后输入reshape为形状 (N, C_in*K_d*K_h*K_w, D_out*H_out*W_out)
        unfolded_result = unfolded_input.reshape(N, C_in*K_d*K_h*K_w, D_out*H_out*W_out)

        # 将权重reshape为 (C_out, C_in*K_d*K_h*K_w)
        weight_reshaped = weight.reshape(C_out, C_in * K_d * K_h * K_w)
        
        # 直接使用批量矩阵乘法，避免冗余转置
        # weight_reshaped形状: (C_out, C_in*K_d*K_h*K_w)
        # unfolded_input形状: (N, C_in*K_d*K_h*K_w, L)
        # 矩阵乘法后直接得到: (N, C_out, L)
        output = weight_reshaped @ unfolded_result
    
    # 将结果reshape为 (N, C_out, D_out, H_out, W_out)
    output = output.reshape(N, C_out, D_out, H_out, W_out)
    
    # 添加偏置项
    if bias is not None:
        if bias.shape != (C_out,):
            raise ValueError(f"bias shape ({bias.shape}) must match output channels ({C_out})")
        
        # 广播偏置到输出张量
        output = output + bias.reshape(1, C_out, 1, 1, 1)
    
    return output

def max_pool1d(input: TN, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False) -> TN | tuple[TN, TN]:
    r"""对输入张量应用1D最大池化。

    参数:
        input (TN): 输入张量，形状为 (N, C, L_in)
        kernel_size (int 或 tuple): 池化核大小
        stride (int 或 tuple, 可选): 池化步长。默认值: kernel_size
        padding (int 或 tuple, 可选): 两侧的隐式零填充。默认值: 0
        dilation (int 或 tuple, 可选): 内核点之间的间距。默认值: 1
        ceil_mode (bool, 可选): 是否使用向上取整计算输出大小。默认值: False
        return_indices (bool, 可选): 是否返回最大值的索引。默认值: False
        
    返回:
        TN 或 tuple(TN, TN): 输出张量，形状为 (N, C, L_out)
            如果 return_indices 为 True，则返回 (output, indices)

    形状转换:
        - 输入: (N, C, L_in)
        - 输出: (N, C, L_out)
        其中:
        L_out = floor((L_in + 2*padding - dilation*(K-1) - 1)/stride + 1)
        当 ceil_mode=True 时，使用 ceil 替代 floor 进行计算
    """
    if not isinstance(input, TN):
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input)}")
    
    if input.ndim != 3:
        raise ValueError(f"Expected 3D tensor for input, but got {input.ndim}D tensor")
    
    # 处理参数
    kernel_size = _single(kernel_size)
    stride = _single(stride) if stride is not None else kernel_size
    padding = _single(padding)
    dilation = _single(dilation)
    
    # 获取输入形状
    N, C, L_in = input.shape
    K = kernel_size
    
    L_out, effective_K = _get_slide_win_count_and_size(L_in, 
                                                       K, 
                                                       dilation, 
                                                       padding, 
                                                       stride, 
                                                       ceil_mode)

    # 检查padding是否超过有效kernel大小的一半（与PyTorch行为一致）
    max_pad = effective_K // 2
    if padding > max_pad:
        raise RuntimeError(f"pad should be at most half of effective kernel size, but got pad={padding}, kernel_size={K} and dilation={dilation}")
    
    # 为1D池化实现展开操作
    # 先对输入进行填充
    if padding != 0:
        # 使用非原地操作创建填充后的输入，确保计算图完整
        # 创建左右填充张量，使用负无穷大填充以正确计算边界窗口的最大值
        left_pad = full((N, C, padding), fill_value=-np.inf, dtype=input.dtype,device=input.device)
        right_pad = full((N, C, padding), fill_value=-np.inf, dtype=input.dtype,device=input.device)
        # 沿时间维度拼接填充和原始输入
        padded_input = concatenate((left_pad, input, right_pad), dim=2)
    else:
        padded_input = input
    
    # 展开输入 - 优化版本
    # 直接生成形状为 (N*C, L_out, K) 的张量，避免多次转置和重塑
    arrlib = input._get_array_lib()
    l_starts = arrlib.arange(L_out) * stride
    k_range = arrlib.arange(K) * dilation
    all_indices = l_starts[:, np.newaxis] + k_range
    all_indices = arrlib.clip(all_indices, 0, padded_input.shape[2] - 1)
    
    # 使用高级索引提取所有内核块，然后直接重塑为 (N*C, L_out, K)
    unfolded_temp = padded_input[:, :, all_indices]
    
    # 直接重塑为 (N*C, L_out, K) 形状，避免不必要的转置
    unfolded_input_reshaped = unfolded_temp.reshape(N * C, L_out, K)
    
    # 在核维度上计算最大值
    max_result = max(unfolded_input_reshaped, dim=2, keepdim=False)
    values = max_result.values
    indices = max_result.indices
    
    # 将输出重塑为 (N, C, L_out)
    output = values.reshape(N, C, L_out)
    
    if return_indices:
        # 修复索引计算，使其与PyTorch一致
        indices_data = indices.data
        
        # 将indices_data转换为形状为(N*C, L_out)
        indices_reshaped = indices_data.reshape(N*C, L_out)
        
        # 创建网格坐标 (L_out)
        grid_l = arrlib.arange(L_out).reshape(1, L_out).repeat(N*C, axis=0)
        
        # 计算核内坐标
        indices_k = indices_reshaped
        
        # 计算输入位置坐标
        input_l = grid_l * stride - padding + indices_k * dilation
        
        # 将输入坐标转换为展平的索引 (L_in)
        # 确保坐标在有效范围内
        input_l = arrlib.clip(input_l, 0, L_in - 1)
        
        flattened_indices = input_l.reshape(N, C, L_out)
        flattened_indices = tensor(flattened_indices, device=input.device)
        return output, flattened_indices
    else:
        return output

def max_pool2d(input: TN, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False) -> TN | tuple[TN, TN]:
    r"""对输入张量应用2D最大池化。

    参数:
        input (TN): 输入张量，形状为 (N, C, H_in, W_in)
        kernel_size (int 或 tuple): 池化核大小
        stride (int 或 tuple, 可选): 池化步长。默认值: kernel_size
        padding (int 或 tuple, 可选): 两侧的隐式零填充。默认值: 0
        dilation (int 或 tuple, 可选): 内核点之间的间距。默认值: 1
        ceil_mode (bool, 可选): 是否使用向上取整计算输出大小。默认值: False
        return_indices (bool, 可选): 是否返回最大值的索引。默认值: False
        
    返回:
        TN 或 tuple(TN, TN): 输出张量，形状为 (N, C, H_out, W_out)
            如果 return_indices 为 True，则返回 (output, indices)

    形状转换:
        - 输入: (N, C, H_in, W_in)
        - 输出: (N, C, H_out, W_out)
        其中:
        H_out = floor((H_in + 2*padding[0] - dilation[0]*(K_h-1) - 1)/stride[0] + 1)
        W_out = floor((W_in + 2*padding[1] - dilation[1]*(K_w-1) - 1)/stride[1] + 1)
        当 ceil_mode=True 时，使用 ceil 替代 floor 进行计算
    """
    if not isinstance(input, TN):
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input)}")
    
    if input.ndim != 4:
        raise ValueError(f"Expected 4D tensor for input, but got {input.ndim}D tensor")
    
    # 参数处理
    kernel_size = _pair(kernel_size)
    padding = _pair(padding)
    stride = _pair(stride) if stride is not None else kernel_size
    dilation = _pair(dilation)
    # 检查膨胀率是否大于0
    if dilation[0] <= 0 or dilation[1] <= 0:
        raise RuntimeError(f"dilation should be greater than zero, but got dilation_height: {dilation[0]} dilation_width: {dilation[1]}")
    
    # 获取输入形状
    N, C, H_in, W_in = input.shape
    kh, kw = kernel_size
    
    # 使用unfold2d函数展开输入为(N, C, kernel_size[0], kernel_size[1], H_out, W_out)
    # 支持ceil_mode、check_pad，并获取H_out和W_out
    unfolded_input, H_out, W_out = unfold2d(input, 
                                            kernel_size=kernel_size, 
                                            dilation=dilation, 
                                            padding=padding, 
                                            padvalue=float('-inf'), 
                                            stride=stride, 
                                            ceil_mode=ceil_mode,
                                            check_pad=True)
    
    # 直接reshape返回池化函数所需的形状 (N*C, kh*kw, H_out*W_out)
    unfolded_result = unfolded_input.reshape(N*C, kh*kw, H_out*W_out)
    
    # 在核维度上计算最大值
    max_result = max(unfolded_result, dim=1, keepdim=False)
    values = max_result.values
    indices = max_result.indices
    
    # 将输出重塑为 (N, C, H_out, W_out)
    output = values.reshape(N, C, H_out, W_out)
    
    if return_indices:
        # 计算每个输出位置对应的输入位置
        indices_data = indices.data
        
        # 将indices_data转换为形状为(N*C, H_out*W_out)
        indices_reshaped = indices_data.reshape(N*C, H_out*W_out)
        
        # 创建网格坐标 (H_out, W_out)
        arrlib = input._get_array_lib()
        grid_y, grid_x = arrlib.meshgrid(arrlib.arange(H_out), arrlib.arange(W_out), indexing='ij')
        grid_y = grid_y.reshape(1, H_out*W_out).repeat(N*C, axis=0)
        grid_x = grid_x.reshape(1, H_out*W_out).repeat(N*C, axis=0)
        
        # 计算核内坐标
        indices_kh = indices_reshaped // kw
        indices_kw = indices_reshaped % kw
        
        # 计算输入位置坐标
        input_y = grid_y * stride[0] - padding[0] + indices_kh * dilation[0]
        input_x = grid_x * stride[1] - padding[1] + indices_kw * dilation[1]
        
        # 将输入坐标转换为展平的索引 (H*W)        
        flattened_indices = input_y * W_in + input_x
        flattened_indices = flattened_indices.reshape(N, C, H_out, W_out)
        flattened_indices = tensor(flattened_indices, device=input.device)
        return output, flattened_indices
    else:
        return output

def max_pool3d(input: TN, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False) -> TN | tuple[TN, TN]:
    r"""对输入张量应用3D最大池化。

    参数:
        input (TN): 输入张量，形状为 (N, C, D_in, H_in, W_in)
        kernel_size (int 或 tuple): 池化核大小
        stride (int 或 tuple, 可选): 池化步长。默认值: kernel_size
        padding (int 或 tuple, 可选): 两侧的隐式零填充。默认值: 0
        dilation (int 或 tuple, 可选): 内核点之间的间距。默认值: 1
        ceil_mode (bool, 可选): 是否使用向上取整计算输出大小。默认值: False
        return_indices (bool, 可选): 是否返回最大值的索引。默认值: False
        
    返回:
        TN 或 tuple(TN, TN): 输出张量，形状为 (N, C, D_out, H_out, W_out)
            如果 return_indices 为 True，则返回 (output, indices)

    形状转换:
        - 输入: (N, C, D_in, H_in, W_in)
        - 输出: (N, C, D_out, H_out, W_out)
        其中:
        D_out = floor((D_in + 2*padding[0] - dilation[0]*(K_d-1) - 1)/stride[0] + 1)
        H_out = floor((H_in + 2*padding[1] - dilation[1]*(K_h-1) - 1)/stride[1] + 1)
        W_out = floor((W_in + 2*padding[2] - dilation[2]*(K_w-1) - 1)/stride[2] + 1)
        当 ceil_mode=True 时，使用 ceil 替代 floor 进行计算
    """
    if not isinstance(input, TN):
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input)}")
    
    if input.ndim != 5:
        raise ValueError(f"Expected 5D tensor for input, but got {input.ndim}D tensor")
    
    # 参数处理
    kernel_size = _triple(kernel_size)
    padding = _triple(padding)
    stride = _triple(stride) if stride is not None else kernel_size
    dilation = _triple(dilation)
    
    # 检查膨胀率是否大于0
    if dilation[0] <= 0 or dilation[1] <= 0 or dilation[2] <= 0:
        raise RuntimeError(f"dilation should be greater than zero, but got dilation_depth: {dilation[0]} dilation_height: {dilation[1]} dilation_width: {dilation[2]}")
    
    # 获取输入形状
    N, C, D_in, H_in, W_in = input.shape
    kd, kh, kw = kernel_size
    
    # 使用unfold3d函数展开输入为(N, C, kd, kh, kw, D_out, H_out, W_out)
    # 支持ceil_mode、check_pad，并获取D_out、H_out和W_out
    unfolded_input, D_out, H_out, W_out = unfold3d(input, 
                                                    kernel_size=kernel_size, 
                                                    dilation=dilation, 
                                                    padding=padding, 
                                                    padvalue=float('-inf'), 
                                                    stride=stride, 
                                                    ceil_mode=ceil_mode,
                                                    check_pad=True)
    
    # 直接reshape返回池化函数所需的形状 (N*C, kd*kh*kw, D_out*H_out*W_out)
    unfolded_result = unfolded_input.reshape(N*C, kd*kh*kw, D_out*H_out*W_out)
    
    # 在核维度上计算最大值
    max_result = max(unfolded_result, dim=1, keepdim=False)
    values = max_result.values
    indices = max_result.indices
    
    # 将输出重塑为 (N, C, D_out, H_out, W_out)
    output = values.reshape(N, C, D_out, H_out, W_out)
    
    if return_indices:
        # 计算每个输出位置对应的输入位置
        indices_data = indices.data
        
        # 将indices_data转换为形状为(N*C, D_out*H_out*W_out)
        indices_reshaped = indices_data.reshape(N*C, D_out*H_out*W_out)
        
        # 创建网格坐标 (D_out, H_out, W_out)
        arrlib = input._get_array_lib()
        grid_d, grid_h, grid_w = arrlib.meshgrid(arrlib.arange(D_out), arrlib.arange(H_out), arrlib.arange(W_out), indexing='ij')
        grid_d = grid_d.reshape(1, D_out*H_out*W_out).repeat(N*C, axis=0)
        grid_h = grid_h.reshape(1, D_out*H_out*W_out).repeat(N*C, axis=0)
        grid_w = grid_w.reshape(1, D_out*H_out*W_out).repeat(N*C, axis=0)
        
        # 计算核内坐标
        indices_k = indices_reshaped
        indices_d = indices_k // (kh * kw)
        indices_h = (indices_k % (kh * kw)) // kw
        indices_w = indices_k % kw
        
        # 计算输入坐标
        input_d = grid_d * stride[0] + indices_d * dilation[0] - padding[0]
        input_h = grid_h * stride[1] + indices_h * dilation[1] - padding[1]
        input_w = grid_w * stride[2] + indices_w * dilation[2] - padding[2]
        
        # 计算线性索引
        input_indices = (input_d * H_in + input_h) * W_in + input_w
        
        # 重塑为最终形状
        indices_output = input_indices.reshape(N, C, D_out, H_out, W_out)
        
        return output, indices_output
    
    return output

def avg_pool1d(input: TN, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) -> TN:
    r"""对输入张量应用1D平均池化。
    
    参数:
        input (TN): 输入张量，形状为 (N, C, L_in)
        kernel_size (int 或 tuple): 池化核大小
        stride (int 或 tuple, 可选): 步长。默认值: None（与kernel_size相同）
        padding (int 或 tuple, 可选): 两侧的隐式零填充。默认值: 0
        ceil_mode (bool, 可选): 是否使用向上取整计算输出形状。默认值: False
        count_include_pad (bool, 可选): 是否在计算平均值时包含填充区域。默认值: True
        divisor_override (int, 可选): 如果指定，将使用此值代替kernel_size进行除法。默认值: None
    
    返回:
        TN: 池化后的张量，形状为 (N, C, L_out)
    """
    if not isinstance(input, TN):
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input)}")
    
    if input.ndim != 3:
        raise ValueError(f"Expected 3D tensor for input, but got {input.ndim}D tensor")
    
    # 参数处理
    kernel_size = _single(kernel_size)
    stride = _single(stride) if stride is not None else kernel_size
    padding = _single(padding)
    dilation = 1  # avg_pool1d不支持dilation参数，固定为1
    
    N, C, L_in = input.shape
    K = kernel_size
    
    L_out, effective_K = _get_slide_win_count_and_size(L_in, 
                                                       K, 
                                                       dilation, 
                                                       padding, 
                                                       stride, 
                                                       ceil_mode)

    # 检查padding是否超过有效kernel大小的一半（与PyTorch行为一致）
    max_pad = effective_K // 2
    if padding > max_pad:
        raise RuntimeError(f"pad should be at most half of effective kernel size, but got pad={padding}, kernel_size={K} and dilation={dilation}")
    
    # 先对输入进行填充
    if padding != 0 or ceil_mode:
        # 计算基础填充后的形状（仅包括显式padding）
        base_padded_length = L_in + 2 * padding
        
        # 当ceil_mode=True时，计算需要的最小填充大小 - 与im2col一致
        if ceil_mode:
            min_padded_length = (L_out - 1) * stride + effective_K
        else:
            min_padded_length = base_padded_length
        
        # 计算需要额外添加的填充（只在右侧）
        # 显式padding已经包含了左侧的填充
        extra_pad_right = builtins.max(0, min_padded_length - base_padded_length)
        
        # 计算最终填充后的形状
        final_padded_length = base_padded_length + extra_pad_right
        
        # 创建最终填充后的张量
        padded_shape = (N, C, final_padded_length)
        padded_input = zeros(padded_shape, dtype=input.dtype,device=input.device)
        
        # 将原始数据复制到填充后的张量中（考虑显式padding和额外填充）
        data_index = (slice(None), slice(None), slice(padding, padding + L_in))
        padded_input.setat_(data_index, input)
    else:
        padded_input = input
    
    # 展开输入 - 优化版本
    # 使用numpy生成所有内核索引的起始位置
    arrlib = input._get_array_lib()
    l_starts = arrlib.arange(L_out) * stride
    
    # 预计算所有内核块的索引范围
    k_range = arrlib.arange(K) * dilation
    
    # 一次性生成所有需要的索引
    all_indices = l_starts[:, np.newaxis] + k_range
    
    # 确保索引在有效范围内
    all_indices = arrlib.clip(all_indices, 0, padded_input.shape[2] - 1)
    
    # 使用高级索引一次性提取所有内核块
    # 首先创建一个形状为 (N, C, L_out, K) 的临时张量
    unfolded_temp = padded_input[:, :, all_indices]
    
    # 直接重塑为 (N*C, L_out, K) 形状，避免不必要的转置
    unfolded_input_reshaped = unfolded_temp.reshape(N * C, L_out, K)
    
    # 计算平均值
    if divisor_override is not None:
        # 使用指定的除数，确保除数与输入张量具有相同的数据类型
        divisor_tensor = tensor(divisor_override, dtype=input.dtype,device=input.device)
        avg_values = unfolded_input_reshaped.sum(dim=2) / divisor_tensor
    elif not count_include_pad:
        # 不包含填充区域时，使用实际元素数作为除数
        # 创建一个与输入相同的张量来计算有效区域
        input_mask = ones_like(input)
        
        # 对掩码进行填充，与输入的填充方式相同
        if padding != 0 or ceil_mode:
            mask_padded_shape = (N, C, final_padded_length)
            mask_padded_input = zeros(mask_padded_shape, dtype=input.dtype,device=input.device)
            mask_data_index = (slice(None), slice(None), slice(padding, padding + L_in))
            mask_padded_input.setat_(mask_data_index, input_mask)
        else:
            mask_padded_input = input_mask
        
        # 使用与输入相同的优化方式展开掩码
        # 先创建形状为 (N, C, L_out, K) 的临时张量
        mask_unfolded_temp = mask_padded_input[:, :, all_indices]
        
        # 直接重塑为 (N*C, L_out, K) 形状，避免不必要的转置
        mask_unfolded_reshaped = mask_unfolded_temp.reshape(N * C, L_out, K)
        
        # 计算每个输出位置的有效元素数
        effective_counts = mask_unfolded_reshaped.sum(dim=2)
        
        # 计算平均值，使用有效元素数作为除数
        avg_values = unfolded_input_reshaped.sum(dim=2) / effective_counts
    else:
        # 包含填充区域时的处理，与avg_pool2d保持一致
        # 辅助函数：向量化计算所有窗口位置的有效元素数
        def compute_effective_counts_vectorized():
            # 生成所有窗口的起始位置
            l_starts = arrlib.arange(L_out) * stride
            
            # 计算每个窗口的结束位置
            l_ends = l_starts + effective_K
            
            # 显式padding后的输入边界
            padded_l_start = 0
            padded_l_end = L_in + 2 * padding
            
            # 计算窗口与显式padding后输入的交集（向量化操作）
            valid_l_starts = arrlib.maximum(l_starts, padded_l_start)
            valid_l_ends = arrlib.minimum(l_ends, padded_l_end)
            valid_l = arrlib.maximum(0, valid_l_ends - valid_l_starts)
            
            return valid_l
                
        # 向量化计算所有窗口的有效元素数
        all_counts = compute_effective_counts_vectorized()
        
        # 将numpy数组转换为Tensor并扩展到批次和通道维度
        effective_counts = tensor(all_counts, dtype=input.dtype,device=input.device)
        effective_counts = effective_counts.unsqueeze(0)  # 添加维度变为(1, L_out)
        effective_counts = effective_counts.expand(N*C, -1)  # 扩展到(N*C, L_out)
        
        # 计算平均值
        avg_values = unfolded_input_reshaped.sum(dim=2) / effective_counts
    
    # 将输出重塑为 (N, C, L_out)
    output = avg_values.reshape(N, C, L_out)
    return output

def avg_pool2d(input: TN, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) -> TN:
    r"""对输入张量应用2D平均池化。
    
    参数:
        input (TN): 输入张量，形状为 (N, C, H_in, W_in)
        kernel_size (int 或 tuple): 池化核大小
        stride (int 或 tuple, 可选): 步长。默认值: None（与kernel_size相同）
        padding (int 或 tuple, 可选): 两侧的隐式零填充。默认值: 0
        ceil_mode (bool, 可选): 是否使用向上取整计算输出形状。默认值: False
        count_include_pad (bool, 可选): 是否在计算平均值时包含填充区域。默认值: True
        divisor_override (int, 可选): 如果指定，将使用此值代替kernel_size的乘积进行除法。默认值: None
    
    返回:
        TN: 池化后的张量，形状为 (N, C, H_out, W_out)
    """
    if not isinstance(input, TN):
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input)}")
    
    if input.ndim != 4:
        raise ValueError(f"Expected 4D tensor for input, but got {input.ndim}D tensor")
    
    # 参数处理
    kernel_size = _pair(kernel_size)
    padding = _pair(padding)
    stride = _pair(stride) if stride is not None else kernel_size
    dilation = _pair(1)
    
    N, C, H_in, W_in = input.shape
    kh, kw = kernel_size
        
    # 使用unfold2d函数提取滑动块，展开为(N, C, kernel_size[0], kernel_size[1], H_out, W_out)
    # 支持ceil_mode、check_pad，并获取H_out和W_out
    unfolded_input, H_out, W_out = unfold2d(input, 
                                            kernel_size=kernel_size, 
                                            dilation=dilation,
                                            padding=padding, 
                                            padvalue=0.0, 
                                            stride=stride, 
                                            ceil_mode=ceil_mode,
                                            check_pad=True)
    
    # 直接返回池化函数所需的形状 (N*C, kh*kw, H_out*W_out)
    # 只需要一次reshape即可
    unfolded_result = unfolded_input.reshape(N*C, kh*kw, H_out*W_out)

    # 计算平均值
    if divisor_override is not None:
        # 使用指定的除数，确保除数与输入张量具有相同的数据类型
        divisor_tensor = tensor(divisor_override, dtype=input.dtype,device=input.device)
        avg_values = unfolded_result.sum(dim=1) / divisor_tensor
    elif not count_include_pad:
        # 不包含填充区域时，使用实际元素数作为除数
        # 创建一个与输入相同的张量来计算有效区域
        input_mask = ones_like(input)
        
        # 对掩码进行unfold操作，与输入的unfold操作相同，直接返回池化所需形状
        unfolded_mask, H_out, W_out = unfold2d(input_mask, 
                                            kernel_size=kernel_size, 
                                            dilation=dilation, 
                                            padding=padding, 
                                            padvalue=0.0, 
                                            stride=stride, 
                                            ceil_mode=ceil_mode)
        
        unfolded_mask_reshaped = unfolded_mask.reshape(N*C, kh*kw, H_out*W_out)
        
        # 计算每个输出位置的有效元素数
        effective_counts = unfolded_mask_reshaped.sum(dim=1)
        
        # 计算平均值，使用有效元素数作为除数
        avg_values = unfolded_result.sum(dim=1) / effective_counts
    else:
        # count_include_pad=True时的处理
        # 辅助函数：向量化计算所有窗口位置的有效元素数
        def compute_effective_counts_vectorized():
            # 生成所有窗口的坐标网格 (H_out, W_out)
            arrlib = input._get_array_lib()
            i_indices, j_indices = arrlib.meshgrid(arrlib.arange(H_out), arrlib.arange(W_out), indexing='ij')
            
            # 计算每个窗口的起始和结束位置
            h_start = i_indices * stride[0]
            h_end = h_start + kh
            w_start = j_indices * stride[1]
            w_end = w_start + kw
            
            # 显式padding后的输入边界
            padded_h_start = 0
            padded_h_end = H_in + 2 * padding[0]
            padded_w_start = 0
            padded_w_end = W_in + 2 * padding[1]
            
            # 计算窗口与显式padding后输入的交集
            valid_h_start = arrlib.maximum(h_start, padded_h_start)
            valid_h_end = arrlib.minimum(h_end, padded_h_end)
            valid_h = arrlib.maximum(0, valid_h_end - valid_h_start)
            
            valid_w_start = arrlib.maximum(w_start, padded_w_start)
            valid_w_end = arrlib.minimum(w_end, padded_w_end)
            valid_w = arrlib.maximum(0, valid_w_end - valid_w_start)
            
            # 计算每个窗口的有效元素数
            counts = valid_h * valid_w
            
            # 将结果展平为1D数组
            return counts.flatten()
        
        # 使用向量化方法计算所有窗口的有效元素数
        all_counts = compute_effective_counts_vectorized()
        
        # 创建结果张量并重复到所有N*C批次
        effective_counts = tensor(all_counts, dtype=input.dtype,device=input.device)
        effective_counts = effective_counts.unsqueeze(0)  # 添加一个维度，形状变为(1, total_windows)
        effective_counts = effective_counts.expand(N*C, -1)  # 扩展到N*C批次
        
        # 计算平均值
        avg_values = unfolded_result.sum(dim=1) / effective_counts
    
    # 将输出重塑为 (N, C, H_out, W_out)
    output = avg_values.reshape(N, C, H_out, W_out)
    
    return output

def avg_pool3d(input: TN, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) -> TN:
    r"""对输入张量应用3D平均池化。
    
    参数:
        input (TN): 输入张量，形状为 (N, C, D_in, H_in, W_in)
        kernel_size (int 或 tuple): 池化核大小
        stride (int 或 tuple, 可选): 步长。默认值: None（与kernel_size相同）
        padding (int 或 tuple, 可选): 各侧的隐式零填充。默认值: 0
        ceil_mode (bool, 可选): 是否使用向上取整计算输出形状。默认值: False
        count_include_pad (bool, 可选): 是否在计算平均值时包含填充区域。默认值: True
        divisor_override (int, 可选): 如果指定，将使用此值代替kernel_size的乘积进行除法。默认值: None
    
    返回:
        TN: 池化后的张量，形状为 (N, C, D_out, H_out, W_out)
    """
    if not isinstance(input, TN):
        raise TypeError(f"Expected input type to be TN tensor, but received type: {type(input)}")
    
    if input.ndim != 5:
        raise ValueError(f"Expected 5D tensor for input, but got {input.ndim}D tensor")
    
    # 参数处理
    kernel_size = _triple(kernel_size)
    padding = _triple(padding)
    stride = _triple(stride) if stride is not None else kernel_size
    dilation = _triple(1)
    
    N, C, D_in, H_in, W_in = input.shape
    kd, kh, kw = kernel_size
        
    # 使用unfold3d函数提取滑动块，展开为(N, C, kd, kh, kw, D_out, H_out, W_out)
    # 支持ceil_mode、check_pad，并获取D_out、H_out和W_out
    unfolded_input, D_out, H_out, W_out = unfold3d(input, 
                                                  kernel_size=kernel_size, 
                                                  dilation=dilation,
                                                  padding=padding, 
                                                  padvalue=0.0, 
                                                  stride=stride, 
                                                  ceil_mode=ceil_mode,
                                                  check_pad=True)
    
    # 直接返回池化函数所需的形状 (N*C, kd*kh*kw, D_out*H_out*W_out)
    # 只需要一次reshape即可
    unfolded_result = unfolded_input.reshape(N*C, kd*kh*kw, D_out*H_out*W_out)

    # 计算平均值
    if divisor_override is not None:
        # 使用指定的除数，确保除数与输入张量具有相同的数据类型
        divisor_tensor = tensor(divisor_override, dtype=input.dtype,device=input.device)
        avg_values = unfolded_result.sum(dim=1) / divisor_tensor
    elif not count_include_pad:
        # 不包含填充区域时，使用实际元素数作为除数
        # 创建一个与输入相同的张量来计算有效区域
        input_mask = ones_like(input)
        
        # 对掩码进行unfold操作，与输入的unfold操作相同，直接返回池化所需形状
        unfolded_mask, D_out, H_out, W_out = unfold3d(input_mask, 
                                                      kernel_size=kernel_size, 
                                                      dilation=dilation, 
                                                      padding=padding, 
                                                      padvalue=0.0, 
                                                      stride=stride, 
                                                      ceil_mode=ceil_mode)
        
        unfolded_mask_reshaped = unfolded_mask.reshape(N*C, kd*kh*kw, D_out*H_out*W_out)
        
        # 计算每个输出位置的有效元素数
        effective_counts = unfolded_mask_reshaped.sum(dim=1)
        
        # 计算平均值，使用有效元素数作为除数
        avg_values = unfolded_result.sum(dim=1) / effective_counts
    else:
        # count_include_pad=True时的处理
        # 辅助函数：向量化计算所有窗口位置的有效元素数
        def compute_effective_counts_vectorized():
            # 生成所有窗口的坐标网格
            arrlib = input._get_array_lib()
            d_indices, h_indices, w_indices = arrlib.meshgrid(arrlib.arange(D_out), arrlib.arange(H_out), arrlib.arange(W_out), indexing='ij')
            
            # 计算每个窗口的起始和结束位置
            d_start = d_indices * stride[0]
            d_end = d_start + kd
            h_start = h_indices * stride[1]
            h_end = h_start + kh
            w_start = w_indices * stride[2]
            w_end = w_start + kw
            
            # 显式padding后的输入边界
            padded_d_start = 0
            padded_d_end = D_in + 2 * padding[0]
            padded_h_start = 0
            padded_h_end = H_in + 2 * padding[1]
            padded_w_start = 0
            padded_w_end = W_in + 2 * padding[2]
            
            # 计算窗口与显式padding后输入的交集
            valid_d_start = arrlib.maximum(d_start, padded_d_start)
            valid_d_end = arrlib.minimum(d_end, padded_d_end)
            valid_d = arrlib.maximum(0, valid_d_end - valid_d_start)
            
            valid_h_start = arrlib.maximum(h_start, padded_h_start)
            valid_h_end = arrlib.minimum(h_end, padded_h_end)
            valid_h = arrlib.maximum(0, valid_h_end - valid_h_start)
            
            valid_w_start = arrlib.maximum(w_start, padded_w_start)
            valid_w_end = arrlib.minimum(w_end, padded_w_end)
            valid_w = arrlib.maximum(0, valid_w_end - valid_w_start)
            
            # 计算每个窗口的有效元素数
            counts = valid_d * valid_h * valid_w
            
            # 将结果展平为1D数组
            return counts.flatten()
        
        # 使用向量化方法计算所有窗口的有效元素数
        all_counts = compute_effective_counts_vectorized()
        
        # 创建结果张量并重复到所有N*C批次
        effective_counts = tensor(all_counts, dtype=input.dtype,device=input.device)
        effective_counts = effective_counts.unsqueeze(0)  # 添加一个维度，形状变为(1, total_windows)
        effective_counts = effective_counts.expand(N*C, -1)  # 扩展到N*C批次
        
        # 计算平均值
        avg_values = unfolded_result.sum(dim=1) / effective_counts
    
    # 将输出重塑为 (N, C, D_out, H_out, W_out)
    output = avg_values.reshape(N, C, D_out, H_out, W_out)

    return output

def batch_norm(input: TN, running_mean: Optional[TN] = None, running_var: Optional[TN] = None, 
              weight: Optional[TN] = None, bias: Optional[TN] = None, training: bool = False, 
              momentum: float = 0.1, eps: float = 1e-5) -> TN:
    """
    对输入张量应用批量归一化。
    
    参数:
        input (TN): 输入张量，形状为 (N, C), (N, C, L), (N, C, H, W) 或 (N, C, D, H, W)
        running_mean (TN, 可选): 运行时均值，形状为 (C,)。默认值: None
        running_var (TN, 可选): 运行时方差，形状为 (C,)。默认值: None
        weight (TN, 可选): 可学习的缩放参数γ，形状为 (C,)。默认值: None
        bias (TN, 可选): 可学习的偏移参数β，形状为 (C,)。默认值: None
        training (bool, 可选): 是否为训练模式。默认值: False
        momentum (float, 可选): 运行时统计量的动量。默认值: 0.1
        eps (float, 可选): 数值稳定性的小常数。默认值: 1e-5
    
    返回:
        TN: 归一化后的张量，形状与输入相同
    
    形状转换:
        - 输入: (N, C), (N, C, L), (N, C, H, W) 或 (N, C, D, H, W)
        - 输出: 与输入形状相同
    """
    if not isinstance(input, TN):
        raise TypeError(f"Expected input to be TN tensor, but received type: {type(input)}")
    
    if input.ndim not in [2, 3, 4, 5]:
        raise ValueError(f"Expected 2D, 3D, 4D or 5D tensor for input, but got {input.ndim}D tensor")
    
    if weight is not None:
        if not isinstance(weight, TN):
            raise TypeError(f"Expected weight to be TN tensor, but received type: {type(weight)}")
        elif weight.device != input.device:
            raise ValueError(f"Expected weight to have device {input.device}, but got {weight.device}")
    
    if bias is not None:
        if not isinstance(bias, TN):
            raise TypeError(f"Expected bias to be TN tensor, but received type: {type(bias)}")
        elif bias.device != input.device:
            raise ValueError(f"Expected bias to have device {input.device}, but got {bias.device}")
    
    # 获取通道数
    C = input.shape[1]
    
    # 根据输入维度确定计算统计量的维度
    if input.ndim == 2:
        # 输入形状: (N, C)
        dims = (0,)  # type: ignore  # 沿着批次维度计算统计量
    elif input.ndim == 3:
        # 输入形状: (N, C, L)
        dims = (0, 2)  # type: ignore  # 沿着批次和序列长度维度计算统计量
    elif input.ndim == 4:
        # 输入形状: (N, C, H, W)
        dims = (0, 2, 3)  # type: ignore  # 沿着批次、高度和宽度维度计算统计量
    else:  # input.ndim == 5
        # 输入形状: (N, C, D, H, W)
        dims = (0, 2, 3, 4)  # type: ignore  # 沿着批次、深度、高度和宽度维度计算统计量
    
    # 训练模式：使用当前批次的统计量
    if training:
        # 计算当前批次的均值和方差
        mean = input.mean(dim=dims, keepdim=False)  # 形状: (C,)
        var_biased = input.var(dim=dims, unbiased=False, keepdim=False)  # 形状: (C,)
        
        # 计算无偏估计用于更新运行时统计量（与PyTorch保持一致）
        # PyTorch使用无偏估计(ddof=1)来更新运行时统计量
        total_elements = input.numel()  # 总元素数
        elements_per_channel = total_elements // C  # 每个通道的元素数
        
        if elements_per_channel > 1:
            unbiased_correction = elements_per_channel / (elements_per_channel - 1)
            var_unbiased = var_biased * unbiased_correction
        else:
            var_unbiased = var_biased
        
        # 更新运行时统计量
        if running_mean is not None and running_var is not None:
            # 使用指数移动平均更新运行时统计量
            # 注意：PyTorch使用无偏方差来更新运行时统计量
            running_mean_new = (1 - momentum) * running_mean + momentum * mean
            running_var_new = (1 - momentum) * running_var + momentum * var_unbiased
            
            # 更新运行时统计量（原地操作）
            running_mean.data = running_mean_new.data
            running_var.data = running_var_new.data
        
        # 前向传播使用有偏方差（与PyTorch保持一致）
        var = var_biased
    else:
        # 评估模式：使用运行时统计量
        if running_mean is None or running_var is None:
            raise ValueError("running_mean and running_var must be provided in evaluation mode")
        mean = running_mean
        var = running_var
    
    # 计算归一化
    # 根据输入维度重塑统计量以匹配输入形状
    if input.ndim == 2:
        # 对于2D输入
        mean_reshaped = mean.reshape(1, -1)  # 形状: (1, C)
        var_reshaped = var.reshape(1, -1)    # 形状: (1, C)
    elif input.ndim == 3:
        # 对于3D输入
        mean_reshaped = mean.reshape(1, -1, 1)  # 形状: (1, C, 1)
        var_reshaped = var.reshape(1, -1, 1)    # 形状: (1, C, 1)
    elif input.ndim == 4:
        # 对于4D输入
        mean_reshaped = mean.reshape(1, -1, 1, 1)  # 形状: (1, C, 1, 1)
        var_reshaped = var.reshape(1, -1, 1, 1)    # 形状: (1, C, 1, 1)
    else:  # input.ndim == 5
        # 对于5D输入
        mean_reshaped = mean.reshape(1, -1, 1, 1, 1)  # 形状: (1, C, 1, 1, 1)
        var_reshaped = var.reshape(1, -1, 1, 1, 1)    # 形状: (1, C, 1, 1, 1)
    
    # 归一化: (x - mean) / sqrt(var + eps)
    inv_std = 1.0 / sqrt(var_reshaped + eps)
    normalized = (input - mean_reshaped) * inv_std
    
    # 应用仿射变换
    if weight is not None and bias is not None:
        # 根据输入维度重塑权重和偏置以匹配输入形状
        if input.ndim == 2:
            # 对于2D输入
            weight_reshaped = weight.reshape(1, -1)  # 形状: (1, C)
            bias_reshaped = bias.reshape(1, -1)     # 形状: (1, C)
        elif input.ndim == 3:
            # 对于3D输入
            weight_reshaped = weight.reshape(1, -1, 1)  # 形状: (1, C, 1)
            bias_reshaped = bias.reshape(1, -1, 1)     # 形状: (1, C, 1)
        elif input.ndim == 4:
            # 对于4D输入
            weight_reshaped = weight.reshape(1, -1, 1, 1)  # 形状: (1, C, 1, 1)
            bias_reshaped = bias.reshape(1, -1, 1, 1)     # 形状: (1, C, 1, 1)
        else:  # input.ndim == 5
            # 对于5D输入
            weight_reshaped = weight.reshape(1, -1, 1, 1, 1)  # 形状: (1, C, 1, 1, 1)
            bias_reshaped = bias.reshape(1, -1, 1, 1, 1)     # 形状: (1, C, 1, 1, 1)
        
        return normalized * weight_reshaped + bias_reshaped
    else:
        return normalized


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    """Layer normalization function that normalizes over specified dimensions.
    Compatible with torch.nn.functional.layer_norm.
    
    Args:
        input: The input tensor to normalize.
        normalized_shape: An integer or tuple specifying the dimensions to normalize over.
        weight: Optional weight tensor (gamma) for affine transformation.
        bias: Optional bias tensor (beta) for affine transformation.
        eps: Small value added to variance to avoid division by zero.
    """
    # 计算需要归一化的维度
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    
    # 确定归一化的维度索引
    if len(normalized_shape) > len(input.shape):
        raise ValueError(f"normalized_shape {normalized_shape} is longer than input shape {input.shape}")
    
    start_dim = len(input.shape) - len(normalized_shape)
    
    # 验证输入形状是否与 normalized_shape 匹配
    for i, dim in enumerate(normalized_shape):
        if input.shape[start_dim + i] != dim:
            raise ValueError(f"normalized_shape {normalized_shape} does not match input shape {input.shape} starting from dimension {start_dim}")
    
    # 计算归一化维度
    dims = tuple(range(start_dim, len(input.shape)))
    
    # 计算均值和方差
    mean_val = input.mean(dim=dims, keepdim=True)
    var_val = input.var(dim=dims, unbiased=False, keepdim=True)
    
    # 归一化
    normalized = (input - mean_val) / sqrt(var_val + eps)
    
    # 应用仿射变换
    if weight is not None and bias is not None:
        # 重塑 weight 和 bias 以匹配输入形状
        reshape_size = (1,) * start_dim + normalized_shape
        weight_reshaped = weight.reshape(reshape_size)
        bias_reshaped = bias.reshape(reshape_size)
        
        return normalized * weight_reshaped + bias_reshaped
    else:
        return normalized

def embedding(
    input: TN,
    weight: TN,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> TN:
    """
    从嵌入矩阵中查找输入索引的嵌入向量（使用自定义梯度跟踪实现）。
    
    参数:
        input (TN): 包含索引的张量，形状为任意维度
        weight (TN): 嵌入矩阵，形状为 (num_embeddings, embedding_dim)
        padding_idx (int, optional): 如果指定，该索引的嵌入向量不参与梯度计算，
                                     且在训练过程中保持不变。默认为None
        max_norm (float, optional): 如果指定，所有嵌入向量的范数超过max_norm时，
                                    将被重归一化到max_norm。默认为None
        norm_type (float, optional): 计算范数时使用的p值，默认为2（L2范数）
        scale_grad_by_freq (bool, optional): 如果为True，梯度将按mini-batch中每个词的频率进行缩放。
                                             默认为False
        sparse (bool, optional): 如果为True，权重的梯度将是稀疏张量。默认为False
    
    返回:
        TN: 输出张量，形状为 (*, embedding_dim)，其中*是输入的形状
    
    注意:
        - 该实现使用自定义的梯度跟踪机制，更便于支持scale_grad_by_freq参数
        - 支持任意维度的输入张量
        - 输入张量的数据类型应为整数类型（通常是long）
        - 如果设置了padding_idx，它必须在[0, num_embeddings-1]范围内
        - 自动支持梯度计算
        - 与PyTorch行为一致：不会自动将padding_idx的输出设为0，需要手动将权重中对应的值设为0
        - 当前版本支持scale_grad_by_freq参数
        - 当前版本不支持sparse参数（会忽略该参数并使用密集存储）
    """
    if not isinstance(input, TN):
        raise TypeError(f"Expected input to be TN tensor, but received type: {type(input)}")
    if not isinstance(weight, TN):
        raise TypeError(f"Expected weight type to be TN tensor, but received type: {type(weight)}")
    if input.device != weight.device:
        raise ValueError("input and weight must have the same device")
    
    # 检查输入数据类型是否为整数类型
    if input.dtype.kind not in ['i', 'u']:
        raise TypeError(f"Expected input tensor to have integer type, but received: {input.dtype}")
    
    # 检查权重矩阵的维度
    if weight.ndim != 2:
        raise ValueError(f"Expected weight to be 2-dimensional, but received: {weight.ndim} dimensions")
    
    arrlib = input._get_array_lib()
    num_embeddings, embedding_dim = weight.shape
    
    # 处理padding_idx
    if padding_idx is not None:
        if padding_idx is not None and padding_idx >= 0:  # type: ignore
            if padding_idx >= num_embeddings:  # type: ignore
                raise ValueError(f"padding_idx ({padding_idx}) must be within num_embeddings ({num_embeddings})")
        elif padding_idx is not None and padding_idx < 0:  # type: ignore
            padding_idx = num_embeddings + padding_idx
            if padding_idx < 0:  # type: ignore
                raise ValueError(f"padding_idx ({padding_idx}) must be within num_embeddings ({num_embeddings})")
    
    weight_data = weight.data

    # 处理max_norm
    if max_norm is not None:
        # 使用张量操作计算每个嵌入向量的范数，保持维度
        norms = arrlib.linalg.norm(weight.data, ord=norm_type, axis=1, keepdims=True)
        # 创建掩码，找出范数超过max_norm的嵌入向量
        mask = norms > max_norm  # type: ignore
        
        # 如果有需要重归一化的嵌入向量
        mask = norms > max_norm  # type: ignore
        # 计算重归一化后的嵌入向量
        normalized = weight.data / norms * max_norm
        # 使用where操作更新权重
        weight_data = arrlib.where(mask, normalized, weight.data)
    
    # PyTorch不会自动将padding_idx的权重设为0，只是不计算其梯度
    # 因此这里不做额外处理，保持与PyTorch行为一致
    
    # 处理scale_grad_by_freq参数，计算频率信息
    freq = None
    if scale_grad_by_freq:
        # 统计每个索引出现的频率
        input_flat = input.flatten()
        unique_indices, counts = unique(input_flat, return_counts=True)  # type: ignore
        
        # 创建频率数组
        freq = ones(num_embeddings,device=input.device)
        
        # 更新出现过的索引的频率
        for idx, count in zip(unique_indices, counts):
            if idx != padding_idx:  # 不考虑padding_idx
                freq[idx] = count
            
    # 获取输入的数据    
    input_data = input.data
    
    # 使用numpy的索引操作直接获取嵌入向量
    output_data = weight_data[input_data]
    
    # 创建输出张量
    requires_grad = (is_grad_enabled() and weight.requires_grad)
    output = tensor(output_data, device=input.device, requires_grad =requires_grad )
    output.is_leaf = not requires_grad

    # 如果需要跟踪梯度
    if requires_grad:
        # 设置fromvars，记录参与计算的变量
        output.fromvars = (weight,)        
        # 设置parms，记录计算所需的参数
        output.parms = ((input, scale_grad_by_freq, freq),)
                
        # 定义反向传播函数
        def _embedding_backward(result_tensor: TN, i: int) -> TN:
            """
            embedding的反向传播函数
            
            参数:
                result_tensor: 结果张量
                i: 当前处理的变量索引
            
            返回:
                梯度张量
            """
            # 获取相关变量和参数
            weight_var = result_tensor.fromvars[i]
            input_tensor,scale_grad,freq_tensor = result_tensor.parms[i]
            grad_value = result_tensor.grad_value
            
            # 初始化权重梯度
            weight_grad = zeros_like(weight_var)
            
            # 处理scale_grad_by_freq
            if scale_grad:
                # 对梯度进行缩放
                freq_values = freq_tensor[input_tensor]
                # 将频率张量扩展一个维度，以便正确广播到嵌入维度
                freq_values_expanded = freq_values.unsqueeze(-1)
                scaled_grad = grad_value / freq_values_expanded
                # 将缩放后的梯度添加到权重梯度中
                weight_grad = weight_grad.addat(input_tensor, scaled_grad)
            else:
                # 直接将梯度添加到权重梯度中
                weight_grad = weight_grad.addat(input_tensor, grad_value)
            
            # 将padding_idx的梯度设为0（与PyTorch行为一致）
            if padding_idx is not None:
                weight_grad[padding_idx] = 0.0
            
            return weight_grad
        
        # 设置gradfuncs，记录反向传播函数
        output.gradfuncs = (_embedding_backward,)
    
    return output

def embedding2(
    input: TN,
    weight: TN,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> TN:
    """
    从嵌入矩阵中查找输入索引的嵌入向量。
    
    参数:
        input (TN): 包含索引的张量，形状为任意维度
        weight (TN): 嵌入矩阵，形状为 (num_embeddings, embedding_dim)
        padding_idx (int, optional): 如果指定，该索引的嵌入向量不参与梯度计算，
                                     且在训练过程中保持不变。默认为None
        max_norm (float, optional): 如果指定，所有嵌入向量的范数超过max_norm时，
                                    将被重归一化到max_norm。默认为None
        norm_type (float, optional): 计算范数时使用的p值，默认为2（L2范数）
        scale_grad_by_freq (bool, optional): 如果为True，梯度将按mini-batch中每个词的频率进行缩放。
                                             默认为False
        sparse (bool, optional): 如果为True，权重的梯度将是稀疏张量。默认为False
    
    返回:
        TN: 输出张量，形状为 (*, embedding_dim)，其中*是输入的形状
    
    形状:
        - 输入: LongTensor类型的任意形状张量，包含要提取的索引
        - 权重: 浮点数类型的嵌入矩阵，形状为 (V, embedding_dim)，
               其中V = max_index + 1，embedding_dim = 嵌入大小
        - 输出: (*, embedding_dim)，其中*是输入的形状
    
    示例:
        >>> # 批量大小为2，每个样本有4个索引
        >>> input = tensor([[1, 2, 4, 5]], dtype='long')
        >>> # 包含10个3维嵌入向量的嵌入矩阵
        >>> embedding_matrix = tensor(np.random.rand(10, 3), dtype='float32')
        >>> output = embedding(input, embedding_matrix)
        >>> # 结果形状: (1, 4, 3)
        
        >>> # 使用padding_idx的示例（与PyTorch行为一致）
        >>> weights = tensor(np.random.rand(10, 3), dtype='float32')
        >>> input = tensor([[0, 2, 0, 5]], dtype='long')
        >>> # PyTorch不会自动将padding_idx的输出设为0，需要手动处理
        >>> weights[0, :] = 0  # 手动将padding_idx对应的嵌入设为0
        >>> output = embedding(input, weights, padding_idx=0)
        >>> # 结果中索引0的嵌入将保持为全0（因为权重已手动设为0）
    
    注意:
        - 支持任意维度的输入张量
        - 输入张量的数据类型应为整数类型（通常是long）
        - 如果设置了padding_idx，它必须在[0, num_embeddings-1]范围内
        - max_norm参数会对权重矩阵进行重归一化
        - 自动支持梯度计算
        - 与PyTorch行为一致：不会自动将padding_idx的输出设为0，需要手动将权重中对应的值设为0
        - 当前版本暂不支持scale_grad_by_freq和sparse参数
    """
    if not isinstance(weight, TN):
        raise TypeError(f"Expected weight type to be TN tensor, but received type: {type(weight)}")
    if not isinstance(input, TN):
        raise TypeError(f"Expected input to be TN tensor, but received type: {type(input)}")
    if input.device != weight.device:
        raise ValueError("input and weight must have the same device")
    
    # 检查输入数据类型是否为整数类型
    # 使用dtype.kind检查，'i'表示有符号整数，'u'表示无符号整数
    if input.dtype.kind not in ['i', 'u']:
        raise TypeError(f"Expected input tensor to have integer type, but received: {input.dtype}")
    
    # 检查权重矩阵的维度
    if weight.ndim != 2:
        raise ValueError(f"Expected weight to be 2-dimensional, but received: {weight.ndim} dimensions")
    
    num_embeddings, embedding_dim = weight.shape
    
    # 处理padding_idx
    if padding_idx is not None:
        if padding_idx is not None and padding_idx >= 0:  # type: ignore
            if padding_idx >= num_embeddings:  # type: ignore
                raise ValueError(f"padding_idx ({padding_idx}) must be within num_embeddings ({num_embeddings})")
        elif padding_idx is not None and padding_idx < 0:  # type: ignore
            padding_idx = num_embeddings + padding_idx
            if padding_idx < 0:  # type: ignore
                raise ValueError(f"padding_idx ({padding_idx}) must be within num_embeddings ({num_embeddings})")
    
    # 处理max_norm
    if max_norm is not None:
        # 使用张量操作计算每个嵌入向量的范数
        # 计算Lp范数，axis=1表示按行计算，keepdims=True保持维度
        norms = weight.norm(norm_type, dim=1, keepdim=True)
        # 创建掩码，找出范数超过max_norm的嵌入向量
        mask = norms > max_norm  # type: ignore
        
        # 如果有需要重归一化的嵌入向量
        if mask.any():
            # 计算重归一化后的嵌入向量
            normalized = weight / norms * max_norm
            # 使用where操作更新权重，保持梯度跟踪
            weight = where(mask, normalized, weight)
    
    # 确保padding_idx的梯度为0（与PyTorch行为一致）
    # 这里使用一个小技巧：将padding_idx对应的权重行分离出来，不参与梯度计算
    if padding_idx is not None and weight.requires_grad:
        # 创建一个与weight相同形状的临时张量
        # 非padding_idx的行保持原样，padding_idx的行使用detach()版本
        # 这样，反向传播时，padding_idx对应的行的梯度不会影响原weight
        temp_weight = where(
            arange(num_embeddings, requires_grad=False)[:, None] == padding_idx,
            weight[padding_idx].detach().broadcast_to(weight.shape[1:]),
            weight
        )
        
        # 使用临时权重张量进行嵌入查找
        output = temp_weight[input]
    else:
        # 正常进行嵌入查找
        output = weight[input]
    
    # PyTorch不会自动将padding_idx的输出设为0，只是不计算其梯度
    # 因此这里不做额外处理，保持与PyTorch行为一致
    
    return output
