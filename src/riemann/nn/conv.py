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
Convolution and Pooling Modules Implementation for the Riemann Library

This module provides implementations of convolution and pooling operations as neural network modules
for the Riemann library. These modules are essential building blocks for convolutional neural networks
(CNNs) and other deep learning architectures that process spatial or sequential data.

Convolution modules apply learnable filters to input data to extract features:
- Conv1d: 1D convolution for sequential or temporal data (e.g., audio, text)
- Conv2d: 2D convolution for image data with spatial dimensions
- Conv3d: 3D convolution for volumetric data (e.g., video, medical images)

Pooling modules reduce spatial dimensions and provide translation invariance:
- MaxPool1d/2d/3d: Maximum pooling, selecting maximum values in pooling regions
- AvgPool1d/2d/3d: Average pooling, computing average values in pooling regions

All modules inherit from the Module base class and implement a forward method
that applies the respective operation from the functional module. The interface
is fully compatible with PyTorch's nn.Module API for easy migration.
"""

from .module import *
from .functional import *

# ==================== 卷积模块 ====================

class Conv1d(Module):
    """一维卷积模块 (1D Convolution Layer)
    
    对输入信号应用一维卷积操作，常用于处理序列数据如音频、文本等时间序列数据。
    通过在输入上滑动可学习的卷积核来提取局部特征模式。
    
    数学公式::
    
        out(N_i, C_{out_j}) = bias(C_{out_j}) + 
        sum_{k=0}^{C_{in}-1} weight(C_{out_j}, k) * input(N_i, k)
        
        其中 * 表示有效的一维互相关操作
    
    Args:
        in_channels (int): 输入通道数。输入张量的通道数，对于序列数据通常为1。
        out_channels (int): 输出通道数。卷积核的数量，决定输出特征图的通道数。
        kernel_size (int or tuple): 卷积核大小。卷积核的长度，控制感受野大小。
        stride (int or tuple, optional): 卷积步长。控制卷积核滑动的步距。
            默认值: 1
        padding (int or tuple, optional): 填充大小。在输入两侧添加的零填充数量。
            默认值: 0
        dilation (int or tuple, optional): 膨胀率。卷积核元素之间的间距。
            默认值: 1
        groups (int, optional): 分组数。控制输入和输出通道之间的连接模式。
            默认值: 1
        bias (bool, optional): 是否使用偏置项。是否为每个输出通道添加可学习的偏置。
            默认值: True
        dtype (np.dtype, optional): 张量的数据类型。
            默认值: None
        device (str or int or Device, optional): 张量所在的设备。
            默认值: None
    
    Shape:
        - Input: 形状为 (N, C_in, L) 的张量
        - Output: 形状为 (N, C_out, L_out) 的张量
        - Weight: 形状为 (C_out, C_in // groups, K) 的张量
        - Bias: 形状为 (C_out,) 的张量
    
    Examples::
    
        >>> # 音频信号处理示例
        >>> m = Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        >>> input = rm.randn(8, 1, 100)  # batch_size=8, channels=1, length=100
        >>> output = m(input)
        >>> print(output.shape)  # (8, 16, 100)
        
        >>> # 文本序列处理示例
        >>> m = Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=2)
        >>> input = rm.randn(32, 64, 50)  # batch_size=32, embed_dim=64, seq_len=50
        >>> output = m(input)
        >>> print(output.shape)  # (32, 128, 23)
    
    Note:
        Conv1d的特点和用途：
        - 主要处理一维序列数据，如音频波形、文本序列、时间序列数据
        - 能够捕获局部时间依赖关系和模式
        - 参数共享机制大大减少了模型参数量
        - 平移不变性使模型能够识别不同位置的模式
        - 常与RNN、Transformer等架构结合使用
        - 在自然语言处理中用于n-gram特征提取
        - 在音频处理中用于频谱特征提取
        - 输出长度计算公式：L_out = floor((L_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, dtype=None, device=None):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # 初始化权重参数
        weight_shape = (out_channels, in_channels // groups, kernel_size if isinstance(kernel_size, int) else kernel_size[0])
        self.weight = Parameter(randn(weight_shape, dtype=dtype, device=device) * 0.1)
        
        # 初始化偏置参数
        if bias:
            self.bias = Parameter(zeros(out_channels, dtype=dtype, device=device))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, in_channels, length)
            
        Returns:
            输出张量，形状为 (batch_size, out_channels, output_length)
        """
        return conv1d(x, self.weight, self.bias, self.stride, self.padding, 
                     self.dilation, self.groups)


class Conv2d(Module):
    """二维卷积模块 (2D Convolution Layer)
    
    对输入信号应用二维卷积操作，是卷积神经网络中最核心和最常用的层。
    广泛应用于图像处理、计算机视觉和各种二维数据处理任务。
    
    数学公式::
    
        out(N_i, C_{out_j}, h, w) = bias(C_{out_j}) + 
        sum_{k=0}^{C_{in}-1} sum_{i=0}^{H_k-1} sum_{j=0}^{W_k-1} 
        weight(C_{out_j}, k, i, j) * input(N_i, k, stride_h * h + dilation_h * i + padding_h, 
                                          stride_w * w + dilation_w * j + padding_w)
        
        其中 H_k, W_k 是卷积核的高度和宽度
    
    Args:
        in_channels (int): 输入通道数。输入图像的通道数，RGB图像为3，灰度图像为1。
        out_channels (int): 输出通道数。卷积核的数量，决定输出特征图的通道数。
        kernel_size (int or tuple): 卷积核大小。可以是整数(正方形核)或(height, width)元组。
        stride (int or tuple, optional): 卷积步长。控制卷积核在高度和宽度方向上的滑动步距。
            默认值: 1
        padding (int or tuple, optional): 填充大小。在输入四周添加的零填充数量。
            默认值: 0
        dilation (int or tuple, optional): 膨胀率。卷积核元素之间的间距，用于增大感受野。
            默认值: 1
        groups (int, optional): 分组数。控制输入和输出通道之间的连接模式，groups=1时为标准卷积。
            默认值: 1
        bias (bool, optional): 是否使用偏置项。是否为每个输出通道添加可学习的偏置。
            默认值: True
        dtype (np.dtype, optional): 张量的数据类型。
            默认值: None
        device (str or int or Device, optional): 张量所在的设备。
            默认值: None
    
    Shape:
        - Input: 形状为 (N, C_in, H_in, W_in) 的张量
        - Output: 形状为 (N, C_out, H_out, W_out) 的张量
        - Weight: 形状为 (C_out, C_in // groups, H_k, W_k) 的张量
        - Bias: 形状为 (C_out,) 的张量
    
    Examples::
    
        >>> # 标准图像卷积示例
        >>> m = Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        >>> input = rm.randn(4, 3, 224, 224)  # batch_size=4, channels=3, height=224, width=224
        >>> output = m(input)
        >>> print(output.shape)  # (4, 64, 224, 224)
        
        >>> # 使用非方形卷积核
        >>> m = Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 3), stride=(2, 1))
        >>> input = rm.randn(8, 3, 64, 64)
        >>> output = m(input)
        >>> print(output.shape)  # (8, 32, 30, 62)
        
        >>> # 分组卷积示例
        >>> m = Conv2d(in_channels=32, out_channels=64, kernel_size=3, groups=2)
        >>> input = rm.randn(16, 32, 28, 28)
        >>> output = m(input)
        >>> print(output.shape)  # (16, 64, 26, 26)
    
    Note:
        Conv2d的特点和用途：
        - 是CNN架构的核心组件，用于提取图像的局部特征
        - 通过参数共享机制大幅减少模型参数量
        - 具有平移不变性，能够识别图像中不同位置的相同特征
        - 可以捕获空间层次结构，从低级边缘特征到高级语义特征
        - 支持多种卷积模式：标准卷积、分组卷积、深度可分离卷积等
        - 膨胀卷积可以在不增加参数的情况下增大感受野
        - 输出尺寸计算公式：H_out = floor((H_in + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1)/stride[0] + 1)
        - W_out = floor((W_in + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1)/stride[1] + 1)
        - 在现代计算机视觉模型中不可或缺的基础组件
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, dtype=None, device=None):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # 初始化权重参数
        if isinstance(kernel_size, int):
            weight_shape = (out_channels, in_channels // groups, kernel_size, kernel_size)
        else:
            weight_shape = (out_channels, in_channels // groups, kernel_size[0], kernel_size[1])
        
        self.weight = Parameter(randn(weight_shape, dtype=dtype, device=device) * 0.1)
        
        # 初始化偏置参数
        if bias:
            self.bias = Parameter(zeros(out_channels, dtype=dtype, device=device))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, in_channels, height, width)
            
        Returns:
            输出张量，形状为 (batch_size, out_channels, output_height, output_width)
        """
        return conv2d(x, self.weight, self.bias, self.stride, self.padding,
                     self.dilation, self.groups)


class Conv3d(Module):
    """三维卷积模块 (3D Convolution Layer)
    
    对输入信号应用三维卷积操作，专门用于处理具有三个空间/时间维度的数据。
    广泛应用于视频分析、医学图像处理、3D数据处理等任务。
    
    数学公式::
    
        out(N_i, C_{out_j}, d, h, w) = bias(C_{out_j}) + 
        sum_{k=0}^{C_{in}-1} sum_{t=0}^{D_k-1} sum_{i=0}^{H_k-1} sum_{j=0}^{W_k-1} 
        weight(C_{out_j}, k, t, i, j) * input(N_i, k, 
            stride_d * d + dilation_d * t + padding_d,
            stride_h * h + dilation_h * i + padding_h, 
            stride_w * w + dilation_w * j + padding_w)
        
        其中 D_k, H_k, W_k 是卷积核的深度、高度和宽度
    
    Args:
        in_channels (int): 输入通道数。输入数据的通道数，如视频的帧通道或医学图像的模态数。
        out_channels (int): 输出通道数。卷积核的数量，决定输出特征图的通道数。
        kernel_size (int or tuple): 卷积核大小。可以是整数(立方体核)或(depth, height, width)元组。
        stride (int or tuple, optional): 卷积步长。控制卷积核在深度、高度和宽度方向上的滑动步距。
            默认值: 1
        padding (int or tuple, optional): 填充大小。在输入各个维度添加的零填充数量。
            默认值: 0
        dilation (int or tuple, optional): 膨胀率。卷积核元素之间的间距，用于增大3D感受野。
            默认值: 1
        groups (int, optional): 分组数。控制输入和输出通道之间的连接模式。
            默认值: 1
        bias (bool, optional): 是否使用偏置项。是否为每个输出通道添加可学习的偏置。
            默认值: True
        dtype (np.dtype, optional): 张量的数据类型。
            默认值: None
        device (str or int or Device, optional): 张量所在的设备。
            默认值: None
    
    Shape:
        - Input: 形状为 (N, C_in, D_in, H_in, W_in) 的张量
        - Output: 形状为 (N, C_out, D_out, H_out, W_out) 的张量
        - Weight: 形状为 (C_out, C_in // groups, D_k, H_k, W_k) 的张量
        - Bias: 形状为 (C_out,) 的张量
    
    Examples::
    
        >>> # 视频数据处理示例
        >>> m = Conv3d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        >>> input = rm.randn(4, 3, 10, 64, 64)  # batch_size=4, channels=3, frames=10, height=64, width=64
        >>> output = m(input)
        >>> print(output.shape)  # (4, 16, 10, 64, 64)
        
        >>> # 医学图像处理示例(MRI/CT)
        >>> m = Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 5, 5), stride=(1, 2, 2))
        >>> input = rm.randn(8, 1, 20, 128, 128)  # batch_size=8, modalities=1, depth=20, height=128, width=128
        >>> output = m(input)
        >>> print(output.shape)  # (8, 32, 18, 62, 62)
        
        >>> # 3D数据特征提取
        >>> m = Conv3d(in_channels=64, out_channels=128, kernel_size=(2, 3, 3), stride=(2, 1, 1))
        >>> input = rm.randn(16, 64, 30, 32, 32)
        >>> output = m(input)
        >>> print(output.shape)  # (16, 128, 15, 30, 30)
    
    Note:
        Conv3d的特点和用途：
        - 专门处理三维数据，能够捕获时空特征或3D空间特征
        - 在视频分析中可以同时捕获时间和空间维度的相关性
        - 在医学图像处理中用于处理3D扫描数据(MRI、CT、PET等)
        - 在科学计算中用于处理3D模拟数据或气象数据
        - 参数量相比2D卷积更大，计算复杂度更高
        - 可以捕获3D局部模式和结构信息
        - 支持时空特征学习，对动作识别、视频理解等任务至关重要
        - 输出尺寸计算：D_out = floor((D_in + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1)/stride[0] + 1)
        - H_out = floor((H_in + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1)/stride[1] + 1)
        - W_out = floor((W_in + 2*padding[2] - dilation[2]*(kernel_size[2]-1) - 1)/stride[2] + 1)
        - 是3D CNN架构的核心组件，广泛应用于多模态数据处理
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, dtype=None, device=None):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # 初始化权重参数
        if isinstance(kernel_size, int):
            weight_shape = (out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size)
        else:
            weight_shape = (out_channels, in_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2])
        
        self.weight = Parameter(randn(*weight_shape, dtype=dtype, device=device) * 0.1)
        
        # 初始化偏置参数
        if bias:
            self.bias = Parameter(zeros(out_channels, dtype=dtype, device=device))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, in_channels, depth, height, width)
            
        Returns:
            输出张量，形状为 (batch_size, out_channels, output_depth, output_height, output_width)
        """
        return conv3d(x, self.weight, self.bias, self.stride, self.padding,
                     self.dilation, self.groups)


# ==================== 最大池化模块 ====================

class MaxPool1d(Module):
    """一维最大池化模块 (1D Max Pooling Layer)
    
    对输入信号应用一维最大池化操作，在指定的滑动窗口内选择最大值作为输出。
    主要用于降低序列数据的维度，提供平移不变性，并保留最显著的特征。
    
    数学公式::
    
        output[i] = max_{j} input[stride * i + j]
        
        其中 j 的取值范围是 [0, kernel_size-1]，考虑padding和dilation
    
    Args:
        kernel_size (int or tuple): 池化窗口大小。滑动窗口的长度，控制池化区域的大小。
        stride (int or tuple, optional): 池化步长。窗口滑动的步距，控制输出序列的长度。
            默认值: kernel_size
        padding (int, optional): 填充大小。在序列两端添加的零填充数量。
            默认值: 0
        dilation (int, optional): 膨胀率。池化窗口元素之间的间距。
            默认值: 1
        ceil_mode (bool, optional): 是否使用向上取整计算输出长度。
            为True时使用ceil，为False时使用floor。
            默认值: False
        return_indices (bool, optional): 是否返回最大值的位置索引。
            常用于最大池化的逆操作。
            默认值: False
    
    Shape:
        - Input: 形状为 (N, C, L_in) 的张量
        - Output: 形状为 (N, C, L_out) 的张量
        - 如果return_indices=True，还返回形状为(N, C, L_out)的索引张量
    
    Examples::
    
        >>> # 序列数据降维示例
        >>> m = MaxPool1d(kernel_size=3, stride=2, padding=1)
        >>> input = rm.randn(4, 16, 100)  # batch_size=4, channels=16, length=100
        >>> output = m(input)
        >>> print(output.shape)  # (4, 16, 50)
        
        >>> # 音频特征提取示例
        >>> m = MaxPool1d(kernel_size=2, stride=2)
        >>> input = rm.randn(8, 1, 1024)  # batch_size=8, channels=1, audio_samples=1024
        >>> output = m(input)
        >>> print(output.shape)  # (8, 1, 512)
        
        >>> # 返回索引的示例
        >>> m = MaxPool1d(kernel_size=3, return_indices=True)
        >>> input = rm.randn(2, 4, 10)
        >>> output, indices = m(input)
        >>> print(output.shape, indices.shape)  # (2, 4, 4) (2, 4, 4)
    
    Note:
        MaxPool1d的特点和用途：
        - 通过选择最大值来保留最显著的特征，具有特征选择作用
        - 提供平移不变性，小范围的平移不会影响输出结果
        - 有效降低数据维度，减少计算量和参数数量
        - 常用于CNN架构中的下采样层
        - 在时间序列数据中用于捕获关键时间点的特征
        - 在音频处理中用于降采样和特征压缩
        - 相比平均池化，对噪声更鲁棒，能突出显著特征
        - 输出长度计算：L_out = floor((L_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
        - 当ceil_mode=True时使用ceil代替floor进行计算
        - 是深度学习中最常用的池化操作之一
    """
    
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 ceil_mode=False, return_indices=False):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, channels, length)
            
        Returns:
            输出张量，形状为 (batch_size, channels, output_length)
            如果return_indices=True，还返回索引张量
        """
        return max_pool1d(x, self.kernel_size, self.stride, self.padding,
                         self.dilation, self.ceil_mode, self.return_indices)


class MaxPool2d(Module):
    """二维最大池化模块 (2D Max Pooling Layer)
    
    对输入信号应用二维最大池化操作，在指定的滑动窗口内选择最大值作为输出。
    是CNN架构中最常用的下采样操作，用于降低空间维度，提供平移不变性。
    
    数学公式::
    
        output[i, j] = max_{m, n} input[stride_h * i + m, stride_w * j + n]
        
        其中 (m, n) 的取值范围是 [0, kernel_height-1] × [0, kernel_width-1]，
        考虑padding和dilation的影响
    
    Args:
        kernel_size (int or tuple): 池化窗口大小。可以是整数(正方形窗口)或(height, width)元组。
        stride (int or tuple, optional): 池化步长。窗口在高度和宽度方向上的滑动步距。
            默认值: kernel_size
        padding (int or tuple, optional): 填充大小。在图像四周添加的零填充数量。
            默认值: 0
        dilation (int or tuple, optional): 膨胀率。池化窗口元素之间的间距。
            默认值: 1
        ceil_mode (bool, optional): 是否使用向上取整计算输出尺寸。
            为True时使用ceil，为False时使用floor。
            默认值: False
        return_indices (bool, optional): 是否返回最大值的位置索引。
            常用于最大池化的逆操作和上采样。
            默认值: False
    
    Shape:
        - Input: 形状为 (N, C, H_in, W_in) 的张量
        - Output: 形状为 (N, C, H_out, W_out) 的张量
        - 如果return_indices=True，还返回形状为(N, C, H_out, W_out)的索引张量
    
    Examples::
    
        >>> # 标准图像下采样示例
        >>> m = MaxPool2d(kernel_size=2, stride=2)
        >>> input = rm.randn(4, 64, 224, 224)  # batch_size=4, channels=64, height=224, width=224
        >>> output = m(input)
        >>> print(output.shape)  # (4, 64, 112, 112)
        
        >>> # 非方形池化窗口示例
        >>> m = MaxPool2d(kernel_size=(3, 2), stride=(2, 2), padding=(1, 0))
        >>> input = rm.randn(8, 32, 64, 64)
        >>> output = m(input)
        >>> print(output.shape)  # (8, 32, 32, 32)
        
        >>> # 返回索引用于上采样
        >>> m = MaxPool2d(kernel_size=3, stride=2, return_indices=True)
        >>> input = rm.randn(2, 16, 32, 32)
        >>> output, indices = m(input)
        >>> print(output.shape, indices.shape)  # (2, 16, 15, 15) (2, 16, 15, 15)
    
    Note:
        MaxPool2d的特点和用途：
        - 通过选择局部区域的最大值来保留最显著的特征
        - 提供平移不变性，小范围的目标移动不会显著影响输出
        - 大幅降低空间维度，减少后续层的计算复杂度
        - 是CNN架构中的标准下采样操作
        - 在图像分类、目标检测等任务中广泛使用
        - 相比平均池化，对边缘和纹理特征更敏感
        - 可以有效抑制噪声，保留强响应特征
        - 输出尺寸计算：H_out = floor((H_in + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1)/stride[0] + 1)
        - W_out = floor((W_in + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1)/stride[1] + 1)
        - 当ceil_mode=True时使用ceil代替floor进行计算
        - 与卷积层配合使用，构建层次化的特征提取网络
    """
    
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 ceil_mode=False, return_indices=False):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, channels, height, width)
            
        Returns:
            输出张量，形状为 (batch_size, channels, output_height, output_width)
            如果return_indices=True，还返回索引张量
        """
        return max_pool2d(x, self.kernel_size, self.stride, self.padding,
                         self.dilation, self.ceil_mode, self.return_indices)


class MaxPool3d(Module):
    """三维最大池化模块 (3D Max Pooling Layer)
    
    对输入信号应用三维最大池化操作，在指定的三维滑动窗口内选择最大值作为输出。
    专门用于处理视频数据、医学图像等3D数据的下采样操作。
    
    数学公式::
    
        output[d, i, j] = max_{t, m, n} input[
            stride_d * d + t, 
            stride_h * i + m, 
            stride_w * j + n
        ]
        
        其中 (t, m, n) 的取值范围是 [0, depth-1] × [0, height-1] × [0, width-1]，
        考虑padding和dilation的影响
    
    Args:
        kernel_size (int or tuple): 池化窗口大小。可以是整数(立方体窗口)或(depth, height, width)元组。
        stride (int or tuple, optional): 池化步长。窗口在深度、高度和宽度方向上的滑动步距。
            默认值: kernel_size
        padding (int or tuple, optional): 填充大小。在3D数据各个维度添加的零填充数量。
            默认值: 0
        dilation (int or tuple, optional): 膨胀率。池化窗口元素之间的间距。
            默认值: 1
        ceil_mode (bool, optional): 是否使用向上取整计算输出尺寸。
            为True时使用ceil，为False时使用floor。
            默认值: False
        return_indices (bool, optional): 是否返回最大值的位置索引。
            常用于3D最大池化的逆操作。
            默认值: False
    
    Shape:
        - Input: 形状为 (N, C, D_in, H_in, W_in) 的张量
        - Output: 形状为 (N, C, D_out, H_out, W_out) 的张量
        - 如果return_indices=True，还返回形状为(N, C, D_out, H_out, W_out)的索引张量
    
    Examples::
    
        >>> # 视频数据下采样示例
        >>> m = MaxPool3d(kernel_size=2, stride=2)
        >>> input = rm.randn(4, 3, 16, 64, 64)  # batch_size=4, channels=3, frames=16, height=64, width=64
        >>> output = m(input)
        >>> print(output.shape)  # (4, 3, 8, 32, 32)
        
        >>> # 医学图像处理示例
        >>> m = MaxPool3d(kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        >>> input = rm.randn(8, 1, 20, 128, 128)  # batch_size=8, modalities=1, depth=20, height=128, width=128
        >>> output = m(input)
        >>> print(output.shape)  # (8, 1, 19, 64, 64)
        
        >>> # 3D特征压缩示例
        >>> m = MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        >>> input = rm.randn(2, 32, 30, 64, 64)
        >>> output = m(input)
        >>> print(output.shape)  # (2, 32, 30, 32, 32)
    
    Note:
        MaxPool3d的特点和用途：
        - 在3D空间中选择最显著的特征，适用于时空数据处理
        - 提供三维平移不变性，小范围的时空移动不会显著影响输出
        - 大幅降低3D数据维度，减少计算复杂度和内存使用
        - 是3D CNN架构中的标准下采样操作
        - 在视频分析中用于时空特征的下采样
        - 在医学图像处理中用于3D扫描数据的降维
        - 相比平均池化，能更好地保留强响应的3D特征
        - 输出尺寸计算：D_out = floor((D_in + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1)/stride[0] + 1)
        - H_out = floor((H_in + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1)/stride[1] + 1)
        - W_out = floor((W_in + 2*padding[2] - dilation[2]*(kernel_size[2]-1) - 1)/stride[2] + 1)
        - 当ceil_mode=True时使用ceil代替floor进行计算
        - 在3D深度学习模型中用于构建层次化的时空特征提取网络
    """
    
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 ceil_mode=False, return_indices=False):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, channels, depth, height, width)
            
        Returns:
            输出张量，形状为 (batch_size, channels, output_depth, output_height, output_width)
            如果return_indices=True，还返回索引张量
        """
        return max_pool3d(x, self.kernel_size, self.stride, self.padding,
                         self.dilation, self.ceil_mode, self.return_indices)


# ==================== 平均池化模块 ====================

class AvgPool1d(Module):
    """一维平均池化模块 (1D Average Pooling Layer)
    
    对输入信号应用一维平均池化操作，在指定的滑动窗口内计算平均值作为输出。
    主要用于降低序列数据的维度，提供平滑的下采样效果。
    
    数学公式::
    
        output[i] = (1 / kernel_size) * sum_{j} input[stride * i + j]
        
        其中 j 的取值范围是 [0, kernel_size-1]，考虑padding和dilation
        如果指定了divisor_override，则使用该值作为除数
    
    Args:
        kernel_size (int or tuple): 池化窗口大小。滑动窗口的长度，控制池化区域的大小。
        stride (int or tuple, optional): 池化步长。窗口滑动的步距，控制输出序列的长度。
            默认值: kernel_size
        padding (int, optional): 填充大小。在序列两端添加的零填充数量。
            默认值: 0
        ceil_mode (bool, optional): 是否使用向上取整计算输出长度。
            为True时使用ceil，为False时使用floor。
            默认值: False
        count_include_pad (bool, optional): 计算平均值时是否包含填充值。
            为True时包含padding，为False时只计算有效元素。
            默认值: True
        divisor_override (int, optional): 如果指定，将用作除数而不是窗口大小。
            用于自定义平均计算的除数。
            默认值: None
    
    Shape:
        - Input: 形状为 (N, C, L_in) 的张量
        - Output: 形状为 (N, C, L_out) 的张量
    
    Examples::
    
        >>> # 序列数据平滑降维示例
        >>> m = AvgPool1d(kernel_size=3, stride=2, padding=1)
        >>> input = rm.randn(4, 16, 100)  # batch_size=4, channels=16, length=100
        >>> output = m(input)
        >>> print(output.shape)  # (4, 16, 50)
        
        >>> # 音频信号平滑处理示例
        >>> m = AvgPool1d(kernel_size=2, stride=2)
        >>> input = rm.randn(8, 1, 1024)  # batch_size=8, channels=1, audio_samples=1024
        >>> output = m(input)
        >>> print(output.shape)  # (8, 1, 512)
        
        >>> # 不包含填充的平均池化
        >>> m = AvgPool1d(kernel_size=3, stride=1, padding=1, count_include_pad=False)
        >>> input = rm.randn(2, 4, 10)
        >>> output = m(input)
        >>> print(output.shape)  # (2, 4, 10)
    
    Note:
        AvgPool1d的特点和用途：
        - 通过计算平均值提供平滑的下采样效果，减少噪声影响
        - 相比最大池化，能更好地保留整体统计信息
        - 提供某种程度的平移不变性，但不如最大池化明显
        - 常用于需要保留全局信息的场景
        - 在时间序列数据中用于趋势分析和降噪
        - 在音频处理中用于信号平滑和降采样
        - 计算复杂度低，实现简单高效
        - 输出长度计算：L_out = floor((L_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
        - 当ceil_mode=True时使用ceil代替floor进行计算
        - count_include_pad参数影响平均值的计算方式
        - 在某些网络架构中用于替代最大池化，提供不同的特征提取方式
    """
    
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, channels, length)
            
        Returns:
            输出张量，形状为 (batch_size, channels, output_length)
        """
        return avg_pool1d(x, self.kernel_size, self.stride, self.padding,
                         self.ceil_mode, self.count_include_pad, self.divisor_override)


class AvgPool2d(Module):
    """二维平均池化模块 (2D Average Pooling Layer)
    
    对输入信号应用二维平均池化操作，在指定的滑动窗口内计算平均值作为输出。
    是CNN架构中常用的下采样操作，用于降低空间维度，提供平滑的特征表示。
    
    数学公式::
    
        output[i, j] = (1 / (kernel_height * kernel_width)) * 
        sum_{m, n} input[stride_h * i + m, stride_w * j + n]
        
        其中 (m, n) 的取值范围是 [0, kernel_height-1] × [0, kernel_width-1]，
        考虑padding和dilation的影响
    
    Args:
        kernel_size (int or tuple): 池化窗口大小。可以是整数(正方形窗口)或(height, width)元组。
        stride (int or tuple, optional): 池化步长。窗口在高度和宽度方向上的滑动步距。
            默认值: kernel_size
        padding (int or tuple, optional): 填充大小。在图像四周添加的零填充数量。
            默认值: 0
        ceil_mode (bool, optional): 是否使用向上取整计算输出尺寸。
            为True时使用ceil，为False时使用floor。
            默认值: False
        count_include_pad (bool, optional): 是否在平均计算中包含填充值。
            为True时包含padding，为False时只计算有效区域。
            默认值: True
        divisor_override (int, optional): 除数覆盖值。如果指定，将用作除数而不是实际窗口大小。
            默认值: None
    
    Shape:
        - Input: 形状为 (N, C, H_in, W_in) 的张量
        - Output: 形状为 (N, C, H_out, W_out) 的张量
    
    Examples::
    
        >>> # 标准图像下采样示例
        >>> m = AvgPool2d(kernel_size=2, stride=2)
        >>> input = rm.randn(4, 64, 224, 224)  # batch_size=4, channels=64, height=224, width=224
        >>> output = m(input)
        >>> print(output.shape)  # (4, 64, 112, 112)
        
        >>> # 非方形池化窗口示例
        >>> m = AvgPool2d(kernel_size=(3, 2), stride=(2, 2), padding=(1, 0))
        >>> input = rm.randn(8, 32, 64, 64)
        >>> output = m(input)
        >>> print(output.shape)  # (8, 32, 32, 32)
        
        >>> # 不包含填充值的平均池化
        >>> m = AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
        >>> input = rm.randn(2, 16, 32, 32)
        >>> output = m(input)
        >>> print(output.shape)  # (2, 16, 16, 16)
    
    Note:
        AvgPool2d的特点和用途：
        - 通过计算局部区域的平均值来提供平滑的特征表示
        - 相比最大池化，对噪声更加鲁棒，提供更平滑的下采样
        - 保留整体统计信息，适合需要全局信息的任务
        - 在CNN架构中常用于渐进式下采样
        - 可以有效减少空间维度，降低后续层的计算复杂度
        - count_include_pad参数控制是否考虑填充值的影响
        - divisor_override允许自定义除数，提供更灵活的归一化方式
        - 输出尺寸计算：H_out = floor((H_in + 2*padding[0] - kernel_size[0])/stride[0] + 1)
        - W_out = floor((W_in + 2*padding[1] - kernel_size[1])/stride[1] + 1)
        - 当ceil_mode=True时使用ceil代替floor进行计算
        - 与最大池化相比，更适合保留背景和整体信息
    """
    
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, channels, height, width)
            
        Returns:
            输出张量，形状为 (batch_size, channels, output_height, output_width)
        """
        return avg_pool2d(x, self.kernel_size, self.stride, self.padding,
                         self.ceil_mode, self.count_include_pad, self.divisor_override)


class AvgPool3d(Module):
    """三维平均池化模块 (3D Average Pooling Layer)
    
    对输入信号应用三维平均池化操作，在指定的滑动窗口内计算平均值作为输出。
    是3D CNN架构中常用的下采样操作，用于降低三维空间维度，提供平滑的特征表示。
    
    数学公式::
    
        output[i, j, k] = (1 / (kernel_depth * kernel_height * kernel_width)) * 
        sum_{m, n, p} input[stride_d * i + m, stride_h * j + n, stride_w * k + p]
        
        其中 (m, n, p) 的取值范围是 [0, kernel_depth-1] × [0, kernel_height-1] × [0, kernel_width-1]，
        考虑padding和dilation的影响
    
    Args:
        kernel_size (int or tuple): 池化窗口大小。可以是整数(正方体窗口)或(depth, height, width)元组。
        stride (int or tuple, optional): 池化步长。窗口在深度、高度和宽度方向上的滑动步距。
            默认值: kernel_size
        padding (int or tuple, optional): 填充大小。在三维数据四周添加的零填充数量。
            默认值: 0
        ceil_mode (bool, optional): 是否使用向上取整计算输出尺寸。
            为True时使用ceil，为False时使用floor。
            默认值: False
        count_include_pad (bool, optional): 是否在平均计算中包含填充值。
            为True时包含padding，为False时只计算有效区域。
            默认值: True
        divisor_override (int, optional): 除数覆盖值。如果指定，将用作除数而不是实际窗口大小。
            默认值: None
    
    Shape:
        - Input: 形状为 (N, C, D_in, H_in, W_in) 的张量
        - Output: 形状为 (N, C, D_out, H_out, W_out) 的张量
    
    Examples::
    
        >>> # 标准3D数据下采样示例
        >>> m = AvgPool3d(kernel_size=2, stride=2)
        >>> input = rm.randn(4, 32, 16, 64, 64)  # batch_size=4, channels=32, depth=16, height=64, width=64
        >>> output = m(input)
        >>> print(output.shape)  # (4, 32, 8, 32, 32)
        
        >>> # 非方体池化窗口示例
        >>> m = AvgPool3d(kernel_size=(2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        >>> input = rm.randn(8, 16, 32, 64, 64)
        >>> output = m(input)
        >>> print(output.shape)  # (8, 16, 31, 32, 32)
        
        >>> # 不包含填充值的平均池化
        >>> m = AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
        >>> input = rm.randn(2, 8, 16, 32, 32)
        >>> output = m(input)
        >>> print(output.shape)  # (2, 8, 8, 16, 16)
    
    Note:
        AvgPool3d的特点和用途：
        - 通过计算三维局部区域的平均值来提供平滑的特征表示
        - 相比3D最大池化，对噪声更加鲁棒，提供更平滑的下采样
        - 保留三维整体统计信息，适合需要全局时空信息的任务
        - 在3D CNN架构中常用于渐进式下采样，如视频处理、医学图像分析
        - 可以有效减少三维空间维度，大幅降低后续层的计算复杂度
        - count_include_pad参数控制是否考虑填充值的影响
        - divisor_override允许自定义除数，提供更灵活的归一化方式
        - 输出尺寸计算：D_out = floor((D_in + 2*padding[0] - kernel_size[0])/stride[0] + 1)
        - H_out = floor((H_in + 2*padding[1] - kernel_size[1])/stride[1] + 1)
        - W_out = floor((W_in + 2*padding[2] - kernel_size[2])/stride[2] + 1)
        - 当ceil_mode=True时使用ceil代替floor进行计算
        - 广泛应用于视频分析、医学影像、3D重建等三维数据处理任务
        - 与3D最大池化相比，更适合保留背景和整体时空信息
    """
    
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, channels, depth, height, width)
            
        Returns:
            输出张量，形状为 (batch_size, channels, output_depth, output_height, output_width)
        """
        return avg_pool3d(x, self.kernel_size, self.stride, self.padding,
                         self.ceil_mode, self.count_include_pad, self.divisor_override)


class AdaptiveAvgPool1d(Module):
    """一维自适应平均池化模块 (1D Adaptive Average Pooling Layer)
    
    对输入信号应用一维自适应平均池化操作，自动计算池化核大小和步长，
    确保输出尺寸始终为指定的output_size。
    
    与标准AvgPool1d不同，自适应池化不需要指定kernel_size和stride，
    而是根据输入尺寸和目标输出尺寸自动计算。
    
    数学公式::
    
        对于每个输出位置i，计算对应的输入区域[start, end]：
        - start = floor(i * L_in / L_out)
        - end = ceil((i + 1) * L_in / L_out)
        - output[i] = mean(input[start:end])
    
    Args:
        output_size (int): 目标输出长度。池化后的输出序列长度。
    
    Shape:
        - Input: 形状为 (N, C, L_in) 的张量
        - Output: 形状为 (N, C, L_out) 的张量，其中L_out = output_size
    
    Examples::
    
        >>> # 将任意长度序列池化为固定长度5
        >>> m = AdaptiveAvgPool1d(output_size=5)
        >>> input = rm.randn(8, 16, 100)  # batch_size=8, channels=16, length=100
        >>> output = m(input)
        >>> print(output.shape)  # (8, 16, 5)
        
        >>> # 全局平均池化（输出长度为1）
        >>> m = AdaptiveAvgPool1d(output_size=1)
        >>> input = rm.randn(8, 16, 100)
        >>> output = m(input)
        >>> print(output.shape)  # (8, 16, 1)
        
        >>> # 输入输出尺寸相同时，直接返回输入
        >>> m = AdaptiveAvgPool1d(output_size=100)
        >>> output = m(input)
        >>> print(output.shape)  # (8, 16, 100)
    
    Note:
        AdaptiveAvgPool1d的特点和用途：
        - 自动计算池化参数，无需手动指定kernel_size和stride
        - 输出尺寸固定，不受输入尺寸影响
        - 常用于需要固定输出尺寸的场景，如分类器的全局池化
        - 当input.size(-1) < output_size时，某些输出位置会复制单个输入元素
        - 是ResNet、VGG等网络中常用的全局池化方式
        - 与标准平均池化相比，更适合处理变长输入
        - 在迁移学习中特别有用，可以处理不同尺寸的输入图像
        - 计算复杂度与输入长度和输出长度的乘积成正比
    """
    
    def __init__(self, output_size: int):
        """
        初始化一维自适应平均池化模块
        
        Args:
            output_size (int): 目标输出长度
        """
        super().__init__()
        self.output_size = output_size
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, channels, length)
            
        Returns:
            输出张量，形状为 (batch_size, channels, output_size)
        """
        return adaptive_avg_pool1d(x, self.output_size)
    
    def extra_repr(self) -> str:
        """返回模块的额外表示信息"""
        return f'output_size={self.output_size}'


class AdaptiveAvgPool2d(Module):
    """二维自适应平均池化模块 (2D Adaptive Average Pooling Layer)
    
    对输入信号应用二维自适应平均池化操作，自动计算池化核大小和步长，
    确保输出尺寸始终为指定的output_size。
    
    与标准AvgPool2d不同，自适应池化不需要指定kernel_size和stride，
    而是根据输入尺寸和目标输出尺寸自动计算。
    
    数学公式::
    
        对于每个输出位置(i, j)，计算对应的输入区域：
        - start_h = floor(i * H_in / H_out), end_h = ceil((i + 1) * H_in / H_out)
        - start_w = floor(j * W_in / W_out), end_w = ceil((j + 1) * W_in / W_out)
        - output[i, j] = mean(input[start_h:end_h, start_w:end_w])
    
    Args:
        output_size (int or tuple[int, int]): 目标输出尺寸。
            - 单个整数：输出正方形 (output_size, output_size)
            - 元组 (H, W)：输出矩形 (H, W)
    
    Shape:
        - Input: 形状为 (N, C, H_in, W_in) 的张量
        - Output: 形状为 (N, C, H_out, W_out) 的张量
    
    Examples::
    
        >>> # 将任意尺寸特征图池化为7x7
        >>> m = AdaptiveAvgPool2d(output_size=(7, 7))
        >>> input = rm.randn(8, 16, 32, 32)  # batch_size=8, channels=16, 32x32
        >>> output = m(input)
        >>> print(output.shape)  # (8, 16, 7, 7)
        
        >>> # 全局平均池化（输出1x1）
        >>> m = AdaptiveAvgPool2d(output_size=1)
        >>> input = rm.randn(8, 16, 32, 32)
        >>> output = m(input)
        >>> print(output.shape)  # (8, 16, 1, 1)
        
        >>> # 输出矩形特征图
        >>> m = AdaptiveAvgPool2d(output_size=(3, 5))
        >>> output = m(input)
        >>> print(output.shape)  # (8, 16, 3, 5)
    
    Note:
        AdaptiveAvgPool2d的特点和用途：
        - 自动计算池化参数，无需手动指定kernel_size和stride
        - 输出尺寸固定，不受输入尺寸影响
        - 是ResNet、VGG、DenseNet等分类网络的标准全局池化方式
        - 常用于将变尺寸特征图转换为固定尺寸，便于后续全连接层处理
        - 支持正方形和矩形两种输出形状
        - 当输入尺寸小于输出尺寸时，某些输出位置会复制单个输入元素
        - 在迁移学习和多尺度训练中特别有用
        - 计算复杂度与输入面积和输出面积的乘积成正比
        - 是现代CNN架构中不可或缺的组件
    """
    
    def __init__(self, output_size: int | tuple[int, int]):
        """
        初始化二维自适应平均池化模块
        
        Args:
            output_size (int or tuple[int, int]): 目标输出尺寸
        """
        super().__init__()
        self.output_size = output_size
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, channels, height, width)
            
        Returns:
            输出张量，形状为 (batch_size, channels, H_out, W_out)
        """
        return adaptive_avg_pool2d(x, self.output_size)
    
    def extra_repr(self) -> str:
        """返回模块的额外表示信息"""
        return f'output_size={self.output_size}'


class AdaptiveAvgPool3d(Module):
    """三维自适应平均池化模块 (3D Adaptive Average Pooling Layer)
    
    对输入信号应用三维自适应平均池化操作，自动计算池化核大小和步长，
    确保输出尺寸始终为指定的output_size。
    
    与标准AvgPool3d不同，自适应池化不需要指定kernel_size和stride，
    而是根据输入尺寸和目标输出尺寸自动计算。
    
    数学公式::
    
        对于每个输出位置(d, i, j)，计算对应的输入区域：
        - start_d = floor(d * D_in / D_out), end_d = ceil((d + 1) * D_in / D_out)
        - start_h = floor(i * H_in / H_out), end_h = ceil((i + 1) * H_in / H_out)
        - start_w = floor(j * W_in / W_out), end_w = ceil((j + 1) * W_in / W_out)
        - output[d, i, j] = mean(input[start_d:end_d, start_h:end_h, start_w:end_w])
    
    Args:
        output_size (int or tuple[int, int, int]): 目标输出尺寸。
            - 单个整数：输出立方体 (output_size, output_size, output_size)
            - 元组 (D, H, W)：输出长方体 (D, H, W)
    
    Shape:
        - Input: 形状为 (N, C, D_in, H_in, W_in) 的张量
        - Output: 形状为 (N, C, D_out, H_out, W_out) 的张量
    
    Examples::
    
        >>> # 将任意尺寸3D特征池化为4x4x4
        >>> m = AdaptiveAvgPool3d(output_size=(4, 4, 4))
        >>> input = rm.randn(2, 8, 16, 32, 32)  # batch=2, channels=8, 16x32x32
        >>> output = m(input)
        >>> print(output.shape)  # (2, 8, 4, 4, 4)
        
        >>> # 全局平均池化（输出1x1x1）
        >>> m = AdaptiveAvgPool3d(output_size=1)
        >>> input = rm.randn(2, 8, 16, 32, 32)
        >>> output = m(input)
        >>> print(output.shape)  # (2, 8, 1, 1, 1)
    
    Note:
        AdaptiveAvgPool3d的特点和用途：
        - 自动计算池化参数，无需手动指定kernel_size和stride
        - 输出尺寸固定，不受输入尺寸影响
        - 常用于3D卷积网络的全局池化，如视频分析、医学图像处理
        - 支持立方体和长方体两种输出形状
        - 在3D CNN架构中用于将变尺寸体数据转换为固定尺寸
        - 计算复杂度与输入体积和输出体积的乘积成正比
    """
    
    def __init__(self, output_size: int | tuple[int, int, int]):
        """
        初始化三维自适应平均池化模块
        
        Args:
            output_size (int or tuple[int, int, int]): 目标输出尺寸
        """
        super().__init__()
        self.output_size = output_size
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, channels, depth, height, width)
            
        Returns:
            输出张量，形状为 (batch_size, channels, D_out, H_out, W_out)
        """
        return adaptive_avg_pool3d(x, self.output_size)
    
    def extra_repr(self) -> str:
        """返回模块的额外表示信息"""
        return f'output_size={self.output_size}'


class AdaptiveMaxPool1d(Module):
    """一维自适应最大池化模块 (1D Adaptive Max Pooling Layer)
    
    对输入信号应用一维自适应最大池化操作，自动计算池化核大小和步长，
    确保输出尺寸始终为指定的output_size。
    
    与标准MaxPool1d不同，自适应池化不需要指定kernel_size和stride，
    而是根据输入尺寸和目标输出尺寸自动计算。
    
    数学公式::
    
        对于每个输出位置i，计算对应的输入区域[start, end]：
        - start = floor(i * L_in / L_out)
        - end = ceil((i + 1) * L_in / L_out)
        - output[i] = max(input[start:end])
    
    Args:
        output_size (int): 目标输出长度。池化后的输出序列长度。
        return_indices (bool, optional): 是否返回最大值的索引。默认值: False
    
    Shape:
        - Input: 形状为 (N, C, L_in) 的张量
        - Output: 形状为 (N, C, L_out) 的张量，其中L_out = output_size
        - 如果return_indices=True，还返回形状为(N, C, L_out)的索引张量
    
    Examples::
    
        >>> # 将任意长度序列池化为固定长度5
        >>> m = AdaptiveMaxPool1d(output_size=5)
        >>> input = rm.randn(8, 16, 100)  # batch_size=8, channels=16, length=100
        >>> output = m(input)
        >>> print(output.shape)  # (8, 16, 5)
        
        >>> # 返回索引
        >>> m = AdaptiveMaxPool1d(output_size=5, return_indices=True)
        >>> output, indices = m(input)
        >>> print(indices.shape)  # (8, 16, 5)
        
        >>> # 全局最大池化（输出长度为1）
        >>> m = AdaptiveMaxPool1d(output_size=1)
        >>> input = rm.randn(8, 16, 100)
        >>> output = m(input)
        >>> print(output.shape)  # (8, 16, 1)
    
    Note:
        AdaptiveMaxPool1d的特点和用途：
        - 自动计算池化参数，无需手动指定kernel_size和stride
        - 输出尺寸固定，不受输入尺寸影响
        - 保留最显著的特征，适合需要突出重要信息的任务
        - 返回的indices可用于adaptive_max_pool1d的逆操作
        - 与自适应平均池化相比，更关注局部最显著特征
    """
    
    def __init__(self, output_size: int, return_indices: bool = False):
        """
        初始化一维自适应最大池化模块
        
        Args:
            output_size (int): 目标输出长度
            return_indices (bool, optional): 是否返回最大值的索引。默认值: False
        """
        super().__init__()
        self.output_size = output_size
        self.return_indices = return_indices
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, channels, length)
            
        Returns:
            输出张量，形状为 (batch_size, channels, output_size)
            如果return_indices=True，还返回索引张量
        """
        return adaptive_max_pool1d(x, self.output_size, self.return_indices)
    
    def extra_repr(self) -> str:
        """返回模块的额外表示信息"""
        return f'output_size={self.output_size}, return_indices={self.return_indices}'


class AdaptiveMaxPool2d(Module):
    """二维自适应最大池化模块 (2D Adaptive Max Pooling Layer)
    
    对输入信号应用二维自适应最大池化操作，自动计算池化核大小和步长，
    确保输出尺寸始终为指定的output_size。
    
    与标准MaxPool2d不同，自适应池化不需要指定kernel_size和stride，
    而是根据输入尺寸和目标输出尺寸自动计算。
    
    数学公式::
    
        对于每个输出位置(i, j)，计算对应的输入区域：
        - start_h = floor(i * H_in / H_out), end_h = ceil((i + 1) * H_in / H_out)
        - start_w = floor(j * W_in / W_out), end_w = ceil((j + 1) * W_in / W_out)
        - output[i, j] = max(input[start_h:end_h, start_w:end_w])
    
    Args:
        output_size (int or tuple[int, int]): 目标输出尺寸。
            - 单个整数：输出正方形 (output_size, output_size)
            - 元组 (H, W)：输出矩形 (H, W)
        return_indices (bool, optional): 是否返回最大值的索引。默认值: False
    
    Shape:
        - Input: 形状为 (N, C, H_in, W_in) 的张量
        - Output: 形状为 (N, C, H_out, W_out) 的张量
        - 如果return_indices=True，indices的形状为(N, C, H_out, W_out)
    
    Examples::
    
        >>> # 将任意尺寸特征图池化为7x7
        >>> m = AdaptiveMaxPool2d(output_size=(7, 7))
        >>> input = rm.randn(8, 16, 32, 32)  # batch_size=8, channels=16, 32x32
        >>> output = m(input)
        >>> print(output.shape)  # (8, 16, 7, 7)
        
        >>> # 返回索引
        >>> m = AdaptiveMaxPool2d(output_size=(7, 7), return_indices=True)
        >>> output, indices = m(input)
        >>> print(indices.shape)  # (8, 16, 7, 7)
        
        >>> # 全局最大池化（输出1x1）
        >>> m = AdaptiveMaxPool2d(output_size=1)
        >>> input = rm.randn(8, 16, 32, 32)
        >>> output = m(input)
        >>> print(output.shape)  # (8, 16, 1, 1)
    
    Note:
        AdaptiveMaxPool2d的特点和用途：
        - 自动计算池化参数，无需手动指定kernel_size和stride
        - 输出尺寸固定，不受输入尺寸影响
        - 保留最显著的局部特征
        - 返回的indices是flatten后的索引，可用于max_unpool2d
        - 与自适应平均池化相比，更关注局部最显著特征
    """
    
    def __init__(self, output_size: int | tuple[int, int], return_indices: bool = False):
        """
        初始化二维自适应最大池化模块
        
        Args:
            output_size (int or tuple[int, int]): 目标输出尺寸
            return_indices (bool, optional): 是否返回最大值的索引。默认值: False
        """
        super().__init__()
        self.output_size = output_size
        self.return_indices = return_indices
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, channels, height, width)
            
        Returns:
            输出张量，形状为 (batch_size, channels, H_out, W_out)
            如果return_indices=True，还返回索引张量
        """
        return adaptive_max_pool2d(x, self.output_size, self.return_indices)
    
    def extra_repr(self) -> str:
        """返回模块的额外表示信息"""
        return f'output_size={self.output_size}, return_indices={self.return_indices}'


class AdaptiveMaxPool3d(Module):
    """三维自适应最大池化模块 (3D Adaptive Max Pooling Layer)
    
    对输入信号应用三维自适应最大池化操作，自动计算池化核大小和步长，
    确保输出尺寸始终为指定的output_size。
    
    与标准MaxPool3d不同，自适应池化不需要指定kernel_size和stride，
    而是根据输入尺寸和目标输出尺寸自动计算。
    
    数学公式::
    
        对于每个输出位置(d, i, j)，计算对应的输入区域：
        - start_d = floor(d * D_in / D_out), end_d = ceil((d + 1) * D_in / D_out)
        - start_h = floor(i * H_in / H_out), end_h = ceil((i + 1) * H_in / H_out)
        - start_w = floor(j * W_in / W_out), end_w = ceil((j + 1) * W_in / W_out)
        - output[d, i, j] = max(input[start_d:end_d, start_h:end_h, start_w:end_w])
    
    Args:
        output_size (int or tuple[int, int, int]): 目标输出尺寸。
            - 单个整数：输出立方体 (output_size, output_size, output_size)
            - 元组 (D, H, W)：输出长方体 (D, H, W)
        return_indices (bool, optional): 是否返回最大值的索引。默认值: False
    
    Shape:
        - Input: 形状为 (N, C, D_in, H_in, W_in) 的张量
        - Output: 形状为 (N, C, D_out, H_out, W_out) 的张量
        - 如果return_indices=True，indices的形状为(N, C, D_out, H_out, W_out)
    
    Examples::
    
        >>> # 将任意尺寸3D特征池化为4x4x4
        >>> m = AdaptiveMaxPool3d(output_size=(4, 4, 4))
        >>> input = rm.randn(2, 8, 16, 32, 32)  # batch=2, channels=8, 16x32x32
        >>> output = m(input)
        >>> print(output.shape)  # (2, 8, 4, 4, 4)
        
        >>> # 返回索引
        >>> m = AdaptiveMaxPool3d(output_size=(4, 4, 4), return_indices=True)
        >>> output, indices = m(input)
        >>> print(indices.shape)  # (2, 8, 4, 4, 4)
        
        >>> # 全局最大池化（输出1x1x1）
        >>> m = AdaptiveMaxPool3d(output_size=1)
        >>> input = rm.randn(2, 8, 16, 32, 32)
        >>> output = m(input)
        >>> print(output.shape)  # (2, 8, 1, 1, 1)
    
    Note:
        AdaptiveMaxPool3d的特点和用途：
        - 自动计算池化参数，无需手动指定kernel_size和stride
        - 输出尺寸固定，不受输入尺寸影响
        - 保留最显著的3D局部特征
        - 返回的indices是flatten后的索引
        - 常用于3D卷积网络的全局池化
    """
    
    def __init__(self, output_size: int | tuple[int, int, int], return_indices: bool = False):
        """
        初始化三维自适应最大池化模块
        
        Args:
            output_size (int or tuple[int, int, int]): 目标输出尺寸
            return_indices (bool, optional): 是否返回最大值的索引。默认值: False
        """
        super().__init__()
        self.output_size = output_size
        self.return_indices = return_indices
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, channels, depth, height, width)
            
        Returns:
            输出张量，形状为 (batch_size, channels, D_out, H_out, W_out)
            如果return_indices=True，还返回索引张量
        """
        return adaptive_max_pool3d(x, self.output_size, self.return_indices)
    
    def extra_repr(self) -> str:
        """返回模块的额外表示信息"""
        return f'output_size={self.output_size}, return_indices={self.return_indices}'
