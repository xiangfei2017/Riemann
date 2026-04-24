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


"""
Kolmogorov-Arnold网络(KAN)的高效实现

此模块提供了Kolmogorov-Arnold网络的高效实现，基于efficient-kan项目适配到Riemann框架。
KAN是一种新型神经网络架构，使用可学习的激活函数替代传统的固定激活函数。

主要特点:
    - 使用B样条基函数作为可学习激活函数
    - 高效的矩阵乘法实现，避免中间变量膨胀
    - 支持自适应网格更新
    - 提供L1正则化和熵正则化

参考:
    - 原始论文: "KAN: Kolmogorov-Arnold Networks"
    - efficient-kan: https://github.com/Blealtan/efficient-kan

示例:
    >>> from riemann.nn.kan import KAN, KANLinear
    >>> # 创建单层KAN
    >>> layer = KANLinear(10, 5, grid_size=5, spline_order=3)
    >>> # 创建多层KAN网络
    >>> model = KAN([28*28, 64, 10], grid_size=5, spline_order=3)
"""

import math
import builtins
from . import functional as F
from ..tensordef import TN, rand, empty, arange, linspace, no_grad, concatenate, sort, float32, int64
from .module import Module, Parameter, ModuleList
from . import init
from .activation import SiLU
from ..linalg import lstsq


class KANLinear(Module):
    """KAN线性层
    
    KANLinear是Kolmogorov-Arnold网络的基本构建块。与传统线性层不同，
    它使用可学习的B样条激活函数替代固定的激活函数。
    
    计算过程:
        输出 = 基函数路径 + 样条路径
        基函数路径: base_activation(x) @ base_weight
        样条路径: b_splines(x) @ scaled_spline_weight
    
    Args:
        in_features: 输入特征维度
        out_features: 输出特征维度
        grid_size: 网格大小，控制B样条的分段数
        spline_order: B样条阶数，控制平滑度
        scale_noise: 噪声缩放系数，用于初始化
        scale_base: 基函数权重缩放系数
        scale_spline: 样条权重缩放系数
        enable_standalone_scale_spline: 是否启用独立的样条缩放
        base_activation: 基函数激活函数，默认为SiLU
        grid_eps: 网格更新时的插值系数
        grid_range: 网格值范围，默认为[-1, 1]
    
    Shape:
        - 输入: (batch_size, in_features)
        - 输出: (batch_size, out_features)
    
    示例:
        >>> layer = KANLinear(10, 5)
        >>> x = rm.randn(4, 10)
        >>> output = layer(x)
        >>> print(output.shape)  # (4, 5)
    """
    
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # 计算网格步长
        h = (grid_range[1] - grid_range[0]) / grid_size
        
        # 创建B样条网格
        # 网格范围扩展到[spline_order, grid_size + spline_order]以支持边界计算
        grid: TN = (
            (
                arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # 基函数路径的权重参数
        self.base_weight = Parameter(empty(out_features, in_features))
        
        # 样条路径的权重参数
        self.spline_weight = Parameter(
            empty(out_features, in_features, grid_size + spline_order)
        )
        
        # 可选的样条缩放参数
        if enable_standalone_scale_spline:
            self.spline_scaler = Parameter(
                empty(out_features, in_features)
            )

        # 保存配置参数
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """重置网络参数
        
        使用Kaiming均匀初始化基函数权重，
        使用噪声初始化样条权重。
        """
        # 使用Kaiming初始化基函数权重
        init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        
        with no_grad():
            # 获取设备信息，确保噪声在与权重相同的设备上创建
            device = self.spline_weight.device
            
            # 生成随机噪声用于样条权重初始化
            noise = (
                (
                    rand(self.grid_size + 1, self.in_features, self.out_features, device=device)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            
            # 通过曲线拟合将噪声转换为样条系数
            self.spline_weight.data = (
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                ).data
            )
            
            # 初始化样条缩放参数
            if self.enable_standalone_scale_spline:
                init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: TN) -> TN:
        """计算B样条基函数
        
        使用de Boor递归公式计算B样条基函数值。
        B样条是分段多项式函数，具有局部支撑性和平滑性。
        
        Args:
            x: 输入张量，形状为(batch_size, in_features)
        
        Returns:
            B样条基函数值，形状为(batch_size, in_features, grid_size + spline_order)
        
        数学公式:
            B样条通过递归定义:
            - 0阶: B_{i,0}(x) = 1 if grid[i] <= x < grid[i+1], else 0
            - k阶: B_{i,k}(x) = w1 * B_{i,k-1}(x) + w2 * B_{i+1,k-1}(x)
            其中w1, w2是基于网格的权重系数
        """
        # 验证输入维度
        assert x.ndim == 2 and x.size(1) == self.in_features

        # 获取网格
        grid: TN = self.grid  # (in_features, grid_size + 2 * spline_order + 1)
        
        # 增加维度用于广播计算
        x = x.unsqueeze(-1)  # (batch_size, in_features, 1)
        
        # 初始化0阶B样条（指示函数）
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        
        # 递归计算高阶B样条
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        # 验证输出形状
        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: TN, y: TN) -> TN:
        """将曲线点转换为B样条系数
        
        使用最小二乘法求解B样条系数，使得B样条曲线通过给定点。
        
        Args:
            x: 输入位置，形状为(batch_size, in_features)
            y: 输出值，形状为(batch_size, in_features, out_features)
        
        Returns:
            B样条系数，形状为(out_features, in_features, grid_size + spline_order)
        
        算法:
            1. 计算B样条基函数矩阵A
            2. 使用最小二乘法求解: A @ coeff = y
            3. 返回系数矩阵
        """
        # 验证输入
        assert x.ndim == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        # 计算B样条基函数并转置
        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        
        # 转置输出
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        
        # 使用最小二乘法求解
        solution = lstsq(A, B)[0]  # (in_features, grid_size + spline_order, out_features)
        
        # 调整维度顺序
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        # 验证输出形状
        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self) -> TN:
        """获取缩放后的样条权重
        
        如果启用了独立缩放，则将样条权重与缩放因子相乘。
        
        Returns:
            缩放后的样条权重
        """
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: TN) -> TN:
        """前向传播
        
        计算KAN层的前向传播，包括基函数路径和样条路径。
        
        Args:
            x: 输入张量，形状为(..., in_features)
        
        Returns:
            输出张量，形状为(..., out_features)
        
        计算过程:
            output = base_activation(x) @ base_weight + b_splines(x) @ spline_weight
        """
        # 验证输入特征维度
        assert x.size(-1) == self.in_features
        
        # 保存原始形状
        original_shape = x.shape
        
        # 展平为2D张量
        x = x.reshape(-1, self.in_features)

        # 基函数路径: 激活后线性变换
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        # 样条路径: B样条基函数后线性变换
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        
        # 合并两个路径
        output = base_output + spline_output
        
        # 恢复原始形状
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @no_grad()
    def update_grid(self, x: TN, margin=0.01):
        """更新B样条网格
        
        根据输入数据分布自适应更新网格，使网格更好地覆盖数据范围。
        结合自适应网格（基于数据分布）和均匀网格。
        
        Args:
            x: 输入数据，形状为(batch_size, in_features)
            margin: 网格边界余量
        
        算法:
            1. 计算当前样条输出
            2. 根据输入数据分布计算自适应网格
            3. 结合均匀网格和自适应网格
            4. 更新网格和样条系数
        """
        # 验证输入
        assert x.ndim == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        # 计算当前B样条输出
        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = splines @ orig_coeff  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # 对每个通道单独排序以收集数据分布
        x_sorted = sort(x, dim=0)[0]
        
        # 计算自适应网格点（基于数据分布）
        grid_adaptive = x_sorted[
            linspace(
                0, batch - 1, self.grid_size + 1, dtype=int64, device=x.device
            )
        ]

        # 计算均匀网格步长
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        
        # 计算均匀网格
        grid_uniform = (
            arange(
                self.grid_size + 1, dtype=float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        # 结合自适应网格和均匀网格
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        
        # 扩展网格边界以支持边界计算
        grid = concatenate(
            [
                grid[:1]
                - uniform_step
                * arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        # 更新网格
        self.grid.copy_(grid.T)
        
        # 更新样条系数以适应新网格
        self.spline_weight.data = self.curve2coeff(x, unreduced_spline_output).data

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0) -> TN:
        """计算正则化损失
        
        计算L1正则化和熵正则化，用于促进稀疏性和可解释性。
        
        Args:
            regularize_activation: L1正则化系数
            regularize_entropy: 熵正则化系数
        
        Returns:
            总正则化损失
        
        说明:
            原始论文中的L1正则化基于输入样本定义，需要展开中间变量，
            与高效实现不兼容。这里使用权重上的L1正则化替代，
            这在神经网络中更常见且与高效实现兼容。
        """
        # 计算L1正则化（基于样条权重的均值）
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        
        # 计算熵正则化
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -(p * p.log()).sum()
        
        # 合并两种正则化
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(Module):
    """Kolmogorov-Arnold网络
    
    多层KAN网络，由多个KANLinear层堆叠而成。
    
    Args:
        layers_hidden: 隐藏层维度列表，如[28*28, 64, 10]
        grid_size: 网格大小
        spline_order: B样条阶数
        scale_noise: 噪声缩放系数
        scale_base: 基函数权重缩放系数
        scale_spline: 样条权重缩放系数
        base_activation: 基函数激活函数
        grid_eps: 网格更新插值系数
        grid_range: 网格值范围
    
    Shape:
        - 输入: (batch_size, layers_hidden[0])
        - 输出: (batch_size, layers_hidden[-1])
    
    示例:
        >>> model = KAN([28*28, 64, 10])
        >>> x = rm.randn(4, 28*28)
        >>> output = model(x)
        >>> print(output.shape)  # (4, 10)
    """
    
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        # 构建网络层
        self.layers = ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: TN, update_grid=False) -> TN:
        """前向传播
        
        Args:
            x: 输入张量
            update_grid: 是否在每个层更新网格
        
        Returns:
            输出张量
        """
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0) -> TN:
        """计算总正则化损失
        
        对所有层的正则化损失求和。
        
        Args:
            regularize_activation: L1正则化系数
            regularize_entropy: 熵正则化系数
        
        Returns:
            总正则化损失
        """
        return builtins.sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
