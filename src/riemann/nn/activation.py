# BSD 3-Clause License
# 
# Copyright (c) 2025, Fei Xiang
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
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
Activation Functions Module Implementation for the Riemann Library

This module provides implementations of common activation functions as neural network modules
for the Riemann library. Activation functions introduce non-linearity into neural networks,
enabling them to learn complex patterns.

Implemented activation functions:
- ReLU: Rectified Linear Unit, the most commonly used activation function
- LeakyReLU: Variant of ReLU with small slope for negative inputs
- RReLU: Randomized LeakyReLU with random slope during training
- PReLU: Parametric ReLU with learnable slope parameters
- Sigmoid: S-shaped activation function mapping to (0,1) range
- Softmax: Normalizes inputs to probabilities summing to 1
- GELU: Gaussian Error Linear Unit, popular in transformer architectures
- LogSoftmax: Logarithm of Softmax, useful for loss calculation
- Softplus: Smooth approximation of ReLU
- Tanh: Hyperbolic tangent function mapping to (-1,1) range

All activation modules inherit from the Module base class and implement a forward method
that applies the respective activation function from the functional module.
"""
from .module import *
from .functional import *

class ReLU(Module):
    """修正线性单元激活模块 (Rectified Linear Unit)
    
    应用修正线性单元函数逐元素到输入张量，是深度学习中最常用的激活函数之一。
    
    数学公式::
    
        ReLU(x) = max(0, x)
    
    Args:
        无参数
    
    Shape:
        - Input: 任意形状的张量 (*)
        - Output: 与输入相同形状的张量 (*)
    
    Examples::
    
        >>> m = ReLU()
        >>> input = rm.randn(2, 3)
        >>> output = m(input)
    
    Note:
        ReLU函数具有以下特性：
        - 计算简单，梯度为0或1，有助于缓解梯度消失问题
        - 对于负值输入输出为0，可能导致"神经元死亡"问题
        - 在正数区间是线性的，有助于梯度传播
        - 是目前深度学习中最主流的激活函数选择
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return relu(x)

class LeakyReLU(Module):
    """泄漏修正线性单元激活模块 (Leaky Rectified Linear Unit)
    
    应用泄漏修正线性单元函数逐元素到输入张量，是ReLU的改进版本，
    为负值输入提供小的非零梯度，缓解神经元死亡问题。
    
    数学公式::
    
        LeakyReLU(x) = max(0, x) + alpha * min(0, x)
                      = { x,         if x >= 0
                        { alpha * x, if x < 0
    
    Args:
        alpha (float, optional): 负值区域的斜率控制参数。
            控制负数输入时的输出大小，通常设置为较小的正值如0.01。
            默认值: 0.01
    
    Shape:
        - Input: 任意形状的张量 (*)
        - Output: 与输入相同形状的张量 (*)
    
    Examples::
    
        >>> m = LeakyReLU(alpha=0.02)
        >>> input = rm.randn(2, 3)
        >>> output = m(input)
    
    Note:
        LeakyReLU的优势：
        - 解决了ReLU的神经元死亡问题，负值时仍有梯度
        - 计算简单，梯度恒定
        - 在某些任务上表现优于ReLU
        - alpha参数需要根据具体任务调整，通常取0.01-0.2之间
    """
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha  # 保存负区斜率参数[3](@ref)
    
    def forward(self, x: TN) -> TN:
        return leaky_relu(x, self.alpha) 
    
class RReLU(Module):
    """随机泄漏修正线性单元激活模块 (Randomized Leaky Rectified Linear Unit)
    
    应用随机泄漏修正线性单元函数逐元素到输入张量，是LeakyReLU的随机化版本。
    在训练时使用随机斜率，在评估时使用固定斜率，提供正则化效果。
    
    数学公式::
    
        RReLU(x) = max(0, x) + alpha * min(0, x)
                  = { x,           if x >= 0
                    { alpha * x,   if x < 0
        
        其中 alpha 在训练时从 U(lower, upper) 均匀分布中随机采样，
        在评估时使用 (lower + upper) / 2
    
    Args:
        lower (float, optional): 随机斜率的下界。
            控制负数输入时的最小斜率，必须为非负数且小于upper。
            默认值: 1/8 ≈ 0.125
        upper (float, optional): 随机斜率的上界。
            控制负数输入时的最大斜率，必须为正数且大于lower。
            默认值: 1/3 ≈ 0.333
    
    Shape:
        - Input: 任意形状的张量 (*)
        - Output: 与输入相同形状的张量 (*)
    
    Examples::
    
        >>> m = RReLU(lower=0.1, upper=0.3)
        >>> input = rm.randn(2, 3)
        >>> output = m(input)  # 训练时使用随机斜率
    
    Note:
        RReLU的特点：
        - 训练时的随机性起到正则化作用，防止过拟合
        - 评估时使用确定性斜率，保证结果一致性
        - 在某些数据集上表现优于LeakyReLU
        - 随机性增加了模型的鲁棒性
    """
    def __init__(self, lower=1/8, upper=1/3):
        super().__init__()
        self.lower, self.upper = lower, upper  # 保存随机斜率范围
    
    def forward(self, x: TN) -> TN:
        # 根据训练状态选择随机/固定斜率
        alpha = np.random.uniform(self.lower, self.upper) if self.training else (self.lower + self.upper)/2
        return leaky_relu(x, alpha)

class PReLU(Module):
    """参数化修正线性单元激活模块 (Parametric Rectified Linear Unit)
    
    应用参数化修正线性单元函数逐元素到输入张量，是LeakyReLU的可学习参数版本。
    负值区域的斜率参数通过反向传播自动学习，适应数据特征。
    
    数学公式::
    
        PReLU(x) = max(0, x) + alpha * min(0, x)
                  = { x,           if x >= 0
                    { alpha_i * x, if x < 0 (按通道应用不同的alpha)
        
        其中 alpha 是可学习的参数，可以为每个通道设置不同的值
    
    Args:
        num_parameters (int, optional): 可学习参数alpha的数量。
            通常设置为1（所有通道共享）或输入张量的通道数。
            默认值: 1
        init (float, optional): alpha参数的初始值。
            初始化负值区域的斜率，通常设置为较小的正值。
            默认值: 0.25
    
    Shape:
        - Input: 任意形状的张量 (*)
        - Output: 与输入相同形状的张量 (*)
        - Alpha: 形状为 (num_parameters,) 的可学习参数
    
    Examples::
    
        >>> m = PReLU(num_parameters=1, init=0.25)
        >>> input = rm.randn(2, 3, 4, 5)
        >>> output = m(input)
        >>> # alpha参数会自动学习
        >>> print(m.alpha)  # 可学习参数
    
    Note:
        PReLU的优势：
        - alpha参数可学习，自适应调整负值区域斜率
        - 参数量很少，计算开销小
        - 在ImageNet等大规模数据集上表现优异
        - 缓解了ReLU的神经元死亡问题
        - 相比LeakyReLU具有更强的表达能力
    """
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        # 初始化可训练参数
        self.alpha = tensor(np.full((num_parameters,), init), requires_grad=True)
    
    def forward(self, x: TN) -> TN:
        # 调用函数式prelu实现并传递参数
        return prelu(x, self.alpha)
    
class Sigmoid(Module):
    """Sigmoid激活模块 (Sigmoid Activation Function)
    
    应用Sigmoid函数逐元素到输入张量，将输入映射到(0,1)区间，
    是经典的神经网络激活函数，常用于二分类问题的输出层。
    
    数学公式::
    
        Sigmoid(x) = 1 / (1 + exp(-x))
                   = 1 / (1 + e^(-x))
    
    导数公式::
    
        Sigmoid'(x) = Sigmoid(x) * (1 - Sigmoid(x))
    
    Args:
        无参数
    
    Shape:
        - Input: 任意形状的张量 (*)
        - Output: 与输入相同形状的张量 (*), 值域为(0,1)
    
    Examples::
    
        >>> m = Sigmoid()
        >>> input = rm.randn(2, 3)
        >>> output = m(input)
        >>> print(output.min(), output.max())  # 输出值在(0,1)之间
    
    Note:
        Sigmoid函数的特点：
        - 输出范围在(0,1)之间，适合表示概率
        - 函数光滑，处处可导
        - 存在梯度消失问题，当输入绝对值较大时梯度接近0
        - 输出不以0为中心，可能导致梯度更新效率低下
        - 在现代深度学习中，更多用于二分类输出层而非隐藏层
        - 计算相对复杂，涉及指数运算
    """
    def forward(self, x: TN) -> TN:
        return sigmoid(x)  # 直接调用全局sigmoid函数

class Softmax(Module):
    """Softmax激活模块 (Softmax Activation Function)
    
    应用Softmax函数沿指定维度对输入张量进行归一化，将输入转换为概率分布。
    常用于多分类问题的输出层，确保各类别概率之和为1。
    
    数学公式::
    
        Softmax(x_i) = exp(x_i) / sum(exp(x_j))
        
        其中求和沿指定维度dim进行
    
    Args:
        dim (int, optional): 应用Softmax的维度。
            沿此维度对输入进行归一化，使该维度上的元素和为1。
            对于N维输入，dim的取值范围为[-N, N-1]。
            默认值: -1 (最后一个维度)
    
    Shape:
        - Input: 任意形状的张量 (*)
        - Output: 与输入相同形状的张量 (*), 指定维度上的值和为1
    
    Examples::
    
        >>> m = Softmax(dim=1)
        >>> input = rm.randn(2, 3)  # batch_size=2, num_classes=3
        >>> output = m(input)
        >>> print(output.sum(dim=1))  # 每行的和为1
        >>> print(output.shape)  # 形状保持不变: (2, 3)
    
    Note:
        Softmax函数的特点：
        - 输出值在(0,1)之间，指定维度上的和为1，可解释为概率分布
        - 保持输入的相对大小关系，较大的输入对应较大的输出概率
        - 对输入的平移不敏感（加上常数后输出不变）
        - 常用于多分类任务的输出层
        - 数值稳定性重要，实际实现中通常会减去最大值防止溢出
        - 与交叉熵损失函数配合使用是标准的多分类配置
    """
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim  # 保存归一化维度参数
    
    def forward(self, x: TN) -> TN:
        return softmax(x, self.dim)  # 调用带维度参数的softmax函数
    
class GELU(Module):
    """高斯误差线性单元激活模块 (Gaussian Error Linear Unit)
    
    应用高斯误差线性单元函数逐元素到输入张量，是Transformer架构中
    最常用的激活函数，通过概率加权的方式实现平滑的非线性变换。
    
    数学公式::
    
        GELU(x) = x * Φ(x)
                 = x * 0.5 * [1 + erf(x / sqrt(2))]
        
        其中 Φ(x) 是标准正态分布的累积分布函数(CDF)
        
        近似计算公式（常用）::
        
        GELU(x) ≈ 0.5x * [1 + tanh(sqrt(2/π) * (x + 0.044715x³))]
    
    Args:
        无参数
    
    Shape:
        - Input: 任意形状的张量 (*)
        - Output: 与输入相同形状的张量 (*)
    
    Examples::
    
        >>> m = GELU()
        >>> input = rm.randn(2, 3)
        >>> output = m(input)
        >>> # 在Transformer中的应用
        >>> transformer_input = rm.randn(10, 512)  # batch_size=10, seq_len=512
        >>> activated = GELU()(transformer_input)
    
    Note:
        GELU函数的特点：
        - 结合了ReLU的概率特性和sigmoid的平滑性
        - 对于负值输入不完全置零，而是根据概率进行衰减
        - 在Transformer、BERT等现代模型中表现优异
        - 计算相对复杂，涉及误差函数或近似计算
        - 相比ReLU具有更好的梯度流动特性
        - 在自然语言处理任务中特别有效
        - 是当前大型语言模型的标准激活函数选择
    """
    def forward(self, x: TN) -> TN:
        return gelu(x)  # 直接调用全局gelu函数

class LogSoftmax(Module):
    """对数Softmax激活模块 (Logarithm of Softmax)
    
    应用LogSoftmax函数沿指定维度对输入张量进行变换，计算Softmax的对数值。
    常用于多分类问题中与负对数似然损失(NLLLoss)配合使用，提高数值稳定性。
    
    数学公式::
    
        LogSoftmax(x_i) = log(Softmax(x_i))
                        = log(exp(x_i) / sum(exp(x_j)))
                        = x_i - log(sum(exp(x_j)))
        
        其中求和沿指定维度dim进行
    
    Args:
        dim (int, optional): 应用LogSoftmax的维度。
            沿此维度对输入进行对数归一化。
            对于N维输入，dim的取值范围为[-N, N-1]。
            默认值: -1 (最后一个维度)
    
    Shape:
        - Input: 任意形状的张量 (*)
        - Output: 与输入相同形状的张量 (*), 值域为(-∞, 0)
    
    Examples::
    
        >>> m = LogSoftmax(dim=1)
        >>> input = rm.randn(2, 3)  # batch_size=2, num_classes=3
        >>> output = m(input)
        >>> print(output.shape)  # 形状保持不变: (2, 3)
        >>> print(output < 0)  # 所有值都为负数
        >>> # 与NLLLoss配合使用
        >>> nll_loss = rm.nn.NLLLoss()
        >>> target = rm.tensor([1, 2])
        >>> loss = nll_loss(output, target)
    
    Note:
        LogSoftmax函数的特点：
        - 输出值为负数，因为log(probability) ≤ 0
        - 与NLLLoss配合使用是标准的多分类训练配置
        - 相比Softmax+NLLLoss的组合，LogSoftmax+NLLLoss数值更稳定
        - 避免了Softmax计算中的数值溢出问题
        - 在计算交叉熵损失时可以直接使用，无需额外log操作
        - 是分类任务中最常用的激活函数之一
    """
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim  # 保存归一化维度参数
    
    def forward(self, x: TN) -> TN:
        return log_softmax(x, self.dim)
    
class Softplus(Module):
    """Softplus激活模块 (Softplus Activation Function)
    
    应用Softplus函数逐元素到输入张量，是ReLU的平滑近似版本。
    通过对数函数和指数函数的组合，实现处处可导的平滑激活。
    
    数学公式::
    
        Softplus(x) = log(1 + exp(beta * x)) / beta
        
        当 beta=1 时的简化形式：
        Softplus(x) = log(1 + exp(x))
        
        当 x 很大时：Softplus(x) ≈ x
        当 x 很小时：Softplus(x) ≈ exp(beta * x) / beta
    
    Args:
        beta (float, optional): 控制函数陡峭程度的参数。
            beta越大，函数越接近ReLU；beta越小，函数越平滑。
            默认值: 1
        threshold (float, optional): 数值稳定性阈值。
            当 beta * x > threshold 时，直接返回 x 以避免数值溢出。
            默认值: 20
    
    Shape:
        - Input: 任意形状的张量 (*)
        - Output: 与输入相同形状的张量 (*), 值域为(0, +∞)
    
    Examples::
    
        >>> m = Softplus(beta=1, threshold=20)
        >>> input = rm.randn(2, 3)
        >>> output = m(input)
        >>> print(output > 0)  # 所有值都为正数
        >>> # 与ReLU的比较
        >>> relu_out = rm.nn.ReLU()(input)
        >>> softplus_out = Softplus()(input)
        >>> # softplus_out是relu_out的平滑版本
    
    Note:
        Softplus函数的特点：
        - 处处可导且连续，是ReLU的光滑近似
        - 输出始终为正数，避免了ReLU的硬截止
        - 当输入很大时近似线性，类似ReLU
        - 当输入很小时近似指数函数，平滑过渡
        - 计算复杂度高于ReLU，涉及指数和对数运算
        - 在需要平滑激活函数的场合很有用
        - beta参数可以调节函数的陡峭程度
    """
    def __init__(self, beta=1, threshold=20):
        super().__init__()
        self.beta = beta
        self.threshold = threshold  # 保存平滑参数
    
    def forward(self, x: TN) -> TN:
        return softplus(x, self.beta, self.threshold)  # 调用带参数的实现

class Tanh(Module):
    """双曲正切激活模块 (Hyperbolic Tangent Activation Function)
    
    应用双曲正切函数逐元素到输入张量，将输入映射到(-1,1)区间。
    是经典的神经网络激活函数，具有零中心化的优良特性。
    
    数学公式::
    
        Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
                = sinh(x) / cosh(x)
                = 2 * sigmoid(2x) - 1
    
    导数公式::
    
        Tanh'(x) = 1 - Tanh²(x)
    
    Args:
        无参数
    
    Shape:
        - Input: 任意形状的张量 (*)
        - Output: 与输入相同形状的张量 (*), 值域为(-1,1)
    
    Examples::
    
        >>> m = Tanh()
        >>> input = rm.randn(2, 3)
        >>> output = m(input)
        >>> print(output.min(), output.max())  # 输出值在(-1,1)之间
        >>> print(output.mean())  # 均值接近0，具有零中心化特性
    
    Note:
        Tanh函数的特点：
        - 输出范围在(-1,1)之间，具有零中心化特性，有利于梯度更新
        - 函数光滑，处处可导，且导数在原点处最大(为1)
        - 存在梯度消失问题，当输入绝对值较大时梯度接近0
        - 相比Sigmoid，零中心化特性使训练更加稳定
        - 在RNN、LSTM等循环神经网络中广泛使用
        - 计算涉及指数运算，复杂度高于ReLU
        - 在现代深度学习中，更多用于特定场景而非通用激活函数
    """
    def forward(self, x: TN) -> TN:
        return tanh(x)  # 直接调用全局tanh函数
