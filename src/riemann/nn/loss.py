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
Loss Functions Module for the Riemann Library

This module provides implementations of common loss functions as neural network modules
for the Riemann library. Loss functions measure the discrepancy between predicted outputs
and target values, serving as optimization objectives during model training.

Implemented loss functions:
- MSELoss: Mean Squared Error loss, commonly used for regression tasks
- L1Loss: Mean Absolute Error loss, more robust to outliers
- SmoothL1Loss: Hybrid of L1 and L2 loss, used in object detection tasks
- CrossEntropyLoss: Direct loss for multi-class classification, combines LogSoftmax and NLLLoss
- BCEWithLogitsLoss: Binary cross entropy loss with built-in Sigmoid, for binary/multi-label classification
- HuberLoss: Similar to SmoothL1Loss but with different parameter definition
- NLLLoss: Negative Log Likelihood loss, typically used with LogSoftmax for multi-class classification

All loss modules inherit from the Module base class and implement a forward method
that calls the respective loss function from the functional module.
"""
from .module import *
from .functional import *


# MSELoss（均方误差）
class MSELoss(Module):
    """均方误差损失模块 (Mean Squared Error Loss)
    
    计算输入和目标之间均方误差的损失函数，是回归任务中最常用的损失函数之一。
    
    数学公式::
    
        loss(x, y) = (x - y)²
    
    如果 reduction 不是 'none'，则：
    
        loss(x, y) = mean((x - y)²)  当 reduction='mean'
        loss(x, y) = sum((x - y)²)   当 reduction='sum'
    
    Args:
        reduction (str, optional): 指定损失聚合方式。可选值：
            - 'none': 不进行聚合，返回每个元素的损失，形状与输入相同
            - 'mean': 返回所有元素损失的平均值
            - 'sum': 返回所有元素损失的总和
            默认值: 'mean'
    
    Shape:
        - Input: 任意形状的张量
        - Target: 与 input 相同形状的张量
        - Output: 如果 reduction 是 'none'，则形状与输入相同；否则为标量
    
    Examples::
    
        >>> loss = MSELoss()
        >>> input = rm.randn(3, 5, requires_grad=True)
        >>> target = rm.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    
    Note:
        该损失函数对大误差比较敏感，因为误差是平方的。在存在异常值的情况下，
        可能需要考虑使用 L1Loss 或 SmoothL1Loss 来获得更稳健的结果。
    """
    def __init__(self, reduction='mean'):
        """初始化MSELoss模块
        
        Args:
            reduction (str, optional): 损失聚合方式，可选值为'none'、'mean'或'sum'
                - 'none': 返回每个元素的损失值，形状与输入相同
                - 'mean': 返回所有元素损失值的平均值
                - 'sum': 返回所有元素损失值的总和
                默认值: 'mean'
        """
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError("reduction must be 'none', 'mean' or 'sum'")
        self.reduction = reduction

    def forward(self, input: TN, target: TN) -> TN:
        return mse_loss(input, target, reduction=self.reduction)
    
class L1Loss(Module):
    """平均绝对误差损失模块 (Mean Absolute Error Loss)
    
    计算输入和目标之间平均绝对误差的损失函数，对异常值具有较好的稳健性。
    
    数学公式::
    
        loss(x, y) = |x - y|
    
    如果 reduction 不是 'none'，则：
    
        loss(x, y) = mean(|x - y|)  当 reduction='mean'
        loss(x, y) = sum(|x - y|)   当 reduction='sum'
    
    Args:
        reduction (str, optional): 指定损失聚合方式。可选值：
            - 'none': 不进行聚合，返回每个元素的损失，形状与输入相同
            - 'mean': 返回所有元素损失的平均值
            - 'sum': 返回所有元素损失的总和
            默认值: 'mean'
    
    Shape:
        - Input: 任意形状的张量
        - Target: 与 input 相同形状的张量
        - Output: 如果 reduction 是 'none'，则形状与输入相同；否则为标量
    
    Examples::
    
        >>> loss = L1Loss()
        >>> input = rm.randn(3, 5, requires_grad=True)
        >>> target = rm.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    
    Note:
        与 MSELoss 相比，L1Loss 对异常值更加稳健，因为它不会对误差进行平方。
        但是在接近零点时，L1Loss 的梯度不可导，这可能导致训练过程中的不稳定。
        如果希望在误差小时有平滑的梯度，可以考虑使用 SmoothL1Loss。
    """
    def __init__(self, reduction='mean'):
        """初始化L1Loss模块
        
        Args:
            reduction (str, optional): 损失聚合方式，可选值为'none'、'mean'或'sum'
                - 'none': 返回每个元素的损失值，形状与输入相同
                - 'mean': 返回所有元素损失值的平均值
                - 'sum': 返回所有元素损失值的总和
                默认值: 'mean'
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, input: TN, target: TN) -> TN:
        return l1_loss(input, target, reduction=self.reduction)
        
class SmoothL1Loss(Module):
    """平滑L1损失模块 (Smooth L1 Loss)
    
    结合了L1损失和L2损失优点的损失函数，在误差较小时使用L2损失，误差较大时使用L1损失。
    
    数学公式::
    
        loss(x, y) = 0.5 * (x - y)² / beta,            当 |x - y| < beta
        loss(x, y) = |x - y| - 0.5 * beta,             当 |x - y| >= beta
    
    如果 reduction 不是 'none'，则对上述结果进行相应的聚合。
    
    Args:
        beta (float, optional): 控制从二次函数到线性函数转换的阈值。
            当误差绝对值小于beta时，使用二次函数形式；否则使用线性函数形式。
            默认值: 1.0
        reduction (str, optional): 指定损失聚合方式。可选值：
            - 'none': 不进行聚合，返回每个元素的损失，形状与输入相同
            - 'mean': 返回所有元素损失的平均值
            - 'sum': 返回所有元素损失的总和
            默认值: 'mean'
    
    Shape:
        - Input: 任意形状的张量
        - Target: 与 input 相同形状的张量
        - Output: 如果 reduction 是 'none'，则形状与输入相同；否则为标量
    
    Examples::
    
        >>> loss = SmoothL1Loss(beta=1.0)
        >>> input = rm.randn(3, 5, requires_grad=True)
        >>> target = rm.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    
    Note:
        SmoothL1Loss 在目标检测任务中经常使用（如Fast R-CNN），
        它在误差较小时具有平滑的梯度（类似L2），在误差较大时对异常值稳健（类似L1）。
        当beta=1.0时，该损失函数等价于PyTorch中的SmoothL1Loss。
    """
    def __init__(self, beta=1.0, reduction='mean'):
        """初始化SmoothL1Loss模块
        
        Args:
            beta (float, optional): 平滑过渡的阈值，默认值为1.0
            reduction (str, optional): 损失聚合方式，可选值为'none'、'mean'或'sum'
                - 'none': 返回每个元素的损失值，形状与输入相同
                - 'mean': 返回所有元素损失值的平均值
                - 'sum': 返回所有元素损失值的总和
                默认值: 'mean'
        """
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, input: TN, target: TN) -> TN:
        return smooth_l1_loss(input, target, beta=self.beta, reduction=self.reduction)
    
class CrossEntropyLoss(Module):
    """交叉熵损失模块 (Cross Entropy Loss)
    
    结合了LogSoftmax和NLLLoss的多分类损失函数，直接计算原始输入和目标之间的交叉熵损失。
    
    该损失函数对于多分类任务非常有用，它期望输入包含每个类别的原始、未归一化的分数，
    并且会内部应用LogSoftmax进行归一化。
    
    数学公式::
    
        loss(x, class) = -log(exp(x[class]) / sum_j(exp(x[j])))
                       = -x[class] + log(sum_j(exp(x[j])))
    
    如果包含权重和标签平滑，则公式会更复杂。
    
    Args:
        weight (TN, optional): 手动指定每个类别的权重。如果提供，形状应为(C,)，
            其中C是类别数。这对于处理类别不平衡问题很有用。
            默认值: None
        ignore_index (int, optional): 指定一个目标值，该值对应的损失不会被计算。
            这在处理填充标记或需要忽略某些样本时很有用。
            默认值: -100
        reduction (str, optional): 指定损失聚合方式。可选值：
            - 'none': 不进行聚合，返回每个样本的损失
            - 'mean': 返回所有样本损失的平均值
            - 'sum': 返回所有样本损失的总和
            默认值: 'mean'
        label_smoothing (float, optional): 标签平滑系数，范围在[0.0, 1.0]之间。
            当大于0时，会使标签更加平滑，有助于防止过拟合。
            0.0表示不进行标签平滑。
            默认值: 0.0
    
    Shape:
        - Input: (N, C)，其中N是批次大小，C是类别数
        - Target: (N,)，包含每个样本的目标类别索引，值在[0, C-1]范围内
        - Output: 标量，或者当reduction='none'时形状为(N,)
    
    Examples::
    
        >>> loss = CrossEntropyLoss()
        >>> input = rm.randn(3, 5, requires_grad=True)  # 3个样本，5个类别
        >>> target = rm.tensor([1, 0, 4])  # 目标类别索引
        >>> output = loss(input, target)
        >>> output.backward()
    
    Note:
        与先应用LogSoftmax再使用NLLLoss相比，直接使用CrossEntropyLoss数值上更稳定。
        该函数期望输入是原始分数（logits），而不是经过softmax的概率。
    """
    def __init__(self, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
        """初始化CrossEntropyLoss模块
        
        Args:
            weight (TN, optional): 各类别的权重张量，形状为[C]，其中C为类别数
            ignore_index (int): 指定要忽略的目标值索引，该索引处的损失不会被计算
            reduction (str, optional): 损失聚合方式，可选值为'none'、'mean'或'sum'
                - 'none': 返回每个元素的损失值，形状与输入相同
                - 'mean': 返回所有元素损失值的平均值
                - 'sum': 返回所有元素损失值的总和
                默认值: 'mean'
            label_smoothing (float, optional): 标签平滑系数，范围在[0,1]之间，用于防止过拟合
                默认值: 0.0
        """
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, input: TN, target: TN) -> TN:
        return cross_entropy(input, target, weight=self.weight,
                 size_average=None, ignore_index=self.ignore_index,
                 reduce=None, reduction=self.reduction,
                 label_smoothing=self.label_smoothing)
    
class BCEWithLogitsLoss(Module):
    """带Sigmoid的二元交叉熵损失模块 (Binary Cross Entropy with Logits Loss)
    
    结合了Sigmoid激活函数和BCELoss的二元分类损失函数，数值上比分别使用Sigmoid和BCELoss更稳定。
    
    该损失函数适用于二元分类或多标签分类任务，它直接接受原始输入（logits），
    内部会应用Sigmoid函数进行概率转换。
    
    数学公式::
    
        loss(x, y) = -[y * log(sigmoid(x)) + (1 - y) * log(1 - sigmoid(x))]
        
        其中 sigmoid(x) = 1 / (1 + exp(-x))
    
    如果包含权重，则：
    
        loss(x, y) = -[pos_weight * y * log(sigmoid(x)) + 
                      (1 - y) * log(1 - sigmoid(x))] * weight
    
    Args:
        weight (TN, optional): 手动指定每个样本的权重。如果提供，形状应与输入相同。
            这对于处理样本不平衡问题很有用。
            默认值: None
        pos_weight (TN, optional): 正类别的权重。如果提供，形状应为(C,)，
            其中C是类别数。这用于增加正类损失的权重，有助于处理类别不平衡。
            默认值: None
        reduction (str, optional): 指定损失聚合方式。可选值：
            - 'none': 不进行聚合，返回每个元素的损失
            - 'mean': 返回所有元素损失的平均值
            - 'sum': 返回所有元素损失的总和
            默认值: 'mean'
    
    Shape:
        - Input: (N, *)，其中N是批次大小，*可以是任意数量的额外维度
        - Target: 与 input 相同形状，值在[0, 1]范围内
        - Output: 标量，或者当reduction='none'时形状与输入相同
    
    Examples::
    
        >>> loss = BCEWithLogitsLoss()
        >>> input = rm.randn(3, requires_grad=True)
        >>> target = rm.tensor([0.0, 1.0, 1.0])
        >>> output = loss(input, target)
        >>> output.backward()
    
    Note:
        与先应用Sigmoid再使用BCELoss相比，直接使用BCEWithLogitsLoss数值上更稳定，
        因为它避免了当sigmoid输出接近0或1时的数值下溢问题。
        该函数期望输入是原始分数（logits），而不是经过sigmoid的概率。
    """
    def __init__(self, weight=None, pos_weight=None, reduction='mean'):
        """初始化BCEWithLogitsLoss模块
        
        Args:
            weight (TN, optional): 每个样本的权重张量，形状与输入相同
            pos_weight (TN, optional): 正类的权重张量，用于处理类别不平衡问题
            reduction (str, optional): 损失聚合方式，可选值为'none'、'mean'或'sum'
                - 'none': 返回每个元素的损失值，形状与输入相同
                - 'mean': 返回所有元素损失值的平均值
                - 'sum': 返回所有元素损失值的总和
                默认值: 'mean'
        """
        super().__init__()
        self.weight = weight
        self.pos_weight = pos_weight
        self.reduction = reduction
        
        # 参数校验
        if pos_weight is not None and not isinstance(pos_weight, TN):
            raise ValueError("pos_weight must be a TN tensor")
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Invalid reduction: {reduction}")

    def forward(self, input: TN, target: TN) -> TN:
        return binary_cross_entropy_with_logits(input, target, 
                            weight=self.weight, 
                            pos_weight=self.pos_weight, 
                            reduction=self.reduction)

class HuberLoss(Module):
    """Huber损失模块 (Huber Loss)
    
    结合了L1损失和L2损失优点的稳健回归损失函数，与SmoothL1Loss类似但参数定义不同。
    
    Huber损失在误差较小时使用二次函数（L2风格），在误差较大时使用线性函数（L1风格），
    从而在保持对异常值稳健性的同时，在误差较小时提供平滑的梯度。
    
    数学公式::
    
        loss(x, y) = 0.5 * (x - y)²,                   当 |x - y| <= delta
        loss(x, y) = delta * (|x - y| - 0.5 * delta),  当 |x - y| > delta
    
    如果 reduction 不是 'none'，则对上述结果进行相应的聚合。
    
    Args:
        delta (float, optional): 控制从二次函数到线性函数转换的阈值。
            当误差绝对值小于等于delta时，使用二次函数形式；否则使用线性函数形式。
            较小的delta使得损失函数更早转换为线性形式，更加稳健。
            默认值: 1.0
        reduction (str, optional): 指定损失聚合方式。可选值：
            - 'none': 不进行聚合，返回每个元素的损失，形状与输入相同
            - 'mean': 返回所有元素损失的平均值
            - 'sum': 返回所有元素损失的总和
            默认值: 'mean'
    
    Shape:
        - Input: 任意形状的张量
        - Target: 与 input 相同形状的张量
        - Output: 如果 reduction 是 'none'，则形状与输入相同；否则为标量
    
    Examples::
    
        >>> loss = HuberLoss(delta=1.0)
        >>> input = rm.randn(3, 5, requires_grad=True)
        >>> target = rm.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    
    Note:
        HuberLoss与SmoothL1Loss在数学上是等价的，只是参数定义不同：
        SmoothL1Loss的beta参数等于HuberLoss的delta参数。
        该损失函数在统计学中很常用，特别是在稳健回归问题中。
    """
    def __init__(self, delta=1.0, reduction='mean'):
        """初始化HuberLoss模块
        
        Args:
            delta (float, optional): 损失函数从二次形式转变为线性形式的阈值，默认值为1.0
            reduction (str, optional): 损失聚合方式，可选值为'none'、'mean'或'sum'
                - 'none': 返回每个元素的损失值，形状与输入相同
                - 'mean': 返回所有元素损失值的平均值
                - 'sum': 返回所有元素损失值的总和
                默认值: 'mean'
        """
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, input: TN, target: TN) -> TN:
        return huber_loss(input, target, 
                        delta=self.delta, 
                        reduction=self.reduction)

class NLLLoss(Module):
    """负对数似然损失模块 (Negative Log Likelihood Loss)
    
    用于多分类任务的损失函数，期望输入已经经过LogSoftmax处理的对数概率。
    
    该损失函数通常与LogSoftmax层配合使用，计算目标类别对应的负对数概率值。
    它适用于需要自定义输入归一化方式的多分类场景。
    
    数学公式::
    
        loss(x, class) = -x[class]
    
    如果包含权重，则：
    
        loss(x, class) = -weight[class] * x[class]
    
    Args:
        weight (TN, optional): 手动指定每个类别的权重。如果提供，形状应为(C,)，
            其中C是类别数。这对于处理类别不平衡问题很有用。
            默认值: None
        ignore_index (int, optional): 指定一个目标值，该值对应的损失不会被计算。
            这在处理填充标记或需要忽略某些样本时很有用。
            默认值: -100
        reduction (str, optional): 指定损失聚合方式。可选值：
            - 'none': 不进行聚合，返回每个样本的损失
            - 'mean': 返回所有样本损失的平均值
            - 'sum': 返回所有样本损失的总和
            默认值: 'mean'
    
    Shape:
        - Input: (N, C)，其中N是批次大小，C是类别数，包含对数概率
        - Target: (N,)，包含每个样本的目标类别索引，值在[0, C-1]范围内
        - Output: 标量，或者当reduction='none'时形状为(N,)
    
    Examples::
    
        >>> m = rm.LogSoftmax(dim=1)
        >>> loss = NLLLoss()
        >>> input = rm.randn(3, 5, requires_grad=True)
        >>> target = rm.tensor([1, 0, 4])
        >>> output = loss(m(input), target)
        >>> output.backward()
    
    Note:
        输入应该是已经经过LogSoftmax处理的对数概率，而不是原始分数。
        如果原始分数需要同时进行归一化和损失计算，建议使用CrossEntropyLoss。
        该损失函数期望的目标值是类别索引，而不是one-hot编码。
    """
    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        """初始化NLLLoss模块
        
        Args:
            weight (TN, optional): 各类别的权重张量，形状为[C]，其中C为类别数
            ignore_index (int): 指定要忽略的目标值索引，该索引处的损失不会被计算
            reduction (str, optional): 损失聚合方式，可选值为'none'、'mean'或'sum'
                - 'none': 返回每个元素的损失值，形状与输入相同
                - 'mean': 返回所有元素损失值的平均值
                - 'sum': 返回所有元素损失值的总和
                默认值: 'mean'
        """
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(self, input: TN, target: TN) -> TN:
        return nll_loss(
            input, 
            target, 
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction
        )