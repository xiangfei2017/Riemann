#!/usr/bin/env python3
"""
Riemann 自动微分自定义梯度示例
使用 Function 类自定义 Sigmoid 函数的梯度

该示例演示了如何使用 Riemann 库的 Function 基类来创建自定义的
可微函数。Function 类是 Riemann 自定义梯度的核心接口，要求实现 forward
和 backward 两个静态方法。

我们以 Sigmoid 函数为例，展示了如何完整地实现前向计算和反向梯度
传播，以及如何正确处理上下文信息保存、链式法则应用等关键步骤。

版本: 1.0
日期: 2025
"""

import riemann as rm
import numpy as np

class CustomSigmoid(rm.autograd.Function):
    """
    使用 Function 类自定义的 Sigmoid 函数类，支持自动微分
    
    Riemann 的 Function 类是自定义梯度的基类，参考了 PyTorch 的 torch.autograd.Function 接口设计
    必须实现 forward 和 backward 两个静态方法
    """
    
    @staticmethod
    def forward(ctx, x: rm.TN) -> rm.TN:
        """
        前向计算函数
        
        Args:
            ctx: 上下文对象，用于保存信息供 backward 方法使用
            x: 输入张量
            
        Returns:
            Sigmoid 函数的计算结果
        """
        # 使用 numpy 进行前向计算，因为 numpy 的指数运算更高效
        exp_val = np.exp(-x.data)  # 计算 e^(-x)
        sig = 1.0 / (1.0 + exp_val)  # 计算 sigmoid
        
        # 将 sigmoid 结果保存到上下文中，以便在 backward 中使用
        ctx.save_for_backward(rm.tensor(sig, dtype=x.dtype))
        
        # 返回与输入具有相同数据类型的张量
        return rm.tensor(sig, dtype=x.dtype)
    
    @staticmethod
    def backward(ctx, grad_output: rm.TN) -> rm.TN:
        """
        反向传播函数
        
        Args:
            ctx: 上下文对象，包含 forward 中保存的信息
            grad_output: 输出张量的梯度
            
        Returns:
            输入张量 x 的梯度
        """
        # 从上下文中获取保存的 sigmoid 值
        sig, = ctx.saved_tensors
        
        # 计算 sigmoid 函数的导数：sigmoid(x) * (1 - sigmoid(x))
        # 然后与输出梯度相乘（链式法则）
        grad_input = grad_output * sig * (1 - sig)
        
        return grad_input

if __name__ == "__main__":
    print("使用 Function 类自定义 Sigmoid 函数梯度示例")
    print("=" * 60)
    
    # 创建输入张量
    x = rm.tensor(2.0, requires_grad=True, dtype='float64')
    print(f"输入 x = {x}")
    print(f"x 的数据类型: {x.dtype}")
    
    # 使用自定义的 Sigmoid 函数
    y = CustomSigmoid.apply(x)
    print(f"\n自定义 Sigmoid(x) = {y}")
    print(f"y 的数据类型: {y.dtype}")
    
    # 执行反向传播
    y.backward()
    print(f"\n反向传播后 x.grad = {x.grad}")
    
    # 使用 Riemann 内置函数验证梯度计算
    print(f"\n{'-' * 50}")
    print("使用内置函数验证梯度计算")
    print('-' * 50)
    
    x2 = rm.tensor(2.0, requires_grad=True, dtype='float64')
    
    # 使用内置函数计算 sigmoid
    exp_val = rm.exp(-x2)
    y2 = 1.0 / (1.0 + exp_val)
    
    y2.backward()
    print(f"内置梯度计算 x.grad = {x2.grad}")
    
    # 比较结果
    print(f"\n{'-' * 50}")
    print("梯度验证结果")
    print('-' * 50)
    print(f"自定义梯度: {x.grad}")
    print(f"内置梯度: {x2.grad}")
    print(f"差值: {rm.abs(x.grad - x2.grad)}")
    print(f"验证通过: {'✓' if rm.abs(x.grad - x2.grad) < 1e-12 else '✗'}")
    print(f"梯度比较: {'一致' if rm.abs(x.grad - x2.grad) < 1e-12 else '不一致'}")
