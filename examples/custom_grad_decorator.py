#!/usr/bin/env python3
"""
Riemann 自动微分自定义梯度示例
使用 track_grad 装饰器自定义 Sigmoid 函数的梯度

该示例演示了如何使用 Riemann 库的 track_grad 装饰器为自定义函数
添加自动微分支持。我们以 Sigmoid 函数为例，实现了其导数计算
并通过装饰器绑定到自定义的 sigmoid 函数。

版本: 1.0
日期: 2025
"""

import riemann as rm
import numpy as np

def sigmoid_derivative(x: rm.TN) -> tuple[rm.TN]:
    """
    Sigmoid函数的导数函数，用于自定义梯度计算。
    公式：d/dx (sigmoid(x)) = sigmoid(x) * (1 - sigmoid(x))
    
    Args:
        x: 输入张量
        
    Returns:
        tuple[rm.TN]: 包含单个元素的元组，表示输入x的梯度
    """
    # 计算sigmoid值
    exp_val = rm.exp(-x)
    sig = 1.0 / (1.0 + exp_val)
    
    # 计算导数
    derivative = sig * (1. - sig)
    
    return (derivative,)

@rm.track_grad(sigmoid_derivative)
def custom_sigmoid(x: rm.TN) -> rm.TN:
    """
    使用track_grad修饰器自定义的sigmoid函数，支持自动微分。
    
    Args:
        x: 输入张量
        
    Returns:
        rm.TN: sigmoid(x)的结果
    """
    # 前向计算：使用numpy实现sigmoid，并确保数据类型与输入一致
    exp_val = np.exp(-x.data)  # 使用numpy进行前向计算
    sig = 1.0 / (1.0 + exp_val)
    
    # 返回与输入具有相同数据类型的张量
    return rm.tensor(sig, dtype=x.dtype)

if __name__ == "__main__":
    print("自定义Sigmoid函数梯度示例")
    print("=" * 40)
    
    # 创建输入张量
    x = rm.tensor(2.0, requires_grad=True, dtype='float64')
    print(f"输入 x = {x}")
    
    # 使用自定义sigmoid函数
    y = custom_sigmoid(x)
    print(f"自定义sigmoid(x) = {y}")
    
    # 执行反向传播
    y.backward()
    print(f"反向传播后 x.grad = {x.grad}")
    
    # 验证梯度计算是否正确
    # 使用Riemann内置的梯度计算作为参考
    x2 = rm.tensor(2.0, requires_grad=True, dtype='float64')
    
    # 使用内置函数计算sigmoid
    exp_val = rm.exp(-x2)
    y2 = 1.0 / (1.0 + exp_val)
    
    y2.backward()
    print(f"内置梯度计算 x.grad = {x2.grad}")
    
    print(f"\n梯度比较结果: {'一致' if (x.grad - x2.grad).abs() < 1e-12 else '不一致'}")
    print(f"差值: {(x.grad - x2.grad).abs()}")