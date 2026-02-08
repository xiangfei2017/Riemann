#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
学习率调度器使用示例

此脚本展示了如何在 Riemann 中使用不同的学习率调度器，包括：
1. StepLR - 按固定步长调整学习率
2. MultiStepLR - 在指定的里程碑处调整学习率
3. ExponentialLR - 按指数衰减调整学习率
4. CosineAnnealingLR - 按余弦退火调整学习率
5. ReduceLROnPlateau - 根据性能指标调整学习率
"""

import numpy as np
import riemann as rm
from riemann.optim import SGD
from riemann.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
)


class SimpleModel(rm.nn.Module):
    """简单的线性模型用于演示"""
    def __init__(self):
        super().__init__()
        self.linear = rm.nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)


def generate_sample_data():
    """生成示例数据"""
    # 生成输入数据
    x = rm.tensor(np.random.randn(100, 10).astype(np.float32))
    # 生成目标数据（线性关系）
    weights = np.random.randn(10, 1).astype(np.float32)
    bias = np.random.randn(1).astype(np.float32)
    y = rm.tensor(np.dot(x.data, weights) + bias)
    return x, y


def train_with_scheduler(model, optimizer, scheduler, x, y, num_epochs=10):
    """使用指定的调度器训练模型"""
    print(f"\n使用 {scheduler.__class__.__name__} 调度器训练模型")
    
    # 记录学习率变化
    lr_history = []
    
    for epoch in range(num_epochs):
        # 前向传播
        output = model(x)
        # 计算损失
        loss = rm.nn.functional.mse_loss(output, y)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        # 打印信息
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.data:.4f}, LR: {current_lr:.6f}")
        # 更新学习率
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(loss.data)
        else:
            scheduler.step()
        # 清零梯度
        optimizer.zero_grad()
    
    print(f"学习率变化: {[round(lr, 6) for lr in lr_history]}")
    return lr_history


def main():
    """主函数"""
    print("Riemann 学习率调度器使用示例")
    print("=" * 60)
    
    # 生成示例数据
    x, y = generate_sample_data()
    
    # 1. 使用 StepLR 调度器
    print("\n1. StepLR 调度器")
    print("-" * 40)
    model1 = SimpleModel()
    optimizer1 = SGD(model1.parameters(), lr=0.1)
    scheduler1 = StepLR(optimizer1, step_size=3, gamma=0.1)
    train_with_scheduler(model1, optimizer1, scheduler1, x, y)
    
    # 2. 使用 MultiStepLR 调度器
    print("\n2. MultiStepLR 调度器")
    print("-" * 40)
    model2 = SimpleModel()
    optimizer2 = SGD(model2.parameters(), lr=0.1)
    scheduler2 = MultiStepLR(optimizer2, milestones=[3, 6, 9], gamma=0.1)
    train_with_scheduler(model2, optimizer2, scheduler2, x, y)
    
    # 3. 使用 ExponentialLR 调度器
    print("\n3. ExponentialLR 调度器")
    print("-" * 40)
    model3 = SimpleModel()
    optimizer3 = SGD(model3.parameters(), lr=0.1)
    scheduler3 = ExponentialLR(optimizer3, gamma=0.9)
    train_with_scheduler(model3, optimizer3, scheduler3, x, y)
    
    # 4. 使用 CosineAnnealingLR 调度器
    print("\n4. CosineAnnealingLR 调度器")
    print("-" * 40)
    model4 = SimpleModel()
    optimizer4 = SGD(model4.parameters(), lr=0.1)
    scheduler4 = CosineAnnealingLR(optimizer4, T_max=5, eta_min=0.001)
    train_with_scheduler(model4, optimizer4, scheduler4, x, y)
    
    # 5. 使用 ReduceLROnPlateau 调度器
    print("\n5. ReduceLROnPlateau 调度器")
    print("-" * 40)
    model5 = SimpleModel()
    optimizer5 = SGD(model5.parameters(), lr=0.1)
    scheduler5 = ReduceLROnPlateau(
        optimizer5, mode='min', factor=0.1, patience=2, threshold=1e-4, verbose=True
    )
    train_with_scheduler(model5, optimizer5, scheduler5, x, y)


if __name__ == "__main__":
    main()
