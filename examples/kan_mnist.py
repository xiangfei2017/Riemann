#!/usr/bin/env python3
"""
MNIST手写数字识别KAN网络示例

本文件实现了一个基于Riemann框架的Kolmogorov-Arnold网络(KAN)，
用于识别MNIST手写数字数据集。KAN使用可学习的B样条激活函数
替代传统的固定激活函数。

网络架构:
- 输入层: 784个神经元 (28×28像素的MNIST图像展平后)
- 隐藏层: 64个神经元，使用KAN层
- 输出层: 10个神经元，对应10个数字类别(0-9)

损失函数: 交叉熵损失函数 (CrossEntropyLoss)
优化器: AdamW优化器，学习率0.001，weight_decay=0.0001
学习率调度: 指数衰减调度器，gamma=0.8

数据预处理:
- 图像数据归一化到[-1, 1]范围
- 标签直接使用类别索引，适配CrossEntropyLoss

特点:
- 使用KAN网络，可学习的激活函数
- 支持自适应网格更新
- 支持批量数据加载，提高训练效率
- 提供详细的训练进度和性能指标
- 包含完整的训练和验证评估流程

作者: Riemann框架示例
日期: 2025
"""
import sys
import os
import time
from tqdm import tqdm

# 添加项目根目录到Python路径，确保可以导入riemann模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(project_root)

# 导入riemann包中的模块
import riemann.nn as nn
import riemann.optim as optim
from riemann.vision import datasets, transforms
from riemann.utils.data import DataLoader
from riemann import cuda
from riemann import no_grad


def clear_screen():
    """跨平台清屏函数（自动识别系统）"""
    if os.name == 'nt':
        os.system('cls')  # Windows系统指令
    else:
        os.system('clear')  # Linux/MacOS指令
    return


def main():
    """
    主函数
    
    执行完整的MNIST手写数字识别流程：
    1. 定义数据预处理变换
    2. 加载训练和验证数据集
    3. 创建数据加载器
    4. 初始化KAN模型
    5. 训练模型多个epoch
    6. 评估模型性能
    7. 输出训练时间和最终准确率
    """
    clear_screen()
    print("MNIST手写数字识别KAN网络示例")
    
    # 定义数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1, 1]
    ])
    
    data_root = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    # 加载训练和验证数据集
    print("加载数据集...")
    train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        transform=transform,
        download=True  # 下载数据集到本地
    )
    val_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        transform=transform,
        download=True  # 下载数据集到本地
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=64,
        shuffle=False
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 检查CUDA可用性
    print("\n初始化模型...")
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 定义模型
    model = nn.KAN([28 * 28, 64, 10])
    model.to(device)
    
    # 定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 定义学习率调度器
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 训练模型
    print("\n开始训练...")
    epochs = 3
    train_start_time = time.time()
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        with tqdm(train_loader) as pbar:
            for i, (images, labels) in enumerate(pbar):
                images = images.view(-1, 28 * 28).to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad(True)
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                
                accuracy = (output.argmax(dim=1) == labels).float().mean()
                pbar.set_postfix(
                    loss=loss.item(),
                    accuracy=accuracy.item(),
                    lr=optimizer.param_groups[0]['lr']
                )
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_accuracy = 0
        with no_grad():
            for images, labels in val_loader:
                images = images.view(-1, 28 * 28).to(device)
                output = model(images)
                val_loss += criterion(output, labels.to(device)).item()
                val_accuracy += (
                    (output.argmax(dim=1) == labels.to(device)).float().mean().item()
                )
        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        
        # 更新学习率
        scheduler.step()
        
        print(f"Epoch {epoch + 1}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    train_end_time = time.time()
    print(f"\n训练总时间: {train_end_time - train_start_time:.2f}秒")


if __name__ == "__main__":
    # 当脚本直接运行时执行主函数
    main()
