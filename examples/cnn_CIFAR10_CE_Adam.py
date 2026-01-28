#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   cnn_CIFAR10_advanced.py
@Time    :   2025/12/27
@Author  :   Advanced CNN
@Version :   1.1
@Desc    :   基于CFAIR10数据集特点优化的高级卷积神经网络示例（业界最佳实践版）
"""

import os
import time
from tqdm import tqdm
from riemann import nn, optim
from riemann.vision import datasets, transforms
from riemann.utils.data import DataLoader
from riemann import tensor, get_default_dtype
from riemann import cuda


def clear_screen():
    """跨平台清屏函数（自动识别系统）"""
    if os.name == 'nt':
        os.system('cls')  # Windows系统指令
    else:
        os.system('clear')  # Linux/MacOS指令
    return


class EasyCIFAR10(datasets.CIFAR10):
    """
    继承自CFAIR10的子类，在初始化时对图像数据应用归一化、标准化转换，
    这样训练过程中不再需要转换数据，可节省训练时间。
    """
    
    def __init__(self, root, train=True):
        """
        初始化预处理的CIFAR-10数据集。
        
        参数:
            root (str): 数据集的根目录
            train (bool): 是否加载训练集，默认为True
        """
        # 定义图像转换：将PIL图像转换为张量并标准化
        def tensor_transform(img):
            # 将图像转换为张量
            tensor_img = transforms.ToTensor()(img)
            # 使用CFAIR10数据集的实际均值和标准差进行标准化
            # 根据数据分析：R(134.57,69.84), G(135.19,69.56), B(136.95,66.95)
            # 转换为0-1范围的均值和标准差
            normalized_img = transforms.Normalize(
                mean=(134.57/255, 135.19/255, 136.95/255), 
                std=(69.84/255, 69.56/255, 66.95/255)
            )(tensor_img)
            return normalized_img
        
        def tensor_label_transform(label):
            return tensor(label, dtype=get_default_dtype())
        
        # 初始化父类，传入转换函数
        super().__init__(root, train=train, transform=tensor_transform, target_transform=tensor_label_transform)
        
        # 预处理所有数据
        print("Transforming CFAIR-10 to EasyCFAIR10 ...")
        self.data_list = []
        # 使用父类的长度，而不是子类的长度
        parent_length = super().__len__()
        for i in tqdm(range(parent_length)):
            # 通过父类的__getitem__获取转换后的数据
            self.data_list.append(super().__getitem__(i))
    
    def __len__(self):
        """
        返回数据集的大小。
        
        返回:
            int: 数据集中的样本数量
        """
        return len(self.data_list)
    
    def __getitem__(self, index):
        """
        获取指定索引的样本。
        
        参数:
            index (int): 样本索引
            
        返回:
            tuple: (image, target) 预处理后的图像张量和目标张量的元组
        """
        return self.data_list[index]


class AdvancedConvNet(nn.Module):
    """
    基于CIFAR10数据集特点优化的高级卷积神经网络（业界最佳实践版）
    
    优化策略：
    1. 采用经典的VGG风格网络结构，逐步增加通道数
    2. 使用适当的卷积核大小和步长，平衡特征提取和计算效率
    3. 合理设置全连接层参数，避免过拟合
    4. 完整使用批量归一化和dropout等正则化技术
    5. 使用ReLU激活函数，提高训练稳定性
    
    网络结构:
        - 输入块: Conv(3->32, 3x3) -> ReLU -> Conv(32->32, 3x3) -> ReLU -> MaxPool(2x2)
        - 中间块1: Conv(32->64, 3x3) -> ReLU -> Conv(64->64, 3x3) -> ReLU -> MaxPool(2x2)
        - 中间块2: Conv(64->128, 3x3) -> ReLU -> Conv(128->128, 3x3) -> ReLU -> MaxPool(2x2)
        - 全连接层: Linear(128*4*4->512) -> ReLU -> Dropout -> Linear(512->10)
    
    方法:
        forward(inputs): 前向传播计算
        train_step(inputs, targets): 执行一步训练
        evaluate(dataloader): 评估模型在数据集上的性能
    """
    def __init__(self):
        """
        初始化分类器模型
        
        基于CIFAR10数据集特点创建优化的网络结构。
        """
        super().__init__()
    
        # 输入块：提取低级特征
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)  # 32x32 -> 32x32
        self.bn1 = nn.BatchNorm2d(num_features=32)  # 添加批量归一化
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)  # 32x32 -> 32x32
        self.bn2 = nn.BatchNorm2d(num_features=32)  # 添加批量归一化
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16
        self.dropout1 = nn.Dropout(p=0.2)
        
        # 中间块1：提取中级特征
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # 16x16 -> 16x16
        self.bn3 = nn.BatchNorm2d(num_features=64)  # 添加批量归一化
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)  # 16x16 -> 16x16
        self.bn4 = nn.BatchNorm2d(num_features=64)  # 添加批量归一化
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8
        self.dropout2 = nn.Dropout(p=0.3)
        
        # 中间块2：提取高级特征
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # 8x8 -> 8x8
        self.bn5 = nn.BatchNorm2d(num_features=128)  # 添加批量归一化
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)  # 8x8 -> 8x8
        self.bn6 = nn.BatchNorm2d(num_features=128)  # 添加批量归一化
        self.relu6 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8 -> 4x4
        self.dropout3 = nn.Dropout(p=0.4)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # 全连接层
        self.bn7 = nn.BatchNorm1d(num_features=512)  # 添加批量归一化
        self.relu_fc1 = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 10)
        
        # 交叉熵损失函数，适用于多分类任务
        self.loss_func = nn.CrossEntropyLoss()
        # 使用Adam优化器，自适应学习率
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=5e-4)
    
    def forward(self, inputs):
        """
        前向传播计算
        
        参数:
            inputs (Tensor): 输入张量，形状为(batch_size, 3, 32, 32)
                           表示一批CIFAR-10彩色图像数据
        
        返回:
            Tensor: 输出张量，形状为(batch_size, 10)
                   表示每个样本属于10个类别的未归一化对数概率
        """
        # 输入块：提取低级特征
        x = self.conv1(inputs)
        x = self.bn1(x)  # 添加批量归一化
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)  # 添加批量归一化
        x = self.relu2(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # 中间块1：提取中级特征
        x = self.conv3(x)
        x = self.bn3(x)  # 添加批量归一化
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)  # 添加批量归一化
        x = self.relu4(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # 中间块2：提取高级特征
        x = self.conv5(x)
        x = self.bn5(x)  # 添加批量归一化
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.bn6(x)  # 添加批量归一化
        x = self.relu6(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # 展平并全连接
        x = x.reshape(x.shape[0], -1)  # 展平
        x = self.fc1(x)
        x = self.bn7(x)  # 添加批量归一化
        x = self.relu_fc1(x)
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        
        return x
    
    def train_step(self, inputs, targets):
        """
        模型训练方法
        
        执行一步训练，包括前向传播、损失计算、反向传播和参数更新。
        
        参数:
            inputs (Tensor): 输入张量，形状为(batch_size, 3, 32, 32)
                           表示一批CIFAR-10彩色图像数据
            targets (Tensor): 目标张量，形状为(batch_size,)
                           表示每个样本的类别索引，取值范围为[0, 9]
        
        返回:
            Tensor: 损失值，形状为(1,)
                   表示当前批次的平均损失
        """
        # 前向传播
        outputs = self.forward(inputs)
        
        # 计算损失
        loss = self.loss_func(outputs, targets)
        
        # 反向传播和优化
        self.optimizer.zero_grad(True)
        loss.backward()
        self.optimizer.step()
        
        return loss
    
    def evaluate(self, dataloader, cuda_available=False):
        """
        评估模型在数据集上的性能
        
        计算模型在给定数据加载器上的准确率和平均损失。
        
        参数:
            dataloader (DataLoader): 数据加载器，提供批次数据
            cuda_available (bool): 是否使用CUDA设备
        
        返回:
            tuple: (accuracy, loss)
                accuracy (float): 分类准确率，取值范围为[0, 1]
                loss (float): 平均损失值
        """
        correct = 0
        total = 0
        total_loss = 0.0
        
        # 不计算梯度，节省内存和计算资源
        for batch in dataloader:
            img_tensors, target_tensors = batch
            
            # 将批量数据移动到GPU
            if cuda_available:
                img_tensors = img_tensors.to('cuda')
                target_tensors = target_tensors.to('cuda')
            
            # 前向传播
            outputs = self.forward(img_tensors)
            
            # 计算损失
            loss = self.loss_func(outputs, target_tensors)
            total_loss += loss.item()
            
            # 计算准确率
            predicted = outputs.argmax(dim=1)
            total += target_tensors.size(0)
            
            # 确保predicted和target_tensors在同一设备上
            if cuda_available:
                # 如果在GPU上，确保两者都是Cupy数组
                correct += (predicted == target_tensors).sum().item()
            else:
                # 如果在CPU上，确保两者都是NumPy数组
                correct += (predicted.data == target_tensors.data).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        
        return accuracy, avg_loss


def main():
    """
    主函数
    
    执行完整的CIFAR-10图像识别流程：
    1. 加载预处理的数据集
    2. 创建数据加载器
    3. 初始化高级卷积神经网络模型
    4. 训练模型几个epoch
    5. 评估模型性能
    6. 输出训练时间和最终准确率
    """
    # 清屏
    clear_screen()
    print("CIFAR-10图像识别高级卷积神经网络示例（基于数据集特点优化）")
    
    # 检查CUDA可用性
    CUDA_AVAILABLE = cuda.CUPY_AVAILABLE
    device = 'cuda' if CUDA_AVAILABLE else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载数据集
    print("加载数据集...")
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    
    # 创建预处理的数据集
    train_dataset = EasyCIFAR10(root=data_path, train=True)
    test_dataset = EasyCIFAR10(root=data_path, train=False)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"类别: {train_dataset.classes}")
    
    # 创建数据加载器
    batch_size = 128  # 调整批次大小，平衡内存使用和训练稳定性
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    print("\n初始化模型...")
    model = AdvancedConvNet()
    
    # 将模型移动到指定设备
    if CUDA_AVAILABLE:
        model.to(device)
        # 重新初始化优化器，确保它引用GPU上的参数
        model.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    
    # 训练参数
    num_epochs = 5  # 增加训练轮数，确保模型充分收敛
    best_accuracy = 0.0
    
    print("\n开始训练...")
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # 训练阶段
        model.train()  # 设置为训练模式
        total_loss = 0.0
        batch_count = 0
        
        # 使用tqdm显示进度，不添加额外的打印语句
        with tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch") as pbar:
            for batch_idx, batch in enumerate(pbar):
                img_tensors, target_tensors = batch
                
                # 将批量数据移动到GPU
                if CUDA_AVAILABLE:
                    img_tensors = img_tensors.to(device)
                    target_tensors = target_tensors.to(device)
                
                # 执行一步训练
                loss = model.train_step(img_tensors, target_tensors)
                total_loss += loss.item()
                batch_count += 1
                
                # 更新进度条显示当前损失
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{total_loss/batch_count:.4f}"})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{num_epochs} 完成, 平均损失: {avg_loss:.4f}")
        
        # 评估阶段
        model.eval()  # 设置为评估模式
        test_accuracy, test_loss = model.evaluate(test_loader, CUDA_AVAILABLE)
        print(f"测试集准确率: {test_accuracy:.4f}, 测试损失: {test_loss:.4f}")
        
        # 保存最佳模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            print(f"新的最佳准确率: {best_accuracy:.4f}")
        
        print("-" * 50)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"训练总时间: {training_time:.2f}秒")
    print(f"最佳测试准确率: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()