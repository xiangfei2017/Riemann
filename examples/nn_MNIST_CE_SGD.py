#!/usr/bin/env python3
"""
MNIST手写数字识别神经网络示例（改进版）

本文件实现了一个基于Riemann框架的简单神经网络，用于识别MNIST手写数字数据集。
使用更新后的MNIST类和transforms模块，采用与nn_CrossEntropy_SGD.py相同的网络结构和优化器参数。

网络架构:
- 输入层: 784个神经元 (28×28像素的MNIST图像展平后)
- 隐藏层: 200个神经元，使用ReLU激活函数
- 输出层: 10个神经元，对应10个数字类别(0-9)

损失函数: 交叉熵损失函数 (CrossEntropyLoss)
优化器: 随机梯度下降 (SGD)，学习率0.001，动量0.9

数据预处理:
- transforms.ToTensor(): 将PIL图像转换为张量并归一化到[0,1]范围
- transforms.Normalize(): 使用MNIST数据集的均值(0.1307)和标准差(0.3081)进行标准化

特点:
- 使用DataLoader进行批量数据加载，提高训练效率
- 在模型中使用nn.Flatten()进行输入展平，而不是在forward方法中
- 支持GPU加速计算（如果可用）
- 提供详细的训练进度和性能指标

作者: Riemann框架示例
日期: 2025
"""
import sys,os,time
from tqdm import tqdm

# 添加项目根目录到Python路径，确保可以导入riemann模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src'))
sys.path.append(project_root)

# 导入riemann包中的模块
import riemann.nn as nn
import riemann.optim as opt
from riemann.vision.datasets import MNIST
from riemann.vision import transforms
from riemann.utils.data import DataLoader

class Classifier(nn.Module):
    """
    MNIST手写数字分类器
    
    这是一个简单的前馈神经网络，用于识别MNIST手写数字数据集中的图像。
    网络结构包含一个隐藏层，使用ReLU激活函数和交叉熵损失函数。
    
    网络架构:
    - 输入层: 784个神经元 (28×28像素的MNIST图像展平后)
    - 隐藏层: 200个神经元，使用ReLU激活函数
    - 输出层: 10个神经元，对应10个数字类别(0-9)
    
    属性:
        model (nn.Sequential): 包含网络各层的序列容器
        loss_func (nn.CrossEntropyLoss): 交叉熵损失函数
        optimizer (opt.SGD): 随机梯度下降优化器
    
    方法:
        forward(inputs): 前向传播计算
        train_step(inputs, targets): 执行一步训练
        evaluate(dataloader): 评估模型在数据集上的性能
    """
    def __init__(self):
        """
        初始化分类器模型
        
        创建网络各层、损失函数和优化器。
        网络结构使用nn.Sequential容器，包含展平层、线性层和激活函数。
        """
        super().__init__()
    
        # 定义神经网络结构：输入层(784维)→隐藏层(200个神经元)→输出层(10个神经元)
        self.model = nn.Sequential(
            nn.Flatten(),        # 展平操作，将输入从(1, 28, 28)变为(1, 784)
            nn.Linear(784, 200),  # 输入层到隐藏层
            nn.ReLU(),           # ReLU激活函数，与nn_CrossEntropy_SGD.py相同
            nn.Linear(200, 10)   # 隐藏层到输出层
        )

        # 交叉熵损失函数，适用于多分类任务
        self.loss_func = nn.CrossEntropyLoss()
        # 随机梯度下降优化器，学习率0.001，momentum=0.9，与nn_CrossEntropy_SGD.py相同
        self.optimizer = opt.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def forward(self,inputs):
        """
        前向传播计算
        
        将输入数据通过网络模型，得到预测输出。
        输入数据会先经过展平层，然后通过两个线性层和激活函数。
        
        参数:
            inputs (Tensor): 输入张量，形状为(batch_size, 1, 28, 28)
                           表示一批MNIST图像数据
        
        返回:
            Tensor: 输出张量，形状为(batch_size, 10)
                   表示每个样本属于10个类别的未归一化对数概率
        """
        return self.model(inputs)
    
    def train_step(self, inputs, targets): 
        """
        模型训练方法
        
        执行一步训练，包括前向传播、损失计算、反向传播和参数更新。
        使用交叉熵损失函数和SGD优化器进行参数更新。
        
        参数:
            inputs (Tensor): 输入张量，形状为(batch_size, 1, 28, 28)
                           表示一批MNIST图像数据
            targets (Tensor): 目标张量，形状为(batch_size,)
                             表示每个样本的真实类别标签(0-9的整数)
        
        返回:
            Tensor: 标量损失值，表示当前批次的平均损失
        """
        outputs = self.forward(inputs)
        # targets已经是整数标签，无需转换
        loss = self.loss_func(outputs, targets)
        self.optimizer.zero_grad(True)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def evaluate(self, dataloader):
        """
        模型评估方法
        
        在给定的数据集上评估模型的性能，计算平均损失和准确率。
        该方法不会更新模型参数，仅用于性能评估。
        
        参数:
            dataloader (DataLoader): 数据加载器，提供批量的测试数据
        
        返回:
            tuple: 包含两个元素的元组
                - accuracy (float): 模型在数据集上的准确率(0-1之间)
                - avg_loss (float): 模型在数据集上的平均损失
        """
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            img_tensors, target_tensors = batch
            outputs = self.forward(img_tensors)
            
            # 计算损失
            loss = self.loss_func(outputs, target_tensors)
            total_loss += loss.item()
            
            # 计算准确率
            predicted = outputs.argmax(dim=1)
            total += target_tensors.size(0)
            correct += (predicted == target_tensors).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        return accuracy, avg_loss

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
    2. 加载训练和测试数据集
    3. 创建数据加载器
    4. 初始化神经网络模型
    5. 训练模型多个epoch
    6. 评估模型性能
    7. 输出训练时间和最终准确率
    """
    clear_screen()
    print("MNIST手写数字识别神经网络示例（SGD优化器版）")
    
    # 定义数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化参数
    ])
    
    data_root = os.path.join(os.path.dirname(__file__), '..','data')
    # 加载训练和测试数据集
    print("加载数据集...")
    train_dataset = MNIST(
        root=data_root,
        train=True,
        transform=transform
    )
    test_dataset = MNIST(
        root=data_root,
        train=False,
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=100,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 创建模型
    print("\n初始化模型...")
    model = Classifier()
    
    # 训练模型
    print("\n开始训练...")
    epochs = 3
    train_start_time = time.time()
    
    for epoch in range(epochs):
        model.train()  # 设置为训练模式
        epoch_loss = 0.0
        num_batches = len(train_loader)
        
        # 训练一个epoch，使用单一进度条
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            img_tensors, target_tensors = batch
            loss = model.train_step(img_tensors, target_tensors)
            epoch_loss += loss.item()
            
            # 更新进度条显示当前损失
            current_loss = loss.item()
            progress_bar.set_postfix({'Loss': f'{current_loss:.4f}'})
        
        # 计算平均损失
        avg_loss = epoch_loss / num_batches
        print(f'Epoch {epoch+1}/{epochs} 完成, 平均损失: {avg_loss:.4f}')
        
        # 在测试集上评估
        model.eval()  # 设置为评估模式
        test_accuracy, test_loss = model.evaluate(test_loader)
        print(f'测试集准确率: {test_accuracy:.4f}, 测试损失: {test_loss:.4f}')
        print('-' * 50)
    
    train_end_time = time.time()
    print(f"训练总时间: {train_end_time - train_start_time:.2f}秒")

if __name__ == "__main__":
    # 当脚本直接运行时执行主函数
    main()