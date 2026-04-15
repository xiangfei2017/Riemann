"""
CIFAR-10图像识别卷积神经网络示例（简化LeNet版）

本示例展示了如何使用Riemann深度学习框架构建一个简化版LeNet卷积神经网络，
用于CIFAR-10图像分类任务。

主要组件：
    - LeNet: 简化版LeNet卷积神经网络模型
        * 2个卷积层 + 1个全连接层
        * 针对32x32输入优化
    - EasyCIFAR10: 预处理的CIFAR10数据集
    - 数据预处理: 归一化和标准化

网络结构:
     Conv(3->32) -> ReLU -> MaxPool  # 32x32 -> 16x16
     Conv(32->64) -> ReLU -> MaxPool # 16x16 -> 8x8
     Flatten
     FC(4096->10)  # 直接输出到10个类别

使用方法:
    运行本脚本将自动加载CIFAR-10数据集，
    训练模型并在测试集上评估性能。
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
    继承自CIFAR10的子类，在初始化时对图像数据应用归一化、标准化转换，
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
            # 使用CIFAR10数据集的实际均值和标准差进行标准化
            # 根据数据分析：R(134.57,69.84), G(135.19,69.56), B(136.95,66.95)
            # 转换为0-1范围的均值和标准差
            normalized_img = transforms.Normalize(
                mean=(134.57/255, 135.19/255, 136.95/255), 
                std=(69.84/255, 69.56/255, 66.95/255)
            )(tensor_img)
            return normalized_img
        
        def tensor_label_transform(label):
            return tensor(label, dtype=get_default_dtype())
        
        # 初始化父类，不传入转换函数，直接获取PIL Image
        super().__init__(root, train=train, transform=None, target_transform=None)
        
        # 预处理所有数据，直接更新父类的data_list
        print("Transforming CIFAR-10 to EasyCIFAR10 ...")
        for i in tqdm(range(len(self.data_list))):
            img, target = self.data_list[i]  # 获取PIL Image和标签
            # 应用转换并更新
            self.data_list[i] = (tensor_transform(img), tensor_label_transform(target))
    
    # __len__和__getitem__直接继承自父类CIFAR10，无需重写


class LeNet(nn.Module):
    """
    简化版 LeNet 卷积神经网络用于 CIFAR-10 图像分类
    
    网络结构（针对 32x32 输入优化）:
        - 卷积层1: Conv(3->32, 3x3) -> ReLU -> MaxPool(2x2)  # 32x32 -> 16x16
        - 卷积层2: Conv(32->64, 3x3) -> ReLU -> MaxPool(2x2) # 16x16 -> 8x8
        - 展平
        - 全连接层: Linear(64*8*8 -> 10)  # 直接输出到10个类别
    
    方法:
        forward(inputs): 前向传播计算
        train_step(inputs, targets): 执行一步训练
        evaluate(dataloader): 评估模型在数据集上的性能
    """
    def __init__(self):
        """
        初始化 LeNet 模型
        
        使用 Sequential 容器构建网络，代码更简洁。
        """
        super().__init__()
        
        # 特征提取部分：2个卷积层，使用 Sequential 容器
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 分类器部分：1个全连接层，直接输出到10个类别
        self.classifier = nn.Linear(64 * 8 * 8, 10)

        # 交叉熵损失函数，适用于多分类任务
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, inputs):
        """
        前向传播计算
        
        参数:
            inputs (Tensor): 输入张量，形状为(batch_size, 3, 32, 32)
        
        返回:
            Tensor: 输出张量，形状为(batch_size, 10)
        """
        # 特征提取: (batch_size, 3, 32, 32) -> (batch_size, 64, 8, 8)
        x = self.features(inputs)
        
        # 展平: (batch_size, 64, 8, 8) -> (batch_size, 64*8*8)
        x = x.reshape(x.shape[0], -1)
        
        # 分类: (batch_size, 64*8*8) -> (batch_size, 10)
        x = self.classifier(x)
        return x
    
    def train_step(self, inputs, targets):
        """
        模型训练方法
        
        执行一步训练，包括前向传播、损失计算、反向传播和参数更新。
        使用交叉熵损失函数和SGD优化器进行参数更新。
        
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
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss
    
    def evaluate(self, dataloader, device):
        """
        评估模型在数据集上的性能
        
        计算模型在给定数据加载器上的准确率和平均损失。
        
        参数:
            dataloader (DataLoader): 数据加载器，提供批次数据
            cuda_available (bool): 是否在CUDA设备上运行
        
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
            img_tensors = img_tensors.to(device)
            target_tensors = target_tensors.to(device)
            
            # 前向传播
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


def main():
    """
    主函数
    
    执行完整的CIFAR-10图像识别流程：
    1. 加载预处理的数据集
    2. 创建数据加载器
    3. 初始化极简的卷积神经网络模型
    4. 训练模型几个epoch
    5. 评估模型性能
    6. 输出训练时间和最终准确率
    """
    clear_screen()
    print("CIFAR-10图像识别极简卷积神经网络示例（预处理版）")
    
    # 检查CUDA可用性
    CUDA_AVAILABLE = cuda.CUPY_AVAILABLE
    device = 'cuda' if CUDA_AVAILABLE else 'cpu'
    print(f"使用设备: {device}")
    
    data_root = os.path.join(os.path.dirname(__file__), '..','data')
    # 加载训练和测试数据集
    print("加载数据集...")
    train_dataset = EasyCIFAR10(
        root=data_root,
        train=True
    )
    test_dataset = EasyCIFAR10(
        root=data_root,
        train=False
    )
    
    # 创建数据加载器，使用极小的批次大小
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=256,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=16,
        shuffle=False
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"类别: {train_dataset.classes}")
    
    # 创建模型
    print("\n初始化模型...")
    model = LeNet()
    
    # 将模型移动到指定设备
    model.to(device)
    
    # 初始化优化器，确保它引用正确设备上的参数
    model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 训练模型
    print("\n开始训练...")
    epochs = 5  # 训练epoch数
    train_start_time = time.time()
    
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        model.train()  # 设置为训练模式
        epoch_loss = 0.0
        num_batches = len(train_loader)
        
        # 训练一个epoch，使用单一进度条
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            img_tensors, target_tensors = batch
            
            # 将批量数据移动到GPU
            img_tensors = img_tensors.to(device)
            target_tensors = target_tensors.to(device)
            
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
        test_accuracy, test_loss = model.evaluate(test_loader, device)
        print(f'测试集准确率: {test_accuracy:.4f}, 测试损失: {test_loss:.4f}')
        
        # 保存最佳模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            print(f'新的最佳准确率: {best_accuracy:.4f}')
        
        print('-' * 50)
    
    train_end_time = time.time()
    print(f"训练总时间: {train_end_time - train_start_time:.2f}秒")
    print(f"最佳测试准确率: {best_accuracy:.4f}")

if __name__ == "__main__":
    # 当脚本直接运行时执行主函数
    main()