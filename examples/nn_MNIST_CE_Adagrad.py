"""MNIST手写数字识别神经网络示例（Adagrad优化器版）

本文件实现了一个基于Riemann框架的简单神经网络，用于识别MNIST手写数字数据集。
使用交叉熵损失函数和Adagrad优化算法，相比SGD通常收敛更快。
包含数据集加载、模型定义、训练和测试功能，支持批量数据加载和处理。

主要功能：
1. 使用riemann.vision.datasets.MNIST加载标准MNIST数据集
2. 构建简单的全连接神经网络模型
3. 实现训练循环和测试评估
4. 支持批量数据加载和处理
5. 使用Adagrad优化器进行参数更新

依赖：
- riemann：自定义深度学习框架
- riemann.vision：计算机视觉模块，包含MNIST数据集和变换
- tqdm：进度条显示

用法：
    python nn_CrossEntropy_Adagrad.py
"""
import sys,os,time
from tqdm import tqdm

# 添加项目根目录到Python路径，确保可以导入riemann模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src'))
sys.path.append(project_root)

# 导入riemann包中的模块
import riemann.nn as nn
import riemann.optim as opt
from riemann.utils.data import DataLoader
from riemann.vision.datasets import MNIST
from riemann.vision import transforms
from riemann import cuda

class Classifier(nn.Module):
    """
    MNIST手写数字分类器（Adagrad优化器版）
    
    这是一个简单的前馈神经网络，用于识别MNIST手写数字数据集中的图像。
    网络结构包含一个隐藏层，使用ReLU激活函数和交叉熵损失函数。
    使用Adagrad优化器进行参数更新，相比SGD通常收敛更快。
    
    网络架构:
    - 输入层: 784个神经元 (28×28像素的MNIST图像展平后)
    - 隐藏层: 200个神经元，使用ReLU激活函数
    - 输出层: 10个神经元，对应10个数字类别(0-9)
    
    属性:
        model (nn.Sequential): 包含网络各层的序列容器
        loss_func (nn.CrossEntropyLoss): 交叉熵损失函数
        optimizer (opt.Adagrad): Adagrad优化器
    
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
        使用Adagrad优化器进行参数更新，相比SGD通常收敛更快。
        """
        super().__init__()
    
        # 定义神经网络结构：输入层(784维)→隐藏层(200个神经元)→输出层(10个神经元)
        self.model = nn.Sequential(
            nn.Flatten(),        # 展平操作，将输入从(1, 28, 28)变为(1, 784)
            nn.Linear(784, 200),  # 输入层到隐藏层
            nn.ReLU(),           # ReLU激活函数
            nn.Linear(200, 10)   # 隐藏层到输出层
        )

        # 交叉熵损失函数，适用于多分类任务
        self.loss_func = nn.CrossEntropyLoss()
        # 优化器将在设备移动后初始化
    
    def forward(self, inputs):
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
        使用交叉熵损失函数和Adagrad优化器进行参数更新。
        
        参数:
            inputs (Tensor): 输入张量，形状为(batch_size, 1, 28, 28)
                           表示一批MNIST图像数据
            targets (Tensor): 目标张量，形状为(batch_size,)
                             表示每个样本的真实类别标签(0-9的整数)
        
        返回:
            Tensor: 标量损失值，表示当前批次的平均损失
        """
        outputs = self.forward(inputs)
        loss = self.loss_func(outputs, targets)
        self.optimizer.zero_grad(True)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def evaluate(self, dataloader,device):
        """
        模型评估方法
        
        在给定的数据集上评估模型的性能，计算平均损失和准确率。
        该方法不会更新模型参数，仅用于性能评估。
        
        参数:
            dataloader (DataLoader): 数据加载器，提供批量的测试数据
            device (str): 模型运行设备对象
        
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
            
            # 将批量数据移动到GPU
            img_tensors = img_tensors.to(device)
            target_tensors = target_tensors.to(device)
            
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
    print("MNIST手写数字识别神经网络示例（Adagrad优化器版）")
    
    # 检查CUDA可用性
    CUDA_AVAILABLE = cuda.CUPY_AVAILABLE
    device = 'cuda' if CUDA_AVAILABLE else 'cpu'
    print(f"使用设备: {device}")
    
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
        transform=transform,
        download=True  # 下载数据集到本地
    )
    test_dataset = MNIST(
        root=data_root,
        train=False,
        transform=transform,
        download=True  # 下载数据集到本地
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=100,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=100,
        shuffle=False
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 创建模型
    print("\n初始化模型...")
    model = Classifier()
    
    # 将模型移动到指定设备
    model.to(device)
    
    # 初始化优化器，确保它引用正确设备上的参数
    model.optimizer = opt.Adagrad(model.parameters(), 
                                lr=0.01,       # 学习率，Adagrad通常使用比SGD更高的学习率
                                weight_decay=0.0001,  # 权重衰减，用于L2正则化
                                eps=1e-10,      # 数值稳定性参数，防止除零错误
                                initial_accumulator_value=0.0)  # 累加器初始值
    
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
        print('-' * 50)
    
    train_end_time = time.time()
    print(f"训练总时间: {train_end_time - train_start_time:.2f}秒")

# 将主程序代码放在 if __name__ == '__main__': 条件内
if __name__ == '__main__':
    main()