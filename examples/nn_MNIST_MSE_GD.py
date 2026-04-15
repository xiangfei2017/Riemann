"""
使用均方误差损失和梯度下降优化器在MNIST数据集上进行手写数字识别

本示例展示了如何使用Riemann深度学习框架构建一个简单的神经网络分类器，
用于MNIST手写数字识别任务。该示例使用均方误差损失函数和标准梯度下降(GD)优化器。

主要功能：
1. 构建一个包含两个隐藏层的全连接神经网络
2. 使用ReLU作为隐藏层激活函数，Sigmoid作为输出层激活函数
3. 使用均方误差(MSE)作为损失函数
4. 使用标准梯度下降(GD)优化器进行参数更新
5. 在MNIST训练集上进行训练，并在测试集上评估模型性能
6. 展示完整的训练流程：数据加载、模型训练、性能评估

网络结构：
- 输入层: 784个神经元 (28x28像素的图像展平)
- 隐藏层1: 200个神经元，使用ReLU激活函数
- 隐藏层2: 200个神经元，使用ReLU激活函数
- 输出层: 10个神经元，对应10个数字类别，使用Sigmoid激活函数

超参数：
- 学习率: 0.01
- 训练轮数: 1
- 优化器: 梯度下降(GD)
- 损失函数: 均方误差(MSE)
"""

import sys,os,time
from tqdm import tqdm

# 添加项目根目录到Python路径，确保可以导入riemann模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src'))
sys.path.append(project_root)

# 导入riemann包中的模块
import riemann.nn as nn
import riemann.optim as opt
from riemann.vision.datasets import *

class Classifier(nn.Module):
    """
    简单的神经网络分类器，用于MNIST手写数字识别。
    
    网络结构:
    - 输入层: 784个神经元 (28x28像素的图像展平)
    - 隐藏层1: 200个神经元，使用ReLU激活函数
    - 隐藏层2: 200个神经元，使用ReLU激活函数
    - 输出层: 10个神经元，对应10个数字类别，使用Sigmoid激活函数
    
    损失函数: 均方误差损失 (MSELoss)
    优化器: 梯度下降 (GD)，学习率0.01
    """
    
    def __init__(self):
        """
        初始化神经网络分类器。
        """
        super().__init__()
        
        # 定义网络结构
        self.model = nn.Sequential(
            nn.Linear(784,200),
            nn.ReLU(),
            nn.Linear(200,10),
            nn.Sigmoid()
        )

        self.loss_func = nn.MSELoss()
        self.optimizer = opt.GD(self.parameters(), lr=0.01)        

    def forward(self,inputs):
        return self.model(inputs)
    
    def train_step(self,inputs,targets): 
        outputs = self.forward(inputs)
        loss = self.loss_func(outputs,targets)
        self.optimizer.zero_grad(True)
        loss.backward()
        self.optimizer.step()

def clear_screen():
    """跨平台清屏函数（自动识别系统）"""
    if os.name == 'nt':
        os.system('cls')  # Windows系统指令
    else:
        os.system('clear')  # Linux/MacOS指令
    return

clear_screen()

print("loading training data...")
starttime = time.time()
# 设置数据集路径
data_root = os.path.join(os.path.dirname(__file__), '..','data')

# 使用继承自MNIST的预处理版本，复用MNIST代码同时保持高效
minist_dataset = EasyMNIST(root=data_root, train=True,download=True)
endtime = time.time()
print(f"loading seconds:{endtime-starttime:.2f}")
print(f"dataset size: {len(minist_dataset)}")

C = Classifier()

print("starting training...")
starttime = time.time()
epochs = 1
for epoch in range(epochs):
    for img_tensor,target_tensor in tqdm(minist_dataset,desc=f'Epoch {epoch+1} train'):
        C.train_step(img_tensor,target_tensor)
        pass
    pass

endtime = time.time()
print(f"train seconds:{endtime-starttime:.2f}")

# test the neural network
minist_test_dataset = EasyMNIST(root=data_root, train=False)
print("start testing...")
starttime = time.time()

scores = 0
test_samples = len(minist_test_dataset)
for img_tensor,target_tensor in tqdm(minist_test_dataset):
    output = C.forward(img_tensor)
    if output.argmax() == target_tensor.argmax():
        scores += 1

endtime = time.time()
perf = scores / test_samples
print(f"test seconds:{endtime-starttime:.2f}")
print(f"accuracy: {perf:.4f}")