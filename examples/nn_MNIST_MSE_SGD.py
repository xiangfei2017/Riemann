"""
使用均方误差损失和随机梯度下降优化器在MNIST数据集上进行手写数字识别

本示例展示了如何使用Riemann深度学习框架构建一个简单的神经网络分类器，
用于MNIST手写数字识别任务。该示例使用均方误差损失函数和随机梯度下降(SGD)优化器，
并引入了动量(momentum)来加速训练过程。

主要功能：
1. 构建一个包含两个隐藏层的全连接神经网络
2. 使用ReLU作为隐藏层激活函数，Sigmoid作为输出层激活函数
3. 使用均方误差(MSE)作为损失函数
4. 使用随机梯度下降(SGD)优化器进行参数更新，并添加动量加速收敛
5. 使用DataLoader进行批量数据加载和打乱
6. 在MNIST训练集上进行训练，并在测试集上评估模型性能
7. 展示完整的训练流程：数据加载、模型训练、性能评估

网络结构：
- 输入层: 784个神经元 (28x28像素的图像展平)
- 隐藏层1: 200个神经元，使用ReLU激活函数
- 隐藏层2: 200个神经元，使用ReLU激活函数
- 输出层: 10个神经元，对应10个数字类别，使用Sigmoid激活函数

超参数：
- 学习率: 0.01
- 动量: 0.9
- 训练轮数: 1
- 批量大小: 1
- 优化器: 随机梯度下降(SGD) with momentum
- 损失函数: 均方误差(MSE)

与nn_MNIST_MSE_GD.py的区别：
1. 使用SGD优化器而非GD优化器，添加了动量参数
2. 使用DataLoader进行数据加载，支持批量处理和数据打乱
3. 主程序代码放在if __name__ == '__main__'条件内，提高代码可重用性
"""

import sys,os,time
from tqdm import tqdm

# 添加项目根目录到Python路径，确保可以导入riemann模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src'))
sys.path.append(project_root)

# 导入riemann包中的模块
import riemann.nn as nn
import riemann.optim as opt
from riemann.vision.datasets import EasyMNIST
from riemann.utils.data import DataLoader

# 定义 Classifier 类
class Classifier(nn.Module):
    # 类定义保持不变
    def __init__(self):
        super().__init__()
    
        self.model = nn.Sequential(
            nn.Linear(784,200),
            nn.ReLU(),
            nn.Linear(200,10),
            nn.Sigmoid()
        )

        self.loss_func = nn.MSELoss()
        # 随机梯度算法优化器
        self.optimizer = opt.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def forward(self,inputs):
        # 输入已经在数据加载时reshape为(1, 784)，直接使用
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

# 将主程序代码放在 if __name__ == '__main__': 条件内
if __name__ == '__main__':
    clear_screen()
    # 加载数据
    print("loading training data...")
    starttime = time.time()
    # 设置数据集路径
    data_root = os.path.join(os.path.dirname(__file__), '..','data')
    
    minist_dataset = EasyMNIST(root=data_root, train=True,download=True)
    endtime = time.time()
    print(f"loading seconds:{endtime-starttime:.2f}")

    # 创建模型和数据加载器
    C = Classifier()
    dataloader = DataLoader(minist_dataset, batch_size=1, shuffle=True, num_workers=0)

    # 训练循环
    starttime = time.time()
    epochs = 1
    for epoch in range(epochs):
        for batch in tqdm(dataloader,desc=f'Epoch {epoch+1} train'):
            img_tensors,target_tensors = batch
            C.train_step(img_tensors,target_tensors)
            pass
        pass

    endtime = time.time()
    print(f"train seconds:{endtime-starttime:.2f}")

    print("\nloading test data...")
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
    print (f"performance = {perf:.2f}")
        
