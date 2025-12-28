import sys,os,time
from tqdm import tqdm

# 添加项目根目录到Python路径，确保可以导入riemann模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src'))
sys.path.append(project_root)

# 导入riemann包中的模块
import riemann.nn as nn
import riemann.optim as opt
from riemann.tensordef import tensor
from riemann.vision.datasets import EasyMNIST
from riemann.utils.data import DataLoader
import numpy as np

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
    
    minist_dataset = EasyMNIST(root=data_root, train=True)
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
        
