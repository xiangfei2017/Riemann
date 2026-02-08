import riemann as rm
import numpy as np

# 1. 定义一个简单的神经网络模型
class SimpleNet(rm.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = rm.nn.Linear(10, 100)
        self.fc2 = rm.nn.Linear(100, 100)
        self.fc3 = rm.nn.Linear(100, 1)
    
    def forward(self, x):
        x = rm.nn.functional.relu(self.fc1(x))
        x = rm.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 2. 生成模拟数据
def generate_data(num_samples=1000):
    # 生成随机输入
    X = np.random.randn(num_samples, 10).astype(np.float32)
    # 生成标签 (简单的线性关系 + 噪声)
    y = np.sum(X[:, :3], axis=1, keepdims=True) + np.random.randn(num_samples, 1).astype(np.float32) * 0.1
    return X, y

# 3. 训练函数 (带梯度修剪)
def train_with_gradient_clipping():
    print("=== 训练神经网络 (带梯度修剪) ===")
    
    # 创建模型、损失函数和优化器
    model = SimpleNet()
    criterion = rm.nn.MSELoss()
    optimizer = rm.optim.SGD(model.parameters(), lr=0.01)
    
    # 生成数据
    X, y = generate_data()
    
    # 训练参数
    num_epochs = 10
    batch_size = 32
    clip_value = 1.0  # 梯度修剪阈值
    
    # 记录梯度范数
    grad_norms = []
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        # 随机打乱数据
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, len(X), batch_size):
            # 获取批次数据
            batch_end = min(i + batch_size, len(X))
            batch_X = X_shuffled[i:batch_end]
            batch_y = y_shuffled[i:batch_end]
            
            # 转换为Riemann张量
            inputs = rm.tensor(batch_X)
            targets = rm.tensor(batch_y)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 应用梯度修剪 (按范数裁剪)
            rm.utils.data.clip_grad_norm_(model.parameters(), clip_value)
            
            # 计算并记录梯度范数 (修剪后)
            grad_norm_after = rm.utils.data.clip_grad_norm_(model.parameters(), float('inf'))
            grad_norms.append(grad_norm_after)
            
            # 更新参数
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(X) // batch_size)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    return grad_norms

# 4. 训练函数 (无梯度修剪)
def train_without_gradient_clipping():
    print("\n=== 训练神经网络 (无梯度修剪) ===")
    
    # 创建模型、损失函数和优化器
    model = SimpleNet()
    criterion = rm.nn.MSELoss()
    optimizer = rm.optim.SGD(model.parameters(), lr=0.01)
    
    # 生成数据
    X, y = generate_data()
    
    # 训练参数
    num_epochs = 10
    batch_size = 32
    
    # 记录梯度范数
    grad_norms = []
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        # 随机打乱数据
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, len(X), batch_size):
            # 获取批次数据
            batch_end = min(i + batch_size, len(X))
            batch_X = X_shuffled[i:batch_end]
            batch_y = y_shuffled[i:batch_end]
            
            # 转换为Riemann张量
            inputs = rm.tensor(batch_X)
            targets = rm.tensor(batch_y)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 计算并记录梯度范数
            grad_norm = rm.utils.data.clip_grad_norm_(model.parameters(), float('inf'))
            grad_norms.append(grad_norm)
            
            # 直接更新参数 (无梯度修剪)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(X) // batch_size)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    return grad_norms

# 5. 分析梯度范数
def analyze_grad_norms(with_clipping, without_clipping):
    print("\n=== 梯度范数分析 ===")
    
    # 计算统计信息
    max_with_clipping = max(with_clipping)
    mean_with_clipping = np.mean(with_clipping)
    
    max_without_clipping = max(without_clipping)
    mean_without_clipping = np.mean(without_clipping)
    
    print(f"带梯度修剪:")
    print(f"  最大梯度范数: {max_with_clipping:.4f}")
    print(f"  平均梯度范数: {mean_with_clipping:.4f}")
    
    print(f"\n无梯度修剪:")
    print(f"  最大梯度范数: {max_without_clipping:.4f}")
    print(f"  平均梯度范数: {mean_without_clipping:.4f}")
    
    # 检查梯度修剪效果
    if max_with_clipping <= 1.0 + 1e-6:
        print("\n✅ 梯度修剪成功: 所有梯度范数都被限制在阈值以下")
    else:
        print("\n❌ 梯度修剪失败: 存在超过阈值的梯度范数")

# 6. 主函数
if __name__ == "__main__":
    # 训练带梯度修剪的模型
    grad_norms_with_clipping = train_with_gradient_clipping()
    
    # 训练无梯度修剪的模型
    grad_norms_without_clipping = train_without_gradient_clipping()
    
    # 分析梯度范数
    analyze_grad_norms(grad_norms_with_clipping, grad_norms_without_clipping)
    
    print("\n训练完成!")
