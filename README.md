# Riemann

Riemann是一个轻量级的自动求导库及神经网络编程框架，支持标量/向量/张量的自动梯度跟踪，提供搭建神经网络所需的常用组件，接口灵活易扩展、兼容PyTorch，专为神经网络相关的学习、教育和研究目的而设计。

## 主要功能

- **自动求导**：支持实数和复数的标量、向量、张量的前向计算和反向自动求导
- **梯度计算**：支持反向传播算法计算梯度，提供`grad`和`backward`函数用于高效梯度计算，支持标量、向量、矩阵、多维张量计算中的反向梯度跟踪，支持雅可比矩阵(Jacobian)和海森矩阵(Hessian)计算，支持通过`track_grad`修饰器或`Function`类自定义梯度跟踪函数
- **张量操作**：提供丰富的张量操作功能，包括：加减乘除、初等函数、索引操作、形状操作、维度扩缩、堆叠/分割
- **神经网络组件**：包含基本的神经网络模块、激活函数、损失函数和优化器
- **计算机视觉支持**：提供常用的数据集类和图像变换功能，支持MNIST和CIFAR10等数据集加载和预处理

## PyTorch接口兼容性

Riemann库设计时注重与PyTorch接口的兼容性，同名的函数和类接口保持一致，方便PyTorch用户快速上手：

- **张量操作**：支持与PyTorch同名的张量操作函数和方法，如`tensor()`、`grad()`、`backward()`等
- **神经网络组件**：`nn`模块中的层、激活函数和损失函数与PyTorch保持接口兼容
- **优化器**：`optim`模块中的优化器（如SGD、Adam等）接口与PyTorch保持一致
- **自动微分机制**：`requires_grad`、反向传播机制与PyTorch相似
- **计算机视觉**：`vision`模块中的数据集和变换与torchvision保持接口兼容

这种设计使得熟悉PyTorch的用户可以轻松迁移到Riemann库进行开发和研究工作。

## 安装方法

### 从PyPI安装（待发布）

```bash
pip install riemann
```

### 源码安装与开发环境配置

```bash
# 从Gitee获取Riemann库源码
git clone https://gitee.com/xfcode2021/Riemann.git
cd Riemann
# 使用开发模式安装包及其核心依赖（-e表示可编辑模式，修改代码后无需重新安装）
pip install -e .

# 如果需要运行tests目录下的测试代码，还需安装测试依赖
pip install -e .[tests]
```

### 示例代码依赖

运行examples目录中的示例代码需要安装以下额外依赖：

```bash
pip install tqdm pillow
```

- **tqdm**: 用于神经网络训练示例中的进度条显示
- **pillow**: 用于计算机视觉示例中的图像处理功能（提供PIL模块）

numpy已作为核心依赖包含在包安装中，无需额外安装。

各个示例文件中的依赖可能略有不同，具体依赖信息也会在每个示例文件的头部注释中说明。

## 快速开始

```python
# 导入riemann库
from riemann import tensor
from riemann.autograd import grad

# 创建张量
t = tensor([1.0, 2.0, 3.0], requires_grad=True)

# 定义函数
def f(x):
    return (x ** 2.0).sum()

# 计算梯度
output = f(t)
grad_f = grad(output, t)[0]
print("梯度:", grad_f)  # 输出: 梯度: [2. 4. 6.]

# 反向传播示例
x = tensor([1.0, 2.0], requires_grad=True)
y = tensor([3.0, 4.0], requires_grad=True)
z = (x * y).sum()
z.backward()
print("x的梯度:", x.grad)  # 输出: x的梯度: [3. 4.]
print("y的梯度:", y.grad)  # 输出: y的梯度: [1. 2.]
```

## 核心功能介绍

### 1. 张量操作
- 提供张量创建函数（tensor, zeros, ones, random等，支持复数张量）
- 支持基本的数学运算（加减乘除幂运算，指数、对数、三角、双曲等初等函数）
- 支持向量、矩阵运算（批量矩阵乘法、向量点积、矩阵行列式、矩阵逆、矩阵分解等）
- 支持张量形状重塑、维度扩缩、索引和切片、元素收集/散射、拼接/分割等操作

### 2. 自动求导
- **backward方法**：触发反向传播计算梯度
- **grad函数**：计算函数相对于输入的梯度
- **track_track修饰器和Function类**：支持自定义梯度跟踪函数

### 3. 雅可比矩阵和海森矩阵
- 支持多输入多输出函数的雅可比矩阵计算
- 提供海森矩阵计算功能用于二阶导数
- 高效计算雅可比-向量乘积和向量-雅可比乘积
- 支持海森-向量乘积和向量-海森乘积计算

### 4. 线性代数模块
- 提供矩阵分解及其反向梯度跟踪（SVD、PLU, QR等）
- 支持求矩阵逆、广义逆、行列式、特征值/特征向量
- 矩阵范数、条件数计算
- 支持线性方程组求解、最小二乘求解

### 5. 神经网络模块
- 基本层（Linear, Flatten, Dropout, BatchNorm等）
- 激活函数（ReLU, Sigmoid,Softmax等）
- 卷积池化层（Conv1d/2d/3d, MaxPool1d/2d/3d, AvgPool1d/2d/3d等）
- 损失函数（MSE, CrossEntropy等）
- 优化器（SGD, Adam, Adagrad, LBFGS等）
- 网络模块容器（Sequential, ModuleList, ModuleDict等）

### 6. 计算机视觉模块
- 数据集类：MNIST、CIFAR10等常用数据集的加载和预处理
- 图像变换：Resize、Crop、Flip、Rotate、Normalize等图像预处理操作

## 项目结构

```
Riemann/
├── src/
│   └── riemann/              # 核心源代码
│       ├── autograd/         # 自动微分相关模块
│       ├── nn/               # 神经网络模块
│       ├── utils/            # 工具函数
│       ├── vision/           # 计算机视觉模块
│       ├── __init__.py       # 包配置文件
│       ├── dtype.py          # 数据类型定义
│       ├── gradmode.py       # 梯度模式控制
│       ├── linalg.py         # 线性代数函数
│       ├── optim.py          # 优化器
│       ├── serialization.py  # 对象保存与加载
│       └── tensordef.py      # 张量定义和核心操作
├── data/                     # 训练测试数据集文件目录
├── docs/                     # 项目文档目录
├── tests/                    # 测试文件
├── examples/                     # 示例代码
│   ├── backward_demo.py          # 反向传播示例
│   ├── grad_demo.py              # 梯度计算示例
│   ├── custom_grad_decorator.py  # 自定义梯度跟踪函数示例
│   ├── optimizers_comparison.py  # 优化器比较示例
│   ├── nn_MNIST_CE_SGD.py        # 神经网络训练示例
│   ├── cnn_CIFAR10_CE_SGD.py     # 卷积网路训练示例
│   └── ...
├── README.md                 # 项目文档
├── LICENSE                   # 许可证文件
└── pyproject.toml            # 项目配置和依赖管理
```

## 更多示例

### 雅可比矩阵计算

```python
from riemann import tensor
from riemann.autograd.functional import jacobian

# 定义函数
def f(x):
    return x ** 2.0

# 创建输入张量
x = tensor([1.0, 2.0, 3.0], requires_grad=True)

# 计算雅可比矩阵
jacob = jacobian(f, x)
print("雅可比矩阵:", jacob)
```

### 神经网络训练

```python
# 神经网络训练示例：训练一个网络用于求两数之和，这个网络模型本质上是一个线性回归
from riemann import tensor
from riemann.nn import Linear, MSELoss
from riemann.optim import SGD

# 创建模型
model = Linear(2, 1)
criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01)

# 训练数据
inputs = tensor([[1.0, 2.0], [3.0, 4.0]])
targets = tensor([[3.0], [7.0]])

# 训练循环
for epoch in range(100):
    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item()}')

# 预测
new_input = tensor([[5.0, 6.0]])
prediction = model(new_input)
print(f'预测结果: {prediction.item()}')
```

### 计算机视觉支持

Riemann提供了丰富的计算机视觉功能，包括常用的数据集类和图像变换功能：

### 支持的数据集

- **MNIST**：手写数字识别数据集
- **CIFAR10**：10类彩色图像数据集

### 支持的图像变换

- **基础变换**：ToTensor, ToPILImage, Normalize
- **几何变换**：Resize, CenterCrop, RandomResizedCrop, FiveCrop, TenCrop, Pad
- **随机变换**：RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomGrayscale
- **颜色变换**：ColorJitter, Grayscale
- **组合变换**：Compose, Lambda

### 计算机视觉数据集加载

```python
from riemann.vision.datasets import MNIST, CIFAR10
from riemann.vision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip

# 定义MNIST变换
mnist_transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))  # MNIST标准化参数
])

# 定义CIFAR10训练集变换（包含数据增强）
cifar10_train_transform = Compose([
    RandomHorizontalFlip(),  # 随机水平翻转
    ToTensor(),
    Normalize((0.5279, 0.5303, 0.5373), (0.2739, 0.2728, 0.2625))  # CIFAR10实际标准化参数
])

# 定义CIFAR10测试集变换（不包含数据增强）
cifar10_test_transform = Compose([
    ToTensor(),
    Normalize((0.5279, 0.5303, 0.5373), (0.2739, 0.2728, 0.2625))  # CIFAR10实际标准化参数
])

# 加载MNIST数据集
train_dataset = MNIST(root='data', train=True, transform=mnist_transform)
test_dataset = MNIST(root='data', train=False, transform=mnist_transform)

# 加载CIFAR10数据集
cifar10_train = CIFAR10(root='data', train=True, transform=cifar10_train_transform)
cifar10_test = CIFAR10(root='data', train=False, transform=cifar10_test_transform)

print(f"MNIST训练集大小: {len(train_dataset)}")
print(f"CIFAR10测试集大小: {len(cifar10_test)}")
```

### CNN示例

```python
import riemann as r
from riemann.vision.datasets import CIFAR10
from riemann.vision.transforms import *
from riemann.nn import *
from riemann.optim import SGD

# 定义CNN模型
class SimpleCNN(r.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 只使用1个卷积层
        self.conv1 = Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = Flatten()
        # 经过池化后，图像尺寸从32x32变为16x16，通道数为16
        self.fc1 = Linear(16 * 16 * 16, 10)  # 直接输出10个类别
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        return x

# 加载数据
# 训练集使用数据增强，测试集不使用
train_transform = Compose([
    RandomHorizontalFlip(),  # 随机水平翻转
    ToTensor(),
    Normalize((0.5279, 0.5303, 0.5373), (0.2739, 0.2728, 0.2625))  # CIFAR10实际标准化参数
])

test_transform = Compose([
    ToTensor(),
    Normalize((0.5279, 0.5303, 0.5373), (0.2739, 0.2728, 0.2625))  # CIFAR10实际标准化参数
])

train_dataset = CIFAR10(root='data', train=True, transform=train_transform)
test_dataset = CIFAR10(root='data', train=False, transform=test_transform)

# 减小批次大小和数据量以加快测试
train_loader = r.utils.DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = r.utils.DataLoader(test_dataset, batch_size=100, shuffle=False)

# 创建模型、损失函数和优化器
model = SimpleCNN()
criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.01)

# 训练循环
from tqdm import tqdm

for epoch in range(3):  # 训练3代
    total_loss = 0
    # 使用tqdm显示进度条
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/3")
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        # 前向传播
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 更新进度条显示当前损失
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss/len(train_loader)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

# 模型评估（推理测试）
model.eval()  # 设置为评估模式
correct = 0
total = 0

# 使用tqdm显示测试进度
test_progress_bar = tqdm(test_loader, desc="Testing")

with r.no_grad():  # 禁用梯度计算
    for data, target in test_progress_bar:
        # 前向传播
        outputs = model(data)
        
        # 获取预测结果
        _, predicted = r.max(outputs, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # 更新进度条显示当前准确率
        current_accuracy = 100 * correct / total
        test_progress_bar.set_postfix({"Accuracy": f"{current_accuracy:.2f}%"})

# 输出最终测试准确率
test_accuracy = 100 * correct / total
print(f"测试集准确率: {test_accuracy:.2f}% ({correct}/{total})")

# 单个样本推理示例
sample_data, sample_target = next(iter(test_loader))
sample_output = model(sample_data[:1])  # 只取第一个样本
_, predicted_class = r.max(sample_output, 1)
print(f"样本预测类别: {predicted_class.item()}, 实际类别: {sample_target[0].item()}")

print("CNN训练和推理测试完成！")
```

### 图像变换示例

```python
from riemann.vision.transforms import Compose, ToTensor, Normalize, Resize, CenterCrop, RandomHorizontalFlip
from PIL import Image

# 定义变换序列
transform = Compose([
    Resize(256),               # 调整图像大小
    CenterCrop(224),           # 中心裁剪
    RandomHorizontalFlip(),    # 随机水平翻转
    ToTensor(),                # 转换为张量
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 加载并变换图像
image = Image.open("example.jpg")
transformed_image = transform(image)
print(f"变换后的图像形状: {transformed_image.shape}")  # 输出: (3, 224, 224)
```

## 示例说明

Riemann提供了丰富的示例代码，位于`examples/`目录：

- **backward_demo.py**: backward函数使用演示
- **grad_demo.py**: grad函数使用演示
- **hessian.py**: 海森矩阵计算示例
- **jacobian.py**: 雅可比矩阵计算示例
- **nn_MNIST_CE_SGD.py**: 基于MNIST的手写数字识别神经网络示例（交叉熵损失 + SGD优化）
- **nn_MNIST_CE_Adam.py**: 基于MNIST的手写数字识别神经网络示例（交叉熵损失 + Adam优化）
- **nn_MNIST_CE_Adagrad.py**: 基于MNIST的手写数字识别神经网络示例（交叉熵损失 + Adagrad优化）
- **nn_MNIST_MSE_GD.py**: 基于MNIST的手写数字识别神经网络示例（均方误差 + 梯度下降）
- **nn_MNIST_MSE_SGD.py**: 基于MNIST的手写数字识别神经网络示例（均方误差 + SGD优化）
- **nn_MNIST_MSE_LBFGS.py**: 基于MNIST的手写数字识别神经网络示例（均方误差 + LBFGS优化）
- **cnn_CIFAR10_CE_SGD.py**: 基于CIFAR10的卷积神经网络示例（交叉熵损失 + SGD优化）
- **cnn_CIFAR10_CE_Adam.py**: 基于CIFAR10的卷积神经网络示例（交叉熵损失 + Adam优化）
- **optimizers_comparison.py**: 优化器性能比较
- **scatter.py**: 张量散射操作示例
- **custom_grad_decorator.py**: 使用@track_grad装饰器自定义梯度跟踪函数示例
- **custom_grad_FunctionClass.py**: 使用Function类自定义梯度跟踪函数示例

## 测试方法

### 使用pytest批量运行测试

Riemann使用pytest作为测试框架。您可以使用以下命令批量运行所有测试：

```bash
# 运行tests目录下所有测试文件
pytest tests

# 运行特定测试文件
pytest tests/test_010_grad.py

# 运行特定测试类或方法
pytest tests/test_011_jacobian.py::TestJacobianFunctions::test_single_input_single_output

# 运行测试并生成覆盖率报告
pytest --cov=riemann tests/

# 运行vision模块测试
pytest tests/test_052_vision.py
```

### 单独运行测试脚本

您也可以单独运行测试脚本：

```bash
cd tests
python test_010_grad.py
# 运行其他测试脚本
```

## 技术特点

- **高效实现**：优化的自动微分算法
- **易用API**：简洁明了的接口设计，与PyTorch接口保持高度兼容
- **灵活扩展**：支持自定义操作和导数规则
- **完整测试**：全面的单元测试覆盖
- **计算机视觉支持**：提供常用数据集和图像变换功能

## 应用场景

- **深度学习研究**：自定义模型和算法开发
- **科学计算**：复杂数学模型的梯度计算
- **优化问题求解**：梯度下降和Adam等优化算法
- **计算机视觉**：图像分类、目标检测等视觉任务
- **教育教学**：自动微分和深度学习原理学习

## 贡献指南

欢迎提交Issue和Pull Request！在贡献代码前，请确保通过所有测试。

## 第三方依赖及许可证

### Core Dependencies

| Library | Version Requirement | License Type | Notes                                     |
|---------|---------------------|--------------|-------------------------------------------|
| NumPy   | >=1.20.0            | BSD 3-Clause | Core numerical computation library        |
| SciPy   | >=1.7.0             | BSD 3-Clause | Linear algebra algorithms (LU, SVD, etc.) |

### Testing Dependencies

| Library  | Version Requirement | Purpose           | License Type  | Notes               |
|----------|---------------------|-------------------|---------------|---------------------|
| PyTorch  | >=2.0.0             | Result comparison | BSD 3-Clause  | Used for verifying  |
|          |                     | validation        |               | calculation results |
| pytest   | >=7.0.0             | Testing framework | MIT           | Used for organizing |
|          |                     |                   |               | and running tests   |

### Vision Dependencies

| Library | Version Requirement | Purpose          | License Type | Notes               |
|---------|---------------------|------------------|--------------|---------------------|
| Pillow  | >=8.0.0             | Image processing | BSD 3-Clause | Used for image      |
|         |                     |                  |              | loading/saving      |

*Note: This project also utilizes Python's standard library components (like unittest) for testing, which don't require separate installation.*

*Note: Details of the BSD 3-Clause license for NumPy, PyTorch and Pillow can be found on their official websites.*

## 许可证

本项目采用BSD 3-Clause许可证。详见LICENSE文件。

## 贡献

欢迎提交问题报告和拉取请求。请确保所有贡献都符合项目的编码标准。

## 联系方式

作者: Fei Xiang
邮箱: xfeix@outlook.com
Gitee: https://gitee.com/xfcode2021