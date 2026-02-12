# Riemann

Riemann是一个轻量级的自动求导库及神经网络编程框架，支持标量/向量/张量的自动梯度跟踪，提供搭建神经网络所需的常用组件，接口兼容PyTorch，为神经网络相关的学习、教育和研究目的而设计。


## 功能介绍

### 1. 张量操作
- 提供张量创建函数（tensor, zeros, ones, randn, normal等，支持复数张量）
- 支持基本的数学运算（加减乘除幂运算，指数、对数、三角、双曲等初等函数，求和、均值、方差、标准差等统计函数）
- 支持向量、矩阵运算（批量矩阵乘法、向量点积、矩阵行列式、矩阵逆、矩阵分解等）
- 支持张量形状重塑、维度扩缩、索引和切片、元素收集/散射、拼接/分割等操作
- 支持张量序列化/反序列化，方便模型训练和部署

### 2. 自动求导
- **backward方法**：触发反向传播计算梯度
- **grad函数**：计算函数相对于输入的梯度
- **track_grad修饰器和Function类**：支持自定义梯度跟踪函数
- **雅可比矩阵和海森矩阵**：支持多输入多输出函数的雅可比矩阵计算，支持多输入函数的海森矩阵计算

### 3. 线性代数模块
- 提供矩阵分解及其反向梯度跟踪（SVD、PLU, QR等）
- 支持求矩阵逆、广义逆、行列式、特征值/特征向量
- 矩阵范数、条件数计算
- 支持线性方程组求解、最小二乘求解

### 4. 神经网络模块
- 基本层（Linear, Dropout, BatchNorm, LayerNorm, Embedding等）
- 激活函数（ReLU, Sigmoid, Softmax等）
- 卷积池化层（Conv1d/2d/3d, MaxPool1d/2d/3d, AvgPool1d/2d/3d等）
- 损失函数（MSE, CrossEntropy等）
- 优化器（SGD, Adam, Adagrad, LBFGS等）
- 网络模块容器（Sequential, ModuleList, ModuleDict等）

### 5. 计算机视觉模块
- 数据集类：
  - **MNIST**：手写数字识别数据集
  - **CIFAR10**：10类彩色图像数据集

- 图像变换：
  - **基础变换**：ToTensor, ToPILImage, Normalize
  - **几何变换**：Resize, CenterCrop, RandomResizedCrop, FiveCrop, TenCrop, Pad
  - **随机变换**：RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomGrayscale
  - **颜色变换**：ColorJitter, Grayscale
  - **组合变换**：Compose, Lambda

### 6. CUDA/GPU支持
- 提供GPU加速，支持张量、模型在CPU和GPU之间迁移
- 优化的GPU计算性能


## 应用场景

- **深度学习研究**：自定义模型和算法开发
- **科学计算**：复杂数学模型的梯度计算
- **优化问题求解**：梯度下降和Adam等优化算法
- **计算机视觉**：图像分类、目标检测等视觉任务
- **教育教学**：自动微分和深度学习原理学习


## PyTorch接口兼容性

Riemann库设计时注重与PyTorch接口的兼容性，同名的函数和类接口保持一致，方便PyTorch用户快速上手：

- **张量操作**：支持与PyTorch同名的张量操作函数和方法，如`tensor()`、`grad()`、`backward()`等
- **神经网络组件**：`nn`模块中的层、激活函数和损失函数与PyTorch保持接口兼容
- **优化器**：`optim`模块中的优化器（如SGD、Adam等）接口与PyTorch保持一致
- **自动微分机制**：`requires_grad`、反向传播机制与PyTorch相似
- **计算机视觉**：`vision`模块中的数据集和变换与torchvision保持接口兼容

这种设计使得熟悉PyTorch的用户可以轻松迁移到Riemann库进行开发和研究工作。


## 项目文件夹结构

```
Riemann/
├── src/
│   └── riemann/              # 核心源代码
│       ├── autograd/         # 自动微分相关模块
│       ├── nn/               # 神经网络模块
│       ├── optim/            # 优化器模块
│       ├── utils/            # 工具函数
│       ├── vision/           # 计算机视觉模块
│       ├── __init__.py       # 包配置文件
│       ├── dtype.py          # 数据类型定义
│       ├── gradmode.py       # 梯度模式控制
│       ├── linalg.py         # 线性代数函数
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
│   ├── mnist_demo.py             # MNIST手写数字识别GUI示例
│   ├── nn_MNIST_CE_SGD.py        # 神经网络训练示例
│   ├── cnn_CIFAR10_CE_SGD.py     # 卷积网路训练示例
│   └── ...
├── README.md                 # 项目文档
├── LICENSE                   # 许可证文件
└── pyproject.toml            # 项目配置和依赖管理
```

### 代码目录

代码目录 `src/riemann/` 是Riemann库的核心源代码所在地，包含了所有主要功能模块：

- **autograd/**: 实现自动微分功能，包括反向传播算法和梯度计算
- **nn/**: 神经网络相关组件，如各种层、激活函数和损失函数
- **optim/**: 优化器模块，如SGD、Adam等优化器，以及学习率调度器
- **utils/**: 工具函数，包括数据集Dataset类和数据加载器Dataloader类等
- **vision/**: 计算机视觉相关功能，包括数据集和图像变换
- **核心文件**: 如 `tensordef.py`（张量定义）、`linalg.py`（线性代数）等

### tests目录

`tests/` 目录包含了大量测试用例，用于验证Riemann库的各种功能是否正常工作：

- 测试用例按模块分类，覆盖了自动微分、张量操作、神经网络等各个功能模块
- 测试用例既可以使用pytest批量运行，也可以作为独立脚本单独运行
- 提供了详细的测试覆盖，确保库的稳定性和可靠性

### 文档docs目录

`docs/` 目录用于存放项目文档：

- 包含详细的API文档
- 提供使用指南和教程
- 记录项目架构和设计决策
- 帮助用户和开发者更好地理解和使用Riemann库

### examples目录

`examples/` 目录包含了各种示例代码，展示如何使用Riemann库的不同功能：

- **基础示例**: 如反向传播示例（backward_demo.py）、梯度计算示例（grad_demo.py）
- **自定义梯度示例**: 如custom_grad_decorator.py，展示如何使用装饰器自定义梯度
- **优化器示例**: 如optimizers_comparison.py，比较不同优化器的性能
- **神经网络示例**: 如nn_MNIST_CE_SGD.py（MNIST手写数字识别）、cnn_CIFAR10_CE_SGD.py（CIFAR10图像分类）
- **GUI应用示例**: 如mnist_demo.py，提供图形界面的MNIST手写数字识别应用

这些示例代码为用户提供了实际使用Riemann库的参考，帮助用户快速上手和理解库的功能。

Riemann提供了丰富的示例代码，位于`examples/`目录：

- **backward_demo.py**: backward函数使用演示
- **grad_demo.py**: grad函数使用演示
- **hessian.py**: 海森矩阵计算示例
- **jacobian.py**: 雅可比矩阵计算示例
- **mnist_demo.py**: MNIST手写数字识别GUI示例（图形界面训练与识别）
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


## 安装方法

### 从PyPI安装

#### 基本安装

直接运行以下命令安装Riemann及其核心依赖：

```bash
pip install riemann
```

#### 安装CUDA依赖

如果需要使用GPU加速，需要显式指定CUDA依赖选项：

```bash
# 安装默认CUDA版本（推荐CUDA 12.x）
pip install riemann[cuda]

# 对于特定CUDA版本的用户，也可以直接安装对应版本的依赖
# CUDA 13.x
pip install riemann[cuda13]
# CUDA 12.x
pip install riemann[cuda12]
# CUDA 11.x
pip install riemann[cuda11]
# CUDA 10.x (仅Linux)
pip install riemann[cuda10]
```

#### 安装其他可选依赖

```bash
# 安装测试依赖（用于运行tests目录下的测试代码）
pip install riemann[tests]

# 安装通用cupy依赖（适用于macOS、ARM64等不支持CUDA的平台）
pip install riemann[cupy]
```

### 源码安装与开发环境配置

```bash
# 从Gitee获取Riemann库源码
git clone https://gitee.com/xfcode2021/Riemann.git
cd Riemann
# 使用开发模式安装包及其核心依赖（-e表示可编辑模式，修改代码后无需重新安装）
pip install -e .

# 安装测试依赖
pip install -e .[tests]

# 安装CUDA依赖
# 注意：使用CUDA加速前，请确保已安装对应版本的CUDA驱动
pip install -e .[cuda]

# 安装特定版本的CUDA依赖
# CUDA 13.x
pip install -e .[cuda13]
# CUDA 12.x
pip install -e .[cuda12]
# CUDA 11.x
pip install -e .[cuda11]
# CUDA 10.x (仅Linux)
pip install -e .[cuda10]

# 安装通用cupy依赖（适用于macOS、ARM64等平台）
pip install -e .[cupy]
```

### 依赖说明

#### 核心依赖

直接运行`pip install riemann`会自动安装以下核心依赖：
- **numpy>=1.20.0**: 核心数值计算库
- **pillow>=8.0.0**: 用于计算机视觉中的图像处理功能
- **tqdm>=4.0.0**: 用于神经网络训练中的进度条显示

#### CUDA依赖

CUDA依赖不会自动安装，需要显式指定：
- **cupy-cuda13x**: 适用于CUDA 13.x的Linux和Windows平台
- **cupy-cuda12x**: 适用于CUDA 12.x的Linux和Windows平台
- **cupy-cuda11x**: 适用于CUDA 11.x的Linux和Windows平台
- **cupy-cuda10x**: 适用于CUDA 10.x的Linux平台
- **cupy**: 通用版本（适用于macOS、ARM64等不支持CUDA的平台）

#### 平台兼容性

- **CUDA支持**: 仅在Linux或Windows系统，且x86_64/AMD64架构下可用
- **macOS系统**: 不支持NVIDIA CUDA，会自动使用CPU模式
- **ARM架构**: 对于NVIDIA Jetson等设备可能支持CUDA，但需要安装对应ARM版本的CUDA驱动

### CUDA驱动安装说明

1. **检查CUDA驱动版本**
   - Windows系统：右键点击桌面 → NVIDIA控制面板 → 帮助 → 系统信息 → 驱动程序版本
   - Linux系统：终端运行 `nvidia-smi` 或 `nvcc --version`

2. **下载并安装对应版本的CUDA驱动**
   - 访问NVIDIA官方网站：https://developer.nvidia.com/cuda-toolkit-archive
   - 根据你的GPU型号和操作系统选择合适的驱动版本
   - 安装过程中请按照向导提示完成

3. **验证CUDA驱动安装**
   - 安装完成后，重启电脑并运行：
     - Windows: `nvidia-smi`
     - Linux: `nvidia-smi` 或 `nvcc --version`
   - 确认输出中显示正确的CUDA版本信息

### 验证安装

安装完成后，可以运行以下代码验证：

```python
import riemann as r
print("CUDA可用:", r.cuda.is_available())
print("使用设备:", r.device('cuda' if r.cuda.is_available() else 'cpu'))
```

如果CUDA安装成功，会显示`CUDA可用: True`，否则会显示`CUDA可用: False`并自动使用CPU模式。


## 使用方法

### riemann包的模块结构

```
riemann                  # 主包
├── autograd             # 自动微分模块
│   └── functional       # 自动微分函数式接口
├── linalg               # 线性代数模块
├── nn                   # 神经网络模块
│   └── functional       # 神经网络函数
├── optim                # 优化器模块
│   └── lr_scheduler     # 学习率调度器模块
├── utils                # 工具函数模块
│   └── data             # 数据处理工具
├── vision               # 计算机视觉模块
│   ├── datasets         # 数据集类
│   └── transforms       # 图像变换操作
└── cuda                 # CUDA/GPU支持
```

### 模块导入示例

**整体导入riemann模块：**

```python
import riemann as r

# 使用张量创建函数
t = r.tensor([1.0, 2.0, 3.0])

# 使用自动微分功能
x = r.tensor([1.0, 2.0], requires_grad=True)
y = x ** 2
y.sum().backward()
print(x.grad)  # 输出: [2. 4.]
```

**按模块树导入需要的函数和类：**

```python
# 导入张量相关功能
from riemann import tensor, zeros, ones, randn

# 导入自动微分功能
from riemann.autograd import grad, backward
from riemann.autograd.functional import jacobian, hessian

# 导入线性代数功能
from riemann import linalg
from riemann.linalg import svd, det, inv

# 导入神经网络组件
from riemann.nn import Linear, Conv2d, ReLU, CrossEntropyLoss
from riemann.nn.functional import relu, cross_entropy

# 导入优化器
from riemann.optim import SGD, Adam, Adagrad

# 导入计算机视觉功能
from riemann.vision.datasets import MNIST, CIFAR10
from riemann.vision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip

# 导入CUDA支持
from riemann import cuda
from riemann.cuda import is_available, Device
```

### 例1：求导数（梯度）示例

```python
# 例1：求导数（梯度）示例
# 本示例展示了两种计算梯度的方法：
# 1. 使用grad函数直接计算函数相对于输入的梯度
# 2. 使用backward方法通过反向传播计算梯度

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

### 例2：计算雅可比矩阵示例

```python
# 例2：计算雅可比矩阵示例
# 本示例展示了如何使用jacobian函数计算函数相对于输入的雅可比矩阵
# 雅可比矩阵是函数输出对输入的偏导数矩阵，对于多输入多输出函数非常重要

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

### 例3：简单神经网络训练示例

```python
# 例3：简单神经网络训练示例
# 本示例展示了如何训练一个简单的神经网络（线性回归模型）来求两数之和
# 包括模型创建、损失函数定义、优化器配置、训练循环和预测过程

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

### 例4：简单卷积神经网络训练示例

```python
# 例4：简单卷积神经网络训练示例
# 本示例展示了如何使用卷积神经网络（CNN）训练CIFAR10图像分类模型
# 包括模型定义、数据加载与预处理、训练循环、模型评估和单个样本推理

import riemann as r
from riemann.vision.datasets import CIFAR10
from riemann.vision.transforms import *
from riemann.nn import *
from riemann.optim import SGD
from tqdm import tqdm

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
test_loader = r.utils.DataLoader(test_dataset, batch_size=1, shuffle=False)

# 创建模型、损失函数和优化器
model = Sequential(
    Conv2d(3, 16, kernel_size=3, padding=1),
    ReLU(),
    MaxPool2d(kernel_size=2, stride=2),
    Flatten(),
    Linear(16 * 16 * 16, 10)
)
criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.01)

# 训练循环
for epoch in range(3):  # 训练3代
    total_loss = 0
    # 使用tqdm显示进度条
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/3")
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        # 前向传播
        output = model(data)
        loss = criterion(output, target)   # 计算输出与目标标签间的损失
        
        # 反向传播和优化器更新
        optimizer.zero_grad()   # 清空训练参数的梯度
        loss.backward()         # 计算loss对训练参数的梯度
        optimizer.step()        # 更新训练参数
        
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
        predicted = outputs.argmax(dim=1)  # 获取每个样本的预测类别
        total += target.size(0)  # 累加测试样本数
        correct += (predicted == target).sum().item() # 累加正确预测的样本数
        
        # 更新进度条显示当前准确率
        current_accuracy = 100 * correct / total
        test_progress_bar.set_postfix({"Accuracy": f"{current_accuracy:.2f}%"})

# 输出最终测试准确率
test_accuracy = 100 * correct / total
print(f"测试集准确率: {test_accuracy:.2f}% ({correct}/{total})")

# 单个样本推理示例
sample_data, sample_target = next(iter(test_loader))
sample_output = model(sample_data[:1])  # 只取第一个样本
predicted_class = sample_output.argmax(dim=1)
print(f"样本预测类别: {predicted_class.item()}, 实际类别: {sample_target[0].item()}")

print("CNN训练和推理测试完成！")
```

### 例5：图像变换示例

```python
# 例5：图像变换示例
# 本示例展示了如何使用Riemann的图像变换功能对图像进行预处理
# 包括调整大小、中心裁剪、随机水平翻转、转换为张量和标准化等操作

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

### 例6：GPU加速示例

```python
# 例6：GPU加速示例
# 本示例展示了如何在Riemann中使用GPU加速神经网络训练
# 包括设备检测与设置、模型和数据的设备迁移、GPU上的训练和评估过程

import riemann as r
from riemann.nn import Linear, Flatten, ReLU, Sequential, CrossEntropyLoss
from riemann.optim import Adam
from riemann.vision.datasets import MNIST
from riemann.vision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm

# 检查CUDA是否可用
device = r.device('cuda' if r.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载MNIST数据集
transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])

train_dataset = MNIST(root='data', train=True, transform=transform)
test_dataset = MNIST(root='data', train=False, transform=transform)

train_loader = r.utils.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = r.utils.DataLoader(test_dataset, batch_size=1, shuffle=False)

# 创建模型并移动到指定设备
model = Sequential(
    Flatten(),
    Linear(28*28, 128),
    ReLU(),
    Linear(128, 10)
)
model.to(device)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        # 将数据移动到指定设备
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

# 模型评估
model.eval()
correct = 0
total = 0

with r.no_grad():
    test_progress_bar = tqdm(test_loader, desc="Testing")
    for data, target in test_progress_bar:
        # 将数据移动到指定设备
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        outputs = model(data)
        
        # 获取预测结果
        predicted = outputs.argmax(dim=1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        current_accuracy = 100 * correct / total
        test_progress_bar.set_postfix({"Accuracy": f"{current_accuracy:.2f}%"})

# 输出最终测试准确率
test_accuracy = 100 * correct / total
print(f"测试集准确率: {test_accuracy:.2f}% ({correct}/{total})")
```

## 测试方法

Riemann项目文件夹中tests目录下包括覆盖所有功能的测试用例，这些测试用例既可以用pytest批量运行，也可以作独立脚本运行。

### 使用pytest批量运行测试

您可以使用以下命令批量运行所有测试：

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


## 第三方依赖及许可证

### Core Dependencies

| Library | Version Requirement | License Type | Notes                                     |
|---------|---------------------|--------------|-------------------------------------------|
| NumPy   | >=1.20.0            | BSD 3-Clause | Core numerical computation library        |
| Pillow  | >=8.0.0             | BSD 3-Clause | Image processing library                  |
| tqdm    | >=4.0.0             | MIT          | Progress bar for training and data loading|

### Testing Dependencies
| Library    | Version Requirement | Purpose           | License Type  | Notes                                     |
|------------|---------------------|-------------------|---------------|-------------------------------------------|
| PyTorch    | >=2.0.0             | Result comparison | BSD 3-Clause  | Used for verifying calculation results    |
| torchvision| >=0.15.0            | Computer vision   | BSD 3-Clause  | PyTorch's computer vision library         |
| pytest     | >=7.0.0             | Testing framework | MIT           | Used for organizing and running tests     |

### Optional CUDA Dependencies

| Library      | Version Requirement | Purpose           | License Type | Notes                                     |
|--------------|---------------------|-------------------|--------------|-------------------------------------------|
| cupy-cuda12x | Latest              | GPU acceleration  | MIT          | For CUDA 12.x on Linux x86_64             |
| cupy-cuda11x | Latest              | GPU acceleration  | MIT          | For CUDA 11.x on Linux x86_64             |
| cupy-cuda10x | Latest              | GPU acceleration  | MIT          | For CUDA 10.x on Linux x86_64             |
| cupy         | Latest              | GPU acceleration  | MIT          | For other platforms (including Windows)   |

*Note: This project also utilizes Python's standard library components (like unittest) for testing, which don't require separate installation.*

*Note: Details of the BSD 3-Clause license for NumPy, PyTorch and Pillow can be found on their official websites.*

## 许可证

本项目采用BSD 3-Clause许可证。详见LICENSE文件。

## 贡献指南

欢迎提交Issue和Pull Request！在贡献代码前，请确保所有贡献都符合项目的编码标准，并通过所有测试。

## 联系方式

作者: Fei Xiang
邮箱: xfeix@outlook.com
Gitee: https://gitee.com/xfcode2021
