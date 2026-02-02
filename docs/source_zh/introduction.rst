简介
====

Riemann 是一个轻量级的自动求导库及神经网络编程框架，支持标量/向量/张量的自动梯度跟踪，提供搭建神经网络所需的常用组件，接口兼容 PyTorch，为神经网络相关的学习、教育和研究目的而设计。

主要功能
--------

张量操作

- 提供张量创建函数（tensor, zeros, ones, randn, normal 等，支持复数张量）
- 支持基本的数学运算（加减乘除幂运算，指数、对数、三角、双曲等初等函数，求和、均值、方差、标准差等统计函数）
- 支持向量、矩阵运算（批量矩阵乘法、向量点积、矩阵行列式、矩阵逆、矩阵分解等）
- 支持张量形状重塑、维度扩缩、索引和切片、元素收集/散射、拼接/分割等操作
- 支持张量序列化/反序列化，方便模型训练和部署

自动求导

- **backward 方法**：触发反向传播计算梯度
- **grad 函数**：计算函数相对于输入的梯度
- **track_grad 修饰器和 Function 类**：支持自定义梯度跟踪函数
- **雅可比矩阵和海森矩阵**：支持多输入多输出函数的雅可比矩阵计算，支持多输入函数的海森矩阵计算

线性代数模块

- 提供矩阵分解及其反向梯度跟踪（SVD、PLU, QR 等）
- 支持求矩阵逆、广义逆、行列式、特征值/特征向量
- 矩阵范数、条件数计算
- 支持线性方程组求解、最小二乘求解

神经网络模块

- 基本层（Linear, Dropout, BatchNorm, LayerNorm, Embedding 等）
- 激活函数（ReLU, Sigmoid, Softmax 等）
- 卷积池化层（Conv1d/2d/3d, MaxPool1d/2d/3d, AvgPool1d/2d/3d 等）
- 损失函数（MSE, CrossEntropy 等）
- 优化器（SGD, Adam, Adagrad, LBFGS 等）
- 网络模块容器（Sequential, ModuleList, ModuleDict 等）

计算机视觉模块

- 数据集类：
  - **MNIST**：手写数字识别数据集
  - **CIFAR10**：10 类彩色图像数据集

- 图像变换：
  - **基础变换**：ToTensor, ToPILImage, Normalize
  - **几何变换**：Resize, CenterCrop, RandomResizedCrop, FiveCrop, TenCrop, Pad
  - **随机变换**：RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomGrayscale
  - **颜色变换**：ColorJitter, Grayscale
  - **组合变换**：Compose, Lambda

CUDA/GPU 支持

- 提供 GPU 加速，支持张量、模型在 CPU 和 GPU 之间迁移
- 优化的 GPU 计算性能

应用场景
--------

- **深度学习研究**：自定义模型和算法开发
- **科学计算**：复杂数学模型的梯度计算
- **优化问题求解**：梯度下降和 Adam 等优化算法
- **计算机视觉**：图像分类、目标检测等视觉任务
- **教育教学**：自动微分和深度学习原理学习

PyTorch 接口兼容性
-------------------

Riemann 库设计时注重与 PyTorch 接口的兼容性，同名的函数和类接口保持一致，方便 PyTorch 用户快速上手：

- **张量操作**：支持与 PyTorch 同名的张量操作函数和方法，如 `tensor()`、`grad()`、`backward()` 等
- **神经网络组件**：`nn` 模块中的层、激活函数和损失函数与 PyTorch 保持接口兼容
- **优化器**：`optim` 模块中的优化器（如 SGD、Adam 等）接口与 PyTorch 保持一致
- **自动微分机制**：`requires_grad`、反向传播机制与 PyTorch 相似
- **计算机视觉**：`vision` 模块中的数据集和变换与 torchvision 保持接口兼容

这种设计使得熟悉 PyTorch 的用户可以轻松迁移到 Riemann 库进行开发和研究工作。