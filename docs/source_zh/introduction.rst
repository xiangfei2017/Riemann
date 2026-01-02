简介
====

Riemann 是一个开源的轻量级自动微分库，专为学习、教育和研究自动微分和神经网络相关内容而设计。

主要功能
--------

张量操作

- 提供张量创建函数（tensor, zeros, ones, random等，支持复数张量）
- 支持基本的数学运算（加减乘除幂运算，指数、对数、三角、双曲等初等函数）
- 支持向量、矩阵运算（批量矩阵乘法、向量点积、矩阵行列式、矩阵逆、矩阵分解等）
- 支持张量形状重塑、维度扩缩、索引和切片、元素收集/散射、拼接/分割等操作

自动求导

- **backward方法**：触发反向传播计算梯度
- **grad函数**：计算函数相对于输入的梯度
- **track_track修饰器和Function类**：支持自定义梯度跟踪函数

雅可比矩阵和海森矩阵

- 支持多输入多输出函数的雅可比矩阵计算
- 提供海森矩阵计算功能用于二阶导数
- 高效计算雅可比-向量乘积和向量-雅可比乘积
- 支持海森-向量乘积和向量-海森乘积计算

线性代数模块

- 提供矩阵分解及其反向梯度跟踪（SVD、PLU, QR等）
- 支持求矩阵逆、广义逆、行列式、特征值/特征向量
- 矩阵范数、条件数计算
- 支持线性方程组求解、最小二乘求解

神经网络模块

- 基本层（Linear, Flatten, Dropout, BatchNorm等）
- 激活函数（ReLU, Sigmoid,Softmax等）
- 卷积池化层（Conv1d/2d/3d, MaxPool1d/2d/3d, AvgPool1d/2d/3d等）
- 损失函数（MSE, CrossEntropy等）
- 优化器（SGD, Adam, Adagrad, LBFGS等）
- 网络模块容器（Sequential, ModuleList, ModuleDict等）

计算机视觉模块

- 数据集类：MNIST、CIFAR10等常用数据集的加载和预处理
- 图像变换：Resize、Crop、Flip、Rotate、Normalize等图像预处理操作

PyTorch 接口兼容性
-------------------

Riemann 库在设计时考虑了与 PyTorch 接口的兼容性。同名函数和类保持一致的接口，使 PyTorch 用户能够快速上手：

- **张量操作**：支持与 PyTorch 同名的张量操作函数和方法，如 `tensor()`、`grad()`、`backward()` 等
- **神经网络组件**：`nn` 模块中的层、激活函数和损失函数与 PyTorch 保持接口兼容
- **优化器**：`optim` 模块中的优化器（如 SGD、Adam 等）与 PyTorch 保持一致的接口
- **自动微分机制**：`requires_grad`、计算图构建和反向传播机制与 PyTorch 类似
- **计算机视觉**：`vision` 模块中的数据集和变换与 torchvision 保持接口兼容

这种设计使得熟悉 PyTorch 的用户能够轻松迁移到 Riemann 库进行开发和研究工作。