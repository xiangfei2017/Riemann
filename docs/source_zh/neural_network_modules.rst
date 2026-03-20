如何搭建神经网络
================

Riemann 通过 ``riemann.nn`` 包提供了一套全面的神经网络模块。这些模块是创建和训练神经网络的构建块。

本章节将详细讲解如何使用 Riemann 搭建、训练和评估一个完整的神经网络。我们将以 **MNIST手写数字识别** 为例，演示从数据准备到模型评估的完整流程。

步骤1：数据准备
----------------

在构建神经网络之前，需要准备好数据集。Riemann 提供了 ``Dataset`` 和 ``DataLoader`` 接口用于数据加载和处理。

理解 Dataset
~~~~~~~~~~~~

``Dataset`` 是用于表示数据集的抽象基类。它定义了两个子类必须实现的核心方法：

- ``__len__()``: 返回数据集的样本数量
- ``__getitem__(idx)``: 根据索引返回一个样本

**自定义数据集**

如果你需要处理自己的数据，可以通过继承 ``Dataset`` 类并实现上述两个方法来创建自定义数据集。例如：

.. code-block:: python

    from riemann.utils.data import Dataset
    
    class MyDataset(Dataset):
        def __init__(self, data, labels, transform=None):
            self.data = data
            self.labels = labels
            self.transform = transform
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            sample = self.data[idx]
            label = self.labels[idx]
            
            if self.transform:
                sample = self.transform(sample)
            
            return sample, label

**使用内置数据集**

Riemann 为常见任务提供了内置数据集，简化了数据加载过程。对于计算机视觉任务，可以使用 ``riemann.vision.datasets`` 中的数据集。本示例将直接使用 Riemann 提供的 MNIST 手写数字数据集：

.. code-block:: python

    from riemann.vision.datasets import MNIST

使用 Transforms 进行数据变换
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``transforms`` 用于数据预处理和增强。可以使用 ``transforms.Compose`` 组合多个变换操作：

.. code-block:: python

    from riemann.vision import transforms

    # 定义数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),           # 将图像转换为张量
        transforms.Normalize((0.1307,), (0.3081,))  # 使用均值和标准差进行归一化
    ])

**关键概念：**

- ``ToTensor()``: 将 PIL Image 或 numpy 数组转换为张量，并将像素值从 [0, 255] 缩放到 [0.0, 1.0]
- ``Normalize(mean, std)``: 使用均值和标准差对张量进行归一化: ``output = (input - mean) / std``

加载 MNIST 数据集
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # 加载训练集和测试集
    train_dataset = MNIST(
        root='./data',      # 数据存储/加载目录
        train=True,         # True 表示训练集，False 表示测试集
        transform=transform # 要应用的数据变换
    )
    
    test_dataset = MNIST(
        root='./data',
        train=False,
        transform=transform
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

使用 DataLoader 进行批量处理
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``DataLoader`` 用于批量加载数据，支持数据打乱和自动批处理。它是连接 Dataset 和训练循环的桥梁，能够高效地将数据组织成批次供模型训练使用。

**为什么需要 DataLoader？**

在神经网络训练中，我们通常不会一次性将所有数据输入模型，而是采用**批次训练（Batch Training）**的方式：

1. **内存效率**：大规模数据集可能无法一次性加载到内存中，批次处理可以分块加载数据
2. **训练稳定性**：小批次数据的梯度估计噪声有助于逃离局部最优，大批量数据的梯度估计更稳定
3. **硬件利用率**：现代 GPU/CPU 对矩阵运算有高度优化，批次处理可以充分利用并行计算能力
4. **收敛速度**：适当的批次大小可以加速模型收敛

**示例代码：**

.. code-block:: python

    from riemann.utils.data import DataLoader

    # 创建训练数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=100,     # 每批次的样本数量
        shuffle=True        # 每个 epoch 打乱数据
    )
    
    # 创建测试数据加载器
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,       # 测试时逐个处理样本
        shuffle=False       # 测试数据不需要打乱
    )

**关键参数：**

- ``dataset``: 要加载数据的数据集
- ``batch_size``: 每批加载的样本数。训练时通常设置为 32、64、100 等；测试时可设置为 1 或更大的值
- ``shuffle``: 设置为 True 以在每个 epoch 重新打乱数据顺序，有助于防止模型记住数据顺序，提高泛化能力

步骤2：构建神经网络
--------------------

Riemann 中的神经网络通过继承 ``nn.Module`` 并实现 ``forward`` 方法来构建。

理解 nn.Module
~~~~~~~~~~~~~~~

``nn.Module`` 是所有神经网络模块的基类。它提供了：

- **参数管理**: 自动跟踪可学习参数
- **子模块管理**: 支持嵌套模块
- **设备管理**: 支持 CPU/GPU 执行
- **训练/评估模式**: ``train()`` 和 ``eval()`` 方法

定义网络架构
~~~~~~~~~~~~

对于 MNIST 分类任务，我们构建一个简单的前馈神经网络。在深入代码之前，让我们先理解每个组件的作用。

**网络组件详解**

1. **Sequential 容器**

``nn.Sequential`` 是一个按顺序执行模块的容器。它将多个层按顺序堆叠，数据会依次通过每一层。使用 Sequential 的好处是代码简洁、结构清晰，特别适合简单的顺序网络。

2. **Flatten 层**

``nn.Flatten`` 用于展平输入张量。MNIST 图像是 28×28 像素的二维图像，但全连接层需要一维向量输入。Flatten 将形状为 ``(batch_size, 1, 28, 28)`` 的张量转换为 ``(batch_size, 784)``，其中 784 = 28 × 28。

3. **Linear 层（全连接层）**

``nn.Linear(in_features, out_features)`` 是全连接层，执行线性变换：``output = input @ weight.T + bias``

- 权重矩阵 ``weight`` 的形状为 ``(out_features, in_features)``
- 偏置向量 ``bias`` 的形状为 ``(out_features,)``
- 这些参数会在训练过程中自动学习和更新

4. **激活函数（ReLU）**

``nn.ReLU()`` 是修正线性单元激活函数，定义为：``f(x) = max(0, x)``

- **作用**：引入非线性，使网络能够学习复杂的模式。如果没有激活函数，多层线性变换等价于单层线性变换，无法学习非线性关系
- **优点**：计算简单、缓解梯度消失问题、加速收敛

5. **损失函数（CrossEntropyLoss）**

``nn.CrossEntropyLoss`` 是多分类任务的损失函数，它结合了 LogSoftmax 和 NLLLoss（负对数似然损失）：

- **作用**：衡量模型预测与真实标签之间的差异
- **计算**：``loss = -log(softmax(output)[target_class])``
- **目标**：通过最小化损失函数，使模型预测更接近真实标签

6. **优化器（Adam）**

``opt.Adam`` 是一种自适应学习率优化算法：

- **作用**：根据计算出的梯度更新网络参数，使损失函数逐渐减小
- **原理**：结合动量（Momentum）和 RMSProp 的优点，为每个参数维护独立的学习率
- **参数**：
  - ``lr`` (learning rate): 学习率，控制参数更新的步长
  - ``betas``: 动量系数，控制梯度累积的速度
  - ``weight_decay``: L2 正则化系数，防止过拟合

**代码实现：**

.. code-block:: python

    import riemann.nn as nn
    import riemann.optim as opt

    class Classifier(nn.Module):
        """
        MNIST 手写数字分类器
        
        网络架构：
        - 输入层: 784 个神经元 (28x28 像素展平)
        - 隐藏层: 200 个神经元，使用 ReLU 激活
        - 输出层: 10 个神经元 (对应数字 0-9)
        """
        def __init__(self):
            super().__init__()
            
            # 使用 Sequential 容器定义网络层
            self.model = nn.Sequential(
                nn.Flatten(),           # 将 (batch, 1, 28, 28) 展平为 (batch, 784)
                nn.Linear(784, 200),    # 输入层到隐藏层：784维 -> 200维
                nn.ReLU(),              # 激活函数：引入非线性
                nn.Linear(200, 10)      # 隐藏层到输出层：200维 -> 10维（10个数字类别）
            )
            
            # 定义多分类任务的损失函数
            self.loss_func = nn.CrossEntropyLoss()
            
            # 使用 Adam 算法定义优化器
            self.optimizer = opt.Adam(
                self.parameters(),      # 要优化的参数（所有 Linear 层的 weight 和 bias）
                lr=0.001,               # 学习率：控制参数更新的步长
                betas=(0.9, 0.999),     # 动量系数
                weight_decay=0.0001     # L2 正则化：防止过拟合
            )
        
        def forward(self, inputs):
            """
            前向传播
            
            参数:
                inputs: 形状为 (batch_size, 1, 28, 28) 的张量
            
            返回:
                形状为 (batch_size, 10) 的张量 - 未归一化的 logits
            """
            return self.model(inputs)

**关键概念总结：**

- ``nn.Sequential``: 按顺序执行模块的容器，简化网络定义
- ``nn.Flatten``: 展平多维输入，适配全连接层
- ``nn.Linear``: 全连接层，包含可学习的权重和偏置
- ``nn.ReLU``: 非线性激活函数，使网络能够学习复杂模式
- ``nn.CrossEntropyLoss``: 分类损失函数，衡量预测与真实值的差异
- ``opt.Adam``: 自适应优化器，自动调整参数更新步长

步骤3：训练网络
----------------

训练涉及多次迭代数据集（epoch），计算预测、计算损失并更新参数。

实现训练步骤
~~~~~~~~~~~~

.. code-block:: python

    class Classifier(nn.Module):
        # ... 上面的 __init__ 和 forward 方法 ...
        
        def train_step(self, inputs, targets):
            """
            执行一步训练
            
            参数:
                inputs: 一批图像，形状为 (batch_size, 1, 28, 28)
                targets: 一批标签，形状为 (batch_size,)
            
            返回:
                loss: 标量损失值
            """
            # 前向传播：计算预测
            outputs = self.forward(inputs)
            
            # 计算损失
            loss = self.loss_func(outputs, targets)
            
            # 反向传播：计算梯度
            self.optimizer.zero_grad(True)  # 清除之前的梯度
            loss.backward()                  # 计算梯度
            
            # 更新参数
            self.optimizer.step()
            
            return loss

完整的训练循环
~~~~~~~~~~~~~~

.. code-block:: python

    # 创建模型实例
    model = Classifier()
    
    # 训练配置
    epochs = 3
    
    # 训练循环
    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        epoch_loss = 0.0
        num_batches = len(train_loader)
        
        # 遍历批次
        for batch_idx, batch in enumerate(train_loader):
            img_tensors, target_tensors = batch
            
            # 执行训练步骤
            loss = model.train_step(img_tensors, target_tensors)
            epoch_loss += loss.item()
            
            # 每100个批次打印进度
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, '
                      f'Batch {batch_idx}/{num_batches}, '
                      f'Loss: {loss.item():.4f}')
        
        # 计算该 epoch 的平均损失
        avg_loss = epoch_loss / num_batches
        print(f'Epoch {epoch+1}/{epochs} 完成, 平均损失: {avg_loss:.4f}')

**训练过程说明：参数如何一步一步学习**

神经网络的训练本质上是一个**优化问题**：通过不断调整网络参数（权重和偏置），使损失函数的值最小化。让我们详细看看这个过程：

**1. 前向传播（Forward Propagation）**

输入数据从输入层经过隐藏层传递到输出层，逐层计算得到预测结果：

- 输入图像经过 Flatten 展平为 784 维向量
- 通过第一个 Linear 层：``h1 = x @ W1.T + b1`` (784维 -> 200维)
- 经过 ReLU 激活：``h1_relu = max(0, h1)``
- 通过第二个 Linear 层：``output = h1_relu @ W2.T + b2`` (200维 -> 10维)
- 输出 10 个 logits，代表每个数字（0-9）的预测分数

**2. 损失计算（Loss Computation）**

计算预测结果与真实标签之间的差异：

- 使用 CrossEntropyLoss 计算损失值
- 损失值越大，表示预测与真实值差距越大
- 训练目标就是最小化这个损失值

**3. 反向传播（Backward Propagation）**

计算损失函数对每个参数的梯度（偏导数）：

- 从输出层开始，逐层向后计算梯度
- 使用链式法则：``∂L/∂W = ∂L/∂output * ∂output/∂W``
- 梯度告诉我们：如何调整参数才能使损失减小
- 梯度为正，表示增大该参数会增大损失；梯度为负则相反

**4. 参数更新（Parameter Update）**

优化器根据梯度更新参数：

- **梯度下降原理**：``W_new = W_old - lr * gradient``
- ``lr`` 是学习率，控制更新步长
- Adam 优化器还会考虑历史梯度信息，自适应调整每个参数的学习率
- 更新后，网络参数变得更优，预测能力更强

**训练循环的本质**

通过成千上万次的"前向传播 → 计算损失 → 反向传播 → 更新参数"循环，网络逐渐学会从输入图像中提取特征并正确分类。这个过程类似于学生通过不断练习和纠正错误来提高成绩。

步骤4：评估与推理
------------------

训练完成后，在测试集上评估模型以衡量其泛化性能。

评估方法
~~~~~~~~

.. code-block:: python

    class Classifier(nn.Module):
        # ... 前面的方法 ...
        
        def evaluate(self, dataloader):
            """
            评估模型性能
            
            参数:
                dataloader: 提供测试数据的 DataLoader
            
            返回:
                accuracy: 分类准确率 (0-1)
                avg_loss: 数据集上的平均损失
            """
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in dataloader:
                img_tensors, target_tensors = batch
                
                # 前向传播
                outputs = self.forward(img_tensors)
                
                # 计算损失
                loss = self.loss_func(outputs, target_tensors)
                total_loss += loss.item()
                
                # 计算准确率
                predicted = outputs.argmax(dim=1)  # 获取预测的类别
                total += target_tensors.size(0)
                correct += (predicted == target_tensors).sum().item()
            
            accuracy = correct / total
            avg_loss = total_loss / len(dataloader)
            return accuracy, avg_loss

运行评估
~~~~~~~~

.. code-block:: python

    # 设置模型为评估模式
    model.eval()
    
    # 在测试集上评估
    test_accuracy, test_loss = model.evaluate(test_loader)
    print(f'测试准确率: {test_accuracy:.4f}')
    print(f'测试损失: {test_loss:.4f}')

**关键点：**

- ``model.eval()``: 将模型设置为评估模式（禁用 dropout 等）
- ``outputs.argmax(dim=1)``: 获取最大值所在的索引（预测的类别）
- 评估过程不应修改模型参数

**准确率与哪些因素有关？**

模型的准确率（Accuracy）是衡量模型性能的重要指标，表示预测正确的样本占总样本的比例。准确率受多种因素影响：

**1. 网络架构因素**

- **网络深度和宽度**：层数更多、神经元更多的网络通常有更强的表达能力，但也更容易过拟合
- **激活函数选择**：ReLU、Sigmoid、Tanh 等不同激活函数影响网络的学习能力和收敛速度
- **层间连接方式**：全连接、卷积、循环等不同结构适用于不同类型的数据

**2. 训练相关因素**

- **训练轮数（Epochs）**：训练不足会导致欠拟合，训练过多可能导致过拟合
- **批次大小（Batch Size）**：影响梯度估计的准确性和训练稳定性
- **学习率（Learning Rate）**：过大导致震荡不收敛，过小导致收敛缓慢
- **优化器选择**：SGD、Adam、RMSprop 等不同优化器有不同的收敛特性

**3. 数据相关因素**

- **数据质量**：噪声、错误标注会降低模型性能
- **数据量**：更多的训练数据通常能带来更好的泛化能力
- **数据分布**：训练集和测试集分布不一致会导致性能下降
- **数据预处理**：归一化、数据增强等预处理手段对准确率有显著影响

**4. 正则化因素**

- **L1/L2 正则化**：防止过拟合，提高泛化能力
- **Dropout**：随机丢弃神经元，减少共适应
- **早停（Early Stopping）**：在验证集性能开始下降前停止训练

**5. 初始化因素**

- **权重初始化**：良好的初始化（如 Xavier、He 初始化）可以加速收敛并提高最终性能

理解这些因素有助于你在实际应用中诊断问题并优化模型性能。

步骤5：完整示例
----------------

以下是 MNIST 手写数字识别的完整可运行代码：

.. code-block:: python

    import sys
    import os
    import time
    
    # 导入 Riemann 模块
    import riemann.nn as nn
    import riemann.optim as opt
    from riemann.vision.datasets import MNIST
    from riemann.vision import transforms
    from riemann.utils.data import DataLoader


    class Classifier(nn.Module):
        """MNIST 手写数字分类器"""
        
        def __init__(self):
            super().__init__()
            
            # 网络架构
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 200),
                nn.ReLU(),
                nn.Linear(200, 10)
            )
            
            # 损失函数和优化器
            self.loss_func = nn.CrossEntropyLoss()
            self.optimizer = opt.Adam(
                self.parameters(),
                lr=0.001,
                betas=(0.9, 0.999),
                weight_decay=0.0001
            )
        
        def forward(self, inputs):
            return self.model(inputs)
        
        def train_step(self, inputs, targets):
            outputs = self.forward(inputs)
            loss = self.loss_func(outputs, targets)
            self.optimizer.zero_grad(True)
            loss.backward()
            self.optimizer.step()
            return loss
        
        def evaluate(self, dataloader):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in dataloader:
                img_tensors, target_tensors = batch
                outputs = self.forward(img_tensors)
                
                loss = self.loss_func(outputs, target_tensors)
                total_loss += loss.item()
                
                predicted = outputs.argmax(dim=1)
                total += target_tensors.size(0)
                correct += (predicted == target_tensors).sum().item()
            
            accuracy = correct / total
            avg_loss = total_loss / len(dataloader)
            return accuracy, avg_loss


    def main():
        print("MNIST 手写数字识别")
        
        # 步骤1：数据准备
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        print("加载数据集...")
        train_dataset = MNIST(root='./data', train=True, transform=transform)
        test_dataset = MNIST(root='./data', train=False, transform=transform)
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        
        # 步骤2：创建模型
        print("\n初始化模型...")
        model = Classifier()
        
        # 步骤3：训练
        print("\n开始训练...")
        epochs = 3
        train_start_time = time.time()
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = len(train_loader)
            
            for batch_idx, batch in enumerate(train_loader):
                img_tensors, target_tensors = batch
                loss = model.train_step(img_tensors, target_tensors)
                epoch_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, '
                          f'Batch {batch_idx}/{num_batches}, '
                          f'Loss: {loss.item():.4f}')
            
            avg_loss = epoch_loss / num_batches
            print(f'Epoch {epoch+1}/{epochs} 完成, '
                  f'平均损失: {avg_loss:.4f}')
            
            # 步骤4：评估
            model.eval()
            test_accuracy, test_loss = model.evaluate(test_loader)
            print(f'测试准确率: {test_accuracy:.4f}, '
                  f'测试损失: {test_loss:.4f}')
            print('-' * 50)
        
        train_end_time = time.time()
        print(f"训练总时间: {train_end_time - train_start_time:.2f} 秒")


    if __name__ == "__main__":
        main()

预期输出
~~~~~~~~

运行完整示例时，你应该看到类似以下的输出：

.. code-block:: text

    MNIST 手写数字识别
    加载数据集...
    训练集大小: 60000
    测试集大小: 10000
    
    初始化模型...
    
    开始训练...
    Epoch 1/3, Batch 0/600, Loss: 2.3124
    Epoch 1/3, Batch 100/600, Loss: 0.5231
    Epoch 1/3, Batch 200/600, Loss: 0.3412
    Epoch 1/3, Batch 300/600, Loss: 0.2894
    Epoch 1/3, Batch 400/600, Loss: 0.2543
    Epoch 1/3, Batch 500/600, Loss: 0.1987
    Epoch 1/3 完成, 平均损失: 0.3124
    测试准确率: 0.9123, 测试损失: 0.2987
    --------------------------------------------------
    Epoch 2/3, Batch 0/600, Loss: 0.1876
    Epoch 2/3, Batch 100/600, Loss: 0.1654
    ...
    测试准确率: 0.9456, 测试损失: 0.1876
    --------------------------------------------------
    Epoch 3/3 完成
    测试准确率: 0.9567, 测试损失: 0.1456
    --------------------------------------------------
    训练总时间: 45.23 秒

关键概念总结
------------

Dataset 和 DataLoader
~~~~~~~~~~~~~~~~~~~~~

- **Dataset**: 数据表示的抽象基类，需要实现 ``__len__`` 和 ``__getitem__``
- **DataLoader**: 高效处理批处理、打乱和加载数据
- **Transforms**: 数据增强和归一化的预处理流程

神经网络组件
~~~~~~~~~~~~

- **nn.Module**: 所有神经网络模块的基类
- **nn.Sequential**: 按顺序堆叠层的容器
- **nn.Linear**: 全连接层
- **nn.ReLU**: 引入非线性的激活函数
- **nn.CrossEntropyLoss**: 多分类任务的损失函数

训练过程
~~~~~~~~

- **前向传播**: 计算模型预测
- **损失计算**: 衡量预测与目标之间的差异
- **反向传播**: 通过反向传播计算梯度
- **优化器步骤**: 更新模型参数

评估
~~~~

- **model.eval()**: 将模型设置为评估模式
- **argmax**: 从输出 logits 获取预测的类别
- **准确率**: 正确预测的百分比

Module 类与容器
===============

Riemann 中的所有神经网络模块都继承自 ``nn.Module`` 类，它是构建神经网络的基础。本节将详细介绍 Module 类的核心功能、参数管理以及各种容器类的使用方法。

Module 类核心功能
-----------------

``nn.Module`` 类提供了以下核心功能：

- **参数管理**：自动跟踪和管理可学习参数
- **子模块管理**：支持嵌套子模块，形成层次化结构
- **设备管理**：支持将模块移动到不同设备（CPU/GPU）
- **前向传播**：定义数据通过网络的流动路径
- **状态管理**：支持训练/评估模式切换

Module 类主要方法
-----------------

.. list-table:: Module 类主要方法
   :widths: 20 30 50
   :header-rows: 1

   * - 方法名
     - 描述
     - 使用示例
   * - ``__init__()``
     - 初始化模块，创建核心数据结构
     - ``super(MyModule, self).__init__()``
   * - ``forward(*args, **kwargs)``
     - 定义前向传播逻辑，子类必须实现
     - ``def forward(self, x): return self.layer(x)``
   * - ``__call__(*args, **kwargs)``
     - 模块调用接口，内部调用forward方法
     - ``output = model(input_data)``
   * - ``parameters(recurse=True)``
     - 返回所有参数的迭代器
     - ``for param in model.parameters(): print(param.shape)``
   * - ``named_parameters(prefix='', recurse=True)``
     - 返回带名称的参数迭代器
     - ``for name, param in model.named_parameters(): print(name, param.shape)``
   * - ``buffers(recurse=True)``
     - 返回所有缓冲区的迭代器
     - ``for buffer in model.buffers(): print(buffer.shape)``
   * - ``named_buffers(prefix='', recurse=True)``
     - 返回带名称的缓冲区迭代器
     - ``for name, buffer in model.named_buffers(): print(name, buffer.shape)``
   * - ``children()``
     - 返回直接子模块的迭代器
     - ``for child in model.children(): print(child)``
   * - ``modules()``
     - 返回所有子模块的迭代器（包括自身）
     - ``for module in model.modules(): print(module)``
   * - ``named_modules(prefix='', recurse=True)``
     - 返回带名称的模块迭代器
     - ``for name, module in model.named_modules(): print(name, module)``
   * - ``train(mode=True)``
     - 设置模块为训练模式
     - ``model.train()``
   * - ``eval()``
     - 设置模块为评估模式
     - ``model.eval()``
   * - ``to(device)``
     - 将模块移动到指定设备
     - ``model.to('cuda')``
   * - ``cuda()``
     - 将模块移动到 CUDA 设备
     - ``model.cuda()``
   * - ``cpu()``
     - 将模块移动到 CPU 设备
     - ``model.cpu()``
   * - ``zero_grad(set_to_none=False)``
     - 清除所有参数的梯度
     - ``model.zero_grad()``
   * - ``requires_grad_(requires_grad=True)``
     - 设置参数是否需要梯度
     - ``model.requires_grad_(False)  # 冻结参数``
   * - ``state_dict(destination=None, prefix='', keep_vars=False)``
     - 返回模块状态字典
     - ``state = model.state_dict()``
   * - ``load_state_dict(state_dict)``
     - 将状态字典加载到模块
     - ``model.load_state_dict(state)``
   * - ``register_parameter(name, param)``
     - 向模块注册参数
     - ``self.register_parameter('weight', nn.Parameter(rm.randn(10, 5)))``
   * - ``register_buffer(name, tensor)``
     - 向模块注册缓冲区
     - ``self.register_buffer('running_mean', rm.zeros(10))``
   * - ``add_module(name, module)``
     - 显式添加子模块
     - ``self.add_module('linear', nn.Linear(10, 5))``

创建自定义模块
----------------

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    class MyNetwork(nn.Module):
        def __init__(self):
            super(MyNetwork, self).__init__()
            # 定义子模块
            self.linear1 = nn.Linear(10, 50)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(50, 1)
        
        def forward(self, x):
            # 定义前向传播逻辑
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x
    
    # 创建实例
    model = MyNetwork()
    print(model)

容器类
------

Riemann 提供了多个容器类来组织和管理模块：

Sequential
~~~~~~~~~~

``Sequential`` 容器按顺序执行模块，适用于简单的线性网络结构：

**参数**：

- 接受模块列表或关键字参数

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 方法1：使用模块列表
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # 方法2：使用关键字参数
    model = nn.Sequential(
        linear1=nn.Linear(10, 20),
        relu=nn.ReLU(),
        linear2=nn.Linear(20, 5)
    )
    
    # 前向传播
    x = rm.randn(32, 10)
    output = model(x)
    print(output.shape)  # [32, 5]

ModuleList
~~~~~~~~~~

``ModuleList`` 容器存储模块列表，支持按索引访问，适用于需要动态控制前向传播的场景：

**参数**：

- ``modules``: 模块列表（可选）

**主要方法**：

- ``append(module)``: 添加模块
- ``extend(modules)``: 扩展模块列表
- ``insert(index, module)``: 插入模块

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建模块列表
    layers = nn.ModuleList([
        nn.Linear(10, 20),
        nn.ReLU()
    ])
    
    # 添加更多模块
    layers.append(nn.Linear(20, 10))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(10, 5))
    
    # 前向传播
    x = rm.randn(32, 10)
    for i, layer in enumerate(layers):
        x = layer(x)
        print(f"After layer {i}: {x.shape}")
    
    print(f"Final output shape: {x.shape}")  # [32, 5]

ModuleDict
~~~~~~~~~~

``ModuleDict`` 容器使用字典存储模块，支持按键访问，适用于需要根据条件选择不同模块的场景：

**参数**：

- ``modules``: 模块字典（可选）

**主要方法**：

- ``update(modules)``: 更新模块字典
- ``pop(key)``: 移除并返回指定键的模块

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建模块字典
    layers = nn.ModuleDict({
        'linear1': nn.Linear(10, 20),
        'relu': nn.ReLU(),
        'linear2': nn.Linear(20, 5)
    })
    
    # 添加新模块
    layers.update({'dropout': nn.Dropout(p=0.5)})
    
    # 前向传播
    x = rm.randn(32, 10)
    x = layers['linear1'](x)
    x = layers['relu'](x)
    x = layers['dropout'](x)
    x = layers['linear2'](x)
    
    print(x.shape)  # [32, 5]

ParameterList
~~~~~~~~~~~~~

``ParameterList`` 容器专门用于存储参数列表：

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建参数列表
    params = nn.ParameterList([
        nn.Parameter(rm.randn(10, 20)),
        nn.Parameter(rm.randn(20))
    ])
    
    # 添加更多参数
    params.append(nn.Parameter(rm.randn(20, 5)))
    
    # 索引访问
    weight = params[0]
    bias = params[1]

ParameterDict
~~~~~~~~~~~~~

``ParameterDict`` 容器专门用于存储参数字典：

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建参数字典
    params = nn.ParameterDict({
        'w1': nn.Parameter(rm.randn(10, 20)),
        'b1': nn.Parameter(rm.randn(20)),
        'w2': nn.Parameter(rm.randn(20, 5)),
        'b2': nn.Parameter(rm.randn(5))
    })
    
    # 按键访问
    weight1 = params['w1']
    bias1 = params['b1']

激活函数
========

激活函数是神经网络中的重要组件，引入非线性特性使网络能够学习复杂的函数映射。

激活函数列表
------------

.. list-table:: Riemann 支持的激活函数
   :widths: 15 20 25 25 15
   :header-rows: 1

   * - 函数名
     - 描述
     - 应用场景
     - 参数含义
     - 备注
   * - ``ReLU``
     - 修正线性单元，输出 max(0, x)
     - 大多数深度学习模型的默认选择
     - 无参数
     - 可能产生"死亡神经元"问题
   * - ``LeakyReLU``
     - 带泄露的 ReLU，负区域有小斜率
     - 解决 ReLU 的死亡神经元问题
     - ``negative_slope``: 负区域斜率，默认 0.01
     - 计算成本略高于 ReLU
   * - ``Sigmoid``
     - S 型激活函数，输出 (0, 1)
     - 二分类任务的输出层
     - 无参数
     - 存在梯度消失问题
   * - ``Tanh``
     - 双曲正切函数，输出 (-1, 1)
     - RNN 等序列模型
     - 无参数
     - 仍有梯度消失问题，但比 Sigmoid 轻
   * - ``Softmax``
     - 归一化指数函数，输出概率分布
     - 多分类任务的输出层
     - ``dim``: 计算维度，默认 -1
     - 通常与交叉熵损失配合使用
   * - ``GELU``
     - 高斯误差线性单元
     - Transformer 模型的默认选择
     - 无参数
     - 计算成本较高

损失函数
========

损失函数用于衡量模型预测与真实目标值之间的差异，是模型训练的核心组件。

损失函数列表
------------

.. list-table:: Riemann 支持的损失函数
   :widths: 15 20 25 25 15
   :header-rows: 1

   * - 函数名
     - 描述
     - 应用场景
     - 参数含义
     - 备注
   * - ``MSELoss``
     - 均方误差损失
     - 回归任务
     - ``reduction``: 聚合方式，默认 'mean'
     - 对异常值敏感
   * - ``L1Loss``
     - L1 损失（绝对误差）
     - 对异常值不敏感的回归任务
     - ``reduction``: 聚合方式，默认 'mean'
     - 在原点处梯度不连续
   * - ``CrossEntropyLoss``
     - 交叉熵损失，结合 log_softmax 和 nll_loss
     - 多分类任务
     - ``weight``: 类别权重
       ``ignore_index``: 忽略的目标值
       ``reduction``: 聚合方式，默认 'mean'
     - 输入为原始 logits，不需要 softmax
   * - ``BCEWithLogitsLoss``
     - 带 logits 的二元交叉熵损失
     - 二分类任务
     - ``weight``: 样本权重
       ``pos_weight``: 正类权重
     - 输入为原始 logits，不需要 sigmoid
   * - ``HuberLoss``
     - Huber 损失，对异常值鲁棒
     - 对异常值敏感的回归任务
     - ``delta``: 阈值，默认 1.0
     - 计算成本适中

基础网络层
==========

线性层 (Linear)
---------------

线性层（又称全连接层）对输入数据进行仿射变换：

**参数**：
- ``in_features``: 输入特征维度
- ``out_features``: 输出特征维度
- ``bias``: 是否使用偏置，默认 True

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建线性层
    linear = nn.Linear(in_features=20, out_features=10)
    
    # 前向传播
    x = rm.randn(32, 20)
    output = linear(x)
    print(output.shape)  # [32, 10]

Dropout 层
----------

Dropout 层通过随机停用神经元来防止过拟合：

**参数**：
- ``p``: Dropout 概率，默认 0.5

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建 dropout 层
    dropout = nn.Dropout(p=0.5)
    
    # 前向传播（训练模式）
    x = rm.randn(4, 16)
    dropout.train()
    output_train = dropout(x)
    
    # 前向传播（评估模式）
    dropout.eval()
    output_eval = dropout(x)

Flatten 层
----------

Flatten 层将输入张量展平：

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    flatten = nn.Flatten()
    
    # 将 (batch, 1, 28, 28) 展平为 (batch, 784)
    x = rm.randn(32, 1, 28, 28)
    output = flatten(x)
    print(output.shape)  # [32, 784]
