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

**数据根目录管理**

为了方便管理数据存储位置，Riemann 提供了 ``get_data_root()`` 工具函数，用于获取项目的数据根目录：

.. code-block:: python

    from riemann.utils import get_data_root
    
    # 获取数据根目录路径
    data_root = get_data_root()
    print(f"数据根目录: {data_root}")
    # 输出示例: D:\\code\\Riemann\\data

该函数会自动定位到项目根目录下的 ``data`` 文件夹，避免在不同环境中手动指定路径的问题。

**加载数据集**

.. code-block:: python

    from riemann.utils import get_data_root
    
    # 加载训练集和测试集
    train_dataset = MNIST(
        root=get_data_root(),  # 使用工具函数获取数据根目录
        train=True,            # True 表示训练集，False 表示测试集
        transform=transform    # 要应用的数据变换
    )
    
    test_dataset = MNIST(
        root=get_data_root(),
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
- **钩子管理**：支持注册前向/反向传播钩子，用于调试、特征提取和梯度修改

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
   * - ``register_forward_pre_hook(hook)``
     - 注册前向传播前钩子
     - ``handle = model.register_forward_pre_hook(my_hook)``
   * - ``register_forward_hook(hook)``
     - 注册前向传播后钩子
     - ``handle = model.register_forward_hook(my_hook)``
   * - ``register_full_backward_pre_hook(hook)``
     - 注册反向传播前钩子
     - ``handle = model.register_full_backward_pre_hook(my_hook)``
   * - ``register_full_backward_hook(hook)``
     - 注册反向传播后钩子
     - ``handle = model.register_full_backward_hook(my_hook)``
   * - ``apply(fn)``
     - 递归地将函数应用到所有子模块
     - ``model.apply(init_weights)``
   * - ``get_parameter(target)``
     - 获取指定名称的参数
     - ``param = model.get_parameter('layer1.weight')``
   * - ``get_submodule(target)``
     - 获取指定名称的子模块
     - ``module = model.get_submodule('layer1.conv1')``
   * - ``get_buffer(target)``
     - 获取指定名称的缓冲区
     - ``buffer = model.get_buffer('bn1.running_mean')``
   * - ``has_parameter(target)``
     - 检查参数是否存在
     - ``if model.has_parameter('weight'): ...``
   * - ``has_buffer(target)``
     - 检查缓冲区是否存在
     - ``if model.has_buffer('running_mean'): ...``
   * - ``set_parameter(name, param)``
     - 设置指定名称的参数
     - ``model.set_parameter('weight', new_param)``
   * - ``set_buffer(name, tensor)``
     - 设置指定名称的缓冲区
     - ``model.set_buffer('running_mean', new_tensor)``
   * - ``delete_parameter(target)``
     - 删除指定名称的参数
     - ``model.delete_parameter('old_weight')``
   * - ``delete_buffer(target)``
     - 删除指定名称的缓冲区
     - ``model.delete_buffer('old_buffer')``
   * - ``copy()``
     - 创建模块的浅拷贝
     - ``new_model = model.copy()``
   * - ``deepcopy()``
     - 创建模块的深拷贝
     - ``new_model = model.deepcopy()``

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

线性层（又称全连接层）对输入数据进行仿射变换，是神经网络中最基础的层之一。

**用途**：

- 实现输入到输出的线性变换：``output = input @ weight.T + bias``
- 常用于特征变换、分类器的最后一层、以及网络中的维度转换
- 是构建多层感知机（MLP）的基础组件

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

Dropout 层通过随机停用神经元来防止过拟合，是一种常用的正则化技术。

**用途**：

- 防止神经网络过拟合，提高模型泛化能力
- 在训练时随机将部分神经元输出置为零，强迫网络学习更鲁棒的特征表示
- 常用于全连接层之后，特别是在深层网络中

**参数**：

- ``p``: Dropout 概率，默认 0.5，表示每个神经元被丢弃的概率

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

Dropout2d 层
------------

Dropout2d 层以通道为单位随机丢弃整个特征图，适用于卷积神经网络。

**用途**：

- 专门用于 2D 卷积特征图（形状为 ``(N, C, H, W)``）的正则化
- 以通道为单位随机丢弃，而非单个像素，保持特征图的空间相关性
- 常用于卷积层之后，防止卷积网络过拟合

**参数**：

- ``p``: Dropout 概率，默认 0.5

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建 Dropout2d 层
    dropout2d = nn.Dropout2d(p=0.5)
    
    # 前向传播（输入形状为 [N, C, H, W]）
    x = rm.randn(4, 16, 32, 32)
    dropout2d.train()
    output = dropout2d(x)
    print(output.shape)  # [4, 16, 32, 32]

Dropout3d 层
------------

Dropout3d 层以通道为单位随机丢弃整个 3D 特征图，适用于 3D 卷积神经网络。

**用途**：

- 专门用于 3D 卷积特征图（形状为 ``(N, C, D, H, W)``）的正则化
- 以通道为单位随机丢弃整个 3D 特征体
- 常用于视频处理、3D 医学图像等 3D 卷积网络中

**参数**：

- ``p``: Dropout 概率，默认 0.5

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建 Dropout3d 层
    dropout3d = nn.Dropout3d(p=0.5)
    
    # 前向传播（输入形状为 [N, C, D, H, W]）
    x = rm.randn(4, 16, 8, 32, 32)
    dropout3d.train()
    output = dropout3d(x)
    print(output.shape)  # [4, 16, 8, 32, 32]

Flatten 层
----------

Flatten 层将输入张量在指定维度范围内进行展平操作。

**用途**：

- 将多维张量展平为一维或低维张量，常用于连接卷积层和全连接层
- 保留批次维度，将空间维度和通道维度合并为特征向量
- 是 CNN 架构中连接卷积部分和全连接部分的桥梁

**参数**：

- ``start_dim``: 开始展平的维度，默认 1
- ``end_dim``: 结束展平的维度，默认 -1

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    flatten = nn.Flatten()
    
    # 将 (batch, 1, 28, 28) 展平为 (batch, 784)
    x = rm.randn(32, 1, 28, 28)
    output = flatten(x)
    print(output.shape)  # [32, 784]

BatchNorm1d 层
--------------

一维批量归一化层，对 2D 或 3D 输入的通道维度进行归一化。

**用途**：

- 加速神经网络训练收敛，允许使用更大的学习率
- 减少对初始化的敏感性，提高训练稳定性
- 具有一定的正则化效果，减少对 Dropout 的依赖
- 常用于全连接层之后或 1D 卷积层之后

**参数**：

- ``num_features``: 特征数量（通道数 C）
- ``eps``: 数值稳定性的小常数，默认 1e-5
- ``momentum``: 运行时统计量的动量，默认 0.1
- ``affine``: 是否使用可学习的仿射参数，默认 True
- ``track_running_stats``: 是否跟踪运行时均值和方差，默认 True

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建 BatchNorm1d 层
    bn = nn.BatchNorm1d(num_features=100)
    
    # 2D 输入 (N, C)
    x = rm.randn(20, 100)
    output = bn(x)
    print(output.shape)  # [20, 100]
    
    # 3D 输入 (N, C, L)
    x = rm.randn(20, 100, 35)
    output = bn(x)
    print(output.shape)  # [20, 100, 35]

BatchNorm2d 层
--------------

二维批量归一化层，对 4D 输入 ``(N, C, H, W)`` 的通道维度进行归一化。

**用途**：

- 专门用于 2D 卷积神经网络，对每个通道的特征图进行归一化
- 加速 CNN 训练，提高模型泛化能力
- 是构建现代 CNN（如 ResNet、DenseNet）的关键组件
- 通常放置在卷积层之后、激活函数之前

**参数**：

- ``num_features``: 特征数量（通道数 C）
- ``eps``: 数值稳定性的小常数，默认 1e-5
- ``momentum``: 运行时统计量的动量，默认 0.1
- ``affine``: 是否使用可学习的仿射参数，默认 True
- ``track_running_stats``: 是否跟踪运行时均值和方差，默认 True

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建 BatchNorm2d 层
    bn = nn.BatchNorm2d(num_features=64)
    
    # 4D 输入 (N, C, H, W)
    x = rm.randn(16, 64, 32, 32)
    output = bn(x)
    print(output.shape)  # [16, 64, 32, 32]

BatchNorm3d 层
--------------

三维批量归一化层，对 5D 输入 ``(N, C, D, H, W)`` 的通道维度进行归一化。

**用途**：

- 专门用于 3D 卷积神经网络，如视频处理、3D 医学图像分析
- 对每个通道的 3D 特征体进行归一化
- 是 3D CNN 架构（如 C3D、I3D）的重要组成部分

**参数**：

- ``num_features``: 特征数量（通道数 C）
- ``eps``: 数值稳定性的小常数，默认 1e-5
- ``momentum``: 运行时统计量的动量，默认 0.1
- ``affine``: 是否使用可学习的仿射参数，默认 True
- ``track_running_stats``: 是否跟踪运行时均值和方差，默认 True

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建 BatchNorm3d 层
    bn = nn.BatchNorm3d(num_features=32)
    
    # 5D 输入 (N, C, D, H, W)
    x = rm.randn(8, 32, 4, 16, 16)
    output = bn(x)
    print(output.shape)  # [8, 32, 4, 16, 16]

LayerNorm 层
------------

层归一化层，对单个样本的所有特征进行归一化。

**用途**：

- 对单个样本的特征进行归一化，不依赖批次统计量
- 适用于批次大小为 1 或动态变化的场景
- 是 Transformer 模型的核心组件，用于替代 BatchNorm
- 在自然语言处理任务中广泛使用

**参数**：

- ``normalized_shape``: 需要归一化的维度，可以是整数或元组
- ``eps``: 数值稳定性的小常数，默认 1e-5
- ``affine``: 是否使用可学习的仿射参数，默认 True

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建 LayerNorm 层
    ln = nn.LayerNorm(normalized_shape=128)
    
    # 输入可以是任意形状，最后维度需要匹配 normalized_shape
    x = rm.randn(20, 128)
    output = ln(x)
    print(output.shape)  # [20, 128]
    
    # 多维输入
    x = rm.randn(20, 10, 128)
    output = ln(x)
    print(output.shape)  # [20, 10, 128]

Embedding 层
------------

嵌入层，将整数索引转换为固定大小的密集向量表示。

**用途**：

- 将离散的整数索引（如词索引）映射为连续的向量表示
- 是处理分类特征和序列数据（如文本、用户ID）的基础组件
- 在 NLP 任务中作为词嵌入层使用
- 支持填充索引（padding_idx）不参与梯度计算

**参数**：

- ``num_embeddings``: 嵌入向量的数量（词典大小）
- ``embedding_dim``: 每个嵌入向量的维度
- ``padding_idx``: 填充索引，该索引的嵌入向量不参与梯度计算，默认 None
- ``max_norm``: 嵌入向量的最大范数，超过则重归一化，默认 None
- ``norm_type``: 计算范数的 p 值，默认 2（L2 范数）
- ``scale_grad_by_freq``: 是否按频率缩放梯度，默认 False

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建 Embedding 层，词典大小为 10000，嵌入维度为 128
    embedding = nn.Embedding(num_embeddings=10000, embedding_dim=128)
    
    # 输入是整数索引
    input_indices = rm.tensor([1, 5, 10, 100])
    output = embedding(input_indices)
    print(output.shape)  # [4, 128]
    
    # 使用 padding_idx
    embedding_with_pad = nn.Embedding(10000, 128, padding_idx=0)
    input_with_pad = rm.tensor([0, 1, 2, 0])  # 0 是填充索引
    output = embedding_with_pad(input_with_pad)

模块钩子管理
============

Riemann 提供了强大的模块钩子机制，允许用户在模块的前向传播和反向传播过程中插入自定义逻辑。钩子机制是调试、监控和修改网络行为的强大工具。

钩子类型概述
------------

Riemann 支持四种类型的模块钩子，分别在前向传播和反向传播的不同阶段执行：

.. list-table::
   :header-rows: 1
   :widths: 20 25 35 30

   * - 钩子类型
     - 注册方法
     - 执行时机
     - 可修改的值
   * - 前向预处理钩子
     - ``register_forward_pre_hook``
     - 在 ``forward`` 方法执行之前调用
     - 模块输入（``input``）
   * - 前向钩子
     - ``register_forward_hook``
     - 在 ``forward`` 方法执行之后调用
     - 模块输出（``output``）
   * - 反向预处理钩子
     - ``register_full_backward_pre_hook``
     - 当模块所有需要梯度的输出都收到梯度时调用（不参与损失计算的输出不需要等待）
     - 输出梯度（``grad_output``）
   * - 反向钩子
     - ``register_full_backward_hook``
     - 当模块所有需要梯度的输入都收到梯度时调用
     - 输入梯度（``grad_input``）

钩子执行顺序
~~~~~~~~~~~~

前向传播阶段的钩子执行顺序：

.. code-block:: text

    register_forward_pre_hook → forward → register_forward_hook

反向传播阶段的钩子执行顺序：

.. code-block:: text

    register_full_backward_pre_hook → (计算 grad_input) → register_full_backward_hook

前向预处理钩子 (register_forward_pre_hook)
------------------------------------------

**用途**：

- 在模块前向计算之前修改或检查输入数据
- 实现输入预处理、数据验证或调试信息打印
- 常用于动态调整输入范围、添加噪声或记录中间状态

**钩子函数签名**：

.. code-block:: python

    hook(module, input) -> None or modified input

**参数说明**：

- ``module``：当前被调用的模块实例
- ``input``：包含所有输入张量的元组（即使只有一个输入也是元组形式）

**返回值**：

- ``None``：表示不修改输入，使用原始输入继续执行
- ``Tensor`` 或 ``tuple``：返回修改后的输入，将替换原始输入传递给 ``forward``

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # 定义前向预处理钩子：打印输入信息
    def print_input_hook(module, input):
        print(f"模块 {module._get_name()} 的输入形状: {input[0].shape}")
        return None  # 不修改输入

    # 定义前向预处理钩子：修改输入
    def double_input_hook(module, input):
        # 将输入乘以2
        return (input[0] * 2,)

    # 创建线性层并注册钩子
    linear = nn.Linear(10, 5)
    handle1 = linear.register_forward_pre_hook(print_input_hook)
    handle2 = linear.register_forward_pre_hook(double_input_hook)

    # 前向传播
    x = rm.ones(2, 10)
    output = linear(x)  # 实际使用的是 x * 2

    # 移除钩子
    handle1.remove()
    handle2.remove()

前向钩子 (register_forward_hook)
--------------------------------

**用途**：

- 在模块前向计算之后修改或检查输出数据
- 实现特征提取、输出监控和调试
- 常用于记录中间层特征、分析激活分布

**钩子函数签名**：

.. code-block:: python

    hook(module, input, output) -> None or modified output

**参数说明**：

- ``module``：当前被调用的模块实例
- ``input``：传递给 ``forward`` 的所有输入张量的元组
  
  - **始终是元组**：即使是单输入模块，``input`` 也是包含一个元素的元组：``(input_tensor,)``
  - 多输入模块：``(input1, input2, ...)``
  - **注意**：如果前向预处理钩子修改了输入，这里将是修改后的版本，而非原始输入
  
- ``output``：``forward`` 方法的返回值
  
  - 单输出模块：单个张量
  - 多输出模块：张量元组 ``(output1, output2, ...)``

**返回值**：

- ``None``：表示不修改输出，使用原始输出作为模块返回值
- ``Tensor`` 或 ``tuple``：返回修改后的输出，将替换原始输出
  
  - 单输出模块返回张量
  - 多输出模块返回相同结构的元组

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # 定义前向钩子：特征提取器
    class FeatureExtractor:
        def __init__(self):
            self.features = []
        
        def hook(self, module, input, output):
            self.features.append(output.clone())
            return None

    # 创建模型并注册特征提取钩子
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

    extractor = FeatureExtractor()
    handle = model[0].register_forward_hook(extractor.hook)

    # 前向传播
    x = rm.randn(4, 784)
    output = model(x)

    # 查看提取的特征
    print(f"第一层输出形状: {extractor.features[0].shape}")

    # 移除钩子
    handle.remove()

反向预处理钩子 (register_full_backward_pre_hook)
------------------------------------------------

**用途**：

- 在反向传播开始时修改或检查输出梯度（``grad_output``）
- 实现梯度裁剪、梯度缩放或梯度监控
- 常用于防止梯度爆炸、调整梯度流

**钩子函数签名**：

.. code-block:: python

    hook(module, grad_output) -> None or modified grad_output

**参数说明**：

- ``module``：当前反向传播的模块实例
- ``grad_output``：包含所有输出梯度的元组
  
  - 单输出模块：``(grad_output_tensor,)``
  - 多输出模块：``(grad_output1, grad_output2, ...)``
  - 对于不需要梯度的输出，对应位置为 ``None``

**返回值**：

- ``None``：表示不修改梯度，使用原始 ``grad_output`` 继续计算
- ``tuple``：返回修改后的 ``grad_output``，将用于后续梯度计算
  
  **重要**：如果只想修改部分梯度，返回的元组必须包含**所有**输出梯度。对于不想修改的位置，返回原始梯度值；如果返回 ``None``，该位置梯度将被**清零**（设为0）

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # 定义反向预处理钩子：梯度裁剪
    def clip_grad_hook(module, grad_output):
        # 裁剪梯度，防止梯度爆炸
        clipped = tuple(
            g.clip(-1, 1) if g is not None else None 
            for g in grad_output
        )
        return clipped

    # 定义反向预处理钩子：打印梯度信息
    def print_grad_hook(module, grad_output):
        print(f"输出梯度形状: {grad_output[0].shape}")
        print(f"输出梯度值范围: [{grad_output[0].min()}, {grad_output[0].max()}]")
        return None

    # 创建线性层并注册钩子
    linear = nn.Linear(10, 5)
    handle = linear.register_full_backward_pre_hook(clip_grad_hook)

    # 前向传播和反向传播
    x = rm.randn(2, 10)
    x.requires_grad = True
    output = linear(x)
    output.sum().backward()  # 梯度将被裁剪到 [-1, 1] 范围

    # 移除钩子
    handle.remove()

反向钩子 (register_full_backward_hook)
--------------------------------------

**用途**：

- 在反向传播结束时修改或检查输入梯度（``grad_input``）
- 实现梯度监控、调试和可视化
- 常用于分析梯度流向、检测梯度消失或爆炸

**钩子函数签名**：

.. code-block:: python

    hook(module, grad_input, grad_output) -> None or modified grad_input

**参数说明**：

- ``module``：当前反向传播的模块实例
- ``grad_input``：包含所有输入梯度的元组
  
  - 单输入模块：``(grad_input_tensor,)``
  - 多输入模块：``(grad_input1, grad_input2, ...)``
  - 对于不需要梯度的输入，对应位置为 ``None``

- ``grad_output``：包含所有输出梯度的元组
  
  - **注意**：如果反向预处理钩子修改了梯度，这里将是修改后的版本

**返回值**：

- ``None``：表示不修改梯度，使用原始 ``grad_input`` 继续传播
- ``tuple``：返回修改后的 ``grad_input``，将替换原始梯度传播给前一层
  
  **重要**：如果只想修改部分梯度，返回的元组必须包含**所有**输入梯度。对于不想修改的位置，返回原始梯度值；如果返回 ``None``，该位置梯度将被**清零**（设为0）
  
  .. note::
     Riemann 的这一行为与 PyTorch 不同。在 PyTorch 中，返回 ``None`` 会使该位置梯度保持为 ``None``。
     Riemann 选择清零的方式是为了：
     
     1. **语义一致性**：与反向预处理钩子行为保持一致（返回 ``None`` 均表示清零）
     2. **实用性**：清零是阻止梯度传播的直观方式，而 ``None`` 需要额外处理
     3. **安全性**：梯度为 0 是合法数值，不会导致后续计算错误

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # 定义反向钩子：梯度监控器
    class GradientMonitor:
        def __init__(self):
            self.gradients = []
        
        def hook(self, module, grad_input, grad_output):
            self.gradients.append({
                'module': module._get_name(),
                'grad_input': [g.clone() if g is not None else None for g in grad_input],
                'grad_output': [g.clone() if g is not None else None for g in grad_output]
            })
            return None

    # 创建模型并注册梯度监控钩子
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

    monitor = GradientMonitor()
    for layer in model:
        layer.register_full_backward_hook(monitor.hook)

    # 前向传播和反向传播
    x = rm.randn(4, 784)
    x.requires_grad = True
    output = model(x)
    output.sum().backward()

    # 查看记录的梯度信息
    for grad_info in monitor.gradients:
        print(f"模块: {grad_info['module']}")
        print(f"输入梯度形状: {[g.shape if g is not None else None for g in grad_info['grad_input']]}")

钩子注册与移除
--------------

**注册钩子**：

所有钩子注册方法都返回一个 ``RemovableHandle`` 对象，可用于后续移除钩子：

.. code-block:: python

    # 注册钩子并获取句柄
    handle = module.register_forward_hook(hook_function)
    
    # 使用句柄移除钩子
    handle.remove()

**使用上下文管理器**：

``RemovableHandle`` 支持上下文管理器协议，可以使用 ``with`` 语句自动管理钩子的生命周期：

.. code-block:: python

    with module.register_forward_hook(hook_function) as handle:
        # 在此范围内钩子有效
        output = module(input)
        # 退出 with 块时自动移除钩子

**多钩子管理**：

一个模块可以注册多个同类型钩子，它们按注册顺序依次执行：

.. code-block:: python

    def hook1(module, input):
        print("钩子1")
        return None
    
    def hook2(module, input):
        print("钩子2")
        return None
    
    module.register_forward_pre_hook(hook1)
    module.register_forward_pre_hook(hook2)
    
    # 执行顺序: hook1 -> hook2

典型应用场景
------------

**1. 特征可视化**

特征可视化是深度学习中常用的技术，用于理解神经网络在不同层学习到的模式。通过在卷积层上注册前向钩子，可以捕获并可视化中间特征图。

**使用场景**：

- 可视化不同卷积滤波器检测到的特征（边缘、纹理、形状）
- 通过检查中间表示来调试模型行为
- 为研究或演示目的创建特征图

**示例**：捕获并可视化CNN的特征图（使用真实MNIST数据）

.. code-block:: python

    import riemann.nn as nn
    from riemann.vision.datasets import EasyMNIST
    from riemann.utils import get_data_root
    import matplotlib.pyplot as plt

    # 设置中文字体（如果需要显示中文标题）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 加载MNIST数据集
    print("加载MNIST数据集...")
    train_dataset = EasyMNIST(root=get_data_root(), train=True, onehot_label=False)

    # 获取一个样本（手写数字图像）
    sample_data, sample_label = train_dataset[0]
    print(f"样本标签: {int(sample_label)}")

    # 将展平的数据重塑为28x28图像
    sample_image = sample_data.reshape(28, 28)

    # 创建一个简单的CNN用于演示
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.fc = nn.Linear(32 * 28 * 28, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    # 用于存储激活值的字典
    activations = {}

    def get_activation(name):
        """创建保存激活值的钩子函数"""
        def hook(module, input, output):
            # 使用detach避免保存计算图
            activations[name] = output.detach()
        return hook

    # 创建模型并注册钩子
    model = SimpleCNN()
    model.conv1.register_forward_hook(get_activation('conv1'))
    model.conv2.register_forward_hook(get_activation('conv2'))

    # 使用MNIST样本进行前向传播
    # 将样本重塑为 [batch_size, channels, height, width]
    input_image = sample_image.unsqueeze(0).unsqueeze(0)  # [1, 1, 28, 28]
    output = model(input_image)

    # 现在 activations['conv1'] 包含 conv1 层的特征图
    # 形状: [1, 16, 28, 28] - 16个28x28的特征图
    print(f"Conv1 激活值形状: {activations['conv1'].shape}")
    print(f"模型预测: {output.argmax(dim=1).item()}")

    # 可视化 conv1 的前8个特征图
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(activations['conv1'][0, i].numpy(), cmap='viridis')
        ax.set_title(f'Filter {i}')
        ax.axis('off')
    plt.suptitle('Conv1 Layer Feature Maps', fontsize=14)
    plt.tight_layout()
    plt.show()

**2. 梯度检查**

梯度检查对于调试训练问题至关重要。无效的梯度（NaN或Inf值）可能导致训练静默失败或产生意外结果。通过使用反向钩子，可以在训练期间实时监控梯度。

**使用场景**：

- 及早检测梯度爆炸或梯度消失
- 识别哪些层产生了无效梯度
- 出现问题时自动停止训练或调整学习率

**示例**：全面的梯度监控与自动训练停止

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    class GradientChecker:
        """全面监控各种问题的梯度检查器"""
        
        def __init__(self, threshold=1e3):
            self.threshold = threshold  # 梯度爆炸阈值
            self.has_nan_inf = False
            self.layer_stats = {}
        
        def hook(self, module, grad_input, grad_output):
            module_name = module._get_name()
            
            # 检查 grad_output 中的 NaN 或 Inf
            for i, g in enumerate(grad_output):
                if g is not None:
                    if rm.isnan(g).any():
                        print(f"错误: 在 {module_name} 的 grad_output[{i}] 中检测到 NaN")
                        self.has_nan_inf = True
                    if rm.isinf(g).any():
                        print(f"错误: 在 {module_name} 的 grad_output[{i}] 中检测到 Inf")
                        self.has_nan_inf = True
                    
                    # 检查梯度爆炸
                    grad_norm = g.norm().item()
                    if grad_norm > self.threshold:
                        print(f"警告: {module_name} 发生梯度爆炸: 范数={grad_norm:.2f}")
            
            # 同时检查 grad_input
            for i, g in enumerate(grad_input):
                if g is not None:
                    if rm.isnan(g).any() or rm.isinf(g).any():
                        print(f"错误: {module_name} 的 grad_input[{i}] 包含无效梯度")
                        self.has_nan_inf = True
            
            # 存储统计信息
            self.layer_stats[module_name] = {
                'grad_output_norms': [g.norm().item() if g is not None else 0 for g in grad_output],
                'grad_input_norms': [g.norm().item() if g is not None else 0 for g in grad_input]
            }
            
            return None  # 不修改梯度，仅监控

    # 在训练中使用
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

    checker = GradientChecker(threshold=100.0)
    
    # 在所有层上注册钩子
    for layer in model:
        layer.register_full_backward_hook(checker.hook)

    # 带有梯度检查的训练循环
    for epoch in range(10):
        # ... 前向传播 ...
        # loss = criterion(output, target)
        # loss.backward()
        
        # 在优化器步骤前检查梯度是否有效
        if checker.has_nan_inf:
            print(f"第 {epoch} 轮: 由于无效梯度而停止训练")
            break
        
        # optimizer.step()

**3. 权重统计监控**

监控训练期间的权重统计信息有助于理解网络的学习情况。权重分布的突然变化可能表明存在初始化不良、学习率问题或过拟合等问题。

**使用场景**：

- 跟踪训练轮次间的权重分布变化
- 检测死亡神经元（权重卡在零值）
- 识别潜在的过拟合（权重变得过大）
- 验证正确的权重初始化

**示例**：全面的权重和激活监控

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    class TrainingMonitor:
        """监控训练期间的权重、偏置和激活值"""
        
        def __init__(self):
            self.history = []
        
        def forward_hook(self, module, input, output):
            """监控前向传播统计信息"""
            stats = {
                'module': module._get_name(),
                'input_mean': input[0].mean().item() if input[0] is not None else 0,
                'output_mean': output.mean().item(),
                'output_std': output.std().item()
            }
            
            # 监控权重（如果可用）
            if hasattr(module, 'weight') and module.weight is not None:
                w = module.weight.data
                stats.update({
                    'weight_mean': w.mean().item(),
                    'weight_std': w.std().item(),
                    'weight_min': w.min().item(),
                    'weight_max': w.max().item(),
                    'dead_neurons': (w.abs() < 1e-6).sum().item()  # 接近零的权重
                })
            
            # 监控偏置（如果可用）
            if hasattr(module, 'bias') and module.bias is not None:
                b = module.bias.data
                stats.update({
                    'bias_mean': b.mean().item(),
                    'bias_std': b.std().item()
                })
            
            self.history.append(stats)
            
            # 对潜在问题打印警告
            if stats.get('weight_std', 0) > 10:
                print(f"警告: {stats['module']} 的权重标准差过高: {stats['weight_std']:.2f}")
            if stats.get('dead_neurons', 0) > 0:
                print(f"信息: {stats['module']} 有 {stats['dead_neurons']} 个死亡神经元")
        
        def print_summary(self):
            """打印监控统计信息摘要"""
            print("\n=== 训练监控摘要 ===")
            for stats in self.history[-5:]:  # 显示最近5条记录
                print(f"\n{stats['module']}:")
                if 'weight_mean' in stats:
                    print(f"  权重: 均值={stats['weight_mean']:.4f}, 标准差={stats['weight_std']:.4f}")
                if 'bias_mean' in stats:
                    print(f"  偏置: 均值={stats['bias_mean']:.4f}, 标准差={stats['bias_std']:.4f}")
                print(f"  激活值: 均值={stats['output_mean']:.4f}, 标准差={stats['output_std']:.4f}")

    # 使用示例
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

    monitor = TrainingMonitor()
    
    # 在所有线性层上注册前向钩子
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            layer.register_forward_hook(monitor.forward_hook)

    # 在训练期间，统计信息会自动收集
    # 训练结束后，查看摘要
    # monitor.print_summary()

注意事项
--------

1. **钩子返回值**

   如果钩子不需要修改数据，应该返回 ``None`` 以避免不必要的副作用。当需要修改时，返回与输入结构相同的张量或元组。

2. **多输入/多输出模块处理**

   对于具有多个输入或输出的模块，钩子会接收包含所有输入/输出的元组：

   - **多输入模块**：``input`` 和 ``grad_input`` 是包含所有输入张量/梯度的元组
   - **多输出模块**：``output`` 和 ``grad_output`` 是包含所有输出张量/梯度的元组
   - **重要**：
     
     - 对于多输出模块，反向预处理钩子仅在所有输出梯度（requires_grad=True且参与损失函数计算的输出）都就绪时才被调用
     - 对于多输入模块，反向钩子仅在所有输入梯度（requires_grad=True的输入）都就绪时才被调用
     
     这确保钩子接收到完整的梯度信息
   
   修改梯度时，即使只修改部分元素，也要始终返回结构完整的元组。

3. **多钩子调用链**

   一个模块可以注册多个同类型钩子，它们按注册顺序形成调用链：

   - **前向预处理钩子链**：前一个钩子的返回值作为下一个钩子的输入
     
     - 如果钩子返回 ``None``：使用原始输入传递给下一个钩子
     - 如果钩子返回非 ``None``：使用该返回值作为下一个钩子的输入
     - 最后一个钩子的输出决定最终传递给 ``forward`` 的输入
   
   - **前向钩子链**：前一个钩子的返回值作为下一个钩子的输出
     
     - 如果钩子返回 ``None``：使用原始输出传递给下一个钩子
     - 如果钩子返回非 ``None``：使用该返回值作为下一个钩子的输出
     - 最后一个钩子的输出决定模块的最终返回值
   
   - **反向预处理钩子链**：可以修改 ``grad_output``
     
     - 钩子接收的 ``grad_output`` 可能是前一个钩子修改后的版本
     - 如果钩子返回 ``None``：使用当前 ``grad_output`` 计算 ``grad_input``
     - 如果钩子返回非 ``None``：使用该返回值替换 ``grad_output`` 并计算 ``grad_input``
   
   - **反向钩子链**：可以修改 ``grad_input``
     
     - 钩子接收的 ``grad_input`` 是计算完成的输入梯度
     - 钩子接收的 ``grad_output`` 可能是反向预处理钩子修改后的版本
     - 如果钩子返回 ``None``：使用原始 ``grad_input`` 传播给前一层
     - 如果钩子返回非 ``None``：使用该返回值替换 ``grad_input`` 并传播给前一层

   .. code-block:: python

       # 多输出模块示例
       class MultiOutputModule(nn.Module):
           def forward(self, x):
               return x * 2, x * 3  # 两个输出
       
       module = MultiOutputModule()
       
       def grad_hook(module, grad_input, grad_output):
           # grad_output 包含两个输出的梯度
           # 只有当两个输出梯度都就绪时才调用此钩子
           print(f"输出1梯度形状: {grad_output[0].shape}")
           print(f"输出2梯度形状: {grad_output[1].shape}")
           return None
       
       module.register_full_backward_hook(grad_hook)

3. **梯度计算流程**

   理解梯度计算流程有助于正确使用反向钩子：

   - 反向预处理钩子（``register_full_backward_pre_hook``）
     
     在 ``grad_input`` 计算之前调用。修改 ``grad_output`` 会影响模块输入的梯度计算方式
     
   - 反向钩子（``register_full_backward_hook``）
     
     在 ``grad_input`` 计算之后调用。修改 ``grad_input`` 会影响传播到前一层的梯度

   .. code-block:: text

       反向传播流程：
       
       1. 输出梯度从上游到达
       2. 调用 register_full_backward_pre_hook（可修改 grad_output）
       3. 使用（可能已修改的）grad_output 计算 grad_input
       4. 调用 register_full_backward_hook（可修改 grad_input）
       5. 将修改后的 grad_input 传播到前一层

4. **钩子执行条件**

   反向钩子有特定的执行条件，以确保有意义的梯度修改：

   - 至少有一个模块输入需要梯度，**或者**
   - 模块必须有需要梯度的参数
   
   如果两个条件都不满足，反向钩子不会被调用，因为没有梯度可以修改。

5. **性能考虑**

   - 钩子会增加额外的函数调用开销。对于生产环境推理，应移除所有调试和监控钩子
   - 在钩子中避免执行耗时操作，特别是在训练循环中
   - 当在同一模块上注册多个钩子时，它们会顺序执行，开销会累积

6. **内存管理**

   - 在钩子中保存张量引用时要注意内存泄漏问题。保存的张量会保留计算图
   - 存储张量供后续分析时，始终使用 ``.clone()`` 或 ``.detach()`` 创建副本
   - 反向传播完成后，缓存的梯度会自动清理

7. **与计算图的交互**

   在钩子中修改梯度时，要注意计算图：

   - 修改后的梯度会流入后续计算
   - 对于梯度裁剪，确保操作不会破坏梯度流
   - 对于梯度监控，使用 ``.detach()`` 避免影响计算图

   .. code-block:: python

       # 安全的梯度裁剪（保持梯度流）
       def safe_clip_hook(module, grad_output):
           clipped = tuple(
               g.clip(-1, 1) if g is not None else None 
               for g in grad_output
           )
           return clipped
       
       # 安全的梯度监控（不影响计算图）
       def safe_monitor_hook(module, grad_input, grad_output):
           # 存储前先detach，避免内存泄漏
           stored_grads = [g.detach().clone() if g is not None else None 
                          for g in grad_output]
           # ... 分析 stored_grads ...
           return None

卷积网络
========

卷积神经网络（CNN）是深度学习中最重要和广泛应用的架构之一，特别适用于处理具有网格结构的数据，如图像、视频和序列数据。Riemann 提供了完整的卷积网络组件，包括一维、二维、三维卷积层和池化层。

卷积层
------

卷积层通过可学习的卷积核在输入数据上滑动，提取局部特征模式。Riemann 支持三种维度的卷积操作：

.. list-table:: 卷积层类型
   :header-rows: 1
   :widths: 20 40 40

   * - 卷积层
     - 适用数据类型
     - 典型应用场景
   * - ``Conv1d``
     - 一维序列数据 (N, C, L)
     - 音频处理、文本序列、时间序列
   * - ``Conv2d``
     - 二维图像数据 (N, C, H, W)
     - 图像分类、目标检测、图像分割
   * - ``Conv3d``
     - 三维体数据 (N, C, D, H, W)
     - 视频分析、医学图像、3D重建

Conv1d 层
~~~~~~~~~

**用途**：

- 处理一维序列数据，如音频波形、文本序列、时间序列
- 捕获局部时间依赖关系和模式
- 在自然语言处理中用于 n-gram 特征提取

**参数**：

- ``in_channels``：输入通道数
- ``out_channels``：输出通道数（卷积核数量）
- ``kernel_size``：卷积核大小
- ``stride``：卷积步长，默认 1
- ``padding``：填充大小，默认 0
- ``dilation``：膨胀率，默认 1
- ``groups``：分组数，默认 1（标准卷积）
- ``bias``：是否使用偏置，默认 True

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # 音频信号处理
    conv1d = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
    audio = rm.randn(8, 1, 1000)  # batch=8, channels=1, samples=1000
    output = conv1d(audio)
    print(output.shape)  # [8, 16, 1000]

Conv2d 层
~~~~~~~~~

**用途**：

- CNN 架构的核心组件，提取图像的局部特征
- 从低级边缘特征到高级语义特征的层次化特征提取
- 支持标准卷积、分组卷积、深度可分离卷积等

**参数**：

- ``in_channels``：输入通道数（如 RGB 图像为 3）
- ``out_channels``：输出通道数
- ``kernel_size``：卷积核大小（整数或元组）
- ``stride``：卷积步长，默认 1
- ``padding``：填充大小，默认 0
- ``dilation``：膨胀率，用于增大感受野，默认 1
- ``groups``：分组数，默认 1
- ``bias``：是否使用偏置，默认 True

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # 标准图像卷积
    conv2d = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
    image = rm.randn(4, 3, 224, 224)  # batch=4, RGB, height=224, width=224
    output = conv2d(image)
    print(output.shape)  # [4, 64, 224, 224]

Conv3d 层
~~~~~~~~~

**用途**：

- 处理三维数据，如视频、医学图像（MRI、CT）
- 捕获时空特征或 3D 空间特征
- 在视频分析中同时捕获时间和空间维度的相关性

**参数**：

- ``in_channels``：输入通道数
- ``out_channels``：输出通道数
- ``kernel_size``：卷积核大小（整数或三元组）
- ``stride``：卷积步长，默认 1
- ``padding``：填充大小，默认 0
- ``dilation``：膨胀率，默认 1
- ``groups``：分组数，默认 1
- ``bias``：是否使用偏置，默认 True

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # 视频数据处理
    conv3d = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    video = rm.randn(2, 3, 10, 64, 64)  # batch=2, RGB, frames=10, height=64, width=64
    output = conv3d(video)
    print(output.shape)  # [2, 16, 10, 64, 64]

池化层
------

池化层用于降低特征图的空间维度，减少计算量，并提供平移不变性。Riemann 提供最大池化和平均池化两种主要操作，每种都有标准和自适应变体。

.. list-table:: 池化层类型
   :header-rows: 1
   :widths: 20 40 40

   * - 池化层
     - 操作类型
     - 特点
   * - ``MaxPool1d/2d/3d``
     - 取窗口内最大值
     - 保留显著特征，对噪声鲁棒
   * - ``AdaptiveMaxPool1d/2d/3d``
     - 自适应最大池化
     - 自动计算池化参数，输出固定尺寸
   * - ``AvgPool1d/2d/3d``
     - 取窗口内平均值
     - 平滑下采样，保留整体信息
   * - ``AdaptiveAvgPool1d/2d/3d``
     - 自适应平均池化
     - 自动计算池化参数，输出固定尺寸

最大池化层
~~~~~~~~~~

最大池化层在池化窗口内选择最大值，保留最显著的特征，对噪声具有鲁棒性。Riemann 提供标准最大池化和自适应最大池化两种类型。

标准最大池化
^^^^^^^^^^^^

MaxPool1d 层
++++++++++++

**用途**：

- 对序列数据应用一维最大池化，在滑动窗口内选择最大值
- 降低序列维度，同时保留最显著的特征
- 为时间序列和序列数据提供平移不变性

**参数**：

- ``kernel_size``：池化窗口大小
- ``stride``：池化步长，默认等于 kernel_size
- ``padding``：填充大小，默认 0
- ``dilation``：膨胀率，默认 1
- ``ceil_mode``：是否向上取整计算输出长度，默认 False
- ``return_indices``：是否返回最大值位置索引，默认 False

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # 序列数据下采样
    maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
    features = rm.randn(4, 16, 100)  # batch=4, channels=16, length=100
    output = maxpool(features)
    print(output.shape)  # [4, 16, 50]

MaxPool2d 层
++++++++++++

**用途**：

- 通过选择局部区域的最大值来保留最显著的特征
- 提供平移不变性
- 大幅降低空间维度，减少后续层计算复杂度

**参数**：

- ``kernel_size``：池化窗口大小
- ``stride``：池化步长，默认等于 kernel_size
- ``padding``：填充大小，默认 0
- ``dilation``：膨胀率，默认 1
- ``ceil_mode``：是否向上取整计算输出尺寸，默认 False
- ``return_indices``：是否返回最大值位置索引，默认 False

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # 标准图像下采样
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    features = rm.randn(4, 64, 224, 224)
    output = maxpool(features)
    print(output.shape)  # [4, 64, 112, 112]

MaxPool3d 层
++++++++++++

**用途**：

- 对视频、医学图像等三维数据应用三维最大池化
- 降低三维空间维度，同时保留最显著的时空特征
- 提供三维平移不变性

**参数**：

- ``kernel_size``：池化窗口大小（可以是整数或 depth, height, width 元组）
- ``stride``：池化步长，默认等于 kernel_size
- ``padding``：填充大小，默认 0
- ``dilation``：膨胀率，默认 1
- ``ceil_mode``：是否向上取整计算输出尺寸，默认 False
- ``return_indices``：是否返回最大值位置索引，默认 False

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # 视频数据下采样
    maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
    features = rm.randn(4, 3, 16, 64, 64)  # batch=4, channels=3, frames=16, height=64, width=64
    output = maxpool(features)
    print(output.shape)  # [4, 3, 8, 32, 32]

自适应最大池化
^^^^^^^^^^^^^^

自适应池化层根据指定的输出尺寸自动计算池化核大小和步长，确保输出尺寸始终为指定值，无需手动计算池化参数。

AdaptiveMaxPool1d 层
++++++++++++++++++++

**用途**：

- 对序列数据应用一维自适应最大池化
- 保留序列中最显著的特征，同时映射到固定长度
- 适用于需要保留最大值信息的序列任务

**参数**：

- ``output_size``：输出序列长度
- ``return_indices``：是否返回最大值位置索引，默认 False

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # 自适应最大池化
    adaptive_pool = nn.AdaptiveMaxPool1d(output_size=10)
    features = rm.randn(4, 16, 50)
    output = adaptive_pool(features)
    print(output.shape)  # [4, 16, 10]

AdaptiveMaxPool2d 层
++++++++++++++++++++

**用途**：

- 对图像数据应用二维自适应最大池化
- 保留局部区域最显著的特征
- 适用于需要保留空间最大值信息的视觉任务

**参数**：

- ``output_size``：输出尺寸，可以是整数 (H, W) 元组或单个整数
- ``return_indices``：是否返回最大值位置索引，默认 False

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # 自适应最大池化
    adaptive_pool = nn.AdaptiveMaxPool2d(output_size=(7, 7))
    features = rm.randn(4, 64, 224, 224)
    output = adaptive_pool(features)
    print(output.shape)  # [4, 64, 7, 7]

AdaptiveMaxPool3d 层
++++++++++++++++++++

**用途**：

- 对三维数据应用三维自适应最大池化
- 保留三维空间中最显著的特征
- 适用于视频分析、医学图像等三维数据处理

**参数**：

- ``output_size``：输出尺寸，可以是整数 (D, H, W) 元组或单个整数
- ``return_indices``：是否返回最大值位置索引，默认 False

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # 三维自适应最大池化
    adaptive_pool = nn.AdaptiveMaxPool3d(output_size=(4, 7, 7))
    features = rm.randn(4, 32, 16, 64, 64)
    output = adaptive_pool(features)
    print(output.shape)  # [4, 32, 4, 7, 7]

平均池化层
~~~~~~~~~~

平均池化层计算池化窗口内的平均值，提供平滑的下采样效果，保留整体统计信息。Riemann 提供标准平均池化和自适应平均池化两种类型。

标准平均池化
^^^^^^^^^^^^

AvgPool1d 层
++++++++++++

**用途**：

- 对序列数据应用一维平均池化，在滑动窗口内计算平均值
- 为序列数据提供平滑的下采样效果
- 保留整体统计信息

**参数**：

- ``kernel_size``：池化窗口大小
- ``stride``：池化步长，默认等于 kernel_size
- ``padding``：填充大小，默认 0
- ``ceil_mode``：是否向上取整，默认 False
- ``count_include_pad``：计算平均值时是否包含填充值，默认 True
- ``divisor_override``：自定义平均计算的除数，默认 None

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # 平滑序列下采样
    avgpool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
    features = rm.randn(4, 16, 100)  # batch=4, channels=16, length=100
    output = avgpool(features)
    print(output.shape)  # [4, 16, 50]

AvgPool2d 层
++++++++++++

**用途**：

- 通过计算局部区域的平均值提供平滑的特征表示
- 相比最大池化，对噪声更加鲁棒
- 保留整体统计信息

**参数**：

- ``kernel_size``：池化窗口大小
- ``stride``：池化步长，默认等于 kernel_size
- ``padding``：填充大小，默认 0
- ``ceil_mode``：是否向上取整，默认 False
- ``count_include_pad``：计算平均值时是否包含填充值，默认 True

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # 平滑下采样
    avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
    features = rm.randn(4, 64, 224, 224)
    output = avgpool(features)
    print(output.shape)  # [4, 64, 112, 112]

AvgPool3d 层
++++++++++++

**用途**：

- 对视频、医学图像等三维数据应用三维平均池化
- 提供平滑的三维下采样，同时保留整体时空信息
- 相比三维最大池化，对噪声更加鲁棒

**参数**：

- ``kernel_size``：池化窗口大小（可以是整数或 depth, height, width 元组）
- ``stride``：池化步长，默认等于 kernel_size
- ``padding``：填充大小，默认 0
- ``ceil_mode``：是否向上取整，默认 False
- ``count_include_pad``：计算平均值时是否包含填充值，默认 True
- ``divisor_override``：自定义平均计算的除数，默认 None

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # 三维数据平滑下采样
    avgpool = nn.AvgPool3d(kernel_size=2, stride=2)
    features = rm.randn(4, 32, 16, 64, 64)  # batch=4, channels=32, depth=16, height=64, width=64
    output = avgpool(features)
    print(output.shape)  # [4, 32, 8, 32, 32]

自适应平均池化
^^^^^^^^^^^^^^

自适应池化层根据指定的输出尺寸自动计算池化核大小和步长，确保输出尺寸始终为指定值，无需手动计算池化参数。

AdaptiveAvgPool1d 层
++++++++++++++++++++

**用途**：

- 对序列数据应用一维自适应平均池化
- 将任意长度的序列映射到指定的固定长度
- 常用于序列模型的输出层，统一不同长度序列的维度

**参数**：

- ``output_size``：输出序列长度，可以是整数或 None（表示保持原尺寸）

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # 将不同长度的序列映射到固定长度 10
    adaptive_pool = nn.AdaptiveAvgPool1d(output_size=10)
    
    # 输入序列长度为 50
    features = rm.randn(4, 16, 50)  # batch=4, channels=16, length=50
    output = adaptive_pool(features)
    print(output.shape)  # [4, 16, 10]
    
    # 输入序列长度为 100，输出仍为 10
    features = rm.randn(4, 16, 100)
    output = adaptive_pool(features)
    print(output.shape)  # [4, 16, 10]

AdaptiveAvgPool2d 层
++++++++++++++++++++

**用途**：

- 对图像数据应用二维自适应平均池化
- 将任意尺寸的特征图映射到指定的固定尺寸
- 常用于 CNN 末尾，将不同尺寸的图像特征转换为固定维度

**参数**：

- ``output_size``：输出尺寸，可以是整数 (H, W) 元组或单个整数（表示正方形输出）

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # 将任意尺寸的特征图映射到 7x7
    adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
    
    # 输入尺寸为 224x224
    features = rm.randn(4, 64, 224, 224)
    output = adaptive_pool(features)
    print(output.shape)  # [4, 64, 7, 7]
    
    # 输入尺寸为 128x128，输出仍为 7x7
    features = rm.randn(4, 64, 128, 128)
    output = adaptive_pool(features)
    print(output.shape)  # [4, 64, 7, 7]

AdaptiveAvgPool3d 层
++++++++++++++++++++

**用途**：

- 对视频、医学图像等三维数据应用三维自适应平均池化
- 将任意尺寸的三维特征图映射到指定的固定尺寸
- 常用于 3D CNN 末尾，统一不同尺寸的三维特征

**参数**：

- ``output_size``：输出尺寸，可以是整数 (D, H, W) 元组或单个整数（表示立方体输出）

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # 将任意尺寸的三维特征映射到 4x7x7
    adaptive_pool = nn.AdaptiveAvgPool3d(output_size=(4, 7, 7))
    
    # 输入尺寸为 16x64x64
    features = rm.randn(4, 32, 16, 64, 64)
    output = adaptive_pool(features)
    print(output.shape)  # [4, 32, 4, 7, 7]

MNIST 手写体识别示例
--------------------

以下是一个完整的 MNIST 手写体识别 CNN 模型示例，包含训练和推理全过程：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    import riemann.optim as opt
    from riemann.vision.datasets import MNIST
    from riemann.vision import transforms
    from riemann.utils.data import DataLoader
    from riemann import cuda

    class MNISTNet(nn.Module):
        """MNIST 手写体识别网络"""
        
        def __init__(self):
            super().__init__()
            # 特征提取层
            self.features = nn.Sequential(
                # 第一层卷积: 1@28x28 -> 32@28x28
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 32@14x14
                
                # 第二层卷积: 32@14x14 -> 64@14x14
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 64@7x7
            )
            
            # 分类器
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 10)
            )
            
            # 损失函数
            self.loss_func = nn.CrossEntropyLoss()
        
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
        
        def train_step(self, inputs, targets):
            """单步训练"""
            outputs = self.forward(inputs)
            loss = self.loss_func(outputs, targets)
            self.optimizer.zero_grad(True)
            loss.backward()
            self.optimizer.step()
            return loss
        
        def evaluate(self, dataloader, device):
            """评估模型性能"""
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in dataloader:
                img_tensors, target_tensors = batch
                # 将数据移动到指定设备
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


    def main():
        """主函数：完整的训练和推理流程"""
        print("MNIST 手写体识别 CNN 示例")
        
        # 检查CUDA可用性
        CUDA_AVAILABLE = cuda.CUPY_AVAILABLE
        device = 'cuda' if CUDA_AVAILABLE else 'cpu'
        print(f"使用设备: {device}")
        
        # 1. 数据准备
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST 数据集的均值和标准差
        ])
        
        # 加载数据集
        train_dataset = MNIST(root='./data', train=True, transform=transform)
        test_dataset = MNIST(root='./data', train=False, transform=transform)
        
        # 创建数据加载器（批次大小512以提升效率）
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
        
        # 2. 创建模型并移动到设备
        model = MNISTNet()
        model.to(device)
        print(f"模型结构:\n{model}")
        
        # 初始化优化器（在模型移动到设备后）
        model.optimizer = opt.Adam(model.parameters(), lr=0.001)
        
        # 3. 训练模型
        num_epochs = 5
        print(f"\n开始训练，共 {num_epochs} 个 epoch...")
        
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                # 将数据移动到指定设备
                images = images.to(device)
                labels = labels.to(device)
                
                loss = model.train_step(images, labels)
                train_loss += loss.item()
                
                if batch_idx % 50 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], "
                          f"Batch [{batch_idx}/{len(train_loader)}], "
                          f"Loss: {loss.item():.4f}")
            
            # 评估阶段
            model.eval()
            test_accuracy, test_loss = model.evaluate(test_loader, device)
            avg_train_loss = train_loss / len(train_loader)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] 完成: "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Test Loss: {test_loss:.4f}, "
                  f"Test Accuracy: {test_accuracy*100:.2f}%")
        
        # 4. 推理演示
        print("\n推理演示:")
        model.eval()
        
        # 获取一批测试数据
        test_images, test_labels = next(iter(test_loader))
        # 将数据移动到指定设备
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)
        
        # 前向传播
        with rm.no_grad():
            outputs = model(test_images[:5])
            predictions = outputs.argmax(dim=1)
        
        print(f"预测结果: {predictions.tolist()}")
        print(f"真实标签: {test_labels[:5].tolist()}")
        print(f"预测准确率: {(predictions == test_labels[:5]).sum().item() / 5 * 100:.2f}%")

    if __name__ == "__main__":
        main()

CIFAR-10 图像分类示例
---------------------

以下是一个完整的 CIFAR-10 图像分类 CNN 模型示例，包含训练和推理全过程：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    import riemann.optim as opt
    from riemann.vision.datasets import CIFAR10
    from riemann.vision import transforms
    from riemann.utils.data import DataLoader
    from riemann import cuda

    class CIFAR10Net(nn.Module):
        """CIFAR-10 图像分类网络（简化版）"""
        
        def __init__(self):
            super().__init__()
            # 特征提取层（简化结构，减少卷积层数）
            self.features = nn.Sequential(
                # 第一层: 3@32x32 -> 32@16x16
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.25),
                
                # 第二层: 32@16x16 -> 64@8x8
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.25),
                
                # 第三层: 64@8x8 -> 128@4x4
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.25),
            )
            
            # 分类器
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 10)
            )
            
            # 损失函数
            self.loss_func = nn.CrossEntropyLoss()
        
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
        
        def train_step(self, inputs, targets):
            """单步训练"""
            outputs = self.forward(inputs)
            loss = self.loss_func(outputs, targets)
            self.optimizer.zero_grad(True)
            loss.backward()
            self.optimizer.step()
            return loss
        
        def evaluate(self, dataloader, device):
            """评估模型性能"""
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in dataloader:
                img_tensors, target_tensors = batch
                # 将数据移动到指定设备
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


    def main():
        """主函数：完整的训练和推理流程"""
        print("CIFAR-10 图像分类 CNN 示例")
        
        # 检查CUDA可用性
        CUDA_AVAILABLE = cuda.CUPY_AVAILABLE
        device = 'cuda' if CUDA_AVAILABLE else 'cpu'
        print(f"使用设备: {device}")
        
        # 1. 数据准备
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # RGB三通道归一化
        ])
        
        # 加载数据集
        train_dataset = CIFAR10(root='./data', train=True, transform=transform)
        test_dataset = CIFAR10(root='./data', train=False, transform=transform)
        
        # 创建数据加载器（批次大小512以提升效率）
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
        
        # 2. 创建模型并移动到设备
        model = CIFAR10Net()
        model.to(device)
        print(f"模型结构:\n{model}")
        
        # 初始化优化器（在模型移动到设备后）
        model.optimizer = opt.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # 3. 训练模型
        num_epochs = 5
        print(f"\n开始训练，共 {num_epochs} 个 epoch...")
        
        best_accuracy = 0.0
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                # 将数据移动到指定设备
                images = images.to(device)
                labels = labels.to(device)
                
                loss = model.train_step(images, labels)
                train_loss += loss.item()
                
                if batch_idx % 50 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], "
                          f"Batch [{batch_idx}/{len(train_loader)}], "
                          f"Loss: {loss.item():.4f}")
            
            # 评估阶段
            model.eval()
            test_accuracy, test_loss = model.evaluate(test_loader, device)
            avg_train_loss = train_loss / len(train_loader)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] 完成: "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Test Loss: {test_loss:.4f}, "
                  f"Test Accuracy: {test_accuracy*100:.2f}%")
            
            # 保存最佳模型
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                print(f"  -> 最佳模型更新! 准确率: {best_accuracy*100:.2f}%")
        
        print(f"\n训练完成! 最佳测试准确率: {best_accuracy*100:.2f}%")
        
        # 4. 推理演示
        print("\n推理演示:")
        model.eval()
        
        # 获取一批测试数据
        test_images, test_labels = next(iter(test_loader))
        # 将数据移动到指定设备
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)
        
        # 类别名称
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
        
        # 前向传播
        with rm.no_grad():
            outputs = model(test_images[:5])
            predictions = outputs.argmax(dim=1)
        
        print(f"预测类别: {[classes[p] for p in predictions.tolist()]}")
        print(f"真实类别: {[classes[t] for t in test_labels[:5].tolist()]}")
        print(f"预测准确率: {(predictions == test_labels[:5]).sum().item() / 5 * 100:.2f}%")

    if __name__ == "__main__":
        main()

卷积网络设计要点
----------------

1. **感受野设计**：
   
   - 通过堆叠多个小卷积核（如 3x3）替代大卷积核，可以减少参数量同时保持相同的感受野
   - 使用膨胀卷积（dilation）可以在不增加参数的情况下增大感受野

2. **下采样策略**：
   
   - 使用池化层（MaxPool/AvgPool）或步长大于1的卷积进行下采样
   - 下采样可以逐步减小特征图尺寸，增加特征通道数，提取更高级的特征

3. **归一化和正则化**：
   
   - 在卷积层后使用 BatchNorm 可以加速训练并提高模型稳定性
   - 使用 Dropout 可以防止过拟合

4. **激活函数选择**：
   
   - ReLU 是最常用的激活函数，计算简单且能有效缓解梯度消失
   - 在深层网络中，LeakyReLU 或 GELU 可能表现更好

Transformer
===========

Transformer 是一种基于注意力机制的深度学习架构，最初用于自然语言处理任务，现已广泛应用于计算机视觉、语音识别等领域。Riemann 提供了完整的 Transformer 组件，与 PyTorch 接口兼容。

Transformer 架构概述
--------------------

Transformer 由编码器（Encoder）和解码器（Decoder）两部分组成：

- **编码器**：将输入序列编码为连续的表示（memory）
- **解码器**：根据编码器的输出和已生成的目标序列，自回归地生成输出序列

.. code-block:: text

    输入序列 → [编码器] → Memory → [解码器] → 输出序列
                      ↑___________↓
                        交叉注意力

多头注意力机制 (MultiheadAttention)
-----------------------------------

多头注意力是 Transformer 的核心组件，允许模型同时关注来自不同表示子空间的信息。

**原理**：

多头注意力将输入的 Query、Key、Value 分别投影到多个子空间（头），在每个子空间独立计算注意力，然后将结果拼接并再次投影：

.. code-block:: text

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W^O
    where head_i = Attention(Q @ W_i^Q, K @ W_i^K, V @ W_i^V)
    
    Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V

**用途**：

- 捕获序列中不同位置之间的依赖关系
- 自注意力机制允许每个位置关注序列中的所有位置
- 多头设计使模型能够关注不同类型的信息

**参数**：

- ``embed_dim``：输入和输出的维度
- ``num_heads``：注意力头的数量，必须能整除 embed_dim
- ``dropout``：注意力权重的 dropout 概率，默认 0.0
- ``bias``：是否使用偏置，默认 True
- ``add_bias_kv``：是否为 key 和 value 添加可学习的偏置，默认 False
- ``add_zero_attn``：是否在 key 和 value 序列末尾添加零向量，默认 False
- ``kdim``：Key 的维度，默认 None（使用 embed_dim）
- ``vdim``：Value 的维度，默认 None（使用 embed_dim）
- ``batch_first``：输入格式是否为 (batch, seq, feature)，默认 False

**前向传播参数**：

- ``query``, ``key``, ``value``：输入张量，形状取决于 batch_first
- ``attn_mask``：注意力掩码，支持 2D (tgt_len, src_len) 或 3D (batch*num_heads, tgt_len, src_len)
- ``key_padding_mask``：key 的填充掩码，支持 bool 或 float 类型，形状 (batch, src_len)
- ``is_causal``：是否使用因果掩码（防止关注未来位置），默认 False
- ``need_weights``：是否返回注意力权重，默认 True
- ``average_attn_weights``：是否对多头注意力权重取平均，默认 True

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # 创建多头注意力层
    mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

    # 输入张量
    batch_size, seq_len, embed_dim = 2, 10, 512
    query = rm.randn(batch_size, seq_len, embed_dim)
    key = rm.randn(batch_size, seq_len, embed_dim)
    value = rm.randn(batch_size, seq_len, embed_dim)

    # 前向传播
    output, attn_weights = mha(query, key, value)
    print(f"输出形状: {output.shape}")  # [2, 10, 512]
    print(f"注意力权重形状: {attn_weights.shape}")  # [2, 8, 10, 10] (batch, num_heads, tgt_len, src_len)

**使用掩码的示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

    batch_size, seq_len, embed_dim = 2, 10, 512
    query = rm.randn(batch_size, seq_len, embed_dim)
    key = rm.randn(batch_size, seq_len, embed_dim)
    value = rm.randn(batch_size, seq_len, embed_dim)

    # 因果掩码（用于自回归模型）
    output, attn_weights = mha(query, key, value, is_causal=True)

    # 自定义注意力掩码（2D）
    attn_mask = rm.zeros(seq_len, seq_len)
    attn_mask[0, 5:] = float('-inf')  # 第0个位置不能关注第5个及之后的位置
    output, _ = mha(query, key, value, attn_mask=attn_mask)

    # key 填充掩码（忽略填充位置）
    key_padding_mask = rm.zeros(batch_size, seq_len)
    key_padding_mask[0, 8:] = float('-inf')  # 第0个样本的第8个及之后位置为填充
    output, _ = mha(query, key, value, key_padding_mask=key_padding_mask)

Transformer 编码器
------------------

编码器由多个相同的编码器层堆叠而成，每个编码器层包含：

1. **多头自注意力**：处理输入序列内部的关系
2. **前馈网络**：对每个位置独立进行非线性变换
3. **残差连接和层归一化**：稳定训练

**两种归一化模式**：

- **Post-LN** （默认）：先执行子层，再归一化（原始 Transformer 论文）
- **Pre-LN** ：先归一化，再执行子层（训练更稳定）

**组件**：

- ``TransformerEncoderLayer``：单个编码器层
- ``TransformerEncoder``：由 N 个编码器层组成的完整编码器

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # 创建编码器层
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=512, nhead=8, dim_feedforward=2048, 
        dropout=0.1, batch_first=True
    )

    # 创建编码器（6层）
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

    # 输入序列 (batch=2, seq_len=10, d_model=512)
    src = rm.randn(2, 10, 512)
    
    # 前向传播
    output = encoder(src)
    print(f"编码器输出形状: {output.shape}")  # [2, 10, 512]

Transformer 解码器
------------------

解码器由多个相同的解码器层堆叠而成，每个解码器层包含：

1. **掩码多头自注意力**：防止关注未来位置（自回归）
2. **交叉注意力**：关注编码器的输出（memory）
3. **前馈网络**：非线性变换
4. **残差连接和层归一化**

**组件**：

- ``TransformerDecoderLayer``：单个解码器层
- ``TransformerDecoder``：由 N 个解码器层组成的完整解码器

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # 创建解码器层
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=512, nhead=8, dim_feedforward=2048,
        dropout=0.1, batch_first=True
    )

    # 创建解码器（6层）
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

    # 目标序列 (batch=2, tgt_len=20, d_model=512)
    tgt = rm.randn(2, 20, 512)
    
    # 编码器输出 (batch=2, src_len=10, d_model=512)
    memory = rm.randn(2, 10, 512)
    
    # 前向传播
    output = decoder(tgt, memory)
    print(f"解码器输出形状: {output.shape}")  # [2, 20, 512]

完整 Transformer 模型
---------------------

Riemann 提供了完整的 Transformer 模型，包含编码器和解码器。

**参数**：

- ``d_model``：模型维度，默认 512
- ``nhead``：注意力头数，默认 8
- ``num_encoder_layers``：编码器层数，默认 6
- ``num_decoder_layers``：解码器层数，默认 6
- ``dim_feedforward``：前馈网络维度，默认 2048
- ``dropout``：dropout 概率，默认 0.1
- ``activation``：激活函数，'relu' 或 'gelu'，默认 'relu'
- ``batch_first``：输入格式，默认 False

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # 创建 Transformer 模型
    transformer = nn.Transformer(
        d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
        dim_feedforward=2048, dropout=0.1, batch_first=True
    )

    # 源序列 (batch=2, src_len=10, d_model=512)
    src = rm.randn(2, 10, 512)
    
    # 目标序列 (batch=2, tgt_len=20, d_model=512)
    tgt = rm.randn(2, 20, 512)
    
    # 前向传播
    output = transformer(src, tgt)
    print(f"Transformer 输出形状: {output.shape}")  # [2, 20, 512]

机器翻译示例
------------

以下是一个完整的机器翻译模型示例，展示了 Transformer 在训练和推理中的使用：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    class TransformerTranslationModel(nn.Module):
        """Transformer 机器翻译模型"""
        
        def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, 
                     num_encoder_layers=6, num_decoder_layers=6, max_seq_len=100):
            super().__init__()
            self.d_model = d_model
            
            # 词嵌入层
            self.src_embedding = nn.Embedding(src_vocab_size, d_model)
            self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
            
            # 位置编码（简化为可学习参数）
            self.pos_encoding = nn.Embedding(max_seq_len, d_model)
            
            # Transformer
            self.transformer = nn.Transformer(
                d_model=d_model, nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=2048, dropout=0.1,
                batch_first=True
            )
            
            # 输出投影
            self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        
        def forward(self, src, tgt, src_mask=None, tgt_mask=None):
            """训练前向传播"""
            # 添加位置编码
            src_pos = rm.arange(src.shape[1]).expand(src.shape[0], -1)
            tgt_pos = rm.arange(tgt.shape[1]).expand(tgt.shape[0], -1)
            
            src_emb = self.src_embedding(src) + self.pos_encoding(src_pos)
            tgt_emb = self.tgt_embedding(tgt) + self.pos_encoding(tgt_pos)
            
            # Transformer 前向传播
            output = self.transformer(src_emb, tgt_emb, src_mask=src_mask, tgt_mask=tgt_mask)
            
            # 投影到词表维度
            logits = self.output_proj(output)
            return logits
        
        def generate(self, src, max_len=50, start_token=1, end_token=2):
            """推理：自回归生成翻译结果"""
            self.eval()
            
            # 编码源序列
            src_pos = rm.arange(src.shape[1]).expand(src.shape[0], -1)
            src_emb = self.src_embedding(src) + self.pos_encoding(src_pos)
            memory = self.transformer.encoder(src_emb)
            
            # 自回归生成
            tgt = rm.full((src.shape[0], 1), start_token, dtype=rm.int64)
            
            for _ in range(max_len):
                # 生成因果掩码（上三角为True，防止关注未来位置）
                tgt_mask = rm.full((tgt.shape[1], tgt.shape[1]), float('-inf'))
                tgt_mask = tgt_mask.triu(diagonal=1)  # 上三角（不包括对角线）设为-inf
                
                # 解码
                tgt_pos = rm.arange(tgt.shape[1]).expand(tgt.shape[0], -1)
                tgt_emb = self.tgt_embedding(tgt) + self.pos_encoding(tgt_pos)
                output = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                
                # 预测下一个词
                logits = self.output_proj(output[:, -1, :])
                next_token = logits.argmax(dim=-1, keepdim=True)
                
                # 添加到序列
                tgt = rm.concatenate([tgt, next_token], dim=1)
                
                # 检查是否生成结束符
                if (next_token == end_token).all():
                    break
            
            return tgt

    # 创建模型
    model = TransformerTranslationModel(
        src_vocab_size=10000, tgt_vocab_size=10000,
        d_model=512, nhead=8, num_encoder_layers=6
    )

    # 模拟训练数据
    src = rm.randint(0, 10000, (2, 20))  # 源序列
    tgt = rm.randint(0, 10000, (2, 25))  # 目标序列

    # 训练前向传播
    logits = model(src, tgt)
    print(f"训练输出形状: {logits.shape}")  # [2, 25, 10000]

    # 推理生成
    generated = model.generate(src, max_len=30)
    print(f"生成序列形状: {generated.shape}")

编码器与解码器的区别
--------------------

.. list-table:: 编码器 vs 解码器
   :header-rows: 1
   :widths: 25 35 40

   * - 特性
     - 编码器 (Encoder)
     - 解码器 (Decoder)
   * - **注意力类型**
     - 仅自注意力
     - 自注意力 + 交叉注意力
   * - **掩码**
     - 无掩码（可看全部输入）
     - 因果掩码（只能看已生成部分）
   * - **输入**
     - 源序列
     - 目标序列 + 编码器输出
   * - **输出**
     - 连续表示 (memory)
     - 目标序列的下一个词预测
   * - **使用场景**
     - 文本分类、特征提取
     - 机器翻译、文本生成

**训练与推理的配合**：

1. **训练阶段**：
   
   - 编码器一次性处理完整的源序列
   - 解码器使用教师强制（teacher forcing），输入真实的目标序列
   - 并行计算所有位置的输出

2. **推理阶段**：
   
   - 编码器同样一次性处理源序列
   - 解码器自回归生成，每次生成一个词
   - 使用已生成的词作为下一步的输入
   - 直到生成结束符或达到最大长度

注意事项
--------

1. **位置编码**：
   
   - Transformer 本身不包含位置信息，需要额外添加位置编码
   - 可以使用正弦/余弦位置编码或可学习的位置嵌入

2. **掩码使用**：
   
   - ``src_key_padding_mask``：忽略源序列中的填充位置
   - ``tgt_key_padding_mask``：忽略目标序列中的填充位置
   - ``tgt_mask`` （因果掩码）：防止解码器关注未来位置

3. **内存优化**：
   
   - 注意力计算复杂度为 O(n²)，长序列会消耗大量内存
   - 可以考虑使用稀疏注意力或分块注意力优化

4. **初始化**：
   
   - Transformer 对初始化敏感，使用 Xavier/Glorot 初始化
   - Riemann 的 Transformer 组件已包含适当的初始化

KAN网络
-------

Kolmogorov-Arnold网络（KAN）是一种新型神经网络架构，使用可学习的B样条激活函数替代传统的固定激活函数。KAN基于Kolmogorov-Arnold表示定理，该定理证明任何多元连续函数都可以表示为单变量连续函数的组合。

KAN网络原理
~~~~~~~~~~~~

**核心思想**

传统MLP使用固定的非线性激活函数（如ReLU、Sigmoid）：

.. math::

    \text{MLP: } x \mapsto \sum_i w_i \cdot \sigma(\text{激活函数}(x))

KAN使用可学习的单变量函数替代固定激活：

.. math::

    \text{KAN: } x \mapsto \sum_i \phi_i(x_i) \cdot w_i

其中 :math:`\phi_i` 是可学习的B样条函数。

**双路径计算**

KANLinear层包含两条计算路径：

1. **基函数路径**：使用固定激活函数（如SiLU）提供基础非线性
2. **样条路径**：使用可学习的B样条函数提供灵活的非线性变换

.. code-block:: text

      输入 x
        │
        ├──────→ [基函数路径] ────→ SiLU(x) @ base_weight   ──────┐
        │         　　　　　                                      ├──→ 相加 ──→ 输出
        └──────→ [样条路径　] ────→ B-splines(x) @ spline_weight ─┘

**B样条基函数**

B样条（B-spline）是分段多项式函数，具有局部支撑性和平滑性。KAN使用de Boor递归公式计算B样条基函数。

**阶数与多项式次数的区别**

B样条的"阶数"（order）与多项式的"次数"（degree）是不同的概念：

- **阶数（k）**：B样条的递归阶数，决定基函数的复杂程度
- **次数（degree）**：实际多项式的最高幂次，等于 ``阶数 - 1``

例如，3阶B样条对应2次（二次）多项式，具有连续的一阶导数。

**de Boor递归算法**

de Boor算法是计算B样条基函数的标准方法，通过递归定义：

零阶基函数（指示函数）：

.. math::

    B_{i,0}(x) = \begin{cases} 
    1 & \text{if } t_i \leq x < t_{i+1} \\
    0 & \text{otherwise}
    \end{cases}

k阶基函数（递归定义）：

.. math::

    B_{i,k}(x) = \frac{x - t_i}{t_{i+k} - t_i} B_{i,k-1}(x) + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1,k-1}(x)

其中 :math:`t_i` 是网格节点（knots）。

**算法特性**：

1. **局部支撑性**：每个基函数只在有限区间非零，修改一个控制点只影响局部区域
2. **单位分解性**：所有基函数在任意点的和为1，保证数值稳定性
3. **连续性**：k阶B样条具有 :math:`C^{k-2}` 连续性（k-2阶连续导数）

**B样条网格的可解释性**

B样条网格具有天然的可解释性优势：

1. **可视化理解**：
   
   每个基函数的形状直观可见，可以绘制出基函数曲线，观察其对输出的贡献

2. **局部控制**：
   
   每个网格区间对应一个局部基函数，权重的物理意义明确：控制该区间内的函数形状

3. **平滑性保证**：
   
   由于B样条的连续性，学习到的函数天然平滑，不会出现神经网络常见的锯齿状输出

4. **符号表达**：
   
   B样条可以转换为分段多项式表达式，便于提取显式的数学公式

**自适应网格**

KAN支持自适应网格更新，根据输入数据分布动态调整网格点位置，使网络更好地拟合数据分布。

**为什么需要自适应网格**

固定网格在数据分布不均匀时存在问题：

1. **数据稀疏区域**：网格点过于密集，浪费计算资源
2. **数据密集区域**：网格点过于稀疏，拟合精度不足
3. **边界效应**：固定网格可能无法覆盖数据的实际范围

自适应网格通过动态调整，使网格点集中在数据密集区域，提高拟合效率。

**自适应网格算法**

Riemann的KAN实现采用以下自适应策略：

**步骤1：计算当前样条输出**

.. code-block:: python

    # 计算当前B样条基函数
    splines = self.b_splines(x)  # (batch, in_features, coeff)
    
    # 计算未聚合的样条输出
    unreduced_spline_output = splines @ orig_coeff

**步骤2：构建自适应网格**

基于输入数据的实际分布计算自适应网格点：

.. math::

    \text{grid}_{\text{adaptive}} = \text{sorted_data}\left[\text{linspace}(0, N-1, \text{grid_size}+1)\right]

其中 ``sorted_data`` 是按值排序后的输入数据。

**步骤3：构建均匀网格**

计算覆盖数据范围的均匀网格：

.. math::

    \text{grid}_{\text{uniform}} = \text{linspace}(\min(x)-\epsilon, \max(x)+\epsilon, \text{grid_size}+1)

**步骤4：混合网格**

结合自适应网格和均匀网格，使用插值系数 ``grid_eps`` 控制混合比例：

.. math::

    \text{grid} = \text{grid_eps} \cdot \text{grid}_{\text{uniform}} + (1 - \text{grid_eps}) \cdot \text{grid}_{\text{adaptive}}

**步骤5：扩展边界**

在网格两端扩展额外的节点以支持边界计算：

.. code-block:: python

    # 在网格前后添加额外的节点
    grid = concatenate([
        grid[:1] - step * arange(spline_order, 0, -1),
        grid,
        grid[-1:] + step * arange(1, spline_order + 1)
    ])

**步骤6：更新样条系数**

使用最小二乘法将旧网格上的样条函数映射到新网格：

.. code-block:: python

    # 使用curve2coeff将曲线点转换为新网格的B样条系数
    self.spline_weight.data = self.curve2coeff(x, unreduced_spline_output).data

**算法优势**：

1. **数据驱动**：网格自动适应数据分布，无需人工调整
2. **平滑过渡**：混合策略避免网格突变导致的训练不稳定
3. **计算高效**：只在必要时更新（如每20个epoch），避免频繁计算开销

KAN网络适用场景
~~~~~~~~~~~~~~~~

KAN网络特别适合以下场景：

**1. 可解释性要求高的任务**

- 科学计算和物理建模
- 需要理解特征重要性的任务
- 可解释AI（XAI）应用

**2. 函数拟合和符号回归**

- 发现数据背后的数学公式
- 物理定律发现
- 方程拟合

**3. 小样本学习**

- 参数效率更高
- 在数据量有限时表现更好
- 避免过拟合

**4. 需要平滑输出的任务**

- B样条提供平滑的函数近似
- 适合需要连续导数的应用
- 物理模拟和控制系统

**5. 与传统MLP的比较**

+--------------+-------------------+-------------------+
| 特性         | MLP               | KAN               |
+==============+===================+===================+
| 激活函数     | 固定              | 可学习（B样条）   |
+--------------+-------------------+-------------------+
| 可解释性     | 较低              | 较高              |
+--------------+-------------------+-------------------+
| 参数效率     | 一般              | 较高              |
+--------------+-------------------+-------------------+
| 训练速度     | 快                | 较慢              |
+--------------+-------------------+-------------------+
| 适用场景     | 通用              | 科学计算、解释性  |
+--------------+-------------------+-------------------+

Riemann的KAN模块
~~~~~~~~~~~~~~~~~

Riemann在 ``riemann.nn.kan`` 模块中提供了完整的KAN实现：

**主要组件**：

- ``KANLinear``：KAN线性层，核心构建块
- ``KAN``：多层KAN网络容器

**特性**：

- 高效的矩阵乘法实现
- 支持自适应网格更新
- 提供L1正则化和熵正则化
- 与Riemann自动微分完全兼容

KANLinear模块
~~~~~~~~~~~~~~

**结构说明**

KANLinear是KAN网络的基本构建块，包含以下参数：

**参数**：

- ``in_features``：输入特征维度
- ``out_features``：输出特征维度
- ``grid_size``：网格大小，控制B样条的分段数（默认5）
- ``spline_order``：B样条阶数，控制平滑度（默认3）
- ``scale_noise``：噪声缩放系数，用于初始化（默认0.1）
- ``scale_base``：基函数权重缩放系数（默认1.0）
- ``scale_spline``：样条权重缩放系数（默认1.0）
- ``enable_standalone_scale_spline``：是否启用独立的样条缩放（默认True）
- ``base_activation``：基函数激活函数（默认SiLU）
- ``grid_eps``：网格更新时的插值系数（默认0.02）
- ``grid_range``：网格值范围（默认[-1, 1]）

**内部参数**：

- ``base_weight``：基函数路径权重，形状 ``(out_features, in_features)``
- ``spline_weight``：样条路径权重，形状 ``(out_features, in_features, grid_size + spline_order)``
- ``spline_scaler``：样条缩放因子，形状 ``(out_features, in_features)`` ，可选
- ``grid``：B样条网格点，形状 ``(in_features, grid_size + 2*spline_order + 1)``

**使用方法**

基本使用示例：

.. code-block:: python

    import riemann as rm
    from riemann.nn import KANLinear

    # 创建KAN线性层
    layer = KANLinear(
        in_features=10,
        out_features=5,
        grid_size=5,
        spline_order=3
    )

    # 输入数据
    x = rm.randn(4, 10)  # (batch_size, in_features)

    # 前向传播
    output = layer(x)
    print(f"输出形状: {output.shape}")  # (4, 5)

带自适应网格更新的训练：

.. code-block:: python

    # 训练过程中更新网格
    for epoch in range(num_epochs):
        for batch in dataloader:
            x, y = batch
            
            # 每几个epoch更新一次网格
            if epoch % 20 == 0:
                layer.update_grid(x)
            
            output = layer(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

使用正则化：

.. code-block:: python

    # 计算正则化损失
    reg_loss = layer.regularization_loss(
        regularize_activation=1.0,
        regularize_entropy=1.0
    )
    
    # 总损失 = 任务损失 + 正则化损失
    total_loss = task_loss + 0.01 * reg_loss

KAN容器模块
~~~~~~~~~~~~

**结构说明**

KAN是多层KAN网络的容器，自动堆叠多个KANLinear层。

**参数**：

- ``layers_hidden``：隐藏层维度列表，如 ``[28*28, 64, 10]``
- ``grid_size``：网格大小（默认5）
- ``spline_order``：B样条阶数（默认3）
- ``scale_noise``：噪声缩放系数（默认0.1）
- ``scale_base``：基函数权重缩放系数（默认1.0）
- ``scale_spline``：样条权重缩放系数（默认1.0）
- ``base_activation``：基函数激活函数（默认SiLU）
- ``grid_eps``：网格更新插值系数（默认0.02）
- ``grid_range``：网格值范围（默认[-1, 1]）

**使用方法**

创建多层KAN网络：

.. code-block:: python

    from riemann.nn import KAN

    # 创建多层KAN网络
    model = KAN(
        layers_hidden=[784, 64, 32, 10],
        grid_size=5,
        spline_order=3
    )

    # 输入数据
    x = rm.randn(4, 784)

    # 前向传播
    output = model(x)
    print(f"输出形状: {output.shape}")  # (4, 10)

训练时更新网格：

.. code-block:: python

    # 前向传播时更新网格
    output = model(x, update_grid=True)

完整训练示例：

.. code-block:: python

    import riemann as rm
    from riemann.nn import KAN
    from riemann.optim import Adam

    # 创建模型
    model = KAN([784, 64, 10], grid_size=5, spline_order=3)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = rm.nn.CrossEntropyLoss()

    # 训练循环
    for epoch in range(100):
        for batch_idx, (data, target) in enumerate(train_loader):
            # 展平图像
            data = data.view(data.size(0), -1)
            
            # 每20个epoch更新网格
            update_grid = (epoch % 20 == 0 and batch_idx == 0)
            
            # 前向传播
            output = model(data, update_grid=update_grid)
            
            # 计算损失
            loss = criterion(output, target)
            
            # 添加正则化
            reg_loss = model.regularization_loss()
            total_loss = loss + 0.01 * reg_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

KAN设计建议
~~~~~~~~~~~~

1. **网格大小选择**：
   
   - 小网格（3-5）：适合简单函数，参数少
   - 大网格（10-20）：适合复杂函数，但参数多
   - 建议从5开始，根据任务调整

2. **样条阶数选择**：
   
   - 1阶：线性样条，不连续导数
   - 3阶：三次样条，推荐默认值
   - 5阶：更高平滑度，但计算量大

3. **网格更新策略**：
   
   - 训练初期频繁更新（每10-20 epoch）
   - 训练后期减少更新频率
   - 避免每个batch都更新（计算开销大）

4. **正则化使用**：
   
   - L1正则化促进稀疏性
   - 熵正则化促进选择性
   - 正则化系数建议0.001-0.01
