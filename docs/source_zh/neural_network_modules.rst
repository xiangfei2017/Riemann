如何搭建神经网络
================

Riemann 通过 ``riemann.nn`` 包提供了一套全面的神经网络模块。这些模块是创建和训练神经网络的构建块。

快速开始
--------

本章节将详细讲解如何使用 Riemann 搭建、训练和评估一个完整的神经网络，包括数据集准备、网络构建、训练过程和推理评估等步骤。

数据集准备
~~~~~~~~~~

在使用 Riemann 构建神经网络之前，首先需要准备数据集。Riemann 提供了 `Dataset` 接口，用于定义数据加载和处理的标准方法。

Dataset 接口介绍
^^^^^^^^^^^^^^^^

`Dataset` 是一个抽象基类，用于表示数据集。要创建自定义数据集，你需要继承 `Dataset` 类并实现以下两个核心方法：

- ``__len__()``: 返回数据集的样本数量
- ``__getitem__(idx)``: 根据索引返回一个样本

构建自定义数据集
^^^^^^^^^^^^^^^^^^

以下是创建自定义数据集的详细示例：

.. code-block:: python

    import riemann as rm
    import numpy as np
    from riemann.utils.data import Dataset, DataLoader

    # 自定义数据集类
    class SimpleDataset(Dataset):
        def __init__(self, num_samples=1000):
            """
            初始化数据集
            
            :param num_samples: 数据集样本数量
            """
            # 生成随机输入数据
            self.inputs = rm.randn(num_samples, 10)
            # 生成对应的目标值（简单线性映射）
            weights = rm.randn(10, 2)
            biases = rm.randn(2)
            self.targets = self.inputs @ weights + biases
            
        def __len__(self):
            """
            返回数据集的样本数量
            """
            return len(self.inputs)
        
        def __getitem__(self, idx):
            """
            根据索引返回一个样本
            
            :param idx: 样本索引
            :return: 输入数据和目标值的元组
            """
            return self.inputs[idx], self.targets[idx]

    # 创建数据集实例
    train_dataset = SimpleDataset(1000)
    test_dataset = SimpleDataset(200)

    # 查看数据集信息
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 获取单个样本
    sample_input, sample_target = train_dataset[0]
    print(f"样本输入形状: {sample_input.shape}")
    print(f"样本目标形状: {sample_target.shape}")

高级数据集示例
^^^^^^^^^^^^^^^^

以下是一个更复杂的数据集示例，包含数据预处理和变换：

.. code-block:: python

    class ImageDataset(Dataset):
        def __init__(self, image_paths, labels, transform=None):
            """
            图像数据集
            
            :param image_paths: 图像路径列表
            :param labels: 标签列表
            :param transform: 数据变换函数（可选）
            """
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            # 这里应该是实际的图像加载代码
            # 为了示例，我们生成随机数据
            image = rm.randn(3, 32, 32)  # 模拟 3x32x32 的 RGB 图像
            label = self.labels[idx]
            
            # 应用数据变换
            if self.transform:
                image = self.transform(image)
            
            return image, label

使用 DataLoader
~~~~~~~~~~~~~~~

`DataLoader` 用于批量加载数据，支持多线程数据加载、数据打乱和自动批处理。

DataLoader 参数说明
^^^^^^^^^^^^^^^^^^^

`DataLoader` 接受以下主要参数：

- ``dataset``: 要加载的数据集实例
- ``batch_size``: 每个批次的样本数量，默认 1
- ``shuffle``: 是否在每个 epoch 开始时打乱数据，默认 False
- ``num_workers``: 用于数据加载的子进程数，默认 0（主进程加载）
- ``drop_last``: 如果数据集大小不能被批次大小整除，是否丢弃最后一个不完整的批次，默认 False
- ``pin_memory``: 是否将加载的数据复制到 CUDA 固定内存中，加速数据传输到 GPU，默认 False
- ``timeout``: 数据加载超时时间，默认 0
- ``worker_init_fn``: 每个工作进程初始化时调用的函数，默认 None
- ``multiprocessing_context``: 多进程上下文，默认 None

DataLoader 使用示例
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=2,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False,
        num_workers=1
    )

    # 遍历 DataLoader
    print("遍历训练数据加载器:")
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Input shape {inputs.shape}, Target shape {targets.shape}")
        if batch_idx == 2:  # 只打印前3个批次
            break

    # 在训练循环中使用
    print("\n在训练循环中使用:")
    num_epochs = 2
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 这里是训练代码
            if batch_idx % 10 == 0:  # 每10个批次打印一次
                print(f"  Batch {batch_idx}/{len(train_loader)}")
            # 实际训练中，这里会执行前向传播、损失计算、反向传播等
            break  # 为了示例，只执行一个批次

使用 pin_memory 加速
^^^^^^^^^^^^^^^^^^^^

如果使用 GPU 训练，可以启用 `pin_memory` 来加速数据传输：

.. code-block:: python

    # 为 GPU 训练优化的 DataLoader
    gpu_train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True  # 启用固定内存
    )

    # 在训练循环中
    if rm.cuda.is_available():
        device = rm.device('cuda')
        for inputs, targets in gpu_train_loader:
            # 数据已经在固定内存中，传输到 GPU 会更快
            inputs, targets = inputs.to(device), targets.to(device)
            # 训练步骤...

构建神经网络
~~~~~~~~~~~~~

神经网络是由多个层组成的模型，用于学习数据中的模式。在 Riemann 中，我们使用 ``nn.Module`` 类来构建神经网络。

神经网络构建步骤:

1. **导入必要的模块**：导入 ``riemann.nn`` 模块，它包含了各种网络层和激活函数
2. **定义网络类**：继承 ``nn.Module`` 类
3. **初始化网络层**：在 ``__init__`` 方法中定义网络的各个层
4. **定义前向传播**：在 ``forward`` 方法中定义数据如何通过网络流动
5. **创建网络实例**：实例化定义的网络类

基础网络构建示例
^^^^^^^^^^^^^^^^^

.. code-block:: python

    import riemann.nn as nn

    class SimpleNet(nn.Module):
        def __init__(self):
            """
            初始化简单的全连接神经网络
            
            网络结构：
            - 输入层：10 个特征
            - 隐藏层 1：50 个神经元，使用 ReLU 激活函数
            - 隐藏层 2：20 个神经元，使用 ReLU 激活函数
            - 输出层：2 个神经元（适用于回归任务）
            """
            super(SimpleNet, self).__init__()
            # 定义网络层
            self.fc1 = nn.Linear(10, 50)  # 输入层到第一个隐藏层
            self.relu = nn.ReLU()          # 激活函数
            self.fc2 = nn.Linear(50, 20)   # 第一个隐藏层到第二个隐藏层
            self.fc3 = nn.Linear(20, 2)    # 第二个隐藏层到输出层
        
        def forward(self, x):
            """
            定义前向传播过程
            
            :param x: 输入数据，形状为 [batch_size, 10]
            :return: 输出数据，形状为 [batch_size, 2]
            """
            # 前向传播
            x = self.fc1(x)  # 通过第一个全连接层
            x = self.relu(x) # 应用 ReLU 激活函数
            x = self.fc2(x)  # 通过第二个全连接层
            x = self.relu(x) # 应用 ReLU 激活函数
            x = self.fc3(x)  # 通过输出层
            return x

    # 创建网络实例
    model = SimpleNet()
    print(model)  # 打印网络结构

分类网络示例
^^^^^^^^^^^^

对于分类任务，我们需要调整输出层和激活函数：

.. code-block:: python

    class ClassificationNet(nn.Module):
        def __init__(self, num_classes=10):
            """
            初始化分类神经网络
            
            :param num_classes: 分类任务的类别数量
            """
            super(ClassificationNet, self).__init__()
            self.fc1 = nn.Linear(10, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, num_classes)  # 输出层大小等于类别数
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            # 注意：对于分类任务，我们通常在损失函数中处理激活函数
            # 使用 CrossEntropyLoss 时，不需要在这里应用 softmax
            return x

使用优化器
~~~~~~~~~~~

优化器用于根据损失函数的梯度更新网络参数，从而使模型逐渐学习到更好的表示。

常用优化器
^^^^^^^^^^

Riemann 提供了多种优化器，每种都有其特点和适用场景：

- **SGD**：随机梯度下降，基础优化器
- **Adam**：自适应矩估计，结合了动量和自适应学习率
- **RMSprop**：均方根传播，适用于递归神经网络
- **Adagrad**：自适应学习率，适用于稀疏数据

优化器使用示例
^^^^^^^^^^^^^^^

.. code-block:: python

    from riemann.optim import SGD, Adam, RMSprop

    # 创建 SGD 优化器
    # lr: 学习率，控制参数更新的步长
    # momentum: 动量，加速优化过程
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 或者使用 Adam 优化器
    # betas: 用于计算梯度和梯度平方的移动平均值的系数
    # weight_decay: 权重衰减，用于正则化
    # optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.0001)

    # 或者使用 RMSprop 优化器
    # optimizer = RMSprop(model.parameters(), lr=0.001, alpha=0.99, eps=1e-08)

学习率调度
^^^^^^^^^^

学习率是一个重要的超参数，通常需要随着训练的进行而调整：

.. code-block:: python

    # 简单的学习率调度示例
    initial_lr = 0.01
    optimizer = SGD(model.parameters(), lr=initial_lr, momentum=0.9)

    # 在训练过程中调整学习率
    for epoch in range(num_epochs):
        # 每 5 个 epoch 学习率减半
        if epoch % 5 == 0 and epoch > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
        
        # 训练代码...
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

定义损失函数
~~~~~~~~~~~~~

损失函数用于衡量模型预测与真实值之间的差异，是模型优化的目标。

损失函数选择指南
^^^^^^^^^^^^^^^^^

.. list-table:: 损失函数选择指南
   :widths: 20 30 50
   :header-rows: 1

   * - 任务类型
     - 推荐损失函数
     - 适用场景
   * - 回归任务
     - MSELoss
     - 预测连续值，如房价、温度等
   * - 回归任务
     - L1Loss
     - 对异常值不敏感的回归任务
   * - 回归任务
     - HuberLoss
     - 结合 MSE 和 L1 的优点，对异常值鲁棒
   * - 分类任务
     - CrossEntropyLoss
     - 多分类任务，输出为类别概率
   * - 分类任务
     - BCEWithLogitsLoss
     - 二分类任务，输出为0或1的概率

损失函数使用示例
^^^^^^^^^^^^^^^^^

.. code-block:: python

    import riemann.nn as nn

    # 对于回归任务
    # MSELoss: 均方误差损失，计算预测值与真实值之差的平方的平均值
    criterion = nn.MSELoss()

    # 对于分类任务
    # CrossEntropyLoss: 交叉熵损失，结合了 log_softmax 和 nll_loss
    # criterion = nn.CrossEntropyLoss()

    # 对于二分类任务
    # BCEWithLogitsLoss: 带 logits 的二元交叉熵损失
    # criterion = nn.BCEWithLogitsLoss()

    # 对于对异常值敏感的回归任务
    # HuberLoss: Huber 损失，在误差较小时使用 MSE，误差较大时使用 L1
    # criterion = nn.HuberLoss(delta=1.0)

训练网络
~~~~~~~~

训练网络是一个迭代过程，包括前向传播、损失计算、反向传播和参数更新四个主要步骤。

完整训练循环详解
^^^^^^^^^^^^^^^^^

.. code-block:: python

    num_epochs = 10  # 训练轮数
    
    for epoch in range(num_epochs):
        # 设置模型为训练模式
        # 这会启用 dropout 和 batch normalization 等训练特有的行为
        model.train()
        
        running_loss = 0.0  # 累计损失
        
        # 遍历数据加载器
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 1. 清零梯度
            # 每次迭代前必须清零梯度，否则梯度会累积
            optimizer.zero_grad()
            
            # 2. 前向传播
            # 将输入数据通过网络，得到预测值
            outputs = model(inputs)
            
            # 3. 计算损失
            # 衡量预测值与真实值之间的差异
            loss = criterion(outputs, targets)
            
            # 4. 反向传播
            # 计算损失对所有可学习参数的梯度
            loss.backward()
            
            # 5. 更新参数
            # 根据计算出的梯度更新网络参数
            optimizer.step()
            
            # 累加损失
            running_loss += loss.item()
            
            # 打印批次信息
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # 计算并打印每个 epoch 的平均损失
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

训练技巧
^^^^^^^^

1. **早停**：当验证损失不再下降时停止训练，防止过拟合
2. **正则化**：使用权重衰减、dropout 等方法防止过拟合
3. **批量归一化**：加速训练并提高模型稳定性
4. **梯度裁剪**：防止梯度爆炸，特别是在循环神经网络中
5. **混合精度训练**：使用半精度浮点数加速训练

推理与评估
~~~~~~~~~~~

模型训练完成后，需要在测试集上评估其性能，确保模型能够 generalization到未见数据。

模型评估步骤
^^^^^^^^^^^^

1. **设置模型为评估模式**：禁用 dropout 和 batch normalization 的训练行为
2. **使用 no_grad 上下文**：禁用梯度计算，节省内存和计算资源
3. **遍历测试数据**：计算模型在测试集上的性能指标
4. **计算评估指标**：根据任务类型选择合适的指标

评估示例
^^^^^^^^

.. code-block:: python

    # 设置模型为评估模式
    model.eval()
    
    # 评估指标
    test_loss = 0.0
    correct = 0
    total = 0
    
    # 使用 no_grad 上下文，禁用梯度计算
    with rm.no_grad():
        for inputs, targets in test_loader:
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            # 对于分类任务，计算准确率
            # _, predicted = rm.max(outputs, dim=1)
            # total += targets.size(0)
            # correct += (predicted == targets).sum().item()
    
    # 计算平均损失
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    
    # 对于分类任务，计算准确率
    # accuracy = 100 * correct / total
    # print(f"Test Accuracy: {accuracy:.2f}%")

保存和加载模型
^^^^^^^^^^^^^^

训练好的模型可以保存到磁盘，以便后续使用：

.. code-block:: python

    # 保存模型
    rm.save(model.state_dict(), 'model.pth')
    print("Model saved successfully!")

    # 加载模型
    # 创建模型实例
    loaded_model = SimpleNet()
    # 加载保存的参数
    loaded_model.load_state_dict(rm.load('model.pth'))
    # 设置为评估模式
    loaded_model.eval()
    print("Model loaded successfully!")

    # 使用加载的模型进行推理
    with rm.no_grad():
        # 示例输入
        sample_input = rm.randn(1, 10)
        # 模型预测
        prediction = loaded_model(sample_input)
        print(f"Sample prediction: {prediction}")

其它说明
--------

使用 CUDA
~~~~~~~~~

如果系统支持 CUDA，可以将模型和数据移至 GPU 上运行：

.. code-block:: python

    # 检查 CUDA 是否可用
    if rm.cuda.is_available():
        device = rm.device('cuda')
        print("Using CUDA")
    else:
        device = rm.device('cpu')
        print("Using CPU")

    # 将模型移至设备
    model.to(device)

    # 在训练循环中，将数据也移至设备
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        # 训练步骤...

神经网络基本组件
~~~~~~~~~~~~~~~~

- **输入层**：接收原始数据
- **隐藏层**：提取数据特征，层数和神经元数量决定了模型的表达能力
- **输出层**：产生最终预测结果
- **激活函数**：引入非线性，使网络能够学习复杂映射
- **损失函数**：衡量预测与真实值的差异
- **优化器**：根据损失函数的梯度更新网络参数

激活函数选择
~~~~~~~~~~~~~

- **ReLU**：适用于大多数场景，计算高效，缓解梯度消失问题
- **LeakyReLU**：解决 ReLU 的 "死亡神经元" 问题
- **Sigmoid**：适用于二分类任务的输出层
- **Softmax**：适用于多分类任务的输出层
- **Tanh**：输出范围在 [-1, 1]，比 Sigmoid 有更好的梯度特性
- **GELU**：在 Transformer 模型中表现优异

损失函数选择
~~~~~~~~~~~~~

- **MSELoss**：适用于回归任务
- **L1Loss**：对异常值不敏感，适用于某些回归任务
- **CrossEntropyLoss**：适用于多分类任务
- **BCEWithLogitsLoss**：适用于二分类任务
- **HuberLoss**：结合了 MSE 和 L1 的优点，对异常值鲁棒

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
     - 返回所有子模块（包括自身）的迭代器
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
     - 清空所有参数的梯度
     - ``model.zero_grad()``
   * - ``requires_grad_(requires_grad=True)``
     - 设置参数是否需要计算梯度
     - ``model.requires_grad_(False)  # 冻结参数``
   * - ``state_dict(destination=None, prefix='', keep_vars=False)``
     - 返回模块的状态字典
     - ``state = model.state_dict()``
   * - ``register_parameter(name, param)``
     - 注册参数到模块
     - ``self.register_parameter('weight', Parameter(rm.randn(10, 5)))``
   * - ``register_buffer(name, tensor)``
     - 注册缓冲区到模块
     - ``self.register_buffer('running_mean', rm.zeros(10))``
   * - ``add_module(name, module)``
     - 显式添加子模块
     - ``self.add_module('linear', Linear(10, 5))``
   * - ``_get_name()``
     - 获取模块类名
     - ``print(model._get_name())  # 输出类名``
   * - ``register_parameters_batch(**parameters)``
     - 批量注册参数
     - ``self.register_parameters_batch(weight=Parameter(rm.randn(10, 5)), bias=Parameter(rm.zeros(5)))``
   * - ``register_buffers_batch(**buffers)``
     - 批量注册缓冲区
     - ``self.register_buffers_batch(running_mean=rm.zeros(10), running_var=rm.ones(10))``
   * - ``clear_cache()``
     - 清除属性访问缓存
     - ``model.clear_cache()``
   * - ``enable_cache(enabled=True)``
     - 启用或禁用属性缓存
     - ``model.enable_cache(False)``

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

Parameter 类
------------

``Parameter`` 类用于包装张量，使其成为模块的可学习参数：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    class CustomLayer(nn.Module):
        def __init__(self, in_features, out_features):
            super(CustomLayer, self).__init__()
            # 创建可学习参数
            self.weight = nn.Parameter(rm.randn(out_features, in_features))
            self.bias = nn.Parameter(rm.zeros(out_features))
        
        def forward(self, x):
            return x @ self.weight.T + self.bias

    # 使用自定义层
    layer = CustomLayer(10, 5)
    print(layer.weight.shape)  # (5, 10)
    print(layer.bias.shape)    # (5,)

容器类
------

Riemann 提供了几种容器类来组织和管理模块：

Sequential
----------

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
    
    # 方法2：使用关键字参数（PyTorch 风格）
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
----------

``ModuleList`` 容器存储模块列表，允许通过索引访问，适用于需要动态控制前向传播的场景：

**参数**：
- ``modules``：模块列表（可选）

**主要方法**：
- ``append(module)``：添加模块
- ``extend(modules)``：扩展模块列表
- ``insert(index, module)``：插入模块

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
----------

``ModuleDict`` 容器使用字典存储模块，允许通过键访问，适用于需要根据条件选择不同模块的场景：

**参数**：
- ``modules``：模块字典（可选）

**主要方法**：
- ``update(modules)``：更新模块字典
- ``pop(key)``：移除并返回指定键的模块

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
    x = layers['dropout'](x)  # 使用新添加的模块
    x = layers['linear2'](x)
    
    print(x.shape)  # [32, 5]

容器类的选择
-----------------

- **Sequential**：适用于简单的线性网络，代码简洁
- **ModuleList**：适用于需要动态调整模块顺序或数量的场景
- **ModuleDict**：适用于需要根据条件选择不同模块的场景

混合使用容器类
----------------

可以根据网络结构的复杂度，混合使用不同的容器类：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    class ComplexNetwork(nn.Module):
        def __init__(self):
            super(ComplexNetwork, self).__init__()
            
            # 使用 Sequential 定义特征提取器
            self.feature_extractor = nn.Sequential(
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, 25),
                nn.ReLU()
            )
            
            # 使用 ModuleList 定义多个分类头
            self.classifiers = nn.ModuleList([
                nn.Linear(25, 10),  # 分类任务1
                nn.Linear(25, 5),   # 分类任务2
                nn.Linear(25, 1)    # 回归任务
            ])
            
            # 使用 ModuleDict 定义不同的激活函数
            self.activations = nn.ModuleDict({
                'relu': nn.ReLU(),
                'sigmoid': nn.Sigmoid(),
                'softmax': nn.Softmax(dim=1)
            })
        
        def forward(self, x, task_type):
            x = self.feature_extractor(x)
            
            if task_type == 'classification1':
                x = self.classifiers[0](x)
                x = self.activations['softmax'](x)
            elif task_type == 'classification2':
                x = self.classifiers[1](x)
                x = self.activations['softmax'](x)
            elif task_type == 'regression':
                x = self.classifiers[2](x)
                x = self.activations['relu'](x)
            
            return x

    # 使用混合容器网络
    model = ComplexNetwork()
    x = rm.randn(32, 100)
    
    # 执行分类任务1
    output1 = model(x, 'classification1')
    print(f"Classification1 output shape: {output1.shape}")  # [32, 10]
    
    # 执行回归任务
    output3 = model(x, 'regression')
    print(f"Regression output shape: {output3.shape}")      # [32, 1]

激活函数
========

激活函数是神经网络中的重要组成部分，它们引入非线性特性，使网络能够学习复杂的函数映射。Riemann 提供了多种激活函数，适用于不同的场景和任务。

激活函数列表
--------------

.. list-table:: Riemann 支持的激活函数
   :widths: 15 20 25 25 15
   :header-rows: 1

   * - 函数名
     - 描述
     - 适用场景
     - 参数含义
     - 注意事项
   * - ``ReLU``
     - 修正线性单元，输出 max(0, x)
     - 大多数深度学习模型的默认选择
     - 无参数
     - 可能产生 "死亡神经元" 问题
   * - ``LeakyReLU``
     - 带泄漏的 ReLU，负区间有小斜率
     - 解决 ReLU 的死亡神经元问题
     - ``negative_slope``: 负区间斜率，默认 0.01
     - 计算开销略高于 ReLU
   * - ``RReLU``
     - 随机泄漏 ReLU，训练时斜率随机
     - 作为正则化手段，防止过拟合
     - ``lower``: 斜率下限，默认 1/8
       ``upper``: 斜率上限，默认 1/3
       ``training``: 是否为训练模式
     - 测试时使用固定斜率
   * - ``PReLU``
     - 参数化 ReLU，斜率可学习
     - 需要学习负区间斜率的场景
     - ``num_parameters``: 可学习参数数量，默认 1
       ``init``: 初始斜率值，默认 0.25
     - 可能导致过拟合，需要谨慎使用
   * - ``Sigmoid``
     - S 型激活函数，输出 (0, 1)
     - 二分类任务的输出层
     - 无参数
     - 存在梯度消失问题
   * - ``Tanh``
     - 双曲正切函数，输出 (-1, 1)
     - RNN 等序列模型
     - 无参数
     - 仍存在梯度消失问题，但比 Sigmoid 轻
   * - ``Softmax``
     - 归一化指数函数，输出概率分布
     - 多分类任务的输出层
     - ``dim``: 计算维度，默认 -1
     - 通常与交叉熵损失一起使用
   * - ``LogSoftmax``
     - Softmax 的对数形式
     - 与 NLLLoss 一起使用，提高数值稳定性
     - ``dim``: 计算维度，默认 -1
     - 输出为对数概率
   * - ``GELU``
     - 高斯误差线性单元
     - Transformer 模型的默认选择
     - 无参数
     - 计算开销较高
   * - ``Softplus``
     - ReLU 的平滑近似
     - 需要平滑激活函数的场景
     - ``beta``: 曲线陡峭度，默认 1.0
       ``threshold``: 线性近似阈值，默认 20.0
     - 计算开销较高

激活函数使用示例
------------------

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建各种激活函数
    relu = nn.ReLU()
    leaky_relu = nn.LeakyReLU(negative_slope=0.01)
    prelu = nn.PReLU(num_parameters=1)
    sigmoid = nn.Sigmoid()
    tanh = nn.Tanh()
    softmax = nn.Softmax(dim=1)
    gelu = nn.GELU()
    log_softmax = nn.LogSoftmax(dim=1)
    softplus = nn.Softplus(beta=1.0)
    rrelu = nn.RReLU(lower=0.1, upper=0.3)
    
    # 测试输入
    x = rm.randn(4, 10)
    
    # 使用激活函数
    output_relu = relu(x)
    output_leaky = leaky_relu(x)
    output_prelu = prelu(x)
    output_sigmoid = sigmoid(x)
    output_tanh = tanh(x)
    output_softmax = softmax(x)
    output_gelu = gelu(x)
    output_log_softmax = log_softmax(x)
    output_softplus = softplus(x)
    output_rrelu = rrelu(x)
    
    # 验证输出形状
    print(f"ReLU output shape: {output_relu.shape}")  # [4, 10]
    print(f"Softmax output sum: {rm.sum(output_softmax, dim=1)}")  # 应接近 [1, 1, 1, 1]

损失函数
========

损失函数用于衡量模型预测值与真实目标值之间的差异，是模型训练的核心组成部分。Riemann 提供了多种损失函数，适用于不同类型的任务。

损失函数列表
------------

.. list-table:: Riemann 支持的损失函数
   :widths: 15 20 25 25 15
   :header-rows: 1

   * - 函数名
     - 描述
     - 适用场景
     - 参数含义
     - 注意事项
   * - ``MSELoss``
     - 均方误差损失
     - 回归任务
     - ``size_average``: 已弃用
       ``reduce``: 已弃用
       ``reduction``: 聚合方式，默认 'mean'
     - 对异常值敏感
   * - ``L1Loss``
     - L1 损失（绝对误差）
     - 对异常值不敏感的回归任务
     - ``size_average``: 已弃用
       ``reduce``: 已弃用
       ``reduction``: 聚合方式，默认 'mean'
     - 梯度在原点不连续
   * - ``SmoothL1Loss``
     - 平滑 L1 损失，结合 MSE 和 L1 的优点
     - 目标检测等任务
     - ``size_average``: 已弃用
       ``reduce``: 已弃用
       ``reduction``: 聚合方式，默认 'mean'
       ``beta``: 平滑阈值，默认 1.0
     - 计算开销适中
   * - ``CrossEntropyLoss``
     - 交叉熵损失，结合了 log_softmax 和 nll_loss
     - 多分类任务
     - ``weight``: 类别权重
       ``size_average``: 已弃用
       ``ignore_index``: 忽略的目标值，默认 -100
       ``reduce``: 已弃用
       ``reduction``: 聚合方式，默认 'mean'
       ``label_smoothing``: 标签平滑程度，默认 0.0
     - 输入为原始 logits，不需要 softmax
   * - ``BCEWithLogitsLoss``
     - 带 logits 的二元交叉熵损失
     - 二分类任务
     - ``weight``: 样本权重
       ``size_average``: 已弃用
       ``reduce``: 已弃用
       ``reduction``: 聚合方式，默认 'mean'
       ``pos_weight``: 正类权重
     - 输入为原始 logits，不需要 sigmoid
   * - ``HuberLoss``
     - Huber 损失，对异常值鲁棒
     - 对异常值敏感的回归任务
     - ``delta``: 阈值，默认 1.0
       ``size_average``: 已弃用
       ``reduce``: 已弃用
       ``reduction``: 聚合方式，默认 'mean'
     - 计算开销适中
   * - ``NLLLoss``
     - 负对数似然损失
     - 与 LogSoftmax 一起使用的分类任务
     - ``weight``: 类别权重
       ``size_average``: 已弃用
       ``ignore_index``: 忽略的目标值，默认 -100
       ``reduce``: 已弃用
       ``reduction``: 聚合方式，默认 'mean'
     - 输入为对数概率

损失函数使用示例
------------------

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建各种损失函数
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    smooth_l1_loss = nn.SmoothL1Loss(beta=1.0)
    cross_entropy_loss = nn.CrossEntropyLoss()
    bce_with_logits_loss = nn.BCEWithLogitsLoss()
    huber_loss = nn.HuberLoss(delta=1.0)
    nll_loss = nn.NLLLoss()
    
    # 回归任务测试数据
    reg_preds = rm.randn(4, 1)
    reg_targets = rm.randn(4, 1)
    
    # 分类任务测试数据
    cls_preds = rm.randn(4, 10)
    cls_targets = rm.randint(0, 10, (4,))
    
    # 二分类任务测试数据
    binary_preds = rm.randn(4, 1)
    binary_targets = rm.randint(0, 2, (4, 1)).float()
    
    # 计算各种损失
    loss_mse = mse_loss(reg_preds, reg_targets)
    loss_l1 = l1_loss(reg_preds, reg_targets)
    loss_smooth_l1 = smooth_l1_loss(reg_preds, reg_targets)
    loss_ce = cross_entropy_loss(cls_preds, cls_targets)
    loss_bce = bce_with_logits_loss(binary_preds, binary_targets)
    loss_huber = huber_loss(reg_preds, reg_targets)
    
    # 计算 NLLLoss（需要先计算 log_softmax）
    log_softmax = nn.LogSoftmax(dim=1)
    logits = log_softmax(cls_preds)
    loss_nll = nll_loss(logits, cls_targets)
    
    # 打印损失值
    print(f"MSE Loss: {loss_mse.item():.4f}")
    print(f"L1 Loss: {loss_l1.item():.4f}")
    print(f"Cross Entropy Loss: {loss_ce.item():.4f}")
    print(f"BCE With Logits Loss: {loss_bce.item():.4f}")

基本网络层
==========

基本网络层是构建神经网络的基础组件，包括全连接层、dropout 层、展平层等。这些层在各种神经网络架构中都有广泛的应用。

线性层（Linear）
----------------

线性层（也称为全连接层）对输入数据执行仿射变换：

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
    x = rm.randn(32, 20)  # 32 个样本的批次
    output = linear(x)
    print(output.shape)  # [32, 10]

Dropout 层
----------

Dropout 层通过随机失活神经元来防止过拟合：

**参数**：
- ``p``: 失活概率，默认 0.5

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建 dropout 层
    dropout = nn.Dropout(p=0.5)
    
    # 前向传播（训练模式）
    x = rm.randn(4, 16)
    output_train = dropout(x)
    
    # 前向传播（评估模式）
    dropout.eval()
    output_eval = dropout(x)
    
    print(output_train.shape)  # [4, 16]
    print(output_eval.shape)   # [4, 16]

展平层（Flatten）
------------------

展平层将多维张量展平为二维张量（批次维度保持不变）：

**参数**：
- 无参数

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建展平层
    flatten = nn.Flatten()
    
    # 前向传播
    x = rm.randn(4, 16, 8, 8)  # 4 个样本，16 通道，8x8 特征图
    output = flatten(x)
    print(output.shape)  # [4, 1024] (16*8*8)

批量归一化层（BatchNorm1d）
---------------------------

批量归一化层对输入进行归一化，加速训练并提高模型稳定性：

**参数**：
- ``num_features``: 特征数量
- ``eps``: 数值稳定性参数，默认 1e-5
- ``momentum``: 动量参数，默认 0.1
- ``affine``: 是否使用可学习的仿射参数，默认 True
- ``track_running_stats``: 是否跟踪运行统计信息，默认 True

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建批量归一化层
    batch_norm = nn.BatchNorm1d(num_features=16)
    
    # 前向传播
    x = rm.randn(4, 16)  # 4 个样本，16 特征
    output = batch_norm(x)
    print(output.shape)  # [4, 16]

卷积网络相关模块
================

卷积网络是处理图像、语音等网格结构数据的强大工具。Riemann 提供了丰富的卷积和池化层，支持 1D、2D 和 3D 数据。

卷积层
------

卷积层通过滑动窗口提取局部特征，是卷积神经网络的核心组件。

Conv1d
~~~~~~

一维卷积层，适用于序列数据如音频、文本等：

**参数**：

- ``in_channels``: 输入通道数
- ``out_channels``: 输出通道数
- ``kernel_size``: 卷积核大小
- ``stride``: 步长，默认 1
- ``padding``: 填充，默认 0
- ``dilation``: 膨胀率，默认 1
- ``groups``: 分组卷积组数，默认 1
- ``bias``: 是否使用偏置，默认 True

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建一维卷积层
    conv1d = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
    
    # 前向传播
    x = rm.randn(10, 16, 50)  # [batch_size, channels, length]
    output = conv1d(x)
    print(output.shape)  # [10, 32, 50] (有填充)

Conv2d
~~~~~~

二维卷积层，适用于图像数据：

**参数**：

- ``in_channels``: 输入通道数
- ``out_channels``: 输出通道数
- ``kernel_size``: 卷积核大小
- ``stride``: 步长，默认 1
- ``padding``: 填充，默认 0
- ``dilation``: 膨胀率，默认 1
- ``groups``: 分组卷积组数，默认 1
- ``bias``: 是否使用偏置，默认 True

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建二维卷积层
    conv2d = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    
    # 前向传播
    x = rm.randn(4, 3, 32, 32)  # [batch_size, channels, height, width]
    output = conv2d(x)
    print(output.shape)  # [4, 16, 32, 32] (有填充)

Conv3d
~~~~~~

三维卷积层，适用于 3D 数据如视频、医学影像等：

**参数**：

- ``in_channels``: 输入通道数
- ``out_channels``: 输出通道数
- ``kernel_size``: 卷积核大小
- ``stride``: 步长，默认 1
- ``padding``: 填充，默认 0
- ``dilation``: 膨胀率，默认 1
- ``groups``: 分组卷积组数，默认 1
- ``bias``: 是否使用偏置，默认 True

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建三维卷积层
    conv3d = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    
    # 前向传播
    x = rm.randn(2, 3, 16, 16, 16)  # [batch_size, channels, depth, height, width]
    output = conv3d(x)
    print(output.shape)  # [2, 16, 16, 16, 16] (有填充)

池化层
------

池化层用于减少特征图的空间维度，同时保留重要信息。

MaxPool1d
~~~~~~~~~

一维最大池化层：

**参数**：

- ``kernel_size``: 池化核大小
- ``stride``: 步长，默认 kernel_size
- ``padding``: 填充，默认 0
- ``dilation``: 膨胀率，默认 1
- ``ceil_mode``: 是否使用向上取整，默认 False
- ``return_indices``: 是否返回最大值索引，默认 False

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建一维最大池化层
    max_pool1d = nn.MaxPool1d(kernel_size=2, stride=2)
    
    # 前向传播
    x = rm.randn(4, 16, 50)  # [batch_size, channels, length]
    output = max_pool1d(x)
    print(output.shape)  # [4, 16, 25]

MaxPool2d
~~~~~~~~~

二维最大池化层：

**参数**：

- ``kernel_size``: 池化核大小
- ``stride``: 步长，默认 kernel_size
- ``padding``: 填充，默认 0
- ``dilation``: 膨胀率，默认 1
- ``ceil_mode``: 是否使用向上取整，默认 False
- ``return_indices``: 是否返回最大值索引，默认 False

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建二维最大池化层
    max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
    
    # 前向传播
    x = rm.randn(4, 16, 32, 32)  # [batch_size, channels, height, width]
    output = max_pool2d(x)
    print(output.shape)  # [4, 16, 16, 16]

MaxPool3d
~~~~~~~~~

三维最大池化层：

**参数**：

- ``kernel_size``: 池化核大小
- ``stride``: 步长，默认 kernel_size
- ``padding``: 填充，默认 0
- ``dilation``: 膨胀率，默认 1
- ``ceil_mode``: 是否使用向上取整，默认 False
- ``return_indices``: 是否返回最大值索引，默认 False

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建三维最大池化层
    max_pool3d = nn.MaxPool3d(kernel_size=2, stride=2)
    
    # 前向传播
    x = rm.randn(2, 16, 16, 16, 16)  # [batch_size, channels, depth, height, width]
    output = max_pool3d(x)
    print(output.shape)  # [2, 16, 8, 8, 8]

AvgPool1d
~~~~~~~~~

一维平均池化层：

**参数**：

- ``kernel_size``: 池化核大小
- ``stride``: 步长，默认 kernel_size
- ``padding``: 填充，默认 0
- ``ceil_mode``: 是否使用向上取整，默认 False
- ``count_include_pad``: 是否包含填充值，默认 True
- ``divisor_override``: 自定义除数，默认 None

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建一维平均池化层
    avg_pool1d = nn.AvgPool1d(kernel_size=2, stride=2)
    
    # 前向传播
    x = rm.randn(4, 16, 50)  # [batch_size, channels, length]
    output = avg_pool1d(x)
    print(output.shape)  # [4, 16, 25]

AvgPool2d
~~~~~~~~~

二维平均池化层：

**参数**：

- ``kernel_size``: 池化核大小
- ``stride``: 步长，默认 kernel_size
- ``padding``: 填充，默认 0
- ``ceil_mode``: 是否使用向上取整，默认 False
- ``count_include_pad``: 是否包含填充值，默认 True
- ``divisor_override``: 自定义除数，默认 None

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建二维平均池化层
    avg_pool2d = nn.AvgPool2d(kernel_size=2, stride=2)
    
    # 前向传播
    x = rm.randn(4, 16, 32, 32)  # [batch_size, channels, height, width]
    output = avg_pool2d(x)
    print(output.shape)  # [4, 16, 16, 16]

AvgPool3d
~~~~~~~~~

三维平均池化层：

**参数**：

- ``kernel_size``: 池化核大小
- ``stride``: 步长，默认 kernel_size
- ``padding``: 填充，默认 0
- ``ceil_mode``: 是否使用向上取整，默认 False
- ``count_include_pad``: 是否包含填充值，默认 True
- ``divisor_override``: 自定义除数，默认 None

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建三维平均池化层
    avg_pool3d = nn.AvgPool3d(kernel_size=2, stride=2)
    
    # 前向传播
    x = rm.randn(2, 16, 16, 16, 16)  # [batch_size, channels, depth, height, width]
    output = avg_pool3d(x)
    print(output.shape)  # [2, 16, 8, 8, 8]

Transformer
===========

Transformer 是一种基于自注意力机制的深度学习架构，最初用于自然语言处理任务，现已成为序列建模的主流方法。Riemann 提供了完整的 Transformer 组件，与 PyTorch 接口兼容。

MultiheadAttention
------------------

多头注意力机制，允许模型同时关注来自不同表示子空间的信息。

**功能描述**：
实现多头注意力机制，通过并行计算多组注意力权重来捕获输入序列的不同方面特征。

**参数**：

- ``embed_dim`` (int): 输入和输出向量的维度大小，必须能被 ``num_heads`` 整除
- ``num_heads`` (int): 多头注意力中使用的头部数量
- ``dropout`` (float, optional): 训练过程中对注意力权重应用的 dropout 概率，默认为 0.0
- ``bias`` (bool, optional): 是否在投影层中添加偏置项，默认为 True
- ``add_bias_kv`` (bool, optional): 是否在 key 和 value 序列的末尾添加可学习的偏置项，默认为 False
- ``add_zero_attn`` (bool, optional): 是否在注意力权重中添加一列零，默认为 False
- ``kdim`` (int, optional): key 向量的维度，默认为 None（使用 embed_dim）
- ``vdim`` (int, optional): value 向量的维度，默认为 None（使用 embed_dim）
- ``batch_first`` (bool, optional): 输入输出的形状格式，默认为 False（seq_len, batch_size, embed_dim）

**注意事项**：

- ``embed_dim`` 必须能被 ``num_heads`` 整除
- 当 ``batch_first=True`` 时，输入形状为 (batch_size, seq_len, embed_dim)
- 支持自注意力和交叉注意力两种模式

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建多头注意力层
    mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.1)
    
    # 自注意力模式
    query = rm.randn(10, 32, 512)  # [seq_len, batch_size, embed_dim]
    key = query
    value = query
    output, attn_weights = mha(query, key, value)
    print(output.shape)  # [10, 32, 512]
    
    # 交叉注意力模式
    query = rm.randn(10, 32, 512)  # 目标序列
    key = rm.randn(20, 32, 512)    # 源序列
    value = rm.randn(20, 32, 512)  # 源序列
    output, attn_weights = mha(query, key, value)
    print(output.shape)  # [10, 32, 512]
    
    # 使用 batch_first=True
    mha_bf = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
    query = rm.randn(32, 10, 512)  # [batch_size, seq_len, embed_dim]
    output, _ = mha_bf(query, query, query)
    print(output.shape)  # [32, 10, 512]

TransformerEncoderLayer
-----------------------

Transformer 编码器的单个层，由自注意力机制和前馈网络组成。

**功能描述**：
实现 Transformer 编码器的单个层，包含多头自注意力子层和前馈神经网络子层，每个子层后都有残差连接和层归一化。

**参数**：

- ``d_model`` (int): 输入和输出特征的维度大小
- ``nhead`` (int): 多头注意力中使用的头部数量
- ``dim_feedforward`` (int, optional): 前馈网络中隐藏层的维度大小，默认为 2048
- ``dropout`` (float, optional): 训练过程中对各层输出应用的 dropout 概率，默认为 0.1
- ``activation`` (str, optional): 前馈网络中使用的激活函数类型，'relu' 或 'gelu'，默认为 'relu'
- ``layer_norm_eps`` (float, optional): 层归一化中使用的 epsilon 值，默认为 1e-05
- ``batch_first`` (bool, optional): 输入输出的形状格式，默认为 False
- ``norm_first`` (bool, optional): 是否使用 Pre-LN 模式，默认为 False（Post-LN 模式）
- ``bias`` (bool, optional): 是否在所有线性层中添加偏置项，默认为 True

**注意事项**：

- ``norm_first=False`` 时使用 Post-LN 模式（原始 Transformer 论文）：先进行注意力/前馈计算，然后残差连接，最后层归一化
- ``norm_first=True`` 时使用 Pre-LN 模式：先层归一化，然后进行注意力/前馈计算，最后残差连接
- Pre-LN 模式通常训练更稳定

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建 Transformer 编码器层（Post-LN 模式）
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1
    )
    
    # 前向传播
    src = rm.randn(10, 32, 512)  # [seq_len, batch_size, d_model]
    output = encoder_layer(src)
    print(output.shape)  # [10, 32, 512]
    
    # 使用 Pre-LN 模式
    encoder_layer_prenorm = nn.TransformerEncoderLayer(
        d_model=512, nhead=8, norm_first=True
    )
    output = encoder_layer_prenorm(src)
    print(output.shape)  # [10, 32, 512]

TransformerDecoderLayer
-----------------------

Transformer 解码器的单个层，由自注意力机制、交叉注意力机制和前馈网络组成。

**功能描述**：
实现 Transformer 解码器的单个层，包含三个子层：掩码多头自注意力、多头交叉注意力和前馈神经网络，每个子层后都有残差连接和层归一化。

**参数**：

- ``d_model`` (int): 输入和输出特征的维度大小
- ``nhead`` (int): 多头注意力中使用的头部数量
- ``dim_feedforward`` (int, optional): 前馈网络中隐藏层的维度大小，默认为 2048
- ``dropout`` (float, optional): 训练过程中对各层输出应用的 dropout 概率，默认为 0.1
- ``activation`` (str, optional): 前馈网络中使用的激活函数类型，'relu' 或 'gelu'，默认为 'relu'
- ``layer_norm_eps`` (float, optional): 层归一化中使用的 epsilon 值，默认为 1e-05
- ``batch_first`` (bool, optional): 输入输出的形状格式，默认为 False
- ``norm_first`` (bool, optional): 是否使用 Pre-LN 模式，默认为 False
- ``bias`` (bool, optional): 是否在所有线性层中添加偏置项，默认为 True

**注意事项**：

- 解码器层需要同时接收目标序列（tgt）和编码器输出（memory）
- 自注意力使用掩码防止关注未来位置（用于自回归生成）
- 交叉注意力用于关注编码器输出的信息

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建 Transformer 解码器层
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1
    )
    
    # 前向传播
    tgt = rm.randn(20, 32, 512)     # [tgt_len, batch_size, d_model] 目标序列
    memory = rm.randn(10, 32, 512)  # [src_len, batch_size, d_model] 编码器输出
    output = decoder_layer(tgt, memory)
    print(output.shape)  # [20, 32, 512]

TransformerEncoder
------------------

由 N 个 TransformerEncoderLayer 层堆叠而成的 Transformer 编码器。

**功能描述**：
将输入序列通过多个编码器层进行特征提取，每个编码器层都包含自注意力机制和前馈网络。

**参数**：

- ``encoder_layer`` (TransformerEncoderLayer): 单个编码器层实例，将被克隆 num_layers 次
- ``num_layers`` (int): 编码器层的数量
- ``norm`` (Module, optional): 最后的层归一化层，默认为 None

**注意事项**：

- 编码器层会被深拷贝，因此传入的 encoder_layer 不会被修改
- 可以添加最终的层归一化来稳定输出

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建编码器层
    encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    
    # 创建编码器（6层）
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    
    # 前向传播
    src = rm.randn(10, 32, 512)  # [seq_len, batch_size, d_model]
    output = transformer_encoder(src)
    print(output.shape)  # [10, 32, 512]
    
    # 带最终层归一化的编码器
    encoder_norm = nn.LayerNorm(512)
    transformer_encoder_norm = nn.TransformerEncoder(
        encoder_layer, num_layers=6, norm=encoder_norm
    )
    output = transformer_encoder_norm(src)
    print(output.shape)  # [10, 32, 512]

TransformerDecoder
------------------

由 N 个 TransformerDecoderLayer 层堆叠而成的 Transformer 解码器。

**功能描述**：

将目标序列通过多个解码器层进行特征提取，每个解码器层都包含自注意力、交叉注意力和前馈网络。

**参数**：

- ``decoder_layer`` (TransformerDecoderLayer): 单个解码器层实例，将被克隆 num_layers 次
- ``num_layers`` (int): 解码器层的数量
- ``norm`` (Module, optional): 最后的层归一化层，默认为 None

**注意事项**：

- 解码器需要编码器的输出（memory）作为交叉注意力的输入
- 适用于序列到序列的生成任务

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建解码器层
    decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
    
    # 创建解码器（6层）
    transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    
    # 前向传播
    tgt = rm.randn(20, 32, 512)     # [tgt_len, batch_size, d_model] 目标序列
    memory = rm.randn(10, 32, 512)  # [src_len, batch_size, d_model] 编码器输出
    output = transformer_decoder(tgt, memory)
    print(output.shape)  # [20, 32, 512]

Transformer
-----------

完整的 Transformer 模型，包含编码器和解码器。

**功能描述**：
实现完整的 Transformer 架构，是编码器-解码器结构的端到端实现，适用于序列到序列的任务，如机器翻译、文本摘要等。

**参数**：

- ``d_model`` (int, optional): 编码器/解码器输入的特征维度，默认为 512
- ``nhead`` (int, optional): 多头注意力模型中的头数，默认为 8
- ``num_encoder_layers`` (int, optional): 编码器中子编码器层的数量，默认为 6
- ``num_decoder_layers`` (int, optional): 解码器中子解码器层的数量，默认为 6
- ``dim_feedforward`` (int, optional): 前馈网络模型的维度，默认为 2048
- ``dropout`` (float, optional): dropout 值，默认为 0.1
- ``activation`` (str, optional): 编码器/解码器中间层的激活函数，'relu' 或 'gelu'，默认为 'relu'
- ``custom_encoder`` (Module, optional): 自定义编码器，默认为 None
- ``custom_decoder`` (Module, optional): 自定义解码器，默认为 None
- ``layer_norm_eps`` (float, optional): 层归一化组件中的 eps 值，默认为 1e-05
- ``batch_first`` (bool, optional): 输入输出张量是否为 (batch, seq, feature) 格式，默认为 False
- ``norm_first`` (bool, optional): 是否在其他注意力和前馈操作之前执行 LayerNorm，默认为 False
- ``bias`` (bool, optional): Linear 和 LayerNorm 层是否学习加性偏置，默认为 True

**注意事项**：

- 如果提供了 ``custom_encoder`` 或 ``custom_decoder``，将使用自定义模块替代默认的编码器/解码器
- 完整的 Transformer 适用于序列到序列的任务
- 对于仅编码器任务（如 BERT），可以只使用 TransformerEncoder
- 对于仅解码器任务（如 GPT），可以只使用 TransformerDecoder

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建完整的 Transformer 模型
    transformer_model = nn.Transformer(
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1
    )
    
    # 前向传播
    src = rm.randn(10, 32, 512)     # [src_len, batch_size, d_model] 源序列
    tgt = rm.randn(20, 32, 512)     # [tgt_len, batch_size, d_model] 目标序列
    output = transformer_model(src, tgt)
    print(output.shape)  # [20, 32, 512]
    
    # 使用 batch_first=True
    transformer_model_bf = nn.Transformer(
        d_model=512, nhead=8, batch_first=True
    )
    src = rm.randn(32, 10, 512)     # [batch_size, src_len, d_model]
    tgt = rm.randn(32, 20, 512)     # [batch_size, tgt_len, d_model]
    output = transformer_model_bf(src, tgt)
    print(output.shape)  # [32, 20, 512]

示例
====

用于图像分类的简单 CNN
-----------------------

.. code-block:: python

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

用于时间序列预测的 Transformer
-------------------------------

.. code-block:: python

    # 本示例展示了如何使用 Transformer 模型进行时间序列预测
    # 包括数据准备、模型构建、训练和预测

    import riemann as rm
    import riemann.nn as nn
    from riemann.optim import Adam
    from riemann.utils.data import Dataset, DataLoader
    import numpy as np

    # 生成时间序列数据
    def generate_time_series_data(num_samples, seq_length, pred_length):
        """
        生成时间序列数据
        
        :param num_samples: 样本数量
        :param seq_length: 输入序列长度
        :param pred_length: 预测序列长度
        :return: 输入序列和目标序列
        """
        # 生成正弦波数据作为示例
        t = np.linspace(0, 100, num_samples + seq_length + pred_length)
        data = np.sin(t) + 0.1 * np.random.randn(len(t))
        
        inputs = []
        targets = []
        
        for i in range(num_samples):
            inputs.append(data[i:i+seq_length])
            targets.append(data[i+seq_length:i+seq_length+pred_length])
        
        return np.array(inputs), np.array(targets)

    # 自定义时间序列数据集
    class TimeSeriesDataset(Dataset):
        def __init__(self, num_samples=1000, seq_length=50, pred_length=10):
            self.inputs, self.targets = generate_time_series_data(
                num_samples, seq_length, pred_length
            )
            # 转换为 Riemann 张量
            self.inputs = rm.tensor(self.inputs, dtype=rm.float32)
            self.targets = rm.tensor(self.targets, dtype=rm.float32)
            
        def __len__(self):
            return len(self.inputs)
        
        def __getitem__(self, idx):
            return self.inputs[idx], self.targets[idx]

    # 简化的Transformer时间序列预测模型（仅使用编码器）
    class TransformerTimeSeriesModel(nn.Module):
        def __init__(self, input_dim=1, d_model=64, nhead=4, 
                     num_layers=2, dim_feedforward=128, pred_length=10):
            """
            Transformer 时间序列预测模型（简化版）
            
            仅使用Transformer编码器，将序列映射到预测值
            
            :param input_dim: 输入特征维度
            :param d_model: Transformer 模型维度
            :param nhead: 多头注意力头数
            :param num_layers: 编码器层数
            :param dim_feedforward: 前馈网络维度
            :param pred_length: 预测序列长度
            """
            super(TransformerTimeSeriesModel, self).__init__()
            
            self.d_model = d_model
            self.pred_length = pred_length
            
            # 输入嵌入层：将输入维度映射到 d_model 维度
            self.input_embedding = nn.Linear(input_dim, d_model)
            
            # 位置编码参数（可学习的位置编码）
            self.pos_encoding = nn.Parameter(rm.randn(100, d_model) * 0.01)
            
            # Transformer 编码器
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward,
                dropout=0.1,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, 
                num_layers=num_layers
            )
            
            # 输出层：将 d_model 维度映射到 pred_length * input_dim
            self.output_layer = nn.Linear(d_model, pred_length * input_dim)
            
        def forward(self, src):
            """
            前向传播
            
            :param src: 输入序列 [batch_size, src_len, input_dim]
            :return: 预测序列 [batch_size, pred_length, input_dim]
            """
            batch_size, src_len, input_dim = src.shape
            
            # 输入嵌入
            src = self.input_embedding(src)  # [batch_size, src_len, d_model]
            
            # 添加位置编码
            src = src + self.pos_encoding[:src_len, :].unsqueeze(0)
            
            # 编码器
            memory = self.transformer_encoder(src)  # [batch_size, src_len, d_model]
            
            # 取最后一个时间步的输出
            last_output = memory[:, -1, :]  # [batch_size, d_model]
            
            # 输出层
            output = self.output_layer(last_output)  # [batch_size, pred_length * input_dim]
            
            # 重塑为 [batch_size, pred_length, input_dim]
            output = output.view(batch_size, self.pred_length, input_dim)
            
            return output

    # 创建数据集和数据加载器
    train_dataset = TimeSeriesDataset(num_samples=1000, seq_length=50, pred_length=10)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 创建模型
    model = TransformerTimeSeriesModel(
        input_dim=1, 
        d_model=64, 
        nhead=4, 
        num_layers=2,
        dim_feedforward=128,
        pred_length=10
    )

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 添加特征维度
            inputs = inputs.unsqueeze(-1)  # [batch_size, seq_len, 1]
            targets = targets.unsqueeze(-1)  # [batch_size, pred_len, 1]
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    # 预测示例
    model.eval()
    with rm.no_grad():
        # 获取一个测试样本
        test_input, test_target = train_dataset[0]
        test_input = test_input.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
        
        # 进行预测
        prediction = model(test_input)
        
        print(f"\n输入序列形状: {test_input.shape}")
        print(f"预测序列形状: {prediction.shape}")
        print(f"真实目标形状: {test_target.shape}")
        
        # 计算预测误差
        test_target = test_target.unsqueeze(-1)
        error = rm.mean((prediction - test_target) ** 2).item()
        print(f"预测均方误差: {error:.6f}")
        
        # 打印目标值和预测值
        print("\n===== Prediction Results Comparison =====")
        print(f"{'Step':<10} {'Target':<15} {'Prediction':<15} {'Error':<15}")
        print("-" * 55)
        
        pred_values = prediction.squeeze().tolist()
        target_values = test_target.squeeze().tolist()
        
        for i in range(len(target_values)):
            target_val = target_values[i]
            pred_val = pred_values[i] if isinstance(pred_values, list) else pred_values
            diff = target_val - pred_val
            print(f"{i+1:<10} {target_val:<15.6f} {pred_val:<15.6f} {diff:<15.6f}")
        
        print("-" * 55)
