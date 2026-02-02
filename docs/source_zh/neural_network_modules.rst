神经网络模块
============

Riemann 通过 ``riemann.nn`` 包提供了一套全面的神经网络模块。这些模块是创建和训练神经网络的构建块。

快速开始
--------

本章节将详细讲解如何使用 Riemann 搭建、训练和评估一个完整的神经网络，包括数据集准备、网络构建、训练过程和推理评估等步骤。

数据集准备
~~~~~~~~~~

在使用 Riemann 构建神经网络之前，首先需要准备数据集。Riemann 支持与 PyTorch 类似的数据处理方式：

.. code-block:: python

    import riemann as rm
    import numpy as np
    from riemann.utils.data import Dataset, DataLoader

    # 自定义数据集类
    class SimpleDataset(Dataset):
        def __init__(self, num_samples=1000):
            # 生成随机输入数据
            self.inputs = rm.randn(num_samples, 10)
            # 生成对应的目标值（简单线性映射）
            weights = rm.randn(10, 2)
            biases = rm.randn(2)
            self.targets = self.inputs @ weights + biases
            
        def __len__(self):
            return len(self.inputs)
        
        def __getitem__(self, idx):
            return self.inputs[idx], self.targets[idx]

    # 创建数据集实例
    train_dataset = SimpleDataset(1000)
    test_dataset = SimpleDataset(200)

使用 DataLoader
~~~~~~~~~~~~~~~

DataLoader 用于批量加载数据，支持多线程数据加载和数据打乱：

.. code-block:: python

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False
    )

    # 遍历 DataLoader
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Input shape {inputs.shape}, Target shape {targets.shape}")
        break

构建神经网络
~~~~~~~~~~~~~

使用 Riemann 的 ``nn.Module`` 类和各种网络层来构建神经网络：

.. code-block:: python

    import riemann.nn as nn

    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            # 定义网络层
            self.fc1 = nn.Linear(10, 50)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(50, 20)
            self.fc3 = nn.Linear(20, 2)
        
        def forward(self, x):
            # 定义前向传播
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            return x

    # 创建网络实例
    model = SimpleNet()
    print(model)

使用优化器
~~~~~~~~~~~

Riemann 提供了各种优化器来更新网络参数：

.. code-block:: python

    from riemann.optim import SGD

    # 创建优化器
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 或者使用 Adam 优化器
    # from riemann.optim import Adam
    # optimizer = Adam(model.parameters(), lr=0.001)

定义损失函数
~~~~~~~~~~~~~

根据任务类型选择合适的损失函数：

.. code-block:: python

    # 对于回归任务
    criterion = nn.MSELoss()

    # 对于分类任务
    # criterion = nn.CrossEntropyLoss()

训练网络
~~~~~~~~

完整的训练循环：

.. code-block:: python

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()  # 设置为训练模式
        running_loss = 0.0
        
        for inputs, targets in train_loader:
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
            
            running_loss += loss.item()
        
        # 计算平均损失
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

推理与评估
~~~~~~~~~~~

在测试集上评估模型性能：

.. code-block:: python

    model.eval()  # 设置为评估模式
    test_loss = 0.0

    with rm.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")

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
------

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
------

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
------

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
---------

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
---------

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
---------

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
---------

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
---------

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
---------

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

示例
====

用于图像分类的简单 CNN
-----------------------

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super(SimpleCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.classifier = nn.Sequential(
                nn.Linear(32 * 8 * 8, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)  # 展平
            x = self.classifier(x)
            return x
    
    # 创建模型
    model = SimpleCNN(num_classes=10)
    
    # 前向传播
    x = rm.randn(4, 3, 32, 32)  # 4 张 RGB 图像的批次
    output = model(x)
    print(output.shape)  # [4, 10]

用于序列数据的简单 RNN
-----------------------

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    class SimpleRNN(nn.Module):
        def __init__(self, input_size=10, hidden_size=50, num_classes=2):
            super(SimpleRNN, self).__init__()
            self.hidden_size = hidden_size
            # 注意：Riemann 目前尚未实现 RNN 层，这里仅作为示例结构
            # 实际使用时需要使用现有层组合或等待官方实现
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            # 简单的前馈网络模拟
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    # 创建模型
    model = SimpleRNN()
    
    # 前向传播
    x = rm.randn(32, 10)  # 32 个样本，每个样本 10 个特征
    output = model(x)
    print(output.shape)  # [32, 2]
