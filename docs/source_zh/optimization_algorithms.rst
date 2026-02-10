优化器
======

Riemann 通过 ``riemann.optim`` 包提供了多种优化算法。这些优化器用于在训练过程中更新神经网络的参数。

优化器基础
----------

本节介绍了 Riemann 中优化器的基本概念和使用模式。

Riemann 中的所有优化器都继承自 ``optim.Optimizer`` 类。要使用优化器，您需要：

1. 创建一个优化器实例，指定要优化的参数
2. 定义损失函数
3. 清零梯度
4. 计算损失并调用 ``backward()``
5. 调用优化器的 ``step()`` 方法

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    import riemann.optim as optim
    
    # 创建一个简单的模型
    model = nn.Linear(10, 1)
    
    # 创建一个优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 定义损失函数
    loss_fn = nn.MSELoss()
    
    # 训练循环
    for epoch in range(num_epochs):
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()

GD (梯度下降)
--------------

GD 是最基本的优化算法，它沿着负梯度的方向更新参数。

基本 GD
~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # 创建模型
    model = nn.Linear(10, 1)
    
    # 创建 GD 优化器
    optimizer = optim.GD(model.parameters(), lr=0.01)
    
    # 训练循环
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

带权重衰减的 GD
~~~~~~~~~~~~~~~~

权重衰减（L2 正则化）可以添加以防止过拟合。

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # 创建带权重衰减的 GD 优化器
    optimizer = optim.GD(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    # 训练循环
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

SGD (随机梯度下降)
------------------

SGD 是梯度下降的一种变体，它在每次迭代中使用数据子集更新参数。它是机器学习中最基础的优化算法之一。

**算法说明：**

1. 在小批量数据上计算梯度：`∇θL(θ)`
2. 应用权重衰减：`θ = θ * (1 - η * weight_decay)`
3. 更新参数：`θ = θ - η * ∇θL(θ)`
4. 可选：应用动量和/或 Nesterov 动量

**优点：**

- 简单且计算效率高
- 实现直接明了
- 除学习率外无需调整其他超参数
- 与动量一起使用时对大型数据集有效

**适用场景：**

- 训练大型数据集上的大型模型
- 计算资源有限时
- 作为比较其他优化算法的基线
- 使用精心设计的学习率调度进行模型微调时

基本 SGD
~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # 创建模型
    model = nn.Linear(10, 1)
    
    # 创建 SGD 优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 训练循环
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

带动量的 SGD
~~~~~~~~~~~~

动量有助于加速 SGD 在相关方向上的前进，并抑制振荡。

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # 创建带动量的 SGD 优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # 训练循环
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

带 Nesterov 动量的 SGD
~~~~~~~~~~~~~~~~~~~~~~

Nesterov 动量是动量的一种变体，通过在动量向量方向上向前看，可以提供更好的收敛性。

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # 创建带 Nesterov 动量的 SGD 优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    
    # 训练循环
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

带 closure 参数的 SGD
~~~~~~~~~~~~~~~~~~~~~

SGD 优化器支持 ``closure`` 参数，这是一个重新评估模型并返回损失的函数。这在需要在每个优化步骤中多次计算损失的算法中很有用，或者用于自定义优化逻辑。

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # 创建 SGD 优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 定义 closure 函数
    def closure():
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        return loss
    
    # 使用 closure 的训练循环
    for epoch in range(num_epochs):
        loss = optimizer.step(closure)
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

何时使用带 closure 的 SGD
~~~~~~~~~~~~~~~~~~~~~~~~~~

closure 参数在以下场景中特别有用：

- 实现需要多次前向/反向传播的自定义优化逻辑
- 使用需要多次计算损失梯度的算法时
- 在不同优化器之间创建一致的接口（例如，LBFGS 总是需要 closure）
- 通过在不同阶段检查损失值来调试优化问题

Adam (自适应矩估计)
------------------------

Adam 结合了 AdaGrad 和 RMSProp 算法的最佳特性，提供了一个能够在噪声问题上处理稀疏梯度的优化算法。它同时维护梯度的移动平均值（动量）和梯度平方的移动平均值（自适应学习率）。

**算法说明：**

1. 计算一阶矩估计：`m = β1*m + (1-β1)*∇θL(θ)`
2. 计算二阶矩估计：`v = β2*v + (1-β2)*(∇θL(θ))²`
3. 应用偏差校正：`m̂ = m/(1-β1^t)`，`v̂ = v/(1-β2^t)`
4. 应用权重衰减：`θ = θ * (1 - η * weight_decay)`
5. 更新参数：`θ = θ - η*m̂/(√v̂ + ε)`

**优点：**

- 每个参数的自适应学习率
- 结合了动量和自适应学习率的优点
- 适合处理稀疏梯度和噪声问题
- 通常不需要大量超参数调优

**适用场景：**

- 训练具有许多参数的深度神经网络
- 具有稀疏梯度的问题（如 NLP 任务）
- 大规模机器学习问题
- 希望快速收敛时
- 作为大多数深度学习任务的默认优化器

基本 Adam
~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # 创建 Adam 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

带权重衰减的 Adam
~~~~~~~~~~~~~~~~~~

权重衰减（L2 正则化）可以添加以防止过拟合。

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # 创建带权重衰减的 Adam 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 训练循环
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

Adagrad
-------

AdaGrad 根据参数调整学习率，对与频繁出现特征相关的参数执行较小的更新。它维护一个基于历史平方梯度和的按参数缩放的学习率。

**算法说明：**

1. 计算梯度：`∇θL(θ)`
2. 累积平方梯度：`G = G + (∇θL(θ))²`
3. 应用权重衰减：`θ = θ * (1 - η * weight_decay)`
4. 更新参数：`θ = θ - η*∇θL(θ)/(√G + ε)`
5. 可选：应用学习率衰减

**优点：**

- 每个参数的自适应学习率
- 自动为不同特征缩放学习率
- 适合处理一些特征不频繁出现的稀疏数据
- 无需为不同参数手动调整学习率

**适用场景：**

- 在稀疏数据集上训练（如文本分类）
- 具有不同频率特征的问题
- 希望学习率能自动适应特征重要性时
- 作为自适应学习率方法的基线

基本 Adagrad
~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # 创建 AdaGrad 优化器
    optimizer = optim.Adagrad(model.parameters(), lr=0.01)
    
    # 训练循环
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

LBFGS
-----

LBFGS 是一种拟牛顿法，它使用有限的内存来近似 BFGS 算法。它维护最近梯度评估的历史记录来近似 Hessian 矩阵，使得优化比完整的牛顿法更高效。

**算法说明：**

1. 使用闭包函数计算梯度
2. 维护最近的（位置，梯度）对历史
3. 使用此历史近似逆 Hessian 矩阵
4. 基于近似逆 Hessian 计算搜索方向
5. 执行线搜索以找到最佳步长
6. 更新参数并重复

**优点：**

- 与完整 BFGS 相比内存效率更高
- 对于平滑目标函数，收敛速度快于一阶方法
- 使用二阶信息而无需显式计算 Hessian
- 适合中小型问题

**适用场景：**

- 训练中小型模型
- 具有平滑目标函数的问题
- 需要比一阶方法更快收敛时
- 批量优化，可计算完整梯度
- 内存有限但仍希望获得二阶优化优势时

**LBFGS 参数详细说明：**

- **lr**: 学习率（默认值：1.0）。与其他优化器不同，LBFGS 将其用作线搜索期间的步长乘数。
- **max_iter**: 每个优化步骤的最大迭代次数（默认值：20）。控制算法尝试找到最佳步长的次数。
- **max_eval**: 每个优化步骤的最大函数评估次数（默认值：None）。如果为 None，则默认为 max_iter * 1.25。
- **tolerance_grad**: 梯度收敛阈值（默认值：1e-05）。当梯度范数低于此阈值时，优化停止。
- **tolerance_change**: 参数变化收敛阈值（默认值：1e-09）。当参数更新低于此阈值时，优化停止。
- **history_size**: 历史缓冲区大小（默认值：100）。控制用于近似 Hessian 矩阵的内存量。
- **line_search_fn**: 线搜索函数（默认值：None）。用于替代默认实现的自定义线搜索函数。

基本 LBFGS
~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # 创建 LBFGS 优化器
    optimizer = optim.LBFGS(model.parameters(), lr=1.0)
    
    # 定义 LBFGS 的闭包函数
    def closure():
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        return loss
    
    # 训练循环
    for epoch in range(num_epochs):
        loss = optimizer.step(closure)
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

带自定义参数的 LBFGS
~~~~~~~~~~~~~~~~~~~~~

您可以自定义 LBFGS 参数以更好地适应特定的优化问题：

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # 创建带自定义参数的 LBFGS 优化器
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=1.0,               # 学习率（步长乘数）
        max_iter=50,          # 每个步骤的最大迭代次数
        max_eval=None,        # 最大函数评估次数（None = max_iter * 1.25）
        tolerance_grad=1e-06,  # 梯度收敛阈值
        tolerance_change=1e-09, # 参数变化收敛阈值
        history_size=100,      # Hessian 近似内存
        line_search_fn=None    # 使用默认线搜索
    )
    
    # 定义闭包函数
    def closure():
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        return loss
    
    # 训练循环
    for epoch in range(num_epochs):
        loss = optimizer.step(closure)
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

LBFGS 参数调优技巧
~~~~~~~~~~~~~~~~~~~

- **lr**: 从 1.0 开始，并根据收敛情况调整。对于病态问题，较小的值（0.1-0.5）可能更稳定。
- **max_iter**: 对于更复杂的优化问题，增加此值，但要注意计算时间的增加。
- **history_size**: 较大的值可能会改善收敛性，但会增加内存使用。对于小型模型，100-200 通常足够。
- **tolerance_grad** 和 **tolerance_change**: 根据所需的精度进行调整。较小的值会导致更精确的优化，但可能需要更多迭代。
- **line_search_fn**: 仅在有特定线搜索要求时自定义。默认实现对大多数问题效果良好。

AdamW（Adam with Weight Decay）
-------------------------------

AdamW 是 Adam 的改进版本，它将权重衰减作为独立的正则化项处理，而不是像 Adam 中那样修改梯度。这使得权重衰减能够更有效地作为 L2 正则化，避免了 Adam 中原有的权重衰减副作用。

**算法说明：**

1. 计算一阶矩估计：`m = β1*m + (1-β1)*∇θL(θ)`
2. 计算二阶矩估计：`v = β2*v + (1-β2)*(∇θL(θ))²`
3. 应用偏差校正：`m̂ = m/(1-β1^t)`，`v̂ = v/(1-β2^t)`
4. 应用权重衰减：`θ = θ * (1 - η * weight_decay)`
5. 更新参数：`θ = θ - η*m̂/(√v̂ + ε)`

**优点：**

- 权重衰减作为正则化器更有效
- 更好的泛化性能
- 避免了 Adam 中的权重衰减问题
- 适合现代深度学习架构

**适用场景：**

- 训练深度神经网络
- 迁移学习和微调
- 参数众多、容易过拟合的模型
- 任何使用 Adam 的场景，尤其是需要权重衰减的情况

基本 AdamW
~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # 创建 AdamW 优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # 训练循环
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

自定义 betas 的 AdamW
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # 创建带有自定义 betas 的 AdamW 优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
    
    # 训练循环
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

RMSprop（Root Mean Square Propagation）
---------------------------------------

RMSprop 是一种自适应学习率优化算法，它为每个参数维护一个梯度平方的移动平均值。它特别适合递归神经网络（RNN）和其他序列模型。

**算法说明：**

1. 计算梯度平方的移动平均值：`v = α*v + (1-α)*(∇θL(θ))²`
2. 更新参数：`θ = θ - η*∇θL(θ)/(√v + ε)`
3. 可选：应用动量和/或使用中心化版本

**优点：**

- 每个参数的自适应学习率
- 对非平稳目标函数有效
- 适合 RNN 和序列模型
- 有助于解决梯度消失/爆炸问题

**适用场景：**

- 训练递归神经网络（RNN）
- 训练长短期记忆（LSTM）网络
- 训练门控循环单元（GRU）
- 任何处理序列数据的模型
- 梯度可能随时间显著变化的模型

基本 RMSprop
~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # 创建 RMSprop 优化器
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99)
    
    # 训练循环
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

带动量的 RMSprop
~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # 创建带动量的 RMSprop 优化器
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, momentum=0.9)
    
    # 训练循环
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

带中心化梯度的 RMSprop
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # 创建带中心化梯度的 RMSprop 优化器
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, centered=True)
    
    # 训练循环
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

优化器方法
----------

清零梯度
~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 清零梯度
    optimizer.zero_grad()

步进
~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 执行单次优化步骤
    optimizer.step()

状态字典
~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 获取优化器状态
    state_dict = optimizer.state_dict()
    
    # 加载优化器状态
    optimizer.load_state_dict(state_dict)

参数组
------

参数组是 Riemann 优化器的一个强大特性，它允许您为模型中的不同参数集配置不同的超参数。这在许多场景中非常有用，例如：

- 为模型的不同层设置不同的学习率
- 为权重和偏置参数设置不同的超参数
- 在微调预训练模型时，为不同部分设置不同的学习率

参数组的基本结构
~~~~~~~~~~~~~~~~

参数组通过一个字典列表来定义，每个字典包含以下内容：

- `params`：要优化的参数集合
- 其他键值对：为该参数组指定的超参数（如 `lr`、`weight_decay` 等）

基本用法示例
~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # 创建模型
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    
    # 创建参数组
    optimizer = optim.SGD([
        {'params': model[0].parameters(), 'lr': 0.01},
        {'params': model[2].parameters(), 'lr': 0.001}
    ], momentum=0.9)
    
    # 训练循环
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

在上面的示例中：
- 第一个参数组包含模型的第一层（`model[0]`）的所有参数，学习率设置为 0.01
- 第二个参数组包含模型的第三层（`model[2]`）的所有参数，学习率设置为 0.001
- 两个参数组共享 `momentum=0.9` 这个超参数

为权重和偏置设置不同的超参数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # 创建模型
    model = nn.Linear(10, 1)
    
    # 分离权重和偏置参数
    weight_params = [p for name, p in model.named_parameters() if 'weight' in name]
    bias_params = [p for name, p in model.named_parameters() if 'bias' in name]
    
    # 创建参数组
    optimizer = optim.SGD([
        {'params': weight_params, 'lr': 0.01, 'weight_decay': 1e-4},
        {'params': bias_params, 'lr': 0.02, 'weight_decay': 0}
    ])

在这个示例中：
- 权重参数使用较小的学习率（0.01）和权重衰减（1e-4）
- 偏置参数使用较大的学习率（0.02）且没有权重衰减

在预训练模型中使用参数组
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # 创建预训练模型（简化示例）
    class PretrainedModel(nn.Module):
        def __init__(self):
            super(PretrainedModel, self).__init__()
            # 假设这部分是预训练的特征提取器
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            # 假设这部分是新添加的分类器
            self.classifier = nn.Linear(64, 10)
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    model = PretrainedModel()
    
    # 创建参数组
    optimizer = optim.SGD([
        {'params': model.features.parameters(), 'lr': 0.001},  # 预训练特征使用较小的学习率
        {'params': model.classifier.parameters(), 'lr': 0.01}  # 新分类器使用较大的学习率
    ], momentum=0.9)

在这个示例中：
- 预训练的特征提取器使用较小的学习率（0.001），以避免破坏已学习的特征
- 新添加的分类器使用较大的学习率（0.01），以加快其收敛速度

参数组的工作原理
~~~~~~~~~~~~~~~~

当您使用参数组创建优化器时：

1. 优化器会为每个参数组单独维护状态
2. 在每次 `step()` 调用时，优化器会根据每个参数组的超参数更新相应的参数
3. 如果某个超参数没有在参数组中指定，优化器会使用构造函数中提供的默认值

最佳实践
~~~~~~~~

1. **清晰命名**：使用 `named_parameters()` 来根据参数名称创建参数组，提高代码可读性
2. **合理分组**：根据参数的性质和重要性进行分组，例如：
   - 不同层的参数
   - 权重和偏置参数
   - 预训练和新添加的参数
3. **学习率调度**：参数组的学习率可以与学习率调度器一起使用，调度器会根据各自的初始学习率进行调整
4. **超参数搜索**：使用参数组可以更灵活地进行超参数搜索，为不同部分的参数找到最佳配置

梯度裁剪
--------

梯度裁剪是一种防止深度网络中梯度爆炸的技术，通过限制梯度的大小来确保训练过程的稳定性。在训练深层神经网络时，梯度可能会变得非常大，导致参数更新幅度过大，进而使训练过程发散。

梯度裁剪的作用
~~~~~~~~~~~~~~

- **防止梯度爆炸**：限制梯度的最大值，避免参数更新幅度过大
- **提高训练稳定性**：使训练过程更加平稳，减少训练波动
- **加快收敛速度**：在某些情况下可以帮助模型更快地收敛
- **允许使用更大的学习率**：通过限制梯度，可以使用更大的初始学习率

按范数裁剪
~~~~~~~~~~

按范数裁剪是通过计算梯度的L2范数，并将其限制在一个最大范数内来实现的。这种方法会保持梯度的方向不变，只调整其大小。

**参数说明**：
- `parameters`：要裁剪梯度的参数集合
- `max_norm`：梯度的最大范数
- `norm_type`：范数的类型，默认为2（L2范数）
- `error_if_nonfinite`：如果梯度包含非有限值（如NaN或inf），是否抛出错误，默认为False

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 按范数裁剪梯度
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

**使用场景**：
- 适用于大多数深度神经网络训练
- 特别是在使用RNN或LSTM等循环神经网络时
- 当你观察到训练损失出现NaN或inf时

**实际应用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    import riemann.optim as optim

    # 创建模型
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )

    # 创建优化器
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 训练循环
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            
            # 在优化器step前执行梯度裁剪
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

按值裁剪
~~~~~~~~

按值裁剪是通过将梯度的每个元素限制在一个指定的范围内来实现的。这种方法会直接截断梯度中的极端值。

**参数说明**：
- `parameters`：要裁剪梯度的参数集合
- `clip_value`：梯度的最大绝对值
- `error_if_nonfinite`：如果梯度包含非有限值（如NaN或inf），是否抛出错误，默认为False

**使用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 按值裁剪梯度
    nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

**使用场景**：
- 当你希望直接控制梯度的最大绝对值时
- 当梯度中存在极端异常值时
- 对于某些特定的网络架构，如GAN中的判别器

**实际应用示例**：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    import riemann.optim as optim

    # 创建模型
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )

    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 训练循环
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            
            # 在优化器step前执行梯度裁剪
            nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
            
            optimizer.step()

梯度裁剪的最佳实践
~~~~~~~~~~~~~~~~~~

1. **选择合适的裁剪方法**：
   - 按范数裁剪（``clip_grad_norm_``）：适用于大多数情况，特别是RNN
   - 按值裁剪（``clip_grad_value_``）：适用于梯度中有极端值的情况

2. **设置合理的裁剪阈值**：
   - 按范数裁剪：max_norm通常设置在0.5到5.0之间
   - 按值裁剪：clip_value通常设置在0.1到1.0之间

3. **裁剪时机**：
   - 必须在 ``loss.backward()`` 之后、``optimizer.step()`` 之前执行
   - 对于每个batch都应该执行梯度裁剪

4. **与其他技术结合**：
   - 与学习率调度器结合使用
   - 与参数组结合使用，可以为不同层设置不同的裁剪策略

5. **监控效果**：
   - 观察训练损失是否变得更加稳定
   - 检查是否还出现梯度爆炸的情况
   - 调整裁剪阈值以获得最佳效果

示例
----

使用 Adam 训练神经网络
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    import riemann.optim as optim
    
    # 创建模型
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss()
    
    # 训练循环
    for epoch in range(50):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            
            # 梯度裁剪
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}')

为不同层使用不同的学习率进行训练
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    import riemann.optim as optim
    
    # 创建预训练模型（简化示例）
    class PretrainedModel(nn.Module):
        def __init__(self):
            super(PretrainedModel, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Linear(64, 10)
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    model = PretrainedModel()
    
    # 创建具有不同学习率的参数组
    optimizer = optim.SGD([
        {'params': model.features.parameters(), 'lr': 0.001},  # 预训练特征使用较低的学习率
        {'params': model.classifier.parameters(), 'lr': 0.01}  # 新分类器使用较高的学习率
    ], momentum=0.9)
    
    # 训练循环
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

自定义优化器
------------

您可以通过继承 ``optim.Optimizer`` 来创建自定义优化器。

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    class CustomSGD(optim.Optimizer):
        def __init__(self, params, lr=0.01, momentum=0):
            defaults = dict(lr=lr, momentum=momentum)
            super(CustomSGD, self).__init__(params, defaults)
        
        def step(self, closure=None):
            loss = None
            if closure is not None:
                loss = closure()
            
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    d_p = p.grad.data
                    if group['momentum'] != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = rm.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(group['momentum']).add_(d_p)
                            d_p = buf
                    
                    p.data.add_(-group['lr'], d_p)
            
            return loss
    
    # 使用自定义优化器
    optimizer = CustomSGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # 训练循环
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

学习率调度器
=============

学习率调度器用于在训练过程中动态调整学习率，这对于模型的收敛和性能优化非常重要。Riemann 提供了多种学习率调度器，每种都有其特定的调整策略。

学习率调度器类型
-----------------

StepLR
~~~~~~

**功能**：按照固定的步长和衰减因子调整学习率。

**参数**：
- `optimizer`：要调整学习率的优化器
- `step_size`：学习率衰减的步长
- `gamma`：学习率衰减因子，默认为 0.1
- `last_epoch`：最后一个 epoch 的索引，默认为 -1

**使用场景**：适用于需要在固定间隔降低学习率的场景。

MultiStepLR
~~~~~~~~~~~

**功能**：在指定的里程碑处调整学习率。

**参数**：
- `optimizer`：要调整学习率的优化器
- `milestones`：学习率衰减的里程碑列表
- `gamma`：学习率衰减因子，默认为 0.1
- `last_epoch`：最后一个 epoch 的索引，默认为 -1

**使用场景**：适用于需要在特定 epoch 处降低学习率的场景。

ExponentialLR
~~~~~~~~~~~~~

**功能**：按照指数衰减调整学习率。

**参数**：
- `optimizer`：要调整学习率的优化器
- `gamma`：学习率衰减因子
- `last_epoch`：最后一个 epoch 的索引，默认为 -1

**使用场景**：适用于需要学习率持续平滑下降的场景。

CosineAnnealingLR
~~~~~~~~~~~~~~~~~

**功能**：按照余弦函数的形状调整学习率。

**参数**：
- `optimizer`：要调整学习率的优化器
- `T_max`：余弦退火的周期
- `eta_min`：最小学习率，默认为 0
- `last_epoch`：最后一个 epoch 的索引，默认为 -1

**使用场景**：适用于需要学习率先下降后上升的场景，有助于跳出局部最优。

ReduceLROnPlateau
~~~~~~~~~~~~~~~~~

**功能**：当指标停止改善时调整学习率。

**参数**：
- `optimizer`：要调整学习率的优化器
- `mode`：模式，'min' 或 'max'，默认为 'min'
- `factor`：学习率衰减因子，默认为 0.1
- `patience`：指标停止改善后等待的 epoch 数，默认为 10
- `threshold`：衡量新最佳值的阈值，默认为 1e-4
- `threshold_mode`：阈值模式，'rel' 或 'abs'，默认为 'rel'
- `cooldown`：学习率降低后恢复正常操作前的等待 epoch 数，默认为 0
- `min_lr`：最小学习率，默认为 0
- `eps`：学习率变化的最小值，默认为 1e-8

**使用场景**：适用于需要根据验证指标动态调整学习率的场景。

学习率调度器的使用方法
-----------------------

学习率调度器的基本使用流程如下：

1. 创建优化器
2. 创建学习率调度器，传入优化器和相关参数
3. 在训练循环中，先调用优化器的 `step()` 方法更新参数
4. 然后调用调度器的 `step()` 方法更新学习率

调度器与优化器的配合使用
---------------------------

- **顺序**：先调用 `optimizer.step()`，再调用 `scheduler.step()`
- **参数组**：调度器会根据每个参数组的初始学习率进行调整
- **状态保存**：调度器的状态可以通过 `state_dict()` 和 `load_state_dict()` 方法保存和加载
- **特殊情况**：`ReduceLROnPlateau` 调度器需要在 `step()` 方法中传入验证指标

完整示例代码
-------------

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    import riemann.optim as optim
    from riemann.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau

    # 创建模型
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )

    # 创建优化器
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # 创建学习率调度器（选择一种）
    # 1. StepLR 示例
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # 2. MultiStepLR 示例
    # scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    
    # 3. ExponentialLR 示例
    # scheduler = ExponentialLR(optimizer, gamma=0.99)
    
    # 4. CosineAnnealingLR 示例
    # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0.001)
    
    # 5. ReduceLROnPlateau 示例（需要在 step() 中传入验证损失）
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    # 定义损失函数
    loss_fn = nn.MSELoss()

    # 生成示例数据
    inputs = rm.randn(100, 10)
    targets = rm.randn(100, 1)

    # 训练循环
    num_epochs = 100
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 更新学习率
        scheduler.step()  # 对于 ReduceLROnPlateau，使用 scheduler.step(loss.item())
        
        # 打印信息
        if (epoch + 1) % 10 == 0:
            current_lr = scheduler.get_lr()[0] if hasattr(scheduler, 'get_lr') else optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, LR: {current_lr:.6f}')

学习率调度器的最佳实践
-----------------------

1. **选择合适的调度器**：根据任务特点选择合适的学习率调度策略
2. **设置合理的初始学习率**：初始学习率不宜过大或过小
3. **结合验证集**：使用 `ReduceLROnPlateau` 时，应基于验证集指标而非训练集指标
4. **学习率下限**：为 `ReduceLROnPlateau` 设置合理的 `min_lr`，防止学习率过小导致训练停滞
5. **预热阶段**：对于大型模型，可以考虑在训练初期使用较小的学习率进行预热
6. **参数组配合**：与参数组结合使用时，确保每个参数组的初始学习率设置合理
