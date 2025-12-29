优化算法
=========

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
~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~

SGD 是梯度下降的一种变体，它在每次迭代中使用数据子集更新参数。

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
~~~~~~~~~~~~~~~~~~~~~~~~

Nesterov 动量是动量的一种变体，可以提供更好的收敛性。

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

Adam (自适应矩估计)
~~~~~~~~~~~~~~~~~~~~~

Adam 结合了 AdaGrad 和 RMSProp 算法的最佳特性，提供了一个能够在噪声问题上处理稀疏梯度的优化算法。

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

AdaGrad 根据参数调整学习率，对与频繁出现特征相关的参数执行较小的更新。

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

LBFGS 是一种拟牛顿方法，使用有限的内存量近似 BFGS 算法。

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # 创建 LBFGS 优化器
    optimizer = optim.LBFGS(model.parameters(), lr=1.0)
    
    # 为 LBFGS 定义闭包函数
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

参数组允许您为不同的参数集使用不同的超参数。

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

梯度裁剪
--------

梯度裁剪防止深度网络中的梯度爆炸。

按范数裁剪
~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 按范数裁剪梯度
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

按值裁剪
~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 按值裁剪梯度
    nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

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
~~~~~~~~~~~~

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