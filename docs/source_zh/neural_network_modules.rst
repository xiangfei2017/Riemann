神经网络模块
============

Riemann 通过 ``riemann.nn`` 包提供了一套全面的神经网络模块。这些模块是创建和训练神经网络的构建块。

模块基础
~~~~~~~~

Riemann 中的所有神经网络模块都继承自 ``nn.Module`` 类。这个基类提供了参数管理、前向传播定义和梯度计算的功能。

创建自定义模块
~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    class MyNetwork(nn.Module):
        def __init__(self):
            super(MyNetwork, self).__init__()
            self.linear1 = nn.Linear(10, 5)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(5, 1)
        
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x
    
    # 创建实例
    model = MyNetwork()
    print(model)

模块参数
~~~~~~~~

参数是模块的可学习方面。它们会自动跟踪以进行梯度计算：

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建线性层
    linear = nn.Linear(10, 5)
    
    # 访问参数
    for name, param in linear.named_parameters():
        print(f"{name}: {param.shape}")
    
    # 检查张量是否为参数
    print(linear.weight.requires_grad)  # True

线性层
~~~~~~

线性层对输入数据执行仿射变换。

全连接层
~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建线性层
    linear = nn.Linear(in_features=20, out_features=10)
    
    # 前向传播
    x = rm.randn(32, 20)  # 32 个样本的批次
    output = linear(x)
    print(output.shape)  # [32, 10]

卷积层
~~~~~~

卷积层对于处理图像等空间数据至关重要。

一维卷积
~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建一维卷积层
    conv1d = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
    
    # 前向传播
    x = rm.randn(10, 16, 50)  # [batch_size, channels, length]
    output = conv1d(x)
    print(output.shape)  # [10, 32, 48] (假设无填充)

二维卷积
~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建二维卷积层
    conv2d = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    
    # 前向传播
    x = rm.randn(4, 3, 32, 32)  # [batch_size, channels, height, width]
    output = conv2d(x)
    print(output.shape)  # [4, 16, 32, 32] (有填充)

三维卷积
~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建三维卷积层
    conv3d = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3)
    
    # 前向传播
    x = rm.randn(2, 3, 16, 16, 16)  # [batch_size, channels, depth, height, width]
    output = conv3d(x)
    print(output.shape)  # [2, 16, 14, 14, 14] (假设无填充)

池化层
~~~~~~

池化层减少空间维度。

最大池化
~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建最大池化层
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    # 前向传播
    x = rm.randn(4, 16, 32, 32)
    output = maxpool(x)
    print(output.shape)  # [4, 16, 16, 16]

平均池化
~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建平均池化层
    avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    # 前向传播
    x = rm.randn(4, 16, 32, 32)
    output = avgpool(x)
    print(output.shape)  # [4, 16, 16, 16]

归一化层
~~~~~~~~

归一化层有助于稳定训练。

批量归一化
~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建批量归一化层
    batch_norm = nn.BatchNorm2d(num_features=16)
    
    # 前向传播
    x = rm.randn(4, 16, 32, 32)
    output = batch_norm(x)
    print(output.shape)  # [4, 16, 32, 32]

激活函数
~~~~~~~~

Riemann 提供了各种激活函数。

ReLU 及其变体
~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建激活函数
    relu = nn.ReLU()
    leaky_relu = nn.LeakyReLU(negative_slope=0.01)
    prelu = nn.PReLU(num_parameters=1)  # 可学习参数
    elu = nn.ELU(alpha=1.0)
    
    # 前向传播
    x = rm.randn(4, 16)
    output_relu = relu(x)
    output_leaky = leaky_relu(x)
    output_prelu = prelu(x)
    output_elu = elu(x)
    
    print(output_relu.shape)  # [4, 16]

Sigmoid 和 Tanh
~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建激活函数
    sigmoid = nn.Sigmoid()
    tanh = nn.Tanh()
    
    # 前向传播
    x = rm.randn(4, 16)
    output_sigmoid = sigmoid(x)
    output_tanh = tanh(x)
    
    print(output_sigmoid.shape)  # [4, 16]

Softmax
~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建 softmax 层
    softmax = nn.Softmax(dim=1)
    
    # 前向传播
    x = rm.randn(4, 10)
    output = softmax(x)
    
    # 验证概率和为 1
    print(rm.sum(output, dim=1))  # tensor([1., 1., 1., 1.])

Dropout 层
~~~~~~~~~~

Dropout 层有助于防止过拟合。

Dropout
~~~~~~~

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

容器模块
~~~~~~~~

容器模块有助于组织其他模块。

Sequential
~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建顺序模型
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # 前向传播
    x = rm.randn(32, 10)
    output = model(x)
    
    print(output.shape)  # [32, 5]

ModuleList
~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建模块列表
    layers = nn.ModuleList([
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    ])
    
    # 前向传播
    x = rm.randn(32, 10)
    for layer in layers:
        x = layer(x)
    
    print(x.shape)  # [32, 5]

ModuleDict
~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建模块字典
    layers = nn.ModuleDict({
        'linear1': nn.Linear(10, 20),
        'relu': nn.ReLU(),
        'linear2': nn.Linear(20, 5)
    })
    
    # 前向传播
    x = rm.randn(32, 10)
    x = layers['linear1'](x)
    x = layers['relu'](x)
    x = layers['linear2'](x)
    
    print(x.shape)  # [32, 5]

损失函数
~~~~~~~~

损失函数衡量预测值与目标值之间的差异。

MSE 损失
~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建 MSE 损失
    mse_loss = nn.MSELoss()
    
    # 前向传播
    predictions = rm.randn(32, 10)
    targets = rm.randn(32, 10)
    loss = mse_loss(predictions, targets)
    
    print(loss.item())  # 标量值

交叉熵损失
~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建交叉熵损失
    ce_loss = nn.CrossEntropyLoss()
    
    # 前向传播
    predictions = rm.randn(32, 10)  # 原始分数（logits）
    targets = rm.randint(0, 10, (32,))  # 类别索引
    loss = ce_loss(predictions, targets)
    
    print(loss.item())  # 标量值

二元交叉熵损失
~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # 创建二元交叉熵损失
    bce_loss = nn.BCELoss()
    
    # 前向传播
    predictions = rm.sigmoid(rm.randn(32, 1))  # 概率
    targets = rm.randint(0, 2, (32, 1)).float()  # 二元目标
    loss = bce_loss(predictions, targets)
    
    print(loss.item())  # 标量值

示例
----

用于图像分类的简单 CNN
~~~~~~~~~~~~~~~~~~~~~~~

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