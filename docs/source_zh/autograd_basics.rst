自动微分基础
============

Riemann 的自动微分引擎，类似于 PyTorch 的 autograd，允许自动计算张量操作的梯度。这对于训练神经网络和其他优化任务至关重要。

梯度跟踪
--------

默认情况下，张量不跟踪其操作。要启用梯度跟踪，请在创建张量时设置 ``requires_grad=True``：

.. code-block:: python

    import riemann as rm
    
    # 不跟踪梯度的张量
    x = rm.tensor([1., 2., 3.])
    print(x.requires_grad)  # False
    
    # 跟踪梯度的张量
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    print(x.requires_grad)  # True

您也可以在现有张量上启用或禁用梯度跟踪：

.. code-block:: python

    x = rm.tensor([1., 2., 3.])
    print(x.requires_grad)  # False
    
    # 启用梯度跟踪
    x.requires_grad_(True)
    print(x.requires_grad)  # True

计算图
=======

当您对具有 ``requires_grad=True`` 的张量执行操作时，Riemann 会构建一个计算图，跟踪每个张量是如何计算的。这个图用于在反向传播期间计算梯度。

.. code-block:: python

    import riemann as rm
    
    # 创建带梯度跟踪的张量
    x = rm.tensor(2.0, requires_grad=True)
    y = rm.tensor(3.0, requires_grad=True)
    
    # 构建计算图
    z = x * y + x ** 2
    
    # 打印计算图信息
    print(z.requires_grad)  # True

计算梯度
--------

要计算梯度，请在输出张量上调用 ``backward()`` 方法：

.. code-block:: python

    import riemann as rm
    
    # 创建带梯度跟踪的张量
    x = rm.tensor(2.0, requires_grad=True)
    y = rm.tensor(3.0, requires_grad=True)
    
    # 定义计算
    z = x * y + x ** 2
    
    # 计算梯度
    z.backward()
    
    # 访问梯度
    print(x.grad)  # dz/dx = y + 2*x = 3 + 4 = 7
    print(y.grad)  # dz/dy = x = 2

对于标量输出，可以直接调用 ``backward()``。对于非标量输出，需要提供梯度参数：

.. code-block:: python

    import riemann as rm
    
    # 创建带梯度跟踪的张量
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    
    # 定义产生非标量输出的计算
    y = x * 2
    
    # 计算相对于向量的梯度
    gradient = rm.tensor([1., 1., 1.])  # 和的梯度
    y.backward(gradient)
    
    # 访问梯度
    print(x.grad)  # [2., 2., 2.]

梯度累积
--------

默认情况下，梯度会累积。这意味着如果您多次调用 ``backward()``，梯度会累加：

.. code-block:: python

    import riemann as rm
    
    # 创建带梯度跟踪的张量
    x = rm.tensor(1.0, requires_grad=True)
    
    # 第一次计算
    y = x * 2
    y.backward()
    print(x.grad)  # 2
    
    # 第二次计算
    y = x * 3
    y.backward()
    print(x.grad)  # 2 + 3 = 5 (梯度累积)
    
    # 清除梯度
    if x.grad is not None:
        x.grad.zero_()
    print(x.grad)  # 0

禁用梯度跟踪
============

有时您需要执行不跟踪梯度的操作，例如在评估期间。您可以使用几种方法：

使用 ``torch.no_grad()`` 上下文管理器：

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    
    with rm.no_grad():
        y = x * 2
        print(y.requires_grad)  # False

使用 ``detach()`` 创建不跟踪梯度的新张量：

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    y = x.detach()
    print(y.requires_grad)  # False

原地操作和梯度
--------------

原地操作可能会影响梯度计算。以下是一些重要的注意事项：

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    
    # 这将引发错误，因为不允许对叶子变量进行原地操作
    try:
        x.add_(1)  # 这将引发错误
    except RuntimeError as e:
        print(f"错误: {e}")
    
    # 相反，创建一个新张量
    y = x + 1
    print(y.requires_grad)  # True

高阶梯度
========

Riemann 支持通过设置 ``create_graph=True`` 计算高阶导数：

.. code-block:: python

    import riemann as rm
    
    # 创建带梯度跟踪的张量
    x = rm.tensor(2.0, requires_grad=True)
    
    # 一阶计算
    y = x ** 3
    
    # 计算带图创建的一阶梯度
    dy_dx = rm.grad(y, x, create_graph=True)[0]
    print(dy_dx)  # 12
    
    # 计算二阶梯度
    d2y_dx2 = rm.grad(dy_dx, x)[0]
    print(d2y_dx2)  # 12

梯度计算技巧
============

1. **内存管理**：梯度计算使用内存来存储计算图。当您不需要梯度时，使用 ``no_grad()`` 或 ``detach()`` 来节省内存。

常见陷阱
--------

1. **原地操作**：避免对需要梯度的张量进行原地操作。

2. **分离张量**：一旦分离，张量将失去其梯度历史。

3. **非标量输出**：在非标量输出上调用 ``backward()`` 时，记得提供梯度参数。

4. **内存泄漏**：长时间运行带梯度跟踪的计算可能会消耗大量内存。

示例
----

简单神经网络训练
================

.. code-block:: python

    import riemann as rm
    
    # 创建简单数据集
    X = rm.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    y = rm.tensor([[0.0], [1.0], [0.0]], requires_grad=True)
    
    # 初始化权重和偏置
    w = rm.randn(2, 1, requires_grad=True)
    b = rm.randn(1, requires_grad=True)
    
    # 前向传播
    predictions = rm.matmul(X, w) + b
    loss = rm.mean((predictions - y) ** 2)
    
    # 反向传播
    loss.backward()
    
    # 更新权重（简单梯度下降）
    learning_rate = 0.01
    with rm.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    
    print(f"损失: {loss.item():.4f}")
