自动微分基础
============

Riemann 的自动微分引擎，允许自动计算张量操作的梯度。这对于训练神经网络和其他优化任务至关重要。

梯度跟踪
--------

默认情况下，张量计算不跟踪其梯度。要启用梯度跟踪，请在创建张量时设置 ``requires_grad=True``：

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
    z = x * y + x ** 2.
    
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
    z = x * y + x ** 2.
    
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
    y = x * 2.
    
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
    y = x * 2.
    y.backward()
    print(x.grad)  # 2
    
    # 第二次计算
    y = x * 3.
    y.backward()
    print(x.grad)  # 2 + 3 = 5 (梯度累积)
    
    # 清除梯度
    if x.grad is not None:
        x.grad.zero_()
    print(x.grad)  # 0

禁用梯度跟踪
============

有时您需要执行不跟踪梯度的操作，例如在评估期间。您可以使用几种方法：

使用 ``riemann.no_grad()`` 上下文管理器：

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    
    with rm.no_grad():
        y = x * 2.
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
        x.add_(1.)  # 这将引发错误
    except RuntimeError as e:
        print(f"错误: {e}")
    
    # 相反，创建一个新张量
    y = x + 1.
    print(y.requires_grad)  # True

高阶导数
--------

Riemann 支持通过设置 ``create_graph=True`` 来计算高阶导数：

.. code-block:: python

    import riemann as rm
    
    # 创建支持梯度跟踪的张量
    x = rm.tensor(2.0, requires_grad=True)
    
    # 一阶计算
    y = x ** 3.
    
    # 使用图创建计算一阶梯度
    dy_dx = rm.autograd.grad(y, x, create_graph=True)[0]
    print(dy_dx)  # 12
    
    # 计算二阶梯度
    d2y_dx2 = rm.autograd.grad(dy_dx, x)[0]
    print(d2y_dx2)  # 12

此外，Riemann 还提供了两个便捷的高阶导数计算工具：`d()` 方法和 `higher_order_grad()` 函数。

``d()`` 方法
~~~~~~~~~~~~

张量对象的 ``d()`` 方法用于计算当前标量张量对指定多个标量张量的混合偏导数。它可以方便地计算多阶混合导数。

.. code-block:: python

    import riemann as rm
    
    # 创建支持梯度跟踪的张量
    x = rm.tensor(2.0, requires_grad=True)
    y = rm.tensor(3.0, requires_grad=True)
    
    # 定义函数 f = x^3 * y^2
    f = x ** 3. * y ** 2.
    
    # 计算混合偏导数 d²f/dxdy
    d2f_dxdy = f.d(x, y)
    print(d2f_dxdy)  # 72.0
    
    # 计算三阶混合偏导数 d³f/dx²dy
    d3f_dx2dy = f.d(x, x, y)
    print(d3f_dx2dy)  # 72.0

``higher_order_grad()`` 函数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`higher_order_grad()` 函数用于计算标量张量输出相对于输入张量的 n 阶导数。它提供了一种便捷的方式来直接计算指定阶数的导数。

.. code-block:: python

    import riemann as rm
    
    # 创建支持梯度跟踪的张量
    x = rm.tensor(2.0, requires_grad=True)
    
    # 定义函数 y = x^3
    y = x ** 3.
    
    # 计算二阶导数
    d2y_dx2 = rm.autograd.higher_order_grad(y, x, 2)[0]
    print(d2y_dx2)  # 12.0
    
    # 计算三阶导数
    d3y_dx3 = rm.autograd.higher_order_grad(y, x, 3)[0]
    print(d3y_dx3)  # 6.0
    
    # 多个输入的情况
    x1 = rm.tensor(1.0, requires_grad=True)
    x2 = rm.tensor(2.0, requires_grad=True)
    z = x1 ** 2. + x2 ** 3.
    grads = rm.autograd.higher_order_grad(z, [x1, x2], 2)
    print(grads)  # (2.0, 12.0)

功能函数 (Functional API)
--------------------------

Riemann 还在 `riemann.autograd.functional` 模块中提供了一系列功能函数，用于计算更高级的导数结构，如雅可比矩阵、Hessian 矩阵、雅可比向量积等。

``jacobian()`` 函数
~~~~~~~~~~~~~~~~~~~~

`jacobian()` 函数用于计算函数从输入到输出的雅可比矩阵 (Jacobian matrix)，展示了函数输出对输入的所有一阶偏导数。

.. code-block:: python

    import riemann as rm
    
    # 定义函数 f = x^2
    def f(x):
        return x ** 2
    
    # 创建输入张量
    x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    # 计算雅可比矩阵
    jac = rm.autograd.functional.jacobian(f, x)
    print(jac)
    print(jac.shape)  # (3, 3)  # 对于向量输入，结果形状为 (n_inputs, n_outputs)

``hessian()`` 函数
~~~~~~~~~~~~~~~~~~~~~~

``hessian()`` 函数用于计算标量值函数的 Hessian 矩阵，显示函数相对于其输入的所有二阶偏导数。

.. code-block:: python

    import riemann as rm
    
    # 定义函数 f = x^3
    def f(x):
        return x ** 3
    
    # 创建输入张量
    x = rm.tensor(2.0, requires_grad=True)
    
    # 计算 Hessian 矩阵
    hess = rm.autograd.functional.hessian(f, x)
    print(hess)
    print(hess.shape)  # (1, 1)  # 对于标量输入，形状为 (输入大小, 输入大小)

``derivative()`` 函数
~~~~~~~~~~~~~~~~~~~~~~~~~

``derivative()`` 函数用于为给定函数计算导函数。它创建一个新函数，当调用时返回原函数在指定输入处的导数。

.. code-block:: python

    import riemann as rm
    
    # 定义函数 f = x^2
    def f(x):
        return x ** 2
    
    # 创建导函数
    df = rm.autograd.functional.derivative(f)
    
    # 测试导函数
    x = rm.tensor(2.0, requires_grad=True)
    print(df(x))  # 应返回 tensor(4.0)
    
    # 多输入示例
    def g(x, y):
        return x * y + x ** 2
    
    dg = rm.autograd.functional.derivative(g)
    x = rm.tensor(2.0, requires_grad=True)
    y = rm.tensor(3.0, requires_grad=True)
    print(dg(x, y))

``jvp()`` (雅可比向量积) 函数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``jvp()`` 函数计算雅可比矩阵与给定向量的乘积。

.. code-block:: python

    import riemann as rm
    
    # 定义函数 f = x^2
    def f(x):
        return x ** 2
    
    # 创建输入张量
    x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    # 定义 v 向量
    v = rm.tensor([1.0, 1.0, 1.0])
    
    # 计算 jvp
    f_x, jvp_val = rm.autograd.functional.jvp(f, x, v)
    print(jvp_val)

``vjp()`` (向量雅可比积) 函数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`vjp()` 函数用于计算给定向量与雅可比矩阵的乘积 (Vector-Jacobian Product)。

.. code-block:: python

    import riemann as rm
    
    # 定义函数 f = x^2
    def f(x):
        return x ** 2
    
    # 创建输入张量
    x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    # 定义 v 向量
    v = rm.tensor([1.0, 1.0, 1.0])
    
    # 计算 vjp
    f_x, vjp_val = rm.autograd.functional.vjp(f, x, v)
    print(vjp_val)

``hvp()`` (Hessian-向量积) 和 ``vhp()`` 函数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`hvp()` 和 `vhp()` 函数分别用于计算 Hessian-向量积 (Hessian-Vector Product) 和向量-Hessian-积 (Vector-Hessian Product)。由于 Hessian 矩阵是对称的，`hvp()` 和 `vhp()` 实际上是相同的。

.. code-block:: python

    import riemann as rm
    
    # 定义标量值函数
    def f(x):
        return (x ** 3).sum()
    
    # 创建输入张量
    x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    # 定义 v 向量
    v = rm.tensor([1.0, 1.0, 1.0])
    
    # 计算 hvp
    f_x, hvp_val = rm.autograd.functional.hvp(f, x, v)
    print(hvp_val)

    # vhp 的计算方式与 hvp 相同
    f_x, vhp_val = rm.autograd.functional.vhp(f, x, v)
    print(vhp_val)

自定义梯度函数
-------------------------

Riemann 提供三种方式来实现带梯度跟踪的自定义函数：

1. **使用 Riemann 张量函数（自动梯度）**
   如果你的自定义函数使用现有的 Riemann 张量函数实现，你无需编写任何梯度代码即可自动获得梯度跟踪能力：
   
   .. code-block:: python

       import riemann as rm
       
       def my_custom_function(x):
           """一个自动获得梯度支持的自定义函数"""
           return rm.exp(rm.sin(x)) + x**2.
       
       # 测试自动梯度跟踪
       x = rm.tensor(1.0, requires_grad=True)
       y = my_custom_function(x)
       y.backward()
       print(f"梯度: {x.grad}")  # 将自动计算正确的梯度

2. **使用 track_grad 装饰器**
   使用 ``track_grad`` 装饰器来包装你的函数，并提供显式的梯度计算：
   
   .. code-block:: python

       import riemann as rm
       import numpy as np
       
       def sigmoid_derivative(x):
           """sigmoid 函数的梯度"""
           sig = 1. / (1. + np.exp(-x.data))
           return (rm.tensor(sig * (1. - sig)),)
       
       @rm.track_grad(sigmoid_derivative)
       def custom_sigmoid(x):
           """带梯度支持的自定义 sigmoid 函数"""
           return rm.tensor(1. / (1. + np.exp(-x.data)))
       
       # 测试自定义 sigmoid 函数的梯度
       x = rm.tensor(0.0, requires_grad=True)
       y = custom_sigmoid(x)
       y.backward()
       print(f"Sigmoid 输出: {y}")  # 应为 0.5
       print(f"Sigmoid 梯度: {x.grad}")  # 应为 0.25

3. **使用 Function 类**
   对于更复杂的情况，你可以继承 ``Function`` 类并实现 forward 和 backward 方法：
   
   .. code-block:: python

       import riemann as rm
       import numpy as np
       
       class CustomSigmoid(rm.autograd.Function):
           @staticmethod
           def forward(ctx, x):
               """sigmoid 函数的前向计算"""
               sig = 1. / (1. + np.exp(-x.data))
               ctx.save_for_backward(rm.tensor(sig))
               return rm.tensor(sig)
           
           @staticmethod
           def backward(ctx, grad_output):
               """sigmoid 函数的反向计算"""
               sig, = ctx.saved_tensors
               return grad_output * sig * (1. - sig)
       
       # 测试 CustomSigmoid
       x = rm.tensor(0.0, requires_grad=True)
       y = CustomSigmoid.apply(x)
       y.backward()
       print(f"Sigmoid 输出: {y}")  # 应为 0.5
       print(f"Sigmoid 梯度: {x.grad}")  # 应为 0.25

梯度检查
-----------------

使用 ``gradcheck`` 函数来验证你的自定义梯度函数是否正确：

.. code-block:: python

    import riemann as rm
    
    # 定义一个用于梯度检查的测试函数
    def test_function(x):
        return CustomSigmoid.apply(x)
    
    # 执行梯度检查
    x = rm.tensor(0.0, requires_grad=True)
    check_passed = rm.gradcheck(test_function, (x,))
    print(f"梯度检查通过: {check_passed}")

Gradcheck 通过比较解析梯度和有限差分法计算的数值梯度来验证梯度计算的正确性。

梯度计算技巧
-------------------------

1. **内存管理**：梯度计算使用内存来存储计算图。当你不需要梯度时，使用 ``no_grad()`` 或 ``detach()`` 来节省内存。

常见误区
------------------

1. **原地操作**：避免对需要梯度跟踪的张量进行原地操作。

2. **张量分离**：分离后的张量会失去梯度历史。

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