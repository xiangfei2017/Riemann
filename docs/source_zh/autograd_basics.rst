自动求导基础
============

Riemann 的自动微分引擎，允许自动计算张量操作的梯度。这对于训练神经网络和其他优化任务至关重要。

梯度跟踪
--------

默认情况下，张量计算不跟踪其梯度。要启用梯度跟踪，请在创建张量时设置 ``requires_grad=True``。

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

计算梯度
--------

要计算梯度，请在输出张量上调用 ``backward()`` 方法。

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

默认情况下，梯度会累积。这意味着如果您多次调用 ``backward()``，梯度会累加。

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
--------------

有时你需要在不跟踪梯度的情况下执行操作，例如在评估模型时。你可以使用以下几种方法：

使用 ``riemann.no_grad()`` 上下文管理器：

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    
    with rm.no_grad():
        y = x * 2.
        print(y.requires_grad)  # False

张量计算图分离与数据复制方法
-------------------------------------------------

Riemann 提供了多种张量方法用于打断计算图依赖和复制张量数据。每个方法在以下方面具有不同的特性：

- 是否创建新的张量对象或原地修改
- 是否与原张量共享数据
- 是否保留梯度跟踪

以下是这些关键方法的单独说明和示例：

1. **detach()**：创建一个与原张量共享数据但从计算图中分离的新张量

detach() 方法返回一个新的张量对象，它与原张量共享相同的数据内存，但已从计算图中断开连接。这意味着：

- 修改分离后的张量会同时修改原张量
- 梯度不会通过分离后的张量反向传播

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    y = x * 2.
    
    # 将 y 从计算图中分离
    detached_y = y.detach()
    
    print(f"detached_y: {detached_y}")
    print(f"detached_y.requires_grad: {detached_y.requires_grad}")
    print(f"修改 detached_y 会修改 y: {id(detached_y.data) == id(y.data)}")

**特点**：创建新张量对象，与原张量共享内存，禁用梯度跟踪

2. **detach_()**：原地操作，将当前张量从计算图中分离

detach_() 是 detach() 的原地版本。它不会创建新张量，而是直接修改当前张量，使其与计算图断开连接。

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    y = x * 2.
    
    print(f"detach_() 前: y.requires_grad = {y.requires_grad}")
    y.detach_()  # 原地操作
    print(f"detach_() 后: y.requires_grad = {y.requires_grad}")

**特点**：原地修改张量（不创建新对象），与原张量共享内存（同一张量），禁用梯度跟踪

3. **clone()**：创建一个数据复制的新张量，保持计算图依赖关系

clone() 方法创建一个完全独立的新张量对象，拥有自己的数据内存，但保留了原张量的计算图依赖关系。这意味着对克隆张量的操作可以反向传播梯度到原始张量。

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    y = x * 2.
    
    cloned_y = y.clone()
    
    print(f"cloned_y: {cloned_y}")
    print(f"cloned_y.requires_grad: {cloned_y.requires_grad}")
    print(f"修改 cloned_y 不会修改 y: {id(cloned_y.data) != id(y.data)}")
    
    # 演示梯度可以通过克隆张量传递到原始张量
    loss = cloned_y.sum()
    loss.backward()
    print(f"x.grad after backward(): {x.grad}")  # 梯度从克隆张量传递到了 x

**特点**：创建新张量对象，复制数据（不共享内存），保留梯度跟踪

4. **copy()**：创建一个数据复制的新张量，与计算图完全分离

copy() 方法创建一个新的张量对象，拥有自己的数据内存，并且与计算图完全分离。这相当于调用 clone().detach_()，适用于创建独立的、不跟踪梯度的张量副本。

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    y = x * 2.
    
    copied_y = y.copy()
    
    print(f"copied_y: {copied_y}")
    print(f"copied_y.requires_grad: {copied_y.requires_grad}")
    print(f"修改 copied_y 不会修改 y: {id(copied_y.data) != id(y.data)}")

**特点**：创建新张量对象，复制数据（不共享内存），禁用梯度跟踪

5. 各方法关键差异对比

下表总结了这四个方法的关键差异：

+----------------+----------------------+------------------------+-------------------------------+
| 方法           | 是否创建新对象？     | 是否与原张量共享内存？ | 是否支持梯度跟踪？            |
+================+======================+========================+===============================+
| detach()       | 是                   | 是                     | 否                            |
+----------------+----------------------+------------------------+-------------------------------+
| detach_()      | 否                   | 不适用，同一张量       | 否                            |
+----------------+----------------------+------------------------+-------------------------------+
| clone()        | 是                   | 否                     | 是                            |
+----------------+----------------------+------------------------+-------------------------------+
| copy()         | 是                   | 否                     | 否                            |
+----------------+----------------------+------------------------+-------------------------------+

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    
    # 使用 detach() - 创建新张量，共享数据，与图分离
    y1 = x.detach()
    print(f"detach() 结果: y1 = {y1}, requires_grad={y1.requires_grad}")
    
    # 使用 detach_() - 原地操作，修改当前张量
    x2 = rm.tensor([1., 2., 3.], requires_grad=True)
    print(f"detach_() 之前: x2.requires_grad={x2.requires_grad}")
    x2.detach_()
    print(f"detach_() 之后: x2.requires_grad={x2.requires_grad}")
    
    # 使用 clone() - 创建新张量，复制数据，保持图依赖
    y2 = x.clone()
    print(f"clone() 结果: y2 = {y2}, requires_grad={y2.requires_grad}")
    
    # 使用 copy() - 创建新张量，复制数据，与图分离
    y3 = x.copy()
    print(f"copy() 结果: y3 = {y3}, requires_grad={y3.requires_grad}")

这些方法的主要区别：

- **数据共享**: detach() 与原始张量共享数据，而 clone() 和 copy() 创建新的数据副本
- **原地操作**: detach_() 是原地操作，直接修改原张量，其他方法创建新张量
- **梯度跟踪**: clone() 保持梯度跟踪（如果原张量需要梯度），其他方法禁用梯度跟踪
- **独立副本**: copy() 创建一个完全独立的新张量对象，不与原张量共享数据，也不保留计算图依赖关系

原地操作和梯度
--------------

原地操作可能会影响梯度计算。以下是重要的注意事项：

1. **带梯度跟踪的叶子节点张量**：不允许对需要梯度跟踪的叶子节点张量执行原地操作，这会破坏反向传播所需的计算图
2. **带梯度跟踪的非叶子节点张量**：允许对需要梯度跟踪的非叶子节点张量（中间结果）执行原地操作

示例：

.. code-block:: python

    import riemann as rm
    
    # 1. 示例：带梯度跟踪的叶子节点张量不允许原地操作
    x = rm.tensor([1., 2., 3.], requires_grad=True)  # 叶子节点张量
    
    try:
        x.add_(1.)  # 这将引发错误
    except RuntimeError as e:
        print(f"叶子节点张量原地操作错误: {e}")
    
    # 2. 示例：非叶子节点张量允许原地操作
    y = x * 2.  # 非叶子节点张量
    print(f"非叶子节点张量原地加法前: y = {y}")
    y.add_(3.)  # 非叶子节点张量的原地操作
    print(f"非叶子节点张量原地加法后: y = {y}")
    
    # 计算非叶子节点张量原地操作后的梯度
    z = y.sum()
    z.backward()
    print(f"x (叶子张量) 的梯度: x.grad = {x.grad}")
    
    # 清除梯度
    x.grad.zero_()
    
    # 3. 示例：使用张量索引赋值进行原地操作
    y = x * 2.  # 非叶子节点张量
    print(f"索引赋值前: y = {y}")
    y[0] = 100.  # 使用索引进行原地赋值
    print(f"索引赋值后: y = {y}")
    
    # 计算索引赋值后的梯度
    z = y.sum()
    z.backward()
    print(f"索引赋值后 x 的梯度: x.grad = {x.grad}")
    
    # 清除梯度
    x.grad.zero_()
    
    # 4. 示例：原地操作中的梯度跟踪
    x = rm.tensor(2.0, requires_grad=True)  # 叶子张量
    y = rm.tensor(3.0, requires_grad=True)  # 叶子张量
    
    a = x * y  # 中间张量
    a.mul_(2.)  # 原地乘法
    b = a + x  # 最终张量
    
    b.backward()
    
    print(f"x (左值) 的梯度: x.grad = {x.grad}")
    print(f"y (右值) 的梯度: y.grad = {y.grad}")

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

求导功能函数 (Functional API)
------------------------------

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
   如果你的自定义函数使用现有的 Riemann 张量函数实现，你无需编写任何梯度代码即可自动获得梯度跟踪能力。
   
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

常见错误
--------

1. **原地操作**：避免对需要梯度跟踪的叶子节点张量进行原地操作。

2. **清除张量的计算依赖**：detach后的张量会失去计算图依赖关系，无法进行反向传播梯度计算。

3. **非标量输出**：在非标量输出上调用 ``backward()`` 时，记得提供梯度参数。

4. **内存泄漏**：长时间运行带梯度跟踪的计算可能会消耗大量内存。

示例
----

Rosenbrock 函数优化 (香蕉函数)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rosenbrock 函数（也称为香蕉函数）是一个经典的非凸优化问题。该函数在 (1, 1) 处取得最小值 0。

以下是使用 Riemann 的自动微分和 Adam 优化器来优化 Rosenbrock 函数的示例：

.. code-block:: python

    import riemann as rm
    from riemann import optim

    # 定义 Rosenbrock 函数（香蕉函数）
    def rosenbrock_2d(x, y):
        """2D 情况下的 Rosenbrock 函数"""
        return 100. * (y - x**2.)**2. + (1. - x)**2.

    # 初始化需要梯度跟踪的参数
    x = rm.tensor(-1.2, requires_grad=True)  # 从点 (-1.2, 1.0) 开始
    y = rm.tensor(1.0, requires_grad=True)
    params = [x, y]

    # 设置优化器
    optimizer = optim.Adam(params, lr=0.05)

    print("优化 Rosenbrock 函数（香蕉函数）:")
    print(f"初始 x: {x.item():.4f}, y: {y.item():.4f}")
    print(f"初始损失: {rosenbrock_2d(x, y).item():.4f}")

    # 执行优化过程
    for i in range(1000):
        loss = rosenbrock_2d(x, y)
        
        # 重置梯度
        optimizer.zero_grad()
        
        # 自动计算梯度
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 每 200 次迭代打印一次进度
        if i % 200 == 0:
            print(f"迭代次数 {i}: 损失 = {loss.item():.8f}, x = {x.item():.8f}, y = {y.item():.8f}")

    # 打印最终结果
    print(f"\n优化完成!")
    print(f"最终 x: {x.item():.10f}, y: {y.item():.10f}")
    print(f"最终损失: {loss.item():.10f}")
    print(f"理论最小值: x=1.0, y=1.0, 损失=0.0")
