自动求导基础
============

Riemann 的自动微分引擎能够自动记录张量计算的过程，形成计算图，并通过反向传播算法高效地计算导数。这对于训练神经网络和其他优化任务至关重要。

**核心概念**

- **计算图（Computation Graph）**：Riemann 在后台自动构建的有向图，记录了张量之间的运算关系。每个节点代表一个张量，边代表运算操作。

- **前向计算（Forward Pass）**：从输入张量开始，沿着计算图执行运算，最终得到输出结果的过程。

- **反向传播（Backward Propagation）**：从输出张量开始，沿着计算图的反方向传播梯度，计算每个输入张量的导数。

- **梯度（Gradient）**：标量张量对其他张量的偏导数，表示输出相对于输入的变化率。

- **叶子节点张量（Leaf Node Tensor）**：用户直接创建的张量（例如通过 ``rm.tensor()``），且 ``requires_grad=True``。这些通常是模型参数。

- **中间节点张量（Intermediate Node Tensor）**：通过对其他张量进行运算而创建的张量。默认情况下，不会保留中间节点的梯度。

**梯度计算方法**

Riemann 提供了两种计算梯度的方法：

1. **backward() 方法**：适合一次求出多个张量的梯度。调用后，所有参与计算的叶子节点张量的梯度会被计算并存储在各自的 ``grad`` 属性中。

2. **grad() 函数**：适合对指定的张量求导。可以精确控制需要计算哪些张量的梯度，返回一个包含梯度的元组，不会修改张量的 ``grad`` 属性。

梯度跟踪开关
------------

默认情况下，张量计算不跟踪其梯度。要启用梯度跟踪开关，请在创建张量时设置 ``requires_grad=True``。

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

Riemann 提供了两种计算梯度的方法：``backward()`` 方法和 ``grad()`` 函数。

使用 backward() 方法
~~~~~~~~~~~~~~~~~~~~

``backward()`` 方法适合一次求出多个张量的梯度。调用后，梯度会自动存储在参与计算的叶子节点张量的 ``grad`` 属性中。

**函数签名**

.. code-block:: python

    tensor_object.backward(gradient=None, retain_graph=False, create_graph=False)

**参数说明**

- **gradient** （可选）：当输出张量不是标量时，需要提供一个与输出张量形状相同的梯度张量。对于标量输出，可以省略此参数，默认为 ``None`` （相当于传入标量 1）。
- **retain_graph** （可选）：是否保留计算图。默认为 ``False`` ，表示反向传播后释放计算图。如果需要多次调用 ``backward()`` ，应设置为 ``True`` 。
- **create_graph** （可选）：是否记录梯度的计算图，以便后续计算高阶导数，默认为 ``False`` 。

**使用场景**

- 训练神经网络时，一次计算所有可训练参数的梯度
- 需要多次反向传播时（如梯度累积）
- 计算高阶导数时

**注意事项**

- 只有 ``requires_grad=True`` 的 **叶子节点张量** 才会被计算梯度
- **中间节点张量** 默认不会保留梯度，如果需要计算中间节点的梯度，需要调用 ``retain_grad()`` 方法
- 梯度会累积到 ``grad`` 属性中，多次调用 ``backward()`` 前需要手动清零梯度

**示例 1：标量输出的梯度计算**

.. code-block:: python

    import riemann as rm
    
    # 创建带梯度跟踪的张量（叶子节点）
    x = rm.tensor(2.0, requires_grad=True)
    y = rm.tensor(3.0, requires_grad=True)
    
    # 定义计算（中间节点）
    z = x * y + x ** 2.
    
    # 计算梯度
    z.backward()
    
    # 访问梯度
    print(x.grad)  # dz/dx = y + 2*x = 3 + 4 = 7
    print(y.grad)  # dz/dy = x = 2

**示例 2：非标量输出的梯度计算**

.. code-block:: python

    import riemann as rm
    
    # 创建带梯度跟踪的张量
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    
    # 定义产生非标量输出的计算
    y = x * 2.
    
    # 计算相对于向量的梯度，需要传入梯度参数
    gradient = rm.tensor([1., 1., 1.])  # 雅可比向量积的向量
    y.backward(gradient)
    
    # 访问梯度
    print(x.grad)  # [2., 2., 2.]

**示例 3：保留中间节点的梯度**

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor(2.0, requires_grad=True)
    y = x * 3  # 中间节点
    z = y ** 2  # 输出
    
    # 保留中间节点 y 的梯度
    y.retain_grad()
    
    z.backward()
    
    print(x.grad)  # dz/dx = 36
    print(y.grad)  # dz/dy = 12（因为调用了 retain_grad()）

使用 grad() 函数
~~~~~~~~~~~~~~~~

``grad()`` 函数适合对指定的张量求导，可以精确控制需要计算哪些张量的梯度。

**函数签名**

.. code-block:: python

    riemann.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=False, allow_unused=False)

**参数说明**

- **outputs** ：输出张量（标量或张量），用于计算梯度的起点
- **inputs** ：输入张量或张量元组，指定需要计算梯度的张量
- **grad_outputs** （可选）：当 ``outputs`` 不是标量时，需要提供梯度张量
- **retain_graph** （可选）：是否保留计算图，默认为 ``False``
- **create_graph** （可选）：是否记录梯度的计算图，以便后续计算高阶导数，默认为 ``False``
- **allow_unused** （可选）：是否允许某些输入张量未被使用，默认为 ``False``

**使用场景**

- 只需要计算特定张量的梯度，而不是所有叶子节点的梯度
- 不想修改张量的 ``grad`` 属性时
- 需要更灵活地控制梯度计算过程

**注意事项**

- 梯度以 **元组形式返回** ，顺序与 ``inputs`` 参数的顺序一致
- 只有 ``inputs`` 中指定的张量才会计算梯度
- **不会修改** 输入张量的 ``grad`` 属性
- 中间节点即使调用了 ``retain_grad()`` ，也不会在 ``grad()`` 中自动计算梯度，必须显式指定

**示例 1：计算指定张量的梯度**

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor(2.0, requires_grad=True)
    y = rm.tensor(3.0, requires_grad=True)
    z = rm.tensor(4.0, requires_grad=True)
    
    # 定义计算
    w = x * y + z
    
    # 只计算 x 和 y 的梯度，不计算 z 的梯度
    grads = rm.autograd.grad(w, (x, y))
    
    print(grads)  # (tensor(3.), tensor(2.))
    print(x.grad)  # None（grad() 不会修改 grad 属性）

**示例 2：非标量输出的梯度计算**

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    y = x * 2
    
    # 对于非标量输出，需要提供 grad_outputs
    grad_outputs = rm.tensor([1., 1., 1.])
    grads = rm.autograd.grad(y, x, grad_outputs=grad_outputs)
    
    print(grads)  # (tensor([2., 2., 2.]),)

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

梯度计算上下文控制
------------------

Riemann 提供了灵活的梯度计算上下文控制机制，通过上下文管理器和修饰器，可以方便地启用或禁用梯度跟踪。这在模型推理（需要禁用梯度以节省内存）和训练（需要启用梯度）场景中非常有用。

is_grad_enabled() 函数
~~~~~~~~~~~~~~~~~~~~~~

``is_grad_enabled()`` 函数用于检查当前是否启用了梯度计算。

.. code-block:: python

    import riemann as rm
    
    # 检查当前梯度状态
    print(rm.is_grad_enabled())  # True（默认启用）
    
    with rm.no_grad():
        print(rm.is_grad_enabled())  # False

no_grad() 上下文管理器/装饰器
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``no_grad()`` 用于暂时禁用梯度计算。在这个上下文中，所有计算将不会追踪梯度，适用于推理阶段，可显著减少内存使用并加速计算。

**作为上下文管理器使用：**

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    
    with rm.no_grad():
        y = x * 2.
        print(y.requires_grad)  # False

**作为函数装饰器使用：**

.. code-block:: python

    import riemann as rm
    
    @rm.no_grad
    def inference(model, x):
        # 函数内的计算不会追踪梯度
        return model(x)

enable_grad() 上下文管理器/装饰器
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``enable_grad()`` 用于暂时启用梯度计算。可用于在 ``no_grad`` 上下文中临时启用梯度计算。

**作为上下文管理器使用：**

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    
    with rm.no_grad():
        # 这里禁用了梯度
        print(rm.is_grad_enabled())  # False
        
        with rm.enable_grad():
            # 这里临时启用了梯度
            y = x * 2.
            print(y.requires_grad)  # True
        
        # 回到禁用梯度的状态
        print(rm.is_grad_enabled())  # False

**作为函数装饰器使用：**

.. code-block:: python

    import riemann as rm
    
    @rm.enable_grad
    def train_step(model, x, target, loss_fn):
        # 函数内的计算会追踪梯度
        pred = model(x)
        loss = loss_fn(pred, target)
        loss.backward()
        return loss

set_grad_enabled() 上下文管理器/装饰器
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``set_grad_enabled(mode)`` 是最灵活的梯度控制函数，可以显式地启用或禁用梯度计算。

**参数：**

- **mode** (bool): ``True`` 启用梯度计算，``False`` 禁用梯度计算

**作为上下文管理器使用：**

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    
    # 禁用梯度
    with rm.set_grad_enabled(False):
        y = x * 2.
        print(y.requires_grad)  # False
    
    # 启用梯度
    with rm.set_grad_enabled(True):
        y = x * 2.
        print(y.requires_grad)  # True

**作为函数装饰器使用：**

.. code-block:: python

    import riemann as rm
    
    @rm.set_grad_enabled(False)
    def inference(model, x):
        return model(x)
    
    @rm.set_grad_enabled(True)
    def train(model, x, target, loss_fn):
        pred = model(x)
        loss = loss_fn(pred, target)
        loss.backward()
        return loss

嵌套使用上下文管理器
~~~~~~~~~~~~~~~~~~~~

梯度控制上下文管理器支持嵌套使用，内层上下文会临时覆盖外层的设置：

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    
    with rm.no_grad():  # 外层：禁用梯度
        y1 = x * 2.
        print(f"外层 no_grad: y1.requires_grad = {y1.requires_grad}")  # False
        
        with rm.enable_grad():  # 内层：启用梯度
            y2 = x * 3.
            print(f"内层 enable_grad: y2.requires_grad = {y2.requires_grad}")  # True
        
        # 回到外层上下文
        y3 = x * 4.
        print(f"回到外层: y3.requires_grad = {y3.requires_grad}")  # False

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
   使用 ``track_grad`` 装饰器来包装你的函数，并提供显式的梯度计算。

   **梯度函数接口要求：**

   传递给 ``track_grad`` 的梯度函数必须遵循以下接口要求：

   - **参数**：必须接受与前向函数相同的参数（相同的名称和顺序）
   - **返回值**：必须返回一个 tuple（元组），包含每个输入张量的梯度（偏导数）
   - **元组元素**：每个元素对应相应输入张量的梯度。对于不需要梯度的张量，该位置返回 ``None``
   - **梯度计算**：梯度应计算为输出对每个输入的偏导数

   **单输入示例：**

   .. code-block:: python

       import riemann as rm
       import numpy as np
       
       def sigmoid_derivative(x):
           """sigmoid 函数的梯度：返回包含一个元素的元组"""
           sig = 1. / (1. + np.exp(-x.data))
           return (rm.tensor(sig * (1. - sig)),)  # 注意：必须返回元组
       
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

   **多输入示例：**

   .. code-block:: python

       import riemann as rm
       
       def multiply_derivative(x, y):
           """乘法函数的梯度：d(xy)/dx = y, d(xy)/dy = x"""
           return (y, x)  # 返回包含每个输入梯度的元组
       
       @rm.track_grad(multiply_derivative)
       def custom_multiply(x, y):
           """带梯度支持的自定义乘法函数"""
           return x * y
       
       # 使用多个输入进行测试
       x = rm.tensor(2.0, requires_grad=True)
       y = rm.tensor(3.0, requires_grad=True)
       z = custom_multiply(x, y)
       z.backward()
       print(f"z = {z}")  # 应为 6.0
       print(f"dz/dx = {x.grad}")  # 应为 3.0 (y)
       print(f"dz/dy = {y.grad}")  # 应为 2.0 (x)

3. **使用 Function 类**
   对于更复杂的情况，你可以继承 ``Function`` 类并实现 ``forward`` 和 ``backward`` 静态方法。

   **Function 类接口：**

   要使用 ``Function`` 类创建自定义函数，必须实现两个静态方法：

   **forward(ctx, *inputs)**

   - **用途**：执行前向计算
   - **参数**：

     - ``ctx``：上下文对象，用于保存反向传播所需的信息。使用 ``ctx.save_for_backward()`` 保存反向传播需要的张量
     - ``*inputs``：输入张量（可变数量的参数）
   - **返回**：前向计算的输出张量
   - **用法**：在此实现自定义计算逻辑，并使用 ``ctx.save_for_backward()`` 保存梯度计算所需的任何张量

   **backward(ctx, grad_output)**

   - **用途**：执行反向（梯度）计算
   - **参数**：

     - ``ctx``：上下文对象，包含前向传播期间保存的信息。通过 ``ctx.saved_tensors`` 访问保存的张量
     - ``grad_output``：输出张量的梯度（来自计算图中后续层）
   - **返回**：梯度元组，每个输入张量对应一个梯度。每个梯度应为 ``grad_output`` 与局部梯度（偏导数）的乘积
   - **用法**：使用链式法则计算梯度：``grad_input = grad_output * local_gradient``

   **示例：**

   .. code-block:: python

       import riemann as rm
       import numpy as np
       
       class CustomSigmoid(rm.autograd.Function):
           @staticmethod
           def forward(ctx, x):
               """sigmoid 函数的前向计算
               
               参数：
                   ctx: 用于保存张量的上下文对象
                   x: 输入张量
               
               返回：
                   应用 sigmoid 后的输出张量
               """
               sig = 1. / (1. + np.exp(-x.data))
               ctx.save_for_backward(rm.tensor(sig))  # 保存用于反向传播
               return rm.tensor(sig)
           
           @staticmethod
           def backward(ctx, grad_output):
               """sigmoid 函数的反向计算
               
               参数：
                   ctx: 包含保存张量的上下文对象
                   grad_output: 来自输出侧的梯度
               
               返回：
                   关于输入的梯度
               """
               sig, = ctx.saved_tensors  # 检索保存的张量
               # 链式法则：grad_input = grad_output * local_gradient
               # sigmoid 的 local_gradient：sig * (1 - sig)
               return grad_output * sig * (1. - sig)
       
       # 测试 CustomSigmoid
       x = rm.tensor(0.0, requires_grad=True)
       y = CustomSigmoid.apply(x)  # 使用 apply() 调用函数
       y.backward()
       print(f"Sigmoid 输出: {y}")  # 应为 0.5
       print(f"Sigmoid 梯度: {x.grad}")  # 应为 0.25

   **关键点：**

   - 始终对 ``forward`` 和 ``backward`` 方法使用 ``@staticmethod`` 装饰器
   - 在 ``forward`` 中使用 ``ctx.save_for_backward()`` 保存梯度计算所需的张量
   - 在 ``backward`` 中通过 ``ctx.saved_tensors`` 访问保存的张量（返回元组）
   - ``backward`` 方法必须返回一个元组，包含 ``forward`` 每个输入对应的梯度
   - 使用 ``ClassName.apply(*inputs)`` 调用函数，而不是实例化类

高级计算图操作
--------------

Riemann 提供了几个高级函数用于操作计算图，这些函数可以在不影响前向计算值或反向梯度值的情况下修改计算图。这些函数对于将原本不参与梯度计算的张量连接到计算图非常有用。

fwbw_all_zero 函数
~~~~~~~~~~~~~~~~~~

``fwbw_all_zero`` 函数在前向传播中返回值为 0.0 的标量张量，在反向传播中返回与输入形状相同的全零张量。

**用途：**
使用此函数可以将张量添加到计算图中，而不会影响前向计算结果或反向梯度值。

**使用方法：**
将 ``fwbw_all_zero(x)`` 添加到任何张量，以"无损"方式将 ``x`` 包含到计算图中。

**示例：**

.. code-block:: python

    import riemann as rm
    
    a = rm.tensor([1.0, 2.0], requires_grad=True)
    x = rm.tensor([3.0, 4.0], requires_grad=True)
    
    # 将 x 添加到计算图中，但不改变 a 的值
    a = a + rm.fwbw_all_zero(x)
    
    # a 的值保持不变，但 x 现在在计算图中
    print(f"a = {a}")  # 输出: [1.0, 2.0]
    
    # 当调用 backward 时，x 将收到零梯度
    a.sum().backward()
    print(f"x.grad = {x.grad}")  # 输出: [0.0, 0.0]

attach_zero_grad_sources 方法
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``attach_zero_grad_sources`` 方法将多个张量作为来源张量依附到目标张量。这不改变张量的值，但允许它在反向传播期间向这些来源传递零梯度。

**用途：**
将张量连接到计算图，使它们在调用 backward 时收到零梯度而不是 None。

**参数：**

- ``sources``：要依附的来源张量集合，可以是元组、列表或集合。只有 ``requires_grad=True`` 的张量（且不是张量本身）会被依附。

**返回：**
修改后的张量（self），便于链式调用。

**示例：**

.. code-block:: python

    import riemann as rm
    
    a = rm.tensor([1.0, 2.0], requires_grad=True)
    b = rm.tensor([3.0, 4.0], requires_grad=True)
    c = rm.tensor([5.0, 6.0], requires_grad=True)
    
    # 将 a 和 b 依附到 c 的计算图
    c.attach_zero_grad_sources([a, b])
    
    # c 的值不变，但 backward 会向 a 和 b 传递零梯度
    result = (c * 2).sum()
    result.backward()
    
    print(f"a.grad = {a.grad}")  # 输出: [0.0, 0.0]
    print(f"b.grad = {b.grad}")  # 输出: [0.0, 0.0]
    print(f"c.grad = {c.grad}")  # 输出: [2.0, 2.0]

share_grad_map 函数
~~~~~~~~~~~~~~~~~~~

``share_grad_map`` 函数将一组张量连接到一个共享的计算图，而不改变现有前向计算值或反向梯度值。

**用途：**
确保组中的所有张量都参与计算图并收到梯度（对于不直接参与计算的张量为零），而不是 None。

**参数：**

- ``tensors``：要连接的张量元组或列表。必须是元组或列表（不是集合）以保持顺序。

**返回：**
具有相同值但连接到共享计算图的张量元组或列表。注意：``requires_grad=True`` 的张量会被克隆（不会被原地修改）。

**行为：**

- ``requires_grad=True`` 的张量会被克隆，并将所有其他张量作为零梯度来源依附到克隆张量
- 不需要梯度的张量或非 TN 对象保持不变
- 所有连接的张量互相接收零梯度

**示例：**

.. code-block:: python

    import riemann as rm
    
    a = rm.tensor([1.0, 2.0], requires_grad=True)
    b = rm.tensor([3.0, 4.0], requires_grad=True)
    c = rm.tensor([5.0, 6.0], requires_grad=True)
    
    # 定义一个只使用 a 和 b 的函数
    def func(a, b, c):
        return (a * b).sum()
    
    # share_grad_map 之前：c 不参与，收到 None
    y1 = func(a, b, c)
    y1.backward()
    print(f"c.grad = {c.grad}")  # 输出: None
    
    # 重置张量
    a = rm.tensor([1.0, 2.0], requires_grad=True)
    b = rm.tensor([3.0, 4.0], requires_grad=True)
    c = rm.tensor([5.0, 6.0], requires_grad=True)
    
    # share_grad_map 之后：所有张量连接，c 收到零梯度
    a_new, b_new, c_new = rm.share_grad_map((a, b, c))
    y2 = func(a_new, b_new, c_new)
    y2.backward()
    print(f"c.grad = {c_new.grad}")  # 输出: [0.0, 0.0]
    
    # 验证：前向值相同，a 和 b 的梯度不变
    assert float(y1.data) == float(y2.data)
    assert (a_new.grad == rm.tensor([3., 4.])).all()
    assert (b_new.grad == rm.tensor([1., 2.])).all()

**使用场景：**

这些函数在以下场景特别有用：

1. **多任务学习**：当某些参数不参与特定任务的损失计算，但你希望它们收到零梯度而不是 None，以便进行梯度累积。

2. **条件计算**：当某些张量在条件判断下被使用，但你希望无论条件如何都有一致的梯度行为。

3. **梯度监控**：当你想要监控组中所有参数的梯度，即使那些不直接参与特定计算的参数。

4. **参数共享**：当实现复杂的参数共享方案，其中所有共享参数应连接到同一个计算图。

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
