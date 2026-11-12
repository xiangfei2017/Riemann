自动求导基础
============

Riemann 的自动求导引擎能够自动记录张量计算的过程，形成计算图，并通过反向传播算法高效地计算导数。这对于训练神经网络和其他优化任务至关重要。

**核心概念**

- 计算图：Riemann 在后台自动构建的有向图，记录了张量之间的运算关系。每个节点代表一个张量，边代表运算操作。

- 前向计算：从输入张量开始，沿着计算图执行运算，最终得到输出结果的过程。

- 反向传播：从输出张量开始，沿着计算图的反方向传播梯度，计算每个输入张量的导数。

- 梯度：标量张量对其他张量的偏导数，表示输出相对于输入的变化率。

- 叶子节点张量：用户直接创建的张量（例如通过 ``rm.tensor()``），且 ``requires_grad=True``。这些通常是模型参数。

- 中间节点张量：通过对其他张量进行运算而创建的张量。默认情况下，不会保留中间节点的梯度。

**梯度计算方法**

Riemann 提供了两种计算梯度的方法：

1. **backward() 方法**: 适合一次求出多个张量的梯度。调用后，所有参与计算的叶子节点张量的梯度会被计算并存储在各自的 ``grad`` 属性中。

2. **grad() 函数**: 适合对指定的张量求导。可以精确控制需要计算哪些张量的梯度，返回一个包含梯度的元组，不会修改张量的 ``grad`` 属性。

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

- gradient （可选）：当输出张量不是标量时，需要提供一个与输出张量形状相同的梯度张量。对于标量输出，可以省略此参数，默认为 ``None`` （相当于传入标量 1）。
- retain_graph （可选）：是否保留计算图。默认为 ``False`` ，表示反向传播后释放计算图。如果需要多次调用 ``backward()`` ，应设置为 ``True`` 。
- create_graph （可选）：是否记录梯度的计算图，以便后续计算高阶导数，默认为 ``False`` 。

**使用场景**

- 训练神经网络时，一次计算所有可训练参数的梯度
- 需要多次反向传播时（如梯度累积）
- 计算高阶导数时

**注意事项**

- 只有 ``requires_grad=True`` 的 **叶子节点张量** 才会被计算梯度
- 中间节点张量 默认不会保留梯度，如果需要计算中间节点的梯度，需要调用 ``retain_grad()`` 方法
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

- outputs ：输出张量（标量或张量），用于计算梯度的起点
- inputs ：输入张量或张量元组，指定需要计算梯度的张量
- grad_outputs （可选）：当 ``outputs`` 不是标量时，需要提供梯度张量
- retain_graph （可选）：是否保留计算图，默认为 ``False``
- create_graph （可选）：是否记录梯度的计算图，以便后续计算高阶导数，默认为 ``False``
- allow_unused （可选）：是否允许某些输入张量未被使用，默认为 ``False``

**使用场景**

- 只需要计算特定张量的梯度，而不是所有叶子节点的梯度
- 不想修改张量的 ``grad`` 属性时
- 需要更灵活地控制梯度计算过程

**注意事项**

- 梯度以 **元组形式返回** ，顺序与 ``inputs`` 参数的顺序一致
- 只有 ``inputs`` 中指定的张量才会计算梯度
- 不会修改 输入张量的 ``grad`` 属性
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

- mode (bool): ``True`` 启用梯度计算，``False`` 禁用梯度计算

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

1. **detach()**: 创建一个与原张量共享数据但从计算图中分离的新张量

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
    print(f"修改 detached_y 会修改 y: {detached_y.array().ctypes.data == y.detach().array().ctypes.data}")

**特点**: 创建新张量对象，与原张量共享内存，禁用梯度跟踪

2. **detach_()**: 原地操作，将当前张量从计算图中分离

detach_() 是 detach() 的原地版本。它不会创建新张量，而是直接修改当前张量，使其与计算图断开连接。

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    y = x * 2.
    
    print(f"detach_() 前: y.requires_grad = {y.requires_grad}")
    y.detach_()  # 原地操作
    print(f"detach_() 后: y.requires_grad = {y.requires_grad}")

**特点**: 原地修改张量（不创建新对象），与原张量共享内存（同一张量），禁用梯度跟踪

3. **clone()**: 创建一个数据复制的新张量，保持计算图依赖关系

clone() 方法创建一个完全独立的新张量对象，拥有自己的数据内存，但保留了原张量的计算图依赖关系。这意味着对克隆张量的操作可以反向传播梯度到原始张量。

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    y = x * 2.
    
    cloned_y = y.clone()
    
    print(f"cloned_y: {cloned_y}")
    print(f"cloned_y.requires_grad: {cloned_y.requires_grad}")
    print(f"修改 cloned_y 不会修改 y: {cloned_y.detach().array().ctypes.data != y.detach().array().ctypes.data}")
    
    # 演示梯度可以通过克隆张量传递到原始张量
    loss = cloned_y.sum()
    loss.backward()
    print(f"x.grad after backward(): {x.grad}")  # 梯度从克隆张量传递到了 x

**特点**: 创建新张量对象，复制数据（不共享内存），保留梯度跟踪

4. **copy()**: 创建一个数据复制的新张量，与计算图完全分离

copy() 方法创建一个新的张量对象，拥有自己的数据内存，并且与计算图完全分离。这相当于调用 clone().detach_()，适用于创建独立的、不跟踪梯度的张量副本。

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    y = x * 2.
    
    copied_y = y.copy()
    
    print(f"copied_y: {copied_y}")
    print(f"copied_y.requires_grad: {copied_y.requires_grad}")
    print(f"修改 copied_y 不会修改 y: {copied_y.detach().array().ctypes.data != y.detach().array().ctypes.data}")

**特点**: 创建新张量对象，复制数据（不共享内存），禁用梯度跟踪

5. 各方法关键差异对比

下表总结了这四个方法的关键差异：

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 方法
     - 是否创建新对象？
     - 是否与原张量共享内存？
     - 是否支持梯度跟踪？
   * - detach()
     - 是
     - 是
     - 否
   * - detach_()
     - 否
     - 不适用，同一张量
     - 否
   * - clone()
     - 是
     - 否
     - 是
   * - copy()
     - 是
     - 否
     - 否

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

- 数据共享: detach() 与原始张量共享数据，而 clone() 和 copy() 创建新的数据副本
- 原地操作: detach_() 是原地操作，直接修改原张量，其他方法创建新张量
- 梯度跟踪: clone() 保持梯度跟踪（如果原张量需要梯度），其他方法禁用梯度跟踪
- 独立副本: copy() 创建一个完全独立的新张量对象，不与原张量共享数据，也不保留计算图依赖关系

原地操作和梯度
--------------

原地操作可能会影响梯度计算。以下是重要的注意事项：

1. **带梯度跟踪的叶子节点张量**: 不允许对需要梯度跟踪的叶子节点张量执行原地操作，这会破坏反向传播所需的计算图
2. **带梯度跟踪的非叶子节点张量**: 允许对需要梯度跟踪的非叶子节点张量（中间结果）执行原地操作
3. **一元原地操作**: ``neg_()`` 的反向将流向历史的梯度取负；``abs_()`` 的反向按操作前元素的符号缩放梯度（元素为 0 处梯度为 0）
4. **取余与取整除法**: ``%=``（``remainder_``）支持梯度，右值梯度为 ``-grad * floor(x / y)``；``//=`` 与 PyTorch 一致，反向会抛 ``RuntimeError``：floor_divide 未实现梯度

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

此外，Riemann 还提供了两种更便捷的高阶导数计算方式：``TN.d()`` 方法和 ``func.grad`` 函数链式调用。

``TN.d()`` 方法
~~~~~~~~~~~~~~~~

张量对象的 ``d()`` 方法是计算高阶混合偏导数最方便的接口。它将对多个变量的各阶求导自然地串在一起：

**功能用途**

- 计算当前标量张量对指定变量的**任意阶数偏导数**或**混合偏导数**
- 变量可以是**任意形状**（标量、向量、矩阵），不限于 0-d 标量
- 同一变量多次出现时**阶数自动累加**
- 显式阶数规格：``(var, k)`` 形式直接指定对该变量求 k 阶

**接口签名**

``current_scalar.d(*args, create_graph=False)``

参数:
    - ``*args``: 求导规格，两种形式可混用
        - ``TN`` — 对该变量求一阶
        - ``(TN, k)`` — 对该变量求 **k** 阶 (k ≥ 1)
    - ``create_graph``: ``bool``, 默认 ``False``。是否追踪结果计算图，以支持继续高阶微分。

返回值:
    - ``TN``，形状为各变量形状的扁平拼接：``shape = ⊕ shape(var_i) × order_i``

**示例代码**

.. code-block:: python

    import riemann as rm
    import numpy as np

    # === 标量变量（兼容旧版） ===
    x = rm.tensor(2.0, requires_grad=True)
    y = rm.tensor(3.0, requires_grad=True)
    f = x ** 3 * y ** 2

    print(f.d(x))                 # 一阶 df/dx = 3x²y² = 108
    print(f.d(x, y))              # 混合偏导 d²f/dxdy = 6x²y = 72
    print(f.d((x, 2)))            # 显式二阶 d²f/dx² = 6xy² = 108
    print(f.d(x, x))              # 同变量重复 = f.d((x, 2)) — 6xy² = 108
    print(f.d((x, 2), y))         # 显式多阶 d³f/∂x²∂y = 12xy = 72

    # === 向量变量（核心能力） ===
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    f = (x ** 3).sum()

    print(f.d(x))                 # 梯度向量 [3, 12, 27]
    print(f.d(x, x))              # Hessian 矩阵 diag([6, 12, 18])
    print(f.d((x, 3)))            # 三阶张量 shape=(3,3,3)，对角线 [24, 48, 72]

    # === 多向量混合偏导 ===
    x = rm.tensor([1., 2.], requires_grad=True)
    y = rm.tensor([3., 4.], requires_grad=True)
    f = (x * y).sum() ** 2

    print(f.d(x, y))              # (2, 2) 交叉偏导矩阵 [[28, 12], [8, 38]]

    # === 标量 + 向量 ===
    x = rm.tensor(2.0, requires_grad=True)
    y = rm.tensor([1., 2., 3.], requires_grad=True)
    f = x * (y ** 2).sum()

    print(f.d(x, y))              # d²f/∂x∂y = [2, 4, 6], shape=(3,)

**注意事项**

1. ``self`` 必须是**标量张量**（``ndim == 0``）且 ``requires_grad=True``
2. 所有求导变量必须 ``requires_grad=True``
3. ``args`` 至少要有一个，否则抛 ``ValueError``
4. ``(var, k)`` 形式的 k 必须 ≥ 1
5. 结果形状 = 各变量形状按 args 顺序拼接，如 ``f.d(x, y)`` 其中 x∈Rⁿ、y∈Rᵐ → shape=(n, m)
6. ``create_graph=True`` 保留计算图，可继续链式高阶微分；注意内存开销随阶数指数增长
7. 当对**张量**（ndim>0）做高阶偏导时，内部会逐元素标量 grad 再 stack 拼回形状

使用 ``func`` 模块函数计算高阶导数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``func`` 模块的柯里化设计（多数函数返回 Callable）天然适合链式嵌套，
以函数式方式灵活组合 ``grad``、``jacrev``、``jacfwd``、``hessian`` 等函数，
表达任意阶的高阶导数。以下展示几种最典型的场景，数值均已验证。

**示例 1: 向量输入 → 向量输出的 Jacobian 与二阶张量**

``func.jacrev`` / ``func.jacfwd`` 直接对向量输出函数求导，
天然处理向量输入 → 向量输出的高阶导数场景。
将 ``jacrev`` 再次嵌套自身，即可得到 Jacobian 的 Jacobian —— 三阶张量。

.. code-block:: python

    import riemann as rm
    from riemann import func

    # f(x) = [x₀³, x₁³, x₂³]，逐元素三次方，向量输入/向量输出
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    print(x)
    f = lambda xv: xv ** 3   # shape: (3,) → (3,)

    # ===== 一阶: Jacobian 矩阵 =====
    # J_{ij} = ∂f_i/∂x_j = 3x_i² · δ_{ij} （对角矩阵）
    J = func.jacrev(f)(x)
    print(J)
    # J.shape = (3, 3)
    # J = diag([3·1², 3·2², 3·3²]) = diag([3, 12, 27])

    # jacfwd 等价（前向模式，对这类小向量效率相近）
    J_fwd = func.jacfwd(f)(x)
    print(J_fwd)
    # → 与 J 完全相同

    # ===== 二阶: Jacobian 的 Jacobian —— 三阶张量 =====
    # T_{ijk} = ∂²f_i/∂x_j∂x_k
    # 对 f_i = x_i³: 仅当 i=j=k 时 T_{ijk} = 6x_i，其余为 0
    T = func.jacrev(lambda xv: func.jacrev(f, create_graph=True)(xv))(x)
    print(T)
    # T.shape = (3, 3, 3)
    # T[0,0,0] = 6·1 = 6,  T[1,1,1] = 6·2 = 12,  T[2,2,2] = 6·3 = 18
    # 所有其他元素 = 0

**示例 2: 向量输入 → 标量输出的 Hessian**

这是优化问题中最常见的高阶场景——目标函数是向量输入的标量值，
需要 Hessian 矩阵。
用 ``func.jacrev(func.grad(f))`` 嵌套一步构造 Hessian，
也可以用 ``func.hessian`` 直接得到 Hessian 矩阵。

.. code-block:: python

    import riemann as rm
    from riemann import func

    # 最小二乘 f(x) = ||A·x - b||², x ∈ R², A ∈ R^{3×2}, b ∈ R³
    A = rm.tensor([[1., 2.], [3., 4.], [5., 6.]], requires_grad=False)
    b = rm.tensor([1., 2., 3.], requires_grad=False)
    x = rm.tensor([1., 2.], requires_grad=True)

    def f_least_squares(xv):
        residual = A @ xv - b        # shape (3,)
        return (residual ** 2).sum() # 标量输出

    # ===== Hessian 矩阵 =====
    # 最小二乘的 Hessian 恒等于 2·AᵀA（与 x 无关！）
    H = func.jacrev(func.grad(f_least_squares, create_graph=True))(x)
    # H.shape = (2, 2)
    # H = 2·AᵀA = [[70, 88], [88, 112]]

**示例 3: 多输入函数的各输入独立高阶 —— ``f(x, y) = x³y⁴``**

通过 ``argnums`` 指定对哪个输入位置求导，再嵌套 ``func.grad`` 实现高阶：

.. code-block:: python

    import riemann as rm
    from riemann import func

    x = rm.tensor(2.0, requires_grad=True)
    print(x)
    y = rm.tensor(3.0, requires_grad=True)
    print(y)

    f = lambda xv, yv: xv ** 3 * yv ** 4

    # 对 x 二阶: ∂²f/∂x² = 6xy⁴ = 6·2·81 = 972
    d2x = func.grad(
        lambda xv, yv: func.grad(f, argnums=0, create_graph=True)(xv, yv),
        argnums=0
    )(x, y)   # → 972.0
    print(d2x)

    # 对 y 二阶: ∂²f/∂y² = 12x³y² = 12·8·9 = 864
    d2y = func.grad(
        lambda xv, yv: func.grad(f, argnums=1, create_graph=True)(xv, yv),
        argnums=1
    )(x, y)   # → 864.0
    print(d2y)

    # 同时对两个输入求一阶（argnums 为 tuple）
    dx, dy = func.grad(f, argnums=(0, 1), create_graph=True)(x, y)
    print(dx, dy)
    # dx = 3x²y⁴ = 972.0,  dy = x³·4y³ = 864.0

**示例 4: ``func.grad`` + ``has_aux`` 传递辅助值 + 继续高阶**

当 ``f`` 返回 ``(value, aux)`` 二元组时，``has_aux=True`` 让 ``func.grad``
只从 ``value`` 计算梯度，同时把 ``aux`` 原样传回：

.. code-block:: python

    import riemann as rm
    from riemann import func
    from riemann.autograd import grad

    x = rm.tensor(2.0, requires_grad=True)
    print(x)

    # f(x) = log(x), 辅助值是 1.0
    def f_with_aux(xv):
        return rm.log(xv), rm.tensor(1.0)   # (value, aux)

    # 构建带 has_aux 的高阶梯度函数
    grad_fn = func.grad(f_with_aux, has_aux=True, create_graph=True)
    g1, aux = grad_fn(x)
    print(g1, aux)
    # g1 = 1/x = 0.5,  aux = 1.0（原样返回）

    # g1 带着计算图，可以继续对它求导
    g2 = grad(g1, x, allow_unused=True)[0]
    print(g2)
    # g2 = -1/x² = -0.25

**示例 5: ``jacrev(grad(f))`` — 一次性构造 Hessian 矩阵**

对向量变量的二阶导数，可以用 ``func.jacrev(func.grad(f))`` 一步算出
完整的 Hessian 矩阵，比手动两次 ``func.grad`` 更高效：

.. code-block:: python

    import riemann as rm
    from riemann import func

    # f(x) = ||x||⁴ = Σ x_i⁴,  x ∈ R³
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    f = lambda xv: (xv ** 4).sum()

    # 梯度函数 g = ∇f
    # Hessian = Jacobian_of_forward(∇f)
    H = func.jacrev(func.grad(f, create_graph=True))(x)
    print(H)

    # H.shape = (3, 3)
    # H 是对角矩阵，对角元 12·x_i² = [12, 48, 108]
    #
    # 数学原理: H_ij = ∂²f/∂x_i∂x_j = ∂(∇f)_i/∂x_j = Jac(∇f)_{ij}

前向模式（``jacfwd``）对同一个问题结果相同：

.. code-block:: python

    import riemann as rm
    from riemann import func

    x = rm.tensor([1., 2., 3.], requires_grad=True)
    f = lambda xv: (xv ** 4).sum()
    H_fwd = func.jacfwd(func.grad(f, create_graph=True))(x)
    print(H_fwd)
    # → 与 H 完全相同

**示例 6: 批量高阶 —— ``func.vmap`` 包高阶函数**

对一批输入同时求高阶导数，用 ``func.vmap`` 将标量高阶函数向量化后批量执行：

.. code-block:: python

    import riemann as rm
    from riemann import func
    from riemann.autograd import grad

    xs = rm.tensor([1., 2., 3., 4.], requires_grad=True)
    print(xs)

    # 标量版本的二阶导数函数: x → f''(x) = 12x²
    def scalar_second(xv):
        f = xv ** 4
        g1 = grad(f, xv, create_graph=True, allow_unused=True)[0]
        g2 = grad(g1, xv, allow_unused=True)[0]
        return g2

    # vmap 批量应用
    batch_f2 = func.vmap(scalar_second)(xs)
    print(batch_f2)
    # → [12·1², 12·2², 12·3², 12·4²] = [12, 48, 108, 192]

或者先对批量求和转标量再反向（需要手动传 ``grad_outputs`` 给第二次 ``grad``）：

.. code-block:: python

    import riemann as rm
    from riemann.autograd import grad

    xs = rm.tensor([1., 2., 3., 4.], requires_grad=True)
    f_sum = (xs ** 4).sum()
    g1 = grad(f_sum, xs, create_graph=True, allow_unused=True)[0]
    g2 = grad(g1, xs, allow_unused=True,
              grad_outputs=rm.ones_like(g1))[0]   # 必须传 grad_outputs！
    print(g2)
    # → 同样 [12, 48, 108, 192]

**示例 7: ``TN.d()`` 与 ``func.grad`` 高阶链式等价性**

两种方式对同一计算求导结果完全一致：

.. code-block:: python

    import riemann as rm
    from riemann import func

    x = rm.tensor(2.0, requires_grad=True)
    y = rm.tensor(3.0, requires_grad=True)
    f = x ** 3 * y ** 2

    # 方式 A: TN.d() —— 最简洁
    d_x2 = f.d(x, x)
    print(d_x2)
    # → 6xy² = 6·2·9 = 108.0

    # 方式 B: func.grad 函数式高阶链
    f_fn = lambda xv, yv: xv ** 3 * yv ** 2
    gf = func.grad(f_fn, argnums=0, create_graph=True)
    ggf = func.grad(lambda xv, yv: gf(xv, yv), argnums=0)
    fg_x2 = ggf(x, y)
    print(fg_x2)
    # → 也是 108.0

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 场景
     - 推荐方式
     - 理由
   * - 标量计算图上的纯高阶导数
     - ``f.d((x, n))``
     - 一行搞定，内部自动管理 create_graph 链
   * - 多变量混合偏导
     - ``f.d(x, y, z)``
     - 自动管理顺序和阶数累加
   * - 有显式 fn 的函数式编程
     - ``func.grad`` 链式
     - 与其他 func 原语（vmap, jacrev, jvp）自由组合
   * - 向量输入到向量输出 Jacobian
     - ``func.jacrev(f)`` / ``func.jacfwd(f)``
     - grad 只支持标量输出，向量输出必须用 jacrev/jacfwd
   * - 向量输入到向量输出二阶张量
     - ``func.jacrev(func.jacrev(f))``
     - Jacobian 的 Jacobian = 二阶偏导张量，一次算出
   * - 向量输入到标量输出 Hessian
     - ``func.jacrev(func.grad(fn))``
     - 结构化、一次算出、比手动两次 grad 高效
   * - 批量高阶导数
     - ``func.vmap(scalar_high_order)``
     - 批量处理自然，无需手动管理 batch 维
   * - 需要辅助值传递
     - ``func.grad(fn, has_aux=True)``
     - 求导同时透传非梯度信息
   * - 复杂控制流中的高阶
     - 直接 ``autograd.grad`` 循环
     - 最底层、最灵活，想插什么逻辑都可以

autograd.functional 模块求导函数
---------------------------------

Riemann 在 ``riemann.autograd.functional`` 模块中提供了一系列底层功能函数，
用于计算更高级的导数结构，如雅可比矩阵、Hessian 矩阵、雅可比向量积等。
这些函数接收**具体的输入张量**和**前向函数**，直接返回导数结果。

``jacobian()`` 函数
~~~~~~~~~~~~~~~~~~~~

``jacobian()`` 函数用于计算函数从输入到输出的雅可比矩阵 (Jacobian matrix)，展示了函数输出对输入的所有一阶偏导数。

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
    print(jac.shape)  # (3, 3)  # 对于向量输入/向量输出，结果形状为 (n_outputs, n_inputs)

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
    print(hess.shape)  # ()  # 标量输入返回标量二阶导

``derivative()`` 函数
~~~~~~~~~~~~~~~~~~~~~~~~~

``derivative()`` 函数用于为给定函数创建导函数。返回的新函数在调用时，计算原函数在给定输入处的雅可比矩阵（即导数）。

返回值始终为 tuple：每个输入对应一个导数张量；若原函数有多个输出，则每个输出再嵌套一层 tuple，与 ``jacobian()`` 的多输入/多输出返回结构一致。

.. code-block:: python

    import riemann as rm
    
    # 定义函数 f = x^2
    def f(x):
        return x ** 2
    
    # 创建导函数
    df = rm.autograd.functional.derivative(f)
    
    # 测试导函数
    x = rm.tensor(2.0, requires_grad=True)
    print(df(x))  # (tensor(4.),) — 始终返回 tuple，标量输入的导数为 0 维张量
    
    # 多输入示例
    def g(x, y):
        return x * y + x ** 2
    
    x = rm.tensor(2.0, requires_grad=True)
    y = rm.tensor(3.0, requires_grad=True)
    dg = rm.autograd.functional.derivative(g)
    print(dg(x, y))  # (tensor(7.), tensor(2.)) — 依次为 ∂g/∂x = y + 2x、∂g/∂y = x
    
    # create_graph=True 可在此基础上继续求高阶导数
    d2f = rm.autograd.functional.derivative(lambda t: t ** 3, create_graph=True)
    print(d2f(x))  # (tensor(12.),) — 3x^2，结果 requires_grad=True，可再次求导

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

``vjp()`` 函数用于计算给定向量与雅可比矩阵的乘积 (Vector-Jacobian Product)。

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

``hvp()`` 和 ``vhp()`` 函数分别用于计算 Hessian-向量积 (Hessian-Vector Product) 和向量-Hessian-积 (Vector-Hessian Product)。由于 Hessian 矩阵是对称的，``hvp()`` 和 ``vhp()`` 实际上是相同的。

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

func 模块求导函数
------------------

``riemann.func`` 模块提供了一套**函数式自动求导原语**，接口设计参考了 PyTorch 的 ``torch.func``。
与底层的 ``autograd.functional`` 相比，``func`` 模块采用更现代的 **柯里化（curry）风格**: 
先传入函数和参数配置，返回一个**包裹函数**；调用包裹函数时才真正执行前向和反向计算。

这种风格的优势在于：返回的包裹函数本身也是可组合的，可以与 ``vmap``、高阶 ``grad`` 等原语自由嵌套，
构建出 ``vmap(jacrev(grad(f)))`` 这样的复合变换。

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 特性
     - autograd.functional
     - func
   * - 调用风格
     - ``fn(f, *inputs)`` 传入具体输入
     - ``fn(f, ...)(*inputs)`` 返回包裹函数
   * - argnums 支持
     - 部分函数支持多输入位置 tuple
     - 全面支持，返回值按 argnums 过滤
   * - has_aux
     - 不支持
     - grad / grad_and_value 支持
   * - create_graph
     - 部分函数支持
     - 所有函数统一支持
   * - 组合能力
     - 不易与 vmap 等复合
     - 返回 Callable，可自由嵌套
   * - 默认 cotangents=None
     - 向量输出会报错
     - 自动填全 1，对齐 torch.func.vjp
   * - 非 leaf 输入
     - 需张量为 leaf 且 requires_grad=True
     - 自动 detach().requires_grad_(True) 处理

**导入方式**: 

.. code-block:: python

    import riemann as rm
    from riemann import func
    # 或简写
    from riemann.func import grad, jvp, vjp, jacrev, jacfwd, vmap, functional_call

``grad()`` — 标量值函数的梯度
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``func.grad`` 返回一个包裹函数，计算标量值函数在指定参数位置的梯度。

**函数签名**: 

.. code-block:: python

    riemann.func.grad(func, argnums=0, has_aux=False, create_graph=False) -> Callable

**参数**: 

- func：可微函数，输出必须是标量张量（``ndim == 0``），否则抛出 ``RuntimeError``
- argnums：需要求导的参数位置，可以是整数或整数元组，默认 ``0``：

  - 整数：只对该位置的参数求导，例如 ``argnums=1`` 表示对 ``func`` 的第二个参数求导
  - 整数元组：对元组中每个位置的参数分别求导，例如 ``argnums=(0, 1)``
  - 非法取值：``()``、``None``、含非整数元素的元组、list 等，在调用导函数时抛出 ``RuntimeError``，错误信息与 ``torch.func.grad`` 一致（``()`` 报 ``argnums must be non-empty``；``None`` 报 ``argnums must be int or Tuple[int, ...], got: <class 'NoneType'>``）
- has_aux：如果为 ``True``，``func`` 必须返回 ``(value, aux)`` 二元组。``value`` 是 ``func`` 的前向计算输出，``grad`` 要求它必须是标量张量；``aux`` 是 ``func`` 顺带输出的辅助结果（可以是任意对象，不参与求导）。导函数只从 ``value`` 计算梯度，并把 ``aux`` 原样透传在返回值中。这样可以在同一次前向计算中顺带获得不需要求导的结果（如损失值、统计量或其它中间输出），无需重复计算
- create_graph：如果为 ``True``，梯度张量会被追踪，可用于高阶微分

**返回**: ``func`` 的导函数（一个函数对象），它接受与 ``func`` 相同的参数。调用导函数时：

- ``has_aux=False``：返回 ``grad_result``
- ``has_aux=True``：返回 ``(grad_result, aux)``，其中 ``aux`` 为 ``func`` 原样透传的辅助结果
- ``grad_result`` 是标量输出 ``value`` 关于 ``argnums`` 所选参数的梯度：只含 1 个参数位置时为单个梯度张量（形状与对应输入张量相同），含多个位置时为多个梯度张量构成的元组（顺序与 ``argnums`` 一致）

.. note::

    三层输出容易混淆：``func`` 的输出是 ``value``（``has_aux=True`` 时为 ``(value, aux)``）；``func.grad(...)`` 的输出是一个函数对象（导函数）；导函数的输出才是 ``grad_result``（``has_aux=True`` 时为 ``(grad_result, aux)``）。

**示例**: 

.. code-block:: python

    import riemann as rm
    from riemann import func

    # 基本用法
    def f(x, y):
        return (x ** 2).sum() + x @ y  # 标量输出

    x = rm.tensor([1., 2.], requires_grad=True)
    y = rm.tensor([3., 4.], requires_grad=True)

    g = func.grad(f, argnums=0)      # 只对 x 求导
    grad_x = g(x, y)
    print(grad_x)  # 2*x + y = [2+3, 4+4] = [5, 8]

    # 多输入求导：argnums=(0, 1) 同时对两个参数求梯度
    g_both = func.grad(f, argnums=(0, 1))
    gx, gy = g_both(x, y)
    print(gx)  # 同 grad_x
    print(gy)  # x = [1., 2.]

    # has_aux：同时返回辅助数据
    def f_with_aux(x):
        return (x ** 3).sum(), float((x ** 2).sum() ** 0.5)

    g_aux = func.grad(f_with_aux, has_aux=True)
    grad, aux = g_aux(rm.tensor([1., 2., 3.], requires_grad=True))
    print(grad)  # [3., 12., 27.]
    print(aux)   # sqrt(1+4+9) ≈ 3.7417

    # create_graph=True 支持高阶导：内层 grad 返回带计算图的梯度，
    # 外层 grad 需要标量输出，因此对梯度做 .sum() 标量化
    grad_fn = func.grad(lambda x: (x ** 4).sum(), create_graph=True)
    g_higher = func.grad(lambda x: grad_fn(x).sum())
    h = g_higher(rm.tensor([1., 2., 3.], requires_grad=True))
    print(h)  # 二阶导 12*x² = [12., 48., 108.]

**注意事项**: 

- func 输出必须是**标量**（``ndim == 0``），否则抛出 ``RuntimeError``
- 返回值的形态取决于 ``argnums`` 中的参数位置数量：1 个位置返回单个张量，多个位置返回元组

``grad_and_value()`` — 同时返回梯度和函数值
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

比分开调用 ``grad`` + 前向更高效：内部只做一次前向 + 一次反向。

**函数签名**: 

.. code-block:: python

    riemann.func.grad_and_value(func, argnums=0, has_aux=False, create_graph=False) -> Callable

**参数**: 与 ``func.grad`` 相同，逐项说明如下：

- func：可微函数，输出必须是标量张量（``ndim == 0``），否则抛出 ``RuntimeError``
- argnums：需要求导的参数位置，整数或整数元组，默认 ``0``；整数表示只对该位置求导，整数元组表示对多个位置分别求导；非法取值（``()``、``None``、含非整数元素的元组、list 等）在调用导函数时抛 ``RuntimeError``
- has_aux：为 ``True`` 时 ``func`` 必须返回 ``(value, aux)`` 二元组，``value`` 是参与求导的标量输出，``aux`` 是不参与求导、原样透传的辅助结果
- create_graph：为 ``True`` 时梯度张量保留计算图，可继续求高阶导数

**返回**: ``func`` 的导函数（一个函数对象），它接受与 ``func`` 相同的参数。调用导函数时返回：

- ``has_aux=False``：``(grad_result, value)``
- ``has_aux=True``：``(grad_result, value, aux)``

其中 ``value`` 是 ``func`` 的前向输出（标量张量），``aux`` 是原样透传的辅助结果；``grad_result`` 的形态由 ``argnums`` 决定（同 ``func.grad``：1 个位置为单个梯度张量，多个位置为梯度元组）

**示例**: 

.. code-block:: python

    import riemann as rm
    from riemann import func

    def loss(params):
        return (params ** 2).sum()

    gav = func.grad_and_value(loss)
    grad, val = gav(rm.tensor([1., 2., 3.], requires_grad=True))
    # grad = [2., 4., 6.], val = 14.0

``jvp()`` — Jacobian-vector 积
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

计算 Jacobian 矩阵与给定向量的乘积。jvp 是真正的前向模式原语：先完成一次前向计算，
再沿计算图从输入到输出正向传播 tangent，无需反向传播，
比显式构造完整 Jacobian 再乘向量更高效（尤其当 Jacobian 矩阵很大时）。

**函数签名**: 

.. code-block:: python

    riemann.func.jvp(func, primals, tangents, create_graph=False, strict=False) -> (output, jvp_result)

**参数**: 

- func：可微函数，输入为一个或多个张量，输出可为单个张量或张量元组（输出可以是标量，也可以是非标量）
- primals：**求值点** (primal point)，即函数的前向输入，单个张量或元组/list（与 ``func`` 的多输入一一对应）。数学上 ``primal`` 指“原始值” x₀，与之相对的是无穷小扰动项（tangent）；传入的张量不需要 ``requires_grad=True``，内部会自动处理
- tangents：**切向量** (tangent vector)，与 ``primals`` 逐项对应的扰动方向 v，要求每一项的形状与对应输入完全相同（否则抛 ``RuntimeError``）；单输入时传单个张量，多输入时须传与 ``primals`` 等长的元组/list。把输入看成一条过求值点的曲线 x(t) = x₀ + t·v，那么 v 就是该曲线在 t = 0 处的切向量，而 JVP 正是沿该切向的方向导数：

  ``JVP = J(x₀)·v = d/dt f(x₀ + t·v) |_(t=0)``

  即“雅可比矩阵 × 切向量”，结果是输出空间中曲线 f(x₀ + t·v) 在 t = 0 处的切向量——这也正是（前向模式）自动微分的传播方式
- create_graph（bool，默认 ``False``）：为 ``True`` 时 JVP 结果保留计算图且可继续求导（支持高阶微分）；``False`` 时结果不带梯度追踪
- strict（bool，默认 ``False``）：严格模式。当某个输出与某个输入完全无关（该输入对应梯度为 ``None``）时，``False`` 按数学语义贡献 0；``True`` 则抛出 ``RuntimeError``

**返回**: 二元组 ``(output, jvp_result)``

- ``output``：**前向计算的输出值**，即 ``func(*primals)`` 的结果（单输出为一个张量，多输出为张量元组）。它只是求值点处的函数值，与导数无关，随返回值一并给出是为了避免重复执行前向计算
- ``jvp_result``：JVP 结果 ``J(x₀)·v``，**形状与 output 相同**；单输出时为单个张量，多输出时为与 output 逐项对应的元组

**示例**: 

.. code-block:: python

    import riemann as rm
    from riemann import func

    def f(x):
        return x ** 3  # f: R^n -> R^n, 向量输出

    x = rm.tensor([1., 2., 3.], requires_grad=True)
    v = rm.tensor([0.5, 1., 0.5], requires_grad=False)

    out, jvp_val = func.jvp(f, x, v)
    print(out)      # [1., 8., 27.] —— 前向输出 f(x)
    print(jvp_val)  # 3*x^2 * v = [3*0.5, 12*1, 27*0.5] = [1.5, 12., 13.5]

    # 多输入
    def g(x, y):
        return (x * y).sum()

    out2, jvp2 = func.jvp(g,
                          (rm.tensor([1., 2.], requires_grad=True),
                           rm.tensor([3., 4.], requires_grad=True)),
                          (rm.tensor([1., 0.], requires_grad=False),
                           rm.tensor([0., 1.], requires_grad=False)))
    print(jvp2)  # 1*3 + 2*1 = 5.0  (∂g/∂x·vx + ∂g/∂y·vy)

``vjp()`` — vector-Jacobian 积
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

计算给定向量与 Jacobian 矩阵的乘积（JVP 的反向对偶）。
对于标量输出 ``f: R^n -> R``，``vjp(f, x)`` 等价于 ``grad(f)(x)`` （默认 cotangent 为全 1）。

**函数签名**: 

.. code-block:: python

    riemann.func.vjp(func, primals, cotangents=None, create_graph=False, strict=False) -> (output, vjp_result)

**参数**: 

- func：可微函数，输入为一个或多个张量，输出可为单个张量或张量元组（输出可以是标量，也可以是非标量）
- primals：**求值点** (primal point)，即函数的前向输入，单个张量或元组/list（与 ``func`` 的多输入一一对应）。与 JVP 中的 ``primals`` 含义相同，是未加扰动的“原始值” x₀；不需要 ``requires_grad=True``，内部会自动处理
- cotangents：**余切向量** (cotangent vector)，输出空间的权向量 w，与 ``func`` 的每个输出逐项对应，且形状必须与对应输出完全相同（否则抛 ``RuntimeError``）；单输出时传单个张量，多输出时传与输出等长的元组/list。为 ``None`` 时自动按输出形状填全 1（标量输出填标量 1，对齐 ``torch.func.vjp`` 行为）。

  数学上，前向的雅可比 J 把输入切向量映射为输出切向量，而余切向量属于输出切空间的**对偶空间**：它不表示“往哪个方向走”，而是一个线性泛函，作用在输出增量 dy 上给出标量 wᵀ·dy。VJP 计算的就是

  ``VJP = wᵀ·J(x₀)``

  它等价于标量函数 L(x) = wᵀ·f(x) 在 x₀ 处的梯度 ``∇L(x₀)``。反向模式自动微分每一步传播的正是这种余切向量（即通常所说的“上游梯度”），因此 VJP 是反向模式的核心原语
- create_graph（bool，默认 ``False``）：为 ``True`` 时 ``output`` 与 ``vjp_result`` 均以可微方式计算（保留计算图），可继续求高阶导数；``False`` 时结果不带梯度追踪
- strict（bool，默认 ``False``）：严格模式。当某个输入与所有输出都无关（梯度为 ``None``）时，``False`` 返回同形状零张量（数学上 VJP 即为 0）；``True`` 则抛出 ``RuntimeError``

**返回**: 二元组 ``(output, vjp_result)``

- ``output``：**前向计算的输出值**，即 ``func(*primals)`` 的结果（单输出为一个张量，多输出为张量元组）。它只是求值点处的函数值，与导数无关，随返回值一并给出是为了避免重复执行前向计算
- ``vjp_result``：VJP 结果 ``wᵀ·J(x₀)``，形状与对应输入相同；单输入时为单个张量，多输入时为与 ``primals`` 逐项对应的元组。当 ``func`` 输出为标量且 ``cotangents=None`` 时（即 w = 1），它就退化为普通梯度

**示例**: 

.. code-block:: python

    import riemann as rm
    from riemann import func

    def f(x):
        return (x ** 2).sum() + x.sum()   # 标量输出

    x = rm.tensor([1., 2., 3.], requires_grad=True)

    # 默认 cotangent=None 等价于 [1., 1., 1.]
    out, vjp_val = func.vjp(f, x)
    print(out)      # 20.0 —— 前向输出 f(x)
    print(vjp_val)  # 2x + 1 = [3., 5., 7.]

    # 自定义 cotangent（标量输出对应标量 cotangent，相当于把函数输出加权求和）
    out2, vjp_val2 = func.vjp(f, x, rm.tensor(2.0))
    print(vjp_val2)  # 2 * (2x + 1) = [6., 10., 14.]

    # 向量输出也支持默认 cotangent
    def g(x):
        return x ** 2   # R^3 -> R^3, 向量输出

    out3, vjp_val3 = func.vjp(g, x)     # cotangent=None 自动填全1
    print(vjp_val3)  # J^T @ ones = [2., 4., 6.]

``jacrev()`` / ``jacfwd()`` — Jacobian 矩阵
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

两种方式计算 Jacobian 矩阵：

- jacrev：反向模式自动求导，由 vmap(vjp) 实现。对 m 维输出做 m 次反向传播，适合输入维度小、输出维度大的场景
- jacfwd：正向模式自动求导，由逐列 jvp 实现（等价于 vmap(jvp)，但共享一次前向求值与局部 VJP 图缓存）。对 n 维输入做 n 次前向传播，适合输入维度大、输出维度小的场景

**函数签名**: 

.. code-block:: python

    riemann.func.jacrev(func, argnums=0, create_graph=False, strict=True) -> Callable
    riemann.func.jacfwd(func, argnums=0, create_graph=False) -> Callable

**参数**: 

- func：可微函数，输入为一个或多个张量，输出可为单个张量或张量元组
- argnums：需要计算 Jacobian 的输入位置，整数或整数元组，默认 ``0``；非法取值（``()``、``None``、含非整数元素的元组、list 等）在调用导函数时抛 ``RuntimeError``，错误信息与 ``torch.func`` 一致（``torch.func.jacrev`` 对 ``()`` 和 ``None`` 均抛 ``RuntimeError``；``torch.func.jacfwd`` 对 ``()`` 抛 ``RuntimeError``，对 ``None`` 则按全部参数求导）
- create_graph（bool，默认 ``False``）：为 ``True`` 时 Jacobian 的每个元素都保留计算图，可再对其求导（例如 ``func.jacrev(func.jacrev(f, create_graph=True))`` 得到二阶导数）
- strict（bool，默认 ``True``，仅 ``jacrev`` 支持）：严格模式。当某个输出与某个输入完全无关（该输入对应的梯度为 ``None``）时，``True`` 抛出 ``RuntimeError``；``False`` 时该部分按数学语义填 0。``jacfwd`` 内部通过 ``vmap+jvp`` 实现，没有此参数

**返回**: 一个包裹函数，接受与 ``func`` 相同的参数；调用后返回 Jacobian 矩阵（形状见下）。只含 1 个参数位置时返回单个张量，含多个位置时返回元组（顺序与 ``argnums`` 一致）；``jacfwd`` 在 ``argnums=None`` 时按全部参数求导，此时即使只有 1 个参数也返回元组

**返回形状**: 对于输出形状 ``(y1, y2, ...)`` 和输入形状 ``(x1, x2, ...)``，
Jacobian 形状为 ``(y1, y2, ..., x1, x2, ...)``。

**示例**: 

.. code-block:: python

    import riemann as rm
    from riemann import func

    # 标量输出 f: R^3 -> R，Jacobian 形状为 (1, 3) — 等价于梯度
    f = lambda x: (x ** 3).sum()
    jac = func.jacrev(f)(rm.tensor([1., 2., 3.], requires_grad=True))
    print(jac.shape)  # (3,) —— 单元素输入被 unwrap
    print(jac)        # [3., 12., 27.]

    # 向量输出 f: R^3 -> R^2，Jacobian 形状为 (2, 3)
    g = lambda x: rm.stack([x.sum(), (x ** 2).sum()])
    jac_g = func.jacrev(g)(rm.tensor([1., 2., 3.], requires_grad=True))
    print(jac_g.shape)  # (2, 3)

**注意事项**: 

- jacfwd 在 Riemann 中通过逐列 ``jvp`` 实现（共享一次前向求值与局部 VJP 图缓存，等价于 ``vmap(jvp)``），时间复杂度正比于输入维度；若输出维度远大于输入维度，建议使用 ``jacrev``
- 多输入场景（``argnums=(0, 1)``）下，返回值为元组，每个元素对应一个输入的 Jacobian

``hessian()`` — Hessian 矩阵
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

计算标量值函数的 Hessian 矩阵（二阶偏导数）。实现方式为 ``jacrev(grad(func))``：
先由反向模式 ``grad`` 得到梯度，再由反向模式 ``jacrev`` 对梯度求 Jacobian。
（本库的前向模式由局部 VJP 转置模拟，``jacrev(jacfwd(func))`` 会为每个基向量生成
独立列子图并使外层反向遍历成本按 O(n²) 放大，因此改用反向叠反向。）

**函数签名**: 

.. code-block:: python

    riemann.func.hessian(func, argnums=0, create_graph=False, strict=True) -> Callable

**参数**: 

- func：可微函数，输出必须是标量张量（``ndim == 0``），向量输出会抛 ``RuntimeError``
- argnums：需要计算 Hessian 的输入位置，整数或整数元组，默认 ``0``；非法取值（``()``、``None``、含非整数元素的元组、list 等）在调用导函数时抛 ``RuntimeError``，错误信息与 ``torch.func.hessian`` 一致
- create_graph（bool，默认 ``False``）：为 ``True`` 时 Hessian 的每个元素都保留计算图，可继续求高阶导数；``False`` 时结果不带梯度追踪
- strict（bool，默认 ``True``）：严格模式。当函数对某个输入的梯度为 ``None``（即输出与该输入无关）时报错；当 ``argnums`` 未覆盖全部参数时，内部会自动放宽为 ``False``

**返回**: 一个包裹函数，接受与 ``func`` 相同的参数；调用后返回 Hessian 矩阵。只含 1 个参数位置时返回单个张量，含多个位置时返回元组（顺序与 ``argnums`` 一致）

**返回形状**: 对于形状 ``(n,)`` 的输入，Hessian 形状为 ``(n, n)``。

**示例**: 

.. code-block:: python

    import riemann as rm
    from riemann import func

    f = lambda x: (x ** 4).sum()
    h = func.hessian(f)(rm.tensor([1., 2., 3.], requires_grad=True))
    print(h.shape)  # (3, 3)
    # f''(x) = diag(12 * x^2) = diag([12, 48, 108])

**注意事项**: 

- 仅支持标量值函数，向量输出会抛错
- 当 ``argnums`` 不是全部参数位置时，内部自动放宽严格模式（cross-Hessian 零值不再报错）

``vmap()`` — 向量化映射
~~~~~~~~~~~~~~~~~~~~~~~~

将原本作用于**单个样本**的函数扩展到**批次维度**上执行。
底层通过循环 + 堆叠实现，接口对齐 ``torch.func.vmap``，方便与 ``jacrev`` 等函数式原语组合。

**函数签名**: 

.. code-block:: python

    riemann.func.vmap(func, in_dims=0, out_dims=0) -> Callable

**参数**: 

- func：需要被批处理的函数。它作用于单个样本（形如 ``func(x)``），输入/输出可以是张量或张量元组
- in_dims：每个输入的批次维度，默认 ``0``。
  - ``int``：所有输入共用该维度位置
  - ``tuple[int|None, ...]``：每个输入各自的维度位置；``None`` 表示该输入不批处理（直接透传，适合模型参数等共享量）
- out_dims：每个输出的批次维度，默认 ``0``，语义同 ``in_dims``；返回的每个输出都会沿该维度堆叠

**返回**: 一个包裹函数，接受带批次维的输入，返回带批次维的输出。单输出时返回单个张量，多输出（tuple）时返回与之结构相同的元组

**示例**: 

.. code-block:: python

    import riemann as rm
    from riemann import func

    # 基本用法：标量输出的批处理
    f = lambda x: (x ** 2).sum()
    x_batch = rm.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
    batched_f = func.vmap(f)
    result = batched_f(x_batch)
    print(result)  # [14., 77.]

    # 多输入 + 不同 in_dims
    def g(x, y):
        return (x * y).sum()

    x_batch = rm.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    y_batch = rm.tensor([[10., 20.], [30., 40.]], requires_grad=True)
    batched_g = func.vmap(g, in_dims=(0, 0))
    print(batched_g(x_batch, y_batch))  # [50., 250.]

    # 多输入 + in_dims=None 某输入不批处理（广播语义）
    y_single = rm.tensor([10., 20.], requires_grad=True)
    batched_g2 = func.vmap(g, in_dims=(0, None))
    print(batched_g2(x_batch, y_single))  # [30., 110.] —— y_single 被每个样本共享

    # 与 jacrev 组合：批量求 Jacobian
    def scalar_fn(x):
        return (x ** 2).sum()

    batched_jac = func.vmap(func.jacrev(scalar_fn))
    jac_batch = batched_jac(x_batch)
    print(jac_batch.shape)  # (2, 3) —— 每个样本的 ∇f = 2x

**注意事项**: 

- vmap 内部是**顺序循环** + 堆叠，并非 GPU 上真正的 SIMT 并行；
  但与 ``jacrev`` / ``hessian`` 等组合时能正确追踪梯度
- 批次维通常是 ``dim=0``；若指定负维度（如 ``dim=-1``），会自动规范化
- 返回 tuple 输出的函数也支持 vmap，每个输出沿 ``out_dims`` 独立堆叠

``functional_call()`` — 无状态 Module 调用
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

以**自定义参数值**调用 Module 的 ``forward`` 方法，不修改 Module 自身的参数属性。
这在需要对 Module 参数本身求导（如 hypernetworks、meta-learning）时至关重要——
常规 ``module(x)`` 调用会在 forward 内部引用 ``self.weight`` 等属性，
而 ``functional_call`` 允许你传入一组替代参数值。

**函数签名**: 

.. code-block:: python

    riemann.func.functional_call(module, parameter_values, args, kwargs=None) -> Any

**参数**: 

- module：带有 ``forward`` 和 ``named_parameters`` 方法的对象（通常是 ``riemann.nn.Module``）
- parameter_values：与 ``module.named_parameters()`` 顺序一致的参数值（list/tuple/TN）；为 ``None`` 时直接使用 Module 自身参数（退化为普通 ``forward`` 调用）；长度与参数数量不一致时抛 ``ValueError``
- args：传给 ``forward`` 的位置参数；传入单个张量时会自动包装成单元素元组，传入 list 会转成 tuple
- kwargs（dict，可选，默认 ``None``）：传给 ``forward`` 的关键字参数字典

**返回**: ``module.forward(*args, **kwargs)`` 的返回值（单输出为张量，多输出为元组，具体由 Module 自身的 ``forward`` 决定）

**示例**: 

.. code-block:: python

    import riemann as rm
    from riemann import func

    # 使用框架内置的 Linear 层，weight / bias 自动注册到 _parameters
    layer = rm.nn.Linear(3, 2)
    x = rm.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=False)

    # 用自定义参数调用（不修改 layer.weight 和 layer.bias）
    alt_weight = rm.tensor([[1., 0., 0.], [0., 0., 1.]], requires_grad=True)
    alt_bias = rm.tensor([0.5, -0.5], requires_grad=True)

    out = func.functional_call(layer, [alt_weight, alt_bias], x)
    print("out:", out)
    # layer.weight 和 layer.bias 仍是原值

    # 与 grad 组合：对替代参数求梯度
    loss = func.grad_and_value(lambda *params:
        (func.functional_call(layer, params, x) ** 2).sum()
    )
    g, v = loss(alt_weight, alt_bias)

    print("grad:", g)
    print("value:", v)

**注意事项**: 

- parameter_values 的长度必须与 ``module.named_parameters()`` 返回的参数数量一致
- 调用结束后 Module 参数会**自动恢复原值**（内部使用 try-finally）
- 对于没有参数的 Module，``parameter_values=None`` 时退化为普通 ``forward`` 调用

组合使用与高阶模式
~~~~~~~~~~~~~~~~~~

``func`` 模块的核心优势是**所有函数都返回 Callable**，这使得它们可以自由组合。
下面是几个常见的高阶用法模式：

**模式 1：vmap + jacrev — 批量求导**

.. code-block:: python

    import riemann as rm
    from riemann import func

    def loss(x):
        return (x ** 2).sum()

    x_batch = rm.tensor([[1., 2.], [3., 4.], [5., 6.]], requires_grad=True)
    grads = func.vmap(func.jacrev(loss))(x_batch)
    print(grads.shape)  # (3, 2)

**模式 2：jacrev(jacrev(f)) — 通过 jacrev 构造 Hessian**

.. code-block:: python

    import riemann as rm
    from riemann import func

    def f(x):
        return (x ** 4).sum()

    x = rm.tensor([1., 2., 3.], requires_grad=True)
    h_via_jac = func.jacrev(func.jacrev(f, create_graph=True))(x)
    print(h_via_jac)
    # 对角矩阵，对角元 [12·1², 12·2², 12·3²] = [12, 48, 108]

**模式 3：functional_call + grad — 对 Module 参数求梯度**

.. code-block:: python

    import riemann as rm
    from riemann import func

    layer = rm.nn.Linear(3, 2)
    x = rm.tensor([1., 2., 3.])
    alt_weight = rm.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], requires_grad=True)
    alt_bias = rm.tensor([0.01, 0.02], requires_grad=True)

    def param_loss(weight, bias):
        y = func.functional_call(layer, [weight, bias], x)
        return (y ** 2).sum()

    g_w, g_b = func.grad(param_loss, argnums=(0, 1))(alt_weight, alt_bias)
    print(g_w.shape, g_b.shape)

**模式 4：create_graph 链 — 高阶微分**

.. code-block:: python

    import riemann as rm
    from riemann import func

    f = lambda x: (x ** 4).sum()
    x = rm.tensor([1., 2., 3.], requires_grad=True)

    # 一阶
    g = func.grad(f, create_graph=True)
    print(g(x))
    # 二阶：对梯度再求 grad，同样需要 create_graph=True 才能继续链式高阶
    h = func.grad(lambda a: g(a).sum(), create_graph=True)
    print(h(x))
    # 三阶：继续叠加
    third = func.grad(lambda a: h(a).sum())(x)
    print(third)

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 场景
     - 推荐
     - 理由
   * - 只想快速算一次 jacobian/hessian
     - ``autograd.functional``
     - 直接传入输入，少一层包裹
   * - 需要与 vmap 组合、批量求导
     - ``func``
     - 返回 Callable 可嵌套
   * - 需要 has_aux / argnums 多输入
     - ``func``
     - ``autograd.functional`` 不支持
   * - 需要对 Module 参数求梯度
     - ``func.functional_call``
     - 底层模块无此功能
   * - 向量输出 vjp 默认 cotangent
     - ``func.vjp``
     - 自动填全 1，更安全
   * - create_graph 高阶微分
     - ``func``
     - 所有函数统一支持

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

   传递给 ``track_grad`` 的梯度函数签名固定为 ``grad_func(result, *args, **kwargs)``：

   - **参数**: 第一个参数 ``result`` 是前向函数的输出张量；其后参数必须与前向函数的输入参数同名同序
   - **返回值**: 返回一个 tuple，按 ``grad_func`` 签名中的 TN 参数顺序给出每个输入张量的梯度
   - **元组元素**: 只有 TN 参数占位，非 TN 参数（标量、布尔值等）不占位；
     不需要梯度的 TN 参数对应位置返回 ``None``
   - **链式法则**: 返回值必须已包含上游梯度，可通过 ``result.grad_value`` 获取；
     同时可以读取 ``result`` 的其他属性（shape、dtype、device 等）
   - **调用顺序无关**: 内部通过 ``inspect.signature`` 建立“参数名 → 实参值”的映射，
     因此调用时关键字参数顺序、位置参数与关键字参数混用都不会影响梯度对应关系

   **场景 1：单输入函数**

   .. code-block:: python

       import riemann as rm
       import numpy as np
       
       def sigmoid_derivative(result, x):
           """sigmoid 函数的梯度：返回包含一个元素的元组"""
           # result 即 sigmoid 输出，直接复用可省去重复计算
           return (result.grad_value * result * (1. - result),)
       
       @rm.track_grad(sigmoid_derivative)
       def custom_sigmoid(x):
           """带梯度支持的自定义 sigmoid 函数"""
           return rm.tensor(1. / (1. + np.exp(-x.array())))
       
       # 测试自定义 sigmoid 函数的梯度
       x = rm.tensor(0.0, requires_grad=True)
       y = custom_sigmoid(x)
       y.backward()
       print(f"Sigmoid 输出: {y}")  # 应为 0.5
       print(f"Sigmoid 梯度: {x.grad}")  # 应为 0.25

   **场景 2：多输入函数（关键字参数顺序打乱）**

   ``grad_func`` 的返回值按函数签名顺序排列，与调用时传参顺序无关。

   .. code-block:: python

       import riemann as rm
       
       def divide_derivative(result, x, y):
           """除法梯度：d(x/y)/dx = 1/y, d(x/y)/dy = -x/y²"""
           return (
               result.grad_value / y,
               result.grad_value * (-x / y**2.),
           )
       
       @rm.track_grad(divide_derivative)
       def custom_divide(x, y):
           return x / y
       
       x = rm.tensor(5.0, requires_grad=True)
       y = rm.tensor(2.0, requires_grad=True)
       # 关键字顺序与签名 (x, y) 相反，梯度仍能正确对应
       z = custom_divide(y=y, x=x)
       z.backward()
       print(f"z = {z}")        # 2.5
       print(f"dz/dx = {x.grad}")  # 0.5
       print(f"dz/dy = {y.grad}")  # -1.25

   **场景 3：TN 与非 TN 参数混合、带默认参数**

   非 TN 参数不占用梯度元组的位置；默认参数在未显式传入时同样按签名参与映射。

   .. code-block:: python

       import riemann as rm
       
       def scaled_pow_derivative(result, x, scale=1.0, exponent=2.0):
           """d(scale * x^n)/dx = scale * n * x^(n-1)"""
           local = exponent * x ** (exponent - 1.)
           return (result.grad_value * scale * local,)
       
       @rm.track_grad(scaled_pow_derivative)
       def scaled_pow(x, scale=1.0, exponent=2.0):
           return scale * x ** exponent
       
       x = rm.tensor(2.0, requires_grad=True)
       y = scaled_pow(x, exponent=3.0)   # scale 使用默认值 1.0
       y.backward()
       print(f"y = {y}")        # 8.0
       print(f"dy/dx = {x.grad}")  # 12.0 (3 * 2²)

   **场景 4：变长参数（\*args 与 \*\*kwargs）**

   当 ``grad_func`` 使用 ``*args`` / ``**kwargs`` 时，返回的梯度元组按签名顺序依次展平。

   .. code-block:: python

       import riemann as rm
       
       # *args：每个输入的梯度都是上游梯度
       def sum_derivative(result, *args):
           return tuple(result.grad_value for _ in args)
       
       @rm.track_grad(sum_derivative)
       def custom_sum(*xs):
           return rm.tensor(sum(x.array() for x in xs))
       
       a = rm.tensor(1.0, requires_grad=True)
       b = rm.tensor(2.0, requires_grad=True)
       s = custom_sum(a, b)
       s.backward()
       print(f"da = {a.grad}, db = {b.grad}")  # 1.0, 1.0
       
       # **kwargs：按参数名取对应权重
       WEIGHTS = {"a": 2.0, "b": 3.0}
       
       def weighted_sum_derivative(result, **kwargs):
           return tuple(result.grad_value * WEIGHTS[name] for name in kwargs)
       
       @rm.track_grad(weighted_sum_derivative)
       def custom_weighted_sum(**kwargs):
           return rm.tensor(sum(WEIGHTS[name] * v.array() for name, v in kwargs.items()))
       
       va = rm.tensor(1.0, requires_grad=True)
       vb = rm.tensor(1.0, requires_grad=True)
       w = custom_weighted_sum(a=va, b=vb)
       w.backward()
       print(f"dva = {va.grad}, dvb = {vb.grad}")  # 2.0, 3.0

   **场景 5：部分输入不需要梯度**

   .. code-block:: python

       import riemann as rm
       
       def add_derivative(result, x, y):
           return (result.grad_value, result.grad_value)
       
       @rm.track_grad(add_derivative)
       def custom_add(x, y):
           return x + y
       
       x = rm.tensor(2.0, requires_grad=True)
       y = rm.tensor(3.0, requires_grad=False)  # 不需要梯度
       z = custom_add(x, y)
       z.backward()
       print(f"dz/dx = {x.grad}")  # 1.0
       print(f"dz/dy = {y.grad}")  # None

   **场景 6：矩阵乘法（自定义链式法则）**

   这是 ``track_grad`` 的核心优势：反向传播中可以直接使用矩阵乘 ``@``、转置等运算，
   而不是被限制为“上游梯度逐元素相乘”。

   .. code-block:: python

       import riemann as rm
       
       def matmul_derivative(result, x, y):
           """矩阵乘反向：dL/dx = dL/dY @ Yᵀ, dL/dY = Xᵀ @ dL/dY"""
           g = result.grad_value
           return (g @ y.T, x.T @ g)
       
       @rm.track_grad(matmul_derivative)
       def custom_matmul(x, y):
           return rm.matmul(x, y)
       
       x = rm.tensor([[1., 2.], [3., 4.]], requires_grad=True)
       y = rm.tensor([[5., 6.], [7., 8.]], requires_grad=True)
       z = custom_matmul(x, y)
       z.sum().backward()
       print(f"dx = {x.grad}")  # [[11., 15.], [11., 15.]]
       print(f"dy = {y.grad}")  # [[4., 4.], [6., 6.]]

   **注意事项：**

   - 梯度函数返回的是“已经乘以上游梯度的最终梯度”，漏乘 ``result.grad_value`` 会丢失链式法则的上游部分
   - 元组顺序只由 ``grad_func`` 签名决定，与调用时的传参顺序无关；非 TN 参数不占位
   - ``backward`` 每次调用都会按当前上游梯度重新计算 ``grad_func``，
     因此支持 ``retain_graph=True`` 下的重复反向传播
   - 若前向函数完全由 Riemann 张量算子构成（已自动求导），通常无需使用 ``track_grad``；
     它主要用于包装外部算子或手动定义前向/反向的计算

3. **使用 Function 类**
   对于更复杂的情况，你可以继承 ``Function`` 类并实现 ``forward`` 和 ``backward`` 静态方法。

   **Function 类接口：**

   要使用 ``Function`` 类创建自定义函数，必须实现两个静态方法：

   **forward(ctx, \*inputs)**

   - **用途**: 执行前向计算
   - **参数**: 

     - ``ctx``：上下文对象，用于保存反向传播所需的信息。使用 ``ctx.save_for_backward()`` 保存反向传播需要的张量
     - ``*inputs``：输入张量（可变数量的参数）
   - **返回**: 前向计算的输出张量
   - **用法**: 在此实现自定义计算逻辑，并使用 ``ctx.save_for_backward()`` 保存梯度计算所需的任何张量

   **backward(ctx, grad_output)**

   - **用途**: 执行反向（梯度）计算
   - **参数**: 

     - ``ctx``：上下文对象，包含前向传播期间保存的信息。通过 ``ctx.saved_tensors`` 访问保存的张量
     - ``grad_output``：输出张量的梯度（来自计算图中后续层）
   - **返回**: 梯度元组，每个输入张量对应一个梯度。每个梯度应为 ``grad_output`` 与局部梯度（偏导数）的乘积
   - **用法**: 使用链式法则计算梯度：``grad_input = grad_output * local_gradient``

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
               sig = 1. / (1. + np.exp(-x.detach().array()))
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

手动修改计算图
--------------

Riemann 提供了用于手动操作计算图的函数。这些函数专为特殊场景设计，用于在不影响前向计算值或反向传播梯度值的情况下将张量连接到计算图。这些函数是底层工具，通常用于框架内部（如 Riemann 的钩子处理机制），而非普通用户场景。

share_grad_map 函数
~~~~~~~~~~~~~~~~~~~

``share_grad_map`` 函数将一组张量全连接映射为个数一样的另一组张量，两组张量间对应位置恒等映射，前向透传张量值，反向透传梯度值，相当于clone关系，其它连接上，前向透传0值（即不影响新张量值），反向传递0梯度。

**用途：**
确保组中的所有张量都参与计算图并收到梯度（对于不直接参与计算的张量为零），而不是 None。这在希望某些张量收到零梯度而不改变现有计算图的前向或反向计算值时特别有用。

**核心机制：**

1. 对于每个需要梯度的张量，创建一个克隆。克隆张量通过 ``clone`` 操作依赖原始张量（梯度透传）
2. 将所有其他张量（排除自身）作为零梯度来源依附到克隆张量
3. 这样每个张量既保持与原始张量的梯度关系，又与其他张量形成零梯度连接

**参数：**

- tensors：要连接的张量元组或列表。必须是元组或列表（不是集合）以保持顺序。

**返回：**
具有相同值但连接到共享计算图的张量元组或列表。注意：``requires_grad=True`` 的张量会被克隆（不会被原地修改）。

**行为：**

- requires_grad=True 的张量会被克隆，并将所有其他张量作为零梯度来源依附到克隆张量
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
    # 注意：梯度回传到原始张量 a/b/c，克隆张量.grad 为 None
    print(f"c.grad (origin): {c.grad}")  # 输出: [0.0, 0.0]
    
    # 验证：前向值相同，a 和 b 的梯度不变
    assert y1.item() == y2.item()
    assert (a.grad == rm.tensor([3., 4.])).all()
    assert (b.grad == rm.tensor([1., 2.])).all()
    assert (c.grad == rm.tensor([0., 0.])).all()

**典型使用场景：**

1. **模块钩子处理**: 在 Riemann 的模块钩子机制中，``share_grad_map`` 用于产生新的模块输出张量，接替模块原有的输出张量。当模块只有部分张量参与或贡献损失函数计算时，``share_grad_map`` 产生模块新的输出，不改变前向计算和反向梯度计算的数值，但可以确保原来不参与损失函数计算的输出张量现在可以接收到零梯度了，从而依赖该输出的模块输入张量也会收到零梯度。

2. **多任务学习**: 当某些参数不参与特定任务的损失计算，但你希望它们收到零梯度而不是 None，以便进行梯度累积。

3. **条件计算**: 当某些张量在条件判断下被使用，但你希望无论条件如何都有一致的梯度行为。

4. **梯度监控**: 当你想要监控组中所有参数的梯度，即使那些不直接参与特定计算的参数。

**注意：** 这是一个用于手动构建计算图的底层函数。大多数用户应该依赖 Riemann 的自动图构建，而不是直接使用此函数。

辅助函数和方法
~~~~~~~~~~~~~~

以下函数和方法由 ``share_grad_map`` 内部使用，用户很少需要直接调用：

**attach_zero_grad_sources 方法**

将多个张量作为来源张量依附到目标张量。这不改变张量的值，但允许它在反向传播期间向这些来源传递零梯度。内部用于将张量连接到计算图，使它们收到零梯度而不是 None。

梯度检查
-----------------

可以用有限差分法手动验证自定义梯度函数是否正确：

.. code-block:: python

    import riemann as rm
    import numpy as np

    # 手动数值梯度检查（类似 PyTorch 的 gradcheck 原理）
    def numerical_grad(func, x, eps=1e-5):
        x_flat = x._array.astype(np.float64).copy()
        grad_flat = np.zeros_like(x_flat)
        it = np.nditer(x_flat, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            old_val = x_flat[idx]
            x_flat[idx] = old_val + eps
            f_plus = func(rm.tensor(x_flat.astype(np.float32))).item()
            x_flat[idx] = old_val - eps
            f_minus = func(rm.tensor(x_flat.astype(np.float32))).item()
            x_flat[idx] = old_val
            grad_flat[idx] = (f_plus - f_minus) / (2 * eps)
            it.iternext()
        return rm.tensor(grad_flat.astype(np.float32))

    # 定义一个简单函数测试
    def f(x):
        return (x ** 3).sum()

    x = rm.tensor([1., 2., 3.], requires_grad=True)
    y = f(x)
    y.backward()

    num_grad = numerical_grad(f, x)
    print(f"解析梯度: {x.grad}")
    print(f"数值梯度: {num_grad}")
    print(f"最大误差: {(x.grad - num_grad).abs().max()}")
    # 最大误差应小于 1e-4

Gradcheck 通过比较解析梯度和有限差分法计算的数值梯度来验证梯度计算的正确性。Riemann 目前未提供内置的 ``gradcheck`` 函数，可使用上述数值梯度检查方法代替。

梯度计算技巧
-------------------------

1. **内存管理**: 梯度计算使用内存来存储计算图。当你不需要梯度时，使用 ``no_grad()`` 或 ``detach()`` 来节省内存。

常见错误
--------

1. **原地操作**: 避免对需要梯度跟踪的叶子节点张量进行原地操作。

2. **清除张量的计算依赖**: detach后的张量会失去计算图依赖关系，无法进行反向传播梯度计算。

3. **非标量输出**: 在非标量输出上调用 ``backward()`` 时，记得提供梯度参数。

4. **内存泄漏**: 长时间运行带梯度跟踪的计算可能会消耗大量内存。

示例
----

Rosenbrock 函数优化 (香蕉函数)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rosenbrock 函数（也称为香蕉函数）是一个经典的非凸优化问题。该函数在 (1, 1) 处取得最小值 0。

以下是使用 Riemann 的自动求导和 Adam 优化器来优化 Rosenbrock 函数的示例：

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
