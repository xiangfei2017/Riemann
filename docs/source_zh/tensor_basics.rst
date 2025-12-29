张量基础
========

张量是 Riemann 中的核心数据结构，类似于 NumPy 数组，但具有额外的自动微分和梯度跟踪功能。

创建张量
--------

从数据创建
~~~~~~~~~~

您可以直接从 Python 列表或 NumPy 数组创建张量：

.. code-block:: python

    import riemann as rm
    import numpy as np
    
    # 从 Python 列表创建
    x = rm.tensor([1, 2, 3])
    print(x)  # tensor([1, 2, 3])
    
    # 从 NumPy 数组创建
    np_array = np.array([1, 2, 3])
    x = rm.tensor(np_array)
    print(x)  # tensor([1, 2, 3])

指定数据类型
~~~~~~~~~~~~

创建张量时可以指定数据类型：

.. code-block:: python

    # Float32 张量（默认）
    x = rm.tensor([1, 2, 3], dtype=rm.float32)
    
    # Float64 张量
    x = rm.tensor([1, 2, 3], dtype=rm.float64)
    
    # 复数张量
    x = rm.tensor([1+2j, 3+4j], dtype=rm.complex64)

特殊张量
~~~~~~~~

Riemann 提供了创建特殊张量的函数：

.. code-block:: python

    # 零张量
    x = rm.zeros(3, 4)
    
    # 全一张量
    x = rm.ones(2, 3)
    
    # 单位矩阵
    x = rm.eye(3)
    
    # 随机张量
    x = rm.randn(2, 3)  # 正态分布
    x = rm.rand(2, 3)   # 均匀分布 [0, 1)

张量属性
--------

张量有几个重要的属性：

.. code-block:: python

    x = rm.tensor([[1, 2, 3], [4, 5, 6]], dtype=rm.float32, requires_grad=True)
    
    # 形状
    print(x.shape)  # (2, 3)
    
    # 数据类型
    print(x.dtype)  # float32
    
    # 维度数量
    print(x.ndim)  # 2
    
    # 元素总数
    print(x.numel())  # 6
    
    # 梯度跟踪
    print(x.requires_grad)  # True

张量操作
--------

基本算术运算
~~~~~~~~~~~~

张量支持标准算术运算：

.. code-block:: python

    x = rm.tensor([1, 2, 3])
    y = rm.tensor([4, 5, 6])
    
    # 加法
    z = x + y
    
    # 减法
    z = x - y
    
    # 乘法（逐元素）
    z = x * y
    
    # 除法
    z = x / y
    
    # 矩阵乘法
    a = rm.tensor([[1, 2], [3, 4]])
    b = rm.tensor([[5, 6], [7, 8]])
    c = a @ b  # 矩阵乘法

数学函数
~~~~~~~~

Riemann 提供了广泛的数学函数：

.. code-block:: python

    x = rm.tensor([0, rm.pi/4, rm.pi/2])
    
    # 三角函数
    y = rm.sin(x)
    y = rm.cos(x)
    y = rm.tan(x)
    
    # 指数和对数函数
    y = rm.exp(x)
    y = rm.log(x)
    y = rm.sqrt(x)
    
    # 双曲函数
    y = rm.sinh(x)
    y = rm.cosh(x)
    y = rm.tanh(x)

形状操作
~~~~~~~~

.. code-block:: python

    x = rm.tensor([[1, 2, 3], [4, 5, 6]])
    
    # 重塑
    y = x.reshape(3, 2)
    
    # 转置
    y = x.T
    
    # 压缩和扩展维度
    y = x.unsqueeze(0)  # 在位置 0 添加维度
    y = y.squeeze(0)   # 在位置 0 移除维度
    
    # 拼接
    a = rm.tensor([1, 2, 3])
    b = rm.tensor([4, 5, 6])
    c = rm.cat([a, b], dim=0)  # tensor([1, 2, 3, 4, 5, 6])

梯度跟踪
--------

启用梯度跟踪
~~~~~~~~~~~~

要启用自动微分，请在创建张量时设置 ``requires_grad=True``：

.. code-block:: python

    x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * 2
    z = y.sum()
    
    # 计算梯度
    z.backward()
    print(x.grad)  # tensor([2., 2., 2.])

禁用梯度跟踪
~~~~~~~~~~~~

当不需要梯度时，可以禁用梯度跟踪以提高性能：

.. code-block:: python

    x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    # 方法 1：使用 no_grad 上下文
    with rm.no_grad():
        y = x * 2  # 此操作不进行梯度跟踪
    
    # 方法 2：使用 requires_grad_
    x.requires_grad_(False)
    y = x * 2  # 不进行梯度跟踪

原地操作
--------

原地操作直接修改张量而不创建新张量：

.. code-block:: python

    x = rm.tensor([1, 2, 3])
    
    # 原地加法
    x += 1  # 同 x.add_(1)
    
    # 原地乘法
    x *= 2  # 同 x.mul_(2)
    
    # 原地赋值
    x[0] = 10

注意：原地操作在梯度跟踪时可能会有问题。当 ``requires_grad=True`` 时请谨慎使用。

保存和加载张量
--------------

您可以使用 Riemann 的序列化函数保存和加载张量：

.. code-block:: python

    # 创建张量
    x = rm.tensor([1, 2, 3])
    
    # 保存到文件
    rm.save(x, 'tensor.pt')
    
    # 从文件加载
    y = rm.load('tensor.pt')
    print(y)  # tensor([1, 2, 3])