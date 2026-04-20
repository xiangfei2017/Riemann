爱因斯坦求和约定 (einsum)
==========================

爱因斯坦求和约定是数学和物理学中描述张量运算的简洁表示法，由阿尔伯特·爱因斯坦在1916年提出。在深度学习框架中，``einsum`` 函数利用这一约定，提供了一种统一、优雅且强大的方式来表达各种张量运算。

什么是爱因斯坦求和约定
--------------------------

**数学背景**

在传统的张量代数中，复杂的张量运算通常涉及大量的求和符号和索引标记。爱因斯坦求和约定通过省略求和符号，使表达式更加简洁明了。其核心规则是：**当同一个索引在一个项中出现两次时，表示对该索引进行求和**。

例如，传统的矩阵乘法可以表示为：

.. math::

    C_{ik} = \sum_{j} A_{ij} B_{jk}

使用爱因斯坦求和约定，可以简化为：

.. math::

    C_{ik} = A_{ij} B_{jk}

这里，索引 ``j`` 在等式右边出现了两次（分别在 ``A`` 和 ``B`` 中），表示对 ``j`` 进行求和。

**einsum 的核心思想**

Riemann 中的 ``einsum`` 函数实现了这一约定，允许用户通过简洁的字符串方程来描述复杂的张量运算。其优势包括：

1. **统一性**：一个函数可以替代多种不同的张量运算
2. **可读性**：方程字符串直观地表达了运算的数学含义
3. **灵活性**：支持任意维度的张量和复杂的索引操作
4. **效率**：内部优化确保高性能计算

einsum 方程字符串语法
----------------------

**基本语法结构**

einsum 方程字符串遵循以下格式：

.. code-block:: python

    "输入1索引,输入2索引,...->输出索引"

其中：

- **输入索引**：用字母表示对应输入张量的维度，如 ``ij`` 表示2D张量（矩阵）
- **输出索引**：指定输出张量的维度，省略时表示按字母顺序排列
- **省略号 ``...``**：表示任意数量的批量维度（batch dimensions）
- **重复索引**：表示在该维度上进行求和（缩并）

**索引规则详解**

1. **唯一索引**：如果一个索引只在一个输入中出现，且在输出中也出现，表示该维度被保留
2. **重复索引**：如果一个索引在多个输入中出现，表示在这些维度上进行元素级乘法后求和
3. **缺失索引**：如果输入中的索引未在输出中出现，表示对该维度进行求和归约

**示例解析**

.. code-block:: python

    # 矩阵乘法: ij,jk->ik
    # i: A的行索引, j: A的列索引/B的行索引(求和), k: B的列索引
    # 结果C的维度为 (i, k)
    C = rm.einsum('ij,jk->ik', A, B)

    # 批量矩阵乘法: ...ij,...jk->...ik
    # ... 表示任意批量维度
    C = rm.einsum('...ij,...jk->...ik', A, B)

    # 迹运算: ii->
    # i 重复出现，表示对角线元素求和
    trace = rm.einsum('ii->', A)

    # 对角线提取: ii->i
    # 保留对角线元素，结果为向量
    diag = rm.einsum('ii->i', A)

einsum 计算场景分类
--------------------

下表详细列出了 einsum 可以替代的各种计算场景：

**基础矩阵运算**

.. list-table:: 基础矩阵运算
    :widths: 20 25 35 20
    :header-rows: 1

    * - 运算类型
      - 数学描述
      - einsum 方程
      - 等价函数
    * - 矩阵乘法
      - :math:`C_{ik} = \sum_j A_{ij} B_{jk}`
      - ``ij,jk->ik``
      - ``rm.matmul(A, B)``
    * - 批量矩阵乘法
      - :math:`C_{bik} = \sum_j A_{bij} B_{bjk}`
      - ``bij,bjk->bik``
      - ``rm.matmul(A, B)``
    * - 通用批量矩阵乘
      - 支持任意批量维度
      - ``...ij,...jk->...ik``
      - ``rm.matmul(A, B)``
    * - 向量点积
      - :math:`c = \sum_i a_i b_i`
      - ``i,i->``
      - ``rm.dot(a, b)``
    * - 向量外积
      - :math:`C_{ij} = a_i b_j`
      - ``i,j->ij``
      - ``rm.outer(a, b)``

**矩阵属性提取**

.. list-table:: 矩阵属性提取
    :widths: 20 25 35 20
    :header-rows: 1

    * - 运算类型
      - 数学描述
      - einsum 方程
      - 等价函数
    * - 矩阵迹
      - :math:`\text{tr}(A) = \sum_i A_{ii}`
      - ``ii->``
      - ``rm.trace(A)``
    * - 对角线提取
      - :math:`\text{diag}(A)_i = A_{ii}`
      - ``ii->i``
      - ``rm.diag(A)``
    * - 批量矩阵迹
      - :math:`\text{tr}(A_b) = \sum_i A_{bii}`
      - ``bii->b``
      - ``rm.trace(A)``
    * - 批量对角线提取
      - :math:`\text{diag}(A_b)_i = A_{bii}`
      - ``bii->bi``
      - ``rm.diag(A)``

**转置与维度重排**

.. list-table:: 转置与维度重排
    :widths: 20 25 35 20
    :header-rows: 1

    * - 运算类型
      - 数学描述
      - einsum 方程
      - 等价函数
    * - 矩阵转置
      - :math:`C_{ji} = A_{ij}`
      - ``ij->ji``
      - ``A.T`` 或 ``rm.transpose(A)``
    * - 高维转置
      - :math:`C_{jki} = A_{ijk}`
      - ``ijk->jki``
      - ``rm.permute(A, (1, 2, 0))``
    * - 批量转置
      - :math:`C_{bji} = A_{bij}`
      - ``...ij->...ji``
      - ``A.mT``

**张量缩并与求和**

.. list-table:: 张量缩并与求和
    :widths: 20 25 35 20
    :header-rows: 1

    * - 运算类型
      - 数学描述
      - einsum 方程
      - 等价函数
    * - 全元素求和
      - :math:`s = \sum_{i,j} A_{ij}`
      - ``ij->``
      - ``rm.sum(A)``
    * - 按行求和
      - :math:`s_i = \sum_j A_{ij}`
      - ``ij->i``
      - ``rm.sum(A, dim=1)``
    * - 按列求和
      - :math:`s_j = \sum_i A_{ij}`
      - ``ij->j``
      - ``rm.sum(A, dim=0)``
    * - 张量缩并
      - :math:`C_{ijm} = \sum_{k,l} A_{ijkl} B_{jklm}`
      - ``ijkl,jklm->ijm``
      - 无直接等价
    * - 自缩并
      - :math:`C_i = \sum_j A_{ij} B_{ij}`
      - ``ij,ij->i``
      - ``(A * B).sum(dim=1)``

**特殊矩阵运算**

.. list-table:: 特殊矩阵运算
    :widths: 20 25 35 20
    :header-rows: 1

    * - 运算类型
      - 数学描述
      - einsum 方程
      - 等价函数
    * - Hadamard积
      - :math:`C_{ij} = A_{ij} B_{ij}`
      - ``ij,ij->ij``
      - ``A * B``
    * - Frobenius内积
      - :math:`\langle A, B \rangle_F = \sum_{i,j} A_{ij} B_{ij}`
      - ``ij,ij->``
      - ``(A * B).sum()``
    * - Kronecker积
      - :math:`C_{ikjl} = A_{ij} B_{kl}`
      - ``ij,kl->ikjl``
      - ``rm.kron(A, B)``
    * - 恒等复制
      - :math:`C_{ij} = A_{ij}`
      - ``ij->ij``
      - ``A.clone()``

**多操作数运算**

.. list-table:: 多操作数运算
    :widths: 20 25 35 20
    :header-rows: 1

    * - 运算类型
      - 数学描述
      - einsum 方程
      - 说明
    * - 三操作数链式
      - :math:`C_{il} = \sum_{j,k} A_{ij} B_{jk} C_{kl}`
      - ``ij,jk,kl->il``
      - 连续矩阵乘法
    * - 四操作数链式
      - :math:`C_{im} = \sum_{j,k,l} A_{ij} B_{jk} C_{kl} D_{lm}`
      - ``ij,jk,kl,lm->im``
      - 长链矩阵乘法
    * - 多操作数混合
      - :math:`C_i = \sum_{j,k} A_{ij} B_{jk} C_{ik}`
      - ``ij,jk,ik->i``
      - 复杂混合运算
    * - 批量三操作数
      - 支持任意批量维度
      - ``...ij,...jk,...kl->...il``
      - 批量链式乘法

**重复索引运算**

.. list-table:: 重复索引运算
    :widths: 20 25 35 20
    :header-rows: 1

    * - 运算类型
      - 数学描述
      - einsum 方程
      - 说明
    * - 前两个索引重复
      - :math:`C_j = \sum_{i} A_{iij}`
      - ``iij->j``
      - 提取特定对角线
    * - 多索引重复
      - :math:`C_{ij} = \sum_{k,l} A_{iijj}`
      - ``iijj->ij``
      - 高维对角线提取
    * - 非连续索引重复
      - :math:`s = \sum_{i,j} A_{ijji}`
      - ``ijji->``
      - 反对角线求和

**1D向量运算**

.. list-table:: 1D向量运算
    :widths: 20 25 35 20
    :header-rows: 1

    * - 运算类型
      - 数学描述
      - einsum 方程
      - 等价函数
    * - 向量点积
      - :math:`c = \sum_i a_i b_i`
      - ``i,i->``
      - ``rm.dot(a, b)``
    * - 向量外积
      - :math:`C_{ij} = a_i b_j`
      - ``i,j->ij``
      - ``rm.outer(a, b)``
    * - 矩阵乘向量
      - :math:`c_i = \sum_j A_{ij} b_j`
      - ``ij,j->i``
      - ``rm.matmul(A, b)``
    * - 向量乘矩阵
      - :math:`c_j = \sum_i a_i A_{ij}`
      - ``i,ij->j``
      - ``rm.matmul(a, A)``
    * - 批量矩阵乘向量
      - :math:`C_{bi} = \sum_j A_{bij} b_{bj}`
      - ``bij,bj->bi``
      - ``rm.matmul(A, b.unsqueeze(-1)).squeeze(-1)``

einsum 使用示例
----------------

**示例1：矩阵乘法**

.. code-block:: python

    import riemann as rm

    # 创建矩阵
    A = rm.tensor([[1, 2], [3, 4], [5, 6]])  # 3x2
    B = rm.tensor([[7, 8, 9], [10, 11, 12]])  # 2x3

    # 矩阵乘法: (3x2) @ (2x3) = (3x3)
    C = rm.einsum('ij,jk->ik', A, B)
    print("矩阵乘法结果:")
    print(C)
    # 输出:
    # tensor([[ 27,  30,  33],
    #         [ 61,  68,  75],
    #         [ 95, 106, 117]])

**示例2：批量矩阵乘法**

.. code-block:: python

    import riemann as rm

    # 创建批量矩阵 (2个3x4矩阵)
    A = rm.randn(2, 3, 4)
    # 创建批量矩阵 (2个4x5矩阵)
    B = rm.randn(2, 4, 5)

    # 批量矩阵乘法
    C = rm.einsum('bij,bjk->bik', A, B)
    print(f"批量矩阵乘法结果形状: {C.shape}")  # (2, 3, 5)

    # 使用省略号支持更多批量维度
    A = rm.randn(2, 3, 4, 5)  # 2x3个4x5矩阵
    B = rm.randn(2, 3, 5, 6)  # 2x3个5x6矩阵
    C = rm.einsum('...ij,...jk->...ik', A, B)
    print(f"通用批量乘法结果形状: {C.shape}")  # (2, 3, 4, 6)

**示例3：迹和对角线运算**

.. code-block:: python

    import riemann as rm

    # 创建方阵
    A = rm.tensor([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

    # 计算迹（对角线元素之和）
    trace = rm.einsum('ii->', A)
    print(f"矩阵迹: {trace}")  # 1 + 5 + 9 = 15

    # 提取对角线元素
    diag = rm.einsum('ii->i', A)
    print(f"对角线元素: {diag}")  # [1, 5, 9]

    # 批量矩阵迹
    batch_A = rm.randn(4, 3, 3)  # 4个3x3矩阵
    batch_trace = rm.einsum('bii->b', batch_A)
    print(f"批量迹形状: {batch_trace.shape}")  # (4,)

**示例4：转置和维度重排**

.. code-block:: python

    import riemann as rm

    # 创建矩阵
    A = rm.tensor([[1, 2, 3],
                   [4, 5, 6]])

    # 矩阵转置
    A_T = rm.einsum('ij->ji', A)
    print("转置结果:")
    print(A_T)
    # 输出:
    # tensor([[1, 4],
    #         [2, 5],
    #         [3, 6]])

    # 高维转置
    B = rm.randn(2, 3, 4)
    B_perm = rm.einsum('ijk->jki', B)
    print(f"高维转置形状: {B_perm.shape}")  # (3, 4, 2)

**示例5：向量运算**

.. code-block:: python

    import riemann as rm

    # 创建向量
    a = rm.tensor([1, 2, 3])
    b = rm.tensor([4, 5, 6])

    # 向量点积
    dot_product = rm.einsum('i,i->', a, b)
    print(f"点积: {dot_product}")  # 1*4 + 2*5 + 3*6 = 32

    # 向量外积
    outer_product = rm.einsum('i,j->ij', a, b)
    print("外积结果:")
    print(outer_product)
    # 输出:
    # tensor([[ 4,  5,  6],
    #         [ 8, 10, 12],
    #         [12, 15, 18]])

    # 矩阵乘向量
    A = rm.tensor([[1, 2, 3],
                   [4, 5, 6]])
    c = rm.einsum('ij,j->i', A, a)
    print(f"矩阵乘向量: {c}")  # [14, 32]

**示例6：张量缩并**

.. code-block:: python

    import riemann as rm

    # 创建3D张量
    A = rm.randn(2, 3, 4)
    B = rm.randn(3, 4, 5)

    # 张量缩并：在维度1和2上求和
    C = rm.einsum('ijk,jkl->il', A, B)
    print(f"张量缩并结果形状: {C.shape}")  # (2, 5)

    # 更复杂的缩并
    D = rm.randn(2, 3, 4, 5)
    E = rm.randn(3, 4, 5, 6)
    F = rm.einsum('ijkl,jklm->im', D, E)
    print(f"复杂缩并结果形状: {F.shape}")  # (2, 6)

**示例7：Hadamard积和Frobenius内积**

.. code-block:: python

    import riemann as rm

    # 创建矩阵
    A = rm.tensor([[1, 2],
                   [3, 4]])
    B = rm.tensor([[5, 6],
                   [7, 8]])

    # Hadamard积（逐元素乘法）
    hadamard = rm.einsum('ij,ij->ij', A, B)
    print("Hadamard积:")
    print(hadamard)
    # 输出:
    # tensor([[ 5, 12],
    #         [21, 32]])

    # Frobenius内积
    frobenius = rm.einsum('ij,ij->', A, B)
    print(f"Frobenius内积: {frobenius}")  # 5 + 12 + 21 + 32 = 70

**示例8：多操作数运算**

.. code-block:: python

    import riemann as rm

    # 创建多个矩阵
    A = rm.randn(3, 4)
    B = rm.randn(4, 5)
    C = rm.randn(5, 6)
    D = rm.randn(6, 7)

    # 四操作数链式乘法
    result = rm.einsum('ij,jk,kl,lm->im', A, B, C, D)
    print(f"四操作数链式结果形状: {result.shape}")  # (3, 7)

    # 等价于：
    # temp1 = rm.matmul(A, B)
    # temp2 = rm.matmul(temp1, C)
    # result = rm.matmul(temp2, D)

**示例9：带梯度跟踪的einsum**

.. code-block:: python

    import riemann as rm

    # 创建需要梯度的张量
    A = rm.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    B = rm.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    # 执行einsum运算
    C = rm.einsum('ij,jk->ik', A, B)

    # 计算损失并反向传播
    loss = C.sum()
    loss.backward()

    print("A的梯度:")
    print(A.grad)
    print("B的梯度:")
    print(B.grad)

**示例10：隐式输出（省略输出索引）**

.. code-block:: python

    import riemann as rm

    A = rm.tensor([[1, 2], [3, 4]])
    B = rm.tensor([[5, 6], [7, 8]])

    # 隐式输出：省略->后的部分
    # 结果按字母顺序排列索引
    C = rm.einsum('ij,jk', A, B)  # 等价于 'ij,jk->ik'
    print("隐式输出结果:")
    print(C)

    # 批量隐式输出
    A = rm.randn(2, 3, 4)
    B = rm.randn(2, 4, 5)
    C = rm.einsum('...ij,...jk', A, B)  # 等价于 '...ij,...jk->...ik'
    print(f"批量隐式输出形状: {C.shape}")  # (2, 3, 5)

einsum 性能优化建议
--------------------

1. **优先使用简单方程**：对于常见的矩阵乘法，直接使用 ``rm.matmul`` 可能更高效
2. **避免不必要的复制**：einsum 会尽可能返回视图而非副本
3. **批量操作优于循环**：使用 ``...`` 表示批量维度，避免显式循环
4. **链式运算合并**：多个矩阵乘法可以合并为一个einsum调用，减少中间结果
5. **预编译方程**：对于重复使用的相同方程，einsum 会自动缓存优化

注意事项
--------

1. **索引字母限制**：索引使用小写字母（a-z），最多支持26个不同索引
2. **维度匹配**：重复索引的维度大小必须一致
3. **设备一致性**：所有输入张量必须在同一设备上
4. **数据类型**：einsum 遵循常规的类型提升规则
5. **梯度跟踪**：支持自动微分，可以正常计算梯度
