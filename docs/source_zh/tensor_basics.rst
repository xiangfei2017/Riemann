张量基础
========

什么是张量
----------

张量是 Riemann 中的核心数据结构，本质上是一个0维或多维的数组，用于量化描述客观事物，如文本、图像、视频、语音等等。

在 Riemann 中，张量具有以下特点：

- **多维数组结构**：支持0维（标量）、1维（向量）、2维（矩阵）以及更高维度的数组表示
- **数学运算支持**：支持加减乘除、内积等基本数学运算，以及各种常见数学函数
- **形状变换能力**：支持张量形状重塑、维度扩缩、索引和切片等操作
- **自动梯度跟踪**：内置自动微分机制，支持梯度计算和反向传播
- **设备兼容性**：支持在CPU和GPU等不同设备上运行

张量是构建神经网络和梯度下降类算法的基础。您熟悉的数学中的0维标量、1维向量、2维矩阵都是张量的特殊形式。

需要注意的是，Riemann 中的张量与张量代数或张量分析里说的张量不完全等价，主要是运算规则上有些差异。这里提到的张量主要服务于神经网络相关的计算，其本质是多维数组、支持多种运算符和函数、支持自动梯度跟踪。

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

指定设备
~~~~~~~~

创建张量时可以指定设备：

.. code-block:: python

    # CPU 张量（默认）
    x = rm.tensor([1, 2, 3], device='cpu')
    
    # CUDA 张量
    x = rm.tensor([1, 2, 3], device='cuda')
    
    # 指定 CUDA 设备索引
    x = rm.tensor([1, 2, 3], device='cuda:0')

启用梯度跟踪
~~~~~~~~~~~~

创建张量时可以指定是否需要梯度跟踪：

.. code-block:: python

    # 不需要梯度（默认）
    x = rm.tensor([1, 2, 3], requires_grad=False)
    
    # 需要梯度（仅对浮点类型有效）
    x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)

tensor函数参数详解
~~~~~~~~~~~~~~~~~~

**tensor函数签名**:

.. code-block:: python

    def tensor(data, dtype=None, device=None, requires_grad=False) -> TN

**参数说明**:

- **data**: 可以是任意可转换为numpy数组的数据，包括列表、元组、标量、numpy数组等
- **dtype**: 可选，指定张量的数据类型。如果为None，则根据data的类型自动推断
- **device**: 可选，指定张量所在的设备，可以是'cpu'、'cuda'、'cuda:0'、整数索引或Device对象。如果为None，则使用当前设备上下文或默认设备
- **requires_grad**: 可选，布尔值，指定是否需要计算该张量的梯度，默认为False

**参数为None时的处理逻辑**:

**dtype为None时**:

- 如果data是numpy数组或cupy数组，保留原始数据类型
- 如果data是Python标量：
  - bool → bool
  - int → int64
  - float → 默认浮点类型（默认为float32）
  - complex → 默认复数类型（默认为complex64）
- 如果data是Python列表或元组，根据元素类型推断数据类型（选择能够容纳所有元素的最小类型）

**device为None时**:

- 首先检查是否在CUDA设备上下文中
- 如果在CUDA上下文中，使用当前CUDA设备
- 否则使用默认设备（默认为CPU）

**使用示例**:

.. code-block:: python

    # 基本用法
    x = rm.tensor([1, 2, 3])
    
    # 完整参数示例
    x = rm.tensor(
        data=[1.0, 2.0, 3.0],
        dtype=rm.float32,
        device='cuda',
        requires_grad=True
    )

**默认数据类型和默认设备的查询与设置**:

**默认数据类型**:

.. code-block:: python

    # 获取当前默认浮点类型
    default_dtype = rm.get_default_dtype()
    print(default_dtype)  # 默认为 float32
    
    # 设置默认浮点类型
    rm.set_default_dtype(rm.float64)
    print(rm.get_default_dtype())  # 现在为 float64

**默认设备**:

.. code-block:: python

    # 获取当前默认设备
    default_device = rm.get_default_device()
    print(default_device)  # 默认为 device(type='cpu', index=None)
    
    # 设置默认设备
    rm.set_default_device('cuda')
    print(rm.get_default_device())  # 现在为 device(type='cuda', index=0)
    
    # 设置指定CUDA设备为默认设备
    rm.set_default_device('cuda:1')
    print(rm.get_default_device())  # 现在为 device(type='cuda', index=1)

**示例：使用默认设置创建张量**:

.. code-block:: python

    # 设置默认设备为CUDA
    rm.set_default_device('cuda')
    
    # 设置默认数据类型为float64
    rm.set_default_dtype(rm.float64)
    
    # 创建张量时不指定device和dtype
    # 会使用默认设置
    x = rm.tensor([1.0, 2.0, 3.0])
    print(x.device)  # cuda:0
    print(x.dtype)   # float64

数据类型和设备初始化
~~~~~~~~~~~~~~~~~~~~

**dtype对象初始化**:

Riemann支持多种方式初始化数据类型：

.. code-block:: python

    # 使用Riemann内置dtype
    dtype = rm.float32
    dtype = rm.float64
    dtype = rm.int32
    dtype = rm.int64
    dtype = rm.complex64
    dtype = rm.complex128
    
    # 使用NumPy dtype
    import numpy as np
    dtype = np.float32
    dtype = np.dtype('float64')
    
    # 使用字符串
    dtype = 'float32'
    dtype = 'float64'

**Device对象初始化**:

Riemann的Device对象可以通过以下方式初始化：

.. code-block:: python

    # 使用字符串
    device = rm.device('cpu')
    device = rm.device('cuda')
    device = rm.device('cuda:0')
    
    # 使用整数索引（仅CUDA）
    device = rm.device(0)  # 等价于 'cuda:0'
    
    # 通过Device构造函数
    from riemann import Device
    device = Device('cpu')
    device = Device('cuda:1')

**Device对象属性**:

.. code-block:: python

    device = rm.device('cuda:0')
    print(device.type)  # 'cuda'
    print(device.index)  # 0
    
    device = rm.device('cpu')
    print(device.type)  # 'cpu'
    print(device.index)  # None

Device上下文管理
~~~~~~~~~~~~~~~~

Riemann支持使用上下文管理器来临时切换设备，在with块内创建的张量会默认使用指定的设备：

.. code-block:: python

    import riemann as rm
    
    # 在CPU上创建张量
    x = rm.tensor([1, 2, 3])
    print(x.device)  # cpu
    
    # 临时切换到CUDA设备
    with rm.device('cuda'):
        # 在CUDA上创建张量
        y = rm.tensor([4, 5, 6])
        print(y.device)  # cuda:0
        
        # 不指定device参数时，默认使用上下文设备
        z = rm.tensor([7, 8, 9])
        print(z.device)  # cuda:0
    
    # 退出上下文后，恢复默认设备
    w = rm.tensor([10, 11, 12])
    print(w.device)  # cpu

**上下文管理的优势**:

- 避免在每次创建张量时重复指定device参数
- 确保代码块内的所有张量都在同一设备上
- 自动恢复之前的设备状态，避免设备状态混乱

**使用场景**:

.. code-block:: python

    # 示例：在CUDA上执行计算密集型操作
    with rm.device('cuda'):
        # 创建输入张量
        input_data = rm.tensor([[1.0, 2.0], [3.0, 4.0]])
        
        # 执行计算
        result = rm.matmul(input_data, input_data)
        
        # 计算结果会自动在CUDA上
        print(result.device)  # cuda:0

特殊张量
~~~~~~~~

Riemann 提供了丰富的特殊张量创建函数，下表列出了所有支持的函数及其功能：

.. list-table:: 特殊张量创建函数
    :widths: 20 60 20
    :header-rows: 1

    * - 函数名
      - 功能描述
      - 示例
    * - ``zeros``
      - 创建指定形状的全零张量
      - ``zeros(3, 4)``
    * - ``zeros_like``
      - 创建与给定张量形状相同的全零张量
      - ``zeros_like(x)``
    * - ``ones``
      - 创建指定形状的全一张量
      - ``ones(2, 3)``
    * - ``ones_like``
      - 创建与给定张量形状相同的全一张量
      - ``ones_like(x)``
    * - ``empty``
      - 创建指定形状的未初始化张量
      - ``empty(2, 3)``
    * - ``empty_like``
      - 创建与给定张量形状相同的未初始化张量
      - ``empty_like(x)``
    * - ``full``
      - 创建指定形状并填充指定值的张量
      - ``full((2, 3), 5)``
    * - ``full_like``
      - 创建与给定张量形状相同并填充指定值的张量
      - ``full_like(x, 5)``
    * - ``eye``
      - 创建单位矩阵
      - ``eye(3)``
    * - ``rand``
      - 创建均匀分布 [0, 1) 的随机张量
      - ``rand(2, 3)``
    * - ``randn``
      - 创建标准正态分布的随机张量
      - ``randn(2, 3)``
    * - ``randint``
      - 创建指定范围内的整数随机张量
      - ``randint(0, 10, (2, 3))``
    * - ``normal``
      - 创建指定均值和标准差的正态分布随机张量
      - ``normal(0, 1, (2, 3))``
    * - ``randperm``
      - 创建从 0 到 n-1 的随机排列张量
      - ``randperm(5)``
    * - ``arange``
      - 创建按步长递增的一维张量
      - ``arange(0, 10, 2)``
    * - ``linspace``
      - 创建指定范围内的等间隔一维张量
      - ``linspace(0, 1, 5)``
    * - ``from_numpy``
      - 从 NumPy 或 CuPy 数组创建张量
      - ``from_numpy(np_array)``

**使用示例**:

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

    # 填充张量
    x = rm.full((2, 3), 5)  # 创建值为 5 的 2x3 张量

    # 序列张量
    x = rm.arange(0, 10, 2)  # 0, 2, 4, 6, 8
    x = rm.linspace(0, 1, 5)  # 0, 0.25, 0.5, 0.75, 1.0

    # 从 NumPy 数组创建
    import numpy as np
    np_array = np.array([1, 2, 3])
    x = rm.from_numpy(np_array)

特殊张量的默认参数行为
~~~~~~~~~~~~~~~~~~~~~~~~

当创建特殊张量时，如果不指定 `dtype` 和 `device` 参数，会根据函数类型使用不同的默认行为：

无参考张量的创建函数（如 zeros, ones, rand 等）：

- 当不指定 `dtype` 和 `device` 参数时，函数行为与 `tensor()` 函数的行为一致
- 默认数据类型为 `float32`
- 默认设备为当前的设备上下文或默认设备设置

有参考张量的 like 类函数（如 zeros_like, ones_like 等）：

- 当不指定 `dtype` 和 `device` 参数时，使用参考张量的 `dtype` 和 `device` 来创建张量
- 这确保了新创建的张量与参考张量具有相同的数据类型和设备

**默认数据类型 (dtype)**:

.. code-block:: python

    # 默认创建 float32 张量
    x = rm.zeros(3, 4)
    print(x.dtype)  # float32
    
    # 显式指定数据类型
    x = rm.zeros(3, 4, dtype=rm.float64)
    print(x.dtype)  # float64

**默认设备 (device)**:

- 当不指定 `device` 参数时，会使用当前的设备上下文或默认设备设置
- 这与 `tensor()` 函数的默认行为一致

.. code-block:: python

    # 默认使用当前设备上下文或默认设备
    x = rm.zeros(3, 4)
    print(x.device)  # 默认为 cpu
    
    # 在 CUDA 上下文内创建
    with rm.device('cuda'):
        x = rm.zeros(3, 4)
        print(x.device)  # cuda:0
    
    # 显式指定设备
    x = rm.zeros(3, 4, device='cuda')
    print(x.device)  # cuda:0

**完整参数示例**:

.. code-block:: python

    # 指定所有参数
    x = rm.zeros(
        3, 4,            # 形状
        dtype=rm.float32,  # 数据类型
        device='cuda',    # 设备
        requires_grad=True  # 梯度跟踪
    )
    
    # 随机张量示例
    x = rm.randn(
        2, 3,            # 形状
        dtype=rm.float64,  # 数据类型
        device='cpu',     # 设备
        requires_grad=False  # 梯度跟踪
    )

张量属性与状态
--------------

张量具有多种属性和状态检测函数，用于获取张量的基本信息和检测其状态。

**张量属性**

.. list-table:: 张量属性
  :widths: 15 45 40
  :header-rows: 1

  * - 属性名
    - 功能描述
    - 示例
  * - ``dtype``
    - 张量的数据类型
    - ``x.dtype`` → ``float32``
  * - ``device``
    - 张量所在的设备
    - ``x.device`` → ``cpu`` 或 ``cuda:0``
  * - ``ndim``
    - 张量的维度数量
    - ``x.ndim`` → ``2``
  * - ``shape``
    - 张量的形状
    - ``x.shape`` → ``(2, 3)``
  * - ``size``
    - 张量在指定维度上的大小
    - ``x.size(0)`` → ``2``
  * - ``numel``
    - 张量的元素总数
    - ``x.numel()`` → ``6``
  * - ``is_leaf``
    - 张量是否为计算图中的叶子节点
    - ``x.is_leaf`` → ``True``
  * - ``requires_grad``
    - 张量是否需要梯度跟踪
    - ``x.requires_grad`` → ``True``

**状态检测函数**

.. list-table:: 状态检测函数
  :widths: 15 45 40
  :header-rows: 1

  * - 函数名
    - 功能描述
    - 示例
  * - ``is_floating_point``
    - 检测张量是否为浮点类型
    - ``x.is_floating_point()`` → ``True``
  * - ``is_complex``
    - 检测张量是否为复数类型
    - ``x.is_complex()`` → ``False``
  * - ``isreal``
    - 检测张量是否为实数类型
    - ``x.isreal()`` → ``True``
  * - ``isinf``
    - 检测张量元素是否为无穷大
    - ``x.isinf()`` → 布尔张量
  * - ``isnan``
    - 检测张量元素是否为 NaN
    - ``x.isnan()`` → 布尔张量
  * - ``is_cuda``
    - 检测张量是否在 CUDA 设备上
    - ``x.is_cuda`` → ``False``
  * - ``is_cpu``
    - 检测张量是否在 CPU 设备上
    - ``x.is_cpu`` → ``True``
  * - ``type``
    - 获取或设置张量的数据类型
    - ``x.type()`` → ``float32`` 或 ``x.type(rm.float64)``
  * - ``is_contiguous``
    - 检测张量是否为连续存储
    - ``x.is_contiguous()`` → ``True``

**属性与状态检测示例**

.. code-block:: python

    import riemann as rm

    # 创建一个张量
    x = rm.tensor([[1, 2, 3], [4, 5, 6]], dtype=rm.float32, requires_grad=True)
    print("原始张量:", x)

    # 1. 基本属性
    print("\n1. 基本属性:")
    print("数据类型:", x.dtype)
    print("设备:", x.device)
    print("维度数量:", x.ndim)
    print("形状:", x.shape)
    print("第0维大小:", x.size(0))
    print("元素总数:", x.numel())
    print("是否为叶子节点:", x.is_leaf)
    print("是否需要梯度跟踪:", x.requires_grad)

    # 2. 状态检测
    print("\n2. 状态检测:")
    print("是否为浮点类型:", x.is_floating_point())
    print("是否为复数类型:", x.is_complex())
    print("是否为实数类型:", x.isreal())
    print("是否在 CUDA 设备上:", x.is_cuda)
    print("是否在 CPU 设备上:", x.is_cpu)
    print("是否为连续存储:", x.is_contiguous())

    # 3. 特殊值检测
    print("\n3. 特殊值检测:")
    y = rm.tensor([1.0, float('inf'), float('nan')])
    print("张量:", y)
    print("是否为无穷大:", y.isinf())
    print("是否为 NaN:", y.isnan())

    # 4. 类型操作
    print("\n4. 类型操作:")
    print("当前类型:", x.type())
    x_double = x.type(rm.float64)
    print("转换为 float64 后的类型:", x_double.type())

张量运算
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

Riemann 提供了广泛的数学函数，以下是常用的数学函数列表：

**基本数学函数**

.. list-table:: 基本数学函数
  :widths: 15 45 40
  :header-rows: 1

  * - 函数名
    - 功能描述
    - 示例
  * - ``abs``
    - 计算绝对值
    - ``rm.abs(x)``
  * - ``sqrt``
    - 计算平方根
    - ``rm.sqrt(x)``
  * - ``square``
    - 计算平方
    - ``rm.square(x)``
  * - ``exp``
    - 计算指数函数
    - ``rm.exp(x)``
  * - ``exp2``
    - 计算 2 的幂
    - ``rm.exp2(x)``
  * - ``log``
    - 计算自然对数
    - ``rm.log(x)``
  * - ``log10``
    - 计算以 10 为底的对数
    - ``rm.log10(x)``
  * - ``log2``
    - 计算以 2 为底的对数
    - ``rm.log2(x)``
  * - ``sign``
    - 计算符号函数
    - ``rm.sign(x)``
  * - ``ceil``
    - 向上取整
    - ``rm.ceil(x)``
  * - ``floor``
    - 向下取整
    - ``rm.floor(x)``
  * - ``round``
    - 四舍五入
    - ``rm.round(x)``
  * - ``trunc``
    - 截断小数部分
    - ``rm.trunc(x)``

**三角函数**

.. list-table:: 三角函数
  :widths: 15 45 40
  :header-rows: 1

  * - 函数名
    - 功能描述
    - 示例
  * - ``sin``
    - 计算正弦值
    - ``rm.sin(x)``
  * - ``cos``
    - 计算余弦值
    - ``rm.cos(x)``
  * - ``tan``
    - 计算正切值
    - ``rm.tan(x)``
  * - ``arcsin``
    - 计算反正弦值
    - ``rm.arcsin(x)``
  * - ``arccos``
    - 计算反余弦值
    - ``rm.arccos(x)``
  * - ``arctan``
    - 计算反正切值
    - ``rm.arctan(x)``
  * - ``arctan2``
    - 计算两个张量的反正切值
    - ``rm.arctan2(y, x)``

**双曲函数**

.. list-table:: 双曲函数
  :widths: 15 45 40
  :header-rows: 1

  * - 函数名
    - 功能描述
    - 示例
  * - ``sinh``
    - 计算双曲正弦值
    - ``rm.sinh(x)``
  * - ``cosh``
    - 计算双曲余弦值
    - ``rm.cosh(x)``
  * - ``tanh``
    - 计算双曲正切值
    - ``rm.tanh(x)``
  * - ``arcsinh``
    - 计算反双曲正弦值
    - ``rm.arcsinh(x)``
  * - ``arccosh``
    - 计算反双曲余弦值
    - ``rm.arccosh(x)``
  * - ``arctanh``
    - 计算反双曲正切值
    - ``rm.arctanh(x)``

**数学函数示例**

.. code-block:: python

    import riemann as rm

    # 创建示例张量
    x = rm.tensor([-2.5, 0.0, 1.5, 3.0])
    print("原始张量:", x)

    # 基本数学函数
    print("\n1. 基本数学函数:")
    print("绝对值:", rm.abs(x))
    print("平方根:", rm.sqrt(rm.abs(x)))
    print("平方:", rm.square(x))
    print("指数:", rm.exp(x))
    print("自然对数:", rm.log(rm.abs(x) + 1e-10))

    # 取整函数
    print("\n2. 取整函数:")
    print("向上取整:", rm.ceil(x))
    print("向下取整:", rm.floor(x))
    print("四舍五入:", rm.round(x))
    print("截断:", rm.trunc(x))

    # 三角函数
    print("\n3. 三角函数:")
    angles = rm.tensor([0, rm.pi/4, rm.pi/2, rm.pi])
    print("角度:", angles)
    print("正弦:", rm.sin(angles))
    print("余弦:", rm.cos(angles))
    print("正切:", rm.tan(angles))

    # 双曲函数
    print("\n4. 双曲函数:")
    print("双曲正弦:", rm.sinh(x))
    print("双曲余弦:", rm.cosh(x))
    print("双曲正切:", rm.tanh(x))

统计函数
~~~~~~~~~

Riemann 提供了多种张量统计函数，用于张量分析。以下是常用的统计函数列表：

**常用统计函数**

.. list-table:: 统计函数
  :widths: 15 45 40
  :header-rows: 1

  * - 函数名
    - 功能描述
    - 示例
  * - ``sum``
    - 计算张量元素的和
    - ``rm.sum(x)``
  * - ``sumall``
    - 计算多个张量的和
    - ``rm.sumall(x, y, z)``
  * - ``mean``
    - 计算张量元素的平均值
    - ``rm.mean(x)``
  * - ``var``
    - 计算张量元素的方差
    - ``rm.var(x)``
  * - ``std``
    - 计算张量元素的标准差
    - ``rm.std(x)``
  * - ``norm``
    - 计算张量的范数
    - ``rm.norm(x)``
  * - ``max``
    - 计算张量元素的最大值
    - ``rm.max(x)``
  * - ``min``
    - 计算张量元素的最小值
    - ``rm.min(x)``
  * - ``maximum``
    - 计算两个张量的元素级最大值
    - ``rm.maximum(x, y)``
  * - ``minimum``
    - 计算两个张量的元素级最小值
    - ``rm.minimum(x, y)``
  * - ``where``
    - 根据条件选择元素
    - ``rm.where(condition, x, y)``
  * - ``clamp``
    - 将张量值限制在一个范围内
    - ``rm.clamp(x, min, max)``
  * - ``sort``
    - 对张量元素进行排序
    - ``rm.sort(x)``
  * - ``argsort``
    - 返回对张量排序的索引
    - ``rm.argsort(x)``
  * - ``argmax``
    - 返回最大值的索引
    - ``rm.argmax(x)``
  * - ``argmin``
    - 返回最小值的索引
    - ``rm.argmin(x)``
  * - ``prod``
    - 计算张量元素的乘积
    - ``rm.prod(x)``
  * - ``dot``
    - 计算两个张量的点积
    - ``rm.dot(x, y)``

**统计函数示例**

.. code-block:: python

    import riemann as rm

    # 创建示例张量
    x = rm.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    y = rm.tensor([[2.0, 1.0, 4.0], [3.0, 6.0, 5.0], [8.0, 7.0, 10.0]])
    z = rm.tensor([[1.0, 3.0, 2.0], [4.0, 2.0, 1.0], [3.0, 4.0, 5.0]])
    print("原始张量 x:", x)
    print("原始张量 y:", y)
    print("原始张量 z:", z)

    # 1. sum 函数
    print("\n1. sum 函数:")
    print("所有元素的和:", rm.sum(x))
    print("沿轴 0 的和:", rm.sum(x, dim=0))
    print("沿轴 1 的和:", rm.sum(x, dim=1))

    # 2. sumall 函数
    print("\n2. sumall 函数:")
    print("多个张量的和:", rm.sumall(x, y, z))

    # 3. mean 函数
    print("\n3. mean 函数:")
    print("所有元素的平均值:", rm.mean(x))
    print("沿轴 0 的平均值:", rm.mean(x, dim=0))

    # 4. max 和 min 函数
    print("\n4. max 和 min 函数:")
    print("最大值:", rm.max(x))
    print("最小值:", rm.min(x))
    print("沿轴 0 的最大值:", rm.max(x, dim=0))
    print("沿轴 1 的最小值:", rm.min(x, dim=1))

    # 5. where 函数
    print("\n5. where 函数:")
    condition = x > 5
    result = rm.where(condition, x, y)
    print("条件 (x > 5):", condition)
    print("where 结果:", result)

**where 函数详细示例**

where 函数有两种主要用法：
1. 当不提供 x 和 y 时，返回满足条件的元素的索引
2. 当提供 x 和 y 时，根据条件从 x 和 y 中选择元素

.. code-block:: python

    import riemann as rm

    # 创建示例张量
    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = rm.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
    print("原始张量 x:")
    print(x)
    print("原始张量 y:")
    print(y)

    # 用法 1: 只提供条件，返回满足条件的元素索引
    print("\n用法 1: 只提供条件")
    condition = x > 5
    indices = rm.where(condition)
    print("条件 (x > 5):")
    print(condition)
    print("满足条件的元素索引:")
    print("行索引:", indices[0])
    print("列索引:", indices[1])
    print("索引元组:", indices)

    # 用法 2: 提供条件、x 和 y，根据条件选择元素
    print("\n用法 2: 提供条件、x 和 y")
    
    # 基本用法
    result1 = rm.where(condition, x, y)
    print("基本用法结果 (x > 5 时取 x，否则取 y):")
    print(result1)
    
    # 使用标量作为 x 或 y
    result2 = rm.where(condition, 100, y)
    print("\n使用标量作为 x 的结果 (x > 5 时取 100，否则取 y):")
    print(result2)
    
    result3 = rm.where(condition, x, 0)
    print("\n使用标量作为 y 的结果 (x > 5 时取 x，否则取 0):")
    print(result3)

    # 使用不同形状的张量（会自动广播）
    print("\n使用不同形状的张量")
    condition_1d = rm.tensor([True, False, True])  # 1D 条件
    x_1d = rm.tensor([100, 200, 300])  # 1D x
    
    result4 = rm.where(condition_1d, x_1d, y)
    print("1D 条件和 1D x 与 2D y 的结果:")
    print(result4)

    # 带梯度跟踪的 where 函数
    print("\n带梯度跟踪的 where 函数")
    x_grad = rm.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    y_grad = rm.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], requires_grad=True)
    condition_grad = x_grad > 3.0
    
    result_grad = rm.where(condition_grad, x_grad, y_grad)
    print("带梯度的结果:")
    print(result_grad)
    
    # 计算梯度
    sum_result = rm.sum(result_grad)
    sum_result.backward()
    
    print("\n梯度计算结果:")
    print("x_grad 的梯度:")
    print(x_grad.grad)
    print("y_grad 的梯度:")
    print(y_grad.grad)

**sumall 函数的效率优势**

`sumall` 函数比使用张量加法运算更高效，特别是在梯度跟踪方面，因为：

1. **减小计算图**：使用 `sumall` 时，计算图减小到只有一层，无论有多少个张量相加。
2. **可扩展的效率**：使用张量加法运算符 (`+`) 时，计算图会随着张量的增加而线性变大，导致图复杂度增加。
3. **更快的梯度跟踪**：`sumall` 的简单图结构在反向传播时会产生更快的梯度计算，尤其是在对多个张量求和时。

**梯度跟踪效率示例**

.. code-block:: python

    import riemann as rm

    # 创建带梯度跟踪的张量
    x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = rm.tensor([4.0, 5.0, 6.0], requires_grad=True)
    z = rm.tensor([7.0, 8.0, 9.0], requires_grad=True)
    w = rm.tensor([10.0, 11.0, 12.0], requires_grad=True)

    # 使用 sumall（更高效）
    print("\n使用 sumall:")
    result_sumall = rm.sumall(x, y, z, w)
    result_sumall.backward()
    print("x.grad:", x.grad)
    print("y.grad:", y.grad)
    print("z.grad:", z.grad)
    print("w.grad:", w.grad)

    # 重置梯度
    x.grad = None
    y.grad = None
    z.grad = None
    w.grad = None

    # 使用加法运算符（效率较低）
    print("\n使用加法运算符:")
    result_addition = x + y + z + w
    result_addition.backward()
    print("x.grad:", x.grad)
    print("y.grad:", y.grad)
    print("z.grad:", z.grad)
    print("w.grad:", w.grad)

**其他统计函数示例**

.. code-block:: python

    import riemann as rm

    # 创建示例张量
    x = rm.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    y = rm.tensor([[2.0, 1.0, 4.0], [3.0, 6.0, 5.0], [8.0, 7.0, 10.0]])
    print("原始张量 x:", x)
    print("原始张量 y:", y)

    # 1. clamp 函数
    print("\n1. clamp 函数:")
    clamped = rm.clamp(x, min=3.0, max=7.0)
    print("限制在 3 和 7 之间:", clamped)

    # 2. argmax 函数
    print("\n2. argmax 函数:")
    print("最大值的索引:", rm.argmax(x))
    print("沿轴 0 的最大值索引:", rm.argmax(x, dim=0))

    # 3. maximum 函数
    print("\n3. maximum 函数:")
    max_result = rm.maximum(x, y)
    print("x 和 y 的元素级最大值:", max_result)

    # 4. sort 和 argsort 函数
    print("\n4. sort 和 argsort 函数:")
    sorted_x, indices = rm.sort(x, dim=1, return_indices=True)
    print("沿轴 1 排序:", sorted_x)
    print("排序索引:", indices)
    
    argsorted = rm.argsort(x, dim=1)
    print("沿轴 1 的排序索引:", argsorted)

**带梯度跟踪的统计函数示例**

.. code-block:: python

    import riemann as rm

    # 创建带梯度跟踪的张量
    x = rm.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    print("带梯度的原始张量:", x)

    # 1. sum 带梯度跟踪
    print("\n1. sum 带梯度跟踪:")
    sum_result = rm.sum(x)
    print("sum 结果:", sum_result)
    sum_result.backward()
    print("sum 的梯度:", x.grad)

    # 重置梯度
    x.grad = None

    # 2. mean 带梯度跟踪
    print("\n2. mean 带梯度跟踪:")
    mean_result = rm.mean(x)
    print("mean 结果:", mean_result)
    mean_result.backward()
    print("mean 的梯度:", x.grad)

    # 重置梯度
    x.grad = None

    # 3. max 带梯度跟踪
    print("\n3. max 带梯度跟踪:")
    max_result = rm.max(x)
    print("max 结果:", max_result)
    max_result.backward()
    print("max 的梯度:", x.grad)

张量比较运算符
~~~~~~~~~~~~~~~~

Riemann 支持多种张量比较运算符，用于比较张量元素。

.. list-table:: 张量比较运算符
  :widths: 15 35 25 25
  :header-rows: 1

  * - 运算符
    - 功能描述
    - 示例
    - 结果类型
  * - ``==``
    - 等于
    - ``x == y``
    - 布尔张量
  * - ``!=``
    - 不等于
    - ``x != y``
    - 布尔张量
  * - ``<``
    - 小于
    - ``x < y``
    - 布尔张量
  * - ``<=``
    - 小于等于
    - ``x <= y``
    - 布尔张量
  * - ``>``
    - 大于
    - ``x > y``
    - 布尔张量
  * - ``>=``
    - 大于等于
    - ``x >= y``
    - 布尔张量

**比较运算符示例**

.. code-block:: python

    import riemann as rm

    # 创建示例张量
    x = rm.tensor([1, 2, 3, 4])
    y = rm.tensor([2, 2, 2, 2])
    print("x:", x)
    print("y:", y)

    # 比较运算
    print("\n比较运算结果:")
    print("x == y:", x == y)
    print("x != y:", x != y)
    print("x < y:", x < y)
    print("x <= y:", x <= y)
    print("x > y:", x > y)
    print("x >= y:", x >= y)

张量逻辑运算符
~~~~~~~~~~~~~~~~

Riemann 支持多种张量逻辑运算符，用于对布尔张量进行逻辑操作。

.. list-table:: 张量逻辑运算符
  :widths: 15 35 25 25
  :header-rows: 1

  * - 运算符
    - 功能描述
    - 示例
    - 结果类型
  * - ``&``
    - 逻辑与
    - ``x & y``
    - 布尔张量
  * - ``|``
    - 逻辑或
    - ``x | y``
    - 布尔张量
  * - ``^``
    - 逻辑异或
    - ``x ^ y``
    - 布尔张量
  * - ``~``
    - 逻辑非
    - ``~x``
    - 布尔张量

**逻辑运算符示例**

.. code-block:: python

    import riemann as rm

    # 创建布尔张量
    x = rm.tensor([True, True, False, False])
    y = rm.tensor([True, False, True, False])
    print("x:", x)
    print("y:", y)

    # 逻辑运算
    print("\n逻辑运算结果:")
    print("x & y:", x & y)
    print("x | y:", x | y)
    print("x ^ y:", x ^ y)
    print("~x:", ~x)

张量位操作运算符
~~~~~~~~~~~~~~~~

Riemann 支持多种张量位操作运算符，用于对整数张量进行位操作。

.. list-table:: 张量位操作运算符
  :widths: 15 35 25 25
  :header-rows: 1

  * - 运算符
    - 功能描述
    - 示例
    - 结果类型
  * - ``&``
    - 按位与
    - ``x & y``
    - 整数张量
  * - ``|``
    - 按位或
    - ``x | y``
    - 整数张量
  * - ``^``
    - 按位异或
    - ``x ^ y``
    - 整数张量
  * - ``~``
    - 按位取反
    - ``~x``
    - 整数张量
  * - ``<<``
    - 左移
    - ``x << y``
    - 整数张量
  * - ``>>``
    - 右移
    - ``x >> y``
    - 整数张量

**位操作运算符示例**

.. code-block:: python

    import riemann as rm

    # 创建整数张量
    x = rm.tensor([1, 3, 5, 7], dtype=rm.int32)
    y = rm.tensor([1, 2, 3, 4], dtype=rm.int32)
    print("x:", x)
    print("y:", y)

    # 位操作
    print("\n位操作结果:")
    print("x & y:", x & y)
    print("x | y:", x | y)
    print("x ^ y:", x ^ y)
    print("~x:", ~x)
    print("x << 1:", x << 1)
    print("x >> 1:", x >> 1)

张量检查和比较函数
~~~~~~~~~~~~~~~~~~

Riemann 提供了多种张量检查和比较函数，用于检查张量的属性或比较多个张量。

.. list-table:: 张量检查和比较函数
  :widths: 15 45 40
  :header-rows: 1

  * - 函数名
    - 功能描述
    - 示例
  * - ``all``
    - 检查张量所有元素是否为真
    - ``rm.all(x)``
  * - ``any``
    - 检查张量是否有任何元素为真
    - ``rm.any(x)``
  * - ``allclose``
    - 检查两个张量是否在容差范围内相等
    - ``rm.allclose(x, y, rtol=1e-05, atol=1e-08)``
  * - ``equal``
    - 检查两个张量是否元素级相等
    - ``rm.equal(x, y)``
  * - ``not_equal``
    - 检查两个张量是否元素级不相等
    - ``rm.not_equal(x, y)``
  * - ``nonzero``
    - 返回非零元素的索引
    - ``rm.nonzero(x)``
  * - ``unique``
    - 返回张量中的唯一元素
    - ``rm.unique(x)``

**检查和比较函数示例**

.. code-block:: python

    import riemann as rm

    # 创建示例张量
    x = rm.tensor([True, True, True])
    y = rm.tensor([True, False, True])
    z = rm.tensor([1.0, 2.0, 3.0])
    w = rm.tensor([1.0, 2.0000001, 3.0])

    print("x:", x)
    print("y:", y)
    print("z:", z)
    print("w:", w)

    # 检查函数
    print("\n1. 检查函数:")
    print("all(x):", rm.all(x))
    print("any(y):", rm.any(y))

    # 比较函数
    print("\n2. 比较函数:")
    print("equal(z, w):", rm.equal(z, w))
    print("not_equal(z, w):", rm.not_equal(z, w))
    print("allclose(z, w):", rm.allclose(z, w))
    print("allclose(z, w, rtol=1e-03):", rm.allclose(z, w, rtol=1e-03))

形状和维度操作函数
~~~~~~~~~~~~~~~~~~

以下表格列出了 Riemann 支持的所有与形状、维度操作有关的函数：

.. list-table:: 形状和维度操作函数
  :widths: 15 55 30
  :header-rows: 1

  * - 函数名
    - 功能描述
    - 示例
  * - ``reshape``
    - 改变张量的形状，不改变数据，支持-1自动推理
    - ``x.reshape(3, 2)`` 或 ``x.reshape(-1, 2)``
  * - ``view``
    - reshape的别名，返回具有相同数据但不同形状的视图
    - ``x.view(3, 2)``
  * - ``flatten``
    - 将张量的指定维度范围展平为一维
    - ``x.flatten(start_dim=0, end_dim=-1)``
  * - ``squeeze``
    - 移除大小为1的维度
    - ``x.squeeze()`` 或 ``x.squeeze(0)``
  * - ``unsqueeze``
    - 在指定位置添加大小为1的维度
    - ``x.unsqueeze(0)``
  * - ``expand``
    - 扩展张量到指定形状，只能扩展大小为1的维度
    - ``x.expand(3, 4)`` 或 ``x.expand(-1, 4)``
  * - ``expand_as``
    - 扩展张量到与另一个张量相同的形状
    - ``x.expand_as(y)``
  * - ``repeat``
    - 沿指定维度重复张量的元素
    - ``x.repeat(2, 3)``
  * - ``transpose``
    - 交换张量的两个指定维度
    - ``x.transpose(0, 1)``
  * - ``permute``
    - 按照指定的维度顺序重新排列张量的维度
    - ``x.permute(2, 0, 1)``
  * - ``flip``
    - 沿指定维度翻转张量
    - ``x.flip([0, 1])``
  * - ``T``
    - 张量转置属性，对于高维张量反转整个维度顺序
    - ``x.T``
  * - ``mT``
    - 矩阵转置属性，只交换最后两个维度
    - ``x.mT``
  * - ``H``
    - 张量共轭转置属性
    - ``x.H``
  * - ``mH``
    - 矩阵共轭转置属性，最后两个维度的共轭转置
    - ``x.mH``
  * - ``cat`` / ``concatenate``
    - 沿指定维度连接张量序列
    - ``rm.cat([x, y], dim=0)``
  * - ``stack``
    - 沿新维度堆叠张量序列
    - ``rm.stack([x, y], dim=0)``
  * - ``vstack``
    - 垂直堆叠张量，一维张量作为行堆叠，多维张量沿第0轴连接
    - ``rm.vstack([x, y])``
  * - ``hstack``
    - 水平堆叠张量，一维张量水平连接，多维张量沿第1轴连接
    - ``rm.hstack([x, y])``
  * - ``split``
    - 沿指定维度将张量分割成多个块
    - ``rm.split(x, 2, dim=0)``

张量类型转换
------------

Riemann 提供了多种函数用于张量类型转换，包括数据类型转换和设备切换。

**数据类型转换函数**

.. list-table:: 数据类型转换函数
  :widths: 15 45 40
  :header-rows: 1

  * - 函数名
    - 功能描述
    - 示例
  * - ``type``
    - 将张量转换为指定数据类型
    - ``x.type(rm.float64)``
  * - ``type_as``
    - 将张量转换为与另一个张量相同的数据类型
    - ``x.type_as(y)``
  * - ``to``
    - 通用转换函数，可转换数据类型和设备
    - ``x.to(rm.float32)`` 或 ``x.to('cuda')``
  * - ``bool``
    - 将张量转换为布尔类型
    - ``x.bool()``
  * - ``float``
    - 将张量转换为 float32 类型
    - ``x.float()``
  * - ``double``
    - 将张量转换为 float64 类型
    - ``x.double()``
  * - ``real``
    - 返回复数张量的实部
    - ``x.real()``
  * - ``imag``
    - 返回复数张量的虚部
    - ``x.imag()``
  * - ``conj``
    - 返回复数张量的共轭
    - ``x.conj()``

**设备切换函数**

.. list-table:: 设备切换函数
  :widths: 15 45 40
  :header-rows: 1

  * - 函数名
    - 功能描述
    - 示例
  * - ``cuda``
    - 将张量移至 CUDA 设备
    - ``x.cuda()`` 或 ``x.cuda(0)``
  * - ``cpu``
    - 将张量移至 CPU 设备
    - ``x.cpu()``
  * - ``to``
    - 通用设备切换函数
    - ``x.to('cuda')`` 或 ``x.to('cpu')``

**to() 函数详细参数说明**

.. list-table:: to() 函数参数
  :widths: 15 30 40 15
  :header-rows: 1
  :align: center

  * - 参数名
    - 类型
    - 描述
    - 默认值
  * - ``other``
    - TN
    - 另一个张量，使用其device和dtype作为目标进行迁移
    - None
  * - ``device``
    - str 或 Device
    - 目标设备，可以是字符串（如'cpu'、'cuda'）或Device对象
    - None
  * - ``dtype``
    - dtype
    - 目标数据类型，可以是Python类型、NumPy dtype、字符串或Riemann dtype
    - None
  * - ``non_blocking``
    - bool
    - 如果为True且数据在固定内存中，则复制到GPU可以与主机计算异步进行。仅适用于CPU -> GPU的传输
    - False
  * - ``copy``
    - bool
    - 如果为True，则总是返回副本，即使设备和数据类型相同
    - False

**to() 函数使用示例**

.. code-block:: python

    import riemann as rm

    # 转换数据类型
    x = rm.tensor([1.0, 2.0, 3.0], dtype=rm.float32)
    y = x.to(rm.float64)
    print(f"转换后数据类型: {y.dtype}")

    # 转换设备
    x = rm.tensor([1.0, 2.0, 3.0], device='cpu')
    y = x.to('cuda')
    print(f"转换后设备: {y.device}")

    # 同时转换数据类型和设备
    x = rm.tensor([1.0, 2.0, 3.0], dtype=rm.float32, device='cpu')
    y = x.to(rm.float64, device='cuda')
    print(f"转换后数据类型: {y.dtype}, 设备: {y.device}")

    # 使用关键字参数
    x = rm.tensor([1.0, 2.0, 3.0])
    y = x.to(dtype=rm.float64, device='cuda')

    # 从另一个张量复制dtype和device
    x = rm.tensor([1.0, 2.0, 3.0], dtype=rm.float64, device='cuda')
    y = rm.tensor([4.0, 5.0, 6.0])
    z = y.to(x)
    print(f"从x复制后: dtype={z.dtype}, device={z.device}")

    # 强制复制
    y = x.to(copy=True)

**non_blocking 参数使用示例**

.. code-block:: python

    import riemann as rm

    # 创建CPU上的张量
    x = rm.tensor([1.0, 2.0, 3.0], device='cpu')

    # 异步传输到GPU
    # 注意：异步传输需要数据在固定内存中
    # 实际使用中，建议在传输后同步设备以确保数据已传输完成
    y = x.to('cuda', non_blocking=True)

    # 执行一些CPU计算
    # 这些计算可以与数据传输并行进行
    cpu_result = x * 2

    # 同步设备，确保数据传输完成
    # 在访问GPU上的张量之前必须同步
    rm.cuda.synchronize()

    # 现在可以安全地使用GPU上的张量
    gpu_result = y * 2

**类型转换示例**

.. code-block:: python

    import riemann as rm

    # 创建一个整型张量
    x = rm.tensor([1, 2, 3], dtype=rm.int32)
    print("原始张量:", x)
    print("原始数据类型:", x.dtype)
    print("原始设备:", x.device)

    # 1. 数据类型转换
    print("\n1. 数据类型转换:")
    x_float = x.float()
    print("转换为 float32:", x_float)
    print("数据类型:", x_float.dtype)

    x_double = x.double()
    print("\n转换为 float64:", x_double)
    print("数据类型:", x_double.dtype)

    x_bool = x.bool()
    print("\n转换为 bool:", x_bool)
    print("数据类型:", x_bool.dtype)

    # 2. 使用 to 函数转换
    print("\n2. 使用 to 函数转换:")
    x_to_float = x.to(rm.float32)
    print("使用 to 转换为 float32:", x_to_float.dtype)

    # 3. 复数相关转换
    print("\n3. 复数相关转换:")
    z = rm.tensor([1+2j, 3+4j], dtype=rm.complex64)
    print("复数张量:", z)
    print("实部:", z.real())
    print("虚部:", z.imag())
    print("共轭:", z.conj())

    # 4. 设备切换（如果有 CUDA 可用）
    print("\n4. 设备切换:")
    if rm.cuda.is_available():
        x_cuda = x.cuda()
        print("移至 CUDA 设备:", x_cuda.device)
        
        x_back_to_cpu = x_cuda.cpu()
        print("移回 CPU 设备:", x_back_to_cpu.device)
    else:
        print("CUDA 不可用，跳过设备切换示例")

**类型转换的注意事项**

1. **数据类型转换**：

   - 从高精度类型转换为低精度类型可能会导致精度损失
   - 从整数类型转换为浮点数类型是安全的
   - 从浮点数类型转换为整数类型会截断小数部分

2. **设备切换**：

   - 设备切换会创建新的张量副本，消耗内存和时间
   - 确保在进行设备切换时，目标设备可用
   - 不同设备上的张量不能直接进行运算，需要先统一设备

3. **复数转换**：

   - ``real()`` 和 ``imag()`` 函数返回复数张量的实部和虚部，结果为浮点类型
   - ``conj()`` 函数返回复数张量的共轭，结果仍为复数类型

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

梯度上下文管理
~~~~~~~~~~~~~~~~

Riemann 提供了多种梯度上下文管理工具，用于在特定代码块中控制梯度跟踪的行为。这些工具可以通过 `with` 语句或装饰器的方式使用。

**使用 with 语句控制梯度上下文**

.. code-block:: python

    import riemann as rm

    # 创建需要梯度跟踪的张量
    x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)

    # 1. 使用 no_grad() 禁用梯度跟踪
    print("\n1. 使用 no_grad() 禁用梯度跟踪:")
    with rm.no_grad():
        y = x * 2
        print("y.requires_grad:", y.requires_grad)  # False

    # 2. 使用 enable_grad() 启用梯度跟踪
    print("\n2. 使用 enable_grad() 启用梯度跟踪:")
    with rm.no_grad():
        # 在此上下文中，梯度跟踪默认是禁用的
        z = x + 1
        print("z.requires_grad:", z.requires_grad)  # False
        
        # 但可以在内部启用梯度跟踪
        with rm.enable_grad():
            w = x * 3
            print("w.requires_grad:", w.requires_grad)  # True

    # 3. 使用 set_grad_enabled() 手动设置梯度跟踪状态
    print("\n3. 使用 set_grad_enabled() 手动设置梯度跟踪状态:")
    with rm.set_grad_enabled(True):
        a = x * 4
        print("a.requires_grad:", a.requires_grad)  # True
    
    with rm.set_grad_enabled(False):
        b = x * 5
        print("b.requires_grad:", b.requires_grad)  # False

**使用装饰器控制梯度上下文**

除了 `with` 语句，Riemann 还提供了装饰器形式的梯度上下文管理工具，用于控制整个函数的梯度跟踪行为。

.. code-block:: python

    import riemann as rm

    # 创建需要梯度跟踪的张量
    x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)

    # 使用 @no_grad 装饰器禁用函数内的梯度跟踪
    @rm.no_grad
    def inference_fn(tensor):
        """推理函数，不需要梯度跟踪"""
        result = tensor * 2 + 1
        print("inference_fn: result.requires_grad =", result.requires_grad)
        return result

    # 使用 @enable_grad 装饰器启用函数内的梯度跟踪
    @rm.enable_grad
    def training_fn(tensor):
        """训练函数，需要梯度跟踪"""
        result = tensor * 3 + 2
        print("training_fn: result.requires_grad =", result.requires_grad)
        return result

    # 测试装饰器效果
    print("\n测试 @no_grad 装饰器:")
    output1 = inference_fn(x)
    
    print("\n测试 @enable_grad 装饰器:")
    output2 = training_fn(x)

**梯度上下文管理的应用场景**

1. **推理阶段**：在模型推理时禁用梯度跟踪，提高性能并节省内存
2. **部分计算**：在复杂计算中，只对需要的部分启用梯度跟踪
3. **嵌套上下文**：在不同层级的代码中灵活切换梯度跟踪状态
4. **函数级别控制**：通过装饰器为整个函数设置统一的梯度跟踪策略

索引操作
--------

Riemann 支持多种张量索引操作方式，用于获取数组元素或片段。以下是常见的索引方式：

**1. 整数索引**

整数索引用于获取张量中特定位置的单个元素。对于多维张量，可以使用逗号分隔的多个整数索引。

.. code-block:: python

    import riemann as rm

    # 创建张量
    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("x:", x)

    # 整数索引
    print("x[0, 0]:", x[0, 0])  # 获取第一行第一列元素
    print("x[1, 2]:", x[1, 2])  # 获取第二行第三列元素

**2. 负整数索引**

负整数索引从张量末尾开始计数，-1 表示最后一个元素，-2 表示倒数第二个元素，以此类推。

.. code-block:: python

    import riemann as rm

    # 创建张量
    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("x:", x)

    # 负整数索引
    print("x[-1, -1]:", x[-1, -1])  # 获取最后一行最后一列元素
    print("x[-2, -3]:", x[-2, -3])  # 获取倒数第二行倒数第三列元素

**3. 切片索引**

切片索引用于获取张量的连续片段，使用冒号（:）表示范围。格式为 `start:end:step`，其中 start 是起始索引，end 是结束索引（不包含），step 是步长。

.. code-block:: python

    import riemann as rm

    # 创建张量
    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("x:", x)

    # 切片索引
    print("x[:, 0]:", x[:, 0])  # 获取所有行的第一列
    print("x[0, :]:", x[0, :])  # 获取第一行的所有列
    print("x[1:, 1:]:", x[1:, 1:])  # 获取从第二行开始，第二列开始的子张量
    print("x[::2, ::2]:", x[::2, ::2])  # 获取隔行隔列的元素

**4. 整数数组索引**

整数数组索引用于根据整数数组指定的位置获取元素，返回的张量形状与索引数组相同。

.. code-block:: python

    import riemann as rm

    # 创建张量
    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("x:", x)

    # 整数数组索引
    indices = rm.tensor([0, 1, 2])
    print("x[indices, indices]:", x[indices, indices])  # 获取对角线元素

**5. 布尔索引**

布尔索引用于根据布尔数组指定的条件获取元素，返回的是满足条件的元素组成的一维张量。

.. code-block:: python

    import riemann as rm

    # 创建张量
    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("x:", x)

    # 布尔索引
    mask = x > 5
    print("mask:", mask)
    print("x[mask]:", x[mask])  # 获取大于 5 的元素

**6. 混合索引**

混合索引是指在同一个索引表达式中使用多种索引方式，例如同时使用整数索引和切片索引。

.. code-block:: python

    import riemann as rm

    # 创建张量
    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("x:", x)

    # 混合索引
    print("x[0, 1:]:", x[0, 1:])  # 获取第一行的第二列及以后的元素
    print("x[1:, 0]:", x[1:, 0])  # 获取第二行及以后的第一列元素

**7. 索引相关函数**

Riemann 提供了多个与索引相关的函数，用于按索引收集或散射数据：

.. list-table:: 索引相关函数
  :widths: 15 45 40
  :header-rows: 1

  * - 函数名
    - 功能描述
    - 示例
  * - ``gather``
    - 按索引收集数据
    - ``input.gather(dim, index)``
  * - ``scatter``
    - 按索引散射数据（非原地）
    - ``input.scatter(dim, index, src)``
  * - ``scatter_``
    - 按索引散射数据（原地）
    - ``input.scatter_(dim, index, src)``
  * - ``scatter_add``
    - 按索引散射并累加数据（非原地）
    - ``input.scatter_add(dim, index, src)``
  * - ``scatter_add_``
    - 按索引散射并累加数据（原地）
    - ``input.scatter_add_(dim, index, src)``
  * - ``setat``
    - 在指定索引处设置值（非原地）
    - ``input.setat(indices, value)``
  * - ``setat_``
    - 在指定索引处设置值（原地）
    - ``input.setat_(indices, value)``
  * - ``addat``
    - 在指定索引处添加值（非原地）
    - ``input.addat(indices, value)``
  * - ``addat_``
    - 在指定索引处添加值（原地）
    - ``input.addat_(indices, value)``
  * - ``subat``
    - 在指定索引处减去值（非原地）
    - ``input.subat(indices, value)``
  * - ``subat_``
    - 在指定索引处减去值（原地）
    - ``input.subat_(indices, value)``
  * - ``mulat``
    - 在指定索引处乘以值（非原地）
    - ``input.mulat(indices, value)``
  * - ``mulat_``
    - 在指定索引处乘以值（原地）
    - ``input.mulat_(indices, value)``
  * - ``divat``
    - 在指定索引处除以值（非原地）
    - ``input.divat(indices, value)``
  * - ``divat_``
    - 在指定索引处除以值（原地）
    - ``input.divat_(indices, value)``
  * - ``powat``
    - 在指定索引处进行幂运算（非原地）
    - ``input.powat(indices, value)``
  * - ``powat_``
    - 在指定索引处进行幂运算（原地）
    - ``input.powat_(indices, value)``

**gather 函数示例**

``gather`` 函数用于从输入张量的指定维度中按索引收集数据。

.. code-block:: python

    import riemann as rm

    # 创建输入张量
    input = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("input:", input)

    # 定义索引
    index = rm.tensor([[0, 1], [1, 2]])
    print("index:", index)

    # 按维度 0 收集数据
    output = input.gather(0, index)
    print("gather along dim 0:", output)

    # 按维度 1 收集数据
    output = input.gather(1, index)
    print("gather along dim 1:", output)

**scatter 函数示例**

``scatter`` 函数用于将源张量的数据按索引散射到目标张量的指定维度。

.. code-block:: python

    import riemann as rm

    # 创建目标张量
    input = rm.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    print("input:", input)

    # 定义索引和源张量
    index = rm.tensor([[0, 1], [1, 2]])
    src = rm.tensor([[10, 20], [30, 40]])
    print("index:", index)
    print("src:", src)

    # 按维度 0 散射数据（非原地）
    output = input.scatter(0, index, src)
    print("scatter along dim 0:", output)

    # 按维度 1 散射数据（非原地）
    output = input.scatter(1, index, src)
    print("scatter along dim 1:", output)

**scatter_ 函数示例**

``scatter_`` 是 ``scatter`` 的原地版本，直接修改输入张量。

.. code-block:: python

    import riemann as rm

    # 创建目标张量
    input = rm.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    print("input:", input)

    # 定义索引和源张量
    index = rm.tensor([[0, 1], [1, 2]])
    src = rm.tensor([[10, 20], [30, 40]])
    print("index:", index)
    print("src:", src)

    # 按维度 1 散射数据（原地）
    input.scatter_(1, index, src)
    print("after scatter_ along dim 1:", input)

**scatter_add 函数示例**

``scatter_add`` 函数用于将源张量的数据按索引散射并累加到目标张量的指定维度。

.. code-block:: python

    import riemann as rm

    # 创建目标张量
    input = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("input:", input)

    # 定义索引和源张量
    index = rm.tensor([[0, 1], [1, 2]])
    src = rm.tensor([[10, 20], [30, 40]])
    print("index:", index)
    print("src:", src)

    # 按维度 1 散射并累加数据（非原地）
    output = input.scatter_add(1, index, src)
    print("scatter_add along dim 1:", output)

**scatter_add_ 函数示例**

``scatter_add_`` 是 ``scatter_add`` 的原地版本，直接修改输入张量。

.. code-block:: python

    import riemann as rm

    # 创建目标张量
    input = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("input:", input)

    # 定义索引和源张量
    index = rm.tensor([[0, 1], [1, 2]])
    src = rm.tensor([[10, 20], [30, 40]])
    print("index:", index)
    print("src:", src)

    # 按维度 1 散射并累加数据（原地）
    input.scatter_add_(1, index, src)
    print("after scatter_add_ along dim 1:", input)

**setat 和 setat_ 函数示例**

``setat`` 函数用于在指定索引处设置值（非原地），而 ``setat_`` 是其原地版本。

.. code-block:: python

    import riemann as rm

    # 创建输入张量
    input = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("input:", input)

    # 1. 使用 setat（非原地）
    print("\n1. 使用 setat（非原地）:")
    # 在指定索引处设置值
    indices = (0, 1)  # 第 0 行，第 1 列
    value = 99
    output = input.setat(indices, value)
    print("setat 结果:", output)
    print("原始输入保持不变:", input)

    # 2. 使用 setat_（原地）
    print("\n2. 使用 setat_（原地）:")
    # 在指定索引处设置值
    indices = (1, 2)  # 第 1 行，第 2 列
    value = 88
    input.setat_(indices, value)
    print("setat_ 后:", input)

    # 3. 使用 setat 设置多个索引
    print("\n3. 使用 setat 设置多个索引:")
    input = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    indices = [[0, 0], [2, 2]]  # (0,0) 和 (2,2)
    value = 100
    output = input.setat(indices, value)
    print("setat 设置多个索引:", output)

**索引操作注意事项**

1. **索引越界**：使用超出张量范围的索引会导致错误。
2. **内存布局**：不同的索引方式可能会影响返回张量的内存布局，某些索引操作可能会返回原始张量的视图，而不是副本。
3. **梯度跟踪**：对于需要梯度跟踪的张量，某些索引操作可能会影响梯度计算，特别是原地操作。
4. **性能考虑**：对于大型张量，整数数组索引和布尔索引可能会比切片索引慢，因为它们会创建新的张量副本。
5. **gather/scatter 函数参数**：
   - ``dim``：指定操作的维度
   - ``index``：指定索引位置
   - ``src``：指定源数据（仅用于 scatter 相关函数）
   - 对于原地操作（带下划线的函数），需要确保输入张量不是叶子节点，否则会导致错误。

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

原地操作函数和运算符
~~~~~~~~~~~~~~~~~~~~

Riemann 支持以下原地操作函数和运算符：

.. list-table:: 原地操作函数和运算符
  :widths: 15 35 30 20
  :header-rows: 1

  * - 函数名
    - 功能描述
    - 等效运算符
    - 示例
  * - ``add_``
    - 原地加法
    - ``+=``
    - ``x.add_(y)`` 或 ``x += y``
  * - ``sub_``
    - 原地减法
    - ``-=``
    - ``x.sub_(y)`` 或 ``x -= y``
  * - ``mul_``
    - 原地乘法
    - ``*=``
    - ``x.mul_(y)`` 或 ``x *= y``
  * - ``div_``
    - 原地除法
    - ``/=``
    - ``x.div_(y)`` 或 ``x /= y``
  * - ``pow_``
    - 原地幂运算
    - ``**=``
    - ``x.pow_(y)`` 或 ``x **= y``
  * - ``zero_``
    - 原地将所有元素置为0
    - 无
    - ``x.zero_()``
  * - ``fill_``
    - 原地将所有元素填充为指定值
    - 无
    - ``x.fill_(5)``
  * - ``copy_``
    - 原地从另一个张量复制数据
    - 无
    - ``x.copy_(y)``
  * - ``detach_``
    - 原地分离梯度，使张量不再跟踪梯度
    - 无
    - ``x.detach_()``
  * - ``masked_fill_``
    - 原地根据掩码填充值
    - 无
    - ``x.masked_fill_(mask, value)``
  * - ``fill_diagonal_``
    - 原地填充对角线元素
    - 无
    - ``x.fill_diagonal_(value)``
  * - ``setat_``
    - 原地设置指定位置的值
    - ``x[index] = val``
    - ``x.setat_(index, val)``
  * - ``addat_``
    - 原地在指定位置执行加法
    - ``x[index] += val``
    - ``x.addat_(index, val)``
  * - ``subat_``
    - 原地在指定位置执行减法
    - ``x[index] -= val``
    - ``x.subat_(index, val)``
  * - ``mulat_``
    - 原地在指定位置执行乘法
    - ``x[index] *= val``
    - ``x.mulat_(index, val)``
  * - ``divat_``
    - 原地在指定位置执行除法
    - ``x[index] /= val``
    - ``x.divat_(index, val)``
  * - ``powat_``
    - 原地在指定位置执行幂运算
    - ``x[index] **= val``
    - ``x.powat_(index, val)``
  * - ``scatter_``
    - 原地根据索引分散值
    - 无
    - ``x.scatter_(dim, index, src)``
  * - ``scatter_add_``
    - 原地根据索引分散并累加值
    - 无
    - ``x.scatter_add_(dim, index, src)``

原地操作的注意事项
~~~~~~~~~~~~~~~~~~

使用原地操作时需要注意以下几点：

1. **带梯度跟踪属性的叶子节点限制**

   - 对于 ``requires_grad=True`` 的叶子节点张量，不允许执行原地操作
   - 这是因为原地操作会修改张量的值，可能会破坏梯度计算的正确性

2. **右值的梯度跟踪**

   - 原地操作的右值（如 ``x += y`` 中的 ``y``）的梯度可以正常跟踪
   - 这意味着即使使用原地操作，右值张量的梯度计算不受影响

3. **原地操作对象的梯度跟踪**

   - 对于非叶子节点的张量，原地操作的梯度跟踪结果比较复杂
   - 特别是按索引对数组赋值（如 ``x[index] = val``）时，梯度计算可能会出现意外行为
   - 建议在需要梯度跟踪的场景中谨慎使用原地操作

4. **推荐使用场景**

   - 对新建的无梯度跟踪属性的张量（``requires_grad=False``），可以使用原地操作
   - 对 ``clone()`` 或 ``copy()`` 后的对象，这些对象不是可梯度跟踪的叶子节点，可以使用原地操作
   - 在不需要梯度计算的推理阶段，使用原地操作可以节省内存

5. **内存优化**

   - 原地操作不会创建新的张量对象，因此可以节省内存
   - 在处理大型张量时，适当使用原地操作可以显著减少内存使用

6. **链式操作**

   - 原地操作返回``self``，因此可以进行链式调用
   - 例如：``x.add_(y).mul_(z)`` 是可行的， ``(x + y) * z`` 是非原地操作的链式调用

原地操作的梯度跟踪示例
~~~~~~~~~~~~~~~~~~~~~~~~

以下是按索引对数组原地赋值的梯度跟踪示例：

.. code-block:: python

    import riemann as rm

    # 创建需要梯度跟踪的张量
    x0 = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = rm.tensor([10.0, 20.0, 30.0], requires_grad=True)

    # 打印原始值
    print("原始值:")
    print("x0:", x0)
    print("y:", y)

    # 对x进行clone，使其不再是叶子节点，可以执行原地操作
    x = x0.clone()
    print("\nx.clone()后，x不再是叶子节点")
    print("x.is_leaf:", x.is_leaf)

    # 按索引进行原地赋值
    print("\n执行原地赋值 x[1] = y[0]")
    x[1] = y[0]

    # 打印赋值后的值
    print("\n赋值后:")
    print("x0:", x0)
    print("x:", x)
    print("y:", y)

    # 计算损失函数
    loss = x.sum()
    print("\n损失值:", loss)

    # 反向传播计算梯度
    loss.backward()

    # 打印梯度
    print("\n梯度跟踪结果:")
    print("x0.grad:", x0.grad)  # 左值方向的梯度
    print("y.grad:", y.grad)  # 右值方向的梯度

**输出结果分析**:

- 原地赋值后，`x` 的值变为 `[1.0, 10.0, 3.0]`，而 `y` 的值保持不变
- 梯度计算结果显示：
  - `x0.grad` 为 `[1.0, 0.0, 1.0]`，说明除了原地赋值的位置外，其他位置的梯度正常跟踪
  - `y.grad` 为 `[1.0, 0.0, 0.0]`，说明右值方向的梯度正常跟踪

**结论**:

- 右值方向的梯度跟踪不受原地操作影响，正常工作
- 左值方向的梯度跟踪在原地赋值位置可能会出现异常行为
- 对于带梯度跟踪属性的叶子节点，必须先clone()后才能执行原地操作
- 因此，在需要精确梯度计算的场景中，应谨慎使用原地操作

对角化操作
--------------

Riemann 提供了多种对角化操作函数，用于处理张量的对角线元素、三角部分等。以下是常用的对角化操作函数：

**diagonal 函数**

从输入张量中提取对角线元素。

.. code-block:: python

    import riemann as rm
    
    # 创建示例张量
    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("原始张量:")
    print(x)
    
    # 提取主对角线
    print("\n主对角线:")
    print(rm.diagonal(x))  # tensor([1, 5, 9])
    
    # 提取偏移对角线
    print("\n偏移对角线 (offset=1):")
    print(rm.diagonal(x, offset=1))  # tensor([2, 6])
    
    # 提取负偏移对角线
    print("\n负偏移对角线 (offset=-1):")
    print(rm.diagonal(x, offset=-1))  # tensor([4, 8])

**diag 函数**

提取张量的对角线元素或从1D张量创建对角矩阵。

.. code-block:: python

    import riemann as rm
    
    # 从1D张量创建对角矩阵
    v = rm.tensor([1, 2, 3])
    print("\n从1D张量创建对角矩阵:")
    print(rm.diag(v))
    # 输出:
    # tensor([[1, 0, 0],
    #         [0, 2, 0],
    #         [0, 0, 3]])
    
    # 提取2D张量的对角线元素
    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("\n提取2D张量的对角线元素:")
    print(rm.diag(x))  # tensor([1, 5, 9])

**batch_diag 函数**

从批量1D张量生成批量对角矩阵。

.. code-block:: python

    import riemann as rm
    
    # 创建批量1D张量
    batch_v = rm.tensor([[1, 2], [3, 4]])
    print("\n批量1D张量:")
    print(batch_v)
    
    # 生成批量对角矩阵
    print("\n批量对角矩阵:")
    print(rm.batch_diag(batch_v))
    # 输出:
    # tensor([[[1, 0],
    #          [0, 2]],
    #         
    #         [[3, 0],
    #          [0, 4]]])

**fill_diagonal 函数**

用指定值填充张量指定维度之间的对角线元素，返回新张量。

.. code-block:: python

    import riemann as rm
    
    # 创建示例张量
    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("\n原始张量:")
    print(x)
    
    # 用0填充主对角线
    print("\n用0填充主对角线:")
    print(rm.fill_diagonal(x, 0))
    # 输出:
    # tensor([[0, 2, 3],
    #         [4, 0, 6],
    #         [7, 8, 0]])
    
    # 用5填充偏移对角线
    print("\n用5填充偏移对角线 (offset=1):")
    print(rm.fill_diagonal(x, 5, offset=1))

**fill_diagonal_ 函数**

用指定值原地填充张量指定维度之间的对角线元素，返回原张量。

.. code-block:: python

    import riemann as rm
    
    # 创建示例张量
    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("\n原始张量:")
    print(x)
    
    # 原地用0填充主对角线
    print("\n原地用0填充主对角线:")
    result = rm.fill_diagonal_(x, 0)
    print(result)
    print("原张量是否被修改:")
    print(x)

**tril 函数**

提取张量的下三角部分（包括对角线）。

.. code-block:: python

    import riemann as rm
    
    # 创建示例张量
    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("\n原始张量:")
    print(x)
    
    # 提取下三角部分
    print("\n下三角部分:")
    print(rm.tril(x))
    # 输出:
    # tensor([[1, 0, 0],
    #         [4, 5, 0],
    #         [7, 8, 9]])
    
    # 提取偏移下三角部分
    print("\n偏移下三角部分 (diagonal=-1):")
    print(rm.tril(x, diagonal=-1))

**triu 函数**

提取张量的上三角部分（包括对角线）。

.. code-block:: python

    import riemann as rm
    
    # 创建示例张量
    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("\n原始张量:")
    print(x)
    
    # 提取上三角部分
    print("\n上三角部分:")
    print(rm.triu(x))
    # 输出:
    # tensor([[1, 2, 3],
    #         [0, 5, 6],
    #         [0, 0, 9]])
    
    # 提取偏移上三角部分
    print("\n偏移上三角部分 (diagonal=1):")
    print(rm.triu(x, diagonal=1))

**函数参数说明**

.. list-table:: 对角化操作函数参数
    :widths: 15 35 25 25
    :header-rows: 1

    * - 函数名
      - 主要参数
      - 默认值
      - 说明
    * - ``diagonal``
      - input, offset, dim1, dim2
      - offset=0, dim1=0, dim2=1
      - 提取指定维度间的对角线元素
    * - ``diag``
      - input, offset
      - offset=0
      - 提取对角线元素或创建对角矩阵
    * - ``batch_diag``
      - v
      - 无
      - 从批量1D张量生成批量对角矩阵
    * - ``fill_diagonal``
      - input, value, offset, dim1, dim2
      - offset=0, dim1=-2, dim2=-1
      - 填充对角线元素，返回新张量
    * - ``fill_diagonal_``
      - input, value, offset, dim1, dim2
      - offset=0, dim1=-2, dim2=-1
      - 原地填充对角线元素，返回原张量
    * - ``tril``
      - input_tensor, diagonal
      - diagonal=0
      - 提取下三角部分
    * - ``triu``
      - input_tensor, diagonal
      - diagonal=0
      - 提取上三角部分

**注意事项**

1. ``diagonal`` 函数：
   - 输入张量必须至少是2维的
   - dim1 和 dim2 不能相同
   - 支持负索引（-1表示最后一个维度）

2. ``diag`` 函数：
   - 当输入是1D张量时，返回对角矩阵
   - 当输入是2D张量时，返回对角线元素
   - 不支持3D及以上维度的输入

3. ``batch_diag`` 函数：
   - 输入张量的最后一维是对角线元素的长度
   - 输出张量的形状为 ``(*, n, n)``，其中 n 是输入张量的最后一维大小

4. ``fill_diagonal`` 和 ``fill_diagonal_`` 函数：
   - input 张量必须至少是2维的
   - dim1 和 dim2 不能相同
   - 支持负索引（默认填充最后两个维度的对角线）
   - ``fill_diagonal_`` 是原地操作，会修改原张量

5. ``tril`` 和 ``triu`` 函数：
   - diagonal 参数控制对角线的偏移量
   - diagonal=0 表示主对角线
   - diagonal>0 表示主对角线以上
   - diagonal<0 表示主对角线以下

保存和加载张量
--------------

Riemann 提供了与 PyTorch 兼容的序列化功能，支持保存和加载张量、参数、模块状态以及训练检查点。这些功能使用 ZIP 格式进行序列化，确保跨平台兼容性和高效的存储。

基本用法
~~~~~~~~

保存和加载单个张量：

.. code-block:: python

    import riemann as rm
    
    # 创建张量
    x = rm.tensor([1, 2, 3])
    
    # 保存到文件
    rm.save(x, 'tensor.pt')
    
    # 从文件加载
    y = rm.load('tensor.pt')
    print(y)  # tensor([1, 2, 3])

保存多维张量
~~~~~~~~~~~~

可以保存任意形状和维度的张量：

.. code-block:: python

    # 创建多维张量
    matrix = rm.randn(3, 4)
    tensor_3d = rm.randn(2, 3, 4)
    
    # 保存多维张量
    rm.save(matrix, 'matrix.pt')
    rm.save(tensor_3d, 'tensor_3d.pt')
    
    # 加载并验证
    loaded_matrix = rm.load('matrix.pt')
    loaded_tensor_3d = rm.load('tensor_3d.pt')
    
    print(f"矩阵形状: {loaded_matrix.shape}")  # (3, 4)
    print(f"3D张量形状: {loaded_tensor_3d.shape}")  # (2, 3, 4)

保存模型状态字典
~~~~~~~~~~~~~~~~

在训练深度学习模型时，通常需要保存模型的参数状态：

.. code-block:: python

    # 创建一个简单的神经网络
    model = rm.nn.Sequential(
        rm.nn.Linear(10, 64),
        rm.nn.ReLU(),
        rm.nn.Linear(64, 10)
    )
    
    # 保存模型状态字典
    rm.save(model.state_dict(), 'model_weights.pt')
    
    # 创建新模型并加载权重
    new_model = rm.nn.Sequential(
        rm.nn.Linear(10, 64),
        rm.nn.ReLU(),
        rm.nn.Linear(64, 10)
    )
    new_model.load_state_dict(rm.load('model_weights.pt'))

保存训练检查点
~~~~~~~~~~~~~~

训练过程中，可以保存包含模型状态、优化器状态和训练进度的完整检查点：

.. code-block:: python

    # 假设正在进行模型训练
    model = rm.nn.Linear(10, 5)
    optimizer = rm.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练若干轮次
    for epoch in range(10):
        # ... 训练代码 ...
        pass
    
    # 保存完整的训练检查点
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': 0.5,  # 当前损失值
    }
    rm.save(checkpoint, 'checkpoint.pt')
    
    # 从检查点恢复训练
    checkpoint = rm.load('checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']
    
    print(f"从第 {start_epoch} 轮继续训练，上次损失: {loss}")

设备映射加载
~~~~~~~~~~~~

当在不同设备（CPU/GPU）之间加载模型时，可以使用 ``map_location`` 参数指定加载位置：

.. code-block:: python

    # 在GPU上保存的张量，在CPU上加载
    # 假设在GPU上训练并保存
    # rm.save(gpu_tensor, 'gpu_tensor.pt')
    
    # 在CPU上加载
    cpu_tensor = rm.load('gpu_tensor.pt', map_location='cpu')
    
    # 使用字典进行设备映射
    map_location = {'cuda:0': 'cpu', 'cuda:1': 'cpu'}
    cpu_tensor = rm.load('model.pt', map_location=map_location)

保存多个张量
~~~~~~~~~~~~

可以将多个张量保存在同一个文件中：

.. code-block:: python

    # 创建多个张量
    tensor_a = rm.randn(3, 3)
    tensor_b = rm.randn(4, 4)
    tensor_c = rm.tensor([1, 2, 3, 4, 5])
    
    # 保存为字典
    tensor_dict = {
        'weights': tensor_a,
        'biases': tensor_b,
        'labels': tensor_c
    }
    rm.save(tensor_dict, 'tensors.pt')
    
    # 加载并访问各个张量
    loaded_dict = rm.load('tensors.pt')
    weights = loaded_dict['weights']
    biases = loaded_dict['biases']
    labels = loaded_dict['labels']

注意事项
~~~~~~~~

1. **文件格式**：Riemann 使用 ZIP 格式进行序列化，文件扩展名通常为 ``.pt`` 或 ``.pth``

2. **兼容性**：序列化格式与 PyTorch 兼容，可以加载 PyTorch 保存的张量（部分限制）

3. **设备信息**：保存的张量会保留设备信息（CPU/GPU），加载时可以通过 ``map_location`` 重新映射

4. **梯度信息**：保存张量时会保留梯度计算图信息（requires_grad 属性）

5. **大文件处理**：对于大型模型，建议使用检查点机制分块保存，避免内存不足