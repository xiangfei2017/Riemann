API 参考
========

本节提供了 Riemann 库中所有函数、类和模块的全面参考。

张量操作
--------

张量创建函数
~~~~~~~~~~~~

.. function:: riemann.tensor(data, dtype=None, requires_grad=False)

   从数据创建张量。

   :param data: 初始化张量的数据
   :type data: array_like
   :param dtype: 张量的期望数据类型
   :type dtype: numpy.dtype, optional
   :param requires_grad: 是否跟踪此张量上的操作
   :type requires_grad: bool, optional
   :return: 包含给定数据的张量
   :rtype: riemann.TN

.. function:: riemann.zeros(*shape, dtype=None, requires_grad=False)

   创建一个填充零的张量。

   :param shape: 张量的形状
   :type shape: int 或整数元组
   :param dtype: 张量的期望数据类型
   :type dtype: numpy.dtype, optional
   :param requires_grad: 是否跟踪此张量上的操作
   :type requires_grad: bool, optional
   :return: 填充零的张量
   :rtype: riemann.TN

.. function:: riemann.ones(*shape, dtype=None, requires_grad=False)

   创建一个填充一的张量。

   :param shape: 张量的形状
   :type shape: int 或整数元组
   :param dtype: 张量的期望数据类型
   :type dtype: numpy.dtype, optional
   :param requires_grad: 是否跟踪此张量上的操作
   :type requires_grad: bool, optional
   :return: 填充一的张量
   :rtype: riemann.TN

.. function:: riemann.empty(*shape, dtype=None, requires_grad=False)

   创建一个未初始化的张量。

   :param shape: 张量的形状
   :type shape: int 或整数元组
   :param dtype: 张量的期望数据类型
   :type dtype: numpy.dtype, optional
   :param requires_grad: 是否跟踪此张量上的操作
   :type requires_grad: bool, optional
   :return: 未初始化的张量
   :rtype: riemann.TN

.. function:: riemann.full(*shape, fill_value, dtype=None, requires_grad=False)

   创建一个填充特定值的张量。

   :param shape: 张量的形状
   :type shape: int 或整数元组
   :param fill_value: 填充张量的值
   :type fill_value: 标量
   :param dtype: 张量的期望数据类型
   :type dtype: numpy.dtype, optional
   :param requires_grad: 是否跟踪此张量上的操作
   :type requires_grad: bool, optional
   :return: 填充指定值的张量
   :rtype: riemann.TN

.. function:: riemann.eye(n, m=None, dtype=None, requires_grad=False)

   创建一个对角线为一，其余为零的二维张量。

   :param n: 行数
   :type n: int
   :param m: 列数（默认为 n）
   :type m: int, optional
   :param dtype: 张量的期望数据类型
   :type dtype: numpy.dtype, optional
   :param requires_grad: 是否跟踪此张量上的操作
   :type requires_grad: bool, optional
   :return: 对角线为一的二维张量
   :rtype: riemann.TN

.. function:: riemann.zeros_like(tsr, dtype=None, requires_grad=False)

   创建一个与输入张量形状相同的零张量。

   :param tsr: 参考张量
   :type tsr: riemann.TN
   :param dtype: 张量的期望数据类型
   :type dtype: numpy.dtype, optional
   :param requires_grad: 是否跟踪此张量上的操作
   :type requires_grad: bool, optional
   :return: 与输入张量形状相同的零张量
   :rtype: riemann.TN

.. function:: riemann.ones_like(tsr, dtype=None, requires_grad=False)

   创建一个与输入张量形状相同的一张量。

   :param tsr: 参考张量
   :type tsr: riemann.TN
   :param dtype: 张量的期望数据类型
   :type dtype: numpy.dtype, optional
   :param requires_grad: 是否跟踪此张量上的操作
   :type requires_grad: bool, optional
   :return: 与输入张量形状相同的一张量
   :rtype: riemann.TN

.. function:: riemann.empty_like(tsr, dtype=None, requires_grad=False)

   创建一个与输入张量形状相同的未初始化张量。

   :param tsr: 参考张量
   :type tsr: riemann.TN
   :param dtype: 张量的期望数据类型
   :type dtype: numpy.dtype, optional
   :param requires_grad: 是否跟踪此张量上的操作
   :type requires_grad: bool, optional
   :return: 与输入张量形状相同的未初始化张量
   :rtype: riemann.TN

.. function:: riemann.full_like(tsr, fill_value, dtype=None, requires_grad=False)

   创建一个与输入张量形状相同的填充特定值的张量。

   :param tsr: 参考张量
   :type tsr: riemann.TN
   :param fill_value: 填充张量的值
   :type fill_value: 标量
   :param dtype: 张量的期望数据类型
   :type dtype: numpy.dtype, optional
   :param requires_grad: 是否跟踪此张量上的操作
   :type requires_grad: bool, optional
   :return: 与输入张量形状相同的填充指定值的张量
   :rtype: riemann.TN

随机数生成
~~~~~~~~~~

.. function:: riemann.rand(*size, requires_grad=False, dtype=None)

   创建一个填充来自 [0, 1) 均匀分布的随机数的张量。

   :param size: 张量的形状
   :type size: int 或整数元组
   :param requires_grad: 是否跟踪此张量上的操作
   :type requires_grad: bool, optional
   :param dtype: 张量的期望数据类型
   :type dtype: numpy.dtype, optional
   :return: 填充随机值的张量
   :rtype: riemann.TN

.. function:: riemann.randn(*size, requires_grad=False, dtype=None)

   创建一个填充来自标准正态分布的随机数的张量。

   :param size: 张量的形状
   :type size: int 或整数元组
   :param requires_grad: 是否跟踪此张量上的操作
   :type requires_grad: bool, optional
   :param dtype: 张量的期望数据类型
   :type dtype: numpy.dtype, optional
   :return: 填充随机值的张量
   :rtype: riemann.TN

.. function:: riemann.randint(low, high, size, requires_grad=False, dtype=int64)

   创建一个填充从 low（包含）到 high（不包含）的随机整数的张量。

   :param low: 要抽取的最小整数
   :type low: int
   :param high: 要抽取的最大整数加一
   :type high: int
   :param size: 张量的形状
   :type size: int 或整数元组
   :param requires_grad: 是否跟踪此张量上的操作
   :type requires_grad: bool, optional
   :param dtype: 张量的期望数据类型
   :type dtype: numpy.dtype, optional
   :return: 填充随机整数的张量
   :rtype: riemann.TN

.. function:: riemann.randperm(n, requires_grad=False, dtype=int64)

   创建一个包含 0 到 n-1 随机顺序数字的张量。

   :param n: 上限（不包含）
   :type n: int
   :param requires_grad: 是否跟踪此张量上的操作
   :type requires_grad: bool, optional
   :param dtype: 张量的期望数据类型
   :type dtype: numpy.dtype, optional
   :return: 包含随机排列整数的张量
   :rtype: riemann.TN

.. function:: riemann.normal(mean, std, size, dtype=None)

   创建一个填充来自正态分布的随机数的张量。

   :param mean: 正态分布的均值
   :type mean: float
   :param std: 正态分布的标准差
   :type std: float
   :param size: 张量的形状
   :type size: int 或整数元组
   :param dtype: 张量的期望数据类型
   :type dtype: numpy.dtype, optional
   :return: 填充随机值的张量
   :rtype: riemann.TN

随机种子控制
~~~~~~~~~~~~

.. function:: riemann.manual_seed(seed)

   设置随机数生成器的种子，用于确保随机操作的可重复性。

   :param seed: 随机种子值
   :type seed: int
   :return: 随机数生成器对象
   :rtype: torch.Generator

序列和范围函数
~~~~~~~~~~~~~~

.. function:: riemann.arange(start, end=None, step=1.0, dtype=None, requires_grad=False)

   创建一个从 start 到 end 以 step 为步长的 1-D 张量。

   :param start: 起始值
   :type start: float
   :param end: 结束值（不包含）
   :type end: float, optional
   :param step: 值之间的间距
   :type step: float, optional
   :param dtype: 张量的期望数据类型
   :type dtype: numpy.dtype, optional
   :param requires_grad: 是否跟踪此张量上的操作
   :type requires_grad: bool, optional
   :return: 包含等间距值的 1-D 张量
   :rtype: riemann.TN

.. function:: riemann.linspace(start, end, steps=100, endpoint=True, dtype=None, requires_grad=False)

   创建一个在给定区间内包含等间距值的 1-D 张量。

   :param start: 起始值
   :type start: float
   :param end: 结束值
   :type end: float
   :param steps: 要生成的样本数
   :type steps: int, optional
   :param endpoint: 是否包含结束值
   :type endpoint: bool, optional
   :param dtype: 张量的期望数据类型
   :type dtype: numpy.dtype, optional
   :param requires_grad: 是否跟踪此张量上的操作
   :type requires_grad: bool, optional
   :return: 包含等间距值的 1-D 张量
   :rtype: riemann.TN

张量属性
~~~~~~~~

.. method:: riemann.TN.dtype

   返回张量的数据类型。

   :return: 张量的数据类型
   :rtype: numpy.dtype

.. method:: riemann.TN.real

   返回复数张量的实部。

   :return: 包含实部的张量
   :rtype: riemann.TN

.. method:: riemann.TN.imag

   返回复数张量的虚部。

   :return: 包含虚部的张量
   :rtype: riemann.TN

.. method:: riemann.TN.shape

   返回张量的形状。

   :return: 张量各维度大小的元组
   :rtype: tuple

.. method:: riemann.TN.ndim

   返回张量的维度数。

   :return: 张量的维度数
   :rtype: int

.. method:: riemann.TN.device

   返回张量所在的设备。

   :return: 张量所在的设备对象
   :rtype: Device

.. method:: riemann.TN.is_cuda

   检查张量是否在CUDA设备上。

   :return: 如果张量在CUDA设备上返回True，否则返回False
   :rtype: bool

.. method:: riemann.TN.is_cpu

   检查张量是否在CPU设备上。

   :return: 如果张量在CPU设备上返回True，否则返回False
   :rtype: bool

.. method:: riemann.TN.is_leaf

   检查张量是否为叶子节点。

   :return: 如果张量为叶子节点返回True，否则返回False
   :rtype: bool

.. method:: riemann.TN.is_floating_point()

   检查张量是否为浮点类型。

   :return: 如果张量是浮点类型返回True，否则返回False
   :rtype: bool

.. method:: riemann.TN.is_complex()

   检查张量是否为复数类型。

   :return: 如果张量是复数类型返回True，否则返回False
   :rtype: bool

.. method:: riemann.TN.isreal()

   判断张量元素是否为实数。

   :return: 布尔张量，True表示对应元素为实数
   :rtype: riemann.TN

.. method:: riemann.TN.isinf()

   判断张量元素是否为无穷大。

   :return: 布尔张量，True表示对应元素为无穷大
   :rtype: riemann.TN

.. method:: riemann.TN.isnan()

   判断张量元素是否为NaN（非数值）。

   :return: 布尔张量，True表示对应元素为NaN
   :rtype: riemann.TN

.. method:: riemann.TN.conj()

   返回张量的复数共轭。

   :return: 包含共轭元素的张量
   :rtype: riemann.TN

.. method:: riemann.TN.size(dim=None)

   返回张量的大小。

   :param dim: 要查询的维度索引，如果为None返回整个形状
   :type dim: int, optional
   :return: 如果dim为None返回形状元组，否则返回指定维度的大小
   :rtype: tuple or int

.. method:: riemann.TN.numel()

   返回张量中元素的总数。

   :return: 张量中元素的数量
   :rtype: int

.. method:: riemann.TN.is_contiguous()

   检查张量的内存布局是否是连续的。

   :return: 如果张量是连续的返回True，否则返回False
   :rtype: bool

张量形状操作
~~~~~~~~~~~~

.. function:: riemann.reshape(input, shape)

   返回一个具有相同数据但不同形状的张量。

   :param input: 输入张量
   :type input: riemann.TN
   :param shape: 新形状
   :type shape: 整数元组
   :return: 具有新形状的张量
   :rtype: riemann.TN

.. function:: riemann.squeeze(input, dim=None)

   从张量的形状中移除大小为 1 的维度。

   :param input: 输入张量
   :type input: riemann.TN
   :param dim: 要压缩的维度
   :type dim: int, optional
   :return: 压缩维度后的张量
   :rtype: riemann.TN

.. function:: riemann.unsqueeze(input, dim)

   在指定位置插入大小为 1 的维度。

   :param input: 输入张量
   :type input: riemann.TN
   :param dim: 要扩展的维度
   :type dim: int
   :return: 扩展维度后的张量
   :rtype: riemann.TN

.. function:: riemann.transpose(input, dim0, dim1)

   交换张量的两个维度。

   :param input: 输入张量
   :type input: riemann.TN
   :param dim0: 要交换的第一个维度
   :type dim0: int
   :param dim1: 要交换的第二个维度
   :type dim1: int
   :return: 交换维度后的张量
   :rtype: riemann.TN

.. attribute:: riemann.TN.mT

   矩阵转置，即张量最后两个维度间的转置。

   对于行向量 (1, n)，转置为列向量 (n, 1)；对于列向量 (n, 1)，转置为行向量 (1, n)。

   :return: 转置后的张量
   :rtype: riemann.TN

.. function:: riemann.is_contiguous(input)

   检查张量的内存是否连续。

   :param input: 输入张量
   :type input: riemann.TN
   :return: 内存是否连续
   :rtype: bool

.. function:: riemann.contiguous(input)

   返回一个内存连续的张量。

   :param input: 输入张量
   :type input: riemann.TN
   :return: 内存连续的张量
   :rtype: riemann.TN

.. function:: riemann.gather(input, dim, index)

   沿指定维度收集张量中的元素。

   :param input: 输入张量
   :type input: riemann.TN
   :param dim: 收集的维度
   :type dim: int
   :param index: 索引张量
   :type index: riemann.TN
   :return: 收集后的张量
   :rtype: riemann.TN

.. function:: riemann.scatter(input, dim, index, src)

   沿指定维度将源张量中的元素分散到目标张量中。

   :param input: 目标张量
   :type input: riemann.TN
   :param dim: 分散的维度
   :type dim: int
   :param index: 索引张量
   :type index: riemann.TN
   :param src: 源张量
   :type src: riemann.TN
   :return: 分散后的张量
   :rtype: riemann.TN

.. function:: riemann.broadcast_to(input, size)

   将张量广播到新形状。

   :param input: 输入张量
   :type input: riemann.TN
   :param size: 目标形状
   :type size: 整数元组
   :return: 广播后的张量
   :rtype: riemann.TN

.. function:: riemann.flip(input, dims)

   沿指定维度反转元素的顺序。

   :param input: 输入张量
   :type input: riemann.TN
   :param dims: 要翻转的维度
   :type dims: int 的列表或元组
   :return: 翻转后的张量
   :rtype: riemann.TN

.. function:: riemann.split(ts, split_indices, dim=0)

   将张量拆分为多个子张量。

   :param ts: 输入张量
   :type ts: riemann.TN
   :param split_indices: 拆分的索引
   :type split_indices: int 或整数列表
   :param dim: 沿其拆分的维度
   :type dim: int, optional
   :return: 张量列表
   :rtype: riemann.TN 的列表

.. function:: riemann.stack(tensors, dim=0)

   沿新维度堆叠张量。

   :param tensors: 要堆叠的张量序列
   :type tensors: riemann.TN 的序列
   :param dim: 要插入的维度
   :type dim: int, optional
   :return: 堆叠后的张量
   :rtype: riemann.TN

.. function:: riemann.cat(tensors, dim=0)

   沿现有维度连接张量。

   :param tensors: 要连接的张量序列
   :type tensors: riemann.TN 的序列
   :param dim: 沿其连接的维度
   :type dim: int, optional
   :return: 连接后的张量
   :rtype: riemann.TN

.. function:: riemann.concatenate(tensors, dim=0)

   沿现有维度连接张量。

   :param tensors: 要连接的张量序列
   :type tensors: riemann.TN 的序列
   :param dim: 沿其连接的维度
   :type dim: int, optional
   :return: 连接后的张量
   :rtype: riemann.TN

.. function:: riemann.vstack(tensors)

   垂直堆叠张量（按行）。

   :param tensors: 要堆叠的张量序列
   :type tensors: riemann.TN 的序列
   :return: 垂直堆叠的张量
   :rtype: riemann.TN

.. function:: riemann.hstack(tensors)

   水平堆叠张量（按列）。

   :param tensors: 要堆叠的张量序列
   :type tensors: riemann.TN 的序列
   :return: 水平堆叠的张量
   :rtype: riemann.TN

张量运算符
~~~~~~~~~~

Riemann框架支持丰富的张量运算符，包括算术运算符、比较运算符、位运算符和原地运算符。这些运算符可以直接作用于张量对象，支持自动微分，并遵循Python的运算符优先级规则。

算术运算符
``````````

.. method:: __add__(other)

   张量加法运算符，等价于 `+`。

   :param other: 另一个张量或标量值
   :type other: riemann.TN 或 int 或 float 或 complex
   :return: 加法结果张量
   :rtype: riemann.TN

.. method:: __radd__(other)

   反向张量加法运算符，当左操作数不是张量时使用。

   :param other: 非张量左操作数
   :type other: int 或 float 或 complex
   :return: 加法结果张量
   :rtype: riemann.TN

.. method:: __sub__(other)

   张量减法运算符，等价于 `-`。

   :param other: 另一个张量或标量值
   :type other: riemann.TN 或 int 或 float 或 complex
   :return: 减法结果张量
   :rtype: riemann.TN

.. method:: __rsub__(other)

   反向张量减法运算符，当左操作数不是张量时使用。

   :param other: 非张量左操作数
   :type other: int 或 float 或 complex
   :return: 减法结果张量
   :rtype: riemann.TN

.. method:: __mul__(other)

   张量乘法运算符，等价于 `*`。

   :param other: 另一个张量或标量值
   :type other: riemann.TN 或 int 或 float 或 complex
   :return: 乘法结果张量
   :rtype: riemann.TN

.. method:: __rmul__(other)

   反向张量乘法运算符，当左操作数不是张量时使用。

   :param other: 非张量左操作数
   :type other: int 或 float 或 complex
   :return: 乘法结果张量
   :rtype: riemann.TN

.. method:: __matmul__(other)

   张量矩阵乘法运算符，等价于 `@`。

   :param other: 另一个张量
   :type other: riemann.TN
   :return: 矩阵乘法结果张量
   :rtype: riemann.TN
   :raises RuntimeError: 如果任一操作数是标量

.. method:: __rmatmul__(other)

   反向张量矩阵乘法运算符，当左操作数不是张量时使用。

   :param other: 非张量左操作数
   :type other: int 或 float 或 complex
   :return: 矩阵乘法结果张量
   :rtype: riemann.TN

.. method:: __truediv__(other)

   张量除法运算符，等价于 `/`。

   :param other: 另一个张量或标量值
   :type other: riemann.TN 或 int 或 float 或 complex
   :return: 除法结果张量
   :rtype: riemann.TN

.. method:: __rtruediv__(other)

   反向张量除法运算符，当左操作数不是张量时使用。

   :param other: 非张量左操作数
   :type other: int 或 float 或 complex
   :return: 除法结果张量
   :rtype: riemann.TN

.. method:: __pow__(other)

   张量幂运算符，等价于 `**`。

   :param other: 指数张量或标量值
   :type other: riemann.TN 或 int 或 float 或 complex
   :return: 幂运算结果张量
   :rtype: riemann.TN

.. method:: __rpow__(other)

   反向张量幂运算符，当左操作数不是张量时使用。

   :param other: 非张量左操作数
   :type other: int 或 float 或 complex
   :return: 幂运算结果张量
   :rtype: riemann.TN

.. method:: __pos__()

   张量正号运算符，等价于 `+`。

   :return: 原张量
   :rtype: riemann.TN

.. method:: __neg__()

   张量负号运算符，等价于 `-`。

   :return: 取负结果张量
   :rtype: riemann.TN

比较运算符
``````````

.. method:: __lt__(other)

   张量小于运算符，等价于 `<`。

   :param other: 另一个张量或标量值
   :type other: riemann.TN 或 int 或 float 或 complex
   :return: 布尔结果张量
   :rtype: riemann.TN
   :raises TypeError: 如果other是None

.. method:: __le__(other)

   张量小于等于运算符，等价于 `<=`。

   :param other: 另一个张量或标量值
   :type other: riemann.TN 或 int 或 float 或 complex
   :return: 布尔结果张量
   :rtype: riemann.TN
   :raises TypeError: 如果other是None

.. method:: __gt__(other)

   张量大于运算符，等价于 `>`。

   :param other: 另一个张量或标量值
   :type other: riemann.TN 或 int 或 float 或 complex
   :return: 布尔结果张量
   :rtype: riemann.TN
   :raises TypeError: 如果other是None

.. method:: __ge__(other)

   张量大于等于运算符，等价于 `>=`。

   :param other: 另一个张量或标量值
   :type other: riemann.TN 或 int 或 float 或 complex
   :return: 布尔结果张量
   :rtype: riemann.TN
   :raises TypeError: 如果other是None

.. method:: __eq__(other)

   张量等于运算符，等价于 `==`。

   :param other: 另一个张量或标量值
   :type other: riemann.TN 或 int 或 float 或 complex
   :return: 布尔结果张量
   :rtype: riemann.TN

.. method:: __ne__(other)

   张量不等于运算符，等价于 `!=`。

   :param other: 另一个张量或标量值
   :type other: riemann.TN 或 int 或 float 或 complex
   :return: 布尔结果张量
   :rtype: riemann.TN

位运算符
````````

.. method:: __and__(other)

   张量按位与运算符，等价于 `&`。

   :param other: 另一个张量或标量值
   :type other: riemann.TN 或 int
   :return: 按位与结果张量
   :rtype: riemann.TN

.. method:: __or__(other)

   张量按位或运算符，等价于 `|`。

   :param other: 另一个张量或标量值
   :type other: riemann.TN 或 int
   :return: 按位或结果张量
   :rtype: riemann.TN

.. method:: __xor__(other)

   张量按位异或运算符，等价于 `^`。

   :param other: 另一个张量或标量值
   :type other: riemann.TN 或 int
   :return: 按位异或结果张量
   :rtype: riemann.TN

.. method:: __invert__()

   张量按位取反运算符，等价于 `~`。

   :return: 按位取反结果张量
   :rtype: riemann.TN

.. method:: __lshift__(other)

   张量左移位运算符，等价于 `<<`。

   :param other: 移位位数
   :type other: riemann.TN 或 int
   :return: 左移位结果张量
   :rtype: riemann.TN

.. method:: __rshift__(other)

   张量右移位运算符，等价于 `>>`。

   :param other: 移位位数
   :type other: riemann.TN 或 int
   :return: 右移位结果张量
   :rtype: riemann.TN

原地运算符
``````````

.. method:: __iadd__(other)

   张量原地加法运算符，等价于 `+=`。

   :param other: 另一个张量或标量值
   :type other: riemann.TN 或 int 或 float 或 complex
   :return: 原地修改后的张量
   :rtype: riemann.TN
   :raises RuntimeError: 如果张量是需要梯度的叶子节点

.. method:: __isub__(other)

   张量原地减法运算符，等价于 `-=`。

   :param other: 另一个张量或标量值
   :type other: riemann.TN 或 int 或 float 或 complex
   :return: 原地修改后的张量
   :rtype: riemann.TN
   :raises RuntimeError: 如果张量是需要梯度的叶子节点

.. method:: __imul__(other)

   张量原地乘法运算符，等价于 `*=`。

   :param other: 另一个张量或标量值
   :type other: riemann.TN 或 int 或 float 或 complex
   :return: 原地修改后的张量
   :rtype: riemann.TN
   :raises RuntimeError: 如果张量是需要梯度的叶子节点

.. method:: __itruediv__(other)

   张量原地除法运算符，等价于 `/=`。

   :param other: 另一个张量或标量值
   :type other: riemann.TN 或 int 或 float 或 complex
   :return: 原地修改后的张量
   :rtype: riemann.TN
   :raises RuntimeError: 如果张量是需要梯度的叶子节点

.. method:: __ipow__(other)

   张量原地幂运算符，等价于 `**=`。

   :param other: 指数张量或标量值
   :type other: riemann.TN 或 int 或 float 或 complex
   :return: 原地修改后的张量
   :rtype: riemann.TN
   :raises RuntimeError: 如果张量是需要梯度的叶子节点

数学运算
~~~~~~~~

.. function:: riemann.matmul(input, other)

   两个张量的矩阵乘法。

   :param input: 第一个张量
   :type input: riemann.TN
   :param other: 第二个张量
   :type other: riemann.TN
   :return: 张量的矩阵乘积
   :rtype: riemann.TN

.. function:: riemann.dot(x, y)

   计算两个张量的点积。

   :param x: 第一个张量
   :type x: riemann.TN
   :param y: 第二个张量
   :type y: riemann.TN
   :return: 点积结果
   :rtype: riemann.TN

.. function:: riemann.sum(x, dim=None, keepdim=False)

   计算跨维度元素的总和。

   :param x: 输入张量
   :type x: riemann.TN
   :param dim: 要求和的维度
   :type dim: int 或整数元组, optional
   :param keepdim: 是否保持缩减的维度
   :type keepdim: bool, optional
   :return: 元素的总和
   :rtype: riemann.TN

.. function:: riemann.prod(x, dim=None, keepdim=False)

   计算跨维度元素的乘积。

   :param x: 输入张量
   :type x: riemann.TN
   :param dim: 要相乘的维度
   :type dim: int 或整数元组, optional
   :param keepdim: 是否保持缩减的维度
   :type keepdim: bool, optional
   :return: 元素的乘积
   :rtype: riemann.TN

.. function:: riemann.mean(x, dim=None, keepdim=False)

   计算跨维度元素的平均值。

   :param x: 输入张量
   :type x: riemann.TN
   :param dim: 要平均的维度
   :type dim: int 或整数元组, optional
   :param keepdim: 是否保持缩减的维度
   :type keepdim: bool, optional
   :return: 元素的平均值
   :rtype: riemann.TN

.. function:: riemann.var(x, dim=None, unbiased=True, keepdim=False)

   计算跨维度元素的方差。

   :param x: 输入张量
   :type x: riemann.TN
   :param dim: 要计算方差的维度
   :type dim: int 或整数元组, optional
   :param unbiased: 是否使用无偏估计
   :type unbiased: bool, optional
   :param keepdim: 是否保持缩减的维度
   :type keepdim: bool, optional
   :return: 元素的方差
   :rtype: riemann.TN

.. function:: riemann.std(x, dim=None, unbiased=True, keepdim=False)

   计算跨维度元素的标准差。

   :param x: 输入张量
   :type x: riemann.TN
   :param dim: 要计算标准差的维度
   :type dim: int 或整数元组, optional
   :param unbiased: 是否使用无偏估计
   :type unbiased: bool, optional
   :param keepdim: 是否保持缩减的维度
   :type keepdim: bool, optional
   :return: 元素的标准差
   :rtype: riemann.TN

.. function:: riemann.norm(x, p="fro", dim=None, keepdim=False)

   计算张量的范数。

   :param x: 输入张量
   :type x: riemann.TN
   :param p: 范数阶数
   :type p: int, float, str, optional
   :param dim: 要计算范数的维度
   :type dim: int 或整数元组, optional
   :param keepdim: 是否保持缩减的维度
   :type keepdim: bool, optional
   :return: 张量的范数
   :rtype: riemann.TN

.. function:: riemann.max(x, dim=None, keepdim=False, *, out=None)

   计算跨维度元素的最大值。

   :param x: 输入张量
   :type x: riemann.TN
   :param dim: 要查找最大值的维度
   :type dim: int, optional
   :param keepdim: 是否保持缩减的维度
   :type keepdim: bool, optional
   :param out: 输出张量
   :type out: riemann.TN, optional
   :return: 最大值
   :rtype: riemann.TN

.. function:: riemann.min(x, dim=None, keepdim=False, *, out=None)

   计算跨维度元素的最小值。

   :param x: 输入张量
   :type x: riemann.TN
   :param dim: 要查找最小值的维度
   :type dim: int, optional
   :param keepdim: 是否保持缩减的维度
   :type keepdim: bool, optional
   :param out: 输出张量
   :type out: riemann.TN, optional
   :return: 最小值
   :rtype: riemann.TN

.. function:: riemann.argmax(x, dim=None, keepdim=False, *, out=None)

   计算跨维度元素的最大值索引。

   :param x: 输入张量
   :type x: riemann.TN
   :param dim: 要查找最大值索引的维度
   :type dim: int, optional
   :param keepdim: 是否保持缩减的维度
   :type keepdim: bool, optional
   :param out: 输出张量
   :type out: riemann.TN, optional
   :return: 最大值索引
   :rtype: riemann.TN

.. function:: riemann.argmin(x, dim=None, keepdim=False, *, out=None)

   计算跨维度元素的最小值索引。

   :param x: 输入张量
   :type x: riemann.TN
   :param dim: 要查找最小值索引的维度
   :type dim: int, optional
   :param keepdim: 是否保持缩减的维度
   :type keepdim: bool, optional
   :param out: 输出张量
   :type out: riemann.TN, optional
   :return: 最小值索引
   :rtype: riemann.TN

.. function:: riemann.abs(x)

   计算每个元素的绝对值。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素的绝对值
   :rtype: riemann.TN

.. function:: riemann.sqrt(x)

   计算每个元素的平方根。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素的平方根
   :rtype: riemann.TN

.. function:: riemann.pow(input, exponent)

   将每个元素提升到幂。

   :param input: 输入张量
   :type input: riemann.TN
   :param exponent: 指数值
   :type exponent: riemann.TN 或标量
   :return: 输入张量的幂
   :rtype: riemann.TN

.. function:: riemann.log(x)

   计算每个元素的自然对数。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素的自然对数
   :rtype: riemann.TN

.. function:: riemann.log1p(x)

   计算每个元素加一的自然对数。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素加一的自然对数
   :rtype: riemann.TN

.. function:: riemann.log2(x)

   计算每个元素的以2为底的对数。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素的以2为底的对数
   :rtype: riemann.TN

.. function:: riemann.log10(x)

   计算每个元素的以10为底的对数。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素的以10为底的对数
   :rtype: riemann.TN

.. function:: riemann.exp(x)

   计算每个元素的指数。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素的指数
   :rtype: riemann.TN

.. function:: riemann.exp2(x)

   计算每个元素的2的幂。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素的2的幂
   :rtype: riemann.TN

.. function:: riemann.square(x)

   计算每个元素的平方。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素的平方
   :rtype: riemann.TN

.. function:: riemann.sin(x)

   计算每个元素的正弦。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素的正弦
   :rtype: riemann.TN

.. function:: riemann.cos(x)

   计算每个元素的余弦。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素的余弦
   :rtype: riemann.TN

.. function:: riemann.tan(x)

   计算每个元素的正切。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素的正切
   :rtype: riemann.TN

.. function:: riemann.cot(x)

   计算每个元素的余切。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素的余切
   :rtype: riemann.TN

.. function:: riemann.sec(x)

   计算每个元素的正割。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素的正割
   :rtype: riemann.TN

.. function:: riemann.csc(x)

   计算每个元素的余割。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素的余割
   :rtype: riemann.TN

.. function:: riemann.asin(x)

   计算每个元素的反正弦。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素的反正弦
   :rtype: riemann.TN

.. function:: riemann.acos(x)

   计算每个元素的反余弦。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素的反余弦
   :rtype: riemann.TN

.. function:: riemann.atan(x)

   计算每个元素的反正切。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素的反正切
   :rtype: riemann.TN

.. function:: riemann.sinh(x)

   计算每个元素的双曲正弦。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素的双曲正弦
   :rtype: riemann.TN

.. function:: riemann.cosh(x)

   计算每个元素的双曲余弦。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素的双曲余弦
   :rtype: riemann.TN

.. function:: riemann.tanh(x)

   计算每个元素的双曲正切。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素的双曲正切
   :rtype: riemann.TN

.. function:: riemann.coth(x)

   计算每个元素的双曲余切。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素的双曲余切
   :rtype: riemann.TN

.. function:: riemann.sech(x)

   计算每个元素的双曲正割。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素的双曲正割
   :rtype: riemann.TN

.. function:: riemann.csch(x)

   计算每个元素的双曲余割。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素的双曲余割
   :rtype: riemann.TN

.. function:: riemann.arcsinh(x)

   计算每个元素的反双曲正弦。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素的反双曲正弦
   :rtype: riemann.TN

.. function:: riemann.arccosh(x)

   计算每个元素的反双曲余弦。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素的反双曲余弦
   :rtype: riemann.TN

.. function:: riemann.arctanh(x)

   计算每个元素的反双曲正切。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素的反双曲正切
   :rtype: riemann.TN

.. function:: riemann.ceil(x)

   向上取整，返回不小于每个元素的最小整数。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 向上取整后的张量
   :rtype: riemann.TN

.. function:: riemann.floor(x)

   向下取整，返回不大于每个元素的最大整数。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 向下取整后的张量
   :rtype: riemann.TN

.. function:: riemann.round(x)

   四舍五入到最近的整数。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 四舍五入后的张量
   :rtype: riemann.TN

.. function:: riemann.trunc(x)

   截断小数部分，返回整数部分。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 截断后的张量
   :rtype: riemann.TN

.. function:: riemann.sign(x)

   计算每个元素的符号。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素的符号
   :rtype: riemann.TN

.. function:: riemann.where(cond, x=None, y=None)

   根据条件从 x 或 y 中选择元素。

   :param cond: 条件张量
   :type cond: riemann.TN
   :param x: 当条件为 True 时选择的张量
   :type x: riemann.TN, optional
   :param y: 当条件为 False 时选择的张量
   :type y: riemann.TN, optional
   :return: 从 x 或 y 中选取元素组成的张量
   :rtype: riemann.TN

.. function:: riemann.clamp(x, min=None, max=None, out=None)

   将所有元素限制在指定范围内。

   :param x: 输入张量
   :type x: riemann.TN
   :param min: 最小值
   :type min: float, optional
   :param max: 最大值
   :type max: float, optional
   :param out: 输出张量
   :type out: riemann.TN, optional
   :return: 元素被限制在指定范围内的张量
   :rtype: riemann.TN

.. function:: riemann.masked_fill(input, mask, value)

   根据掩码填充值到张量中。

   :param input: 输入张量
   :type input: riemann.TN
   :param mask: 掩码张量，形状与输入张量相同
   :type mask: riemann.TN
   :param value: 填充的值
   :type value: scalar
   :return: 填充后的张量
   :rtype: riemann.TN

.. function:: riemann.maximum(input, other)

   计算两个张量的元素级最大值。

   :param input: 第一个输入张量
   :type input: riemann.TN
   :param other: 第二个输入张量
   :type other: riemann.TN
   :return: 元素级最大值组成的张量
   :rtype: riemann.TN

.. function:: riemann.minimum(input, other)

   计算两个张量的元素级最小值。

   :param input: 第一个输入张量
   :type input: riemann.TN
   :param other: 第二个输入张量
   :type other: riemann.TN
   :return: 元素级最小值组成的张量
   :rtype: riemann.TN

.. function:: riemann.diagonal(input, offset=0, dim1=-2, dim2=-1)

   返回张量的对角线。

   :param input: 输入张量
   :type input: riemann.TN
   :param offset: 对角线的偏移量
   :type offset: int, optional
   :param dim1: 对角线的第一个维度
   :type dim1: int, optional
   :param dim2: 对角线的第二个维度
   :type dim2: int, optional
   :return: 张量的对角线
   :rtype: riemann.TN

.. function:: riemann.diag(input, offset=0)

   返回二维张量的对角线或构造对角矩阵。

   :param input: 输入张量
   :type input: riemann.TN
   :param offset: 对角线的偏移量
   :type offset: int, optional
   :return: 张量的对角线或对角矩阵
   :rtype: riemann.TN

.. function:: riemann.fill_diagonal(input, value, offset=0, dim1=-2, dim2=-1)

   用指定值填充张量的对角线。

   :param input: 输入张量
   :type input: riemann.TN
   :param value: 填充对角线的值
   :type value: 标量
   :param offset: 对角线的偏移量
   :type offset: int, optional
   :param dim1: 对角线的第一个维度
   :type dim1: int, optional
   :param dim2: 对角线的第二个维度
   :type dim2: int, optional
   :return: 对角线被填充的张量
   :rtype: riemann.TN

.. function:: riemann.batch_diag(v)

   返回张量的批处理对角线。

   :param v: 输入张量
   :type v: riemann.TN
   :return: 张量的批处理对角线
   :rtype: riemann.TN

.. function:: riemann.nonzero(input, *, as_tuple=False)

   返回非零元素的索引。

   :param input: 输入张量
   :type input: riemann.TN
   :param as_tuple: 是否以张量元组的形式返回
   :type as_tuple: bool, optional
   :return: 非零元素的索引
   :rtype: riemann.TN 或 riemann.TN 的元组

.. function:: riemann.tril(input_tensor, diagonal=0)

   返回矩阵的下三角部分。

   :param input_tensor: 输入张量
   :type input_tensor: riemann.TN
   :param diagonal: 对角线偏移量
   :type diagonal: int, optional
   :return: 矩阵的下三角部分
   :rtype: riemann.TN

.. function:: riemann.triu(input_tensor, diagonal=0)

   返回矩阵的上三角部分。

   :param input_tensor: 输入张量
   :type input_tensor: riemann.TN
   :param diagonal: 对角线偏移量
   :type diagonal: int, optional
   :return: 矩阵的上三角部分
   :rtype: riemann.TN

.. function:: riemann.cumsum(input, dim, *, dtype=None)

   计算张量沿指定维度的累积和。

   :param input: 输入张量
   :type input: riemann.TN
   :param dim: 计算累积和的维度
   :type dim: int
   :param dtype: 输出张量的数据类型
   :type dtype: riemann.dtype, optional
   :return: 累积和结果
   :rtype: riemann.TN

.. function:: riemann.unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None)

   返回张量中的唯一元素。

   :param input: 输入张量
   :type input: riemann.TN
   :param sorted: 是否对唯一值进行排序
   :type sorted: bool, optional
   :param return_inverse: 是否返回逆索引
   :type return_inverse: bool, optional
   :param return_counts: 是否返回每个唯一值的计数
   :type return_counts: bool, optional
   :param dim: 沿哪个维度查找唯一值，默认为None（展平后查找）
   :type dim: int, optional
   :return: 唯一值，如果指定return_inverse或return_counts则返回元组
   :rtype: riemann.TN or tuple

.. function:: riemann.broadcast_tensors(*tensors)

   广播多个张量到相同的形状。

   :param tensors: 要广播的张量序列
   :type tensors: riemann.TN
   :return: 广播后的张量列表
   :rtype: list of riemann.TN

.. function:: riemann.repeat(input, repeats, dim=None)

   沿指定维度重复张量元素。

   :param input: 输入张量
   :type input: riemann.TN
   :param repeats: 每个元素的重复次数
   :type repeats: int
   :param dim: 要重复的维度，默认为None（展平后重复）
   :type dim: int, optional
   :return: 重复后的张量
   :rtype: riemann.TN

.. function:: riemann.outer(input, vec2)

   计算两个向量的外积。

   :param input: 第一个输入向量
   :type input: riemann.TN
   :param vec2: 第二个输入向量
   :type vec2: riemann.TN
   :return: 外积矩阵
   :rtype: riemann.TN

比较运算
~~~~~~~~

.. function:: riemann.equal(a, b)

   计算元素级的相等性。

   :param a: 第一个张量
   :type a: riemann.TN
   :param b: 第二个张量
   :type b: riemann.TN
   :return: 指示相等性的布尔张量
   :rtype: bool

.. function:: riemann.not_equal(a, b)

   计算元素级的不等性。

   :param a: 第一个张量
   :type a: riemann.TN
   :param b: 第二个张量
   :type b: riemann.TN
   :return: 指示不等性的布尔张量
   :rtype: bool

.. function:: riemann.allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False)

   如果两个张量在容差范围内元素级相等，则返回 True。

   :param a: 第一个张量
   :type a: riemann.TN
   :param b: 第二个张量
   :type b: riemann.TN
   :param rtol: 相对容差
   :type rtol: float, optional
   :param atol: 绝对容差
   :type atol: float, optional
   :param equal_nan: 是否将 NaN 值视为相等
   :type equal_nan: bool, optional
   :return: 张量是否接近
   :rtype: bool

.. function:: riemann.isinf(x)

   元素级测试是否为无穷大。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 指示无穷大的布尔张量
   :rtype: riemann.TN

.. function:: riemann.isnan(x)

   元素级测试是否为 NaN。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 指示 NaN 的布尔张量
   :rtype: riemann.TN

.. function:: riemann.isreal(x)

   元素级测试是否为实数。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 指示实数的布尔张量
   :rtype: riemann.TN

排序运算
~~~~~~~~

.. function:: riemann.sort(input, dim=-1, descending=False, stable=False, *, out=None)

   沿给定维度对张量元素进行排序。

   :param input: 输入张量
   :type input: riemann.TN
   :param dim: 排序的维度
   :type dim: int, optional
   :param descending: 是否按降序排序
   :type descending: bool, optional
   :param stable: 是否使用稳定排序
   :type stable: bool, optional
   :param out: 输出张量
   :type out: riemann.TN, optional
   :return: 排序后的张量和索引
   :rtype: riemann.TN, riemann.TN

.. function:: riemann.argsort(input, dim=-1, descending=False, stable=False, *, out=None)

   沿给定维度返回将张量排序的索引。

   :param input: 输入张量
   :type input: riemann.TN
   :param dim: 排序的维度
   :type dim: int, optional
   :param descending: 是否按降序排序
   :type descending: bool, optional
   :param stable: 是否使用稳定排序
   :type stable: bool, optional
   :param out: 输出张量
   :type out: riemann.TN, optional
   :return: 排序索引
   :rtype: riemann.TN

原地操作
~~~~~~~~

.. method:: riemann.TN.setat_(index, val)

   原地设置张量指定位置的值。

   :param index: 索引，指定要设置值的位置
   :type index: int, slice, tuple, or array
   :param val: 要设置的值
   :type val: riemann.TN, numpy.ndarray, list, or scalar
   :return: 原地修改后的张量
   :rtype: riemann.TN

.. method:: riemann.TN.addat_(index, val)

   原地将值加到张量指定位置。

   :param index: 索引，指定要操作的位置
   :type index: int, slice, tuple, or array
   :param val: 要添加的值
   :type val: riemann.TN, numpy.ndarray, list, or scalar
   :return: 原地修改后的张量
   :rtype: riemann.TN

.. method:: riemann.TN.subat_(index, val)

   原地从张量指定位置减去值。

   :param index: 索引，指定要操作的位置
   :type index: int, slice, tuple, or array
   :param val: 要减去的值
   :type val: riemann.TN, numpy.ndarray, list, or scalar
   :return: 原地修改后的张量
   :rtype: riemann.TN

.. method:: riemann.TN.mulat_(index, val)

   原地将张量指定位置的值乘以给定值。

   :param index: 索引，指定要操作的位置
   :type index: int, slice, tuple, or array
   :param val: 要乘的值
   :type val: riemann.TN, numpy.ndarray, list, or scalar
   :return: 原地修改后的张量
   :rtype: riemann.TN

.. method:: riemann.TN.divat_(index, val)

   原地将张量指定位置的值除以给定值。

   :param index: 索引，指定要操作的位置
   :type index: int, slice, tuple, or array
   :param val: 除数
   :type val: riemann.TN, numpy.ndarray, list, or scalar
   :return: 原地修改后的张量
   :rtype: riemann.TN

.. method:: riemann.TN.powat_(index, val)

   原地对张量指定位置的值进行幂运算。

   :param index: 索引，指定要操作的位置
   :type index: int, slice, tuple, or array
   :param val: 指数
   :type val: riemann.TN, numpy.ndarray, list, or scalar
   :return: 原地修改后的张量
   :rtype: riemann.TN

.. method:: riemann.TN.scatter_(dim, index, src=None, *, value=None)
   :no-index:

   原地将值按照索引填充到张量中。

   :param dim: 沿着哪个维度进行索引
   :type dim: int
   :param index: 索引张量
   :type index: riemann.TN
   :param src: 源张量，提供要填充的值
   :type src: riemann.TN, optional
   :param value: 标量值，提供要填充的值
   :type value: int, float, complex, optional
   :return: 原地修改后的张量
   :rtype: riemann.TN

.. method:: riemann.TN.scatter_add_(dim, index, src)
   :no-index:

   原地将值按照索引累加到张量中。

   :param dim: 沿着哪个维度进行索引
   :type dim: int
   :param index: 索引张量
   :type index: riemann.TN
   :param src: 源张量，提供要累加的值
   :type src: riemann.TN
   :return: 原地修改后的张量
   :rtype: riemann.TN

.. method:: riemann.TN.requires_grad_(requires_grad=True)

   原地设置张量是否需要计算梯度。

   :param requires_grad: 是否需要计算梯度
   :type requires_grad: bool, optional
   :return: 原地修改后的张量
   :rtype: riemann.TN

.. method:: riemann.TN.add_(other)

   原地加法操作。

   :param other: 要加到当前张量上的值
   :type other: riemann.TN, numpy.ndarray, list, or scalar
   :return: 原地修改后的张量
   :rtype: riemann.TN

.. method:: riemann.TN.sub_(other)

   原地减法操作。

   :param other: 要从当前张量中减去的值
   :type other: riemann.TN, numpy.ndarray, list, or scalar
   :return: 原地修改后的张量
   :rtype: riemann.TN

.. method:: riemann.TN.mul_(other)

   原地乘法操作。

   :param other: 要与当前张量相乘的值
   :type other: riemann.TN, numpy.ndarray, list, or scalar
   :return: 原地修改后的张量
   :rtype: riemann.TN

.. method:: riemann.TN.div_(other)

   原地除法操作。

   :param other: 除数
   :type other: riemann.TN, numpy.ndarray, list, or scalar
   :return: 原地修改后的张量
   :rtype: riemann.TN

.. method:: riemann.TN.pow_(other)

   原地幂运算操作。

   :param other: 指数
   :type other: riemann.TN, numpy.ndarray, list, or scalar
   :return: 原地修改后的张量
   :rtype: riemann.TN

.. method:: riemann.TN.detach_()

   原地断开张量与计算图的连接。

   :return: 原地修改后的张量
   :rtype: riemann.TN

.. method:: riemann.TN.copy_(src)

   原地复制源张量到当前张量。

   :param src: 源张量
   :type src: riemann.TN, numpy.ndarray, list, or scalar
   :return: 原地修改后的张量
   :rtype: riemann.TN

.. method:: riemann.TN.zero_()

   原地将张量所有元素设置为0。

   :return: 原地修改后的张量
   :rtype: riemann.TN

.. method:: riemann.TN.fill_(value)

   原地将张量所有元素填充为指定值。

   :param value: 填充值
   :type value: riemann.TN, numpy.ndarray, list, or scalar
   :return: 原地修改后的张量
   :rtype: riemann.TN

.. method:: riemann.TN.clamp_(min=None, max=None)

   原地将张量元素限制在指定范围内。

   :param min: 最小值
   :type min: float, optional
   :param max: 最大值
   :type max: float, optional
   :return: 原地修改后的张量
   :rtype: riemann.TN

.. function:: riemann.masked_fill_(input, mask, value)

   原地版本的 masked_fill 函数，根据掩码填充值到张量中。

   :param input: 输入张量
   :type input: riemann.TN
   :param mask: 掩码张量，形状与输入张量相同
   :type mask: riemann.TN
   :param value: 填充的值
   :type value: scalar
   :return: 原地修改后的张量
   :rtype: riemann.TN

.. function:: riemann.fill_diagonal_(input, value, offset=0, dim1=-2, dim2=-1)

   fill_diagonal 的原地版本。

   :param input: 输入张量
   :type input: riemann.TN
   :param value: 填充对角线的值
   :type value: 标量
   :param offset: 对角线的偏移量
   :type offset: int, optional
   :param dim1: 对角线的第一个维度
   :type dim1: int, optional
   :param dim2: 对角线的第二个维度
   :type dim2: int, optional
   :return: 对角线被填充的输入张量
   :rtype: riemann.TN

收集与散射函数
~~~~~~~~~~~~~~

.. method:: riemann.TN.gather(dim, index)

   根据指定的维度和索引收集元素。

   :param dim: 收集维度
   :type dim: int
   :param index: 索引张量
   :type index: riemann.TN
   :return: 收集后的张量
   :rtype: riemann.TN

.. method:: riemann.TN.scatter(dim, index, src=None, *, value=None)

   将值按照索引填充到新张量中。

   :param dim: 沿着哪个维度进行索引
   :type dim: int
   :param index: 索引张量
   :type index: riemann.TN
   :param src: 源张量，提供要填充的值
   :type src: riemann.TN, optional
   :param value: 标量值，提供要填充的值
   :type value: int, float, complex, optional
   :return: 填充后的新张量
   :rtype: riemann.TN

.. method:: riemann.TN.scatter_(dim, index, src=None, *, value=None)

   原地将值按照索引填充到张量中。

   :param dim: 沿着哪个维度进行索引
   :type dim: int
   :param index: 索引张量
   :type index: riemann.TN
   :param src: 源张量，提供要填充的值
   :type src: riemann.TN, optional
   :param value: 标量值，提供要填充的值
   :type value: int, float, complex, optional
   :return: 原地修改后的张量
   :rtype: riemann.TN

.. method:: riemann.TN.scatter_add(dim, index, src)

   将值按照索引累加到新张量中。

   :param dim: 沿着哪个维度进行索引
   :type dim: int
   :param index: 索引张量
   :type index: riemann.TN
   :param src: 源张量，提供要累加的值
   :type src: riemann.TN
   :return: 累加后的新张量
   :rtype: riemann.TN

.. method:: riemann.TN.scatter_add_(dim, index, src)

   原地将值按照索引累加到张量中。

   :param dim: 沿着哪个维度进行索引
   :type dim: int
   :param index: 索引张量
   :type index: riemann.TN
   :param src: 源张量，提供要累加的值
   :type src: riemann.TN
   :return: 原地修改后的张量
   :rtype: riemann.TN

数据转换
~~~~~~~~
.. function:: riemann.from_numpy(arr, requires_grad=False)

   将 NumPy 数组转换为 Riemann 张量。

   :param arr: 输入 NumPy 数组
   :type arr: numpy.ndarray
   :param requires_grad: 是否跟踪此张量上的操作
   :type requires_grad: bool, optional
   :return: Riemann 张量
   :rtype: riemann.TN

.. function:: riemann.item(tensor)

   将单个元素的张量转换为 Python 标量。

   :param tensor: 输入张量
   :type tensor: riemann.TN
   :return: Python 标量
   :rtype: int, float, etc.

.. method:: riemann.TN.tolist()

   将张量转换为 Python 列表。

   :return: Python 列表或标量
   :rtype: list, int, float, complex

.. method:: riemann.TN.numpy()

   将张量转换为 NumPy 数组。

   :return: NumPy 数组
   :rtype: numpy.ndarray

.. method:: riemann.TN.to(*args, **kwargs)

   将张量转换为指定的数据类型和/或设备。

   :param dtype: 目标数据类型
   :type dtype: dtype, optional
   :param device: 目标设备
   :type device: str, Device, optional
   :return: 转换后的张量
   :rtype: riemann.TN

.. method:: riemann.TN.type(dtype=None)

   返回或转换张量的数据类型。

   :param dtype: 数据类型，如果为None则返回当前数据类型
   :type dtype: dtype, optional
   :return: 如果dtype为None返回当前数据类型，否则返回转换后的数据类型
   :rtype: dtype or riemann.TN

.. method:: riemann.TN.type_as(other)

   将张量转换为与另一个张量相同的数据类型。

   :param other: 目标数据类型的参考张量
   :type other: riemann.TN
   :return: 转换后的数据类型
   :rtype: riemann.TN

.. method:: riemann.TN.bool()

   将张量转换为布尔类型。

   :return: 布尔类型的张量
   :rtype: riemann.TN

.. method:: riemann.TN.float()

   将张量转换为单精度浮点类型(float32)。

   :return: float32类型的张量
   :rtype: riemann.TN

.. method:: riemann.TN.double()

   将张量转换为双精度浮点类型(float64)。

   :return: float64类型的张量
   :rtype: riemann.TN

副本函数
~~~~~~~~

.. function:: riemann.clone(tensor)

   返回张量的副本。

   :param tensor: 输入张量
   :type tensor: riemann.TN
   :return: 张量副本
   :rtype: riemann.TN

.. method:: riemann.TN.copy()

   返回张量的副本，不共享内存，也不依赖原张量。

   :return: 张量副本
   :rtype: riemann.TN

.. function:: riemann.detach(tensor)

   从计算图中分离张量，停止梯度跟踪。

   :param tensor: 输入张量
   :type tensor: riemann.TN
   :return: 分离后的张量
   :rtype: riemann.TN

数据类型
--------

.. module:: riemann.dtype

预定义数据类型
~~~~~~~~~~~~~~

.. data:: float16

   16位浮点数据类型

.. data:: float32

   32位浮点数据类型

.. data:: float64

   64位浮点数据类型

.. data:: complex64

   64位复数数据类型

.. data:: complex128

   128位复数数据类型

.. data:: half

   float16的别名

.. data:: float_

   float32的别名

.. data:: double

   float64的别名

.. data:: complex_

   complex64的别名

.. data:: int8

   8位有符号整数数据类型

.. data:: int16

   16位有符号整数数据类型

.. data:: int32

   32位有符号整数数据类型

.. data:: int64

   64位有符号整数数据类型

.. data:: uint8

   8位无符号整数数据类型

.. data:: uint16

   16位无符号整数数据类型

.. data:: uint32

   32位无符号整数数据类型

.. data:: uint64

   64位无符号整数数据类型

.. data:: short

   int16的别名

.. data:: int_

   int32的别名

.. data:: long

   int64的别名

.. data:: bool_

   布尔数据类型

默认数据类型函数
~~~~~~~~~~~~~~~~

.. function:: riemann.set_default_dtype(dtype)

   设置默认的浮点数据类型

   :param dtype: 要设置为默认的浮点数据类型
   :type dtype: numpy.dtype

.. function:: riemann.get_default_dtype()

   获取当前默认的浮点数据类型

   :return: 当前默认的浮点数据类型
   :rtype: numpy.dtype

.. function:: riemann.get_default_complex()

   根据默认的浮点类型推导默认的复数数据类型

   :return: 默认的复数数据类型
   :rtype: numpy.dtype

数据类型检查函数
~~~~~~~~~~~~~~~~

.. function:: riemann.is_numeric_array(numpy_arr)

   检查NumPy数组是否具有数值数据类型

   :param numpy_arr: 要检查的NumPy数组
   :type numpy_arr: numpy.ndarray
   :return: 数组是否具有数值数据类型
   :rtype: bool

.. function:: riemann.is_number(v)

   检查值是否为数值类型

   :param v: 要检查的值
   :type v: Any
   :return: 值是否为数值类型
   :rtype: bool

.. function:: riemann.is_float_or_complex(dtype)

   检查数据类型是否为浮点或复数类型

   :param dtype: 要检查的数据类型
   :type dtype: numpy.dtype
   :return: 数据类型是否为浮点或复数类型
   :rtype: bool

.. function:: riemann.is_scalar(value)

   检查值是否为标量（包括Riemann张量标量）

   :param value: 要检查的值
   :type value: Any
   :return: 值是否为标量
   :rtype: bool

数据类型推断
~~~~~~~~~~~~

.. function:: riemann.infer_data_type(v)

   从Python值、NumPy数组或值集合中推断适当的数据类型

   :param v: 要推断数据类型的值或值集合
   :type v: Any
   :return: 推断的数据类型
   :rtype: numpy.dtype

梯度模式控制
------------

.. module:: riemann.gradmode

.. function:: riemann.is_grad_enabled()

   获取当前线程的梯度计算状态

   :return: 当前梯度计算模式（True表示启用，False表示禁用）
   :rtype: bool

.. function:: riemann.no_grad(func=None)

   上下文管理器，用于暂时禁用梯度计算

   也可以作为函数装饰器使用，禁用被装饰函数内所有计算的梯度追踪。

   :param func: 可选，如果提供，则将no_grad作为装饰器应用于该函数
   :type func: callable, optional
   :return: 如果未提供func，则返回上下文管理器实例；如果提供了func，则返回装饰后的函数

   示例：
   
   .. code-block:: python
      
      # 作为上下文管理器使用
      with riemann.no_grad():
          # 这段代码中的计算不会追踪梯度
          output = model(input_data)
      
      # 作为装饰器使用
      @riemann.no_grad
      def inference(x):
          # 函数内的计算不会追踪梯度
          return model(x)

.. function:: riemann.enable_grad(func=None)

   上下文管理器，用于暂时启用梯度计算

   也可以作为函数装饰器使用，确保被装饰函数内的计算追踪梯度。

   :param func: 可选，如果提供，则将enable_grad作为装饰器应用于该函数
   :type func: callable, optional
   :return: 如果未提供func，则返回上下文管理器实例；如果提供了func，则返回装饰后的函数

   示例：
   
   .. code-block:: python
      
      # 作为上下文管理器使用
      with riemann.enable_grad():
          # 这段代码中的计算会追踪梯度
          output = model(input_data)
          loss = loss_fn(output, target)
          loss.backward()
      
      # 作为装饰器使用
      @riemann.enable_grad
      def train_step(x, y):
          # 函数内的计算会追踪梯度
          pred = model(x)
          loss = loss_fn(pred, y)
          loss.backward()
          return loss

.. function:: riemann.set_grad_enabled(mode=True, func=None)

   上下文管理器，用于设置梯度计算模式

   类似于PyTorch的set_grad_enabled()，可以显式地启用或禁用梯度计算。
   支持作为上下文管理器或装饰器使用，提供最灵活的梯度控制方式。

   :param mode: 如果为True，则启用梯度计算；如果为False，则禁用梯度计算
   :type mode: bool
   :param func: 可选，当作为装饰器使用时传入的函数
   :type func: callable, optional
   :return: 如果func为None，返回上下文管理器实例；如果提供了func参数，返回包装后的函数

   示例：
   
   .. code-block:: python
      
      # 作为上下文管理器使用
      with riemann.set_grad_enabled(False):
          # 这段代码中的计算不会追踪梯度
          output = model(input_data)
      
      with riemann.set_grad_enabled(True):
          # 这段代码中的计算会追踪梯度
          output = model(input_data)
          loss = loss_fn(output, target)
          loss.backward()
      
      # 作为装饰器使用
      @riemann.set_grad_enabled(False)
      def inference(x):
          return model(x)
      
      @riemann.set_grad_enabled(True)
      def train(x, y):
          pred = model(x)
          loss = loss_fn(pred, y)
          loss.backward()
          return loss

序列化
------

.. module:: riemann.serialization

.. function:: riemann.save(obj, f, pickle_module=None, pickle_protocol=2, use_new_zipfile_serialization=True)

   将对象保存到磁盘文件。

   此函数使用pickle序列化将Riemann张量、参数、模块或任何Python对象保存到磁盘文件。

   :param obj: 要保存的对象。可以是张量、参数、模块或任何可pickle的对象
   :type obj: Any
   :param f: 要写入的文件路径或类文件对象
   :type f: str, os.PathLike, 或类文件对象
   :param pickle_module: 用于pickle的模块（默认：pickle）
   :type pickle_module: Any, optional
   :param pickle_protocol: Pickle协议版本（默认：2）
   :type pickle_protocol: int, optional
   :param use_new_zipfile_serialization: 是否使用基于zip的序列化（默认：True）
   :type use_new_zipfile_serialization: bool, optional

   示例：
       >>> import riemann as rm
       >>> # 保存张量
       >>> tensor = rm.randn(3, 4)
       >>> rm.save(tensor, 'tensor.pt')
       >>> 
       >>> # 保存模块
       >>> model = rm.nn.Linear(10, 5)
       >>> rm.save(model.state_dict(), 'model_weights.pt')
       >>> 
       >>> # 保存多个对象
       >>> rm.save({
       ...     'model': model.state_dict(),
       ...     'optimizer_state': optimizer.state_dict(),
       ...     'epoch': 10
       ... }, 'checkpoint.pt')

.. function:: riemann.load(f, map_location=None, pickle_module=None, **pickle_load_args)

   从磁盘文件加载对象。

   此函数使用pickle反序列化从磁盘文件加载Riemann张量、参数、模块或任何Python对象。

   :param f: 要读取的文件路径或类文件对象
   :type f: str, os.PathLike, 或类文件对象
   :param map_location: 用于重新映射存储位置的函数或字典
   :type map_location: Any, optional
   :param pickle_module: 用于unpickle的模块（默认：pickle）
   :type pickle_module: Any, optional
   :param \**pickle_load_args: 传递给pickle.load的额外参数
   :return: 加载的对象

   示例：
       >>> import riemann as rm
       >>> # 加载张量
       >>> tensor = rm.load('tensor.pt')
       >>> 
       >>> # 加载模型权重
       >>> state_dict = rm.load('model_weights.pt')
       >>> model.load_state_dict(state_dict)
       >>> 
       >>> # 加载检查点
       >>> checkpoint = rm.load('checkpoint.pt')
       >>> model.load_state_dict(checkpoint['model'])
       >>> optimizer.load_state_dict(checkpoint['optimizer_state'])
       >>> epoch = checkpoint['epoch']

CUDA支持
--------

.. module:: riemann.cuda

.. class:: riemann.cuda.Device(device='cpu')

   表示一个设备（CPU或CUDA GPU）。

   :param device: 设备类型或索引。可以是：
       - 字符串：'cpu'、'cuda'或'cuda:0'、'cuda:1'
       - 整数：CUDA设备索引
   :type device: str 或 int

   示例：
       >>> import riemann as rm
       >>> # 创建CPU设备
       >>> cpu_device = rm.Device('cpu')
       >>> # 创建CUDA设备
       >>> cuda_device = rm.Device('cuda')
       >>> # 创建特定的CUDA设备
       >>> cuda_device_1 = rm.Device('cuda:1')
       >>> # 通过type和index参数创建CUDA设备
       >>> cuda_device_2 = rm.Device('cuda', 2)

   .. method:: __enter__()

      进入设备上下文。

   .. method:: __exit__(exc_type, exc_val, exc_tb)

      退出设备上下文。

   .. method:: __eq__(other)

      与另一个设备比较。

   .. method:: __str__()

      返回设备的字符串表示。

   .. method:: __repr__()

      返回设备的官方字符串表示。

.. function:: riemann.cuda.is_available()

   检查CUDA是否可用。

   :return: 如果CUDA可用返回True，否则返回False
   :rtype: bool

.. function:: riemann.cuda.device_count()

   返回可用的CUDA设备数量。

   :return: 可用的CUDA设备数量
   :rtype: int

.. function:: riemann.cuda.current_device()

   返回当前CUDA设备的索引。

   :return: 当前CUDA设备的索引
   :rtype: int

.. function:: riemann.cuda.get_device_name(device_idx)

   返回给定索引的CUDA设备名称。

   :param device_idx: CUDA设备的索引
   :type device_idx: int
   :return: CUDA设备的名称
   :rtype: str

.. function:: riemann.cuda.set_device(device_idx)

   设置当前CUDA设备。

   :param device_idx: 要设置为当前的CUDA设备索引
   :type device_idx: int

.. function:: riemann.cuda.empty_cache()

   清空CUDA缓存。

.. function:: riemann.cuda.synchronize(device=None)

   等待当前CUDA设备上的所有操作完成。

   :param device: 要同步的设备，默认为当前设备
   :type device: str, int, or Device, optional

.. function:: riemann.cuda.is_in_cuda_context()

   检查当前线程是否在CUDA设备上下文中。

   :return: 如果在CUDA设备上下文中返回True，否则返回False
   :rtype: bool

.. function:: riemann.memory_allocated(device_idx=None)

   返回给定CUDA设备上分配的内存量。

   :param device_idx: CUDA设备的索引。如果为None，使用当前设备
   :type device_idx: int, optional
   :return: 分配的内存量（字节）
   :rtype: int

.. function:: riemann.get_default_device()

   获取张量创建的默认设备。

   :return: 默认设备
   :rtype: Device

.. function:: riemann.set_default_device(device)

   设置张量创建的默认设备。

   :param device: 要设置为默认的设备。可以是：
       - 字符串：'cpu'、'cuda'或'cuda:0'、'cuda:1'
       - 整数：CUDA设备索引
       - Device对象
   :type device: str, int, 或 Device

   示例：
       >>> import riemann as rm
       >>> rm.get_default_device()
       device(type='cpu', index=None)
       >>> rm.set_default_device('cuda')
       >>> rm.get_default_device()
       device(type='cuda', index=0)
       >>> rm.set_default_device('cuda:1')
       >>> rm.get_default_device()
       device(type='cuda', index=1)

自动微分
--------

.. module:: riemann.autograd

梯度计算
~~~~~~~~

.. function:: riemann.autograd.backward(self, gradient=None, retain_graph=False, create_graph=False)

   执行反向模式自动微分（反向传播）。

   从当前张量开始，通过计算图向后传播梯度，为所有叶节点或设置了retains_grad=True的中间节点计算并存储梯度。

   :param self: 触发反向传播的张量
   :type self: riemann.TN
   :param gradient: 输出张量的梯度，默认为None
   :type gradient: riemann.TN or None, optional
   :param retain_graph: 此参数用于PyTorch兼容性，Riemann反向传播不依赖于此参数
   :type retain_graph: bool, optional
   :param create_graph: 是否在梯度计算过程中创建计算图，设置为True以进行高阶导数计算
   :type create_graph: bool, optional

.. function:: riemann.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=False, allow_unused=False)

   计算并返回输出相对于输入的梯度。

   这是Riemann框架中的核心梯度计算函数。与backward()方法类似，
   但它直接返回计算出的梯度张量，而不是将它们存储在输入张量的.grad属性中。
   这使其更适合于高级梯度计算场景，如计算雅可比矩阵、
   海森矩阵等。

   :param outputs: 要计算梯度的输出张量
   :type outputs: riemann.TN
   :param inputs: 要计算梯度的输入张量或输入张量列表/元组
   :type inputs: riemann.TN or list/tuple of riemann.TN
   :param grad_outputs: 输出张量的梯度，默认为None
   :type grad_outputs: riemann.TN or None, optional
   :param retain_graph: 此参数用于PyTorch兼容性，Riemann反向传播不依赖于此参数
   :type retain_graph: bool, optional
   :param create_graph: 是否在梯度计算过程中创建计算图
   :type create_graph: bool, optional
   :param allow_unused: 是否允许未使用的输入
   :type allow_unused: bool, optional
   :return: 对应于输入的梯度张量元组
   :rtype: tuple of riemann.TN

.. function:: riemann.autograd.higher_order_grad(outputs, inputs, n, create_graph=False)

   计算标量张量输出相对于inputs中每个张量的n阶导数。

   此函数通过递归调用grad()来计算高阶导数。对于每个输入张量，
   它计算n阶导数并返回与输入列表对应的导数元组。

   :param outputs: 要计算高阶导数的标量张量输出
   :type outputs: riemann.TN
   :param inputs: 要计算高阶导数的输入张量或输入张量列表/元组
   :type inputs: riemann.TN or list/tuple of riemann.TN
   :param n: 导数的阶数，必须是非负整数
   :type n: int
   :param create_graph: 是否在梯度计算过程中创建计算图
   :type create_graph: bool, optional
   :return: 对应于输入的n阶导数张量元组
   :rtype: tuple of riemann.TN

.. function:: riemann.autograd.gradcheck(func, inputs, eps=1e-6, atol=1e-5, rtol=1e-3, raise_exception=True, check_sparse_nnz=False, fast_mode=False)

   通过比较数值梯度和解析梯度来验证给定函数的梯度计算是否正确。

   此函数通过向输入参数添加小扰动来计算数值梯度（使用有限差分），
   并将其与使用自动微分计算的解析梯度进行比较。

   :param func: 要验证梯度的函数
   :type func: callable
   :param inputs: 测试用的输入张量元组
   :type inputs: tuple of riemann.TN
   :param eps: 数值梯度计算的小扰动
   :type eps: float, optional
   :param atol: 绝对误差容差
   :type atol: float, optional
   :param rtol: 相对误差容差
   :type rtol: float, optional
   :param raise_exception: 如果梯度检查失败是否抛出异常
   :type raise_exception: bool, optional
   :param check_sparse_nnz: 是否检查稀疏张量非零元素（当前版本不支持）
   :type check_sparse_nnz: bool, optional
   :param fast_mode: 是否使用快速模式（仅检查第一个元素）
   :type fast_mode: bool, optional
   :return: 如果梯度检查通过则为True，否则为False
   :rtype: bool

.. function:: riemann.track_grad(grad_func)

   创建一个梯度跟踪装饰器，用于为函数添加自动微分支持。

   这个装饰器工厂接收一个梯度函数，返回一个修饰器，该修饰器可以将普通的张量运算函数
   转换为支持自动微分的函数。它会自动创建反向传播函数，并管理梯度计算图的构建。

   :param grad_func: 梯度计算函数，接收与前向函数相同的输入参数，
                    返回一个元组，包含每个输入张量对应的梯度（偏导数）
                    元组内元素需要与前向函数的输入张量一一对应，对于不需要梯度的张量，对应的梯度值应为None
   :type grad_func: callable
   :return: 一个修饰器函数，用于包装前向计算函数，使其支持自动微分
   :rtype: callable

   示例：
   
   .. code-block:: python
      
      # 定义单输入导数函数（d/dx log(x) = 1/x）
      def _log_derivative(x):
          return (1. / x.conj(),)
      
      # 使用track_grad修饰器创建支持自动微分的对数函数
      @track_grad(_log_derivative)
      def mylog(x):
          return tensor(np.log(x.data))
      
      # 使用带自动微分的对数函数
      x = tensor(2., requires_grad=True)
      y = mylog(x)
      y.backward()
      print(f'x.grad = {x.grad}')  # 输出: x.grad = 0.5
      
      # 定义多输入导数函数（d/dx (x + y) = 1, d/dy (x + y) = 1）
      def _add_derivative(x, y):
          return (tensor(1.), tensor(1.))
      
      # 使用track_grad修饰器创建支持自动微分的加法函数
      @track_grad(_add_derivative)
      def myadd(x, y):
          return tensor(x.data + y.data)
      
      # 使用带自动微分的加法函数
      x = tensor(2., requires_grad=True)
      y = tensor(3., requires_grad=True)
      z = myadd(x, y)
      z.backward()
      print(f'x.grad = {x.grad}')  # 输出: x.grad = 1.0
      print(f'y.grad = {y.grad}')  # 输出: y.grad = 1.0

.. class:: riemann.autograd.Function

   Riemann框架中用于自定义梯度实现的基类，设计了与PyTorch的torch.autograd.Function类似的接口。

   要使用此类，继承它并实现forward和backward静态方法：
   - forward: 执行前向计算，返回输出张量
   - backward: 接收输出梯度，返回输入梯度

   示例：
   
   .. code-block:: python
      
      class MyFunction(Function):
          @staticmethod
          def forward(ctx, input1, input2):
              ctx.save_for_backward(input1, input2)
              output = input1 * input2
              return output
          
          @staticmethod
          def backward(ctx, grad_output):
              input1, input2 = ctx.saved_tensors
              grad_input1 = grad_output * input2
              grad_input2 = grad_output * input1
              return grad_input1, grad_input2

求导功能函数
~~~~~~~~~~~~

.. function:: riemann.autograd.functional.jacobian(func, inputs, create_graph=False, strict=True)

   计算函数的雅可比矩阵。

   该函数计算给定函数在输入点处的雅可比矩阵，支持单个或多个输入、
   单个或多个输出的情况，并与PyTorch的jacobian函数行为保持兼容。

   :param func: 要计算雅可比矩阵的函数
   :type func: callable
   :param inputs: 函数的输入张量或张量列表/元组
   :type inputs: riemann.TN or list/tuple of riemann.TN
   :param create_graph: 是否在梯度计算过程中创建计算图
   :type create_graph: bool, optional
   :param strict: 是否严格遵循PyTorch的行为规范
   :type strict: bool, optional
   :return: 对应于输入/输出类型的雅可比矩阵表示
   :rtype: riemann.TN or list/tuple of riemann.TN

.. function:: riemann.autograd.functional.hessian(func, inputs, create_graph=False, strict=True)

   计算函数的 Hessian 矩阵。

   该函数计算给定函数在输入点处的 Hessian 矩阵，即梯度的雅可比矩阵。
   它支持单个或多个输入的情况，并与PyTorch的hessian函数行为保持兼容。

   :param func: 要计算 Hessian 矩阵的标量值函数
   :type func: callable
   :param inputs: 函数的输入张量或张量列表/元组
   :type inputs: riemann.TN or list/tuple of riemann.TN
   :param create_graph: 是否在梯度计算过程中创建计算图
   :type create_graph: bool, optional
   :param strict: 如果为True，当检测到输出与输入无关时会引发错误
   :type strict: bool, optional
   :return: 对应于输入类型的 Hessian 矩阵表示
   :rtype: riemann.TN or list/tuple of riemann.TN

.. function:: riemann.autograd.functional.jvp(func, inputs, v=None, create_graph=False, strict=False)

   计算 Jacobian 向量乘积（Jacobian-Vector Product）。

   :param func: 要计算 JVP 的函数
   :type func: callable
   :param inputs: 函数的输入张量或张量列表/元组
   :type inputs: riemann.TN or list/tuple of riemann.TN
   :param v: 与 Jacobian 矩阵相乘的向量
   :type v: riemann.TN or list/tuple of riemann.TN, optional
   :param create_graph: 是否在梯度计算过程中创建计算图，用于高阶导数计算
   :type create_graph: bool, optional
   :param strict: 是否对未使用的输入引发错误
   :type strict: bool, optional
   :return: 函数输出和 JVP 值
   :rtype: tuple of (riemann.TN, riemann.TN or list/tuple of riemann.TN)

.. function:: riemann.autograd.functional.vjp(func, inputs, v=None, create_graph=False, strict=False)

   计算向量 Jacobian 乘积（Vector-Jacobian Product）。

   :param func: 要计算 VJP 的函数
   :type func: callable
   :param inputs: 函数的输入张量或张量列表/元组
   :type inputs: riemann.TN or list/tuple of riemann.TN
   :param v: 与 Jacobian 矩阵相乘的向量
   :type v: riemann.TN or list/tuple of riemann.TN, optional
   :param create_graph: 是否在梯度计算过程中创建计算图，用于高阶导数计算
   :type create_graph: bool, optional
   :param strict: 是否对未使用的输入引发错误
   :type strict: bool, optional
   :return: 函数输出和 VJP 值
   :rtype: tuple of (riemann.TN, riemann.TN or list/tuple of riemann.TN)

.. function:: riemann.autograd.functional.hvp(func, inputs, v, create_graph=False, strict=False)

   计算 Hessian 向量乘积（Hessian-Vector Product）。

   :param func: 要计算 HVP 的标量值函数
   :type func: callable
   :param inputs: 函数的输入张量或张量列表/元组
   :type inputs: riemann.TN or list/tuple of riemann.TN
   :param v: 与 Hessian 矩阵相乘的向量
   :type v: riemann.TN or list/tuple of riemann.TN
   :param create_graph: 是否在梯度计算过程中创建计算图，用于高阶导数计算
   :type create_graph: bool, optional
   :param strict: 是否对未使用的输入引发错误
   :type strict: bool, optional
   :return: 函数输出和 HVP 值
   :rtype: tuple of (riemann.TN, riemann.TN or list/tuple of riemann.TN)

.. function:: riemann.autograd.functional.vhp(func, inputs, v, create_graph=False, strict=False)

   计算向量 Hessian 乘积（Vector-Hessian Product）。

   :param func: 要计算 VHP 的标量值函数
   :type func: callable
   :param inputs: 函数的输入张量或张量列表/元组
   :type inputs: riemann.TN or list/tuple of riemann.TN
   :param v: 与 Hessian 矩阵相乘的向量
   :type v: riemann.TN or list/tuple of riemann.TN
   :param create_graph: 是否在梯度计算过程中创建计算图，用于高阶导数计算
   :type create_graph: bool, optional
   :param strict: 是否对未使用的输入引发错误
   :type strict: bool, optional
   :return: 函数输出和 VHP 值
   :rtype: tuple of (riemann.TN, riemann.TN or list/tuple of riemann.TN)

.. function:: riemann.autograd.functional.derivative(func, create_graph=False)

   计算函数的导数函数。

   该函数返回一个新函数，该新函数在调用时会计算原始函数func在输入点处的导数。
   支持func的输入为单个或多个张量，返回为单个或多个张量或标量。
   内部基于jacobian函数实现导数计算。

   :param func: 要求导的函数
   :type func: callable
   :param create_graph: 是否在梯度计算中创建计算图，默认为False
   :type create_graph: bool, optional
   :return: 导函数，该函数接受与原函数相同的输入
   :rtype: callable

上下文管理器
------------

.. function:: riemann.no_grad()

   上下文管理器，用于禁用梯度计算。在这个上下文中的操作不会被记录在计算图中。

.. function:: riemann.enable_grad()

   上下文管理器，用于启用梯度计算。

.. function:: riemann.set_grad_enabled(mode)

   上下文管理器，根据 mode 参数启用或禁用梯度计算。

   :param mode: 启用梯度计算为 True，禁用为 False
   :type mode: bool

线性代数模块
------------

``riemann.linalg`` 模块提供了各种线性代数运算，包括矩阵乘法、分解、求解等。

矩阵运算
~~~~~~~~

.. function:: riemann.linalg.matmul(a, b)

   计算两个张量的矩阵乘积。

   :param a: 第一个输入张量
   :type a: riemann.TN
   :param b: 第二个输入张量
   :type b: riemann.TN
   :return: 矩阵乘积结果
   :rtype: riemann.TN

.. function:: riemann.linalg.cross(a, b, dim=-1)

   计算两个张量的叉积（向量积）。

   :param a: 第一个输入张量
   :type a: riemann.TN
   :param b: 第二个输入张量
   :type b: riemann.TN
   :param dim: 计算叉积的维度，默认为-1
   :type dim: int, optional
   :return: 叉积结果
   :rtype: riemann.TN

范数计算
~~~~~~~~

.. function:: riemann.linalg.norm(A, ord=None, dim=None, keepdim=False)

   计算张量或矩阵的范数。

   :param A: 输入张量
   :type A: riemann.TN
   :param ord: 范数的阶，默认为Frobenius范数
   :type ord: int or float or str, optional
   :param dim: 计算范数的维度，默认为None（计算所有元素的范数）
   :type dim: int or tuple, optional
   :param keepdim: 是否保留输出的维度
   :type keepdim: bool, optional
   :return: 范数值
   :rtype: riemann.TN

.. function:: riemann.linalg.vector_norm(x, ord=2, dim=None, keepdim=False)

   计算向量范数。

   :param x: 输入张量
   :type x: riemann.TN
   :param ord: 范数的阶，默认为2（L2范数）
   :type ord: float, optional
   :param dim: 计算范数的维度
   :type dim: int or tuple, optional
   :param keepdim: 是否保留输出的维度
   :type keepdim: bool, optional
   :return: 范数值
   :rtype: riemann.TN

.. function:: riemann.linalg.matrix_norm(A, ord='fro', dim=(-2, -1), keepdim=False)

   计算矩阵范数。

   :param A: 输入张量
   :type A: riemann.TN
   :param ord: 范数的阶，默认为'fro'（Frobenius范数）
   :type ord: str or int, optional
   :param dim: 矩阵所在的维度，默认为(-2, -1)
   :type dim: tuple, optional
   :param keepdim: 是否保留输出的维度
   :type keepdim: bool, optional
   :return: 矩阵范数值
   :rtype: riemann.TN

.. function:: riemann.linalg.cond(A, p=None)

   计算矩阵的条件数。

   :param A: 输入矩阵
   :type A: riemann.TN
   :param p: 范数类型，默认为None（2-范数条件数）
   :type p: int or float or str, optional
   :return: 条件数
   :rtype: riemann.TN

.. function:: riemann.linalg.svdvals(A)

   计算矩阵的奇异值。

   :param A: 输入矩阵
   :type A: riemann.TN
   :return: 奇异值
   :rtype: riemann.TN

矩阵分解
~~~~~~~~

.. function:: riemann.linalg.det(A)

   计算矩阵的行列式。

   :param A: 输入矩阵
   :type A: riemann.TN
   :return: 行列式值
   :rtype: riemann.TN

.. function:: riemann.linalg.inv(A)

   计算方阵的逆矩阵。

   :param A: 输入方阵
   :type A: riemann.TN
   :return: 逆矩阵
   :rtype: riemann.TN

.. function:: riemann.linalg.skew(A)

   计算矩阵的斜对称部分。

   :param A: 输入矩阵
   :type A: riemann.TN
   :return: 斜对称矩阵
   :rtype: riemann.TN

.. function:: riemann.linalg.svd(A, full_matrices=True)

   计算矩阵的奇异值分解（SVD）。

   :param A: 输入矩阵
   :type A: riemann.TN
   :param full_matrices: 是否返回完整的U和Vh矩阵
   :type full_matrices: bool, optional
   :return: (U, S, Vh)元组
   :rtype: tuple

.. function:: riemann.linalg.pinv(A, rcond=1e-15)

   计算矩阵的Moore-Penrose伪逆。

   :param A: 输入矩阵
   :type A: riemann.TN
   :param rcond: 奇异值阈值
   :type rcond: float, optional
   :return: 伪逆矩阵
   :rtype: riemann.TN

特征值分解
~~~~~~~~~~~~

.. function:: riemann.linalg.eig(A)

   计算方阵的特征值和特征向量。

   :param A: 输入方阵
   :type A: riemann.TN
   :return: (特征值, 特征向量)元组
   :rtype: tuple

.. function:: riemann.linalg.eigh(A, UPLO='L')

   计算厄米特矩阵（或实对称矩阵）的特征值和特征向量。

   :param A: 输入厄米特矩阵
   :type A: riemann.TN
   :param UPLO: 指定使用上三角('U')还是下三角('L')部分
   :type UPLO: str, optional
   :return: (特征值, 特征向量)元组
   :rtype: tuple

线性方程组求解
~~~~~~~~~~~~~~~~

.. function:: riemann.linalg.lstsq(A, b, rcond=None)

   计算最小二乘解。

   :param A: 系数矩阵
   :type A: riemann.TN
   :param b: 右侧向量或矩阵
   :type b: riemann.TN
   :param rcond: 奇异值阈值
   :type rcond: float, optional
   :return: 最小二乘解
   :rtype: riemann.TN

.. function:: riemann.linalg.lu(A, pivot=True)

   计算矩阵的LU分解。

   :param A: 输入矩阵
   :type A: riemann.TN
   :param pivot: 是否进行主元选取
   :type pivot: bool, optional
   :return: (P, L, U)元组
   :rtype: tuple

.. function:: riemann.linalg.solve(A, b)

   求解线性方程组 Ax = b。

   :param A: 系数矩阵
   :type A: riemann.TN
   :param b: 右侧向量或矩阵
   :type b: riemann.TN
   :return: 解向量或矩阵
   :rtype: riemann.TN

.. function:: riemann.linalg.qr(A, mode='reduced')

   计算矩阵的QR分解。

   :param A: 输入矩阵
   :type A: riemann.TN
   :param mode: 分解模式，'reduced'或'complete'
   :type mode: str, optional
   :return: (Q, R)元组
   :rtype: tuple

.. function:: riemann.linalg.cholesky(A, upper=False)

   计算正定矩阵的Cholesky分解。

   :param A: 输入正定矩阵
   :type A: riemann.TN
   :param upper: 是否返回上三角矩阵，默认为False（下三角）
   :type upper: bool, optional
   :return: Cholesky因子
   :rtype: riemann.TN

神经网络模块
------------

基础类
~~~~~~

.. class:: riemann.nn.Module()

   神经网络模块的基础类。

   .. method:: __init__()

      初始化模块。

   .. method:: forward(*args, **kwargs)

      定义前向传播。

      :param args: 输入参数
      :param kwargs: 关键字参数
      :return: 前向传播的输出

   .. method:: parameters()

      获取所有可训练参数。

      :return: 参数列表
      :rtype: list of riemann.TN

.. class:: riemann.nn.Parameter(data=None, requires_grad=True)

   可训练参数类，用于存储模型参数。

   :param data: 参数数据
   :type data: array_like, optional
   :param requires_grad: 是否跟踪梯度
   :type requires_grad: bool, optional

容器模块
~~~~~~~~

.. class:: riemann.nn.Sequential(*modules)

   按顺序应用多个模块的容器。

   :param modules: 模块列表
   :type modules: list of riemann.Module

.. class:: riemann.nn.ModuleList(modules=None)

   用于存储模块列表的容器类。

   该容器允许以列表形式存储多个模块，并提供便捷的访问和迭代方法。所有子模块都会被正确注册，以便在参数列表中出现。

   :param modules: 用于初始化的模块列表
   :type modules: list of riemann.Module, 可选

.. class:: riemann.nn.ModuleDict(modules=None)

   用于存储模块字典的容器类。

   该容器允许使用字符串键存储模块，并提供类似字典的访问方法。所有子模块都会被正确注册。

   :param modules: 用于初始化的模块字典
   :type modules: dict of {str: riemann.Module}, 可选

.. class:: riemann.nn.ParameterList(parameters=None)

   用于存储参数列表的容器类。

   该容器允许以列表形式存储多个参数。所有参数都会被正确注册。

   :param parameters: 用于初始化的参数列表
   :type parameters: list of riemann.Parameter, 可选

.. class:: riemann.nn.ParameterDict(parameters=None)

   用于存储参数字典的容器类。

   该容器允许使用字符串键存储参数，并提供类似字典的访问方法。所有参数都会被正确注册。

   :param parameters: 用于初始化的参数字典
   :type parameters: dict of {str: riemann.Parameter}, 可选

线性层
~~~~~~

.. class:: riemann.nn.Linear(in_features, out_features, bias=True)

   全连接线性层。

   :param in_features: 输入特征数量
   :type in_features: int
   :param out_features: 输出特征数量
   :type out_features: int
   :param bias: 是否包含偏置项
   :type bias: bool, optional

卷积层
~~~~~~

.. class:: riemann.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

   一维卷积层。

   对一维输入应用卷积运算，提取特征并生成新的特征图。

   :param in_channels: 输入的通道数
   :type in_channels: int
   :param out_channels: 输出的通道数
   :type out_channels: int
   :param kernel_size: 卷积核大小
   :type kernel_size: int 或 tuple
   :param stride: 步长
   :type stride: int 或 tuple, 可选
   :param padding: 输入四周的零填充
   :type padding: int 或 tuple, 可选
   :param dilation: 核元素之间的间隔
   :type dilation: int 或 tuple, 可选
   :param groups: 输入通道与输出通道之间的分组
   :type groups: int, 可选
   :param bias: 是否使用偏置项
   :type bias: bool, 可选
   :param padding_mode: 填充模式
   :type padding_mode: str, 可选

.. class:: riemann.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

   二维卷积层。

   对二维输入应用卷积运算，提取图像特征并生成新的特征图。

   :param in_channels: 输入的通道数
   :type in_channels: int
   :param out_channels: 输出的通道数
   :type out_channels: int
   :param kernel_size: 卷积核大小
   :type kernel_size: int 或 tuple
   :param stride: 步长
   :type stride: int 或 tuple, 可选
   :param padding: 输入四周的零填充
   :type padding: int 或 tuple, 可选
   :param dilation: 核元素之间的间隔
   :type dilation: int 或 tuple, 可选
   :param groups: 输入通道与输出通道之间的分组
   :type groups: int, 可选
   :param bias: 是否使用偏置项
   :type bias: bool, 可选
   :param padding_mode: 填充模式
   :type padding_mode: str, 可选

.. class:: riemann.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

   三维卷积层。

   对三维输入应用卷积运算，常用于视频、体积数据等的特征提取。

   :param in_channels: 输入的通道数
   :type in_channels: int
   :param out_channels: 输出的通道数
   :type out_channels: int
   :param kernel_size: 卷积核大小
   :type kernel_size: int 或 tuple
   :param stride: 步长
   :type stride: int 或 tuple, 可选
   :param padding: 输入四周的零填充
   :type padding: int 或 tuple, 可选
   :param dilation: 核元素之间的间隔
   :type dilation: int 或 tuple, 可选
   :param padding_mode: 填充模式
   :type padding_mode: str, 可选

池化层
~~~~~~

.. class:: riemann.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

   一维最大池化层。

   对一维输入应用最大池化，用于提取序列数据的关键特征并减少数据维度。

   :param kernel_size: 池化窗口大小
   :type kernel_size: int 或 tuple
   :param stride: 池化窗口移动步长
   :type stride: int 或 tuple, 可选
   :param padding: 输入四周的零填充
   :type padding: int 或 tuple, 可选
   :param dilation: 池化窗口元素之间的间隔
   :type dilation: int 或 tuple, 可选
   :param return_indices: 是否返回最大值的索引
   :type return_indices: bool, 可选
   :param ceil_mode: 是否使用向上取整来计算输出形状
   :type ceil_mode: bool, 可选

.. class:: riemann.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

   二维最大池化层。

   对二维输入应用最大池化，常用于图像数据的特征提取和维度减少。

   :param kernel_size: 池化窗口大小
   :type kernel_size: int 或 tuple
   :param stride: 池化窗口移动步长
   :type stride: int 或 tuple, 可选
   :param padding: 输入四周的零填充
   :type padding: int 或 tuple, 可选
   :param dilation: 池化窗口元素之间的间隔
   :type dilation: int 或 tuple, 可选
   :param return_indices: 是否返回最大值的索引
   :type return_indices: bool, 可选
   :param ceil_mode: 是否使用向上取整来计算输出形状
   :type ceil_mode: bool, 可选

.. class:: riemann.nn.MaxPool3d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

   三维最大池化层。

   对三维输入应用最大池化，常用于视频、体积数据等的特征提取。

   :param kernel_size: 池化窗口大小
   :type kernel_size: int 或 tuple
   :param stride: 池化窗口移动步长
   :type stride: int 或 tuple, 可选
   :param padding: 输入四周的零填充
   :type padding: int 或 tuple, 可选
   :param dilation: 池化窗口元素之间的间隔
   :type dilation: int 或 tuple, 可选
   :param return_indices: 是否返回最大值的索引
   :type return_indices: bool, 可选
   :param ceil_mode: 是否使用向上取整来计算输出形状
   :type ceil_mode: bool, 可选

.. class:: riemann.nn.AvgPool1d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)

   一维平均池化层。

   对一维输入应用平均池化，用于平滑序列数据特征并减少数据维度。

   :param kernel_size: 池化窗口大小
   :type kernel_size: int 或 tuple
   :param stride: 池化窗口移动步长
   :type stride: int 或 tuple, 可选
   :param padding: 输入四周的零填充
   :type padding: int 或 tuple, 可选
   :param ceil_mode: 是否使用向上取整来计算输出形状
   :type ceil_mode: bool, 可选
   :param count_include_pad: 计算平均值时是否包含零填充
   :type count_include_pad: bool, 可选
   :param divisor_override: 如果指定，将使用该值作为分母
   :type divisor_override: int, 可选

.. class:: riemann.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)

   二维平均池化层。

   对二维输入应用平均池化，常用于图像数据的特征平滑和维度减少。

   :param kernel_size: 池化窗口大小
   :type kernel_size: int 或 tuple
   :param stride: 池化窗口移动步长
   :type stride: int 或 tuple, 可选
   :param padding: 输入四周的零填充
   :type padding: int 或 tuple, 可选
   :param ceil_mode: 是否使用向上取整来计算输出形状
   :type ceil_mode: bool, 可选
   :param count_include_pad: 计算平均值时是否包含零填充
   :type count_include_pad: bool, 可选
   :param divisor_override: 如果指定，将使用该值作为分母
   :type divisor_override: int, 可选

.. class:: riemann.nn.AvgPool3d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)

   三维平均池化层。

   对三维输入应用平均池化，常用于视频、体积数据等的特征平滑和维度减少。

   :param kernel_size: 池化窗口大小
   :type kernel_size: int 或 tuple
   :param stride: 池化窗口移动步长
   :type stride: int 或 tuple, 可选
   :param padding: 输入四周的零填充
   :type padding: int 或 tuple, 可选
   :param ceil_mode: 是否使用向上取整来计算输出形状
   :type ceil_mode: bool, 可选
   :param count_include_pad: 计算平均值时是否包含零填充
   :type count_include_pad: bool, 可选
   :param divisor_override: 如果指定，将使用该值作为分母
   :type divisor_override: int, 可选

自适应池化层
~~~~~~~~~~~~

.. class:: riemann.nn.AdaptiveAvgPool1d(output_size)

   一维自适应平均池化层。

   根据指定的输出尺寸自动计算池化核大小和步长，确保输出尺寸始终为指定值。

   :param output_size: 输出序列长度，可以是整数或 None（表示保持原尺寸）
   :type output_size: int 或 tuple

.. class:: riemann.nn.AdaptiveAvgPool2d(output_size)

   二维自适应平均池化层。

   根据指定的输出尺寸自动计算池化核大小和步长，常用于将任意尺寸的特征图转换为固定尺寸。

   :param output_size: 输出尺寸，可以是整数或二元组 (H, W)，或 None（表示保持原尺寸）
   :type output_size: int 或 tuple

.. class:: riemann.nn.AdaptiveAvgPool3d(output_size)

   三维自适应平均池化层。

   根据指定的输出尺寸自动计算池化核大小和步长，常用于体积数据的特征提取。

   :param output_size: 输出尺寸，可以是整数或三元组 (D, H, W)，或 None（表示保持原尺寸）
   :type output_size: int 或 tuple

.. class:: riemann.nn.AdaptiveMaxPool1d(output_size, return_indices=False)

   一维自适应最大池化层。

   根据指定的输出尺寸自动计算池化核大小和步长，确保输出尺寸始终为指定值。

   :param output_size: 输出序列长度，可以是整数或 None（表示保持原尺寸）
   :type output_size: int 或 tuple
   :param return_indices: 是否返回最大值的索引
   :type return_indices: bool, 可选

.. class:: riemann.nn.AdaptiveMaxPool2d(output_size, return_indices=False)

   二维自适应最大池化层。

   根据指定的输出尺寸自动计算池化核大小和步长，常用于将任意尺寸的特征图转换为固定尺寸。

   :param output_size: 输出尺寸，可以是整数或二元组 (H, W)，或 None（表示保持原尺寸）
   :type output_size: int 或 tuple
   :param return_indices: 是否返回最大值的索引
   :type return_indices: bool, 可选

.. class:: riemann.nn.AdaptiveMaxPool3d(output_size, return_indices=False)

   三维自适应最大池化层。

   根据指定的输出尺寸自动计算池化核大小和步长，常用于体积数据的特征提取。

   :param output_size: 输出尺寸，可以是整数或三元组 (D, H, W)，或 None（表示保持原尺寸）
   :type output_size: int 或 tuple
   :param return_indices: 是否返回最大值的索引
   :type return_indices: bool, 可选

归一化层
~~~~~~~~

.. class:: riemann.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

   一维批量归一化层。

   对二维或三维输入张量的通道维度进行归一化，使其具有零均值和单位方差，提升训练收敛性和模型泛化能力。

   :param num_features: 特征数量（通道维度）
   :type num_features: int
   :param eps: 避免除零的微小值
   :type eps: float, 可选
   :param momentum: 运行统计的动量
   :type momentum: float, 可选
   :param affine: 是否包含可学习的仿射参数
   :type affine: bool, 可选
   :param track_running_stats: 是否跟踪运行均值和方差
   :type track_running_stats: bool, 可选

.. class:: riemann.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1)

   二维批量归一化层。

   :param num_features: 特征数量
   :type num_features: int
   :param eps: 避免除零的微小值
   :type eps: float, optional
   :param momentum: 运行统计的动量
   :type momentum: float, optional

.. class:: riemann.nn.BatchNorm3d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

   三维批量归一化层。

   对五维输入张量 (N, C, D, H, W) 的通道维度进行归一化，使其具有零均值和单位方差，提升训练收敛性和模型泛化能力。

   :param num_features: 特征数量（通道维度）
   :type num_features: int
   :param eps: 避免除零的微小值
   :type eps: float, 可选
   :param momentum: 运行统计的动量
   :type momentum: float, 可选
   :param affine: 是否包含可学习的仿射参数
   :type affine: bool, 可选
   :param track_running_stats: 是否跟踪运行均值和方差
   :type track_running_stats: bool, 可选

.. class:: riemann.nn.LayerNorm(normalized_shape, eps=1e-05, affine=True, device=None, dtype=None)

   层归一化层，对指定维度进行归一化处理。

   与 torch.nn.LayerNorm 兼容，对输入张量的指定维度进行归一化，使其具有零均值和单位方差。

   :param normalized_shape: 整数或元组，指定需要归一化的维度
   :type normalized_shape: int 或 tuple
   :param eps: 为避免除零而添加到方差的小值
   :type eps: float, optional
   :param affine: 是否包含可学习的仿射参数（gamma 和 beta）
   :type affine: bool, optional
   :param device: 参数和缓冲区的设备
   :type device: optional
   :param dtype: 参数和缓冲区的数据类型
   :type dtype: optional

.. class:: riemann.nn.Flatten(start_dim=1, end_dim=-1)

   展平张量维度的层，移除从 start_dim 到 end_dim 的所有维度。

   通常用于卷积层之后、全连接层之前，将多维度的卷积结果展平为一维向量。

   :param start_dim: 开始展平的维度
   :type start_dim: int, 可选
   :param end_dim: 结束展平的维度
   :type end_dim: int, 可选

激活函数模块
~~~~~~~~~~~~

.. class:: riemann.nn.ReLU(inplace=False)

   ReLU 激活函数

   应用整流线性单元函数逐元素处理输入，ReLU(x) = max(0, x)

   :param inplace: 是否就地操作
   :type inplace: bool, 可选

.. class:: riemann.nn.LeakyReLU(negative_slope=0.01, inplace=False)

   Leaky ReLU 激活函数

   应用带泄漏的整流线性单元函数逐元素处理输入，LeakyReLU(x) = max(x, negative_slope * x)

   :param negative_slope: 负数区域的斜率
   :type negative_slope: float, 可选
   :param inplace: 是否就地操作
   :type inplace: bool, 可选

.. class:: riemann.nn.RReLU(lower=0.125, upper=0.3333333333333333, inplace=False)

   随机化 Leaky ReLU 激活函数

   应用随机化带泄漏的整流线性单元函数逐元素处理输入

   :param lower: 均匀分布的下限
   :type lower: float, 可选
   :param upper: 均匀分布的上限
   :type upper: float, 可选
   :param inplace: 是否就地操作
   :type inplace: bool, 可选

.. class:: riemann.nn.PReLU(num_parameters=1, init=0.25)

   参数化 ReLU 激活函数

   应用参数化的整流线性单元函数逐元素处理输入，a 是可学习的参数

   :param num_parameters: 可学习参数的数量
   :type num_parameters: int, 可选
   :param init: 参数的初始值
   :type init: float, 可选

.. class:: riemann.nn.Sigmoid()

   Sigmoid 激活函数

   应用 Sigmoid 函数逐元素处理输入，将值映射到 [0, 1] 区间

.. class:: riemann.nn.Tanh()

   Tanh 激活函数

   应用双曲正切函数逐元素处理输入，将值映射到 [-1, 1] 区间

.. class:: riemann.nn.Softmax(dim=None)

   Softmax 激活函数

   在指定维度上应用 Softmax 函数

   :param dim: 应用 softmax 的维度
   :type dim: int, 可选

.. class:: riemann.nn.LogSoftmax(dim=None)

   Log-Softmax 激活函数

   在指定维度上应用 Log-Softmax 函数

   :param dim: 应用 log-softmax 的维度
   :type dim: int, 可选

.. class:: riemann.nn.GELU()

   高斯误差线性单元激活函数

   应用高斯误差线性单元函数逐元素处理输入，GELU(x) = x * Φ(x)，其中 Φ 是标准正态分布的累积分布函数

.. class:: riemann.nn.Softplus(beta=1, threshold=20)

   Softplus 激活函数

   应用 Softplus 激活函数逐元素处理输入，Softplus(x) = (1 / beta) * log(1 + exp(beta * x))

   :param beta: 线性部分的斜率
   :type beta: float, 可选
   :param threshold: 数值稳定性的阈值
   :type threshold: float, 可选

Dropout 层
~~~~~~~~~~

.. class:: riemann.nn.Dropout(p=0.5)

   Dropout 层，用于防止过拟合。

   :param p: 丢弃概率
   :type p: float, optional

.. class:: riemann.nn.Dropout2d(p=0.5, inplace=False)

   2D dropout层，用于防止过拟合。

   在训练期间，以概率p随机将输入张量的整个通道置为0，并对剩余通道乘以1/(1-p)进行缩放。
   在评估期间，不执行任何操作。

   :param p: 丢弃概率
   :type p: float, optional
   :param inplace: 是否原地操作
   :type inplace: bool, optional

.. class:: riemann.nn.Dropout3d(p=0.5, inplace=False)

   3D dropout层，用于防止过拟合。

   在训练期间，以概率p随机将输入张量的整个通道置为0，并对剩余通道乘以1/(1-p)进行缩放。
   在评估期间，不执行任何操作。

   :param p: 丢弃概率
   :type p: float, optional
   :param inplace: 是否原地操作
   :type inplace: bool, optional

嵌入层
~~~~~~

.. class:: riemann.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, dtype=None, device=None)

   嵌入层，将整数索引转换为密集向量。

   嵌入层是神经网络中处理分类特征和序列数据的基础组件。

   :param num_embeddings: 嵌入向量的数量，即词典大小
   :type num_embeddings: int
   :param embedding_dim: 每个嵌入向量的维度
   :type embedding_dim: int
   :param padding_idx: 如果指定，该索引的嵌入向量不参与梯度计算，且在训练过程中保持不变
   :type padding_idx: int, 可选
   :param max_norm: 如果指定，所有嵌入向量的范数超过max_norm时，将被重归一化到max_norm
   :type max_norm: float, 可选
   :param norm_type: 计算范数时使用的p值，默认为2（L2范数）
   :type norm_type: float, 可选
   :param scale_grad_by_freq: 如果为True，梯度将按mini-batch中每个词的频率进行缩放
   :type scale_grad_by_freq: bool, 可选
   :param sparse: 如果为True，权重的梯度将是稀疏张量
   :type sparse: bool, 可选
   :param dtype: 嵌入权重的数据类型
   :type dtype: np.dtype, 可选
   :param device: 嵌入权重的设备
   :type device: str|int|Device, 可选

损失函数模块
~~~~~~~~~~~~

.. class:: riemann.nn.L1Loss(size_average=None, reduce=None, reduction='mean')

   平均绝对误差损失，计算输入与目标值之间的绝对值误差。

   :param size_average: 已弃用
   :type size_average: bool, optional
   :param reduce: 已弃用
   :type reduce: bool, optional
   :param reduction: 指定输出的聚合方式
   :type reduction: str, optional

.. class:: riemann.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

   均方误差损失，计算输入与目标值之间的平方误差。

   :param size_average: 已弃用
   :type size_average: bool, optional
   :param reduce: 已弃用
   :type reduce: bool, optional
   :param reduction: 指定输出的聚合方式
   :type reduction: str, optional

.. class:: riemann.nn.NLLLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

   负对数似然损失，用于分类任务中的概率预测。

   :param weight: 每个类别的手动缩放权重
   :type weight: riemann.TN, optional
   :param size_average: 已弃用
   :type size_average: bool, optional
   :param ignore_index: 指定要忽略的目标值
   :type ignore_index: int, optional
   :param reduce: 已弃用
   :type reduce: bool, optional
   :param reduction: 指定输出的聚合方式
   :type reduction: str, optional

.. class:: riemann.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

   交叉熵损失，结合了 LogSoftmax 和 NLLLoss 于一个类中，常用于多分类任务。

   :param weight: 每个类别的手动缩放权重
   :type weight: riemann.TN, optional
   :param size_average: 已弃用
   :type size_average: bool, optional
   :param ignore_index: 指定要忽略的目标值
   :type ignore_index: int, optional
   :param reduce: 已弃用
   :type reduce: bool, optional
   :param reduction: 指定输出的聚合方式
   :type reduction: str, optional

.. class:: riemann.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')

   二元交叉熵损失，计算目标与输出之间的二元分类误差。

   :param weight: 每个批次元素的手动缩放权重
   :type weight: riemann.TN, optional
   :param size_average: 已弃用
   :type size_average: bool, optional
   :param reduce: 已弃用
   :type reduce: bool, optional
   :param reduction: 指定输出的聚合方式
   :type reduction: str, optional

.. class:: riemann.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean')

   带logits的二元交叉熵损失，直接对输入logits计算二元交叉熵。

   :param weight: 每个批次元素的手动缩放权重
   :type weight: riemann.TN, optional
   :param size_average: 已弃用
   :type size_average: bool, optional
   :param reduce: 已弃用
   :type reduce: bool, optional
   :param reduction: 指定输出的聚合方式
   :type reduction: str, optional

.. class:: riemann.nn.HuberLoss(delta=1.0, size_average=None, reduce=None, reduction='mean')

   Huber损失函数，当误差小于delta时使用平方误差，否则使用线性误差。

   :param delta: 损失函数从二次变为线性的阈值
   :type delta: float, optional
   :param size_average: 已弃用
   :type size_average: bool, optional
   :param reduce: 已弃用
   :type reduce: bool, optional
   :param reduction: 指定输出的聚合方式
   :type reduction: str, optional

.. class:: riemann.nn.SmoothL1Loss(beta=1.0, size_average=None, reduce=None, reduction='mean')

   平滑L1损失，结合了L1损失和L2损失的优点，在小误差时使用二次损失，大误差时使用线性损失。

   :param beta: 控制从二次损失到线性损失的转换阈值
   :type beta: float, optional
   :param size_average: 已弃用
   :type size_average: bool, optional
   :param reduce: 已弃用
   :type reduce: bool, optional
   :param reduction: 指定输出的聚合方式
   :type reduction: str, optional

Transformer模块
~~~~~~~~~~~~~~~

.. class:: riemann.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None)

   多头注意力机制，允许模型同时关注来自不同表示子空间的信息。

   :param embed_dim: 输入和输出向量的维度，必须能被num_heads整除
   :type embed_dim: int
   :param num_heads: 注意力头的数量
   :type num_heads: int
   :param dropout: 注意力权重的dropout概率
   :type dropout: float, 可选
   :param bias: 是否在投影层添加偏置
   :type bias: bool, 可选
   :param add_bias_kv: 是否在key和value序列末尾添加可学习的偏置
   :type add_bias_kv: bool, 可选
   :param add_zero_attn: 是否在注意力权重中添加一列零
   :type add_zero_attn: bool, 可选
   :param kdim: key向量的维度，默认为embed_dim
   :type kdim: int, 可选
   :param vdim: value向量的维度，默认为embed_dim
   :type vdim: int, 可选
   :param batch_first: 输入输出形状是否为(batch, seq, feature)而非(seq, batch, feature)
   :type batch_first: bool, 可选
   :param device: 张量设备
   :type device: 可选
   :param dtype: 张量数据类型
   :type dtype: 可选

.. class:: riemann.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-05, batch_first=False, norm_first=False, bias=True, device=None, dtype=None)

   Transformer编码器的单层，由自注意力机制和前馈网络组成。

   :param d_model: 输入和输出特征的维度
   :type d_model: int
   :param nhead: 注意力头的数量
   :type nhead: int
   :param dim_feedforward: 前馈网络隐藏层的维度
   :type dim_feedforward: int, 可选
   :param dropout: dropout概率
   :type dropout: float, 可选
   :param activation: 激活函数类型，'relu'或'gelu'
   :type activation: str, 可选
   :param layer_norm_eps: 层归一化的epsilon值
   :type layer_norm_eps: float, 可选
   :param batch_first: 输入输出形状是否为(batch, seq, feature)
   :type batch_first: bool, 可选
   :param norm_first: 是否使用Pre-LN模式
   :type norm_first: bool, 可选
   :param bias: 是否在线性层添加偏置
   :type bias: bool, 可选
   :param device: 张量设备
   :type device: 可选
   :param dtype: 张量数据类型
   :type dtype: 可选

.. class:: riemann.nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-05, batch_first=False, norm_first=False, bias=True, device=None, dtype=None)

   Transformer解码器的单层，由自注意力、交叉注意力和前馈网络组成。

   :param d_model: 输入和输出特征的维度
   :type d_model: int
   :param nhead: 注意力头的数量
   :type nhead: int
   :param dim_feedforward: 前馈网络隐藏层的维度
   :type dim_feedforward: int, 可选
   :param dropout: dropout概率
   :type dropout: float, 可选
   :param activation: 激活函数类型，'relu'或'gelu'
   :type activation: str, 可选
   :param layer_norm_eps: 层归一化的epsilon值
   :type layer_norm_eps: float, 可选
   :param batch_first: 输入输出形状是否为(batch, seq, feature)
   :type batch_first: bool, 可选
   :param norm_first: 是否使用Pre-LN模式
   :type norm_first: bool, 可选
   :param bias: 是否在线性层添加偏置
   :type bias: bool, 可选
   :param device: 张量设备
   :type device: 可选
   :param dtype: 张量数据类型
   :type dtype: 可选

.. class:: riemann.nn.TransformerEncoder(encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True)

   由N个堆叠的TransformerEncoderLayer组成的Transformer编码器。

   :param encoder_layer: 单个编码器层实例，将被克隆
   :type encoder_layer: TransformerEncoderLayer
   :param num_layers: 编码器层的数量
   :type num_layers: int
   :param norm: 最后的层归一化，可选
   :type norm: Module, 可选
   :param enable_nested_tensor: 是否启用嵌套张量优化（仅接口兼容）
   :type enable_nested_tensor: bool, 可选
   :param mask_check: 是否进行掩码检查（仅接口兼容）
   :type mask_check: bool, 可选

.. class:: riemann.nn.TransformerDecoder(decoder_layer, num_layers, norm=None)

   由N个堆叠的TransformerDecoderLayer组成的Transformer解码器。

   :param decoder_layer: 单个解码器层实例，将被克隆
   :type decoder_layer: TransformerDecoderLayer
   :param num_layers: 解码器层的数量
   :type num_layers: int
   :param norm: 最后的层归一化，可选
   :type norm: Module, 可选

.. class:: riemann.nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', custom_encoder=None, custom_decoder=None, layer_norm_eps=1e-05, batch_first=False, norm_first=False, bias=True, device=None, dtype=None)

   包含编码器和解码器的完整Transformer架构。

   :param d_model: 编码器/解码器输入的维度
   :type d_model: int, 可选
   :param nhead: 注意力头的数量
   :type nhead: int, 可选
   :param num_encoder_layers: 编码器层的数量
   :type num_encoder_layers: int, 可选
   :param num_decoder_layers: 解码器层的数量
   :type num_decoder_layers: int, 可选
   :param dim_feedforward: 前馈网络的维度
   :type dim_feedforward: int, 可选
   :param dropout: dropout值
   :type dropout: float, 可选
   :param activation: 激活函数，'relu'或'gelu'
   :type activation: str, 可选
   :param custom_encoder: 自定义编码器模块
   :type custom_encoder: Module, 可选
   :param custom_decoder: 自定义解码器模块
   :type custom_decoder: Module, 可选
   :param layer_norm_eps: 层归一化的epsilon值
   :type layer_norm_eps: float, 可选
   :param batch_first: 输入输出形状是否为(batch, seq, feature)
   :type batch_first: bool, 可选
   :param norm_first: 是否在注意力和前馈操作之前执行LayerNorm
   :type norm_first: bool, 可选
   :param bias: 线性层和LayerNorm层是否学习加性偏置
   :type bias: bool, 可选
   :param device: 张量设备
   :type device: 可选
   :param dtype: 张量数据类型
   :type dtype: 可选

函数式接口
~~~~~~~~~~

``riemann.nn.functional`` 模块提供了各种神经网络操作的函数式实现。

线性函数
````````

.. function:: riemann.nn.functional.linear(input, weight, bias=None)

   应用线性变换：y = xA^T + b

   :param input: 输入张量，形状为 (\*, in_features)
   :type input: riemann.TN
   :param weight: 权重张量，形状为 (out_features, in_features)
   :type weight: riemann.TN
   :param bias: 偏置张量，形状为 (out_features,). 默认值: None
   :type bias: riemann.TN, optional
   :return: 输出张量，形状为 (\*, out_features)
   :rtype: riemann.TN

激活函数
````````

.. function:: riemann.nn.functional.sigmoid(input)

   应用逐元素的 sigmoid 函数：sigmoid(x) = 1 / (1 + exp(-x))

   :param input: 输入张量
   :type input: riemann.TN
   :return: 输出张量
   :rtype: riemann.TN

.. function:: riemann.nn.functional.silu(input)

   应用 Sigmoid Linear Unit (SiLU) 激活函数：silu(x) = x * sigmoid(x)

   :param input: 输入张量
   :type input: riemann.TN
   :return: 输出张量
   :rtype: riemann.TN

.. function:: riemann.nn.functional.tanh(input)

   应用双曲正切激活函数：tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

   :param input: 输入张量
   :type input: riemann.TN

Dropout函数
````````````

.. function:: riemann.nn.functional.dropout(input, p=0.5, training=True, inplace=False)

   在训练期间，以概率p随机将输入张量的元素置为0，并对剩余元素乘以1/(1-p)进行缩放。
   在评估期间，不执行任何操作。

   :param input: 输入张量
   :type input: riemann.TN
   :param p: 丢弃概率
   :type p: float, optional
   :param training: 是否在训练模式
   :type training: bool, optional
   :param inplace: 是否原地操作
   :type inplace: bool, optional
   :return: 应用dropout后的张量
   :rtype: riemann.TN

.. function:: riemann.nn.functional.dropout2d(input, p=0.5, training=True, inplace=False)

   在训练期间，以概率p随机将输入张量的整个通道置为0，并对剩余通道乘以1/(1-p)进行缩放。
   在评估期间，不执行任何操作。

   :param input: 输入张量，形状为 (N, C, H, W)
   :type input: riemann.TN
   :param p: 丢弃概率
   :type p: float, optional
   :param training: 是否在训练模式
   :type training: bool, optional
   :param inplace: 是否原地操作
   :type inplace: bool, optional
   :return: 应用dropout后的张量
   :rtype: riemann.TN

.. function:: riemann.nn.functional.dropout3d(input, p=0.5, training=True, inplace=False)

   在训练期间，以概率p随机将输入张量的整个通道置为0，并对剩余通道乘以1/(1-p)进行缩放。
   在评估期间，不执行任何操作。

   :param input: 输入张量，形状为 (N, C, D, H, W)
   :type input: riemann.TN
   :param p: 丢弃概率
   :type p: float, optional
   :param training: 是否在训练模式
   :type training: bool, optional
   :param inplace: 是否原地操作
   :type inplace: bool, optional
   :return: 应用dropout后的张量
   :rtype: riemann.TN

归一化函数
``````````

.. function:: riemann.nn.functional.batch_norm(input, running_mean=None, running_var=None, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5)

   对输入张量应用批量归一化。

   :param input: 输入张量，形状为 (N, C), (N, C, L), (N, C, H, W) 或 (N, C, D, H, W)
   :type input: riemann.TN
   :param running_mean: 运行时均值，形状为 (C,)
   :type running_mean: riemann.TN, optional
   :param running_var: 运行时方差，形状为 (C,)
   :type running_var: riemann.TN, optional
   :param weight: 可学习的缩放参数γ，形状为 (C,)
   :type weight: riemann.TN, optional
   :param bias: 可学习的偏移参数β，形状为 (C,)
   :type bias: riemann.TN, optional
   :param training: 是否为训练模式
   :type training: bool, optional
   :param momentum: 运行时统计量的动量
   :type momentum: float, optional
   :param eps: 数值稳定性的小常数
   :type eps: float, optional
   :return: 归一化后的张量，形状与输入相同
   :rtype: riemann.TN

.. function:: riemann.nn.functional.layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05)

   对输入张量的指定维度应用层归一化。

   :param input: 输入张量
   :type input: riemann.TN
   :param normalized_shape: 整数或元组，指定要归一化的维度
   :type normalized_shape: int 或 tuple
   :param weight: 可选的权重张量（γ），用于仿射变换
   :type weight: riemann.TN, optional
   :param bias: 可选的偏置张量（β），用于仿射变换
   :type bias: riemann.TN, optional
   :param eps: 为避免除零而添加到方差的小值
   :type eps: float, optional
   :return: 归一化后的张量，形状与输入相同
   :rtype: riemann.TN

嵌入函数
````````

.. function:: riemann.nn.functional.embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)

   从嵌入矩阵中查找输入索引的嵌入向量。

   :param input: 包含索引的张量，形状为任意维度
   :type input: riemann.TN
   :param weight: 嵌入矩阵，形状为 (num_embeddings, embedding_dim)
   :type weight: riemann.TN
   :param padding_idx: 如果指定，该索引的嵌入向量不参与梯度计算，且在训练过程中保持不变
   :type padding_idx: int, optional
   :param max_norm: 如果指定，所有嵌入向量的范数超过max_norm时，将被重归一化到max_norm
   :type max_norm: float, optional
   :param norm_type: 计算范数时使用的p值，默认为2（L2范数）
   :type norm_type: float, optional
   :param scale_grad_by_freq: 如果为True，梯度将按mini-batch中每个词的频率进行缩放
   :type scale_grad_by_freq: bool, optional
   :param sparse: 如果为True，权重的梯度将是稀疏张量
   :type sparse: bool, optional
   :return: 输出张量，形状为 (``*``, embedding_dim)，其中``*``是输入的形状
   :rtype: riemann.TN

.. function:: riemann.nn.functional.softmax(input, dim)

   沿指定维度应用 softmax 函数

   :param input: 输入张量
   :type input: riemann.TN
   :param dim: 计算 softmax 的维度
   :type dim: int
   :return: 输出张量
   :rtype: riemann.TN

.. function:: riemann.nn.functional.log_softmax(input, dim=-1)

   应用对数 softmax 函数以获得数值稳定性

   :param input: 输入张量
   :type input: riemann.TN
   :param dim: 计算 log_softmax 的维度
   :type dim: int, optional
   :return: 输出张量
   :rtype: riemann.TN

.. function:: riemann.nn.functional.relu(input)

   应用整流线性单元激活函数：relu(x) = max(0, x)

   :param input: 输入张量
   :type input: riemann.TN
   :return: 输出张量
   :rtype: riemann.TN

.. function:: riemann.nn.functional.leaky_relu(input, alpha=0.01)

   应用带泄漏的整流线性单元激活函数

   :param input: 输入张量
   :type input: riemann.TN
   :param alpha: 负区域的斜率
   :type alpha: float, optional
   :return: 输出张量
   :rtype: riemann.TN

.. function:: riemann.nn.functional.prelu(input, alpha)

   应用参数化整流线性单元激活函数

   :param input: 输入张量
   :type input: riemann.TN
   :param alpha: 可学习的参数张量
   :type alpha: riemann.TN
   :return: 输出张量
   :rtype: riemann.TN

.. function:: riemann.nn.functional.rrelu(input, lower=1.0/8.0, upper=1.0/3.0, training=True)

   应用随机化整流线性单元激活函数

   :param input: 输入张量
   :type input: riemann.TN
   :param lower: 均匀分布的下限
   :type lower: float, optional
   :param upper: 均匀分布的上限
   :type upper: float, optional
   :param training: 是否使用随机化的 alpha（训练）或固定的 alpha（评估）
   :type training: bool, optional
   :return: 输出张量
   :rtype: riemann.TN

.. function:: riemann.nn.functional.gelu(input)

   应用高斯误差线性单元激活函数

   :param input: 输入张量
   :type input: riemann.TN
   :return: 输出张量
   :rtype: riemann.TN

.. function:: riemann.nn.functional.softplus(input, beta=1.0, threshold=20.0)

   应用 Softplus 激活函数：softplus(x) = (1 / beta) * log(1 + exp(beta * x))

   :param input: 输入张量
   :type input: riemann.TN
   :param beta: 线性部分的斜率
   :type beta: float, optional
   :param threshold: 数值稳定性的阈值
   :type threshold: float, optional
   :return: 输出张量
   :rtype: riemann.TN

损失函数
````````

.. function:: riemann.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='mean')

   计算均方误差损失

   :param input: 输入张量
   :type input: riemann.TN
   :param target: 目标张量
   :type target: riemann.TN
   :param size_average: 已弃用
   :type size_average: bool, optional
   :param reduce: 已弃用
   :type reduce: bool, optional
   :param reduction: 指定输出的聚合方式
   :type reduction: str, optional
   :return: 损失值
   :rtype: riemann.TN

.. function:: riemann.nn.functional.l1_loss(input, target, size_average=None, reduce=None, reduction='mean')

   计算 L1（绝对误差）损失

   :param input: 输入张量
   :type input: riemann.TN
   :param target: 目标张量
   :type target: riemann.TN
   :param size_average: 已弃用
   :type size_average: bool, optional
   :param reduce: 已弃用
   :type reduce: bool, optional
   :param reduction: 指定输出的聚合方式
   :type reduction: str, optional
   :return: 损失值
   :rtype: riemann.TN

.. function:: riemann.nn.functional.smooth_l1_loss(input, target, size_average=None, reduce=None, reduction='mean', beta=1.0)

   计算平滑 L1 损失

   :param input: 输入张量
   :type input: riemann.TN
   :param target: 目标张量
   :type target: riemann.TN
   :param size_average: 已弃用
   :type size_average: bool, optional
   :param reduce: 已弃用
   :type reduce: bool, optional
   :param reduction: 指定输出的聚合方式
   :type reduction: str, optional
   :param beta: 损失函数从二次变为线性的阈值
   :type beta: float, optional
   :return: 损失值
   :rtype: riemann.TN

.. function:: riemann.nn.functional.cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)

   计算交叉熵损失

   :param input: 输入张量
   :type input: riemann.TN
   :param target: 目标张量
   :type target: riemann.TN
   :param weight: 每个类别的手动缩放权重
   :type weight: riemann.TN, optional
   :param size_average: 已弃用
   :type size_average: bool, optional
   :param ignore_index: 指定要忽略的目标值
   :type ignore_index: int, optional
   :param reduce: 已弃用
   :type reduce: bool, optional
   :param reduction: 指定输出的聚合方式
   :type reduction: str, optional
   :param label_smoothing: 标签平滑的量
   :type label_smoothing: float, optional
   :return: 损失值
   :rtype: riemann.TN

.. function:: riemann.nn.functional.binary_cross_entropy_with_logits(input, target, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)

   计算带 logits 的二元交叉熵损失

   :param input: 输入张量
   :type input: riemann.TN
   :param target: 目标张量
   :type target: riemann.TN
   :param weight: 每个批次元素的手动缩放权重
   :type weight: riemann.TN, optional
   :param size_average: 已弃用
   :type size_average: bool, optional
   :param reduce: 已弃用
   :type reduce: bool, optional
   :param reduction: 指定输出的聚合方式
   :type reduction: str, optional
   :param pos_weight: 正类的权重
   :type pos_weight: riemann.TN, optional
   :return: 损失值
   :rtype: riemann.TN

.. function:: riemann.nn.functional.huber_loss(input, target, delta=1.0, size_average=None, reduce=None, reduction='mean')

   计算 Huber 损失

   :param input: 输入张量
   :type input: riemann.TN
   :param target: 目标张量
   :type target: riemann.TN
   :param delta: 损失函数从二次变为线性的阈值
   :type delta: float, optional
   :param size_average: 已弃用
   :type size_average: bool, optional
   :param reduce: 已弃用
   :type reduce: bool, optional
   :param reduction: 指定输出的聚合方式
   :type reduction: str, optional
   :return: 损失值
   :rtype: riemann.TN

.. function:: riemann.nn.functional.nll_loss(input, target, weight=None, ignore_index=-100, reduction='mean')

   计算负对数似然损失

   :param input: 输入张量
   :type input: riemann.TN
   :param target: 目标张量
   :type target: riemann.TN
   :param weight: 每个类别的手动缩放权重
   :type weight: riemann.TN, optional
   :param ignore_index: 指定要忽略的目标值
   :type ignore_index: int, optional
   :param reduction: 指定输出的聚合方式
   :type reduction: str, optional
   :return: 损失值
   :rtype: riemann.TN

卷积函数
````````

.. function:: riemann.nn.functional.conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)

   对输入信号应用 1D 卷积

   :param input: 输入张量，形状为 (N, C_in, L_in)
   :type input: riemann.TN
   :param weight: 权重张量，形状为 (C_out, C_in/groups, K)
   :type weight: riemann.TN
   :param bias: 偏置张量，形状为 (C_out). 默认值: None
   :type bias: riemann.TN, optional
   :param stride: 卷积步长
   :type stride: int or tuple, optional
   :param padding: 添加到输入两侧的零填充
   :type padding: int or tuple, optional
   :param dilation: 内核元素之间的间距
   :type dilation: int or tuple, optional
   :param groups: 从输入通道到输出通道的阻塞连接数
   :type groups: int, optional
   :return: 输出张量，形状为 (N, C_out, L_out)
   :rtype: riemann.TN

.. function:: riemann.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)

   对输入图像应用 2D 卷积

   :param input: 输入张量，形状为 (N, C_in, H_in, W_in)
   :type input: riemann.TN
   :param weight: 权重张量，形状为 (C_out, C_in/groups, K_h, K_w)
   :type weight: riemann.TN
   :param bias: 偏置张量，形状为 (C_out). 默认值: None
   :type bias: riemann.TN, optional
   :param stride: 卷积步长
   :type stride: int or tuple, optional
   :param padding: 添加到输入两侧的零填充
   :type padding: int or tuple, optional
   :param dilation: 内核元素之间的间距
   :type dilation: int or tuple, optional
   :param groups: 从输入通道到输出通道的阻塞连接数
   :type groups: int, optional
   :return: 输出张量，形状为 (N, C_out, H_out, W_out)
   :rtype: riemann.TN

.. function:: riemann.nn.functional.conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)

   对输入体积应用 3D 卷积

   :param input: 输入张量，形状为 (N, C_in, D_in, H_in, W_in)
   :type input: riemann.TN
   :param weight: 权重张量，形状为 (C_out, C_in/groups, K_d, K_h, K_w)
   :type weight: riemann.TN
   :param bias: 偏置张量，形状为 (C_out). 默认值: None
   :type bias: riemann.TN, optional
   :param stride: 卷积步长
   :type stride: int or tuple, optional
   :param padding: 添加到输入所有侧面的零填充
   :type padding: int or tuple, optional
   :param dilation: 内核元素之间的间距
   :type dilation: int or tuple, optional
   :param groups: 从输入通道到输出通道的阻塞连接数
   :type groups: int, optional
   :return: 输出张量，形状为 (N, C_out, D_out, H_out, W_out)
   :rtype: riemann.TN

池化函数
````````

.. function:: riemann.nn.functional.max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

   对输入信号应用 1D 最大池化

   :param input: 输入张量，形状为 (N, C, L_in)
   :type input: riemann.TN
   :param kernel_size: 池化窗口大小
   :type kernel_size: int or tuple
   :param stride: 池化窗口移动步长。默认值: kernel_size
   :type stride: int or tuple, optional
   :param padding: 添加到输入两侧的零填充
   :type padding: int or tuple, optional
   :param dilation: 池化窗口元素之间的间距
   :type dilation: int or tuple, optional
   :param ceil_mode: 是否使用向上取整来计算输出形状
   :type ceil_mode: bool, optional
   :param return_indices: 是否返回最大值的索引
   :type return_indices: bool, optional
   :return: 输出张量，形状为 (N, C, L_out)，如果 return_indices 为 True，则返回元组 (TN, TN)
   :rtype: riemann.TN or tuple

.. function:: riemann.nn.functional.max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

   对输入图像应用 2D 最大池化

   :param input: 输入张量，形状为 (N, C, H_in, W_in)
   :type input: riemann.TN
   :param kernel_size: 池化窗口大小
   :type kernel_size: int or tuple
   :param stride: 池化窗口移动步长。默认值: kernel_size
   :type stride: int or tuple, optional
   :param padding: 添加到输入两侧的零填充
   :type padding: int or tuple, optional
   :param dilation: 池化窗口元素之间的间距
   :type dilation: int or tuple, optional
   :param ceil_mode: 是否使用向上取整来计算输出形状
   :type ceil_mode: bool, optional
   :param return_indices: 是否返回最大值的索引
   :type return_indices: bool, optional
   :return: 输出张量，形状为 (N, C, H_out, W_out)，如果 return_indices 为 True，则返回元组 (TN, TN)
   :rtype: riemann.TN or tuple

.. function:: riemann.nn.functional.max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

   对输入体积数据应用 3D 最大池化

   :param input: 输入张量，形状为 (N, C, D_in, H_in, W_in)
   :type input: riemann.TN
   :param kernel_size: 池化窗口大小
   :type kernel_size: int or tuple
   :param stride: 池化窗口移动步长。默认值: kernel_size
   :type stride: int or tuple, optional
   :param padding: 添加到输入所有侧面的零填充
   :type padding: int or tuple, optional
   :param dilation: 池化窗口元素之间的间距
   :type dilation: int or tuple, optional
   :param ceil_mode: 是否使用向上取整来计算输出形状
   :type ceil_mode: bool, optional
   :param return_indices: 是否返回最大值的索引
   :type return_indices: bool, optional
   :return: 输出张量，形状为 (N, C, D_out, H_out, W_out)，如果 return_indices 为 True，则返回元组 (TN, TN)
   :rtype: riemann.TN or tuple

.. function:: riemann.nn.functional.avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)

   对输入信号应用 1D 平均池化

   :param input: 输入张量，形状为 (N, C, L_in)
   :type input: riemann.TN
   :param kernel_size: 池化窗口大小
   :type kernel_size: int or tuple
   :param stride: 池化窗口移动步长。默认值: kernel_size
   :type stride: int or tuple, optional
   :param padding: 添加到输入两侧的零填充
   :type padding: int or tuple, optional
   :param ceil_mode: 是否使用向上取整来计算输出形状
   :type ceil_mode: bool, optional
   :param count_include_pad: 计算平均值时是否包含零填充
   :type count_include_pad: bool, optional
   :param divisor_override: 如果指定，将使用该值作为分母
   :type divisor_override: int, optional
   :return: 输出张量，形状为 (N, C, L_out)
   :rtype: riemann.TN

.. function:: riemann.nn.functional.avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)

   对输入图像应用 2D 平均池化

   :param input: 输入张量，形状为 (N, C, H_in, W_in)
   :type input: riemann.TN
   :param kernel_size: 池化窗口大小
   :type kernel_size: int or tuple
   :param stride: 池化窗口移动步长。默认值: kernel_size
   :type stride: int or tuple, optional
   :param padding: 添加到输入两侧的零填充
   :type padding: int or tuple, optional
   :param ceil_mode: 是否使用向上取整来计算输出形状
   :type ceil_mode: bool, optional
   :param count_include_pad: 计算平均值时是否包含零填充
   :type count_include_pad: bool, optional
   :param divisor_override: 如果指定，将使用该值作为分母
   :type divisor_override: int, optional
   :return: 输出张量，形状为 (N, C, H_out, W_out)
   :rtype: riemann.TN

.. function:: riemann.nn.functional.avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)

   对输入体积数据应用 3D 平均池化

   :param input: 输入张量，形状为 (N, C, D_in, H_in, W_in)
   :type input: riemann.TN
   :param kernel_size: 池化窗口大小
   :type kernel_size: int or tuple
   :param stride: 池化窗口移动步长。默认值: kernel_size
   :type stride: int or tuple, optional
   :param padding: 添加到输入所有侧面的零填充
   :type padding: int or tuple, optional
   :param ceil_mode: 是否使用向上取整来计算输出形状
   :type ceil_mode: bool, optional
   :param count_include_pad: 计算平均值时是否包含零填充
   :type count_include_pad: bool, optional
   :param divisor_override: 如果指定，将使用该值作为分母
   :type divisor_override: int, optional
   :return: 输出张量，形状为 (N, C, D_out, H_out, W_out)
   :rtype: riemann.TN

工具函数
````````

.. function:: riemann.nn.functional.one_hot(target, num_classes)

   将类别索引转换为 one-hot 编码张量

   :param target: 目标张量，形状为 (N, \*)
   :type target: riemann.TN
   :param num_classes: 类别数量
   :type num_classes: int
   :return: One-hot 编码张量，形状为 (N, \*, num_classes)
   :rtype: riemann.TN

.. function:: riemann.nn.functional.unfold(input, kernel_size, dilation=1, padding=0, stride=1)

   从批处理输入张量中提取滑动局部块

   :param input: 输入张量，形状为 (N, C, H, W)
   :type input: riemann.TN
   :param kernel_size: 滑动块大小
   :type kernel_size: int or tuple
   :param dilation: 内核元素之间的间距
   :type dilation: int or tuple, optional
   :param padding: 添加到输入两侧的零填充
   :type padding: int or tuple, optional
   :param stride: 滑动块的步长
   :type stride: int or tuple, optional
   :return: 展开后的张量，形状为 (N, C * kernel_size[0] * kernel_size[1], L)
   :rtype: riemann.TN

.. function:: riemann.nn.functional.fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1)

   将展开的张量折叠回原始形状

   :param input: 输入张量，形状为 (N, C * kernel_size[0] * kernel_size[1], L)
   :type input: riemann.TN
   :param output_size: 输出张量大小 (H, W)
   :type output_size: int or tuple
   :param kernel_size: 滑动块大小
   :type kernel_size: int or tuple
   :param dilation: 内核元素之间的间距
   :type dilation: int or tuple, optional
   :param padding: 添加到输入两侧的零填充
   :type padding: int or tuple, optional
   :param stride: 滑动块的步长
   :type stride: int or tuple, optional
   :return: 折叠后的张量，形状为 (N, C, H, W)
   :rtype: riemann.TN

.. function:: riemann.nn.functional.unfold2d(input, kernel_size, dilation=1, padding=0, stride=1)

   从2D输入张量中提取滑动局部块（unfold的2D专用版本）

   :param input: 输入张量，形状为 (N, C, H, W)
   :type input: riemann.TN
   :param kernel_size: 滑动块大小
   :type kernel_size: int or tuple
   :param dilation: 内核元素之间的间距
   :type dilation: int or tuple, optional
   :param padding: 添加到输入两侧的零填充
   :type padding: int or tuple, optional
   :param stride: 滑动块的步长
   :type stride: int or tuple, optional
   :return: 展开后的张量，形状为 (N, C * kernel_size[0] * kernel_size[1], L)
   :rtype: riemann.TN

.. function:: riemann.nn.functional.unfold3d(input, kernel_size, dilation=1, padding=0, stride=1)

   从3D输入张量中提取滑动局部块（unfold的3D专用版本）

   :param input: 输入张量，形状为 (N, C, D, H, W)
   :type input: riemann.TN
   :param kernel_size: 滑动块大小
   :type kernel_size: int or tuple
   :param dilation: 内核元素之间的间距
   :type dilation: int or tuple, optional
   :param padding: 添加到输入所有侧面的零填充
   :type padding: int or tuple, optional
   :param stride: 滑动块的步长
   :type stride: int or tuple, optional
   :return: 展开后的张量，形状为 (N, C * kernel_size[0] * kernel_size[1] * kernel_size[2], L)
   :rtype: riemann.TN

数据集
------

数据集类
~~~~~~~~

.. class:: riemann.utils.Dataset

   抽象数据集基类，定义了所有数据集必须实现的标准接口。

   .. method:: __len__()

      返回数据集中的样本数量。

   .. method:: __getitem__(index)

      根据给定索引获取数据集中的单个样本。

.. class:: riemann.utils.TensorDataset(*tensors)

   简单的张量数据集实现，将多个张量的第一个维度作为数据集维度。

   :param \*tensors: 可变数量的张量，所有张量的第一个维度大小必须相同
   :type \*tensors: riemann.TN

   .. method:: __len__()

      返回数据集的大小，即张量的第一个维度大小。

   .. method:: __getitem__(index)

      获取指定索引处的样本数据。

数据加载器
~~~~~~~~~~

.. class:: riemann.utils.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=None, drop_last=False)

   高效的数据加载器，支持批次处理、数据洗牌和多进程加载。

   :param dataset: 要加载数据的数据集
   :type dataset: riemann.utils.Dataset
   :param batch_size: 每个批次的大小，默认为1
   :type batch_size: int, optional
   :param shuffle: 是否在每个 epoch 开始时洗牌数据，默认为False
   :type shuffle: bool, optional
   :param num_workers: 数据加载的工作进程数，0表示主进程加载，默认为0
   :type num_workers: int, optional
   :param collate_fn: 批次处理函数，用于将样本组合成批次，默认使用default_collate
   :type collate_fn: callable, optional
   :param drop_last: 如果数据集大小不能被批次大小整除，是否丢弃最后一个不完整的批次，默认为False
   :type drop_last: bool, optional

   .. method:: __len__()

      返回数据加载器的批次数目。

   .. method:: __iter__()

      返回数据加载器的迭代器。

数据集工具函数
~~~~~~~~~~~~~~

.. function:: riemann.utils.default_collate(batch)

   默认的批次处理函数，将一批样本数据转换为适合模型输入的张量格式。

   :param batch: 一个批次的样本列表，每个样本可以是各种数据类型。
   :type batch: list
   :return: 根据输入类型组合成的批次数据。

.. function:: riemann.utils.clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False)

   按范数裁剪梯度。

   :param parameters: 需要裁剪梯度的参数集合
   :type parameters: Iterable[riemann.TN]
   :param max_norm: 梯度的最大范数
   :type max_norm: float or int
   :param norm_type: 范数类型，默认为2（L2范数）
   :type norm_type: float or int, optional
   :param error_if_nonfinite: 如果梯度包含非有限值（如NaN或inf），是否抛出错误，默认为False
   :type error_if_nonfinite: bool, optional
   :return: 裁剪前的梯度范数
   :rtype: float

.. function:: riemann.utils.clip_grad_value_(parameters, clip_value, error_if_nonfinite=False)

   按值裁剪梯度。

   :param parameters: 需要裁剪梯度的参数集合
   :type parameters: Iterable[riemann.TN]
   :param clip_value: 梯度裁剪的阈值
   :type clip_value: float or int
   :param error_if_nonfinite: 如果梯度包含非有限值（如NaN或inf），是否抛出错误，默认为False
   :type error_if_nonfinite: bool, optional

视觉
----

.. module:: riemann.vision

数据集
~~~~~~

.. class:: riemann.vision.datasets.MNIST(root, train=True, transform=None, target_transform=None)

   MNIST数据集类，用于加载和处理MNIST手写数字数据集。

   :param root: 数据集的根目录
   :type root: str
   :param train: 是否加载训练集，默认为True
   :type train: bool
   :param transform: 应用于图像的变换函数
   :type transform: callable, optional
   :param target_transform: 应用于目标的变换函数
   :type target_transform: callable, optional

   .. method:: __len__()

      返回数据集中的样本数量。

   .. method:: __getitem__(index)

      根据给定索引获取数据集中的单个样本。

.. class:: riemann.vision.datasets.EasyMNIST(root, train=True, onehot_label=True, download=False)

   继承自MNIST的子类，在初始化时对图像数据应用归一化、标准化、展开转换，对标签作onehot编码或转换为标量张量。

   :param root: 数据集的根目录
   :type root: str
   :param train: 是否加载训练集，默认为True
   :type train: bool
   :param onehot_label: 是否使用one-hot编码的标签，默认为True
   :type onehot_label: bool
   :param download: 如果数据集不存在是否下载，默认为False
   :type download: bool, optional

   .. method:: __len__()

      返回数据集的大小。

   .. method:: __getitem__(index)

      获取指定索引处的样本数据。

.. class:: riemann.vision.datasets.FashionMNIST(root, train=True, transform=None, target_transform=None, download=False)

   Fashion-MNIST数据集类，用于加载和处理Fashion-MNIST时尚产品数据集。

   :param root: 数据集的根目录
   :type root: str
   :param train: 是否加载训练集，默认为True
   :type train: bool
   :param transform: 应用于图像的变换函数
   :type transform: callable, optional
   :param target_transform: 应用于目标的变换函数
   :type target_transform: callable, optional
   :param download: 如果数据集不存在是否下载，默认为False
   :type download: bool, optional

   .. attribute:: classes

      类别名称列表：['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

   .. method:: __len__()

      返回数据集中的样本数量。

   .. method:: __getitem__(index)

      根据给定索引获取数据集中的单个样本。

.. class:: riemann.vision.datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=False)

   CIFAR-10数据集类，用于加载和处理CIFAR-10图像数据集。

   :param root: 数据集的根目录
   :type root: str
   :param train: 是否加载训练集，默认为True
   :type train: bool
   :param transform: 应用于图像的变换函数
   :type transform: callable, optional
   :param target_transform: 应用于目标的变换函数
   :type target_transform: callable, optional
   :param download: 如果数据集不存在是否下载，默认为False
   :type download: bool, optional

   .. method:: __len__()

      返回数据集的大小。

   .. method:: __getitem__(index)

      获取指定索引处的样本数据。

.. class:: riemann.vision.datasets.Flowers102(root, split='train', transform=None, target_transform=None, download=False)

   Oxford 102 Flower数据集类，用于加载和处理花卉分类数据集。

   :param root: 数据集的根目录
   :type root: str
   :param split: 数据集划分（'train'、'val' 或 'test'），默认为'train'
   :type split: str, optional
   :param transform: 应用于图像的变换函数
   :type transform: callable, optional
   :param target_transform: 应用于目标的变换函数
   :type target_transform: callable, optional
   :param download: 如果数据集不存在是否下载，默认为False
   :type download: bool, optional

   .. method:: __len__()

      返回数据集的大小。

   .. method:: __getitem__(index)

      获取指定索引处的样本数据。

.. class:: riemann.vision.datasets.OxfordIIITPet(root, split='trainval', target_types='category', transform=None, target_transform=None, download=False)

   Oxford-IIIT Pet数据集类，用于加载和处理宠物分类数据集。

   :param root: 数据集的根目录
   :type root: str
   :param split: 数据集划分（'trainval' 或 'test'），默认为'trainval'
   :type split: str, optional
   :param target_types: 目标类型（'category'、'binary-category' 或 'segmentation'），默认为'category'
   :type target_types: str 或 list, optional
   :param transform: 应用于图像的变换函数
   :type transform: callable, optional
   :param target_transform: 应用于目标的变换函数
   :type target_transform: callable, optional
   :param download: 如果数据集不存在是否下载，默认为False
   :type download: bool, optional

   .. method:: __len__()

      返回数据集的大小。

   .. method:: __getitem__(index)

      获取指定索引处的样本数据。

.. class:: riemann.vision.datasets.LFWPeople(root, split='10fold', image_set='funneled', transform=None, target_transform=None, download=False)

   LFW People数据集类，用于加载和处理人脸识别数据集。

   :param root: 数据集的根目录
   :type root: str
   :param split: 数据集划分（'10fold'、'train' 或 'test'），默认为'10fold'
   :type split: str, optional
   :param image_set: 图像对齐类型（'original'、'funneled' 或 'deepfunneled'），默认为'funneled'
   :type image_set: str, optional
   :param transform: 应用于图像的变换函数
   :type transform: callable, optional
   :param target_transform: 应用于目标的变换函数
   :type target_transform: callable, optional
   :param download: 如果数据集不存在是否下载，默认为False
   :type download: bool, optional

   .. attribute:: classes

      人物名称列表。

   .. method:: __len__()

      返回数据集的大小。

   .. method:: __getitem__(index)

      获取指定索引处的样本数据。

.. class:: riemann.vision.datasets.SVHN(root, split='train', transform=None, target_transform=None, download=False)

   SVHN（街景门牌号码）数据集类，用于加载和处理数字识别数据集。

   :param root: 数据集的根目录
   :type root: str
   :param split: 数据集划分（'train'、'test' 或 'extra'），默认为'train'
   :type split: str, optional
   :param transform: 应用于图像的变换函数
   :type transform: callable, optional
   :param target_transform: 应用于目标的变换函数
   :type target_transform: callable, optional
   :param download: 如果数据集不存在是否下载，默认为False
   :type download: bool, optional

   .. method:: __len__()

      返回数据集的大小。

   .. method:: __getitem__(index)

      获取指定索引处的样本数据。

.. class:: riemann.vision.datasets.DatasetFolder(root, loader, extensions=None, transform=None, target_transform=None, is_valid_file=None, allow_empty=False)

   通用文件夹数据集类，用于从文件夹加载自定义数据集。

   :param root: 数据集的根目录路径
   :type root: str
   :param loader: 图像加载函数
   :type loader: callable
   :param extensions: 允许的文件扩展名元组
   :type extensions: tuple, optional
   :param transform: 应用于图像的变换函数
   :type transform: callable, optional
   :param target_transform: 应用于目标的变换函数
   :type target_transform: callable, optional
   :param is_valid_file: 验证文件是否有效的函数
   :type is_valid_file: callable, optional
   :param allow_empty: 是否允许空文件夹，默认为False
   :type allow_empty: bool

   .. attribute:: classes

      类别名称列表。

   .. attribute:: class_to_idx

      类别名称到索引的映射字典。

   .. method:: __len__()

      返回数据集中的样本数量。

   .. method:: __getitem__(index)

      根据给定索引获取数据集中的单个样本。

.. class:: riemann.vision.datasets.ImageFolder(root, transform=None, target_transform=None, loader=None, is_valid_file=None)

   图像文件夹数据集类，继承自 DatasetFolder，用于从文件夹加载图像数据集。

   :param root: 数据集的根目录路径
   :type root: str
   :param transform: 应用于图像的变换函数
   :type transform: callable, optional
   :param target_transform: 应用于目标的变换函数
   :type target_transform: callable, optional
   :param loader: 图像加载函数，默认为 PIL Image 加载
   :type loader: callable, optional
   :param is_valid_file: 验证文件是否有效的函数
   :type is_valid_file: callable, optional

   .. attribute:: classes

      类别名称列表。

   .. attribute:: class_to_idx

      类别名称到索引的映射字典。

   .. method:: __len__()

      返回数据集中的样本数量。

   .. method:: __getitem__(index)

      根据给定索引获取数据集中的单个样本。

图像变换
~~~~~~~~

.. module:: riemann.vision.transforms

.. class:: riemann.vision.transforms.Transform

   所有变换类的基类。

   .. method:: __call__(img)

      执行变换。

.. class:: riemann.vision.transforms.Compose(transforms)

   将多个变换组合成一个变换。

   :param transforms: 要组合的变换列表
   :type transforms: list of Transform objects

.. class:: riemann.vision.transforms.ToTensor

   将PIL图像或NumPy数组转换为TN张量。

.. class:: riemann.vision.transforms.ToPILImage

   将TN张量或NumPy数组转换为PIL图像。

.. class:: riemann.vision.transforms.Normalize(mean, std, inplace=False)

   使用均值和标准差标准化张量。

   :param mean: 每个通道的均值
   :type mean: sequence
   :param std: 每个通道的标准差
   :type std: sequence
   :param inplace: 是否原地操作，默认为False
   :type inplace: bool, optional

.. class:: riemann.vision.transforms.Resize(size, interpolation=BILINEAR)

   调整PIL图像大小。

   :param size: 目标大小。如果是int，则较小边会被调整为该大小，保持宽高比。如果是(h, w)，则直接调整为该大小。
   :type size: int or tuple
   :param interpolation: 插值方法，默认为BILINEAR
   :type interpolation: int, optional

.. class:: riemann.vision.transforms.CenterCrop(size)

   中心裁剪。

   :param size: 裁剪大小。如果是int，则裁剪为正方形(size, size)。如果是(h, w)，则裁剪为该大小。
   :type size: int or tuple

.. class:: riemann.vision.transforms.RandomHorizontalFlip(p=0.5)

   随机水平翻转。

   :param p: 翻转概率，默认为0.5
   :type p: float, optional

.. class:: riemann.vision.transforms.RandomVerticalFlip(p=0.5)

   随机垂直翻转。

   :param p: 翻转概率，默认为0.5
   :type p: float, optional

.. class:: riemann.vision.transforms.RandomRotation(degrees, resample=NEAREST, expand=False, center=None)

   随机旋转。

   :param degrees: 旋转角度范围。如果是int，则在(-degrees, degrees)范围内选择。如果是(min, max)，则在(min, max)范围内选择。
   :type degrees: int or tuple
   :param resample: 重采样方法，默认为NEAREST
   :type resample: int, optional
   :param expand: 是否扩展图像以适应旋转，默认为False
   :type expand: bool, optional
   :param center: 旋转中心，默认为图像中心
   :type center: tuple, optional

.. class:: riemann.vision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)

   随机颜色变换。

   :param brightness: 亮度调整因子
   :type brightness: float or tuple
   :param contrast: 对比度调整因子
   :type contrast: float or tuple
   :param saturation: 饱和度调整因子
   :type saturation: float or tuple
   :param hue: 色调调整因子
   :type hue: float or tuple

.. class:: riemann.vision.transforms.Grayscale(num_output_channels=1)

   将图像转换为灰度。

   :param num_output_channels: 输出通道数，1或3，默认为1
   :type num_output_channels: int

.. class:: riemann.vision.transforms.RandomGrayscale(p=0.1)

   随机转换为灰度。

   :param p: 转换为灰度的概率，默认为0.1
   :type p: float, optional

.. class:: riemann.vision.transforms.RandomCrop(size, padding=None)

   在随机位置裁剪图像。

   :param size: 裁剪大小。如果是int，则裁剪为正方形(size, size)。如果是(h, w)，则裁剪为该大小。
   :type size: int or tuple
   :param padding: 填充大小，默认为None
   :type padding: int or tuple, optional

.. class:: riemann.vision.transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(3/4, 4/3), interpolation=BILINEAR)

   随机裁剪并调整大小。

   :param size: 目标大小。如果是int，则调整为正方形(size, size)。如果是(h, w)，则调整为该大小。
   :type size: int or tuple
   :param scale: 裁剪面积相对于原图的比例范围，默认为(0.08, 1.0)
   :type scale: tuple, optional
   :param ratio: 裁剪的宽高比范围，默认为(3/4, 4/3)
   :type ratio: tuple, optional
   :param interpolation: 插值方法，默认为BILINEAR
   :type interpolation: int, optional

.. class:: riemann.vision.transforms.FiveCrop(size)

   五裁剪。

   :param size: 裁剪大小。如果是int，则裁剪为正方形(size, size)。如果是(h, w)，则裁剪为该大小。
   :type size: int or tuple

.. class:: riemann.vision.transforms.TenCrop(size, vertical_flip=False)

   十裁剪。

   :param size: 裁剪大小。如果是int，则裁剪为正方形(size, size)。如果是(h, w)，则裁剪为该大小。
   :type size: int or tuple
   :param vertical_flip: 是否包括垂直翻转版本，默认为False
   :type vertical_flip: bool, optional

.. class:: riemann.vision.transforms.Pad(padding, fill=0, padding_mode='constant')

   填充。

   :param padding: 填充大小。如果是int，则在所有方向填充相同大小。如果是(pad_l, pad_r, pad_t, pad_b)，则分别指定左右上下的填充大小。如果是(pad_h, pad_w)，则分别指定高度和宽度方向的填充大小。
   :type padding: int or tuple
   :param fill: 填充值，默认为0
   :type fill: int or tuple
   :param padding_mode: 填充模式，默认为'constant'
   :type padding_mode: str, optional

.. class:: riemann.vision.transforms.Lambda(lambd)

   使用用户定义的lambda函数作为变换。

   :param lambd: Lambda函数
   :type lambd: function

.. class:: riemann.vision.transforms.PILToTensor

   将PIL Image转换为张量（不缩放）。

.. class:: riemann.vision.transforms.ConvertImageDtype(dtype)

   转换图像数据类型。

   :param dtype: 目标数据类型
   :type dtype: torch.dtype

.. class:: riemann.vision.transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))

   对图像应用高斯模糊。

   :param kernel_size: 高斯核的大小
   :type kernel_size: int or tuple
   :param sigma: 高斯核的标准差范围
   :type sigma: tuple, optional

.. class:: riemann.vision.transforms.RandomAffine(degrees, translate=None, scale=None, shear=None, resample=NEAREST, fillcolor=0)

   随机仿射变换。

   :param degrees: 旋转角度范围
   :type degrees: float or tuple
   :param translate: 平移范围
   :type translate: tuple, optional
   :param scale: 缩放范围
   :type scale: tuple, optional
   :param shear: 剪切角度范围
   :type shear: float or tuple, optional
   :param resample: 重采样模式
   :type resample: int, optional
   :param fillcolor: 填充颜色
   :type fillcolor: int, optional

.. class:: riemann.vision.transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=BILINEAR, fill=0)

   随机透视变换。

   :param distortion_scale: 失真程度
   :type distortion_scale: float
   :param p: 应用变换的概率
   :type p: float
   :param interpolation: 插值模式
   :type interpolation: int
   :param fill: 填充值
   :type fill: int or tuple

.. class:: riemann.vision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)

   随机擦除，用于数据增强。

   :param p: 应用擦除的概率
   :type p: float
   :param scale: 擦除区域面积范围
   :type scale: tuple
   :param ratio: 擦除区域长宽比范围
   :type ratio: tuple
   :param value: 擦除填充值
   :type value: int or float or tuple
   :param inplace: 是否原地操作
   :type inplace: bool

.. class:: riemann.vision.transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET)

   自动数据增强，基于学习策略。

   :param policy: 增强策略
   :type policy: AutoAugmentPolicy

.. class:: riemann.vision.transforms.RandAugment(num_ops=2, magnitude=9, num_magnitude_bins=31, interpolation=BILINEAR, fill=None)

   随机数据增强。

   :param num_ops: 操作数量
   :type num_ops: int
   :param magnitude: 增强幅度
   :type magnitude: int
   :param num_magnitude_bins: 幅度分箱数
   :type num_magnitude_bins: int
   :param interpolation: 插值模式
   :type interpolation: int
   :param fill: 填充值
   :type fill: int or tuple or None

.. class:: riemann.vision.transforms.TrivialAugmentWide(num_magnitude_bins=31, interpolation=BILINEAR, fill=None)

   宽范围简单增强。

   :param num_magnitude_bins: 幅度分箱数
   :type num_magnitude_bins: int
   :param interpolation: 插值模式
   :type interpolation: int
   :param fill: 填充值
   :type fill: int or tuple or None

.. class:: riemann.vision.transforms.SanitizeBoundingBox(labels_format='xyxy', min_size=1)

   边界框清理。

   :param labels_format: 边界框格式
   :type labels_format: str
   :param min_size: 最小尺寸
   :type min_size: int

.. class:: riemann.vision.transforms.Invert

   颜色反转。

.. class:: riemann.vision.transforms.Posterize(bits)

   减少颜色位数。

   :param bits: 保留的位数
   :type bits: int

.. class:: riemann.vision.transforms.Solarize(threshold)

   反转高于阈值的像素。

   :param threshold: 阈值
   :type threshold: int

.. class:: riemann.vision.transforms.Equalize

   直方图均衡化。

.. class:: riemann.vision.transforms.AutoContrast

   自动对比度调整。

.. class:: riemann.vision.transforms.Sharpness(sharpness_factor)

   锐度调整。

   :param sharpness_factor: 锐度因子
   :type sharpness_factor: float

.. class:: riemann.vision.transforms.Brightness(brightness_factor)

   亮度调整。

   :param brightness_factor: 亮度因子
   :type brightness_factor: float

.. class:: riemann.vision.transforms.Contrast(contrast_factor)

   对比度调整。

   :param contrast_factor: 对比度因子
   :type contrast_factor: float

.. class:: riemann.vision.transforms.Saturation(saturation_factor)

   饱和度调整。

   :param saturation_factor: 饱和度因子
   :type saturation_factor: float

.. class:: riemann.vision.transforms.Hue(hue_factor)

   色调调整。

   :param hue_factor: 色调因子
   :type hue_factor: float

优化
----

优化器
~~~~~~

.. class:: riemann.optim.Optimizer(params, defaults)

   所有优化器的基类。

   :param params: 待优化参数的迭代器或定义参数组的字典列表
   :type params: Iterable[riemann.TN or riemann.nn.Parameter] or List[Dict[str, Any]]
   :param defaults: 优化器的默认超参数
   :type defaults: Dict[str, Any]

   .. method:: step(closure=None)

      执行单个优化步骤

      :param closure: 重新评估模型并返回损失的闭包
      :type closure: callable, optional
      :return: 如果提供了closure，则返回损失值，否则返回None
      :rtype: float or None

   .. method:: zero_grad(set_to_none=False)

      将所有参数的梯度设置为零

      :param set_to_none: 是否将梯度设置为None而不是零
      :type set_to_none: bool, optional

   .. method:: add_param_group(param_group)

      向优化器添加参数组

      :param param_group: 要添加的参数组
      :type param_group: Dict[str, Any]

   .. method:: state_dict()

      返回优化器的状态字典

      :return: 优化器状态
      :rtype: Dict[str, Any]

   .. method:: load_state_dict(state_dict)

      加载优化器状态

      :param state_dict: 要加载的状态字典
      :type state_dict: Dict[str, Any]

.. class:: riemann.optim.GD(params, lr=0.01, weight_decay=0.0)

   梯度下降（Gradient Descent）优化器

   :param params: 待优化参数的迭代器或定义参数组的字典列表
   :type params: Iterable[riemann.TN or riemann.nn.Parameter] or List[Dict[str, Any]]
   :param lr: 学习率
   :type lr: float, optional
   :param weight_decay: 权重衰减（L2正则化）系数
   :type weight_decay: float, optional

   .. method:: step()

      执行单个优化步骤

.. class:: riemann.optim.SGD(params, lr=0.01, momentum=0.0, weight_decay=0.0, dampening=0.0, nesterov=False)

   随机梯度下降（Stochastic Gradient Descent）优化器

   :param params: 待优化参数的迭代器或定义参数组的字典列表
   :type params: Iterable[riemann.TN or riemann.nn.Parameter] or List[Dict[str, Any]]
   :param lr: 学习率
   :type lr: float, optional
   :param momentum: 动量因子
   :type momentum: float, optional
   :param weight_decay: 权重衰减（L2正则化）系数
   :type weight_decay: float, optional
   :param dampening: 动量抑制系数
   :type dampening: float, optional
   :param nesterov: 是否启用Nesterov动量
   :type nesterov: bool, optional

   .. method:: step(closure=None)

      执行单个优化步骤

      :param closure: 重新评估模型并返回损失的闭包
      :type closure: callable, optional
      :return: 如果提供了closure，则返回损失值，否则返回None
      :rtype: float or None

.. class:: riemann.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

   Adam（Adaptive Moment Estimation）优化器

   :param params: 待优化参数的迭代器或定义参数组的字典列表
   :type params: Iterable[riemann.TN or riemann.nn.Parameter] or List[Dict[str, Any]]
   :param lr: 学习率
   :type lr: float, optional
   :param betas: 用于计算梯度及其平方的运行平均值的系数
   :type betas: Tuple[float, float], optional
   :param eps: 添加到分母以提高数值稳定性的项
   :type eps: float, optional
   :param weight_decay: 权重衰减（L2正则化）系数
   :type weight_decay: float, optional
   :param amsgrad: 是否使用AMSGrad变体
   :type amsgrad: bool, optional

   .. method:: step()

      执行单个优化步骤

.. class:: riemann.optim.Adagrad(params, lr=0.01, lr_decay=0.0, weight_decay=0.0, initial_accumulator_value=0.0, eps=1e-10)

   Adagrad（Adaptive Gradient Algorithm）优化器

   :param params: 待优化参数的迭代器或定义参数组的字典列表
   :type params: Iterable[riemann.TN or riemann.nn.Parameter] or List[Dict[str, Any]]
   :param lr: 学习率
   :type lr: float, optional
   :param lr_decay: 学习率衰减
   :type lr_decay: float, optional
   :param weight_decay: 权重衰减（L2正则化）系数
   :type weight_decay: float, optional
   :param initial_accumulator_value: 累加器的初始值
   :type initial_accumulator_value: float, optional
   :param eps: 添加到分母以提高数值稳定性的项
   :type eps: float, optional

   .. method:: step()

      执行单个优化步骤

.. class:: riemann.optim.LBFGS(params, lr=1.0, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)

   L-BFGS（Limited-memory Broyden-Fletcher-Goldfarb-Shanno）优化器

   :param params: 待优化参数的迭代器或定义参数组的字典列表
   :type params: Iterable[riemann.TN or riemann.nn.Parameter] or List[Dict[str, Any]]
   :param lr: 学习率
   :type lr: float, optional
   :param max_iter: 每个优化步骤的最大迭代次数
   :type max_iter: int, optional
   :param max_eval: 每个优化步骤的最大函数评估次数
   :type max_eval: int, optional
   :param tolerance_grad: 梯度收敛阈值
   :type tolerance_grad: float, optional
   :param tolerance_change: 参数变化收敛阈值
   :type tolerance_change: float, optional
   :param history_size: 历史缓冲区大小
   :type history_size: int, optional
   :param line_search_fn: 线搜索函数
   :type line_search_fn: callable, optional

   .. method:: step(closure)

      执行单个优化步骤

      :param closure: 重新评估模型并返回损失的闭包
      :type closure: callable
      :return: 损失值
      :rtype: float

.. class:: riemann.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False)

   AdamW（Adam with Weight Decay）优化器

   Adam的改进版本，将权重衰减作为独立的正则化项处理，而非Adam中的梯度修改。
   这使得权重衰减能够更有效地作为L2正则化，避免了Adam中原有的权重衰减副作用。

   :param params: 待优化参数的迭代器或定义参数组的字典列表
   :type params: Iterable[riemann.TN or riemann.nn.Parameter] or List[Dict[str, Any]]
   :param lr: 学习率
   :type lr: float, optional
   :param betas: 用于计算梯度及其平方的运行平均值的系数
   :type betas: Tuple[float, float], optional
   :param eps: 添加到分母以提高数值稳定性的项
   :type eps: float, optional
   :param weight_decay: 权重衰减（L2正则化）系数
   :type weight_decay: float, optional
   :param amsgrad: 是否使用AMSGrad变体
   :type amsgrad: bool, optional

   .. method:: step()

      执行单个优化步骤

.. class:: riemann.optim.RMSprop(params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False)

   RMSprop（Root Mean Square Propagation）优化器

   自适应学习率优化器，特别适用于递归神经网络（RNN）。
   它通过维护梯度平方的移动平均值来调整每个参数的学习率。

   :param params: 待优化参数的迭代器或定义参数组的字典列表
   :type params: Iterable[riemann.TN or riemann.nn.Parameter] or List[Dict[str, Any]]
   :param lr: 学习率
   :type lr: float, optional
   :param alpha: 平滑常数，用于计算梯度平方的指数移动平均值
   :type alpha: float, optional
   :param eps: 添加到分母以提高数值稳定性的项
   :type eps: float, optional
   :param weight_decay: 权重衰减（L2正则化）系数
   :type weight_decay: float, optional
   :param momentum: 动量因子
   :type momentum: float, optional
   :param centered: 是否使用中心化的RMSprop（使用梯度的移动平均值）
   :type centered: bool, optional

   .. method:: step()

      执行单个优化步骤

学习率调度器
~~~~~~~~~~~~

.. class:: riemann.optim.lr_scheduler.LRScheduler(optimizer, last_epoch=-1, verbose=False)

   所有学习率调度器的基类

   :param optimizer: 要调整学习率的优化器
   :type optimizer: riemann.optim.Optimizer
   :param last_epoch: 最后一个epoch的索引
   :type last_epoch: int, optional
   :param verbose: 是否打印学习率更新
   :type verbose: bool, optional

   .. method:: step(epoch=None)

      执行单个调度步骤

      :param epoch: 当前epoch索引
      :type epoch: int, optional

   .. method:: get_lr()

      返回当前epoch的学习率

      :return: 每个参数组的学习率
      :rtype: List[float]

   .. method:: get_last_lr()

      返回最后计算的学习率

      :return: 每个参数组的学习率
      :rtype: List[float]

   .. method:: state_dict()

      返回调度器的状态字典

      :return: 调度器状态
      :rtype: Dict[str, Any]

   .. method:: load_state_dict(state_dict)

      加载调度器状态

      :param state_dict: 要加载的状态字典
      :type state_dict: Dict[str, Any]

.. class:: riemann.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False)

   每隔step_size个epoch将学习率衰减一个因子

   :param optimizer: 要调整学习率的优化器
   :type optimizer: riemann.optim.Optimizer
   :param step_size: 学习率衰减的周期
   :type step_size: int
   :param gamma: 学习率衰减的乘法因子
   :type gamma: float, optional
   :param last_epoch: 最后一个epoch的索引
   :type last_epoch: int, optional
   :param verbose: 是否打印学习率更新
   :type verbose: bool, optional

.. class:: riemann.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False)

   在指定的milestones处将学习率衰减一个因子

   :param optimizer: 要调整学习率的优化器
   :type optimizer: riemann.optim.Optimizer
   :param milestones: epoch索引列表
   :type milestones: List[int]
   :param gamma: 学习率衰减的乘法因子
   :type gamma: float, optional
   :param last_epoch: 最后一个epoch的索引
   :type last_epoch: int, optional
   :param verbose: 是否打印学习率更新
   :type verbose: bool, optional

.. class:: riemann.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1, verbose=False)

   指数衰减学习率

   :param optimizer: 要调整学习率的优化器
   :type optimizer: riemann.optim.Optimizer
   :param gamma: 学习率衰减的乘法因子
   :type gamma: float
   :param last_epoch: 最后一个epoch的索引
   :type last_epoch: int, optional
   :param verbose: 是否打印学习率更新
   :type verbose: bool, optional

.. class:: riemann.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False)

   使用余弦函数退火学习率

   :param optimizer: 要调整学习率的优化器
   :type optimizer: riemann.optim.Optimizer
   :param T_max: 最大迭代次数
   :type T_max: int
   :param eta_min: 最小学习率
   :type eta_min: float, optional
   :param last_epoch: 最后一个epoch的索引
   :type last_epoch: int, optional
   :param verbose: 是否打印学习率更新
   :type verbose: bool, optional

.. class:: riemann.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)

   当指标停止改善时减少学习率

   :param optimizer: 要调整学习率的优化器
   :type optimizer: riemann.optim.Optimizer
   :param mode: 'min'或'max'之一
   :type mode: str, optional
   :param factor: 学习率衰减的乘法因子
   :type factor: float, optional
   :param patience: 学习率将在多少个没有改善的epoch后减少
   :type patience: int, optional
   :param verbose: 是否打印学习率更新
   :type verbose: bool, optional
   :param threshold: 测量新最优值的阈值
   :type threshold: float, optional
   :param threshold_mode: 'rel'或'abs'之一
   :type threshold_mode: str, optional
   :param cooldown: 学习率减少后恢复正常操作前要等待的epoch数
   :type cooldown: int, optional
   :param min_lr: 最小学习率
   :type min_lr: float or List[float], optional
   :param eps: 应用于lr的最小衰减
   :type eps: float, optional

   .. method:: step(metrics, epoch=None)

      执行单个调度步骤

      :param metrics: 要检查的指标值
      :type metrics: float
      :param epoch: 当前epoch索引
      :type epoch: int, optional

