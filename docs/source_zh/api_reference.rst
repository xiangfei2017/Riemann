API 参考
========

本节提供了 Riemann 库中所有函数、类和模块的全面参考。

张量操作
~~~~~~~~

张量创建函数
------------

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
----------

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

序列和范围函数
--------------

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

张量形状操作
------------

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

数学运算
--------

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

.. function:: riemann.exp(x)

   计算每个元素的指数。

   :param x: 输入张量
   :type x: riemann.TN
   :return: 每个元素的指数
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