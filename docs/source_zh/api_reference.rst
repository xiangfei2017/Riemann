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

比较运算
--------

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
--------

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

工具函数
--------

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

.. function:: riemann.clone(tensor)

   返回张量的副本。

   :param tensor: 输入张量
   :type tensor: riemann.TN
   :return: 张量副本
   :rtype: riemann.TN

.. function:: riemann.detach(tensor)

   从计算图中分离张量，停止梯度跟踪。

   :param tensor: 输入张量
   :type tensor: riemann.TN
   :return: 分离后的张量
   :rtype: riemann.TN

自动微分
--------

.. function:: riemann.grad(loss, variables)

   计算标量损失相对于变量的梯度。

   :param loss: 标量损失张量
   :type loss: riemann.TN
   :param variables: 要计算梯度的变量列表
   :type variables: list of riemann.TN
   :return: 梯度列表
   :rtype: list of riemann.TN

梯度计算
~~~~~~~~

.. function:: riemann.backward(loss, variables=None)

   执行反向传播以计算梯度。

   :param loss: 标量损失张量
   :type loss: riemann.TN
   :param variables: 要计算梯度的变量列表
   :type variables: list of riemann.TN, optional

.. function:: riemann.grad_check(function, inputs, eps=1e-05, atol=1e-05, rtol=0.001)

   验证函数的梯度计算是否正确。

   :param function: 要检查梯度的函数
   :type function: callable
   :param inputs: 函数的输入张量
   :type inputs: list of riemann.TN
   :param eps: 用于有限差分的微小值
   :type eps: float, optional
   :param atol: 绝对容差
   :type atol: float, optional
   :param rtol: 相对容差
   :type rtol: float, optional
   :return: 梯度是否正确
   :rtype: bool

求导功能函数
~~~~~~~~~~~~

.. function:: riemann.autograd.functional.jacobian(func, inputs)

   计算函数相对于输入的雅可比矩阵。

   :param func: 要计算雅可比矩阵的函数
   :type func: callable
   :param inputs: 函数的输入张量或张量序列
   :type inputs: riemann.TN or sequence of riemann.TN
   :return: 雅可比矩阵
   :rtype: riemann.TN

.. function:: riemann.autograd.functional.hessian(func, inputs)

   计算函数相对于输入的 Hessian 矩阵。

   :param func: 要计算 Hessian 矩阵的标量值函数
   :type func: callable
   :param inputs: 函数的输入张量或张量序列
   :type inputs: riemann.TN or sequence of riemann.TN
   :return: Hessian 矩阵
   :rtype: riemann.TN

.. function:: riemann.autograd.functional.jvp(func, inputs, v)

   计算 Jacobian 向量乘积（Jacobian-Vector Product）。

   :param func: 要计算 JVP 的函数
   :type func: callable
   :param inputs: 函数的输入张量或张量序列
   :type inputs: riemann.TN or sequence of riemann.TN
   :param v: 与 Jacobian 矩阵相乘的向量
   :type v: riemann.TN or sequence of riemann.TN
   :return: 函数输出和 JVP 值
   :rtype: tuple of (riemann.TN, riemann.TN or sequence of riemann.TN)

.. function:: riemann.autograd.functional.vjp(func, inputs, v)

   计算向量 Jacobian 乘积（Vector-Jacobian Product）。

   :param func: 要计算 VJP 的函数
   :type func: callable
   :param inputs: 函数的输入张量或张量序列
   :type inputs: riemann.TN or sequence of riemann.TN
   :param v: 与 Jacobian 矩阵相乘的向量
   :type v: riemann.TN or sequence of riemann.TN
   :return: 函数输出和 VJP 值
   :rtype: tuple of (riemann.TN, riemann.TN or sequence of riemann.TN)

.. function:: riemann.autograd.functional.hvp(func, inputs, v)

   计算 Hessian 向量乘积（Hessian-Vector Product）。

   :param func: 要计算 HVP 的标量值函数
   :type func: callable
   :param inputs: 函数的输入张量或张量序列
   :type inputs: riemann.TN or sequence of riemann.TN
   :param v: 与 Hessian 矩阵相乘的向量
   :type v: riemann.TN or sequence of riemann.TN
   :return: 函数输出和 HVP 值
   :rtype: tuple of (riemann.TN, riemann.TN or sequence of riemann.TN)

.. function:: riemann.autograd.functional.vhp(func, inputs, v)

   计算向量 Hessian 乘积（Vector-Hessian Product）。

   :param func: 要计算 VHP 的标量值函数
   :type func: callable
   :param inputs: 函数的输入张量或张量序列
   :type inputs: riemann.TN or sequence of riemann.TN
   :param v: 与 Hessian 矩阵相乘的向量
   :type v: riemann.TN or sequence of riemann.TN
   :return: 函数输出和 VHP 值
   :rtype: tuple of (riemann.TN, riemann.TN or sequence of riemann.TN)

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
~~~~~~~~~~~~

.. function:: riemann.no_grad()

   上下文管理器，用于禁用梯度计算。在这个上下文中的操作不会被记录在计算图中。

.. function:: riemann.enable_grad()

   上下文管理器，用于启用梯度计算。

.. function:: riemann.set_grad_enabled(mode)

   上下文管理器，根据 mode 参数启用或禁用梯度计算。

   :param mode: 启用梯度计算为 True，禁用为 False
   :type mode: bool

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
==========

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
~~~~~~~~

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

.. note:: LayerNorm 尚未在 Riemann 中实现。

.. class:: riemann.nn.Flatten(start_dim=1, end_dim=-1)

   展平张量维度的层，移除从 start_dim 到 end_dim 的所有维度。

   通常用于卷积层之后、全连接层之前，将多维度的卷积结果展平为一维向量。

   :param start_dim: 开始展平的维度
   :type start_dim: int, 可选
   :param end_dim: 结束展平的维度
   :type end_dim: int, 可选

激活函数
~~~~~~~~

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

.. note:: Dropout2d 和 Dropout3d 尚未在 Riemann 中实现。

损失函数
--------

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

优化器
=========

.. class:: riemann.optim.Optimizer

   所有优化器的基类。

.. class:: riemann.optim.Adagrad(params, lr=0.01, lr_decay=0.0, weight_decay=0.0, initial_accumulator_value=0.0, eps=1e-10)
  
   Adagrad（Adaptive Gradient Algorithm）优化器
   
   自适应学习率优化算法，为每个参数维护独立的学习率。
   根据历史梯度的平方和来自适应调整学习率，梯度大的参数学习率小，梯度小的参数学习率大。
   
   参数更新公式: θ = θ - (η / (√(G + ε))) * ∇θL(θ)
   其中，G是历史梯度的平方和，η是学习率，ε是数值稳定参数
   
   适用场景:
       - 处理稀疏数据的任务
       - 训练词嵌入模型和某些类型的CNN
       - 需要不同参数有不同学习率的场景
      
   初始化Adagrad优化器
   
   参数:
       params: 待优化参数组 (需包含requires_grad=True的TN对象)
       lr: 学习率 (默认: 0.01)
       lr_decay: 学习率衰减率 (默认: 0.0)
       weight_decay: 权重衰减 (L2正则化系数) (默认: 0.0)
       initial_accumulator_value: 累加器的初始值 (默认: 0.0)
       eps: 数值稳定性参数，防止除零错误 (默认: 1e-10)
   
   异常:
       ValueError: 当任何参数值无效时抛出
   
.. class:: riemann.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
   
   Adam（Adaptive Moment Estimation）优化器
   
   结合了动量法和自适应学习率的优化算法，维护每个参数的学习率和动量项。
   支持偏差校正和AMSGrad变体。
   
   参数更新公式:
       m = β1*m + (1-β1)*∇θL(θ)  # 一阶矩估计
       v = β2*v + (1-β2)*(∇θL(θ))²  # 二阶矩估计
       m̂ = m/(1-β1^t)  # 偏差校正的一阶矩估计
       v̂ = v/(1-β2^t)  # 偏差校正的二阶矩估计
       θ = θ - η*m̂/(√v̂ + ε)  # 参数更新
   
   适用场景:
       - 需要快速收敛的深度学习任务
       - 非平稳目标函数
       - 大规模数据和参数的场景
      
   初始化Adam优化器
   
   参数:
       params: 待优化参数组 (需包含requires_grad=True的TN对象)
       lr: 学习率 (默认1e-3)
       betas: 用于计算一阶和二阶矩估计的系数，格式为(beta1, beta2) (默认(0.9, 0.999))
       eps: 数值稳定性参数，防止除零错误 (默认1e-8)
       weight_decay: 权重衰减系数 (默认0)
       amsgrad: 是否使用AMSGrad变体 (默认False)
   
   异常:
       ValueError: 当学习率、betas、eps或权重衰减系数为负数时抛出
   
.. class:: riemann.optim.GD(params, lr=0.01, weight_decay=0.0)
   
   普通梯度下降（Gradient Descent）优化器
   
   最简单的优化算法，每次迭代都沿着负梯度方向更新参数。
   支持权重衰减（L2正则化）以防止过拟合。
   
   参数更新公式: θ = θ - η * ∇θL(θ)
   其中，θ是参数，η是学习率，∇θL(θ)是损失函数关于参数的梯度
   
   适用场景:
       - 小规模数据集和简单模型
       - 作为其他复杂优化算法的基础对比
      
   初始化梯度下降优化器
   
   参数:
       params: 待优化参数组 (需包含requires_grad=True的TN对象)
       lr: 学习率 (默认0.01)
       weight_decay: L2正则化系数 (默认0.0)
   
   异常:
       ValueError: 当学习率或权重衰减系数为负数时抛出
   
.. class:: riemann.optim.LBFGS(params, lr=1.0, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)
   
   L-BFGS（Limited-memory Broyden–Fletcher–Goldfarb–Shanno）优化器
   
   一种拟牛顿优化算法，通过维护有限数量的历史梯度和参数变化来近似Hessian矩阵的逆。
   相比于标准BFGS，它使用更少的内存，适用于大规模优化问题。
   
   注意：与其他优化器不同，LBFGS需要一个闭包函数来重新计算损失和梯度。
   
   参数:
       params: 待优化的参数组
       lr: 学习率（默认: 1.0）
       max_iter: 每个优化步骤中的最大迭代次数（默认: 20）
       max_eval: 每个优化步骤中的最大函数评估次数（默认: None，即max_iter * 1.25）
       tolerance_grad: 梯度收敛阈值（默认: 1e-5）
       tolerance_change: 参数变化收敛阈值（默认: 1e-9）
       history_size: 历史记录大小（默认: 100）
       line_search_fn: 线搜索函数（默认: None，使用内置的强Wolfe条件线搜索）
      
   初始化LBFGS优化器
   
   参数:
       params: 待优化参数组
       lr: 学习率
       max_iter: 每个优化步骤中的最大迭代次数
       max_eval: 每个优化步骤中的最大函数评估次数
       tolerance_grad: 梯度收敛阈值
       tolerance_change: 参数变化收敛阈值
       history_size: 历史记录大小
       line_search_fn: 线搜索函数
   
.. class:: riemann.optim.SGD(params: Union[Iterable[Union[riemann.tensordef.TN, riemann.nn.module.Parameter]], List[Dict[str, Any]]], lr: float = 0.01, momentum: float = 0.0, weight_decay: float = 0.0, dampening: float = 0.0, nesterov: bool = False) -> None
   
   随机梯度下降（Stochastic Gradient Descent）优化器
   
   梯度下降的随机版本，每次迭代使用一个批次的数据计算梯度。
   支持动量（momentum）以加速收敛和减少震荡，支持Nesterov动量。
   
   参数更新公式: 
       - 标准动量: v = μ*v + ∇θL(θ), θ = θ - η*v
       - Nesterov动量: v = μ*v + ∇θL(θ) + μ*η*∇θL(θ), θ = θ - η*v
   
   适用场景:
       - 大规模数据集训练
       - 需要逃离局部最优解的复杂优化问题
       - 大多数深度学习任务的默认选择
      
   初始化随机梯度下降优化器
   
   参数:
       params: 待优化参数组 (需包含requires_grad=True的TN对象)
       lr: 学习率 (默认0.01)
       momentum: 动量系数 (默认0.0)
       weight_decay: L2正则化系数 (默认0.0)
       dampening: 动量抑制系数 (默认0.0)
       nesterov: 是否启用Nesterov动量 (默认False)
   
   异常:
       ValueError: 当学习率、动量系数、权重衰减系数或抑制系数为负数时抛出
       ValueError: 当启用nesterov但momentum为0时抛出
   
损失函数关于参数的梯度
   
   适用场景:
       - 小规模数据集和简单模型
       - 作为其他复杂优化算法的基础对比
      
   初始化梯度下降优化器
   
   参数:
       params: 待优化参数组 (需包含requires_grad=True的TN对象)
       lr: 学习率 (默认0.01)
       weight_decay: L2正则化系数 (默认0.0)
   
   异常:
       ValueError: 当学习率或权重衰减系数为负数时抛出
