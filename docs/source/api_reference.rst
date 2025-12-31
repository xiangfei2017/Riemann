API Reference
=============

This section provides a comprehensive reference for all functions, classes, and modules in the Riemann library.

Tensor Operations
~~~~~~~~~~~~~~~~~

Tensor Creation Functions
--------------------------

.. function:: riemann.tensor(data, dtype=None, requires_grad=False)

   Creates a tensor from data.

   :param data: Data to initialize the tensor
   :type data: array_like
   :param dtype: Desired data type of the tensor
   :type dtype: numpy.dtype, optional
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :return: A tensor containing the given data
   :rtype: riemann.TN

.. function:: riemann.zeros(*shape, dtype=None, requires_grad=False)

   Creates a tensor filled with zeros.

   :param shape: Shape of the tensor
   :type shape: int or tuple of ints
   :param dtype: Desired data type of the tensor
   :type dtype: numpy.dtype, optional
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :return: A tensor filled with zeros
   :rtype: riemann.TN

.. function:: riemann.ones(*shape, dtype=None, requires_grad=False)

   Creates a tensor filled with ones.

   :param shape: Shape of the tensor
   :type shape: int or tuple of ints
   :param dtype: Desired data type of the tensor
   :type dtype: numpy.dtype, optional
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :return: A tensor filled with ones
   :rtype: riemann.TN

.. function:: riemann.empty(*shape, dtype=None, requires_grad=False)

   Creates an uninitialized tensor.

   :param shape: Shape of the tensor
   :type shape: int or tuple of ints
   :param dtype: Desired data type of the tensor
   :type dtype: numpy.dtype, optional
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :return: An uninitialized tensor
   :rtype: riemann.TN

.. function:: riemann.full(*shape, fill_value, dtype=None, requires_grad=False)

   Creates a tensor filled with a specific value.

   :param shape: Shape of the tensor
   :type shape: int or tuple of ints
   :param fill_value: Value to fill the tensor with
   :type fill_value: scalar
   :param dtype: Desired data type of the tensor
   :type dtype: numpy.dtype, optional
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :return: A tensor filled with the specified value
   :rtype: riemann.TN

.. function:: riemann.eye(n, m=None, dtype=None, requires_grad=False)

   Creates a 2-D tensor with ones on the diagonal and zeros elsewhere.

   :param n: Number of rows
   :type n: int
   :param m: Number of columns (default is n)
   :type m: int, optional
   :param dtype: Desired data type of the tensor
   :type dtype: numpy.dtype, optional
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :return: A 2-D tensor with ones on the diagonal
   :rtype: riemann.TN

.. function:: riemann.zeros_like(tsr, dtype=None, requires_grad=False)

   Creates a tensor of zeros with the same shape as the input tensor.

   :param tsr: Reference tensor
   :type tsr: riemann.TN
   :param dtype: Desired data type of the tensor
   :type dtype: numpy.dtype, optional
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :return: A tensor of zeros with the same shape
   :rtype: riemann.TN

.. function:: riemann.ones_like(tsr, dtype=None, requires_grad=False)

   Creates a tensor of ones with the same shape as the input tensor.

   :param tsr: Reference tensor
   :type tsr: riemann.TN
   :param dtype: Desired data type of the tensor
   :type dtype: numpy.dtype, optional
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :return: A tensor of ones with the same shape
   :rtype: riemann.TN

.. function:: riemann.empty_like(tsr, dtype=None, requires_grad=False)

   Creates an uninitialized tensor with the same shape as the input tensor.

   :param tsr: Reference tensor
   :type tsr: riemann.TN
   :param dtype: Desired data type of the tensor
   :type dtype: numpy.dtype, optional
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :return: An uninitialized tensor with the same shape
   :rtype: riemann.TN

.. function:: riemann.full_like(tsr, fill_value, dtype=None, requires_grad=False)

   Creates a tensor filled with a value with the same shape as the input tensor.

   :param tsr: Reference tensor
   :type tsr: riemann.TN
   :param fill_value: Value to fill the tensor with
   :type fill_value: scalar
   :param dtype: Desired data type of the tensor
   :type dtype: numpy.dtype, optional
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :return: A tensor filled with the specified value
   :rtype: riemann.TN

Random Number Generation
------------------------

.. function:: riemann.rand(*size, requires_grad=False, dtype=None)

   Creates a tensor filled with random numbers from a uniform distribution on [0, 1).

   :param size: Shape of the tensor
   :type size: int or tuple of ints
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :param dtype: Desired data type of the tensor
   :type dtype: numpy.dtype, optional
   :return: A tensor filled with random values
   :rtype: riemann.TN

.. function:: riemann.randn(*size, requires_grad=False, dtype=None)

   Creates a tensor filled with random numbers from a standard normal distribution.

   :param size: Shape of the tensor
   :type size: int or tuple of ints
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :param dtype: Desired data type of the tensor
   :type dtype: numpy.dtype, optional
   :return: A tensor filled with random values
   :rtype: riemann.TN

.. function:: riemann.randint(low, high, size, requires_grad=False, dtype=int64)

   Creates a tensor filled with random integers from low (inclusive) to high (exclusive).

   :param low: Lowest integer to be drawn
   :type low: int
   :param high: One above the highest integer to be drawn
   :type high: int
   :param size: Shape of the tensor
   :type size: int or tuple of ints
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :param dtype: Desired data type of the tensor
   :type dtype: numpy.dtype, optional
   :return: A tensor filled with random integers
   :rtype: riemann.TN

.. function:: riemann.randperm(n, requires_grad=False, dtype=int64)

   Creates a tensor containing the numbers 0 to n-1 in random order.

   :param n: Upper bound (exclusive)
   :type n: int
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :param dtype: Desired data type of the tensor
   :type dtype: numpy.dtype, optional
   :return: A tensor with random permutation of integers
   :rtype: riemann.TN

.. function:: riemann.normal(mean, std, size, dtype=None)

   Creates a tensor filled with random numbers from a normal distribution.

   :param mean: Mean of the normal distribution
   :type mean: float
   :param std: Standard deviation of the normal distribution
   :type std: float
   :param size: Shape of the tensor
   :type size: int or tuple of ints
   :param dtype: Desired data type of the tensor
   :type dtype: numpy.dtype, optional
   :return: A tensor filled with random values
   :rtype: riemann.TN

Sequential and Range Functions
------------------------------

.. function:: riemann.arange(start, end=None, step=1.0, dtype=None, requires_grad=False)

   Creates a 1-D tensor with values from start to end with step.

   :param start: Starting value
   :type start: float
   :param end: End value (exclusive)
   :type end: float, optional
   :param step: Spacing between values
   :type step: float, optional
   :param dtype: Desired data type of the tensor
   :type dtype: numpy.dtype, optional
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :return: A 1-D tensor with evenly spaced values
   :rtype: riemann.TN

.. function:: riemann.linspace(start, end, steps=100, endpoint=True, dtype=None, requires_grad=False)

   Creates a 1-D tensor with evenly spaced values over a given interval.

   :param start: Starting value
   :type start: float
   :param end: End value
   :type end: float
   :param steps: Number of samples to generate
   :type steps: int, optional
   :param endpoint: Whether to include the end value
   :type endpoint: bool, optional
   :param dtype: Desired data type of the tensor
   :type dtype: numpy.dtype, optional
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :return: A 1-D tensor with evenly spaced values
   :rtype: riemann.TN

Tensor Shape Operations
-----------------------

.. function:: riemann.reshape(input, shape)

   Returns a tensor with the same data but a different shape.

   :param input: Input tensor
   :type input: riemann.TN
   :param shape: New shape
   :type shape: tuple of ints
   :return: Tensor with new shape
   :rtype: riemann.TN

.. function:: riemann.squeeze(input, dim=None)

   Removes dimensions of size 1 from the shape of a tensor.

   :param input: Input tensor
   :type input: riemann.TN
   :param dim: Dimension to squeeze
   :type dim: int, optional
   :return: Tensor with squeezed dimensions
   :rtype: riemann.TN

.. function:: riemann.unsqueeze(input, dim)

   Inserts a dimension of size 1 at the specified position.

   :param input: Input tensor
   :type input: riemann.TN
   :param dim: Dimension to unsqueeze
   :type dim: int
   :return: Tensor with unsqueezed dimension
   :rtype: riemann.TN

.. function:: riemann.transpose(input, dim0, dim1)

   Swaps two dimensions of a tensor.

   :param input: Input tensor
   :type input: riemann.TN
   :param dim0: First dimension to swap
   :type dim0: int
   :param dim1: Second dimension to swap
   :type dim1: int
   :return: Tensor with swapped dimensions
   :rtype: riemann.TN

.. function:: riemann.broadcast_to(input, size)

   Broadcasts a tensor to a new shape.

   :param input: Input tensor
   :type input: riemann.TN
   :param size: Target shape
   :type size: tuple of ints
   :return: Broadcasted tensor
   :rtype: riemann.TN

.. function:: riemann.flip(input, dims)

   Reverse the order of elements along specified dimensions.

   :param input: Input tensor
   :type input: riemann.TN
   :param dims: Dimensions to flip
   :type dims: list or tuple of int
   :return: Flipped tensor
   :rtype: riemann.TN

.. function:: riemann.split(ts, split_indices, dim=0)

   Splits a tensor into multiple sub-tensors.

   :param ts: Input tensor
   :type ts: riemann.TN
   :param split_indices: Indices at which to split
   :type split_indices: int or list of ints
   :param dim: Dimension along which to split
   :type dim: int, optional
   :return: List of tensors
   :rtype: list of riemann.TN

.. function:: riemann.stack(tensors, dim=0)

   Stacks tensors along a new dimension.

   :param tensors: Sequence of tensors to stack
   :type tensors: sequence of riemann.TN
   :param dim: Dimension to insert
   :type dim: int, optional
   :return: Stacked tensor
   :rtype: riemann.TN

.. function:: riemann.cat(tensors, dim=0)

   Concatenates tensors along an existing dimension.

   :param tensors: Sequence of tensors to concatenate
   :type tensors: sequence of riemann.TN
   :param dim: Dimension along which to concatenate
   :type dim: int, optional
   :return: Concatenated tensor
   :rtype: riemann.TN

.. function:: riemann.concatenate(tensors, dim=0)

   Concatenates tensors along an existing dimension.

   :param tensors: Sequence of tensors to concatenate
   :type tensors: sequence of riemann.TN
   :param dim: Dimension along which to concatenate
   :type dim: int, optional
   :return: Concatenated tensor
   :rtype: riemann.TN

.. function:: riemann.vstack(tensors)

   Stacks tensors vertically (row-wise).

   :param tensors: Sequence of tensors to stack
   :type tensors: sequence of riemann.TN
   :return: Vertically stacked tensor
   :rtype: riemann.TN

.. function:: riemann.hstack(tensors)

   Stacks tensors horizontally (column-wise).

   :param tensors: Sequence of tensors to stack
   :type tensors: sequence of riemann.TN
   :return: Horizontally stacked tensor
   :rtype: riemann.TN

Mathematical Operations
-----------------------

.. function:: riemann.matmul(input, other)

   Matrix multiplication of two tensors.

   :param input: First tensor
   :type input: riemann.TN
   :param other: Second tensor
   :type other: riemann.TN
   :return: Matrix product of the tensors
   :rtype: riemann.TN

.. function:: riemann.dot(x, y)

   Computes the dot product of two tensors.

   :param x: First tensor
   :type x: riemann.TN
   :param y: Second tensor
   :type y: riemann.TN
   :return: Dot product result
   :rtype: riemann.TN

.. function:: riemann.sum(x, dim=None, keepdim=False)

   Computes the sum of elements across dimensions.

   :param x: Input tensor
   :type x: riemann.TN
   :param dim: Dimension(s) to sum over
   :type dim: int or tuple of ints, optional
   :param keepdim: Whether to keep the reduced dimensions
   :type keepdim: bool, optional
   :return: Sum of elements
   :rtype: riemann.TN

.. function:: riemann.prod(x, dim=None, keepdim=False)

   Computes the product of elements across dimensions.

   :param x: Input tensor
   :type x: riemann.TN
   :param dim: Dimension(s) to multiply over
   :type dim: int or tuple of ints, optional
   :param keepdim: Whether to keep the reduced dimensions
   :type keepdim: bool, optional
   :return: Product of elements
   :rtype: riemann.TN

.. function:: riemann.mean(x, dim=None, keepdim=False)

   Computes the mean of elements across dimensions.

   :param x: Input tensor
   :type x: riemann.TN
   :param dim: Dimension(s) to average over
   :type dim: int or tuple of ints, optional
   :param keepdim: Whether to keep the reduced dimensions
   :type keepdim: bool, optional
   :return: Mean of elements
   :rtype: riemann.TN

.. function:: riemann.var(x, dim=None, unbiased=True, keepdim=False)

   Computes the variance of elements across dimensions.

   :param x: Input tensor
   :type x: riemann.TN
   :param dim: Dimension(s) to compute variance over
   :type dim: int or tuple of ints, optional
   :param unbiased: Whether to use unbiased estimation
   :type unbiased: bool, optional
   :param keepdim: Whether to keep the reduced dimensions
   :type keepdim: bool, optional
   :return: Variance of elements
   :rtype: riemann.TN

.. function:: riemann.std(x, dim=None, unbiased=True, keepdim=False)

   Computes the standard deviation of elements across dimensions.

   :param x: Input tensor
   :type x: riemann.TN
   :param dim: Dimension(s) to compute standard deviation over
   :type dim: int or tuple of ints, optional
   :param unbiased: Whether to use unbiased estimation
   :type unbiased: bool, optional
   :param keepdim: Whether to keep the reduced dimensions
   :type keepdim: bool, optional
   :return: Standard deviation of elements
   :rtype: riemann.TN

.. function:: riemann.norm(x, p="fro", dim=None, keepdim=False)

   Computes the norm of a tensor.

   :param x: Input tensor
   :type x: riemann.TN
   :param p: Norm order
   :type p: int, float, str, optional
   :param dim: Dimension(s) to compute norm over
   :type dim: int or tuple of ints, optional
   :param keepdim: Whether to keep the reduced dimensions
   :type keepdim: bool, optional
   :return: Norm of the tensor
   :rtype: riemann.TN

.. function:: riemann.max(x, dim=None, keepdim=False, *, out=None)

   Computes the maximum value of elements across dimensions.

   :param x: Input tensor
   :type x: riemann.TN
   :param dim: Dimension(s) to find maximum over
   :type dim: int, optional
   :param keepdim: Whether to keep the reduced dimensions
   :type keepdim: bool, optional
   :param out: Output tensor
   :type out: riemann.TN, optional
   :return: Maximum value
   :rtype: riemann.TN

.. function:: riemann.min(x, dim=None, keepdim=False, *, out=None)

   Computes the minimum value of elements across dimensions.

   :param x: Input tensor
   :type x: riemann.TN
   :param dim: Dimension(s) to find minimum over
   :type dim: int, optional
   :param keepdim: Whether to keep the reduced dimensions
   :type keepdim: bool, optional
   :param out: Output tensor
   :type out: riemann.TN, optional
   :return: Minimum value
   :rtype: riemann.TN

.. function:: riemann.abs(x)

   Computes the absolute value of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Absolute value of each element
   :rtype: riemann.TN

.. function:: riemann.sqrt(x)

   Computes the square root of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Square root of each element
   :rtype: riemann.TN

.. function:: riemann.pow(input, exponent)

   Raises each element to a power.

   :param input: Input tensor
   :type input: riemann.TN
   :param exponent: Exponent value
   :type exponent: riemann.TN or scalar
   :return: Input tensor raised to the power
   :rtype: riemann.TN

.. function:: riemann.log(x)

   Computes the natural logarithm of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Natural logarithm of each element
   :rtype: riemann.TN

.. function:: riemann.log1p(x)

   Computes the natural logarithm of one plus each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Natural logarithm of one plus each element
   :rtype: riemann.TN

.. function:: riemann.exp(x)

   Computes the exponential of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Exponential of each element
   :rtype: riemann.TN

.. function:: riemann.sin(x)

   Computes the sine of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Sine of each element
   :rtype: riemann.TN

.. function:: riemann.cos(x)

   Computes the cosine of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Cosine of each element
   :rtype: riemann.TN

.. function:: riemann.tan(x)

   Computes the tangent of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Tangent of each element
   :rtype: riemann.TN

.. function:: riemann.cot(x)

   Computes the cotangent of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Cotangent of each element
   :rtype: riemann.TN

.. function:: riemann.sec(x)

   Computes the secant of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Secant of each element
   :rtype: riemann.TN

.. function:: riemann.csc(x)

   Computes the cosecant of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Cosecant of each element
   :rtype: riemann.TN

.. function:: riemann.asin(x)

   Computes the inverse sine of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Inverse sine of each element
   :rtype: riemann.TN

.. function:: riemann.acos(x)

   Computes the inverse cosine of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Inverse cosine of each element
   :rtype: riemann.TN

.. function:: riemann.atan(x)

   Computes the inverse tangent of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Inverse tangent of each element
   :rtype: riemann.TN

.. function:: riemann.sinh(x)

   Computes the hyperbolic sine of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Hyperbolic sine of each element
   :rtype: riemann.TN

.. function:: riemann.cosh(x)

   Computes the hyperbolic cosine of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Hyperbolic cosine of each element
   :rtype: riemann.TN

.. function:: riemann.tanh(x)

   Computes the hyperbolic tangent of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Hyperbolic tangent of each element
   :rtype: riemann.TN

.. function:: riemann.coth(x)

   Computes the hyperbolic cotangent of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Hyperbolic cotangent of each element
   :rtype: riemann.TN

.. function:: riemann.sech(x)

   Computes the hyperbolic secant of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Hyperbolic secant of each element
   :rtype: riemann.TN

.. function:: riemann.csch(x)

   Computes the hyperbolic cosecant of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Hyperbolic cosecant of each element
   :rtype: riemann.TN

.. function:: riemann.arcsinh(x)

   Computes the inverse hyperbolic sine of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Inverse hyperbolic sine of each element
   :rtype: riemann.TN

.. function:: riemann.arccosh(x)

   Computes the inverse hyperbolic cosine of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Inverse hyperbolic cosine of each element
   :rtype: riemann.TN

.. function:: riemann.arctanh(x)

   Computes the inverse hyperbolic tangent of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Inverse hyperbolic tangent of each element
   :rtype: riemann.TN

.. function:: riemann.sign(x)

   Computes the sign of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Sign of each element
   :rtype: riemann.TN

.. function:: riemann.where(cond, x=None, y=None)

   Returns elements from x or y depending on cond.

   :param cond: Condition tensor
   :type cond: riemann.TN
   :param x: Tensor to select from where condition is True
   :type x: riemann.TN, optional
   :param y: Tensor to select from where condition is False
   :type y: riemann.TN, optional
   :return: Tensor with elements from x or y
   :rtype: riemann.TN

.. function:: riemann.clamp(x, min=None, max=None, out=None)

   Clamps all elements to a specified range.

   :param x: Input tensor
   :type x: riemann.TN
   :param min: Minimum value
   :type min: float, optional
   :param max: Maximum value
   :type max: float, optional
   :param out: Output tensor
   :type out: riemann.TN, optional
   :return: Tensor with elements clamped to the range
   :rtype: riemann.TN

.. function:: riemann.maximum(input, other)

   Computes the element-wise maximum of two tensors.

   :param input: First input tensor
   :type input: riemann.TN
   :param other: Second input tensor
   :type other: riemann.TN
   :return: Tensor with element-wise maximum values
   :rtype: riemann.TN

.. function:: riemann.minimum(input, other)

   Computes the element-wise minimum of two tensors.

   :param input: First input tensor
   :type input: riemann.TN
   :param other: Second input tensor
   :type other: riemann.TN
   :return: Tensor with element-wise minimum values
   :rtype: riemann.TN

.. function:: riemann.diagonal(input, offset=0, dim1=-2, dim2=-1)

   Returns the diagonal of a tensor.

   :param input: Input tensor
   :type input: riemann.TN
   :param offset: Offset of the diagonal
   :type offset: int, optional
   :param dim1: First dimension of the diagonal
   :type dim1: int, optional
   :param dim2: Second dimension of the diagonal
   :type dim2: int, optional
   :return: Diagonal of the tensor
   :rtype: riemann.TN

.. function:: riemann.diag(input, offset=0)

   Returns the diagonal of a 2-D tensor or constructs a diagonal matrix.

   :param input: Input tensor
   :type input: riemann.TN
   :param offset: Offset of the diagonal
   :type offset: int, optional
   :return: Diagonal of the tensor or diagonal matrix
   :rtype: riemann.TN

.. function:: riemann.fill_diagonal(input, value, offset=0, dim1=-2, dim2=-1)

   Fills the diagonal of a tensor with a specified value.

   :param input: Input tensor
   :type input: riemann.TN
   :param value: Value to fill the diagonal with
   :type value: scalar
   :param offset: Offset of the diagonal
   :type offset: int, optional
   :param dim1: First dimension of the diagonal
   :type dim1: int, optional
   :param dim2: Second dimension of the diagonal
   :type dim2: int, optional
   :return: Tensor with filled diagonal
   :rtype: riemann.TN

.. function:: riemann.fill_diagonal_(input, value, offset=0, dim1=-2, dim2=-1)

   In-place version of fill_diagonal.

   :param input: Input tensor
   :type input: riemann.TN
   :param value: Value to fill the diagonal with
   :type value: scalar
   :param offset: Offset of the diagonal
   :type offset: int, optional
   :param dim1: First dimension of the diagonal
   :type dim1: int, optional
   :param dim2: Second dimension of the diagonal
   :type dim2: int, optional
   :return: Input tensor with filled diagonal
   :rtype: riemann.TN

.. function:: riemann.batch_diag(v)

   Returns the batched diagonal of a tensor.

   :param v: Input tensor
   :type v: riemann.TN
   :return: Batched diagonal of the tensor
   :rtype: riemann.TN

.. function:: riemann.nonzero(input, *, as_tuple=False)

   Returns the indices of non-zero elements.

   :param input: Input tensor
   :type input: riemann.TN
   :param as_tuple: Whether to return as a tuple of tensors
   :type as_tuple: bool, optional
   :return: Indices of non-zero elements
   :rtype: riemann.TN or tuple of riemann.TN

.. function:: riemann.tril(input_tensor, diagonal=0)

   Returns the lower triangular part of a matrix.

   :param input_tensor: Input tensor
   :type input_tensor: riemann.TN
   :param diagonal: Diagonal offset
   :type diagonal: int, optional
   :return: Lower triangular part of the matrix
   :rtype: riemann.TN

.. function:: riemann.triu(input_tensor, diagonal=0)

   Returns the upper triangular part of a matrix.

   :param input_tensor: Input tensor
   :type input_tensor: riemann.TN
   :param diagonal: Diagonal offset
   :type diagonal: int, optional
   :return: Upper triangular part of the matrix
   :rtype: riemann.TN

Comparison Operations
---------------------

.. function:: riemann.equal(a, b)

   Computes element-wise equality.

   :param a: First tensor
   :type a: riemann.TN
   :param b: Second tensor
   :type b: riemann.TN
   :return: Boolean tensor indicating equality
   :rtype: bool

.. function:: riemann.not_equal(a, b)

   Computes element-wise inequality.

   :param a: First tensor
   :type a: riemann.TN
   :param b: Second tensor
   :type b: riemann.TN
   :return: Boolean tensor indicating inequality
   :rtype: bool

.. function:: riemann.allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False)

   Returns True if two tensors are element-wise equal within a tolerance.

   :param a: First tensor
   :type a: riemann.TN
   :param b: Second tensor
   :type b: riemann.TN
   :param rtol: Relative tolerance
   :type rtol: float, optional
   :param atol: Absolute tolerance
   :type atol: float, optional
   :param equal_nan: Whether to compare NaN values as equal
   :type equal_nan: bool, optional
   :return: Whether tensors are close
   :rtype: bool

.. function:: riemann.isinf(x)

   Tests element-wise for infinity.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Boolean tensor indicating infinity
   :rtype: riemann.TN

.. function:: riemann.isnan(x)

   Tests element-wise for NaN.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Boolean tensor indicating NaN
   :rtype: riemann.TN

.. function:: riemann.isreal(x)

   Tests element-wise for real numbers.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Boolean tensor indicating real numbers
   :rtype: riemann.TN

Sorting Operations
------------------

.. function:: riemann.sort(input, dim=-1, descending=False, stable=False, *, out=None)

   Sorts the elements of a tensor along a given dimension.

   :param input: Input tensor
   :type input: riemann.TN
   :param dim: Dimension to sort along
   :type dim: int, optional
   :param descending: Whether to sort in descending order
   :type descending: bool, optional
   :param stable: Whether to use stable sort
   :type stable: bool, optional
   :param out: Output tensor
   :type out: riemann.TN, optional
   :return: Sorted tensor and indices
   :rtype: tuple of (riemann.TN, riemann.TN)

.. function:: riemann.argsort(input, dim=-1, descending=False, stable=False, *, out=None)

   Returns the indices that sort a tensor along a given dimension.

   :param input: Input tensor
   :type input: riemann.TN
   :param dim: Dimension to sort along
   :type dim: int, optional
   :param descending: Whether to sort in descending order
   :type descending: bool, optional
   :param stable: Whether to use stable sort
   :type stable: bool, optional
   :param out: Output tensor
   :type out: riemann.TN, optional
   :return: Indices that sort the tensor
   :rtype: riemann.TN

.. function:: riemann.unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None)

   Returns the unique elements of a tensor.

   :param input: Input tensor
   :type input: riemann.TN
   :param sorted: Whether to return unique elements in sorted order
   :type sorted: bool, optional
   :param return_inverse: Whether to return indices to reconstruct input
   :type return_inverse: bool, optional
   :param return_counts: Whether to return counts of each unique element
   :type return_counts: bool, optional
   :param dim: Dimension to apply unique to
   :type dim: int, optional
   :return: Unique elements and optionally indices and counts
   :rtype: riemann.TN or tuple of riemann.TN

Utility Functions
------------------

.. function:: riemann.from_numpy(arr)

   Creates a tensor from a NumPy array.

   :param arr: NumPy array
   :type arr: numpy.ndarray
   :return: Tensor with the same data as the NumPy array
   :rtype: riemann.TN

.. function:: riemann.cumsum(input, dim, *, dtype=None, out=None)

   Computes the cumulative sum of elements along a dimension.

   :param input: Input tensor
   :type input: riemann.TN
   :param dim: Dimension to cumsum over
   :type dim: int
   :param dtype: Desired data type of the output
   :type dtype: numpy.dtype, optional
   :param out: Output tensor
   :type out: riemann.TN, optional
   :return: Cumulative sum of elements
   :rtype: riemann.TN

Automatic Differentiation
-------------------------

Gradient Computation
~~~~~~~~~~~~~~~~~~~~

.. function:: riemann.autograd.backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False)

   Computes gradients of tensors with respect to graph leaves.

   :param tensors: Tensors to compute gradients for
   :type tensors: riemann.TN or sequence of riemann.TN
   :param grad_tensors: Gradients with respect to tensors
   :type grad_tensors: riemann.TN or sequence of riemann.TN, optional
   :param retain_graph: Whether to retain the computation graph
   :type retain_graph: bool, optional
   :param create_graph: Whether to create a graph of the gradient computation
   :type create_graph: bool, optional

.. function:: riemann.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False)

   Computes and returns the gradient of outputs with respect to inputs.

   :param outputs: Outputs of the differentiated function
   :type outputs: riemann.TN or sequence of riemann.TN
   :param inputs: Inputs with respect to which to compute gradients
   :type inputs: riemann.TN or sequence of riemann.TN
   :param grad_outputs: Gradients with respect to outputs
   :type grad_outputs: riemann.TN or sequence of riemann.TN, optional
   :param retain_graph: Whether to retain the computation graph
   :type retain_graph: bool, optional
   :param create_graph: Whether to create a graph of the gradient computation
   :type create_graph: bool, optional
   :param only_inputs: Whether to compute gradients only with respect to inputs
   :type only_inputs: bool, optional
   :param allow_unused: Whether to allow unused inputs
   :type allow_unused: bool, optional
   :return: Gradients with respect to inputs
   :rtype: tuple of riemann.TN


Derivative Functions
~~~~~~~~~~~~~~~~~~~~

.. function:: riemann.autograd.functional.jacobian(func, inputs, create_graph=False, strict=False)

   Computes the Jacobian matrix of a function.

   The Jacobian matrix represents the first-order partial derivatives of a function mapping a vector to a vector.

   :param func: Function that takes tensor or tensor list and returns tensor or tensor list
   :type func: callable
   :param inputs: Input tensor(s)
   :type inputs: riemann.TN or sequence of riemann.TN
   :param create_graph: Whether to create a graph of the computation for higher-order derivatives
   :type create_graph: bool, optional
   :param strict: Whether to raise errors on unused inputs
   :type strict: bool, optional
   :return: Jacobian matrix representation
   :rtype: riemann.TN or sequence of riemann.TN

.. function:: riemann.autograd.functional.hessian(func, inputs, create_graph=False, strict=False)

   Computes the Hessian matrix of a scalar-valued function.

   The Hessian matrix represents the second-order partial derivatives of a scalar function.

   :param func: Function that takes tensor or tensor list and returns scalar tensor
   :type func: callable
   :param inputs: Input tensor(s)
   :type inputs: riemann.TN or sequence of riemann.TN
   :param create_graph: Whether to create a graph of the computation for higher-order derivatives
   :type create_graph: bool, optional
   :param strict: Whether to raise errors on unused inputs
   :type strict: bool, optional
   :return: Hessian matrix representation
   :rtype: riemann.TN or sequence of riemann.TN

.. function:: riemann.autograd.functional.jvp(func, inputs, v, create_graph=False, strict=False)

   Computes Jacobian-vector product (JVP).

   :param func: Function that takes tensor or tensor list and returns tensor or tensor list
   :type func: callable
   :param inputs: Input tensor(s)
   :type inputs: riemann.TN or sequence of riemann.TN
   :param v: Vector to multiply with Jacobian
   :type v: riemann.TN or sequence of riemann.TN
   :param create_graph: Whether to create a graph of the computation for higher-order derivatives
   :type create_graph: bool, optional
   :param strict: Whether to raise errors on unused inputs
   :type strict: bool, optional
   :return: Tuple of function output and JVP result
   :rtype: tuple

.. function:: riemann.autograd.functional.vjp(func, inputs, v=None, create_graph=False, strict=False)

   Computes vector-Jacobian product (VJP).

   :param func: Function that takes tensor or tensor list and returns tensor or tensor list
   :type func: callable
   :param inputs: Input tensor(s)
   :type inputs: riemann.TN or sequence of riemann.TN
   :param v: Vector to multiply with Jacobian
   :type v: riemann.TN or sequence of riemann.TN, optional
   :param create_graph: Whether to create a graph of the computation for higher-order derivatives
   :type create_graph: bool, optional
   :param strict: Whether to raise errors on unused inputs
   :type strict: bool, optional
   :return: Tuple of function output and VJP result
   :rtype: tuple

.. function:: riemann.autograd.functional.hvp(func, inputs, v, create_graph=False, strict=False)

   Computes Hessian-vector product (HVP).

   :param func: Function that takes tensor or tensor list and returns scalar tensor
   :type func: callable
   :param inputs: Input tensor(s)
   :type inputs: riemann.TN or sequence of riemann.TN
   :param v: Vector to multiply with Hessian
   :type v: riemann.TN or sequence of riemann.TN
   :param create_graph: Whether to create a graph of the computation for higher-order derivatives
   :type create_graph: bool, optional
   :param strict: Whether to raise errors on unused inputs
   :type strict: bool, optional
   :return: Tuple of function output and HVP result
   :rtype: tuple

.. function:: riemann.autograd.functional.vhp(func, inputs, v, create_graph=False, strict=False)

   Computes vector-Hessian product (VHP).

   :param func: Function that takes tensor or tensor list and returns scalar tensor
   :type func: callable
   :param inputs: Input tensor(s)
   :type inputs: riemann.TN or sequence of riemann.TN
   :param v: Vector to multiply with Hessian
   :type v: riemann.TN or sequence of riemann.TN
   :param create_graph: Whether to create a graph of the computation for higher-order derivatives
   :type create_graph: bool, optional
   :param strict: Whether to raise errors on unused inputs
   :type strict: bool, optional
   :return: Tuple of function output and VHP result

.. function:: riemann.autograd.functional.derivative(func, create_graph=False)

   Computes the derivative of a function.

   Returns a new function that computes the derivative of the original function func at the input point when called.
   Supports func with single or multiple tensor inputs and single/multiple tensor or scalar outputs.
   Internally implemented based on jacobian function.

   :param func: Function to compute derivative for
   :type func: callable
   :param create_graph: Whether to create a computation graph for higher-order derivatives
   :type create_graph: bool, optional
   :return: Derivative function with the same input arguments as original func
   :rtype: callable

Context Managers
~~~~~~~~~~~~~~~~~

.. class:: riemann.no_grad

   Context manager that disables gradient calculation.

   .. code-block:: python

      with riemann.no_grad():
          # Operations inside this block won't track gradients
          x = riemann.tensor([1., 2., 3.], requires_grad=True)
          y = x * 2
          print(y.requires_grad)  # False

.. class:: riemann.enable_grad

   Context manager that enables gradient calculation.

   .. code-block:: python

      with riemann.enable_grad():
          # Operations inside this block will track gradients
          x = riemann.tensor([1., 2., 3.])
          y = x * 2
          print(y.requires_grad)  # True

Neural Network Modules
----------------------

Base Classes
~~~~~~~~~~~~

.. class:: riemann.nn.Module

   Base class for all neural network modules.

   Methods:

   .. method:: forward(*input)

      Defines the computation performed at every call.

      :param input: Input tensors
      :type input: riemann.TN
      :return: Output tensor
      :rtype: riemann.TN

   .. method:: parameters(recurse=True)

      Returns an iterator over module parameters.

      :param recurse: Whether to include parameters of submodules
      :type recurse: bool
      :return: Iterator over parameters
      :rtype: iterator

   .. method:: zero_grad()

      Sets gradients of all model parameters to zero.

   .. method:: train(mode=True)

      Sets the module in training mode.

      :param mode: Whether to set training mode
      :type mode: bool

   .. method:: eval()

      Sets the module in evaluation mode.


.. class:: riemann.Parameter(data=None, requires_grad=True)

   Parameter class for trainable parameters in neural networks.

   Wrapper around tensor objects that marks them as trainable parameters, ensuring they are registered and appear in the model's parameter list.

   :param data: Initial tensor data
   :type data: riemann.Tensor, optional
   :param requires_grad: Whether this parameter requires gradient computation
   :type requires_grad: bool, optional

Container Modules
~~~~~~~~~~~~~~~~~

.. class:: riemann.nn.Sequential(*args)

   A sequential container that passes data through a sequence of modules.

   :param args: Modules to add to the container
   :type args: riemann.nn.Module

.. class:: riemann.nn.ModuleList(modules=None)

   Holds submodules in a list.

   :param modules: Modules to add to the list
   :type modules: iterable, optional

.. class:: riemann.nn.ModuleDict(modules=None)

   Holds submodules in a dictionary.

   :param modules: Modules to add to the dictionary
   :type modules: dict, optional

Linear Layers
~~~~~~~~~~~~~

.. class:: riemann.nn.Linear(in_features, out_features, bias=True)

   Applies a linear transformation to the incoming data.

   :param in_features: Size of each input sample
   :type in_features: int
   :param out_features: Size of each output sample
   :type out_features: int
   :param bias: Whether to include an additive bias
   :type bias: bool

Convolutional Layers
~~~~~~~~~~~~~~~~~~~~

.. class:: riemann.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

   Applies a 1D convolution over an input signal.

   :param in_channels: Number of channels in the input image
   :type in_channels: int
   :param out_channels: Number of channels produced by the convolution
   :type out_channels: int
   :param kernel_size: Size of the convolving kernel
   :type kernel_size: int or tuple
   :param stride: Stride of the convolution
   :type stride: int or tuple, optional
   :param padding: Zero-padding added to both sides of the input
   :type padding: int or tuple, optional
   :param dilation: Spacing between kernel elements
   :type dilation: int or tuple, optional
   :param groups: Number of blocked connections from input to output channels
   :type groups: int, optional
   :param bias: Whether to include a bias term
   :type bias: bool, optional

.. class:: riemann.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

   Applies a 2D convolution over an input image.

   :param in_channels: Number of channels in the input image
   :type in_channels: int
   :param out_channels: Number of channels produced by the convolution
   :type out_channels: int
   :param kernel_size: Size of the convolving kernel
   :type kernel_size: int or tuple
   :param stride: Stride of the convolution
   :type stride: int or tuple, optional
   :param padding: Zero-padding added to both sides of the input
   :type padding: int or tuple, optional
   :param dilation: Spacing between kernel elements
   :type dilation: int or tuple, optional
   :param groups: Number of blocked connections from input to output channels
   :type groups: int, optional
   :param bias: Whether to include a bias term
   :type bias: bool, optional

.. class:: riemann.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

   Applies a 3D convolution over an input volume.

   :param in_channels: Number of channels in the input image
   :type in_channels: int
   :param out_channels: Number of channels produced by the convolution
   :type out_channels: int
   :param kernel_size: Size of the convolving kernel
   :type kernel_size: int or tuple
   :param stride: Stride of the convolution
   :type stride: int or tuple, optional
   :param padding: Zero-padding added to both sides of the input
   :type padding: int or tuple, optional
   :param dilation: Spacing between kernel elements
   :type dilation: int or tuple, optional
   :param groups: Number of blocked connections from input to output channels
   :type groups: int, optional
   :param bias: Whether to include a bias term
   :type bias: bool, optional

Pooling Layers
~~~~~~~~~~~~~~

.. class:: riemann.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

   Applies a 1D max pooling over an input signal.

   :param kernel_size: Size of the window
   :type kernel_size: int or tuple
   :param stride: Stride of the window
   :type stride: int or tuple, optional
   :param padding: Zero-padding added to both sides of the input
   :type padding: int or tuple, optional
   :param dilation: Spacing between kernel elements
   :type dilation: int or tuple, optional
   :param return_indices: Whether to return the indices of the max values
   :type return_indices: bool, optional
   :param ceil_mode: Whether to use ceil instead of floor to compute output shape
   :type ceil_mode: bool, optional

.. class:: riemann.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

   Applies a 2D max pooling over an input image.

   :param kernel_size: Size of the window
   :type kernel_size: int or tuple
   :param stride: Stride of the window
   :type stride: int or tuple, optional
   :param padding: Zero-padding added to both sides of the input
   :type padding: int or tuple, optional
   :param dilation: Spacing between kernel elements
   :type dilation: int or tuple, optional
   :param return_indices: Whether to return the indices of the max values
   :type return_indices: bool, optional
   :param ceil_mode: Whether to use ceil instead of floor to compute output shape
   :type ceil_mode: bool, optional

.. class:: riemann.nn.AvgPool1d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)

   Applies a 1D average pooling over an input signal.

   :param kernel_size: Size of the window
   :type kernel_size: int or tuple
   :param stride: Stride of the window
   :type stride: int or tuple, optional
   :param padding: Zero-padding added to both sides of the input
   :type padding: int or tuple, optional
   :param ceil_mode: Whether to use ceil instead of floor to compute output shape
   :type ceil_mode: bool, optional
   :param count_include_pad: Whether to include zero-padding in the averaging calculation
   :type count_include_pad: bool, optional

.. class:: riemann.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)

   Applies a 2D average pooling over an input image.

   :param kernel_size: Size of the window
   :type kernel_size: int or tuple
   :param stride: Stride of the window
   :type stride: int or tuple, optional
   :param padding: Zero-padding added to both sides of the input
   :type padding: int or tuple, optional
   :param ceil_mode: Whether to use ceil instead of floor to compute output shape
   :type ceil_mode: bool, optional
   :param count_include_pad: Whether to include zero-padding in the averaging calculation
   :type count_include_pad: bool, optional
   :param divisor_override: If specified, it will be used as divisor
   :type divisor_override: int, optional

.. class:: riemann.nn.MaxPool3d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

   Applies a 3D max pooling over an input volumetric data.

   :param kernel_size: Size of the window
   :type kernel_size: int or tuple
   :param stride: Stride of the window
   :type stride: int or tuple, optional
   :param padding: Zero-padding added to all sides of the input
   :type padding: int or tuple, optional
   :param dilation: Spacing between kernel elements
   :type dilation: int or tuple, optional
   :param return_indices: Whether to return the indices of the max values
   :type return_indices: bool, optional
   :param ceil_mode: Whether to use ceil instead of floor to compute output shape
   :type ceil_mode: bool, optional

.. class:: riemann.nn.AvgPool3d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)

   Applies a 3D average pooling over an input volumetric data.

   :param kernel_size: Size of the window
   :type kernel_size: int or tuple
   :param stride: Stride of the window
   :type stride: int or tuple, optional
   :param padding: Zero-padding added to all sides of the input
   :type padding: int or tuple, optional
   :param ceil_mode: Whether to use ceil instead of floor to compute output shape
   :type ceil_mode: bool, optional
   :param count_include_pad: Whether to include zero-padding in the averaging calculation    
   :type count_include_pad: bool, optional
   :param divisor_override: If specified, it will be used as divisor
   :type divisor_override: int, optional

Normalization Layers
~~~~~~~~~~~~~~~~~~~~

.. class:: riemann.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

   Applies Batch Normalization over a 2D or 3D input.

   :param num_features: Number of features C from an expected input of size (N, C, L) or (N, C, L, S)
   :type num_features: int
   :param eps: A value added to the denominator for numerical stability
   :type eps: float, optional
   :param momentum: Value used for the running_mean and running_var computation
   :type momentum: float, optional
   :param affine: Whether to use learnable affine parameters
   :type affine: bool, optional
   :param track_running_stats: Whether to track the running mean and variance
   :type track_running_stats: bool, optional

.. class:: riemann.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

   Applies Batch Normalization over a 4D input.

   :param num_features: Number of features C from an expected input of size (N, C, H, W)
   :type num_features: int
   :param eps: A value added to the denominator for numerical stability
   :type eps: float, optional
   :param momentum: Value used for the running_mean and running_var computation
   :type momentum: float, optional
   :param affine: Whether to use learnable affine parameters
   :type affine: bool, optional
   :param track_running_stats: Whether to track the running mean and variance
   :type track_running_stats: bool, optional

.. class:: riemann.nn.BatchNorm3d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

   Three-dimensional batch normalization layer.

   Normalizes the channel dimension of a 5D input tensor (N, C, D, H, W) to have zero mean and unit variance,
   improving training convergence and model generalization capabilities.

   :param num_features: Number of features (channel dimension)
   :type num_features: int
   :param eps: Small value to avoid division by zero
   :type eps: float, optional
   :param momentum: Momentum for running statistics
   :type momentum: float, optional
   :param affine: Whether to include learnable affine parameters
   :type affine: bool, optional
   :param track_running_stats: Whether to track running mean and variance
   :type track_running_stats: bool, optional

Note: LayerNorm is not yet implemented in Riemann.

.. class:: riemann.nn.Flatten(start_dim=1, end_dim=-1)

   Layer that flattens tensor dimensions by removing all dimensions from start_dim to end_dim.

   Typically used after convolution layers and before fully connected layers to convert multi-dimensional convolution outputs into 1D vectors.

   :param start_dim: Starting dimension to flatten
   :type start_dim: int, optional
   :param end_dim: Ending dimension to flatten
   :type end_dim: int, optional

Activation Functions
~~~~~~~~~~~~~~~~~~~~

.. class:: riemann.nn.ReLU(inplace=False)

   Applies the Rectified Linear Unit activation function element-wise.

   ReLU(x) = max(0, x)

   :param inplace: Whether to do the operation in-place
   :type inplace: bool, optional

.. class:: riemann.nn.LeakyReLU(negative_slope=0.01, inplace=False)

   Applies the Leaky Rectified Linear Unit activation function element-wise.

   LeakyReLU(x) = max(x, negative_slope * x)

   :param negative_slope: Slope of the negative region
   :type negative_slope: float, optional
   :param inplace: Whether to do the operation in-place
   :type inplace: bool, optional

.. class:: riemann.nn.RReLU(lower=0.125, upper=0.3333333333333333, inplace=False)

   Applies the Randomized Leaky Rectified Linear Unit activation function element-wise.

   RReLU(x) = max(x, a * x) where a is sampled uniformly from [lower, upper]

   :param lower: Lower bound of the uniform distribution
   :type lower: float, optional
   :param upper: Upper bound of the uniform distribution
   :type upper: float, optional
   :param inplace: Whether to do the operation in-place
   :type inplace: bool, optional

.. class:: riemann.nn.PReLU(num_parameters=1, init=0.25)

   Applies the Parametric Rectified Linear Unit activation function element-wise.

   PReLU(x) = max(x, a * x) where a is a learnable parameter

   :param num_parameters: Number of learnable parameters
   :type num_parameters: int, optional
   :param init: Initial value of the parameter
   :type init: float, optional

.. class:: riemann.nn.Sigmoid

   Applies the element-wise function: sigmoid(x) = 1 / (1 + exp(-x))

.. class:: riemann.nn.Tanh

   Applies the element-wise function: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

.. class:: riemann.nn.Softmax(dim=None)

   Applies the Softmax function to an n-dimensional input tensor.

   :param dim: Dimension along which Softmax will be computed
   :type dim: int, optional

.. class:: riemann.nn.LogSoftmax(dim=None)

   Applies the LogSoftmax function to an n-dimensional input tensor.

   :param dim: Dimension along which LogSoftmax will be computed
   :type dim: int, optional

.. class:: riemann.nn.GELU

   Applies the Gaussian Error Linear Unit activation function element-wise.

   GELU(x) = x * Φ(x), where Φ is the cumulative distribution function of the standard normal distribution

.. class:: riemann.nn.Softplus(beta=1, threshold=20)

   Applies the Softplus activation function element-wise.

   Softplus(x) = (1 / beta) * log(1 + exp(beta * x))

   :param beta: Slope of the linear part
   :type beta: float, optional
   :param threshold: Threshold value for numerical stability
   :type threshold: float, optional

Dropout Layers
~~~~~~~~~~~~~~

.. class:: riemann.nn.Dropout(p=0.5, inplace=False)

   Randomly zeroes some of the elements of the input tensor with probability p.

   :param p: Probability of an element to be zeroed
   :type p: float, optional
   :param inplace: Whether to do the operation in-place
   :type inplace: bool, optional

Note: Dropout2d and Dropout3d are not yet implemented in Riemann.

Loss Functions
--------------

.. class:: riemann.nn.L1Loss(size_average=None, reduce=None, reduction='mean')

   Creates a criterion that measures the mean absolute error between each element in the input and target.

   :param size_average: Deprecated
   :type size_average: bool, optional
   :param reduce: Deprecated
   :type reduce: bool, optional
   :param reduction: Specifies the reduction to apply to the output
   :type reduction: str, optional

.. class:: riemann.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

   Creates a criterion that measures the mean squared error between each element in the input and target.

   :param size_average: Deprecated
   :type size_average: bool, optional
   :param reduce: Deprecated
   :type reduce: bool, optional
   :param reduction: Specifies the reduction to apply to the output
   :type reduction: str, optional

.. class:: riemann.nn.NLLLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

   Creates a criterion that measures the negative log likelihood loss.

   :param weight: Manual rescaling weight given to each class
   :type weight: riemann.TN, optional
   :param size_average: Deprecated
   :type size_average: bool, optional
   :param ignore_index: Specifies a target value that is ignored
   :type ignore_index: int, optional
   :param reduce: Deprecated
   :type reduce: bool, optional
   :param reduction: Specifies the reduction to apply to the output
   :type reduction: str, optional

.. class:: riemann.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

   Combines LogSoftmax and NLLLoss in one single class.

   :param weight: Manual rescaling weight given to each class
   :type weight: riemann.TN, optional
   :param size_average: Deprecated
   :type size_average: bool, optional
   :param ignore_index: Specifies a target value that is ignored
   :type ignore_index: int, optional
   :param reduce: Deprecated
   :type reduce: bool, optional
   :param reduction: Specifies the reduction to apply to the output
   :type reduction: str, optional

.. class:: riemann.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')

   Creates a criterion that measures the binary cross entropy between the target and the output.

   :param weight: Manual rescaling weight given to each batch element
   :type weight: riemann.TN, optional
   :param size_average: Deprecated
   :type size_average: bool, optional
   :param reduce: Deprecated
   :type reduce: bool, optional
   :param reduction: Specifies the reduction to apply to the output
   :type reduction: str, optional

.. class:: riemann.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean')

   Computes binary cross entropy between the target and the input logits.

   :param weight: Manual rescaling weight given to each batch element
   :type weight: riemann.TN, optional
   :param size_average: Deprecated
   :type size_average: bool, optional
   :param reduce: Deprecated
   :type reduce: bool, optional
   :param reduction: Specifies the reduction to apply to the output
   :type reduction: str, optional

.. class:: riemann.nn.HuberLoss(delta=1.0, size_average=None, reduce=None, reduction='mean')

   Creates a criterion that uses a squared term if the absolute element-wise error is less than delta, otherwise an L1 term.

   :param delta: Threshold where the loss function changes from quadratic to linear
   :type delta: float, optional
   :param size_average: Deprecated
   :type size_average: bool, optional
   :param reduce: Deprecated
   :type reduce: bool, optional
   :param reduction: Specifies the reduction to apply to the output
   :type reduction: str, optional

.. class:: riemann.nn.SmoothL1Loss(beta=1.0, size_average=None, reduce=None, reduction='mean')

   Creates a criterion that combines L1 loss and L2 loss, using quadratic loss for small errors and linear loss for large errors.

   :param beta: Threshold controlling the transition from quadratic to linear loss
   :type beta: float, optional
   :param size_average: Deprecated
   :type size_average: bool, optional
   :param reduce: Deprecated
   :type reduce: bool, optional
   :param reduction: Specifies the reduction to apply to the output
   :type reduction: str, optional

Optimizer
=========

.. class:: riemann.optim.Optimizer

   Base class for all optimizers.


.. class:: riemann.optim.Adagrad(params, lr=0.01, lr_decay=0.0, weight_decay=0.0, initial_accumulator_value=0.0, eps=1e-10)

   Adagrad (Adaptive Gradient Algorithm) optimizer
   
   An adaptive learning rate optimization algorithm that adjusts learning rates based on the frequency of each parameter's updates.
   Parameters that are updated infrequently (sparse features) get larger learning rates, and vice versa.
   
   Parameter update formula: θ_i = θ_i - (η / (√G_ii + ε)) * ∇θ_i L(θ)
   where G_ii is the sum of squares of gradients for parameter i
   
   Application scenarios:
       - Training with sparse data
       - Problems where some features are infrequently updated
       - Online learning tasks

   Initialize Adagrad optimizer
   
   Parameters:
       params: Parameter group to be optimized
       lr: Learning rate (default: 0.01)
       eps: Epsilon value to prevent division by zero (default: 1e-10)
       weight_decay: L2 regularization coefficient (default: 0)
   
   Exceptions:
       ValueError: Thrown when learning rate, weight decay, or epsilon is negative


.. class:: riemann.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

   Adam (Adaptive Moment Estimation) optimizer
   
   An adaptive learning rate optimization algorithm that computes individual learning rates for different parameters based on first and second moment estimates.
   Combines the advantages of AdaGrad (handling sparse gradients) and RMSProp (handling non-stationary objectives).
   
   Parameter update formula: θ = θ - η * m̂ / (√v̂ + ε)
   where m̂ and v̂ are the bias-corrected first and second moment estimates
   
   Application scenarios:
       - Most deep learning tasks
       - Training neural networks with large, sparse datasets
       - Situations where adaptive learning rates are beneficial

   Initialize Adam optimizer
   
   Parameters:
       params: Parameter group to be optimized
       lr: Learning rate (default: 0.001)
       betas: Tuple of (beta1, beta2) for first and second moment estimates (default: (0.9, 0.999))
       eps: Epsilon value to prevent division by zero (default: 1e-08)
       weight_decay: L2 regularization coefficient (default: 0)
   
   Exceptions:
       ValueError: Thrown when learning rate, weight decay, or epsilon is negative
       ValueError: Thrown when beta parameters are not between 0 and 1 (exclusive)


.. class:: riemann.optim.GD(params, lr=0.01, weight_decay=0.0)

   Gradient Descent optimizer
   
   A basic optimization algorithm that updates parameters in the direction of the negative gradient.
   Supports L2 regularization (weight decay) to prevent overfitting.
   
   Parameter update formula: θ = θ - η * ∇θL(θ)
   where θ is the parameter, η is learning rate, and ∇θL(θ) is the gradient of loss function with respect to the parameter
   
   Application scenarios:
       - Small datasets and simple models
       - As a baseline for comparison with more complex optimization algorithms

   Initialize Gradient Descent optimizer
   
   Parameters:
       params: Parameter group to be optimized (must contain TN objects with requires_grad=True)
       lr: Learning rate (default 0.01)
       weight_decay: L2 regularization coefficient (default 0.0)
   
   Exceptions:
       ValueError: Thrown when learning rate or weight decay coefficient is negative


.. class:: riemann.optim.LBFGS(params, lr=1.0, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)

   L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) optimizer
   
   A quasi-Newton optimization algorithm that approximates the inverse of Hessian matrix by maintaining a limited number of historical gradients and parameter changes.
   Compared to standard BFGS, it uses less memory and is suitable for large-scale optimization problems.
   
   Note: Unlike other optimizers, LBFGS requires a closure function to recompute loss and gradients.

   Initialize LBFGS optimizer
   
   Parameters:
       params: Parameter group to be optimized
       lr: Learning rate (default: 1.0)
       max_iter: Maximum number of iterations per optimization step (default: 20)
       max_eval: Maximum number of function evaluations per optimization step (default: None, i.e., max_iter * 1.25)
       tolerance_grad: Gradient convergence threshold (default: 1e-5)
       tolerance_change: Parameter change convergence threshold (default: 1e-9)
       history_size: History record size (default: 100)
       line_search_fn: Line search function (default: None, using built-in strong Wolfe conditions line search)

.. class:: riemann.optim.SGD(params: Union[Iterable[Union[riemann.tensordef.TN, riemann.nn.module.Parameter]], List[Dict[str, Any]]], lr: float = 0.01, momentum: float = 0.0, weight_decay: float = 0.0, dampening: float = 0.0, nesterov: bool = False) -> None

   Stochastic Gradient Descent (SGD) optimizer
   
   A stochastic version of Gradient Descent that computes gradients using one batch of data per iteration.
   Supports momentum to accelerate convergence and reduce oscillations, and supports Nesterov momentum.
   
   Parameter update formulas:
       - Standard momentum: v = μ*v + ∇θL(θ), θ = θ - η*v
       - Nesterov momentum: v = μ*v + ∇θL(θ) + μ*η*∇θL(θ), θ = θ - η*v
   
   Application scenarios:
       - Training on large-scale datasets
       - Complex optimization problems that need to escape local minima
       - Default choice for most deep learning tasks

   Initialize Stochastic Gradient Descent optimizer
   
   Parameters:
       params: Parameter group to be optimized (must contain TN objects with requires_grad=True)
       lr: Learning rate (default 0.01)
       momentum: Momentum coefficient (default 0.0)
       weight_decay: L2 regularization coefficient (default 0.0)
       dampening: Momentum dampening coefficient (default 0.0)
       nesterov: Whether to enable Nesterov momentum (default False)
   
   Exceptions:
       ValueError: Thrown when learning rate, momentum coefficient, weight decay coefficient or dampening coefficient is negative
       ValueError: Thrown when nesterov is enabled but momentum is 0

