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

Note: LayerNorm is not yet implemented in Riemann.

Activation Functions
~~~~~~~~~~~~~~~~~~~~

.. class:: riemann.nn.ReLU(inplace=False)

   Applies the rectified linear unit function element-wise.

   :param inplace: Whether to do the operation in-place
   :type inplace: bool, optional

.. class:: riemann.nn.Sigmoid

   Applies the element-wise function: sigmoid(x) = 1 / (1 + exp(-x))

.. class:: riemann.nn.Tanh

   Applies the element-wise function: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

.. class:: riemann.nn.Softmax(dim=None)

   Applies the Softmax function to an n-dimensional input tensor.

   :param dim: Dimension along which Softmax will be computed
   :type dim: int

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

.. class:: riemann.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

   Creates a criterion that measures the mean squared error between each element in the input and target.

   :param size_average: Deprecated
   :type size_average: bool, optional
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

Optimization Algorithms
------------------------

Base Class
~~~~~~~~~~

.. class:: riemann.optim.Optimizer(params, defaults)

   Base class for all optimizers.

   :param params: Iterable of parameters to optimize or dicts defining parameter groups
   :type params: iterable
   :param defaults: Default values for optimization options
   :type defaults: dict

   Methods:

   .. method:: step(closure=None)

      Performs a single optimization step.

      :param closure: A closure that reevaluates the model and returns the loss
      :type closure: callable, optional
      :return: The loss value if closure is provided
      :rtype: riemann.Tensor

   .. method:: zero_grad()

      Clears the gradients of all optimized parameters.

   .. method:: state_dict()

      Returns the state of the optimizer as a dictionary.

      :return: State dictionary
      :rtype: dict

   .. method:: load_state_dict(state_dict)

      Loads the optimizer state.

      :param state_dict: Optimizer state
      :type state_dict: dict

Specific Optimizers
~~~~~~~~~~~~~~~~~~~

.. class:: riemann.optim.SGD(params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False)

   Implements stochastic gradient descent.

   :param params: Iterable of parameters to optimize or dicts defining parameter groups
   :type params: iterable
   :param lr: Learning rate
   :type lr: float
   :param momentum: Momentum factor
   :type momentum: float, optional
   :param dampening: Dampening for momentum
   :type dampening: float, optional
   :param weight_decay: Weight decay (L2 penalty)
   :type weight_decay: float, optional
   :param nesterov: Whether to enable Nesterov momentum
   :type nesterov: bool, optional

.. class:: riemann.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

   Implements Adam algorithm.

   :param params: Iterable of parameters to optimize or dicts defining parameter groups
   :type params: iterable
   :param lr: Learning rate
   :type lr: float
   :param betas: Coefficients used for computing running averages of gradient and its square
   :type betas: tuple of float, optional
   :param eps: Term added to the denominator to improve numerical stability
   :type eps: float, optional
   :param weight_decay: Weight decay (L2 penalty)
   :type weight_decay: float, optional
   :param amsgrad: Whether to use the AMSGrad variant of this algorithm
   :type amsgrad: bool, optional

.. class:: riemann.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)

   Implements Adagrad algorithm.

   :param params: Iterable of parameters to optimize or dicts defining parameter groups
   :type params: iterable
   :param lr: Learning rate
   :type lr: float
   :param lr_decay: Learning rate decay
   :type lr_decay: float, optional
   :param weight_decay: Weight decay (L2 penalty)
   :type weight_decay: float, optional
   :param initial_accumulator_value: Initial value for the accumulator
   :type initial_accumulator_value: float, optional
   :param eps: Term added to the denominator to improve numerical stability
   :type eps: float, optional

Learning Rate Schedulers
------------------------

Note: Learning rate schedulers are not yet implemented in Riemann.

Data Loading
------------

.. class:: riemann.utils.data.Dataset

   An abstract class representing a Dataset.

   Methods:

   .. method:: __getitem__(index)

      Returns the sample at the given index.

      :param index: Index
      :type index: int
      :return: Sample at the given index
      :rtype: Any

   .. method:: __len__()

      Returns the size of the dataset.

      :return: Size of the dataset
      :rtype: int

.. class:: riemann.utils.data.TensorDataset(*tensors)

   Dataset wrapping tensors.

   :param tensors: Tensors that have the same size of the first dimension
   :type tensors: riemann.Tensor

.. class:: riemann.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)

   Combines a dataset and a sampler, and provides an iterable over the given dataset.

   :param dataset: Dataset from which to load the data
   :type dataset: riemann.utils.data.Dataset
   :param batch_size: How many samples per batch to load
   :type batch_size: int, optional
   :param shuffle: Whether to shuffle the data at every epoch
   :type shuffle: bool, optional
   :param sampler: Defines the strategy to draw samples from the dataset
   :type sampler: riemann.utils.data.Sampler, optional
   :param batch_sampler: Like sampler, but returns a batch of indices at a time
   :type batch_sampler: riemann.utils.data.Sampler, optional
   :param num_workers: How many subprocesses to use for data loading
   :type num_workers: int, optional
   :param collate_fn: Merges a list of samples to form a mini-batch
   :type collate_fn: callable, optional
   :param pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory
   :type pin_memory: bool, optional
   :param drop_last: Whether to drop the last incomplete batch
   :type drop_last: bool, optional
   :param timeout: Timeout value for collecting a batch from workers
   :type timeout: float, optional
   :param worker_init_fn: If not None, this will be called on each worker subprocess
   :type worker_init_fn: callable, optional

Computer Vision
---------------

.. class:: riemann.vision.datasets.MNIST(root, train=True, transform=None, target_transform=None)

   MNIST dataset.

   :param root: Root directory of dataset where MNIST files are stored
   :type root: str
   :param train: Whether to use training set
   :type train: bool, optional
   :param transform: Optional transform to be applied on a sample
   :type transform: callable, optional
   :param target_transform: Optional transform to be applied on a target
   :type target_transform: callable, optional

.. class:: riemann.vision.datasets.CIFAR10(root, train=True, transform=None, target_transform=None)

   CIFAR-10 dataset.

   :param root: Root directory of dataset where CIFAR-10 files are stored
   :type root: str
   :param train: Whether to use training set
   :type train: bool, optional
   :param transform: Optional transform to be applied on a sample
   :type transform: callable, optional
   :param target_transform: Optional transform to be applied on a target
   :type target_transform: callable, optional

Note: ImageFolder is not yet implemented in Riemann.

.. class:: riemann.vision.transforms.Compose(transforms)

   Composes several transforms together.

   :param transforms: List of transforms to compose
   :type transforms: list of callables

Linear Algebra
--------------

.. function:: riemann.linalg.matmul(A, B, *, out=None)

   Computes the matrix product of two tensors.

   The behavior depends on the dimensions of the input tensors:
   - 1D × 1D: Vector inner product (scalar)
   - 2D × 1D: Matrix-vector product
   - 1D × 2D: Vector-matrix product
   - 2D × 2D: Matrix multiplication
   - ≥3D: Batched matrix multiplication (with broadcasting support)

   :param A: First tensor
   :type A: riemann.TN
   :param B: Second tensor
   :type B: riemann.TN
   :param out: Output tensor (optional)
   :type out: riemann.TN, optional
   :return: Matrix product of the tensors
   :rtype: riemann.TN

   Examples:
     >>> A = tensor([[1, 2], [3, 4]])
     >>> B = tensor([[5, 6], [7, 8]])
     >>> C = matmul(A, B)  # Returns [[19, 22], [43, 50]]
     
     >>> a = tensor([1, 2, 3])
     >>> b = tensor([4, 5, 6])
     >>> c = matmul(a, b)  # Returns scalar 32
     
     >>> A = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # Shape (2, 2, 2)
     >>> B = tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])  # Shape (2, 2, 2)
     >>> C = matmul(A, B)  # Returns tensor of shape (2, 2, 2)

.. function:: riemann.linalg.inv(input)

   Computes the inverse of a square matrix.

   :param input: Input matrix
   :type input: riemann.TN
   :return: Inverse of the matrix
   :rtype: riemann.TN

.. function:: riemann.linalg.pinv(input, rcond=1e-15)

   Computes the pseudoinverse of a matrix.

   :param input: Input matrix
   :type input: riemann.TN
   :param rcond: Cutoff for small singular values
   :type rcond: float, optional
   :return: Pseudoinverse of the matrix
   :rtype: riemann.TN

.. function:: riemann.linalg.det(input)

   Computes the determinant of a square matrix.

   :param input: Input matrix
   :type input: riemann.TN
   :return: Determinant of the matrix
   :rtype: riemann.TN

.. function:: riemann.linalg.svd(input, full_matrices=True, compute_uv=True)

   Computes the singular value decomposition of a matrix.

   :param input: Input matrix
   :type input: riemann.TN
   :param full_matrices: Whether to compute full-sized U and V
   :type full_matrices: bool, optional
   :param compute_uv: Whether to compute U and V in addition to singular values
   :type compute_uv: bool, optional
   :return: Singular value decomposition
   :rtype: tuple of riemann.TN

.. function:: riemann.linalg.eig(input)

   Computes the eigenvalues and eigenvectors of a square matrix.

   :param input: Input square matrix of shape (\*, n, n) where \* is zero or more batch dimensions
   :type input: riemann.TN
   :return: Tuple containing eigenvalues and eigenvectors
   :rtype: tuple of riemann.TN

.. function:: riemann.linalg.qr(input)

   Computes the QR decomposition of a matrix.

   :param input: Input matrix
   :type input: riemann.TN
   :return: Q and R matrices
   :rtype: tuple of riemann.TN

.. function:: riemann.linalg.solve(input, other)

   Solves a linear system of equations.

   :param input: Coefficient matrix
   :type input: riemann.TN
   :param other: Right-hand side tensor
   :type other: riemann.TN
   :return: Solution tensor
   :rtype: riemann.TN

.. function:: riemann.linalg.norm(input, p='fro', dim=None, keepdim=False, out=None)

   Computes the norm of a tensor.

   :param input: Input tensor
   :type input: riemann.TN
   :param p: The order of norm
   :type p: int, float, or str, optional
   :param dim: Dimensions to reduce
   :type dim: int or tuple of ints, optional
   :param keepdim: Whether to keep the reduced dimensions
   :type keepdim: bool, optional
   :param out: Output tensor
   :type out: riemann.TN, optional
   :return: Norm of the tensor
   :rtype: riemann.TN

.. function:: riemann.linalg.vector_norm(x, ord=None, dim=None, keepdim=False, out=None, dtype=None)

   Computes the vector norm of a tensor.

   :param x: Input tensor
   :type x: riemann.TN
   :param ord: Order of the norm. Supported values: None (default, 2-norm), 0, 1, 2, -1, -2, inf, -inf
   :type ord: int, float, or None, optional
   :param dim: Dimension(s) to reduce over
   :type dim: int or tuple of ints, optional
   :param keepdim: Whether to keep the reduced dimensions
   :type keepdim: bool, optional
   :param out: Output tensor
   :type out: riemann.TN, optional
   :param dtype: Data type of the output tensor
   :type dtype: numpy.dtype, optional
   :return: Vector norm of the tensor
   :rtype: riemann.TN

.. function:: riemann.linalg.matrix_norm(A, ord=None, dim=None, keepdim=False, out=None, dtype=None)

   Computes the matrix norm of a tensor.

   :param A: Input tensor
   :type A: riemann.TN
   :param ord: Order of the norm. Supported values: None (default, Frobenius), 1, 2, -1, -2, inf, -inf, 'fro', 'nuc'
   :type ord: int, float, or str, optional
   :param dim: Dimensions to reduce over, must be a tuple of length 2
   :type dim: tuple of ints, optional
   :param keepdim: Whether to keep the reduced dimensions
   :type keepdim: bool, optional
   :param out: Output tensor
   :type out: riemann.TN, optional
   :param dtype: Data type of the output tensor
   :type dtype: numpy.dtype, optional
   :return: Matrix norm of the tensor
   :rtype: riemann.TN

.. function:: riemann.linalg.cond(A, p=None, *, out=None)

   Computes the condition number of a matrix with respect to a matrix norm.

   :param A: Input tensor of shape (\*, m, n) where \* is zero or more batch dimensions
   :type A: riemann.TN
   :param p: Matrix norm to use. Supported values: None (default, 2-norm), 2, -2, 'fro', 'nuc', inf, -inf, 1, -1
   :type p: int, float, or str, optional
   :param out: Output tensor
   :type out: riemann.TN, optional
   :return: Condition number of the matrix
   :rtype: riemann.TN

.. function:: riemann.linalg.svdvals(A)

   Returns the singular values of a matrix.

   :param A: Input tensor of shape (\*, m, n) where \* is zero or more batch dimensions
   :type A: riemann.TN
   :return: Singular values of the matrix, shape (\*, k) where k = min(m, n)
   :rtype: riemann.TN

.. function:: riemann.linalg.skew(A)

   Computes the skew-symmetric part of a square matrix.

   The skew-symmetric part is defined as: (A - A^T) / 2, where A^T is the transpose of A.

   :param A: Input square matrix
   :type A: riemann.TN
   :return: Skew-symmetric matrix
   :rtype: riemann.TN

.. function:: riemann.linalg.eigh(A)

   Computes the eigenvalues and eigenvectors of a complex Hermitian (or real symmetric) matrix.

   :param A: Input Hermitian or symmetric matrix of shape (\*, n, n) where \* is zero or more batch dimensions
   :type A: riemann.TN
   :return: Tuple containing eigenvalues and eigenvectors
   :rtype: tuple of riemann.TN

.. function:: riemann.linalg.lu(A, *, pivot=True, out=None)

   Computes the LU decomposition of a matrix.

   If pivot=True, computes A = PLU where P is the permutation matrix.
   If pivot=False, computes A = LU without row exchanges.

   :param A: Input matrix of shape (\*, m, n) where \* is zero or more batch dimensions
   :type A: riemann.TN
   :param pivot: Whether to use pivoting
   :type pivot: bool, optional
   :param out: Output tensor
   :type out: riemann.TN, optional
   :return: LU decomposition of the matrix
   :rtype: riemann.TN

.. function:: riemann.linalg.lstsq(A, B, *, rcond=None, out=None)

   Solves the linear least squares problem: min_X ||AX - B||2^2

   :param A: Coefficient matrix of shape (\*, m, n) where \* is zero or more batch dimensions
   :type A: riemann.TN
   :param B: Right-hand side matrix or vector of shape (\*, m, k) or (\*, m)
   :type B: riemann.TN
   :param rcond: Cutoff for small singular values
   :type rcond: float, optional
   :param out: Output tuple containing (X, residuals, rank, singular_values)
   :type out: tuple, optional
   :return: Solution to the least squares problem
   :rtype: riemann.TN