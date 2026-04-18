API Reference
=============

This section provides a comprehensive reference for all functions, classes, and modules in the Riemann library.

Tensor Operations
-----------------

Tensor Creation Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: riemann.tensor(data, dtype=None, requires_grad=False)

   Create a tensor from data.

   :param data: Data to initialize the tensor
   :type data: array_like
   :param dtype: Expected data type of the tensor
   :type dtype: numpy.dtype, optional
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :return: Tensor containing the given data
   :rtype: riemann.TN

.. function:: riemann.zeros(*shape, dtype=None, requires_grad=False)

   Create a tensor filled with zeros.

   :param shape: Shape of the tensor
   :type shape: int or tuple of integers
   :param dtype: Expected data type of the tensor
   :type dtype: numpy.dtype, optional
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :return: Tensor filled with zeros
   :rtype: riemann.TN

.. function:: riemann.ones(*shape, dtype=None, requires_grad=False)

   Create a tensor filled with ones.

   :param shape: Shape of the tensor
   :type shape: int or tuple of integers
   :param dtype: Expected data type of the tensor
   :type dtype: numpy.dtype, optional
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :return: Tensor filled with ones
   :rtype: riemann.TN

.. function:: riemann.empty(*shape, dtype=None, requires_grad=False)

   Create an uninitialized tensor.

   :param shape: Shape of the tensor
   :type shape: int or tuple of integers
   :param dtype: Expected data type of the tensor
   :type dtype: numpy.dtype, optional
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :return: Uninitialized tensor
   :rtype: riemann.TN

.. function:: riemann.full(*shape, fill_value, dtype=None, requires_grad=False)

   Create a tensor filled with a specific value.

   :param shape: Shape of the tensor
   :type shape: int or tuple of integers
   :param fill_value: Value to fill the tensor with
   :type fill_value: scalar
   :param dtype: Expected data type of the tensor
   :type dtype: numpy.dtype, optional
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :return: Tensor filled with the specified value
   :rtype: riemann.TN

.. function:: riemann.eye(n, m=None, dtype=None, requires_grad=False)

   Create a 2D tensor with ones on the diagonal and zeros elsewhere.

   :param n: Number of rows
   :type n: int
   :param m: Number of columns (default: n)
   :type m: int, optional
   :param dtype: Expected data type of the tensor
   :type dtype: numpy.dtype, optional
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :return: 2D tensor with ones on the diagonal
   :rtype: riemann.TN

.. function:: riemann.zeros_like(tsr, dtype=None, requires_grad=False)

   Create a zero tensor with the same shape as the input tensor.

   :param tsr: Reference tensor
   :type tsr: riemann.TN
   :param dtype: Expected data type of the tensor
   :type dtype: numpy.dtype, optional
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :return: Zero tensor with the same shape as the input tensor
   :rtype: riemann.TN

.. function:: riemann.ones_like(tsr, dtype=None, requires_grad=False)

   Create a one tensor with the same shape as the input tensor.

   :param tsr: Reference tensor
   :type tsr: riemann.TN
   :param dtype: Expected data type of the tensor
   :type dtype: numpy.dtype, optional
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :return: One tensor with the same shape as the input tensor
   :rtype: riemann.TN

.. function:: riemann.empty_like(tsr, dtype=None, requires_grad=False)

   Create an uninitialized tensor with the same shape as the input tensor.

   :param tsr: Reference tensor
   :type tsr: riemann.TN
   :param dtype: Expected data type of the tensor
   :type dtype: numpy.dtype, optional
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :return: Uninitialized tensor with the same shape as the input tensor
   :rtype: riemann.TN

.. function:: riemann.full_like(tsr, fill_value, dtype=None, requires_grad=False)

   Create a tensor with the same shape as the input tensor, filled with a specific value.

   :param tsr: Reference tensor
   :type tsr: riemann.TN
   :param fill_value: Value to fill the tensor with
   :type fill_value: scalar
   :param dtype: Expected data type of the tensor
   :type dtype: numpy.dtype, optional
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :return: Tensor with the same shape as the input tensor, filled with the specified value
   :rtype: riemann.TN

Random Number Generation
~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: riemann.rand(*size, requires_grad=False, dtype=None)

   Create a tensor filled with random numbers from a uniform distribution over [0, 1).

   :param size: Shape of the tensor
   :type size: int or tuple of integers
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :param dtype: Expected data type of the tensor
   :type dtype: numpy.dtype, optional
   :return: Tensor filled with random values
   :rtype: riemann.TN

.. function:: riemann.randn(*size, requires_grad=False, dtype=None)

   Create a tensor filled with random numbers from a standard normal distribution.

   :param size: Shape of the tensor
   :type size: int or tuple of integers
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :param dtype: Expected data type of the tensor
   :type dtype: numpy.dtype, optional
   :return: Tensor filled with random values
   :rtype: riemann.TN

.. function:: riemann.randint(low, high, size, requires_grad=False, dtype=int64)

   Create a tensor filled with random integers from low (inclusive) to high (exclusive).

   :param low: Minimum integer to draw
   :type low: int
   :param high: Maximum integer plus one to draw
   :type high: int
   :param size: Shape of the tensor
   :type size: int or tuple of integers
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :param dtype: Expected data type of the tensor
   :type dtype: numpy.dtype, optional
   :return: Tensor filled with random integers
   :rtype: riemann.TN

.. function:: riemann.randperm(n, requires_grad=False, dtype=int64)

   Create a tensor containing numbers from 0 to n-1 in random order.

   :param n: Upper bound (exclusive)
   :type n: int
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :param dtype: Expected data type of the tensor
   :type dtype: numpy.dtype, optional
   :return: Tensor containing randomly permuted integers
   :rtype: riemann.TN

.. function:: riemann.normal(mean, std, size, dtype=None)

   Create a tensor filled with random numbers from a normal distribution.

   :param mean: Mean of the normal distribution
   :type mean: float
   :param std: Standard deviation of the normal distribution
   :type std: float
   :param size: Shape of the tensor
   :type size: int or tuple of integers
   :param dtype: Expected data type of the tensor
   :type dtype: numpy.dtype, optional
   :return: Tensor filled with random values
   :rtype: riemann.TN

Random Seed Control
~~~~~~~~~~~~~~~~~~~

.. function:: riemann.manual_seed(seed)

   Set the seed for the random number generator to ensure reproducibility of random operations.

   :param seed: Random seed value
   :type seed: int
   :return: Random number generator object
   :rtype: torch.Generator

Sequence and Range Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: riemann.arange(start, end=None, step=1.0, dtype=None, requires_grad=False)

   Create a 1-D tensor of evenly spaced values from start to end with step.

   :param start: Start value
   :type start: float
   :param end: End value (exclusive)
   :type end: float, optional
   :param step: Spacing between values
   :type step: float, optional
   :param dtype: Expected data type of the tensor
   :type dtype: numpy.dtype, optional
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :return: 1-D tensor containing evenly spaced values
   :rtype: riemann.TN

.. function:: riemann.linspace(start, end, steps=100, endpoint=True, dtype=None, requires_grad=False)

   Create a 1-D tensor of evenly spaced values within a given interval.

   :param start: Start value
   :type start: float
   :param end: End value
   :type end: float
   :param steps: Number of samples to generate
   :type steps: int, optional
   :param endpoint: Whether to include the end value
   :type endpoint: bool, optional
   :param dtype: Expected data type of the tensor
   :type dtype: numpy.dtype, optional
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :return: 1-D tensor containing evenly spaced values
   :rtype: riemann.TN

Tensor Attributes
~~~~~~~~~~~~~~~~~

.. method:: riemann.TN.dtype

   Return the data type of the tensor.

   :return: Data type of the tensor
   :rtype: numpy.dtype

.. method:: riemann.TN.real

   Return the real part of a complex tensor.

   :return: Tensor containing the real parts
   :rtype: riemann.TN

.. method:: riemann.TN.imag

   Return the imaginary part of a complex tensor.

   :return: Tensor containing the imaginary parts
   :rtype: riemann.TN

.. method:: riemann.TN.shape

   Return the shape of the tensor.

   :return: Tuple of tensor dimensions
   :rtype: tuple

.. method:: riemann.TN.ndim

   Return the number of dimensions of the tensor.

   :return: Number of dimensions of the tensor
   :rtype: int

.. method:: riemann.TN.device

   Return the device where the tensor is located.

   :return: Device object where the tensor is located
   :rtype: Device

.. method:: riemann.TN.is_cuda

   Check if the tensor is on a CUDA device.

   :return: True if the tensor is on a CUDA device, False otherwise
   :rtype: bool

.. method:: riemann.TN.is_cpu

   Check if the tensor is on a CPU device.

   :return: True if the tensor is on a CPU device, False otherwise
   :rtype: bool

.. method:: riemann.TN.is_leaf

   Check if the tensor is a leaf node.

   :return: True if the tensor is a leaf node, False otherwise
   :rtype: bool

.. method:: riemann.TN.is_floating_point()

   Check if the tensor is of floating point type.

   :return: True if the tensor is of floating point type, False otherwise
   :rtype: bool

.. method:: riemann.TN.is_complex()

   Check if the tensor is of complex type.

   :return: True if the tensor is of complex type, False otherwise
   :rtype: bool

.. method:: riemann.TN.isreal()

   Determine if tensor elements are real numbers.

   :return: Boolean tensor, True indicates the corresponding element is real
   :rtype: riemann.TN

.. method:: riemann.TN.isinf()

   Determine if tensor elements are infinity.

   :return: Boolean tensor, True indicates the corresponding element is infinity
   :rtype: riemann.TN

.. method:: riemann.TN.isnan()

   Determine if tensor elements are NaN (Not a Number).

   :return: Boolean tensor, True indicates the corresponding element is NaN
   :rtype: riemann.TN

.. method:: riemann.TN.conj()

   Return the complex conjugate of the tensor.

   :return: Tensor containing conjugate elements
   :rtype: riemann.TN

.. method:: riemann.TN.size(dim=None)

   Return the size of the tensor.

   :param dim: Dimension index to query, if None return the entire shape
   :type dim: int, optional
   :return: Shape tuple if dim is None, otherwise size of the specified dimension
   :rtype: tuple or int

.. method:: riemann.TN.numel()

   Return the total number of elements in the tensor.

   :return: Number of elements in the tensor
   :rtype: int

.. method:: riemann.TN.is_contiguous()

   Check if the memory layout of the tensor is contiguous.

   :return: True if the tensor is contiguous, False otherwise
   :rtype: bool

Tensor Shape Operations
~~~~~~~~~~~~~~~~~~~~~~~

.. function:: riemann.reshape(input, shape)

   Return a tensor with the same data but different shape.

   :param input: Input tensor
   :type input: riemann.TN
   :param shape: New shape
   :type shape: tuple of integers
   :return: Tensor with the new shape
   :rtype: riemann.TN

.. function:: riemann.squeeze(input, dim=None)

   Remove dimensions of size 1 from the tensor shape.

   :param input: Input tensor
   :type input: riemann.TN
   :param dim: Dimension to squeeze
   :type dim: int, optional
   :return: Tensor with squeezed dimensions
   :rtype: riemann.TN

.. function:: riemann.unsqueeze(input, dim)

   Insert a dimension of size 1 at the specified position.

   :param input: Input tensor
   :type input: riemann.TN
   :param dim: Dimension to expand
   :type dim: int
   :return: Tensor with expanded dimensions
   :rtype: riemann.TN

.. function:: riemann.transpose(input, dim0, dim1)

   Swap two dimensions of the tensor.

   :param input: Input tensor
   :type input: riemann.TN
   :param dim0: First dimension to swap
   :type dim0: int
   :param dim1: Second dimension to swap
   :type dim1: int
   :return: Tensor with swapped dimensions
   :rtype: riemann.TN

.. attribute:: riemann.TN.mT

   Matrix transpose, i.e., transpose between the last two dimensions of the tensor.

   For a row vector (1, n), transpose to column vector (n, 1); for a column vector (n, 1), transpose to row vector (1, n).

   :return: Transposed tensor
   :rtype: riemann.TN

.. function:: riemann.is_contiguous(input)

   Check if the tensor memory is contiguous.

   :param input: Input tensor
   :type input: riemann.TN
   :return: Whether the memory is contiguous
   :rtype: bool

.. function:: riemann.contiguous(input)

   Return a tensor with contiguous memory.

   :param input: Input tensor
   :type input: riemann.TN
   :return: Tensor with contiguous memory
   :rtype: riemann.TN

.. function:: riemann.gather(input, dim, index)

   Gather elements from the tensor along a specified dimension.

   :param input: Input tensor
   :type input: riemann.TN
   :param dim: Dimension to gather
   :type dim: int
   :param index: Index tensor
   :type index: riemann.TN
   :return: Gathered tensor
   :rtype: riemann.TN

.. function:: riemann.scatter(input, dim, index, src)

   Scatter elements from the source tensor into the target tensor along a specified dimension.

   :param input: Target tensor
   :type input: riemann.TN
   :param dim: Dimension to scatter
   :type dim: int
   :param index: Index tensor
   :type index: riemann.TN
   :param src: Source tensor
   :type src: riemann.TN
   :return: Scattered tensor
   :rtype: riemann.TN

.. function:: riemann.broadcast_to(input, size)

   Broadcast the tensor to a new shape.

   :param input: Input tensor
   :type input: riemann.TN
   :param size: Target shape
   :type size: tuple of integers
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

   Split a tensor into multiple sub-tensors.

   :param ts: Input tensor
   :type ts: riemann.TN
   :param split_indices: Indices to split
   :type split_indices: int or list of integers
   :param dim: Dimension along which to split
   :type dim: int, optional
   :return: List of tensors
   :rtype: List of riemann.TN

.. function:: riemann.stack(tensors, dim=0)

   Stack tensors along a new dimension.

   :param tensors: Sequence of tensors to stack
   :type tensors: Sequence of riemann.TN
   :param dim: Dimension to insert
   :type dim: int, optional
   :return: Stacked tensor
   :rtype: riemann.TN

.. function:: riemann.cat(tensors, dim=0)

   Concatenate tensors along an existing dimension.

   :param tensors: Sequence of tensors to concatenate
   :type tensors: Sequence of riemann.TN
   :param dim: Dimension along which to concatenate
   :type dim: int, optional
   :return: Concatenated tensor
   :rtype: riemann.TN

.. function:: riemann.concatenate(tensors, dim=0)

   Concatenate tensors along an existing dimension.

   :param tensors: Sequence of tensors to concatenate
   :type tensors: Sequence of riemann.TN
   :param dim: Dimension along which to concatenate
   :type dim: int, optional
   :return: Concatenated tensor
   :rtype: riemann.TN

.. function:: riemann.vstack(tensors)

   Stack tensors vertically (row-wise).

   :param tensors: Sequence of tensors to stack
   :type tensors: Sequence of riemann.TN
   :return: Vertically stacked tensor
   :rtype: riemann.TN

.. function:: riemann.hstack(tensors)

   Stack tensors horizontally (column-wise).

   :param tensors: Sequence of tensors to stack
   :type tensors: Sequence of riemann.TN
   :return: Horizontally stacked tensor
   :rtype: riemann.TN

.. function:: riemann.dstack(tensors)

   Stack tensors in depth (along dimension 2).

   1D tensors are first reshaped to (1, N, 1), 2D tensors are reshaped to (M, N, 1), then stacked along dimension 2.

   :param tensors: Sequence of tensors to stack
   :type tensors: Sequence of riemann.TN
   :return: Depth-stacked tensor
   :rtype: riemann.TN

.. function:: riemann.tensor_split(input, indices_or_sections, dim=0)

   Split a tensor into multiple sub-tensors along a specified dimension.

   When ``indices_or_sections`` is an integer, it specifies the number of sections to split the tensor into.
   When ``indices_or_sections`` is a list, it specifies the indices at which to split.

   :param input: Input tensor
   :type input: riemann.TN
   :param indices_or_sections: Split parameter (number of sections or list of indices)
   :type indices_or_sections: int or list[int]
   :param dim: Dimension along which to split, defaults to 0
   :type dim: int, optional
   :return: Tuple of split tensors
   :rtype: tuple[riemann.TN, ...]

.. function:: riemann.vsplit(input, indices_or_sections)

   Split a tensor vertically (along dimension 0).

   Splits the tensor along dimension 0 (vertical direction) into multiple sub-tensors.

   :param input: Input tensor
   :type input: riemann.TN
   :param indices_or_sections: Split parameter (number of sections or list of indices)
   :type indices_or_sections: int or list[int]
   :return: Tuple of split tensors
   :rtype: tuple[riemann.TN, ...]

.. function:: riemann.hsplit(input, indices_or_sections)

   Split a tensor horizontally (along dimension 1).

   Splits the tensor along dimension 1 (horizontal direction) into multiple sub-tensors.

   :param input: Input tensor
   :type input: riemann.TN
   :param indices_or_sections: Split parameter (number of sections or list of indices)
   :type indices_or_sections: int or list[int]
   :return: Tuple of split tensors
   :rtype: tuple[riemann.TN, ...]

.. function:: riemann.dsplit(input, indices_or_sections)

   Split a tensor in depth (along dimension 2).

   Splits a 3D+ tensor along dimension 2 (depth direction) into multiple sub-tensors.

   :param input: Input tensor (at least 3D)
   :type input: riemann.TN
   :param indices_or_sections: Split parameter (number of sections or list of indices)
   :type indices_or_sections: int or list[int]
   :return: Tuple of split tensors
   :rtype: tuple[riemann.TN, ...]

Tensor Operators
~~~~~~~~~~~~~~~~

The Riemann framework supports a rich set of tensor operators, including arithmetic operators, comparison operators, bitwise operators, and in-place operators. These operators can directly act on tensor objects, support automatic differentiation, and follow Python's operator precedence rules.

Arithmetic Operators
````````````````````

.. method:: __add__(other)

   Tensor addition operator, equivalent to `+`.

   :param other: Another tensor or scalar value
   :type other: riemann.TN or int or float or complex
   :return: Addition result tensor
   :rtype: riemann.TN

.. method:: __radd__(other)

   Reverse tensor addition operator, used when the left operand is not a tensor.

   :param other: Non-tensor left operand
   :type other: int or float or complex
   :return: Addition result tensor
   :rtype: riemann.TN

.. method:: __sub__(other)

   Tensor subtraction operator, equivalent to `-`.

   :param other: Another tensor or scalar value
   :type other: riemann.TN or int or float or complex
   :return: Subtraction result tensor
   :rtype: riemann.TN

.. method:: __rsub__(other)

   Reverse tensor subtraction operator, used when the left operand is not a tensor.

   :param other: Non-tensor left operand
   :type other: int or float or complex
   :return: Subtraction result tensor
   :rtype: riemann.TN

.. method:: __mul__(other)

   Tensor multiplication operator, equivalent to `*`.

   :param other: Another tensor or scalar value
   :type other: riemann.TN or int or float or complex
   :return: Multiplication result tensor
   :rtype: riemann.TN

.. method:: __rmul__(other)

   Reverse tensor multiplication operator, used when the left operand is not a tensor.

   :param other: Non-tensor left operand
   :type other: int or float or complex
   :return: Multiplication result tensor
   :rtype: riemann.TN

.. method:: __matmul__(other)

   Tensor matrix multiplication operator, equivalent to `@`.

   :param other: Another tensor
   :type other: riemann.TN
   :return: Matrix multiplication result tensor
   :rtype: riemann.TN
   :raises RuntimeError: If either operand is a scalar

.. method:: __rmatmul__(other)

   Reverse tensor matrix multiplication operator, used when the left operand is not a tensor.

   :param other: Non-tensor left operand
   :type other: int or float or complex
   :return: Matrix multiplication result tensor
   :rtype: riemann.TN

.. method:: __truediv__(other)

   Tensor division operator, equivalent to `/`.

   :param other: Another tensor or scalar value
   :type other: riemann.TN or int or float or complex
   :return: Division result tensor
   :rtype: riemann.TN

.. method:: __rtruediv__(other)

   Reverse tensor division operator, used when the left operand is not a tensor.

   :param other: Non-tensor left operand
   :type other: int or float or complex
   :return: Division result tensor
   :rtype: riemann.TN

.. method:: __pow__(other)

   Tensor power operator, equivalent to `**`.

   :param other: Exponent tensor or scalar value
   :type other: riemann.TN or int or float or complex
   :return: Power operation result tensor
   :rtype: riemann.TN

.. method:: __rpow__(other)

   Reverse tensor power operator, used when the left operand is not a tensor.

   :param other: Non-tensor left operand
   :type other: int or float or complex
   :return: Power operation result tensor
   :rtype: riemann.TN

.. method:: __pos__()

   Tensor unary plus operator, equivalent to `+`.

   :return: Original tensor
   :rtype: riemann.TN

.. method:: __neg__()

   Tensor unary minus operator, equivalent to `-`.

   :return: Negated result tensor
   :rtype: riemann.TN

Comparison Operators
````````````````````

.. method:: __lt__(other)

   Tensor less than operator, equivalent to `<`.

   :param other: Another tensor or scalar value
   :type other: riemann.TN or int or float or complex
   :return: Boolean result tensor
   :rtype: riemann.TN
   :raises TypeError: If other is None

.. method:: __le__(other)

   Tensor less than or equal operator, equivalent to `<=`.

   :param other: Another tensor or scalar value
   :type other: riemann.TN or int or float or complex
   :return: Boolean result tensor
   :rtype: riemann.TN
   :raises TypeError: If other is None

.. method:: __gt__(other)

   Tensor greater than operator, equivalent to `>`.

   :param other: Another tensor or scalar value
   :type other: riemann.TN or int or float or complex
   :return: Boolean result tensor
   :rtype: riemann.TN
   :raises TypeError: If other is None

.. method:: __ge__(other)

   Tensor greater than or equal operator, equivalent to `>=`.

   :param other: Another tensor or scalar value
   :type other: riemann.TN or int or float or complex
   :return: Boolean result tensor
   :rtype: riemann.TN
   :raises TypeError: If other is None

.. method:: __eq__(other)

   Tensor equality operator, equivalent to `==`.

   :param other: Another tensor or scalar value
   :type other: riemann.TN or int or float or complex
   :return: Boolean result tensor
   :rtype: riemann.TN

.. method:: __ne__(other)

   Tensor inequality operator, equivalent to `!=`.

   :param other: Another tensor or scalar value
   :type other: riemann.TN or int or float or complex
   :return: Boolean result tensor
   :rtype: riemann.TN

Bitwise Operators
`````````````````

.. method:: __and__(other)

   Tensor bitwise AND operator, equivalent to `&`.

   :param other: Another tensor or scalar value
   :type other: riemann.TN or int
   :return: Bitwise AND result tensor
   :rtype: riemann.TN

.. method:: __or__(other)

   Tensor bitwise OR operator, equivalent to `|`.

   :param other: Another tensor or scalar value
   :type other: riemann.TN or int
   :return: Bitwise OR result tensor
   :rtype: riemann.TN

.. method:: __xor__(other)

   Tensor bitwise XOR operator, equivalent to `^`.

   :param other: Another tensor or scalar value
   :type other: riemann.TN or int
   :return: Bitwise XOR result tensor
   :rtype: riemann.TN

.. method:: __invert__()

   Tensor bitwise NOT operator, equivalent to `~`.

   :return: Bitwise NOT result tensor
   :rtype: riemann.TN

.. method:: __lshift__(other)

   Tensor left shift operator, equivalent to `<<`.

   :param other: Number of bits to shift
   :type other: riemann.TN or int
   :return: Left shift result tensor
   :rtype: riemann.TN

.. method:: __rshift__(other)

   Tensor right shift operator, equivalent to `>>`.

   :param other: Number of bits to shift
   :type other: riemann.TN or int
   :return: Right shift result tensor
   :rtype: riemann.TN

In-place Operators
``````````````````

.. method:: __iadd__(other)

   Tensor in-place addition operator, equivalent to `+=`.

   :param other: Another tensor or scalar value
   :type other: riemann.TN or int or float or complex
   :return: In-place modified tensor
   :rtype: riemann.TN
   :raises RuntimeError: If the tensor is a leaf node that requires gradients

.. method:: __isub__(other)

   Tensor in-place subtraction operator, equivalent to `-=`.

   :param other: Another tensor or scalar value
   :type other: riemann.TN or int or float or complex
   :return: In-place modified tensor
   :rtype: riemann.TN
   :raises RuntimeError: If the tensor is a leaf node that requires gradients

.. method:: __imul__(other)

   Tensor in-place multiplication operator, equivalent to `*=`.

   :param other: Another tensor or scalar value
   :type other: riemann.TN or int or float or complex
   :return: In-place modified tensor
   :rtype: riemann.TN
   :raises RuntimeError: If the tensor is a leaf node that requires gradients

.. method:: __itruediv__(other)

   Tensor in-place division operator, equivalent to `/=`.

   :param other: Another tensor or scalar value
   :type other: riemann.TN or int or float or complex
   :return: In-place modified tensor
   :rtype: riemann.TN
   :raises RuntimeError: If the tensor is a leaf node that requires gradients

.. method:: __ipow__(other)

   Tensor in-place power operator, equivalent to `**=`.

   :param other: Exponent tensor or scalar value
   :type other: riemann.TN or int or float or complex
   :return: In-place modified tensor
   :rtype: riemann.TN
   :raises RuntimeError: If the tensor is a leaf node that requires gradients

Mathematical Operations
~~~~~~~~~~~~~~~~~~~~~~~

.. function:: riemann.matmul(input, other)

   Matrix multiplication of two tensors.

   :param input: First tensor
   :type input: riemann.TN
   :param other: Second tensor
   :type other: riemann.TN
   :return: Matrix product of the tensors
   :rtype: riemann.TN

.. function:: riemann.dot(x, y)

   Calculate the dot product of two tensors.

   :param x: First tensor
   :type x: riemann.TN
   :param y: Second tensor
   :type y: riemann.TN
   :return: Dot product result
   :rtype: riemann.TN

.. function:: riemann.sum(x, dim=None, keepdim=False)

   Calculate the sum of elements across dimensions.

   :param x: Input tensor
   :type x: riemann.TN
   :param dim: Dimensions to sum
   :type dim: int or tuple of integers, optional
   :param keepdim: Whether to keep the reduced dimensions
   :type keepdim: bool, optional
   :return: Sum of elements
   :rtype: riemann.TN

.. function:: riemann.prod(x, dim=None, keepdim=False)

   Calculate the product of elements across dimensions.

   :param x: Input tensor
   :type x: riemann.TN
   :param dim: Dimensions to multiply
   :type dim: int or tuple of integers, optional
   :param keepdim: Whether to keep the reduced dimensions
   :type keepdim: bool, optional
   :return: Product of elements
   :rtype: riemann.TN

.. function:: riemann.mean(x, dim=None, keepdim=False)

   Calculate the mean of elements across dimensions.

   :param x: Input tensor
   :type x: riemann.TN
   :param dim: Dimensions to average
   :type dim: int or tuple of integers, optional
   :param keepdim: Whether to keep the reduced dimensions
   :type keepdim: bool, optional
   :return: Mean of elements
   :rtype: riemann.TN

.. function:: riemann.var(x, dim=None, unbiased=True, keepdim=False)

   Calculate the variance of elements across dimensions.

   :param x: Input tensor
   :type x: riemann.TN
   :param dim: Dimensions to calculate variance
   :type dim: int or tuple of integers, optional
   :param unbiased: Whether to use unbiased estimation
   :type unbiased: bool, optional
   :param keepdim: Whether to keep the reduced dimensions
   :type keepdim: bool, optional
   :return: Variance of elements
   :rtype: riemann.TN

.. function:: riemann.std(x, dim=None, unbiased=True, keepdim=False)

   Calculate the standard deviation of elements across dimensions.

   :param x: Input tensor
   :type x: riemann.TN
   :param dim: Dimensions to calculate standard deviation
   :type dim: int or tuple of integers, optional
   :param unbiased: Whether to use unbiased estimation
   :type unbiased: bool, optional
   :param keepdim: Whether to keep the reduced dimensions
   :type keepdim: bool, optional
   :return: Standard deviation of elements
   :rtype: riemann.TN

.. function:: riemann.norm(x, p="fro", dim=None, keepdim=False)

   Calculate the norm of the tensor.

   :param x: Input tensor
   :type x: riemann.TN
   :param p: Norm order
   :type p: int, float, str, optional
   :param dim: Dimensions to calculate norm
   :type dim: int or tuple of integers, optional
   :param keepdim: Whether to keep the reduced dimensions
   :type keepdim: bool, optional
   :return: Norm of the tensor
   :rtype: riemann.TN

.. function:: riemann.max(x, dim=None, keepdim=False, *, out=None)

   Calculate the maximum value of elements across dimensions.

   :param x: Input tensor
   :type x: riemann.TN
   :param dim: Dimension to find maximum
   :type dim: int, optional
   :param keepdim: Whether to keep the reduced dimensions
   :type keepdim: bool, optional
   :param out: Output tensor
   :type out: riemann.TN, optional
   :return: Maximum value
   :rtype: riemann.TN

.. function:: riemann.min(x, dim=None, keepdim=False, *, out=None)

   Calculate the minimum value of elements across dimensions.

   :param x: Input tensor
   :type x: riemann.TN
   :param dim: Dimension to find minimum
   :type dim: int, optional
   :param keepdim: Whether to keep the reduced dimensions
   :type keepdim: bool, optional
   :param out: Output tensor
   :type out: riemann.TN, optional
   :return: Minimum value
   :rtype: riemann.TN

.. function:: riemann.argmax(x, dim=None, keepdim=False, *, out=None)

   Calculate the indices of maximum values across dimensions.

   :param x: Input tensor
   :type x: riemann.TN
   :param dim: Dimension to find maximum indices
   :type dim: int, optional
   :param keepdim: Whether to keep the reduced dimensions
   :type keepdim: bool, optional
   :param out: Output tensor
   :type out: riemann.TN, optional
   :return: Indices of maximum values
   :rtype: riemann.TN

.. function:: riemann.argmin(x, dim=None, keepdim=False, *, out=None)

   Calculate the indices of minimum values across dimensions.

   :param x: Input tensor
   :type x: riemann.TN
   :param dim: Dimension to find minimum indices
   :type dim: int, optional
   :param keepdim: Whether to keep the reduced dimensions
   :type keepdim: bool, optional
   :param out: Output tensor
   :type out: riemann.TN, optional
   :return: Indices of minimum values
   :rtype: riemann.TN

.. function:: riemann.abs(x)

   Calculate the absolute value of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Absolute value of each element
   :rtype: riemann.TN

.. function:: riemann.sqrt(x)

   Calculate the square root of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Square root of each element
   :rtype: riemann.TN

.. function:: riemann.pow(input, exponent)

   Raise each element to a power.

   :param input: Input tensor
   :type input: riemann.TN
   :param exponent: Exponent value
   :type exponent: riemann.TN or scalar
   :return: Power of the input tensor
   :rtype: riemann.TN

.. function:: riemann.log(x)

   Calculate the natural logarithm of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Natural logarithm of each element
   :rtype: riemann.TN

.. function:: riemann.log1p(x)

   Calculate the natural logarithm of each element plus one.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Natural logarithm of each element plus one
   :rtype: riemann.TN

.. function:: riemann.log2(x)

   Calculate the base-2 logarithm of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Base-2 logarithm of each element
   :rtype: riemann.TN

.. function:: riemann.log10(x)

   Calculate the base-10 logarithm of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Base-10 logarithm of each element
   :rtype: riemann.TN

.. function:: riemann.exp(x)

   Calculate the exponential of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Exponential of each element
   :rtype: riemann.TN

.. function:: riemann.exp2(x)

   Calculate 2 raised to the power of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: 2 raised to the power of each element
   :rtype: riemann.TN

.. function:: riemann.square(x)

   Calculate the square of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Square of each element
   :rtype: riemann.TN

.. function:: riemann.sin(x)

   Calculate the sine of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Sine of each element
   :rtype: riemann.TN

.. function:: riemann.cos(x)

   Calculate the cosine of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Cosine of each element
   :rtype: riemann.TN

.. function:: riemann.tan(x)

   Calculate the tangent of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Tangent of each element
   :rtype: riemann.TN

.. function:: riemann.cot(x)

   Calculate the cotangent of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Cotangent of each element
   :rtype: riemann.TN

.. function:: riemann.sec(x)

   Calculate the secant of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Secant of each element
   :rtype: riemann.TN

.. function:: riemann.csc(x)

   Calculate the cosecant of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Cosecant of each element
   :rtype: riemann.TN

.. function:: riemann.asin(x)

   Calculate the arcsine of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Arcsine of each element
   :rtype: riemann.TN

.. function:: riemann.acos(x)

   Calculate the arccosine of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Arccosine of each element
   :rtype: riemann.TN

.. function:: riemann.atan(x)

   Calculate the arctangent of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Arctangent of each element
   :rtype: riemann.TN

.. function:: riemann.sinh(x)

   Calculate the hyperbolic sine of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Hyperbolic sine of each element
   :rtype: riemann.TN

.. function:: riemann.cosh(x)

   Calculate the hyperbolic cosine of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Hyperbolic cosine of each element
   :rtype: riemann.TN

.. function:: riemann.tanh(x)

   Calculate the hyperbolic tangent of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Hyperbolic tangent of each element
   :rtype: riemann.TN

.. function:: riemann.coth(x)

   Calculate the hyperbolic cotangent of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Hyperbolic cotangent of each element
   :rtype: riemann.TN

.. function:: riemann.sech(x)

   Calculate the hyperbolic secant of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Hyperbolic secant of each element
   :rtype: riemann.TN

.. function:: riemann.csch(x)

   Calculate the hyperbolic cosecant of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Hyperbolic cosecant of each element
   :rtype: riemann.TN

.. function:: riemann.arcsinh(x)

   Calculate the inverse hyperbolic sine of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Inverse hyperbolic sine of each element
   :rtype: riemann.TN

.. function:: riemann.arccosh(x)

   Calculate the inverse hyperbolic cosine of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Inverse hyperbolic cosine of each element
   :rtype: riemann.TN

.. function:: riemann.arctanh(x)

   Calculate the inverse hyperbolic tangent of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Inverse hyperbolic tangent of each element
   :rtype: riemann.TN

.. function:: riemann.ceil(x)

   Round up each element to the smallest integer greater than or equal to the element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Ceil of each element
   :rtype: riemann.TN

.. function:: riemann.floor(x)

   Round down each element to the largest integer less than or equal to the element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Floor of each element
   :rtype: riemann.TN

.. function:: riemann.round(x)

   Round each element to the nearest integer.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Rounded tensor
   :rtype: riemann.TN

.. function:: riemann.trunc(x)

   Truncate the decimal part of each element, returning the integer part.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Truncated tensor
   :rtype: riemann.TN

.. function:: riemann.sign(x)

   Calculate the sign of each element.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Sign of each element
   :rtype: riemann.TN

.. function:: riemann.where(cond, x=None, y=None)

   Select elements from x or y based on condition.

   :param cond: Condition tensor
   :type cond: riemann.TN
   :param x: Tensor to select when condition is True
   :type x: riemann.TN, optional
   :param y: Tensor to select when condition is False
   :type y: riemann.TN, optional
   :return: Tensor composed of elements selected from x or y
   :rtype: riemann.TN

.. function:: riemann.clamp(x, min=None, max=None, out=None)

   Clamp all elements within a specified range.

   :param x: Input tensor
   :type x: riemann.TN
   :param min: Minimum value
   :type min: float, optional
   :param max: Maximum value
   :type max: float, optional
   :param out: Output tensor
   :type out: riemann.TN, optional
   :return: Tensor with elements clamped within the specified range
   :rtype: riemann.TN

.. function:: riemann.masked_fill(input, mask, value)

   Fill values into the tensor according to a mask.

   :param input: Input tensor
   :type input: riemann.TN
   :param mask: Mask tensor, same shape as input tensor
   :type mask: riemann.TN
   :param value: Value to fill
   :type value: scalar
   :return: Filled tensor
   :rtype: riemann.TN

.. function:: riemann.maximum(input, other)

   Calculate the element-wise maximum of two tensors.

   :param input: First input tensor
   :type input: riemann.TN
   :param other: Second input tensor
   :type other: riemann.TN
   :return: Tensor composed of element-wise maximum values
   :rtype: riemann.TN

.. function:: riemann.minimum(input, other)

   Calculate the element-wise minimum of two tensors.

   :param input: First input tensor
   :type input: riemann.TN
   :param other: Second input tensor
   :type other: riemann.TN
   :return: Tensor composed of element-wise minimum values
   :rtype: riemann.TN

.. function:: riemann.diagonal(input, offset=0, dim1=-2, dim2=-1)

   Return the diagonal of the tensor.

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

   Return the diagonal of a 2D tensor or construct a diagonal matrix.

   :param input: Input tensor
   :type input: riemann.TN
   :param offset: Offset of the diagonal
   :type offset: int, optional
   :return: Diagonal of the tensor or diagonal matrix
   :rtype: riemann.TN

.. function:: riemann.fill_diagonal(input, value, offset=0, dim1=-2, dim2=-1)

   Fill the diagonal of the tensor with a specified value.

   :param input: Input tensor
   :type input: riemann.TN
   :param value: Value to fill the diagonal
   :type value: scalar
   :param offset: Offset of the diagonal
   :type offset: int, optional
   :param dim1: First dimension of the diagonal
   :type dim1: int, optional
   :param dim2: Second dimension of the diagonal
   :type dim2: int, optional
   :return: Tensor with filled diagonal
   :rtype: riemann.TN

.. function:: riemann.batch_diag(v)

   Return the batch diagonal of the tensor.

   :param v: Input tensor
   :type v: riemann.TN
   :return: Batch diagonal of the tensor
   :rtype: riemann.TN

.. function:: riemann.nonzero(input, *, as_tuple=False)

   Return the indices of non-zero elements.

   :param input: Input tensor
   :type input: riemann.TN
   :param as_tuple: Whether to return as a tuple of tensors
   :type as_tuple: bool, optional
   :return: Indices of non-zero elements
   :rtype: riemann.TN or tuple of riemann.TN

.. function:: riemann.tril(input_tensor, diagonal=0)

   Return the lower triangular part of the matrix.

   :param input_tensor: Input tensor
   :type input_tensor: riemann.TN
   :param diagonal: Diagonal offset
   :type diagonal: int, optional
   :return: Lower triangular part of the matrix
   :rtype: riemann.TN

.. function:: riemann.triu(input_tensor, diagonal=0)

   Return the upper triangular part of the matrix.

   :param input_tensor: Input tensor
   :type input_tensor: riemann.TN
   :param diagonal: Diagonal offset
   :type diagonal: int, optional
   :return: Upper triangular part of the matrix
   :rtype: riemann.TN

.. function:: riemann.cumsum(input, dim, *, dtype=None)

   Calculate the cumulative sum of a tensor along a specified dimension.

   :param input: Input tensor
   :type input: riemann.TN
   :param dim: Dimension along which to compute cumulative sum
   :type dim: int
   :param dtype: Data type of the output tensor
   :type dtype: riemann.dtype, optional
   :return: Cumulative sum result
   :rtype: riemann.TN

.. function:: riemann.unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None)

   Return the unique elements of a tensor.

   :param input: Input tensor
   :type input: riemann.TN
   :param sorted: Whether to sort the unique values
   :type sorted: bool, optional
   :param return_inverse: Whether to return inverse indices
   :type return_inverse: bool, optional
   :param return_counts: Whether to return counts of each unique value
   :type return_counts: bool, optional
   :param dim: Dimension along which to find unique values, default is None (flattened)
   :type dim: int, optional
   :return: Unique values, or tuple if return_inverse or return_counts is specified
   :rtype: riemann.TN or tuple

.. function:: riemann.broadcast_tensors(*tensors)

   Broadcast multiple tensors to the same shape.

   :param tensors: Sequence of tensors to broadcast
   :type tensors: riemann.TN
   :return: List of broadcasted tensors
   :rtype: list of riemann.TN

.. function:: riemann.repeat(input, repeats, dim=None)

   Repeat tensor elements along a specified dimension.

   :param input: Input tensor
   :type input: riemann.TN
   :param repeats: Number of repetitions for each element
   :type repeats: int
   :param dim: Dimension along which to repeat, default is None (flattened)
   :type dim: int, optional
   :return: Repeated tensor
   :rtype: riemann.TN

.. function:: riemann.outer(input, vec2)

   Compute the outer product of two vectors.

   :param input: First input vector
   :type input: riemann.TN
   :param vec2: Second input vector
   :type vec2: riemann.TN
   :return: Outer product matrix
   :rtype: riemann.TN

Comparison Operations
~~~~~~~~~~~~~~~~~~~~~

.. function:: riemann.equal(a, b)

   Calculate element-wise equality.

   :param a: First tensor
   :type a: riemann.TN
   :param b: Second tensor
   :type b: riemann.TN
   :return: Boolean tensor indicating equality
   :rtype: bool

.. function:: riemann.not_equal(a, b)

   Calculate element-wise inequality.

   :param a: First tensor
   :type a: riemann.TN
   :param b: Second tensor
   :type b: riemann.TN
   :return: Boolean tensor indicating inequality
   :rtype: bool

.. function:: riemann.allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False)

   Return True if two tensors are element-wise equal within a tolerance.

   :param a: First tensor
   :type a: riemann.TN
   :param b: Second tensor
   :type b: riemann.TN
   :param rtol: Relative tolerance
   :type rtol: float, optional
   :param atol: Absolute tolerance
   :type atol: float, optional
   :param equal_nan: Whether to treat NaN values as equal
   :type equal_nan: bool, optional
   :return: Whether the tensors are close
   :rtype: bool

.. function:: riemann.isinf(x)

   Element-wise test for infinity.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Boolean tensor indicating infinity
   :rtype: riemann.TN

.. function:: riemann.isnan(x)

   Element-wise test for NaN.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Boolean tensor indicating NaN
   :rtype: riemann.TN

.. function:: riemann.isreal(x)

   Element-wise test for real numbers.

   :param x: Input tensor
   :type x: riemann.TN
   :return: Boolean tensor indicating real numbers
   :rtype: riemann.TN

Sorting Operations
~~~~~~~~~~~~~~~~~~

.. function:: riemann.sort(input, dim=-1, descending=False, stable=False, *, out=None)

   Sort tensor elements along a given dimension.

   :param input: Input tensor
   :type input: riemann.TN
   :param dim: Dimension to sort
   :type dim: int, optional
   :param descending: Whether to sort in descending order
   :type descending: bool, optional
   :param stable: Whether to use stable sort
   :type stable: bool, optional
   :param out: Output tensor
   :type out: riemann.TN, optional
   :return: Sorted tensor and indices
   :rtype: riemann.TN, riemann.TN

.. function:: riemann.argsort(input, dim=-1, descending=False, stable=False, *, out=None)

   Return indices that would sort the tensor along a given dimension.

   :param input: Input tensor
   :type input: riemann.TN
   :param dim: Dimension to sort
   :type dim: int, optional
   :param descending: Whether to sort in descending order
   :type descending: bool, optional
   :param stable: Whether to use stable sort
   :type stable: bool, optional
   :param out: Output tensor
   :type out: riemann.TN, optional
   :return: Sorting indices
   :rtype: riemann.TN

In-place Operations
~~~~~~~~~~~~~~~~~~~

.. method:: riemann.TN.setat_(index, val)

   In-place set values at specified positions in the tensor.

   :param index: Index specifying positions to set values
   :type index: int, slice, tuple, or array
   :param val: Values to set
   :type val: riemann.TN, numpy.ndarray, list, or scalar
   :return: In-place modified tensor
   :rtype: riemann.TN

.. method:: riemann.TN.addat_(index, val)

   In-place add values to specified positions in the tensor.

   :param index: Index specifying positions to operate
   :type index: int, slice, tuple, or array
   :param val: Values to add
   :type val: riemann.TN, numpy.ndarray, list, or scalar
   :return: In-place modified tensor
   :rtype: riemann.TN

.. method:: riemann.TN.subat_(index, val)

   In-place subtract values from specified positions in the tensor.

   :param index: Index specifying positions to operate
   :type index: int, slice, tuple, or array
   :param val: Values to subtract
   :type val: riemann.TN, numpy.ndarray, list, or scalar
   :return: In-place modified tensor
   :rtype: riemann.TN

.. method:: riemann.TN.mulat_(index, val)

   In-place multiply specified positions in the tensor by given values.

   :param index: Index specifying positions to operate
   :type index: int, slice, tuple, or array
   :param val: Values to multiply
   :type val: riemann.TN, numpy.ndarray, list, or scalar
   :return: In-place modified tensor
   :rtype: riemann.TN

.. method:: riemann.TN.divat_(index, val)

   In-place divide specified positions in the tensor by given values.

   :param index: Index specifying positions to operate
   :type index: int, slice, tuple, or array
   :param val: Divisor
   :type val: riemann.TN, numpy.ndarray, list, or scalar
   :return: In-place modified tensor
   :rtype: riemann.TN

.. method:: riemann.TN.powat_(index, val)

   In-place exponentiate specified positions in the tensor.

   :param index: Index specifying positions to operate
   :type index: int, slice, tuple, or array
   :param val: Exponent
   :type val: riemann.TN, numpy.ndarray, list, or scalar
   :return: In-place modified tensor
   :rtype: riemann.TN

.. method:: riemann.TN.scatter_(dim, index, src=None, *, value=None)
   :no-index:

   In-place fill values into the tensor according to indices.

   :param dim: Dimension along which to index
   :type dim: int
   :param index: Index tensor
   :type index: riemann.TN
   :param src: Source tensor providing values to fill
   :type src: riemann.TN, optional
   :param value: Scalar value providing values to fill
   :type value: int, float, complex, optional
   :return: In-place modified tensor
   :rtype: riemann.TN

.. method:: riemann.TN.scatter_add_(dim, index, src)
   :no-index:

   In-place accumulate values into the tensor according to indices.

   :param dim: Dimension along which to index
   :type dim: int
   :param index: Index tensor
   :type index: riemann.TN
   :param src: Source tensor providing values to accumulate
   :type src: riemann.TN
   :return: In-place modified tensor
   :rtype: riemann.TN

.. method:: riemann.TN.requires_grad_(requires_grad=True)

   In-place set whether the tensor requires gradient calculation.

   :param requires_grad: Whether to require gradient calculation
   :type requires_grad: bool, optional
   :return: In-place modified tensor
   :rtype: riemann.TN

.. method:: riemann.TN.add_(other)

   In-place addition operation.

   :param other: Value to add to the current tensor
   :type other: riemann.TN, numpy.ndarray, list, or scalar
   :return: In-place modified tensor
   :rtype: riemann.TN

.. method:: riemann.TN.sub_(other)

   In-place subtraction operation.

   :param other: Value to subtract from the current tensor
   :type other: riemann.TN, numpy.ndarray, list, or scalar
   :return: In-place modified tensor
   :rtype: riemann.TN

.. method:: riemann.TN.mul_(other)

   In-place multiplication operation.

   :param other: Value to multiply with the current tensor
   :type other: riemann.TN, numpy.ndarray, list, or scalar
   :return: In-place modified tensor
   :rtype: riemann.TN

.. method:: riemann.TN.div_(other)

   In-place division operation.

   :param other: Divisor
   :type other: riemann.TN, numpy.ndarray, list, or scalar
   :return: In-place modified tensor
   :rtype: riemann.TN

.. method:: riemann.TN.pow_(other)

   In-place power operation.

   :param other: Exponent
   :type other: riemann.TN, numpy.ndarray, list, or scalar
   :return: In-place modified tensor
   :rtype: riemann.TN

.. method:: riemann.TN.detach_()

   In-place detach the tensor from the computation graph.

   :return: In-place modified tensor
   :rtype: riemann.TN

.. method:: riemann.TN.copy_(src)

   In-place copy source tensor to current tensor.

   :param src: Source tensor
   :type src: riemann.TN, numpy.ndarray, list, or scalar
   :return: In-place modified tensor
   :rtype: riemann.TN

.. method:: riemann.TN.zero_()

   In-place set all elements of the tensor to 0.

   :return: In-place modified tensor
   :rtype: riemann.TN

.. method:: riemann.TN.fill_(value)

   In-place fill all elements of the tensor with a specified value.

   :param value: Fill value
   :type value: riemann.TN, numpy.ndarray, list, or scalar
   :return: In-place modified tensor
   :rtype: riemann.TN

.. method:: riemann.TN.clamp_(min=None, max=None)

   In-place clamp tensor elements within a specified range.

   :param min: Minimum value
   :type min: float, optional
   :param max: Maximum value
   :type max: float, optional
   :return: In-place modified tensor
   :rtype: riemann.TN

.. function:: riemann.masked_fill_(input, mask, value)

   In-place version of masked_fill function, fill values into the tensor according to a mask.

   :param input: Input tensor
   :type input: riemann.TN
   :param mask: Mask tensor, same shape as input tensor
   :type mask: riemann.TN
   :param value: Value to fill
   :type value: scalar
   :return: In-place modified tensor
   :rtype: riemann.TN

.. function:: riemann.fill_diagonal_(input, value, offset=0, dim1=-2, dim2=-1)

   In-place version of fill_diagonal.

   :param input: Input tensor
   :type input: riemann.TN
   :param value: Value to fill the diagonal
   :type value: scalar
   :param offset: Offset of the diagonal
   :type offset: int, optional
   :param dim1: First dimension of the diagonal
   :type dim1: int, optional
   :param dim2: Second dimension of the diagonal
   :type dim2: int, optional
   :return: Input tensor with filled diagonal
   :rtype: riemann.TN

Gather and Scatter Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. method:: riemann.TN.gather(dim, index)

   Gather elements according to specified dimension and indices.

   :param dim: Gathering dimension
   :type dim: int
   :param index: Index tensor
   :type index: riemann.TN
   :return: Gathered tensor
   :rtype: riemann.TN

.. method:: riemann.TN.scatter(dim, index, src=None, *, value=None)

   Fill values into a new tensor according to indices.

   :param dim: Dimension along which to index
   :type dim: int
   :param index: Index tensor
   :type index: riemann.TN
   :param src: Source tensor providing values to fill
   :type src: riemann.TN, optional
   :param value: Scalar value providing values to fill
   :type value: int, float, complex, optional
   :return: New tensor with filled values
   :rtype: riemann.TN

.. method:: riemann.TN.scatter_(dim, index, src=None, *, value=None)

   In-place fill values into the tensor according to indices.

   :param dim: Dimension along which to index
   :type dim: int
   :param index: Index tensor
   :type index: riemann.TN
   :param src: Source tensor providing values to fill
   :type src: riemann.TN, optional
   :param value: Scalar value providing values to fill
   :type value: int, float, complex, optional
   :return: In-place modified tensor
   :rtype: riemann.TN

.. method:: riemann.TN.scatter_add(dim, index, src)

   Accumulate values into a new tensor according to indices.

   :param dim: Dimension along which to index
   :type dim: int
   :param index: Index tensor
   :type index: riemann.TN
   :param src: Source tensor providing values to accumulate
   :type src: riemann.TN
   :return: New tensor with accumulated values
   :rtype: riemann.TN

.. method:: riemann.TN.scatter_add_(dim, index, src)

   In-place accumulate values into the tensor according to indices.

   :param dim: Dimension along which to index
   :type dim: int
   :param index: Index tensor
   :type index: riemann.TN
   :param src: Source tensor providing values to accumulate
   :type src: riemann.TN
   :return: In-place modified tensor
   :rtype: riemann.TN

Data Conversion
~~~~~~~~~~~~~~~

.. function:: riemann.from_numpy(arr, requires_grad=False)

   Convert a NumPy array to a Riemann tensor.

   :param arr: Input NumPy array
   :type arr: numpy.ndarray
   :param requires_grad: Whether to track operations on this tensor
   :type requires_grad: bool, optional
   :return: Riemann tensor
   :rtype: riemann.TN

.. function:: riemann.item(tensor)

   Convert a single-element tensor to a Python scalar.

   :param tensor: Input tensor
   :type tensor: riemann.TN
   :return: Python scalar
   :rtype: int, float, etc.

.. method:: riemann.TN.tolist()

   Convert the tensor to a Python list.

   :return: Python list or scalar
   :rtype: list, int, float, complex

.. method:: riemann.TN.numpy()

   Convert the tensor to a NumPy array.

   :return: NumPy array
   :rtype: numpy.ndarray

.. method:: riemann.TN.to(*args, **kwargs)

   Convert the tensor to a specified data type and/or device.

   :param dtype: Target data type
   :type dtype: dtype, optional
   :param device: Target device
   :type device: str, Device, optional
   :return: Converted tensor
   :rtype: riemann.TN

.. method:: riemann.TN.type(dtype=None)

   Return or convert the data type of the tensor.

   :param dtype: Data type, if None return current data type
   :type dtype: dtype, optional
   :return: Current data type if dtype is None, otherwise converted data type
   :rtype: dtype or riemann.TN

.. method:: riemann.TN.type_as(other)

   Convert the tensor to the same data type as another tensor.

   :param other: Reference tensor for target data type
   :type other: riemann.TN
   :return: Converted data type
   :rtype: riemann.TN

.. method:: riemann.TN.bool()

   Convert the tensor to boolean type.

   :return: Boolean type tensor
   :rtype: riemann.TN

.. method:: riemann.TN.float()

   Convert the tensor to single-precision floating point type (float32).

   :return: float32 type tensor
   :rtype: riemann.TN

.. method:: riemann.TN.double()

   Convert the tensor to double-precision floating point type (float64).

   :return: float64 type tensor
   :rtype: riemann.TN

Copy Functions
~~~~~~~~~~~~~~

.. function:: riemann.clone(tensor)

   Return a copy of the tensor.

   :param tensor: Input tensor
   :type tensor: riemann.TN
   :return: Tensor copy
   :rtype: riemann.TN

.. method:: riemann.TN.copy()

   Return a copy of the tensor, not sharing memory and not dependent on the original tensor.

   :return: Tensor copy
   :rtype: riemann.TN

.. function:: riemann.detach(tensor)

   Detach the tensor from the computation graph, stopping gradient tracking.

   :param tensor: Input tensor
   :type tensor: riemann.TN
   :return: Detached tensor
   :rtype: riemann.TN

Data Types
----------

.. module:: riemann.dtype

Predefined Data Types
~~~~~~~~~~~~~~~~~~~~~

.. function:: riemann.is_numeric_array(numpy_arr)

   Check if a NumPy array has a numeric data type

   :param numpy_arr: The NumPy array to check
   :type numpy_arr: numpy.ndarray
   :return: Whether the array has a numeric data type
   :rtype: bool

.. function:: riemann.is_number(v)

   Check if a value is a numeric type

   :param v: The value to check
   :type v: Any
   :return: Whether the value is a numeric type
   :rtype: bool

.. function:: riemann.is_float_or_complex(dtype)

   Check if a data type is a floating point or complex number type

   :param dtype: The data type to check
   :type dtype: numpy.dtype
   :return: Whether the data type is a floating point or complex number type
   :rtype: bool

.. function:: riemann.is_scalar(value)

   Check if a value is a scalar (including Riemann tensor scalars)

   :param value: The value to check
   :type value: Any
   :return: Whether the value is a scalar
   :rtype: bool

Data Type Inference
~~~~~~~~~~~~~~~~~~~

.. function:: riemann.infer_data_type(v)

   Infer an appropriate data type from Python values, NumPy arrays, or collections of values

   :param v: The value or collection of values from which to infer the data type
   :type v: Any
   :return: The inferred data type
   :rtype: numpy.dtype

Gradient Mode Control
---------------------

.. module:: riemann.gradmode

.. function:: riemann.is_grad_enabled()

   Get the gradient computation state for the current thread

   :return: The current gradient computation mode (True for enabled, False for disabled)
   :rtype: bool

.. function:: riemann.no_grad(func=None)

   Context manager to temporarily disable gradient computation

   Can also be used as a function decorator, disabling gradient tracking for all computations within the decorated function.

   :param func: Optional, if provided, applies no_grad as a decorator to the function
   :type func: callable, optional
   :return: If func is not provided, returns a context manager instance; if func is provided, returns the decorated function

   Example:
   
   .. code-block:: python
      
      # Used as a context manager
      with riemann.no_grad():
          # Computations within this block will not track gradients
          output = model(input_data)
      
      # Used as a decorator
      @riemann.no_grad
      def inference(x):
          # Computations within this function will not track gradients
          return model(x)

.. function:: riemann.enable_grad(func=None)

   Context manager to temporarily enable gradient computation

   Can also be used as a function decorator, ensuring that computations within the decorated function track gradients.

   :param func: Optional, if provided, applies enable_grad as a decorator to the function
   :type func: callable, optional
   :return: If func is not provided, returns a context manager instance; if func is provided, returns the decorated function

   Example:
   
   .. code-block:: python
      
      # Used as a context manager
      with riemann.enable_grad():
          # Computations within this block will track gradients
          output = model(input_data)
          loss = loss_fn(output, target)
          loss.backward()
      
      # Used as a decorator
      @riemann.enable_grad
      def train_step(x, y):
          # Computations within this function will track gradients
          pred = model(x)
          loss = loss_fn(pred, y)
          loss.backward()
          return loss

.. function:: riemann.set_grad_enabled(mode=True, func=None)

   Context manager to set the gradient computation mode

   Similar to PyTorch's set_grad_enabled(), it can explicitly enable or disable gradient computation.
   Supports usage as both a context manager and a decorator, providing the most flexible way to control gradients.

   :param mode: If True, enables gradient computation; if False, disables gradient computation
   :type mode: bool
   :param func: Optional, the function passed when used as a decorator
   :type func: callable, optional
   :return: If func is None, returns a context manager instance; if the func parameter is provided, returns the wrapped function

   Example:
   
   .. code-block:: python
      
      # Used as a context manager
      with riemann.set_grad_enabled(False):
          # Computations within this block will not track gradients
          output = model(input_data)
      
      with riemann.set_grad_enabled(True):
          # Computations within this block will track gradients
          output = model(input_data)
          loss = loss_fn(output, target)
          loss.backward()
      
      # Used as a decorator
      @riemann.set_grad_enabled(False)
      def inference(x):
          return model(x)
      
      @riemann.set_grad_enabled(True)
      def train(x, y):
          pred = model(x)
          loss = loss_fn(pred, y)
          loss.backward()
          return loss


Serialization
-------------

.. module:: riemann.serialization

.. function:: riemann.save(obj, f, pickle_module=None, pickle_protocol=2, use_new_zipfile_serialization=True)

   Save an object to a disk file.

   This function uses pickle serialization to save Riemann tensors, parameters, modules, or any Python objects to a disk file.

   :param obj: The object to save. Can be a tensor, parameter, module, or any picklable object
   :type obj: Any
   :param f: File path or file-like object to write to
   :type f: str, os.PathLike, or file-like object
   :param pickle_module: Module to use for pickling (default: pickle)
   :type pickle_module: Any, optional
   :param pickle_protocol: Pickle protocol version (default: 2)
   :type pickle_protocol: int, optional
   :param use_new_zipfile_serialization: Whether to use zip-based serialization (default: True)
   :type use_new_zipfile_serialization: bool, optional

   Example:
       >>> import riemann as rm
       >>> # Save a tensor
       >>> tensor = rm.randn(3, 4)
       >>> rm.save(tensor, 'tensor.pt')
       >>> 
       >>> # Save a module
       >>> model = rm.nn.Linear(10, 5)
       >>> rm.save(model.state_dict(), 'model_weights.pt')
       >>> 
       >>> # Save multiple objects
       >>> rm.save({
       ...     'model': model.state_dict(),
       ...     'optimizer_state': optimizer.state_dict(),
       ...     'epoch': 10
       ... }, 'checkpoint.pt')

.. function:: riemann.load(f, map_location=None, pickle_module=None, **pickle_load_args)

   Load an object from a disk file.

   This function uses pickle deserialization to load Riemann tensors, parameters, modules, or any Python objects from a disk file.

   :param f: File path or file-like object to read from
   :type f: str, os.PathLike, or file-like object
   :param map_location: Function or dictionary for remapping storage locations
   :type map_location: Any, optional
   :param pickle_module: Module to use for unpickling (default: pickle)
   :type pickle_module: Any, optional
   :param \**pickle_load_args: Additional arguments passed to pickle.load
   :return: The loaded object

   Example:
       >>> import riemann as rm
       >>> # Load a tensor
       >>> tensor = rm.load('tensor.pt')
       >>> 
       >>> # Load model weights
       >>> state_dict = rm.load('model_weights.pt')
       >>> model.load_state_dict(state_dict)
       >>> 
       >>> # Load a checkpoint
       >>> checkpoint = rm.load('checkpoint.pt')
       >>> model.load_state_dict(checkpoint['model'])
       >>> optimizer.load_state_dict(checkpoint['optimizer_state'])
       >>> epoch = checkpoint['epoch']

CUDA Support
------------

.. module:: riemann.cuda

.. class:: riemann.cuda.Device(device='cpu')

   Represents a device (CPU or CUDA GPU).

   :param device: Device type or index. Can be:
       - String: 'cpu', 'cuda' or 'cuda:0', 'cuda:1'
       - Integer: CUDA device index
   :type device: str or int

   Example:
       >>> import riemann as rm
       >>> # Create a CPU device
       >>> cpu_device = rm.Device('cpu')
       >>> # Create a CUDA device
       >>> cuda_device = rm.Device('cuda')
       >>> # Create a specific CUDA device
       >>> cuda_device_1 = rm.Device('cuda:1')
       >>> # Create a CUDA device by type and index
       >>> cuda_device_2 = rm.Device('cuda', 2)

   .. method:: __enter__()

      Enter the device context.

   .. method:: __exit__(exc_type, exc_val, exc_tb)

      Exit the device context.

   .. method:: __eq__(other)

      Compare with another device.

   .. method:: __str__()

      Return the string representation of the device.

   .. method:: __repr__()

      Return the official string representation of the device.

.. function:: riemann.cuda.is_available()

   Check if CUDA is available.

   :return: True if CUDA is available, False otherwise
   :rtype: bool

.. function:: riemann.cuda.device_count()

   Return the number of available CUDA devices.

   :return: Number of available CUDA devices
   :rtype: int

.. function:: riemann.cuda.current_device()

   Return the index of the current CUDA device.

   :return: Index of the current CUDA device
   :rtype: int

.. function:: riemann.cuda.get_device_name(device_idx)

   Return the name of the CUDA device at the given index.

   :param device_idx: Index of the CUDA device
   :type device_idx: int
   :return: Name of the CUDA device
   :rtype: str

.. function:: riemann.cuda.set_device(device_idx)

   Set the current CUDA device.

   :param device_idx: CUDA device index to set as current
   :type device_idx: int

.. function:: riemann.cuda.empty_cache()

   Clear the CUDA cache.

.. function:: riemann.cuda.synchronize(device=None)

   Wait for all operations on the current CUDA device to complete.

   :param device: Device to synchronize, default is current device
   :type device: str, int, or Device, optional

.. function:: riemann.cuda.is_in_cuda_context()

   Check if the current thread is in a CUDA device context.

   :return: True if in a CUDA device context, False otherwise
   :rtype: bool

.. function:: riemann.memory_allocated(device_idx=None)

   Return the amount of memory allocated on the given CUDA device.

   :param device_idx: Index of the CUDA device. If None, use the current device
   :type device_idx: int, optional
   :return: Amount of allocated memory (bytes)
   :rtype: int

.. function:: riemann.get_default_device()

   Get the default device for tensor creation.

   :return: The default device
   :rtype: Device

.. function:: riemann.set_default_device(device)

   Set the default device for tensor creation.

   :param device: The device to set as default. Can be:
       - String: 'cpu', 'cuda' or 'cuda:0', 'cuda:1'
       - Integer: CUDA device index
       - Device object
   :type device: str, int, or Device

   Example:
       >>> import riemann as rm
       >>> rm.get_default_device()
       device(type='cpu', index=None)
       >>> rm.set_default_device('cuda')
       >>> rm.get_default_device()
       device(type='cuda', index=0)
       >>> rm.set_default_device('cuda:1')
       >>> rm.get_default_device()
       device(type='cuda', index=1)

Automatic Differentiation
-------------------------

.. module:: riemann.autograd

Gradient Computation
~~~~~~~~~~~~~~~~~~~~

.. function:: riemann.autograd.backward(self, gradient=None, retain_graph=False, create_graph=False)

   Perform reverse-mode automatic differentiation (backpropagation).

   Starting from the current tensor, propagate gradients backward through the computation graph, 
   computing and storing gradients for all leaf nodes or intermediate nodes with retains_grad=True.

   :param self: The tensor that triggers backpropagation
   :type self: riemann.TN
   :param gradient: Gradient of the output tensor, defaults to None
   :type gradient: riemann.TN or None, optional
   :param retain_graph: This parameter is for PyTorch compatibility, Riemann backpropagation does not rely on it
   :type retain_graph: bool, optional
   :param create_graph: Whether to create a computation graph during gradient computation, set to True for higher-order derivative calculations
   :type create_graph: bool, optional

.. function:: riemann.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=False, allow_unused=False)

   Compute and return the gradients of outputs with respect to inputs.

   This is the core gradient computation function in the Riemann framework. Similar to the backward() method,
   but it directly returns the computed gradient tensors instead of storing them in the .grad attribute of input tensors.
   This makes it more suitable for advanced gradient computation scenarios, such as calculating Jacobian matrices,
   Hessian matrices, etc.

   :param outputs: Output tensor(s) for which to compute gradients
   :type outputs: riemann.TN
   :param inputs: Input tensor(s) or list/tuple of input tensors for which to compute gradients
   :type inputs: riemann.TN or list/tuple of riemann.TN
   :param grad_outputs: Gradient(s) of the output tensor(s), defaults to None
   :type grad_outputs: riemann.TN or None, optional
   :param retain_graph: This parameter is for PyTorch compatibility, Riemann backpropagation does not rely on it
   :type retain_graph: bool, optional
   :param create_graph: Whether to create a computation graph during gradient computation
   :type create_graph: bool, optional
   :param allow_unused: Whether to allow unused inputs
   :type allow_unused: bool, optional
   :return: Tuple of gradient tensors corresponding to the inputs
   :rtype: tuple of riemann.TN

.. function:: riemann.autograd.higher_order_grad(outputs, inputs, n, create_graph=False)

   Compute the n-th order derivative of a scalar tensor output with respect to each tensor in inputs.

   This function computes higher-order derivatives by recursively calling grad(). For each input tensor,
   it computes the n-th order derivative and returns a tuple of derivatives corresponding to the input list.

   :param outputs: Scalar tensor output for which to compute higher-order derivatives
   :type outputs: riemann.TN
   :param inputs: Input tensor(s) or list/tuple of input tensors for which to compute higher-order derivatives
   :type inputs: riemann.TN or list/tuple of riemann.TN
   :param n: Order of the derivative, must be a non-negative integer
   :type n: int
   :param create_graph: Whether to create a computation graph during gradient computation
   :type create_graph: bool, optional
   :return: Tuple of n-th order derivative tensors corresponding to the inputs
   :rtype: tuple of riemann.TN

.. function:: riemann.autograd.gradcheck(func, inputs, eps=1e-6, atol=1e-5, rtol=1e-3, raise_exception=True, check_sparse_nnz=False, fast_mode=False)

   Verify the correctness of gradient computation for a given function by comparing numerical and analytical gradients.

   This function computes numerical gradients using finite differences by adding small perturbations to input parameters,
   and compares them with analytical gradients computed using automatic differentiation.

   :param func: Function for which to verify gradients
   :type func: callable
   :param inputs: Tuple of input tensors for testing
   :type inputs: tuple of riemann.TN
   :param eps: Small perturbation for numerical gradient computation
   :type eps: float, optional
   :param atol: Absolute error tolerance
   :type atol: float, optional
   :param rtol: Relative error tolerance
   :type rtol: float, optional
   :param raise_exception: Whether to raise an exception if gradient check fails
   :type raise_exception: bool, optional
   :param check_sparse_nnz: Whether to check sparse tensor non-zero elements (not supported in current version)
   :type check_sparse_nnz: bool, optional
   :param fast_mode: Whether to use fast mode (only check the first element)
   :type fast_mode: bool, optional
   :return: True if gradient check passes, False otherwise
   :rtype: bool

.. function:: riemann.track_grad(grad_func)

   Create a gradient tracking decorator for adding automatic differentiation support to functions.

   This decorator factory receives a gradient function and returns a decorator that can convert ordinary tensor operation functions
   into functions that support automatic differentiation. It automatically creates backpropagation functions and manages gradient computation graph construction.

   :param grad_func: Gradient computation function that receives the same input parameters as the forward function,
                    returns a tuple containing gradients (partial derivatives) for each input tensor.
                    Elements in the tuple must correspond one-to-one with the input tensors of the forward function. For tensors that don't require gradients, the corresponding gradient value should be None.
   :type grad_func: callable
   :return: A decorator function for wrapping forward computation functions to support automatic differentiation
   :rtype: callable

   Example:
   
   .. code-block:: python
      
      # Define single-input derivative function (d/dx log(x) = 1/x)
      def _log_derivative(x):
          return (1. / x.conj(),)
      
      # Use track_grad decorator to create automatic differentiation-supported log function
      @track_grad(_log_derivative)
      def mylog(x):
          return tensor(np.log(x.data))
      
      # Use automatic differentiation-supported log function
      x = tensor(2., requires_grad=True)
      y = mylog(x)
      y.backward()
      print(f'x.grad = {x.grad}')  # Output: x.grad = 0.5
      
      # Define multi-input derivative function (d/dx (x + y) = 1, d/dy (x + y) = 1)
      def _add_derivative(x, y):
          return (tensor(1.), tensor(1.))
      
      # Use track_grad decorator to create automatic differentiation-supported addition function
      @track_grad(_add_derivative)
      def myadd(x, y):
          return tensor(x.data + y.data)
      
      # Use automatic differentiation-supported addition function
      x = tensor(2., requires_grad=True)
      y = tensor(3., requires_grad=True)
      z = myadd(x, y)
      z.backward()
      print(f'x.grad = {x.grad}')  # Output: x.grad = 1.0
      print(f'y.grad = {y.grad}')  # Output: y.grad = 1.0

.. class:: riemann.autograd.Function

   Base class for custom gradient implementations in the Riemann framework, designed with an interface similar to PyTorch's torch.autograd.Function.

   To use this class, inherit from it and implement the forward and backward static methods:
   - forward: Perform forward computation, return output tensor(s)
   - backward: Receive output gradient(s), return input gradient(s)

   Example:
   
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

Functional Differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: riemann.autograd.functional.jacobian(func, inputs, create_graph=False, strict=True)

   Compute the Jacobian matrix of a function.

   This function computes the Jacobian matrix of a given function at the input point, supporting single or multiple inputs,
   single or multiple outputs, and maintains compatibility with PyTorch's jacobian function behavior.

   :param func: Function for which to compute the Jacobian matrix
   :type func: callable
   :param inputs: Input tensor(s) or list/tuple of input tensors for the function
   :type inputs: riemann.TN or list/tuple of riemann.TN
   :param create_graph: Whether to create a computation graph during gradient computation
   :type create_graph: bool, optional
   :param strict: Whether to strictly follow PyTorch's behavior specifications
   :type strict: bool, optional
   :return: Jacobian matrix representation corresponding to the input/output types
   :rtype: riemann.TN or list/tuple of riemann.TN

.. function:: riemann.autograd.functional.hessian(func, inputs, create_graph=False, strict=True)

   Compute the Hessian matrix of a function.

   This function computes the Hessian matrix of a given function at the input point, which is the Jacobian matrix of the gradient.
   It supports single or multiple inputs and maintains compatibility with PyTorch's hessian function behavior.

   :param func: Scalar-valued function for which to compute the Hessian matrix
   :type func: callable
   :param inputs: Input tensor(s) or list/tuple of input tensors for the function
   :type inputs: riemann.TN or list/tuple of riemann.TN
   :param create_graph: Whether to create a computation graph during gradient computation
   :type create_graph: bool, optional
   :param strict: If True, raises an error when output is detected to be independent of input
   :type strict: bool, optional
   :return: Hessian matrix representation corresponding to the input types
   :rtype: riemann.TN or list/tuple of riemann.TN

.. function:: riemann.autograd.functional.jvp(func, inputs, v=None, create_graph=False, strict=False)

   Compute the Jacobian-Vector Product (JVP).

   :param func: Function for which to compute JVP
   :type func: callable
   :param inputs: Input tensor(s) or list/tuple of input tensors for the function
   :type inputs: riemann.TN or list/tuple of riemann.TN
   :param v: Vector to multiply with the Jacobian matrix
   :type v: riemann.TN or list/tuple of riemann.TN, optional
   :param create_graph: Whether to create a computation graph during gradient computation, for higher-order derivative calculations
   :type create_graph: bool, optional
   :param strict: Whether to raise an error for unused inputs
   :type strict: bool, optional
   :return: Function output and JVP value
   :rtype: tuple of (riemann.TN, riemann.TN or list/tuple of riemann.TN)

.. function:: riemann.autograd.functional.vjp(func, inputs, v=None, create_graph=False, strict=False)

   Compute the Vector-Jacobian Product (VJP).

   :param func: Function for which to compute VJP
   :type func: callable
   :param inputs: Input tensor(s) or list/tuple of input tensors for the function
   :type inputs: riemann.TN or list/tuple of riemann.TN
   :param v: Vector to multiply with the Jacobian matrix
   :type v: riemann.TN or list/tuple of riemann.TN, optional
   :param create_graph: Whether to create a computation graph during gradient computation, for higher-order derivative calculations
   :type create_graph: bool, optional
   :param strict: Whether to raise an error for unused inputs
   :type strict: bool, optional
   :return: Function output and VJP value
   :rtype: tuple of (riemann.TN, riemann.TN or list/tuple of riemann.TN)

.. function:: riemann.autograd.functional.hvp(func, inputs, v, create_graph=False, strict=False)

   Compute the Hessian-Vector Product (HVP).

   :param func: Scalar-valued function for which to compute HVP
   :type func: callable
   :param inputs: Input tensor(s) or list/tuple of input tensors for the function
   :type inputs: riemann.TN or list/tuple of riemann.TN
   :param v: Vector to multiply with the Hessian matrix
   :type v: riemann.TN or list/tuple of riemann.TN
   :param create_graph: Whether to create a computation graph during gradient computation, for higher-order derivative calculations
   :type create_graph: bool, optional
   :param strict: Whether to raise an error for unused inputs
   :type strict: bool, optional
   :return: Function output and HVP value
   :rtype: tuple of (riemann.TN, riemann.TN or list/tuple of riemann.TN)

.. function:: riemann.autograd.functional.vhp(func, inputs, v, create_graph=False, strict=False)

   Compute the Vector-Hessian Product (VHP).

   :param func: Scalar-valued function for which to compute VHP
   :type func: callable
   :param inputs: Input tensor(s) or list/tuple of input tensors for the function
   :type inputs: riemann.TN or list/tuple of riemann.TN
   :param v: Vector to multiply with the Hessian matrix
   :type v: riemann.TN or list/tuple of riemann.TN
   :param create_graph: Whether to create a computation graph during gradient computation, for higher-order derivative calculations
   :type create_graph: bool, optional
   :param strict: Whether to raise an error for unused inputs
   :type strict: bool, optional
   :return: Function output and VHP value
   :rtype: tuple of (riemann.TN, riemann.TN or list/tuple of riemann.TN)

.. function:: riemann.autograd.functional.derivative(func, create_graph=False)

   Compute the derivative function of a function.

   This function returns a new function that, when called, computes the derivative of the original function func at the input point.
   Supports func with single or multiple tensor inputs, returning single or multiple tensors or scalars.
   Internally implements derivative computation based on the jacobian function.

   :param func: Function to differentiate
   :type func: callable
   :param create_graph: Whether to create a computation graph during gradient computation, defaults to False
   :type create_graph: bool, optional
   :return: Derivative function that accepts the same inputs as the original function
   :rtype: callable

Context Managers
----------------

.. function:: riemann.no_grad()

   Context manager to disable gradient computation. Operations within this context will not be recorded in the computation graph.

.. function:: riemann.enable_grad()

   Context manager to enable gradient computation.

.. function:: riemann.set_grad_enabled(mode)

   Context manager that enables or disables gradient computation based on the mode parameter.

   :param mode: True to enable gradient computation, False to disable
   :type mode: bool

Linear Algebra Module
---------------------

The ``riemann.linalg`` module provides various linear algebra operations, including matrix multiplication, decomposition, and solving.

Matrix Operations
~~~~~~~~~~~~~~~~~

.. function:: riemann.linalg.matmul(a, b)

   Compute the matrix product of two tensors.

   :param a: First input tensor
   :type a: riemann.TN
   :param b: Second input tensor
   :type b: riemann.TN
   :return: Matrix product result
   :rtype: riemann.TN

.. function:: riemann.linalg.cross(a, b, dim=-1)

   Compute the cross product (vector product) of two tensors.

   :param a: First input tensor
   :type a: riemann.TN
   :param b: Second input tensor
   :type b: riemann.TN
   :param dim: Dimension along which to compute cross product, default is -1
   :type dim: int, optional
   :return: Cross product result
   :rtype: riemann.TN

Norm Computation
~~~~~~~~~~~~~~~~

.. function:: riemann.linalg.norm(A, ord=None, dim=None, keepdim=False)

   Compute the norm of a tensor or matrix.

   :param A: Input tensor
   :type A: riemann.TN
   :param ord: Order of norm, default is Frobenius norm
   :type ord: int or float or str, optional
   :param dim: Dimension along which to compute norm, default is None (compute norm of all elements)
   :type dim: int or tuple, optional
   :param keepdim: Whether to keep the reduced dimensions
   :type keepdim: bool, optional
   :return: Norm value
   :rtype: riemann.TN

.. function:: riemann.linalg.vector_norm(x, ord=2, dim=None, keepdim=False)

   Compute the vector norm.

   :param x: Input tensor
   :type x: riemann.TN
   :param ord: Order of norm, default is 2 (L2 norm)
   :type ord: float, optional
   :param dim: Dimension along which to compute norm
   :type dim: int or tuple, optional
   :param keepdim: Whether to keep the reduced dimensions
   :type keepdim: bool, optional
   :return: Norm value
   :rtype: riemann.TN

.. function:: riemann.linalg.matrix_norm(A, ord='fro', dim=(-2, -1), keepdim=False)

   Compute the matrix norm.

   :param A: Input tensor
   :type A: riemann.TN
   :param ord: Order of norm, default is 'fro' (Frobenius norm)
   :type ord: str or int, optional
   :param dim: Dimensions of the matrix, default is (-2, -1)
   :type dim: tuple, optional
   :param keepdim: Whether to keep the reduced dimensions
   :type keepdim: bool, optional
   :return: Matrix norm value
   :rtype: riemann.TN

.. function:: riemann.linalg.cond(A, p=None)

   Compute the condition number of a matrix.

   :param A: Input matrix
   :type A: riemann.TN
   :param p: Type of norm, default is None (2-norm condition number)
   :type p: int or float or str, optional
   :return: Condition number
   :rtype: riemann.TN

.. function:: riemann.linalg.svdvals(A)

   Compute the singular values of a matrix.

   :param A: Input matrix
   :type A: riemann.TN
   :return: Singular values
   :rtype: riemann.TN

Matrix Decomposition
~~~~~~~~~~~~~~~~~~~~

.. function:: riemann.linalg.det(A)

   Compute the determinant of a matrix.

   :param A: Input matrix
   :type A: riemann.TN
   :return: Determinant value
   :rtype: riemann.TN

.. function:: riemann.linalg.inv(A)

   Compute the inverse of a square matrix.

   :param A: Input square matrix
   :type A: riemann.TN
   :return: Inverse matrix
   :rtype: riemann.TN

.. function:: riemann.linalg.skew(A)

   Compute the skew-symmetric part of a matrix.

   :param A: Input matrix
   :type A: riemann.TN
   :return: Skew-symmetric matrix
   :rtype: riemann.TN

.. function:: riemann.linalg.svd(A, full_matrices=True)

   Compute the singular value decomposition (SVD) of a matrix.

   :param A: Input matrix
   :type A: riemann.TN
   :param full_matrices: Whether to return full U and Vh matrices
   :type full_matrices: bool, optional
   :return: Tuple of (U, S, Vh)
   :rtype: tuple

.. function:: riemann.linalg.pinv(A, rcond=1e-15)

   Compute the Moore-Penrose pseudoinverse of a matrix.

   :param A: Input matrix
   :type A: riemann.TN
   :param rcond: Singular value threshold
   :type rcond: float, optional
   :return: Pseudoinverse matrix
   :rtype: riemann.TN

Eigenvalue Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: riemann.linalg.eig(A)

   Compute the eigenvalues and eigenvectors of a square matrix.

   :param A: Input square matrix
   :type A: riemann.TN
   :return: Tuple of (eigenvalues, eigenvectors)
   :rtype: tuple

.. function:: riemann.linalg.eigh(A, UPLO='L')

   Compute the eigenvalues and eigenvectors of a Hermitian (or real symmetric) matrix.

   :param A: Input Hermitian matrix
   :type A: riemann.TN
   :param UPLO: Specifies whether to use upper ('U') or lower ('L') triangular part
   :type UPLO: str, optional
   :return: Tuple of (eigenvalues, eigenvectors)
   :rtype: tuple

Linear Equation Solving
~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: riemann.linalg.lstsq(A, b, rcond=None)

   Compute the least-squares solution.

   :param A: Coefficient matrix
   :type A: riemann.TN
   :param b: Right-hand side vector or matrix
   :type b: riemann.TN
   :param rcond: Singular value threshold
   :type rcond: float, optional
   :return: Least-squares solution
   :rtype: riemann.TN

.. function:: riemann.linalg.lu(A, pivot=True)

   Compute the LU decomposition of a matrix.

   :param A: Input matrix
   :type A: riemann.TN
   :param pivot: Whether to perform pivoting
   :type pivot: bool, optional
   :return: Tuple of (P, L, U)
   :rtype: tuple

.. function:: riemann.linalg.solve(A, b)

   Solve the linear equation system Ax = b.

   :param A: Coefficient matrix
   :type A: riemann.TN
   :param b: Right-hand side vector or matrix
   :type b: riemann.TN
   :return: Solution vector or matrix
   :rtype: riemann.TN

.. function:: riemann.linalg.qr(A, mode='reduced')

   Compute the QR decomposition of a matrix.

   :param A: Input matrix
   :type A: riemann.TN
   :param mode: Decomposition mode, 'reduced' or 'complete'
   :type mode: str, optional
   :return: Tuple of (Q, R)
   :rtype: tuple

.. function:: riemann.linalg.cholesky(A, upper=False)

   Compute the Cholesky decomposition of a positive-definite matrix.

   :param A: Input positive-definite matrix
   :type A: riemann.TN
   :param upper: Whether to return upper triangular matrix, default is False (lower)
   :type upper: bool, optional
   :return: Cholesky factor
   :rtype: riemann.TN

Neural Network Modules
----------------------

Base Classes
~~~~~~~~~~~~

.. class:: riemann.nn.Module()

   Base class for all neural network modules.

   .. method:: __init__()

      Initialize the module.

   .. method:: forward(*args, **kwargs)

      Define the forward pass of the module.

      :param args: Input arguments
      :param kwargs: Keyword arguments
      :return: Output of the forward pass

   .. method:: parameters()

      Get all trainable parameters.

      :return: List of parameters
      :rtype: list of riemann.TN

.. class:: riemann.nn.Parameter(data=None, requires_grad=True)

   Trainable parameter class for storing model parameters.

   :param data: Parameter data
   :type data: array_like, optional
   :param requires_grad: Whether to track gradients
   :type requires_grad: bool, optional

Container Modules
~~~~~~~~~~~~~~~~~~

.. class:: riemann.nn.Sequential(*modules)

   Container that applies a sequence of modules in order.

   :param modules: List of modules
   :type modules: list of riemann.Module

.. class:: riemann.nn.ModuleList(modules=None)

   Container class for storing a list of modules.

   This container allows storing multiple modules in list form and provides convenient access and iteration methods. All submodules are properly registered to appear in the parameter list.

   :param modules: List of modules for initialization
   :type modules: list of riemann.Module, optional

.. class:: riemann.nn.ModuleDict(modules=None)

   Container class for storing a dictionary of modules.

   This container allows storing modules using string keys and provides dictionary-like access methods. All submodules are properly registered.

   :param modules: Dictionary of modules for initialization
   :type modules: dict of {str: riemann.Module}, optional

.. class:: riemann.nn.ParameterList(parameters=None)

   Container class for storing a list of parameters.

   This container allows storing multiple parameters in list form. All parameters are properly registered to appear in the parameter list.

   :param parameters: List of parameters for initialization
   :type parameters: list of riemann.Parameter, optional

.. class:: riemann.nn.ParameterDict(parameters=None)

   Container class for storing a dictionary of parameters.

   This container allows storing parameters using string keys and provides dictionary-like access methods. All parameters are properly registered.

   :param parameters: Dictionary of parameters for initialization
   :type parameters: dict of {str: riemann.Parameter}, optional

Linear Layers
~~~~~~~~~~~~~

.. class:: riemann.nn.Linear(in_features, out_features, bias=True)

   Fully connected linear layer.

   :param in_features: Number of input features
   :type in_features: int
   :param out_features: Number of output features
   :type out_features: int
   :param bias: Whether to include bias term
   :type bias: bool, optional

Convolutional Layers
~~~~~~~~~~~~~~~~~~~~

.. class:: riemann.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

   1D convolutional layer.

   Applies convolution operations to 1D inputs, extracting features and generating new feature maps.

   :param in_channels: Number of input channels
   :type in_channels: int
   :param out_channels: Number of output channels
   :type out_channels: int
   :param kernel_size: Kernel size
   :type kernel_size: int or tuple
   :param stride: Stride
   :type stride: int or tuple, optional
   :param padding: Zero-padding added to all sides of the input
   :type padding: int or tuple, optional
   :param dilation: Spacing between kernel elements
   :type dilation: int or tuple, optional
   :param groups: Grouping between input and output channels
   :type groups: int, optional
   :param bias: Whether to use bias term
   :type bias: bool, optional
   :param padding_mode: Padding mode
   :type padding_mode: str, optional

.. class:: riemann.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

   2D convolutional layer.

   Applies convolution operations to 2D inputs, extracting image features and generating new feature maps.

   :param in_channels: Number of input channels
   :type in_channels: int
   :param out_channels: Number of output channels
   :type out_channels: int
   :param kernel_size: Kernel size
   :type kernel_size: int or tuple
   :param stride: Stride
   :type stride: int or tuple, optional
   :param padding: Zero-padding added to all sides of the input
   :type padding: int or tuple, optional
   :param dilation: Spacing between kernel elements
   :type dilation: int or tuple, optional
   :param groups: Grouping between input and output channels
   :type groups: int, optional
   :param bias: Whether to use bias term
   :type bias: bool, optional
   :param padding_mode: Padding mode
   :type padding_mode: str, optional

.. class:: riemann.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

   3D convolutional layer.

   Applies convolution operations to 3D inputs, commonly used for feature extraction in video, volumetric data, etc.

   :param in_channels: Number of input channels
   :type in_channels: int
   :param out_channels: Number of output channels
   :type out_channels: int
   :param kernel_size: Kernel size
   :type kernel_size: int or tuple
   :param stride: Stride
   :type stride: int or tuple, optional
   :param padding: Zero-padding added to all sides of the input
   :type padding: int or tuple, optional
   :param dilation: Spacing between kernel elements
   :type dilation: int or tuple, optional
   :param padding_mode: Padding mode
   :type padding_mode: str, optional

Pooling Layers
~~~~~~~~~~~~~~~

.. class:: riemann.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

   1D max pooling layer.

   Applies max pooling to 1D inputs, used for extracting key features from sequence data and reducing data dimensions.

   :param kernel_size: Pooling window size
   :type kernel_size: int or tuple
   :param stride: Stride of the pooling window
   :type stride: int or tuple, optional
   :param padding: Zero-padding added to all sides of the input
   :type padding: int or tuple, optional
   :param dilation: Spacing between pooling window elements
   :type dilation: int or tuple, optional
   :param return_indices: Whether to return indices of maximum values
   :type return_indices: bool, optional
   :param ceil_mode: Whether to use ceiling instead of floor to compute output shape
   :type ceil_mode: bool, optional

.. class:: riemann.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

   2D max pooling layer.

   Applies max pooling to 2D inputs, commonly used for feature extraction and dimension reduction in image data.

   :param kernel_size: Pooling window size
   :type kernel_size: int or tuple
   :param stride: Stride of the pooling window
   :type stride: int or tuple, optional
   :param padding: Zero-padding added to all sides of the input
   :type padding: int or tuple, optional
   :param dilation: Spacing between pooling window elements
   :type dilation: int or tuple, optional
   :param return_indices: Whether to return indices of maximum values
   :type return_indices: bool, optional
   :param ceil_mode: Whether to use ceiling instead of floor to compute output shape
   :type ceil_mode: bool, optional

.. class:: riemann.nn.MaxPool3d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

   3D max pooling layer.

   Applies max pooling to 3D inputs, commonly used for feature extraction in video, volumetric data, etc.

   :param kernel_size: Pooling window size
   :type kernel_size: int or tuple
   :param stride: Stride of the pooling window
   :type stride: int or tuple, optional
   :param padding: Zero-padding added to all sides of the input
   :type padding: int or tuple, optional
   :param dilation: Spacing between pooling window elements
   :type dilation: int or tuple, optional
   :param return_indices: Whether to return indices of maximum values
   :type return_indices: bool, optional
   :param ceil_mode: Whether to use ceiling instead of floor to compute output shape
   :type ceil_mode: bool, optional

.. class:: riemann.nn.AvgPool1d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)

   1D average pooling layer.

   Applies average pooling to 1D inputs, used for smoothing sequence data features and reducing data dimensions.

   :param kernel_size: Pooling window size
   :type kernel_size: int or tuple
   :param stride: Stride of the pooling window
   :type stride: int or tuple, optional
   :param padding: Zero-padding added to all sides of the input
   :type padding: int or tuple, optional
   :param ceil_mode: Whether to use ceiling instead of floor to compute output shape
   :type ceil_mode: bool, optional
   :param count_include_pad: Whether to include zero-padding when calculating average
   :type count_include_pad: bool, optional
   :param divisor_override: If specified, will be used as denominator
   :type divisor_override: int, optional

.. class:: riemann.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)

   2D average pooling layer.

   Applies average pooling to 2D inputs, commonly used for feature smoothing and dimension reduction in image data.

   :param kernel_size: Pooling window size
   :type kernel_size: int or tuple
   :param stride: Stride of the pooling window
   :type stride: int or tuple, optional
   :param padding: Zero-padding added to all sides of the input
   :type padding: int or tuple, optional
   :param ceil_mode: Whether to use ceiling instead of floor to compute output shape
   :type ceil_mode: bool, optional
   :param count_include_pad: Whether to include zero-padding when calculating average
   :type count_include_pad: bool, optional
   :param divisor_override: If specified, will be used as denominator
   :type divisor_override: int, optional

.. class:: riemann.nn.AvgPool3d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)

   3D average pooling layer.

   Applies average pooling to 3D inputs, commonly used for feature smoothing and dimension reduction in video, volumetric data, etc.

   :param kernel_size: Pooling window size
   :type kernel_size: int or tuple
   :param stride: Stride of the pooling window
   :type stride: int or tuple, optional
   :param padding: Zero-padding added to all sides of the input
   :type padding: int or tuple, optional
   :param ceil_mode: Whether to use ceiling instead of floor to compute output shape
   :type ceil_mode: bool, optional
   :param count_include_pad: Whether to include zero-padding when calculating average
   :type count_include_pad: bool, optional
   :param divisor_override: If specified, will be used as denominator
   :type divisor_override: int, optional

Adaptive Pooling Layers
~~~~~~~~~~~~~~~~~~~~~~~

.. class:: riemann.nn.AdaptiveAvgPool1d(output_size)

   1D adaptive average pooling layer.

   Automatically computes pooling kernel size and stride based on the specified output size, ensuring the output dimensions are always fixed.

   :param output_size: Output sequence length, can be an integer or None (indicating maintaining original size)
   :type output_size: int or tuple

.. class:: riemann.nn.AdaptiveAvgPool2d(output_size)

   2D adaptive average pooling layer.

   Automatically computes pooling kernel size and stride based on the specified output size, commonly used to convert feature maps of arbitrary sizes to fixed dimensions.

   :param output_size: Output size, can be an integer or a tuple (H, W), or None (indicating maintaining original size)
   :type output_size: int or tuple

.. class:: riemann.nn.AdaptiveAvgPool3d(output_size)

   3D adaptive average pooling layer.

   Automatically computes pooling kernel size and stride based on the specified output size, commonly used for feature extraction in volumetric data.

   :param output_size: Output size, can be an integer or a tuple (D, H, W), or None (indicating maintaining original size)
   :type output_size: int or tuple

.. class:: riemann.nn.AdaptiveMaxPool1d(output_size, return_indices=False)

   1D adaptive max pooling layer.

   Automatically computes pooling kernel size and stride based on the specified output size, ensuring the output dimensions are always fixed.

   :param output_size: Output sequence length, can be an integer or None (indicating maintaining original size)
   :type output_size: int or tuple
   :param return_indices: Whether to return indices of maximum values
   :type return_indices: bool, optional

.. class:: riemann.nn.AdaptiveMaxPool2d(output_size, return_indices=False)

   2D adaptive max pooling layer.

   Automatically computes pooling kernel size and stride based on the specified output size, commonly used to convert feature maps of arbitrary sizes to fixed dimensions.

   :param output_size: Output size, can be an integer or a tuple (H, W), or None (indicating maintaining original size)
   :type output_size: int or tuple
   :param return_indices: Whether to return indices of maximum values
   :type return_indices: bool, optional

.. class:: riemann.nn.AdaptiveMaxPool3d(output_size, return_indices=False)

   3D adaptive max pooling layer.

   Automatically computes pooling kernel size and stride based on the specified output size, commonly used for feature extraction in volumetric data.

   :param output_size: Output size, can be an integer or a tuple (D, H, W), or None (indicating maintaining original size)
   :type output_size: int or tuple
   :param return_indices: Whether to return indices of maximum values
   :type return_indices: bool, optional

Normalization Layers
~~~~~~~~~~~~~~~~~~~~

.. class:: riemann.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

   1D batch normalization layer.

   Normalizes the channel dimension of 2D or 3D input tensors to have zero mean and unit variance, improving training convergence and model generalization.

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

.. class:: riemann.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1)

   2D batch normalization layer.

   :param num_features: Number of features
   :type num_features: int
   :param eps: Small value to avoid division by zero
   :type eps: float, optional
   :param momentum: Momentum for running statistics
   :type momentum: float, optional

.. class:: riemann.nn.BatchNorm3d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

   3D batch normalization layer.

   Normalizes the channel dimension of 5D input tensors (N, C, D, H, W) to have zero mean and unit variance, improving training convergence and model generalization.

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

.. class:: riemann.nn.LayerNorm(normalized_shape, eps=1e-05, affine=True, device=None, dtype=None)

   Layer normalization layer, normalizes specified dimensions.

   Compatible with torch.nn.LayerNorm, normalizes specified dimensions of input tensors to have zero mean and unit variance.

   :param normalized_shape: Integer or tuple specifying dimensions to normalize
   :type normalized_shape: int or tuple
   :param eps: Small value added to variance to avoid division by zero
   :type eps: float, optional
   :param affine: Whether to include learnable affine parameters (gamma and beta)
   :type affine: bool, optional
   :param device: Device for parameters and buffers
   :type device: optional
   :param dtype: Data type for parameters and buffers
   :type dtype: optional

.. class:: riemann.nn.Flatten(start_dim=1, end_dim=-1)

   Layer that flattens tensor dimensions, removing all dimensions from start_dim to end_dim.

   Typically used after convolutional layers and before fully connected layers to flatten multi-dimensional convolution results into 1D vectors.

   :param start_dim: Dimension to start flattening from
   :type start_dim: int, optional
   :param end_dim: Dimension to end flattening at
   :type end_dim: int, optional

Activation Function Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: riemann.nn.ReLU(inplace=False)

   ReLU activation function

   Applies the rectified linear unit function element-wise: ReLU(x) = max(0, x)

   :param inplace: Whether to perform operation in-place
   :type inplace: bool, optional

.. class:: riemann.nn.LeakyReLU(negative_slope=0.01, inplace=False)

   Leaky ReLU activation function

   Applies the leaky rectified linear unit function element-wise: LeakyReLU(x) = max(x, negative_slope * x)

   :param negative_slope: Slope of the negative region
   :type negative_slope: float, optional
   :param inplace: Whether to perform operation in-place
   :type inplace: bool, optional

.. class:: riemann.nn.RReLU(lower=0.125, upper=0.3333333333333333, inplace=False)

   Randomized Leaky ReLU activation function

   Applies the randomized leaky rectified linear unit function element-wise

   :param lower: Lower bound of uniform distribution
   :type lower: float, optional
   :param upper: Upper bound of uniform distribution
   :type upper: float, optional
   :param inplace: Whether to perform operation in-place
   :type inplace: bool, optional

.. class:: riemann.nn.PReLU(num_parameters=1, init=0.25)

   Parametric ReLU activation function

   Applies the parametric rectified linear unit function element-wise, where a is a learnable parameter

   :param num_parameters: Number of learnable parameters
   :type num_parameters: int, optional
   :param init: Initial value of parameters
   :type init: float, optional

.. class:: riemann.nn.Sigmoid()

   Sigmoid activation function

   Applies the sigmoid function element-wise, mapping values to the [0, 1] range

.. class:: riemann.nn.Tanh()

   Tanh activation function

   Applies the hyperbolic tangent function element-wise, mapping values to the [-1, 1] range

.. class:: riemann.nn.Softmax(dim=None)

   Softmax activation function

   Applies the softmax function along the specified dimension

   :param dim: Dimension to apply softmax
   :type dim: int, optional

.. class:: riemann.nn.LogSoftmax(dim=None)

   Log-Softmax activation function

   Applies the log-softmax function along the specified dimension

   :param dim: Dimension to apply log-softmax
   :type dim: int, optional

.. class:: riemann.nn.GELU()

   Gaussian Error Linear Unit activation function

   Applies the Gaussian Error Linear Unit function element-wise: GELU(x) = x * Φ(x), where Φ is the cumulative distribution function of the standard normal distribution

.. class:: riemann.nn.Softplus(beta=1, threshold=20)

   Softplus activation function

   Applies the Softplus activation function element-wise: Softplus(x) = (1 / beta) * log(1 + exp(beta * x))

   :param beta: Slope of the linear part
   :type beta: float, optional
   :param threshold: Threshold for numerical stability
   :type threshold: float, optional

Dropout Layers
~~~~~~~~~~~~~~~

.. class:: riemann.nn.Dropout(p=0.5)

   Dropout layer for preventing overfitting.

   :param p: Dropout probability
   :type p: float, optional

.. class:: riemann.nn.Dropout2d(p=0.5, inplace=False)

   2D dropout layer for preventing overfitting.

   During training, randomly zeroes entire channels of the input tensor with probability p, and scales remaining channels by 1/(1-p).
   During evaluation, no operation is performed.

   :param p: Dropout probability
   :type p: float, optional
   :param inplace: Whether to perform operation in-place
   :type inplace: bool, optional

.. class:: riemann.nn.Dropout3d(p=0.5, inplace=False)

   3D dropout layer for preventing overfitting.

   During training, randomly zeroes entire channels of the input tensor with probability p, and scales remaining channels by 1/(1-p).
   During evaluation, no operation is performed.

   :param p: Dropout probability
   :type p: float, optional
   :param inplace: Whether to perform operation in-place
   :type inplace: bool, optional

Embedding Layer
~~~~~~~~~~~~~~~~

.. class:: riemann.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, dtype=None, device=None)

   Embedding layer that converts integer indices to dense vectors.
   
   The embedding layer is a fundamental component in neural networks for handling categorical features and sequence data.
   
   :param num_embeddings: Number of embedding vectors, i.e., vocabulary size
   :type num_embeddings: int
   :param embedding_dim: Dimension of each embedding vector
   :type embedding_dim: int
   :param padding_idx: If specified, embedding vectors at this index do not participate in gradient computation and remain unchanged during training
   :type padding_idx: int, optional
   :param max_norm: If specified, all embedding vectors with norm exceeding max_norm will be renormalized to max_norm
   :type max_norm: float, optional
   :param norm_type: p-value for norm calculation, defaults to 2 (L2 norm)
   :type norm_type: float, optional
   :param scale_grad_by_freq: If True, gradients will be scaled by frequency of each word in mini-batch
   :type scale_grad_by_freq: bool, optional
   :param sparse: If True, gradient of weight will be a sparse tensor
   :type sparse: bool, optional
   :param dtype: Data type for embedding weights
   :type dtype: np.dtype, optional
   :param device: Device for embedding weights
   :type device: str|int|Device, optional

Loss Function Modules
~~~~~~~~~~~~~~~~~~~~~

.. class:: riemann.nn.L1Loss(size_average=None, reduce=None, reduction='mean')

   Mean absolute error loss, computes absolute error between input and target values.

   :param size_average: Deprecated
   :type size_average: bool, optional
   :param reduce: Deprecated
   :type reduce: bool, optional
   :param reduction: Specifies the reduction to apply to the output
   :type reduction: str, optional

.. class:: riemann.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

   Mean squared error loss, computes squared error between input and target values.

   :param size_average: Deprecated
   :type size_average: bool, optional
   :param reduce: Deprecated
   :type reduce: bool, optional
   :param reduction: Specifies the reduction to apply to the output
   :type reduction: str, optional

.. class:: riemann.nn.NLLLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

   Negative log likelihood loss, used for probability prediction in classification tasks.

   :param weight: Manual scaling weight for each class
   :type weight: riemann.TN, optional
   :param size_average: Deprecated
   :type size_average: bool, optional
   :param ignore_index: Specifies target value to ignore
   :type ignore_index: int, optional
   :param reduce: Deprecated
   :type reduce: bool, optional
   :param reduction: Specifies the reduction to apply to the output
   :type reduction: str, optional

.. class:: riemann.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

   Cross entropy loss, combines LogSoftmax and NLLLoss in one class, commonly used for multi-class classification tasks.

   :param weight: Manual scaling weight for each class
   :type weight: riemann.TN, optional
   :param size_average: Deprecated
   :type size_average: bool, optional
   :param ignore_index: Specifies target value to ignore
   :type ignore_index: int, optional
   :param reduce: Deprecated
   :type reduce: bool, optional
   :param reduction: Specifies the reduction to apply to the output
   :type reduction: str, optional

.. class:: riemann.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')

   Binary cross entropy loss, computes binary classification error between target and output.

   :param weight: Manual scaling weight for each batch element
   :type weight: riemann.TN, optional
   :param size_average: Deprecated
   :type size_average: bool, optional
   :param reduce: Deprecated
   :type reduce: bool, optional
   :param reduction: Specifies the reduction to apply to the output
   :type reduction: str, optional

.. class:: riemann.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean')

   Binary cross entropy loss with logits, computes binary cross entropy directly on input logits.

   :param weight: Manual scaling weight for each batch element
   :type weight: riemann.TN, optional
   :param size_average: Deprecated
   :type size_average: bool, optional
   :param reduce: Deprecated
   :type reduce: bool, optional
   :param reduction: Specifies the reduction to apply to the output
   :type reduction: str, optional

.. class:: riemann.nn.HuberLoss(delta=1.0, size_average=None, reduce=None, reduction='mean')

   Huber loss function, uses squared error when error is less than delta, otherwise uses linear error.

   :param delta: Threshold at which the loss function changes from quadratic to linear
   :type delta: float, optional
   :param size_average: Deprecated
   :type size_average: bool, optional
   :param reduce: Deprecated
   :type reduce: bool, optional
   :param reduction: Specifies the reduction to apply to the output
   :type reduction: str, optional

.. class:: riemann.nn.SmoothL1Loss(beta=1.0, size_average=None, reduce=None, reduction='mean')

   Smooth L1 loss, combines advantages of L1 and L2 losses, uses quadratic loss for small errors and linear loss for large errors.

   :param beta: Threshold controlling transition from quadratic to linear loss
   :type beta: float, optional
   :param size_average: Deprecated
   :type size_average: bool, optional
   :param reduce: Deprecated
   :type reduce: bool, optional
   :param reduction: Specifies the reduction to apply to the output
   :type reduction: str, optional

Transformer Modules
~~~~~~~~~~~~~~~~~~~

.. class:: riemann.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None)

   Multi-head attention mechanism, allows the model to attend to information from different representation subspaces.

   :param embed_dim: Dimension of input and output vectors, must be divisible by num_heads
   :type embed_dim: int
   :param num_heads: Number of attention heads
   :type num_heads: int
   :param dropout: Dropout probability for attention weights
   :type dropout: float, optional
   :param bias: Whether to add bias to projection layers
   :type bias: bool, optional
   :param add_bias_kv: Whether to add learnable bias to key and value sequences
   :type add_bias_kv: bool, optional
   :param add_zero_attn: Whether to add a column of zeros to attention weights
   :type add_zero_attn: bool, optional
   :param kdim: Dimension of key vectors, defaults to embed_dim
   :type kdim: int, optional
   :param vdim: Dimension of value vectors, defaults to embed_dim
   :type vdim: int, optional
   :param batch_first: Whether input/output shape is (batch, seq, feature) instead of (seq, batch, feature)
   :type batch_first: bool, optional
   :param device: Device for tensors
   :type device: optional
   :param dtype: Data type for tensors
   :type dtype: optional

.. class:: riemann.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-05, batch_first=False, norm_first=False, bias=True, device=None, dtype=None)

   Single layer of Transformer encoder, consisting of self-attention mechanism and feed-forward network.

   :param d_model: Dimension of input and output features
   :type d_model: int
   :param nhead: Number of attention heads
   :type nhead: int
   :param dim_feedforward: Dimension of feed-forward hidden layer
   :type dim_feedforward: int, optional
   :param dropout: Dropout probability
   :type dropout: float, optional
   :param activation: Activation function type, 'relu' or 'gelu'
   :type activation: str, optional
   :param layer_norm_eps: Epsilon value for layer normalization
   :type layer_norm_eps: float, optional
   :param batch_first: Whether input/output shape is (batch, seq, feature)
   :type batch_first: bool, optional
   :param norm_first: Whether to use Pre-LN mode
   :type norm_first: bool, optional
   :param bias: Whether to add bias to linear layers
   :type bias: bool, optional
   :param device: Device for tensors
   :type device: optional
   :param dtype: Data type for tensors
   :type dtype: optional

.. class:: riemann.nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-05, batch_first=False, norm_first=False, bias=True, device=None, dtype=None)

   Single layer of Transformer decoder, consisting of self-attention, cross-attention, and feed-forward network.

   :param d_model: Dimension of input and output features
   :type d_model: int
   :param nhead: Number of attention heads
   :type nhead: int
   :param dim_feedforward: Dimension of feed-forward hidden layer
   :type dim_feedforward: int, optional
   :param dropout: Dropout probability
   :type dropout: float, optional
   :param activation: Activation function type, 'relu' or 'gelu'
   :type activation: str, optional
   :param layer_norm_eps: Epsilon value for layer normalization
   :type layer_norm_eps: float, optional
   :param batch_first: Whether input/output shape is (batch, seq, feature)
   :type batch_first: bool, optional
   :param norm_first: Whether to use Pre-LN mode
   :type norm_first: bool, optional
   :param bias: Whether to add bias to linear layers
   :type bias: bool, optional
   :param device: Device for tensors
   :type device: optional
   :param dtype: Data type for tensors
   :type dtype: optional

.. class:: riemann.nn.TransformerEncoder(encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True)

   Transformer encoder consisting of N stacked TransformerEncoderLayer layers.

   :param encoder_layer: Single encoder layer instance to be cloned
   :type encoder_layer: TransformerEncoderLayer
   :param num_layers: Number of encoder layers
   :type num_layers: int
   :param norm: Final layer normalization, optional
   :type norm: Module, optional
   :param enable_nested_tensor: Whether to enable nested tensor optimization (interface compatibility only)
   :type enable_nested_tensor: bool, optional
   :param mask_check: Whether to perform mask checking (interface compatibility only)
   :type mask_check: bool, optional

.. class:: riemann.nn.TransformerDecoder(decoder_layer, num_layers, norm=None)

   Transformer decoder consisting of N stacked TransformerDecoderLayer layers.

   :param decoder_layer: Single decoder layer instance to be cloned
   :type decoder_layer: TransformerDecoderLayer
   :param num_layers: Number of decoder layers
   :type num_layers: int
   :param norm: Final layer normalization, optional
   :type norm: Module, optional

.. class:: riemann.nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', custom_encoder=None, custom_decoder=None, layer_norm_eps=1e-05, batch_first=False, norm_first=False, bias=True, device=None, dtype=None)

   Complete Transformer architecture containing both encoder and decoder.

   :param d_model: Dimension of encoder/decoder inputs
   :type d_model: int, optional
   :param nhead: Number of attention heads
   :type nhead: int, optional
   :param num_encoder_layers: Number of encoder layers
   :type num_encoder_layers: int, optional
   :param num_decoder_layers: Number of decoder layers
   :type num_decoder_layers: int, optional
   :param dim_feedforward: Dimension of feed-forward network
   :type dim_feedforward: int, optional
   :param dropout: Dropout value
   :type dropout: float, optional
   :param activation: Activation function, 'relu' or 'gelu'
   :type activation: str, optional
   :param custom_encoder: Custom encoder module
   :type custom_encoder: Module, optional
   :param custom_decoder: Custom decoder module
   :type custom_decoder: Module, optional
   :param layer_norm_eps: Epsilon value for layer normalization
   :type layer_norm_eps: float, optional
   :param batch_first: Whether input/output shape is (batch, seq, feature)
   :type batch_first: bool, optional
   :param norm_first: Whether to perform LayerNorm before attention and feed-forward
   :type norm_first: bool, optional
   :param bias: Whether linear and LayerNorm layers learn additive bias
   :type bias: bool, optional
   :param device: Device for tensors
   :type device: optional
   :param dtype: Data type for tensors
   :type dtype: optional

Functional Interface
~~~~~~~~~~~~~~~~~~~~

The ``riemann.nn.functional`` module provides functional implementations of various neural network operations.

Linear Functions
````````````````
.. function:: riemann.nn.functional.linear(input, weight, bias=None)

   Applies linear transformation: y = xA^T + b

   :param input: Input tensor with shape ``(*, in_features)``
   :type input: riemann.TN
   :param weight: Weight tensor with shape ``(out_features, in_features)``
   :type weight: riemann.TN
   :param bias: Bias tensor with shape ``(out_features)``. Default: None
   :type bias: riemann.TN, optional
   :return: Output tensor with shape ``(*, out_features)``
   :rtype: riemann.TN

Activation Functions
````````````````````
.. function:: riemann.nn.functional.sigmoid(input)

   Applies element-wise sigmoid function: sigmoid(x) = 1 / (1 + exp(-x))

   :param input: Input tensor
   :type input: riemann.TN
   :return: Output tensor
   :rtype: riemann.TN

.. function:: riemann.nn.functional.silu(input)

   Applies Sigmoid Linear Unit (SiLU) activation function: silu(x) = x * sigmoid(x)

   :param input: Input tensor
   :type input: riemann.TN
   :return: Output tensor
   :rtype: riemann.TN

.. function:: riemann.nn.functional.tanh(input)

   Applies hyperbolic tangent activation function: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

   :param input: Input tensor
   :type input: riemann.TN

Dropout Functions
`````````````````

.. function:: riemann.nn.functional.dropout(input, p=0.5, training=True, inplace=False)

   During training, randomly zeroes elements of the input tensor with probability p, and scales remaining elements by 1/(1-p).
   During evaluation, no operation is performed.

   :param input: Input tensor
   :type input: riemann.TN
   :param p: Dropout probability
   :type p: float, optional
   :param training: Whether in training mode
   :type training: bool, optional
   :param inplace: Whether to perform operation in-place
   :type inplace: bool, optional
   :return: Tensor after applying dropout
   :rtype: riemann.TN

.. function:: riemann.nn.functional.dropout2d(input, p=0.5, training=True, inplace=False)

   During training, randomly zeroes entire channels of the input tensor with probability p, and scales remaining channels by 1/(1-p).
   During evaluation, no operation is performed.

   :param input: Input tensor with shape (N, C, H, W)
   :type input: riemann.TN
   :param p: Dropout probability
   :type p: float, optional
   :param training: Whether in training mode
   :type training: bool, optional
   :param inplace: Whether to perform operation in-place
   :type inplace: bool, optional
   :return: Tensor after applying dropout
   :rtype: riemann.TN

.. function:: riemann.nn.functional.dropout3d(input, p=0.5, training=True, inplace=False)

   During training, randomly zeroes entire channels of the input tensor with probability p, and scales remaining channels by 1/(1-p).
   During evaluation, no operation is performed.

   :param input: Input tensor with shape (N, C, D, H, W)
   :type input: riemann.TN
   :param p: Dropout probability
   :type p: float, optional
   :param training: Whether in training mode
   :type training: bool, optional
   :param inplace: Whether to perform operation in-place
   :type inplace: bool, optional
   :return: Tensor after applying dropout
   :rtype: riemann.TN

Normalization Functions
```````````````````````
.. function:: riemann.nn.functional.batch_norm(input, running_mean=None, running_var=None, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5)

Applies batch normalization to the input tensor.

:param input: Input tensor with shape (N, C), (N, C, L), (N, C, H, W) or (N, C, D, H, W)

:type input: riemann.TN

:param running_mean: Running mean with shape (C,)

:type running_mean: riemann.TN, optional

:param running_var: Running variance with shape (C,)

:type running_var: riemann.TN, optional

:param weight: Learnable scaling parameter γ with shape (C,)

:type weight: riemann.TN, optional

:param bias: Learnable offset parameter β with shape (C,)

:type bias: riemann.TN, optional

:param training: Whether in training mode

:type training: bool, optional

:param momentum: Momentum for running statistics

:type momentum: float, optional

:param eps: Small constant for numerical stability

:type eps: float, optional

:return: Normalized tensor with same shape as input

:rtype: riemann.TN

.. function:: riemann.nn.functional.layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05)

Applies layer normalization to specified dimensions of the input tensor.

:param input: Input tensor

:type input: riemann.TN

:param normalized_shape: Integer or tuple specifying dimensions to normalize

:type normalized_shape: int or tuple

:param weight: Optional weight tensor (γ) for affine transformation

:type weight: riemann.TN, optional

:param bias: Optional bias tensor (β) for affine transformation

:type bias: riemann.TN, optional

:param eps: Small value added to variance to avoid division by zero

:type eps: float, optional

:return: Normalized tensor with same shape as input

:rtype: riemann.TN

Embedding Functions
```````````````````
.. function:: riemann.nn.functional.embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)

   Looks up embedding vectors for input indices from an embedding matrix.

   :param input: Tensor containing indices with arbitrary shape
   :type input: riemann.TN
   :param weight: Embedding matrix with shape (num_embeddings, embedding_dim)
   :type weight: riemann.TN
   :param padding_idx: If specified, embedding vectors at this index do not participate in gradient computation and remain unchanged during training
   :type padding_idx: int, optional
   :param max_norm: If specified, all embedding vectors with norm exceeding max_norm will be renormalized to max_norm
   :type max_norm: float, optional
   :param norm_type: p-value for norm calculation, defaults to 2 (L2 norm)
   :type norm_type: float, optional
   :param scale_grad_by_freq: If True, gradients will be scaled by frequency of each word in mini-batch
   :type scale_grad_by_freq: bool, optional
   :param sparse: If True, gradient of weight will be a sparse tensor
   :type sparse: bool, optional
   :return: Output tensor with shape (``*``, embedding_dim), where ``*`` is the shape of input
   :rtype: riemann.TN

.. function:: riemann.nn.functional.softmax(input, dim)

   Applies softmax function along the specified dimension

   :param input: Input tensor
   :type input: riemann.TN
   :param dim: Dimension to compute softmax
   :type dim: int
   :return: Output tensor
   :rtype: riemann.TN

.. function:: riemann.nn.functional.log_softmax(input, dim=-1)

   Applies log softmax function for numerical stability

   :param input: Input tensor
   :type input: riemann.TN
   :param dim: Dimension to compute log_softmax
   :type dim: int, optional
   :return: Output tensor
   :rtype: riemann.TN

.. function:: riemann.nn.functional.relu(input)

   Applies rectified linear unit activation function: relu(x) = max(0, x)

   :param input: Input tensor
   :type input: riemann.TN
   :return: Output tensor
   :rtype: riemann.TN

.. function:: riemann.nn.functional.leaky_relu(input, alpha=0.01)

   Applies leaky rectified linear unit activation function

   :param input: Input tensor
   :type input: riemann.TN
   :param alpha: Slope of the negative region
   :type alpha: float, optional
   :return: Output tensor
   :rtype: riemann.TN

.. function:: riemann.nn.functional.prelu(input, alpha)

   Applies parametric rectified linear unit activation function

   :param input: Input tensor
   :type input: riemann.TN
   :param alpha: Learnable parameter tensor
   :type alpha: riemann.TN
   :return: Output tensor
   :rtype: riemann.TN

.. function:: riemann.nn.functional.rrelu(input, lower=1.0/8.0, upper=1.0/3.0, training=True)

   Applies randomized rectified linear unit activation function

   :param input: Input tensor
   :type input: riemann.TN
   :param lower: Lower bound of uniform distribution
   :type lower: float, optional
   :param upper: Upper bound of uniform distribution
   :type upper: float, optional
   :param training: Whether to use randomized alpha (training) or fixed alpha (evaluation)
   :type training: bool, optional
   :return: Output tensor
   :rtype: riemann.TN

.. function:: riemann.nn.functional.gelu(input)

   Applies Gaussian Error Linear Unit activation function

   :param input: Input tensor
   :type input: riemann.TN
   :return: Output tensor
   :rtype: riemann.TN

.. function:: riemann.nn.functional.softplus(input, beta=1.0, threshold=20.0)

   Applies Softplus activation function: softplus(x) = (1 / beta) * log(1 + exp(beta * x))

   :param input: Input tensor
   :type input: riemann.TN
   :param beta: Slope of the linear part
   :type beta: float, optional
   :param threshold: Threshold for numerical stability
   :type threshold: float, optional
   :return: Output tensor
   :rtype: riemann.TN

Loss Functions
``````````````

.. function:: riemann.nn.functional.l1_loss(input, target, size_average=None, reduce=None, reduction='mean')

Compute L1 (absolute error) loss

:param input: Input tensor

:type input: riemann.TN

:param target: Target tensor

:type target: riemann.TN

:param size_average: Deprecated

:type size_average: bool, optional

:param reduce: Deprecated

:type reduce: bool, optional

:param reduction: Specifies the reduction to apply to the output

:type reduction: str, optional

:return: Loss value

:rtype: riemann.TN

.. function:: riemann.nn.functional.smooth_l1_loss(input, target, size_average=None, reduce=None, reduction='mean', beta=1.0)

Compute smooth L1 loss

:param input: Input tensor

:type input: riemann.TN

:param target: Target tensor

:type target: riemann.TN

:param size_average: Deprecated

:type size_average: bool, optional

:param reduce: Deprecated

:type reduce: bool, optional

:param reduction: Specifies the reduction to apply to the output

:type reduction: str, optional

:param beta: Threshold at which the loss function changes from quadratic to linear

:type beta: float, optional

:return: Loss value

:rtype: riemann.TN

.. function:: riemann.nn.functional.cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)

Compute cross entropy loss

:param input: Input tensor

:type input: riemann.TN

:param target: Target tensor

:type target: riemann.TN

:param weight: Manual scaling weight for each class

:type weight: riemann.TN, optional

:param size_average: Deprecated

:type size_average: bool, optional

:param ignore_index: Specifies target value to ignore

:type ignore_index: int, optional

:param reduce: Deprecated

:type reduce: bool, optional

:param reduction: Specifies the reduction to apply to the output

:type reduction: str, optional

:param label_smoothing: Amount of label smoothing

:type label_smoothing: float, optional

:return: Loss value

:rtype: riemann.TN

.. function:: riemann.nn.functional.binary_cross_entropy_with_logits(input, target, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)

Compute binary cross entropy loss with logits

:param input: Input tensor

:type input: riemann.TN

:param target: Target tensor

:type target: riemann.TN

:param weight: Manual scaling weight for each batch element

:type weight: riemann.TN, optional

:param size_average: Deprecated

:type size_average: bool, optional

:param reduce: Deprecated

:type reduce: bool, optional

:param reduction: Specifies the reduction to apply to the output

:type reduction: str, optional

:param pos_weight: Weight of positive class

:type pos_weight: riemann.TN, optional

:return: Loss value

:rtype: riemann.TN

.. function:: riemann.nn.functional.huber_loss(input, target, delta=1.0, size_average=None, reduce=None, reduction='mean')

Compute Huber loss

:param input: Input tensor

:type input: riemann.TN

:param target: Target tensor

:type target: riemann.TN

:param delta: Threshold at which the loss function changes from quadratic to linear

:type delta: float, optional

:param size_average: Deprecated

:type size_average: bool, optional

:param reduce: Deprecated

:type reduce: bool, optional

:param reduction: Specifies the reduction to apply to the output

:type reduction: str, optional

:return: Loss value

:rtype: riemann.TN

.. function:: riemann.nn.functional.nll_loss(input, target, weight=None, ignore_index=-100, reduction='mean')

Compute negative log likelihood loss

:param input: Input tensor

:type input: riemann.TN

:param target: Target tensor

:type target: riemann.TN

:param weight: Manual scaling weight for each class

:type weight: riemann.TN, optional

:param ignore_index: Specifies target value to ignore

:type ignore_index: int, optional

:param reduction: Specifies the reduction to apply to the output

:type reduction: str, optional

:return: Loss value

:rtype: riemann.TN

Convolution Functions
`````````````````````

.. function:: riemann.nn.functional.conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)

   Apply 1D convolution to input signals

   :param input: Input tensor with shape (N, C_in, L_in)
   :type input: riemann.TN
   :param weight: Weight tensor with shape (C_out, C_in/groups, K)
   :type weight: riemann.TN
   :param bias: Bias tensor with shape (C_out). Default: None
   :type bias: riemann.TN, optional
   :param stride: Convolution stride
   :type stride: int or tuple, optional
   :param padding: Zero-padding added to both sides of the input
   :type padding: int or tuple, optional
   :param dilation: Spacing between kernel elements
   :type dilation: int or tuple, optional
   :param groups: Number of blocked connections from input channels to output channels
   :type groups: int, optional
   :return: Output tensor with shape (N, C_out, L_out)
   :rtype: riemann.TN

.. function:: riemann.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)

   Apply 2D convolution to input images

   :param input: Input tensor with shape (N, C_in, H_in, W_in)
   :type input: riemann.TN
   :param weight: Weight tensor with shape (C_out, C_in/groups, K_h, K_w)
   :type weight: riemann.TN
   :param bias: Bias tensor with shape (C_out). Default: None
   :type bias: riemann.TN, optional
   :param stride: Convolution stride
   :type stride: int or tuple, optional
   :param padding: Zero-padding added to both sides of the input
   :type padding: int or tuple, optional
   :param dilation: Spacing between kernel elements
   :type dilation: int or tuple, optional
   :param groups: Number of blocked connections from input channels to output channels
   :type groups: int, optional
   :return: Output tensor with shape (N, C_out, H_out, W_out)
   :rtype: riemann.TN

.. function:: riemann.nn.functional.conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)

   Apply 3D convolution to input volumes

   :param input: Input tensor with shape (N, C_in, D_in, H_in, W_in)
   :type input: riemann.TN
   :param weight: Weight tensor with shape (C_out, C_in/groups, K_d, K_h, K_w)
   :type weight: riemann.TN
   :param bias: Bias tensor with shape (C_out). Default: None
   :type bias: riemann.TN, optional
   :param stride: Convolution stride
   :type stride: int or tuple, optional
   :param padding: Zero-padding added to all sides of the input
   :type padding: int or tuple, optional
   :param dilation: Spacing between kernel elements
   :type dilation: int or tuple, optional
   :param groups: Number of blocked connections from input channels to output channels
   :type groups: int, optional
   :return: Output tensor with shape (N, C_out, D_out, H_out, W_out)
   :rtype: riemann.TN

Pooling Functions
`````````````````

.. function:: riemann.nn.functional.max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

   Apply 1D max pooling to input signals

   :param input: Input tensor with shape (N, C, L_in)
   :type input: riemann.TN
   :param kernel_size: Pooling window size
   :type kernel_size: int or tuple
   :param stride: Stride of the pooling window. Default: kernel_size
   :type stride: int or tuple, optional
   :param padding: Zero-padding added to both sides of the input
   :type padding: int or tuple, optional
   :param dilation: Spacing between pooling window elements
   :type dilation: int or tuple, optional
   :param ceil_mode: Whether to use ceiling instead of floor to compute output shape
   :type ceil_mode: bool, optional
   :param return_indices: Whether to return indices of maximum values
   :type return_indices: bool, optional
   :return: Output tensor with shape (N, C, L_out), or tuple (TN, TN) if return_indices is True
   :rtype: riemann.TN or tuple

.. function:: riemann.nn.functional.max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

   Apply 2D max pooling to input images

   :param input: Input tensor with shape (N, C, H_in, W_in)
   :type input: riemann.TN
   :param kernel_size: Pooling window size
   :type kernel_size: int or tuple
   :param stride: Stride of the pooling window. Default: kernel_size
   :type stride: int or tuple, optional
   :param padding: Zero-padding added to both sides of the input
   :type padding: int or tuple, optional
   :param dilation: Spacing between pooling window elements
   :type dilation: int or tuple, optional
   :param ceil_mode: Whether to use ceiling instead of floor to compute output shape
   :type ceil_mode: bool, optional
   :param return_indices: Whether to return indices of maximum values
   :type return_indices: bool, optional
   :return: Output tensor with shape (N, C, H_out, W_out), or tuple (TN, TN) if return_indices is True
   :rtype: riemann.TN or tuple

.. function:: riemann.nn.functional.max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

   Apply 3D max pooling to input volume data

   :param input: Input tensor with shape (N, C, D_in, H_in, W_in)
   :type input: riemann.TN
   :param kernel_size: Pooling window size
   :type kernel_size: int or tuple
   :param stride: Stride of the pooling window. Default: kernel_size
   :type stride: int or tuple, optional
   :param padding: Zero-padding added to all sides of the input
   :type padding: int or tuple, optional
   :param dilation: Spacing between pooling window elements
   :type dilation: int or tuple, optional
   :param ceil_mode: Whether to use ceiling instead of floor to compute output shape
   :type ceil_mode: bool, optional
   :param return_indices: Whether to return indices of maximum values
   :type return_indices: bool, optional
   :return: Output tensor with shape (N, C, D_out, H_out, W_out), or tuple (TN, TN) if return_indices is True
   :rtype: riemann.TN or tuple

.. function:: riemann.nn.functional.avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)

   Apply 1D average pooling to input signals

   :param input: Input tensor with shape (N, C, L_in)
   :type input: riemann.TN
   :param kernel_size: Pooling window size
   :type kernel_size: int or tuple
   :param stride: Stride of the pooling window. Default: kernel_size
   :type stride: int or tuple, optional
   :param padding: Zero-padding added to both sides of the input
   :type padding: int or tuple, optional
   :param ceil_mode: Whether to use ceiling instead of floor to compute output shape
   :type ceil_mode: bool, optional
   :param count_include_pad: Whether to include zero-padding when calculating average
   :type count_include_pad: bool, optional
   :param divisor_override: If specified, will be used as denominator
   :type divisor_override: int, optional
   :return: Output tensor with shape (N, C, L_out)
   :rtype: riemann.TN

.. function:: riemann.nn.functional.avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)

   Apply 2D average pooling to input images

   :param input: Input tensor with shape (N, C, H_in, W_in)
   :type input: riemann.TN
   :param kernel_size: Pooling window size
   :type kernel_size: int or tuple
   :param stride: Stride of the pooling window. Default: kernel_size
   :type stride: int or tuple, optional
   :param padding: Zero-padding added to both sides of the input
   :type padding: int or tuple, optional
   :param ceil_mode: Whether to use ceiling instead of floor to compute output shape
   :type ceil_mode: bool, optional
   :param count_include_pad: Whether to include zero-padding when calculating average
   :type count_include_pad: bool, optional
   :param divisor_override: If specified, will be used as denominator
   :type divisor_override: int, optional
   :return: Output tensor with shape (N, C, H_out, W_out)
   :rtype: riemann.TN

.. function:: riemann.nn.functional.avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)

   Apply 3D average pooling to input volume data

   :param input: Input tensor with shape (N, C, D_in, H_in, W_in)
   :type input: riemann.TN
   :param kernel_size: Pooling window size
   :type kernel_size: int or tuple
   :param stride: Stride of the pooling window. Default: kernel_size
   :type stride: int or tuple, optional
   :param padding: Zero-padding added to all sides of the input
   :type padding: int or tuple, optional
   :param ceil_mode: Whether to use ceiling instead of floor to compute output shape
   :type ceil_mode: bool, optional
   :param count_include_pad: Whether to include zero-padding when calculating average
   :type count_include_pad: bool, optional
   :param divisor_override: If specified, will be used as denominator
   :type divisor_override: int, optional
   :return: Output tensor with shape (N, C, D_out, H_out, W_out)
   :rtype: riemann.TN

Utility Functions
`````````````````
.. function:: riemann.nn.functional.one_hot(target, num_classes)

   Convert category indices to one-hot encoded tensors

   :param target: Target tensor with shape ``(N, *)``
   :type target: riemann.TN
   :param num_classes: Number of classes
   :type num_classes: int
   :return: One-hot encoded tensor with shape ``(N, *, num_classes)``
   :rtype: riemann.TN

.. function:: riemann.nn.functional.unfold(input, kernel_size, dilation=1, padding=0, stride=1)

   Extract sliding local blocks from batched input tensor

   :param input: Input tensor with shape (N, C, H, W)
   :type input: riemann.TN
   :param kernel_size: Sliding block size
   :type kernel_size: int or tuple
   :param dilation: Spacing between kernel elements
   :type dilation: int or tuple, optional
   :param padding: Zero-padding added to both sides of the input
   :type padding: int or tuple, optional
   :param stride: Stride of the sliding block
   :type stride: int or tuple, optional
   :return: Unfolded tensor with shape (N, C * kernel_size[0] * kernel_size[1], L)
   :rtype: riemann.TN

.. function:: riemann.nn.functional.fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1)

Fold the unfolded tensor back to its original shape

:param input: Input tensor with shape (N, C * kernel_size[0] * kernel_size[1], L)

:type input: riemann.TN

:param output_size: Output tensor size (H, W)

:type output_size: int or tuple

:param kernel_size: Sliding block size

:type kernel_size: int or tuple

:param dilation: Spacing between kernel elements

:type dilation: int or tuple, optional

:param padding: Zero-padding added to both sides of the input

:type padding: int or tuple, optional

:param stride: Stride of the sliding block
:type stride: int or tuple, optional
:return: Folded tensor with shape (N, C, H, W)
:rtype: riemann.TN

.. function:: riemann.nn.functional.unfold2d(input, kernel_size, dilation=1, padding=0, stride=1)

   Extract sliding local blocks from 2D input tensor (2D-specific version of unfold)

   :param input: Input tensor with shape (N, C, H, W)
   :type input: riemann.TN
   :param kernel_size: Sliding block size
   :type kernel_size: int or tuple
   :param dilation: Spacing between kernel elements
   :type dilation: int or tuple, optional
   :param padding: Zero-padding added to both sides of the input
   :type padding: int or tuple, optional
   :param stride: Stride of the sliding block
   :type stride: int or tuple, optional
   :return: Unfolded tensor with shape (N, C * kernel_size[0] * kernel_size[1], L)
   :rtype: riemann.TN

.. function:: riemann.nn.functional.unfold3d(input, kernel_size, dilation=1, padding=0, stride=1)

   Extract sliding local blocks from 3D input tensor (3D-specific version of unfold)

   :param input: Input tensor with shape (N, C, D, H, W)
   :type input: riemann.TN
   :param kernel_size: Sliding block size
   :type kernel_size: int or tuple
   :param dilation: Spacing between kernel elements
   :type dilation: int or tuple, optional
   :param padding: Zero-padding added to all sides of the input
   :type padding: int or tuple, optional
   :param stride: Stride of the sliding block
   :type stride: int or tuple, optional
   :return: Unfolded tensor with shape (N, C * kernel_size[0] * kernel_size[1] * kernel_size[2], L)
   :rtype: riemann.TN

Datasets
--------

Dataset Classes
~~~~~~~~~~~~~~~

.. class:: riemann.utils.Dataset

   Abstract base class for datasets, defining the standard interface that all datasets must implement.

   .. method:: __len__()

      Return the number of samples in the dataset.

   .. method:: __getitem__(index)

      Get a single sample from the dataset at the given index.

.. class:: riemann.utils.TensorDataset(*tensors)

   Simple tensor dataset implementation that uses the first dimension of multiple tensors as the dataset dimension.

   :param \*tensors: Variable number of tensors, all tensors must have the same size in the first dimension
   :type \*tensors: riemann.TN

   .. method:: __len__()

      Return the size of the dataset, which is the size of the first dimension of the tensors.

   .. method:: __getitem__(index)

      Get sample data at the specified index.

Data Loaders
~~~~~~~~~~~~

.. class:: riemann.utils.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=None, drop_last=False)

   Efficient data loader supporting batch processing, data shuffling, and multi-process loading.

   :param dataset: Dataset to load data from
   :type dataset: riemann.utils.Dataset
   :param batch_size: Size of each batch, defaults to 1
   :type batch_size: int, optional
   :param shuffle: Whether to shuffle the data at the beginning of each epoch, defaults to False
   :type shuffle: bool, optional
   :param num_workers: Number of worker processes for data loading, 0 means loading in the main process, defaults to 0
   :type num_workers: int, optional
   :param collate_fn: Batch processing function for combining samples into batches, defaults to default_collate
   :type collate_fn: callable, optional
   :param drop_last: Whether to drop the last incomplete batch if dataset size is not divisible by batch size, defaults to False
   :type drop_last: bool, optional

   .. method:: __len__()

      Return the number of batches in the data loader.

   .. method:: __iter__()

      Return an iterator for the data loader.

Dataset Utility Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: riemann.utils.default_collate(batch)

Default batch processing function that converts a batch of sample data into tensor format suitable for model input.

:param batch: List of samples in a batch, each sample can be various data types

:type batch: list

:return: Batch data combined according to input type

.. function:: riemann.utils.clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False)

Clip gradients by norm.

:param parameters: Collection of parameters whose gradients need to be clipped

:type parameters: Iterable[riemann.TN]

:param max_norm: Maximum norm of gradients

:type max_norm: float or int

:param norm_type: Type of norm, defaults to 2 (L2 norm)

:type norm_type: float or int, optional

:param error_if_nonfinite: Whether to throw an error if gradients contain non-finite values (such as NaN or inf), defaults to False

:type error_if_nonfinite: bool, optional

:return: Gradient norm before clipping

:rtype: float

.. function:: riemann.utils.clip_grad_value_(parameters, clip_value, error_if_nonfinite=False)

Clip gradients by value.

:param parameters: Collection of parameters whose gradients need to be clipped

:type parameters: Iterable[riemann.TN]

:param clip_value: Threshold value for gradient clipping

:type clip_value: float or int

:param error_if_nonfinite: Whether to throw an error if gradients contain non-finite values (such as NaN or inf), defaults to False

:type error_if_nonfinite: bool, optional

Vision
------

.. module:: riemann.vision

Datasets
~~~~~~~~

.. class:: riemann.vision.datasets.MNIST(root, train=True, transform=None, target_transform=None)

   MNIST dataset class for loading and processing the MNIST handwritten digit dataset.

   :param root: Root directory of the dataset
   :type root: str
   :param train: Whether to load the training set, defaults to True
   :type train: bool
   :param transform: Transformation function applied to images
   :type transform: callable, optional
   :param target_transform: Transformation function applied to targets
   :type target_transform: callable, optional

   .. method:: __len__()

      Return the number of samples in the dataset.

   .. method:: __getitem__(index)

      Get a single sample from the dataset at the given index.

.. class:: riemann.vision.datasets.EasyMNIST(root, train=True, onehot_label=True, download=False)

   Subclass inherited from MNIST, applies normalization, standardization, and flattening transformations to image data during initialization, and performs one-hot encoding or conversion to scalar tensors for labels.

   :param root: Root directory of the dataset
   :type root: str
   :param train: Whether to load the training set, defaults to True
   :type train: bool
   :param onehot_label: Whether to use one-hot encoded labels, defaults to True
   :type onehot_label: bool
   :param download: Whether to download the dataset if not found, defaults to False
   :type download: bool, optional

   .. method:: __len__()

      Return the size of the dataset.

   .. method:: __getitem__(index)

      Get sample data at the specified index.

.. class:: riemann.vision.datasets.FashionMNIST(root, train=True, transform=None, target_transform=None, download=False)

   Fashion-MNIST dataset class for loading and processing the Fashion-MNIST fashion product dataset.

   :param root: Root directory of the dataset
   :type root: str
   :param train: Whether to load the training set, defaults to True
   :type train: bool
   :param transform: Transformation function applied to images
   :type transform: callable, optional
   :param target_transform: Transformation function applied to targets
   :type target_transform: callable, optional
   :param download: Whether to download the dataset if not found, defaults to False
   :type download: bool, optional

   .. attribute:: classes

      List of class names: ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

   .. method:: __len__()

      Return the number of samples in the dataset.

   .. method:: __getitem__(index)

      Get a single sample from the dataset at the given index.

.. class:: riemann.vision.datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=False)

   CIFAR-10 dataset class for loading and processing the CIFAR-10 image dataset.

   :param root: Root directory of the dataset
   :type root: str
   :param train: Whether to load the training set, defaults to True
   :type train: bool
   :param transform: Transformation function applied to images
   :type transform: callable, optional
   :param target_transform: Transformation function applied to targets
   :type target_transform: callable, optional
   :param download: Whether to download the dataset if not found, defaults to False
   :type download: bool, optional

   .. method:: __len__()

      Return the size of the dataset.

   .. method:: __getitem__(index)

      Get sample data at the specified index.

.. class:: riemann.vision.datasets.Flowers102(root, split='train', transform=None, target_transform=None, download=False)

   Oxford 102 Flower dataset class for loading and processing the flower classification dataset.

   :param root: Root directory of the dataset
   :type root: str
   :param split: Dataset split ('train', 'val', or 'test'), defaults to 'train'
   :type split: str, optional
   :param transform: Transformation function applied to images
   :type transform: callable, optional
   :param target_transform: Transformation function applied to targets
   :type target_transform: callable, optional
   :param download: Whether to download the dataset if not found, defaults to False
   :type download: bool, optional

   .. method:: __len__()

      Return the size of the dataset.

   .. method:: __getitem__(index)

      Get sample data at the specified index.

.. class:: riemann.vision.datasets.OxfordIIITPet(root, split='trainval', target_types='category', transform=None, target_transform=None, download=False)

   Oxford-IIIT Pet dataset class for loading and processing the pet classification dataset.

   :param root: Root directory of the dataset
   :type root: str
   :param split: Dataset split ('trainval' or 'test'), defaults to 'trainval'
   :type split: str, optional
   :param target_types: Type of target ('category', 'binary-category', or 'segmentation'), defaults to 'category'
   :type target_types: str or list, optional
   :param transform: Transformation function applied to images
   :type transform: callable, optional
   :param target_transform: Transformation function applied to targets
   :type target_transform: callable, optional
   :param download: Whether to download the dataset if not found, defaults to False
   :type download: bool, optional

   .. method:: __len__()

      Return the size of the dataset.

   .. method:: __getitem__(index)

      Get sample data at the specified index.

.. class:: riemann.vision.datasets.LFWPeople(root, split='10fold', image_set='funneled', transform=None, target_transform=None, download=False)

   LFW People dataset class for loading and processing the face recognition dataset.

   :param root: Root directory of the dataset
   :type root: str
   :param split: Dataset split ('10fold', 'train', or 'test'), defaults to '10fold'
   :type split: str, optional
   :param image_set: Image alignment type ('original', 'funneled', or 'deepfunneled'), defaults to 'funneled'
   :type image_set: str, optional
   :param transform: Transformation function applied to images
   :type transform: callable, optional
   :param target_transform: Transformation function applied to targets
   :type target_transform: callable, optional
   :param download: Whether to download the dataset if not found, defaults to False
   :type download: bool, optional

   .. attribute:: classes

      List of person names.

   .. method:: __len__()

      Return the size of the dataset.

   .. method:: __getitem__(index)

      Get sample data at the specified index.

.. class:: riemann.vision.datasets.SVHN(root, split='train', transform=None, target_transform=None, download=False)

   SVHN (Street View House Numbers) dataset class for loading and processing the digit recognition dataset.

   :param root: Root directory of the dataset
   :type root: str
   :param split: Dataset split ('train', 'test', or 'extra'), defaults to 'train'
   :type split: str, optional
   :param transform: Transformation function applied to images
   :type transform: callable, optional
   :param target_transform: Transformation function applied to targets
   :type target_transform: callable, optional
   :param download: Whether to download the dataset if not found, defaults to False
   :type download: bool, optional

   .. method:: __len__()

      Return the size of the dataset.

   .. method:: __getitem__(index)

      Get sample data at the specified index.

.. class:: riemann.vision.datasets.DatasetFolder(root, loader, extensions=None, transform=None, target_transform=None, is_valid_file=None, allow_empty=False)

   Generic folder dataset class for loading custom datasets from folders.

   :param root: Root directory path of the dataset
   :type root: str
   :param loader: Image loading function
   :type loader: callable
   :param extensions: Tuple of allowed file extensions
   :type extensions: tuple, optional
   :param transform: Transformation function applied to images
   :type transform: callable, optional
   :param target_transform: Transformation function applied to targets
   :type target_transform: callable, optional
   :param is_valid_file: Function to validate if a file is valid
   :type is_valid_file: callable, optional
   :param allow_empty: Whether to allow empty folders, defaults to False
   :type allow_empty: bool

   .. attribute:: classes

      List of class names.

   .. attribute:: class_to_idx

      Dictionary mapping class names to indices.

   .. method:: __len__()

      Return the number of samples in the dataset.

   .. method:: __getitem__(index)

      Get a single sample from the dataset at the given index.

.. class:: riemann.vision.datasets.ImageFolder(root, transform=None, target_transform=None, loader=None, is_valid_file=None)

   Image folder dataset class, inherited from DatasetFolder, for loading image datasets from folders.

   :param root: Root directory path of the dataset
   :type root: str
   :param transform: Transformation function applied to images
   :type transform: callable, optional
   :param target_transform: Transformation function applied to targets
   :type target_transform: callable, optional
   :param loader: Image loading function, defaults to PIL Image loader
   :type loader: callable, optional
   :param is_valid_file: Function to validate if a file is valid
   :type is_valid_file: callable, optional

   .. attribute:: classes

      List of class names.

   .. attribute:: class_to_idx

      Dictionary mapping class names to indices.

   .. method:: __len__()

      Return the number of samples in the dataset.

   .. method:: __getitem__(index)

      Get a single sample from the dataset at the given index.

Image Transforms
~~~~~~~~~~~~~~~~

.. module:: riemann.vision.transforms

.. class:: riemann.vision.transforms.Transform

   Base class for all transformation classes.

   .. method:: __call__(img)

      Execute the transformation.

.. class:: riemann.vision.transforms.Compose(transforms)

   Combine multiple transformations into a single transformation.

   :param transforms: List of transformations to combine
   :type transforms: list of Transform objects

.. class:: riemann.vision.transforms.ToTensor

   Convert PIL image or NumPy array to TN tensor.

.. class:: riemann.vision.transforms.ToPILImage

   Convert TN tensor or NumPy array to PIL image.

.. class:: riemann.vision.transforms.Normalize(mean, std, inplace=False)

   Normalize tensor using mean and standard deviation.

   :param mean: Mean for each channel
   :type mean: sequence
   :param std: Standard deviation for each channel
   :type std: sequence
   :param inplace: Whether to perform operation in-place, defaults to False
   :type inplace: bool, optional

.. class:: riemann.vision.transforms.Resize(size, interpolation=BILINEAR)

   Resize PIL image.

   :param size: Target size. If int, the smaller edge will be resized to this size while maintaining aspect ratio. If (h, w), resize directly to this size.
   :type size: int or tuple
   :param interpolation: Interpolation method, defaults to BILINEAR
   :type interpolation: int, optional

.. class:: riemann.vision.transforms.CenterCrop(size)

   Center crop.

   :param size: Crop size. If int, crop to square (size, size). If (h, w), crop to this size.
   :type size: int or tuple

.. class:: riemann.vision.transforms.RandomHorizontalFlip(p=0.5)

   Random horizontal flip.

   :param p: Flip probability, defaults to 0.5
   :type p: float, optional

.. class:: riemann.vision.transforms.RandomVerticalFlip(p=0.5)

   Random vertical flip.

   :param p: Flip probability, defaults to 0.5
   :type p: float, optional

.. class:: riemann.vision.transforms.RandomRotation(degrees, resample=NEAREST, expand=False, center=None)

   Random rotation.

   :param degrees: Rotation angle range. If int, select from (-degrees, degrees). If (min, max), select from (min, max).
   :type degrees: int or tuple
   :param resample: Resampling method, defaults to NEAREST
   :type resample: int, optional
   :param expand: Whether to expand image to accommodate rotation, defaults to False
   :type expand: bool, optional
   :param center: Rotation center, defaults to image center
   :type center: tuple, optional

.. class:: riemann.vision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)

   Random color transformation.

   :param brightness: Brightness adjustment factor
   :type brightness: float or tuple
   :param contrast: Contrast adjustment factor
   :type contrast: float or tuple
   :param saturation: Saturation adjustment factor
   :type saturation: float or tuple
   :param hue: Hue adjustment factor
   :type hue: float or tuple

.. class:: riemann.vision.transforms.Grayscale(num_output_channels=1)

   Convert image to grayscale.

   :param num_output_channels: Number of output channels, 1 or 3, defaults to 1
   :type num_output_channels: int

.. class:: riemann.vision.transforms.RandomGrayscale(p=0.1)

   Randomly convert to grayscale.

   :param p: Probability of converting to grayscale, defaults to 0.1
   :type p: float, optional

.. class:: riemann.vision.transforms.RandomCrop(size, padding=None)

   Crop image at random position.

   :param size: Crop size. If int, crop to square (size, size). If (h, w), crop to this size.
   :type size: int or tuple
   :param padding: Padding size, defaults to None
   :type padding: int or tuple, optional

.. class:: riemann.vision.transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(3/4, 4/3), interpolation=BILINEAR)

   Random crop and resize.

   :param size: Target size. If int, resize to square (size, size). If (h, w), resize to this size.
   :type size: int or tuple
   :param scale: Crop area ratio range relative to original image, defaults to (0.08, 1.0)
   :type scale: tuple, optional
   :param ratio: Crop aspect ratio range, defaults to (3/4, 4/3)
   :type ratio: tuple, optional
   :param interpolation: Interpolation method, defaults to BILINEAR
   :type interpolation: int, optional

.. class:: riemann.vision.transforms.FiveCrop(size)

   Five crop.

   :param size: Crop size. If int, crop to square (size, size). If (h, w), crop to this size.
   :type size: int or tuple

.. class:: riemann.vision.transforms.TenCrop(size, vertical_flip=False)

   Ten crop.

   :param size: Crop size. If int, crop to square (size, size). If (h, w), crop to this size.
   :type size: int or tuple
   :param vertical_flip: Whether to include vertical flip version, defaults to False
   :type vertical_flip: bool, optional

.. class:: riemann.vision.transforms.Pad(padding, fill=0, padding_mode='constant')

   Padding.

   :param padding: Padding size. If int, pad all sides equally. If (pad_l, pad_r, pad_t, pad_b), specify left, right, top, bottom padding respectively. If (pad_h, pad_w), specify height and width padding respectively.
   :type padding: int or tuple
   :param fill: Fill value, defaults to 0
   :type fill: int or tuple
   :param padding_mode: Padding mode, defaults to 'constant'
   :type padding_mode: str, optional

.. class:: riemann.vision.transforms.Lambda(lambd)

   Use user-defined lambda function as transformation.

   :param lambd: Lambda function
   :type lambd: function

.. class:: riemann.vision.transforms.PILToTensor

   Convert PIL Image to tensor (without scaling).

.. class:: riemann.vision.transforms.ConvertImageDtype(dtype)

   Convert image data type.

   :param dtype: Target data type
   :type dtype: torch.dtype

.. class:: riemann.vision.transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))

   Apply Gaussian blur to image.

   :param kernel_size: Size of Gaussian kernel
   :type kernel_size: int or tuple
   :param sigma: Standard deviation range of Gaussian kernel
   :type sigma: tuple, optional

.. class:: riemann.vision.transforms.RandomAffine(degrees, translate=None, scale=None, shear=None, resample=NEAREST, fillcolor=0)

   Random affine transformation.

   :param degrees: Rotation angle range
   :type degrees: float or tuple
   :param translate: Translation range
   :type translate: tuple, optional
   :param scale: Scale range
   :type scale: tuple, optional
   :param shear: Shear angle range
   :type shear: float or tuple, optional
   :param resample: Resampling mode
   :type resample: int, optional
   :param fillcolor: Fill color
   :type fillcolor: int, optional

.. class:: riemann.vision.transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=BILINEAR, fill=0)

   Random perspective transformation.

   :param distortion_scale: Distortion degree
   :type distortion_scale: float
   :param p: Probability of applying transformation
   :type p: float
   :param interpolation: Interpolation mode
   :type interpolation: int
   :param fill: Fill value
   :type fill: int or tuple

.. class:: riemann.vision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)

   Random erasing for data augmentation.

   :param p: Probability of applying erasing
   :type p: float
   :param scale: Erasing area range
   :type scale: tuple
   :param ratio: Erasing aspect ratio range
   :type ratio: tuple
   :param value: Erasing fill value
   :type value: int or float or tuple
   :param inplace: Whether to operate in-place
   :type inplace: bool

.. class:: riemann.vision.transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET)

   Automatic data augmentation based on learning policy.

   :param policy: Augmentation policy
   :type policy: AutoAugmentPolicy

.. class:: riemann.vision.transforms.RandAugment(num_ops=2, magnitude=9, num_magnitude_bins=31, interpolation=BILINEAR, fill=None)

   Random data augmentation.

   :param num_ops: Number of operations
   :type num_ops: int
   :param magnitude: Augmentation magnitude
   :type magnitude: int
   :param num_magnitude_bins: Number of magnitude bins
   :type num_magnitude_bins: int
   :param interpolation: Interpolation mode
   :type interpolation: int
   :param fill: Fill value
   :type fill: int or tuple or None

.. class:: riemann.vision.transforms.TrivialAugmentWide(num_magnitude_bins=31, interpolation=BILINEAR, fill=None)

   Wide range simple augmentation.

   :param num_magnitude_bins: Number of magnitude bins
   :type num_magnitude_bins: int
   :param interpolation: Interpolation mode
   :type interpolation: int
   :param fill: Fill value
   :type fill: int or tuple or None

.. class:: riemann.vision.transforms.SanitizeBoundingBox(labels_format='xyxy', min_size=1)

   Sanitize bounding boxes.

   :param labels_format: Bounding box format
   :type labels_format: str
   :param min_size: Minimum size
   :type min_size: int

.. class:: riemann.vision.transforms.Invert

   Invert colors.

.. class:: riemann.vision.transforms.Posterize(bits)

   Reduce color bits.

   :param bits: Number of bits to keep
   :type bits: int

.. class:: riemann.vision.transforms.Solarize(threshold)

   Invert pixels above threshold.

   :param threshold: Threshold value
   :type threshold: int

.. class:: riemann.vision.transforms.Equalize

   Histogram equalization.

.. class:: riemann.vision.transforms.AutoContrast

   Auto contrast adjustment.

.. class:: riemann.vision.transforms.Sharpness(sharpness_factor)

   Sharpness adjustment.

   :param sharpness_factor: Sharpness factor
   :type sharpness_factor: float

.. class:: riemann.vision.transforms.Brightness(brightness_factor)

   Brightness adjustment.

   :param brightness_factor: Brightness factor
   :type brightness_factor: float

.. class:: riemann.vision.transforms.Contrast(contrast_factor)

   Contrast adjustment.

   :param contrast_factor: Contrast factor
   :type contrast_factor: float

.. class:: riemann.vision.transforms.Saturation(saturation_factor)

   Saturation adjustment.

   :param saturation_factor: Saturation factor
   :type saturation_factor: float

.. class:: riemann.vision.transforms.Hue(hue_factor)

   Hue adjustment.

   :param hue_factor: Hue factor
   :type hue_factor: float

Optimization
------------

Optimizers
~~~~~~~~~~

.. class:: riemann.optim.Optimizer(params, defaults)

   Base class for all optimizers.

   :param params: Iterator of parameters to optimize or list of dictionaries defining parameter groups
   :type params: Iterable[riemann.TN or riemann.nn.Parameter] or List[Dict[str, Any]]
   :param defaults: Default hyperparameters for the optimizer
   :type defaults: Dict[str, Any]

   .. method:: step(closure=None)

      Perform a single optimization step

      :param closure: A closure that reevaluates the model and returns the loss
      :type closure: callable, optional
      :return: Loss value if closure is provided, otherwise None
      :rtype: float or None

   .. method:: zero_grad(set_to_none=False)

      Set the gradients of all optimized parameters to zero

      :param set_to_none: Whether to set gradients to None instead of zero
      :type set_to_none: bool, optional

   .. method:: add_param_group(param_group)

      Add a parameter group to the optimizer

      :param param_group: Parameter group to add
      :type param_group: Dict[str, Any]

   .. method:: state_dict()

      Return the optimizer's state dictionary

      :return: Optimizer state
      :rtype: Dict[str, Any]

   .. method:: load_state_dict(state_dict)

      Load the optimizer state

      :param state_dict: State dictionary to load
      :type state_dict: Dict[str, Any]

.. class:: riemann.optim.GD(params, lr=0.01, weight_decay=0.0)

   Gradient Descent optimizer

   :param params: Iterator of parameters to optimize or list of dictionaries defining parameter groups
   :type params: Iterable[riemann.TN or riemann.nn.Parameter] or List[Dict[str, Any]]
   :param lr: Learning rate
   :type lr: float, optional
   :param weight_decay: Weight decay (L2 regularization) coefficient
   :type weight_decay: float, optional

   .. method:: step()

      Perform a single optimization step

.. class:: riemann.optim.SGD(params, lr=0.01, momentum=0.0, weight_decay=0.0, dampening=0.0, nesterov=False)

   Stochastic Gradient Descent optimizer

   :param params: Iterator of parameters to optimize or list of dictionaries defining parameter groups
   :type params: Iterable[riemann.TN or riemann.nn.Parameter] or List[Dict[str, Any]]
   :param lr: Learning rate
   :type lr: float, optional
   :param momentum: Momentum factor
   :type momentum: float, optional
   :param weight_decay: Weight decay (L2 regularization) coefficient
   :type weight_decay: float, optional
   :param dampening: Dampening for momentum
   :type dampening: float, optional
   :param nesterov: Whether to enable Nesterov momentum
   :type nesterov: bool, optional

   .. method:: step(closure=None)

      Perform a single optimization step

      :param closure: A closure that reevaluates the model and returns the loss
      :type closure: callable, optional
      :return: Loss value if closure is provided, otherwise None
      :rtype: float or None

.. class:: riemann.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

   Adam (Adaptive Moment Estimation) optimizer

   :param params: Iterator of parameters to optimize or list of dictionaries defining parameter groups
   :type params: Iterable[riemann.TN or riemann.nn.Parameter] or List[Dict[str, Any]]
   :param lr: Learning rate
   :type lr: float, optional
   :param betas: Coefficients used for computing running averages of gradient and its square
   :type betas: Tuple[float, float], optional
   :param eps: Term added to the denominator to improve numerical stability
   :type eps: float, optional
   :param weight_decay: Weight decay (L2 regularization) coefficient
   :type weight_decay: float, optional
   :param amsgrad: Whether to use the AMSGrad variant
   :type amsgrad: bool, optional

   .. method:: step()

      Perform a single optimization step

.. class:: riemann.optim.Adagrad(params, lr=0.01, lr_decay=0.0, weight_decay=0.0, initial_accumulator_value=0.0, eps=1e-10)

   Adagrad (Adaptive Gradient Algorithm) optimizer

   :param params: Iterator of parameters to optimize or list of dictionaries defining parameter groups
   :type params: Iterable[riemann.TN or riemann.nn.Parameter] or List[Dict[str, Any]]
   :param lr: Learning rate
   :type lr: float, optional
   :param lr_decay: Learning rate decay
   :type lr_decay: float, optional
   :param weight_decay: Weight decay (L2 regularization) coefficient
   :type weight_decay: float, optional
   :param initial_accumulator_value: Initial value for the accumulator
   :type initial_accumulator_value: float, optional
   :param eps: Term added to the denominator to improve numerical stability
   :type eps: float, optional

   .. method:: step()

      Perform a single optimization step

.. class:: riemann.optim.LBFGS(params, lr=1.0, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)

   L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) optimizer

   :param params: Iterator of parameters to optimize or list of dictionaries defining parameter groups
   :type params: Iterable[riemann.TN or riemann.nn.Parameter] or List[Dict[str, Any]]
   :param lr: Learning rate
   :type lr: float, optional
   :param max_iter: Maximum number of iterations per optimization step
   :type max_iter: int, optional
   :param max_eval: Maximum number of function evaluations per optimization step
   :type max_eval: int, optional
   :param tolerance_grad: Gradient tolerance for convergence
   :type tolerance_grad: float, optional
   :param tolerance_change: Parameter change tolerance for convergence
   :type tolerance_change: float, optional
   :param history_size: Update history size
   :type history_size: int, optional
   :param line_search_fn: Line search function
   :type line_search_fn: callable, optional

   .. method:: step(closure)

      Perform a single optimization step

      :param closure: A closure that reevaluates the model and returns the loss
      :type closure: callable
      :return: Loss value
      :rtype: float

.. class:: riemann.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False)

   AdamW (Adam with Weight Decay) optimizer

   An improved version of Adam that treats weight decay as a separate regularization term
   instead of modifying the gradients in Adam. This allows weight decay to more effectively
   act as L2 regularization, avoiding the weight decay side effects present in Adam.

   :param params: Iterator of parameters to optimize or list of dictionaries defining parameter groups
   :type params: Iterable[riemann.TN or riemann.nn.Parameter] or List[Dict[str, Any]]
   :param lr: Learning rate
   :type lr: float, optional
   :param betas: Coefficients used for computing running averages of gradient and its square
   :type betas: Tuple[float, float], optional
   :param eps: Term added to the denominator to improve numerical stability
   :type eps: float, optional
   :param weight_decay: Weight decay (L2 regularization) coefficient
   :type weight_decay: float, optional
   :param amsgrad: Whether to use the AMSGrad variant
   :type amsgrad: bool, optional

   .. method:: step()

      Perform a single optimization step

.. class:: riemann.optim.RMSprop(params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False)

   RMSprop (Root Mean Square Propagation) optimizer

   An adaptive learning rate optimizer particularly suitable for recurrent neural networks (RNNs).
   It adjusts the learning rate for each parameter by maintaining a moving average of squared gradients.

   :param params: Iterator of parameters to optimize or list of dictionaries defining parameter groups
   :type params: Iterable[riemann.TN or riemann.nn.Parameter] or List[Dict[str, Any]]
   :param lr: Learning rate
   :type lr: float, optional
   :param alpha: Smoothing constant used for computing the exponential moving average of squared gradients
   :type alpha: float, optional
   :param eps: Term added to the denominator to improve numerical stability
   :type eps: float, optional
   :param weight_decay: Weight decay (L2 regularization) coefficient
   :type weight_decay: float, optional
   :param momentum: Momentum factor
   :type momentum: float, optional
   :param centered: Whether to use centered RMSprop (using a moving average of gradients)
   :type centered: bool, optional

   .. method:: step()

      Perform a single optimization step

Learning Rate Schedulers
~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: riemann.optim.lr_scheduler.LRScheduler(optimizer, last_epoch=-1, verbose=False)

   Base class for all learning rate schedulers

   :param optimizer: Optimizer whose learning rate will be adjusted
   :type optimizer: riemann.optim.Optimizer
   :param last_epoch: The index of last epoch
   :type last_epoch: int, optional
   :param verbose: Whether to print learning rate updates
   :type verbose: bool, optional

   .. method:: step(epoch=None)

      Perform a single scheduler step

      :param epoch: Current epoch index
      :type epoch: int, optional

   .. method:: get_lr()

      Get the current learning rate for the current epoch

      :return: Learning rate for each parameter group
      :rtype: List[float]

   .. method:: get_last_lr()

      Return the last computed learning rate

      :return: Learning rate for each parameter group
      :rtype: List[float]

   .. method:: state_dict()

      Return the scheduler's state dictionary

      :return: Scheduler state
      :rtype: Dict[str, Any]

   .. method:: load_state_dict(state_dict)

      Load the scheduler state

      :param state_dict: State dictionary to load
      :type state_dict: Dict[str, Any]

.. class:: riemann.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False)

   Decays the learning rate by gamma every step_size epochs

   :param optimizer: Optimizer whose learning rate will be adjusted
   :type optimizer: riemann.optim.Optimizer
   :param step_size: Period of learning rate decay
   :type step_size: int
   :param gamma: Multiplicative factor of learning rate decay
   :type gamma: float, optional
   :param last_epoch: The index of last epoch
   :type last_epoch: int, optional
   :param verbose: Whether to print learning rate updates
   :type verbose: bool, optional

.. class:: riemann.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False)

   Decays the learning rate by gamma at specified milestones

   :param optimizer: Optimizer whose learning rate will be adjusted
   :type optimizer: riemann.optim.Optimizer
   :param milestones: List of epoch indices
   :type milestones: List[int]
   :param gamma: Multiplicative factor of learning rate decay
   :type gamma: float, optional
   :param last_epoch: The index of last epoch
   :type last_epoch: int, optional
   :param verbose: Whether to print learning rate updates
   :type verbose: bool, optional

.. class:: riemann.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1, verbose=False)

   Exponentially decays the learning rate

   :param optimizer: Optimizer whose learning rate will be adjusted
   :type optimizer: riemann.optim.Optimizer
   :param gamma: Multiplicative factor of learning rate decay
   :type gamma: float
   :param last_epoch: The index of last epoch
   :type last_epoch: int, optional
   :param verbose: Whether to print learning rate updates
   :type verbose: bool, optional

.. class:: riemann.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False)

   Anneals the learning rate using cosine function

   :param optimizer: Optimizer whose learning rate will be adjusted
   :type optimizer: riemann.optim.Optimizer
   :param T_max: Maximum number of iterations
   :type T_max: int
   :param eta_min: Minimum learning rate
   :type eta_min: float, optional
   :param last_epoch: The index of last epoch
   :type last_epoch: int, optional
   :param verbose: Whether to print learning rate updates
   :type verbose: bool, optional

.. class:: riemann.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)

   Reduces learning rate when a metric has stopped improving

   :param optimizer: Optimizer whose learning rate will be adjusted
   :type optimizer: riemann.optim.Optimizer
   :param mode: One of 'min' or 'max'
   :type mode: str, optional
   :param factor: Multiplicative factor of learning rate reduction
   :type factor: float, optional
   :param patience: Number of epochs with no improvement after which learning rate will be reduced
   :type patience: int, optional
   :param verbose: Whether to print learning rate updates
   :type verbose: bool, optional
   :param threshold: Threshold for measuring the new optimum
   :type threshold: float, optional
   :param threshold_mode: One of 'rel' or 'abs'
   :type threshold_mode: str, optional
   :param cooldown: Number of epochs to wait before resuming normal operation after learning rate has been reduced
   :type cooldown: int, optional
   :param min_lr: Minimum learning rate
   :type min_lr: float or List[float], optional
   :param eps: Minimal decay applied to lr
   :type eps: float, optional

   .. method:: step(metrics, epoch=None)

      Perform a single scheduler step

      :param metrics: Metric value to check
      :type metrics: float
      :param epoch: Current epoch index
      :type epoch: int, optional


