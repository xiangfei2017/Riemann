Tensor Basics
=============

Tensors are the core data structure in Riemann, similar to NumPy arrays but with additional capabilities for automatic differentiation and gradient tracking.

Creating Tensors
----------------

From Data
~~~~~~~~~~

You can create tensors directly from Python lists or NumPy arrays:

.. code-block:: python

    import riemann as rm
    import numpy as np
    
    # From Python list
    x = rm.tensor([1, 2, 3])
    print(x)  # tensor([1, 2, 3])
    
    # From NumPy array
    np_array = np.array([1, 2, 3])
    x = rm.tensor(np_array)
    print(x)  # tensor([1, 2, 3])

With Specific Data Types
~~~~~~~~~~~~~~~~~~~~~~~~

You can specify the data type when creating tensors:

.. code-block:: python

    # Float32 tensor (default)
    x = rm.tensor([1, 2, 3], dtype=rm.float32)
    
    # Float64 tensor
    x = rm.tensor([1, 2, 3], dtype=rm.float64)
    
    # Complex tensor
    x = rm.tensor([1+2j, 3+4j], dtype=rm.complex64)

Special Tensors
~~~~~~~~~~~~~~~

Riemann provides functions to create special tensors:

.. code-block:: python

    # Zeros tensor
    x = rm.zeros(3, 4)
    
    # Ones tensor
    x = rm.ones(2, 3)
    
    # Identity matrix
    x = rm.eye(3)
    
    # Random tensor
    x = rm.randn(2, 3)  # Normal distribution
    x = rm.rand(2, 3)   # Uniform distribution [0, 1)

Tensor Attributes
-----------------

Tensors have several important attributes:

.. code-block:: python

    x = rm.tensor([[1, 2, 3], [4, 5, 6]], dtype=rm.float32, requires_grad=True)
    
    # Shape
    print(x.shape)  # (2, 3)
    
    # Data type
    print(x.dtype)  # float32
    
    # Number of dimensions
    print(x.ndim)  # 2
    
    # Total number of elements
    print(x.numel())  # 6
    
    # Gradient tracking
    print(x.requires_grad)  # True

Tensor Operations
-----------------

Basic Arithmetic
~~~~~~~~~~~~~~~~

Tensors support standard arithmetic operations:

.. code-block:: python

    x = rm.tensor([1, 2, 3])
    y = rm.tensor([4, 5, 6])
    
    # Addition
    z = x + y
    
    # Subtraction
    z = x - y
    
    # Multiplication (element-wise)
    z = x * y
    
    # Division
    z = x / y
    
    # Matrix multiplication
    a = rm.tensor([[1, 2], [3, 4]])
    b = rm.tensor([[5, 6], [7, 8]])
    c = a @ b  # Matrix multiplication

Mathematical Functions
~~~~~~~~~~~~~~~~~~~~~~

Riemann provides a wide range of mathematical functions:

.. code-block:: python

    x = rm.tensor([0, rm.pi/4, rm.pi/2])
    
    # Trigonometric functions
    y = rm.sin(x)
    y = rm.cos(x)
    y = rm.tan(x)
    
    # Exponential and logarithmic functions
    y = rm.exp(x)
    y = rm.log(x)
    y = rm.sqrt(x)
    
    # Hyperbolic functions
    y = rm.sinh(x)
    y = rm.cosh(x)
    y = rm.tanh(x)

Shape Operations
~~~~~~~~~~~~~~~~

.. code-block:: python

    x = rm.tensor([[1, 2, 3], [4, 5, 6]])
    
    # Reshape
    y = x.reshape(3, 2)
    
    # Transpose
    y = x.T
    
    # Squeeze and unsqueeze
    y = x.unsqueeze(0)  # Add dimension at position 0
    y = y.squeeze(0)   # Remove dimension at position 0
    
    # Concatenation
    a = rm.tensor([1, 2, 3])
    b = rm.tensor([4, 5, 6])
    c = rm.cat([a, b], dim=0)  # tensor([1, 2, 3, 4, 5, 6])

Gradient Tracking
-----------------

Enabling Gradient Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To enable automatic differentiation, set ``requires_grad=True`` when creating a tensor:

.. code-block:: python

    x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * 2
    z = y.sum()
    
    # Compute gradients
    z.backward()
    print(x.grad)  # tensor([2., 2., 2.])

Disabling Gradient Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can disable gradient tracking for performance when you don't need gradients:

.. code-block:: python

    x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    # Method 1: Using no_grad context
    with rm.no_grad():
        y = x * 2  # No gradient tracking for this operation
    
    # Method 2: Using requires_grad_
    x.requires_grad_(False)
    y = x * 2  # No gradient tracking

In-place Operations
-------------------

In-place operations modify the tensor directly without creating a new tensor:

.. code-block:: python

    x = rm.tensor([1, 2, 3])
    
    # In-place addition
    x += 1  # Same as x.add_(1)
    
    # In-place multiplication
    x *= 2  # Same as x.mul_(2)
    
    # In-place assignment
    x[0] = 10

Note: In-place operations can be problematic with gradient tracking. Use them with caution when ``requires_grad=True``.

Saving and Loading Tensors
--------------------------

You can save and load tensors using Riemann's serialization functions:

.. code-block:: python

    # Create tensor
    x = rm.tensor([1, 2, 3])
    
    # Save to file
    rm.save(x, 'tensor.pt')
    
    # Load from file
    y = rm.load('tensor.pt')
    print(y)  # tensor([1, 2, 3])