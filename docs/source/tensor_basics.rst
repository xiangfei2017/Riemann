Tensor Basics
=============

What is a Tensor
----------------

Tensors are the core data structure in Riemann, essentially a 0-dimensional or multi-dimensional array used to quantify and describe objective things such as text, images, videos, audio, and more.

In Riemann, tensors have the following characteristics:

- **Multi-dimensional array structure**: Supports 0-dimensional (scalar), 1-dimensional (vector), 2-dimensional (matrix), and higher-dimensional array representations
- **Mathematical operation support**: Supports basic mathematical operations such as addition, subtraction, multiplication, division, inner product, and various common mathematical functions
- **Shape transformation capabilities**: Supports tensor shape reshaping, dimension expansion/reduction, indexing and slicing operations
- **Automatic gradient tracking**: Built-in automatic differentiation mechanism that supports gradient calculation and backpropagation
- **Device compatibility**: Supports running on different devices such as CPU and GPU

Tensors are the foundation for building neural networks and gradient descent algorithms. The 0-dimensional scalars, 1-dimensional vectors, and 2-dimensional matrices you are familiar with in mathematics are all special forms of tensors.

It should be noted that tensors in Riemann are not exactly equivalent to tensors in tensor algebra or tensor analysis, mainly due to some differences in operation rules. The tensors mentioned here primarily serve neural network-related computations, and their essence is multi-dimensional arrays that support various operators and functions, as well as automatic gradient tracking.

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

With Specific Device
~~~~~~~~~~~~~~~~~~~~

You can specify the device when creating tensors:

.. code-block:: python

    # CPU tensor (default)
    x = rm.tensor([1, 2, 3], device='cpu')
    
    # CUDA tensor
    x = rm.tensor([1, 2, 3], device='cuda')
    
    # Specify CUDA device index
    x = rm.tensor([1, 2, 3], device='cuda:0')

With Gradient Tracking
~~~~~~~~~~~~~~~~~~~~~~

You can specify whether gradient tracking is needed when creating tensors:

.. code-block:: python

    # No gradient tracking (default)
    x = rm.tensor([1, 2, 3], requires_grad=False)
    
    # With gradient tracking (only valid for float types)
    x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)

Tensor Function Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

**tensor function signature**:

.. code-block:: python

    def tensor(data, dtype=None, device=None, requires_grad=False) -> TN

**Parameter explanation**:

- **data**: Can be any data that can be converted to a numpy array, including lists, tuples, scalars, numpy arrays, etc.
- **dtype**: Optional, specifies the data type of the tensor. If None, the data type is inferred from the input data.
- **device**: Optional, specifies the device where the tensor is located, can be 'cpu', 'cuda', 'cuda:0', integer index, or Device object. If None, uses the current device context or default device.
- **requires_grad**: Optional, boolean value, specifies whether gradient calculation is needed for this tensor, default is False.

**Handling of None parameters**:

**When dtype is None**:

- If data is a numpy array or cupy array, preserves the original data type
- If data is a Python scalar:
  - bool → bool
  - int → int64
  - float → default floating-point type (default: float32)
  - complex → default complex type (default: complex64)
- If data is a Python list or tuple, infers data type based on element types (chooses the smallest type that can accommodate all elements)

**When device is None**:

- First checks if in a CUDA device context
- If in CUDA context, uses the current CUDA device
- Otherwise uses the default device (default: CPU)

**Usage examples**:

.. code-block:: python

    # Basic usage
    x = rm.tensor([1, 2, 3])
    
    # Full parameter example
    x = rm.tensor(
        data=[1.0, 2.0, 3.0],
        dtype=rm.float32,
        device='cuda',
        requires_grad=True
    )

**Querying and Setting Defaults**:

**Default data type**:

.. code-block:: python

    # Get current default floating-point type
    default_dtype = rm.get_default_dtype()
    print(default_dtype)  # Default is float32
    
    # Set default floating-point type
    rm.set_default_dtype(rm.float64)
    print(rm.get_default_dtype())  # Now float64

**Default device**:

.. code-block:: python

    # Get current default device
    default_device = rm.get_default_device()
    print(default_device)  # Default is device(type='cpu', index=None)
    
    # Set default device
    rm.set_default_device('cuda')
    print(rm.get_default_device())  # Now device(type='cuda', index=0)
    
    # Set specific CUDA device as default
    rm.set_default_device('cuda:1')
    print(rm.get_default_device())  # Now device(type='cuda', index=1)

**Example: Creating tensors with default settings**:

.. code-block:: python

    # Set default device to CUDA
    rm.set_default_device('cuda')
    
    # Set default data type to float64
    rm.set_default_dtype(rm.float64)
    
    # Create tensor without specifying device and dtype
    # Will use default settings
    x = rm.tensor([1.0, 2.0, 3.0])
    print(x.device)  # cuda:0
    print(x.dtype)   # float64

Data Type and Device Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**dtype object initialization**:

Riemann supports multiple ways to initialize data types:

.. code-block:: python

    # Using Riemann built-in dtype
    dtype = rm.float32
    dtype = rm.float64
    dtype = rm.int32
    dtype = rm.int64
    dtype = rm.complex64
    dtype = rm.complex128
    
    # Using NumPy dtype
    import numpy as np
    dtype = np.float32
    dtype = np.dtype('float64')
    
    # Using string
    dtype = 'float32'
    dtype = 'float64'

**Device object initialization**:

Riemann's Device object can be initialized in the following ways:

.. code-block:: python

    # Using string
    device = rm.device('cpu')
    device = rm.device('cuda')
    device = rm.device('cuda:0')
    
    # Using integer index (CUDA only)
    device = rm.device(0)  # Equivalent to 'cuda:0'
    
    # Through Device constructor
    from riemann import Device
    device = Device('cpu')
    device = Device('cuda:1')

**Device object attributes**:

.. code-block:: python

    device = rm.device('cuda:0')
    print(device.type)  # 'cuda'
    print(device.index)  # 0
    
    device = rm.device('cpu')
    print(device.type)  # 'cpu'
    print(device.index)  # None

Device Context Management
~~~~~~~~~~~~~~~~~~~~~~~~~

Riemann supports using context managers to temporarily switch devices. Tensors created inside the with block will use the specified device by default:

.. code-block:: python

    import riemann as rm
    
    # Create tensor on CPU
    x = rm.tensor([1, 2, 3])
    print(x.device)  # cpu
    
    # Temporarily switch to CUDA device
    with rm.device('cuda'):
        # Create tensor on CUDA
        y = rm.tensor([4, 5, 6])
        print(y.device)  # cuda:0
        
        # When device parameter is not specified, uses context device
        z = rm.tensor([7, 8, 9])
        print(z.device)  # cuda:0
    
    # After exiting context, default device is restored
    w = rm.tensor([10, 11, 12])
    print(w.device)  # cpu

**Advantages of context management**:

- Avoids repeatedly specifying device parameter for each tensor creation
- Ensures all tensors in the code block are on the same device
- Automatically restores previous device state, avoiding device state confusion

**Usage scenarios**:

.. code-block:: python

    # Example: Execute compute-intensive operations on CUDA
    with rm.device('cuda'):
        # Create input tensor
        input_data = rm.tensor([[1.0, 2.0], [3.0, 4.0]])
        
        # Perform computation
        result = rm.matmul(input_data, input_data)
        
        # Result is automatically on CUDA
        print(result.device)  # cuda:0

Special Tensors
~~~~~~~~~~~~~~~

Riemann provides a rich set of special tensor creation functions. The table below lists all supported functions and their capabilities:

.. list-table:: Special Tensor Creation Functions
    :widths: 20 60 20
    :header-rows: 1

    * - Function
      - Description
      - Example
    * - ``zeros``
      - Creates a tensor filled with zeros
      - ``zeros(3, 4)``
    * - ``zeros_like``
      - Creates a tensor of zeros with the same shape as the input
      - ``zeros_like(x)``
    * - ``ones``
      - Creates a tensor filled with ones
      - ``ones(2, 3)``
    * - ``ones_like``
      - Creates a tensor of ones with the same shape as the input
      - ``ones_like(x)``
    * - ``empty``
      - Creates an uninitialized tensor
      - ``empty(2, 3)``
    * - ``empty_like``
      - Creates an uninitialized tensor with the same shape as the input
      - ``empty_like(x)``
    * - ``full``
      - Creates a tensor filled with a specified value
      - ``full((2, 3), 5)``
    * - ``full_like``
      - Creates a tensor filled with a specified value, with the same shape as the input
      - ``full_like(x, 5)``
    * - ``eye``
      - Creates an identity matrix
      - ``eye(3)``
    * - ``rand``
      - Creates a tensor with uniform distribution [0, 1)
      - ``rand(2, 3)``
    * - ``randn``
      - Creates a tensor with standard normal distribution
      - ``randn(2, 3)``
    * - ``randint``
      - Creates a tensor with random integers in a specified range
      - ``randint(0, 10, (2, 3))``
    * - ``normal``
      - Creates a tensor with normal distribution of specified mean and std
      - ``normal(0, 1, (2, 3))``
    * - ``randperm``
      - Creates a tensor with random permutation of integers from 0 to n-1
      - ``randperm(5)``
    * - ``arange``
      - Creates a 1D tensor with evenly spaced values
      - ``arange(0, 10, 2)``
    * - ``linspace``
      - Creates a 1D tensor with specified number of evenly spaced values
      - ``linspace(0, 1, 5)``

**Usage examples**:

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
    
    # Filled tensor
    x = rm.full((2, 3), 5)  # Creates 2x3 tensor filled with 5
    
    # Sequence tensors
    x = rm.arange(0, 10, 2)  # 0, 2, 4, 6, 8
    x = rm.linspace(0, 1, 5)  # 0, 0.25, 0.5, 0.75, 1.0

Default Parameter Behavior for Special Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When creating special tensors, if you don't specify `dtype` and `device` parameters, different default behaviors are used based on the function type:

**Tensor Creation Functions Without Reference Tensor** (e.g., zeros, ones, rand, etc.):

- When `dtype` and `device` parameters are not specified, the function behavior is consistent with the `tensor()` function
- Default data type is `float32`
- Default device is the current device context or default device setting

**Reference Tensor-based "like" Functions** (e.g., zeros_like, ones_like, etc.):

- When `dtype` and `device` parameters are not specified, the `dtype` and `device` of the reference tensor are used to create the tensor
- This ensures that the newly created tensor has the same data type and device as the reference tensor

**Default Data Type (dtype)**:

.. code-block:: python

    # Default creates float32 tensor
    x = rm.zeros(3, 4)
    print(x.dtype)  # float32
    
    # Explicitly specify data type
    x = rm.zeros(3, 4, dtype=rm.float64)
    print(x.dtype)  # float64

**Default Device (device)**:

- When not specifying the `device` parameter, the current device context or default device setting is used
- This is consistent with the default behavior of the `tensor()` function

.. code-block:: python

    # Default uses current device context or default device
    x = rm.zeros(3, 4)
    print(x.device)  # Defaults to cpu
    
    # Create within CUDA context
    with rm.device('cuda'):
        x = rm.zeros(3, 4)
        print(x.device)  # cuda:0
    
    # Explicitly specify device
    x = rm.zeros(3, 4, device='cuda')
    print(x.device)  # cuda:0

**Complete Parameter Example**:

.. code-block:: python

    # Specify all parameters
    x = rm.zeros(
        3, 4,            # Shape
        dtype=rm.float32,  # Data type
        device='cuda',    # Device
        requires_grad=True  # Gradient tracking
    )
    
    # Random tensor example
    x = rm.randn(
        2, 3,            # Shape
        dtype=rm.float64,  # Data type
        device='cpu',     # Device
        requires_grad=False  # Gradient tracking
    )

Tensor Attributes and States
----------------------------

Tensors have various attributes and state detection functions for obtaining basic information and detecting states.

**Tensor Attributes**

.. list-table:: Tensor Attributes
  :widths: 15 45 40
  :header-rows: 1

  * - Attribute
    - Description
    - Example
  * - ``dtype``
    - Tensor data type
    - ``x.dtype`` → ``float32``
  * - ``device``
    - Device where tensor is located
    - ``x.device`` → ``cpu`` or ``cuda:0``
  * - ``ndim``
    - Number of tensor dimensions
    - ``x.ndim`` → ``2``
  * - ``shape``
    - Tensor shape
    - ``x.shape`` → ``(2, 3)``
  * - ``size``
    - Size of tensor in specified dimension
    - ``x.size(0)`` → ``2``
  * - ``numel``
    - Total number of tensor elements
    - ``x.numel()`` → ``6``
  * - ``is_leaf``
    - Whether tensor is a leaf node in computation graph
    - ``x.is_leaf`` → ``True``
  * - ``requires_grad``
    - Whether tensor requires gradient tracking
    - ``x.requires_grad`` → ``True``

**State Detection Functions**

.. list-table:: State Detection Functions
  :widths: 15 45 40
  :header-rows: 1

  * - Function
    - Description
    - Example
  * - ``is_floating_point``
    - Detect if tensor is floating point type
    - ``x.is_floating_point()`` → ``True``
  * - ``is_complex``
    - Detect if tensor is complex type
    - ``x.is_complex()`` → ``False``
  * - ``isreal``
    - Detect if tensor is real type
    - ``x.isreal()`` → ``True``
  * - ``isinf``
    - Detect if tensor elements are infinite
    - ``x.isinf()`` → boolean tensor
  * - ``isnan``
    - Detect if tensor elements are NaN
    - ``x.isnan()`` → boolean tensor
  * - ``is_cuda``
    - Detect if tensor is on CUDA device
    - ``x.is_cuda`` → ``False``
  * - ``is_cpu``
    - Detect if tensor is on CPU device
    - ``x.is_cpu`` → ``True``
  * - ``type``
    - Get or set tensor data type
    - ``x.type()`` → ``float32`` or ``x.type(rm.float64)``
  * - ``is_contiguous``
    - Detect if tensor is stored contiguously
    - ``x.is_contiguous()`` → ``True``

**Attributes and State Detection Examples**

.. code-block:: python

    import riemann as rm

    # Create a tensor
    x = rm.tensor([[1, 2, 3], [4, 5, 6]], dtype=rm.float32, requires_grad=True)
    print("Original tensor:", x)

    # 1. Basic attributes
    print("\n1. Basic attributes:")
    print("Data type:", x.dtype)
    print("Device:", x.device)
    print("Number of dimensions:", x.ndim)
    print("Shape:", x.shape)
    print("Size of dimension 0:", x.size(0))
    print("Total number of elements:", x.numel())
    print("Is leaf node:", x.is_leaf)
    print("Requires gradient:", x.requires_grad)

    # 2. State detection
    print("\n2. State detection:")
    print("Is floating point:", x.is_floating_point())
    print("Is complex:", x.is_complex())
    print("Is real:", x.isreal())
    print("Is on CUDA:", x.is_cuda)
    print("Is on CPU:", x.is_cpu)
    print("Is contiguous:", x.is_contiguous())

    # 3. Special value detection
    print("\n3. Special value detection:")
    y = rm.tensor([1.0, float('inf'), float('nan')])
    print("Tensor:", y)
    print("Is infinite:", y.isinf())
    print("Is NaN:", y.isnan())

    # 4. Type operations
    print("\n4. Type operations:")
    print("Current type:", x.type())
    x_double = x.type(rm.float64)
    print("Type after conversion:", x_double.type())

Tensor Computations
-------------------

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

Riemann provides a wide range of mathematical functions. Here are the commonly used ones:

**Basic Mathematical Functions**

.. list-table:: Basic Mathematical Functions
  :widths: 15 45 40
  :header-rows: 1

  * - Function
    - Description
    - Example
  * - ``abs``
    - Compute absolute value
    - ``rm.abs(x)``
  * - ``sqrt``
    - Compute square root
    - ``rm.sqrt(x)``
  * - ``square``
    - Compute square
    - ``rm.square(x)``
  * - ``exp``
    - Compute exponential function
    - ``rm.exp(x)``
  * - ``exp2``
    - Compute 2 to the power
    - ``rm.exp2(x)``
  * - ``log``
    - Compute natural logarithm
    - ``rm.log(x)``
  * - ``log10``
    - Compute base-10 logarithm
    - ``rm.log10(x)``
  * - ``log2``
    - Compute base-2 logarithm
    - ``rm.log2(x)``
  * - ``sign``
    - Compute sign function
    - ``rm.sign(x)``
  * - ``ceil``
    - Round up
    - ``rm.ceil(x)``
  * - ``floor``
    - Round down
    - ``rm.floor(x)``
  * - ``round``
    - Round to nearest integer
    - ``rm.round(x)``
  * - ``trunc``
    - Truncate decimal part
    - ``rm.trunc(x)``

**Trigonometric Functions**

.. list-table:: Trigonometric Functions
  :widths: 15 45 40
  :header-rows: 1

  * - Function
    - Description
    - Example
  * - ``sin``
    - Compute sine
    - ``rm.sin(x)``
  * - ``cos``
    - Compute cosine
    - ``rm.cos(x)``
  * - ``tan``
    - Compute tangent
    - ``rm.tan(x)``
  * - ``arcsin``
    - Compute arcsine
    - ``rm.arcsin(x)``
  * - ``arccos``
    - Compute arccosine
    - ``rm.arccos(x)``
  * - ``arctan``
    - Compute arctangent
    - ``rm.arctan(x)``
  * - ``arctan2``
    - Compute arctangent of two tensors
    - ``rm.arctan2(y, x)``

**Hyperbolic Functions**

.. list-table:: Hyperbolic Functions
  :widths: 15 45 40
  :header-rows: 1

  * - Function
    - Description
    - Example
  * - ``sinh``
    - Compute hyperbolic sine
    - ``rm.sinh(x)``
  * - ``cosh``
    - Compute hyperbolic cosine
    - ``rm.cosh(x)``
  * - ``tanh``
    - Compute hyperbolic tangent
    - ``rm.tanh(x)``
  * - ``arcsinh``
    - Compute inverse hyperbolic sine
    - ``rm.arcsinh(x)``
  * - ``arccosh``
    - Compute inverse hyperbolic cosine
    - ``rm.arccosh(x)``
  * - ``arctanh``
    - Compute inverse hyperbolic tangent
    - ``rm.arctanh(x)``

**Mathematical Functions Example**

.. code-block:: python

    import riemann as rm

    # Create example tensor
    x = rm.tensor([-2.5, 0.0, 1.5, 3.0])
    print("Original tensor:", x)

    # Basic mathematical functions
    print("\n1. Basic mathematical functions:")
    print("Absolute value:", rm.abs(x))
    print("Square root:", rm.sqrt(rm.abs(x)))
    print("Square:", rm.square(x))
    print("Exponential:", rm.exp(x))
    print("Natural logarithm:", rm.log(rm.abs(x) + 1e-10))

    # Rounding functions
    print("\n2. Rounding functions:")
    print("Ceiling:", rm.ceil(x))
    print("Floor:", rm.floor(x))
    print("Round:", rm.round(x))
    print("Truncate:", rm.trunc(x))

    # Trigonometric functions
    print("\n3. Trigonometric functions:")
    angles = rm.tensor([0, rm.pi/4, rm.pi/2, rm.pi])
    print("Angles:", angles)
    print("Sine:", rm.sin(angles))
    print("Cosine:", rm.cos(angles))
    print("Tangent:", rm.tan(angles))

    # Hyperbolic functions
    print("\n4. Hyperbolic functions:")
    print("Hyperbolic sine:", rm.sinh(x))
    print("Hyperbolic cosine:", rm.cosh(x))
    print("Hyperbolic tangent:", rm.tanh(x))

Statistical Functions
~~~~~~~~~~~~~~~~~~~~~

Riemann provides various statistical functions for tensor analysis. Here are the commonly used ones:

**Common Statistical Functions**

.. list-table:: Statistical Functions
  :widths: 15 45 40
  :header-rows: 1

  * - Function
    - Description
    - Example
  * - ``sum``
    - Compute sum of tensor elements
    - ``rm.sum(x)``
  * - ``sumall``
    - Compute sum of multiple tensors
    - ``rm.sumall(x, y, z)``
  * - ``mean``
    - Compute mean of tensor elements
    - ``rm.mean(x)``
  * - ``var``
    - Compute variance of tensor elements
    - ``rm.var(x)``
  * - ``std``
    - Compute standard deviation of tensor elements
    - ``rm.std(x)``
  * - ``norm``
    - Compute norm of tensor
    - ``rm.norm(x)``
  * - ``max``
    - Compute maximum value of tensor elements
    - ``rm.max(x)``
  * - ``min``
    - Compute minimum value of tensor elements
    - ``rm.min(x)``
  * - ``maximum``
    - Compute element-wise maximum of two tensors
    - ``rm.maximum(x, y)``
  * - ``minimum``
    - Compute element-wise minimum of two tensors
    - ``rm.minimum(x, y)``
  * - ``where``
    - Select elements based on condition
    - ``rm.where(condition, x, y)``
  * - ``clamp``
    - Clamp tensor values to a range
    - ``rm.clamp(x, min, max)``
  * - ``sort``
    - Sort tensor elements
    - ``rm.sort(x)``
  * - ``argsort``
    - Return indices that sort tensor
    - ``rm.argsort(x)``
  * - ``argmax``
    - Return index of maximum value
    - ``rm.argmax(x)``
  * - ``argmin``
    - Return index of minimum value
    - ``rm.argmin(x)``

**Statistical Functions Example**

.. code-block:: python

    import riemann as rm

    # Create example tensor
    x = rm.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    y = rm.tensor([[2.0, 1.0, 4.0], [3.0, 6.0, 5.0], [8.0, 7.0, 10.0]])
    z = rm.tensor([[1.0, 3.0, 2.0], [4.0, 2.0, 1.0], [3.0, 4.0, 5.0]])
    print("Original tensor x:", x)
    print("Original tensor y:", y)
    print("Original tensor z:", z)

    # 1. sum function
    print("\n1. sum function:")
    print("Sum of all elements:", rm.sum(x))
    print("Sum along axis 0:", rm.sum(x, dim=0))
    print("Sum along axis 1:", rm.sum(x, dim=1))

    # 2. sumall function
    print("\n2. sumall function:")
    print("Sum of multiple tensors:", rm.sumall(x, y, z))

    # 3. mean function
    print("\n3. mean function:")
    print("Mean of all elements:", rm.mean(x))
    print("Mean along axis 0:", rm.mean(x, dim=0))

    # 4. max and min functions
    print("\n4. max and min functions:")
    print("Maximum value:", rm.max(x))
    print("Minimum value:", rm.min(x))
    print("Maximum along axis 0:", rm.max(x, dim=0))
    print("Minimum along axis 1:", rm.min(x, dim=1))

    # 5. where function
    print("\n5. where function:")
    condition = x > 5
    result = rm.where(condition, x, y)
    print("Condition (x > 5):", condition)
    print("Where result:", result)

**where Function Detailed Example**

The where function has two main use cases:
1. When only condition is provided, returns indices of elements that satisfy the condition
2. When condition, x, and y are provided, selects elements from x and y based on the condition

.. code-block:: python

    import riemann as rm

    # Create example tensors
    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = rm.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
    print("Original tensor x:")
    print(x)
    print("Original tensor y:")
    print(y)

    # Use case 1: Only provide condition, return indices of elements that satisfy the condition
    print("\nUse case 1: Only provide condition")
    condition = x > 5
    indices = rm.where(condition)
    print("Condition (x > 5):")
    print(condition)
    print("Indices of elements that satisfy the condition:")
    print("Row indices:", indices[0])
    print("Column indices:", indices[1])
    print("Index tuple:", indices)

    # Use case 2: Provide condition, x, and y, select elements based on condition
    print("\nUse case 2: Provide condition, x, and y")
    
    # Basic usage
    result1 = rm.where(condition, x, y)
    print("Basic usage result (take x where x > 5, otherwise take y):")
    print(result1)
    
    # Use scalar as x or y
    result2 = rm.where(condition, 100, y)
    print("\nResult with scalar as x (take 100 where x > 5, otherwise take y):")
    print(result2)
    
    result3 = rm.where(condition, x, 0)
    print("\nResult with scalar as y (take x where x > 5, otherwise take 0):")
    print(result3)

    # Use tensors of different shapes (broadcasting will be applied)
    print("\nUsing tensors of different shapes")
    condition_1d = rm.tensor([True, False, True])  # 1D condition
    x_1d = rm.tensor([100, 200, 300])  # 1D x
    
    result4 = rm.where(condition_1d, x_1d, y)
    print("Result with 1D condition and 1D x vs 2D y:")
    print(result4)

    # where function with gradient tracking
    print("\nwhere function with gradient tracking")
    x_grad = rm.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    y_grad = rm.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], requires_grad=True)
    condition_grad = x_grad > 3.0
    
    result_grad = rm.where(condition_grad, x_grad, y_grad)
    print("Result with gradient tracking:")
    print(result_grad)
    
    # Compute gradients
    sum_result = rm.sum(result_grad)
    sum_result.backward()
    
    print("\nGradient computation result:")
    print("Gradient of x_grad:")
    print(x_grad.grad)
    print("Gradient of y_grad:")
    print(y_grad.grad)

**sumall Function Efficiency Advantage**

The `sumall` function is more efficient than using tensor addition operations, especially in gradient tracking, because:

1. **Reduced Computation Graph**: When using `sumall`, the computation graph is reduced to a single layer, regardless of the number of tensors being summed.
2. **Scalable Efficiency**: With tensor addition operators (`+`), the computation graph grows linearly with each additional tensor, leading to increased graph complexity.
3. **Faster Gradient Tracking**: The simpler graph structure of `sumall` results in much faster gradient computation during backpropagation, especially when summing many tensors.

**Gradient Tracking Efficiency Example**

.. code-block:: python

    import riemann as rm

    # Create tensors with gradient tracking
    x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = rm.tensor([4.0, 5.0, 6.0], requires_grad=True)
    z = rm.tensor([7.0, 8.0, 9.0], requires_grad=True)
    w = rm.tensor([10.0, 11.0, 12.0], requires_grad=True)

    # Using sumall (more efficient)
    print("\nUsing sumall:")
    result_sumall = rm.sumall(x, y, z, w)
    result_sumall.backward()
    print("x.grad:", x.grad)
    print("y.grad:", y.grad)
    print("z.grad:", z.grad)
    print("w.grad:", w.grad)

    # Reset gradients
    x.grad = None
    y.grad = None
    z.grad = None
    w.grad = None

    # Using addition operators (less efficient)
    print("\nUsing addition operators:")
    result_addition = x + y + z + w
    result_addition.backward()
    print("x.grad:", x.grad)
    print("y.grad:", y.grad)
    print("z.grad:", z.grad)
    print("w.grad:", w.grad)

**Other Statistical Functions Example**

.. code-block:: python

    import riemann as rm

    # Create example tensor
    x = rm.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    y = rm.tensor([[2.0, 1.0, 4.0], [3.0, 6.0, 5.0], [8.0, 7.0, 10.0]])
    print("Original tensor x:", x)
    print("Original tensor y:", y)

    # 1. clamp function
    print("\n1. clamp function:")
    clamped = rm.clamp(x, min=3.0, max=7.0)
    print("Clamped between 3 and 7:", clamped)

    # 2. argmax function
    print("\n2. argmax function:")
    print("Index of maximum value:", rm.argmax(x))
    print("Indices of maximum along axis 0:", rm.argmax(x, dim=0))

    # 3. maximum function
    print("\n3. maximum function:")
    max_result = rm.maximum(x, y)
    print("Element-wise maximum of x and y:", max_result)

    # 4. sort and argsort functions
    print("\n4. sort and argsort functions:")
    sorted_x, indices = rm.sort(x, dim=1, return_indices=True)
    print("Sorted along axis 1:", sorted_x)
    print("Sort indices:", indices)
    
    argsorted = rm.argsort(x, dim=1)
    print("Argsort along axis 1:", argsorted)

**Statistical Functions with Gradient Tracking**

.. code-block:: python

    import riemann as rm

    # Create tensor with gradient tracking
    x = rm.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    print("Original tensor with grad:", x)

    # 1. sum with gradient tracking
    print("\n1. sum with gradient tracking:")
    sum_result = rm.sum(x)
    print("Sum result:", sum_result)
    sum_result.backward()
    print("Gradient of sum:", x.grad)

    # Reset gradients
    x.grad = None

    # 2. mean with gradient tracking
    print("\n2. mean with gradient tracking:")
    mean_result = rm.mean(x)
    print("Mean result:", mean_result)
    mean_result.backward()
    print("Gradient of mean:", x.grad)

    # Reset gradients
    x.grad = None

    # 3. max with gradient tracking
    print("\n3. max with gradient tracking:")
    max_result = rm.max(x)
    print("Max result:", max_result)
    max_result.backward()
    print("Gradient of max:", x.grad)

Tensor Comparison Operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Riemann supports various tensor comparison operators for comparing tensor elements.

.. list-table:: Tensor Comparison Operators
  :widths: 15 35 25 25
  :header-rows: 1

  * - Operator
    - Description
    - Example
    - Result Type
  * - ``==``
    - Equal
    - ``x == y``
    - Boolean tensor
  * - ``!=``
    - Not equal
    - ``x != y``
    - Boolean tensor
  * - ``<``
    - Less than
    - ``x < y``
    - Boolean tensor
  * - ``<=``
    - Less than or equal
    - ``x <= y``
    - Boolean tensor
  * - ``>``
    - Greater than
    - ``x > y``
    - Boolean tensor
  * - ``>=``
    - Greater than or equal
    - ``x >= y``
    - Boolean tensor

**Comparison Operators Example**

.. code-block:: python

    import riemann as rm

    # Create example tensors
    x = rm.tensor([1, 2, 3, 4])
    y = rm.tensor([2, 2, 2, 2])
    print("x:", x)
    print("y:", y)

    # Comparison operations
    print("\nComparison results:")
    print("x == y:", x == y)
    print("x != y:", x != y)
    print("x < y:", x < y)
    print("x <= y:", x <= y)
    print("x > y:", x > y)
    print("x >= y:", x >= y)

Tensor Logical Operators
~~~~~~~~~~~~~~~~~~~~~~~~

Riemann supports various tensor logical operators for logical operations on boolean tensors.

.. list-table:: Tensor Logical Operators
  :widths: 15 35 25 25
  :header-rows: 1

  * - Operator
    - Description
    - Example
    - Result Type
  * - ``&``
    - Logical AND
    - ``x & y``
    - Boolean tensor
  * - ``|``
    - Logical OR
    - ``x | y``
    - Boolean tensor
  * - ``^``
    - Logical XOR
    - ``x ^ y``
    - Boolean tensor
  * - ``~``
    - Logical NOT
    - ``~x``
    - Boolean tensor

**Logical Operators Example**

.. code-block:: python

    import riemann as rm

    # Create boolean tensors
    x = rm.tensor([True, True, False, False])
    y = rm.tensor([True, False, True, False])
    print("x:", x)
    print("y:", y)

    # Logical operations
    print("\nLogical operation results:")
    print("x & y:", x & y)
    print("x | y:", x | y)
    print("x ^ y:", x ^ y)
    print("~x:", ~x)

Tensor Bitwise Operators
~~~~~~~~~~~~~~~~~~~~~~~~

Riemann supports various tensor bitwise operators for bitwise operations on integer tensors.

.. list-table:: Tensor Bitwise Operators
  :widths: 15 35 25 25
  :header-rows: 1

  * - Operator
    - Description
    - Example
    - Result Type
  * - ``&``
    - Bitwise AND
    - ``x & y``
    - Integer tensor
  * - ``|``
    - Bitwise OR
    - ``x | y``
    - Integer tensor
  * - ``^``
    - Bitwise XOR
    - ``x ^ y``
    - Integer tensor
  * - ``~``
    - Bitwise NOT
    - ``~x``
    - Integer tensor
  * - ``<<``
    - Left shift
    - ``x << y``
    - Integer tensor
  * - ``>>``
    - Right shift
    - ``x >> y``
    - Integer tensor

**Bitwise Operators Example**

.. code-block:: python

    import riemann as rm

    # Create integer tensors
    x = rm.tensor([1, 3, 5, 7], dtype=rm.int32)
    y = rm.tensor([1, 2, 3, 4], dtype=rm.int32)
    print("x:", x)
    print("y:", y)

    # Bitwise operations
    print("\nBitwise operation results:")
    print("x & y:", x & y)
    print("x | y:", x | y)
    print("x ^ y:", x ^ y)
    print("~x:", ~x)
    print("x << 1:", x << 1)
    print("x >> 1:", x >> 1)


Tensor Check and Comparison Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Riemann provides various tensor check and comparison functions for checking tensor properties or comparing multiple tensors.

.. list-table:: Tensor Check and Comparison Functions
  :widths: 15 45 40
  :header-rows: 1

  * - Function
    - Description
    - Example
  * - ``all``
    - Check if all elements are true
    - ``rm.all(x)``
  * - ``any``
    - Check if any element is true
    - ``rm.any(x)``
  * - ``allclose``
    - Check if two tensors are equal within tolerance
    - ``rm.allclose(x, y, rtol=1e-05, atol=1e-08)``
  * - ``equal``
    - Check if two tensors are element-wise equal
    - ``rm.equal(x, y)``
  * - ``not_equal``
    - Check if two tensors are element-wise not equal
    - ``rm.not_equal(x, y)``
  * - ``nonzero``
    - Return indices of non-zero elements
    - ``rm.nonzero(x)``
  * - ``unique``
    - Return unique elements in tensor
    - ``rm.unique(x)``

**Check and Comparison Functions Example**

.. code-block:: python

    import riemann as rm

    # Create example tensors
    x = rm.tensor([True, True, True])
    y = rm.tensor([True, False, True])
    z = rm.tensor([1.0, 2.0, 3.0])
    w = rm.tensor([1.0, 2.0000001, 3.0])

    print("x:", x)
    print("y:", y)
    print("z:", z)
    print("w:", w)

    # Check functions
    print("\n1. Check functions:")
    print("all(x):", rm.all(x))
    print("any(y):", rm.any(y))

    # Comparison functions
    print("\n2. Comparison functions:")
    print("equal(z, w):", rm.equal(z, w))
    print("not_equal(z, w):", rm.not_equal(z, w))
    print("allclose(z, w):", rm.allclose(z, w))
    print("allclose(z, w, rtol=1e-03):", rm.allclose(z, w, rtol=1e-03))

Shape and Dimension Manipulation Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following table lists all shape and dimension manipulation functions supported by Riemann:

.. list-table:: Shape and Dimension Manipulation Functions
  :widths: 15 55 30
  :header-rows: 1

  * - Function
    - Description
    - Example
  * - ``reshape``
    - Changes tensor shape without modifying data, supports -1 for auto-inference
    - ``x.reshape(3, 2)`` or ``x.reshape(-1, 2)``
  * - ``view``
    - Alias for reshape, returns a view with the same data but different shape
    - ``x.view(3, 2)``
  * - ``flatten``
    - Flattens a range of tensor dimensions into one dimension
    - ``x.flatten(start_dim=0, end_dim=-1)``
  * - ``squeeze``
    - Removes dimensions of size 1
    - ``x.squeeze()`` or ``x.squeeze(0)``
  * - ``unsqueeze``
    - Adds a dimension of size 1 at the specified position
    - ``x.unsqueeze(0)``
  * - ``expand``
    - Expands tensor to specified shape, can only expand dimensions of size 1
    - ``x.expand(3, 4)`` or ``x.expand(-1, 4)``
  * - ``expand_as``
    - Expands tensor to the same shape as another tensor
    - ``x.expand_as(y)``
  * - ``repeat``
    - Repeats tensor elements along specified dimensions
    - ``x.repeat(2, 3)``
  * - ``transpose``
    - Swaps two specified dimensions of the tensor
    - ``x.transpose(0, 1)``
  * - ``permute``
    - Rearranges tensor dimensions according to specified order
    - ``x.permute(2, 0, 1)``
  * - ``flip``
    - Flips tensor along specified dimensions
    - ``x.flip([0, 1])``
  * - ``T``
    - Tensor transpose property, reverses entire dimension order for high-dimensional tensors
    - ``x.T``
  * - ``mT``
    - Matrix transpose property, swaps only the last two dimensions
    - ``x.mT``
  * - ``H``
    - Tensor conjugate transpose property
    - ``x.H``
  * - ``mH``
    - Matrix conjugate transpose property, conjugate transpose of last two dimensions
    - ``x.mH``
  * - ``cat`` / ``concatenate``
    - Concatenates tensor sequence along specified dimension
    - ``rm.cat([x, y], dim=0)``
  * - ``stack``
    - Stacks tensor sequence along a new dimension
    - ``rm.stack([x, y], dim=0)``
  * - ``vstack``
    - Vertically stacks tensors, 1D tensors as rows, multi-dimensional along axis 0
    - ``rm.vstack([x, y])``
  * - ``hstack``
    - Horizontally stacks tensors, 1D tensors concatenated, multi-dimensional along axis 1
    - ``rm.hstack([x, y])``

Tensor Type Conversion
----------------------

Riemann provides various functions for tensor type conversion, including data type conversion and device switching.

**Data Type Conversion Functions**

.. list-table:: Data Type Conversion Functions
  :widths: 15 45 40
  :header-rows: 1

  * - Function
    - Description
    - Example
  * - ``type``
    - Convert tensor to specified data type
    - ``x.type(rm.float64)``
  * - ``type_as``
    - Convert tensor to the same data type as another tensor
    - ``x.type_as(y)``
  * - ``to``
    - General conversion function, can convert data type and device
    - ``x.to(rm.float32)`` or ``x.to('cuda')``
  * - ``bool``
    - Convert tensor to boolean type
    - ``x.bool()``
  * - ``float``
    - Convert tensor to float32 type
    - ``x.float()``
  * - ``double``
    - Convert tensor to float64 type
    - ``x.double()``
  * - ``real``
    - Return real part of complex tensor
    - ``x.real()``
  * - ``imag``
    - Return imaginary part of complex tensor
    - ``x.imag()``
  * - ``conj``
    - Return complex conjugate of complex tensor
    - ``x.conj()``

**Device Switching Functions**

.. list-table:: Device Switching Functions
  :widths: 15 45 40
  :header-rows: 1

  * - Function
    - Description
    - Example
  * - ``cuda``
    - Move tensor to CUDA device
    - ``x.cuda()`` or ``x.cuda(0)``
  * - ``cpu``
    - Move tensor to CPU device
    - ``x.cpu()``
  * - ``to``
    - General device switching function
    - ``x.to('cuda')`` or ``x.to('cpu')``

**to() Function Detailed Parameters**

.. list-table:: to() Function Parameters
  :widths: 15 30 40 15
  :header-rows: 1
  :align: center

  * - Parameter
    - Type
    - Description
    - Default
  * - ``other``
    - TN
    - Another tensor, use its device and dtype as target for migration
    - None
  * - ``device``
    - str or Device
    - Target device, can be a string (e.g. 'cpu', 'cuda') or Device object
    - None
  * - ``dtype``
    - dtype
    - Target data type, can be Python type, NumPy dtype, string or Riemann dtype
    - None
  * - ``non_blocking``
    - bool
    - If True and data is in pinned memory, copying to GPU can be asynchronous with host computation. Only applicable for CPU -> GPU transfers
    - False
  * - ``copy``
    - bool
    - If True, always return a copy, even if device and dtype are the same
    - False

**to() Function Usage Examples**

.. code-block:: python

    import riemann as rm

    # Convert data type
    x = rm.tensor([1.0, 2.0, 3.0], dtype=rm.float32)
    y = x.to(rm.float64)
    print(f"Converted dtype: {y.dtype}")

    # Convert device
    x = rm.tensor([1.0, 2.0, 3.0], device='cpu')
    y = x.to('cuda')
    print(f"Converted device: {y.device}")

    # Convert both dtype and device
    x = rm.tensor([1.0, 2.0, 3.0], dtype=rm.float32, device='cpu')
    y = x.to(rm.float64, device='cuda')
    print(f"Converted dtype: {y.dtype}, device: {y.device}")

    # Use keyword arguments
    x = rm.tensor([1.0, 2.0, 3.0])
    y = x.to(dtype=rm.float64, device='cuda')

    # Copy dtype and device from another tensor
    x = rm.tensor([1.0, 2.0, 3.0], dtype=rm.float64, device='cuda')
    y = rm.tensor([4.0, 5.0, 6.0])
    z = y.to(x)
    print(f"Copied from x: dtype={z.dtype}, device={z.device}")

    # Force copy
    y = x.to(copy=True)

**non_blocking Parameter Usage Example**

.. code-block:: python

    import riemann as rm

    # Create tensor on CPU
    x = rm.tensor([1.0, 2.0, 3.0], device='cpu')

    # Asynchronous transfer to GPU
    # Note: Asynchronous transfer requires data to be in pinned memory
    # In practice, it's recommended to synchronize device after transfer to ensure completion
    y = x.to('cuda', non_blocking=True)

    # Perform some CPU computations
    # These can run in parallel with data transfer
    cpu_result = x * 2

    # Synchronize device to ensure data transfer is complete
    # Must synchronize before accessing the GPU tensor
    rm.cuda.synchronize()

    # Now it's safe to use the GPU tensor
    gpu_result = y * 2

**Type Conversion Examples**

.. code-block:: python

    import riemann as rm

    # Create an integer tensor
    x = rm.tensor([1, 2, 3], dtype=rm.int32)
    print("Original tensor:", x)
    print("Original data type:", x.dtype)
    print("Original device:", x.device)

    # 1. Data type conversion
    print("\n1. Data type conversion:")
    x_float = x.float()
    print("Convert to float32:", x_float)
    print("Data type:", x_float.dtype)

    x_double = x.double()
    print("\nConvert to float64:", x_double)
    print("Data type:", x_double.dtype)

    x_bool = x.bool()
    print("\nConvert to bool:", x_bool)
    print("Data type:", x_bool.dtype)

    # 2. Using to function for conversion
    print("\n2. Using to function for conversion:")
    x_to_float = x.to(rm.float32)
    print("Using to convert to float32:", x_to_float.dtype)

    # 3. Complex number related conversions
    print("\n3. Complex number related conversions:")
    z = rm.tensor([1+2j, 3+4j], dtype=rm.complex64)
    print("Complex tensor:", z)
    print("Real part:", z.real())
    print("Imaginary part:", z.imag())
    print("Conjugate:", z.conj())

    # 4. Device switching (if CUDA is available)
    print("\n4. Device switching:")
    if rm.cuda.is_available():
        x_cuda = x.cuda()
        print("Move to CUDA device:", x_cuda.device)
        
        x_back_to_cpu = x_cuda.cpu()
        print("Move back to CPU device:", x_back_to_cpu.device)
    else:
        print("CUDA not available, skipping device switching example")

**Notes on Type Conversion**

1. **Data Type Conversion**:

   - Converting from higher precision to lower precision types may cause precision loss
   - Converting from integer to floating point types is safe
   - Converting from floating point to integer types will truncate the decimal part

2. **Device Switching**:

   - Device switching creates a new tensor copy, consuming memory and time
   - Ensure the target device is available before switching
   - Tensors on different devices cannot be directly operated on, they need to be unified first

3. **Complex Number Conversion**:

   - ``real()`` and ``imag()`` functions return the real and imaginary parts of complex tensors, resulting in floating point types
   - ``conj()`` function returns the complex conjugate of complex tensors, resulting in complex type

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

Gradient Context Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Riemann provides various gradient context management tools to control gradient tracking behavior within specific code blocks. These tools can be used with `with` statements or decorators.

**Using with Statements for Gradient Context Control**

.. code-block:: python

    import riemann as rm

    # Create tensor with gradient tracking
    x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)

    # 1. Using no_grad() to disable gradient tracking
    print("\n1. Using no_grad() to disable gradient tracking:")
    with rm.no_grad():
        y = x * 2
        print("y.requires_grad:", y.requires_grad)  # False

    # 2. Using enable_grad() to enable gradient tracking
    print("\n2. Using enable_grad() to enable gradient tracking:")
    with rm.no_grad():
        # In this context, gradient tracking is disabled by default
        z = x + 1
        print("z.requires_grad:", z.requires_grad)  # False
        
        # But we can enable gradient tracking internally
        with rm.enable_grad():
            w = x * 3
            print("w.requires_grad:", w.requires_grad)  # True

    # 3. Using set_grad_enabled() to manually set gradient tracking state
    print("\n3. Using set_grad_enabled() to manually set gradient tracking state:")
    with rm.set_grad_enabled(True):
        a = x * 4
        print("a.requires_grad:", a.requires_grad)  # True
    
    with rm.set_grad_enabled(False):
        b = x * 5
        print("b.requires_grad:", b.requires_grad)  # False

**Using Decorators for Gradient Context Control**

In addition to `with` statements, Riemann also provides gradient context management tools in the form of decorators, which control gradient tracking behavior for entire functions.

.. code-block:: python

    import riemann as rm

    # Create tensor with gradient tracking
    x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)

    # Using @no_grad decorator to disable gradient tracking in the function
    @rm.no_grad
    def inference_fn(tensor):
        """Inference function, no gradient tracking needed"""
        result = tensor * 2 + 1
        print("inference_fn: result.requires_grad =", result.requires_grad)
        return result

    # Using @enable_grad decorator to enable gradient tracking in the function
    @rm.enable_grad
    def training_fn(tensor):
        """Training function, gradient tracking needed"""
        result = tensor * 3 + 2
        print("training_fn: result.requires_grad =", result.requires_grad)
        return result

    # Test decorator effects
    print("\nTesting @no_grad decorator:")
    output1 = inference_fn(x)
    
    print("\nTesting @enable_grad decorator:")
    output2 = training_fn(x)

**Application Scenarios for Gradient Context Management**

1. **Inference Phase**: Disable gradient tracking during model inference to improve performance and save memory
2. **Partial Computation**: Only enable gradient tracking for the parts that need it in complex calculations
3. **Nested Contexts**: Flexibly switch gradient tracking states at different code levels
4. **Function-Level Control**: Set a unified gradient tracking strategy for entire functions through decorators

Indexing Operations
-------------------

Riemann supports various tensor indexing operations for accessing array elements or slices. Here are the common indexing methods:

**1. Integer Indexing**

Integer indexing is used to access a single element at a specific position in the tensor. For multi-dimensional tensors, multiple integer indices can be used separated by commas.

.. code-block:: python

    import riemann as rm

    # Create tensor
    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("x:", x)

    # Integer indexing
    print("x[0, 0]:", x[0, 0])  # Get element at first row, first column
    print("x[1, 2]:", x[1, 2])  # Get element at second row, third column

**2. Negative Integer Indexing**

Negative integer indexing counts from the end of the tensor, where -1 represents the last element, -2 represents the second last element, and so on.

.. code-block:: python

    import riemann as rm

    # Create tensor
    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("x:", x)

    # Negative integer indexing
    print("x[-1, -1]:", x[-1, -1])  # Get element at last row, last column
    print("x[-2, -3]:", x[-2, -3])  # Get element at second last row, third last column

**3. Slice Indexing**

Slice indexing is used to access contiguous segments of the tensor, using colons (:) to represent ranges. The format is `start:end:step`, where start is the starting index, end is the ending index (exclusive), and step is the step size.

.. code-block:: python

    import riemann as rm

    # Create tensor
    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("x:", x)

    # Slice indexing
    print("x[:, 0]:", x[:, 0])  # Get first column of all rows
    print("x[0, :]:", x[0, :])  # Get all columns of first row
    print("x[1:, 1:]:", x[1:, 1:])  # Get sub-tensor starting from second row and second column
    print("x[::2, ::2]:", x[::2, ::2])  # Get elements at every other row and column

**4. Integer Array Indexing**

Integer array indexing is used to access elements at positions specified by an integer array, returning a tensor with the same shape as the index array.

.. code-block:: python

    import riemann as rm

    # Create tensor
    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("x:", x)

    # Integer array indexing
    indices = rm.tensor([0, 1, 2])
    print("x[indices, indices]:", x[indices, indices])  # Get diagonal elements

**5. Boolean Indexing**

Boolean indexing is used to access elements that satisfy conditions specified by a boolean array, returning a 1D tensor of elements that meet the conditions.

.. code-block:: python

    import riemann as rm

    # Create tensor
    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("x:", x)

    # Boolean indexing
    mask = x > 5
    print("mask:", mask)
    print("x[mask]:", x[mask])  # Get elements greater than 5

**6. Mixed Indexing**

Mixed indexing refers to using multiple indexing methods in the same indexing expression, such as using both integer indexing and slice indexing.

.. code-block:: python

    import riemann as rm

    # Create tensor
    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("x:", x)

    # Mixed indexing
    print("x[0, 1:]:", x[0, 1:])  # Get elements from second column onwards in first row
    print("x[1:, 0]:", x[1:, 0])  # Get elements from first column in second row and onwards

**7. Indexing-Related Functions**

Riemann provides several indexing-related functions for gathering or scattering data by indices:

.. list-table:: Indexing-Related Functions
  :widths: 15 45 40
  :header-rows: 1

  * - Function
    - Description
    - Example
  * - ``gather``
    - Gather data by indices
    - ``input.gather(dim, index)``
  * - ``scatter``
    - Scatter data by indices (non-in-place)
    - ``input.scatter(dim, index, src)``
  * - ``scatter_``
    - Scatter data by indices (in-place)
    - ``input.scatter_(dim, index, src)``
  * - ``scatter_add``
    - Scatter and accumulate data by indices (non-in-place)
    - ``input.scatter_add(dim, index, src)``
  * - ``scatter_add_``
    - Scatter and accumulate data by indices (in-place)
    - ``input.scatter_add_(dim, index, src)``
  * - ``setat``
    - Set values at specified indices (non-in-place)
    - ``rm.setat(input, indices, value)``
  * - ``setat_``
    - Set values at specified indices (in-place)
    - ``input.setat_(indices, value)``
  * - ``addat``
    - Add values at specified indices (non-in-place)
    - ``input.addat(indices, value)``
  * - ``addat_``
    - Add values at specified indices (in-place)
    - ``input.addat_(indices, value)``
  * - ``subat``
    - Subtract values at specified indices (non-in-place)
    - ``input.subat(indices, value)``
  * - ``subat_``
    - Subtract values at specified indices (in-place)
    - ``input.subat_(indices, value)``
  * - ``mulat``
    - Multiply values at specified indices (non-in-place)
    - ``input.mulat(indices, value)``
  * - ``mulat_``
    - Multiply values at specified indices (in-place)
    - ``input.mulat_(indices, value)``
  * - ``divat``
    - Divide values at specified indices (non-in-place)
    - ``input.divat(indices, value)``
  * - ``divat_``
    - Divide values at specified indices (in-place)
    - ``input.divat_(indices, value)``
  * - ``powat``
    - Raise values at specified indices to power (non-in-place)
    - ``input.powat(indices, value)``
  * - ``powat_``
    - Raise values at specified indices to power (in-place)
    - ``input.powat_(indices, value)``

**gather Function Example**

The ``gather`` function is used to gather data from the specified dimension of the input tensor by indices.

.. code-block:: python

    import riemann as rm

    # Create input tensor
    input = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("input:", input)

    # Define indices
    index = rm.tensor([[0, 1], [1, 2]])
    print("index:", index)

    # Gather data along dimension 0
    output = input.gather(0, index)
    print("gather along dim 0:", output)

    # Gather data along dimension 1
    output = input.gather(1, index)
    print("gather along dim 1:", output)

**scatter Function Example**

The ``scatter`` function is used to scatter data from the source tensor to the specified dimension of the target tensor by indices.

.. code-block:: python

    import riemann as rm

    # Create target tensor
    input = rm.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    print("input:", input)

    # Define indices and source tensor
    index = rm.tensor([[0, 1], [1, 2]])
    src = rm.tensor([[10, 20], [30, 40]])
    print("index:", index)
    print("src:", src)

    # Scatter data along dimension 0 (non-in-place)
    output = input.scatter(0, index, src)
    print("scatter along dim 0:", output)

    # Scatter data along dimension 1 (non-in-place)
    output = input.scatter(1, index, src)
    print("scatter along dim 1:", output)

**scatter_ Function Example**

``scatter_`` is the in-place version of ``scatter``, which directly modifies the input tensor.

.. code-block:: python

    import riemann as rm

    # Create target tensor
    input = rm.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    print("input:", input)

    # Define indices and source tensor
    index = rm.tensor([[0, 1], [1, 2]])
    src = rm.tensor([[10, 20], [30, 40]])
    print("index:", index)
    print("src:", src)

    # Scatter data along dimension 1 (in-place)
    input.scatter_(1, index, src)
    print("after scatter_ along dim 1:", input)

**scatter_add Function Example**

The ``scatter_add`` function is used to scatter and accumulate data from the source tensor to the specified dimension of the target tensor by indices.

.. code-block:: python

    import riemann as rm

    # Create target tensor
    input = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("input:", input)

    # Define indices and source tensor
    index = rm.tensor([[0, 1], [1, 2]])
    src = rm.tensor([[10, 20], [30, 40]])
    print("index:", index)
    print("src:", src)

    # Scatter and accumulate data along dimension 1 (non-in-place)
    output = input.scatter_add(1, index, src)
    print("scatter_add along dim 1:", output)

**scatter_add_ Function Example**

``scatter_add_`` is the in-place version of ``scatter_add``, which directly modifies the input tensor.

.. code-block:: python

    import riemann as rm

    # Create target tensor
    input = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("input:", input)

    # Define indices and source tensor
    index = rm.tensor([[0, 1], [1, 2]])
    src = rm.tensor([[10, 20], [30, 40]])
    print("index:", index)
    print("src:", src)

    # Scatter and accumulate data along dimension 1 (in-place)
    input.scatter_add_(1, index, src)
    print("after scatter_add_ along dim 1:", input)

**setat and setat_ Function Examples**

The ``setat`` function is used to set values at specified indices (non-in-place), while ``setat_`` is the in-place version.

.. code-block:: python

    import riemann as rm

    # Create input tensor
    input = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("input:", input)

    # 1. Using setat (non-in-place)
    print("\n1. Using setat (non-in-place):")
    # Set values at specified indices
    indices = (0, 1)  # Row 0, column 1
    value = 99
    output = input.setat(indices, value)
    print("setat result:", output)
    print("original input unchanged:", input)

    # 2. Using setat_ (in-place)
    print("\n2. Using setat_ (in-place):")
    # Set values at specified indices
    indices = (1, 2)  # Row 1, column 2
    value = 88
    input.setat_(indices, value)
    print("after setat_:", input)

    # 3. Using setat with multiple indices
    print("\n3. Using setat with multiple indices:")
    input = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    indices = [[0, 0], [2, 2]]  # (0,0) and (2,2)
    value = 100
    output = input.setat(indices, value)
    print("setat with multiple indices:", output)

**Indexing Operation Notes**

1. **Index Out of Bounds**：Using indices beyond the tensor range will result in an error.
2. **Memory Layout**：Different indexing methods may affect the memory layout of the returned tensor. Some indexing operations may return a view of the original tensor instead of a copy.
3. **Gradient Tracking**：For tensors requiring gradient tracking, some indexing operations may affect gradient calculation, especially in-place operations.
4. **Performance Considerations**：For large tensors, integer array indexing and boolean indexing may be slower than slice indexing because they create new tensor copies.
5. **gather/scatter Function Parameters**：
   - ``dim``：Specifies the dimension of the operation
   - ``index``：Specifies the index positions
   - ``src``：Specifies the source data (only for scatter-related functions)
   - For in-place operations (functions with underscores), ensure the input tensor is not a leaf node, otherwise an error will occur.

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

In-place Operations Functions and Operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Riemann supports the following in-place operations functions and operators:

.. list-table:: In-place Operations Functions and Operators
  :widths: 15 35 30 20
  :header-rows: 1

  * - Function
    - Description
    - Equivalent Operator
    - Example
  * - ``add_``
    - In-place addition
    - ``+=``
    - ``x.add_(y)`` or ``x += y``
  * - ``sub_``
    - In-place subtraction
    - ``-=``
    - ``x.sub_(y)`` or ``x -= y``
  * - ``mul_``
    - In-place multiplication
    - ``*=``
    - ``x.mul_(y)`` or ``x *= y``
  * - ``div_``
    - In-place division
    - ``/=``
    - ``x.div_(y)`` or ``x /= y``
  * - ``pow_``
    - In-place power operation
    - ``**=``
    - ``x.pow_(y)`` or ``x **= y``
  * - ``zero_``
    - In-place set all elements to 0
    - None
    - ``x.zero_()``
  * - ``fill_``
    - In-place fill all elements with a specified value
    - None
    - ``x.fill_(5)``
  * - ``copy_``
    - In-place copy data from another tensor
    - None
    - ``x.copy_(y)``
  * - ``detach_``
    - In-place detach gradients, making tensor no longer track gradients
    - None
    - ``x.detach_()``
  * - ``masked_fill_``
    - In-place fill values based on mask
    - None
    - ``x.masked_fill_(mask, value)``
  * - ``fill_diagonal_``
    - In-place fill diagonal elements
    - None
    - ``x.fill_diagonal_(value)``
  * - ``setat_``
    - In-place set value at specified position
    - ``x[index] = val``
    - ``x.setat_(index, val)``
  * - ``addat_``
    - In-place perform addition at specified position
    - ``x[index] += val``
    - ``x.addat_(index, val)``
  * - ``subat_``
    - In-place perform subtraction at specified position
    - ``x[index] -= val``
    - ``x.subat_(index, val)``
  * - ``mulat_``
    - In-place perform multiplication at specified position
    - ``x[index] *= val``
    - ``x.mulat_(index, val)``
  * - ``divat_``
    - In-place perform division at specified position
    - ``x[index] /= val``
    - ``x.divat_(index, val)``
  * - ``powat_``
    - In-place perform power operation at specified position
    - ``x[index] **= val``
    - ``x.powat_(index, val)``
  * - ``scatter_``
    - In-place scatter values according to index
    - None
    - ``x.scatter_(dim, index, src)``
  * - ``scatter_add_``
    - In-place scatter and accumulate values according to index
    - None
    - ``x.scatter_add_(dim, index, src)``

Notes on In-place Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using in-place operations, please note the following:

1. **Leaf Node Restrictions with Gradient Tracking**

   - For leaf node tensors with ``requires_grad=True``, in-place operations are not allowed
   - This is because in-place operations modify tensor values, which may compromise the correctness of gradient calculation

2. **Gradient Tracking for Right-hand Side Values**

   - The gradient of the right-hand side value (such as ``y`` in ``x += y``) can be tracked normally
   - This means that even when using in-place operations, the gradient calculation of the right-hand side tensor is not affected

3. **Gradient Tracking for In-place Operation Objects**

   - For non-leaf node tensors, the gradient tracking result of in-place operations is more complex
   - Especially when assigning values to arrays by index (such as ``x[index] = val``), gradient calculation may produce unexpected behavior
   - It is recommended to use in-place operations with caution in scenarios where gradient tracking is required

4. **Recommended Usage Scenarios**

   - In-place operations can be used on newly created tensors without gradient tracking attributes ( requires_grad=False )
   - For objects after ``clone()`` or ``copy()``, which are not leaf nodes with ``requires_grad=True``, so in-place operations can be used
   - In the inference phase where gradient calculation is not needed, using in-place operations can save memory

5. **Memory Optimization**

   - In-place operations do not create new tensor objects, so they can save memory
   - When processing large tensors, appropriate use of in-place operations can significantly reduce memory usage

6. **Chained Operations**

   - In-place operations return self , so they can be chained
   - For example: x.add_(y).mul_(z) is valid, while (x + y) * z is a chain of non-in-place operations

Gradient Tracking Example for In-place Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is an example of gradient tracking for in-place array assignment by index:

.. code-block:: python

    import riemann as rm

    # Create tensors with gradient tracking
    x0 = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = rm.tensor([10.0, 20.0, 30.0], requires_grad=True)

    # Print original values
    print("Original values:")
    print("x0:", x0)
    print("y:", y)

    # Clone x to make it no longer a leaf node, allowing in-place operations
    x = x0.clone()
    print("\nAfter x.clone(), x is no longer a leaf node")
    print("x.is_leaf:", x.is_leaf)

    # Perform in-place assignment by index
    print("\nPerforming in-place assignment x[1] = y[0]")
    x[1] = y[0]

    # Print values after assignment
    print("\nAfter assignment:")
    print("x0:", x0)
    print("x:", x)
    print("y:", y)

    # Calculate loss function
    loss = x.sum()
    print("\nLoss value:", loss)

    # Backward propagation to calculate gradients
    loss.backward()

    # Print gradients
    print("\nGradient tracking results:")
    print("x0.grad:", x0.grad)  # Gradient in left-hand side direction
    print("y.grad:", y.grad)  # Gradient in right-hand side direction

**Output Analysis**:

- After in-place assignment, the value of `x` becomes `[1.0, 10.0, 3.0]`, while the value of `y` remains unchanged
- Gradient calculation results show:
  - `x0.grad` is `[1.0, 0.0, 1.0]`, indicating that gradients are normally tracked except at the in-place assignment position
  - `y.grad` is `[1.0, 0.0, 0.0]`, indicating that gradients in the right-hand side direction are normally tracked

**Conclusion**:

- Gradient tracking for the right-hand side is not affected by in-place operations and works normally
- Gradient tracking for the left-hand side may exhibit abnormal behavior at the in-place assignment position
- For leaf nodes with gradient tracking, you must clone() them before performing in-place operations
- Therefore, in-place operations should be used with caution in scenarios requiring precise gradient calculation

Diagonalization Operations
--------------------------

Riemann provides various diagonalization operation functions for handling tensor diagonal elements, triangular parts, etc. Here are the commonly used diagonalization operation functions:

**diagonal Function**

Extracts diagonal elements from the input tensor.

.. code-block:: python

    import riemann as rm
    
    # Create example tensor
    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("Original tensor:")
    print(x)
    
    # Extract main diagonal
    print("\nMain diagonal:")
    print(rm.diagonal(x))  # tensor([1, 5, 9])
    
    # Extract offset diagonal
    print("\nOffset diagonal (offset=1):")
    print(rm.diagonal(x, offset=1))  # tensor([2, 6])
    
    # Extract negative offset diagonal
    print("\nNegative offset diagonal (offset=-1):")
    print(rm.diagonal(x, offset=-1))  # tensor([4, 8])

**diag Function**

Extracts diagonal elements from a tensor or creates a diagonal matrix from a 1D tensor.

.. code-block:: python

    import riemann as rm
    
    # Create diagonal matrix from 1D tensor
    v = rm.tensor([1, 2, 3])
    print("\nCreate diagonal matrix from 1D tensor:")
    print(rm.diag(v))
    # Output:
    # tensor([[1, 0, 0],
    #         [0, 2, 0],
    #         [0, 0, 3]])
    
    # Extract diagonal elements from 2D tensor
    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("\nExtract diagonal elements from 2D tensor:")
    print(rm.diag(x))  # tensor([1, 5, 9])

**batch_diag Function**

Generates batch diagonal matrices from batch 1D tensors.

.. code-block:: python

    import riemann as rm
    
    # Create batch 1D tensor
    batch_v = rm.tensor([[1, 2], [3, 4]])
    print("\nBatch 1D tensor:")
    print(batch_v)
    
    # Generate batch diagonal matrices
    print("\nBatch diagonal matrices:")
    print(rm.batch_diag(batch_v))
    # Output:
    # tensor([[[1, 0],
    #          [0, 2]],
    #         
    #         [[3, 0],
    #          [0, 4]]])

**fill_diagonal Function**

Fills the diagonal elements between specified dimensions with a given value, returns a new tensor.

.. code-block:: python

    import riemann as rm
    
    # Create example tensor
    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("\nOriginal tensor:")
    print(x)
    
    # Fill main diagonal with 0
    print("\nFill main diagonal with 0:")
    print(rm.fill_diagonal(x, 0))
    # Output:
    # tensor([[0, 2, 3],
    #         [4, 0, 6],
    #         [7, 8, 0]])
    
    # Fill offset diagonal with 5
    print("\nFill offset diagonal with 5 (offset=1):")
    print(rm.fill_diagonal(x, 5, offset=1))

**fill_diagonal_ Function**

In-place fills the diagonal elements between specified dimensions with a given value, returns the original tensor.

.. code-block:: python

    import riemann as rm
    
    # Create example tensor
    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("\nOriginal tensor:")
    print(x)
    
    # In-place fill main diagonal with 0
    print("\nIn-place fill main diagonal with 0:")
    result = rm.fill_diagonal_(x, 0)
    print(result)
    print("Is original tensor modified:")
    print(x)

**tril Function**

Extracts the lower triangular part of the tensor (including the diagonal).

.. code-block:: python

    import riemann as rm
    
    # Create example tensor
    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("\nOriginal tensor:")
    print(x)
    
    # Extract lower triangular part
    print("\nLower triangular part:")
    print(rm.tril(x))
    # Output:
    # tensor([[1, 0, 0],
    #         [4, 5, 0],
    #         [7, 8, 9]])
    
    # Extract offset lower triangular part
    print("\nOffset lower triangular part (diagonal=-1):")
    print(rm.tril(x, diagonal=-1))

**triu Function**

Extracts the upper triangular part of the tensor (including the diagonal).

.. code-block:: python

    import riemann as rm
    
    # Create example tensor
    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("\nOriginal tensor:")
    print(x)
    
    # Extract upper triangular part
    print("\nUpper triangular part:")
    print(rm.triu(x))
    # Output:
    # tensor([[1, 2, 3],
    #         [0, 5, 6],
    #         [0, 0, 9]])
    
    # Extract offset upper triangular part
    print("\nOffset upper triangular part (diagonal=1):")
    print(rm.triu(x, diagonal=1))

**Function Parameter Description**

.. list-table:: Diagonalization Operation Function Parameters
    :widths: 15 35 25 25
    :header-rows: 1

    * - Function Name
      - Main Parameters
      - Default Values
      - Description
    * - ``diagonal``
      - input, offset, dim1, dim2
      - offset=0, dim1=0, dim2=1
      - Extracts diagonal elements between specified dimensions
    * - ``diag``
      - input, offset
      - offset=0
      - Extracts diagonal elements or creates diagonal matrix
    * - ``batch_diag``
      - v
      - None
      - Generates batch diagonal matrices from batch 1D tensors
    * - ``fill_diagonal``
      - input, value, offset, dim1, dim2
      - offset=0, dim1=-2, dim2=-1
      - Fills diagonal elements, returns new tensor
    * - ``fill_diagonal_``
      - input, value, offset, dim1, dim2
      - offset=0, dim1=-2, dim2=-1
      - In-place fills diagonal elements, returns original tensor
    * - ``tril``
      - input_tensor, diagonal
      - diagonal=0
      - Extracts lower triangular part
    * - ``triu``
      - input_tensor, diagonal
      - diagonal=0
      - Extracts upper triangular part

**Notes**

1. ``diagonal`` Function:
   - Input tensor must be at least 2-dimensional
   - dim1 and dim2 cannot be the same
   - Supports negative indices (-1 represents the last dimension)

2. ``diag`` Function:
   - When input is 1D tensor, returns diagonal matrix
   - When input is 2D tensor, returns diagonal elements
   - Does not support 3D or higher-dimensional inputs

3. ``batch_diag`` Function:
   - The last dimension of the input tensor is the length of the diagonal elements
   - The output tensor shape is ``(*, n, n)``, where n is the size of the last dimension of the input tensor

4. ``fill_diagonal`` and ``fill_diagonal_`` Functions:
   - input tensor must be at least 2-dimensional
   - dim1 and dim2 cannot be the same
   - Support negative indices (default fills the diagonal of the last two dimensions)
   - ``fill_diagonal_`` is an in-place operation that modifies the original tensor

5. ``tril`` and ``triu`` Functions:
   - The diagonal parameter controls the offset of the diagonal
   - diagonal=0 represents the main diagonal
   - diagonal>0 represents above the main diagonal
   - diagonal<0 represents below the main diagonal

Saving and Loading Tensors
--------------------------

Riemann provides PyTorch-compatible serialization functionality for saving and loading tensors, parameters, module states, and training checkpoints. These features use ZIP format for serialization, ensuring cross-platform compatibility and efficient storage.

Basic Usage
~~~~~~~~~~~

Save and load a single tensor:

.. code-block:: python

    import riemann as rm
    
    # Create tensor
    x = rm.tensor([1, 2, 3])
    
    # Save to file
    rm.save(x, 'tensor.pt')
    
    # Load from file
    y = rm.load('tensor.pt')
    print(y)  # tensor([1, 2, 3])

Saving Multi-dimensional Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can save tensors of any shape and dimension:

.. code-block:: python

    # Create multi-dimensional tensors
    matrix = rm.randn(3, 4)
    tensor_3d = rm.randn(2, 3, 4)
    
    # Save multi-dimensional tensors
    rm.save(matrix, 'matrix.pt')
    rm.save(tensor_3d, 'tensor_3d.pt')
    
    # Load and verify
    loaded_matrix = rm.load('matrix.pt')
    loaded_tensor_3d = rm.load('tensor_3d.pt')
    
    print(f"Matrix shape: {loaded_matrix.shape}")  # (3, 4)
    print(f"3D tensor shape: {loaded_tensor_3d.shape}")  # (2, 3, 4)

Saving Model State Dict
~~~~~~~~~~~~~~~~~~~~~~~

When training deep learning models, you typically need to save the model's parameter state:

.. code-block:: python

    # Create a simple neural network
    model = rm.nn.Sequential(
        rm.nn.Linear(10, 64),
        rm.nn.ReLU(),
        rm.nn.Linear(64, 10)
    )
    
    # Save model state dict
    rm.save(model.state_dict(), 'model_weights.pt')
    
    # Create new model and load weights
    new_model = rm.nn.Sequential(
        rm.nn.Linear(10, 64),
        rm.nn.ReLU(),
        rm.nn.Linear(64, 10)
    )
    new_model.load_state_dict(rm.load('model_weights.pt'))

Saving Training Checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~~~~

During training, you can save complete checkpoints containing model state, optimizer state, and training progress:

.. code-block:: python

    # Assume model training is in progress
    model = rm.nn.Linear(10, 5)
    optimizer = rm.optim.Adam(model.parameters(), lr=0.001)
    
    # Train for several epochs
    for epoch in range(10):
        # ... training code ...
        pass
    
    # Save complete training checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': 0.5,  # Current loss value
    }
    rm.save(checkpoint, 'checkpoint.pt')
    
    # Resume training from checkpoint
    checkpoint = rm.load('checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']
    
    print(f"Resuming training from epoch {start_epoch}, last loss: {loss}")

Device Mapping for Loading
~~~~~~~~~~~~~~~~~~~~~~~~~~

When loading models between different devices (CPU/GPU), you can use the ``map_location`` parameter to specify the loading location:

.. code-block:: python

    # Load tensor saved on GPU to CPU
    # Assume training and saving on GPU
    # rm.save(gpu_tensor, 'gpu_tensor.pt')
    
    # Load on CPU
    cpu_tensor = rm.load('gpu_tensor.pt', map_location='cpu')
    
    # Use dictionary for device mapping
    map_location = {'cuda:0': 'cpu', 'cuda:1': 'cpu'}
    cpu_tensor = rm.load('model.pt', map_location=map_location)

Saving Multiple Tensors
~~~~~~~~~~~~~~~~~~~~~~~

You can save multiple tensors in a single file:

.. code-block:: python

    # Create multiple tensors
    tensor_a = rm.randn(3, 3)
    tensor_b = rm.randn(4, 4)
    tensor_c = rm.tensor([1, 2, 3, 4, 5])
    
    # Save as dictionary
    tensor_dict = {
        'weights': tensor_a,
        'biases': tensor_b,
        'labels': tensor_c
    }
    rm.save(tensor_dict, 'tensors.pt')
    
    # Load and access individual tensors
    loaded_dict = rm.load('tensors.pt')
    weights = loaded_dict['weights']
    biases = loaded_dict['biases']
    labels = loaded_dict['labels']

Important Notes
~~~~~~~~~~~~~~~

1. **File Format**: Riemann uses ZIP format for serialization, with file extensions typically being ``.pt`` or ``.pth``

2. **Compatibility**: The serialization format is compatible with PyTorch, and can load PyTorch-saved tensors (with some limitations)

3. **Device Information**: Saved tensors retain device information (CPU/GPU), which can be remapped using ``map_location`` during loading

4. **Gradient Information**: When saving tensors, gradient computation graph information (requires_grad attribute) is preserved

5. **Large File Handling**: For large models, it is recommended to use checkpoint mechanisms to save in chunks to avoid memory issues