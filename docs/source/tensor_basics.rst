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

Riemann provides three functions — ``tensor``, ``as_tensor``, and ``from_numpy`` — for creating tensors from Python array-like data (such as lists and tuples) or scalars, and from NumPy/CuPy arrays or scalars:

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

The key differences between the three functions lie in **memory sharing behavior** and **supported input types**:

.. list-table:: Comparison of tensor, as_tensor, and from_numpy
    :widths: 20 27 27 26
    :header-rows: 1

    * - Item
      - ``tensor``
      - ``as_tensor``
      - ``from_numpy``
    * - Supported input types
      - Python lists, tuples, scalars; NumPy/CuPy arrays or scalars
      - Python lists, tuples, scalars; NumPy/CuPy arrays or scalars; existing tensors
      - NumPy/CuPy arrays only
    * - Memory sharing with input
      - Never shares (always copies data)
      - Shares when possible (when dtype and device match)
      - Always shares
    * - dtype parameter
      - Supported
      - Supported
      - Not supported (keeps the array's dtype)
    * - device parameter
      - Supported
      - Supported
      - Not supported (device is determined by the array)
    * - requires_grad parameter
      - Supported
      - Supported
      - Not supported (always False)

**tensor: always copies data**

Tensors created by ``tensor`` never share memory with the input data; modifying the tensor does not affect the original data. This is the safest way to create tensors:

.. code-block:: python

    arr = np.array([1.0, 2.0, 3.0])
    x = rm.tensor(arr)
    x[0] = 0
    print(arr)  # [1. 2. 3.], the original array is unchanged

**as_tensor: shares memory when possible**

When the input data's dtype and device match the target, tensors created by ``as_tensor`` share memory with the input; data is only copied when a type or device conversion is needed. In addition, ``as_tensor`` supports creating a new tensor from an existing tensor:

.. code-block:: python

    # Shares memory when dtype and device match
    arr = np.array([1.0, 2.0, 3.0])
    x = rm.as_tensor(arr)
    x[0] = 0
    print(arr)  # [0. 2. 3.], the original array is modified

    # Create from an existing tensor (shares the underlying data)
    t = rm.tensor([1.0, 2.0, 3.0])
    y = rm.as_tensor(t)

**from_numpy: always shares memory**

``from_numpy`` only accepts NumPy/CuPy arrays. The created tensor always shares memory with the original array, which is useful when the tensor and the array need to stay in sync (a CuPy array automatically produces a CUDA tensor):

.. code-block:: python

    arr = np.array([1.0, 2.0, 3.0])
    x = rm.from_numpy(arr)
    x[0] = 0
    print(arr)  # [0. 2. 3.], the original array is modified

.. note::

    Sharing memory means that modifying the tensor also modifies the original array (and vice versa). If you want to prevent the original data from being accidentally modified, use the ``tensor`` function.

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
- **device**: Optional, specifies the device where the tensor is located, can be 'cpu', 'cuda', 'cuda:0', integer index, or Device object. If None, the device is resolved from the input data: tensors (TN) and cupy arrays keep their source device, while other data (numpy arrays, lists, scalars) use the default device.
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

- If data is a tensor (TN) or a cupy array, its source device is kept (aligned with PyTorch)
- Otherwise (numpy arrays, lists, scalars, etc.), the default device is used
- The default device is resolved as: device override inside a ``with Device(...)`` context > global default set by ``set_default_device`` > CPU

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

Riemann supports using context managers to temporarily switch the default device. Inside the with block, the context device becomes the return value of ``get_default_device()`` and takes precedence over the global default set by ``set_default_device()``. Tensors created in the block from device-less data (lists, scalars, numpy arrays) use the context device:

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

**Notes on context management**:

- The context override takes precedence over the global default device set by ``set_default_device()``
- The context only affects data without device semantics (lists, scalars, numpy arrays, etc.) and functions without a reference tensor (zeros, ones, etc.); ``tensor()`` and ``as_tensor()`` called with a tensor or a cupy array always keep its source device, regardless of the context
- For a CUDA device, entering the context also switches the current CuPy device, so native CuPy code inside the block runs on the same device

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
    * - ``from_numpy``
      - Creates a tensor from a NumPy or CuPy array
      - ``from_numpy(np_array)``

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

    # From NumPy array
    import numpy as np
    np_array = np.array([1, 2, 3])
    x = rm.from_numpy(np_array)

Default Parameter Behavior for Special Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When creating special tensors, if you don't specify `dtype` and `device` parameters, different default behaviors are used based on the function type:

**Tensor Creation Functions Without Reference Tensor** (e.g., zeros, ones, rand, etc.):

- When `dtype` and `device` parameters are not specified, the function behavior is consistent with the `tensor()` function
- Default data type is `float32`
- Default device follows ``get_default_device()``: the device override inside a ``with Device(...)`` context takes precedence, then the global default set by ``set_default_device``, then CPU

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

- When not specifying the `device` parameter, the default device is used (the device override inside a ``with Device(...)`` context takes precedence over the global default device)
- This is consistent with the default behavior of the `tensor()` function for device-less data

.. code-block:: python

    # Default uses the default device (context override > global default)
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
  * - ``is_leaf``
    - Whether tensor is a leaf node in computation graph
    - ``x.is_leaf`` → ``True``
  * - ``requires_grad``
    - Whether tensor requires gradient tracking
    - ``x.requires_grad`` → ``True``
  * - ``is_cuda``
    - Whether tensor is located on a CUDA device
    - ``x.is_cuda`` → ``False``
  * - ``is_cpu``
    - Whether tensor is located on the CPU
    - ``x.is_cpu`` → ``True``
  * - ``real``
    - Real part of a complex tensor; returns the tensor itself for real tensors
    - ``z.real`` → tensor
  * - ``imag``
    - Imaginary part of a complex tensor; raises RuntimeError for real tensors
    - ``z.imag`` → tensor
  * - ``data``
    - New tensor sharing the underlying data but detached from the computation graph
    - ``x.data`` → tensor sharing memory

**State Detection and Query Functions**

.. list-table:: State Detection and Query Functions
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
  * - ``is_contiguous``
    - Detect if tensor is stored contiguously
    - ``x.is_contiguous()`` → ``True``
  * - ``type``
    - Get or set tensor data type
    - ``x.type()`` → ``float32`` or ``x.type(rm.float64)``
  * - ``size``
    - Size of the tensor, or of the specified dimension
    - ``x.size(0)`` → ``2``
  * - ``numel``
    - Total number of tensor elements
    - ``x.numel()`` → ``6``

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

Interoperability with NumPy/CuPy/PyTorch Tensors
------------------------------------------------

Riemann's cross-framework conversion is built entirely on the array-exchange protocols of the Python ecosystem, rather than on a separate converter for each framework. Understanding these protocols tells you exactly when memory is shared and when data is copied.

**Protocols Behind the Conversion**

.. list-table:: Array-exchange protocols implemented by Riemann tensors
  :widths: 26 26 48
  :header-rows: 1

  * - Protocol
    - Triggered by
    - Purpose and behavior
  * - ``__array__``
    - ``np.asarray(x)``, ``np.array(x)``
    - NumPy array protocol, exposing the object as an ``ndarray``. For CPU tensors it returns the underlying array (zero copy); for CUDA tensors it fetches data back to host memory via ``cp.asnumpy``, so the result is always a copy.
  * - ``__cuda_array_interface__``
    - ``cp.asarray(x)``, ``cp.array(x)``, Numba and other GPU libraries
    - CUDA Array Interface (CAI). A dict describing the device pointer, shape, strides and dtype, letting CuPy take over Riemann's device memory without copying. CPU tensors do not provide this attribute.
  * - ``__dlpack__``
    - ``torch.from_dlpack(x)``, ``cp.from_dlpack(x)``, ``np.from_dlpack(x)``
    - DLPack cross-framework standard. A capsule carries "device + pointer + shape + strides + dtype" for both CPU and GPU, making it the universal zero-copy channel between torch, JAX, CuPy and NumPy.
  * - ``__dlpack_device__``
    - Companion query of ``__dlpack__``
    - Returns ``(device_type, device_id)``, e.g. ``(1, 0)`` for CPU and ``(2, 0)`` for the first GPU. Consumers use it to locate the data and pick the right backend.
  * - ``__buffer__``
    - ``memoryview(x)``, ``bytes(x)``, ``np.frombuffer(...)``
    - Native Python buffer protocol (PEP 688), exposing element format, shape, strides and writability. Buffers can only point at host memory, so a CUDA tensor yields a CPU copy. This protocol can only be implemented in pure Python on Python 3.12+; on 3.11 and earlier ``memoryview``/``bytes``/``np.frombuffer`` do not work, so use ``.array()`` or ``np.asarray()`` instead.

.. note::
   Riemann uses only two storage backends: NumPy (CPU) and CuPy (GPU). ``__array__`` serves the CPU
   direction, ``__cuda_array_interface__`` serves the GPU direction, and
   ``__dlpack__``/``__dlpack_device__``/``__buffer__`` provide the generic cross-framework channel.
   In addition, ``TN.array()`` returns the device-native array directly (``np.ndarray`` on CPU,
   ``cp.ndarray`` on CUDA, zero copy), while ``TN.numpy()`` follows PyTorch semantics: CPU tensors
   return the underlying array, CUDA tensors raise ``TypeError`` and require ``.cpu()`` first.

**Case 1: NumPy array → Riemann tensor**

.. code-block:: python

    import numpy as np
    import riemann as rm

    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    # tensor: always copies
    t1 = rm.tensor(arr, device='cpu')
    print(np.shares_memory(t1.array(), arr))          # False

    # as_tensor: shares memory when dtype and device match
    t2 = rm.as_tensor(arr, device='cpu')
    print(np.shares_memory(t2.array(), arr))          # True
    t2.array()[0] = 42.0
    print(arr[0])                                     # 42.0, visible both ways

    # as_tensor: copies when dtype differs
    t3 = rm.as_tensor(arr, dtype=rm.float64)
    print(np.shares_memory(t3.array(), arr))          # False

    # from_dlpack: zero copy (NumPy 2.x implements DLPack too)
    t4 = rm.from_dlpack(arr)
    print(np.shares_memory(t4.array(), arr))          # True

**Case 2: CuPy array → Riemann tensor**

.. code-block:: python

    import cupy as cp
    import riemann as rm

    garr = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)

    # tensor: copies, keeping the source device and dtype
    t = rm.tensor(garr)
    print(t.device, t.dtype)                          # cuda:0 float32
    print(t.array().data.ptr == garr.data.ptr)        # False

    # as_tensor: same device and dtype, shares device memory
    s = rm.as_tensor(garr)
    print(s.array().data.ptr == garr.data.ptr)        # True

    # from_dlpack: zero copy
    d = rm.from_dlpack(garr)
    print(d.array().data.ptr == garr.data.ptr)        # True

    # as_tensor: dtype differs, copies
    c = rm.as_tensor(garr, dtype=rm.float64)
    print(c.array().data.ptr == garr.data.ptr)        # False

.. note::
   Never use ``is`` to test device-memory sharing. Protocol round trips may return a new wrapper
   object, but the device memory is the same; compare ``data.ptr`` (CuPy) or ``data_ptr()`` (PyTorch).

**Case 3: PyTorch CPU tensor → Riemann tensor**

.. code-block:: python

    import numpy as np
    import riemann as rm
    import torch

    src = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

    # tensor: copies
    t = rm.tensor(src, device='cpu')
    print(np.shares_memory(t.array(), src.numpy()))   # False

    # as_tensor: same dtype, shares memory
    s = rm.as_tensor(src, device='cpu')
    print(np.shares_memory(s.array(), src.numpy()))   # True
    s.array()[0] = 42.0
    print(float(src[0]))                              # 42.0

    # from_dlpack: zero copy, dtype taken from the source
    d = rm.from_dlpack(src)
    print(d.dtype, np.shares_memory(d.array(), src.numpy()))   # float32 True

**Case 4: PyTorch CUDA tensor → Riemann tensor**

.. code-block:: python

    import riemann as rm
    import torch

    src_gpu = torch.tensor([1.0, 2.0], dtype=torch.float32, device='cuda:0')

    # tensor: copies, but keeps the source device
    t = rm.tensor(src_gpu)
    print(t.device)                                   # cuda:0
    print(t.array().data.ptr == src_gpu.data_ptr())   # False

    # as_tensor: shares device memory
    s = rm.as_tensor(src_gpu)
    print(s.array().data.ptr == src_gpu.data_ptr())   # True

    # from_dlpack: zero copy
    d = rm.from_dlpack(src_gpu)
    print(d.device, d.dtype)                          # cuda:0 float32
    print(d.array().data.ptr == src_gpu.data_ptr())   # True

.. note::
   When ``rm.from_dlpack`` receives CUDA data but CuPy is not installed or no CUDA device is
   visible, it raises ``RuntimeError`` naming the device the source data lives on.

**Case 5: Riemann tensor → NumPy array**

.. code-block:: python

    import numpy as np
    import riemann as rm

    t = rm.tensor([1.0, 2.0], dtype=rm.float32)       # CPU tensor

    # np.array: copies
    a1 = np.array(t)
    print(np.shares_memory(a1, t.array()))            # False

    # np.asarray: zero-copy view
    a2 = np.asarray(t)
    print(np.shares_memory(a2, t.array()))            # True
    a2[0] = 42.0
    print(t.array()[0])                               # 42.0, visible both ways

    # np.asarray with a different dtype: copies
    a3 = np.asarray(t, dtype=np.float64)
    print(np.shares_memory(a3, t.array()))            # False

    # CUDA tensor: __array__ brings data back to host, returning an independent copy
    g = rm.tensor([1.0, 2.0], dtype=rm.float32, device='cuda')
    a4 = np.asarray(g)
    print(a4.dtype, a4)                               # float32 [1. 2.]

**Case 6: Riemann tensor → CuPy array**

.. code-block:: python

    import cupy as cp
    import riemann as rm

    g = rm.tensor([1.0, 2.0], dtype=rm.float32, device='cuda')

    # cp.array: copies
    c1 = cp.array(g)
    print(c1.data.ptr == g.array().data.ptr)          # False

    # cp.asarray: zero copy
    c2 = cp.asarray(g)
    print(c2.data.ptr == g.array().data.ptr)          # True
    c2[0] = 42.0
    print(float(g.array()[0]))                        # 42.0, visible both ways

    # cp.asarray with a different dtype: copies
    c3 = cp.asarray(g, dtype=cp.float64)
    print(c3.data.ptr == g.array().data.ptr)          # False

    # cp.from_dlpack: zero copy
    c4 = cp.from_dlpack(g)
    print(c4.data.ptr == g.array().data.ptr)          # True

**Case 7: Riemann tensor → PyTorch tensor**

.. code-block:: python

    import numpy as np
    import riemann as rm
    import torch

    # CPU tensor
    t = rm.tensor([1.0, 2.0], dtype=rm.float32)

    p1 = torch.tensor(t)                              # copies
    print(np.shares_memory(p1.numpy(), t.array()))    # False

    p2 = torch.as_tensor(t)                           # same dtype, shares memory
    print(np.shares_memory(p2.numpy(), t.array()))    # True

    p3 = torch.from_dlpack(t)                         # zero copy
    print(np.shares_memory(p3.numpy(), t.array()))    # True

    # CUDA tensor
    g = rm.tensor([1.0, 2.0], dtype=rm.float32, device='cuda')

    p4 = torch.as_tensor(g)                           # shares device memory
    print(p4.data_ptr() == g.array().data.ptr)        # True

    p5 = torch.from_dlpack(g)                         # zero copy
    print(p5.device, p5.dtype)                        # cuda:0 torch.float32

.. note::
   ``torch.as_tensor`` and ``torch.from_dlpack`` return tensors that share memory with the Riemann
   tensor, so an in-place assignment on the PyTorch side (e.g. ``p2[0] = 42``) changes the Riemann
   tensor's data directly.

**Case 8: Buffer protocol (memoryview / bytes / np.frombuffer; Python 3.12+)**

.. note::
   PEP 688 only allows pure-Python objects to implement the buffer protocol from Python 3.12 on;
   on earlier versions use zero-copy entries such as ``np.asarray(t)`` or ``t.array()`` instead.

.. code-block:: python

    import numpy as np
    import riemann as rm

    t = rm.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=rm.float32)

    mv = memoryview(t)
    print(mv.format, mv.itemsize, mv.ndim, mv.shape, mv.readonly)
    # 'f' 4 2 (2, 2) False

    # A CPU tensor's buffer points at the tensor's memory, so writes take effect at once
    mv[0, 0] = 42.0
    print(t.array()[0, 0])                            # 42.0

    # bytes: raw bytes in C order
    print(len(bytes(t)), bytes(t) == t.array().tobytes())   # 16 True

    # np.frombuffer: build a zero-copy NumPy view over the buffer
    view = np.frombuffer(memoryview(t), dtype=np.float32)
    print(np.shares_memory(view, t.array()))          # True
    view[1] = 7.0
    print(t.array()[0, 1])                            # 7.0

    # Scalar tensor: zero-dimensional buffer
    s = rm.tensor(3.5)
    print(memoryview(s).ndim, memoryview(s).shape)    # 0 ()

    # CUDA tensor: buffers can only live in host memory, so this is a CPU copy
    g = rm.tensor([1.0, 2.0], dtype=rm.float32, device='cuda')
    gmv = memoryview(g)
    print(gmv.shape)                                  # (2,)
    gmv[0] = 9.0
    print(float(g.array()[0]))                        # 1.0, the write never reaches device memory

**Zero-Copy Cheat Sheet**

.. list-table:: External data → Riemann tensor
  :widths: 22 22 22 34
  :header-rows: 1

  * - Conversion
    - dtype/device match
    - dtype mismatch
    - Notes
  * - ``rm.tensor``
    - copy
    - copy (with dtype cast)
    - always copies; device and dtype follow the source
  * - ``rm.as_tensor``
    - shares memory
    - copy
    - sharing requires both dtype and device to match
  * - ``rm.from_dlpack``
    - shares memory
    - dtype cannot be specified
    - zero copy; dtype and device come from the source

.. list-table:: Riemann tensor → external data
  :widths: 20 26 26 28
  :header-rows: 1

  * - Target
    - Call
    - CPU tensor
    - CUDA tensor
  * - NumPy
    - ``np.asarray``
    - shares memory
    - copy (brought back to host)
  * - NumPy
    - ``np.array``
    - copy
    - copy
  * - CuPy
    - ``cp.asarray``
    - copy (moved to device)
    - shares device memory
  * - CuPy
    - ``cp.array``
    - copy
    - copy
  * - PyTorch
    - ``torch.as_tensor``
    - shares memory
    - shares device memory
  * - PyTorch
    - ``torch.from_dlpack``
    - shares memory
    - shares device memory
  * - PyTorch
    - ``torch.tensor``
    - copy
    - copy
  * - Buffer
    - ``memoryview`` / ``bytes``
    - shares memory (writable)
    - host copy (writes are not propagated)

Tensor Computations
-------------------

Basic Arithmetic
~~~~~~~~~~~~~~~~

Tensors support standard arithmetic operations, each with a corresponding functional form:

.. list-table:: Basic Arithmetic Operations
  :widths: 20 25 55
  :header-rows: 1

  * - Operator
    - Function
    - Description
  * - ``x + y``
    - ``rm.add(x, y)``
    - Element-wise addition
  * - ``x - y``
    - ``rm.sub(x, y)``
    - Element-wise subtraction
  * - ``x * y``
    - ``rm.mul(x, y)``
    - Element-wise multiplication
  * - ``x / y``
    - ``rm.div(x, y)``
    - Element-wise true division
  * - ``x ** y``
    - ``rm.pow(x, y)``
    - Element-wise power
  * - ``x // y``
    - ``rm.floor_divide(x, y)``
    - Element-wise floor division (no backward gradient, as in PyTorch)
  * - ``x % y``
    - ``rm.remainder(x, y)``
    - Element-wise remainder (the sign follows the divisor)
  * - ``x @ y``
    - ``rm.matmul(x, y)``
    - Matrix multiplication

Operators are thin wrappers around these module-level functions; ``rm.add`` / ``rm.sub``
additionally accept the keyword-only ``alpha`` parameter and compute
``x + alpha * y`` / ``x - alpha * y`` (passing ``alpha`` positionally raises ``TypeError``,
consistent with PyTorch).

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
    
    # Floor division and remainder
    z = x // y
    z = x % y
    
    # Addition with alpha, equivalent to x + 2 * y
    z = rm.add(x, y, alpha=2)
    
    # Matrix multiplication
    a = rm.tensor([[1, 2], [3, 4]])
    b = rm.tensor([[5, 6], [7, 8]])
    c = a @ b  # Matrix multiplication

Tensor Product Operations
~~~~~~~~~~~~~~~~~~~~~~~~~

Riemann supports various tensor product operations, each with different mathematical meanings and application scenarios:

**1. Dot Product (Inner Product)**

The dot product of two vectors is a scalar, calculated as:

.. math::

    \mathbf{a} \cdot \mathbf{b} = \sum_{i} a_i b_i

.. code-block:: python

    import riemann as rm
    
    a = rm.tensor([1, 2, 3])
    b = rm.tensor([4, 5, 6])
    
    # Using dot function
    result = rm.dot(a, b)  # tensor(32.)
    
    # Or using einsum
    result = rm.einsum('i,i->', a, b)  # tensor(32.)

**2. Outer Product**

The outer product of two vectors is a matrix:

.. math::

    (\mathbf{a} \otimes \mathbf{b})_{ij} = a_i b_j

.. code-block:: python

    a = rm.tensor([1, 2, 3])
    b = rm.tensor([4, 5])
    
    # Using outer function
    result = rm.outer(a, b)  # shape: (3, 2)
    
    # Or using einsum
    result = rm.einsum('i,j->ij', a, b)  # shape: (3, 2)

**3. Hadamard Product (Element-wise Multiplication)**

Element-wise product of two tensors with the same shape:

.. math::

    (\mathbf{A} \circ \mathbf{B})_{ij} = A_{ij} B_{ij}

.. code-block:: python

    A = rm.tensor([[1, 2], [3, 4]])
    B = rm.tensor([[5, 6], [7, 8]])
    
    # Using * operator
    C = A * B  # tensor([[5, 12], [21, 32]])
    
    # Or using einsum
    C = rm.einsum('ij,ij->ij', A, B)  # tensor([[5, 12], [21, 32]])

**4. Kronecker Product**

The Kronecker product of two tensors is a block matrix:

.. math::

    \mathbf{A} \otimes \mathbf{B} = \begin{bmatrix} a_{11}\mathbf{B} & a_{12}\mathbf{B} & \cdots \\ a_{21}\mathbf{B} & a_{22}\mathbf{B} & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix}

.. code-block:: python

    A = rm.tensor([[1, 2], [3, 4]])
    B = rm.tensor([[0, 5], [6, 7]])
    
    # Using kron function
    C = rm.kron(A, B)
    # tensor([[ 0,  5,  0, 10],
    #         [ 6,  7, 12, 14],
    #         [ 0, 15,  0, 20],
    #         [18, 21, 24, 28]])

**5. Matrix Multiplication**

Standard matrix multiplication:

.. math::

    (\mathbf{A} \mathbf{B})_{ik} = \sum_{j} A_{ij} B_{jk}

.. code-block:: python

    A = rm.tensor([[1, 2], [3, 4]])
    B = rm.tensor([[5, 6], [7, 8]])
    
    # Using @ operator
    C = A @ B  # tensor([[19, 22], [43, 50]])
    
    # Or using matmul function
    C = rm.matmul(A, B)
    
    # Or using einsum
    C = rm.einsum('ij,jk->ik', A, B)

**6. Cross Product**

The cross product of two 3D vectors is a vector perpendicular to both input vectors:

.. math::

    \mathbf{a} \times \mathbf{b} = \begin{bmatrix} a_2 b_3 - a_3 b_2 \\ a_3 b_1 - a_1 b_3 \\ a_1 b_2 - a_2 b_1 \end{bmatrix}

.. code-block:: python

    a = rm.tensor([1., 2., 3.])
    b = rm.tensor([4., 5., 6.])
    
    # Using cross function
    result = rm.cross(a, b)  # tensor([-3., 6., -3.])
    
    # Batch cross product
    a_batch = rm.tensor([[1., 2., 3.], [4., 5., 6.]])
    b_batch = rm.tensor([[4., 5., 6.], [1., 2., 3.]])
    result = rm.cross(a_batch, b_batch)  # shape: (2, 3)

**Product Operations Comparison Table**

.. list-table:: Tensor Product Operations Comparison
    :widths: 20 25 30 25
    :header-rows: 1

    * - Operation Type
      - Input Shapes
      - Output Shape
      - Function/Operator
    * - Dot Product
      - (n,), (n,)
      - ()
      - ``dot``, ``einsum('i,i->')``
    * - Outer Product
      - (m,), (n,)
      - (m, n)
      - ``outer``, ``einsum('i,j->ij')``
    * - Hadamard Product
      - (m, n), (m, n)
      - (m, n)
      - ``*``, ``einsum('ij,ij->ij')``
    * - Kronecker Product
      - (m, n), (p, q)
      - (m*p, n*q)
      - ``kron``
    * - Matrix Multiplication
      - (m, n), (n, p)
      - (m, p)
      - ``@``, ``matmul``, ``einsum('ij,jk->ik')``
    * - Cross Product
      - (3,), (3,) or (..., 3), (..., 3)
      - (3,) or (..., 3)
      - ``cross``

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
  * - ``pow``
    - Compute power of each element
    - ``rm.pow(x, y)``
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
  * - ``prod``
    - Compute product of tensor elements
    - ``rm.prod(x)``
  * - ``dot``
    - Compute dot product of two tensors
    - ``rm.dot(x, y)``

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

**sum Function**

**Description:** Computes the sum of tensor elements. Commonly used in loss computation, feature aggregation, and statistical summarization.

**Parameters:**

.. list-table::
  :widths: 15 20 65
  :header-rows: 1

  * - Parameter
    - Type
    - Description
  * - ``x``
    - TN
    - Input tensor
  * - ``dim``
    - int, tuple, None
    - Dimension along which to sum. None means sum over all elements (default), int means sum along the specified axis, tuple means sum along multiple axes
  * - ``keepdim``
    - bool
    - Whether to keep the original number of dimensions, default is False

**Returns:** Tensor containing the sum

**Example:**

.. code-block:: python

    import riemann as rm

    x = rm.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print("Sum of all elements:", rm.sum(x))
    # tensor(21.)

    print("Sum along axis 0:", rm.sum(x, dim=0))
    # tensor([5., 7., 9.])

    print("Sum along axis 1:", rm.sum(x, dim=1, keepdim=True))
    # tensor([[ 6.],
    #         [15.]])

**mean Function**

**Description:** Computes the mean of tensor elements. Commonly used in data normalization, feature statistics, and loss computation.

**Parameters:**

.. list-table::
  :widths: 15 20 65
  :header-rows: 1

  * - Parameter
    - Type
    - Description
  * - ``x``
    - TN
    - Input tensor
  * - ``dim``
    - int, tuple, None
    - Dimension along which to compute the mean. None means over all elements (default)
  * - ``keepdim``
    - bool
    - Whether to keep the original number of dimensions, default is False

**Returns:** Tensor containing the mean

**Example:**

.. code-block:: python

    import riemann as rm

    x = rm.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print("Mean of all elements:", rm.mean(x))
    # tensor(3.5)

    print("Mean along axis 0:", rm.mean(x, dim=0))
    # tensor([2.5, 3.5, 4.5])

**var Function**

**Description:** Computes the variance of tensor elements. Commonly used in data distribution analysis and normalization.

**Parameters:**

.. list-table::
  :widths: 15 20 65
  :header-rows: 1

  * - Parameter
    - Type
    - Description
  * - ``x``
    - TN
    - Input tensor
  * - ``dim``
    - int, tuple, None
    - Dimension along which to compute the variance. None means over all elements (default)
  * - ``unbiased``
    - bool
    - Whether to use unbiased estimation (Bessel's correction), default is True
  * - ``keepdim``
    - bool
    - Whether to keep the original number of dimensions, default is False

**Returns:** Tensor containing the variance

**Example:**

.. code-block:: python

    import riemann as rm

    x = rm.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print("Variance of all elements:", rm.var(x))
    # tensor(3.5)

    print("Variance along axis 0:", rm.var(x, dim=0))
    # tensor([4.5, 4.5, 4.5])

**max/min Functions**

**Description:** Computes the maximum or minimum of tensor elements. Commonly used in extreme value detection, feature selection, and data range analysis.

**Parameters:**

.. list-table::
  :widths: 15 20 65
  :header-rows: 1

  * - Parameter
    - Type
    - Description
  * - ``x``
    - TN
    - Input tensor
  * - ``dim``
    - int, tuple, None
    - Dimension along which to find the extreme value. None returns the global extreme value (default); specifying a dimension returns the extreme values and their indices along that dimension
  * - ``keepdim``
    - bool
    - Whether to keep the original number of dimensions, default is False

**Returns:** When dim is None, returns a scalar tensor; when dim is specified, returns a namedtuple containing ``values`` (extreme value tensor) and ``indices`` (index tensor)

**Example:**

.. code-block:: python

    import riemann as rm

    x = rm.tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])
    print("Maximum value:", rm.max(x))
    # tensor(6.)

    print("Minimum value:", rm.min(x))
    # tensor(1.)

    result = rm.max(x, dim=0)
    print("Max values along axis 0:", result.values)
    # tensor([4., 5., 6.])
    print("Max indices along axis 0:", result.indices)
    # tensor([1, 0, 1])

    result = rm.min(x, dim=1)
    print("Min values along axis 1:", result.values)
    # tensor([1., 2.])
    print("Min indices along axis 1:", result.indices)
    # tensor([0, 1])

**argmax/argmin Functions**

**Description:** Returns the index of the maximum or minimum value in the tensor. Commonly used in classification prediction and extreme value localization. argmax/argmin are tensor methods.

**Parameters:**

.. list-table::
  :widths: 15 20 65
  :header-rows: 1

  * - Parameter
    - Type
    - Description
  * - ``dim``
    - int, None
    - Dimension along which to find the extreme value index. None returns the global flattened index (default)
  * - ``keepdim``
    - bool
    - Whether to keep the original number of dimensions, default is False

**Returns:** Index tensor (int64 type)

**Example:**

.. code-block:: python

    import riemann as rm

    x = rm.tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])
    print("Global max index:", x.argmax())
    # tensor(5)

    print("Max indices along axis 0:", x.argmax(dim=0))
    # tensor([1, 0, 1])

    print("Min indices along axis 1:", x.argmin(dim=1))
    # tensor([0, 1])

**where Function**

**Description:** Selects elements based on a condition, or returns the indices of elements that satisfy the condition. Commonly used in conditional filtering, masking, and piecewise function implementation.

**Parameters:**

.. list-table::
  :widths: 15 30 55
  :header-rows: 1

  * - Parameter
    - Type
    - Description
  * - ``cond``
    - TN
    - Condition tensor (boolean type)
  * - ``x``
    - TN, int, float, None
    - Value selected when condition is True, default is None
  * - ``y``
    - TN, int, float, None
    - Value selected when condition is False, default is None

**Returns:** When both x and y are None, returns a tuple of index tensors (tuple of TN); otherwise returns a tensor with elements selected based on the condition (TN)

**Example:**

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

**clamp_ Function**

**Description:** Clamps each element of the tensor in-place to the range [min, max] (in-place operation, directly modifies tensor data). Commonly used in gradient clipping and data range limiting.

**Parameters:**

.. list-table::
  :widths: 15 30 55
  :header-rows: 1

  * - Parameter
    - Type
    - Description
  * - ``min``
    - float, TN, None
    - Lower bound, can be a scalar or tensor, None means no lower bound, default is None
  * - ``max``
    - float, TN, None
    - Upper bound, can be a scalar or tensor, None means no upper bound, default is None

**Returns:** Returns the tensor itself (self), with elements clamped in-place.

**Example:**

.. code-block:: python

    import riemann as rm

    x = rm.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    print("Original tensor:", x)
    # tensor([1., 2., 3., 4., 5.])

    x.clamp_(min=2.0, max=4.0)
    print("After in-place clamping:", x)
    # tensor([2., 2., 3., 4., 4.])

**clamp Function**

**Description:** Clamps each element of the tensor to the range [min, max] and returns a new tensor (non-in-place operation, does not modify the original tensor). Commonly used in gradient clipping and data range limiting.

**Parameters:**

.. list-table::
  :widths: 15 30 55
  :header-rows: 1

  * - Parameter
    - Type
    - Description
  * - ``x``
    - TN
    - Input tensor
  * - ``min``
    - float, TN, None
    - Lower bound, can be a scalar or tensor, None means no lower bound, default is None
  * - ``max``
    - float, TN, None
    - Upper bound, can be a scalar or tensor, None means no upper bound, default is None

**Returns:** Returns a new tensor with elements clamped to the [min, max] range. The original tensor is not modified.

**Example:**

.. code-block:: python

    import riemann as rm

    x = rm.tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])
    print("Original tensor:", x)
    # tensor([[1., 5., 3.],
    #         [4., 2., 6.]])

    result = rm.clamp(x, min=2.0, max=5.0)
    print("Clamped result:", result)
    # tensor([[2., 5., 3.],
    #         [4., 2., 5.]])

    print("Original tensor unchanged:", x)
    # tensor([[1., 5., 3.],
    #         [4., 2., 6.]])

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

Tensor Bitwise Operators
~~~~~~~~~~~~~~~~~~~~~~~~

Riemann supports various tensor bitwise operators for bitwise operations on integer tensors.
On boolean tensors, bitwise operations act in the True/False domain and are equivalent to the corresponding logical operations.
When the left operand is a scalar, the reflected operators (e.g. ``5 & x``, ``True | x``) also work, matching PyTorch behavior.

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


Tensor Logical Functions
~~~~~~~~~~~~~~~~~~~~~~~~

Logical functions evaluate elements as booleans (non-zero is True) and always return boolean tensors, independent of the input dtype.
They differ from bitwise operators: ``&`` operates on integer bits, while ``logical_and`` treats each element as a whole truth value.

.. list-table:: Tensor Logical Functions
  :widths: 25 45 30
  :header-rows: 1

  * - Function
    - Description
    - Example
  * - ``logical_and``
    - Logical AND
    - ``rm.logical_and(x, y)``
  * - ``logical_or``
    - Logical OR
    - ``rm.logical_or(x, y)``
  * - ``logical_xor``
    - Logical XOR
    - ``rm.logical_xor(x, y)``
  * - ``logical_not``
    - Logical NOT
    - ``rm.logical_not(x)``

**Logical Functions Example**

.. code-block:: python

    import riemann as rm

    # Integer tensors: non-zero treated as True
    x = rm.tensor([1, 0, 3, 0])
    y = rm.tensor([2, 0, 0, 4])

    # Logical operations return boolean tensors
    print(rm.logical_and(x, y))     # tensor([ True, False, False, False])
    print(rm.logical_or(x, y))      # tensor([ True, False,  True,  True])
    print(rm.logical_xor(x, y))     # tensor([False, False,  True,  True])
    print(rm.logical_not(x))        # tensor([False,  True, False,  True])

    # ``other`` can be a scalar, broadcastable like a tensor
    print(rm.logical_and(x, 2))     # tensor([ True, False,  True, False])

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
  * - ``split``
    - Splits tensor into chunks along specified dimension (by chunk size)
    - ``rm.split(x, 2, dim=0)`` or ``rm.split(x, [2, 3], dim=0)``
  * - ``tensor_split``
    - Splits tensor into chunks along specified dimension (by number of sections or indices)
    - ``rm.tensor_split(x, 3, dim=0)`` or ``rm.tensor_split(x, [2, 4], dim=0)``
  * - ``vsplit``
    - Vertically splits tensor (along dimension 0), splits tensor into multiple subtensors
    - ``rm.vsplit(x, 3)`` or ``rm.vsplit(x, [2, 4])``
  * - ``hsplit``
    - Horizontally splits tensor (along dimension 1), splits tensor into multiple subtensors
    - ``rm.hsplit(x, 2)`` or ``rm.hsplit(x, [1, 3])``
  * - ``dsplit``
    - Depth splits tensor (along dimension 2), splits 3D+ tensor into multiple subtensors
    - ``rm.dsplit(x, 3)`` or ``rm.dsplit(x, [2, 4])``
  * - ``dstack``
    - Depth stacks tensors along dimension 2 (1D/2D tensors will be reshaped first)
    - ``rm.dstack([x, y])``

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

Indexing operations fall into two categories: reading by index (``x[idx]``) and writing by index (the ``setat``/``scatter`` function families as well as the ``x[idx] = val`` syntax). This section covers "Reading by Index", "Writing by Index" and "Summary of Indexing Functions"; both reading and writing support automatic differentiation.

.. _index-read:

Reading by Index
~~~~~~~~~~~~~~~~

Read operations are implemented by ``TN.__getitem__`` and accept the same index expressions as NumPy.

**1. Integer and Negative Integer Indexing**

Integer indexing accesses the element at a specific position; for multi-dimensional tensors, multiple integer indices are separated by commas. Integer indexing reduces the number of dimensions by one (using integers for every dimension yields a 0-d scalar tensor). Negative indices count from the end, where ``-1`` denotes the last element.

.. code-block:: python

    import riemann as rm

    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    print(x[0, 0])    # tensor(1): a 0-d tensor
    print(x[1, 2])    # tensor(6)
    print(x[-1, -1])  # tensor(9): last row, last column
    print(x[-2, -3])  # tensor(4)
    print(x[0])       # tensor([1, 2, 3]): one dimension removed

**2. Slice Indexing**

Slices use the ``start:end:step`` form, where ``end`` is exclusive and ``step`` may be negative:

.. code-block:: python

    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    print(x[:, 0])      # tensor([1, 4, 7]): column 0 of every row
    print(x[1:, 1:])    # tensor([[5, 6], [8, 9]])
    print(x[::2, ::2])  # tensor([[1, 3], [7, 9]]): every other row and column
    print(x[::-1])      # rows in reverse order

**3. Integer Array Indexing**

Integer arrays (``TN`` or integer-typed ``numpy.ndarray``) follow NumPy's fancy indexing rules: multiple index arrays are combined by broadcasting, and the result shape is the broadcast shape of the index arrays.

.. code-block:: python

    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    idx = rm.tensor([0, 1, 2])
    print(x[idx, idx])            # tensor([1, 5, 9]): diagonal elements
    print(x[rm.tensor([2, 0])])   # tensor([[7, 8, 9], [1, 2, 3]]): reorder rows

**4. Boolean Indexing**

Boolean indexing uses a boolean tensor with the same shape as a mask and returns a 1-D tensor holding the elements that satisfy the condition:

.. code-block:: python

    x = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    mask = x > 5
    print(mask)
    print(x[mask])  # tensor([6, 7, 8, 9]): returns a 1-D tensor

**5. Mixed Indexing**

A single index expression may combine integers, slices, integer arrays, boolean masks and the ellipsis ``...``:

.. code-block:: python

    y = rm.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    print(y[0, 1:])      # tensor([[3, 4]]): second row of block 0
    print(y[..., 0])     # tensor([[1, 3], [5, 7]]): column 0 of every matrix
    print(y[0, [1, 0]])  # tensor([[3, 4], [1, 2]]): reorder rows inside block 0

**Views and Copies**

An indexing result is either a view of the original tensor (sharing the underlying data) or a copy (holding freshly copied data). The rules match NumPy:

.. list-table:: Index Forms and Result Type
  :widths: 26 32 25 17
  :header-rows: 1

  * - Index form
    - Example
    - Result shape
    - View/Copy
  * - Integer, slice, ellipsis (basic indexing)
    - ``x[0]``, ``x[1:3]``, ``x[..., 0]``
    - Dimensions indexed by integers are removed
    - View
  * - Integer array
    - ``x[[0, 2]]``, ``x[idx]``
    - Broadcast shape of the index arrays
    - Copy
  * - Boolean mask
    - ``x[mask]``
    - 1-D (number of matching elements)
    - Copy
  * - Mixed indexing
    - ``x[0, [1, 0]]``
    - Follows NumPy rules
    - Copy if integer arrays or boolean masks are involved, otherwise a view

The difference between a view and a copy is whether the underlying data is shared: when the original tensor is modified, a view changes accordingly while a copy does not.

.. code-block:: python

    x = rm.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])

    y = x[0:2]     # slice: view
    z = x[[0, 2]]  # integer array: copy

    x.setat_((0, 0), 99.)   # modify the original tensor in place

    print(y)  # tensor([[99., 2., 3.], [4., 5., 6.]]): the view follows
    print(z)  # tensor([[1., 2., 3.], [7., 8., 9.]]): the copy is unaffected

The internal flag ``_is_view`` on the result records whether it is a view (``True``) or a copy (``False``), and ``_base`` points to the original tensor that produced it.

Practical advice:

- Prefer slices (views) for read-only access to avoid the extra memory and time cost of copying large tensors;
- Use integer-array or boolean indexing, or call ``clone()``/``copy()`` explicitly, when you need independent data;
- Writing in place through a view also modifies the original tensor, so confirm that this is the intended behavior.

**Gathering Along a Dimension: gather**

``gather`` is the function form of "reading by index" and matches the semantics of ``torch.gather``: along the given dimension ``dim``, elements are selected position by position with an ``index`` of the same shape as the output.

Riemann provides two equivalent spellings:

.. code-block:: python

    riemann.gather(input, dim, index)   # module-level function
    input.gather(dim, index)            # TN method

**Parameters**

- ``input``: source tensor (``riemann.TN``) to read from;
- ``dim``: dimension to gather along, negative values are accepted (``-1`` means the last dimension);
- ``index``: index tensor (``riemann.TN``) whose number of dimensions must equal that of ``input``; dtypes other than int64 are converted to int64 automatically; the result has exactly the shape of ``index``;
- ``sparse_grad``: whether to produce a sparse gradient; only ``False`` is supported in the current version, ``True`` raises ``NotImplementedError``;
- ``out``: optional output tensor whose shape, dtype and device must match the result; only the module-level ``riemann.gather`` provides it, and automatic differentiation is not supported when ``out`` is used.

**Behavior**

For every position ``(i0, ..., i(n-1))`` in the output, the value is:

``out[i0, ..., i(n-1)] = input[i0, ..., index[i0, ..., i(n-1)], ..., i(n-1)]``

where the ``dim``-th subscript is given by the value of ``index`` at that position and all other subscripts coincide with the output position. This implies the following constraints:

- values of ``index`` along ``dim`` must lie in ``[0, input.shape[dim])``, otherwise ``IndexError`` is raised;
- for dimensions other than ``dim``, the shape of ``index`` must not exceed the corresponding size of ``input``, otherwise out-of-range indexing raises an error;
- the number of dimensions of ``index`` and ``input`` must be equal, otherwise ``ValueError`` is raised;
- the result is a copy and does not share memory with ``input``.

.. code-block:: python

    input = rm.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    index = rm.tensor([[0, 1], [1, 2]], dtype=rm.int64)

    print(input.gather(0, index))
    # tensor([[1, 5],
    #         [4, 8]])

    print(rm.gather(input, 1, index))
    # tensor([[1, 2],
    #         [5, 6]])

**Use cases**

- picking the predicted values of target classes in cross-entropy and negative log-likelihood losses;
- selecting or reordering elements by a predefined order during sequence decoding;
- assembling outputs from sampled or sorted indices.

**Gradients of gather**

``gather`` is fully differentiable: during backpropagation the upstream gradient is scattered back to the corresponding positions of ``input``; positions that were not selected receive a zero gradient, and a position selected multiple times accumulates gradients.

.. code-block:: python

    input = rm.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], requires_grad=True)
    index = rm.tensor([[0, 1], [1, 2]], dtype=rm.int64)

    out = rm.gather(input, 1, index)
    out.sum().backward()

    print(input.grad)
    # tensor([[1., 1., 0.],
    #         [0., 1., 1.],
    #         [0., 0., 0.]])

**More Selection Functions: index_select, masked_select and take**

Besides ``gather``, Riemann provides three specialized read functions that mirror the corresponding PyTorch interfaces. They are all implemented on top of ``x[idx]`` and therefore share the same indexing rules and support automatic differentiation.

.. list-table:: More Read Functions
  :widths: 18 42 40
  :header-rows: 1

  * - Function
    - Signature
    - Semantics
  * - ``index_select``
    - ``index_select(input, dim, index, *, out=None)``; ``x.index_select(dim, index)``
    - Selects along ``dim`` with a 1-D index; the result keeps the input dimensions and replaces ``dim`` with ``len(index)``
  * - ``masked_select``
    - ``masked_select(input, mask, *, out=None)``; ``x.masked_select(mask)``
    - Selects elements by a broadcastable boolean mask and returns a 1-D copy in row-major order
  * - ``take``
    - ``take(input, index)``; ``x.take(index)``
    - Selects by a flattened row-major index and returns a tensor shaped like ``index``

**index_select**

``index_select`` is equivalent to ``torch.index_select``. ``index`` must be a one-dimensional integer tensor (other integer dtypes are converted to int64 automatically) whose values lie in ``[0, input.shape[dim])``; out-of-range values raise ``IndexError``. The result has the same number of dimensions as ``input``, with the length of ``dim`` equal to ``len(index)``:

.. code-block:: python

    input = rm.tensor([[1, 2, 3], [4, 5, 6]])
    index = rm.tensor([1, 0], dtype=rm.int64)

    print(rm.index_select(input, 0, index))
    # tensor([[4, 5, 6],
    #         [1, 2, 3]])

    print(input.index_select(1, rm.tensor([2, 0], dtype=rm.int64)))
    # tensor([[3, 1],
    #         [6, 4]])

Gradients are scattered back to the selected positions, and a position selected multiple times accumulates gradients.

**masked_select**

``masked_select`` is equivalent to ``torch.masked_select``. ``mask`` must be a boolean tensor whose shape is broadcastable to the shape of ``input``; a non-boolean mask raises ``TypeError``. The result is always one-dimensional and holds the selected elements in row-major order; an all-``False`` mask yields an empty tensor:

.. code-block:: python

    input = rm.tensor([[1., -2., 3.], [-4., 5., -6.]])
    mask = input > 0

    print(rm.masked_select(input, mask))   # tensor([1., 3., 5.])

Gradients flow only to the selected positions; unselected positions receive a zero gradient.

**take**

``take`` is equivalent to ``torch.take``. It treats ``input`` as flattened row-major data and indexes it with ``index``, which may have any shape. Negative indices follow Python semantics (``-1`` selects the last element); values must lie in ``[-input.numel(), input.numel())``, otherwise ``IndexError`` is raised. A 0-d ``index`` returns a 0-d tensor:

.. code-block:: python

    input = rm.tensor([[1., 2.], [3., 4.]])

    print(rm.take(input, rm.tensor([[3, 0], [1, 2]], dtype=rm.int64)))
    # tensor([[4., 1.],
    #         [2., 3.]])

    print(rm.take(input, rm.tensor(2, dtype=rm.int64)))   # tensor(3.)

Gradients are routed through flatten + integer-array indexing, so repeated indices accumulate gradients on the original positions.

**Gradient Tracking for Index Reads**

The result of ``x[idx]`` can take part in automatic differentiation whether it is a view or a copy. During backpropagation the gradient is scattered back to the corresponding positions of the original tensor: positions that were not indexed receive a zero gradient, and positions indexed repeatedly accumulate gradients. Being a view or a copy only affects memory sharing, not the correctness of gradient propagation.

.. code-block:: python

    import riemann as rm

    # Integer array indexing: copy
    a = rm.tensor([1., 2., 3.], requires_grad=True)
    b = a[[0, 2]]
    (b * b).sum().backward()
    print(a.grad)   # tensor([2., 0., 6.]): only the selected positions have gradients

    # Repeated index: gradients accumulate
    a = rm.tensor([1., 2., 3.], requires_grad=True)
    b = a[[1, 1]]
    (b * b).sum().backward()
    print(a.grad)   # tensor([0., 8., 0.])

    # Slice indexing: view
    a = rm.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
    b = a[:, 1]
    (b * 2).sum().backward()
    print(a.grad)
    # tensor([[0., 2., 0.],
    #         [0., 2., 0.]])

    # Boolean mask: copy
    a = rm.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
    b = a[a > 3]
    (b * 10).sum().backward()
    print(a.grad)
    # tensor([[ 0.,  0.,  0.],
    #         [10., 10., 10.]])

Writing by Index
~~~~~~~~~~~~~~~~

Write operations are grouped by how the target positions are specified. Every function comes in a non-in-place form and an in-place form (suffixed with ``_``): the non-in-place version returns a new tensor and leaves the original untouched, whereas the in-place version modifies the tensor itself and returns ``self``.

- the ``setat`` family: positions are given by NumPy-style index expressions, matching the ``x[idx] = val`` syntax;
- the ``scatter`` family: positions are given by ``dim`` + ``index``, matching ``torch.scatter``/``torch.scatter_add`` semantics.

**1. The setat Family**

.. list-table:: Functions in the setat Family
  :widths: 18 30 16 36
  :header-rows: 1

  * - Function
    - Semantics
    - In-place version
    - Equivalent syntax
  * - ``setat``
    - Set the given positions to a value
    - ``setat_``
    - ``x[idx] = val``
  * - ``addat``
    - Add a value at the given positions
    - ``addat_``
    - ``x[idx] += val``
  * - ``subat``
    - Subtract a value at the given positions
    - ``subat_``
    - ``x[idx] -= val``
  * - ``mulat``
    - Multiply the given positions by a value
    - ``mulat_``
    - ``x[idx] *= val``
  * - ``divat``
    - Divide the given positions by a value
    - ``divat_``
    - ``x[idx] /= val``
  * - ``powat``
    - Raise the given positions to a power
    - ``powat_``
    - ``x[idx] **= val``

**Parameters**

- ``indices``: NumPy-style index expression supporting integers, slices, integer arrays, boolean masks and tuples thereof, with exactly the same rules as "Reading by Index";
- ``val``: right-hand value, which may be a scalar, a ``riemann.TN`` or an object convertible to ``TN``; its shape must match the selected region (NumPy broadcasting applies when possible).

**Behavior**

- ``setat`` follows overwrite semantics: when the same position is written multiple times, the last write wins;
- ``addat`` follows ``numpy.add.at`` semantics: repeated writes to the same position accumulate, which makes it the natural choice for duplicate indices; ``subat``/``mulat``/``divat``/``powat`` behave likewise, applying the corresponding operation to the target position each time.

.. code-block:: python

    import riemann as rm

    x = rm.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])

    # Single position: setat returns a new tensor, x itself is unchanged
    print(x.setat((0, 1), 99.))
    # tensor([[ 1., 99.,  3.],
    #         [ 4.,  5.,  6.],
    #         [ 7.,  8.,  9.]])

    # Slice region: set rows 0~1 and columns 1~2 to zero
    print(x.setat((slice(0, 2), slice(1, 3)), rm.zeros((2, 2))))
    # tensor([[1., 0., 0.],
    #         [4., 0., 0.],
    #         [7., 8., 9.]])

    # Duplicate indices: setat overwrites while addat accumulates
    b = rm.tensor([1., 2., 3.])
    print(b.setat([1, 1], rm.tensor([10., 20.])))  # tensor([ 1., 20.,  3.])
    print(b.addat([1, 1], rm.tensor([10., 20.])))  # tensor([ 1., 32.,  3.])

**Use cases**

- updating a small number of positions, such as masked assignment, padding or writing fixed constants;
- accumulating statistics with duplicate indices (``addat``), for example projecting several samples onto the same position;
- for tidier "write by category" scenarios, prefer the ``scatter`` family below.

**2. The scatter Family**

The ``scatter`` family describes the target positions by ``dim`` + ``index``, matching PyTorch's ``torch.Tensor.scatter``/``scatter_add`` and fitting batch writes by category or by position.

.. list-table:: Functions in the scatter Family
  :widths: 22 36 14 28
  :header-rows: 1

  * - Function
    - Semantics
    - In-place version
    - Duplicate indices
  * - ``scatter``
    - Write src (or value) along dim according to index
    - ``scatter_``
    - Last write wins
  * - ``scatter_add``
    - Add src along dim according to index
    - ``scatter_add_``
    - Accumulate

**Parameters**

- ``dim``: dimension to scatter along, negative values are accepted;
- ``index``: index tensor (``riemann.TN``) whose dtype is normalized to int64; **its number of dimensions must equal that of ``self``**; its shape determines the region actually written;
- ``src``: source data, either a scalar or a ``TN`` with the same shape as ``index``; its dtype must equal that of ``self``, otherwise ``RuntimeError`` is raised; mutually exclusive with ``value`` (``scatter_add``/``scatter_add_`` only accept ``src`` and have no ``value``);
- ``value``: a scalar (int/float/complex or a NumPy scalar; a 0-d ``TN`` is also accepted when passed as a keyword argument), mutually exclusive with ``src``; internally a source tensor of the same shape as ``index`` filled with ``value`` is created. A scalar may also be passed as the third positional argument, in which case it is treated as ``value``; the positional form only accepts int/float/complex and NumPy scalars;
- returns: ``scatter``/``scatter_add`` return a new tensor with the same shape as ``self``, while ``scatter_``/``scatter_add_`` return ``self``.

**Behavior**

Each position ``i`` of ``index`` corresponds to one write. The target position is determined by two parts: the subscript along ``dim`` is taken from ``index[i]``, while the subscripts of the remaining dimensions are the coordinates of ``i`` itself, that is:

``self[..., index[i] at dim, ...] op= src[i]``

- values of ``index`` along ``dim`` must lie in ``[0, self.shape[dim])``, otherwise ``IndexError`` is raised;
- the shape of ``index`` may be smaller than that of ``self``, in which case only the region covered by ``index`` is written and the rest keeps its original values;
- ``scatter`` overwrites, so multiple writes to the same target let the last write win, whereas ``scatter_add`` accumulates;
- internally the implementation converts indirect indices into direct indices and then calls ``setat``/``addat``, so all rules of the ``setat`` family apply to ``scatter`` as well.

.. code-block:: python

    import riemann as rm

    base = rm.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    index = rm.tensor([[0, 1], [1, 2]], dtype=rm.int64)
    src = rm.tensor([[10., 20.], [30., 40.]])

    # dim=0: the row number comes from index, the column number is the current coordinate
    print(base.scatter(0, index, src))
    # tensor([[10.,  0.,  0.],
    #         [30., 20.,  0.],
    #         [ 0., 40.,  0.]])

    # dim=1: the column number comes from index, the row number is the current coordinate
    print(base.scatter(1, index, src))
    # tensor([[10., 20.,  0.],
    #         [ 0., 30., 40.],
    #         [ 0.,  0.,  0.]])

    # Use the value parameter instead of src
    print(base.scatter(1, index, value=5.))
    # tensor([[5., 5., 0.],
    #         [0., 5., 5.],
    #         [0., 0., 0.]])

    # In-place version: modifies base directly
    base.scatter_(1, index, src)
    print(base)
    # tensor([[10., 20.,  0.],
    #         [ 0., 30., 40.],
    #         [ 0.,  0.,  0.]])

.. code-block:: python

    # scatter_add: accumulates on duplicate indices, useful for statistics/aggregation
    base = rm.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    print(base.scatter_add(1, index, src))
    # tensor([[11., 22.,  3.],
    #         [ 4., 35., 46.],
    #         [ 7.,  8.,  9.]])

**Use cases**

- building one-hot encodings or writing labels and weights back by class index;
- rebuilding vocabularies/embeddings by writing batch vectors back into a global table;
- message aggregation in graph neural networks and attention pooling, where ``scatter_add`` sums several sources targeting the same position.

**3. In-place Assignment Syntax and Limits**

``x[idx] = val``, ``x[idx] += val`` and similar syntax are equivalent to the corresponding ``setat_``/``addat_`` series and count as in-place writes, which come with two limits:

- in-place writes on a leaf tensor with ``requires_grad=True`` raise ``RuntimeError`` because they would corrupt the computation graph; call ``clone()`` first to obtain a non-leaf tensor;
- in-place writes do not create new graph nodes, so switch to the non-in-place versions (``setat``/``scatter`` and so on) when exact gradients are required afterwards.

.. code-block:: python

    x0 = rm.tensor([1., 2., 3.], requires_grad=True)
    x = x0.clone()      # non-leaf tensor, in-place writes are allowed

    x[1] = 10.          # equivalent to x.setat_(1, 10.)
    print(x)            # tensor([ 1., 10.,  3.], requires_grad=True)
    print(x0)           # tensor([1., 2., 3.], requires_grad=True): x0 is untouched

**Gradient Tracking for Index Writes**

The non-in-place writers (``setat``, ``addat``, ``subat``, ``mulat``, ``divat``, ``powat``, ``scatter``, ``scatter_add``) all support automatic differentiation; they behave like "read + compute + write back", so gradients flow to both the written tensor and the right-hand value:

- for the **written tensor**: positions overwritten by ``setat``/``scatter`` no longer contribute to the forward result, hence their gradients are zero while untouched positions keep the upstream gradient; ``addat``/``scatter_add`` keep the original values in the computation, so every position keeps its gradient;
- for the **right-hand value** (``val`` or ``src``): each written element receives the upstream gradient of its position; with ``addat``/``scatter_add`` repeated writes to the same position accumulate gradients.

Note the duplicate-index subtlety of ``setat``: the forward pass keeps only the last written value, while the backward pass sends gradients back to every duplicated right-hand element, so the two are inconsistent. Use ``addat``/``scatter_add`` whenever duplicate indices are possible.

.. code-block:: python

    import riemann as rm

    # setat: overwritten positions get zero gradient, the value gets one gradient per write
    base = rm.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    val = rm.tensor([10., 20.], requires_grad=True)
    out = base.setat(([0, 1], [0, 1]), val)   # writes (0,0) and (1,1)
    (out * 2).sum().backward()
    print(base.grad)
    # tensor([[0., 2.],
    #         [2., 0.]])
    print(val.grad)   # tensor([2., 2.])

.. code-block:: python

    # scatter: gradients of the value flow back element by element, overwritten positions of the base get zero
    base = rm.zeros((2, 3), requires_grad=True)
    index = rm.tensor([[0, 1], [1, 0]], dtype=rm.int64)
    src = rm.tensor([[1., 2.], [3., 4.]], requires_grad=True)

    out = base.scatter(1, index, src)
    out.sum().backward()
    print(base.grad)
    # tensor([[0., 0., 1.],
    #         [0., 0., 1.]])
    print(src.grad)
    # tensor([[1., 1.],
    #         [1., 1.]])

.. code-block:: python

    # scatter_add: original values still take part in the computation, so every position keeps its gradient
    base = rm.ones((2, 3), requires_grad=True)
    index = rm.tensor([[0, 0], [0, 0]], dtype=rm.int64)   # duplicate indices
    src = rm.tensor([[1., 2.], [3., 4.]], requires_grad=True)

    out = base.scatter_add(1, index, src)
    out.sum().backward()
    print(base.grad)
    # tensor([[1., 1., 1.],
    #         [1., 1., 1.]])
    print(src.grad)
    # tensor([[1., 1.],
    #         [1., 1.]])

.. code-block:: python

    # Gradients also work with the in-place x[idx] = val syntax
    x0 = rm.tensor([1., 2., 3.], requires_grad=True)
    y = rm.tensor([5., 6.], requires_grad=True)

    x = x0.clone()
    x[[0, 2]] = y
    x.sum().backward()

    print(x0.grad)  # tensor([0., 1., 0.]): overwritten positions get no gradient
    print(y.grad)   # tensor([1., 1.])

Summary of Indexing Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The table below summarizes Riemann's indexing functions. The "In-place" column lists the corresponding in-place version and the "Gradients" column states whether the non-in-place version supports automatic differentiation.

.. list-table:: Summary of Indexing Functions
  :widths: 16 34 14 36
  :header-rows: 1

  * - Function
    - Signature
    - In-place version
    - Notes and gradients
  * - ``gather``
    - ``gather(input, dim, index, *, sparse_grad=False, out=None)``; ``x.gather(dim, index)``
    - None
    - Reads along a dimension and returns a copy shaped like index; gradients supported
  * - ``index_select``
    - ``index_select(input, dim, index, *, out=None)``; ``x.index_select(dim, index)``
    - None
    - Selects along a dimension with a 1-D index; gradients supported
  * - ``masked_select``
    - ``masked_select(input, mask, *, out=None)``; ``x.masked_select(mask)``
    - None
    - Selects elements by a broadcastable boolean mask and returns a 1-D copy; gradients supported
  * - ``take``
    - ``take(input, index)``; ``x.take(index)``
    - None
    - Selects by a flattened row-major index, returns a tensor shaped like index; gradients supported
  * - ``index_put_``
    - ``x.index_put_(indices, values, accumulate=False)``
    - None (in-place only)
    - Writes values by an index tuple, overwriting or accumulating; gradients supported for values
  * - ``scatter``
    - ``x.scatter(dim, index, src=None, *, value=None)``
    - ``scatter_``
    - Overwrites along a dimension; gradients supported (non-in-place)
  * - ``scatter_add``
    - ``x.scatter_add(dim, index, src)``
    - ``scatter_add_``
    - Adds along a dimension, accumulating duplicate indices; gradients supported (non-in-place)
  * - ``setat``
    - ``x.setat(indices, val)``
    - ``setat_``
    - Overwrites by NumPy index expression; gradients supported (non-in-place)
  * - ``addat``
    - ``x.addat(indices, val)``
    - ``addat_``
    - Adds by NumPy index expression, accumulating duplicate indices; gradients supported (non-in-place)
  * - ``subat``
    - ``x.subat(indices, val)``
    - ``subat_``
    - Subtracts by NumPy index expression; gradients supported (non-in-place)
  * - ``mulat``
    - ``x.mulat(indices, val)``
    - ``mulat_``
    - Multiplies by NumPy index expression; gradients supported (non-in-place)
  * - ``divat``
    - ``x.divat(indices, val)``
    - ``divat_``
    - Divides by NumPy index expression; gradients supported (non-in-place)
  * - ``powat``
    - ``x.powat(indices, val)``
    - ``powat_``
    - Raises positions to a power by NumPy index expression; gradients supported (non-in-place)

**Related Functions**

The following functions also produce "indices" but belong to extremum, sorting and conditional operations; see the corresponding subsections of the Tensor Operations chapter:

- ``argmax``/``argmin``: indices of the minimum and maximum values;
- ``argsort``: indices that sort the tensor;
- ``topk``: the top-k values together with their indices;
- ``nonzero``: indices of the non-zero elements;
- ``where``: selects elements by a condition, and returns the index tuple of matching elements when the condition is None.

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
  * - ``floor_divide_``
    - In-place floor division
    - ``//=``
    - ``x.floor_divide_(y)`` or ``x //= y``
  * - ``remainder_``
    - In-place remainder
    - ``%=``
    - ``x.remainder_(y)`` or ``x %= y``
  * - ``neg_``
    - In-place negation (boolean tensors are not supported; unsigned integers wrap around)
    - None
    - ``x.neg_()``
  * - ``abs_``
    - In-place absolute value (boolean and complex tensors are not supported)
    - None
    - ``x.abs_()``
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
  * - ``index_put_``
    - In-place write values according to an index tuple
    - None
    - ``x.index_put_(indices, values, accumulate=False)``

.. note::

   ``neg_()`` / ``abs_()`` are unary in-place operations: boolean tensors are not supported
   (raises ``NotImplementedError``); ``abs_()`` does not support complex tensors
   (raises ``RuntimeError``, consistent with PyTorch); ``neg_()`` wraps unsigned integers
   modulo the bit width. Both modify the data in place, return the tensor itself, and can be chained.

.. note::

   ``index_put_`` takes an index tuple such as ``(rows, cols)`` or ``(mask,)`` and follows ``torch.Tensor.index_put_``: with ``accumulate=False`` it overwrites (``setat_`` semantics, the last write wins), and with ``accumulate=True`` it accumulates duplicate indices (``addat_`` semantics). The dtype of ``values`` must match the target tensor, otherwise ``RuntimeError`` is raised. It only exists in the in-place form; use ``setat``/``addat`` when a non-in-place result is required.

.. code-block:: python

    x = rm.tensor([[1., 2.], [3., 4.]])
    rows = rm.tensor([0, 1], dtype=rm.int64)
    cols = rm.tensor([1, 0], dtype=rm.int64)

    # Overwrite the selected positions
    print(x.clone().index_put_((rows, cols), rm.tensor([10., 20.])))
    # tensor([[ 1., 10.],
    #         [20.,  4.]])

    # Accumulate a broadcast scalar onto the selected positions
    print(x.clone().index_put_((rows, cols), rm.tensor(5.), accumulate=True))
    # tensor([[1., 7.],
    #         [8., 4.]])

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

Einstein Summation Convention (einsum)
--------------------------------------

The Einstein summation convention is a concise notation for tensor operations in mathematics and physics. Riemann's ``einsum`` function leverages this convention to provide a unified, elegant, and powerful way to express various tensor operations.

**Core Concept**

The core rule of the Einstein summation convention is: **when the same index appears twice in a term, it indicates summation over that index**.

For example, matrix multiplication :math:`C_{ik} = \sum_{j} A_{ij} B_{jk}` can be written as ``ij,jk->ik``.

**Basic Syntax**

.. code-block:: python

    # Matrix multiplication
    C = rm.einsum('ij,jk->ik', A, B)

    # Batch matrix multiplication
    C = rm.einsum('bij,bjk->bik', A, B)

    # Using ellipsis to support arbitrary batch dimensions
    C = rm.einsum('...ij,...jk->...ik', A, B)

    # Matrix trace
    trace = rm.einsum('ii->', A)

    # Diagonal extraction
    diag = rm.einsum('ii->i', A)

**Operations einsum Can Replace**

einsum can uniformly express various tensor operations:

.. list-table:: einsum Computation Scenarios
    :widths: 25 35 40
    :header-rows: 1

    * - Operation Type
      - einsum Equation
      - Description
    * - Matrix Multiplication
      - ``ij,jk->ik``
      - Standard matrix multiplication
    * - Batch Matrix Multiplication
      - ``...ij,...jk->...ik``
      - Supports arbitrary batch dimensions
    * - Matrix Trace
      - ``ii->``
      - Sum of diagonal elements
    * - Diagonal Extraction
      - ``ii->i``
      - Extract diagonal elements as vector
    * - Matrix Transpose
      - ``ij->ji``
      - Swap rows and columns
    * - Vector Dot Product
      - ``i,i->``
      - Vector inner product
    * - Vector Outer Product
      - ``i,j->ij``
      - Generate rank-1 matrix
    * - Hadamard Product
      - ``ij,ij->ij``
      - Element-wise multiplication
    * - Frobenius Inner Product
      - ``ij,ij->``
      - Matrix inner product
    * - Sum All Elements
      - ``ij->``
      - Sum of all elements
    * - Sum Along Rows/Columns
      - ``ij->i`` / ``ij->j``
      - Sum along specified dimension
    * - Multi-Operand Chain
      - ``ij,jk,kl->il``
      - Sequential matrix multiplication

**Usage Examples**

.. code-block:: python

    import riemann as rm

    # Matrix multiplication
    A = rm.tensor([[1, 2], [3, 4]])
    B = rm.tensor([[5, 6], [7, 8]])
    C = rm.einsum('ij,jk->ik', A, B)

    # Batch matrix multiplication
    batch_A = rm.randn(2, 3, 4)
    batch_B = rm.randn(2, 4, 5)
    batch_C = rm.einsum('bij,bjk->bik', batch_A, batch_B)

    # Trace operation
    trace = rm.einsum('ii->', A)

    # Vector operations
    a = rm.tensor([1, 2, 3])
    b = rm.tensor([4, 5, 6])
    dot = rm.einsum('i,i->', a, b)  # Dot product
    outer = rm.einsum('i,j->ij', a, b)  # Outer product

**Detailed Documentation**

For complete documentation on einsum, including detailed syntax rules, all computation scenarios, and more examples, please refer to :doc:`einsum_chapter`.