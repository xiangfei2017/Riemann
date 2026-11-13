Introduction
============

What is Riemann?
----------------

**Riemann** is a lightweight automatic differentiation library and neural network programming framework, designed for deep learning-related learning, education, and research.

What Can Riemann Do?
~~~~~~~~~~~~~~~~~~~~

- **Tensor Computation**: Support mathematical operations for 0 to multi-dimensional tensors, complex tensors, and CPU/GPU heterogeneous computing
- **Automatic Differentiation**: Automatic gradient tracking based on backpropagation algorithm, support for custom gradient functions
- **Neural Networks**: Common components for building neural networks (layers, activation functions, loss functions, optimizers, etc.)
- **Computer Vision**: Image dataset loading and image transformation functions
- **Linear Algebra**: Matrix decomposition, eigenvalue computation, linear equation solving, etc.

Riemann's core value lies in its **simplicity** and **learnability** — clear code structure makes it easy to understand the internal workings of deep learning frameworks, making it an ideal tool for learning and researching automatic differentiation and neural network implementations.

Key Features
------------

Tensor Operations
~~~~~~~~~~~~~~~~~

- Tensor creation functions (tensor, zeros, ones, randn, normal, etc., supporting complex tensors)
- Basic mathematical operations (addition, subtraction, multiplication, division, power operations, elementary functions like exponential, logarithmic, trigonometric, hyperbolic functions, statistical functions like sum, mean, variance, standard deviation)
- Vector and matrix operations (batch matrix multiplication, vector dot product, matrix determinant, matrix inverse, matrix decomposition)
- Tensor shape reshaping, dimension expansion/reduction, indexing and slicing, element gathering/scattering, concatenation/splitting
- Tensor serialization/deserialization for easy model training and deployment

Automatic Differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~

- **backward method**: Triggers backpropagation to compute gradients
- **grad function**: Computes gradients of functions with respect to inputs
- **func module**: torch.func-style functional autodiff API — curried ``grad``, ``jvp``, ``vjp``, ``jacrev``, ``jacfwd``, ``hessian``, ``vmap``, ``functional_call``, suitable for chained higher-order derivatives
- **track_grad decorator**: Use ``@track_grad`` decorator to customize gradient tracking functions (Riemann-specific)
- **Function class**: Customize forward and backward propagation by inheriting from Function class (Riemann-specific)
- **Jacobian and Hessian Matrices**: Supports Jacobian matrix computation for multi-input multi-output functions, and Hessian matrix computation for multi-input functions

Linear Algebra Module
~~~~~~~~~~~~~~~~~~~~~

- Matrix decomposition and backward gradient tracking (SVD, PLU, QR, etc.)
- Matrix inverse, pseudo-inverse, determinant, eigenvalues/eigenvectors
- Matrix norms and condition number computation
- Linear equation solving and least squares solving

Neural Network Modules
~~~~~~~~~~~~~~~~~~~~~~

- Basic layers (Linear, Dropout, BatchNorm, LayerNorm, Embedding, various normalization layers)
- Activation functions (ReLU, Sigmoid, Softmax, Tanh, GELU, SiLU, etc.)
- Loss functions (MSE, CrossEntropy, BCE, L1Loss, NLLLoss, etc.)
- Convolution and pooling (Conv1d/2d/3d, MaxPool1d/2d/3d, AvgPool1d/2d/3d, AdaptivePool, etc.)
- Transformer (MultiheadAttention, TransformerEncoder, TransformerDecoder, Transformer, etc.)
- KAN Networks (KAN, KANLinear etc.)
- Optimizers (SGD, Adam, Adagrad, AdamW, RMSprop, LBFGS, etc.)
- Learning rate schedulers (StepLR, ExponentialLR, CosineAnnealingLR, etc.)
- Network module containers (Sequential, ModuleList, ModuleDict, etc.)

Computer Vision Module
~~~~~~~~~~~~~~~~~~~~~~

- Dataset classes:

  - **MNIST**: Handwritten digit recognition dataset
  - **CIFAR10**: 10-class color image dataset
  - **ImageFolder**: Load image dataset from folder (organized by class subfolders)
  - **DatasetFolder**: Generic folder dataset base class

- Image transformations (40+ transforms):

  - **Type Conversion**: ToTensor, PILToTensor, ToPILImage, ConvertImageDtype
  - **Geometric Transforms**: Resize, CenterCrop, RandomCrop, RandomResizedCrop, FiveCrop, TenCrop, Pad
  - **Flip and Rotation**: RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomAffine, RandomPerspective
  - **Color Transforms**: ColorJitter, Grayscale, RandomGrayscale, Invert, Posterize, Solarize, Equalize, AutoContrast, Sharpness, Brightness, Contrast, Saturation, Hue
  - **Data Augmentation**: AutoAugment, RandAugment, TrivialAugmentWide, RandomErasing
  - **Other Transforms**: Normalize, GaussianBlur, Lambda, SanitizeBoundingBox

CUDA/GPU Support
~~~~~~~~~~~~~~~~

- GPU acceleration, supporting tensor and model migration between CPU and GPU
- Optimized GPU computing performance
- Support for Windows and Linux platforms (macOS CPU-only mode)

Riemann Package Module Structure
--------------------------------

.. code-block:: text

    riemann                  # Main package
    ├── autograd             # Automatic differentiation module
    │   └── functional       # Automatic differentiation functional interface
    ├── core                 # Core tensor subpackage
    │   ├── tensordef        # Tensor definition (TN class), dynamic graph and autograd
    │   ├── tensorfuncs      # Functional tensor operations (one-way dependency on tensordef)
    │   ├── dtype            # Data type definitions
    │   ├── gradmode         # Gradient mode control
    │   └── serialization    # Object saving and loading
    ├── func                 # torch.func-style functional autodiff API
    ├── linalg               # Linear algebra module
    ├── nn                   # Neural network module
    │   └── functional       # Neural network functions
    ├── optim                # Optimizer module
    │   └── lr_scheduler     # Learning rate scheduler module
    ├── utils                # Utility functions module
    │   └── data             # Data processing tools
    ├── vision               # Computer vision module
    │   ├── datasets         # Dataset classes
    │   └── transforms       # Image transformation operations
    └── cuda                 # CUDA/GPU support

Module Import Examples
----------------------

**Import the entire riemann module:**

.. code-block:: python

    import riemann as r

    # Use tensor creation functions
    t = r.tensor([1.0, 2.0, 3.0])

    # Use automatic differentiation functionality
    x = r.tensor([1.0, 2.0], requires_grad=True)
    y = x ** 2
    y.sum().backward()
    print(x.grad)  # Output: [2. 4.]

**Import required functions and classes by module tree:**

.. code-block:: python

    # Import tensor-related functionality
    from riemann import tensor, zeros, ones, randn

    # Import automatic differentiation functionality
    from riemann.autograd import grad, backward
    from riemann.autograd.functional import jacobian, hessian

    # Import functional autodiff API (torch.func-style, curried, supports higher-order)
    from riemann import func
    from riemann.func import grad as fgrad, jvp, vjp, jacrev, jacfwd, hessian, vmap, functional_call

    # Import linear algebra functionality
    from riemann import linalg
    from riemann.linalg import svd, det, inv

    # Import neural network components
    from riemann.nn import Linear, Conv2d, ReLU, CrossEntropyLoss
    from riemann.nn.functional import relu, cross_entropy

    # Import optimizers
    from riemann.optim import SGD, Adam, Adagrad

    # Import computer vision functionality
    from riemann.vision.datasets import MNIST, CIFAR10, ImageFolder
    from riemann.vision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip

    # Import CUDA support
    from riemann import cuda
    from riemann.cuda import is_available, Device

Application Scenarios
---------------------

- **Deep Learning Research**: Custom model and algorithm development
- **Scientific Computing**: Gradient computation for complex mathematical models, and optimization problem solving
- **Computer Vision**: Image classification, object detection, and other vision tasks
- **Education and Teaching**: Learning automatic differentiation and deep learning principles

PyTorch Compatibility
---------------------

Riemann is designed with PyTorch interface compatibility in mind. Functions and classes with the same name maintain consistent interfaces, making it easy for PyTorch users to get started quickly.

Compatible Features
~~~~~~~~~~~~~~~~~~~

- **Tensor Operations**: Supports tensor operation functions and methods with the same names as PyTorch
- **Neural Network Components**: Layers, activation functions, and loss functions in the ``nn`` module maintain interface compatibility with PyTorch
- **Optimizers**: Optimizers in the ``optim`` module (such as SGD, Adam, etc.) maintain consistent interfaces with PyTorch
- **Automatic Differentiation Mechanism**: ``requires_grad`` and backpropagation mechanisms are similar to PyTorch
- **torch.func Compatibility**: Top-level ``func`` module mirrors ``torch.func`` API — curried ``grad``, ``jacrev``/``jacfwd``, ``jvp``/``vjp``, ``hessian``, ``vmap``, ``functional_call``
- **Computer Vision**: Datasets and transforms in the ``vision`` module maintain interface compatibility with torchvision
- **Cross-framework interoperability**: Riemann tensors can be converted to and from NumPy arrays, CuPy arrays and other tensor frameworks such as PyTorch, or share data with them, through ``r.tensor()``, ``r.as_tensor()``, ``r.from_dlpack()``, ``TN.array()`` and ``TN.numpy()``, built on the ``__array__``, ``__cuda_array_interface__``, ``__dlpack__`` and ``__buffer__`` protocols (``__buffer__`` requires Python 3.12+). Conversion is zero-copy when device and dtype match

PyTorch Features Not Supported by Riemann
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As a lightweight framework, Riemann **does not support** the following advanced PyTorch features:

- **Distributed Training**: Does not support DataParallel, DistributedDataParallel, and other multi-GPU distributed training
- **JIT Compilation**: Does not support TorchScript compilation and optimization
- **Quantization**: Does not support model quantization (INT8, etc.)
- **ONNX Export**: Does not support exporting to ONNX format
- **Mobile Deployment**: Does not support TorchMobile, Core ML, and other mobile deployment
- **Advanced Optimizers**: Some advanced optimizers (such as Adamax, SparseAdam) are not supported
- **Advanced Dynamic Graph Features**: Such as certain complex control flows and dynamic shape operations

Riemann-Specific Features
~~~~~~~~~~~~~~~~~~~~~~~~~

Riemann provides some features that PyTorch does not have:

- **KAN Network Support**: Kolmogorov-Arnold Networks (KAN) implementation, including KAN, KANLinear modules
- **Custom Gradient Decorator** (``@track_grad``): Quickly add automatic differentiation support to functions using decorators, without defining a complete Function class
- **New functions**: ``sumall`` (add multiple tensors or non-tensors and return the total sum), ``isum`` (intelligent sum function that automatically selects the appropriate summation method based on parameters), ``d()`` (tensor method for chaining variables to compute higher-order mixed partial derivatives), etc.
- **Concise Code Structure**: Less code volume, clearer structure, easier for learning and research

Installation Guide
------------------

Riemann has **not** been published on PyPI, so ``pip install riemann`` does **not** work. Please install from source.

**Python version**: Riemann requires Python >= 3.9 and supports Python 3.9 ~ 3.14.

Dependencies
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 34 50

   * - Category
     - Dependencies
     - Notes
   * - Base
     - numpy>=1.20.0, pillow>=8.0.0, tqdm>=4.0.0
     - Required to run Riemann
   * - Test
     - pytest>=7.0.0, torch>=2.0.0, torchvision>=0.15.0
     - Only needed to run the test suite
   * - CUDA
     - cupy-cuda12x (default), cupy-cuda13x, cupy-cuda11x, cupy-cuda10x
     - Only needed for NVIDIA GPU acceleration

Installing from Source
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/xiangfei2017/Riemann.git
    cd Riemann

    # Base install (-e is editable mode: no reinstall after code changes)
    pip install -e .

    # Optional: test dependencies
    pip install -e .[tests]

    # Optional: CUDA dependencies (CUDA 12.x by default; cuda13 / cuda11 / cuda10 also available)
    pip install -e .[cuda]

Installing Dependencies Manually
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If pip/conda fails due to network or dependency resolution issues, install packages one by one:

.. code-block:: bash

    # Base dependencies
    pip install numpy pillow tqdm
    # Test dependencies
    pip install pytest torch torchvision
    # CUDA dependencies (choose by the version shown by nvcc --version)
    pip install cupy-cuda12x

- On a system-wide environment the OS package manager also works (e.g. ``apt install python3-numpy``, ``brew install numpy``); **a virtual environment can only use pip**;
- Install torch separately with the command from the `PyTorch website <https://pytorch.org/get-started/locally/>`_;
- Conda users can create the env with ``conda create -n riemann python=3.10``, but CuPy is still best installed via pip.

Then finish the Riemann install with ``pip install -e . --no-deps``.

CUDA Dependencies and Platform Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CUDA dependencies are not installed automatically. Install them explicitly with ``pip install -e .[cuda]`` and make sure a matching NVIDIA driver is installed.

.. list-table::
   :header-rows: 1
   :widths: 18 18 18 46

   * - Platform
     - Architecture
     - CUDA Support
     - Notes
   * - Linux
     - x86_64/AMD64
     - ✅ Supported
     - Use ``pip install cupy-cudaXX``
   * - Windows
     - x86_64/AMD64
     - ✅ Supported
     - Use ``pip install cupy-cudaXX``
   * - macOS
     - x86_64/ARM64
     - ❌ Not supported
     - No NVIDIA driver; CPU mode is used automatically
   * - Linux (ARM64)
     - aarch64/arm64
     - ⚠️ Source compilation required
     - NVIDIA Jetson, etc. require CuPy compiled from source
   * - Windows (ARM64)
     - aarch64/arm64
     - ❌ Not supported
     - CUDA is not supported yet

Installing the CUDA Driver
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Run ``nvcc --version`` to check the installed CUDA Toolkit version (``nvidia-smi`` shows the highest version supported by the driver, which may differ from the installed one).
2. Download and install the matching driver/toolkit from the `CUDA Toolkit Archive <https://developer.nvidia.com/cuda-toolkit-archive>`_.
3. Reboot and run ``nvcc --version`` again to confirm.

Verifying Installation
~~~~~~~~~~~~~~~~~~~~~~

After installation, you can verify with the following code:

.. code-block:: python

    import riemann as r
    print("CUDA Available:", r.cuda.is_available())
    print("Using Device:", r.device('cuda' if r.cuda.is_available() else 'cpu'))

If CUDA is installed successfully, it will display ``CUDA Available: True``, otherwise it will display ``CUDA Available: False`` and automatically use CPU mode.
