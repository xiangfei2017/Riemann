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
- **Unique Features**: ``sumall`` function (add multiple tensors or non-tensors), ``isum`` function (intelligent sum)

Automatic Differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~

- **backward method**: Triggers backpropagation to compute gradients
- **grad function**: Computes gradients of functions with respect to inputs
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
- **Scientific Computing**: Gradient computation for complex mathematical models
- **Optimization Problem Solving**: Gradient descent and Adam optimization algorithms
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
- **Computer Vision**: Datasets and transforms in the ``vision`` module maintain interface compatibility with torchvision

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

- **Custom Gradient Decorator** (``@track_grad``): Quickly add automatic differentiation support to functions using decorators, without defining a complete Function class
- **Advanced Computation Graph Building**: More flexible computation graph building and management mechanisms
- **sumall function**: Add multiple tensors or non-tensors and return the total sum
- **isum function**: Intelligent sum function that automatically selects the appropriate summation method based on parameters
- **Concise Code Structure**: Less code volume, clearer structure, easier for learning and research

Installation Guide
------------------

Riemann installation includes the following parts:

1. **Core Package**: Riemann main library, including core functions such as tensor computation, automatic differentiation, and neural networks
2. **Core Dependencies**: NumPy, Pillow, tqdm, and other required dependencies
3. **CUDA Dependencies** (optional): CuPy library for GPU acceleration
4. **Test Dependencies** (optional): pytest and other testing frameworks

Installing with Conda (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Conda better manages complex dependencies, especially CUDA-related packages:

.. code-block:: bash

    # Create a new conda environment
    conda create -n riemann python=3.10
    conda activate riemann

    # Install core dependencies
    conda install numpy pillow tqdm

    # Install Riemann from source (choose GitHub or Gitee mirror)
    # GitHub source (international access)
    git clone https://github.com/xiangfei2017/Riemann.git
    # Or Gitee source (faster for China users)
    # git clone https://gitee.com/xfcode2021/Riemann.git
    cd Riemann
    pip install -e .

    # Install test dependencies (optional)
    pip install -e .[tests]

    # Install CUDA dependencies (optional)
    pip install -e .[cuda]

Installing with pip
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Get Riemann library source (choose GitHub or Gitee mirror)
    # GitHub source (international access)
    git clone https://github.com/xiangfei2017/Riemann.git
    # Or Gitee source (faster for China users)
    # git clone https://gitee.com/xfcode2021/Riemann.git
    cd Riemann
    
    # Install package and its core dependencies in development mode
    pip install -e .

    # Install test dependencies
    pip install -e .[tests]

CUDA Support Installation Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::

    To enable CUDA acceleration in Riemann, the following **three conditions** must be met, all are required:

    1. **NVIDIA GPU Hardware**: The computer must be equipped with an NVIDIA graphics card
    2. **CUDA Driver**: NVIDIA CUDA driver compatible with the GPU must be installed
    3. **CuPy Library**: CuPy library matching the CUDA version must be installed

    Installing CuPy alone does not enable CUDA support in Riemann. Hardware and drivers must be properly installed first.

**CUDA Installation Steps:**

1. **Check GPU Hardware**
   Confirm that the computer is equipped with an NVIDIA graphics card that supports CUDA.

2. **Install CUDA Driver**

   - **Windows/Linux**: Visit `NVIDIA website <https://developer.nvidia.com/cuda-toolkit-archive>`_ to download and install the appropriate CUDA Toolkit version
   - Verification after installation: Run ``nvcc --version`` to check CUDA version
   - Note: ``nvidia-smi`` shows the highest CUDA version supported by the driver, while ``nvcc --version`` shows the actually installed version

3. **Install CuPy Library**

   Select the appropriate CuPy package based on your installed CUDA version:

   .. code-block:: bash

       # Install CUDA dependencies (auto-detect CUDA version and install matching CuPy)
       pip install -e .[cuda]

       # Or install specific CUDA version dependencies
       pip install -e .[cuda13]  # CUDA 13.x
       pip install -e .[cuda12]  # CUDA 12.x
       pip install -e .[cuda11]  # CUDA 11.x
       pip install -e .[cuda10]  # CUDA 10.x (Linux only)

**Version Compatibility:**

- CuPy version must match the CUDA Toolkit version
- Different CUDA versions are not compatible; ensure you select the correct version
- CUDA 11.x or 12.x is recommended for better compatibility

Dependency Notes
~~~~~~~~~~~~~~~~

**Core Dependencies** (automatically installed):

- **numpy>=1.20.0**: Core numerical computing library
- **pillow>=8.0.0**: Image processing functionality
- **tqdm>=4.0.0**: Progress bar display

**CUDA Dependencies** (require explicit installation, and CUDA driver must be installed first):

- **cupy-cuda13x**: For CUDA 13.x
- **cupy-cuda12x**: For CUDA 12.x
- **cupy-cuda11x**: For CUDA 11.x
- **cupy-cuda10x**: For CUDA 10.x (Linux only)

Platform Compatibility
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Platform
     - Architecture
     - CUDA Support
     - Installation Method
   * - Linux
     - x86_64/AMD64
     - ✅ Supported
     - ``pip install -e .[cuda]``
   * - Windows
     - x86_64/AMD64
     - ✅ Supported
     - ``pip install -e .[cuda]``
   * - macOS
     - x86_64/ARM64
     - ❌ Not Supported
     - No NVIDIA GPU driver, use CPU mode
   * - Linux (ARM64)
     - aarch64/arm64
     - ⚠️ Source compilation required
     - NVIDIA Jetson, etc. require CuPy compilation from source

Verifying Installation
~~~~~~~~~~~~~~~~~~~~~~

After installation, you can verify with the following code:

.. code-block:: python

    import riemann as r
    print("CUDA Available:", r.cuda.is_available())
    print("Using Device:", r.device('cuda' if r.cuda.is_available() else 'cpu'))

If CUDA is installed successfully, it will display ``CUDA Available: True``, otherwise it will display ``CUDA Available: False`` and automatically use CPU mode.
