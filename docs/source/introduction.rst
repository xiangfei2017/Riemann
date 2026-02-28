Introduction
============

Riemann is a lightweight automatic differentiation library and neural network programming framework that supports automatic gradient tracking for scalars/vectors/tensors, provides common components for building neural networks, has PyTorch-compatible interfaces, and is designed for neural network-related learning, education, and research purposes.

Key Features
------------

Tensor Operations

- Provides tensor creation functions (tensor, zeros, ones, randn, normal, etc.) with support for complex tensors
- Supports basic mathematical operations (addition, subtraction, multiplication, division, exponentiation, elementary functions like exponential, logarithmic, trigonometric, hyperbolic functions, statistical functions like sum, mean, variance, standard deviation, etc.)
- Supports vector and matrix operations (batch matrix multiplication, vector dot product, matrix determinant, matrix inverse, matrix factorization, etc.)
- Supports tensor shape reshaping, dimension expansion/reduction, indexing and slicing, element gathering/scattering, concatenation/splitting, etc.
- Supports tensor serialization/deserialization for easy model training and deployment

Automatic Differentiation

- **backward method**: Triggers backpropagation to compute gradients
- **grad function**: Computes gradients of functions with respect to inputs
- **track_grad Decorator and Function Class**: Support custom gradient tracking functions
- **Jacobian and Hessian Matrices**: Supports Jacobian matrix computation for multi-input multi-output functions, and Hessian matrix computation for multi-input functions

Linear Algebra Module

- Provides matrix factorization and backward gradient tracking (SVD, PLU, QR, etc.)
- Supports calculation of matrix inverse, pseudo-inverse, determinant, eigenvalues/eigenvectors
- Matrix norm and condition number computation
- Supports linear equation solving and least squares solving

Neural Network Modules

- Basic layers (Linear, Dropout, BatchNorm, LayerNorm, Embedding, etc.)
- Activation functions (ReLU, Sigmoid, Softmax, etc.)
- Loss functions (MSE, CrossEntropy, etc.)
- Convolution and pooling (Conv1d/2d/3d, MaxPool1d/2d/3d, AvgPool1d/2d/3d, etc.)
- Transformer (MultiheadAttention, TransformerEncoder, TransformerDecoder, Transformer, etc.)
- Optimizers (SGD, Adam, Adagrad, LBFGS, etc.)
- Network module containers (Sequential, ModuleList, ModuleDict, etc.)

Computer Vision Module

- Dataset classes:
  - **MNIST**: Handwritten digit recognition dataset
  - **CIFAR10**: 10-class color image dataset

- Image transformations:
  - **Basic transformations**: ToTensor, ToPILImage, Normalize
  - **Geometric transformations**: Resize, CenterCrop, RandomResizedCrop, FiveCrop, TenCrop, Pad
  - **Random transformations**: RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomGrayscale
  - **Color transformations**: ColorJitter, Grayscale
  - **Composed transformations**: Compose, Lambda

CUDA/GPU Support

- Provides GPU acceleration, supporting tensor and model migration between CPU and GPU
- Optimized GPU computing performance

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
    from riemann.vision.datasets import MNIST, CIFAR10
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

PyTorch Interface Compatibility
-------------------------------

Riemann library is designed with PyTorch interface compatibility in mind. Functions and classes with the same name maintain consistent interfaces, making it easy for PyTorch users to get started quickly:

- **Tensor Operations**: Supports tensor operation functions and methods with the same names as PyTorch, such as `tensor()`, `grad()`, `backward()`, etc.
- **Neural Network Components**: Layers, activation functions, and loss functions in the `nn` module maintain interface compatibility with PyTorch
- **Optimizers**: Optimizers in the `optim` module (such as SGD, Adam, etc.) maintain consistent interfaces with PyTorch
- **Automatic Differentiation Mechanism**: `requires_grad` and backpropagation mechanisms are similar to PyTorch
- **Computer Vision**: Datasets and transforms in the `vision` module maintain interface compatibility with torchvision

This design allows users familiar with PyTorch to easily migrate to the Riemann library for development and research work.