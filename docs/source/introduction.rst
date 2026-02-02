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
- Convolution and pooling layers (Conv1d/2d/3d, MaxPool1d/2d/3d, AvgPool1d/2d/3d, etc.)
- Loss functions (MSE, CrossEntropy, etc.)
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