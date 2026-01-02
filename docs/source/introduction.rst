Introduction
============

Riemann is an open-source lightweight automatic differentiation library designed specifically for learning, educational, and research purposes related to automatic differentiation and neural networks.

Key Features
------------

Tensor Operations

- Provides tensor creation functions (tensor, zeros, ones, random, etc.) with support for complex tensors
- Supports basic mathematical operations (addition, subtraction, multiplication, division, exponentiation, elementary functions like exponential, logarithmic, trigonometric, hyperbolic functions, etc.)
- Supports vector and matrix operations (batch matrix multiplication, vector dot product, matrix determinant, matrix inverse, matrix factorization, etc.)
- Supports tensor shape reshaping, dimension expansion/reduction, indexing and slicing, element gathering/scattering, concatenation/splitting, etc.

Automatic Differentiation

- **backward method**: Triggers backpropagation to compute gradients
- **grad function**: Computes gradients of functions with respect to inputs
- **track_grad Decorator and Function Class**: Support custom gradient tracking functions

Jacobian and Hessian Matrices

- Supports Jacobian matrix computation for multi-input multi-output functions
- Provides Hessian matrix computation functionality for second-order derivatives
- Efficient computation of Jacobian-vector products and vector-Jacobian products
- Supports Hessian-vector product and vector-Hessian product computation

Linear Algebra Module

- Provides matrix factorization and backward gradient tracking (SVD, PLU, QR, etc.)
- Supports calculation of matrix inverse, pseudo-inverse, determinant, eigenvalues/eigenvectors
- Matrix norm and condition number computation
- Supports linear equation solving and least squares solving

Neural Network Modules

- Basic layers (Linear, Flatten, Dropout, BatchNorm, etc.)
- Activation functions (ReLU, Sigmoid, Softmax, etc.)
- Convolution and pooling layers (Conv1d/2d/3d, MaxPool1d/2d/3d, AvgPool1d/2d/3d, etc.)
- Loss functions (MSE, CrossEntropy, etc.)
- Optimizers (SGD, Adam, Adagrad, LBFGS, etc.)
- Network module containers (Sequential, ModuleList, ModuleDict, etc.)

Computer Vision Module

- Dataset classes: Loading and preprocessing of commonly used datasets such as MNIST, CIFAR10
- Image transformations: Resize, Crop, Flip, Rotate, Normalize and other image preprocessing operations

PyTorch Interface Compatibility
-------------------------------

Riemann library is designed with PyTorch interface compatibility in mind. Functions and classes with the same name maintain consistent interfaces, making it easy for PyTorch users to get started quickly:

- **Tensor Operations**: Supports tensor operation functions and methods with the same names as PyTorch, such as `tensor()`, `grad()`, `backward()`, etc.
- **Neural Network Components**: Layers, activation functions, and loss functions in the `nn` module maintain interface compatibility with PyTorch
- **Optimizers**: Optimizers in the `optim` module (such as SGD, Adam, etc.) maintain consistent interfaces with PyTorch
- **Automatic Differentiation Mechanism**: `requires_grad`, computation graph construction, and backpropagation mechanisms are similar to PyTorch
- **Computer Vision**: Datasets and transforms in the `vision` module maintain interface compatibility with torchvision

This design allows users familiar with PyTorch to easily migrate to the Riemann library for development and research work.