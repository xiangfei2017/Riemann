Introduction
============

Riemann is an open-source lightweight automatic differentiation library designed specifically for learning, educational, and research purposes related to automatic differentiation and neural networks.

Key Features
------------

- **Automatic Differentiation**: Supports forward and reverse automatic differentiation for scalars, vectors, and tensors of real and complex numbers
- **Gradient Computation**: Supports backpropagation algorithm for gradient computation, providing `grad` and `backward` functions for efficient gradient calculation
- **Tensor Operations**: Provides rich tensor operation functions, including arithmetic operations, elementary functions, indexing operations, shape operations, dimension expansion/reduction, stacking/splitting
- **Higher-order Derivatives**: Supports Jacobian and Hessian matrix computation, as well as JVP (Jacobian-vector product), VJP (vector-Jacobian product), HVP (Hessian-vector product), and VHP (vector-Hessian product)
- **Neural Network Components**: Contains basic neural network modules, activation functions, loss functions, and optimizers
- **Computer Vision Support**: Provides common dataset classes and image transformation functions, supporting loading and preprocessing of datasets like MNIST and CIFAR10

PyTorch Interface Compatibility
-------------------------------

Riemann library is designed with PyTorch interface compatibility in mind. Functions and classes with the same name maintain consistent interfaces, making it easy for PyTorch users to get started quickly:

- **Tensor Operations**: Supports tensor operation functions and methods with the same names as PyTorch, such as `tensor()`, `grad()`, `backward()`, etc.
- **Neural Network Components**: Layers, activation functions, and loss functions in the `nn` module maintain interface compatibility with PyTorch
- **Optimizers**: Optimizers in the `optim` module (such as SGD, Adam, etc.) maintain consistent interfaces with PyTorch
- **Automatic Differentiation Mechanism**: `requires_grad`, computation graph construction, and backpropagation mechanisms are similar to PyTorch
- **Computer Vision**: Datasets and transforms in the `vision` module maintain interface compatibility with torchvision

This design allows users familiar with PyTorch to easily migrate to the Riemann library for development and research work.