# Riemann

**Language**: [English](README.md) | [中文](README_zh.md)

Riemann is a neural network programming framework similar to PyTorch. It supports automatic differentiation in tensor computations, provides components for building neural networks, and is designed for learning, education, and research related to neural networks.


## Feature Introduction

### 1. Tensor Computation and Automatic Differentiation
- Supports mathematical operations on zero- to multi-dimensional tensors, including operations on complex-valued tensors and CUDA support.
- Linear algebra module supports matrix operations, including batch matrix multiplication, matrix decomposition, eigenvalues and determinants, matrix inverse and pseudoinverse, linear system solving, norms and ranks, etc.
- Supports tensor shape reshaping, dimension expansion/contraction, indexing and slicing, gather/scatter operations, concatenation/splitting, etc.
- Supports automatic gradient tracking based on the backpropagation algorithm, function differentiation with respect to inputs, and user-defined gradient functions.
- Supports tensor serialization/deserialization for convenient model training and deployment.

### 2. Neural Network Module
- Basic layers (Linear, Dropout, BatchNorm, LayerNorm, Embedding, etc.)
- Activation functions (ReLU, Sigmoid, Softmax, etc.)
- Loss functions (MSE, CrossEntropy, etc.)
- Convolution and pooling layers (Conv1d/2d/3d, MaxPool1d/2d/3d, AvgPool1d/2d/3d, etc.)
- Transformer (MultiheadAttention, TransformerEncoder, TransformerDecoder, Transformer, etc.)
- Optimizers (SGD, Adam, Adagrad, LBFGS, etc.)
- Network module containers (Sequential, ModuleList, ModuleDict, etc.)


## Application Scenarios

- **Deep learning research**: Custom model and algorithm development
- **Scientific computing**: Gradient calculation for complex mathematical models
- **Optimization problem solving**: Gradient descent and Adam optimization algorithms
- **Computer vision**: Image classification, object detection and other vision tasks
- **Education and teaching**: Learning automatic differentiation and deep learning principles


## PyTorch Interface Compatibility

Riemann library is designed with a focus on compatibility with PyTorch interfaces, with consistent interfaces for functions and classes with the same names, making it easy for PyTorch users to get started:

- **Tensor operations**: Supports tensor operation functions and methods with the same names as PyTorch, such as `tensor()`, `grad()`, `backward()`, etc.
- **Neural network components**: Layers, activation functions and loss functions in the `nn` module are interface-compatible with PyTorch
- **Optimizers**: Optimizers (such as SGD, Adam, etc.) in the `optim` module have consistent interfaces with PyTorch
- **Automatic differentiation mechanism**: `requires_grad` and backpropagation mechanism are similar to PyTorch
- **Computer vision**: Datasets and transforms in the `vision` module are interface-compatible with torchvision

This design allows users familiar with PyTorch to easily migrate to the Riemann library for development work.


## Project Folder Structure

```
Riemann/
├── src/
│   └── riemann/              # Core source code
│       ├── autograd/         # Automatic differentiation related modules
│       ├── nn/               # Neural network modules
│       ├── optim/            # Optimizers
│       ├── utils/            # Utility functions
│       ├── vision/           # Computer vision modules
│       ├── __init__.py       # Package configuration file
│       ├── dtype.py          # Data type definitions
│       ├── gradmode.py       # Gradient mode control
│       ├── linalg.py         # Linear algebra functions
│       ├── serialization.py  # Object saving and loading
│       └── tensordef.py      # Tensor definition and core operations
├── data/                     # Training and testing dataset directory
├── docs/                     # Project documentation directory
├── tests/                    # Test files
├── examples/                     # Example code
│   ├── backward_demo.py          # Backward propagation example
│   ├── grad_demo.py              # Gradient calculation example
│   ├── custom_grad_decorator.py  # Custom gradient tracking function example
│   ├── optimizers_comparison.py  # Optimizer comparison example
│   ├── mnist_demo.py             # MNIST handwritten digit recognition GUI example
│   ├── nn_MNIST_CE_SGD.py        # Neural network training example
│   ├── cnn_CIFAR10_CE_SGD.py     # Convolutional network training example
│   └── ...
├── README.md                 # Project documentation
├── LICENSE                   # License file
└── pyproject.toml            # Project configuration and dependency management
```

### Code Directory

The code directory `src/riemann/` is the location of Riemann library's core source code, containing all major functional modules:

- **autograd/**: Implements automatic differentiation functionality, including backpropagation algorithm and gradient calculation
- **nn/**: Neural network-related components, such as various layers, activation functions, and loss functions
- **optim/**: Optimizers, such as SGD, Adam, etc., and learning rate schedulers
- **utils/**: Utility functions, including Dataset class and Dataloader class for data loading, etc.
- **vision/**: Computer vision-related functionality, including datasets and image transforms
- **Core files**: Such as `tensordef.py` (tensor definition), `linalg.py` (linear algebra), etc.

### tests Directory

The `tests/` directory contains a large number of test cases for verifying that various functions of the Riemann library work correctly:

- Test cases are classified by module, covering various functional modules such as automatic differentiation, tensor operations, neural networks, etc.
- Test cases can be run in batches using pytest or run individually as standalone scripts
- Provides detailed test coverage to ensure library stability and reliability

### docs Directory

The `docs/` directory is used to store project documentation:

- Contains detailed API documentation
- Provides usage guides and tutorials
- Records project architecture and design decisions
- Helps users and developers better understand and use the Riemann library

### examples Directory

The `examples/` directory contains various example codes that demonstrate how to use different features of the Riemann library:

- **Basic examples**: Such as backward propagation example (backward_demo.py), gradient calculation example (grad_demo.py)
- **Custom gradient examples**: Such as custom_grad_decorator.py, demonstrating how to use decorators to customize gradients
- **Optimizer examples**: Such as optimizers_comparison.py, comparing the performance of different optimizers
- **Neural network examples**: Such as nn_MNIST_CE_SGD.py (MNIST handwritten digit recognition), cnn_CIFAR10_CE_SGD.py (CIFAR10 image classification)
- **GUI application examples**: Such as mnist_demo.py, providing a graphical interface for MNIST handwritten digit recognition

These example codes provide references for users to actually use the Riemann library, helping users quickly get started and understand the library's functionality.

Riemann provides rich example codes located in the `examples/` directory:

- **backward_demo.py**: Backward function usage demonstration
- **grad_demo.py**: Grad function usage demonstration
- **hessian.py**: Hessian matrix calculation example
- **jacobian.py**: Jacobian matrix calculation example
- **mnist_demo.py**: MNIST handwritten digit recognition GUI example (graphical interface training and recognition)
- **nn_MNIST_CE_SGD.py**: Neural network training example based on MNIST (cross-entropy loss + SGD optimization)
- **nn_MNIST_CE_Adam.py**: Neural network training example based on MNIST (cross-entropy loss + Adam optimization)
- **nn_MNIST_CE_Adagrad.py**: Neural network training example based on MNIST (cross-entropy loss + Adagrad optimization)
- **nn_MNIST_MSE_GD.py**: Neural network training example based on MNIST (mean squared error + gradient descent)
- **nn_MNIST_MSE_SGD.py**: Neural network training example based on MNIST (mean squared error + SGD optimization)
- **nn_MNIST_MSE_LBFGS.py**: Neural network training example based on MNIST (mean squared error + LBFGS optimization)
- **cnn_CIFAR10_CE_SGD.py**: Convolutional neural network example based on CIFAR10 (cross-entropy loss + SGD optimization)
- **cnn_CIFAR10_CE_Adam.py**: Convolutional neural network example based on CIFAR10 (cross-entropy loss + Adam optimization)
- **optimizers_comparison.py**: Optimizer performance comparison
- **scatter.py**: Tensor scatter operation example
- **custom_grad_decorator.py**: Example of using @track_grad decorator to customize gradient tracking functions
- **custom_grad_FunctionClass.py**: Example of using Function class to customize gradient tracking functions


## Installation Method

### Source Installation and Development Environment Configuration

```bash
# Get Riemann library source code from GitHub
git clone https://github.com/xiangfei2017/Riemann.git
cd Riemann
# Install package and its core dependencies in development mode (-e means editable mode, no need to reinstall after modifying code)
pip install -e .

# Install test dependencies
pip install -e .[tests]

# Install CUDA dependencies
# Note: Before using CUDA acceleration, please ensure you have installed the corresponding version of CUDA driver
pip install -e .[cuda]

# Install specific CUDA version dependencies
# CUDA 13.x
pip install -e .[cuda13]
# CUDA 12.x
pip install -e .[cuda12]
# CUDA 11.x
pip install -e .[cuda11]
# CUDA 10.x (Linux only)
pip install -e .[cuda10]

# Install general CuPy dependency (for macOS, ARM64, etc.)
pip install -e .[cupy]
```

### Dependency Description

#### Core Dependencies

Running `pip install riemann` will automatically install the following core dependencies:
- **numpy>=1.20.0**: Core numerical computation library
- **pillow>=8.0.0**: Used for image processing functionality in computer vision
- **tqdm>=4.0.0**: Used for progress bar display in neural network training

#### CUDA Dependencies

CUDA dependencies are not automatically installed and need to be explicitly specified:
- **cupy-cuda13x**: For CUDA 13.x on Linux and Windows platforms
- **cupy-cuda12x**: For CUDA 12.x on Linux and Windows platforms
- **cupy-cuda11x**: For CUDA 11.x on Linux and Windows platforms
- **cupy-cuda10x**: For CUDA 10.x on Linux platforms
- **cupy**: General version (for macOS, ARM64, and other platforms that don't support CUDA)

#### Platform Compatibility

- **CUDA support**: Only available on Linux or Windows systems with x86_64/AMD64 architecture
- **macOS systems**: NVIDIA CUDA is not supported, will automatically use CPU mode
- **ARM architecture**: For devices like NVIDIA Jetson, CUDA may be supported, but you need to install the corresponding ARM version of CUDA driver

### CUDA Driver Installation Instructions

1. **Check CUDA Driver Version**
   - Windows system: Right-click on desktop → NVIDIA Control Panel → Help → System Information → Driver Version
   - Linux system: Run in terminal `nvidia-smi` or `nvcc --version`

2. **Download and Install Corresponding CUDA Driver**
   - Visit NVIDIA official website: https://developer.nvidia.com/cuda-toolkit-archive
   - Select appropriate driver version based on your GPU model and operating system
   - Follow the wizard instructions to complete the installation

3. **Verify CUDA Driver Installation**
   - After installation, restart your computer and run:
     - Windows: `nvidia-smi`
     - Linux: `nvidia-smi` or `nvcc --version`
   - Confirm that the output shows the correct CUDA version information

### Verify Installation

After installation, you can run the following code to verify:

```python
import riemann as r
print("CUDA available:", r.cuda.is_available())
print("Using device:", r.device('cuda' if r.cuda.is_available() else 'cpu'))
```

If CUDA is installed successfully, it will display `CUDA available: True`, otherwise it will display `CUDA available: False` and automatically use CPU mode.




## Usage Method

### Detailed Documentation

Riemann's detailed usage guide is available online:
- **English documentation**: https://riemann-en.readthedocs.io/en/latest/
- **Chinese documentation**: https://riemann-zh.readthedocs.io/zh-cn/latest/

The documentation source files are located in the `docs` directory, written in reStructuredText format. You can also read the instructions in README.md and README_en.md in the docs directory to manually build the documentation in HTML format locally.

### Riemann Package Module Structure

```
riemann                  # Main package
├── autograd             # Automatic differentiation module
│   └── functional       # Automatic differentiation functional interface
├── linalg               # Linear algebra module
├── nn                   # Neural network module
│   └── functional       # Neural network functions
├── optim                # Optimizer module
│   └── lr_scheduler     # Learning rate scheduler module
├── utils                # Utility function module
│   └── data             # Data processing tools
├── vision               # Computer vision module
│   ├── datasets         # Dataset classes
│   └── transforms       # Image transform operations
└── cuda                 # CUDA/GPU support
```

### Module Import Examples

**Import the entire riemann module:**

```python
import riemann as r

# Use tensor creation function
t = r.tensor([1.0, 2.0, 3.0])

# Use automatic differentiation functionality
x = r.tensor([1.0, 2.0], requires_grad=True)
y = x ** 2
y.sum().backward()
print(x.grad)  # Output: [2. 4.]
```

**Import required functions and classes by module tree:**

```python
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
```

### Example 1: Derivative (Gradient) Calculation Example

```python
# Example 1: Derivative (Gradient) Calculation Example
# This example demonstrates two methods of calculating gradients:
# 1. Using the grad function to directly calculate the gradient of a function with respect to input
# 2. Using the backward method to calculate gradients through backpropagation

# Import riemann library
from riemann import tensor
from riemann.autograd import grad

# Create tensor
t = tensor([1.0, 2.0, 3.0], requires_grad=True)

# Define function
def f(x):
    return (x ** 2.0).sum()

# Calculate gradient
output = f(t)
grad_f = grad(output, t)[0]
print("Gradient:", grad_f)  # Output: Gradient: [2. 4. 6.]

# Backward propagation example
x = tensor([1.0, 2.0], requires_grad=True)
y = tensor([3.0, 4.0], requires_grad=True)
z = (x * y).sum()
z.backward()
print("x's gradient:", x.grad)  # Output: x's gradient: [3. 4.]
print("y's gradient:", y.grad)  # Output: y's gradient: [1. 2.]
```

### Example 2: Jacobian Matrix Calculation Example

```python
# Example 2: Jacobian Matrix Calculation Example
# This example demonstrates how to use the jacobian function to calculate the Jacobian matrix of a function with respect to input
# The Jacobian matrix is a matrix of partial derivatives of function outputs with respect to inputs, which is very important for multi-input multi-output functions

from riemann import tensor
from riemann.autograd.functional import jacobian

# Define function
def f(x):
    return x ** 2.0

# Create input tensor
x = tensor([1.0, 2.0, 3.0], requires_grad=True)

# Calculate Jacobian matrix
jacob = jacobian(f, x)
print("Jacobian matrix:", jacob)
```

### Example 3: Simple Neural Network Training Example

```python
# Example 3: Simple Neural Network Training Example
# This example demonstrates how to train a simple neural network (linear regression model) to sum two numbers
# Including model creation, loss function definition, optimizer configuration, training loop, and prediction process

# Neural network training example: training a network for summing two numbers, this network model is essentially a linear regression
from riemann import tensor
from riemann.nn import Linear, MSELoss
from riemann.optim import SGD

# Create model
model = Linear(2, 1)
criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01)

# Training data
inputs = tensor([[1.0, 2.0], [3.0, 4.0]])
targets = tensor([[3.0], [7.0]])

# Training loop
for epoch in range(100):
    # Forward propagation
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward propagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item()}')

# Prediction
new_input = tensor([[5.0, 6.0]])
prediction = model(new_input)
print(f'Prediction result: {prediction.item()}')
```

### Example 4: Simple Convolutional Neural Network Training Example

```python
# Example 4: Simple Convolutional Neural Network Training Example
# This example demonstrates how to use a convolutional neural network (CNN) to train a CIFAR10 image classification model
# Including model definition, data loading and preprocessing, training loop, model evaluation, and single sample inference

import riemann as r
from riemann.vision.datasets import CIFAR10
from riemann.vision.transforms import *
from riemann.nn import *
from riemann.optim import SGD
from tqdm import tqdm

# Load data
# Use data augmentation for training set, not for test set
train_transform = Compose([
    RandomHorizontalFlip(),  # Random horizontal flip
    ToTensor(),
    Normalize((0.5279, 0.5303, 0.5373), (0.2739, 0.2728, 0.2625))  # CIFAR10 actual normalization parameters
])

test_transform = Compose([
    ToTensor(),
    Normalize((0.5279, 0.5303, 0.5373), (0.2739, 0.2728, 0.2625))  # CIFAR10 actual normalization parameters
])

train_dataset = CIFAR10(root='data', train=True, transform=train_transform)
test_dataset = CIFAR10(root='data', train=False, transform=test_transform)

# Reduce batch size and data volume to speed up testing
train_loader = r.utils.DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = r.utils.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Create model, loss function, and optimizer
model = Sequential(
    Conv2d(3, 16, kernel_size=3, padding=1),
    ReLU(),
    MaxPool2d(kernel_size=2, stride=2),
    Flatten(),
    Linear(16 * 16 * 16, 10)
)
criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(3):  # Train for 3 epochs
    total_loss = 0
    # Use tqdm to display progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/3")
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        # Forward propagation
        output = model(data)
        loss = criterion(output, target)   # Calculate loss between output and target labels
        
        # Backward propagation and optimizer update
        optimizer.zero_grad()   # Clear gradients of training parameters
        loss.backward()         # Calculate gradients of loss with respect to training parameters
        optimizer.step()        # Update training parameters
        
        total_loss += loss.item()
        
        # Update progress bar to display current loss
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss/len(train_loader)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

# Model evaluation (inference test)
model.eval()  # Set to evaluation mode
correct = 0
total = 0

# Use tqdm to display test progress
test_progress_bar = tqdm(test_loader, desc="Testing")

with r.no_grad():  # Disable gradient calculation
    for data, target in test_progress_bar:
        # Forward propagation
        outputs = model(data)
        
        # Get prediction results
        predicted = outputs.argmax(dim=1)  # Get predicted class for each sample
        total += target.size(0)  # Accumulate test sample count
        correct += (predicted == target).sum().item() # Accumulate correctly predicted sample count
        
        # Update progress bar to display current accuracy
        current_accuracy = 100 * correct / total
        test_progress_bar.set_postfix({"Accuracy": f"{current_accuracy:.2f}%"})

# Output final test accuracy
test_accuracy = 100 * correct / total
print(f"Test set accuracy: {test_accuracy:.2f}% ({correct}/{total})")

# Single sample inference example
sample_data, sample_target = next(iter(test_loader))
sample_output = model(sample_data[:1])  # Only take the first sample
predicted_class = sample_output.argmax(dim=1)
print(f"Sample predicted class: {predicted_class.item()}, actual class: {sample_target[0].item()}")

print("CNN training and inference test completed!")
```

### Example 5: GPU Acceleration Example

```python
# Example 6: GPU Acceleration Example
# This example demonstrates how to use GPU acceleration for neural network training in Riemann
# Including device detection and setup, model and data device migration, training and evaluation process on GPU

import riemann as r
from riemann.nn import Linear, Flatten, ReLU, Sequential, CrossEntropyLoss
from riemann.optim import Adam
from riemann.vision.datasets import MNIST
from riemann.vision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm

# Check if CUDA is available
device = r.device('cuda' if r.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load MNIST dataset
transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])

train_dataset = MNIST(root='data', train=True, transform=transform)
test_dataset = MNIST(root='data', train=False, transform=transform)

train_loader = r.utils.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = r.utils.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Create model and move to specified device
model = Sequential(
    Flatten(),
    Linear(28*28, 128),
    ReLU(),
    Linear(128, 10)
)
model.to(device)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        # Move data to specified device
        data, target = data.to(device), target.to(device)
        
        # Forward propagation
        output = model(data)
        loss = criterion(output, target)
        
        # Backward propagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

# Model evaluation
model.eval()
correct = 0
total = 0

with r.no_grad():
    test_progress_bar = tqdm(test_loader, desc="Testing")
    for data, target in test_progress_bar:
        # Move data to specified device
        data, target = data.to(device), target.to(device)
        
        # Forward propagation
        outputs = model(data)
        
        # Get prediction results
        predicted = outputs.argmax(dim=1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        current_accuracy = 100 * correct / total
        test_progress_bar.set_postfix({"Accuracy": f"{current_accuracy:.2f}%"})

# Output final test accuracy
test_accuracy = 100 * correct / total
print(f"Test set accuracy: {test_accuracy:.2f}% ({correct}/{total})")
```

## Testing Method

The tests directory in the Riemann project folder includes test cases that cover all functionalities, which can be run in batches using pytest or as standalone scripts.

### Run Tests in Batches with pytest

You can use the following commands to run all tests in batches:

```bash
# Run all test files in the tests directory
pytest tests

# Run specific test files
pytest tests/test_010_grad.py

# Run specific test classes or methods
pytest tests/test_011_jacobian.py::TestJacobianFunctions::test_single_input_single_output

# Run tests and generate coverage report
pytest --cov=riemann tests/

# Run vision module tests
pytest tests/test_052_vision.py
```

### Run Test Scripts Individually

You can also run test scripts individually:

```bash
cd tests
python test_010_grad.py
# Run other test scripts
```


## Third-party Dependencies and Licenses

### Core Dependencies

| Library | Version Requirement | License Type | Notes                                     |
|---------|---------------------|--------------|-------------------------------------------|
| NumPy   | >=1.20.0            | BSD 3-Clause | Core numerical computation library        |
| Pillow  | >=8.0.0             | BSD 3-Clause | Image processing library                  |
| tqdm    | >=4.0.0             | MIT          | Progress bar for training and data loading|

### Testing Dependencies

| Library    | Version Requirement | Purpose           | License Type  | Notes                                     |
|------------|---------------------|-------------------|---------------|-------------------------------------------|
| PyTorch    | >=2.0.0             | Result comparison | BSD 3-Clause  | Used for verifying calculation results    |
| torchvision| >=0.15.0            | Computer vision   | BSD 3-Clause  | PyTorch's computer vision library         |
| pytest     | >=7.0.0             | Testing framework | MIT           | Used for organizing and running tests     |

### Optional CUDA Dependencies

| Library      | Version Requirement | Purpose           | License Type | Notes                                     |
|--------------|---------------------|-------------------|--------------|-------------------------------------------|
| cupy-cuda12x | Latest              | GPU acceleration  | MIT          | For CUDA 12.x on Linux x86_64             |
| cupy-cuda11x | Latest              | GPU acceleration  | MIT          | For CUDA 11.x on Linux x86_64             |
| cupy-cuda10x | Latest              | GPU acceleration  | MIT          | For CUDA 10.x on Linux x86_64             |
| cupy         | Latest              | GPU acceleration  | MIT          | For other platforms (including Windows)   |

*Note: This project also utilizes Python's standard library components (like unittest) for testing, which don't require separate installation.*

*Note: Details of the BSD 3-Clause license for NumPy, PyTorch and Pillow can be found on their official websites.*

## License

This project adopts the BSD 3-Clause license. See the LICENSE file for details.

## Contribution Guidelines

Welcome to submit Issues and Pull Requests! Before contributing code, please ensure all contributions comply with the project's coding standards and pass all tests.

## Contact Information

Author: Fei Xiang
Email: xfeix@outlook.com
Gitee: https://gitee.com/xfcode2021