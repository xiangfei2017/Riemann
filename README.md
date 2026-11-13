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
- KAN network (KANLinear, KAN)
- Transformer (MultiheadAttention, TransformerEncoder, TransformerDecoder, Transformer, etc.)
- Optimizers (SGD, Adam, Adagrad, LBFGS, etc.)
- Network module containers (Sequential, ModuleList, ModuleDict, etc.)


## Application Scenarios

- **Deep learning research**: Custom model and algorithm development
- **Scientific computing**: Gradient calculation for complex mathematical models, and optimization problem solving
- **Computer vision**: Image classification and other vision tasks
- **Education and teaching**: Learning automatic differentiation and deep learning principles


## PyTorch Interface Compatibility

Riemann library is designed with a focus on compatibility with PyTorch interfaces, with consistent interfaces for functions and classes with the same names, making it easy for PyTorch users to get started:

- **Tensor operations**: Supports tensor operation functions and methods with the same names as PyTorch, such as `tensor()`, `grad()`, `backward()`, etc.
- **Neural network components**: Layers, activation functions and loss functions in the `nn` module are interface-compatible with PyTorch
- **Optimizers**: Optimizers (such as SGD, Adam, etc.) in the `optim` module have consistent interfaces with PyTorch
- **Automatic differentiation mechanism**: `requires_grad` and backpropagation mechanism are similar to PyTorch
- **torch.func compatibility**: The top-level ``func`` module mirrors `torch.func` API — curried ``grad``, ``jacrev``/``jacfwd``, ``jvp``/``vjp``, ``hessian``, ``vmap``, ``functional_call``
- **Computer vision**: Datasets and transforms in the `vision` module are interface-compatible with torchvision
- **Cross-framework interoperability**: Riemann tensors can be converted to and from NumPy arrays, CuPy arrays and other tensor frameworks such as PyTorch, or share data with them, through `r.tensor()`, `r.as_tensor()`, `r.from_dlpack()`, `TN.array()` and `TN.numpy()`, built on the `__array__`, `__cuda_array_interface__`, `__dlpack__` and `__buffer__` protocols. Conversion is zero-copy when device and dtype match

This design allows users familiar with PyTorch to easily migrate to the Riemann library for development work.


## Project Folder Structure

```
Riemann/
├── src/
│   └── riemann/              # Core source code
│       ├── autograd/         # Automatic differentiation related modules
│       ├── core/             # Core tensor subpackage (tensor definition, functional ops, dtype, grad mode, serialization)
│       │   ├── __init__.py       # Aggregated exports of the core subpackage
│       │   ├── tensordef.py      # Tensor definition (TN class), dynamic graph and autograd
│       │   ├── tensorfuncs.py    # Functional tensor operations (one-way dependency on tensordef)
│       │   ├── dtype.py          # Data type definitions
│       │   ├── gradmode.py       # Gradient mode control
│       │   └── serialization.py  # Object saving and loading
│       ├── nn/               # Neural network modules
│       ├── optim/            # Optimizers
│       ├── utils/            # Utility functions
│       ├── vision/           # Computer vision modules
│       ├── __init__.py       # Package configuration file
│       ├── cuda.py           # CUDA/GPU support
│       ├── func.py           # Functional autodiff API
│       └── linalg.py         # Linear algebra functions
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
- **core/**: Core tensor subpackage exposing the tensor API as a whole
  - `tensordef.py`: Tensor definition (TN class), dynamic graph and autograd
  - `tensorfuncs.py`: Functional tensor operations (trigonometric functions, index selection, random generators, einsum, etc.), with a one-way dependency on `tensordef.py`
  - `dtype.py`: Data type definitions
  - `gradmode.py`: Gradient mode control
  - `serialization.py`: Object saving and loading
- **nn/**: Neural network-related components, such as various layers, activation functions, and loss functions
- **optim/**: Optimizers, such as SGD, Adam, etc., and learning rate schedulers
- **utils/**: Utility functions, including Dataset class and Dataloader class for data loading, etc.
- **vision/**: Computer vision-related functionality, including datasets and image transforms
- **`func`**: Functional-style autodiff interface, reference `torch.func`. Provides curried, pure-function APIs such as ``grad``, ``jvp``, ``vjp``, ``jacrev``, ``jacfwd``, ``hessian``, ``hvp``, ``vmap``, ``functional_call`` — suitable for chained higher-order derivatives
- **Core files**: Such as `linalg.py` (linear algebra), `cuda.py` (GPU support), etc.

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

- **backward_demo.py**: Backward function usage demonstration
- **grad_demo.py**: Grad function usage demonstration
- **hessian.py**: Hessian matrix calculation example
- **jacobian.py**: Jacobian matrix calculation example
- **gradient_clipping_demo.py**: Gradient clipping example
- **custom_grad_decorator.py**: Example of using @track_grad decorator to customize gradient tracking functions
- **custom_grad_FunctionClass.py**: Example of using Function class to customize gradient tracking functions
- **scatter.py**: Tensor scatter operation example
- **optimizers_comparison.py**: Optimizer performance comparison
- **nn_lr_scheduler.py**: Learning rate scheduler usage example
- **nn_MNIST_CE_SGD.py**: MNIST handwritten digit recognition (cross-entropy + SGD)
- **nn_MNIST_CE_Adam.py**: MNIST handwritten digit recognition (cross-entropy + Adam)
- **nn_MNIST_CE_Adagrad.py**: MNIST handwritten digit recognition (cross-entropy + Adagrad)
- **nn_MNIST_MSE_GD.py**: MNIST handwritten digit recognition (MSE + gradient descent)
- **nn_MNIST_MSE_SGD.py**: MNIST handwritten digit recognition (MSE + SGD)
- **nn_MNIST_MSE_LBFGS.py**: MNIST handwritten digit recognition (MSE + LBFGS)
- **cnn_CIFAR10_CE_SGD.py**: CIFAR10 image classification CNN (cross-entropy + SGD)
- **cnn_CIFAR10_CE_Adam.py**: CIFAR10 image classification CNN (cross-entropy + Adam)
- **kan_mnist.py**: KAN network MNIST classification example
- **transformer_timeseq.py**: Transformer time series forecasting example
- **mnist_demo.py**: MNIST handwritten digit recognition GUI application


## Installation

Riemann is **not** published on PyPI, so `pip install riemann` will not work. Install it from source.

**Python Version**: Riemann requires Python >= 3.9, supports Python 3.9 ~ 3.14.

### Dependencies

| Category | Packages | Notes |
|----------|----------|-------|
| Core | numpy>=1.20.0, pillow>=8.0.0, tqdm>=4.0.0 | Required to run Riemann |
| Tests | pytest>=7.0.0, torch>=2.0.0, torchvision>=0.15.0 | Only needed to run the test suite |
| CUDA | cupy-cuda12x (default), cupy-cuda13x, cupy-cuda11x, cupy-cuda10x | Only needed for NVIDIA GPU acceleration |

### Install from Source

```bash
git clone https://github.com/xiangfei2017/Riemann.git
cd Riemann

# Base install (-e is editable mode; no reinstall after code changes)
pip install -e .

# Optional: test dependencies
pip install -e .[tests]

# Optional: CUDA dependencies (defaults to CUDA 12.x; or cuda13 / cuda11 / cuda10)
pip install -e .[cuda]
```

### Installing Dependencies Manually

If pip/conda fails due to network or dependency resolution issues, install packages one by one:

```bash
# Core
pip install numpy pillow tqdm
# Tests
pip install pytest torch torchvision
# CUDA (pick the version matching `nvcc --version`)
pip install cupy-cuda12x
```

- On a system-wide environment the OS package manager also works (e.g. `apt install python3-numpy`, `brew install numpy`); **a virtual environment can only use pip**;
- Install torch separately with the command from the [PyTorch website](https://pytorch.org/get-started/locally/);
- Conda users can create the env with `conda create -n riemann python=3.10`, but CuPy is still best installed via pip.

Then finish the Riemann install with `pip install -e . --no-deps`.

### CUDA Dependencies and Platform Compatibility

CUDA support is not installed automatically; use `pip install -e .[cuda]` and make sure a matching NVIDIA driver is installed.

| Platform | Architecture | CUDA Support | Notes |
|----------|--------------|--------------|-------|
| Linux | x86_64/AMD64 | ✅ Supported | `pip install cupy-cudaXX` |
| Windows | x86_64/AMD64 | ✅ Supported | `pip install cupy-cudaXX` |
| macOS | x86_64/ARM64 | ❌ Not supported | No NVIDIA drivers; falls back to CPU |
| Linux (ARM64) | aarch64/arm64 | ⚠️ Build from source | NVIDIA Jetson needs CuPy built from source |
| Windows (ARM64) | aarch64/arm64 | ❌ Not supported | Not supported yet |

### Installing the CUDA Driver

1. Run `nvcc --version` to check the installed CUDA Toolkit version (`nvidia-smi` shows the maximum version supported by the driver, not the installed one).
2. Download and install a matching driver/toolkit from the [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).
3. Restart and run `nvcc --version` again to verify.

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

The documentation source files are located in the `docs` directory, written in reStructuredText format. To build HTML docs locally:

**Windows:**

```batch
cd docs
make.bat html-all     REM build both English and Chinese
make.bat html-en      REM English only
make.bat html-zh      REM Chinese only
make.bat clean-all    REM clean build artifacts
```

**Linux / macOS:**

```bash
cd docs
chmod +x make.sh      # first-time only
./make.sh html-all    # build both English and Chinese
./make.sh html-en     # English only
./make.sh html-zh     # Chinese only
./make.sh clean-all   # clean build artifacts
```

### Riemann Package Module Structure

```
riemann                  # Main package
├── autograd             # Automatic differentiation module
│   └── functional       # Automatic differentiation functional interface
├── func                 # torch.func-style functional autodiff API
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

# Import functional autodiff API (torch.func-style, curried, supports higher-order)
from riemann import func
from riemann.func import grad as fgrad, jvp, vjp, jacrev, jacfwd, hessian, hvp, vmap, functional_call

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

train_dataset = CIFAR10(root='data', train=True, transform=train_transform, download=True)
test_dataset = CIFAR10(root='data', train=False, transform=test_transform, download=True)

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
# Example 5: GPU Acceleration Example
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

train_dataset = MNIST(root='data', train=True, transform=transform, download=True)
test_dataset = MNIST(root='data', train=False, transform=transform, download=True)

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
pytest tests/test_008_autograd.py

# Run specific test classes or methods
pytest tests/test_008_autograd.py::TestAutograd::test_jacobian

# Run tests and generate coverage report
pytest --cov=riemann tests/

# Run vision module tests
pytest tests/test_052_vision.py
```

### Run Test Scripts Individually

You can also run test scripts individually:

```bash
cd tests
python test_008_autograd.py
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

| Library      | Version Requirement | Purpose          | License Type | Notes                                     |
|--------------|---------------------|------------------|--------------|-------------------------------------------|
| cupy-cuda13x | Latest              | GPU acceleration | MIT          | For CUDA 13.x on Linux/Windows x86_64     |
| cupy-cuda12x | Latest              | GPU acceleration | MIT          | For CUDA 12.x on Linux/Windows x86_64     |
| cupy-cuda11x | Latest              | GPU acceleration | MIT          | For CUDA 11.x on Linux/Windows x86_64     |
| cupy-cuda10x | Latest              | GPU acceleration | MIT          | For CUDA 10.x on Linux x86_64 only        |

*Note: This project also utilizes Python's standard library components (like unittest) for testing, which don't require separate installation.*

*Note: Details of the BSD 3-Clause license for NumPy, PyTorch and Pillow can be found on their official websites.*

## License

This project adopts the BSD 3-Clause license. See the LICENSE file for details.

## Contribution Guidelines

Welcome to submit Issues and Pull Requests! Before contributing code, please ensure all contributions comply with the project's coding standards and pass all tests.

## Contact Information

Author: Fei Xiang

GitHub: https://github.com/xiangfei2017

Gitee: https://gitee.com/xfcode2021