# Riemann

Riemann is a lightweight automatic differentiation library and neural network programming framework that supports automatic gradient tracking for scalars, vectors, and tensors. It provides common components needed for building neural networks, with flexible and extensible interfaces, compatible with PyTorch, and specifically designed for learning, education, and research related to neural networks.

## Key Features

- **Automatic Differentiation**: Supports forward computation and backward automatic differentiation for real and complex scalars, vectors, and tensors
- **Gradient Computation**: Supports backpropagation algorithm for gradient calculation, providing `grad` and `backward` functions for efficient gradient computation, supporting backward gradient tracking in scalar, vector, matrix, and multi-dimensional tensor calculations
- **Tensor Operations**: Provides rich tensor operation functionality, including: addition, subtraction, multiplication, division, elementary functions, indexing operations, shape operations, dimension expansion/reduction, stacking/splitting
- **Higher-order Derivatives**: Supports Jacobian matrix and Hessian matrix computation, as well as JVP (Jacobian-vector product), VJP (vector-Jacobian product), HVP (Hessian-vector product), and VHP (vector-Hessian product)
- **Neural Network Components**: Contains basic neural network modules, activation functions, loss functions, and optimizers
- **Computer Vision Support**: Provides commonly used dataset classes and image transformation functions, supporting loading and preprocessing of datasets such as MNIST and CIFAR10

## PyTorch Interface Compatibility

The Riemann library is designed with attention to PyTorch interface compatibility, maintaining consistent interfaces for functions and classes with the same names, making it easy for PyTorch users to get started quickly:

- **Tensor Operations**: Supports tensor operation functions and methods with the same names as PyTorch, such as `tensor()`, `grad()`, `backward()`, etc.
- **Neural Network Components**: Layers, activation functions, and loss functions in the `nn` module maintain interface compatibility with PyTorch
- **Optimizers**: Optimizers in the `optim` module (such as SGD, Adam, etc.) maintain consistent interfaces with PyTorch
- **Automatic Differentiation Mechanism**: `requires_grad`, computation graph construction, and backpropagation mechanisms are similar to PyTorch
- **Computer Vision**: Datasets and transformations in the `vision` module maintain interface compatibility with torchvision

This design allows users familiar with PyTorch to easily migrate to the Riemann library for development and research work.

## Installation

### Install from PyPI (To Be Released)

```bash
pip install riemann
```

### Source Installation and Development Environment Configuration

```bash
# Get Riemann library source code from Gitee
git clone https://gitee.com/xfcode2021/Riemann.git
cd Riemann
# Install the package and its core dependencies in development mode (-e means editable mode, no need to reinstall after code modification)
pip install -e .

# If you need to run test code in the tests directory, you also need to install testing dependencies
pip install -e .[tests]
```

### Example Code Dependencies

Running example code in the examples directory requires installing the following additional dependencies:

```bash
pip install tqdm pillow
```

- **tqdm**: Used for progress bar display in neural network training examples
- **pillow**: Used for image processing functionality in computer vision examples (providing PIL module)

numpy is already included as a core dependency in the package installation and does not need to be installed separately.

Dependencies may vary slightly among different example files, and specific dependency information will be explained in the header comments of each example file.

## Quick Start

```python
# Import riemann library
from riemann import tensor
from riemann.autograd import grad

# Create tensors
t = tensor([1.0, 2.0, 3.0], requires_grad=True)

# Define function
def f(x):
    return (x ** 2.0).sum()

# Calculate gradient
output = f(t)
grad_f = grad(output, t)[0]
print("Gradient:", grad_f)  # Output: Gradient: [2. 4. 6.]

# Backpropagation example
x = tensor([1.0, 2.0], requires_grad=True)
y = tensor([3.0, 4.0], requires_grad=True)
z = (x * y).sum()
z.backward()
print("Gradient of x:", x.grad)  # Output: Gradient of x: [3. 4.]
print("Gradient of y:", y.grad)  # Output: Gradient of y: [1. 2.]
```

## Core Features

### 1. Tensor Operations
- Supports basic mathematical operations (addition, subtraction, multiplication, division, power, exponential functions, logarithmic functions, trigonometric functions, etc.)
- Provides tensor creation functions (zeros, ones, random, etc.)
- Supports tensor shape reshaping, dimension expansion/reduction, indexing and slicing, element gathering/scattering, concatenation/splitting, etc.

### 2. Automatic Differentiation
- **backward method**: Triggers backpropagation to compute gradients
- **grad function**: Computes gradients of functions with respect to inputs
- Supports higher-order derivatives and complex computation graphs

### 3. Jacobian and Hessian Matrices
- Supports Jacobian matrix computation for multi-input multi-output functions
- Provides Hessian matrix computation functionality for second-order derivatives
- Efficient computation of Jacobian-vector products and vector-Jacobian products
- Supports Hessian-vector product and vector-Hessian product computation

### 4. Neural Network Modules
- Basic layers (Linear, ReLU, Sigmoid, etc.)
- Convolution and pooling layers (Conv1d/2d/3d, MaxPool1d/2d/3d, AvgPool1d/2d/3d, etc.)
- Loss functions (MSE, CrossEntropy, etc.)
- Optimizers (SGD, Adam, Adagrad, LBFGS, etc.)

### 5. Computer Vision Module
- Dataset classes: Loading and preprocessing of commonly used datasets such as MNIST, CIFAR10
- Image transformations: Resize, Crop, Flip, Rotate, Normalize and other image preprocessing operations
- torchvision-compatible API design for easy transfer learning

## Project Structure

```
Riemann/
├── src/
│   └── riemann/              # Core source code
│       ├── autograd/         # Automatic differentiation related modules
│       ├── nn/               # Neural network module
│       ├── utils/            # Utility functions
│       ├── vision/           # Computer vision module
│       ├── __init__.py       # Package configuration file
│       ├── dtype.py          # Data type definitions
│       ├── gradmode.py       # Gradient mode control
│       ├── linalg.py         # Linear algebra functions
│       ├── optim.py          # Optimizers
│       ├── serialization.py  # Object saving and loading
│       └── tensordef.py      # Tensor definition and core operations
├── data/                     # Training and test dataset file directory
├── docs/                     # Project documentation directory
├── tests/                    # Test files
├── examples/                 # Example code
│   ├── backward_demo.py      # Backpropagation example
│   ├── grad_demo.py          # Gradient computation example
│   ├── hessian.py            # Hessian matrix computation example
│   ├── jacobian.py           # Jacobian matrix computation example
│   ├── nn_MNIST_CE_SGD.py    # Neural network training example
│   ├── cnn_CIFAR10_CE_SGD.py # CNN training example
│   └── ...
├── README.md                 # Project documentation
├── README_en.md              # English project documentation
├── LICENSE                   # License file
└── pyproject.toml            # Project configuration and dependency management
```

## More Examples

### Jacobian Matrix Computation

```python
from riemann import tensor
from riemann.autograd.functional import jacobian

# Define function
def f(x):
    return x ** 2.0

# Create input tensor
x = tensor([1.0, 2.0, 3.0], requires_grad=True)

# Compute Jacobian matrix
jacob = jacobian(f, x)
print("Jacobian matrix:", jacob)
```

### Neural Network Training

```python
# Neural network training example: Train a network to find the sum of two numbers, this model is essentially linear regression
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
    
    # Backpropagation and optimization
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

### Computer Vision Support

Riemann provides rich computer vision functionality, including commonly used dataset classes and image transformation functions:

### Supported Datasets

- **MNIST**: Handwritten digit recognition dataset
- **CIFAR10**: 10-class color image dataset

### Supported Image Transformations

- **Basic transformations**: ToTensor, ToPILImage, Normalize
- **Geometric transformations**: Resize, CenterCrop, RandomResizedCrop, FiveCrop, TenCrop, Pad
- **Random transformations**: RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomGrayscale
- **Color transformations**: ColorJitter, Grayscale
- **Combined transformations**: Compose, Lambda

### Computer Vision Dataset Loading

```python
from riemann.vision.datasets import MNIST, CIFAR10
from riemann.vision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip

# Define MNIST transformations
mnist_transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))  # MNIST normalization parameters
])

# Define CIFAR10 training set transformations (including data augmentation)
cifar10_train_transform = Compose([
    RandomHorizontalFlip(),  # Random horizontal flip
    ToTensor(),
    Normalize((0.5279, 0.5303, 0.5373), (0.2739, 0.2728, 0.2625))  # Actual CIFAR10 normalization parameters
])

# Define CIFAR10 test set transformations (without data augmentation)
cifar10_test_transform = Compose([
    ToTensor(),
    Normalize((0.5279, 0.5303, 0.5373), (0.2739, 0.2728, 0.2625))  # Actual CIFAR10 normalization parameters
])

# Load MNIST dataset
train_dataset = MNIST(root='data', train=True, transform=mnist_transform)
test_dataset = MNIST(root='data', train=False, transform=mnist_transform)

# Load CIFAR10 dataset
cifar10_train = CIFAR10(root='data', train=True, transform=cifar10_train_transform)
cifar10_test = CIFAR10(root='data', train=False, transform=cifar10_test_transform)

print(f"MNIST training set size: {len(train_dataset)}")
print(f"CIFAR10 test set size: {len(cifar10_test)}")
```

### CNN Example

```python
import riemann as r
from riemann.vision.datasets import CIFAR10
from riemann.vision.transforms import *
from riemann.nn import *
from riemann.optim import SGD

# Define CNN model
class SimpleCNN(r.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Use only 1 convolutional layer
        self.conv1 = Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = Flatten()
        # After pooling, image size changes from 32x32 to 16x16, with 16 channels
        self.fc1 = Linear(16 * 16 * 16, 10)  # Directly output 10 classes
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        return x

# Load data
# Training set uses data augmentation, test set does not
train_transform = Compose([
    RandomHorizontalFlip(),  # Random horizontal flip
    ToTensor(),
    Normalize((0.5279, 0.5303, 0.5373), (0.2739, 0.2728, 0.2625))  # Actual CIFAR10 normalization parameters
])

test_transform = Compose([
    ToTensor(),
    Normalize((0.5279, 0.5303, 0.5373), (0.2739, 0.2728, 0.2625))  # Actual CIFAR10 normalization parameters
])

train_dataset = CIFAR10(root='data', train=True, transform=train_transform)
test_dataset = CIFAR10(root='data', train=False, transform=test_transform)

# Reduce batch size and data amount to speed up testing
train_loader = r.utils.DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = r.utils.DataLoader(test_dataset, batch_size=100, shuffle=False)

# Create model, loss function, and optimizer
model = SimpleCNN()
criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.01)

# Training loop
from tqdm import tqdm

for epoch in range(3):  # Train 3 epochs
    total_loss = 0
    # Use tqdm to display progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/3")
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        # Forward propagation
        output = model(data)
        loss = criterion(output, target)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar to display current loss
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss/len(train_loader)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

# Model evaluation (inference testing)
model.eval()  # Set to evaluation mode
correct = 0
total = 0

# Use tqdm to display test progress
test_progress_bar = tqdm(test_loader, desc="Testing")

with r.no_grad():  # Disable gradient computation
    for data, target in test_progress_bar:
        # Forward propagation
        outputs = model(data)
        
        # Get prediction results
        _, predicted = r.max(outputs, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # Update progress bar to display current accuracy
        current_accuracy = 100 * correct / total
        test_progress_bar.set_postfix({"Accuracy": f"{current_accuracy:.2f}%"})

# Output final test accuracy
test_accuracy = 100 * correct / total
print(f"Test set accuracy: {test_accuracy:.2f}% ({correct}/{total})")

# Single sample inference example
sample_data, sample_target = next(iter(test_loader))
sample_output = model(sample_data[:1])  # Take only the first sample
_, predicted_class = r.max(sample_output, 1)
print(f"Sample predicted class: {predicted_class.item()}, Actual class: {sample_target[0].item()}")

print("CNN training and inference testing completed!")
```

### Image Transformation Example

```python
from riemann.vision.transforms import Compose, ToTensor, Normalize, Resize, CenterCrop, RandomHorizontalFlip
from PIL import Image

# Define transformation sequence
transform = Compose([
    Resize(256),               # Resize image
    CenterCrop(224),           # Center crop
    RandomHorizontalFlip(),    # Random horizontal flip
    ToTensor(),                # Convert to tensor
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Load and transform image
image = Image.open("example.jpg")
transformed_image = transform(image)
print(f"Transformed image shape: {transformed_image.shape}")  # Output: (3, 224, 224)
```

## Example Descriptions

Riemann provides rich example code located in the `examples/` directory:

- **backward_demo.py**: Demonstration of backward function usage
- **grad_demo.py**: Demonstration of grad function usage
- **hessian.py**: Hessian matrix computation example
- **jacobian.py**: Jacobian matrix computation example
- **nn_MNIST_CE_SGD.py**: Handwritten digit recognition neural network example based on MNIST
- **cnn_CIFAR10_CE_SGD.py**: Convolutional neural network example based on CIFAR10
- **optimizers_comparison.py**: Optimizer performance comparison
- **scatter.py**: Tensor scatter operation example

## Testing Methods

### Running Tests in Batches with pytest

Riemann uses pytest as its testing framework. You can use the following commands to run all tests in batches:

```bash
# Run all test files in the tests directory
pytest tests

# Run specific test file
pytest tests/test_010_grad.py

# Run specific test class or method
pytest tests/test_011_jacobian.py::TestJacobianFunctions::test_single_input_single_output

# Run tests and generate coverage report
pytest --cov=riemann tests/

# Run vision module tests
pytest tests/test_052_vision.py
```

### Running Test Scripts Individually

You can also run test scripts individually:

```bash
cd tests
python test_010_grad.py
# Run other test scripts
```

## Technical Features

- **Efficient Implementation**: Optimized automatic differentiation algorithms
- **Easy-to-use API**: Concise and clear interface design
- **Flexible Extension**: Supports custom operations and derivative rules
- **Comprehensive Testing**: Full unit test coverage
- **PyTorch Compatible**: Highly compatible with PyTorch interfaces
- **Computer Vision Support**: Provides commonly used datasets and image transformation functions

## Application Scenarios

- **Deep Learning Research**: Custom model and algorithm development
- **Scientific Computing**: Gradient computation for complex mathematical models
- **Optimization Problem Solving**: Gradient descent and Adam optimization algorithms
- **Computer Vision**: Image classification, object detection, and other vision tasks
- **Education and Teaching**: Learning automatic differentiation and deep learning principles

## Contributing Guide

Issues and Pull Requests are welcome! Before contributing code, please ensure all tests pass.

## Third-party Dependencies and Licenses

### Core Dependencies

| Library | Version Requirement | License Type | Notes                                     |
|---------|---------------------|--------------|-------------------------------------------|
| NumPy   | >=1.20.0            | BSD 3-Clause | Core numerical computation library        |
| SciPy   | >=1.7.0             | BSD 3-Clause | Linear algebra algorithms (LU, SVD, etc.) |

### Testing Dependencies

| Library  | Version Requirement | Purpose           | License Type  | Notes               |
|----------|---------------------|-------------------|---------------|---------------------|
| PyTorch  | >=2.0.0             | Result comparison | BSD 3-Clause  | Used for verifying  |
|          |                     | validation        |               | calculation results |
| pytest   | >=7.0.0             | Testing framework | MIT           | Used for organizing |
|          |                     |                   |               | and running tests   |

### Vision Dependencies

| Library | Version Requirement | Purpose          | License Type | Notes               |
|---------|---------------------|------------------|--------------|---------------------|
| Pillow  | >=8.0.0             | Image processing | BSD 3-Clause | Used for image      |
|         |                     |                  |              | loading/saving      |

*Note: This project also utilizes Python's standard library components (like unittest) for testing, which don't require separate installation.*

*Note: Details of the BSD 3-Clause license for NumPy, PyTorch and Pillow can be found on their official websites.*

## License

This project is licensed under the BSD 3-Clause License. See the LICENSE file for details.

## Contributing

Issues and pull requests are welcome. Please ensure all contributions comply with the project's coding standards.

## Contact

Author: Fei Xiang
Email: xfeix@outlook.com
Gitee: https://gitee.com/xfcode2021