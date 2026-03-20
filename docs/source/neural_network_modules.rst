How to Build a Neural Network
===============================

Riemann provides a comprehensive set of neural network modules through the ``riemann.nn`` package. These modules are building blocks for creating and training neural networks.

This section provides a step-by-step guide on how to build, train, and evaluate a complete neural network using Riemann. We will use **MNIST handwritten digit recognition** as an example to demonstrate the entire process from data preparation to model evaluation.

Step 1: Data Preparation
------------------------

Before building a neural network, you need to prepare your dataset. Riemann provides the ``Dataset`` and ``DataLoader`` interfaces for data loading and processing.

Understanding Dataset
~~~~~~~~~~~~~~~~~~~~~

``Dataset`` is an abstract base class used to represent a dataset. It defines two core methods that subclasses must implement:

- ``__len__()``: Returns the number of samples in the dataset
- ``__getitem__(idx)``: Returns a sample based on the index

**Why use Dataset?**

The Dataset abstraction provides a unified interface for accessing data, allowing the training loop to handle different data sources (images, text, audio, etc.) in the same way. It also enables lazy loading, where data is only loaded into memory when needed.

**Using Built-in Datasets vs. Custom Datasets**

Riemann provides built-in datasets for common tasks. For computer vision tasks, you can use datasets from ``riemann.vision.datasets``:

.. code-block:: python

    from riemann.vision.datasets import MNIST

    # Use Riemann's built-in MNIST dataset
    train_dataset = MNIST(root='./data', train=True, transform=transform)

If you need to use your own data, you can create a custom dataset by inheriting from ``Dataset``:

.. code-block:: python

    from riemann.utils.data import Dataset
    import riemann as rm

    class MyCustomDataset(Dataset):
        """Example of a custom dataset for structured data"""
        
        def __init__(self, data_path):
            # Load your data here
            self.data = rm.load(data_path)  # Example: load from file
            self.labels = rm.load_labels(data_path)
        
        def __len__(self):
            """Return the total number of samples"""
            return len(self.data)
        
        def __getitem__(self, idx):
            """Return a single sample (data, label)"""
            return self.data[idx], self.labels[idx]

**In this tutorial**, we use Riemann's built-in MNIST dataset, which automatically downloads and manages the data for us.

Data Transformation with Transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``transforms`` is used for data preprocessing and augmentation. You can compose multiple transformations using ``transforms.Compose``:

.. code-block:: python

    from riemann.vision import transforms

    # Define data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),           # Convert image to tensor
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std
    ])

**Key Concepts:**

- ``ToTensor()``: Converts PIL Image or numpy array to tensor and scales pixel values from [0, 255] to [0.0, 1.0]. This normalization is important because neural networks work best with small input values (typically 0-1 or -1 to 1).

- ``Normalize(mean, std)``: Normalizes tensor with mean and standard deviation: ``output = (input - mean) / std``. For MNIST, the values (0.1307, 0.3081) are pre-computed statistics of the dataset. Normalization helps the network learn faster by ensuring all features are on a similar scale.

Loading MNIST Dataset
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Load training and test datasets
    train_dataset = MNIST(
        root='./data',      # Directory to store/load data
        train=True,         # True for training set, False for test set
        transform=transform # Data transformation to apply
    )
    
    test_dataset = MNIST(
        root='./data',
        train=False,
        transform=transform
    )
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

Using DataLoader for Batch Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``DataLoader`` is used for batch loading of data, supporting data shuffling and automatic batching:

.. code-block:: python

    from riemann.utils.data import DataLoader

    # Create DataLoader for training
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=100,     # Number of samples per batch
        shuffle=True        # Shuffle data at every epoch
    )
    
    # Create DataLoader for testing
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,       # Process one sample at a time for testing
        shuffle=False       # No need to shuffle test data
    )

**Why use DataLoader and batch training?**

Training on batches (mini-batches) instead of single samples offers several advantages:

1. **Computational Efficiency**: Processing multiple samples together allows for better utilization of hardware (CPU/GPU) through vectorized operations.

2. **Memory Efficiency**: You don't need to load the entire dataset into memory at once. DataLoader loads batches on-demand.

3. **More Stable Gradients**: Gradients computed from a batch are less noisy than those from a single sample, leading to more stable training.

4. **Generalization**: Batch training with shuffling helps the model generalize better by seeing data in different orders each epoch.

**Key Parameters:**

- ``dataset``: The dataset to load data from
- ``batch_size``: How many samples per batch to load. Common values are 32, 64, 128, or 256. Larger batches give more stable gradients but require more memory.
- ``shuffle``: Set to True to have the data reshuffled at every epoch. This prevents the model from learning the order of data and improves generalization.

Step 2: Building the Neural Network
-----------------------------------

Neural networks in Riemann are built by inheriting from ``nn.Module`` and implementing the ``forward`` method.

Understanding nn.Module
~~~~~~~~~~~~~~~~~~~~~~~

``nn.Module`` is the base class for all neural network modules. It provides:

- **Parameter Management**: Automatically tracks learnable parameters (weights and biases)
- **Submodule Management**: Supports nested modules, allowing complex architectures
- **Device Management**: Supports CPU/GPU execution with simple ``.to('cuda')`` calls
- **Training/Evaluation Modes**: ``train()`` and ``eval()`` methods control behaviors like dropout

Defining the Network Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For MNIST classification, we'll build a simple feedforward neural network:

.. code-block:: python

    import riemann.nn as nn
    import riemann.optim as opt

    class Classifier(nn.Module):
        """
        MNIST Handwritten Digit Classifier
        
        Network Architecture:
        - Input Layer: 784 neurons (28x28 pixels flattened)
        - Hidden Layer: 200 neurons with ReLU activation
        - Output Layer: 10 neurons (for digits 0-9)
        """
        def __init__(self):
            super().__init__()
            
            # Define network layers using Sequential container
            self.model = nn.Sequential(
                nn.Flatten(),           # Flatten (1, 28, 28) to (1, 784)
                nn.Linear(784, 200),    # Input to hidden layer
                nn.ReLU(),              # Activation function
                nn.Linear(200, 10)      # Hidden to output layer
            )
            
            # Define loss function for multi-class classification
            self.loss_func = nn.CrossEntropyLoss()
            
            # Define optimizer with Adam algorithm
            self.optimizer = opt.Adam(
                self.parameters(),      # Parameters to optimize
                lr=0.001,               # Learning rate
                betas=(0.9, 0.999),     # Coefficients for running averages
                weight_decay=0.0001     # L2 regularization
            )
        
        def forward(self, inputs):
            """
            Forward pass
            
            Args:
                inputs: Tensor of shape (batch_size, 1, 28, 28)
            
            Returns:
                Tensor of shape (batch_size, 10) - unnormalized logits
            """
            return self.model(inputs)

**Understanding Each Component:**

1. **nn.Sequential**: A container that executes modules in sequence. It's like a pipeline where data flows from the first layer to the last. This simplifies the forward pass definition.

2. **nn.Flatten**: Flattens the input tensor from (batch, 1, 28, 28) to (batch, 784). MNIST images are 28x28 pixels, but neural networks expect a 1D vector as input. Flatten reshapes the data without changing its values.

3. **nn.Linear**: A fully connected (dense) layer that applies the transformation ``y = xW^T + b``, where W is the weight matrix and b is the bias vector. 
   - ``nn.Linear(784, 200)`` means 784 inputs (flattened image) → 200 outputs (hidden neurons)
   - The network learns the optimal weights during training

4. **nn.ReLU (Activation Function)**: Rectified Linear Unit applies ``f(x) = max(0, x)``. Activation functions introduce non-linearity, allowing the network to learn complex patterns. Without activation functions, multiple linear layers would collapse into a single linear transformation.

5. **nn.CrossEntropyLoss**: The loss function measures how wrong the model's predictions are. Cross-entropy is ideal for multi-class classification because:
   - It penalizes confident wrong predictions heavily
   - It works directly with raw model outputs (logits), no need for softmax
   - It combines LogSoftmax and Negative Log-Likelihood for numerical stability

6. **Adam Optimizer**: Adam (Adaptive Moment Estimation) is an optimization algorithm that:
   - Adapts the learning rate for each parameter individually
   - Uses momentum to accelerate convergence
   - Combines the benefits of AdaGrad and RMSProp
   - The learning rate (lr=0.001) controls how big each update step is

Step 3: Training the Network
----------------------------

Training involves iterating over the dataset multiple times (epochs), computing predictions, calculating loss, and updating parameters.

How Parameters Learn: The Core Mechanism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Neural networks learn by iteratively adjusting their parameters (weights and biases) to minimize the loss function. Here's how it works:

1. **Forward Pass**: Input data flows through the network, producing predictions
2. **Loss Calculation**: Compare predictions with true labels to compute error
3. **Backward Pass (Backpropagation)**: Calculate gradients - how much each parameter contributed to the error
4. **Parameter Update**: Adjust parameters in the direction that reduces loss

This process repeats for thousands or millions of iterations until the model converges.

Training Step Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class Classifier(nn.Module):
        # ... __init__ and forward methods from above ...
        
        def train_step(self, inputs, targets):
            """
            Execute one training step
            
            Args:
                inputs: Batch of images, shape (batch_size, 1, 28, 28)
                targets: Batch of labels, shape (batch_size,)
            
            Returns:
                loss: Scalar loss value
            """
            # Forward pass: compute predictions
            outputs = self.forward(inputs)
            
            # Compute loss
            loss = self.loss_func(outputs, targets)
            
            # Backward pass: compute gradients
            self.optimizer.zero_grad(True)  # Clear previous gradients
            loss.backward()                  # Compute gradients
            
            # Update parameters
            self.optimizer.step()
            
            return loss

**Detailed Explanation of Each Step:**

1. **Forward Pass**: The input batch (100 images) goes through the network:
   - Flatten: (100, 1, 28, 28) → (100, 784)
   - Linear + ReLU: (100, 784) → (100, 200)
   - Linear: (100, 200) → (100, 10) - raw scores for each digit

2. **Loss Computation**: CrossEntropyLoss compares the predicted scores with the true labels. It produces a single number representing "how wrong" the predictions are.

3. **zero_grad(True)**: Clears gradients from the previous iteration. Gradients accumulate by default, so we must clear them before computing new ones.

4. **backward()**: Computes gradients of the loss with respect to all parameters using the chain rule of calculus. This tells us which direction to adjust each parameter to reduce loss.

5. **optimizer.step()**: Updates all parameters using the computed gradients. The optimizer applies the learning rate and any momentum/adaptive scaling.

Complete Training Loop
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create model instance
    model = Classifier()
    
    # Training configuration
    epochs = 3
    
    # Training loop
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        epoch_loss = 0.0
        num_batches = len(train_loader)
        
        # Iterate over batches
        for batch_idx, batch in enumerate(train_loader):
            img_tensors, target_tensors = batch
            
            # Execute training step
            loss = model.train_step(img_tensors, target_tensors)
            epoch_loss += loss.item()
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, '
                      f'Batch {batch_idx}/{num_batches}, '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate average loss for the epoch
        avg_loss = epoch_loss / num_batches
        print(f'Epoch {epoch+1}/{epochs} completed, Average Loss: {avg_loss:.4f}')

**Training Process Explained:**

An **epoch** is one complete pass through the entire training dataset. With 60,000 training images and a batch size of 100, we have 600 batches per epoch.

The **loss value** should decrease over time:
- High loss (2.0+) = model is guessing randomly
- Medium loss (0.5-1.0) = model is learning but still makes mistakes
- Low loss (0.1-0.3) = model is confident and mostly correct

Step 4: Evaluation and Inference
--------------------------------

After training, evaluate the model on the test set to measure its generalization performance.

Understanding Accuracy and Model Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**What is Accuracy?**

Accuracy is the percentage of correctly classified samples out of all samples. For MNIST with 10,000 test images, if the model correctly classifies 9,500, the accuracy is 95%.

**Factors Affecting Accuracy:**

1. **Model Architecture**: More layers/neurons can learn complex patterns but may overfit
2. **Training Duration**: Too few epochs = underfitting; too many = overfitting
3. **Learning Rate**: Too high = unstable training; too low = slow convergence
4. **Data Quality**: Clean, well-labeled data produces better models
5. **Regularization**: Techniques like weight_decay prevent overfitting
6. **Data Augmentation**: Transformations during training improve generalization

**Overfitting vs. Underfitting:**

- **Underfitting**: Training accuracy is low. The model is too simple or hasn't trained long enough.
- **Overfitting**: Training accuracy is high but test accuracy is low. The model memorized training data instead of learning general patterns.
- **Good Fit**: Both training and test accuracy are high and close to each other.

Evaluation Method
~~~~~~~~~~~~~~~~~

.. code-block:: python

    class Classifier(nn.Module):
        # ... previous methods ...
        
        def evaluate(self, dataloader):
            """
            Evaluate model performance
            
            Args:
                dataloader: DataLoader providing test data
            
            Returns:
                accuracy: Classification accuracy (0-1)
                avg_loss: Average loss over the dataset
            """
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in dataloader:
                img_tensors, target_tensors = batch
                
                # Forward pass
                outputs = self.forward(img_tensors)
                
                # Compute loss
                loss = self.loss_func(outputs, target_tensors)
                total_loss += loss.item()
                
                # Compute accuracy
                predicted = outputs.argmax(dim=1)  # Get predicted class
                total += target_tensors.size(0)
                correct += (predicted == target_tensors).sum().item()
            
            accuracy = correct / total
            avg_loss = total_loss / len(dataloader)
            return accuracy, avg_loss

**How Accuracy is Calculated:**

1. **outputs.argmax(dim=1)**: For each sample, find the index of the highest score. This index is the predicted digit (0-9).

2. **Compare with targets**: Check if ``predicted == target_tensors`` to get a boolean tensor of correct/incorrect predictions.

3. **Sum and divide**: Count correct predictions and divide by total samples to get accuracy.

Running Evaluation
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Set model to evaluation mode
    model.eval()
    
    # Evaluate on test set
    test_accuracy, test_loss = model.evaluate(test_loader)
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test Loss: {test_loss:.4f}')

**Key Points:**

- ``model.eval()``: Sets the model to evaluation mode. This disables dropout (if used) and batch normalization updates. It's crucial for consistent evaluation results.

- **Test vs. Training Performance**: Test accuracy is usually slightly lower than training accuracy. A small gap (1-3%) is normal. A large gap indicates overfitting.

- **Loss vs. Accuracy**: Loss measures confidence; accuracy measures correctness. A model can have high accuracy but high loss if it's uncertain about correct predictions, or low accuracy but low loss if it's confidently wrong.

Step 5: Complete Example
------------------------

Here is the complete runnable code for MNIST handwritten digit recognition:

.. code-block:: python

    import sys
    import os
    import time
    
    # Import Riemann modules
    import riemann.nn as nn
    import riemann.optim as opt
    from riemann.vision.datasets import MNIST
    from riemann.vision import transforms
    from riemann.utils.data import DataLoader


    class Classifier(nn.Module):
        """MNIST Handwritten Digit Classifier"""
        
        def __init__(self):
            super().__init__()
            
            # Network architecture
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 200),
                nn.ReLU(),
                nn.Linear(200, 10)
            )
            
            # Loss function and optimizer
            self.loss_func = nn.CrossEntropyLoss()
            self.optimizer = opt.Adam(
                self.parameters(),
                lr=0.001,
                betas=(0.9, 0.999),
                weight_decay=0.0001
            )
        
        def forward(self, inputs):
            return self.model(inputs)
        
        def train_step(self, inputs, targets):
            outputs = self.forward(inputs)
            loss = self.loss_func(outputs, targets)
            self.optimizer.zero_grad(True)
            loss.backward()
            self.optimizer.step()
            return loss
        
        def evaluate(self, dataloader):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in dataloader:
                img_tensors, target_tensors = batch
                outputs = self.forward(img_tensors)
                
                loss = self.loss_func(outputs, target_tensors)
                total_loss += loss.item()
                
                predicted = outputs.argmax(dim=1)
                total += target_tensors.size(0)
                correct += (predicted == target_tensors).sum().item()
            
            accuracy = correct / total
            avg_loss = total_loss / len(dataloader)
            return accuracy, avg_loss


    def main():
        print("MNIST Handwritten Digit Recognition")
        
        # Step 1: Data preparation
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        print("Loading datasets...")
        train_dataset = MNIST(root='./data', train=True, transform=transform)
        test_dataset = MNIST(root='./data', train=False, transform=transform)
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
        
        print(f"Training set size: {len(train_dataset)}")
        print(f"Test set size: {len(test_dataset)}")
        
        # Step 2: Create model
        print("\nInitializing model...")
        model = Classifier()
        
        # Step 3: Training
        print("\nStarting training...")
        epochs = 3
        train_start_time = time.time()
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = len(train_loader)
            
            for batch_idx, batch in enumerate(train_loader):
                img_tensors, target_tensors = batch
                loss = model.train_step(img_tensors, target_tensors)
                epoch_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, '
                          f'Batch {batch_idx}/{num_batches}, '
                          f'Loss: {loss.item():.4f}')
            
            avg_loss = epoch_loss / num_batches
            print(f'Epoch {epoch+1}/{epochs} completed, '
                  f'Average Loss: {avg_loss:.4f}')
            
            # Step 4: Evaluation
            model.eval()
            test_accuracy, test_loss = model.evaluate(test_loader)
            print(f'Test Accuracy: {test_accuracy:.4f}, '
                  f'Test Loss: {test_loss:.4f}')
            print('-' * 50)
        
        train_end_time = time.time()
        print(f"Total training time: {train_end_time - train_start_time:.2f} seconds")


    if __name__ == "__main__":
        main()

Expected Output
~~~~~~~~~~~~~~~

When you run the complete example, you should see output similar to:

.. code-block:: text

    MNIST Handwritten Digit Recognition
    Loading datasets...
    Training set size: 60000
    Test set size: 10000
    
    Initializing model...
    
    Starting training...
    Epoch 1/3, Batch 0/600, Loss: 2.3124
    Epoch 1/3, Batch 100/600, Loss: 0.5231
    Epoch 1/3, Batch 200/600, Loss: 0.3412
    Epoch 1/3, Batch 300/600, Loss: 0.2894
    Epoch 1/3, Batch 400/600, Loss: 0.2543
    Epoch 1/3, Batch 500/600, Loss: 0.1987
    Epoch 1/3 completed, Average Loss: 0.3124
    Test Accuracy: 0.9123, Test Loss: 0.2987
    --------------------------------------------------
    Epoch 2/3, Batch 0/600, Loss: 0.1876
    Epoch 2/3, Batch 100/600, Loss: 0.1654
    ...
    Test Accuracy: 0.9456, Test Loss: 0.1876
    --------------------------------------------------
    Epoch 3/3 completed
    Test Accuracy: 0.9567, Test Loss: 0.1456
    --------------------------------------------------
    Total training time: 45.23 seconds

**Interpreting the Results:**

- **Epoch 1**: Loss decreases from ~2.3 to ~0.3, accuracy ~91%. The model is learning basic patterns.
- **Epoch 2**: Loss ~0.18, accuracy ~94%. The model is refining its understanding.
- **Epoch 3**: Loss ~0.14, accuracy ~95%. The model has converged to a good solution.

A final accuracy of 95-97% is excellent for this simple network. More complex architectures (CNNs) can achieve 99%+.

Key Concepts Summary
--------------------

Dataset and DataLoader
~~~~~~~~~~~~~~~~~~~~~~

- **Dataset**: Abstract base class for data representation, requires ``__len__`` and ``__getitem__``. Use built-in datasets for common tasks or create custom datasets for your own data.
- **DataLoader**: Handles batching, shuffling, and loading data efficiently. Batch training improves computational efficiency and gradient stability.
- **Transforms**: Preprocessing pipeline for data augmentation and normalization. Essential for preparing data for neural network training.

Neural Network Components
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **nn.Module**: Base class for all neural network modules. Manages parameters and provides training infrastructure.
- **nn.Sequential**: Container for stacking layers sequentially. Simplifies forward pass definition.
- **nn.Flatten**: Reshapes multi-dimensional input (images) into 1D vectors for fully connected layers.
- **nn.Linear**: Fully connected layer that learns linear transformations. Core building block of neural networks.
- **nn.ReLU**: Activation function introducing non-linearity. Enables learning of complex patterns.
- **nn.CrossEntropyLoss**: Loss function for multi-class classification. Measures prediction error.
- **Optimizer (Adam)**: Algorithm for updating parameters based on gradients. Adam adapts learning rates per parameter.

Training Process
~~~~~~~~~~~~~~~~

- **Forward Pass**: Compute model predictions by propagating input through the network.
- **Loss Calculation**: Measure difference between predictions and targets using loss function.
- **Backward Pass**: Compute gradients via backpropagation to determine how to adjust parameters.
- **Parameter Update**: Optimizer adjusts parameters using gradients and learning rate.
- **Epoch**: One complete pass through the training dataset. Multiple epochs are needed for convergence.

Evaluation
~~~~~~~~~~

- **model.eval()**: Set model to evaluation mode (disables dropout, etc.).
- **argmax**: Get predicted class from output logits by selecting the highest score.
- **Accuracy**: Percentage of correct predictions. Test accuracy measures generalization to unseen data.
- **Overfitting**: When training accuracy is much higher than test accuracy. Use regularization to prevent it.

Module Class and Containers
===========================

All neural network modules in Riemann inherit from the ``nn.Module`` class, which is the foundation for building neural networks. This section details the core functionality, parameter management, and usage methods of various container classes.

Module Class Core Functionality
-------------------------------

The ``nn.Module`` class provides the following core functionalities:

- **Parameter Management**: Automatically tracks and manages learnable parameters
- **Submodule Management**: Supports nested submodules, forming hierarchical structures
- **Device Management**: Supports moving modules to different devices (CPU/GPU)
- **Forward Propagation**: Defines the data flow path through the network
- **State Management**: Supports training/evaluation mode switching

Module Class Main Methods
-------------------------

.. list-table:: Module Class Main Methods
   :widths: 20 30 50
   :header-rows: 1

   * - Method Name
     - Description
     - Usage Example
   * - ``__init__()``
     - Initialize the module, create core data structures
     - ``super(MyModule, self).__init__()``
   * - ``forward(*args, **kwargs)``
     - Define forward propagation logic, must be implemented by subclasses
     - ``def forward(self, x): return self.layer(x)``
   * - ``__call__(*args, **kwargs)``
     - Module call interface, internally calls forward method
     - ``output = model(input_data)``
   * - ``parameters(recurse=True)``
     - Return iterator over all parameters
     - ``for param in model.parameters(): print(param.shape)``
   * - ``named_parameters(prefix='', recurse=True)``
     - Return iterator over named parameters
     - ``for name, param in model.named_parameters(): print(name, param.shape)``
   * - ``buffers(recurse=True)``
     - Return iterator over all buffers
     - ``for buffer in model.buffers(): print(buffer.shape)``
   * - ``named_buffers(prefix='', recurse=True)``
     - Return iterator over named buffers
     - ``for name, buffer in model.named_buffers(): print(name, buffer.shape)``
   * - ``children()``
     - Return iterator over direct submodules
     - ``for child in model.children(): print(child)``
   * - ``modules()``
     - Return iterator over all submodules (including self)
     - ``for module in model.modules(): print(module)``
   * - ``named_modules(prefix='', recurse=True)``
     - Return iterator over named modules
     - ``for name, module in model.named_modules(): print(name, module)``
   * - ``train(mode=True)``
     - Set module to training mode
     - ``model.train()``
   * - ``eval()``
     - Set module to evaluation mode
     - ``model.eval()``
   * - ``to(device)``
     - Move module to specified device
     - ``model.to('cuda')``
   * - ``cuda()``
     - Move module to CUDA device
     - ``model.cuda()``
   * - ``cpu()``
     - Move module to CPU device
     - ``model.cpu()``
   * - ``zero_grad(set_to_none=False)``
     - Clear gradients of all parameters
     - ``model.zero_grad()``
   * - ``requires_grad_(requires_grad=True)``
     - Set whether parameters require gradients
     - ``model.requires_grad_(False)  # Freeze parameters``
   * - ``state_dict(destination=None, prefix='', keep_vars=False)``
     - Return module state dictionary
     - ``state = model.state_dict()``
   * - ``load_state_dict(state_dict)``
     - Load state dictionary into module
     - ``model.load_state_dict(state)``
   * - ``register_parameter(name, param)``
     - Register parameter to module
     - ``self.register_parameter('weight', nn.Parameter(rm.randn(10, 5)))``
   * - ``register_buffer(name, tensor)``
     - Register buffer to module
     - ``self.register_buffer('running_mean', rm.zeros(10))``
   * - ``add_module(name, module)``
     - Explicitly add submodule
     - ``self.add_module('linear', nn.Linear(10, 5))``

Creating Custom Modules
-----------------------

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    class MyNetwork(nn.Module):
        def __init__(self):
            super(MyNetwork, self).__init__()
            # Define submodules
            self.linear1 = nn.Linear(10, 50)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(50, 1)
        
        def forward(self, x):
            # Define forward propagation logic
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x
    
    # Create instance
    model = MyNetwork()
    print(model)

Container Classes
-----------------

Riemann provides several container classes to organize and manage modules:

Sequential
~~~~~~~~~~

The ``Sequential`` container executes modules in sequence, suitable for simple linear network structures:

**Parameters**:

- Accepts module lists or keyword arguments

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Method 1: Using module list
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # Method 2: Using keyword arguments
    model = nn.Sequential(
        linear1=nn.Linear(10, 20),
        relu=nn.ReLU(),
        linear2=nn.Linear(20, 5)
    )
    
    # Forward pass
    x = rm.randn(32, 10)
    output = model(x)
    print(output.shape)  # [32, 5]

ModuleList
~~~~~~~~~~

The ``ModuleList`` container stores module lists, allowing access by index, suitable for scenarios requiring dynamic control of forward propagation:

**Parameters**:

- ``modules``: Module list (optional)

**Main Methods**:

- ``append(module)``: Add module
- ``extend(modules)``: Extend module list
- ``insert(index, module)``: Insert module

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create module list
    layers = nn.ModuleList([
        nn.Linear(10, 20),
        nn.ReLU()
    ])
    
    # Add more modules
    layers.append(nn.Linear(20, 10))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(10, 5))
    
    # Forward pass
    x = rm.randn(32, 10)
    for i, layer in enumerate(layers):
        x = layer(x)
        print(f"After layer {i}: {x.shape}")
    
    print(f"Final output shape: {x.shape}")  # [32, 5]

ModuleDict
~~~~~~~~~~

The ``ModuleDict`` container uses a dictionary to store modules, allowing access by key, suitable for scenarios requiring selection of different modules based on conditions:

**Parameters**:

- ``modules``: Module dictionary (optional)

**Main Methods**:

- ``update(modules)``: Update module dictionary
- ``pop(key)``: Remove and return module with specified key

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create module dictionary
    layers = nn.ModuleDict({
        'linear1': nn.Linear(10, 20),
        'relu': nn.ReLU(),
        'linear2': nn.Linear(20, 5)
    })
    
    # Add new module
    layers.update({'dropout': nn.Dropout(p=0.5)})
    
    # Forward pass
    x = rm.randn(32, 10)
    x = layers['linear1'](x)
    x = layers['relu'](x)
    x = layers['dropout'](x)
    x = layers['linear2'](x)
    
    print(x.shape)  # [32, 5]

ParameterList
~~~~~~~~~~~~~

The ``ParameterList`` container is specifically designed for storing parameter lists:

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create parameter list
    params = nn.ParameterList([
        nn.Parameter(rm.randn(10, 20)),
        nn.Parameter(rm.randn(20))
    ])
    
    # Add more parameters
    params.append(nn.Parameter(rm.randn(20, 5)))
    
    # Index access
    weight = params[0]
    bias = params[1]

ParameterDict
~~~~~~~~~~~~~

The ``ParameterDict`` container is specifically designed for storing parameter dictionaries:

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create parameter dictionary
    params = nn.ParameterDict({
        'w1': nn.Parameter(rm.randn(10, 20)),
        'b1': nn.Parameter(rm.randn(20)),
        'w2': nn.Parameter(rm.randn(20, 5)),
        'b2': nn.Parameter(rm.randn(5))
    })
    
    # Access by key
    weight1 = params['w1']
    bias1 = params['b1']

Activation Functions
====================

Activation functions are important components in neural networks, introducing non-linear characteristics that enable networks to learn complex function mappings.

Activation Function List
------------------------

.. list-table:: Activation Functions Supported by Riemann
   :widths: 15 20 25 25 15
   :header-rows: 1

   * - Function Name
     - Description
     - Application Scenarios
     - Parameter Meanings
     - Notes
   * - ``ReLU``
     - Rectified Linear Unit, outputs max(0, x)
     - Default choice for most deep learning models
     - No parameters
     - May produce "dying neuron" problem
   * - ``LeakyReLU``
     - ReLU with leak, small slope in negative region
     - Solving ReLU's dying neuron problem
     - ``negative_slope``: Slope in negative region, default 0.01
     - Slightly higher computational cost than ReLU
   * - ``Sigmoid``
     - S-shaped activation function, outputs (0, 1)
     - Output layer in binary classification tasks
     - No parameters
     - Suffers from gradient vanishing problem
   * - ``Tanh``
     - Hyperbolic tangent function, outputs (-1, 1)
     - RNN and other sequence models
     - No parameters
     - Still has gradient vanishing problem, but less severe than Sigmoid
   * - ``Softmax``
     - Normalized exponential function, outputs probability distribution
     - Output layer in multi-class classification tasks
     - ``dim``: Calculation dimension, default -1
     - Usually used with cross-entropy loss
   * - ``GELU``
     - Gaussian Error Linear Unit
     - Default choice in Transformer models
     - No parameters
     - Higher computational cost

Loss Functions
==============

Loss functions are used to measure the difference between model predictions and true target values, and are core components of model training.

Loss Function List
------------------

.. list-table:: Loss Functions Supported by Riemann
   :widths: 15 20 25 25 15
   :header-rows: 1

   * - Function Name
     - Description
     - Application Scenarios
     - Parameter Meanings
     - Notes
   * - ``MSELoss``
     - Mean Squared Error loss
     - Regression tasks
     - ``reduction``: Aggregation method, default 'mean'
     - Sensitive to outliers
   * - ``L1Loss``
     - L1 loss (absolute error)
     - Regression tasks insensitive to outliers
     - ``reduction``: Aggregation method, default 'mean'
     - Gradient discontinuous at origin
   * - ``CrossEntropyLoss``
     - Cross entropy loss, combining log_softmax and nll_loss
     - Multi-class classification tasks
     - ``weight``: Class weights
       ``ignore_index``: Ignored target value
       ``reduction``: Aggregation method, default 'mean'
     - Input is raw logits, no need for softmax
   * - ``BCEWithLogitsLoss``
     - Binary cross entropy loss with logits
     - Binary classification tasks
     - ``weight``: Sample weights
       ``pos_weight``: Positive class weight
     - Input is raw logits, no need for sigmoid
   * - ``HuberLoss``
     - Huber loss, robust to outliers
     - Regression tasks sensitive to outliers
     - ``delta``: Threshold, default 1.0
     - Moderate computational cost

Basic Network Layers
====================

Linear Layer (Linear)
---------------------

Linear layer (also known as fully connected layer) performs affine transformation on input data:

**Parameters**:
- ``in_features``: Input feature dimension
- ``out_features``: Output feature dimension
- ``bias``: Whether to use bias, default True

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create linear layer
    linear = nn.Linear(in_features=20, out_features=10)
    
    # Forward pass
    x = rm.randn(32, 20)
    output = linear(x)
    print(output.shape)  # [32, 10]

Dropout Layer
-------------

Dropout layer prevents overfitting by randomly deactivating neurons:

**Parameters**:
- ``p``: Dropout probability, default 0.5

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create dropout layer
    dropout = nn.Dropout(p=0.5)
    
    # Forward pass (training mode)
    x = rm.randn(4, 16)
    dropout.train()
    output_train = dropout(x)
    
    # Forward pass (evaluation mode)
    dropout.eval()
    output_eval = dropout(x)

Flatten Layer
-------------

Flatten layer flattens the input tensor:

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    flatten = nn.Flatten()
    
    # Flatten (batch, 1, 28, 28) to (batch, 784)
    x = rm.randn(32, 1, 28, 28)
    output = flatten(x)
    print(output.shape)  # [32, 784]
