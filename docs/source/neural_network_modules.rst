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

**Data Root Directory Management**

To simplify data storage location management, Riemann provides the ``get_data_root()`` utility function to get the project's data root directory:

.. code-block:: python

    from riemann.utils import get_data_root
    
    # Get the data root directory path
    data_root = get_data_root()
    print(f"Data root: {data_root}")
    # Example output: D:\\code\\Riemann\\data

This function automatically locates the ``data`` folder under the project root directory, avoiding the need to manually specify paths in different environments.

**Loading the Dataset**

.. code-block:: python

    from riemann.utils import get_data_root
    
    # Load training and test datasets
    train_dataset = MNIST(
        root=get_data_root(),  # Use utility function to get data root
        train=True,            # True for training set, False for test set
        transform=transform    # Data transformation to apply
    )
    
    test_dataset = MNIST(
        root=get_data_root(),
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
- **Hook Management**: Supports registering forward/backward hooks for debugging, feature extraction, and gradient modification

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
   * - ``register_forward_pre_hook(hook)``
     - Register forward pre-hook
     - ``handle = model.register_forward_pre_hook(my_hook)``
   * - ``register_forward_hook(hook)``
     - Register forward post-hook
     - ``handle = model.register_forward_hook(my_hook)``
   * - ``register_full_backward_pre_hook(hook)``
     - Register backward pre-hook
     - ``handle = model.register_full_backward_pre_hook(my_hook)``
   * - ``register_full_backward_hook(hook)``
     - Register backward post-hook
     - ``handle = model.register_full_backward_hook(my_hook)``
   * - ``apply(fn)``
     - Recursively apply function to all submodules
     - ``model.apply(init_weights)``
   * - ``get_parameter(target)``
     - Get parameter by name
     - ``param = model.get_parameter('layer1.weight')``
   * - ``get_submodule(target)``
     - Get submodule by name
     - ``module = model.get_submodule('layer1.conv1')``
   * - ``get_buffer(target)``
     - Get buffer by name
     - ``buffer = model.get_buffer('bn1.running_mean')``
   * - ``has_parameter(target)``
     - Check if parameter exists
     - ``if model.has_parameter('weight'): ...``
   * - ``has_buffer(target)``
     - Check if buffer exists
     - ``if model.has_buffer('running_mean'): ...``
   * - ``set_parameter(name, param)``
     - Set parameter by name
     - ``model.set_parameter('weight', new_param)``
   * - ``set_buffer(name, tensor)``
     - Set buffer by name
     - ``model.set_buffer('running_mean', new_tensor)``
   * - ``delete_parameter(target)``
     - Delete parameter by name
     - ``model.delete_parameter('old_weight')``
   * - ``delete_buffer(target)``
     - Delete buffer by name
     - ``model.delete_buffer('old_buffer')``
   * - ``copy()``
     - Create shallow copy of module
     - ``new_model = model.copy()``
   * - ``deepcopy()``
     - Create deep copy of module
     - ``new_model = model.deepcopy()``

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

Linear layer (also known as fully connected layer) performs affine transformation on input data. It is one of the most fundamental layers in neural networks.

**Purpose**:

- Implements linear transformation: ``output = input @ weight.T + bias``
- Commonly used for feature transformation, final classification layer, and dimension conversion in networks
- Basic building block for constructing Multi-Layer Perceptrons (MLP)

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

Dropout layer prevents overfitting by randomly deactivating neurons. It is a commonly used regularization technique.

**Purpose**:

- Prevents neural network overfitting and improves model generalization
- Randomly sets some neuron outputs to zero during training, forcing the network to learn more robust feature representations
- Commonly used after fully connected layers, especially in deep networks

**Parameters**:

- ``p``: Dropout probability, default 0.5, representing the probability of each neuron being dropped

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

Dropout2d Layer
---------------

Dropout2d layer randomly drops entire feature maps at the channel level, suitable for convolutional neural networks.

**Purpose**:

- Specifically designed for regularization of 2D convolutional feature maps (shape ``(N, C, H, W)``)
- Drops entire channels randomly rather than individual pixels, preserving spatial correlation of feature maps
- Commonly used after convolutional layers to prevent overfitting in CNNs

**Parameters**:

- ``p``: Dropout probability, default 0.5

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create Dropout2d layer
    dropout2d = nn.Dropout2d(p=0.5)
    
    # Forward pass (input shape [N, C, H, W])
    x = rm.randn(4, 16, 32, 32)
    dropout2d.train()
    output = dropout2d(x)
    print(output.shape)  # [4, 16, 32, 32]

Dropout3d Layer
---------------

Dropout3d layer randomly drops entire 3D feature maps at the channel level, suitable for 3D convolutional neural networks.

**Purpose**:

- Specifically designed for regularization of 3D convolutional feature maps (shape ``(N, C, D, H, W)``)
- Drops entire 3D feature volumes at the channel level
- Commonly used in 3D convolutional networks for video processing, 3D medical imaging, etc.

**Parameters**:

- ``p``: Dropout probability, default 0.5

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create Dropout3d layer
    dropout3d = nn.Dropout3d(p=0.5)
    
    # Forward pass (input shape [N, C, D, H, W])
    x = rm.randn(4, 16, 8, 32, 32)
    dropout3d.train()
    output = dropout3d(x)
    print(output.shape)  # [4, 16, 8, 32, 32]

Flatten Layer
-------------

Flatten layer flattens the input tensor within a specified dimension range.

**Purpose**:

- Flattens multi-dimensional tensors into 1D or lower-dimensional tensors, commonly used to connect convolutional and fully connected layers
- Preserves batch dimension while merging spatial and channel dimensions into feature vectors
- Bridge between convolutional and fully connected parts in CNN architectures

**Parameters**:

- ``start_dim``: Starting dimension for flattening, default 1
- ``end_dim``: Ending dimension for flattening, default -1

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    flatten = nn.Flatten()
    
    # Flatten (batch, 1, 28, 28) to (batch, 784)
    x = rm.randn(32, 1, 28, 28)
    output = flatten(x)
    print(output.shape)  # [32, 784]

BatchNorm1d Layer
-----------------

1D batch normalization layer, normalizes the channel dimension for 2D or 3D inputs.

**Purpose**:

- Accelerates neural network training convergence and allows larger learning rates
- Reduces sensitivity to initialization and improves training stability
- Provides some regularization effect, reducing dependence on Dropout
- Commonly used after fully connected layers or 1D convolutional layers

**Parameters**:

- ``num_features``: Number of features (channel count C)
- ``eps``: Small constant for numerical stability, default 1e-5
- ``momentum``: Momentum for running statistics, default 0.1
- ``affine``: Whether to use learnable affine parameters, default True
- ``track_running_stats``: Whether to track running mean and variance, default True

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create BatchNorm1d layer
    bn = nn.BatchNorm1d(num_features=100)
    
    # 2D input (N, C)
    x = rm.randn(20, 100)
    output = bn(x)
    print(output.shape)  # [20, 100]
    
    # 3D input (N, C, L)
    x = rm.randn(20, 100, 35)
    output = bn(x)
    print(output.shape)  # [20, 100, 35]

BatchNorm2d Layer
-----------------

2D batch normalization layer, normalizes the channel dimension for 4D inputs ``(N, C, H, W)``.

**Purpose**:

- Specifically designed for 2D convolutional neural networks, normalizes each channel's feature map
- Accelerates CNN training and improves model generalization
- Key component for building modern CNNs (e.g., ResNet, DenseNet)
- Usually placed after convolutional layers and before activation functions

**Parameters**:

- ``num_features``: Number of features (channel count C)
- ``eps``: Small constant for numerical stability, default 1e-5
- ``momentum``: Momentum for running statistics, default 0.1
- ``affine``: Whether to use learnable affine parameters, default True
- ``track_running_stats``: Whether to track running mean and variance, default True

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create BatchNorm2d layer
    bn = nn.BatchNorm2d(num_features=64)
    
    # 4D input (N, C, H, W)
    x = rm.randn(16, 64, 32, 32)
    output = bn(x)
    print(output.shape)  # [16, 64, 32, 32]

BatchNorm3d Layer
-----------------

3D batch normalization layer, normalizes the channel dimension for 5D inputs ``(N, C, D, H, W)``.

**Purpose**:

- Specifically designed for 3D convolutional neural networks, such as video processing and 3D medical image analysis
- Normalizes 3D feature volumes for each channel
- Important component of 3D CNN architectures (e.g., C3D, I3D)

**Parameters**:

- ``num_features``: Number of features (channel count C)
- ``eps``: Small constant for numerical stability, default 1e-5
- ``momentum``: Momentum for running statistics, default 0.1
- ``affine``: Whether to use learnable affine parameters, default True
- ``track_running_stats``: Whether to track running mean and variance, default True

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create BatchNorm3d layer
    bn = nn.BatchNorm3d(num_features=32)
    
    # 5D input (N, C, D, H, W)
    x = rm.randn(8, 32, 4, 16, 16)
    output = bn(x)
    print(output.shape)  # [8, 32, 4, 16, 16]

LayerNorm Layer
---------------

Layer normalization layer, normalizes all features of a single sample.

**Purpose**:

- Normalizes features of individual samples without relying on batch statistics
- Suitable for scenarios with batch size of 1 or dynamically changing batch sizes
- Core component of Transformer models, used as an alternative to BatchNorm
- Widely used in natural language processing tasks

**Parameters**:

- ``normalized_shape``: Dimensions to normalize, can be an integer or tuple
- ``eps``: Small constant for numerical stability, default 1e-5
- ``affine``: Whether to use learnable affine parameters, default True

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create LayerNorm layer
    ln = nn.LayerNorm(normalized_shape=128)
    
    # Input can be any shape, last dimension must match normalized_shape
    x = rm.randn(20, 128)
    output = ln(x)
    print(output.shape)  # [20, 128]
    
    # Multi-dimensional input
    x = rm.randn(20, 10, 128)
    output = ln(x)
    print(output.shape)  # [20, 10, 128]

Embedding Layer
---------------

Embedding layer, converts integer indices to fixed-size dense vector representations.

**Purpose**:

- Maps discrete integer indices (such as word indices) to continuous vector representations
- Basic component for processing categorical features and sequential data (such as text, user IDs)
- Used as word embedding layer in NLP tasks
- Supports padding index (padding_idx) not participating in gradient computation

**Parameters**:

- ``num_embeddings``: Number of embedding vectors (vocabulary size)
- ``embedding_dim``: Dimension of each embedding vector
- ``padding_idx``: Padding index, embedding vectors at this index do not participate in gradient computation, default None
- ``max_norm``: Maximum norm of embedding vectors, re-normalized if exceeded, default None
- ``norm_type``: p-value for norm calculation, default 2 (L2 norm)
- ``scale_grad_by_freq``: Whether to scale gradients by frequency, default False

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create Embedding layer, vocabulary size 10000, embedding dimension 128
    embedding = nn.Embedding(num_embeddings=10000, embedding_dim=128)
    
    # Input is integer indices
    input_indices = rm.tensor([1, 5, 10, 100])
    output = embedding(input_indices)
    print(output.shape)  # [4, 128]
    
    # Using padding_idx
    embedding_with_pad = nn.Embedding(10000, 128, padding_idx=0)
    input_with_pad = rm.tensor([0, 1, 2, 0])  # 0 is the padding index
    output = embedding_with_pad(input_with_pad)

Module Hook Management
======================

Riemann provides a powerful module hook mechanism that allows users to insert custom logic during the forward and backward propagation of modules. The hook mechanism is a powerful tool for debugging, monitoring, and modifying network behavior.

Hook Types Overview
-------------------

Riemann supports four types of module hooks, executed at different stages of forward and backward propagation:

.. list-table::
   :header-rows: 1
   :widths: 25 30 30 30

   * - Hook Type
     - Registration Method
     - Execution Timing
     - Modifiable Value
   * - Forward Pre-Hook
     - ``register_forward_pre_hook``
     - Called before ``forward`` method execution
     - Module input (``input``)
   * - Forward Hook
     - ``register_forward_hook``
     - Called after ``forward`` method execution
     - Module output (``output``)
   * - Full Backward Pre-Hook
     - ``register_full_backward_pre_hook``
     - Called when all module outputs requiring gradients have received gradients
     - Output gradients (``grad_output``)
   * - Full Backward Hook
     - ``register_full_backward_hook``
     - Called when all module inputs requiring gradients have received gradients
     - Input gradients (``grad_input``)

Hook Execution Order
~~~~~~~~~~~~~~~~~~~~

Hook execution order during forward propagation:

.. code-block:: text

    register_forward_pre_hook → forward → register_forward_hook

Hook execution order during backward propagation:

.. code-block:: text

    register_full_backward_pre_hook → (compute grad_input) → register_full_backward_hook

Forward Pre-Hook (register_forward_pre_hook)
--------------------------------------------

**Purpose**:

- Modify or inspect input data before the module's forward computation
- Implement input preprocessing, data validation, or debug information printing
- Commonly used for dynamically adjusting input ranges, adding noise, or recording intermediate states

**Hook Function Signature**:

.. code-block:: python

    hook(module, input) -> None or modified input

**Parameters**:

- ``module``: The module instance being called
- ``input``: A tuple containing all input tensors (even a single input is wrapped in a tuple)

**Return Value**:

- ``None``: Indicates no modification to input, continue execution with original input
- ``Tensor`` or ``tuple``: Returns modified input, which will replace the original input passed to ``forward``

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # Define forward pre-hook: print input information
    def print_input_hook(module, input):
        print(f"Input shape for module {module._get_name()}: {input[0].shape}")
        return None  # Do not modify input

    # Define forward pre-hook: modify input
    def double_input_hook(module, input):
        # Multiply input by 2
        return (input[0] * 2,)

    # Create linear layer and register hooks
    linear = nn.Linear(10, 5)
    handle1 = linear.register_forward_pre_hook(print_input_hook)
    handle2 = linear.register_forward_pre_hook(double_input_hook)

    # Forward propagation
    x = rm.ones(2, 10)
    output = linear(x)  # Actually uses x * 2

    # Remove hooks
    handle1.remove()
    handle2.remove()

Forward Hook (register_forward_hook)
------------------------------------

**Purpose**:

- Modify or inspect output data after the module's forward computation
- Implement feature extraction, output monitoring, and debugging
- Commonly used for recording intermediate layer features and analyzing activation distributions

**Hook Function Signature**:

.. code-block:: python

    hook(module, input, output) -> None or modified output

**Parameters**:

- ``module``: The module instance being called
- ``input``: A tuple containing all input tensors passed to ``forward``
  
  - **Always a tuple**: Even for single-input modules, ``input`` is a tuple with one element: ``(input_tensor,)``
  - Multi-input modules: ``(input1, input2, ...)``
  - **Note**: If a forward pre-hook modified the input, this will be the modified version, not the original input
  
- ``output``: The return value of ``forward`` method
  
  - Single-output modules: A single tensor
  - Multi-output modules: A tuple of tensors ``(output1, output2, ...)``

**Return Value**:

- ``None``: Indicates no modification to output, use original output as module return value
- ``Tensor`` or ``tuple``: Returns modified output, which will replace the original output
  
  - For single-output modules, return a tensor
  - For multi-output modules, return a tuple with the same structure

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # Define forward hook: feature extractor
    class FeatureExtractor:
        def __init__(self):
            self.features = []
        
        def hook(self, module, input, output):
            self.features.append(output.clone())
            return None

    # Create model and register feature extraction hook
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

    extractor = FeatureExtractor()
    handle = model[0].register_forward_hook(extractor.hook)

    # Forward propagation
    x = rm.randn(4, 784)
    output = model(x)

    # View extracted features
    print(f"First layer output shape: {extractor.features[0].shape}")

    # Remove hook
    handle.remove()

Full Backward Pre-Hook (register_full_backward_pre_hook)
--------------------------------------------------------

**Purpose**:

- Modify or inspect output gradients (``grad_output``) at the start of backward propagation
- Implement gradient clipping, gradient scaling, or gradient monitoring
- Commonly used for preventing gradient explosion and adjusting gradient flow

**Hook Function Signature**:

.. code-block:: python

    hook(module, grad_output) -> None or modified grad_output

**Parameters**:

- ``module``: The module instance in backward propagation
- ``grad_output``: A tuple containing all output gradients
  
  - Single-output module: ``(grad_output_tensor,)``
  - Multi-output module: ``(grad_output1, grad_output2, ...)``
  - For outputs that don't require gradients, the corresponding position is ``None``

**Return Value**:

- ``None``: Indicates no modification to gradients, continue computation with original ``grad_output``
- ``tuple``: Returns modified ``grad_output``, which will be used for subsequent gradient computation
  
  **Important**: If you only want to modify some gradients, the returned tuple must contain **all** output gradients. For positions you don't want to modify, return the original gradient value; if you return ``None`` for a position, that gradient will be **zeroed** (set to 0)

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # Define backward pre-hook: gradient clipping
    def clip_grad_hook(module, grad_output):
        # Clip gradients to prevent explosion
        clipped = tuple(
            g.clip(-1, 1) if g is not None else None 
            for g in grad_output
        )
        return clipped

    # Define backward pre-hook: print gradient information
    def print_grad_hook(module, grad_output):
        print(f"Output gradient shape: {grad_output[0].shape}")
        print(f"Output gradient value range: [{grad_output[0].min()}, {grad_output[0].max()}]")
        return None

    # Create linear layer and register hook
    linear = nn.Linear(10, 5)
    handle = linear.register_full_backward_pre_hook(clip_grad_hook)

    # Forward and backward propagation
    x = rm.randn(2, 10)
    x.requires_grad = True
    output = linear(x)
    output.sum().backward()  # Gradients will be clipped to [-1, 1] range

    # Remove hook
    handle.remove()

Full Backward Hook (register_full_backward_hook)
------------------------------------------------

**Purpose**:

- Modify or inspect input gradients (``grad_input``) at the end of backward propagation
- Implement gradient monitoring, debugging, and visualization
- Commonly used for analyzing gradient flow and detecting vanishing or exploding gradients

**Hook Function Signature**:

.. code-block:: python

    hook(module, grad_input, grad_output) -> None or modified grad_input

**Parameters**:

- ``module``: The module instance in backward propagation
- ``grad_input``: A tuple containing all input gradients
  
  - Single-input module: ``(grad_input_tensor,)``
  - Multi-input module: ``(grad_input1, grad_input2, ...)``
  - For inputs that don't require gradients, the corresponding position is ``None``

- ``grad_output``: A tuple containing all output gradients
  
  - **Note**: If a backward pre-hook modified the gradients, this will be the modified version

**Return Value**:

- ``None``: Indicates no modification to gradients, continue propagation with original ``grad_input``
- ``tuple``: Returns modified ``grad_input``, which will replace the original gradients propagated to the previous layer
  
  **Important**: If you only want to modify some gradients, the returned tuple must contain **all** input gradients. For positions you don't want to modify, return the original gradient value; if you return ``None`` for a position, that gradient will be **zeroed** (set to 0)
  
  .. note::
     This behavior differs from PyTorch. In PyTorch, returning ``None`` for a position keeps the gradient as ``None``.
     Riemann chooses to zero out the gradient for the following reasons:
     
     1. **Semantic Consistency**: Consistent with backward pre-hook behavior (returning ``None`` means zeroing)
     2. **Practicality**: Zeroing is an intuitive way to block gradient propagation, while ``None`` requires extra handling
     3. **Safety**: A gradient of 0 is a valid numeric value that won't cause errors in subsequent computations

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # Define backward hook: gradient monitor
    class GradientMonitor:
        def __init__(self):
            self.gradients = []
        
        def hook(self, module, grad_input, grad_output):
            self.gradients.append({
                'module': module._get_name(),
                'grad_input': [g.clone() if g is not None else None for g in grad_input],
                'grad_output': [g.clone() if g is not None else None for g in grad_output]
            })
            return None

    # Create model and register gradient monitoring hooks
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

    monitor = GradientMonitor()
    for layer in model:
        layer.register_full_backward_hook(monitor.hook)

    # Forward and backward propagation
    x = rm.randn(4, 784)
    x.requires_grad = True
    output = model(x)
    output.sum().backward()

    # View recorded gradient information
    for grad_info in monitor.gradients:
        print(f"Module: {grad_info['module']}")
        print(f"Input gradient shapes: {[g.shape if g is not None else None for g in grad_info['grad_input']]}")

Hook Registration and Removal
-----------------------------

**Registering Hooks**:

All hook registration methods return a ``RemovableHandle`` object that can be used to remove the hook later:

.. code-block:: python

    # Register hook and get handle
    handle = module.register_forward_hook(hook_function)
    
    # Remove hook using handle
    handle.remove()

**Using Context Managers**:

``RemovableHandle`` supports the context manager protocol, allowing automatic management of hook lifecycle using ``with`` statements:

.. code-block:: python

    with module.register_forward_hook(hook_function) as handle:
        # Hook is active within this scope
        output = module(input)
        # Hook is automatically removed when exiting the with block

**Managing Multiple Hooks**:

A module can register multiple hooks of the same type, which are executed in registration order:

.. code-block:: python

    def hook1(module, input):
        print("Hook 1")
        return None
    
    def hook2(module, input):
        print("Hook 2")
        return None
    
    module.register_forward_pre_hook(hook1)
    module.register_forward_pre_hook(hook2)
    
    # Execution order: hook1 -> hook2

Typical Application Scenarios
-----------------------------

**1. Feature Visualization**

Feature visualization is a common technique in deep learning to understand what patterns a neural network learns at different layers. By registering forward hooks on convolutional layers, you can capture and visualize intermediate feature maps.

**Use Cases**:

- Visualizing what features different convolutional filters detect (edges, textures, shapes)
- Debugging model behavior by inspecting intermediate representations
- Creating feature maps for research or presentation purposes

**Example**: Capturing and visualizing feature maps from a CNN (using real MNIST data)

.. code-block:: python

    import riemann.nn as nn
    from riemann.vision.datasets import EasyMNIST
    from riemann.utils import get_data_root
    import matplotlib.pyplot as plt

    # Load MNIST dataset
    print("Loading MNIST dataset...")
    train_dataset = EasyMNIST(root=get_data_root(), train=True, onehot_label=False)

    # Get a sample (handwritten digit image)
    sample_data, sample_label = train_dataset[0]
    print(f"Sample label: {int(sample_label)}")

    # Reshape flattened data back to 28x28 image
    sample_image = sample_data.reshape(28, 28)

    # Create a simple CNN for demonstration
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.fc = nn.Linear(32 * 28 * 28, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    # Dictionary to store activations
    activations = {}

    def get_activation(name):
        """Create a hook function that saves activations"""
        def hook(module, input, output):
            # Detach to avoid saving computation graph
            activations[name] = output.detach()
        return hook

    # Create model and register hooks
    model = SimpleCNN()
    model.conv1.register_forward_hook(get_activation('conv1'))
    model.conv2.register_forward_hook(get_activation('conv2'))

    # Forward pass with MNIST sample
    # Reshape to [batch_size, channels, height, width]
    input_image = sample_image.unsqueeze(0).unsqueeze(0)  # [1, 1, 28, 28]
    output = model(input_image)

    # Now activations['conv1'] contains the feature maps from conv1 layer
    # Shape: [1, 16, 28, 28] - 16 feature maps of size 28x28
    print(f"Conv1 activations shape: {activations['conv1'].shape}")
    print(f"Model prediction: {output.argmax(dim=1).item()}")

    # Visualize the first 8 feature maps from conv1
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(activations['conv1'][0, i].numpy(), cmap='viridis')
        ax.set_title(f'Filter {i}')
        ax.axis('off')
    plt.suptitle('Conv1 Layer Feature Maps', fontsize=14)
    plt.tight_layout()
    plt.show()

**2. Gradient Checking**

Gradient checking is essential for debugging training issues. Invalid gradients (NaN or Inf values) can cause training to fail silently or produce unexpected results. By using backward hooks, you can monitor gradients in real-time during training.

**Use Cases**:

- Detecting gradient explosion or vanishing gradients early
- Identifying which layers produce invalid gradients
- Automatically stopping training or adjusting learning rate when issues occur

**Example**: Comprehensive gradient monitoring with automatic training stop

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    class GradientChecker:
        """A comprehensive gradient checker that monitors for various issues"""
        
        def __init__(self, threshold=1e3):
            self.threshold = threshold  # Threshold for gradient explosion
            self.has_nan_inf = False
            self.layer_stats = {}
        
        def hook(self, module, grad_input, grad_output):
            module_name = module._get_name()
            
            # Check for NaN or Inf in grad_output
            for i, g in enumerate(grad_output):
                if g is not None:
                    if rm.isnan(g).any():
                        print(f"ERROR: NaN detected in {module_name} grad_output[{i}]")
                        self.has_nan_inf = True
                    if rm.isinf(g).any():
                        print(f"ERROR: Inf detected in {module_name} grad_output[{i}]")
                        self.has_nan_inf = True
                    
                    # Check for gradient explosion
                    grad_norm = g.norm().item()
                    if grad_norm > self.threshold:
                        print(f"WARNING: Gradient explosion in {module_name}: norm={grad_norm:.2f}")
            
            # Check grad_input as well
            for i, g in enumerate(grad_input):
                if g is not None:
                    if rm.isnan(g).any() or rm.isinf(g).any():
                        print(f"ERROR: Invalid gradient in {module_name} grad_input[{i}]")
                        self.has_nan_inf = True
            
            # Store statistics
            self.layer_stats[module_name] = {
                'grad_output_norms': [g.norm().item() if g is not None else 0 for g in grad_output],
                'grad_input_norms': [g.norm().item() if g is not None else 0 for g in grad_input]
            }
            
            return None  # Don't modify gradients, just monitor

    # Usage in training
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

    checker = GradientChecker(threshold=100.0)
    
    # Register hooks on all layers
    for layer in model:
        layer.register_full_backward_hook(checker.hook)

    # Training loop with gradient checking
    for epoch in range(10):
        # ... forward pass ...
        # loss = criterion(output, target)
        # loss.backward()
        
        # Check if gradients are valid before optimizer step
        if checker.has_nan_inf:
            print(f"Epoch {epoch}: Stopping due to invalid gradients")
            break
        
        # optimizer.step()

**3. Weight Statistics Monitoring**

Monitoring weight statistics during training helps understand how the network is learning. Sudden changes in weight distribution can indicate issues like poor initialization, learning rate problems, or overfitting.

**Use Cases**:

- Tracking weight distribution changes over training epochs
- Detecting dead neurons (weights stuck at zero)
- Identifying potential overfitting (weights growing too large)
- Validating proper weight initialization

**Example**: Comprehensive weight and activation monitoring

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    class TrainingMonitor:
        """Monitor weights, biases, and activations during training"""
        
        def __init__(self):
            self.history = []
        
        def forward_hook(self, module, input, output):
            """Monitor forward pass statistics"""
            stats = {
                'module': module._get_name(),
                'input_mean': input[0].mean().item() if input[0] is not None else 0,
                'output_mean': output.mean().item(),
                'output_std': output.std().item()
            }
            
            # Monitor weights if available
            if hasattr(module, 'weight') and module.weight is not None:
                w = module.weight.data
                stats.update({
                    'weight_mean': w.mean().item(),
                    'weight_std': w.std().item(),
                    'weight_min': w.min().item(),
                    'weight_max': w.max().item(),
                    'dead_neurons': (w.abs() < 1e-6).sum().item()  # Near-zero weights
                })
            
            # Monitor bias if available
            if hasattr(module, 'bias') and module.bias is not None:
                b = module.bias.data
                stats.update({
                    'bias_mean': b.mean().item(),
                    'bias_std': b.std().item()
                })
            
            self.history.append(stats)
            
            # Print warnings for potential issues
            if stats.get('weight_std', 0) > 10:
                print(f"WARNING: {stats['module']} weights have high std: {stats['weight_std']:.2f}")
            if stats.get('dead_neurons', 0) > 0:
                print(f"INFO: {stats['module']} has {stats['dead_neurons']} dead neurons")
        
        def print_summary(self):
            """Print summary of monitored statistics"""
            print("\n=== Training Monitor Summary ===")
            for stats in self.history[-5:]:  # Show last 5 records
                print(f"\n{stats['module']}:")
                if 'weight_mean' in stats:
                    print(f"  Weight: mean={stats['weight_mean']:.4f}, std={stats['weight_std']:.4f}")
                if 'bias_mean' in stats:
                    print(f"  Bias: mean={stats['bias_mean']:.4f}, std={stats['bias_std']:.4f}")
                print(f"  Activation: mean={stats['output_mean']:.4f}, std={stats['output_std']:.4f}")

    # Usage example
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

    monitor = TrainingMonitor()
    
    # Register forward hooks on all linear layers
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            layer.register_forward_hook(monitor.forward_hook)

    # During training, statistics are automatically collected
    # After training, view the summary
    # monitor.print_summary()

Important Notes
---------------

1. **Hook Return Values**

   If a hook doesn't need to modify data, it should return ``None`` to avoid unnecessary side effects. When modification is needed, return a tensor or tuple with the same structure as the input.

2. **Multi-Input/Multi-Output Module Handling**

   For modules with multiple inputs or outputs, hooks receive tuples containing all inputs/outputs:

   - **Multi-input modules**: ``input`` and ``grad_input`` are tuples containing all input tensors/gradients
   - **Multi-output modules**: ``output`` and ``grad_output`` are tuples containing all output tensors/gradients
   - **Important**:
     
     - For multi-output modules, **backward pre-hooks** are called only when **all output gradients** (outputs with requires_grad=True that participate in loss computation) are ready
     - For multi-input modules, **backward hooks** are called only when **all input gradients** (inputs with requires_grad=True) are ready
     
     This ensures the hook receives complete gradient information
   
   When modifying gradients, always return a complete tuple with the same structure, even if you only modify some elements.

3. **Multiple Hooks Chain**

   A module can register multiple hooks of the same type, which form a call chain in registration order:

   - **Forward Pre-Hook Chain**: The return value of the previous hook becomes the input to the next hook
     
     - If a hook returns ``None``: The original input is passed to the next hook
     - If a hook returns non-``None``: That return value is used as the next hook's input
     - The last hook's output determines the final input passed to ``forward``
   
   - **Forward Hook Chain**: The return value of the previous hook becomes the output to the next hook
     
     - If a hook returns ``None``: The original output is passed to the next hook
     - If a hook returns non-``None``: That return value is used as the next hook's output
     - The last hook's output determines the module's final return value
   
   - **Backward Pre-Hook Chain**: Can modify ``grad_output``
     
     - The ``grad_output`` received by a hook may be the modified version from the previous hook
     - If a hook returns ``None``: The current ``grad_output`` is used to compute ``grad_input``
     - If a hook returns non-``None``: That return value replaces ``grad_output`` and is used to compute ``grad_input``
   
   - **Backward Hook Chain**: Can modify ``grad_input``
     
     - The ``grad_input`` received by a hook is the computed input gradient
     - The ``grad_output`` received by a hook may be the modified version from backward pre-hooks
     - If a hook returns ``None``: The original ``grad_input`` is propagated to the previous layer
     - If a hook returns non-``None``: That return value replaces ``grad_input`` and is propagated to the previous layer

   .. code-block:: python

       # Multi-output module example
       class MultiOutputModule(nn.Module):
           def forward(self, x):
               return x * 2, x * 3  # Two outputs
       
       module = MultiOutputModule()
       
       def grad_hook(module, grad_input, grad_output):
           # grad_output contains gradients for BOTH outputs
           # This hook is called only when both output gradients are ready
           print(f"Output 1 grad shape: {grad_output[0].shape}")
           print(f"Output 2 grad shape: {grad_output[1].shape}")
           return None
       
       module.register_full_backward_hook(grad_hook)

3. **Gradient Computation Flow**

   Understanding the gradient computation flow helps correctly use backward hooks:

   - **Backward pre-hooks** (``register_full_backward_pre_hook``): Called before ``grad_input`` computation. Modifying ``grad_output`` affects how gradients are computed for module inputs
   - **Backward hooks** (``register_full_backward_hook``): Called after ``grad_input`` computation. Modifying ``grad_input`` affects gradients propagated to previous layers

   .. code-block:: text

       Backward propagation flow:
       
       1. Output gradients arrive from upstream
       2. register_full_backward_pre_hook called (can modify grad_output)
       3. Compute grad_input using (possibly modified) grad_output
       4. register_full_backward_hook called (can modify grad_input)
       5. Modified grad_input propagated to previous layers

4. **Hook Execution Conditions**

   Backward hooks have specific execution conditions to ensure meaningful gradient modification:

   - At least one module input must require gradients, **OR**
   - The module must have parameters that require gradients
   
   If neither condition is met, backward hooks won't be called because there's no gradient to modify.

5. **Performance Considerations**

   - Hooks add extra function call overhead. For production inference, remove all debugging and monitoring hooks
   - Avoid time-consuming operations in hooks, especially in training loops
   - When multiple hooks are registered on the same module, they execute sequentially, compounding the overhead

6. **Memory Management**

   - Be careful about memory leaks when saving tensor references in hooks. Saved tensors retain the computation graph
   - Always use ``.clone()`` or ``.detach()`` to create copies when storing tensors for later analysis
   - Cached gradients are automatically cleaned up after backward propagation completes

7. **Interaction with Computational Graph**

   When modifying gradients in hooks, be aware of the computational graph:

   - Modified gradients flow into subsequent computations
   - For gradient clipping, ensure the operation doesn't break gradient flow
   - For gradient monitoring, use ``.detach()`` to avoid affecting the graph

   .. code-block:: python

       # Safe gradient clipping (preserves gradient flow)
       def safe_clip_hook(module, grad_output):
           clipped = tuple(
               g.clip(-1, 1) if g is not None else None 
               for g in grad_output
           )
           return clipped
       
       # Safe gradient monitoring (doesn't affect graph)
       def safe_monitor_hook(module, grad_input, grad_output):
           # Detach before storing to avoid memory leak
           stored_grads = [g.detach().clone() if g is not None else None 
                          for g in grad_output]
           # ... analyze stored_grads ...
           return None

Convolutional Networks
======================

Convolutional Neural Networks (CNNs) are one of the most important and widely used architectures in deep learning, particularly suitable for processing grid-structured data such as images, videos, and sequential data. Riemann provides a complete set of convolutional network components, including 1D, 2D, and 3D convolution layers and pooling layers.

Convolution Layers
------------------

Convolution layers extract local feature patterns by sliding learnable convolutional kernels over input data. Riemann supports three dimensions of convolution operations:

.. list-table:: Convolution Layer Types
   :header-rows: 1
   :widths: 20 40 40

   * - Convolution Layer
     - Applicable Data Types
     - Typical Application Scenarios
   * - ``Conv1d``
     - 1D sequential data (N, C, L)
     - Audio processing, text sequences, time series
   * - ``Conv2d``
     - 2D image data (N, C, H, W)
     - Image classification, object detection, image segmentation
   * - ``Conv3d``
     - 3D volumetric data (N, C, D, H, W)
     - Video analysis, medical imaging, 3D reconstruction

Conv1d Layer
~~~~~~~~~~~~

**Purpose**:

- Process 1D sequential data such as audio waveforms, text sequences, and time series
- Capture local temporal dependencies and patterns
- Used for n-gram feature extraction in natural language processing

**Parameters**:

- ``in_channels``: Number of input channels
- ``out_channels``: Number of output channels (number of convolutional kernels)
- ``kernel_size``: Size of the convolutional kernel
- ``stride``: Convolution stride, default 1
- ``padding``: Padding size, default 0
- ``dilation``: Dilation rate, default 1
- ``groups``: Number of groups, default 1 (standard convolution)
- ``bias``: Whether to use bias, default True

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # Audio signal processing
    conv1d = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
    audio = rm.randn(8, 1, 1000)  # batch=8, channels=1, samples=1000
    output = conv1d(audio)
    print(output.shape)  # [8, 16, 1000]

Conv2d Layer
~~~~~~~~~~~~

**Purpose**:

- Core component of CNN architecture for extracting local image features
- Hierarchical feature extraction from low-level edge features to high-level semantic features
- Supports standard convolution, grouped convolution, depthwise separable convolution, etc.

**Parameters**:

- ``in_channels``: Number of input channels (e.g., 3 for RGB images)
- ``out_channels``: Number of output channels
- ``kernel_size``: Convolutional kernel size (integer or tuple)
- ``stride``: Convolution stride, default 1
- ``padding``: Padding size, default 0
- ``dilation``: Dilation rate for increasing receptive field, default 1
- ``groups``: Number of groups, default 1
- ``bias``: Whether to use bias, default True

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # Standard image convolution
    conv2d = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
    image = rm.randn(4, 3, 224, 224)  # batch=4, RGB, height=224, width=224
    output = conv2d(image)
    print(output.shape)  # [4, 64, 224, 224]

Conv3d Layer
~~~~~~~~~~~~

**Purpose**:

- Process 3D data such as videos and medical images (MRI, CT)
- Capture spatiotemporal features or 3D spatial features
- Simultaneously capture temporal and spatial correlations in video analysis

**Parameters**:

- ``in_channels``: Number of input channels
- ``out_channels``: Number of output channels
- ``kernel_size``: Convolutional kernel size (integer or triple tuple)
- ``stride``: Convolution stride, default 1
- ``padding``: Padding size, default 0
- ``dilation``: Dilation rate, default 1
- ``groups``: Number of groups, default 1
- ``bias``: Whether to use bias, default True

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # Video data processing
    conv3d = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    video = rm.randn(2, 3, 10, 64, 64)  # batch=2, RGB, frames=10, height=64, width=64
    output = conv3d(video)
    print(output.shape)  # [2, 16, 10, 64, 64]

Pooling Layers
--------------

Pooling layers are used to reduce the spatial dimensions of feature maps, decrease computational cost, and provide translation invariance. Riemann provides three types of pooling operations: max pooling, average pooling, and adaptive pooling.

.. list-table:: Pooling Layer Types
   :header-rows: 1
   :widths: 20 40 40

   * - Pooling Layer
     - Operation Type
     - Characteristics
   * - ``MaxPool1d/2d/3d``
     - Maximum value in window
     - Preserves salient features, robust to noise
   * - ``AvgPool1d/2d/3d``
     - Average value in window
     - Smooth downsampling, preserves overall information
   * - ``AdaptiveMaxPool1d/2d/3d``
     - Adaptive max pooling
     - Auto-computes pooling parameters, fixed output size
   * - ``AdaptiveAvgPool1d/2d/3d``
     - Adaptive average pooling
     - Auto-computes pooling parameters, fixed output size

Max Pooling Layers
~~~~~~~~~~~~~~~~~~

Max pooling layers select the maximum value within the pooling window, preserving the most salient features and providing robustness to noise. Riemann provides both standard max pooling and adaptive max pooling.

Standard Max Pooling
^^^^^^^^^^^^^^^^^^^^

MaxPool1d Layer
++++++++++++++++

**Purpose**:

- Apply 1D max pooling to sequence data, selecting the maximum value within the sliding window
- Reduce sequence dimensionality while preserving the most salient features
- Provide translation invariance for time series and sequential data

**Parameters**:

- ``kernel_size``: Pooling window size
- ``stride``: Pooling stride, defaults to kernel_size
- ``padding``: Padding size, default 0
- ``dilation``: Dilation rate, default 1
- ``ceil_mode``: Whether to use ceiling for output length calculation, default False
- ``return_indices``: Whether to return the indices of maximum values, default False

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # Sequence data downsampling
    maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
    features = rm.randn(4, 16, 100)  # batch=4, channels=16, length=100
    output = maxpool(features)
    print(output.shape)  # [4, 16, 50]

MaxPool2d Layer
++++++++++++++++

**Purpose**:

- Preserve the most salient features by selecting the maximum value in local regions
- Provide translation invariance
- Significantly reduce spatial dimensions and subsequent layer computational complexity

**Parameters**:

- ``kernel_size``: Pooling window size
- ``stride``: Pooling stride, defaults to kernel_size
- ``padding``: Padding size, default 0
- ``dilation``: Dilation rate, default 1
- ``ceil_mode``: Whether to use ceiling for output size calculation, default False
- ``return_indices``: Whether to return the indices of maximum values, default False

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # Standard image downsampling
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    features = rm.randn(4, 64, 224, 224)
    output = maxpool(features)
    print(output.shape)  # [4, 64, 112, 112]

MaxPool3d Layer
++++++++++++++++

**Purpose**:

- Apply 3D max pooling for volumetric data such as video and medical images
- Reduce 3D spatial dimensions while preserving the most salient spatiotemporal features
- Provide 3D translation invariance

**Parameters**:

- ``kernel_size``: Pooling window size (can be int or tuple of depth, height, width)
- ``stride``: Pooling stride, defaults to kernel_size
- ``padding``: Padding size, default 0
- ``dilation``: Dilation rate, default 1
- ``ceil_mode``: Whether to use ceiling for output size calculation, default False
- ``return_indices``: Whether to return the indices of maximum values, default False

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # Video data downsampling
    maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
    features = rm.randn(4, 3, 16, 64, 64)  # batch=4, channels=3, frames=16, height=64, width=64
    output = maxpool(features)
    print(output.shape)  # [4, 3, 8, 32, 32]

Adaptive Max Pooling
^^^^^^^^^^^^^^^^^^^^

Adaptive pooling layers automatically compute the pooling kernel size and stride based on the specified output size, ensuring the output dimensions are always fixed without manual calculation of pooling parameters.

AdaptiveMaxPool1d Layer
++++++++++++++++++++++++

**Purpose**:

- Apply 1D adaptive max pooling to sequence data
- Preserve the most salient features in sequences while mapping to fixed length
- Suitable for sequence tasks requiring preservation of maximum value information

**Parameters**:

- ``output_size``: Output sequence length
- ``return_indices``: Whether to return the indices of maximum values, default False

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # Adaptive max pooling
    adaptive_pool = nn.AdaptiveMaxPool1d(output_size=10)
    features = rm.randn(4, 16, 50)
    output = adaptive_pool(features)
    print(output.shape)  # [4, 16, 10]

AdaptiveMaxPool2d Layer
++++++++++++++++++++++++

**Purpose**:

- Apply 2D adaptive max pooling to image data
- Preserve the most salient features in local regions
- Suitable for vision tasks requiring preservation of spatial maximum value information

**Parameters**:

- ``output_size``: Output size, can be an integer tuple (H, W) or a single integer
- ``return_indices``: Whether to return the indices of maximum values, default False

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # Adaptive max pooling
    adaptive_pool = nn.AdaptiveMaxPool2d(output_size=(7, 7))
    features = rm.randn(4, 64, 224, 224)
    output = adaptive_pool(features)
    print(output.shape)  # [4, 64, 7, 7]

AdaptiveMaxPool3d Layer
++++++++++++++++++++++++

**Purpose**:

- Apply 3D adaptive max pooling to 3D data
- Preserve the most salient features in 3D space
- Suitable for video analysis, medical images, and other 3D data processing

**Parameters**:

- ``output_size``: Output size, can be an integer tuple (D, H, W) or a single integer
- ``return_indices``: Whether to return the indices of maximum values, default False

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # 3D adaptive max pooling
    adaptive_pool = nn.AdaptiveMaxPool3d(output_size=(4, 7, 7))
    features = rm.randn(4, 32, 16, 64, 64)
    output = adaptive_pool(features)
    print(output.shape)  # [4, 32, 4, 7, 7]

Average Pooling Layers
~~~~~~~~~~~~~~~~~~~~~~

Average pooling layers compute the average value within the pooling window, providing smooth downsampling and preserving overall statistical information. Riemann provides both standard average pooling and adaptive average pooling.

Standard Average Pooling
^^^^^^^^^^^^^^^^^^^^^^^^

AvgPool1d Layer
++++++++++++++++

**Purpose**:

- Apply 1D average pooling to sequence data, computing the average within the sliding window
- Provide smooth downsampling for sequential data
- Preserve overall statistical information

**Parameters**:

- ``kernel_size``: Pooling window size
- ``stride``: Pooling stride, defaults to kernel_size
- ``padding``: Padding size, default 0
- ``ceil_mode``: Whether to use ceiling, default False
- ``count_include_pad``: Whether to include padding values in average calculation, default True
- ``divisor_override``: Custom divisor for average computation, default None

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # Smooth sequence downsampling
    avgpool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
    features = rm.randn(4, 16, 100)  # batch=4, channels=16, length=100
    output = avgpool(features)
    print(output.shape)  # [4, 16, 50]

AvgPool2d Layer
++++++++++++++++

**Purpose**:

- Provide smooth feature representation by computing the average of local regions
- More robust to noise compared to max pooling
- Preserve overall statistical information

**Parameters**:

- ``kernel_size``: Pooling window size
- ``stride``: Pooling stride, defaults to kernel_size
- ``padding``: Padding size, default 0
- ``ceil_mode``: Whether to use ceiling, default False
- ``count_include_pad``: Whether to include padding values in average calculation, default True

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # Smooth downsampling
    avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
    features = rm.randn(4, 64, 224, 224)
    output = avgpool(features)
    print(output.shape)  # [4, 64, 112, 112]

AvgPool3d Layer
++++++++++++++++

**Purpose**:

- Apply 3D average pooling for volumetric data such as video and medical images
- Provide smooth 3D downsampling while preserving overall spatiotemporal information
- More robust to noise compared to 3D max pooling

**Parameters**:

- ``kernel_size``: Pooling window size (can be int or tuple of depth, height, width)
- ``stride``: Pooling stride, defaults to kernel_size
- ``padding``: Padding size, default 0
- ``ceil_mode``: Whether to use ceiling, default False
- ``count_include_pad``: Whether to include padding values in average calculation, default True
- ``divisor_override``: Custom divisor for average computation, default None

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # 3D data smooth downsampling
    avgpool = nn.AvgPool3d(kernel_size=2, stride=2)
    features = rm.randn(4, 32, 16, 64, 64)  # batch=4, channels=32, depth=16, height=64, width=64
    output = avgpool(features)
    print(output.shape)  # [4, 32, 8, 32, 32]

Adaptive Average Pooling
^^^^^^^^^^^^^^^^^^^^^^^^

Adaptive pooling layers automatically compute the pooling kernel size and stride based on the specified output size, ensuring the output dimensions are always fixed without manual calculation of pooling parameters.

AdaptiveAvgPool1d Layer
++++++++++++++++++++++++

**Purpose**:

- Apply 1D adaptive average pooling to sequence data
- Map sequences of arbitrary length to a specified fixed length
- Commonly used in the output layer of sequence models to unify dimensions of different length sequences

**Parameters**:

- ``output_size``: Output sequence length, can be an integer or None (indicating maintaining original size)

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # Map sequences of different lengths to fixed length 10
    adaptive_pool = nn.AdaptiveAvgPool1d(output_size=10)
    
    # Input sequence length 50
    features = rm.randn(4, 16, 50)  # batch=4, channels=16, length=50
    output = adaptive_pool(features)
    print(output.shape)  # [4, 16, 10]
    
    # Input sequence length 100, output still 10
    features = rm.randn(4, 16, 100)
    output = adaptive_pool(features)
    print(output.shape)  # [4, 16, 10]

AdaptiveAvgPool2d Layer
++++++++++++++++++++++++

**Purpose**:

- Apply 2D adaptive average pooling to image data
- Map feature maps of arbitrary sizes to a specified fixed size
- Commonly used at the end of CNNs to convert image features of different sizes to fixed dimensions

**Parameters**:

- ``output_size``: Output size, can be an integer tuple (H, W) or a single integer (indicating square output)

**Usage Example**

MNIST Handwritten Digit Recognition Example
-------------------------------------------

Below is a complete CNN model example for MNIST handwritten digit recognition, including full training and inference workflows:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    import riemann.optim as opt
    from riemann.vision.datasets import MNIST
    from riemann.vision import transforms
    from riemann.utils.data import DataLoader
    from riemann import cuda

    class MNISTNet(nn.Module):
        """MNIST Handwritten Digit Recognition Network"""
        
        def __init__(self):
            super().__init__()
            # Feature extraction layers
            self.features = nn.Sequential(
                # First convolution: 1@28x28 -> 32@28x28
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 32@14x14
                
                # Second convolution: 32@14x14 -> 64@14x14
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 64@7x7
            )
            
            # Classifier
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 10)
            )
            
            # Loss function
            self.loss_func = nn.CrossEntropyLoss()
        
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
        
        def train_step(self, inputs, targets):
            """Single training step"""
            outputs = self.forward(inputs)
            loss = self.loss_func(outputs, targets)
            self.optimizer.zero_grad(True)
            loss.backward()
            self.optimizer.step()
            return loss
        
        def evaluate(self, dataloader, device):
            """Evaluate model performance"""
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in dataloader:
                img_tensors, target_tensors = batch
                # Move data to device
                img_tensors = img_tensors.to(device)
                target_tensors = target_tensors.to(device)
                
                outputs = self.forward(img_tensors)
                
                # Compute loss
                loss = self.loss_func(outputs, target_tensors)
                total_loss += loss.item()
                
                # Compute accuracy
                predicted = outputs.argmax(dim=1)
                total += target_tensors.size(0)
                correct += (predicted == target_tensors).sum().item()
            
            accuracy = correct / total
            avg_loss = total_loss / len(dataloader)
            return accuracy, avg_loss


    def main():
        """Main function: complete training and inference workflow"""
        print("MNIST Handwritten Digit Recognition CNN Example")
        
        # Check CUDA availability
        CUDA_AVAILABLE = cuda.CUPY_AVAILABLE
        device = 'cuda' if CUDA_AVAILABLE else 'cpu'
        print(f"Using device: {device}")
        
        # 1. Data preparation
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # Mean and std for MNIST
        ])
        
        # Load datasets
        train_dataset = MNIST(root='./data', train=True, transform=transform)
        test_dataset = MNIST(root='./data', train=False, transform=transform)
        
        # Create data loaders (batch size 256 for better efficiency)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        # 2. Create model and move to device
        model = MNISTNet()
        model.to(device)
        print(f"Model structure:\n{model}")
        
        # Initialize optimizer (after moving model to device)
        model.optimizer = opt.Adam(model.parameters(), lr=0.001)
        
        # 3. Train model
        num_epochs = 5
        print(f"\nStarting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                # Move data to device
                images = images.to(device)
                labels = labels.to(device)
                
                loss = model.train_step(images, labels)
                train_loss += loss.item()
                
                if batch_idx % 50 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], "
                          f"Batch [{batch_idx}/{len(train_loader)}], "
                          f"Loss: {loss.item():.4f}")
            
            # Evaluation phase
            model.eval()
            test_accuracy, test_loss = model.evaluate(test_loader, device)
            avg_train_loss = train_loss / len(train_loader)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] completed: "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Test Loss: {test_loss:.4f}, "
                  f"Test Accuracy: {test_accuracy*100:.2f}%")
        
        # 4. Inference demonstration
        print("\nInference demonstration:")
        model.eval()
        
        # Get a batch of test data
        test_images, test_labels = next(iter(test_loader))
        # Move data to device
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)
        
        # Forward propagation
        with rm.no_grad():
            outputs = model(test_images[:5])
            predictions = outputs.argmax(dim=1)
        
        print(f"Predictions: {predictions.tolist()}")
        print(f"True labels: {test_labels[:5].tolist()}")
        print(f"Prediction accuracy: {(predictions == test_labels[:5]).sum().item() / 5 * 100:.2f}%")

    if __name__ == "__main__":
        main()

CIFAR-10 Image Classification Example
-------------------------------------

Below is a complete CNN model example for CIFAR-10 image classification, including full training and inference workflows:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    import riemann.optim as opt
    from riemann.vision.datasets import CIFAR10
    from riemann.vision import transforms
    from riemann.utils.data import DataLoader
    from riemann import cuda

    class CIFAR10Net(nn.Module):
        """CIFAR-10 Image Classification Network (Simplified)"""
        
        def __init__(self):
            super().__init__()
            # Feature extraction layers (simplified, fewer conv layers)
            self.features = nn.Sequential(
                # First layer: 3@32x32 -> 32@16x16
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.25),
                
                # Second layer: 32@16x16 -> 64@8x8
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.25),
                
                # Third layer: 64@8x8 -> 128@4x4
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.25),
            )
            
            # Classifier
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 10)
            )
            
            # Loss function
            self.loss_func = nn.CrossEntropyLoss()
        
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
        
        def train_step(self, inputs, targets):
            """Single training step"""
            outputs = self.forward(inputs)
            loss = self.loss_func(outputs, targets)
            self.optimizer.zero_grad(True)
            loss.backward()
            self.optimizer.step()
            return loss
        
        def evaluate(self, dataloader, device):
            """Evaluate model performance"""
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in dataloader:
                img_tensors, target_tensors = batch
                # Move data to device
                img_tensors = img_tensors.to(device)
                target_tensors = target_tensors.to(device)
                
                outputs = self.forward(img_tensors)
                
                # Compute loss
                loss = self.loss_func(outputs, target_tensors)
                total_loss += loss.item()
                
                # Compute accuracy
                predicted = outputs.argmax(dim=1)
                total += target_tensors.size(0)
                correct += (predicted == target_tensors).sum().item()
            
            accuracy = correct / total
            avg_loss = total_loss / len(dataloader)
            return accuracy, avg_loss


    def main():
        """Main function: complete training and inference workflow"""
        print("CIFAR-10 Image Classification CNN Example")
        
        # Check CUDA availability
        CUDA_AVAILABLE = cuda.CUPY_AVAILABLE
        device = 'cuda' if CUDA_AVAILABLE else 'cpu'
        print(f"Using device: {device}")
        
        # 1. Data preparation
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels
        ])
        
        # Load datasets
        train_dataset = CIFAR10(root='./data', train=True, transform=transform)
        test_dataset = CIFAR10(root='./data', train=False, transform=transform)
        
        # Create data loaders (batch size 512 for better efficiency)
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

        # 2. Create model and move to device
        model = CIFAR10Net()
        model.to(device)
        print(f"Model structure:\n{model}")

        # Initialize optimizer (after moving model to device)
        model.optimizer = opt.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        # 3. Train model
        num_epochs = 5
        print(f"\nStarting training for {num_epochs} epochs...")
        
        best_accuracy = 0.0
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                # Move data to device
                images = images.to(device)
                labels = labels.to(device)
                
                loss = model.train_step(images, labels)
                train_loss += loss.item()
                
                if batch_idx % 50 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], "
                          f"Batch [{batch_idx}/{len(train_loader)}], "
                          f"Loss: {loss.item():.4f}")
            
            # Evaluation phase
            model.eval()
            test_accuracy, test_loss = model.evaluate(test_loader, device)
            avg_train_loss = train_loss / len(train_loader)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] completed: "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Test Loss: {test_loss:.4f}, "
                  f"Test Accuracy: {test_accuracy*100:.2f}%")
            
            # Save best model
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                print(f"  -> Best model updated! Accuracy: {best_accuracy*100:.2f}%")
        
        print(f"\nTraining completed! Best test accuracy: {best_accuracy*100:.2f}%")
        
        # 4. Inference demonstration
        print("\nInference demonstration:")
        model.eval()
        
        # Get a batch of test data
        test_images, test_labels = next(iter(test_loader))
        # Move data to device
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)
        
        # Class names
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Forward propagation
        with rm.no_grad():
            outputs = model(test_images[:5])
            predictions = outputs.argmax(dim=1)
        
        print(f"Predicted classes: {[classes[p] for p in predictions.tolist()]}")
        print(f"True classes: {[classes[t] for t in test_labels[:5].tolist()]}")
        print(f"Prediction accuracy: {(predictions == test_labels[:5]).sum().item() / 5 * 100:.2f}%")

    if __name__ == "__main__":
        main()

CNN Design Guidelines
---------------------

1. **Receptive Field Design**:
   
   - Stack multiple small convolutional kernels (e.g., 3x3) instead of large kernels to reduce parameters while maintaining the same receptive field
   - Use dilated convolution to increase receptive field without adding parameters

2. **Downsampling Strategy**:
   
   - Use pooling layers (MaxPool/AvgPool) or convolutions with stride > 1 for downsampling
   - Downsampling gradually reduces feature map size and increases feature channels to extract higher-level features

3. **Normalization and Regularization**:
   
   - Using BatchNorm after convolutional layers can accelerate training and improve model stability
   - Use Dropout to prevent overfitting

4. **Activation Function Selection**:
   
   - ReLU is the most commonly used activation function, computationally simple and effective for mitigating vanishing gradients
   - LeakyReLU or GELU may perform better in deep networks

Transformer
===========

Transformer is a deep learning architecture based on attention mechanisms, originally designed for natural language processing tasks but now widely applied in computer vision, speech recognition, and other fields. Riemann provides complete Transformer components compatible with PyTorch interfaces.

Transformer Architecture Overview
---------------------------------

Transformer consists of two parts: Encoder and Decoder:

- **Encoder**: Encodes the input sequence into a continuous representation (memory)
- **Decoder**: Generates output sequences autoregressively based on the encoder's output and previously generated target sequences

.. code-block:: text

    Input Sequence → [Encoder] → Memory → [Decoder] → Output Sequence
                          ↑_____________↓
                           Cross Attention

MultiheadAttention Mechanism
----------------------------

Multi-head attention is the core component of Transformer, allowing the model to simultaneously attend to information from different representation subspaces.

**Principle**:

Multi-head attention projects Query, Key, and Value inputs into multiple subspaces (heads), computes attention independently in each subspace, then concatenates the results and projects again:

.. code-block:: text

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W^O
    where head_i = Attention(Q @ W_i^Q, K @ W_i^K, V @ W_i^V)
    
    Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V

**Purpose**:

- Capture dependencies between different positions in sequences
- Self-attention mechanism allows each position to attend to all positions in the sequence
- Multi-head design enables the model to attend to different types of information

**Parameters**:

- ``embed_dim``: Input and output dimension
- ``num_heads``: Number of attention heads
- ``dropout``: Dropout probability for attention weights, default 0.0
- ``bias``: Whether to use bias, default True
- ``batch_first``: Whether input format is (batch, seq, feature), default False
- ``kdim``: Key dimension, default None (uses embed_dim)
- ``vdim``: Value dimension, default None (uses embed_dim)

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # Create multi-head attention layer
    mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

    # Input tensors
    batch_size, seq_len, embed_dim = 2, 10, 512
    query = rm.randn(batch_size, seq_len, embed_dim)
    key = rm.randn(batch_size, seq_len, embed_dim)
    value = rm.randn(batch_size, seq_len, embed_dim)

    # Forward propagation
    output, attn_weights = mha(query, key, value)
    print(f"Output shape: {output.shape}")  # [2, 10, 512]
    print(f"Attention weights shape: {attn_weights.shape}")  # [2, 10, 10]

Transformer Encoder
-------------------

The encoder consists of multiple identical encoder layers stacked together. Each encoder layer contains:

1. **Multi-head self-attention**: Processes relationships within the input sequence
2. **Feed-forward network**: Applies non-linear transformations independently to each position
3. **Residual connections and layer normalization**: Stabilizes training

**Two Normalization Modes**:

- **Post-LN** (default): Execute sublayer first, then normalize (original Transformer paper)
- **Pre-LN**: Normalize first, then execute sublayer (more stable training)

**Components**:

- ``TransformerEncoderLayer``: Single encoder layer
- ``TransformerEncoder``: Complete encoder composed of N encoder layers

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # Create encoder layer
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=512, nhead=8, dim_feedforward=2048, 
        dropout=0.1, batch_first=True
    )

    # Create encoder (6 layers)
    encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

    # Input sequence (batch=2, seq_len=10, d_model=512)
    src = rm.randn(2, 10, 512)
    
    # Forward propagation
    output = encoder(src)
    print(f"Encoder output shape: {output.shape}")  # [2, 10, 512]

Transformer Decoder
-------------------

The decoder consists of multiple identical decoder layers stacked together. Each decoder layer contains:

1. **Masked multi-head self-attention**: Prevents attending to future positions (autoregressive)
2. **Cross-attention**: Attends to encoder output (memory)
3. **Feed-forward network**: Non-linear transformation
4. **Residual connections and layer normalization**

**Components**:

- ``TransformerDecoderLayer``: Single decoder layer
- ``TransformerDecoder``: Complete decoder composed of N decoder layers

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # Create decoder layer
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=512, nhead=8, dim_feedforward=2048,
        dropout=0.1, batch_first=True
    )

    # Create decoder (6 layers)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

    # Target sequence (batch=2, tgt_len=20, d_model=512)
    tgt = rm.randn(2, 20, 512)
    
    # Encoder output (batch=2, src_len=10, d_model=512)
    memory = rm.randn(2, 10, 512)
    
    # Forward propagation
    output = decoder(tgt, memory)
    print(f"Decoder output shape: {output.shape}")  # [2, 20, 512]

Complete Transformer Model
--------------------------

Riemann provides a complete Transformer model containing both encoder and decoder.

**Parameters**:

- ``d_model``: Model dimension, default 512
- ``nhead``: Number of attention heads, default 8
- ``num_encoder_layers``: Number of encoder layers, default 6
- ``num_decoder_layers``: Number of decoder layers, default 6
- ``dim_feedforward``: Feed-forward network dimension, default 2048
- ``dropout``: Dropout probability, default 0.1
- ``activation``: Activation function, 'relu' or 'gelu', default 'relu'
- ``batch_first``: Input format, default False

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    # Create Transformer model
    transformer = nn.Transformer(
        d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
        dim_feedforward=2048, dropout=0.1, batch_first=True
    )

    # Source sequence (batch=2, src_len=10, d_model=512)
    src = rm.randn(2, 10, 512)
    
    # Target sequence (batch=2, tgt_len=20, d_model=512)
    tgt = rm.randn(2, 20, 512)
    
    # Forward propagation
    output = transformer(src, tgt)
    print(f"Transformer output shape: {output.shape}")  # [2, 20, 512]

Machine Translation Example
---------------------------

Below is a complete machine translation model example demonstrating the use of Transformer in training and inference:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn

    class TransformerTranslationModel(nn.Module):
        """Transformer Machine Translation Model"""
        
        def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, 
                     num_encoder_layers=6, num_decoder_layers=6, max_seq_len=100):
            super().__init__()
            self.d_model = d_model
            
            # Word embedding layers
            self.src_embedding = nn.Embedding(src_vocab_size, d_model)
            self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
            
            # Positional encoding (simplified as learnable parameters)
            self.pos_encoding = nn.Embedding(max_seq_len, d_model)
            
            # Transformer
            self.transformer = nn.Transformer(
                d_model=d_model, nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=2048, dropout=0.1,
                batch_first=True
            )
            
            # Output projection
            self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        
        def forward(self, src, tgt, src_mask=None, tgt_mask=None):
            """Training forward propagation"""
            # Add positional encoding
            src_pos = rm.arange(src.shape[1]).expand(src.shape[0], -1)
            tgt_pos = rm.arange(tgt.shape[1]).expand(tgt.shape[0], -1)
            
            src_emb = self.src_embedding(src) + self.pos_encoding(src_pos)
            tgt_emb = self.tgt_embedding(tgt) + self.pos_encoding(tgt_pos)
            
            # Transformer forward propagation
            output = self.transformer(src_emb, tgt_emb, src_mask=src_mask, tgt_mask=tgt_mask)
            
            # Project to vocabulary dimension
            logits = self.output_proj(output)
            return logits
        
        def generate(self, src, max_len=50, start_token=1, end_token=2):
            """Inference: Autoregressive generation of translation results"""
            self.eval()
            
            # Encode source sequence
            src_pos = rm.arange(src.shape[1]).expand(src.shape[0], -1)
            src_emb = self.src_embedding(src) + self.pos_encoding(src_pos)
            memory = self.transformer.encoder(src_emb)
            
            # Autoregressive generation
            tgt = rm.full((src.shape[0], 1), start_token, dtype=rm.int64)
            
            for _ in range(max_len):
                # Generate causal mask
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1])
                
                # Decode
                tgt_pos = rm.arange(tgt.shape[1]).expand(tgt.shape[0], -1)
                tgt_emb = self.tgt_embedding(tgt) + self.pos_encoding(tgt_pos)
                output = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                
                # Predict next token
                logits = self.output_proj(output[:, -1, :])
                next_token = logits.argmax(dim=-1, keepdim=True)
                
                # Add to sequence
                tgt = rm.concatenate([tgt, next_token], dim=1)
                
                # Check if end token is generated
                if (next_token == end_token).all():
                    break
            
            return tgt

    # Create model
    model = TransformerTranslationModel(
        src_vocab_size=10000, tgt_vocab_size=10000,
        d_model=512, nhead=8, num_encoder_layers=6
    )

    # Simulate training data
    src = rm.randint(0, 10000, (2, 20))  # Source sequence
    tgt = rm.randint(0, 10000, (2, 25))  # Target sequence

    # Training forward propagation
    logits = model(src, tgt)
    print(f"Training output shape: {logits.shape}")  # [2, 25, 10000]

    # Inference generation
    generated = model.generate(src, max_len=30)
    print(f"Generated sequence shape: {generated.shape}")

Differences Between Encoder and Decoder
---------------------------------------

.. list-table:: Encoder vs Decoder
   :header-rows: 1
   :widths: 25 35 40

   * - Characteristic
     - Encoder
     - Decoder
   * - **Attention Type**
     - Self-attention only
     - Self-attention + Cross-attention
   * - **Masking**
     - No mask (can see all input)
     - Causal mask (cannot see future positions)
   * - **Input**
     - Source sequence
     - Target sequence + Encoder output
   * - **Application Scenarios**
     - Text classification, sentiment analysis, feature extraction
     - Machine translation, text generation, summarization

Training and Inference Workflow
-------------------------------

**Training Phase**:

1. **Encoder** processes the entire source sequence to generate memory
2. **Decoder** receives the target sequence (teacher forcing) and encoder memory
3. Use causal mask to prevent decoder from attending to future positions
4. Compute loss and backpropagate

**Inference Phase**:

1. **Encoder** processes the source sequence to generate memory
2. **Decoder** generates autoregressively:
   
   - Start with start token
   - Generate next token based on previously generated tokens and encoder memory
   - Repeat until end token or maximum length

.. code-block:: text

    Training:
    Source: [I, love, you] ──→ Encoder ──→ Memory
    Target: [我, 爱, 你] ───→ Decoder ──→ Output
                                   ↑
                                   └── Memory (from Encoder)
    
    Inference:
    Source: [I, love, you] ──→ Encoder ──→ Memory
                                              ↓
    Generated: [我] ───────→ Decoder ──→ [爱]
       ↑                                      ↓
       └──────────────────────────────── [你]

Transformer Design Guidelines
-----------------------------

1. **Model Dimension Selection**:
   
   - ``d_model`` is typically 512 or 768, balancing model capacity and computational cost
   - ``num_heads`` should divide ``d_model`` evenly (e.g., 512/8=64)

2. **Layer Depth**:
   
   - Standard configuration is 6 encoder layers + 6 decoder layers
   - Encoder-only models (e.g., BERT) can use 12-24 layers
   - Decoder-only models (e.g., GPT) can use 12-96 layers

3. **Positional Encoding**:
   
   - Essential for Transformer as it has no inherent sequential information
   - Can use learnable positional embeddings or sinusoidal encoding
   - Some modern variants use Rotary Position Embedding (RoPE)

4. **Attention Mask Usage**:
   
   - ``src_mask``: Used when source sequence contains padding
   - ``tgt_mask``: Causal mask to prevent attending to future positions
   - ``memory_mask``: Controls which encoder positions decoder can attend to

5. **Optimization Tips**:
   
   - Use learning rate warmup to stabilize early training
   - Label smoothing can improve generalization
   - Gradient clipping prevents gradient explosion
