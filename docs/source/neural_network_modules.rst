How to Build a Neural Network
===============================

Riemann provides a comprehensive set of neural network modules through the ``riemann.nn`` package. These modules are building blocks for creating and training neural networks.

Quick Start
-----------

This section provides a step-by-step guide on how to build, train, and evaluate a complete neural network using Riemann, including dataset preparation, network construction, training process, and inference evaluation.

Dataset Preparation
~~~~~~~~~~~~~~~~~~~

Before building a neural network with Riemann, you first need to prepare your dataset. Riemann provides a `Dataset` interface for defining standard methods for data loading and processing.

Dataset Interface Introduction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Dataset` is an abstract base class used to represent a dataset. To create a custom dataset, you need to inherit from the `Dataset` class and implement the following two core methods:

- ``__len__()``: Returns the number of samples in the dataset
- ``__getitem__(idx)``: Returns a sample based on the index

Building Custom Datasets
^^^^^^^^^^^^^^^^^^^^^^^^

Here's a detailed example of creating a custom dataset:

.. code-block:: python

    import riemann as rm
    import numpy as np
    from riemann.utils.data import Dataset, DataLoader

    # Custom dataset class
    class SimpleDataset(Dataset):
        def __init__(self, num_samples=1000):
            """
            Initialize the dataset
            
            :param num_samples: Number of samples in the dataset
            """
            # Generate random input data
            self.inputs = rm.randn(num_samples, 10)
            # Generate corresponding target values (simple linear mapping)
            weights = rm.randn(10, 2)
            biases = rm.randn(2)
            self.targets = self.inputs @ weights + biases
            
        def __len__(self):
            """
            Return the number of samples in the dataset
            """
            return len(self.inputs)
        
        def __getitem__(self, idx):
            """
            Return a sample based on the index
            
            :param idx: Sample index
            :return: Tuple of input data and target value
            """
            return self.inputs[idx], self.targets[idx]

    # Create dataset instances
    train_dataset = SimpleDataset(1000)
    test_dataset = SimpleDataset(200)

    # Check dataset information
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # Get a single sample
    sample_input, sample_target = train_dataset[0]
    print(f"Sample input shape: {sample_input.shape}")
    print(f"Sample target shape: {sample_target.shape}")

Advanced Dataset Example
^^^^^^^^^^^^^^^^^^^^^^^^

Here's a more complex dataset example with data preprocessing and transformations:

.. code-block:: python

    class ImageDataset(Dataset):
        def __init__(self, image_paths, labels, transform=None):
            """
            Image dataset
            
            :param image_paths: List of image paths
            :param labels: List of labels
            :param transform: Data transformation function (optional)
            """
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            # This should be actual image loading code
            # For example purposes, we generate random data
            image = rm.randn(3, 32, 32)  # Simulate a 3x32x32 RGB image
            label = self.labels[idx]
            
            # Apply data transformation
            if self.transform:
                image = self.transform(image)
            
            return image, label

Using DataLoader
~~~~~~~~~~~~~~~~

`DataLoader` is used for batch loading of data, supporting multi-threaded data loading, data shuffling, and automatic batching.

DataLoader Parameter Description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`DataLoader` accepts the following main parameters:

- ``dataset``: Dataset instance to load
- ``batch_size``: Number of samples per batch, default 1
- ``shuffle``: Whether to shuffle data at the start of each epoch, default False
- ``num_workers``: Number of subprocesses for data loading, default 0 (main process)
- ``drop_last``: Whether to drop the last incomplete batch if dataset size isn't divisible by batch size, default False
- ``pin_memory``: Whether to copy loaded data to CUDA pinned memory for faster GPU transfer, default False
- ``timeout``: Data loading timeout, default 0
- ``worker_init_fn``: Function called when initializing each worker process, default None
- ``multiprocessing_context``: Multiprocessing context, default None

DataLoader Usage Example
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=2,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False,
        num_workers=1
    )

    # Iterate through DataLoader
    print("Iterating through training data loader:")
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Input shape {inputs.shape}, Target shape {targets.shape}")
        if batch_idx == 2:  # Print only first 3 batches
            break

    # Using in training loop
    print("\nUsing in training loop:")
    num_epochs = 2
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Training code here
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"  Batch {batch_idx}/{len(train_loader)}")
            # In actual training, forward pass, loss calculation, backpropagation, etc.
            break  # For example purposes, only execute one batch

Using pin_memory for Acceleration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If training with GPU, you can enable `pin_memory` to speed up data transfer:

.. code-block:: python

    # DataLoader optimized for GPU training
    gpu_train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True  # Enable pinned memory
    )

    # In training loop
    if rm.cuda.is_available():
        device = rm.device('cuda')
        for inputs, targets in gpu_train_loader:
            # Data is already in pinned memory, faster transfer to GPU
            inputs, targets = inputs.to(device), targets.to(device)
            # Training steps...

Building a Neural Network
~~~~~~~~~~~~~~~~~~~~~~~~~

Neural networks are models composed of multiple layers that learn patterns in data. In Riemann, we use the ``nn.Module`` class to build neural networks.

Neural Network Building Steps:

1. **Import necessary modules**: Import the ``riemann.nn`` module, which contains various network layers and activation functions
2. **Define network class**: Inherit from the ``nn.Module`` class
3. **Initialize network layers**: Define the network's layers in the ``__init__`` method
4. **Define forward propagation**: Define how data flows through the network in the ``forward`` method
5. **Create network instance**: Instantiate the defined network class

Basic Network Building Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import riemann.nn as nn

    class SimpleNet(nn.Module):
        def __init__(self):
            """
            Initialize a simple fully connected neural network
            
            Network structure:
            - Input layer: 10 features
            - Hidden layer 1: 50 neurons, using ReLU activation
            - Hidden layer 2: 20 neurons, using ReLU activation
            - Output layer: 2 neurons (suitable for regression tasks)
            """
            super(SimpleNet, self).__init__()
            # Define network layers
            self.fc1 = nn.Linear(10, 50)  # Input layer to first hidden layer
            self.relu = nn.ReLU()          # Activation function
            self.fc2 = nn.Linear(50, 20)   # First hidden layer to second hidden layer
            self.fc3 = nn.Linear(20, 2)    # Second hidden layer to output layer
        
        def forward(self, x):
            """
            Define the forward propagation process
            
            :param x: Input data, shape [batch_size, 10]
            :return: Output data, shape [batch_size, 2]
            """
            # Forward propagation
            x = self.fc1(x)  # Through first fully connected layer
            x = self.relu(x) # Apply ReLU activation function
            x = self.fc2(x)  # Through second fully connected layer
            x = self.relu(x) # Apply ReLU activation function
            x = self.fc3(x)  # Through output layer
            return x

    # Create network instance
    model = SimpleNet()
    print(model)  # Print network structure

Classification Network Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For classification tasks, we need to adjust the output layer and activation function:

.. code-block:: python

    class ClassificationNet(nn.Module):
        def __init__(self, num_classes=10):
            """
            Initialize a classification neural network
            
            :param num_classes: Number of classes for the classification task
            """
            super(ClassificationNet, self).__init__()
            self.fc1 = nn.Linear(10, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, num_classes)  # Output layer size equals number of classes
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            # Note: For classification tasks, we typically handle activation in the loss function
            # When using CrossEntropyLoss, no need to apply softmax here
            return x

Using Optimizers
~~~~~~~~~~~~~~~~

Optimizers are used to update network parameters based on the gradients of the loss function, allowing the model to gradually learn better representations.

Common Optimizers
^^^^^^^^^^^^^^^^^

Riemann provides multiple optimizers, each with its own characteristics and suitable scenarios:

- **SGD**: Stochastic Gradient Descent, the basic optimizer
- **Adam**: Adaptive Moment Estimation, combines momentum and adaptive learning rate
- **RMSprop**: Root Mean Square Propagation, suitable for recurrent neural networks
- **Adagrad**: Adaptive Gradient Algorithm, suitable for sparse data

Optimizer Usage Example
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from riemann.optim import SGD, Adam, RMSprop

    # Create SGD optimizer
    # lr: Learning rate, controls the step size of parameter updates
    # momentum: Momentum, accelerates the optimization process
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Or use Adam optimizer
    # betas: Coefficients for computing moving averages of gradient and its square
    # weight_decay: Weight decay, used for regularization
    # optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.0001)

    # Or use RMSprop optimizer
    # optimizer = RMSprop(model.parameters(), lr=0.001, alpha=0.99, eps=1e-08)

Learning Rate Scheduling
^^^^^^^^^^^^^^^^^^^^^^^^

Learning rate is an important hyperparameter that often needs adjustment during training:

.. code-block:: python

    # Simple learning rate scheduling example
    initial_lr = 0.01
    optimizer = SGD(model.parameters(), lr=initial_lr, momentum=0.9)

    # Adjust learning rate during training
    for epoch in range(num_epochs):
        # Halve the learning rate every 5 epochs
        if epoch % 5 == 0 and epoch > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
        
        # Training code...
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

Defining Loss Functions
~~~~~~~~~~~~~~~~~~~~~~~

Loss functions measure the difference between model predictions and ground truth, serving as the optimization target for the model.

Loss Function Selection Guide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Loss Function Selection Guide
   :widths: 20 30 50
   :header-rows: 1

   * - Task Type
     - Recommended Loss Function
     - Application Scenarios
   * - Regression
     - MSELoss
     - Predicting continuous values, such as house prices, temperature, etc.
   * - Regression
     - L1Loss
     - Regression tasks insensitive to outliers
   * - Regression
     - HuberLoss
     - Combines advantages of MSE and L1, robust to outliers
   * - Classification
     - CrossEntropyLoss
     - Multi-class classification tasks, outputs class probabilities
   * - Classification
     - BCEWithLogitsLoss
     - Binary classification tasks, outputs probability of 0 or 1

Loss Function Usage Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import riemann.nn as nn

    # For regression tasks
    # MSELoss: Mean Squared Error loss, calculates the average of squared differences
    criterion = nn.MSELoss()

    # For classification tasks
    # CrossEntropyLoss: Cross entropy loss, combines log_softmax and nll_loss
    # criterion = nn.CrossEntropyLoss()

    # For binary classification tasks
    # BCEWithLogitsLoss: Binary cross entropy loss with logits
    # criterion = nn.BCEWithLogitsLoss()

    # For regression tasks sensitive to outliers
    # HuberLoss: Huber loss, uses MSE for small errors and L1 for large errors
    # criterion = nn.HuberLoss(delta=1.0)

Training the Network
~~~~~~~~~~~~~~~~~~~~

Training a network is an iterative process consisting of four main steps: forward propagation, loss calculation, backward propagation, and parameter update.

Complete Training Loop Explained
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    num_epochs = 10  # Number of training epochs
    
    for epoch in range(num_epochs):
        # Set model to training mode
        # This enables dropout and batch normalization behaviors specific to training
        model.train()
        
        running_loss = 0.0  # Accumulated loss
        
        # Iterate through data loader
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 1. Zero gradients
            # Gradients must be zeroed before each iteration to prevent accumulation
            optimizer.zero_grad()
            
            # 2. Forward pass
            # Pass input data through the network to get predictions
            outputs = model(inputs)
            
            # 3. Calculate loss
            # Measure the difference between predictions and ground truth
            loss = criterion(outputs, targets)
            
            # 4. Backward pass
            # Calculate gradients of the loss with respect to all learnable parameters
            loss.backward()
            
            # 5. Update parameters
            # Update network parameters based on the calculated gradients
            optimizer.step()
            
            # Accumulate loss
            running_loss += loss.item()
            
            # Print batch information
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Calculate and print average loss for each epoch
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

Training Tips
^^^^^^^^^^^^^

1. **Early stopping**: Stop training when validation loss no longer improves to prevent overfitting
2. **Regularization**: Use weight decay, dropout, etc. to prevent overfitting
3. **Batch normalization**: Accelerates training and improves model stability
4. **Gradient clipping**: Prevents gradient explosion, especially in recurrent neural networks
5. **Mixed precision training**: Uses half-precision floating points to speed up training

Inference and Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~

After model training is complete, it's necessary to evaluate its performance on a test set to ensure the model can generalize to unseen data.

Model Evaluation Steps
^^^^^^^^^^^^^^^^^^^^^^

1. **Set model to evaluation mode**: Disables dropout and batch normalization training behaviors
2. **Use no_grad context**: Disables gradient calculation to save memory and computational resources
3. **Iterate through test data**: Calculate performance metrics on the test set
4. **Calculate evaluation metrics**: Choose appropriate metrics based on task type

Evaluation Example
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Set model to evaluation mode
    model.eval()
    
    # Evaluation metrics
    test_loss = 0.0
    correct = 0
    total = 0
    
    # Use no_grad context to disable gradient calculation
    with rm.no_grad():
        for inputs, targets in test_loader:
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            # For classification tasks, calculate accuracy
            # _, predicted = rm.max(outputs, dim=1)
            # total += targets.size(0)
            # correct += (predicted == targets).sum().item()
    
    # Calculate average loss
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    
    # For classification tasks, calculate accuracy
    # accuracy = 100 * correct / total
    # print(f"Test Accuracy: {accuracy:.2f}%")

Saving and Loading Models
^^^^^^^^^^^^^^^^^^^^^^^^^

Trained models can be saved to disk for later use:

.. code-block:: python

    # Save model
    rm.save(model.state_dict(), 'model.pth')
    print("Model saved successfully!")

    # Load model
    # Create model instance
    loaded_model = SimpleNet()
    # Load saved parameters
    loaded_model.load_state_dict(rm.load('model.pth'))
    # Set to evaluation mode
    loaded_model.eval()
    print("Model loaded successfully!")

    # Use loaded model for inference
    with rm.no_grad():
        # Sample input
        sample_input = rm.randn(1, 10)
        # Model prediction
        prediction = loaded_model(sample_input)
        print(f"Sample prediction: {prediction}")

Other Notes
-----------

Using CUDA
~~~~~~~~~~

If CUDA is available, you can move the model and data to GPU for execution:

.. code-block:: python

    # Check if CUDA is available
    if rm.cuda.is_available():
        device = rm.device('cuda')
        print("Using CUDA")
    else:
        device = rm.device('cpu')
        print("Using CPU")

    # Move model to device
    model.to(device)

    # In the training loop, also move data to device
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        # Training steps...

Neural Network Basic Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Input Layer**: Receives raw data
- **Hidden Layers**: Extract data features, the number of layers and neurons determines the model's expressive capacity
- **Output Layer**: Produces final prediction results
- **Activation Functions**: Introduce non-linearity, enabling networks to learn complex mappings
- **Loss Functions**: Measure the difference between predictions and ground truth
- **Optimizers**: Update network parameters based on loss gradients

Activation Function Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **ReLU**: Suitable for most scenarios, computationally efficient, mitigates gradient vanishing problem
- **LeakyReLU**: Solves the "dying neuron" problem of ReLU
- **Sigmoid**: Suitable for output layers in binary classification tasks
- **Softmax**: Suitable for output layers in multi-class classification tasks
- **Tanh**: Output range in [-1, 1], better gradient properties than Sigmoid
- **GELU**: Performs excellently in Transformer models

Loss Function Selection
~~~~~~~~~~~~~~~~~~~~~~~

- **MSELoss**: Suitable for regression tasks
- **L1Loss**: Insensitive to outliers, suitable for some regression tasks
- **CrossEntropyLoss**: Suitable for multi-class classification tasks
- **BCEWithLogitsLoss**: Suitable for binary classification tasks
- **HuberLoss**: Combines advantages of MSE and L1, robust to outliers

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
   * - ``register_parameter(name, param)``
     - Register parameter to module
     - ``self.register_parameter('weight', Parameter(rm.randn(10, 5)))``
   * - ``register_buffer(name, tensor)``
     - Register buffer to module
     - ``self.register_buffer('running_mean', rm.zeros(10))``
   * - ``add_module(name, module)``
     - Explicitly add submodule
     - ``self.add_module('linear', Linear(10, 5))``
   * - ``_get_name()``
     - Get module class name
     - ``print(model._get_name())  # Output class name``
   * - ``register_parameters_batch(**parameters)``
     - Batch register parameters
     - ``self.register_parameters_batch(weight=Parameter(rm.randn(10, 5)), bias=Parameter(rm.zeros(5)))``
   * - ``register_buffers_batch(**buffers)``
     - Batch register buffers
     - ``self.register_buffers_batch(running_mean=rm.zeros(10), running_var=rm.ones(10))``
   * - ``clear_cache()``
     - Clear attribute access cache
     - ``model.clear_cache()``
   * - ``enable_cache(enabled=True)``
     - Enable or disable attribute cache
     - ``model.enable_cache(False)``
   * - ``register_forward_pre_hook(hook)``
     - Register forward pre-hook, executed before forward method call
     - ``handle = model.register_forward_pre_hook(lambda module, input: print(f'Input: {input}'))``
   * - ``register_forward_hook(hook)``
     - Register forward hook, executed after forward method call
     - ``handle = model.register_forward_hook(lambda module, input, output: print(f'Output: {output}'))``
   * - ``register_full_backward_pre_hook(hook)``
     - Register backward pre-hook, executed before backward propagation starts
     - ``handle = model.register_full_backward_pre_hook(lambda module, grad_output: print(f'Grad output: {grad_output}'))``
   * - ``register_full_backward_hook(hook)``
     - Register backward hook, executed after backward propagation completes
     - ``handle = model.register_full_backward_hook(lambda module, grad_input, grad_output: print(f'Grad input: {grad_input}'))``

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

Parameter Class
---------------

The ``Parameter`` class is used to wrap tensors, making them learnable parameters of the module:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    class CustomLayer(nn.Module):
        def __init__(self, in_features, out_features):
            super(CustomLayer, self).__init__()
            # Create learnable parameters
            self.weight = nn.Parameter(rm.randn(out_features, in_features))
            self.bias = nn.Parameter(rm.zeros(out_features))
        
        def forward(self, x):
            return x @ self.weight.T + self.bias

    # Use custom layer
    layer = CustomLayer(10, 5)
    print(layer.weight.shape)  # (5, 10)
    print(layer.bias.shape)    # (5,)

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
    
    # Method 2: Using keyword arguments (PyTorch style)
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
    x = layers['dropout'](x)  # Use newly added module
    x = layers['linear2'](x)
    
    print(x.shape)  # [32, 5]

ParameterList
~~~~~~~~~~~~~

The ``ParameterList`` container is specifically designed for storing parameter lists, allowing access by index, suitable for scenarios requiring management of multiple parameters:

**Constructor Parameters**:

- ``parameters`` (iterable, optional): An iterable of parameters. Can be a list, tuple, generator, or any iterable object where each element must be a ``Parameter`` object. If ``None``, creates an empty list. Default: ``None``

**Main Methods**:

- ``append(parameter)``: Append parameter to the end of the list
  
  - **Parameters**:
    
    - ``parameter`` (Parameter): The parameter to add, must be of type ``Parameter``
    
  - **Description**: The parameter is automatically registered to the module with the registration name being the string representation of the current list length (e.g., ``'0'``, ``'1'``, ``'2'``, etc.)
  
  - **Exceptions**:
    
    - ``TypeError``: If ``parameter`` is not a ``Parameter`` object

- ``extend(parameters)``: Extend the parameter list
  
  - **Parameters**:
    
    - ``parameters`` (iterable): An iterable containing ``Parameter`` objects
    
  - **Description**: Adds multiple parameters to the end of the list, with each parameter calling the ``append()`` method in sequence
  
  - **Exceptions**:
    
    - ``TypeError``: If any element in ``parameters`` is not a ``Parameter`` object

- ``__getitem__(idx)``: Index access to parameters
  
  - **Parameters**:
    
    - ``idx`` (int): Parameter index, must be an integer, supports negative indexing (e.g., ``-1`` for the last parameter)
    
  - **Returns**:
    
    - ``Parameter``: The parameter at the specified index
    
  - **Exceptions**:
    
    - ``IndexError``: If the index is out of range
    - ``TypeError``: If ``idx`` is not an integer type

- ``__len__()``: Get the length of the parameter list
  
  - **Returns**:
    
    - ``int``: The number of parameters in the list

- ``__iter__()``: Iterator support
  
  - **Returns**:
    
    - ``iterator``: An iterator over the parameters, supporting ``for`` loop iteration

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create empty parameter list
    params = nn.ParameterList()
    print(len(params))  # 0
    
    # Create from list
    params = nn.ParameterList([
        nn.Parameter(rm.randn(10, 20)),
        nn.Parameter(rm.randn(20))
    ])
    print(len(params))  # 2
    
    # Create from generator
    params = nn.ParameterList(nn.Parameter(rm.randn(i, i+1)) for i in range(3))
    print(len(params))  # 3
    
    # Add more parameters
    params.append(nn.Parameter(rm.randn(20, 5)))
    params.append(nn.Parameter(rm.randn(5)))
    print(len(params))  # 5
    
    # Batch add parameters
    new_params = [nn.Parameter(rm.randn(5, 3)), nn.Parameter(rm.randn(3))]
    params.extend(new_params)
    print(len(params))  # 7
    
    # Index access (supports negative indexing)
    weight1 = params[0]      # First parameter
    bias1 = params[1]        # Second parameter
    last_param = params[-1]  # Last parameter
    
    # Iterative access
    for i, param in enumerate(params):
        print(f"Parameter {i}: {param.shape}")
    
    # Verify parameters are registered
    for name, param in params.named_parameters():
        print(f"{name}: {param.shape}")
    
    # Use in module
    class MultiLayerNetwork(nn.Module):
        def __init__(self):
            super(MultiLayerNetwork, self).__init__()
            self.params = nn.ParameterList([
                nn.Parameter(rm.randn(10, 20)),
                nn.Parameter(rm.randn(20)),
                nn.Parameter(rm.randn(20, 5)),
                nn.Parameter(rm.randn(5))
            ])
        
        def forward(self, x):
            x = x @ self.params[0] + self.params[1]
            x = x @ self.params[2] + self.params[3]
            return x
    
    model = MultiLayerNetwork()
    x = rm.randn(32, 10)
    output = model(x)
    print(output.shape)  # [32, 5]

ParameterDict
~~~~~~~~~~~~~

The ``ParameterDict`` container is specifically designed for storing parameter dictionaries, allowing access by key, suitable for scenarios requiring management of parameters by name:

**Constructor Parameters**:

- ``parameters`` (dict, optional): A dictionary of parameters. Keys must be of string type, values must be ``Parameter`` objects. If ``None``, creates an empty parameter dictionary. Default: ``None``

**Main Methods**:

- ``__setitem__(key, parameter)``: Set parameter
  
  - **Parameters**:
    
    - ``key`` (str): Parameter key, must be of string type
    - ``parameter`` (Parameter): The parameter to set, must be of type ``Parameter``
    
  - **Description**: The parameter is automatically registered to the module with the registration name being the specified key
  
  - **Exceptions**:
    
    - ``TypeError``: If ``key`` is not of string type, or ``parameter`` is not a ``Parameter`` object

- ``__getitem__(key)``: Get parameter by key
  
  - **Parameters**:
    
    - ``key`` (str): Parameter key, must be of string type
    
  - **Returns**:
    
    - ``Parameter``: The parameter for the specified key
    
  - **Exceptions**:
    
    - ``KeyError``: If the specified key does not exist in the dictionary
    - ``TypeError``: If ``key`` is not of string type

- ``update(parameters)``: Update parameter dictionary
  
  - **Parameters**:
    
    - ``parameters`` (dict): A dictionary containing ``Parameter`` objects, keys must be of string type
    
  - **Description**: For each key-value pair in the dictionary, calls the ``__setitem__`` method to add the parameter. If the key already exists, the original parameter is overwritten
  
  - **Exceptions**:
    
    - ``TypeError``: If ``parameters`` is not a dictionary type, or keys are not strings, or values are not ``Parameter`` objects

- ``keys()``: Get all parameter keys
  
  - **Returns**:
    
    - ``dict_keys``: A view of parameter keys, containing all keys of string type

- ``items()``: Get all parameter items
  
  - **Returns**:
    
    - ``dict_items``: A view of parameter items, containing ``(key, Parameter)`` tuples

- ``values()``: Get all parameter values
  
  - **Returns**:
    
    - ``dict_values``: A view of parameter values, containing all ``Parameter`` objects

- ``__iter__()``: Iterator support
  
  - **Returns**:
    
    - ``iterator``: An iterator over parameter keys (of string type)

- ``__len__()``: Get parameter dictionary length
  
  - **Returns**:
    
    - ``int``: The number of parameters in the dictionary

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create empty parameter dictionary
    params = nn.ParameterDict()
    print(len(params))  # 0
    
    # Create from dictionary
    params = nn.ParameterDict({
        'w1': nn.Parameter(rm.randn(10, 20)),
        'b1': nn.Parameter(rm.randn(20)),
        'w2': nn.Parameter(rm.randn(20, 5)),
        'b2': nn.Parameter(rm.randn(5))
    })
    print(len(params))  # 4
    
    # Dynamically add parameters
    params['scale'] = nn.Parameter(rm.randn(1))
    params['shift'] = nn.Parameter(rm.randn(1))
    print(len(params))  # 6
    
    # Batch add/update parameters
    params.update({
        'new_w': nn.Parameter(rm.randn(5, 3)),
        'new_b': nn.Parameter(rm.randn(3))
    })
    print(len(params))  # 8
    
    # Overwrite existing parameter
    params.update({'w1': nn.Parameter(rm.randn(10, 20))})
    
    # Access by key
    weight1 = params['w1']
    bias1 = params['b1']
    
    # Use variable as key
    w_key = 'encoder_weight'
    params[w_key] = nn.Parameter(rm.randn(20, 10))
    encoder_w = params[w_key]
    
    # Iterate over keys
    for name in params:
        print(f"Key: {name}")
    
    # Iterate over key-value pairs
    for name, param in params.items():
        print(f"{name}: {param.shape}")
    
    # Iterate over values
    for param in params.values():
        print(f"Shape: {param.shape}")
    
    # Membership check
    print('w1' in params.keys())  # True
    print('nonexistent' in params.keys())  # False
    
    # Verify parameters are registered
    for name, param in params.named_parameters():
        print(f"{name}: {param.shape}")
    
    # Use in module
    class NamedParameterNetwork(nn.Module):
        def __init__(self):
            super(NamedParameterNetwork, self).__init__()
            self.params = nn.ParameterDict({
                'encoder_w': nn.Parameter(rm.randn(10, 20)),
                'encoder_b': nn.Parameter(rm.randn(20)),
                'decoder_w': nn.Parameter(rm.randn(20, 5)),
                'decoder_b': nn.Parameter(rm.randn(5))
            })
        
        def forward(self, x):
            x = x @ self.params['encoder_w'] + self.params['encoder_b']
            x = x @ self.params['decoder_w'] + self.params['decoder_b']
            return x
    
    model = NamedParameterNetwork()
    x = rm.randn(32, 10)
    output = model(x)
    print(output.shape)  # [32, 5]

Container Class Selection
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Sequential**: Suitable for simple linear networks, concise code
- **ModuleList**: Suitable for scenarios requiring dynamic adjustment of module order or quantity
- **ModuleDict**: Suitable for scenarios requiring selection of different modules based on conditions
- **ParameterList**: Suitable for scenarios requiring management of multiple parameters
- **ParameterDict**: Suitable for scenarios requiring management of parameters by name

Mixing Container Classes
~~~~~~~~~~~~~~~~~~~~~~~~

You can mix different container classes based on the complexity of the network structure:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    class ComplexNetwork(nn.Module):
        def __init__(self):
            super(ComplexNetwork, self).__init__()
            
            # Use Sequential to define feature extractor
            self.feature_extractor = nn.Sequential(
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, 25),
                nn.ReLU()
            )
            
            # Use ModuleList to define multiple classification heads
            self.classifiers = nn.ModuleList([
                nn.Linear(25, 10),  # Classification task 1
                nn.Linear(25, 5),   # Classification task 2
                nn.Linear(25, 1)    # Regression task
            ])
            
            # Use ModuleDict to define different activation functions
            self.activations = nn.ModuleDict({
                'relu': nn.ReLU(),
                'sigmoid': nn.Sigmoid(),
                'softmax': nn.Softmax(dim=1)
            })
        
        def forward(self, x, task_type):
            x = self.feature_extractor(x)
            
            if task_type == 'classification1':
                x = self.classifiers[0](x)
                x = self.activations['softmax'](x)
            elif task_type == 'classification2':
                x = self.classifiers[1](x)
                x = self.activations['softmax'](x)
            elif task_type == 'regression':
                x = self.classifiers[2](x)
                x = self.activations['relu'](x)
            
            return x

    # Use mixed container network
    model = ComplexNetwork()
    x = rm.randn(32, 100)
    
    # Execute classification task 1
    output1 = model(x, 'classification1')
    print(f"Classification1 output shape: {output1.shape}")  # [32, 10]
    
    # Execute regression task
    output3 = model(x, 'regression')
    print(f"Regression output shape: {output3.shape}")      # [32, 1]

Activation Functions
====================

Activation functions are important components in neural networks, introducing non-linear characteristics that enable networks to learn complex function mappings. Riemann provides multiple activation functions suitable for different scenarios and tasks.

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
   * - ``RReLU``
     - Randomized leaky ReLU, slope random during training
     - As regularization method to prevent overfitting
     - ``lower``: Lower bound of slope, default 1/8
       ``upper``: Upper bound of slope, default 1/3
       ``training``: Whether in training mode
     - Fixed slope during testing
   * - ``PReLU``
     - Parameterized ReLU, slope learnable
     - Scenarios needing to learn negative region slope
     - ``num_parameters``: Number of learnable parameters, default 1
       ``init``: Initial slope value, default 0.25
     - May cause overfitting, use cautiously
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
   * - ``LogSoftmax``
     - Logarithm of softmax function
     - Used with NLLLoss for numerical stability
     - ``dim``: Calculation dimension, default -1
     - Outputs log probabilities
   * - ``GELU``
     - Gaussian Error Linear Unit
     - Default choice in Transformer models
     - No parameters
     - Higher computational cost
   * - ``Softplus``
     - Smooth approximation to ReLU
     - Scenarios needing smooth activation functions
     - ``beta``: Curve steepness, default 1.0
       ``threshold``: Linear approximation threshold, default 20.0
     - Higher computational cost

Activation Function Usage Examples
----------------------------------

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create various activation functions
    relu = nn.ReLU()
    leaky_relu = nn.LeakyReLU(negative_slope=0.01)
    prelu = nn.PReLU(num_parameters=1)
    sigmoid = nn.Sigmoid()
    tanh = nn.Tanh()
    softmax = nn.Softmax(dim=1)
    gelu = nn.GELU()
    log_softmax = nn.LogSoftmax(dim=1)
    softplus = nn.Softplus(beta=1.0)
    rrelu = nn.RReLU(lower=0.1, upper=0.3)
    
    # Test input
    x = rm.randn(4, 10)
    
    # Use activation functions
    output_relu = relu(x)
    output_leaky = leaky_relu(x)
    output_prelu = prelu(x)
    output_sigmoid = sigmoid(x)
    output_tanh = tanh(x)
    output_softmax = softmax(x)
    output_gelu = gelu(x)
    output_log_softmax = log_softmax(x)
    output_softplus = softplus(x)
    output_rrelu = rrelu(x)
    
    # Verify output shape
    print(f"ReLU output shape: {output_relu.shape}")  # [4, 10]
    print(f"Softmax output sum: {rm.sum(output_softmax, dim=1)}")  # Should be close to [1, 1, 1, 1]

Loss Functions
==============

Loss functions are used to measure the difference between model predictions and true target values, and are core components of model training. Riemann provides multiple loss functions suitable for different types of tasks.

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
     - ``size_average``: Deprecated
       ``reduce``: Deprecated
       ``reduction``: Aggregation method, default 'mean'
     - Sensitive to outliers
   * - ``L1Loss``
     - L1 loss (absolute error)
     - Regression tasks insensitive to outliers
     - ``size_average``: Deprecated
       ``reduce``: Deprecated
       ``reduction``: Aggregation method, default 'mean'
     - Gradient discontinuous at origin
   * - ``SmoothL1Loss``
     - Smooth L1 loss, combining advantages of MSE and L1
     - Object detection and other tasks
     - ``size_average``: Deprecated
       ``reduce``: Deprecated
       ``reduction``: Aggregation method, default 'mean'
       ``beta``: Smoothing threshold, default 1.0
     - Moderate computational cost
   * - ``CrossEntropyLoss``
     - Cross entropy loss, combining log_softmax and nll_loss
     - Multi-class classification tasks
     - ``weight``: Class weights
       ``size_average``: Deprecated
       ``ignore_index``: Ignored target value, default -100
       ``reduce``: Deprecated
       ``reduction``: Aggregation method, default 'mean'
       ``label_smoothing``: Label smoothing degree, default 0.0
     - Input is raw logits, no need for softmax
   * - ``BCEWithLogitsLoss``
     - Binary cross entropy loss with logits
     - Binary classification tasks
     - ``weight``: Sample weights
       ``size_average``: Deprecated
       ``reduce``: Deprecated
       ``reduction``: Aggregation method, default 'mean'
       ``pos_weight``: Positive class weight
     - Input is raw logits, no need for sigmoid
   * - ``HuberLoss``
     - Huber loss, robust to outliers
     - Regression tasks sensitive to outliers
     - ``delta``: Threshold, default 1.0
       ``size_average``: Deprecated
       ``reduce``: Deprecated
       ``reduction``: Aggregation method, default 'mean'
     - Moderate computational cost
   * - ``NLLLoss``
     - Negative log likelihood loss
     - Classification tasks used with LogSoftmax
     - ``weight``: Class weights
       ``size_average``: Deprecated
       ``ignore_index``: Ignored target value, default -100
       ``reduce``: Deprecated
       ``reduction``: Aggregation method, default 'mean'
     - Input is log probabilities

Loss Function Usage Examples
----------------------------

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create various loss functions
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    smooth_l1_loss = nn.SmoothL1Loss(beta=1.0)
    cross_entropy_loss = nn.CrossEntropyLoss()
    bce_with_logits_loss = nn.BCEWithLogitsLoss()
    huber_loss = nn.HuberLoss(delta=1.0)
    nll_loss = nn.NLLLoss()
    
    # Regression task test data
    reg_preds = rm.randn(4, 1)
    reg_targets = rm.randn(4, 1)
    
    # Classification task test data
    cls_preds = rm.randn(4, 10)
    cls_targets = rm.randint(0, 10, (4,))
    
    # Binary classification task test data
    binary_preds = rm.randn(4, 1)
    binary_targets = rm.randint(0, 2, (4, 1)).float()
    
    # Calculate various losses
    loss_mse = mse_loss(reg_preds, reg_targets)
    loss_l1 = l1_loss(reg_preds, reg_targets)
    loss_smooth_l1 = smooth_l1_loss(reg_preds, reg_targets)
    loss_ce = cross_entropy_loss(cls_preds, cls_targets)
    loss_bce = bce_with_logits_loss(binary_preds, binary_targets)
    loss_huber = huber_loss(reg_preds, reg_targets)
    
    # Calculate NLLLoss (need to compute log_softmax first)
    log_softmax = nn.LogSoftmax(dim=1)
    logits = log_softmax(cls_preds)
    loss_nll = nll_loss(logits, cls_targets)
    
    # Print loss values
    print(f"MSE Loss: {loss_mse.item():.4f}")
    print(f"L1 Loss: {loss_l1.item():.4f}")
    print(f"Cross Entropy Loss: {loss_ce.item():.4f}")
    print(f"BCE With Logits Loss: {loss_bce.item():.4f}")

Basic Network Layers
====================

Basic network layers are fundamental components for building neural networks, including fully connected layers, dropout layers, flatten layers, etc. These layers are widely used in various neural network architectures.

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
    x = rm.randn(32, 20)  # Batch of 32 samples
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
    output_train = dropout(x)
    
    # Forward pass (evaluation mode)
    dropout.eval()
    output_eval = dropout(x)
    
    print(output_train.shape)  # [4, 16]
    print(output_eval.shape)   # [4, 16]

Flatten Layer
-------------

Flatten layer flattens multi-dimensional tensors into 2D tensors (batch dimension remains unchanged):

**Parameters**:
- No parameters

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create flatten layer
    flatten = nn.Flatten()
    
    # Forward pass
    x = rm.randn(4, 16, 8, 8)  # 4 samples, 16 channels, 8x8 feature maps
    output = flatten(x)
    print(output.shape)  # [4, 1024] (16*8*8)

Batch Normalization Layer (BatchNorm1d)
----------------------------------------

Batch normalization layer normalizes inputs, accelerating training and improving model stability:

**Parameters**:
- ``num_features``: Number of features
- ``eps``: Numerical stability parameter, default 1e-5
- ``momentum``: Momentum parameter, default 0.1
- ``affine``: Whether to use learnable affine parameters, default True
- ``track_running_stats``: Whether to track running statistics, default True

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create batch normalization layer
    batch_norm = nn.BatchNorm1d(num_features=16)
    
    # Forward pass
    x = rm.randn(4, 16)  # 4 samples, 16 features
    output = batch_norm(x)
    print(output.shape)  # [4, 16]

Convolutional Network Modules
==============================

Convolutional networks are powerful tools for processing grid-structured data such as images. Riemann provides rich convolution and pooling layers, supporting 1D, 2D, and 3D data.

Convolutional Layers
--------------------

Convolutional layers extract local features through sliding windows and are core components of convolutional neural networks.

Conv1d
~~~~~~

1D convolutional layer, suitable for sequence data such as audio, text, etc.:

**Parameters**:

- ``in_channels``: Input channels
- ``out_channels``: Output channels
- ``kernel_size``: Convolution kernel size
- ``stride``: Stride, default 1
- ``padding``: Padding, default 0
- ``dilation``: Dilation rate, default 1
- ``groups``: Group convolution count, default 1
- ``bias``: Whether to use bias, default True

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create 1D convolution layer
    conv1d = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
    
    # Forward pass
    x = rm.randn(10, 16, 50)  # [batch_size, channels, length]
    output = conv1d(x)
    print(output.shape)  # [10, 32, 50] (with padding)

Conv2d
~~~~~~

2D convolutional layer, suitable for image data:

**Parameters**:

- ``in_channels``: Input channels
- ``out_channels``: Output channels
- ``kernel_size``: Convolution kernel size
- ``stride``: Stride, default 1
- ``padding``: Padding, default 0
- ``dilation``: Dilation rate, default 1
- ``groups``: Group convolution count, default 1
- ``bias``: Whether to use bias, default True

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create 2D convolution layer
    conv2d = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    
    # Forward pass
    x = rm.randn(4, 3, 32, 32)  # [batch_size, channels, height, width]
    output = conv2d(x)
    print(output.shape)  # [4, 16, 32, 32] (with padding)

Conv3d
~~~~~~

3D convolutional layer, suitable for 3D data such as video, medical imaging, etc.:

**Parameters**:

- ``in_channels``: Input channels
- ``out_channels``: Output channels
- ``kernel_size``: Convolution kernel size
- ``stride``: Stride, default 1
- ``padding``: Padding, default 0
- ``dilation``: Dilation rate, default 1
- ``groups``: Group convolution count, default 1
- ``bias``: Whether to use bias, default True

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create 3D convolution layer
    conv3d = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    
    # Forward pass
    x = rm.randn(2, 3, 16, 16, 16)  # [batch_size, channels, depth, height, width]
    output = conv3d(x)
    print(output.shape)  # [2, 16, 16, 16, 16] (with padding)

Pooling Layers
--------------

Pooling layers reduce the spatial dimensions of feature maps while preserving important information.

MaxPool1d
~~~~~~~~~

1D max pooling layer:

**Parameters**:

- ``kernel_size``: Pooling kernel size
- ``stride``: Stride, default kernel_size
- ``padding``: Padding, default 0
- ``dilation``: Dilation rate, default 1
- ``ceil_mode``: Whether to use ceiling mode, default False
- ``return_indices``: Whether to return max indices, default False

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create 1D max pooling layer
    max_pool1d = nn.MaxPool1d(kernel_size=2, stride=2)
    
    # Forward pass
    x = rm.randn(4, 16, 50)  # [batch_size, channels, length]
    output = max_pool1d(x)
    print(output.shape)  # [4, 16, 25]

MaxPool2d
~~~~~~~~~

2D max pooling layer:

**Parameters**:

- ``kernel_size``: Pooling kernel size
- ``stride``: Stride, default kernel_size
- ``padding``: Padding, default 0
- ``dilation``: Dilation rate, default 1
- ``ceil_mode``: Whether to use ceiling mode, default False
- ``return_indices``: Whether to return max indices, default False

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create 2D max pooling layer
    max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
    
    # Forward pass
    x = rm.randn(4, 16, 32, 32)  # [batch_size, channels, height, width]
    output = max_pool2d(x)
    print(output.shape)  # [4, 16, 16, 16]

MaxPool3d
~~~~~~~~~

3D max pooling layer:

**Parameters**:

- ``kernel_size``: Pooling kernel size
- ``stride``: Stride, default kernel_size
- ``padding``: Padding, default 0
- ``dilation``: Dilation rate, default 1
- ``ceil_mode``: Whether to use ceiling mode, default False
- ``return_indices``: Whether to return max indices, default False

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create 3D max pooling layer
    max_pool3d = nn.MaxPool3d(kernel_size=2, stride=2)
    
    # Forward pass
    x = rm.randn(2, 16, 16, 16, 16)  # [batch_size, channels, depth, height, width]
    output = max_pool3d(x)
    print(output.shape)  # [2, 16, 8, 8, 8]

AvgPool1d
~~~~~~~~~

1D average pooling layer:

**Parameters**:

- ``kernel_size``: Pooling kernel size
- ``stride``: Stride, default kernel_size
- ``padding``: Padding, default 0
- ``ceil_mode``: Whether to use ceiling mode, default False
- ``count_include_pad``: Whether to include padding values, default True
- ``divisor_override``: Custom divisor, default None

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create 1D average pooling layer
    avg_pool1d = nn.AvgPool1d(kernel_size=2, stride=2)
    
    # Forward pass
    x = rm.randn(4, 16, 50)  # [batch_size, channels, length]
    output = avg_pool1d(x)
    print(output.shape)  # [4, 16, 25]

AvgPool2d
~~~~~~~~~

2D average pooling layer:

**Parameters**:

- ``kernel_size``: Pooling kernel size
- ``stride``: Stride, default kernel_size
- ``padding``: Padding, default 0
- ``ceil_mode``: Whether to use ceiling mode, default False
- ``count_include_pad``: Whether to include padding values, default True
- ``divisor_override``: Custom divisor, default None

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create 2D average pooling layer
    avg_pool2d = nn.AvgPool2d(kernel_size=2, stride=2)
    
    # Forward pass
    x = rm.randn(4, 16, 32, 32)  # [batch_size, channels, height, width]
    output = avg_pool2d(x)
    print(output.shape)  # [4, 16, 16, 16]

AvgPool3d
~~~~~~~~~

3D average pooling layer:

**Parameters**:

- ``kernel_size``: Pooling kernel size
- ``stride``: Stride, default kernel_size
- ``padding``: Padding, default 0
- ``ceil_mode``: Whether to use ceiling mode, default False
- ``count_include_pad``: Whether to include padding values, default True
- ``divisor_override``: Custom divisor, default None

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create 3D average pooling layer
    avg_pool3d = nn.AvgPool3d(kernel_size=2, stride=2)
    
    # Forward pass
    x = rm.randn(2, 16, 16, 16, 16)  # [batch_size, channels, depth, height, width]
    output = avg_pool3d(x)
    print(output.shape)  # [2, 16, 8, 8, 8]

Transformer
===========

Transformer is a deep learning architecture based on self-attention mechanisms, originally designed for natural language processing tasks and now the mainstream method for sequence modeling. Riemann provides complete Transformer components compatible with PyTorch's interface.

MultiheadAttention
------------------

Multi-head attention mechanism that allows the model to simultaneously focus on information from different representation subspaces.

**Description**:
Implements multi-head attention mechanism by computing multiple sets of attention weights in parallel to capture different aspects of the input sequence.

**Parameters**:

- ``embed_dim`` (int): Dimension of input and output vectors, must be divisible by ``num_heads``
- ``num_heads`` (int): Number of attention heads
- ``dropout`` (float, optional): Dropout probability applied to attention weights during training, default is 0.0
- ``bias`` (bool, optional): Whether to add bias terms in projection layers, default is True
- ``add_bias_kv`` (bool, optional): Whether to add learnable bias to key and value sequences, default is False
- ``add_zero_attn`` (bool, optional): Whether to add a column of zeros to attention weights, default is False
- ``kdim`` (int, optional): Dimension of key vectors, default is None (uses embed_dim)
- ``vdim`` (int, optional): Dimension of value vectors, default is None (uses embed_dim)
- ``batch_first`` (bool, optional): Input/output format, default is False (seq_len, batch_size, embed_dim)

**Notes**:

- ``embed_dim`` must be divisible by ``num_heads``
- When ``batch_first=True``, input shape is (batch_size, seq_len, embed_dim)
- Supports both self-attention and cross-attention modes

**Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create multi-head attention layer
    mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.1)
    
    # Self-attention mode
    query = rm.randn(10, 32, 512)  # [seq_len, batch_size, embed_dim]
    key = query
    value = query
    output, attn_weights = mha(query, key, value)
    print(output.shape)  # [10, 32, 512]
    
    # Cross-attention mode
    query = rm.randn(10, 32, 512)  # Target sequence
    key = rm.randn(20, 32, 512)    # Source sequence
    value = rm.randn(20, 32, 512)  # Source sequence
    output, attn_weights = mha(query, key, value)
    print(output.shape)  # [10, 32, 512]
    
    # Using batch_first=True
    mha_bf = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
    query = rm.randn(32, 10, 512)  # [batch_size, seq_len, embed_dim]
    output, _ = mha_bf(query, query, query)
    print(output.shape)  # [32, 10, 512]

TransformerEncoderLayer
-----------------------

A single layer of Transformer encoder, consisting of self-attention mechanism and feed-forward network.

**Description**:
Implements a single layer of Transformer encoder, containing multi-head self-attention sublayer and feed-forward neural network sublayer, each followed by residual connection and layer normalization.

**Parameters**:

- ``d_model`` (int): Dimension of input and output features
- ``nhead`` (int): Number of attention heads
- ``dim_feedforward`` (int, optional): Dimension of hidden layer in feed-forward network, default is 2048
- ``dropout`` (float, optional): Dropout probability applied to layer outputs during training, default is 0.1
- ``activation`` (str, optional): Activation function type in feed-forward network, 'relu' or 'gelu', default is 'relu'
- ``layer_norm_eps`` (float, optional): Epsilon value for layer normalization, default is 1e-05
- ``batch_first`` (bool, optional): Input/output format, default is False
- ``norm_first`` (bool, optional): Whether to use Pre-LN mode, default is False (Post-LN mode)
- ``bias`` (bool, optional): Whether to add bias in all linear layers, default is True

**Notes**:

- ``norm_first=False`` uses Post-LN mode (original Transformer paper): attention/feed-forward first, then residual connection, then layer normalization
- ``norm_first=True`` uses Pre-LN mode: layer normalization first, then attention/feed-forward, then residual connection
- Pre-LN mode usually provides more stable training

**Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create Transformer encoder layer (Post-LN mode)
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1
    )
    
    # Forward pass
    src = rm.randn(10, 32, 512)  # [seq_len, batch_size, d_model]
    output = encoder_layer(src)
    print(output.shape)  # [10, 32, 512]
    
    # Using Pre-LN mode
    encoder_layer_prenorm = nn.TransformerEncoderLayer(
        d_model=512, nhead=8, norm_first=True
    )
    output = encoder_layer_prenorm(src)
    print(output.shape)  # [10, 32, 512]

TransformerDecoderLayer
-----------------------

A single layer of Transformer decoder, consisting of self-attention, cross-attention mechanisms and feed-forward network.

**Description**:
Implements a single layer of Transformer decoder, containing three sublayers: masked multi-head self-attention, multi-head cross-attention, and feed-forward neural network, each followed by residual connection and layer normalization.

**Parameters**:

- ``d_model`` (int): Dimension of input and output features
- ``nhead`` (int): Number of attention heads
- ``dim_feedforward`` (int, optional): Dimension of hidden layer in feed-forward network, default is 2048
- ``dropout`` (float, optional): Dropout probability applied to layer outputs during training, default is 0.1
- ``activation`` (str, optional): Activation function type in feed-forward network, 'relu' or 'gelu', default is 'relu'
- ``layer_norm_eps`` (float, optional): Epsilon value for layer normalization, default is 1e-05
- ``batch_first`` (bool, optional): Input/output format, default is False
- ``norm_first`` (bool, optional): Whether to use Pre-LN mode, default is False
- ``bias`` (bool, optional): Whether to add bias in all linear layers, default is True

**Notes**:

- Decoder layer needs to receive both target sequence (tgt) and encoder output (memory)
- Self-attention uses masking to prevent attending to future positions (for autoregressive generation)
- Cross-attention is used to attend to encoder output information

**Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create Transformer decoder layer
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1
    )
    
    # Forward pass
    tgt = rm.randn(20, 32, 512)     # [tgt_len, batch_size, d_model] Target sequence
    memory = rm.randn(10, 32, 512)  # [src_len, batch_size, d_model] Encoder output
    output = decoder_layer(tgt, memory)
    print(output.shape)  # [20, 32, 512]

TransformerEncoder
------------------

Transformer encoder consisting of N stacked TransformerEncoderLayer layers.

**Description**:
Passes input sequence through multiple encoder layers for feature extraction, where each encoder layer contains self-attention mechanism and feed-forward network.

**Parameters**:

- ``encoder_layer`` (TransformerEncoderLayer): Single encoder layer instance, will be cloned num_layers times
- ``num_layers`` (int): Number of encoder layers
- ``norm`` (Module, optional): Final layer normalization layer, default is None

**Notes**:

- Encoder layers are deep-copied, so the passed encoder_layer will not be modified
- Final layer normalization can be added to stabilize output

**Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create encoder layer
    encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    
    # Create encoder (6 layers)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    
    # Forward pass
    src = rm.randn(10, 32, 512)  # [seq_len, batch_size, d_model]
    output = transformer_encoder(src)
    print(output.shape)  # [10, 32, 512]
    
    # Encoder with final layer normalization
    encoder_norm = nn.LayerNorm(512)
    transformer_encoder_norm = nn.TransformerEncoder(
        encoder_layer, num_layers=6, norm=encoder_norm
    )
    output = transformer_encoder_norm(src)
    print(output.shape)  # [10, 32, 512]

TransformerDecoder
------------------

Transformer decoder consisting of N stacked TransformerDecoderLayer layers.

**Description**:
Passes target sequence through multiple decoder layers for feature extraction, where each decoder layer contains self-attention, cross-attention, and feed-forward network.

**Parameters**:

- ``decoder_layer`` (TransformerDecoderLayer): Single decoder layer instance, will be cloned num_layers times
- ``num_layers`` (int): Number of decoder layers
- ``norm`` (Module, optional): Final layer normalization layer, default is None

**Notes**:

- Decoder needs encoder output (memory) as input for cross-attention
- Suitable for sequence-to-sequence generation tasks

**Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create decoder layer
    decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
    
    # Create decoder (6 layers)
    transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    
    # Forward pass
    tgt = rm.randn(20, 32, 512)     # [tgt_len, batch_size, d_model] Target sequence
    memory = rm.randn(10, 32, 512)  # [src_len, batch_size, d_model] Encoder output
    output = transformer_decoder(tgt, memory)
    print(output.shape)  # [20, 32, 512]

Transformer
-----------

Complete Transformer model containing encoder and decoder.

**Description**:
Implements the complete Transformer architecture, an end-to-end implementation of encoder-decoder structure, suitable for sequence-to-sequence tasks such as machine translation, text summarization, etc.

**Parameters**:

- ``d_model`` (int, optional): Feature dimension of encoder/decoder input, default is 512
- ``nhead`` (int, optional): Number of attention heads, default is 8
- ``num_encoder_layers`` (int, optional): Number of encoder layers, default is 6
- ``num_decoder_layers`` (int, optional): Number of decoder layers, default is 6
- ``dim_feedforward`` (int, optional): Dimension of feed-forward network, default is 2048
- ``dropout`` (float, optional): Dropout value, default is 0.1
- ``activation`` (str, optional): Activation function for encoder/decoder intermediate layers, 'relu' or 'gelu', default is 'relu'
- ``custom_encoder`` (Module, optional): Custom encoder, default is None
- ``custom_decoder`` (Module, optional): Custom decoder, default is None
- ``layer_norm_eps`` (float, optional): Epsilon value for layer normalization, default is 1e-05
- ``batch_first`` (bool, optional): Whether input/output tensors are (batch, seq, feature) format, default is False
- ``norm_first`` (bool, optional): Whether to execute LayerNorm before other attention and feed-forward operations, default is False
- ``bias`` (bool, optional): Whether Linear and LayerNorm layers learn additive bias, default is True

**Notes**:

- If ``custom_encoder`` or ``custom_decoder`` is provided, custom modules will be used instead of default encoder/decoder
- Complete Transformer is suitable for sequence-to-sequence tasks
- For encoder-only tasks (like BERT), can use TransformerEncoder only
- For decoder-only tasks (like GPT), can use TransformerDecoder only

**Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create complete Transformer model
    transformer_model = nn.Transformer(
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1
    )
    
    # Forward pass
    src = rm.randn(10, 32, 512)     # [src_len, batch_size, d_model] Source sequence
    tgt = rm.randn(20, 32, 512)     # [tgt_len, batch_size, d_model] Target sequence
    output = transformer_model(src, tgt)
    print(output.shape)  # [20, 32, 512]
    
    # Using batch_first=True
    transformer_model_bf = nn.Transformer(
        d_model=512, nhead=8, batch_first=True
    )
    src = rm.randn(32, 10, 512)     # [batch_size, src_len, d_model]
    tgt = rm.randn(32, 20, 512)     # [batch_size, tgt_len, d_model]
    output = transformer_model_bf(src, tgt)
    print(output.shape)  # [32, 20, 512]

Examples
========

Simple CNN for Image Classification
-----------------------------------

.. code-block:: python

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

Transformer for Time Series Prediction
---------------------------------------

.. code-block:: python

    # This example demonstrates how to use a Transformer model for time series prediction
    # Including data preparation, model construction, training and prediction

    import riemann as rm
    import riemann.nn as nn
    from riemann.optim import Adam
    from riemann.utils.data import Dataset, DataLoader
    import numpy as np

    # Generate time series data
    def generate_time_series_data(num_samples, seq_length, pred_length):
        """
        Generate time series data
        
        :param num_samples: Number of samples
        :param seq_length: Input sequence length
        :param pred_length: Prediction sequence length
        :return: Input sequences and target sequences
        """
        # Generate sine wave data as an example
        t = np.linspace(0, 100, num_samples + seq_length + pred_length)
        data = np.sin(t) + 0.1 * np.random.randn(len(t))
        
        inputs = []
        targets = []
        
        for i in range(num_samples):
            inputs.append(data[i:i+seq_length])
            targets.append(data[i+seq_length:i+seq_length+pred_length])
        
        return np.array(inputs), np.array(targets)

    # Custom time series dataset
    class TimeSeriesDataset(Dataset):
        def __init__(self, num_samples=1000, seq_length=50, pred_length=10):
            self.inputs, self.targets = generate_time_series_data(
                num_samples, seq_length, pred_length
            )
            # Convert to Riemann tensors
            self.inputs = rm.tensor(self.inputs, dtype=rm.float32)
            self.targets = rm.tensor(self.targets, dtype=rm.float32)
            
        def __len__(self):
            return len(self.inputs)
        
        def __getitem__(self, idx):
            return self.inputs[idx], self.targets[idx]

    # Simplified Transformer time series prediction model (encoder only)
    class TransformerTimeSeriesModel(nn.Module):
        def __init__(self, input_dim=1, d_model=64, nhead=4, 
                     num_layers=2, dim_feedforward=128, pred_length=10):
            """
            Transformer time series prediction model (simplified version)
            
            Uses only Transformer encoder to map sequences to predictions
            
            :param input_dim: Input feature dimension
            :param d_model: Transformer model dimension
            :param nhead: Number of multi-head attention heads
            :param num_layers: Number of encoder layers
            :param dim_feedforward: Feedforward network dimension
            :param pred_length: Prediction sequence length
            """
            super(TransformerTimeSeriesModel, self).__init__()
            
            self.d_model = d_model
            self.pred_length = pred_length
            
            # Input embedding layer: map input dimension to d_model dimension
            self.input_embedding = nn.Linear(input_dim, d_model)
            
            # Positional encoding parameter (learnable positional encoding)
            self.pos_encoding = nn.Parameter(rm.randn(100, d_model) * 0.01)
            
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward,
                dropout=0.1,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, 
                num_layers=num_layers
            )
            
            # Output layer: map d_model dimension to pred_length * input_dim
            self.output_layer = nn.Linear(d_model, pred_length * input_dim)
            
        def forward(self, src):
            """
            Forward pass
            
            :param src: Input sequence [batch_size, src_len, input_dim]
            :return: Prediction sequence [batch_size, pred_length, input_dim]
            """
            batch_size, src_len, input_dim = src.shape
            
            # Input embedding
            src = self.input_embedding(src)  # [batch_size, src_len, d_model]
            
            # Add positional encoding
            src = src + self.pos_encoding[:src_len, :].unsqueeze(0)
            
            # Encoder
            memory = self.transformer_encoder(src)  # [batch_size, src_len, d_model]
            
            # Take the last time step output
            last_output = memory[:, -1, :]  # [batch_size, d_model]
            
            # Output layer
            output = self.output_layer(last_output)  # [batch_size, pred_length * input_dim]
            
            # Reshape to [batch_size, pred_length, input_dim]
            output = output.view(batch_size, self.pred_length, input_dim)
            
            return output

    # Create dataset and data loader
    train_dataset = TimeSeriesDataset(num_samples=1000, seq_length=50, pred_length=10)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Create model
    model = TransformerTimeSeriesModel(
        input_dim=1, 
        d_model=64, 
        nhead=4, 
        num_layers=2,
        dim_feedforward=128,
        pred_length=10
    )

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Add feature dimension
            inputs = inputs.unsqueeze(-1)  # [batch_size, seq_len, 1]
            targets = targets.unsqueeze(-1)  # [batch_size, pred_len, 1]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    # Prediction example
    model.eval()
    with rm.no_grad():
        # Get a test sample
        test_input, test_target = train_dataset[0]
        test_input = test_input.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
        
        # Make prediction
        prediction = model(test_input)
        
        print(f"\nInput sequence shape: {test_input.shape}")
        print(f"Prediction sequence shape: {prediction.shape}")
        print(f"True target shape: {test_target.shape}")
        
        # Calculate prediction error
        test_target = test_target.unsqueeze(-1)
        error = rm.mean((prediction - test_target) ** 2).item()
        print(f"Prediction MSE: {error:.6f}")
        
        # Print target and prediction values
        print("\n===== Prediction Results Comparison =====")
        print(f"{'Step':<10} {'Target':<15} {'Prediction':<15} {'Error':<15}")
        print("-" * 55)
        
        pred_values = prediction.squeeze().tolist()
        target_values = test_target.squeeze().tolist()
        
        for i in range(len(target_values)):
            target_val = target_values[i]
            pred_val = pred_values[i] if isinstance(pred_values, list) else pred_values
            diff = target_val - pred_val
            print(f"{i+1:<10} {target_val:<15.6f} {pred_val:<15.6f} {diff:<15.6f}")
        
        print("-" * 55)