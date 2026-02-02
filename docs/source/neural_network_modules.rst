Neural Network Modules
=======================

Riemann provides a comprehensive set of neural network modules through the ``riemann.nn`` package. These modules are building blocks for creating and training neural networks.

Quick Start
-----------

This section provides a step-by-step guide on how to build, train, and evaluate a complete neural network using Riemann, including dataset preparation, network construction, training process, and inference evaluation.

Dataset Preparation
~~~~~~~~~~~~~~~~~~~

Before building a neural network with Riemann, you first need to prepare your dataset. Riemann supports data processing similar to PyTorch:

.. code-block:: python

    import riemann as rm
    import numpy as np
    from riemann.utils.data import Dataset, DataLoader

    # Custom dataset class
    class SimpleDataset(Dataset):
        def __init__(self, num_samples=1000):
            # Generate random input data
            self.inputs = rm.randn(num_samples, 10)
            # Generate corresponding target values (simple linear mapping)
            weights = rm.randn(10, 2)
            biases = rm.randn(2)
            self.targets = self.inputs @ weights + biases
            
        def __len__(self):
            return len(self.inputs)
        
        def __getitem__(self, idx):
            return self.inputs[idx], self.targets[idx]

    # Create dataset instances
    train_dataset = SimpleDataset(1000)
    test_dataset = SimpleDataset(200)

Using DataLoader
~~~~~~~~~~~~~~~~

DataLoader is used for batch loading of data, supporting multi-threaded data loading and data shuffling:

.. code-block:: python

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False
    )

    # Iterate through DataLoader
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Input shape {inputs.shape}, Target shape {targets.shape}")
        break

Building a Neural Network
~~~~~~~~~~~~~~~~~~~~~~~~~

Use Riemann's ``nn.Module`` class and various network layers to build a neural network:

.. code-block:: python

    import riemann.nn as nn

    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            # Define network layers
            self.fc1 = nn.Linear(10, 50)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(50, 20)
            self.fc3 = nn.Linear(20, 2)
        
        def forward(self, x):
            # Define forward propagation
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            return x

    # Create network instance
    model = SimpleNet()
    print(model)

Using Optimizers
~~~~~~~~~~~~~~~~

Riemann provides various optimizers to update network parameters:

.. code-block:: python

    from riemann.optim import SGD

    # Create optimizer
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Or use Adam optimizer
    # from riemann.optim import Adam
    # optimizer = Adam(model.parameters(), lr=0.001)

Defining Loss Functions
~~~~~~~~~~~~~~~~~~~~~~~

Choose the appropriate loss function based on the task type:

.. code-block:: python

    # For regression tasks
    criterion = nn.MSELoss()

    # For classification tasks
    # criterion = nn.CrossEntropyLoss()

Training the Network
---------------------

Complete training loop:

.. code-block:: python

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()  # Set to training mode
        running_loss = 0.0
        
        for inputs, targets in train_loader:
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
            
            running_loss += loss.item()
        
        # Calculate average loss
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

Inference and Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate model performance on the test set:

.. code-block:: python

    model.eval()  # Set to evaluation mode
    test_loss = 0.0

    with rm.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")

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
----------

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
----------

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
----------

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

Container Class Selection
-------------------------

- **Sequential**: Suitable for simple linear networks, concise code
- **ModuleList**: Suitable for scenarios requiring dynamic adjustment of module order or quantity
- **ModuleDict**: Suitable for scenarios requiring selection of different modules based on conditions

Mixing Container Classes
------------------------

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
------

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
------

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
------

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
---------

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
---------

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
---------

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
---------

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
---------

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
---------

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

Examples
========

Simple CNN for Image Classification
-----------------------------------

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super(SimpleCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.classifier = nn.Sequential(
                nn.Linear(32 * 8 * 8, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)  # Flatten
            x = self.classifier(x)
            return x
    
    # Create model
    model = SimpleCNN(num_classes=10)
    
    # Forward pass
    x = rm.randn(4, 3, 32, 32)  # Batch of 4 RGB images
    output = model(x)
    print(output.shape)  # [4, 10]

Simple RNN for Sequence Data
----------------------------

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    class SimpleRNN(nn.Module):
        def __init__(self, input_size=10, hidden_size=50, num_classes=2):
            super(SimpleRNN, self).__init__()
            self.hidden_size = hidden_size
            # Note: Riemann has not yet implemented RNN layers, this is just an example structure
            # When using, you need to use existing layer combinations or wait for official implementation
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            # Simple feed-forward network simulation
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    # Create model
    model = SimpleRNN()
    
    # Forward pass
    x = rm.randn(32, 10)  # 32 samples, each with 10 features
    output = model(x)
    print(output.shape)  # [32, 2]