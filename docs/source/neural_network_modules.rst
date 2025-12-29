Neural Network Modules
=======================

Riemann provides a comprehensive set of neural network modules through the ``riemann.nn`` package. These modules are building blocks for creating and training neural networks.

Module Basics
-------------

All neural network modules in Riemann inherit from the ``nn.Module`` class. This base class provides functionality for parameter management, forward pass definition, and gradient computation.

Creating a Custom Module
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    class MyNetwork(nn.Module):
        def __init__(self):
            super(MyNetwork, self).__init__()
            self.linear1 = nn.Linear(10, 5)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(5, 1)
        
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x
    
    # Create an instance
    model = MyNetwork()
    print(model)

Module Parameters
~~~~~~~~~~~~~~~~~

Parameters are the learnable aspects of a module. They are automatically tracked for gradient computation:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create a linear layer
    linear = nn.Linear(10, 5)
    
    # Access parameters
    for name, param in linear.named_parameters():
        print(f"{name}: {param.shape}")
    
    # Check if a tensor is a parameter
    print(linear.weight.requires_grad)  # True

Linear Layers
-------------

Linear layers perform affine transformations on input data.

Fully Connected Layer
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create a linear layer
    linear = nn.Linear(in_features=20, out_features=10)
    
    # Forward pass
    x = rm.randn(32, 20)  # Batch of 32 samples
    output = linear(x)
    print(output.shape)  # [32, 10]

Convolutional Layers
--------------------

Convolutional layers are essential for processing spatial data like images.

1D Convolution
~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create a 1D convolution layer
    conv1d = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
    
    # Forward pass
    x = rm.randn(10, 16, 50)  # [batch_size, channels, length]
    output = conv1d(x)
    print(output.shape)  # [10, 32, 48] (assuming no padding)

2D Convolution
~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create a 2D convolution layer
    conv2d = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    
    # Forward pass
    x = rm.randn(4, 3, 32, 32)  # [batch_size, channels, height, width]
    output = conv2d(x)
    print(output.shape)  # [4, 16, 32, 32] (with padding)

3D Convolution
~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create a 3D convolution layer
    conv3d = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3)
    
    # Forward pass
    x = rm.randn(2, 3, 16, 16, 16)  # [batch_size, channels, depth, height, width]
    output = conv3d(x)
    print(output.shape)  # [2, 16, 14, 14, 14] (assuming no padding)

Pooling Layers
--------------

Pooling layers reduce spatial dimensions.

Max Pooling
~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create a max pooling layer
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    # Forward pass
    x = rm.randn(4, 16, 32, 32)
    output = maxpool(x)
    print(output.shape)  # [4, 16, 16, 16]

Average Pooling
~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create an average pooling layer
    avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    # Forward pass
    x = rm.randn(4, 16, 32, 32)
    output = avgpool(x)
    print(output.shape)  # [4, 16, 16, 16]

Normalization Layers
--------------------

Normalization layers help stabilize training.

Batch Normalization
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create a batch normalization layer
    batch_norm = nn.BatchNorm2d(num_features=16)
    
    # Forward pass
    x = rm.randn(4, 16, 32, 32)
    output = batch_norm(x)
    print(output.shape)  # [4, 16, 32, 32]

Activation Functions
--------------------

Riemann provides various activation functions.

ReLU and Variants
~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create activation functions
    relu = nn.ReLU()
    leaky_relu = nn.LeakyReLU(negative_slope=0.01)
    prelu = nn.PReLU(num_parameters=1)  # Learnable parameter
    elu = nn.ELU(alpha=1.0)
    
    # Forward pass
    x = rm.randn(4, 16)
    output_relu = relu(x)
    output_leaky = leaky_relu(x)
    output_prelu = prelu(x)
    output_elu = elu(x)
    
    print(output_relu.shape)  # [4, 16]

Sigmoid and Tanh
~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create activation functions
    sigmoid = nn.Sigmoid()
    tanh = nn.Tanh()
    
    # Forward pass
    x = rm.randn(4, 16)
    output_sigmoid = sigmoid(x)
    output_tanh = tanh(x)
    
    print(output_sigmoid.shape)  # [4, 16]

Softmax
~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create a softmax layer
    softmax = nn.Softmax(dim=1)
    
    # Forward pass
    x = rm.randn(4, 10)
    output = softmax(x)
    
    # Verify that probabilities sum to 1
    print(rm.sum(output, dim=1))  # tensor([1., 1., 1., 1.])

Dropout Layers
--------------

Dropout layers help prevent overfitting.

Dropout
~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create a dropout layer
    dropout = nn.Dropout(p=0.5)
    
    # Forward pass (training mode)
    x = rm.randn(4, 16)
    output_train = dropout(x)
    
    # Forward pass (evaluation mode)
    dropout.eval()
    output_eval = dropout(x)
    
    print(output_train.shape)  # [4, 16]
    print(output_eval.shape)   # [4, 16]

Container Modules
-----------------

Container modules help organize other modules.

Sequential
~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create a sequential model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # Forward pass
    x = rm.randn(32, 10)
    output = model(x)
    
    print(output.shape)  # [32, 5]

ModuleList
~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create a module list
    layers = nn.ModuleList([
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    ])
    
    # Forward pass
    x = rm.randn(32, 10)
    for layer in layers:
        x = layer(x)
    
    print(x.shape)  # [32, 5]

ModuleDict
~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create a module dictionary
    layers = nn.ModuleDict({
        'linear1': nn.Linear(10, 20),
        'relu': nn.ReLU(),
        'linear2': nn.Linear(20, 5)
    })
    
    # Forward pass
    x = rm.randn(32, 10)
    x = layers['linear1'](x)
    x = layers['relu'](x)
    x = layers['linear2'](x)
    
    print(x.shape)  # [32, 5]

Loss Functions
--------------

Loss functions measure the difference between predictions and targets.

MSE Loss
~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create MSE loss
    mse_loss = nn.MSELoss()
    
    # Forward pass
    predictions = rm.randn(32, 10)
    targets = rm.randn(32, 10)
    loss = mse_loss(predictions, targets)
    
    print(loss.item())  # Scalar value

Cross-Entropy Loss
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create cross-entropy loss
    ce_loss = nn.CrossEntropyLoss()
    
    # Forward pass
    predictions = rm.randn(32, 10)  # Raw scores (logits)
    targets = rm.randint(0, 10, (32,))  # Class indices
    loss = ce_loss(predictions, targets)
    
    print(loss.item())  # Scalar value

Binary Cross-Entropy Loss
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Create binary cross-entropy loss
    bce_loss = nn.BCELoss()
    
    # Forward pass
    predictions = rm.sigmoid(rm.randn(32, 1))  # Probabilities
    targets = rm.randint(0, 2, (32, 1)).float()  # Binary targets
    loss = bce_loss(predictions, targets)
    
    print(loss.item())  # Scalar value

Examples
--------

Simple CNN for Image Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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