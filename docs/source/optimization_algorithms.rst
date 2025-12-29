Optimization Algorithms
=======================

Riemann provides a variety of optimization algorithms through the ``riemann.optim`` package. These optimizers are used to update the parameters of neural networks during training.

Optimizer Basics
----------------

This section covers the fundamental concepts and usage patterns for optimizers in Riemann.

All optimizers in Riemann inherit from the ``optim.Optimizer`` class. To use an optimizer, you need to:

1. Create an optimizer instance with the parameters to optimize
2. Define a loss function
3. Zero the gradients
4. Compute the loss and call ``backward()``
5. Call the optimizer's ``step()`` method

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    import riemann.optim as optim
    
    # Create a simple model
    model = nn.Linear(10, 1)
    
    # Create an optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Define loss function
    loss_fn = nn.MSELoss()
    
    # Training loop
    for epoch in range(num_epochs):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()

GD (Gradient Descent)
---------------------

GD is the most basic optimization algorithm that updates parameters in the direction of the negative gradient.

Basic GD
~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # Create model
    model = nn.Linear(10, 1)
    
    # Create GD optimizer
    optimizer = optim.GD(model.parameters(), lr=0.01)
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

GD with Weight Decay
~~~~~~~~~~~~~~~~~~~~

Weight decay (L2 regularization) can be added to prevent overfitting.

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # Create GD optimizer with weight decay
    optimizer = optim.GD(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

SGD (Stochastic Gradient Descent)
----------------------------------

SGD is a variant of gradient descent that updates parameters using a subset of data at each iteration.

Basic SGD
~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # Create model
    model = nn.Linear(10, 1)
    
    # Create SGD optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

SGD with Momentum
~~~~~~~~~~~~~~~~~

Momentum helps accelerate SGD in the relevant direction and dampens oscillations.

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # Create SGD optimizer with momentum
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

SGD with Nesterov Momentum
~~~~~~~~~~~~~~~~~~~~~~~~~~

Nesterov momentum is a variant of momentum that can provide better convergence.

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # Create SGD optimizer with Nesterov momentum
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

Adam (Adaptive Moment Estimation)
---------------------------------

Adam combines the best properties of AdaGrad and RMSProp algorithms to provide an optimization algorithm that can handle sparse gradients on noisy problems.

Basic Adam
~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # Create Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

Adam with Weight Decay
~~~~~~~~~~~~~~~~~~~~~~

Weight decay (L2 regularization) can be added to prevent overfitting.

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # Create Adam optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

Adagrad
-------

AdaGrad adapts the learning rate to the parameters, performing smaller updates for parameters associated with frequently occurring features.

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # Create AdaGrad optimizer
    optimizer = optim.Adagrad(model.parameters(), lr=0.01)
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

LBFGS
-----

LBFGS is a quasi-Newton method that approximates the BFGS algorithm using a limited amount of memory.

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # Create LBFGS optimizer
    optimizer = optim.LBFGS(model.parameters(), lr=1.0)
    
    # Define closure function for LBFGS
    def closure():
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        return loss
    
    # Training loop
    for epoch in range(num_epochs):
        loss = optimizer.step(closure)
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

Optimizer Methods
-----------------

Zero Gradients
~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Zero gradients
    optimizer.zero_grad()

Step
~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Perform a single optimization step
    optimizer.step()

State Dict
~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Get optimizer state
    state_dict = optimizer.state_dict()
    
    # Load optimizer state
    optimizer.load_state_dict(state_dict)

Parameter Groups
----------------

Parameter groups allow you to use different hyperparameters for different sets of parameters.

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # Create model
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    
    # Create parameter groups
    optimizer = optim.SGD([
        {'params': model[0].parameters(), 'lr': 0.01},
        {'params': model[2].parameters(), 'lr': 0.001}
    ], momentum=0.9)
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

Gradient Clipping
-----------------

Gradient clipping prevents exploding gradients in deep networks.

Clip by Norm
~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Clip gradients by norm
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

Clip by Value
~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Clip gradients by value
    nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

Examples
--------

Training a Neural Network with Adam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    import riemann.optim as optim
    
    # Create model
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Define loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(50):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}')

Training with Different Learning Rates for Different Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    import riemann.optim as optim
    
    # Create a pre-trained model (simplified example)
    class PretrainedModel(nn.Module):
        def __init__(self):
            super(PretrainedModel, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Linear(64, 10)
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    model = PretrainedModel()
    
    # Create parameter groups with different learning rates
    optimizer = optim.SGD([
        {'params': model.features.parameters(), 'lr': 0.001},  # Lower LR for pretrained features
        {'params': model.classifier.parameters(), 'lr': 0.01}  # Higher LR for new classifier
    ], momentum=0.9)
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

Custom Optimizer
----------------

You can create custom optimizers by inheriting from ``optim.Optimizer``.

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    class CustomSGD(optim.Optimizer):
        def __init__(self, params, lr=0.01, momentum=0):
            defaults = dict(lr=lr, momentum=momentum)
            super(CustomSGD, self).__init__(params, defaults)
        
        def step(self, closure=None):
            loss = None
            if closure is not None:
                loss = closure()
            
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    d_p = p.grad.data
                    if group['momentum'] != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = rm.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(group['momentum']).add_(d_p)
                            d_p = buf
                    
                    p.data.add_(-group['lr'], d_p)
            
            return loss
    
    # Use custom optimizer
    optimizer = CustomSGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()