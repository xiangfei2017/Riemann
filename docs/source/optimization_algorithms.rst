Optimizers
==========

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

SGD is a variant of gradient descent that updates parameters using a subset of data at each iteration. It is one of the most fundamental optimization algorithms in machine learning.

**Algorithm Explanation:**

1. Compute gradients on a mini-batch: `∇θL(θ)`
2. Apply weight decay: `θ = θ * (1 - η * weight_decay)`
3. Update parameters: `θ = θ - η * ∇θL(θ)`
4. Optional: Apply momentum and/or Nesterov momentum

**Advantages:**

- Simple and computationally efficient
- Straightforward implementation
- No hyperparameters to tune except learning rate
- Effective for large datasets when used with momentum

**Suitable Scenarios:**

- Training large models with large datasets
- When computational resources are limited
- As a baseline for comparing other optimization algorithms
- When fine-tuning models with careful learning rate scheduling

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

Nesterov momentum is a variant of momentum that can provide better convergence by looking ahead in the direction of the momentum vector.

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

SGD with Closure Parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~

The SGD optimizer supports a ``closure`` parameter, which is a function that reevaluates the model and returns the loss. This is useful for algorithms that need to compute the loss multiple times per optimization step, or for custom optimization logic.

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # Create SGD optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Define closure function
    def closure():
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        return loss
    
    # Training loop using closure
    for epoch in range(num_epochs):
        loss = optimizer.step(closure)
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

When to Use Closure with SGD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The closure parameter is particularly useful in scenarios such as:

- Implementing custom optimization logic that requires multiple forward/backward passes
- Using SGD with algorithms that need to compute loss gradients multiple times
- Creating consistent interfaces across different optimizers (e.g., LBFGS always requires a closure)
- Debugging optimization issues by inspecting loss values at different stages

Adam (Adaptive Moment Estimation)
---------------------------------

Adam combines the best properties of AdaGrad and RMSProp algorithms to provide an optimization algorithm that can handle sparse gradients on noisy problems. It maintains both a moving average of the gradients (momentum) and a moving average of the squared gradients (adaptive learning rate).

**Algorithm Explanation:**

1. Compute first moment estimate: `m = β1*m + (1-β1)*∇θL(θ)`
2. Compute second moment estimate: `v = β2*v + (1-β2)*(∇θL(θ))²`
3. Apply bias correction: `m̂ = m/(1-β1^t)`, `v̂ = v/(1-β2^t)`
4. Apply weight decay: `θ = θ * (1 - η * weight_decay)`
5. Update parameters: `θ = θ - η*m̂/(√v̂ + ε)`

**Advantages:**

- Adaptive learning rate per parameter
- Combines benefits of momentum and adaptive learning rates
- Well-suited for sparse gradients and noisy problems
- Generally works well without extensive hyperparameter tuning

**Suitable Scenarios:**

- Training deep neural networks with many parameters
- Problems with sparse gradients (e.g., NLP tasks)
- Large-scale machine learning problems
- When quick convergence is desired
- As a default optimizer for most deep learning tasks

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

AdaGrad adapts the learning rate to the parameters, performing smaller updates for parameters associated with frequently occurring features. It maintains a per-parameter learning rate that is scaled based on the historical sum of squared gradients.

**Algorithm Explanation:**

1. Compute gradients: `∇θL(θ)`
2. Accumulate squared gradients: `G = G + (∇θL(θ))²`
3. Apply weight decay: `θ = θ * (1 - η * weight_decay)`
4. Update parameters: `θ = θ - η*∇θL(θ)/(√G + ε)`
5. Optional: Apply learning rate decay

**Advantages:**

- Adaptive learning rate per parameter
- Automatically scales learning rates for different features
- Well-suited for sparse data where some features are infrequent
- No need to manually tune learning rate for different parameters

**Suitable Scenarios:**

- Training on sparse datasets (e.g., text classification)
- Problems with features of varying frequencies
- When you want learning rates to automatically adapt to feature importance
- As a baseline for adaptive learning rate methods

Basic Adagrad
~~~~~~~~~~~~~

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

LBFGS is a quasi-Newton method that approximates the BFGS algorithm using a limited amount of memory. It maintains a history of recent gradient evaluations to approximate the Hessian matrix, allowing for more efficient optimization than full Newton methods.

**Algorithm Explanation:**

1. Compute gradients using the closure function
2. Maintain a history of recent (position, gradient) pairs
3. Approximate the inverse Hessian matrix using this history
4. Compute search direction based on the approximate inverse Hessian
5. Perform line search to find optimal step size
6. Update parameters and repeat

**Advantages:**

- Memory efficient compared to full BFGS
- Faster convergence than first-order methods for smooth objectives
- Uses second-order information without explicitly computing the Hessian
- Well-suited for small to medium-sized problems

**Suitable Scenarios:**

- Training small to medium-sized models
- Problems with smooth objective functions
- When you need faster convergence than first-order methods
- For batch optimization where you can compute the full gradient
- When memory is limited but you still want second-order optimization benefits

**LBFGS Parameters Detailed Explanation:**

- **lr**: Learning rate (default: 1.0). Unlike other optimizers, LBFGS uses this as a step size multiplier during line search.
- **max_iter**: Maximum number of iterations per optimization step (default: 20). Controls how many times the algorithm will attempt to find an optimal step.
- **max_eval**: Maximum number of function evaluations per optimization step (default: None). If None, defaults to max_iter * 1.25.
- **tolerance_grad**: Gradient tolerance for convergence (default: 1e-05). The optimization stops when the gradient norm falls below this threshold.
- **tolerance_change**: Parameter change tolerance for convergence (default: 1e-09). The optimization stops when parameter updates fall below this threshold.
- **history_size**: Update history size (default: 100). Controls how much memory is used to approximate the Hessian matrix.
- **line_search_fn**: Line search function (default: None). Custom line search function to use instead of the default implementation.

Basic LBFGS
~~~~~~~~~~~

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

LBFGS with Custom Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can customize LBFGS parameters to better suit your specific optimization problem:

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # Create LBFGS optimizer with custom parameters
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=1.0,               # Learning rate (step size multiplier)
        max_iter=50,          # Maximum iterations per step
        max_eval=None,        # Maximum function evaluations (None = max_iter * 1.25)
        tolerance_grad=1e-06,  # Gradient convergence threshold
        tolerance_change=1e-09, # Parameter change convergence threshold
        history_size=100,      # Hessian approximation memory
        line_search_fn=None    # Use default line search
    )
    
    # Define closure function
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

Parameter Tuning Tips for LBFGS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **lr**: Start with 1.0 and adjust based on convergence. For ill-conditioned problems, smaller values (0.1-0.5) may be more stable.
- **max_iter**: Increase for more complex optimization problems, but be aware of increased computation time.
- **history_size**: Larger values may improve convergence but increase memory usage. For small models, 100-200 is typically sufficient.
- **tolerance_grad** and **tolerance_change**: Adjust based on desired precision. Smaller values lead to more precise optimization but may require more iterations.
- **line_search_fn**: Only customize if you have specific line search requirements. The default implementation works well for most problems.

AdamW (Adam with Weight Decay)
-------------------------------

AdamW is an improved version of Adam that treats weight decay as a separate regularization term instead of modifying the gradients. This allows weight decay to more effectively act as L2 regularization, avoiding the weight decay side effects present in Adam.

**Algorithm Explanation:**

1. Compute first moment estimate: `m = β1*m + (1-β1)*∇θL(θ)`
2. Compute second moment estimate: `v = β2*v + (1-β2)*(∇θL(θ))²`
3. Apply bias correction: `m̂ = m/(1-β1^t)`, `v̂ = v/(1-β2^t)`
4. Apply weight decay: `θ = θ * (1 - η * weight_decay)`
5. Update parameters: `θ = θ - η*m̂/(√v̂ + ε)`

**Advantages:**

- More effective weight decay as a regularizer
- Better generalization performance
- Avoids Adam's weight decay issues
- Well-suited for modern deep learning architectures

**Suitable Scenarios:**

- Training deep neural networks
- Transfer learning and fine-tuning
- Models with many parameters prone to overfitting
- Any scenario where Adam is used, especially with weight decay

Basic AdamW
~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # Create AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

AdamW with Custom Betas
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # Create AdamW optimizer with custom betas
    optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

RMSprop (Root Mean Square Propagation)
---------------------------------------

RMSprop is an adaptive learning rate optimization algorithm that maintains a moving average of the squared gradients for each parameter. It is particularly well-suited for recurrent neural networks (RNNs) and other sequence models.

**Algorithm Explanation:**

1. Compute moving average of squared gradients: `v = α*v + (1-α)*(∇θL(θ))²`
2. Update parameters: `θ = θ - η*∇θL(θ)/(√v + ε)`
3. Optional: Apply momentum and/or use centered version

**Advantages:**

- Adaptive learning rate per parameter
- Effective for non-stationary objectives
- Well-suited for RNNs and sequence models
- Helps with the vanishing/exploding gradient problem

**Suitable Scenarios:**

- Training recurrent neural networks (RNNs)
- Training long short-term memory (LSTM) networks
- Training gated recurrent units (GRUs)
- Any model with sequence data
- Models where gradients can vary significantly over time

Basic RMSprop
~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # Create RMSprop optimizer
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99)
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

RMSprop with Momentum
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # Create RMSprop optimizer with momentum
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, momentum=0.9)
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

RMSprop with Centered Gradients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # Create RMSprop optimizer with centered gradients
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, centered=True)
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

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

Parameter groups are a powerful feature of Riemann optimizers that allow you to configure different hyperparameters for different sets of parameters in your model. This is particularly useful in scenarios such as:

- Setting different learning rates for different layers of your model
- Applying different hyperparameters to weight and bias parameters
- Using different learning rates when fine-tuning pre-trained models

Basic Structure of Parameter Groups
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameter groups are defined through a list of dictionaries, where each dictionary contains:

- `params`：The set of parameters to optimize
- Other key-value pairs：Hyperparameters specific to this parameter group (e.g., `lr`, `weight_decay`)

Basic Usage Example
~~~~~~~~~~~~~~~~~~~

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

In the example above:
- The first parameter group contains all parameters from the first layer (`model[0]`) with a learning rate of 0.01
- The second parameter group contains all parameters from the third layer (`model[2]`) with a learning rate of 0.001
- Both parameter groups share the `momentum=0.9` hyperparameter

Setting Different Hyperparameters for Weights and Biases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # Create model
    model = nn.Linear(10, 1)
    
    # Separate weight and bias parameters
    weight_params = [p for name, p in model.named_parameters() if 'weight' in name]
    bias_params = [p for name, p in model.named_parameters() if 'bias' in name]
    
    # Create parameter groups
    optimizer = optim.SGD([
        {'params': weight_params, 'lr': 0.01, 'weight_decay': 1e-4},
        {'params': bias_params, 'lr': 0.02, 'weight_decay': 0}
    ])

In this example:
- Weight parameters use a smaller learning rate (0.01) and weight decay (1e-4)
- Bias parameters use a larger learning rate (0.02) and no weight decay

Using Parameter Groups in Pre-trained Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.optim as optim
    
    # Create a pre-trained model (simplified example)
    class PretrainedModel(nn.Module):
        def __init__(self):
            super(PretrainedModel, self).__init__()
            # Assume this part is a pre-trained feature extractor
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            # Assume this part is a newly added classifier
            self.classifier = nn.Linear(64, 10)
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    model = PretrainedModel()
    
    # Create parameter groups
    optimizer = optim.SGD([
        {'params': model.features.parameters(), 'lr': 0.001},  # Lower LR for pre-trained features
        {'params': model.classifier.parameters(), 'lr': 0.01}  # Higher LR for new classifier
    ], momentum=0.9)

In this example:
- The pre-trained feature extractor uses a smaller learning rate (0.001) to avoid disrupting learned features
- The newly added classifier uses a larger learning rate (0.01) to speed up its convergence

How Parameter Groups Work
~~~~~~~~~~~~~~~~~~~~~~~~~

When you create an optimizer with parameter groups:

1. The optimizer maintains separate states for each parameter group
2. During each `step()` call, the optimizer updates parameters according to the hyperparameters of their respective groups
3. If a hyperparameter is not specified in a parameter group, the optimizer uses the default value provided in the constructor

Best Practices
~~~~~~~~~~~~~~

1. **Clear Naming**：Use `named_parameters()` to create parameter groups based on parameter names for better code readability
2. **Logical Grouping**：Group parameters based on their nature and importance, such as:
   - Parameters from different layers
   - Weight and bias parameters
   - Pre-trained and newly added parameters
3. **Learning Rate Scheduling**：Parameter group learning rates can be used with learning rate schedulers, which will adjust them based on their initial learning rates
4. **Hyperparameter Search**：Parameter groups enable more flexible hyperparameter searches by allowing different configurations for different parts of your model

Gradient Clipping
-----------------

Gradient clipping is a technique to prevent exploding gradients in deep networks by limiting the size of gradients to ensure training stability. During the training of deep neural networks, gradients can become extremely large, leading to excessive parameter updates and divergence of the training process.

Benefits of Gradient Clipping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Prevents Exploding Gradients**: Limits the maximum gradient value, avoiding excessive parameter updates
- **Prevents Exploding Gradients**: Limits the maximum gradient value, avoiding excessive parameter updates
- **Improves Training Stability**: Makes the training process more stable and reduces training fluctuations
- **Speeds Up Convergence**: Can help the model converge faster in some cases
- **Allows Larger Learning Rates**: By limiting gradients, larger initial learning rates can be used

Clip by Norm
~~~~~~~~~~~~

Clipping by norm works by calculating the L2 norm of the gradients and limiting it to a maximum norm. This method preserves the direction of the gradients while adjusting their magnitude.

**Parameter Description**:
- `parameters`: The parameter collection whose gradients to clip
- `max_norm`: The maximum norm of the gradients
- `norm_type`: The type of norm to use, default is 2 (L2 norm)
- `error_if_nonfinite`: Whether to throw an error if gradients contain non-finite values (like NaN or inf), default is False

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Clip gradients by norm
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

**Use Cases**:
- Suitable for most deep neural network training
- Especially useful when using RNNs or LSTMs
- When you observe NaN or inf in training loss

**Practical Application Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    import riemann.optim as optim

    # Create model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )

    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            
            # Clip gradients before optimizer step
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

Clip by Value
~~~~~~~~~~~~~

Clipping by value works by limiting each element of the gradients to a specified range. This method directly truncates extreme values in the gradients.

**Parameter Description**:
- `parameters`: The parameter collection whose gradients to clip
- `clip_value`: The maximum absolute value of gradients
- `error_if_nonfinite`: Whether to throw an error if gradients contain non-finite values (like NaN or inf), default is False

**Usage Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    
    # Clip gradients by value
    nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

**Use Cases**:
- When you want to directly control the maximum absolute value of gradients
- When there are extreme outliers in the gradients
- For specific network architectures like discriminators in GANs

**Practical Application Example**:

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    import riemann.optim as optim

    # Create model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            
            # Clip gradients before optimizer step
            nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
            
            optimizer.step()

Best Practices for Gradient Clipping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Choose the Right Clipping Method**:
   - Clip by norm (``clip_grad_norm_``): Suitable for most cases, especially RNNs
   - Clip by value (``clip_grad_value_``): Suitable for cases with extreme gradient values

2. **Set Reasonable Clipping Thresholds**:
   - Clip by norm: max_norm is typically set between 0.5 and 5.0
   - Clip by value: clip_value is typically set between 0.1 and 1.0

3. **Clipping Timing**:
   - Must be executed after `loss.backward()` and before `optimizer.step()`
   - Should be applied to every batch

4. **Combining with Other Techniques**:
   - Use with learning rate schedulers
   - Use with parameter groups to apply different clipping strategies to different layers

5. **Monitoring Effectiveness**:
   - Observe if training loss becomes more stable
   - Check if exploding gradients still occur
   - Adjust clipping thresholds for optimal results

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

Learning Rate Schedulers
========================

Learning rate schedulers are used to dynamically adjust the learning rate during training, which is crucial for model convergence and performance optimization. Riemann provides several learning rate schedulers, each with its specific adjustment strategy.

Types of Learning Rate Schedulers
---------------------------------

StepLR
~~~~~~

**Function**: Adjusts the learning rate by a fixed step size and decay factor.

**Parameters**:

- `optimizer`: The optimizer whose learning rate is to be adjusted
- `step_size`: The step size for learning rate decay
- `gamma`: The learning rate decay factor, default is 0.1
- `last_epoch`: The index of the last epoch, default is -1

**Use Case**: Suitable for scenarios where learning rate needs to be reduced at fixed intervals.

MultiStepLR
~~~~~~~~~~~

**Function**: Adjusts the learning rate at specified milestones.

**Parameters**:

- `optimizer`: The optimizer whose learning rate is to be adjusted
- `milestones`: A list of milestones for learning rate decay
- `gamma`: The learning rate decay factor, default is 0.1
- `last_epoch`: The index of the last epoch, default is -1

**Use Case**: Suitable for scenarios where learning rate needs to be reduced at specific epochs.

ExponentialLR
~~~~~~~~~~~~~

**Function**: Adjusts the learning rate with exponential decay.

**Parameters**:

- `optimizer`: The optimizer whose learning rate is to be adjusted
- `gamma`: The learning rate decay factor
- `last_epoch`: The index of the last epoch, default is -1

**Use Case**: Suitable for scenarios where a continuous smooth decay of learning rate is needed.

CosineAnnealingLR
~~~~~~~~~~~~~~~~~

**Function**: Adjusts the learning rate according to the shape of a cosine function.

**Parameters**:

- `optimizer`: The optimizer whose learning rate is to be adjusted
- `T_max`: The period of cosine annealing
- `eta_min`: The minimum learning rate, default is 0
- `last_epoch`: The index of the last epoch, default is -1

**Use Case**: Suitable for scenarios where learning rate needs to first decrease and then increase, helping to escape local optima.

ReduceLROnPlateau
~~~~~~~~~~~~~~~~~

**Function**: Adjusts the learning rate when a metric stops improving.

**Parameters**:

- `optimizer`: The optimizer whose learning rate is to be adjusted
- `mode`: Mode, 'min' or 'max', default is 'min'
- `factor`: The learning rate decay factor, default is 0.1
- `patience`: Number of epochs with no improvement after which learning rate will be reduced, default is 10
- `threshold`: Threshold for measuring new best, default is 1e-4
- `threshold_mode`: Threshold mode, 'rel' or 'abs', default is 'rel'
- `cooldown`: Number of epochs to wait before resuming normal operation after lr has been reduced, default is 0
- `min_lr`: Minimum learning rate, default is 0
- `eps`: Minimum change in learning rate to qualify as an improvement, default is 1e-8

**Use Case**: Suitable for scenarios where learning rate needs to be dynamically adjusted based on validation metrics.

Using Learning Rate Schedulers
------------------------------

The basic usage flow of learning rate schedulers is as follows:

1. Create an optimizer
2. Create a learning rate scheduler, passing in the optimizer and relevant parameters
3. In the training loop, first call the optimizer's `step()` method to update parameters
4. Then call the scheduler's `step()` method to update the learning rate

Scheduler and Optimizer Interaction
-----------------------------------

- **Order**: It is recommended to call `optimizer.step()` first, then `scheduler.step()`
- **Parameter Groups**: The scheduler will adjust each parameter group based on its initial learning rate
- **State Saving**: The scheduler's state can be saved and loaded using `state_dict()` and `load_state_dict()` methods
- **Special Case**: The `ReduceLROnPlateau` scheduler requires a validation metric to be passed in the `step()` method

Complete Example Code
---------------------

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    import riemann.optim as optim
    from riemann.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau

    # Create model
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )

    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # Create learning rate scheduler (choose one)
    # 1. StepLR example
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # 2. MultiStepLR example
    # scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    
    # 3. ExponentialLR example
    # scheduler = ExponentialLR(optimizer, gamma=0.99)
    
    # 4. CosineAnnealingLR example
    # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0.001)
    
    # 5. ReduceLROnPlateau example (needs validation loss in step())
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    # Define loss function
    loss_fn = nn.MSELoss()

    # Generate example data
    inputs = rm.randn(100, 10)
    targets = rm.randn(100, 1)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        scheduler.step()  # For ReduceLROnPlateau, use scheduler.step(loss.item())
        
        # Print information
        if (epoch + 1) % 10 == 0:
            current_lr = scheduler.get_lr()[0] if hasattr(scheduler, 'get_lr') else optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, LR: {current_lr:.6f}')

Best Practices for Learning Rate Schedulers
-------------------------------------------

1. **Choose the Right Scheduler**: Select the appropriate learning rate scheduling strategy based on task characteristics
2. **Set a Reasonable Initial Learning Rate**: The initial learning rate should not be too large or too small
3. **Combine with Validation Set**: When using `ReduceLROnPlateau`, it should be based on validation metrics rather than training metrics
4. **Learning Rate Lower Bound**: Set a reasonable `min_lr` for `ReduceLROnPlateau` to prevent training stagnation due to extremely small learning rates
5. **Warm-up Phase**: For large models, consider using a smaller learning rate for warm-up in the early training stages
6. **Parameter Group Compatibility**: When used with parameter groups, ensure that each parameter group has a reasonable initial learning rate
