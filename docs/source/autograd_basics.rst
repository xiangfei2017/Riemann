Automatic Differentiation Basics
================================

Riemann's automatic differentiation engine, similar to PyTorch's autograd, allows for automatic computation of gradients for tensor operations. This is essential for training neural networks and other optimization tasks.

Gradient Tracking
-----------------

By default, tensors don't track their operations. To enable gradient tracking, set ``requires_grad=True`` when creating a tensor:

.. code-block:: python

    import riemann as rm
    
    # Tensor without gradient tracking
    x = rm.tensor([1., 2., 3.])
    print(x.requires_grad)  # False
    
    # Tensor with gradient tracking
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    print(x.requires_grad)  # True

You can also enable or disable gradient tracking on existing tensors:

.. code-block:: python

    x = rm.tensor([1., 2., 3.])
    print(x.requires_grad)  # False
    
    # Enable gradient tracking
    x.requires_grad_(True)
    print(x.requires_grad)  # True

Computational Graph
-------------------

When you perform operations on tensors with ``requires_grad=True``, Riemann builds a computational graph that tracks how each tensor was computed. This graph is used to compute gradients during backpropagation.

.. code-block:: python

    import riemann as rm
    
    # Create tensors with gradient tracking
    x = rm.tensor(2.0, requires_grad=True)
    y = rm.tensor(3.0, requires_grad=True)
    
    # Build computational graph
    z = x * y + x ** 2
    
    # Print computational graph information
    print(z.requires_grad)  # True

Computing Gradients
-------------------

To compute gradients, call the ``backward()`` method on the output tensor:

.. code-block:: python

    import riemann as rm
    
    # Create tensors with gradient tracking
    x = rm.tensor(2.0, requires_grad=True)
    y = rm.tensor(3.0, requires_grad=True)
    
    # Define computation
    z = x * y + x ** 2
    
    # Compute gradients
    z.backward()
    
    # Access gradients
    print(x.grad)  # dz/dx = y + 2*x = 3 + 4 = 7
    print(y.grad)  # dz/dy = x = 2

For scalar outputs, you can call ``backward()`` directly. For non-scalar outputs, you need to provide a gradient argument:

.. code-block:: python

    import riemann as rm
    
    # Create tensors with gradient tracking
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    
    # Define computation that produces a non-scalar output
    y = x * 2
    
    # Compute gradients with respect to a vector
    gradient = rm.tensor([1., 1., 1.])  # Gradient of the sum
    y.backward(gradient)
    
    # Access gradients
    print(x.grad)  # [2., 2., 2.]

Gradient Accumulation
---------------------

Gradients are accumulated by default. This means that if you call ``backward()`` multiple times, the gradients will add up:

.. code-block:: python

    import riemann as rm
    
    # Create tensor with gradient tracking
    x = rm.tensor(1.0, requires_grad=True)
    
    # First computation
    y = x * 2
    y.backward()
    print(x.grad)  # 2
    
    # Second computation
    y = x * 3
    y.backward()
    print(x.grad)  # 2 + 3 = 5 (gradients accumulate)
    
    # Clear gradients
    if x.grad is not None:
        x.grad.zero_()
    print(x.grad)  # 0

Disabling Gradient Tracking
---------------------------

Sometimes you need to perform operations without tracking gradients, for example during evaluation. You can use several methods:

Using ``torch.no_grad()`` context manager:

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    
    with rm.no_grad():
        y = x * 2
        print(y.requires_grad)  # False

Using ``detach()`` to create a new tensor without gradient tracking:

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    y = x.detach()
    print(y.requires_grad)  # False

In-place Operations and Gradients
---------------------------------

In-place operations can affect gradient computation. Here are some important considerations:

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    
    # This will raise an error because in-place operations on leaf variables are not allowed
    try:
        x.add_(1)  # This will raise an error
    except RuntimeError as e:
        print(f"Error: {e}")
    
    # Instead, create a new tensor
    y = x + 1
    print(y.requires_grad)  # True

Higher-Order Gradients
----------------------

Riemann supports computing higher-order derivatives by setting ``create_graph=True``:

.. code-block:: python

    import riemann as rm
    
    # Create tensor with gradient tracking
    x = rm.tensor(2.0, requires_grad=True)
    
    # First-order computation
    y = x ** 3
    
    # Compute first-order gradients with graph creation
    dy_dx = rm.grad(y, x, create_graph=True)[0]
    print(dy_dx)  # 12
    
    # Compute second-order gradients
    d2y_dx2 = rm.grad(dy_dx, x)[0]
    print(d2y_dx2)  # 12

Gradient Computation Tips
-------------------------

1. **Memory Management**: Gradient computation uses memory to store the computational graph. Use ``no_grad()`` or ``detach()`` when you don't need gradients to save memory.

Common Pitfalls
---------------

1. **In-place Operations**: Avoid in-place operations on tensors that require gradients.

2. **Detaching Tensors**: Once detached, tensors lose their gradient history.

3. **Non-scalar Outputs**: Remember to provide gradient arguments when calling ``backward()`` on non-scalar outputs.

4. **Memory Leaks**: Long-running computations with gradient tracking can consume significant memory.

Examples
--------

Simple Neural Network Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    
    # Create simple dataset
    X = rm.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    y = rm.tensor([[0.0], [1.0], [0.0]], requires_grad=True)
    
    # Initialize weights and bias
    w = rm.randn(2, 1, requires_grad=True)
    b = rm.randn(1, requires_grad=True)
    
    # Forward pass
    predictions = rm.matmul(X, w) + b
    loss = rm.mean((predictions - y) ** 2)
    
    # Backward pass
    loss.backward()
    
    # Update weights (simple gradient descent)
    learning_rate = 0.01
    with rm.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    
    print(f"Loss: {loss.item():.4f}")
