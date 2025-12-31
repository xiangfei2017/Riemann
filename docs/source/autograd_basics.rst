Automatic Differentiation Basics
================================

Riemann's automatic differentiation engine, allows for automatic computation of gradients for tensor operations. This is essential for training neural networks and other optimization tasks.

Gradient Tracking
-----------------

By default, tensors don't track their gradients. To enable gradient tracking, set ``requires_grad=True`` when creating a tensor:

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
    z = x * y + x ** 2.
    
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
    z = x * y + x ** 2.
    
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
    y = x * 2.
    
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
    y = x * 2.
    y.backward()
    print(x.grad)  # 2
    
    # Second computation
    y = x * 3.
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
        y = x * 2.
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
        x.add_(1.)  # This will raise an error
    except RuntimeError as e:
        print(f"Error: {e}")
    
    # Instead, create a new tensor
    y = x + 1.
    print(y.requires_grad)  # True

Higher-Order Gradients
----------------------

Riemann supports computing higher-order derivatives by setting ``create_graph=True``:

.. code-block:: python

    import riemann as rm
    
    # Create tensor with gradient tracking
    x = rm.tensor(2.0, requires_grad=True)
    
    # First-order computation
    y = x ** 3.
    
    # Compute first-order gradients with graph creation
    dy_dx = rm.autograd.grad(y, x, create_graph=True)[0]
    print(dy_dx)  # 12
    
    # Compute second-order gradients
    d2y_dx2 = rm.autograd.grad(dy_dx, x)[0]
    print(d2y_dx2)  # 12

Additionally, Riemann provides two convenient tools for higher-order derivative computation: the ``d()`` method and ``higher_order_grad()`` function.

``d()`` Method
~~~~~~~~~~~~~~

The ``d()`` method of tensor objects is used to compute mixed partial derivatives of the current scalar tensor with respect to multiple scalar tensors. It allows for easy computation of multi-order mixed derivatives.

.. code-block:: python

    import riemann as rm
    
    # Create tensors with gradient tracking
    x = rm.tensor(2.0, requires_grad=True)
    y = rm.tensor(3.0, requires_grad=True)
    
    # Define function f = x^3 * y^2
    f = x ** 3 * y ** 2
    
    # Compute mixed partial derivative d²f/dxdy
    d2f_dxdy = f.d(x, y)
    print(d2f_dxdy)  # 72.0
    
    # Compute third-order mixed partial derivative d³f/dx²dy
    d3f_dx2dy = f.d(x, x, y)
    print(d3f_dx2dy)  # 72.0

``higher_order_grad()`` Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``higher_order_grad()`` function is used to compute n-th order derivatives of scalar tensor outputs with respect to input tensors. It provides a convenient way to directly compute derivatives of a specified order.

.. code-block:: python

    import riemann as rm
    
    # Create tensor with gradient tracking
    x = rm.tensor(2.0, requires_grad=True)
    
    # Define function y = x^3
    y = x ** 3
    
    # Compute second-order derivative
    d2y_dx2 = rm.autograd.higher_order_grad(y, x, 2)[0]
    print(d2y_dx2)  # 12.0
    
    # Compute third-order derivative
    d3y_dx3 = rm.autograd.higher_order_grad(y, x, 3)[0]
    print(d3y_dx3)  # 6.0
    
    # Multiple inputs case
    x1 = rm.tensor(1.0, requires_grad=True)
    x2 = rm.tensor(2.0, requires_grad=True)
    z = x1 ** 2 + x2 ** 3
    grads = rm.autograd.higher_order_grad(z, [x1, x2], 2)
    print(grads)  # (2.0, 12.0)

Functional API
--------------

Riemann also provides a set of functional API functions in ``riemann.autograd.functional`` module for computing more advanced derivative structures, such as Jacobian matrices, Hessian matrices, Jacobian-vector products, etc.

``jacobian()`` Function
~~~~~~~~~~~~~~~~~~~~~~~~

The ``jacobian()`` function computes the Jacobian matrix of a function from input to output, showing all first-order partial derivatives of the function output with respect to the input.

.. code-block:: python

    import riemann as rm
    
    # Define function f = x^2
    def f(x):
        return x ** 2
    
    # Create input tensor
    x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    # Compute Jacobian matrix
    jac = rm.autograd.functional.jacobian(f, x)
    print(jac)
    print(jac.shape)  # (3, 3)  # For vector input, shape is (n_inputs, n_outputs)

``hessian()`` Function
~~~~~~~~~~~~~~~~~~~~~~

The ``hessian()`` function computes the Hessian matrix of a scalar-valued function, showing all second-order partial derivatives of the function with respect to its inputs.

.. code-block:: python

    import riemann as rm
    
    # Define function f = x^3
    def f(x):
        return x ** 3
    
    # Create input tensor
    x = rm.tensor(2.0, requires_grad=True)
    
    # Compute Hessian matrix
    hess = rm.autograd.functional.hessian(f, x)
    print(hess)
    print(hess.shape)  # (1, 1)  # For scalar input, shape is (input_size, input_size)

``derivative()`` Function
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``derivative()`` function computes a derivative function for the given function. It creates a new function that, when called, returns the derivative of the original function at the specified inputs.

.. code-block:: python

    import riemann as rm
    
    # Define function f = x^2
    def f(x):
        return x ** 2.
    
    # Create derivative function
    df = rm.autograd.functional.derivative(f)
    
    # Test the derivative function
    x = rm.tensor(2.0, requires_grad=True)
    print(df(x))  # Should return tensor(4.0)
    
    # Multi-input example
    def g(x, y):
        return x * y + x ** 2.
    
    dg = rm.autograd.functional.derivative(g)
    x = rm.tensor(2.0, requires_grad=True)
    y = rm.tensor(3.0, requires_grad=True)
    print(dg(x, y))

``jvp()`` (Jacobian-Vector Product) Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``jvp()`` function computes the product of a Jacobian matrix with a given vector.

.. code-block:: python

    import riemann as rm
    
    # Define function f = x^2
    def f(x):
        return x ** 2
    
    # Create input tensor
    x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    # Define v vector
    v = rm.tensor([1.0, 1.0, 1.0])
    
    # Compute jvp
    f_x, jvp_val = rm.autograd.functional.jvp(f, x, v)
    print(jvp_val)

``vjp()`` (Vector-Jacobian Product) Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``vjp()`` function computes the product of a given vector with a Jacobian matrix.

.. code-block:: python

    import riemann as rm
    
    # Define function f = x^2
    def f(x):
        return x ** 2
    
    # Create input tensor
    x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    # Define v vector
    v = rm.tensor([1.0, 1.0, 1.0])
    
    # Compute vjp
    f_x, vjp_val = rm.autograd.functional.vjp(f, x, v)
    print(vjp_val)

``hvp()`` (Hessian-Vector Product) and ``vhp()`` Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``hvp()`` and ``vhp()`` functions compute Hessian-Vector Product and Vector-Hessian Product respectively. Since the Hessian matrix is symmetric, ``hvp()`` and ``vhp()`` are effectively the same.

.. code-block:: python

    import riemann as rm
    
    # Define scalar-valued function
    def f(x):
        return (x ** 3).sum()
    
    # Create input tensor
    x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    # Define v vector
    v = rm.tensor([1.0, 1.0, 1.0])
    
    # Compute hvp
    f_x, hvp_val = rm.autograd.functional.hvp(f, x, v)
    print(hvp_val)

    # vhp computes the same result as hvp
    f_x, vhp_val = rm.autograd.functional.vhp(f, x, v)
    print(vhp_val)

Custom Gradient Functions
-------------------------

Riemann provides three ways to implement custom functions with gradient tracking support:

1. **Using Riemann Tensor Functions (Automatic Gradients)**
   If you implement your custom function using existing Riemann tensor functions, you get gradient tracking automatically without writing any gradient code:
   
   .. code-block:: python

       import riemann as rm
       
       def my_custom_function(x):
           """A custom function that automatically gets gradient support"""
           return rm.exp(rm.sin(x)) + x**2.
       
       # Test automatic gradient tracking
       x = rm.tensor(1.0, requires_grad=True)
       y = my_custom_function(x)
       y.backward()
       print(f"Gradient: {x.grad}")  # Will automatically compute correct gradient

2. **Using track_grad Decorator**
   Use the ``track_grad`` decorator to wrap your function and provide explicit gradient computation:
   
   .. code-block:: python

       import riemann as rm
       import numpy as np
       
       def sigmoid_derivative(x):
           """Gradient function for sigmoid"""
           sig = 1. / (1. + np.exp(-x.data))
           return (rm.tensor(sig * (1. - sig)),)
       
       @rm.track_grad(sigmoid_derivative)
       def custom_sigmoid(x):
           """Custom sigmoid function with gradient support"""
           return rm.tensor(1. / (1. + np.exp(-x.data)))
       
       # Test custom sigmoid with gradient
       x = rm.tensor(0.0, requires_grad=True)
       y = custom_sigmoid(x)
       y.backward()
       print(f"Sigmoid output: {y}")  # Should be 0.5
       print(f"Sigmoid gradient: {x.grad}")  # Should be 0.25

3. **Using Function Class**
   For more complex cases, you can subclass ``Function`` and implement both forward and backward methods:
   
   .. code-block:: python

       import riemann as rm
       import numpy as np
       
       class CustomSigmoid(rm.autograd.Function):
           @staticmethod
           def forward(ctx, x):
               """Forward computation for sigmoid"""
               sig = 1. / (1. + np.exp(-x.data))
               ctx.save_for_backward(rm.tensor(sig))
               return rm.tensor(sig)
           
           @staticmethod
           def backward(ctx, grad_output):
               """Backward computation for sigmoid"""
               sig, = ctx.saved_tensors
               return grad_output * sig * (1. - sig)
       
       # Test CustomSigmoid
       x = rm.tensor(0.0, requires_grad=True)
       y = CustomSigmoid.apply(x)
       y.backward()
       print(f"Sigmoid output: {y}")  # Should be 0.5
       print(f"Sigmoid gradient: {x.grad}")  # Should be 0.25

Gradient Checking
-----------------

Use the ``gradcheck`` function to verify your custom gradient functions are correct:

.. code-block:: python

    import riemann as rm
    
    # Define a test function for gradcheck
    def test_function(x):
        return CustomSigmoid.apply(x)
    
    # Perform gradient check
    x = rm.tensor(0.0, requires_grad=True)
    check_passed = rm.gradcheck(test_function, (x,))
    print(f"Gradient check passed: {check_passed}")

Gradcheck verifies that your analytical gradient computation matches the numerical gradient computed using finite difference method.

Gradient Computation Tips
-------------------------

1. **Memory Management**: Gradient computation uses memory to store the computational graph. Use ``no_grad()`` or ``detach()`` when you don't need gradients to save memory.

Common Pitfalls
------------------

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