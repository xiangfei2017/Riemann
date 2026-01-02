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

Using ``riemann.no_grad()`` context manager:

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    
    with rm.no_grad():
        y = x * 2.
        print(y.requires_grad)  # False

Tensor Methods for Graph Detaching and Data Copying
---------------------------------------------------

Riemann provides several tensor methods for managing computation graph dependencies, and copying tensor data. Each method has distinct characteristics related to:
- Whether it creates a new tensor object or modifies in-place
- Whether it shares data with the original tensor
- Whether gradient tracking is preserved

Here are the key methods explained with individual examples:

1. **detach()**: Create a new tensor that shares data with the original but is detached from the computation graph

The detach() method returns a new tensor object that shares the same data memory as the original tensor, but is disconnected from the computation graph. This means:
- Changes to the detached tensor will modify the original tensor
- No gradients will be backpropagated through the detached tensor

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    y = x * 2.
    
    # Detach y from the computation graph
    detached_y = y.detach()
    
    print(f"detached_y: {detached_y}")
    print(f"detached_y.requires_grad: {detached_y.requires_grad}")
    print(f"Modifying detached_y will modify y: {id(detached_y.data) == id(y.data)}")

**Characteristics**: Creates new tensor object, shares memory with original, disables gradient tracking

2. **detach_()**: In-place operation that detaches the current tensor from the computation graph

The detach_() method is an in-place version of detach(). Instead of creating a new tensor, it modifies the current tensor to disconnect it from the computation graph.

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    y = x * 2.
    
    print(f"Before detach_(): y.requires_grad = {y.requires_grad}")
    y.detach_()  # In-place operation
    print(f"After detach_(): y.requires_grad = {y.requires_grad}")

**Characteristics**: Modifies tensor in-place (no new object), shares memory with original (same tensor), disables gradient tracking

3. **clone()**: Create a new tensor with copied data that maintains computation graph dependencies

The clone() method creates a completely new tensor object with its own data memory, but preserves the computation graph dependencies from the original tensor. This means operations on the cloned tensor can backpropagate gradients to the original tensor.

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    y = x * 2.
    
    cloned_y = y.clone()
    
    print(f"cloned_y: {cloned_y}")
    print(f"cloned_y.requires_grad: {cloned_y.requires_grad}")
    print(f"Modifying cloned_y won't modify y: {id(cloned_y.data) != id(y.data)}")
    
    # Demonstrate gradient can propagate through cloned tensor to original tensor
    loss = cloned_y.sum()
    loss.backward()
    print(f"x.grad after backward(): {x.grad}")  # Gradient propagates from cloned tensor to x

**Characteristics**: Creates new tensor object, copies data (no memory sharing), preserves gradient tracking

4. **copy()**: Create a new tensor with copied data that is detached from the computation graph

The copy() method creates a new tensor object with its own data memory and is completely detached from the computation graph. This is equivalent to calling clone().detach_() and is useful for creating independent tensor copies without gradient tracking.

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    y = x * 2.
    
    copied_y = y.copy()
    
    print(f"copied_y: {copied_y}")
    print(f"copied_y.requires_grad: {copied_y.requires_grad}")
    print(f"Modifying copied_y won't modify y: {id(copied_y.data) != id(y.data)}")

**Characteristics**: Creates new tensor object, copies data (no memory sharing), disables gradient tracking

5. Key Differences Between Methods

The following table summarizes the key differences between these four methods:

+----------------+----------------------+------------------------+-------------------------------+
| Method         | Creates New Object?  | Shares Memory with     | Supports Gradient Tracking?   |
|                |                      | Original Tensor?       |                               |
+================+======================+========================+===============================+
| detach()       | Yes                  | Yes                    | No                            |
+----------------+----------------------+------------------------+-------------------------------+
| detach_()      | No                   | N/A (same tensor)      | No                            |
+----------------+----------------------+------------------------+-------------------------------+
| clone()        | Yes                  | No                     | Yes                           |
+----------------+----------------------+------------------------+-------------------------------+
| copy()         | Yes                  | No                     | No                            |
+----------------+----------------------+------------------------+-------------------------------+

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    
    # Using detach() - creates new tensor, shares data, detached from graph
    y1 = x.detach()
    print(f"detach() result: y1 = {y1}, requires_grad={y1.requires_grad}")
    
    # Using detach_() - in-place operation, modifies current tensor
    x2 = rm.tensor([1., 2., 3.], requires_grad=True)
    print(f"Before detach_(): x2.requires_grad={x2.requires_grad}")
    x2.detach_()
    print(f"After detach_(): x2.requires_grad={x2.requires_grad}")
    
    # Using clone() - creates new tensor, copies data, maintains graph dependency
    y2 = x.clone()
    print(f"clone() result: y2 = {y2}, requires_grad={y2.requires_grad}")
    
    # Using copy() - creates new tensor, copies data, detached from graph
    y3 = x.copy()
    print(f"copy() result: y3 = {y3}, requires_grad={y3.requires_grad}")

Key differences between these methods:

- **Data Sharing**: detach() shares data with original, while clone() and copy() create new data copies
- **In-place Operation**: detach_() modifies the tensor in-place, others create new tensors
- **Gradient Tracking**: clone() maintains gradient tracking (if original requires it), others disable gradient tracking
- **Independent Copy**: copy() creates a completely independent new tensor object that does not share data with the original tensor nor preserves computational graph dependencies

In-place Operations and Gradients
---------------------------------

In-place operations can affect gradient computation. Here are important considerations:

1. **Leaf Variables with Gradient Tracking**: In-place operations are NOT allowed on leaf tensors that require gradient tracking, as this would destroy the computational graph necessary for backpropagation.
2. **Non-Leaf Variables with Gradient Tracking**: In-place operations are allowed on non-leaf tensors (intermediate results) that require gradient tracking.

Examples:

.. code-block:: python

    import riemann as rm
    
    # 1. Example: In-place operations on leaf tensors are NOT allowed
    x = rm.tensor([1., 2., 3.], requires_grad=True)  # Leaf tensor
    
    try:
        x.add_(1.)  # This will raise an error
    except RuntimeError as e:
        print(f"Error on leaf tensor in-place operation: {e}")
    
    # 2. Example: In-place operations on non-leaf tensors ARE allowed
    y = x * 2.  # Non-leaf tensor
    print(f"Before in-place add on non-leaf tensor: y = {y}")
    y.add_(3.)  # In-place operation on non-leaf tensor
    print(f"After in-place add on non-leaf tensor: y = {y}")
    
    # Compute gradient after in-place operation on non-leaf tensor
    z = y.sum()
    z.backward()
    print(f"Gradient of x (leaf tensor): x.grad = {x.grad}")
    
    # Clear gradients
    x.grad.zero_()
    
    # 3. Example: In-place assignment using tensor indexing on non-leaf tensors
    y = x * 2.  # Non-leaf tensor
    print(f"Before in-place indexing assignment: y = {y}")
    y[0] = 100.  # In-place indexing assignment
    print(f"After in-place indexing assignment: y = {y}")
    
    # Compute gradient after indexing assignment
    z = y.sum()
    z.backward()
    print(f"Gradient of x after indexing assignment: x.grad = {x.grad}")
    
    # Clear gradients
    x.grad.zero_()
    
    # 4. Example: Gradient tracking with in-place operations
    x = rm.tensor(2.0, requires_grad=True)  # Leaf tensor
    y = rm.tensor(3.0, requires_grad=True)  # Leaf tensor
    
    a = x * y  # Intermediate tensor
    a.mul_(2.)  # In-place multiply
    b = a + x  # Final tensor
    
    b.backward()
    
    print(f"Gradient of x (left value): x.grad = {x.grad}")
    print(f"Gradient of y (right value): y.grad = {y.grad}")

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

Gradient functions (Functional API)
-----------------------------------

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

Rosenbrock Function Optimization (Banana Function)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rosenbrock function (also known as banana function) is a classic non-convex optimization problem. The function has its minimum at (1, 1) with value 0.

Here's an example of optimizing the Rosenbrock function using Riemann's automatic differentiation and Adam optimizer:

.. code-block:: python

    import riemann as rm
    from riemann import optim

    # Define the Rosenbrock function (banana function)
    def rosenbrock_2d(x, y):
        """Rosenbrock function for 2D case"""
        return 100. * (y - x**2.)**2. + (1. - x)**2.

    # Initialize parameters with gradient tracking
    x = rm.tensor(-1.2, requires_grad=True)  # Start from point (-1.2, 1.0)
    y = rm.tensor(1.0, requires_grad=True)
    params = [x, y]

    # Setup optimizer
    optimizer = optim.Adam(params, lr=0.05)

    print("Optimizing Rosenbrock function (banana function):")
    print(f"Initial x: {x.item():.4f}, y: {y.item():.4f}")
    print(f"Initial loss: {rosenbrock_2d(x, y).item():.4f}")

    # Perform optimization
    for i in range(1000):
        loss = rosenbrock_2d(x, y)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Compute gradients automatically
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Print progress every 200 iterations
        if i % 200 == 0:
            print(f"Iteration {i}: loss = {loss.item():.8f}, x = {x.item():.8f}, y = {y.item():.8f}")

    # Print final results
    print(f"\nOptimization completed!")
    print(f"Final x: {x.item():.10f}, y: {y.item():.10f}")
    print(f"Final loss: {loss.item():.10f}")
    print(f"Theoretical minimum: x=1.0, y=1.0, loss=0.0")