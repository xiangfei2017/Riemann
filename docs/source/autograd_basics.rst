Automatic Differentiation Basics
================================

Riemann's automatic differentiation engine automatically records tensor computations, building a computation graph, and efficiently computes derivatives through backpropagation. This is essential for training neural networks and other optimization tasks.

**Core Concepts**

- **Computation Graph**: A directed graph automatically constructed by Riemann in the background that records the relationships between tensor operations. Each node represents a tensor, and edges represent operations.

- **Forward Pass**: The process of executing operations starting from input tensors along the computation graph to obtain the final output.

- **Backward Propagation (Backprop)**: The process of propagating gradients backward along the computation graph starting from the output tensor to compute derivatives for each input tensor.

- **Gradient**: The partial derivative of a scalar output tensor with respect to other tensors, representing the rate of change of the output relative to the input.

- **Leaf Node Tensor**: A tensor created directly by the user (e.g., through ``rm.tensor()``) with ``requires_grad=True``. These are typically model parameters.

- **Intermediate Node Tensor**: A tensor created as a result of operations on other tensors. By default, gradients for intermediate nodes are not retained.

**Gradient Computation Methods**

Riemann provides two methods for computing gradients:

1. **backward() method**: Suitable for computing gradients of multiple tensors at once. After calling, gradients for all participating leaf node tensors are computed and stored in their respective ``grad`` attributes.

2. **grad() function**: Suitable for computing gradients of specific tensors. Allows precise control over which tensors' gradients to compute, returning a tuple of gradients without modifying the tensors' ``grad`` attributes.

Gradient Tracking Switch
------------------------

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

Riemann provides two methods for computing gradients: the ``backward()`` method and the ``grad()`` function.

Using the backward() Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``backward()`` method is suitable for computing gradients of multiple tensors at once. After calling, gradients are automatically stored in the ``grad`` attributes of participating leaf node tensors.

**Function Signature**:

.. code-block:: python

    tensor_object.backward(gradient=None, retain_graph=False, create_graph=False)

**Parameters**:

- **gradient** (optional): When the output tensor is not a scalar, a gradient tensor with the same shape as the output is required. For scalar outputs, this parameter can be omitted, defaulting to ``None`` (equivalent to passing scalar 1).
- **retain_graph** (optional): Kept only for PyTorch compatibility. Riemann backpropagation does not rely on this argument; ``backward()`` can be called repeatedly with the default value, without setting it to ``True``.
- **create_graph** (optional): Whether to record the computation graph of gradients for subsequent computation of higher-order derivatives, defaults to ``False``.

**Differences from torch.Tensor.backward**:

- ``retain_graph`` is kept only for compatibility and does not affect repeated backward passes.
- ``TN.backward`` does not accept an ``inputs`` argument; use ``riemann.autograd.backward(..., inputs=...)`` when gradients should accumulate only into specific tensors.
- The semantics of ``gradient`` match torch.

**Use Cases**:

- Training neural networks, computing gradients for all trainable parameters at once
- When multiple backward passes are needed (e.g., gradient accumulation)
- Computing higher-order derivatives

**Important Notes**:

- Only **leaf node tensors** with ``requires_grad=True`` will have their gradients computed
- **Intermediate node tensors** do not retain gradients by default; call ``retain_grad()`` if you need gradients for intermediate nodes
- Gradients accumulate in the ``grad`` attribute; manually zero gradients before multiple ``backward()`` calls

**Example 1: Gradient Computation for Scalar Output**

.. code-block:: python

    import riemann as rm
    
    # Create tensors with gradient tracking (leaf nodes)
    x = rm.tensor(2.0, requires_grad=True)
    y = rm.tensor(3.0, requires_grad=True)
    
    # Define computation (intermediate node)
    z = x * y + x ** 2.
    
    # Compute gradients
    z.backward()
    
    # Access gradients
    print(x.grad)  # dz/dx = y + 2*x = 3 + 4 = 7
    print(y.grad)  # dz/dy = x = 2

**Example 2: Gradient Computation for Non-Scalar Output**

.. code-block:: python

    import riemann as rm
    
    # Create tensors with gradient tracking
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    
    # Define computation that produces a non-scalar output
    y = x * 2.
    
    # Compute gradients with respect to a vector, gradient argument required
    gradient = rm.tensor([1., 1., 1.])  # Vector for Jacobian-vector product
    y.backward(gradient)
    
    # Access gradients
    print(x.grad)  # [2., 2., 2.]

**Example 3: Retaining Gradients for Intermediate Nodes**

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor(2.0, requires_grad=True)
    y = x * 3  # Intermediate node
    z = y ** 2  # Output
    
    # Retain gradients for intermediate node y
    y.retain_grad()
    
    z.backward()
    
    print(x.grad)  # dz/dx = 36
    print(y.grad)  # dz/dy = 12 (because retain_grad() was called)

Using the grad() Function
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``grad()`` function is suitable for computing gradients of specific tensors, allowing precise control over which tensors' gradients to compute.

**Function Signature**:

.. code-block:: python

    riemann.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=False, allow_unused=False)

**Parameters**:

- **outputs**: Output tensor(s) (scalar or tensor) or a sequence of output tensors, the starting point(s) for gradient computation
- **inputs**: Input tensor or tuple of tensors, specifying which tensors to compute gradients for
- **grad_outputs** (optional): Initial cotangents (initial/upstream gradients) for the roots. Explicit cotangents can have any shape but must match the corresponding root in shape and device; a single tensor is allowed for a single root, and multiple roots require a sequence of the same length as ``outputs`` (elements may be TN or None). When the whole argument is ``None`` or an element is ``None``, that position uses an implicit 1.0, which is allowed only if the corresponding root is a real scalar; non-scalar roots must provide an explicit cotangent.
- **retain_graph** (optional): Kept only for PyTorch compatibility and does not affect repeated calls, defaults to ``False``
- **create_graph** (optional): Whether to record the computation graph of gradients for subsequent computation of higher-order derivatives, defaults to ``False``
- **allow_unused** (optional): Whether to allow some input tensors to be unused, defaults to ``False``

**Differences from torch.autograd.grad**:

- ``retain_graph`` is kept only for compatibility; optional torch arguments such as ``materialize_grads`` and ``is_grads_batched`` are not implemented.
- The tuple return value, the fact that ``.grad`` is not written, and ``allow_unused`` semantics match torch.
- Multiple roots, duplicate roots, and roots that depend on each other are also handled the same way as in torch; when ``grad_outputs=None`` or an element of the sequence is ``None``, that position uses an implicit 1.0, so the corresponding root must be a real scalar (non-scalar roots must provide an explicit cotangent).

**Use Cases**:

- When you only need gradients for specific tensors, not all leaf nodes
- When you don't want to modify the ``grad`` attributes of tensors
- When you need more flexible control over the gradient computation process

**Important Notes**:

- Gradients are returned as a **tuple**, in the same order as the ``inputs`` parameter
- Only tensors specified in ``inputs`` will have their gradients computed
- **Does not modify** the ``grad`` attributes of input tensors
- Intermediate nodes, even with ``retain_grad()`` called, will not automatically have gradients computed in ``grad()``; they must be explicitly specified
- ``outputs`` supports multiple roots: duplicate roots sum their ``grad_outputs``; a root that is also a predecessor of another root waits until the upstream gradient arrives

**Example 1: Computing Gradients for Specific Tensors**

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor(2.0, requires_grad=True)
    y = rm.tensor(3.0, requires_grad=True)
    z = rm.tensor(4.0, requires_grad=True)
    
    # Define computation
    w = x * y + z
    
    # Only compute gradients for x and y, not z
    grads = rm.autograd.grad(w, (x, y))
    
    print(grads)  # (tensor(3.), tensor(2.))
    print(x.grad)  # None (grad() does not modify grad attributes)

**Example 2: Gradient Computation for Non-Scalar Output**

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    y = x * 2
    
    # For non-scalar outputs, grad_outputs must be provided
    grad_outputs = rm.tensor([1., 1., 1.])
    grads = rm.autograd.grad(y, x, grad_outputs=grad_outputs)
    
    print(grads)  # (tensor([2., 2., 2.]),)

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

Gradient Computation Context Control
------------------------------------

Riemann provides a flexible gradient computation context control mechanism through functions and context managers, allowing convenient enabling or disabling of gradient tracking. This is useful in model inference (where gradients should be disabled to save memory) and training (where gradients are needed) scenarios.

is_grad_enabled() Function
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``is_grad_enabled()`` function checks whether gradient computation is currently enabled.

.. code-block:: python

    import riemann as rm
    
    # Check current gradient status
    print(rm.is_grad_enabled())  # True (enabled by default)
    
    with rm.no_grad():
        print(rm.is_grad_enabled())  # False

no_grad() Context Manager/Decorator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``no_grad()`` temporarily disables gradient computation. In this context, all computations will not track gradients, which is suitable for inference phases and can significantly reduce memory usage and accelerate computation.

**As a context manager:**

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    
    with rm.no_grad():
        y = x * 2.
        print(y.requires_grad)  # False

**As a function decorator:**

.. code-block:: python

    import riemann as rm
    
    @rm.no_grad
    def inference(model, x):
        # Computations within the function will not track gradients
        return model(x)

enable_grad() Context Manager/Decorator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``enable_grad()`` temporarily enables gradient computation. Can be used to temporarily enable gradients within a ``no_grad`` context.

**As a context manager:**

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    
    with rm.no_grad():
        # Gradients are disabled here
        print(rm.is_grad_enabled())  # False
        
        with rm.enable_grad():
            # Gradients are temporarily enabled here
            y = x * 2.
            print(y.requires_grad)  # True
        
        # Back to disabled state
        print(rm.is_grad_enabled())  # False

**As a function decorator:**

.. code-block:: python

    import riemann as rm
    
    @rm.enable_grad
    def train_step(model, x, target, loss_fn):
        # Computations within the function will track gradients
        pred = model(x)
        loss = loss_fn(pred, target)
        loss.backward()
        return loss

set_grad_enabled() Context Manager/Decorator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``set_grad_enabled(mode)`` is the most flexible gradient control function, allowing explicit enabling or disabling of gradient computation.

**Parameters:**

- **mode** (bool): ``True`` to enable gradient computation, ``False`` to disable

**As a context manager:**

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    
    # Disable gradients
    with rm.set_grad_enabled(False):
        y = x * 2.
        print(y.requires_grad)  # False
    
    # Enable gradients
    with rm.set_grad_enabled(True):
        y = x * 2.
        print(y.requires_grad)  # True

**As a function decorator:**

.. code-block:: python

    import riemann as rm
    
    @rm.set_grad_enabled(False)
    def inference(model, x):
        return model(x)
    
    @rm.set_grad_enabled(True)
    def train(model, x, target, loss_fn):
        pred = model(x)
        loss = loss_fn(pred, target)
        loss.backward()
        return loss

Nested Context Managers
~~~~~~~~~~~~~~~~~~~~~~~

Gradient control context managers support nested usage, where inner contexts temporarily override outer settings:

.. code-block:: python

    import riemann as rm
    
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    
    with rm.no_grad():  # Outer: disable gradients
        y1 = x * 2.
        print(f"Outer no_grad: y1.requires_grad = {y1.requires_grad}")  # False
        
        with rm.enable_grad():  # Inner: enable gradients
            y2 = x * 3.
            print(f"Inner enable_grad: y2.requires_grad = {y2.requires_grad}")  # True
        
        # Back to outer context
        y3 = x * 4.
        print(f"Back to outer: y3.requires_grad = {y3.requires_grad}")  # False

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
    print(f"Modifying detached_y will modify y: {detached_y.array().ctypes.data == y.detach().array().ctypes.data}")

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
    print(f"Modifying cloned_y won't modify y: {cloned_y.detach().array().ctypes.data != y.detach().array().ctypes.data}")
    
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
    print(f"Modifying copied_y won't modify y: {copied_y.detach().array().ctypes.data != y.detach().array().ctypes.data}")

**Characteristics**: Creates new tensor object, copies data (no memory sharing), disables gradient tracking

5. Key Differences Between Methods

The following table summarizes the key differences between these four methods:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - 
     - 
     - Original Tensor?
     - 
   * - detach()
     - Yes
     - Yes
     - No
   * - detach_()
     - No
     - N/A (same tensor)
     - No
   * - clone()
     - Yes
     - No
     - Yes
   * - copy()
     - Yes
     - No
     - No

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
3. **Unary in-place operations**: the backward of ``neg_()`` negates the gradient flowing to the history; the backward of ``abs_()`` scales the gradient by the sign of the pre-operation element (the gradient is 0 where the element is 0).
4. **Remainder and floor division**: ``%=`` (``remainder_``) supports gradients, where the gradient w.r.t. the right operand is ``-grad * floor(x / y)``; ``//=`` matches PyTorch and raises ``RuntimeError`` in backward (floor_divide has no derivative).
5. **In-place operations under forward AD**: while ``func.jvp`` is tracing, in-place modification is also allowed on leaf tensors (including ``x[...] = y``). In-place operators do not create a new node; they are recorded as ordered steps applied to the tangent:
   ``add_/sub_/mul_/div_/pow_/remainder_/copy_/fill_/abs_/neg_`` and index assignment/compound assignment are supported;
   ``floor_divide_`` raises (no forward rule) and ``zero_`` does not participate. Steps are applied in order,
   aliased views of the same tensor are not double-counted, and the original leaf check is restored after tracing.

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

Additionally, Riemann provides two more convenient ways to compute higher-order derivatives: the ``TN.d()`` method and chained ``func.grad`` calls.

``TN.d()`` Method
~~~~~~~~~~~~~~~~~

``TN.d()`` is a **Riemann-specific** tensor method with no PyTorch equivalent;
it is equivalent to ``riemann.autograd.diff(self, ...)``. It is the most
convenient interface for computing higher-order mixed partial derivatives and
naturally chains multi-order derivatives with respect to multiple variables.

**Purpose**

- Compute arbitrary-order partial derivatives or mixed partial derivatives of the current scalar tensor with respect to specified variables
- Variables can be of **any shape** (scalar, vector, matrix), not limited to 0-d scalars
- When the same variable appears multiple times, orders are **automatically accumulated**
- Explicit order specification: ``(var, k)`` form directly requests k-th order derivative with respect to that variable

**Signature**

``current_scalar.d(*args, create_graph=False)``

Parameters:
    - ``*args``: Derivative specifications, two forms can be mixed
        - ``TN`` — first-order derivative with respect to this variable
        - ``(TN, k)`` — k-th order derivative with respect to this variable (k ≥ 1)
    - ``create_graph``: ``bool``, default ``False``. Whether to track the computation graph of the result so it can be further differentiated.

Returns:
    - ``TN`` whose shape is the flat concatenation of each variable's shape:
      ``shape = ⊕ shape(var_i) × order_i``

**Examples**

.. code-block:: python

    import riemann as rm
    import numpy as np

    # === Scalar variables (backward compatible) ===
    x = rm.tensor(2.0, requires_grad=True)
    print(x)
    y = rm.tensor(3.0, requires_grad=True)
    print(y)
    f = x ** 3 * y ** 2

    print(f.d(x))                 # first-order df/dx = 3x²y² = 108
    print(f.d(x, y))              # mixed partial d²f/dxdy = 6x²y = 72
    print(f.d((x, 2)))            # explicit second-order d²f/dx² = 6xy² = 108
    print(f.d(x, x))              # repeated = f.d((x, 2)) — 6xy² = 108
    print(f.d((x, 2), y))         # explicit mixed d³f/∂x²∂y = 12xy = 72

    # === Vector variables (core capability) ===
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    f = (x ** 3).sum()

    print(f.d(x))                 # gradient vector [3, 12, 27]
    print(f.d(x, x))              # Hessian matrix diag([6, 12, 18])
    print(f.d((x, 3)))            # third-order tensor shape=(3,3,3), diagonal [24, 48, 72]

    # === Multi-vector mixed partial ===
    x = rm.tensor([1., 2.], requires_grad=True)
    y = rm.tensor([3., 4.], requires_grad=True)
    f = (x * y).sum() ** 2

    print(f.d(x, y))              # (2, 2) cross partial matrix [[28, 12], [8, 38]]

    # === Scalar + Vector ===
    x = rm.tensor(2.0, requires_grad=True)
    y = rm.tensor([1., 2., 3.], requires_grad=True)
    f = x * (y ** 2).sum()

    print(f.d(x, y))              # d²f/∂x∂y = [2, 4, 6], shape=(3,)

**Notes**

1. ``self`` must be a **scalar tensor** (``ndim == 0``) with ``requires_grad=True``
2. All differentiable variables must have ``requires_grad=True``
3. At least one arg is required; otherwise raises ``ValueError``
4. For ``(var, k)`` form, k must be ≥ 1
5. Result shape is the concatenation of each variable's shape in args order; e.g., ``f.d(x, y)`` with x∈Rⁿ, y∈Rᵐ → shape=(n, m)
6. ``create_graph=True`` keeps the computation graph for further higher-order differentiation; memory cost grows exponentially with the order
7. When computing higher-order derivatives of **tensors** (ndim>0), the implementation performs per-element scalar grads and stacks them back into shape internally

Higher-Order Derivatives with ``func`` Module Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most functions in the ``func`` module return a Callable, making them naturally
suited for functional-style nested composition of ``grad``, ``jacrev``,
``jacfwd``, ``hessian`` and more — express arbitrary-order derivatives
elegantly. Below are the most typical scenarios with verified numerical results.

**Example 1: Vector input → Vector output — Jacobian and second-order tensor**

``func.jacrev`` / ``func.jacfwd`` directly differentiate vector-output functions,
handling vector input → vector output higher-order derivatives naturally.
Nesting ``jacrev`` inside itself yields the Jacobian-of-Jacobian — a third-order tensor.

.. code-block:: python

    import riemann as rm
    from riemann import func

    # f(x) = [x₀³, x₁³, x₂³], elementwise cubic, vector in → vector out
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    print(x)
    f = lambda xv: xv ** 3   # shape: (3,) → (3,)

    # ===== First-order: Jacobian matrix =====
    # J_{ij} = ∂f_i/∂x_j = 3x_i² · δ_{ij}  (diagonal)
    J = func.jacrev(f)(x)
    print(J)
    # J.shape = (3, 3)
    # J = diag([3·1², 3·2², 3·3²]) = diag([3, 12, 27])

    # jacfwd is equivalent (forward mode, comparable efficiency for small vectors)
    J_fwd = func.jacfwd(f)(x)
    print(J_fwd)
    # → identical to J

    # ===== Second-order: Jacobian-of-Jacobian — third-order tensor =====
    # T_{ijk} = ∂²f_i/∂x_j∂x_k
    # For f_i = x_i³:  non-zero only when i = j = k, T_{iii} = 6x_i
    T = func.jacrev(lambda xv: func.jacrev(f)(xv))(x)
    print(T)
    # T.shape = (3, 3, 3)
    # T[0,0,0] = 6·1 = 6,  T[1,1,1] = 6·2 = 12,  T[2,2,2] = 6·3 = 18
    # All other entries = 0

**Example 2: Vector input → Scalar output — Hessian**

The most common higher-order scenario in optimization: scalar-valued objective
over vector input, requiring the Hessian matrix.
Use ``func.jacrev(func.grad(f))`` to construct the Hessian in one nested call,
or ``func.hessian`` to obtain the Hessian matrix directly.

.. code-block:: python

    import riemann as rm
    from riemann import func

    # Least squares  f(x) = ||A·x - b||²,  x ∈ R²,  A ∈ R^{3×2},  b ∈ R³
    A = rm.tensor([[1., 2.], [3., 4.], [5., 6.]], requires_grad=False)
    b = rm.tensor([1., 2., 3.], requires_grad=False)
    x = rm.tensor([1., 2.], requires_grad=True)

    def f_least_squares(xv):
        residual = A @ xv - b        # shape (3,)
        return (residual ** 2).sum() # scalar output

    # ===== Hessian matrix =====
    # The Hessian of least squares is always 2·AᵀA (independent of x!)
    H = func.jacrev(func.grad(f_least_squares))(x)
    # H.shape = (2, 2)
    # H = 2·AᵀA = [[70, 88], [88, 112]]

**Example 3: Multi-input higher-order per input — ``f(x, y) = x³y⁴``**

Use ``argnums`` to specify which input position to differentiate, then nest ``func.grad``:

.. code-block:: python

    import riemann as rm
    from riemann import func

    x = rm.tensor(2.0, requires_grad=True)
    print(x)
    y = rm.tensor(3.0, requires_grad=True)
    print(y)

    f = lambda xv, yv: xv ** 3 * yv ** 4

    # 2nd order w.r.t. x:  ∂²f/∂x² = 6xy⁴ = 6·2·81 = 972
    d2x = func.grad(
        lambda xv, yv: func.grad(f, argnums=0)(xv, yv),
        argnums=0
    )(x, y)   # → 972.0
    print(d2x)

    # 2nd order w.r.t. y:  ∂²f/∂y² = 12x³y² = 12·8·9 = 864
    d2y = func.grad(
        lambda xv, yv: func.grad(f, argnums=1)(xv, yv),
        argnums=1
    )(x, y)   # → 864.0
    print(d2y)

    # 1st order w.r.t. both inputs simultaneously (argnums as tuple)
    dx, dy = func.grad(f, argnums=(0, 1))(x, y)
    print(dx, dy)
    # dx = 3x²y⁴ = 972.0,  dy = x³·4y³ = 864.0

**Example 4: ``func.grad`` + ``has_aux`` with higher-order continuation**

When ``f`` returns a ``(value, aux)`` pair, ``has_aux=True`` makes ``func.grad``
compute gradients only from ``value`` while passing ``aux`` through unchanged:

.. code-block:: python

    import riemann as rm
    from riemann import func
    from riemann.autograd import grad

    x = rm.tensor(2.0, requires_grad=True)
    print(x)

    # f(x) = log(x),  auxiliary value is 1.0
    def f_with_aux(xv):
        return rm.log(xv), rm.tensor(1.0)   # (value, aux)

    # Build a high-order grad function with has_aux
    grad_fn = func.grad(f_with_aux, has_aux=True)
    g1, aux = grad_fn(x)
    print(g1, aux)
    # g1 = 1/x = 0.5,  aux = 1.0 (passed through unchanged)

    # g1 carries the computation graph — continue differentiating it
    g2 = grad(g1, x, allow_unused=True)[0]
    print(g2)
    # g2 = -1/x² = -0.25

**Example 5: ``jacrev(grad(f))`` — construct the Hessian matrix in one shot**

For second-order derivatives of vector variables, ``func.jacrev(func.grad(f))``
yields the complete Hessian matrix directly — more efficient than two manual
``func.grad`` calls:

.. code-block:: python

    import riemann as rm
    from riemann import func

    # f(x) = ||x||⁴ = Σ x_i⁴,  x ∈ R³
    x = rm.tensor([1., 2., 3.], requires_grad=True)
    f = lambda xv: (xv ** 4).sum()

    # Gradient function g = ∇f
    # Hessian = Jacobian_of_forward(∇f)
    H = func.jacrev(func.grad(f))(x)
    print(H)

    # H.shape = (3, 3)
    # H is diagonal with entries 12·x_i² = [12, 48, 108]
    #
    # Why this works:  H_ij = ∂²f/∂x_i∂x_j = ∂(∇f)_i/∂x_j = Jac(∇f)_{ij}

The forward-mode variant (``jacfwd``) produces the same result:

.. code-block:: python

    import riemann as rm
    from riemann import func

    x = rm.tensor([1., 2., 3.], requires_grad=True)
    f = lambda xv: (xv ** 4).sum()
    H_fwd = func.jacfwd(func.grad(f))(x)
    print(H_fwd)
    # → identical to H

**Example 6: Batched higher-order — wrap a higher-order function in ``func.vmap``**

Apply higher-order derivatives to a batch of inputs by vectorizing a scalar
higher-order function with ``func.vmap``:

.. code-block:: python

    import riemann as rm
    from riemann import func
    from riemann.autograd import grad

    xs = rm.tensor([1., 2., 3., 4.], requires_grad=True)
    print(xs)

    # Scalar 2nd-order function:  x → f''(x) = 12x²
    def scalar_second(xv):
        f = xv ** 4
        g1 = grad(f, xv, create_graph=True, allow_unused=True)[0]
        g2 = grad(g1, xv, allow_unused=True)[0]
        return g2

    # vmap applies it to every batch element
    batch_f2 = func.vmap(scalar_second)(xs)
    # → [12·1², 12·2², 12·3², 12·4²] = [12, 48, 108, 192]

Alternatively, sum the batch into a scalar then differentiate — this requires
passing ``grad_outputs`` manually to the second ``grad`` call because the
intermediate result is a vector:

.. code-block:: python

    f_sum = (xs ** 4).sum()
    g1 = grad(f_sum, xs, create_graph=True, allow_unused=True)[0]
    g2 = grad(g1, xs, allow_unused=True,
              grad_outputs=rm.ones_like(g1))[0]   # grad_outputs required here!
    print(g2)
    # → also [12, 48, 108, 192]

**Example 7: ``TN.d()`` vs chained ``func.grad`` — equivalence**

Both approaches produce identical results for the same computation:

.. code-block:: python

    import riemann as rm
    from riemann import func

    x = rm.tensor(2.0, requires_grad=True)
    y = rm.tensor(3.0, requires_grad=True)
    f = x ** 3 * y ** 2

    # Approach A: TN.d() — most concise
    d_x2 = f.d(x, x)
    print(d_x2)
    # → 6xy² = 6·2·9 = 108.0

    # Approach B: func.grad functional chain
    f_fn = lambda xv, yv: xv ** 3 * yv ** 2
    gf = func.grad(f_fn, argnums=0)
    ggf = func.grad(lambda xv, yv: gf(xv, yv), argnums=0)
    fg_x2 = ggf(x, y)
    print(fg_x2)
    # → also 108.0

**Which approach to choose**

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Scenario
     - Recommended
     - Reason
   * - Pure higher-order on scalar graph node
     - ``f.d((x, n))``
     - One-liner; results keep the graph, so nesting works
   * - Mixed partials on multiple variables
     - ``f.d(x, y, z)``
     - Order and accumulation managed automatically
   * - Functional style with explicit fn
     - Chained ``func.grad``
     - Free composition with vmap, jacrev, jvp
   * - Vector in to Vector out, Jacobian
     - ``func.jacrev(f)`` / ``func.jacfwd(f)``
     - grad only supports scalar outputs
   * - Vector in to Vector out, 2nd-order
     - ``func.jacrev(func.jacrev(f))``
     - Jacobian-of-Jacobian = 2nd-order partial
   * - Vector in to Scalar out, Hessian
     - ``func.jacrev(func.grad(fn))``
     - Structured, one-shot; more efficient
   * - Batched higher-order
     - ``func.vmap(scalar_high_order)``
     - Natural batching, no manual batch dim management
   * - Need auxiliary value passthrough
     - ``func.grad(fn, has_aux=True)``
     - Differentiate while passing non-gradient info
   * - Higher-order in complex control flow
     - Direct ``autograd.grad`` loop
     - Lowest level, most flexible

autograd Module Derivative Functions
------------------------------------

The ``riemann.autograd`` module provides low-level reverse-mode autodiff and
gradient functions, and exposes higher-level derivative structures through the
``riemann.autograd.functional`` submodule. This chapter is organized as follows:

- Backpropagation and gradients: ``backward()``, ``grad()``, ``diff()``
- Gradient checking: ``gradcheck()``
- ``autograd.functional`` submodule: ``jacobian()``, ``hessian()``,
  ``derivative()``, ``jvp()``, ``vjp()``, ``hvp()``, ``vhp()``

**Terminology: cotangents and initial gradients**

``grad_outputs`` / ``grad_tensors`` are cotangent vectors on the output side in
mathematical terms. They can be understood simply as the "initial gradient" or
"upstream gradient". The term comes from differential geometry: the tangent
space describes directions of increments, and its dual space is the cotangent
space, whose elements are linear functionals acting on increments. In practice,
just treat them as tensors with the same shape as the corresponding output that
weight the backward pass; no geometry background is required.

**Terminology: root**

A "root" is the starting point of backpropagation: it is both an output tensor
produced by the forward computation and the node from which ``backward()`` /
``grad()`` starts traversing the computation graph. A single output is one root;
multiple outputs are multiple roots. Roots may depend on each other, or refer to
the same tensor (duplicate roots).

``backward()`` Function
~~~~~~~~~~~~~~~~~~~~~~~

**Function**

Runs reverse-mode autodiff from the roots in ``tensors`` and accumulates
gradients into the ``.grad`` attribute of the relevant leaf tensors.

**Differences from torch.autograd.backward**

- The interface matches torch, except that the deprecated ``grad_variables``
  argument is removed.
- ``retain_graph`` is kept only for compatibility; Riemann does not rely on it
  and supports repeated backward passes without it.
- Gradient accumulation into ``.grad`` follows the same semantics as torch.

**Signature**

.. code-block:: python

    riemann.autograd.backward(tensors, grad_tensors=None, retain_graph=None,
                              create_graph=False, inputs=None)

**Parameters**

- ``tensors``: A single TN or a sequence of TNs, the roots of backpropagation.
- ``grad_tensors``: Initial cotangents (initial/upstream gradients) for each root.
  Explicit cotangents can have any shape but must match the corresponding root in
  shape and device; a single TN is allowed only for a single root; otherwise pass
  a sequence with the same length as ``tensors`` (elements may be TN or None).
  When the whole argument is ``None`` or an element is ``None``, that position
  uses an implicit 1.0, which is allowed only if the corresponding root is a real
  scalar; non-scalar roots must provide an explicit cotangent. Duplicate roots
  sum their cotangents.
- ``retain_graph``: Kept only for PyTorch compatibility; Riemann does not rely
  on this flag.
- ``create_graph``: Whether to build a graph for the gradient computation, used
  for higher-order derivatives.
- ``inputs``: A single TN or a sequence. When provided, only these tensors
  accumulate ``.grad``. When ``None``, leaves always accumulate, while
  intermediate nodes accumulate only if ``retains_grad=True``.

**Returns**

``None``. Gradients are written to the ``.grad`` attributes and accumulate
across repeated calls.

**Example**

.. code-block:: python

    import riemann as rm

    # Single root
    x = rm.tensor([1.0, 2.0], requires_grad=True)
    y = (x * x).sum()
    rm.autograd.backward(y)
    print(x.grad)              # tensor([2., 4.])

    # Multiple roots: contributions from all roots are accumulated
    x2 = rm.tensor([1.0, 2.0], requires_grad=True)
    y2 = (x2 * x2).sum()
    z2 = (x2 * 3.0).sum()
    rm.autograd.backward([y2, z2])
    print(x2.grad)             # tensor([5., 7.])

``grad()`` Function
~~~~~~~~~~~~~~~~~~~

**Function**

Computes gradients of one or more outputs with respect to ``inputs`` and returns
them as a tuple, without modifying any ``.grad`` attribute. Multiple roots,
duplicate roots, and roots that depend on each other are supported.

**Differences from torch.autograd.grad**

- The interface is essentially the same, but ``retain_graph`` is kept only for
  compatibility and does not affect repeated calls.
- Torch options such as ``materialize_grads`` and ``is_grads_batched`` are not
  implemented.
- The tuple return value, the fact that ``.grad`` is not written, and
  ``allow_unused`` semantics match torch.
- Multiple roots, duplicate roots, and roots that depend on each other are
  handled the same way as in torch; when ``grad_outputs=None`` or an element of
  the sequence is ``None``, that position uses an implicit 1.0, so the corresponding root must be a real scalar (non-scalar roots must provide an explicit cotangent).

**Signature**

.. code-block:: python

    riemann.autograd.grad(outputs, inputs, retain_graph=False, grad_outputs=None,
                          create_graph=False, allow_unused=False)

**Parameters**

- ``outputs``: A single TN or a sequence of TNs (multiple roots); every root
  must have ``requires_grad=True``.
- ``inputs``: A single TN or a sequence, the inputs to differentiate with
  respect to.
- ``retain_graph``: Kept only for PyTorch compatibility.
- ``grad_outputs``: Initial cotangents (initial/upstream gradients) for the roots.
  Explicit cotangents can have any shape but must match the corresponding root in
  shape and device; a single TN is allowed only for a single root; otherwise pass
  a sequence with the same length as ``outputs`` (elements may be TN or None).
  When the whole argument is ``None`` or an element is ``None``, that position
  uses an implicit 1.0, which is allowed only if the corresponding root is a real
  scalar; non-scalar roots must provide an explicit cotangent. Explicit
  cotangents are not detached.
- ``create_graph``: Whether to build a graph for the gradient computation, used
  for higher-order derivatives.
- ``allow_unused``: When ``False``, an error is raised if an input is unrelated
  to all outputs; when ``True``, the corresponding result is ``None``.

**Returns**

``tuple[TN, ...]`` in the same order as ``inputs``. Duplicate inputs appear
repeatedly with equal values; unrelated inputs are ``None`` when
``allow_unused=True``.

**Example**

.. code-block:: python

    import riemann as rm

    x = rm.tensor([1.0, 2.0], requires_grad=True)
    y1 = x.sum()
    y2 = (x * x).sum()

    # Multiple roots: d(y1)/dx + d(y2)/dx
    gx, = rm.autograd.grad([y1, y2], x)
    print(gx)                  # tensor([3., 5.]): 1 + 2x

    # Duplicate inputs: two identical gradients
    g1, g2 = rm.autograd.grad(y1, [x, x])
    print(g1, g2)              # tensor([1., 1.]) tensor([1., 1.])

``diff()`` Function (underlying implementation of ``TN.d``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function**

Computes high-order or mixed partial derivatives of a scalar tensor with respect
to the specified variables. ``TN.d(...)`` is equivalent to
``riemann.autograd.diff(tensor, ...)``.

``diff()`` is **Riemann-specific**: PyTorch has no function with this name;
higher-order derivatives are typically assembled from ``torch.autograd.grad``
or ``torch.func`` transforms.

**Signature**

.. code-block:: python

    riemann.autograd.diff(tensor, *args, create_graph=False)

**Parameters**

- ``tensor``: A scalar TN with ``requires_grad=True``.
- ``*args``: Derivative specifications, which may be mixed: ``TN`` means a
  first derivative with respect to that variable; ``(TN, k)`` means order ``k``.
  Repeated variables accumulate their orders.
- ``create_graph``: Whether to track the graph of the derivative result so that
  higher derivatives can be taken.

**Returns**

A TN whose shape is the concatenation of each variable's shape, repeated by the
corresponding order. For example, in ``f.d(x, y)`` with ``x.shape=(2,)`` and
``y.shape=(3,)``, the result is a ``(2, 3)`` mixed-partial matrix.

**Example**

.. code-block:: python

    import riemann as rm

    # Scalar high-order derivative
    x = rm.tensor(3.0, requires_grad=True)
    f = x ** 3
    print(rm.autograd.diff(f, x))          # 27.0  (3x^2)
    print(rm.autograd.diff(f, (x, 2)))     # 18.0  (6x)

    # Mixed partials for multiple vectors
    xv = rm.tensor([1.0, 2.0], requires_grad=True)
    yv = rm.tensor([3.0, 4.0], requires_grad=True)
    fv = (xv * yv).sum() ** 2
    print(rm.autograd.diff(fv, xv, yv).shape)   # (2, 2)

``gradcheck()`` Function
~~~~~~~~~~~~~~~~~~~~~~~~

**Function**

Validates analytical gradients against a numerical Jacobian computed by central
differences. As in PyTorch, every output element is compared separately instead
of summing multiple outputs into a weaker check.

**Signature**

.. code-block:: python

    riemann.autograd.gradcheck(func, inputs, *, eps=1e-6, atol=1e-5, rtol=1e-3,
                               raise_exception=True, nondet_tol=0.0,
                               check_undefined_grad=True, check_grad_dtypes=False,
                               check_batched_grad=False,
                               check_batched_forward_grad=False,
                               check_forward_ad=False, check_backward_ad=True,
                               fast_mode=False, masked=None)

**Parameters**

- ``func``: The function to check; returns a single TN or a tuple/list of TNs.
- ``inputs``: A single TN or a sequence of TNs. Every input must have
  ``requires_grad=True`` and be ``float64`` (double precision).
- ``eps``: Central-difference step, default ``1e-6``.
- ``atol`` / ``rtol``: Absolute/relative tolerance, default ``1e-5`` / ``1e-3``.
- ``raise_exception``: Whether to raise ``RuntimeError`` on failure, default ``True``.
- ``nondet_tol``: Non-determinism tolerance; absolute errors below it are
  accepted, default ``0.0``.
- ``check_undefined_grad``: Whether to treat an analytical ``None`` gradient as
  zero during comparison, default ``True``.
- ``check_grad_dtypes``: Whether to also check that gradient dtypes match the
  inputs, default ``False``.
- ``check_backward_ad``: Whether to run the reverse-mode check; only ``True`` is
  supported.
- ``fast_mode``: Fast mode using random directional projections, default
  ``False`` (full element-wise check).
- ``check_batched_grad``, ``check_batched_forward_grad``, ``check_forward_ad``,
  ``masked``: Not implemented in this version; enabling them raises
  ``NotImplementedError``.

**Returns**

``bool``: ``True`` if the check passes; ``False`` if it fails when
``raise_exception=False``.

**Example**

.. code-block:: python

    import riemann as rm

    def f(x, y):
        return x * y

    x = rm.tensor(2.0, requires_grad=True, dtype=rm.float64)
    y = rm.tensor(3.0, requires_grad=True, dtype=rm.float64)
    print(rm.autograd.gradcheck(f, (x, y)))   # True

``autograd.functional.jacobian()`` Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function**

Computes the Jacobian matrix of a function from inputs to outputs, showing all
first-order partial derivatives of the outputs with respect to the inputs.
Single/multiple inputs and single/multiple outputs are supported.

**Signature**

.. code-block:: python

    riemann.autograd.functional.jacobian(func, inputs, create_graph=False, strict=True)

**Parameters**

- ``func``: Forward function taking TN(s) and returning a TN or a sequence of TNs.
- ``inputs``: A single TN or a sequence of TNs.
- ``create_graph``: Whether to build a graph for the derivative computation.
- ``strict``: When ``True``, an error is raised if an output is unrelated to an
  input; when ``False``, the corresponding Jacobian block is filled with zeros.

**Returns**

The structure follows the input/output structure:

- Single input, single output: a TN of shape ``output.shape + input.shape``.
- Single input, multiple outputs: ``tuple[TN]`` in output order.
- Multiple inputs, single output: ``tuple[TN]`` in input order, each of shape
  ``output.shape + input_i.shape``.
- Multiple inputs, multiple outputs: ``tuple[tuple[TN, ...], ...]``, outer by
  output and inner by input.

**Example**

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
    print(jac.shape)  # (3, 3)  # For vector input/vector output, shape is (n_outputs, n_inputs)

``autograd.functional.hessian()`` Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function**

Computes the Hessian matrix (the Jacobian of the gradient) of a scalar-valued
function, showing all second-order partial derivatives with respect to the
inputs. Single/multiple inputs are supported.

**Signature**

.. code-block:: python

    riemann.autograd.functional.hessian(func, inputs, create_graph=False, strict=True)

**Parameters**

- ``func``: Forward function returning a scalar TN.
- ``inputs``: A single TN or a sequence of TNs.
- ``create_graph``: Whether to build a graph for the derivative computation.
- ``strict``: When ``True``, an error is raised if the output is unrelated to an
  input; when ``False``, the corresponding block is filled with zeros.

**Returns**

- Single input: a TN of shape ``input.shape + input.shape``.
- Multiple inputs: ``tuple[tuple[TN, ...], ...]`` where ``result[i][j]`` is
  ``∂²f / (∂input_i ∂input_j)`` with shape ``input_i.shape + input_j.shape``.

**Example**

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
    print(hess.shape)  # ()  # Scalar input returns a scalar second derivative

``autograd.functional.derivative()`` Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function**

Returns the derivative function of the given function. When the returned
function is called, it computes the Jacobian (i.e., the derivative) of the
original function at the given inputs. It is implemented on top of
``jacobian()``.

**Signature**

.. code-block:: python

    riemann.autograd.functional.derivative(func, create_graph=False)

**Parameters**

- ``func``: Forward function taking one or more TNs.
- ``create_graph``: Whether to build a graph for the derivative computation.

**Returns**

A callable accepting the same inputs as the original function. Its result is
always a ``tuple``: one derivative tensor per input; for multiple outputs, an
extra tuple level is nested per output, consistent with the multi-input /
multi-output return structure of ``jacobian()``.

**Example**

.. code-block:: python

    import riemann as rm
    
    # Define function f = x^2
    def f(x):
        return x ** 2.
    
    # Create derivative function
    df = rm.autograd.functional.derivative(f)
    
    # Test the derivative function
    x = rm.tensor(2.0, requires_grad=True)
    print(df(x))  # (tensor(4.),) — always a tuple; scalar input yields a 0-D derivative
    
    # Multi-input example
    def g(x, y):
        return x * y + x ** 2.
    
    x = rm.tensor(2.0, requires_grad=True)
    y = rm.tensor(3.0, requires_grad=True)
    dg = rm.autograd.functional.derivative(g)
    print(dg(x, y))  # (tensor(7.), tensor(2.)) — ∂g/∂x = y + 2x, ∂g/∂y = x
    
    # create_graph=True enables higher-order derivatives
    d2f = rm.autograd.functional.derivative(lambda t: t ** 3., create_graph=True)
    print(d2f(x))  # (tensor(12.),) — 3x^2; requires_grad=True so it can be differentiated again

``autograd.functional.jvp()`` Function (Jacobian-Vector Product)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function**

Computes the Jacobian-vector product J·v at the given point.

**Signature**

.. code-block:: python

    riemann.autograd.functional.jvp(func, inputs, v=None, create_graph=False, strict=False)

**Parameters**

- ``func``: Forward function taking TN(s) and returning a TN or a sequence of TNs.
- ``inputs``: A single TN or a sequence of TNs.
- ``v``: Input direction; its shape must match the corresponding input. For a
  single scalar input it may be omitted (defaults to 1.0); for multiple inputs a
  sequence with the same length as ``inputs`` is required.
- ``create_graph``: Whether to build a graph for the derivative computation.
- ``strict``: When ``True``, an error is raised if an output is unrelated to an
  input.

**Returns**

``(func_output, jvp_result)``. ``func_output`` has the same structure as the
return value of ``func``; ``jvp_result`` is a TN for a single output, or a tuple
in output order for multiple outputs.

**Example**

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

``autograd.functional.vjp()`` Function (Vector-Jacobian Product)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function**

Computes the vector-Jacobian product vᵀJ for a given vector.

**Signature**

.. code-block:: python

    riemann.autograd.functional.vjp(func, inputs, v=None, create_graph=False, strict=False)

**Parameters**

- ``func``: Forward function taking TN(s) and returning a TN or a sequence of TNs.
- ``inputs``: A single TN or a sequence of TNs.
- ``v``: Output-side cotangent; its shape must match the corresponding output of
  ``func``. When ``None``, it is automatically filled with ones matching each
  output; for multiple outputs, a sequence with the same length as the outputs
  is required.
- ``create_graph``: Whether to build a graph for the derivative computation.
- ``strict``: When ``True``, an error is raised if an output is unrelated to an
  input; when ``False``, the corresponding input receives a zero tensor.

**Returns**

``(func_output, vjp_result)``. ``func_output`` has the same structure as the
return value of ``func``; ``vjp_result`` is a TN for a single input, or a tuple
in input order for multiple inputs.

**Example**

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

``autograd.functional.hvp()`` Function (Hessian-Vector Product)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function**

Computes the product of the Hessian matrix with a vector v as H·v, without
explicitly constructing the full Hessian.

**Signature**

.. code-block:: python

    riemann.autograd.functional.hvp(func, inputs, v, create_graph=False, strict=False)

**Parameters**

- ``func``: Forward function that must return a scalar TN.
- ``inputs``: A single TN or a sequence of TNs.
- ``v``: A vector with the same shape as ``inputs`` (a sequence of the same
  length for multiple inputs).
- ``create_graph``: Whether to build a graph for the derivative computation.
- ``strict``: When ``True``, an error is raised if the output is unrelated to an
  input; when ``False``, the result for an unused input is a zero tensor of the
  same shape.

**Returns**

``(func_output, hvp_result)``. ``func_output`` is a scalar TN; ``hvp_result`` is
a TN with the same shape as the input for a single input, or a tuple in input
order for multiple inputs.

**Example**

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

``autograd.functional.vhp()`` Function (Vector-Hessian Product)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function**

Computes the product of a vector v with the Hessian matrix as vᵀH. Because the
Hessian is symmetric, the numerical result is the same as ``hvp()``.

**Signature**

.. code-block:: python

    riemann.autograd.functional.vhp(func, inputs, v, create_graph=False, strict=False)

**Parameters**

- ``func``: Forward function that must return a scalar TN.
- ``inputs``: A single TN or a sequence of TNs.
- ``v``: A vector with the same shape as ``inputs`` (a sequence of the same
  length for multiple inputs).
- ``create_graph``: Whether to build a graph for the derivative computation.
- ``strict``: When ``True``, an error is raised if the output is unrelated to an
  input; when ``False``, the result for an unused input is a zero tensor of the
  same shape.

**Returns**

``(func_output, vhp_result)``. ``func_output`` is a scalar TN; ``vhp_result`` is
a TN with the same shape as the input for a single input, or a tuple in input
order for multiple inputs.

**Example**

.. code-block:: python

    import riemann as rm

    def f(x):
        return (x ** 3).sum()

    x = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
    v = rm.tensor([1.0, 1.0, 1.0])

    f_x, vhp_val = rm.autograd.functional.vhp(f, x, v)
    print(vhp_val)             # tensor([6., 12., 18.]): H = diag(6x)

End-to-End Example: From Function Definition to Higher-Order Derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following complete example walks through the common interfaces in this
chapter: define a forward function → forward computation → first-order gradient →
Hessian / higher-order derivative → numerical check → accumulate into ``.grad``.
Use ``f(w, x) = (w·x)^2`` so the results can be verified by hand.

.. code-block:: python

    import riemann as rm

    # 1. Define the forward function
    def loss_fn(w, x):
        return (w * x).sum() ** 2

    # 2. Create leaf tensors; float64 is used for gradcheck
    w = rm.tensor([1.0, 2.0], requires_grad=True, dtype=rm.float64)
    x = rm.tensor([3.0, 4.0], requires_grad=True, dtype=rm.float64)

    # 3. Forward computation: w·x = 11, loss = 121
    loss = loss_fn(w, x)
    print(loss)                    # tensor(121., dtype=float64, ...)

    # 4. First-order gradients (returned as a tuple, .grad untouched)
    gw, gx = rm.autograd.grad(loss, [w, x])
    print(gw)                      # tensor([66., 88.], dtype=float64)
    print(gx)                      # tensor([22., 44.], dtype=float64)

    # 5. Higher-order derivative: Hessian with respect to w
    hess_w = rm.autograd.functional.hessian(lambda wv: loss_fn(wv, x), w)
    print(hess_w)
    # tensor([[18., 24.],
    #         [24., 32.]], dtype=float64)

    # 6. Obtain the same second-order result with diff
    d2 = rm.autograd.diff(loss, (w, 2))
    print(d2)                      # same as the Hessian above

    # 7. Numerically validate the analytical gradients
    print(rm.autograd.gradcheck(lambda wv: loss_fn(wv, x), w))   # True

    # 8. Training-style usage: backpropagate and accumulate into .grad
    loss = loss_fn(w, x)
    loss.backward()
    print(w.grad)                  # tensor([66., 88.], dtype=float64)

**Result notes**

- ``∂f/∂w = 2(w·x)x = [66, 88]`` and ``∂f/∂x = 2(w·x)w = [22, 44]``.
- ``H = ∂²f/∂w² = 2xxᵀ = [[18, 24], [24, 32]]``.
- ``grad()`` and ``functional.hessian()`` do not write ``.grad``; only
  ``backward()`` accumulates into ``w.grad``.

func Module Derivative Functions
--------------------------------

The ``riemann.func`` module provides a set of **functional autodiff primitives**,
designed after PyTorch's ``torch.func``. Compared to the low-level
``autograd.functional`` module, ``func`` adopts a modern **currying style**:
you first pass in the function and configuration, which returns a **wrapped
function**; the wrapped function only executes the forward and backward pass
when called with concrete inputs.

The key advantage of this style is that wrapped functions are themselves
composable — they can be freely nested with ``vmap``, higher-order ``grad``,
and other primitives to build compound transformations like
``vmap(jacrev(grad(f)))``.

Key Differences from autograd.functional
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Feature
     - ``autograd.functional``
     - ``func``
   * - Call style
     - ``fn(f, *inputs)`` — pass inputs directly
     - ``fn(f, ...)(*inputs)`` — returns Callable
   * - argnums support
     - Some functions support multi-input tuples
     - Fully supported; results filtered by argnums
   * - has_aux
     - Not supported
     - Supported by ``grad`` / ``grad_and_value``
   * - Higher-order derivatives
     - Needs explicit ``create_graph=True``
     - No ``create_graph`` param for grad / jacrev / jacfwd / hessian; results can be re-differentiated
   * - Composability
     - Hard to combine with ``vmap`` etc.
     - All functions return Callable; nest freely
   * - vjp cotangent
     - Requires explicit cotangents
     - Explicit cotangents; ``vjp_fn`` returns a tuple
   * - Non-leaf inputs
     - Requires leaf tensors with ``requires_grad``
     - Auto ``detach().requires_grad_(True)``

**Import**:

.. code-block:: python

    import riemann as rm
    from riemann import func
    # or import specific functions
    from riemann.func import grad, jvp, vjp, jacrev, jacfwd, vmap, functional_call

``grad()`` — Gradient of a Scalar-Valued Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``func.grad`` returns a wrapped function that computes the gradient of a
scalar-valued function with respect to the specified argument positions.

**Function Signature**:

.. code-block:: python

    riemann.func.grad(func, argnums=0, has_aux=False) -> Callable

**Parameters**:

- **func**: Differentiable function that returns a scalar tensor (``ndim == 0``); otherwise a ``RuntimeError`` is raised
- **argnums** (int or tuple[int,...]): Argument positions to differentiate, default ``0``:

  - int: differentiate with respect to that single argument, e.g. ``argnums=1`` for the second argument of ``func``
  - tuple of ints: differentiate with respect to each listed argument position, e.g. ``argnums=(0, 1)``
  - invalid values: ``()``, ``None``, tuples containing non-int elements, lists, etc. raise ``RuntimeError`` when the derivative function is called, with messages identical to ``torch.func.grad`` (``()`` gives ``argnums must be non-empty``; ``None`` gives ``argnums must be int or Tuple[int, ...], got: <class 'NoneType'>``)
- **has_aux** (bool): If ``True``, ``func`` must return a ``(value, aux)`` pair. ``value`` is ``func``'s forward-computation output and ``grad`` requires it to be a scalar tensor; ``aux`` is an additional result produced by ``func`` (any object, not differentiated). The derivative function computes gradients only from ``value`` and passes ``aux`` through unchanged in its return value, so non-differentiable results (such as loss values, statistics, or other intermediate outputs) can be obtained from the same forward pass without recomputing them

**Returns**: The derivative function of ``func`` (a function object) that
accepts the same arguments as ``func``. When called, it returns:

- ``has_aux=False``: ``grad_result``
- ``has_aux=True``: ``(grad_result, aux)``, where ``aux`` is passed through unchanged from ``func``
- ``grad_result`` is the gradient of the scalar output ``value`` with respect to the arguments selected by ``argnums``: exactly one selected position returns a single gradient tensor (same shape as the corresponding input), while multiple positions return a tuple of gradients in the order of ``argnums``

As in ``torch.func.grad``, the selected arguments do not need to carry ``requires_grad`` (an internal differentiable copy is used, leaving the user's tensor untouched). If ``value`` is disconnected from a selected argument, or ``value`` itself is not differentiable (a constant or already ``detach()``-ed), that position returns a **zero tensor** of the same shape instead of raising an error.

.. note::

    Three outputs are easy to confuse: ``func``'s output is ``value`` (``(value, aux)`` when ``has_aux=True``); ``func.grad(...)``'s output is a function object (the derivative function); and the derivative function's output is ``grad_result`` (``(grad_result, aux)`` when ``has_aux=True``).

**Example**:

.. code-block:: python

    import riemann as rm
    from riemann import func

    # Basic usage
    def f(x, y):
        return (x ** 2).sum() + x @ y   # scalar output

    x = rm.tensor([1., 2.], requires_grad=True)
    y = rm.tensor([3., 4.], requires_grad=True)

    g = func.grad(f, argnums=0)      # differentiate with respect to x only
    grad_x = g(x, y)
    print(grad_x)  # 2*x + y = [5, 8]

    # Multi-input differentiation: argnums=(0, 1) returns a tuple
    g_both = func.grad(f, argnums=(0, 1))
    gx, gy = g_both(x, y)
    print(gx)  # same as grad_x
    print(gy)  # x = [1., 2.]

    # has_aux: return auxiliary data alongside the gradient
    def f_with_aux(x):
        return (x ** 3).sum(), float((x ** 2).sum() ** 0.5)

    g_aux = func.grad(f_with_aux, has_aux=True)
    grad, aux = g_aux(rm.tensor([1., 2., 3.], requires_grad=True))
    print(grad)  # [3., 12., 27.]
    print(aux)   # sqrt(1+4+9) ≈ 3.7417

    # Higher-order derivatives: the inner grad
    # returns a gradient that still has a computation graph; the outer
    # grad needs a scalar output, so we .sum() the gradient to scalarize
    grad_fn = func.grad(lambda x: (x ** 4).sum())
    g_higher = func.grad(lambda x: grad_fn(x).sum())
    h = g_higher(rm.tensor([1., 2., 3.], requires_grad=True))
    print(h)  # Second derivative: 12*x² = [12., 48., 108.]

**Notes**:

- ``func`` output **must be a scalar** (``ndim == 0``), otherwise a ``RuntimeError`` is raised
- The form of the return value depends on the number of positions in ``argnums``: one position returns a single tensor, multiple positions return a tuple

``grad_and_value()`` — Gradient and Function Value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

More efficient than calling ``grad`` separately: internally performs one
forward + one backward pass instead of two.

**Function Signature**:

.. code-block:: python

    riemann.func.grad_and_value(func, argnums=0, has_aux=False) -> Callable

**Parameters**: Same as ``func.grad`` (invalid ``argnums`` values raise ``RuntimeError`` when the derivative function is called)

**Returns**: The derivative function of ``func`` (a function object) accepting
the same arguments as ``func``; when called it returns ``(grad_result, value)``,
or ``(grad_result, value, aux)`` when ``has_aux=True``. The form of
``grad_result`` is determined by ``argnums`` (same as ``func.grad``)

As in ``func.grad``, the selected arguments do not need ``requires_grad``; disconnected or
non-differentiable positions return **zero tensors** instead of raising an error.

**Example**:

.. code-block:: python

    import riemann as rm
    from riemann import func

    def loss(params):
        return (params ** 2).sum()

    gav = func.grad_and_value(loss)
    grad, val = gav(rm.tensor([1., 2., 3.], requires_grad=True))
    # grad = [2., 4., 6.], val = 14.0

``jvp()`` — Jacobian-Vector Product
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Computes the product of the Jacobian matrix with a given vector. ``jvp`` is a
true forward-mode primitive: it performs one forward evaluation, then propagates
the tangent forward through the computation graph without any backward pass,
which is more efficient than explicitly constructing the full Jacobian and then
multiplying.

Implementation-wise, ``jvp`` is driven by per-operator direct forward rules
(``_jvp_rule``): the PRIMAL phase builds the graph and registers rules, then the
TANGENT phase traverses the topological order once and computes each node's
tangent. Missing rules raise an error (all built-in operators are covered).

In-place operators are applied as ordered steps on the
tangent: ``add_/sub_/mul_/div_/pow_/remainder_/copy_/fill_/abs_/neg_`` and index
assignments are supported, while ``//=`` (``floor_divide_``) raises because it has
no derivative, consistent with the forward rules.

**Function Signature**:

.. code-block:: python

    riemann.func.jvp(func, primals, tangents, *, strict=False, has_aux=False) -> (output, jvp_result[, aux])

**Parameters**:

- **func**: Differentiable function
- **primals**: Function input tuple (a single input must still be wrapped in a 1-tuple, matching ``torch.func.jvp``)
- **tangents**: Tangent vector tuple with the same length as ``primals``; each entry has the same shape as its primal
- **strict** (bool, keyword-only): Raise error when an output is completely independent of an input
- **has_aux** (bool): When ``True``, ``func`` must return ``(output, aux)``; only
  ``output`` is differentiated and ``aux`` is appended to the return value

**Returns**: ``(output, jvp_result)``, or ``(output, jvp_result, aux)`` when
``has_aux=True`` — output value and JVP result (single tensor for single input,
tuple for multiple inputs)

**Example**:

.. code-block:: python

    import riemann as rm
    from riemann import func

    def f(x):
        return x ** 3  # f: R^n -> R^n, vector output

    x = rm.tensor([1., 2., 3.], requires_grad=True)
    v = rm.tensor([0.5, 1., 0.5], requires_grad=False)

    out, jvp_val = func.jvp(f, (x,), (v,))
    print(out)      # [1., 8., 27.]
    print(jvp_val)  # 3*x^2 * v = [1.5, 12., 13.5]

    # Multi-input
    def g(x, y):
        return (x * y).sum()

    out2, jvp_result = func.jvp(g,
                          (rm.tensor([1., 2.], requires_grad=True),
                           rm.tensor([3., 4.], requires_grad=True)),
                          (rm.tensor([1., 0.], requires_grad=False),
                           rm.tensor([0., 1.], requires_grad=False)))
    print(jvp_result)  # 1*3 + 2*1 = 5.0

``vjp()`` — Vector-Jacobian Product
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Computes the product of a given vector with the Jacobian matrix (the reverse
dual of JVP). For scalar outputs ``f: R^n -> R``, ``vjp_fn(1)`` is equivalent
to ``grad(f)(x)``.

**Function Signature**:

.. code-block:: python

    riemann.func.vjp(func, *primals, has_aux=False) -> (output, vjp_fn[, aux])

**Parameters**:

- **primals**: Evaluation point inputs, in the positional order of ``func`` (must be tensors; ``requires_grad`` is handled internally)
- **has_aux** (bool): When ``True``, ``func`` must return ``(output, aux)``; only
  ``output`` is differentiated and ``aux`` is appended to the return value

**Returns**: ``(output, vjp_fn)``, or ``(output, vjp_fn, aux)`` when ``has_aux=True``.
Call ``vjp_fn(cotangents, retain_graph=True, create_graph=None)``: ``cotangents``
must match every output in structure and shape, and the result is always a tuple
(a 1-tuple for a single primal, matching ``torch.func``). ``create_graph=None``
uses the current grad mode; ``retain_graph`` is accepted for interface parity only.

**Example**:

.. code-block:: python

    import riemann as rm
    from riemann import func

    def f(x):
        return (x ** 2).sum() + x.sum()   # scalar output

    x = rm.tensor([1., 2., 3.], requires_grad=True)

    # The forward pass runs once; vjp_fn can be called repeatedly
    out, vjp_fn = func.vjp(f, x)
    print(out)                         # 20.0 — forward output f(x)
    print(vjp_fn(rm.tensor(1.0))[0])   # 2x + 1 = [3., 5., 7.]

    # Custom cotangent (equivalent to a weighted sum of the outputs)
    print(vjp_fn(rm.tensor(2.0))[0])   # 2 * (2x + 1) = [6., 10., 14.]

    # Vector output
    def g(x):
        return x ** 2   # R^3 -> R^3, vector output

    out3, vjp_fn_g = func.vjp(g, x)
    print(vjp_fn_g(rm.ones_like(x, requires_grad=False))[0])  # J^T @ ones = [2., 4., 6.]

``linearize()`` — Linearization from a Single Forward Pass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Linearizes a function at ``primals`` and returns ``(primal_out, linear_fn)``.
The forward pass runs once and records the primal graph; ``linear_fn`` can then
be called repeatedly with different tangents along the same graph to obtain
``J(x)·v`` (equivalent to ``torch.func.linearize``). This is also the semantics
used internally by ``jacfwd``: a constant term plus a linear term, i.e.
``func(x + v) ≈ primal_out + linear_fn(v)``.

**Function Signature**:

.. code-block:: python

    riemann.func.linearize(func, *primals) -> tuple[output, Callable]

**Parameters**:

- **func**: Function to linearize; takes tensors and returns one tensor or a tuple of tensors
- **primals**: Evaluation-point inputs, in the same positional order as ``func``; all must be tensors

**Returns**: ``(primal_out, linear_fn)``. ``primal_out`` has the same structure as
``func``'s output; ``linear_fn(*tangents)`` returns ``J(x)·v``, and the number and
shapes of the tangents must match ``primals``.

``linear_fn`` is linear in the tangents (it keeps only first-order dependence), so its
result can be differentiated once more, while a second derivative of ``linear_fn`` is
always a **zero tensor** (consistent with ``torch.func``). For genuine higher-order
derivatives, differentiate the original ``func`` with ``grad`` / ``jvp``.

**Example**:

.. code-block:: python

    import riemann as rm
    from riemann import func

    x = rm.tensor([1., 2., 3.], requires_grad=True)
    out, lin = func.linearize(lambda t: t * t, x)
    print(out)                            # [1., 4., 9.]
    print(lin(rm.tensor([1., 0., 1.])))   # [2., 0., 6.] — J(x)·v = 2x⊙v
    print(lin(rm.tensor([1., 1., 1.])))   # [2., 4., 6.] — same graph reused

``jacrev()`` / ``jacfwd()`` — Jacobian Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two ways to compute the Jacobian matrix:

- **``jacrev``**: Reverse-mode autodiff, implemented via vmap(vjp). Performs
  ``m`` backward passes for ``m``-dimensional output. Best when input dimension
  is small and output dimension is large
- **``jacfwd``**: Forward-mode autodiff, implemented as linearize + vmap(jvp) on
  top of the direct-rule ``func.jvp`` primitive. The primal graph is recorded
  once; each input basis vector then only re-propagates its one-hot tangent
  through the operators' ``_jvp_rule``. Cost is proportional to ``n`` tangent
  propagations for ``n`` input dimensions (no repeated forward pass). Best when
  input dimension is large and output dimension is small

**Function Signature**:

.. code-block:: python

    riemann.func.jacrev(func, argnums=0, *, has_aux=False, chunk_size=None) -> Callable
    riemann.func.jacfwd(func, argnums=0, has_aux=False, *, randomness='error') -> Callable

**Parameters**:

- **func**: Differentiable function
- **argnums**: Input position(s) whose Jacobian is computed; int or tuple of ints, default ``0``. Invalid values (``()``, ``None``, tuples containing non-int elements, lists, etc.) raise ``RuntimeError`` when the derivative function is called, with messages identical to ``torch.func`` (``torch.func.jacrev`` raises for both ``()`` and ``None``; ``torch.func.jacfwd`` raises for ``()`` but treats ``None`` as all arguments)
- **has_aux**: When ``True``, ``func`` must return ``(output, aux)``; only ``output`` is differentiated and the result becomes ``(Jacobian, aux)`` (keyword-only for ``jacrev``)
- **chunk_size**: Keyword-only, ``jacrev`` only. Number of output basis vectors processed per chunk, to bound peak memory; ``None`` means process all at once. The result is identical to the unchunked computation; non-positive or non-int values raise ``ValueError``. Like ``torch.func``, ``jacfwd`` has no ``chunk_size``
- **randomness**: Keyword-only, ``jacfwd`` only. Accepts ``'error'`` / ``'different'`` / ``'same'``; invalid values raise ``RuntimeError`` when the derivative function is called (matching ``torch.func``); no randomness re-parameterization is performed

**Return Shape**: For output shape ``(y1, y2, ...)`` and input shape
``(x1, x2, ...)``, the Jacobian shape is ``(y1, y2, ..., x1, x2, ...)``.

**Example**:

.. code-block:: python

    import riemann as rm
    from riemann import func

    # Scalar output f: R^3 -> R, Jacobian shape (1, 3) — equivalent to gradient
    f = lambda x: (x ** 3).sum()
    jac = func.jacrev(f)(rm.tensor([1., 2., 3.], requires_grad=True))
    print(jac.shape)  # (3,) — unwrapped for single input
    print(jac)        # [3., 12., 27.]

    # Vector output f: R^3 -> R^2, Jacobian shape (2, 3)
    g = lambda x: rm.stack([x.sum(), (x ** 2).sum()])
    jac_g = func.jacrev(g)(rm.tensor([1., 2., 3.], requires_grad=True))
    print(jac_g.shape)  # (2, 3)

**Notes**:

- ``jacfwd`` in Riemann records the primal graph once and reuses it for every
  basis vector (linearize + per-column tangent propagation, equivalent to
  ``vmap(jvp)``); time complexity is proportional to input dimension. Use
  ``jacrev`` when output dimension is much larger than input dimension
- Multi-input (``argnums=(0, 1)``) returns a tuple, one Jacobian per input

``hessian()`` — Hessian Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Computes the Hessian matrix (second-order partial derivatives) of a
scalar-valued function. Implemented as ``jacrev(grad(func))``: reverse-mode
``grad`` first produces the gradient, then reverse-mode ``jacrev``
differentiates that gradient. (Using ``jacrev(jacfwd(func))`` instead would
re-propagate every basis vector inside each outer reverse pass, making the total
O(n²); reverse-over-reverse avoids that.)

**Function Signature**:

.. code-block:: python

    riemann.func.hessian(func, argnums=0) -> Callable

**Parameters**:

- **func**: Differentiable function that returns a scalar tensor
- **argnums**: Input position(s) whose Hessian is computed; int or tuple of ints, default ``0``. Invalid values (``()``, ``None``, tuples containing non-int elements, lists, etc.) raise ``RuntimeError`` when the derivative function is called, with messages identical to ``torch.func.hessian``
- No ``create_graph`` / ``strict`` / ``has_aux`` parameter: results always keep the graph and disconnected Hessian blocks are returned as zeros (matching ``torch.func``)

**Return Shape**: For input shape ``(n,)``, Hessian shape is ``(n, n)``.

**Example**:

.. code-block:: python

    import riemann as rm
    from riemann import func

    f = lambda x: (x ** 4).sum()
    h = func.hessian(f)(rm.tensor([1., 2., 3.], requires_grad=True))
    print(h.shape)  # (3, 3)
    # f''(x) = diag(12 * x^2) = diag([12, 48, 108])

**Notes**:

- Only supports scalar-valued functions; vector output raises an error
- When ``argnums`` is not all argument positions, only the selected input blocks are computed and returned

``vmap()`` — Vectorized Map
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extends a function designed for **single samples** to operate over a
**batch dimension**. Internally implemented via a loop + stack, with an
API matching ``torch.func.vmap``, enabling composition with function
primitives like ``jacrev``.

**Function Signature**:

.. code-block:: python

    riemann.func.vmap(func, in_dims=0, out_dims=0, randomness='error', *, chunk_size=None) -> Callable

**Parameters**:

- **in_dims**: Batch dimension for each input. Possible values:
  ``int`` — all inputs share this dimension;
  ``tuple[int|None, ...]`` — per-input dimension position, whose length must equal the
  number of inputs; ``None`` means the input is not batched (passed through directly,
  useful for shared quantities like model parameters). A top-level ``in_dims=None`` is
  invalid (matching ``torch.func``).
- **out_dims**: Batch dimension for each output; same semantics as ``in_dims``. When given
  as a tuple its length must equal the number of outputs and its elements cannot be ``None``
- **randomness**: Accepts ``'error'`` / ``'different'`` / ``'same'``; invalid values raise
  ``RuntimeError`` when the wrapped function is created (matching ``torch.func``); no
  randomness re-parameterization is performed
- **chunk_size** (keyword-only): Number of batch samples processed per chunk, controlling loop and
  stacking granularity; ``None`` means the whole batch at once. The result is
  identical to the unchunked computation; non-positive or non-int values raise ``ValueError``

**Example**:

.. code-block:: python

    import riemann as rm
    from riemann import func

    # Basic usage: scalar output batching
    f = lambda x: (x ** 2).sum()
    x_batch = rm.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
    batched_f = func.vmap(f)
    result = batched_f(x_batch)
    print(result)  # [14., 77.]

    # Multi-input with different in_dims
    def g(x, y):
        return (x * y).sum()

    x_batch = rm.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    y_batch = rm.tensor([[10., 20.], [30., 40.]], requires_grad=True)
    batched_g = func.vmap(g, in_dims=(0, 0))
    print(batched_g(x_batch, y_batch))  # [50., 250.]

    # Multi-input with in_dims=None for broadcast semantics
    y_single = rm.tensor([10., 20.], requires_grad=True)
    batched_g2 = func.vmap(g, in_dims=(0, None))
    print(batched_g2(x_batch, y_single))  # [30., 110.]

    # Composition with jacrev: batched Jacobian
    def scalar_fn(x):
        return (x ** 2).sum()

    batched_jac = func.vmap(func.jacrev(scalar_fn))
    jac_batch = batched_jac(x_batch)
    print(jac_batch.shape)  # (2, 3) — ∇f = 2x per sample

**Notes**:

- ``vmap`` is implemented as a **sequential loop** + stack, not true SIMT
  parallelism on GPU; but it correctly tracks gradients when composed
  with ``jacrev`` / ``hessian``
- Batch dimension is typically ``dim=0``; negative dimensions (e.g.
  ``dim=-1``) are normalized automatically
- Functions returning tuple outputs are also supported; each output is
  stacked independently along ``out_dims``

``functional_call()`` — Stateless Module Invocation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calls a Module's ``forward`` with **custom parameters and buffers** without
modifying the Module's own state. This is essential when you need to
differentiate through the Module's parameters themselves (e.g., hypernetworks,
meta-learning) — a regular ``module(x)`` call captures ``self.weight`` etc. in
the forward graph, while ``functional_call`` lets you substitute values by name.
The API matches ``torch.func.functional_call``.

**Function Signature**:

.. code-block:: python

    riemann.func.functional_call(module, parameter_and_buffer_dicts,
                                 args=None, kwargs=None,
                                 *, tie_weights=True, strict=False) -> Any

**Parameters**:

- **module**: Object with ``forward`` and ``named_parameters`` methods
  (typically a ``riemann.nn.Module``)
- **parameter_and_buffer_dicts**: ``{name: tensor}`` following state_dict naming
  (e.g. ``"weight"``, ``"l1.bias"``). Parameters and buffers may live in the same
  dict, or be split across multiple disjoint dicts. ``None`` falls back to the
  Module's own state. A positional tensor sequence in ``named_parameters()`` order
  is still accepted for backward compatibility
- **args** / **kwargs**: Positional and keyword arguments for ``forward``
- **tie_weights**: When ``True`` (default), tied weights are respected: providing
  one name of a tied group updates the whole group, and conflicting objects raise
  ``ValueError``. When ``False``, only explicitly given names are replaced
- **strict**: When ``True``, the provided names must exactly match the Module's
  parameter/buffer names; missing or extra keys raise ``RuntimeError``

**Example**:

.. code-block:: python

    import riemann as rm
    from riemann import func

    # Use the built-in Linear layer; weight/bias are auto-registered to _parameters
    layer = rm.nn.Linear(3, 2)
    x = rm.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=False)

    # Call with custom parameters (does not mutate layer.weight/bias)
    alt_weight = rm.tensor([[1., 0., 0.], [0., 0., 1.]], requires_grad=True)
    alt_bias = rm.tensor([0.5, -0.5], requires_grad=True)

    out = func.functional_call(layer, [alt_weight, alt_bias], x)
    print("out:", out)
    # layer.weight and layer.bias are unchanged

    # Combine with grad: differentiate with respect to substitute parameters
    loss = func.grad_and_value(lambda *params:
        (func.functional_call(layer, params, x) ** 2).sum()
    )
    g, v = loss(alt_weight, alt_bias)
    print("grad:", g)
    print("value:", v)

**Notes**:

- Names follow state_dict conventions with dotted paths for nested modules
  (e.g. ``"l1.weight"``); buffers such as BatchNorm ``running_mean`` are supported
- Parameters and buffers are **automatically restored** after the call (try-finally)
- For parameterless modules, a regular ``forward`` call is performed
- The implementation temporarily swaps and restores state, so it is not
  thread-safe and does not truly parallelize under ``vmap``; use
  ``stack_module_state`` for cross-model batching

``stack_module_state()`` — Stack Multiple Module States
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stacks the parameters and buffers of several homogeneous Modules along a new
dimension 0, returning ``(params, buffers)`` for use with
``functional_call`` + ``vmap`` (model ensembles).

**Function Signature**:

.. code-block:: python

    riemann.func.stack_module_state(models) -> tuple[dict, dict]

**Parameters**:

- **models**: A sequence or ModuleList of Modules, at least one; all must be the
  same class, in the same training/eval mode, with identical parameter/buffer
  names (otherwise ``RuntimeError`` is raised)

**Returns**: ``(params, buffers)``, two ``{name: stacked tensor}`` dicts;
``requires_grad`` semantics of the original parameters are preserved

**Example**:

.. code-block:: python

    import riemann as rm
    from riemann import func

    # Substitute parameters and buffers by name
    layer = rm.nn.Linear(3, 2)
    x = rm.tensor([[1., 2., 3.]])
    y = func.functional_call(
        layer, {"weight": rm.zeros(2, 3), "bias": rm.zeros(2)}, (x,))
    print(y)  # all zeros

    # Ensemble: stack states, then batch the forward pass with vmap
    models = [rm.nn.Linear(3, 2) for _ in range(4)]
    params, buffers = func.stack_module_state(models)
    print(params["weight"].shape)  # (4, 2, 3)

    batch_out = func.vmap(
        lambda w, b, inp: func.functional_call(
            models[0], {"weight": w, "bias": b}, (inp,)),
        in_dims=(0, 0, None))(params["weight"], params["bias"], x)
    print(batch_out.shape)  # (4, 1, 2)

Composition and Higher-Order Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The core strength of ``func`` is that **all functions return a Callable**,
making them freely composable. Here are common higher-order patterns:

**Pattern 1: vmap + jacrev — batched differentiation**

.. code-block:: python

    import riemann as rm
    from riemann import func

    def loss(x):
        return (x ** 2).sum()

    x_batch = rm.tensor([[1., 2.], [3., 4.], [5., 6.]], requires_grad=True)
    grads = func.vmap(func.jacrev(loss))(x_batch)
    print(grads.shape)  # (3, 2)

**Pattern 2: jacrev(jacrev(f)) — Hessian via nested jacrev**

.. code-block:: python

    import riemann as rm
    from riemann import func

    def f(x):
        return (x ** 4).sum()

    x = rm.tensor([1., 2., 3.], requires_grad=True)
    h_via_jac = func.jacrev(func.jacrev(f))(x)
    print(h_via_jac)
    # Diagonal matrix with entries [12·1², 12·2², 12·3²] = [12, 48, 108]

**Pattern 3: functional_call + grad — differentiate through Module parameters**

.. code-block:: python

    import riemann as rm
    from riemann import func

    layer = rm.nn.Linear(3, 2)
    x = rm.tensor([1., 2., 3.])
    alt_weight = rm.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], requires_grad=True)
    alt_bias = rm.tensor([0.01, 0.02], requires_grad=True)

    def param_loss(weight, bias):
        y = func.functional_call(layer, [weight, bias], x)
        return (y ** 2).sum()

    g_w, g_b = func.grad(param_loss, argnums=(0, 1))(alt_weight, alt_bias)
    print(g_w.shape, g_b.shape)

**Pattern 4: nested differentiation — higher-order derivatives**

.. code-block:: python

    import riemann as rm
    from riemann import func

    f = lambda x: (x ** 4).sum()
    x = rm.tensor([1., 2., 3.], requires_grad=True)

    # First-order
    g = func.grad(f)
    print(g(x))
    # Second-order: differentiate the gradient (results keep the graph by default)
    h = func.grad(lambda a: g(a).sum())
    print(h(x))
    # Third-order: keep stacking
    third = func.grad(lambda a: h(a).sum())(x)
    print(third)

func Module vs autograd.functional: When to Use Which?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Scenario
     - Recommendation
     - Reason
   * - Quick one-shot jacobian / hessian
     - ``autograd.functional``
     - Pass inputs directly, one less wrapper
   * - Compose with vmap, batch differentiation
     - ``func``
     - Returns Callable; nest freely
   * - has_aux or multi-input argnums
     - ``func``
     - Not supported by lower module
   * - Differentiate through Module params
     - ``func.functional_call``
     - Not available in lower module
   * - vjp for vector outputs
     - ``func.vjp``
     - Explicit cotangents; ``vjp_fn`` returns a tuple
   * - Higher-order derivatives (nesting)
     - ``func``
     - Results keep the graph; nest freely

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

2. **Using register_grad_rule Decorator**
   ``register_grad_rule`` is a **Riemann-specific** operator-gradient registration
   decorator with no PyTorch equivalent. Use it to wrap your function and
   provide explicit gradient computation.

   **Gradient Function Interface Requirements:**

   The gradient function passed to ``register_grad_rule`` has the fixed signature ``grad_func(result, *args, **kwargs)``:

   - **Parameters**: The first parameter ``result`` is the output tensor of the forward function; the remaining parameters must match the forward function (same names and order)
   - **Return Value**: Must return a ``tuple`` containing the gradient for each input tensor, ordered by the TN parameters in the ``grad_func`` signature
   - **Tuple Elements**: Only TN parameters occupy a position; non-TN parameters (scalars, booleans, etc.) do not.
     For TN tensors that don't require gradients, return ``None`` for that position
   - **Chain Rule**: Returned gradients must already include the upstream gradient from the chain rule, available via ``result.grad_value``.
     Other attributes of ``result`` (shape, dtype, device, etc.) can also be used
   - **Call Order Independent**: The decorator builds a "parameter name → argument value" mapping via ``inspect.signature``,
     so keyword argument order and mixing positional/keyword arguments do not affect gradient alignment

   **Scenario 1: Single-input function**

   .. code-block:: python

       import riemann as rm
       import numpy as np
       
       def sigmoid_derivative(result, x):
           """Gradient function for sigmoid: returns tuple with one element"""
           # result is the sigmoid output; reuse it to avoid recomputation
           return (result.grad_value * result * (1. - result),)
       
       @rm.register_grad_rule(sigmoid_derivative)
       def custom_sigmoid(x):
           """Custom sigmoid function with gradient support"""
           return rm.tensor(1. / (1. + np.exp(-x.array())))
       
       # Test custom sigmoid with gradient
       x = rm.tensor(0.0, requires_grad=True)
       y = custom_sigmoid(x)
       y.backward()
       print(f"Sigmoid output: {y}")  # Should be 0.5
       print(f"Sigmoid gradient: {x.grad}")  # Should be 0.25

   **Scenario 2: Multiple inputs (keyword order shuffled)**

   The ``grad_func`` return tuple follows the function signature order, independent of call order.

   .. code-block:: python

       import riemann as rm
       
       def divide_derivative(result, x, y):
           """Division gradients: d(x/y)/dx = 1/y, d(x/y)/dy = -x/y²"""
           return (
               result.grad_value / y,
               result.grad_value * (-x / y**2.),
           )
       
       @rm.register_grad_rule(divide_derivative)
       def custom_divide(x, y):
           return x / y
       
       x = rm.tensor(5.0, requires_grad=True)
       y = rm.tensor(2.0, requires_grad=True)
       # Keyword order is the reverse of the (x, y) signature; gradients still align
       z = custom_divide(y=y, x=x)
       z.backward()
       print(f"z = {z}")           # 2.5
       print(f"dz/dx = {x.grad}")  # 0.5
       print(f"dz/dy = {y.grad}")  # -1.25

   **Scenario 3: Mixed TN and non-TN parameters, with defaults**

   Non-TN parameters do not occupy gradient tuple positions; defaults are included in the
   signature mapping when not explicitly passed.

   .. code-block:: python

       import riemann as rm
       
       def scaled_pow_derivative(result, x, scale=1.0, exponent=2.0):
           """d(scale * x^n)/dx = scale * n * x^(n-1)"""
           local = exponent * x ** (exponent - 1.)
           return (result.grad_value * scale * local,)
       
       @rm.register_grad_rule(scaled_pow_derivative)
       def scaled_pow(x, scale=1.0, exponent=2.0):
           return scale * x ** exponent
       
       x = rm.tensor(2.0, requires_grad=True)
       y = scaled_pow(x, exponent=3.0)   # scale uses default 1.0
       y.backward()
       print(f"y = {y}")           # 8.0
       print(f"dy/dx = {x.grad}")  # 12.0 (3 * 2²)

   **Scenario 4: Variable-length arguments (*args and **kwargs)**

   When ``grad_func`` uses ``*args`` / ``**kwargs``, the returned gradient tuple is flattened
   in signature order.

   .. code-block:: python

       import riemann as rm
       
       # *args: every input gets the upstream gradient
       def sum_derivative(result, *args):
           return tuple(result.grad_value for _ in args)
       
       @rm.register_grad_rule(sum_derivative)
       def custom_sum(*xs):
           return rm.tensor(sum(x.array() for x in xs))
       
       a = rm.tensor(1.0, requires_grad=True)
       b = rm.tensor(2.0, requires_grad=True)
       s = custom_sum(a, b)
       s.backward()
       print(f"da = {a.grad}, db = {b.grad}")  # 1.0, 1.0
       
       # **kwargs: look up the weight by parameter name
       WEIGHTS = {"a": 2.0, "b": 3.0}
       
       def weighted_sum_derivative(result, **kwargs):
           return tuple(result.grad_value * WEIGHTS[name] for name in kwargs)
       
       @rm.register_grad_rule(weighted_sum_derivative)
       def custom_weighted_sum(**kwargs):
           return rm.tensor(sum(WEIGHTS[name] * v.array() for name, v in kwargs.items()))
       
       va = rm.tensor(1.0, requires_grad=True)
       vb = rm.tensor(1.0, requires_grad=True)
       w = custom_weighted_sum(a=va, b=vb)
       w.backward()
       print(f"dva = {va.grad}, dvb = {vb.grad}")  # 2.0, 3.0

   **Scenario 5: Some inputs do not require gradients**

   .. code-block:: python

       import riemann as rm
       
       def add_derivative(result, x, y):
           return (result.grad_value, result.grad_value)
       
       @rm.register_grad_rule(add_derivative)
       def custom_add(x, y):
           return x + y
       
       x = rm.tensor(2.0, requires_grad=True)
       y = rm.tensor(3.0, requires_grad=False)  # no gradient needed
       z = custom_add(x, y)
       z.backward()
       print(f"dz/dx = {x.grad}")  # 1.0
       print(f"dz/dy = {y.grad}")  # None

   **Scenario 6: Matrix multiplication (custom chain rule)**

   This is the key advantage of ``register_grad_rule``: backward computation may directly use matrix
   multiplication ``@``, transpose, and other operations instead of being limited to
   element-wise multiplication with the upstream gradient.

   .. code-block:: python

       import riemann as rm
       
       def matmul_derivative(result, x, y):
           """Matrix multiplication backward: dL/dX = dL/dY @ Yᵀ, dL/dY = Xᵀ @ dL/dY"""
           g = result.grad_value
           return (g @ y.T, x.T @ g)
       
       @rm.register_grad_rule(matmul_derivative)
       def custom_matmul(x, y):
           return rm.matmul(x, y)
       
       x = rm.tensor([[1., 2.], [3., 4.]], requires_grad=True)
       y = rm.tensor([[5., 6.], [7., 8.]], requires_grad=True)
       z = custom_matmul(x, y)
       z.sum().backward()
       print(f"dx = {x.grad}")  # [[11., 15.], [11., 15.]]
       print(f"dy = {y.grad}")  # [[4., 4.], [6., 6.]]

   **Notes:**

   - The gradient function must return the final gradient *including* the upstream gradient;
     omitting ``result.grad_value`` loses the upstream part of the chain rule
   - Tuple order is determined solely by the ``grad_func`` signature, never by call order;
     non-TN parameters do not occupy a position
   - Each ``backward`` call recomputes ``grad_func`` with the current upstream gradient,
     so repeated backward calls with ``retain_graph=True`` are supported
   - If the forward function is composed entirely of Riemann tensor operators (already differentiable),
     ``register_grad_rule`` is usually unnecessary; it is intended for wrapping external operators
     or manually defined forward/backward computations

3. **Using register_jvp_rule Decorator (Forward Rules)**
   When an operator cannot be expressed with existing TN operations, or an external
   operator needs forward-mode support, register a direct forward rule with
   ``@rm.register_jvp_rule(jvp_func)``. It is fully independent of
   ``register_grad_rule``: either decorator can be used alone, and both can decorate
   the same function in any order without overwriting each other.

   **jvp_func Interface:**

   - Call shape mirrors the original function with the output first:
     ``jvp_func(result, *args, **kwargs)``, where ``result`` is the output tensor
   - Each **TN argument** is passed as ``TangentPair(primal, tangent)`` (a
     ``tuple`` subclass, so it unpacks directly); ``tangent`` is ``None`` when that
     input direction is zero. Non-TN arguments (including tuple parameters such as
     ``dim=(0, 1)``) are passed through unchanged, so the rule can read their values
     directly
   - Return the output tangent; its shape must equal ``result``'s. Return ``None`` if
     the output does not depend on any differentiable input (treated as no tangent).
     The rule never touches the internal ``JVPFrame``
   - The rule does not need ``fromvars``: positional parameters, keyword parameters,
     ``*args``/``**kwargs`` and defaults are paired by the original signature, and
     inputs with ``requires_grad=False`` are still passed to the rule
   - Variadic rules can use ``isinstance(arg, TangentPair)`` to tell paired TN
     arguments apart from ordinary tuple arguments such as ``(0, 1)``

   Example:

   .. code-block:: python

       import riemann as rm

       def _my_tanh_jvp(result, x):
           x, x_t = x                   # TN argument = (primal, tangent)
           if x_t is None:              # None means zero direction
               return None
           # d(tanh(x))/dx = 1 - tanh(x)^2, reusing the forward output
           return (1.0 - result * result) * x_t

       @rm.register_jvp_rule(_my_tanh_jvp)
       def my_tanh(x):
           return rm.tanh(x)

       x = rm.tensor([1., 2.], requires_grad=True)
       out, tan = rm.func.jvp(my_tanh, x, rm.tensor([0.5, -0.5]))
       print(tan)  # (1 - tanh(x)^2) * v

   **Notes:**

   - A rule only defines how this node's tangent follows from incoming tangents; the
     framework performs the chain rule in topological order
   - Forward and backward rules can be implemented independently; with only
     ``register_jvp_rule``, gradients still come from TN operations inside the function
   - As with ``register_grad_rule``, the decorated function should create a new node
     rather than returning an input unchanged

4. **Using Function Class**
   For more complex cases, you can subclass ``Function`` and implement ``forward`` and
   ``backward`` static methods; implement the optional ``jvp`` static method for custom
   forward-mode behavior.

   **Function Class Interface:**

   To create a custom function using the ``Function`` class, implement at least
   ``forward`` and ``backward``:

   **forward(ctx, *inputs)**

   - **Purpose**: Performs the forward computation
   - **Parameters**:

     - ``ctx``: Context object used to save information for the backward pass. Use ``ctx.save_for_backward()`` to store tensors needed in backward
     - ``*inputs``: Input tensors (variable number of arguments)
   - **Returns**: Output tensor(s) of the forward computation
   - **Usage**: Implement your custom computation logic here and save any tensors needed for gradient computation using ``ctx.save_for_backward()``

   **backward(ctx, grad_output)**

   - **Purpose**: Performs the backward (gradient) computation
   - **Parameters**:

     - ``ctx``: Context object containing information saved during forward pass. Access saved tensors via ``ctx.saved_tensors``
     - ``grad_output``: Gradient of the output tensor (from subsequent layers in the computation graph)
   - **Returns**: Tuple of gradients, one for each input tensor. Each gradient should be the product of ``grad_output`` and the local gradient (partial derivative)
   - **Usage**: Compute gradients using the chain rule: ``grad_input = grad_output * local_gradient``

   **jvp(ctx, *input_tangents) (optional, forward mode)**

   - **Purpose**: Register a forward rule for the custom function; once implemented,
     ``func.jvp`` calls it directly
   - **Parameters**:

     - ``ctx``: the same context object used by ``backward``
     - ``input_tangents``: one entry per ``forward`` input; pass ``None`` for
       non-tensor inputs
   - **Returns**: the output tangent; a tensor or ``None`` (non-differentiable output)
     for a single output — ``apply`` currently supports only single-output functions
   - **Saved tensors**: inside ``jvp``, ``ctx.saved_tensors`` contains only tensors
     saved with ``save_for_forward()``; tensors saved with ``save_for_backward()``
     remain visible only in ``backward``
   - **When not implemented**: ``func.jvp`` raises when it reaches the node; forward
     mode requires an explicit ``jvp`` static method

   **setup_context(ctx, inputs, output) (optional, new style)**

   Besides ``forward(ctx, *inputs)``, the PyTorch 1.13+ style is supported:
   ``forward(*inputs)`` performs only the computation and
   ``setup_context(ctx, inputs, output)`` calls ``save_for_backward``/
   ``save_for_forward``. ``apply`` detects and uses the corresponding style; do not
   mix the two.

   **Other context members:**

   - ``ctx.needs_input_grad``: a per-input boolean tuple written by ``apply`` before
     any callback; non-tensor positions are ``False``, and all entries are ``False``
     under ``set_grad_enabled(False)``
   - ``ctx.save_for_forward(*tensors)``: only TN or ``None`` are accepted; store
     non-tensors as ordinary ``ctx`` attributes

   **Example:**

   .. code-block:: python

       import riemann as rm
       import numpy as np
       
       class CustomSigmoid(rm.autograd.Function):
           @staticmethod
           def forward(ctx, x):
               """Forward computation for sigmoid
               
               Args:
                   ctx: Context object for saving tensors
                   x: Input tensor
               
               Returns:
                   Output tensor after applying sigmoid
               """
               sig = 1. / (1. + np.exp(-x.detach().array()))
               ctx.save_for_backward(rm.tensor(sig))  # Save for backward
               return rm.tensor(sig)
           
           @staticmethod
           def backward(ctx, grad_output):
               """Backward computation for sigmoid
               
               Args:
                   ctx: Context object with saved tensors
                   grad_output: Gradient from output side
               
               Returns:
                   Gradient with respect to input
               """
               sig, = ctx.saved_tensors  # Retrieve saved tensor
               # Chain rule: grad_input = grad_output * local_gradient
               # local_gradient for sigmoid: sig * (1 - sig)
               return grad_output * sig * (1. - sig)
       
       # Test CustomSigmoid
       x = rm.tensor(0.0, requires_grad=True)
       y = CustomSigmoid.apply(x)  # Use apply() to call the function
       y.backward()
       print(f"Sigmoid output: {y}")  # Should be 0.5
       print(f"Sigmoid gradient: {x.grad}")  # Should be 0.25

   **Key Points:**

   - Always use ``@staticmethod`` decorator for both ``forward`` and ``backward`` methods
   - Use ``ctx.save_for_backward()`` in ``forward`` to save tensors needed for gradient computation
   - Access saved tensors in ``backward`` via ``ctx.saved_tensors`` (returns a tuple)
   - The ``backward`` method must return a tuple with one gradient for each input to ``forward``
   - Call the function using ``ClassName.apply(*inputs)``, not by instantiating the class

Manually Modifying the Computational Graph
------------------------------------------

Riemann provides functions for manually manipulating the computational graph. These functions are designed for special use cases where you need to connect tensors to the computational graph without affecting forward computation values or backward gradient values. These are low-level tools typically used in framework internals (such as Riemann's hook handling mechanism) rather than common user scenarios.

map_value_connect_grad Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``map_value_connect_grad`` function fully connects and maps a group of tensors to another group of tensors with the same count. The corresponding positions between the two groups have an identity mapping: forward pass transparently transmits tensor values, backward pass transparently transmits gradient values (equivalent to a clone relationship). For other connections, forward pass transmits zero values (i.e., does not affect the new tensor values), backward pass transmits zero gradients.

**Purpose:**
Ensure all tensors in a group participate in the computational graph and receive gradients (zero for tensors not directly involved in computation) rather than None. This is particularly useful when you want certain tensors to receive zero gradients without changing the existing computational graph's forward or backward computation values.

**Core Mechanism:**

1. For each tensor that requires gradients, create a clone. The cloned tensor depends on the original tensor through the ``clone`` operation (gradient passes through)
2. Attach all other tensors (excluding itself) as zero-gradient sources to the cloned tensor
3. This way, each tensor maintains its gradient relationship with the original tensor while forming zero-gradient connections with other tensors

**Parameters:**

- ``tensors``: A tuple or list of tensors to connect. Must be tuple or list (not set) to preserve order.

**Returns:**
A tuple or list of tensors with the same values but connected to a shared computational graph. Note: tensors with ``requires_grad=True`` are cloned (not modified in place).

**Behavior:**

- Tensors with ``requires_grad=True`` are cloned, and all other tensors are attached as zero-gradient sources to the cloned tensor
- Tensors without gradients or non-TN objects remain unchanged
- All connected tensors receive zero gradients from each other

**Example:**

.. code-block:: python

    import riemann as rm
    
    a = rm.tensor([1.0, 2.0], requires_grad=True)
    b = rm.tensor([3.0, 4.0], requires_grad=True)
    c = rm.tensor([5.0, 6.0], requires_grad=True)
    
    # Define a function that only uses a and b
    def func(a, b, c):
        return (a * b).sum()
    
    # Before map_value_connect_grad: c doesn't participate, receives None
    y1 = func(a, b, c)
    y1.backward()
    print(f"c.grad = {c.grad}")  # Output: None
    
    # Reset tensors
    a = rm.tensor([1.0, 2.0], requires_grad=True)
    b = rm.tensor([3.0, 4.0], requires_grad=True)
    c = rm.tensor([5.0, 6.0], requires_grad=True)
    
    # After map_value_connect_grad: all tensors connected, c receives zero gradient
    a_new, b_new, c_new = rm.map_value_connect_grad((a, b, c))
    y2 = func(a_new, b_new, c_new)
    y2.backward()
    # Note: gradients flow back to the ORIGINAL tensors a/b/c;
    # cloned tensors .grad is None
    print(f"c.grad (origin): {c.grad}")  # Output: [0.0, 0.0]
    
    # Verify: forward values are identical, a and b gradients unchanged
    assert y1.item() == y2.item()
    assert (a.grad == rm.tensor([3., 4.])).all()
    assert (b.grad == rm.tensor([1., 2.])).all()
    assert (c.grad == rm.tensor([0., 0.])).all()

**Typical Use Cases:**

1. **Module Hook Handling**: In Riemann's module hook mechanism, ``map_value_connect_grad`` is used to create new module output tensors to replace the original output tensors. When only some tensors of a module participate in or contribute to the loss function computation, ``map_value_connect_grad`` produces a new module output without changing forward computation or backward gradient values. This ensures that output tensors that previously didn't participate in loss computation can now receive zero gradients, and input tensors depending on these outputs will also receive zero gradients.

2. **Multi-task Learning**: When some parameters don't participate in certain task's loss computation but you want them to receive zero gradients rather than None for gradient accumulation.

3. **Conditional Computation**: When some tensors are conditionally used in forward pass but you want consistent gradient behavior regardless of the condition.

4. **Gradient Monitoring**: When you want to monitor gradients of all parameters in a group, even those not directly involved in a specific computation.

**Note:** This is a low-level function for manually building computational graphs. Most users should rely on Riemann's automatic graph construction rather than using this function directly.

Supporting Functions and Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following are low-level computational graph helpers that user code rarely needs to call directly:

**attach_zero_grad_sources Function**

Returns a new tensor with the same value that passes gradients through to the input tensor and sends zero gradients to the given sources during backward. It does not modify the original tensor's computational graph, and sources not involved in the loss receive zero gradients instead of None.

Gradient Checking
-----------------

You can manually verify custom gradient functions using the finite difference method:

.. code-block:: python

    import riemann as rm
    import numpy as np

    # Manual numerical gradient check (same principle as PyTorch's gradcheck)
    def numerical_grad(func, x, eps=1e-5):
        x_flat = x._array.astype(np.float64).copy()
        grad_flat = np.zeros_like(x_flat)
        it = np.nditer(x_flat, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            old_val = x_flat[idx]
            x_flat[idx] = old_val + eps
            f_plus = func(rm.tensor(x_flat.astype(np.float32))).item()
            x_flat[idx] = old_val - eps
            f_minus = func(rm.tensor(x_flat.astype(np.float32))).item()
            x_flat[idx] = old_val
            grad_flat[idx] = (f_plus - f_minus) / (2 * eps)
            it.iternext()
        return rm.tensor(grad_flat.astype(np.float32))

    # Test with a simple function
    def f(x):
        return (x ** 3).sum()

    x = rm.tensor([1., 2., 3.], requires_grad=True)
    y = f(x)
    y.backward()

    num_grad = numerical_grad(f, x)
    print(f"Analytical grad: {x.grad}")
    print(f"Numerical grad:   {num_grad}")
    print(f"Max error:        {(x.grad - num_grad).abs().max()}")
    # Max error should be < 1e-4

Gradcheck verifies that your analytical gradient computation matches the numerical gradient computed using finite differences. Riemann does not yet provide a built-in ``gradcheck`` function; use the numerical gradient check above instead.

Gradient Computation Tips
-------------------------

1. **Memory Management**: Gradient computation uses memory to store the computational graph. Use ``no_grad()`` or ``detach()`` when you don't need gradients to save memory.

Common Pitfalls
------------------

1. **In-place Operations**: Avoid performing in-place operations on leaf node tensors that require gradient tracking.

2. **Detaching Tensors from Computational Graph**: After detaching, tensors lose their computational graph dependencies and cannot perform backward propagation for gradient calculation.

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