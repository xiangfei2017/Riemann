Einstein Summation Convention (einsum)
======================================

The Einstein summation convention is a concise notation for tensor operations in mathematics and physics, introduced by Albert Einstein in 1916. In deep learning frameworks, the ``einsum`` function leverages this convention to provide a unified, elegant, and powerful way to express various tensor operations.

What is the Einstein Summation Convention
------------------------------------------

**Mathematical Background**

In traditional tensor algebra, complex tensor operations typically involve numerous summation symbols and index notations. The Einstein summation convention simplifies expressions by omitting summation symbols. The core rule is: **when the same index appears twice in a term, it indicates summation over that index**.

For example, traditional matrix multiplication can be written as:

.. math::

    C_{ik} = \sum_{j} A_{ij} B_{jk}

Using the Einstein summation convention, this simplifies to:

.. math::

    C_{ik} = A_{ij} B_{jk}

Here, the index ``j`` appears twice on the right-hand side (in both ``A`` and ``B``), indicating summation over ``j``.

**Core Concepts of einsum**

The ``einsum`` function in Riemann implements this convention, allowing users to describe complex tensor operations through concise string equations. Its advantages include:

1. **Universality**: One function can replace many different tensor operations
2. **Readability**: The equation string intuitively expresses the mathematical meaning of the operation
3. **Flexibility**: Supports tensors of arbitrary dimensions and complex index manipulations
4. **Efficiency**: Internal optimizations ensure high-performance computation

einsum Equation String Syntax
------------------------------

**Basic Syntax Structure**

The einsum equation string follows this format:

.. code-block:: python

    "input1_indices,input2_indices,...->output_indices"

Where:

- **Input indices**: Letters representing the dimensions of corresponding input tensors, e.g., ``ij`` for a 2D tensor (matrix)
- **Output indices**: Specify the dimensions of the output tensor; if omitted, indices are arranged in alphabetical order
- **Ellipsis ``...``**: Represents any number of batch dimensions
- **Repeated indices**: Indicate summation (contraction) over that dimension

**Detailed Index Rules**

1. **Unique indices**: If an index appears in only one input and also in the output, that dimension is preserved
2. **Repeated indices**: If an index appears in multiple inputs, it indicates element-wise multiplication followed by summation over those dimensions
3. **Missing indices**: If an index from the input does not appear in the output, it indicates summation reduction over that dimension

**Example Analysis**

.. code-block:: python

    # Matrix multiplication: ij,jk->ik
    # i: row index of A, j: column index of A/row index of B (summed), k: column index of B
    # Result C has dimensions (i, k)
    C = rm.einsum('ij,jk->ik', A, B)

    # Batch matrix multiplication: ...ij,...jk->...ik
    # ... represents any batch dimensions
    C = rm.einsum('...ij,...jk->...ik', A, B)

    # Trace operation: ii->
    # i appears twice, indicating summation of diagonal elements
    trace = rm.einsum('ii->', A)

    # Diagonal extraction: ii->i
    # Preserves diagonal elements, result is a vector
    diag = rm.einsum('ii->i', A)

Classification of einsum Computation Scenarios
-----------------------------------------------

The following tables detail the various computation scenarios that einsum can replace:

**Basic Matrix Operations**

.. list-table:: Basic Matrix Operations
    :widths: 20 25 35 20
    :header-rows: 1

    * - Operation Type
      - Mathematical Description
      - einsum Equation
      - Equivalent Function
    * - Matrix Multiplication
      - :math:`C_{ik} = \sum_j A_{ij} B_{jk}`
      - ``ij,jk->ik``
      - ``rm.matmul(A, B)``
    * - Batch Matrix Multiplication
      - :math:`C_{bik} = \sum_j A_{bij} B_{bjk}`
      - ``bij,bjk->bik``
      - ``rm.matmul(A, B)``
    * - General Batch Matrix Multiplication
      - Supports arbitrary batch dimensions
      - ``...ij,...jk->...ik``
      - ``rm.matmul(A, B)``
    * - Vector Dot Product
      - :math:`c = \sum_i a_i b_i`
      - ``i,i->``
      - ``rm.dot(a, b)``
    * - Vector Outer Product
      - :math:`C_{ij} = a_i b_j`
      - ``i,j->ij``
      - ``rm.outer(a, b)``

**Matrix Property Extraction**

.. list-table:: Matrix Property Extraction
    :widths: 20 25 35 20
    :header-rows: 1

    * - Operation Type
      - Mathematical Description
      - einsum Equation
      - Equivalent Function
    * - Matrix Trace
      - :math:`\text{tr}(A) = \sum_i A_{ii}`
      - ``ii->``
      - ``rm.trace(A)``
    * - Diagonal Extraction
      - :math:`\text{diag}(A)_i = A_{ii}`
      - ``ii->i``
      - ``rm.diag(A)``
    * - Batch Matrix Trace
      - :math:`\text{tr}(A_b) = \sum_i A_{bii}`
      - ``bii->b``
      - ``rm.trace(A)``
    * - Batch Diagonal Extraction
      - :math:`\text{diag}(A_b)_i = A_{bii}`
      - ``bii->bi``
      - ``rm.diag(A)``

**Transpose and Dimension Permutation**

.. list-table:: Transpose and Dimension Permutation
    :widths: 20 25 35 20
    :header-rows: 1

    * - Operation Type
      - Mathematical Description
      - einsum Equation
      - Equivalent Function
    * - Matrix Transpose
      - :math:`C_{ji} = A_{ij}`
      - ``ij->ji``
      - ``A.T`` or ``rm.transpose(A)``
    * - High-Dimensional Transpose
      - :math:`C_{jki} = A_{ijk}`
      - ``ijk->jki``
      - ``rm.permute(A, (1, 2, 0))``
    * - Batch Transpose
      - :math:`C_{bji} = A_{bij}`
      - ``...ij->...ji``
      - ``A.mT``

**Tensor Contraction and Summation**

.. list-table:: Tensor Contraction and Summation
    :widths: 20 25 35 20
    :header-rows: 1

    * - Operation Type
      - Mathematical Description
      - einsum Equation
      - Equivalent Function
    * - Sum All Elements
      - :math:`s = \sum_{i,j} A_{ij}`
      - ``ij->``
      - ``rm.sum(A)``
    * - Sum Along Rows
      - :math:`s_i = \sum_j A_{ij}`
      - ``ij->i``
      - ``rm.sum(A, dim=1)``
    * - Sum Along Columns
      - :math:`s_j = \sum_i A_{ij}`
      - ``ij->j``
      - ``rm.sum(A, dim=0)``
    * - Tensor Contraction
      - :math:`C_{ijm} = \sum_{k,l} A_{ijkl} B_{jklm}`
      - ``ijkl,jklm->ijm``
      - No direct equivalent
    * - Self-Contraction
      - :math:`C_i = \sum_j A_{ij} B_{ij}`
      - ``ij,ij->i``
      - ``(A * B).sum(dim=1)``

**Special Matrix Operations**

.. list-table:: Special Matrix Operations
    :widths: 20 25 35 20
    :header-rows: 1

    * - Operation Type
      - Mathematical Description
      - einsum Equation
      - Equivalent Function
    * - Hadamard Product
      - :math:`C_{ij} = A_{ij} B_{ij}`
      - ``ij,ij->ij``
      - ``A * B``
    * - Frobenius Inner Product
      - :math:`\langle A, B \rangle_F = \sum_{i,j} A_{ij} B_{ij}`
      - ``ij,ij->``
      - ``(A * B).sum()``
    * - Kronecker Product
      - :math:`C_{ikjl} = A_{ij} B_{kl}`
      - ``ij,kl->ikjl``
      - ``rm.kron(A, B)``
    * - Identity Copy
      - :math:`C_{ij} = A_{ij}`
      - ``ij->ij``
      - ``A.clone()``

**Multi-Operand Operations**

.. list-table:: Multi-Operand Operations
    :widths: 20 25 35 20
    :header-rows: 1

    * - Operation Type
      - Mathematical Description
      - einsum Equation
      - Description
    * - Three-Operand Chain
      - :math:`C_{il} = \sum_{j,k} A_{ij} B_{jk} C_{kl}`
      - ``ij,jk,kl->il``
      - Sequential matrix multiplication
    * - Four-Operand Chain
      - :math:`C_{im} = \sum_{j,k,l} A_{ij} B_{jk} C_{kl} D_{lm}`
      - ``ij,jk,kl,lm->im``
      - Long chain matrix multiplication
    * - Multi-Operand Mixed
      - :math:`C_i = \sum_{j,k} A_{ij} B_{jk} C_{ik}`
      - ``ij,jk,ik->i``
      - Complex mixed operation
    * - Batch Three-Operand
      - Supports arbitrary batch dimensions
      - ``...ij,...jk,...kl->...il``
      - Batch chain multiplication

**Repeated Index Operations**

.. list-table:: Repeated Index Operations
    :widths: 20 25 35 20
    :header-rows: 1

    * - Operation Type
      - Mathematical Description
      - einsum Equation
      - Description
    * - First Two Indices Repeated
      - :math:`C_j = \sum_{i} A_{iij}`
      - ``iij->j``
      - Extract specific diagonal
    * - Multiple Indices Repeated
      - :math:`C_{ij} = \sum_{k,l} A_{iijj}`
      - ``iijj->ij``
      - High-dimensional diagonal extraction
    * - Non-Contiguous Index Repeat
      - :math:`s = \sum_{i,j} A_{ijji}`
      - ``ijji->``
      - Anti-diagonal summation

**1D Vector Operations**

.. list-table:: 1D Vector Operations
    :widths: 20 25 35 20
    :header-rows: 1

    * - Operation Type
      - Mathematical Description
      - einsum Equation
      - Equivalent Function
    * - Vector Dot Product
      - :math:`c = \sum_i a_i b_i`
      - ``i,i->``
      - ``rm.dot(a, b)``
    * - Vector Outer Product
      - :math:`C_{ij} = a_i b_j`
      - ``i,j->ij``
      - ``rm.outer(a, b)``
    * - Matrix-Vector Multiplication
      - :math:`c_i = \sum_j A_{ij} b_j`
      - ``ij,j->i``
      - ``rm.matmul(A, b)``
    * - Vector-Matrix Multiplication
      - :math:`c_j = \sum_i a_i A_{ij}`
      - ``i,ij->j``
      - ``rm.matmul(a, A)``
    * - Batch Matrix-Vector Multiplication
      - :math:`C_{bi} = \sum_j A_{bij} b_{bj}`
      - ``bij,bj->bi``
      - ``rm.matmul(A, b.unsqueeze(-1)).squeeze(-1)``

einsum Usage Examples
----------------------

**Example 1: Matrix Multiplication**

.. code-block:: python

    import riemann as rm

    # Create matrices
    A = rm.tensor([[1, 2], [3, 4], [5, 6]])  # 3x2
    B = rm.tensor([[7, 8, 9], [10, 11, 12]])  # 2x3

    # Matrix multiplication: (3x2) @ (2x3) = (3x3)
    C = rm.einsum('ij,jk->ik', A, B)
    print("Matrix multiplication result:")
    print(C)
    # Output:
    # tensor([[ 27,  30,  33],
    #         [ 61,  68,  75],
    #         [ 95, 106, 117]])

**Example 2: Batch Matrix Multiplication**

.. code-block:: python

    import riemann as rm

    # Create batch matrices (2 matrices of 3x4)
    A = rm.randn(2, 3, 4)
    # Create batch matrices (2 matrices of 4x5)
    B = rm.randn(2, 4, 5)

    # Batch matrix multiplication
    C = rm.einsum('bij,bjk->bik', A, B)
    print(f"Batch matrix multiplication result shape: {C.shape}")  # (2, 3, 5)

    # Use ellipsis to support more batch dimensions
    A = rm.randn(2, 3, 4, 5)  # 2x3 matrices of 4x5
    B = rm.randn(2, 3, 5, 6)  # 2x3 matrices of 5x6
    C = rm.einsum('...ij,...jk->...ik', A, B)
    print(f"General batch multiplication result shape: {C.shape}")  # (2, 3, 4, 6)

**Example 3: Trace and Diagonal Operations**

.. code-block:: python

    import riemann as rm

    # Create square matrix
    A = rm.tensor([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

    # Compute trace (sum of diagonal elements)
    trace = rm.einsum('ii->', A)
    print(f"Matrix trace: {trace}")  # 1 + 5 + 9 = 15

    # Extract diagonal elements
    diag = rm.einsum('ii->i', A)
    print(f"Diagonal elements: {diag}")  # [1, 5, 9]

    # Batch matrix trace
    batch_A = rm.randn(4, 3, 3)  # 4 matrices of 3x3
    batch_trace = rm.einsum('bii->b', batch_A)
    print(f"Batch trace shape: {batch_trace.shape}")  # (4,)

**Example 4: Transpose and Dimension Permutation**

.. code-block:: python

    import riemann as rm

    # Create matrix
    A = rm.tensor([[1, 2, 3],
                   [4, 5, 6]])

    # Matrix transpose
    A_T = rm.einsum('ij->ji', A)
    print("Transpose result:")
    print(A_T)
    # Output:
    # tensor([[1, 4],
    #         [2, 5],
    #         [3, 6]])

    # High-dimensional transpose
    B = rm.randn(2, 3, 4)
    B_perm = rm.einsum('ijk->jki', B)
    print(f"High-dimensional transpose shape: {B_perm.shape}")  # (3, 4, 2)

**Example 5: Vector Operations**

.. code-block:: python

    import riemann as rm

    # Create vectors
    a = rm.tensor([1, 2, 3])
    b = rm.tensor([4, 5, 6])

    # Vector dot product
    dot_product = rm.einsum('i,i->', a, b)
    print(f"Dot product: {dot_product}")  # 1*4 + 2*5 + 3*6 = 32

    # Vector outer product
    outer_product = rm.einsum('i,j->ij', a, b)
    print("Outer product result:")
    print(outer_product)
    # Output:
    # tensor([[ 4,  5,  6],
    #         [ 8, 10, 12],
    #         [12, 15, 18]])

    # Matrix-vector multiplication
    A = rm.tensor([[1, 2, 3],
                   [4, 5, 6]])
    c = rm.einsum('ij,j->i', A, a)
    print(f"Matrix-vector multiplication: {c}")  # [14, 32]

**Example 6: Tensor Contraction**

.. code-block:: python

    import riemann as rm

    # Create 3D tensors
    A = rm.randn(2, 3, 4)
    B = rm.randn(3, 4, 5)

    # Tensor contraction: sum over dimensions 1 and 2
    C = rm.einsum('ijk,jkl->il', A, B)
    print(f"Tensor contraction result shape: {C.shape}")  # (2, 5)

    # More complex contraction
    D = rm.randn(2, 3, 4, 5)
    E = rm.randn(3, 4, 5, 6)
    F = rm.einsum('ijkl,jklm->im', D, E)
    print(f"Complex contraction result shape: {F.shape}")  # (2, 6)

**Example 7: Hadamard Product and Frobenius Inner Product**

.. code-block:: python

    import riemann as rm

    # Create matrices
    A = rm.tensor([[1, 2],
                   [3, 4]])
    B = rm.tensor([[5, 6],
                   [7, 8]])

    # Hadamard product (element-wise multiplication)
    hadamard = rm.einsum('ij,ij->ij', A, B)
    print("Hadamard product:")
    print(hadamard)
    # Output:
    # tensor([[ 5, 12],
    #         [21, 32]])

    # Frobenius inner product
    frobenius = rm.einsum('ij,ij->', A, B)
    print(f"Frobenius inner product: {frobenius}")  # 5 + 12 + 21 + 32 = 70

**Example 8: Multi-Operand Operations**

.. code-block:: python

    import riemann as rm

    # Create multiple matrices
    A = rm.randn(3, 4)
    B = rm.randn(4, 5)
    C = rm.randn(5, 6)
    D = rm.randn(6, 7)

    # Four-operand chain multiplication
    result = rm.einsum('ij,jk,kl,lm->im', A, B, C, D)
    print(f"Four-operand chain result shape: {result.shape}")  # (3, 7)

    # Equivalent to:
    # temp1 = rm.matmul(A, B)
    # temp2 = rm.matmul(temp1, C)
    # result = rm.matmul(temp2, D)

**Example 9: einsum with Gradient Tracking**

.. code-block:: python

    import riemann as rm

    # Create tensors requiring gradients
    A = rm.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    B = rm.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    # Perform einsum operation
    C = rm.einsum('ij,jk->ik', A, B)

    # Compute loss and backpropagate
    loss = C.sum()
    loss.backward()

    print("Gradient of A:")
    print(A.grad)
    print("Gradient of B:")
    print(B.grad)

**Example 10: Implicit Output (Omitting Output Indices)**

.. code-block:: python

    import riemann as rm

    A = rm.tensor([[1, 2], [3, 4]])
    B = rm.tensor([[5, 6], [7, 8]])

    # Implicit output: omit the part after ->
    # Result indices are arranged in alphabetical order
    C = rm.einsum('ij,jk', A, B)  # Equivalent to 'ij,jk->ik'
    print("Implicit output result:")
    print(C)

    # Batch implicit output
    A = rm.randn(2, 3, 4)
    B = rm.randn(2, 4, 5)
    C = rm.einsum('...ij,...jk', A, B)  # Equivalent to '...ij,...jk->...ik'
    print(f"Batch implicit output shape: {C.shape}")  # (2, 3, 5)

einsum Performance Optimization Tips
-------------------------------------

1. **Prefer simple equations**: For common matrix multiplications, using ``rm.matmul`` directly may be more efficient
2. **Avoid unnecessary copies**: einsum returns views rather than copies whenever possible
3. **Batch operations over loops**: Use ``...`` to represent batch dimensions instead of explicit loops
4. **Chain operation fusion**: Multiple matrix multiplications can be combined into one einsum call, reducing intermediate results
5. **Pre-compiled equations**: For repeatedly used equations, einsum automatically caches optimizations

Notes
-----

1. **Index letter limitation**: Indices use lowercase letters (a-z), supporting up to 26 different indices
2. **Dimension matching**: Dimensions of repeated indices must be consistent
3. **Device consistency**: All input tensors must be on the same device
4. **Data types**: einsum follows standard type promotion rules
5. **Gradient tracking**: Supports automatic differentiation and normal gradient computation
