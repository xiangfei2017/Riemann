Linear Algebra
==============

Riemann provides comprehensive linear algebra operations through the ``riemann.linalg`` module. These operations are essential for many machine learning and scientific computing applications.

Matrix Operations
-----------------

Matrix Multiplication
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # Matrix multiplication
    A = rm.randn(3, 4)
    B = rm.randn(4, 5)
    C = linalg.matmul(A, B)  # or simply use A @ B
    print(C.shape)  # (3, 5)
    
    # Batch matrix multiplication
    A_batch = rm.randn(10, 3, 4)
    B_batch = rm.randn(10, 4, 5)
    C_batch = linalg.matmul(A_batch, B_batch)
    print(C_batch.shape)  # (10, 3, 5)

Matrix Transpose
~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    
    # Transpose matrix
    A = rm.randn(3, 4)
    A_T = A.T  # or A.transpose()
    print(A_T.shape)  # (4, 3)
    
    # Transpose batch matrix
    A_batch = rm.randn(10, 3, 4)
    A_batch_T = A_batch.transpose(1, 2)  # Swap dimensions 1 and 2
    print(A_batch_T.shape)  # (10, 4, 3)

Matrix Inverse
~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # Matrix inverse
    A = rm.randn(3, 3)
    A_inv = linalg.inv(A)
    print(A_inv.shape)  # (3, 3)
    
    # Verify A @ A_inv ≈ I
    identity = linalg.matmul(A, A_inv)
    print(identity)  # Should be close to identity matrix
    
    # Pseudoinverse of non-square matrix
    A = rm.randn(3, 4)
    A_pinv = linalg.pinv(A)
    print(A_pinv.shape)  # (4, 3)

Matrix Determinant
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # Matrix determinant
    A = rm.randn(3, 3)
    det = linalg.det(A)
    print(det)  # Scalar value
    
    # Batch determinants
    A_batch = rm.randn(10, 3, 3)
    det_batch = linalg.det(A_batch)
    print(det_batch.shape)  # (10,)

Decompositions
--------------

Singular Value Decomposition (SVD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function**: ``linalg.svd(A, full_matrices=True)``

**Description**: Computes the singular value decomposition of a matrix.

**Parameters**:
- ``A``: Input tensor (matrix or batch of matrices)
- ``full_matrices``: If True, returns full-sized U and Vh matrices. If False, returns reduced-sized matrices.

**Returns**:
- ``U``: Unitary matrix(ces)
- ``S``: Singular values as a 1D tensor
- ``Vh``: Unitary matrix(ces) (conjugate transpose of V)

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # SVD of matrix
    A = rm.randn(4, 3)
    U, S, Vh = linalg.svd(A)
    print(U.shape, S.shape, Vh.shape)  # (4, 4), (3,), (3, 3)
    
    # Reconstruct matrix
    A_reconstructed = U @ rm.diag(S) @ Vh
    print(rm.allclose(A, A_reconstructed))  # True
    
    # Reduced SVD
    U, S, Vh = linalg.svd(A, full_matrices=False)
    print(U.shape, S.shape, Vh.shape)  # (4, 3), (3,), (3, 3)

Eigen Value Decomposition(EVD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function**: ``linalg.eig(A)``

**Description**: Computes the eigenvalues and eigenvectors of a square matrix.

**Parameters**:
- ``A``: Input tensor (square matrix or batch of square matrices)

**Returns**:
- ``eigenvalues``: Eigenvalues of the matrix as a 1D tensor
- ``eigenvectors``: Eigenvectors of the matrix, where each column is an eigenvector

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # Eigendecomposition of symmetric matrix
    A = rm.randn(3, 3)
    A = A + A.T  # Make it symmetric
    eigenvalues, eigenvectors = linalg.eig(A)
    print(eigenvalues.shape, eigenvectors.shape)  # (3,), (3, 3)
    
    # Reconstruct matrix
    A_reconstructed = eigenvectors @ rm.diag(eigenvalues) @ eigenvectors.T
    print(rm.allclose(A, A_reconstructed))  # True

QR Decomposition
~~~~~~~~~~~~~~~~

**Function**: ``linalg.qr(A)``

**Description**: Computes the QR decomposition of a matrix, decomposing it into an orthogonal matrix Q and an upper triangular matrix R.

**Parameters**:
- ``A``: Input tensor (matrix or batch of matrices)

**Returns**:
- ``Q``: Orthogonal matrix(ces)
- ``R``: Upper triangular matrix(ces)

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # QR decomposition
    A = rm.randn(4, 3)
    Q, R = linalg.qr(A)
    print(Q.shape, R.shape)  # (4, 3), (3, 3)
    
    # Reconstruct matrix
    A_reconstructed = Q @ R
    print(rm.allclose(A, A_reconstructed))  # True

LU Decomposition
~~~~~~~~~~~~~~~~

**Function**: ``linalg.lu(A)``

**Description**: Computes the LU decomposition of a matrix with partial pivoting, decomposing it into a permutation matrix P, a lower triangular matrix L, and an upper triangular matrix U such that A = P @ L @ U.

**Parameters**:
- ``A``: Input tensor (matrix or batch of matrices)

**Returns**:
- ``P``: Permutation matrix(ces)
- ``L``: Lower triangular matrix(ces)
- ``U``: Upper triangular matrix(ces)

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # LU decomposition with pivoting (A = PLU)
    A = rm.randn(4, 4)
    P, L, U = linalg.lu(A)
    print(P.shape, L.shape, U.shape)  # (4, 4), (4, 4), (4, 4)
    
    # Reconstruct matrix
    A_reconstructed = P @ L @ U
    print(rm.allclose(A, A_reconstructed))  # True
    
    # LU decomposition for rectangular matrices
    A_rect = rm.randn(3, 5)
    P, L, U = linalg.lu(A_rect)
    print(P.shape, L.shape, U.shape)  # (3, 3), (3, 3), (3, 5)
    
    # Note: pivot=False is not yet implemented
    # P, L, U = linalg.lu(A, pivot=False)  # Throws NotImplementedError

Cholesky Decomposition
~~~~~~~~~~~~~~~~~~~~~~

**Function**: ``linalg.cholesky(A, upper=False)``

**Description**: Computes the Cholesky decomposition of a symmetric positive-definite matrix, decomposing it into a lower triangular matrix L such that A = L @ L.T (or upper triangular matrix U such that A = U.T @ U if upper=True).

**Parameters**:
- ``A``: Input tensor (symmetric positive-definite matrix or batch of such matrices)
- ``upper``: If True, returns an upper triangular matrix. If False (default), returns a lower triangular matrix.

**Returns**:
- ``L``: Lower or upper triangular matrix(ces) such that A = L @ L.T (or U.T @ U)

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # Cholesky decomposition of symmetric positive-definite matrix
    A = rm.randn(3, 3)
    A = A @ A.T  # Make it symmetric positive-definite
    L = linalg.cholesky(A)
    print(L.shape)  # (3, 3)
    
    # Reconstruct matrix
    A_reconstructed = L @ L.T
    print(rm.allclose(A, A_reconstructed))  # True
    
    # Upper triangular Cholesky decomposition
    U = linalg.cholesky(A, upper=True)
    A_reconstructed_upper = U.T @ U
    print(rm.allclose(A, A_reconstructed_upper))  # True
    
    # Batch Cholesky decomposition
    A_batch = rm.randn(4, 3, 3)
    A_batch = A_batch @ A_batch.transpose(1, 2)  # Make symmetric positive-definite
    L_batch = linalg.cholesky(A_batch)
    print(L_batch.shape)  # (4, 3, 3)

Norms and Vector Products
-------------------------

Vector Norms
~~~~~~~~~~~~

**Function**: ``linalg.norm(x, ord=None)``

**Description**: Computes the norm of a vector or matrix.

**Parameters**:
- ``x``: Input tensor (vector, matrix, or batch of vectors/matrices)
- ``ord``: Norm order. For vectors: 1, 2, inf, etc. For matrices: 'fro' (Frobenius), 'nuc' (nuclear), 2 (spectral), etc.

**Returns**:
- Norm of the input tensor as a scalar or tensor of scalars for batches

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # Vector norms
    v = rm.randn(5)
    
    # L1 norm
    l1_norm = linalg.norm(v, ord=1)
    
    # L2 norm (Euclidean norm)
    l2_norm = linalg.norm(v, ord=2)
    
    # Infinity norm
    inf_norm = linalg.norm(v, ord=float('inf'))
    
    print(l1_norm, l2_norm, inf_norm)

Matrix Norms
~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # Matrix norms
    A = rm.randn(3, 4)
    
    # Frobenius norm
    frobenius_norm = linalg.norm(A, ord='fro')
    
    # Nuclear norm (sum of singular values)
    nuclear_norm = linalg.norm(A, ord='nuc')
    
    # Spectral norm (maximum singular value)
    spectral_norm = linalg.norm(A, ord=2)
    
    print(frobenius_norm, nuclear_norm, spectral_norm)

Inner Product
~~~~~~~~~~~~~

**Function**: ``linalg.dot(a, b)``

**Description**: Computes the dot product of two vectors, or batch of vectors.

**Parameters**:
- ``a``: First input tensor (vector or batch of vectors)
- ``b``: Second input tensor (vector or batch of vectors)

**Returns**:
- Dot product of the input vectors as a scalar or tensor of scalars for batches

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # Dot product between vectors
    v1 = rm.randn(5)
    v2 = rm.randn(5)
    dot_product = linalg.dot(v1, v2)
    
    # Cosine similarity
    cosine_sim = linalg.dot(v1, v2) / (linalg.norm(v1) * linalg.norm(v2))
    
    # Euclidean distance between vectors (using norm)
    euclidean_dist = linalg.norm(v1 - v2)
    
    print(dot_product, cosine_sim, euclidean_dist)

Vector Cross Product
~~~~~~~~~~~~~~~~~~~~

**Function**: ``linalg.cross(a, b, dim=-1)``

**Description**: Computes the cross product of two 3-dimensional vectors.

**Parameters**:
- ``a``: First input tensor (3D vector or batch of 3D vectors)
- ``b``: Second input tensor (3D vector or batch of 3D vectors)
- ``dim``: Dimension along which to compute the cross product (default: -1)

**Returns**:
- Cross product of the input vectors as a tensor of the same shape

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # Cross product of 3D vectors
    v1 = rm.tensor([1.0, 2.0, 3.0])
    v2 = rm.tensor([4.0, 5.0, 6.0])
    cross_product = linalg.cross(v1, v2)
    print(cross_product)  # [-3.  6. -3.]
    
    # Batch cross product
    v1_batch = rm.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    v2_batch = rm.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    cross_product_batch = linalg.cross(v1_batch, v2_batch)
    print(cross_product_batch.shape)  # (2, 3)

Linear Systems
--------------

Solving Linear Systems
~~~~~~~~~~~~~~~~~~~~~~

**Function**: ``linalg.solve(A, b)``

**Description**: Solves a linear system of equations Ax = b.

**Parameters**:
- ``A``: Coefficient matrix (square matrix or batch of square matrices)
- ``b``: Right-hand side vector or matrix

**Returns**:
- Solution vector or matrix x such that Ax = b

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # Solve Ax = b
    A = rm.randn(3, 3)
    b = rm.randn(3)
    x = linalg.solve(A, b)
    print(x.shape)  # (3,)
    
    # Verify solution
    print(rm.allclose(A @ x, b))  # True
    
    # Solve batch linear systems
    A_batch = rm.randn(10, 3, 3)
    b_batch = rm.randn(10, 3)
    x_batch = linalg.solve(A_batch, b_batch)
    print(x_batch.shape)  # (10, 3)

Least Squares Solutions
~~~~~~~~~~~~~~~~~~~~~~~

**Function**: ``linalg.lstsq(A, b)``

**Description**: Computes the least squares solution to a linear system Ax = b.

**Parameters**:
- ``A``: Coefficient matrix
- ``b``: Right-hand side vector or matrix

**Returns**:
- Tuple containing the solution vector/matrix, residuals, rank, and singular values

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # Solve overdetermined system Ax = b using least squares
    A = rm.randn(5, 3)  # More equations than unknowns
    b = rm.randn(5)
    x = linalg.lstsq(A, b)
    print(x[0].shape)  # (3,)
    
    # Solve underdetermined system Ax = b
    A = rm.randn(3, 5)  # Fewer equations than unknowns
    b = rm.randn(3)
    x = linalg.lstsq(A, b)
    print(x[0].shape)  # (5,)

Special Matrices
----------------

Identity Matrix
~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    
    # Create identity matrix
    I = rm.eye(3)
    print(I)  # 3x3 identity matrix
    
    # Batch identity matrices
    I_batch = rm.eye(3, batch_shape=(4,))
    print(I_batch.shape)  # (4, 3, 3)

Diagonal Matrix
~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    
    # Create diagonal matrix from vector
    v = rm.randn(3)
    D = rm.diag(v)
    print(D.shape)  # (3, 3)
    
    # Extract diagonal from matrix
    A = rm.randn(3, 3)
    diag = rm.diag(A)
    print(diag.shape)  # (3,)

Triangular Matrix
~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # Create upper triangular matrix
    A = rm.randn(3, 3)
    upper = linalg.triu(A)
    print(upper)
    
    # Create lower triangular matrix
    lower = linalg.tril(A)
    print(lower)

Examples
--------

Principal Component Analysis (PCA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # Generate random data
    X = rm.randn(100, 5)  # 100 samples, 5 features
    
    # Center data
    X_centered = X - rm.mean(X, dim=0)
    
    # Compute covariance matrix
    cov_matrix = linalg.matmul(X_centered.T, X_centered) / (X.shape[0] - 1.)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = linalg.eig(cov_matrix)
    
    # Sort eigenvalues in descending order
    sorted_indices = rm.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Project data onto principal components
    X_pca = linalg.matmul(X_centered, eigenvectors)
    
    print(X_pca.shape)  # (100, 5)
    print("Explained variance ratio:", eigenvalues / rm.sum(eigenvalues))

Linear Regression
~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # Generate synthetic data
    n_samples, n_features = 100, 5
    X = rm.randn(n_samples, n_features)
    true_weights = rm.randn(n_features)
    y = linalg.matmul(X, true_weights) + 0.1 * rm.randn(n_samples)
    
    # Solve for weights using normal equations: w = (X^T X)^{-1} X^T y
    XtX = linalg.matmul(X.T, X)
    XtY = linalg.matmul(X.T, y)
    weights = linalg.solve(XtX, XtY)
    
    # Calculate predictions and MSE
    predictions = linalg.matmul(X, weights)
    mse = rm.mean((predictions - y) ** 2)
    
    print("True weights:", true_weights)
    print("Estimated weights:", weights)
    print("MSE:", mse.item())

Solving Differential Equations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # Solve linear ODE system: dx/dt = Ax
    # Use simple Euler method for numerical integration
    
    # Define system matrix
    A = rm.tensor([[-1.0, 2.0], [0.0, -3.0]])
    
    # Initial condition
    x0 = rm.tensor([1.0, 1.0])
    
    # Time parameters
    dt = 0.01  # Time step
    t_end = 1.0  # End time
    num_steps = int(t_end / dt)
    
    # Perform numerical integration using Euler method
    x = x0.clone()
    solutions = [x.clone()]
    
    for i in range(num_steps):
        # dx/dt = Ax, so x(t+dt) ≈ x(t) + dt * A * x(t)
        dx = linalg.matmul(A, x)
        x = x + dt * dx
        solutions.append(x.clone())
    
    solutions = rm.stack(solutions)
    print(solutions.shape)  # (101, 2)

Kalman Filter
~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # Simplified Kalman Filter implementation
    class KalmanFilter:
        def __init__(self, F, H, Q, R, x0, P0):
            self.F = F  # State transition matrix
            self.H = H  # Measurement matrix
            self.Q = Q  # Process noise covariance
            self.R = R  # Measurement noise covariance
            self.x = x0  # Initial state estimate
            self.P = P0  # Initial error covariance
        
        def predict(self):
            # Predict state
            self.x = linalg.matmul(self.F, self.x)
            # Predict error covariance
            self.P = linalg.matmul(self.F, linalg.matmul(self.P, self.F.T)) + self.Q
        
        def update(self, z):
            # Calculate Kalman gain
            S = linalg.matmul(self.H, linalg.matmul(self.P, self.H.T)) + self.R
            K = linalg.matmul(self.P, linalg.matmul(self.H.T, linalg.inv(S)))
            
            # Update state estimate
            y = z - linalg.matmul(self.H, self.x)  # Innovation
            self.x = self.x + linalg.matmul(K, y)
            
            # Update error covariance
            I = rm.eye(self.P.shape[0])
            self.P = linalg.matmul(I - linalg.matmul(K, self.H), self.P)
    
    # Example usage
    F = rm.tensor([[1.0, 1.0], [0.0, 1.0]])  # State transition
    H = rm.tensor([[1.0, 0.0]])              # Measurement
    Q = 0.01 * rm.eye(2)                     # Process noise
    R = 0.1 * rm.eye(1)                      # Measurement noise
    x0 = rm.tensor([0.0, 0.0])                # Initial state
    P0 = rm.eye(2)                            # Initial covariance
    
    kf = KalmanFilter(F, H, Q, R, x0, P0)
    
    # Simulated measurements
    measurements = [rm.tensor([i + 0.1 * rm.randn(1).item()]) for i in range(10)]
    
    # Run filter
    for z in measurements:
        kf.predict()
        kf.update(z)
        print(f"State estimate: {kf.x}")