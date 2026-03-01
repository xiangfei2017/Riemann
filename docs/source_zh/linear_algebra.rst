线性代数
========

Riemann 通过 ``riemann.linalg`` 模块提供了全面的线性代数运算。这些运算对于许多机器学习和科学计算应用至关重要。

矩阵运算
--------

矩阵乘法
~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # 矩阵乘法
    A = rm.randn(3, 4)
    B = rm.randn(4, 5)
    C = linalg.matmul(A, B)  # 或者简单地使用 A @ B
    print(C.shape)  # (3, 5)
    
    # 批量矩阵乘法
    A_batch = rm.randn(10, 3, 4)
    B_batch = rm.randn(10, 4, 5)
    C_batch = linalg.matmul(A_batch, B_batch)
    print(C_batch.shape)  # (10, 3, 5)

矩阵转置
~~~~~~~~

.. code-block:: python

    import riemann as rm
    
    # 转置矩阵
    A = rm.randn(3, 4)
    A_T = A.T  # 或者 A.transpose()
    print(A_T.shape)  # (4, 3)
    
    # 转置批量矩阵
    A_batch = rm.randn(10, 3, 4)
    A_batch_T = A_batch.transpose(1, 2)  # 交换维度 1 和 2
    print(A_batch_T.shape)  # (10, 4, 3)

矩阵求逆
~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # 矩阵逆
    A = rm.randn(3, 3)
    A_inv = linalg.inv(A)
    print(A_inv.shape)  # (3, 3)
    
    # 验证 A @ A_inv ≈ I
    identity = linalg.matmul(A, A_inv)
    print(identity)  # 应该接近单位矩阵
    
    # 非方阵的伪逆
    A = rm.randn(3, 4)
    A_pinv = linalg.pinv(A)
    print(A_pinv.shape)  # (4, 3)

矩阵行列式
~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # 矩阵行列式
    A = rm.randn(3, 3)
    det = linalg.det(A)
    print(det)  # 标量值
    
    # 批量行列式
    A_batch = rm.randn(10, 3, 3)
    det_batch = linalg.det(A_batch)
    print(det_batch.shape)  # (10,)

分解
----

奇异值分解 (SVD)
~~~~~~~~~~~~~~~~~~

**函数**: ``linalg.svd(A, full_matrices=True)``

**描述**: 计算矩阵的奇异值分解。

**参数**:

- ``A``: 输入张量（矩阵或批量矩阵）
- ``full_matrices``: 如果为True，返回全尺寸的U和Vh矩阵。如果为False，返回简化尺寸的矩阵。

**返回值**:

- ``U``: 酉矩阵
- ``S``: 奇异值，作为一维张量
- ``Vh``: 酉矩阵（V的共轭转置）

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # 矩阵的 SVD
    A = rm.randn(4, 3)
    U, S, Vh = linalg.svd(A)
    print(U.shape, S.shape, Vh.shape)  # (4, 4), (3,), (3, 3)
    
    # 重构矩阵
    A_reconstructed = U @ rm.diag(S) @ Vh
    print(rm.allclose(A, A_reconstructed))  # True
    
    # 简化 SVD
    U, S, Vh = linalg.svd(A, full_matrices=False)
    print(U.shape, S.shape, Vh.shape)  # (4, 3), (3,), (3, 3)

特征值分解
~~~~~~~~~~

**函数**: ``linalg.eig(A)``

**描述**: 计算方阵的特征值和特征向量。

**参数**:

- ``A``: 输入张量（方阵或批量方阵）

**返回值**:

- ``eigenvalues``: 矩阵的特征值，作为一维张量
- ``eigenvectors``: 矩阵的特征向量，每一列是一个特征向量

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # 对称矩阵的特征值分解
    A = rm.randn(3, 3)
    A = A + A.T  # 使其对称
    eigenvalues, eigenvectors = linalg.eig(A)
    print(eigenvalues.shape, eigenvectors.shape)  # (3,), (3, 3)
    
    # 重构矩阵
    A_reconstructed = eigenvectors @ rm.diag(eigenvalues) @ eigenvectors.T
    print(rm.allclose(A, A_reconstructed))  # True

QR 分解
~~~~~~~

**函数**: ``linalg.qr(A)``

**描述**: 计算矩阵的QR分解，将矩阵分解为正交矩阵Q和上三角矩阵R。

**参数**:

- ``A``: 输入张量（矩阵或批量矩阵）

**返回值**:

- ``Q``: 正交矩阵
- ``R``: 上三角矩阵

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # QR 分解
    A = rm.randn(4, 3)
    Q, R = linalg.qr(A)
    print(Q.shape, R.shape)  # (4, 3), (3, 3)
    
    # 重构矩阵
    A_reconstructed = Q @ R
    print(rm.allclose(A, A_reconstructed))  # True

LU 分解
~~~~~~~

**函数**: ``linalg.lu(A)``

**描述**: 计算矩阵的LU分解（带部分主元），将矩阵分解为置换矩阵P、下三角矩阵L和上三角矩阵U，使得A = P @ L @ U。

**参数**:

- ``A``: 输入张量（矩阵或批量矩阵）

**返回值**:

- ``P``: 置换矩阵
- ``L``: 下三角矩阵
- ``U``: 上三角矩阵

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # 带主元的 LU 分解 (A = PLU)
    A = rm.randn(4, 4)
    P, L, U = linalg.lu(A)
    print(P.shape, L.shape, U.shape)  # (4, 4), (4, 4), (4, 4)
    
    # 重构矩阵
    A_reconstructed = P @ L @ U
    print(rm.allclose(A, A_reconstructed))  # True
    
    # 矩形矩阵的 LU 分解
    A_rect = rm.randn(3, 5)
    P, L, U = linalg.lu(A_rect)
    print(P.shape, L.shape, U.shape)  # (3, 3), (3, 3), (3, 5)
    
    # 注意：pivot=False 尚未实现
    # P, L, U = linalg.lu(A, pivot=False)  # 抛出 NotImplementedError

Cholesky 分解
~~~~~~~~~~~~~

**函数**: ``linalg.cholesky(A, upper=False)``

**描述**: 计算对称正定矩阵的Cholesky分解，将矩阵分解为下三角矩阵L，使得A = L @ L.T（如果upper=True，则分解为上三角矩阵U，使得A = U.T @ U）。

**参数**:

- ``A``: 输入张量（对称正定矩阵或批量对称正定矩阵）
- ``upper``: 如果为True，返回上三角矩阵。如果为False（默认），返回下三角矩阵。

**返回值**:

- ``L``: 下三角或上三角矩阵，使得A = L @ L.T（或U.T @ U）

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # 对称正定矩阵的 Cholesky 分解
    A = rm.randn(3, 3)
    A = A @ A.T  # 使其对称正定
    L = linalg.cholesky(A)
    print(L.shape)  # (3, 3)
    
    # 重构矩阵
    A_reconstructed = L @ L.T
    print(rm.allclose(A, A_reconstructed))  # True
    
    # 上三角 Cholesky 分解
    U = linalg.cholesky(A, upper=True)
    A_reconstructed_upper = U.T @ U
    print(rm.allclose(A, A_reconstructed_upper))  # True
    
    # 批量 Cholesky 分解
    A_batch = rm.randn(4, 3, 3)
    A_batch = A_batch @ A_batch.transpose(1, 2)  # 使其对称正定
    L_batch = linalg.cholesky(A_batch)
    print(L_batch.shape)  # (4, 3, 3)

范数和向量乘积
--------------

向量范数
~~~~~~~~

**函数**: ``linalg.norm(x, ord=None)``

**描述**: 计算向量或矩阵的范数。

**参数**:

- ``x``: 输入张量（向量、矩阵或批量向量/矩阵）
- ``ord``: 范数阶数。对于向量：1, 2, inf等。对于矩阵：'fro'（Frobenius）、'nuc'（核）、2（谱）等。

**返回值**:

- 输入张量的范数，作为标量或批量标量张量

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # 向量范数
    v = rm.randn(5)
    
    # L1 范数
    l1_norm = linalg.norm(v, ord=1)
    
    # L2 范数（欧几里得范数）
    l2_norm = linalg.norm(v, ord=2)
    
    # 无穷范数
    inf_norm = linalg.norm(v, ord=float('inf'))
    
    print(l1_norm, l2_norm, inf_norm)

矩阵范数
~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # 矩阵范数
    A = rm.randn(3, 4)
    
    # Frobenius 范数
    frobenius_norm = linalg.norm(A, ord='fro')
    
    # 核范数（奇异值之和）
    nuclear_norm = linalg.norm(A, ord='nuc')
    
    # 谱范数（最大奇异值）
    spectral_norm = linalg.norm(A, ord=2)
    
    print(frobenius_norm, nuclear_norm, spectral_norm)

内积
~~~~

**函数**: ``linalg.dot(a, b)``

**描述**: 计算两个向量或批量向量的点积。

**参数**:

- ``a``: 第一个输入张量（向量或批量向量）
- ``b``: 第二个输入张量（向量或批量向量）

**返回值**:

- 输入向量的点积，作为标量或批量标量张量

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # 向量之间的点积
    v1 = rm.randn(5)
    v2 = rm.randn(5)
    dot_product = linalg.dot(v1, v2)
    
    # 余弦相似度
    cosine_sim = linalg.dot(v1, v2) / (linalg.norm(v1) * linalg.norm(v2))
    
    # 向量之间的欧几里得距离（使用范数）
    euclidean_dist = linalg.norm(v1 - v2)
    
    print(dot_product, cosine_sim, euclidean_dist)

矢量交叉积
~~~~~~~~~~

**函数**: ``linalg.cross(a, b, dim=-1)``

**描述**: 计算两个三维向量的叉积。

**参数**:

- ``a``: 第一个输入张量（三维向量或批量三维向量）
- ``b``: 第二个输入张量（三维向量或批量三维向量）
- ``dim``: 计算叉积的维度（默认：-1）

**返回值**:

- 输入向量的叉积，形状与输入相同的张量

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # 三维向量的叉积
    v1 = rm.tensor([1.0, 2.0, 3.0])
    v2 = rm.tensor([4.0, 5.0, 6.0])
    cross_product = linalg.cross(v1, v2)
    print(cross_product)  # [-3.  6. -3.]
    
    # 批量叉积
    v1_batch = rm.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    v2_batch = rm.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    cross_product_batch = linalg.cross(v1_batch, v2_batch)
    print(cross_product_batch.shape)  # (2, 3)

线性系统
--------

求解线性系统
~~~~~~~~~~~~

**函数**: ``linalg.solve(A, b)``

**描述**: 求解线性方程组 Ax = b。

**参数**:

- ``A``: 系数矩阵（方阵或批量方阵）
- ``b``: 右侧向量或矩阵

**返回值**:

- 解向量或矩阵 x，使得 Ax = b

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # 求解 Ax = b
    A = rm.randn(3, 3)
    b = rm.randn(3)
    x = linalg.solve(A, b)
    print(x.shape)  # (3,)
    
    # 验证解
    print(rm.allclose(A @ x, b))  # True
    
    # 求解批量线性系统
    A_batch = rm.randn(10, 3, 3)
    b_batch = rm.randn(10, 3)
    x_batch = linalg.solve(A_batch, b_batch)
    print(x_batch.shape)  # (10, 3)

最小二乘解
~~~~~~~~~~

**函数**: ``linalg.lstsq(A, b)``

**描述**: 计算线性系统 Ax = b 的最小二乘解。

**参数**:

- ``A``: 系数矩阵
- ``b``: 右侧向量或矩阵

**返回值**:

- 包含解向量/矩阵、残差、秩和奇异值的元组

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # 使用最小二乘法求解超定系统 Ax = b
    A = rm.randn(5, 3)  # 方程数多于未知数
    b = rm.randn(5)
    x = linalg.lstsq(A, b)
    print(x[0].shape)  # (3,)
    
    # 求解欠定系统 Ax = b
    A = rm.randn(3, 5)  # 方程数少于未知数
    b = rm.randn(3)
    x = linalg.lstsq(A, b)
    print(x[0].shape)  # (5,)

特殊矩阵
--------

单位矩阵
~~~~~~~~

.. code-block:: python

    import riemann as rm
    
    # 创建单位矩阵
    I = rm.eye(3)
    print(I)  # 3x3 单位矩阵
    
    # 批量单位矩阵
    I_batch = rm.eye(3, batch_shape=(4,))
    print(I_batch.shape)  # (4, 3, 3)

对角矩阵
~~~~~~~~

.. code-block:: python

    import riemann as rm
    
    # 从向量创建对角矩阵
    v = rm.randn(3)
    D = rm.diag(v)
    print(D.shape)  # (3, 3)
    
    # 从矩阵提取对角线
    A = rm.randn(3, 3)
    diag = rm.diag(A)
    print(diag.shape)  # (3,)

三角矩阵
~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # 创建上三角矩阵
    A = rm.randn(3, 3)
    upper = linalg.triu(A)
    print(upper)
    
    # 创建下三角矩阵
    lower = linalg.tril(A)
    print(lower)

示例
----

主成分分析 (PCA)
~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # 生成随机数据
    X = rm.randn(100, 5)  # 100 个样本，5 个特征
    
    # 中心化数据
    X_centered = X - rm.mean(X, dim=0)
    
    # 计算协方差矩阵
    cov_matrix = linalg.matmul(X_centered.T, X_centered) / (X.shape[0] - 1.)
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = linalg.eig(cov_matrix)
    
    # 按降序排序特征值
    sorted_indices = rm.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # 将数据投影到主成分上
    X_pca = linalg.matmul(X_centered, eigenvectors)
    
    print(X_pca.shape)  # (100, 5)
    print("解释方差比:", eigenvalues / rm.sum(eigenvalues))

线性回归
~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # 生成合成数据
    n_samples, n_features = 100, 5
    X = rm.randn(n_samples, n_features)
    true_weights = rm.randn(n_features)
    y = linalg.matmul(X, true_weights) + 0.1 * rm.randn(n_samples)
    
    # 使用正规方程求解权重：w = (X^T X)^(-1) X^T y
    XtX = linalg.matmul(X.T, X)
    XtY = linalg.matmul(X.T, y)
    weights = linalg.solve(XtX, XtY)
    
    # 计算预测和 MSE
    predictions = linalg.matmul(X, weights)
    mse = rm.mean((predictions - y) ** 2)
    
    print("真实权重:", true_weights)
    print("估计权重:", weights)
    print("MSE:", mse.item())

求解微分方程
~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # 求解线性常微分方程组：dx/dt = Ax
    # 使用简单的欧拉方法进行数值积分
    
    # 定义系统矩阵
    A = rm.tensor([[-1.0, 2.0], [0.0, -3.0]])
    
    # 初始条件
    x0 = rm.tensor([1.0, 1.0])
    
    # 时间参数
    dt = 0.01  # 时间步长
    t_end = 1.0  # 结束时间
    num_steps = int(t_end / dt)
    
    # 使用欧拉方法进行数值积分
    x = x0.clone()
    solutions = [x.clone()]
    
    for i in range(num_steps):
        # dx/dt = Ax，所以 x(t+dt) ≈ x(t) + dt * A * x(t)
        dx = linalg.matmul(A, x)
        x = x + dt * dx
        solutions.append(x.clone())
    
    solutions = rm.stack(solutions)
    print(solutions.shape)  # (101, 2)

卡尔曼滤波器
~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # 简化的卡尔曼滤波器实现
    class KalmanFilter:
        def __init__(self, F, H, Q, R, x0, P0):
            self.F = F  # 状态转移矩阵
            self.H = H  # 观测矩阵
            self.Q = Q  # 过程噪声协方差
            self.R = R  # 测量噪声协方差
            self.x = x0  # 初始状态估计
            self.P = P0  # 初始误差协方差
        
        def predict(self):
            # 预测状态
            self.x = linalg.matmul(self.F, self.x)
            # 预测误差协方差
            self.P = linalg.matmul(self.F, linalg.matmul(self.P, self.F.T)) + self.Q
        
        def update(self, z):
            # 计算卡尔曼增益
            S = linalg.matmul(self.H, linalg.matmul(self.P, self.H.T)) + self.R
            K = linalg.matmul(self.P, linalg.matmul(self.H.T, linalg.inv(S)))
            
            # 更新状态估计
            y = z - linalg.matmul(self.H, self.x)  # 新息
            self.x = self.x + linalg.matmul(K, y)
            
            # 更新误差协方差
            I = rm.eye(self.P.shape[0])
            self.P = linalg.matmul(I - linalg.matmul(K, self.H), self.P)
    
    # 示例用法
    F = rm.tensor([[1.0, 1.0], [0.0, 1.0]])  # 状态转移
    H = rm.tensor([[1.0, 0.0]])              # 观测
    Q = 0.01 * rm.eye(2)                     # 过程噪声
    R = 0.1 * rm.eye(1)                      # 测量噪声
    x0 = rm.tensor([0.0, 0.0])                # 初始状态
    P0 = rm.eye(2)                            # 初始协方差
    
    kf = KalmanFilter(F, H, Q, R, x0, P0)
    
    # 模拟测量
    measurements = [rm.tensor([i + 0.1 * rm.randn(1).item()]) for i in range(10)]
    
    # 运行滤波器
    for z in measurements:
        kf.predict()
        kf.update(z)
        print(f"状态估计: {kf.x}")