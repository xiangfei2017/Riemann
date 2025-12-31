线性代数
========

Riemann 通过 ``riemann.linalg`` 模块提供了全面的线性代数运算。这些运算对于许多机器学习和科学计算应用至关重要。

矩阵运算
~~~~~~~~

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

范数和度量
~~~~~~~~~~

向量范数
~~~~~~~~

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

距离度量
~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.linalg as linalg
    
    # 向量之间的欧几里得距离
    v1 = rm.randn(5)
    v2 = rm.randn(5)
    euclidean_dist = linalg.norm(v1 - v2)
    
    # 余弦相似度
    cosine_sim = linalg.dot(v1, v2) / (linalg.norm(v1) * linalg.norm(v2))
    
    print(euclidean_dist, cosine_sim)

线性系统
--------

求解线性系统
~~~~~~~~~~~~

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