简介
====

Riemann 是什么？
----------------

**Riemann** 是一个轻量级的自动求导库及神经网络编程框架，专为深度学习相关的学习、教育和研究而设计。

Riemann 能做什么？
~~~~~~~~~~~~~~~~~~

- **张量计算**：支持 0 到多维张量的数学运算，支持复数张量，支持 CPU/GPU 异构计算
- **自动求导**：基于反向传播算法实现自动梯度跟踪，支持自定义梯度函数
- **神经网络**：提供搭建神经网络所需的常用组件（层、激活函数、损失函数、优化器等）
- **计算机视觉**：提供图像数据集加载和图像变换功能
- **线性代数**：支持矩阵分解、特征值计算、线性方程组求解等

Riemann 的核心价值在于其**简洁性**和**可学习性**——代码结构清晰，便于理解深度学习框架的内部工作原理，是学习和研究自动微分及神经网络实现的理想工具。

主要功能
--------

张量操作
~~~~~~~~

- 提供张量创建函数（tensor, zeros, ones, randn, normal 等，支持复数张量）
- 支持基本的数学运算（加减乘除幂运算，指数、对数、三角、双曲等初等函数，求和、均值、方差、标准差等统计函数）
- 支持向量、矩阵运算（批量矩阵乘法、向量点积、矩阵行列式、矩阵逆、矩阵分解等）
- 支持张量形状重塑、维度扩缩、索引和切片、元素收集/散射、拼接/分割等操作
- 支持张量序列化/反序列化，方便模型训练和部署
- **特有功能**：``sumall`` 函数（将多个张量或非张量相加）、``isum`` 函数（智能求和）

自动求导
~~~~~~~~

- **backward 方法**：触发反向传播计算梯度
- **grad 函数**：计算函数相对于输入的梯度
- **track_grad 装饰器**：使用 ``@track_grad`` 装饰器自定义梯度跟踪函数（Riemann 特有）
- **Function 类**：通过继承 Function 类自定义前向和反向传播（Riemann 特有）
- **雅可比矩阵和海森矩阵**：支持多输入多输出函数的雅可比矩阵计算，支持多输入函数的海森矩阵计算

线性代数模块
~~~~~~~~~~~~

- 提供矩阵分解及其反向梯度跟踪（SVD、PLU, QR 等）
- 支持求矩阵逆、广义逆、行列式、特征值/特征向量
- 矩阵范数、条件数计算
- 支持线性方程组求解、最小二乘求解

神经网络模块
~~~~~~~~~~~~

- 基本层（Linear, Dropout, BatchNorm, LayerNorm, Embedding, 各类归一化层等）
- 激活函数（ReLU, Sigmoid, Softmax, Tanh, GELU, SiLU 等）
- 损失函数（MSE, CrossEntropy, BCE, L1Loss, NLLLoss 等）
- 卷积池化（Conv1d/2d/3d, MaxPool1d/2d/3d, AvgPool1d/2d/3d, AdaptivePool 等）
- Transformer(MultiheadAttention, TransformerEncoder, TransformerDecoder, Transformer等)
- 优化器（SGD, Adam, Adagrad, AdamW, RMSprop, LBFGS 等）
- 学习率调度器（StepLR, ExponentialLR, CosineAnnealingLR 等）
- 网络模块容器（Sequential, ModuleList, ModuleDict 等）

计算机视觉模块
~~~~~~~~~~~~~~

- 数据集类：

  - **MNIST**：手写数字识别数据集
  - **CIFAR10**：10 类彩色图像数据集
  - **ImageFolder**：从文件夹加载图像数据集（按类别分子文件夹）
  - **DatasetFolder**：通用文件夹数据集基类

- 图像变换（40+ 种变换）：

  - **类型转换**：ToTensor, PILToTensor, ToPILImage, ConvertImageDtype
  - **几何变换**：Resize, CenterCrop, RandomCrop, RandomResizedCrop, FiveCrop, TenCrop, Pad
  - **翻转旋转**：RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomAffine, RandomPerspective
  - **颜色变换**：ColorJitter, Grayscale, RandomGrayscale, Invert, Posterize, Solarize, Equalize, AutoContrast, Sharpness, Brightness, Contrast, Saturation, Hue
  - **数据增强**：AutoAugment, RandAugment, TrivialAugmentWide, RandomErasing
  - **其他变换**：Normalize, GaussianBlur, Lambda, SanitizeBoundingBox

CUDA/GPU 支持
~~~~~~~~~~~~~

- 提供 GPU 加速，支持张量、模型在 CPU 和 GPU 之间迁移
- 优化的 GPU 计算性能
- 支持 Windows 和 Linux 平台（macOS 仅支持 CPU 模式）

riemann包的模块结构
--------------------

.. code-block:: text

    riemann                  # 主包
    ├── autograd             # 自动微分模块
    │   └── functional       # 自动微分函数式接口
    ├── linalg               # 线性代数模块
    ├── nn                   # 神经网络模块
    │   └── functional       # 神经网络函数
    ├── optim                # 优化器模块
    │   └── lr_scheduler     # 学习率调度器模块
    ├── utils                # 工具函数模块
    │   └── data             # 数据处理工具
    ├── vision               # 计算机视觉模块
    │   ├── datasets         # 数据集类
    │   └── transforms       # 图像变换操作
    └── cuda                 # CUDA/GPU支持

模块导入示例
------------

**整体导入riemann模块：**

.. code-block:: python

    import riemann as r

    # 使用张量创建函数
    t = r.tensor([1.0, 2.0, 3.0])

    # 使用自动微分功能
    x = r.tensor([1.0, 2.0], requires_grad=True)
    y = x ** 2
    y.sum().backward()
    print(x.grad)  # 输出: [2. 4.]

**按模块树导入需要的函数和类：**

.. code-block:: python

    # 导入张量相关功能
    from riemann import tensor, zeros, ones, randn

    # 导入自动微分功能
    from riemann.autograd import grad, backward
    from riemann.autograd.functional import jacobian, hessian

    # 导入线性代数功能
    from riemann import linalg
    from riemann.linalg import svd, det, inv

    # 导入神经网络组件
    from riemann.nn import Linear, Conv2d, ReLU, CrossEntropyLoss
    from riemann.nn.functional import relu, cross_entropy

    # 导入优化器
    from riemann.optim import SGD, Adam, Adagrad

    # 导入计算机视觉功能
    from riemann.vision.datasets import MNIST, CIFAR10, ImageFolder
    from riemann.vision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip

    # 导入CUDA支持
    from riemann import cuda
    from riemann.cuda import is_available, Device

应用场景
--------

- **深度学习研究**：自定义模型和算法开发
- **科学计算**：复杂数学模型的梯度计算
- **优化问题求解**：梯度下降和 Adam 等优化算法
- **计算机视觉**：图像分类、目标检测等视觉任务
- **教育教学**：自动微分和深度学习原理学习

PyTorch 兼容性说明
------------------

Riemann 设计时注重与 PyTorch 接口的兼容性，同名的函数和类保持一致的接口，方便 PyTorch 用户快速上手。

兼容特性
~~~~~~~~

- **张量操作**：支持与 PyTorch 同名的张量操作函数和方法
- **神经网络组件**：``nn`` 模块中的层、激活函数和损失函数与 PyTorch 保持接口兼容
- **优化器**：``optim`` 模块中的优化器（如 SGD、Adam 等）接口与 PyTorch 保持一致
- **自动微分机制**：``requires_grad``、反向传播机制与 PyTorch 相似
- **计算机视觉**：``vision`` 模块中的数据集和变换与 torchvision 保持接口兼容

Riemann 不支持的 PyTorch 特性
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Riemann 作为轻量级框架，以下 PyTorch 的高级特性**暂不支持**：

- **分布式训练**：不支持 DataParallel、DistributedDataParallel 等多 GPU 分布式训练
- **JIT 编译**：不支持 TorchScript 编译和优化
- **量化**：不支持模型量化（INT8 等）
- **ONNX 导出**：不支持导出为 ONNX 格式
- **移动端部署**：不支持 TorchMobile、Core ML 等移动端部署
- **高级优化器**：部分高级优化器（如 Adamax、SparseAdam）暂不支持
- **动态图高级特性**：如某些复杂的控制流和动态形状操作

Riemann 特有功能
~~~~~~~~~~~~~~~~

Riemann 提供了一些 PyTorch 没有的功能：

- **自定义梯度装饰器** (``@track_grad``)：使用装饰器快速为函数添加自动微分支持，无需定义完整的 Function 类
- **高级计算图构建**：更灵活的计算图构建和管理机制
- **sumall 函数**：将多个张量或非张量相加，返回总和
- **isum 函数**：智能求和函数，根据参数自动选择合适的求和方式
- **简洁的代码结构**：代码量更少，结构更清晰，便于学习和研究

安装指南
--------

Riemann 安装包括以下几个部分：

1. **核心包**：Riemann 主库，包含张量计算、自动求导、神经网络等核心功能
2. **核心依赖**：NumPy、Pillow、tqdm 等必需依赖
3. **CUDA 依赖** (可选)：CuPy 库，用于 GPU 加速
4. **测试依赖** (可选)：pytest 等测试框架

使用 Conda 安装（推荐）
~~~~~~~~~~~~~~~~~~~~~~~

Conda 能更好地管理复杂依赖，特别是 CUDA 相关包：

.. code-block:: bash

    # 创建新的 conda 环境
    conda create -n riemann python=3.10
    conda activate riemann

    # 安装核心依赖
    conda install numpy pillow tqdm

    # 从源码安装 Riemann（可选择 GitHub 或 Gitee 源）
    # GitHub 源（国际访问）
    git clone https://github.com/xiangfei2017/Riemann.git
    # 或 Gitee 源（国内访问更快）
    # git clone https://gitee.com/xfcode2021/Riemann.git
    cd Riemann
    pip install -e .

    # 安装测试依赖（可选）
    pip install -e .[tests]

    # 安装 CUDA 依赖（可选）
    pip install -e .[cuda]

使用 pip 安装
~~~~~~~~~~~~~

.. code-block:: bash

    # 获取 Riemann 库源码（可选择 GitHub 或 Gitee 源）
    # GitHub 源（国际访问）
    git clone https://github.com/xiangfei2017/Riemann.git
    # 或 Gitee 源（国内访问更快）
    # git clone https://gitee.com/xfcode2021/Riemann.git
    cd Riemann
    
    # 使用开发模式安装包及其核心依赖
    pip install -e .

    # 安装测试依赖
    pip install -e .[tests]

CUDA 支持安装说明
~~~~~~~~~~~~~~~~~

.. important::

    要使 Riemann 支持 CUDA 加速，必须满足以下 **三个条件**，缺一不可：

    1. **NVIDIA GPU 硬件**：计算机必须配备 NVIDIA 显卡
    2. **CUDA 驱动**：必须安装与 GPU 兼容的 NVIDIA CUDA 驱动程序
    3. **CuPy 库**：必须安装与 CUDA 版本匹配的 CuPy 库

    仅安装 CuPy 并不能使 Riemann 支持 CUDA，必须先确保硬件和驱动已正确安装。

**CUDA 安装步骤：**

1. **检查 GPU 硬件**
   确认计算机配备 NVIDIA 显卡，且显卡支持 CUDA。

2. **安装 CUDA 驱动**

   - **Windows/Linux**: 访问 `NVIDIA 官网 <https://developer.nvidia.com/cuda-toolkit-archive>`_ 下载并安装对应版本的 CUDA Toolkit
   - 安装后验证：运行 ``nvcc --version`` 查看 CUDA 版本
   - 注意：``nvidia-smi`` 显示的是驱动支持的最高 CUDA 版本，``nvcc --version`` 显示的是实际安装的版本

3. **安装 CuPy 库**

   根据已安装的 CUDA 版本，选择对应的 CuPy 包：

   .. code-block:: bash

       # 安装 CUDA 依赖（自动检测 CUDA 版本并安装对应 CuPy）
       pip install -e .[cuda]

       # 或安装特定版本的 CUDA 依赖
       pip install -e .[cuda13]  # CUDA 13.x
       pip install -e .[cuda12]  # CUDA 12.x
       pip install -e .[cuda11]  # CUDA 11.x
       pip install -e .[cuda10]  # CUDA 10.x (仅 Linux)

**版本配套关系：**

- CuPy 版本必须与 CUDA Toolkit 版本匹配
- 不同 CUDA 版本之间不兼容，请确保选择正确的版本
- 建议使用 CUDA 11.x 或 12.x 以获得更好的兼容性

依赖说明
~~~~~~~~

**核心依赖** (自动安装)：

- **numpy>=1.20.0**: 核心数值计算库
- **pillow>=8.0.0**: 图像处理功能
- **tqdm>=4.0.0**: 进度条显示

**CUDA 依赖** (需显式安装，且需要先安装 CUDA 驱动)：

- **cupy-cuda13x**: 适用于 CUDA 13.x
- **cupy-cuda12x**: 适用于 CUDA 12.x
- **cupy-cuda11x**: 适用于 CUDA 11.x
- **cupy-cuda10x**: 适用于 CUDA 10.x (仅 Linux)

平台兼容性
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - 平台
     - 架构
     - CUDA 支持
     - 安装方式
   * - Linux
     - x86_64/AMD64
     - ✅ 支持
     - ``pip install -e .[cuda]``
   * - Windows
     - x86_64/AMD64
     - ✅ 支持
     - ``pip install -e .[cuda]``
   * - macOS
     - x86_64/ARM64
     - ❌ 不支持
     - 无 NVIDIA GPU 驱动，使用 CPU 模式
   * - Linux (ARM64)
     - aarch64/arm64
     - ⚠️ 需源码编译
     - NVIDIA Jetson 等需从源码编译 CuPy

验证安装
~~~~~~~~

安装完成后，可以运行以下代码验证：

.. code-block:: python

    import riemann as r
    print("CUDA 可用:", r.cuda.is_available())
    print("使用设备:", r.device('cuda' if r.cuda.is_available() else 'cpu'))

如果 CUDA 安装成功，会显示 ``CUDA 可用: True``，否则会显示 ``CUDA 可用: False`` 并自动使用 CPU 模式。
