简介
====

Riemann 是什么？
----------------

**Riemann** 是一个轻量级的自动求导库及神经网络编程框架，专为深度学习相关的学习、教育和研究而设计。

Riemann 能做什么？
~~~~~~~~~~~~~~~~~~

- **张量计算**：支持 0 到多维张量的数学运算，支持复数张量，支持 CPU/GPU 异构计算
- **自动求导**：基于反向传播算法实现自动梯度跟踪，支持自定义梯度函数
- **神经网络**：提供搭建神经网络所需的组件（层、激活函数、损失函数、优化器等）
- **计算机视觉**：提供图像数据集加载和图像变换功能
- **线性代数**：支持矩阵分解、特征值计算、线性方程组求解等

Riemann 的核心价值在于其**简洁性**和**可学习性**——代码结构清晰，便于理解深度学习框架的内部工作原理，是学习和研究自动求导及神经网络实现的理想工具。

主要功能
--------

张量操作
~~~~~~~~

- 提供张量创建函数（tensor, zeros, ones, randn, normal 等，支持复数张量）
- 支持基本的数学运算（加减乘除幂运算，指数、对数、三角、双曲等初等函数，求和、均值、方差、标准差等统计函数）
- 支持向量、矩阵运算（批量矩阵乘法、向量点积、矩阵行列式、矩阵逆、矩阵分解等）
- 支持张量形状重塑、维度扩缩、索引和切片、元素收集/散射、拼接/分割等操作
- 支持张量序列化/反序列化，方便模型训练和部署

自动求导
~~~~~~~~

- **backward 方法**：触发反向传播计算梯度
- **grad 函数**：计算函数相对于输入的梯度
- **func 模块**：torch.func 风格的函数式自动求导 API，提供柯里化的 ``grad``、``jvp``、``vjp``、``jacrev``、``jacfwd``、``hessian``、``vmap``、``functional_call`` 等函数，适合链式高阶求导
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
- Transformer（MultiheadAttention, TransformerEncoder, TransformerDecoder, Transformer等）
- KAN网络（KAN, KANLinear等）
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
    ├── autograd             # 自动求导模块
    │   └── functional       # 自动求导函数式接口
    ├── core                 # 核心张量子包
    │   ├── tensordef        # 张量定义（TN 类）、动态计算图与自动微分
    │   ├── tensorfuncs      # 函数式张量运算（仅单向依赖 tensordef）
    │   ├── dtype            # 数据类型定义
    │   ├── gradmode         # 梯度模式控制
    │   └── serialization    # 对象保存与加载
    ├── func                 # torch.func 风格函数式自动求导 API
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

    # 使用自动求导功能
    x = r.tensor([1.0, 2.0], requires_grad=True)
    y = x ** 2
    y.sum().backward()
    print(x.grad)  # 输出: [2. 4.]

**按模块树导入需要的函数和类：**

.. code-block:: python

    # 导入张量相关功能
    from riemann import tensor, zeros, ones, randn

    # 导入自动求导功能
    from riemann.autograd import grad, backward
    from riemann.autograd.functional import jacobian, hessian

    # 导入函数式自动求导 API（torch.func 风格，柯里化，支持高阶求导）
    from riemann import func
    from riemann.func import grad as fgrad, jvp, vjp, jacrev, jacfwd, hessian, vmap, functional_call

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
- **科学计算**：复杂数学模型的梯度计算，以及优化问题求解
- **计算机视觉**：图像分类、目标检测等视觉任务
- **教育教学**：自动求导和深度学习原理学习

PyTorch 兼容性说明
------------------

Riemann 设计时注重与 PyTorch 接口的兼容性，同名的函数和类保持一致的接口，方便 PyTorch 用户快速上手。

兼容特性
~~~~~~~~

- **张量操作**：支持与 PyTorch 同名的张量操作函数和方法
- **神经网络组件**：``nn`` 模块中的层、激活函数和损失函数与 PyTorch 保持接口兼容
- **优化器**：``optim`` 模块中的优化器（如 SGD、Adam 等）接口与 PyTorch 保持一致
- **自动求导机制**：``requires_grad``、反向传播机制与 PyTorch 相似
- **torch.func 兼容**：顶层 ``func`` 模块对齐 ``torch.func`` 接口——柯里化的 ``grad``、``jacrev``/``jacfwd``、``jvp``/``vjp``、``hessian``、``vmap``、``functional_call``
- **计算机视觉**：``vision`` 模块中的数据集和变换与 torchvision 保持接口兼容
- **跨框架互操作**：Riemann 张量可与 NumPy 数组、CuPy 数组以及 PyTorch 等其它张量框架互相转换或共享数据，入口包括 ``r.tensor()``、``r.as_tensor()``、``r.from_dlpack()``、``TN.array()``、``TN.numpy()``，底层依托 ``__array__``、``__cuda_array_interface__``、``__dlpack__``、``__buffer__`` 协议（``__buffer__`` 需 Python 3.12+）；设备与 dtype 一致时为零拷贝

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

- **KAN 网络支持**：Kolmogorov-Arnold Networks (KAN) 网络实现，包括 KAN、KANLinear等模块
- **自定义梯度装饰器** (``@track_grad``)：使用装饰器快速为函数添加自动求导支持，无需定义完整的 Function 类
- **新函数**：``sumall`` （将多个张量或非张量相加，返回总和）、``isum`` （智能求和函数，根据参数自动选择合适的求和方式）、``d()`` （张量方法，按变量链式求高阶混合偏导数）等
- **简洁的代码结构**：代码量更少，结构更清晰，便于学习和研究

安装指南
--------

Riemann 尚未发布到 PyPI，**不支持** ``pip install riemann`` 安装方式，请从源码安装。

**Python 版本**：Riemann 要求 Python >= 3.9，支持 Python 3.9 ~ 3.14。

依赖一览
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 34 50

   * - 类别
     - 依赖
     - 说明
   * - 基础依赖
     - numpy>=1.20.0、pillow>=8.0.0、tqdm>=4.0.0
     - 运行 Riemann 必需
   * - 测试依赖
     - pytest>=7.0.0、torch>=2.0.0、torchvision>=0.15.0
     - 仅运行测试时需要
   * - CUDA 依赖
     - cupy-cuda12x（默认）、cupy-cuda13x、cupy-cuda11x、cupy-cuda10x
     - 仅 NVIDIA GPU 加速时需要

源码安装
~~~~~~~~

.. code-block:: bash

    git clone https://gitee.com/xfcode2021/Riemann.git
    cd Riemann

    # 基础安装（-e 为可编辑模式，修改代码后无需重装）
    pip install -e .

    # 可选：测试依赖
    pip install -e .[tests]

    # 可选：CUDA 依赖（默认 CUDA 12.x，也可指定 cuda13 / cuda11 / cuda10）
    pip install -e .[cuda]

手动安装依赖
~~~~~~~~~~~~

若 pip/conda 因网络或依赖解析失败，可手动逐个安装：

.. code-block:: bash

    # 基础依赖
    pip install numpy pillow tqdm
    # 测试依赖
    pip install pytest torch torchvision
    # CUDA 依赖（按 nvcc --version 显示的版本选择）
    pip install cupy-cuda12x

- 系统环境可用操作系统包管理器安装（如 ``apt install python3-numpy``、``brew install numpy``）；**自建虚拟环境只能用 pip 安装**；
- torch 请按 `PyTorch 官网 <https://pytorch.org/get-started/locally/>`_ 给出的命令单独安装；
- Conda 环境可用 ``conda create -n riemann python=3.10`` 创建，但 CuPy 仍建议用 pip 安装。

装好依赖后执行 ``pip install -e . --no-deps`` 完成 Riemann 本体安装。

CUDA 依赖与平台兼容性
~~~~~~~~~~~~~~~~~~~~~

CUDA 依赖不会自动安装，需用 ``pip install -e .[cuda]`` 显式安装，并保证已装对应版本的 NVIDIA 驱动。

.. list-table::
   :header-rows: 1
   :widths: 18 18 18 46

   * - 平台
     - 架构
     - CUDA 支持
     - 说明
   * - Linux
     - x86_64/AMD64
     - ✅ 支持
     - 用 ``pip install cupy-cudaXX``
   * - Windows
     - x86_64/AMD64
     - ✅ 支持
     - 用 ``pip install cupy-cudaXX``
   * - macOS
     - x86_64/ARM64
     - ❌ 不支持
     - 无 NVIDIA 驱动，自动使用 CPU 模式
   * - Linux (ARM64)
     - aarch64/arm64
     - ⚠️ 需源码编译
     - NVIDIA Jetson 等需从源码编译 CuPy
   * - Windows (ARM64)
     - aarch64/arm64
     - ❌ 不支持
     - 暂不支持 CUDA

CUDA 驱动安装
~~~~~~~~~~~~~

1. 运行 ``nvcc --version`` 查看已安装的 CUDA Toolkit 版本（``nvidia-smi`` 显示的是驱动支持的最高版本，不等于已安装版本）。
2. 从 `CUDA Toolkit Archive <https://developer.nvidia.com/cuda-toolkit-archive>`_ 下载并安装匹配版本的驱动/工具包。
3. 重启后再次运行 ``nvcc --version`` 确认版本。

验证安装
~~~~~~~~

安装完成后，可以运行以下代码验证：

.. code-block:: python

    import riemann as r
    print("CUDA 可用:", r.cuda.is_available())
    print("使用设备:", r.device('cuda' if r.cuda.is_available() else 'cpu'))

如果 CUDA 安装成功，会显示 ``CUDA 可用: True``，否则会显示 ``CUDA 可用: False`` 并自动使用 CPU 模式。
