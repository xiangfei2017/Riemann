计算机视觉
==========

Riemann 通过 ``riemann.vision`` 模块为计算机视觉任务提供全面的支持，包括常用数据集、图像变换和数据加载工具。

.. contents:: 目录
   :local:
   :depth: 2

概述
----

``riemann.vision`` 模块包含以下主要组件：

- **数据集 (datasets)**: MNIST、CIFAR-10、Flowers102、OxfordIIITPet、LFWPeople、SVHN、ImageFolder 等常用数据集
- **图像变换 (transforms)**: 图像预处理和数据增强的变换操作
- **数据加载**: 与 ``DataLoader`` 无缝集成，支持批量加载和并行处理

快速开始
--------

.. code-block:: python

    import riemann as rm
    from riemann.vision import datasets, transforms
    from riemann.utils.data import DataLoader

    # 定义数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载数据集
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 遍历数据
    for images, labels in train_loader:
        print(f"图像批次形状: {images.shape}")  # [64, 1, 28, 28]
        print(f"标签批次形状: {labels.shape}")  # [64]
        break

数据集 (Datasets)
-----------------

Riemann 提供了多种常用的计算机视觉数据集，所有数据集都继承自 ``Dataset`` 类，可与 ``DataLoader`` 配合使用。

数据集概览
~~~~~~~~~~

.. list-table:: 支持的数据集
   :header-rows: 1
   :widths: 20 35 15 30

   * - 数据集
     - 描述
     - 大小
     - 下载源
   * - MNIST
     - 手写数字识别（0-9），28×28 灰度图像
     - 60,000 训练 / 10,000 测试
     - AWS S3 (ossci-datasets)
   * - FashionMNIST
     - 时尚产品图像（10 类），28×28 灰度图像
     - 60,000 训练 / 10,000 测试
     - Zalando Research
   * - CIFAR-10
     - 10 类物体识别，32×32 彩色图像
     - 50,000 训练 / 10,000 测试
     - 多伦多大学
   * - CIFAR-100
     - 100 类物体识别（含 20 个超类），32×32 彩色图像
     - 50,000 训练 / 10,000 测试
     - 多伦多大学
   * - Flowers102
     - 102 种花卉分类数据集
     - 1,020 训练 / 1,020 验证 / 6,149 测试
     - Oxford VGG
   * - OxfordIIITPet
     - 37 种宠物品种（猫和狗）分类
     - 约 7,000 张图像（每类约 200 张）
     - Oxford VGG
   * - LFWPeople
     - 人脸识别数据集，包含多个身份
     - 13,233 张图像 / 5,749 人
     - UMass Amherst
   * - SVHN
     - 街景门牌号码，32×32 彩色图像
     - 73,257 训练 / 26,032 测试 / 531,131 额外
     - 斯坦福大学
   * - ImageFolder
     - 通用文件夹式数据集加载器
     - 用户自定义
     - 本地文件
   * - DatasetFolder
     - 通用文件夹数据集，支持自定义加载器
     - 用户自定义
     - 本地文件

MNIST 数据集
~~~~~~~~~~~~

手写数字识别数据集，包含 60,000 张训练图像和 10,000 张测试图像，图像尺寸为 28×28 像素。

**参数说明**:

- ``root`` (str): 数据存储根目录
- ``train`` (bool): ``True`` 加载训练集，``False`` 加载测试集
- ``transform`` (callable, optional): 图像变换函数
- ``target_transform`` (callable, optional): 标签变换函数
- ``download`` (bool, optional): 如果为 True，从互联网下载数据集

**使用示例**:

.. code-block:: python

    from riemann.vision.datasets import MNIST
    from riemann.utils.data import DataLoader

    # 加载训练集和测试集
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

EasyMNIST（预处理 MNIST）
^^^^^^^^^^^^^^^^^^^^^^^^^

EasyMNIST 是 MNIST 的预处理版本，在初始化时应用归一化、标准化和展平。标签可以转换为 one-hot 编码。由于在初始化时一次性完成转换，而不是在每个 epoch 中重复处理，这可以节省训练过程中的预处理时间。

**参数说明**:

- ``root`` (str): 数据存储根目录
- ``train`` (bool): ``True`` 加载训练集，``False`` 加载测试集
- ``onehot_label`` (bool): 如果为 True，将标签转换为 one-hot 编码（默认: True）
- ``download`` (bool, optional): 如果为 True，从互联网下载数据集

**使用示例**:

.. code-block:: python

    from riemann.vision.datasets import EasyMNIST

    # 加载 EasyMNIST，使用 one-hot 标签（默认）
    train_dataset = EasyMNIST(root='./data', train=True, onehot_label=True, download=True)
    
    # 加载，使用标量标签
    test_dataset = EasyMNIST(root='./data', train=False, onehot_label=False, download=True)

    # 数据已经预处理（归一化、展平）
    image, label = train_dataset[0]
    print(f"图像形状: {image.shape}")  # [784] - 展平后
    print(f"标签形状: {label.shape}")  # [10] - 如果 onehot_label=True 则为 one-hot

FashionMNIST 数据集
~~~~~~~~~~~~~~~~~~~

Fashion-MNIST 是 Zalando 文章图像的数据集，包含 60,000 个训练示例和 10,000 个测试示例。每个示例都是 28×28 的灰度图像，与 10 个类别之一相关联。它被设计为 MNIST 的直接替代品。

**类别**: T-shirt/top（T恤/上衣）、Trouser（裤子）、Pullover（套衫）、Dress（连衣裙）、Coat（外套）、Sandal（凉鞋）、Shirt（衬衫）、Sneaker（运动鞋）、Bag（包）、Ankle boot（短靴）

**参数说明**:

- ``root`` (str): 数据存储根目录
- ``train`` (bool): ``True`` 加载训练集，``False`` 加载测试集
- ``transform`` (callable, optional): 图像变换函数
- ``target_transform`` (callable, optional): 标签变换函数
- ``download`` (bool, optional): 如果为 True，从互联网下载数据集

**使用示例**:

.. code-block:: python

    from riemann.vision.datasets import FashionMNIST

    # 加载 FashionMNIST 数据集
    train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    print(f"类别: {train_dataset.classes}")

CIFAR-10 数据集
~~~~~~~~~~~~~~~

包含 60,000 张 32×32 彩色图像，分为 10 个类别（飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车）。

**参数说明**:

- ``root`` (str): 数据存储根目录
- ``train`` (bool): ``True`` 加载训练集（50,000 张），``False`` 加载测试集（10,000 张）
- ``transform`` (callable, optional): 图像变换函数
- ``target_transform`` (callable, optional): 标签变换函数
- ``download`` (bool, optional): 如果为 True，从互联网下载数据集

**使用示例**:

.. code-block:: python

    from riemann.vision.datasets import CIFAR10

    # 加载 CIFAR-10 数据集
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

CIFAR-100 数据集
~~~~~~~~~~~~~~~~

包含 60,000 张 32×32 彩色图像，分为 100 个类别。每个类别有 600 张图像（500 张用于训练，100 张用于测试）。CIFAR-100 有 100 个细分类别和 20 个超类。

**参数说明**:

- ``root`` (str): 数据存储根目录
- ``train`` (bool): ``True`` 加载训练集（50,000 张），``False`` 加载测试集（10,000 张）
- ``transform`` (callable, optional): 图像变换函数
- ``target_transform`` (callable, optional): 标签变换函数
- ``download`` (bool, optional): 如果为 True，从互联网下载数据集
- ``coarse`` (bool, optional): 如果为 True，使用 20 个超类标签；否则使用 100 个细分类别标签（默认：False）

**使用示例**:

.. code-block:: python

    from riemann.vision.datasets import CIFAR100

    # 加载 CIFAR-100 使用细分类别标签（100 类）
    train_dataset = CIFAR100(root='./data', train=True, download=True, coarse=False)
    test_dataset = CIFAR100(root='./data', train=False, download=True, coarse=False)

    # 加载 CIFAR-100 使用超类标签（20 类）
    train_dataset_coarse = CIFAR100(root='./data', train=True, download=True, coarse=True)

Flowers102 数据集
~~~~~~~~~~~~~~~~~

Oxford 102 Flower 是一个图像分类数据集，包含 102 个花卉类别。这些花卉选自英国常见的花卉。每个类别包含 40 到 258 张图像。图像具有较大的尺度、姿态和光照变化。

**注意**: 此类需要 ``scipy`` 来从 ``.mat`` 格式加载目标文件。

**参数说明**:

- ``root`` (str): 数据集根目录
- ``split`` (str, optional): 数据集划分，支持 ``"train"``（默认）、``"val"`` 或 ``"test"``
- ``transform`` (callable, optional): 图像变换函数
- ``target_transform`` (callable, optional): 目标变换函数
- ``download`` (bool, optional): 如果为 True，从互联网下载数据集

**数据集统计**:

- 训练集：1,020 张图像
- 验证集：1,020 张图像
- 测试集：6,149 张图像
- 总计：8,189 张图像，102 个类别

**使用示例**:

.. code-block:: python

    from riemann.vision.datasets import Flowers102

    # 加载 Flowers102 数据集
    train_dataset = Flowers102(root='./data', split='train', download=True, transform=transforms.ToTensor())
    val_dataset = Flowers102(root='./data', split='val', download=True, transform=transforms.ToTensor())
    test_dataset = Flowers102(root='./data', split='test', download=True, transform=transforms.ToTensor())

    print(f"训练样本数: {len(train_dataset)}")  # 1020
    print(f"验证样本数: {len(val_dataset)}")  # 1020
    print(f"测试样本数: {len(test_dataset)}")  # 6149

OxfordIIITPet 数据集
~~~~~~~~~~~~~~~~~~~~

Oxford-IIIT Pet 数据集是一个包含 37 个类别的宠物数据集，每个类别约有 200 张图像。图像在尺度、姿态和光照方面有很大变化。所有图像都有相关的真实标注，包括物种（猫或狗）、品种和像素级 trimap 分割。

**参数说明**:

- ``root`` (str): 数据集根目录
- ``split`` (str, optional): 数据集划分，支持 ``"trainval"``（默认）或 ``"test"``
- ``target_types`` (str 或 list, optional): 要使用的目标类型。可以是 ``"category"``（默认）、``"binary-category"`` 或 ``"segmentation"``。也可以是列表，以输出包含所有指定目标类型的元组。
- ``transform`` (callable, optional): 图像变换函数
- ``target_transform`` (callable, optional): 目标变换函数
- ``download`` (bool, optional): 如果为 True，从互联网下载数据集

**目标类型**:

- ``category`` (int): 37 个宠物类别之一的标签
- ``binary-category`` (int): 猫（0）或狗（1）的二元标签
- ``segmentation`` (PIL Image): 图像的分割 trimap

**使用示例**:

.. code-block:: python

    from riemann.vision.datasets import OxfordIIITPet

    # 加载类别标签
    dataset = OxfordIIITPet(root='./data', split='trainval', target_types='category', download=True)
    
    # 加载二元分类（猫 vs 狗）
    dataset_bin = OxfordIIITPet(root='./data', split='trainval', target_types='binary-category', download=True)
    
    # 加载分割掩码
    dataset_seg = OxfordIIITPet(root='./data', split='trainval', target_types='segmentation', download=True)
    
    # 加载多个目标类型
    dataset_multi = OxfordIIITPet(root='./data', split='trainval', 
                                   target_types=['category', 'segmentation'], download=True)

LFWPeople 数据集
~~~~~~~~~~~~~~~~

LFW（Labeled Faces in the Wild）People 数据集包含从网络上收集的 13,233 张人脸图像。图像被组织成 5,749 个不同的身份。该数据集专为面部识别研究而设计。

**参数说明**:

- ``root`` (str): 数据集根目录
- ``split`` (str, optional): 数据集划分，支持 ``"10fold"``（默认）、``"train"`` 或 ``"test"``
- ``image_set`` (str, optional): 图像对齐类型，支持 ``"original"``、``"funneled"``（默认）或 ``"deepfunneled"``
- ``transform`` (callable, optional): 图像变换函数
- ``target_transform`` (callable, optional): 目标变换函数
- ``download`` (bool, optional): 如果为 True，从互联网下载数据集

**图像集**:

- ``original``: 未对齐的原始图像
- ``funneled``: 几何归一化的人脸图像（默认）
- ``deepfunneled``: 深度 funneled 图像，对齐效果更好

**使用示例**:

.. code-block:: python

    from riemann.vision.datasets import LFWPeople

    # 加载 LFWPeople 数据集，使用 funneled 图像
    train_dataset = LFWPeople(root='./data', split='train', image_set='funneled', download=True)
    test_dataset = LFWPeople(root='./data', split='test', image_set='funneled', download=True)

    print(f"类别数（人数）: {len(train_dataset.classes)}")
    print(f"训练样本数: {len(train_dataset)}")

SVHN 数据集
~~~~~~~~~~~

SVHN（Street View House Numbers）数据集包含从 Google 街景收集的门牌号 32×32 彩色图像。数据集包含 10 个数字类别（0-9）。

**注意**: 此类需要 ``scipy`` 来从 ``.mat`` 格式加载数据。

**参数说明**:

- ``root`` (str): 数据集根目录
- ``split`` (str): 数据集划分，支持 ``"train"``、``"test"`` 或 ``"extra"``
- ``transform`` (callable, optional): 图像变换函数
- ``target_transform`` (callable, optional): 目标变换函数
- ``download`` (bool, optional): 如果为 True，从互联网下载数据集

**数据集统计**:

- 训练集：73,257 张图像
- 测试集：26,032 张图像
- 额外集：531,131 张额外图像（较简单的样本）

**使用示例**:

.. code-block:: python

    from riemann.vision.datasets import SVHN

    # 加载 SVHN 数据集
    train_dataset = SVHN(root='./data', split='train', download=True, transform=transforms.ToTensor())
    test_dataset = SVHN(root='./data', split='test', download=True, transform=transforms.ToTensor())
    
    # 额外划分，包含更多训练数据
    extra_dataset = SVHN(root='./data', split='extra', download=True, transform=transforms.ToTensor())

ImageFolder 数据集
~~~~~~~~~~~~~~~~~~

从本地文件夹加载图像数据集，适用于自定义数据集。文件夹结构应按类别组织：

.. code-block:: text

    root/
    ├── class_a/
    │   ├── img1.jpg
    │   └── img2.png
    ├── class_b/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── class_c/
        └── img1.jpg

**参数说明**:

- ``root`` (str): 数据集根目录路径
- ``transform`` (callable, optional): 图像变换函数
- ``target_transform`` (callable, optional): 标签变换函数
- ``loader`` (callable, optional): 图像加载函数，默认为 PIL Image 加载
- ``is_valid_file`` (callable, optional): 验证文件是否有效的函数

**使用示例**:

.. code-block:: python

    from riemann.vision.datasets import ImageFolder

    # 从文件夹加载自定义数据集
    dataset = ImageFolder(
        root='./custom_dataset',
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    )

    print(f"类别数: {len(dataset.classes)}")  # ['class_a', 'class_b', 'class_c']
    print(f"类别到索引映射: {dataset.class_to_idx}")  # {'class_a': 0, 'class_b': 1, 'class_c': 2}

DatasetFolder 数据集
~~~~~~~~~~~~~~~~~~~~

通用的文件夹数据集类，与 ``ImageFolder`` 类似，但允许自定义图像加载器。

**参数说明**:

- ``root`` (str): 数据集根目录路径
- ``loader`` (callable): 图像加载函数
- ``extensions`` (tuple, optional): 允许的文件扩展名元组
- ``transform`` (callable, optional): 图像变换函数
- ``target_transform`` (callable, optional): 标签变换函数
- ``is_valid_file`` (callable, optional): 验证文件是否有效的函数
- ``allow_empty`` (bool): 是否允许空文件夹，默认 ``False``

**使用示例**:

.. code-block:: python

    from riemann.vision.datasets import DatasetFolder, default_loader

    # 使用自定义加载器
    dataset = DatasetFolder(
        root='./custom_dataset',
        loader=default_loader,
        extensions=('.jpg', '.png'),
        transform=transforms.ToTensor()
    )

default_loader
~~~~~~~~~~~~~~

``default_loader`` 是 ImageFolder 和 DatasetFolder 使用的默认图像加载函数。它会根据文件扩展名自动选择合适的加载方式：

- PIL 可以处理的图像格式（如 .jpg, .png, .bmp 等）：使用 PIL.Image.open() 加载并转换为 RGB 模式
- 其他格式：尝试使用 PIL 加载

**用途说明**:

``default_loader`` 主要用于 ``ImageFolder`` 和 ``DatasetFolder`` 的 ``loader`` 参数，用于指定加载图像的方法。当使用这两个数据集类时，如果不指定 ``loader`` 参数，默认就会使用 ``default_loader``。

**使用示例**:

.. code-block:: python

    from riemann.vision.datasets import DatasetFolder, default_loader

    # 使用 default_loader 加载图像
    image = default_loader('path/to/image.jpg')
    
    # 在 DatasetFolder 中使用
    dataset = DatasetFolder(
        root='./custom_dataset',
        loader=default_loader,  # 指定使用 default_loader
        extensions=('.jpg', '.png')
    )

图像变换 (Transforms)
---------------------

``riemann.vision.transforms`` 提供了丰富的图像变换操作，用于数据预处理和数据增强。

变换概览
~~~~~~~~

.. list-table:: 支持的图像变换
   :header-rows: 1
   :widths: 25 35 40

   * - 变换类
     - 说明
     - 类别
   * - Compose
     - 将多个变换组合成一个
     - 工具类
   * - PILToTensor
     - 将 PIL Image 转换为张量（不缩放）
     - 类型转换
   * - ToTensor
     - 将 PIL Image 或 numpy.ndarray 转换为张量（缩放到 [0, 1]）
     - 类型转换
   * - ToPILImage
     - 将张量转换为 PIL Image
     - 类型转换
   * - ConvertImageDtype
     - 将图像转换为指定数据类型
     - 类型转换
   * - Normalize
     - 使用均值和标准差对张量进行标准化
     - 标准化
   * - Resize
     - 调整图像大小到指定尺寸
     - 几何变换
   * - CenterCrop
     - 从图像中心裁剪
     - 几何变换
   * - RandomHorizontalFlip
     - 随机水平翻转图像
     - 数据增强
   * - RandomVerticalFlip
     - 随机垂直翻转图像
     - 数据增强
   * - RandomRotation
     - 随机旋转图像
     - 数据增强
   * - ColorJitter
     - 随机调整亮度、对比度、饱和度、色调
     - 数据增强
   * - Grayscale
     - 将图像转换为灰度图像
     - 颜色变换
   * - RandomGrayscale
     - 随机将图像转换为灰度图像
     - 数据增强
   * - RandomCrop
     - 随机裁剪图像到指定尺寸
     - 数据增强
   * - RandomResizedCrop
     - 随机裁剪并调整图像大小
     - 数据增强
   * - FiveCrop
     - 将图像裁剪为 5 个区域（四角 + 中心）
     - 几何变换
   * - TenCrop
     - 将图像裁剪为 10 个区域（五裁剪 + 水平翻转）
     - 几何变换
   * - Pad
     - 使用指定值填充图像
     - 几何变换
   * - Lambda
     - 应用自定义 lambda 函数
     - 工具类
   * - GaussianBlur
     - 对图像应用高斯模糊
     - 滤波器
   * - RandomAffine
     - 随机仿射变换
     - 数据增强
   * - RandomPerspective
     - 随机透视变换
     - 数据增强
   * - RandomErasing
     - 随机擦除矩形区域
     - 数据增强
   * - AutoAugment
     - AutoAugment 数据增强策略
     - 自动增强
   * - RandAugment
     - RandAugment 数据增强策略
     - 自动增强
   * - TrivialAugmentWide
     - TrivialAugmentWide 数据增强策略
     - 自动增强
   * - SanitizeBoundingBox
     - 清理和验证边界框
     - 目标检测
   * - Invert
     - 反转图像颜色
     - 颜色变换
   * - Posterize
     - 减少每个颜色通道的位数
     - 颜色变换
   * - Solarize
     - 反转高于阈值的像素
     - 颜色变换
   * - Equalize
     - 均衡化图像直方图
     - 颜色变换
   * - AutoContrast
     - 最大化图像对比度
     - 颜色变换
   * - Sharpness
     - 调整图像锐度
     - 颜色变换
   * - Brightness
     - 调整图像亮度
     - 颜色变换
   * - Contrast
     - 调整图像对比度
     - 颜色变换
   * - Saturation
     - 调整图像饱和度
     - 颜色变换
   * - Hue
     - 调整图像色调
     - 颜色变换

Compose
~~~~~~~

将多个变换组合在一起，按顺序应用。

**参数说明**:

- ``transforms`` (list): 要组合的变换对象列表

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 定义变换流程
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

PILToTensor
~~~~~~~~~~~

将 PIL Image 转换为张量，不进行缩放。与 ToTensor 不同，PILToTensor 不会将值从 [0, 255] 缩放到 [0.0, 1.0]。

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 将 PIL Image 转换为张量（值范围 [0, 255]）
    pil_to_tensor = transforms.PILToTensor()
    tensor_img = pil_to_tensor(pil_image)

ToTensor
~~~~~~~~

将 PIL Image 或 numpy.ndarray 转换为张量。将值从 [0, 255] 缩放到 [0.0, 1.0]。

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 将 PIL Image 转换为张量（值范围 [0, 1]）
    to_tensor = transforms.ToTensor()
    tensor_img = to_tensor(pil_image)

ToPILImage
~~~~~~~~~~

将张量转换为 PIL Image。

**参数说明**:

- ``mode`` (str, optional): 输出图像的颜色模式

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 将张量转换为 PIL Image
    to_pil = transforms.ToPILImage()
    pil_img = to_pil(tensor)

ConvertImageDtype
~~~~~~~~~~~~~~~~~

将图像转换为指定数据类型。

**参数说明**:

- ``dtype`` (dtype): 目标数据类型

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 转换为 float32
    convert_dtype = transforms.ConvertImageDtype(dtype='float32')
    converted_img = convert_dtype(img)

Normalize
~~~~~~~~~

使用均值和标准差对张量进行标准化。

**参数说明**:

- ``mean`` (sequence): 每个通道的均值
- ``std`` (sequence): 每个通道的标准差

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 使用 ImageNet 统计数据进行标准化
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    normalized_img = normalize(tensor_img)

Resize
~~~~~~

调整图像大小到指定尺寸。

**参数说明**:

- ``size`` (int or tuple): 目标尺寸。如果是 int，短边调整为该尺寸；如果是 tuple，(高度, 宽度)。

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 调整到特定尺寸
    resize = transforms.Resize((224, 224))
    resized_img = resize(pil_image)

    # 按短边调整
    resize = transforms.Resize(256)
    resized_img = resize(pil_image)

CenterCrop
~~~~~~~~~~

从图像中心裁剪。

**参数说明**:

- ``size`` (int or tuple): 裁剪尺寸

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 中心裁剪为 224x224
    center_crop = transforms.CenterCrop(224)
    cropped_img = center_crop(pil_image)

RandomHorizontalFlip
~~~~~~~~~~~~~~~~~~~~

随机水平翻转图像。

**参数说明**:

- ``p`` (float): 翻转概率（默认：0.5）

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 以 50% 概率翻转
    hflip = transforms.RandomHorizontalFlip(p=0.5)
    flipped_img = hflip(pil_image)

RandomVerticalFlip
~~~~~~~~~~~~~~~~~~

随机垂直翻转图像。

**参数说明**:

- ``p`` (float): 翻转概率（默认：0.5）

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 以 50% 概率翻转
    vflip = transforms.RandomVerticalFlip(p=0.5)
    flipped_img = vflip(pil_image)

RandomRotation
~~~~~~~~~~~~~~

随机旋转图像。

**参数说明**:

- ``degrees`` (sequence or float): 旋转角度范围 (-degrees, +degrees)

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 在 -15 到 15 度之间随机旋转
    rotation = transforms.RandomRotation(degrees=15)
    rotated_img = rotation(pil_image)

ColorJitter
~~~~~~~~~~~

随机调整亮度、对比度、饱和度和色调。

**参数说明**:

- ``brightness`` (float): 亮度抖动因子
- ``contrast`` (float): 对比度抖动因子
- ``saturation`` (float): 饱和度抖动因子
- ``hue`` (float): 色调抖动因子

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 随机调整颜色
    jitter = transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    )
    jittered_img = jitter(pil_image)

Grayscale
~~~~~~~~~

将图像转换为灰度图像。

**参数说明**:

- ``num_output_channels`` (int): 输出通道数（1 或 3）

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 转换为灰度图像（1 通道）
    gray = transforms.Grayscale(num_output_channels=1)
    gray_img = gray(pil_image)

RandomGrayscale
~~~~~~~~~~~~~~~

随机将图像转换为灰度图像。

**参数说明**:

- ``p`` (float): 转换概率（默认：0.1）

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 以 10% 概率转换为灰度图像
    gray = transforms.RandomGrayscale(p=0.1)
    gray_img = gray(pil_image)

RandomCrop
~~~~~~~~~~

随机裁剪图像到指定尺寸。

**参数说明**:

- ``size`` (int or tuple): 裁剪尺寸
- ``padding`` (int, optional): 填充尺寸

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 带填充的随机裁剪
    crop = transforms.RandomCrop(224, padding=4)
    cropped_img = crop(pil_image)

RandomResizedCrop
~~~~~~~~~~~~~~~~~

随机裁剪并调整图像大小。

**参数说明**:

- ``size`` (int or tuple): 目标尺寸
- ``scale`` (tuple): 裁剪的缩放范围
- ``ratio`` (tuple): 宽高比范围

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 随机裁剪并调整大小
    crop = transforms.RandomResizedCrop(224, scale=(0.08, 1.0))
    cropped_img = crop(pil_image)

FiveCrop
~~~~~~~~

将图像裁剪为 5 个区域（四角 + 中心）。

**参数说明**:

- ``size`` (int or tuple): 裁剪尺寸

**使用示例**:

.. code-block:: python

    import riemann as rm
    from riemann.vision import transforms

    # 五裁剪
    five_crop = transforms.FiveCrop(224)
    crops = five_crop(pil_image)  # 返回 5 个图像的元组
    
    # 堆叠成批次
    tensor_crops = rm.stack([transforms.ToTensor()(crop) for crop in crops])

TenCrop
~~~~~~~

将图像裁剪为 10 个区域（五裁剪 + 水平翻转）。

**参数说明**:

- ``size`` (int or tuple): 裁剪尺寸
- ``vertical_flip`` (bool): 是否也应用垂直翻转

**使用示例**:

.. code-block:: python

    import riemann as rm
    from riemann.vision import transforms

    # 十裁剪
    ten_crop = transforms.TenCrop(224)
    crops = ten_crop(pil_image)  # 返回 10 个图像的元组

Pad
~~~

使用指定值填充图像。

**参数说明**:

- ``padding`` (int or tuple): 填充尺寸
- ``fill`` (int or tuple): 填充值

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 填充图像
    pad = transforms.Pad(padding=4, fill=0)
    padded_img = pad(pil_image)

Lambda
~~~~~~

应用自定义 lambda 函数。

**参数说明**:

- ``lambd`` (function): 要应用的 lambda 函数

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 自定义 lambda 变换
    lambd = transforms.Lambda(lambda x: x.rotate(45))
    transformed_img = lambd(pil_image)

GaussianBlur
~~~~~~~~~~~~

对图像应用高斯模糊。

**参数说明**:

- ``kernel_size`` (int): 高斯核大小
- ``sigma`` (float or tuple): 标准差

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 应用高斯模糊
    blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    blurred_img = blur(pil_image)

RandomAffine
~~~~~~~~~~~~

随机仿射变换。

**参数说明**:

- ``degrees`` (float or tuple): 旋转角度
- ``translate`` (tuple): 平移范围
- ``scale`` (tuple): 缩放范围
- ``shear`` (float or tuple): 剪切范围

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 随机仿射变换
    affine = transforms.RandomAffine(
        degrees=15,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    )
    transformed_img = affine(pil_image)

RandomPerspective
~~~~~~~~~~~~~~~~~

随机透视变换。

**参数说明**:

- ``distortion_scale`` (float): 扭曲程度
- ``p`` (float): 应用变换的概率

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 随机透视
    perspective = transforms.RandomPerspective(distortion_scale=0.5, p=0.5)
    transformed_img = perspective(pil_image)

RandomErasing
~~~~~~~~~~~~~

随机擦除矩形区域。

**参数说明**:

- ``p`` (float): 应用的概率
- ``scale`` (tuple): 擦除区域范围
- ``ratio`` (tuple): 宽高比范围
- ``value`` (str or float): 擦除值

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 随机擦除（通常用于张量）
    erasing = transforms.RandomErasing(p=0.5, scale=(0.02, 0.33))
    erased_tensor = erasing(tensor_img)

AutoAugment
~~~~~~~~~~~

AutoAugment 数据增强策略。

**参数说明**:

- ``policy`` (str): 使用的策略（'imagenet', 'cifar10', 'svhn'）

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 使用 ImageNet 策略的 AutoAugment
    auto_augment = transforms.AutoAugment(policy='imagenet')
    augmented_img = auto_augment(pil_image)

RandAugment
~~~~~~~~~~~

RandAugment 数据增强策略。

**参数说明**:

- ``num_ops`` (int): 操作数量
- ``magnitude`` (int): 操作强度

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # RandAugment
    rand_augment = transforms.RandAugment(num_ops=2, magnitude=9)
    augmented_img = rand_augment(pil_image)

TrivialAugmentWide
~~~~~~~~~~~~~~~~~~

TrivialAugmentWide 数据增强策略。

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # TrivialAugmentWide
    trivial_augment = transforms.TrivialAugmentWide()
    augmented_img = trivial_augment(pil_image)

SanitizeBoundingBox
~~~~~~~~~~~~~~~~~~~

清理和验证边界框。

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 清理边界框
    sanitize = transforms.SanitizeBoundingBox()
    sanitized_boxes = sanitize(boxes, image_size)

Invert
~~~~~~

反转图像颜色。

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 反转图像
    invert = transforms.Invert()
    inverted_img = invert(pil_image)

Posterize
~~~~~~~~~

减少每个颜色通道的位数。

**参数说明**:

- ``bits`` (int): 保留的位数

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 色调分离
    posterize = transforms.Posterize(bits=4)
    posterized_img = posterize(pil_image)

Solarize
~~~~~~~~

反转高于阈值的像素。

**参数说明**:

- ``threshold`` (int): 阈值

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 曝光
    solarize = transforms.Solarize(threshold=128)
    solarized_img = solarize(pil_image)

Equalize
~~~~~~~~

均衡化图像直方图。

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 均衡化图像
    equalize = transforms.Equalize()
    equalized_img = equalize(pil_image)

AutoContrast
~~~~~~~~~~~~

最大化图像对比度。

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 自动对比度
    auto_contrast = transforms.AutoContrast()
    contrasted_img = auto_contrast(pil_image)

Sharpness
~~~~~~~~~

调整图像锐度。

**参数说明**:

- ``sharpness_factor`` (float): 锐度因子

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 调整锐度
    sharpness = transforms.Sharpness(sharpness_factor=2.0)
    sharpened_img = sharpness(pil_image)

Brightness
~~~~~~~~~~

调整图像亮度。

**参数说明**:

- ``brightness_factor`` (float): 亮度因子

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 调整亮度
    brightness = transforms.Brightness(brightness_factor=1.5)
    brightened_img = brightness(pil_image)

Contrast
~~~~~~~~

调整图像对比度。

**参数说明**:

- ``contrast_factor`` (float): 对比度因子

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 调整对比度
    contrast = transforms.Contrast(contrast_factor=1.5)
    contrasted_img = contrast(pil_image)

Saturation
~~~~~~~~~~

调整图像饱和度。

**参数说明**:

- ``saturation_factor`` (float): 饱和度因子

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 调整饱和度
    saturation = transforms.Saturation(saturation_factor=1.5)
    saturated_img = saturation(pil_image)

Hue
~~~

调整图像色调。

**参数说明**:

- ``hue_factor`` (float): 色调因子（-0.5 到 0.5）

**使用示例**:

.. code-block:: python

    from riemann.vision import transforms

    # 调整色调
    hue = transforms.Hue(hue_factor=0.1)
    hue_adjusted_img = hue(pil_image)

完整示例
--------

以下示例展示了如何使用 Riemann 的计算机视觉模块进行常见的深度学习任务。

图像分类完整训练流程
~~~~~~~~~~~~~~~~~~~~

本示例演示了使用 CIFAR-10 数据集进行图像分类的完整流程，包括数据加载、数据增强、模型定义、训练和评估。

**流程说明**:

1. **数据预处理**: 使用随机裁剪、水平翻转和颜色抖动进行数据增强
2. **标准化**: 使用 ImageNet 统计数据进行标准化
3. **模型定义**: 简单的卷积神经网络
4. **训练循环**: 标准的训练流程，包括前向传播、损失计算、反向传播和参数更新

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    import riemann.optim as optim
    from riemann.vision import datasets, transforms
    from riemann.utils.data import DataLoader

    # 定义训练数据变换（包含数据增强）
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),      # 随机裁剪并调整大小
        transforms.RandomHorizontalFlip(),       # 随机水平翻转
        transforms.ColorJitter(                  # 颜色抖动（数据增强）
            brightness=0.2, 
            contrast=0.2
        ),
        transforms.ToTensor(),                   # 转换为张量
        transforms.Normalize(                    # 标准化
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 定义测试数据变换（不包含数据增强）
    test_transform = transforms.Compose([
        transforms.Resize(256),                  # 调整大小
        transforms.CenterCrop(224),              # 中心裁剪
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 加载 CIFAR-10 数据集
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,           # 训练时打乱数据
        num_workers=4           # 使用4个子进程加载数据
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False
    )

    # 定义卷积神经网络模型
    model = nn.Sequential(
        # 第一个卷积块
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # 第二个卷积块
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # 全连接层
        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 10)  # CIFAR-10 有10个类别
    )

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=0.01, 
        momentum=0.9
    )

    # 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()   # 清空梯度
            loss.backward()         # 计算梯度
            optimizer.step()        # 更新参数
            
            running_loss += loss.item()
            
            # 每100个批次打印一次进度
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/100:.4f}')
                running_loss = 0.0
        
        print(f'Epoch {epoch+1} 完成')

    print('训练完成！')

使用 ImageFolder 加载自定义数据集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

当你有自己的图像数据集时，可以使用 ``ImageFolder`` 方便地加载。只需要将图像按文件夹组织，每个文件夹代表一个类别。

**文件夹结构要求**:

.. code-block:: text

    custom_dataset/
    ├── class_a/           # 类别 A 的图像
    │   ├── img1.jpg
    │   └── img2.png
    ├── class_b/           # 类别 B 的图像
    │   ├── img1.jpg
    │   └── img2.jpg
    └── class_c/           # 类别 C 的图像
        └── img1.jpg

**加载示例**:

.. code-block:: python

    from riemann.vision import datasets, transforms
    from riemann.utils.data import DataLoader

    # 定义数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 使用 ImageFolder 加载数据集
    dataset = datasets.ImageFolder(
        root='./custom_dataset',
        transform=transform
    )

    # 查看数据集信息
    print(f"类别数: {len(dataset.classes)}")
    print(f"类别名称: {dataset.classes}")
    print(f"类别到索引映射: {dataset.class_to_idx}")
    print(f"样本总数: {len(dataset)}")

    # 创建数据加载器
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 遍历数据
    for images, labels in loader:
        print(f"图像批次形状: {images.shape}")  # [32, 3, 224, 224]
        print(f"标签批次形状: {labels.shape}")  # [32]
        break

创建自定义数据集类
~~~~~~~~~~~~~~~~~~

当 ``ImageFolder`` 无法满足需求时，你可以继承 ``Dataset`` 类创建自定义数据集。以下示例展示了如何创建一个从文件夹加载图像的自定义数据集。

**适用场景**:

- 需要自定义文件组织方式
- 需要从其他数据源（如数据库、网络）加载数据
- 需要进行复杂的预处理

.. code-block:: python

    from riemann.utils.data import Dataset
    from PIL import Image
    import os

    class CustomImageDataset(Dataset):
        """
        自定义图像数据集类
        
        从文件夹加载图像，文件夹结构为：
        root/
            label1/
                image1.jpg
                image2.jpg
            label2/
                image1.jpg
        """
        
        def __init__(self, root_dir, transform=None):
            """
            参数:
                root_dir (str): 数据集根目录
                transform (callable, optional): 图像变换函数
            """
            self.root_dir = root_dir
            self.transform = transform
            self.images = []
            self.labels = []
            
            # 扫描文件夹，收集所有图像路径和标签
            for label in sorted(os.listdir(root_dir)):
                label_dir = os.path.join(root_dir, label)
                if os.path.isdir(label_dir):
                    for img_name in os.listdir(label_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                            self.images.append(os.path.join(label_dir, img_name))
                            self.labels.append(int(label))
            
            print(f"加载了 {len(self.images)} 张图像，共 {len(set(self.labels))} 个类别")

        def __len__(self):
            """返回数据集大小"""
            return len(self.images)

        def __getitem__(self, idx):
            """
            获取指定索引的样本
            
            参数:
                idx (int): 样本索引
                
            返回:
                tuple: (图像, 标签)
            """
            # 加载图像
            img_path = self.images[idx]
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            
            return image, label

    # 使用自定义数据集
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = CustomImageDataset(
        root_dir='./custom_data',
        transform=transform
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=2
    )

    # 测试数据加载
    for images, labels in loader:
        print(f"批次图像形状: {images.shape}")
        print(f"批次标签: {labels}")
        break
