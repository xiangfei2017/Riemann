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

图像变换 (Transforms)
---------------------

``riemann.vision.transforms`` 提供了丰富的图像变换操作，用于数据预处理和数据增强。

变换组合 (Compose)
~~~~~~~~~~~~~~~~~~

将多个变换组合在一起，按顺序应用。

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

基本变换
~~~~~~~~

**ToTensor**: 将 PIL Image 或 numpy.ndarray 转换为张量

.. code-block:: python

    # PIL Image -> Tensor (值范围 [0, 1])
    tensor = transforms.ToTensor()(pil_image)

**ToPILImage**: 将张量转换为 PIL Image

.. code-block:: python

    # Tensor -> PIL Image
    pil_image = transforms.ToPILImage()(tensor)

**Resize**: 调整图像大小

.. code-block:: python

    # 调整为指定尺寸
    resize = transforms.Resize((224, 224))
    
    # 按短边等比例调整
    resize = transforms.Resize(256)

**CenterCrop**: 中心裁剪

.. code-block:: python

    # 从图像中心裁剪指定尺寸
    crop = transforms.CenterCrop(224)

**Normalize**: 标准化

.. code-block:: python

    # 使用均值和标准差进行标准化
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

数据增强变换
~~~~~~~~~~~~

**RandomResizedCrop**: 随机缩放裁剪

.. code-block:: python

    # 随机裁剪并缩放到指定尺寸
    crop = transforms.RandomResizedCrop(224, scale=(0.08, 1.0))

**RandomHorizontalFlip**: 随机水平翻转

.. code-block:: python

    # 以 50% 概率水平翻转
    flip = transforms.RandomHorizontalFlip(p=0.5)

**RandomVerticalFlip**: 随机垂直翻转

.. code-block:: python

    # 以 50% 概率垂直翻转
    flip = transforms.RandomVerticalFlip(p=0.5)

**RandomRotation**: 随机旋转

.. code-block:: python

    # 随机旋转 (-15, 15) 度
    rotation = transforms.RandomRotation(degrees=15)

**ColorJitter**: 颜色抖动

.. code-block:: python

    # 随机调整亮度、对比度、饱和度和色调
    jitter = transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    )

**RandomCrop**: 随机裁剪

.. code-block:: python

    # 随机裁剪到指定尺寸
    crop = transforms.RandomCrop(224, padding=4)

**RandomGrayscale**: 随机灰度化

.. code-block:: python

    # 以 10% 概率转换为灰度图像
    gray = transforms.RandomGrayscale(p=0.1)

高级变换
~~~~~~~~

**FiveCrop**: 五裁剪（四角和中心）

.. code-block:: python

    import riemann as rm
    from riemann.vision import transforms

    # 对图像进行五裁剪
    five_crop = transforms.FiveCrop(224)
    image = rm.randn(3, 256, 256)  # [C, H, W]
    crops = five_crop(image)  # 返回5个张量的元组

**TenCrop**: 十裁剪（五裁剪 + 水平翻转）

.. code-block:: python

    # 对图像进行十裁剪
    ten_crop = transforms.TenCrop(224, vertical_flip=False)
    crops = ten_crop(image)  # 返回10个张量的元组

**Pad**: 图像填充

.. code-block:: python

    # 对图像进行填充
    pad = transforms.Pad(padding=4, fill=0)

AutoAugment 系列
~~~~~~~~~~~~~~~~

**AutoAugment**: 自动数据增强

.. code-block:: python

    from riemann.vision.transforms import AutoAugment, AutoAugmentPolicy

    # 使用 ImageNet 策略
    augment = AutoAugment(policy=AutoAugmentPolicy.IMAGENET)
    augmented_image = augment(image)

**RandAugment**: 随机数据增强

.. code-block:: python

    from riemann.vision.transforms import RandAugment

    # 随机选择变换组合
    augment = RandAugment(num_ops=2, magnitude=9)
    augmented_image = augment(image)

**TrivialAugmentWide**: 宽范围简单增强

.. code-block:: python

    from riemann.vision.transforms import TrivialAugmentWide

    # 宽范围简单增强
    augment = TrivialAugmentWide()
    augmented_image = augment(image)

实用工具
--------

**default_loader**: 默认图像加载器

.. code-block:: python

    from riemann.vision.datasets import default_loader

    # 加载图像文件
    image = default_loader('path/to/image.jpg')

完整示例
--------

图像分类训练流程
~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    import riemann.optim as optim
    from riemann.vision import datasets, transforms
    from riemann.utils.data import DataLoader

    # 定义数据变换
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        transform=test_transform
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 定义模型
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 10)
    )

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 训练循环
    for epoch in range(10):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

自定义数据集
~~~~~~~~~~~~

.. code-block:: python

    from riemann.utils.data import Dataset
    from PIL import Image
    import os

    class CustomImageDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.images = []
            self.labels = []
            
            # 加载数据列表
            for label in os.listdir(root_dir):
                label_dir = os.path.join(root_dir, label)
                if os.path.isdir(label_dir):
                    for img_name in os.listdir(label_dir):
                        self.images.append(os.path.join(label_dir, img_name))
                        self.labels.append(int(label))

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_path = self.images[idx]
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label

    # 使用自定义数据集
    dataset = CustomImageDataset(
        root_dir='./custom_data',
        transform=transforms.ToTensor()
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

API 参考
--------

数据集类
~~~~~~~~

.. list-table:: 数据集类
   :header-rows: 1
   :widths: 25 75

   * - 类名
     - 说明
   * - ``MNIST``
     - MNIST 手写数字数据集（包含 EasyMNIST 预处理版本）
   * - ``FashionMNIST``
     - Fashion-MNIST 时尚产品数据集
   * - ``CIFAR10``
     - CIFAR-10 图像分类数据集
   * - ``Flowers102``
     - Oxford 102 Flower 花卉分类数据集
   * - ``OxfordIIITPet``
     - Oxford-IIIT Pet 宠物分类数据集
   * - ``LFWPeople``
     - LFW People 人脸识别数据集
   * - ``SVHN``
     - Street View House Numbers 街景门牌号数据集
   * - ``ImageFolder``
     - 从文件夹加载图像数据集
   * - ``DatasetFolder``
     - 通用文件夹数据集基类

变换类
~~~~~~

.. list-table:: 变换类
   :header-rows: 1
   :widths: 25 75

   * - 类名
     - 说明
   * - **组合变换**
     -
   * - ``Compose``
     - 组合多个变换，按顺序应用
   * - **类型转换**
     -
   * - ``ToTensor``
     - 将 PIL Image 或 numpy.ndarray 转换为张量
   * - ``PILToTensor``
     - 将 PIL Image 转换为张量（不缩放）
   * - ``ToPILImage``
     - 将张量转换为 PIL Image
   * - ``ConvertImageDtype``
     - 转换图像数据类型
   * - **几何变换**
     -
   * - ``Resize``
     - 调整图像大小
   * - ``CenterCrop``
     - 中心裁剪
   * - ``RandomCrop``
     - 随机裁剪
   * - ``RandomResizedCrop``
     - 随机缩放裁剪
   * - ``FiveCrop``
     - 五裁剪（四角和中心）
   * - ``TenCrop``
     - 十裁剪（五裁剪 + 水平翻转）
   * - ``Pad``
     - 图像填充
   * - **翻转与旋转**
     -
   * - ``RandomHorizontalFlip``
     - 随机水平翻转
   * - ``RandomVerticalFlip``
     - 随机垂直翻转
   * - ``RandomRotation``
     - 随机旋转
   * - ``RandomAffine``
     - 随机仿射变换
   * - ``RandomPerspective``
     - 随机透视变换
   * - **颜色变换**
     -
   * - ``ColorJitter``
     - 颜色抖动（亮度、对比度、饱和度、色调）
   * - ``Grayscale``
     - 转换为灰度图像
   * - ``RandomGrayscale``
     - 随机转换为灰度图像
   * - ``Invert``
     - 颜色反转
   * - ``Posterize``
     - 减少颜色位数
   * - ``Solarize``
     - 反转高于阈值的像素
   * - ``Equalize``
     - 直方图均衡化
   * - ``AutoContrast``
     - 自动对比度调整
   * - ``Sharpness``
     - 锐度调整
   * - ``Brightness``
     - 亮度调整
   * - ``Contrast``
     - 对比度调整
   * - ``Saturation``
     - 饱和度调整
   * - ``Hue``
     - 色调调整
   * - **标准化与归一化**
     -
   * - ``Normalize``
     - 使用均值和标准差进行标准化
   * - **数据增强（高级）**
     -
   * - ``AutoAugment``
     - 自动数据增强（基于学习策略）
   * - ``RandAugment``
     - 随机数据增强
   * - ``TrivialAugmentWide``
     - 宽范围简单增强
   * - **其他变换**
     -
   * - ``Lambda``
     - 应用自定义 lambda 函数
   * - ``GaussianBlur``
     - 高斯模糊
   * - ``RandomErasing``
     - 随机擦除（用于数据增强）
   * - ``SanitizeBoundingBox``
     - 边界框清理
