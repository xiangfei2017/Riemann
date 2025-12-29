计算机视觉
==========

Riemann 通过 ``riemann.vision`` 包为计算机视觉任务提供了全面的工具。这包括数据集、变换和图像处理实用程序。

数据集
~~~~~~

Riemann 包含几个流行的计算机视觉数据集。

MNIST
~~~~~

.. code-block:: python

    import riemann as rm
    from riemann.vision.datasets import MNIST
    from riemann.utils.data import DataLoader
    
    # 加载 MNIST 数据集
    train_dataset = MNIST(root='./data', train=True, transform=None)
    test_dataset = MNIST(root='./data', train=False, transform=None)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 遍历数据
    for images, labels in train_loader:
        print(images.shape)  # [64, 1, 28, 28]
        print(labels.shape)  # [64]
        break

CIFAR-10
~~~~~~~~

.. code-block:: python

    import riemann as rm
    from riemann.vision.datasets import CIFAR10
    from riemann.utils.data import DataLoader
    
    # 加载 CIFAR-10 数据集
    train_dataset = CIFAR10(root='./data', train=True, transform=None)
    test_dataset = CIFAR10(root='./data', train=False, transform=None)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 遍历数据
    for images, labels in train_loader:
        print(images.shape)  # [64, 3, 32, 32]
        print(labels.shape)  # [64]
        break



变换
-----

变换用于预处理和增强图像数据。

基本变换
~~~~~~~~

.. code-block:: python

    import riemann as rm
    from riemann.vision import transforms
    
    # 定义变换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 将变换应用于数据集
    dataset = ImageFolder(root='./custom_dataset', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

用于数据增强的随机变换
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    from riemann.vision import transforms
    
    # 定义增强变换
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 将变换应用于数据集
    dataset = CIFAR10(root='./data', train=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

自定义变换
~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    from riemann.vision import transforms
    import numpy as np
    
    class CustomTransform:
        def __init__(self, factor=1.0):
            self.factor = factor
        
        def __call__(self, img):
            # 转换为 numpy 数组
            img_array = np.array(img, dtype=np.float32)
            # 应用自定义变换
            img_array = img_array * self.factor
            # 转换回 PIL 图像
            return transforms.ToPILImage()(img_array)
    
    # 定义包含自定义变换的变换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        CustomTransform(factor=1.2),
        transforms.ToTensor(),
    ])
    
    # 将变换应用于数据集
    dataset = CIFAR10(root='./data', train=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

图像处理
~~~~~~~~

Riemann 提供了各种图像处理实用程序。

图像转换
~~~~~~~~

.. code-block:: python

    import riemann as rm
    from riemann.vision import transforms
    
    # 将 PIL 图像转换为张量
    pil_img = ...  # PIL 图像
    tensor_img = transforms.ToTensor()(pil_img)
    
    # 将张量转换为 PIL 图像
    pil_img = transforms.ToPILImage()(tensor_img)



示例
----

使用数据增强的图像分类
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    import riemann.optim as optim
    from riemann.vision import datasets, transforms
    from riemann.utils.data import DataLoader
    
    # 定义训练和测试的变换
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=test_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 创建一个简单的 CNN 模型
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super(SimpleCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 28 * 28, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(512, num_classes),
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), 256 * 28 * 28)
            x = self.classifier(x)
            return x
    
    model = SimpleCNN(num_classes=10)  # CIFAR-10 有 10 个类别
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # 训练循环
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch {epoch}, Loss: {running_loss/len(train_loader):.4f}')
    
    # 评估
    model.eval()
    correct = 0
    total = 0
    with rm.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = rm.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy: {100 * correct / total:.2f}%')

图像分类的自定义数据集
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    from riemann.utils.data import Dataset, DataLoader
    from riemann.vision import transforms
    import os
    from PIL import Image
    
    class CustomImageDataset(Dataset):
        def __init__(self, csv_file, root_dir, transform=None):
            self.annotations = pd.read_csv(csv_file)
            self.root_dir = root_dir
            self.transform = transform
        
        def __len__(self):
            return len(self.annotations)
        
        def __getitem__(self, idx):
            img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
            image = Image.open(img_path).convert("RGB")
            y_label = rm.tensor(self.annotations.iloc[idx, 1])
            
            if self.transform:
                image = self.transform(image)
            
            return (image, y_label)
    
    # 定义变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # 创建数据集和数据加载器
    dataset = CustomImageDataset(csv_file='labels.csv', root_dir='images/', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 在训练循环中使用
    for images, labels in dataloader:
        print(images.shape)  # [32, 3, 224, 224]
        print(labels.shape)  # [32]
        break