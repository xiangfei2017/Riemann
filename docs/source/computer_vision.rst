Computer Vision
===============

Riemann provides comprehensive tools for computer vision tasks through the ``riemann.vision`` package. This includes datasets, transforms, and utilities for image processing.

Datasets
--------

Riemann includes several popular computer vision datasets.

MNIST
~~~~~

.. code-block:: python

    import riemann as rm
    from riemann.vision.datasets import MNIST
    from riemann.utils.data import DataLoader
    
    # Load MNIST dataset
    train_dataset = MNIST(root='./data', train=True, transform=None)
    test_dataset = MNIST(root='./data', train=False, transform=None)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Iterate through data
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
    
    # Load CIFAR-10 dataset
    train_dataset = CIFAR10(root='./data', train=True, transform=None)
    test_dataset = CIFAR10(root='./data', train=False, transform=None)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Iterate through data
    for images, labels in train_loader:
        print(images.shape)  # [64, 3, 32, 32]
        print(labels.shape)  # [64]
        break



Transforms
----------

Transforms are used to preprocess and augment image data.

Basic Transforms
~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    from riemann.vision import transforms
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms to dataset
    dataset = ImageFolder(root='./custom_dataset', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

Random Transforms for Data Augmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    from riemann.vision import transforms
    
    # Define augmentation transforms
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms to dataset
    dataset = CIFAR10(root='./data', train=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

Custom Transforms
~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    from riemann.vision import transforms
    import numpy as np
    
    class CustomTransform:
        def __init__(self, factor=1.0):
            self.factor = factor
        
        def __call__(self, img):
            # Convert to numpy array
            img_array = np.array(img, dtype=np.float32)
            # Apply custom transformation
            img_array = img_array * self.factor
            # Convert back to PIL Image
            return transforms.ToPILImage()(img_array)
    
    # Define transforms with custom transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        CustomTransform(factor=1.2),
        transforms.ToTensor(),
    ])
    
    # Apply transforms to dataset
    dataset = CIFAR10(root='./data', train=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

Image Processing
----------------

Riemann provides various image processing utilities.

Image Conversion
~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    from riemann.vision import transforms
    
    # Convert PIL Image to tensor
    pil_img = ...  # PIL Image
    tensor_img = transforms.ToTensor()(pil_img)
    
    # Convert tensor to PIL Image
    pil_img = transforms.ToPILImage()(tensor_img)



Examples
--------

Image Classification with Data Augmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    import riemann.optim as optim
    from riemann.vision import datasets, transforms
    from riemann.utils.data import DataLoader
    
    # Define transforms for training and testing
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
    
    # Load datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create a simple CNN model
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
    
    model = SimpleCNN(num_classes=10)  # CIFAR-10 has 10 classes
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Training loop
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
    
    # Evaluation
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

Custom Dataset for Image Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Create dataset and dataloader
    dataset = CustomImageDataset(csv_file='labels.csv', root_dir='images/', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Use in training loop
    for images, labels in dataloader:
        print(images.shape)  # [32, 3, 224, 224]
        print(labels.shape)  # [32]
        break