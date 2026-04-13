Computer Vision
===============

Riemann provides comprehensive support for computer vision tasks through the ``riemann.vision`` module, including popular datasets, image transformations, and data loading utilities.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

The ``riemann.vision`` module includes the following main components:

- **Datasets**: Popular datasets like MNIST, CIFAR-10, ImageFolder, etc.
- **Transforms**: Image preprocessing and data augmentation operations
- **Data Loading**: Seamless integration with ``DataLoader``, supporting batch loading and parallel processing

Quick Start
-----------

.. code-block:: python

    import riemann as rm
    from riemann.vision import datasets, transforms
    from riemann.utils.data import DataLoader

    # Define data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load dataset
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Iterate through data
    for images, labels in train_loader:
        print(f"Image batch shape: {images.shape}")  # [64, 1, 28, 28]
        print(f"Label batch shape: {labels.shape}")  # [64]
        break

Datasets
--------

Riemann provides various popular computer vision datasets. All datasets inherit from the ``Dataset`` class and can be used with ``DataLoader``.

MNIST Dataset
~~~~~~~~~~~~~

Handwritten digit recognition dataset containing 60,000 training images and 10,000 test images, with image size of 28×28 pixels.

**Parameters**:

- ``root`` (str): Root directory for data storage
- ``train`` (bool): ``True`` to load training set, ``False`` to load test set
- ``transform`` (callable, optional): Image transformation function
- ``target_transform`` (callable, optional): Label transformation function

**Usage Example**:

.. code-block:: python

    from riemann.vision.datasets import MNIST
    from riemann.utils.data import DataLoader

    # Load training and test sets
    train_dataset = MNIST(root='./data', train=True, transform=transforms.ToTensor())
    test_dataset = MNIST(root='./data', train=False, transform=transforms.ToTensor())

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

CIFAR-10 Dataset
~~~~~~~~~~~~~~~~

Contains 60,000 32×32 color images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

**Parameters**:

- ``root`` (str): Root directory for data storage
- ``train`` (bool): ``True`` to load training set (50,000 images), ``False`` to load test set (10,000 images)
- ``transform`` (callable, optional): Image transformation function
- ``target_transform`` (callable, optional): Label transformation function

**Usage Example**:

.. code-block:: python

    from riemann.vision.datasets import CIFAR10

    # Load CIFAR-10 dataset
    train_dataset = CIFAR10(root='./data', train=True, transform=transforms.ToTensor())
    test_dataset = CIFAR10(root='./data', train=False, transform=transforms.ToTensor())

ImageFolder Dataset
~~~~~~~~~~~~~~~~~~~

Load image datasets from local folders, suitable for custom datasets. Folder structure should be organized by class:

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

**Parameters**:

- ``root`` (str): Root directory path of the dataset
- ``transform`` (callable, optional): Image transformation function
- ``target_transform`` (callable, optional): Label transformation function
- ``loader`` (callable, optional): Image loading function, defaults to PIL Image loader
- ``is_valid_file`` (callable, optional): Function to validate if a file is valid

**Usage Example**:

.. code-block:: python

    from riemann.vision.datasets import ImageFolder

    # Load custom dataset from folder
    dataset = ImageFolder(
        root='./custom_dataset',
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    )

    print(f"Number of classes: {len(dataset.classes)}")  # ['class_a', 'class_b', 'class_c']
    print(f"Class to index mapping: {dataset.class_to_idx}")  # {'class_a': 0, 'class_b': 1, 'class_c': 2}

DatasetFolder
~~~~~~~~~~~~~

Generic folder dataset class, similar to ``ImageFolder`` but allows custom image loaders.

**Parameters**:

- ``root`` (str): Root directory path of the dataset
- ``loader`` (callable): Image loading function
- ``extensions`` (tuple, optional): Tuple of allowed file extensions
- ``transform`` (callable, optional): Image transformation function
- ``target_transform`` (callable, optional): Label transformation function
- ``is_valid_file`` (callable, optional): Function to validate if a file is valid
- ``allow_empty`` (bool): Whether to allow empty folders, default ``False``

**Usage Example**:

.. code-block:: python

    from riemann.vision.datasets import DatasetFolder, default_loader

    # Use custom loader
    dataset = DatasetFolder(
        root='./custom_dataset',
        loader=default_loader,
        extensions=('.jpg', '.png'),
        transform=transforms.ToTensor()
    )

Transforms
----------

``riemann.vision.transforms`` provides rich image transformation operations for data preprocessing and data augmentation.

Compose
~~~~~~~

Combine multiple transformations and apply them in sequence.

.. code-block:: python

    from riemann.vision import transforms

    # Define transformation pipeline
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

Basic Transforms
~~~~~~~~~~~~~~~~

**ToTensor**: Convert PIL Image or numpy.ndarray to tensor

.. code-block:: python

    # PIL Image -> Tensor (value range [0, 1])
    tensor = transforms.ToTensor()(pil_image)

**ToPILImage**: Convert tensor to PIL Image

.. code-block:: python

    # Tensor -> PIL Image
    pil_image = transforms.ToPILImage()(tensor)

**Resize**: Resize image

.. code-block:: python

    # Resize to specified size
    resize = transforms.Resize((224, 224))
    
    # Resize proportionally by shorter side
    resize = transforms.Resize(256)

**CenterCrop**: Center crop

.. code-block:: python

    # Crop specified size from image center
    crop = transforms.CenterCrop(224)

**Normalize**: Normalization

.. code-block:: python

    # Normalize using mean and standard deviation
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

Data Augmentation
~~~~~~~~~~~~~~~~~

**RandomResizedCrop**: Random resized crop

.. code-block:: python

    # Random crop and resize to specified size
    crop = transforms.RandomResizedCrop(224, scale=(0.08, 1.0))

**RandomHorizontalFlip**: Random horizontal flip

.. code-block:: python

    # Flip horizontally with 50% probability
    flip = transforms.RandomHorizontalFlip(p=0.5)

**RandomVerticalFlip**: Random vertical flip

.. code-block:: python

    # Flip vertically with 50% probability
    flip = transforms.RandomVerticalFlip(p=0.5)

**RandomRotation**: Random rotation

.. code-block:: python

    # Random rotation between (-15, 15) degrees
    rotation = transforms.RandomRotation(degrees=15)

**ColorJitter**: Color jitter

.. code-block:: python

    # Randomly adjust brightness, contrast, saturation, and hue
    jitter = transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    )

**RandomCrop**: Random crop

.. code-block:: python

    # Random crop to specified size
    crop = transforms.RandomCrop(224, padding=4)

**RandomGrayscale**: Random grayscale

.. code-block:: python

    # Convert to grayscale with 10% probability
    gray = transforms.RandomGrayscale(p=0.1)

Advanced Transforms
~~~~~~~~~~~~~~~~~~~

**FiveCrop**: Five crop (corners and center)

.. code-block:: python

    import riemann as rm
    from riemann.vision import transforms

    # Five crop on image
    five_crop = transforms.FiveCrop(224)
    image = rm.randn(3, 256, 256)  # [C, H, W]
    crops = five_crop(image)  # Returns tuple of 5 tensors

**TenCrop**: Ten crop (five crop + horizontal flip)

.. code-block:: python

    # Ten crop on image
    ten_crop = transforms.TenCrop(224, vertical_flip=False)
    crops = ten_crop(image)  # Returns tuple of 10 tensors

**Pad**: Image padding

.. code-block:: python

    # Pad image
    pad = transforms.Pad(padding=4, fill=0)

AutoAugment Family
~~~~~~~~~~~~~~~~~~

**AutoAugment**: Automatic data augmentation

.. code-block:: python

    from riemann.vision.transforms import AutoAugment, AutoAugmentPolicy

    # Use ImageNet policy
    augment = AutoAugment(policy=AutoAugmentPolicy.IMAGENET)
    augmented_image = augment(image)

**RandAugment**: Random data augmentation

.. code-block:: python

    from riemann.vision.transforms import RandAugment

    # Randomly select transformation combination
    augment = RandAugment(num_ops=2, magnitude=9)
    augmented_image = augment(image)

**TrivialAugmentWide**: Wide range simple augmentation

.. code-block:: python

    from riemann.vision.transforms import TrivialAugmentWide

    # Wide range simple augmentation
    augment = TrivialAugmentWide()
    augmented_image = augment(image)

Utilities
---------

**default_loader**: Default image loader

.. code-block:: python

    from riemann.vision.datasets import default_loader

    # Load image file
    image = default_loader('path/to/image.jpg')

Complete Examples
-----------------

Image Classification Training Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    import riemann.optim as optim
    from riemann.vision import datasets, transforms
    from riemann.utils.data import DataLoader

    # Define data transformations
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

    # Load datasets
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

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define model
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

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Training loop
    for epoch in range(10):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

Custom Dataset
~~~~~~~~~~~~~~

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
            
            # Load data list
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

    # Use custom dataset
    dataset = CustomImageDataset(
        root_dir='./custom_data',
        transform=transforms.ToTensor()
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

API Reference
-------------

Dataset Classes
~~~~~~~~~~~~~~~

.. list-table:: Dataset Classes
   :header-rows: 1
   :widths: 25 75

   * - Class Name
     - Description
   * - ``MNIST``
     - MNIST handwritten digit dataset
   * - ``EasyMNIST``
     - Simplified MNIST dataset
   * - ``CIFAR10``
     - CIFAR-10 image classification dataset
   * - ``ImageFolder``
     - Load image dataset from folder
   * - ``DatasetFolder``
     - Generic folder dataset base class

Transform Classes
~~~~~~~~~~~~~~~~~

.. list-table:: Transform Classes
   :header-rows: 1
   :widths: 25 75

   * - Class Name
     - Description
   * - **Composition**
     -
   * - ``Compose``
     - Compose multiple transforms and apply sequentially
   * - **Type Conversion**
     -
   * - ``ToTensor``
     - Convert PIL Image or numpy.ndarray to tensor
   * - ``PILToTensor``
     - Convert PIL Image to tensor (without scaling)
   * - ``ToPILImage``
     - Convert tensor to PIL Image
   * - ``ConvertImageDtype``
     - Convert image data type
   * - **Geometric Transforms**
     -
   * - ``Resize``
     - Resize image
   * - ``CenterCrop``
     - Center crop
   * - ``RandomCrop``
     - Random crop
   * - ``RandomResizedCrop``
     - Random resized crop
   * - ``FiveCrop``
     - Five crop (corners and center)
   * - ``TenCrop``
     - Ten crop (five crop + horizontal flip)
   * - ``Pad``
     - Image padding
   * - **Flip and Rotation**
     -
   * - ``RandomHorizontalFlip``
     - Random horizontal flip
   * - ``RandomVerticalFlip``
     - Random vertical flip
   * - ``RandomRotation``
     - Random rotation
   * - ``RandomAffine``
     - Random affine transformation
   * - ``RandomPerspective``
     - Random perspective transformation
   * - **Color Transforms**
     -
   * - ``ColorJitter``
     - Color jitter (brightness, contrast, saturation, hue)
   * - ``Grayscale``
     - Convert to grayscale
   * - ``RandomGrayscale``
     - Randomly convert to grayscale
   * - ``Invert``
     - Invert colors
   * - ``Posterize``
     - Reduce color bits
   * - ``Solarize``
     - Invert pixels above threshold
   * - ``Equalize``
     - Histogram equalization
   * - ``AutoContrast``
     - Auto contrast adjustment
   * - ``Sharpness``
     - Sharpness adjustment
   * - ``Brightness``
     - Brightness adjustment
   * - ``Contrast``
     - Contrast adjustment
   * - ``Saturation``
     - Saturation adjustment
   * - ``Hue``
     - Hue adjustment
   * - **Normalization**
     -
   * - ``Normalize``
     - Normalize with mean and standard deviation
   * - **Advanced Augmentation**
     -
   * - ``AutoAugment``
     - Automatic data augmentation (learning-based policy)
   * - ``RandAugment``
     - Random data augmentation
   * - ``TrivialAugmentWide``
     - Wide range simple augmentation
   * - **Other Transforms**
     -
   * - ``Lambda``
     - Apply custom lambda function
   * - ``GaussianBlur``
     - Gaussian blur
   * - ``RandomErasing``
     - Random erasing (for data augmentation)
   * - ``SanitizeBoundingBox``
     - Sanitize bounding boxes