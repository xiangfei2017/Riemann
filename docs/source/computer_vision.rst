Computer Vision
===============

Riemann provides comprehensive support for computer vision tasks through the ``riemann.vision`` module, including popular datasets, image transformations, and data loading utilities.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

The ``riemann.vision`` module includes the following main components:

- **Datasets**: Popular datasets like MNIST, CIFAR-10, Flowers102, OxfordIIITPet, LFWPeople, SVHN, ImageFolder, etc.
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

Dataset Overview
~~~~~~~~~~~~~~~~

.. list-table:: Supported Datasets
   :header-rows: 1
   :widths: 20 35 15 30

   * - Dataset
     - Description
     - Size
     - Download Source
   * - MNIST
     - Handwritten digit recognition (0-9), 28×28 grayscale images
     - 60,000 train / 10,000 test
     - AWS S3 (ossci-datasets)
   * - FashionMNIST
     - Fashion product images (10 categories), 28×28 grayscale
     - 60,000 train / 10,000 test
     - Zalando Research
   * - CIFAR-10
     - 10-class object recognition, 32×32 color images
     - 50,000 train / 10,000 test
     - University of Toronto
   * - CIFAR-100
     - 100-class object recognition with 20 superclasses, 32×32 color images
     - 50,000 train / 10,000 test
     - University of Toronto
   * - Flowers102
     - 102 flower categories classification
     - 1,020 train / 1,020 val / 6,149 test
     - Oxford VGG
   * - OxfordIIITPet
     - 37 pet breeds (cats and dogs) classification
     - ~7,000 images (~200 per class)
     - Oxford VGG
   * - LFWPeople
     - Face recognition dataset with multiple identities
     - 13,233 images / 5,749 people
     - UMass Amherst
   * - SVHN
     - Street View House Numbers, 32×32 color images
     - 73,257 train / 26,032 test / 531,131 extra
     - Stanford University
   * - ImageFolder
     - Generic folder-based dataset loader
     - User-defined
     - Local files
   * - DatasetFolder
     - Generic folder dataset with custom loader
     - User-defined
     - Local files

MNIST Dataset
~~~~~~~~~~~~~

Handwritten digit recognition dataset containing 60,000 training images and 10,000 test images, with image size of 28×28 pixels.

**Parameters**:

- ``root`` (str): Root directory for data storage
- ``train`` (bool): ``True`` to load training set, ``False`` to load test set
- ``transform`` (callable, optional): Image transformation function
- ``target_transform`` (callable, optional): Label transformation function
- ``download`` (bool, optional): If True, downloads the dataset from the internet

**Usage Example**:

.. code-block:: python

    from riemann.vision.datasets import MNIST
    from riemann.utils.data import DataLoader

    # Load training and test sets
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

EasyMNIST (Preprocessed MNIST)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

EasyMNIST is a preprocessed version of MNIST that applies normalization, standardization, and flattening during initialization. Labels can be converted to one-hot encoding. This saves preprocessing time during training as transformations are applied once at initialization rather than during each epoch.

**Parameters**:

- ``root`` (str): Root directory for data storage
- ``train`` (bool): ``True`` to load training set, ``False`` to load test set
- ``onehot_label`` (bool): If True, convert labels to one-hot encoding (default: True)
- ``download`` (bool, optional): If True, downloads the dataset from the internet

**Usage Example**:

.. code-block:: python

    from riemann.vision.datasets import EasyMNIST

    # Load EasyMNIST with one-hot labels (default)
    train_dataset = EasyMNIST(root='./data', train=True, onehot_label=True, download=True)
    
    # Load with scalar labels
    test_dataset = EasyMNIST(root='./data', train=False, onehot_label=False, download=True)

    # Data is already preprocessed (normalized, flattened)
    image, label = train_dataset[0]
    print(f"Image shape: {image.shape}")  # [784] - flattened
    print(f"Label shape: {label.shape}")  # [10] - one-hot if onehot_label=True

FashionMNIST Dataset
~~~~~~~~~~~~~~~~~~~~

Fashion-MNIST is a dataset of Zalando's article images consisting of 60,000 training examples and 10,000 test examples. Each example is a 28×28 grayscale image, associated with a label from 10 classes. It is designed to be a drop-in replacement for MNIST.

**Classes**: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

**Parameters**:

- ``root`` (str): Root directory for data storage
- ``train`` (bool): ``True`` to load training set, ``False`` to load test set
- ``transform`` (callable, optional): Image transformation function
- ``target_transform`` (callable, optional): Label transformation function
- ``download`` (bool, optional): If True, downloads the dataset from the internet

**Usage Example**:

.. code-block:: python

    from riemann.vision.datasets import FashionMNIST

    # Load FashionMNIST dataset
    train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    print(f"Classes: {train_dataset.classes}")

CIFAR-10 Dataset
~~~~~~~~~~~~~~~~

Contains 60,000 32×32 color images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

**Parameters**:

- ``root`` (str): Root directory for data storage
- ``train`` (bool): ``True`` to load training set (50,000 images), ``False`` to load test set (10,000 images)
- ``transform`` (callable, optional): Image transformation function
- ``target_transform`` (callable, optional): Label transformation function
- ``download`` (bool, optional): If True, downloads the dataset from the internet

**Usage Example**:

.. code-block:: python

    from riemann.vision.datasets import CIFAR10

    # Load CIFAR-10 dataset
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

CIFAR-100 Dataset
~~~~~~~~~~~~~~~~~

Contains 60,000 32×32 color images in 100 classes. Each class has 600 images (500 for training, 100 for testing). CIFAR-100 has 100 fine-grained classes and 20 superclasses.

**Parameters**:

- ``root`` (str): Root directory for data storage
- ``train`` (bool): ``True`` to load training set (50,000 images), ``False`` to load test set (10,000 images)
- ``transform`` (callable, optional): Image transformation function
- ``target_transform`` (callable, optional): Label transformation function
- ``download`` (bool, optional): If True, downloads the dataset from the internet
- ``coarse`` (bool, optional): If True, uses 20 superclass labels; otherwise uses 100 fine-grained class labels (default: False)

**Usage Example**:

.. code-block:: python

    from riemann.vision.datasets import CIFAR100

    # Load CIFAR-100 with fine-grained labels (100 classes)
    train_dataset = CIFAR100(root='./data', train=True, download=True, coarse=False)
    test_dataset = CIFAR100(root='./data', train=False, download=True, coarse=False)

    # Load CIFAR-100 with superclass labels (20 classes)
    train_dataset_coarse = CIFAR100(root='./data', train=True, download=True, coarse=True)

Flowers102 Dataset
~~~~~~~~~~~~~~~~~~

Oxford 102 Flower is an image classification dataset consisting of 102 flower categories. The flowers were chosen to be flowers commonly occurring in the United Kingdom. Each class consists of between 40 and 258 images. The images have large scale, pose and light variations.

**Note**: This class requires ``scipy`` to load target files from ``.mat`` format.

**Parameters**:

- ``root`` (str): Root directory of the dataset
- ``split`` (str, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``
- ``transform`` (callable, optional): Image transformation function
- ``target_transform`` (callable, optional): Target transformation function
- ``download`` (bool, optional): If True, downloads the dataset from the internet

**Dataset Statistics**:

- Train: 1,020 images
- Validation: 1,020 images  
- Test: 6,149 images
- Total: 8,189 images across 102 classes

**Usage Example**:

.. code-block:: python

    from riemann.vision.datasets import Flowers102

    # Load Flowers102 dataset
    train_dataset = Flowers102(root='./data', split='train', download=True, transform=transforms.ToTensor())
    val_dataset = Flowers102(root='./data', split='val', download=True, transform=transforms.ToTensor())
    test_dataset = Flowers102(root='./data', split='test', download=True, transform=transforms.ToTensor())

    print(f"Train samples: {len(train_dataset)}")  # 1020
    print(f"Validation samples: {len(val_dataset)}")  # 1020
    print(f"Test samples: {len(test_dataset)}")  # 6149

OxfordIIITPet Dataset
~~~~~~~~~~~~~~~~~~~~~

The Oxford-IIIT Pet Dataset is a 37 category pet dataset with roughly 200 images for each class. The images have large variations in scale, pose and lighting. All images have an associated ground truth annotation of species (cat or dog), breed, and pixel-level trimap segmentation.

**Parameters**:

- ``root`` (str): Root directory of the dataset
- ``split`` (str, optional): The dataset split, supports ``"trainval"`` (default) or ``"test"``
- ``target_types`` (str or list, optional): Types of target to use. Can be ``"category"`` (default), ``"binary-category"``, or ``"segmentation"``. Can also be a list to output a tuple with all specified target types.
- ``transform`` (callable, optional): Image transformation function
- ``target_transform`` (callable, optional): Target transformation function
- ``download`` (bool, optional): If True, downloads the dataset from the internet

**Target Types**:

- ``category`` (int): Label for one of the 37 pet categories
- ``binary-category`` (int): Binary label for cat (0) or dog (1)
- ``segmentation`` (PIL Image): Segmentation trimap of the image

**Usage Example**:

.. code-block:: python

    from riemann.vision.datasets import OxfordIIITPet

    # Load with category labels
    dataset = OxfordIIITPet(root='./data', split='trainval', target_types='category', download=True)
    
    # Load with binary classification (cat vs dog)
    dataset_bin = OxfordIIITPet(root='./data', split='trainval', target_types='binary-category', download=True)
    
    # Load with segmentation masks
    dataset_seg = OxfordIIITPet(root='./data', split='trainval', target_types='segmentation', download=True)
    
    # Load with multiple target types
    dataset_multi = OxfordIIITPet(root='./data', split='trainval', 
                                   target_types=['category', 'segmentation'], download=True)

LFWPeople Dataset
~~~~~~~~~~~~~~~~~

LFW (Labeled Faces in the Wild) People dataset contains 13,233 face images collected from the web. The images are organized into 5,749 different identities. This dataset is designed for face recognition research.

**Parameters**:

- ``root`` (str): Root directory of the dataset
- ``split`` (str, optional): The dataset split, supports ``"10fold"`` (default), ``"train"``, or ``"test"``
- ``image_set`` (str, optional): The image alignment type, supports ``"original"``, ``"funneled"`` (default), or ``"deepfunneled"``
- ``transform`` (callable, optional): Image transformation function
- ``target_transform`` (callable, optional): Target transformation function
- ``download`` (bool, optional): If True, downloads the dataset from the internet

**Image Sets**:

- ``original``: Original images without alignment
- ``funneled``: Geometrically normalized face images (default)
- ``deepfunneled``: Deep funneled images with better alignment

**Usage Example**:

.. code-block:: python

    from riemann.vision.datasets import LFWPeople

    # Load LFWPeople dataset with funneled images
    train_dataset = LFWPeople(root='./data', split='train', image_set='funneled', download=True)
    test_dataset = LFWPeople(root='./data', split='test', image_set='funneled', download=True)

    print(f"Number of classes (people): {len(train_dataset.classes)}")
    print(f"Train samples: {len(train_dataset)}")

SVHN Dataset
~~~~~~~~~~~~

SVHN (Street View House Numbers) dataset contains 32×32 color images of house numbers collected from Google Street View. The dataset includes 10 digit classes (0-9).

**Note**: This class requires ``scipy`` to load data from ``.mat`` format.

**Parameters**:

- ``root`` (str): Root directory of the dataset
- ``split`` (str): The dataset split, supports ``"train"``, ``"test"``, or ``"extra"``
- ``transform`` (callable, optional): Image transformation function
- ``target_transform`` (callable, optional): Target transformation function
- ``download`` (bool, optional): If True, downloads the dataset from the internet

**Dataset Statistics**:

- Train: 73,257 images
- Test: 26,032 images
- Extra: 531,131 additional images (less difficult samples)

**Usage Example**:

.. code-block:: python

    from riemann.vision.datasets import SVHN

    # Load SVHN dataset
    train_dataset = SVHN(root='./data', split='train', download=True, transform=transforms.ToTensor())
    test_dataset = SVHN(root='./data', split='test', download=True, transform=transforms.ToTensor())
    
    # Also available: extra split with additional training data
    extra_dataset = SVHN(root='./data', split='extra', download=True, transform=transforms.ToTensor())

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

    # Five crop: returns a tuple of 5 images
    five_crop = transforms.FiveCrop(224)
    
    # Apply to image
    crops = five_crop(image)  # (top_left, top_right, bottom_left, bottom_right, center)
    
    # Stack into batch
    tensor_crops = rm.stack([transforms.ToTensor()(crop) for crop in crops])

**TenCrop**: Ten crop (FiveCrop + horizontal flips)

.. code-block:: python

    # Ten crop: returns a tuple of 10 images
    ten_crop = transforms.TenCrop(224)

**Pad**: Pad image

.. code-block:: python

    # Pad image with specified padding
    pad = transforms.Pad(padding=4, fill=0)

**Lambda**: Custom transform using lambda

.. code-block:: python

    # Define custom transform
    custom_transform = transforms.Lambda(lambda x: x.rotate(90))

Complete Example
----------------

Complete image classification training workflow:

.. code-block:: python

    import riemann as rm
    from riemann.vision import datasets, transforms
    from riemann.utils.data import DataLoader
    from riemann.nn import Module, Linear, ReLU, CrossEntropyLoss
    from riemann.optim import SGD

    # Define training and test transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
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

    # Load dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                      download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                     download=True, transform=test_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, 
                              shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, 
                             shuffle=False, num_workers=4)

    # Define model
    class SimpleNet(Module):
        def __init__(self):
            super().__init__()
            self.fc1 = Linear(224 * 224 * 3, 256)
            self.relu = ReLU()
            self.fc2 = Linear(256, 10)
        
        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    model = SimpleNet()
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(10):
        for images, labels in train_loader:
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1} completed")

API Reference
-------------

.. py:module:: riemann.vision.datasets

.. py:class:: MNIST
   
   MNIST handwritten digits dataset. Includes EasyMNIST variant with preprocessing.

.. py:class:: FashionMNIST
   
   Fashion-MNIST dataset.

.. py:class:: CIFAR10
   
   CIFAR-10 dataset.

.. py:class:: Flowers102
   
   Oxford 102 Flower dataset.

.. py:class:: OxfordIIITPet
   
   Oxford-IIIT Pet dataset.

.. py:class:: LFWPeople
   
   Labeled Faces in the Wild People dataset.

.. py:class:: SVHN
   
   Street View House Numbers dataset.

.. py:class:: ImageFolder
   
   Generic data loader for images from local folders.

.. py:class:: DatasetFolder
   
   Generic data loader for custom image formats.

.. py:module:: riemann.vision.transforms
   :noindex:

.. py:class:: Compose
   
   Compose multiple transforms.

.. py:class:: ToTensor
   
   Convert PIL Image or numpy.ndarray to tensor.

.. py:class:: ToPILImage
   
   Convert tensor to PIL Image.

.. py:class:: Resize
   
   Resize image.

.. py:class:: CenterCrop
   
   Center crop image.

.. py:class:: RandomResizedCrop
   
   Random resized crop.

.. py:class:: RandomHorizontalFlip
   
   Random horizontal flip.

.. py:class:: Normalize
   
   Normalize tensor.
