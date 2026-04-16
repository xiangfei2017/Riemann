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

default_loader
~~~~~~~~~~~~~~

``default_loader`` is the default image loading function used by ``ImageFolder`` and ``DatasetFolder``. It automatically selects the appropriate loading method based on file extension:

- Image formats supported by PIL (e.g., .jpg, .png, .bmp, etc.): Loaded using PIL.Image.open() and converted to RGB mode
- Other formats: Attempts to load using PIL

**Purpose**:

``default_loader`` is mainly used for the ``loader`` parameter of ``ImageFolder`` and ``DatasetFolder`` to specify the image loading method. When using these two dataset classes, if the ``loader`` parameter is not specified, ``default_loader`` will be used by default.

**Usage Example**:

.. code-block:: python

    from riemann.vision.datasets import DatasetFolder, default_loader

    # Load image using default_loader
    image = default_loader('path/to/image.jpg')
    
    # Use in DatasetFolder
    dataset = DatasetFolder(
        root='./custom_dataset',
        loader=default_loader,  # Specify to use default_loader
        extensions=('.jpg', '.png')
    )

Transforms
----------

``riemann.vision.transforms`` provides rich image transformation operations for data preprocessing and data augmentation.

Transform Overview
~~~~~~~~~~~~~~~~~~

.. list-table:: Supported Transforms
   :header-rows: 1
   :widths: 25 35 40

   * - Transform
     - Description
     - Category
   * - Compose
     - Combine multiple transforms into one
     - Utility
   * - PILToTensor
     - Convert PIL Image to tensor without scaling
     - Conversion
   * - ToTensor
     - Convert PIL Image or numpy.ndarray to tensor (scales to [0, 1])
     - Conversion
   * - ToPILImage
     - Convert tensor to PIL Image
     - Conversion
   * - ConvertImageDtype
     - Convert image to specified data type
     - Conversion
   * - Normalize
     - Normalize tensor with mean and std
     - Normalization
   * - Resize
     - Resize image to specified size
     - Geometric
   * - CenterCrop
     - Crop image from center
     - Geometric
   * - RandomHorizontalFlip
     - Randomly flip image horizontally
     - Augmentation
   * - RandomVerticalFlip
     - Randomly flip image vertically
     - Augmentation
   * - RandomRotation
     - Randomly rotate image by angle
     - Augmentation
   * - ColorJitter
     - Randomly change brightness, contrast, saturation, hue
     - Augmentation
   * - Grayscale
     - Convert image to grayscale
     - Color
   * - RandomGrayscale
     - Randomly convert image to grayscale
     - Augmentation
   * - RandomCrop
     - Randomly crop image to specified size
     - Augmentation
   * - RandomResizedCrop
     - Random crop and resize image
     - Augmentation
   * - FiveCrop
     - Crop image into 5 regions (4 corners + center)
     - Geometric
   * - TenCrop
     - Crop image into 10 regions (FiveCrop + flips)
     - Geometric
   * - Pad
     - Pad image with specified value
     - Geometric
   * - Lambda
     - Apply custom lambda function
     - Utility
   * - GaussianBlur
     - Apply Gaussian blur to image
     - Filter
   * - RandomAffine
     - Random affine transformation
     - Augmentation
   * - RandomPerspective
     - Random perspective transformation
     - Augmentation
   * - RandomErasing
     - Randomly erase rectangular regions
     - Augmentation
   * - AutoAugment
     - AutoAugment data augmentation policy
     - Auto Augmentation
   * - RandAugment
     - RandAugment data augmentation policy
     - Auto Augmentation
   * - TrivialAugmentWide
     - TrivialAugmentWide data augmentation policy
     - Auto Augmentation
   * - SanitizeBoundingBox
     - Sanitize and validate bounding boxes
     - Detection
   * - Invert
     - Invert image colors
     - Color
   * - Posterize
     - Reduce number of bits for each color channel
     - Color
   * - Solarize
     - Invert pixels above threshold
     - Color
   * - Equalize
     - Equalize image histogram
     - Color
   * - AutoContrast
     - Maximize image contrast
     - Color
   * - Sharpness
     - Adjust image sharpness
     - Color
   * - Brightness
     - Adjust image brightness
     - Color
   * - Contrast
     - Adjust image contrast
     - Color
   * - Saturation
     - Adjust image saturation
     - Color
   * - Hue
     - Adjust image hue
     - Color

Compose
~~~~~~~

Combine multiple transformations and apply them in sequence.

**Parameters**:

- ``transforms`` (list): List of transform objects to compose

**Usage Example**:

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

PILToTensor
~~~~~~~~~~~

Convert PIL Image to tensor without scaling. Unlike ToTensor, PILToTensor does not scale values from [0, 255] to [0.0, 1.0].

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Convert PIL Image to tensor (values in [0, 255])
    pil_to_tensor = transforms.PILToTensor()
    tensor_img = pil_to_tensor(pil_image)

ToTensor
~~~~~~~~

Convert PIL Image or numpy.ndarray to tensor. Scales values from [0, 255] to [0.0, 1.0].

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Convert PIL Image to tensor (values in [0, 1])
    to_tensor = transforms.ToTensor()
    tensor_img = to_tensor(pil_image)

ToPILImage
~~~~~~~~~~

Convert tensor to PIL Image.

**Parameters**:

- ``mode`` (str, optional): Color mode of the output image

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Convert tensor to PIL Image
    to_pil = transforms.ToPILImage()
    pil_img = to_pil(tensor)

ConvertImageDtype
~~~~~~~~~~~~~~~~~

Convert image to specified data type.

**Parameters**:

- ``dtype`` (dtype): Target data type

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Convert to float32
    convert_dtype = transforms.ConvertImageDtype(dtype='float32')
    converted_img = convert_dtype(img)

Normalize
~~~~~~~~~

Normalize tensor with mean and standard deviation.

**Parameters**:

- ``mean`` (sequence): Mean values for each channel
- ``std`` (sequence): Standard deviation values for each channel

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Normalize using ImageNet statistics
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    normalized_img = normalize(tensor_img)

Resize
~~~~~~

Resize image to specified size.

**Parameters**:

- ``size`` (int or tuple): Target size. If int, smaller edge is resized to size. If tuple, (height, width).

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Resize to specific size
    resize = transforms.Resize((224, 224))
    resized_img = resize(pil_image)

    # Resize by shorter side
    resize = transforms.Resize(256)
    resized_img = resize(pil_image)

CenterCrop
~~~~~~~~~~

Crop image from center.

**Parameters**:

- ``size`` (int or tuple): Crop size

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Center crop to 224x224
    center_crop = transforms.CenterCrop(224)
    cropped_img = center_crop(pil_image)

RandomHorizontalFlip
~~~~~~~~~~~~~~~~~~~~

Randomly flip image horizontally.

**Parameters**:

- ``p`` (float): Probability of flipping (default: 0.5)

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Flip with 50% probability
    hflip = transforms.RandomHorizontalFlip(p=0.5)
    flipped_img = hflip(pil_image)

RandomVerticalFlip
~~~~~~~~~~~~~~~~~~

Randomly flip image vertically.

**Parameters**:

- ``p`` (float): Probability of flipping (default: 0.5)

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Flip with 50% probability
    vflip = transforms.RandomVerticalFlip(p=0.5)
    flipped_img = vflip(pil_image)

RandomRotation
~~~~~~~~~~~~~~

Randomly rotate image by angle.

**Parameters**:

- ``degrees`` (sequence or float): Range of degrees (-degrees, +degrees)

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Rotate between -15 and 15 degrees
    rotation = transforms.RandomRotation(degrees=15)
    rotated_img = rotation(pil_image)

ColorJitter
~~~~~~~~~~~

Randomly change brightness, contrast, saturation, and hue.

**Parameters**:

- ``brightness`` (float): Brightness jitter factor
- ``contrast`` (float): Contrast jitter factor
- ``saturation`` (float): Saturation jitter factor
- ``hue`` (float): Hue jitter factor

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Randomly adjust color
    jitter = transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    )
    jittered_img = jitter(pil_image)

Grayscale
~~~~~~~~~

Convert image to grayscale.

**Parameters**:

- ``num_output_channels`` (int): Number of output channels (1 or 3)

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Convert to grayscale (1 channel)
    gray = transforms.Grayscale(num_output_channels=1)
    gray_img = gray(pil_image)

RandomGrayscale
~~~~~~~~~~~~~~~

Randomly convert image to grayscale.

**Parameters**:

- ``p`` (float): Probability of conversion (default: 0.1)

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Convert to grayscale with 10% probability
    gray = transforms.RandomGrayscale(p=0.1)
    gray_img = gray(pil_image)

RandomCrop
~~~~~~~~~~

Randomly crop image to specified size.

**Parameters**:

- ``size`` (int or tuple): Crop size
- ``padding`` (int, optional): Padding size

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Random crop with padding
    crop = transforms.RandomCrop(224, padding=4)
    cropped_img = crop(pil_image)

RandomResizedCrop
~~~~~~~~~~~~~~~~~

Random crop and resize image.

**Parameters**:

- ``size`` (int or tuple): Target size
- ``scale`` (tuple): Scale range for cropping
- ``ratio`` (tuple): Aspect ratio range

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Random resized crop
    crop = transforms.RandomResizedCrop(224, scale=(0.08, 1.0))
    cropped_img = crop(pil_image)

FiveCrop
~~~~~~~~

Crop image into 5 regions (4 corners + center).

**Parameters**:

- ``size`` (int or tuple): Crop size

**Usage Example**:

.. code-block:: python

    import riemann as rm
    from riemann.vision import transforms

    # Five crop
    five_crop = transforms.FiveCrop(224)
    crops = five_crop(pil_image)  # Returns tuple of 5 images
    
    # Stack into batch
    tensor_crops = rm.stack([transforms.ToTensor()(crop) for crop in crops])

TenCrop
~~~~~~~

Crop image into 10 regions (FiveCrop + horizontal flips).

**Parameters**:

- ``size`` (int or tuple): Crop size
- ``vertical_flip`` (bool): Also apply vertical flip

**Usage Example**:

.. code-block:: python

    import riemann as rm
    from riemann.vision import transforms

    # Ten crop
    ten_crop = transforms.TenCrop(224)
    crops = ten_crop(pil_image)  # Returns tuple of 10 images

Pad
~~~

Pad image with specified value.

**Parameters**:

- ``padding`` (int or tuple): Padding size
- ``fill`` (int or tuple): Fill value

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Pad image
    pad = transforms.Pad(padding=4, fill=0)
    padded_img = pad(pil_image)

Lambda
~~~~~~

Apply custom lambda function.

**Parameters**:

- ``lambd`` (function): Lambda function to apply

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Custom lambda transform
    lambd = transforms.Lambda(lambda x: x.rotate(45))
    transformed_img = lambd(pil_image)

GaussianBlur
~~~~~~~~~~~~

Apply Gaussian blur to image.

**Parameters**:

- ``kernel_size`` (int): Gaussian kernel size
- ``sigma`` (float or tuple): Standard deviation

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Apply Gaussian blur
    blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    blurred_img = blur(pil_image)

RandomAffine
~~~~~~~~~~~~

Random affine transformation.

**Parameters**:

- ``degrees`` (float or tuple): Rotation degrees
- ``translate`` (tuple): Translation range
- ``scale`` (tuple): Scale range
- ``shear`` (float or tuple): Shear range

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Random affine transformation
    affine = transforms.RandomAffine(
        degrees=15,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    )
    transformed_img = affine(pil_image)

RandomPerspective
~~~~~~~~~~~~~~~~~

Random perspective transformation.

**Parameters**:

- ``distortion_scale`` (float): Distortion scale
- ``p`` (float): Probability of applying transform

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Random perspective
    perspective = transforms.RandomPerspective(distortion_scale=0.5, p=0.5)
    transformed_img = perspective(pil_image)

RandomErasing
~~~~~~~~~~~~~

Randomly erase rectangular regions.

**Parameters**:

- ``p`` (float): Probability of applying
- ``scale`` (tuple): Erasing area range
- ``ratio`` (tuple): Aspect ratio range
- ``value`` (str or float): Erasing value

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Random erasing (typically used on tensors)
    erasing = transforms.RandomErasing(p=0.5, scale=(0.02, 0.33))
    erased_tensor = erasing(tensor_img)

AutoAugment
~~~~~~~~~~~

AutoAugment data augmentation policy.

**Parameters**:

- ``policy`` (str): Policy to use ('imagenet', 'cifar10', 'svhn')

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # AutoAugment with ImageNet policy
    auto_augment = transforms.AutoAugment(policy='imagenet')
    augmented_img = auto_augment(pil_image)

RandAugment
~~~~~~~~~~~

RandAugment data augmentation policy.

**Parameters**:

- ``num_ops`` (int): Number of operations
- ``magnitude`` (int): Magnitude of operations

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # RandAugment
    rand_augment = transforms.RandAugment(num_ops=2, magnitude=9)
    augmented_img = rand_augment(pil_image)

TrivialAugmentWide
~~~~~~~~~~~~~~~~~~

TrivialAugmentWide data augmentation policy.

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # TrivialAugmentWide
    trivial_augment = transforms.TrivialAugmentWide()
    augmented_img = trivial_augment(pil_image)

SanitizeBoundingBox
~~~~~~~~~~~~~~~~~~~

Sanitize and validate bounding boxes.

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Sanitize bounding boxes
    sanitize = transforms.SanitizeBoundingBox()
    sanitized_boxes = sanitize(boxes, image_size)

Invert
~~~~~~

Invert image colors.

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Invert image
    invert = transforms.Invert()
    inverted_img = invert(pil_image)

Posterize
~~~~~~~~~

Reduce number of bits for each color channel.

**Parameters**:

- ``bits`` (int): Number of bits to keep

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Posterize image
    posterize = transforms.Posterize(bits=4)
    posterized_img = posterize(pil_image)

Solarize
~~~~~~~~

Invert pixels above threshold.

**Parameters**:

- ``threshold`` (int): Threshold value

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Solarize image
    solarize = transforms.Solarize(threshold=128)
    solarized_img = solarize(pil_image)

Equalize
~~~~~~~~

Equalize image histogram.

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Equalize image
    equalize = transforms.Equalize()
    equalized_img = equalize(pil_image)

AutoContrast
~~~~~~~~~~~~

Maximize image contrast.

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Auto contrast
    auto_contrast = transforms.AutoContrast()
    contrasted_img = auto_contrast(pil_image)

Sharpness
~~~~~~~~~

Adjust image sharpness.

**Parameters**:

- ``sharpness_factor`` (float): Sharpness factor

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Adjust sharpness
    sharpness = transforms.Sharpness(sharpness_factor=2.0)
    sharpened_img = sharpness(pil_image)

Brightness
~~~~~~~~~~

Adjust image brightness.

**Parameters**:

- ``brightness_factor`` (float): Brightness factor

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Adjust brightness
    brightness = transforms.Brightness(brightness_factor=1.5)
    brightened_img = brightness(pil_image)

Contrast
~~~~~~~~

Adjust image contrast.

**Parameters**:

- ``contrast_factor`` (float): Contrast factor

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Adjust contrast
    contrast = transforms.Contrast(contrast_factor=1.5)
    contrasted_img = contrast(pil_image)

Saturation
~~~~~~~~~~

Adjust image saturation.

**Parameters**:

- ``saturation_factor`` (float): Saturation factor

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Adjust saturation
    saturation = transforms.Saturation(saturation_factor=1.5)
    saturated_img = saturation(pil_image)

Hue
~~~

Adjust image hue.

**Parameters**:

- ``hue_factor`` (float): Hue factor (-0.5 to 0.5)

**Usage Example**:

.. code-block:: python

    from riemann.vision import transforms

    # Adjust hue
    hue = transforms.Hue(hue_factor=0.1)
    hue_adjusted_img = hue(pil_image)

Complete Examples
-----------------

The following examples demonstrate how to use Riemann's computer vision module for common deep learning tasks.

Image Classification Training Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates the complete workflow of image classification using the CIFAR-10 dataset, including data loading, data augmentation, model definition, training, and evaluation.

**Pipeline Overview**:

1. **Data Preprocessing**: Use random cropping, horizontal flipping, and color jittering for data augmentation
2. **Normalization**: Normalize using ImageNet statistics
3. **Model Definition**: Simple convolutional neural network
4. **Training Loop**: Standard training flow including forward propagation, loss calculation, backward propagation, and parameter updates

.. code-block:: python

    import riemann as rm
    import riemann.nn as nn
    import riemann.optim as optim
    from riemann.vision import datasets, transforms
    from riemann.utils.data import DataLoader

    # Define training data transforms (with data augmentation)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),      # Random crop and resize
        transforms.RandomHorizontalFlip(),       # Random horizontal flip
        transforms.ColorJitter(                  # Color jitter (data augmentation)
            brightness=0.2, 
            contrast=0.2
        ),
        transforms.ToTensor(),                   # Convert to tensor
        transforms.Normalize(                    # Normalize
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Define test data transforms (without data augmentation)
    test_transform = transforms.Compose([
        transforms.Resize(256),                  # Resize
        transforms.CenterCrop(224),              # Center crop
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load CIFAR-10 dataset
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

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,           # Shuffle data during training
        num_workers=4           # Use 4 subprocesses to load data
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False
    )

    # Define convolutional neural network model
    model = nn.Sequential(
        # First convolutional block
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # Second convolutional block
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # Fully connected layer
        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 10)  # CIFAR-10 has 10 classes
    )

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=0.01, 
        momentum=0.9
    )

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Forward propagation
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward propagation and optimization
            optimizer.zero_grad()   # Clear gradients
            loss.backward()         # Compute gradients
            optimizer.step()        # Update parameters
            
            running_loss += loss.item()
            
            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/100:.4f}')
                running_loss = 0.0
        
        print(f'Epoch {epoch+1} completed')

    print('Training completed!')

Loading Custom Dataset with ImageFolder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you have your own image dataset, you can use ``ImageFolder`` for convenient loading. Just organize images by folders, with each folder representing a class.

**Required Folder Structure**:

.. code-block:: text

    custom_dataset/
    ├── class_a/           # Images for class A
    │   ├── img1.jpg
    │   └── img2.png
    ├── class_b/           # Images for class B
    │   ├── img1.jpg
    │   └── img2.jpg
    └── class_c/           # Images for class C
        └── img1.jpg

**Loading Example**:

.. code-block:: python

    from riemann.vision import datasets, transforms
    from riemann.utils.data import DataLoader

    # Define data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load dataset using ImageFolder
    dataset = datasets.ImageFolder(
        root='./custom_dataset',
        transform=transform
    )

    # View dataset information
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"Class names: {dataset.classes}")
    print(f"Class to index mapping: {dataset.class_to_idx}")
    print(f"Total samples: {len(dataset)}")

    # Create data loader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Iterate through data
    for images, labels in loader:
        print(f"Image batch shape: {images.shape}")  # [32, 3, 224, 224]
        print(f"Label batch shape: {labels.shape}")  # [32]
        break

Creating Custom Dataset Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``ImageFolder`` cannot meet your needs, you can inherit from the ``Dataset`` class to create a custom dataset. The following example shows how to create a custom dataset that loads images from folders.

**Applicable Scenarios**:

- Need custom file organization
- Need to load data from other sources (e.g., database, network)
- Need complex preprocessing

.. code-block:: python

    from riemann.utils.data import Dataset
    from PIL import Image
    import os

    class CustomImageDataset(Dataset):
        """
        Custom image dataset class
        
        Load images from folders with structure:
        root/
            label1/
                image1.jpg
                image2.jpg
            label2/
                image1.jpg
        """
        
        def __init__(self, root_dir, transform=None):
            """
            Parameters:
                root_dir (str): Root directory of dataset
                transform (callable, optional): Image transform function
            """
            self.root_dir = root_dir
            self.transform = transform
            self.images = []
            self.labels = []
            
            # Scan folders, collect all image paths and labels
            for label in sorted(os.listdir(root_dir)):
                label_dir = os.path.join(root_dir, label)
                if os.path.isdir(label_dir):
                    for img_name in os.listdir(label_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                            self.images.append(os.path.join(label_dir, img_name))
                            self.labels.append(int(label))
            
            print(f"Loaded {len(self.images)} images, {len(set(self.labels))} classes")

        def __len__(self):
            """Return dataset size"""
            return len(self.images)

        def __getitem__(self, idx):
            """
            Get sample at specified index
            
            Parameters:
                idx (int): Sample index
                
            Returns:
                tuple: (image, label)
            """
            # Load image
            img_path = self.images[idx]
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            return image, label

    # Use custom dataset
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

    # Test data loading
    for images, labels in loader:
        print(f"Batch image shape: {images.shape}")
        print(f"Batch labels: {labels}")
        break
