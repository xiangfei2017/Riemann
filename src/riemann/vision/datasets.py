# BSD 3-Clause License
#
# Copyright (c) 2025, Fei Xiang
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Riemann Library Vision Module: Dataset Classes

This module provides dataset classes for commonly used computer vision datasets,
implementing data loading and preprocessing functionality compatible with the
Riemann automatic differentiation framework. These dataset classes are designed
to work seamlessly with the DataLoader from the utils.data module and provide
efficient data access for training and evaluation of computer vision models.

Main features:
    - MNIST dataset: Handwritten digit recognition dataset with IDX-UBYTE format support
    - EasyMNIST dataset: Simplified version of MNIST with basic functionality
    - CIFAR10 dataset: Object recognition dataset with image and label loading
    - Compatibility with PyTorch-style transforms and data loading pipelines
    - Efficient data loading with progress tracking and error handling

Using this module, you can easily load and preprocess standard vision datasets,
implement custom datasets for your specific needs, and integrate with the
Riemann framework for building and training computer vision models.
"""

import os
import numpy as np
import struct
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, List, Any, Dict, cast
from tqdm import tqdm
from PIL import Image
from ..tensordef import *
from ..utils.data import *
from .transforms import *


def check_md5(filepath: str, expected_md5: str) -> bool:
    """
    检查文件的MD5校验值。
    
    参数:
        filepath (str): 文件路径
        expected_md5 (str): 期望的MD5值
        
    返回:
        bool: MD5是否匹配
    """
    import hashlib
    
    if not os.path.exists(filepath):
        return False
    
    md5_hash = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    
    return md5_hash.hexdigest() == expected_md5


def download_file(url: str, filepath: str, expected_md5: Optional[str] = None) -> bool:
    """
    从URL下载文件，支持MD5校验和进度条显示。
    
    参数:
        url (str): 下载URL
        filepath (str): 保存路径
        expected_md5 (str, optional): 期望的MD5值，用于校验
        
    返回:
        bool: 是否成功下载（或文件已存在且校验通过）
        
    异常:
        RuntimeError: 下载失败时抛出
    """
    import urllib.request
    
    filename = os.path.basename(filepath)
    
    # 检查文件是否存在且完整
    if os.path.exists(filepath):
        if expected_md5 is None or check_md5(filepath, expected_md5):
            print(f"{filename} already downloaded and verified.")
            return True
        else:
            print(f"{filename} exists but MD5 mismatch, re-downloading...")
            os.remove(filepath)
    
    # 下载文件（带进度条）
    try:
        # 尝试使用tqdm显示进度条
        try:
            from tqdm import tqdm
            
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                chunk_size = 1024 * 32  # 32KB chunks
                
                with open(filepath, "wb") as fh, tqdm(
                    total=total_size, unit="B", unit_scale=True, desc=filename
                ) as pbar:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        fh.write(chunk)
                        pbar.update(len(chunk))
        except ImportError:
            # 如果没有tqdm，使用简单的urlretrieve
            urllib.request.urlretrieve(url, filepath)
                
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")
    
    return True

def extract_archive(filepath: str, extract_dir: str, archive_type: str = 'auto') -> None:
    """
    解压压缩文件。
    
    参数:
        filepath (str): 压缩文件路径
        extract_dir (str): 解压目标目录
        archive_type (str): 压缩类型，'auto'自动检测，或'gzip'、'tar'
        
    异常:
        RuntimeError: 解压失败时抛出
        ValueError: 不支持的压缩类型
    """
    import gzip
    import tarfile
    
    filename = os.path.basename(filepath)
    
    # 自动检测压缩类型
    if archive_type == 'auto':
        if filepath.endswith('.gz') and not filepath.endswith('.tar.gz'):
            archive_type = 'gzip'
        elif filepath.endswith(('.tar.gz', '.tgz')):
            archive_type = 'tar'
        else:
            raise ValueError(f"Cannot auto-detect archive type for {filename}")
    
    print(f"Extracting {filename}...")
    try:
        if archive_type == 'gzip':
            # gzip单文件解压
            extracted_file = filepath.replace('.gz', '')
            with gzip.open(filepath, 'rb') as f_in:
                with open(extracted_file, 'wb') as f_out:
                    f_out.write(f_in.read())
            print(f"Extracted {os.path.basename(extracted_file)}")
        elif archive_type == 'tar':
            # tar.gz解压
            with tarfile.open(filepath, 'r:gz') as tar:
                tar.extractall(extract_dir)
            print(f"Extracted to {extract_dir}")
        else:
            raise ValueError(f"Unsupported archive type: {archive_type}")
    except Exception as e:
        raise RuntimeError(f"Failed to extract {filepath}: {e}")


class MNIST(Dataset):
    """
    MNIST数据集类，用于加载和处理MNIST手写数字数据集。
    
    这个类支持transform和target_transform参数，
    不提供这两个参数时，数据集每行是元组(PIL图像、整型标签)。
    数据集从标准的IDX-UBYTE格式文件中加载，包括训练集和测试集。
    
    参数:
        root (str): 数据集的根目录，应包含MNIST目录，其中存放着
                   train-images-idx3-ubyte, train-labels-idx1-ubyte,
                   t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte文件
        train (bool): 如果为True，则加载训练集；如果为False，则加载测试集
        transform (callable, optional): 接收PIL图像并返回变换后图像的函数
        target_transform (callable, optional): 接收目标并返回变换后目标的函数
        download (bool, optional): 如果为True，从互联网下载数据集并放入root目录
    """
    
    # MNIST数据集资源URL和MD5校验值
    resources = [
        ("https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02620c"),
    ]
    
    # 训练集和测试集文件映射
    train_files = ("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
    test_files = ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
    
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        """
        初始化MNIST数据集。
        
        根据train参数加载相应的数据集，并处理数据格式。
        
        参数:
            root (str): 数据集的根目录
            train (bool): 是否加载训练集，默认为True
            transform (callable, optional): 应用于图像的变换函数
            target_transform (callable, optional): 应用于目标的变换函数
            download (bool, optional): 是否下载数据集，默认为False
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data_list = []
        
        # 构建MNIST数据目录路径
        mnist_dir = os.path.join(root, 'MNIST', 'raw')
        
        # 如果需要，下载数据集
        if download:
            self._download(mnist_dir)
        
        if self.train:
            # 训练集文件路径
            image_file = os.path.join(mnist_dir, self.train_files[0])
            label_file = os.path.join(mnist_dir, self.train_files[1])
        else:
            # 测试集文件路径
            image_file = os.path.join(mnist_dir, self.test_files[0])
            label_file = os.path.join(mnist_dir, self.test_files[1])
        
        # 检查文件是否存在
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"Image file does not exist: {image_file}. "
                                   f"You can set download=True to download it.")
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file does not exist: {label_file}. "
                                   f"You can set download=True to download it.")
        
        # 读取标签文件
        with open(label_file, 'rb') as f:
            # 读取文件头
            magic, n = struct.unpack('>II', f.read(8))
            if magic != 2049:
                raise ValueError(f"Invalid magic number {magic} in label file {label_file}")
            
            # 读取所有标签
            labels = np.fromfile(f, dtype=np.uint8)
        
        # 读取图像文件
        with open(image_file, 'rb') as f:
            # 读取文件头
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            if magic != 2051:
                raise ValueError(f"Invalid magic number {magic} in image file {image_file}")
            
            # 读取所有图像数据
            images = np.fromfile(f, dtype=np.uint8)
            images = images.reshape(len(labels), 28, 28)
        
        # 将图像和标签组合成数据列表，预转换为PIL Image以提高后续访问效率
        print("Loading MNIST data...")
        for img_numpy, label in tqdm(zip(images, labels), total=len(labels)):
            # 预先将numpy数组转换为PIL Image，避免在__getitem__时重复转换
            img = Image.fromarray(img_numpy, mode='L')
            self.data_list.append((img, int(label)))
    
    def _download(self, mnist_dir):
        """
        下载MNIST数据集文件。
        
        从指定的URL下载数据集文件，解压并保存到指定目录。
        如果文件已存在且MD5校验通过，则跳过下载。
        
        参数:
            mnist_dir (str): MNIST数据保存目录
        """
        # 创建目录
        os.makedirs(mnist_dir, exist_ok=True)
        
        for url, md5 in self.resources:
            filename = os.path.basename(url)
            filepath = os.path.join(mnist_dir, filename)
            extracted_file = filepath.replace('.gz', '')
            
            # 检查解压后的文件是否已存在
            if os.path.exists(extracted_file):
                print(f"{os.path.basename(extracted_file)} already exists, skipping download.")
                continue
            
            # 下载文件（使用通用工具函数）
            download_file(url, filepath, md5)
            
            # 解压文件（使用通用工具函数）
            extract_archive(filepath, mnist_dir, 'gzip')
    
    def __len__(self):
        """
        返回数据集的大小。
        
        返回:
            int: 数据集中的样本数量
        """
        return len(self.data_list)
    
    def __getitem__(self, index):
        """
        获取指定索引的样本。
        
        参数:
            index (int): 样本索引
            
        返回:
            tuple: (image, target) 图像和目标的元组
        """
        # 直接获取预转换的PIL Image和标签
        img, label = self.data_list[index]
        
        # 应用transform，如果提供的话
        if self.transform is not None:
            img = self.transform(img)
        
        # 应用target_transform，如果提供的话
        if self.target_transform is not None:
            label = self.target_transform(label)
            
        return img, label

class EasyMNIST(MNIST):
    """
    继承自MNIST的子类，在初始化时对图像数据应用归一化、标准化、展开转换，对标签作onehot编码或转换为标量张量
    这样训练过程中不再需要转换数据，可节省训练时间。
    """

    def __init__(self, root, train=True, onehot_label=True, download=False):
        """
        初始化预处理的MNIST数据集。

        参数:
            root (str): 数据集的根目录
            train (bool): 是否加载训练集，默认为True
            onehot_label (bool): 是否将标签转换为one-hot编码，默认为True
            download (bool, optional): 如果为True，从互联网下载数据集并放入root目录，默认为False
        """
        # 定义图像转换：将PIL图像转换为张量，标准化并展平
        def flatten_transform(img):
            # 将图像转换为张量
            tensor_img = ToTensor()(img)
            # 应用MNIST标准化参数
            normalized_img = Normalize((0.1307,), (0.3081,))(tensor_img)
            # 展平为一维向量
            return normalized_img.flatten()

        # 定义目标转换：将标签转换为one-hot编码的张量
        def one_hot_transform(label):
            # 创建one-hot编码的目标张量，使用标准的0和1
            target = zeros((10,), dtype=get_default_dtype())
            target[label] = 1.0
            return target

        def tensor_label_transform(label):
            return tensor(label, dtype=get_default_dtype())

        # 初始化父类，不传入转换函数，直接获取PIL Image，传入download参数
        super().__init__(root, train=train, transform=None, target_transform=None, download=download)

        # 预处理所有数据，直接更新父类的data_list
        print("Transforming MNIST to EasyMNIST ...")
        for i in tqdm(range(len(self.data_list))):
            img, label = self.data_list[i]  # 获取PIL Image和标签
            # 应用转换并更新
            if onehot_label:
                self.data_list[i] = (flatten_transform(img), one_hot_transform(label))
            else:
                self.data_list[i] = (flatten_transform(img), tensor_label_transform(label))

    # __len__和__getitem__直接继承自父类MNIST，无需重写


class FashionMNIST(MNIST):
    """
    Fashion-MNIST数据集类，用于加载和处理时尚产品图像数据集。
    
    Fashion-MNIST是MNIST的替代品，包含10个类别的时尚产品图像。
    数据集包含70000张28x28灰度图像，其中60000张用于训练，10000张用于测试。
    数据格式与MNIST完全一致，可以直接替换MNIST使用。
    
    参数:
        root (str): 数据集的根目录
        train (bool): 如果为True，则加载训练集；如果为False，则加载测试集
        transform (callable, optional): 接收PIL图像并返回变换后图像的函数
        target_transform (callable, optional): 接收目标并返回变换后目标的函数
        download (bool, optional): 如果为True，从互联网下载数据集并放入root目录
    """
    
    # Fashion-MNIST数据集资源URL和MD5校验值
    resources = [
        ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
        ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
        ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
    ]
    
    # 类别名称
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        """
        初始化Fashion-MNIST数据集。
        
        参数:
            root (str): 数据集的根目录
            train (bool): 是否加载训练集，默认为True
            transform (callable, optional): 应用于图像的变换函数
            target_transform (callable, optional): 应用于目标的变换函数
            download (bool, optional): 是否下载数据集，默认为False
        """
        # 构建FashionMNIST数据目录路径（与MNIST不同）
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data_list = []
        
        # FashionMNIST使用FashionMNIST/raw目录
        fashion_dir = os.path.join(root, 'FashionMNIST', 'raw')
        
        # 如果需要，下载数据集
        if download:
            self._download(fashion_dir)
        
        if self.train:
            # 训练集文件路径
            image_file = os.path.join(fashion_dir, self.train_files[0])
            label_file = os.path.join(fashion_dir, self.train_files[1])
        else:
            # 测试集文件路径
            image_file = os.path.join(fashion_dir, self.test_files[0])
            label_file = os.path.join(fashion_dir, self.test_files[1])
        
        # 检查文件是否存在
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"Image file does not exist: {image_file}. "
                                   f"You can set download=True to download it.")
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file does not exist: {label_file}. "
                                   f"You can set download=True to download it.")
        
        # 读取标签文件
        with open(label_file, 'rb') as f:
            # 读取文件头
            magic, n = struct.unpack('>II', f.read(8))
            if magic != 2049:
                raise ValueError(f"Invalid magic number {magic} in label file {label_file}")
            
            # 读取所有标签
            labels = np.fromfile(f, dtype=np.uint8)
        
        # 读取图像文件
        with open(image_file, 'rb') as f:
            # 读取文件头
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            if magic != 2051:
                raise ValueError(f"Invalid magic number {magic} in image file {image_file}")
            
            # 读取所有图像数据
            images = np.fromfile(f, dtype=np.uint8)
            images = images.reshape(len(labels), 28, 28)
        
        # 将图像和标签组合成数据列表，预转换为PIL Image以提高后续访问效率
        print("Loading Fashion-MNIST data...")
        for img_numpy, label in tqdm(zip(images, labels), total=len(labels)):
            # 预先将numpy数组转换为PIL Image，避免在__getitem__时重复转换
            img = Image.fromarray(img_numpy, mode='L')
            self.data_list.append((img, int(label)))
    
    def _download(self, fashion_dir):
        """
        下载Fashion-MNIST数据集文件。
        
        参数:
            fashion_dir (str): Fashion-MNIST数据保存目录
        """
        # 创建目录
        os.makedirs(fashion_dir, exist_ok=True)
        
        for url, md5 in self.resources:
            filename = os.path.basename(url)
            filepath = os.path.join(fashion_dir, filename)
            extracted_file = filepath.replace('.gz', '')
            
            # 检查解压后的文件是否已存在
            if os.path.exists(extracted_file):
                print(f"{os.path.basename(extracted_file)} already exists, skipping download.")
                continue
            
            # 下载文件（使用通用工具函数）
            download_file(url, filepath, md5)
            
            # 解压文件（使用通用工具函数）
            extract_archive(filepath, fashion_dir, 'gzip')


class CIFAR10(Dataset):
    """
    CIFAR-10数据集类，用于加载和处理CIFAR-10图像数据集。
    
    CIFAR-10数据集包含60000张32x32彩色图像，分为10个类别，每个类别6000张图像。
    其中50000张用于训练，10000张用于测试。
    
    参数:
        root (str): 数据集的根目录，应包含data_batch_1到data_batch_5、test_batch和batches.meta文件
        train (bool): 如果为True，则加载训练集；如果为False，则加载测试集
        transform (callable, optional): 接收PIL图像并返回变换后图像的函数
        target_transform (callable, optional): 接收目标并返回变换后目标的函数
        download (bool, optional): 如果为True，从互联网下载数据集并放入root目录
    """
    
    # CIFAR-10数据集资源URL和MD5校验值
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    md5 = "c58f30108f718f92721af3b95e74349a"
    
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        """
        初始化CIFAR-10数据集。
        
        根据train参数加载相应的数据集，并处理数据格式。
        
        参数:
            root (str): 数据集的根目录
            train (bool): 是否加载训练集，默认为True
            transform (callable, optional): 应用于图像的变换函数
            target_transform (callable, optional): 应用于目标的变换函数
            download (bool, optional): 是否下载数据集，默认为False
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        
        # 构建CIFAR10数据目录路径
        cifar10_dir = os.path.join(root, 'cifar-10-batches-py')
        
        # 如果需要，下载数据集
        if download:
            self._download(root)
        
        # 加载类别名称
        meta_file = os.path.join(cifar10_dir, 'batches.meta')
        if os.path.exists(meta_file):
            meta_data = self._load_batch(meta_file)
            self.classes = meta_data['label_names']
        else:
            # 默认类别名称
            self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
        
        # 预先将numpy数组转换为PIL Image，避免在__getitem__时重复转换
        # 采用流式处理：逐批次加载、转换、释放内存，避免内存翻倍
        print("Loading and converting CIFAR-10 data...")
        self.data_list = []
        
        if self.train:
            # 加载训练数据，从5个批次文件中逐批加载和转换
            for i in range(1, 6):
                batch_file = os.path.join(cifar10_dir, f'data_batch_{i}')
                if not os.path.exists(batch_file):
                    raise FileNotFoundError(f"Training batch file does not exist: {batch_file}. "
                                           f"You can set download=True to download it.")
                
                batch_data = self._load_batch(batch_file)
                batch_images = batch_data['data']  # (10000, 3072)
                batch_labels = batch_data['labels']  # list of 10000 ints
                
                # 立即转换当前批次，不保留原始numpy数据
                for j in tqdm(range(len(batch_images)), desc=f"Converting batch {i}/5", leave=False):
                    img_array = batch_images[j].reshape(3, 32, 32).transpose(1, 2, 0)
                    img = Image.fromarray(img_array, mode='RGB')
                    self.data_list.append((img, int(batch_labels[j])))
                
                # 显式删除批次数据，允许垃圾回收
                del batch_data, batch_images, batch_labels
                print(f"Loaded and converted {os.path.basename(batch_file)}")
        else:
            # 加载测试数据
            test_file = os.path.join(cifar10_dir, 'test_batch')
            if not os.path.exists(test_file):
                raise FileNotFoundError(f"Test batch file does not exist: {test_file}. "
                                       f"You can set download=True to download it.")
            
            batch_data = self._load_batch(test_file)
            batch_images = batch_data['data']
            batch_labels = batch_data['labels']
            
            # 立即转换
            for j in tqdm(range(len(batch_images)), desc="Converting test batch", leave=False):
                img_array = batch_images[j].reshape(3, 32, 32).transpose(1, 2, 0)
                img = Image.fromarray(img_array, mode='RGB')
                self.data_list.append((img, int(batch_labels[j])))
            
            del batch_data, batch_images, batch_labels
            print(f"Loaded and converted {os.path.basename(test_file)}")
    
    def _download(self, root):
        """
        下载CIFAR-10数据集文件。
        
        从指定的URL下载数据集文件，解压并保存到指定目录。
        如果文件已存在且MD5校验通过，则跳过下载。
        
        参数:
            root (str): 数据集保存的根目录
        """
        # 创建目录
        os.makedirs(root, exist_ok=True)
        
        # 检查数据目录是否已存在
        cifar10_dir = os.path.join(root, 'cifar-10-batches-py')
        if os.path.exists(cifar10_dir):
            # 检查关键文件是否存在
            required_files = ['batches.meta', 'test_batch'] + [f'data_batch_{i}' for i in range(1, 6)]
            if all(os.path.exists(os.path.join(cifar10_dir, f)) for f in required_files):
                print(f"CIFAR-10 data already exists, skipping download.")
                return
        
        filepath = os.path.join(root, self.filename)
        
        # 下载文件（使用通用工具函数）
        download_file(self.url, filepath, self.md5)
        
        # 解压文件（使用通用工具函数）
        extract_archive(filepath, root, 'tar')
    
    def _load_batch(self, file_path):
        """
        加载CIFAR-10批次文件。
        
        参数:
            file_path (str): 批次文件路径
            
        返回:
            dict: 包含数据和标签的字典
        """
        import pickle
        with open(file_path, 'rb') as f:
            # 兼容Python 2和3的pickle加载
            try:
                batch_dict = pickle.load(f, encoding='latin1')
            except (UnicodeDecodeError, TypeError):
                batch_dict = pickle.load(f)
        
        return batch_dict
    
    def __len__(self):
        """
        返回数据集的大小。
        
        返回:
            int: 数据集中的样本数量
        """
        return len(self.data_list)
    
    def __getitem__(self, index):
        """
        获取指定索引的样本。
        
        参数:
            index (int): 样本索引
            
        返回:
            tuple: (image, target) 图像和目标的元组
        """
        # 直接获取预转换的PIL Image和标签
        img, target = self.data_list[index]
        
        # 应用transform，如果提供的话
        if self.transform is not None:
            img = self.transform(img)
        
        # 应用target_transform，如果提供的话
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target


class Flowers102(Dataset):
    """
    Oxford 102 Flower 数据集类。
    
    Oxford 102 Flower 是一个图像分类数据集，包含102种花卉类别。
    这些花卉选自英国常见的花卉。每个类别包含40到258张图像。
    
    图像具有较大的尺度、姿态和光照变化。此外，有些类别内部差异很大，
    且存在几个非常相似的类别。
    
    参数:
        root (str): 数据集的根目录
        split (str, optional): 数据集划分，支持 "train" (默认)、"val"、"test"
        transform (callable, optional): 应用于图像的变换函数
        target_transform (callable, optional): 应用于目标的变换函数
        download (bool, optional): 如果为True，从互联网下载数据集
    
    注意:
        此类需要 scipy 来从 .mat 格式加载目标文件。
    """
    
    # 数据集资源URL和MD5校验值
    _download_url_prefix = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
    _file_dict = {
        "image": ("102flowers.tgz", "52808999861908f626f3c1f4e79d11fa"),
        "label": ("imagelabels.mat", "e0620be6f572b9609742df49c70aed4d"),
        "setid": ("setid.mat", "a5357ecc9cb78c4bef273ce3793fc85c"),
    }
    _splits_map = {"train": "trnid", "val": "valid", "test": "tstid"}
    
    def __init__(self, root, split="train", transform=None, target_transform=None, download=False):
        """
        初始化 Flowers102 数据集。
        
        参数:
            root (str): 数据集的根目录
            split (str): 数据集划分，默认为 "train"
            transform (callable, optional): 应用于图像的变换函数
            target_transform (callable, optional): 应用于目标的变换函数
            download (bool, optional): 是否下载数据集，默认为False
        """
        import scipy.io as sio
        
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # 验证split参数
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}. Must be one of ('train', 'val', 'test')")
        
        # 构建数据目录路径
        self._base_folder = os.path.join(root, "flowers-102")
        self._images_folder = os.path.join(self._base_folder, "jpg")
        
        # 如果需要，下载数据集
        if download:
            self._download()
        
        # 检查数据完整性
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")
        
        # 加载数据集划分信息
        set_ids = sio.loadmat(os.path.join(self._base_folder, self._file_dict["setid"][0]), squeeze_me=True)
        image_ids = set_ids[self._splits_map[split]].tolist()
        
        # 加载标签信息
        labels = sio.loadmat(os.path.join(self._base_folder, self._file_dict["label"][0]), squeeze_me=True)
        image_id_to_label = dict(enumerate((labels["labels"] - 1).tolist(), 1))
        
        # 构建图像文件路径和标签列表
        self._labels = []
        self._image_files = []
        for image_id in image_ids:
            self._labels.append(image_id_to_label[image_id])
            self._image_files.append(os.path.join(self._images_folder, f"image_{image_id:05d}.jpg"))
    
    def _check_integrity(self):
        """检查数据完整性"""
        if not os.path.isdir(self._images_folder):
            return False
        
        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            filepath = os.path.join(self._base_folder, filename)
            if not check_md5(filepath, md5):
                return False
        return True
    
    def _download(self):
        """下载数据集"""
        if self._check_integrity():
            return
        
        # 创建基础目录
        os.makedirs(self._base_folder, exist_ok=True)
        
        # 下载并解压图像文件
        image_file, image_md5 = self._file_dict["image"]
        image_url = f"{self._download_url_prefix}{image_file}"
        image_path = os.path.join(self._base_folder, image_file)
        
        download_file(image_url, image_path, image_md5)
        extract_archive(image_path, self._base_folder, 'tar')
        
        # 下载标签和划分文件
        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            url = f"{self._download_url_prefix}{filename}"
            filepath = os.path.join(self._base_folder, filename)
            download_file(url, filepath, md5)
    
    def __len__(self):
        """返回数据集大小"""
        return len(self._image_files)
    
    def __getitem__(self, idx):
        """获取指定索引的样本"""
        image_file, label = self._image_files[idx], self._labels[idx]
        image = Image.open(image_file).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label


class OxfordIIITPet(Dataset):
    """
    Oxford-IIIT Pet 数据集类。
    
    Oxford-IIIT Pet 数据集包含37种宠物（猫和狗）品种的高质量图像。
    数据集包含超过7000张图像，每个类别约200张图像。
    
    参数:
        root (str): 数据集的根目录
        split (str, optional): 数据集划分，支持 "trainval" (默认) 或 "test"
        target_types (str or list, optional): 目标类型，可以是 "category" (默认)、
            "binary-category" 或 "segmentation"，也可以是列表
        transform (callable, optional): 应用于图像的变换函数
        target_transform (callable, optional): 应用于目标的变换函数
        download (bool, optional): 如果为True，从互联网下载数据集
    """
    
    # 数据集资源URL和MD5校验值
    _RESOURCES = (
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f"),
    )
    _VALID_TARGET_TYPES = ("category", "binary-category", "segmentation")
    
    def __init__(self, root, split="trainval", target_types="category", 
                 transform=None, target_transform=None, download=False):
        """
        初始化 OxfordIIITPet 数据集。
        
        参数:
            root (str): 数据集的根目录
            split (str): 数据集划分，默认为 "trainval"
            target_types (str or list): 目标类型，默认为 "category"
            transform (callable, optional): 应用于图像的变换函数
            target_transform (callable, optional): 应用于目标的变换函数
            download (bool, optional): 是否下载数据集，默认为False
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # 验证split参数
        if split not in ("trainval", "test"):
            raise ValueError(f"Invalid split: {split}. Must be one of ('trainval', 'test')")
        
        # 处理target_types
        if isinstance(target_types, str):
            target_types = [target_types]
        self._target_types = []
        for target_type in target_types:
            if target_type not in self._VALID_TARGET_TYPES:
                raise ValueError(f"Invalid target_type: {target_type}")
            self._target_types.append(target_type)
        
        # 构建数据目录路径
        self._base_folder = os.path.join(root, "oxford-iiit-pet")
        self._images_folder = os.path.join(self._base_folder, "images")
        self._anns_folder = os.path.join(self._base_folder, "annotations")
        self._segs_folder = os.path.join(self._anns_folder, "trimaps")
        
        # 如果需要，下载数据集
        if download:
            self._download()
        
        # 检查数据是否存在
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")
        
        # 加载图像ID和标签
        image_ids = []
        self._labels = []
        self._bin_labels = []
        
        with open(os.path.join(self._anns_folder, f"{self.split}.txt"), 'r') as f:
            for line in f:
                parts = line.strip().split()
                image_id = parts[0]
                label = int(parts[1]) - 1  # 转换为0-based
                bin_label = int(parts[2]) - 1  # 转换为0-based
                image_ids.append(image_id)
                self._labels.append(label)
                self._bin_labels.append(bin_label)
        
        # 构建类别名称列表
        self.bin_classes = ["Cat", "Dog"]
        
        # 从图像ID中提取类别名称
        raw_classes = {}
        for image_id, label in zip(image_ids, self._labels):
            class_name = " ".join(part.title() for part in image_id.rsplit("_", 1)[0].split("_"))
            raw_classes[label] = class_name
        
        # 按标签排序构建类别列表
        self.classes = [raw_classes[i] for i in sorted(raw_classes.keys())]
        self.bin_class_to_idx = dict(zip(self.bin_classes, range(len(self.bin_classes))))
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        
        # 构建图像和分割掩码路径列表
        self._images = [os.path.join(self._images_folder, f"{image_id}.jpg") for image_id in image_ids]
        self._segs = [os.path.join(self._segs_folder, f"{image_id}.png") for image_id in image_ids]
    
    def _check_exists(self):
        """检查数据是否存在"""
        return (os.path.isdir(self._meta_folder) if hasattr(self, '_meta_folder') else True) and \
               os.path.isdir(self._images_folder) and \
               os.path.isdir(self._anns_folder)
    
    def _download(self):
        """下载数据集"""
        if self._check_exists():
            return
        
        # 创建基础目录
        os.makedirs(self._base_folder, exist_ok=True)
        
        # 下载并解压图像文件
        url, md5 = self._RESOURCES[0]  # images.tar.gz
        filename = os.path.basename(url)
        filepath = os.path.join(self._base_folder, filename)
        download_file(url, filepath, md5)
        extract_archive(filepath, self._base_folder, 'tar')
        
        # 下载并解压标注文件
        url, md5 = self._RESOURCES[1]  # annotations.tar.gz
        filename = os.path.basename(url)
        filepath = os.path.join(self._base_folder, filename)
        download_file(url, filepath, md5)
        extract_archive(filepath, self._base_folder, 'tar')
    
    def __len__(self):
        """返回数据集大小"""
        return len(self._images)
    
    def __getitem__(self, idx):
        """获取指定索引的样本"""
        image = Image.open(self._images[idx]).convert("RGB")
        
        target = []
        for target_type in self._target_types:
            if target_type == "category":
                target.append(self._labels[idx])
            elif target_type == "binary-category":
                target.append(self._bin_labels[idx])
            else:  # segmentation
                target.append(Image.open(self._segs[idx]))
        
        if not target:
            target = None
        elif len(target) == 1:
            target = target[0]
        else:
            target = tuple(target)
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            target = self.target_transform(target)
        
        return image, target


class LFWPeople(Dataset):
    """
    LFW (Labeled Faces in the Wild) People 数据集类。
    
    LFW 数据集包含从互联网上收集的人脸图像，用于人脸识别研究。
    数据集包含超过13,000张从网络上收集的人脸图像。
    
    参数:
        root (str): 数据集的根目录
        split (str, optional): 数据集划分，支持 "train"、"test"、"10fold" (默认)
        image_set (str, optional): 图像对齐类型，支持 "original"、"funneled" (默认)、"deepfunneled"
        transform (callable, optional): 应用于图像的变换函数
        target_transform (callable, optional): 应用于目标的变换函数
        download (bool, optional): 如果为True，从互联网下载数据集
    """
    
    # 数据集基础目录
    base_folder = "lfw-py"
    download_url_prefix = "http://vis-www.cs.umass.edu/lfw/"
    
    # 图像文件字典: (images_dir, filename, md5)
    file_dict = {
        "original": ("lfw", "lfw.tgz", "a17d05bd522c52d84eca14327a23d494"),
        "funneled": ("lfw_funneled", "lfw-funneled.tgz", "1b42dfed7d15c9b2dd63d5e5840c86ad"),
        "deepfunneled": ("lfw-deepfunneled", "lfw-deepfunneled.tgz", "68331da3eb755a505a502b5aacb3c201"),
    }
    
    # 校验文件MD5
    checksums = {
        "pairs.txt": "9f1ba174e4e1c508ff7cdf10ac338a7d",
        "pairsDevTest.txt": "5132f7440eb68cf58910c8a45a2ac10b",
        "pairsDevTrain.txt": "4f27cbf15b2da4a85c1907eb4181ad21",
        "people.txt": "450f0863dd89e85e73936a6d71a3474b",
        "peopleDevTest.txt": "e4bf5be0a43b5dcd9dc5ccfcb8fb19c5",
        "peopleDevTrain.txt": "54eaac34beb6d042ed3a7d883e247a21",
        "lfw-names.txt": "a6d0a479bd074669f656265a6e693f6d",
    }
    
    # 划分对应的标注文件后缀
    annot_file = {"10fold": "", "train": "DevTrain", "test": "DevTest"}
    names = "lfw-names.txt"
    
    def __init__(self, root, split="10fold", image_set="funneled", 
                 transform=None, target_transform=None, download=False):
        """
        初始化 LFWPeople 数据集。
        
        参数:
            root (str): 数据集的根目录
            split (str): 数据集划分，默认为 "10fold"
            image_set (str): 图像对齐类型，默认为 "funneled"
            transform (callable, optional): 应用于图像的变换函数
            target_transform (callable, optional): 应用于目标的变换函数
            download (bool, optional): 是否下载数据集，默认为False
        """
        self.root = os.path.join(root, self.base_folder)
        self.split = split
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        
        # 验证参数
        if split not in ("10fold", "train", "test"):
            raise ValueError(f"Invalid split: {split}")
        if image_set not in self.file_dict:
            raise ValueError(f"Invalid image_set: {image_set}")
        
        # 获取图像目录和文件名
        images_dir, self.filename, self.md5 = self.file_dict[image_set]
        self.images_dir = os.path.join(self.root, images_dir)
        
        # 标注文件
        self.labels_file = f"people{self.annot_file[split]}.txt"
        
        # 如果需要，下载数据集
        if download:
            self.download()
        
        # 检查数据完整性
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")
        
        # 加载数据
        self.data = []
        self.class_to_idx = {}
        self.classes = []
        
        with open(os.path.join(self.root, self.labels_file), 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                person_name = parts[0]
                num_images = int(parts[1])
                
                # 为每个人分配一个类别索引
                if person_name not in self.class_to_idx:
                    self.class_to_idx[person_name] = len(self.classes)
                    self.classes.append(person_name)
                
                label = self.class_to_idx[person_name]
                
                # 添加每个人的所有图像
                for i in range(1, num_images + 1):
                    img_path = os.path.join(self.images_dir, person_name, f"{person_name}_{i:04d}.jpg")
                    if os.path.exists(img_path):
                        self.data.append((img_path, label))
    
    def _check_integrity(self):
        """检查数据完整性"""
        # 检查压缩包文件MD5
        if not check_md5(os.path.join(self.root, self.filename), self.md5):
            return False
        
        # 检查标注文件MD5
        if not check_md5(os.path.join(self.root, self.labels_file), self.checksums[self.labels_file]):
            return False
        
        return check_md5(os.path.join(self.root, self.names), self.checksums[self.names])
    
    def download(self):
        """下载数据集"""
        if self._check_integrity():
            return
        
        # 创建根目录
        os.makedirs(self.root, exist_ok=True)
        
        # 下载并解压图像文件
        url = f"{self.download_url_prefix}{self.filename}"
        filepath = os.path.join(self.root, self.filename)
        download_file(url, filepath, self.md5)
        extract_archive(filepath, self.root, 'tar')
        
        # 下载标注文件
        labels_url = f"{self.download_url_prefix}{self.labels_file}"
        labels_path = os.path.join(self.root, self.labels_file)
        download_file(labels_url, labels_path, self.checksums[self.labels_file])
        
        # 下载names文件
        names_url = f"{self.download_url_prefix}{self.names}"
        names_path = os.path.join(self.root, self.names)
        download_file(names_url, names_path, self.checksums[self.names])
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取指定索引的样本"""
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label


class CIFAR100(CIFAR10):
    """
    CIFAR-100数据集类，用于加载和处理CIFAR-100图像数据集。
    
    CIFAR-100数据集包含60000张32x32彩色图像，分为100个类别，每个类别600张图像。
    其中50000张用于训练，10000张用于测试。
    
    与CIFAR-10不同，CIFAR-100有100个细分类别和20个超类。
    
    参数:
        root (str): 数据集的根目录
        train (bool): 如果为True，则加载训练集；如果为False，则加载测试集
        transform (callable, optional): 接收PIL图像并返回变换后图像的函数
        target_transform (callable, optional): 接收目标并返回变换后目标的函数
        download (bool, optional): 如果为True，从互联网下载数据集并放入root目录
        coarse (bool, optional): 如果为True，使用20个超类标签；否则使用100个细分类别标签
    """
    
    # CIFAR-100数据集资源URL和MD5校验值
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, coarse=False):
        """
        初始化CIFAR-100数据集。
        
        参数:
            root (str): 数据集的根目录
            train (bool): 是否加载训练集，默认为True
            transform (callable, optional): 应用于图像的变换函数
            target_transform (callable, optional): 应用于目标的变换函数
            download (bool, optional): 是否下载数据集，默认为False
            coarse (bool, optional): 是否使用超类标签，默认为False
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.coarse = coarse
        
        # 构建CIFAR100数据目录路径
        cifar100_dir = os.path.join(root, 'cifar-100-python')
        
        # 如果需要，下载数据集
        if download:
            self._download_cifar100(root)
        
        # 加载类别名称
        meta_file = os.path.join(cifar100_dir, 'meta')
        if os.path.exists(meta_file):
            meta_data = self._load_batch(meta_file)
            self.classes = meta_data['fine_label_names'] if not coarse else meta_data['coarse_label_names']
        else:
            self.classes = None
        
        # 预先将numpy数组转换为PIL Image
        print("Loading and converting CIFAR-100 data...")
        self.data_list = []
        
        if self.train:
            # 加载训练数据
            train_file = os.path.join(cifar100_dir, 'train')
            if not os.path.exists(train_file):
                raise FileNotFoundError(f"Training file does not exist: {train_file}. "
                                       f"You can set download=True to download it.")
            
            batch_data = self._load_batch(train_file)
            batch_images = batch_data['data']
            # 根据coarse参数选择标签类型
            batch_labels = batch_data['coarse_labels'] if coarse else batch_data['fine_labels']
            
            # 立即转换
            for j in tqdm(range(len(batch_images)), desc="Converting train data", leave=False):
                img_array = batch_images[j].reshape(3, 32, 32).transpose(1, 2, 0)
                img = Image.fromarray(img_array, mode='RGB')
                self.data_list.append((img, int(batch_labels[j])))
            
            del batch_data, batch_images, batch_labels
            print(f"Loaded and converted train data")
        else:
            # 加载测试数据
            test_file = os.path.join(cifar100_dir, 'test')
            if not os.path.exists(test_file):
                raise FileNotFoundError(f"Test file does not exist: {test_file}. "
                                       f"You can set download=True to download it.")
            
            batch_data = self._load_batch(test_file)
            batch_images = batch_data['data']
            # 根据coarse参数选择标签类型
            batch_labels = batch_data['coarse_labels'] if coarse else batch_data['fine_labels']
            
            # 立即转换
            for j in tqdm(range(len(batch_images)), desc="Converting test data", leave=False):
                img_array = batch_images[j].reshape(3, 32, 32).transpose(1, 2, 0)
                img = Image.fromarray(img_array, mode='RGB')
                self.data_list.append((img, int(batch_labels[j])))
            
            del batch_data, batch_images, batch_labels
            print(f"Loaded and converted test data")
    
    def _download_cifar100(self, root):
        """
        下载CIFAR-100数据集文件。
        
        参数:
            root (str): 数据集保存的根目录
        """
        # 创建目录
        os.makedirs(root, exist_ok=True)
        
        # 检查数据目录是否已存在
        cifar100_dir = os.path.join(root, 'cifar-100-python')
        if os.path.exists(cifar100_dir):
            # 检查关键文件是否存在
            required_files = ['meta', 'test', 'train']
            if all(os.path.exists(os.path.join(cifar100_dir, f)) for f in required_files):
                print(f"CIFAR-100 data already exists, skipping download.")
                return
        
        filepath = os.path.join(root, self.filename)
        
        # 下载文件（使用通用工具函数）
        download_file(self.url, filepath, self.md5)
        
        # 解压文件（使用通用工具函数）
        extract_archive(filepath, root, 'tar')


class DatasetFolder(Dataset):
    """
    通用数据文件夹数据集类，参照 torchvision.datasets.DatasetFolder 实现。

    从目录结构中加载数据，目录结构应为：
    root/class_x/xxx.ext
    root/class_x/xxy.ext
    root/class_x/xxz.ext
    root/class_y/123.ext
    root/class_y/nsdf3.ext
    root/class_y/asd932_.ext

    参数:
        root (str or Path): 根目录路径
        loader (callable): 加载样本文件的函数，接收文件路径作为输入
        extensions (tuple, optional): 允许的文件扩展名元组，如 ('.jpg', '.jpeg', '.png')
        transform (callable, optional): 应用于样本的变换函数
        target_transform (callable, optional): 应用于目标的变换函数
        is_valid_file (callable, optional): 判断文件是否有效的函数，接收文件路径作为输入
        allow_empty (bool, optional): 是否允许空文件夹，默认为False
    """

    def __init__(
        self,
        root: Union[str, Path],
        loader: Callable[[str], Any],
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
    ):
        """
        初始化DatasetFolder数据集。

        参数:
            root: 根目录路径
            loader: 加载样本文件的函数
            extensions: 允许的文件扩展名元组
            transform: 应用于样本的变换函数
            target_transform: 应用于目标的变换函数
            is_valid_file: 判断文件是否有效的函数
            allow_empty: 是否允许空文件夹
        """
        super().__init__()

        self.root = str(root)
        self.loader = loader
        self.extensions = extensions
        self.transform = transform
        self.target_transform = target_transform
        self.is_valid_file = is_valid_file
        self.allow_empty = allow_empty

        # 验证参数
        if extensions is not None and is_valid_file is not None:
            raise ValueError("Both extensions and is_valid_file cannot be specified at the same time")

        # 构建is_valid_file函数
        if is_valid_file is None:
            if extensions is not None:
                def is_valid_file(x: str) -> bool:
                    return self.has_file_allowed_extension(x, cast(Tuple[str, ...], self.extensions))
            else:
                # 如果没有指定extensions和is_valid_file，则接受所有文件
                def is_valid_file(x: str) -> bool:
                    return True

        self.is_valid_file = is_valid_file

        # 加载样本
        self.samples, self.class_to_idx = self.make_dataset(
            self.root, self.is_valid_file, self.allow_empty
        )

        if len(self.samples) == 0 and not self.allow_empty:
            msg = f"Found 0 files in subfolders of: {self.root}\n"
            if self.extensions is not None:
                msg += f"Supported extensions are: {', '.join(self.extensions)}"
            raise RuntimeError(msg)

        self.targets = [s[1] for s in self.samples]
        self.classes = list(self.class_to_idx.keys())

    @staticmethod
    def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
        """
        检查文件是否具有允许的扩展名。

        参数:
            filename: 文件名
            extensions: 允许的扩展名或扩展名元组

        返回:
            bool: 如果文件具有允许的扩展名则返回True
        """
        return filename.lower().endswith(extensions)

    @staticmethod
    def make_dataset(
        directory: str,
        is_valid_file: Callable[[str], bool],
        allow_empty: bool = False,
    ) -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
        """
        从目录中创建数据集。

        参数:
            directory: 根目录路径
            is_valid_file: 判断文件是否有效的函数
            allow_empty: 是否允许空文件夹

        返回:
            tuple: (samples, class_to_idx)
                - samples: 样本列表，每个元素是 (path, class_index) 元组
                - class_to_idx: 类名到索引的映射字典
        """
        directory = os.path.expanduser(directory)

        # 检查目录是否存在
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        # 获取所有类别（子目录）
        classes = sorted([d.name for d in os.scandir(directory) if d.is_dir()])

        if not allow_empty and len(classes) == 0:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}")

        # 创建类名到索引的映射
        class_to_idx: Dict[str, int] = {cls_name: i for i, cls_name in enumerate(classes)}

        # 收集所有样本
        samples: List[Tuple[str, int]] = []
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)

            if not os.path.isdir(target_dir):
                continue

            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        samples.append((path, class_index))

        return samples, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        获取指定索引的样本。

        参数:
            index: 样本索引

        返回:
            tuple: (sample, target) 样本和目标类别的元组
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        """
        返回数据集的大小。

        返回:
            int: 数据集中的样本数量
        """
        return len(self.samples)

    def __repr__(self) -> str:
        """
        返回数据集的字符串表示。

        返回:
            str: 数据集的描述信息
        """
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Root location: {self.root}")
        if self.extensions is not None:
            body.append(f"Extensions: {self.extensions}")
        body.append(f"Number of classes: {len(self.classes)}")
        lines = [head] + ["    " + line for line in body]
        return "\n".join(lines)


def default_loader(path: str) -> Any:
    """
    默认的图像加载器，使用PIL打开图像。

    参数:
        path: 图像文件路径

    返回:
        PIL.Image: 打开的图像
    """
    from PIL import Image
    return Image.open(path).convert('RGB')


class ImageFolder(DatasetFolder):
    """
    图像文件夹数据集类，继承自DatasetFolder，参照 torchvision.datasets.ImageFolder 实现。

    从目录结构中加载图像数据，目录结构应为：
    root/dog/xxx.png
    root/dog/xxy.png
    root/dog/xxz.png
    root/cat/123.png
    root/cat/nsdf3.png
    root/cat/asd932_.png

    参数:
        root (str or Path): 根目录路径
        transform (callable, optional): 应用于图像的变换函数
        target_transform (callable, optional): 应用于目标的变换函数
        loader (callable, optional): 加载图像的函数，默认为default_loader
        is_valid_file (callable, optional): 判断文件是否有效的函数
        allow_empty (bool, optional): 是否允许空文件夹，默认为False
    """

    # 默认支持的图像扩展名
    IMG_EXTENSIONS: Tuple[str, ...] = (
        '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'
    )

    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
    ):
        """
        初始化ImageFolder数据集。

        参数:
            root: 根目录路径
            transform: 应用于图像的变换函数
            target_transform: 应用于目标的变换函数
            loader: 加载图像的函数
            is_valid_file: 判断文件是否有效的函数
            allow_empty: 是否允许空文件夹
        """
        super().__init__(
            root=root,
            loader=loader,
            extensions=self.IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
        )

    def __repr__(self) -> str:
        """
        返回数据集的字符串表示。

        返回:
            str: 数据集的描述信息
        """
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Root location: {self.root}")
        body.append(f"Number of classes: {len(self.classes)}")
        lines = [head] + ["    " + line for line in body]
        return "\n".join(lines)


class SVHN(Dataset):
    """
    SVHN (Street View House Numbers) 数据集类。
    
    SVHN数据集包含从Google街景图像中裁剪出的房屋门牌号图像。
    数据集包含10个数字类别（0-9），图像大小为32x32彩色图像。
    
    注意：原始SVHN数据集中标签10表示数字0，但在这个实现中，
    我们将标签0映射到数字0，以兼容PyTorch的损失函数（期望标签范围[0, C-1]）。
    
    参数:
        root (str): 数据集的根目录
        split (str): 数据集划分，'train'、'test'或'extra'
        transform (callable, optional): 接收PIL图像并返回变换后图像的函数
        target_transform (callable, optional): 接收目标并返回变换后目标的函数
        download (bool, optional): 如果为True，从互联网下载数据集并放入root目录
    """
    
    # SVHN数据集资源URL和MD5校验值
    url = "http://ufldl.stanford.edu/housenumbers/"
    filename = "{}_32x32.mat"
    files = {
        'train': ("train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"),
        'test': ("test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"),
        'extra': ("extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"),
    }
    
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        """
        初始化SVHN数据集。
        
        参数:
            root (str): 数据集的根目录
            split (str): 数据集划分，'train'、'test'或'extra'，默认为'train'
            transform (callable, optional): 应用于图像的变换函数
            target_transform (callable, optional): 应用于目标的变换函数
            download (bool, optional): 是否下载数据集，默认为False
        """
        import scipy.io as sio
        
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # 验证split参数
        if split not in self.files:
            raise ValueError(f"Invalid split: {split}. Must be one of {list(self.files.keys())}")
        
        # 构建SVHN数据目录路径
        svhn_dir = os.path.join(root, 'SVHN')
        
        # 如果需要，下载数据集
        if download:
            self._download(svhn_dir)
        
        # 加载数据文件
        filename, _ = self.files[split]
        filepath = os.path.join(svhn_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"SVHN data file does not exist: {filepath}. "
                                   f"You can set download=True to download it.")
        
        # 加载.mat文件
        print(f"Loading SVHN {split} data...")
        mat_data = sio.loadmat(filepath)
        
        # 提取图像和标签
        # SVHN的图像格式是 (N, 32, 32, 3)，需要转换
        images = mat_data['X']  # (32, 32, 3, N)
        labels = mat_data['y'].flatten()  # (N,)
        
        # 转换图像格式从 (H, W, C, N) 到 (N, H, W, C)
        images = images.transpose(3, 0, 1, 2)
        
        # 处理标签：原始SVHN中标签10表示数字0，我们将其转换为0
        labels = labels.copy()
        labels[labels == 10] = 0
        
        # 预先将numpy数组转换为PIL Image
        print("Converting SVHN data...")
        self.data_list = []
        for i in tqdm(range(len(images)), leave=False):
            img_array = images[i]
            img = Image.fromarray(img_array, mode='RGB')
            self.data_list.append((img, int(labels[i])))
        
        print(f"Loaded {len(self.data_list)} SVHN {split} samples")
    
    def _download(self, svhn_dir):
        """
        下载SVHN数据集文件。
        
        参数:
            svhn_dir (str): SVHN数据保存目录
        """
        # 创建目录
        os.makedirs(svhn_dir, exist_ok=True)
        
        filename, md5 = self.files[self.split]
        filepath = os.path.join(svhn_dir, filename)
        
        # 检查文件是否已存在
        if os.path.exists(filepath):
            if check_md5(filepath, md5):
                print(f"{filename} already downloaded and verified.")
                return
            else:
                print(f"{filename} exists but MD5 mismatch, re-downloading...")
                os.remove(filepath)
        
        # 下载文件
        url = self.url + filename
        download_file(url, filepath, md5)
    
    def __len__(self):
        """
        返回数据集的大小。
        
        返回:
            int: 数据集中的样本数量
        """
        return len(self.data_list)
    
    def __getitem__(self, index):
        """
        获取指定索引的样本。
        
        参数:
            index (int): 样本索引
            
        返回:
            tuple: (image, target) 图像和目标的元组
        """
        # 直接获取预转换的PIL Image和标签
        img, target = self.data_list[index]
        
        # 应用transform，如果提供的话
        if self.transform is not None:
            img = self.transform(img)
        
        # 应用target_transform，如果提供的话
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target

