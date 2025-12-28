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
from tqdm import tqdm
from ..tensordef import *
from ..utils.data import *
from .transforms import *

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
    """
    
    def __init__(self, root, train=True, transform=None, target_transform=None):
        """
        初始化MNIST数据集。
        
        根据train参数加载相应的数据集，并处理数据格式。
        
        参数:
            root (str): 数据集的根目录
            train (bool): 是否加载训练集，默认为True
            transform (callable, optional): 应用于图像的变换函数
            target_transform (callable, optional): 应用于目标的变换函数
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data_list = []
        
        # 构建MNIST数据目录路径
        mnist_dir = os.path.join(root, 'MNIST','raw')
        
        if self.train:
            # 训练集文件路径
            image_file = os.path.join(mnist_dir, 'train-images-idx3-ubyte')
            label_file = os.path.join(mnist_dir, 'train-labels-idx1-ubyte')
        else:
            # 测试集文件路径
            image_file = os.path.join(mnist_dir, 't10k-images-idx3-ubyte')
            label_file = os.path.join(mnist_dir, 't10k-labels-idx1-ubyte')
        
        # 检查文件是否存在
        if not os.path.exists(image_file):
            raise FileNotFoundError(f"Image file does not exist: {image_file}")
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file does not exist: {label_file}")
        
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
        
        # 将图像和标签组合成数据列表
        print("Loading MNIST data...")
        for img, label in tqdm(zip(images, labels), total=len(labels)):
            self.data_list.append((img, label))
    
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
        # 获取存储的numpy数组和标签
        img_numpy, label = self.data_list[index]
        
        # 将numpy数组转换为PIL图像
        from PIL import Image
        img = Image.fromarray(img_numpy, mode='L')
        
        # 将标签从numpy.uint8转换为Python的int类型
        label = int(label)
        
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
    
    def __init__(self, root, train=True, onehot_label=True):
        """
        初始化预处理的MNIST数据集。
        
        参数:
            root (str): 数据集的根目录
            train (bool): 是否加载训练集，默认为True
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
        
        # 初始化父类，传入转换函数
        if onehot_label:
            super().__init__(root, train=train, transform=flatten_transform, target_transform=one_hot_transform)
        else:
            super().__init__(root, train=train, transform=flatten_transform, target_transform=tensor_label_transform)
        
        # 预处理所有数据
        print("Transforming MNIST to EasyMNIST ...")
        for i in tqdm(range(super().__len__())):
            # 通过父类的__getitem__获取转换后的数据
            self.data_list[i] = super().__getitem__(i)
    
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
            tuple: (image, target) 预处理后的图像张量和目标张量的元组
        """
        return self.data_list[index]

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
    """
    
    def __init__(self, root, train=True, transform=None, target_transform=None):
        """
        初始化CIFAR-10数据集。
        
        根据train参数加载相应的数据集，并处理数据格式。
        
        参数:
            root (str): 数据集的根目录
            train (bool): 是否加载训练集，默认为True
            transform (callable, optional): 应用于图像的变换函数
            target_transform (callable, optional): 应用于目标的变换函数
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        
        # 构建CIFAR10数据目录路径
        cifar10_dir = os.path.join(root, 'cifar-10-batches-py')
        
        # 加载数据
        self.data = []  # 存储图像数据
        self.targets = []  # 存储标签
        
        if self.train:
            # 加载训练数据，从5个批次文件中加载
            for i in range(1, 6):
                batch_file = os.path.join(cifar10_dir, f'data_batch_{i}')
                if not os.path.exists(batch_file):
                    raise FileNotFoundError(f"Training batch file does not exist: {batch_file}")
                
                batch_data = self._load_batch(batch_file)
                self.data.append(batch_data['data'])
                self.targets.extend(batch_data['labels'])
                
                print(f"Loading {os.path.basename(batch_file)}...")
        else:
            # 加载测试数据
            test_file = os.path.join(cifar10_dir, 'test_batch')
            if not os.path.exists(test_file):
                raise FileNotFoundError(f"Test batch file does not exist: {test_file}")
            
            batch_data = self._load_batch(test_file)
            self.data.append(batch_data['data'])
            self.targets.extend(batch_data['labels'])
            
            print(f"Loading {os.path.basename(test_file)}...")
        
        # 合并所有批次数据
        self.data = np.concatenate(self.data, axis=0)
        self.targets = np.array(self.targets)
        
        # 加载类别名称
        meta_file = os.path.join(cifar10_dir, 'batches.meta')
        if os.path.exists(meta_file):
            meta_data = self._load_batch(meta_file)
            self.classes = meta_data['label_names']
        else:
            # 默认类别名称
            self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
    
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
        return len(self.data)
    
    def __getitem__(self, index):
        """
        获取指定索引的样本。
        
        参数:
            index (int): 样本索引
            
        返回:
            tuple: (image, target) 图像和目标的元组
        """
        # 获取图像数据和标签
        img_array = self.data[index]
        target = self.targets[index]
        
        # 重塑图像数据为32x32x3格式
        # CIFAR-10数据格式：前1024为红色通道，接下来1024为绿色，最后1024为蓝色
        img_array = img_array.reshape(3, 32, 32).transpose(1, 2, 0)
        
        # 将numpy数组转换为PIL图像
        from PIL import Image
        img = Image.fromarray(img_array, mode='RGB')
        
        # 将标签从numpy类型转换为Python的int类型
        target = int(target)
        
        # 应用transform，如果提供的话
        if self.transform is not None:
            img = self.transform(img)
        
        # 应用target_transform，如果提供的话
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target

