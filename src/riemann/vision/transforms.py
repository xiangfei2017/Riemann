# BSD 3-Clause License
#
# Copyright (c) 2025, Fei Xiang
# All rights reserved.

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
Riemann Library Vision Module: Image Transforms

This module provides various image transformation functions for data augmentation
and preprocessing, designed to work with the Riemann automatic differentiation
framework. The transforms are compatible with PyTorch's torchvision.transforms
API and adapted to work with the Riemann TN (Tensor) data structure.

Main features:
    - Basic transformations: Cropping, flipping, rotation, resizing, etc.
    - Color transformations: Brightness, contrast, saturation adjustments
    - Data augmentation: Random transformations, normalization, etc.
    - Tensor transformations: Conversion between PIL images, NumPy arrays, and TN tensors
    - Composable transformations: Chain multiple transforms together

Usage examples:
    >>> import riemann.vision.transforms as transforms
    >>> transform = transforms.Compose([
    ...     transforms.Resize((256, 256)),
    ...     transforms.RandomHorizontalFlip(),
    ...     transforms.ToTensor(),
    ...     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ... ])
    >>> transformed_image = transform(original_image)

Using this module, you can create flexible data preprocessing pipelines for
computer vision tasks, enhance your training data with various augmentation
techniques, and seamlessly integrate with the Riemann framework for building
and training computer vision models.
"""

from __future__ import annotations
import random
import math
from typing import Callable, List, Tuple, Union, Optional, Any
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from ..tensordef import TN, tensor
from ..dtype import get_default_dtype


class Transform:
    """所有变换类的基类
    
    所有变换类都应该继承这个基类，并实现__call__方法。
    """
    
    def __call__(self, img: Any) -> Any:
        """执行变换
        
        Args:
            img: 输入图像，可以是PIL图像、NumPy数组或TN张量
            
        Returns:
            变换后的图像
        """
        raise NotImplementedError
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class Compose(Transform):
    """将多个变换组合成一个变换
    
    Args:
        transforms (list of Transform objects): 要组合的变换列表
        
    Example:
        >>> transforms.Compose([
        ...     transforms.CenterCrop(10),
        ...     transforms.ToTensor(),
        ... ])
    """
    
    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms
    
    def __call__(self, img: Any) -> Any:
        for t in self.transforms:
            img = t(img)
        return img
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(Transform):
    """将PIL图像或NumPy数组转换为TN张量
    
    将PIL图像(H x W x C)或NumPy数组(H x W x C)转换为TN张量(C x H x W)。
    如果输入是灰度图像(H x W)，则输出张量形状为(1 x H x W)。
    
    转换后的张量值会被缩放到[0.0, 1.0]范围内。
    """
    
    def __call__(self, pic: Any) -> TN:
        """
        Args:
            pic (PIL.Image or numpy.ndarray): 要转换的图像
            
        Returns:
            TN: 转换后的张量
        """
        if isinstance(pic, TN):
            return pic
            
        if isinstance(pic, np.ndarray):
            # NumPy数组处理
            if pic.ndim == 2:
                # 灰度图像
                img = tensor(pic, dtype=get_default_dtype()).unsqueeze(0)
            elif pic.ndim == 3:
                # 彩色图像，HWC -> CHW
                img = tensor(pic.transpose(2, 0, 1), dtype=get_default_dtype())
            else:
                raise ValueError(f'Unsupported array shape: {pic.shape}')
        elif isinstance(pic, Image.Image):
            # PIL图像处理
            if pic.mode == 'I':
                # 32位整数图像
                img = tensor(np.array(pic, np.int32, copy=False), dtype=get_default_dtype())
            elif pic.mode == 'I;16':
                # 16位整数图像
                img = tensor(np.array(pic, np.int16, copy=False), dtype=get_default_dtype())
            elif pic.mode == 'F':
                # 32位浮点图像
                img = tensor(np.array(pic, np.float32, copy=False), dtype=get_default_dtype())
            else:
                # 标准图像模式
                img = tensor(pic, dtype=get_default_dtype())
            
            # 如果是PIL图像，确保数据在[0, 255]范围内并缩放到[0, 1]
            if pic.mode in ['L', 'P', 'I', 'I;16']:
                # 灰度图像，添加通道维度
                img = img.unsqueeze(0) / 255.0
            elif pic.mode in ['RGB', 'RGBA', 'CMYK', 'YCbCr']:
                # 彩色图像，HWC -> CHW
                if img.ndim == 3:
                    img = img.permute(2, 0, 1) / 255.0
                else:
                    img = img / 255.0
        else:
            raise TypeError(f'Unsupported type: {type(pic)}')
        
        return img
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class ToPILImage(Transform):
    """将TN张量或NumPy数组转换为PIL图像
    
    将TN张量(C x H x W)或NumPy数组(H x W x C)转换为PIL图像。
    张量值应该在[0.0, 1.0]范围内，会被缩放到[0, 255]。
    """
    
    def __call__(self, pic: Any) -> Image.Image:
        """
        Args:
            pic (TN or numpy.ndarray): 要转换的张量或数组
            
        Returns:
            PIL.Image: 转换后的PIL图像
        """
        if isinstance(pic, TN):
            # TN张量处理
            np_pic = pic.data
            if np_pic.ndim == 3 and np_pic.shape[0] == 1:
                # 单通道图像，移除通道维度
                np_pic = np_pic.squeeze(0)
            elif np_pic.ndim == 3:
                # 多通道图像，CHW -> HWC
                np_pic = np_pic.transpose(1, 2, 0)
        elif isinstance(pic, np.ndarray):
            np_pic = pic
        else:
            raise TypeError(f'Unsupported type: {type(pic)}')
        
        # 确保值在[0, 255]范围内
        if np_pic.dtype != np.uint8:
            np_pic = np.clip(np_pic * 255, 0, 255).astype(np.uint8)
        
        # 转换为PIL图像
        if np_pic.ndim == 2:
            # 灰度图像
            return Image.fromarray(np_pic, mode='L')
        elif np_pic.ndim == 3:
            # 彩色图像
            if np_pic.shape[2] == 3:
                return Image.fromarray(np_pic, mode='RGB')
            elif np_pic.shape[2] == 4:
                return Image.fromarray(np_pic, mode='RGBA')
        
        raise ValueError(f'Unsupported array shape: {np_pic.shape}')
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class Normalize(Transform):
    """使用均值和标准差标准化张量
    
    给定均值(mean)和标准差(std)，对张量进行标准化：
        output = (input - mean) / std
    
    Args:
        mean (sequence): 每个通道的均值
        std (sequence): 每个通道的标准差
        inplace (bool, optional): 是否原地操作。默认值：False
        
    Example:
        >>> transforms.Normalize(mean=[0.485, 0.456, 0.406],
        ...                      std=[0.229, 0.224, 0.225])
    """
    
    def __init__(self, mean: List[float], std: List[float], inplace: bool = False) -> None:
        self.mean = mean
        self.std = std
        self.inplace = inplace
    
    def __call__(self, tensor: TN) -> TN:
        """
        Args:
            tensor (TN): 要标准化的张量，形状为(C x H x W)
            
        Returns:
            TN: 标准化后的张量
        """
        if not isinstance(tensor, TN):
            raise TypeError(f'Input should be a TN tensor, got {type(tensor)}')
        
        if tensor.ndim != 3:
            raise ValueError(f'Expected tensor to be 3D, got {tensor.ndim}D')
        
        if len(self.mean) != tensor.shape[0] or len(self.std) != tensor.shape[0]:
            raise ValueError(f'Expected mean and std to have {tensor.shape[0]} elements, '
                           f'but got {len(self.mean)} and {len(self.std)}')
        
        if self.inplace:
            for i, (m, s) in enumerate(zip(self.mean, self.std)):
                tensor.data[i] = (tensor.data[i] - m) / s
            return tensor
        else:
            result = []
            for i, (m, s) in enumerate(zip(self.mean, self.std)):
                result.append((tensor.data[i] - m) / s)
            from ..tensordef import tensor
            return tensor(np.stack(result, axis=0))
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'


class Resize(Transform):
    """调整PIL图像大小
    
    Args:
        size (int or tuple): 目标大小。如果是int，则较小边会被调整为该大小，
                           保持宽高比。如果是(h, w)，则直接调整为该大小。
        interpolation (int, optional): 插值方法。默认值：PIL.Image.BILINEAR
        
    Example:
        >>> transforms.Resize(256)  # 较小边调整为256
        >>> transforms.Resize((256, 256))  # 直接调整为256x256
    """
    
    def __init__(self, size: Union[int, Tuple[int, int]], 
                 interpolation: int = Image.BILINEAR) -> None:
        if isinstance(size, int):
            self.size = size
        else:
            self.size = size
        self.interpolation = interpolation
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): 要调整大小的图像
            
        Returns:
            PIL.Image: 调整大小后的图像
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f'Expected PIL.Image, got {type(img)}')
        
        if isinstance(self.size, int):
            # 计算保持宽高比的新大小
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
            else:
                oh = self.size
                ow = int(self.size * w / h)
            return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size})'


class CenterCrop(Transform):
    """中心裁剪
    
    Args:
        size (int or tuple): 裁剪大小。如果是int，则裁剪为正方形(size, size)。
                           如果是(h, w)，则裁剪为该大小。
        
    Example:
        >>> transforms.CenterCrop(224)  # 裁剪为224x224
        >>> transforms.CenterCrop((224, 256))  # 裁剪为224x256
    """
    
    def __init__(self, size: Union[int, Tuple[int, int]]) -> None:
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): 要裁剪的图像
            
        Returns:
            PIL.Image: 裁剪后的图像
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f'Expected PIL.Image, got {type(img)}')
        
        width, height = img.size
        crop_height, crop_width = self.size
        
        if crop_width > width or crop_height > height:
            raise ValueError(f'Required crop size {self.size} is larger than '
                           f'input image size {(width, height)}')
        
        # 计算裁剪位置
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height
        
        return img.crop((left, top, right, bottom))
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size})'


class RandomHorizontalFlip(Transform):
    """随机水平翻转
    
    以给定的概率随机水平翻转图像。
    
    Args:
        p (float, optional): 翻转概率。默认值：0.5
    """
    
    def __init__(self, p: float = 0.5) -> None:
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): 要翻转的图像
            
        Returns:
            PIL.Image: 可能翻转后的图像
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f'Expected PIL.Image, got {type(img)}')
        
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p})'


class RandomVerticalFlip(Transform):
    """随机垂直翻转
    
    以给定的概率随机垂直翻转图像。
    
    Args:
        p (float, optional): 翻转概率。默认值：0.5
    """
    
    def __init__(self, p: float = 0.5) -> None:
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): 要翻转的图像
            
        Returns:
            PIL.Image: 可能翻转后的图像
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f'Expected PIL.Image, got {type(img)}')
        
        if random.random() < self.p:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p})'


class RandomRotation(Transform):
    """随机旋转
    
    随机旋转图像一个角度。
    
    Args:
        degrees (int or tuple): 旋转角度范围。如果是int，则在(-degrees, degrees)范围内选择。
                               如果是(min, max)，则在(min, max)范围内选择。
        resample (int, optional): 重采样方法。默认值：PIL.Image.NEAREST
        expand (bool, optional): 是否扩展图像以适应旋转。默认值：False
        center (tuple, optional): 旋转中心。默认值：图像中心
        
    Example:
        >>> transforms.RandomRotation(30)  # 在(-30, 30)度范围内随机旋转
        >>> transforms.RandomRotation((10, 30))  # 在(10, 30)度范围内随机旋转
    """
    
    def __init__(self, degrees: Union[int, Tuple[int, int]], 
                 resample: int = Image.NEAREST, expand: bool = False, 
                 center: Optional[Tuple[float, float]] = None) -> None:
        if isinstance(degrees, int):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): 要旋转的图像
            
        Returns:
            PIL.Image: 旋转后的图像
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f'Expected PIL.Image, got {type(img)}')
        
        angle = random.uniform(self.degrees[0], self.degrees[1])
        return img.rotate(angle, self.resample, self.expand, self.center)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(degrees={self.degrees})'


class ColorJitter(Transform):
    """随机颜色变换
    
    随机改变图像的亮度、对比度、饱和度和色调。
    
    Args:
        brightness (float or tuple): 亮度调整因子。如果是float，则在(0, brightness)范围内选择。
                                    如果是(min, max)，则在(min, max)范围内选择。
        contrast (float or tuple): 对比度调整因子，同上。
        saturation (float or tuple): 饱和度调整因子，同上。
        hue (float or tuple): 色调调整因子，如果是float，则在(-hue, hue)范围内选择。
                             如果是(min, max)，则在(min, max)范围内选择。
                             应该在[-0.5, 0.5]范围内。
        
    Example:
        >>> transforms.ColorJitter(brightness=0.2, contrast=0.2)
        >>> transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    """
    
    def __init__(self, brightness: Union[float, Tuple[float, float]] = 0,
                 contrast: Union[float, Tuple[float, float]] = 0,
                 saturation: Union[float, Tuple[float, float]] = 0,
                 hue: Union[float, Tuple[float, float]] = 0) -> None:
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5))
    
    def _check_input(self, value: Union[float, Tuple[float, float]], name: str, 
                     center: float = 1, bound: Tuple[float, float] = (0, float('inf'))) -> Tuple[float, float]:
        if isinstance(value, (float, int)):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - value, center + value]
        elif isinstance(value, tuple) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        else:
            raise TypeError(f"{name} should be a single number or a tuple/list of length 2.")
        
        return value
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): 要变换的图像
            
        Returns:
            PIL.Image: 变换后的图像
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f'Expected PIL.Image, got {type(img)}')
        
        # 随机应用变换
        fn_idx = list(range(4))
        random.shuffle(fn_idx)
        
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
                img = ImageEnhance.Brightness(img).enhance(brightness_factor)
            
            elif fn_id == 1 and self.contrast is not None:
                contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
                img = ImageEnhance.Contrast(img).enhance(contrast_factor)
            
            elif fn_id == 2 and self.saturation is not None:
                saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
                img = ImageEnhance.Color(img).enhance(saturation_factor)
            
            elif fn_id == 3 and self.hue is not None:
                hue_factor = random.uniform(self.hue[0], self.hue[1])
                img = ImageEnhance.Color(img).enhance(hue_factor)
        
        return img
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'brightness={self.brightness}, '
                f'contrast={self.contrast}, '
                f'saturation={self.saturation}, '
                f'hue={self.hue})')


class Grayscale(Transform):
    """将图像转换为灰度
    
    Args:
        num_output_channels (int): 输出通道数，1或3。默认值：1
        
    Example:
        >>> transforms.Grayscale()  # 转换为单通道灰度图像
        >>> transforms.Grayscale(3)  # 转换为3通道灰度图像
    """
    
    def __init__(self, num_output_channels: int = 1) -> None:
        if num_output_channels not in (1, 3):
            raise ValueError('num_output_channels should be either 1 or 3')
        self.num_output_channels = num_output_channels
    
    def __call__(self, img: Any) -> Any:
        """
        Args:
            img (PIL.Image or TN): 要转换的图像
            
        Returns:
            PIL.Image or TN: 转换后的图像
        """
        if isinstance(img, TN):
            # TN张量处理
            if img.ndim != 3:
                raise ValueError(f'Expected tensor to be 3D, got {img.ndim}D')
            
            if img.shape[0] == 1:
                # 已经是单通道
                if self.num_output_channels == 3:
                    return img.expand(3, -1, -1)
                return img
            elif img.shape[0] == 3:
                # RGB转灰度
                r, g, b = img.data[0], img.data[1], img.data[2]
                gray = 0.299 * r + 0.587 * g + 0.114 * b
                gray_tensor = tensor(gray, dtype=img.dtype, requires_grad=img.requires_grad)
                
                if self.num_output_channels == 3:
                    return gray_tensor.expand(3, -1, -1)
                return gray_tensor.unsqueeze(0)
            else:
                raise ValueError(f'Expected tensor to have 1 or 3 channels, got {img.shape[0]}')
        elif isinstance(img, Image.Image):
            # PIL图像处理
            return img.convert('L' if self.num_output_channels == 1 else 'RGB')
        else:
            raise TypeError(f'Expected PIL.Image or TN tensor, got {type(img)}')
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_output_channels={self.num_output_channels})'


class RandomGrayscale(Transform):
    """随机转换为灰度
    
    以给定的概率随机将图像转换为灰度。
    
    Args:
        p (float, optional): 转换为灰度的概率。默认值：0.1
        
    Example:
        >>> transforms.RandomGrayscale(p=0.2)  # 20%概率转换为灰度
    """
    
    def __init__(self, p: float = 0.1) -> None:
        self.p = p
        self.grayscale = Grayscale(num_output_channels=3)
    
    def __call__(self, img: Any) -> Any:
        """
        Args:
            img (PIL.Image or TN): 要转换的图像
            
        Returns:
            PIL.Image or TN: 可能转换后的图像
        """
        if random.random() < self.p:
            return self.grayscale(img)
        return img
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p})'


class RandomCrop(Transform):
    """随机裁剪
    
    在随机位置裁剪图像。
    
    Args:
        size (int or tuple): 裁剪大小。如果是int，则裁剪为正方形(size, size)。
                           如果是(h, w)，则裁剪为该大小。
        padding (int or tuple, optional): 填充大小。默认值：None
        
    Example:
        >>> transforms.RandomCrop(224)  # 随机裁剪为224x224
        >>> transforms.RandomCrop((224, 256))  # 随机裁剪为224x256
    """
    
    def __init__(self, size: Union[int, Tuple[int, int]], 
                 padding: Optional[Union[int, Tuple[int, int, int, int]]] = None) -> None:
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.padding = padding
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): 要裁剪的图像
            
        Returns:
            PIL.Image: 裁剪后的图像
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f'Expected PIL.Image, got {type(img)}')
        
        if self.padding is not None:
            img = ImageOps.expand(img, border=self.padding, fill=0)
        
        width, height = img.size
        crop_height, crop_width = self.size
        
        if crop_width > width or crop_height > height:
            raise ValueError(f'Required crop size {self.size} is larger than '
                           f'input image size {(width, height)}')
        
        # 随机选择裁剪位置
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        right = left + crop_width
        bottom = top + crop_height
        
        return img.crop((left, top, right, bottom))
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size}, padding={self.padding})'


class RandomResizedCrop(Transform):
    """随机裁剪并调整大小
    
    随机裁剪图像并调整到指定大小。
    
    Args:
        size (int or tuple): 目标大小。如果是int，则调整为正方形(size, size)。
                           如果是(h, w)，则调整为该大小。
        scale (tuple, optional): 裁剪面积相对于原图的比例范围。默认值：(0.08, 1.0)
        ratio (tuple, optional): 裁剪的宽高比范围。默认值：(3/4, 4/3)
        interpolation (int, optional): 插值方法。默认值：PIL.Image.BILINEAR
        
    Example:
        >>> transforms.RandomResizedCrop(224)  # 随机裁剪并调整为224x224
    """
    
    def __init__(self, size: Union[int, Tuple[int, int]], 
                 scale: Tuple[float, float] = (0.08, 1.0),
                 ratio: Tuple[float, float] = (3. / 4., 4. / 3.),
                 interpolation: int = Image.BILINEAR) -> None:
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): 要裁剪的图像
            
        Returns:
            PIL.Image: 裁剪并调整大小后的图像
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f'Expected PIL.Image, got {type(img)}')
        
        width, height = img.size
        target_width, target_height = self.size
        
        # 随机选择面积和宽高比
        area = width * height
        target_area = random.uniform(*self.scale) * area
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        aspect_ratio = math.exp(random.uniform(*log_ratio))
        
        # 计算裁剪的宽度和高度
        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))
        
        # 确保裁剪大小不超过原图
        if 0 < w <= width and 0 < h <= height:
            # 随机选择裁剪位置
            left = random.randint(0, width - w)
            top = random.randint(0, height - h)
            right = left + w
            bottom = top + h
            
            img = img.crop((left, top, right, bottom))
        
        # 调整大小
        return img.resize((target_width, target_height), self.interpolation)
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(size={self.size}, scale={self.scale}, '
                f'ratio={self.ratio})')


class FiveCrop(Transform):
    """五裁剪
    
    从图像的四个角和中心裁剪指定大小的区域。
    
    Args:
        size (int or tuple): 裁剪大小。如果是int，则裁剪为正方形(size, size)。
                           如果是(h, w)，则裁剪为该大小。
        
    Example:
        >>> transforms.FiveCrop(224)  # 裁剪为5个224x224的图像
    """
    
    def __init__(self, size: Union[int, Tuple[int, int]]) -> None:
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
    
    def __call__(self, img: Image.Image) -> Tuple[Image.Image, ...]:
        """
        Args:
            img (PIL.Image): 要裁剪的图像
            
        Returns:
            tuple: 包含5个裁剪图像的元组
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f'Expected PIL.Image, got {type(img)}')
        
        width, height = img.size
        crop_height, crop_width = self.size
        
        if crop_width > width or crop_height > height:
            raise ValueError(f'Required crop size {self.size} is larger than '
                           f'input image size {(width, height)}')
        
        # 计算裁剪位置
        tl = (0, 0, crop_width, crop_height)  # 左上角
        tr = (width - crop_width, 0, width, crop_height)  # 右上角
        bl = (0, height - crop_height, crop_width, height)  # 左下角
        br = (width - crop_width, height - crop_height, width, height)  # 右下角
        center = ((width - crop_width) // 2, (height - crop_height) // 2,
                  (width + crop_width) // 2, (height + crop_height) // 2)  # 中心
        
        return (img.crop(tl), img.crop(tr), img.crop(bl), img.crop(br), img.crop(center))
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size})'


class TenCrop(Transform):
    """十裁剪
    
    从图像的四个角和中心裁剪指定大小的区域，并包括它们的水平翻转版本。
    
    Args:
        size (int or tuple): 裁剪大小。如果是int，则裁剪为正方形(size, size)。
                           如果是(h, w)，则裁剪为该大小。
        vertical_flip (bool, optional): 是否包括垂直翻转版本。默认值：False
        
    Example:
        >>> transforms.TenCrop(224)  # 裁剪为10个224x224的图像
    """
    
    def __init__(self, size: Union[int, Tuple[int, int]], vertical_flip: bool = False) -> None:
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.vertical_flip = vertical_flip
    
    def __call__(self, img: Image.Image) -> Tuple[Image.Image, ...]:
        """
        Args:
            img (PIL.Image): 要裁剪的图像
            
        Returns:
            tuple: 包含10个裁剪图像的元组
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f'Expected PIL.Image, got {type(img)}')
        
        # 获取五裁剪
        five_crops = FiveCrop(self.size)(img)
        
        # 获取水平翻转版本
        h_flipped_crops = tuple(crop.transpose(Image.FLIP_LEFT_RIGHT) for crop in five_crops)
        
        if self.vertical_flip:
            # 获取垂直翻转版本
            v_flipped_crops = tuple(crop.transpose(Image.FLIP_TOP_BOTTOM) for crop in five_crops)
            # 获取水平垂直翻转版本
            hv_flipped_crops = tuple(crop.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM) 
                                    for crop in five_crops)
            return five_crops + h_flipped_crops + v_flipped_crops + hv_flipped_crops
        else:
            return five_crops + h_flipped_crops
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size}, vertical_flip={self.vertical_flip})'


class Pad(Transform):
    """填充
    
    在图像周围填充指定大小的像素。
    
    Args:
        padding (int or tuple): 填充大小。如果是int，则在所有方向填充相同大小。
                               如果是(pad_l, pad_r, pad_t, pad_b)，则分别指定左右上下的填充大小。
                               如果是(pad_h, pad_w)，则分别指定高度和宽度方向的填充大小。
        fill (int or tuple): 填充值。默认值：0
        padding_mode (str, optional): 填充模式。默认值：'constant'
        
    Example:
        >>> transforms.Pad(4)  # 在所有方向填充4像素
        >>> transforms.Pad((2, 4))  # 高度方向填充2像素，宽度方向填充4像素
    """
    
    def __init__(self, padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]], 
                 fill: Union[int, Tuple[int, ...]] = 0, padding_mode: str = 'constant') -> None:
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): 要填充的图像
            
        Returns:
            PIL.Image: 填充后的图像
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f'Expected PIL.Image, got {type(img)}')
        
        return ImageOps.expand(img, border=self.padding, fill=self.fill)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(padding={self.padding}, fill={self.fill})'


class Lambda(Transform):
    """Lambda变换
    
    使用用户定义的lambda函数作为变换。
    
    Args:
        lambd (function): Lambda函数
        
    Example:
        >>> transforms.Lambda(lambda x: x.convert('L'))  # 转换为灰度图像
    """
    
    def __init__(self, lambd: Callable) -> None:
        self.lambd = lambd
    
    def __call__(self, img: Any) -> Any:
        return self.lambd(img)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'