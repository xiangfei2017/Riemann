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
from typing import Callable, Any
from enum import Enum
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

# 处理PIL库版本兼容性问题
try:
    # 对于较新版本的PIL (Pillow >= 10.0)
    from PIL.Image import Resampling as PilResampling
    BILINEAR = PilResampling.BILINEAR
    NEAREST = PilResampling.NEAREST
    LANCZOS = PilResampling.LANCZOS
    BICUBIC = PilResampling.BICUBIC
except ImportError:
    # 对于旧版本的PIL
    PilResampling = None  # type: ignore
    BILINEAR = Image.BILINEAR  # type: ignore
    NEAREST = Image.NEAREST  # type: ignore
    LANCZOS = Image.LANCZOS  # type: ignore
    BICUBIC = Image.BICUBIC  # type: ignore

# 定义其他常用常量
FLIP_LEFT_RIGHT = Image.FLIP_LEFT_RIGHT  # type: ignore
FLIP_TOP_BOTTOM = Image.FLIP_TOP_BOTTOM  # type: ignore


class InterpolationMode(Enum):
    """插值模式枚举类
    
    定义图像变换中使用的插值方法，与torchvision.transforms.InterpolationMode兼容。
    
    成员:
        NEAREST: 最近邻插值 - 最快的插值方法，但质量较低
        NEAREST_EXACT: 精确最近邻插值 - 与OpenCV的INTER_NEAREST行为一致
        BILINEAR: 双线性插值 - 默认插值方法，质量和速度的平衡
        BICUBIC: 双三次插值 - 质量较高，但速度较慢
        BOX: 盒式滤波 - 用于缩小图像
        HAMMING: Hamming窗滤波 - 用于缩小图像
        LANCZOS: Lanczos滤波 - 质量最高，但速度最慢
    """
    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"


def _interpolation_mode_to_pil(mode: InterpolationMode) -> int:
    """将InterpolationMode转换为PIL的插值常量"""
    mapping = {
        InterpolationMode.NEAREST: NEAREST,
        InterpolationMode.BILINEAR: BILINEAR,
        InterpolationMode.BICUBIC: BICUBIC,
        InterpolationMode.LANCZOS: LANCZOS,
    }
    if mode in mapping:
        return mapping[mode]
    # 对于不直接支持的类型，使用默认值
    return BILINEAR

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
    
    def __init__(self, transforms: list[Callable]) -> None:
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


class PILToTensor(Transform):
    """将PIL图像转换为TN张量（不缩放）
    
    与ToTensor不同，PILToTensor不进行[0, 255]到[0.0, 1.0]的缩放。
    输出张量的数据类型与输入图像的数据类型一致。
    
    将PIL图像(H x W x C)转换为TN张量(C x H x W)。
    如果输入是灰度图像(H x W)，则输出张量形状为(1 x H x W)。
    
    Args:
        pic (PIL.Image): 要转换的PIL图像
        
    Returns:
        TN: 转换后的张量，值范围[0, 255]，类型为uint8或原始类型
        
    Example:
        >>> transform = transforms.PILToTensor()
        >>> tensor_img = transform(pil_image)  # 值在[0, 255]范围内
    """
    
    def __call__(self, pic: Image.Image) -> TN:
        if not isinstance(pic, Image.Image):
            raise TypeError(f'Input should be PIL.Image, got {type(pic)}')
        
        # 转换为numpy数组
        np_img = np.array(pic)
        
        # 处理不同维度的图像
        if np_img.ndim == 2:
            # 灰度图像，添加通道维度
            np_img = np_img[np.newaxis, :, :]
        elif np_img.ndim == 3:
            # 彩色图像，HWC -> CHW
            np_img = np_img.transpose(2, 0, 1)
        
        return tensor(np_img)
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


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


class ConvertImageDtype(Transform):
    """将图像张量转换为指定的数据类型
    
    与ToTensor不同，这个变换只改变数据类型，不改变值的范围。
    适用于已经转换为张量的图像，需要改变其数据类型的场景。
    
    Args:
        dtype: 目标数据类型，如np.float32, np.float64, np.uint8等
        
    Example:
        >>> transform = transforms.ConvertImageDtype(np.float32)
        >>> float_tensor = transform(uint8_tensor)
    """
    
    def __init__(self, dtype: np.dtype) -> None:
        self.dtype = dtype
    
    def __call__(self, tensor_obj: TN) -> TN:
        """
        Args:
            tensor_obj (TN): 输入张量
            
        Returns:
            TN: 转换数据类型后的张量
        """
        if not isinstance(tensor_obj, TN):
            raise TypeError(f'Input should be a TN tensor, got {type(tensor_obj)}')
        
        return tensor(tensor_obj.data.astype(self.dtype))
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(dtype={self.dtype})'


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
    
    def __init__(self, mean: list[float], std: list[float], inplace: bool = False) -> None:
        self.mean = mean
        self.std = std
        self.inplace = inplace
    
    def __call__(self, tensor_obj: TN) -> TN:
        """
        Args:
            tensor_obj (TN): 要标准化的张量，形状为(C x H x W)
            
        Returns:
            TN: 标准化后的张量
        """
        if not isinstance(tensor_obj, TN):
            raise TypeError(f'Input should be a TN tensor, got {type(tensor)}')
        
        if tensor_obj.ndim != 3:
            raise ValueError(f'Expected tensor to be 3D, got {tensor_obj.ndim}D')
        
        if len(self.mean) != tensor_obj.shape[0] or len(self.std) != tensor_obj.shape[0]:
            raise ValueError(f'Expected mean and std to have {tensor_obj.shape[0]} elements, '
                           f'but got {len(self.mean)} and {len(self.std)}')
        
        if self.inplace:
            for i, (m, s) in enumerate(zip(self.mean, self.std)):
                tensor_obj.data[i] = (tensor_obj.data[i] - m) / s
            return tensor_obj
        else:
            result = []
            for i, (m, s) in enumerate(zip(self.mean, self.std)):
                result.append((tensor_obj.data[i] - m) / s)
            return tensor(np.stack(result, axis=0))
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'


class Resize(Transform):
    """调整PIL图像大小
    
    Args:
        size (int or tuple): 目标大小。如果是int，则较小边会被调整为该大小，
                           保持宽高比。如果是(h, w)，则直接调整为该大小。
        interpolation (int, optional): 插值方法。默认值：BILINEAR
        
    Example:
        >>> transforms.Resize(256)  # 较小边调整为256
        >>> transforms.Resize((256, 256))  # 直接调整为256x256
    """
    
    def __init__(self, size: int | tuple[int, int],
                 interpolation: int = BILINEAR) -> None:
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
    
    def __init__(self, size: int | tuple[int, int]) -> None:
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
            return img.transpose(FLIP_LEFT_RIGHT)
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
            return img.transpose(FLIP_TOP_BOTTOM)
        return img
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p})'


class RandomRotation(Transform):
    """随机旋转
    
    随机旋转图像一个角度。
    
    Args:
        degrees (int or tuple): 旋转角度范围。如果是int，则在(-degrees, degrees)范围内选择。
                               如果是(min, max)，则在(min, max)范围内选择。
        resample (int, optional): 重采样方法。默认值：NEAREST
        expand (bool, optional): 是否扩展图像以适应旋转。默认值：False
        center (tuple, optional): 旋转中心。默认值：图像中心
        
    Example:
        >>> transforms.RandomRotation(30)  # 在(-30, 30)度范围内随机旋转
        >>> transforms.RandomRotation((10, 30))  # 在(10, 30)度范围内随机旋转
    """
    
    def __init__(self, degrees: int | tuple[int, int],
                 resample: int = NEAREST, expand: bool = False,
                 center: tuple[float, float] | None = None) -> None:
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
        return img.rotate(angle, self.resample, self.expand, self.center)  # type: ignore
    
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
    
    def __init__(self, brightness: float | tuple[float, float] = 0,
                 contrast: float | tuple[float, float] = 0,
                 saturation: float | tuple[float, float] = 0,
                 hue: float | tuple[float, float] = 0) -> None:
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5))
    
    def _check_input(self, value: float | tuple[float, float], name: str,
                     center: float = 1, bound: tuple[float, float] = (0, float('inf'))) -> tuple[float, float]:
        if isinstance(value, (float, int)):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value_tuple = (center - value, center + value)
        elif isinstance(value, tuple) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
            value_tuple = value
        else:
            raise TypeError(f"{name} should be a single number or a tuple/list of length 2.")
        
        return value_tuple
    
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
    
    def __init__(self, size: int | tuple[int, int],
                 padding: int | tuple[int, int, int, int] | None = None) -> None:
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
        interpolation (int, optional): 插值方法。默认值：BILINEAR
        
    Example:
        >>> transforms.RandomResizedCrop(224)  # 随机裁剪并调整为224x224
    """
    
    def __init__(self, size: int | tuple[int, int],
                 scale: tuple[float, float] = (0.08, 1.0),
                 ratio: tuple[float, float] = (3. / 4., 4. / 3.),
                 interpolation: int = BILINEAR) -> None:
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
    
    def __init__(self, size: int | tuple[int, int]) -> None:
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
    
    def __call__(self, img: Image.Image) -> tuple[Image.Image, ...]:
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
    
    def __init__(self, size: int | tuple[int, int], vertical_flip: bool = False) -> None:
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.vertical_flip = vertical_flip
    
    def __call__(self, img: Image.Image) -> tuple[Image.Image, ...]:
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
        h_flipped_crops = tuple(crop.transpose(FLIP_LEFT_RIGHT) for crop in five_crops)
        
        if self.vertical_flip:
            # 获取垂直翻转版本
            v_flipped_crops = tuple(crop.transpose(FLIP_TOP_BOTTOM) for crop in five_crops)
            # 获取水平垂直翻转版本
            hv_flipped_crops = tuple(crop.transpose(FLIP_LEFT_RIGHT).transpose(FLIP_TOP_BOTTOM) 
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
    
    def __init__(self, padding: int | tuple[int, int] | tuple[int, int, int, int],
                 fill: int | tuple[int, ...] = 0, padding_mode: str = 'constant') -> None:
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


class GaussianBlur(Transform):
    """高斯模糊
    
    使用高斯滤波器对图像进行模糊处理。
    
    Args:
        kernel_size (int or sequence): 高斯核的大小。必须是正奇数。
                                      如果是int，则使用相同的核大小进行高度和宽度方向的模糊。
                                      如果是(sequence, sequence)，则分别指定高度和宽度方向的核大小。
        sigma (float or sequence): 高斯核的标准差。如果是float，则使用相同的标准差。
                                  如果是(sequence, sequence)，则分别指定高度和宽度方向的标准差。
                                  如果为None，则自动根据kernel_size计算：sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
                                  
    Example:
        >>> transforms.GaussianBlur(kernel_size=5, sigma=2.0)
        >>> transforms.GaussianBlur(kernel_size=(5, 7), sigma=(1.0, 2.0))
    """
    
    def __init__(self, kernel_size: int | tuple[int, int], sigma: float | tuple[float, float] | None = None) -> None:
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): 要模糊的图像
            
        Returns:
            PIL.Image: 模糊后的图像
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f'Expected PIL.Image, got {type(img)}')
        
        # 处理kernel_size
        if isinstance(self.kernel_size, int):
            kx = ky = self.kernel_size
        else:
            kx, ky = self.kernel_size
        
        # 验证kernel_size是奇数
        if kx % 2 == 0 or ky % 2 == 0:
            raise ValueError(f'kernel_size must be odd, got {self.kernel_size}')
        
        # 处理sigma
        if self.sigma is None:
            # 自动计算sigma
            sigma_x = 0.3 * ((kx - 1) * 0.5 - 1) + 0.8
            sigma_y = 0.3 * ((ky - 1) * 0.5 - 1) + 0.8
        elif isinstance(self.sigma, (int, float)):
            sigma_x = sigma_y = float(self.sigma)
        else:
            sigma_x, sigma_y = self.sigma
        
        # 使用PIL的GaussianBlur
        return img.filter(ImageFilter.GaussianBlur(radius=max(sigma_x, sigma_y)))
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(kernel_size={self.kernel_size}, sigma={self.sigma})'


class RandomAffine(Transform):
    """随机仿射变换
    
    对图像进行随机的仿射变换，包括旋转、平移、缩放和剪切。
    
    Args:
        degrees (sequence or float): 旋转角度范围。如果是float，则旋转范围为(-degrees, degrees)。
                                    如果是(sequence, sequence)，则旋转范围为(min, max)。
        translate (tuple, optional): 平移范围，以图像尺寸的比例表示。例如(0.1, 0.1)表示在宽度和高度方向上最多平移10%。
        scale (tuple, optional): 缩放范围。例如(0.8, 1.2)表示缩放因子在0.8到1.2之间。
        shear (sequence or float, optional): 剪切角度范围。可以是单个float或sequence。
        interpolation (InterpolationMode): 插值模式。默认值：InterpolationMode.NEAREST。
        fill (int or tuple): 填充值。默认值：0。
        center (tuple, optional): 旋转中心点。如果为None，则使用图像中心。
        
    Example:
        >>> transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))
    """
    
    def __init__(self, degrees: float | tuple[float, float], 
                 translate: tuple[float, float] | None = None,
                 scale: tuple[float, float] | None = None,
                 shear: float | tuple[float, float] | tuple[float, float, float, float] | None = None,
                 interpolation: InterpolationMode = InterpolationMode.NEAREST,
                 fill: int | tuple[int, ...] = 0,
                 center: tuple[float, float] | None = None) -> None:
        # 处理degrees
        if isinstance(degrees, (int, float)):
            self.degrees = (-float(degrees), float(degrees))
        else:
            self.degrees = degrees
        
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.interpolation = interpolation
        self.fill = fill
        self.center = center
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): 要变换的图像
            
        Returns:
            PIL.Image: 变换后的图像
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f'Expected PIL.Image, got {type(img)}')
        
        width, height = img.size
        
        # 随机旋转角度
        angle = random.uniform(self.degrees[0], self.degrees[1])
        
        # 随机平移
        if self.translate is not None:
            max_dx = self.translate[0] * width
            max_dy = self.translate[1] * height
            translations = (random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy))
        else:
            translations = (0, 0)
        
        # 随机缩放
        if self.scale is not None:
            scale = random.uniform(self.scale[0], self.scale[1])
        else:
            scale = 1.0
        
        # 随机剪切
        if self.shear is not None:
            if isinstance(self.shear, (int, float)):
                shear = (random.uniform(-self.shear, self.shear), 0)
            elif len(self.shear) == 2:
                shear = (random.uniform(self.shear[0], self.shear[1]), 0)
            else:
                shear = (random.uniform(self.shear[0], self.shear[1]), 
                        random.uniform(self.shear[2], self.shear[3]))
        else:
            shear = (0, 0)
        
        # 计算变换矩阵
        center = self.center if self.center is not None else (width / 2, height / 2)
        
        # 构建仿射变换矩阵
        # PIL的transform方法使用6元组 (a, b, c, d, e, f) 表示变换矩阵
        # x' = ax + by + c
        # y' = dx + ey + f
        
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # 旋转矩阵
        rot_matrix = [cos_a, -sin_a, 0, sin_a, cos_a, 0]
        
        # 缩放
        scale_matrix = [scale, 0, 0, 0, scale, 0]
        
        # 构建完整的仿射变换矩阵
        # PIL的transform方法使用6元组 (a, b, c, d, e, f) 表示变换矩阵
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # 组合旋转、缩放、剪切
        shear_x_rad = math.radians(shear[0])
        shear_y_rad = math.radians(shear[1])
        
        a = cos_a * scale - sin_a * shear_y_rad * scale
        b = -sin_a * scale + cos_a * shear_x_rad * scale
        c = translations[0]
        d = sin_a * scale + cos_a * shear_y_rad * scale
        e = cos_a * scale + sin_a * shear_x_rad * scale
        f = translations[1]
        
        # 调整中心点
        cx, cy = center
        c = c - cx * a - cy * b + cx
        f = f - cx * d - cy * e + cy
        
        matrix = (a, b, c, d, e, f)
        
        return img.transform(img.size, Image.AFFINE, matrix, 
                           resample=_interpolation_mode_to_pil(self.interpolation), 
                           fillcolor=self.fill)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(degrees={self.degrees}, translate={self.translate}, scale={self.scale}, shear={self.shear})'


class RandomPerspective(Transform):
    """随机透视变换
    
    对图像进行随机的透视变换。
    
    Args:
        distortion_scale (float): 扭曲程度，范围[0, 1]。默认值：0.5。
        p (float): 应用变换的概率。默认值：0.5。
        interpolation (InterpolationMode): 插值模式。默认值：InterpolationMode.BILINEAR。
        fill (int or tuple): 填充值。默认值：0。
        
    Example:
        >>> transforms.RandomPerspective(distortion_scale=0.5, p=0.5)
    """
    
    def __init__(self, distortion_scale: float = 0.5, p: float = 0.5,
                 interpolation: InterpolationMode = InterpolationMode.BILINEAR,
                 fill: int | tuple[int, ...] = 0) -> None:
        self.distortion_scale = distortion_scale
        self.p = p
        self.interpolation = interpolation
        self.fill = fill
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): 要变换的图像
            
        Returns:
            PIL.Image: 变换后的图像
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f'Expected PIL.Image, got {type(img)}')
        
        if random.random() >= self.p:
            return img
        
        width, height = img.size
        
        # 计算扭曲范围
        half_width = width // 2
        half_height = height // 2
        
        # 随机生成四个角的偏移
        topleft = (random.randint(0, int(self.distortion_scale * half_width)),
                   random.randint(0, int(self.distortion_scale * half_height)))
        topright = (random.randint(width - int(self.distortion_scale * half_width), width),
                    random.randint(0, int(self.distortion_scale * half_height)))
        botright = (random.randint(width - int(self.distortion_scale * half_width), width),
                    random.randint(height - int(self.distortion_scale * half_height), height))
        botleft = (random.randint(0, int(self.distortion_scale * half_width)),
                   random.randint(height - int(self.distortion_scale * half_height), height))
        
        # 定义原始四个角和目标四个角
        coeffs = self._find_coeffs(
            [(0, 0), (width, 0), (width, height), (0, height)],
            [topleft, topright, botright, botleft]
        )
        
        return img.transform(img.size, Image.PERSPECTIVE, coeffs,
                           resample=_interpolation_mode_to_pil(self.interpolation),
                           fillcolor=self.fill)
    
    def _find_coeffs(self, pa: list[tuple[float, float]], pb: list[tuple[float, float]]) -> tuple:
        """计算透视变换的系数"""
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

        A = np.array(matrix, dtype=float)
        B = np.array(pb).reshape(8)

        res = np.dot(np.linalg.inv(A.T @ A) @ A.T, B)
        return tuple(np.array(res).reshape(8))
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(distortion_scale={self.distortion_scale}, p={self.p})'


class RandomErasing(Transform):
    """随机擦除
    
    随机选择图像中的一个矩形区域，并用随机值或给定值填充。
    常用于数据增强，模拟遮挡情况。
    
    Args:
        p (float): 应用擦除的概率。默认值：0.5。
        scale (tuple): 擦除区域的面积范围，相对于图像面积的比例。默认值：(0.02, 0.33)。
        ratio (tuple): 擦除区域的宽高比范围。默认值：(0.3, 3.3)。
        value (int or float or str): 填充值。可以是数字，或"random"表示随机值。默认值：0。
        inplace (bool): 是否原地修改。默认值：False。
        
    Example:
        >>> transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    """
    
    def __init__(self, p: float = 0.5, scale: tuple[float, float] = (0.02, 0.33),
                 ratio: tuple[float, float] = (0.3, 3.3),
                 value: int | float | str = 0, inplace: bool = False) -> None:
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace
    
    def __call__(self, img: TN) -> TN:
        """
        Args:
            img (TN): 输入张量，形状为(C, H, W)
            
        Returns:
            TN: 擦除后的张量
        """
        if not isinstance(img, TN):
            raise TypeError(f'Expected TN tensor, got {type(img)}')
        
        if random.random() >= self.p:
            return img
        
        if not self.inplace:
            img = img.clone()
        
        # 获取图像尺寸
        if img.ndim == 3:
            c, h, w = img.shape
        else:
            raise ValueError(f'Expected 3D tensor (C, H, W), got {img.ndim}D')
        
        area = h * w
        
        # 尝试找到合适的擦除区域
        for _ in range(10):
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            
            eh = int(round(math.sqrt(target_area / aspect_ratio)))
            ew = int(round(math.sqrt(target_area * aspect_ratio)))
            
            if eh < h and ew < w:
                # 随机选择位置
                i = random.randint(0, h - eh)
                j = random.randint(0, w - ew)
                
                # 填充值
                if self.value == 'random':
                    fill_value = random.random()
                else:
                    fill_value = self.value
                
                # 执行擦除
                img.data[:, i:i+eh, j:j+ew] = fill_value
                break
        
        return img
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p}, scale={self.scale}, ratio={self.ratio})'


class AutoAugment(Transform):
    """AutoAugment 自动增强
    
    使用预定义的策略自动选择和应用数据增强操作。
    基于Google的AutoAugment论文实现。
    
    Args:
        policy (str): 使用的策略。可选：'imagenet', 'cifar10', 'svhn'。默认值：'imagenet'。
        interpolation (InterpolationMode): 插值模式。默认值：InterpolationMode.NEAREST。
        fill (int or tuple): 填充值。默认值：None。
        
    Example:
        >>> transforms.AutoAugment(policy='imagenet')
    """
    
    def __init__(self, policy: str = 'imagenet',
                 interpolation: InterpolationMode = InterpolationMode.NEAREST,
                 fill: int | tuple[int, ...] | None = None) -> None:
        self.policy = policy
        self.interpolation = interpolation
        self.fill = fill if fill is not None else 0
        
        # 定义策略
        self.policies = self._get_policies()
    
    def _get_policies(self) -> list:
        """获取预定义的策略"""
        # 简化的策略定义
        if self.policy == 'imagenet':
            return [
                [('Posterize', 0.4, 8), ('Rotate', 0.6, 9)],
                [('Solarize', 0.6, 5), ('AutoContrast', 0.6, None)],
                [('Equalize', 0.8, None), ('Equalize', 0.6, None)],
                [('Posterize', 0.6, 7), ('Posterize', 0.6, 6)],
                [('Equalize', 0.4, None), ('Solarize', 0.2, 4)],
            ]
        elif self.policy == 'cifar10':
            return [
                [('Invert', 0.1, None), ('Contrast', 0.2, 6)],
                [('Rotate', 0.7, 2), ('TranslateX', 0.3, 9)],
                [('Sharpness', 0.8, 1), ('Sharpness', 0.9, 3)],
                [('ShearY', 0.5, 8), ('TranslateY', 0.7, 9)],
                [('AutoContrast', 0.5, None), ('Equalize', 0.9, None)],
            ]
        else:  # svhn
            return [
                [('ShearX', 0.9, 4), ('Invert', 0.2, None)],
                [('ShearY', 0.9, 8), ('Invert', 0.7, None)],
                [('Equalize', 0.6, None), ('Solarize', 0.6, 6)],
                [('Invert', 0.9, None), ('Equalize', 0.6, None)],
                [('Equalize', 0.9, None), ('Equalize', 0.6, None)],
            ]
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): 输入图像
            
        Returns:
            PIL.Image: 增强后的图像
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f'Expected PIL.Image, got {type(img)}')
        
        # 随机选择一个策略
        policy = random.choice(self.policies)
        
        # 应用策略中的操作
        for op_name, prob, magnitude in policy:
            if random.random() < prob:
                img = self._apply_op(img, op_name, magnitude)
        
        return img
    
    def _apply_op(self, img: Image.Image, op_name: str, magnitude: int | None) -> Image.Image:
        """应用单个操作"""
        if op_name == 'Rotate':
            angle = magnitude if magnitude is not None else 0
            return img.rotate(angle, fillcolor=self.fill)
        elif op_name == 'ShearX':
            m = magnitude / 10.0 if magnitude is not None else 0
            return img.transform(img.size, Image.AFFINE, (1, m, 0, 0, 1, 0), fillcolor=self.fill)
        elif op_name == 'ShearY':
            m = magnitude / 10.0 if magnitude is not None else 0
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, m, 1, 0), fillcolor=self.fill)
        elif op_name == 'TranslateX':
            pixels = magnitude if magnitude is not None else 0
            return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), fillcolor=self.fill)
        elif op_name == 'TranslateY':
            pixels = magnitude if magnitude is not None else 0
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), fillcolor=self.fill)
        elif op_name == 'Posterize':
            bits = magnitude if magnitude is not None else 4
            return ImageOps.posterize(img, bits)
        elif op_name == 'Solarize':
            threshold = magnitude * 25 if magnitude is not None else 128
            return ImageOps.solarize(img, threshold)
        elif op_name == 'Equalize':
            return ImageOps.equalize(img)
        elif op_name == 'AutoContrast':
            return ImageOps.autocontrast(img)
        elif op_name == 'Contrast':
            factor = magnitude / 10.0 if magnitude is not None else 1.0
            return ImageEnhance.Contrast(img).enhance(factor)
        elif op_name == 'Sharpness':
            factor = magnitude / 10.0 if magnitude is not None else 1.0
            return ImageEnhance.Sharpness(img).enhance(factor)
        elif op_name == 'Invert':
            return ImageOps.invert(img)
        else:
            return img
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(policy={self.policy})'


class RandAugment(Transform):
    """RandAugment 随机增强
    
    随机选择和应用数据增强操作，比AutoAugment更简单但效果相当。
    基于RandAugment论文实现。
    
    Args:
        num_ops (int): 要应用的操作数量。默认值：2。
        magnitude (int): 所有操作的幅度。默认值：9。
        num_magnitude_bins (int): 幅度的离散级别数。默认值：31。
        interpolation (InterpolationMode): 插值模式。默认值：InterpolationMode.NEAREST。
        fill (int or tuple): 填充值。默认值：None。
        
    Example:
        >>> transforms.RandAugment(num_ops=2, magnitude=9)
    """
    
    def __init__(self, num_ops: int = 2, magnitude: int = 9, num_magnitude_bins: int = 31,
                 interpolation: InterpolationMode = InterpolationMode.NEAREST,
                 fill: int | tuple[int, ...] | None = None) -> None:
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill if fill is not None else 0
        
        # 定义可用的操作
        self.ops = [
            'Rotate', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY',
            'Posterize', 'Solarize', 'Equalize', 'AutoContrast', 'Contrast', 'Sharpness', 'Invert'
        ]
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): 输入图像
            
        Returns:
            PIL.Image: 增强后的图像
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f'Expected PIL.Image, got {type(img)}')
        
        # 随机选择num_ops个操作
        selected_ops = random.sample(self.ops, min(self.num_ops, len(self.ops)))
        
        for op_name in selected_ops:
            img = self._apply_op(img, op_name)
        
        return img
    
    def _apply_op(self, img: Image.Image, op_name: str) -> Image.Image:
        """应用单个操作"""
        # 计算实际幅度
        mag = self.magnitude / self.num_magnitude_bins
        
        if op_name == 'Rotate':
            angle = mag * 30  # 最大30度
            angle = random.choice([-1, 1]) * angle
            return img.rotate(angle, fillcolor=self.fill)
        elif op_name == 'ShearX':
            m = mag * 0.3
            m = random.choice([-1, 1]) * m
            return img.transform(img.size, Image.AFFINE, (1, m, 0, 0, 1, 0), fillcolor=self.fill)
        elif op_name == 'ShearY':
            m = mag * 0.3
            m = random.choice([-1, 1]) * m
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, m, 1, 0), fillcolor=self.fill)
        elif op_name == 'TranslateX':
            pixels = mag * img.size[0] / 3
            pixels = random.choice([-1, 1]) * int(pixels)
            return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), fillcolor=self.fill)
        elif op_name == 'TranslateY':
            pixels = mag * img.size[1] / 3
            pixels = random.choice([-1, 1]) * int(pixels)
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), fillcolor=self.fill)
        elif op_name == 'Posterize':
            bits = int((1 - mag) * 4) + 4
            return ImageOps.posterize(img, bits)
        elif op_name == 'Solarize':
            threshold = int((1 - mag) * 256)
            return ImageOps.solarize(img, threshold)
        elif op_name == 'Equalize':
            return ImageOps.equalize(img)
        elif op_name == 'AutoContrast':
            return ImageOps.autocontrast(img)
        elif op_name == 'Contrast':
            factor = 1 + mag
            return ImageEnhance.Contrast(img).enhance(factor)
        elif op_name == 'Sharpness':
            factor = 1 + mag
            return ImageEnhance.Sharpness(img).enhance(factor)
        elif op_name == 'Invert':
            return ImageOps.invert(img)
        else:
            return img
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_ops={self.num_ops}, magnitude={self.magnitude})'


class TrivialAugmentWide(Transform):
    """TrivialAugmentWide 简单宽范围增强
    
    对每个图像随机应用一个增强操作，使用最大幅度。
    这是最简单但有效的自动增强方法。
    
    Args:
        num_magnitude_bins (int): 幅度的离散级别数。默认值：31。
        interpolation (InterpolationMode): 插值模式。默认值：InterpolationMode.NEAREST。
        fill (int or tuple): 填充值。默认值：None。
        
    Example:
        >>> transforms.TrivialAugmentWide()
    """
    
    def __init__(self, num_magnitude_bins: int = 31,
                 interpolation: InterpolationMode = InterpolationMode.NEAREST,
                 fill: int | tuple[int, ...] | None = None) -> None:
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill if fill is not None else 0
        
        # 定义可用的操作
        self.ops = [
            'Rotate', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY',
            'Posterize', 'Solarize', 'Equalize', 'AutoContrast', 'Contrast', 'Sharpness', 'Invert'
        ]
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): 输入图像
            
        Returns:
            PIL.Image: 增强后的图像
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f'Expected PIL.Image, got {type(img)}')
        
        # 随机选择一个操作
        op_name = random.choice(self.ops)
        
        # 使用最大幅度
        mag = 1.0
        
        return self._apply_op(img, op_name, mag)
    
    def _apply_op(self, img: Image.Image, op_name: str, mag: float) -> Image.Image:
        """应用单个操作"""
        if op_name == 'Rotate':
            angle = random.choice([-1, 1]) * mag * 30
            return img.rotate(angle, fillcolor=self.fill)
        elif op_name == 'ShearX':
            m = random.choice([-1, 1]) * mag * 0.3
            return img.transform(img.size, Image.AFFINE, (1, m, 0, 0, 1, 0), fillcolor=self.fill)
        elif op_name == 'ShearY':
            m = random.choice([-1, 1]) * mag * 0.3
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, m, 1, 0), fillcolor=self.fill)
        elif op_name == 'TranslateX':
            pixels = random.choice([-1, 1]) * int(mag * img.size[0] / 3)
            return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), fillcolor=self.fill)
        elif op_name == 'TranslateY':
            pixels = random.choice([-1, 1]) * int(mag * img.size[1] / 3)
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), fillcolor=self.fill)
        elif op_name == 'Posterize':
            bits = 4
            return ImageOps.posterize(img, bits)
        elif op_name == 'Solarize':
            threshold = 128
            return ImageOps.solarize(img, threshold)
        elif op_name == 'Equalize':
            return ImageOps.equalize(img)
        elif op_name == 'AutoContrast':
            return ImageOps.autocontrast(img)
        elif op_name == 'Contrast':
            factor = 1 + mag
            return ImageEnhance.Contrast(img).enhance(factor)
        elif op_name == 'Sharpness':
            factor = 1 + mag
            return ImageEnhance.Sharpness(img).enhance(factor)
        elif op_name == 'Invert':
            return ImageOps.invert(img)
        else:
            return img
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_magnitude_bins={self.num_magnitude_bins})'


class SanitizeBoundingBox(Transform):
    """边界框清理
    
    清理无效的边界框，移除面积过小或坐标无效的边界框。
    同时可以移除对应的标签。
    
    Args:
        min_size (float): 边界框的最小尺寸（宽或高）。默认值：1.0。
        labels_getter (callable or None): 获取标签的函数。如果为None，则不处理标签。
        
    Example:
        >>> transform = transforms.SanitizeBoundingBox(min_size=1.0)
        >>> boxes, labels = transform(boxes, labels)
    """
    
    def __init__(self, min_size: float = 1.0, labels_getter: Callable | None = None) -> None:
        self.min_size = min_size
        self.labels_getter = labels_getter
    
    def __call__(self, boxes: TN, labels: TN | None = None) -> tuple[TN, TN | None]:
        """
        Args:
            boxes (TN): 边界框张量，形状为(N, 4)，格式为(x1, y1, x2, y2)
            labels (TN, optional): 标签张量，形状为(N,)
            
        Returns:
            tuple: (清理后的边界框, 清理后的标签)
        """
        if not isinstance(boxes, TN):
            raise TypeError(f'Expected TN tensor for boxes, got {type(boxes)}')
        
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f'Expected boxes shape (N, 4), got {boxes.shape}')
        
        # 计算宽度和高度
        x1, y1, x2, y2 = boxes.data[:, 0], boxes.data[:, 1], boxes.data[:, 2], boxes.data[:, 3]
        width = x2 - x1
        height = y2 - y1
        
        # 创建有效掩码
        valid_mask = (width >= self.min_size) & (height >= self.min_size) & (x1 < x2) & (y1 < y2)
        valid_mask = valid_mask & (x1 >= 0) & (y1 >= 0)  # 确保坐标非负
        
        # 过滤边界框
        valid_boxes = tensor(boxes.data[valid_mask])
        
        # 过滤标签
        if labels is not None:
            valid_labels = tensor(labels.data[valid_mask])
        else:
            valid_labels = None
        
        return valid_boxes, valid_labels
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(min_size={self.min_size})'


class Invert(Transform):
    """反转图像颜色
    
    对图像进行颜色反转（负片效果）。
    
    Example:
        >>> transform = transforms.Invert()
        >>> inverted_img = transform(img)
    """
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): 要反转的图像
            
        Returns:
            PIL.Image: 反转后的图像
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f'Expected PIL.Image, got {type(img)}')
        
        return ImageOps.invert(img)
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class Posterize(Transform):
    """减少图像颜色的位数
    
    减少每个颜色通道的位数，产生海报化效果。
    
    Args:
        bits (int): 保留的位数，范围[0, 8]。值越小，效果越明显。
        
    Example:
        >>> transform = transforms.Posterize(bits=4)
        >>> posterized_img = transform(img)
    """
    
    def __init__(self, bits: int) -> None:
        if not 0 <= bits <= 8:
            raise ValueError(f'bits must be between 0 and 8, got {bits}')
        self.bits = bits
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): 要处理的图像
            
        Returns:
            PIL.Image: 处理后的图像
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f'Expected PIL.Image, got {type(img)}')
        
        return ImageOps.posterize(img, self.bits)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(bits={self.bits})'


class Solarize(Transform):
    """反转高于阈值的所有像素值
    
    对图像进行太阳化效果处理，反转高于阈值的所有像素值。
    
    Args:
        threshold (int): 阈值，范围[0, 256]。默认值：128。
        
    Example:
        >>> transform = transforms.Solarize(threshold=128)
        >>> solarized_img = transform(img)
    """
    
    def __init__(self, threshold: int = 128) -> None:
        if not 0 <= threshold <= 256:
            raise ValueError(f'threshold must be between 0 and 256, got {threshold}')
        self.threshold = threshold
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): 要处理的图像
            
        Returns:
            PIL.Image: 处理后的图像
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f'Expected PIL.Image, got {type(img)}')
        
        return ImageOps.solarize(img, self.threshold)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(threshold={self.threshold})'


class Equalize(Transform):
    """均衡化图像直方图
    
    对图像进行直方图均衡化，增强对比度。
    
    Example:
        >>> transform = transforms.Equalize()
        >>> equalized_img = transform(img)
    """
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): 要均衡化的图像
            
        Returns:
            PIL.Image: 均衡化后的图像
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f'Expected PIL.Image, got {type(img)}')
        
        return ImageOps.equalize(img)
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class AutoContrast(Transform):
    """自动调整图像对比度
    
    自动调整图像对比度，使图像使用完整的颜色范围。
    
    Example:
        >>> transform = transforms.AutoContrast()
        >>> adjusted_img = transform(img)
    """
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): 要调整的图像
            
        Returns:
            PIL.Image: 调整后的图像
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f'Expected PIL.Image, got {type(img)}')
        
        return ImageOps.autocontrast(img)
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class Sharpness(Transform):
    """调整图像锐度
    
    调整图像的锐度。
    
    Args:
        sharpness_factor (float): 锐度因子。0表示模糊，1表示原始图像，大于1表示锐化。
        
    Example:
        >>> transform = transforms.Sharpness(sharpness_factor=2.0)
        >>> sharpened_img = transform(img)
    """
    
    def __init__(self, sharpness_factor: float) -> None:
        self.sharpness_factor = sharpness_factor
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): 要调整的图像
            
        Returns:
            PIL.Image: 调整后的图像
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f'Expected PIL.Image, got {type(img)}')
        
        return ImageEnhance.Sharpness(img).enhance(self.sharpness_factor)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(sharpness_factor={self.sharpness_factor})'


class Brightness(Transform):
    """调整图像亮度
    
    调整图像的亮度。
    
    Args:
        brightness_factor (float): 亮度因子。0表示黑色图像，1表示原始图像，大于1表示更亮。
        
    Example:
        >>> transform = transforms.Brightness(brightness_factor=1.5)
        >>> brightened_img = transform(img)
    """
    
    def __init__(self, brightness_factor: float) -> None:
        self.brightness_factor = brightness_factor
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): 要调整的图像
            
        Returns:
            PIL.Image: 调整后的图像
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f'Expected PIL.Image, got {type(img)}')
        
        return ImageEnhance.Brightness(img).enhance(self.brightness_factor)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(brightness_factor={self.brightness_factor})'


class Contrast(Transform):
    """调整图像对比度
    
    调整图像的对比度。
    
    Args:
        contrast_factor (float): 对比度因子。0表示灰色图像，1表示原始图像，大于1表示更高对比度。
        
    Example:
        >>> transform = transforms.Contrast(contrast_factor=1.5)
        >>> adjusted_img = transform(img)
    """
    
    def __init__(self, contrast_factor: float) -> None:
        self.contrast_factor = contrast_factor
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): 要调整的图像
            
        Returns:
            PIL.Image: 调整后的图像
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f'Expected PIL.Image, got {type(img)}')
        
        return ImageEnhance.Contrast(img).enhance(self.contrast_factor)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(contrast_factor={self.contrast_factor})'


class Saturation(Transform):
    """调整图像饱和度
    
    调整图像的饱和度。
    
    Args:
        saturation_factor (float): 饱和度因子。0表示灰度图像，1表示原始图像，大于1表示更高饱和度。
        
    Example:
        >>> transform = transforms.Saturation(saturation_factor=1.5)
        >>> adjusted_img = transform(img)
    """
    
    def __init__(self, saturation_factor: float) -> None:
        self.saturation_factor = saturation_factor
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): 要调整的图像
            
        Returns:
            PIL.Image: 调整后的图像
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f'Expected PIL.Image, got {type(img)}')
        
        return ImageEnhance.Color(img).enhance(self.saturation_factor)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(saturation_factor={self.saturation_factor})'


class Hue(Transform):
    """调整图像色调
    
    调整图像的色调。
    
    Args:
        hue_factor (float): 色调因子，范围[-0.5, 0.5]。0表示原始图像，正数表示向红色偏移，负数表示向蓝色偏移。
        
    Example:
        >>> transform = transforms.Hue(hue_factor=0.1)
        >>> adjusted_img = transform(img)
    """
    
    def __init__(self, hue_factor: float) -> None:
        if not -0.5 <= hue_factor <= 0.5:
            raise ValueError(f'hue_factor must be between -0.5 and 0.5, got {hue_factor}')
        self.hue_factor = hue_factor
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL.Image): 要调整的图像
            
        Returns:
            PIL.Image: 调整后的图像
        """
        if not isinstance(img, Image.Image):
            raise TypeError(f'Expected PIL.Image, got {type(img)}')
        
        # 转换为HSV，调整H通道，再转回RGB
        import colorsys
        
        # 将图像转换为numpy数组
        img_array = np.array(img)
        
        # 调整色调
        result = np.zeros_like(img_array, dtype=np.float32)
        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                r, g, b = img_array[i, j] / 255.0
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                h = (h + self.hue_factor) % 1.0
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                result[i, j] = [r * 255, g * 255, b * 255]
        
        return Image.fromarray(result.astype(np.uint8))
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(hue_factor={self.hue_factor})'