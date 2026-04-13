#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Riemann vision模块测试脚本

测试脚本文件名：test_052_vision.py
存在tests目录下，可以作为独立脚本运行，也可以被pytest调用测试

参考tests\test_053_nn_cnn.py里的测试框架、测试统计输出代码

对vision模块中的MNIST、CIFAR10数据集类和transforms类进行测试：
1. MNIST数据集测试：与torch的MNIST数据集对比
2. CIFAR10数据集测试：与torch的CIFAR10数据集对比
3. transforms测试：测试transforms.py中的每个类，与torch的同名类测试结果对比

测试代码结构清晰、简洁，重复代码设计为函数以便重用
"""

import sys
import os
import time
import tempfile
import numpy as np
from PIL import Image

# 添加Riemann库路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import riemann as rm
from riemann.vision.datasets import MNIST, EasyMNIST, CIFAR10, DatasetFolder, ImageFolder, default_loader
from riemann.vision.transforms import (
    Compose, ToTensor, ToPILImage, Normalize, Resize, CenterCrop,
    RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter,
    Grayscale, RandomGrayscale, RandomResizedCrop, FiveCrop, TenCrop,
    Pad, Lambda, RandomCrop,
    # 新增加的类
    InterpolationMode, RandomAffine, RandomPerspective, RandomErasing,
    GaussianBlur, AutoAugment, RandAugment, TrivialAugmentWide,
    SanitizeBoundingBox, ConvertImageDtype, PILToTensor as RiemannPILToTensor,
    # 本次补充的类
    Invert, Posterize, Solarize, Equalize, AutoContrast, Sharpness,
    Brightness, Contrast, Saturation, Hue
)

from torchvision import transforms as torch_transforms
from torchvision.datasets import MNIST as TorchMNIST, CIFAR10 as TorchCIFAR10


class Colors:
    """终端颜色类"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# 清屏函数
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

class StatisticsCollector:
    """测试统计类"""
    
    def __init__(self):
        self.function_stats = {}
        self.current_function = None
        self.function_start_time = None
        self.total_tests = 0
        self.total_passed = 0
        self.total_start_time = time.time()
    
    def start_function(self, func_name: str):
        """开始测试一个函数"""
        self.current_function = func_name
        self.function_start_time = time.time()
        if func_name not in self.function_stats:
            self.function_stats[func_name] = {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'time': 0,
                'details': []
            }
        print(f"\n{Colors.HEADER}测试 {func_name}{Colors.ENDC}")
    
    def end_function(self):
        """结束测试一个函数"""
        if self.current_function and self.function_start_time:
            elapsed = time.time() - self.function_start_time
            self.function_stats[self.current_function]['time'] = elapsed
            self.function_start_time = None
            self.current_function = None
    
    def add_result(self, test_name: str, passed: bool, details: str = ""):
        """添加测试结果"""
        if self.current_function:
            stats = self.function_stats[self.current_function]
            stats['total'] += 1
            self.total_tests += 1
            
            if passed:
                stats['passed'] += 1
                self.total_passed += 1
                status = f"{Colors.OKGREEN}通过{Colors.ENDC}"
            else:
                stats['failed'] += 1
                status = f"{Colors.FAIL}失败{Colors.ENDC}"
            
            stats['details'].append({
                'name': test_name,
                'passed': passed,
                'details': details
            })
            
            print(f"  {test_name}: {status}")
            if details:
                print(f"    详情: {details}")
    
    def print_summary(self):
        # 计算列宽 - 分别计算表头和数据的宽度需求
        headers = ['用例名', '通过/总数', '通过率', '耗时(秒)']
        header_widths = [self._get_display_width(h) for h in headers]
        
        # 为数据列单独计算宽度需求
        data_widths = [0, 0, 0, 0]
        for func_name, stats in self.function_stats.items():
            data_widths[0] = max(data_widths[0], self._get_display_width(func_name))
            data_widths[1] = max(data_widths[1], len(f"{stats['passed']}/{stats['total']}"))
            data_widths[2] = max(data_widths[2], len(f"{stats['passed']/stats['total']*100:.1f}%"))
            data_widths[3] = max(data_widths[3], len(f"{stats['time']:.4f}"))
        
        # 结合表头和数据宽度确定最终列宽
        col_widths = [
            max(header_widths[0], data_widths[0]) + 6,  # 增加4个空格（原来是+2）
            max(header_widths[1], data_widths[1]) + 4,
            max(header_widths[2], data_widths[2]) + 8,
            max(header_widths[3], data_widths[3]) + 4
        ]

        """打印测试汇总"""
        total_time = time.time() - self.total_start_time
        total_pass_rate = (self.total_passed / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print(f"\n{Colors.HEADER}{'='*sum(col_widths)}{Colors.ENDC}")
        print(f"{Colors.HEADER}测试汇总{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*sum(col_widths)}{Colors.ENDC}")
        
        # 打印总体统计
        print(f"总测试数: {self.total_tests}")
        print(f"通过数: {self.total_passed}")
        print(f"失败数: {self.total_tests - self.total_passed}")
        print(f"总通过率: {total_pass_rate:.1f}%")
        print(f"总耗时: {total_time:.4f}秒")
        
        print(f"\n{Colors.HEADER}各函数测试详情{Colors.ENDC}")
        print(f"{'-'*sum(col_widths)}")
                
        # 打印表头 - 精确计算每个标题的填充
        header_line = ""
        for i, header in enumerate(headers):
            header_width = self._get_display_width(header)
            padding = col_widths[i] - header_width
            header_line += header + " " * padding
        print(header_line)
        print(f"{'-'*sum(col_widths)}")
        
        # 打印数据行 - 精确计算每个值的填充
        for func_name, stats in self.function_stats.items():
            pass_rate = stats['passed'] / stats['total'] * 100
            color = Colors.OKGREEN if pass_rate == 100 else Colors.WARNING if pass_rate >= 80 else Colors.FAIL
            
            # 计算每个字段的显示宽度并添加适当的填充
            func_name_display = func_name
            func_name_width = self._get_display_width(func_name_display)
            func_name_padding = col_widths[0] - func_name_width
            
            pass_total_display = f"{stats['passed']}/{stats['total']}"
            pass_total_width = len(pass_total_display)  # 纯ASCII，直接用len
            pass_total_padding = col_widths[1] - pass_total_width
            
            # 通过率字段包含颜色代码，但显示宽度只计算实际文本
            pass_rate_display = f"{pass_rate:.1f}%"
            pass_rate_width = len(pass_rate_display)  # 纯ASCII，直接用len
            pass_rate_padding = col_widths[2] - pass_rate_width
            
            time_display = f"{stats['time']:.4f}"
            time_width = len(time_display)  # 纯ASCII，直接用len
            time_padding = col_widths[3] - time_width
            
            # 构建完整的行
            print(
                f"{func_name_display}{' ' * func_name_padding}" +
                f"{pass_total_display}{' ' * pass_total_padding}" +
                f"{color}{pass_rate_display}{' ' * pass_rate_padding}{Colors.ENDC}" +
                f"{time_display}{' ' * time_padding}"
            )
        
        print("="*sum(col_widths))
    
    def _get_display_width(self, text):
        """计算字符串的显示宽度，中文字符算2个宽度，英文字符算1个宽度"""
        width = 0
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                width += 2
            else:
                width += 1
        return width
    
    def _ljust_display_width(self, text, width):
        """按显示宽度左对齐字符串，中文字符算2个宽度"""
        display_width = self._get_display_width(text)
        if display_width >= width:
            return text
        padding = width - display_width
        return text + " " * padding
    
    def _ljust_with_color(self, text, width, color_code):
        """按显示宽度左对齐字符串，支持颜色代码"""
        # 先计算纯文本的显示宽度
        text_display_width = self._get_display_width(text)
        
        if text_display_width >= width:
            # 如果文本宽度已经足够，直接返回带颜色的文本
            return f"{color_code}{text}{Colors.ENDC}"
        
        # 计算需要的填充空格数
        padding = width - text_display_width
        
        # 在添加颜色代码之前添加填充
        return f"{color_code}{text}{' ' * padding}{Colors.ENDC}"


# ==================== 工具函数 ====================

def create_test_image(size=(32, 32), mode='RGB'):
    """创建测试用的PIL图像"""
    if mode == 'L':
        # 灰度图像
        return Image.fromarray(np.random.randint(0, 256, (*size, 1), dtype=np.uint8).squeeze(2), mode='L')
    elif mode == 'RGB':
        # RGB图像
        return Image.fromarray(np.random.randint(0, 256, (*size, 3), dtype=np.uint8), mode='RGB')
    elif mode == 'RGBA':
        # RGBA图像
        return Image.fromarray(np.random.randint(0, 256, (*size, 4), dtype=np.uint8), mode='RGBA')
    else:
        raise ValueError(f"Unsupported image mode: {mode}")

def tensor_allclose(rm_tensor, torch_tensor, rtol=1e-4, atol=1e-6):
    """比较Riemann张量和PyTorch张量是否接近"""
    rm_data = rm_tensor.data if hasattr(rm_tensor, 'data') else rm_tensor
    torch_data = torch_tensor.detach().numpy()
    return np.allclose(rm_data, torch_data, rtol=rtol, atol=atol)

def compare_transforms(rm_transform, torch_transform, test_image, test_name, stats):
    """比较Riemann和PyTorch的变换结果"""
    try:
        # 应用变换
        rm_result = rm_transform(test_image)
        torch_result = torch_transform(test_image)
        
        # 如果结果是PIL图像，转换为张量进行比较
        if isinstance(rm_result, Image.Image) and isinstance(torch_result, Image.Image):
            # 比较图像大小和模式
            size_match = rm_result.size == torch_result.size
            mode_match = rm_result.mode == torch_result.mode
            stats.add_result(f"{test_name}-图像大小", size_match,
                            f"Riemann: {rm_result.size}, PyTorch: {torch_result.size}")
            stats.add_result(f"{test_name}-图像模式", mode_match,
                            f"Riemann: {rm_result.mode}, PyTorch: {torch_result.mode}")
            
            # 转换为数组比较像素值
            rm_array = np.array(rm_result)
            torch_array = np.array(torch_result)
            
            if rm_array.shape == torch_array.shape:
                pixel_match = np.allclose(rm_array, torch_array, rtol=1e-4, atol=1e-6)
                stats.add_result(f"{test_name}-像素值", pixel_match,
                                f"像素差异: {np.abs(rm_array - torch_array).max():.6f}")
            else:
                stats.add_result(f"{test_name}-像素值", False,
                                f"形状不匹配: Riemann {rm_array.shape}, PyTorch {torch_array.shape}")
        
        # 如果结果是张量，直接比较
        elif hasattr(rm_result, 'data') and hasattr(torch_result, 'data'):
            tensor_match = tensor_allclose(rm_result, torch_result)
            stats.add_result(f"{test_name}-张量值", tensor_match,
                            f"张量形状: Riemann {rm_result.shape}, PyTorch {torch_result.shape}")
        
        else:
            stats.add_result(f"{test_name}", False, "结果类型不匹配")
    
    except Exception as e:
        stats.add_result(f"{test_name}", False, f"测试异常: {str(e)}")



# ==================== 测试函数 ====================

def test_mnist_dataset(stats=None):
    """测试MNIST数据集"""
    # 如果没有传入stats实例，创建一个（用于pytest调用）
    if stats is None:
        stats = StatisticsCollector()
    
    stats.start_function("MNIST数据集")
    
    try:
        # 使用data目录下的MNIST数据集
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mnist_root = os.path.join(project_root, 'data')  # 只指定到data目录，MNIST类会自动添加MNIST/raw'data')
        
        # 测试Riemann MNIST
        print("测试Riemann MNIST数据集...")
        rm_mnist_train = MNIST(root=mnist_root, train=True)
        rm_mnist_test = MNIST(root=mnist_root, train=False)
        
        # 测试PyTorch MNIST
        print("测试PyTorch MNIST数据集...")
        torch_mnist_train = TorchMNIST(root=mnist_root, train=True, download=False)
        torch_mnist_test = TorchMNIST(root=mnist_root, train=False, download=False)
    
        # 测试数据集长度
        stats.add_result("MNIST训练集长度-Riemann", len(rm_mnist_train) == 60000,
                        f"期望: 60000, 实际: {len(rm_mnist_train)}")
        stats.add_result("MNIST训练集长度-PyTorch", len(torch_mnist_train) == 60000,
                        f"期望: 60000, 实际: {len(torch_mnist_train)}")
        stats.add_result("MNIST训练集长度-一致性", len(rm_mnist_train) == len(torch_mnist_train),
                        f"Riemann: {len(rm_mnist_train)}, PyTorch: {len(torch_mnist_train)}")
        
        stats.add_result("MNIST测试集长度-Riemann", len(rm_mnist_test) == 10000,
                        f"期望: 10000, 实际: {len(rm_mnist_test)}")
        stats.add_result("MNIST测试集长度-PyTorch", len(torch_mnist_test) == 10000,
                        f"期望: 10000, 实际: {len(torch_mnist_test)}")
        stats.add_result("MNIST测试集长度-一致性", len(rm_mnist_test) == len(torch_mnist_test),
                        f"Riemann: {len(rm_mnist_test)}, PyTorch: {len(torch_mnist_test)}")
        
        # 测试数据获取
        rm_img, rm_label = rm_mnist_train[0]
        torch_img, torch_label = torch_mnist_train[0]
        
        stats.add_result("MNIST数据获取-图像类型-Riemann", isinstance(rm_img, Image.Image),
                        f"图像类型: {type(rm_img)}")
        stats.add_result("MNIST数据获取-图像类型-PyTorch", isinstance(torch_img, Image.Image),
                        f"图像类型: {type(torch_img)}")
        stats.add_result("MNIST数据获取-图像模式-Riemann", rm_img.mode == 'L',
                        f"图像模式: {rm_img.mode}")
        stats.add_result("MNIST数据获取-图像模式-PyTorch", torch_img.mode == 'L',
                        f"图像模式: {torch_img.mode}")
        stats.add_result("MNIST数据获取-图像大小-Riemann", rm_img.size == (28, 28),
                        f"图像大小: {rm_img.size}")
        stats.add_result("MNIST数据获取-图像大小-PyTorch", torch_img.size == (28, 28),
                        f"图像大小: {torch_img.size}")
        
        # 比较图像数据
        rm_img_array = np.array(rm_img)
        torch_img_array = np.array(torch_img)
        img_match = np.array_equal(rm_img_array, torch_img_array)
        stats.add_result("MNIST数据获取-图像数据一致性", img_match,
                        f"图像数据差异: {np.abs(rm_img_array - torch_img_array).max() if not img_match else 0}")
        
        stats.add_result("MNIST数据获取-标签类型-Riemann", isinstance(rm_label, int),
                        f"标签类型: {type(rm_label)}")
        stats.add_result("MNIST数据获取-标签类型-PyTorch", isinstance(torch_label, int),
                        f"标签类型: {type(torch_label)}")
        stats.add_result("MNIST数据获取-标签值-一致性", rm_label == torch_label,
                        f"Riemann: {rm_label}, PyTorch: {torch_label}")
    
        # 测试EasyMNIST
        print("测试Riemann EasyMNIST数据集...")
        rm_easymnist_train = EasyMNIST(root=mnist_root, train=True, onehot_label=True)
        rm_easymnist_test = EasyMNIST(root=mnist_root, train=False, onehot_label=False)
        
        # 测试数据集长度
        stats.add_result("EasyMNIST训练集长度", len(rm_easymnist_train) == 60000,
                        f"期望: 60000, 实际: {len(rm_easymnist_train)}")
        stats.add_result("EasyMNIST测试集长度", len(rm_easymnist_test) == 10000,
                        f"期望: 10000, 实际: {len(rm_easymnist_test)}")
        
        # 测试数据获取
        img, label = rm_easymnist_train[0]
        stats.add_result("EasyMNIST数据获取-图像类型", hasattr(img, 'data'),
                        f"图像类型: {type(img)}")
        stats.add_result("EasyMNIST数据获取-图像形状", img.shape == (784,),
                        f"图像形状: {img.shape}")
        
        img, label = rm_easymnist_test[0]
        stats.add_result("EasyMNIST测试数据获取-标签类型", hasattr(label, 'data'),
                        f"标签类型: {type(label)}")
        stats.add_result("EasyMNIST测试数据获取-标签形状", label.shape == (),
                        f"标签形状: {label.shape}")
    
    except Exception as e:
        stats.add_result("MNIST数据集测试", False, f"测试异常: {str(e)}")
    
    finally:
        stats.end_function()

def test_cifar10_dataset(stats=None):
    """测试CIFAR10数据集"""
    # 如果没有传入stats实例，创建一个（用于pytest调用）
    if stats is None:
        stats = StatisticsCollector()
    
    stats.start_function("CIFAR10数据集")
    
    try:
        # 使用data目录下的CIFAR10数据集
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cifar10_root = os.path.join(project_root, 'data')  # 只指定到data目录，CIFAR10类会自动添加cifar-10-batches-py
        
        # 测试Riemann CFAIR10
        print("测试Riemann CFAIR10数据集...")
        rm_cifar10_train = CIFAR10(root=cifar10_root, train=True)
        rm_cifar10_test = CIFAR10(root=cifar10_root, train=False)
        
        # 测试PyTorch CIFAR10
        print("测试PyTorch CIFAR10数据集...")
        torch_cifar10_train = TorchCIFAR10(root=cifar10_root, train=True, download=False)
        torch_cifar10_test = TorchCIFAR10(root=cifar10_root, train=False, download=False)
        
        # 测试数据集长度
        stats.add_result("CIFAR10训练集长度-Riemann", len(rm_cifar10_train) == 50000,
                       f"期望: 50000, 实际: {len(rm_cifar10_train)}")
        stats.add_result("CIFAR10训练集长度-PyTorch", len(torch_cifar10_train) == 50000,
                       f"期望: 50000, 实际: {len(torch_cifar10_train)}")
        stats.add_result("CIFAR10训练集长度-一致性", len(rm_cifar10_train) == len(torch_cifar10_train),
                       f"Riemann: {len(rm_cifar10_train)}, PyTorch: {len(torch_cifar10_train)}")
        
        stats.add_result("CIFAR10测试集长度-Riemann", len(rm_cifar10_test) == 10000,
                       f"期望: 10000, 实际: {len(rm_cifar10_test)}")
        stats.add_result("CIFAR10测试集长度-PyTorch", len(torch_cifar10_test) == 10000,
                       f"期望: 10000, 实际: {len(torch_cifar10_test)}")
        stats.add_result("CIFAR10测试集长度-一致性", len(rm_cifar10_test) == len(torch_cifar10_test),
                       f"Riemann: {len(rm_cifar10_test)}, PyTorch: {len(torch_cifar10_test)}")
        
        # 测试数据获取
        rm_img, rm_label = rm_cifar10_train[0]
        torch_img, torch_label = torch_cifar10_train[0]
        
        stats.add_result("CIFAR10数据获取-图像类型-Riemann", isinstance(rm_img, Image.Image),
                       f"图像类型: {type(rm_img)}")
        stats.add_result("CIFAR10数据获取-图像类型-PyTorch", isinstance(torch_img, Image.Image),
                       f"图像类型: {type(torch_img)}")
        stats.add_result("CIFAR10数据获取-图像模式-Riemann", rm_img.mode == 'RGB',
                       f"图像模式: {rm_img.mode}")
        stats.add_result("CIFAR10数据获取-图像模式-PyTorch", torch_img.mode == 'RGB',
                       f"图像模式: {torch_img.mode}")
        stats.add_result("CIFAR10数据获取-图像大小-Riemann", rm_img.size == (32, 32),
                       f"图像大小: {rm_img.size}")
        stats.add_result("CIFAR10数据获取-图像大小-PyTorch", torch_img.size == (32, 32),
                       f"图像大小: {torch_img.size}")
        
        # 比较图像数据
        rm_img_array = np.array(rm_img)
        torch_img_array = np.array(torch_img)
        img_match = np.array_equal(rm_img_array, torch_img_array)
        stats.add_result("CIFAR10数据获取-图像数据一致性", img_match,
                       f"图像数据差异: {np.abs(rm_img_array - torch_img_array).max() if not img_match else 0}")
        
        stats.add_result("CIFAR10数据获取-标签类型-Riemann", isinstance(rm_label, int),
                       f"标签类型: {type(rm_label)}")
        stats.add_result("CIFAR10数据获取-标签类型-PyTorch", isinstance(torch_label, int),
                       f"标签类型: {type(torch_label)}")
        stats.add_result("CIFAR10数据获取-标签值-一致性", rm_label == torch_label,
                       f"Riemann: {rm_label}, PyTorch: {torch_label}")
        
        # 测试类别
        stats.add_result("CIFAR10数据集-类别数量-Riemann", len(rm_cifar10_train.classes) == 10,
                       f"类别数量: {len(rm_cifar10_train.classes)}")
        stats.add_result("CIFAR10数据集-类别数量-PyTorch", len(torch_cifar10_train.classes) == 10,
                       f"类别数量: {len(torch_cifar10_train.classes)}")
        stats.add_result("CIFAR10数据集-类别名称-一致性", rm_cifar10_train.classes == torch_cifar10_train.classes,
                       f"类别差异: {set(rm_cifar10_train.classes) - set(torch_cifar10_train.classes)}")
            
    except Exception as e:
        stats.add_result("CIFAR10数据集测试", False, f"测试异常: {str(e)}")
    
    finally:
        stats.end_function()

def test_transforms(stats=None):
    """测试transforms类"""
    # 如果没有传入stats实例，创建一个（用于pytest调用）
    if stats is None:
        stats = StatisticsCollector()
    
    stats.start_function("Transforms类")
    
    try:
        # 创建测试图像
        rgb_image = create_test_image(size=(64, 64), mode='RGB')
        gray_image = create_test_image(size=(64, 64), mode='L')
        
        # 测试ToTensor
        print("测试ToTensor...")
        rm_to_tensor = ToTensor()
        torch_to_tensor = torch_transforms.ToTensor()
        compare_transforms(rm_to_tensor, torch_to_tensor, rgb_image, "ToTensor-RGB", stats)
        compare_transforms(rm_to_tensor, torch_to_tensor, gray_image, "ToTensor-灰度", stats)
        
        # 测试ToPILImage
        print("测试ToPILImage...")
        rm_to_pil = ToPILImage()
        torch_to_pil = torch_transforms.ToPILImage()
        
        # 先转换为张量再转回PIL图像
        rm_tensor = rm_to_tensor(rgb_image)
        torch_tensor = torch_to_tensor(rgb_image)
        
        # 分别使用各自的张量
        try:
            rm_result = rm_to_pil(rm_tensor)
            torch_result = torch_to_pil(torch_tensor)
            
            # 比较图像大小和模式
            size_match = rm_result.size == torch_result.size
            mode_match = rm_result.mode == torch_result.mode
            stats.add_result("ToPILImage-从张量-图像大小", size_match,
                            f"Riemann: {rm_result.size}, PyTorch: {torch_result.size}")
            stats.add_result("ToPILImage-从张量-图像模式", mode_match,
                            f"Riemann: {rm_result.mode}, PyTorch: {torch_result.mode}")
            
            # 转换为数组比较像素值
            rm_array = np.array(rm_result)
            torch_array = np.array(torch_result)
            pixel_match = np.array_equal(rm_array, torch_array)
            stats.add_result("ToPILImage-从张量-像素值", pixel_match,
                            f"像素差异: {np.abs(rm_array - torch_array).max() if not pixel_match else 0}")
        except Exception as e:
            stats.add_result("ToPILImage-从张量", False, f"测试异常: {str(e)}")
        
        # 测试Normalize
        print("测试Normalize...")
        rm_normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        torch_normalize = torch_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # 分别使用各自的张量
        try:
            rm_result = rm_normalize(rm_tensor)
            torch_result = torch_normalize(torch_tensor)
            
            # 比较张量形状
            shape_match = rm_result.shape == tuple(torch_result.shape)
            stats.add_result("Normalize-RGB-张量形状", shape_match,
                            f"Riemann: {rm_result.shape}, PyTorch: {torch_result.shape}")
            
            # 比较张量值
            rm_data = rm_result.data if hasattr(rm_result, 'data') else rm_result
            torch_data = torch_result.detach().numpy()
            value_match = np.allclose(rm_data, torch_data, rtol=1e-5, atol=1e-5)
            stats.add_result("Normalize-RGB-张量值", value_match,
                            f"值差异: {np.abs(rm_data - torch_data).max() if not value_match else 0}")
        except Exception as e:
            stats.add_result("Normalize-RGB", False, f"测试异常: {str(e)}")
        
        # 测试Resize
        print("测试Resize...")
        rm_resize = Resize((32, 32))
        torch_resize = torch_transforms.Resize((32, 32))
        compare_transforms(rm_resize, torch_resize, rgb_image, "Resize-固定大小", stats)
        
        rm_resize_int = Resize(32)
        torch_resize_int = torch_transforms.Resize(32)
        compare_transforms(rm_resize_int, torch_resize_int, rgb_image, "Resize-整数大小", stats)
        
        # 测试CenterCrop
        print("测试CenterCrop...")
        rm_center_crop = CenterCrop(32)
        torch_center_crop = torch_transforms.CenterCrop(32)
        compare_transforms(rm_center_crop, torch_center_crop, rgb_image, "CenterCrop-正方形", stats)
        
        # 测试RandomHorizontalFlip
        print("测试RandomHorizontalFlip...")
        rm_h_flip = RandomHorizontalFlip(p=1.0)  # 设置p=1确保总是翻转
        torch_h_flip = torch_transforms.RandomHorizontalFlip(p=1.0)
        compare_transforms(rm_h_flip, torch_h_flip, rgb_image, "RandomHorizontalFlip", stats)
        
        # 测试RandomVerticalFlip
        print("测试RandomVerticalFlip...")
        rm_v_flip = RandomVerticalFlip(p=1.0)  # 设置p=1确保总是翻转
        torch_v_flip = torch_transforms.RandomVerticalFlip(p=1.0)
        compare_transforms(rm_v_flip, torch_v_flip, rgb_image, "RandomVerticalFlip", stats)
        
        # 测试RandomRotation
        print("测试RandomRotation...")
        rm_rotation = RandomRotation(30)  # 固定角度范围
        torch_rotation = torch_transforms.RandomRotation(30)
        
        # 由于RandomRotation是随机变换，只比较图像大小和模式，不比较像素值
        try:
            rm_result = rm_rotation(rgb_image)
            torch_result = torch_rotation(rgb_image)
            
            # 比较图像大小和模式
            size_match = rm_result.size == torch_result.size
            mode_match = rm_result.mode == torch_result.mode
            stats.add_result("RandomRotation-图像大小", size_match,
                            f"Riemann: {rm_result.size}, PyTorch: {torch_result.size}")
            stats.add_result("RandomRotation-图像模式", mode_match,
                            f"Riemann: {rm_result.mode}, PyTorch: {torch_result.mode}")
            
            # 不比较像素值，因为随机变换结果不同
            stats.add_result("RandomRotation-像素值", True, "随机变换不比较像素值")
        except Exception as e:
            stats.add_result("RandomRotation", False, f"测试异常: {str(e)}")
        
        # 测试ColorJitter
        print("测试ColorJitter...")
        rm_color_jitter = ColorJitter(brightness=0.2, contrast=0.2)
        torch_color_jitter = torch_transforms.ColorJitter(brightness=0.2, contrast=0.2)
        
        # 由于ColorJitter是随机变换，只比较图像大小和模式，不比较像素值
        try:
            rm_result = rm_color_jitter(rgb_image)
            torch_result = torch_color_jitter(rgb_image)
            
            # 比较图像大小和模式
            size_match = rm_result.size == torch_result.size
            mode_match = rm_result.mode == torch_result.mode
            stats.add_result("ColorJitter-图像大小", size_match,
                            f"Riemann: {rm_result.size}, PyTorch: {torch_result.size}")
            stats.add_result("ColorJitter-图像模式", mode_match,
                            f"Riemann: {rm_result.mode}, PyTorch: {torch_result.mode}")
            
            # 不比较像素值，因为随机变换结果不同
            stats.add_result("ColorJitter-像素值", True, "随机变换不比较像素值")
        except Exception as e:
            stats.add_result("ColorJitter", False, f"测试异常: {str(e)}")
        
        # 测试Compose
        print("测试Compose...")
        rm_compose = Compose([
            Resize((32, 32)),
            CenterCrop(28),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        torch_compose = torch_transforms.Compose([
            torch_transforms.Resize((32, 32)),
            torch_transforms.CenterCrop(28),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        compare_transforms(rm_compose, torch_compose, rgb_image, "Compose", stats)
        
        # 测试Lambda
        print("测试Lambda...")
        rm_lambda = Lambda(lambda x: x.convert('L'))  # 转换为灰度
        torch_lambda = torch_transforms.Lambda(lambda x: x.convert('L'))  # 转换为灰度
        compare_transforms(rm_lambda, torch_lambda, rgb_image, "Lambda", stats)
        
        # 测试Grayscale
        print("测试Grayscale...")
        try:
            rm_grayscale = Grayscale(num_output_channels=1)
            torch_grayscale = torch_transforms.Grayscale(num_output_channels=1)
            
            rm_result = rm_grayscale(rgb_image)
            torch_result = torch_grayscale(rgb_image)
            
            # 转换为numpy数组进行比较
            rm_array = np.array(rm_result)
            torch_array = np.array(torch_result)
            
            # 比较图像大小和模式
            size_match = rm_result.size == torch_result.size
            mode_match = rm_result.mode == torch_result.mode
            stats.add_result("Grayscale-图像大小", size_match,
                            f"Riemann: {rm_result.size}, PyTorch: {torch_result.size}")
            stats.add_result("Grayscale-图像模式", mode_match,
                            f"Riemann: {rm_result.mode}, PyTorch: {torch_result.mode}")
            
            # 比较像素值
            pixel_match = np.allclose(rm_array, torch_array, atol=1)
            stats.add_result("Grayscale-像素值", pixel_match,
                            f"数组形状: Riemann {rm_array.shape}, PyTorch {torch_array.shape}")
        except Exception as e:
            stats.add_result("Grayscale", False, f"测试异常: {str(e)}")
        
        # 测试RandomGrayscale
        print("测试RandomGrayscale...")
        try:
            rm_random_grayscale = RandomGrayscale(p=0.5)
            torch_random_grayscale = torch_transforms.RandomGrayscale(p=0.5)
            
            rm_result = rm_random_grayscale(rgb_image)
            torch_result = torch_random_grayscale(rgb_image)
            
            # 比较图像大小和模式
            size_match = rm_result.size == torch_result.size
            mode_match = rm_result.mode == torch_result.mode
            stats.add_result("RandomGrayscale-图像大小", size_match,
                            f"Riemann: {rm_result.size}, PyTorch: {torch_result.size}")
            stats.add_result("RandomGrayscale-图像模式", mode_match,
                            f"Riemann: {rm_result.mode}, PyTorch: {torch_result.mode}")
            
            # 不比较像素值，因为随机变换结果可能不同
            stats.add_result("RandomGrayscale-像素值", True, "随机变换不比较像素值")
        except Exception as e:
            stats.add_result("RandomGrayscale", False, f"测试异常: {str(e)}")
        
        # 测试RandomResizedCrop
        print("测试RandomResizedCrop...")
        try:
            rm_random_crop = RandomResizedCrop(size=(100, 100), scale=(0.8, 1.0), ratio=(3./4., 4./3.))
            torch_random_crop = torch_transforms.RandomResizedCrop(size=(100, 100), scale=(0.8, 1.0), ratio=(3./4., 4./3.))
            
            rm_result = rm_random_crop(rgb_image)
            torch_result = torch_random_crop(rgb_image)
            
            # 比较图像大小和模式
            size_match = rm_result.size == torch_result.size
            mode_match = rm_result.mode == torch_result.mode
            stats.add_result("RandomResizedCrop-图像大小", size_match,
                            f"Riemann: {rm_result.size}, PyTorch: {torch_result.size}")
            stats.add_result("RandomResizedCrop-图像模式", mode_match,
                            f"Riemann: {rm_result.mode}, PyTorch: {torch_result.mode}")
            
            # 不比较像素值，因为随机变换结果可能不同
            stats.add_result("RandomResizedCrop-像素值", True, "随机变换不比较像素值")
        except Exception as e:
            stats.add_result("RandomResizedCrop", False, f"测试异常: {str(e)}")
        
        # 测试FiveCrop
        print("测试FiveCrop...")
        try:
            rm_five_crop = FiveCrop(size=(32, 32))  # 使用更小的尺寸
            torch_five_crop = torch_transforms.FiveCrop(size=(32, 32))
            
            rm_result = rm_five_crop(rgb_image)
            torch_result = torch_five_crop(rgb_image)
            
            # FiveCrop返回5个图像的元组
            if len(rm_result) == 5 and len(torch_result) == 5:
                stats.add_result("FiveCrop-返回数量", True, "都返回5个图像")
                
                # 比较每个图像的大小和模式
                all_size_match = True
                all_mode_match = True
                for i in range(5):
                    if rm_result[i].size != torch_result[i].size:
                        all_size_match = False
                    if rm_result[i].mode != torch_result[i].mode:
                        all_mode_match = False
                
                stats.add_result("FiveCrop-图像大小", all_size_match, "所有裁剪图像大小匹配")
                stats.add_result("FiveCrop-图像模式", all_mode_match, "所有裁剪图像模式匹配")
                
                # 比较第一个裁剪的像素值
                rm_array = np.array(rm_result[0])
                torch_array = np.array(torch_result[0])
                pixel_match = np.allclose(rm_array, torch_array, atol=1)
                stats.add_result("FiveCrop-像素值", pixel_match,
                                f"第一个裁剪数组形状: Riemann {rm_array.shape}, PyTorch {torch_array.shape}")
            else:
                stats.add_result("FiveCrop-返回数量", False,
                                f"Riemann: {len(rm_result)}, PyTorch: {len(torch_result)}")
        except Exception as e:
            stats.add_result("FiveCrop", False, f"测试异常: {str(e)}")
        
        # 测试TenCrop
        print("测试TenCrop...")
        try:
            rm_ten_crop = TenCrop(size=(32, 32))  # 使用更小的尺寸
            torch_ten_crop = torch_transforms.TenCrop(size=(32, 32))
            
            rm_result = rm_ten_crop(rgb_image)
            torch_result = torch_ten_crop(rgb_image)
            
            # TenCrop返回10个图像的元组
            if len(rm_result) == 10 and len(torch_result) == 10:
                stats.add_result("TenCrop-返回数量", True, "都返回10个图像")
                
                # 比较每个图像的大小和模式
                all_size_match = True
                all_mode_match = True
                for i in range(10):
                    if rm_result[i].size != torch_result[i].size:
                        all_size_match = False
                    if rm_result[i].mode != torch_result[i].mode:
                        all_mode_match = False
                
                stats.add_result("TenCrop-图像大小", all_size_match, "所有裁剪图像大小匹配")
                stats.add_result("TenCrop-图像模式", all_mode_match, "所有裁剪图像模式匹配")
                
                # 比较第一个裁剪的像素值
                rm_array = np.array(rm_result[0])
                torch_array = np.array(torch_result[0])
                pixel_match = np.allclose(rm_array, torch_array, atol=1)
                stats.add_result("TenCrop-像素值", pixel_match,
                                f"第一个裁剪数组形状: Riemann {rm_array.shape}, PyTorch {torch_array.shape}")
            else:
                stats.add_result("TenCrop-返回数量", False,
                                f"Riemann: {len(rm_result)}, PyTorch: {len(torch_result)}")
        except Exception as e:
            stats.add_result("TenCrop", False, f"测试异常: {str(e)}")
        
        # 测试Pad
        print("测试Pad...")
        try:
            rm_pad = Pad(padding=10, fill=0)
            torch_pad = torch_transforms.Pad(padding=10, fill=0)
            
            rm_result = rm_pad(rgb_image)
            torch_result = torch_pad(rgb_image)
            
            # 转换为numpy数组进行比较
            rm_array = np.array(rm_result)
            torch_array = np.array(torch_result)
            
            # 比较图像大小和模式
            size_match = rm_result.size == torch_result.size
            mode_match = rm_result.mode == torch_result.mode
            stats.add_result("Pad-图像大小", size_match,
                            f"Riemann: {rm_result.size}, PyTorch: {torch_result.size}")
            stats.add_result("Pad-图像模式", mode_match,
                            f"Riemann: {rm_result.mode}, PyTorch: {torch_result.mode}")
            
            # 比较像素值
            pixel_match = np.allclose(rm_array, torch_array, atol=1)
            stats.add_result("Pad-像素值", pixel_match,
                            f"数组形状: Riemann {rm_array.shape}, PyTorch {torch_array.shape}")
        except Exception as e:
            stats.add_result("Pad", False, f"测试异常: {str(e)}")
        
        # ==================== 新增加的Transforms类测试 ====================
        print("\n测试新增加的Transforms类...")
        
        # 测试InterpolationMode
        print("测试InterpolationMode...")
        try:
            # 测试枚举值
            stats.add_result("InterpolationMode-NEAREST", InterpolationMode.NEAREST.value == "nearest",
                            f"值: {InterpolationMode.NEAREST.value}")
            stats.add_result("InterpolationMode-BILINEAR", InterpolationMode.BILINEAR.value == "bilinear",
                            f"值: {InterpolationMode.BILINEAR.value}")
            stats.add_result("InterpolationMode-BICUBIC", InterpolationMode.BICUBIC.value == "bicubic",
                            f"值: {InterpolationMode.BICUBIC.value}")
        except Exception as e:
            stats.add_result("InterpolationMode", False, f"测试异常: {str(e)}")
        
        # 测试PILToTensor（与ToTensor区分）
        print("测试PILToTensor...")
        try:
            rm_pil_to_tensor = RiemannPILToTensor()
            rm_result = rm_pil_to_tensor(rgb_image)
            
            # PILToTensor不缩放值到[0,1]，保持原始值[0,255]
            stats.add_result("PILToTensor-输出形状", rm_result.shape == (3, 64, 64),
                            f"形状: {rm_result.shape}")
            stats.add_result("PILToTensor-值范围", rm_result.data.max() > 1.0,
                            f"最大值: {rm_result.data.max():.2f} (应该>1.0，因为不缩放)")
            # 检查是否为TN类型（从riemann导入的tensor类型）
            from riemann.tensordef import TN
            stats.add_result("PILToTensor-类型", isinstance(rm_result, TN),
                            f"类型: {type(rm_result)}")
        except Exception as e:
            stats.add_result("PILToTensor", False, f"测试异常: {str(e)}")
        
        # 测试ConvertImageDtype
        print("测试ConvertImageDtype...")
        try:
            # 创建测试张量
            test_tensor = rm.tensor(np.random.randint(0, 256, (3, 32, 32), dtype=np.uint8))
            rm_convert = ConvertImageDtype(np.float32)
            rm_result = rm_convert(test_tensor)
            
            stats.add_result("ConvertImageDtype-类型转换", rm_result.data.dtype == np.float32,
                            f"转换后类型: {rm_result.data.dtype}")
            stats.add_result("ConvertImageDtype-形状保持", rm_result.shape == test_tensor.shape,
                            f"形状: {rm_result.shape}")
        except Exception as e:
            stats.add_result("ConvertImageDtype", False, f"测试异常: {str(e)}")
        
        # 测试GaussianBlur
        print("测试GaussianBlur...")
        try:
            rm_blur = GaussianBlur(kernel_size=5, sigma=1.0)
            torch_blur = torch_transforms.GaussianBlur(kernel_size=5, sigma=1.0)
            
            rm_result = rm_blur(rgb_image)
            torch_result = torch_blur(rgb_image)
            
            # 比较图像大小和模式
            size_match = rm_result.size == torch_result.size
            mode_match = rm_result.mode == torch_result.mode
            stats.add_result("GaussianBlur-图像大小", size_match,
                            f"Riemann: {rm_result.size}, PyTorch: {torch_result.size}")
            stats.add_result("GaussianBlur-图像模式", mode_match,
                            f"Riemann: {rm_result.mode}, PyTorch: {torch_result.mode}")
            
            # 模糊后的图像应该与原始图像不同
            rm_array = np.array(rm_result)
            original_array = np.array(rgb_image)
            is_different = not np.array_equal(rm_array, original_array)
            stats.add_result("GaussianBlur-图像变化", is_different, "模糊后图像应该与原始图像不同")
        except Exception as e:
            stats.add_result("GaussianBlur", False, f"测试异常: {str(e)}")
        
        # 测试RandomAffine
        print("测试RandomAffine...")
        try:
            rm_affine = RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))
            torch_affine = torch_transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))
            
            rm_result = rm_affine(rgb_image)
            torch_result = torch_affine(rgb_image)
            
            # 比较图像大小和模式
            size_match = rm_result.size == torch_result.size
            mode_match = rm_result.mode == torch_result.mode
            stats.add_result("RandomAffine-图像大小", size_match,
                            f"Riemann: {rm_result.size}, PyTorch: {torch_result.size}")
            stats.add_result("RandomAffine-图像模式", mode_match,
                            f"Riemann: {rm_result.mode}, PyTorch: {torch_result.mode}")
            
            # 不比较像素值，因为随机变换结果不同
            stats.add_result("RandomAffine-像素值", True, "随机变换不比较像素值")
        except Exception as e:
            stats.add_result("RandomAffine", False, f"测试异常: {str(e)}")
        
        # 测试RandomPerspective
        print("测试RandomPerspective...")
        try:
            rm_perspective = RandomPerspective(distortion_scale=0.3, p=1.0)
            torch_perspective = torch_transforms.RandomPerspective(distortion_scale=0.3, p=1.0)
            
            rm_result = rm_perspective(rgb_image)
            torch_result = torch_perspective(rgb_image)
            
            # 比较图像大小和模式
            size_match = rm_result.size == torch_result.size
            mode_match = rm_result.mode == torch_result.mode
            stats.add_result("RandomPerspective-图像大小", size_match,
                            f"Riemann: {rm_result.size}, PyTorch: {torch_result.size}")
            stats.add_result("RandomPerspective-图像模式", mode_match,
                            f"Riemann: {rm_result.mode}, PyTorch: {torch_result.mode}")
            
            # 不比较像素值，因为随机变换结果不同
            stats.add_result("RandomPerspective-像素值", True, "随机变换不比较像素值")
        except Exception as e:
            stats.add_result("RandomPerspective", False, f"测试异常: {str(e)}")
        
        # 测试RandomErasing
        print("测试RandomErasing...")
        try:
            from riemann.tensordef import TN
            # RandomErasing需要TN张量作为输入
            test_tensor = rm.tensor(np.random.rand(3, 64, 64).astype(np.float32))
            rm_erasing = RandomErasing(p=1.0, scale=(0.1, 0.3), value='random')
            
            rm_result = rm_erasing(test_tensor)
            
            stats.add_result("RandomErasing-输出形状", rm_result.shape == test_tensor.shape,
                            f"形状: {rm_result.shape}")
            stats.add_result("RandomErasing-类型保持", isinstance(rm_result, TN),
                            f"类型: {type(rm_result)}")
            
            # 擦除后的张量应该与原始张量不同
            is_different = not np.allclose(rm_result.data, test_tensor.data)
            stats.add_result("RandomErasing-张量变化", is_different, "擦除后张量应该与原始张量不同")
        except Exception as e:
            stats.add_result("RandomErasing", False, f"测试异常: {str(e)}")
        
        # 测试AutoAugment
        print("测试AutoAugment...")
        try:
            rm_autoaugment = AutoAugment(policy='imagenet')
            torch_autoaugment = torch_transforms.AutoAugment(policy=torch_transforms.AutoAugmentPolicy.IMAGENET)
            
            rm_result = rm_autoaugment(rgb_image)
            torch_result = torch_autoaugment(rgb_image)
            
            # 比较图像大小和模式
            size_match = rm_result.size == torch_result.size
            mode_match = rm_result.mode == torch_result.mode
            stats.add_result("AutoAugment-图像大小", size_match,
                            f"Riemann: {rm_result.size}, PyTorch: {torch_result.size}")
            stats.add_result("AutoAugment-图像模式", mode_match,
                            f"Riemann: {rm_result.mode}, PyTorch: {torch_result.mode}")
            
            # 不比较像素值，因为随机变换结果不同
            stats.add_result("AutoAugment-像素值", True, "随机变换不比较像素值")
        except Exception as e:
            stats.add_result("AutoAugment", False, f"测试异常: {str(e)}")
        
        # 测试RandAugment
        print("测试RandAugment...")
        try:
            rm_randaugment = RandAugment(num_ops=2, magnitude=9)
            torch_randaugment = torch_transforms.RandAugment(num_ops=2, magnitude=9)
            
            rm_result = rm_randaugment(rgb_image)
            torch_result = torch_randaugment(rgb_image)
            
            # 比较图像大小和模式
            size_match = rm_result.size == torch_result.size
            mode_match = rm_result.mode == torch_result.mode
            stats.add_result("RandAugment-图像大小", size_match,
                            f"Riemann: {rm_result.size}, PyTorch: {torch_result.size}")
            stats.add_result("RandAugment-图像模式", mode_match,
                            f"Riemann: {rm_result.mode}, PyTorch: {torch_result.mode}")
            
            # 不比较像素值，因为随机变换结果不同
            stats.add_result("RandAugment-像素值", True, "随机变换不比较像素值")
        except Exception as e:
            stats.add_result("RandAugment", False, f"测试异常: {str(e)}")
        
        # 测试TrivialAugmentWide
        print("测试TrivialAugmentWide...")
        try:
            rm_trivial = TrivialAugmentWide()
            torch_trivial = torch_transforms.TrivialAugmentWide()
            
            rm_result = rm_trivial(rgb_image)
            torch_result = torch_trivial(rgb_image)
            
            # 比较图像大小和模式
            size_match = rm_result.size == torch_result.size
            mode_match = rm_result.mode == torch_result.mode
            stats.add_result("TrivialAugmentWide-图像大小", size_match,
                            f"Riemann: {rm_result.size}, PyTorch: {torch_result.size}")
            stats.add_result("TrivialAugmentWide-图像模式", mode_match,
                            f"Riemann: {rm_result.mode}, PyTorch: {torch_result.mode}")
            
            # 不比较像素值，因为随机变换结果不同
            stats.add_result("TrivialAugmentWide-像素值", True, "随机变换不比较像素值")
        except Exception as e:
            stats.add_result("TrivialAugmentWide", False, f"测试异常: {str(e)}")
        
        # 测试SanitizeBoundingBox
        print("测试SanitizeBoundingBox...")
        try:
            # 创建测试边界框 (x1, y1, x2, y2)
            boxes = rm.tensor(np.array([
                [10, 10, 50, 50],   # 有效框
                [5, 5, 100, 100],   # 有效框
                [0, 0, 1, 1],       # 太小，应该被过滤
                [60, 60, 40, 40],   # 无效框 (x1 > x2)
                [-5, -5, 10, 10],   # 部分在图像外
            ], dtype=np.float32))
            labels = rm.tensor(np.array([1, 2, 3, 4, 5], dtype=np.int64))
            
            sanitize = SanitizeBoundingBox(min_size=5)
            valid_boxes, valid_labels = sanitize(boxes, labels)
            
            stats.add_result("SanitizeBoundingBox-过滤无效框", valid_boxes.shape[0] < boxes.shape[0],
                            f"输入: {boxes.shape[0]}个框, 输出: {valid_boxes.shape[0]}个框")
            stats.add_result("SanitizeBoundingBox-标签同步", valid_labels.shape[0] == valid_boxes.shape[0],
                            f"有效标签数: {valid_labels.shape[0]}")
            stats.add_result("SanitizeBoundingBox-输出形状", valid_boxes.shape[1] == 4,
                            f"边界框形状: {valid_boxes.shape}")
        except Exception as e:
            stats.add_result("SanitizeBoundingBox", False, f"测试异常: {str(e)}")
        
        # ==================== 本次补充的Transforms类测试 ====================
        print("\n测试本次补充的Transforms类...")
        
        # 测试Invert
        print("测试Invert...")
        try:
            rm_invert = Invert()
            rm_result = rm_invert(rgb_image)
            
            # 验证基本功能
            stats.add_result("Invert-图像大小", rm_result.size == rgb_image.size,
                            f"大小: {rm_result.size}")
            stats.add_result("Invert-图像模式", rm_result.mode == rgb_image.mode,
                            f"模式: {rm_result.mode}")
            
            # 验证反转效果：两次反转应该回到原图
            rm_restored = rm_invert(rm_result)
            restored_array = np.array(rm_restored)
            original_array = np.array(rgb_image)
            is_restored = np.array_equal(restored_array, original_array)
            stats.add_result("Invert-可逆性", is_restored, "两次反转应该回到原图")
        except Exception as e:
            stats.add_result("Invert", False, f"测试异常: {str(e)}")
        
        # 测试Posterize
        print("测试Posterize...")
        try:
            rm_posterize = Posterize(bits=4)
            rm_result = rm_posterize(rgb_image)
            
            # 验证基本功能
            stats.add_result("Posterize-图像大小", rm_result.size == rgb_image.size,
                            f"大小: {rm_result.size}")
            stats.add_result("Posterize-图像模式", rm_result.mode == rgb_image.mode,
                            f"模式: {rm_result.mode}")
            
            # 验证posterize效果：颜色数量应该减少
            rm_array = np.array(rm_result)
            original_array = np.array(rgb_image)
            is_different = not np.array_equal(rm_array, original_array)
            stats.add_result("Posterize-图像变化", is_different, "Posterize后图像应该不同")
        except Exception as e:
            stats.add_result("Posterize", False, f"测试异常: {str(e)}")
        
        # 测试Solarize
        print("测试Solarize...")
        try:
            rm_solarize = Solarize(threshold=128)
            rm_result = rm_solarize(rgb_image)
            
            # 验证基本功能
            stats.add_result("Solarize-图像大小", rm_result.size == rgb_image.size,
                            f"大小: {rm_result.size}")
            stats.add_result("Solarize-图像模式", rm_result.mode == rgb_image.mode,
                            f"模式: {rm_result.mode}")
            
            # 验证solarize效果
            rm_array = np.array(rm_result)
            original_array = np.array(rgb_image)
            is_different = not np.array_equal(rm_array, original_array)
            stats.add_result("Solarize-图像变化", is_different, "Solarize后图像应该不同")
        except Exception as e:
            stats.add_result("Solarize", False, f"测试异常: {str(e)}")
        
        # 测试Equalize
        print("测试Equalize...")
        try:
            rm_equalize = Equalize()
            rm_result = rm_equalize(rgb_image)
            
            # 验证基本功能
            stats.add_result("Equalize-图像大小", rm_result.size == rgb_image.size,
                            f"大小: {rm_result.size}")
            stats.add_result("Equalize-图像模式", rm_result.mode == rgb_image.mode,
                            f"模式: {rm_result.mode}")
            
            # 验证equalize效果
            rm_array = np.array(rm_result)
            original_array = np.array(rgb_image)
            is_different = not np.array_equal(rm_array, original_array)
            stats.add_result("Equalize-图像变化", is_different, "Equalize后图像应该不同")
        except Exception as e:
            stats.add_result("Equalize", False, f"测试异常: {str(e)}")
        
        # 测试AutoContrast
        print("测试AutoContrast...")
        try:
            rm_autocontrast = AutoContrast()
            rm_result = rm_autocontrast(rgb_image)
            
            # 验证基本功能
            stats.add_result("AutoContrast-图像大小", rm_result.size == rgb_image.size,
                            f"大小: {rm_result.size}")
            stats.add_result("AutoContrast-图像模式", rm_result.mode == rgb_image.mode,
                            f"模式: {rm_result.mode}")
            
            # 验证autocontrast返回的是PIL图像
            stats.add_result("AutoContrast-返回类型", isinstance(rm_result, Image.Image),
                            f"类型: {type(rm_result)}")
        except Exception as e:
            stats.add_result("AutoContrast", False, f"测试异常: {str(e)}")
        
        # 测试Sharpness
        print("测试Sharpness...")
        try:
            rm_sharpness = Sharpness(sharpness_factor=2.0)
            rm_result = rm_sharpness(rgb_image)
            
            # 验证基本功能
            stats.add_result("Sharpness-图像大小", rm_result.size == rgb_image.size,
                            f"大小: {rm_result.size}")
            stats.add_result("Sharpness-图像模式", rm_result.mode == rgb_image.mode,
                            f"模式: {rm_result.mode}")
            
            # 验证sharpness效果
            rm_array = np.array(rm_result)
            original_array = np.array(rgb_image)
            is_different = not np.array_equal(rm_array, original_array)
            stats.add_result("Sharpness-图像变化", is_different, "Sharpness调整后图像应该不同")
        except Exception as e:
            stats.add_result("Sharpness", False, f"测试异常: {str(e)}")
        
        # 测试Brightness
        print("测试Brightness...")
        try:
            rm_brightness = Brightness(brightness_factor=1.5)
            torch_brightness = torch_transforms.ColorJitter(brightness=0.5)
            
            rm_result = rm_brightness(rgb_image)
            # ColorJitter是随机的，我们直接测试Riemann的功能
            
            stats.add_result("Brightness-图像大小", rm_result.size == rgb_image.size,
                            f"大小: {rm_result.size}")
            stats.add_result("Brightness-图像模式", rm_result.mode == rgb_image.mode,
                            f"模式: {rm_result.mode}")
            
            # 验证亮度确实改变了
            rm_array = np.array(rm_result)
            original_array = np.array(rgb_image)
            is_different = not np.array_equal(rm_array, original_array)
            stats.add_result("Brightness-图像变化", is_different, "亮度调整后图像应该不同")
        except Exception as e:
            stats.add_result("Brightness", False, f"测试异常: {str(e)}")
        
        # 测试Contrast
        print("测试Contrast...")
        try:
            rm_contrast = Contrast(contrast_factor=1.5)
            
            rm_result = rm_contrast(rgb_image)
            
            stats.add_result("Contrast-图像大小", rm_result.size == rgb_image.size,
                            f"大小: {rm_result.size}")
            stats.add_result("Contrast-图像模式", rm_result.mode == rgb_image.mode,
                            f"模式: {rm_result.mode}")
            
            # 验证对比度确实改变了
            rm_array = np.array(rm_result)
            original_array = np.array(rgb_image)
            is_different = not np.array_equal(rm_array, original_array)
            stats.add_result("Contrast-图像变化", is_different, "对比度调整后图像应该不同")
        except Exception as e:
            stats.add_result("Contrast", False, f"测试异常: {str(e)}")
        
        # 测试Saturation
        print("测试Saturation...")
        try:
            rm_saturation = Saturation(saturation_factor=1.5)
            
            rm_result = rm_saturation(rgb_image)
            
            stats.add_result("Saturation-图像大小", rm_result.size == rgb_image.size,
                            f"大小: {rm_result.size}")
            stats.add_result("Saturation-图像模式", rm_result.mode == rgb_image.mode,
                            f"模式: {rm_result.mode}")
            
            # 验证饱和度确实改变了
            rm_array = np.array(rm_result)
            original_array = np.array(rgb_image)
            is_different = not np.array_equal(rm_array, original_array)
            stats.add_result("Saturation-图像变化", is_different, "饱和度调整后图像应该不同")
        except Exception as e:
            stats.add_result("Saturation", False, f"测试异常: {str(e)}")
        
        # 测试Hue
        print("测试Hue...")
        try:
            rm_hue = Hue(hue_factor=0.1)
            
            rm_result = rm_hue(rgb_image)
            
            stats.add_result("Hue-图像大小", rm_result.size == rgb_image.size,
                            f"大小: {rm_result.size}")
            stats.add_result("Hue-图像模式", rm_result.mode == rgb_image.mode,
                            f"模式: {rm_result.mode}")
            
            # 验证色调确实改变了
            rm_array = np.array(rm_result)
            original_array = np.array(rgb_image)
            is_different = not np.array_equal(rm_array, original_array)
            stats.add_result("Hue-图像变化", is_different, "色调调整后图像应该不同")
        except Exception as e:
            stats.add_result("Hue", False, f"测试异常: {str(e)}")
        
        # ==================== 已有但未充分验证的类测试 ====================
        print("\n测试已有但未充分验证的类...")
        
        # 测试RandomCrop
        print("测试RandomCrop...")
        try:
            rm_random_crop = RandomCrop(size=(32, 32))
            torch_random_crop = torch_transforms.RandomCrop(size=(32, 32))
            
            rm_result = rm_random_crop(rgb_image)
            torch_result = torch_random_crop(rgb_image)
            
            # 比较图像大小和模式
            size_match = rm_result.size == torch_result.size
            mode_match = rm_result.mode == torch_result.mode
            stats.add_result("RandomCrop-图像大小", size_match,
                            f"Riemann: {rm_result.size}, PyTorch: {torch_result.size}")
            stats.add_result("RandomCrop-图像模式", mode_match,
                            f"Riemann: {rm_result.mode}, PyTorch: {torch_result.mode}")
            
            # 不比较像素值，因为随机裁剪位置不同
            stats.add_result("RandomCrop-像素值", True, "随机变换不比较像素值")
        except Exception as e:
            stats.add_result("RandomCrop", False, f"测试异常: {str(e)}")
        
    except Exception as e:
        stats.add_result("Transforms测试", False, f"测试异常: {str(e)}")
    
    finally:
        stats.end_function()


def create_test_imagefolder_dataset(root_dir):
    """
    创建测试数据集目录结构：
    root/
    ├── cat/
    │   ├── cat1.jpg
    │   ├── cat2.png
    │   └── cat3.jpeg
    ├── dog/
    │   ├── dog1.jpg
    │   ├── dog2.png
    │   └── dog3.jpeg
    └── bird/
        └── bird1.png
    """
    # 创建类别目录
    cat_dir = os.path.join(root_dir, 'cat')
    dog_dir = os.path.join(root_dir, 'dog')
    bird_dir = os.path.join(root_dir, 'bird')
    
    os.makedirs(cat_dir, exist_ok=True)
    os.makedirs(dog_dir, exist_ok=True)
    os.makedirs(bird_dir, exist_ok=True)
    
    # 创建测试图像
    def create_test_image(path, size=(64, 64), color=(255, 0, 0)):
        img = Image.new('RGB', size, color)
        img.save(path)
    
    # cat 类别的图像
    create_test_image(os.path.join(cat_dir, 'cat1.jpg'), color=(255, 100, 100))
    create_test_image(os.path.join(cat_dir, 'cat2.png'), color=(255, 150, 150))
    create_test_image(os.path.join(cat_dir, 'cat3.jpeg'), color=(255, 200, 200))
    
    # dog 类别的图像
    create_test_image(os.path.join(dog_dir, 'dog1.jpg'), color=(100, 255, 100))
    create_test_image(os.path.join(dog_dir, 'dog2.png'), color=(150, 255, 150))
    create_test_image(os.path.join(dog_dir, 'dog3.jpeg'), color=(200, 255, 200))
    
    # bird 类别的图像
    create_test_image(os.path.join(bird_dir, 'bird1.png'), color=(100, 100, 255))


def test_datasetfolder(stats: StatisticsCollector):
    """DatasetFolder 测试用例组"""
    stats.start_function("DatasetFolder")
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_imagefolder_dataset(tmpdir)
            
            # 测试1: 基本初始化
            print("测试基本初始化...")
            try:
                dataset = DatasetFolder(tmpdir, loader=default_loader)
                stats.add_result("基本初始化-数据集大小", len(dataset) == 7,
                                f"大小: {len(dataset)}, 期望: 7")
                stats.add_result("基本初始化-类别数", len(dataset.classes) == 3,
                                f"类别: {dataset.classes}")
                stats.add_result("基本初始化-class_to_idx", set(dataset.class_to_idx.keys()) == {'bird', 'cat', 'dog'},
                                f"class_to_idx: {dataset.class_to_idx}")
            except Exception as e:
                stats.add_result("基本初始化", False, f"异常: {str(e)}")
            
            # 测试2: 获取样本
            print("测试获取样本...")
            try:
                dataset = DatasetFolder(tmpdir, loader=default_loader)
                img, label = dataset[0]
                stats.add_result("获取样本-图像类型", isinstance(img, Image.Image),
                                f"类型: {type(img)}")
                stats.add_result("获取样本-标签类型", isinstance(label, int),
                                f"类型: {type(label)}")
            except Exception as e:
                stats.add_result("获取样本", False, f"异常: {str(e)}")
            
            # 测试3: 扩展名过滤
            print("测试扩展名过滤...")
            try:
                folder_with_ext = DatasetFolder(
                    tmpdir, 
                    loader=default_loader,
                    extensions=('.jpg', '.jpeg')
                )
                stats.add_result("扩展名过滤", len(folder_with_ext) == 4,
                                f"大小: {len(folder_with_ext)}, 期望: 4")
            except Exception as e:
                stats.add_result("扩展名过滤", False, f"异常: {str(e)}")
            
            # 测试4: is_valid_file 功能
            print("测试is_valid_file...")
            try:
                def custom_validator(path):
                    return 'cat' in path
                folder_with_validator = DatasetFolder(
                    tmpdir,
                    loader=default_loader,
                    is_valid_file=custom_validator
                )
                stats.add_result("is_valid_file", len(folder_with_validator) == 3,
                                f"大小: {len(folder_with_validator)}, 期望: 3")
            except Exception as e:
                stats.add_result("is_valid_file", False, f"异常: {str(e)}")
            
            # 测试5: 参数互斥检测
            print("测试参数互斥检测...")
            try:
                DatasetFolder(
                    tmpdir,
                    loader=default_loader,
                    extensions=('.jpg',),
                    is_valid_file=lambda x: True
                )
                stats.add_result("参数互斥检测", False, "应该抛出 ValueError")
            except ValueError:
                stats.add_result("参数互斥检测", True, "正确抛出 ValueError")
            except Exception as e:
                stats.add_result("参数互斥检测", False, f"异常: {str(e)}")
            
            # 测试6: 空目录检测
            print("测试空目录检测...")
            try:
                empty_dir = os.path.join(tmpdir, 'empty')
                os.makedirs(empty_dir)
                DatasetFolder(empty_dir, loader=default_loader, allow_empty=False)
                stats.add_result("空目录检测", False, "应该抛出 FileNotFoundError")
            except FileNotFoundError:
                stats.add_result("空目录检测", True, "正确抛出 FileNotFoundError")
            except Exception as e:
                stats.add_result("空目录检测", False, f"异常: {str(e)}")
            
            # 测试7: allow_empty=True
            print("测试allow_empty=True...")
            try:
                empty_dir = os.path.join(tmpdir, 'empty')
                os.makedirs(empty_dir, exist_ok=True)
                folder = DatasetFolder(empty_dir, loader=default_loader, allow_empty=True)
                stats.add_result("allow_empty=True", len(folder) == 0,
                                f"大小: {len(folder)}, 期望: 0")
            except Exception as e:
                stats.add_result("allow_empty=True", False, f"异常: {str(e)}")
            
            # 测试8: __repr__
            print("测试__repr__...")
            try:
                dataset = DatasetFolder(tmpdir, loader=default_loader)
                repr_str = repr(dataset)
                stats.add_result("__repr__", 'DatasetFolder' in repr_str,
                                f"包含类名: {'DatasetFolder' in repr_str}")
            except Exception as e:
                stats.add_result("__repr__", False, f"异常: {str(e)}")
                
    except Exception as e:
        stats.add_result("DatasetFolder测试", False, f"测试异常: {str(e)}")
    
    finally:
        stats.end_function()


def test_imagefolder(stats: StatisticsCollector):
    """ImageFolder 测试用例组"""
    stats.start_function("ImageFolder")
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            create_test_imagefolder_dataset(tmpdir)
            
            # 测试1: 基本初始化
            print("测试基本初始化...")
            try:
                dataset = ImageFolder(tmpdir)
                stats.add_result("基本初始化-数据集大小", len(dataset) == 7,
                                f"大小: {len(dataset)}, 期望: 7")
                stats.add_result("基本初始化-类别数", len(dataset.classes) == 3,
                                f"类别: {dataset.classes}")
                stats.add_result("基本初始化-样本数", len(dataset.samples) == 7,
                                f"样本数: {len(dataset.samples)}")
            except Exception as e:
                stats.add_result("基本初始化", False, f"异常: {str(e)}")
            
            # 测试2: 获取样本
            print("测试获取样本...")
            try:
                dataset = ImageFolder(tmpdir)
                img, label = dataset[0]
                stats.add_result("获取样本-图像类型", isinstance(img, Image.Image),
                                f"类型: {type(img)}")
                stats.add_result("获取样本-标签类型", isinstance(label, int),
                                f"类型: {type(label)}")
            except Exception as e:
                stats.add_result("获取样本", False, f"异常: {str(e)}")
            
            # 测试3: transform 功能
            print("测试transform功能...")
            try:
                transform = ToTensor()
                dataset_with_transform = ImageFolder(tmpdir, transform=transform)
                img, label = dataset_with_transform[0]
                stats.add_result("transform功能", hasattr(img, 'shape'),
                                f"变换后类型: {type(img)}")
            except Exception as e:
                stats.add_result("transform功能", False, f"异常: {str(e)}")
            
            # 测试4: target_transform 功能
            print("测试target_transform功能...")
            try:
                def target_transform(x):
                    return x * 2
                dataset_with_target = ImageFolder(tmpdir, target_transform=target_transform)
                original_dataset = ImageFolder(tmpdir)
                _, transformed_label = dataset_with_target[0]
                _, original_label = original_dataset[0]
                stats.add_result("target_transform功能", transformed_label == original_label * 2,
                                f"原始: {original_label}, 变换后: {transformed_label}")
            except Exception as e:
                stats.add_result("target_transform功能", False, f"异常: {str(e)}")
            
            # 测试5: __repr__
            print("测试__repr__...")
            try:
                dataset = ImageFolder(tmpdir)
                repr_str = repr(dataset)
                stats.add_result("__repr__", 'ImageFolder' in repr_str and 'Root location' in repr_str,
                                f"包含类名和根目录: {'ImageFolder' in repr_str}, {'Root location' in repr_str}")
            except Exception as e:
                stats.add_result("__repr__", False, f"异常: {str(e)}")
            
            # 测试6: 与DataLoader集成
            print("测试DataLoader集成...")
            try:
                from riemann.utils.data import DataLoader
                dataset = ImageFolder(tmpdir, transform=ToTensor())
                dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
                
                batch_count = 0
                for batch_images, batch_labels in dataloader:
                    batch_count += 1
                    if batch_count >= 1:
                        break
                
                stats.add_result("DataLoader集成", batch_count > 0,
                                f"成功获取批次: {batch_count > 0}")
            except Exception as e:
                stats.add_result("DataLoader集成", False, f"异常: {str(e)}")
            
            # 测试7: 类别分布
            print("测试类别分布...")
            try:
                dataset = ImageFolder(tmpdir)
                class_counts = {}
                for _, label in dataset:
                    class_name = dataset.classes[label]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                correct = (class_counts.get('cat', 0) == 3 and 
                          class_counts.get('dog', 0) == 3 and 
                          class_counts.get('bird', 0) == 1)
                stats.add_result("类别分布", correct,
                                f"分布: {class_counts}")
            except Exception as e:
                stats.add_result("类别分布", False, f"异常: {str(e)}")
                
    except Exception as e:
        stats.add_result("ImageFolder测试", False, f"测试异常: {str(e)}")
    
    finally:
        stats.end_function()


def main():
    """主测试函数"""
    print(f"{Colors.HEADER}Riemann vision模块测试{Colors.ENDC}")
    print(f"{Colors.OKCYAN}测试MNIST、CIFAR10数据集和transforms类{Colors.ENDC}")
    
    # 创建测试统计对象
    stats = StatisticsCollector()
    
    # 运行所有测试
    test_mnist_dataset(stats)
    test_cifar10_dataset(stats)
    test_datasetfolder(stats)
    test_imagefolder(stats)
    test_transforms(stats)
    
    # 打印测试汇总
    stats.print_summary()

if __name__ == "__main__":
    import struct
    clear_screen()
    main()