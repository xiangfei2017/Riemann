#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Riemann nn.conv 卷积和池化函数测试脚本

测试脚本文件名：test_051_nn_cnn.py
存在tests目录下，可以作为独立脚本运行，也可以被pytest调用测试

参考tests\test_005_shape_operations.py里的测试框架、测试统计输出代码

对1D、2D、3D场景，各设计一个测试用例，每个测试用例里用nn模块里的模块构建一个基本的卷积网络，
其中包括1个卷积层、1个ReLU激活函数层、1个池化层、1个展平层、1个全连接层、1个输出层、1个交叉熵损失函数层

每个用例包含两个子用例，分别使用Max池化和均值池化

每个子用例均要比较riemann的网络和torch的网络，二者输入数据一样，网络结构完全一样，
比较前向函数结果值和权重参数的反向跟踪梯度

测试代码结构清晰、简洁，重复代码设计为函数以便重用
"""

import sys
import os
import time
import numpy as np
from typing import Tuple, Dict, Any

# 添加Riemann库路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# 导入riemann模块
try:
    import riemann as rm
    # 从rm.cuda获取cupy引用和CUDA可用性
    CUDA_AVAILABLE = rm.cuda.CUPY_AVAILABLE
    cp = rm.cuda.cp
    from riemann.nn import (
        Conv1d, Conv2d, Conv3d, 
        MaxPool1d, MaxPool2d, MaxPool3d, 
        AvgPool1d, AvgPool2d, AvgPool3d,
        ReLU,Linear, CrossEntropyLoss
    )
except ImportError as e:
    print(f"无法导入riemann模块: {e}")
    print("请确保项目路径设置正确")
    sys.exit(1)

# 尝试导入PyTorch进行比较
try:
    import torch
    import torch.nn as tnn
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        print("PyTorch CUDA可用")
    else:
        print("PyTorch CUDA不可用")
except ImportError:
    print("警告: 无法导入PyTorch，将只测试riemann的形状操作函数")
    TORCH_AVAILABLE = False


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


# ==================== 网络定义 ====================

# 1D卷积网络 - 确保Riemann和PyTorch网络结构完全一致
class RiemannCNN1D(rm.nn.Module):
    """Riemann 1D卷积网络"""
    def __init__(self, pool_type='max'):
        super().__init__()
        # 使用与PyTorch完全相同的层参数
        self.conv1 = Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = ReLU()
        if pool_type == 'max':
            self.pool = MaxPool1d(kernel_size=2, stride=2)
        else:
            self.pool = AvgPool1d(kernel_size=2, stride=2)
        
        # 计算展平后的特征维度：输入长度50 -> 卷积后50 -> 池化后25
        self.flatten_size = 16 * 25  # out_channels * pooled_length
        self.fc1 = Linear(self.flatten_size, 64)
        self.relu_fc = ReLU()  # 添加激活函数，增加非线性表达能力
        self.fc2 = Linear(64, 10)  # 10分类
        self.criterion = CrossEntropyLoss()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.flatten(1)  # 展平除batch维外的所有维度
        x = self.fc1(x)
        x = self.relu_fc(x)  # 添加非线性激活
        x = self.fc2(x)
        return x


class TorchCNN1D(tnn.Module):
    """PyTorch 1D卷积网络 - 与Riemann网络结构完全一致"""
    def __init__(self, pool_type='max'):
        super().__init__()
        # 使用与Riemann完全相同的层参数
        self.conv1 = tnn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = tnn.ReLU()
        if pool_type == 'max':
            self.pool = tnn.MaxPool1d(kernel_size=2, stride=2)
        else:
            self.pool = tnn.AvgPool1d(kernel_size=2, stride=2)
        
        # 计算展平后的特征维度：输入长度50 -> 卷积后50 -> 池化后25
        self.flatten_size = 16 * 25  # out_channels * pooled_length
        self.fc1 = tnn.Linear(self.flatten_size, 64)
        self.relu_fc = tnn.ReLU()  # 添加激活函数，增加非线性表达能力
        self.fc2 = tnn.Linear(64, 10)
        self.criterion = tnn.CrossEntropyLoss()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu_fc(x)  # 添加非线性激活
        x = self.fc2(x)
        return x


# 2D卷积网络 - 确保Riemann和PyTorch网络结构完全一致
class RiemannCNN2D(rm.nn.Module):
    """Riemann 2D卷积网络"""
    def __init__(self, pool_type='max'):
        super().__init__()
        # 使用与PyTorch完全相同的层参数
        self.conv1 = Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = ReLU()
        if pool_type == 'max':
            self.pool = MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = AvgPool2d(kernel_size=2, stride=2)
        
        # 计算展平后的特征维度：输入32x32 -> 卷积后32x32 -> 池化后16x16
        self.flatten_size = 16 * 16 * 16  # out_channels * height * width
        self.fc1 = Linear(self.flatten_size, 64)
        self.relu_fc = ReLU()  # 添加激活函数，增加非线性表达能力
        self.fc2 = Linear(64, 10)  # 10分类
        self.criterion = CrossEntropyLoss()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.flatten(1)  # 展平除batch维外的所有维度
        x = self.fc1(x)
        x = self.relu_fc(x)  # 添加非线性激活
        x = self.fc2(x)
        return x


class TorchCNN2D(tnn.Module):
    """PyTorch 2D卷积网络 - 与Riemann网络结构完全一致"""
    def __init__(self, pool_type='max'):
        super().__init__()
        # 使用与Riemann完全相同的层参数
        self.conv1 = tnn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = tnn.ReLU()
        if pool_type == 'max':
            self.pool = tnn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = tnn.AvgPool2d(kernel_size=2, stride=2)
        
        # 计算展平后的特征维度：输入32x32 -> 卷积后32x32 -> 池化后16x16
        self.flatten_size = 16 * 16 * 16  # out_channels * height * width
        self.fc1 = tnn.Linear(self.flatten_size, 64)
        self.relu_fc = tnn.ReLU()  # 添加激活函数，增加非线性表达能力
        self.fc2 = tnn.Linear(64, 10)
        self.criterion = tnn.CrossEntropyLoss()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu_fc(x)  # 添加非线性激活
        x = self.fc2(x)
        return x


# 3D卷积网络 - 确保Riemann和PyTorch网络结构完全一致
class RiemannCNN3D(rm.nn.Module):
    """Riemann 3D卷积网络"""
    def __init__(self, pool_type='max'):
        super().__init__()
        # 使用与PyTorch完全相同的层参数
        self.conv1 = Conv3d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.relu = ReLU()
        if pool_type == 'max':
            self.pool = MaxPool3d(kernel_size=2, stride=2)
        else:
            self.pool = AvgPool3d(kernel_size=2, stride=2)
        
        # 计算展平后的特征维度：输入16x16x16 -> 卷积后16x16x16 -> 池化后8x8x8
        self.flatten_size = 8 * 8 * 8 * 8  # out_channels * depth * height * width
        self.fc1 = Linear(self.flatten_size, 32)
        self.relu_fc = ReLU()  # 添加激活函数，增加非线性表达能力
        self.fc2 = Linear(32, 5)  # 5分类
        self.criterion = CrossEntropyLoss()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.flatten(1)  # 展平除batch维外的所有维度
        x = self.fc1(x)
        x = self.relu_fc(x)  # 添加非线性激活
        x = self.fc2(x)
        return x


class TorchCNN3D(tnn.Module):
    """PyTorch 3D卷积网络 - 与Riemann网络结构完全一致"""
    def __init__(self, pool_type='max'):
        super().__init__()
        # 使用与Riemann完全相同的层参数
        self.conv1 = tnn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.relu = tnn.ReLU()
        if pool_type == 'max':
            self.pool = tnn.MaxPool3d(kernel_size=2, stride=2)
        else:
            self.pool = tnn.AvgPool3d(kernel_size=2, stride=2)
        
        # 计算展平后的特征维度：输入16x16x16 -> 卷积后16x16x16 -> 池化后8x8x8
        self.flatten_size = 8 * 8 * 8 * 8  # out_channels * depth * height * width
        self.fc1 = tnn.Linear(self.flatten_size, 32)
        self.relu_fc = tnn.ReLU()  # 添加激活函数，增加非线性表达能力
        self.fc2 = tnn.Linear(32, 5)
        self.criterion = tnn.CrossEntropyLoss()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu_fc(x)  # 添加非线性激活
        x = self.fc2(x)
        return x


# ==================== 工具函数 ====================

def copy_weights_torch_to_riemann(torch_layer, riemann_layer, device="cpu"):
    """将PyTorch层的权重复制到Riemann层，使用深拷贝避免内存共享"""
    if hasattr(torch_layer, 'weight') and hasattr(riemann_layer, 'weight'):
        # 现在Linear层的权重格式与PyTorch一致，都是 [out_features, in_features]
        if isinstance(torch_layer, tnn.Linear) and isinstance(riemann_layer, Linear):
            torch_weight = torch_layer.weight.detach().cpu().numpy()  # [out_features, in_features]
            # 使用深拷贝创建独立的numpy数组，避免内存共享
            torch_weight_copy = np.array(torch_weight, copy=True)
            # 创建一个新的Riemann张量
            if device == "cuda" and CUDA_AVAILABLE:
                # 创建CPU张量，然后移动到CUDA设备
                new_weight_tensor = rm.tensor(torch_weight_copy, requires_grad=True)
                # 移动到CUDA设备，然后detach_()并设置is_leaf=True
                new_weight_tensor = new_weight_tensor.to(device).detach_()
                new_weight_tensor.requires_grad = True
                new_weight_tensor.is_leaf = True
                # 创建Parameter对象
                new_weight = rm.nn.Parameter(new_weight_tensor, requires_grad=True)
            else:
                # 创建CPU张量
                new_weight_tensor = rm.tensor(torch_weight_copy, requires_grad=True)
                new_weight_tensor.is_leaf = True
                # 创建Parameter对象
                new_weight = rm.nn.Parameter(new_weight_tensor, requires_grad=True)
            # 替换权重
            riemann_layer.weight = new_weight
        else:
            # 对于卷积层，权重格式相同
            torch_weight = torch_layer.weight.detach().cpu().numpy()
            # 使用深拷贝创建独立的numpy数组，避免内存共享
            torch_weight_copy = np.array(torch_weight, copy=True)
            # 创建一个新的Riemann张量
            if device == "cuda" and CUDA_AVAILABLE:
                # 创建CPU张量，然后移动到CUDA设备
                new_weight_tensor = rm.tensor(torch_weight_copy, requires_grad=True)
                # 移动到CUDA设备，然后detach_()并设置is_leaf=True
                new_weight_tensor = new_weight_tensor.to(device).detach_()
                new_weight_tensor.requires_grad = True
                new_weight_tensor.is_leaf = True
                # 创建Parameter对象
                new_weight = rm.nn.Parameter(new_weight_tensor, requires_grad=True)
            else:
                # 创建CPU张量
                new_weight_tensor = rm.tensor(torch_weight_copy, requires_grad=True)
                new_weight_tensor.is_leaf = True
                # 创建Parameter对象
                new_weight = rm.nn.Parameter(new_weight_tensor, requires_grad=True)
            # 替换权重
            riemann_layer.weight = new_weight
    
    if hasattr(torch_layer, 'bias') and hasattr(riemann_layer, 'bias'):
        if torch_layer.bias is not None:
            torch_bias = torch_layer.bias.detach().cpu().numpy()
            # 使用深拷贝创建独立的numpy数组，避免内存共享
            torch_bias_copy = np.array(torch_bias, copy=True)
            # 创建一个新的Riemann张量
            if device == "cuda" and CUDA_AVAILABLE:
                # 创建CPU张量，然后移动到CUDA设备
                new_bias_tensor = rm.tensor(torch_bias_copy, requires_grad=True)
                # 移动到CUDA设备，然后detach_()并设置is_leaf=True
                new_bias_tensor = new_bias_tensor.to(device).detach_()
                new_bias_tensor.requires_grad = True
                new_bias_tensor.is_leaf = True
                # 创建Parameter对象
                new_bias = rm.nn.Parameter(new_bias_tensor, requires_grad=True)
            else:
                # 创建CPU张量
                new_bias_tensor = rm.tensor(torch_bias_copy, requires_grad=True)
                new_bias_tensor.is_leaf = True
                # 创建Parameter对象
                new_bias = rm.nn.Parameter(new_bias_tensor, requires_grad=True)
            # 替换偏置
            riemann_layer.bias = new_bias
        else:
            riemann_layer.register_parameter('bias', None)


def tensor_allclose(rm_tensor, torch_tensor, rtol=1e-4, atol=1e-6):
    """比较Riemann张量和PyTorch张量是否接近"""
    rm_data = rm_tensor.data if hasattr(rm_tensor, 'data') else rm_tensor
    torch_data = torch_tensor.detach().numpy()
    return np.allclose(rm_data, torch_data, rtol=rtol, atol=atol)


def compare_values(rm_result, torch_result, atol=1e-6, rtol=1e-6):
    """比较Riemann和PyTorch的值是否接近"""
    # 处理None值的情况
    if not TORCH_AVAILABLE:
        # 如果没有PyTorch，只检查riemann结果是否存在
        return rm_result is not None
    
    if rm_result is None and torch_result is None:
        return True
    if rm_result is None or torch_result is None:
        return False
    
    # 处理嵌套元组/列表的情况
    if isinstance(rm_result, (list, tuple)) and isinstance(torch_result, (list, tuple)):
        if len(rm_result) != len(torch_result):
            return False
        
        all_passed = True
        for r, t in zip(rm_result, torch_result):
            if not compare_values(r, t, atol, rtol):
                all_passed = False
                break
        
        return all_passed
    
    # 转换为numpy数组
    try:
        # 处理Riemann结果
        if hasattr(rm_result, 'is_cuda') and rm_result.is_cuda:
            # 如果是CUDA张量，先移动到CPU
            rm_data = rm_result.detach().cpu().numpy()
        else:
            rm_data = rm_result.detach().numpy()
        
        # 处理PyTorch结果
        if hasattr(torch_result, 'is_cuda') and torch_result.is_cuda:
            # 如果是CUDA张量，先移动到CPU
            torch_data = torch_result.detach().cpu().numpy()
        else:
            torch_data = torch_result.detach().numpy()
    except Exception as e:
        print(f"比较值转换错误: {e}")
        return False
    
    # 处理形状不匹配的情况
    try:
        np.testing.assert_allclose(rm_data, torch_data, rtol=rtol, atol=atol)
        return True
    except AssertionError:
        return False


def compare_gradients(torch_param, riemann_param, name="parameter"):
    """比较PyTorch和Riemann参数的梯度"""
    if torch_param.grad is None and riemann_param.grad is None:
        return True, "两个梯度都为None"
    
    if torch_param.grad is None:
        return False, f"PyTorch梯度为None"
    
    # 对于Riemann，即使grad为None，也尝试从data.grad获取
    if riemann_param.grad is None:
        # 尝试从data.grad获取梯度
        if hasattr(riemann_param, 'data') and hasattr(riemann_param.data, 'grad'):
            riemann_grad = riemann_param.data.grad
            if riemann_grad is None:
                return False, f"Riemann梯度为None"
        else:
            return False, f"Riemann梯度为None"
    else:
        riemann_grad = riemann_param.grad
    
    # 处理CUDA张量
    if hasattr(torch_param.grad, 'is_cuda') and torch_param.grad.is_cuda:
        torch_grad = torch_param.grad.detach().cpu().numpy()
    else:
        torch_grad = torch_param.grad.detach().numpy()
    
    if hasattr(riemann_grad, 'is_cuda') and riemann_grad.is_cuda:
        riemann_grad_data = riemann_grad.detach().cpu().numpy()
    else:
        riemann_grad_data = riemann_grad.data if hasattr(riemann_grad, 'data') else riemann_grad
    
    # 确保riemann_grad_data是numpy数组
    if not isinstance(riemann_grad_data, np.ndarray):
        try:
            riemann_grad_data = np.array(riemann_grad_data)
        except Exception as e:
            return False, f"无法转换Riemann梯度为numpy数组: {e}"
    
    close_result = np.allclose(torch_grad, riemann_grad_data, rtol=1e-4, atol=1e-6)
    diff = np.abs(torch_grad - riemann_grad_data).max()
    
    return close_result, f"梯度差异: {diff:.6f}"


def compare_network(rm_net, torch_net, input_data, targets, pool_type, network_type, stats, device="cpu"):
    """测试Riemann网络与PyTorch网络的比较"""
    # 确保网络在正确的设备上初始化
    # 先创建新的网络实例，确保它们在正确的设备上
    if network_type == "1D":
        rm_net = RiemannCNN1D(pool_type=pool_type)
        torch_net = TorchCNN1D(pool_type=pool_type)
    elif network_type == "2D":
        rm_net = RiemannCNN2D(pool_type=pool_type)
        torch_net = TorchCNN2D(pool_type=pool_type)
    elif network_type == "3D":
        rm_net = RiemannCNN3D(pool_type=pool_type)
        torch_net = TorchCNN3D(pool_type=pool_type)
    
    # 将网络移动到指定设备
    if device == "cuda":
        if CUDA_AVAILABLE:
            # 先移动整个网络
            rm_net.to(device)
            # 再确保所有层都移动到CUDA设备
            rm_net.conv1.to(device)
            rm_net.fc1.to(device)
            rm_net.fc2.to(device)
            # 确保池化层也移动到CUDA设备
            rm_net.pool.to(device)
        if torch.cuda.is_available():
            torch_net.to(device)
    
    # 复制权重，确保两个网络参数完全一致
    copy_weights_torch_to_riemann(torch_net.conv1, rm_net.conv1, device=device)
    copy_weights_torch_to_riemann(torch_net.fc1, rm_net.fc1, device=device)
    copy_weights_torch_to_riemann(torch_net.fc2, rm_net.fc2, device=device)
    
    # 确保输入数据完全一致：使用深拷贝创建独立的numpy数组，避免内存共享
    input_data_copy = np.array(input_data, copy=True)
    rm_input = rm.tensor(input_data_copy, requires_grad=True, device=device)
    torch_input = torch.tensor(input_data_copy, requires_grad=True, device=device if device == "cuda" and torch.cuda.is_available() else "cpu")
    
    # 确保目标数据完全一致：使用深拷贝创建独立的numpy数组，避免内存共享
    targets_copy = np.array(targets, copy=True)
    rm_targets = rm.tensor(targets_copy, device=device)
    torch_targets = torch.tensor(targets_copy, dtype=torch.long, device=device if device == "cuda" and torch.cuda.is_available() else "cpu")
    
    # 验证输入张量形状一致
    input_shape_match = rm_input.shape == torch_input.shape
    stats.add_result(f"网络比较 {network_type} {pool_type.upper()}池化-{device}-输入形状", input_shape_match,
                    f"Riemann: {rm_input.shape}, PyTorch: {torch_input.shape}")
    
    # 前向传播
    rm_output = rm_net(rm_input)
    torch_output = torch_net(torch_input)
    
    # 比较前向传播结果
    forward_close = compare_values(rm_output, torch_output)
    stats.add_result(f"网络比较 {network_type} {pool_type.upper()}池化-{device}-前向传播", forward_close, 
                    f"输出形状: rm={rm_output.shape}, torch={torch_output.shape}")
    
    # 计算损失
    rm_loss = rm_net.criterion(rm_output, rm_targets)
    torch_loss = torch_net.criterion(torch_output, torch_targets)
    
    # 比较损失
    loss_close = compare_values(rm_loss, torch_loss)
    stats.add_result(f"网络比较 {network_type} {pool_type.upper()}池化-{device}-损失计算", loss_close,
                    f"损失值: rm={rm_loss.data:.6f}, torch={torch_loss.item():.6f}")
    
    # 执行完整的梯度测试，无论设备类型
    # 反向传播
    rm_loss.backward()
    torch_loss.backward()
    
    # 比较梯度
    gradient_tests_passed = 0
    gradient_tests_total = 0
    
    # 比较conv1梯度
    gradient_tests_total += 2
    conv1_weight_close, conv1_weight_msg = compare_gradients(torch_net.conv1.weight, rm_net.conv1.weight, "conv1.weight")
    stats.add_result(f"网络比较 {network_type} {pool_type.upper()}池化-{device}-conv1.weight梯度", conv1_weight_close, conv1_weight_msg)
    if conv1_weight_close:
        gradient_tests_passed += 1
    
    if torch_net.conv1.bias is not None:
        conv1_bias_close, conv1_bias_msg = compare_gradients(torch_net.conv1.bias, rm_net.conv1.bias, "conv1.bias")
        stats.add_result(f"网络比较 {network_type} {pool_type.upper()}池化-{device}-conv1.bias梯度", conv1_bias_close, conv1_bias_msg)
        if conv1_bias_close:
            gradient_tests_passed += 1
    
    # 比较fc1梯度
    gradient_tests_total += 2
    fc1_weight_close, fc1_weight_msg = compare_gradients(torch_net.fc1.weight, rm_net.fc1.weight, "fc1.weight")
    stats.add_result(f"网络比较 {network_type} {pool_type.upper()}池化-{device}-fc1.weight梯度", fc1_weight_close, fc1_weight_msg)
    if fc1_weight_close:
        gradient_tests_passed += 1
    
    fc1_bias_close, fc1_bias_msg = compare_gradients(torch_net.fc1.bias, rm_net.fc1.bias, "fc1.bias")
    stats.add_result(f"网络比较 {network_type} {pool_type.upper()}池化-{device}-fc1.bias梯度", fc1_bias_close, fc1_bias_msg)
    if fc1_bias_close:
        gradient_tests_passed += 1
    
    # 比较fc2梯度
    gradient_tests_total += 2
    fc2_weight_close, fc2_weight_msg = compare_gradients(torch_net.fc2.weight, rm_net.fc2.weight, "fc2.weight")
    stats.add_result(f"网络比较 {network_type} {pool_type.upper()}池化-{device}-fc2.weight梯度", fc2_weight_close, fc2_weight_msg)
    if fc2_weight_close:
        gradient_tests_passed += 1
    
    fc2_bias_close, fc2_bias_msg = compare_gradients(torch_net.fc2.bias, rm_net.fc2.bias, "fc2.bias")
    stats.add_result(f"网络比较 {network_type} {pool_type.upper()}池化-{device}-fc2.bias梯度", fc2_bias_close, fc2_bias_msg)
    if fc2_bias_close:
        gradient_tests_passed += 1
    
    # 梯度总体通过率
    gradient_pass_rate = gradient_tests_passed / gradient_tests_total * 100
    stats.add_result(f"网络比较 {network_type} {pool_type.upper()}池化-{device}-梯度总体", gradient_pass_rate >= 90,
                    f"梯度通过率: {gradient_pass_rate:.1f}% ({gradient_tests_passed}/{gradient_tests_total})")


# ==================== 测试函数 ====================

def test_cnn_1d(stats=None):
    """测试1D卷积网络"""
    # 如果没有传入stats实例，创建一个（用于pytest调用）
    if stats is None:
        stats = StatisticsCollector()
    
    stats.start_function("1D卷积网络")
    
    # 设置随机种子确保可重现性
    np.random.seed(42)
    
    # 测试数据（只生成一次）
    batch_size = 4
    input_length = 50
    input_data = np.random.randn(batch_size, 1, input_length).astype(np.float32)
    targets = np.random.randint(0, 10, batch_size)
    
    print(f"输入数据形状: {input_data.shape}")
    print(f"目标: {targets}")
    print(f"输入数据范围: [{input_data.min():.6f}, {input_data.max():.6f}]")
    print()
    
    pool_types = ['max', 'avg']
    devices = ["cpu"]
    if CUDA_AVAILABLE and torch.cuda.is_available():
        devices.append("cuda")
    
    for pool_type in pool_types:
        for device in devices:
            print(f"测试1D {pool_type.upper()}池化 - {device}...")
            
            # 直接进行网络比较
            print(f"  直接比较Riemann网络与PyTorch网络...")
            rm_net = RiemannCNN1D(pool_type=pool_type)
            torch_net = TorchCNN1D(pool_type=pool_type)
            compare_network(rm_net, torch_net, input_data, targets, pool_type, "1D", stats, device=device)
            
    stats.end_function()


def test_cnn_2d(stats=None):
    """测试2D卷积网络"""
    # 如果没有传入stats实例，创建一个（用于pytest调用）
    if stats is None:
        stats = StatisticsCollector()
    
    stats.start_function("2D卷积网络")
    
    try:
        # 设置随机种子确保可重现性
        np.random.seed(42)
        
        # 测试数据
        batch_size = 4
        input_shape = (3, 32, 32)  # 3通道，32x32图像
        input_data = np.random.randn(batch_size, *input_shape).astype(np.float32)
        targets = np.random.randint(0, 10, batch_size)
        
        print(f"输入数据形状: {input_data.shape}")
        print(f"目标: {targets}")
        print(f"输入数据范围: [{input_data.min():.6f}, {input_data.max():.6f}]")
        print()
        
        pool_types = ['max', 'avg']
        devices = ["cpu"]
        if CUDA_AVAILABLE and torch.cuda.is_available():
            devices.append("cuda")
        
        for pool_type in pool_types:
            for device in devices:
                print(f"测试2D {pool_type.upper()}池化 - {device}...")
                
                # 修复：直接进行网络比较，避免重复使用PyTorch网络
                print(f"  直接比较Riemann网络与PyTorch网络...")
                rm_net = RiemannCNN2D(pool_type=pool_type)
                torch_net = TorchCNN2D(pool_type=pool_type)
                compare_network(rm_net, torch_net, input_data, targets, pool_type, "2D", stats, device=device)
                
    finally:
        stats.end_function()


def test_cnn_3d(stats=None):
    """测试3D卷积网络"""
    # 如果没有传入stats实例，创建一个（用于pytest调用）
    if stats is None:
        stats = StatisticsCollector()
    
    stats.start_function("3D卷积网络")
    
    try:
        # 设置随机种子确保可重现性
        np.random.seed(42)
        
        # 测试数据（只生成一次）
        batch_size = 2  # 3D数据内存占用较大，减小batch size
        input_shape = (1, 16, 16, 16)  # 1通道，16x16x16体积
        input_data = np.random.randn(batch_size, *input_shape).astype(np.float32)
        targets = np.random.randint(0, 5, batch_size)
        
        print(f"输入数据形状: {input_data.shape}")
        print(f"目标: {targets}")
        print(f"输入数据范围: [{input_data.min():.6f}, {input_data.max():.6f}]")
        print()
        
        pool_types = ['max', 'avg']
        devices = ["cpu"]
        if CUDA_AVAILABLE and torch.cuda.is_available():
            devices.append("cuda")
        
        for pool_type in pool_types:
            for device in devices:
                print(f"测试3D {pool_type.upper()}池化 - {device}...")
                
                # 修复：直接进行网络比较，避免重复使用PyTorch网络
                print(f"  直接比较Riemann网络与PyTorch网络...")
                rm_net = RiemannCNN3D(pool_type=pool_type)
                torch_net = TorchCNN3D(pool_type=pool_type)
                compare_network(rm_net, torch_net, input_data, targets, pool_type, "3D", stats, device=device)
                
    finally:
        stats.end_function()


def main():
    """主测试函数"""
    print(f"{Colors.HEADER}Riemann nn.conv 卷积和池化函数测试{Colors.ENDC}")
    print(f"{Colors.OKCYAN}测试1D、2D和3D卷积网络的前向传播和反向传播{Colors.ENDC}")
    
    # 创建测试统计对象
    stats = StatisticsCollector()
    
    # 运行所有测试
    test_cnn_1d(stats)
    test_cnn_2d(stats)
    test_cnn_3d(stats)
    
    # 打印测试汇总
    stats.print_summary()

if __name__ == "__main__":
    clear_screen()
    main()