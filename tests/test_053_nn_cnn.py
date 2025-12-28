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

import riemann as rm
from riemann.nn import Module
from riemann.nn import Conv1d, Conv2d, Conv3d, MaxPool1d, MaxPool2d, MaxPool3d, AvgPool1d, AvgPool2d, AvgPool3d
from riemann.nn import ReLU
from riemann.nn import Linear
from riemann.nn import CrossEntropyLoss

import torch
import torch.nn as tnn


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

def copy_weights_torch_to_riemann(torch_layer, riemann_layer):
    """将PyTorch层的权重复制到Riemann层，使用深拷贝避免内存共享"""
    if hasattr(torch_layer, 'weight') and hasattr(riemann_layer, 'weight'):
        # 现在Linear层的权重格式与PyTorch一致，都是 [out_features, in_features]
        if isinstance(torch_layer, tnn.Linear) and isinstance(riemann_layer, Linear):
            torch_weight = torch_layer.weight.detach().numpy()  # [out_features, in_features]
            # 使用深拷贝创建独立的numpy数组，避免内存共享
            torch_weight_copy = np.array(torch_weight, copy=True)
            # 直接复制，不需要转置
            riemann_layer.weight.data = torch_weight_copy
        else:
            # 对于卷积层，权重格式相同
            torch_weight = torch_layer.weight.detach().numpy()
            # 使用深拷贝创建独立的numpy数组，避免内存共享
            torch_weight_copy = np.array(torch_weight, copy=True)
            # 直接赋值numpy数组，避免创建嵌套TN对象
            riemann_layer.weight.data = torch_weight_copy
    
    if hasattr(torch_layer, 'bias') and hasattr(riemann_layer, 'bias'):
        if torch_layer.bias is not None:
            torch_bias = torch_layer.bias.detach().numpy()
            # 使用深拷贝创建独立的numpy数组，避免内存共享
            torch_bias_copy = np.array(torch_bias, copy=True)
            # 直接赋值numpy数组，避免创建嵌套TN对象
            riemann_layer.bias.data = torch_bias_copy
        else:
            riemann_layer.register_parameter('bias', None)


def tensor_allclose(rm_tensor, torch_tensor, rtol=1e-4, atol=1e-6):
    """比较Riemann张量和PyTorch张量是否接近"""
    rm_data = rm_tensor.data if hasattr(rm_tensor, 'data') else rm_tensor
    torch_data = torch_tensor.detach().numpy()
    return np.allclose(rm_data, torch_data, rtol=rtol, atol=atol)


def compare_gradients(torch_param, riemann_param, name="parameter"):
    """比较PyTorch和Riemann参数的梯度"""
    if torch_param.grad is None and riemann_param.grad is None:
        return True, "两个梯度都为None"
    
    if torch_param.grad is None or riemann_param.grad is None:
        return False, f"梯度状态不一致: torch={torch_param.grad is not None}, riemann={riemann_param.grad is not None}"
    
    torch_grad = torch_param.grad.detach().numpy()
    riemann_grad = riemann_param.grad.data if hasattr(riemann_param.grad, 'data') else riemann_param.grad
    
    close_result = np.allclose(torch_grad, riemann_grad, rtol=1e-4, atol=1e-6)
    diff = np.abs(torch_grad - riemann_grad).max()
    
    return close_result, f"梯度差异: {diff:.6f}"


def compare_network(rm_net, torch_net, input_data, targets, pool_type, network_type, stats):
    """测试Riemann网络与PyTorch网络的比较"""
    try:
        # 复制权重，确保两个网络参数完全一致
        copy_weights_torch_to_riemann(torch_net.conv1, rm_net.conv1)
        copy_weights_torch_to_riemann(torch_net.fc1, rm_net.fc1)
        copy_weights_torch_to_riemann(torch_net.fc2, rm_net.fc2)
        
        # 确保输入数据完全一致：使用深拷贝创建独立的numpy数组，避免内存共享
        input_data_copy = np.array(input_data, copy=True)
        rm_input = rm.tensor(input_data_copy, requires_grad=True)
        torch_input = torch.tensor(input_data_copy, requires_grad=True)
        
        # 确保目标数据完全一致：使用深拷贝创建独立的numpy数组，避免内存共享
        targets_copy = np.array(targets, copy=True)
        rm_targets = rm.tensor(targets_copy)
        torch_targets = torch.tensor(targets_copy, dtype=torch.long)
        
        # 验证输入张量形状一致
        input_shape_match = rm_input.shape == torch_input.shape
        stats.add_result(f"网络比较 {network_type} {pool_type.upper()}池化-输入形状", input_shape_match,
                        f"Riemann: {rm_input.shape}, PyTorch: {torch_input.shape}")
        
        # 前向传播
        rm_output = rm_net(rm_input)
        torch_output = torch_net(torch_input)
        
        # 比较前向传播结果
        forward_close = tensor_allclose(rm_output, torch_output, rtol=1e-4, atol=1e-6)
        stats.add_result(f"网络比较 {network_type} {pool_type.upper()}池化-前向传播", forward_close, 
                        f"输出形状: rm={rm_output.shape}, torch={torch_output.shape}")
        
        # 计算损失
        rm_loss = rm_net.criterion(rm_output, rm_targets)
        torch_loss = torch_net.criterion(torch_output, torch_targets)
        
        # 比较损失
        loss_close = tensor_allclose(rm_loss, torch_loss, rtol=1e-4, atol=1e-6)
        stats.add_result(f"网络比较 {network_type} {pool_type.upper()}池化-损失计算", loss_close,
                        f"损失值: rm={rm_loss.data:.6f}, torch={torch_loss.item():.6f}")
        
        # 反向传播
        rm_loss.backward()
        torch_loss.backward()
        
        # 比较梯度
        gradient_tests_passed = 0
        gradient_tests_total = 0
        
        # 比较conv1梯度
        gradient_tests_total += 2
        conv1_weight_close, conv1_weight_msg = compare_gradients(torch_net.conv1.weight, rm_net.conv1.weight, "conv1.weight")
        stats.add_result(f"网络比较 {network_type} {pool_type.upper()}池化-conv1.weight梯度", conv1_weight_close, conv1_weight_msg)
        if conv1_weight_close:
            gradient_tests_passed += 1
        
        if torch_net.conv1.bias is not None:
            conv1_bias_close, conv1_bias_msg = compare_gradients(torch_net.conv1.bias, rm_net.conv1.bias, "conv1.bias")
            stats.add_result(f"网络比较 {network_type} {pool_type.upper()}池化-conv1.bias梯度", conv1_bias_close, conv1_bias_msg)
            if conv1_bias_close:
                gradient_tests_passed += 1
        
        # 比较fc1梯度
        gradient_tests_total += 2
        fc1_weight_close, fc1_weight_msg = compare_gradients(torch_net.fc1.weight, rm_net.fc1.weight, "fc1.weight")
        stats.add_result(f"网络比较 {network_type} {pool_type.upper()}池化-fc1.weight梯度", fc1_weight_close, fc1_weight_msg)
        if fc1_weight_close:
            gradient_tests_passed += 1
        
        fc1_bias_close, fc1_bias_msg = compare_gradients(torch_net.fc1.bias, rm_net.fc1.bias, "fc1.bias")
        stats.add_result(f"网络比较 {network_type} {pool_type.upper()}池化-fc1.bias梯度", fc1_bias_close, fc1_bias_msg)
        if fc1_bias_close:
            gradient_tests_passed += 1
        
        # 比较fc2梯度
        gradient_tests_total += 2
        fc2_weight_close, fc2_weight_msg = compare_gradients(torch_net.fc2.weight, rm_net.fc2.weight, "fc2.weight")
        stats.add_result(f"网络比较 {network_type} {pool_type.upper()}池化-fc2.weight梯度", fc2_weight_close, fc2_weight_msg)
        if fc2_weight_close:
            gradient_tests_passed += 1
        
        fc2_bias_close, fc2_bias_msg = compare_gradients(torch_net.fc2.bias, rm_net.fc2.bias, "fc2.bias")
        stats.add_result(f"网络比较 {network_type} {pool_type.upper()}池化-fc2.bias梯度", fc2_bias_close, fc2_bias_msg)
        if fc2_bias_close:
            gradient_tests_passed += 1
        
        # 梯度总体通过率
        gradient_pass_rate = gradient_tests_passed / gradient_tests_total * 100
        stats.add_result(f"网络比较 {network_type} {pool_type.upper()}池化-梯度总体", gradient_pass_rate >= 90,
                        f"梯度通过率: {gradient_pass_rate:.1f}% ({gradient_tests_passed}/{gradient_tests_total})")
        
    except Exception as e:
        stats.add_result(f"网络比较 {network_type} {pool_type.upper()}池化", False, f"测试异常: {str(e)}")


# ==================== 测试函数 ====================

def test_cnn_1d(stats=None):
    """测试1D卷积网络"""
    # 如果没有传入stats实例，创建一个（用于pytest调用）
    if stats is None:
        stats = StatisticsCollector()
    
    stats.start_function("1D卷积网络")
    
    try:
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
        
        for pool_type in pool_types:
            print(f"测试1D {pool_type.upper()}池化...")
            
            # 修复：直接进行网络比较，避免重复使用PyTorch网络
            print(f"  直接比较Riemann网络与PyTorch网络...")
            rm_net = RiemannCNN1D(pool_type=pool_type)
            torch_net = TorchCNN1D(pool_type=pool_type)
            compare_network(rm_net, torch_net, input_data, targets, pool_type, "1D", stats)
            
    finally:
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
        
        for pool_type in pool_types:
            print(f"测试2D {pool_type.upper()}池化...")
            
            # 修复：直接进行网络比较，避免重复使用PyTorch网络
            print(f"  直接比较Riemann网络与PyTorch网络...")
            rm_net = RiemannCNN2D(pool_type=pool_type)
            torch_net = TorchCNN2D(pool_type=pool_type)
            compare_network(rm_net, torch_net, input_data, targets, pool_type, "2D", stats)
            
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
        
        for pool_type in pool_types:
            print(f"测试3D {pool_type.upper()}池化...")
            
            # 修复：直接进行网络比较，避免重复使用PyTorch网络
            print(f"  直接比较Riemann网络与PyTorch网络...")
            rm_net = RiemannCNN3D(pool_type=pool_type)
            torch_net = TorchCNN3D(pool_type=pool_type)
            compare_network(rm_net, torch_net, input_data, targets, pool_type, "3D", stats)
            
    finally:
        stats.end_function()


def main():
    """主测试函数"""
    print(f"{Colors.HEADER}Riemann nn.conv 卷积和池化函数测试{Colors.ENDC}")
    print(f"{Colors.OKCYAN}测试1D、2D、3D卷积网络的前向传播和反向传播{Colors.ENDC}")
    
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