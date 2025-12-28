#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Riemann nn.BatchNorm 批量归一化模块测试脚本

测试脚本文件名：test_051_nn_batchnorm.py
存在tests目录下，可以作为独立脚本运行，也可以被pytest调用

参考tests\test_051_nn_cnn.py里的测试框架、测试统计输出代码

对BatchNorm1d、BatchNorm2d、BatchNorm3d三个类模块，各设计一个测试用例，
每个测试用例包含多个子用例，用于覆盖参数组合（训练/评估模式、affine参数、track_running_stats参数等）

每个子用例均要比较riemann的BatchNorm模块和torch的BatchNorm模块，二者输入数据一样，模块参数完全一样，
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
from riemann.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d
from riemann.nn import MSELoss

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


# ==================== 工具函数 ====================

def copy_weights_torch_to_riemann(torch_layer, riemann_layer):
    """将PyTorch层的权重复制到Riemann层，使用深拷贝避免内存共享"""
    if hasattr(torch_layer, 'weight') and hasattr(riemann_layer, 'weight'):
        if torch_layer.weight is not None:
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
    
    # 复制运行时统计量
    if hasattr(torch_layer, 'running_mean') and hasattr(riemann_layer, 'running_mean'):
        if torch_layer.running_mean is not None:
            running_mean = torch_layer.running_mean.detach().numpy()
            running_mean_copy = np.array(running_mean, copy=True)
            riemann_layer.running_mean.data = running_mean_copy
    
    if hasattr(torch_layer, 'running_var') and hasattr(riemann_layer, 'running_var'):
        if torch_layer.running_var is not None:
            running_var = torch_layer.running_var.detach().numpy()
            running_var_copy = np.array(running_var, copy=True)
            riemann_layer.running_var.data = running_var_copy


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


def compare_batch_norm(rm_bn, torch_bn, input_data, test_name, stats, mode="train"):
    """测试Riemann BatchNorm与PyTorch BatchNorm的比较"""
    try:
        # 复制权重和运行时统计量，确保两个模块参数完全一致
        copy_weights_torch_to_riemann(torch_bn, rm_bn)
        
        # 设置模式
        if mode == "train":
            rm_bn.train()
            torch_bn.train()
        else:
            rm_bn.eval()
            torch_bn.eval()
        
        # 确保输入数据完全一致：使用深拷贝创建独立的numpy数组，避免内存共享
        input_data_copy = np.array(input_data, copy=True)
        rm_input = rm.tensor(input_data_copy, requires_grad=True)
        torch_input = torch.tensor(input_data_copy, requires_grad=True)
        
        # 验证输入张量形状一致
        input_shape_match = rm_input.shape == torch_input.shape
        stats.add_result(f"{test_name}-输入形状", input_shape_match,
                        f"Riemann: {rm_input.shape}, PyTorch: {torch_input.shape}")
        
        # 前向传播
        rm_output = rm_bn(rm_input)
        torch_output = torch_bn(torch_input)
        
        # 比较输出
        forward_close = tensor_allclose(rm_output, torch_output, rtol=1e-4, atol=1e-6)
        stats.add_result(f"{test_name}-前向传播", forward_close, 
                        f"输出形状: rm={rm_output.shape}, torch={torch_output.shape}")
        
        # 计算损失
        target_data = np.random.randn(*rm_output.shape).astype(np.float32)
        target_rm = rm.tensor(target_data)
        target_torch = torch.tensor(target_data)
        
        rm_loss = MSELoss()(rm_output, target_rm)
        torch_loss = tnn.MSELoss()(torch_output, target_torch)
        
        # 比较损失
        loss_close = tensor_allclose(rm_loss, torch_loss, rtol=1e-4, atol=1e-6)
        stats.add_result(f"{test_name}-损失计算", loss_close,
                        f"损失值: rm={rm_loss.data:.6f}, torch={torch_loss.item():.6f}")
        
        # 反向传播
        rm_loss.backward()
        torch_loss.backward()
        
        # 比较梯度
        gradient_tests_passed = 0
        gradient_tests_total = 0
        
        # 比较权重梯度
        if hasattr(rm_bn, 'weight') and rm_bn.weight is not None:
            gradient_tests_total += 1
            weight_close, weight_msg = compare_gradients(torch_bn.weight, rm_bn.weight, "weight")
            stats.add_result(f"{test_name}-weight梯度", weight_close, weight_msg)
            if weight_close:
                gradient_tests_passed += 1
        
        # 比较偏置梯度
        if hasattr(rm_bn, 'bias') and rm_bn.bias is not None:
            gradient_tests_total += 1
            bias_close, bias_msg = compare_gradients(torch_bn.bias, rm_bn.bias, "bias")
            stats.add_result(f"{test_name}-bias梯度", bias_close, bias_msg)
            if bias_close:
                gradient_tests_passed += 1
        
        # 梯度总体通过率
        if gradient_tests_total > 0:
            gradient_pass_rate = gradient_tests_passed / gradient_tests_total * 100
            stats.add_result(f"{test_name}-梯度总体", gradient_pass_rate >= 90,
                            f"梯度通过率: {gradient_pass_rate:.1f}% ({gradient_tests_passed}/{gradient_tests_total})")
        
    except Exception as e:
        stats.add_result(f"{test_name}", False, f"测试异常: {str(e)}")


# ==================== 测试函数 ====================

def test_batch_norm_1d(stats=None):
    """测试BatchNorm1d模块"""
    # 如果没有传入stats实例，创建一个（用于pytest调用）
    if stats is None:
        stats = StatisticsCollector()
    
    stats.start_function("BatchNorm1d模块")
    
    try:
        # 设置随机种子确保可重现性
        np.random.seed(42)
        
        # 测试参数组合
        test_configs = [
            {"affine": True, "track_running_stats": True, "mode": "train", "desc": "训练模式+affine+跟踪统计"},
            {"affine": True, "track_running_stats": True, "mode": "eval", "desc": "评估模式+affine+跟踪统计"},
            {"affine": False, "track_running_stats": True, "mode": "train", "desc": "训练模式+无affine+跟踪统计"},
            {"affine": True, "track_running_stats": False, "mode": "train", "desc": "训练模式+affine+不跟踪统计"},
        ]
        
        # 测试2D输入 (N, C)
        print("\n测试2D输入 (N, C):")
        batch_size, num_features = 4, 3
        input_data_2d = np.random.randn(batch_size, num_features).astype(np.float32)
        
        for i, config in enumerate(test_configs):
            print(f"  配置{i+1}: {config['desc']}")
            
            # 创建Riemann和PyTorch的BatchNorm1d
            rm_bn = BatchNorm1d(num_features, affine=config['affine'], track_running_stats=config['track_running_stats'])
            torch_bn = tnn.BatchNorm1d(num_features, affine=config['affine'], track_running_stats=config['track_running_stats'])
            
            test_name = f"BatchNorm1d-2D-配置{i+1}"
            compare_batch_norm(rm_bn, torch_bn, input_data_2d, test_name, stats, config['mode'])
        
        # 测试3D输入 (N, C, L)
        print("\n测试3D输入 (N, C, L):")
        batch_size, num_features, seq_len = 2, 3, 4
        input_data_3d = np.random.randn(batch_size, num_features, seq_len).astype(np.float32)
        
        for i, config in enumerate(test_configs):
            print(f"  配置{i+1}: {config['desc']}")
            
            # 创建Riemann和PyTorch的BatchNorm1d
            rm_bn = BatchNorm1d(num_features, affine=config['affine'], track_running_stats=config['track_running_stats'])
            torch_bn = tnn.BatchNorm1d(num_features, affine=config['affine'], track_running_stats=config['track_running_stats'])
            
            test_name = f"BatchNorm1d-3D-配置{i+1}"
            compare_batch_norm(rm_bn, torch_bn, input_data_3d, test_name, stats, config['mode'])
            
    finally:
        stats.end_function()


def test_batch_norm_2d(stats=None):
    """测试BatchNorm2d模块"""
    # 如果没有传入stats实例，创建一个（用于pytest调用）
    if stats is None:
        stats = StatisticsCollector()
    
    stats.start_function("BatchNorm2d模块")
    
    try:
        # 设置随机种子确保可重现性
        np.random.seed(42)
        
        # 测试参数组合
        test_configs = [
            {"affine": True, "track_running_stats": True, "mode": "train", "desc": "训练模式+affine+跟踪统计"},
            {"affine": True, "track_running_stats": True, "mode": "eval", "desc": "评估模式+affine+跟踪统计"},
            {"affine": False, "track_running_stats": True, "mode": "train", "desc": "训练模式+无affine+跟踪统计"},
            {"affine": True, "track_running_stats": False, "mode": "train", "desc": "训练模式+affine+不跟踪统计"},
        ]
        
        # 测试4D输入 (N, C, H, W)
        print("\n测试4D输入 (N, C, H, W):")
        batch_size, num_features, height, width = 2, 3, 4, 4
        input_data_4d = np.random.randn(batch_size, num_features, height, width).astype(np.float32)
        
        for i, config in enumerate(test_configs):
            print(f"  配置{i+1}: {config['desc']}")
            
            # 创建Riemann和PyTorch的BatchNorm2d
            rm_bn = BatchNorm2d(num_features, affine=config['affine'], track_running_stats=config['track_running_stats'])
            torch_bn = tnn.BatchNorm2d(num_features, affine=config['affine'], track_running_stats=config['track_running_stats'])
            
            test_name = f"BatchNorm2d-4D-配置{i+1}"
            compare_batch_norm(rm_bn, torch_bn, input_data_4d, test_name, stats, config['mode'])
            
    finally:
        stats.end_function()


def test_batch_norm_3d(stats=None):
    """测试BatchNorm3d模块"""
    # 如果没有传入stats实例，创建一个（用于pytest调用）
    if stats is None:
        stats = StatisticsCollector()
    
    stats.start_function("BatchNorm3d模块")
    
    try:
        # 设置随机种子确保可重现性
        np.random.seed(42)
        
        # 测试参数组合
        test_configs = [
            {"affine": True, "track_running_stats": True, "mode": "train", "desc": "训练模式+affine+跟踪统计"},
            {"affine": True, "track_running_stats": True, "mode": "eval", "desc": "评估模式+affine+跟踪统计"},
            {"affine": False, "track_running_stats": True, "mode": "train", "desc": "训练模式+无affine+跟踪统计"},
            {"affine": True, "track_running_stats": False, "mode": "train", "desc": "训练模式+affine+不跟踪统计"},
        ]
        
        # 测试5D输入 (N, C, D, H, W)
        print("\n测试5D输入 (N, C, D, H, W):")
        batch_size, num_features, depth, height, width = 2, 3, 4, 4, 4
        input_data_5d = np.random.randn(batch_size, num_features, depth, height, width).astype(np.float32)
        
        for i, config in enumerate(test_configs):
            print(f"  配置{i+1}: {config['desc']}")
            
            # 创建Riemann和PyTorch的BatchNorm3d
            rm_bn = BatchNorm3d(num_features, affine=config['affine'], track_running_stats=config['track_running_stats'])
            torch_bn = tnn.BatchNorm3d(num_features, affine=config['affine'], track_running_stats=config['track_running_stats'])
            
            test_name = f"BatchNorm3d-5D-配置{i+1}"
            compare_batch_norm(rm_bn, torch_bn, input_data_5d, test_name, stats, config['mode'])
            
    finally:
        stats.end_function()


def main():
    """主测试函数"""
    print(f"{Colors.HEADER}Riemann nn.BatchNorm 批量归一化模块测试{Colors.ENDC}")
    print(f"{Colors.OKCYAN}测试BatchNorm1d、BatchNorm2d、BatchNorm3d模块的前向传播和反向传播{Colors.ENDC}")
    
    # 创建测试统计对象
    stats = StatisticsCollector()
    
    # 运行所有测试
    test_batch_norm_1d(stats)
    test_batch_norm_2d(stats)
    test_batch_norm_3d(stats)
    
    # 打印测试汇总
    stats.print_summary()

if __name__ == "__main__":
    clear_screen()
    main()