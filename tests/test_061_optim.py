#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试优化后的Optimizer类功能

测试内容:
1. 基础SGD优化器功能
2. Dampening和Nesterov支持
3. Closure参数支持
4. 状态管理和序列化
5. 错误处理和边界检查
6. 类型提示验证
"""

import sys
import os
import numpy as np
from pathlib import Path
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# 导入riemann模块
try:
    from riemann import tensor, Parameter
    from riemann.optim import SGD, Adam
    from riemann.optim.lr_scheduler import (
        StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
    )
    from riemann.nn import Module
    import unittest
    # 从rm.cuda获取cupy引用和CUDA可用性
    try:
        import riemann as rm
        CUDA_AVAILABLE = rm.cuda.CUPY_AVAILABLE
        cp = rm.cuda.cp
    except Exception:
        CUDA_AVAILABLE = False
except ImportError as e:
    print(f"无法导入riemann模块: {e}")
    print("请确保项目路径设置正确")
    sys.exit(1)

# 尝试导入PyTorch进行比较
try:
    import torch
    import torch.optim as torch_optim
    TORCH_AVAILABLE = True

    # 在模块级别进行PyTorch预热，避免在测试计时中包含初始化开销
    print("预热PyTorch系统...")
    warmup_start = time.time()
    
    # 执行简单的PyTorch操作以触发初始化
    warmup_input = torch.tensor([[0.0]], requires_grad=True)
    warmup_output = warmup_input.sum()
    warmup_output.backward()
    
    # 清理资源
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"PyTorch预热完成，耗时: {time.time() - warmup_start:.4f}秒")
    
except ImportError:
    print("警告: 无法导入PyTorch，将只测试riemann的优化器")
    TORCH_AVAILABLE = False

# 定义颜色类用于美化输出
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# 清屏函数
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# 测试统计类
class StatisticsCollector:
    def __init__(self):
        self.total_cases = 0
        self.passed_cases = 0
        self.total_time = 0.0
        self.function_stats = {}
        self.current_function = None
        self.current_function_start_time = 0
        self.current_test_details = []  # 存储当前测试的详细信息
    
    def start_function(self, function_name):
        self.current_function = function_name
        self.current_function_start_time = time.time()
        self.current_test_details = []  # 重置详细信息列表
        
        if function_name not in self.function_stats:
            self.function_stats[function_name] = {"total": 0, "passed": 0, "time": 0.0}
    
    def add_result(self, case_name, passed, details=None):
        self.total_cases += 1
        if passed:
            self.passed_cases += 1
        
        if self.current_function:
            self.function_stats[self.current_function]["total"] += 1
            if passed:
                self.function_stats[self.current_function]["passed"] += 1
                
            # 记录测试详情
            status = "通过" if passed else "失败"
            status_color = Colors.OKGREEN if passed else Colors.FAIL
            self.current_test_details.append({
                "name": case_name,
                "status": status,
                "color": status_color,
                "details": details
            })
    
    def end_function(self):
        if self.current_function:
            elapsed = time.time() - self.current_function_start_time
            self.function_stats[self.current_function]["time"] += elapsed
            self.total_time += elapsed
    
    def _get_display_width(self, text):
        """计算字符串的显示宽度，中文字符算2个宽度，英文字符算1个宽度"""
        width = 0
        for char in text:
            # 判断是否为中文字符（CJK统一表意文字范围）
            if '\u4e00' <= char <= '\u9fff':
                width += 2
            else:
                width += 1
        return width
    
    def print_summary(self):
        # 定义各列的标题
        headers = ['用例名', '通过/总数', '通过率', '耗时(秒)']
        
        # 计算各列标题的显示宽度
        header_widths = [self._get_display_width(h) for h in headers]
        
        # 计算数据行中各列的最大显示宽度
        max_func_name_width = header_widths[0]
        for func_name in self.function_stats.keys():
            max_func_name_width = max(max_func_name_width, self._get_display_width(func_name))
        
        # 为各列设置最终宽度，标题宽度和内容宽度的最大值，并留出适当间距
        col_widths = [
            max(max_func_name_width, header_widths[0]) + 2,  # 用例名列
            header_widths[1] + 4,  # 通过/总数列
            header_widths[2] + 4,  # 通过率列
            header_widths[3] + 4   # 耗时列
        ]
        
        total_width = sum(col_widths)
        
        print("\n" + "="*total_width)
        print(f"{Colors.BOLD}测试统计摘要{Colors.ENDC}")
        print("="*total_width)
        print(f"总测试用例数: {self.total_cases}")
        print(f"通过测试用例数: {Colors.OKGREEN if self.passed_cases == self.total_cases else Colors.FAIL}{self.passed_cases}{Colors.ENDC}")
        print(f"测试通过率: {Colors.OKGREEN if self.passed_cases == self.total_cases else Colors.FAIL}{self.passed_cases/self.total_cases*100:.2f}%{Colors.ENDC}")
        print(f"总耗时: {self.total_time:.4f} 秒")
        print("\n各用例测试详情:")
        print("-"*total_width)
        
        # 打印表头 - 精确计算每个标题的填充
        header_line = ""
        for i, header in enumerate(headers):
            header_width = self._get_display_width(header)
            padding = col_widths[i] - header_width
            header_line += header + " " * padding
        print(header_line)
        print("-"*total_width)
        
        # 打印数据行 - 精确计算每个值的填充
        for func_name, stats in self.function_stats.items():
            pass_rate = stats["passed"]/stats["total"]*100 if stats["total"] > 0 else 0
            status_color = Colors.OKGREEN if pass_rate == 100 else Colors.FAIL
            
            # 计算每个字段的显示宽度并添加适当的填充
            func_name_display = func_name
            func_name_width = self._get_display_width(func_name_display)
            func_name_padding = col_widths[0] - func_name_width
            
            pass_total_display = f"{stats['passed']}/{stats['total']}"
            pass_total_width = self._get_display_width(pass_total_display)
            pass_total_padding = col_widths[1] - pass_total_width
            
            # 通过率字段包含颜色代码，但显示宽度只计算实际文本
            pass_rate_display = f"{pass_rate:.2f}"
            pass_rate_width = self._get_display_width(pass_rate_display)
            pass_rate_padding = col_widths[2] - pass_rate_width
            
            time_display = f"{stats['time']:.4f}"
            time_width = self._get_display_width(time_display)
            time_padding = col_widths[3] - time_width
            
            # 构建完整的行
            print(
                f"{func_name_display}{' ' * func_name_padding}" +
                f"{pass_total_display}{' ' * pass_total_padding}" +
                f"{status_color}{pass_rate_display}{' ' * pass_rate_padding}{Colors.ENDC}" +
                f"{time_display}{' ' * time_padding}"
            )
        
        print("="*total_width)

# 全局统计实例
stats = StatisticsCollector()
# 判断是否作为独立脚本运行
IS_RUNNING_AS_SCRIPT = False

# 比较值的函数
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
        # 增加容差参数
        np.testing.assert_allclose(rm_data, torch_data, rtol=rtol, atol=atol)
        return True
    except AssertionError:
        return False

class SimpleModel(Module):
    """简单的线性模型用于测试"""
    
    def __init__(self):
        super().__init__()
        self.weight = Parameter(tensor(np.random.randn(10, 5).astype(np.float32) * 0.1))
        self.bias = Parameter(tensor(np.zeros(5).astype(np.float32)))
    
    def forward(self, x):
        # 使用矩阵乘法，需要建立计算图
        return x @ self.weight + self.bias

class TestOptimizedOptimizer(unittest.TestCase):
    """测试优化后的Optimizer类"""
    
    def setUp(self):
        """设置测试环境"""
        np.random.seed(42)
        self.model = SimpleModel()
        self.x = tensor(np.random.randn(3, 10).astype(np.float32))
        self.y = tensor(np.random.randn(3, 5).astype(np.float32))
        
        # 如果是独立脚本运行，则开始记录函数统计
        if IS_RUNNING_AS_SCRIPT:
            self.current_test_name = self._testMethodName
            stats.start_function(self.current_test_name)
            print(f"\n{Colors.HEADER}开始测试: {self.current_test_name}{Colors.ENDC}")
                
    def tearDown(self):
        # 如果是独立脚本运行，则结束函数统计
        if IS_RUNNING_AS_SCRIPT:
            stats.end_function()
            print(f"{Colors.OKBLUE}测试完成: {self.current_test_name}{Colors.ENDC}")
    
    def test_sgd_optimizer(self):
        """测试SGD优化器的所有功能"""
        # 基础功能测试用例
        basic_cases = [
            {"name": "基础SGD功能", "lr": 0.01, "momentum": 0.0, "weight_decay": 0.0, "dampening": 0.0, "nesterov": False},
            {"name": "不同学习率", "lr": 0.1, "momentum": 0.0, "weight_decay": 0.0, "dampening": 0.0, "nesterov": False},
            {"name": "带权重衰减", "lr": 0.01, "momentum": 0.0, "weight_decay": 0.0001, "dampening": 0.0, "nesterov": False},
        ]
        
        # 动量和抑制功能测试用例
        momentum_cases = [
            {"name": "基础动量", "momentum": 0.9, "dampening": 0.0},
            {"name": "动量和抑制", "momentum": 0.9, "dampening": 0.1},
            {"name": "不同动量值", "momentum": 0.5, "dampening": 0.0},
        ]
        
        # Nesterov动量测试用例
        nesterov_cases = [
            {"name": "Nesterov动量", "momentum": 0.9, "nesterov": True},
            {"name": "无Nesterov动量", "momentum": 0.9, "nesterov": False},
        ]
        
        # Closure参数支持测试用例
        closure_cases = [
            {"name": "基础closure支持", "lr": 0.01},
            {"name": "不同学习率的closure", "lr": 0.1},
        ]
        
        # 状态字典测试用例
        state_dict_cases = [
            {"name": "基础状态字典功能", "lr": 0.01, "momentum": 0.9},
            {"name": "不同动量值的状态字典", "lr": 0.01, "momentum": 0.5},
        ]
        
        # 梯度清零功能测试用例
        zero_grad_cases = [
            {"name": "基础梯度清零", "set_to_none": False},
            {"name": "set_to_none=True", "set_to_none": True},
        ]
        
        # 错误处理测试用例
        error_handling_cases = [
            {"name": "负学习率", "lr": -0.01, "should_raise": True},
            {"name": "负动量", "lr": 0.01, "momentum": -0.1, "should_raise": True},
            {"name": "负权重衰减", "lr": 0.01, "weight_decay": -0.0001, "should_raise": True},
            {"name": "负抑制", "lr": 0.01, "momentum": 0.9, "dampening": -0.1, "should_raise": True},
            {"name": "无动量的Nesterov", "lr": 0.01, "momentum": 0.0, "nesterov": True, "should_raise": True},
        ]
        
        # 运行基础功能测试
        print(f"\n{Colors.BOLD}基础功能测试{Colors.ENDC}")
        for case in basic_cases:
            case_name = f"SGD - {case['name']}"
            start_time = time.time()
            try:
                # 创建SGD优化器
                optimizer = SGD(
                    self.model.parameters(), 
                    lr=case['lr'],
                    momentum=case['momentum'],
                    weight_decay=case['weight_decay'],
                    dampening=case['dampening'],
                    nesterov=case['nesterov']
                )
                
                # 检查参数组设置
                self.assertEqual(len(optimizer.param_groups), 1)
                group = optimizer.param_groups[0]
                self.assertEqual(group['lr'], case['lr'])
                self.assertEqual(group['momentum'], case['momentum'])
                self.assertEqual(group['weight_decay'], case['weight_decay'])
                self.assertEqual(group['dampening'], case['dampening'])
                self.assertEqual(group['nesterov'], case['nesterov'])
                
                # 检查状态初始化
                self.assertIsInstance(optimizer.state, dict)
                
                passed = True
                
            except Exception as e:
                passed = False
                print(f"  错误: {str(e)}")
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"SGD测试失败: {case_name}")
        
        # 运行动量和抑制功能测试
        print(f"\n{Colors.BOLD}动量和抑制功能测试{Colors.ENDC}")
        for case in momentum_cases:
            case_name = f"SGD - {case['name']}"
            start_time = time.time()
            try:
                # 测试带动量的SGD
                optimizer = SGD(
                    self.model.parameters(), 
                    lr=0.01, 
                    momentum=case['momentum'], 
                    dampening=case['dampening']
                )
                
                group = optimizer.param_groups[0]
                self.assertEqual(group['momentum'], case['momentum'])
                self.assertEqual(group['dampening'], case['dampening'])
                
                # 执行一步优化
                output = self.model(self.x)
                diff = output - self.y
                loss = (diff * diff).mean()
                loss.backward()
                
                optimizer.step()
                
                # 检查动量状态是否正确初始化
                for param in self.model.parameters():
                    param_id = id(param)
                    if param_id in optimizer.state:
                        self.assertIn('velocity', optimizer.state[param_id])
                        self.assertEqual(optimizer.state[param_id]['velocity'].shape, param.data.shape)
                
                passed = True
                
            except Exception as e:
                passed = False
                print(f"  错误: {str(e)}")
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"SGD动量测试失败: {case_name}")
        
        # 运行Nesterov动量测试
        print(f"\n{Colors.BOLD}Nesterov动量测试{Colors.ENDC}")
        for case in nesterov_cases:
            case_name = f"SGD - {case['name']}"
            start_time = time.time()
            try:
                # 测试Nesterov动量
                optimizer = SGD(
                    self.model.parameters(), 
                    lr=0.01, 
                    momentum=case['momentum'], 
                    nesterov=case['nesterov']
                )
                
                group = optimizer.param_groups[0]
                self.assertEqual(group['momentum'], case['momentum'])
                self.assertEqual(group['nesterov'], case['nesterov'])
                
                # 执行一步优化
                output = self.model(self.x)
                diff = output - self.y
                loss = (diff * diff).mean()
                loss.backward()
                
                initial_weight = self.model.weight.data.copy()
                optimizer.step()
                
                # 检查参数是否更新
                self.assertFalse(np.allclose(initial_weight, self.model.weight.data))
                
                passed = True
                
            except Exception as e:
                passed = False
                print(f"  错误: {str(e)}")
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"SGD Nesterov测试失败: {case_name}")
        
        # 运行Closure参数支持测试
        print(f"\n{Colors.BOLD}Closure参数支持测试{Colors.ENDC}")
        for case in closure_cases:
            case_name = f"SGD - {case['name']}"
            start_time = time.time()
            try:
                optimizer = SGD(self.model.parameters(), lr=case['lr'])
                
                def closure():
                    output = self.model(self.x)
                    diff = output - self.y
                    loss = (diff * diff).mean()
                    loss.backward()
                    return loss.data.item()
                
                # 测试带closure的step
                loss_value = optimizer.step(closure)
                self.assertIsNotNone(loss_value)
                self.assertIsInstance(loss_value, float)
                
                passed = True
                
            except Exception as e:
                passed = False
                print(f"  错误: {str(e)}")
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"SGD closure测试失败: {case_name}")
        
        # 运行状态字典测试
        print(f"\n{Colors.BOLD}状态字典测试{Colors.ENDC}")
        for case in state_dict_cases:
            case_name = f"SGD - {case['name']}"
            start_time = time.time()
            try:
                optimizer = SGD(
                    self.model.parameters(), 
                    lr=case['lr'], 
                    momentum=case['momentum']
                )
                
                # 执行几步优化以产生状态
                for _ in range(3):
                    output = self.model(self.x)
                    diff = output - self.y
                    loss = (diff * diff).mean()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                
                # 保存状态
                state_dict = optimizer.state_dict()
                
                # 检查状态字典结构
                self.assertIn('state', state_dict)
                self.assertIn('param_groups', state_dict)
                
                # 创建新的优化器并加载状态
                new_optimizer = SGD(self.model.parameters(), lr=0.001)  # 不同的学习率
                new_optimizer.load_state_dict(state_dict)
                
                # 检查加载后的状态
                self.assertEqual(len(new_optimizer.param_groups), len(optimizer.param_groups))
                self.assertEqual(new_optimizer.param_groups[0]['lr'], case['lr'])  # 应该被覆盖
                
                passed = True
                
            except Exception as e:
                passed = False
                print(f"  错误: {str(e)}")
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"SGD状态字典测试失败: {case_name}")
        
        # 运行梯度清零功能测试
        print(f"\n{Colors.BOLD}梯度清零功能测试{Colors.ENDC}")
        for case in zero_grad_cases:
            case_name = f"SGD - {case['name']}"
            start_time = time.time()
            try:
                optimizer = SGD(self.model.parameters(), lr=0.01)
                
                # 计算梯度
                output = self.model(self.x)
                diff = output - self.y
                loss = (diff * diff).mean()
                loss.backward()
                
                # 检查梯度存在
                for param in self.model.parameters():
                    self.assertIsNotNone(param.grad)
                
                # 清零梯度
                optimizer.zero_grad(set_to_none=case['set_to_none'])
                
                # 检查梯度被清零
                for param in self.model.parameters():
                    if case['set_to_none']:
                        self.assertIsNone(param.grad)
                    else:
                        if param.grad is not None:
                            self.assertTrue(np.allclose(param.grad.data, 0))
                
                passed = True
                
            except Exception as e:
                passed = False
                print(f"  错误: {str(e)}")
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"SGD梯度清零测试失败: {case_name}")
        
        # 运行错误处理测试
        print(f"\n{Colors.BOLD}错误处理测试{Colors.ENDC}")
        for case in error_handling_cases:
            case_name = f"SGD - {case['name']}"
            start_time = time.time()
            try:
                # 提取参数
                lr = case.get('lr', 0.01)
                momentum = case.get('momentum', 0.0)
                weight_decay = case.get('weight_decay', 0.0)
                dampening = case.get('dampening', 0.0)
                nesterov = case.get('nesterov', False)
                
                if case['should_raise']:
                    # 应该抛出异常
                    with self.assertRaises(ValueError):
                        SGD(
                            self.model.parameters(), 
                            lr=lr,
                            momentum=momentum,
                            weight_decay=weight_decay,
                            dampening=dampening,
                            nesterov=nesterov
                        )
                else:
                    # 不应该抛出异常
                    optimizer = SGD(
                        self.model.parameters(), 
                        lr=lr,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        dampening=dampening,
                        nesterov=nesterov
                    )
                    self.assertIsInstance(optimizer, SGD)
                
                passed = True
                
            except Exception as e:
                passed = False
                print(f"  错误: {str(e)}")
            
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"SGD错误处理测试失败: {case_name}")

    def test_lr_schedulers(self):
        """测试学习率调度器与 PyTorch 的行为一致性"""
        # 学习率调度器测试用例
        scheduler_cases = [
            {
                "name": "StepLR",
                "params": {
                    "step_size": 3,
                    "gamma": 0.1
                },
                "initial_lr": 0.1,
                "num_epochs": 10
            },
            {
                "name": "MultiStepLR",
                "params": {
                    "milestones": [3, 6, 9],
                    "gamma": 0.1
                },
                "initial_lr": 0.1,
                "num_epochs": 10
            },
            {
                "name": "ExponentialLR",
                "params": {
                    "gamma": 0.9
                },
                "initial_lr": 0.1,
                "num_epochs": 10
            },
            {
                "name": "CosineAnnealingLR",
                "params": {
                    "T_max": 5,
                    "eta_min": 0.001
                },
                "initial_lr": 0.1,
                "num_epochs": 10
            },
            {
                "name": "ReduceLROnPlateau",
                "params": {
                    "mode": 'min',
                    "factor": 0.1,
                    "patience": 2,
                    "threshold": 1e-4
                },
                "initial_lr": 0.1,
                "num_epochs": 10,
                "losses": [1.0, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
            }
        ]

        # 运行学习率调度器测试
        for case in scheduler_cases:
            scheduler_name = case["name"]
            print(f"\n{Colors.BOLD}{scheduler_name}调度器测试{Colors.ENDC}")
            
            initial_lr = case["initial_lr"]
            num_epochs = case["num_epochs"]
            scheduler_params = case["params"]
            
            # 创建 Riemann 模型和优化器
            class SimpleModel(Module):
                def __init__(self):
                    super().__init__()
                    self.linear = Parameter(tensor(np.random.randn(10, 1).astype(np.float32) * 0.1))
                    self.bias = Parameter(tensor(np.zeros(1).astype(np.float32)))
                
                def forward(self, x):
                    return x @ self.linear + self.bias
            
            rm_model = SimpleModel()
            rm_optimizer = SGD(rm_model.parameters(), lr=initial_lr)
            
            # 创建 Riemann 调度器
            if scheduler_name == "StepLR":
                rm_scheduler = StepLR(rm_optimizer, **scheduler_params)
            elif scheduler_name == "MultiStepLR":
                rm_scheduler = MultiStepLR(rm_optimizer, **scheduler_params)
            elif scheduler_name == "ExponentialLR":
                rm_scheduler = ExponentialLR(rm_optimizer, **scheduler_params)
            elif scheduler_name == "CosineAnnealingLR":
                rm_scheduler = CosineAnnealingLR(rm_optimizer, **scheduler_params)
            elif scheduler_name == "ReduceLROnPlateau":
                rm_scheduler = ReduceLROnPlateau(rm_optimizer, **scheduler_params)
            
            # 记录 Riemann 学习率
            rm_lrs = []
            
            # 记录初始学习率
            rm_lrs.append(rm_optimizer.param_groups[0]['lr'])
            
            # 测试 ReduceLROnPlateau
            if scheduler_name == "ReduceLROnPlateau":
                losses = case.get("losses", [])
                
                for epoch in range(num_epochs):
                    rm_optimizer.step()
                    rm_scheduler.step(losses[epoch])
                    if epoch < num_epochs - 1:
                        rm_lrs.append(rm_optimizer.param_groups[0]['lr'])
            else:
                # 其他调度器
                for epoch in range(num_epochs):
                    rm_optimizer.step()
                    rm_scheduler.step()
                    if epoch < num_epochs - 1:
                        rm_lrs.append(rm_optimizer.param_groups[0]['lr'])
            
            # 如果 PyTorch 可用，创建 PyTorch 模型和优化器并比较
            if TORCH_AVAILABLE:
                import torch.nn as torch_nn
                import torch.optim.lr_scheduler as torch_lr_scheduler
                
                class TorchSimpleModel(torch_nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.linear = torch_nn.Linear(10, 1)
                
                torch_model = TorchSimpleModel()
                torch_optimizer = torch_optim.SGD(torch_model.parameters(), lr=initial_lr)
                
                # 创建 PyTorch 调度器
                if scheduler_name == "StepLR":
                    torch_scheduler = torch_lr_scheduler.StepLR(torch_optimizer, **scheduler_params)
                elif scheduler_name == "MultiStepLR":
                    torch_scheduler = torch_lr_scheduler.MultiStepLR(torch_optimizer, **scheduler_params)
                elif scheduler_name == "ExponentialLR":
                    torch_scheduler = torch_lr_scheduler.ExponentialLR(torch_optimizer, **scheduler_params)
                elif scheduler_name == "CosineAnnealingLR":
                    torch_scheduler = torch_lr_scheduler.CosineAnnealingLR(torch_optimizer, **scheduler_params)
                elif scheduler_name == "ReduceLROnPlateau":
                    # 移除 verbose 参数，因为 PyTorch 的某些版本可能不支持
                    torch_params = scheduler_params.copy()
                    torch_scheduler = torch_lr_scheduler.ReduceLROnPlateau(torch_optimizer, **torch_params)
                
                # 记录 PyTorch 学习率
                torch_lrs = []
                torch_lrs.append(torch_optimizer.param_groups[0]['lr'])
                
                # 测试 ReduceLROnPlateau
                if scheduler_name == "ReduceLROnPlateau":
                    losses = case.get("losses", [])
                    
                    for epoch in range(num_epochs):
                        torch_optimizer.step()
                        torch_scheduler.step(losses[epoch])
                        if epoch < num_epochs - 1:
                            torch_lrs.append(torch_optimizer.param_groups[0]['lr'])
                else:
                    # 其他调度器
                    for epoch in range(num_epochs):
                        torch_optimizer.step()
                        torch_scheduler.step()
                        if epoch < num_epochs - 1:
                            torch_lrs.append(torch_optimizer.param_groups[0]['lr']) 
                
                # 比较学习率
                print(f"  Riemann 学习率: {[round(lr, 6) for lr in rm_lrs]}")
                print(f"  PyTorch 学习率: {[round(lr, 6) for lr in torch_lrs]}")
                
                # 检查是否一致
                all_match = all(np.isclose(rm_lr, torch_lr) for rm_lr, torch_lr in zip(rm_lrs, torch_lrs))
                status = "通过" if all_match else "失败"
                print(f"  测试结果: {Colors.OKGREEN if all_match else Colors.FAIL}{status}{Colors.ENDC}")
                
                # 记录测试结果
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(f"{scheduler_name} 调度器", all_match)
                
                # 断言确保测试通过
                self.assertTrue(all_match, f"{scheduler_name} 调度器测试失败")
            else:
                print(f"  Riemann 学习率: {[round(lr, 6) for lr in rm_lrs]}")
                print("  PyTorch 不可用，跳过比较")
                
                # 记录测试结果
                if IS_RUNNING_AS_SCRIPT:
                    stats.add_result(f"{scheduler_name} 调度器", True)
                
                # 即使没有 PyTorch，也要确保学习率列表不为空
                self.assertTrue(len(rm_lrs) > 0, f"{scheduler_name} 调度器测试失败")

if __name__ == '__main__':
    # 设置为独立脚本运行模式
    IS_RUNNING_AS_SCRIPT = True
    
    # 清屏
    clear_screen()
    
    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行优化器测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}PyTorch 可用: {TORCH_AVAILABLE}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}CUDA 可用: {CUDA_AVAILABLE}{Colors.ENDC}")
    print()
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOptimizedOptimizer)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出
    result = runner.run(suite)
    
    # 打印统计信息
    stats.print_summary()
    
    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)
