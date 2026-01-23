#!/usr/bin/env python3
"""
Riemann nn.Module 全功能测试套件

本测试文件全面验证 Riemann nn.Module 类的所有函数和方法，
确保与 PyTorch nn.Module 的行为一致。

测试覆盖的功能模块：
1. 参数管理 - parameters(), named_parameters(), get_parameter(), set_parameter(), delete_parameter(), has_parameter()
2. 缓冲区管理 - buffers(), named_buffers(), get_buffer(), set_buffer(), delete_buffer(), has_buffer()
3. 子模块管理 - children(), modules(), named_modules(), get_submodule(), set_submodule(), delete_submodule(), has_submodule()
4. 状态管理 - state_dict(), load_state_dict()
5. 模式控制 - train(), eval(), training属性
6. 梯度管理 - zero_grad(), requires_grad_()
7. 类型转换 - type(), float(), double(), half()
8. 函数应用 - apply()
9. 字符串表示 - extra_repr(), __repr__()
10. 模块复制 - copy(), deepcopy(), __copy__(), __deepcopy__()
11. 属性管理 - __setattr__(), __getattr__(), __delattr__()
12. 前向传播 - forward(), __call__()
"""

import unittest
import numpy as np
import time
import sys
import os
import copy
from typing import Any, Dict, List

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# 导入riemann模块
try:
    import riemann as rm
    from riemann.nn import Parameter, Module, Linear, Sequential, ModuleList, ModuleDict, Dropout
    RIEMANN_AVAILABLE = True
except ImportError as e:
    print(f"无法导入riemann模块: {e}")
    RIEMANN_AVAILABLE = False
    sys.exit(1)

# 尝试导入PyTorch进行比较
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    
    # PyTorch预热
    print("预热PyTorch系统...")
    warmup_start = time.time()
    warmup_input = torch.tensor([[0.0]], requires_grad=True)
    warmup_output = warmup_input.sum()
    warmup_output.backward()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"PyTorch预热完成，耗时: {time.time() - warmup_start:.4f}秒")
    
except ImportError:
    print("警告: 无法导入PyTorch，将只测试riemann的功能")
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
        self.current_test_details = []
    
    def start_function(self, function_name):
        self.current_function = function_name
        self.current_function_start_time = time.time()
        self.current_test_details = []
        
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
        # 计算需要添加的空格数
        padding = width - display_width
        return text + " " * padding
    
    def print_summary(self):
        headers = ['用例名', '通过/总数', '通过率', '耗时(秒)']
        header_widths = [self._get_display_width(h) for h in headers]
        
        max_func_name_width = header_widths[0]
        for func_name in self.function_stats.keys():
            max_func_name_width = max(max_func_name_width, self._get_display_width(func_name))
        
        col_widths = [
            max(max_func_name_width, header_widths[0]) + 2,
            header_widths[1] + 4,
            header_widths[2] + 4,
            header_widths[3] + 4
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
        
        header_line = ""
        for i, header in enumerate(headers):
            header_line += self._ljust_display_width(header, col_widths[i])
        print(header_line)
        print("-"*total_width)
        
        for func_name, stats in self.function_stats.items():
            pass_rate = stats["passed"] / stats["total"] * 100
            pass_rate_color = Colors.OKGREEN if pass_rate == 100 else Colors.WARNING if pass_rate >= 80 else Colors.FAIL
            
            # 构建每一列的内容
            # 第一列：用例名（按显示宽度左对齐）
            col1 = self._ljust_display_width(func_name, col_widths[0])
            
            # 第二列：通过/总数（左对齐）
            col2 = f"{stats['passed']}/{stats['total']}".ljust(col_widths[1])
            
            # 第三列：通过率（左对齐，颜色代码不影响显示宽度）
            pass_rate_str = f"{pass_rate:.1f}%"
            pass_rate_display = pass_rate_str.ljust(col_widths[2])
            col3 = f"{pass_rate_color}{pass_rate_display}{Colors.ENDC}"
            
            # 第四列：耗时（左对齐）
            col4 = f"{stats['time']:.4f}".ljust(col_widths[3])
            
            # 组合所有列
            line = col1 + col2 + col3 + col4
            print(line)
        
        print("="*total_width)

# 全局统计收集器
stats = StatisticsCollector()

# 辅助函数
def tensor_allclose(tensor1, tensor2, rtol=1e-05, atol=1e-08):
    """比较两个张量是否近似相等"""
    if hasattr(tensor1, 'data'):
        tensor1 = tensor1.data
    if hasattr(tensor2, 'data'):
        tensor2 = tensor2.data
    return np.allclose(tensor1, tensor2, rtol=rtol, atol=atol)

def compare_with_pytorch(rm_result, torch_result, name="result"):
    """与PyTorch结果比较"""
    if not TORCH_AVAILABLE:
        return True
    
    if isinstance(rm_result, list) and isinstance(torch_result, list):
        if len(rm_result) != len(torch_result):
            print(f"列表长度不匹配: {len(rm_result)} vs {len(torch_result)}")
            return False
        for i, (rm_item, torch_item) in enumerate(zip(rm_result, torch_result)):
            if not tensor_allclose(rm_item, torch_item):
                print(f"列表元素{i}不匹配")
                return False
        return True
    else:
        return tensor_allclose(rm_result, torch_result)

# ==================== 测试用的模块类 ====================
class ModuleForTest(rm.nn.Module):
    """用于测试的简单模块"""
    def __init__(self):
        super().__init__()
        self.param1 = rm.nn.Parameter(rm.randn(10, 5))
        self.param2 = rm.nn.Parameter(rm.randn(5))
        self.buffer1 = rm.randn(5)  # 不使用Parameter，会自动注册为buffer
        self.submodule = SubModuleForTest()
    
    def forward(self, x):
        return self.submodule(x + self.buffer1)

class SubModuleForTest(rm.nn.Module):
    """用于测试的子模块"""
    def __init__(self):
        super().__init__()
        self.linear = Linear(5, 3)
    
    def forward(self, x):
        return self.linear(x)

class CustomModule(rm.nn.Module):
    """自定义模块用于测试"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = Linear(in_features, out_features)
        # 注册缓冲区
        self.register_buffer('running_mean', rm.zeros(out_features))
        self.register_buffer('running_var', rm.ones(out_features))
    
    def forward(self, x):
        return self.linear(x)

class NestedModule(rm.nn.Module):
    """嵌套模块用于测试"""
    def __init__(self):
        super().__init__()
        self.layer1 = CustomModule(10, 8)
        self.layer2 = CustomModule(8, 5)
        self.layer3 = SubModuleForTest()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)

# ==================== 参数管理测试 ====================
def test_parameters_management():
    """测试参数管理相关函数"""
    stats.start_function("参数管理")
    
    try:
        # 测试1: parameters() 基本功能
        print("测试 parameters() 基本功能...")
        module = CustomModule(10, 5)
        params = list(module.parameters())
        
        expected_count = 2  # weight and bias
        actual_count = len(params)
        passed = actual_count == expected_count
        stats.add_result("parameters()基本功能", passed, f"期望{expected_count}个参数，实际{actual_count}个")
        
        # 测试2: parameters() recurse=False
        print("测试 parameters() recurse=False...")
        nested = NestedModule()
        params_recurse = list(nested.parameters(recurse=True))
        params_no_recurse = list(nested.parameters(recurse=False))
        
        # NestedModule没有直接参数，只有子模块参数
        passed = len(params_no_recurse) == 0 and len(params_recurse) > 0
        stats.add_result("parameters()递归控制", passed, f"recurse=False: {len(params_no_recurse)}, recurse=True: {len(params_recurse)}")
        
        # 测试3: named_parameters()
        print("测试 named_parameters()...")
        named_params = dict(module.named_parameters())
        
        expected_keys = {'linear.weight', 'linear.bias'}
        actual_keys = set(named_params.keys())
        passed = expected_keys == actual_keys
        stats.add_result("named_parameters()键名", passed, f"期望{expected_keys}，实际{actual_keys}")
        
        # 测试4: get_parameter()
        print("测试 get_parameter()...")
        weight = module.get_parameter('linear.weight')
        passed = weight is not None and hasattr(weight, 'data')
        stats.add_result("get_parameter()获取", passed, f"成功获取linear.weight参数")
        
        # 测试5: get_parameter() 不存在的参数
        try:
            module.get_parameter('nonexistent')
            passed = False
        except AttributeError:
            passed = True
        stats.add_result("get_parameter()异常处理", passed, "正确抛出AttributeError")
        
        # 测试6: set_parameter()
        print("测试 set_parameter()...")
        new_param = Parameter(rm.randn(5, 10))
        module.set_parameter('linear.weight', new_param)
        retrieved = module.get_parameter('linear.weight')
        passed = retrieved is new_param
        stats.add_result("set_parameter()设置", passed, "成功设置新参数")
        
        # 测试7: has_parameter()
        print("测试 has_parameter()...")
        has_weight = module.has_parameter('linear.weight')
        has_nonexistent = module.has_parameter('nonexistent')
        passed = has_weight and not has_nonexistent
        stats.add_result("has_parameter()检查", passed, f"linear.weight: {has_weight}, nonexistent: {has_nonexistent}")
        
        # 测试8: delete_parameter()
        print("测试 delete_parameter()...")
        module.register_parameter('temp_param', Parameter(rm.randn(3, 3)))
        before_delete = module.has_parameter('temp_param')
        module.delete_parameter('temp_param')
        after_delete = module.has_parameter('temp_param')
        passed = before_delete and not after_delete
        stats.add_result("delete_parameter()删除", passed, f"删除前: {before_delete}, 删除后: {after_delete}")
        
    except Exception as e:
        print(f"参数管理测试出现异常: {e}")
        stats.add_result("参数管理异常", False, str(e))
    
    finally:
        stats.end_function()

# ==================== 缓冲区管理测试 ====================
def test_buffers_management():
    """测试缓冲区管理相关函数"""
    stats.start_function("缓冲区管理")
    
    try:
        # 测试1: buffers() 基本功能
        print("测试 buffers() 基本功能...")
        module = CustomModule(10, 5)
        buffers = list(module.buffers())
        
        expected_count = 2  # running_mean and running_var
        actual_count = len(buffers)
        passed = actual_count == expected_count
        stats.add_result("buffers()基本功能", passed, f"期望{expected_count}个缓冲区，实际{actual_count}个")
        
        # 测试2: named_buffers()
        print("测试 named_buffers()...")
        named_buffers = dict(module.named_buffers())
        
        expected_keys = {'running_mean', 'running_var'}
        actual_keys = set(named_buffers.keys())
        passed = expected_keys == actual_keys
        stats.add_result("named_buffers()键名", passed, f"期望{expected_keys}，实际{actual_keys}")
        
        # 测试3: get_buffer()
        print("测试 get_buffer()...")
        running_mean = module.get_buffer('running_mean')
        passed = running_mean is not None and hasattr(running_mean, 'data')
        stats.add_result("get_buffer()获取", passed, f"成功获取running_mean缓冲区")
        
        # 测试4: set_buffer()
        print("测试 set_buffer()...")
        new_buffer = rm.ones(5)
        module.set_buffer('running_mean', new_buffer)
        retrieved = module.get_buffer('running_mean')
        passed = tensor_allclose(retrieved, new_buffer)
        stats.add_result("set_buffer()设置", passed, "成功设置新缓冲区")
        
        # 测试5: has_buffer()
        print("测试 has_buffer()...")
        has_mean = module.has_buffer('running_mean')
        has_nonexistent = module.has_buffer('nonexistent')
        passed = has_mean and not has_nonexistent
        stats.add_result("has_buffer()检查", passed, f"running_mean: {has_mean}, nonexistent: {has_nonexistent}")
        
        # 测试6: delete_buffer()
        print("测试 delete_buffer()...")
        module.register_buffer('temp_buffer', rm.ones(3))
        before_delete = module.has_buffer('temp_buffer')
        module.delete_buffer('temp_buffer')
        after_delete = module.has_buffer('temp_buffer')
        passed = before_delete and not after_delete
        stats.add_result("delete_buffer()删除", passed, f"删除前: {before_delete}, 删除后: {after_delete}")
        
    except Exception as e:
        print(f"缓冲区管理测试出现异常: {e}")
        stats.add_result("缓冲区管理异常", False, str(e))
    
    finally:
        stats.end_function()

# ==================== 子模块管理测试 ====================
def test_submodules_management():
    """测试子模块管理相关函数"""
    stats.start_function("子模块管理")
    
    try:
        # 测试1: children()
        print("测试 children()...")
        nested = NestedModule()
        children = list(nested.children())
        
        expected_count = 3  # layer1, layer2, seq
        actual_count = len(children)
        passed = actual_count == expected_count
        stats.add_result("children()直接子模块", passed, f"期望{expected_count}个子模块，实际{actual_count}个")
        
        # 测试2: modules()
        print("测试 modules()...")
        modules = list(nested.modules())
        
        # 应该包含自身和所有子模块
        passed = len(modules) > 0 and any(m is nested for m in modules)
        stats.add_result("modules()所有模块", passed, f"总模块数: {len(modules)}, 包含自身: {any(m is nested for m in modules)}")
        
        # 测试3: named_modules()
        print("测试 named_modules()...")
        named_modules = dict(nested.named_modules())
        
        passed = '' in named_modules and named_modules[''] is nested
        stats.add_result("named_modules()命名", passed, f"包含空键名: {'' in named_modules}, 指向自身: {named_modules.get('') is nested}")
        
        # 测试4: get_submodule()
        print("测试 get_submodule()...")
        layer1 = nested.get_submodule('layer1')
        passed = layer1 is nested.layer1
        stats.add_result("get_submodule()获取", passed, "成功获取layer1子模块")
        
        # 测试5: set_submodule()
        print("测试 set_submodule()...")
        new_module = CustomModule(5, 3)
        nested.set_submodule('new_layer', new_module)
        retrieved = nested.get_submodule('new_layer')
        passed = retrieved is new_module
        stats.add_result("set_submodule()设置", passed, "成功设置新子模块")
        
        # 测试6: has_submodule()
        print("测试 has_submodule()...")
        has_layer1 = nested.has_submodule('layer1')
        has_nonexistent = nested.has_submodule('nonexistent')
        passed = has_layer1 and not has_nonexistent
        stats.add_result("has_submodule()检查", passed, f"layer1: {has_layer1}, nonexistent: {has_nonexistent}")
        
        # 测试7: delete_submodule()
        print("测试 delete_submodule()...")
        nested.set_submodule('temp_layer', CustomModule(2, 2))
        before_delete = nested.has_submodule('temp_layer')
        nested.delete_submodule('temp_layer')
        after_delete = nested.has_submodule('temp_layer')
        passed = before_delete and not after_delete
        stats.add_result("delete_submodule()删除", passed, f"删除前: {before_delete}, 删除后: {after_delete}")
        
    except Exception as e:
        print(f"子模块管理测试出现异常: {e}")
        stats.add_result("子模块管理异常", False, str(e))
    
    finally:
        stats.end_function()

# ==================== 状态管理测试 ====================
def test_state_management():
    """测试状态管理相关函数"""
    stats.start_function("状态管理")
    
    try:
        # 测试1: state_dict() 基本功能
        print("测试 state_dict() 基本功能...")
        module = CustomModule(10, 5)
        state_dict = module.state_dict()
        
        # 应该包含参数和缓冲区
        expected_keys = {'linear.weight', 'linear.bias', 'running_mean', 'running_var'}
        actual_keys = set(state_dict.keys())
        passed = expected_keys.issubset(actual_keys)
        stats.add_result("state_dict()基本功能", passed, f"期望包含{expected_keys}，实际包含{actual_keys}")
        
        # 测试2: load_state_dict() 基本功能
        print("测试 load_state_dict() 基本功能...")
        new_module = CustomModule(10, 5)
        new_module.load_state_dict(state_dict)
        
        # 比较权重
        weight_close = tensor_allclose(module.linear.weight, new_module.linear.weight)
        bias_close = tensor_allclose(module.linear.bias, new_module.linear.bias)
        passed = weight_close and bias_close
        stats.add_result("load_state_dict()加载", passed, f"weight匹配: {weight_close}, bias匹配: {bias_close}")
        
        # 测试3: state_dict() prefix参数
        print("测试 state_dict() prefix参数...")
        prefixed_dict = module.state_dict(prefix='test.')
        
        has_prefix = all(key.startswith('test.') for key in prefixed_dict.keys())
        passed = has_prefix
        stats.add_result("state_dict()前缀", passed, f"所有键都有前缀: {has_prefix}")
        
    except Exception as e:
        print(f"状态管理测试出现异常: {e}")
        stats.add_result("状态管理异常", False, str(e))
    
    finally:
        stats.end_function()

# ==================== 模式控制测试 ====================
def test_mode_control():
    """测试训练/评估模式控制"""
    stats.start_function("模式控制")
    
    try:
        # 测试1: train() 方法
        print("测试 train() 方法...")
        module = CustomModule(10, 5)
        
        module.train()  # 默认应该是训练模式
        training_after_train = module.training
        
        module.eval()  # 切换到评估模式
        training_after_eval = module.training
        
        module.train(False)  # 显式设置为评估模式
        training_after_train_false = module.training
        
        passed = (training_after_train and not training_after_eval and not training_after_train_false)
        stats.add_result("train()/eval()模式切换", passed, 
                        f"train(): {training_after_train}, eval(): {training_after_eval}, train(False): {training_after_train_false}")
        
        # 测试2: 子模块模式继承
        print("测试子模块模式继承...")
        nested = NestedModule()
        
        nested.eval()
        child_training_states = [child.training for child in nested.children()]
        all_eval = not any(child_training_states)
        
        nested.train()
        child_training_states = [child.training for child in nested.children()]
        all_train = all(child_training_states)
        
        passed = all_eval and all_train
        stats.add_result("子模块模式继承", passed, f"eval()后子模块全评估: {all_eval}, train()后子模块全训练: {all_train}")
        
    except Exception as e:
        print(f"模式控制测试出现异常: {e}")
        stats.add_result("模式控制异常", False, str(e))
    
    finally:
        stats.end_function()

# ==================== 梯度管理测试 ====================
def test_gradient_management():
    """测试梯度管理相关函数"""
    stats.start_function("梯度管理")
    
    try:
        # 测试1: zero_grad() 基本功能
        print("测试 zero_grad() 基本功能...")
        module = CustomModule(10, 5)
        
        # 模拟有梯度的情况
        if hasattr(module.linear.weight, 'grad') and module.linear.weight.grad is not None:
            initial_grad = module.linear.weight.grad.copy()
        else:
            # 手动设置梯度
            module.linear.weight.grad = rm.ones_like(module.linear.weight)
            initial_grad = module.linear.weight.grad.copy()
        
        module.zero_grad()
        grad_after_zero = module.linear.weight.grad
        
        passed = grad_after_zero is None or (hasattr(grad_after_zero, 'abs') and grad_after_zero.abs().max().item() == 0.0)
        stats.add_result("zero_grad()清零梯度", passed, "梯度已清零")
        
        # 测试2: zero_grad(set_to_none=True)
        print("测试 zero_grad(set_to_none=True)...")
        # 重新设置梯度
        module.linear.weight.grad = rm.ones_like(module.linear.weight)
        module.zero_grad(set_to_none=True)
        grad_after_none = module.linear.weight.grad
        
        passed = grad_after_none is None
        stats.add_result("zero_grad()设置为None", passed, f"梯度为None: {grad_after_none is None}")
        
        # 测试3: requires_grad_()
        print("测试 requires_grad_()...")
        original_requires_grad = module.linear.weight.requires_grad
        
        module.requires_grad_(False)
        requires_grad_false = module.linear.weight.requires_grad
        
        module.requires_grad_(True)
        requires_grad_true = module.linear.weight.requires_grad
        
        passed = (original_requires_grad and not requires_grad_false and requires_grad_true)
        stats.add_result("requires_grad_()切换", passed, 
                        f"原始: {original_requires_grad}, False: {requires_grad_false}, True: {requires_grad_true}")
        
    except Exception as e:
        print(f"梯度管理测试出现异常: {e}")
        stats.add_result("梯度管理异常", False, str(e))
    
    finally:
        stats.end_function()

# ==================== 类型转换测试 ====================
def test_type_conversion():
    """测试数据类型转换函数"""
    stats.start_function("类型转换")
    
    try:
        # 测试1: type() 方法
        print("测试 type() 方法...")
        module = CustomModule(10, 5)
        
        # 转换为float32
        module.type(np.float32)
        weight_dtype = module.linear.weight.data.dtype
        passed = weight_dtype == np.float32
        stats.add_result("type()转换为float32", passed, f"权重类型: {weight_dtype}")
        
        # 测试2: float() 方法
        print("测试 float() 方法...")
        module.float()
        weight_dtype_after_float = module.linear.weight.data.dtype
        passed = weight_dtype_after_float == np.float32
        stats.add_result("float()转换", passed, f"权重类型: {weight_dtype_after_float}")
        
        # 测试3: double() 方法
        print("测试 double() 方法...")
        module.double()
        weight_dtype_after_double = module.linear.weight.data.dtype
        passed = weight_dtype_after_double == np.float64
        stats.add_result("double()转换", passed, f"权重类型: {weight_dtype_after_double}")
        
        # 测试4: half() 方法 (如果支持)
        print("测试 half() 方法...")
        try:
            module.half()
            weight_dtype_after_half = module.linear.weight.data.dtype
            passed = weight_dtype_after_half == np.float16
            stats.add_result("half()转换", passed, f"权重类型: {weight_dtype_after_half}")
        except Exception as e:
            stats.add_result("half()转换", False, f"不支持float16: {e}")
        
    except Exception as e:
        print(f"类型转换测试出现异常: {e}")
        stats.add_result("类型转换异常", False, str(e))
    
    finally:
        stats.end_function()

# ==================== 函数应用测试 ====================
def test_apply_function():
    """测试apply()函数"""
    stats.start_function("函数应用")
    
    try:
        # 测试1: apply() 基本功能
        print("测试 apply() 基本功能...")
        nested = NestedModule()
        
        module_count = 0
        def count_modules(m):
            nonlocal module_count
            module_count += 1
        
        nested.apply(count_modules)
        passed = module_count > 0
        stats.add_result("apply()递归应用", passed, f"统计到{module_count}个模块")
        
        # 测试2: apply() 链式调用
        print("测试 apply() 链式调用...")
        
        def set_attr(m):
            setattr(m, 'test_attr', True)
        
        def check_attr(m):
            return hasattr(m, 'test_attr')
        
        result = nested.apply(set_attr)
        
        # 检查所有模块都有test_attr
        all_have_attr = all(check_attr(m) for m in nested.modules())
        passed = result is nested and all_have_attr
        stats.add_result("apply()链式调用", passed, f"返回自身: {result is nested}, 所有模块有属性: {all_have_attr}")
        
        # 测试3: apply() 与自定义函数
        print("测试 apply() 与自定义函数...")
        
        custom_count = 0
        def count_custom(m):
            nonlocal custom_count
            if isinstance(m, CustomModule):
                custom_count += 1
        
        nested.apply(count_custom)
        passed = custom_count > 0
        stats.add_result("apply()类型检测", passed, f"统计到{custom_count}个CustomModule")
        
    except Exception as e:
        print(f"函数应用测试出现异常: {e}")
        stats.add_result("函数应用异常", False, str(e))
    
    finally:
        stats.end_function()

# ==================== 字符串表示测试 ====================
def test_string_representation():
    """测试字符串表示相关函数"""
    stats.start_function("字符串表示")
    
    try:
        # 测试1: extra_repr() 基本功能
        print("测试 extra_repr() 基本功能...")
        module = CustomModule(10, 5)
        extra_repr = module.extra_repr()
        
        # CustomModule没有实现extra_repr，所以应该返回空字符串
        passed = extra_repr == ''
        stats.add_result("extra_repr()内容", passed, f"extra_repr: '{extra_repr}'")
        
        # 测试2: __repr__() 基本功能
        print("测试 __repr__() 基本功能...")
        repr_str = repr(module)
        
        contains_class_name = 'CustomModule' in repr_str
        contains_extra_info = extra_repr in repr_str
        passed = contains_class_name and contains_extra_info
        stats.add_result("__repr__()基本格式", passed, f"包含类名: {contains_class_name}, 包含额外信息: {contains_extra_info}")
        
        # 测试3: __repr__() 嵌套模块
        print("测试 __repr__() 嵌套模块...")
        nested = NestedModule()
        nested_repr = repr(nested)
        
        contains_nested = 'layer1' in nested_repr and 'layer2' in nested_repr
        has_proper_indentation = '\n' in nested_repr and '    ' in nested_repr
        passed = contains_nested and has_proper_indentation
        stats.add_result("__repr__()嵌套格式", passed, f"包含子模块: {contains_nested}, 有缩进: {has_proper_indentation}")
        
        # 测试4: __repr__() 空模块
        print("测试 __repr__() 空模块...")
        empty_module = Module()
        empty_repr = repr(empty_module)
        
        passed = empty_repr == 'Module()'
        stats.add_result("__repr__()空模块", passed, f"空模块repr: '{empty_repr}'")
        
    except Exception as e:
        print(f"字符串表示测试出现异常: {e}")
        stats.add_result("字符串表示异常", False, str(e))
    
    finally:
        stats.end_function()

# ==================== 模块复制测试 ====================
def test_module_copying():
    """测试模块复制相关函数"""
    stats.start_function("模块复制")
    
    try:
        # 测试1: copy() 浅拷贝
        print("测试 copy() 浅拷贝...")
        module = CustomModule(10, 5)
        copied = module.copy()
        
        is_different_object = copied is not module
        shares_parameters = copied.linear.weight is module.linear.weight
        passed = is_different_object and shares_parameters
        stats.add_result("copy()浅拷贝", passed, f"不同对象: {is_different_object}, 共享参数: {shares_parameters}")
        
        # 测试2: deepcopy() 深拷贝
        print("测试 deepcopy() 深拷贝...")
        deep_copied = module.deepcopy()
        
        is_different_object = deep_copied is not module
        separate_parameters = deep_copied.linear.weight is not module.linear.weight
        same_values = tensor_allclose(deep_copied.linear.weight, module.linear.weight)
        passed = is_different_object and separate_parameters and same_values
        stats.add_result("deepcopy()深拷贝", passed, 
                        f"不同对象: {is_different_object}, 独立参数: {separate_parameters}, 值相同: {same_values}")
        
        # 测试3: __copy__() 支持
        print("测试 __copy__() 支持...")
        copy_copied = copy.copy(module)
        
        is_different_object = copy_copied is not module
        shares_parameters = copy_copied.linear.weight is module.linear.weight
        passed = is_different_object and shares_parameters
        stats.add_result("__copy__()支持", passed, f"不同对象: {is_different_object}, 共享参数: {shares_parameters}")
        
        # 测试4: __deepcopy__() 支持
        print("测试 __deepcopy__() 支持...")
        try:
            deepcopy_copied = copy.deepcopy(module)
            
            is_different_object = deepcopy_copied is not module
            separate_parameters = deepcopy_copied.linear.weight is not module.linear.weight
            same_values = tensor_allclose(deepcopy_copied.linear.weight, module.linear.weight)
            passed = is_different_object and separate_parameters and same_values
            stats.add_result("__deepcopy__()支持", passed, 
                            f"不同对象: {is_different_object}, 独立参数: {separate_parameters}, 值相同: {same_values}")
        except Exception as deep_copy_error:
            print(f"深拷贝测试失败: {deep_copy_error}")
            stats.add_result("__deepcopy__()支持", False, f"深拷贝异常: {deep_copy_error}")
        
    except Exception as e:
        print(f"模块复制测试出现异常: {e}")
        stats.add_result("模块复制异常", False, str(e))
    
    finally:
        stats.end_function()

# ==================== 属性管理测试 ====================
def test_attribute_management():
    """测试属性管理相关函数"""
    stats.start_function("属性管理")
    
    try:
        # 测试1: __setattr__ 参数注册
        print("测试 __setattr__ 参数注册...")
        module = Module()
        
        # 设置Parameter应该自动注册
        module.test_param = Parameter(rm.randn(3, 3))
        has_param = module.has_parameter('test_param')
        passed = has_param
        stats.add_result("__setattr__参数注册", passed, f"自动注册参数: {has_param}")
        
        # 测试2: __setattr__ 子模块注册
        print("测试 __setattr__ 子模块注册...")
        module.test_module = CustomModule(5, 3)
        has_submodule = module.has_submodule('test_module')
        passed = has_submodule
        stats.add_result("__setattr__子模块注册", passed, f"自动注册子模块: {has_submodule}")
        
        # 测试3: __getattr__ 参数访问
        print("测试 __getattr__ 参数访问...")
        retrieved_param = module.test_param
        passed = retrieved_param is module._parameters['test_param']
        stats.add_result("__getattr__参数访问", passed, f"正确访问参数: {retrieved_param is module._parameters['test_param']}")
        
        # 测试4: __getattr__ 子模块访问
        print("测试 __getattr__ 子模块访问...")
        retrieved_module = module.test_module
        passed = retrieved_module is module._modules['test_module']
        stats.add_result("__getattr__子模块访问", passed, f"正确访问子模块: {retrieved_module is module._modules['test_module']}")
        
        # 测试5: __delattr__ 参数删除
        print("测试 __delattr__ 参数删除...")
        del module.test_param
        has_param_after_del = module.has_parameter('test_param')
        passed = not has_param_after_del
        stats.add_result("__delattr__参数删除", passed, f"删除后不存在: {not has_param_after_del}")
        
        # 测试6: __delattr__ 子模块删除
        print("测试 __delattr__ 子模块删除...")
        del module.test_module
        has_module_after_del = module.has_submodule('test_module')
        passed = not has_module_after_del
        stats.add_result("__delattr__子模块删除", passed, f"删除后不存在: {not has_module_after_del}")
        
    except Exception as e:
        print(f"属性管理测试出现异常: {e}")
        stats.add_result("属性管理异常", False, str(e))
    
    finally:
        stats.end_function()

# ==================== 前向传播测试 ====================
def test_forward_propagation():
    """测试前向传播相关函数"""
    stats.start_function("前向传播")
    
    try:
        # 测试1: forward() 基本功能
        print("测试 forward() 基本功能...")
        module = CustomModule(10, 5)
        input_data = rm.randn(2, 10)
        
        output = module.forward(input_data)
        expected_shape = (2, 5)
        actual_shape = output.shape
        passed = actual_shape == expected_shape
        stats.add_result("forward()基本功能", passed, f"输出形状: 期望{expected_shape}, 实际{actual_shape}")
        
        # 测试2: __call__ 调用
        print("测试 __call__ 调用...")
        call_output = module(input_data)
        outputs_equal = tensor_allclose(output, call_output)
        passed = outputs_equal
        stats.add_result("__call__调用", passed, f"forward()和__call__输出一致: {outputs_equal}")
        
        # 测试3: Sequential 前向传播
        print("测试 Sequential 前向传播...")
        seq = Sequential(
            CustomModule(10, 8),
            CustomModule(8, 5),
            CustomModule(5, 1)
        )
        
        seq_input = rm.randn(3, 10)
        seq_output = seq(seq_input)
        expected_seq_shape = (3, 1)
        actual_seq_shape = seq_output.shape
        passed = actual_seq_shape == expected_seq_shape
        stats.add_result("Sequential前向传播", passed, f"输出形状: 期望{expected_seq_shape}, 实际{actual_seq_shape}")
        
        # 测试4: 嵌套模块前向传播
        print("测试嵌套模块前向传播...")
        nested = NestedModule()
        nested_input = rm.randn(4, 10)  # 修正输入维度
        nested_output = nested(nested_input)
        expected_nested_shape = (4, 3)  # 修正输出维度，因为最后是Linear(5, 3)
        actual_nested_shape = nested_output.shape
        passed = actual_nested_shape == expected_nested_shape
        stats.add_result("嵌套模块前向传播", passed, f"输出形状: 期望{expected_nested_shape}, 实际{actual_nested_shape}")
        
    except Exception as e:
        print(f"前向传播测试出现异常: {e}")
        stats.add_result("前向传播异常", False, str(e))
    
    finally:
        stats.end_function()

# ==================== to() 方法测试 ====================
def test_to_method():
    """测试Module的to()方法"""
    stats.start_function("to()方法测试")
    
    try:
        # 检查CUDA是否可用
        CUDA_AVAILABLE = rm.cuda.CUPY_AVAILABLE
        print(f"CUDA可用状态: {CUDA_AVAILABLE}")
        
        # 测试1: 创建模型
        print("测试创建模型...")
        module = CustomModule(10, 5)
        input_data = rm.randn(2, 10)
        
        # 测试2: 模型在CPU上的前向传播
        print("测试模型在CPU上的前向传播...")
        cpu_output = module(input_data)
        expected_shape = (2, 5)
        actual_shape = cpu_output.shape
        passed = actual_shape == expected_shape
        stats.add_result("CPU前向传播", passed, f"输出形状: 期望{expected_shape}, 实际{actual_shape}")
        
        # 测试3: 如果CUDA可用，测试模型移动到CUDA
        if CUDA_AVAILABLE:
            print("测试模型移动到CUDA...")
            try:
                # 将模型移动到CUDA
                cuda_module = module.to('cuda')
                
                # 检查模型参数是否在CUDA上
                params_on_cuda = True
                for i, param in enumerate(cuda_module.parameters()):
                    if not hasattr(param.data, 'device'):
                        params_on_cuda = False
                        break
                    # 检查device属性的字符串表示是否包含'cuda'
                    device_str = str(param.data.device)
                    if 'cuda' not in device_str.lower():
                        params_on_cuda = False
                        break
                
                buffers_on_cuda = True
                for i, buffer in enumerate(cuda_module.buffers()):
                    if not hasattr(buffer.data, 'device'):
                        buffers_on_cuda = False
                        break
                    # 检查device属性的字符串表示是否包含'cuda'
                    device_str = str(buffer.data.device)
                    if 'cuda' not in device_str.lower():
                        buffers_on_cuda = False
                        break
                
                passed = params_on_cuda and buffers_on_cuda
                stats.add_result("模型移动到CUDA", passed, f"参数在CUDA: {params_on_cuda}, 缓冲区在CUDA: {buffers_on_cuda}")
                
                # 测试4: 模型在CUDA上的前向传播
                print("测试模型在CUDA上的前向传播...")
                cuda_input = input_data.to('cuda')
                cuda_output = cuda_module(cuda_input)
                actual_shape = cuda_output.shape
                passed = actual_shape == expected_shape
                stats.add_result("CUDA前向传播", passed, f"输出形状: 期望{expected_shape}, 实际{actual_shape}")
                
                # 测试5: 模型从CUDA移动回CPU
                print("测试模型从CUDA移动回CPU...")
                cpu_module_back = cuda_module.to('cpu')
                
                # 检查模型参数是否在CPU上
                params_on_cpu = True
                for param in cpu_module_back.parameters():
                    if hasattr(param.data, 'device') and param.data.device == 'cuda':
                        params_on_cpu = False
                        break
                
                buffers_on_cpu = True
                for buffer in cpu_module_back.buffers():
                    if hasattr(buffer.data, 'device') and buffer.data.device == 'cuda':
                        buffers_on_cpu = False
                        break
                
                passed = params_on_cpu and buffers_on_cpu
                stats.add_result("模型移动回CPU", passed, f"参数在CPU: {params_on_cpu}, 缓冲区在CPU: {buffers_on_cpu}")
                
                # 测试6: 模型在CPU上的前向传播（移动后）
                print("测试模型在CPU上的前向传播（移动后）...")
                cpu_output_back = cpu_module_back(input_data)
                actual_shape = cpu_output_back.shape
                passed = actual_shape == expected_shape
                stats.add_result("CPU前向传播（移动后）", passed, f"输出形状: 期望{expected_shape}, 实际{actual_shape}")
                
            except Exception as e:
                print(f"CUDA测试出现异常: {e}")
                stats.add_result("CUDA测试异常", False, str(e))
        else:
            print("CUDA不可用，跳过CUDA相关测试")
            stats.add_result("CUDA不可用", True, "跳过CUDA相关测试")
        
        # 测试7: 测试to()方法的dtype参数
        print("测试to()方法的dtype参数...")
        try:
            # 将模型转换为float32
            float32_module = module.to(np.float32)
            
            # 检查模型参数是否为float32
            params_float32 = all(param.data.dtype == np.float32 for param in float32_module.parameters())
            buffers_float32 = all(buffer.data.dtype == np.float32 for buffer in float32_module.buffers())
            passed = params_float32 and buffers_float32
            stats.add_result("to()方法dtype转换", passed, f"参数为float32: {params_float32}, 缓冲区为float32: {buffers_float32}")
            
            # 测试8: 模型在float32上的前向传播
            print("测试模型在float32上的前向传播...")
            float32_input = input_data.to(np.float32)
            float32_output = float32_module(float32_input)
            actual_shape = float32_output.shape
            passed = actual_shape == expected_shape
            stats.add_result("float32前向传播", passed, f"输出形状: 期望{expected_shape}, 实际{actual_shape}")
            
        except Exception as e:
            print(f"dtype转换测试出现异常: {e}")
            stats.add_result("dtype转换测试异常", False, str(e))
        
    except Exception as e:
        print(f"to()方法测试出现异常: {e}")
        stats.add_result("to()方法测试异常", False, str(e))
    
    finally:
        stats.end_function()

# ==================== 主测试函数 ====================
def run_all_tests():
    """运行所有测试"""
    print(f"{Colors.BOLD}Riemann nn.Module 全功能测试套件{Colors.ENDC}")
    print("="*80)
    print(f"Riemann版本: 可用")
    print(f"PyTorch版本: {'可用' if TORCH_AVAILABLE else '不可用'}")
    print("="*80)
    
    # 设置随机种子确保可重复性
    np.random.seed(42)
    if TORCH_AVAILABLE:
        torch.manual_seed(42)
    
    # 运行所有测试
    test_functions = [
        test_parameters_management,
        test_buffers_management,
        test_submodules_management,
        test_state_management,
        test_mode_control,
        test_gradient_management,
        test_type_conversion,
        test_apply_function,
        test_string_representation,
        test_module_copying,
        test_attribute_management,
        test_forward_propagation,
        test_to_method
    ]
    
    start_time = time.time()
    
    for test_func in test_functions:
        try:
            test_func()
        except Exception as e:
            print(f"{Colors.FAIL}测试函数 {test_func.__name__} 执行失败: {e}{Colors.ENDC}")
            stats.add_result(f"{test_func.__name__}失败", False, str(e))
    
    end_time = time.time()
    
    # 打印最终统计
    stats.print_summary()
    
    # 返回是否所有测试都通过
    all_passed = stats.passed_cases == stats.total_cases
    if all_passed:
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}🎉 所有测试通过！{Colors.ENDC}")
    else:
        print(f"\n{Colors.FAIL}{Colors.BOLD}❌ 有 {stats.total_cases - stats.passed_cases} 个测试失败{Colors.ENDC}")
    
    print(f"总耗时: {end_time - start_time:.4f} 秒")
    return all_passed

if __name__ == "__main__":
    clear_screen()
    # 支持作为独立脚本运行
    success = run_all_tests()
    sys.exit(0 if success else 1)