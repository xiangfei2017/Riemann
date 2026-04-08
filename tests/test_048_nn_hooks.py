#!/usr/bin/env python3
"""
Riemann nn.Module 钩子处理测试套件（重构版）

测试架构：
1. 钩子注册与管理
2. 前向预处理钩子（forward_pre_hook）
3. 前向钩子（forward_hook）
4. 反向预处理钩子（backward_pre_hook）
5. 反向钩子（backward_hook）
6. 组合钩子（同时注册反向预处理+反向钩子）
7. 复杂场景测试
"""

import numpy as np
import time
import sys
import os
import pytest

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


# ==================== pytest fixtures ====================

@pytest.fixture
def stats():
    """提供StatisticsCollector实例的fixture"""
    return StatisticsCollector()

# 导入riemann模块
try:
    import riemann as rm
    from riemann.nn import Parameter, Module
    RIEMANN_AVAILABLE = True
except ImportError as e:
    print(f"无法导入riemann模块: {e}")
    RIEMANN_AVAILABLE = False


# ==================== 工具类和函数 ====================

class Colors:
    """颜色类用于美化输出"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class StatisticsCollector:
    """测试统计类"""
    
    def __init__(self):
        self.total_cases = 0
        self.passed_cases = 0
        self.total_time = 0.0
        self.function_stats = {}
        self.current_function = None
        self.current_function_start_time = 0
        self.current_test_details = []
        self.function_test_details = {}

    def start_function(self, function_name):
        self.current_function = function_name
        self.current_function_start_time = time.time()
        self.current_test_details = []

        if function_name not in self.function_stats:
            self.function_stats[function_name] = {"total": 0, "passed": 0, "time": 0.0}

        if function_name not in self.function_test_details:
            self.function_test_details[function_name] = []

        print(f"\n{Colors.BOLD}{function_name}{Colors.ENDC}")
        print("-" * 80)

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
            test_detail = {
                "name": case_name,
                "status": status,
                "color": status_color,
                "details": details
            }
            self.current_test_details.append(test_detail)
            self.function_test_details[self.current_function].append(test_detail)

            print(f"  {case_name} [{status_color}{status}{Colors.ENDC}]" + (f" - {details}" if details else ""))

    def end_function(self):
        if self.current_function:
            elapsed = time.time() - self.current_function_start_time
            self.function_stats[self.current_function]["time"] += elapsed
            self.total_time += elapsed
            self.current_function = None

    def _get_display_width(self, text):
        """计算字符串的显示宽度"""
        width = 0
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                width += 2
            else:
                width += 1
        return width

    def _ljust_display_width(self, text, width):
        """按显示宽度左对齐字符串"""
        display_width = self._get_display_width(text)
        if display_width >= width:
            return text
        padding = width - display_width
        return text + " " * padding

    def print_summary(self):
        """打印测试摘要"""
        headers = ['用例组', '通过/总数', '通过率', '耗时(秒)']
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
        print(f"通过测试用例数: {self.passed_cases}")
        print(f"测试通过率: {100*self.passed_cases/self.total_cases:.2f}%")
        print(f"总耗时: {self.total_time:.4f} 秒")
        print("\n各用例组详情:")
        print("-"*total_width)
        
        header_line = ""
        for i, header in enumerate(headers):
            header_line += self._ljust_display_width(header, col_widths[i])
        print(header_line)
        print("-"*total_width)
        
        for func_name, stats_data in self.function_stats.items():
            pass_rate = 100*stats_data["passed"]/stats_data["total"] if stats_data["total"] > 0 else 0
            
            col1 = self._ljust_display_width(func_name, col_widths[0])
            col2 = f"{stats_data['passed']}/{stats_data['total']}".ljust(col_widths[1])
            col3 = f"{pass_rate:.1f}%".ljust(col_widths[2])
            col4 = f"{stats_data['time']:.4f}".ljust(col_widths[3])
            
            print(f"{col1}{col2}{col3}{col4}")
        
        print("="*total_width)

    def raise_if_failed(self):
        """如果有失败的测试，抛出异常"""
        if self.passed_cases < self.total_cases:
            failed = self.total_cases - self.passed_cases
            raise AssertionError(f"有 {failed} 个测试用例失败")


def tensor_allclose(a, b, rtol=1e-5, atol=1e-8):
    """比较两个张量是否接近"""
    if a is None or b is None:
        return a is b
    if not hasattr(a, 'data') or not hasattr(b, 'data'):
        return False
    return np.allclose(a.data, b.data, rtol=rtol, atol=atol)


# ==================== 测试模块定义 ====================

class SimpleLinearModule(Module):
    """简单线性模块"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(rm.ones((out_features, in_features)))
        self.bias = Parameter(rm.ones((out_features,)))

    def forward(self, x):
        return x @ self.weight.T + self.bias


class TwoInputModule(Module):
    """双输入模块"""
    def __init__(self):
        super().__init__()
        self.weight = Parameter(rm.ones((10, 5)))

    def forward(self, x1, x2):
        return (x1 + x2) @ self.weight


class MultiInputModule(Module):
    """多输入模块"""
    def __init__(self):
        super().__init__()
        self.weight = Parameter(rm.ones((5, 10)))

    def forward(self, x, y):
        return (x + y) @ self.weight.T


class MultiOutputModule(Module):
    """多输出模块"""
    def __init__(self):
        super().__init__()
        self.weight = Parameter(rm.ones((10, 5)))

    def forward(self, x):
        return x @ self.weight, x @ self.weight * 2


class NoParamModule(Module):
    """无参数模块"""
    def forward(self, x1, x2):
        return x1 * 2 + x2 * 3


class IdentityPassThroughModule(Module):
    """输入透传模块"""
    def __init__(self):
        super().__init__()
        self.weight = Parameter(rm.ones((5, 10)))

    def forward(self, x1, x2):
        out1 = x1  # 透传
        out2 = x2 @ self.weight.T
        return out1, out2


class InnerModule(Module):
    """内部模块"""
    def __init__(self):
        super().__init__()
        self.weight = Parameter(rm.ones((5, 10)))

    def forward(self, x):
        return x @ self.weight.T


class OuterModule(Module):
    """外部模块"""
    def __init__(self):
        super().__init__()
        self.inner = InnerModule()

    def forward(self, x):
        return self.inner(x)


class ThreeInputThreeOutputModule(Module):
    """3输入3输出模块"""
    def __init__(self):
        super().__init__()
        self.weight1 = Parameter(rm.ones(2))
        self.weight2 = Parameter(rm.ones(3))
        self.weight3 = Parameter(rm.ones(4))

    def forward(self, x1, x2, x3):
        out1 = x1 * self.weight1
        out2 = x2 * self.weight2
        out3 = x3 * self.weight3
        return out1, out2, out3


# ==================== pytest测试函数 ====================

def test_hook_registration_and_management(stats):
    """测试钩子注册与管理"""
    stats.start_function("钩子注册与管理")
    
    try:
        # 测试1: register_forward_pre_hook
        print("测试 register_forward_pre_hook...")
        module = SimpleLinearModule(10, 5)
        hook_called = False
        
        def forward_pre_hook(module, input):
            nonlocal hook_called
            hook_called = True
            return input
        
        handle = module.register_forward_pre_hook(forward_pre_hook)
        input_data = rm.randn(2, 10)
        output = module(input_data)
        
        passed = hook_called
        stats.add_result("register_forward_pre_hook调用", passed, f"前向预处理钩子被调用: {hook_called}")
        
        # 测试2: 移除前向预处理钩子
        print("测试 forward_pre_hook 移除...")
        hook_called = False
        handle.remove()
        input_data = rm.randn(2, 10)
        output = module(input_data)
        
        passed = not hook_called
        stats.add_result("forward_pre_hook移除", passed, f"前向预处理钩子被正确移除: {not hook_called}")
        
        # 测试3: register_forward_hook
        print("测试 register_forward_hook...")
        module = SimpleLinearModule(10, 5)
        hook_called = False
        
        def forward_hook(module, input, output):
            nonlocal hook_called
            hook_called = True
        
        handle = module.register_forward_hook(forward_hook)
        input_data = rm.randn(2, 10)
        output = module(input_data)
        
        passed = hook_called
        stats.add_result("register_forward_hook调用", passed, f"前向钩子被调用: {hook_called}")
        
        # 测试4: 移除前向钩子
        print("测试 forward_hook 移除...")
        hook_called = False
        handle.remove()
        input_data = rm.randn(2, 10)
        output = module(input_data)
        
        passed = not hook_called
        stats.add_result("forward_hook移除", passed, f"前向钩子被正确移除: {not hook_called}")
        
        # 测试5: register_full_backward_pre_hook
        print("测试 register_full_backward_pre_hook...")
        module = SimpleLinearModule(10, 5)
        hook_called = False
        
        def backward_pre_hook(module, grad_output):
            nonlocal hook_called
            hook_called = True
            return grad_output
        
        handle = module.register_full_backward_pre_hook(backward_pre_hook)
        input_data = rm.randn(2, 10, requires_grad=True)
        output = module(input_data)
        loss = output.sum()
        loss.backward()
        
        passed = hook_called
        stats.add_result("register_full_backward_pre_hook调用", passed, f"反向预处理钩子被调用: {hook_called}")
        
        # 测试6: 移除反向预处理钩子
        print("测试 backward_pre_hook 移除...")
        hook_called = False
        handle.remove()
        input_data = rm.randn(2, 10, requires_grad=True)
        output = module(input_data)
        loss = output.sum()
        loss.backward()
        
        passed = not hook_called
        stats.add_result("register_full_backward_pre_hook移除", passed, f"反向预处理钩子被正确移除: {not hook_called}")
        
        # 测试7: register_full_backward_hook
        print("测试 register_full_backward_hook...")
        module = SimpleLinearModule(10, 5)
        hook_called = False
        
        def backward_hook(module, grad_input, grad_output):
            nonlocal hook_called
            hook_called = True
            return grad_input
        
        handle = module.register_full_backward_hook(backward_hook)
        input_data = rm.randn(2, 10, requires_grad=True)
        output = module(input_data)
        loss = output.sum()
        loss.backward()
        
        passed = hook_called
        stats.add_result("register_full_backward_hook调用", passed, f"反向钩子被调用: {hook_called}")
        
        # 测试8: 移除反向钩子
        print("测试 backward_hook 移除...")
        hook_called = False
        handle.remove()
        input_data = rm.randn(2, 10, requires_grad=True)
        output = module(input_data)
        loss = output.sum()
        loss.backward()
        
        passed = not hook_called
        stats.add_result("register_full_backward_hook移除", passed, f"反向钩子被正确移除: {not hook_called}")
        
        # 测试9: 多个前向预处理钩子同时注册
        print("测试多个前向预处理钩子同时注册...")
        module = SimpleLinearModule(10, 5)
        hook_count = 0
        
        def hook1(m, input):
            nonlocal hook_count
            hook_count += 1
            return input
        
        def hook2(m, input):
            nonlocal hook_count
            hook_count += 1
            return input
        
        handle1 = module.register_forward_pre_hook(hook1)
        handle2 = module.register_forward_pre_hook(hook2)
        
        input_data = rm.randn(2, 10)
        output = module(input_data)
        
        passed = hook_count == 2
        stats.add_result("多个前向预处理钩子同时注册", passed, f"所有钩子被调用: {hook_count == 2}")
        
        handle1.remove()
        handle2.remove()
        
        # 测试10: 多个前向钩子同时注册
        print("测试多个前向钩子同时注册...")
        module = SimpleLinearModule(10, 5)
        hook1_called = False
        hook2_called = False
        
        def forward_hook1(m, input, output):
            nonlocal hook1_called
            hook1_called = True
            return output
        
        def forward_hook2(m, input, output):
            nonlocal hook2_called
            hook2_called = True
            return output
        
        handle1 = module.register_forward_hook(forward_hook1)
        handle2 = module.register_forward_hook(forward_hook2)
        
        input_data = rm.randn(2, 10)
        output = module(input_data)
        
        passed = hook1_called and hook2_called
        stats.add_result("多个前向钩子同时注册", passed, f"所有钩子被调用: hook1={hook1_called}, hook2={hook2_called}")
        
        handle1.remove()
        handle2.remove()
        
        # 测试11: 多个反向预处理钩子同时注册
        print("测试多个反向预处理钩子同时注册...")
        module = SimpleLinearModule(10, 5)
        hook_count = 0
        
        def pre_hook1(m, grad_output):
            nonlocal hook_count
            hook_count += 1
            return grad_output
        
        def pre_hook2(m, grad_output):
            nonlocal hook_count
            hook_count += 1
            return grad_output
        
        handle1 = module.register_full_backward_pre_hook(pre_hook1)
        handle2 = module.register_full_backward_pre_hook(pre_hook2)
        
        input_data = rm.randn(2, 10, requires_grad=True)
        output = module(input_data)
        loss = output.sum()
        loss.backward()
        
        passed = hook_count == 2
        stats.add_result("多个反向预处理钩子同时注册", passed, f"所有钩子被调用: {hook_count == 2}")
        
        handle1.remove()
        handle2.remove()
        
        # 测试12: 多个反向钩子同时注册
        print("测试多个反向钩子同时注册...")
        module = SimpleLinearModule(10, 5)
        hook1_called = False
        hook2_called = False
        
        def backward_hook1(m, grad_input, grad_output):
            nonlocal hook1_called
            hook1_called = True
            return grad_input
        
        def backward_hook2(m, grad_input, grad_output):
            nonlocal hook2_called
            hook2_called = True
            return grad_input
        
        handle1 = module.register_full_backward_hook(backward_hook1)
        handle2 = module.register_full_backward_hook(backward_hook2)
        
        input_data = rm.randn(2, 10, requires_grad=True)
        output = module(input_data)
        loss = output.sum()
        loss.backward()
        
        passed = hook1_called and hook2_called
        stats.add_result("多个反向钩子同时注册", passed, f"所有钩子被调用: hook1={hook1_called}, hook2={hook2_called}")
        
        handle1.remove()
        handle2.remove()
        
    finally:
        stats.end_function()


def test_forward_pre_hooks(stats):
    """测试前向预处理钩子"""
    stats.start_function("前向预处理钩子")
    
    try:
        # 测试1: 基本功能
        print("测试前向预处理钩子基本功能...")
        module = SimpleLinearModule(10, 5)
        hook_called = False
        
        def pre_hook(m, input):
            nonlocal hook_called
            hook_called = True
            return input
        
        module.register_forward_pre_hook(pre_hook)
        input_data = rm.randn(2, 10)
        output = module(input_data)
        
        passed = hook_called and output is not None
        stats.add_result("前向预处理钩子基本功能", passed, f"钩子调用: {hook_called}, 输出有效: {output is not None}")
        
        # 测试2: 多输入
        print("测试前向预处理钩子 - 多输入...")
        module = MultiInputModule()
        hook_called = False
        input_received = None
        
        def pre_hook_multi(m, input):
            nonlocal hook_called, input_received
            hook_called = True
            input_received = input
            return input
        
        module.register_forward_pre_hook(pre_hook_multi)
        x = rm.randn(2, 10)
        y = rm.randn(2, 10)
        output = module(x, y)
        
        passed = hook_called and input_received is not None and len(input_received) == 2
        stats.add_result("前向预处理钩子 - 多输入", passed, 
                        f"钩子被调用: {hook_called}, 收到输入: {input_received is not None}")
        
        # 测试3: 修改输入
        print("测试前向预处理钩子 - 修改输入...")
        module = SimpleLinearModule(10, 5)
        input_modified = False
        
        def pre_hook_modify(m, input):
            nonlocal input_modified
            input_modified = True
            return (input[0] * 2,)
        
        module.register_forward_pre_hook(pre_hook_modify)
        input_data = rm.ones(2, 10)
        output = module(input_data)
        expected = input_data * 2 @ module.weight.T + module.bias
        
        passed = input_modified and tensor_allclose(output, expected)
        stats.add_result("前向预处理钩子 - 修改输入", passed, 
                        f"输入被修改: {input_modified}, 输出正确: {tensor_allclose(output, expected)}")
        
        # 测试4: 返回None
        print("测试前向预处理钩子 - 返回None...")
        module = SimpleLinearModule(10, 5)
        hook_called = False
        
        def pre_hook_none(m, input):
            nonlocal hook_called
            hook_called = True
            return None
        
        module.register_forward_pre_hook(pre_hook_none)
        input_data = rm.ones(2, 10)
        output = module(input_data)
        expected = input_data @ module.weight.T + module.bias
        
        passed = hook_called and tensor_allclose(output, expected)
        stats.add_result("前向预处理钩子 - 返回None", passed, 
                        f"钩子被调用: {hook_called}, 输出正确: {tensor_allclose(output, expected)}")
        
        # 测试5: 验证输入修改效果
        print("测试前向预处理钩子 - 验证输入修改效果...")
        module = SimpleLinearModule(10, 5)
        hook_called = False
        
        def pre_hook_verify_modify(m, input):
            nonlocal hook_called
            hook_called = True
            # 将输入乘以2
            modified_input = (input[0] * 2,)
            return modified_input
        
        module.register_forward_pre_hook(pre_hook_verify_modify)
        
        input_data = rm.ones(2, 10)
        output = module(input_data)
        
        # 验证输出（输入被乘以2，所以输出应该是原来的2倍）
        expected_output = (input_data * 2) @ module.weight.data.T + module.bias.data
        output_correct = tensor_allclose(output, expected_output)
        passed = hook_called and output_correct
        stats.add_result("前向预处理钩子 - 验证输入修改效果", passed, 
                        f"钩子被调用: {hook_called}, 输出正确: {output_correct}")
        
        # 测试6: 同一模块多次调用
        print("测试前向预处理钩子 - 同一模块多次调用...")
        module = SimpleLinearModule(10, 5)
        call_count = 0
        
        def count_pre_hook(m, input):
            nonlocal call_count
            call_count += 1
            return input
        
        module.register_forward_pre_hook(count_pre_hook)
        
        # 多次调用
        for i in range(3):
            input_data = rm.randn(2, 10)
            output = module(input_data)
        
        passed = call_count == 3
        stats.add_result("前向预处理钩子 - 同一模块多次调用", passed, 
                        f"钩子被调用次数: {call_count}")
        
        # 测试7: 多个钩子同时注册
        print("测试前向预处理钩子 - 多个钩子同时注册...")
        module = SimpleLinearModule(10, 5)
        hook_count = 0
        
        def hook1(m, input):
            nonlocal hook_count
            hook_count += 1
            return input
        
        def hook2(m, input):
            nonlocal hook_count
            hook_count += 1
            return input
        
        handle1 = module.register_forward_pre_hook(hook1)
        handle2 = module.register_forward_pre_hook(hook2)
        
        input_data = rm.randn(2, 10)
        output = module(input_data)
        
        passed = hook_count == 2
        stats.add_result("前向预处理钩子 - 多个钩子同时注册", passed, 
                        f"所有钩子被调用: {hook_count == 2}")
        
        handle1.remove()
        handle2.remove()
        
    finally:
        stats.end_function()


def test_forward_hooks(stats):
    """测试前向钩子"""
    stats.start_function("前向钩子")
    
    try:
        # 测试1: 基本功能
        print("测试前向钩子基本功能...")
        module = SimpleLinearModule(10, 5)
        hook_called = False
        
        def forward_hook(m, input, output):
            nonlocal hook_called
            hook_called = True
        
        module.register_forward_hook(forward_hook)
        input_data = rm.randn(2, 10)
        output = module(input_data)
        
        passed = hook_called and output is not None
        stats.add_result("前向钩子基本功能", passed, 
                        f"钩子调用: {hook_called}, 输出有效: {output is not None}")
        
        # 测试2: 多输入
        print("测试前向钩子 - 多输入...")
        module = MultiInputModule()
        hook_called = False
        
        def forward_hook_multi(m, input, output):
            nonlocal hook_called
            hook_called = True
        
        module.register_forward_hook(forward_hook_multi)
        x = rm.randn(2, 10)
        y = rm.randn(2, 10)
        output = module(x, y)
        
        passed = hook_called and output is not None
        stats.add_result("前向钩子 - 多输入", passed, 
                        f"钩子被调用: {hook_called}, 收到输出: {output is not None}")
        
        # 测试3: 修改输出
        print("测试前向钩子 - 修改输出...")
        module = SimpleLinearModule(10, 5)
        output_modified = False
        
        def forward_hook_modify(m, input, output):
            nonlocal output_modified
            output_modified = True
            output[:] = output * 2
        
        module.register_forward_hook(forward_hook_modify)
        input_data = rm.ones(2, 10)
        output = module(input_data)
        expected = (input_data @ module.weight.T + module.bias) * 2
        
        passed = output_modified and tensor_allclose(output, expected)
        stats.add_result("前向钩子 - 修改输出", passed, 
                        f"输出被修改: {output_modified}, 输出正确: {tensor_allclose(output, expected)}")
        
        # 测试4: 返回None
        print("测试前向钩子 - 返回None...")
        module = SimpleLinearModule(10, 5)
        hook_called = False
        
        def forward_hook_none(m, input, output):
            nonlocal hook_called
            hook_called = True
            return None
        
        module.register_forward_hook(forward_hook_none)
        input_data = rm.randn(2, 10)
        output = module(input_data)
        expected = input_data @ module.weight.T + module.bias
        
        passed = hook_called and tensor_allclose(output, expected)
        stats.add_result("前向钩子 - 返回None", passed, 
                        f"钩子被调用: {hook_called}, 输出正确: {tensor_allclose(output, expected)}")
        
        # 测试5: 多输出模块
        print("测试前向钩子 - 多输出模块...")
        module = MultiOutputModule()
        hook_called = False
        output_received = None
        
        def forward_hook_multi_output(m, input, output):
            nonlocal hook_called, output_received
            hook_called = True
            output_received = output
            return output
        
        module.register_forward_hook(forward_hook_multi_output)
        input_data = rm.randn(2, 10)
        out1, out2 = module(input_data)
        
        passed = hook_called and output_received is not None
        stats.add_result("前向钩子 - 多输出模块", passed, 
                        f"钩子被调用: {hook_called}, 收到输出: {output_received is not None}")
        
        # 测试6: 验证输出修改效果
        print("测试前向钩子 - 验证输出修改效果...")
        module = SimpleLinearModule(10, 5)
        hook_called = False
        
        def forward_hook_verify_modify(m, input, output):
            nonlocal hook_called
            hook_called = True
            # 将输出乘以3
            return output * 3
        
        module.register_forward_hook(forward_hook_verify_modify)
        
        input_data = rm.ones(2, 10)
        output = module(input_data)
        
        # 验证输出（输出被乘以3）
        expected_output = (input_data @ module.weight.T + module.bias) * 3
        output_correct = tensor_allclose(output, expected_output)
        passed = hook_called and output_correct
        stats.add_result("前向钩子 - 验证输出修改效果", passed, 
                        f"钩子被调用: {hook_called}, 输出正确: {output_correct}")
        
        # 测试7: 同一模块多次调用
        print("测试前向钩子 - 同一模块多次调用...")
        module = SimpleLinearModule(10, 5)
        call_count = 0
        
        def count_forward_hook(m, input, output):
            nonlocal call_count
            call_count += 1
            return output
        
        module.register_forward_hook(count_forward_hook)
        
        # 多次调用
        for i in range(3):
            input_data = rm.randn(2, 10)
            output = module(input_data)
        
        passed = call_count == 3
        stats.add_result("前向钩子 - 同一模块多次调用", passed, 
                        f"钩子被调用次数: {call_count}")
        
    finally:
        stats.end_function()


def test_backward_pre_hooks(stats):
    """测试反向预处理钩子"""
    stats.start_function("反向预处理钩子")
    
    try:
        # 测试1: 基本功能
        print("测试反向预处理钩子基本功能...")
        module = SimpleLinearModule(10, 5)
        hook_called = False
        
        def backward_pre_hook(m, grad_output):
            nonlocal hook_called
            hook_called = True
            return grad_output
        
        module.register_full_backward_pre_hook(backward_pre_hook)
        input_data = rm.randn(2, 10, requires_grad=True)
        output = module(input_data)
        loss = output.sum()
        loss.backward()
        
        passed = hook_called
        stats.add_result("反向预处理钩子基本功能", passed, f"钩子调用: {hook_called}")
        
        # 测试2: 修改梯度
        print("测试反向预处理钩子 - 修改梯度...")
        module = SimpleLinearModule(10, 5)
        hook_called = False
        
        def backward_pre_hook_modify(m, grad_output):
            nonlocal hook_called
            hook_called = True
            return tuple(g * 2 if g is not None else None for g in grad_output)
        
        module.register_full_backward_pre_hook(backward_pre_hook_modify)
        input_data = rm.ones(2, 10, requires_grad=True)
        output = module(input_data)
        output.sum().backward()
        
        expected_grad = rm.ones((2, 5)) @ module.weight.data * 2
        grad_correct = tensor_allclose(input_data.grad, expected_grad)
        
        passed = hook_called and grad_correct
        stats.add_result("反向预处理钩子 - 修改梯度", passed, 
                        f"钩子调用: {hook_called}, 梯度正确: {grad_correct}")
        
        # 测试3: 多输出模块
        print("测试反向预处理钩子 - 多输出模块...")
        module = MultiOutputModule()
        hook_called = False
        
        def backward_pre_hook_multi(m, grad_output):
            nonlocal hook_called
            hook_called = True
            return grad_output
        
        module.register_full_backward_pre_hook(backward_pre_hook_multi)
        input_data = rm.randn(2, 10, requires_grad=True)
        out1, out2 = module(input_data)
        (out1 + out2).sum().backward()
        
        passed = hook_called
        stats.add_result("反向预处理钩子 - 多输出模块", passed, f"钩子调用: {hook_called}")
        
        # 测试4: 无参数模块 + 输入无需梯度
        print("测试无参数模块 + 反向预处理钩子（输入无需梯度）...")
        module = NoParamModule()
        hook_called = False
        
        def backward_pre_hook_no_param(m, grad_output):
            nonlocal hook_called
            hook_called = True
            return grad_output
        
        module.register_full_backward_pre_hook(backward_pre_hook_no_param)
        x1 = rm.tensor([1.0, 2.0, 3.0])
        x2 = rm.tensor([4.0, 5.0, 6.0])
        out = module(x1, x2)
        out_grad = rm.tensor(out.data, requires_grad=True)
        out_grad.backward(rm.ones_like(out_grad))
        
        passed = not hook_called
        stats.add_result("无参数模块 + 反向预处理钩子（输入无需梯度）", passed, 
                        f"钩子未调用（符合预期）: {not hook_called}")
        
        # 测试5: 无参数模块 + 输入需要梯度
        print("测试无参数模块 + 反向预处理钩子（输入需要梯度）...")
        module = NoParamModule()
        hook_called = False
        
        def pre_hook_no_param_grad(m, grad_output):
            nonlocal hook_called
            hook_called = True
            return grad_output
        
        module.register_full_backward_pre_hook(pre_hook_no_param_grad)
        x1 = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
        x2 = rm.tensor([4.0, 5.0, 6.0], requires_grad=True)
        out = module(x1, x2)
        out.backward(rm.ones_like(out))
        
        passed = hook_called
        stats.add_result("无参数模块 + 反向预处理钩子（输入需要梯度）", passed, 
                        f"钩子调用: {hook_called}")
        
        # 测试6: 多次调用
        print("测试反向预处理钩子 - 多次调用...")
        module = SimpleLinearModule(10, 5)
        hook_count = 0
        
        def count_pre_hook(m, grad_output):
            nonlocal hook_count
            hook_count += 1
            return grad_output
        
        module.register_full_backward_pre_hook(count_pre_hook)
        
        for i in range(3):
            input_data = rm.randn(2, 10, requires_grad=True)
            output = module(input_data)
            output.sum().backward()
        
        passed = hook_count == 3
        stats.add_result("反向预处理钩子 - 多次调用", passed, 
                        f"钩子被调用次数: {hook_count}")
        
        # 测试7: 验证grad_output值
        print("测试反向预处理钩子 - 验证grad_output值...")
        module = SimpleLinearModule(10, 5)
        hook_called = False
        grad_output_values = None
        
        def verify_pre_hook(m, grad_output):
            nonlocal hook_called, grad_output_values
            hook_called = True
            grad_output_values = grad_output
            return grad_output
        
        module.register_full_backward_pre_hook(verify_pre_hook)
        
        input_data = rm.randn(2, 10, requires_grad=True)
        output = module(input_data)
        loss = output.sum()
        loss.backward()
        
        # 验证grad_output的值（对于sum()损失，grad_output应该是全1）
        grad_output_correct = (grad_output_values is not None and 
                               tensor_allclose(grad_output_values[0], rm.ones((2, 5))))
        passed = hook_called and grad_output_correct
        stats.add_result("反向预处理钩子 - 验证grad_output值", passed, 
                        f"钩子调用: {hook_called}, grad_output正确: {grad_output_correct}")
        
        # 测试8: 多输入模块
        print("测试反向预处理钩子 - 多输入模块...")
        module = TwoInputModule()
        hook_called = False
        
        def multi_input_pre_hook(m, grad_output):
            nonlocal hook_called
            hook_called = True
            return grad_output
        
        module.register_full_backward_pre_hook(multi_input_pre_hook)
        
        x1 = rm.ones(2, 10, requires_grad=True)
        x2 = rm.ones(2, 10, requires_grad=True)
        output = module(x1, x2)
        output.sum().backward()
        
        passed = hook_called
        stats.add_result("反向预处理钩子 - 多输入模块", passed, 
                        f"钩子调用: {hook_called}")
        
        # 测试9: 单输出模块 + 不修改
        print("测试反向预处理钩子 - 单输出模块 + 不修改...")
        module = SimpleLinearModule(10, 5)
        hook_called = False
        
        def pre_hook_no_modify(m, grad_output):
            nonlocal hook_called
            hook_called = True
            return None  # 不修改
        
        module.register_full_backward_pre_hook(pre_hook_no_modify)
        input_data = rm.ones(2, 10, requires_grad=True)
        output = module(input_data)
        output.sum().backward()
        
        passed = hook_called
        stats.add_result("反向预处理钩子 - 单输出模块 + 不修改", passed, 
                        f"钩子调用: {hook_called}")
        
        # 测试10: 多输出模块 + 部分修改
        print("测试反向预处理钩子 - 多输出模块 + 部分修改...")
        module = MultiOutputModule()
        hook_called = False
        grad_modified = False
        
        def pre_hook_partial_modify(m, grad_output):
            nonlocal hook_called, grad_modified
            hook_called = True
            # 只修改第一个输出梯度
            modified = (grad_output[0] * 2,) + grad_output[1:]
            grad_modified = True
            return modified
        
        module.register_full_backward_pre_hook(pre_hook_partial_modify)
        input_data = rm.ones(2, 10, requires_grad=True)
        out1, out2 = module(input_data)
        (out1 + out2).sum().backward()
        
        passed = hook_called and grad_modified
        stats.add_result("反向预处理钩子 - 多输出模块 + 部分修改", passed, 
                        f"钩子调用: {hook_called}, 梯度被修改: {grad_modified}")
        
        # 测试11: 多输出模块 + 全部修改
        print("测试反向预处理钩子 - 多输出模块 + 全部修改...")
        module = MultiOutputModule()
        hook_called = False
        grad_modified = False
        
        def pre_hook_all_modify(m, grad_output):
            nonlocal hook_called, grad_modified
            hook_called = True
            # 修改所有输出梯度
            modified = tuple(g * 2 for g in grad_output)
            grad_modified = True
            return modified
        
        module.register_full_backward_pre_hook(pre_hook_all_modify)
        input_data = rm.ones(2, 10, requires_grad=True)
        out1, out2 = module(input_data)
        (out1 + out2).sum().backward()
        
        passed = hook_called and grad_modified
        stats.add_result("反向预处理钩子 - 多输出模块 + 全部修改", passed, 
                        f"钩子调用: {hook_called}, 梯度被修改: {grad_modified}")
        
        # 测试12: 有参数模块 + 输入无需梯度但有参数
        print("测试反向预处理钩子 - 有参数模块 + 输入无需梯度...")
        module = SimpleLinearModule(10, 5)
        hook_called = False
        
        def pre_hook_with_param(m, grad_output):
            nonlocal hook_called
            hook_called = True
            return grad_output
        
        module.register_full_backward_pre_hook(pre_hook_with_param)
        input_data = rm.ones(2, 10, requires_grad=False)
        output = module(input_data)
        output.sum().backward()
        
        passed = hook_called
        stats.add_result("反向预处理钩子 - 有参数模块 + 输入无需梯度", passed, 
                        f"钩子调用: {hook_called}")
        
    finally:
        stats.end_function()


def test_backward_hooks(stats):
    """测试反向钩子"""
    stats.start_function("反向钩子")
    
    try:
        # 测试1: 基本功能
        print("测试反向钩子基本功能...")
        module = SimpleLinearModule(10, 5)
        hook_called = False
        
        def backward_hook(m, grad_input, grad_output):
            nonlocal hook_called
            hook_called = True
            return grad_input
        
        module.register_full_backward_hook(backward_hook)
        input_data = rm.randn(2, 10, requires_grad=True)
        output = module(input_data)
        loss = output.sum()
        loss.backward()
        
        passed = hook_called
        stats.add_result("反向钩子基本功能", passed, f"钩子调用: {hook_called}")
        
        # 测试2: 修改梯度
        print("测试反向钩子 - 修改梯度...")
        module = SimpleLinearModule(10, 5)
        hook_called = False
        
        def backward_hook_modify(m, grad_input, grad_output):
            nonlocal hook_called
            hook_called = True
            return tuple(g * 3 if g is not None else None for g in grad_input)
        
        module.register_full_backward_hook(backward_hook_modify)
        input_data = rm.ones(2, 10, requires_grad=True)
        output = module(input_data)
        output.sum().backward()
        
        expected_grad = rm.ones((2, 5)) @ module.weight.data * 3
        grad_correct = tensor_allclose(input_data.grad, expected_grad)
        
        passed = hook_called and grad_correct
        stats.add_result("反向钩子 - 修改梯度", passed, 
                        f"钩子调用: {hook_called}, 梯度正确: {grad_correct}")
        
        # 测试3: 嵌套模块
        print("测试嵌套模块反向钩子...")
        module = OuterModule()
        inner_hook_called = False
        outer_hook_called = False
        
        def inner_backward_hook(module, grad_input, grad_output):
            nonlocal inner_hook_called
            inner_hook_called = True
            return grad_input
        
        def outer_backward_hook(module, grad_input, grad_output):
            nonlocal outer_hook_called
            outer_hook_called = True
            return grad_input
        
        module.inner.register_full_backward_hook(inner_backward_hook)
        module.register_full_backward_hook(outer_backward_hook)
        
        input_data = rm.ones(2, 10, requires_grad=True)
        output = module(input_data)
        loss = output.sum()
        loss.backward()
        
        passed = inner_hook_called and outer_hook_called
        stats.add_result("嵌套模块反向钩子", passed, 
                        f"内部钩子调用: {inner_hook_called}, 外部钩子调用: {outer_hook_called}")
        
        # 测试4: 无参数模块 + 输入无需梯度
        print("测试无参数模块反向钩子（输入无需梯度）...")
        module = NoParamModule()
        hook_called = False
        
        def backward_hook_no_param(m, grad_input, grad_output):
            nonlocal hook_called
            hook_called = True
            return grad_input
        
        module.register_full_backward_hook(backward_hook_no_param)
        x1 = rm.tensor([1.0, 2.0, 3.0])
        x2 = rm.tensor([4.0, 5.0, 6.0])
        out = module(x1, x2)
        out_grad = rm.tensor(out.data, requires_grad=True)
        out_grad.backward(rm.ones_like(out_grad))
        
        passed = not hook_called
        stats.add_result("无参数模块反向钩子（输入无需梯度）", passed, 
                        f"钩子未调用（符合预期）: {not hook_called}")
        
        # 测试5: 多模块共享输入
        print("测试多模块共享输入 + 反向钩子...")
        
        class Module1(Module):
            def __init__(self):
                super().__init__()
                self.weight = Parameter(rm.ones((5, 10)))
            def forward(self, x):
                return x @ self.weight.T
        
        class Module2(Module):
            def __init__(self):
                super().__init__()
                self.weight = Parameter(rm.ones((3, 10)))
            def forward(self, x):
                return x @ self.weight.T
        
        module1 = Module1()
        module2 = Module2()
        hook1_called = False
        hook2_called = False
        
        def hook1(m, grad_input, grad_output):
            nonlocal hook1_called
            hook1_called = True
            return grad_input
        
        def hook2(m, grad_input, grad_output):
            nonlocal hook2_called
            hook2_called = True
            return grad_input
        
        module1.register_full_backward_hook(hook1)
        module2.register_full_backward_hook(hook2)
        
        x = rm.ones(2, 10, requires_grad=True)
        out1 = module1(x)
        out2 = module2(x)
        (out1.sum() + out2.sum()).backward()
        
        passed = hook1_called and hook2_called
        stats.add_result("多模块共享输入 + 反向钩子", passed, 
                        f"模块1钩子调用: {hook1_called}, 模块2钩子调用: {hook2_called}")
        
        # 测试6: 返回None
        print("测试反向钩子 - 返回None...")
        module = SimpleLinearModule(10, 5)
        hook_called = False
        
        def backward_hook_none(m, grad_input, grad_output):
            nonlocal hook_called
            hook_called = True
            return None
        
        module.register_full_backward_hook(backward_hook_none)
        input_data = rm.ones(2, 10, requires_grad=True)
        output = module(input_data)
        output.sum().backward()
        
        expected_grad = rm.ones((2, 5)) @ module.weight.data
        grad_correct = tensor_allclose(input_data.grad, expected_grad)
        
        passed = hook_called and grad_correct
        stats.add_result("反向钩子 - 返回None", passed, 
                        f"钩子调用: {hook_called}, 梯度正确: {grad_correct}")
        
        # 测试7: 无参数模块 + 输入需要梯度
        print("测试无参数模块反向钩子（输入需要梯度）...")
        module = NoParamModule()
        hook_called = False
        
        def backward_hook_no_param_grad(m, grad_input, grad_output):
            nonlocal hook_called
            hook_called = True
            return grad_input
        
        module.register_full_backward_hook(backward_hook_no_param_grad)
        x1 = rm.tensor([1.0, 2.0, 3.0], requires_grad=True)
        x2 = rm.tensor([4.0, 5.0, 6.0], requires_grad=True)
        out = module(x1, x2)
        out.backward(rm.ones_like(out))
        
        passed = hook_called
        stats.add_result("无参数模块反向钩子（输入需要梯度）", passed, 
                        f"钩子调用: {hook_called}")
        
        # 测试8: 修改输入梯度
        print("测试反向钩子 - 修改输入梯度...")
        module = SimpleLinearModule(10, 5)
        hook_called = False
        
        def backward_hook_modify_grad(m, grad_input, grad_output):
            nonlocal hook_called
            hook_called = True
            return tuple(g * 2 if g is not None else None for g in grad_input)
        
        module.register_full_backward_hook(backward_hook_modify_grad)
        
        input_data = rm.ones(2, 10, requires_grad=True)
        output = module(input_data)
        output.sum().backward()
        
        expected_grad = rm.ones((2, 5)) @ module.weight.data * 2
        grad_correct = tensor_allclose(input_data.grad, expected_grad)
        passed = hook_called and grad_correct
        stats.add_result("反向钩子 - 修改输入梯度", passed, 
                        f"钩子调用: {hook_called}, 梯度正确: {grad_correct}")
        
        # 测试9: 多次前向传播后的backward
        print("测试多次前向传播后的backward...")
        module = SimpleLinearModule(10, 5)
        hook_call_count = 0
        
        def backward_hook_count(m, grad_input, grad_output):
            nonlocal hook_call_count
            hook_call_count += 1
            return grad_input
        
        module.register_full_backward_hook(backward_hook_count)
        
        for i in range(3):
            input_data = rm.randn(2, 10, requires_grad=True)
            output = module(input_data)
            loss = output.sum()
            loss.backward()
        
        passed = hook_call_count == 3
        stats.add_result("多次前向传播后的backward", passed, 
                        f"钩子被调用次数: {hook_call_count}")
        
        # 测试10: 单输入模块 + 部分输入需要梯度
        print("测试反向钩子 - 单输入模块 + 部分输入需要梯度...")
        module = TwoInputModule()
        hook_called = False
        
        def backward_hook_partial(m, grad_input, grad_output):
            nonlocal hook_called
            hook_called = True
            return grad_input
        
        module.register_full_backward_hook(backward_hook_partial)
        
        x = rm.ones(2, 10, requires_grad=True)
        y = rm.ones(2, 10, requires_grad=False)
        output = module(x, y)
        output.sum().backward()
        
        passed = hook_called and x.grad is not None
        stats.add_result("反向钩子 - 单输入模块 + 部分输入需要梯度", passed, 
                        f"钩子调用: {hook_called}, x.grad不为None: {x.grad is not None}")
        
        # 测试11: 单输入模块 + 不修改
        print("测试反向钩子 - 单输入模块 + 不修改...")
        module = SimpleLinearModule(10, 5)
        hook_called = False
        
        def backward_hook_no_modify(m, grad_input, grad_output):
            nonlocal hook_called
            hook_called = True
            return None  # 不修改
        
        module.register_full_backward_hook(backward_hook_no_modify)
        input_data = rm.ones(2, 10, requires_grad=True)
        output = module(input_data)
        output.sum().backward()
        
        passed = hook_called
        stats.add_result("反向钩子 - 单输入模块 + 不修改", passed, 
                        f"钩子调用: {hook_called}")
        
        # 测试12: 单输入模块 + 修改输入梯度（×3）
        print("测试反向钩子 - 单输入模块 + 修改输入梯度（×3）...")
        module = SimpleLinearModule(10, 5)
        hook_called = False
        
        def backward_hook_modify_x3(m, grad_input, grad_output):
            nonlocal hook_called
            hook_called = True
            return tuple(g * 3 if g is not None else None for g in grad_input)
        
        module.register_full_backward_hook(backward_hook_modify_x3)
        input_data = rm.ones(2, 10, requires_grad=True)
        output = module(input_data)
        output.sum().backward()
        
        expected_grad = rm.ones((2, 5)) @ module.weight.data * 3
        grad_correct = tensor_allclose(input_data.grad, expected_grad)
        passed = hook_called and grad_correct
        stats.add_result("反向钩子 - 单输入模块 + 修改输入梯度（×3）", passed, 
                        f"钩子调用: {hook_called}, 梯度正确: {grad_correct}")
        
        # 测试13: 多输入模块 + 部分修改
        print("测试反向钩子 - 多输入模块 + 部分修改...")
        module = TwoInputModule()
        hook_called = False
        grad_modified = False
        
        def backward_hook_partial_modify(m, grad_input, grad_output):
            nonlocal hook_called, grad_modified
            hook_called = True
            # 只修改第一个输入梯度
            modified = (grad_input[0] * 2,) + grad_input[1:] if len(grad_input) > 1 else grad_input
            grad_modified = True
            return modified
        
        module.register_full_backward_hook(backward_hook_partial_modify)
        x = rm.ones(2, 10, requires_grad=True)
        y = rm.ones(2, 10, requires_grad=True)
        output = module(x, y)
        output.sum().backward()
        
        # TwoInputModule: output = (x + y) @ weight, weight shape is (10, 5)
        # grad_input = grad_output @ weight.T = ones(2, 5) @ (10, 5).T = ones(2, 5) @ (5, 10) = ones(2, 10)
        expected_x_grad = rm.ones((2, 5)) @ module.weight.data.T * 2
        expected_y_grad = rm.ones((2, 5)) @ module.weight.data.T
        x_grad_correct = tensor_allclose(x.grad, expected_x_grad)
        y_grad_correct = tensor_allclose(y.grad, expected_y_grad)
        passed = hook_called and grad_modified and x_grad_correct and y_grad_correct
        stats.add_result("反向钩子 - 多输入模块 + 部分修改", passed, 
                        f"钩子调用: {hook_called}, x梯度正确: {x_grad_correct}, y梯度正确: {y_grad_correct}")
        
        # 测试14: 多输入模块 + 全部修改
        print("测试反向钩子 - 多输入模块 + 全部修改...")
        module = TwoInputModule()
        hook_called = False
        grad_modified = False
        
        def backward_hook_all_modify(m, grad_input, grad_output):
            nonlocal hook_called, grad_modified
            hook_called = True
            # 修改所有输入梯度
            modified = tuple(g * 2 if g is not None else None for g in grad_input)
            grad_modified = True
            return modified
        
        module.register_full_backward_hook(backward_hook_all_modify)
        x = rm.ones(2, 10, requires_grad=True)
        y = rm.ones(2, 10, requires_grad=True)
        output = module(x, y)
        output.sum().backward()
        
        # TwoInputModule: output = (x + y) @ weight, weight shape is (10, 5)
        # grad_input = grad_output @ weight.T = ones(2, 5) @ (10, 5).T = ones(2, 5) @ (5, 10) = ones(2, 10)
        expected_grad = rm.ones((2, 5)) @ module.weight.data.T * 2
        x_grad_correct = tensor_allclose(x.grad, expected_grad)
        y_grad_correct = tensor_allclose(y.grad, expected_grad)
        passed = hook_called and grad_modified and x_grad_correct and y_grad_correct
        stats.add_result("反向钩子 - 多输入模块 + 全部修改", passed, 
                        f"钩子调用: {hook_called}, x梯度正确: {x_grad_correct}, y梯度正确: {y_grad_correct}")
        
        # 测试15: 有参数模块 + 输入无需梯度但有参数
        print("测试反向钩子 - 有参数模块 + 输入无需梯度...")
        module = SimpleLinearModule(10, 5)
        hook_called = False
        
        def backward_hook_with_param(m, grad_input, grad_output):
            nonlocal hook_called
            hook_called = True
            return grad_input
        
        module.register_full_backward_hook(backward_hook_with_param)
        input_data = rm.ones(2, 10, requires_grad=False)
        output = module(input_data)
        output.sum().backward()
        
        passed = hook_called
        stats.add_result("反向钩子 - 有参数模块 + 输入无需梯度", passed, 
                        f"钩子调用: {hook_called}")
        
        # 测试16: 单输入 requires_grad=True
        print("测试反向钩子 - 单输入 requires_grad=True...")
        module = SimpleLinearModule(10, 5)
        hook_called = False
        grad_input_received = None
        grad_output_received = None
        
        def backward_hook_single_input(m, grad_input, grad_output):
            nonlocal hook_called, grad_input_received, grad_output_received
            hook_called = True
            grad_input_received = grad_input
            grad_output_received = grad_output
            return grad_input
        
        module.register_full_backward_hook(backward_hook_single_input)
        x = rm.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]], requires_grad=True)
        y = module(x)
        loss = y.sum()
        loss.backward()
        
        passed = (hook_called and grad_input_received is not None and 
                 grad_output_received is not None and
                 isinstance(grad_input_received, tuple) and
                 isinstance(grad_output_received, tuple) and
                 len(grad_input_received) == 1 and
                 len(grad_output_received) == 1)
        stats.add_result("反向钩子 - 单输入 requires_grad=True", passed, 
                        f"钩子被调用: {hook_called}, 收到grad_input: {grad_input_received is not None}")
        
        # 测试17: 多输入，部分 requires_grad=False
        print("测试反向钩子 - 多输入，部分 requires_grad=False...")
        module = TwoInputModule()
        hook_called = False
        grad_input_length = None
        grad_input_none_count = 0
        
        def backward_hook_partial_grad(m, grad_input, grad_output):
            nonlocal hook_called, grad_input_length, grad_input_none_count
            hook_called = True
            grad_input_length = len(grad_input)
            grad_input_none_count = sum(1 for g in grad_input if g is None)
            return grad_input
        
        module.register_full_backward_hook(backward_hook_partial_grad)
        x = rm.ones((2, 10), requires_grad=True)
        y = rm.ones((2, 10), requires_grad=False)
        output = module(x, y)
        output.sum().backward()
        
        # 验证：grad_input长度为2，且没有None（与PyTorch行为一致）
        passed = hook_called and grad_input_length == 2 and grad_input_none_count == 0
        stats.add_result("反向钩子 - 多输入，部分 requires_grad=False", passed, 
                        f"钩子被调用: {hook_called}, grad_input长度: {grad_input_length}, None数量: {grad_input_none_count}")
        
        # 测试18: 多输入，全部 requires_grad=False
        print("测试反向钩子 - 多输入，全部 requires_grad=False...")
        module = TwoInputModule()
        hook_called = False
        grad_input_all_none = False
        
        def backward_hook_all_no_grad(m, grad_input, grad_output):
            nonlocal hook_called, grad_input_all_none
            hook_called = True
            grad_input_all_none = all(g is None for g in grad_input)
            return grad_input
        
        module.register_full_backward_hook(backward_hook_all_no_grad)
        x = rm.ones((2, 10), requires_grad=False)
        y = rm.ones((2, 10), requires_grad=False)
        output = module(x, y)
        output.sum().backward()
        
        # 与PyTorch行为一致：当所有输入requires_grad=False且模块无参数时，钩子被调用且grad_input为None
        passed = hook_called and grad_input_all_none
        stats.add_result("反向钩子 - 多输入，全部 requires_grad=False", passed, 
                        f"钩子被调用: {hook_called}, 所有grad_input为None: {grad_input_all_none}")
        
    finally:
        stats.end_function()


def test_combined_hooks(stats):
    """测试组合钩子（同时注册反向预处理+反向钩子）"""
    stats.start_function("组合钩子")
    
    try:
        # 测试1: 同时注册两种钩子
        print("测试同时注册反向预处理钩子和反向钩子...")
        module = SimpleLinearModule(10, 5)
        pre_hook_called = False
        backward_hook_called = False
        
        def pre_hook(m, grad_output):
            nonlocal pre_hook_called
            pre_hook_called = True
            return grad_output
        
        def backward_hook(m, grad_input, grad_output):
            nonlocal backward_hook_called
            backward_hook_called = True
            return grad_input
        
        module.register_full_backward_pre_hook(pre_hook)
        module.register_full_backward_hook(backward_hook)
        
        input_data = rm.ones(2, 10, requires_grad=True)
        output = module(input_data)
        output.sum().backward()
        
        passed = pre_hook_called and backward_hook_called
        stats.add_result("同时注册两种钩子", passed, 
                        f"预处理钩子调用: {pre_hook_called}, 反向钩子调用: {backward_hook_called}")
        
        # 测试2: 两种钩子都修改梯度
        print("测试两种钩子都修改梯度...")
        module = SimpleLinearModule(10, 5)
        pre_hook_called = False
        backward_hook_called = False
        
        def pre_hook_modify(m, grad_output):
            nonlocal pre_hook_called
            pre_hook_called = True
            return tuple(g * 2 if g is not None else None for g in grad_output)
        
        def backward_hook_modify(m, grad_input, grad_output):
            nonlocal backward_hook_called
            backward_hook_called = True
            return tuple(g * 3 if g is not None else None for g in grad_input)
        
        module.register_full_backward_pre_hook(pre_hook_modify)
        module.register_full_backward_hook(backward_hook_modify)
        
        input_data = rm.ones(2, 10, requires_grad=True)
        output = module(input_data)
        output.sum().backward()
        
        # 预处理钩子×2，反向钩子×3，总共×6
        expected_grad = rm.ones((2, 5)) @ module.weight.data * 6
        grad_correct = tensor_allclose(input_data.grad, expected_grad)
        
        passed = pre_hook_called and backward_hook_called and grad_correct
        stats.add_result("两种钩子都修改梯度", passed, 
                        f"预处理钩子调用: {pre_hook_called}, 反向钩子调用: {backward_hook_called}, 梯度正确: {grad_correct}")
        
        # 测试3: 多输入多输出 + 输入透传
        print("测试多输入多输出 + 输入透传...")
        module = IdentityPassThroughModule()
        pre_hook_called = False
        backward_hook_called = False
        pre_hook_has_none = False
        backward_hook_has_none = False
        
        def pre_hook_identity(m, grad_output):
            nonlocal pre_hook_called, pre_hook_has_none
            pre_hook_called = True
            pre_hook_has_none = any(g is None for g in grad_output)
            return grad_output
        
        def backward_hook_identity(m, grad_input, grad_output):
            nonlocal backward_hook_called, backward_hook_has_none
            backward_hook_called = True
            backward_hook_has_none = any(g is None for g in grad_output)
            return grad_input
        
        module.register_full_backward_pre_hook(pre_hook_identity)
        module.register_full_backward_hook(backward_hook_identity)
        
        x1 = rm.ones(2, 10, requires_grad=True)
        x2 = rm.ones(2, 10, requires_grad=True)
        out1, out2 = module(x1, x2)
        loss = out2.sum()
        loss.backward()
        
        passed = (pre_hook_called and backward_hook_called and 
                 pre_hook_has_none and backward_hook_has_none and
                 x2.grad is not None)
        stats.add_result("多输入多输出 + 输入透传", passed, 
                        f"预处理钩子调用: {pre_hook_called}, 反向钩子调用: {backward_hook_called}, "
                        f"预处理钩子收到None: {pre_hook_has_none}, 反向钩子收到None: {backward_hook_has_none}")
        
        # 测试4: 预处理修改输出梯度 + 反向钩子不修改
        print("测试预处理修改输出梯度 + 反向钩子不修改...")
        module = SimpleLinearModule(10, 5)
        pre_hook_called = False
        backward_hook_called = False
        
        def pre_hook_modify_out(m, grad_output):
            nonlocal pre_hook_called
            pre_hook_called = True
            return tuple(g * 2 for g in grad_output)
        
        def backward_hook_no_modify(m, grad_input, grad_output):
            nonlocal backward_hook_called
            backward_hook_called = True
            return None  # 不修改
        
        module.register_full_backward_pre_hook(pre_hook_modify_out)
        module.register_full_backward_hook(backward_hook_no_modify)
        
        input_data = rm.ones((2, 10), requires_grad=True)
        output = module(input_data)
        loss = output.sum()
        loss.backward()
        
        # 验证梯度：预处理钩子x2影响反向传播
        expected_grad = rm.full_like(input_data, 10.0)  # weight.sum() * 2 (pre-hook修改)
        grad_correct = rm.allclose(input_data.grad, expected_grad)
        
        passed = pre_hook_called and backward_hook_called and grad_correct
        stats.add_result("预处理修改输出梯度 + 反向钩子不修改", passed, 
                        f"预处理钩子调用: {pre_hook_called}, 反向钩子调用: {backward_hook_called}, 梯度正确: {grad_correct}")
        
        # 测试5: 预处理不修改 + 反向钩子修改输入梯度
        print("测试预处理不修改 + 反向钩子修改输入梯度...")
        module = SimpleLinearModule(10, 5)
        pre_hook_called = False
        backward_hook_called = False
        
        def pre_hook_no_modify2(m, grad_output):
            nonlocal pre_hook_called
            pre_hook_called = True
            return None  # 不修改
        
        def backward_hook_modify_in(m, grad_input, grad_output):
            nonlocal backward_hook_called
            backward_hook_called = True
            return tuple(g * 3 if g is not None else None for g in grad_input)
        
        module.register_full_backward_pre_hook(pre_hook_no_modify2)
        module.register_full_backward_hook(backward_hook_modify_in)
        
        input_data = rm.ones((2, 10), requires_grad=True)
        output = module(input_data)
        loss = output.sum()
        loss.backward()
        
        # 验证梯度：反向钩子x3修改
        expected_grad = rm.full_like(input_data, 15.0)  # weight.sum() * 3 (backward hook修改)
        grad_correct = rm.allclose(input_data.grad, expected_grad)
        
        passed = pre_hook_called and backward_hook_called and grad_correct
        stats.add_result("预处理不修改 + 反向钩子修改输入梯度", passed, 
                        f"预处理钩子调用: {pre_hook_called}, 反向钩子调用: {backward_hook_called}, 梯度正确: {grad_correct}")
        
        # 测试6: 预处理修改输出梯度 + 反向钩子修改输入梯度
        print("测试预处理修改输出梯度 + 反向钩子修改输入梯度...")
        module = SimpleLinearModule(10, 5)
        pre_hook_called = False
        backward_hook_called = False
        
        def pre_hook_modify_out2(m, grad_output):
            nonlocal pre_hook_called
            pre_hook_called = True
            return tuple(g * 2 for g in grad_output)
        
        def backward_hook_modify_in2(m, grad_input, grad_output):
            nonlocal backward_hook_called
            backward_hook_called = True
            return tuple(g * 3 if g is not None else None for g in grad_input)
        
        module.register_full_backward_pre_hook(pre_hook_modify_out2)
        module.register_full_backward_hook(backward_hook_modify_in2)
        
        input_data = rm.ones((2, 10), requires_grad=True)
        output = module(input_data)
        loss = output.sum()
        loss.backward()
        
        # 验证梯度：预处理x2 * 反向x3 = x6
        expected_grad = rm.full_like(input_data, 30.0)  # weight.sum() * 2 * 3
        grad_correct = rm.allclose(input_data.grad, expected_grad)
        
        passed = pre_hook_called and backward_hook_called and grad_correct
        stats.add_result("预处理修改输出梯度 + 反向钩子修改输入梯度", passed, 
                        f"预处理钩子调用: {pre_hook_called}, 反向钩子调用: {backward_hook_called}, 梯度正确: {grad_correct}")
        
        # 测试7: 多输出模块 + 两种钩子都修改
        print("测试多输出模块 + 两种钩子都修改...")
        
        class MultiOutputBothHooksModule(Module):
            def __init__(self):
                super().__init__()
                self.weight1 = Parameter(rm.ones((10, 5)))
                self.weight2 = Parameter(rm.ones((10, 5)))
            
            def forward(self, x):
                out1 = x @ self.weight1
                out2 = x @ self.weight2
                return out1, out2
        
        module = MultiOutputBothHooksModule()
        pre_hook_called = False
        backward_hook_called = False
        
        def pre_hook_multi(m, grad_output):
            nonlocal pre_hook_called
            pre_hook_called = True
            return tuple(g * 2 for g in grad_output)
        
        def backward_hook_multi(m, grad_input, grad_output):
            nonlocal backward_hook_called
            backward_hook_called = True
            return tuple(g * 3 if g is not None else None for g in grad_input)
        
        module.register_full_backward_pre_hook(pre_hook_multi)
        module.register_full_backward_hook(backward_hook_multi)
        
        input_data = rm.ones((2, 10), requires_grad=True)
        out1, out2 = module(input_data)
        loss = out1.sum() + out2.sum()
        loss.backward()
        
        # 验证梯度：两个输出都参与，预处理x2 * 反向x3 = x6
        # 每个输出贡献 weight.sum() = 5，两个输出共10，再x6 = 60
        expected_grad = rm.full_like(input_data, 60.0)
        grad_correct = rm.allclose(input_data.grad, expected_grad)
        
        passed = pre_hook_called and backward_hook_called and grad_correct
        stats.add_result("多输出模块 + 两种钩子都修改", passed, 
                        f"预处理钩子调用: {pre_hook_called}, 反向钩子调用: {backward_hook_called}, 梯度正确: {grad_correct}")
        
    finally:
        stats.end_function()


def test_complex_scenarios(stats):
    """测试复杂场景"""
    stats.start_function("复杂场景")
    
    try:
        # 测试1: 3输入3输出，部分输出不参与损失
        print("测试复杂场景 - 3输入3输出部分输出不参与损失...")
        module = ThreeInputThreeOutputModule()
        pre_hook_called = False
        backward_hook_called = False
        
        def backward_pre_hook(module, grad_output):
            nonlocal pre_hook_called
            pre_hook_called = True
            return grad_output
        
        def backward_hook(module, grad_input, grad_output):
            nonlocal backward_hook_called
            backward_hook_called = True
            return None
        
        module.register_full_backward_pre_hook(backward_pre_hook)
        module.register_full_backward_hook(backward_hook)
        
        x1 = rm.tensor([1., 2.], requires_grad=True)
        x2 = rm.tensor([4., 5., 6.], requires_grad=True)
        x3 = rm.tensor([7., 8., 9., 10.], requires_grad=True)
        out1, out2, out3 = module(x1, x2, x3)
        loss = out2.sum() + out3.sum()
        loss.backward()
        
        passed = (pre_hook_called and backward_hook_called and 
                 x1.grad is not None and x2.grad is not None and x3.grad is not None)
        stats.add_result("复杂场景 - 3输入3输出部分输出不参与损失", passed, 
                        f"钩子调用: {pre_hook_called and backward_hook_called}, "
                        f"x1.grad: {x1.grad is not None}, x2.grad: {x2.grad is not None}, x3.grad: {x3.grad is not None}")
        
        # 测试2: 多次前向传播
        print("测试多次前向传播 + 反向钩子...")
        module = SimpleLinearModule(10, 5)
        hook_call_count = 0
        
        def backward_hook_count(module, grad_input, grad_output):
            nonlocal hook_call_count
            hook_call_count += 1
            return grad_input
        
        module.register_full_backward_hook(backward_hook_count)
        
        x1 = rm.ones(2, 10, requires_grad=True)
        x2 = rm.ones(2, 10, requires_grad=True)
        x3 = rm.ones(2, 10, requires_grad=True)
        y1 = module(x1)
        y2 = module(x2)
        y3 = module(x3)
        
        loss = y1.sum() + y2.sum() + y3.sum()
        loss.backward()
        
        passed = hook_call_count == 3
        stats.add_result("多次前向传播 + 反向钩子", passed, 
                        f"钩子调用次数: {hook_call_count} (预期3)")
        
        # 测试3: 缓存清理
        print("测试单次前向传播后缓存清理...")
        module = SimpleLinearModule(10, 5)
        
        def backward_hook_cache(module, grad_input, grad_output):
            return grad_input
        
        module.register_full_backward_hook(backward_hook_cache)
        
        input_data = rm.ones(2, 10, requires_grad=True)
        output = module(input_data)
        
        cache_before = len(module._backward_hook_cache)
        output.sum().backward()
        cache_after = len(module._backward_hook_cache)
        
        passed = cache_before > 0 and cache_after == 0
        stats.add_result("单次前向传播后缓存清理", passed, 
                        f"前向后: {cache_before}, 反向后: {cache_after}")
        
        # 测试4: 复杂嵌套模块的钩子调用顺序
        print("测试复杂嵌套模块的钩子调用顺序...")
        
        class MiddleModule(Module):
            def __init__(self):
                super().__init__()
                self.inner = InnerModule()
            
            def forward(self, x):
                return self.inner(x) * 2
        
        class OuterComplexModule(Module):
            def __init__(self):
                super().__init__()
                self.middle = MiddleModule()
            
            def forward(self, x):
                return self.middle(x) + 1
        
        module = OuterComplexModule()
        call_order = []
        
        def inner_hook(module, grad_input, grad_output):
            call_order.append("inner")
            return grad_input
        
        def middle_hook(module, grad_input, grad_output):
            call_order.append("middle")
            return grad_input
        
        def outer_hook(module, grad_input, grad_output):
            call_order.append("outer")
            return grad_input
        
        module.middle.inner.register_full_backward_hook(inner_hook)
        module.middle.register_full_backward_hook(middle_hook)
        module.register_full_backward_hook(outer_hook)
        
        input_data = rm.randn(2, 10, requires_grad=True)
        output = module(input_data)
        loss = output.sum()
        loss.backward()
        
        all_hooks_called = len(call_order) == 3 and set(call_order) == {"outer", "middle", "inner"}
        passed = all_hooks_called
        stats.add_result("复杂嵌套模块的钩子调用顺序", passed, 
                        f"调用顺序: {call_order}, 所有钩子被调用: {all_hooks_called}")
        
        # 测试5: 参数共享模块的钩子行为
        print("测试参数共享模块的钩子行为...")
        
        class SharedWeightModule(Module):
            def __init__(self):
                super().__init__()
                self.weight = Parameter(rm.ones((5, 10)))
            
            def forward(self, x):
                return x @ self.weight.T
        
        shared = SharedWeightModule()
        
        class MainSharedModule(Module):
            def __init__(self):
                super().__init__()
                self.shared1 = shared
                self.shared2 = shared
            
            def forward(self, x):
                out1 = self.shared1(x)
                out2 = self.shared2(x)
                return out1 + out2
        
        module = MainSharedModule()
        hook1_called = False
        hook2_called = False
        
        def hook1(module, grad_input, grad_output):
            nonlocal hook1_called
            hook1_called = True
            return grad_input
        
        def hook2(module, grad_input, grad_output):
            nonlocal hook2_called
            hook2_called = True
            return grad_input
        
        module.shared1.register_full_backward_hook(hook1)
        module.shared2.register_full_backward_hook(hook2)
        
        input_data = rm.randn(2, 10, requires_grad=True)
        output = module(input_data)
        loss = output.sum()
        loss.backward()
        
        passed = hook1_called and hook2_called
        stats.add_result("参数共享模块的钩子行为", passed, 
                        f"钩子1被调用: {hook1_called}, 钩子2被调用: {hook2_called}")
        
        # 测试6: 反向钩子验证grad_input值
        print("测试反向钩子 - 验证grad_input值...")
        module = SimpleLinearModule(10, 5)
        hook_called = False
        grad_input_values = None
        
        def backward_hook_verify_grad(module, grad_input, grad_output):
            nonlocal hook_called, grad_input_values
            hook_called = True
            grad_input_values = grad_input
            return grad_input
        
        module.register_full_backward_hook(backward_hook_verify_grad)
        
        input_data = rm.ones((2, 10), requires_grad=True)
        output = module(input_data)
        loss = output.sum()
        loss.backward()
        
        expected_grad_input = rm.ones((2, 5)) @ module.weight.data
        grad_input_correct = (grad_input_values is not None and 
                             len(grad_input_values) == 1 and
                             tensor_allclose(grad_input_values[0], expected_grad_input))
        passed = hook_called and grad_input_correct
        stats.add_result("反向钩子 - 验证grad_input值", passed, 
                        f"钩子调用: {hook_called}, grad_input正确: {grad_input_correct}")
        
        # 测试7: 反向预处理钩子验证grad_output值
        print("测试反向预处理钩子 - 验证grad_output值...")
        module = MultiOutputModule()
        hook_called = False
        grad_output_values = None
        
        def pre_hook_verify_grad(module, grad_output):
            nonlocal hook_called, grad_output_values
            hook_called = True
            grad_output_values = grad_output
            return grad_output
        
        module.register_full_backward_pre_hook(pre_hook_verify_grad)
        
        input_data = rm.ones((2, 10), requires_grad=True)
        out1, out2 = module(input_data)
        (out1 + out2).sum().backward()
        
        # 对于sum()损失，grad_output应该是全1
        grad_output_correct = (grad_output_values is not None and 
                              len(grad_output_values) == 2 and
                              tensor_allclose(grad_output_values[0], rm.ones((2, 5))) and
                              tensor_allclose(grad_output_values[1], rm.ones((2, 5))))
        passed = hook_called and grad_output_correct
        stats.add_result("反向预处理钩子 - 验证grad_output值", passed, 
                        f"钩子调用: {hook_called}, grad_output正确: {grad_output_correct}")
        
        # 测试8: 多个反向预处理钩子级联调用
        print("测试多个反向预处理钩子级联调用...")
        module = SimpleLinearModule(10, 5)
        hook_calls = []
        hook2_received_correct = False
        hook3_received_correct = False
        
        def hook1_cascade(module, grad_output):
            hook_calls.append('hook1')
            modified = tuple(g * 2 for g in grad_output)
            return modified
        
        def hook2_cascade(module, grad_output):
            nonlocal hook2_received_correct
            hook_calls.append('hook2')
            hook2_received_correct = rm.allclose(grad_output[0], rm.full_like(grad_output[0], 2.0))
            modified = tuple(g + 1 for g in grad_output)
            return modified
        
        def hook3_cascade(module, grad_output):
            nonlocal hook3_received_correct
            hook_calls.append('hook3')
            hook3_received_correct = rm.allclose(grad_output[0], rm.full_like(grad_output[0], 3.0))
            return None
        
        module.register_full_backward_pre_hook(hook1_cascade)
        module.register_full_backward_pre_hook(hook2_cascade)
        module.register_full_backward_pre_hook(hook3_cascade)
        
        input_data = rm.ones((2, 10), requires_grad=True)
        output = module(input_data)
        loss = output.sum()
        loss.backward()
        
        correct_order = hook_calls == ['hook1', 'hook2', 'hook3']
        expected_grad = rm.full_like(input_data, 15.0)  # 5 * 3 = 15
        grad_correct = rm.allclose(input_data.grad, expected_grad)
        
        passed = correct_order and hook2_received_correct and hook3_received_correct and grad_correct
        stats.add_result("多个反向预处理钩子级联调用", passed, 
                        f"调用顺序: {hook_calls}, hook2收到修改后梯度: {hook2_received_correct}, "
                        f"hook3收到修改后梯度: {hook3_received_correct}, 梯度正确: {grad_correct}")
        
        # 测试9: 多个反向预处理钩子与反向钩子级联
        print("测试多个反向预处理钩子与反向钩子级联...")
        module = SimpleLinearModule(5, 3)
        pre_hook1_grad = None
        pre_hook2_grad = None
        pre_hook3_grad = None
        backward_hook_grad = None
        
        def pre_hook1_verify(module, grad_output):
            nonlocal pre_hook1_grad
            pre_hook1_grad = grad_output[0].clone() if grad_output[0] is not None else None
            modified = tuple(g * 2 if g is not None else None for g in grad_output)
            return modified
        
        def pre_hook2_verify(module, grad_output):
            nonlocal pre_hook2_grad
            pre_hook2_grad = grad_output[0].clone() if grad_output[0] is not None else None
            modified = tuple(g + 3 if g is not None else None for g in grad_output)
            return modified
        
        def pre_hook3_verify(module, grad_output):
            nonlocal pre_hook3_grad
            pre_hook3_grad = grad_output[0].clone() if grad_output[0] is not None else None
            modified = tuple(g - 1 if g is not None else None for g in grad_output)
            return modified
        
        def backward_hook_verify(module, grad_input, grad_output):
            nonlocal backward_hook_grad
            backward_hook_grad = grad_output[0].clone() if grad_output[0] is not None else None
            return None
        
        module.register_full_backward_pre_hook(pre_hook1_verify)
        module.register_full_backward_pre_hook(pre_hook2_verify)
        module.register_full_backward_pre_hook(pre_hook3_verify)
        module.register_full_backward_hook(backward_hook_verify)
        
        input_data = rm.ones((2, 5), requires_grad=True)
        output = module(input_data)
        loss = output.sum()
        loss.backward()
        
        pre_hook1_received_correct = rm.allclose(pre_hook1_grad, rm.ones_like(pre_hook1_grad))
        pre_hook2_received_correct = rm.allclose(pre_hook2_grad, rm.full_like(pre_hook2_grad, 2.0))
        pre_hook3_received_correct = rm.allclose(pre_hook3_grad, rm.full_like(pre_hook3_grad, 5.0))
        backward_hook_received_correct = rm.allclose(backward_hook_grad, rm.full_like(backward_hook_grad, 4.0))
        
        passed = (pre_hook1_received_correct and 
                 pre_hook2_received_correct and 
                 pre_hook3_received_correct and 
                 backward_hook_received_correct)
        stats.add_result("多个反向预处理钩子与反向钩子级联", passed,
                        f"pre_hook1收到1: {pre_hook1_received_correct}, "
                        f"pre_hook2收到2: {pre_hook2_received_correct}, "
                        f"pre_hook3收到5: {pre_hook3_received_correct}, "
                        f"backward_hook收到4: {backward_hook_received_correct}")
        
        # 测试10: 多输出模块 - 部分输出requires_grad=False
        print("测试多输出模块 - 部分输出requires_grad=False...")
        
        class MixedOutputModule(Module):
            def __init__(self):
                super().__init__()
                self.weight1 = Parameter(rm.ones((10, 5)))
                self.weight2 = Parameter(rm.ones((10, 5)))
            
            def forward(self, x):
                y1 = x @ self.weight1
                y2 = x @ self.weight2
                y2 = y2.detach()
                return y1, y2
        
        module = MixedOutputModule()
        hook_called = False
        grad_output_count = 0
        
        def backward_pre_hook_mixed(module, grad_output):
            nonlocal hook_called, grad_output_count
            hook_called = True
            grad_output_count = len(grad_output)
            return grad_output
        
        module.register_full_backward_pre_hook(backward_pre_hook_mixed)
        
        input_data = rm.ones((2, 10), requires_grad=True)
        y1, y2 = module(input_data)
        y2_requires_grad = y2.requires_grad
        loss = y1.sum()
        loss.backward()
        
        passed = hook_called and grad_output_count == 1 and not y2_requires_grad
        stats.add_result("多输出模块 - 部分输出requires_grad=False", passed,
                        f"钩子被调用: {hook_called}, 收到梯度数量: {grad_output_count}, y2.requires_grad: {y2_requires_grad}")
        
        # 测试11: 多次前向传播的多输出模块
        print("测试多次前向传播的多输出模块...")
        module = MultiOutputModule()
        hook_call_count = 0
        hook_outputs = []
        
        def backward_pre_hook_multi_out(module, grad_output):
            nonlocal hook_call_count, hook_outputs
            hook_call_count += 1
            hook_outputs.append(len(grad_output))
            return grad_output
        
        module.register_full_backward_pre_hook(backward_pre_hook_multi_out)
        
        x1 = rm.ones((2, 10), requires_grad=True)
        y1_a, y1_b = module(x1)
        
        x2 = rm.ones((2, 10), requires_grad=True)
        y2_a, y2_b = module(x2)
        
        loss = y1_a.sum() + y1_b.sum() + y2_a.sum() + y2_b.sum()
        loss.backward()
        
        passed = hook_call_count == 2 and all(count == 2 for count in hook_outputs)
        stats.add_result("多次前向传播的多输出模块", passed,
                        f"钩子调用次数: {hook_call_count}, 每次输出数量: {hook_outputs}")
        
        # 测试12: 多次前向传播后缓存清理
        print("测试多次前向传播后缓存清理...")
        module = SimpleLinearModule(10, 5)
        hook_call_count = 0
        
        def backward_hook_multi_cache(module, grad_input, grad_output):
            nonlocal hook_call_count
            hook_call_count += 1
            return grad_input
        
        module.register_full_backward_hook(backward_hook_multi_cache)
        
        input_data = rm.ones((2, 10), requires_grad=True)
        output1 = module(input_data)
        cache_after_1st = len(module._backward_hook_cache)
        
        output2 = module(input_data)
        cache_after_2nd = len(module._backward_hook_cache)
        
        output3 = module(input_data)
        cache_after_3rd = len(module._backward_hook_cache)
        
        loss2 = output2.sum()
        loss2.backward()
        cache_after_2nd_backward = len(module._backward_hook_cache)
        
        loss1 = output1.sum()
        loss1.backward()
        cache_after_1st_backward = len(module._backward_hook_cache)
        
        loss3 = output3.sum()
        loss3.backward()
        cache_after_3rd_backward = len(module._backward_hook_cache)
        
        passed = (cache_after_3rd == 3 and
                 cache_after_2nd_backward == 2 and
                 cache_after_1st_backward == 1 and
                 cache_after_3rd_backward == 0 and
                 hook_call_count == 3)
        stats.add_result("多次前向传播后缓存清理", passed,
                        f"3次前向后: {cache_after_3rd}, 2nd反向后: {cache_after_2nd_backward}, "
                        f"1st反向后: {cache_after_1st_backward}, 3rd反向后: {cache_after_3rd_backward}, "
                        f"钩子调用: {hook_call_count}")
        
        # 测试13: 多输出模块缓存清理
        print("测试多输出模块缓存清理...")
        module = MultiOutputModule()
        hook_called = False
        
        def backward_hook_multi_out_cache(module, grad_input, grad_output):
            nonlocal hook_called
            hook_called = True
            return grad_input
        
        module.register_full_backward_hook(backward_hook_multi_out_cache)
        
        input_data = rm.ones((2, 10), requires_grad=True)
        out1, out2 = module(input_data)
        cache_after_forward = len(module._backward_hook_cache)
        
        loss = out1.sum() + out2.sum()
        loss.backward()
        cache_after_backward = len(module._backward_hook_cache)
        
        passed = cache_after_forward == 2 and cache_after_backward == 0 and hook_called
        stats.add_result("多输出模块缓存清理", passed,
                        f"前向后: {cache_after_forward}, 反向后: {cache_after_backward}, 钩子调用: {hook_called}")
        
        # 测试14: 嵌套模块缓存清理
        print("测试嵌套模块缓存清理...")
        
        class InnerCacheModule(Module):
            def __init__(self):
                super().__init__()
                self.weight = Parameter(rm.ones((5, 10)))
            
            def forward(self, x):
                return x @ self.weight.T
        
        class OuterCacheModule(Module):
            def __init__(self):
                super().__init__()
                self.inner = InnerCacheModule()
            
            def forward(self, x):
                return self.inner(x)
        
        outer = OuterCacheModule()
        inner_hook_called = False
        outer_hook_called = False
        
        def inner_hook_cache(module, grad_input, grad_output):
            nonlocal inner_hook_called
            inner_hook_called = True
            return grad_input
        
        def outer_hook_cache(module, grad_input, grad_output):
            nonlocal outer_hook_called
            outer_hook_called = True
            return grad_input
        
        outer.inner.register_full_backward_hook(inner_hook_cache)
        outer.register_full_backward_hook(outer_hook_cache)
        
        input_data = rm.ones((2, 10), requires_grad=True)
        output = outer(input_data)
        inner_cache_before = len(outer.inner._backward_hook_cache)
        outer_cache_before = len(outer._backward_hook_cache)
        
        loss = output.sum()
        loss.backward()
        inner_cache_after = len(outer.inner._backward_hook_cache)
        outer_cache_after = len(outer._backward_hook_cache)
        
        passed = (inner_cache_before > 0 and
                 outer_cache_before > 0 and
                 inner_cache_after == 0 and
                 outer_cache_after == 0 and
                 inner_hook_called and
                 outer_hook_called)
        stats.add_result("嵌套模块缓存清理", passed,
                        f"内部模块: 前{inner_cache_before}->后{inner_cache_after}, "
                        f"外部模块: 前{outer_cache_before}->后{outer_cache_after}, "
                        f"钩子调用: 内{inner_hook_called}/外{outer_hook_called}")
        
        # 测试15: 无参数模块缓存清理
        print("测试无参数模块缓存清理...")
        
        class SingleInputNoParamModule(Module):
            def forward(self, x):
                return x * 2
        
        module = SingleInputNoParamModule()
        hook_called = False
        
        def backward_hook_no_param_cache(module, grad_input, grad_output):
            nonlocal hook_called
            hook_called = True
            return grad_input
        
        module.register_full_backward_hook(backward_hook_no_param_cache)
        
        input_data = rm.ones((2, 10), requires_grad=True)
        output = module(input_data)
        cache_after_forward = len(module._backward_hook_cache)
        
        loss = output.sum()
        loss.backward()
        cache_after_backward = len(module._backward_hook_cache)
        
        passed = cache_after_forward > 0 and cache_after_backward == 0 and hook_called
        stats.add_result("无参数模块缓存清理", passed,
                        f"前向后: {cache_after_forward}, 反向后: {cache_after_backward}, 钩子调用: {hook_called}")
        
        # 测试16: 钩子抛出异常
        print("测试钩子抛出异常...")
        module = SimpleLinearModule(10, 5)
        exception_raised = False
        exception_message_correct = False
        
        def pre_hook_raise(m, grad_output):
            raise ValueError("测试异常：预处理钩子")
        
        module.register_full_backward_pre_hook(pre_hook_raise)
        
        x = rm.ones((2, 10), requires_grad=True)
        out = module(x)
        loss = out.sum()
        try:
            loss.backward()
        except ValueError as e:
            exception_raised = True
            exception_message_correct = "测试异常" in str(e)
        
        passed = exception_raised and exception_message_correct
        stats.add_result("钩子抛出异常", passed,
                        f"异常抛出: {exception_raised}, 异常信息正确: {exception_message_correct}")
        
        # 测试17: 多次前向传播 + 反向钩子 + 分别backward
        print("测试多次前向传播 + 反向钩子 + 分别backward...")
        module = SimpleLinearModule(10, 5)
        hook_called = 0
        
        def backward_hook_separate(module, grad_input, grad_output):
            nonlocal hook_called
            hook_called += 1
            return grad_input
        
        module.register_full_backward_hook(backward_hook_separate)
        
        input_data = rm.ones((2, 10), requires_grad=True)
        output1 = module(input_data)
        output2 = module(input_data)
        output3 = module(input_data)
        
        loss2 = output2.sum()
        loss2.backward()
        
        loss1 = output1.sum()
        loss1.backward()
        
        loss3 = output3.sum()
        loss3.backward()
        
        hook_called_correct = hook_called == 3
        expected_grad = rm.ones((2, 5)) @ module.weight.data * 3
        grad_correct = tensor_allclose(input_data.grad, expected_grad)
        cache_cleaned = len(module._backward_hook_cache) == 0
        
        passed = hook_called_correct and grad_correct and cache_cleaned
        stats.add_result("多次前向传播 + 反向钩子 + 分别backward", passed,
                        f"钩子调用{hook_called}次(预期3), 梯度正确: {grad_correct}, 缓存清理: {cache_cleaned}")
        
        # 测试18: 单输出模块 + 部分输出不参与损失
        print("测试单输出模块 + 部分输出不参与损失...")
        module = SimpleLinearModule(10, 5)
        pre_hook_called = False
        backward_hook_called = False
        
        def pre_hook_partial_loss(m, grad_output):
            nonlocal pre_hook_called
            pre_hook_called = True
            return grad_output
        
        def backward_hook_partial_loss(m, grad_input, grad_output):
            nonlocal backward_hook_called
            backward_hook_called = True
            return grad_input
        
        module.register_full_backward_pre_hook(pre_hook_partial_loss)
        module.register_full_backward_hook(backward_hook_partial_loss)
        
        x = rm.ones((2, 10), requires_grad=True)
        out = module(x)
        
        mask = rm.zeros_like(out)
        mask[0, 0] = 1
        partial_out = out * mask
        loss = partial_out.sum()
        loss.backward()
        
        passed = pre_hook_called and backward_hook_called and x.grad is not None
        stats.add_result("单输出模块 + 部分输出不参与损失", passed,
                        f"预处理钩子调用: {pre_hook_called}, 反向钩子调用: {backward_hook_called}, 梯度计算: {x.grad is not None}")
        
    finally:
        stats.end_function()


# ==================== 主函数 ====================

def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*80)
    print(f"{Colors.BOLD}Riemann nn.Module 钩子处理测试套件（重构版）{Colors.ENDC}")
    print("="*80)
    
    start_time = time.time()
    
    # 创建共享的统计收集器
    stats = StatisticsCollector()
    
    test_functions = [
        ("钩子注册与管理", test_hook_registration_and_management),
        ("前向预处理钩子", test_forward_pre_hooks),
        ("前向钩子", test_forward_hooks),
        ("反向预处理钩子", test_backward_pre_hooks),
        ("反向钩子", test_backward_hooks),
        ("组合钩子", test_combined_hooks),
        ("复杂场景", test_complex_scenarios),
    ]
    
    results = []
    for name, test_func in test_functions:
        try:
            test_func(stats)
            results.append((name, True))
        except AssertionError as e:
            print(f"\n{Colors.FAIL}{name} 失败: {e}{Colors.ENDC}")
            results.append((name, False))
    
    elapsed = time.time() - start_time
    
    # 在最后输出汇总的测试统计摘要
    stats.print_summary()
    
    # 输出总体测试结果
    print("\n" + "="*80)
    print(f"{Colors.BOLD}总体测试结果{Colors.ENDC}")
    print("="*80)
    for name, passed in results:
        status = f"{Colors.OKGREEN}通过{Colors.ENDC}" if passed else f"{Colors.FAIL}失败{Colors.ENDC}"
        print(f"  {name}: {status}")
    print(f"\n总耗时: {elapsed:.2f}秒")
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print(f"\n{Colors.OKGREEN}所有测试通过！{Colors.ENDC}")
    else:
        print(f"\n{Colors.FAIL}部分测试失败！{Colors.ENDC}")
    
    return all_passed and stats.passed_cases == stats.total_cases


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
