#!/usr/bin/env python3
"""
Riemann nn.Container 容器类全功能测试套件

本测试文件全面验证 Riemann nn.Module 容器类的所有功能，
包括Sequential、ModuleList、ModuleDict、ParameterList、ParameterDict。
确保与 PyTorch nn.Module 容器类的行为一致。

测试覆盖的功能模块：
1. Sequential - 顺序容器，支持模块按顺序执行
2. ModuleList - 模块列表容器，支持动态模块管理
3. ModuleDict - 模块字典容器，支持命名模块管理
4. ParameterList - 参数列表容器，支持动态参数管理
5. ParameterDict - 参数字典容器，支持命名参数管理
"""

import numpy as np
import time
import sys
import os
from collections import OrderedDict

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# 导入riemann模块
try:
    import riemann as rm
    from riemann.nn import Parameter, Module, Linear, ReLU, Sequential, ModuleList, ModuleDict, ParameterList, ParameterDict
    RIEMANN_AVAILABLE = True
except ImportError as e:
    print(f"无法导入riemann模块: {e}")
    RIEMANN_AVAILABLE = False
    sys.exit(1)


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
        self.function_test_details = {}  # 存储每个函数的测试用例
    
    def start_function(self, function_name):
        self.current_function = function_name
        self.current_function_start_time = time.time()
        self.current_test_details = []
        
        if function_name not in self.function_stats:
            self.function_stats[function_name] = {"total": 0, "passed": 0, "time": 0.0}
        
        if function_name not in self.function_test_details:
            self.function_test_details[function_name] = []
        
        # 打印用例组开始信息，每组之间用空行隔开
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
            
            # 实时打印用例执行状态（使用颜色）
            print(f"  {case_name} [{status_color}{status}{Colors.ENDC}]" + (f" - {details}" if details else ""))
    
    def end_function(self):
        if self.current_function:
            elapsed = time.time() - self.current_function_start_time
            self.function_stats[self.current_function]["time"] += elapsed
            self.total_time += elapsed
            # 保存当前函数的测试用例
            self.function_test_details[self.current_function] = self.current_test_details.copy()
    
    def raise_if_failed(self):
        """检查当前函数是否有失败的测试，如果有则抛出AssertionError"""
        if self.current_function:
            function_stats = self.function_stats.get(self.current_function, {})
            total = function_stats.get("total", 0)
            passed = function_stats.get("passed", 0)
            if passed < total:
                failed_count = total - passed
                raise AssertionError(f"{self.current_function} 中有 {failed_count} 个测试用例失败")
    
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

# ==================== Sequential测试 ====================
def test_sequential():
    """测试Sequential顺序容器的功能"""
    stats.start_function("Sequential测试")
    
    try:
        # 测试1: 从模块创建Sequential
        print("测试从模块创建Sequential...")
        seq = Sequential(Linear(10, 20), ReLU(), Linear(20, 5))
        passed = len(seq) == 3
        stats.add_result("从模块创建", passed, f"模块数量: {len(seq)}")
        
        # 测试2: 从列表创建Sequential
        print("测试从列表创建Sequential...")
        modules = [Linear(10, 20), ReLU(), Linear(20, 5)]
        seq = Sequential(*modules)
        passed = len(seq) == 3
        stats.add_result("从列表创建", passed, f"模块数量: {len(seq)}")
        
        # 测试3: 从OrderedDict创建Sequential
        print("测试从OrderedDict创建Sequential...")
        modules = OrderedDict([
            ('fc1', Linear(10, 20)),
            ('relu', ReLU()),
            ('fc2', Linear(20, 5))
        ])
        seq = Sequential(modules)
        passed = len(seq) == 3 and list(seq._modules.keys()) == ['fc1', 'relu', 'fc2']
        stats.add_result("从OrderedDict创建", passed, f"模块数量: {len(seq)}, 键: {list(seq._modules.keys())}")
        
        # 测试4: 前向传播
        print("测试前向传播...")
        seq = Sequential(Linear(10, 20), ReLU(), Linear(20, 5))
        x = rm.randn(2, 10)
        output = seq(x)
        passed = hasattr(output, 'shape') and output.shape == (2, 5)
        stats.add_result("前向传播", passed, f"输出形状: {output.shape if hasattr(output, 'shape') else 'N/A'}")
        
        # 测试5: 迭代访问
        print("测试迭代访问...")
        seq = Sequential(Linear(10, 20), ReLU(), Linear(20, 5))
        module_count = 0
        for module in seq:
            module_count += 1
        passed = module_count == 3
        stats.add_result("迭代访问", passed, f"迭代模块数量: {module_count}")
        
        # 测试6: 索引访问
        print("测试索引访问...")
        seq = Sequential(Linear(10, 20), ReLU(), Linear(20, 5))
        first = seq[0]
        second = seq[1]
        third = seq[2]
        passed = (isinstance(first, Linear) and isinstance(second, ReLU) and isinstance(third, Linear))
        stats.add_result("索引访问", passed, "可以通过索引访问各层")
        
        # 测试7: 负数索引
        print("测试负数索引...")
        seq = Sequential(Linear(10, 20), ReLU(), Linear(20, 5))
        last = seq[-1]
        second_last = seq[-2]
        passed = isinstance(last, Linear) and isinstance(second_last, ReLU)
        stats.add_result("负数索引", passed, "支持负数索引访问")
        
        # 测试8: 长度获取
        print("测试长度获取...")
        seq = Sequential(Linear(10, 20), ReLU())
        passed = len(seq) == 2
        stats.add_result("长度获取", passed, f"长度: {len(seq)}")
        
        # 测试9: 动态添加模块
        print("测试动态添加模块...")
        seq = Sequential(Linear(10, 20))
        seq.add_module('1', ReLU())
        seq.add_module('2', Linear(20, 5))
        passed = len(seq) == 3
        stats.add_result("动态添加模块", passed, f"添加后长度: {len(seq)}")
        
        # 测试10: 空Sequential
        print("测试空Sequential...")
        seq = Sequential()
        passed = len(seq) == 0
        stats.add_result("空Sequential", passed, f"空容器长度: {len(seq)}")
        
        # 测试11: 索引越界检查
        print("测试索引越界检查...")
        seq = Sequential(Linear(10, 20))
        try:
            _ = seq[5]
            stats.add_result("索引越界检查", False, "应该抛出IndexError")
        except IndexError:
            stats.add_result("索引越界检查", True, "正确抛出IndexError")
        
        # 测试12: 类型检查
        print("测试类型检查...")
        try:
            _ = seq["invalid"]
            stats.add_result("类型检查", False, "应该抛出TypeError")
        except TypeError:
            stats.add_result("类型检查", True, "正确抛出TypeError")
        
        # 测试13: 在自定义模块中使用
        print("测试在自定义模块中使用...")
        class TestModule(Module):
            def __init__(self):
                super().__init__()
                self.seq = Sequential(
                    Linear(10, 20),
                    ReLU(),
                    Linear(20, 5)
                )
            
            def forward(self, x):
                return self.seq(x)
        
        module = TestModule()
        x = rm.randn(2, 10)
        output = module(x)
        passed = hasattr(output, 'shape') and output.shape == (2, 5)
        stats.add_result("自定义模块中使用", passed, f"输出形状: {output.shape if hasattr(output, 'shape') else 'N/A'}")
        
    except Exception as e:
        print(f"Sequential测试出现异常: {e}")
        stats.add_result("Sequential测试异常", False, str(e))
    
    finally:
        stats.end_function()
        stats.raise_if_failed()


# ==================== ModuleList测试 ====================
def test_module_list():
    """测试ModuleList模块列表容器的功能"""
    stats.start_function("ModuleList测试")
    
    try:
        # 测试1: 创建空ModuleList
        print("测试创建空ModuleList...")
        modules = ModuleList()
        passed = len(modules) == 0
        stats.add_result("空ModuleList创建", passed, f"空列表长度: {len(modules)}")
        
        # 测试2: 从列表创建ModuleList
        print("测试从列表创建ModuleList...")
        modules = ModuleList([Linear(10, 20), ReLU(), Linear(20, 5)])
        passed = len(modules) == 3
        stats.add_result("从列表创建", passed, f"列表长度: {len(modules)}")
        
        # 测试3: append方法
        print("测试append方法...")
        modules = ModuleList()
        modules.append(Linear(10, 20))
        modules.append(ReLU())
        passed = len(modules) == 2
        stats.add_result("append方法", passed, f"添加后长度: {len(modules)}")
        
        # 测试4: extend方法
        print("测试extend方法...")
        modules = ModuleList([Linear(10, 20)])
        modules.extend([ReLU(), Linear(20, 5)])
        passed = len(modules) == 3
        stats.add_result("extend方法", passed, f"扩展后长度: {len(modules)}")
        
        # 测试5: 索引访问
        print("测试索引访问...")
        modules = ModuleList([Linear(10, 20), ReLU(), Linear(20, 5)])
        first = modules[0]
        second = modules[1]
        third = modules[2]
        passed = (isinstance(first, Linear) and isinstance(second, ReLU) and isinstance(third, Linear))
        stats.add_result("索引访问", passed, "可以通过索引访问各模块")
        
        # 测试6: 负数索引
        print("测试负数索引...")
        modules = ModuleList([Linear(10, 20), ReLU(), Linear(20, 5)])
        last = modules[-1]
        second_last = modules[-2]
        passed = isinstance(last, Linear) and isinstance(second_last, ReLU)
        stats.add_result("负数索引", passed, "支持负数索引访问")
        
        # 测试7: insert方法
        print("测试insert方法...")
        modules = ModuleList([Linear(10, 20), Linear(20, 5)])
        modules.insert(1, ReLU())
        passed = len(modules) == 3 and isinstance(modules[1], ReLU)
        stats.add_result("insert方法", passed, f"插入后长度: {len(modules)}, 位置1是ReLU: {isinstance(modules[1], ReLU)}")
        
        # 测试8: pop方法
        print("测试pop方法...")
        modules = ModuleList([Linear(10, 20), ReLU(), Linear(20, 5)])
        popped = modules.pop(1)
        passed = len(modules) == 2 and isinstance(popped, ReLU)
        stats.add_result("pop方法", passed, f"弹出后长度: {len(modules)}, 弹出的是ReLU: {isinstance(popped, ReLU)}")
        
        # 测试9: pop默认参数（最后一个）
        print("测试pop默认参数...")
        modules = ModuleList([Linear(10, 20), ReLU(), Linear(20, 5)])
        popped = modules.pop()
        passed = len(modules) == 2 and isinstance(popped, Linear)
        stats.add_result("pop默认参数", passed, f"弹出后长度: {len(modules)}, 弹出最后一个: {isinstance(popped, Linear)}")
        
        # 测试10: clear方法
        print("测试clear方法...")
        modules = ModuleList([Linear(10, 20), ReLU()])
        modules.clear()
        passed = len(modules) == 0
        stats.add_result("clear方法", passed, f"清空后长度: {len(modules)}")
        
        # 测试11: index方法
        print("测试index方法...")
        relu = ReLU()
        modules = ModuleList([Linear(10, 20), relu, Linear(20, 5)])
        idx = modules.index(relu)
        passed = idx == 1
        stats.add_result("index方法", passed, f"ReLU的索引: {idx}")
        
        # 测试12: remove方法
        print("测试remove方法...")
        relu = ReLU()
        modules = ModuleList([Linear(10, 20), relu, Linear(20, 5)])
        modules.remove(relu)
        passed = len(modules) == 2
        stats.add_result("remove方法", passed, f"移除后长度: {len(modules)}")
        
        # 测试13: __iadd__方法
        print("测试__iadd__方法...")
        modules = ModuleList([Linear(10, 20)])
        modules += [ReLU(), Linear(20, 5)]
        passed = len(modules) == 3
        stats.add_result("__iadd__方法", passed, f"+= 后长度: {len(modules)}")
        
        # 测试14: 迭代访问
        print("测试迭代访问...")
        modules = ModuleList([Linear(10, 20), ReLU(), Linear(20, 5)])
        count = 0
        for module in modules:
            count += 1
        passed = count == 3
        stats.add_result("迭代访问", passed, f"迭代模块数量: {count}")
        
        # 测试15: 在自定义模块中使用
        print("测试在自定义模块中使用...")
        class TestModule(Module):
            def __init__(self):
                super().__init__()
                self.layers = ModuleList([
                    Linear(10, 20),
                    ReLU(),
                    Linear(20, 5)
                ])
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        module = TestModule()
        x = rm.randn(2, 10)
        output = module(x)
        passed = hasattr(output, 'shape') and output.shape == (2, 5)
        stats.add_result("自定义模块中使用", passed, f"输出形状: {output.shape if hasattr(output, 'shape') else 'N/A'}")
        
        # 测试16: 参数注册
        print("测试参数注册...")
        modules = ModuleList([Linear(10, 20), Linear(20, 5)])
        named_params = list(modules.named_parameters())
        passed = len(named_params) > 0  # Linear层有参数
        stats.add_result("参数注册", passed, f"注册参数数量: {len(named_params)}")
        
    except Exception as e:
        print(f"ModuleList测试出现异常: {e}")
        stats.add_result("ModuleList测试异常", False, str(e))
    
    finally:
        stats.end_function()
        stats.raise_if_failed()


# ==================== ModuleDict测试 ====================
def test_module_dict():
    """测试ModuleDict模块字典容器的功能"""
    stats.start_function("ModuleDict测试")
    
    try:
        # 测试1: 创建空ModuleDict
        print("测试创建空ModuleDict...")
        modules = ModuleDict()
        passed = len(modules) == 0
        stats.add_result("空ModuleDict创建", passed, f"空字典长度: {len(modules)}")
        
        # 测试2: 从字典创建ModuleDict
        print("测试从字典创建ModuleDict...")
        modules = ModuleDict({
            'fc1': Linear(10, 20),
            'relu': ReLU(),
            'fc2': Linear(20, 5)
        })
        passed = len(modules) == 3
        stats.add_result("从字典创建", passed, f"字典长度: {len(modules)}")
        
        # 测试3: __setitem__方法
        print("测试__setitem__方法...")
        modules = ModuleDict()
        modules['fc1'] = Linear(10, 20)
        modules['relu'] = ReLU()
        passed = len(modules) == 2
        stats.add_result("__setitem__方法", passed, f"添加后长度: {len(modules)}")
        
        # 测试4: __getitem__方法
        print("测试__getitem__方法...")
        modules = ModuleDict({
            'fc1': Linear(10, 20),
            'relu': ReLU()
        })
        fc1 = modules['fc1']
        relu = modules['relu']
        passed = isinstance(fc1, Linear) and isinstance(relu, ReLU)
        stats.add_result("__getitem__方法", passed, "可以通过键访问各模块")
        
        # 测试5: update方法（dict）
        print("测试update方法（dict）...")
        modules = ModuleDict({'fc1': Linear(10, 20)})
        modules.update({'relu': ReLU(), 'fc2': Linear(20, 5)})
        passed = len(modules) == 3
        stats.add_result("update方法(dict)", passed, f"更新后长度: {len(modules)}")
        
        # 测试6: update方法（ModuleDict）
        print("测试update方法（ModuleDict）...")
        modules1 = ModuleDict({'fc1': Linear(10, 20)})
        modules2 = ModuleDict({'fc2': Linear(20, 5)})
        modules1.update(modules2)
        passed = len(modules1) == 2 and 'fc1' in modules1 and 'fc2' in modules1
        stats.add_result("update方法(ModuleDict)", passed, f"更新后长度: {len(modules1)}")
        
        # 测试7: keys方法
        print("测试keys方法...")
        modules = ModuleDict({'fc1': Linear(10, 20), 'fc2': Linear(20, 5)})
        keys = list(modules.keys())
        passed = len(keys) == 2 and all(isinstance(k, str) for k in keys)
        stats.add_result("keys方法", passed, f"键: {keys}")
        
        # 测试8: values方法
        print("测试values方法...")
        modules = ModuleDict({'fc1': Linear(10, 20), 'relu': ReLU()})
        values = list(modules.values())
        passed = len(values) == 2
        stats.add_result("values方法", passed, f"值数量: {len(values)}")
        
        # 测试9: items方法
        print("测试items方法...")
        modules = ModuleDict({'fc1': Linear(10, 20), 'fc2': Linear(20, 5)})
        items = list(modules.items())
        passed = len(items) == 2 and all(isinstance(item, tuple) and len(item) == 2 for item in items)
        stats.add_result("items方法", passed, f"项数量: {len(items)}")
        
        # 测试10: __iter__方法
        print("测试__iter__方法...")
        modules = ModuleDict({'fc1': Linear(10, 20), 'fc2': Linear(20, 5)})
        keys = []
        for key in modules:
            keys.append(key)
        passed = len(keys) == 2 and 'fc1' in keys and 'fc2' in keys
        stats.add_result("__iter__方法", passed, f"迭代键: {keys}")
        
        # 测试11: __len__方法
        print("测试__len__方法...")
        modules = ModuleDict({'fc1': Linear(10, 20)})
        passed = len(modules) == 1
        stats.add_result("__len__方法", passed, f"长度: {len(modules)}")
        
        # 测试12: pop方法
        print("测试pop方法...")
        modules = ModuleDict({'fc1': Linear(10, 20), 'fc2': Linear(20, 5)})
        popped = modules.pop('fc1')
        passed = len(modules) == 1 and isinstance(popped, Linear)
        stats.add_result("pop方法", passed, f"弹出后长度: {len(modules)}")
        
        # 测试13: clear方法
        print("测试clear方法...")
        modules = ModuleDict({'fc1': Linear(10, 20), 'fc2': Linear(20, 5)})
        modules.clear()
        passed = len(modules) == 0
        stats.add_result("clear方法", passed, f"清空后长度: {len(modules)}")
        
        # 测试14: 在自定义模块中使用
        print("测试在自定义模块中使用...")
        class TestModule(Module):
            def __init__(self):
                super().__init__()
                self.layers = ModuleDict({
                    'fc1': Linear(10, 20),
                    'relu': ReLU(),
                    'fc2': Linear(20, 5)
                })
            
            def forward(self, x):
                x = self.layers['fc1'](x)
                x = self.layers['relu'](x)
                x = self.layers['fc2'](x)
                return x
        
        module = TestModule()
        x = rm.randn(2, 10)
        output = module(x)
        passed = hasattr(output, 'shape') and output.shape == (2, 5)
        stats.add_result("自定义模块中使用", passed, f"输出形状: {output.shape if hasattr(output, 'shape') else 'N/A'}")
        
        # 测试15: 参数注册
        print("测试参数注册...")
        modules = ModuleDict({'fc1': Linear(10, 20), 'fc2': Linear(20, 5)})
        named_params = list(modules.named_parameters())
        passed = len(named_params) > 0
        stats.add_result("参数注册", passed, f"注册参数数量: {len(named_params)}")
        
    except Exception as e:
        print(f"ModuleDict测试出现异常: {e}")
        stats.add_result("ModuleDict测试异常", False, str(e))
    
    finally:
        stats.end_function()
        stats.raise_if_failed()


# ==================== ParameterList测试 ====================
def test_parameter_list():
    """测试ParameterList参数列表容器的功能"""
    stats.start_function("ParameterList测试")
    
    try:
        # 测试1: 创建空ParameterList
        print("测试创建空ParameterList...")
        params = ParameterList()
        passed = len(params) == 0
        stats.add_result("空ParameterList创建", passed, f"空列表长度: {len(params)}")
        
        # 测试2: 从列表创建ParameterList
        print("测试从列表创建ParameterList...")
        params = ParameterList([
            Parameter(rm.randn(10, 20)),
            Parameter(rm.randn(20))
        ])
        passed = len(params) == 2
        stats.add_result("从列表创建", passed, f"列表长度: {len(params)}")
        
        # 测试3: append方法
        print("测试append方法...")
        params.append(Parameter(rm.randn(20, 5)))
        params.append(Parameter(rm.randn(5)))
        passed = len(params) == 4
        stats.add_result("append方法", passed, f"添加后长度: {len(params)}")
        
        # 测试4: 索引访问
        print("测试索引访问...")
        params = ParameterList([
            Parameter(rm.randn(10, 20)),
            Parameter(rm.randn(20)),
            Parameter(rm.randn(20, 5)),
            Parameter(rm.randn(5))
        ])
        weight1 = params[0]
        bias1 = params[1]
        weight2 = params[2]
        bias2 = params[3]
        passed = (hasattr(weight1, 'shape') and hasattr(bias1, 'shape') and 
                  hasattr(weight2, 'shape') and hasattr(bias2, 'shape'))
        stats.add_result("索引访问", passed, "所有参数都可以通过索引访问")
        
        # 测试5: 负数索引
        print("测试负数索引...")
        params = ParameterList([
            Parameter(rm.randn(10, 20)),
            Parameter(rm.randn(20)),
            Parameter(rm.randn(20, 5))
        ])
        last_param = params[-1]
        second_last = params[-2]
        passed = hasattr(last_param, 'shape') and hasattr(second_last, 'shape')
        stats.add_result("负数索引", passed, "支持负数索引访问")
        
        # 测试6: __setitem__方法
        print("测试__setitem__方法...")
        params = ParameterList([
            Parameter(rm.randn(10, 20)),
            Parameter(rm.randn(20))
        ])
        new_param = Parameter(rm.randn(10, 30))
        params[0] = new_param
        passed = params[0] is new_param
        stats.add_result("__setitem__方法", passed, "可以替换参数")
        
        # 测试7: extend方法
        print("测试extend方法...")
        params = ParameterList([Parameter(rm.randn(10, 20))])
        new_params = [Parameter(rm.randn(20, 5)), Parameter(rm.randn(5))]
        params.extend(new_params)
        passed = len(params) == 3
        stats.add_result("extend方法", passed, f"扩展后长度: {len(params)}")
        
        # 测试8: 迭代访问
        print("测试迭代访问...")
        params = ParameterList([
            Parameter(rm.randn(10, 20)),
            Parameter(rm.randn(20)),
            Parameter(rm.randn(20, 5))
        ])
        param_count = 0
        for param in params:
            param_count += 1
            assert hasattr(param, 'shape')
        passed = param_count == 3
        stats.add_result("迭代访问", passed, f"迭代参数数量: {param_count}")
        
        # 测试9: 参数注册
        print("测试参数注册...")
        params = ParameterList([
            Parameter(rm.randn(10, 20)),
            Parameter(rm.randn(20)),
            Parameter(rm.randn(20, 5))
        ])
        named_params = list(params.named_parameters())
        passed = len(named_params) == 3
        stats.add_result("参数注册", passed, f"注册参数数量: {len(named_params)}")
        
        # 测试10: 在模块中使用
        print("测试在模块中使用...")
        class TestModule(Module):
            def __init__(self):
                super().__init__()
                self.params = ParameterList([
                    Parameter(rm.randn(10, 20)),
                    Parameter(rm.randn(20))
                ])
            
            def forward(self, x):
                return x @ self.params[0] + self.params[1]
        
        module = TestModule()
        x = rm.randn(2, 10)
        output = module(x)
        passed = hasattr(output, 'shape') and output.shape == (2, 20)
        stats.add_result("模块中使用", passed, f"输出形状: {output.shape if hasattr(output, 'shape') else 'N/A'}")
        
        # 测试11: 类型检查
        print("测试类型检查...")
        params = ParameterList()
        try:
            params.append(rm.randn(5, 5))
            stats.add_result("类型检查", False, "应该拒绝非Parameter对象")
        except TypeError:
            stats.add_result("类型检查", True, "正确拒绝非Parameter对象")
        
        # 测试12: __setitem__类型检查
        print("测试__setitem__类型检查...")
        params = ParameterList([Parameter(rm.randn(10, 20))])
        try:
            params[0] = rm.randn(10, 20)
            stats.add_result("__setitem__类型检查", False, "应该拒绝非Parameter对象")
        except TypeError:
            stats.add_result("__setitem__类型检查", True, "正确拒绝非Parameter对象")
        
    except Exception as e:
        print(f"ParameterList测试出现异常: {e}")
        stats.add_result("ParameterList测试异常", False, str(e))
    
    finally:
        stats.end_function()
        stats.raise_if_failed()


# ==================== ParameterDict测试 ====================
def test_parameter_dict():
    """测试ParameterDict参数字典容器的功能"""
    stats.start_function("ParameterDict测试")
    
    try:
        # 测试1: 创建空ParameterDict
        print("测试创建空ParameterDict...")
        params = ParameterDict()
        passed = len(params) == 0
        stats.add_result("空ParameterDict创建", passed, f"空字典长度: {len(params)}")
        
        # 测试2: 从字典创建ParameterDict
        print("测试从字典创建ParameterDict...")
        params = ParameterDict({
            'w1': Parameter(rm.randn(10, 20)),
            'b1': Parameter(rm.randn(20))
        })
        passed = len(params) == 2
        stats.add_result("从字典创建", passed, f"字典长度: {len(params)}")
        
        # 测试3: __setitem__方法
        print("测试__setitem__方法...")
        params = ParameterDict()
        params['w1'] = Parameter(rm.randn(10, 20))
        params['b1'] = Parameter(rm.randn(20))
        passed = len(params) == 2
        stats.add_result("__setitem__方法", passed, f"添加后长度: {len(params)}")
        
        # 测试4: __getitem__方法
        print("测试__getitem__方法...")
        params = ParameterDict({
            'w1': Parameter(rm.randn(10, 20)),
            'b1': Parameter(rm.randn(20))
        })
        weight1 = params['w1']
        bias1 = params['b1']
        passed = (hasattr(weight1, 'shape') and hasattr(bias1, 'shape'))
        stats.add_result("__getitem__方法", passed, "所有参数都可以通过键访问")
        
        # 测试5: update方法
        print("测试update方法...")
        params = ParameterDict({'w1': Parameter(rm.randn(10, 20))})
        params.update({
            'b1': Parameter(rm.randn(20)),
            'w2': Parameter(rm.randn(20, 5))
        })
        passed = len(params) == 3
        stats.add_result("update方法", passed, f"更新后长度: {len(params)}")
        
        # 测试6: keys方法
        print("测试keys方法...")
        params = ParameterDict({
            'w1': Parameter(rm.randn(10, 20)),
            'b1': Parameter(rm.randn(20))
        })
        keys = list(params.keys())
        passed = len(keys) == 2 and all(isinstance(k, str) for k in keys)
        stats.add_result("keys方法", passed, f"键数量: {len(keys)}, 都是字符串: {all(isinstance(k, str) for k in keys)}")
        
        # 测试7: items方法
        print("测试items方法...")
        params = ParameterDict({
            'w1': Parameter(rm.randn(10, 20)),
            'b1': Parameter(rm.randn(20))
        })
        items = list(params.items())
        passed = len(items) == 2 and all(isinstance(item, tuple) and len(item) == 2 for item in items)
        stats.add_result("items方法", passed, f"项数量: {len(items)}, 都是元组: {all(isinstance(item, tuple) for item in items)}")
        
        # 测试8: values方法
        print("测试values方法...")
        params = ParameterDict({
            'w1': Parameter(rm.randn(10, 20)),
            'b1': Parameter(rm.randn(20))
        })
        values = list(params.values())
        passed = len(values) == 2 and all(hasattr(v, 'shape') for v in values)
        stats.add_result("values方法", passed, f"值数量: {len(values)}, 都是参数: {all(hasattr(v, 'shape') for v in values)}")
        
        # 测试9: 迭代访问
        print("测试迭代访问...")
        params = ParameterDict({
            'w1': Parameter(rm.randn(10, 20)),
            'b1': Parameter(rm.randn(20))
        })
        key_count = 0
        for key in params:
            key_count += 1
            assert isinstance(key, str)
        passed = key_count == 2
        stats.add_result("迭代访问", passed, f"迭代键数量: {key_count}")
        
        # 测试10: 参数注册
        print("测试参数注册...")
        params = ParameterDict({
            'w1': Parameter(rm.randn(10, 20)),
            'b1': Parameter(rm.randn(20))
        })
        named_params = list(params.named_parameters())
        passed = len(named_params) == 2
        stats.add_result("参数注册", passed, f"注册参数数量: {len(named_params)}")
        
        # 测试11: 在模块中使用
        print("测试在模块中使用...")
        class TestModule(Module):
            def __init__(self):
                super().__init__()
                self.params = ParameterDict({
                    'w1': Parameter(rm.randn(10, 20)),
                    'b1': Parameter(rm.randn(20))
                })
            
            def forward(self, x):
                return x @ self.params['w1'] + self.params['b1']
        
        module = TestModule()
        x = rm.randn(2, 10)
        output = module(x)
        passed = hasattr(output, 'shape') and output.shape == (2, 20)
        stats.add_result("模块中使用", passed, f"输出形状: {output.shape if hasattr(output, 'shape') else 'N/A'}")
        
        # 测试12: 类型检查
        print("测试类型检查...")
        params = ParameterDict()
        try:
            params['test'] = rm.randn(5, 5)
            stats.add_result("类型检查", False, "应该拒绝非Parameter对象")
        except TypeError:
            stats.add_result("类型检查", True, "正确拒绝非Parameter对象")
        
        # 测试13: 键类型检查
        print("测试键类型检查...")
        params = ParameterDict()
        try:
            params[123] = Parameter(rm.randn(5, 5))
            stats.add_result("键类型检查", False, "应该拒绝非字符串键")
        except TypeError:
            stats.add_result("键类型检查", True, "正确拒绝非字符串键")
        
        # 测试14: 覆盖参数
        print("测试覆盖参数...")
        params = ParameterDict({'w1': Parameter(rm.randn(10, 20))})
        old_param = params['w1']
        params['w1'] = Parameter(rm.randn(10, 20))
        new_param = params['w1']
        passed = old_param is not new_param
        stats.add_result("覆盖参数", passed, "参数被正确覆盖")
        
        # 测试15: pop方法
        print("测试pop方法...")
        params = ParameterDict({
            'w1': Parameter(rm.randn(10, 20)),
            'b1': Parameter(rm.randn(20))
        })
        popped = params.pop('w1')
        passed = len(params) == 1 and hasattr(popped, 'shape')
        stats.add_result("pop方法", passed, f"弹出后长度: {len(params)}")
        
        # 测试16: clear方法
        print("测试clear方法...")
        params = ParameterDict({
            'w1': Parameter(rm.randn(10, 20)),
            'b1': Parameter(rm.randn(20))
        })
        params.clear()
        passed = len(params) == 0
        stats.add_result("clear方法", passed, f"清空后长度: {len(params)}")
        
    except Exception as e:
        print(f"ParameterDict测试出现异常: {e}")
        stats.add_result("ParameterDict测试异常", False, str(e))
    
    finally:
        stats.end_function()
        stats.raise_if_failed()


# ==================== 测试运行入口 ====================
def run_all_tests():
    """运行所有测试"""
    print(f"{Colors.BOLD}Riemann nn.Container 容器类全功能测试套件{Colors.ENDC}")
    print("="*80)
    print(f"Riemann版本: 可用")
    print("="*80)
    
    # 设置随机种子确保可重复性
    np.random.seed(42)
    
    # 运行所有测试
    test_functions = [
        test_sequential,
        test_module_list,
        test_module_dict,
        test_parameter_list,
        test_parameter_dict
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
