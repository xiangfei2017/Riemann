import unittest
import time
import sys
import os
import numpy as np

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','src')))

# 导入riemann模块
try:
    import riemann as rm
    from riemann import no_grad, enable_grad, is_grad_enabled, set_grad_enabled
    RM_AVAILABLE = True
except ImportError:
    print("无法导入riemann模块，请确保项目路径设置正确")
    RM_AVAILABLE = False

# 导入PyTorch模块（如果可用）
try:
    import torch
    import torch.autograd
    TORCH_AVAILABLE = True
except ImportError:
    print("无法导入PyTorch模块，将仅测试Riemann功能")
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
    
    # 修改StatisticsCollector类的start_function方法
    def start_function(self, function_name):
        self.current_function = function_name
        self.current_function_start_time = time.time()
        
        if function_name not in self.function_stats:
            # 为每个函数创建独立的测试详情列表
            self.function_stats[function_name] = {
                "total": 0, 
                "passed": 0, 
                "time": 0.0,
                "details": []  # 为每个函数单独存储测试详情
            }
    
    # 修改add_result方法
    def add_result(self, case_name, passed, details=None):
        self.total_cases += 1
        if passed:
            self.passed_cases += 1
        
        if self.current_function:
            self.function_stats[self.current_function]["total"] += 1
            if passed:
                self.function_stats[self.current_function]["passed"] += 1
                
            # 将测试详情添加到对应函数的统计中
            status = "通过" if passed else "失败"
            status_color = Colors.OKGREEN if passed else Colors.FAIL
            self.function_stats[self.current_function]["details"].append({
                "name": case_name,
                "status": status,
                "color": status_color,
                "details": details,
                "passed": passed  # 添加passed字段用于错误信息显示
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
    
    # 修改StatisticsCollector类的print_summary方法
    def print_summary(self):
        # 定义各列的标题
        headers = ['用例名', '通过/总数', '通过率', '耗时(秒)']
        
        # 计算各列标题的显示宽度
        header_widths = [self._get_display_width(h) for h in headers]
        
        # 计算数据行中各列的最大显示宽度
        max_func_name_width = header_widths[0]
        max_detail_name_width = 0
        for func_name, stats in self.function_stats.items():
            max_func_name_width = max(max_func_name_width, self._get_display_width(func_name))
            # 同时计算最长的测试详情名称
            for detail in stats.get("details", []):
                detail_width = self._get_display_width(f"  {detail['name']}")
                max_detail_name_width = max(max_detail_name_width, detail_width)
        
        # 为各列设置最终宽度，标题宽度和内容宽度的最大值，并留出适当间距
        col_widths = [
            max(max(max_func_name_width, max_detail_name_width), header_widths[0]) + 2,  # 用例名列
            header_widths[1] + 4,  # 通过/总数列
            header_widths[2] + 4,  # 通过率列
            header_widths[3] + 4   # 耗时列
        ]
        
        total_width = sum(col_widths)
        
        # 输出分隔线和标题（首尾使用'='）
        print("\n" + "="*total_width)
        print(f"{Colors.BOLD}测试统计摘要{Colors.ENDC}")
        print("="*total_width)
        print(f"总测试用例数: {self.total_cases}")
        print(f"通过测试用例数: {Colors.OKGREEN if self.passed_cases == self.total_cases else Colors.FAIL}{self.passed_cases}{Colors.ENDC}")
        print(f"测试通过率: {Colors.OKGREEN if self.passed_cases == self.total_cases else Colors.FAIL}{self.passed_cases/self.total_cases*100:.2f}%{Colors.ENDC}")
        print(f"总耗时: {self.total_time:.4f} 秒")
        print("\n各用例测试详情:")
        print("-"*total_width)
        
        # 打印表头 - 左对齐
        header_line = ""
        for i, header in enumerate(headers):
            header_width = self._get_display_width(header)
            padding = col_widths[i] - header_width
            header_line += header + " " * padding
        print(header_line)
        print("-"*total_width)
        
        # 打印数据行 - 精确计算每个值的填充
        for i, (func_name, stats) in enumerate(self.function_stats.items()):
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
            pass_rate_display = f"{pass_rate:.2f}%"
            pass_rate_width = self._get_display_width(pass_rate_display)
            pass_rate_padding = col_widths[2] - pass_rate_width
            
            time_display = f"{stats['time']:.4f}s"
            time_width = self._get_display_width(time_display)
            time_padding = col_widths[3] - time_width
            
            # 构建完整的行
            print(
                f"{func_name_display}{' ' * func_name_padding}" +
                f"{pass_total_display}{' ' * pass_total_padding}" +
                f"{status_color}{pass_rate_display}{' ' * pass_rate_padding}{Colors.ENDC}" +
                f"{time_display}{' ' * time_padding}"
            )
            
            # 输出详细测试结果 - 使用函数自己的详情列表
            for detail in stats.get("details", []):  # 这里改为使用stats["details"]
                detail_display = f"{detail['name']}"
                detail_width = self._get_display_width(detail_display)
                detail_padding = col_widths[0] - detail_width
                
                status_display = f"{detail['color']}{detail['status']}{Colors.ENDC}"
                status_width = self._get_display_width(f"{detail['status']}")  # 不计算颜色代码宽度
                status_padding = col_widths[1] - status_width
                
                print(
                    f"{detail_display}{' ' * detail_padding}" +
                    f"{status_display}{' ' * status_padding}"
                )
                
                if detail['details'] and not detail['passed']:
                    for err_detail in detail['details']:
                        error_display = f"  错误信息: {err_detail}"
                        error_width = self._get_display_width(error_display)
                        error_padding = col_widths[0] - error_width
                        print(f"{error_display}{' ' * error_padding}")
            
            # 中间行使用'-'，只有首尾使用'='
            if i == len(self.function_stats) - 1:  # 最后一行用'='
                print("="*total_width)
            else:  # 中间行用'-'
                print("-"*total_width)

# 创建全局统计实例
stats = StatisticsCollector()

# 标识是否作为独立脚本运行
IS_RUNNING_AS_SCRIPT = False

# 比较Riemann和PyTorch的结果
def compare_grad_context_behaviors(rm_result, torch_result):
    """比较Riemann和PyTorch在梯度上下文控制中的行为"""
    if not TORCH_AVAILABLE or torch_result is None:
        return True  # 如果PyTorch不可用，则默认通过
    
    # 对于布尔值结果直接比较
    if isinstance(rm_result, bool) and isinstance(torch_result, bool):
        return rm_result == torch_result
    
    # 对于张量属性的比较
    if hasattr(rm_result, 'requires_grad') and hasattr(torch_result, 'requires_grad'):
        return rm_result.requires_grad == torch_result.requires_grad
    
    # 对于列表或元组形式的结果进行递归比较
    if isinstance(rm_result, (list, tuple)) and isinstance(torch_result, (list, tuple)):
        if len(rm_result) != len(torch_result):
            return False
        return all(compare_grad_context_behaviors(r, t) for r, t in zip(rm_result, torch_result))
    
    # 默认比较值
    return rm_result == torch_result

class TestGradContext(unittest.TestCase):
    def setUp(self):
        # 设置随机种子以确保结果可重复
        np.random.seed(42)
        if TORCH_AVAILABLE:
            torch.manual_seed(42)
        # 获取当前测试方法名
        self.current_test_name = self._testMethodName
        # 如果是独立脚本运行，则开始记录函数统计
        if IS_RUNNING_AS_SCRIPT:
            stats.start_function(self.current_test_name)
            print(f"\n{Colors.HEADER}开始测试: {self.current_test_name}{Colors.ENDC}")
                
    def tearDown(self):
        # 如果是独立脚本运行，则结束函数统计
        if IS_RUNNING_AS_SCRIPT:
            stats.end_function()
            print(f"{Colors.OKBLUE}测试完成: {self.current_test_name}{Colors.ENDC}")
    
    def test_default_behavior(self):
        """测试默认的梯度控制行为"""
        case_name = "默认的梯度控制行为"
        start_time = time.time()
        try:
            # 测试默认梯度状态
            rm_grad_enabled = is_grad_enabled()
            torch_grad_enabled = torch.is_grad_enabled() if TORCH_AVAILABLE else True
            
            # 创建requires_grad=True的张量
            x_np = 2.0
            rm_x = rm.tensor(x_np, requires_grad=True)
            if TORCH_AVAILABLE:
                torch_x = torch.tensor(x_np, requires_grad=True)
            else:
                torch_x = None
            
            # 执行计算
            rm_y = rm_x * 2
            rm_z = rm_y + 3
            
            torch_y = torch_x * 2 if TORCH_AVAILABLE else None
            torch_z = torch_y + 3 if TORCH_AVAILABLE else None
            
            # 收集结果
            rm_result = (rm_grad_enabled, rm_x.requires_grad, rm_y.requires_grad, rm_z.requires_grad)
            torch_result = (torch_grad_enabled, torch_x.requires_grad, torch_y.requires_grad, torch_z.requires_grad) if TORCH_AVAILABLE else None
            
            # 比较结果
            passed = compare_grad_context_behaviors(rm_result, torch_result)
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  行为比较: 失败")
                    print(f"  Riemann: {rm_result}")
                    print(f"  PyTorch: {torch_result}")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"梯度默认行为不匹配: {case_name}")
                
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_no_grad_context(self):
        """测试no_grad上下文管理器"""
        case_name = "no_grad上下文管理器"
        start_time = time.time()
        try:
            # 创建requires_grad=True的张量
            x_np = 2.0
            rm_x = rm.tensor(x_np, requires_grad=True)
            if TORCH_AVAILABLE:
                torch_x = torch.tensor(x_np, requires_grad=True)
            else:
                torch_x = None
            
            # 在no_grad上下文中执行计算
            with no_grad():
                rm_grad_enabled = is_grad_enabled()
                rm_y = rm_x * 2
                rm_z = rm_y + 3
                rm_new_tensor = rm.tensor(x_np, requires_grad=True)
            
            if TORCH_AVAILABLE:
                with torch.no_grad():
                    torch_grad_enabled = torch.is_grad_enabled()
                    torch_y = torch_x * 2
                    torch_z = torch_y + 3
                    torch_new_tensor = torch.tensor(x_np, requires_grad=True)
            else:
                torch_grad_enabled = None
                torch_y = None
                torch_z = None
                torch_new_tensor = None
            
            # 收集结果
            rm_result = (
                rm_grad_enabled,
                rm_x.requires_grad,
                rm_y.requires_grad,
                rm_z.requires_grad,
                rm_new_tensor.requires_grad
            )
            
            torch_result = (
                torch_grad_enabled,
                torch_x.requires_grad,
                torch_y.requires_grad,
                torch_z.requires_grad,
                torch_new_tensor.requires_grad
            ) if TORCH_AVAILABLE else None
            
            # 比较结果
            passed = compare_grad_context_behaviors(rm_result, torch_result)
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  行为比较: 失败")
                    print(f"  Riemann: {rm_result}")
                    print(f"  PyTorch: {torch_result}")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"no_grad上下文行为不匹配: {case_name}")
                
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_enable_grad_context(self):
        """测试enable_grad上下文管理器（嵌套在no_grad中）"""
        case_name = "enable_grad上下文嵌套"
        start_time = time.time()
        try:
            # 创建requires_grad=True的张量
            x_np = 2.0
            rm_x = rm.tensor(x_np, requires_grad=True)
            if TORCH_AVAILABLE:
                torch_x = torch.tensor(x_np, requires_grad=True)
            else:
                torch_x = None
            
            # 嵌套上下文管理器测试
            with no_grad():
                # 外层no_grad
                rm_y = rm_x * 2
                
                # 内层enable_grad
                with enable_grad():
                    rm_grad_enabled_inner = is_grad_enabled()
                    rm_z = rm_y + 3
                    rm_new_tensor = rm.tensor(x_np, requires_grad=True)
                
                # 回到外层no_grad
                rm_grad_enabled_outer = is_grad_enabled()
                rm_w = rm_z * 4
            
            if TORCH_AVAILABLE:
                with torch.no_grad():
                    # 外层no_grad
                    torch_y = torch_x * 2
                    
                    # 内层enable_grad
                    with torch.enable_grad():
                        torch_grad_enabled_inner = torch.is_grad_enabled()
                        torch_z = torch_y + 3
                        torch_new_tensor = torch.tensor(x_np, requires_grad=True)
                    
                    # 回到外层no_grad
                    torch_grad_enabled_outer = torch.is_grad_enabled()
                    torch_w = torch_z * 4
            else:
                torch_grad_enabled_inner = None
                torch_grad_enabled_outer = None
                torch_y = None
                torch_z = None
                torch_w = None
                torch_new_tensor = None
            
            # 收集结果
            rm_result = (
                rm_grad_enabled_inner,
                rm_grad_enabled_outer,
                rm_y.requires_grad,
                rm_z.requires_grad,
                rm_w.requires_grad,
                rm_new_tensor.requires_grad
            )
            
            torch_result = (
                torch_grad_enabled_inner,
                torch_grad_enabled_outer,
                torch_y.requires_grad,
                torch_z.requires_grad,
                torch_w.requires_grad,
                torch_new_tensor.requires_grad
            ) if TORCH_AVAILABLE else None
            
            # 比较结果
            passed = compare_grad_context_behaviors(rm_result, torch_result)
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  行为比较: 失败")
                    print(f"  Riemann: {rm_result}")
                    print(f"  PyTorch: {torch_result}")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"enable_grad嵌套上下文行为不匹配: {case_name}")
                
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_no_grad_decorator(self):
        """测试no_grad装饰器"""
        case_name = "no_grad装饰器"
        start_time = time.time()
        try:
        # 定义测试函数
            @no_grad  # 修复：移除括号
            def riemann_function():
                x = rm.tensor(2.0, requires_grad=True)
                result = x * 3 + 4
                return (is_grad_enabled(), x.requires_grad, result.requires_grad)
            
            # 执行函数
            rm_result = riemann_function()
            
            # PyTorch版本
            if TORCH_AVAILABLE:
                @torch.no_grad  # 修复：移除括号
                def torch_function():
                    x = torch.tensor(2.0, requires_grad=True)
                    result = x * 3 + 4
                    return (torch.is_grad_enabled(), x.requires_grad, result.requires_grad)
                
                torch_result = torch_function()
            else:
                torch_result = None
            
            # 比较结果
            passed = compare_grad_context_behaviors(rm_result, torch_result)
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  行为比较: 失败")
                    print(f"  Riemann: {rm_result}")
                    print(f"  PyTorch: {torch_result}")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"no_grad装饰器行为不匹配: {case_name}")
                
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_enable_grad_decorator(self):
        """测试enable_grad装饰器"""
        case_name = "enable_grad装饰器"
        start_time = time.time()
        try:
            # 定义测试函数
            @enable_grad  # 修复：移除括号
            def riemann_function():
                x = rm.tensor(2.0, requires_grad=True)
                result = x * 3 + 4
                return (is_grad_enabled(), x.requires_grad, result.requires_grad)
            
            # 测试1: 正常调用
            rm_result_normal = riemann_function()
            
            # 测试2: 在no_grad上下文中调用
            with no_grad():
                rm_result_in_no_grad = riemann_function()
            
            # PyTorch版本
            if TORCH_AVAILABLE:
                @torch.enable_grad  # 修复：移除括号
                def torch_function():
                    x = torch.tensor(2.0, requires_grad=True)
                    result = x * 3 + 4
                    return (torch.is_grad_enabled(), x.requires_grad, result.requires_grad)
                
                torch_result_normal = torch_function()
                
                with torch.no_grad():
                    torch_result_in_no_grad = torch_function()
            else:
                torch_result_normal = None
                torch_result_in_no_grad = None
            
            # 收集结果
            rm_result = (rm_result_normal, rm_result_in_no_grad)
            torch_result = (torch_result_normal, torch_result_in_no_grad) if TORCH_AVAILABLE else None
            
            # 比较结果
            passed = compare_grad_context_behaviors(rm_result, torch_result)
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  行为比较: 失败")
                    print(f"  Riemann正常: {rm_result_normal}")
                    print(f"  Riemann在no_grad中: {rm_result_in_no_grad}")
                    if TORCH_AVAILABLE:
                        print(f"  PyTorch正常: {torch_result_normal}")
                        print(f"  PyTorch在no_grad中: {torch_result_in_no_grad}")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"enable_grad装饰器行为不匹配: {case_name}")
                
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_nested_decorators(self):
        """测试嵌套装饰器"""
        case_name = "嵌套装饰器"
        start_time = time.time()
        try:
            # 定义嵌套装饰器测试函数
            @no_grad  # 修复：移除括号
            def outer_riemann_function():
                # 外层no_grad
                x = rm.tensor(2.0, requires_grad=True)
                intermediate = x * 2
                
                # 调用内层函数
                result = inner_riemann_function(intermediate)
                
                # 回到外层no_grad
                final = result + 10
                return (
                    is_grad_enabled(),
                    x.requires_grad,
                    intermediate.requires_grad,
                    result.requires_grad,
                    final.requires_grad
                )
            
            @enable_grad  # 修复：移除括号
            def inner_riemann_function(input_tensor):
                # 内层enable_grad
                return input_tensor * 3 + 4
            
            # 执行函数
            rm_result = outer_riemann_function()
            
            # PyTorch版本
            if TORCH_AVAILABLE:
                @torch.no_grad  # 修复：移除括号
                def outer_torch_function():
                    # 外层no_grad
                    x = torch.tensor(2.0, requires_grad=True)
                    intermediate = x * 2
                    
                    # 调用内层函数
                    result = inner_torch_function(intermediate)
                    
                    # 回到外层no_grad
                    final = result + 10
                    return (
                        torch.is_grad_enabled(),
                        x.requires_grad,
                        intermediate.requires_grad,
                        result.requires_grad,
                        final.requires_grad
                    )
                
                @torch.enable_grad  # 修复：移除括号
                def inner_torch_function(input_tensor):
                    # 内层enable_grad
                    return input_tensor * 3 + 4
                
                torch_result = outer_torch_function()
            else:
                torch_result = None
            
            # 比较结果
            passed = compare_grad_context_behaviors(rm_result, torch_result)
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  行为比较: 失败")
                    print(f"  Riemann: {rm_result}")
                    print(f"  PyTorch: {torch_result}")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"嵌套装饰器行为不匹配: {case_name}")
                
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_set_grad_enabled_context(self):
        """测试set_grad_enabled上下文管理器"""
        case_name = "set_grad_enabled上下文管理器"
        start_time = time.time()
        try:
            # 创建requires_grad=True的张量
            x_np = 2.0
            rm_x = rm.tensor(x_np, requires_grad=True)
            if TORCH_AVAILABLE:
                torch_x = torch.tensor(x_np, requires_grad=True)
            else:
                torch_x = None
            
            # 测试1: set_grad_enabled(False)
            with set_grad_enabled(False):
                rm_grad_enabled_false = is_grad_enabled()
                rm_y = rm_x * 2
            
            # 测试2: set_grad_enabled(True)
            with set_grad_enabled(True):
                rm_grad_enabled_true = is_grad_enabled()
                rm_z = rm_x * 3
            
            # 测试3: 嵌套set_grad_enabled
            with set_grad_enabled(False):
                rm_w1 = rm_x * 4
                with set_grad_enabled(True):
                    rm_grad_enabled_nested = is_grad_enabled()
                    rm_w2 = rm_x * 5
                    rm_new_tensor = rm.tensor(x_np, requires_grad=True)
                rm_w3 = rm_x * 6
            
            # PyTorch版本
            if TORCH_AVAILABLE:
                # 测试1: set_grad_enabled(False)
                with torch.set_grad_enabled(False):
                    torch_grad_enabled_false = torch.is_grad_enabled()
                    torch_y = torch_x * 2
                
                # 测试2: set_grad_enabled(True)
                with torch.set_grad_enabled(True):
                    torch_grad_enabled_true = torch.is_grad_enabled()
                    torch_z = torch_x * 3
                
                # 测试3: 嵌套set_grad_enabled
                with torch.set_grad_enabled(False):
                    torch_w1 = torch_x * 4
                    with torch.set_grad_enabled(True):
                        torch_grad_enabled_nested = torch.is_grad_enabled()
                        torch_w2 = torch_x * 5
                        torch_new_tensor = torch.tensor(x_np, requires_grad=True)
                    torch_w3 = torch_x * 6
            else:
                torch_grad_enabled_false = None
                torch_grad_enabled_true = None
                torch_grad_enabled_nested = None
                torch_y = None
                torch_z = None
                torch_w1 = None
                torch_w2 = None
                torch_w3 = None
                torch_new_tensor = None
            
            # 收集结果
            rm_result = (
                rm_grad_enabled_false,
                rm_grad_enabled_true,
                rm_grad_enabled_nested,
                rm_y.requires_grad,
                rm_z.requires_grad,
                rm_w1.requires_grad,
                rm_w2.requires_grad,
                rm_w3.requires_grad,
                rm_new_tensor.requires_grad
            )
            
            torch_result = (
                torch_grad_enabled_false,
                torch_grad_enabled_true,
                torch_grad_enabled_nested,
                torch_y.requires_grad,
                torch_z.requires_grad,
                torch_w1.requires_grad,
                torch_w2.requires_grad,
                torch_w3.requires_grad,
                torch_new_tensor.requires_grad
            ) if TORCH_AVAILABLE else None
            
            # 比较结果
            passed = compare_grad_context_behaviors(rm_result, torch_result)
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  行为比较: 失败")
                    print(f"  Riemann: {rm_result}")
                    print(f"  PyTorch: {torch_result}")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"set_grad_enabled上下文行为不匹配: {case_name}")
                
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise
    
    def test_set_grad_enabled_decorator(self):
        """测试set_grad_enabled装饰器"""
        case_name = "set_grad_enabled装饰器"
        start_time = time.time()
        try:
            # 定义测试函数
            @set_grad_enabled(False)
            def inference_without_grad():
                x = rm.tensor(2.0, requires_grad=True)
                result = x * 3 + 4
                return (is_grad_enabled(), x.requires_grad, result.requires_grad)
            
            @set_grad_enabled(True)
            def compute_with_grad_explicit():
                x = rm.tensor(2.0, requires_grad=True)
                result = x * 3 + 4
                return (is_grad_enabled(), x.requires_grad, result.requires_grad)
            
            # 测试1: 禁用梯度的函数
            rm_result_disabled = inference_without_grad()
            
            # 测试2: 启用梯度的函数
            rm_result_enabled = compute_with_grad_explicit()
            
            # 测试3: 在no_grad上下文中调用启用梯度的函数
            with no_grad():
                rm_result_in_no_grad = compute_with_grad_explicit()
            
            # PyTorch版本
            if TORCH_AVAILABLE:
                @torch.set_grad_enabled(False)
                def torch_inference_without_grad():
                    x = torch.tensor(2.0, requires_grad=True)
                    result = x * 3 + 4
                    return (torch.is_grad_enabled(), x.requires_grad, result.requires_grad)
                
                @torch.set_grad_enabled(True)
                def torch_compute_with_grad_explicit():
                    x = torch.tensor(2.0, requires_grad=True)
                    result = x * 3 + 4
                    return (torch.is_grad_enabled(), x.requires_grad, result.requires_grad)
                
                torch_result_disabled = torch_inference_without_grad()
                torch_result_enabled = torch_compute_with_grad_explicit()
                
                with torch.no_grad():
                    torch_result_in_no_grad = torch_compute_with_grad_explicit()
            else:
                torch_result_disabled = None
                torch_result_enabled = None
                torch_result_in_no_grad = None
            
            # 收集结果
            rm_result = (rm_result_disabled, rm_result_enabled, rm_result_in_no_grad)
            torch_result = (torch_result_disabled, torch_result_enabled, torch_result_in_no_grad) if TORCH_AVAILABLE else None
            
            # 比较结果
            passed = compare_grad_context_behaviors(rm_result, torch_result)
            time_taken = time.time() - start_time
            
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, passed)
                status = "通过" if passed else "失败"
                print(f"测试用例: {case_name} - {Colors.OKGREEN if passed else Colors.FAIL}{status}{Colors.ENDC} ({time_taken:.4f}秒)")
                if not passed:
                    print(f"  行为比较: 失败")
                    print(f"  Riemann禁用梯度: {rm_result_disabled}")
                    print(f"  Riemann启用梯度: {rm_result_enabled}")
                    print(f"  Riemann在no_grad中: {rm_result_in_no_grad}")
                    if TORCH_AVAILABLE:
                        print(f"  PyTorch禁用梯度: {torch_result_disabled}")
                        print(f"  PyTorch启用梯度: {torch_result_enabled}")
                        print(f"  PyTorch在no_grad中: {torch_result_in_no_grad}")
            
            # 断言确保测试通过
            self.assertTrue(passed, f"set_grad_enabled装饰器行为不匹配: {case_name}")
                
        except Exception as e:
            time_taken = time.time() - start_time
            if IS_RUNNING_AS_SCRIPT:
                stats.add_result(case_name, False, [str(e)])
                print(f"测试用例: {case_name} - {Colors.FAIL}错误{Colors.ENDC} ({time_taken:.4f}秒) - {str(e)}")
            raise

if __name__ == '__main__':
    # 设置为独立脚本运行模式
    IS_RUNNING_AS_SCRIPT = True
    
    # 清屏
    clear_screen()
    
    print(f"{Colors.HEADER}{Colors.BOLD}===== 开始运行梯度上下文控制测试 ====={Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}测试框架: Riemann vs PyTorch{Colors.ENDC}")
    print()
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestGradContext)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=0)  # 禁用默认输出，使用自定义输出
    result = runner.run(test_suite)
    
    # 打印测试统计摘要
    stats.print_summary()
    
    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)